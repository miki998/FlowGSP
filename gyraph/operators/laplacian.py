"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np, hermitian, warnings
from .base import Operator
from .jordan_destroy import destroy_jordan_blocks_laplacian
from typing import Optional, Any


class Laplacian(Operator):
    """
    Laplacian operator for graph signal processing.

    Computes the directed graph Laplacian ``L = D - A`` and its
    eigendecomposition, which defines the Graph Fourier basis.
    Jordan-block perturbation is applied automatically for
    non-diagonalizable matrices.

    Parameters
    ----------
    graph : Graph
        The graph object this operator is associated with.
    name : str, optional
        Human-readable name for the operator.
    params : dict, optional
        Additional operator parameters passed to the base class.
    in_degree : bool, optional
        If ``True`` (default) the degree matrix uses in-degrees; otherwise
        out-degrees are used.
    normalize : str, optional
        Normalization applied to the adjacency matrix before the Laplacian
        is formed. One of ``'left'``, ``'right'``, or ``'symmetric'``.
        ``None`` (default) skips normalization.
    """

    def __init__(
        self,
        graph: Any,
        name: Optional[str] = None,
        params: Optional[dict] = None,
        in_degree: bool = True,
        normalize: Optional[str] = None,
    ):
        super().__init__(graph, name=name, params=params)
        if normalize is not None:
            if normalize not in ["right", "left", "symmetric"]:
                raise ValueError(
                    "normalize must be one of ['right', 'left', 'symmetric']"
                )
            self.params["normalize"] = normalize

        self.compute_operator(in_degree=in_degree)
        self.compute_basis()

    def compute_operator(self, in_degree: bool = True):
        """Compute the Laplacian operator for the graph."""
        self.graph.adj_matrix = self.sanitize_operator(self.graph.adj_matrix)
        self.graph.adj_matrix = self.normalize_operator(
            self.graph.adj_matrix, order=self.params["normalize"]
        )
        self.M = self.compute_directed_laplacian(
            self.graph.adj_matrix, in_degree=in_degree
        )

        self.params["in_degree"] = in_degree

    def compute_basis(self):
        """
        Compute the Graph Fourier basis via eigendecomposition of the Laplacian.

        Sets ``self.U``, ``self.V``, ``self.Uinv``, and ``self.frequencies``
        (ordered by eigenvalue magnitude). For symmetric Laplacians
        ``self.Uinv = self.U^H``; otherwise it is computed via matrix inversion.
        """
        if self.is_symmetric():
            self.V, self.U = np.linalg.eigh(self.M)
        else:
            try:
                self.V, self.U = np.linalg.eig(self.M)
                self.Uinv = np.linalg.inv(self.U)
                cond_number = np.linalg.cond(self.U)
                if cond_number > 1e3:  # You can adjust this threshold as needed
                    if self.graph.debug:
                        warnings.warn(
                            f"The condition number of U is too high: {int(cond_number)}. Attempting to destroy Jordan blocks."
                        )
                modified_M = destroy_jordan_blocks_laplacian(self.M, max_iter=1000)
                if self.graph.debug:
                    warnings.warn(
                        f"Attention! The Laplacian matrix has been modified to destroy Jordan blocks. {np.sum(self.M != modified_M)} numbers of edges modified."
                    )
                self.M = modified_M
                self.V, self.U = np.linalg.eig(self.M)
                cond_number = np.linalg.cond(self.U)
                if cond_number > 1e3:
                    warnings.warn(
                        f"The condition number of U is still too high after attempting to destroy Jordan blocks: {int(cond_number)}. Consider further reducing the threshold or investigating the graph structure."
                    )
            except np.linalg.LinAlgError:
                if self.graph.debug:
                    warnings.warn(
                        "Matrix is not diagonalizable, attempting to destroy Jordan blocks."
                    )
                modified_M = destroy_jordan_blocks_laplacian(self.M)
                if self.graph.debug:
                    warnings.warn(
                        f"Attention! The Laplacian matrix has been modified to destroy Jordan blocks. {np.sum(self.M != modified_M)} numbers of edges modified."
                    )
                self.M = modified_M
                try:
                    self.V, self.U = np.linalg.eig(self.M)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError(
                        "Matrix is still not diagonalizable after attempting to destroy Jordan blocks."
                    )

        self.frequencies = np.abs(self.V)
        # Sort eigenvalues and eigenvectors
        if not np.all(np.abs(self.V - 1) < 1e-10):  # If not a perfect cycle
            self.V = self.V[np.argsort(self.frequencies)]
            self.U = self.U[:, np.argsort(self.frequencies)]
            self.frequencies = np.sort(
                self.frequencies
            )  # Sort frequencies in ascending order

        # Compute inverse Fourier transform
        if self.is_symmetric():
            self.Uinv = hermitian(self.U)
        else:
            self.Uinv = np.linalg.inv(self.U)
        # Final condition number
        cond_number = np.linalg.cond(self.U)

        self.imaginaries = np.abs(self.V.imag) >= 1e-8
        self.name = "Laplacian"
        self.params["cond_number"] = cond_number

    def compute_directed_laplacian(
        self, A: np.ndarray, in_degree: bool = True
    ) -> np.ndarray:
        """
        Compute the directed Laplacian matrix for a given adjacency matrix A.

        The directed Laplacian is defined as L = D - A, where D is a diagonal matrix containing the in-degree of each node, and A is the adjacency matrix.

        Parameters
        ----------
        A : ndarray
            Adjacency matrix
        in_degree : bool
            Flag to compute in-degree or out-degree

        Returns
        -------
        ret : ndarray
            Directed Laplacian matrix
        """
        if np.any(A.imag != 0):
            raise ValueError("Complex values in laplacian matrix")
        elif np.any(np.diag(A) != 0):
            warnings.warn(
                "Diagonal entries in adjacency matrix are not zero, this is not a valid adjacency matrix."
            )

        if in_degree:
            deg = A.sum(axis=1).astype(float)
        else:
            deg = A.sum(axis=0).astype(float)
        ret = np.diag(deg) - A.astype(float)

        return ret

    def heat_kernel(self, alpha: float = 0.001) -> np.ndarray:
        """
        Spectral heat-diffusion kernel ``K = 1 - alpha * Re(V)``.

        Parameters
        ----------
        alpha : float, optional
            Diffusion coefficient. Large values may cause instability
            (a warning is issued when ``|alpha * Re(V)| >= 1``). Default is ``0.001``.

        Returns
        -------
        kernel : np.ndarray, shape (N,)
            Per-frequency gain vector.
        """
        kernel_shift = alpha * self.V.real

        if np.any(np.abs(kernel_shift) >= 1):
            warnings.warn(
                "The heat kernel may be unstable due to large alpha. Consider reducing the value of alpha to ensure stability."
            )

        kernel = np.ones(self.graph.N) - kernel_shift
        return kernel

    def low_pass_kernel(self, limfreq: int, factor: int = 1) -> np.ndarray:
        """
        Rectangular ideal low-pass kernel in the graph frequency domain.

        Parameters
        ----------
        limfreq : int
            Cutoff frequency index (inclusive).
        factor : int, optional
            Gain applied to the passband. Default is 1.

        Returns
        -------
        kernel : np.ndarray, shape (N,)
            Per-frequency gain vector (passband = ``factor``, stopband = 0).
        """

        kernel = np.zeros(self.graph.N)
        pad = 1 if limfreq < self.conjugate_frequency(limfreq) else 2
        kernel[: limfreq + pad] = factor

        return kernel

    def high_pass_kernel(self, limfreq: int, factor: int = 1) -> np.ndarray:
        """
        Rectangular ideal high-pass kernel in the graph frequency domain.

        Parameters
        ----------
        limfreq : int
            Cutoff frequency index (exclusive lower bound).
        factor : int, optional
            Gain applied to the passband. Default is 1.

        Returns
        -------
        kernel : np.ndarray, shape (N,)
            Per-frequency gain vector (complement of :meth:`low_pass_kernel`).
        """
        kernel = 1 - self.low_pass_kernel(limfreq, factor=1)
        kernel *= factor

        return kernel
