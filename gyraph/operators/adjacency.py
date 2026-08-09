"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np, hermitian, TV, warnings
from .base import Operator
from .jordan_destroy import destroy_jordan_blocks, destroy_zero_eigenvals
from typing import Optional, Any


class Adjacency(Operator):
    """
    Adjacency operator for graph signal processing.

    Builds the (optionally normalized) adjacency matrix of a graph and
    computes its eigendecomposition to serve as the Graph Fourier basis.

    Parameters
    ----------
    graph : Graph
        The graph object this operator is associated with.
    name : str, optional
        Human-readable name for the operator.
    params : dict, optional
        Additional operator parameters passed to the base class.
    decomposition : str, optional
        Eigendecomposition method — ``'eig'`` (default) uses
        :func:`numpy.linalg.eig`; ``'jordan'`` falls back to sympy's
        Jordan normal form for non-diagonalizable matrices.
    normalize : str, optional
        Normalization applied to the adjacency matrix before
        decomposition. One of ``'left'``, ``'right'``, or
        ``'symmetric'``. ``None`` (default) skips normalization.
    partial : bool, optional
        If ``True``, skip the eigendecomposition step on construction.
        Useful when only the operator matrix is needed. Default is
        ``False``.
    """

    def __init__(
        self,
        graph: Any,
        name: Optional[str] = None,
        params: Optional[dict] = None,
        decomposition: str = "eig",
        normalize: Optional[str] = None,
        partial: bool = False,
    ):
        super().__init__(graph, name=name, params=params)
        if normalize is not None:
            if normalize not in ["right", "left", "symmetric"]:
                raise ValueError(
                    "normalize must be one of ['right', 'left', 'symmetric']"
                )
            self.params["normalize"] = normalize

        self.compute_operator()
        if not partial:
            self.compute_basis(decomposition=decomposition)

    def compute_operator(self):
        """Compute the Adjacency operator for the graph."""
        self.graph.adj_matrix = self.sanitize_operator(self.graph.adj_matrix)
        self.graph.adj_matrix = self.normalize_operator(
            self.graph.adj_matrix, order=self.params["normalize"]
        )
        self.M = self.graph.adj_matrix

    def compute_basis(self, decomposition: str = "eig"):
        """
        Compute the Graph Fourier basis via eigendecomposition of the adjacency matrix.

        Sets ``self.U`` (eigenvectors), ``self.V`` (eigenvalues), ``self.Uinv``
        (inverse Fourier basis), and ``self.frequencies`` (total variation
        ordering). For symmetric operators ``self.Uinv = self.U^H``; otherwise
        it is computed via matrix inversion. Jordan-block perturbation is
        applied automatically when the condition number of ``U`` is too large.

        Parameters
        ----------
        decomposition : str, optional
            ``'eig'`` (default) or ``'jordan'``.
        """
        if self.is_symmetric():
            self.V, self.U = np.linalg.eigh(self.M)
        else:
            if decomposition == "eig":
                try:
                    self.V, self.U = np.linalg.eig(self.M)
                    cond_number = np.linalg.cond(self.U)
                    if cond_number > 1e3:  # You can adjust this threshold as needed
                        if self.graph.debug:
                            warnings.warn(
                                f"The condition number of U is too high: {int(cond_number)}. Attempting to destroy Jordan blocks."
                            )
                        modified_M = destroy_jordan_blocks(self.M, max_iter=1000)
                        modified_M = destroy_zero_eigenvals(modified_M, max_iter=1000)
                        if self.graph.debug:
                            warnings.warn(
                                f"Attention! The Laplacian matrix has been modified to destroy Jordan blocks. {np.sum(self.M != modified_M)} numbers of edges modified."
                            )
                        self.M = modified_M
                        self.V, self.U = np.linalg.eig(self.M)
                        cond_number = np.linalg.cond(self.U)
                        if cond_number > 1e3:
                            warnings.warn(
                                f"Warning: The condition number of U is still too high after attempting to destroy Jordan blocks: {int(cond_number)}. Consider further reducing the threshold or investigating the graph structure."
                            )
                except np.linalg.LinAlgError:
                    if self.graph.debug:
                        warnings.warn(
                            "Matrix is not diagonalizable, attempting to destroy Jordan blocks."
                        )
                    modified_M = destroy_jordan_blocks(self.M)
                    modified_M = destroy_zero_eigenvals(
                        modified_M
                    )  # Making sure no zero eigenvalues for invertibility
                    if self.graph.debug:
                        warnings.warn(
                            f"Attention! The Laplacian matrix has been modified to destroy Jordan blocks.{np.sum(self.M != modified_M)} numbers of edges modified."
                        )
                    self.M = modified_M
                    try:
                        self.V, self.U = np.linalg.eig(self.M)
                    except np.linalg.LinAlgError:
                        raise np.linalg.LinAlgError(
                            "Matrix is still not diagonalizable after attempting to destroy Jordan blocks."
                        )
            else:
                from sympy import Matrix

                m = Matrix(self.M.astype(float))  # Cast to float to avoid sympy error
                P, J = m.jordan_form()
                self.V = np.array(J).astype(complex)
                self.U = np.array(P).astype(complex)

        # We take the frequencies non normalized by max eigenvalues
        self.frequencies = np.array(
            [
                TV(self.U[:, k], self.M, norm="L1", lbd_flag=False)
                for k in range(self.graph.N)
            ]
        )
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
        self.name = "Adjacency"
        self.params["cond_number"] = cond_number
        self.params["decomposition"] = decomposition

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
