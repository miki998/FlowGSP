"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np, hermitian, sobolev, warnings, low_rank_approximation_m
from .base import Operator
from .jordan_destroy import destroy_jordan_blocks_laplacian
from typing import Optional, Any


class AdvectionDiffusion(Operator):
    """
    Class for the Advection-Diffusion operator on a directed graph.
    This class inherits from the Operator base class and implements the specific methods
    for the Advection-Diffusion operator, including the computation of the basis and the Graph Fourier Transform.
    The Advection-Diffusion operator (Divergence Free) is defined as L = D - A, where D is a diagonal matrix containing the in-degree of each node,
    and A is the adjacency matrix of the graph.
    """

    def __init__(
        self,
        graph: Any,
        name: Optional[str] = None,
        params: Optional[dict] = None,
        in_degree: bool = True,
        normalize: Optional[str] = None,
        k_rank: Optional[int] = None,
        partial: bool = False,
        rcond: Optional[float] = 1e-10,
    ):
        super().__init__(graph, name=name, params=params)
        if normalize is not None:
            if normalize not in ["right", "left", "symmetric"]:
                raise ValueError(
                    "normalize must be one of ['right', 'left', 'symmetric']"
                )
            self.params["normalize"] = normalize

        self.compute_operator(in_degree=in_degree, divergence_free=True)

        # If partial is True but k_rank is None then do not compute any spectral decomposition.
        if (not partial) and k_rank is not None:
            partial = True  # If k_rank is specified, then partial must be True to compute the low-rank approximation of the spectral decomposition.

        if not partial:
            self.compute_basis()
            self.reconstruct_advection_diffusion_operators(rcond=rcond)
        elif partial and (
            k_rank is not None
        ):  # If not partial, then compute the full spectral decomposition. If k_rank is not None, then compute the low-rank approximation of the spectral decomposition.
            self.reconstruct_advection_diffusion_operators(
                k_rank=k_rank, rcond=rcond if k_rank is None else 1e-15
            )  # Use default rcond for pseudo-inverse to increase computation speed.

    def compute_operator(self, in_degree: bool = True, divergence_free: bool = True):
        """Compute the Advection-Diffusion operator for the graph."""
        if divergence_free:
            # If normalizing, then normalize before computing the Advection Diffusion
            self.graph.adj_matrix = self.sanitize_operator(self.graph.adj_matrix)
            self.graph.adj_matrix = self.normalize_operator(
                self.graph.adj_matrix, order=self.params["normalize"]
            )
            self.M = self.compute_directed_laplacian(
                self.graph.adj_matrix, in_degree=in_degree
            )
        else:
            raise NotImplementedError("Non divergence free theory not developed yet.")

        self.params["in_degree"] = in_degree

    def compute_basis(self):
        """
        Compute the basis for the Laplacian operator.
        The basis is computed as the eigenvectors of the Laplacian matrix.
        """
        if self.is_symmetric():
            self.V, self.U = np.linalg.eigh(self.M)
        else:
            try:
                self.V, self.U = np.linalg.eig(self.M)
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
                modified_M = destroy_jordan_blocks_laplacian(self.M, max_iter=1000)
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

        self.radial_frequencies = np.abs(self.V)
        # Sort eigenvalues and eigenvectors
        if not np.all(np.abs(self.V - 1) < 1e-10):  # If not a perfect cycle
            self.V = self.V[np.argsort(self.radial_frequencies)]
            self.U = self.U[:, np.argsort(self.radial_frequencies)]
            self.radial_frequencies = np.sort(self.radial_frequencies)
            self.angular_frequencies = np.abs(np.angle(self.V))

            self.radial_order = np.argsort(self.radial_frequencies)
            self.angular_order = np.argsort(self.angular_frequencies)[::-1]

        # Compute inverse Fourier transform
        if self.is_symmetric():
            self.Uinv = hermitian(self.U)
        else:
            self.Uinv = np.linalg.inv(self.U)

        # Final condition number
        cond_number = np.linalg.cond(self.U)

        self.imaginaries = np.abs(self.V.imag) >= 1e-8
        self.name = "AdvectionDiffusion"
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
            # raise ValueError("Not an Adjacency matrix")

        if in_degree:
            deg = A.sum(axis=1).astype(float)
        else:
            deg = A.sum(axis=0).astype(float)
        ret = np.diag(deg) - A.astype(float)

        return ret

    def reconstruct_advection_diffusion_operators(
        self, rcond: float = 1e-10, k_rank: Optional[int] = None
    ):
        """
        Reconstruct the Advection-Diffusion operator using either the symmetric and anti-symmetric decomposition or the spectral decomposition.
        If k_rank is None, then compute the full spectral decomposition. If k_rank is not None and positive, then compute the low-rank approximation of the spectral decomposition. If k_rank is not None and non-positive, then compute the symmetric and anti-symmetric decomposition.
        """
        if k_rank is not None and (k_rank > self.graph.N):
            raise ValueError(
                f"k_rank must be less than or equal to the number of nodes in the graph. Given k_rank={k_rank} and number of nodes={self.graph.N}."
            )
        elif k_rank is not None and k_rank <= 0:
            self.sym_antisym_reconstruct(rcond=rcond)
        else:
            self.spectral_reconstruct_operators(rcond=rcond, k_rank=k_rank)

    def sym_antisym_reconstruct(self, rcond: float = 1e-10):
        """
        Reconstruct the Advection-Diffusion operator using the symmetric and anti-symmetric decomposition.
        The Advection-Diffusion operator is reconstructed as P + Q, where P is the symmetric part and Q is the anti-symmetric part of the operator.
        """
        self.P = (self.M + self.M.T) / 2
        self.Q = (self.M - self.M.T) / 2
        self.Z = self.Q @ np.linalg.pinv(self.P, rcond=rcond)

    def spectral_reconstruct_operators(
        self, rcond: float = 1e-10, k_rank: Optional[int] = None
    ):
        """
        Reconstruct the Advection-Diffusion operator using the spectral decomposition.
        The Advection-Diffusion operator is reconstructed as P + Q, where P is the real part and Q is the imaginary part of the operator.
        """
        if k_rank is None:
            self.P = (self.U @ np.diag(self.V.real) @ self.Uinv).real
            self.Q = (self.U @ (1j * np.diag(self.V.imag)) @ self.Uinv).real
            self.Z = (
                self.U
                @ (1j * np.diag(self.V.imag))
                @ np.linalg.pinv(np.diag(self.V.real), rcond=rcond)
                @ self.Uinv
            ).real
        else:
            approx_eigvals, approx_eigvecs = low_rank_approximation_m(
                self.M,
                k_rank,
                which="S",  # 'S' for smallest magnitude eigenvalues -> from experiment of operator approximation in nn
            )
            safe_indices = (
                []
            )  # Sanitize the approximated eigenvalues and eigenvectors to ensure numerical stability, e.g., by thresholding large values.
            for i in range(len(approx_eigvals)):
                if np.abs(approx_eigvecs[i]).max() < 1e5:
                    safe_indices.append(i)
            if len(safe_indices) < k_rank:
                warnings.warn(
                    f"Only {len(safe_indices)} out of {k_rank} approximated eigenvalues/eigenvectors are considered safe for reconstruction. Consider reducing k_rank or adjusting the approximation method for better stability."
                )
            safe_indices = np.array(safe_indices)
            approx_eigvals = approx_eigvals[safe_indices]
            approx_eigvecs = approx_eigvecs[:, safe_indices]

            self.V_approx = approx_eigvals
            self.U_approx = approx_eigvecs

            self.P = (
                approx_eigvecs
                @ np.diag(approx_eigvals.real)
                @ np.linalg.pinv(approx_eigvecs)
            ).real
            self.Q = (
                approx_eigvecs
                @ (1j * np.diag(approx_eigvals.imag))
                @ np.linalg.pinv(approx_eigvecs)
            ).real

            self.Z = (
                approx_eigvecs
                @ (1j * np.diag(approx_eigvals.imag))
                @ np.linalg.pinv(np.diag(approx_eigvals.real), rcond=rcond)
                @ np.linalg.pinv(approx_eigvecs)
            ).real

    def heat_transport_kernel(self, alpha: float, beta: float) -> np.ndarray:
        """
        Compute the heat transport kernel for the Advection-Diffusion operator.
        The heat transport kernel is defined as K = 1 - alpha * V.real - 1j * beta * V.imag, where V is the eigenvalues of the Laplacian matrix.
        """
        kernel_shift = alpha * self.V.real + 1j * beta * self.V.imag

        if np.any(np.abs(kernel_shift) >= 1):
            warnings.warn(
                "The heat transport kernel may be unstable due to large alpha or beta. Consider reducing the values of alpha and beta to ensure stability."
            )

        kernel = np.ones(self.graph.N).astype(complex) - kernel_shift
        return kernel

    def transport_kernel(self, alpha: float) -> np.ndarray:
        """
        Compute the transport kernel for the Advection-Diffusion operator.
        The transport kernel is defined as K = 1 - 1j * alpha * V.imag, where V is the eigenvalues of the Laplacian matrix.
        """
        kernel_shift = 1j * alpha * self.V.imag

        if np.any(np.abs(kernel_shift) >= 1):
            warnings.warn(
                "The transport kernel may be unstable due to large alpha. Consider reducing the value of alpha to ensure stability."
            )

        kernel = np.ones(self.graph.N).astype(complex) - kernel_shift
        return kernel

    def heat_kernel(self, alpha: float) -> np.ndarray:
        """
        Compute the heat kernel for the Advection-Diffusion operator.
        The heat kernel is defined as K = 1 - alpha * V.real, where V is the eigenvalues of the Laplacian matrix.
        """
        kernel_shift = alpha * self.V.real

        if np.any(np.abs(kernel_shift) >= 1):
            warnings.warn(
                "The heat kernel may be unstable due to large alpha. Consider reducing the value of alpha to ensure stability."
            )

        kernel = np.ones(self.graph.N).astype(complex) - kernel_shift
        return kernel

    def low_pass_kernel(
        self, limfreq: int, mode: str = "radial", factor: int = 1
    ) -> np.ndarray:
        """
        Compute the low-pass kernel for the Advection-Diffusion operator. Rectangular ideal low-pass filter.
        """
        nimaginaries = self.imaginaries.sum()
        if mode == "radial":
            kernel = np.zeros(self.graph.N)
            pad = 1 if limfreq < self.conjugate_frequency(limfreq) else 2
            kernel[: limfreq + pad] = factor
        else:
            kernel = np.zeros(self.graph.N)
            if limfreq >= nimaginaries:
                limfreq = nimaginaries - 1
            pad = 1 if limfreq < self.graph.operator.conjugate_frequency(limfreq) else 2
            kernel[self.angular_order[: limfreq + pad]] = factor

        return kernel

    def high_pass_kernel(
        self, limfreq: int, mode: str = "radial", factor: int = 1
    ) -> np.ndarray:
        """
        Compute the high-pass kernel for the Advection-Diffusion operator. Rectangular ideal high-pass filter.
        """
        kernel = 1 - self.low_pass_kernel(limfreq, mode=mode, factor=1)
        kernel *= factor

        return kernel

    def radial_smoothness(
        self, signal: np.ndarray, norm: str = "L2", normalize: bool = False
    ) -> float:
        """
        Compute the radial smoothness of a signal with respect to the Advection-Diffusion operator.
        The radial smoothness is defined as the L2 norm of the Laplacian operator applied to the signal.

        Parameters
        ----------
        signal : ndarray
            The signal for which to compute radial smoothness.
        norm : str, optional
            The type of norm to use. Can be 'L1' or 'L2'. Default is 'L2'.
        normalize : bool, optional
            Whether to normalize by the L2 norm of the signal. Default is False.

        Returns
        -------
        smoothness : float
            The radial smoothness of the signal.
        """
        smoothness = sobolev(signal, self.M, norm=norm, normalize=normalize)
        return smoothness

    def angular_smoothness(
        self, signal: np.ndarray, norm: str = "L2", normalize: bool = False
    ) -> float:
        """
        Compute the angular smoothness of a signal with respect to the Advection-Diffusion operator.
        The angular smoothness is defined as the L2 norm of the Laplacian operator applied to the signal.

        Parameters
        ----------
        signal : ndarray
            The signal for which to compute angular smoothness.
        norm : str, optional
            The type of norm to use. Can be 'L1' or 'L2'. Default is 'L2'.
        normalize : bool, optional
            Whether to normalize by the L2 norm of the signal. Default is False.
        rcond : float, optional
            The reciprocal condition number for the pseudo-inverse. Default is 1e-10.

        Returns
        -------
        smoothness : float
            The angular smoothness of the signal.
        """
        smoothness = sobolev(signal, self.Z, norm=norm, normalize=normalize)
        return smoothness
