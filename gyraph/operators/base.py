"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from typing import Optional, Dict, Any, List
from gyraph.utils import np, hermitian


class Operator:
    """
    Base class for graph filters.
    This class provides a template for creating various types of filters
    that can be applied to signals on graphs.

    Attributes
    ----------
    graph : Any
        The graph object this operator is associated with.
    name : str, optional
        Name of the operator.
    params : dict
        Dictionary of operator parameters including normalization method.
    M : np.ndarray, optional
        Matrix representation of the operator.
    U : np.ndarray, optional
        Fourier basis (eigenvectors of the operator).
    V : np.ndarray, optional
        Eigenvalues of the operator.
    Uinv : np.ndarray, optional
        Inverse of the Fourier basis, if precomputed.
    """

    def __init__(
        self,
        graph: Any,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.graph = graph
        self.name = name
        self.params = params if params is not None else {}

        self.params["normalize"] = None  # Normalization method for the operator
        self.M: Optional[np.ndarray] = None  # Matrix of the operator
        self.U: Optional[np.ndarray] = None  # Fourier basis
        self.V: Optional[np.ndarray] = None  # Eigenvalues of the operator
        self.Uinv: Optional[
            np.ndarray
        ] = None  # Inverse of the Fourier basis, if precomputed

    def compute_basis(self) -> None:
        """
        Compute the basis for the operator.
        This method should be overridden by subclasses to compute the basis
        for the specific operator type.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def GFT(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the Graph Fourier Transform of the input signal.

        Applies ``self.Uinv @ signal``. ``self.Uinv`` must be set by
        calling :meth:`compute_basis` first.

        Parameters
        ----------
        signal : np.ndarray, shape (N,) or (N, T)
            Input signal defined on graph vertices.

        Returns
        -------
        ret : np.ndarray
            Graph Fourier coefficients of ``signal``.
        """
        ret = self.Uinv @ signal
        return ret

    def inverseGFT(self, coef: np.ndarray) -> np.ndarray:
        """
        Compute the inverse Graph Fourier Transform.

        Applies ``self.U @ coef`` to reconstruct a vertex-domain signal
        from its frequency-domain coefficients.

        Parameters
        ----------
        coef : np.ndarray, shape (N,) or (N, T)
            Graph Fourier coefficients.

        Returns
        -------
        ret : np.ndarray
            Reconstructed signal defined on graph vertices.
        """
        ret = self.U @ coef
        return ret

    def sanitize_operator(self, A: np.ndarray) -> np.ndarray:
        """
        Sanitize the operator matrix by removing self-loops and ensuring non-negativity.

        Parameters
        ----------
        A : numpy.ndarray
            The operator matrix to be sanitized.

        Returns
        -------
        sanitized_A : numpy.ndarray
            The sanitized operator matrix.
        """
        # Remove self-loops by setting diagonal to zero
        A_no_self_loops = A - np.diag(np.diag(A))

        # Ensure non-negativity by setting negative values to zero
        sanitized_A = np.maximum(A_no_self_loops, 0)

        return sanitized_A

    def normalize_operator(
        self, A: np.ndarray, order: Optional[str] = "left"
    ) -> np.ndarray:
        """
        Normalize the operator matrix by in-degrees / out-degrees / symmetric

        Parameters
        ----------
        A : numpy.ndarray
            The operator matrix to be normalized.
        order : str
            The normalization method. Can be "right", "left", or "symmetric".

        Returns
        -------
        normA : numpy.ndarray
            The normalized operator matrix.
        """
        if np.any(A.imag != 0):
            raise ValueError("Complex values in adjacency matrix")

        elif order == "right":
            outdegrees = np.sum(A, axis=0)
            factors_in = np.diag(
                np.divide(1, outdegrees, where=np.abs(outdegrees) > 1e-10)
            )
            normA = A @ factors_in

        elif order == "left":
            indegrees = np.sum(A, axis=1)
            factors_out = np.diag(
                np.divide(1, indegrees, where=np.abs(indegrees) > 1e-10)
            )
            normA = factors_out @ A

        elif order == "symmetric":
            indegrees = np.sum(A, axis=1)
            outdegrees = np.sum(A, axis=0)

            indegrees = np.sqrt(np.abs(indegrees))
            outdegrees = np.sqrt(np.abs(outdegrees))

            factors_in = np.diag(
                np.divide(1, indegrees, where=np.abs(indegrees) > 1e-10)
            )
            factors_out = np.diag(
                np.divide(1, outdegrees, where=np.abs(outdegrees) > 1e-10)
            )
            normA = factors_out @ A @ factors_in

        elif order is None:
            normA = A
        self.params["normalize"] = order

        return normA

    def conjugate_frequency(self, idx: int) -> int:
        """
        Return conjugate frequency of the idx-th harmonic

        Parameters
        ----------
        idx : int
            Index of the harmonic

        Returns
        -------
        ret : int
            Index of the conjugate frequency
        """
        if self.V.ndim != 1:
            raise ValueError("Input must be 1D array")

        cf = np.where(np.abs(hermitian(self.V[idx]) - self.V) < 1e-8)[0]
        cf = list(set(cf) - set([idx]))
        if len(cf) == 0:
            ret = idx
        else:
            ret = cf[0]
        return ret

    def eigvalues_pairs(self) -> List[np.ndarray]:
        """
        Compute a list of groups (pairs or singletons) of complex conjugate eigenvalues.

        This function takes a numpy array `V` representing the eigenvalues of a graph Laplacian,
        and returns a list of groups of indices where the corresponding eigenvalues are either
        complex conjugate pairs or singletons (real eigenvalues).

        Parameters
        ----------

        Returns
        -------
        tasks: List[numpy.ndarray]
            A list of groups of indices where the corresponding eigenvalues are either
            complex conjugate pairs or singletons.
        """
        if self.V.ndim != 1:
            raise ValueError("Input must be 1D array")

        indexes = np.arange(self.V.shape[0])
        assigned = []
        pairs = []
        for idx in indexes:
            if idx in assigned:
                continue
            condition = (np.abs(self.V[idx].real - self.V.real) < 1e-8) & (
                np.abs(self.V[idx].imag + self.V.imag) < 1e-8
            )
            gp = np.where(condition)[0]
            if len(gp) == 1:
                g = gp[0]
                if g == idx:
                    # If only one element matches, it is a singleton
                    pairs.append(gp)
                    assigned += list(gp)

                # If two elements match, they are a conjugate pair
                else:
                    pairs.append(np.array([idx, g]))
                    assigned += list([idx, g])
            else:
                # More than two elements match, then overlap of reals
                # Separately add them in the pairs
                for g in gp:
                    pairs.append(np.array([g]))
                assigned += list(gp)

        return pairs

    def is_symmetric(self) -> bool:
        """
        Check if the operator matrix is symmetric.

        Returns
        -------
        bool
            True if the operator matrix is symmetric, False otherwise.
        """
        if self.M is None:
            raise ValueError("Operator matrix M is not yet defined.")

        return np.allclose(self.M, hermitian(self.M))

    def normality(self) -> float:
        """
        Compute the normality of a matrix M.

        Returns
        -------
        float
            Normality measure of the operator matrix. Returns 0 for a normal matrix
            (one that commutes with its conjugate transpose) and larger values indicate
            greater departure from normality. The theoretical maximum depends on the
            matrix structure.
        """
        if self.M is None:
            raise ValueError("Operator matrix M is not yet defined.")
        return np.linalg.norm(self.M @ self.M.T - self.M.T @ self.M, ord="fro") / (
            np.linalg.norm(self.M, ord="fro") ** 2
        )

    def __repr__(self) -> str:
        return f"<Operator(name={self.name}, params={self.params})>"
