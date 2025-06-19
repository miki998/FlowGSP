"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from ..utils import np, hermitian

class Operator:
    """
    Base class for graph filters.
    This class provides a template for creating various types of filters
    that can be applied to signals on graphs.
    """

    def __init__(self, graph, name=None, params=None):
        self.graph = graph
        self.name = name
        self.params = params if params is not None else {}

        self.M = None # Matrix of the operator
        self.U = None # Fourier basis
        self.V = None # Eigenvalues of the operator
        self.Uinv = None # Inverse of the Fourier basis, if precomputed

    def compute_basis(self):
        """
        Compute the basis for the operator.
        This method should be overridden by subclasses to compute the basis
        for the specific operator type.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def GFT(self, signal):
        """
        Compute the Graph Fourier Transform of the input signal.

        Parameters
        ----------
        signal : numpy array
            Input signal defined on graph vertices 

        U : numpy array
            Graph Fourier basis (eigenvectors of graph Laplacian)

        herm : bool
            If True, use Hermitian transpose of U instead of U 

        Uinv : numpy array
            Inverse of U to avoid matrix inversion if precomputed
            Default is None to compute the inverse on the fly

        Returns
        -------
        ret : numpy array 
            Graph Fourier transform of signal 
        """
        ret = self.Uinv @ signal
        return ret
    
    def inverseGFT(self, coef):
        """
        Compute inverse graph Fourier transform of a signal.

        Given the graph Fourier coefficients `coef` and the graph 
        Fourier basis `U`, this function computes the inverse transform 
        to reconstruct the original signal defined on the vertices.

        Parameters
        ----------
        coef : numpy array
            Graph Fourier coefficients of signal

        U : numpy array
            Graph Fourier basis (eigenvectors of graph Laplacian)

        Returns
        -------  
        ret : numpy array
            Reconstructed signal defined on graph vertices
        """
        ret = self.U @ coef
        return ret

    def normalize_operator(self, A:np.ndarray, order:str="left"):
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

        if order == "right":
            outdegrees = np.sum(A, axis=0)
            factors_in = np.diag(np.divide(1, outdegrees, where=np.abs(outdegrees) > 1e-10))
            normA = A @ factors_in

        if order == "left":
            indegrees = np.sum(A, axis=1)
            factors_out = np.diag(np.divide(1, indegrees, where=np.abs(indegrees) > 1e-10))
            normA = factors_out @ A

        if order == "symmetric":
            indegrees = np.sum(A, axis=1)
            outdegrees = np.sum(A, axis=0)

            indegrees = np.sqrt(np.abs(indegrees))
            outdegrees = np.sqrt(np.abs(outdegrees))

            factors_in = np.diag(np.divide(1, indegrees, where=np.abs(indegrees) > 1e-10))
            factors_out = np.diag(np.divide(1, outdegrees, where=np.abs(outdegrees) > 1e-10))
            normA = factors_out @ A @ factors_in

        self.params["normalization"] = order

        return normA

    def conjugate_frequency(self, idx:int):
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
        ret = cf[0]
        return ret

    def eigvalues_pairs(self):
        """
        Compute a list of groups (pairs or singletons) of complex conjugate eigenvalues.

        This function takes a numpy array `V` representing the eigenvalues of a graph Laplacian,
        and returns a list of groups of indices where the corresponding eigenvalues are either
        complex conjugate pairs or singletons (real eigenvalues).

        Parameters
        ----------

        Returns
        -------
        tasks: numpy.ndarray
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
            condition = (np.abs(self.V[idx].real - self.V.real) < 1e-8) & (np.abs(self.V[idx].imag + self.V.imag) < 1e-8)
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

    def is_symmetric(self):
        """
        Check if the operator matrix is symmetric.

        Returns
        -------
        bool
            True if the operator matrix is symmetric, False otherwise.
        """
        if self.M is None:
            raise ValueError("Operator matrix M is not defined.")
        
        return np.allclose(self.M, hermitian(self.M))

    def __repr__(self):
        return f"<Operator(name={self.name}, params={self.params})>"