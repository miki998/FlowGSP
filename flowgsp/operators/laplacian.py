"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from ..utils import np, hermitian
from .base import Operator
from .jordan_destroy import destroy_jordan_blocks_laplacian
from typing import Optional

class Laplacian(Operator):
    """
    Class for the Laplacian operator on a graph.
    This class inherits from the Operator base class and implements the specific methods
    for the Laplacian operator, including the computation of the basis and the Graph Fourier Transform.
    """

    def __init__(self, graph, name=None, params=None, 
                 in_degree:bool=True, normalize:Optional[str]=None):
        super().__init__(graph, name=name, params=params)
        if normalize is not None:
            if normalize not in ['right', 'left', 'symmetric']:
                raise ValueError("normalize must be one of ['right', 'left', 'symmetric']")
            self.params['normalize'] = normalize
        self.compute_basis(in_degree=in_degree)

    def compute_basis(self, in_degree:bool=True):
        """
        Compute the basis for the Laplacian operator.
        The basis is computed as the eigenvectors of the Laplacian matrix.
        """
        self.graph.adj_matrix = self.normalize_operator(self.graph.adj_matrix, order=self.params['normalize'])
        self.M = self.compute_directed_laplacian(self.graph.adj_matrix, in_degree=in_degree)
        
        if self.is_symmetric():
            self.V, self.U = np.linalg.eig(self.M)
        else:
            try:
                self.V, self.U = np.linalg.eig(self.M)
                self.Uinv = np.linalg.inv(self.U)
            except np.linalg.LinAlgError as e:
                print("Matrix is not diagonalizable, attempting to destroy Jordan blocks.")
                self.M = destroy_jordan_blocks_laplacian(self.M)
                print('Attention! The Laplacian matrix has been modified to destroy Jordan blocks.')
                self.V, self.U = np.linalg.eig(self.M)

        self.frequencies = np.abs(self.V)
        # Sort eigenvalues and eigenvectors
        if not np.all(np.abs(self.V - 1) < 1e-10): # If not a perfect cycle
            self.V = self.V[np.argsort(self.frequencies)]
            self.U = self.U[:, np.argsort(self.frequencies)]
            self.frequencies = np.sort(self.frequencies) # Sort frequencies in ascending order

        cond_number = np.linalg.cond(self.U)
        if cond_number > 1e3:  # You can adjust this threshold as needed
            print(f"Warning: The condition number of U is too high: {int(cond_number)}.")

        # Compute inverse Fourier transform
        if self.is_symmetric():
            self.Uinv = hermitian(self.U)
        else:
            self.Uinv = np.linalg.inv(self.U)

        self.imaginaries = np.abs(self.V.imag) >= 1e-8
        self.name = "Laplacian"
        self.params['in_degree'] = in_degree
        self.params['cond_number'] = cond_number
    
    def compute_directed_laplacian(self, A:np.ndarray, in_degree:bool=True):
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
            raise ValueError("Not an Adjacency matrix")
        
        if in_degree:
            deg = A.sum(axis=1).astype(float)
        else:
            deg = A.sum(axis=0).astype(float)
        ret = np.diag(deg) - A.astype(float)

        return ret

    def heat_kernel(self, alpha:float=0.001):
        """
        Compute the heat kernel for the Laplacian operator.
        The heat kernel is defined as K = 1 - alpha * V.real, where V is the eigenvalues of the Laplacian matrix.
        """
        kernel = np.ones(self.graph.N) - alpha * self.V.real
        return kernel