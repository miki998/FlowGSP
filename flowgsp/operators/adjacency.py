"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import np, hermitian, TV
from .base import Operator
from .jordan_destroy import destroy_jordan_blocks, destroy_zero_eigenvals
from typing import Optional

class Adjacency(Operator):
    """
    Class for the Laplacian operator on a graph.
    This class inherits from the Operator base class and implements the specific methods
    for the Laplacian operator, including the computation of the basis and the Graph Fourier Transform.
    """

    def __init__(self, graph, name=None, params=None, 
                 in_degree:bool=True, decomposition:str='eig',
                 normalize:Optional[str]=None):
        super().__init__(graph, name=name, params=params)
        if normalize is not None:
            if normalize not in ['right', 'left', 'symmetric']:
                raise ValueError("normalize must be one of ['right', 'left', 'symmetric']")
            self.params['normalize'] = normalize
        self.compute_basis(in_degree=in_degree, decomposition=decomposition)

    def compute_basis(self, in_degree:bool=True, decomposition:str='eig'):
        """
        Compute the basis for the Laplacian operator.
        The basis is computed as the eigenvectors of the Laplacian matrix.
        """
        self.graph.adj_matrix = self.normalize_operator(self.graph.adj_matrix, order=self.params['normalize'])
        self.M = self.graph.adj_matrix
        if self.is_symmetric():
            self.V, self.U = np.linalg.eigh(self.M)
        else:
            if decomposition == 'eig':
                try:
                    self.V, self.U = np.linalg.eig(self.M)
                except np.linalg.LinAlgError as e:
                    print("Matrix is not diagonalizable, attempting to destroy Jordan blocks.")
                    self.M = destroy_jordan_blocks(self.M)
                    self.M = destroy_zero_eigenvals(self.M) # Making sure no zero eigenvalues for invertibility
                    print('Attention! The Laplacian matrix has been modified to destroy Jordan blocks.')
                    self.V, self.U = np.linalg.eig(self.M)
            else:
                from sympy import Matrix
                m = Matrix(self.M.astype(float)) # Cast to float to avoid sympy error
                P, J = m.jordan_form()
                self.V = np.array(J).astype(complex)
                self.U = np.array(P).astype(complex)

        # We take the frequencies non normalized by max eigenvalues
        self.frequencies = np.array([TV(self.U[:,k], self.M, norm='L1', lbd_flag=False) 
                                        for k in range(self.graph.N)])
        # Sort eigenvalues and eigenvectors
        if not np.all(np.abs(self.V - 1) < 1e-10): # If not a perfect cycle
            self.V = self.V[np.argsort(self.frequencies)]
            self.U = self.U[:, np.argsort(self.frequencies)]
            self.frequencies = np.sort(self.frequencies) # Sort frequencies in ascending order

        # Compute inverse Fourier transform
        if self.is_symmetric():
            self.Uinv = hermitian(self.U)
        else:
            self.Uinv = np.linalg.inv(self.U)

        cond_number = np.linalg.cond(self.U)
        if cond_number > 1e3:  # You can adjust this threshold as needed
            print(f"Warning: The condition number of U is too high: {int(cond_number)}.")

        self.imaginaries = np.abs(self.V.imag) >= 1e-8
        self.name = "Adjacency"
        self.params['in_degree'] = in_degree
        self.params['cond_number'] = cond_number
        self.params['decomposition'] = decomposition