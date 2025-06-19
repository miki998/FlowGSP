"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

# TODO: Add implementation of "Design of polynomial approximated filters for signals on directed graphs"
# 10.1109/GlobalSIP.2017.8309036

from ..utils import *
from typing import Union
from .graph_filter import GraphFilter

class PolynomialFilter(GraphFilter):
    """
    A class for polynomial graph filters that applies a polynomial of the graph shift operator (GSO)
    to a signal in vertex domain.
    This class inherits from GraphFilter and implements the specific methods
    for polynomial filtering, including the computation of polynomial coefficients
    and the application of the filter to a signal.
    """
    def __init__(self, graph, name=None, params=None, order:Optional[int]=None):
        super().__init__(graph, name=name, params=params)
        self.name = "PolynomialFilter"
        if order is None:
            self.params['order'] = np.sqrt(self.graph.N)
        else:
            self.params['order'] = order
        self.precompute_polynomial()

    def apply(self, signal:np.ndarray, 
              kernel:np.ndarray, 
              return_coefs:bool=False,
              rcond:float=1e-8):
        """
        Applies the polynomial filter to a signal on a (undirected & directed) graph.
        
        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            kernel (np.ndarray): The graph filter kernel.
            return_coefs (bool): Whether to return the polynomial coefficients.
            rcond (float): The cutoff for the pseudo-inverse.
        
        Returns
        -------
            filtered_signal (np.ndarray): The filtered signal after applying the polynomial filter. + coefs if return_coefs is True.
        """
        if return_coefs:
            graph_filter, coefs = self.polynomial_filter(kernel, return_coefs=return_coefs, rcond=rcond)
            return graph_filter @ signal, coefs
        else:
            graph_filter = self.polynomial_filter(kernel, return_coefs=return_coefs, rcond=rcond)
            return graph_filter @ signal
        
    def polynomial_filter(self, kernel:np.ndarray,
                          return_coefs:bool=False,
                          rcond:float=1e-8):
        """
        Applies a polynomial graph filter to a signal on a (undirected & directed) graph.
        
        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            kernel (np.ndarray): The graph filter kernel.
            return_coefs (bool): Whether to return the polynomial coefficients.
            rcond (float): The cutoff for the pseudo-inverse.
        
        Returns
        -------
            graph_filter (np.ndarray): graph filter
        """

        assert kernel.ndim == 1, "The kernel must be a 1D array."
        deg = self.params['order']
        _, c = self.get_polynomial_coefficients(kernel, deg=deg, rcond=rcond)
        
        graph_filter = np.sum([c[i] * self.powers_of_M[i] for i in range(deg)], axis=0)
        
        if return_coefs:
            return graph_filter, c
        return graph_filter

    def precompute_polynomial(self):
        """
        Precompute and store the list of powers of the graph shift operator matrix.
        """
        M = self.graph.operator.M
        order = int(self.params['order'])
        self.powers_of_M = [np.eye(M.shape[0], dtype=M.dtype)]
        for i in range(1, order):
            self.powers_of_M.append(self.powers_of_M[-1] @ M)

    def get_polynomial_coefficients(self, kernel:np.ndarray,
                                    deg:float, rcond:float=1e-8):
        """
        
        Simply solve for (c_i) the system spectral with filter P (i.e kernel) and A=UVU^{-1}
        P = \sum_i\geq 0 c_i V^i

        Paramters
        ---------
        kernel: np.ndarray
            The filter kernel.
        V: np.ndarray
            The eigenvectors of the graph Laplacian.
        deg: int
            The minimum polynomial degree.
        rcond: float
            The cutoff for the pseudo-inverse.
        
        Returns
        ---------
        vdm_optim: np.ndarray
            The Vandermonde matrix.
        c_optim: np.ndarray
            The polynomial coefficients.
        """
        if deg >= 0:
            vdm_optim = self.vandermonde_matrix(deg)
            c_optim = np.linalg.pinv(vdm_optim, rcond=rcond) @ kernel
        else:
            c_optim = None
            vdm_optim = None
            best_reconstruct = np.inf
            for k in range(1, kernel.shape[0]+1):
                vdm = self.vandermonde_matrix(k)
                c = np.linalg.pinv(vdm, rcond=rcond) @ kernel
                reconstruct_error = np.abs(vdm @ c - kernel).sum()
                if reconstruct_error < best_reconstruct:
                    best_reconstruct = reconstruct_error
                    c_optim = c
                    vdm_optim = vdm

        return vdm_optim, c_optim

    def vandermonde_matrix(self, dim:int):
        """
        Computes the Vandermonde matrix of a vector.

        Parameters
        ----------
            dim (int): The dimension of the Vandermonde matrix.

        Returns
        -------
            vdm (np.ndarray): The Vandermonde matrix.
        """

        vdm = np.zeros((self.graph.N, dim)).astype(complex)
        for sidx in range(dim):
            vdm [:, sidx] = self.graph.operator.V ** sidx
        return vdm

    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"


