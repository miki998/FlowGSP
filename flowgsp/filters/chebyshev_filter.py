"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import *
from .graph_filter import GraphFilter

class ChebyshevFilter(GraphFilter):
    """
    A class for Chebyshev graph filters that applies a Chebyshev polynomial
    of the graph shift operator (GSO) to a signal in vertex domain.
    This class inherits from GraphFilter and implements the specific methods
    for Chebyshev filtering, including the computation of Chebyshev polynomials
    and the application of the filter to a signal.
    # TODO: Test implementation of ChebyshevFilter.
    """
    def __init__(self, graph, params=None):
        self.__init__(graph, name=None, params=params)
        self.name = "ChebyshevFilter"
        raise NotImplementedError("ChebyshevFilter is an abstract class and cannot be instantiated directly.")

    def apply(self, signal, coefs):
        """
        Override the apply method for child classes to implement specific filtering logic.
        """
        raise NotImplementedError("The apply method must be implemented in the child class.")
    
    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"
    

    def chebyshev_polynomial(self, L:np.ndarray, V:np.ndarray, deg:int, return_polyeig:bool=False):
        """
        Computes the Chebyshev polynomial of degree n.

        Parameters
        ----------
            L (np.ndarray): Laplacian matrix.
            V (np.ndarray): The eigenvalues of the graph Laplacian.
            deg (int): The degree of the Chebyshev polynomial.

        Returns
        -------
            list: List of the Chebyshev polynomials of degree n.
        """
        Lc = 2*(L / np.abs(V).max()) - 1
        assert V.ndim == 1, "The eigenvalues must be a 1D array."
        Vc = np.diag(2*(V / np.abs(V).max()) - 1)
        if deg == 0:
            return [np.eye(Lc.shape[0])]
        elif deg == 1:
            return [np.eye(Lc.shape[0]), Lc]
        else:
            Tn_2 = np.eye(Lc.shape[0])
            Tn_1 = Lc
            polynomials = [np.eye(Lc.shape[0]), Lc]

            Vn_2 = np.eye(Lc.shape[0])
            Vn_1 = Vc
            polynomials_eig = [np.diag(Vn_2), np.diag(Vn_1)]
            for i in range(2, deg):
                Tn = 2 * Lc * Tn_1 - Tn_2
                Tn_2, Tn_1 = deepcopy(Tn_1), deepcopy(Tn)
                polynomials.append(Tn)

                Vn = 2 * Vc * Vn_1 - Vn_2
                Vn_2, Vn_1 = deepcopy(Vn_1), deepcopy(Vn)
                polynomials_eig.append(np.diag(Vn))

            if return_polyeig:
                return np.array(polynomials), np.array(polynomials_eig)
            return np.array(polynomials)

    def get_chebyshev_coefficients_interpolate(self, kernel:np.ndarray, V:np.ndarray, 
                                            deg:int, nbsample_points:int=100):
        """
        Computes the Chebyshev coefficients of a kernel using interpolation.

        Parameters
        ----------
            kernel (np.ndarray): The kernel to compute the Chebyshev coefficients of.
            V (np.ndarray): The eigenvectors of the graph Laplacian.
            deg (int): The degree of the Chebyshev polynomial.
            nbsample_points (int): The number of sample points to use.

        Returns
        -------
            list: List of the Chebyshev coefficients.
        """
        assert kernel.ndim == 1, "The kernel must be a 1D array."
        if np.iscomplexobj(kernel):
            raise ValueError("The kernel must be a real-valued array.")
        coefs = []
        sample_points_kernel = np.array([np.pi * (n-1/2)/ nbsample_points for n in range(nbsample_points)])
        kernel_positions = V.real
        N = kernel.shape[0]

        for m in range(deg):
            interpolated_values = np.interp(sample_points_kernel, kernel_positions, kernel)
            coef = (2 / nbsample_points) * np.sum(interpolated_values * np.cos(m * sample_points_kernel))
            coefs.append(coef)
        return np.array(coefs)

    def get_chebyshev_coefficients(self, func, deg:int, nbsample_points:int=100):
        """
        Computes the Chebyshev coefficients of a function.

        Parameters
        ----------
            function (function): The function to compute the Chebyshev coefficients of.
            deg (int): The degree of the Chebyshev polynomial.
            nbsample_points (int): The number of sample points to use.

        Returns
        -------
            list: List of the Chebyshev coefficients.
        """
        assert callable(type(func))

        coefs = []
        sample_points_kernel = np.array([np.pi * (n-1/2)/ nbsample_points for n in range(nbsample_points)])

        for m in range(deg):
            evaluated = func(sample_points_kernel)
            coef = (2 / nbsample_points) * np.sum(evaluated * np.cos(m * sample_points_kernel))
            coefs.append(coef)
        return np.array(coefs)

    def chebyshev_filter(self, A:np.ndarray, kernel:Optional[np.ndarray], V:np.ndarray, 
                        deg:int, nbsample_points:int=100, return_coefs:bool=False):
        """
        Applies a Chebyshev graph filter to a signal on a (undirected & directed) graph.
        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            U (np.ndarray): The eigenvectors of the graph Laplacian.
            Uinv (np.ndarray): The inverse of the eigenvectors of the graph Laplacian.
            kernel (np.ndarray): The graph filter kernel.
            V (np.ndarray): The eigenvectors of the graph Laplacian.
            minpolydeg (int): The degree of the polynomial.
            normalize_gso (bool): Whether to normalize the graph shift operator.
            return_coefs (bool): Whether to return the polynomial coefficients.
        Returns
        -------
            graph_filter (np.ndarray): graph filter
        """
        if callable(type(kernel)):
            c = self.get_chebyshev_coefficients(kernel, deg, nbsample_points)

        elif isinstance(kernel, np.ndarray):
            assert kernel.ndim == 1, "The kernel must be a 1D array."
            c = self.get_chebyshev_coefficients_interpolate(kernel, V, deg, nbsample_points)
        else:
            raise ValueError("The kernel must be a 1D array or a callable function.")
        
        polynomials = self.chebyshev_polynomial(A, V, deg)
        graph_filter = np.sum([c[i] * polynomials[i] for i in range(deg)], axis=0)
        if return_coefs:
            return graph_filter, c
        return graph_filter