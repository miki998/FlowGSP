"""
Copyright © 2026 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np, hermitian
from .graph_filter import GraphFilter


class TikhonovFilter(GraphFilter):
    """
    A Tikhonov filter for graph signals.
    This filter applies a Tikhonov regularization technique to signals on graphs,
    which is particularly useful for denoising signals in the presence of noise.
    This filter is designed to work with both undirected and directed graphs.
    NOTE: Tikhonov is derived from a prior that uses either the AD operator or the Laplacian operator.
    The choice of other operators is not considered here and may produce unexpected results.
    """

    def __init__(self, graph, params=None):
        super().__init__(graph, name=None, params=params)
        self.name = "TikhonovFilter"

    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"

    def tikhonov_filter(
        self, noise_covariance: np.ndarray, lbd: float, prior: str = "radial"
    ) -> np.ndarray:
        """
        Applies a graph tikhonov filter to a signal on a (undirected & directed) graph.
        The signal and noise PSD are defined with respect to the GSO
        We use the following formulation ((\Sigma_\eta)^{-1} + \lambda f(L))^{-1}\Sigma_\eta^{-1}
        In the case of decorrelated noise, this reduces to the classical form ((\sigma^2)^{-1}I + \lambda f(L))^{-1}(\sigma^2)^{-1}I
        Parameters
        ----------
            noise_covariance (np.ndarray): The noise covariance matrix.
            lbd (float): The regularization parameter.
            prior (str): The type of prior to use for regularization. Default is "radial". (Applicable only when graph is directed and AD operator is used)
        Returns
        -------
            np.ndarray: The filtered signal.
        """  # noqa W605
        from gyraph.operators import AdvectionDiffusion

        if noise_covariance.ndim != 2:
            raise ValueError("Noise covariance must be a 2D array.")

        if self.graph.is_directed():
            if prior == "radial" or (
                not isinstance(self.graph.operator, AdvectionDiffusion)
            ):
                f_L = hermitian(self.graph.operator.M) @ self.graph.operator.M
            elif prior == "angular" and isinstance(
                self.graph.operator, AdvectionDiffusion
            ):
                Qinv = np.linalg.pinv(self.graph.operator.Q)
                f_L = hermitian(self.graph.operator.P @ Qinv) @ (
                    self.graph.operator.P @ Qinv
                )
            else:
                raise ValueError("Invalid prior type. Choose 'radial' or 'angular'.")
        elif not self.graph.is_directed():
            f_L = self.graph.operator.M

        noise_precision = np.linalg.pinv(noise_covariance)

        graph_filter = np.linalg.pinv(noise_precision + lbd * f_L) @ noise_precision

        return graph_filter

    def apply_tikhonov(
        self,
        signal: np.ndarray,
        noise_covariance: np.ndarray,
        lbd: float,
        prior: str = "radial",
        return_kernel: bool = False,
    ) -> np.ndarray:
        """
        Applies a graph tikhonov filter to a signal on a (undirected & directed) graph.
        The signal and noise PSD are defined with respect to the GSO

        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            noise_covariance (np.ndarray): The noise covariance matrix.
            lbd (float): The regularization parameter.
            prior (str): The type of prior to use for regularization. Default is "radial". (Applicable only when graph is directed and AD operator is used)

        Returns
        -------
            np.ndarray: The filtered signal.
        """
        graph_filter = self.tikhonov_filter(noise_covariance, lbd, prior)
        if return_kernel:
            return graph_filter @ signal, graph_filter
        else:
            return graph_filter @ signal
