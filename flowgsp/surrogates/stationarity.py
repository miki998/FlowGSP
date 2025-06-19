"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from ..utils import *
from typing import Optional

class Stationary:
    """
    A class to represent a stationary process on a graph.
    This class provides methods to estimate the covariance and auto-correlation
    of graph samples, check for stationarity, and compute the stationary level.
    It also provides methods to generate white noise in the directed graph domain
    and compute translation and localization operators.
    """

    def __init__(self, graph, params=None):
        self.graph = graph

    def estimate_covariance(self, samples:np.ndarray):
        """
        Estimate the covariance matrix of the graph samples.
        Parameters:
        -----------
        samples: np.ndarray
            The graph samples to be checked for stationarity. 
        Returns:
        --------
        np.ndarray
            The estimated covariance matrix of the graph samples.
        """
        # Accelerated covariance estimation for samples (assumed shape: [num_samples, num_features])
        # This computes the sample covariance matrix (features x features)
        est_covar = np.cov(samples, rowvar=False, bias=True)
        return est_covar

    def estimate_psd(self, est_covar:np.ndarray):
        """
        Estimate the psd matrix of the graph samples.
        Parameters:
        -----------
        est_covar: np.ndarray
            The estimated covariance matrix of the graph samples.
        Uinv: np.ndarray
            The matrix of eigenvectors of the graph Laplacian.
        Returns:
        --------
        np.ndarray
            The estimated psd matrix of the graph samples.
        """
        
        est_psd = self.graph.operator.Uinv @ est_covar @ hermitian(self.graph.operator.Uinv)
        return est_psd

    def is_stationary(self, graph_samples:np.ndarray,
                    eps_diag:float=0.5, eps_mean:float=0.5, 
                    verbose:bool=False, return_auto:bool=False):
        """
        Check if the graph samples are stationary. 
        The stationarity is checked by comparing the nodal mean and the off-diagonal auto-correlation of the graph samples.

        Parameters:
        -----------
        graph_samples: np.ndarray
            The graph samples to be checked for stationarity. 
            The graph samples are assumed to be in the spectral domain.
        Uinv: np.ndarray
            The matrix of eigenvectors of the graph Laplacian.
        eps_diag: float
            The threshold for the off-diagonal auto-correlation.
        eps_mean: float
            The threshold for the nodal mean.
        verbose: bool
            Whether to print the off-diagonal auto-correlation and nodal mean.
        return_auto: bool
            Whether to return the off-diagonal auto-correlation.

        Returns:
        --------
        bool
            Whether the graph samples are stationary.
        """
        if graph_samples.ndim == 1:
            # Compute nodal mean estimate
            mean_est = np.abs(graph_samples - np.mean(graph_samples)).max() # considering worst case scenario

            # Compute off-diagonal auto-correlation estimate
            covar_est = np.outer(graph_samples, graph_samples)
            auto_corr_est = self.graph.operator.Uinv @ covar_est @ hermitian(self.graph.operator.Uinv)
            auto_corr_diag = np.diag(auto_corr_est)
            off_diag_est = np.abs(auto_corr_est - np.diag(auto_corr_diag)).max() / auto_corr_diag.max()
            
            first_order = mean_est < eps_mean
            second_order = off_diag_est < eps_diag
            return first_order and second_order

        # Compute nodal mean estimate
        mean_vector = np.mean(graph_samples, axis=0)
        mean_est = np.abs(mean_vector - mean_vector.mean()).max() # considering worst case scenario

        # Compute off-diagonal auto-correlation estimate
        covar_est = np.mean([np.outer(sample, sample) for sample in graph_samples], axis=0)
        auto_corr_est = self.graph.operator.Uinv @ covar_est @ hermitian(self.graph.operator.Uinv)
        auto_corr_diag = np.abs(np.diag(auto_corr_est))
        off_diag_est = np.abs(auto_corr_est - np.diag(auto_corr_diag)).max() / auto_corr_diag.max() # as a percentage of the diagonal entries
        
        if verbose:
            print(f"1st order cond = {np.round(mean_est,5)}")
            print(f"2nd order cond = {np.round(off_diag_est,5)}")

        first_order = mean_est < eps_mean
        second_order = off_diag_est < eps_diag
        if return_auto:
            return first_order and second_order, auto_corr_est
        return first_order and second_order

    def stationary_level(self, graph_samples:np.ndarray, 
                         covar_est:Optional[np.ndarray]=None, 
                         return_auto:bool=False):
        """
        Compute the ratio of the nodal mean to the off-diagonal auto-correlation of the graph samples.
        This ratio is used to quantify the stationarity of the graph samples.

        Parameters:
        -----------
        graph_samples: np.ndarray
            The graph samples to be checked for stationarity. 
            The graph samples are assumed to be in the spectral domain.
        Uinv: np.ndarray
            The matrix of eigenvectors of the graph Laplacian.
        return_auto: bool
            Whether to return the off-diagonal auto-correlation.

        Returns:
        --------
        float
            The ratio of the nodal mean to the off-diagonal auto-correlation of the graph samples
        """

        if graph_samples.ndim == 1:
            if covar_est is None:
                covar_est = np.outer(graph_samples, graph_samples)
        else:
            if covar_est is None:
                covar_est = np.mean([np.outer(sample, sample) for sample in graph_samples], axis=0)

        auto_corr_est = self.graph.operator.Uinv @ covar_est @ hermitian(self.graph.operator.Uinv)
        auto_corr_diag = np.abs(np.diag(auto_corr_est))
        diag_power = np.linalg.norm(auto_corr_diag)
        off_diag_power = np.linalg.norm(auto_corr_est)

        if return_auto:
            return diag_power / off_diag_power, auto_corr_est
        return diag_power / off_diag_power

    def translation_operator(self, kernel:np.ndarray, i:int):
        """
        Compute translation operator
        The translation operator is defined as the product of the kernel and the inverse of the eigenvector matrix.
        The translation operator is used to compute the local graph Laplacian.

        Parameters:
        -----------
        kernel: np.ndarray
            The kernel of the graph Laplacian.
        i: int
            The index of the node to compute the translation operator for.

        Returns:
        --------
        gL: np.ndarray
            The translation operator for the graph Laplacian.
        """

        delta = np.zeros(self.graph.operator.U.shape[0]).astype(complex)
        delta[i] = 1.0 + 0j
        spectral_local = self.graph.operator.Uinv @ delta
        
        gL = self.graph.operator.U @ (kernel * spectral_local)
        return gL

    def localization_operator(self, kernel:np.ndarray, i:int):
        """
        Compute Localization operator
        The localization operator is defined as the product of the kernel and the translation operator.
        The localization operator is used to compute the local graph Laplacian.

        Parameters:
        -----------
        kernel: np.ndarray
            The kernel of the graph Laplacian.
        i: int
            The index of the node to compute the localization operator for.

        Returns:
        --------
        gL: np.ndarray
            The localization operator for the graph Laplacian.
        """

        gL = (self.graph.operator.U @ np.diag(kernel) @ self.graph.operator.Uinv)[i]
        return gL

    def white_noise_generator(self, nb_repeat:int, seed:int=99, controlled_covariance:bool=False):
        """
        Generate white noise in graph domain.
        Sampled from a multivariate normal distribution with covariance matrix
        Default to Gaussian distribution with zero mean
        -> Future: can be extended to other distributions if needed.

        Parameters:
        -----------
        nb_repeat: int
            The number of samples to generate.
        U: np.ndarray
            The matrix of eigenvectors of the graph Laplacian.
        seed: int
            The seed for the random number generator. 
        controlled_covariance: bool
            Whether to use controlled covariance for the white noise generation.

        Returns:
        --------
        ret_z: np.ndarray
            The generated white noise samples in the directed graph domain.
        """


        np.random.seed(seed)
        N = self.graph.operator.U.shape[0]
        # Generating White Noise equivalent in directed graph
        covariance_dir = (self.graph.operator.U @ hermitian(self.graph.operator.U)).real

        ret_z = np.random.multivariate_normal(np.zeros(N), covariance_dir, size=nb_repeat)
        return ret_z

        if controlled_covariance:
            raise  NotImplementedError("Controlled covariance needs to be verified yet.")
            tmpV, tmpU = np.linalg.eig(covariance_dir)
            assert np.all(tmpV > 0) # check covariance dir is semi-positive definite so that z = U @ sqrt(V) @ x is real

            if nb_repeat == 1:
                z = tmpU @ np.sqrt(np.diag(tmpV)) @ np.random.normal(0, 1, N)
                return z
            
            ret_z = []
            for _ in range(nb_repeat):
                z = tmpU @ np.sqrt(np.diag(tmpV)) @ np.random.normal(0, 1, N)
                ret_z.append(z)
            ret_z = np.array(ret_z)
            return ret_z