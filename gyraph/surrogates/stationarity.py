"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np, hermitian, deepcopy
from typing import Optional, Tuple, Union
from gyraph.graphs import Graph


class Stationary:
    """
    A class to represent a stationary process on a graph.
    This class provides methods to estimate the covariance and auto-correlation
    of graph samples, check for stationarity, and compute the stationary level.
    It also provides methods to generate white noise in the directed graph domain
    and compute translation and localization operators.
    """

    def __init__(self, graph: Graph, params: Optional[dict] = None):
        self.graph = graph
        self.params = params if params is not None else {}

    def exact_covariance(self, kernel: np.ndarray) -> np.ndarray:
        """
        Compute the exact covariance matrix from a given kernel

        Parameters
        ----------
        kernel : np.ndarray
            Kernel representing PSD

        Returns
        -------
        np.ndarray
            The exact covariance matrix.
        """
        if kernel.ndim == 1:
            psd = np.diag(kernel)
        elif kernel.ndim == 2:
            psd = deepcopy(kernel)
        else:
            raise ValueError("Input matrix is not a covariance matrix")

        exact_cov = self.graph.operator.U @ psd @ hermitian(self.graph.operator.U)
        return exact_cov

    def estimate_covariance(self, samples: np.ndarray) -> np.ndarray:
        """
        Estimate the covariance matrix of the graph samples.

        Parameters
        ----------
        samples : np.ndarray
            The graph samples to be checked for stationarity.

        Returns
        -------
        np.ndarray
            The estimated covariance matrix of the graph samples.
        """
        # Accelerated covariance estimation for samples (assumed shape: [num_samples, num_features])
        # This computes the sample covariance matrix (features x features)
        if samples.ndim < 2:
            est_covar = np.outer(samples, samples)
        else:
            est_covar = np.cov(samples, rowvar=False, bias=True)
        return est_covar

    def estimate_psd(self, est_covar: np.ndarray) -> np.ndarray:
        """
        Estimate the psd matrix of the graph samples.

        Parameters
        ----------
        est_covar : np.ndarray
            The estimated covariance matrix of the graph samples.

        Returns
        -------
        np.ndarray
            The estimated psd matrix of the graph samples.
        """
        if est_covar.ndim != 2:
            raise ValueError("Input matrix is not a covariance matrix")

        est_psd = (
            self.graph.operator.Uinv @ est_covar @ hermitian(self.graph.operator.Uinv)
        )
        return est_psd

    def is_stationary(
        self,
        graph_samples: np.ndarray,
        eps_diag: float = 0.5,
        eps_mean: float = 0.5,
        verbose: bool = False,
        return_auto: bool = False,
    ) -> Union[bool, Tuple[bool, np.ndarray]]:
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
            mean_est = np.abs(
                graph_samples - np.mean(graph_samples)
            ).max()  # considering worst case scenario

            # Compute off-diagonal auto-correlation estimate
            covar_est = np.outer(graph_samples, graph_samples)
            auto_corr_est = (
                self.graph.operator.Uinv
                @ covar_est
                @ hermitian(self.graph.operator.Uinv)
            )
            auto_corr_diag = np.diag(auto_corr_est)
            off_diag_est = (
                np.abs(auto_corr_est - np.diag(auto_corr_diag)).max()
                / auto_corr_diag.max()
            )

            first_order = mean_est < eps_mean
            second_order = off_diag_est < eps_diag
            if return_auto:
                return first_order and second_order, auto_corr_est
            return first_order and second_order

        # Compute nodal mean estimate
        mean_vector = np.mean(graph_samples, axis=0)
        mean_est = np.abs(
            mean_vector - mean_vector.mean()
        ).max()  # considering worst case scenario

        # Compute off-diagonal auto-correlation estimate
        covar_est = np.mean(
            [np.outer(sample, sample) for sample in graph_samples], axis=0
        )
        auto_corr_est = (
            self.graph.operator.Uinv @ covar_est @ hermitian(self.graph.operator.Uinv)
        )
        auto_corr_diag = np.abs(np.diag(auto_corr_est))
        off_diag_est = (
            np.abs(auto_corr_est - np.diag(auto_corr_diag)).max() / auto_corr_diag.max()
        )  # as a percentage of the diagonal entries

        if verbose:
            print(f"1st order cond = {np.round(mean_est, 5)}")
            print(f"2nd order cond = {np.round(off_diag_est, 5)}")

        first_order = mean_est < eps_mean
        second_order = off_diag_est < eps_diag
        if return_auto:
            return first_order and second_order, auto_corr_est
        return first_order and second_order

    def stationary_level(
        self,
        graph_samples: np.ndarray,
        covar_est: Optional[np.ndarray] = None,
        return_auto: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray]]:
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
                covar_est = np.mean(
                    [np.outer(sample, sample) for sample in graph_samples], axis=0
                )

        auto_corr_est = (
            self.graph.operator.Uinv @ covar_est @ hermitian(self.graph.operator.Uinv)
        )
        auto_corr_diag = np.abs(np.diag(auto_corr_est))
        diag_power = np.linalg.norm(auto_corr_diag)
        off_diag_power = np.linalg.norm(auto_corr_est)

        if return_auto:
            return diag_power / off_diag_power, auto_corr_est
        return diag_power / off_diag_power

    def translation_operator(self, kernel: np.ndarray, i: int) -> np.ndarray:
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
        if kernel.ndim == 1:
            kernel = np.diag(kernel)
        delta = np.zeros(self.graph.operator.U.shape[0]).astype(complex)
        delta[i] = 1.0 + 0j
        spectral_local = self.graph.operator.Uinv @ delta

        gL = self.graph.operator.U @ (kernel * spectral_local)
        return gL

    def localization_operator(self, kernel: np.ndarray, i: int) -> np.ndarray:
        """
        Compute Localization operator

        Parameters:
        -----------
        kernel: np.ndarray
            The kernel of the graph Laplacian.
        i: int
            The index of the node to compute the localization operator for.

        Returns:
        --------
        gL: np.ndarray
            The localization operator for the concerned GSO.
        """
        if kernel.ndim == 1:
            kernel = np.diag(kernel)
        gL = (self.graph.operator.U @ kernel @ hermitian(self.graph.operator.U))[i]
        return gL

    def psd_realization_generator(
        self, psd: np.ndarray, nb_repeat: int, seed: int = 99
    ) -> np.ndarray:
        """
        Generate samples following a given PSD in graph domain.
        Sampled from a multivariate normal distribution with covariance matrix
        Default to Gaussian distribution with zero mean
        -> Future: can be extended to other distributions if needed.

        Parameters:
        -----------
        psd: np.ndarray
            The psd (vector) precising the behaviour of stationary process
        nb_repeat: int
            The number of samples to generate.
        seed: int
            The seed for the random number generator.

        Returns:
        --------
        ret_z: np.ndarray
            The generated white noise samples in the directed graph domain.
        """
        np.random.seed(seed)
        N = self.graph.operator.U.shape[0]
        if psd.ndim == 1:
            spectral_covariance = np.diag(psd)
        elif psd.ndim == 2:
            spectral_covariance = deepcopy(psd)
        else:
            raise ValueError("PSD is neither a matrix nor a vector")

        # Generating White Noise equivalent in directed graph
        covariance_dir = (
            self.graph.operator.U
            @ spectral_covariance
            @ hermitian(self.graph.operator.U)
        ).real
        # TODO: Catch special case of weird spectral covariance inputs

        ret_z = np.random.multivariate_normal(
            np.zeros(N), covariance_dir, size=nb_repeat
        )
        return ret_z

    def white_noise_generator(self, nb_repeat: int, seed: int = 99) -> np.ndarray:
        """
        Generate white noise in graph domain.
        Sampled from a multivariate normal distribution with covariance matrix
        Default to Gaussian distribution with zero mean
        -> Future: can be extended to other distributions if needed.

        Parameters:
        -----------
        nb_repeat: int
            The number of samples to generate.
        seed: int
            The seed for the random number generator.

        Returns:
        --------
        ret_z: np.ndarray
            The generated white noise samples in the directed graph domain.
        """
        np.random.seed(seed)
        # Generating White Noise equivalent in directed graph
        covariance_dir = (self.graph.operator.U @ hermitian(self.graph.operator.U)).real

        ret_z = np.random.multivariate_normal(
            np.zeros(self.graph.N), covariance_dir, size=nb_repeat
        )
        return ret_z

    def var_generator(
        self,
        A: np.ndarray,
        active_nodes: list,
        amplitude_nodes: list,
        time_nodes: list,
        n_iter: int,
        time_noise: list,
        add_noise: str = "gaussian",
        gamma: float = 1,
        seed: int = 99,
    ) -> np.ndarray:
        """
        Generates a sequence of directed graph signals over time using a graph spreading process.

        Parameters
        ----------
            A (numpy.ndarray): The adjacency matrix of the graph.
            active_nodes (list): A list of indices of the active nodes in the graph.
            amplitude_nodes (list): A list of amplitudes to be applied to the active nodes.
            time_nodes (list): A list of time steps at which the active node amplitudes should be applied.
            n_iter (int): The number of time steps to simulate.
            add_noise (str): Specifies the type of noise to add.
            time_noise (list): A list of time steps at which Gaussian noise should be added.
            gamma (float, optional): A scaling factor for the adjacency matrix. Defaults to 1.
            seed (int, optional): A seed for the random number generator. Defaults to 99.

        Returns
        -------
            directed_logs (numpy.ndarray): A 2D array of shape (n_iter, graphdim) containing the sequence of directed graph signals.
        """
        np.random.seed(seed)

        if add_noise not in ["gaussian", "graph", None]:
            raise ValueError("add_noise must be either 'gaussian' or 'graph' or None")

        if add_noise == "graph":
            random_generators = self.white_noise_generator(n_iter, seed=seed)
        elif add_noise == "gaussian":
            random_generators = [
                np.random.normal(0, 1, self.graph.N) for _ in range(n_iter)
            ]
        elif add_noise is None:
            random_generators = []

        # Initial condition
        initial_cond = random_generators[0]

        initial_directed = deepcopy(initial_cond)
        directed_logs = [initial_directed]

        # Defining GSO
        muA = gamma * A

        # Generating the diffusion processes
        for _iter in range(n_iter - 1):
            if (_iter in time_noise) and (add_noise is not None):
                source_random = random_generators[_iter + 1]
            else:
                source_random = np.zeros(self.graph.N)

            # Spreading process
            initial_directed = muA @ directed_logs[-1]

            # Node Inherent process
            initial_directed += source_random
            if _iter in time_nodes:
                for lidx, l in enumerate(active_nodes):
                    initial_directed[l] += amplitude_nodes[lidx]

            directed_logs.append(initial_directed)

        directed_logs = np.array(directed_logs)
        return directed_logs
