"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np, hermitian
from .spectral_filter import SpectralFilter


class WienerFilter(SpectralFilter):
    """
    A Wiener filter for graph signals.
    This filter applies a Wiener filtering technique to signals on graphs,
    which is particularly useful for denoising signals in the presence of noise.
    It uses the graph Fourier transform (GFT) to transform the signal into the spectral domain,
    applies the Wiener filter kernel, and then transforms it back to the spatial domain.
    This filter is designed to work with both undirected and directed graphs.
    """

    def __init__(self, graph, params=None):
        super().__init__(graph, name=None, params=params)

        self.name = "WienerFilter"

        # Precompute mixing matrix Q
        if hasattr(self.graph.operator, "P"):
            Lsym = (self.graph.operator.P + self.graph.operator.P.T) / 2
        else:
            Lsym = (self.graph.operator.M + self.graph.operator.M.T) / 2

        _, Usym = np.linalg.eig(Lsym)
        self.Q = self.graph.operator.Uinv @ Usym

    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"

    def wiener_filter(
        self, kernel_h: np.ndarray, x_psd: np.ndarray, noise_psd: np.ndarray
    ) -> np.ndarray:
        """
        Applies a graph wiener filter to a signal on a (undirected & directed) graph.
        The signal and noise PSD are defined with respect to the GSO

        Parameters
        ----------
            kernel_h (np.ndarray): The graph filter kernel.
            x_psd (np.ndarray): The power spectral density of the signal.
            noise_psd (np.ndarray): The power spectral density of the noise.

        Returns
        -------
            np.ndarray: The filtered signal.
        """
        if kernel_h.ndim == 2:
            kernel_h = np.diag(kernel_h)
        if x_psd.ndim == 2:
            x_psd = np.diag(x_psd)
        if noise_psd.ndim == 2:
            noise_psd = np.diag(noise_psd)

        nsize = kernel_h.shape[0]
        g_kernel = np.zeros(nsize, dtype=complex)
        for n in range(nsize):
            g_kernel[n] = (
                kernel_h[n] * x_psd[n] / (kernel_h[n] ** 2 * x_psd[n] + noise_psd[n])
            )
        g_kernel = np.diag(g_kernel)
        return g_kernel

    def wiener_filter_AD(
        self,
        kernel_h: np.ndarray,
        x_psd: np.ndarray,
        noise_eta_psd: np.ndarray,
        noise_eps_psd: np.ndarray,
    ) -> np.ndarray:
        """
        Applies a graph wiener filter for denoising additive GWSS and DGWSS signals.

        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            U (np.ndarray): The eigenvectors of the graph Laplacian.
            Uinv (np.ndarray): The inverse of the eigenvectors of the graph Laplacian.
            kernel_h (np.ndarray): The graph filter kernel.
            x_psd (np.ndarray): The power spectral density of the signal.
            noise_eta_psd (np.ndarray): The power spectral density of the DGWSS noise.
            noise_eps_psd (np.ndarray): The power spectral density of the GWSS noise.
                NOTE that the  noise_eps_psd is from the perspective of the undirected eigenmodes
            mixingQ (np.ndarray): The mixing matrix.
        Returns
        -------
            filtered (np.ndarray): The filtered signal.
            g_kernel (np.ndarray): The graph filter kernel.

        """
        if kernel_h.ndim == 2:
            kernel_h = np.diag(kernel_h)

        if self.Q is None:
            raise ValueError(
                "Mixing matrix Q is not precomputed. Ensure the graph is of AdvectionDiffusion type."
            )

        x_psd = np.diag(
            np.diag(x_psd)
        )  # Ensure diagonal even if actually not Stationary
        mixingQ = self.Q

        E = mixingQ @ noise_eps_psd @ hermitian(mixingQ)
        gram = hermitian(self.graph.operator.U) @ self.graph.operator.U
        M = E * (gram).T

        g_kernel = np.diag(
            np.linalg.inv(np.abs(kernel_h) ** 2 * x_psd + noise_eta_psd + M)
            @ (np.abs(kernel_h) ** 2 * np.diag(x_psd))
        )
        return g_kernel

    def apply_wiener(
        self,
        signal: np.ndarray,
        kernel_h: np.ndarray,
        x_psd: np.ndarray,
        noise_psd: np.ndarray,
        return_kernel: bool = False,
    ) -> np.ndarray:
        """
        Applies a graph wiener filter to a signal on a (undirected & directed) graph.
        The signal and noise PSD are defined with respect to the GSO

        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            kernel_h (np.ndarray): The graph filter kernel.
            x_psd (np.ndarray): The power spectral density of the signal.
            noise_psd (np.ndarray): The power spectral density of the noise.

        Returns
        -------
            np.ndarray: The filtered signal.
        """
        kernel = self.wiener_filter(kernel_h, x_psd, noise_psd)
        if return_kernel:
            return self.apply(signal, kernel), kernel
        else:
            return self.apply(signal, kernel)

    def apply_wiener_AD(
        self,
        signal: np.ndarray,
        kernel_h: np.ndarray,
        x_psd: np.ndarray,
        noise_eta_psd: np.ndarray,
        noise_eps_psd: np.ndarray,
        return_kernel: bool = False,
    ) -> np.ndarray:
        """
        Applies a graph wiener filter for denoising additive GWSS and DGWSS signals.

        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            U (np.ndarray): The eigenvectors of the graph Laplacian.
            Uinv (np.ndarray): The inverse of the eigenvectors of the graph Laplacian.
            kernel_h (np.ndarray): The graph filter kernel.
            x_psd (np.ndarray): The power spectral density of the signal.
            noise_eta_psd (np.ndarray): The power spectral density of the DGWSS noise.
            noise_eps_psd (np.ndarray): The power spectral density of the GWSS noise.
                NOTE that the  noise_eps_psd is from the perspective of the undirected eigenmodes
            mixingQ (np.ndarray): The mixing matrix.
            return_kernel (bool): Whether to return the kernel.
        Returns
        -------
            filtered (np.ndarray): The filtered signal.
            g_kernel (np.ndarray): The graph filter kernel.
        """
        kernel = self.wiener_filter_AD(
            kernel_h,
            x_psd,
            noise_eta_psd,
            noise_eps_psd,
        )
        if return_kernel:
            return self.apply(signal, kernel), kernel
        else:
            return self.apply(signal, kernel)
