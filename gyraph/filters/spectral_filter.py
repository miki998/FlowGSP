"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np
from .filter import Filter


class SpectralFilter(Filter):
    """
    A spectral filter that applies a kernel in the spectral domain to a signal.
    This filter uses the graph Fourier transform (GFT) to transform the signal
    into the spectral domain, applies the kernel, and then transforms it back
    to the spatial domain.
    """

    def __init__(self, graph, name=None, params=None):
        super().__init__(graph, name=name, params=params)
        if name is None:
            self.name = "SpectralFilter"
        else:
            self.name = name

    def apply(self, signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        This method applies a spectral kernel to the signal in the spectral domain.

        Parameters
        ----------
        signal : array_like
            The input signal to be filtered.
        kernel : array_like
            The spectral kernel to apply to the signal. This can be a 1D array
            representing a diagonal matrix in the spectral domain.
        Returns
        -------
        filtered : array_like
            The filtered signal after applying the spectral kernel.
        """
        if kernel.ndim == 1:
            kernel = np.diag(kernel)

        coef = self.graph.operator.GFT(signal)
        filtered = self.graph.operator.inverseGFT(kernel @ coef)

        return filtered

    def phase_filter(self, phase: np.ndarray) -> np.ndarray:
        """
        Create a phase filter for the spectral domain.
        This filter applies a phase shift to the eigenvalues of the graph operator,
        effectively rotating the eigenvalues in the complex plane.
        Parameters
        ----------
        phase : np.ndarray
            The phase shift to apply to the eigenvalues.
        Returns
        -------
        np.ndarray
            A diagonal matrix representing the phase filter in the spectral domain.
        """
        cond = self.graph.operator.V.imag
        cond *= self.graph.operator.imaginaries  # mask real eigenvalues
        negatives, positives = np.array((cond < 0), dtype=float), np.array(
            (cond > 0), dtype=float
        )

        filter_p = negatives * np.exp(1j * phase) + positives * np.exp(-1j * phase)
        filter_p = np.diag(filter_p)

        return filter_p

    def phase_shift(self, phase: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        Apply a phase shift to the frequency domain representation of a signal.
        Generalization of Hilbert Transform with general phase shift in GFT domain
        Parameters
        ----------
        phase : float, np.ndarray
            The phase shift to apply.
        signal : np.ndarray
            The input signal.

        Returns
        -------
        np.ndarray
            The signal with the phase shift applied.
        """
        filter_p = self.phase_filter(phase)
        return self.apply(signal, filter_p)

    def transform_in_real(self, kernel: np.ndarray) -> np.ndarray:
        """
        Transform the filter into a real-valued filter in the spatial domain.
        i.e that the kernel respects conjugate symmetry wrt to the graph operator's eigenvalues.
        """
        conj_constraint = self.graph.operator.eigvalues_pairs()

        for pair in conj_constraint:
            if len(pair) == 1:
                kernel[pair[0]] = np.sign(kernel[pair[0]]) * np.abs(kernel[pair[0]])
            elif len(pair) == 2:
                if kernel.dtype != np.complex128:
                    kernel = kernel.astype(np.complex128)
                amplitude = (np.abs(kernel[pair[0]]) + np.abs(kernel[pair[1]])) / 2
                angle = (np.angle(kernel[pair[0]]) - np.angle(kernel[pair[1]])) / 2
                kernel[pair[0]] = amplitude * np.exp(1j * angle)
                kernel[pair[1]] = np.conj(kernel[pair[0]])
            else:
                raise ValueError("Invalid conjugate pair length")

        return kernel

    def estimate_transfer_function(
        self, signal: np.ndarray, filtered_signal: np.ndarray
    ) -> np.ndarray:
        """
        Estimate the transfer function of the filter given an input signal and its filtered version.
        This is done by taking the ratio of the GFT of the filtered signal to the GFT of the input signal.
        Parameters
        ----------
        signal : np.ndarray
            The input signal before filtering.
        filtered_signal : np.ndarray
            The output signal after filtering.

        Returns
        -------
        np.ndarray
            The estimated transfer function in the spectral domain.
        """
        coef_input = self.graph.operator.GFT(signal)
        coef_filtered = self.graph.operator.GFT(filtered_signal)

        poles = np.abs(coef_input) < 1e-7
        zeros = np.abs(coef_filtered) < 1e-7

        arbitrary_indices = np.zeros_like(coef_input).astype(
            bool
        )  # array of indices representing undefined points in the transfer function
        absorbing_indices = np.zeros_like(coef_input).astype(
            bool
        )  # array of indices representing exact zeros in the transfer function

        for fidx in range(len(coef_input)):
            if poles[fidx] and (not zeros[fidx]):
                arbitrary_indices[fidx] = True
            elif zeros[fidx]:
                absorbing_indices[fidx] = True

        print(
            f"Poles at indices: {np.where(poles)[0]}, Zeros at indices: {np.where(zeros)[0]}"
        )
        if np.any(arbitrary_indices):
            self.logger.warning(
                f"Estimated transfer function has undefined values at indices {np.where(arbitrary_indices)[0]} due to poles in the input signal's GFT. Setting these values to the mean of the transfer function energy outside of poles."
            )

        # Avoid division by zero by setting the transfer function to zero at poles and to one at zeros (or any arbitrary value, since the transfer function is undefined at poles and zeros)
        transfer_function = np.divide(
            coef_filtered, coef_input, out=np.zeros_like(coef_filtered), where=~poles
        )
        transfer_function[arbitrary_indices] = np.mean(
            transfer_function[~arbitrary_indices]
        )  # here mean of transfer function energy outside of poles were taken but any arbitrary value could be used, since the transfer function is undefined at poles
        transfer_function[absorbing_indices] = 0.0

        return transfer_function

    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"
