"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import *
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

    def apply(self, signal, kernel):
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

    def phase_filter(self, phase:np.ndarray):
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
        cond *= self.graph.operator.imaginaries # mask real eigenvalues
        negatives, positives = np.array((cond < 0), dtype=float), np.array((cond > 0), dtype=float)
        
        filter_p = negatives * np.exp(1j * phase) + positives * np.exp(-1j * phase)
        filter_p = np.diag(filter_p)

        return filter_p
    
    def phase_shift(self, phase:np.ndarray, signal:np.ndarray):
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
    
    def transform_in_real(self, kernel:np.ndarray):
        """
        Transform the filter into a real-valued filter in the spatial domain.
        i.e that the kernel respects conjugate symmetry wrt to the graph operator's eigenvalues.
        """
        conj_constraint = self.graph.operator.eigvalues_pairs()

        for pair in conj_constraint:
            if len(pair) == 1:
                kernel[pair[0]] = np.sign(kernel[pair[0]]) * np.abs(kernel[pair[0]])
            elif len(pair) == 2:
                amplitude = (np.abs(kernel[pair[0]]) + np.abs(kernel[pair[1]])) / 2
                angle = (np.angle(kernel[pair[0]]) - np.angle(kernel[pair[1]])) / 2
                kernel[pair[0]] = amplitude * np.exp(1j * angle)
                kernel[pair[1]] = np.conj(kernel[pair[0]])
            else:
                raise ValueError("Invalid conjugate pair length")

        return kernel

    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"