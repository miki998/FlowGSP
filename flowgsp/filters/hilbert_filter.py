"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import *
from .spectral_filter import SpectralFilter
    
class HilbertFilter(SpectralFilter):
    """
    A spectral filter that applies a kernel in the spectral domain to a signal.
    This filter uses the graph Fourier transform (GFT) to transform the signal
    into the spectral domain, applies the kernel, and then transforms it back
    to the spatial domain.
    """
    def __init__(self, graph, params=None):
        super().__init__(graph, name=None, params=params)
        self.name = "HilbertFilter"
    
    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"

    def hilbert_filter(self):
        """
        Compute filter for Hilbert Transform i.e in GFT domain apply rotation 90

        Returns
        -------
        filter_H : numpy.ndarray
            Diagonal filter matrix for Hilbert transform in GFT domain
        """
        filter_H = self.phase_filter(np.pi/2)
        return filter_H

    def hilbert_transform(self, signal:np.ndarray):
        """
        Compute the Hilbert transform of the input signal in the graph 
        Fourier transform (GFT) domain.

        The signal is first transformed to the GFT domain using the 
        eigenvector matrices U and V. The Hilbert transform filter is 
        applied in the GFT domain. The result is then inverse transformed 
        back to the vertex domain.

        Parameters
        ----------
        signal : numpy.ndarray
            Input signal in the vertex domain

        Returns
        -------
        ret : numpy.ndarray
            Hilbert transformed signal in the vertex domain
        """
        return self.apply(signal, self.hilbert_filter())

    def analytical_signal(self, signal:np.ndarray):
        """
        Compute the analytical signal.

        The analytical signal is computed by taking the Hilbert transform of the 
        input signal, and adding it with a 90 degree phase shift to the original 
        signal.

        Parameters
        ----------
        signal : numpy array
            Input signal 
        U : numpy array
            Graph Fourier transform eigenvector matrix
        V : numpy array
            Graph Fourier transform eigenvalues
        Uinv : numpy array, optional
            Inverse graph Fourier transform eigenvector matrix

        Returns
        -------
        ret : numpy array
            Analytical signal
        """
        xh = self.hilbert_transform(signal)
        ret = signal + 1j * xh

        return ret

    def graph_instant_frequency(self, signal:np.ndarray):
        """
        Compute generalized instant frequency on graph support.
        
        Parameters:
        ----------
        signal : numpy.ndarray
            Input signal.
        
        Returns:
        -------
        numpy.ndarray
            Generalized instant frequency on the graph.
        """
        angle = np.angle(self.analytical_signal(signal))
        adj = self.graph.operator.adjacency_matrix

        ret = np.zeros_like(angle)
        for i in range(len(angle)):
            # neighbours = np.where(adj[:,i])[0]
            neighbours = np.where(adj[i])[0]
            neighbours_angle = angle[neighbours]
            acc = 0
            for n in neighbours_angle:
                if n < angle[i]: 
                    acc += n
                else:
                    modified_n = np.unwrap([angle[i], n])[-1]
                    acc += modified_n
            ret[i] = acc/len(neighbours_angle) - angle[i]

        ret = np.abs(ret)
        return ret

    def demodulating_bydivision(self, signal:np.ndarray):
        """Compute the demodulation by direct division on node domain.

        Parameters
        ----------
        signal : numpy array
            Input signal to demodulate

        Returns
        -------
        ret : numpy array
            Demodulated signal
        """
        instant_amplitude = np.abs(self.analytical_signal(signal))
        assert signal.shape == instant_amplitude.shape

        div = deepcopy(instant_amplitude)

        # Taking care of division by 0, if the modulator is 0
        # then we map to 0 by default the original signal
        div[div == 0] = np.inf
        ret = signal / div

        return ret