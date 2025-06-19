"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from .utils import *
from .numericals import hermitian

def dirichlet(signal:np.ndarray, L:np.ndarray, normalize:bool=True):
    """
    Compute the Dirichlet energy of a signal with respect to a graph Laplacian.
    (Dirichlet-2 Energy)
    "Shuman, D. I., Narang, S. K., Frossard, P., Ortega, A., & Vandergheynst, P. (2013). 
    The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. 
    IEEE signal processing magazine, 30(3), 83-98."
    S(x)=x^T L x

    Parameters
    ----------
    signal : array_like
        The signal to compute total variation for. 
    L : array_like
        The laplacian operator.
    normalize : bool, optional
        Whether to normalize the total variation by the L2 norm of the signal.
        Default is True. 

    Returns
    -------
    smoothness : float
        The dirichlet energy of the signal.
    """

    div = 1
    if normalize:
        div = np.linalg.norm(signal)
        if div < 1e-10:
            return 0

    smoothness = (hermitian(signal) @ L @ signal) / div
    return smoothness

def TV(signal:np.ndarray, A:np.ndarray, norm:str="L1", 
                     normalize:bool=False, 
                     lbd_flag:bool=False):
    """
    Compute the shift difference of a signal with respect to a graph adjacency.
    "Sandryhaila, A., & Moura, J. M. (2014). 
    Discrete signal processing on graphs: Frequency analysis. 
    IEEE Transactions on signal processing, 62(12), 3042-3054.
    "
    S(x)=||x-Ax||_1 or ||x-Ax||_2
    
    Parameters
    ----------
    signal : array_like
        The signal to compute total variation for.
    A : array_like
        The adjacency matrix of the graph.
    norm : {'L1', 'L2'}, optional
        The norm to use for the difference. Default is 'L1'.
    normalize : bool, optional
        Whether to normalize the total variation by the L2 norm of the signal.
        Default is True.

    Returns
    -------
    smoothness : float
        The shift difference of the signal, which is a measure of smoothness.
    """

    div = 1
    if normalize:
        div = np.linalg.norm(signal)
        if div < 1e-10:
            return 0

    if lbd_flag:
            lbd_max = np.abs(np.linalg.eigvals(A)).max()
    else:
        lbd_max = 1.0

    if norm == "L1":
        smoothness = np.abs(signal - A/lbd_max @ signal).sum() / div
    else:
        smoothness = np.linalg.norm(signal - A/lbd_max @ signal) / div
    return smoothness

def sobolev(signal:np.ndarray, L:np.ndarray, norm:str="L2",
            normalize:bool=False):
    """
    Compute the Sobolev norm of a signal with respect to a graph Laplacian.
    "Singh, R., Chakraborty, A., & Manoj, B. S. (2016, June). 
    Graph Fourier transform based on directed Laplacian. 
    In 2016 International Conference on Signal Processing and Communications (SPCOM) (pp. 1-5). IEEE."
    S(x)=||Lx||_2 and also S(x) = ||Lx||_1

    Parameters
    ----------
    signal : ndarray
        The signal for which to compute total variation.
    L : ndarray
        The graph Laplacian operator.
    normalize : bool, optional  
        Whether to normalize by the L2 norm of the signal. Default is False.

    Returns
    -------
    smoothness : float
        The Sobolev norm of the signal, which is the square root of the total variation.
    """

    div = 1
    if normalize:
        div = np.linalg.norm(signal)
        if div < 1e-10:
            return 0
    if norm == "L1":
        smoothness = np.linalg.norm(L @ signal, ord=1) / div
    elif norm == "L2":
        smoothness = np.linalg.norm(L @ signal) / div
    else:
        raise ValueError("Unsupported norm type. Use 'L1' or 'L2'.")
    
    return smoothness

def directed_variation(signal:np.ndarray, A:np.ndarray, normalize:bool=False):
    """
    Compute the directed variation of a signal with respect to a graph Laplacian.
    "Shafipour, R., Khodabakhsh, A., Mateos, G., & Nikolova, E. (2018). 
    A directed graph Fourier transform with spread frequency components. 
    IEEE Transactions on signal processing, 67(4), 946-960."
    S(x)=sum_{i,j} A_{ij} max(x_i - x_j, 0)
    
    Parameters
    ----------
    signal : ndarray
        The signal for which to compute total variation.
    A : ndarray
        The graph adjacency matrix.
    normalize : bool, optional  
        Whether to normalize by the L2 norm of the signal. Default is False.

    Returns
    -------
    smoothness : float
        The directed variation of the signal.
    """
    assert signal.shape[0] == A.shape[0], "Signal s and adjacency matrix A must have the same number of nodes"
    assert np.isrealobj(signal), "Signal s must be real"
    
    if normalize:
        div = np.linalg.norm(signal)
        if div < 1e-10:
            return 0
        
    smoothness = np.sum(A * np.maximum(signal[:, None] - signal[None, :], 0))
    smoothness /= div
    return smoothness