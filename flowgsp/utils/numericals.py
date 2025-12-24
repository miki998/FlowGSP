"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

import numpy as np

from copy import deepcopy
from typing import Union

# TODO Organize this file better in terms of functionality


def antisymmetry(M):
    """Compute the antisymmetry of a matrix M."""
    return np.linalg.norm(M + hermitian(M), ord="fro") / (
        np.linalg.norm(M, ord="fro") ** 2
    )


def symmetry(M):
    """Compute the symmetry of a matrix M."""
    return np.linalg.norm(M - hermitian(M), ord="fro") / (
        np.linalg.norm(M, ord="fro") ** 2
    )


def hermitian(A: np.ndarray):
    """
    Compute the Hermitian (conjugate transpose) of a matrix.

    Parameters
    ----------
    A : numpy array
        Input matrix

    Returns
    -------
    ret : numpy array
        Hermitian of A
    """
    ret = np.conjugate(A).T
    return ret


def laplacian_to_adj(L: np.ndarray):
    """
    Compute the adjacency matrix from the Laplacian matrix.

    Parameters
    ----------
    L : ndarray
        Laplacian matrix

    Returns
    -------
    A : ndarray
        Adjacency matrix
    """
    if np.any(L.imag != 0):
        raise ValueError("Complex values in laplacian matrix")
    # elif np.all(np.diag(L) == 0): # Commented out as some Laplacians may have zero diagonal entries
    #     raise ValueError("Not a Laplacian matrix")

    A = np.diag(np.diag(L)) - L

    return A


def normalize(a: np.ndarray, **kwargs):
    """
    Normlize the input array to have zero mean and unit variance.

    Parameters
    ----------
    a : np.ndarray
        Input array, can be real or complex valued.

    Returns
    -------
    ret : np.ndarray
        Output with 0 mean and unit variance along axis of choice.
    """

    tmp = a - np.mean(a, **kwargs)
    ret = tmp / np.std(a, **kwargs)
    return ret


def standardize(a: np.ndarray, **kwargs):
    """
    Standardize the input array to have zero min and unit max.

    Parameters
    ----------
    a : np.ndarray
        Input array, can be real or complex valued.

    Returns
    -------
    ret : np.ndarray
        Output with 0 min and 1 max along axis of choice.
    """

    tmp = a - np.min(a, **kwargs)
    ret = tmp / np.max(tmp, **kwargs)
    return ret


def no_decimal(array: np.ndarray, tol: float = 1e-10):
    """
    Sets array values below a tolerance to zero.

    Maps real and imaginary parts of complex array separately.
    Keeps complex array structure.

    Parameters
    ----------
    array : np.ndarray
        Input array, can be real or complex valued.

    tol : float, optional
        Tolerance below which values are set to zero.

    Returns
    -------
    ret : np.ndarray
        Output with small values set to zero.
    """

    ret_real = deepcopy(array.real)
    ret_image = deepcopy(array.imag)
    ret_real[np.abs(ret_real) < tol] = 0.0
    ret_image[np.abs(ret_image) < tol] = 0.0

    ret = ret_real + 1j * ret_image

    return ret


def signed_amplitude(complexarr: np.ndarray):
    """
    Compute signed amplitude of a complex array.

    Takes the absolute value and multiplies by the sign of the
    real part to compute a signed amplitude.

    Prints a warning if the real part contains any zeros,
    as taking the sign of zero is undefined.

    Parameters
    ----------
    complexarr : np.ndarray
        Input complex array

    Returns
    -------
    ret : np.ndarray
        Signed amplitude of complexarr
    """

    # if np.sum(complexarr.real == 0) > 0 and :
    #     print("There are 0 valued real parts in the input array.")
    ret = np.abs(complexarr) * np.sign(complexarr.real)
    return ret


def smooth_1d(y: np.ndarray, box_pts: int):
    """
    Applies a 1D smoothing filter to the input array `y` using a box filter of size `box_pts`.

    Parameters
    ----------
    y : np.ndarray
        The input array to be smoothed.
    box_pts : int
        The size of the box filter to use for smoothing.

    Returns
    -------
    y_smooth : np.ndarray
        The smoothed version of the input array `y`.
    """

    assert y.ndim == 1, "Input array must be 1D"
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(np.pad(y, (0, len(box) - 1), "edge"), box, mode="valid")
    return y_smooth


def spatial_smooth(signal, coords, size=1e-3):
    """
    Applies spatial smoothing to a signal based on given coordinates.

    Parameters
    ----------
    signal : array-like
        The input signal to be smoothed.
    coords : array-like
        The coordinates corresponding to each point in the signal.
    size : float, optional
        The smoothing size parameter. Points within this distance from each other
        will be averaged. Default is 1e-3.

    Returns
    -------
    array-like
        The smoothed signal.
    """
    ret = deepcopy(signal)
    if size <= 0:
        return ret
    for k in range(len(coords)):
        ret[k] = ret[np.linalg.norm(coords[k] - coords, axis=1) < size].mean()
    return ret


def signaltonoise_dB(a: np.ndarray):
    """
    Compute the signal-to-noise ratio (SNR) of an array in decibels (dB).

    Parameters
    ----------
    a : np.ndarray
        Input array, can be real or complex valued.

    Returns
    -------
    snr_dB : float
        Signal-to-noise ratio in decibels
    """

    a = np.asanyarray(a)
    m = a.mean()
    sd = a.std()
    snr_dB = 20 * np.log10(abs(m / sd))

    return snr_dB


def estimate_snr(
    signal: Union[np.ndarray, float],
    noise: Union[np.ndarray, float],
    return_decibel: bool = True,
    max_snr: float = 200,
):
    """
    Estimate the signal-to-noise ratio (SNR) of a signal and noise array.

    Parameters
    ----------
    signal : np.ndarray, float
        Input signal array, can be real or complex valued.
    noise : np.ndarray, float
        Input noise array, can be real or complex valued.
    return_decibel : bool, optional
        Whether to return the SNR in decibels.
    max_snr : float, optional
        Maximum SNR value to return to avoid division by zero.

    Returns
    -------
    ratio : float
        Signal-to-noise ratio
    """
    if isinstance(signal + 0.0, float) and isinstance(noise + 0.0, float):
        if noise == 0:
            return max_snr  # Max SNR value to avoid division by zero
        ratio = signal / noise
    elif isinstance(signal, np.ndarray) and isinstance(noise, np.ndarray):
        est_signal = np.abs(signal).mean()
        est_noise = np.abs(noise).mean()
        if est_noise == 0:
            return max_snr  # Max SNR value to avoid division by zero
        ratio = est_signal / est_noise
    else:
        raise ValueError("Both signal and noise must be either numpy arrays or floats.")

    if return_decibel:
        ratio = 10 * np.log10(ratio)
    return ratio
