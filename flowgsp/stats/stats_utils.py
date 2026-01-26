"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

import numpy as np


def p_value(null_distrib: np.ndarray, statistic: float, two_tail: bool = False):
    """
    Calculates the p-value for a given test statistic and null distribution.

    Parameters:
    -----------
    null_distrib : np.ndarray
        The null distribution to compare the statistic against.
    statistic : float
        The test statistic value.
    two_tail : bool, optional
        Whether to calculate a two-tailed p-value, by default False.

    Returns:
    --------
    score : float
        The calculated p-value.
    """

    rc = null_distrib > statistic
    lc = null_distrib < statistic

    score_r = np.mean(rc)
    score_l = np.mean(lc)
    score = np.min([score_r, score_l])

    if two_tail:
        score *= 2
        score = np.min([score, 1])

    return score


def circular_stats(alpha_deg, weights=None):
    """
    Compute the (optionally weighted) circular mean and circular variance of a set of angles in degrees.

    Parameters:
    -----------
    alpha_deg: np.ndarray
        Set of angles in degrees.
    weights: np.ndarray or None, optional
        Non-negative weights for each observation. If None, uniform weights are used.

    Returns:
    --------
    mean_angle_deg: float
        Circular mean angle in degrees.
    variance: float
        Circular variance (1 - resultant length).
    """
    alpha = np.deg2rad(alpha_deg)

    if weights is None:
        w = np.ones_like(alpha, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != alpha.shape:
            raise ValueError("weights must have the same shape as the input angles")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

    w_sum = np.sum(w)
    if w_sum == 0:
        raise ValueError("sum of weights must be positive")
    w = w / w_sum  # normalize

    # Weighted mean on the circle
    sin_mean = np.sum(w * np.sin(alpha))
    cos_mean = np.sum(w * np.cos(alpha))
    mean_angle = np.arctan2(sin_mean, cos_mean)

    # Weighted resultant length and variance
    R = np.abs(np.sum(w * np.exp(1j * alpha)))
    variance = 1 - R

    mean_angle_deg = np.rad2deg(mean_angle) % 360
    return mean_angle_deg, variance


def circular_correlation(alpha_deg, beta_deg, weights=None):
    """
    Compute the (optionally weighted) circular correlation coefficient between two sets of angles in degrees.

    Parameters:
    -----------
    alpha_deg: np.ndarray
        First set of angles in degrees.
    beta_deg: np.ndarray
        Second set of angles in degrees.
    weights: np.ndarray or None, optional
        Non-negative weights for each observation. If None, uniform weights are used.

    Returns:
    --------
    corr: float
        Circular correlation coefficient.
    """
    alpha = np.deg2rad(alpha_deg)
    beta = np.deg2rad(beta_deg)

    # Prepare weights
    if weights is None:
        w = np.ones_like(alpha, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != alpha.shape:
            raise ValueError("weights must have the same shape as the input angles")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    w_sum = np.sum(w)
    if w_sum == 0:
        return 0.0  # avoid division by zero
    w = w / w_sum  # normalize

    # Weighted circular means
    alpha_bar = np.arctan2(np.sum(w * np.sin(alpha)), np.sum(w * np.cos(alpha)))
    beta_bar = np.arctan2(np.sum(w * np.sin(beta)), np.sum(w * np.cos(beta)))

    # Weighted covariance and variances
    sin_alpha = np.sin(alpha - alpha_bar)
    sin_beta = np.sin(beta - beta_bar)
    num = np.sum(w * sin_alpha * sin_beta)
    den = np.sqrt(np.sum(w * sin_alpha**2) * np.sum(w * sin_beta**2))

    return num / (den + 1e-12)


def sample_circular_complex_gaussian(mean, covariance, n_samples=1, seed=None):
    """
    Sample from a circular complex Gaussian distribution CN(mean, covariance).

    Parameters:
    -----------
    mean: np.ndarray
        (n,) complex mean vector
    covariance: np.ndarray
        (n, n) Hermitian positive semi-definite covariance matrix
    n_samples: int, optional
        Number of samples to draw
    seed: int, optional
        Random seed for reproducibility

    Returns:
    --------
    samples: np.ndarray
        (n, n_samples) complex-valued samples
    """
    mean = np.asarray(mean)
    n = mean.shape[0]
    covariance = np.asarray(covariance)

    # Ensure the covariance matrix is Hermitian
    assert np.allclose(
        covariance, covariance.T.conj()
    ), "Covariance matrix must be Hermitian."

    # Use a local random generator for reproducibility
    rng = np.random.default_rng(seed)

    # Sample from standard circular complex Gaussian: each part N(0, 0.5)
    z = rng.normal(scale=np.sqrt(0.5), size=(n, n_samples)) + 1j * rng.normal(
        scale=np.sqrt(0.5), size=(n, n_samples)
    )

    # Use eigh for Hermitian matrices to avoid Cholesky issues with identity or singular matrices
    eigvals, eigvecs = np.linalg.eigh(covariance)
    eigvals = np.clip(eigvals, 0, None)  # Ensure non-negative
    sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

    samples = mean[:, None] + sqrt_cov @ z

    # Verify circularity: E[Re(z)^2] == E[Im(z)^2] and E[Re(z)Im(z)] == 0
    if n_samples > 1000:  # Only check for large enough samples
        z_flat = z.reshape(n, -1)
        re = z_flat.real
        im = z_flat.imag
        # assert np.allclose(np.mean(re**2, axis=1), np.mean(im**2, axis=1), rtol=5e-2), "Not circular: variances differ"
        assert np.allclose(
            np.mean(re * im, axis=1), 0, atol=1e-1
        ), "Not circular: covariance not zero"

    return samples.squeeze() if n_samples == 1 else samples
