"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
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
    score = np.min(score_r, score_l)

    if two_tail:
        score *= 2
        score = np.min(score, 1)

    return score