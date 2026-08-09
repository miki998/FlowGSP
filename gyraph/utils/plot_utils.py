"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL

Plotting utilities for gyraph visualizations.
"""

import itertools
import matplotlib  # noqa: F401
from matplotlib import colors
import matplotlib.pyplot as plt  # noqa: F401
from matplotlib.pyplot import cm  # noqa: F401


def unique_color_generator():
    """
    Yield an infinite sequence of unique color strings.

    Iterates through Matplotlib's TABLEAU_COLORS then CSS4_COLORS,
    cycling back to the start once exhausted.

    Yields
    ------
    str
        Color name string compatible with Matplotlib.
    """
    list_colors = list(colors.TABLEAU_COLORS) + list(colors.CSS4_COLORS)
    for color in itertools.cycle(list_colors):
        yield color
