"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

import itertools
import matplotlib.colors as mcolors

def unique_color_generator():
    colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
    for color in itertools.cycle(colors):
        yield color