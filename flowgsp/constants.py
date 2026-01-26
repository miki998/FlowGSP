"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

"""
Constants definition for simple accesses
"""

from . utils import plt, matplotlib


# VARIABLES
EPS = 1e-6
INF = 1e10
VMIN_EPS, VMAX_EPS = -0.25, 0.25

# PLOTTING - PREFERENCES
import scienceplots

plt.style.use(['science','ieee', 'no-latex'])


# for better visualisation of graphs. Comment out if not needed
matplotlib.rcParams["figure.dpi"] = 300
# default font sizes (adjust as needed)
_DEFAULT_TITLE_SIZE = 7
_DEFAULT_LABEL_SIZE = 6
_DEFAULT_TICK_SIZE = 5
_DEFAULT_LEGEND_SIZE = 5


matplotlib.rcParams.update(
    {
        "font.size": _DEFAULT_LABEL_SIZE,
        "axes.titlesize": _DEFAULT_TITLE_SIZE,
        "axes.labelsize": _DEFAULT_LABEL_SIZE,
        "xtick.labelsize": _DEFAULT_TICK_SIZE,
        "ytick.labelsize": _DEFAULT_TICK_SIZE,
        "legend.fontsize": _DEFAULT_LEGEND_SIZE,
    }
)
