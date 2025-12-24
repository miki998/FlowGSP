"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

"""
Constants definition for simple accesses
"""

from . utils import plt, matplotlib


# VARIABLES
EPS = 1e-6

# PLOTTING - PREFERENCES
import scienceplots
plt.style.use(['science','ieee', 'no-latex'])
VMIN_EPS, VMAX_EPS = -0.25, 0.25

# for better visualisation of graphs. Comment out if not needed
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['figure.figsize'] = [3, 2]

# WARNINGS - SUPPORT
# # Suppress FutureWarning messages
# warnings.simplefilter(action='ignore', category=FutureWarning)