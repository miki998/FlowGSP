"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

import os
import os.path as op
import sys
import pickle
from tqdm import tqdm
from copy import deepcopy
from typing import Optional

import torch
import numpy as np
from numpy.linalg import matrix_rank

from math import comb
import scipy.io as sio
from sympy import Matrix
from scipy.stats import zscore
from scipy.stats import pearsonr

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


import networkx as nx

import cv2
import pandas as pd
import seaborn as sns
import netplotbrain

from nilearn.plotting import plot_epi, show
from nilearn.connectome import ConnectivityMeasure

from joblib import Parallel, delayed

def save(pickle_filename:str, anything:Optional[np.ndarray]):
    """
    Pickle array

    Parameters
    ----------
    pickle_filename : str
        The filename to save the pickled array to
    anything : Optional[np.ndarray]
        The array to pickle

    Returns
    -------
    None

    """
    with open(pickle_filename, "wb") as handle:
        pickle.dump(anything, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load(pickle_filename:str):
    """
    Loads a pickled array from a file.

    Parameters
    ----------
    pickle_filename : str
        The path to the pickled file to load.

    Returns
    -------
    b : Any
        The unpickled object loaded from the file.
    """
    with open(pickle_filename, "rb") as handle:
        b = pickle.load(handle)
    return b