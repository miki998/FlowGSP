"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

import os  # noqa: F401
import os.path as op  # noqa: F401
import sys  # noqa: F401
import pickle
import json
import torch as _torch
import warnings  # noqa: F401
from tqdm import tqdm  # noqa: F401
from copy import deepcopy  # noqa: F401
from typing import Optional, Any  # noqa: F401

import torch  # noqa: F401
import numpy as np  # noqa: F401
from numpy.linalg import matrix_rank  # noqa: F401

from math import comb  # noqa: F401
import scipy.io as sio  # noqa: F401
from sympy import Matrix  # noqa: F401
from scipy.stats import zscore  # noqa: F401
from scipy.stats import pearsonr  # noqa: F401
from scipy.io import loadmat  # noqa: F401

import networkx as nx  # noqa: F401

import pandas as pd  # noqa: F401
import seaborn as sns  # noqa: F401

import cv2  # noqa: F401

from joblib import Parallel, delayed  # noqa: F401


def save(pickle_filename: str, anything: Any) -> None:
    """
    Pickle array

    Parameters
    ----------
    pickle_filename : str
        The filename to save the pickled array to
    anything : Any
        The object to pickle

    Returns
    -------
    None

    """
    with open(pickle_filename, "wb") as handle:
        pickle.dump(anything, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(pickle_filename: str) -> Any:
    """
    Loads a pickled array from a file.

    Parameters
    ----------
    pickle_filename : str
        The path to the pickled file to load.

    Returns
    -------
    Any
        The unpickled object loaded from the file.
    """
    with open(pickle_filename, "rb") as handle:
        b = pickle.load(handle)
    return b


def save_json(json_filename: str, anything, indent: int = 4) -> None:
    """
    Save a Python object to a JSON file. Handles common non-JSON types:
    - numpy.ndarray -> stored with metadata so it can be restored
    - numpy scalars -> converted to native Python scalars
    - pandas.DataFrame / pandas.Series -> stored as dicts to reconstruct
    - torch.Tensor -> stored via numpy conversion (if torch available)

    Parameters
    ----------
    json_filename : str
        Path to the output JSON file.
    anything : Any
        The object to save.
    indent : int
        JSON indentation (default 4).
    """

    def _default(o):
        # numpy arrays
        if isinstance(o, np.ndarray):
            return {
                "__ndarray__": True,
                "dtype": str(o.dtype),
                "shape": o.shape,
                "data": o.tolist(),
            }
        # numpy scalars
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        # pandas
        try:
            if isinstance(o, pd.DataFrame):
                return {"__pd_dataframe__": True, "data": o.to_dict(orient="list")}
            if isinstance(o, pd.Series):
                return {"__pd_series__": True, "data": o.to_dict()}
        except Exception:
            pass
        # torch tensors (if torch present)
        try:
            import torch as _torch  # local import to avoid hard dependency

            if isinstance(o, _torch.Tensor):
                arr = o.detach().cpu().numpy()
                return {
                    "__torch__": True,
                    "dtype": str(arr.dtype),
                    "shape": arr.shape,
                    "data": arr.tolist(),
                }
        except Exception:
            pass
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(anything, f, default=_default, indent=indent, ensure_ascii=False)


def load_json(json_filename: str) -> Any:
    """
    Load an object saved by save_json. Reconstructs numpy arrays, pandas DataFrames/Series,
    and torch.Tensors (if torch is available).

    Parameters
    ----------
    json_filename : str
        Path to the JSON file to load.

    Returns
    -------
    Any
        The reconstructed Python object.
    """

    def _object_hook(d):
        if "__ndarray__" in d:
            arr = np.array(d["data"], dtype=np.dtype(d["dtype"]))
            try:
                arr = arr.reshape(tuple(d["shape"]))
            except Exception:
                pass
            return arr
        if "__pd_dataframe__" in d:
            return pd.DataFrame(d["data"])
        if "__pd_series__" in d:
            return pd.Series(d["data"])
        if "__torch__" in d:
            try:
                arr = np.array(d["data"], dtype=np.dtype(d["dtype"]))
                try:
                    arr = arr.reshape(tuple(d["shape"]))
                except Exception:
                    pass
                return _torch.from_numpy(arr)
            except Exception:
                # fallback to numpy array if torch not available
                arr = np.array(d["data"], dtype=np.dtype(d["dtype"]))
                try:
                    arr = arr.reshape(tuple(d["shape"]))
                except Exception:
                    pass
                return arr
        return d

    with open(json_filename, "r", encoding="utf-8") as f:
        return json.load(f, object_hook=_object_hook)
