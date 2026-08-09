"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
Modified from https://github.com/pygsp/pygsp
"""
import numpy as np
from scipy import sparse, spatial


def nearest_neighbour_graph(
    Xin,
    k: int = 10,
    epsilon: float = 0.01,
    NNtype: str = "knn",
    center: bool = True,
    rescale: bool = True,
    sigma: float = None,
    dist_type: str = "euclidean",
    order: int = 0,
) -> sparse.csc_matrix:
    r"""Nearest-neighbor graph from given point cloud.

    Parameters
    ----------
    Xin : ndarray
        Input points, Should be an `N`-by-`d` matrix, where `N` is the number
        of nodes in the graph and `d` is the dimension of the feature space.
    NNtype : string, optional
        Type of nearest neighbor graph to create. The options are 'knn' for
        k-Nearest Neighbors or 'radius' for epsilon-Nearest Neighbors (default
        is 'knn').
    use_flann : bool, optional
        Use Fast Library for Approximate Nearest Neighbors (FLANN) or not.
        (default is False)
    center : bool, optional
        Center the data so that it has zero mean (default is True)
    rescale : bool, optional
        Rescale the data so that it lies in a l2-sphere (default is True)
    k : int, optional
        Number of neighbors for knn (default is 10)
    sigma : float, optional
        Width of the similarity kernel.
        By default, it is set to the average of the nearest neighbor distance.
    epsilon : float, optional
        Radius for the epsilon-neighborhood search (default is 0.01)
    plotting : dict, optional
        Dictionary of plotting parameters. See :obj:`pygsp.plotting`.
        (default is {})
    symmetrize_type : string, optional
        Type of symmetrization to use for the adjacency matrix. See
        :func:`pygsp.utils.symmetrization` for the options.
        (default is 'average')
    dist_type : string, optional
        Type of distance to compute. See
        :func:`pyflann.index.set_distance_type` for possible options.
        (default is 'euclidean')
    order : float, optional
        Only used if dist_type is 'minkowski'; represents the order of the
        Minkowski distance. (default is 0)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> X = np.random.default_rng(42).uniform(size=(30, 2))
    >>> G = graphs.NNGraph(X)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=5)
    >>> _ = G.plot(ax=axes[1])

    """

    N, d = np.shape(Xin)
    Xout = Xin

    if k >= N:
        raise ValueError(
            "The number of neighbors (k={}) must be smaller "
            "than the number of nodes ({}).".format(k, N)
        )

    if center:
        Xout = Xin - np.kron(np.ones((N, 1)), np.mean(Xin, axis=0))

    if rescale:
        bounding_radius = 0.5 * np.linalg.norm(
            np.amax(Xout, axis=0) - np.amin(Xout, axis=0), 2
        )
        scale = np.power(N, 1.0 / float(min(d, 3))) / 10.0
        Xout *= scale / bounding_radius

    # Translate distance type string to corresponding Minkowski order.
    dist_translation = {
        "euclidean": 2,
        "manhattan": 1,
        "max_dist": np.inf,
        "minkowski": order,
    }

    if NNtype == "knn":
        spi = np.zeros((N * k))
        spj = np.zeros((N * k))
        spv = np.zeros((N * k))

        kdt = spatial.KDTree(Xout)
        D, NN = kdt.query(Xout, k=(k + 1), p=dist_translation[dist_type])

        if sigma is None:
            sigma = np.mean(D[:, 1:])  # Discard distance to self.

        for i in range(N):
            spi[i * k : (i + 1) * k] = np.kron(np.ones((k)), i)
            spj[i * k : (i + 1) * k] = NN[i, 1:]
            spv[i * k : (i + 1) * k] = np.exp(-np.power(D[i, 1:], 2) / float(sigma))

    elif NNtype == "radius":
        kdt = spatial.KDTree(Xout)
        D, NN = kdt.query(
            Xout, k=None, distance_upper_bound=epsilon, p=dist_translation[dist_type]
        )
        if sigma is None:
            # Discard distance to self.
            sigma = np.mean([np.mean(d[1:]) for d in D])
        count = 0
        for i in range(N):
            count = count + len(NN[i])

        spi = np.zeros((count))
        spj = np.zeros((count))
        spv = np.zeros((count))

        start = 0
        for i in range(N):
            leng = len(NN[i]) - 1
            spi[start : start + leng] = np.kron(np.ones((leng)), i)
            spj[start : start + leng] = NN[i][1:]
            spv[start : start + leng] = np.exp(-np.power(D[i][1:], 2) / float(sigma))
            start = start + leng

    else:
        raise ValueError("Unknown NNtype {}".format(NNtype))

    W = sparse.csc_matrix((spv, (spi, spj)), shape=(N, N))

    # Sanity check
    if np.shape(W)[0] != np.shape(W)[1]:
        raise ValueError("Weight matrix W is not square")

    # Enforce symmetry. Note that checking symmetry with
    # np.abs(W - W.T).sum() is as costly as the symmetrization itself.
    W = (W + W.T) / 2

    return W
