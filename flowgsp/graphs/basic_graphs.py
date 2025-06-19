"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import *
from typing import Union

def create_cycle_graph(N:Union[int, tuple], graph_type:Union[str, int]):
    """
    Generate Adjacency matrix of a graph of N nodes.

    Supported graph types are:
    - line (0)
    - cycle (1)
    - bicycle (2)
    - tricycle (3)

    Parameters
    ----------
    N : int
        Number of nodes in graph
    graph_type : str
        Type of graph to generate. Options are "line", "cycle", "bicycle", "tricycle".

    Returns
    -------
    G : networkx.Graph
        The generated graph.
    pos : dict
        The positions of the nodes in the graph.
    """
    if graph_type == "line" or graph_type == 0:
        A = np.diag(np.ones(N - 1))
        A = np.concatenate([A, np.zeros((1, N - 1))])
        bound = np.zeros((N, 1))
        A = np.concatenate([bound, A], axis=1)

    elif graph_type == "cycle" or graph_type == 1:
        A = np.diag(np.ones(N - 1))
        A = np.concatenate([A, np.zeros((1, N - 1))])
        bound = np.zeros((N, 1))
        bound[-1] = 1.0
        A = np.concatenate([bound, A], axis=1)

    elif graph_type == "bicycle" or graph_type == 2:
        A = np.diag(np.ones(N - 1))
        A = np.concatenate([A, np.zeros((1, N - 1))])
        bound = np.zeros((N, 1))
        bound[-1] = 1.0
        A = np.concatenate([bound, A], axis=1)

        # Adding one sub cycle
        if N <= 12:
            A[N // 2, 0] = 1
        else:
            A[3 * N // 6, 5 * N // 6] = 1

    elif graph_type == "tricycle" or graph_type == 3:
        A = np.diag(np.ones(N - 1))
        A = np.concatenate([A, np.zeros((1, N - 1))])
        bound = np.zeros((N, 1))
        bound[-1] = 1.0
        A = np.concatenate([bound, A], axis=1)

        # Adding two sub cycle
        A[N // 6, 2 * N // 6] = 1
        A[4 * N // 6, 5 * N // 6] = 1

    else:
        print("Not supported format : use either cycle / bicycle / tricycle")
        raise IndexError
    
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    pos = nx.kamada_kawai_layout(G)
    return G, pos

