"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import np, nx
from typing import Union
import warnings

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

def assymetric_erdos_renyi_graph(N:int, density:float = 0.05, ratio_directed:float = 0.1,
                                 degree_bias:float=0.0, ratio_bias:float=0.0, 
                                 base:str='undirected', seed:int=99):
    """
    Generate an variant of asymmetric Erdos-Renyi graph - with chain graph initial backbone.

    Parameters
    ----------
    N : int
        Number of nodes in the graph.
    density : float, optional
        Density of the graph, by default 0.05.
    ratio_directed : float, optional
        Ratio of directed edges to total edges, by default 0.1.
    degree_bias : float, optional
        Degree bias for selected nodes, by default 0.0.
    ratio_bias : float, optional
        Ratio of nodes to apply degree bias, by default 0.0.
    seed : int, optional
        Random seed for reproducibility, by default 99.

    Returns
    -------
    G : networkx.Graph
        The generated asymmetric Erdos-Renyi graph.
    pos : dict
        The positions of the nodes in the graph.
    """

    G, pos = create_cycle_graph(N, graph_type=1)
    if base not in ['undirected', 'directed']:
        raise ValueError("Base must be either 'undirected' or 'directed'.")
    
    backbone = nx.to_numpy_array(G)
    if base == 'undirected':
        backbone = backbone + backbone.T  # make it undirected

    nb_edges = int(density * N * (N - 1) / 2)  # number of edges to add
    max_directed_edges = int(ratio_directed * nb_edges)
    count_directed = 0
    edge_count = 0
    
    np.random.seed(seed)  # for reproducibility
    # Selecting nodes to bias the degree distribution
    if degree_bias != 0:
        nodes_to_bias = np.random.randint(0, N, size=int(N * ratio_bias))  # nodes to bias
        bias = np.ones(N)
        bias[nodes_to_bias] = bias[nodes_to_bias] * degree_bias  # increase degree of selected nodes
        bias /= np.sum(bias)  # normalize bias to sum to 1
    else:
        bias = np.ones(N) / N

    while edge_count < nb_edges:
        # UNCOMMENT THIS TO BIAS ONLY DIRECTED EDGES
        # if count_directed < max_directed_edges: # adding directed edges
        #     u = np.random.choice(np.arange(0, N), p=bias)
        #     v = np.random.randint(0, N)
        # else:  # adding undirected edges
        #     u = np.random.randint(0, N)
        #     v = np.random.randint(0, N)
        
        u = np.random.choice(np.arange(0, N), p=bias)
        v = np.random.randint(0, N)

        if (u != v) and (backbone[u, v] == 0) and (backbone[v, u] == 0) and not ((u-v) in [-1, 1, -N, N]): # ensure no self-loop and no parallel edges
            # Populate with edges
            if (count_directed < max_directed_edges):  # add directed edge
                backbone[u, v] = 1.0
                count_directed += 1
            else:  # add undirected edge
                backbone[u, v] = 1.0
                backbone[v, u] = 1.0
            edge_count += 1

    if count_directed != max_directed_edges:
        warnings.warn("Directed edges count mismatch. Expected {}, got {}.".format(max_directed_edges, count_directed))

    G = nx.from_numpy_array(backbone, create_using=nx.DiGraph)
    return G, pos