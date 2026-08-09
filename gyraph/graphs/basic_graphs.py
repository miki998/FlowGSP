"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np, nx, deepcopy
from typing import Union
from gyraph.operators.jordan_destroy import destroy_jordan_blocks
from typing import Tuple


def create_cycle_graph(
    N: Union[int, tuple], graph_type: Union[str, int]
) -> Tuple[nx.Graph, dict]:
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
        raise IndexError("Graph type not recognized.")

    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    pos = nx.kamada_kawai_layout(G)
    return G, pos


def create_flower_graph(
    Nr: int, Nc: int, diagonalizable: bool = True
) -> Tuple[nx.Graph, dict]:
    """
    Generate a flower graph with Nr nodes per petal and Nc petals.
    Parameters
    ----------
    Nr : int
        Number of nodes per petal.
    Nc : int
        Number of petals.
    Returns
    -------
    G : networkx.Graph
        The generated flower graph.
    pos : dict
        The positions of the nodes in the graph.
    """
    Gcycle, _ = create_cycle_graph(Nr, graph_type=1)
    cycle = nx.to_numpy_array(Gcycle, nodelist=range(Nr))

    A = np.zeros((Nc * Nr, Nc * Nr))
    for k in range(Nc):
        tmp = deepcopy(cycle)
        if k in np.arange(Nc):
            tmp[-1][0] = 0
        A[k * Nr : (k + 1) * Nr, k * Nr : (k + 1) * Nr] = tmp
        if k + 1 == Nc:
            A[0, k * Nr] = 1
        else:
            A[(k + 1) * Nr, k * Nr] = 1

    if diagonalizable:
        A = destroy_jordan_blocks(A, prefer_nodes=[0 + Nr * k for k in range(Nc)])

    G = nx.from_numpy_array(A, create_using=nx.MultiDiGraph())
    pos = nx.kamada_kawai_layout(G)
    return G, pos


def create_directed_torus(
    Nr: int, Nc: int, directed: bool = True
) -> Tuple[nx.Graph, dict]:
    """
    Generate the adjacency matrix of a directed torus graph with Nr rows and Nc columns. Mainly used in testing and in GHT paper.
    Parameters
    ----------
    Nr : int
        Number of rows.
    Nc : int
        Number of columns.
    directed : bool, optional
        Whether the graph is directed or undirected. Default is True.
    Returns
    -------
    G : networkx.Graph
        The generated torus graph.
    pos : dict
        The positions of the nodes in the graph.
    """
    Gcycle, _ = create_cycle_graph(Nr, graph_type="cycle")
    cycle = nx.to_numpy_array(Gcycle, nodelist=range(Nr))
    A = np.zeros((Nc * Nr, Nc * Nr))

    for k in range(Nc):
        A[k * Nr : (k + 1) * Nr, k * Nr : (k + 1) * Nr] = deepcopy(cycle)

    for k in range(Nc):
        for s in range(Nr):
            if k * Nr + s + Nr >= (Nr * Nc):
                continue
            A[k * Nr + s, k * Nr + s + Nr] = 1.0

            if k * Nr + s + (Nc - 1) * Nr >= (Nr * Nc):
                continue
            A[k * Nr + s + (Nc - 1) * Nr, k * Nr + s] = 1.0

    pos = {k: ((-k % Nr), k // Nr) for k in range(0, Nr * Nc)}
    if not directed:
        G = nx.from_numpy_array(A.T + A, create_using=nx.Graph)
    else:
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    return G, pos


def create_barbell_graph(
    N: int, directed: bool = True, weight: float = 1.0
) -> Tuple[nx.Graph, dict]:
    """
    Generate the adjacency matrix of a barbell graph with N nodes. The graph consists of two cliques of size N//2 connected by a single edge.
    Parameters
    ----------
    N : int
        Total number of nodes in the graph. Should be even.
    directed : bool, optional
        Whether the graph is directed or undirected. Default is True.
    weight : float, optional
        Weight of the connecting edge. Default is 1.0.
    Returns
    -------
    G : networkx.Graph
        The generated barbell graph.
    pos : dict
        The positions of the nodes in the graph.
    """
    if N % 2 != 0:
        raise ValueError("N should be even for barbell graph.")

    G = nx.barbell_graph(N // 2, 0)
    if directed:
        G = G.to_directed()
        A = nx.to_numpy_array(G, nodelist=range(N))
        A[N // 2 - 1, N // 2] = weight
        A[N // 2, N // 2 - 1] = 0
        A[N - 1, 0] = weight
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    pos = {}
    # First clique in circle on the left
    angle_left = np.linspace(0, 2 * np.pi, N // 2, endpoint=False)
    for i in range(N // 2):
        pos[i] = (np.cos(angle_left[i]), np.sin(angle_left[i]))

    # Second clique in circle on the right, shifted away
    angle_right = np.linspace(0, 2 * np.pi, N // 2, endpoint=False)
    for i in range(N // 2, N):
        pos[i] = (3 + np.cos(angle_right[i - N // 2]), np.sin(angle_right[i - N // 2]))

    return G, pos


def create_long_barbell_graph(N: int, chain_length: int) -> Tuple[nx.Graph, dict]:
    """
    Generate the adjacency matrix of a long barbell graph with N nodes and a chain of specified length connecting the two cliques.
    Parameters
    ----------
    N : int
        Total number of nodes in the graph. Should be even and greater than 2*chain_length.
    chain_length : int
        Length of the chain connecting the two cliques. Should be less than N//2.
    Returns
    -------
    G : networkx.Graph
        The generated long barbell graph.
    pos : dict
        The positions of the nodes in the graph.
    """

    if N % 2 != 0:
        raise ValueError("N should be even for barbell graph.")
    if chain_length >= N // 2:
        raise ValueError("Chain length should be less than N//2.")
    clique_N = (N - 2 * chain_length) // 2  # Adjust N to account for chain nodes

    node_idx = [
        clique_N // 5,
        clique_N // 3 + 1,
        clique_N - clique_N // 3 - 1,
        clique_N - clique_N // 5,
    ]  # Connect the middle nodes of the cliques to the chain

    G, pos, infos = create_special_long_barbell_graph_(
        clique_N,
        clique_N,
        chain_length,
        chain_length,
        chain12_from=node_idx[0],
        chain12_to=node_idx[1],
        chain21_from=node_idx[2],
        chain21_to=node_idx[3],
        chain_height=0.2,
        clique_distance=3,
        clique_radius=0.8,
    )

    return G, pos, infos


def create_special_long_barbell_graph_(
    clique1_size: int,
    clique2_size: int,
    chain12_len: int,
    chain21_len: int,
    chain12_from: int = 0,
    chain12_to: int = 0,
    chain21_from: int = 0,
    chain21_to: int = 0,
    clique_radius: float = 1.2,
    clique_distance: float = 8.0,
    chain_height: float = 2.5,
):
    """
    Generate a long barbell graph with two cliques connected by two chains.
    Parameters
    ----------
    clique1_size : int
        Number of nodes in the first clique.
    clique2_size : int
        Number of nodes in the second clique.
    chain12_len : int
        Length of the chain connecting clique 1 to clique 2.
    chain21_len : int
        Length of the chain connecting clique 2 to clique 1.
    chain12_from : int, optional
        Index of the node in clique 1 from which the chain to clique 2 starts. Default is 0.
    chain12_to : int, optional
        Index of the node in clique 2 to which the chain from clique 1 connects. Default is 0.
    chain21_from : int, optional
        Index of the node in clique 2 from which the chain to clique 1 starts. Default is 0.
    chain21_to : int, optional
        Index of the node in clique 1 to which the chain from clique 2 connects. Default is 0.
    clique_radius : float, optional
        Radius of the circles in which the cliques are arranged. Default is 1.2.
    clique_distance : float, optional
        Distance between the centers of the two cliques. Default is 8.0.
    chain_height : float, optional
        Height at which the chains are placed above and below the cliques. Default is 2.5.
    Returns
    -------
    G : networkx.Graph
        The generated long barbell graph.
    pos : dict
        The positions of the nodes in the graph.
    """
    import math

    if clique1_size < 1 or clique2_size < 1:
        raise ValueError("Each clique must contain at least 1 node.")
    if chain12_len < 0 or chain21_len < 0:
        raise ValueError("Chain lengths must be >= 0.")

    if not (0 <= chain12_from < clique1_size):
        raise ValueError("chain12_from out of range for clique1.")
    if not (0 <= chain12_to < clique2_size):
        raise ValueError("chain12_to out of range for clique2.")
    if not (0 <= chain21_from < clique2_size):
        raise ValueError("chain21_from out of range for clique2.")
    if not (0 <= chain21_to < clique1_size):
        raise ValueError("chain21_to out of range for clique1.")

    G = nx.DiGraph()
    pos = {}

    # Node ids
    offset = 0
    clique1_nodes = list(range(offset, offset + clique1_size))
    offset += clique1_size

    clique2_nodes = list(range(offset, offset + clique2_size))
    offset += clique2_size

    chain12_nodes = list(range(offset, offset + chain12_len))
    offset += chain12_len

    chain21_nodes = list(range(offset, offset + chain21_len))
    offset += chain21_len

    # Add all nodes
    G.add_nodes_from(clique1_nodes + clique2_nodes + chain12_nodes + chain21_nodes)

    # Directed clique 1
    for u in clique1_nodes:
        for v in clique1_nodes:
            if u != v:
                G.add_edge(u, v)

    # Directed clique 2
    for u in clique2_nodes:
        for v in clique2_nodes:
            if u != v:
                G.add_edge(u, v)

    # Chain 1 -> 2
    start_12 = clique1_nodes[chain12_from]
    end_12 = clique2_nodes[chain12_to]

    if chain12_len == 0:
        G.add_edge(start_12, end_12)
    else:
        G.add_edge(start_12, chain12_nodes[0])
        for i in range(chain12_len - 1):
            G.add_edge(chain12_nodes[i], chain12_nodes[i + 1])
        G.add_edge(chain12_nodes[-1], end_12)

    # Chain 2 -> 1
    start_21 = clique2_nodes[chain21_from]
    end_21 = clique1_nodes[chain21_to]

    if chain21_len == 0:
        G.add_edge(start_21, end_21)
    else:
        G.add_edge(start_21, chain21_nodes[0])
        for i in range(chain21_len - 1):
            G.add_edge(chain21_nodes[i], chain21_nodes[i + 1])
        G.add_edge(chain21_nodes[-1], end_21)

    # -------------------------
    # Hand-crafted positions
    # -------------------------

    left_center = (0.0, 0.0)
    right_center = (clique_distance, 0.0)

    def circle_positions(nodes, center_x, center_y, radius):
        n = len(nodes)
        if n == 1:
            return {nodes[0]: (center_x, center_y)}
        out = {}
        for i, node in enumerate(nodes):
            theta = 2.0 * math.pi * i / n
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            out[node] = (x, y)
        return out

    pos.update(
        circle_positions(clique1_nodes, left_center[0], left_center[1], clique_radius)
    )
    pos.update(
        circle_positions(clique2_nodes, right_center[0], right_center[1], clique_radius)
    )

    # Chain 1 -> 2 goes above
    if chain12_len > 0:
        x0, _ = pos[start_12]
        x1, _ = pos[end_12]
        xs = [x0 + (i + 1) * (x1 - x0) / (chain12_len + 1) for i in range(chain12_len)]
        for node, x in zip(chain12_nodes, xs):
            pos[node] = (x, chain_height)

    # Chain 2 -> 1 goes below
    if chain21_len > 0:
        x0, _ = pos[start_21]
        x1, _ = pos[end_21]
        xs = [x0 + (i + 1) * (x1 - x0) / (chain21_len + 1) for i in range(chain21_len)]
        for node, x in zip(chain21_nodes, xs):
            pos[node] = (x, -chain_height)

    info = {
        "clique1_nodes": clique1_nodes,
        "clique2_nodes": clique2_nodes,
        "chain12_nodes": chain12_nodes,
        "chain21_nodes": chain21_nodes,
        "chain12_from_node": start_12,
        "chain12_to_node": end_12,
        "chain21_from_node": start_21,
        "chain21_to_node": end_21,
    }

    return G, pos, info


def assymetric_erdos_renyi_graph(
    N: int,
    density: float = 0.05,
    ratio_directed: float = 0.1,
    degree_bias: float = 0.0,
    ratio_bias: float = 0.0,
    base: str = "undirected",
    seed: int = 99,
) -> Tuple[nx.Graph, dict]:
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
    if base not in ["undirected", "directed"]:
        raise ValueError("Base must be either 'undirected' or 'directed'.")

    backbone = nx.to_numpy_array(G)
    if base == "undirected":
        backbone = backbone + backbone.T  # make it undirected

    nb_edges = int(density * N * (N - 1) / 2)  # number of edges to add
    max_directed_edges = int(ratio_directed * nb_edges)
    count_directed = 0
    edge_count = 0

    np.random.seed(seed)  # for reproducibility
    # Selecting nodes to bias the degree distribution
    if degree_bias != 0:
        nodes_to_bias = np.random.randint(
            0, N, size=int(N * ratio_bias)
        )  # nodes to bias
        bias = np.ones(N)
        bias[nodes_to_bias] = (
            bias[nodes_to_bias] * degree_bias
        )  # increase degree of selected nodes
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

        if (
            (u != v)
            and (backbone[u, v] == 0)
            and (backbone[v, u] == 0)
            and not ((u - v) in [-1, 1, -N, N])
        ):  # ensure no self-loop and no parallel edges
            # Populate with edges
            if count_directed < max_directed_edges:  # add directed edge
                backbone[u, v] = 1.0
                count_directed += 1
            else:  # add undirected edge
                backbone[u, v] = 1.0
                backbone[v, u] = 1.0
            edge_count += 1

    G = nx.from_numpy_array(backbone, create_using=nx.DiGraph)
    return G, pos
