"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from ..utils import *

def combine_graphs(A: np.ndarray, B: np.ndarray, nodes_listA: list, nodes_listB: list):
    """
    Combine graphs by union and adding edges between corresponding nodes.
    Elements in nodes_listA and nodes_listB are indices of nodes to connect 
    between graphs A and B respectively. Negative indexes refer to going from 
    B to A while positive indexes refer to going from A to B.

    Parameters
    ----------
    A : np.ndarray
        Graph A adjacency matrix
    B : np.ndarray
        Graph B adjacency matrix
    
    nodes_listA : list
        Nodes in A to connect
    nodes_listB : list    
        Nodes in B to connect

    Returns
    -------
    ret : np.ndarray
        Combined graph adjacency matrix
    """

    a = nx.convert_matrix.from_numpy_array(A, create_using=nx.DiGraph)
    b = nx.convert_matrix.from_numpy_array(B, create_using=nx.DiGraph)

    c = nx.union(a, b, rename=("a-", "b-"))
    for k in range(len(nodes_listA)):
        nA, nB = nodes_listA[k], nodes_listB[k]
        if nA < 0 and nB < 0:
            c.add_edge(f"b-{-nB}", f"a-{-nA}")
        else:
            c.add_edge(f"a-{nA}", f"b-{nB}")

    ret = np.array(nx.adjacency_matrix(c).todense())
    return ret

def get_cycles(G: nx.Graph, start_idx: int, max_depth: int, verbose:bool=True):
    """
    Find all cycles reachable from a start node within a given maximum depth.

    Parameters
    ----------
    G : networkx.Graph
        The graph to search for cycles.
    start_idx : int 
        The index of the node to start the search from.
    max_depth : int
        The maximum depth to search for cycles.
    verbose : bool
        Whether to print progress updates.

    Returns
    -------
    unique_cycles : list
        A list of lists, where each inner list represents a cycle path.
    """
    from collections import Counter

    def findPaths(G, u, n):
        if n == 0:
            return [[u]]
        # paths = [ [u] + path for neighbor in G.neighbors(u) for path in findPaths(G,neighbor,n-1) if u not in path]
        paths = [
            [u] + path
            for neighbor in G.neighbors(u)
            for path in findPaths(G, neighbor, n - 1)
        ]
        return paths

    allpaths = findPaths(G, start_idx, max_depth)

    # 1. Search for cycles
    if verbose: print(f"Finding cycles up to depth {max_depth} from node {start_idx}...")
    paths_with_cycles = np.where(np.sum(np.array(allpaths) == start_idx, axis=1) > 1)[0]
    paths_with_cycles = np.array(allpaths)[paths_with_cycles]

    # 2. Trim the sequences to only keep the cycles
    if verbose: print(f"Trimming paths to isolate cycles...")
    trimed_paths = []
    for k in range(len(paths_with_cycles)):
        cstart, cend = np.where(paths_with_cycles[k] == start_idx)[0][[0, 1]]
        sequence = paths_with_cycles[k][cstart : cend + 1]

        if (np.array(list(Counter(sequence).values())) > 1).sum() == 1:
            trimed_paths.append(sequence)

    # 3. Remove repeating sequences
    if verbose: print("Removing repeating cycles...")
    unique_cycles = []
    add_flag = True
    for p in trimed_paths:
        for cur in unique_cycles:
            if np.any(p == cur):
                add_flag = False
        if add_flag:
            unique_cycles.append(p)
        add_flag = True

    # 4. Verify that all inputs are indeed cycles and remove the last value to close the loop
    if verbose: print("Verifying cycles and closing loops...")
    unique_cycles = [p[:-1] for p in unique_cycles if p[0] == p[-1]]
        
    return unique_cycles

def var_generator(A:np.ndarray, active_nodes:list, amplitude_nodes:list, 
                  time_nodes:list, n_iter:int,
                  add_noise:bool, time_noise:list, 
                  gamma:float=1, seed:int=99):
    """
    Generates a sequence of directed graph signals over time using a graph spreading process.

    Parameters
    ----------
        A (numpy.ndarray): The adjacency matrix of the graph.
        active_nodes (list): A list of indices of the active nodes in the graph.
        amplitude_nodes (list): A list of amplitudes to be applied to the active nodes.
        time_nodes (list): A list of time steps at which the active node amplitudes should be applied.
        n_iter (int): The number of time steps to simulate.
        add_noise (bool): Whether to add Gaussian noise to the graph signals.
        time_noise (list): A list of time steps at which Gaussian noise should be added.
        gamma (float, optional): A scaling factor for the adjacency matrix. Defaults to 1.
        seed (int, optional): A seed for the random number generator. Defaults to 99.

    Returns
    -------
        directed_logs (numpy.ndarray): A 2D array of shape (n_iter, graphdim) containing the sequence of directed graph signals.
    """
    np.random.seed(seed)

    graphdim = len(A)

    # Initial condition: Implemented to be Gaussian Impulse
    initial_cond = np.random.normal(0, 1, graphdim)
    initial_directed = deepcopy(initial_cond)
    directed_logs = [initial_directed]

    # Defining GSO
    muA = gamma * A

    # Generating the diffusion processes
    for _iter in range(n_iter - 1):

        if (_iter in time_noise) and add_noise:
            source_random = np.random.normal(0, 1, graphdim)
        else:
            source_random = np.zeros(graphdim)

        # Spreading process
        initial_directed = muA @ directed_logs[-1]

        # Node Inherent process
        initial_directed += source_random
        if _iter in time_nodes:
            for lidx, l in enumerate(active_nodes):
                initial_directed[l] += amplitude_nodes[lidx]

        directed_logs.append(initial_directed)

    directed_logs = np.array(directed_logs)
    return directed_logs