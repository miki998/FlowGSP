"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import np, nx, deepcopy


def upsample_scheme_graph(
    A: np.ndarray, upsample_factor: int, weight: float = 1
) -> np.ndarray:
    """
    Upsample a graph by inserting edges between existing nodes.
    This function takes an adjacency matrix of a graph and upsamples it by inserting
    edges between existing nodes. The upsample factor determines how many new edges
    are added between each pair of existing nodes.

    Parameters
    ----------
    A : np.ndarray
        The adjacency matrix of the graph to be upsampled.
    upsample_factor : int
        The factor by which to upsample the graph. For example, if upsample_factor is 2,
        then one new edge will be added between each pair of existing nodes.

    Returns
    -------
    A : np.ndarray
        The upsampled adjacency matrix of the graph.

    Notes
    -----
    This function is mainly used for binary graphs, but can also work with weighted graphs.
    However, be aware of corner cases, such as small weights.
    """
    S = deepcopy(A)
    N = S.shape[0]

    for i in range(N):
        for j in range(N):
            if A[i, j] != 0 and A[j, i] == 0:
                # If there is a directed edge from i to j, we add upsample_factor-1 edges
                for k in range(1, upsample_factor):
                    S = np.insert(S, len(S), 0, axis=1)
                    S = np.insert(S, len(S), 0, axis=0)
                    if k > 1:
                        # Attach the new nodes among themselves
                        S[-1, -2] = weight
                        S[-2, -1] = weight
                if upsample_factor > 1:  # Only attach back if extra nodes were added
                    # Attach back to node i and j
                    S[i, -upsample_factor + 1] = weight
                    S[-upsample_factor + 1, i] = weight
                    S[j, -1] = weight
                    S[-1, j] = weight

    return S


def combine_graphs(
    A: np.ndarray, B: np.ndarray, nodes_listA: list, nodes_listB: list
) -> np.ndarray:
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


def get_cycles(
    G: nx.Graph, start_idx: int, max_depth: int, verbose: bool = False
) -> list:
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
    if verbose:
        print(f"Finding cycles up to depth {max_depth} from node {start_idx}...")
    paths_with_cycles = np.where(np.sum(np.array(allpaths) == start_idx, axis=1) > 1)[0]
    paths_with_cycles = np.array(allpaths)[paths_with_cycles]

    # 2. Trim the sequences to only keep the cycles
    if verbose:
        print("Trimming paths to isolate cycles...")
    trimed_paths = []
    for k in range(len(paths_with_cycles)):
        cstart, cend = np.where(paths_with_cycles[k] == start_idx)[0][[0, 1]]
        sequence = paths_with_cycles[k][cstart : cend + 1]

        if (np.array(list(Counter(sequence).values())) > 1).sum() == 1:
            trimed_paths.append(sequence)

    # 3. Remove repeating sequences
    if verbose:
        print("Removing repeating cycles...")
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
    if verbose:
        print("Verifying cycles and closing loops...")
    unique_cycles = [p[:-1] for p in unique_cycles if p[0] == p[-1]]

    return unique_cycles
