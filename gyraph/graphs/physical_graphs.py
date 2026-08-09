"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from gyraph.utils import (
    np,
    nx,
    op,
    loadmat,
    nearest_neighbour_graph,
    load,
    Union,
    Optional,
)
from typing import Tuple


def create_torus_graph(length: Union[int, tuple]) -> Tuple[nx.Graph, dict]:
    """
    Create a toroidal graph with given length.
    The graph is a 2D grid with periodic boundary conditions.
    Each node is connected to its neighbors in the grid.

    Parameters
    ----------
    length : int, Optional[tuple]
        Length of each side of the grid.
        Or a tuple (length_x, length_y) for non-square grids.

    Returns
    -------
    G : networkx.Graph
        The generated toroidal graph.
    pos : dict
        The positions of the nodes in the graph.
    """
    if isinstance(length, (list, tuple)) and len(length) == 2:
        G = nx.grid_2d_graph(length[0], length[1], periodic=True)
    else:
        G = nx.grid_2d_graph(length, length, periodic=True)

    pos = {
        nidx: (n[0], n[1]) for nidx, n in enumerate(G.nodes())
    }  # The adjacency matrix is indexed as in G.nodes()
    # Remap node labels from (i, j) tuples to integer indices for consistency
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G, pos


def create_torus_laminar_flow_graph(
    length: int, vertical_radius: int = -1, horizontal_radius: int = -1
) -> Tuple[nx.Graph, dict]:
    """
    Create a toroidal graph with directed edges representing a laminar flow.

    Parameters
    ----------
    length : int
        Length of each side of the grid.
    vertical_radius : int, optional
        Vertical radius for directed edges in the middle of the grid. Default is -1 (no directed edges).
    horizontal_radius : int, optional
        Horizontal radius for directed edges in the middle of the grid. Default is -1 (no directed edges).

    Returns
    -------
    G : networkx.Graph
        The generated toroidal graph.
    pos : dict
        The positions of the nodes in the graph.
    """
    if (vertical_radius != -1) or (horizontal_radius != -1):
        G = nx.grid_2d_graph(length, length, periodic=True, create_using=nx.DiGraph())
        # Add directed vertical edges in the middle
        for i in range(length):
            for j in range(length):
                if (horizontal_radius != -1) and (
                    length // 2 - horizontal_radius
                    < j
                    < length // 2 + horizontal_radius
                ):
                    G.add_edge((i, j), ((i + 1) % length, j), direction="o")
                    G.remove_edge(((i + 1) % length, j), (i, j))

                if (vertical_radius != -1) and (
                    length // 2 - vertical_radius < i < length // 2 + vertical_radius
                ):
                    G.add_edge((i, j), (i, (j + 1) % length), direction="o")
                    G.remove_edge((i, (j + 1) % length), (i, j))

    else:
        G = nx.grid_2d_graph(length, length, periodic=True)
    pos = {
        nidx: (n[0], n[1]) for nidx, n in enumerate(G.nodes())
    }  # The adjacency matrix is indexed as in G.nodes()
    # Remap node labels from (i, j) tuples to integer indices for consistency
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    return G, pos


def create_torus_vortex_graph(
    length: int = 60, radius: int = 15
) -> Tuple[nx.DiGraph, dict]:
    """
    Create a toroidal graph with directed edges representing a vortex flow.
    The graph is a 2D grid with periodic boundary conditions.
    Each node is connected to its neighbors in the grid, and additional directed edges
    are added to form a vortex pattern.

    Parameters
    ----------
    length : int
        Length of each side of the grid.
    radius : int
        Radius of the vortex. Nodes within this radius will be connected in a circular pattern.
    Returns
    -------
    G_torus_directed : networkx.DiGraph
        The generated toroidal graph with directed edges.
    pos_torus : dict
        The positions of the nodes in the graph.
    """
    G_torus, pos_torus = create_torus_graph(length)
    points = np.array(list(pos_torus.values()))
    centers = points.mean(axis=0)

    for nidx in pos_torus.keys():
        pos_torus[nidx] = (
            pos_torus[nidx][0] - centers[0],
            pos_torus[nidx][1] - centers[1],
        )

    G_torus_directed = nx.DiGraph()

    # Add nodes
    G_torus_directed.add_nodes_from(G_torus.nodes())

    # Add undirected edges
    for edge in G_torus.edges():
        G_torus_directed.add_edge(edge[0], edge[1])
        G_torus_directed.add_edge(edge[1], edge[0])

    # Draw additional edges to form a circle
    # Select points that are within a radius then link them
    r1, r2 = 0, radius  # Define the radius for selecting points to form a circle
    center = np.array([0, 0])  # Center of the circle

    # Find nodes within the specified radius
    nodes_within_radius = [
        node
        for node, coord in pos_torus.items()
        if (np.linalg.norm(np.array(coord) - center, ord=2) <= r2)
        and (np.linalg.norm(np.array(coord) - center, ord=2) >= (r1))
    ]

    # Create edges between consecutive nodes within the radius
    max_distance = 2  # Define the maximum distance for linking nodes

    for i, node1 in enumerate(nodes_within_radius):
        for j, node2 in enumerate(nodes_within_radius):
            if (
                i != j
                and np.linalg.norm(
                    np.array(pos_torus[node1]) - np.array(pos_torus[node2]), ord=1
                )
                <= max_distance
            ):
                c1, c2 = (
                    pos_torus[node1][0] + 1j * pos_torus[node1][1],
                    pos_torus[node2][0] + 1j * pos_torus[node2][1],
                )
                # same_radius = np.linalg.norm(pos_torus[node1] - center, ord=1) == np.linalg.norm(pos_torus[node2] - center, ord=1)
                same_radius = (
                    np.abs(
                        np.linalg.norm(pos_torus[node1] - center, ord=2)
                        - np.linalg.norm(pos_torus[node2] - center, ord=2)
                    )
                    < 1
                )
                angle_diff = np.angle(c2 / c1)

                if same_radius and (angle_diff < 0):
                    G_torus_directed.add_edge(node1, node2)

    return G_torus_directed, pos_torus


def create_torus_multi_vortex_graph(
    length: int, centers_vortices: list, radiuses: list
) -> Tuple[nx.DiGraph, dict]:
    """
    Create a toroidal graph with directed edges representing multiple vortex flows.
    The graph is a 2D grid with periodic boundary conditions.
    Each node is connected to its neighbors in the grid, and additional directed edges
    are added to form a vortex pattern.

    Parameters
    ----------
    length : int
        Length of each side of the grid.
    centers_vortices : list
        List of centers for each vortex. Each center is a tuple (x, y).
    radiuses : list
        List of radii for each vortex. Each radius corresponds to a center in `centers_vortices`.
        Nodes within this radius will be connected in a circular pattern.
    Returns
    -------
    G_torus_directed : networkx.DiGraph
        The generated toroidal graph with directed edges.
    pos_torus : dict
        The positions of the nodes in the graph.
    """
    G_torus, pos_torus = create_torus_graph(length)
    points = np.array(list(pos_torus.values()))
    centers = points.mean(axis=0)

    for nidx in pos_torus.keys():
        pos_torus[nidx] = (
            pos_torus[nidx][0] - centers[0],
            pos_torus[nidx][1] - centers[1],
        )

    G_torus_directed = nx.DiGraph()

    # Add nodes
    G_torus_directed.add_nodes_from(G_torus.nodes())

    # Add undirected edges
    for edge in G_torus.edges():
        G_torus_directed.add_edge(edge[0], edge[1])
        G_torus_directed.add_edge(edge[1], edge[0])

    # Draw additional edges to form a circle
    # Select points that are within a radius then link them

    for r, center in zip(radiuses, centers_vortices):
        # Find nodes within the specified radius
        nodes_within_radius = [
            node
            for node, coord in pos_torus.items()
            if (np.linalg.norm(np.array(coord) - center, ord=2) <= r)
            and (np.linalg.norm(np.array(coord) - center, ord=2) >= (0))
        ]

        # Create edges between consecutive nodes within the radius
        max_distance = 2  # Define the maximum distance for linking nodes

        for i, node1 in enumerate(nodes_within_radius):
            for j, node2 in enumerate(nodes_within_radius):
                if (
                    i != j
                    and np.linalg.norm(
                        np.array(pos_torus[node1]) - np.array(pos_torus[node2]), ord=1
                    )
                    <= max_distance
                ):
                    c1, c2 = (
                        pos_torus[node1][0] + 1j * pos_torus[node1][1],
                        pos_torus[node2][0] + 1j * pos_torus[node2][1],
                    )
                    # same_radius = np.linalg.norm(pos_torus[node1] - center, ord=1) == np.linalg.norm(pos_torus[node2] - center, ord=1)
                    same_radius = (
                        np.abs(
                            np.linalg.norm(pos_torus[node1] - center, ord=2)
                            - np.linalg.norm(pos_torus[node2] - center, ord=2)
                        )
                        < 1
                    )
                    angle_diff = np.angle(c2 / c1)

                    if same_radius and (angle_diff < 0):
                        G_torus_directed.add_edge(node1, node2)

    return G_torus_directed, pos_torus


def create_vortex_graph(radius: int = 15) -> Tuple[nx.DiGraph, dict]:
    """
    Create a graph with directed edges representing a vortex flow.
    This is used to create the vortex component for non-toroidal graphs, on meshes or point clouds.

    Parameters
    ----------
    length : int
        Length of each side of the grid.
    radius : int
        Radius of the vortex. Nodes within this radius will be connected in a circular pattern.
    Returns
    -------
    G_torus_directed : networkx.DiGraph
        The generated toroidal graph with directed edges.
    pos_torus : dict
        The positions of the nodes in the graph.
    """
    G_torus, pos_torus = create_torus_graph(radius * 2 + 1)
    points = np.array(list(pos_torus.values()))
    centers = points.mean(axis=0)

    for nidx in pos_torus.keys():
        pos_torus[nidx] = (
            pos_torus[nidx][0] - centers[0],
            pos_torus[nidx][1] - centers[1],
        )

    G_torus_directed = nx.DiGraph()

    # Add nodes
    G_torus_directed.add_nodes_from(G_torus.nodes())

    # Draw additional edges to form a circle
    # Select points that are within a radius then link them
    r1, r2 = 0, radius  # Define the radius for selecting points to form a circle
    center = np.array([0, 0])  # Center of the circle

    # Find nodes within the specified radius
    nodes_within_radius = [
        node
        for node, coord in pos_torus.items()
        if (np.linalg.norm(np.array(coord) - center, ord=2) <= r2)
        and (np.linalg.norm(np.array(coord) - center, ord=2) >= (r1))
    ]

    # Create edges between consecutive nodes within the radius
    max_distance = 2  # Define the maximum distance for linking nodes

    for i, node1 in enumerate(nodes_within_radius):
        for j, node2 in enumerate(nodes_within_radius):
            if (
                i != j
                and np.linalg.norm(
                    np.array(pos_torus[node1]) - np.array(pos_torus[node2]), ord=1
                )
                <= max_distance
            ):
                c1, c2 = (
                    pos_torus[node1][0] + 1j * pos_torus[node1][1],
                    pos_torus[node2][0] + 1j * pos_torus[node2][1],
                )
                # same_radius = np.linalg.norm(pos_torus[node1] - center, ord=1) == np.linalg.norm(pos_torus[node2] - center, ord=1)
                same_radius = (
                    np.abs(
                        np.linalg.norm(pos_torus[node1] - center, ord=2)
                        - np.linalg.norm(pos_torus[node2] - center, ord=2)
                    )
                    < 1
                )
                if np.isclose(c1, 0):
                    continue
                angle_diff = np.angle(c2 / c1)

                if same_radius and (angle_diff < 0):
                    G_torus_directed.add_edge(node1, node2)

    return G_torus_directed, pos_torus


def create_bunny_graph(
    path_to_resources: str,
    k: int = 10,
    epsilon: float = 0.01,
    NNtype: str = "knn",
    center: bool = True,
    rescale: bool = True,
    sigma: float = None,
    dist_type: str = "euclidean",
    order: int = 0,
    return_cloud: bool = False,
) -> Tuple[nx.Graph, dict, np.ndarray]:
    """
    Create a graph from the bunny dataset.

    Parameters
    ----------
    path_to_resources : str
        Path to the directory containing the bunny dataset.
    **kwargs:
        Additional arguments that can be found in nearest_neighbour_graph documentation.

    Returns
    -------
    G : networkx.Graph
        The generated 3d bunny graph.
    pos : dict
        The positions of the nodes in the graph.
    """

    metadata = loadmat(path_to_resources + "/mesh_graphs_data/bunny.mat")
    A = nearest_neighbour_graph(
        metadata["bunny"],
        k=k,
        epsilon=epsilon,
        NNtype=NNtype,
        center=center,
        rescale=rescale,
        sigma=sigma,
        dist_type=dist_type,
        order=order,
    ).toarray()

    G = nx.from_numpy_array(A)
    positions = {
        i: tuple(metadata["bunny"][i, [0, 1]])
        for i in range(metadata["bunny"].shape[0])
    }
    nx.set_node_attributes(G, positions, "pos")

    if return_cloud:
        return G, positions, metadata["bunny"]
    return G, positions


def create_dragon_graph(
    path_to_resources: str,
    k: int = 10,
    epsilon: float = 0.01,
    NNtype: str = "knn",
    center: bool = True,
    rescale: bool = True,
    sigma: float = None,
    dist_type: str = "euclidean",
    order: int = 0,
    return_cloud: bool = False,
) -> Tuple[nx.Graph, dict, np.ndarray]:
    """
    Create a graph from the dragon dataset.

    Parameters
    ----------
    path_to_resources : str
        Path to the directory containing the bunny dataset.
    **kwargs:
        Additional arguments that can be found in nearest_neighbour_graph documentation.

    Returns
    -------
    G : networkx.Graph
        The generated 3d bunny graph.
    pos : dict
        The positions of the nodes in the graph.
    """

    metadata = load(
        op.join(path_to_resources, "mesh_graphs_data/dragon_sampled2500.pkl")
    )
    A = nearest_neighbour_graph(
        metadata,
        k=k,
        epsilon=epsilon,
        NNtype=NNtype,
        center=center,
        rescale=rescale,
        sigma=sigma,
        dist_type=dist_type,
        order=order,
    ).toarray()

    G = nx.from_numpy_array(A)
    positions = {i: tuple(metadata[i, [0, 1]]) for i in range(metadata.shape[0])}
    nx.set_node_attributes(G, positions, "pos")

    if return_cloud:
        return G, positions, metadata
    return G, positions


def create_cube_graph(
    nb_pts: int = 300,
    nb_dim: int = 3,
    seed: Optional[int] = None,
    k: int = 10,
    epsilon: float = 0.01,
    NNtype: str = "knn",
    center: bool = True,
    rescale: bool = True,
    sigma: float = None,
    dist_type: str = "euclidean",
    order: int = 0,
    return_cloud: bool = False,
) -> Tuple[nx.Graph, dict, Optional[np.ndarray]]:
    r"""Hyper-cube (NN-graph).
    https://github.com/epfl-lts2/pygsp/blob/master/pygsp/graphs/nngraphs/cube.py

    Parameters
    ----------
    radius : float
        Edge lenght (default = 1)
    nb_pts : int
        Number of vertices (default = 300)
    nb_dim : int
        Dimension (default = 3)
    sampling : string
        Variance of the distance kernel (default = 'random')
        (Can now only be 'random')
    seed : int
        Seed for the random number generator (for reproducible graphs).
    **kwargs:
        Additional arguments that can be found in nearest_neighbour_graph documentation.

    Returns
    -------
    G : networkx.Graph
        The generated 3d bunny graph.
    pos : dict
        The positions of the nodes in the graph.
    """

    rs = np.random.RandomState(seed)

    if nb_dim > 3:
        raise NotImplementedError("Dimension > 3 not supported yet!")

    # Random Sampling
    if nb_dim == 2:
        pts = rs.rand(nb_pts, nb_dim)

    elif nb_dim == 3:
        n = nb_pts // 6

        pts = np.zeros((n * 6, 3))
        pts[:n, 1:] = rs.rand(n, 2)
        pts[n : 2 * n, :] = np.concatenate((np.ones((n, 1)), rs.rand(n, 2)), axis=1)

        pts[2 * n : 3 * n, :] = np.concatenate(
            (rs.rand(n, 1), np.zeros((n, 1)), rs.rand(n, 1)), axis=1
        )
        pts[3 * n : 4 * n, :] = np.concatenate(
            (rs.rand(n, 1), np.ones((n, 1)), rs.rand(n, 1)), axis=1
        )

        pts[4 * n : 5 * n, :2] = rs.rand(n, 2)
        pts[5 * n : 6 * n, :] = np.concatenate((rs.rand(n, 2), np.ones((n, 1))), axis=1)

    A = nearest_neighbour_graph(
        pts,
        k=k,
        epsilon=epsilon,
        NNtype=NNtype,
        center=center,
        rescale=rescale,
        sigma=sigma,
        dist_type=dist_type,
        order=order,
    ).toarray()

    G = nx.from_numpy_array(A)
    positions = {i: tuple(pts[i, [0, 1]]) for i in range(pts.shape[0])}
    nx.set_node_attributes(G, positions, "pos")

    if return_cloud:
        return G, positions, pts
    return G, positions


def create_sphere_graph(
    nb_pts: int = 300,
    nb_dim: int = 3,
    seed: Optional[int] = None,
    k: int = 10,
    epsilon: float = 0.01,
    NNtype: str = "knn",
    center: bool = True,
    rescale: bool = True,
    sigma: float = None,
    dist_type: str = "euclidean",
    order: int = 0,
    return_cloud: bool = False,
) -> Tuple[nx.Graph, dict, Optional[np.ndarray]]:
    r"""Hyper-cube (NN-graph).
    https://github.com/epfl-lts2/pygsp/blob/master/pygsp/graphs/nngraphs/cube.py

    Parameters
    ----------
    radius : float
        Edge lenght (default = 1)
    nb_pts : int
        Number of vertices (default = 300)
    nb_dim : int
        Dimension (default = 3)
    sampling : string
        Variance of the distance kernel (default = 'random')
        (Can now only be 'random')
    seed : int
        Seed for the random number generator (for reproducible graphs).
    **kwargs:
        Additional arguments that can be found in nearest_neighbour_graph documentation.

    Returns
    -------
    G : networkx.Graph
        The generated 3d bunny graph.
    pos : dict
        The positions of the nodes in the graph.
    """

    rs = np.random.RandomState(seed)

    if nb_dim > 3:
        raise NotImplementedError("Dimension > 3 not supported yet!")
    # Random Sampling
    rs = np.random.RandomState(seed)
    pts = rs.normal(0, 1, (nb_pts, nb_dim))

    for i in range(nb_pts):
        pts[i] /= np.linalg.norm(pts[i])

    A = nearest_neighbour_graph(
        pts,
        k=k,
        epsilon=epsilon,
        NNtype=NNtype,
        center=center,
        rescale=rescale,
        sigma=sigma,
        dist_type=dist_type,
        order=order,
    ).toarray()

    G = nx.from_numpy_array(A)
    positions = {i: tuple(pts[i, [0, 1]]) for i in range(pts.shape[0])}
    nx.set_node_attributes(G, positions, "pos")

    if return_cloud:
        return G, positions, pts
    return G, positions


# Map regular 3D surface
def create_inverted_parabola_grid(
    grid_size: int, parabola_scale: float, curve_scale: float
) -> Tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D grid and map it to a 3D inverted parabola surface.

    Returns
    -------
    positions_3d : dict
        Dictionary mapping node indices to 3D coordinates (x, y, z).
    positions_2d : dict
        Dictionary mapping node indices to 2D grid coordinates.
    """
    # Create a 2D grid
    x = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
    y = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
    X, Y = np.meshgrid(x, y)

    # Create inverted parabola: z = -(x^2 + y^2) / scale
    # This creates a bowl shape pointing downward

    Z = -((curve_scale * X) ** 2 + (curve_scale * Y) ** 2) / parabola_scale

    # Create node positions
    positions_3d = {}
    positions_2d = {}
    node_idx = 0

    for i in range(grid_size):
        for j in range(grid_size):
            positions_3d[node_idx] = (X[i, j], Y[i, j], Z[i, j])
            positions_2d[node_idx] = (i, j)
            node_idx += 1

    return positions_3d, positions_2d, X, Y, Z


def create_two_holes_curvature(
    grid_size: int, parabola_scale: float
) -> Tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D grid and map it to a 3D inverted parabola surface.

    Returns
    -------
    positions_3d : dict
        Dictionary mapping node indices to 3D coordinates (x, y, z).
    positions_2d : dict
        Dictionary mapping node indices to 2D grid coordinates.
    """
    # Create a 2D grid
    x = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
    y = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
    X, Y = np.meshgrid(x, y)

    # Create two hole curve: z = sin(pi*x) * sin(pi*y) / scale
    Z = np.sin(np.pi * X / X.max()) * np.sin(np.pi * Y / Y.max()) / parabola_scale

    # Create node positions
    positions_3d = {}
    positions_2d = {}
    node_idx = 0

    for i in range(grid_size):
        for j in range(grid_size):
            positions_3d[node_idx] = (X[i, j], Y[i, j], Z[i, j])
            positions_2d[node_idx] = (i, j)
            node_idx += 1

    return positions_3d, positions_2d, X, Y, Z


def create_hyperbolic_paraboloid_grid(
    grid_size: int, parabola_scale: float, curve_scale: float
) -> Tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a 2D grid and map it to a 3D hyperbolic paraboloid (saddle) surface.

    The hyperbolic paraboloid is defined as z = (x² - y²) / scale, creating
    a saddle shape with positive curvature in one direction and negative in the other.

    Returns
    -------
    positions_3d : dict
        Dictionary mapping node indices to 3D coordinates (x, y, z).
    positions_2d : dict
        Dictionary mapping node indices to 2D grid coordinates.
    X, Y, Z : ndarray
        Meshgrid arrays for the surface coordinates.
    """
    # Create a 2D grid
    x = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
    y = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
    X, Y = np.meshgrid(x, y)

    # Create hyperbolic paraboloid (saddle surface): z = (x² - y²) / scale
    # This creates a saddle shape with opposite curvatures
    Z = ((curve_scale * X) ** 2 - (curve_scale * Y) ** 2) / parabola_scale

    # Create node positions
    positions_3d = {}
    positions_2d = {}
    node_idx = 0

    for i in range(grid_size):
        for j in range(grid_size):
            positions_3d[node_idx] = (X[i, j], Y[i, j], Z[i, j])
            positions_2d[node_idx] = (i, j)
            node_idx += 1

    return positions_3d, positions_2d, X, Y, Z


def create_vortex_graph_surface(
    positions_3d: dict,
    positions_2d: dict,
    grid_size: int,
    vortex_radius: float,
    max_distance: float,
) -> Tuple[nx.DiGraph, list, list]:
    """
    Create a directed graph with vortex flow on the parabola surface.

    Parameters
    ----------
    positions_3d : dict
        3D positions of nodes.
    positions_2d : dict
        2D grid positions of nodes.

    Returns
    -------
    G : networkx.DiGraph
        The directed graph with vortex flow.
    """
    G = nx.DiGraph()
    num_nodes = len(positions_3d)
    G.add_nodes_from(range(num_nodes))

    # Center of the grid (center of vortex)
    center_2d = np.array([grid_size / 2, grid_size / 2])

    # Create vortex edges
    vortex_edges = []
    cross_vectors = []

    for node1 in range(num_nodes):
        pos1_2d = np.array(positions_2d[node1])

        # Check if node is within vortex radius (in 2D)
        dist_from_center = np.linalg.norm(pos1_2d - center_2d)

        if dist_from_center <= vortex_radius:
            for node2 in range(num_nodes):
                if node1 == node2:
                    continue

                pos2_2d = np.array(positions_2d[node2])

                # Check if node2 is also within vortex radius
                dist2_from_center = np.linalg.norm(pos2_2d - center_2d)

                if dist2_from_center <= vortex_radius:
                    # Check distance between nodes (in 2D grid)
                    distance_2d = np.linalg.norm(pos1_2d - pos2_2d)

                    if distance_2d <= max_distance:
                        # Compute angle for vortex direction
                        # Vector from center to node1
                        vec_to_node1 = pos1_2d - center_2d
                        # Vector from node1 to node2
                        vec_between = pos2_2d - pos1_2d

                        # For vortex, we want edges that rotate counter-clockwise
                        # Use cross product to determine direction
                        cross = (
                            vec_to_node1[0] * vec_between[1]
                            - vec_to_node1[1] * vec_between[0]
                        )

                        # Also check that nodes are at similar distances from center
                        radius_diff = abs(dist_from_center - dist2_from_center)

                        # Add edge if it follows vortex pattern
                        if cross < 0 and radius_diff < 2.0:
                            vortex_edges.append((node1, node2))
                            G.add_edge(node1, node2, weight=1)
                            cross_vectors.append(cross)

    return G, vortex_edges, cross_vectors


def create_mesh_graph(
    positions_3d: dict,
    positions_2d: dict,
    grid_size: int,
    connect_diagonal: bool = False,
) -> nx.Graph:
    """
    Create an undirected nearest-neighbor (mesh) graph from node positions.

    Parameters
    ----------
    positions_3d : dict
        3D positions of nodes keyed by node index.
    positions_2d : dict
        2D grid indices of nodes keyed by node index.
    connect_diagonal : bool, optional
        If True, also connect diagonal neighbors (8-neighborhood), otherwise
        only 4-neighborhood is used.

    Returns
    -------
    G : networkx.Graph
        Undirected mesh graph with edges between nearest neighbors.
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(positions_3d)))

    # Fast lookup from (i, j) grid coordinate to node index
    idx_from_ij = {ij: n for n, ij in positions_2d.items()}

    for node, (i, j) in positions_2d.items():
        neighbor_ij = []
        # 4-neighborhood
        if i + 1 < grid_size:
            neighbor_ij.append((i + 1, j))
        if j + 1 < grid_size:
            neighbor_ij.append((i, j + 1))
        # Optional 8-neighborhood
        if connect_diagonal:
            if i + 1 < grid_size and j + 1 < grid_size:
                neighbor_ij.append((i + 1, j + 1))
            if i + 1 < grid_size and j - 1 >= 0:
                neighbor_ij.append((i + 1, j - 1))

        for nij in neighbor_ij:
            neigh = idx_from_ij.get(nij)
            if neigh is None:
                continue
            # Weight by Euclidean distance in 3D
            # p1 = np.asarray(positions_3d[node])
            # p2 = np.asarray(positions_3d[neigh])
            # dist = float(np.linalg.norm(p1 - p2))
            # G.add_edge(node, neigh, weight=dist)
            G.add_edge(node, neigh, weight=1)

    return G
