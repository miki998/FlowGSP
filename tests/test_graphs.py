import unittest
import numpy as np

# Import graph modules to increase coverage
import networkx as nx
from gyraph.graphs import Graph
from gyraph.graphs import basic_graphs
from gyraph.graphs import physical_graphs
from gyraph.graphs import graph_utils


class TestGraphs(unittest.TestCase):
    """
    Test cases for gyraph.graphs module.
    """

    def setUp(self):
        """Set up test fixtures for graph tests."""
        self.G, _ = basic_graphs.create_cycle_graph(10, 1)
        self.graph = Graph(adj_matrix=nx.adjacency_matrix(self.G).todense())

    def test_init_graph(self):
        """
        Test graph initialization and basic properties.
        """
        with self.assertRaises(ValueError):
            Graph()

        graph = Graph(adj_matrix=nx.adjacency_matrix(self.G).todense())

        graph.adj_matrix = None
        with self.assertRaises(ValueError):
            graph.set_operator()

    def test_draw_graph(self):
        """
        Test graph drawing functionality.
        """
        import matplotlib.pyplot as plt

        self.graph.draw()
        plt.close()

        self.graph.draw_signal(np.ones(self.graph.N))
        plt.close()

    def test_assymetry_level(self):
        """
        Test graph asymmetry level computation.
        """
        self.graph.assymetry_level()

    def test_assymetry_edge_level(self):
        """
        Test graph asymmetry edge level computation.
        """
        self.graph.assymetry_edge_level()

    def test_degree_entropy(self):
        """
        Test graph degree entropy computation.
        """
        self.graph.degree_entropy()
        self.graph.degree_entropy(degree_type="out")

    def test_ratio_entropy(self):
        """
        Test graph ratio entropy computation.
        """
        self.graph.ratio_entropy()


class TestBasicGraphs(unittest.TestCase):
    """
    Test cases for gyraph.graphs.basic_graphs module.
    """

    def setUp(self):
        """Set up test fixtures for basic graphs tests."""
        self.graph_sizes = [3, 5, 15]

    def test_cyclic_graphs(self):
        """
        Test ring graph generation.
        """
        # Verify module is importable
        self.assertIsNotNone(basic_graphs)
        for graph_types in ["line", "cycle", "bicycle", "tricycle", 0, 1, 2, 3]:
            for N in self.graph_sizes:
                G, _ = basic_graphs.create_cycle_graph(N, graph_types)
                self.assertEqual(len(G.nodes), N)

        self.assertRaises(IndexError, basic_graphs.create_cycle_graph, 10, "gibberish")

    def test_create_flower_graph(self):
        """
        Test flower graph generation.
        """
        # Verify module is importable
        self.assertIsNotNone(basic_graphs)
        for N in self.graph_sizes:
            G, _ = basic_graphs.create_flower_graph(N, N)
            self.assertEqual(len(G.nodes), N * N)

    def test_create_directed_torus(self):
        """
        Test directed torus graph generation.
        """
        # Verify module is importable
        self.assertIsNotNone(basic_graphs)
        for N in self.graph_sizes:
            G, _ = basic_graphs.create_directed_torus(N // 2 + 1, N // 2 + 1)
            self.assertEqual(len(G.nodes), (N // 2 + 1) * (N // 2 + 1))

            G, _ = basic_graphs.create_directed_torus(
                N // 2 + 1, N // 2 + 1, directed=False
            )
            self.assertEqual(len(G.nodes), (N // 2 + 1) * (N // 2 + 1))

    def test_erdos_renyi_graph(self):
        """
        Test Erdos-Renyi random graph generation.
        #TODO: Add more tests for directed and weighted graphs and edge cases.
        Test timeout cases as well.
        """
        for N in self.graph_sizes:
            for p in [0.1]:
                G, pos = basic_graphs.assymetric_erdos_renyi_graph(N, p)
                self.assertEqual(len(G.nodes), N)

        basic_graphs.assymetric_erdos_renyi_graph(
            10, density=0.05, ratio_directed=0.05, degree_bias=0.1, base="undirected"
        )

        with self.assertRaises(ValueError):
            basic_graphs.assymetric_erdos_renyi_graph(
                10, density=0.05, ratio_directed=0.05, base="invalid"
            )


class TestPhysicalGraphs(unittest.TestCase):
    """
    Test cases for gyraph.graphs.physical_graphs module.
    """

    def setUp(self):
        """Set up test fixtures for physical graphs tests."""
        self.len = 10
        self.radius = 3
        self.centers = np.array([(2, 2), (7, 7)])

        self.path_to_resources = "./tests/test_graphs/"

    def test_torus_graphs(self):
        """
        Test toroidal graph generation.
        """
        # Verify module is importable
        self.assertIsNotNone(physical_graphs)

        G, pos = physical_graphs.create_torus_graph(self.len)
        self.assertEqual(len(G.nodes), self.len * self.len)

        G, pos = physical_graphs.create_torus_laminar_flow_graph(self.len)
        G, pos = physical_graphs.create_torus_laminar_flow_graph(
            self.len, vertical_radius=self.radius
        )
        self.assertEqual(len(G.nodes), self.len * self.len)

        G, pos = physical_graphs.create_torus_vortex_graph(self.len, radius=self.radius)
        self.assertEqual(len(G.nodes), self.len * self.len)

        G, pos = physical_graphs.create_torus_multi_vortex_graph(
            self.len,
            centers_vortices=self.centers,
            radiuses=[self.radius] * len(self.centers),
        )
        self.assertEqual(len(G.nodes), self.len * self.len)

        # Testing the instance creation of non-toroidal vortex graph
        G, pos = physical_graphs.create_vortex_graph(radius=self.radius)

    def test_mesh_graphs(self):
        """
        Test mesh graph generation.
        #TODO : Add more tests for different mesh types and edge cases.
        """
        physical_graphs.create_bunny_graph(self.path_to_resources, return_cloud=True)
        physical_graphs.create_bunny_graph(self.path_to_resources, return_cloud=False)

        physical_graphs.create_dragon_graph(self.path_to_resources, return_cloud=True)
        physical_graphs.create_dragon_graph(self.path_to_resources, return_cloud=False)

        (
            positions_3d,
            positions_2d,
            X,
            Y,
            Z,
        ) = physical_graphs.create_inverted_parabola_grid(self.len, 1, 1)
        G = physical_graphs.create_mesh_graph(
            positions_3d=positions_3d, positions_2d=positions_2d, grid_size=self.len
        )
        self.assertEqual(len(G.nodes), self.len * self.len)

        for dim in [2, 3]:
            physical_graphs.create_cube_graph(nb_pts=100, nb_dim=dim, return_cloud=True)
            physical_graphs.create_cube_graph(
                nb_pts=100, nb_dim=dim, return_cloud=False
            )

            physical_graphs.create_sphere_graph(
                nb_pts=100, nb_dim=dim, return_cloud=True
            )
            physical_graphs.create_sphere_graph(
                nb_pts=100, nb_dim=dim, return_cloud=False
            )

        self.assertRaises(
            NotImplementedError, physical_graphs.create_cube_graph, nb_pts=100, nb_dim=4
        )
        self.assertRaises(
            NotImplementedError,
            physical_graphs.create_sphere_graph,
            nb_pts=100,
            nb_dim=4,
        )

    def test_create_inverted_parabola_grid(self):
        """
        Test inverted parabola grid graph generation.
        """
        (
            positions_3d,
            positions_2d,
            X,
            Y,
            Z,
        ) = physical_graphs.create_inverted_parabola_grid(self.len, 1, 1)
        self.assertEqual(len(positions_3d), self.len * self.len)
        self.assertEqual(len(positions_2d), self.len * self.len)

    def test_create_two_holes_curvature(self):
        """
        Test two holes curvature graph generation.
        """
        (
            positions_3d,
            positions_2d,
            X,
            Y,
            Z,
        ) = physical_graphs.create_two_holes_curvature(self.len, 1)
        self.assertEqual(len(positions_3d), self.len * self.len)
        self.assertEqual(len(positions_2d), self.len * self.len)

    def test_create_hyperbolic_paraboloid_grid(self):
        """
        Test hyperbolic paraboloid grid graph generation.
        """
        (
            positions_3d,
            positions_2d,
            X,
            Y,
            Z,
        ) = physical_graphs.create_hyperbolic_paraboloid_grid(self.len, 1, 1)
        self.assertEqual(len(positions_3d), self.len * self.len)
        self.assertEqual(len(positions_2d), self.len * self.len)

    def test_create_vortex_graph_surface(self):
        """
        Test vortex graph surface generation.
        """
        (
            positions_3d,
            positions_2d,
            X,
            Y,
            Z,
        ) = physical_graphs.create_hyperbolic_paraboloid_grid(self.len, 1, 1)

        G, vortex_edges, cross_vectors = physical_graphs.create_vortex_graph_surface(
            positions_3d,
            positions_2d,
            self.len,
            vortex_radius=self.radius,
            max_distance=1,
        )
        self.assertEqual(len(G.nodes), self.len * self.len)


class TestGraphUtils(unittest.TestCase):
    """
    Test cases for gyraph.graphs.graph_utils module.
    """

    def setUp(self):
        """Set up test fixtures for graph utils tests."""
        G, _ = basic_graphs.create_cycle_graph(10, 3)
        self.A = nx.adjacency_matrix(G).todense()

        self.G1, _ = basic_graphs.create_cycle_graph(10, 3)
        self.G2, _ = basic_graphs.create_cycle_graph(10, 1)

    def test_upsample_scheme_graph(self):
        """
        Test graph upsampling scheme.
        """
        # Verify module is importable
        self.assertIsNotNone(graph_utils)
        upsampled_A = graph_utils.upsample_scheme_graph(
            self.A, upsample_factor=3, weight=1
        )
        self.assertIsInstance(upsampled_A, np.ndarray)

    def test_combine_graphs(self):
        """
        Test graph combination.
        """
        combined_graph = graph_utils.combine_graphs(
            nx.adjacency_matrix(self.G1).todense(),
            nx.adjacency_matrix(self.G2).todense(),
            nodes_listA=[0, 1, 2],
            nodes_listB=[-1, -2, -3],
        )
        self.assertIsNotNone(combined_graph)

    def test_get_cycles(self):
        """
        Test cycle detection in graphs.
        """
        cycles = graph_utils.get_cycles(self.G1, 0, 2, verbose=False)
        self.assertIsInstance(cycles, list)


class TestBarbellGraphs(unittest.TestCase):
    """
    Test cases for the barbell generators in gyraph.graphs.basic_graphs.
    """

    def test_create_barbell_graph(self):
        """Barbell graph: two cliques of N/2 nodes, with node positions."""
        for N in [6, 10]:
            G, pos = basic_graphs.create_barbell_graph(N)
            self.assertEqual(len(G.nodes), N)
            self.assertEqual(len(pos), N)

    def test_create_barbell_graph_directed(self):
        """Directed variant rewires the bridge and back-edge with `weight`."""
        N, weight = 10, 2.0
        G, _ = basic_graphs.create_barbell_graph(N, directed=True, weight=weight)
        self.assertTrue(G.is_directed())
        A = nx.to_numpy_array(G, nodelist=range(N))
        # Bridge between cliques is one-way with the requested weight
        self.assertEqual(A[N // 2 - 1, N // 2], weight)
        self.assertEqual(A[N // 2, N // 2 - 1], 0.0)
        # Closing edge from last node back to node 0
        self.assertEqual(A[N - 1, 0], weight)

    def test_create_barbell_graph_odd_raises(self):
        with self.assertRaises(ValueError):
            basic_graphs.create_barbell_graph(7)

    def test_create_long_barbell_graph(self):
        """Long barbell: N nodes total, plus chain/clique bookkeeping infos."""
        N, chain_length = 20, 3
        G, pos, infos = basic_graphs.create_long_barbell_graph(N, chain_length)
        self.assertEqual(len(G.nodes), N)
        self.assertEqual(len(pos), N)
        for key in [
            "clique1_nodes",
            "clique2_nodes",
            "chain12_nodes",
            "chain21_nodes",
        ]:
            self.assertIn(key, infos)
        self.assertEqual(len(infos["chain12_nodes"]), chain_length)
        self.assertEqual(len(infos["chain21_nodes"]), chain_length)
        # Cliques and chains partition the node set
        n_covered = (
            len(infos["clique1_nodes"])
            + len(infos["clique2_nodes"])
            + len(infos["chain12_nodes"])
            + len(infos["chain21_nodes"])
        )
        self.assertEqual(n_covered, N)

    def test_create_long_barbell_graph_invalid_inputs(self):
        with self.assertRaises(ValueError):
            basic_graphs.create_long_barbell_graph(21, 3)  # odd N
        with self.assertRaises(ValueError):
            basic_graphs.create_long_barbell_graph(10, 6)  # chain too long


class TestNearestNeighbourGraph(unittest.TestCase):
    """
    Test cases for gyraph.utils.graph_utils.nearest_neighbour_graph.
    """

    def setUp(self):
        from gyraph.utils.graph_utils import nearest_neighbour_graph

        self.nearest_neighbour_graph = nearest_neighbour_graph
        self.X = np.random.default_rng(42).uniform(size=(30, 2))

    def test_knn_graph(self):
        """kNN weight matrix: square, symmetric, non-negative Gaussian weights."""
        W = self.nearest_neighbour_graph(self.X, k=4)
        self.assertEqual(W.shape, (30, 30))
        self.assertAlmostEqual(np.abs(W - W.T).max(), 0.0)
        dense = W.toarray()
        self.assertTrue(np.all(dense >= 0.0))
        self.assertTrue(np.all(dense <= 1.0 + 1e-12))  # exp(-d^2/sigma) <= 1
        # Every node keeps at least its k outgoing neighbours (symmetrized)
        self.assertTrue(np.all((dense > 0).sum(axis=1) >= 4))

    def test_knn_graph_options(self):
        """Distance types and no-center/no-rescale variants must run."""
        for dist_type in ["euclidean", "manhattan", "max_dist"]:
            W = self.nearest_neighbour_graph(
                self.X, k=3, dist_type=dist_type, center=False, rescale=False
            )
            self.assertEqual(W.shape, (30, 30))

    def test_k_too_large_raises(self):
        with self.assertRaises(ValueError):
            self.nearest_neighbour_graph(self.X, k=30)

    def test_unknown_nntype_raises(self):
        with self.assertRaises(ValueError):
            self.nearest_neighbour_graph(self.X, NNtype="gibberish")


if __name__ == "__main__":
    unittest.main()
