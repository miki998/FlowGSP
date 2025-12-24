import unittest
import numpy as np

# Import graph modules to increase coverage
from flowgsp.graphs import basic_graphs, nx
from flowgsp.graphs import graph_utils


class TestBasicGraphs(unittest.TestCase):
    """
    Test cases for flowgsp.graphs.basic_graphs module.
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
                G, pos = basic_graphs.create_cycle_graph(N, graph_types)
                self.assertEqual(len(G.nodes), N)

        self.assertRaises(IndexError, basic_graphs.create_cycle_graph, 10, "gibberish")

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


class TestGraphUtils(unittest.TestCase):
    """
    Test cases for flowgsp.graphs.graph_utils module.
    """

    def setUp(self):
        """Set up test fixtures for graph utils tests."""
        G, _ = basic_graphs.create_cycle_graph(10, 3)
        self.A = nx.adjacency_matrix(G).todense()

        self.G1, _ = basic_graphs.create_cycle_graph(10, 3)
        self.G2, _ = basic_graphs.create_cycle_graph(10, 1)

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


if __name__ == "__main__":
    unittest.main()
