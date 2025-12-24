import unittest

from flowgsp.utils import load, op, np
from flowgsp.graphs import Graph
from flowgsp.operators import (
    Adjacency,
    Laplacian,
)


class TestAdjacency(unittest.TestCase):
    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "usa_graph_diagonal.pkl"))
        self.symA = self.A + self.A.T

        graph = Graph(adj_matrix=self.A, debug=False)
        graph_sym = Graph(adj_matrix=self.symA, debug=False)

        self.graph_adj = Adjacency(graph=graph)
        self.graph_adj_sym = Adjacency(graph=graph, normalize="symmetric")
        self.graph_sym = Adjacency(graph=graph_sym)

        with self.assertRaises(ValueError):
            Adjacency(graph=graph, normalize="gibberish")  # Should raise ValueError

    def test_compute_basis(self):
        """
        Test the compute_basis method for Adjacency operator.
        # TODO: Add assertions to verify the correctness of the basis computation.
        This is currently a quick test
        """
        self.graph_adj.compute_basis()
        self.graph_adj_sym.compute_basis()

    def test_compute_kernels(self):
        """
        Test the compute_kernels method for Adjacency operator.
        # TODO: Test behaviours of the kernels computed.
        """
        lowpass = self.graph_adj.low_pass_kernel(self.graph_adj.graph.N // 3)
        highpass = self.graph_adj_sym.high_pass_kernel(self.graph_adj.graph.N // 3)

        self.assertEqual(lowpass.shape[0], self.graph_adj.graph.N)
        self.assertEqual(highpass.shape[0], self.graph_adj.graph.N)


class TestLaplacian(unittest.TestCase):
    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "usa_graph_diagonal.pkl"))
        self.symA = self.A + self.A.T

        graph = Graph(adj_matrix=self.A, debug=False)
        graph_sym = Graph(adj_matrix=self.symA, debug=False)

        self.graph_lap = Laplacian(graph=graph)
        self.graph_lap_right = Laplacian(graph=graph, normalize="right")
        self.graph_sym = Laplacian(graph=graph_sym)

        with self.assertRaises(ValueError):
            Laplacian(graph=graph, normalize="gibberish")  # Should raise ValueError

    def test_compute_basis(self):
        """
        Test the compute_basis method for Adjacency operator.
        # TODO: Add assertions to verify the correctness of the basis computation.
        This is currently a quick test
        """
        self.graph_lap.compute_basis()
        self.graph_sym.compute_basis()

    def test_compute_directed_laplacian(self):
        """
        Test the compute_directed_laplacian method for Laplacian operator.
        # TODO: Add assertions to verify the correctness of the directed Laplacian computation.
        This is currently a quick test
        """
        self.graph_lap.compute_directed_laplacian(self.A, in_degree=True)
        self.graph_lap.compute_directed_laplacian(self.A, in_degree=False)

    def test_compute_kernels(self):
        """
        Test the compute_kernels method for Adjacency operator.
        # TODO: Test behaviours of the kernels computed.
        """
        lowpass = self.graph_lap.low_pass_kernel(self.graph_lap.graph.N // 3)
        highpass = self.graph_lap.high_pass_kernel(self.graph_lap.graph.N // 3)
        heat_kernel = self.graph_lap.heat_kernel(self.graph_lap.graph.N // 3)

        self.assertEqual(lowpass.shape[0], self.graph_lap.graph.N)
        self.assertEqual(highpass.shape[0], self.graph_lap.graph.N)
        self.assertEqual(heat_kernel.shape[0], self.graph_lap.graph.N)

if __name__ == "__main__":
    unittest.main()
