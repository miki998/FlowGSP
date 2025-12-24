import unittest

from flowgsp.utils import load, op, np
from flowgsp.graphs import Graph

from flowgsp.filters import SpectralFilter, PolynomialFilter


class TestFilterFunctions(unittest.TestCase):
    def setUp(self):
        # Setup common test data
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("laplacian")

        self.spectral_filter = SpectralFilter(graph=self.graph)
        self.graph_filter = PolynomialFilter(graph=self.graph)

        self.signal = np.arange(self.graph.N)

    def test_spectral_filter_directed(self):
        kernel = np.eye(self.graph.N)
        filtered_signal = self.spectral_filter.apply(self.signal, kernel)
        self.assertEqual(filtered_signal.shape, self.signal.shape)

    def test_vandermonde_matrix(self):
        dim = 3
        vdm = self.graph_filter.vandermonde_matrix(self.graph.operator.V, dim)
        self.assertEqual(vdm.shape, (len(self.graph.operator.V), dim))

    def test_get_polynomial_coefficients(self):
        kernel = np.ones(self.graph.N)
        minpolydeg = 3
        vdm, c = self.graph_filter.get_polynomial_coefficients(kernel, minpolydeg)
        self.assertEqual(vdm.shape, (len(self.graph.operator.V), minpolydeg))
        self.assertEqual(c.shape, (minpolydeg,))

    def test_polynomial_filter(self):
        kernel = np.ones(self.graph.N)
        graph_filter, c = self.graph_filter.polynomial_filter(kernel, return_coefs=True)
        self.assertEqual(graph_filter.shape, (self.graph.N, self.graph.N))
        self.assertEqual(len(c), self.graph_filter.params["order"])


if __name__ == "__main__":
    unittest.main()
