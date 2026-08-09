import unittest

from gyraph.utils import load, op, np
from gyraph.graphs import Graph

from gyraph.filters import TikhonovFilter


class TestTikhonovFilterFunctions(unittest.TestCase):
    """
    Test cases for Tikhonov filter and related functionalities.
    """

    def setUp(self):
        """Set up test fixtures for Tikhonov filter tests."""
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("advection_diffusion")

        self.graph_lap = Graph(
            adj_matrix=self.A + self.A.T
        )  # Make it undirected for Laplacian
        self.graph_lap.set_operator("laplacian")

        self.tikhonov_filter = TikhonovFilter(graph=self.graph)
        self.tikhonov_filter_lap = TikhonovFilter(graph=self.graph_lap)

        self.signal = np.arange(self.graph.N)
        self.noise_covariance = np.eye(self.graph.N) * 0.1
        self.lbd = 0.5

    def test_tikhonov_apply(self):
        """
        Test Tikhonov filter kernel computation.
        """
        filtered = self.tikhonov_filter.apply_tikhonov(
            self.signal,
            noise_covariance=self.noise_covariance,
            lbd=self.lbd,
            prior="radial",
            return_kernel=False,
        )

        filtered, kernel = self.tikhonov_filter.apply_tikhonov(
            self.signal,
            noise_covariance=self.noise_covariance,
            lbd=self.lbd,
            prior="radial",
            return_kernel=True,
        )
        self.assertEqual(filtered.shape, self.signal.shape)
        self.assertEqual(kernel.shape, (self.graph.N, self.graph.N))

        filtered = self.tikhonov_filter.apply_tikhonov(
            self.signal,
            noise_covariance=self.noise_covariance,
            lbd=self.lbd,
            prior="angular",
            return_kernel=False,
        )
        self.assertEqual(filtered.shape, self.signal.shape)

        filtered = self.tikhonov_filter_lap.apply_tikhonov(
            self.signal,
            noise_covariance=self.noise_covariance,
            lbd=self.lbd,
            prior="radial",
            return_kernel=False,
        )
        self.assertEqual(filtered.shape, self.signal.shape)

        with self.assertRaises(ValueError):
            self.tikhonov_filter.apply_tikhonov(
                self.signal,
                noise_covariance=np.eye(self.graph.N) * 0.1,
                lbd=self.lbd,
                prior="invalid_prior",
                return_kernel=False,
            )

        with self.assertRaises(ValueError):
            self.tikhonov_filter.apply_tikhonov(
                self.signal,
                noise_covariance=np.ones(self.graph.N),
                lbd=self.lbd,
                prior="angular",
                return_kernel=False,
            )


if __name__ == "__main__":
    unittest.main()
