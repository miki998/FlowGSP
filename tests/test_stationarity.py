import unittest

import flowgsp
from flowgsp.utils import load, op, np

class TestStationarity(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, 'bretagne_graph.pkl'))['struct']
        self.graph = flowgsp.graphs.Graph(adj_matrix=self.A)
        self.graph.set_operator('adjacency')
        self.stationarity = flowgsp.surrogates.Stationary(graph=self.graph)

        np.random.seed(99)
        self.graph_samples = np.random.random((self.graph.N))
        self.kernel = np.array([1.0, 0.5])
        self.eps_diag = 0.5
        self.eps_mean = 0.5

    def test_estimate_covariance(self):
        result = self.stationarity.estimate_covariance(self.graph_samples)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.graph.N, self.graph.N))

    def test_estimate_autocorrelation(self):
        est_covar = self.stationarity.estimate_covariance(self.graph_samples)
        result = self.stationarity.estimate_psd(est_covar)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.graph.N, self.graph.N))

    def test_stationary_level(self):
        result = self.stationarity.stationary_level(self.graph_samples)
        self.assertIsInstance(result, float)

    def test_directed_wn_generator_single(self):
        result = self.stationarity.white_noise_generator(1)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, self.graph.N))

    def test_directed_wn_generator_multiple(self):
        result = self.stationarity.white_noise_generator(5)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, self.graph.N))

if __name__ == '__main__':
    unittest.main()