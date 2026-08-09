import unittest

import gyraph
from gyraph.utils import load, op, np


class TestStationarity(unittest.TestCase):
    def setUp(self):
        # Common setup for tests
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = gyraph.graphs.Graph(adj_matrix=self.A)
        self.graph.set_operator("adjacency")
        self.stationarity = gyraph.surrogates.Stationary(graph=self.graph)

        np.random.seed(99)
        self.graph_samples = np.random.random((self.graph.N))
        self.graph_samples_2D = np.random.random((3, self.graph.N))
        self.kernel = np.ones(self.graph.N)
        self.eps_diag = 0.5
        self.eps_mean = 0.5

    def test_exact_covariance(self):
        """
        Test exact covariance computation from kernel.
        """
        result = self.stationarity.exact_covariance(np.diag(self.kernel))
        result = self.stationarity.exact_covariance(self.kernel)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.graph.N, self.graph.N))

        with self.assertRaises(ValueError):
            self.stationarity.exact_covariance(np.ones((3, 3, 3)))

    def test_estimate_covariance(self):
        """
        Test covariance estimation from graph signals.
        """
        # Single sample graph signals
        result = self.stationarity.estimate_covariance(self.graph_samples)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.graph.N, self.graph.N))

        # Multi-samples graph signals
        result = self.stationarity.estimate_covariance(self.graph_samples_2D)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.graph.N, self.graph.N))

    def test_estimate_psd(self):
        """
        Test PSD estimation from covariance matrix.
        """
        # Estimate covariance first
        est_covar = self.stationarity.estimate_covariance(self.graph_samples)
        result = self.stationarity.estimate_psd(est_covar)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.graph.N, self.graph.N))

        # Using wrong covariance shape
        with self.assertRaises(ValueError):
            self.stationarity.estimate_psd(np.ones((3, 3, 3)))

    def test_is_stationary(self):
        """
        Test stationarity check.
        """
        # With auto return and single sample
        result, auto = self.stationarity.is_stationary(
            self.graph_samples, self.eps_diag, self.eps_mean, return_auto=True
        )
        result = self.stationarity.is_stationary(
            self.graph_samples, self.eps_diag, self.eps_mean, return_auto=False
        )
        self.assertIsNotNone(result)

        # With multi-sample signals
        result, auto = self.stationarity.is_stationary(
            self.graph_samples_2D, self.eps_diag, self.eps_mean, return_auto=True
        )
        result = self.stationarity.is_stationary(
            self.graph_samples_2D, self.eps_diag, self.eps_mean, return_auto=False
        )
        self.assertIsNotNone(result)

    def test_stationary_level(self):
        """
        Test stationary level computation.
        """
        result, auto = self.stationarity.stationary_level(
            self.graph_samples, return_auto=True
        )
        result = self.stationarity.stationary_level(
            self.graph_samples, return_auto=False
        )
        self.assertIsInstance(result, float)

        result = self.stationarity.stationary_level(self.graph_samples_2D)
        self.assertIsInstance(result, float)

    def test_translation_localization(self):
        """
        Test translation localization computation.
        """
        result = self.stationarity.translation_operator(self.kernel, 0)
        result = self.stationarity.translation_operator(np.diag(self.kernel), 0)
        self.assertIsInstance(result, np.ndarray)

        result = self.stationarity.localization_operator(self.kernel, 0)
        result = self.stationarity.localization_operator(np.diag(self.kernel), 0)
        self.assertIsInstance(result, np.ndarray)

    def test_psd_realization_generator(self):
        """
        Test PSD realization generator.
        """
        result = self.stationarity.psd_realization_generator(self.kernel, 10)
        result = self.stationarity.psd_realization_generator(np.diag(self.kernel), 10)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (10, self.graph.N))

        with self.assertRaises(ValueError):
            self.stationarity.psd_realization_generator(np.ones((3, 3, 3)), 10)

    def test_directed_wn_generator(self):
        """
        Test white noise generator for multiple samples.
        """
        result = self.stationarity.white_noise_generator(5)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (5, self.graph.N))

    def test_var_generator(self):
        """
        Test Vector AutoRegressive generator.
        """
        timecourse = self.stationarity.var_generator(self.A, [], [], [], 10, [])
        self.assertIsInstance(timecourse, np.ndarray)
        self.assertEqual(timecourse.shape, (10, self.graph.N))


if __name__ == "__main__":
    unittest.main()
