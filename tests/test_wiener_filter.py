import unittest

from gyraph.utils import load, op, np
from gyraph.graphs import Graph

from gyraph.filters import WienerFilter


class TestWienerFilterFunctions(unittest.TestCase):
    """
    Test cases for Wiener filter and related functionalities.
    """

    def setUp(self):
        """Set up test fixtures for Wiener filter tests."""
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("advection_diffusion")

        self.graph_lap = Graph(
            adj_matrix=self.A + self.A.T
        )  # Make it undirected for Laplacian
        self.graph_lap.set_operator("laplacian")

        self.wiener_filter = WienerFilter(graph=self.graph)
        self.wiener_filter_lap = WienerFilter(graph=self.graph_lap)

        self.signal = np.arange(self.graph.N)

        self.kernel_h = np.ones(self.graph.N)
        self.x_psd = np.ones(self.graph.N)
        self.noise_eta_psd = np.ones(self.graph.N) * 0.1
        self.noise_eps_psd = np.ones(self.graph.N) * 0.05

    def test_wiener_filter(self):
        """
        Test Wiener filter kernel computation.
        """
        g_kernel = self.wiener_filter.wiener_filter(
            self.kernel_h, self.x_psd, self.noise_eps_psd
        )
        self.assertEqual(g_kernel.shape[0], self.kernel_h.shape[0])

    def test_wiener_filter_AD(self):
        """
        Test Wiener filter for advection-diffusion operator.
        """
        filtered, g_kernel = self.wiener_filter.apply_wiener_AD(
            self.signal,
            self.kernel_h,
            self.x_psd,
            self.noise_eta_psd,
            self.noise_eps_psd,
            return_kernel=True,
        )

        self.assertEqual(filtered.shape, self.signal.shape)
        self.assertEqual(g_kernel.shape, self.kernel_h.shape)

        filtered, g_kernel = self.wiener_filter.apply_wiener_AD(
            self.signal,
            np.diag(self.kernel_h),
            self.x_psd,
            self.noise_eta_psd,
            self.noise_eps_psd,
            return_kernel=True,
        )
        self.assertEqual(filtered.shape, self.signal.shape)
        self.assertEqual(g_kernel.shape, self.kernel_h.shape)

        filtered = self.wiener_filter.apply_wiener_AD(
            self.signal,
            np.diag(self.kernel_h),
            np.diag(self.x_psd),
            self.noise_eta_psd,
            self.noise_eps_psd,
            return_kernel=False,
        )

        self.assertEqual(filtered.shape, self.signal.shape)

        filtered = self.wiener_filter_lap.apply_wiener_AD(
            self.signal,
            np.diag(self.kernel_h),
            np.diag(self.x_psd),
            self.noise_eta_psd,
            self.noise_eps_psd,
            return_kernel=False,
        )

        self.assertEqual(filtered.shape, self.signal.shape)

    def test_wiener_apply(self):
        """
        Test Wiener filter application.
        """
        # Instantiate to check that it runs without error
        filtered_signal = self.wiener_filter.apply_wiener(
            self.signal,
            self.kernel_h,
            self.x_psd,
            self.noise_eps_psd,
            return_kernel=False,
        )
        filtered_signal, g_kernel = self.wiener_filter.apply_wiener(
            self.signal,
            self.kernel_h,
            self.x_psd,
            self.noise_eps_psd,
            return_kernel=True,
        )
        self.assertEqual(filtered_signal.shape, self.signal.shape)
        self.assertEqual(g_kernel.shape[0], self.kernel_h.shape[0])

        filtered_signal, g_kernel = self.wiener_filter.apply_wiener(
            self.signal,
            np.diag(self.kernel_h),
            np.diag(self.x_psd),
            np.diag(self.noise_eps_psd),
            return_kernel=True,
        )
        self.assertEqual(filtered_signal.shape, self.signal.shape)
        self.assertEqual(g_kernel.shape[0], self.kernel_h.shape[0])


if __name__ == "__main__":
    unittest.main()
