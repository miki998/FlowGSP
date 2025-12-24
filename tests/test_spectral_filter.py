import unittest

from flowgsp.utils import load, op, np
from flowgsp.graphs import Graph

from flowgsp.filters import SpectralFilter


class TestSpectralFilterFunctions(unittest.TestCase):
    def setUp(self):
        # Setup common test data
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("adjacency")

        self.spectral_filter = SpectralFilter(graph=self.graph)
        self.spectral_filter_naming = SpectralFilter(
            graph=self.graph, name="MySpectralFilter"
        )

        self.signal = np.arange(self.graph.N)
        self.kernel = np.eye(self.graph.N)

    def test_spectral_apply(self):
        filtered_signal = self.spectral_filter.apply(self.signal, self.kernel)
        self.assertEqual(filtered_signal.shape, self.signal.shape)

    def test_phase_filter(self):
        phase_filter = self.spectral_filter.phase_filter(
            np.pi / 4 * np.ones(self.graph.N)
        )
        self.assertEqual(phase_filter.shape[0], self.graph.N)

    def test_phase_shift(self):
        phase_filtered = self.spectral_filter.phase_shift(np.pi / 4, self.signal)
        self.assertEqual(phase_filtered.shape, self.signal.shape)

    def test_transform_in_real(self):
        self.spectral_filter.transform_in_real(np.eye(self.graph.N))


if __name__ == "__main__":
    unittest.main()
