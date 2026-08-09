import unittest

from gyraph.utils import load, op, np
from gyraph.graphs import Graph

from gyraph.filters import HilbertFilter


class TestHilbertFilterFunctions(unittest.TestCase):
    def setUp(self):
        # Setup common test data
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("advection_diffusion")

        self.hilbert = HilbertFilter(graph=self.graph)

        self.signal = np.arange(self.graph.N)

    def test_hilbert_filter(self):
        """
        Test the hilbert filter.
        """
        kernel = self.hilbert.hilbert_filter()
        self.assertEqual(kernel.shape[0], self.signal.shape[0])

    def test_hilbert_transform(self):
        """
        Test the Hilbert transform.
        """
        filtered_signal = self.hilbert.hilbert_transform(self.signal)
        self.assertEqual(filtered_signal.shape, self.signal.shape)

    def test_analytical_signal(self):
        """
        Test the analytical signal.
        """
        filtered_signal = self.hilbert.analytical_signal(self.signal)
        self.assertEqual(filtered_signal.shape, self.signal.shape)

    def test_graph_instant_frequency(self):
        """
        Test the graph instantaneous frequency.
        """
        inst_freq = self.hilbert.graph_instant_frequency(self.signal)
        self.assertEqual(inst_freq.shape, self.signal.shape)

    def test_demodulating_by_division(self):
        """
        Test the demodulation by division.
        """
        demodulated = self.hilbert.demodulating_by_division(self.signal)
        self.assertEqual(demodulated.shape, self.signal.shape)


if __name__ == "__main__":
    unittest.main()
