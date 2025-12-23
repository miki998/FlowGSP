import unittest

import numpy as np
import flowgsp
from flowgsp.utils import load, op, p_value, np

class TestSurrogate(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, 'bretagne_graph.pkl'))['struct']
        self.graph = flowgsp.graphs.Graph(adj_matrix=self.A)
        self.graph.set_operator("adjacency")
        self.surrogates = flowgsp.surrogates.Surrogate(graph=self.graph)

        self.N = self.graph.N
        self.seed = 42
        self.signal = np.arange(self.graph.N)
        self.nrands = 5
        np.random.seed(self.seed)

    def test_p_value(self):
        null_distrib = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        statistic = 3.5
        result = p_value(null_distrib, statistic)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

    def test_randomizer_phase(self):
        result = self.surrogates.randomizer_phase(self.N)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.N, self.N))

    def test_randomizer_phase_onlysign(self):
        result = self.surrogates.randomizer_phase(self.N, onlysign=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.N, self.N))

    def test_randomizer_phase_conj(self):
        result = self.surrogates.randomizer_phase(self.N, conj=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.N, self.N))

    def test_randomize_direct(self):
        result = self.surrogates.phase_randomize(self.signal)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.signal.shape)

    def test_dir_random_surrogate(self):
        result = self.surrogates.directed_random_surrogate(self.signal, self.nrands)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.nrands, len(self.signal)))

    def test_dir_random_surrogate_normalize(self):
        result = self.surrogates.directed_random_surrogate(self.signal, self.nrands, normalize=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.nrands, len(self.signal)))

    def test_naive_random_surrogate(self):
        result = self.surrogates.naive_random_surrogate(self.signal, self.nrands)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.nrands, len(self.signal)))

    def test_undir_random_surrogate(self):
        result = self.surrogates.undirected_random_surrogate(self.signal, self.nrands)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), self.nrands)
        for surrogate in result:
            self.assertIsInstance(surrogate, np.ndarray)
            self.assertEqual(surrogate.shape, self.signal.shape)

if __name__ == '__main__':
    unittest.main()