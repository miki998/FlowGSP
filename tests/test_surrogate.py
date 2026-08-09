import unittest

import gyraph
from gyraph.utils import load, op, p_value, np


class TestSurrogate(unittest.TestCase):
    """
    Test cases for the Surrogate class in gyraph.surrogates.surrogate module.
    """

    def setUp(self):
        """
        Common setup for tests
        """
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = gyraph.graphs.Graph(adj_matrix=self.A)
        self.graph.set_operator("adjacency")
        self.surrogates = gyraph.surrogates.Surrogate(graph=self.graph)

        self.graph_u = gyraph.graphs.Graph(adj_matrix=self.A + self.A.T)
        self.graph_u.set_operator("adjacency")
        self.surrogates_u = gyraph.surrogates.Surrogate(graph=self.graph_u)

        self.N = self.graph.N
        self.seed = 42
        self.signal = np.arange(self.graph.N)
        self.nrands = 5

    def test_no_set_operator_warning(self):
        """
        Test that ValueError is raised when graph operator is not set.
        """
        graph_no_op = gyraph.graphs.Graph(adj_matrix=self.A)
        with self.assertRaises(ValueError):
            _ = gyraph.surrogates.Surrogate(graph=graph_no_op)

    def test_p_value(self):
        """
        Test the p-value calculation.
        #TODO: This is to be moved to test_utils.py
        """
        null_distrib = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        statistic = 3.5
        result = p_value(null_distrib, statistic)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

    def test_randomizer_phase(self):
        """
        Test the randomizer_phase method.
        """
        # Test default parameters
        result = self.surrogates.randomizer_phase(self.N, self.seed)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.N, self.N))

        # Test with onlysign=True
        result = self.surrogates.randomizer_phase(self.N, self.seed, onlysign=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.N, self.N))

        # Test with conj=True
        result = self.surrogates.randomizer_phase(self.N, self.seed, conj=True)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.N, self.N))

    def test_randomize_direct(self):
        """
        Test the direct phase randomization method.
        """
        result = self.surrogates.phase_randomize(self.signal, seed=self.seed)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.signal.shape)

    def test_dir_random_surrogate(self):
        """
        Test the directed random surrogate generation method.
        """
        # Test the directed random surrogate generation method.
        result = self.surrogates.directed_random_surrogate(
            self.signal, self.nrands, seed=self.seed
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.nrands, len(self.signal)))

        # Test with normalization
        result = self.surrogates.directed_random_surrogate(
            self.signal, self.nrands, seed=self.seed, normalize=True
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.nrands, len(self.signal)))

    def test_naive_random_surrogate(self):
        """
        Test the naive random surrogate generation method.
        """
        result = self.surrogates.naive_random_surrogate(
            self.signal, self.nrands, seed=self.seed
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.nrands, len(self.signal)))

    def test_undir_random_surrogate(self):
        """
        Test the undirected random surrogate generation method.
        """
        result = self.surrogates_u.undirected_random_surrogate(
            self.signal, self.nrands, seed=self.seed
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), self.nrands)
        for surrogate in result:
            self.assertIsInstance(surrogate, np.ndarray)
            self.assertEqual(surrogate.shape, self.signal.shape)


if __name__ == "__main__":
    unittest.main()
