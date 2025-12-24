import unittest

from flowgsp.utils import np

# Import the module to increase coverage
from flowgsp.utils import (
    p_value,
    dirichlet,
    TV,
    sobolev,
    directed_variation,
)


class TestStats(unittest.TestCase):
    """
    Test cases for flowgsp.utils.stats_utils functions.
    """

    def setUp(self):
        """Set up test fixtures for stats_utils tests."""
        self.test_graphs_path = "./tests/test_graphs/"
        self.null_distrib = np.random.rand(1000)
        self.statistic = 0.5

    def test_p_value(self):
        """
        Test the p_value function.
        #TODO: expand test cases
        """
        # Test cases for p_value function
        self.assertIsInstance(p_value(self.null_distrib, self.statistic), float)
        self.assertIsInstance(
            p_value(self.null_distrib, self.statistic, two_tail=True), float
        )

class TestMetrics(unittest.TestCase):
    """
    Test cases for flowgsp.utils.metrics functions.
    """

    def setUp(self):
        """Set up test fixtures for metrics tests."""
        self.signal = np.random.rand(10)
        self.A = np.random.rand(10, 10)
        self.L = np.random.rand(10, 10)
        self.L = (self.L + self.L.T) / 2  # Make it symmetric

    def test_dirichlet(self):
        """
        Test the dirichlet function.
        """
        smoothness = dirichlet(self.signal, self.L, normalize=True)
        self.assertIsInstance(smoothness, float)

    def test_TV(self):
        """
        Test the TV function.
        """
        smoothness_L1 = TV(self.signal, self.A, norm="L1", normalize=True)
        smoothness_L2 = TV(self.signal, self.A, norm="L2", normalize=True)
        self.assertIsInstance(smoothness_L1, float)
        self.assertIsInstance(smoothness_L2, float)

    def test_sobolev(self):
        """
        Test the sobolev function.
        """
        smoothness = sobolev(self.signal, self.L, norm="L2", normalize=True)
        self.assertIsInstance(smoothness, float)

    def test_directed_variation(self):
        """
        Test the directed_variation function.
        """
        smoothness = directed_variation(self.signal, self.L, normalize=True)
        self.assertIsInstance(smoothness, float)


if __name__ == "__main__":
    unittest.main()
