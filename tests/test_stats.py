import unittest

import numpy as np
from gyraph.stats.stats_utils import (
    p_value,
    circular_stats,
    circular_correlation,
    sample_circular_complex_gaussian,
)


class TestStats(unittest.TestCase):
    def setUp(self):
        self.nb_samples = 1000

    def test_sample_circular_complex_gaussian(self):
        """Test sampling from circular complex Gaussian distribution."""
        samples = sample_circular_complex_gaussian(
            np.zeros(10), np.eye(10), n_samples=self.nb_samples
        )
        self.assertEqual(samples.shape[1], self.nb_samples)

    def test_circular_stats(self):
        """Test circular statistics computation."""
        samples = sample_circular_complex_gaussian(
            np.zeros(10), np.eye(10), n_samples=self.nb_samples
        )
        angles = np.rad2deg(np.angle(samples))  # Get the angles of the complex samples
        mean, var = circular_stats(angles)
        self.assertIsInstance(mean, float)
        self.assertIsInstance(var, float)

    def test_circular_correlation(self):
        """Test circular correlation computation."""
        samples1 = sample_circular_complex_gaussian(
            np.zeros(10), np.eye(10), n_samples=self.nb_samples
        )
        samples2 = sample_circular_complex_gaussian(
            np.zeros(10), np.eye(10), n_samples=self.nb_samples
        )

        angles1 = np.rad2deg(np.angle(samples1))  # Get the angles of the first samples
        angles2 = np.rad2deg(np.angle(samples2))  # Get the angles of the second samples
        corr = circular_correlation(angles1, angles2)
        self.assertIsInstance(corr, float)

    def test_p_value(self):
        """Test p-value computation."""
        np.random.seed(99)
        p_val = p_value(np.random.randn(100), 0.1)
        self.assertIsInstance(p_val, float)
        p_val = p_value(np.random.randn(100), 0.1, two_tail=True)
        self.assertIsInstance(p_val, float)


if __name__ == "__main__":
    unittest.main()
