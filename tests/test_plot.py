import unittest

# Import the module to increase coverage
from flowgsp.utils import unique_color_generator


class TestPlot(unittest.TestCase):
    """
    Test cases for flowgsp.filters.ChebyshevFilter class.
    """

    def setUp(self):
        """Set up test fixtures for Chebyshev filter tests."""
        self.test_graphs_path = "./tests/test_graphs/"

    def test_unique_color_generator(self):
        """
        Test the unique_color_generator function.
        """
        color_gen = unique_color_generator()
        colors = [next(color_gen) for _ in range(10)]
        self.assertEqual(len(colors), 10)
        self.assertEqual(len(set(colors)), 10)  # Ensure all colors are unique


if __name__ == "__main__":
    unittest.main()
