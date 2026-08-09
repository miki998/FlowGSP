import unittest

from gyraph.utils import np
from gyraph.operators import (
    destroy_zero_eigenvals,
    destroy_jordan_blocks,
    destroy_jordan_blocks_laplacian,
)


class TestDestroyZeroEigenvals(unittest.TestCase):
    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"
        self.test_cases_path = "./tests/test_cases/"

    def test_destroy_zero_eigenvals_no_prefer_nodes(self):
        """
        Test destroy_zero_eigenvals without preferred nodes.
        """
        A = np.array([[0, 1], [0, 0]], dtype=float)
        result = destroy_zero_eigenvals(A)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")

    def test_destroy_zero_eigenvals_with_prefer_nodes(self):
        """
        Test destroy_zero_eigenvals with preferred nodes.
        """
        A = np.array([[0, 1], [0, 0]], dtype=float)
        prefer_nodes = [0]
        result = destroy_zero_eigenvals(A, prefer_nodes=prefer_nodes)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")

    # def test_find_best_pair(self):
    #     """
    #     Test the find_best_pair function.
    #     """
    #     raise NotImplementedError("Test for find_best_pair is not implemented yet.")

    def test_destroy_jordan_blocks(self):
        """
        Test the destroy_jordan_blocks function with a simple matrix.
        """
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)

        # Instantiate to quickly check if function runs without error
        result = destroy_jordan_blocks(A)
        result = destroy_jordan_blocks_laplacian(A)
        self.assertIsInstance(result, np.ndarray)

    def test_destroy_zero_eigenvals_large_matrix(self):
        """
        Test destroy_zero_eigenvals on a larger matrix.
        """
        A = [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
        A = np.array(A, dtype=float)
        # A = np.zeros((5, 5), dtype=float)
        # np.fill_diagonal(A, [0, 1, 2, 3, 4])
        result = destroy_zero_eigenvals(A)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")

    def test_destroy_zero_eigenvals_max_iter(self):
        """
        Test destroy_zero_eigenvals with a maximum iteration limit.
        """
        A = np.array([[0, 1], [0, 0]], dtype=float)
        result = destroy_zero_eigenvals(A, max_iter=10)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")


if __name__ == "__main__":
    unittest.main()
