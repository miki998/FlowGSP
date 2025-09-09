import unittest

from flowgsp.utils import np
from flowgsp.operators import destroy_zero_eigenvals

class TestDestroyZeroEigenvals(unittest.TestCase):

    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"
        self.test_cases_path = "./tests/test_cases/"

    def test_destroy_zero_eigenvals_no_prefer_nodes(self):
        A = np.array([[0, 1], [0, 0]], dtype=float)
        result = destroy_zero_eigenvals(A)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")

    def test_destroy_zero_eigenvals_with_prefer_nodes(self):
        A = np.array([[0, 1], [0, 0]], dtype=float)
        prefer_nodes = [0]
        result = destroy_zero_eigenvals(A, prefer_nodes=prefer_nodes)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")

    def test_destroy_zero_eigenvals_large_matrix(self):
        A = [[0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0]]
        A = np.array(A, dtype=float)
        # A = np.zeros((5, 5), dtype=float)
        # np.fill_diagonal(A, [0, 1, 2, 3, 4])
        result = destroy_zero_eigenvals(A)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")

    def test_destroy_zero_eigenvals_verbose(self):
        A = np.array([[0, 1], [0, 0]], dtype=float)
        result = destroy_zero_eigenvals(A, verbose=True)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")

    def test_destroy_zero_eigenvals_max_iter(self):
        A = np.array([[0, 1], [0, 0]], dtype=float)
        result = destroy_zero_eigenvals(A, max_iter=10)
        D, _ = np.linalg.eig(result)
        self.assertTrue(np.all(np.abs(D) > 1e-6), "Zero eigenvalues were not destroyed")

if __name__ == '__main__':
    unittest.main()