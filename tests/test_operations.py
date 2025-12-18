import unittest

import flowgsp
from flowgsp.utils import load, op, np

class TestOperations(unittest.TestCase):

    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"
        self.test_cases_path = "./tests/test_cases/"

        self.A1 = load(op.join(self.test_graphs_path, 'cycle.pkl'))
        self.A2 = load(op.join(self.test_graphs_path, 'usa_graph_diagonal.pkl'))
        self.L1 = load(op.join(self.test_graphs_path, 'cycle_laplacian.pkl'))
        self.L2 = load(op.join(self.test_graphs_path, 'usa_graph_diagonal_laplacian.pkl'))

        self.U, self.V, self.Uinv = load(op.join(self.test_graphs_path, 'usa_graph_basis.pkl'))


    # TODO: Replace with flowgsp initialization of operator pipeline
    # def test_normalize_adjacency_left(self):
    #     eA1, eA2, eL1, eL2 = load(op.join(self.test_cases_path, 'TC0.pkl'))
    #     rA1 = normalize_adjacency(self.A1, "left")
    #     rA2 = normalize_adjacency(self.A2, "left")
    #     rL1 = normalize_adjacency(self.L1, "left")
    #     rL2 = normalize_adjacency(self.L2, "left")

    #     np.testing.assert_array_almost_equal(eA1, rA1, decimal=6)
    #     np.testing.assert_array_almost_equal(eA2, rA2, decimal=6)
    #     np.testing.assert_array_almost_equal(eL1, rL1, decimal=6)
    #     np.testing.assert_array_almost_equal(eL2, rL2, decimal=6)

    #     complex = np.array([[1j, -1j], [-1j, 1j]], dtype=np.complex64)
    #     np.testing.assert_raises(ValueError, normalize_adjacency, complex, "left")

    # def test_normalize_adjacency_right(self):
    #     eA1, eA2, eL1, eL2 = load(op.join(self.test_cases_path, 'TC1.pkl'))
    #     rA1 = normalize_adjacency(self.A1, "right")
    #     rA2 = normalize_adjacency(self.A2, "right")
    #     rL1 = normalize_adjacency(self.L1, "right")
    #     rL2 = normalize_adjacency(self.L2, "right")

    #     np.testing.assert_array_almost_equal(eA1, rA1, decimal=6)
    #     np.testing.assert_array_almost_equal(eA2, rA2, decimal=6)
    #     np.testing.assert_array_almost_equal(eL1, rL1, decimal=6)
    #     np.testing.assert_array_almost_equal(eL2, rL2, decimal=6)

    #     complex = np.array([[1j, -1j], [-1j, 1j]], dtype=np.complex64)
    #     np.testing.assert_raises(ValueError, normalize_adjacency, complex, "right")

    # def test_normalize_adjacency_symmetric(self):
    #     eA1, eA2, eL1, eL2 = load(op.join(self.test_cases_path, 'TC2.pkl'))
    #     rA1 = normalize_adjacency(self.A1, "symmetric")
    #     rA2 = normalize_adjacency(self.A2, "symmetric")
    #     rL1 = normalize_adjacency(self.L1, "symmetric")
    #     rL2 = normalize_adjacency(self.L2, "symmetric")

    #     np.testing.assert_array_almost_equal(eA1, rA1, decimal=6)
    #     np.testing.assert_array_almost_equal(eA2, rA2, decimal=6)
    #     np.testing.assert_array_almost_equal(eL1, rL1, decimal=6)
    #     np.testing.assert_array_almost_equal(eL2, rL2, decimal=6)

    #     complex = np.array([[1j, -1j], [-1j, 1j]], dtype=np.complex64)
    #     np.testing.assert_raises(ValueError, normalize_adjacency, complex, "symmetric")

    # def test_laplacian_to_adj(self):
    #     eL1, eL2 = load(op.join(self.test_cases_path, 'TC3.pkl'))
    #     rL1 = laplacian_to_adj(self.L1)
    #     rL2 = laplacian_to_adj(self.L2)

    #     np.testing.assert_array_almost_equal(eL1, rL1, decimal=6)
    #     np.testing.assert_array_almost_equal(eL2, rL2, decimal=6)

    #     np.testing.assert_raises(ValueError, laplacian_to_adj, self.A1)
    #     np.testing.assert_raises(ValueError, laplacian_to_adj, self.A2)

    # def test_compute_directed_laplacian_in_degree(self):
    #     eA1, eA2 = load(op.join(self.test_cases_path, 'TC4.pkl'))
    #     rA1 = compute_directed_laplacian(self.A1, in_degree=True)
    #     rA2 = compute_directed_laplacian(self.A2, in_degree=True)

    #     np.testing.assert_array_almost_equal(eA1, rA1, decimal=6)
    #     np.testing.assert_array_almost_equal(eA2, rA2, decimal=6)

    #     np.testing.assert_raises(ValueError, compute_directed_laplacian, self.L1)
    #     np.testing.assert_raises(ValueError, compute_directed_laplacian, self.L2)

    # def test_compute_directed_laplacian_out_degree(self):
    #     eA1, eA2 = load(op.join(self.test_cases_path, 'TC5.pkl'))
    #     rA1 = compute_directed_laplacian(self.A1, in_degree=False)
    #     rA2 = compute_directed_laplacian(self.A2, in_degree=False)

    #     np.testing.assert_array_almost_equal(eA1, rA1, decimal=6)
    #     np.testing.assert_array_almost_equal(eA2, rA2, decimal=6)

    #     np.testing.assert_raises(ValueError, compute_directed_laplacian, self.L1)
    #     np.testing.assert_raises(ValueError, compute_directed_laplacian, self.L2)

    # def test_compute_basis_eig(self):
    #     # NOTE: For now - only checking the eigenvalues
    #     (eUA1, eVA1), (eUA2, eVA2), (eUL1, eVL1), (eUL2, eVL2) = load(op.join(self.test_cases_path, 'TC6.pkl'))
    #     self.A3 = self.A2 + hermitian(self.A2)
    #     self.L3 = self.L2 + hermitian(self.L2)

    #     rUA1, rVA1 = compute_basis(self.A1, method="eig", gso_order='adj')
    #     rUA2, rVA2 = compute_basis(self.A2, method="eig", gso_order='adj')
    #     rUA3, rVA3 = compute_basis(self.A3, method="eig", gso_order='adj')
    #     rUL1, rVL1 = compute_basis(self.L1, method="eig", gso_order='laplacian')
    #     rUL2, rVL2 = compute_basis(self.L2, method="eig", gso_order='laplacian')
    #     rUL3, rVL3 = compute_basis(self.L3, method="eig", gso_order='laplacian')

        
    #     # np.testing.assert_array_almost_equal(rUA1, eUA1, decimal=6)
    #     # np.testing.assert_array_almost_equal(rUA2, eUA2, decimal=6)
    #     # np.testing.assert_array_almost_equal(rUL1, eUL1, decimal=6)
    #     # np.testing.assert_array_almost_equal(rUL2, eUL2, decimal=6)

    #     np.testing.assert_array_almost_equal(np.abs(rVA1), np.abs(eVA1), decimal=6)
    #     np.testing.assert_array_almost_equal(np.abs(rVA2), np.abs(eVA2), decimal=6)
    #     np.testing.assert_array_almost_equal(np.abs(rVL1), np.abs(eVL1), decimal=6)
    #     np.testing.assert_array_almost_equal(np.abs(rVL2), np.abs(eVL2), decimal=6)

    #     # Assert Realness of eigenvectors of hermitian matrices
    #     np.testing.assert_array_almost_equal(rUA3.imag, np.zeros_like(rUA3), decimal=6)
    #     np.testing.assert_array_almost_equal(rUL3.imag, np.zeros_like(rUL3), decimal=6)
        
    #     # NOTE: Laxist test of angle
    #     np.testing.assert_array_almost_equal(np.abs(np.angle(rVA1)) % np.pi, np.abs(np.angle(eVA1)) % np.pi, decimal=6)
    #     np.testing.assert_array_almost_equal(np.abs(np.angle(rVA2)) % np.pi, np.abs(np.angle(eVA2)) % np.pi, decimal=6)
    #     np.testing.assert_array_almost_equal(np.abs(np.angle(rVL1)) % np.pi, np.abs(np.angle(eVL1)) % np.pi, decimal=6)
    #     np.testing.assert_array_almost_equal(np.abs(np.angle(rVL2)) % np.pi, np.abs(np.angle(eVL2)) % np.pi, decimal=6)

    # def test_polar_decomposition(self):
    #     eQ, eF, eP = load(op.join(self.test_cases_path, 'TC7.pkl'))
    #     rQ, rF, rP = polar_decomposition(self.A1)

    #     np.testing.assert_array_almost_equal(eQ, rQ, decimal=6)
    #     np.testing.assert_array_almost_equal(eF, rF, decimal=6)
    #     np.testing.assert_array_almost_equal(eP, rP, decimal=6)

    # def test_hermitian(self):
    #     A = np.array([[1, 1j], [-1j, 1]], dtype=complex)
    #     result = hermitian(A)
    #     expected = np.array([[1, 1j], [-1j, 1]], dtype=complex).T.conj()
    #     np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # def test_conjugate_frequency(self):
    #     V = np.array([[1-1j, 0], 
    #                   [0, 1+1j]], dtype=complex)
    #     result = conjugate_frequency(0, np.diag(V))
    #     expected = 1
    #     self.assertEqual(result, expected)

    #     np.testing.assert_raises(ValueError, eigvalues_pairs, V)

    # def test_eigvalues_pairs(self):
    #     V = np.array([[1-1j, 0, 0], 
    #                   [0, 1+1j, 0],
    #                   [0, 0, -.5]], dtype=complex)
    #     result = eigvalues_pairs(np.diag(V))
    #     expected = [[0, 1], [2]]
    #     for res, exp in zip(result, expected):
    #         np.testing.assert_array_almost_equal(res, exp)

    #     np.testing.assert_raises(ValueError, eigvalues_pairs, V)

    # def test_GFT(self):
    #     expected = load(op.join(self.test_cases_path, 'TC8.pkl'))
    #     signal = np.ones(len(self.A2))
    #     result = GFT(signal, self.U, self.Uinv)

    #     np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # def test_inverseGFT(self):
    #     expected = load(op.join(self.test_cases_path, 'TC9.pkl'))
    #     result = inverseGFT(np.ones(len(self.A2)), self.U)

    #     np.testing.assert_array_almost_equal(result, expected, decimal=6)

if __name__ == '__main__':
    unittest.main()