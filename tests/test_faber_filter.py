import unittest

from gyraph.utils import load, op, np

# Import the module to increase coverage
from gyraph.graphs import Graph
from gyraph.filters.faber_filter import (
    FaberFilter,
    _ellipse_params,
    _faber_mats,
    _faber_vdm,
    _qr_solve,
)


class TestFaberHelpers(unittest.TestCase):
    """Test cases for the module-level Faber helpers."""

    def setUp(self):
        # Complex-conjugate-paired spectrum, as produced by a real matrix
        self.lam = np.array([0.5 + 0.3j, -0.2 - 0.3j, 1.0 + 0j, 0.1 + 0.8j, 0.1 - 0.8j])
        self.c, self.a, self.b, self.delta_J = _ellipse_params(self.lam)
        self.cap = (self.a + self.b) / 2.0

    def test_ellipse_params_bounds_spectrum(self):
        """The semi-axes must bound the spectrum coordinate-wise (they are
        fitted to the axis-aligned bounding box), and delta_J must be the
        Joukowski parameter of the semi-axes."""
        self.assertTrue(np.all(np.abs(self.lam.real - self.c.real) <= self.a + 1e-9))
        self.assertTrue(np.all(np.abs(self.lam.imag - self.c.imag) <= self.b + 1e-9))
        self.assertAlmostEqual(self.delta_J, (self.a - self.b) / (self.a + self.b))
        self.assertLess(abs(self.delta_J), 1.0)

    def test_faber_vdm_startup_and_recurrence(self):
        """Columns must satisfy Phi_0 = 1, Phi_1 = mu, Phi_2 = mu^2 - 2 delta_J."""
        V = _faber_vdm(self.lam, self.c, self.cap, self.delta_J, 4)
        mu = (self.lam - self.c) / self.cap
        self.assertTrue(np.allclose(V[:, 0], 1.0))
        self.assertTrue(np.allclose(V[:, 1], mu))
        self.assertTrue(np.allclose(V[:, 2], mu**2 - 2.0 * self.delta_J))
        self.assertTrue(np.allclose(V[:, 3], mu * V[:, 2] - self.delta_J * V[:, 1]))
        # Degenerate degrees
        self.assertEqual(
            _faber_vdm(self.lam, self.c, self.cap, self.delta_J, 0).shape, (5, 0)
        )
        V1 = _faber_vdm(self.lam, self.c, self.cap, self.delta_J, 1)
        self.assertTrue(np.allclose(V1[:, 0], 1.0))

    def test_faber_mats_match_vdm_on_diagonal_matrix(self):
        """On a diagonal matrix, diag(F_k(M)) must equal the Vandermonde column."""
        mats = _faber_mats(np.diag(self.lam), self.c, self.cap, self.delta_J, 5)
        V = _faber_vdm(self.lam, self.c, self.cap, self.delta_J, 5)
        self.assertEqual(len(mats), 5)
        for k in range(5):
            self.assertTrue(
                np.allclose(np.diag(mats[k]), V[:, k], atol=1e-12), msg=f"k={k}"
            )
        # Degenerate degrees
        for deg in range(4):
            self.assertEqual(
                len(
                    _faber_mats(np.diag(self.lam), self.c, self.cap, self.delta_J, deg)
                ),
                deg,
            )

    def test_qr_solve_exact_system(self):
        """On a well-posed least-squares problem, _qr_solve must recover the
        exact coefficients."""
        rng = np.random.default_rng(0)
        V = rng.standard_normal((10, 3))
        c_true = np.array([1.0, -2.0, 0.5])
        c_est = _qr_solve(V, V @ c_true)
        self.assertTrue(np.allclose(c_est, c_true, atol=1e-10))

    def test_qr_solve_rank_deficient(self):
        """Rank-deficient columns must be truncated instead of blowing up."""
        V = np.ones((6, 3))
        V[:, 1] = np.arange(6)
        V[:, 2] = V[:, 1]  # duplicate column -> rank 2
        c_est = _qr_solve(V, V[:, 0] + V[:, 1])
        self.assertTrue(np.all(np.isfinite(c_est)))


class TestFaberFilter(unittest.TestCase):
    """
    Test cases for gyraph.filters.faber_filter.FaberFilter class.
    """

    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("advection_diffusion")

        self.order = 4
        self.faber = FaberFilter(graph=self.graph, order=self.order)
        self.signal = np.arange(self.graph.N, dtype=float)

    def test_precompute_polynomial(self):
        """Faber basis must have `order` matrices, aliased to powers_of_M,
        with F_0 = I."""
        self.assertEqual(len(self.faber.faber_matrices), self.order)
        self.assertIs(self.faber.powers_of_M, self.faber.faber_matrices)
        self.assertTrue(np.allclose(self.faber.faber_matrices[0], np.eye(self.graph.N)))

    def test_identity_kernel_gives_identity_filter(self):
        """The constant kernel h(lambda) = 1 must produce the identity filter."""
        filt = self.faber.polynomial_filter(np.ones(self.graph.N))
        self.assertTrue(np.allclose(filt, np.eye(self.graph.N), atol=1e-8))

    def test_linear_kernel_reproduces_operator(self):
        """The kernel h(lambda) = lambda must reproduce the operator M
        (exactly representable in the degree-2 Faber basis)."""
        filt, coefs = self.faber.polynomial_filter(
            self.graph.operator.V, return_coefs=True
        )
        self.assertTrue(np.allclose(filt, self.graph.operator.M, atol=1e-8))
        self.assertEqual(len(coefs), self.order)

    def test_apply_method(self):
        """apply with the identity kernel must be a no-op on the signal."""
        filtered = self.faber.apply(self.signal, np.ones(self.graph.N))
        self.assertTrue(np.allclose(filtered, self.signal, atol=1e-8))

    def test_vandermonde_matrix(self):
        """Vandermonde entries must be bounded (Faber property: |Phi_k| <= 2)."""
        vdm = self.faber.vandermonde_matrix(self.graph.operator.V, self.order)
        self.assertEqual(vdm.shape, (self.graph.N, self.order))
        self.assertTrue(np.all(np.abs(vdm) <= 2.0 + 1e-9))

    def test_regression_descent(self):
        """Adam descent must run and return the right shapes and finite loss."""
        recon, coefs, loss = self.faber.regression_descent(
            self.signal, self.signal, n_iter=20
        )
        self.assertEqual(recon.shape, self.signal.shape)
        self.assertEqual(coefs.shape, (self.order,))
        self.assertTrue(np.isfinite(loss))

    def test_repr(self):
        rep = repr(self.faber)
        self.assertIn("FaberFilter", rep)
        self.assertIn("delta_J", rep)


if __name__ == "__main__":
    unittest.main()
