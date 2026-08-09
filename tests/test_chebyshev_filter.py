import unittest

from gyraph.utils import load, op, np

# Import the module to increase coverage
from gyraph.graphs import Graph
from gyraph.filters import ChebyshevFilter
from gyraph.filters.chebyshev_filter import (
    DualChebyshevFilter,
    _cheb_mats,
    _cheb_scaling,
    _cheb_vdm,
)


class TestChebyshevHelpers(unittest.TestCase):
    """Test cases for the module-level Chebyshev helpers."""

    def setUp(self):
        self.lam = np.array([0.5 + 0.3j, -0.2 - 0.3j, 1.0 + 0j, 0.1 + 0.8j, 0.1 - 0.8j])

    def test_cheb_scaling_real_spectrum(self):
        """For a real spectrum the scaling is the standard affine map."""
        eigs = np.array([-1.0, 0.0, 3.0])
        c, R = _cheb_scaling(eigs.astype(complex))
        self.assertAlmostEqual(c.real, 1.0)
        self.assertAlmostEqual(c.imag, 0.0)
        self.assertAlmostEqual(R, 2.0, places=10)

    def test_cheb_scaling_contains_spectrum(self):
        """All scaled eigenvalues must lie in the unit disk."""
        c, R = _cheb_scaling(self.lam)
        self.assertTrue(np.all(np.abs((self.lam - c) / R) <= 1.0 + 1e-12))

    def test_cheb_vdm_recurrence(self):
        """Vandermonde columns must satisfy T_0 = 1, T_1 = x, T_2 = 2x^2 - 1."""
        x = np.linspace(-1, 1, 7)
        C = _cheb_vdm(x, 3)
        self.assertTrue(np.allclose(C[:, 0], 1.0))
        self.assertTrue(np.allclose(C[:, 1], x))
        self.assertTrue(np.allclose(C[:, 2], 2 * x**2 - 1))
        # Degenerate degrees
        self.assertEqual(_cheb_vdm(x, 0).shape, (7, 0))
        self.assertTrue(np.allclose(_cheb_vdm(x, 1)[:, 0], 1.0))

    def test_cheb_mats_match_vdm_on_diagonal_matrix(self):
        """On a diagonal matrix, diag(T_k(M)) must equal the Vandermonde column."""
        c, R = _cheb_scaling(self.lam)
        mats = _cheb_mats(np.diag(self.lam), c, R, 4)
        V = _cheb_vdm((self.lam - c) / R, 4)
        self.assertEqual(len(mats), 4)
        for k in range(4):
            self.assertTrue(np.allclose(np.diag(mats[k]), V[:, k], atol=1e-12))
        # Degenerate degrees
        self.assertEqual(_cheb_mats(np.diag(self.lam), c, R, 0), [])
        self.assertEqual(len(_cheb_mats(np.diag(self.lam), c, R, 1)), 1)


class TestChebyshevFilter(unittest.TestCase):
    """
    Test cases for gyraph.filters.ChebyshevFilter class.
    """

    def setUp(self):
        """Set up test fixtures for Chebyshev filter tests."""
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("advection_diffusion")

        self.order = 4
        self.chebyshev = ChebyshevFilter(graph=self.graph, order=self.order)
        self.signal = np.arange(self.graph.N, dtype=float)

    def test_precompute_polynomial(self):
        """Chebyshev matrix basis must have `order` elements, aliased to powers_of_M."""
        self.assertEqual(len(self.chebyshev.chebyshev_matrices), self.order)
        self.assertIs(self.chebyshev.powers_of_M, self.chebyshev.chebyshev_matrices)
        # T_0 is the identity
        N = self.graph.N
        self.assertTrue(np.allclose(self.chebyshev.chebyshev_matrices[0], np.eye(N)))

    def test_identity_kernel_gives_identity_filter(self):
        """The constant kernel h(lambda) = 1 must produce the identity filter
        (exactly representable as 1 * T_0)."""
        filt = self.chebyshev.polynomial_filter(np.ones(self.graph.N))
        self.assertTrue(np.allclose(filt, np.eye(self.graph.N), atol=1e-8))

    def test_linear_kernel_reproduces_operator(self):
        """The kernel h(lambda) = lambda must reproduce M itself
        (exactly representable as c*T_0 + R*T_1)."""
        filt, coefs = self.chebyshev.polynomial_filter(
            self.graph.operator.V, return_coefs=True
        )
        self.assertTrue(np.allclose(filt, self.graph.operator.M, atol=1e-8))
        self.assertEqual(len(coefs), self.order)

    def test_apply_method(self):
        """apply must equal (polynomial filter) @ signal; identity kernel is a no-op."""
        filtered = self.chebyshev.apply(self.signal, np.ones(self.graph.N))
        self.assertTrue(np.allclose(filtered, self.signal, atol=1e-8))

        filtered, coefs = self.chebyshev.apply(
            self.signal, np.ones(self.graph.N), return_coefs=True
        )
        self.assertEqual(filtered.shape, self.signal.shape)
        self.assertEqual(len(coefs), self.order)

    def test_vandermonde_matrix(self):
        """Vandermonde must have bounded (Chebyshev) entries and correct shape."""
        vdm = self.chebyshev.vandermonde_matrix(self.graph.operator.V, self.order)
        self.assertEqual(vdm.shape, (self.graph.N, self.order))
        self.assertTrue(np.allclose(vdm[:, 0], 1.0))

    def test_regression_descent(self):
        """Adam descent must run, return the right shapes and a finite loss."""
        recon, coefs, loss = self.chebyshev.regression_descent(
            self.signal, self.signal, n_iter=20
        )
        self.assertEqual(recon.shape, self.signal.shape)
        self.assertEqual(coefs.shape, (self.order,))
        self.assertTrue(np.isfinite(loss))

    def test_repr(self):
        self.assertIn("ChebyshevFilter", repr(self.chebyshev))


class TestDualChebyshevFilter(unittest.TestCase):
    """
    Test cases for gyraph.filters.chebyshev_filter.DualChebyshevFilter.
    """

    FILTER_TYPES = ["GAGD", "GQAD", "GQDA", "GA", "GD"]
    MATRIX_ATTRS = {
        "GD": ("powers_of_D",),
        "GA": ("powers_of_A",),
        "GAGD": ("powers_of_P", "powers_of_Q"),
        "GQAD": ("powers_of_R",),
        "GQDA": ("powers_of_R",),
    }

    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("advection_diffusion")

        self.order = 3
        self.signal = np.arange(self.graph.N, dtype=float)
        self.filters = {
            ft: DualChebyshevFilter(graph=self.graph, order=self.order, filter_type=ft)
            for ft in self.FILTER_TYPES
        }

    def test_invalid_filter_type_raises(self):
        with self.assertRaises(ValueError):
            DualChebyshevFilter(
                graph=self.graph, order=self.order, filter_type="gibberish"
            )

    def test_precompute_polynomial(self):
        """Each filter type must populate its Chebyshev matrix list(s)."""
        for ft, filt in self.filters.items():
            for attr in self.MATRIX_ATTRS[ft]:
                mats = getattr(filt, attr)
                self.assertEqual(len(mats), self.order, msg=f"{ft}.{attr}")
                # T_0 is always the identity
                self.assertTrue(
                    np.allclose(mats[0], np.eye(self.graph.N)), msg=f"{ft}.{attr}[0]"
                )

    def test_vandermonde_matrix_compose(self):
        """Composed Vandermonde: (N, 2*order) for GAGD, (N, order) otherwise,
        with real entries bounded by 1 in magnitude (Chebyshev on [-1, 1])."""
        for ft, filt in self.filters.items():
            vdm = filt.vandermonde_matrix_compose(self.order)
            ncols = 2 * self.order if ft == "GAGD" else self.order
            self.assertEqual(vdm.shape, (self.graph.N, ncols), msg=ft)
            self.assertTrue(np.all(np.abs(vdm) <= 1.0 + 1e-9), msg=ft)

    def test_identity_kernel_gives_identity_filter(self):
        """The constant kernel h = 1 must produce the identity filter for
        every filter type (exactly representable as 1 * T_0)."""
        N = self.graph.N
        for ft, filt in self.filters.items():
            gf = filt.polynomial_filter(np.ones(N))
            self.assertTrue(np.allclose(gf, np.eye(N), atol=1e-8), msg=ft)

    def test_gd_linear_kernel_reproduces_diffusion_part(self):
        """GD with kernel h = Re(lambda) must reproduce the diffusion
        operator P (exactly representable as c*T_0 + R*T_1)."""
        gf, coefs = self.filters["GD"].polynomial_filter(
            self.graph.operator.V.real, return_coefs=True
        )
        self.assertTrue(np.allclose(gf, self.graph.operator.P, atol=1e-8))
        self.assertEqual(len(coefs), self.order)

    def test_ga_linear_kernel_reproduces_advection_part(self):
        """GA with kernel h = Im(lambda) must reproduce -i*Q (the -i
        substitution basis makes it exactly representable as R_I * T_1)."""
        gf = self.filters["GA"].polynomial_filter(self.graph.operator.V.imag)
        self.assertTrue(np.allclose(gf, -1j * self.graph.operator.Q, atol=1e-8))

    def test_gagd_linear_kernel_reproduces_diffusion_part(self):
        """GAGD with kernel h = Re(lambda) must also assemble to P, even when
        the coefficient solve spreads mass across both bases."""
        gf = self.filters["GAGD"].polynomial_filter(self.graph.operator.V.real)
        self.assertTrue(np.allclose(gf, self.graph.operator.P, atol=1e-8))

    def test_apply_method(self):
        """apply must equal (polynomial filter) @ signal for every type;
        the identity kernel is a no-op."""
        ones = np.ones(self.graph.N)
        for ft, filt in self.filters.items():
            filtered = filt.apply(self.signal, ones)
            self.assertTrue(np.allclose(filtered, self.signal, atol=1e-8), msg=ft)

            filtered, coefs = filt.apply(self.signal, ones, return_coefs=True)
            ncoef = 2 * self.order if ft == "GAGD" else self.order
            self.assertEqual(filtered.shape, self.signal.shape, msg=ft)
            self.assertEqual(len(coefs), ncoef, msg=ft)

    def test_regression_descent(self):
        """Descent must run for every filter type with correct output shapes."""
        for ft, filt in self.filters.items():
            recon, coefs, loss = filt.regression_descent(
                self.signal, 0.5 * self.signal, n_iter=10
            )
            ncoef = 2 * self.order if ft == "GAGD" else self.order
            self.assertEqual(recon.shape, self.signal.shape, msg=ft)
            self.assertEqual(coefs.shape, (ncoef,), msg=ft)
            self.assertTrue(np.isfinite(loss), msg=ft)

    def test_repr(self):
        self.assertIn("DualChebyshevFilter", repr(self.filters["GD"]))


if __name__ == "__main__":
    unittest.main()
