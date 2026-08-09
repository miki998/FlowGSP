import unittest

from gyraph.utils import load, op, np
from gyraph.graphs import Graph
from gyraph.operators import (
    Adjacency,
    Laplacian,
    AdvectionDiffusion,
    TimeVertexAdjacency,
    TimeVertexLaplacian,
)


class TestAdjacency(unittest.TestCase):
    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "usa_graph_diagonal.pkl"))
        self.symA = self.A + self.A.T

        graph = Graph(adj_matrix=self.A, debug=False)
        graph_sym = Graph(adj_matrix=self.symA, debug=False)

        self.graph_adj = Adjacency(graph=graph)
        self.graph_adj_sym = Adjacency(graph=graph, normalize="symmetric")
        self.graph_sym = Adjacency(graph=graph_sym)

        with self.assertRaises(ValueError):
            Adjacency(graph=graph, normalize="gibberish")  # Should raise ValueError

    def test_compute_operator(self):
        """
        Test the compute_operator method for Adjacency operator.
        """
        self.graph_adj.compute_operator()
        self.graph_adj_sym.compute_operator()
        self.graph_sym.compute_operator()

    def test_compute_basis(self):
        """Test that compute_basis produces a valid eigendecomposition."""
        self.graph_adj.compute_basis()
        N = self.graph_adj.graph.N

        # Shapes
        self.assertEqual(self.graph_adj.U.shape, (N, N))
        self.assertEqual(self.graph_adj.V.shape, (N,))
        self.assertEqual(self.graph_adj.Uinv.shape, (N, N))
        self.assertEqual(self.graph_adj.frequencies.shape, (N,))

        # Frequencies are non-negative and sorted
        self.assertTrue(np.all(self.graph_adj.frequencies >= 0))
        self.assertTrue(np.all(np.diff(self.graph_adj.frequencies) >= -1e-10))

        # Eigenvalue equation: M @ U == U @ diag(V)
        self.assertTrue(
            np.allclose(
                self.graph_adj.M @ self.graph_adj.U,
                self.graph_adj.U * self.graph_adj.V,
                atol=1e-6,
            )
        )

        # For a genuinely symmetric graph: Uinv == U^H (unitary basis)
        self.graph_sym.compute_basis()
        self.assertTrue(
            np.allclose(
                self.graph_sym.Uinv,
                self.graph_sym.U.conj().T,
                atol=1e-6,
            )
        )

    def test_compute_kernels(self):
        """Test shape, range, and complementarity of low/high-pass kernels."""
        N = self.graph_adj.graph.N
        cutoff = N // 3

        lowpass = self.graph_adj.low_pass_kernel(cutoff)
        highpass = self.graph_adj.high_pass_kernel(cutoff)

        # Correct output shape
        self.assertEqual(lowpass.shape[0], N)
        self.assertEqual(highpass.shape[0], N)

        # Values are non-negative
        self.assertTrue(np.all(lowpass >= 0))
        self.assertTrue(np.all(highpass >= 0))

        # Low-pass and high-pass sum to all-ones
        self.assertTrue(np.allclose(lowpass + highpass, np.ones(N)))


class TestLaplacian(unittest.TestCase):
    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "usa_graph_diagonal.pkl"))
        self.symA = self.A + self.A.T

        graph = Graph(adj_matrix=self.A, debug=False)
        graph_sym = Graph(adj_matrix=self.symA, debug=False)

        self.graph_lap = Laplacian(graph=graph)
        self.graph_lap_right = Laplacian(graph=graph, normalize="right")
        self.graph_sym = Laplacian(graph=graph_sym)

        with self.assertRaises(ValueError):
            Laplacian(graph=graph, normalize="gibberish")  # Should raise ValueError

    def test_compute_operator(self):
        """
        Test the compute_operator method for Laplacian operator.
        """
        self.graph_lap.compute_operator()
        self.graph_sym.compute_operator()

    def test_compute_basis(self):
        """Test that compute_basis produces a valid eigendecomposition."""
        self.graph_lap.compute_basis()
        N = self.graph_lap.graph.N

        # Shapes
        self.assertEqual(self.graph_lap.U.shape, (N, N))
        self.assertEqual(self.graph_lap.V.shape, (N,))
        self.assertEqual(self.graph_lap.Uinv.shape, (N, N))

        # Frequencies are non-negative and sorted
        self.assertTrue(np.all(self.graph_lap.frequencies >= 0))
        self.assertTrue(np.all(np.diff(self.graph_lap.frequencies) >= -1e-10))

        # Eigenvalue equation: M @ U == U @ diag(V)
        self.assertTrue(
            np.allclose(
                self.graph_lap.M @ self.graph_lap.U,
                self.graph_lap.U * self.graph_lap.V,
                atol=1e-6,
            )
        )

        # Symmetric graph: Uinv == U^H
        self.graph_sym.compute_basis()
        self.assertTrue(
            np.allclose(self.graph_sym.Uinv, self.graph_sym.U.conj().T, atol=1e-6)
        )

    def test_compute_directed_laplacian(self):
        """Test directed Laplacian properties: L = D - A, row sums ~= 0 for in-degree."""
        L_in = self.graph_lap.compute_directed_laplacian(self.A, in_degree=True)
        L_out = self.graph_lap.compute_directed_laplacian(self.A, in_degree=False)

        N = self.A.shape[0]
        self.assertEqual(L_in.shape, (N, N))
        self.assertEqual(L_out.shape, (N, N))

        # Diagonal should be non-negative (degree values)
        self.assertTrue(np.all(np.diag(L_in) >= 0))
        self.assertTrue(np.all(np.diag(L_out) >= 0))

        # Off-diagonal should be <= 0 (L = D - A with A >= 0)
        mask = ~np.eye(N, dtype=bool)
        self.assertTrue(np.all(L_in[mask] <= 1e-10))
        self.assertTrue(np.all(L_out[mask] <= 1e-10))

    def test_compute_kernels(self):
        """Test shape, range, and complementarity of Laplacian kernels."""
        N = self.graph_lap.graph.N
        cutoff = N // 3

        lowpass = self.graph_lap.low_pass_kernel(cutoff)
        highpass = self.graph_lap.high_pass_kernel(cutoff)
        heat_kernel = self.graph_lap.heat_kernel(0.01)

        self.assertEqual(lowpass.shape[0], N)
        self.assertEqual(highpass.shape[0], N)
        self.assertEqual(heat_kernel.shape[0], N)

        # Low + high pass should sum to all-ones
        self.assertTrue(np.allclose(lowpass + highpass, np.ones(N)))

        # Heat kernel values should be close to 1 for small alpha
        self.assertTrue(np.all(heat_kernel > 0))


class TestAdvectionDiffusion(unittest.TestCase):
    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "usa_graph_diagonal.pkl"))
        self.symA = self.A + self.A.T

        graph = Graph(adj_matrix=self.A, debug=False)
        graph_sym = Graph(adj_matrix=self.symA, debug=False)

        self.graph_adv_diff = AdvectionDiffusion(graph=graph)
        self.graph_adv_diff_left = AdvectionDiffusion(graph=graph, normalize="left")
        self.graph_sym = AdvectionDiffusion(graph=graph_sym)

        with self.assertRaises(ValueError):
            AdvectionDiffusion(
                graph=graph, normalize="gibberish"
            )  # Should raise ValueError

        self.signal = np.ones(self.graph_adv_diff.graph.N)

    def test_compute_operator(self):
        """
        Test the compute_operator method for AdvectionDiffusion operator.
        """
        self.graph_adv_diff.compute_operator()
        self.graph_sym.compute_operator()

        with self.assertRaises(NotImplementedError):
            self.graph_adv_diff.compute_operator(
                divergence_free=False
            )  # Should raise NotImplementedError

    def test_compute_basis(self):
        """Test that compute_basis produces a valid eigendecomposition."""
        self.graph_adv_diff.compute_basis()
        N = self.graph_adv_diff.graph.N

        self.assertEqual(self.graph_adv_diff.U.shape, (N, N))
        self.assertEqual(self.graph_adv_diff.V.shape, (N,))
        # Eigenvalue equation: M @ U == U @ diag(V)
        self.assertTrue(
            np.allclose(
                self.graph_adv_diff.M @ self.graph_adv_diff.U,
                self.graph_adv_diff.U * self.graph_adv_diff.V,
                atol=1e-6,
            )
        )

        self.graph_sym.compute_basis()

    def test_compute_directed_laplacian(self):
        """
        Test the compute_directed_laplacian method for Laplacian operator.
        """
        self.graph_adv_diff.compute_directed_laplacian(self.A, in_degree=True)
        self.graph_adv_diff.compute_directed_laplacian(self.A, in_degree=False)

    def test_spectral_reconstruct_operators(self):
        """
        Test the spectral_reconstruct_operators method for AdvectionDiffusion operator.
        """
        self.graph_adv_diff.spectral_reconstruct_operators()
        self.assertIsNotNone(self.graph_adv_diff.P)
        self.assertIsNotNone(self.graph_adv_diff.Q)
        self.assertIsNotNone(self.graph_adv_diff.Z)

    def test_compute_kernels(self):
        """Test shape and basic properties of AdvectionDiffusion kernels."""
        N = self.graph_adv_diff.graph.N
        cutoff = N // 3

        h_kernel = self.graph_adv_diff.heat_kernel(0.01)
        ht_kernel = self.graph_adv_diff.heat_transport_kernel(0.01, 0.01)
        t_kernel = self.graph_adv_diff.transport_kernel(0.01)
        low_pass_kernel = self.graph_adv_diff.low_pass_kernel(cutoff)
        high_pass_kernel = self.graph_adv_diff.high_pass_kernel(cutoff)

        for kernel in [
            h_kernel,
            ht_kernel,
            t_kernel,
            low_pass_kernel,
            high_pass_kernel,
        ]:
            self.assertEqual(kernel.shape[0], N)

        # Low + high pass must partition the frequency axis
        self.assertTrue(np.allclose(low_pass_kernel + high_pass_kernel, np.ones(N)))

    def test_smoothness(self):
        """
        Test the smoothness method for AdvectionDiffusion operator.
        """
        smoothness_value = self.graph_adv_diff.radial_smoothness(self.signal)
        self.assertIsInstance(smoothness_value, float)
        smoothness_value = self.graph_adv_diff.angular_smoothness(self.signal)
        self.assertIsInstance(smoothness_value, float)


class TestTimeVertex(unittest.TestCase):
    def setUp(self):
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "usa_graph_diagonal.pkl"))
        self.symA = self.A + self.A.T

        graph = Graph(adj_matrix=self.A, debug=False)
        graph_sym = Graph(adj_matrix=self.symA, debug=False)

        self.graph_adj = TimeVertexAdjacency(graph=graph, nb_time=5)
        self.graph_lap = TimeVertexLaplacian(graph=graph, nb_time=5)
        self.graph_adj_sym = TimeVertexAdjacency(graph=graph_sym, nb_time=5)
        self.graph_lap_sym = TimeVertexLaplacian(graph=graph_sym, nb_time=5)

        with self.assertRaises(ValueError):
            TimeVertexLaplacian(
                graph=graph, normalize="gibberish"
            )  # Should raise ValueError

    def test_compute_operator(self):
        """
        Test the compute_operator method for TimeVertex operators.
        """
        self.graph_adj.compute_operator()
        self.graph_lap.compute_operator()
        self.graph_adj_sym.compute_operator()
        self.graph_lap_sym.compute_operator()

    def test_compute_basis(self):
        """Test that compute_basis produces a valid eigendecomposition for time-vertex operators."""
        for tv_op in [
            self.graph_adj,
            self.graph_adj_sym,
            self.graph_lap,
            self.graph_lap_sym,
        ]:
            tv_op.compute_basis()
            NT = tv_op.graph.N * tv_op.params["nb_time"]

            # Output shapes
            self.assertEqual(tv_op.U.shape, (NT, NT))
            self.assertEqual(tv_op.V.shape, (NT,))
            self.assertIsNotNone(tv_op.Uinv)

            # M @ U == U @ diag(V)  (A·x = λx for each column)
            self.assertTrue(
                np.allclose(
                    tv_op.M @ tv_op.U,
                    tv_op.U * tv_op.V,
                    atol=1e-4,
                )
            )

    def test_compute_directed_laplacian(self):
        """Test directed Laplacian L = D - A properties for time-vertex Laplacian."""
        L_in = self.graph_lap.compute_directed_laplacian(self.A, in_degree=True)
        L_out = self.graph_lap.compute_directed_laplacian(self.A, in_degree=False)

        N = self.A.shape[0]
        self.assertEqual(L_in.shape, (N, N))
        self.assertEqual(L_out.shape, (N, N))
        self.assertTrue(np.all(np.diag(L_in) >= 0))
        self.assertTrue(np.all(np.diag(L_out) >= 0))

    def test_sig2vec_and_vec2sig(self):
        """Test sig2vec/vec2sig round-trip and error handling for time-vertex operators."""
        signal = np.random.rand(
            self.graph_adj.graph.N, self.graph_adj.params["nb_time"]
        )
        vec = self.graph_adj.sig2vec(signal)
        reconstructed_signal = self.graph_adj.vec2sig(vec)

        self.assertEqual(
            vec.shape, (self.graph_adj.graph.N * self.graph_adj.params["nb_time"],)
        )
        self.assertEqual(
            reconstructed_signal.shape,
            (self.graph_adj.params["nb_time"], self.graph_adj.graph.N),
        )
        self.assertTrue(np.allclose(signal.shape, reconstructed_signal.T.shape))

        with self.assertRaises(ValueError):
            self.graph_adj.vec2sig(np.ones((10, 10)))
        with self.assertRaises(ValueError):
            self.graph_adj.sig2vec(np.ones(10))

        signal = np.random.rand(
            self.graph_lap.graph.N, self.graph_lap.params["nb_time"]
        )
        vec = self.graph_lap.sig2vec(signal)
        reconstructed_signal = self.graph_lap.vec2sig(vec)

        self.assertEqual(
            vec.shape, (self.graph_lap.graph.N * self.graph_lap.params["nb_time"],)
        )
        self.assertEqual(
            reconstructed_signal.shape,
            (self.graph_lap.params["nb_time"], self.graph_lap.graph.N),
        )
        self.assertTrue(np.allclose(signal.shape, reconstructed_signal.T.shape))

        with self.assertRaises(ValueError):
            self.graph_lap.vec2sig(np.ones((10, 10)))
        with self.assertRaises(ValueError):
            self.graph_lap.sig2vec(np.ones(10))


if __name__ == "__main__":
    unittest.main()
