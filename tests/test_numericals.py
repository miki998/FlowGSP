import unittest
import numpy as np
from gyraph.utils.numericals import (
    no_decimal,
    normalize,
    standardize,
    signed_amplitude,
    smooth_1d,
    signaltonoise_dB,
    hermitian,
    symmetry,
    antisymmetry,
    laplacian_to_adj,
    spatial_smooth,
    peak_snr,
    estimate_snr,
    low_rank_approximation_m,
    low_rank_approximation_ri,
)


class TestNumericals(unittest.TestCase):
    def test_no_decimal_real_values(self):
        array = np.array([1.0, 0.00000000001, 2.0, 0.000000000001])
        result = no_decimal(array)
        expected = np.array([1.0, 0.0, 2.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_no_decimal_complex_values(self):
        array = np.array(
            [
                1.0 + 1.0j,
                0.00000000001 + 0.00000000001j,
                2.0 + 2.0j,
                0.000000000001 + 0.000000000001j,
            ]
        )
        result = no_decimal(array)
        expected = np.array([1.0 + 1.0j, 0.0 + 0.0j, 2.0 + 2.0j, 0.0 + 0.0j])
        np.testing.assert_array_equal(result, expected)

    def test_no_decimal_with_custom_tolerance(self):
        array = np.array([1.0, 0.1, 2.0, 0.01])
        result = no_decimal(array, tol=0.05)
        expected = np.array([1.0, 0.1, 2.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_no_decimal_all_values_below_tolerance(self):
        array = np.array([0.00000000001, 0.000000000001])
        result = no_decimal(array)
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_no_decimal_no_values_below_tolerance(self):
        array = np.array([1.0, 2.0, 3.0])
        result = no_decimal(array)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_no_decimal_empty_array(self):
        array = np.array([])
        result = no_decimal(array)
        expected = np.array([])
        np.testing.assert_array_equal(result, expected)

    def test_normalize_real_values(self):
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize(array)
        expected = (array - np.mean(array)) / np.std(array)
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_complex_values(self):
        array = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
        result = normalize(array)
        expected = (array - np.mean(array)) / np.std(array)
        np.testing.assert_array_almost_equal(result, expected)

    def test_standardize_real_values(self):
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = standardize(array)
        expected = (array - np.min(array)) / np.max(array - np.min(array))
        np.testing.assert_array_almost_equal(result, expected)

    def test_standardize_complex_values(self):
        array = np.array([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
        result = standardize(array)
        expected = (array - np.min(array)) / np.max(array - np.min(array))
        np.testing.assert_array_almost_equal(result, expected)

    def test_signed_amplitude(self):
        array = np.array([1.0 + 1.0j, -2.0 + 2.0j, 0.0 + 3.0j])
        result = signed_amplitude(array)
        expected = np.array([np.abs(1.0 + 1.0j), -np.abs(-2.0 + 2.0j), 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_smooth_1d(self):
        array = np.array([1, 2, 3, 4, 5])
        result = smooth_1d(array, 3)
        expected = np.array([2.0, 3.0, 4.0, 14 / 3, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_signaltonoise_dB(self):
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = signaltonoise_dB(array)
        expected = 20 * np.log10(np.mean(array) / np.std(array))
        self.assertAlmostEqual(result, expected)

    def test_hermitian(self):
        M = np.array([[1.0 + 1.0j, 2.0], [3.0j, 4.0]])
        np.testing.assert_array_equal(hermitian(M), np.conjugate(M).T)

    def test_symmetry_antisymmetry(self):
        # A symmetric matrix has zero symmetry defect
        S = np.array([[1.0, 2.0], [2.0, 3.0]])
        self.assertAlmostEqual(symmetry(S), 0.0)
        self.assertGreater(antisymmetry(S), 0.0)
        # A skew-symmetric matrix has zero antisymmetry defect
        K = np.array([[0.0, 1.0], [-1.0, 0.0]])
        self.assertAlmostEqual(antisymmetry(K), 0.0)
        self.assertGreater(symmetry(K), 0.0)

    def test_laplacian_to_adj(self):
        A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        L = np.diag(A.sum(axis=1)) - A
        np.testing.assert_allclose(laplacian_to_adj(L), A)

    def test_laplacian_to_adj_complex_raises(self):
        with self.assertRaises(ValueError):
            laplacian_to_adj(np.eye(2) + 1j)

    def test_spatial_smooth(self):
        coords = np.array([[0.0, 0.0], [0.0, 0.5], [10.0, 10.0]])
        signal = np.array([1.0, 3.0, 7.0])
        # Sequential in-place averaging: node 0 -> mean(1, 3) = 2,
        # node 1 -> mean(updated 2, 3) = 2.5, node 2 isolated -> unchanged
        result = spatial_smooth(signal, coords, size=1.0)
        np.testing.assert_allclose(result, [2.0, 2.5, 7.0])
        # Non-positive size returns the signal unchanged
        np.testing.assert_allclose(spatial_smooth(signal, coords, size=0), signal)

    def test_peak_snr(self):
        ground = np.ones(4)
        # Zero MSE is capped at 200 dB
        self.assertEqual(peak_snr(ground, ground), 200)
        denoised = np.array([1.0, 1.0, 1.0, 0.5])
        expected = 20 * np.log10(1.0 / np.sqrt(np.mean((ground - denoised) ** 2)))
        self.assertAlmostEqual(peak_snr(ground, denoised), expected)

    def test_peak_snr_along_axis(self):
        ground = np.ones((3, 4))
        denoised = ground.copy()
        denoised[0, 0] = 0.9
        result = peak_snr(ground, denoised, axis=1)
        self.assertEqual(result.shape, (3,))
        # Rows with zero MSE are capped at 200 dB
        np.testing.assert_allclose(result[1:], 200)
        self.assertLess(result[0], 200)

    def test_estimate_snr(self):
        self.assertAlmostEqual(estimate_snr(10.0, 2.0, return_decibel=False), 5.0)
        self.assertAlmostEqual(
            estimate_snr(np.full(5, 10.0), np.ones(5), return_decibel=True), 10.0
        )
        # Zero noise is capped at max_snr
        self.assertEqual(estimate_snr(1.0, 0.0), 200)
        self.assertEqual(estimate_snr(np.ones(3), np.zeros(3)), 200)

    def test_estimate_snr_mixed_types_raises(self):
        with self.assertRaises(ValueError):
            estimate_snr(np.ones(3), 1.0)

    def test_low_rank_approximation_m(self):
        rng = np.random.default_rng(0)
        A = 0.5 * rng.standard_normal((20, 20))
        K = 4
        vals, vecs = low_rank_approximation_m(A, K)
        # Conjugate-pair completion may add one extra eigenvalue
        self.assertIn(len(vals), (K, K + 1))
        self.assertEqual(vecs.shape, (20, len(vals)))
        # Sorted by amplitude
        self.assertTrue(np.all(np.diff(np.abs(vals)) >= -1e-12))
        # Each pair is a genuine eigenpair of A
        for k in range(len(vals)):
            np.testing.assert_allclose(A @ vecs[:, k], vals[k] * vecs[:, k], atol=1e-8)

    def test_low_rank_approximation_ri(self):
        rng = np.random.default_rng(0)
        A = 0.5 * rng.standard_normal((20, 20))
        K1, K2 = 4, 4
        (vals_re, vecs_re), (vals_im, vecs_im) = low_rank_approximation_ri(A, K1, K2)
        self.assertIn(len(vals_re), (K1, K1 + 1))
        self.assertIn(len(vals_im), (K2, K2 + 1))
        self.assertEqual(vecs_re.shape, (20, len(vals_re)))
        self.assertEqual(vecs_im.shape, (20, len(vals_im)))
        for vals, vecs in [(vals_re, vecs_re), (vals_im, vecs_im)]:
            self.assertTrue(np.all(np.diff(np.abs(vals)) >= -1e-12))
            for k in range(len(vals)):
                np.testing.assert_allclose(
                    A @ vecs[:, k], vals[k] * vecs[:, k], atol=1e-8
                )


if __name__ == "__main__":
    unittest.main()
