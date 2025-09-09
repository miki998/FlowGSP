import unittest
import numpy as np
from flowgsp.utils.numericals import *

class TestNumericals(unittest.TestCase):

    def test_no_decimal_real_values(self):
        array = np.array([1.0, 0.00000000001, 2.0, 0.000000000001])
        result = no_decimal(array)
        expected = np.array([1.0, 0.0, 2.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_no_decimal_complex_values(self):
        array = np.array([1.0 + 1.0j, 0.00000000001 + 0.00000000001j, 2.0 + 2.0j, 0.000000000001 + 0.000000000001j])
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
        expected = np.array([2.0, 3.0, 4.0, 14/3, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_signaltonoise_dB(self):
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = signaltonoise_dB(array)
        expected = 20 * np.log10(np.mean(array) / np.std(array))
        self.assertAlmostEqual(result, expected)

if __name__ == '__main__':
    unittest.main()