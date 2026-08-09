import os
import tempfile
import unittest

import pandas as pd
import torch

from gyraph.utils import np

# Import the module to increase coverage
from gyraph.utils import (
    dirichlet,
    TV,
    sobolev,
    directed_variation,
    save,
    load,
    save_json,
    load_json,
)


class TestUtils(unittest.TestCase):
    """
    Test cases for gyraph.utils functions.
    """

    def setUp(self):
        """Set up test fixtures for utils tests."""
        self.signal = np.random.rand(10)
        self.A = np.random.rand(10, 10)
        self.L = np.random.rand(10, 10)
        self.L = (self.L + self.L.T) / 2  # Make it symmetric

    def test_save_json(self):
        """Test the save function for JSON."""
        data = {"key": "value"}
        filename = "test_data.json"
        save_json(filename, data)
        loaded_data = load_json(filename)
        self.assertEqual(data, loaded_data)


class TestSerialization(unittest.TestCase):
    """
    Round-trip tests for the pickle and JSON serialization helpers,
    including the non-JSON types handled by save_json/load_json.
    """

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def test_pickle_roundtrip(self):
        data = {"array": np.arange(5), "nested": [1, "two", 3.0]}
        path = self._path("data.pkl")
        save(path, data)
        loaded = load(path)
        np.testing.assert_array_equal(loaded["array"], data["array"])
        self.assertEqual(loaded["nested"], data["nested"])

    def test_json_ndarray_roundtrip(self):
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        path = self._path("arr.json")
        save_json(path, {"arr": arr})
        loaded = load_json(path)
        np.testing.assert_array_equal(loaded["arr"], arr)
        self.assertEqual(loaded["arr"].dtype, arr.dtype)

    def test_json_numpy_scalars(self):
        path = self._path("scalars.json")
        save_json(path, {"i": np.int64(3), "f": np.float32(1.5), "b": np.bool_(True)})
        loaded = load_json(path)
        self.assertEqual(loaded["i"], 3)
        self.assertEqual(loaded["f"], 1.5)
        self.assertEqual(loaded["b"], True)

    def test_json_pandas_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        series = pd.Series({"x": 1, "y": 2})
        path = self._path("pandas.json")
        save_json(path, {"df": df, "series": series})
        loaded = load_json(path)
        pd.testing.assert_frame_equal(loaded["df"], df)
        pd.testing.assert_series_equal(loaded["series"], series)

    def test_json_torch_roundtrip(self):
        tensor = torch.arange(4, dtype=torch.float64).reshape(2, 2)
        path = self._path("torch.json")
        save_json(path, {"t": tensor})
        loaded = load_json(path)
        self.assertIsInstance(loaded["t"], torch.Tensor)
        self.assertTrue(torch.equal(loaded["t"], tensor))

    def test_json_unserializable_raises(self):
        with self.assertRaises(TypeError):
            save_json(self._path("bad.json"), {"obj": object()})


class TestMetrics(unittest.TestCase):
    """
    Test cases for gyraph.utils.metrics functions.
    """

    def setUp(self):
        """Set up test fixtures for metrics tests."""
        self.signal = np.random.rand(10)
        self.A = np.random.rand(10, 10)
        self.L = np.random.rand(10, 10)
        self.L = (self.L + self.L.T) / 2  # Make it symmetric

    def test_dirichlet(self):
        """
        Test the dirichlet function.
        """
        smoothness = dirichlet(self.signal, self.L, normalize=True)
        self.assertIsInstance(smoothness, float)

    def test_TV(self):
        """
        Test the TV function.
        """
        smoothness_L1 = TV(self.signal, self.A, norm="L1", normalize=True)
        smoothness_L2 = TV(self.signal, self.A, norm="L2", normalize=True)
        self.assertIsInstance(smoothness_L1, float)
        self.assertIsInstance(smoothness_L2, float)

    def test_sobolev(self):
        """
        Test the sobolev function.
        """
        smoothness = sobolev(self.signal, self.L, norm="L2", normalize=True)
        self.assertIsInstance(smoothness, float)

    def test_directed_variation(self):
        """
        Test the directed_variation function.
        """
        smoothness = directed_variation(self.signal, self.L, normalize=True)
        self.assertIsInstance(smoothness, float)


if __name__ == "__main__":
    unittest.main()
