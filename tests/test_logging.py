"""
Test logging system functionality.

This test validates that the logging system works correctly and
can be imported without issues.

Note: This test uses direct module loading via importlib to avoid
importing the full flowgsp package, which has dependencies (torch, etc.)
that may not be available in the test environment. This allows us to
test the logging module in isolation.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from files to avoid dependency issues
import importlib.util  # noqa: E402


def load_logging_config():
    """Load logging_config module directly."""
    spec = importlib.util.spec_from_file_location(
        "logging_config",
        Path(__file__).parent.parent / "flowgsp" / "utils" / "logging_config.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_logging_utils():
    """Load logging_utils module directly."""
    # First ensure logging_config is available
    logging_config = load_logging_config()
    sys.modules["flowgsp.utils.logging_config"] = logging_config

    spec = importlib.util.spec_from_file_location(
        "logging_utils",
        Path(__file__).parent.parent / "experiments" / "logging_utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestLoggingConfig(unittest.TestCase):
    """Test the core logging configuration module."""

    @classmethod
    def setUpClass(cls):
        cls.logging_config = load_logging_config()

    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = self.logging_config.setup_logger("test_logger", level="INFO")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "test_logger")

    def test_setup_logger_with_file(self):
        """Test logger setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = self.logging_config.setup_logger(
                "test_logger", level="INFO", log_file=log_file
            )

            logger.info("Test message")

            # Check file was created
            self.assertTrue(os.path.exists(log_file))

            # Check file has content
            with open(log_file, "r") as f:
                content = f.read()
            self.assertIn("Test message", content)

    def test_configure_experiment_logging(self):
        """Test experiment logging configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self.logging_config.configure_experiment_logging(
                experiment_name="test_exp", verbose=True, results_dir=tmpdir
            )

            self.assertIsNotNone(logger)
            self.assertEqual(logger.name, "test_exp")

    def test_set_library_log_levels(self):
        """Test setting library log levels."""
        # Should not raise an exception
        self.logging_config.set_library_log_levels("ERROR")
        self.logging_config.set_library_log_levels("WARNING")

    def test_disable_enable_logging(self):
        """Test disabling and enabling logging."""
        # Should not raise an exception
        self.logging_config.disable_logging()
        self.logging_config.enable_logging()


class TestLoggingUtils(unittest.TestCase):
    """Test the experiment-specific logging utilities."""

    @classmethod
    def setUpClass(cls):
        cls.logging_utils = load_logging_utils()

    def test_setup_experiment_logger(self):
        """Test experiment logger setup."""
        logger = self.logging_utils.setup_experiment_logger(
            "test_exp", verbose=True, log_to_file=False
        )
        self.assertIsNotNone(logger)

    def test_setup_experiment_logger_with_file(self):
        """Test experiment logger with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = self.logging_utils.setup_experiment_logger(
                "test_exp", verbose=True, log_to_file=True, results_dir=tmpdir
            )

            logger.info("Test message")

            # Check log file was created
            log_file = os.path.join(tmpdir, "test_exp.log")
            self.assertTrue(os.path.exists(log_file))

    def test_experiment_logger_context_manager(self):
        """Test ExperimentLogger context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.logging_utils.ExperimentLogger(
                "test_exp", verbose=True, log_to_file=True, results_dir=tmpdir
            ) as logger:
                logger.info("Test message")

            # Check log file was created
            log_file = os.path.join(tmpdir, "test_exp.log")
            self.assertTrue(os.path.exists(log_file))

    def test_print_section(self):
        """Test print_section helper."""
        logging_config = load_logging_config()
        logger = logging_config.setup_logger("test", level="INFO")

        # Should not raise an exception
        self.logging_utils.print_section(logger, "Test Section")


if __name__ == "__main__":
    unittest.main()
