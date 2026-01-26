"""
Logging configuration for FlowGSP.

This module provides centralized logging configuration for the FlowGSP library
and experiments. It supports multiple log levels, console and file output,
and integrates seamlessly with the existing verbose parameter patterns.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"
EXPERIMENT_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def setup_logger(
    name: str = "flowgsp",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and/or file output.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is "flowgsp".
    level : int or str, optional
        Logging level. Can be an integer (e.g., logging.INFO) or
        a string (e.g., "INFO", "DEBUG"). Default is logging.INFO.
    log_file : str or Path, optional
        Path to log file. If None, file logging is disabled.
    format_string : str, optional
        Custom format string for log messages. If None, uses DEFAULT_FORMAT.
    console : bool, optional
        Whether to output logs to console. Default is True.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = setup_logger("myexperiment", level="DEBUG")
    >>> logger.info("Starting experiment")

    >>> logger = setup_logger("myexp", log_file="results/experiment.log")
    >>> logger.debug("Debug information")
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Set format
    if format_string is None:
        format_string = DEFAULT_FORMAT
    formatter = logging.Formatter(format_string)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "flowgsp") -> logging.Logger:
    """
    Get an existing logger or create a default one.

    Parameters
    ----------
    name : str, optional
        Name of the logger. Default is "flowgsp".

    Returns
    -------
    logging.Logger
        Logger instance.

    Examples
    --------
    >>> logger = get_logger("flowgsp.graphs")
    >>> logger.info("Creating graph")
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


def configure_experiment_logging(
    experiment_name: str,
    verbose: bool = True,
    log_file: Optional[Union[str, Path]] = None,
    results_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """
    Configure logging for an experiment.

    This function provides a convenient way to set up logging for experiments,
    respecting the verbose parameter pattern used throughout the codebase.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (used as logger name).
    verbose : bool, optional
        If True, set log level to INFO. If False, set to WARNING.
        Default is True.
    log_file : str or Path, optional
        Custom path to log file. If None and results_dir is provided,
        creates a log file in results_dir.
    results_dir : str or Path, optional
        Directory for results. If provided and log_file is None,
        creates a log file named "{experiment_name}.log" in this directory.

    Returns
    -------
    logging.Logger
        Configured logger for the experiment.

    Examples
    --------
    >>> logger = configure_experiment_logging(
    ...     "temperature_analysis",
    ...     verbose=True,
    ...     results_dir="results"
    ... )
    >>> logger.info("Starting temperature analysis")
    """
    # Determine log level based on verbose
    level = logging.INFO if verbose else logging.WARNING

    # Determine log file path
    if log_file is None and results_dir is not None:
        results_path = Path(results_dir)
        log_file = results_path / f"{experiment_name}.log"

    # Set up logger
    logger = setup_logger(
        name=experiment_name,
        level=level,
        log_file=log_file,
        format_string=EXPERIMENT_FORMAT,
        console=True,
    )

    return logger


def set_library_log_levels(level: Union[int, str] = logging.WARNING) -> None:
    """
    Set log levels for common noisy third-party libraries.

    This is useful to reduce noise in logs from libraries like matplotlib,
    networkx, numba, urllib3, PIL, and torch.

    Parameters
    ----------
    level : int or str, optional
        Logging level to set. Default is logging.WARNING.

    Examples
    --------
    >>> set_library_log_levels(logging.ERROR)

    Notes
    -----
    Affects the following libraries:
    - matplotlib
    - networkx
    - numba
    - urllib3
    - PIL
    - torch
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.WARNING)

    noisy_libraries = [
        "matplotlib",
        "networkx",
        "numba",
        "urllib3",
        "PIL",
        "torch",
    ]

    for lib_name in noisy_libraries:
        logging.getLogger(lib_name).setLevel(level)


def disable_logging() -> None:
    """
    Disable all logging output.

    This is useful for testing or when running in quiet mode.

    Examples
    --------
    >>> disable_logging()
    """
    logging.disable(logging.CRITICAL)


def enable_logging() -> None:
    """
    Re-enable logging after it was disabled.

    Examples
    --------
    >>> enable_logging()
    """
    logging.disable(logging.NOTSET)
