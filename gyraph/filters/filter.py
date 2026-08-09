"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from typing import Optional, Dict, Any


class Filter:
    """
    Base class for filters.
    This class provides a template for creating various types of filters
    that can be applied to signals on graphs.
    """

    def __init__(
        self,
        graph: Any,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.graph = graph
        self.name = name
        self.params = params if params is not None else {}

    def apply(self, signal):
        """
        Apply the filter to the input signal.
        This method should be overridden by subclasses.

        Parameters
        ----------
        signal : array-like
            Input signal to filter.

        Returns
        -------
        array-like
            Filtered signal.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __repr__(self) -> str:
        return f"<Filter(name={self.name}, params={self.params})>"
