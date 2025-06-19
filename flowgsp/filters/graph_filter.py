"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from ..utils import *
from .filter import Filter

class GraphFilter(Filter):
    """
    A base class for graph filters that applies a polynomial of the GSO
    to a signal in vertex domain
    """
    def __init__(self, graph, name=None, params=None):
        super().__init__(graph, name=name, params=params)
        self.name = "GraphFilter"

    def apply(self, signal):
        """
        Override the apply method for child classes to implement specific filtering logic.
        """
        raise NotImplementedError("The apply method must be implemented in the child class.")
    
    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"