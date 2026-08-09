"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from .base import Operator  # noqa: F401
from .jordan_destroy import (
    find_best_pair,
    destroy_jordan_blocks,
    destroy_jordan_blocks_laplacian,
    destroy_zero_eigenvals,
)  # noqa: F401
from .adjacency import Adjacency  # noqa: F401
from .laplacian import Laplacian  # noqa: F401
from .advection_diffusion import AdvectionDiffusion  # noqa: F401
from .time_vertex_laplacian import (
    TimeVertexLaplacian,
    TimeVertexAdjacency,
)  # noqa: F401

__all__ = [
    "Operator",
    "Adjacency",
    "Laplacian",
    "AdvectionDiffusion",
    "TimeVertexLaplacian",
    "TimeVertexAdjacency",
    "find_best_pair",
    "destroy_jordan_blocks",
    "destroy_jordan_blocks_laplacian",
    "destroy_zero_eigenvals",
]
