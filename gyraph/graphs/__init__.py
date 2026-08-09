"""
Copyright © 2024 Chun Hei Michael Chan, MIPLab EPFL
"""

from .graph import Graph
from .graph_utils import upsample_scheme_graph, combine_graphs, get_cycles
from .basic_graphs import (
    create_cycle_graph,
    create_flower_graph,
    create_directed_torus,
    assymetric_erdos_renyi_graph,
)
from .physical_graphs import (
    create_torus_graph,
    create_torus_laminar_flow_graph,
    create_torus_vortex_graph,
    create_torus_multi_vortex_graph,
    create_vortex_graph,
    create_bunny_graph,
    create_dragon_graph,
    create_cube_graph,
    create_sphere_graph,
    create_inverted_parabola_grid,
    create_two_holes_curvature,
    create_hyperbolic_paraboloid_grid,
    create_vortex_graph_surface,
    create_mesh_graph,
)

__all__ = [
    "Graph",
    "upsample_scheme_graph",
    "combine_graphs",
    "get_cycles",
    "create_cycle_graph",
    "create_flower_graph",
    "create_directed_torus",
    "assymetric_erdos_renyi_graph",
    "create_torus_graph",
    "create_torus_laminar_flow_graph",
    "create_torus_vortex_graph",
    "create_torus_multi_vortex_graph",
    "create_vortex_graph",
    "create_bunny_graph",
    "create_dragon_graph",
    "create_cube_graph",
    "create_sphere_graph",
    "create_inverted_parabola_grid",
    "create_two_holes_curvature",
    "create_hyperbolic_paraboloid_grid",
    "create_vortex_graph_surface",
    "create_mesh_graph",
]
