Graphs
======

.. currentmodule:: gyraph.graphs

The graph object
----------------

.. autoclass:: Graph
   :members:
   :undoc-members:
   :show-inheritance:

Synthetic generators
--------------------

Small, controlled graphs — useful for validating a method before letting it
loose on real data. Each returns a NetworkX graph and a node-position mapping
ready to hand to :class:`Graph`.

.. autofunction:: create_cycle_graph
.. autofunction:: create_directed_torus
.. autofunction:: create_flower_graph
.. autofunction:: assymetric_erdos_renyi_graph

Flow fields and meshes
----------------------

Graphs whose edge directions encode a physical flow, and nearest-neighbour
graphs built from point clouds and surfaces.

.. autofunction:: create_torus_graph
.. autofunction:: create_torus_laminar_flow_graph
.. autofunction:: create_torus_vortex_graph
.. autofunction:: create_torus_multi_vortex_graph
.. autofunction:: create_vortex_graph
.. autofunction:: create_vortex_graph_surface
.. autofunction:: create_mesh_graph
.. autofunction:: create_sphere_graph
.. autofunction:: create_cube_graph
.. autofunction:: create_bunny_graph
.. autofunction:: create_dragon_graph
.. autofunction:: create_inverted_parabola_grid
.. autofunction:: create_hyperbolic_paraboloid_grid
.. autofunction:: create_two_holes_curvature

Graph manipulation
------------------

.. autofunction:: upsample_scheme_graph
.. autofunction:: combine_graphs
.. autofunction:: get_cycles
