Smoothness on directed graphs
=============================

"Smooth" on a graph means *similar values on connected nodes*. On a directed
graph the definition splits: a signal can be smooth along the flow and rough
against it. GyRAPH ships several measures, each answering a slightly different
question.

.. code-block:: python

   import numpy as np
   from gyraph.graphs import Graph, create_directed_torus
   from gyraph.utils import dirichlet, TV, sobolev, directed_variation

   G, pos = create_directed_torus(Nr=8, Nc=6, directed=True)
   graph = Graph(G=G, pos=pos)
   graph.set_operator("laplacian")

   x = np.random.randn(graph.N)
   L = graph.operator.M
   A = graph.adj_matrix

Vertex-domain measures
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Measures
   * - :func:`~gyraph.utils.dirichlet`
     - :math:`x^\top L x` — the classical Dirichlet energy, quadratic
   * - :func:`~gyraph.utils.TV`
     - :math:`\|x - A x\|_p` — total variation as a shift difference,
       ``norm="L1"`` or ``"L2"``
   * - :func:`~gyraph.utils.sobolev`
     - Sobolev norm, penalising higher-order roughness
   * - :func:`~gyraph.utils.directed_variation`
     - only counts variation *along* edge direction — asymmetric by design

.. code-block:: python

   dirichlet(x, L)
   TV(x, A, norm="L1")
   sobolev(x, L, norm="L2")
   directed_variation(x, A)

``directed_variation`` is the one that distinguishes directed GSP from its
symmetric counterpart: it is not invariant to reversing every edge, so it
separates a signal flowing with the graph from one flowing against it.

Radial and angular smoothness
-----------------------------

Under the advection–diffusion operator the complex spectrum splits naturally
into a *radial* part (eigenvalue magnitude — diffusion, how fast a mode decays)
and an *angular* part (eigenvalue phase — advection, how fast a mode rotates).
Smoothness follows the same split:

.. code-block:: python

   graph.set_operator("advection_diffusion")
   op = graph.operator

   op.radial_smoothness(x, norm="L2")
   op.angular_smoothness(x, norm="L2")

   op.radial_frequencies      # |lambda|
   op.angular_frequencies     # |arg(lambda)|

Two orderings of the same harmonics come with it — ``op.radial_order`` and
``op.angular_order`` — so "low-pass" can mean *slowly decaying* or *slowly
rotating*, and you pick which via the ``mode`` argument:

.. code-block:: python

   kernel = op.low_pass_kernel(limfreq=10, mode="radial")
   kernel = op.low_pass_kernel(limfreq=10, mode="angular")

Transport kernels
-----------------

The same operator gives kernels with a physical reading — pure diffusion, pure
transport, or both:

.. code-block:: python

   op.heat_kernel(alpha=0.1)                     # diffusion only
   op.transport_kernel(alpha=0.1)                # advection only
   op.heat_transport_kernel(alpha=0.1, beta=0.5) # both

Feed any of these to a :class:`~gyraph.filters.SpectralFilter` exactly as in
:doc:`filtering`.
