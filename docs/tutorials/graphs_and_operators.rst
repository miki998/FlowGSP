Graphs and operators
====================

Everything in GyRAPH starts from a :class:`~gyraph.graphs.Graph` and a **shift
operator**. The operator is what turns a graph into a Fourier basis.

Building a graph
----------------

A :class:`~gyraph.graphs.Graph` wraps a NetworkX graph together with node
positions used for plotting. You can build it from a NetworkX object or
directly from an adjacency matrix.

.. code-block:: python

   import numpy as np
   from gyraph.graphs import Graph, create_directed_torus

   # A directed torus: 8 rows x 6 columns, asymmetric by construction
   G, pos = create_directed_torus(Nr=8, Nc=6, directed=True)
   graph = Graph(G=G, pos=pos)

   # ... or straight from a matrix
   A = np.array([[0, 1, 0],
                 [0, 0, 1],
                 [1, 0, 0]], dtype=float)
   cycle = Graph(adj_matrix=A)

Generators live in :mod:`gyraph.graphs`. The synthetic ones
(:func:`~gyraph.graphs.create_cycle_graph`,
:func:`~gyraph.graphs.create_directed_torus`,
:func:`~gyraph.graphs.assymetric_erdos_renyi_graph`) are useful for controlled
experiments; the physical ones
(:func:`~gyraph.graphs.create_torus_vortex_graph`,
:func:`~gyraph.graphs.create_sphere_graph`,
:func:`~gyraph.graphs.create_bunny_graph`) produce meshes and flow fields on
surfaces.

How asymmetric is it?
---------------------

Directed GSP only pays off when the graph is genuinely asymmetric, so the
:class:`~gyraph.graphs.Graph` exposes a few diagnostics:

.. code-block:: python

   graph.is_directed()          # does the NetworkX object carry directed edges?
   graph.is_assymmetric()       # is A != A.T ?
   graph.assymetry_level()      # scalar summary of ||A - A.T||
   graph.assymetry_edge_level() # fraction of edges without a reciprocal
   graph.degree_entropy()       # entropy of the in/out-degree distributions

Attaching a shift operator
--------------------------

:meth:`~gyraph.graphs.Graph.set_operator` builds the operator matrix **and**
its eigendecomposition, and stores the result in ``graph.operator``:

.. code-block:: python

   graph.set_operator("adjacency")
   op = graph.operator

   op.M              # the operator matrix itself
   op.V              # eigenvalues  (complex for directed graphs)
   op.U              # graph Fourier basis (columns are eigenvectors)
   op.Uinv           # its inverse — not the conjugate transpose in general
   op.frequencies    # graph frequencies, sorted ascending

Five operators are available:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Name
     - Class
   * - ``"adjacency"``
     - :class:`~gyraph.operators.Adjacency`
   * - ``"laplacian"``
     - :class:`~gyraph.operators.Laplacian`
   * - ``"advection_diffusion"``
     - :class:`~gyraph.operators.AdvectionDiffusion`
   * - ``"time_vertex_laplacian"``
     - :class:`~gyraph.operators.TimeVertexLaplacian`
   * - ``"time_vertex_adjacency"``
     - :class:`~gyraph.operators.TimeVertexAdjacency`

Keyword arguments are forwarded to the operator constructor:

.. code-block:: python

   graph.set_operator("laplacian", normalize="symmetric")
   graph.set_operator("advection_diffusion", divergence_free=True)

Non-normal and non-diagonalizable operators
-------------------------------------------

For a symmetric graph the shift operator is normal, its eigenbasis is
orthonormal and the GFT is unitary. For a directed graph none of that is
guaranteed. Two things can go wrong, and GyRAPH handles both:

*The operator is normal but not symmetric.* Fine — the spectrum is complex and
``Uinv`` is not ``U.conj().T``, which is why GyRAPH always keeps ``Uinv``
explicitly. Measure how far you are from normal with:

.. code-block:: python

   op.normality()    # 0 when M M^H == M^H M, larger otherwise

*The operator is defective* (a repeated eigenvalue without a full set of
eigenvectors, i.e. a non-trivial Jordan block). Then there is no eigenbasis at
all. The helpers in :mod:`gyraph.operators` perturb the graph minimally until
the matrix is diagonalizable:

.. code-block:: python

   from gyraph.operators import destroy_jordan_blocks, destroy_zero_eigenvals

   A_fixed = destroy_jordan_blocks(A)     # smallest edge edits removing Jordan blocks
   A_fixed = destroy_zero_eigenvals(A)    # same, for exactly-zero eigenvalues

``compute_basis`` calls into these automatically when it detects a defective
matrix, so most of the time you never invoke them directly.

The Graph Fourier Transform
---------------------------

Once a basis exists, the transform pair is on the operator:

.. code-block:: python

   x = np.random.randn(graph.N)

   coef = op.GFT(x)                       # Uinv @ x
   assert np.allclose(op.inverseGFT(coef), x)

Because the spectrum is complex, harmonics come in conjugate pairs. Two helpers
make that structure explicit:

.. code-block:: python

   op.conjugate_frequency(3)   # index of the harmonic conjugate to harmonic 3
   op.eigvalues_pairs()        # groups of conjugate eigenvalues (pairs or singletons)

Keeping track of the pairing is what lets you build kernels whose output stays
real — see :doc:`filtering`.

Visualising
-----------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, ax = plt.subplots()
   graph.draw(axes=ax)                    # topology, symmetric vs asymmetric edges coloured
   graph.draw_signal(x, axes=ax, cmap="coolwarm", colorbar=True)
