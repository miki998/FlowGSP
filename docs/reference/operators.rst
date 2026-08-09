Operators
=========

.. currentmodule:: gyraph.operators

A shift operator turns a graph into a spectral domain. Attach one with
:meth:`gyraph.graphs.Graph.set_operator`, which builds the matrix, computes the
eigendecomposition and stores the result in ``graph.operator``.

Every operator exposes the same core attributes — ``M`` (the matrix), ``U``
(the Fourier basis), ``Uinv``, ``V`` (eigenvalues) and ``frequencies`` — and
the same transform pair, :meth:`Operator.GFT` and :meth:`Operator.inverseGFT`.

Base class
----------

.. autoclass:: Operator
   :members:
   :undoc-members:
   :show-inheritance:

Vertex operators
----------------

.. autoclass:: Adjacency
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Laplacian
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: AdvectionDiffusion
   :members:
   :undoc-members:
   :show-inheritance:

Time–vertex operators
---------------------

Joint operators over the product of the graph and a time axis, for signals that
evolve on the graph. ``sig2vec`` / ``vec2sig`` move between the
``(N, T)`` signal layout and the flattened ``(N*T,)`` vector the operator acts
on.

.. autoclass:: TimeVertexLaplacian
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TimeVertexAdjacency
   :members:
   :undoc-members:
   :show-inheritance:

Diagonalisability repair
------------------------

A directed shift operator may be defective — a repeated eigenvalue with too few
eigenvectors — in which case no Fourier basis exists. These helpers find the
smallest edge perturbation that removes the offending Jordan blocks or exact
zero eigenvalues. ``compute_basis`` calls them automatically when needed.

.. autofunction:: destroy_jordan_blocks
.. autofunction:: destroy_jordan_blocks_laplacian
.. autofunction:: destroy_zero_eigenvals
.. autofunction:: find_best_pair
