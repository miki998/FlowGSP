GyRAPH
======

**A Python framework for signal processing on directed graphs.**

.. image:: https://github.com/miki998/GyRAPH/workflows/Launch%20Unittest/badge.svg
   :target: https://github.com/miki998/GyRAPH/actions
   :alt: Tests

.. image:: https://readthedocs.org/projects/gyraph/badge/?version=latest
   :target: https://gyraph.readthedocs.io/en/latest/
   :alt: Documentation

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://github.com/miki998/GyRAPH/blob/main/LICENSE
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.9+

Classical graph signal processing assumes a symmetric graph, so that the shift
operator is diagonalizable in an orthonormal basis and its spectrum is real.
Most real networks — flows, transport, causal or effective connectivity — are
not symmetric.

GyRAPH takes the asymmetry seriously. It provides graph shift operators for
directed graphs, a complex-valued Graph Fourier Transform built on their
(possibly non-orthogonal) eigenbases, a family of filters that operate in the
resulting spectral domain, and statistical tooling — surrogates and
stationarity tests — for hypothesis testing on directed graph signals.

The library builds on NumPy, SciPy and NetworkX, and is designed to slot into
existing analysis pipelines rather than replace them.

.. code-block:: python

   import numpy as np
   from gyraph.graphs import Graph, create_directed_torus
   from gyraph.filters import SpectralFilter

   G, pos = create_directed_torus(Nr=8, Nc=6, directed=True)
   graph = Graph(G=G, pos=pos)
   graph.set_operator("adjacency")     # computes the Graph Fourier basis

   x = np.random.randn(graph.N)
   kernel = np.zeros(graph.N)
   kernel[:10] = 1.0                   # keep the 10 lowest graph frequencies

   x_low = SpectralFilter(graph).apply(x, kernel)

At a glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - What it gives you
   * - :doc:`reference/graphs`
     - A :class:`~gyraph.graphs.Graph` container around NetworkX, synthetic
       generators (cycles, directed tori, asymmetric Erdős–Rényi) and
       mesh/flow graphs (sphere, cube, bunny, vortex fields).
   * - :doc:`reference/operators`
     - Adjacency, directed Laplacian, advection–diffusion and time–vertex
       shift operators, each exposing ``GFT`` / ``inverseGFT`` and an
       automatic Jordan-block fallback for non-diagonalizable matrices.
   * - :doc:`reference/filters`
     - Spectral, polynomial, Chebyshev, Faber, Hilbert, Tikhonov and Wiener
       filters — including approximations that avoid an eigendecomposition.
   * - :doc:`reference/surrogates`
     - Phase-randomised and structure-preserving surrogates, PSD estimation
       and stationarity testing on directed graphs.
   * - :doc:`reference/utils`
     - Smoothness metrics (Dirichlet, total variation, Sobolev, directed
       variation), numerics and publication-styled plotting.

Getting started
---------------

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   tutorials/index
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: About

   papers
   contributing
   citing

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
