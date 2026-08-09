API reference
=============

The library is organised in five subpackages, all re-exported at the top level,
so ``import gyraph`` gives you ``gyraph.graphs``, ``gyraph.operators``,
``gyraph.filters``, ``gyraph.surrogates`` and ``gyraph.utils``.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Subpackage
     - Contents
   * - :doc:`graphs`
     - The ``Graph`` container plus synthetic, mesh and flow-field generators
   * - :doc:`operators`
     - Shift operators and the Graph Fourier Transform
   * - :doc:`filters`
     - Spectral, polynomial and statistical filters
   * - :doc:`surrogates`
     - Surrogate generation, PSD estimation, stationarity testing
   * - :doc:`utils`
     - Smoothness metrics, statistics, numerics, plotting, I/O and logging

.. toctree::
   :maxdepth: 2

   graphs
   operators
   filters
   surrogates
   utils
