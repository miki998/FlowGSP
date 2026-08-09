Examples and data
=================

Runnable scripts and notebooks live in the `examples/
<https://github.com/miki998/GyRAPH/tree/main/examples>`_ directory of the
repository.

Notebooks
---------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Covers
   * - `examples/basic/basics.ipynb
       <https://github.com/miki998/GyRAPH/blob/main/examples/basic/basics.ipynb>`_
     - Graph creation, operators and the Fourier basis, simple filtering,
       visualisation — start here
   * - `examples/basic/01_basic_graph_creation.py
       <https://github.com/miki998/GyRAPH/blob/main/examples/basic/01_basic_graph_creation.py>`_
     - The same first steps as a plain script
   * - `examples/advanced/advanced.ipynb
       <https://github.com/miki998/GyRAPH/blob/main/examples/advanced/advanced.ipynb>`_
     - Polynomial approximations, denoising, surrogates and stationarity on
       real data

Run them after an editable install:

.. code-block:: bash

   pip install -e .
   jupyter notebook examples/basic/basics.ipynb

Bundled datasets
----------------

The `data/ <https://github.com/miki998/GyRAPH/tree/main/data>`_ directory ships
the graphs used throughout the examples and the test suite:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Dataset
     - Description
   * - ``manhattan_graph_data/``
     - Mid-Manhattan road network with NYC taxi flow signals — a directed graph
       where edge direction is literal
   * - ``usa_graph_data/``
     - US state adjacency graph with boundary shapefiles
   * - ``temperature_bretagne_graph_data/``
     - Brittany weather-station network with temperature recordings
