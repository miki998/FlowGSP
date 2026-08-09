Utilities
=========

.. currentmodule:: gyraph.utils

Everything here is re-exported from ``gyraph.utils``, so a single import gets
you the metrics, statistics and plotting helpers.

Smoothness metrics
------------------

.. autofunction:: dirichlet
.. autofunction:: TV
.. autofunction:: sobolev
.. autofunction:: directed_variation

Statistics
----------

.. autofunction:: p_value
.. autofunction:: circular_stats
.. autofunction:: circular_correlation
.. autofunction:: sample_circular_complex_gaussian

Matrix numerics
---------------

.. autofunction:: symmetry
.. autofunction:: antisymmetry
.. autofunction:: hermitian
.. autofunction:: laplacian_to_adj
.. autofunction:: low_rank_approximation_m
.. autofunction:: low_rank_approximation_ri
.. autofunction:: no_decimal

Signal helpers
--------------

.. autofunction:: normalize
.. autofunction:: standardize
.. autofunction:: signed_amplitude
.. autofunction:: smooth_1d
.. autofunction:: spatial_smooth
.. autofunction:: estimate_snr
.. autofunction:: peak_snr
.. autofunction:: signaltonoise_dB

Graph construction from data
----------------------------

.. autofunction:: nearest_neighbour_graph

Plotting
--------

Graph and signal plotting live on the :class:`gyraph.graphs.Graph` object
itself (:meth:`~gyraph.graphs.Graph.draw` and
:meth:`~gyraph.graphs.Graph.draw_signal`). The functions below cover meshes,
surfaces and dynamics.

.. autofunction:: plot_mesh
.. autofunction:: plot_signal_on_regular_surface
.. autofunction:: signal2face
.. autofunction:: create_video_from_images
.. autofunction:: unique_color_generator

Input / output
--------------

.. autofunction:: save
.. autofunction:: load
.. autofunction:: save_json
.. autofunction:: load_json

Logging
-------

.. autofunction:: setup_logger
.. autofunction:: get_logger
.. autofunction:: configure_experiment_logging
.. autofunction:: set_library_log_levels
.. autofunction:: disable_logging
.. autofunction:: enable_logging
