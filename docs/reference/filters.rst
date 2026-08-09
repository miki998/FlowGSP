Filters
=======

.. currentmodule:: gyraph.filters

Filters take a :class:`gyraph.graphs.Graph` whose operator has already been
set, and read the Fourier basis from ``graph.operator``.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Class
     - Use it when
   * - :class:`SpectralFilter`
     - you want the exact response and can afford an eigendecomposition
   * - :class:`PolynomialFilter`
     - you want to avoid the eigendecomposition at apply time
   * - :class:`ChebyshevFilter`, :class:`FaberFilter`
     - the polynomial fit is ill-conditioned at the order you need
   * - :class:`WienerFilter`, :class:`TikhonovFilter`
     - you are denoising and have a model of signal and noise
   * - :class:`HilbertFilter`
     - you need phase, envelope or instantaneous frequency

Base classes
------------

.. autoclass:: Filter
   :members:
   :show-inheritance:

.. autoclass:: GraphFilter
   :members:
   :show-inheritance:

Spectral filtering
------------------

.. autoclass:: SpectralFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HilbertFilter
   :members:
   :undoc-members:
   :show-inheritance:

Polynomial approximations
-------------------------

.. autoclass:: PolynomialFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ChebyshevFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: FaberFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DualPolynomialFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: DualChebyshevFilter
   :members:
   :undoc-members:
   :show-inheritance:

Statistical filtering
---------------------

.. autoclass:: WienerFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: TikhonovFilter
   :members:
   :undoc-members:
   :show-inheritance:
