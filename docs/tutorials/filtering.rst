Filtering signals
=================

All filters take the graph as their first argument and read the Fourier basis
from ``graph.operator``, so call
:meth:`~gyraph.graphs.Graph.set_operator` first.

.. code-block:: python

   import numpy as np
   from gyraph.graphs import Graph, create_directed_torus

   G, pos = create_directed_torus(Nr=8, Nc=6, directed=True)
   graph = Graph(G=G, pos=pos)
   graph.set_operator("adjacency")

   x = np.random.randn(graph.N)

Exact spectral filtering
------------------------

:class:`~gyraph.filters.SpectralFilter` applies a kernel directly in the GFT
domain: transform, multiply, transform back. It is exact, and it costs one
eigendecomposition up front.

.. code-block:: python

   from gyraph.filters import SpectralFilter

   kernel = np.zeros(graph.N)
   kernel[:10] = 1.0                      # keep the 10 lowest graph frequencies

   sfilt = SpectralFilter(graph)
   x_low = sfilt.apply(x, kernel)

Each operator can hand you standard kernels instead of you writing them out:

.. code-block:: python

   op = graph.operator
   kernel = op.low_pass_kernel(limfreq=10)
   kernel = op.high_pass_kernel(limfreq=10)
   kernel = graph.operator.heat_kernel(alpha=0.1)   # Laplacian / advection-diffusion

Keeping the output real
-----------------------

On a directed graph the spectrum is complex, so an arbitrary kernel maps a real
signal to a complex one. :meth:`~gyraph.filters.SpectralFilter.transform_in_real`
symmetrises a kernel over conjugate harmonic pairs so that the filtered signal
stays real:

.. code-block:: python

   kernel_real = sfilt.transform_in_real(kernel)
   x_low = sfilt.apply(x, kernel_real)
   assert np.allclose(x_low.imag, 0, atol=1e-10)

The same machinery generalises the Hilbert transform: a rotation by an
arbitrary phase in the GFT domain.

.. code-block:: python

   x_shifted = sfilt.phase_shift(np.pi / 4, x)

Polynomial approximations
-------------------------

An eigendecomposition is :math:`O(N^3)` and, for a defective operator, may not
exist at all. A polynomial filter sidesteps it: fit the kernel by a degree-``K``
polynomial of the shift operator, then apply that polynomial with ``K``
sparse matrix–vector products.

.. code-block:: python

   from gyraph.filters import PolynomialFilter

   pfilt = PolynomialFilter(graph, order=12)
   x_low_approx = pfilt.apply(x, kernel)

   # Get the fitted coefficients too
   x_low_approx, coefs = pfilt.apply(x, kernel, return_coefs=True)

Fitting on the raw Vandermonde matrix becomes ill-conditioned quickly. Two
better-conditioned bases are provided and are drop-in replacements:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Basis
   * - :class:`~gyraph.filters.PolynomialFilter`
     - monomials :math:`M^k` — simple, conditioning degrades with the order
   * - :class:`~gyraph.filters.ChebyshevFilter`
     - Chebyshev polynomials on a disc covering the spectrum
   * - :class:`~gyraph.filters.FaberFilter`
     - Faber polynomials on an ellipse circumscribing the complex spectrum

.. code-block:: python

   from gyraph.filters import ChebyshevFilter, FaberFilter

   cfilt = ChebyshevFilter(graph, order=20)
   ffilt = FaberFilter(graph, order=20)

   x_cheb = cfilt.apply(x, kernel)
   x_faber = ffilt.apply(x, kernel)

When a single polynomial in one operator is not expressive enough, the dual
variants (:class:`~gyraph.filters.DualPolynomialFilter`,
:class:`~gyraph.filters.DualChebyshevFilter`) fit a joint polynomial over
advection and diffusion parts, selected by ``filter_type`` (``"GA"``, ``"GD"``,
``"GAGD"``, ``"GQAD"``, ``"GQDA"``).

If a least-squares fit of the kernel is not what you want — for instance when
you have input/target pairs rather than a target frequency response — the
coefficients can also be optimised directly in the vertex domain:

.. code-block:: python

   coefs = pfilt.regression_descent(x, target, deg=12, n_iter=500, lr=1e-2)

.. note::

   ``regression_descent`` uses PyTorch. It is the only part of the filtering
   stack that does.

Denoising: Wiener and Tikhonov
------------------------------

Both are statistical filters: they need a model of the signal and of the noise
in the spectral domain.

:class:`~gyraph.filters.WienerFilter` is the minimum-mean-square-error solution
given the signal and noise power spectral densities:

.. code-block:: python

   from gyraph.filters import WienerFilter

   wfilt = WienerFilter(graph)
   x_hat = wfilt.apply_wiener(y, kernel_h=kernel, x_psd=x_psd, noise_psd=noise_psd)

``apply_wiener_AD`` is the variant for advection–diffusion operators, where
the two noise sources (transport and diffusion) get their own PSDs.

:class:`~gyraph.filters.TikhonovFilter` instead regularises with a smoothness
prior and a single trade-off parameter ``lbd``:

.. code-block:: python

   from gyraph.filters import TikhonovFilter

   tfilt = TikhonovFilter(graph)
   x_hat = tfilt.apply_tikhonov(y, noise_covariance=Sigma, lbd=0.1, prior="radial")

Phase and analytic signals
--------------------------

:class:`~gyraph.filters.HilbertFilter` lifts the Hilbert transform to graphs,
which gives you an analytic signal, an envelope, and a notion of instantaneous
frequency on the vertex set:

.. code-block:: python

   from gyraph.filters import HilbertFilter

   hfilt = HilbertFilter(graph)

   x_h = hfilt.hilbert_transform(x)
   x_a = hfilt.analytical_signal(x)          # x + i * H(x)

   envelope = np.abs(x_a)
   freq = hfilt.graph_instant_frequency(x)   # generalised instantaneous frequency

Estimating an unknown filter
----------------------------

Given an input and its filtered output, you can read the transfer function back
out:

.. code-block:: python

   h_hat = sfilt.estimate_transfer_function(x, x_low)
