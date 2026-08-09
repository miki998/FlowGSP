Surrogates and stationarity
===========================

A statistic computed on a graph signal — a correlation, a smoothness value, a
spectral peak — means nothing on its own. You need a null: *what would this
statistic look like if the signal carried no structure beyond what the graph
already imposes?* That null is a surrogate.

GyRAPH generates surrogates by randomising the **phase** of the graph Fourier
coefficients while preserving their magnitude, so the power spectral density
with respect to the graph is left intact and everything else is destroyed.

Generating surrogates
---------------------

.. code-block:: python

   import numpy as np
   from gyraph.graphs import Graph, create_directed_torus
   from gyraph.surrogates import Surrogate

   G, pos = create_directed_torus(Nr=8, Nc=6, directed=True)
   graph = Graph(G=G, pos=pos)
   graph.set_operator("adjacency")

   x = np.random.randn(graph.N)

   surr = Surrogate(graph)
   surrogates = surr.directed_random_surrogate(x, nrands=200, seed=99)

Three schemes are available, in increasing order of how much graph structure
they respect:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Method
     - Preserves
   * - :meth:`~gyraph.surrogates.Surrogate.naive_random_surrogate`
     - the values, permuted across nodes — no spectral structure at all
   * - :meth:`~gyraph.surrogates.Surrogate.undirected_random_surrogate`
     - the spectrum of the symmetrised graph (sign flips on a real basis)
   * - :meth:`~gyraph.surrogates.Surrogate.directed_random_surrogate`
     - the complex spectrum of the directed graph, with conjugate pairs kept
       consistent so the surrogates stay real

Testing a statistic
-------------------

Compute your statistic on the data and on every surrogate, then compare:

.. code-block:: python

   from gyraph.utils import dirichlet, p_value

   graph.set_operator("laplacian")
   L = graph.operator.M

   observed = dirichlet(x, L)
   null = np.array([dirichlet(s, L) for s in surrogates])

   p = p_value(null, observed, two_tail=True)

Stationarity
------------

Phase randomisation is only a valid null if the signal is **stationary with
respect to the graph** — that is, its covariance is diagonalised by the graph
Fourier basis. :class:`~gyraph.surrogates.Stationary` (which
:class:`~gyraph.surrogates.Surrogate` inherits from) lets you check that
assumption rather than take it on faith.

.. code-block:: python

   from gyraph.surrogates import Stationary

   stat = Stationary(graph)

   covar = stat.estimate_covariance(samples)     # samples: (n_realizations, N)
   psd = stat.estimate_psd(covar)                # covariance in the GFT domain

   is_stat = stat.is_stationary(samples, eps_diag=0.5, eps_mean=0.5, verbose=True)
   level = stat.stationary_level(samples)        # continuous version of the same test

The test asks how much energy sits off the diagonal of the covariance matrix
once expressed in the graph Fourier basis: for a stationary process, that
matrix is diagonal and the off-diagonal mass is noise.

Generating stationary processes
-------------------------------

For simulations and for calibrating a test, you often want signals with a
*known* spectral profile:

.. code-block:: python

   noise = stat.white_noise_generator(nb_repeat=500, seed=0)
   samples = stat.psd_realization_generator(psd, nb_repeat=500, seed=0)

   # A vector-autoregressive process spreading over the directed graph
   dynamics = stat.var_generator(
       A=graph.adj_matrix,
       active_nodes=[0, 5],
       amplitude_nodes=[1.0, 0.8],
       time_nodes=[0, 10],
       n_iter=100,
       seed=0,
   )

Localisation and translation
----------------------------

Stationarity also gives meaning to translating a kernel *to a node*, the graph
analogue of shifting a filter in time:

.. code-block:: python

   T_i = stat.translation_operator(kernel, i=12)   # kernel translated to node 12
   L_i = stat.localization_operator(kernel, i=12)

Circular statistics
-------------------

Because directed graph Fourier coefficients are complex, phases are a natural
thing to summarise — and phases are circular, so ordinary means do not apply.
:mod:`gyraph.stats` provides the right tools:

.. code-block:: python

   from gyraph.utils import circular_stats, circular_correlation

   mean, var = circular_stats(angles_deg)
   rho = circular_correlation(angles_deg, other_angles_deg)
