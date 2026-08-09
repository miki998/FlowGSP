Research powered by GyRAPH
==========================

GyRAPH is the reference implementation behind the following publications. The
methods described in each paper map directly onto a module of the library, so
these are the place to look for the derivations, the assumptions and the
validation behind the code.

Graph Diffusion-Advection Operator for Directed Graph Signal Processing
-----------------------------------------------------------------------

*Chun Hei Michael Chan, Alexandre Cionca, Viktor Škultéty, Dimitri Van De Ville*

Connects directed graph Laplacians to diffusion–advection operators from
physics, and derives the resulting frequency orderings and smoothness measures
— radial (diffusion) and angular (advection) — that make flexible filter design
possible on asymmetric graphs.

→ :class:`gyraph.operators.AdvectionDiffusion`, and the radial/angular kernels
and smoothness measures documented in :doc:`tutorials/smoothness`.

reference: `arXiv:2606.16306 <https://arxiv.org/html/2606.16306v1>`_

Statistical Testing on Directed Graphs by Surrogate Data Generation
----------------------------------------------------------------------------------------------------

*Chun Hei Michael Chan, Alexandre Cionca, Dimitri Van De Ville*

Generates surrogate data that leveraging stationary processes on directed graphs, improving statistical detection of irregular node relationships over conventional approaches for undirected graphs and graph unaware.

→ :class:`gyraph.surrogates.Surrogate` and
:class:`gyraph.surrogates.Stationary`, walked through in
:doc:`tutorials/surrogates`.

reference: `IEEE Xplore <https://ieeexplore.ieee.org/document/11626552>`_

Optimal Wiener-Filter Solutions for Denoising of Graph Signals on Directed Graphs
----------------------------------------------------------------------------------------------------

*Chun Hei Michael Chan, Alexandre Cionca, Dimitri Van De Ville*

We propose a Wiener-filter solution for graph signals on directed graphs. Under various stationarity assumptions combining uncorrelated and correlated noise conditions, we show optimal solutions, including a successful proof-of-concept for temperature graph.

→ :class:`gyraph.filters.WienerFilter` and walked through in
:doc:`tutorials/filtering`.

reference: `arXiv:2606.07876 <https://arxiv.org/pdf/2606.07876>`_

Graph Signal Surrogate Generation for Statistical Testing of Covariance Structure on Directed Graphs
----------------------------------------------------------------------------------------------------

*Chun Hei Michael Chan, Alexandre Cionca, Dimitri Van De Ville*

Generates surrogate data that preserves covariance structure on directed
graphs, improving statistical detection of irregular node relationships over
conventional approaches for undirected graphs and graph unaware.

→ :class:`gyraph.surrogates.Surrogate` and
:class:`gyraph.surrogates.Stationary`, walked through in
:doc:`tutorials/surrogates`.

reference: `arXiv:2608.01766 <https://arxiv.org/pdf/2608.01766>`_

Hilbert Transform on Graphs: Let There Be Phase
-----------------------------------------------

*Chun Hei Michael Chan, Alexandre Cionca, Dimitri Van De Ville* —
IEEE Signal Processing Letters

Lifts the Hilbert transform to graph signals, giving a well-defined analytic
signal, envelope and instantaneous frequency on the vertex set.

→ :class:`gyraph.filters.HilbertFilter` and
:meth:`gyraph.filters.SpectralFilter.phase_shift`.

reference: `IEEE Xplore <https://ieeexplore.ieee.org/document/10962535>`_

.. note::

   Using GyRAPH in your own work? Please cite it — see :doc:`citing`. If your
   paper belongs on this page, open a pull request or an issue.
