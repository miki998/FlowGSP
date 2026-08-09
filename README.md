<h1 align="center">GyRAPH</h1>

<p align="center"><strong>A Python framework for signal processing on directed graphs.</strong></p>

<p align="center">
<a href="https://github.com/miki998/GyRAPH/actions"><img src="https://github.com/miki998/GyRAPH/workflows/Launch%20Unittest/badge.svg" alt="Tests"></a>
<img src="./coverage.svg" alt="Coverage">
<a href="https://gyraph.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/gyraph/badge/?version=latest" alt="Documentation"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"></a>
</p>

---

## Overview

**GyRAPH** provides graph shift operators for asymmetric graphs, a complex-valued Graph Fourier
Transform built on their (possibly non-orthogonal) eigenbases, a family of filters
that operate in the resulting spectral domain, and statistical tooling — surrogates
and stationarity tests — for hypothesis testing on directed graph signals.

The library is built on NumPy, SciPy and NetworkX, and is designed to slot into
existing analysis pipelines rather than replace them.

📖 **Documentation: [gyraph.readthedocs.io](https://gyraph.readthedocs.io/)** — tutorials,
full API reference, and the papers behind the methods.

### Key features

| Area | What you get |
| --- | --- |
| **Graph shift operators** | Adjacency, directed Laplacian, advection–diffusion, and time–vertex (joint temporal/spatial) operators — each with normalization options and an automatic Jordan-block perturbation fallback for non-diagonalizable matrices. |
| **Directed Graph Fourier Transform** | Forward and inverse GFT over complex eigenbases, conjugate-frequency pairing, and frequency ordering by eigenvalue magnitude. |
| **Filters** | Spectral, polynomial (and dual-polynomial), Chebyshev, Faber, Hilbert, Tikhonov, and Wiener filters — including approximation schemes that avoid an explicit eigendecomposition. |
| **Surrogates & stationarity** | Phase-randomized and structure-preserving surrogate generation, PSD estimation, and stationarity tests for null-hypothesis testing on graph signals. |
| **Graph construction** | Synthetic generators (cycles, directed tori, asymmetric Erdős–Rényi, vortex and laminar-flow fields) and mesh/surface graphs (sphere, cube, bunny, dragon, hyperbolic paraboloid). |
| **Metrics & visualization** | Dirichlet energy, total variation, Sobolev and directed-variation smoothness measures; publication-styled plotting for graphs, signals, spectra, meshes and dynamics. |

## Installation

```bash
pip install GyRAPH
```

From source, for development:

```bash
git clone https://github.com/miki998/GyRAPH.git && cd GyRAPH && pip install -e .
```

Requires Python 3.9+. Core dependencies (NumPy, SciPy, NetworkX, scikit-learn, pandas,
matplotlib/seaborn/scienceplots, sympy, tqdm) are installed automatically; the mesh and
learning-oriented modules additionally pull in PyTorch, `torch-geometric`, OpenCV and
scikit-image.

## Quickstart

### Building a graph and its Fourier basis

```python
import numpy as np
from gyraph.graphs import Graph, create_directed_torus

# A directed torus: 8 rows x 6 columns, asymmetric by construction
G, pos = create_directed_torus(Nr=8, Nc=6, directed=True)
graph = Graph(G=G, pos=pos)

# Attach a shift operator — this computes the Graph Fourier basis
graph.set_operator("adjacency")            # or "laplacian", "advection_diffusion", ...
op = graph.operator

print(graph)                                # active operator, node and edge counts
print(graph.is_directed(), graph.assymetry_level())
print(op.normality())                       # 0 if the operator is normal, larger otherwise
print(op.V[:5], op.frequencies[:5])         # eigenvalues and graph frequencies
```

`set_operator` accepts `"adjacency"`, `"laplacian"`, `"advection_diffusion"`,
`"time_vertex_laplacian"` and `"time_vertex_adjacency"`, and forwards keyword
arguments to the operator — e.g. `graph.set_operator("laplacian", normalize="symmetric")`.

### Transforming and filtering a signal

```python
from gyraph.filters import SpectralFilter, PolynomialFilter

x = np.random.randn(graph.N)

# Graph Fourier Transform and its inverse
coef = op.GFT(x)
assert np.allclose(op.inverseGFT(coef), x)

# Low-pass in the spectral domain: keep the 10 lowest graph frequencies
kernel = np.zeros(graph.N)
kernel[:10] = 1.0

sfilt = SpectralFilter(graph)
x_low = sfilt.apply(x, kernel)

# Same response, approximated by a degree-K polynomial of the shift operator
# (no eigendecomposition needed at apply time)
pfilt = PolynomialFilter(graph, order=12)
x_low_approx = pfilt.apply(x, kernel)
```

For directed operators the spectrum is complex. `SpectralFilter.transform_in_real`
enforces conjugate symmetry on a kernel so the filtered signal stays real-valued, and
`phase_shift` generalizes the Hilbert transform to arbitrary phase rotations in the GFT
domain.

### Surrogates and stationarity testing

```python
from gyraph.surrogates import Surrogate

surr = Surrogate(graph)

# 200 surrogates preserving the directed spectral structure of x
surrogates = surr.directed_random_surrogate(x, nrands=200, seed=99)

# Test whether a set of realizations is stationary w.r.t. the graph
is_stat = surr.is_stationary(surrogates, eps_diag=0.5, eps_mean=0.5, verbose=True)
```

### Measuring signal smoothness

```python
from gyraph.utils import dirichlet, TV, directed_variation

graph.set_operator("laplacian")
print(dirichlet(x, graph.operator.M))        # x^T L x
print(TV(x, graph.adj_matrix, norm="L1"))    # ||x - Ax||_1
print(directed_variation(x, graph.adj_matrix))
```

## Package layout

```
gyraph/
├── graphs/        Graph container, synthetic generators, mesh & physical-flow graphs
├── operators/     Adjacency, Laplacian, advection–diffusion, time–vertex operators;
│                  Jordan-block and zero-eigenvalue handling
├── filters/       Spectral, polynomial, Chebyshev, Faber, Hilbert, Tikhonov, Wiener
├── surrogates/    Surrogate generation and stationarity/PSD estimation
├── stats/         p-values, circular statistics, complex Gaussian sampling
└── utils/         Numerics, smoothness metrics, plotting, logging configuration
```

Every subpackage is re-exported at the top level, so `import gyraph` gives access to
`gyraph.graphs`, `gyraph.filters`, `gyraph.operators`, `gyraph.surrogates` and
`gyraph.utils`. Importing `gyraph` also applies a publication-oriented matplotlib style
(`science`/`ieee`) — see [`gyraph/constants.py`](gyraph/constants.py) if you would
rather keep your own rcParams.

## Examples and data

Runnable scripts and notebooks live in [`examples/`](examples/) — start with
[`examples/basic/`](examples/basic/) for graph construction, filtering and
visualization, then move to [`examples/advanced/`](examples/advanced/).

The [`data/`](data/) directory ships the graph datasets used throughout the examples
and tests:

- `manhattan_graph_data/` — mid-Manhattan road network with NYC taxi flow signals
- `usa_graph_data/` — US state adjacency graph with boundary shapefiles
- `temperature_bretagne_graph_data/` — Brittany weather-station network

## Development

```bash
pip install -e . && pip install pytest pytest-cov flake8 pre-commit && pre-commit install
```

Run the test suite (154 unit tests, ~84% line coverage):

```bash
python -m unittest discover -s tests/ -p 'test_*.py'
```

With coverage and linting, as CI does:

```bash
coverage run -m unittest discover -s tests/ -p 'test_*.py' && coverage report -m && flake8 . --max-line-length=127
```

Type checking is configured in [`mypy.ini`](mypy.ini). Both workflows —
unit tests with a coverage badge, and packaging — run on every push to `main`
(see [`.github/workflows/`](.github/workflows/)).

Contributions are welcome; please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
branch/commit conventions, NumPy-style docstring requirements and review process, and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening an issue or pull request.

## Research powered by GyRAPH

GyRAPH is the reference implementation behind the following publications — the
derivations, assumptions and validation for the code live there:

- **Graph Diffusion-Advection Operator for Directed Graph Signal Processing** —
  Chan, Cionca, Škultéty, Van De Ville.
  [arXiv:2606.16306](https://arxiv.org/html/2606.16306v1) → `gyraph.operators.AdvectionDiffusion`
- **Graph Signal Surrogate Generation for Statistical Testing of Covariance
  Structure on Directed Graphs** — Chan, Cionca, Van De Ville.
  [arXiv:2608.01766](https://arxiv.org/pdf/2608.01766) → `gyraph.surrogates`
- **Hilbert Transform on Graphs: Let There Be Phase** — Chan, Cionca,
  Van De Ville, *IEEE Signal Processing Letters*.
  [IEEE Xplore](https://ieeexplore.ieee.org/document/11626552) → `gyraph.filters.HilbertFilter`

## Citation

If you use GyRAPH in academic work, please cite it. Machine-readable metadata is in
[CITATION.cff](CITATION.cff):

```bibtex
@software{chan_gyraph,
  author  = {Chan, Chun Hei Michael},
  title   = {{GyRAPH}: Directed Graph Signal Processing Framework},
  url     = {https://github.com/miki998/GyRAPH},
  license = {MIT}
}
```

## Acknowledgment
This project has been partly funded by the Swiss National Science Foundation under Sinergia grant 209470 “Precision mapping of electrical brain network dynamics with application to epilepsy”.

## License

Released under the [MIT License](LICENSE).
Copyright © Chun Hei Michael Chan, [MIP:Lab](https://miplab.epfl.ch/), EPFL.
