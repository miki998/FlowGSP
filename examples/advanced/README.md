# Advanced Examples

This directory contains advanced examples demonstrating sophisticated techniques in directed graph signal processing with GyRAPH.

## Examples Overview

### `advanced.ipynb`

Builds on the basics notebook and walks through two of the more involved parts of the library, both using the adjacency operator on directed cycle graphs:

- **Hilbert transform** - Applying `HilbertFilter` to a graph signal, comparing the original and filtered signals both as time series and drawn on the graph, and inspecting the filter itself alongside the amplitude of the GFT coefficients before and after filtering.
- **Surrogates** - Generating directed random surrogates from an initial signal with the `Surrogate` class, then estimating the covariance and power spectral density of the realizations and reporting the stationarity level. Two graphs are used, so you can compare a random signal against a Dirac signal.

## Running Examples

```bash
jupyter notebook advanced.ipynb
```

## Prerequisites

Work through the [basic examples](../basic/) first — the notebook assumes you are already comfortable instantiating graphs, setting operators, and using filters.

## Additional Resources

- [Documentation](https://gyraph.readthedocs.io/)
- [Tutorials](https://gyraph.readthedocs.io/en/latest/tutorials/index.html)
- [API Reference](https://gyraph.readthedocs.io/en/latest/reference/index.html)