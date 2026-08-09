# GyRAPH Examples

This directory contains example scripts and notebooks demonstrating how to use the GyRAPH framework.

## Directory Structure

### `basic/`
Contains introductory examples for getting started with GyRAPH:
- Basic graph creation and manipulation
- Simple filtering operations
- Graph visualization
- Fundamental signal processing tasks

**Start here if you're new to GyRAPH!**

### `advanced/`
Advanced examples showcasing sophisticated techniques:
- Hilbert transform on directed graphs
- Surrogate signal generation and stationarity analysis

## Quick Start

To run any example:

```bash
# For Python scripts
python examples/basic/01_basic_graph_creation.py

# For Jupyter notebooks
jupyter notebook examples/basic/basics.ipynb
jupyter notebook examples/advanced/advanced.ipynb
```

## Further Reading

Longer-form material lives on the documentation site:
- [Tutorials](https://gyraph.readthedocs.io/en/latest/tutorials/index.html) - Step-by-step walkthroughs
- [API Reference](https://gyraph.readthedocs.io/en/latest/reference/index.html) - Full module and function documentation

## Requirements

All examples require the GyRAPH package to be installed:

```bash
pip install -e .
```

Some advanced examples may require additional dependencies specified in the example file.

## Contributing Examples

We welcome contributions of new examples! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

Good examples:
- Are well-documented with clear explanations
- Include both code and expected output
- Use sample data or generate synthetic data
- Are self-contained and easy to run
- Follow the project's code style
