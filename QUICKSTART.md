# FlowGSP Quick Start Guide

Welcome to FlowGSP! This guide will help you get started with directed graph signal processing in just a few minutes.

## Installation

### Step 1: Install the Package

```bash
# Clone the repository
git clone https://github.com/miki998/digraph_GSP_framework.git
cd digraph_GSP_framework

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install FlowGSP
pip install -e .
```

Note: Installation may take several minutes as it downloads PyTorch and other dependencies.

## Your First Graph Signal Processing

### Example 1: Create a Directed Graph

```python
import numpy as np
import flowgsp

# Create a simple directed cycle graph
A = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
])

# Create graph object
graph = flowgsp.graphs.Graph(adj_matrix=A)
print(f"Created graph with {graph.N} nodes")
```

### Example 2: Apply a Filter

```python
import numpy as np
import flowgsp
from flowgsp.filters import SpectralFilter

# Create a graph
A = np.random.rand(10, 10)
A = (A > 0.7).astype(float)  # Sparse directed graph
graph = flowgsp.graphs.Graph(adj_matrix=A)
graph.set_operator('laplacian')

# Create a random signal on the graph
signal = np.random.randn(graph.N)

# Apply a low-pass filter (adjust based on actual API)
spectre = SpectralFilter(graph)
kernel = np.ones(graph.N)
filtered = spectre(signal, kernel)
```

### Example 3: Graph Visualization

```python
import numpy as np
import flowgsp
import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
A = np.array([
    [0, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0]
])
graph = flowgsp.graphs.Graph(adj_matrix=A)

# Convert to NetworkX for visualization
G = nx.DiGraph(A)
pos = nx.spring_layout(G)

# Plot
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=800, font_size=16, arrows=True,
        arrowsize=20, edge_color='gray', width=2)
plt.title('Directed Graph')
plt.show()
```

## Key Concepts

### 1. Graph Creation

```python
# From adjacency matrix
graph = flowgsp.graphs.Graph(adj_matrix=A)

# From edge list
# graph = flowgsp.graphs.Graph(edge_list=edges)
```

### 2. Graph Operators

Different operators capture different aspects of graph structure:

```python
# Laplacian (measures curvature)
graph.set_operator('laplacian')

# Adjacency (direct connections)
graph.set_operator('adjacency')
```

## Learning Path

### For Beginners

1. **Start Here**: Run `examples/basic/01_basic_graph_creation.py`
2. **Learn the Basics**: Read through `tutorials/basics.ipynb`
3. **Try Examples**: Explore other files in `examples/basic/`
4. **Read Docs**: Check out `docs/functionality.md`

### For Researchers

1. **Review Examples**: Browse `examples/advanced/`
2. **Check Applications**: See `examples/applications/`
3. **Explore Notebooks**: Dive into `notebooks/` for research code
4. **Read Papers**: Check citations in code for theoretical background

## Troubleshooting

### Import Errors

```python
# If you get: ModuleNotFoundError: No module named 'flowgsp'
# Make sure you installed the package:
pip install -e .
```

## Next Steps

### Explore More

- **Examples**: `examples/` - Organized by difficulty and application
- **Tutorials**: `tutorials/` - Step-by-step guides
- **Documentation**: `docs/functionality.md` - Comprehensive reference
- **Research**: `notebooks/` - Advanced research implementations

### Get Help

- **Examples**: Look for similar use cases in `examples/`
- **Issues**: Open an issue on GitHub
- **Email**: Contact miki998chan@gmail.com

### Contribute

- **Read**: `CONTRIBUTING.md` for guidelines
- **Share**: Your examples and use cases
- **Improve**: Documentation and code
- **Report**: Bugs and suggestions

## Useful Commands

```bash
# Run an example
python examples/basic/01_basic_graph_creation.py

# Run tests
python -m unittest discover -s tests/

# Run a Jupyter notebook
jupyter notebook tutorials/basics.ipynb
```

## Citation

If you use FlowGSP in your research:

```bibtex
@software{chan2025flowgsp,
  author = {Chan, Chun Hei Michael},
  title = {FlowGSP: Directed Graph Signal Processing Framework},
  year = {2026},
  url = {https://github.com/miki998/FlowGSP}
}
```

---

**Happy Graph Signal Processing! 🎯📊**

For more information, see the full [README.md](README.md).