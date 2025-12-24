#!/usr/bin/env python3
"""
Basic Graph Creation Example
=============================

This example demonstrates how to create and manipulate directed graphs
using the FlowGSP framework.

Topics covered:
- Creating graphs from adjacency matrices
- Graph properties and attributes
- Setting graph operators
- Basic graph visualization

"""

import numpy as np
import matplotlib.pyplot as plt


def create_simple_graph():
    """Create a simple directed graph."""
    # Note: FlowGSP import will work after package installation
    # For demonstration, we show the expected usage
    
    print("=" * 60)
    print("Example 1: Creating a Simple Directed Graph")
    print("=" * 60)
    
    # Create a simple directed graph with 4 nodes
    # Adjacency matrix where A[i,j] = 1 means edge from i to j
    A = np.array([
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    print("\nAdjacency Matrix:")
    print(A)
    print(f"\nNumber of nodes: {A.shape[0]}")
    print(f"Number of edges: {np.sum(A)}")
    
    # Uncomment when package is installed:
    # import flowgsp
    # graph = flowgsp.graphs.Graph(adj_matrix=A)
    # print(f"\nGraph created with {graph.N} nodes and {graph.ne} edges")
    # print(f"Graph is directed: {graph.is_directed()}")
    
    return A


def create_weighted_graph():
    """Create a weighted directed graph."""
    print("\n" + "=" * 60)
    print("Example 2: Creating a Weighted Directed Graph")
    print("=" * 60)
    
    # Create a weighted graph
    A_weighted = np.array([
        [0.0, 0.5, 0.8, 0.0],
        [0.0, 0.0, 0.3, 0.9],
        [0.0, 0.0, 0.0, 0.4],
        [0.7, 0.0, 0.0, 0.0]
    ])
    
    print("\nWeighted Adjacency Matrix:")
    print(A_weighted)
    
    # Uncomment when package is installed:
    # import flowgsp
    # graph = flowgsp.graphs.Graph(adj_matrix=A_weighted)
    # print(f"\nWeighted graph created")
    # print(f"Edge weights range: [{np.min(A_weighted[A_weighted>0]):.2f}, "
    #       f"{np.max(A_weighted):.2f}]")
    
    return A_weighted


def demonstrate_operators():
    """Demonstrate different graph operators."""
    print("\n" + "=" * 60)
    print("Example 3: Graph Operators")
    print("=" * 60)
    
    # Create a simple graph
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ])
    
    print("\nGraph operators transform graph structure for signal processing.")
    print("\nAvailable operators:")
    print("- 'adjacency': Uses adjacency matrix directly")
    print("- 'laplacian': Graph Laplacian (L = D - A)")

def visualize_graph():
    """Visualize a directed graph."""
    print("\n" + "=" * 60)
    print("Example 4: Graph Visualization")
    print("=" * 60)
    
    # Create a small directed graph for visualization
    A = np.array([
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0]
    ])
    
    print("\nCreating a directed cycle graph with 5 nodes...")
    
    # Uncomment when package is installed:
    # import flowgsp
    # import networkx as nx
    # 
    # # Create graph
    # graph = flowgsp.graphs.Graph(adj_matrix=A)
    # 
    # # Convert to networkx for visualization
    # G = nx.DiGraph(A)
    # 
    # # Plot
    # plt.figure(figsize=(8, 6))
    # pos = nx.circular_layout(G)
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', 
    #         node_size=500, font_size=16, font_weight='bold',
    #         arrows=True, arrowsize=20, arrowstyle='->', 
    #         edge_color='gray', width=2)
    # plt.title("Directed Cycle Graph", fontsize=14, fontweight='bold')
    # plt.tight_layout()
    # plt.savefig('basic_graph_visualization.png', dpi=150, bbox_inches='tight')
    # print("Visualization saved as 'basic_graph_visualization.png'")
    # plt.close()


def graph_properties():
    """Explore graph properties."""
    print("\n" + "=" * 60)
    print("Example 5: Graph Properties")
    print("=" * 60)
    
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    
    print("\nExploring graph properties...")
    
    # Basic properties
    n_nodes = A.shape[0]
    n_edges = np.sum(A)
    in_degrees = np.sum(A, axis=0)
    out_degrees = np.sum(A, axis=1)
    
    print(f"Number of nodes: {n_nodes}")
    print(f"Number of edges: {n_edges}")
    print(f"In-degrees: {in_degrees}")
    print(f"Out-degrees: {out_degrees}")
    print(f"Average degree: {n_edges / n_nodes:.2f}")
    
    # Uncomment when package is installed:
    # import flowgsp
    # 
    # graph = flowgsp.graphs.Graph(adj_matrix=A)
    # graph.compute_fourier_basis()
    # 
    # print(f"\nSpectral properties:")
    # print(f"Number of eigenvalues: {len(graph.e)}")
    # print(f"Eigenvalue range: [{np.min(graph.e):.4f}, {np.max(graph.e):.4f}]")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 60)
    print(" FlowGSP - Basic Graph Creation Examples")
    print("*" * 60)
    print("\nThese examples demonstrate basic graph creation and manipulation.")
    print("Uncomment the FlowGSP import statements after installing the package.")
    print()
    
    # Run examples
    create_simple_graph()
    create_weighted_graph()
    demonstrate_operators()
    visualize_graph()
    graph_properties()
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Install FlowGSP from repository root: cd ../.. && pip install -e .")
    print("2. Uncomment the FlowGSP code in this file")
    print("3. Run this script again to see the full functionality")
    print("4. Explore more examples in the examples/ directory")
    print()


if __name__ == "__main__":
    main()
