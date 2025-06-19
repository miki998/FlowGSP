"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import *
from flowgsp.operators import Adjacency, Laplacian

class Graph:
    """
    A class representing a directed graph using NetworkX.
    This class allows for the creation of a graph from an adjacency matrix,
    adding nodes and edges, drawing the graph, and setting an operator for spectral analysis.
    It serves as a base class for more specialized graph types.
    """
    def __init__(self, G=None, adj_matrix=None, pos=None):
        if G is not None:
            self.G = G
            self.adj_matrix = nx.to_numpy_array(G)
            self.pos = pos if pos is not None else nx.kamada_kawai_layout(G)

        elif adj_matrix is not None:
            self.from_adjacency_matrix(adj_matrix)
            self.adj_matrix = adj_matrix
        else:
            raise ValueError("Either a graph (G) or an adjacency matrix (adj_matrix) must be provided." \
            "Careful not to pass both at the same time, as it will raise an error.")
        
        self.N = self.adj_matrix.shape[0] if self.adj_matrix is not None else 0
        self.name = None
        self.operator = None  # Placeholder for the operator associated with the graph

    def from_adjacency_matrix(self, adj_matrix):
        adj_matrix = np.array(adj_matrix)
        num_nodes = adj_matrix.shape[0]
        self.G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                if adj_matrix[i, j] != 0:
                    self.G.add_edge(i, j, weight=adj_matrix[i, j])

    def add_edge(self, u, v, **attrs):
        self.G.add_edge(u, v, **attrs)

    def add_node(self, n, **attrs):
        self.G.add_node(n, **attrs)
    
    def draw(self, ax:matplotlib.axes.Axes=None, arrow_size:int=10, arrow_width:int=2, **kwds):
        """
        Draw the directed graph using NetworkX's draw function.
        If no axes are provided, a new figure and axes are created.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Separate symmetric (bidirectional) and asymmetric (unidirectional) edges
        edges = list(self.G.edges())
        symmetric_edges = set()
        asymmetric_edges = set()
        for u, v in edges:
            if (v, u) in edges and (v, u) not in symmetric_edges:
                symmetric_edges.add((u, v))
            elif (v, u) not in edges:
                asymmetric_edges.add((u, v))

        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos=self.pos, ax=ax, **kwds)

        # Draw symmetric edges (bidirectional) in one color/style
        nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=list(symmetric_edges), ax=ax, 
                       edge_color='tab:gray', arrows=False)

        # Draw asymmetric edges (unidirectional) in another color/style
        nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=list(asymmetric_edges), ax=ax, 
                       edge_color='tab:red', arrows=True, connectionstyle='arc3,rad=0.0', 
                       arrowsize=arrow_size, width=arrow_width)

        # Draw labels if requested
        if kwds.get("with_labels", False):
            nx.draw_networkx_labels(self.G, pos=self.pos, ax=ax)

    def draw_signal(self, signal:Optional[np.ndarray]=None, cmap:Optional[colors.Colormap]=None, 
               scale:int=100, ax:matplotlib.axes.Axes=None, scolor:Optional[list]=["red", "blue"], 
               colorbar:bool=False, nodetype:bool="size", **kwds):
        """
        Visualize a signal on a directed graph.

        Plots a directed graph with node size and/or color determined by node values.
        Node size is scaled by the 'scale' parameter to be visible.
        Node color is determined by the sign of the node value (positive or negative)
        if a color map is not provided. If a color map is provided, node color 
        is mapped to the normalized node value.

        Parameters
        ----------
        G : networkx.Graph
            Directed graph to plot

        signal : numpy.ndarray
            graph signal, used for size and/or color

        pos : dict, optional
            Node positions for graph layout

        cmap : matplotlib.colors.Colormap, optional
            Color map to use for node colors
        
        scale : float, optional
            Scaling factor for node sizes

        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        
        scolor : list, optional
            Default node colors if cmap not provided

        colorbar : bool, optional
            Whether to draw a colorbar (requires cmap)

        nodetype : str
            - "color" colors is showing the difference between nodes values
            - "size" size of nodes is showing the difference between nodes values

        Returns
        -------
        None
        
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        # Catching case of poor signal input
        if signal is None:
            signal = np.ones(self.N)
        if np.allclose(signal, 0):
            print("Signal is all zeros, plotting graph with default node size and color.")
            signal = np.ones(self.N)

        # Set node colors
        if cmap is None:
            node_color = [scolor[0] if nd > 0 else scolor[1] for nd in signal]
        else:
            normalized_values = signal - signal.min()
            if np.allclose(normalized_values, 0):
                print("Signal is constant, normalizing to avoid division by zero.")
                normalized_values = np.ones_like(signal)
            else:
                normalized_values /= normalized_values.max()
            node_color = [cmap(normalized_values[k]) for k in range(len(normalized_values))]
            
        node_values = scale * np.abs(signal)
        if nodetype == "color":
            nx.draw(self.G,arrows=True,
                    node_color=signal,
                    pos=self.pos,
                    ax=ax,
                    cmap=cmap, 
                    **kwds)
        elif nodetype == "size":
            nx.draw(self.G,arrows=True,
                    node_size=node_values,
                    node_color=node_color,
                    pos=self.pos,
                    ax=ax,
                    cmap=cmap, 
                    **kwds)
        else:
            print("Unsupported input ... plotting nodes with default size and color")

        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            plt.colorbar(sm)

    def set_operator(self, name='adjacency', **kwargs):
        """
        Returns the operator associated with the graph.
        """
        if self.G is None:
            raise ValueError("Graph is not initialized. Please provide a valid graph.")
        if self.adj_matrix is None:
            raise ValueError("Adjacency matrix is not initialized. Please provide a valid adjacency matrix or a Graph.")
        
        self.name = name
        if name == 'adjacency':
            self.operator = Adjacency(self, **kwargs)
        elif name == 'laplacian':
            self.operator = Laplacian(self, **kwargs)
        elif name == 'advection_diffusion':
            self.operator = AdvectionDiffusion(self, **kwargs)
        else:
            raise ValueError(f"Unknown operator name: {name} \
                             (must be one of ['adjacency', 'laplacian', 'advection_diffusion'])")

    def __repr__(self):
        return f"<Current Operator(name={self.name}, num_nodes={self.G.number_of_nodes()}, num_edges={self.G.number_of_edges()})>"