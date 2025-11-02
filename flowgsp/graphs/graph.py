"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

from flowgsp.utils import np, nx, matplotlib, plt, colors, hermitian, Optional, cm
from flowgsp.operators import Adjacency, Laplacian

class Graph:
    """
    A class representing a directed graph using NetworkX.
    This class allows for the creation of a graph from an adjacency matrix,
    adding nodes and edges, drawing the graph, and setting an operator for spectral analysis.
    It serves as a base class for more specialized graph types.
    """
    def __init__(self, G=None, adj_matrix=None, pos=None, is_directed=None):
        if G is not None:
            self.G = G
            self.adj_matrix = nx.to_numpy_array(G)
            self.pos = pos if pos is not None else nx.kamada_kawai_layout(G)

        elif adj_matrix is not None:
            if is_directed is not None:
                if is_directed:
                    self.G = nx.DiGraph()
                else:
                    self.G = nx.Graph()
            else:
                if np.allclose(adj_matrix, hermitian(adj_matrix)):
                    self.G = nx.Graph()
                else:
                    self.G = nx.DiGraph()
            self.from_adjacency_matrix(adj_matrix)
            self.adj_matrix = adj_matrix
            self.pos = pos if pos is not None else nx.kamada_kawai_layout(self.G)

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
            if isinstance(self.G, nx.DiGraph):
                for j in range(num_nodes):
                    if adj_matrix[i, j] != 0:
                        self.G.add_edge(i, j, weight=adj_matrix[i, j])
            else:
                for j in range(i, num_nodes):
                    if adj_matrix[i, j] != 0:
                        self.G.add_edge(i, j, weight=adj_matrix[i, j])

    def add_edge(self, u, v, **attrs):
        self.G.add_edge(u, v, **attrs)

    def add_node(self, n, **attrs):
        self.G.add_node(n, **attrs)
    
    # Set the operator for spectral analysis
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
        else:
            raise ValueError(f"Unknown operator name: {name} \
                             (must be one of ['adjacency', 'laplacian'])")

    # Draw methods
    def draw(self, axes:matplotlib.axes.Axes=None, arrow_size:int=10, arrow_width:int=2, 
             symmetric_color='tab:gray', asymmetric_color='tab:red', edge_alpha=None,
             **kwds):
        """
        Draw the directed graph using NetworkX's draw function.
        If no axes are provided, a new figure and axes are created.
        """
        if axes is None:
            fig, axes = plt.subplots(figsize=(10, 10))
        
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
        nx.draw_networkx_nodes(self.G, pos=self.pos, ax=axes, **kwds)

        # Draw symmetric edges (bidirectional) in one color/style
        nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=list(symmetric_edges), ax=axes, 
                       edge_color=symmetric_color, arrows=False, alpha=edge_alpha)

        # Draw asymmetric edges (unidirectional) in another color/style
        nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=list(asymmetric_edges), ax=axes, 
                       edge_color=asymmetric_color, arrows=True, connectionstyle='arc3,rad=0.0', 
                       arrowsize=arrow_size, width=arrow_width, alpha=edge_alpha)

        # Draw labels if requested
        if kwds.get("with_labels", False):
            nx.draw_networkx_labels(self.G, pos=self.pos, ax=axes)

    def draw_signal(self, signal:Optional[np.ndarray]=None, cmap:Optional[colors.Colormap]=None, 
               scale:int=100, axes:matplotlib.axes.Axes=None, scolor:Optional[list]=["red", "blue"], 
               colorbar:bool=False, nodetype:bool="size", arrow_size:int=10, arrow_width:int=2, 
               un_arrow_width:int=2, 
               symmetric_color='tab:gray', asymmetric_color='tab:red',
               **kwds):
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
        if axes is None:
            fig, axes = plt.subplots(figsize=(10, 10))

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
            if isinstance(cmap, str):
                cmap = cm.get_cmap('viridis')

            normalized_values = signal - signal.min()
            if np.allclose(normalized_values, 0):
                print("Signal is constant, normalizing to avoid division by zero.")
                normalized_values = np.ones_like(signal)
            else:
                normalized_values /= normalized_values.max()
            node_color = [cmap(normalized_values[k]) for k in range(len(normalized_values))]
            
        # Separate symmetric (bidirectional) and asymmetric (unidirectional) edges
        edges = list(self.G.edges())
        symmetric_edges = set()
        asymmetric_edges = set()
        for u, v in edges:
            if (v, u) in edges and (v, u) not in symmetric_edges:
                symmetric_edges.add((u, v))
            elif (v, u) not in edges:
                asymmetric_edges.add((u, v))

        node_values = scale * np.abs(signal)
        if nodetype == "color":
            # Draw nodes
            nx.draw_networkx_nodes(self.G, pos=self.pos, node_color=signal, cmap=cmap, ax=axes, **kwds)
            
        elif nodetype == "size":
            # Draw nodes
            nx.draw_networkx_nodes(self.G, pos=self.pos, node_size=node_values, 
                                   node_color=node_color, cmap=cmap, ax=axes, **kwds)

        else:
            print("Unsupported input ... plotting nodes with default size and color")

        # Draw symmetric edges (bidirectional) in one color/style
        nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=list(symmetric_edges), ax=axes, 
                    edge_color=symmetric_color, arrows=False, width=un_arrow_width)
        # Draw asymmetric edges (unidirectional) in another color/style
        nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=list(asymmetric_edges), ax=axes, 
                    edge_color=asymmetric_color, arrows=True, connectionstyle='arc3,rad=0.0', 
                    arrowsize=arrow_size, width=arrow_width)
        
        if colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            plt.colorbar(sm)

    # Graph Theoric Properties
    def is_directed(self):
        """
        Check if the graph is directed.
        """
        return isinstance(self.G, nx.DiGraph) and (self.assymetry_level() > 0)
    
    def assymetry_level(self, return_number=False, verbose=False):
        """
        Calculate the assymetry of a graph represented by its adjacency matrix A.
        Assymetry is defined as the ratio of the number of asymmetric edges to the total number of edges.
        An edge (i, j) is asymmetric if A[i, j] != A[j, i].
        """
        nb_symmetric = np.sum((self.adj_matrix + self.adj_matrix.T) == 2) // 2 # divide by 2 because we consider 1 edge even though its 2 entries 
        nb_assymetric = np.sum((self.adj_matrix + self.adj_matrix.T) == 1) // 2 # this is re-checked
        if verbose:
            print(f"Number of symmetric edges: {nb_symmetric}, Number of asymmetric edges: {nb_assymetric}")
        if return_number:
            return nb_assymetric / (nb_symmetric + nb_assymetric) if (nb_symmetric + nb_assymetric) > 0 else 0, (nb_symmetric, nb_assymetric)
        return nb_assymetric / (nb_symmetric + nb_assymetric) if (nb_symmetric + nb_assymetric) > 0 else 0

    def degree_entropy(self, degree_type='in'):
        """
        Calculate the indegree entropy of the in-degree and out-degree distributions of a graph.
        """
        if degree_type not in ['in', 'out']:
            raise ValueError("degree_type must be either 'in' or 'out'")
        
        # Calculate the entropy
        if degree_type == 'in':
            in_degree = np.sum(self.adj_matrix, axis=1)
            p = in_degree / np.sum(in_degree) if np.sum(in_degree) > 0 else np.zeros_like(in_degree)
        else:
            out_degree = np.sum(self.adj_matrix, axis=0)
            p = out_degree / np.sum(out_degree) if np.sum(out_degree) > 0 else np.zeros_like(out_degree)

        return -np.sum(p * np.log2(p + 1e-10))  # Adding a small constant to avoid log(0)
        
    def ratio_entropy(self):
        """ 
        Calculate the ratio entropy of the graph.
        The ratio entropy is defined as the entropy of the in-degree divided by the out-degree.
        It is a measure of the balance between incoming and outgoing connections in the graph.
        """
        in_degree = np.sum(self.adj_matrix, axis=1)
        out_degree = np.sum(self.adj_matrix, axis=0)

        out_degree[out_degree == 0] = -1  # Avoid division by zero
        ratio = in_degree / out_degree  # Avoid division by zero
        ratio[ratio < 0] = 1
        ratio = ratio / np.sum(ratio) if np.sum(ratio) > 0 else np.zeros_like(ratio)

        entropy = -np.sum(ratio * np.log2(ratio + 1e-10))
        return entropy

    def __repr__(self):
        return f"<Current Operator(name={self.name}, num_nodes={self.G.number_of_nodes()}, num_edges={self.G.number_of_edges()})>"