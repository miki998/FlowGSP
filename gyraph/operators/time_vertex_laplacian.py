"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""

import gyraph
from gyraph.utils import np, hermitian, nx, TV, warnings
from .base import Operator
from .jordan_destroy import (
    destroy_jordan_blocks_laplacian,
    destroy_jordan_blocks,
    destroy_zero_eigenvals,
)
from typing import Optional, Any


class TimeVertexAdjacency(Operator):
    """
    A class to represent the adjacency operator
    on a graph with time inclusion.
    """

    def __init__(
        self,
        graph: Any,
        nb_time: int = 1,
        name: Optional[str] = None,
        params: Optional[dict] = None,
        normalize: Optional[str] = None,
        partial: bool = False,
    ):
        super().__init__(graph, name=name, params=params)
        if normalize is not None:
            if normalize not in ["right", "left", "symmetric"]:
                raise ValueError(
                    "normalize must be one of ['right', 'left', 'symmetric']"
                )
            self.params["normalize"] = normalize
        if nb_time <= 1:
            raise ValueError("nb_time must be strictly greater than 1")
        self.params["nb_time"] = nb_time
        self.compute_operator()
        if not partial:
            self.compute_basis()

    def compute_operator(self):
        """Compute the adjacency operator for the graph with time inclusion."""
        self.graph.adj_matrix = self.sanitize_operator(self.graph.adj_matrix)
        self.graph.adj_matrix = self.normalize_operator(
            self.graph.adj_matrix, order=self.params["normalize"]
        )
        self.graph_M = self.graph.adj_matrix
        cycle_graph, _ = gyraph.graphs.create_cycle_graph(
            self.params["nb_time"], 1
        )  # Graph type 1 -> cycle
        self.time_M = nx.adjacency_matrix(cycle_graph).toarray()
        self.M = np.kron(self.time_M, self.graph_M)

    def compute_basis(self):
        """
        Compute the basis for the Laplacian operator.
        The basis is computed as the eigenvectors of the Laplacian matrix.
        """
        if self.is_symmetric():
            # M is symmetric iff both time and graph are symmetric
            self.V_T, self.U_T = np.linalg.eigh(self.time_M)
            self.V_G, self.U_G = np.linalg.eigh(self.graph_M)
        else:
            try:
                self.V_T, self.U_T = np.linalg.eig(self.time_M)
                self.V_G, self.U_G = np.linalg.eig(self.graph_M)
            except np.linalg.LinAlgError:
                warnings.warn(
                    "Matrix is not diagonalizable, attempting to destroy Jordan blocks."
                )
                self.graph_M = destroy_jordan_blocks(self.graph_M)
                self.graph_M = destroy_zero_eigenvals(
                    self.graph_M
                )  # Making sure no zero eigenvalues for invertibility
                warnings.warn(
                    "Attention! The Laplacian matrix has been modified to destroy Jordan blocks."
                )
                try:
                    self.V_G, self.U_G = np.linalg.eig(self.graph_M)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError(
                        "Matrix is still not diagonalizable after attempting to destroy Jordan blocks."
                    )

        self.frequencies_time = np.array(
            [
                TV(self.U_T[:, k], self.time_M, norm="L1", lbd_flag=False)
                for k in range(self.params["nb_time"])
            ]
        )
        self.frequencies_space = np.array(
            [
                TV(self.U_G[:, k], self.graph_M, norm="L1", lbd_flag=False)
                for k in range(self.graph.N)
            ]
        )
        # Sort eigenvalues and eigenvectors
        if not np.all(np.abs(self.V_T - 1) < 1e-10):  # If not a perfect cycle
            self.V_T = self.V_T[np.argsort(self.frequencies_time)]
            self.U_T = self.U_T[:, np.argsort(self.frequencies_time)]
            self.frequencies_time = np.sort(
                self.frequencies_time
            )  # Sort frequencies in ascending order

        if not np.all(np.abs(self.V_G - 1) < 1e-10):  # If not a perfect cycle
            self.V_G = self.V_G[np.argsort(self.frequencies_space)]
            self.U_G = self.U_G[:, np.argsort(self.frequencies_space)]
            self.frequencies_space = np.sort(
                self.frequencies_space
            )  # Sort frequencies in ascending order

        # Joining together both basis by Kronecker
        self.U = np.kron(self.U_T, self.U_G)
        self.V = np.diag(np.kron(np.diag(self.V_T), np.diag(self.V_G)))
        self.frequencies = np.diag(
            np.kron(np.diag(self.frequencies_time), np.diag(self.frequencies_space))
        )
        cond_number = np.linalg.cond(self.U)
        if cond_number > 1e3:  # You can adjust this threshold as needed
            if not self.graph.no_print:
                warnings.warn(
                    f"The condition number of U is too high: {int(cond_number)}."
                )

        # Compute inverse Fourier transform
        if self.is_symmetric():
            self.Uinv = hermitian(self.U)
        else:
            self.Uinv = np.linalg.inv(self.U)

        self.imaginaries = np.abs(self.V.imag) >= 1e-8
        self.name = "Time-Vertex-Adjacency"
        self.params["cond_number"] = cond_number

    def sig2vec(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim != 2:
            raise ValueError("Dimensionality of signal is not 2")
        return signal.flatten()

    def vec2sig(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim != 1:
            raise ValueError("Dimensionality of signal is not 1")
        return signal.reshape(self.params["nb_time"], self.graph.N)


class TimeVertexLaplacian(Operator):
    """
    A class to represent the directed / or standard Laplacian
    operator on a graphs with time inclusion.
    #TODO: Make it specific to the real graph Laplacian and move the directed Laplacian
    to the correct file in the future.
    """

    def __init__(
        self,
        graph: Any,
        nb_time: int = 1,
        name: Optional[str] = None,
        params: Optional[dict] = None,
        in_degree: bool = True,
        normalize: Optional[str] = None,
        partial: bool = False,
    ):
        super().__init__(graph, name=name, params=params)
        if normalize is not None:
            if normalize not in ["right", "left", "symmetric"]:
                raise ValueError(
                    "normalize must be one of ['right', 'left', 'symmetric']"
                )
            self.params["normalize"] = normalize
        if nb_time <= 1:
            raise ValueError("nb_time must be strictly greater than 1")
        self.params["nb_time"] = nb_time

        self.compute_operator(in_degree=in_degree)
        if not partial:
            self.compute_basis()

    def compute_operator(self, in_degree: bool = True):
        """Compute the adjacency operator for the graph with time inclusion."""
        self.graph.adj_matrix = self.sanitize_operator(self.graph.adj_matrix)
        self.graph.adj_matrix = self.normalize_operator(
            self.graph.adj_matrix, order=self.params["normalize"]
        )
        self.graph_M = self.compute_directed_laplacian(
            self.graph.adj_matrix, in_degree=in_degree
        )
        cycle_graph, _ = gyraph.graphs.create_cycle_graph(
            self.params["nb_time"], 1
        )  # Graph type 1 -> cycle
        self.time_M = self.compute_directed_laplacian(
            nx.adjacency_matrix(cycle_graph).toarray(), in_degree=in_degree
        )
        self.M = np.kron(self.time_M, self.graph_M)

        self.params["in_degree"] = in_degree

    def compute_basis(self):
        """
        Compute the basis for the Laplacian operator.
        The basis is computed as the eigenvectors of the Laplacian matrix.
        """
        if self.is_symmetric():
            # M is symmetric iff both time and graph are symmetric
            self.V_T, self.U_T = np.linalg.eigh(self.time_M)
            self.V_G, self.U_G = np.linalg.eigh(self.graph_M)
        else:
            try:
                self.V_T, self.U_T = np.linalg.eig(self.time_M)
                self.V_G, self.U_G = np.linalg.eig(self.graph_M)
            except np.linalg.LinAlgError:
                if self.graph.debug:
                    warnings.warn(
                        "Matrix is not diagonalizable, attempting to destroy Jordan blocks."
                    )
                self.graph_M = destroy_jordan_blocks_laplacian(self.graph_M)
                if self.graph.debug:
                    warnings.warn(
                        "Attention! The Laplacian matrix has been modified to destroy Jordan blocks."
                    )
                try:
                    self.V_G, self.U_G = np.linalg.eig(self.graph_M)
                except np.linalg.LinAlgError:
                    raise np.linalg.LinAlgError(
                        "Matrix is still not diagonalizable after attempting to destroy Jordan blocks."
                    )

        self.frequencies_time = np.abs(self.V_T)
        self.frequencies_space = np.abs(self.V_G)
        # Sort eigenvalues and eigenvectors
        if not np.all(np.abs(self.V_T - 1) < 1e-10):  # If not a perfect cycle
            self.V_T = self.V_T[np.argsort(self.frequencies_time)]
            self.U_T = self.U_T[:, np.argsort(self.frequencies_time)]
            self.frequencies_time = np.sort(
                self.frequencies_time
            )  # Sort frequencies in ascending order

        if not np.all(np.abs(self.V_G - 1) < 1e-10):  # If not a perfect cycle
            self.V_G = self.V_G[np.argsort(self.frequencies_space)]
            self.U_G = self.U_G[:, np.argsort(self.frequencies_space)]
            self.frequencies_space = np.sort(
                self.frequencies_space
            )  # Sort frequencies in ascending order

        # Joining together both basis by Kronecker
        self.U = np.kron(self.U_T, self.U_G)
        self.V = np.diag(np.kron(np.diag(self.V_T), np.diag(self.V_G)))
        self.frequencies = np.diag(
            np.kron(np.diag(self.frequencies_time), np.diag(self.frequencies_space))
        )
        cond_number = np.linalg.cond(self.U)
        if cond_number > 1e3:  # You can adjust this threshold as needed
            if self.graph.debug:
                warnings.warn(
                    f"The condition number of U is too high: {int(cond_number)}."
                )

        # Compute inverse Fourier transform
        if self.is_symmetric():
            self.Uinv = hermitian(self.U)
        else:
            self.Uinv = np.linalg.inv(self.U)

        self.imaginaries = np.abs(self.V.imag) >= 1e-8
        self.name = "Time-Vertex-Laplacian"
        self.params["cond_number"] = cond_number

    def compute_directed_laplacian(
        self, A: np.ndarray, in_degree: bool = True
    ) -> np.ndarray:
        """
        Compute the directed Laplacian matrix for a given adjacency matrix A.

        The directed Laplacian is defined as L = D - A, where D is a diagonal matrix containing the in-degree of each node, and A is the adjacency matrix.

        Parameters
        ----------
        A : ndarray
            Adjacency matrix
        in_degree : bool
            Flag to compute in-degree or out-degree

        Returns
        -------
        ret : ndarray
            Directed Laplacian matrix
        """
        if np.any(A.imag != 0):
            raise ValueError("Complex values in laplacian matrix")
        elif np.any(np.diag(A) != 0):
            raise ValueError("Not an Adjacency matrix")

        if in_degree:
            deg = A.sum(axis=1).astype(float)
        else:
            deg = A.sum(axis=0).astype(float)
        ret = np.diag(deg) - A.astype(float)

        return ret

    def sig2vec(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim != 2:
            raise ValueError("Dimensionality of signal is not 2")
        return signal.flatten()

    def vec2sig(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim != 1:
            raise ValueError("Dimensionality of signal is not 1")
        return signal.reshape(self.params["nb_time"], self.graph.N)
