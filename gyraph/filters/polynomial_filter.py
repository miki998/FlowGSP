"""
Copyright © 2025 Chun Hei Michael Chan, MIPLab EPFL
"""


from gyraph.utils import np, torch
from .graph_filter import GraphFilter

from typing import Optional, Tuple


class PolynomialFilter(GraphFilter):
    """
    A class for polynomial graph filters that applies a polynomial of the graph shift operator (GSO)
    to a signal in vertex domain.
    This class inherits from GraphFilter and implements the specific methods
    for polynomial filtering, including the computation of polynomial coefficients
    and the application of the filter to a signal.
    """

    def __init__(
        self,
        graph,
        name=None,
        params=None,
        order: Optional[int] = None,
        scale_operator: Optional[int] = None,
    ):
        super().__init__(graph, name=name, params=params)
        if order is None:
            self.params["order"] = int(np.sqrt(self.graph.N))  # Default order
        else:
            self.params["order"] = order
        self.precompute_polynomial(scale_operator=scale_operator)
        self.name = "PolynomialFilter"

    def apply(
        self,
        signal: np.ndarray,
        kernel: np.ndarray,
        return_coefs: bool = False,
        rcond: float = 1e-8,
    ) -> np.ndarray:
        """
        Applies the polynomial filter to a signal on a (undirected & directed) graph.

        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            kernel (np.ndarray): The graph filter kernel.
            return_coefs (bool): Whether to return the polynomial coefficients.
            rcond (float): The cutoff for the pseudo-inverse.

        Returns
        -------
            filtered_signal (np.ndarray): The filtered signal after applying the polynomial filter. + coefs if return_coefs is True.
        """
        if return_coefs:
            graph_filter, coefs = self.polynomial_filter(
                kernel, return_coefs=return_coefs, rcond=rcond
            )
            return graph_filter @ signal, coefs
        else:
            graph_filter = self.polynomial_filter(
                kernel, return_coefs=return_coefs, rcond=rcond
            )
            return graph_filter @ signal

    def polynomial_filter(
        self, kernel: np.ndarray, return_coefs: bool = False, rcond: float = 1e-8
    ) -> np.ndarray:
        """
        Applies a polynomial graph filter to a signal on a (undirected & directed) graph.

        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            kernel (np.ndarray): The graph filter kernel.
            return_coefs (bool): Whether to return the polynomial coefficients.
            rcond (float): The cutoff for the pseudo-inverse.

        Returns
        -------
            graph_filter (np.ndarray): graph filter
        """

        assert kernel.ndim == 1, "The kernel must be a 1D array."
        deg = self.params["order"]
        _, c = self.get_polynomial_coefficients(kernel, deg=deg, rcond=rcond)

        graph_filter = np.sum([c[i] * self.powers_of_M[i] for i in range(deg)], axis=0)

        if return_coefs:
            return graph_filter, c
        return graph_filter

    def precompute_polynomial(self, scale_operator: Optional[int] = None):
        """
        Precompute and store the list of powers of the graph shift operator matrix.
        """
        M = self.graph.operator.M
        if scale_operator is not None:
            M = M / scale_operator
        order = int(self.params["order"])
        self.powers_of_M = [np.eye(M.shape[0], dtype=M.dtype)]
        for _ in range(1, order):
            self.powers_of_M.append(self.powers_of_M[-1] @ M)

    def get_polynomial_coefficients(
        self, kernel: np.ndarray, deg: float, rcond: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Simply solve for (c_i) the system spectral with filter P (i.e kernel)

        Paramters
        ---------
        kernel: np.ndarray
            The filter kernel.
        deg: int
            The minimum polynomial degree.
        rcond: float
            The cutoff for the pseudo-inverse.

        Returns
        ---------
        vdm_optim: np.ndarray
            The Vandermonde matrix.
        c_optim: np.ndarray
            The polynomial coefficients.
        """
        if deg >= 0:
            vdm_optim = self.vandermonde_matrix(self.graph.operator.V, deg)
            c_optim = np.linalg.pinv(vdm_optim, rcond=rcond) @ kernel
        else:
            c_optim = None
            vdm_optim = None
            best_reconstruct = np.inf
            for k in range(1, kernel.shape[0] + 1):
                vdm = self.vandermonde_matrix(self.graph.operator.V, k)
                c = np.linalg.pinv(vdm, rcond=rcond) @ kernel
                reconstruct_error = np.abs(vdm @ c - kernel).sum()
                if reconstruct_error < best_reconstruct:
                    best_reconstruct = reconstruct_error
                    c_optim = c
                    vdm_optim = vdm

        return vdm_optim, c_optim

    def vandermonde_matrix(self, V: np.ndarray, dim: int) -> np.ndarray:
        """
        Computes the Vandermonde matrix of a vector.

        Parameters
        ----------
            dim (int): The dimension of the Vandermonde matrix.

        Returns
        -------
            vdm (np.ndarray): The Vandermonde matrix.
        """

        vdm = np.zeros((V.shape[0], dim)).astype(complex)
        for sidx in range(dim):
            vdm[:, sidx] = V**sidx
        return vdm

    def regression_descent(
        self,
        signal,
        target,
        deg: Optional[int] = None,
        n_iter: int = 100,
        lr: float = 1e-2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs regression descent to find the optimal polynomial coefficients for approximating a target signal.
        This method does not employ vdm, but directly optimizes the coefficients in the vertex domain.
        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            target (np.ndarray): The target signal to approximate.
            deg (int): The degree of the polynomial.
            n_iter (int): The number of iterations for regression descent.
            lr (float): The learning rate for the optimizer.

        Returns
        -------
            recon (np.ndarray): The reconstructed signal.
            coefs (np.ndarray): The optimal polynomial coefficients found through regression descent.
        """
        # Learn coefficients c such that sum_i c_i * powers_of_M[i] @ signal ~= target
        if deg is None:
            deg = int(self.params["order"])
        # prepare torch tensors (support complex)
        dtype = (
            torch.complex128
            if np.iscomplexobj(signal) or np.iscomplexobj(target)
            else torch.float64
        )
        device = torch.device("cpu")

        signal_t = torch.tensor(signal, dtype=dtype, device=device)
        target_t = torch.tensor(target, dtype=dtype, device=device)

        # stack powers applied to signal for efficiency: shape (deg, N)
        Msig_list = []
        for i in range(deg):
            Mi_t = torch.tensor(self.powers_of_M[i], dtype=dtype, device=device)
            Msig_list.append(Mi_t.matmul(signal_t))
        Msig = torch.stack(Msig_list, dim=0)  # (deg, N)

        # coefficients to learn
        coefs = torch.zeros(deg, dtype=dtype, device=device, requires_grad=True)

        optim = torch.optim.Adam([coefs], lr=lr)
        prev_loss = None
        for _ in range(n_iter):
            optim.zero_grad()
            recon = (coefs.unsqueeze(1) * Msig).sum(dim=0)
            loss = torch.mean(torch.abs(recon - target_t) ** 2)
            loss.backward()
            optim.step()
            if prev_loss is not None and abs(prev_loss - loss.item()) < 1e-9:
                break
            prev_loss = loss.item()

        return recon.detach().numpy(), coefs.detach().numpy(), prev_loss

    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"


class DualPolynomialFilter(PolynomialFilter):
    """
    A class for polynomial graph filters that applies a polynomial of the graph shift operator (GSO)
    to a signal in vertex domain.
    This class inherits from PolynomialFilter and implements the specific methods
    for dual GSO polynomial filtering.
    """

    def __init__(
        self,
        graph,
        name=None,
        filter_type: Optional[str] = None,
        order: Optional[int] = None,
        scale_operator: Optional[int] = None,
    ):
        self.params = {}
        if filter_type not in ["GAGD", "GQAD", "GQDA", "GA", "GD"]:
            raise ValueError(
                "filter_type must be one of 'GAGD', 'GQAD', 'GQDA', 'GA', or 'GD'."
            )
        self.params["filter_type"] = filter_type

        super().__init__(graph, name=name, params=self.params, order=order)
        self.name = "DualPolynomialFilter"
        self.precompute_polynomial(scale_operator)

    def apply(
        self,
        signal: np.ndarray,
        kernel: np.ndarray,
        return_coefs: bool = False,
        rcond: float = 1e-8,
    ) -> np.ndarray:
        """
        Applies the polynomial filter to a signal on a (undirected & directed) graph.

        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            kernel (np.ndarray): The graph filter kernel.
            return_coefs (bool): Whether to return the polynomial coefficients.
            rcond (float): The cutoff for the pseudo-inverse.

        Returns
        -------
            filtered_signal (np.ndarray): The filtered signal after applying the polynomial filter. + coefs if return_coefs is True.
        """
        if return_coefs:
            graph_filter, coefs = self.polynomial_filter(
                kernel, return_coefs=return_coefs, rcond=rcond
            )
            return graph_filter @ signal, coefs
        else:
            graph_filter = self.polynomial_filter(
                kernel, return_coefs=return_coefs, rcond=rcond
            )
            return graph_filter @ signal

    def polynomial_filter(
        self, kernel: np.ndarray, return_coefs: bool = False, rcond: float = 1e-8
    ) -> np.ndarray:
        """
        Applies a polynomial graph filter to a signal on a (undirected & directed) graph.

        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            kernel (np.ndarray): The graph filter kernel.
            return_coefs (bool): Whether to return the polynomial coefficients.
            rcond (float): The cutoff for the pseudo-inverse.

        Returns
        -------
            graph_filter (np.ndarray): graph filter
        """

        assert kernel.ndim == 1, "The kernel must be a 1D array."
        deg = self.params["order"]
        _, c = self.get_polynomial_coefficients(kernel, deg=deg, rcond=rcond)

        if self.params["filter_type"] == "GAGD":
            graph_filter = np.sum(
                [c[i] * self.powers_of_P[i] for i in range(deg)], axis=0
            ) + np.sum([c[deg + i] * self.powers_of_Q[i] for i in range(deg)], axis=0)
        elif self.params["filter_type"] == "GQAD":
            graph_filter = np.sum(
                [c[i] * self.powers_of_R[i] for i in range(deg)], axis=0
            )
        elif self.params["filter_type"] == "GQDA":
            graph_filter = np.sum(
                [c[i] * self.powers_of_R[i] for i in range(deg)], axis=0
            )
        elif self.params["filter_type"] == "GA":
            graph_filter = np.sum(
                [c[i] * self.powers_of_A[i] for i in range(deg)], axis=0
            )
        elif self.params["filter_type"] == "GD":
            graph_filter = np.sum(
                [c[i] * self.powers_of_D[i] for i in range(deg)], axis=0
            )

        if return_coefs:
            return graph_filter, c
        return graph_filter

    def precompute_polynomial(self, scale_operator: Optional[int] = None):
        """
        Precompute and store the list of powers of the graph shift operator matrix.

        Parameters
        ----------
            scale_operator (Optional[int]): A scaling factor for the graph shift operator. If None, no scaling is applied.
        """
        order = int(self.params["order"])
        scale_operator = scale_operator if scale_operator is not None else 1.0

        if self.params["filter_type"] == "GAGD":
            P = self.graph.operator.P / scale_operator
            Q = self.graph.operator.Q / scale_operator
            self.powers_of_P = [np.eye(P.shape[0], dtype=P.dtype)]
            self.powers_of_Q = [np.eye(Q.shape[0], dtype=Q.dtype)]
            for _ in range(1, order):
                self.powers_of_P.append(self.powers_of_P[-1] @ P)
                self.powers_of_Q.append(self.powers_of_Q[-1] @ Q)

        elif self.params["filter_type"] == "GQAD":
            R = self.graph.operator.Z / scale_operator
            self.powers_of_R = [np.eye(R.shape[0], dtype=R.dtype)]
            for _ in range(1, order):
                self.powers_of_R.append(self.powers_of_R[-1] @ R)

        elif self.params["filter_type"] == "GQDA":
            R = (
                self.graph.operator.P
                @ np.linalg.pinv(self.graph.operator.Q)
                / scale_operator
            )  # TODO: in the future remove, this is too ill-posed.
            self.powers_of_R = [np.eye(R.shape[0], dtype=R.dtype)]
            for _ in range(1, order):
                self.powers_of_R.append(self.powers_of_R[-1] @ R)

        elif self.params["filter_type"] == "GA":
            A = self.graph.operator.Q / scale_operator
            self.powers_of_A = [np.eye(A.shape[0], dtype=A.dtype)]
            for _ in range(1, order):
                self.powers_of_A.append(self.powers_of_A[-1] @ A)

        elif self.params["filter_type"] == "GD":
            D = self.graph.operator.P / scale_operator
            self.powers_of_D = [np.eye(D.shape[0], dtype=D.dtype)]
            for _ in range(1, order):
                self.powers_of_D.append(self.powers_of_D[-1] @ D)

    def vandermonde_matrix_compose(self, deg: float, rcond: float = 1e-8) -> np.ndarray:
        """
        Computes the Vandermonde matrix of a vector for the dual polynomial filter.

        Parameters
        ----------
            deg (int): The degree of the Vandermonde matrix.
            rcond (float): The cutoff for the pseudo-inverse.

        Returns
        -------
            vdm (np.ndarray): The Vandermonde matrix.
        """
        if self.params["filter_type"] == "GAGD":
            vdm_imag = self.vandermonde_matrix(1j * self.graph.operator.V.imag, deg)
            vdm_real = self.vandermonde_matrix(self.graph.operator.V.real, deg)
            vdm_optim = np.concatenate((vdm_real, vdm_imag), axis=1)
        elif self.params["filter_type"] == "GQAD":
            Vpinv_real = np.diag(
                np.linalg.pinv(np.diag(self.graph.operator.V.real), rcond=rcond)
            )
            vdm_optim = self.vandermonde_matrix(
                (1j * self.graph.operator.V.imag) * Vpinv_real, deg
            )
        elif self.params["filter_type"] == "GQDA":
            Vpinv_imag = np.diag(
                np.linalg.pinv(np.diag(self.graph.operator.V.imag), rcond=rcond)
            )
            vdm_optim = self.vandermonde_matrix(
                (self.graph.operator.V.real) * Vpinv_imag, deg
            )
        elif self.params["filter_type"] == "GA":
            vdm_optim = self.vandermonde_matrix(1j * self.graph.operator.V.imag, deg)
        elif self.params["filter_type"] == "GD":
            vdm_optim = self.vandermonde_matrix(
                self.graph.operator.V.real.astype(complex), deg
            )

        return vdm_optim

    def get_polynomial_coefficients(
        self, kernel: np.ndarray, deg: float, rcond: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simply solve for (c_i) the system spectral with filter P (i.e kernel)

        Paramters
        ---------
        kernel: np.ndarray
            The filter kernel.
        deg: int
            The minimum polynomial degree.
        rcond: float
            The cutoff for the pseudo-inverse.

        Returns
        ---------
        vdm_optim: np.ndarray
            The Vandermonde matrix.
        c_optim: np.ndarray
            The polynomial coefficients.
        """
        if deg >= 0:
            vdm_optim = self.vandermonde_matrix_compose(deg=deg, rcond=rcond)
            c_optim = np.linalg.pinv(vdm_optim, rcond=rcond) @ kernel
        else:
            c_optim = None
            vdm_optim = None
            best_reconstruct = np.inf
            for k in range(1, kernel.shape[0] + 1):
                vdm = self.vandermonde_matrix_compose(k, rcond=rcond)
                c = np.linalg.pinv(vdm, rcond=rcond) @ kernel
                reconstruct_error = np.abs(vdm @ c - kernel).sum()
                if reconstruct_error < best_reconstruct:
                    best_reconstruct = reconstruct_error
                    c_optim = c
                    vdm_optim = vdm

        return vdm_optim, c_optim

    def regression_descent(
        self,
        signal,
        target,
        deg: Optional[int] = None,
        n_iter: int = 100,
        lr: float = 1e-2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs regression descent to find the optimal polynomial coefficients for approximating a target signal.
        This method does not employ vdm, but directly optimizes the coefficients in the vertex domain.
        Parameters
        ----------
            signal (np.ndarray): The input signal to be filtered.
            target (np.ndarray): The target signal to approximate.
            deg (int): The degree of the polynomial.
            n_iter (int): The number of iterations for regression descent.
            lr (float): The learning rate for the optimizer.

        Returns
        -------
            recon (np.ndarray): The reconstructed signal.
            coefs (np.ndarray): The optimal polynomial coefficients found through regression descent.
        """
        # Learn coefficients c such that sum_i c_i * powers_of_M[i] @ signal ~= target
        if deg is None:
            deg = int(self.params["order"])
        # prepare torch tensors (support complex)
        dtype = (
            torch.complex128
            if np.iscomplexobj(signal) or np.iscomplexobj(target)
            else torch.float64
        )
        device = torch.device("cpu")

        signal_t = torch.tensor(signal, dtype=dtype, device=device)
        target_t = torch.tensor(target, dtype=dtype, device=device)

        if self.params["filter_type"] == "GAGD":
            # stack powers applied to signal for efficiency: shape (deg, N)
            Msig_list = []
            for i in range(deg):
                Mi_t = torch.tensor(self.powers_of_P[i], dtype=dtype, device=device)
                Msig_list.append(Mi_t.matmul(signal_t))
            for i in range(deg):
                Mi_t = torch.tensor(self.powers_of_Q[i], dtype=dtype, device=device)
                Msig_list.append(Mi_t.matmul(signal_t))
            Msig = torch.stack(Msig_list, dim=0)  # (deg * 2, N)
        elif (
            self.params["filter_type"] == "GQAD" or self.params["filter_type"] == "GQDA"
        ):
            Msig_list = []
            for i in range(deg):
                Mi_t = torch.tensor(self.powers_of_R[i], dtype=dtype, device=device)
                Msig_list.append(Mi_t.matmul(signal_t))
            Msig = torch.stack(Msig_list, dim=0)  # (deg, N)
        elif self.params["filter_type"] == "GA":
            Msig_list = []
            for i in range(deg):
                Mi_t = torch.tensor(self.powers_of_A[i], dtype=dtype, device=device)
                Msig_list.append(Mi_t.matmul(signal_t))
            Msig = torch.stack(Msig_list, dim=0)  # (deg, N)
        elif self.params["filter_type"] == "GD":
            Msig_list = []
            for i in range(deg):
                Mi_t = torch.tensor(self.powers_of_D[i], dtype=dtype, device=device)
                Msig_list.append(Mi_t.matmul(signal_t))
            Msig = torch.stack(Msig_list, dim=0)  # (deg, N)

        # coefficients to learn
        coefs = torch.zeros(
            Msig.shape[0], dtype=dtype, device=device, requires_grad=True
        )

        optim = torch.optim.Adam([coefs], lr=lr)
        prev_loss = None
        for _ in range(n_iter):
            optim.zero_grad()
            recon = (coefs.unsqueeze(1) * Msig).sum(dim=0)
            loss = torch.mean(torch.abs(recon - target_t) ** 2)
            loss.backward()
            optim.step()
            if prev_loss is not None and abs(prev_loss - loss.item()) < 1e-9:
                break
            prev_loss = loss.item()

        return recon.detach().numpy(), coefs.detach().numpy(), prev_loss

    def __repr__(self):
        return f"<Filter(name={self.name}, params={self.params})>"
