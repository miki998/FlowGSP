"""
Copyright © 2026 Chun Hei Michael Chan, MIPLab EPFL

chebyshev_filter.py
─────────────────────────────
Numerically stable polynomial graph filters built on the Chebyshev basis.

Replacements for PolynomialFilter and DualPolynomialFilter that
eliminate two sources of numerical failure present in the monomial parents:

  (a) Exponential Vandermonde ill-conditioning: κ(V_monomial) ~ (|λ_max|/|λ_min|)^K.
  (b) Matrix-power overflow: entries of M^k grow as |λ_max|^k.

Both are replaced by the Chebyshev three-term recurrence

    T_{k+1}(M̃) = 2 M̃ T_k(M̃) − T_{k-1}(M̃),   M̃ = (M − cI) / R,

which requires one matrix multiply per step and remains forward-stable for
all k.  The corresponding Vandermonde V[n,k] = T_k((λ_n − c)/R) ∈ [−1, 1]
(real case) inherits the near-orthogonality of Chebyshev polynomials.

Basis selection per filter type (see DualChebyshevFilter docstring).
TODO: Finish implementation for Vandermonde, currently use the regression instead.
"""

from __future__ import annotations

from gyraph.utils import np, Optional, Tuple, torch
from .polynomial_filter import PolynomialFilter


# ══════════════════════════════════════════════════════════════════════════════
# Module-level Chebyshev helpers (shared by both classes)
# ══════════════════════════════════════════════════════════════════════════════


def _cheb_scaling(eigenvalues: np.ndarray) -> Tuple[complex, float]:
    """
    Bounding-box centre c and circumscribed radius R for a set of (possibly
    complex) eigenvalues, so that (eigenvalues − c) / R lies in [−1, 1] for
    real input, or in a disk of radius 1 for complex input.

    For a real spectrum on [λ_min, λ_max] this is exactly the standard
    Chebyshev affine map:  c = (λ_max + λ_min)/2,  R = (λ_max − λ_min)/2.

    Parameters
    ----------
    eigenvalues : (N,) array, real or complex (partial spectrum is accepted e.g array[min, max])

    Returns
    -------
    c : complex  – bounding-box centre
    R : float    – circumscribed radius (> 0)
    """
    re_c = (eigenvalues.real.max() + eigenvalues.real.min()) / 2.0
    im_c = (eigenvalues.imag.max() + eigenvalues.imag.min()) / 2.0
    c = complex(re_c, im_c)
    R = float(np.abs(eigenvalues - c).max()) + 1e-14  # eps guards R = 0
    return c, R


def _cheb_vdm(scaled: np.ndarray, deg: int) -> np.ndarray:
    """
    Chebyshev Vandermonde  C[n, k] = T_k(scaled[n]),  k = 0 … deg−1,
    computed via the three-term recurrence.

    Numerically stable for real arguments in [−1, 1].  Works for complex
    arguments as well (Chebyshev approximation on circumscribed ellipse).

    Parameters
    ----------
    scaled : (N,) array – pre-normalised arguments  (λ − c) / R
    deg    : int        – number of columns
    """
    N = len(scaled)
    dtype = complex if np.iscomplexobj(scaled) else float
    C = np.zeros((N, deg), dtype=dtype)
    if deg == 0:
        return C
    C[:, 0] = 1.0
    if deg == 1:
        return C
    C[:, 1] = scaled
    for k in range(2, deg):
        C[:, k] = 2.0 * scaled * C[:, k - 1] - C[:, k - 2]
    return C


def _cheb_mats(M: np.ndarray, c: complex, R: float, deg: int) -> list:
    """
    Compute  [T_0(M̃), T_1(M̃), …, T_{deg−1}(M̃)]  via the matrix
    three-term recurrence,  M̃ = (M − c·I) / R.

    Stable drop-in replacement for  [I, M, M², …, M^{deg−1}]:
    only one matrix multiply per step, no exponential entry growth.

    Parameters
    ----------
    M   : (N, N) array – operator matrix (real or complex)
    c   : complex      – spectral bounding-box centre
    R   : float        – spectral bounding-box radius
    deg : int          – number of Chebyshev matrices to return
    """
    N = M.shape[0]
    dtype = np.result_type(M.dtype, np.complex128)
    Mt = (M.astype(dtype) - c * np.eye(N, dtype=dtype)) / R  # M̃

    mats: list = []
    if deg == 0:
        return mats

    T_prev = np.eye(N, dtype=dtype)
    mats.append(T_prev.copy())
    if deg == 1:
        return mats

    T_cur = Mt.copy()
    mats.append(T_cur.copy())

    for _ in range(2, deg):
        T_next = 2.0 * Mt @ T_cur - T_prev
        mats.append(T_next.copy())
        T_prev, T_cur = T_cur, T_next

    return mats


# ══════════════════════════════════════════════════════════════════════════════
# ChebyshevFilter
# ══════════════════════════════════════════════════════════════════════════════


class ChebyshevFilter(PolynomialFilter):
    """
    Numerically stable drop-in replacement for PolynomialFilter.

    Monomial basis  {I, M, M², …}  is replaced by the Chebyshev basis

        {T_0(M̃), T_1(M̃), …, T_{K−1}(M̃)},   M̃ = (M − c·I) / R,

    where c and R are the centre and circumscribed radius of the bounding
    box of the spectrum of M.

    Spectral notes
    ──────────────
    Real spectrum:
        The bounding box is [λ_min, λ_max]; T_k is the standard Chebyshev
        polynomial on a real interval — the exact optimal Faber polynomial
        for that segment.

    Complex scattered spectrum:
        The bounding box is a disk; T_k evaluated at complex arguments gives
        the Chebyshev-on-ellipse approximation.  True Faber polynomials for a
        general ellipse do NOT satisfy a constant-coefficient three-term
        recurrence (their α_k are degree-dependent), so this is the correct
        engineering substitute: it is strictly better-conditioned than the
        monomial basis, and optimal only when the spectrum is strongly
        axis-aligned.
    """

    def __init__(
        self,
        graph,
        name=None,
        params=None,
        order: Optional[int] = None,
        scale_operator: Optional[int] = None,
    ):
        # Must exist before super().__init__ triggers precompute_polynomial
        self._cheb_c: complex = 0.0
        self._cheb_R: float = 1.0
        super().__init__(
            graph,
            name=name,
            params=params,
            order=order,
            scale_operator=scale_operator,
        )
        self.name = "ChebyshevFilter"

    # ── core overrides ────────────────────────────────────────────────────────
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

    def precompute_polynomial(self, scale_operator: Optional[int] = None):
        """Build {T_k(M̃)} and store as chebyshev_matrices / powers_of_M."""
        M = self.graph.operator.M
        if scale_operator is not None:
            M = M / scale_operator
        order = int(self.params["order"])

        # Chebyshev scaling from the operator's eigenvalues (unscaled)
        self._cheb_c, self._cheb_R = _cheb_scaling(self.graph.operator.V)
        self.chebyshev_matrices = _cheb_mats(M, self._cheb_c, self._cheb_R, order)

        # Alias keeps parent's polynomial_filter / apply working unchanged
        self.powers_of_M = self.chebyshev_matrices

    def vandermonde_matrix(self, V: np.ndarray, dim: int) -> np.ndarray:
        """Chebyshev Vandermonde  C[n, k] = T_k((V[n] − c) / R)."""
        scaled = (V - self._cheb_c) / self._cheb_R
        return _cheb_vdm(scaled, dim)

    def polynomial_filter(
        self, kernel: np.ndarray, return_coefs: bool = False, rcond: float = 1e-8
    ) -> np.ndarray:
        assert kernel.ndim == 1, "The kernel must be a 1D array."
        deg = self.params["order"]
        _, c = self.get_polynomial_coefficients(kernel, deg=deg, rcond=rcond)
        graph_filter = np.sum(
            [c[i] * self.chebyshev_matrices[i] for i in range(deg)], axis=0
        )
        if return_coefs:
            return graph_filter, c
        return graph_filter

    def regression_descent(
        self,
        signal,
        target,
        deg: Optional[int] = None,
        n_iter: int = 100,
        lr: float = 1e-2,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Adam descent over Chebyshev coefficients in vertex domain."""
        if deg is None:
            deg = int(self.params["order"])

        # Chebyshev matrices may be complex; always use complex128
        dtype = torch.complex128
        device = torch.device("cpu")

        signal_t = torch.tensor(signal, dtype=dtype, device=device)
        target_t = torch.tensor(target, dtype=dtype, device=device)

        Msig_list = [
            torch.tensor(self.chebyshev_matrices[i], dtype=dtype, device=device).matmul(
                signal_t
            )
            for i in range(deg)
        ]
        Msig = torch.stack(Msig_list, dim=0)  # (deg, N)
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


# ══════════════════════════════════════════════════════════════════════════════
# DualChebyshevFilter
# ══════════════════════════════════════════════════════════════════════════════


class DualChebyshevFilter(ChebyshevFilter):
    """
    Numerically stable drop-in replacement for DualPolynomialFilter.

    The monomial powers of each operator are replaced by Chebyshev matrices
    {T_k(M̃)}, with the basis chosen per filter_type to guarantee real (or
    at least bounded) Vandermonde arguments.

    Basis selection rationale
    ─────────────────────────
    GD   Real eigenvalues λR ⊂ [λ_min, λ_max].
         Standard Chebyshev T_k((P − c·I)/R) — exact Faber polynomial for
         the interval.  Vandermonde arguments (λR − c)/R ∈ [−1, 1].

    GA   Q = L_up has purely imaginary eigenvalues i·λI.
         Naïvely T_k(i·λI / R) = cosh(k · arcsinh(λI/R)) grows
         exponentially in k → catastrophically unstable.

         Fix (−i substitution): let  Â = −i·Q,  eigenvalues(Â) = λI ∈ ℝ.
         Chebyshev basis is built from Â; the Vandermonde uses the real
         arguments λI / R_I ∈ [−1, 1].

         Consistency check:
           T_k(Â/R_I) eigenvalue at node n
           = T_k(−i·(i·λI_n)/R_I)
           = T_k(λI_n/R_I)  ✓ (matches Vandermonde column k)

    GAGD GD basis applied to P; GA basis (−i substitution) applied to Q.
         Vandermonde columns are concatenated: [(λR−c)/R_R | λI/R_I].

    GQAD Z = L_up/L_circ has purely imaginary eigenvalues i·λI/λR.
         Same −i substitution: build basis from −i·Z (real eigenvalues
         λI/λR).  Vandermonde arguments (λI/λR − c_Q)/R_Q ∈ [−1, 1].

    GQDA P @ pinv(Q) has purely imaginary eigenvalues −i·λR/λI.
         Same −i substitution: build basis from −i·(P @ pinv(Q)) (real
         eigenvalues λR/λI).  Vandermonde arguments (λR/λI − c_Q)/R_Q.

    Inheritance / compatibility
    ───────────────────────────
    • self.powers_of_{D,A,P,Q,R} hold Chebyshev matrices with the same
      attribute names as DualPolynomialFilter; polynomial_filter assembles
      the graph filter from them per filter_type (overridden below, since
      this class inherits from ChebyshevFilter, not DualPolynomialFilter).
    • get_polynomial_coefficients (overridden below) fits the kernel on
      self.vandermonde_matrix_compose; apply is inherited from
      ChebyshevFilter and dispatches to the overridden polynomial_filter.
    • vandermonde_matrix (monomial) from PolynomialFilter is NOT used;
      vandermonde_matrix_compose calls _cheb_vdm directly.

    Spectral consistency (default scale_operator=None):
      Chebyshev arguments in the Vandermonde and in the precomputed matrices
      are derived from the same eigenvalue expressions, so coefficient fitting
      and matrix assembly are always consistent.
    """

    def __init__(
        self,
        graph,
        name=None,
        filter_type: Optional[str] = None,
        order: Optional[int] = None,
        scale_operator: Optional[int] = None,
    ):
        self._cheb_params: dict = {}  # populated in precompute_polynomial
        self.params = {}
        if filter_type not in ["GAGD", "GQAD", "GQDA", "GA", "GD"]:
            raise ValueError(
                "filter_type must be one of 'GAGD', 'GQAD', 'GQDA', 'GA', or 'GD'."
            )
        self.params["filter_type"] = filter_type

        super().__init__(
            graph,
            name=name,
            params=self.params,
            order=order,
            scale_operator=scale_operator,
        )
        self.name = "DualChebyshevFilter"

    # ── core overrides ────────────────────────────────────────────────────────

    def precompute_polynomial(self, scale_operator: Optional[int] = None):
        """
        Build Chebyshev matrix lists for each filter_type and store in the
        same attributes (powers_of_D, powers_of_A, …) as the parent.
        """
        order = int(self.params["order"])
        s = float(scale_operator) if scale_operator is not None else 1.0

        V = self.graph.operator.V  # eigenvalues of L (complex)
        lam_R = V.real  # eigenvalues of P = L_circ (real)
        lam_I = V.imag  # Im(λ) s.t. eig(Q) = i·λI (purely imag)
        eps = 1e-14

        if self.params["filter_type"] == "GD":
            # ── Real spectrum → standard Chebyshev on [λ_min, λ_max] ────────
            P = self.graph.operator.P / s
            c_R, R_R = _cheb_scaling(lam_R.astype(complex))
            self._cheb_params.update({"c_R": float(c_R.real), "R_R": R_R})
            self.powers_of_D = _cheb_mats(P, c_R, R_R, order)

        elif self.params["filter_type"] == "GA":
            # ── Purely imaginary eigenvalues → −i substitution ───────────────
            # Â = −i·Q  has real eigenvalues λI; T_k(Â/R_I) is stable
            Q = self.graph.operator.Q / s
            R_I = float(np.abs(lam_I).max()) + eps
            self._cheb_params.update({"R_I": R_I})
            self.powers_of_A = _cheb_mats(-1j * Q, 0.0, R_I, order)

        elif self.params["filter_type"] == "GAGD":
            # ── GD basis for P; GA (−i) basis for Q ─────────────────────────
            P = self.graph.operator.P / s
            Q = self.graph.operator.Q / s
            c_R, R_R = _cheb_scaling(lam_R.astype(complex))
            R_I = float(np.abs(lam_I).max()) + eps
            self._cheb_params.update({"c_R": float(c_R.real), "R_R": R_R, "R_I": R_I})
            self.powers_of_P = _cheb_mats(P, c_R, R_R, order)
            self.powers_of_Q = _cheb_mats(-1j * Q, 0.0, R_I, order)

        elif self.params["filter_type"] == "GQAD":
            # ── Z has purely imaginary eigenvalues i·λI/λR → −i substitution ─
            # −i·Z has real eigenvalues λI/λR
            Z = self.graph.operator.Z / s
            eig_q = lam_I / np.where(np.abs(lam_R) > eps, lam_R, eps)  # λI/λR ∈ ℝ
            c_Q, R_Q = _cheb_scaling(eig_q.astype(complex))
            self._cheb_params.update(
                {"c_Q": float(c_Q.real), "R_Q": R_Q, "eig_q": eig_q}
            )
            self.powers_of_R = _cheb_mats(-1j * Z, float(c_Q.real), R_Q, order)

        elif self.params["filter_type"] == "GQDA":
            # ── P@pinv(Q) has purely imaginary eigenvalues −i·λR/λI ──────────
            # −i·(P@pinv(Q)) has real eigenvalues λR/λI
            R_mat = self.graph.operator.P @ np.linalg.pinv(self.graph.operator.Q) / s
            eig_q = lam_R / np.where(np.abs(lam_I) > eps, lam_I, eps)  # λR/λI ∈ ℝ
            c_Q, R_Q = _cheb_scaling(eig_q.astype(complex))
            self._cheb_params.update(
                {"c_Q": float(c_Q.real), "R_Q": R_Q, "eig_q": eig_q}
            )
            self.powers_of_R = _cheb_mats(-1j * R_mat, float(c_Q.real), R_Q, order)

    def vandermonde_matrix_compose(self, deg: int, rcond: float = 1e-8) -> np.ndarray:
        """
        Chebyshev Vandermonde matching the precomputed matrix basis.

        Spectral consistency: column k of the returned matrix equals
        the vector of eigenvalues of self.powers_of_*[k], so the
        pseudoinverse coefficient solve is always consistent with the
        matrix assembly in polynomial_filter.
        """
        p = self._cheb_params
        V = self.graph.operator.V
        lam_R = V.real
        lam_I = V.imag

        if self.params["filter_type"] == "GD":
            scaled = (lam_R - p["c_R"]) / p["R_R"]  # ∈ [−1, 1]
            return _cheb_vdm(scaled, deg)

        elif self.params["filter_type"] == "GA":
            scaled = lam_I / p["R_I"]  # ∈ [−1, 1]  (−i trick)
            return _cheb_vdm(scaled, deg)

        elif self.params["filter_type"] == "GAGD":
            scaled_R = (lam_R - p["c_R"]) / p["R_R"]
            scaled_I = lam_I / p["R_I"]
            return np.concatenate(
                [_cheb_vdm(scaled_R, deg), _cheb_vdm(scaled_I, deg)], axis=1
            )

        elif self.params["filter_type"] == "GQAD":
            # eig_q = λI/λR (real); stored in _cheb_params by precompute
            scaled = (p["eig_q"] - p["c_Q"]) / p["R_Q"]
            return _cheb_vdm(scaled, deg)

        elif self.params["filter_type"] == "GQDA":
            scaled = (p["eig_q"] - p["c_Q"]) / p["R_Q"]
            return _cheb_vdm(scaled, deg)

    def get_polynomial_coefficients(
        self, kernel: np.ndarray, deg: float, rcond: float = 1e-8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the kernel on the composed Chebyshev Vandermonde (spectrally
        consistent with the precomputed matrix basis). Mirrors
        DualPolynomialFilter.get_polynomial_coefficients, which this class
        does not inherit (its parent is ChebyshevFilter).
        """
        if deg >= 0:
            vdm_optim = self.vandermonde_matrix_compose(deg, rcond=rcond)
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

    def polynomial_filter(
        self, kernel: np.ndarray, return_coefs: bool = False, rcond: float = 1e-8
    ) -> np.ndarray:
        """
        Assemble the graph filter from the per-filter_type Chebyshev matrix
        basis, with coefficients fitted on the composed Vandermonde. Mirrors
        DualPolynomialFilter.polynomial_filter with Chebyshev matrices in
        place of monomial powers.
        """
        assert kernel.ndim == 1, "The kernel must be a 1D array."
        deg = self.params["order"]
        _, c = self.get_polynomial_coefficients(kernel, deg=deg, rcond=rcond)

        if self.params["filter_type"] == "GAGD":
            graph_filter = np.sum(
                [c[i] * self.powers_of_P[i] for i in range(deg)], axis=0
            ) + np.sum([c[deg + i] * self.powers_of_Q[i] for i in range(deg)], axis=0)
        elif self.params["filter_type"] in ("GQAD", "GQDA"):
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

    def regression_descent(
        self,
        signal,
        target,
        deg: Optional[int] = None,
        n_iter: int = 100,
        lr: float = 1e-2,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Adam descent over Chebyshev coefficients in vertex domain."""
        if deg is None:
            deg = int(self.params["order"])

        dtype = torch.complex128  # Chebyshev matrices are always complex
        device = torch.device("cpu")

        signal_t = torch.tensor(signal, dtype=dtype, device=device)
        target_t = torch.tensor(target, dtype=dtype, device=device)

        def _stack(mats: list) -> torch.Tensor:
            return torch.stack(
                [
                    torch.tensor(mats[i], dtype=dtype, device=device).matmul(signal_t)
                    for i in range(deg)
                ],
                dim=0,
            )

        if self.params["filter_type"] == "GAGD":
            Msig = torch.cat(
                [_stack(self.powers_of_P), _stack(self.powers_of_Q)], dim=0
            )
        elif self.params["filter_type"] in ("GQAD", "GQDA"):
            Msig = _stack(self.powers_of_R)
        elif self.params["filter_type"] == "GA":
            Msig = _stack(self.powers_of_A)
        elif self.params["filter_type"] == "GD":
            Msig = _stack(self.powers_of_D)

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
