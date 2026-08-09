# """
# stable_polynomial_filters.py
# ─────────────────────────────
# Numerically stable polynomial graph filters.

# FaberFilter  — general complex GSO spectrum
#     Uses the Faber polynomial for the circumscribed ellipse of the spectrum,
#     which is the unique choice that guarantees bounded Vandermonde entries.

# ─── Why Faber for stable filter ──────────────────────────────────────

# The original class used Chebyshev with the circumscribed *disk* radius R:
#     T_k((λ − c)/R),  |λ − c|/R ≤ 1  (inside unit disk).

# This does NOT give bounded Vandermonde entries. For complex z in the unit
# disk, T_k(z) = cosh(k·arccosh(z)) can grow exponentially with k: the
# Chebyshev polynomial is bounded only for REAL z ∈ [−1,1].

# The fix — Faber polynomial for the circumscribed ELLIPSE:
#     Given the bounding ellipse with semi-axes a, b and focal distance
#     f = √(a²−b²), the Faber polynomial is F_k(z) = f^k T_k((z−c)/f).
#     The cap-normalised version satisfies |F_k(λ)| ≤ 1 for all λ INSIDE
#     the ellipse, by the maximum principle applied to the conformal map
#     Φ : C\E → {|w|>1}.  This gives Vandermonde entries bounded by cap^k,
#     which is well-conditioned after normalisation.

# The recurrence is constant-coefficient and identical in form to Chebyshev:
#     G₀ = I,  G₁ = M−cI,  G_{k+1} = 2(M−cI)G_k − f² G_{k-1}

# which is just the standard Chebyshev recurrence with M̃ = (M−cI)/f
# scaled back:  f^k T_k(M̃) satisfies exactly this.

# ─── Degenerate case ───────────────────────────────────────────────────────────

# As b → a (circular spectrum): f → 0 and the Faber polynomials for a disk
# are monomials ((z−c)/R)^k — no polynomial basis helps.  In this limit the
# field-of-values obstruction fully dominates and eigendecomposition-based
# spectral filtering (U diag(h) U^{-1}) is the only real solution.

# ─── Residual limitation ───────────────────────────────────────────────────────

# The Faber construction fixes the coefficient solve (Vandermonde conditioning).
# It does NOT fix the matrix norm growth ‖F_k(M)‖ for non-normal M: this is
# the field-of-values obstruction, intrinsic to any polynomial filter on a
# non-normal operator.  The three-term recurrence evaluates F_k(M) in a
# forward-stable way, but the true value ‖F_k(M)‖ can still be large.
# """

# from __future__ import annotations


# from gyraph.utils import np, Optional, Tuple, torch
# from .polynomial_filter import PolynomialFilter

# import scipy.linalg


# # ══════════════════════════════════════════════════════════════════════════════
# # Shared helpers
# # ══════════════════════════════════════════════════════════════════════════════

# def _ellipse_params(eigenvalues: np.ndarray) -> Tuple[complex, float, float, float]:
#     """
#     Fit the smallest axis-aligned bounding ellipse to a set of eigenvalues
#     (which arrive in conjugate pairs for a real matrix, so the ellipse is
#     symmetric about the real axis).

#     Returns
#     -------
#     c : complex  – centre (imaginary part is 0 for real-matrix spectra)
#     a : float    – real semi-axis   (horizontal)
#     b : float    – imaginary semi-axis (vertical)
#     f : float    – focal distance √(a²−b²);  f = 0 when a = b (disk)

#     Note: we add a small ε to both semi-axes so that f is well-defined
#     even when the spectrum is a single point or a line segment.
#     """
#     eps = 1e-12
#     c_R = (eigenvalues.real.max() + eigenvalues.real.min()) / 2.0
#     c_I = (eigenvalues.imag.max() + eigenvalues.imag.min()) / 2.0
#     c   = complex(c_R, c_I)
#     a   = float(eigenvalues.real.max() - c_R) + eps   # real semi-axis
#     b   = float(np.abs(eigenvalues.imag).max())  + eps # imaginary semi-axis
#     f   = float(np.sqrt(max(a**2 - b**2, 0.0)))       # focal distance, ≥ 0
#     return c, a, b, f


# def _faber_vdm(eigenvalues: np.ndarray, c: complex, f: float,
#                a: float, b: float, deg: int) -> np.ndarray:
#     """
#     Faber–Chebyshev Vandermonde for the ellipse (c, a, b):

#         V[n, k] = F_k(λ_n) / cap^k,   cap = (a + b) / 2

#     where F_k is the cap-normalised Faber polynomial satisfying
#     |F_k(λ)| ≤ 1 for all λ inside the ellipse.

#     For f > 0 (proper ellipse):
#         F_k(λ)/cap^k  =  T_k((λ−c)/f) · (f/cap)^k
#                        =  T_k((λ−c)/f) · (2f/(a+b))^k

#     For f = 0 (disk, a = b):
#         The Faber polynomials degenerate to monomials ((λ−c)/a)^k.
#         Falls back to monomial basis — caller should be aware.

#     The columns are bounded in magnitude by 1 for λ inside the ellipse,
#     giving a well-conditioned Vandermonde (unlike the disk-Chebyshev
#     used in the original StablePolynomialFilter).
#     """
#     N   = len(eigenvalues)
#     cap = (a + b) / 2.0

#     C = np.zeros((N, deg), dtype=complex)
#     if deg == 0:
#         return C

#     if f < 1e-10:
#         # Degenerate: disk, Faber = monomials ((λ−c)/a)^k
#         scaled = (eigenvalues - c) / a
#         C[:, 0] = 1.0
#         for k in range(1, deg):
#             C[:, k] = C[:, k-1] * scaled
#         return C

#     # Faber for proper ellipse: T_k((λ−c)/f) · (f/cap)^k
#     scaled  = (eigenvalues - c) / f     # arguments to Chebyshev
#     f_cap   = f / cap                   # per-step scaling factor

#     # Chebyshev recurrence on (λ−c)/f, then scale each column
#     T = np.zeros((N, deg), dtype=complex)
#     T[:, 0] = 1.0
#     if deg > 1:
#         T[:, 1] = scaled
#     for k in range(2, deg):
#         T[:, k] = 2.0 * scaled * T[:, k-1] - T[:, k-2]

#     # Apply column-wise scaling: column k gets factor (f/cap)^k
#     scale = np.ones(deg)
#     for k in range(1, deg):
#         scale[k] = scale[k-1] * f_cap
#     C = T * scale[np.newaxis, :]
#     return C


# def _faber_mats(M: np.ndarray, c: complex, f: float,
#                 a: float, b: float, deg: int) -> list:
#     """
#     Compute Faber matrix basis  [F_0(M), F_1(M), …, F_{deg−1}(M)]
#     via the constant-coefficient three-term recurrence:

#         G₀ = I
#         G₁ = M − cI
#         G_{k+1} = 2(M − cI) G_k − f² G_{k-1}      (f > 0)

#     These are the MONIC Faber polynomials G_k = f^k T_k((M−cI)/f).
#     The cap-normalised versions F_k = G_k / cap^k match the Vandermonde.

#     One matrix multiply per step; no exponential entry growth from
#     explicit powers.  The evaluation is forward-stable.

#     For f = 0 (disk): degenerates to monomials (M−cI)^k.
#     """
#     N     = M.shape[0]
#     dtype = np.result_type(M.dtype, np.complex128)
#     Mc    = M.astype(dtype) - c * np.eye(N, dtype=dtype)   # M − cI
#     cap   = (a + b) / 2.0

#     mats: list = []
#     if deg == 0:
#         return mats

#     G_prev = np.eye(N, dtype=dtype)    # G_0 = I  (= F_0)
#     mats.append(G_prev.copy())
#     if deg == 1:
#         return mats

#     G_cur = Mc.copy()                  # G_1 = M − cI
#     mats.append(G_cur.copy() / cap)    # store F_1 = G_1 / cap
#     if deg == 2:
#         return mats

#     # f² coefficient for the recurrence
#     f2 = f * f if f >= 1e-10 else 0.0
#     cap_k = cap * cap                  # cap^k running product, starts at cap^2

#     for _ in range(2, deg):
#         G_next = 2.0 * Mc @ G_cur - f2 * G_prev
#         cap_k  *= cap
#         mats.append(G_next / cap_k)    # store cap-normalised F_k = G_k / cap^k
#         G_prev, G_cur = G_cur, G_next

#     return mats


# def _qr_solve(V: np.ndarray, kernel: np.ndarray, rcond: float = 1e-8) -> np.ndarray:
#     """
#     QR-stabilised least-squares  c = argmin ‖V c − kernel‖.

#     V = QR (reduced),  c = R⁻¹ Q^H kernel  (triangular solve + inner products).
#     Rank is detected via diag(R) to handle near-linearly-dependent columns.
#     """
#     Q, R = np.linalg.qr(V, mode='reduced')
#     rhs  = Q.T.conj() @ kernel
#     diag = np.abs(np.diag(R))
#     tol  = rcond * (diag[0] if diag[0] > 0.0 else 1.0)
#     rank = int(np.sum(diag > tol))
#     c    = np.zeros(R.shape[1], dtype=np.complex128)
#     if rank > 0:
#         c[:rank] = scipy.linalg.solve_triangular(R[:rank, :rank], rhs[:rank], lower=False)
#     return c


# # ══════════════════════════════════════════════════════════════════════════════
# # FaberFilter  — Faber on circumscribed ellipse
# # ══════════════════════════════════════════════════════════════════════════════

# class FaberFilter(PolynomialFilter):
#     """
#     Stable polynomial filter for the general complex GSO spectrum, using the
#     Faber polynomial for the circumscribed ellipse of the spectrum.

#     The key property: |F_k(λ)| ≤ 1 for all λ inside the ellipse (by the
#     maximum principle for the conformal map), giving Vandermonde entries
#     bounded by 1 after cap normalisation.  This is what disk-Chebyshev
#     could NOT guarantee (T_k evaluated at complex arguments grows with k).

#     The matrix recurrence  G_{k+1} = 2(M−cI)G_k − f²G_{k-1}  (where
#     G_k = f^k T_k((M−cI)/f) are the monic Faber polynomials) is constant-
#     coefficient and forward-stable: one matrix multiply per step.

#     Degenerate cases:
#         a ≈ b  (circular spectrum): f → 0, Faber → monomials.  No polynomial
#                basis helps — use eigendecomposition-based filtering instead.
#         b = 0  (real spectrum): f = a, Faber = standard Chebyshev T_k.
#     """

#     def __init__(
#         self,
#         graph,
#         name=None,
#         params=None,
#         order: Optional[int] = None,
#         scale_operator: Optional[int] = None,
#     ):
#         # Initialise ellipse params before super().__init__ triggers precompute
#         self._ell_c: complex = 0.0
#         self._ell_a: float   = 1.0
#         self._ell_b: float   = 0.0
#         self._ell_f: float   = 1.0
#         super().__init__(
#             graph, name=name, params=params,
#             order=order, scale_operator=scale_operator,
#         )
#         self.name = "FaberFilter"

#     def precompute_polynomial(self, scale_operator: Optional[int] = None):
#         M     = self.graph.operator.M
#         if scale_operator is not None:
#             M = M / scale_operator
#         order = int(self.params["order"])

#         V = self.graph.operator.V
#         self._ell_c, self._ell_a, self._ell_b, self._ell_f = _ellipse_params(V)

#         # Faber matrix basis (cap-normalised, stored as self.faber_matrices)
#         self.faber_matrices = _faber_mats(
#             M, self._ell_c, self._ell_f, self._ell_a, self._ell_b, order
#         )
#         # Parent-compatible alias so inherited apply() works
#         self.powers_of_M = self.faber_matrices

#         # Faber Vandermonde on eigenvalues (precomputed, used by polynomial_filter)
#         self._V_faber = _faber_vdm(
#             V, self._ell_c, self._ell_f, self._ell_a, self._ell_b, order
#         )

#     def vandermonde_matrix(self, V: np.ndarray, dim: int) -> np.ndarray:
#         """Return precomputed Faber Vandermonde (first dim columns)."""
#         return self._V_faber[:, :dim]

#     def polynomial_filter(
#         self, kernel: np.ndarray, return_coefs: bool = False, rcond: float = 1e-8
#     ) -> np.ndarray:
#         """
#         Fit Faber coefficients via QR-stabilised least-squares, then assemble
#         using the precomputed cap-normalised Faber matrix basis.

#         Two-layer stability:
#           1. Vandermonde bounded (Faber property) → QR solve is well-conditioned.
#           2. Matrix evaluation via recurrence → forward-stable.
#         """
#         assert kernel.ndim == 1, "kernel must be 1D"
#         deg = self.params["order"]

#         # QR-stabilised solve on the bounded Faber Vandermonde
#         c = _qr_solve(self._V_faber, kernel, rcond=rcond)

#         graph_filter = np.sum(
#             [c[i] * self.faber_matrices[i] for i in range(deg)], axis=0
#         )
#         if return_coefs:
#             return graph_filter, c
#         return graph_filter

#     def regression_descent(
#         self,
#         signal,
#         target,
#         deg: Optional[int] = None,
#         n_iter: int = 100,
#         lr: float = 1e-2,
#     ) -> Tuple[np.ndarray, np.ndarray, float]:
#         """Adam descent over Faber coefficients."""
#         if deg is None:
#             deg = int(self.params["order"])
#         dtype  = torch.complex128
#         device = torch.device("cpu")

#         signal_t = torch.tensor(signal, dtype=dtype, device=device)
#         target_t = torch.tensor(target, dtype=dtype, device=device)

#         Msig = torch.stack(
#             [
#                 torch.tensor(self.faber_matrices[i], dtype=dtype, device=device)
#                 .matmul(signal_t)
#                 for i in range(deg)
#             ],
#             dim=0,
#         )
#         coefs     = torch.zeros(deg, dtype=dtype, device=device, requires_grad=True)
#         optim     = torch.optim.Adam([coefs], lr=lr)
#         prev_loss = None

#         for _ in range(n_iter):
#             optim.zero_grad()
#             recon = (coefs.unsqueeze(1) * Msig).sum(dim=0)
#             loss  = torch.mean(torch.abs(recon - target_t) ** 2)
#             loss.backward()
#             optim.step()
#             if prev_loss is not None and abs(prev_loss - loss.item()) < 1e-9:
#                 break
#             prev_loss = loss.item()

#         return recon.detach().numpy(), coefs.detach().numpy(), prev_loss

#     def __repr__(self):
#         f = self._ell_f
#         a, b = self._ell_a, self._ell_b
#         return (
#             f"<Filter(name={self.name}, params={self.params}, "
#             f"ellipse=(a={a:.3g}, b={b:.3g}, f={f:.3g})>"
#         )


"""
faber_filter.py
───────────────
Polynomial graph filter using the Faber polynomial for the circumscribed
ellipse of the operator's spectrum.

Why Faber, not disk-Chebyshev
──────────────────────────────
disk-Chebyshev uses T_k((λ−c)/R) where R = max|λ_n−c|.  For complex
arguments z in the unit disk, T_k(z) = cos(k arccosh z) exhibits cosh-type
growth with k — the disk scaling does NOT give bounded Vandermonde entries.

Faber for the circumscribed ellipse: the conformal map Φ satisfies
|Φ(λ)| ≤ 1 for all λ inside the ellipse.  The Faber polynomial
Φ_k(λ) = Φ(λ)^k|_poly is bounded in magnitude by 2 on the ellipse boundary
(classical result: max_{λ∈∂E} |Φ_k(λ)| ≤ 2), and by the maximum principle
this extends to the interior.  The Vandermonde entries are therefore bounded
by 2 for all eigenvalues inside the circumscribed ellipse — regardless of k.

Recurrence (corrected)
───────────────────────
Let  μ = (M − cI)/cap,  δ_J = (a − b)/(a + b) ∈ [0, 1).

    F_0 = I
    F_1 = μ
    F_2 = μ F_1 − 2δ_J F_0          ← startup: coefficient 2δ_J
    F_{k+1} = μ F_k − δ_J F_{k-1}   ← steady state: coefficient δ_J, k ≥ 2

The asymmetry at startup (2δ_J vs δ_J) follows directly from the
Joukowski parametrisation; it is verified to reproduce Φ_k = Φ^k|_poly
exactly.  Running this recurrence entirely in the normalised domain
avoids the previous bug where intermediate variables held the unnormalised
monic G_k = f^k T_k((M−cI)/f) values (growing as cap^k per step).

Residual limitation
────────────────────
Faber eliminates the Vandermonde conditioning problem in the coefficient
solve.  It does NOT bound ‖F_k(M)‖ for non-normal M: the recurrence is
forward-stable as an algorithm, but if ‖μ‖₂ ≫ ρ(μ) (which occurs when M
is far from normal), the matrix norms can still grow.  This is the
field-of-values obstruction; no polynomial filter basis can remove it.
"""

from __future__ import annotations

from gyraph.utils import np, Optional, Tuple, torch
from .polynomial_filter import PolynomialFilter

import scipy.linalg


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _ellipse_params(eigenvalues: np.ndarray) -> Tuple[complex, float, float, float]:
    """
    Fit the smallest axis-aligned bounding ellipse to the eigenvalue set.
    For a real matrix the eigenvalues come in conjugate pairs, so the
    centre has zero imaginary part.

    Returns (c, a, b, delta_J) where
      c       – centre of the ellipse
      a       – real semi-axis  (horizontal)
      b       – imaginary semi-axis (vertical)
      delta_J – Joukowski parameter (a−b)/(a+b) ∈ [0, 1)
    """
    eps = 1e-12
    c_R = (eigenvalues.real.max() + eigenvalues.real.min()) / 2.0
    c_I = (eigenvalues.imag.max() + eigenvalues.imag.min()) / 2.0
    c = complex(c_R, c_I)
    a = float(eigenvalues.real.max() - c_R) + eps
    b = float(np.abs(eigenvalues.imag).max()) + eps
    delta_J = (a - b) / (a + b)  # ∈ (-1, 1); ≥ 0 when a ≥ b
    return c, a, b, delta_J


def _faber_vdm(
    eigenvalues: np.ndarray, c: complex, cap: float, delta_J: float, deg: int
) -> np.ndarray:
    """
    Faber Vandermonde  V[n, k] = Φ_k(λ_n),  k = 0 … deg−1.

    Scalar recurrence (same coefficients as matrix version):
        v_0 = 1
        v_1 = μ_n  =  (λ_n − c) / cap
        v_2 = μ_n v_1 − 2δ_J
        v_{k+1} = μ_n v_k − δ_J v_{k-1},  k ≥ 2

    Entries are bounded by 2 in magnitude for λ_n inside the ellipse
    (Faber property: max_∂E |Φ_k| ≤ 2 + max principle in interior).
    """
    N = len(eigenvalues)
    mu = (eigenvalues - c) / cap  # (N,) complex
    V = np.zeros((N, deg), dtype=complex)
    if deg == 0:
        return V
    V[:, 0] = 1.0
    if deg == 1:
        return V
    V[:, 1] = mu
    if deg == 2:
        return V
    # k = 1 step: startup coefficient 2δ_J
    V[:, 2] = mu * V[:, 1] - 2.0 * delta_J * V[:, 0]
    # k ≥ 2: steady-state coefficient δ_J
    for k in range(2, deg - 1):
        V[:, k + 1] = mu * V[:, k] - delta_J * V[:, k - 1]
    return V


def _faber_mats(
    M: np.ndarray, c: complex, cap: float, delta_J: float, deg: int
) -> list:
    """
    Faber matrix basis  [F_0(M), F_1(M), …, F_{deg−1}(M)]  in the
    normalised domain throughout.

    Matrix recurrence (runs entirely in normalised variables):
        F_0 = I
        F_1 = μ = (M − cI) / cap
        F_2 = μ F_1 − 2δ_J I          (startup)
        F_{k+1} = μ F_k − δ_J F_{k-1} (k ≥ 2)

    One matrix multiply per step; no intermediate unnormalised values.

    Note: ‖F_k(M)‖ can still grow for non-normal M because ‖μ‖₂ may
    exceed 1 even though all eigenvalues of μ are inside the unit ellipse.
    """
    N = M.shape[0]
    dtype = np.result_type(M.dtype, np.complex128)
    Id = np.eye(N, dtype=dtype)
    mu = (M.astype(dtype) - c * Id) / cap  # normalised operator

    mats: list = []
    if deg == 0:
        return mats

    F_prev = Id.copy()  # F_0
    mats.append(F_prev)
    if deg == 1:
        return mats

    F_cur = mu.copy()  # F_1
    mats.append(F_cur)
    if deg == 2:
        return mats

    # startup: F_2 = μ F_1 − 2δ_J F_0
    F_next = mu @ F_cur - 2.0 * delta_J * F_prev
    mats.append(F_next)
    F_prev, F_cur = F_cur, F_next
    if deg == 3:
        return mats

    # steady state: F_{k+1} = μ F_k − δ_J F_{k-1}, k ≥ 2
    for _ in range(3, deg):
        F_next = mu @ F_cur - delta_J * F_prev
        mats.append(F_next)
        F_prev, F_cur = F_cur, F_next

    return mats


def _qr_solve(V: np.ndarray, kernel: np.ndarray, rcond: float = 1e-8) -> np.ndarray:
    """
    QR-stabilised least-squares  c = argmin ‖Vc − kernel‖.
    V = QR,  c = R⁻¹ Q^H kernel  (triangular solve + inner products).
    """
    Q, R = np.linalg.qr(V, mode="reduced")
    rhs = Q.T.conj() @ kernel
    diag = np.abs(np.diag(R))
    tol = rcond * (diag[0] if diag[0] > 0.0 else 1.0)
    rank = int(np.sum(diag > tol))
    c_out = np.zeros(R.shape[1], dtype=np.complex128)
    if rank > 0:
        c_out[:rank] = scipy.linalg.solve_triangular(
            R[:rank, :rank], rhs[:rank], lower=False
        )
    return c_out


# ══════════════════════════════════════════════════════════════════════════════
# FaberFilter
# ══════════════════════════════════════════════════════════════════════════════


class FaberFilter(PolynomialFilter):
    """
    Polynomial graph filter using the Faber basis for the circumscribed
    ellipse of the operator spectrum.

    Stability properties
    ────────────────────
    (1) Vandermonde conditioning (fixed):
        V[n, k] = Φ_k(λ_n) is bounded by 2 for all λ_n inside the
        circumscribed ellipse (Faber property + maximum principle).
        The QR-stabilised solve then has good conditioning.

    (2) Matrix evaluation (improved, not fully fixed):
        The recurrence runs in the normalised domain μ = (M−cI)/cap,
        avoiding the previous bug where unnormalised G_k grew as cap^k.
        However ‖F_k(M)‖ can still grow when ‖μ‖₂ > 1, which occurs
        whenever M is non-normal (spectral norm exceeds spectral radius).
        This is the field-of-values obstruction; it is intrinsic to all
        polynomial filters on non-normal operators.
    """

    def __init__(
        self,
        graph,
        name=None,
        params=None,
        order: Optional[int] = None,
        scale_operator: Optional[int] = None,
    ):
        self._c: complex = 0.0
        self._cap: float = 1.0
        self._delta_J: float = 0.0
        super().__init__(
            graph,
            name=name,
            params=params,
            order=order,
            scale_operator=scale_operator,
        )
        self.name = "FaberFilter"

    # ── overrides ─────────────────────────────────────────────────────────────

    def precompute_polynomial(self, scale_operator: Optional[int] = None):
        M = self.graph.operator.M
        if scale_operator is not None:
            M = M / scale_operator
        order = int(self.params["order"])

        V = self.graph.operator.V
        self._c, a, b, self._delta_J = _ellipse_params(V)
        self._cap = (a + b) / 2.0

        self.faber_matrices = _faber_mats(M, self._c, self._cap, self._delta_J, order)
        self.powers_of_M = self.faber_matrices  # parent-compatible alias

        self._V_faber = _faber_vdm(V, self._c, self._cap, self._delta_J, order)

    def vandermonde_matrix(self, V: np.ndarray, dim: int) -> np.ndarray:
        return self._V_faber[:, :dim]

    def polynomial_filter(
        self, kernel: np.ndarray, return_coefs: bool = False, rcond: float = 1e-8
    ) -> np.ndarray:
        assert kernel.ndim == 1, "kernel must be 1D"
        deg = self.params["order"]
        c = _qr_solve(self._V_faber, kernel, rcond=rcond)
        graph_filter = np.sum(
            [c[i] * self.faber_matrices[i] for i in range(deg)], axis=0
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
        if deg is None:
            deg = int(self.params["order"])
        dtype = torch.complex128
        device = torch.device("cpu")

        signal_t = torch.tensor(signal, dtype=dtype, device=device)
        target_t = torch.tensor(target, dtype=dtype, device=device)

        Msig = torch.stack(
            [
                torch.tensor(self.faber_matrices[i], dtype=dtype, device=device).matmul(
                    signal_t
                )
                for i in range(deg)
            ],
            dim=0,
        )
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
        return (
            f"<Filter(name={self.name}, params={self.params}, "
            f"cap={self._cap: .3g}, delta_J={self._delta_J: .3g})>"
        )
