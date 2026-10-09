"""Finite-alpha quartic kernel K_alpha(s) on the D'Hoker-Kraus magnetic brane
(paper sec. 5.1 and app. C).

For each beta = alpha B^2 on the bc_alpha.npz grid (B = B_c(beta), background
dk_background.solve_colloc, self-consistent zero mode w0(beta), brane norm
J = int p1 w0^2 dr = 1) the kernel per reciprocal-lattice harmonic s = G^2 is
the derived on-shell half-pairing (paper app. C.3)

    K(s; beta) = C4 - Re[X_c](s; beta) - (1/2) Pi_bare(s; beta) ,

  * C4 = int p1 w0^4 dr (p1 = e^{W-2V}, p2 = U e^W; derivations/dk_quartic.py).
  * X_c = int p1 b_c w0^2 dr with b_c = iG a_y the COUPLED photon of the
    (H_yy, H_zz, a_y) + algebraic (H_rr, H_xr) response (systems/dk_response.py,
    true-Ricci rows).  At beta = 0, X_c -> X (decoupled bhat
    BVP) and K -> C4 - X (the probe limit).
  * Pi_bare = Re int sqrt(-g) S^{MN}[BH=0] h*_MN dr: the BARE (w0-bilinear)
    condensate stress paired with the coupled metric response, at the derived
    weight 1/2 (h_yy = e^{2V}H_yy, h_zz = e^{2W}H_zz, h_rr = H_rr/U,
    h_xr = i e^{2V}H_xr; conjugation on h).

DERIVATION (derivations/dk_pairing.py, paper app. C.3 -- nothing fitted): the
graded quadratic action of the coupled (h, a_y) sector on the DK background,
EL-locked row by row to the linearised system (k = -1, c = 1, k_a = -2 alpha,
cJ = 2 alpha), gives on shell E4 = contact + (1/2)(source pairings) + flux,
with the flux Theta measured to vanish at both ends on the solutions.  In
code objects that is exactly the formula above (NORM = -1/(2 alpha) is
anchored by the beta -> 0 photon sector, so the beta -> 0 limit is the probe limit by
construction, and the bare-stress weight -1/2 is NORM * (1/2)L_S with the
c = 1 lock).

CONVENTIONS fixed by the derivation: the response rows are the linearisation
of the true brane Ricci tensor (not a flat-space port of it), so the (tt) and
(xx) rows are algebraically redundant (app. C.3); and the metric exchange
enters the kernel at weight 1/2, Delta K = -(alpha/2) Gcal per harmonic, which
is also what gauge invariance requires (numerics/dk_kernel_gauge_check.py).

Anchors (asserted before the scan):
  [1] beta = 0: (C4 - X)(s) reproduces the probe kernel scripts/kernel.npz
      to 3e-12 on the full G2grid (paper app. D.4; measured 2.6e-12);
      C4 = 1.58059313; exchange pieces vanish identically.
  [2] (no anchor [2] here) the weight 1/2 and the response normalisation are
      tested by derivations/paper_response_display.py and
      dk_kernel_gauge_check.py, and the strict G = 0 limit by dk_uniform.py.
  [3] The two-source pairing at finite beta: the code objects 2 Re[X_cross] +
      Pi_bare equal the derived source halves of derivations/dk_pairing.py [9],
      and the Green flux vanishes at both ends (asserted at 1e-9).  The swap
      symmetry itself is NOT tested non-trivially: the only admissible second
      source is a multiple of w0 (see anchor3).
  [4] Two methods: Chebyshev collocation (N = 64, spot-checked N = 96) vs
      2nd-order uniform-sigma finite differences at M, 2M with Richardson;
      worst-case |Delta K| quoted over the whole scan.
  [5] The dropped flux: NORM*(1/2)*Theta on the solutions is <= 7e-10 of K
      at the horizon and falls off as sigma^2 at the boundary, on every
      non-probe brane at four harmonics (quoted in paper app. D.4 as below 1e-9 of
      the kernel; asserted at that level).
Constraint health: the emitted closed system carries H_rr', H_xr' as
independently eliminated slots; on exact solutions d(H_rr)/dr = slot
identically (the 2x2 (rr),(xr) elimination is non-degenerate), so the defect
max|d(H_rr)/dr - slot| / max|slot| is a pure numerics monitor (the raw
(tt),(xx) rows are not emitted by systems/dk_response.py).  Stored per point.

FRAME.  sigma = 1/r in [0,1], sigma = 1 horizon, sigma = 0 boundary; brane
measure dr = dsigma/sigma^2.  d/dr = -sigma^2 d/dsigma.  Boundary conditions:
H_yy = H_zz = a_y = 0 at sigma = 0 (normalisable; indicial {0,4}/{0,2}),
degenerate U-row regularity at sigma = 1 (each field's own slope fixed).

Run:  uv run python -m backreaction.numerics.dk_kernel            # full
      uv run python -m backreaction.numerics.dk_kernel --anchors  # anchors only
"""

import sys
import time

import numpy as np
from scipy.linalg import solve
from scipy.sparse import coo_matrix, diags, lil_matrix
from scipy.sparse.linalg import spsolve

import script_paths
from backreaction import paths
from backreaction.numerics import dk_background as dk
from backreaction.systems import dk_response as resp
from backreaction.systems import dk_pairing as dps

BCA_PATH = paths.data("bc_alpha.npz")
KER_PATH = script_paths.data("kernel.npz")  # main tree, not the package
OUT_PATH = paths.data("dk_kernel.npz")
FIG_PATH = paths.figure("dk_kernel.png")

# fixed relative s-grid: s = gamma * B_c(beta), dense at small s (the Gcal-type
# structure is steep below s ~ B_c/2, where the long-wavelength density mode
# and the complete-monotonicity tests live)
# a parameter grid: one value per line would hide its shape
# fmt: off
GAMMA_GRID = np.array([
    0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.90,
    1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
    9.0, 10.0, 11.0, 12.5, 14.0, 16.0, 18.0, 20.0, 23.0, 26.0, 30.0, 35.0,
    40.0, 45.0])
# fmt: on


def tlog(msg):
    """Timestamped progress line (the full scan runs a few minutes)."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cc_weights_01(N):
    """Clenshaw-Curtis weights for int_0^1 f dsigma on Lobatto nodes."""
    th = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    for j in range(N + 1):
        acc = sum(
            (1.0 if 2 * k == N else 2.0) / (4 * k * k - 1) * np.cos(2 * k * th[j])
            for k in range(1, N // 2 + 1)
        )
        w[j] = 1.0 - acc
    w *= 2.0 / N
    w[0] /= 2.0
    w[-1] /= 2.0
    return w * 0.5


def pair_density(U, V, W, Bc, G, w0, w0p, bh, bhp, Hyy, Hzz, Hrr, Hxr):
    """Re[ sqrt(-g) S^{MN}[bhat] h*_MN ] with the systems/dk_response.py field
    conventions; the rr term uses the pole-free product U*S_rr explicitly.
    All args arrays over nodes (U may vanish at the horizon)."""
    e2V, e2W = np.exp(2 * V), np.exp(2 * W)
    sqrtg = np.exp(2 * V + W)
    Syy = Bc * (bh - w0**2) / e2V
    Szz = (-Bc * bh + Bc * w0**2 - U * w0p**2 * e2V) * e2W / e2V**2
    USrr = (Bc * (w0**2 - bh) + U * w0p**2 * e2V) / e2V**2
    # 2 g^xx g^rr S_xr conj(h_xr) = 2 e^{-2V} U (-i Bc bhp e^{-2V}/G)
    #                               conj(i e^{2V} Hxr) = -2 U Bc bhp Hxr*/(G e2V)
    dens = sqrtg * (
        Syy * np.conj(Hyy) / e2V
        + Szz * np.conj(Hzz) / e2W**2 * e2W
        + USrr * np.conj(Hrr)
        - 2.0 * U * Bc * bhp * np.conj(Hxr) / (G * e2V)
    )
    return dens


class Brane:
    """DK background + self-consistent J-normalised w0 at one beta, with the
    abelian and coupled-response BVPs on the Chebyshev sigma grid."""

    def __init__(self, beta, Bc, w0_nodal, N=64):
        self.beta, self.Bc, self.N = float(beta), float(Bc), N
        self.alpha = 0.0 if beta == 0.0 else beta / self.Bc**2
        self.bg = dk.solve_colloc(beta, N=N, tol=1e-12, maxit=80, require=True)
        self.sig = self.bg.sig  # sig[0]=1 hor, sig[N]=0 bnd
        self.D = self.bg.D
        self.D2 = self.D @ self.D
        self.ccw = cc_weights_01(N)
        self.P1 = np.exp(self.bg.Nn - 2 * self.bg.M)  # p1 = sigma * P1
        self.P2 = self.bg.A * np.exp(self.bg.Nn)  # p2 = P2 / sigma^3
        self.P2s = self.D @ self.P2
        self.w0 = np.asarray(w0_nodal, float)
        self.w0p = -(self.sig**2) * (self.D @ self.w0)  # dw0/dr
        # background fields, vectorised once (interior + horizon; bnd masked)
        self.flds = self._fields_on(self.sig[:-1])
        self._eps = 1e-6

    def _fields_on(self, sg):
        f = self.bg.fields_sigma(np.asarray(sg, float))
        return {k: np.real(np.asarray(f[k])) for k in ("U", "Up", "V", "Vp", "W", "Wp")}

    def integ_dr(self, g):
        out = np.zeros(self.N + 1, dtype=complex)
        m = self.sig > 1e-13
        out[m] = np.asarray(g)[m] / self.sig[m] ** 2
        return complex(self.ccw @ out)

    # ---- abelian sector -------------------------------------------------
    def bhat(self, s, w0sq=None):
        """Decoupled induced-field BVP, sigma frame, pole-free operators:
        -sigma P2 b'' + (P2 - sigma P2') b' + s sigma P1 b = s sigma P1 w0^2;
        Dirichlet b(0) = 0, degenerate (P2(1)=0) horizon row = regularity."""
        sig, D, D2 = self.sig, self.D, self.D2
        if w0sq is None:
            w0sq = self.w0**2
        L = (
            np.diag(-sig * self.P2) @ D2
            + np.diag(self.P2 - sig * self.P2s) @ D
            + np.diag(s * sig * self.P1)
        )
        rhs = s * sig * self.P1 * np.asarray(w0sq, float)
        ib = self.N
        L[ib, :] = 0.0
        L[ib, ib] = 1.0
        rhs[ib] = 0.0
        return solve(L, rhs)

    def photon_kernel(self, s):
        b = self.bhat(s)
        C4 = self.integ_dr(self.P1 * self.sig * self.w0**4).real
        X = self.integ_dr(self.P1 * self.sig * b * self.w0**2).real
        return C4 - X, C4, X, b

    # ---- coupled (H_yy, H_zz, a_y) response ------------------------------
    def _X7(self, sk, s, w0v, w0pv):
        f = self.bg.fields_sigma([sk])
        return resp._solve(
            1.0 / sk,
            np.real(f["U"][0]),
            np.real(f["Up"][0]),
            np.real(f["V"][0]),
            np.real(f["Vp"][0]),
            np.real(f["W"][0]),
            np.real(f["Wp"][0]),
            self.Bc,
            np.sqrt(s),
            self.alpha,
            w0v,
            w0pv,
        )

    def coupled_solve(self, s, w0=None):
        """Chebyshev collocation of the closed system.  Interior rows are the
        emitted evolution rows scaled by sigma^4 U (pole-free); the horizon row
        is the degenerate lim U*(row) regularity relation (Richardson in an
        eps = 1e-6 offset); boundary rows are Dirichlet."""
        N, sig, D, D2 = self.N, self.sig, self.D, self.D2
        w0 = self.w0 if w0 is None else np.asarray(w0, float)
        w0p = -(sig**2) * (D @ w0)
        npts = N + 1
        A = np.zeros((3 * npts, 3 * npts), dtype=complex)
        rhs = np.zeros(3 * npts, dtype=complex)
        X7s = [None] * npts
        eps = self._eps
        for k in range(npts):
            sk = sig[k]
            if sk < 1e-13:  # boundary Dirichlet
                for i in range(3):
                    A[i * npts + k, i * npts + k] = 1.0
                continue
            if sk > 1.0 - 1e-13:  # horizon
                s1, s2 = 1.0 - eps, 1.0 - 2 * eps
                w1 = float(self.bg._bary(w0, [s1])[0].real)
                w1p = float(self.bg._bary(w0p, [s1])[0].real)
                w2 = float(self.bg._bary(w0, [s2])[0].real)
                w2p = float(self.bg._bary(w0p, [s2])[0].real)
                X71 = self._X7(s1, s, w1, w1p)
                X72 = self._X7(s2, s, w2, w2p)
                U1 = float(np.real(self.bg.fields_sigma([s1])["U"][0]))
                U2 = float(np.real(self.bg.fields_sigma([s2])["U"][0]))
                Xu = 2.0 * U1 * X71 - U2 * X72  # lim U * coeffs
                X7s[k] = 2.0 * X71 - X72  # finite algebraic slots
                for i, slot in enumerate((4, 5, 6)):
                    row = i * npts + k
                    for j in range(3):
                        A[row, j * npts + k] += Xu[slot][2 * j]
                        A[row, j * npts : (j + 1) * npts] += (
                            Xu[slot][2 * j + 1] * (-(sk**2)) * D[k]
                        )
                    rhs[row] = -Xu[slot][6]
                continue
            X7 = self._X7(sk, s, float(w0[k]), float(w0p[k]))
            X7s[k] = X7
            Uk = self.flds["U"][k]
            wgt = sk**4 * Uk
            for i, slot in enumerate((4, 5, 6)):
                row = i * npts + k
                A[row, i * npts : (i + 1) * npts] += wgt * (
                    sk**4 * D2[k] + 2 * sk**3 * D[k]
                )
                for j in range(3):
                    A[row, j * npts + k] += -wgt * X7[slot][2 * j]
                    A[row, j * npts : (j + 1) * npts] += (
                        -wgt * X7[slot][2 * j + 1] * (-(sk**2)) * D[k]
                    )
                rhs[row] = wgt * X7[slot][6]
        # max-equilibrate before the LU: the raw sigma^4 U (evolution) vs O(1)
        # (Dirichlet/horizon) row weights span ~16 decades and cond(A) grows
        # past 1e19 by N = 96, while the equilibrated matrix sits at cond
        # ~1e2-1e3 -- the scaling, not the system, was the digit killer
        rn = np.max(np.abs(A), axis=1)
        rn[rn == 0.0] = 1.0
        A /= rn[:, None]
        cn = np.max(np.abs(A), axis=0)
        cn[cn == 0.0] = 1.0
        A /= cn[None, :]
        sol = np.linalg.solve(A, rhs / rn) / cn
        H = sol.reshape(3, npts)  # (H_yy, H_zz, a_y)
        dHr = np.array([-(sig**2) * (D @ H[i]) for i in range(3)])
        Hrr = np.zeros(npts, dtype=complex)
        Hxr = np.zeros(npts, dtype=complex)
        Hrrp = np.zeros(npts, dtype=complex)
        Hxrp = np.zeros(npts, dtype=complex)
        for k in range(npts):
            if X7s[k] is None:
                continue
            vec = np.array(
                [H[0][k], dHr[0][k], H[1][k], dHr[1][k], H[2][k], dHr[2][k], 1.0]
            )
            Hrr[k], Hxr[k] = X7s[k][0] @ vec, X7s[k][1] @ vec
            Hrrp[k], Hxrp[k] = X7s[k][2] @ vec, X7s[k][3] @ vec
        # elimination-consistency defect (numerics health; see module docstring)
        m = (sig > 5e-2) & (sig < 1.0 - 1e-13)
        dHrr = -(sig**2) * (D @ Hrr)
        dHxr = -(sig**2) * (D @ Hxr)
        d1 = np.max(np.abs(dHrr[m] - Hrrp[m])) / max(np.max(np.abs(Hrrp[m])), 1e-300)
        d2 = np.max(np.abs(dHxr[m] - Hxrp[m])) / max(np.max(np.abs(Hxrp[m])), 1e-300)
        return dict(
            H=H,
            dHr=dHr,
            Hrr=Hrr,
            Hxr=Hxr,
            defect=(float(d1), float(d2)),
            w0=w0,
            w0p=w0p,
        )

    # ---- the exchange pairings and the kernel -----------------------------
    def exchange(self, s, sol=None, b=None, bp=None):
        """All pairing integrals of the paper sec. 5.1 and app. C exact kernel on the
        coupled response.  Returns (Pib, Xc, P, diag):
          Pib = Re int sqrt(-g) S^{MN}[BH=0] h*_MN dr   (bare-stress pairing)
          Xc  = int p1 b_c w0^2 dr, b_c = iG a_y        (coupled photon)
          P   = the dressed pairing (BH = decoupled bhat) -- diagnostic only:
                its O(alpha) limit is alpha*Gcal (anchor [2] dP/da check)."""
        G = np.sqrt(s)
        if sol is None:
            sol = self.coupled_solve(s)
        if b is None:
            b = self.bhat(s)
            bp = -(self.sig**2) * (self.D @ b)
        Hyy, Hzz, ay = sol["H"]
        n = self.N
        f = self.flds
        w0i, w0pi = sol["w0"][:-1], sol["w0p"][:-1]

        def pairing(bh_arr, bhp_arr):
            dens = pair_density(
                f["U"],
                f["V"],
                f["W"],
                self.Bc,
                G,
                w0i,
                np.real(w0pi),
                bh_arr,
                bhp_arr,
                Hyy[:-1],
                Hzz[:-1],
                sol["Hrr"][:-1],
                sol["Hxr"][:-1],
            )
            full = np.zeros(n + 1, dtype=complex)
            full[:-1] = dens
            return self.integ_dr(full)

        P = pairing(b[:-1], bp[:-1])
        zer = np.zeros(n, dtype=complex)
        Pib = pairing(zer, zer)
        bc = 1j * G * ay
        Xc = self.integ_dr(self.P1 * self.sig * bc * self.w0**2)
        return Pib, Xc, P, dict(defect=sol["defect"])

    def kernel(self, s):
        """paper sec. 5.1 and app. C exact kernel  K = C4 - Re[X_c] - (1/2) Pi_bare."""
        cx, C4, X, b = self.photon_kernel(s)
        bp = -(self.sig**2) * (self.D @ b)
        if self.beta == 0.0:
            return dict(
                K=cx, C4=C4, X=X, P=0.0, Pib=0.0, Xc=X, defect=(0.0, 0.0), imag=0.0
            )
        sol = self.coupled_solve(s)
        Pib, Xc, P, diag = self.exchange(s, sol=sol, b=b, bp=bp)
        K = C4 - Xc.real - 0.5 * Pib.real
        return dict(
            K=K,
            C4=C4,
            X=X,
            P=P.real,
            Pib=Pib.real,
            Xc=Xc.real,
            defect=diag["defect"],
            imag=max(abs(Pib.imag), abs(Xc.imag)),
        )


# ---------------------------------------------------------------------------
# independent method: 2nd-order finite differences on a uniform sigma grid
# ---------------------------------------------------------------------------
class BraneFD:
    """Same BVPs discretised with 3-point finite differences on a uniform
    sigma grid (M+1 points); w0 interpolated from the Chebyshev cache.
    Combine two resolutions with Richardson: K_R = (4 K_{2M} - K_M)/3.
    Both linear systems are assembled sparse and solved by SuperLU (table 4; the
    dense solve made M > 1600 impractical), with the same max-equilibration."""

    def __init__(self, br, M):
        self.br, self.M = br, M
        self.sig = np.linspace(0.0, 1.0, M + 1)  # sig[0]=0 bnd, sig[M]=1 hor
        h = self.sig[1] - self.sig[0]
        self.h = h
        self.w0 = np.real(br.bg._bary(br.w0, self.sig))
        # FD first derivative (interior central, one-sided ends)
        M1 = M + 1
        Dm = np.zeros((M1, M1))
        for k in range(1, M):
            Dm[k, k - 1], Dm[k, k + 1] = -0.5 / h, 0.5 / h
        Dm[0, :3] = np.array([-1.5, 2.0, -0.5]) / h
        Dm[M, -3:] = np.array([0.5, -2.0, 1.5]) / h
        self.Dm = Dm
        self.w0p = -(self.sig**2) * (Dm @ self.w0)  # dw0/dr
        self.flds = br._fields_on(self.sig[1:])  # exclude boundary node
        self.trap = np.full(M1, h)
        self.trap[0] = self.trap[-1] = h / 2.0

    def integ_dr(self, g):
        out = np.zeros(self.M + 1, dtype=complex)
        m = self.sig > 1e-13
        out[m] = np.asarray(g)[m] / self.sig[m] ** 2
        return complex(self.trap @ out)

    def bhat(self, s):
        M, sig, h = self.M, self.sig, self.h
        P1 = np.zeros(M + 1)
        P2 = np.zeros(M + 1)
        f = self.flds
        P1[1:] = np.exp(f["W"] - 2 * f["V"]) * sig[1:]  # p1 = e^{W-2V}
        P2[1:] = f["U"] * np.exp(f["W"]) * sig[1:] ** 3  # p2 = U e^W
        P1[0], P2[0] = 0.0, 0.0
        # -(p2 b')' + s p1 (b - w0^2) = 0 in r; in sigma with d/dr = -s^2 d/ds:
        # -(sig^2 p2 (sig^2 b')')... use the pole-free sigma form of Brane.bhat:
        # -sig P2h b'' + (P2h - sig P2h') b' + s sig P1h b = s sig P1h w0^2,
        # P1h = p1/sig, P2h = p2 sig^3.
        P1h = np.zeros(M + 1)
        P2h = np.zeros(M + 1)
        P1h[1:] = np.exp(f["W"] - 2 * f["V"]) / sig[1:]
        P2h[1:] = f["U"] * np.exp(f["W"]) * sig[1:] ** 3
        # boundary values by extrapolation (smooth, = 1 at sigma=0)
        P1h[0] = 2 * P1h[1] - P1h[2]
        P2h[0] = 2 * P2h[1] - P2h[2]
        dP2h = self.Dm @ P2h
        L = lil_matrix((M + 1, M + 1))
        rhs = np.zeros(M + 1)
        for k in range(1, M):
            c2 = -sig[k] * P2h[k]
            c1 = P2h[k] - sig[k] * dP2h[k]
            L[k, k - 1] += c2 / h**2 - 0.5 * c1 / h
            L[k, k] += -2 * c2 / h**2 + s * sig[k] * P1h[k]
            L[k, k + 1] += c2 / h**2 + 0.5 * c1 / h
            rhs[k] = s * sig[k] * P1h[k] * self.w0[k] ** 2
        L[0, 0] = 1.0  # Dirichlet at boundary
        # horizon: degenerate row (P2h(1) = 0): (P2h - sig dP2h) b' + s P1 b = src
        c1 = P2h[M] - sig[M] * dP2h[M]
        for q, cv in zip((M - 2, M - 1, M), (0.5, -2.0, 1.5), strict=True):
            L[M, q] += c1 * cv / h
        L[M, M] += s * sig[M] * P1h[M]
        rhs[M] = s * sig[M] * P1h[M] * self.w0[M] ** 2
        return spsolve(L.tocsc(), rhs)

    def kernel(self, s):
        br, M, sig, h = self.br, self.M, self.sig, self.h
        b = self.bhat(s)
        P1full = np.zeros(M + 1)
        P1full[1:] = np.exp(self.flds["W"] - 2 * self.flds["V"])
        C4 = self.integ_dr(P1full * self.w0**4).real
        X = self.integ_dr(P1full * b * self.w0**2).real
        cx = C4 - X
        if br.beta == 0.0:
            return dict(K=cx, C4=C4, X=X, P=0.0)
        # coupled solve, FD
        npts = M + 1
        Ar, Ac, Av = [], [], []  # sparse triplets; duplicates are summed

        def add(r, c, v):
            Ar.append(r)
            Ac.append(c)
            Av.append(v)

        rhs = np.zeros(3 * npts, dtype=complex)
        G = np.sqrt(s)
        X7s = [None] * npts
        eps = 1e-6
        stenc_c = {-1: 1.0 / h**2, 0: -2.0 / h**2, 1: 1.0 / h**2}
        for k in range(npts):
            sk = sig[k]
            if k == 0:
                for i in range(3):
                    add(i * npts, i * npts, 1.0)
                continue
            if k == M:
                s1, s2 = 1.0 - eps, 1.0 - 2 * eps
                w1 = float(br.bg._bary(br.w0, [s1])[0].real)
                w1p = float((-(s1**2)) * br.bg._bary(br.D @ br.w0, [s1])[0].real)
                w2 = float(br.bg._bary(br.w0, [s2])[0].real)
                w2p = float((-(s2**2)) * br.bg._bary(br.D @ br.w0, [s2])[0].real)
                X71 = br._X7(s1, s, w1, w1p)
                X72 = br._X7(s2, s, w2, w2p)
                U1 = float(np.real(br.bg.fields_sigma([s1])["U"][0]))
                U2 = float(np.real(br.bg.fields_sigma([s2])["U"][0]))
                Xu = 2.0 * U1 * X71 - U2 * X72
                X7s[k] = 2.0 * X71 - X72
                one_sided = np.array([0.5, -2.0, 1.5]) / h  # d/dsigma at end
                for i, slot in enumerate((4, 5, 6)):
                    row = i * npts + k
                    for j in range(3):
                        add(row, j * npts + k, Xu[slot][2 * j])
                        for q in range(3):
                            add(
                                row,
                                j * npts + k - 2 + q,
                                Xu[slot][2 * j + 1] * (-(sk**2)) * one_sided[q],
                            )
                    rhs[row] = -Xu[slot][6]
                continue
            fk = {q: self.flds[q][k - 1] for q in self.flds}
            X7 = resp._solve(
                1.0 / sk,
                fk["U"],
                fk["Up"],
                fk["V"],
                fk["Vp"],
                fk["W"],
                fk["Wp"],
                br.Bc,
                G,
                br.alpha,
                float(self.w0[k]),
                float(np.real(self.w0p[k])),
            )
            X7s[k] = X7
            wgt = sk**4 * fk["U"]
            for i, slot in enumerate((4, 5, 6)):
                row = i * npts + k
                for dj, cv in stenc_c.items():
                    add(row, i * npts + k + dj, wgt * sk**4 * cv)
                add(row, i * npts + k - 1, wgt * 2 * sk**3 * (-0.5 / h))
                add(row, i * npts + k + 1, wgt * 2 * sk**3 * (0.5 / h))
                for j in range(3):
                    add(row, j * npts + k, -wgt * X7[slot][2 * j])
                    cd = -wgt * X7[slot][2 * j + 1] * (-(sk**2))
                    add(row, j * npts + k - 1, cd * (-0.5 / h))
                    add(row, j * npts + k + 1, cd * (0.5 / h))
                rhs[row] = wgt * X7[slot][6]
        A = coo_matrix(
            (np.array(Av, dtype=complex), (Ar, Ac)), shape=(3 * npts, 3 * npts)
        ).tocsr()
        # same max-equilibration as Brane.coupled_solve (same scaling disease)
        rn = abs(A).max(axis=1).toarray().ravel()
        rn[rn == 0.0] = 1.0
        A = diags(1.0 / rn) @ A
        cn = abs(A).max(axis=0).toarray().ravel()
        cn[cn == 0.0] = 1.0
        A = A @ diags(1.0 / cn)
        solv = spsolve(A.tocsc(), rhs / rn) / cn
        H = solv.reshape(3, npts)
        dHr = np.array([-(sig**2) * (self.Dm @ H[i]) for i in range(3)])
        Hrr = np.zeros(npts, dtype=complex)
        Hxr = np.zeros(npts, dtype=complex)
        for k in range(npts):
            if X7s[k] is None:
                continue
            vec = np.array(
                [H[0][k], dHr[0][k], H[1][k], dHr[1][k], H[2][k], dHr[2][k], 1.0]
            )
            Hrr[k], Hxr[k] = X7s[k][0] @ vec, X7s[k][1] @ vec
        f = self.flds
        zer = np.zeros(npts - 1, dtype=complex)
        dens = pair_density(
            f["U"],
            f["V"],
            f["W"],
            br.Bc,
            G,
            self.w0[1:],
            np.real(self.w0p[1:]),
            zer,
            zer,
            H[0][1:],
            H[1][1:],
            Hrr[1:],
            Hxr[1:],
        )
        full = np.zeros(npts, dtype=complex)
        full[1:] = dens
        Pib = self.integ_dr(full).real
        bc = 1j * G * H[2]
        Xc = self.integ_dr(P1full * bc * self.w0**2).real
        K = C4 - Xc - 0.5 * Pib
        return dict(K=K, C4=C4, X=X, Pib=Pib, Xc=Xc)


def fd_richardson(br, s, M=240):
    kM = BraneFD(br, M).kernel(s)
    k2 = BraneFD(br, 2 * M).kernel(s)
    qs = ("K", "C4", "X", "Pib", "Xc") if br.beta != 0.0 else ("K", "C4", "X")
    return {q: (4.0 * k2[q] - kM[q]) / 3.0 for q in qs}


# ---------------------------------------------------------------------------
# anchors
# ---------------------------------------------------------------------------
def load_cache():
    bca = np.load(BCA_PATH)
    return bca


def make_brane(bca, i, N=64):
    return Brane(
        float(bca["beta"][i]),
        float(bca["B_c"][i]),
        np.asarray(bca["w0"][i], float),
        N=N,
    )


def anchor1(br0, tol=1e-11):
    # Measured 2.6e-12 on the reference platform (macOS arm64; app. D.4 quotes
    # 3e-12).  The tolerance leaves ~4x for other BLAS/CPU rounding.
    ker = np.load(KER_PATH)
    C4_ref = float(ker["C4"])
    G2grid, Xref = ker["G2grid"], ker["X"]
    C4_me = br0.integ_dr(br0.P1 * br0.sig * br0.w0**4).real
    err = 0.0
    for i, s in enumerate(G2grid):
        if s == 0.0:
            continue
        cx, *_ = br0.photon_kernel(float(s))
        err = max(err, abs(cx - (C4_ref - Xref[i])))
    J = br0.integ_dr(br0.P1 * br0.sig * br0.w0**2).real
    tlog(
        f"[1] beta=0 vs kernel.npz: J={J:.10f}, C4={C4_me:.8f} vs {C4_ref:.8f}"
        f" (d {abs(C4_me - C4_ref):.1e}); max|dK| = {err:.2e} over "
        f"{len(G2grid)} points (tol {tol:.0e})"
    )
    assert err <= tol and abs(C4_me - C4_ref) <= tol, "anchor 1 FAILED"
    return err


# ---------------------------------------------------------------------------
# the two-source (polarised) pairing objects of derivations/dk_pairing.py [9]
# ---------------------------------------------------------------------------
_BGQ = ("U", "Up", "V", "Vp", "W", "Wp")


def _slotvec(sol):
    """(H_yy, H_yy', H_zz, H_zz', H_rr, H_xr, a_y, a_y') on the sigma grid."""
    return (
        sol["H"][0],
        sol["dHr"][0],
        sol["H"][1],
        sol["dHr"][1],
        sol["Hrr"],
        sol["Hxr"],
        sol["H"][2],
        sol["dHr"][2],
    )


def two_source_objects(br, s, solA, solB, wsrcA, wsrcB):
    """The polarised objects derived in derivations/dk_pairing.py [9], on the cross configuration
    (H-slots = response A, K-slots = conj(response B)):

        src_k = the K-half of L_S + L_J carrying SOURCE A against RESPONSE B,
        src_h = the H-half carrying SOURCE B against RESPONSE A,
        W     = Theta_K - Theta_H, the Lagrange/Green bilinear concomitant,

    with the exact identity  int src_k - int src_h = [W]_hor^bnd  and, in
    kernel units, R = (1/alpha) int src.  Returns (R_AB, R_BA, W_hor, W_bnd).
    """
    n, sig, D, al = br.N, br.sig, br.D, br.alpha
    # The grid includes the boundary node sigma = 0, where U ~ 1/sigma^2 and
    # V, W ~ -log(sigma) are infinite; the loop below skips that node, so the
    # divide-by-zero there is expected and its inf values are never read.
    with np.errstate(divide="ignore"):
        fl = br.bg.fields_sigma(sig)
    f = {q: np.real(np.asarray(fl[q])) for q in _BGQ}
    G = np.sqrt(s)
    A, Bv = _slotvec(solA), _slotvec(solB)
    wAp = np.real(-(sig**2) * (D @ wsrcA))
    wBp = np.real(-(sig**2) * (D @ wsrcB))
    sk = np.zeros(n + 1, dtype=complex)
    sh = np.zeros(n + 1, dtype=complex)
    Wg = np.zeros(n + 1, dtype=complex)
    Z = 0.0
    for k in range(n + 1):
        if sig[k] <= 1e-13:
            continue
        a = [v[k] for v in A]
        b = [np.conj(v[k]) for v in Bv]
        slots = (
            a[0],
            a[1],
            Z,
            a[2],
            a[3],
            Z,
            a[4],
            Z,
            Z,
            a[5],
            Z,
            Z,
            a[6],
            a[7],
            Z,
            b[0],
            b[1],
            Z,
            b[2],
            b[3],
            Z,
            b[4],
            Z,
            Z,
            b[5],
            Z,
            Z,
            b[6],
            b[7],
            Z,
        )
        bg = (1.0 / sig[k],) + tuple(f[q][k] for q in _BGQ) + (br.Bc, G, al)
        sk[k] = dps.src_k(*bg, wsrcA[k], wAp[k], *slots)
        sh[k] = dps.src_h(*bg, wsrcB[k], wBp[k], *slots)
        Wg[k] = dps.green_flux(*bg, 0.0, 0.0, *slots)
    R_AB = br.integ_dr(sk).real / al
    R_BA = br.integ_dr(sh).real / al
    return R_AB, R_BA, Wg[0].real / al, Wg[n - 1].real / al


def anchor3(bca, s_list=(20.0, 60.0), tol=1e-9):
    """The derived polarised two-source pairing identity (derivations/dk_pairing.py [9]).

    Plain Green reciprocity, R_AB = R_BA with R_AB = 2 Re[X_cross,AB] +
    Pi_bare,AB, is violated by 1.1e-3 on a synthetic second source (a separate
    diagnostic, not part of this code); that proposition is not derived from
    the action.
    derivations/dk_pairing.py [9] derives the genuine statement.  L_2q is BILINEAR
    (one H-slot and one K-slot per monomial), so Euler's theorem applies to
    each block separately with its own flux, and the difference of the two is
    the Lagrange/Green concomitant W = Theta_K - Theta_H, giving

        R_AB - R_BA = (1/alpha) * ([W]_bnd - [W]_hor) ,

    an identity in ten independent slot functions and hence valid with the
    H-slots from one solution and the K-slots from another.  This anchor
    checks the three things that statement actually asserts:

      (a) the code objects ARE the derived source halves:  R_AB computed as
          2 Re[X] + Pi equals (1/alpha) int src_k to machine precision (and
          likewise R_BA vs src_h) -- so 2X + Pi was the right object all along;
      (b) [W] vanishes at BOTH ends on the physical boundary conditions, so
          the identity predicts EXACT reciprocity for admissible sources;
      (c) reciprocity duly holds to machine precision on admissible sources.

    Admissibility is the point a plain reciprocity test misses.  The algebraic elimination
    that defines H_rr, H_xr differentiates the (rr), (xr) rows, so it carries
    w0'' and was emitted using the zero-mode ODE (U e^W w0')' + B e^{W-2V} w0
    = 0.  A profile violating that ODE is not a source of this system at all:
    its "response" fails elimination consistency (2.4e-2 and 6.5e-1 for the
    synthetic w0*(1 - sigma/2), against 5e-9 for w0), so it is not a solution
    and no reciprocity is expected of it.  The regular solutions of that ODE
    are a single ray (the second solution is logarithmic at the horizon), so
    the only admissible second source is a multiple of w0; here it is 0.7 w0.
    CAVEAT: with wB proportional to wA, (c) follows from bilinearity alone and
    is VACUOUS as a reciprocity test; no genuinely distinct admissible source
    exists at fixed (beta, B_c).  The content of this anchor is (a), the code
    pairing against the derived source halves, and (b), the vanishing of the
    Green flux at both ends.  The kernel only evaluates the diagonal, so it
    does not depend on off-diagonal reciprocity.
    """
    imid = int(np.argmin(np.abs(bca["beta"] - 8.0)))
    br = make_brane(bca, imid)
    tlog(
        f"[3] polarised two-source identity at beta = {br.beta}, alpha = {br.alpha:.5f}"
    )
    worst = 0.0
    for s in s_list:
        G = np.sqrt(s)
        wA = br.w0
        wB = 0.7 * br.w0  # admissible: same zero-mode ODE
        sA, sB = br.coupled_solve(s, w0=wA), br.coupled_solve(s, w0=wB)
        # (a) the code pairing vs the derived source halves
        code = {}
        for tag, wsrc, solF in (("AB", wA, sB), ("BA", wB, sA)):
            f = br.flds
            wsp = np.real(-(br.sig**2) * (br.D @ wsrc))
            zer = np.zeros(br.N, dtype=complex)
            dens = pair_density(
                f["U"],
                f["V"],
                f["W"],
                br.Bc,
                G,
                wsrc[:-1],
                wsp[:-1],
                zer,
                zer,
                solF["H"][0][:-1],
                solF["H"][1][:-1],
                solF["Hrr"][:-1],
                solF["Hxr"][:-1],
            )
            full = np.zeros(br.N + 1, dtype=complex)
            full[:-1] = dens
            Xc = br.integ_dr(br.P1 * br.sig * (1j * G * solF["H"][2]) * wsrc**2).real
            code[tag] = 2.0 * Xc + br.integ_dr(full).real
        R_AB, R_BA, W_hor, W_bnd = two_source_objects(br, s, sA, sB, wA, wB)
        d_id = max(abs(R_AB - code["AB"]), abs(R_BA - code["BA"])) / max(
            abs(R_AB), 1e-30
        )
        # (b)+(c) the flux and the reciprocity it predicts
        d_rec = abs(R_AB - R_BA) / max(abs(R_AB), 1e-30)
        d_flux = abs(W_bnd - W_hor) / max(abs(R_AB), 1e-30)
        worst = max(worst, d_id, d_rec, d_flux)
        tlog(
            f"    s = {s:6.1f}: (a) 2X+Pi vs (1/a)int src rel {d_id:.1e}; "
            f"(b) (1/a)[W] = {W_bnd - W_hor:+.2e} (rel {d_flux:.1e}); "
            f"(c) R_AB - R_BA rel {d_rec:.1e}"
        )
        # elimination consistency: what admissibility actually buys
        tlog(
            f"              admissibility (elimination defect): "
            f"A {sA['defect'][0]:.1e},{sA['defect'][1]:.1e}  "
            f"B {sB['defect'][0]:.1e},{sB['defect'][1]:.1e}"
        )
    tlog(f"[3] worst of (a),(b),(c) = {worst:.1e} (tol {tol:.0e})")
    assert worst < tol, "the derived two-source identity failed"
    return worst


def anchor4_spot(bca):
    """Two-method spot check at a mid beta before the scan."""
    imid = int(np.argmin(np.abs(bca["beta"] - 8.0)))
    br = make_brane(bca, imid)
    out = []
    for gam in (0.05, 1.0, 8.0):
        s = gam * br.Bc
        kc = br.kernel(s)
        kf = fd_richardson(br, s)
        d = abs(kc["K"] - kf["K"])
        out.append(d)
        tlog(
            f"[4] beta = {br.beta}, s/Bc = {gam}: K_cheb = {kc['K']:.8f}, "
            f"K_FD-R = {kf['K']:.8f}, |d| = {d:.2e}; Pib = {kc['Pib']:.6f} "
            f"vs {kf['Pib']:.6f}; Xc = {kc['Xc']:.6f} vs {kf['Xc']:.6f}; "
            f"defect = ({kc['defect'][0]:.1e},{kc['defect'][1]:.1e})"
        )
    # N-refinement spot (Chebyshev 64 vs 96)
    br96 = Brane(br.beta, br.Bc, np.zeros(97), N=96)
    br96.w0 = np.real(br.bg._bary(br.w0, br96.sig))
    br96.w0p = -(br96.sig**2) * (br96.D @ br96.w0)
    s = 1.0 * br.Bc
    k64, k96 = br.kernel(s), br96.kernel(s)
    dN = abs(k64["K"] - k96["K"])
    tlog(
        f"[4] N-refinement at s = Bc: K(64) = {k64['K']:.8f}, "
        f"K(96) = {k96['K']:.8f}, d = {dN:.2e}"
    )
    # the scan-wide two-method tolerance (7e-6) and the N = 64 vs 96 spot
    # (measured 6e-13)
    assert max(out) < 7e-6, "anchor 4: collocation vs FD-Richardson"
    assert dN < 1e-11, "anchor 4: N-refinement"
    return max(out)


def anchor5_flux(bca, gammas=(0.02, 1.0, 8.0, 45.0), tol_hor=1e-9):
    """The by-parts flux of the pairing identity vanishes on the solutions.

    The production kernel drops NORM*(1/2)*[Theta]_hor^bnd (paper app. C.3); this
    measures it.  Theta is evaluated on the diagonal configuration (H-slots =
    the coupled response, K-slots = its conjugate) at every non-probe brane of
    the cache and at s/Bc in `gammas`, in kernel units relative to K.  At the
    horizon it is at roundoff; towards the boundary it falls off as sigma^2
    (local exponent from the last two nodes), so [Theta]_bnd = 0 in the limit.
    Theta carries no second derivatives, so the pp slots are passed as zero.
    """
    worst_hor, p_range = 0.0, [np.inf, -np.inf]
    for i in range(len(bca["beta"])):
        if float(bca["beta"][i]) == 0.0:
            continue
        br = make_brane(bca, i)
        n, f = br.N, br.flds
        fac = dps.NORM_TIMES_ALPHA / br.alpha * 0.5
        for gam in gammas:
            s = gam * br.Bc
            sol = br.coupled_solve(s)
            z = np.zeros(n, dtype=complex)
            (Hyy, Hzz, ay), (dyy, dzz, day) = sol["H"], sol["dHr"]
            slots = [
                np.asarray(x)[:n]
                for x in (
                    Hyy,
                    dyy,
                    z,
                    Hzz,
                    dzz,
                    z,
                    sol["Hrr"],
                    z,
                    z,
                    sol["Hxr"],
                    z,
                    z,
                    ay,
                    day,
                    z,
                )
            ]
            th = fac * np.array(
                [
                    dps.theta(
                        1.0 / br.sig[k],
                        *(f[q][k] for q in _BGQ),
                        br.Bc,
                        np.sqrt(s),
                        br.alpha,
                        0.0,
                        0.0,
                        *(x[k] for x in slots),
                        *(np.conj(x[k]) for x in slots),
                    )
                    for k in range(n)
                ]
            )
            Kv = br.kernel(s)["K"]
            p = np.log(abs(th[-2] / th[-1])) / np.log(br.sig[n - 2] / br.sig[n - 1])
            worst_hor = max(worst_hor, abs(th[0]) / Kv)
            p_range = [min(p_range[0], p), max(p_range[1], p)]
    tlog(
        f"[5] flux of the pairing identity on the solutions: horizon "
        f"|NORM/2 Theta|/K <= {worst_hor:.1e}; boundary fall-off sigma^p with "
        f"p in [{p_range[0]:.3f}, {p_range[1]:.3f}] ({len(bca['beta']) - 1} "
        f"branes x {len(gammas)} harmonics)"
    )
    assert worst_hor < tol_hor, "pairing flux does not vanish at the horizon"
    assert p_range[0] > 1.9, "pairing flux does not vanish at the boundary"
    return worst_hor, p_range


# ---------------------------------------------------------------------------
# production scan
# ---------------------------------------------------------------------------
def scan(bca, fd_M=200):
    nb, ns = len(bca["beta"]), len(GAMMA_GRID)
    shape = (nb, ns)
    out = {
        q: np.zeros(shape)
        for q in ("K", "C4", "X", "P", "Pib", "Xc", "K_fd", "def_rr", "def_xr", "imag")
    }
    sgrid = np.zeros(shape)
    t0 = time.time()
    for ib in range(nb):
        tb = time.time()
        br = make_brane(bca, ib)
        for js, gam in enumerate(GAMMA_GRID):
            s = float(gam * br.Bc)
            sgrid[ib, js] = s
            k = br.kernel(s)
            kf = fd_richardson(br, s, M=fd_M)
            out["K"][ib, js] = k["K"]
            out["C4"][ib, js] = k["C4"]
            out["X"][ib, js] = k["X"]
            out["P"][ib, js] = k["P"]
            out["Pib"][ib, js] = k["Pib"]
            out["Xc"][ib, js] = k["Xc"]
            out["K_fd"][ib, js] = kf["K"]
            out["def_rr"][ib, js] = k["defect"][0]
            out["def_xr"][ib, js] = k["defect"][1]
            out["imag"][ib, js] = k["imag"]
        dmax = np.max(np.abs(out["K"][ib] - out["K_fd"][ib]))
        tlog(
            f"scan beta = {br.beta:6.2f} (alpha = {br.alpha:.5f}, "
            f"Bc = {br.Bc:.5f}): K(s->0) = {out['K'][ib, 0]:+.6f}, "
            f"min K = {np.min(out['K'][ib]):+.6f}, two-method max|d| = "
            f"{dmax:.1e}  [{time.time() - tb:.1f}s]"
        )
    tlog(f"scan total {time.time() - t0:.1f}s")
    # Two methods (collocation vs finite differences with Richardson), quoted
    # in paper app. D.4 as agreeing to 6e-6 over the whole scan.
    dall = float(np.max(np.abs(out["K"] - out["K_fd"])))
    tlog(f"scan two-method max|K - K_fd| = {dall:.2e} (quoted 6e-6)")
    if not dall < 7e-6:
        tlog("FAILED: two-method agreement over the scan")
        raise SystemExit(1)
    np.savez(
        OUT_PATH,
        beta=bca["beta"],
        alpha=bca["alpha"],
        B_c=bca["B_c"],
        gamma=GAMMA_GRID,
        sgrid=sgrid,
        K=out["K"],
        C4=out["C4"],
        X=out["X"],
        P=out["P"],
        Pi_bare=out["Pib"],
        X_coupled=out["Xc"],
        K_fd=out["K_fd"],
        defect_rr=out["def_rr"],
        defect_xr=out["def_xr"],
        imag=out["imag"],
        readme=(
            "Exact kernel (paper sec. 5.1, app. C) K(s;beta) = C4 - Re[X_coupled] - "
            "Pi_bare/2 on the DK brane at B = B_c(beta), T = 1/pi, brane "
            "norm J = int p1 w0^2 dr = 1; s = gamma*B_c(beta).  Derived "
            "on-shell half-pairing (derivations/dk_pairing.py; flux vanishes "
            "on the solutions); true-Ricci response rows and exchange weight -(alpha/2) "
            "gcal (paper app. C.3).  X = decoupled"
            "-photon X (diagnostic); P = dressed pairing (diagnostic, "
            "slope alpha*gcal).  K_fd = independent uniform-sigma FD + "
            "Richardson of the same formula.  defect_* = elimination-"
            "consistency monitors."
        ),
    )
    tlog(f"saved {OUT_PATH}")
    return sgrid, out


def figure(bca, sgrid, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
    beta = bca["beta"]
    alpha = bca["alpha"]
    sel = [0, 4, 7, 10, 12, 14, 17, 19, 22]
    cmap = plt.cm.viridis
    for ib in sel:
        c = cmap(ib / (len(beta) - 1.0))
        lbl = f"$\\beta$={beta[ib]:g} ($\\alpha$={alpha[ib]:.3f})"
        ax[0].plot(GAMMA_GRID, out["K"][ib] / out["K"][0], color=c, label=lbl)
        if beta[ib] > 0:
            exch = (out["Xc"][ib] - out["X"][ib]) + 0.5 * out["Pib"][ib]
            ax[1].plot(GAMMA_GRID, exch, color=c, label=lbl)
    ax[0].axhline(0.0, color="k", lw=0.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$s/B_c(\beta)$")
    ax[0].set_ylabel(r"$K(s)/K_0(s)$")
    ax[0].set_title("exact kernel evolution (brane norm $J=1$)")
    ax[0].legend(fontsize=6.5, ncol=2)
    ax[1].axhline(0.0, color="k", lw=0.6)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$s/B_c(\beta)$")
    ax[1].set_ylabel(r"$(X_c - X) + \Pi_{\rm bare}/2$")
    ax[1].set_title("coupled exchange piece of the exact kernel")
    ax[1].legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=160)
    tlog(f"saved {FIG_PATH}")


if __name__ == "__main__":
    t0 = time.time()
    anchors_only = "--anchors" in sys.argv
    bca = load_cache()
    i0 = int(np.argmin(np.abs(bca["beta"])))
    assert abs(float(bca["beta"][i0])) < 1e-14
    br0 = make_brane(bca, i0)
    anchor1(br0)
    anchor3(bca)
    anchor4_spot(bca)
    anchor5_flux(bca)
    tlog(f"anchors done [{time.time() - t0:.1f}s]")
    if not anchors_only:
        sgrid, out = scan(bca)
        figure(bca, sgrid, out)
    tlog(f"total {time.time() - t0:.1f}s")
