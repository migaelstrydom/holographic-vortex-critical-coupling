"""The strict uniform quartic coefficient K_{G=0}(beta) on the D'Hoker-Kraus
brane, the order of the transition into the uniform triangular lattice, and
the coupling alpha_L at which it changes.

The quartic Landau coefficient of a uniform one-flux-quantum lattice is

    F_4 = (1/2) K_{G=0} + C(tau; beta),
    K_{G=0} = C4 - (1/2) Pi_0,   Pi_0 = int sqrt(-g) S^{MN}[w0] h_MN dr,

with h the uniform (G = 0) metric response to the lattice-averaged condensate
stress at fixed temperature and fixed applied field (derivations/dk_uniform.py,
systems/dk_uniform.py; gauge H_tt = 0, fixed T is H_rr(r_h) = 0).  The abelian
field has no G = 0 response (Bianchi identity at fixed applied field), so the
photon screening X that enters K(s) at every s > 0 is absent here.

Two methods for K_{G=0}:
  [A] Chebyshev collocation of the closed (H_xx, H_zz) system on the
      background's own sigma grid (sigma = 1/r), w0 the cached spectral zero
      mode; horizon rows = the single regularity relation + the fixed-T row.
  [B] Shooting in x = ln(r - 1) from the horizon (DOP853), carrying the zero
      mode w0 itself in the same integration (source-free eigenfunction at the
      cached B_c, horizon-regular, normalised at the end by J), the
      particular solution plus the two regular homogeneous ones, and the
      quadratures for C4, J and Pi_0; the two boundary constants are then
      cancelled by superposition.
A third, structurally different route to the same number is the s -> 0 limit
of the G != 0 kernel K(s; beta) of numerics/dk_kernel.py (a different system,
gauge and set of rows), extrapolated cubically from gamma = s/B_c in
{1e-3, 2e-3, 4e-3, 8e-3}.

Anchors:
  [1] beta -> 0: Pi_0 vanishes and K_{G=0} -> C4 = 1.58059313 (probe).
  [2] O(alpha), independent machinery: lim -(1/2) Pi_0/alpha must equal the
      shift of the probe onset eigenvalue,
          int_0^1 [dP w0'^2 - dQ w0^2] du ,
      evaluated with the per-rho^2 fixed-T response of AdS5-Schwarzschild
      (numerics/g0_thermo.solve_quad, u chart) and the first variation of the
      onset operator (systems/bc_shift.dP, dQ).  This tests the brane system, its
      source normalisation and the fixed-T convention against code that shares
      none of them; the Hellmann-Feynman form of the pairing (derivation [7])
      fixes the weight 1/2 independently of the on-shell action.
  [3] beta -> 0 horizon data per alpha: I_d(r_h) = (H_xx - H_zz)(r_h) ->
      -3.9119463021 and delta s/s|_T = (2 H_xx + H_zz)(r_h)/2 -> -B_c
      (the closed forms on AdS5-Schwarzschild, numerics/g0_thermo).
  [4] Methods A and B agree (to 5e-10 C4; measured 1.1e-10).
  [5] K_{G=0} = lim_{s->0} K(s) on the grid and [7'] on every ladder rung.
  [8] alpha_L: the zero of K_{G=0}/2 + C_tri by a secant in ln beta with a
      fresh background per step, and its error budget (methods A vs B, the
      G != 0 extrapolation, C_tri by finite differences (table 4), shell truncation,
      radial resolution, secant residual).  The
      comparison with the zero on the dk_long_wavelength ladder is cache
      consistency: both use the same K_{G=0} and C_tri code.

Run:  uv run python -m backreaction.numerics.dk_uniform           # grid + ladder + alpha_L (~5 min)
      uv run python -m backreaction.numerics.dk_uniform --quick   # grid + anchors only; writes nothing
"""

import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from backreaction import paths
from backreaction.numerics import dk_background as DK
from backreaction.numerics import dk_kernel as KER
from backreaction.systems import dk_uniform as SU

T0 = time.time()
CHECKS = {}
OUT = paths.data("dk_uniform.npz")
LONGWAVE = paths.data("dk_long_wavelength.npz")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def cc_weights_01(N):
    return KER.cc_weights_01(N)


def xsolve(*args):
    """X = M^{-1} R of systems/dk_uniform with row and column max-equilibration:
    the raw 4x4 spans many decades near the horizon and the boundary (cond up
    to 1e24 at r = 1e5), the equilibrated one does not."""
    M = np.array(SU._Mmat(*args), dtype=float)
    R = np.array(SU._Rmat(*args), dtype=float)
    rn = np.max(np.abs(M), axis=1)
    M, R = M / rn[:, None], R / rn[:, None]
    cn = np.max(np.abs(M), axis=0)
    return np.linalg.solve(M / cn[None, :], R) / cn[:, None]


class Uniform:
    """G = 0 response on one brane: background bg (dk_background.Background),
    physical field Bc, J-normalised spectral zero mode on bg's sigma grid."""

    def __init__(self, bg, Bc, w0_nodal, eps=1e-6):
        self.bg, self.Bc = bg, float(Bc)
        self.beta = float(bg.beta)
        self.alpha = 0.0 if self.beta == 0.0 else self.beta / self.Bc**2
        self.N = bg.N
        self.sig, self.D = bg.sig, bg.D
        self.D2 = self.D @ self.D
        self.w0 = np.asarray(w0_nodal, float)
        self.w0s = self.D @ self.w0
        self.eps = eps

    # ---- background + matter at a point ----------------------------------
    def point(self, sk, w=None, wp=None):
        f = self.bg.fields_sigma([sk])
        g = {k: float(np.real(f[k][0])) for k in ("U", "Up", "V", "Vp", "W", "Wp")}
        if w is None:
            w = float(self.bg._bary(self.w0, [sk])[0])
            wp = -(sk**2) * float(self.bg._bary(self.w0s, [sk])[0])
        return g, w, wp

    def X(self, sk, w=None, wp=None, src=True):
        """4x5 map onto (H_xx, H_xx', H_zz, H_zz', 1) of (H_rr, H_rr', H_xx'',
        H_zz'') at sigma = sk; src=False drops the condensate column."""
        g, w, wp = self.point(sk, w, wp)
        Xv = xsolve(
            1.0 / sk,
            g["U"],
            g["Up"],
            g["V"],
            g["Vp"],
            g["W"],
            g["Wp"],
            self.Bc,
            self.alpha,
            w,
            wp,
        )
        if not src:
            Xv = Xv.copy()
            Xv[:, 4] = 0.0
        return Xv, g

    def horizon_rows(self, wfun=None, src=True):
        """(regularity row from lim U*H_xx'', fixed-T row from lim H_rr) as
        coefficient vectors on (H_xx, H_xx', H_zz, H_zz', 1) at r_h, by
        Richardson in the offset eps.  Also returns the rank diagnostic: the
        ratio of the two singular values of the 2x4 block lim U*(H_xx'', H_zz'')."""
        wfun = self._w_at if wfun is None else wfun
        e = self.eps
        X1, g1 = self.X(1.0 - e, *wfun(1.0 - e), src=src)
        X2, g2 = self.X(1.0 - 2 * e, *wfun(1.0 - 2 * e), src=src)
        Xu = 2.0 * g1["U"] * X1 - g2["U"] * X2
        X0 = 2.0 * X1 - X2
        sv = np.linalg.svd(Xu[2:4, :4], compute_uv=False)
        return Xu[2], X0[0], sv[1] / sv[0]

    def _w_at(self, sk):
        w = float(self.bg._bary(self.w0, [sk])[0])
        wp = -(sk**2) * float(self.bg._bary(self.w0s, [sk])[0])
        return w, wp

    # ---- [A] Chebyshev collocation -----------------------------------------
    def colloc(self):
        N, sig, D, D2 = self.N, self.sig, self.D, self.D2
        n = N + 1
        A = np.zeros((2 * n, 2 * n))
        rhs = np.zeros(2 * n)
        Xs = [None] * n
        for k in range(n):
            sk = sig[k]
            if sk < 1e-13:  # boundary: Dirichlet (normalisable, no r^0 term)
                A[k, k] = 1.0
                A[n + k, n + k] = 1.0
                continue
            if sk > 1.0 - 1e-13:  # horizon
                reg, fixT, _ = self.horizon_rows()
                for row, vec in ((k, reg), (n + k, fixT)):
                    for j in range(2):
                        A[row, j * n + k] += vec[2 * j]
                        A[row, j * n : (j + 1) * n] += (
                            vec[2 * j + 1] * (-(sk**2)) * D[k]
                        )
                    rhs[row] = -vec[4]
                continue
            Xv, g = self.X(sk, *self._w_at(sk))
            Xs[k] = Xv
            wgt = g["U"]
            for i in range(2):
                row = i * n + k
                A[row, i * n : (i + 1) * n] += wgt * (sk**4 * D2[k] + 2 * sk**3 * D[k])
                for j in range(2):
                    A[row, j * n + k] -= wgt * Xv[2 + i][2 * j]
                    A[row, j * n : (j + 1) * n] -= (
                        wgt * Xv[2 + i][2 * j + 1] * (-(sk**2)) * D[k]
                    )
                rhs[row] = wgt * Xv[2 + i][4]
        rn = np.max(np.abs(A), axis=1)
        rn[rn == 0] = 1.0
        A /= rn[:, None]
        cn = np.max(np.abs(A), axis=0)
        cn[cn == 0] = 1.0
        A /= cn[None, :]
        solv = np.linalg.solve(A, rhs / rn) / cn
        Hx, Hz = solv[:n], solv[n:]
        dHx, dHz = -(sig**2) * (D @ Hx), -(sig**2) * (D @ Hz)
        Hr = np.zeros(n)
        for k in range(n):
            if Xs[k] is not None:
                Hr[k] = Xs[k][0] @ np.array([Hx[k], dHx[k], Hz[k], dHz[k], 1.0])
        _, fixT, _ = self.horizon_rows()
        Hr[0] = fixT @ np.array([Hx[0], dHx[0], Hz[0], dHz[0], 1.0])
        # pairing and contact integrals in r (dr = dsigma/sigma^2)
        ccw = cc_weights_01(N)
        pi_i = np.zeros(n)
        c4_i = np.zeros(n)
        J_i = np.zeros(n)
        for k in range(n):
            sk = sig[k]
            if sk < 1e-13:
                continue
            skk = min(sk, 1.0 - 1e-12)
            g, w, wp = self.point(skk, *self._w_at(skk))
            pi_i[k] = (
                SU.pi_density(
                    1 / skk,
                    g["U"],
                    g["Up"],
                    g["V"],
                    g["Vp"],
                    g["W"],
                    g["Wp"],
                    self.Bc,
                    self.alpha,
                    w,
                    wp,
                    Hx[k],
                    Hz[k],
                    Hr[k],
                )
                / sk**2
            )
            p1 = np.exp(g["W"] - 2 * g["V"])
            c4_i[k] = p1 * w**4 / sk**2
            J_i[k] = p1 * w**2 / sk**2
        Pi0 = float(ccw @ pi_i)
        C4 = float(ccw @ c4_i)
        J = float(ccw @ J_i)
        return dict(
            Hx=Hx,
            Hz=Hz,
            Hr=Hr,
            Pi0=Pi0,
            C4=C4,
            J=J,
            KG0=C4 - 0.5 * Pi0,
            Id_h=Hx[0] - Hz[0],
            ds_h=Hx[0] + 0.5 * Hz[0],
            Hrr_h=Hr[0],
        )

    # ---- [B] shooting from the horizon ----------------------------------------
    def shoot(self, r_max=1e5, delta=1e-10, rtol=1e-12, atol=1e-14):
        """Integrate w0 (source-free eigenfunction at B_c) and the three H
        solutions outward in x = ln(r - 1).  Returns the same observables as
        colloc(), with w0 normalised by its own J.

        The start at r = 1 + delta is first order in delta (Taylor data, and
        the quadratures begin there), so K_G0 carries an O(delta) error:
        (shoot - colloc)/C4 = -0.774 delta at beta_L, exactly linear from
        delta = 1e-6 to 1e-10.  delta = 1e-10 puts it at 7e-11 C4, near the
        5e-12 floor reached at delta = 1e-11."""
        Bc = self.Bc
        # horizon data for w0: w(1) = 1, w'(1) = -(B e^{-2V}/U') w (regularity)
        g1, _, _ = self.point(1.0 - 1e-14, 0.0, 0.0)
        w_h = 1.0
        wp_h = -(Bc * np.exp(-2 * g1["V"]) / g1["Up"]) * w_h

        def hor_rows(src):
            # rows evaluated with the SHOOTING zero mode (Taylor near r_h)
            reg, fixT, _ = self.horizon_rows(
                wfun=lambda sk: (w_h + (1.0 / sk - 1.0) * wp_h, wp_h), src=src
            )
            return reg, fixT

        def derivs_at_h(hx, hz, src):
            reg, fixT = hor_rows(src)
            Mh = np.array([[reg[1], reg[3]], [fixT[1], fixT[3]]])
            bh = -np.array(
                [
                    reg[0] * hx + reg[2] * hz + reg[4],
                    fixT[0] * hx + fixT[2] * hz + fixT[4],
                ]
            )
            return np.linalg.solve(Mh, bh)

        starts = []
        for hx, hz, src in ((0.0, 0.0, True), (1.0, 0.0, False), (0.0, 1.0, False)):
            d1 = derivs_at_h(hx, hz, src)
            starts.append((hx, d1[0], hz, d1[1]))

        def rhs(xv, y):
            rr = 1.0 + np.exp(xv)
            dr = np.exp(xv)  # dr/dx
            sk = 1.0 / rr
            w, Pw = y[0], y[1]
            g, _, _ = self.point(sk, 0.0, 0.0)
            P = g["U"] * np.exp(g["W"])
            Q = Bc * np.exp(g["W"] - 2 * g["V"])
            wp = Pw / P
            out = np.zeros_like(y)
            out[0] = wp * dr
            out[1] = -Q * w * dr
            Xs = xsolve(
                rr,
                g["U"],
                g["Up"],
                g["V"],
                g["Vp"],
                g["W"],
                g["Wp"],
                Bc,
                self.alpha,
                w,
                wp,
            )
            Hr_vals = []
            for m in range(3):
                b = 2 + 4 * m
                hx, hx1, hz, hz1 = y[b : b + 4]
                vec = np.array([hx, hx1, hz, hz1, 1.0 if m == 0 else 0.0])
                out[b] = hx1 * dr
                out[b + 1] = (Xs[2] @ vec) * dr
                out[b + 2] = hz1 * dr
                out[b + 3] = (Xs[3] @ vec) * dr
                Hr_vals.append(Xs[0] @ vec)
            # quadratures: C4, J, and the pairing of the particular solution plus
            # the two homogeneous ones (superposed at the end)
            p1 = np.exp(g["W"] - 2 * g["V"])
            out[14] = p1 * w**4 * dr
            out[15] = p1 * w**2 * dr
            for m in range(3):
                b = 2 + 4 * m
                out[16 + m] = (
                    SU.pi_density(
                        rr,
                        g["U"],
                        g["Up"],
                        g["V"],
                        g["Vp"],
                        g["W"],
                        g["Wp"],
                        Bc,
                        self.alpha,
                        w,
                        wp,
                        y[b],
                        y[b + 2],
                        Hr_vals[m],
                    )
                    * dr
                )
            return out

        # initial state at r = 1 + delta (first-order Taylor; w carried with P w')
        rr0 = 1.0 + delta
        g0, _, _ = self.point(1.0 / rr0, 0.0, 0.0)
        P0 = g0["U"] * np.exp(g0["W"])
        w0v = w_h + delta * wp_h
        y0 = np.zeros(19)
        y0[0], y0[1] = w0v, P0 * wp_h
        for m, (hx, hx1, hz, hz1) in enumerate(starts):
            b = 2 + 4 * m
            y0[b : b + 4] = (hx + delta * hx1, hx1, hz + delta * hz1, hz1)
        sol = solve_ivp(
            rhs,
            (np.log(delta), np.log(r_max - 1.0)),
            y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        assert sol.success, sol.message
        yf = sol.y[:, -1]
        # superpose: H = Hp + a HA + b HB with H_xx(inf) = H_zz(inf) = 0
        Hp_x, HA_x, HB_x = yf[2], yf[6], yf[10]
        Hp_z, HA_z, HB_z = yf[4], yf[8], yf[12]
        Mb = np.array([[HA_x, HB_x], [HA_z, HB_z]])
        a, b = np.linalg.solve(Mb, -np.array([Hp_x, Hp_z]))
        J = yf[15]
        Pi0 = (yf[16] + a * yf[17] + b * yf[18]) / J**2
        C4 = yf[14] / J**2
        hx_h = (starts[0][0] + a * starts[1][0] + b * starts[2][0]) / J
        hz_h = (starts[0][2] + a * starts[1][2] + b * starts[2][2]) / J
        # (H is linear in the source ~ w^2; dividing by J restores J = 1 units)
        return dict(
            Pi0=Pi0,
            C4=C4,
            J=1.0,
            KG0=C4 - 0.5 * Pi0,
            Id_h=hx_h - hz_h,
            ds_h=hx_h + 0.5 * hz_h,
            w_bnd=yf[0] / np.sqrt(J),
            nfev=sol.nfev,
        )


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def grid_brane(i, bca=None):
    bca = KER.load_cache() if bca is None else bca
    beta = float(bca["beta"][i])
    bg = DK.solve_colloc(beta, N=64, tol=1e-12, maxit=80, require=True)
    return Uniform(bg, float(bca["B_c"][i]), np.asarray(bca["w0"][i], float))


def flat_anchor():
    """-(1/2) Pi_0 / alpha at alpha -> 0 from the AdS5-Schwarzschild machinery (g0_thermo, bc_shift):
    int [dP w0'^2 - dQ w0^2] du with the per-rho^2 fixed-T response."""
    from exchange_kernel import clencurt_weights
    from backreaction.numerics import g0_thermo as g0
    from backreaction.systems import bc_shift as pert

    Qd = g0.solve_quad("R2", N=400)
    u, D = Qd["u"], Qd["D"]
    ht = np.zeros_like(u)
    hx, hz, hu = Qd["Hxx"], Qd["Hzz"], Qd["Huu"]
    dH = (np.zeros_like(u), Qd["V1"], Qd["V2"], D @ hu)
    wcc = clencurt_weights(len(u) - 1) / 2.0
    w0 = g0.MAT.interp(g0.MAT.w0, u)
    w0p = g0.MAT.interp(g0.MAT.w0p, u)
    vals = np.zeros_like(u)
    for k, uk in enumerate(u):
        uk2 = min(max(uk, 1e-5), 1 - 1e-5)
        args = [v[k] for v in (ht, hx, hz, hu)] + [v[k] for v in dH]
        vals[k] = (
            pert.dP(uk2, g0.BC, 1.0, *args) * w0p[k] ** 2
            - pert.dQ(uk2, g0.BC, 1.0, *args) * w0[k] ** 2
        )
    return float(wcc @ vals), float(Qd["Id"]), float(Qd["ds"])


LADDER = [
    (1e2, 112),
    (3e2, 128),
    (1e3, 144),
    (3e3, 160),
    (1e4, 176),
    (3e4, 192),
    (1e5, 208),
    (3e5, 224),
    (1e6, 240),
    (3e6, 256),
    (1e7, 288),
    (3e7, 304),
    (1e8, 320),
]


def continuity(i, bca, gammas=(1e-3, 2e-3, 4e-3, 8e-3), probe=(1e-3, 1e-4)):
    """K(s; beta) of the production kernel (numerics/dk_kernel) at small s,
    extrapolated to s = 0 (cubic in gamma), against K_{G=0}; and the field-by-
    field comparison of the coupled G != 0 response (gauge H_tt = H_xx = 0) with
    the image of the G = 0 response under xi^x = -H_xx^(0)/(2iG) e^{iGx}:
        H_yy -> H_xx^(0),  H_zz -> H_zz^(0),  H_rr -> H_rr^(0),
        bhat = iG a_y -> -(B/2) H_xx^(0)   (delta a_y = B xi^x),
    with the differences O(gamma)."""
    un = grid_brane(i, bca)
    A = un.colloc()
    br = KER.make_brane(bca, i)
    K = np.array([br.kernel(g * br.Bc)["K"] for g in gammas])
    K0 = float(np.polyfit(gammas, K, 3)[-1])
    devs = []
    for g in probe:
        sres = g * br.Bc
        sol = br.coupled_solve(sres)
        Hyy, Hzz, ay = sol["H"]
        bh = 1j * np.sqrt(sres) * ay
        scale = np.max(np.abs(A["Hx"]))
        devs.append(
            max(
                np.max(np.abs(Hyy - A["Hx"])),
                np.max(np.abs(Hzz - A["Hz"])),
                np.max(np.abs(sol["Hrr"][1:-1] - A["Hr"][1:-1])),
                np.max(np.abs(bh + br.Bc * A["Hx"] / 2)),
            )
            / scale
        )
    Xc_pred = -(br.Bc / 2) * br.integ_dr(br.P1 * br.sig * A["Hx"] * br.w0**2).real
    return dict(
        beta=un.beta, KG0=A["KG0"], K0=K0, C4=A["C4"], devs=devs, Xc_pred=Xc_pred
    )


def main():
    quick = "--quick" in sys.argv
    bca = KER.load_cache()
    betas = np.asarray(bca["beta"], float)

    log("[2] O(alpha) reference: g0_thermo response + bc_shift operator variation")
    ref_slope, Id_ref, ds_ref = flat_anchor()
    log(
        f"    int[dP w0'^2 - dQ w0^2] = {ref_slope:+.10f} per alpha rho^2;"
        f" I_d(1) = {Id_ref:.10f}, ds|_T = {ds_ref:.10f}"
    )

    log("grid beta <= 45: methods A (collocation) and B (shooting, own w0)")
    rows = []
    for i, b in enumerate(betas):
        un = grid_brane(i, bca)
        A = un.colloc()
        Bm = un.shoot()
        _, _, rank = un.horizon_rows()
        rows.append((b, un.alpha, un.Bc, A, Bm, rank))
        log(
            f"  beta={b:7.3f} alpha={un.alpha:.6f} C4={A['C4']:.8f} "
            f"Pi0={A['Pi0']:+.8e} K_G0/C4={A['KG0'] / A['C4']:+.9f} "
            f"|dK|A-B/C4={abs(A['KG0'] - Bm['KG0']) / A['C4']:.1e} sv={rank:.0e}"
        )

    A0 = rows[0][3]
    check(
        "[1] beta = 0: Pi_0 = 0 and K_{G=0} = C4 = 1.58059313",
        abs(A0["Pi0"]) < 1e-12 and abs(A0["KG0"] - 1.58059313) < 1e-7,
        f"Pi0 = {A0['Pi0']:.1e}, K_G0 = {A0['KG0']:.9f}",
    )
    # alpha -> 0 limits: quartic fit in alpha over the five smallest nonzero betas
    # (0.02 ... 0.4); the linear two-point Richardson is only good to 1e-4 here
    a5 = np.array([r[1] for r in rows[1:6]])

    def lim0(vals):
        return float(np.polyfit(a5, np.asarray(vals) / a5, 4)[-1])

    slope = lim0([-0.5 * r[3]["Pi0"] for r in rows[1:6]])
    check(
        "[2] lim -(1/2) Pi_0/alpha = probe onset-eigenvalue shift (g0_thermo + bc_shift)",
        abs(slope / ref_slope - 1) < 1e-6,
        f"brane {slope:+.9f} vs flat {ref_slope:+.9f} (rel {abs(slope / ref_slope - 1):.1e})",
    )
    Id0 = lim0([r[3]["Id_h"] for r in rows[1:6]])
    ds0 = lim0([r[3]["ds_h"] for r in rows[1:6]])
    check(
        "[3] beta -> 0 horizon data: I_d/alpha -> -3.9119463021, (ds/s|_T)/alpha -> -B_c",
        abs(Id0 / Id_ref - 1) < 1e-6 and abs(ds0 / ds_ref - 1) < 1e-6,
        f"I_d -> {Id0:.9f}, ds -> {ds0:.9f} (flat machinery: {Id_ref:.9f}, {ds_ref:.9f})",
    )
    worst = max(abs(r[3]["KG0"] - r[4]["KG0"]) / r[3]["C4"] for r in rows)
    check(
        "[4] methods A and B agree on K_{G=0} over the grid",
        worst < 5e-10,
        f"worst |dK_G0|/C4 = {worst:.1e}",
    )
    fixT = max(abs(r[3]["Hrr_h"]) for r in rows)
    rk = max(r[5] for r in rows)
    check(
        "[4'] fixed-T row met (H_rr(r_h) = 0) and rank-1 horizon block",
        fixT < 1e-9 and rk < 1e-6,
        f"max |H_rr(r_h)| = {fixT:.1e}, sv ratio {rk:.0e}",
    )

    log("[5] continuity: K(s -> 0) of the production kernel vs K_{G=0}")
    cont = []
    for i in (3, 10, 22):
        c = continuity(i, bca)
        cont.append(c)
        log(
            f"  beta={c['beta']:5.1f}: K_G0 = {c['KG0']:.10f}, K(s->0) = {c['K0']:.10f}, "
            f"rel diff {abs(c['KG0'] / c['K0'] - 1):.1e}; field deviations at "
            f"gamma = 1e-3, 1e-4: {c['devs'][0]:.2e}, {c['devs'][1]:.2e}; "
            f"X_c(0) predicted -(B/2) int p1 H_xx w0^2 = {c['Xc_pred']:.6f}"
        )
    # in C4 units, as the paper quotes it (6e-9 at beta = 45); 1e-8 C4 is the
    # floor of the s -> 0 extrapolation (two scripts' extrapolations on the same
    # rung differ by up to 1.0e-8 C4)
    dmax = max(abs(c["KG0"] - c["K0"]) / c["C4"] for c in cont)
    check(
        "[5] K_{G=0} = lim_{s->0} K(s) (no relaxed/clamped difference) to 3e-8 C4",
        dmax < 3e-8,
        f"max |K_G0 - K(s->0)|/C4 = {dmax:.1e}",
    )
    check(
        "[6] G -> 0 response = singular-gauge image of the G = 0 response, O(gamma)",
        all(0.08 < c["devs"][1] / c["devs"][0] < 0.12 for c in cont),
        "deviation ratios (gamma 1e-4 vs 1e-3): "
        + ", ".join(f"{c['devs'][1] / c['devs'][0]:.4f}" for c in cont),
    )

    out = dict(
        beta=np.array([r[0] for r in rows]),
        alpha=np.array([r[1] for r in rows]),
        B_c=np.array([r[2] for r in rows]),
        C4=np.array([r[3]["C4"] for r in rows]),
        Pi0=np.array([r[3]["Pi0"] for r in rows]),
        KG0=np.array([r[3]["KG0"] for r in rows]),
        KG0_shoot=np.array([r[4]["KG0"] for r in rows]),
        ds_h=np.array([r[3]["ds_h"] for r in rows]),
        Id_h=np.array([r[3]["Id_h"] for r in rows]),
        slope_brane=slope,
        slope_flat=ref_slope,
    )

    if not quick:
        out.update(ladder_and_alpha_L())

    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        sys.exit(1)
    if quick:
        log("--quick: nothing written")
        return
    out["readme"] = (
        "Strict uniform (G = 0) quartic coefficient K_G0 = C4 - Pi0/2 on the DK "
        "brane at fixed T and B (numerics/dk_uniform.py): KG0 (collocation), "
        "KG0_shoot (shooting with its own w0) on the bc_alpha grid; lad_* on the "
        "dk_long_wavelength ladder: lad_KG0 / lad_KG0_shoot (the two methods), lad_K0_ext "
        "(cubic s -> 0 extrapolation of the G != 0 kernel from gamma = 1e-3..8e-3), "
        "lad_K0_stencil (a quadratic extrapolation from gamma >= 0.02, biased, "
        "reference only), lad_Ctri (exact shell sum on the same rung).  "
        "alpha_fo/beta_fo/Bc_fo = alpha_L, the zero of K_G0/2 + C_tri, with "
        "alpha_fo_unc its error (quadrature sum of budget_vals, named by "
        "budget_keys, each a shift of alpha), beta_fo_unc and Bc_fo_unc the "
        "corresponding errors of beta and B_c u_H^2.  K_G0 equals lim_{s->0} K(s) "
        "(checks [5], [6], [7'])."
    )
    np.savez(OUT, **out)
    log(f"saved {OUT}")
    log("ALL CHECKS PASSED")


def _secant(fun, lo, hi, f0, f1, rungs, tol=1e-10, maxit=16):
    """Regula falsi (Illinois) in ln beta between rungs lo (f > 0) and hi (f < 0)."""
    from backreaction.numerics import dk_long_wavelength as CL

    x0, x1 = np.log(lo.beta), np.log(hi.beta)
    best, fbest, side = None, np.inf, 0
    for _ in range(maxit):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        r = CL.rung_at(float(np.exp(x2)), rungs)
        f2 = fun(r)
        log(
            f"    beta = {r.beta:.10g}  alpha = {r.alpha:.9f}  F4/C4 = {f2:+.3e}  (N = {r.bg.N})"
        )
        if abs(f2) < abs(fbest):
            best, fbest = r, f2
        if abs(f2) < tol:
            break
        if f2 > 0:
            x0, f0 = x2, f2
            if side == +1:
                f1 *= 0.5
            side = +1
        else:
            x1, f1 = x2, f2
            if side == -1:
                f0 *= 0.5
            side = -1
    return best, fbest


def c_tri_all_shells(rung, gmax=130.0):
    """C_tri = (1/2) sum_{G != 0} e^{-gamma_G/2} K(gamma_G B_c) over every
    triangular shell with gamma <= gmax, exact kernel calls (no weight floor)."""
    g1 = 4.0 * np.pi / np.sqrt(3.0)  # gamma of the first shell, cell area 2 pi
    mult = {}
    for m in range(-12, 13):
        for n in range(-12, 13):
            q = m * m + m * n + n * n
            if q and g1 * q <= gmax:
                mult[q] = mult.get(q, 0) + 1
    return 0.5 * sum(c * np.exp(-g1 * q / 2) * rung.K(g1 * q) for q, c in mult.items())


def ladder_and_alpha_L():
    from backreaction.numerics import dk_long_wavelength as CL

    log("[7] ladder rungs (continuation, as numerics/dk_long_wavelength)")
    cl = np.load(LONGWAVE)
    rungs, lad = [], []
    bg = CL.solve_bg(45.0, 96, None, tol=1e-12)
    for beta, N in LADDER:
        bg = CL.solve_bg(beta, N, bg)
        rung = CL.Rung(beta, bg)
        rungs.append(rung)
        un = Uniform(bg, rung.Bc, rung.w0)
        A = un.colloc()
        Bm = un.shoot(r_max=1e6)
        ext = rung.K0_ext()
        old = rung.K0(CL.OLD_STENCIL)
        ctri, _ = rung.cells()
        j = int(np.argmin(np.abs(np.log(cl["beta"] / beta))))
        same = abs(cl["beta"][j] / beta - 1) < 1e-12
        lad.append(
            dict(
                beta=beta,
                alpha=rung.alpha,
                A=A,
                B=Bm,
                ext=ext,
                old=old,
                Ctri=ctri,
                C4r=rung.C4,
                cache=(float(cl["K0"][j]), float(cl["C_tri"][j])) if same else None,
            )
        )
        log(
            f"  beta={beta:8.1e} alpha={rung.alpha:.6f} K_G0/C4={A['KG0'] / A['C4']:+.9f} "
            f"|A-B|/C4={abs(A['KG0'] - Bm['KG0']) / A['C4']:.1e} "
            f"|ext-A|/C4={abs(ext - A['KG0']) / A['C4']:.1e} "
            f"(0.02 stencil {abs(old - A['KG0']) / A['C4']:.1e})  "
            f"(K_G0/2 + C_tri)/C4 = {(0.5 * A['KG0'] + ctri) / A['C4']:+.7f}"
        )
    check(
        "[7] every ladder background accepted",
        all(r.bg.accepted for r in rungs),
    )
    worstAB = max(abs(x["A"]["KG0"] - x["B"]["KG0"]) / x["A"]["C4"] for x in lad)
    check(
        "[7] methods A and B agree on the ladder",
        worstAB < 5e-10,
        f"worst {worstAB:.1e} C4",
    )
    worstExt = max(abs(x["A"]["KG0"] - x["ext"]) / x["A"]["C4"] for x in lad)
    worstOld = max(abs(x["A"]["KG0"] - x["old"]) / x["A"]["C4"] for x in lad)
    check(
        "[7'] K_G0 = lim_{s->0} K(s) of the G != 0 kernel on every rung to beta = 1e8",
        worstExt < 1e-8,
        f"worst |K_G0 - K(s->0)|/C4 = {worstExt:.1e} (cubic extrapolation from "
        f"gamma = 1e-3..8e-3); the quadratic stencil from gamma >= 0.02 is off by "
        f"up to {worstOld:.1e}",
    )
    c4dev = max(abs(x["A"]["C4"] / x["C4r"] - 1) for x in lad)
    cdev = [
        max(
            abs(x["A"]["KG0"] - x["cache"][0]) / x["A"]["C4"],
            abs(x["Ctri"] - x["cache"][1]),
        )
        for x in lad
        if x["cache"] is not None
    ]
    check(
        "[7''] same rungs as dk_long_wavelength.npz (K_0 and C_tri) and the same C4",
        len(cdev) == len(lad) and max(cdev) < 1e-10 and c4dev < 1e-12,
        f"max dev {max(cdev):.1e}, C4 {c4dev:.1e}",
    )

    # ---- [8] alpha_L ---------------------------------------------------------
    def F4(r, method="colloc"):
        return (0.5 * r.KG0(method) + r.cells()[0]) / r.C4

    f = [(0.5 * x["A"]["KG0"] + x["Ctri"]) / x["A"]["C4"] for x in lad]
    k = next(i for i in range(len(f) - 1) if f[i] > 0 > f[i + 1])
    log(
        f"[8] zero of (K_G0/2 + C_tri) bracketed in beta [{lad[k]['beta']:.3g}, {lad[k + 1]['beta']:.3g}]"
    )
    zr, fz = _secant(F4, rungs[k], rungs[k + 1], f[k], f[k + 1], rungs)
    check(
        "[8] secant converged",
        abs(fz) < 1e-10 and zr.bg.accepted,
        f"alpha_L = {zr.alpha:.9f}, beta = {zr.beta:.4f}, residual {fz:+.1e}",
    )

    # local slopes from two fresh neighbours at ln beta +- 0.01
    nb = [CL.rung_at(zr.beta * np.exp(e), rungs) for e in (-0.01, 0.01)]
    fn = [F4(r) for r in nb]
    dF_da = (fn[1] - fn[0]) / (nb[1].alpha - nb[0].alpha)
    da_dlnb = (nb[1].alpha - nb[0].alpha) / 0.02
    dlnBc_da = np.log(nb[1].Bc / nb[0].Bc) / (nb[1].alpha - nb[0].alpha)
    log(
        f"    local slopes: dF4/dalpha = {dF_da:+.5f} (per C4), dalpha/dln beta = {da_dlnb:.6f}, "
        f"dln B_c/dalpha = {dlnBc_da:.4f}"
    )

    budget = {}
    budget["K_G0 collocation vs shooting"] = abs(
        0.5 * (zr.KG0("shoot") - zr.KG0("colloc")) / zr.C4
    ) / abs(dF_da)
    budget["K_G0 vs s->0 limit of the G != 0 kernel"] = abs(
        0.5 * (zr.K0_ext() - zr.KG0("colloc")) / zr.C4
    ) / abs(dF_da)
    ct_sel, _ = zr.cells()
    # C_tri by the second discretisation (finite differences, Richardson
    # M = 1600/3200) at the same shells
    ct_fd, _ = zr.cells_fd()
    log(f"    C_tri: collocation {ct_sel:.12f}, finite differences {ct_fd:.12f}")
    budget["C_tri: collocation vs finite differences"] = (
        abs(ct_fd - ct_sel) / zr.C4 / abs(dF_da)
    )
    budget["shell sum: weight floor 1e-9 vs all shells to gamma = 130"] = (
        abs(c_tri_all_shells(zr) - ct_sel) / zr.C4 / abs(dF_da)
    )
    bgN = CL.solve_bg(zr.beta, zr.bg.N + 48, zr.bg)
    rN = CL.Rung(zr.beta, bgN)
    budget["radial resolution N -> N + 48"] = abs(F4(rN)) / abs(dF_da) + abs(
        rN.alpha - zr.alpha
    )
    log(f"    N = {bgN.N}: F4/C4 = {F4(rN):+.2e}, alpha = {rN.alpha:.10f}")
    # No "background tolerance" entry: the Newton solve stops at its round-off
    # floor (dk_background.residual_floor), far above either tolerance, so a
    # re-solve at tol = 1e-12 returns the same background (moved by 7e-12)
    # and only repeated the secant residual.
    budget["secant residual"] = abs(fz) / abs(dF_da)
    unc = float(np.sqrt(sum(v * v for v in budget.values())))
    for kk, v in budget.items():
        log(f"    delta alpha_L from {kk:<55s} {v:.1e}")
    log(f"    quadrature sum {unc:.1e}; largest term {max(budget.values()):.1e}")
    beta_unc = unc / da_dlnb * zr.beta
    Bc_unc = unc * dlnBc_da * zr.Bc
    check(
        "[8] alpha_L error budget below 3e-7 and every resolution check accepted",
        unc < 3e-7 and bgN.accepted,
        f"alpha_L = {zr.alpha:.7f} +/- {unc:.0e}, beta_L = {zr.beta:.3f} +/- {beta_unc:.1e}, "
        f"B_c u_H^2 = {zr.Bc:.4f} +/- {Bc_unc:.0e}",
    )
    a_cl = float(cl["alpha_first_order"])
    check(
        "[8] reproduces the zero located on the dk_long_wavelength ladder (dk_long_wavelength.npz)",
        abs(a_cl - zr.alpha) < max(2 * unc, 1e-8),
        f"dk_long_wavelength alpha_L = {a_cl:.9f}, here {zr.alpha:.9f}, difference {abs(a_cl - zr.alpha):.1e}",
    )
    return dict(
        lad_beta=np.array([x["beta"] for x in lad]),
        lad_alpha=np.array([x["alpha"] for x in lad]),
        lad_C4=np.array([x["A"]["C4"] for x in lad]),
        lad_KG0=np.array([x["A"]["KG0"] for x in lad]),
        lad_KG0_shoot=np.array([x["B"]["KG0"] for x in lad]),
        lad_K0_ext=np.array([x["ext"] for x in lad]),
        lad_K0_stencil=np.array([x["old"] for x in lad]),
        lad_Ctri=np.array([x["Ctri"] for x in lad]),
        beta_fo=zr.beta,
        alpha_fo=zr.alpha,
        Bc_fo=zr.Bc,
        F4_fo=fz,
        alpha_fo_unc=unc,
        beta_fo_unc=beta_unc,
        Bc_fo_unc=Bc_unc,
        budget_keys=np.array(list(budget.keys())),
        budget_vals=np.array(list(budget.values())),
        dF4_dalpha=dF_da,
    )


if __name__ == "__main__":
    main()
