"""Why d = 3 binds a crossover state and d = 4, 5, 6 do not.

An extra, not in the paper: the paper reports the integer-d thresholds (table
2, sec. 4.3, app. D.2) but not the continuation in d that explains them.

The T = 0 threshold problem of t0_threshold.py -- the polarised lowest-Landau-level mode
at the throat's BF threshold on the flow AdS_{d+1} -> AdS_{d-1} x R^2 --
continued to real d with `derivations/d_continuation.py`.  In the throat the
threshold solution is (r - r0)^{n/2} w -> a + b ln((r - r0)/(r_* - r0))
(n = d - 2, r_* where e^{2C} = 2v); b/a > 0 is one zero deep in the throat,
i.e. one bound state below the BF threshold, and b/a -> 0 is where it leaves
through the continuum edge (a zero-energy resonance: in the Liouville normal
form below, psi -> a + b y in the throat, so a/b is the scattering length at
the edge in units of the crossover scale).

[1] b/a(d) on a real-d grid, two charts.  In each, the background and the two
    throat solutions at threshold, w1 = x^{-n/2} and w2 = x^{-n/2} ln x
    (x = r - r0), are integrated OUTWARD in one pass; with c0_i their boundary
    sources the source-free solution is c0_2 w1 - c0_1 w2, so b = -c0_1 (the
    source of the IR pure-power solution) and a = c0_2 - c0_1 ln x_*:
      M1  domain-wall gauge, variable rho, flow from the linearised
          deformation of d_continuation [4] (DOP853);
      M2  r chart (g_tt g_rr = -1), variables (ln U, V) against ln(r - r0),
          flow from the independently solved r-chart deformation (LSODA).
    d_crit = the zero of b/a, by brentq in each chart.  Anchors: d = 3, 4, 5, 6
    against t0_threshold.py (inward shooting + throat fit, a third method).
[2] Which ingredient sets the sign?
    (a) Mode weight e^{nA} with n free (n = d - 2 physical; n - 1 = eps is
        the exponent of e^{eps Z}), each operator at its own throat threshold
        lam_th = n^2 v/(4 l^2): b/a(eps) on the d = 4 and d = 3 backgrounds,
        the (d_bg, n) map on [3, 4] x [1, 2] and its zero line.
    (b) Throat units t = rho/l: w_tt + n h w_t + (n^2/4) g w = 0 with
        h = l A' (1 -> l) and g = v e^{-2C} (1 -> 0).  A cross-background
        hybrid (h from one d, g from another) is reported and rejected as
        alignment-dependent; the endpoint test keeps the crossover shape of
        one background and moves only the boundary value l of h.
[3] Liouville normal form (d_continuation [5]): V_L(y) - lam_th for d = 3..6
    and for the swapped weights, figure
    `backreaction/figures/d3_crossover_potential.png`.

Run:  uv run python -m backreaction.numerics.d3_crossover_why   (~15 s)
"""

import math
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from backreaction import paths
from backreaction.derivations import d_continuation as dc

T0 = time.time()
CHECKS = {}
CACHE = paths.data("d3_crossover_why.npz")
FIG = paths.figure("d3_crossover_potential.png")
EPS0 = 1e-10
T0_THRESHOLD_BA = {
    3: 0.122641,
    4: -0.035000,
    5: -0.350252,
    6: -0.996391,
}  # b/a, t0_threshold.py


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


_d = dc.d
_ell = sp.lambdify(_d, dc.ELL)
_v = sp.lambdify(_d, dc.VFIX.subs({dc.alpha: dc.ALPHA_STAR, dc.B: 1}))
_p = sp.lambdify(_d, dc.P_EXP)
_ak = sp.lambdify(_d, dc.a_k_closed()[0])


class Background:
    """Fixed point data for real d at B = 1, alpha = alpha_*(d)."""

    def __init__(self, d):
        self.d = d
        self.l, self.v, self.p = float(_ell(d)), float(_v(d)), float(_p(d))
        self.k = self.p / self.l
        self.ak = float(_ak(d))
        self.ast = 16.0 / (d * (d - 1) * (d - 2))

    def lam_th(self, n):
        return n * n * self.v / (4 * self.l**2)


# ---------------------------------------------------------------------------
# Both charts: background + the two throat solutions at threshold,
#     w1 = x^{-n/2},  w2 = x^{-n/2} ln x       (x = r - r0),
# integrated OUTWARD in one pass.  With c0_i their boundary sources, the
# source-free threshold solution is c0_2 w1 - c0_1 w2 = x^{-n/2}(a + b ln(x/x_*)),
#     b = -c0_1,   a = c0_2 - c0_1 ln x_*.
# ---------------------------------------------------------------------------
def _ratio(c1, c2, lnxs):
    return -c1 / (c2 - c1 * lnxs)


class FlowDW(Background):
    """M1: domain-wall gauge, independent variable rho."""

    RHO_END = 60.0

    def __init__(self, d):
        super().__init__(d)
        dd, ast = d, self.ast
        self.C0 = 0.5 * math.log(self.v)
        e = EPS0
        kl1 = self.k + 1 / self.l
        self.y0 = [
            self.ak * e,
            self.C0 + e,
            1 / self.l + self.ak * e * self.k,
            e * self.k,
            self.l + self.ak * e / kl1,
            0.0,
        ]

        def rhs(r, y):
            A, C, Ap, Cp = y[0], y[1], y[2], y[3]
            Phi = ast * math.exp(-4 * C)
            S = (dd - 2) * Ap + 2 * Cp
            return [
                Ap,
                Cp,
                dd + Phi / (dd - 1) - Ap * S,
                dd - Phi * (dd - 2) / (dd - 1) - Cp * S,
                math.exp(A),
                math.exp(-C),
            ]

        self.rhs = rhs
        self._sol = None

    @property
    def sol(self):
        if self._sol is None:
            self._sol = solve_ivp(
                self.rhs,
                [0.0, self.RHO_END],
                self.y0,
                method="DOP853",
                rtol=1e-13,
                atol=1e-15,
                dense_output=True,
            )
            self.rho_end = self._sol.t[-1]
            self.rho_s = brentq(
                lambda r: math.exp(2 * self.fields(r)[1]) - 2 * self.v,
                0.0,
                self.rho_end,
            )
        return self._sol

    def fields(self, rho):
        if rho >= 0:
            return self.sol.sol(rho)
        e = EPS0 * math.exp(self.k * rho)
        kl1 = self.k + 1 / self.l
        return np.array(
            [
                rho / self.l + self.ak * e,
                self.C0 + e,
                1 / self.l + self.ak * self.k * e,
                self.k * e,
                self.l * math.exp(rho / self.l)
                + self.ak * EPS0 * math.exp(kl1 * rho) / kl1,
                rho * math.exp(-self.C0),
            ]
        )

    def rho_deep(self, tol=1e-14):
        return math.log(tol / EPS0) / self.k

    def threshold_ba(self, n=None, lam=None, full=False):
        n = self.d - 2 if n is None else n
        lam = self.lam_th(n) if lam is None else lam
        R0 = self.y0[4]
        A0 = self.y0[0]
        lR = math.log(R0)
        q = R0 ** (-n / 2)
        z0 = self.y0 + [
            q,
            -(n / 2) * q * math.exp(A0) / R0,
            q * lR,
            q * math.exp(A0) / R0 * (1 - (n / 2) * lR),
        ]
        bg = self.rhs
        twov = math.log(2 * self.v) / 2

        def rhs(r, y):
            out = bg(r, y)
            Ap, C = y[2], y[1]
            em = lam * math.exp(-2 * C)
            out += [y[7], -n * Ap * y[7] - em * y[6], y[9], -n * Ap * y[9] - em * y[8]]
            return out

        def cross(r, y):
            return y[1] - twov

        so = solve_ivp(
            rhs,
            [0.0, self.RHO_END],
            z0,
            method="DOP853",
            rtol=1e-13,
            atol=1e-15,
            events=cross,
            dense_output=True,
        )
        lnRs = math.log(so.y_events[0][0][4])
        res = []
        for rr in (self.RHO_END - 12.0, self.RHO_END):
            y = so.sol(rr)
            c1 = y[6] + y[7] / n
            c2 = y[8] + y[9] / n
            res.append(_ratio(c1, c2, lnRs))
        uv = max(abs(so.y[2, -1] - 1), abs(so.y[3, -1] - 1))
        if full:
            return res[1], abs(res[1] - res[0]), uv
        return res[1]


_RC = dc.r_chart()
_Uv, _Vv, _Up, _Vp = _RC["syms"]
_sub = {dc.alpha: dc.ALPHA_STAR, dc.B: 1}
_Upp = sp.lambdify((_d, _Uv, _Vv, _Up, _Vp), _RC["Upp"].subs(_sub), "math")
_Vpp = sp.lambdify((_d, _Uv, _Vv, _Up, _Vp), _RC["Vpp"].subs(_sub), "math")
_rows_r, (_u, _pp) = dc.deformation_r(_RC)
_usol = sp.solve(_rows_r[0].subs(_pp, dc.P_EXP), _u)[0]
_up = sp.lambdify(_d, _usol.subs(_sub))


class FlowR(Background):
    """M2: r chart (g_tt g_rr = -1), variables (ln U, V) against s = ln(r - r0),
    starting data from the r-chart deformation of d_continuation [4]."""

    S_SPAN = 30.0  # ln-range beyond the crossover

    def __init__(self, d):
        super().__init__(d)
        self.up = float(_up(d))
        self.s0 = math.log(1e-12) / self.p  # eps x0^p = 1e-12 with eps = 1
        e = math.exp(self.p * self.s0)
        self.y0 = [
            2 * (self.s0 - math.log(self.l)) + self.up * e,
            0.5 * math.log(self.v) + e,
            2 + self.up * self.p * e,
            self.p * e,
        ]

    def bg(self, s, L, V, Ls, Vs):
        x = math.exp(s)
        U = math.exp(L)
        Up = U * Ls / x
        Vp = Vs / x
        dd = self.d
        return x * x * _Upp(dd, U, V, Up, Vp) / U + Ls - Ls * Ls, x * x * _Vpp(
            dd, U, V, Up, Vp
        ) + Vs

    def threshold_ba(self, n=None, full=False):
        n = self.d - 2 if n is None else n
        lam = self.lam_th(n)
        s0 = self.s0
        x0 = math.exp(s0)
        L0 = self.y0[0]
        P0 = math.exp((n + 1) * L0 / 2)
        q = x0 ** (-n / 2)
        z0 = self.y0 + [
            q,
            -P0 * (n / 2) * q / x0,
            q * s0,
            P0 * q / x0 * (1 - (n / 2) * s0),
        ]
        twov = math.log(2 * self.v) / 2

        def rhs(s, y):
            L, V, Ls, Vs = y[0], y[1], y[2], y[3]
            Lss, Vss = self.bg(s, L, V, Ls, Vs)
            x = math.exp(s)
            P = math.exp((n + 1) * L / 2)
            Q = math.exp((n - 1) * L / 2 - 2 * V)
            return [
                Ls,
                Vs,
                Lss,
                Vss,
                x * y[5] / P,
                -x * lam * Q * y[4],
                x * y[7] / P,
                -x * lam * Q * y[6],
            ]

        def cross(s, y):
            return y[1] - twov

        cross.terminal = True
        a = solve_ivp(
            rhs, [s0, 80.0], z0, method="LSODA", rtol=1e-12, atol=1e-15, events=cross
        )
        s_star = a.t_events[0][0]
        b = solve_ivp(
            rhs,
            [s_star, s_star + self.S_SPAN],
            a.y_events[0][0],
            method="LSODA",
            rtol=1e-12,
            atol=1e-15,
            dense_output=True,
        )
        res = []
        for sm in (s_star + self.S_SPAN - 8.0, s_star + self.S_SPAN):
            y = b.sol(sm)
            x = math.exp(sm)
            P = math.exp((n + 1) * y[0] / 2)
            c1 = y[4] + x * (y[5] / P) / n
            c2 = y[6] + x * (y[7] / P) / n
            res.append(_ratio(c1, c2, s_star))
        uv = max(abs(b.y[2, -1] - 2), abs(b.y[3, -1] - 1))
        if full:
            return res[1], abs(res[1] - res[0]), uv
        return res[1]


# ---------------------------------------------------------------------------
# [3] Liouville potential
# ---------------------------------------------------------------------------
def liouville_profile(fl, n=None):
    n = fl.d - 2 if n is None else n
    _ = fl.sol  # integrate the background
    rho = np.concatenate(
        [
            np.linspace(fl.rho_deep(1e-6) - 6, 0, 600, endpoint=False),
            np.linspace(0, min(fl.rho_end, fl.rho_s + 14), 1400),
        ]
    )
    F = np.array([fl.fields(r) for r in rho]).T
    A, C, Ap, Cp, R, Y = F
    App = np.array([fl.rhs(r, f)[2] for r, f in zip(rho, F.T, strict=True)])
    Cpp = np.array([fl.rhs(r, f)[3] for r, f in zip(rho, F.T, strict=True)])
    s1, s2 = (n * Ap - Cp) / 2, (n * App - Cpp) / 2
    VL = np.exp(2 * C) * (s2 + s1**2 + Cp * s1)
    Yinf = fl.sol.y[5, -1] + math.exp(-fl.sol.y[1, -1])
    dy = Yinf - Y  # y_b - y
    dy_s = Yinf - fl.fields(fl.rho_s)[5]
    return dy / dy_s, (VL - fl.lam_th(n)) * dy_s**2, dy_s


def throat_units_ba(fA, fC, n=1.0, shift=0.0, l_end=None, span=40.0):
    """The threshold mode in throat units t = rho/l at its own threshold,
        w_tt + n h(t) w_t + (n^2/4) g(t) w = 0,   h = l A',  g = v e^{-2C},
    h -> 1, g -> 1 in the throat; h -> l, g -> 0 at the AdS_{d+1} boundary.
    Diagnostics (not consistent geometries):
      * hybrid: h from flow fA, g from flow fC, aligned at their e^{2C} = 2v
        points (t = 0) up to `shift`;
      * endpoint: h's shape kept, its boundary value l rescaled to l_end,
        h -> 1 - (1 - h)(1 - l_end)/(1 - l)  (fA = fC, no alignment involved).
    Returns b/a with x = t (sign-equivalent to the r-chart b/a)."""
    for fl in (fA, fC):
        _ = fl.sol
    tA, tC = fA.rho_s / fA.l, fC.rho_s / fC.l + shift
    t0 = max(fA.rho_deep() / fA.l - tA, fC.rho_deep() / fC.l - tC) - 1.0
    L = fA.l if l_end is None else l_end
    sc = (1 - L) / (1 - fA.l)

    def rhs(t, y):
        h = 1 - (1 - fA.l * fA.fields(fA.l * (t + tA))[2]) * sc
        g = fC.v * math.exp(-2 * fC.fields(fC.l * (t + tC))[1])
        return [
            y[1],
            -n * h * y[1] - n * n / 4 * g * y[0],
            y[3],
            -n * h * y[3] - n * n / 4 * g * y[2],
        ]

    q = math.exp(-n * t0 / 2)
    so = solve_ivp(
        rhs,
        [t0, span],
        [q, -n / 2 * q, t0 * q, q * (1 - n / 2 * t0)],
        method="DOP853",
        rtol=1e-11,
        atol=1e-14,
    )
    c1 = so.y[0, -1] + so.y[1, -1] / (n * L)
    c2 = so.y[2, -1] + so.y[3, -1] / (n * L)
    return _ratio(c1, c2, 0.0)


def main():
    # ---------------- [1] continuous d ----------------
    log("[1] b/a(d) on the real-d grid, M1 (domain wall) and M2 (r chart)")
    dgrid = np.array(
        [
            2.8,
            2.9,
            3.0,
            3.1,
            3.2,
            3.3,
            3.4,
            3.5,
            3.6,
            3.7,
            3.75,
            3.8,
            3.85,
            3.9,
            3.95,
            4.0,
            4.1,
            4.25,
            4.5,
            5.0,
            6.0,
        ]
    )
    rows = []
    for dd in dgrid:
        ba1, drift1, uv1 = FlowDW(dd).threshold_ba(full=True)
        ba2, drift2, uv2 = FlowR(dd).threshold_ba(full=True)
        p = float(_p(dd))
        rows.append((dd, p, ba1, ba2, max(drift1, drift2), uv1, uv2))
        log(f"   d = {dd:5.2f}  p = {p:.6f}  b/a: M1 {ba1:+.9f}  M2 {ba2:+.9f}")
    rows = np.array(rows)
    ba1, ba2 = rows[:, 2], rows[:, 3]
    check(
        "[1] both flows reach AdS_{d+1} at every d",
        rows[:, 5].max() < 1e-10 and rows[:, 6].max() < 1e-10,
        f"max |A'-1|,|C'-1| = {rows[:, 5].max():.0e}; r chart |dlnU/ds-2|,|dV/ds-1| = "
        f"{rows[:, 6].max():.0e}",
    )
    check(
        "[1] boundary sources converged (b/a at two UV cut-offs)",
        rows[:, 4].max() < 1e-9,
        f"max drift {rows[:, 4].max():.0e}",
    )
    dev12 = np.max(np.abs(ba1 - ba2))
    check(
        "[1] M1 and M2 b/a agree on the whole grid",
        dev12 < 1e-8,
        f"max |diff| = {dev12:.1e}",
    )
    anc = {
        int(dd): b
        for dd, b in zip(dgrid, ba1, strict=True)
        if dd in (3.0, 4.0, 5.0, 6.0)
    }
    check(
        "[1] integer d reproduce t0_threshold.py (inward shooting + throat fit)",
        all(abs(anc[k] - T0_THRESHOLD_BA[k]) < 2e-5 for k in T0_THRESHOLD_BA),
        ", ".join(f"d={k}: {anc[k]:+.6f}" for k in sorted(anc)),
    )
    i = np.where(np.diff(np.sign(ba1)) != 0)[0]
    check(
        "[1] b/a changes sign exactly once on [2.8, 6], decreasing",
        len(i) == 1 and np.all(np.diff(ba1[(dgrid >= 3)]) < 0),
        f"between d = {dgrid[i[0]]} and {dgrid[i[0] + 1]}",
    )
    lo, hi = dgrid[i[0]], dgrid[i[0] + 1]
    dc1 = brentq(lambda x: FlowDW(x).threshold_ba(), lo, hi, xtol=1e-12)
    dc2 = brentq(lambda x: FlowR(x).threshold_ba(), lo, hi, xtol=1e-12)
    log(f"   d_crit: M1 {dc1:.10f},  M2 {dc2:.10f}")
    check(
        "[1] d_crit agrees between the two charts",
        abs(dc1 - dc2) < 1e-7,
        f"|diff| = {abs(dc1 - dc2):.1e}",
    )
    d_crit = dc1
    p_crit = float(_p(d_crit))
    log(f"   d_crit = {d_crit:.7f}, p(d_crit) = {p_crit:.6f}")

    # ---------------- [2] the volume factor ----------------
    log("[2] mode weight e^{nA}, n = 1 + eps, each at its own threshold")
    eps = np.linspace(0, 1, 11)
    f4, f3 = FlowDW(4.0), FlowDW(3.0)
    r4, r3 = FlowR(4.0), FlowR(3.0)
    ba_eps4 = np.array([f4.threshold_ba(n=1 + e) for e in eps])
    ba_eps4r = np.array([r4.threshold_ba(n=1 + e) for e in eps])
    ba_eps3 = np.array([f3.threshold_ba(n=1 + e) for e in eps])
    ba_eps3r = np.array([r3.threshold_ba(n=1 + e) for e in eps])
    log("   eps:                 " + " ".join(f"{e:7.2f}" for e in eps))
    log("   d=4 background, b/a: " + " ".join(f"{x:+7.4f}" for x in ba_eps4))
    log("   d=3 background, b/a: " + " ".join(f"{x:+7.4f}" for x in ba_eps3))
    check(
        "[2] eps scans agree between charts",
        max(np.abs(ba_eps4 - ba_eps4r).max(), np.abs(ba_eps3 - ba_eps3r).max()) < 1e-8,
    )
    check(
        "[2] eps = 1 on d = 4 and eps = 0 on d = 3 reproduce the physical values",
        abs(ba_eps4[-1] - anc[4]) < 1e-12 and abs(ba_eps3[0] - anc[3]) < 1e-12,
    )

    def eps_root(fl):
        g = lambda e: fl.threshold_ba(n=1 + e)
        vals = [g(e) for e in eps]
        j = np.where(np.diff(np.sign(vals)) != 0)[0]
        return [brentq(g, eps[k], eps[k + 1], xtol=1e-10) for k in j]

    e4, e3 = eps_root(f4), eps_root(f3)
    log(f"   sign change in eps: d=4 background {e4}, d=3 background {e3}")
    # the narrow claim: on the two PHYSICAL backgrounds the
    # mode weight never changes the sign (it does change the size: on d = 3
    # b/a doubles, 0.123 -> 0.254; the sign-flipping row of the map is d_bg ~ 3.8)
    check(
        "[2] on the d = 3 and d = 4 backgrounds no eps in [0, 1] changes the sign of b/a",
        not e4 and not e3 and max(ba_eps4) < 0 < min(ba_eps3),
        f"d=4: {min(ba_eps4):+.4f} .. {max(ba_eps4):+.4f}; "
        f"d=3: {ba_eps3[0]:+.4f} (eps=0) .. {ba_eps3[-1]:+.4f} (eps=1)",
    )

    log("[2] the (d_bg, n) map on [3, 4] x [1, 2]")
    dbg = np.linspace(3, 4, 11)
    ngrid = np.linspace(1, 2, 11)
    fmap = {dd: FlowDW(dd) for dd in dbg}
    M = np.array([[fmap[dd].threshold_ba(n=nn) for nn in ngrid] for dd in dbg])
    log("   rows d_bg = 3.0 ... 4.0, columns n = 1.0 ... 2.0")
    for dd, row in zip(dbg, M, strict=True):
        log(f"   d_bg = {dd:4.2f}: " + " ".join(f"{x:+.3f}" for x in row))
    # physical diagonal n = d - 2 is M[i, i]
    check(
        "[2] map diagonal = physical b/a(d)",
        np.allclose(np.diag(M), np.interp(dbg, dgrid, ba1), atol=2e-3),
    )
    # sensitivity: d(b/a)/dn vs d(b/a)/d d_bg at the physical crossing
    h = 0.02
    fcr = FlowDW(d_crit)
    dn = (fcr.threshold_ba(n=d_crit - 2 + h) - fcr.threshold_ba(n=d_crit - 2 - h)) / (
        2 * h
    )
    dbgd = (
        FlowDW(d_crit + h).threshold_ba(n=d_crit - 2)
        - FlowDW(d_crit - h).threshold_ba(n=d_crit - 2)
    ) / (2 * h)
    log(
        f"   at d_crit: d(b/a)/dn = {dn:+.4f} (mode weight), d(b/a)/d(d_bg) = {dbgd:+.4f} (background)"
    )

    zl = []
    for nn in (1.0, 1.25, 1.5, 1.75, 2.0):
        zl.append(
            brentq(lambda x, nn=nn: FlowDW(x).threshold_ba(n=nn), 3.5, 4.0, xtol=1e-9)
        )
    log(
        "   zero line of the map, d_bg at which b/a = 0 for mode weight n = 1, 1.25, "
        "1.5, 1.75, 2: " + ", ".join(f"{z:.4f}" for z in zl)
    )
    check(
        "[2] for every mode weight n in [1, 2] the sign flips with the background "
        "inside [3.5, 4]; at fixed background the n-dependence is 10x weaker",
        len(zl) == 5 and abs(dbgd) > 5 * abs(dn),
    )

    log("[2] throat units: which part of the background?")
    f3, f4 = FlowDW(3.0), FlowDW(4.0)
    for nn in (1.0, 2.0):
        pure = [throat_units_ba(f, f, n=nn) for f in (f3, f4)]
        check(
            f"[2] throat-units form reproduces the sign of b/a (n = {nn:g})",
            all(
                np.sign(x) == np.sign(f.threshold_ba(n=nn))
                for x, f in zip(pure, (f3, f4), strict=True)
            ),
        )
    sw = [
        (sh, throat_units_ba(f3, f4, shift=sh), throat_units_ba(f4, f3, shift=sh))
        for sh in (-1.0, 0.0, 1.0)
    ]
    log(
        "   hybrid (h from d_A, g from d_C), n = 1, vs alignment shift: "
        + "; ".join(
            f"shift {sh:+.0f}: (3,4) {x:+.3f}, (4,3) {y:+.3f}" for sh, x, y in sw
        )
    )
    log("   -> the A'/C' split depends on how the two crossovers are aligned: not used")
    lcrit = {}
    for f in (f3, f4):
        for nn in (1.0, 2.0):
            lcrit[(f.d, nn)] = brentq(
                lambda L, f=f, nn=nn: throat_units_ba(f, f, n=nn, l_end=L),
                0.42,
                0.7,
                xtol=1e-9,
            )
    shape = {
        nn: (
            throat_units_ba(f3, f3, n=nn, l_end=f4.l),
            throat_units_ba(f4, f4, n=nn, l_end=f3.l),
        )
        for nn in (1.0, 2.0)
    }
    l_dc = float(_ell(d_crit))
    log(
        "   endpoint test: zero of b/a in l_end at fixed crossover shape: "
        + ", ".join(f"shape d={k[0]:g}, n={k[1]:g}: {v:.4f}" for k, v in lcrit.items())
    )
    log(f"   physical l = l_(d-1): d=3 {f3.l:.4f}, d_crit {l_dc:.4f}, d=4 {f4.l:.4f}")
    for nn in (1.0, 2.0):
        log(
            f"   n = {nn:g}: d=3 shape with l(4): {shape[nn][0]:+.4f};  d=4 shape with l(3): "
            f"{shape[nn][1]:+.4f}"
        )
    check(
        "[2] at fixed crossover shape the sign is set by l = l_(d-1): every zero "
        "l_end lies between l(3) and l(4), and swapping the endpoint swaps the sign",
        all(f3.l < v < f4.l for v in lcrit.values())
        and all(shape[nn][0] < 0 < shape[nn][1] for nn in (1.0, 2.0)),
    )

    # ---------------- [3] the Liouville potential ----------------
    log("[3] Liouville normal form, V_L - 1 for d = 3, 4, 5, 6")
    prof = {}
    ok3 = []
    for dd in (3.0, d_crit, 4.0, 5.0, 6.0):
        fl = FlowDW(dd)
        X, W, dys = liouville_profile(fl)
        prof[dd] = (X, W)
        near = X < 1e-3
        # boundary coefficient: (V_L - 1) dy^2 -> (n^2 - 1)/4 - dy^2
        coef = np.median((W[near] / dys**2) * (X[near] * dys) ** 2)
        far = X > 0.8 * X.max()
        log(
            f"   d = {dd:.4f}: min (V_L-1)(y_b-y_*)^2 = {W.min():+.3f} at "
            f"(y_b-y)/(y_b-y_*) = {X[np.argmin(W)]:.3f}; boundary coefficient "
            f"{coef:+.4f} (expected (d-1)(d-3)/4 = {(dd - 1) * (dd - 3) / 4:+.4f}); "
            f"throat |V_L - 1| at {X.max():.0f} y_*: {np.abs(W[far]).max() / dys**2:.1e}"
        )
        if dd in (3.0, 4.0, 5.0, 6.0):
            ok3.append(
                abs(coef - (dd - 1) * (dd - 3) / 4) <= 2e-3
                and np.abs(W[far]).max() / dys**2 <= 1e-3
            )
    check(
        "[3] boundary barrier (d-1)(d-3)/4 and throat edge V_L = 1",
        len(ok3) == 4 and all(ok3),
    )

    failed = [k for k, v in CHECKS.items() if not v]
    if failed:
        log(
            f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed; FAILED: {failed}"
        )
        log("figure and cache not written: a check failed")
        sys.exit(1)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    cols = {
        3.0: "#c0392b",
        d_crit: "#7f7f7f",
        4.0: "#1f77b4",
        5.0: "#2ca02c",
        6.0: "#9467bd",
    }
    for dd, (X, W) in prof.items():
        lab = f"d = {dd:g}" if dd in (3.0, 4.0, 5.0, 6.0) else f"d = {dd:.4f} (b = 0)"
        ax.plot(
            X,
            W,
            color=cols[dd],
            lw=1.6 if dd != d_crit else 1.1,
            ls="-" if dd != d_crit else "--",
            label=lab,
        )
    ax.set_title("physical operator, n = d - 2", fontsize=10)
    for (dd, nn), st in {
        (3.0, 1.0): ("#c0392b", "-"),
        (3.0, 2.0): ("#c0392b", ":"),
        (4.0, 2.0): ("#1f77b4", "-"),
        (4.0, 1.0): ("#1f77b4", ":"),
    }.items():
        X, W, _ = liouville_profile(FlowDW(dd), n=nn)
        bx.plot(
            X,
            W,
            color=st[0],
            ls=st[1],
            lw=1.6,
            label=f"d = {dd:g} background, n = {nn:g}"
            + (" (physical)" if nn == dd - 2 else ""),
        )
    bx.set_title(
        r"mode weight $e^{nA}$ swapped (each at its own threshold)", fontsize=10
    )
    for a_ in (ax, bx):
        a_.axhline(0, color="k", lw=0.6)
        a_.set_xscale("log")
        a_.set_xlim(1e-2, 15)
        a_.set_ylim(-2.2, 2.5)
        a_.set_xlabel(r"$(y_b - y)/(y_b - y_\star)$  (boundary left, throat right)")
        a_.legend(fontsize=8, loc="upper right")
    ax.set_ylabel(r"$(V_L - \lambda_{\rm th})\,(y_b - y_\star)^2$")
    fig.tight_layout()
    fig.savefig(FIG, dpi=160)
    log(f"   figure written: {FIG}")

    np.savez(
        CACHE,
        d=dgrid,
        p=rows[:, 1],
        b_over_a_dw=ba1,
        b_over_a_r=ba2,
        d_crit=d_crit,
        d_crit_methods=np.array([dc1, dc2]),
        eps=eps,
        b_over_a_eps_d4=ba_eps4,
        b_over_a_eps_d3=ba_eps3,
        map_dbg=dbg,
        map_n=ngrid,
        map_b_over_a=M,
        map_zero_line=np.array(zl),
        l_end_crit=np.array([[k[0], k[1], v] for k, v in lcrit.items()]),
        readme="Threshold-mode throat ratio b/a for real d (two charts), "
        "its zero d_crit (three root-finders), the mode-weight test n = 1 + eps "
        "at the operator's own threshold, and the (d_bg, n) map.",
    )
    log(f"cache written: {CACHE}")
    failed = [k for k, v in CHECKS.items() if not v]
    log(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
