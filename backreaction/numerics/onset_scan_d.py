"""The finite-temperature onset coupling alpha(beta) of the magnetic brane in
AdS_{d+1} for d = 5, 6, from the probe limit to beta ~ 1e115-1e132, by two
independent methods, with d = 4 and d = 3 anchors.

METHOD 1 (horizon shooting, the d = 4 method of `onset_scan.py` in general d).
The brane is integrated outward from a regular horizon in the raw frame
(horizon r = 1, U'(1) = d, V(1) = Z(1) = 0, field b), labelled by
eps = beta_ext(d) - alpha b^2, beta_ext = d(d-1)/(d-2) (eps = beta_ext is the
probe limit, eps -> 0 the extremal throat).  `derivations/onset_scan_d.py`
derives the equations for real d, the horizon series, V = O(eps) and the
cancellation-free magnetic term -d expm1(-4V) + (d-2) eps e^{-4V}/(d-1), so eps
can be taken to 1e-60: there is no eps floor, because U and Z enter the V
equation only through V' (rounding beta_ext - eps in the U, Z equations moves V
in proportion to V).  The polarised lowest-Landau-level mode
(U e^{(d-3)Z} w')' + B_m e^{(d-3)Z-2V} w = 0 (derived from the linearised
Yang-Mills equations) is integrated in the same pass from horizon regularity,
w = 1 - (B_m/d) x + w2 x^2, and B_m is the lowest root of the source
c0 = w + r w'/(d-2) at r_max = r_c e^{32}, r_c = eps^{-1/p(d)}, with p(d) the
exponent of the throat's growing deformation.  The extraction is exact on r^0
and r^{-(d-2)}; its residual on the r^0 branch is (B/v) r^{-2}/(2(d-2)).

METHOD 2 (independent of method 1 except for the horizon series).  The
background is re-integrated with LSODA from the right-hand sides generated
symbolically by the derivation, sampled directly at Chebyshev nodes in
s = ln(r - 1) on [ln 1e-9, ln r_c + 16]; the eigenproblem is solved as a
Chebyshev-collocation generalised eigenvalue problem for psi = w/W,
W = (1 + x)^{-(d-2)/2} (1 + x/r_c)^{-(d-2)/2}, with the horizon condition
w_s = -(B_m x/d) w and the boundary condition c0 = 0 as rows; v_inf from a
least-squares fit in 1/r.  Resolution N = 300 and 400.

FRAME MAP (derived, [6] of the derivation).  With e^{2V} -> v_inf r^2,
U -> r^2 and T = d/(4 pi) in the raw frame (u_H = d/(4 pi T) = 1),
    B_c u_H^2 = B_m/v_inf,  beta = alpha (B u_H^2)^2 = (beta_ext - eps)/v_inf^2,
    alpha = (beta_ext - eps)/B_m^2,  T/sqrt(B) = (d/(4 pi)) sqrt(v_inf/B_m),
and with s = B_m/B_thr - 1, B_thr = d(d-1)/4 the throat's BF value,
    alpha/alpha_* = (1 - eps/beta_ext)/(1 + s)^2,  alpha_* = 16/(d(d-1)(d-2)).
So alpha < alpha_* whenever the onset lies above the throat's BF threshold
(s > 0); s < 0 is a state bound below it (as in d = 3).

CHECKS
 [1] d = 4: the onset_scan.npz points (alpha, beta, B_c) and the collocation
     ladder dk_throat.npz (direct solves at its betas) are reproduced;
 [2] d = 3: the finite-T onset on the exact RN-AdS4 brane
     (t0_threshold.rn_ads4_finite_t) and its exact v_inf = (eps/6)^2 are
     reproduced (an independent check of the frame map);
 [3] for d = 5, 6, every point of the scan: the root is the ground state (no
     zero of w before the normalisable regime, which is verified), U/r^2 -> 1,
     and the Hamiltonian constraint (not used in the integration) holds;
 [4] beta increases as eps decreases; alpha(beta) is strictly increasing,
     s > 0 and alpha < alpha_* on the whole scan;
 [5] methods 1 and 2 agree on B_m, v_inf, alpha and beta at eight eps per d;
 [6] refinement: X0 = 1e-7 -> 1e-6, 1e-8; rtol 1e-12 -> 1e-11; r_max e^{32}
     -> e^{40} r_c; the measured movement of alpha and beta is printed and
     bounded.
 [7] the approach to alpha_*: 1 - alpha/alpha_* and the local slope
     A = d ln beta/d(1/nu), nu = sqrt(s), are printed; A tends to the
     throat-quantisation value 8 pi/(d-2) (the lowest mode fits half a period
     of the throat oscillation r^{-(d-2)/2 +- i(d-2)nu/2} into ln r_c ~ ln(beta)/4)
     from below.

Run:  uv run python -m backreaction.numerics.onset_scan_d   (~24 min; --quick ~12 min)
"""

import sys
import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.optimize import brentq

from backreaction import paths
from backreaction.derivations import onset_scan_d as od

T0 = time.time()
CHECKS = {}
CACHE = paths.data("onset_scan_d.npz")
X0 = 1e-7  # start of the integration, r = 1 + X0
SPAN_UV = 32.0  # e-folds of r beyond r_c = eps^{-1/p}
RTOL = 1e-12
EPS_MIN = 1e-60
M2_X0 = 1e-9
M2_SPAN = 16.0
QUICK = "--quick" in sys.argv


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def p_exponent(d):
    """Growing deformation of AdS_{d-1} x R^2 in the r chart (closed form)."""
    return (d - 2) * (np.sqrt((d - 1) * (d + 15)) - (d - 1)) / (2 * (d - 1))


class Brane:
    """Raw-frame brane + polarised LLL mode in AdS_{d+1}."""

    def __init__(self, d):
        self.d = d
        self.be = d * (d - 1) / (d - 2)
        self.Bthr = d * (d - 1) / 4
        self.ast = 16 / (d * (d - 1) * (d - 2))
        self.p = p_exponent(d)
        ser, _, _ = od.horizon_series(sp.Integer(d))
        self._ser = {str(k): sp.lambdify(od.eps, v, "math") for k, v in ser.items()}
        Upp, Vpp, Zpp = od.closed_form_eps(sp.Integer(d))
        args = (od.Uv, od.Vv, od.Zv, od.Up, od.Vp, od.Zp, od.eps)
        mods = [{"expm1": np.expm1}, "numpy"]
        self._sym = [sp.lambdify(args, e, mods) for e in (Upp, Vpp, Zpp)]

    def coeffs(self, eps):
        c = {k: f(eps) for k, f in self._ser.items()}
        if self.d == 3:  # no z directions
            for k in ("z1", "z2", "z3"):
                c[k] = 0.0
        return c

    def background0(self, eps, x):
        c, d = self.coeffs(eps), self.d
        U = d * x + c["u2"] * x**2 + c["u3"] * x**3 + c["u4"] * x**4
        Up = d + 2 * c["u2"] * x + 3 * c["u3"] * x**2 + 4 * c["u4"] * x**3
        V = c["v1"] * x + c["v2"] * x**2 + c["v3"] * x**3
        Vp = c["v1"] + 2 * c["v2"] * x + 3 * c["v3"] * x**2
        Z = c["z1"] * x + c["z2"] * x**2 + c["z3"] * x**3
        Zp = c["z1"] + 2 * c["z2"] * x + 3 * c["z3"] * x**2
        return [U, V, Z, Up, Vp, Zp], c

    def initial(self, eps, Bm, x0):
        d = self.d
        bg, c = self.background0(eps, x0)
        w1 = -Bm / d
        p2 = c["u2"] + d * (d - 3) * c["z1"]
        q1 = (d - 3) * c["z1"] - 2 * c["v1"]
        w2 = -(2 * p2 * w1 + Bm * (w1 + q1)) / (4 * d)
        w = 1 + w1 * x0 + w2 * x0**2
        wp = w1 + 2 * w2 * x0
        U, Z = bg[0], bg[2]
        return bg + [w, U * np.exp((d - 3) * Z) * wp]

    def t_max(self, eps, span=SPAN_UV):
        return np.log(max(1.0, eps ** (-1 / self.p))) + span

    # ---- method 1 -------------------------------------------------------
    def shoot(self, eps, Bm, t_max, dense=False, x0=X0, rtol=RTOL):
        d, n3 = self.d, self.d - 3
        br = self.be - eps
        c = 1.0 / (d - 1)

        def rhs(t, y):
            x = np.exp(t)
            U, V, Z, Up, Vp, Zp, w, Pi = y
            e4 = np.exp(-4 * V)
            Upp = 2 * c * br * e4 - 2 * Up * Vp - n3 * Up * Zp + 2 * d
            Vpp = (
                (-d * np.expm1(-4 * V) + (d - 2) * c * eps * e4 - Up * Vp) / U
                - 2 * Vp**2
                - n3 * Vp * Zp
            )
            Zpp = (
                ((d + c * br * e4 - Up * Zp) / U - 2 * Vp * Zp - n3 * Zp**2)
                if n3
                else 0.0
            )
            eZ = np.exp(n3 * Z)
            return [
                x * Up,
                x * Vp,
                x * Zp,
                x * Upp,
                x * Vpp,
                x * Zpp,
                x * Pi / (U * eZ),
                -x * Bm * eZ * np.exp(-2 * V) * w,
            ]

        def node(t, y):
            return y[6]

        return solve_ivp(
            rhs,
            [np.log(x0), t_max],
            self.initial(eps, Bm, x0),
            method="DOP853",
            rtol=rtol,
            atol=1e-300,
            dense_output=dense,
            events=node,
        )

    def c0_of(self, y, t):
        U, V, Z, Up, Vp, Zp, w, Pi = y
        r = 1 + np.exp(t)
        return w + r * Pi / (U * np.exp((self.d - 3) * Z)) / (self.d - 2)

    def onset(self, eps, s_guess=None, x0=X0, rtol=RTOL, span=SPAN_UV):
        """Lowest B_m = B_thr (1 + s) with c0 = 0.  With a guess the root is
        bracketed locally, otherwise by the first sign change upward from
        B_m = 0.3 B_thr on a grid geometric in |s|.  Ground state checked
        afterwards (Sturm: no zero)."""
        tm = self.t_max(eps, span)

        def f(s):
            sol = self.shoot(eps, self.Bthr * (1 + s), tm, x0=x0, rtol=rtol)
            return self.c0_of(sol.y[:, -1], sol.t[-1])

        br = None
        if s_guess is not None and s_guess > 0:
            for lo_f, hi_f in ((0.97, 1.03), (0.85, 1.1), (0.5, 1.6)):
                lo, hi = lo_f * s_guess, hi_f * s_guess
                if f(lo) * f(hi) < 0:
                    br = (lo, hi)
                    break
        if br is None:
            # geometric in |s| on both sides of the threshold: near s = 0 the
            # excited states sit at s_n ~ n^2 s_1 and must not be stepped over
            grid = np.concatenate(
                [-np.geomspace(0.7, 1e-6, 120), np.geomspace(1e-6, 3.0, 130)]
            )
            prev = f(grid[0])
            for i in range(1, len(grid)):
                cur = f(grid[i])
                if cur * prev < 0:
                    br = (grid[i - 1], grid[i])
                    break
                prev = cur
            else:
                raise RuntimeError(
                    f"no sign change of c0 at d = {self.d}, eps = {eps:g}"
                )
        s = brentq(f, *br, xtol=1e-17, rtol=1e-15, maxiter=200)
        Bm = self.Bthr * (1 + s)
        sol = self.shoot(eps, Bm, tm, dense=True, x0=x0, rtol=rtol)
        return self._frame(eps, s, Bm, sol, tm)

    def _frame(self, eps, s, Bm, sol, tm):
        d = self.d
        ts = tm - np.array([0.0, np.log(2), np.log(4)])
        rr = 1 + np.exp(ts)
        Y = sol.sol(ts)
        M = np.stack([np.ones(3), 1 / rr, 1 / rr**2], axis=1)
        vinf = np.exp(2 * np.linalg.solve(M, Y[1] - np.log(rr))[0])
        uinf = np.linalg.solve(M, Y[0] / rr**2)[0]
        rc = max(1.0, vinf**-0.5)
        # node window: up to rc e^{Delta}, Delta half-way (in e-folds) to where
        # a 1e-15 root residual in c0 could put a spurious zero
        delta = min(12.0, 0.5 * np.log(1e15) / (d - 2))
        t_w = min(np.log(rc) + delta, tm)
        nodes = int(np.sum(sol.t_events[0] < t_w))
        yw = sol.sol(t_w)
        rw = 1 + np.exp(t_w)
        power = -(rw * yw[7] / (yw[0] * np.exp((d - 3) * yw[2]))) / ((d - 2) * yw[6])
        # Hamiltonian constraint along the solution, relative to d(d-1)
        tg = np.linspace(np.log(X0) + 1.0, tm - 1.0, 400)
        U, V, Z, Up, Vp, Zp = sol.sol(tg)[:6]
        br = self.be - eps
        con = (
            br * np.exp(-4 * V)
            + 2 * Up * Vp
            + (d - 3) * Up * Zp
            + 2 * U * Vp**2
            + 4 * (d - 3) * U * Vp * Zp
            + (d - 3) * (d - 4) * U * Zp**2
            - d * (d - 1)
        )
        scale = np.abs(br * np.exp(-4 * V)) + np.abs(2 * Up * Vp) + d * (d - 1)
        return dict(
            eps=eps,
            s=s,
            Bm=Bm,
            vinf=vinf,
            uinf=uinf,
            nodes=nodes,
            power=power,
            con=float(np.max(np.abs(con) / scale)),
            beta=br / vinf**2,
            alpha=br / Bm**2,
            Bc=Bm / vinf,
            T_sqrtB=d / (4 * np.pi) * np.sqrt(vinf / Bm),
        )

    # ---- method 2 -------------------------------------------------------
    def method2(self, eps, N):
        d, n3 = self.d, self.d - 3
        tm = self.t_max(eps, M2_SPAN)
        s0 = np.log(M2_X0)
        xi = np.cos(np.pi * np.arange(N + 1) / N)
        cc = np.hstack([2, np.ones(N - 1), 2]) * (-1) ** np.arange(N + 1)
        Xm = np.tile(xi, (N + 1, 1)).T
        D = np.outer(cc, 1 / cc) / (Xm - Xm.T + np.eye(N + 1))
        D = D - np.diag(D.sum(1))
        xi, D = xi[::-1], D[::-1, ::-1] * 2 / (tm - s0)
        s = s0 + (tm - s0) * (xi + 1) / 2
        s[0], s[-1] = s0, tm
        tf = tm + np.linspace(-3.0, 0.0, 7)
        allt = np.concatenate([s[1:], tf])
        ut, inv = np.unique(allt, return_inverse=True)
        y0, _ = self.background0(eps, M2_X0)

        def rhs(t, y):
            x = np.exp(t)
            a = (*y, eps)
            return [
                x * y[3],
                x * y[4],
                x * y[5],
                x * self._sym[0](*a),
                x * self._sym[1](*a),
                x * self._sym[2](*a) if n3 else 0.0,
            ]

        sol = solve_ivp(
            rhs, [s0, tm], y0, method="LSODA", rtol=1e-12, atol=1e-300, t_eval=ut
        )
        Y = sol.y[:, inv]
        yb, yf = Y[:, :N], Y[:, N:]
        rr = 1 + np.exp(tf)
        u = np.exp(tm - M2_SPAN) / rr  # r_c/r, so the columns are O(1)
        Af = np.stack([np.ones(7), u, u**2, u**3], axis=1)
        vinf = np.exp(2 * np.linalg.lstsq(Af, yf[1] - np.log(rr), rcond=None)[0][0])
        rc = max(1.0, vinf**-0.5)
        U, V, Z, Up, Vp, Zp = np.concatenate([np.array(y0)[:, None], yb], axis=1)
        x = np.exp(s)
        Pt = U * np.exp(n3 * Z) / x
        g = ((Up + n3 * Zp * U) * np.exp(n3 * Z) - Pt) / Pt  # (dPt/ds)/Pt
        xq = x * np.exp(n3 * Z - 2 * V) / Pt
        h = (d - 2) / 2
        ws = -h * (x / (1 + x) + (x / rc) / (1 + x / rc))
        wsp = -h * (x / (1 + x) ** 2 + (x / rc) / (1 + x / rc) ** 2)
        wss = ws**2 + wsp
        Id = np.eye(N + 1)
        A = D @ D + (2 * ws + g)[:, None] * D + np.diag(wss + g * ws)
        Mm = -np.diag(xq)
        A[0], Mm[0] = D[0] + ws[0] * Id[0], -(M2_X0 / d) * Id[0]
        A[-1] = Id[-1] + (1 + x[-1]) / (x[-1] * (d - 2)) * (D[-1] + ws[-1] * Id[-1])
        Mm[-1] = 0.0
        ev, vec = eig(A, Mm)
        ok = np.isfinite(ev) & (np.abs(ev.imag) < 1e-8 * np.abs(ev)) & (ev.real > 0)
        i = int(np.argmin(np.where(ok, ev.real, np.inf)))
        Bm = ev[i].real
        psi = vec[:, i].real
        nodes = int(
            np.sum(np.diff(np.sign(psi[np.abs(psi) > 1e-12 * np.abs(psi).max()])) != 0)
        )
        br = self.be - eps
        return dict(Bm=Bm, vinf=vinf, nodes=nodes, alpha=br / Bm**2, beta=br / vinf**2)

    # ---- direct solve at a prescribed beta ---------------------------------
    def at_beta(self, beta_t, eps_grid, beta_grid, s_grid):
        i = np.argsort(beta_grid)
        le = np.interp(np.log(beta_t), np.log(beta_grid[i]), np.log(eps_grid[i]))
        s0 = np.interp(np.log(beta_t), np.log(beta_grid[i]), s_grid[i])
        pts = []
        for le_ in (le, le + 1e-4):
            r = self.onset(np.exp(le_), s0)
            pts.append((le_, np.log(r["beta"] / beta_t), r))
        for _ in range(30):
            (l1, f1, _r1), (l2, f2, r2) = pts[-2], pts[-1]
            if abs(f2) < 1e-13:
                return r2
            l3 = l2 - f2 * (l2 - l1) / (f2 - f1)
            r = self.onset(np.exp(l3), r2["s"])
            pts.append((l3, np.log(r["beta"] / beta_t), r))
        raise RuntimeError(f"secant did not converge at beta = {beta_t:g}")


def scan(br, n):
    be = br.be
    eps_grid = np.concatenate(
        [be - be * np.geomspace(1e-3, 0.95, 20), np.geomspace(0.045 * be, EPS_MIN, n)]
    )
    eps_grid = np.unique(eps_grid)[::-1]
    rows, guess = [], None
    for e in eps_grid:
        if len(rows) >= 2:  # extrapolate ln s linearly in ln eps
            (e1, s1), (e2, s2) = [(r["eps"], r["s"]) for r in rows[-2:]]
            guess = np.exp(
                np.log(s2) + (np.log(s2 / s1)) * np.log(e / e2) / np.log(e2 / e1)
            )
        row = br.onset(e, guess)
        rows.append(row)
        guess = row["s"]
        if len(rows) % 25 == 0:
            log(
                f"      d = {br.d}: eps {e:.3e}  beta {row['beta']:.4e}  "
                f"alpha/alpha_* {row['alpha'] / br.ast:.10f}  B_c u_H^2 {row['Bc']:.6e}"
            )
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


def anchor_d4():
    log("=== d = 4 anchor")
    br = Brane(4)
    ref = np.load(paths.data("onset_scan.npz"))
    o = np.argsort(ref["beta"])
    idx = o[np.linspace(0, len(o) - 1, 8).round().astype(int)]
    dev = []
    log("      eps          beta              alpha (here)    alpha (onset_scan.npz)")
    for j in idx:
        r = br.onset(float(ref["eps"][j]), float(ref["nu"][j]) ** 2)
        dv = max(
            abs(r["alpha"] / ref["alpha"][j] - 1),
            abs(r["beta"] / ref["beta"][j] - 1),
            abs(r["Bc"] / ref["B_c"][j] - 1),
        )
        dev.append(dv)
        log(
            f"      {ref['eps'][j]:.3e}   {r['beta']:.6e}   {r['alpha']:.12f}  "
            f"{ref['alpha'][j]:.12f}   rel dev {dv:.1e}"
        )
    check(
        "[1] d=4: onset_scan.npz reproduced (alpha, beta, B_c)",
        max(dev) < 1e-9,
        f"worst relative deviation {max(dev):.1e}",
    )
    lad = np.load(paths.data("dk_throat.npz"))
    s_ref = ref["nu"] ** 2
    devl = []
    rungs = range(0, len(lad["beta"]), 3) if QUICK else range(len(lad["beta"]))
    for j in rungs:
        bL, aL, BL = lad["beta"][j], lad["alpha"][j], lad["B_c"][j]
        r = br.at_beta(float(bL), ref["eps"], ref["beta"], s_ref)
        devl.append(max(abs(r["alpha"] / aL - 1), abs(r["Bc"] / BL - 1)))
    check(
        f"[1] d=4: the collocation ladder ({len(devl)} of {len(lad['beta'])} rungs, beta = "
        f"{lad['beta'].min():g} ... {lad['beta'].max():g}) reproduced",
        max(devl) < 1e-8,
        f"worst relative deviation (alpha, B_c) {max(devl):.1e}",
    )
    return np.array(dev), np.array(devl)


def anchor_d3():
    log("=== d = 3 anchor: exact non-extremal RN-AdS4")
    from backreaction.numerics.t0_threshold import rn_ads4_finite_t

    br = Brane(3)
    qs = np.array([2.0, 2.9, 2.99, 2.999, 2.9999])
    out = []
    for q in qs:
        e = 6.0 - 2 * q
        r = br.onset(e)
        a_rn = rn_ads4_finite_t(q)
        v_ex = (e / 6.0) ** 2
        out.append((q, r["alpha"], a_rn, r["vinf"], v_ex, r["s"], r["nodes"]))
        log(
            f"      q = {q:<7g} alpha {r['alpha']:.12f}  RN-AdS4 {a_rn:.12f}  "
            f"v_inf {r['vinf']:.10e}  exact {v_ex:.10e}  s = {r['s']:+.5f}"
        )
    out = np.array(out)
    da = np.max(np.abs(out[:, 1] / out[:, 2] - 1))
    dv = np.max(np.abs(out[:, 3] / out[:, 4] - 1))
    check(
        "[2] d=3: RN-AdS4 onset alpha and exact v_inf reproduced",
        da < 1e-9 and dv < 1e-9 and np.all(out[:, 6] == 0),
        f"alpha {da:.1e}, v_inf {dv:.1e}; alpha passes 8/3 with s < 0 at q = "
        f"{qs[np.argmax(out[:, 5] < 0)]:g}",
    )
    return out


def main():
    quick = QUICK
    n = 45 if quick else 120
    res = {}
    dev4, devl4 = anchor_d4()
    d3 = anchor_d3()
    for d in (5, 6):
        log(
            f"=== d = {d}: alpha_* = {16 / (d * (d - 1) * (d - 2)):.10f}, "
            f"beta_ext = {d * (d - 1) / (d - 2):.6f}, B_thr = {d * (d - 1) / 4:g}, "
            f"p = {p_exponent(d):.6f}"
        )
        br = Brane(d)
        S = scan(br, n)
        res[d] = (br, S)
        check(
            f"[3] d={d}: every root is the ground state, U/r^2 -> 1, constraint holds",
            np.all(S["nodes"] == 0)
            and np.max(np.abs(S["power"] - 1)) < 0.05
            and np.max(np.abs(S["uinf"] - 1)) < 1e-9
            and np.max(S["con"]) < 1e-9,
            f"{len(S['eps'])} points; max |U/r^2 - 1| = {np.max(np.abs(S['uinf'] - 1)):.1e}, "
            f"constraint {np.max(S['con']):.1e}, normalisable-branch power at the window "
            f"edge within {np.max(np.abs(S['power'] - 1)):.1e} of 1",
        )
        o = np.argsort(-S["eps"])
        beta, alpha = S["beta"][o], S["alpha"][o]
        check(
            f"[4] d={d}: beta increases as eps decreases; alpha(beta) strictly increasing; "
            "s > 0 and alpha < alpha_* throughout",
            np.all(np.diff(beta) > 0)
            and np.all(np.diff(alpha) > 0)
            and np.all(S["s"] > 0)
            and alpha.max() < br.ast,
            f"beta {beta.min():.3e} ... {beta.max():.3e}; alpha/alpha_* "
            f"{alpha.min() / br.ast:.3e} ... {alpha.max() / br.ast:.8f}; "
            f"min relative increment {np.min(np.diff(alpha) / alpha[1:]):.1e}",
        )
        ident = np.max(
            np.abs(alpha / br.ast - (1 - S["eps"][o] / br.be) / (1 + S["s"][o]) ** 2)
        )
        log(
            f"      alpha/alpha_* = (1 - eps/beta_ext)/(1 + s)^2 on the scan to {ident:.1e}"
        )

    # [5] method 2
    log("=== method 2: LSODA background + Chebyshev eigenproblem")
    m2rows = {}
    for d in (5, 6):
        br, S = res[d]
        eps_list = [br.be * 0.5, 1.0, 1e-3, 1e-8, 1e-16, 1e-30, 1e-45, EPS_MIN]
        rows = []
        for e in eps_list:
            r1 = br.onset(
                e, float(np.interp(np.log(e), np.log(S["eps"][::-1]), S["s"][::-1]))
            )
            a, b_ = br.method2(e, 300), br.method2(e, 400)
            dB = abs(b_["Bm"] / r1["Bm"] - 1)
            dv = abs(b_["vinf"] / r1["vinf"] - 1)
            dN = abs(b_["Bm"] / a["Bm"] - 1)
            rows.append(
                (e, r1["beta"], r1["alpha"], b_["alpha"], dB, dv, dN, b_["nodes"])
            )
            log(
                f"      d = {d}  eps {e:.1e}  beta {r1['beta']:.4e}  alpha {r1['alpha']:.12f} "
                f"(shoot) {b_['alpha']:.12f} (Chebyshev)  dB_m {dB:.1e}  dv_inf {dv:.1e}  "
                f"N 300->400 {dN:.1e}"
            )
        rows = np.array(rows)
        m2rows[d] = rows
        check(
            f"[5] d={d}: shooting and Chebyshev agree on B_m and v_inf at {len(rows)} eps "
            f"(beta {rows[:, 1].min():.1e} ... {rows[:, 1].max():.1e}); eigenvector node-free",
            np.max(rows[:, 4]) < 2e-9
            and np.max(rows[:, 5]) < 1e-9
            and np.all(rows[:, 7] == 0),
            f"max |dB_m/B_m| = {np.max(rows[:, 4]):.1e}, max |dv_inf/v_inf| = "
            f"{np.max(rows[:, 5]):.1e}, Chebyshev N 300 -> 400 {np.max(rows[:, 6]):.1e}",
        )

    # [6] refinement
    log("=== refinement (method 1)")
    ref_rows = {}
    for d in (5, 6):
        br, S = res[d]
        out = []
        for e in (1.0, 1e-30, EPS_MIN):
            sg = float(np.interp(np.log(e), np.log(S["eps"][::-1]), S["s"][::-1]))
            base = br.onset(e, sg)
            var = {
                "X0 1e-6": br.onset(e, sg, x0=1e-6),
                "X0 1e-8": br.onset(e, sg, x0=1e-8),
                "rtol 1e-11": br.onset(e, sg, rtol=1e-11),
                "r_max e^40 r_c": br.onset(e, sg, span=40.0),
            }
            mv = {
                k: max(
                    abs(v["alpha"] / base["alpha"] - 1),
                    abs(v["beta"] / base["beta"] - 1),
                )
                for k, v in var.items()
            }
            out.append([mv[k] for k in var])
            log(
                f"      d = {d}  eps {e:.0e}: "
                + ", ".join(f"{k}: {v:.1e}" for k, v in mv.items())
            )
        out = np.array(out)
        ref_rows[d] = out
        check(
            f"[6] d={d}: alpha and beta move by less than 1e-9 under every refinement",
            out.max() < 1e-9,
            f"largest relative movement {out.max():.1e}",
        )

    # approach to alpha_*
    log("=== approach to alpha_*")
    summary = {}
    for d in (5, 6):
        br, S = res[d]
        o = np.argsort(S["beta"])
        lb, nu = np.log(S["beta"][o]), np.sqrt(S["s"][o])
        A = np.gradient(lb, 1 / nu)
        for L in (10, 20, 50, 100, 150, 200, 250, 300):
            if lb.min() < L < lb.max():
                i = np.searchsorted(lb, L)
                log(
                    f"      d = {d}  ln beta {lb[i]:7.2f}  beta {np.exp(lb[i]):.3e}  "
                    f"alpha/alpha_* {S['alpha'][o][i] / br.ast:.8f}  nu {nu[i]:.5f}  "
                    f"A {A[i]:.3f}  T/sqrt(B) {S['T_sqrtB'][o][i]:.3e}"
                )
        summary[d] = dict(
            A_end=A[-3], amax=S["alpha"].max() / br.ast, bmax=S["beta"].max()
        )
        Aq = 8 * np.pi / (d - 2)
        check(
            f"[7] d={d}: the local slope A tends to the throat-quantisation value 8 pi/(d-2) "
            "from below",
            abs(A[-3] / Aq - 1) < 2e-3 and A[-3] < Aq,
            f"A = {A[-3]:.5f} at ln beta = {lb[-3]:.1f}, 8 pi/(d-2) = {Aq:.5f}",
        )
        log(
            f"      d = {d}: alpha_max/alpha_* = {summary[d]['amax']:.8f} at beta = "
            f"{summary[d]['bmax']:.3e}; local A = {A[-3]:.3f} (8 pi/(d-2) = {8 * np.pi / (d - 2):.3f})"
        )

    failed = [k for k, v in CHECKS.items() if not v]
    if quick:
        log("--quick: cache not written")
    elif not failed:
        out = dict(
            readme="Onset coupling of the magnetic brane in AdS_{d+1}, d = 5, 6, by horizon "
            "shooting in the raw frame (eps = beta_ext - alpha b^2, beta_ext = d(d-1)/(d-2)): "
            "beta = alpha (B u_H^2)^2 = (beta_ext - eps)/v_inf^2, alpha = (beta_ext - eps)/B_m^2, "
            "B_c u_H^2 = B_m/v_inf, T/sqrt(B) = (d/4pi) sqrt(v_inf/B_m), s = B_m/B_thr - 1 with "
            "B_thr = d(d-1)/4, alpha/alpha_* = (1 - eps/beta_ext)/(1+s)^2.  m2_d5, m2_d6: rows "
            "(eps, beta, alpha shoot, alpha Chebyshev, |dB_m|/B_m, |dv_inf|/v_inf, Chebyshev "
            "N 300->400, nodes).  refine_d5, refine_d6: rows eps = 1, 1e-30, 1e-60; columns X0 1e-6, "
            "X0 1e-8, rtol 1e-11, r_max e^40 r_c (relative movement).  d3_anchor: rows "
            "(q, alpha, alpha RN-AdS4, v_inf, exact v_inf, s, nodes).  d4_dev: relative "
            "deviation from onset_scan.npz; d4_ladder_dev: from dk_throat.npz.",
            d4_dev=dev4,
            d4_ladder_dev=devl4,
            d3_anchor=d3,
        )
        for d in (5, 6):
            br, S = res[d]
            o = np.argsort(S["beta"])
            for k, kk in (
                ("eps", "eps"),
                ("beta", "beta"),
                ("alpha", "alpha"),
                ("B_c", "Bc"),
                ("s", "s"),
                ("T_sqrtB", "T_sqrtB"),
                ("nodes", "nodes"),
            ):
                out[f"{k}_d{d}"] = S[kk][o]
            out[f"alpha_star_d{d}"] = br.ast
            out[f"m2_d{d}"] = m2rows[d]
            out[f"refine_d{d}"] = ref_rows[d]
        np.savez(CACHE, **out)
        log(f"cache written: {CACHE}")
    log(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if failed:
        sys.exit(1)
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
