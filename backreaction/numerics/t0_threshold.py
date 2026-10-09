"""The zero-temperature threshold problem: alpha_* is the exact T = 0 critical
coupling, not only the point where the throat's Breitenlohner-Freedman bound
is saturated.

THE QUESTION.  Violating the BF bound of the AdS_{d-1} x R^2 throat is
sufficient for a T = 0 instability but not necessary: a bound state can sit in
the crossover region between the throat and the AdS_{d+1} boundary while the
throat's bound holds.  At T = 0 the background is a single geometry (beta sets
only the scale), so with B = 1 fixed in the geometry, varying the coupling at
fixed beta moves only the coefficient of the magnetic-moment term in the
polarised lowest-Landau-level operator,

    -(e^{(d-2)A} w')' = lambda e^{(d-2)A - 2C} w,       lambda = sqrt(alpha_*/alpha),

a Sturm-Liouville problem in lambda on the whole flow (domain-wall gauge of
`derivations/brane_flows.py`, source-free at the AdS_{d+1} boundary).  Its
continuum starts at the throat's BF threshold lambda = 1, and the physical
operator at coupling alpha is the one at lambda = sqrt(alpha_*/alpha).  The
normal phase at T = 0 is stable for alpha >= alpha_* exactly when this problem
has no discrete eigenvalue below lambda = 1.  By Sturm oscillation that number
is the number of zeros of the threshold solution (lambda = 1, source-free at the
boundary), including a zero deep in the throat if its asymptotic form
(r - r0)^{(d-2)/2} w -> a + b ln((r - r0)/(r_* - r0)) has b/a > 0.

THE COMPUTATION.  For d = 3, 4, 5, 6 at B = 1, alpha = alpha_*(d):
 [1] the T = 0 flow, integrated out of the fixed point along its single growing
     deformation (exponent from brane_flows), reaches AdS_{d+1} (A', C' -> 1,
     constraint satisfied);
 [2] the threshold solution, integrated in from the boundary, has no node on
     the flow, and b/a < 0 in the throat, so no node at infinity either; for
     d = 4 b/a is compared with the r-chart value of `dk_throat_matching.py`
     (independent code and gauge; both integrate with DOP853);
 [3] for lambda < 1 (alpha > alpha_*) the boundary solution has no node and
     reaches the throat on the non-normalisable branch, and, independently,
     the IR-normalisable solution integrated outward always has a non-zero
     source c_0 (no bound state), at lambda = 0.5 ... 0.995;
 [4] the threshold solution's throat exponents are degenerate, (r - r0)^{(d-2)/2} w
     exactly linear in ln(r - r0): the numerically built throat saturates the BF
     bound at alpha_*(d) = 16/(d(d-1)(d-2)).

RESULT.  For d = 4, 5, 6 the threshold solution has no zero (b/a = -0.0350,
-0.350, -0.996): alpha_*(d) is the exact T = 0 critical coupling.  For d = 3 it
does NOT hold: the threshold solution has one zero (b/a = +0.123), so a bound
state sits in the AdS2 x R^2 -> AdS4 crossover below the BF threshold.  It
detaches at lambda_c = 0.937047, i.e. the T = 0 critical coupling of the AdS4
magnetic brane is alpha_c(3) = alpha_*(3)/lambda_c^2 = 3.0370 = 1.1389 alpha_*(3).
 [5] the d = 3 result recomputed on the exact extremal magnetic RN-AdS4 brane
     (analytic background, r chart; DOP853 again): same zero, same
     lambda_c to 1e-11;
 [6] at finite temperature on the non-extremal RN-AdS4 brane the onset coupling
     rises past 8/3 and approaches 3.0370 from below as T -> 0; it crosses 8/3
     at q = alpha B^2/2 = 2.98858887 in horizon units, T/sqrt(B) = 7.42e-4
     with T = (3 - q)/(4 pi), B = B_m (independently root-found in
     `d3_blind_check.py`).

Run:  uv run python -m backreaction.numerics.t0_threshold   (~40 s)
"""

import sys
import time

import numpy as np
import sympy as sp
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq

from backreaction import paths
from backreaction.derivations import brane_flows as bf

T0 = time.time()
CHECKS = {}
CACHE = paths.data("t0_threshold.npz")
EPS0 = 1e-10  # size of the deformation at rho = 0
RHO_IR = -60.0
#: the result: d = 3 has one bound state below the AdS2 BF threshold, d >= 4 none
EXPECTED_NODES = {3: 1, 4: 0, 5: 0, 6: 0}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


class Flow:
    """The T = 0 brane at B = 1, alpha = alpha_*(d), in the domain-wall gauge."""

    def __init__(self, d, sign=+1):
        self.d = d
        dw = bf.domain_wall(d)
        L, Vf = bf.fixed_point(d, dw)
        ast = sp.Rational(16, d * (d - 1) * (d - 2))
        sub = {bf.B: 1, bf.alpha: ast}
        self.l = float(L.subs(sub))
        self.v = float(Vf.subs(sub))
        k, a = [(k, a) for k, a in bf.irrelevant(d, dw, L, Vf) if sp.N(k) > 0][0]
        self.k = float(k.subs(sub))
        self.ak = float(sp.N(a.subs(sub)))
        a1, c1, cc = dw["syms"]
        self.fA = sp.lambdify((a1, c1, cc), dw["App"].subs(sub), "numpy")
        self.fC = sp.lambdify((a1, c1, cc), dw["Cpp"].subs(sub), "numpy")
        self.fK = sp.lambdify((a1, c1, cc), dw["con"].subs(sub), "numpy")
        self.C0 = 0.5 * np.log(self.v)
        self.eps = sign * EPS0
        y0 = [
            self.ak * self.eps,
            self.C0 + self.eps,
            1 / self.l + self.ak * self.eps * self.k,
            self.eps * self.k,
        ]

        def rhs(r, y):
            A, C, Ap, Cp = y
            return [Ap, Cp, self.fA(Ap, Cp, C), self.fC(Ap, Cp, C)]

        def blowup(r, y):
            return 50.0 - abs(y[3]) - abs(y[2])

        blowup.terminal = True
        self.sol = solve_ivp(
            rhs,
            [0.0, 120.0],
            y0,
            method="DOP853",
            rtol=1e-13,
            atol=1e-15,
            dense_output=True,
            events=blowup,
        )
        self.rho_end = self.sol.t[-1]

    def fields(self, rho):
        """(A, C, A', C') at rho; below rho = 0 the linearised deformation."""
        rho = np.atleast_1d(np.asarray(rho, float))
        out = np.empty((4, rho.size))
        lo = rho < 0
        e = self.eps * np.exp(self.k * rho[lo])
        out[:, lo] = [
            rho[lo] / self.l + self.ak * e,
            self.C0 + e,
            1 / self.l + self.ak * self.k * e,
            self.k * e,
        ]
        if (~lo).any():
            out[:, ~lo] = self.sol.sol(rho[~lo])
        return out

    def rho_star(self):
        """e^{2C} = 2v."""
        return brentq(
            lambda r: np.exp(2 * self.fields(r)[1][0]) - 2 * self.v, 0.0, self.rho_end
        )


def mode(flow, lam, rho_a, rho_b, w0, dw0, rtol=1e-12):
    d = flow.d

    def rhs(r, y):
        A, C, Ap, Cp = flow.fields(r)[:, 0]
        return [y[1], -(d - 2) * Ap * y[1] - lam * np.exp(-2 * C) * y[0]]

    def node(r, y):
        return y[0]

    return solve_ivp(
        rhs,
        [rho_a, rho_b],
        [w0, dw0],
        method="DOP853",
        rtol=rtol,
        atol=1e-300,
        dense_output=True,
        events=node,
    )


def run_d(d):
    log(
        f"=== d = {d}:  alpha_* = 16/(d(d-1)(d-2)) = {16 / (d * (d - 1) * (d - 2)):.6f}"
    )
    flow = Flow(d)
    out = dict(d=d, l=flow.l, v=flow.v, k=flow.k, p=flow.k * flow.l)
    # [1] the flow reaches AdS_{d+1}
    rs = np.linspace(0, flow.rho_end, 4000)
    A, C, Ap, Cp = flow.fields(rs)
    con = np.max(np.abs(flow.fK(Ap, Cp, C)))
    check(
        f"[1] d={d}: flow from AdS_{d - 1} x R^2 reaches AdS_{d + 1}",
        flow.rho_end > 110
        and abs(Ap[-1] - 1) < 1e-10
        and abs(Cp[-1] - 1) < 1e-10
        and con < 1e-9,
        f"A' - 1 = {Ap[-1] - 1:.1e}, C' - 1 = {Cp[-1] - 1:.1e}, constraint {con:.1e}, "
        f"p = {out['p']:.6f}",
    )
    other = Flow(d, sign=-1)
    log(
        f"      opposite sign of the deformation ends at rho = {other.rho_end:.2f} "
        f"(C' = {other.fields(other.rho_end)[3][0]:.3g}): not asymptotically AdS"
    )
    rho_s = flow.rho_star()
    rho_uv = min(flow.rho_end, rho_s + 45.0)

    # r - r0 = int_{-oo}^rho e^A  (the r chart has g_tt g_rr = -1, U = e^{2A})
    grid = np.concatenate(
        [np.linspace(RHO_IR, 0, 6001)[:-1], np.linspace(0, rho_uv, 20001)]
    )
    Ag = flow.fields(grid)[0]
    R0 = flow.l * np.exp(RHO_IR / flow.l)
    R = R0 + cumulative_trapezoid(np.exp(Ag), grid, initial=0.0)
    Rs = np.interp(rho_s, grid, R)

    # [2] the threshold solution from the boundary inward
    n = d - 2
    th = mode(flow, 1.0, rho_uv, RHO_IR, np.exp(-n * rho_uv), -n * np.exp(-n * rho_uv))
    nodes = len(th.t_events[0])
    sel = (grid > -45) & (grid < -12)
    w = th.sol(grid[sel])[0]
    y = R[sel] ** (n / 2) * w
    xg = np.log(R[sel] / Rs)
    bco, aco = np.polyfit(xg, y, 1)
    lin_err = np.max(np.abs(np.polyval([bco, aco], xg) - y)) / abs(aco)
    out.update(nodes=nodes, b_over_a=bco / aco, lin_err=lin_err)
    # [4] degenerate exponents: R^{(d-2)/2} w exactly linear in ln R in the throat
    check(
        f"[4] d={d}: (r-r0)^((d-2)/2) w is linear in ln(r-r0) in the throat (BF saturation)",
        lin_err < 1e-6,
        f"linear-fit error {lin_err:.1e}",
    )
    # zeros of a + b x deep in the throat (x < grid window) are zeros too
    x_zero = -aco / bco
    tail_node = int(bco / aco > 0 and x_zero < xg.min())
    out["total_nodes"] = nodes + tail_node
    check(
        f"[2] d={d}: zeros of the threshold solution (= bound states below the BF "
        f"threshold): {EXPECTED_NODES[d]}",
        out["total_nodes"] == EXPECTED_NODES[d]
        and (bco / aco < 0) == (EXPECTED_NODES[d] == 0),
        f"{nodes} on the range + {tail_node} in the throat tail, b/a = {bco / aco:+.6f}",
    )
    log(
        f"      threshold solution: {nodes} node(s) on the integration range, b/a = "
        f"{bco / aco:+.6f}"
        + (f", a + b x = 0 at x = {x_zero:.2f}" if bco / aco > 0 else "")
    )

    # the IR-normalisable solution below threshold, integrated outward; its
    # boundary source c0, in units of w(rho_*), vanishes exactly at a bound state
    def source(lam):
        q = np.sqrt(n**2 / 4 - lam * flow.l**2 / flow.v)
        sp_ = (-n / 2 + q) / flow.l
        mi = mode(
            flow, lam, RHO_IR, rho_uv, np.exp(sp_ * RHO_IR), sp_ * np.exp(sp_ * RHO_IR)
        )
        wu, dwu = mi.sol(rho_uv)
        return (wu + dwu / n) / mi.sol(rho_s)[0]

    def nodes_from_boundary(lam):
        m = mode(
            flow, lam, rho_uv, RHO_IR, np.exp(-n * rho_uv), -n * np.exp(-n * rho_uv)
        )
        return len(m.t_events[0])

    lams = np.array([0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995])
    nodes_l = np.array([nodes_from_boundary(x) for x in lams])
    src = np.array([source(x) for x in lams])
    out.update(lams=lams, nodes_below=nodes_l, src_below=src)
    log("      lambda = sqrt(alpha_*/alpha): " + ", ".join(f"{x:g}" for x in lams))
    log("      nodes of the boundary solution: " + ", ".join(str(x) for x in nodes_l))
    log(
        "      source c0/w(rho_*) of the IR-normalisable solution: "
        + ", ".join(f"{x:+.4f}" for x in src)
    )
    if out["total_nodes"] == 0:
        check(
            f"[3] d={d}: below threshold no node and no bound state (c0 never vanishes)",
            np.all(nodes_l == 0) and (np.all(src > 1e-3) or np.all(src < -1e-3)),
        )
        out["lam_c"] = np.nan
    else:
        # a bound state below threshold: locate where it detaches
        i = np.where(np.sign(src[1:]) != np.sign(src[:-1]))[0]
        lam_c = (
            brentq(source, lams[i[0]], lams[i[0] + 1], xtol=1e-12)
            if len(i) == 1
            else np.nan
        )
        out["lam_c"] = lam_c
        jump = np.where(np.diff(nodes_l) != 0)[0]
        check(
            f"[3] d={d}: one bound state below threshold; it detaches at lambda_c "
            "where the boundary solution gains its node",
            len(i) == 1
            and len(jump) == 1
            and lams[jump[0]] < lam_c < lams[jump[0] + 1],
            f"lambda_c = {lam_c:.8f}, alpha_c/alpha_* = 1/lambda_c^2 = {1 / lam_c**2:.6f}",
        )
    return out


def rn_ads4_lambda_c():
    """Independent d = 3 computation on the exact extremal magnetic RN-AdS4 brane
    (r chart, no flow integration): with R_MN + 3 g_MN = alpha(F F - F^2 g/4),
    f = 1 - m/r^3 + alpha B^2/(2 r^4); extremal at r = 1: m = 4, alpha B^2 = 6,
    so at alpha_*(3) = 8/3 the field is B_* = 3/2 and f = (r-1)^2(r^2+2r+3)/r^4.
    Mode: (r^2 f w')' + lambda B_* w / r^2 = 0 (P = U, Q = B e^{-2V}).
    Returns (zeros of the threshold solution, lambda_c)."""
    Bst = 1.5

    def solve(lam, s_a, s_b, y0):
        def rhs(s, y):  # s = ln(r - 1), y = (w, dw/ds)
            x = np.exp(s)
            r = 1 + x
            g = (r * r + 2 * r + 3) / (r * r)
            gp = (-2 * r - 6) / r**3
            u, v = y
            return [
                v,
                (-lam * Bst * u * x / r**2 - x * g * v - x * x * gp * v) / (x * g),
            ]

        return solve_ivp(
            rhs,
            [s_a, s_b],
            y0,
            method="DOP853",
            rtol=1e-12,
            atol=1e-300,
            dense_output=True,
            events=lambda s, y: y[0],
        )

    top, bot = 30.0, -40.0
    xt = np.exp(top)
    th = solve(1.0, top, bot, [1 / (1 + xt), -xt / (1 + xt) ** 2])
    S = np.linspace(-35, -12, 400)
    b, a = np.polyfit(S, np.exp(S / 2) * th.sol(S)[0], 1)
    zeros = len(th.t_events[0]) + int(b / a > 0 and -a / b < S.min())

    def source(lam):  # IR-normalisable (r-1)^{s+}, s+ = (-1 + sqrt(1 - lam))/2
        sp_ = (-1 + np.sqrt(1 - lam)) / 2
        m = solve(lam, bot, top, [np.exp(sp_ * bot), sp_ * np.exp(sp_ * bot)])
        u, v = m.sol(top)
        x = np.exp(top)
        # w = c0 + c1/r: c0 = w + r w_r = w + (1 + x) v / x
        return (u + (1 + x) * v / x) / m.sol(0.0)[0]

    return zeros, brentq(source, 0.85, 0.99, xtol=1e-12)


def rn_ads4_finite_t(q):
    """Onset on the NON-extremal magnetic RN-AdS4 brane, horizon r = 1,
    r^2 f = (r-1)(r^3 + r^2 + r - q)/r^2 with q = alpha B^2/2 in [0, 3): the lowest
    B_m with a regular, source-free zero mode, and alpha_c = 2q/B_m^2."""
    k = 3.0 - q

    def c0(Bm, top=30.0):
        def rhs(t, y):
            x = np.exp(t)
            r = 1 + x
            return [
                x * y[1] * r**2 / (x * (r**3 + r * r + r - q)),
                -x * Bm * y[0] / r**2,
            ]

        x0 = 1e-9
        so = solve_ivp(
            rhs,
            [np.log(x0), top],
            [1 - Bm / k * x0, -Bm * x0],
            method="DOP853",
            rtol=1e-11,
            atol=1e-300,
        )
        w, Pi = so.y[:, -1]
        x = np.exp(top)
        r = 1 + x
        return w + r * Pi * r**2 / (x * (r**3 + r * r + r - q))

    Bs = np.linspace(0.3, 6.0, 120)
    vals = [c0(b) for b in Bs]
    i = next(j for j in range(len(Bs) - 1) if vals[j] * vals[j + 1] < 0)
    Bm = brentq(c0, Bs[i], Bs[i + 1], xtol=1e-14)
    return 2 * q / Bm**2


def main():
    res = [run_d(d) for d in (3, 4, 5, 6)]
    log("=== d = 3 on the exact extremal magnetic RN-AdS4 brane (second method)")
    z_rn, lam_rn = rn_ads4_lambda_c()
    r3 = [r for r in res if r["d"] == 3][0]
    check(
        "[5] d=3: exact RN-AdS4 gives the same bound state and lambda_c (to 1e-11)",
        z_rn == 1 and abs(lam_rn / r3["lam_c"] - 1) < 1e-11,
        f"zeros {z_rn}, lambda_c = {lam_rn:.13f} vs flow {r3['lam_c']:.13f}",
    )
    a3 = (8 / 3) / lam_rn**2
    log(
        f"      d = 3: T = 0 critical coupling alpha_c = alpha_*(3)/lambda_c^2 = "
        f"{a3:.6f}  (BF value 8/3 = {8 / 3:.6f})"
    )
    log("=== d = 3 at finite temperature: onset coupling on non-extremal RN-AdS4")
    qs = np.array([2.0, 2.9, 2.99, 2.999, 2.9999, 2.99999, 2.999999])
    a_fin = np.array([rn_ads4_finite_t(q) for q in qs])
    log("      q = alpha B^2/2: " + ", ".join(f"{q:g}" for q in qs))
    log("      alpha_c(q):      " + ", ".join(f"{a:.5f}" for a in a_fin))
    check(
        "[6] d=3: the finite-T onset coupling rises past 8/3 and stays below alpha_c(T=0)",
        np.all(np.diff(a_fin) > 0) and a_fin[-1] > 8 / 3 and a_fin.max() < a3,
        f"alpha_c = {a_fin[-1]:.5f} at q = {qs[-1]} vs T = 0 value {a3:.5f}",
    )
    q_x = brentq(lambda q: rn_ads4_finite_t(q) - 8 / 3, 2.9, 2.999, xtol=1e-13)
    t_x = (3 - q_x) / (4 * np.pi * np.sqrt(np.sqrt(3 * q_x / 4)))  # B_m^2 = 2q/(8/3)
    check(
        "[6] d=3: the onset coupling crosses 8/3 at T/sqrt(B) = 7.42e-4",
        round(t_x, 6) == 7.42e-4,
        f"q = {q_x:.10f}, T/sqrt(B) = {t_x:.6e}",
    )
    ref = float(np.load(paths.data("dk_throat_matching.npz"))["log_ratio_b_over_a"])
    r4 = [r for r in res if r["d"] == 4][0]
    check(
        "[2] d=4: b/a = -0.0350004 at the doubling point r_*, and agrees with the "
        "r-chart computation of dk_throat_matching to 1e-5",
        round(r4["b_over_a"], 7) == -0.0350004 and abs(r4["b_over_a"] / ref - 1) < 1e-5,
        f"{r4['b_over_a']:.9f} vs {ref:.9f}",
    )
    log(
        "summary   d   p (exponent)   bound states below BF   b/a          alpha_c/alpha_*"
    )
    for r in res:
        log(
            f"          {r['d']}   {r['p']:.6f}      {r['total_nodes']}                       "
            f"{r['b_over_a']:+.6f}    {1 / r['lam_c'] ** 2 if np.isfinite(r['lam_c']) else 1.0:.6f}"
        )
    failed = [k for k, v in CHECKS.items() if not v]
    log(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if failed:
        log("cache not written")
        sys.exit(1)
    np.savez(
        CACHE,
        d=np.array([r["d"] for r in res]),
        p=np.array([r["p"] for r in res]),
        nodes=np.array([r["total_nodes"] for r in res]),
        b_over_a=np.array([r["b_over_a"] for r in res]),
        lams=res[0]["lams"],
        nodes_below=np.array([r["nodes_below"] for r in res]),
        src_below=np.array([r["src_below"] for r in res]),
        lam_c=np.array([r["lam_c"] for r in res]),
        alpha_c_d3=a3,
        d3_finite_t_q=qs,
        d3_finite_t_alpha=a_fin,
        d3_cross_q=q_x,
        d3_cross_T_over_sqrtB=t_x,
        readme="T = 0 threshold problem on the AdS_{d+1} -> AdS_{d-1} x R^2 flow at "
        "alpha_*(d), B = 1: nodes of the source-free threshold solution and its throat "
        "log ratio b/a in (r-r0)^{(d-2)/2} w = a + b ln((r-r0)/(r_*-r0)), r_* where "
        "e^{2C} = 2v; below threshold (lambda = sqrt(alpha_*/alpha) < 1) node counts and "
        "the normalised source of the IR-normalisable solution; d3_cross_*: where the "
        "d = 3 finite-T onset coupling equals 8/3 (q = alpha B^2/2, T = (3 - q)/(4 pi), "
        "B = B_m, horizon r = 1).",
    )
    log(f"cache written: {CACHE}")


if __name__ == "__main__":
    main()
