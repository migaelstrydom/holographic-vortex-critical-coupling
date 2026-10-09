"""The onset coupling alpha(beta) by shooting from the horizon: a dense scan from
the probe limit to beta ~ 1e66, with no continuation.

METHOD.  The brane is integrated outward from a regular horizon in the RAW frame
(horizon r = 1, U'(1) = 4, V(1) = Z(1) = 0, raw field b), the one-parameter
family being labelled by eps = 6 - alpha b^2 in (0, 6] (eps = 6 is the probe
limit, eps -> 0 the extremal throat).  On the onset curve the raw field is the
eigenvalue, b = B_m = 3 b_h, so eps = 6 - 9 alpha b_h^2 = 6 - beta e^{-4V} with
beta e^{-4V} the frame-invariant horizon value.  `derivations/brane_flows.py` derives the
reduced equations (identical to the production system) and shows that V = O(eps)
at every order of the horizon series, so writing the magnetic term as
4 - (2/3)(6 - eps) e^{-4V} = -4 expm1(-4V) + (2/3) eps e^{-4V} removes the only
cancellation: eps can be taken to 1e-24 and beyond.  The polarised
lowest-Landau-level mode (U e^Z w')' + B_m e^{Z-2V} w = 0 is integrated in the
same pass from horizon regularity w'(1) = -B_m/4, and B_m is the lowest root of
the boundary source c_0 = w + (r/2) w' at r_max >> r_c.  The source extraction is
exact for the r^0 and r^-2 solutions and its error is O(r_c/r_max) RELATIVE to
c_0, so the root is unaffected by the O(1/r) terms of the fixed-T frame (the
r^-3, r^-3 ln r terms that the far-field fits of the collocation codes omit).

FRAME MAP.  With e^{2V} -> v_inf r^2 and U -> r^2 (checked), the fixed-T,
boundary-normalised frame of paper sec. 2.2 has
    B_c u_H^2 = B_m / v_inf,   beta = (6 - eps)/v_inf^2,   alpha = (6 - eps)/B_m^2,
    b_h = B_m/3   (= B_c l3^2 e^{-2V(r_p)}),   nu = sqrt(b_h - 1).

CHECKS
 [1] the collocation ladder (`dk_throat.npz`, a different solver, frame and
     boundary treatment) is reproduced at its rungs to the digits it carries;
 [2] alpha(beta) is strictly increasing over the whole scan.  It is below
     alpha_* = 2/3 identically: every root has nu > 0, so b_h > 1 and
     alpha = (6 - eps)/(9 b_h^2) < 2/3 by construction.  What makes the
     statement "no onset above alpha_*" a result is [0]: each root is the
     lowest one (the zero mode has no node), so no lower-lying onset was missed;
 [3] B_c u_H^2 is a convex function of alpha on the scan;
 [4] the local BKT slope A(beta) = d ln beta / d(1/nu) has its minimum near
     ln beta ~ 21 and then rises monotonically, towards 4 pi.  At the quoted
     marks it is computed directly, by central differences in ln eps about a
     solve at the mark (two step sizes, Richardson), and compared with the
     values quoted in the paper and with the derivative of the scan;
 [5] beyond the ladder the secant slope over [L, L + 1] by direct solves agrees
     with the parameter-free matching condition of `dk_throat_matching.py`
     (A_eff at ln beta = L = 50, 100, the same secant);
 [6] refinement: alpha at eps = 1e-2, 1e-12, 1e-24 is unchanged, to the
     printed movement, under a tighter integrator tolerance, a larger outer
     radius r_max (SPAN_UV) and a different series start point X0.
 [7] (app. D.2) beyond the ladder, the second method: the matching condition of
     dk_throat_matching (exact horizon phase, phi_* evaluated on the T = 0
     brane, the T = 0 offset; no finite-T solve) reproduces alpha and b_h at
     beta = 1e24, 1e45, 1e66 and the local slopes at ln beta = 40, 77.5, 150.
     (M) fixes the throat exponent, 1 + nu^2 = sqrt(alpha_*/alpha); b_h and the
     slope's nu = sqrt(b_h - 1) are horizon values, so the matching is
     converted with the exact relation b_h = (1 + nu^2) sqrt(1 - eps/6) before
     the comparison (unconverted, the slope at 40 differs by 7.6e-5, which is
     that O(eps) frame difference).  In the same frame (M) also reproduces the
     ladder's nu, to 6e-7 at 1e11 (dk_throat_matching [4]).

Run:  uv run python -m backreaction.numerics.onset_scan   (~12 min; --quick ~9 min, writes nothing)
"""

import sys
import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from backreaction import paths
from backreaction.derivations import brane_flows as bf

T0 = time.time()
CHECKS = {}
CACHE = paths.data("onset_scan.npz")
X0 = 1e-7  # start of the integration, r = 1 + X0
SPAN_UV = 32.0  # e-folds of r beyond the throat crossover r_c


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


RTOL = 1e-12  # DOP853 relative tolerance
_SERIES = bf.horizon_series()
_SER, _EPS, (_UC, _VC, _ZC) = _SERIES
_ser = {k: sp.lambdify(_EPS, v, "math") for k, v in _SER.items()}
# the mode start, derived in brane_flows.mode_horizon_series (check [5] there)
_WSER, _BM, _, _ = bf.mode_horizon_series(_SERIES)
_w1, _w2 = (sp.lambdify((_BM, _EPS), _WSER[k], "math") for k in sp.symbols("w1 w2"))


def initial(eps, Bm, x0=X0):
    x = x0
    c = {str(k): f(eps) for k, f in _ser.items()}
    U = 4 * x + c["u2"] * x**2 + c["u3"] * x**3 + c["u4"] * x**4
    Up = 4 + 2 * c["u2"] * x + 3 * c["u3"] * x**2 + 4 * c["u4"] * x**3
    V = c["v1"] * x + c["v2"] * x**2 + c["v3"] * x**3
    Vp = c["v1"] + 2 * c["v2"] * x + 3 * c["v3"] * x**2
    Z = c["z1"] * x + c["z2"] * x**2 + c["z3"] * x**3
    Zp = c["z1"] + 2 * c["z2"] * x + 3 * c["z3"] * x**2
    # mode: (P w')' + Q w = 0 with P = U e^Z, Q = Bm e^{Z-2V}; at x -> 0,
    # w = 1 + w1 x + w2 x^2 with w1 = -Bm/4 and w2 = Bm(3 Bm - 2 eps + 36)/192
    w1, w2 = _w1(Bm, eps), _w2(Bm, eps)
    w = 1 + w1 * x + w2 * x**2
    wp = w1 + 2 * w2 * x
    return [U, V, Z, Up, Vp, Zp, w, U * np.exp(Z) * wp]


def shoot(eps, Bm, t_max, dense=False, x0=X0, rtol=RTOL):
    """Integrate background + mode in t = ln(r - 1).  Returns the solve_ivp result."""
    br = 6.0 - eps

    def rhs(t, y):
        x = np.exp(t)
        U, V, Z, Up, Vp, Zp, w, Pi = y
        e4 = np.exp(-4 * V)
        Upp = (2.0 / 3.0) * br * e4 - 2 * Up * Vp - Up * Zp + 8
        Vpp = (
            (-4 * np.expm1(-4 * V) + (2.0 / 3.0) * eps * e4 - Up * Vp) / U
            - 2 * Vp**2
            - Vp * Zp
        )
        Zpp = ((1.0 / 3.0) * br * e4 - Up * Zp + 4) / U - 2 * Vp * Zp - Zp**2
        eZ = np.exp(Z)
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
        initial(eps, Bm, x0),
        method="DOP853",
        rtol=rtol,
        atol=1e-300,
        dense_output=dense,
        events=node,
    )


def source(sol):
    U, V, Z, Up, Vp, Zp, w, Pi = sol.y[:, -1]
    r = 1 + np.exp(sol.t[-1])
    return w + 0.5 * r * Pi / (U * np.exp(Z))


def t_max_for(eps, span=SPAN_UV):
    """ln r_max: the throat ends where eps r^p ~ 1 (p = sqrt(19/3) - 1), plus span."""
    return np.log(max(1.0, eps ** (-1 / 1.5166))) + span


def onset(eps, nu_guess=None, span=SPAN_UV, x0=X0, rtol=RTOL):
    """Lowest B_m = 3(1 + nu^2) with c_0 = 0.  With a guess (the neighbouring scan
    point) the root is bracketed locally; otherwise by the first sign change on a
    geometric nu grid.  That the root is the ground state is checked afterwards
    by Sturm: its solution has no zero."""
    tm = t_max_for(eps, span)
    f = lambda nu: source(shoot(eps, 3 * (1 + nu**2), tm, x0=x0, rtol=rtol))
    br = None
    if nu_guess is not None:
        lo, hi = 0.97 * nu_guess, 1.005 * nu_guess
        if f(lo) * f(hi) < 0:
            br = (lo, hi)
    if br is None:
        grid = np.geomspace(1e-4, 1.0, 40)
        prev = f(grid[0])
        for i in range(1, len(grid)):
            cur = f(grid[i])
            if cur * prev < 0:
                br = (grid[i - 1], grid[i])
                break
            prev = cur
        else:
            raise RuntimeError(f"no sign change of c0 at eps = {eps:g}")
    nu = brentq(f, *br, xtol=1e-15, rtol=1e-15, maxiter=200)
    Bm = 3 * (1 + nu**2)
    sol = shoot(eps, Bm, tm, dense=True, x0=x0, rtol=rtol)
    # zeros away from the far boundary region (a root residual in c0 can put a
    # spurious zero where c0 ~ c2 r^-2)
    nodes = int(np.sum(sol.t_events[0] < tm - span + 12.0))
    # frame: V - ln r -> (1/2) ln v_inf, U/r^2 -> 1; Richardson in 1/r
    ts = tm - np.array([0.0, np.log(2), np.log(4)])
    rr = 1 + np.exp(ts)
    Vs = sol.sol(ts)[1] - np.log(rr)
    Us = sol.sol(ts)[0] / rr**2
    # q(r) = q_inf + c1/r + c2/r^2 through the three points
    M = np.stack([np.ones(3), 1 / rr, 1 / rr**2], axis=1)
    qv = np.linalg.solve(M, Vs)[0]
    qu = np.linalg.solve(M, Us)[0]
    vinf = np.exp(2 * qv)
    return dict(
        eps=eps,
        nu=nu,
        Bm=Bm,
        vinf=vinf,
        uinf=qu,
        nodes=nodes,
        beta=(6 - eps) / vinf**2,
        alpha=(6 - eps) / Bm**2,
        Bc=Bm / vinf,
        b_h=Bm / 3,
    )


def at_beta(beta_t, eps_grid, beta_grid, nu_grid):
    """Direct solve at a prescribed beta: secant in ln eps from the scan."""
    i = np.argsort(beta_grid)
    le = np.interp(np.log(beta_t), np.log(beta_grid[i]), np.log(eps_grid[i]))
    nu0 = np.interp(np.log(beta_t), np.log(beta_grid[i]), nu_grid[i])
    pts = []
    for le_ in (le, le + 1e-4):
        r = onset(np.exp(le_), nu0)
        pts.append((le_, np.log(r["beta"] / beta_t), r))
    for _ in range(30):
        (l1, f1, _r1), (l2, f2, r2) = pts[-2], pts[-1]
        # ln beta itself carries ~1e-13 of noise (the v_inf extrapolation)
        if abs(f2) < 2e-13 or f2 == f1:
            if abs(f2) > 1e-11:
                break
            return r2
        l3 = l2 - f2 * (l2 - l1) / (f2 - f1)
        r = onset(np.exp(l3), r2["nu"])
        pts.append((l3, np.log(r["beta"] / beta_t), r))
    raise RuntimeError(f"secant did not converge at beta = {beta_t:g}")


def local_slope(row, steps=(0.04, 0.02)):
    """A = d ln beta / d(1/nu) at a solved point, by central differences in
    ln eps with two step sizes and one Richardson step (error O(h^4))."""
    le, est = np.log(row["eps"]), []
    for h in steps:
        rp, rm = onset(np.exp(le + h), row["nu"]), onset(np.exp(le - h), row["nu"])
        est.append(
            (np.log(rp["beta"]) - np.log(rm["beta"])) / (1 / rp["nu"] - 1 / rm["nu"])
        )
    return (4 * est[1] - est[0]) / 3, abs(est[1] - est[0])


#: local slopes quoted in the paper (sec. 4.6, app. D.3), at their printed digits
QUOTED_SLOPES = {40.0: 7.89, 77.5: 8.93, 150.0: 10.3}
QUOTED_SLOPE_MIN = 7.43  # the minimum, near ln beta = 21
#: alpha quoted in app. D.2 at exactly these beta
QUOTED_ALPHA = {1e24: 0.6453, 1e45: 0.6591, 1e66: 0.6626}
#: where [7] compares the local slope of the matching condition with A_local
SLOPE_MARKS_MATCH = (40.0, 77.5, 150.0)
#: refinement test: eps values and the allowed movement of alpha
REFINE_EPS = (1e-2, 1e-12, 1e-24)
REFINE_TOL = 1e-13
REFINE_TOL_BETA = 1e-11


def main():
    quick = "--quick" in sys.argv
    n = 80 if quick else 160
    eps_grid = np.concatenate(
        [6 - 6 * np.geomspace(1e-3, 0.95, 25), np.geomspace(5.6, 1e-24, n)]
    )
    eps_grid = np.unique(eps_grid)[::-1]
    rows, guess = [], None
    for eps in eps_grid:
        row = onset(eps, guess)
        rows.append(row)
        guess = row["nu"]
        if len(rows) % 50 == 0:
            log(
                f"      eps {eps:.3e}  beta {row['beta']:.4e}  alpha {row['alpha']:.10f}  "
                f"B_c u_H^2 {row['Bc']:.10g}"
            )
    key = lambda k: np.array([r[k] for r in rows], dtype=float)
    eps, beta, alpha, Bc, b_h, nu = (
        key(k) for k in ("eps", "beta", "alpha", "Bc", "b_h", "nu")
    )
    nodes, uinf = key("nodes"), key("uinf")
    check(
        "[0] every root is the ground state (no zero) and U/r^2 -> 1",
        np.all(nodes == 0) and np.max(np.abs(uinf - 1)) < 1e-9,
        f"max |U/r^2 - 1| = {np.max(np.abs(uinf - 1)):.1e}",
    )
    log(
        f"      scan: {len(rows)} points, beta {beta.min():.3e} ... {beta.max():.3e}, "
        f"alpha {alpha.min():.3e} ... {alpha.max():.8f}"
    )

    # [1] the collocation ladder, by direct solves at its betas
    lad = np.load(paths.data("dk_throat.npz"))
    dev = dict(alpha=[], B_c=[], b_h=[])
    log(
        "      beta         alpha (scan)    ladder        B_c u_H^2 (scan)       ladder"
    )
    for bL, aL, BL, hL in zip(
        lad["beta"], lad["alpha"], lad["B_c"], lad["b_h"], strict=True
    ):
        r = at_beta(bL, eps, beta, nu)
        dev["alpha"].append(abs(r["alpha"] / aL - 1))
        dev["B_c"].append(abs(r["Bc"] / BL - 1))
        dev["b_h"].append(abs(r["b_h"] / hL - 1))
        log(
            f"      {bL:10.4g}   {r['alpha']:.10f}  {aL:.10f}  {r['Bc']:20.10f}  {BL:20.10f}"
        )
    worst = {k: max(v) for k, v in dev.items()}
    check(
        "[1] the collocation ladder is reproduced at every rung: alpha and B_c "
        "to 2e-10, b_h to 3e-9",
        worst["alpha"] < 2e-10 and worst["B_c"] < 1e-10 and worst["b_h"] < 3e-9,
        ", ".join(f"{k} {v:.1e}" for k, v in worst.items()),
    )

    # [2] monotone, bounded by alpha_*
    o = np.argsort(beta)
    check(
        "[2] alpha(beta) strictly increasing on the whole scan",
        np.all(np.diff(alpha[o]) > 0),
        f"min increment {np.min(np.diff(alpha[o])):.2e}, alpha_max = {alpha.max():.8f} "
        f"at beta = {beta.max():.2e} (< 2/3 by construction, nu > 0; [0] is the guard)",
    )
    alpha_q, bh_q, eps_q = {}, {}, {}
    for bq, aq in QUOTED_ALPHA.items():
        rq = at_beta(bq, eps, beta, nu)
        alpha_q[bq], bh_q[bq], eps_q[bq] = rq["alpha"], rq["b_h"], rq["eps"]
        log(f"      beta = {bq:.0e}: alpha = {alpha_q[bq]:.10f}  (quoted {aq})")
    check(
        "[2] alpha at beta = 1e24, 1e45, 1e66 rounds to the quoted 0.6453, 0.6591, 0.6626",
        all(round(alpha_q[b], 4) == a for b, a in QUOTED_ALPHA.items()),
    )
    # [3] convexity of B_c(alpha)
    oa = np.argsort(alpha)
    slope = np.diff(Bc[oa]) / np.diff(alpha[oa])
    check(
        "[3] B_c u_H^2 convex in alpha (chord slopes increasing)",
        np.all(np.diff(slope) > 0),
    )

    # [4] the local BKT slope A = d ln beta / d(1/nu)
    lb, inv = np.log(beta[o]), 1 / nu[o]
    A_loc = np.gradient(lb, inv)
    sel = lb > 10
    imin = np.argmin(A_loc[sel])
    lb_min, A_min = lb[sel][imin], A_loc[sel][imin]
    after = A_loc[sel][imin:]
    check(
        "[4] A (derivative of the scan) has a single minimum near ln beta ~ 21 and "
        "rises monotonically after it, staying below 4 pi",
        18 < lb_min < 25 and np.all(np.diff(after) > 0) and after.max() < 4 * np.pi,
        f"min A = {A_min:.4f} at ln beta = {lb_min:.2f}; A = {after[-1]:.3f} at "
        f"ln beta = {lb[-1]:.1f}",
    )
    marks = [
        16.118,
        18.421,
        20.5,
        21.5,
        22.5,
        25.328,
        26,
        40,
        55,
        70,
        77.5,
        100,
        130,
        150,
    ]
    A_at, A_err, A_grid = [], [], []
    for m in marks:
        a_, e_ = local_slope(at_beta(np.exp(m), eps, beta, nu))
        A_at.append(a_)
        A_err.append(e_)
        A_grid.append(float(np.interp(m, lb, A_loc)))
    A_at = np.array(A_at)
    log("      ln beta   A (direct)   step change   A (scan derivative)")
    for m, a_, e_, g_ in zip(marks, A_at, A_err, A_grid, strict=True):
        log(f"      {m:7.3f}   {a_:.6f}    {e_:.1e}       {g_:.4f}")
    A_d = dict(zip(marks, A_at, strict=True))
    win = [m for m in marks if 16 < m < 26]  # the ladder's last four decades
    win3 = [m for m in win if m > 18]  # its last three
    spread4 = (max(A_d[m] for m in win) - min(A_d[m] for m in win)) / min(A_d.values())
    spread3 = (max(A_d[m] for m in win3) - min(A_d[m] for m in win3)) / min(
        A_d.values()
    )
    log(
        f"      spread of A over beta in [1e7, 1e11]: {spread4:.2%}; over [1e8, 1e11]: "
        f"{spread3:.2%}"
    )
    check(
        "[4] direct local slopes round to the quoted 7.89, 8.93, 10.3 and the minimum "
        "to 7.43; step changes < 1e-4",
        all(round(A_d[m], 2 if q < 10 else 1) == q for m, q in QUOTED_SLOPES.items())
        and round(min(A_d.values()), 2) == QUOTED_SLOPE_MIN
        and max(A_err) < 1e-4,
        f"min {min(A_d.values()):.4f}",
    )
    check(
        "[4] the derivative of the scan agrees with the direct slopes to 3e-3",
        np.max(np.abs(np.array(A_grid) / A_at - 1)) < 3e-3,
        f"worst {np.max(np.abs(np.array(A_grid) / A_at - 1)):.1e}",
    )

    mt = np.load(paths.data("dk_throat_matching.npz"))
    devs = []
    for L in (50, 100):
        j = int(np.where(mt["A_eff_lnbeta"] == L)[0][0])
        # A_eff there is the secant over [L, L + 1]; the same secant by direct solves
        n1 = at_beta(np.exp(L), eps, beta, nu)["nu"]
        n2 = at_beta(np.exp(L + 1.0), eps, beta, nu)["nu"]
        a_dir = 1.0 / (1 / n2 - 1 / n1)
        devs.append(abs(a_dir / mt["A_eff"][j] - 1))
        log(
            f"      ln beta = {L}: direct secant A = {a_dir:.5f}, matching A_eff = "
            f"{mt['A_eff'][j]:.5f}"
        )
    check(
        "[5] direct secant slope agrees with the matching condition to 1e-4",
        max(devs) < 1e-4,
        f"worst relative deviation {max(devs):.1e}",
    )

    # [6] refinement: tolerance, outer radius, start point
    log(
        "      eps        base alpha        rtol 1e-13   SPAN_UV 40   SPAN_UV 24   X0 1e-8     X0 1e-6"
    )
    moves, bmoves = [], []
    for e_ in REFINE_EPS:
        base = onset(e_)
        g = base["nu"]
        var = [
            onset(e_, g, rtol=1e-13),
            onset(e_, g, span=40.0),
            onset(e_, g, span=24.0),
            onset(e_, g, x0=1e-8),
            onset(e_, g, x0=1e-6),
        ]
        mv = [abs(v["alpha"] - base["alpha"]) for v in var]
        moves.append(max(mv))
        bmoves.append(max(abs(v["beta"] / base["beta"] - 1) for v in var))
        log(
            f"      {e_:.0e}   {base['alpha']:.13f}  "
            + "  ".join(f"{m:.1e}" for m in mv)
        )
    check(
        f"[6] alpha moves by < {REFINE_TOL:g} and beta by < {REFINE_TOL_BETA:g} "
        "(relative) under tolerance, r_max and start-point refinement at "
        "eps = 1e-2, 1e-12, 1e-24",
        max(moves) < REFINE_TOL and max(bmoves) < REFINE_TOL_BETA,
        f"largest movement: alpha {max(moves):.1e}, beta {max(bmoves):.1e}",
    )

    # [7] the onset beyond the ladder by the matching condition (app. D.2)
    from backreaction.numerics import dk_throat_matching as TM

    log("[7] beyond the ladder: the matching condition against horizon shooting")
    sol_t0 = TM.t0_brane()[0]
    rs_t0 = TM.brane_rho_star(sol_t0)
    off_t0 = float(mt["offset_t0"])
    # (M) fixes the throat exponent nu, i.e. alpha = (2/3)/(1 + nu^2)^2.  The
    # horizon value b_h = B_m/3 = sqrt((6 - eps)/(9 alpha)) sits eps/12 below
    # 1 + nu^2 (eps = 6 - beta e^{-4V} at the horizon: 1.1e-8 at beta = 1e24),
    # so b_h is compared through that exact frame relation, with the shooting's eps.
    a_match, bh_match, bh_frame, dA = {}, {}, {}, []
    for bq in QUOTED_ALPHA:
        n_m = TM.nu_match(np.log(bq), sol_t0, rs_t0, off_t0, bracket=(0.02, 0.3))
        bh_match[bq] = 1 + n_m**2
        a_match[bq] = (2.0 / 3.0) / bh_match[bq] ** 2
        bh_frame[bq] = bh_match[bq] * np.sqrt(1 - eps_q[bq] / 6)
        log(
            f"      beta = {bq:.0e}: alpha {a_match[bq]:.11f} (matching), "
            f"{alpha_q[bq]:.11f} (shooting), diff {a_match[bq] - alpha_q[bq]:+.1e}; "
            f"1 + nu^2 = {bh_match[bq]:.11f}, b_h {bh_frame[bq]:.11f} vs {bh_q[bq]:.11f} "
            f"(eps = {eps_q[bq]:.2e})"
        )
    # The shooting slope A = d ln beta / d(1/nu) uses the horizon nu = sqrt(b_h - 1),
    # so the matching nu (the throat exponent) is converted the same way, with
    # the eps of a direct solve at each beta; unconverted, the comparison
    # measures the O(eps) frame difference (7.6e-5 at ln beta = 40).
    A_match, A_match_raw, dA_raw = [], [], []
    for m in SLOPE_MARKS_MATCH:
        est, est_raw = [], []
        for h in (0.4, 0.2):
            inv = []
            for lb_ in (m - h, m + h):
                n_t = TM.nu_match(lb_, sol_t0, rs_t0, off_t0, bracket=(0.02, 0.5))
                e_ = at_beta(np.exp(lb_), eps, beta, nu)["eps"]
                n_h = np.sqrt((1 + n_t**2) * np.sqrt(1 - e_ / 6) - 1)
                inv.append((1 / n_t, 1 / n_h))
            est_raw.append(2 * h / (inv[1][0] - inv[0][0]))
            est.append(2 * h / (inv[1][1] - inv[0][1]))
        A_m = (4 * est[1] - est[0]) / 3
        A_raw = (4 * est_raw[1] - est_raw[0]) / 3
        A_match.append(A_m)
        A_match_raw.append(A_raw)
        dA.append(abs(A_m / A_d[m] - 1))
        dA_raw.append(abs(A_raw / A_d[m] - 1))
        log(
            f"      ln beta = {m:5.1f}: local slope {A_m:.9f} (matching, horizon frame), "
            f"{A_d[m]:.9f} (shooting), rel. diff {dA[-1]:.1e}; throat-frame "
            f"matching {A_raw:.9f} ({dA_raw[-1]:.1e})"
        )
    d_alpha = max(abs(a_match[b] - alpha_q[b]) for b in QUOTED_ALPHA)
    d_bh = max(abs(bh_frame[b] - bh_q[b]) for b in QUOTED_ALPHA)
    check(
        "[7] the matching condition reproduces the horizon-shooting alpha and b_h at "
        "beta = 1e24, 1e45, 1e66 to 1e-10, and the local slopes, compared in one "
        "frame, at ln beta = 40 to 3e-8 and at 77.5, 150 to 2e-9",
        d_alpha < 1e-10 and d_bh < 1e-10 and dA[0] < 3e-8 and max(dA[1:]) < 2e-9,
        f"max |d alpha| = {d_alpha:.1e}, max |d b_h| = {d_bh:.1e}; relative slope "
        f"differences " + ", ".join(f"{x:.1e}" for x in dA),
    )

    failed = [k for k, v in CHECKS.items() if not v]
    log(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if failed:
        log("cache not written")
        sys.exit(1)
    if quick:
        log("--quick: cache not written")
        return
    np.savez(
        CACHE,
        eps=eps,
        beta=beta,
        alpha=alpha,
        B_c=Bc,
        b_h=b_h,
        nu=nu,
        lnbeta_marks=np.array(marks),
        A_local=A_at,
        alpha_quoted_beta=np.array(list(alpha_q)),
        alpha_quoted=np.array(list(alpha_q.values())),
        refine_eps=np.array(REFINE_EPS),
        refine_max_move=np.array(moves),
        b_h_quoted=np.array(list(bh_q.values())),
        alpha_match=np.array([a_match[b] for b in QUOTED_ALPHA]),
        b_h_match=np.array([bh_frame[b] for b in QUOTED_ALPHA]),
        nu2_match=np.array([bh_match[b] - 1 for b in QUOTED_ALPHA]),
        eps_quoted=np.array(list(eps_q.values())),
        lnbeta_slope_match=np.array(SLOPE_MARKS_MATCH),
        A_local_match=np.array(A_match),
        A_local_match_throat=np.array(A_match_raw),
        readme="Onset coupling by horizon shooting (raw frame, eps = 6 - alpha b^2): "
        "beta = (6-eps)/v_inf^2, alpha = (6-eps)/B_m^2, B_c u_H^2 = B_m/v_inf, "
        "b_h = B_m/3, nu = sqrt(b_h - 1); local slope A = d ln beta / d(1/nu) at "
        "lnbeta_marks by central differences in ln eps; alpha_quoted at exactly "
        "alpha_quoted_beta; refine_max_move = largest change of alpha under "
        "tolerance, r_max and start-point refinement at refine_eps.  alpha_match, "
        "b_h_match: the matching condition of dk_throat_matching (phi_* evaluated "
        "on the T = 0 brane, the T = 0 offset) at alpha_quoted_beta: "
        "nu2_match the throat exponent squared, alpha_match = (2/3)/(1 + nu2)^2, "
        "b_h_match = (1 + nu2) sqrt(1 - eps/6) with eps_quoted the shooting's eps; "
        "b_h_quoted the shooting b_h there.  A_local_match: the local slope of the "
        "matching condition at lnbeta_slope_match, with nu converted to sqrt(b_h - 1) "
        "by the same relation (eps from a direct solve); A_local_match_throat the same "
        "with the throat exponent unconverted.",
    )
    log(f"cache written: {CACHE}")


if __name__ == "__main__":
    main()
