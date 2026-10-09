"""The finite-temperature onset problem of the magnetic brane in general d: the
raw-frame horizon data, the cancellation-free form of the equations near
extremality, the polarised lowest-Landau-level zero mode with its horizon and
boundary conditions, and the map from the raw frame to the fixed-T,
boundary-normalised quantities.  General-d counterpart of the d = 4 input to
`numerics/onset_scan.py` (`brane_flows.horizon_series`).

Chart (L = 1, d boundary dimensions, z_1..z_{d-3} the directions transverse to
the field; no z directions for d = 3):

    ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2 + dy^2) + e^{2Z} dz_i dz_i,   F_xy = b,

field equations R_MN + d g_MN = alpha (F_MP F_N^P - F^2 g_MN/(2(d-1))).
The background depends on alpha and b only through beta_raw = alpha b^2.

Results
-------
[1] Closed form for real d, built from the Christoffel symbols at d = 3..7 and
    equal to `brane_flows.r_chart(d)` at d = 4, 5, 6:
        U'' = 2 beta_raw e^{-4V}/(d-1) - 2U'V' - (d-3)U'Z' + 2d,
      U V'' = d - (d-2) beta_raw e^{-4V}/(d-1) - U'V' - 2U V'^2 - (d-3)U V'Z',
      U Z'' = d + beta_raw e^{-4V}/(d-1) - U'Z' - 2U V'Z' - (d-3)U Z'^2,
    constraint beta_raw e^{-4V} + 2U'V' + (d-3)U'Z' + 2UV'^2 + 4(d-3)UV'Z'
               + (d-3)(d-4)UZ'^2 = d(d-1).
[2] Near extremality.  V = 0 solves the V equation iff beta_raw = beta_ext(d)
    = d(d-1)/(d-2).  With eps = beta_ext - beta_raw the magnetic term is
        d - (d-2)(beta_ext - eps) e^{-4V}/(d-1)
          = -d expm1(-4V) + (d-2) eps e^{-4V}/(d-1)
    (no cancellation as eps -> 0).  U, U', Z, Z' enter the V equation only
    multiplied by V', so an absolute error in beta_raw in the U and Z equations
    (the rounding of beta_ext - eps) changes V only in proportion to V itself.
    At eps = 0 the exact solution is V = 0 times the AdS_{d-1} black brane with
    l^2 = (d-2)^2/(d(d-1)), rescaled to U'(1) = d.
[3] Raw-frame horizon data: horizon r = 1, U'(1) = d, V(1) = Z(1) = 0.  The
    series in x = r - 1 is solvable order by order for symbolic d; every V
    coefficient is proportional to eps (V = O(eps)); V'(1) = (d-2) eps/(d(d-1));
    the constraint holds order by order; at d = 4 the coefficients equal
    `brane_flows.horizon_series()` exactly.
[4] The zero mode.  The linearised SU(2) Yang-Mills equations about
    A^3_y = b x, with W = A^1 + i A^2, W_x = w(r) e^{-b x^2/2}, reduce for
    d = 3..6 to
        (P w')' + Q w = 0,   P = U e^{(d-3)Z},   Q = b e^{(d-3)Z - 2V},
    exactly when W_y = -i W_x (all other components vanish identically).
    Horizon regularity: w = 1 + w1 x + w2 x^2 with w1 = -b/d and w2 given in
    closed form in the horizon coefficients.
[5] Boundary.  With U -> r^2 (forced by the field equations), e^{2V} -> v r^2,
    the falloffs are w ~ r^0 and w ~ r^{-(d-2)}; the source
        c0 = w + r w'/(d-2)
    annihilates r^{-(d-2)} and keeps r^0 exactly; on the r^0 branch its residual
    is (b/v) r^{-2}/(2(d-2)) relative, for every d (d = 4 included, where the
    branch carries r^{-2} ln r).
[6] Frame map.  The system and the mode equation are invariant under
    (i) r -> r + c, (ii) U -> lam^2 U(1 + (r-1)/lam) with V, Z composed the same
    way, (iii) V -> V + c, b -> b e^{2c}, (iv) Z -> Z + c.  Hence, with
    e^{2V} -> v_inf r^2 in the raw frame and T = U'(1)/(4 pi) = d/(4 pi), i.e.
    u_H = d/(4 pi T) = 1 there, the boundary-normalised quantities are
        B_c u_H^2 = B_m/v_inf,  beta = alpha (B u_H^2)^2 = beta_raw/v_inf^2,
        alpha = beta_raw/B_m^2,  T/sqrt(B) = (d/(4 pi)) sqrt(v_inf/B_m).
    The throat's BF threshold is B_thr = d(d-1)/4, so
        alpha/alpha_* = (1 - eps/beta_ext)/(B_m/B_thr)^2,
    alpha_* = beta_ext/B_thr^2 = 16/(d(d-1)(d-2)).
[7] d = 3 anchor: the non-extremal magnetic RN-AdS4 brane in the raw frame,
    U = lam^2 U_RN(1 + (r-1)/lam), V = ln(1 + (r-1)/lam), lam = 3/(3-q),
    beta_raw = 2q, solves [1]; v_inf = 1/lam^2.

Run:  uv run python -m backreaction.derivations.onset_scan_d   (~2 min)
"""

import sys
import time

import sympy as sp

from backreaction.derivations import brane_flows as bf

T0 = time.time()
CHECKS = {}

dsym = sp.Symbol("d", positive=True)
r = sp.Symbol("r", positive=True)
x, eps = sp.symbols("x epsilon")
b = sp.Symbol("b", positive=True)
Uv, Vv, Zv, Up, Vp, Zp = sp.symbols("Uv Vv Zv Up Vp Zp")
beta_raw = sp.Symbol("beta_raw", positive=True)


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def is_zero(e):
    e = sp.expand(e)
    return e == 0 or sp.simplify(e) == 0


def beta_ext(d):
    return d * (d - 1) / (d - 2)


def b_thr(d):
    return d * (d - 1) / sp.Integer(4)


def closed_form(d=dsym, br=beta_raw):
    """[1]: (U'', V'', Z'', constraint) in the symbols (Uv, Vv, Zv, Up, Vp, Zp)."""
    e4 = sp.exp(-4 * Vv)
    Upp = 2 * br * e4 / (d - 1) - 2 * Up * Vp - (d - 3) * Up * Zp + 2 * d
    Vpp = (
        (d - (d - 2) * br * e4 / (d - 1) - Up * Vp) / Uv - 2 * Vp**2 - (d - 3) * Vp * Zp
    )
    Zpp = (d + br * e4 / (d - 1) - Up * Zp) / Uv - 2 * Vp * Zp - (d - 3) * Zp**2
    con = (
        br * e4
        + 2 * Up * Vp
        + (d - 3) * Up * Zp
        + 2 * Uv * Vp**2
        + 4 * (d - 3) * Uv * Vp * Zp
        + (d - 3) * (d - 4) * Uv * Zp**2
        - d * (d - 1)
    )
    return Upp, Vpp, Zpp, con


def closed_form_eps(d=dsym):
    """[2]: the same with beta_raw = beta_ext - eps and the V source written
    without cancellation; returns (U'', V'', Z'') with an `expm1` node."""
    from sympy.codegen.cfunctions import expm1

    br = beta_ext(d) - eps
    e4 = sp.exp(-4 * Vv)
    Upp, _, Zpp, _ = closed_form(d, br)
    Vpp = (
        (-d * expm1(-4 * Vv) + (d - 2) * eps * e4 / (d - 1) - Up * Vp) / Uv
        - 2 * Vp**2
        - (d - 3) * Vp * Zp
    )
    return Upp, Vpp, Zpp


def christoffel_chart(d):
    """The r-chart system built from the Ricci tensor, d = 3 included."""
    nz = d - 3
    xs = [sp.Symbol(n) for n in ["t", "x", "y"] + [f"z{i}" for i in range(nz)]] + [r]
    U, V, Z = (sp.Function(n)(r) for n in "UVZ")
    gd = [-U, sp.exp(2 * V), sp.exp(2 * V)] + [sp.exp(2 * Z)] * nz + [1 / U]
    eqs = bf.field_equations(d, gd, xs, 1, 2, len(xs) - 1, bf.B, bf.alpha)
    Upp_, Vpp_, Zpp_ = sp.symbols("Upp_ Vpp_ Zpp_")
    rep = {
        U.diff(r, 2): Upp_,
        V.diff(r, 2): Vpp_,
        Z.diff(r, 2): Zpp_,
        U.diff(r): Up,
        V.diff(r): Vp,
        Z.diff(r): Zp,
        U: Uv,
        V: Vv,
        Z: Zv,
    }
    tt, xx, rr = (sp.expand(eqs[i].subs(rep)) for i in (0, 1, -1))
    if nz:
        zz = sp.expand(eqs[3].subs(rep))
        sol = sp.solve([tt, xx, zz], [Upp_, Vpp_, Zpp_], dict=True)[0]
    else:
        sol = sp.solve([tt, xx], [Upp_, Vpp_], dict=True)[0]
        sol[Zpp_] = None
    con = sp.simplify(rr.subs({k: v for k, v in sol.items() if v is not None}))
    return sol[Upp_], sol[Vpp_], sol[Zpp_], con


def horizon_series(d=dsym, order=4):
    """[3]: coefficients of U = d x + u2 x^2 + ..., V = v1 x + ..., Z = z1 x + ...
    in x = r - 1, for beta_raw = beta_ext(d) - eps.  Returns (dict, (uc, vc, zc))."""
    uc = sp.symbols(f"u2:{order + 1}")
    vc = sp.symbols(f"v1:{order + 1}")
    zc = sp.symbols(f"z1:{order + 1}")
    Us = d * x + sum(uc[i] * x ** (i + 2) for i in range(order - 1))
    Vs = sum(vc[i] * x ** (i + 1) for i in range(order))
    Zs = sum(zc[i] * x ** (i + 1) for i in range(order))
    Upp, Vpp, Zpp, con = closed_form(d, beta_ext(d) - eps)
    sub = {Uv: Us, Vv: Vs, Zv: Zs, Up: Us.diff(x), Vp: Vs.diff(x), Zp: Zs.diff(x)}
    res = [
        Us.diff(x, 2) - Upp.subs(sub),
        sp.expand(Us * (Vs.diff(x, 2) - Vpp.subs(sub))),
        sp.expand(Us * (Zs.diff(x, 2) - Zpp.subs(sub))),
    ]
    res = [sp.expand(sp.series(e, x, 0, order).removeO()) for e in res]
    sol = {}
    for j in range(order - 1):
        ej = [sp.simplify(e.coeff(x, j).subs(sol)) for e in res]
        sj = sp.solve(ej, [uc[j], vc[j], zc[j]], dict=True)
        assert len(sj) == 1, (d, j)
        sol.update({k: sp.factor(v) for k, v in sj[0].items()})
    sol = {k: sp.factor(v.subs(sol)) for k, v in sol.items()}
    con_s = sp.expand(sp.series(con.subs(sub).subs(sol), x, 0, order - 1).removeO())
    return sol, (uc, vc, zc), con_s


def mode_horizon(d, ser, uc, vc, zc):
    """[4]: regular zero-mode data w = 1 + w1 x + w2 x^2 at the horizon."""
    Bm = sp.Symbol("B_m", positive=True)
    w1 = -Bm / d
    p2 = ser[uc[0]] + d * (d - 3) * ser[zc[0]]
    q1 = (d - 3) * ser[zc[0]] - 2 * ser[vc[0]]
    w2 = -(2 * p2 * w1 + Bm * (w1 + q1)) / (4 * d)
    return Bm, w1, w2, p2, q1


def ym_mode(d, sgn):
    """[4]: linearised SU(2) YM about A^3_y = b x on the r chart; ansatz
    A^1_x = w g, A^2_y = sgn w g, g = e^{-b x^2/2} (i.e. W_y = i sgn W_x).
    Returns the list of nonzero linearised equations (a, nu, expression)."""
    nz = d - 3
    X = [sp.Symbol(n) for n in ["t", "xx", "yy"] + [f"z{i}" for i in range(nz)]] + [r]
    xc = X[1]
    n = len(X)
    U, V, Z, w = (sp.Function(s)(r) for s in ("U", "V", "Z", "w"))
    gd = [-U, sp.exp(2 * V), sp.exp(2 * V)] + [sp.exp(2 * Z)] * nz + [1 / U]
    gi = [1 / g for g in gd]
    sqg = sp.exp(2 * V + nz * Z)
    eta = sp.Symbol("eta")
    g = sp.exp(-b * xc**2 / 2)
    A = [[0] * n for _ in range(3)]
    A[2][2] = b * xc
    A[0][1] = eta * w * g
    A[1][2] = eta * sgn * w * g
    eps3 = lambda a, bb, c: sp.LeviCivita(a, bb, c)
    F = [
        [
            [
                sp.diff(A[a][nu], X[mu])
                - sp.diff(A[a][mu], X[nu])
                + sum(
                    eps3(a, bb, c) * A[bb][mu] * A[c][nu]
                    for bb in range(3)
                    for c in range(3)
                )
                for nu in range(n)
            ]
            for mu in range(n)
        ]
        for a in range(3)
    ]
    Fup = [
        [[gi[mu] * gi[nu] * F[a][mu][nu] for nu in range(n)] for mu in range(n)]
        for a in range(3)
    ]
    out = []
    for a in range(3):
        for nu in range(n):
            e = sum(sp.diff(sqg * Fup[a][mu][nu], X[mu]) for mu in range(n))
            e += sqg * sum(
                eps3(a, bb, c) * A[bb][mu] * Fup[c][mu][nu]
                for bb in range(3)
                for c in range(3)
                for mu in range(n)
            )
            e = sp.simplify(sp.diff(e, eta).subs(eta, 0))
            if e != 0:
                out.append((a, nu, e))
    P = U * sp.exp(nz * Z)
    Q = b * sp.exp(nz * Z - 2 * V)
    target = sp.diff(P * w.diff(r), r) + Q * w
    return out, target, g, (U, V, Z, w)


def _residuals(d, Ue, Ve, Ze, br, Bmode):
    """Field-equation residuals and the mode operator on (Ue, Ve, Ze)."""
    sub = {Uv: Ue, Vv: Ve, Zv: Ze, Up: Ue.diff(r), Vp: Ve.diff(r), Zp: Ze.diff(r)}
    Upp, Vpp, Zpp, con = closed_form(sp.Integer(d), br)
    wf = sp.Function("wf")(r)
    mode = (
        sp.diff(Ue * sp.exp((d - 3) * Ze) * wf.diff(r), r)
        + Bmode * sp.exp((d - 3) * Ze - 2 * Ve) * wf
    )
    res = [
        Ue.diff(r, 2) - Upp.subs(sub),
        Ve.diff(r, 2) - Vpp.subs(sub),
        (Ze.diff(r, 2) - Zpp.subs(sub)) if d > 3 else sp.Integer(0),
        con.subs(sub),
    ]
    return res, mode


def symmetries_hold(d):
    """[6]: scaling U -> lam^2 U(rho), rho = 1 + (r-1)/lam, maps residuals to
    (1, lam^-2, lam^-2, 1) x residuals at rho; V, Z -> V + c, Z + c with
    beta_raw -> beta_raw e^{4c}, b -> b e^{2c} leaves them unchanged; the mode
    operator is invariant under both (up to the factor e^{(d-3)c})."""
    lam, c = sp.symbols("lambda c", positive=True)
    Uf, Vf, Zf, Wf = (sp.Function(n) for n in ("Uf", "Vf", "Zf", "Wf"))
    R = sp.Symbol("R")
    rho = 1 + (r - 1) / lam
    base, mode0 = _residuals(d, Uf(r), Vf(r), Zf(r), beta_raw, b)
    scal, _ = _residuals(d, lam**2 * Uf(rho), Vf(rho), Zf(rho), beta_raw, b)
    shft, mode2 = _residuals(
        d, Uf(r), Vf(r) + c, Zf(r) + c, beta_raw * sp.exp(4 * c), b * sp.exp(2 * c)
    )
    ok_scal = all(
        is_zero(
            sp.simplify(sc.subs(r, R).doit() - k * bs.subs(r, 1 + (R - 1) / lam).doit())
        )
        for sc, bs, k in zip(scal, base, (1, lam**-2, lam**-2, 1), strict=True)
    )
    ok_shft = all(
        is_zero(sp.simplify(a1 - a0)) for a1, a0 in zip(shft, base, strict=True)
    )

    def P(y):
        return Uf(y) * sp.exp((d - 3) * Zf(y))

    def Q(y):
        return b * sp.exp((d - 3) * Zf(y) - 2 * Vf(y))

    m_scaled = sp.diff(lam**2 * P(rho) * sp.diff(Wf(rho), r), r) + Q(rho) * Wf(rho)
    m_base = (sp.diff(P(R) * sp.diff(Wf(R), R), R) + Q(R) * Wf(R)).subs(R, rho)
    ok_mode = is_zero(sp.simplify(mode2 * sp.exp(-(d - 3) * c) - mode0)) and is_zero(
        sp.simplify((m_scaled - m_base).doit())
    )
    return ok_scal and ok_shft and ok_mode


def main():
    # [1] closed form vs the Christoffel build and brane_flows.r_chart
    log("[1] the r-chart system: closed form in d vs the Ricci build")
    for d in (3, 4, 5, 6, 7):
        Upp, Vpp, Zpp, con = christoffel_chart(d)
        m = {bf.alpha: beta_raw, bf.B: 1}
        cU, cV, cZ, cC = closed_form(sp.Integer(d))
        ok = is_zero(Upp.subs(m) - cU) and is_zero(Vpp.subs(m) - cV)
        if d > 3:
            ok = ok and is_zero(Zpp.subs(m) - cZ)
        ratio = sp.simplify(con.subs(m) * Uv / cC)
        zlab = ", Z''" if d > 3 else ""
        check(
            f"[1] d={d}: U'', V''{zlab} equal the closed form; the rr constraint is it divided by U",
            ok and ratio.is_number and ratio != 0,
            f"U x constraint / closed form = {ratio}",
        )
    for d in (4, 5, 6):
        rc = bf.r_chart(d)
        U2, V2, Z2, Up2, Vp2, Zp2 = rc["syms"]
        m = {
            U2: Uv,
            V2: Vv,
            Z2: Zv,
            Up2: Up,
            Vp2: Vp,
            Zp2: Zp,
            bf.alpha: beta_raw,
            bf.B: 1,
        }
        cU, cV, cZ, _ = closed_form(sp.Integer(d))
        check(
            f"[1] d={d}: equals brane_flows.r_chart",
            all(
                is_zero(rc[k].xreplace(m) - c)
                for k, c in (("Upp", cU), ("Vpp", cV), ("Zpp", cZ))
            ),
        )

    # [2] near extremality
    log("[2] extremality, cancellation-free form, the eps = 0 throat")
    _, Vpp, _, _ = closed_form()
    UVpp = sp.expand(Vpp * Uv)
    be = sp.solve(sp.Eq(UVpp.subs({Vv: 0, Vp: 0}), 0), beta_raw)
    check(
        "[2] V = 0 solves the V equation iff beta_raw = d(d-1)/(d-2)",
        len(be) == 1 and is_zero(be[0] - beta_ext(dsym)),
        f"beta_ext = {sp.factor(be[0])}",
    )
    _, Vpe, _ = closed_form_eps()
    from sympy.codegen.cfunctions import expm1

    diff = (Vpe - Vpp.subs(beta_raw, beta_ext(dsym) - eps)).replace(
        expm1, lambda a: sp.exp(a) - 1
    )
    check("[2] cancellation-free V source equals the original", is_zero(diff))
    dep = [sp.diff(UVpp, s).subs(Vp, 0) for s in (Uv, Up, Zv, Zp)]
    check(
        "[2] U, U', Z, Z' enter U V'' only multiplied by V'",
        all(is_zero(e) for e in dep),
    )
    for d in (3, 4, 5, 6):
        l2 = sp.Rational((d - 2) ** 2, d * (d - 1))
        lam = sp.Rational(d - 2, d - 1)
        rho = 1 + (r - 1) / lam
        Ut = lam**2 * rho**2 * (1 - rho ** (-(d - 2))) / l2
        Zt = sp.log(rho)
        Upp, Vpp_d, Zpp, con = closed_form(sp.Integer(d), beta_ext(sp.Integer(d)))
        sub = {Uv: Ut, Vv: 0, Zv: Zt, Up: Ut.diff(r), Vp: 0, Zp: Zt.diff(r)}
        ok = (
            is_zero(Ut.diff(r, 2) - Upp.subs(sub))
            and is_zero(Vpp_d.subs(sub))
            and is_zero(con.subs(sub))
            and is_zero(Ut.diff(r).subs(r, 1) - d)
        )
        if d > 3:
            ok = ok and is_zero(Zt.diff(r, 2) - Zpp.subs(sub))
        check(
            f"[2] d={d}: eps = 0 solution is V = 0 x the AdS_{d - 1} black brane, U'(1) = d",
            ok,
            f"U = {sp.factor(sp.simplify(Ut))}",
        )

    # [3] horizon series for symbolic d
    log("[3] raw-frame horizon series, symbolic d")
    ser, (uc, vc, zc), con_s = horizon_series()
    check(
        "[3] series solvable to third order for symbolic d",
        all(c in ser for c in uc[:3] + vc[:3] + zc[:3]),
    )
    check(
        "[3] V'(1) = (d-2) eps/(d(d-1))",
        is_zero(ser[vc[0]] - (dsym - 2) * eps / (dsym * (dsym - 1))),
    )
    check(
        "[3] every V coefficient vanishes at eps = 0 (V = O(eps))",
        all(is_zero(ser[c].subs(eps, 0)) for c in vc[:3]),
    )
    check("[3] constraint satisfied order by order (through x^2)", is_zero(con_s))
    log(f"      v1 = {ser[vc[0]]},  z1 = {ser[zc[0]]},  u2 = {ser[uc[0]]}")
    ser4, *_ = bf.horizon_series()
    mine4 = {str(k): v.subs(dsym, 4) for k, v in ser.items()}
    common = [k for k in ser4 if str(k) in mine4]
    check(
        "[3] d=4: equals brane_flows.horizon_series exactly",
        len(common) >= 9 and all(is_zero(ser4[k] - mine4[str(k)]) for k in common),
        f"{len(common)} coefficients compared",
    )
    for d in (3, 5, 6):
        s_d, *_ = horizon_series(sp.Integer(d))
        check(
            f"[3] d={d}: integer-d series equals symbolic-d series",
            all(is_zero(s_d[k] - ser[k].subs(dsym, d)) for k in s_d),
        )

    # [4] the zero mode from Yang-Mills
    log("[4] the polarised LLL zero mode from the linearised YM equations")
    for d in (3, 4, 5, 6):
        res = {}
        for sgn in (-1, +1):
            out, target, g, (U, V, Z, w) = ym_mode(d, sgn)
            res[sgn] = (out, target, g)
        out, target, g = res[-1]
        # remaining equations must all be proportional to the target
        props = [sp.simplify(e / (g * target)) for (_, _, e) in out]
        free = [p.free_symbols for p in props]
        ok = len(out) == 2 and all(p.is_number and p != 0 for p in props)
        del free
        # and the other sign leaves a non-proportional equation (first-order in x)
        out_p, target_p, g_p = res[+1]
        bad = [
            e
            for (_, _, e) in out_p
            if sp.simplify(e / (g_p * target_p)).has(sp.Symbol("xx"))
            or sp.simplify(e / (g_p * target_p)).has(sp.Derivative)
        ]
        check(
            f"[4] d={d}: W_y = -i W_x reduces YM to (U e^((d-3)Z) w')' + b e^((d-3)Z-2V) w = 0; "
            "W_y = +i W_x does not",
            ok and len(bad) > 0,
            f"nonzero components (a, nu) = {[(a_ + 1, n_) for a_, n_, _ in out]}, "
            f"each = g x {props} x target",
        )

    # horizon regularity of the mode
    for d in (sp.Integer(4), dsym):
        s_ = ser if d == dsym else {k: v.subs(dsym, 4) for k, v in ser.items()}
        Bm, w1, w2, p2, q1 = mode_horizon(d, s_, uc, vc, zc)
        Us = d * x + s_[uc[0]] * x**2 + s_[uc[1]] * x**3
        Vs = s_[vc[0]] * x + s_[vc[1]] * x**2
        Zs = s_[zc[0]] * x + s_[zc[1]] * x**2
        P = Us * sp.exp((d - 3) * Zs)
        Q = Bm * sp.exp((d - 3) * Zs - 2 * Vs)
        ws = 1 + w1 * x + w2 * x**2
        resid = sp.series(sp.diff(P * ws.diff(x), x) + Q * ws, x, 0, 2).removeO()
        check(
            f"[4] {'d=4' if d == 4 else 'symbolic d'}: regular mode w = 1 - (B_m/d) x + w2 x^2 "
            "solves through O(x)",
            is_zero(resid),
        )
    log(
        "      w2 = -(2 p2 w1 + B_m (w1 + q1))/(4d),  p2 = u2 + d(d-3) z1,  "
        "q1 = (d-3) z1 - 2 v1"
    )

    # [5] boundary falloffs and the source
    log("[5] boundary falloffs and the source extraction")
    v = sp.Symbol("v", positive=True)
    s = sp.Symbol("s")
    wpow = r**s
    indicial = sp.factor(
        sp.expand(sp.diff(r ** (dsym - 1) * wpow.diff(r), r) / r ** (s + dsym - 3))
    )
    check(
        "[5] exponents 0 and -(d-2) (U -> r^2, e^{2(d-3)Z}-weight cancels)",
        set(sp.solve(indicial, s)) == {0, 2 - dsym},
        f"indicial {indicial}",
    )
    c0 = lambda f: f + r * f.diff(r) / (dsym - 2)
    check(
        "[5] c0 = w + r w'/(d-2) annihilates r^{-(d-2)} and keeps r^0",
        is_zero(c0(r ** (2 - dsym))) and is_zero(c0(sp.Integer(1)) - 1),
    )
    # constant branch to O(r^-2): (r^{d-1} w')' + (b/v) r^{d-5} w = 0
    a2 = sp.Symbol("a2")
    for d in (3, 5, 6, dsym):
        wb = 1 + a2 * r ** (-2)
        eq = sp.expand(
            (sp.diff(r ** (d - 1) * wb.diff(r), r) + (b / v) * r ** (d - 5))
            / r ** (d - 5)
        )
        a2v = (
            sp.solve(eq.subs(r, 1), a2)[0]
            if d != dsym
            else sp.solve(sp.expand(eq), a2)[0]
        )
        resid = sp.simplify(c0(wb).subs(a2, a2v).subs(dsym, d) - 1)
        check(
            f"[5] {'symbolic d' if d == dsym else f'd={d}'}: residual of c0 on the r^0 branch "
            "= (b/v) r^-2/(2(d-2))",
            is_zero(resid - (b / v) / (2 * (d - 2)) / r**2),
        )
    # d = 4: w = 1 + k r^-2 ln r + m r^-2
    k_, m_ = sp.symbols("k m")
    wb = 1 + k_ * sp.log(r) / r**2
    eq = sp.simplify((sp.diff(r**3 * wb.diff(r), r) + (b / v) / r) * r)
    kv = sp.solve(eq, k_)[0]
    resid = sp.simplify((wb + r * wb.diff(r) / 2).subs(k_, kv) - 1)
    check(
        "[5] d=4: the r^-2 ln r branch leaves the same residual (b/v) r^-2/4",
        is_zero(resid - (b / v) / 4 / r**2),
        f"k = {kv}",
    )
    # U -> r^2 forced
    a_ = sp.Symbol("a", positive=True)
    Upp, _, _, _ = closed_form()
    lead = sp.simplify(
        2 * a_
        - Upp.subs(
            {Uv: a_ * r**2, Up: 2 * a_ * r, Vv: sp.log(r), Vp: 1 / r, Zp: 1 / r}
        ).subs(sp.exp(-4 * sp.log(r)), r**-4)
    )
    lead0 = sp.limit(lead, r, sp.oo)
    check(
        "[5] U'' equation forces U -> r^2 at the boundary (no free normalisation)",
        sp.solve(lead0, a_) == [1],
        f"leading condition {sp.factor(lead0)} = 0",
    )

    # [6] symmetries behind the frame map
    log("[6] symmetries and the frame map")
    for d in (3, 4, 5, 6):
        check(
            f"[6] d={d}: equations covariant under r-scaling and under V, Z shifts with b -> b e^(2c); "
            "mode equation invariant",
            symmetries_hold(d),
        )
    al = sp.Rational(16) / (dsym * (dsym - 1) * (dsym - 2))
    check(
        "[6] alpha_* = beta_ext/B_thr^2 = 16/(d(d-1)(d-2)), B_thr = d(d-1)/4 the throat BF value",
        is_zero(beta_ext(dsym) / b_thr(dsym) ** 2 - al),
    )
    # B_thr: on the eps = 0 throat, w ~ r^s at large r: s^2 + (d-2) s + B l^2 = 0
    Bs = sp.Symbol("B_s", positive=True)
    l2 = (dsym - 2) ** 2 / (dsym * (dsym - 1))
    disc = sp.solve(sp.Eq((dsym - 2) ** 2 - 4 * Bs * l2, 0), Bs)
    check("[6] throat BF threshold B_thr = d(d-1)/4", is_zero(disc[0] - b_thr(dsym)))

    # [7] d = 3 anchor: magnetic RN-AdS4 in the raw frame
    log("[7] d = 3: non-extremal magnetic RN-AdS4 in the raw frame")
    q = sp.Symbol("q", positive=True)
    lam3 = 3 / (3 - q)
    rho = 1 + (r - 1) / lam3
    URN = lambda y: y**2 - (1 + q) / y + q / y**2
    Ue = lam3**2 * URN(rho)
    Ve = sp.log(rho)
    Upp, Vpp, _, con = closed_form(sp.Integer(3), 2 * q)
    sub = {Uv: Ue, Vv: Ve, Zv: 0, Up: Ue.diff(r), Vp: Ve.diff(r), Zp: 0}
    ok = (
        is_zero(Ue.diff(r, 2) - Upp.subs(sub))
        and is_zero(Ve.diff(r, 2) - Vpp.subs(sub))
        and is_zero(con.subs(sub))
        and is_zero(Ue.subs(r, 1))
        and is_zero(Ue.diff(r).subs(r, 1) - 3)
        and is_zero(sp.limit(sp.exp(2 * Ve) / r**2, r, sp.oo) - lam3**-2)
        and is_zero(sp.limit(Ue / r**2, r, sp.oo) - 1)
    )
    check(
        "[7] raw-frame RN-AdS4: U(1) = 0, U'(1) = 3, solves the d = 3 system, v_inf = 1/lam^2",
        ok,
    )

    failed = [k for k, v_ in CHECKS.items() if not v_]
    log(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if failed:
        sys.exit(1)
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
