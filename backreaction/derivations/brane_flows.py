"""Reduced Einstein-Maxwell equations of the magnetic brane in general d, derived
from scratch in two gauges, for the zero-temperature threshold argument and the
horizon-shooting onset scan.

Field equations (L = 1; the d = 4 form is the one of `dk_background.py`):

    R_MN + d g_MN = alpha (F_MP F_N^P - F^2 g_MN / (2(d-1))),     F_xy = B,

the trace-reversed Einstein equation of (1/2 kappa^2)(R + d(d-1)) - F^2/(4 g^2)
with alpha = kappa^2/g^2.  The bulk has coordinates (t, x, y, z_1..z_{d-3}, rho).

GAUGE 1 -- domain wall, zero temperature (boost invariance in (t, z_i)):

    ds^2 = d rho^2 + e^{2A}(-dt^2 + dz_i dz_i) + e^{2C}(dx^2 + dy^2).

GAUGE 2 -- finite temperature, the chart of `dk_background.py` generalised:

    ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2 + dy^2) + e^{2Z} dz_i dz_i.

Results
-------
[1] Gauge 2 at d = 4 reproduces the production reduced system of
    `dk_background._build_symbolics` term by term (U'', V'', W'' and the
    constraint).  The equations used by the scan are therefore the published
    ones, re-derived by independent code.
[2] Gauge 1 has the AdS_{d-1} x R^2 fixed point A = rho/l, e^{2C} = v with
        l^2 = (d-2)^2/(d(d-1)),    v = B sqrt(alpha (d-2)/(d(d-1))),
    and on it the polarised lowest-Landau-level operator
        (e^{(d-2)A} w')' + B e^{(d-2)A - 2C} w = 0
    (the gauge-1 form of P = U e^{(d-3)Z}, Q = B e^{(d-3)Z-2V}; checked by the
    change of variables dr = e^A d rho) has exponents w ~ e^{s rho},
        s^2 + (d-2) s / l + B/v = 0,
    whose discriminant vanishes at alpha_*(d) = 16/(d(d-1)(d-2)).
[3] The fixed point has exactly one relevant-towards-the-UV (growing)
    deformation, delta C = e^{k rho}, delta A = a_k e^{k rho}; in the r chart,
    where r - r0 ~ e^{rho/l}, its exponent is p = k l.  For d = 4,
    p = sqrt(19/3) - 1 exactly (the exponent of `dk_throat_matching.py`).
    The k = 0 rho-translation mode is the only other non-decaying one.
[4] Raw-frame horizon data for shooting at d = 4 (horizon r = 1, U'(1) = 4,
    V(1) = W(1) = 0, raw field b, eps = 6 - alpha b^2): the near-horizon
    series of V is proportional to eps at every order computed, so V = 0
    exactly at eps = 0 (the BTZ x R^2 throat), and the shooting can be
    parametrised by eps down to 1e-30 without cancellation.
[5] The regular near-horizon series of the polarised LLL mode on that brane,
    w = 1 + w1 x + w2 x^2 with w1 = -B_m/4 and w2 = B_m(3 B_m - 2 eps + 36)/192,
    the series start of the onset scan (`mode_horizon_series`).

Run:  uv run python -m backreaction.derivations.brane_flows   (~1 min)
"""

import sys
import time

import sympy as sp

T0 = time.time()
CHECKS = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def is_zero(e):
    e = sp.expand(e)
    return e == 0 or sp.simplify(e) == 0


def ricci_diag(gd, xs, rad):
    """Ricci tensor of a diagonal metric diag(gd) depending on xs[rad] only."""
    n = len(xs)
    gi = [1 / g for g in gd]

    def Gam(l, m, k):  # Gamma^l_{mk}
        e = 0
        if m == l:
            e += sp.diff(gd[l], xs[k])
        if k == l:
            e += sp.diff(gd[l], xs[m])
        if m == k:
            e -= sp.diff(gd[m], xs[l])
        return gi[l] * e / 2

    G = [[[Gam(l, m, k) for k in range(n)] for m in range(n)] for l in range(n)]
    R = []
    for m in range(n):
        e = 0
        for l in range(n):
            e += sp.diff(G[l][m][m], xs[l]) - sp.diff(G[l][m][l], xs[m])
            for s in range(n):
                e += G[l][l][s] * G[s][m][m] - G[l][m][s] * G[s][m][l]
        R.append(sp.simplify(e))
    return R


def field_equations(d, gd, xs, ix, iy, rad, B, alpha):
    """Diagonal components of R_MN + d g_MN - alpha(F F - F^2 g/(2(d-1)))."""
    R = ricci_diag(gd, xs, rad)
    F2 = 2 * B**2 / (gd[ix] * gd[iy])
    out = []
    for m in range(len(xs)):
        FF = B**2 / gd[iy] if m == ix else (B**2 / gd[ix] if m == iy else 0)
        out.append(
            sp.simplify(R[m] + d * gd[m] - alpha * (FF - F2 * gd[m] / (2 * (d - 1))))
        )
    return out


B, alpha = sp.symbols("B alpha", positive=True)


# ---------------------------------------------------------------------------
# gauge 1: domain wall, T = 0
# ---------------------------------------------------------------------------
def domain_wall(d):
    """Return (A'', C'', constraint) as expressions in (A', C', C) and the
    symbols used.  The equations do not involve A itself (boost + translation)."""
    rho = sp.Symbol("rho", real=True)
    nz = d - 3
    xs = [sp.Symbol(n) for n in ["t", "x", "y"] + [f"z{i}" for i in range(nz)]] + [rho]
    A = sp.Function("A")(rho)
    C = sp.Function("C")(rho)
    gd = [-sp.exp(2 * A), sp.exp(2 * C), sp.exp(2 * C)] + [sp.exp(2 * A)] * nz + [1]
    eqs = field_equations(d, gd, xs, 1, 2, len(xs) - 1, B, alpha)
    a1, c1, a2, c2, cc = sp.symbols("a1 c1 a2 c2 cc")
    rep = {
        A.diff(rho, 2): a2,
        C.diff(rho, 2): c2,
        A.diff(rho): a1,
        C.diff(rho): c1,
        C: cc,
    }
    tt = sp.expand(sp.simplify(eqs[0] / gd[0]).subs(rep))
    xx = sp.expand(sp.simplify(eqs[1] / gd[1]).subs(rep))
    rr = sp.expand(eqs[-1].subs(rep))
    sol = sp.solve([tt, xx], [a2, c2], dict=True)[0]
    con = sp.simplify(rr.subs(sol))
    assert A not in con.free_symbols
    # the z_i equations coincide with tt (boost invariance) when present
    if nz:
        zz = sp.expand(sp.simplify(eqs[3] / gd[3]).subs(rep))
        assert is_zero(zz - tt), d
    q = lambda e: sp.nsimplify(sp.simplify(e), rational=True)
    return dict(App=q(sol[a2]), Cpp=q(sol[c2]), con=q(con), syms=(a1, c1, cc), rho=rho)


# ---------------------------------------------------------------------------
# gauge 2: finite temperature, r chart
# ---------------------------------------------------------------------------
def r_chart(d):
    r = sp.Symbol("r", real=True)
    nz = d - 3
    xs = [sp.Symbol(n) for n in ["t", "x", "y"] + [f"z{i}" for i in range(nz)]] + [r]
    U = sp.Function("U")(r)
    V = sp.Function("V")(r)
    Z = sp.Function("Z")(r)
    gd = [-U, sp.exp(2 * V), sp.exp(2 * V)] + [sp.exp(2 * Z)] * nz + [1 / U]
    eqs = field_equations(d, gd, xs, 1, 2, len(xs) - 1, B, alpha)
    Uv, Vv, Zv, Up, Vp, Zp, Upp, Vpp, Zpp = sp.symbols("Uv Vv Zv Up Vp Zp Upp Vpp Zpp")
    rep = {
        U.diff(r, 2): Upp,
        V.diff(r, 2): Vpp,
        Z.diff(r, 2): Zpp,
        U.diff(r): Up,
        V.diff(r): Vp,
        Z.diff(r): Zp,
        U: Uv,
        V: Vv,
        Z: Zv,
    }
    tt = sp.expand(eqs[0].subs(rep))
    xx = sp.expand(eqs[1].subs(rep))
    zz = sp.expand(eqs[3].subs(rep))
    rr = sp.expand(eqs[-1].subs(rep))
    sol = sp.solve([tt, xx, zz], [Upp, Vpp, Zpp], dict=True)[0]
    con = sp.simplify(rr.subs(sol))
    return dict(
        Upp=sp.simplify(sol[Upp]),
        Vpp=sp.simplify(sol[Vpp]),
        Zpp=sp.simplify(sol[Zpp]),
        con=con,
        syms=(Uv, Vv, Zv, Up, Vp, Zp),
        r=r,
    )


def fixed_point(d, dw):
    """AdS_{d-1} x R^2 in gauge 1: A' = 1/l, C' = 0, e^{2C} = v."""
    l, v = sp.symbols("l v", positive=True)
    a1, c1, cc = dw["syms"]
    sub = {a1: 1 / l, c1: 0, cc: sp.log(v) / 2}
    eqs = [
        sp.simplify(dw["App"].subs(sub)),
        sp.simplify(dw["Cpp"].subs(sub)),
        sp.simplify(dw["con"].subs(sub)),
    ]
    sol = sp.solve(eqs, [l, v], dict=True)
    sol = [
        s for s in sol if l in s and v in s and s[l].is_positive and s[v].is_positive
    ]
    assert len(sol) == 1, sol
    L, Vf = sol[0][l], sol[0][v]
    assert all(is_zero(e.subs({l: L, v: Vf})) for e in eqs)
    return sp.simplify(L), sp.simplify(Vf)


def irrelevant(d, dw, L, Vf):
    """Linearise gauge 1 about the fixed point: C = C0 + e^{k rho},
    A = rho/l + a e^{k rho}.  Returns the list of (k, a) with k != 0."""
    a1, c1, cc = dw["syms"]
    rho = dw["rho"]
    eps, k, a = sp.symbols("epsilon k a")
    Cl = sp.log(Vf) / 2 + eps * sp.exp(k * rho)
    Al = rho / L + eps * a * sp.exp(k * rho)
    sub = {a1: Al.diff(rho), c1: Cl.diff(rho), cc: Cl}

    def lin(expr, lhs):
        e = sp.diff(lhs - expr.subs(sub), eps).subs(eps, 0)
        return sp.simplify(e * sp.exp(-k * rho))

    eqA = lin(dw["App"], Al.diff(rho, 2))
    eqC = lin(dw["Cpp"], Cl.diff(rho, 2))
    eqK = lin(dw["con"], 0)
    sols = sp.solve([eqA, eqC, eqK], [k, a], dict=True)
    return [(sp.nsimplify(s[k]), s[a]) for s in sols if s[k] != 0]


def horizon_series(order=4):
    """Raw-frame near-horizon series at d = 4 in x = r - 1 and eps = 6 - alpha b^2:
    U = 4x + u2 x^2 + ..., V = v1 x + ..., Z = z1 x + ... .  Returns (coefficient
    dict, eps symbol, (uc, vc, zc))."""
    rc = r_chart(4)
    U2, V2, Z2, Up2, Vp2, Zp2 = rc["syms"]
    x, eps = sp.symbols("x epsilon")
    uc = sp.symbols(f"u2:{order + 1}")
    vc = sp.symbols(f"v1:{order + 1}")
    zc = sp.symbols(f"z1:{order + 1}")
    Us = 4 * x + sum(uc[i] * x ** (i + 2) for i in range(order - 1))
    Vs = sum(vc[i] * x ** (i + 1) for i in range(order))
    Zs = sum(zc[i] * x ** (i + 1) for i in range(order))
    beta_raw = 6 - eps  # alpha b^2 with the code's alpha absorbed
    sub = {
        U2: Us,
        V2: Vs,
        Z2: Zs,
        Up2: Us.diff(x),
        Vp2: Vs.diff(x),
        Zp2: Zs.diff(x),
        alpha: beta_raw,
        B: 1,
    }
    res = [
        sp.series(Us.diff(x, 2) - rc["Upp"].subs(sub), x, 0, order).removeO(),
        sp.series((Us * (Vs.diff(x, 2) - rc["Vpp"].subs(sub))), x, 0, order).removeO(),
        sp.series((Us * (Zs.diff(x, 2) - rc["Zpp"].subs(sub))), x, 0, order).removeO(),
    ]
    eqs = []
    for e in res:
        e = sp.expand(e)
        eqs += [e.coeff(x, j) for j in range(order)]
    # order by order: the x^j coefficients fix (u_{j+2}, v_{j+1}, z_{j+1})
    sol = {}
    for j in range(order - 1):
        ej = [sp.simplify(e.subs(sol)) for e in eqs[j::order]]
        sj = sp.solve(ej, [uc[j], vc[j], zc[j]], dict=True)
        if len(sj) != 1:
            break
        sol.update({k_: sp.factor(v_) for k_, v_ in sj[0].items()})
    sol = {k_: sp.factor(v_.subs(sol)) for k_, v_ in sol.items()}
    return sol, eps, (uc, vc, zc)


# ---------------------------------------------------------------------------
# horizon-shooting start of the polarised LLL mode (d = 4, raw frame)
# ---------------------------------------------------------------------------
def mode_horizon_series(series=None, order=3):
    """Regular near-horizon series of the polarised lowest-Landau-level mode
    (P w')' + Q w = 0, P = U e^Z, Q = B_m e^{Z - 2V}, on the raw-frame brane of
    `horizon_series`:  w = 1 + w1 x + w2 x^2 + ...  (x = r - 1).  Returns
    (dict w_k -> expression in (B_m, eps), B_m symbol, eps symbol, residual), the
    residual being the lowest surviving order of (P w')' + Q w, which must be
    O(x^{order - 1}) once the coefficients are inserted."""
    sol, eps, (uc, vc, zc) = series if series is not None else horizon_series()
    x, Bm = sp.symbols("x B_m")
    nU = len(uc)
    Us = 4 * x + sum(uc[i] * x ** (i + 2) for i in range(nU))
    Vs = sum(vc[i] * x ** (i + 1) for i in range(len(vc)))
    Zs = sum(zc[i] * x ** (i + 1) for i in range(len(zc)))
    Us, Vs, Zs = (e.subs(sol) for e in (Us, Vs, Zs))
    wc = sp.symbols(f"w1:{order}")
    ws = 1 + sum(wc[i] * x ** (i + 1) for i in range(order - 1))
    P = Us * sp.exp(Zs)
    Q = Bm * sp.exp(Zs - 2 * Vs)
    ode = sp.series(sp.diff(P * sp.diff(ws, x), x) + Q * ws, x, 0, order - 1).removeO()
    ode = sp.expand(ode)
    out = {}
    for j in range(order - 1):
        cj = sp.expand(ode.coeff(x, j).subs(out))
        sj = sp.solve(cj, wc[j])
        assert len(sj) == 1, (j, sj)
        out[wc[j]] = sp.factor(sj[0])
    res = sp.expand(ode.subs(out))
    return out, Bm, eps, res


def check_mode_series():
    """[5] the mode start used by `numerics/onset_scan.py`: w1 = -B_m/4 and the
    second-order coefficient in terms of the background coefficients."""
    series = horizon_series()
    sol, eps, (uc, vc, zc) = series
    w, Bm, _, res = mode_horizon_series(series)
    w1, w2 = sp.symbols("w1 w2")
    check("[5] mode series solves (P w')' + Q w = 0 through O(x)", is_zero(res))
    check("[5] w1 = -B_m/4 (horizon regularity)", is_zero(w[w1] + Bm / 4))
    # closed form in the background coefficients: P = 4x + p2 x^2, Q = B_m(1 + q1 x)
    p2 = sol[uc[0]] + 4 * sol[zc[0]]
    q1 = sol[zc[0]] - 2 * sol[vc[0]]
    W1 = -Bm / 4
    w2_closed = -(2 * p2 * W1 + Bm * (W1 + q1)) / 16
    check(
        "[5] w2 = -(2 p2 w1 + B_m (w1 + q1))/16, p2 = u2 + 4 z1, q1 = z1 - 2 v1",
        is_zero(w[w2] - w2_closed),
        f"w2 = {sp.factor(w[w2])}",
    )
    wrong = -(p2 * W1 + Bm * (W1 + q1)) / 8
    check(
        "[5] negative control: -(p2 w1 + B_m (w1 + q1))/8 is not the coefficient",
        not is_zero(w[w2] - wrong),
    )
    return w, Bm, eps


def main():
    # [1] gauge 2 at d = 4 against the production system
    log("[1] r chart, d = 4, against dk_background._build_symbolics")
    rc = r_chart(4)
    from backreaction.numerics import dk_background as dk

    S = dk._SYM["_sym"]
    rr_, Uv, Vv, Wv, Up, Vp, Wp, Bs, als, *_ = S["symbols"]
    U2, V2, Z2, Up2, Vp2, Zp2 = rc["syms"]
    m = {U2: Uv, V2: Vv, Z2: Wv, Up2: Up, Vp2: Vp, Zp2: Wp, B: Bs, alpha: als}
    for mine, theirs in (("Upp", "Upp"), ("Vpp", "Vpp"), ("Zpp", "Wpp")):
        check(
            f"[1] {mine} equals production {theirs}",
            is_zero(rc[mine].xreplace(m) - S[theirs]),
        )
    # constraints may differ by an overall factor: compare after normalising
    ratio = sp.simplify(rc["con"].xreplace(m) / S["constraint"])
    check(
        "[1] constraint proportional to production constraint",
        ratio.is_number,
        f"ratio {ratio}",
    )

    # [2], [3] gauge 1 for d = 3..6
    for d in (3, 4, 5, 6):
        log(f"[2] domain wall, d = {d}")
        dw = domain_wall(d)
        L, Vf = fixed_point(d, dw)
        check(
            f"[2] d={d}: l^2 = (d-2)^2/(d(d-1))",
            is_zero(L**2 - sp.Rational((d - 2) ** 2, d * (d - 1))),
            f"l^2 = {L**2}",
        )
        check(
            f"[2] d={d}: v = B sqrt(alpha (d-2)/(d(d-1)))",
            is_zero(Vf**2 - B**2 * alpha * sp.Rational(d - 2, d * (d - 1))),
        )
        # LLL operator on the fixed point: discriminant of s^2 + (d-2)s/l + B/v
        disc = sp.simplify((d - 2) ** 2 / L**2 - 4 * B / Vf)
        ast = sp.solve(sp.Eq(disc, 0), alpha)
        check(
            f"[2] d={d}: BF saturation at alpha_* = 16/(d(d-1)(d-2))",
            len(ast) == 1 and is_zero(ast[0] - sp.Rational(16, d * (d - 1) * (d - 2))),
            f"alpha_* = {ast}",
        )
        # gauge-1 form of the LLL operator from the r-chart (P, Q):
        # dr = e^A d rho, U = e^{2A}, e^{2Z} = e^{2A}:  P_r = e^{(d-1)A},
        # Q_r = B e^{(d-3)A - 2C}; (P_r w_r)_r + Q_r w = e^{-A}[(e^{(d-2)A} w')' + B e^{(d-2)A-2C} w]
        rho = dw["rho"]
        Af, Cf, wf = (sp.Function(n)(rho) for n in ("Af", "Cf", "wf"))
        Pr = sp.exp((d - 1) * Af)
        Qr = B * sp.exp((d - 3) * Af - 2 * Cf)
        lhs_r = sp.exp(-Af) * sp.diff(Pr * sp.exp(-Af) * wf.diff(rho), rho) + Qr * wf
        lhs_1 = sp.exp(-Af) * (
            sp.diff(sp.exp((d - 2) * Af) * wf.diff(rho), rho)
            + B * sp.exp((d - 2) * Af - 2 * Cf) * wf
        )
        check(
            f"[2] d={d}: LLL operator in the domain-wall gauge",
            is_zero(sp.simplify(lhs_r - lhs_1)),
        )
        irr = irrelevant(d, dw, L, Vf)
        pos = [(k, a) for k, a in irr if sp.N(k) > 0]
        log(f"      deformations: {[(sp.nsimplify(k), sp.N(k, 8)) for k, _ in irr]}")
        check(f"[3] d={d}: exactly one growing deformation", len(pos) == 1)
        k, a = pos[0]
        p = sp.simplify(k * L)
        log(f"      k = {sp.N(k, 10)},  p = k l = {sp.N(p, 10)},  a_k = {sp.N(a, 10)}")
        if d == 4:
            check(
                "[3] d=4: p = sqrt(19/3) - 1",
                is_zero(sp.radsimp(p - (sp.sqrt(sp.Rational(19, 3)) - 1))),
            )

    # [4] raw-frame horizon series at d = 4 in eps = 6 - alpha b^2
    log("[4] raw-frame horizon series, d = 4")
    sol, eps, (uc, vc, zc) = horizon_series()
    order = len(vc)
    check(
        "[4] horizon series solvable order by order",
        all(c in sol for c in vc[: order - 1]),
        f"{len(sol)} coefficients",
    )
    log(f"      v1 = {sp.factor(sol[vc[0]])},  z1 = {sp.factor(sol[zc[0]])}")
    check("[4] V'(1) = eps/6", is_zero(sol[vc[0]] - eps / 6))
    check(
        "[4] every V coefficient vanishes at eps = 0 (V = O(eps))",
        all(is_zero(sol[c].subs(eps, 0)) for c in vc[: order - 1]),
    )
    log("      " + ", ".join(f"{k_} = {v_}" for k_, v_ in sol.items()))

    log("[5] near-horizon series of the polarised LLL mode, d = 4")
    check_mode_series()

    failed = [k for k, v in CHECKS.items() if not v]
    log(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
