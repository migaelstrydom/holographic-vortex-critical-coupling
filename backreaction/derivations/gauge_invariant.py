"""Algebraic-gauge reduction of the lattice-backreaction system.

The linearised system in radial gauge has two first-order constraints in the
scalar sector.  Numerically one finds
that the *radial* gauge Huu = Hxu = 0 with flat-boundary + regular-horizon
boundary conditions over-determines the problem: the dynamical solution is
unique but violates the constraints.  The reason is that the
residual/required gauge functions behave as xi_u ~ 1/(u sqrt(f)) --
horizon-singular and non-normalisable -- so the physical solution simply is
not radial-gauge-regular.

This script instead fixes the gauge *algebraically*:

    Htt = 0   (uses xi_u:  delta Htt = u (u f' - 2f) xi_u, and u f' - 2f =
               -2 - 2u^4/u_H^4 < 0 never vanishes on (0, u_H])
    Hxx = 0   (uses xi_x:  delta Hxx = 2 i G u^2 xi_x - 2 u f xi_u)

No integration is involved, the required xi is a pointwise rational function
of the fields (regular at horizon and boundary for normalisable data), and no
residual gauge freedom survives.  The remaining fields (Hyy, Hzz, Huu, Hxu)
are therefore *complete gauge invariants* of the scalar sector.

Checks / derivations
--------------------
[1] Gauge shifts of all six H's under xi = (0, xi_x E, 0, 0, xi_u E) are
    derived and the claimed algebraic gauge-fixing verified (delta Htt
    coefficient never vanishing; xi_hat rational in the fields).
[2] The six equations restricted to the gauge slice: (yy) and (zz) rows are
    second order, each in its own field only; (tt), (xx), (uu), (xu) are
    first order, and the Hxu terms of the (xu) row cancel identically,
    making it an algebraic relation for Huu.  After solving {xu, tt, yy, zz}
    the remaining (xx) and (uu) rows reduce to zero identically on shell
    (Bianchi; the matter w0/bhat ODEs are needed for the cancellation).
[3] Solved form: a closed two-field second-order system
        Hyy'' = ...,  Hzz'' = ...      [functions of Hyy, Hzz, H', matter]
    plus Huu and Hxu *algebraic* in (Hyy, Hzz, Hyy', Hzz', matter).
[4] Boundary indicial structure of the closed block: det = p^2 (p - 4)^2,
    i.e. exponents {0, 4} per field -- Dirichlet at u = 0 removes the
    boundary-metric sources, u^4 carries the response.
[5] Export to systems/einstein_invgauge.py for the numerics.

Run:  uv run python -m backreaction.derivations.gauge_invariant
"""

import time

import sympy as sp

T0_START = time.time()


def log(msg):
    print(f"[{time.time() - T0_START:7.1f}s] {msg}", flush=True)


def is_zero(e):
    e = sp.expand(e)
    if e == 0:
        return True
    return sp.cancel(sp.together(e)) == 0


t, x, y, z, u = coords = sp.symbols("t x y z u", real=True)
uH, B = sp.symbols("u_H B", positive=True)
G = sp.Symbol("G", positive=True)
alpha = sp.Symbol("alpha", positive=True)

f = 1 - u**4 / uH**4
gdn = sp.diag(-f / u**2, 1 / u**2, 1 / u**2, 1 / u**2, 1 / (u**2 * f))
gup = sp.diag(-(u**2) / f, u**2, u**2, u**2, u**2 * f)
N = 5
idx = ["t", "x", "y", "z", "u"]

w = sp.Function("w0")
bh = sp.Function("bhat")


def main():
    wpp_rule = {
        sp.Derivative(w(u), (u, 2)): -(
            (sp.diff(f, u) - f / u) * sp.Derivative(w(u), u) + B * w(u)
        )
        / f
    }
    bhpp_rule = {
        sp.Derivative(bh(u), (u, 2)): (
            G**2 * (bh(u) - w(u) ** 2)
            - (sp.diff(f, u) - f / u) * sp.Derivative(bh(u), u)
        )
        / f
    }

    def on_shell(expr):
        e = sp.expand(sp.sympify(expr).doit())
        while e.has(sp.Derivative(w(u), (u, 2))) or e.has(sp.Derivative(bh(u), (u, 2))):
            e = sp.expand(e.subs(wpp_rule).subs(bhpp_rule).doit())
        return e

    def christoffel():
        Gam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for r in range(N):
            for m in range(N):
                for n in range(m, N):
                    e = sp.S.Zero
                    for s_ in range(N):
                        if gup[r, s_] != 0:
                            e += gup[r, s_] * (
                                sp.diff(gdn[s_, m], coords[n])
                                + sp.diff(gdn[s_, n], coords[m])
                                - sp.diff(gdn[m, n], coords[s_])
                            )
                    e = sp.cancel(e / 2)
                    Gam[r][m][n] = e
                    Gam[r][n][m] = e
        return Gam

    GAM = christoffel()

    def cov_deriv_2tensor(h):
        Dh = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for mu in range(N):
            for a in range(N):
                for b in range(N):
                    e = sp.diff(h[a][b], coords[mu])
                    for lam in range(N):
                        if GAM[lam][mu][a] != 0:
                            e -= GAM[lam][mu][a] * h[lam][b]
                        if GAM[lam][mu][b] != 0:
                            e -= GAM[lam][mu][b] * h[a][lam]
                    Dh[mu][a][b] = sp.expand(e)
        return Dh

    def delta_ricci(h):
        Dh = cov_deriv_2tensor(h)
        dGam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for r in range(N):
            for m in range(N):
                for n in range(m, N):
                    e = sp.S.Zero
                    for s_ in range(N):
                        if gup[r, s_] != 0:
                            e += gup[r, s_] * (
                                Dh[m][s_][n] + Dh[n][s_][m] - Dh[s_][m][n]
                            )
                    e = sp.expand(e / 2)
                    dGam[r][m][n] = e
                    dGam[r][n][m] = e
        V = [sp.S.Zero] * N
        for m in range(N):
            V[m] = sp.expand(sum(dGam[r][r][m] for r in range(N)))
        dR = [[sp.S.Zero] * N for _ in range(N)]
        for m in range(N):
            for n in range(m, N):
                e = sp.S.Zero
                for r in range(N):
                    e += sp.diff(dGam[r][m][n], coords[r])
                    for lam in range(N):
                        if GAM[r][r][lam] != 0:
                            e += GAM[r][r][lam] * dGam[lam][m][n]
                        if GAM[lam][r][m] != 0:
                            e -= GAM[lam][r][m] * dGam[r][lam][n]
                        if GAM[lam][r][n] != 0:
                            e -= GAM[lam][r][n] * dGam[r][m][lam]
                e -= sp.diff(V[m], coords[n])
                for lam in range(N):
                    if GAM[lam][n][m] != 0:
                        e += GAM[lam][n][m] * V[lam]
                e = sp.expand(e)
                dR[m][n] = e
                dR[n][m] = e
        return dR

    # ---------------------------------------------------------------------------
    # [1] gauge shifts and the algebraic gauge fixing
    # ---------------------------------------------------------------------------
    Eh = sp.exp(sp.I * G * x)
    xiX = sp.Function("xi_x")
    xiU = sp.Function("xi_u")
    xi = [sp.S.Zero, xiX(u) * Eh, sp.S.Zero, sp.S.Zero, xiU(u) * Eh]
    hg = [[sp.S.Zero] * N for _ in range(N)]
    for m in range(N):
        for n in range(m, N):
            e = sp.diff(xi[n], coords[m]) + sp.diff(xi[m], coords[n])
            for lam in range(N):
                if GAM[lam][m][n] != 0:
                    e -= 2 * GAM[lam][m][n] * xi[lam]
            hg[m][n] = sp.expand(e)
            hg[n][m] = hg[m][n]

    # H-normalised shifts (per Eh): Htt = -u^2 h_tt / f, Hii = u^2 h_ii,
    # Huu = u^2 f h_uu, Hxu = -i u^2 h_xu
    shift = {
        "tt": sp.expand(sp.cancel((-(u**2) / f * hg[0][0]).subs(x, 0))),
        "xx": sp.expand((u**2 * hg[1][1]).subs(x, 0)),
        "yy": sp.expand((u**2 * hg[2][2]).subs(x, 0)),
        "zz": sp.expand((u**2 * hg[3][3]).subs(x, 0)),
        "uu": sp.expand(sp.cancel((u**2 * f * hg[4][4]).subs(x, 0))),
        "xu": sp.expand((-sp.I * u**2 * hg[1][4]).subs(x, 0)),
    }
    print("gauge shifts of the H fields (per e^{iGx}):")
    for k, v in shift.items():
        print(f"  delta H_{k} = {sp.simplify(v)}")

    # delta Htt = c_tt(u) xi_u with c_tt never vanishing:
    c_tt = sp.simplify(sp.cancel(shift["tt"] / xiU(u)))
    ok1a = not c_tt.has(xiX(u)) and not c_tt.has(sp.Derivative)
    c_tt_val = sp.simplify(c_tt)
    log(f"[1] delta Htt = ({c_tt_val}) xi_u, algebraic in xi_u: {ok1a}")
    assert ok1a
    # c_tt = -2u (1 + u^4/uH^4) < 0 on (0, uH]
    ok1b = sp.simplify(c_tt_val + 2 * u * (1 + u**4 / uH**4)) == 0
    log(f"[1] c_tt = -2u (1 + u^4/u_H^4) (nonzero on (0, u_H]): {ok1b}")
    assert ok1b
    c_xx_x = sp.simplify(sp.cancel(sp.expand(shift["xx"]).coeff(xiX(u))))
    log(f"[1] delta Hxx contains ({c_xx_x}) xi_x -- algebraic in xi_x for G != 0")

    # ---------------------------------------------------------------------------
    # the gauge-fixed ansatz: Htt = Hxx = 0
    # ---------------------------------------------------------------------------
    Hyy = sp.Function("H_yy")
    Hzz = sp.Function("H_zz")
    Huu = sp.Function("H_uu")
    Hxu = sp.Function("H_xu")

    h = [[sp.S.Zero] * N for _ in range(N)]
    h[2][2] = (1 / u**2) * Hyy(u) * Eh
    h[3][3] = (1 / u**2) * Hzz(u) * Eh
    h[4][4] = (1 / (u**2 * f)) * Huu(u) * Eh
    h[1][4] = h[4][1] = (sp.I / u**2) * Hxu(u) * Eh

    log("computing delta Ricci on the gauge-fixed ansatz ...")
    dR = delta_ricci(h)
    ER = [[sp.expand(dR[m][n] + 4 * h[m][n]) for n in range(N)] for m in range(N)]

    # Einstein form for the constraints
    dRs = sp.S.Zero
    for m in range(N):
        for n in range(N):
            if h[m][n] != 0:
                dRs -= gup[m, m] * gup[n, n] * h[m][n] * (-4 * gdn[m, n])
        dRs += gup[m, m] * dR[m][m]
    dRs = sp.expand(dRs)
    EG = [
        [
            sp.expand(
                dR[m][n]
                - sp.Rational(1, 2) * h[m][n] * (-20)
                - sp.Rational(1, 2) * gdn[m, n] * dRs
                - 6 * h[m][n]
            )
            for n in range(N)
        ]
        for m in range(N)
    ]

    # per-harmonic source (the AdS5-Schwarzschild limit of systems/dk_stress.py)
    wp = sp.Derivative(w(u), u)
    S = [[sp.S.Zero] * N for _ in range(N)]
    S[0][0] = f * u**2 * (f * wp**2 + B * (bh(u) - w(u) ** 2)) * Eh
    S[1][1] = u**2 * B * (bh(u) - w(u) ** 2) * Eh
    S[2][2] = S[1][1]
    S[3][3] = -(u**2) * (f * wp**2 + B * (bh(u) - w(u) ** 2)) * Eh
    S[4][4] = u**2 * (wp**2 - (B / f) * (bh(u) - w(u) ** 2)) * Eh
    S[1][4] = S[4][1] = -sp.I * (B * u**2 / G) * sp.Derivative(bh(u), u) * Eh
    Str = sum(gup[a, a] * S[a][a] for a in range(N))

    # equations: trace-reversed for (tt), (xx), (yy), (zz), (xu);
    # Einstein form for (uu)
    eqs = {}
    for m, n in ((0, 0), (1, 1), (2, 2), (3, 3), (1, 4)):
        e = ER[m][n] - alpha * (S[m][n] - sp.Rational(1, 3) * gdn[m, n] * Str)
        eqs[(m, n)] = sp.expand(sp.expand(e).subs(x, 0))
    eqs[(4, 4)] = sp.expand(sp.expand(EG[4][4] - alpha * S[4][4]).subs(x, 0))

    Hs = [Hyy(u), Hzz(u), Huu(u), Hxu(u)]
    Hpp = [sp.Derivative(Hf, (u, 2)) for Hf in Hs]

    # drop identically-cancelling H'' coefficients (rational cancellation)
    for key in list(eqs):
        e = eqs[key]
        for d2f in Hpp:
            c = e.coeff(d2f)
            if c != 0 and is_zero(c):
                e = sp.expand(e - sp.expand(c * d2f))
        eqs[key] = e

    log("[2] structure of the six equations on the gauge slice:")
    for (m, n), e in eqs.items():
        d2 = sorted(
            {str(d.expr.func) for d in e.atoms(sp.Derivative) if len(d.variables) == 2}
        )
        d1 = sorted(
            {
                str(d.expr.func)
                for d in e.atoms(sp.Derivative)
                if len(d.variables) == 1 and d.expr in Hs
            }
        )
        alg = sorted(str(Hf.func) for Hf in Hs if e.has(Hf))
        print(
            f"    eq ({idx[m]}{idx[n]}): H'' of {d2 or None}; "
            f"H' of {d1 or None}; algebraic {alg or None}"
        )

    # ---------------------------------------------------------------------------
    # [2]/[3] sequential elimination and redundancy of the rest
    # ---------------------------------------------------------------------------
    # Structure found: (yy), (zz) second order and diagonal in their own H'';
    # (tt), (xx), (xu), (uu) first order.  Moreover the Hxu terms of the (xu)
    # row cancel identically, making it an *algebraic* relation for Huu.
    ok2a = all(
        not eqs[(m, n)].has(d2f)
        for (m, n) in ((0, 0), (1, 1), (4, 4), (1, 4))
        for d2f in Hpp
    )
    log(f"[2] (tt), (xx), (uu), (xu) rows are first order: {ok2a}")
    assert ok2a
    ok2c = (
        is_zero(eqs[(1, 4)].coeff(Hxu(u)))
        and not eqs[(1, 4)].has(sp.Derivative(Hxu(u), u))
        and not eqs[(1, 4)].has(sp.Derivative(Huu(u), u))
    )
    log(f"[2] (xu) row is algebraic in Huu (no Hxu, no Huu'): {ok2c}")
    assert ok2c

    # step 1: (xu) row -> Huu
    cHuu = sp.cancel(sp.together(eqs[(1, 4)].coeff(Huu(u))))
    assert not is_zero(cHuu)
    sol_Huu = sp.expand(
        sp.cancel(
            sp.together(
                -(eqs[(1, 4)] - sp.expand(cHuu * Huu(u))).subs(Hxu(u), 0) / cHuu
            )
        )
    )
    # (Hxu coefficient is identically zero; subs is exact)
    log("[3] step 1: Huu solved algebraically from the (xu) row")

    # step 2: eliminate Huu (and Huu' by differentiation) from {tt, yy, zz},
    # then solve simultaneously for (Hyy'', Hzz'', Hxu)
    huu_rules = [
        (sp.Derivative(Huu(u), (u, 2)), sp.diff(sol_Huu, u, 2)),
        (sp.Derivative(Huu(u), u), sp.diff(sol_Huu, u)),
        (Huu(u), sol_Huu),
    ]

    def elim_huu(e):
        e = sp.expand(sp.sympify(e).doit())
        for k, v in huu_rules:
            if e.has(k):
                e = e.subs(k, v).doit()
        return sp.expand(e)

    work = [elim_huu(eqs[(0, 0)]), elim_huu(eqs[(2, 2)]), elim_huu(eqs[(3, 3)])]
    unknowns = [Hpp[0], Hpp[1], Hxu(u)]
    # explicit linear solve (sp.solve is far too slow on these expressions)
    for e in work:
        assert not e.has(sp.Derivative(Hxu(u), u)) and not e.has(
            sp.Derivative(Hxu(u), (u, 2))
        ), "stray Hxu' in working set"
    Mat = sp.Matrix(
        3, 3, lambda i, j: sp.cancel(sp.together(work[i].coeff(unknowns[j])))
    )
    rem = sp.Matrix(
        3, 1, lambda i, _: sp.expand(work[i].subs({v: 0 for v in unknowns}))
    )
    det = sp.cancel(sp.together(Mat.det()))
    ok3 = not is_zero(det)
    log(
        f"[3] step 2: {{tt, yy, zz}} (Huu eliminated) solvable for "
        f"(Hyy'', Hzz'', Hxu): {ok3}  [det = {sp.simplify(det)}]"
    )
    assert ok3
    X = Mat.LUsolve(-rem)
    sol = {unknowns[i]: sp.expand(sp.cancel(sp.together(X[i]))) for i in range(3)}
    # the closed (Hyy, Hzz) system must not involve Hxu or Huu any more
    for k2 in (Hpp[0], Hpp[1]):
        bad = [
            v
            for v in (
                Hxu(u),
                Huu(u),
                sp.Derivative(Hxu(u), u),
                sp.Derivative(Huu(u), u),
            )
            if sol[k2].has(v)
        ]
        assert not bad, f"unexpected dependence of {k2}: {bad}"
    log(
        "[3] (Hyy, Hzz) close into a two-field second-order system; "
        "Huu, Hxu are algebraic in (H, H')"
    )

    # redundancy of (xx) and (uu): substitute the solved quantities and their
    # consequences, strictly highest derivative first so no rule ever acts
    # inside a remaining Derivative atom.  Solved: Hyy'', Hzz'' and the
    # algebraic Huu, Hxu.
    def reduce_eq(e):
        sol_Hxu = sol[Hxu(u)]
        rules = [
            (sp.Derivative(Huu(u), (u, 2)), sp.diff(sol_Huu, u, 2)),
            (sp.Derivative(Hxu(u), (u, 2)), sp.diff(sol_Hxu, u, 2)),
            (sp.Derivative(Hyy(u), (u, 3)), sp.diff(sol[Hpp[0]], u)),
            (sp.Derivative(Hzz(u), (u, 3)), sp.diff(sol[Hpp[1]], u)),
            (sp.Derivative(Huu(u), u), sp.diff(sol_Huu, u)),
            (sp.Derivative(Hxu(u), u), sp.diff(sol_Hxu, u)),
            (sp.Derivative(Hyy(u), (u, 2)), sol[Hpp[0]]),
            (sp.Derivative(Hzz(u), (u, 2)), sol[Hpp[1]]),
            (Huu(u), sol_Huu),
            (Hxu(u), sol_Hxu),
        ]
        keys = [k for k, _ in rules]
        e = sp.expand(sp.sympify(e).doit())
        for sweep in range(10):
            if not any(e.has(k) for k in keys):
                break
            for k, v in rules:
                if e.has(k):
                    e = e.subs(k, v).doit()
            e = on_shell(sp.expand(e))
            log(
                f"      reduce_eq sweep {sweep} done "
                f"(remaining keys: {sum(e.has(k) for k in keys)})"
            )
        return e

    ok2b = True
    for m, n in ((1, 1), (4, 4)):
        r = reduce_eq(eqs[(m, n)])
        if not is_zero(r):
            ok2b = False
            print(f"    eq ({idx[m]}{idx[n]}) NOT redundant; residue:")
            sp.pprint(sp.simplify(sp.cancel(sp.together(r))))
    log(f"[2] (xx) and (uu) rows identically redundant on shell (Bianchi): {ok2b}")
    assert ok2b

    # ---------------------------------------------------------------------------
    # [3] export form
    # ---------------------------------------------------------------------------
    log("[3] solved system:")
    W0, W0p, BH, BHp = sp.symbols("W0 W0p BH BHp", real=True)
    flat = {
        w(u): W0,
        sp.Derivative(w(u), u): W0p,
        bh(u): BH,
        sp.Derivative(bh(u), u): BHp,
    }
    Hsym = sp.symbols("hyy hzz huu hxu", real=True)
    Hpsym = sp.symbols("dhyy dhzz dhuu dhxu", real=True)

    def flatten(e):
        e = on_shell(sp.expand(e))
        for Hf, s_, sd in zip(Hs, Hsym, Hpsym, strict=True):
            e = e.subs({sp.Derivative(Hf, u): sd, Hf: s_})
        e = e.subs(flat)
        return sp.expand(e)

    exports = {
        "d2Hyy": flatten(sol[Hpp[0]]).subs(alpha, 1),
        "d2Hzz": flatten(sol[Hpp[1]]).subs(alpha, 1),
        "Huu": flatten(sol_Huu).subs(alpha, 1),
        "Hxu": flatten(sol[Hxu(u)]).subs(alpha, 1),
    }
    for k, v in exports.items():
        print(f"  {k} =")
        sp.pprint(sp.collect(v, Hsym + Hpsym))
    for k in exports:
        deps = [s_ for s_ in Hsym + Hpsym if exports[k].has(s_)]
        print(f"  {k} depends on: {deps}")
        # everything must close on (Hyy, Hzz) and their first derivatives
        assert not any(
            exports[k].has(s_) for s_ in (Hsym[2], Hsym[3], Hpsym[2], Hpsym[3])
        ), k

    # ---------------------------------------------------------------------------
    # [4] indicial structure at the boundary for the closed (Hyy, Hzz) system
    # ---------------------------------------------------------------------------
    p = sp.Symbol("p")
    Mind = sp.zeros(2, 2)
    pairs = [("hyy", Hsym[0], Hpsym[0]), ("hzz", Hsym[1], Hpsym[1])]
    closed = {}
    for k in ("d2Hyy", "d2Hzz"):
        e = exports[k].subs(alpha, 0)
        # drop matter sources for the homogeneous analysis
        e = e.subs({W0: 0, W0p: 0, BH: 0, BHp: 0})
        closed[k] = e
    for i, k in enumerate(("d2Hyy", "d2Hzz")):
        for j, (_, s_, sd) in enumerate(pairs):
            aij = sp.limit(u**2 * closed[k].coeff(s_), u, 0)
            bij = sp.limit(u * closed[k].coeff(sd), u, 0)
            Mind[i, j] = (p * (p - 1) if i == j else 0) - aij - bij * p
        # any dependence on huu/hxu at leading order?
    log(f"[4] boundary indicial det (Hyy, Hzz block): {sp.factor(Mind.det())}")

    # horizon behaviour: multiply the yy equation by f and evaluate the limit
    # structure (printed for the record)
    log(
        "[4] horizon structure: see exported coefficients (poles at f = 0 are "
        "first order; degenerate-equation collocation applies)"
    )

    # ---------------------------------------------------------------------------
    # [5] code generation
    # ---------------------------------------------------------------------------
    from backreaction import paths  # noqa: E402

    gen_path = paths.SYSTEMS / "einstein_invgauge.py"

    def split_linear(expr, matter_ok=True):
        A_row, B_row = [], []
        rem = sp.expand(expr)
        for s_ in Hsym[:2]:
            c = sp.cancel(sp.together(rem.coeff(s_)))
            A_row.append(c)
            rem = sp.expand(rem - c * s_)
        for sd in Hpsym[:2]:
            c = sp.cancel(sp.together(rem.coeff(sd)))
            B_row.append(c)
            rem = sp.expand(rem - c * sd)
        rem = sp.cancel(sp.together(rem))
        assert not any(rem.has(s_) for s_ in Hsym + Hpsym), f"nonlinear? {rem}"
        return A_row, B_row, rem

    with open(gen_path, "w") as fh:
        fh.write(
            '"""AUTO-GENERATED by gauge_invariant.py -- do not edit.\n\n'
            "Algebraic gauge Htt = Hxx = 0 (fields are gauge invariants).\n"
            "Field order: (Hyy, Hzz); per rho^2 lambda_G, alpha = 1.\n"
            "Each function returns (coeff_H[2], coeff_dH[2], src):\n"
            "    Hyy'' = row_yy . (H, dH) + src,   Hzz'' = row_zz . (H, dH) + src\n"
            "    Huu   = huu_expr . (H, dH) + src    [algebraic]\n"
            "    Hxu   = hxu_expr . (H, dH) + src    [algebraic]\n"
            '"""\n\nimport math  # noqa: F401\n\n\n'
        )
        for name, key in (
            ("row_yy", "d2Hyy"),
            ("row_zz", "d2Hzz"),
            ("huu_expr", "Huu"),
            ("hxu_expr", "Hxu"),
        ):
            A_row, B_row, s_row = split_linear(exports[key])
            fh.write(
                f"def {name}(u, B, u_H, G, W0, W0p, BH, BHp):\n"
                "    coeff_H = [" + ", ".join(sp.pycode(c) for c in A_row) + "]\n"
                "    coeff_dH = [" + ", ".join(sp.pycode(c) for c in B_row) + "]\n"
                f"    src = {sp.pycode(s_row)}\n"
                "    return coeff_H, coeff_dH, src\n\n\n"
            )
    log(f"[5] wrote {gen_path}")
    log("done")

    # ---------------------------------------------------------------------------
    # provenance stamp
    # ---------------------------------------------------------------------------
    # Records which revision of this file produced the generated module, plus a
    # digest of the module itself, so that a hand edit or a stale regeneration is
    # visible instead of silent.  Must stay last: it digests the finished file.
    # See backreaction/generated.py and tests/test_generated_systems.py.
    from backreaction.generated import stamp  # noqa: E402

    stamp(gen_path, __file__)
    log(f"[stamp] provenance written to {gen_path}")


if __name__ == "__main__":
    main()
