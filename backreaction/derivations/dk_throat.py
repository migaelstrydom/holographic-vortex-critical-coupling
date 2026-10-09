"""The AdS3 x R^2 throat of the magnetic brane, and the critical coupling
alpha_* = 2/3.

Setting.  The condensate-free background at finite backreaction is the
D'Hoker-Kraus magnetic brane, the one-parameter family labelled by
beta = alpha B^2 solved numerically by `dk_background.py` in the fixed-T frame
(T = 1/pi, horizon at r_p = 1).  beta -> oo at fixed alpha is the T -> 0 limit,
where the brane develops a long throat interpolating between AdS5 at the
boundary and the exact AdS3 x R^2 fixed point in the deep interior.  This
module derives that fixed point in OUR conventions, puts the paper sec. 3 zero-mode
operator on it, and reads off the onset criterion there.

Conventions.  Identical to `dk_background.py`:

    R_MN + 4 g_MN = alpha (F_MP F_N^P - (1/6) F^2 g_MN),   L = 1,   F_xy = B,
    ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2 + dy^2) + e^{2W} dz^2.

Zero-mode operator (paper sec. 3, derivations/dk_zeromode.py), B the eigenvalue:

    (P w')' + Q w = 0,   P = U e^W,   Q = B e^{W - 2V}.

Results
-------
[1] The fixed point.  U = (r-r0)^2/l3^2, e^{2W} = c_W (r-r0)^2, e^{2V} = v
    (constant) solves the system iff

        l3^2 = 1/3        (AdS3 radius L/sqrt3 -- D'Hoker-Kraus)
        v    = B sqrt(alpha) / sqrt6,

    with c_W and r0 free (z-rescaling and radial translation).  Equivalently,
    in the two scaling-INVARIANT combinations (x,y-rescaling shifts V and B
    together, so neither v nor B alone is frame data):

        alpha B^2 e^{-4V} = 6,        B e^{-2V} = sqrt(6/alpha).

    The BTZ generalisation U = ((r-r0)^2 - h^2)/l3^2 (finite throat
    temperature) solves the same equations with the same l3, v: the throat's
    own temperature does not move the fixed point.

[2] The onset in the throat.  On the fixed point the SL operator has the exact
    Euler form (r^3 w')' + b r w = 0 with the invariant

        b == B l3^2 e^{-2V} = sqrt(2 / (3 alpha))          (B cancels!)

    and indicial exponents w ~ (r-r0)^s,

        s(s + 2) + b = 0,      s = -1 +- sqrt(1 - b).

    In AdS3 language s = -Delta with Delta(Delta - 2) = m^2 l3^2, so the
    condensate's effective AdS3 mass is m^2 l3^2 = -b, and the AdS3
    Breitenlohner-Freedman bound m^2 l3^2 >= -1 is violated iff b > 1:

        UNSTABLE AT T = 0   <=>   alpha < alpha_* = 2/3.

    The physical field B drops out identically: the flux plane freezes at
    exactly the proper size that holds the magnetic-moment coupling fixed, so
    the T = 0 criterion is a statement about the coupling alone.

[3] Near-critical scaling (WKB / matched asymptotics).  For alpha slightly
    below alpha_* the throat exponents are complex, s = -1 +- i nu with

        nu = sqrt(b - 1),      b = sqrt(2/(3 alpha)),

    and the marginal mode needs one half-period of the log-oscillation to fit
    between the IR (horizon) and UV (AdS5 crossover) matching phases.  The
    throat spans ln(r_c/r_p) e-folds with r_c ~ (beta/6)^{1/4} fixed by
    e^{2V} ~ r^2 meeting the frozen v = sqrt(beta/6), so

        (nu/4) ln beta_c  ~  pi + const   =>   ln beta_c ~ 4 pi / nu,

    and, expanding nu near alpha_*,

        alpha_* - alpha  ~  32 pi^2 alpha_* / (ln beta + d)^2
                         =  (64 pi^2 / 3) / (ln beta + d)^2.

    The same counting gives the excited tower at fixed beta: the n-th SL
    eigenvalue B_n satisfies nu_n (1/4) ln(beta/6) ~ n pi + const with
    nu_n = sqrt(B_n l3^2 e^{-2V} - 1).  Both are tested numerically in
    `dk_throat.py`.

CHECKS (all asserted)
---------------------
 [1] the fixed point annihilates the FULL Einstein tensor (all 25 components)
     and the Maxwell equation, not just the diagonal reduced system;
 [2] l3^2 = 1/3, i.e. L_3 = L/sqrt3 -- the D'Hoker-Kraus value;
 [3] it is an exact fixed point of the PRODUCTION reduced ODE system
     (dk_background._build_symbolics), the same equations the numerics solve;
 [4] frame invariance: under (x,y) -> lambda (x,y) both v and B move but
     alpha B^2 e^{-4V} and B e^{-2V} do not;
 [5] the BTZ (finite throat temperature) generalisation solves the same
     equations with unchanged l3, v;
 [6] the indicial equation from the zero-mode operator, and the collision of
     its roots at s = -1 exactly at alpha = 2/3;
 [7] alpha -> 0 gives b -> oo (always unstable): the probe limit condenses at
     T = 0 for any field, as in the probe limit;
 [8] the AdS3 dictionary Delta(Delta-2) = m^2 l3^2, derived from the
     Klein-Gordon operator on the AdS3 factor, matches the zero-mode indicial
     polynomial with m^2 = -B e^{-2V} (and not with the opposite sign), and
     reproduces the same criterion via the standard BF bound.
[12] the BTZ horizon phase of the paper sec. 4.6 matching condition,
     chi(nu) = arg Gamma(-i nu) - 2 arg Gamma((1 - i nu)/2), has the odd
     expansion chi = pi/2 - 2 nu ln 2 + zeta(3) nu^3/4 + O(nu^5): its O(nu)
     correction only shifts ln beta_c by a constant, so chi contributes its
     full pi/2 to the BKT slope already on the ladder (paper sec. 4.6).  Series
     from SymPy, cross-checked against mpmath at nu = 0.05, 0.25, 0.6.

Run:  uv run python -m backreaction.derivations.dk_throat   (~1 min)
"""

import time

import mpmath
import sympy as sp

T0_START = time.time()


def log(msg):
    print(f"[{time.time() - T0_START:7.1f}s] {msg}", flush=True)


def is_zero(e):
    e = sp.expand(e)
    if e == 0:
        return True
    return sp.simplify(sp.cancel(sp.together(e))) == 0


CHECKS = {}


def check(name, ok):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# ===========================================================================
# geometry helpers
# ===========================================================================
def christoffel(g, xs):
    n = len(xs)
    gi = g.inv()
    return [
        [
            [
                sp.simplify(
                    sum(
                        gi[l, s]
                        * (
                            sp.diff(g[s, m], xs[k])
                            + sp.diff(g[s, k], xs[m])
                            - sp.diff(g[m, k], xs[s])
                        )
                        for s in range(n)
                    )
                    / 2
                )
                for k in range(n)
            ]
            for m in range(n)
        ]
        for l in range(n)
    ]


def ricci(g, xs):
    n = len(xs)
    Gam = christoffel(g, xs)
    R = sp.zeros(n, n)
    for m in range(n):
        for k in range(n):
            e = 0
            for l in range(n):
                e += sp.diff(Gam[l][m][k], xs[l]) - sp.diff(Gam[l][m][l], xs[k])
                for s in range(n):
                    e += Gam[l][l][s] * Gam[s][m][k] - Gam[l][k][s] * Gam[s][m][l]
            R[m, k] = sp.simplify(e)
    return R, Gam


def einstein_maxwell_residual(g, xs, F, alpha):
    """R_MN + 4 g_MN - alpha (F_MP F_N^P - F^2 g_MN / 6), and the Maxwell
    equation nabla_M F^{MN} (for a covariantly constant-flux ansatz)."""
    n = len(xs)
    gi = g.inv()
    R, Gam = ricci(g, xs)
    F2 = sp.simplify(
        sum(
            F[m, p] * gi[m, a] * gi[p, b] * F[a, b]
            for m in range(n)
            for p in range(n)
            for a in range(n)
            for b in range(n)
        )
    )
    FF = sp.Matrix(
        n,
        n,
        lambda m, k: sp.simplify(
            sum(F[m, p] * gi[p, s] * F[k, s] for p in range(n) for s in range(n))
        ),
    )
    EOM = sp.Matrix(
        n,
        n,
        lambda m, k: sp.simplify(
            R[m, k] + 4 * g[m, k] - alpha * (FF[m, k] - F2 * g[m, k] / 6)
        ),
    )
    # Maxwell: (1/sqrt(-g)) d_M (sqrt(-g) F^{MN})
    detg = sp.simplify(g.det())
    sq = sp.sqrt(-detg)
    Fup = sp.Matrix(
        n,
        n,
        lambda m, k: sp.simplify(
            sum(gi[m, a] * gi[k, b] * F[a, b] for a in range(n) for b in range(n))
        ),
    )
    maxw = [
        sp.simplify(sum(sp.diff(sq * Fup[m, k], xs[m]) for m in range(n)) / sq)
        for k in range(n)
    ]
    return EOM, maxw, F2


def main():
    # ===========================================================================
    # [1] the AdS3 x R^2 fixed point
    # ===========================================================================
    log("[1] solving for the AdS3 x R^2 fixed point")

    t, r, x, y, z = sp.symbols("t r x y z")
    B, alpha, l3, v, cW, r0, h = sp.symbols("B alpha l3 v c_W r_0 h", positive=True)

    rho = r - r0  # radial translation freedom
    U_fp = rho**2 / l3**2
    g_fp = sp.diag(-U_fp, 1 / U_fp, v, v, cW * rho**2)
    XS = [t, r, x, y, z]

    F = sp.zeros(5, 5)
    F[2, 3] = B
    F[3, 2] = -B

    EOM, maxw, F2 = einstein_maxwell_residual(g_fp, XS, F, alpha)
    log(f"      F^2 = {F2}")

    diag = [sp.simplify(EOM[i, i]) for i in range(5)]
    sol = sp.solve(diag, [l3, v], dict=True)
    log(f"      solutions: {sol}")
    assert len(sol) == 1, sol
    L3, V0 = sol[0][l3], sol[0][v]
    log(f"      l3 = {L3},  v = e^(2V) = {sp.simplify(V0)}")

    sub_fp = {l3: L3, v: V0}
    EOM_fp = sp.Matrix(5, 5, lambda i, j: sp.simplify(EOM[i, j].subs(sub_fp)))
    check(
        "[1] fixed point annihilates all 25 Einstein-equation components",
        all(is_zero(EOM_fp[i, j]) for i in range(5) for j in range(5)),
    )
    check(
        "[1] fixed point solves the Maxwell equation",
        all(is_zero(sp.simplify(m.subs(sub_fp))) for m in maxw),
    )

    check(
        "[2] AdS3 radius l3^2 = 1/3  (L_3 = L/sqrt3, D'Hoker-Kraus)",
        sp.simplify(L3**2 - sp.Rational(1, 3)) == 0,
    )

    # the two frame-invariant statements
    inv_beta = sp.simplify(alpha * B**2 / V0**2)  # alpha B^2 e^{-4V}
    inv_flux = sp.simplify(B / V0)  # B e^{-2V}
    log(f"      alpha B^2 e^(-4V) = {inv_beta};  B e^(-2V) = {sp.simplify(inv_flux)}")
    check("[1] invariant  alpha B^2 e^{-4V} = 6", sp.simplify(inv_beta - 6) == 0)
    check(
        "[1] invariant  B e^{-2V} = sqrt(6/alpha)",
        sp.simplify(inv_flux - sp.sqrt(6 / alpha)) == 0,
    )

    # ---- [4] frame invariance under (x,y) -> lambda (x,y) ---------------------
    # The metric piece e^{2V}(dx^2 + dy^2) is form-invariant if e^{2V} -> e^{2V} /
    # lambda^2; the gauge field A = B x dy is a fixed one-form, so in the new
    # coordinates A = (B/lambda^2) x' dy', i.e. B -> B / lambda^2.  Neither v nor B
    # is frame data on its own; the two combinations below are.  Checked on
    # INDEPENDENT symbols (v, B), then confirmed to be consistent with the solved
    # fixed-point relation v = B sqrt(alpha)/sqrt6, which is itself covariant.
    lam = sp.symbols("lambda", positive=True)
    scal = {B: B / lam**2, v: v / lam**2}
    check(
        "[4] alpha B^2 e^{-4V} invariant under (x,y)-rescaling",
        sp.simplify((alpha * B**2 / v**2).subs(scal) - alpha * B**2 / v**2) == 0,
    )
    check(
        "[4] B e^{-2V} invariant under (x,y)-rescaling",
        sp.simplify((B / v).subs(scal) - B / v) == 0,
    )
    check(
        "[4] the fixed-point relation v = B sqrt(alpha/6) is covariant",
        sp.simplify(
            (v - B * sp.sqrt(alpha) / sp.sqrt(6)).subs(scal) * lam**2
            - (v - B * sp.sqrt(alpha) / sp.sqrt(6))
        )
        == 0,
    )

    # ---- [5] BTZ generalisation (finite throat temperature) -------------------
    log("[5] BTZ throat: U = ((r-r0)^2 - h^2)/l3^2")
    U_btz = (rho**2 - h**2) / l3**2
    g_btz = sp.diag(-U_btz, 1 / U_btz, v, v, cW * rho**2)
    EOM_b, maxw_b, _ = einstein_maxwell_residual(g_btz, XS, F, alpha)
    EOM_b = sp.Matrix(5, 5, lambda i, j: sp.simplify(EOM_b[i, j].subs(sub_fp)))
    check(
        "[5] BTZ throat solves the same equations with the same l3, v",
        all(is_zero(EOM_b[i, j]) for i in range(5) for j in range(5))
        and all(is_zero(sp.simplify(m.subs(sub_fp))) for m in maxw_b),
    )

    # ---- [3] anchor to the PRODUCTION reduced ODE system ----------------------
    log("[3] anchoring against dk_background._build_symbolics")
    from backreaction.numerics import dk_background as dk  # noqa: E402

    S = dk._SYM["_sym"]
    rr_p, Uv, Vv, Wv, Up, Vp, Wp, Bs, als, Upp, Vpp, Wpp = S["symbols"]
    Wfp = sp.log(sp.sqrt(cW) * rho)
    Vfp = sp.log(sp.sqrt(V0))
    prod_sub = {
        rr_p: r,
        Uv: U_fp.subs(sub_fp),
        Vv: Vfp,
        Wv: Wfp,
        Up: sp.diff(U_fp.subs(sub_fp), r),
        Vp: sp.diff(Vfp, r),
        Wp: sp.diff(Wfp, r),
        Bs: B,
        als: alpha,
    }
    resid = []
    for sym, expr in (
        (Upp, sp.diff(U_fp.subs(sub_fp), r, 2)),
        (Vpp, sp.diff(Vfp, r, 2)),
        (Wpp, sp.diff(Wfp, r, 2)),
    ):
        resid.append(sp.simplify(S[str(sym)].subs(prod_sub) - expr))
    con_resid = sp.simplify(S["constraint"].subs(prod_sub))
    check(
        "[3] fixed point solves the production reduced ODEs (U'',V'',W'')",
        all(is_zero(e) for e in resid),
    )
    check(
        "[3] fixed point satisfies the production Hamiltonian constraint",
        is_zero(con_resid),
    )

    # ===========================================================================
    # [6] the zero-mode operator on the throat: indicial equation and BF bound
    # ===========================================================================
    log("[6] the paper sec. 3 zero-mode operator on the fixed point")

    w = sp.Function("w")
    P_sl = sp.simplify(U_fp * sp.exp(Wfp))  # P = U e^W
    Q_sl = sp.simplify(B * sp.exp(Wfp - 2 * Vfp))  # Q = B e^{W-2V}
    P_sl = P_sl.subs(sub_fp)
    Q_sl = Q_sl.subs(sub_fp)
    log(f"      P = {P_sl},  Q = {sp.simplify(Q_sl)}")

    ode = sp.expand(sp.simplify(sp.diff(P_sl * sp.diff(w(r), r), r) + Q_sl * w(r)))
    s = sp.symbols("s")
    ind = sp.simplify(
        sp.expand(ode.subs(w(r), rho**s).doit()) / (sp.sqrt(cW) * rho ** (s + 1))
    )
    ind = sp.simplify(sp.expand(ind))
    log(f"      indicial polynomial (stripped): {sp.factor(ind)}")

    b_inv = sp.symbols("b", positive=True)  # b = B l3^2 e^{-2V}
    b_val = sp.simplify(B * L3**2 / V0)
    log(f"      b = B l3^2 e^(-2V) = {b_val}")
    check(
        "[6] b = sqrt(2/(3 alpha)) -- B cancels identically",
        sp.simplify(b_val - sp.sqrt(2 / (3 * alpha))) == 0,
    )

    ratio = sp.simplify(sp.expand(ind) / sp.expand(s * (s + 2) + b_val))
    check(
        "[6] indicial equation is s(s+2) + b = 0 (up to a positive factor)",
        ratio.free_symbols.isdisjoint({s}) and sp.simplify(ratio) != 0,
    )

    roots = sp.solve(sp.Eq(s * (s + 2) + b_inv, 0), s)
    log(f"      s = {roots}")
    check(
        "[6] roots s = -1 +- sqrt(1 - b)",
        set(sp.simplify(rt) for rt in roots)
        == {sp.simplify(-1 + sp.sqrt(1 - b_inv)), sp.simplify(-1 - sp.sqrt(1 - b_inv))},
    )

    alpha_star = sp.solve(sp.Eq(b_val, 1), alpha)
    log(f"      b = 1  =>  alpha = {alpha_star}")
    check(
        "[6] critical coupling alpha_* = 2/3",
        len(alpha_star) == 1 and sp.simplify(alpha_star[0] - sp.Rational(2, 3)) == 0,
    )
    check(
        "[6] roots collide at s = -1 exactly at alpha = alpha_*",
        all(
            sp.simplify(rt.subs(b_inv, b_val.subs(alpha, sp.Rational(2, 3))) + 1) == 0
            for rt in roots
        ),
    )

    # ---- [8] the AdS3 dictionary route ---------------------------------------
    # A scalar of mass m in AdS3 has Delta(Delta - 2) = m^2 l3^2 and the BF bound
    # m^2 l3^2 >= -1.  Writing w ~ rho^{-Delta} in the throat identifies
    # m^2 l3^2 = -b, so BF violation <=> b > 1 <=> alpha < 2/3.
    Delta = sp.symbols("Delta")
    m2l2 = sp.symbols("m2l2")
    bf = sp.solve(sp.Eq(Delta * (Delta - 2), m2l2), Delta)
    check(
        "[8] AdS3 dictionary: Delta = 1 +- sqrt(1 + m^2 l3^2), complex iff m^2 l3^2 < -1",
        set(sp.simplify(d) for d in bf)
        == {sp.simplify(1 + sp.sqrt(1 + m2l2)), sp.simplify(1 - sp.sqrt(1 + m2l2))},
    )
    # The identification m^2 l3^2 = -b is DERIVED, not assumed.  Build the
    # Klein-Gordon operator of a scalar of mass m on the AdS3 factor of the throat,
    # g3 = diag(-U, 1/U, e^{2W}) in (t, rho, z), straight from the metric:
    #     KG[phi] = (1/sqrt(-g3)) (sqrt(-g3) g3^{rr} phi')' - m^2 phi .
    # Its indicial polynomial at phi ~ rho^{-Delta} must be (i) the AdS3 relation
    # Delta(Delta - 2) - m^2 l3^2 (the dictionary itself, derived here rather than
    # quoted), and (ii) proportional to the zero-mode indicial polynomial of [6] at
    # s = -Delta exactly when m^2 = -B e^{-2V}, i.e. m^2 l3^2 = -b.  Negative
    # control: the opposite sign m^2 = +B e^{-2V} must NOT reproduce [6].
    m2 = sp.symbols("m2", real=True)
    phi = sp.Function("phi")
    U3, W3 = U_fp.subs(sub_fp), Wfp
    sqrtg3 = sp.sqrt(U3 * (1 / U3) * sp.exp(2 * W3))  # sqrt(-g3) = e^W
    kg = sp.diff(sqrtg3 * U3 * sp.diff(phi(r), r), r) / sqrtg3 - m2 * phi(r)
    kg_ind = sp.simplify(
        sp.expand(kg.subs(phi(r), rho ** (-Delta)).doit()) / rho ** (-Delta)
    )
    log(f"      AdS3 Klein-Gordon indicial polynomial: {sp.factor(kg_ind)}")
    check(
        "[8] AdS3 Klein-Gordon on the throat gives Delta(Delta-2) = m^2 l3^2 "
        "(derived from the metric)",
        is_zero(kg_ind - (Delta * (Delta - 2) - m2 * L3**2) / L3**2),
    )
    ind_D = sp.simplify(ind.subs(s, -Delta))

    def _prop(a, b_):
        """a / b_ is a nonzero constant (free of Delta)."""
        q = sp.simplify(sp.cancel(a / b_))
        return q != 0 and Delta not in q.free_symbols

    m2_derived = sp.simplify(-Q_sl / P_sl * U3)  # -B e^{-2V} from the operator
    check(
        "[8] zero-mode indicial polynomial at s = -Delta = KG polynomial with "
        "m^2 = -B e^{-2V}, i.e. m^2 l3^2 = -b",
        is_zero(m2_derived * L3**2 + b_val)
        and _prop(ind_D, kg_ind.subs(m2, m2_derived)),
    )
    check(
        "[8] negative control: m^2 = +B e^{-2V} does NOT reproduce the zero-mode "
        "indicial polynomial",
        not _prop(ind_D, kg_ind.subs(m2, -m2_derived)),
    )
    check(
        "[8] BF: the roots are complex iff the discriminant 4(1 + m^2 l3^2) < 0, "
        "i.e. b > 1 <=> alpha < 2/3",
        is_zero(sp.discriminant(Delta * (Delta - 2) - m2l2, Delta) - 4 * (1 + m2l2))
        and sp.solve(sp.Eq(-b_val, -1), alpha) == [sp.Rational(2, 3)],
    )

    # ---- [7] probe limit ------------------------------------------------------
    check(
        "[7] alpha -> 0: b -> oo (probe always condenses at T = 0)",
        sp.limit(b_val, alpha, 0, "+") == sp.oo,
    )

    # ===========================================================================
    # [9] near-critical scaling constants (WKB counting; quoted, not proved here)
    # ===========================================================================
    log("[9] near-critical scaling")
    a_st = sp.Rational(2, 3)
    nu_near = sp.simplify(sp.sqrt(sp.series(b_val - 1, alpha, a_st, 2).removeO()))
    log(f"      nu = sqrt(b - 1);  near alpha_*:  nu ~ {sp.simplify(nu_near)}")
    c_bkt = sp.simplify(2 * a_st * (4 * sp.pi) ** 2)
    log(
        f"      alpha_* - alpha ~ C / (ln beta + d)^2  with  C = 32 pi^2 alpha_* "
        f"= {c_bkt} = {float(c_bkt):.4f}"
    )
    check(
        "[9] near-critical nu^2 = (alpha_* - alpha)/(2 alpha_*) to leading order",
        sp.simplify(
            sp.limit((b_val - 1) / ((a_st - alpha) / (2 * a_st)), alpha, a_st) - 1
        )
        == 0,
    )

    # ===========================================================================
    # [10] general d: the LLL reduction on an AdS_{d+1} diagonal brane
    # ===========================================================================
    log("[10] general-d polarised-LLL reduction")

    def lll_reduce(d):
        """Repeat the paper sec. 3 (derivations/dk_zeromode.py) reduction in d+1 bulk
        dimensions: coordinates (t, r, x, y, z_1..z_{d-3}), diagonal metric with
        g_xx = g_yy = e^{2V} (flux plane) and g_{z_i z_i} = e^{2W}.  Returns the
        SL pair (P, Q) of (P w')' + Q w = 0."""
        nz = d - 3
        tt, rr = sp.symbols("t_ r_", real=True)
        xx, yy = sp.symbols("x_ y_", real=True)
        zz = list(sp.symbols(f"zz0:{max(nz, 1)}", real=True))[:nz]
        cs = [tt, rr, xx, yy] + zz
        n = len(cs)
        Uf = sp.Function("Uf")(rr)
        Vf = sp.Function("Vf")(rr)
        Wf = sp.Function("Wf")(rr)
        gdn_ = sp.diag(
            *([-Uf, 1 / Uf, sp.exp(2 * Vf), sp.exp(2 * Vf)] + [sp.exp(2 * Wf)] * nz)
        )
        gup_ = gdn_.inv()
        sq = sp.sqrt(sp.simplify(-gdn_.det()))
        wf = sp.Function("wf", real=True)
        ps = sp.exp(-B * xx**2 / 2)
        Am = [[sp.S.Zero] * n for _ in range(3)]
        Am[0][2] = wf(rr) * ps  # A^1_x
        Am[1][3] = -wf(rr) * ps  # A^2_y  (W_y = -i W_x)
        Am[2][3] = B * xx  # A^3_y = B x
        Fs = [[[sp.S.Zero] * n for _ in range(n)] for _ in range(3)]
        for ai in range(3):
            for m in range(n):
                for k in range(n):
                    e = sp.diff(Am[ai][k], cs[m]) - sp.diff(Am[ai][m], cs[k])
                    for bi in range(3):
                        for ci in range(3):
                            lc = sp.LeviCivita(ai + 1, bi + 1, ci + 1)
                            if lc != 0:
                                e += lc * Am[bi][m] * Am[ci][k]
                    Fs[ai][m][k] = sp.expand(e)
        Lg = sp.S.Zero
        for ai in range(3):
            for m in range(n):
                for k in range(n):
                    if Fs[ai][m][k] != 0:
                        Lg += (
                            -sp.Rational(1, 4)
                            * gup_[m, m]
                            * gup_[k, k]
                            * Fs[ai][m][k] ** 2
                        )
        Lg = sp.expand(sq * Lg)
        Wc, Wpc, ss = sp.symbols("Wc Wpc ss", real=True)
        wpr = sp.Derivative(wf(rr), rr)
        Lf = sp.expand(Lg.subs(wpr, Wpc).subs(wf(rr), Wc))
        Lf = sp.expand(Lf.subs({Wc: ss * Wc, Wpc: ss * Wpc}))
        assert is_zero(sp.expand(Lf.coeff(ss, 1))), f"d={d}: O(w) tadpole"
        L2_ = sp.expand(Lf.coeff(ss, 2)).subs({Wc: wf(rr), Wpc: wpr})
        L2_ = sp.simplify(L2_ / ps**2)
        assert is_zero(sp.diff(L2_, xx)), f"d={d}: residual x after LLL reduction"
        Ed = sp.expand(sp.cancel(sp.together(L2_)))
        # sqrt(exp(...)) from sqrt(-g) needs powdenest(force) to collapse: all the
        # metric functions are real exponentials, so the roots are unambiguous.
        den = lambda e: sp.simplify(sp.powdenest(sp.simplify(e), force=True))
        P_raw = sp.simplify(-Ed.coeff(wpr, 2))
        Q_raw = sp.simplify(Ed.coeff(wf(rr), 2))
        assert is_zero(sp.simplify(Ed - (-P_raw * wpr**2 + Q_raw * wf(rr) ** 2)))
        return den(P_raw), den(Q_raw), Uf, Vf, Wf, rr

    for dd in (3, 4, 5, 6):
        Pd, Qd, Uf, Vf, Wf, rr_ = lll_reduce(dd)
        Pref = Uf * sp.exp((dd - 3) * Wf)
        Qref = B * sp.exp((dd - 3) * Wf - 2 * Vf)
        check(
            f"[10] d={dd}: P = U e^{{(d-3)W}} and Q = B e^{{(d-3)W-2V}}",
            is_zero(sp.powdenest(sp.simplify(Pd - Pref), force=True))
            and is_zero(sp.powdenest(sp.simplify(Qd - Qref), force=True)),
        )

    # ===========================================================================
    # [11] general d: the throat fixed point and alpha_*(d) = 16/(d(d-1)(d-2))
    # ===========================================================================
    log("[11] general-d fixed point and the closed form for alpha_*(d)")

    # Product AdS_{d-1} x R^2: R_MN = -(p-1)/l^2 g_MN on the AdS_p factor (p=d-1),
    # R_MN = 0 on the flat flux plane; F^2 = 2B^2/v^2; F_xP F_x^P = B^2/v.
    dsym = sp.Symbol("d", positive=True)
    ls, vs = sp.symbols("l_s v_s", positive=True)
    F2g = 2 * B**2 / vs**2
    eq_ads = -(dsym - 2) / ls**2 + dsym + alpha * F2g / (2 * (dsym - 1))
    eq_pl = dsym * vs - alpha * (B**2 / vs - vs * F2g / (2 * (dsym - 1)))
    solg = sp.solve([eq_ads, eq_pl], [ls, vs], dict=True)
    solg = [s_ for s_ in solg if s_[ls].is_real is not False]
    assert solg, solg
    Lg_, Vg_ = sp.simplify(solg[0][ls]), sp.simplify(solg[0][vs])
    log(f"      l^2 = {sp.simplify(Lg_**2)},  v = {Vg_}")
    check(
        "[11] l^2 = (d-2)^2/(d(d-1))",
        sp.simplify(Lg_**2 - (dsym - 2) ** 2 / (dsym * (dsym - 1))) == 0,
    )
    check(
        "[11] v = B sqrt(alpha (d-2)/(d(d-1)))",
        sp.simplify(Vg_**2 - B**2 * alpha * (dsym - 2) / (dsym * (dsym - 1))) == 0
        and Vg_.is_positive is not False,
    )

    # SL indicial equation in general d: P ~ r^{d-1}, Q ~ B r^{d-3}/v gives
    #   s(s + d - 2) + B l^2 / v = 0,   BF bound of AdS_{d-1} is (d-2)^2/4.
    bg_ = sp.simplify(B * Lg_**2 / Vg_)
    log(f"      B l^2 / v = {bg_}")
    astar_d = sp.solve(sp.Eq(bg_, (dsym - 2) ** 2 / 4), alpha)
    log(f"      alpha_*(d) = {astar_d}")
    check(
        "[11] alpha_*(d) = 16 / (d (d-1) (d-2))",
        len(astar_d) == 1
        and sp.simplify(astar_d[0] - 16 / (dsym * (dsym - 1) * (dsym - 2))) == 0,
    )
    check(
        "[11] d = 4 reproduces alpha_* = 2/3",
        sp.simplify(astar_d[0].subs(dsym, 4) - sp.Rational(2, 3)) == 0,
    )

    # cross-check the symbolic-d product-space shortcut against explicit Ricci
    log("      cross-checking against explicit Ricci computation, d = 3..6")
    for dd in (3, 4, 5, 6):
        nz = dd - 3
        tt, rr = sp.symbols("t2 r2", real=True)
        zs = list(sp.symbols(f"z2_0:{max(nz, 1)}", real=True))[:nz]
        cs = [tt, rr] + list(sp.symbols("x2 y2", real=True)) + zs
        n = len(cs)
        lD, vD, cD = sp.symbols("lD vD cD", positive=True)
        UD = rr**2 / lD**2
        gD = sp.diag(*([-UD, 1 / UD, vD, vD] + [cD * rr**2] * nz))
        EOMd, maxd, F2d = einstein_maxwell_residual(
            gD,
            cs,
            sp.Matrix(
                n,
                n,
                lambda i, j: B if (i, j) == (2, 3) else (-B if (i, j) == (3, 2) else 0),
            ),
            alpha,
        )
        # replace the AdS_5 "+4 g" by the general "+d g".  The loop variables are
        # bound as default arguments rather than captured: sp.Matrix(n, n, f) calls
        # f immediately for every (i, j), so a closure would in fact be correct --
        # but E=EOMd captures the matrix being *reassigned* here by value, which is
        # what the expression means, and it keeps B023 from firing on the whole
        # construct.
        EOMd = sp.Matrix(
            n,
            n,
            lambda i, j, E=EOMd, g=gD, F2=F2d, d=dd: sp.simplify(
                E[i, j]
                - 4 * g[i, j]
                + d * g[i, j]
                - alpha * F2 * g[i, j] / 6
                + alpha * F2 * g[i, j] / (2 * (d - 1))
            ),
        )
        sold = sp.solve(
            [sp.simplify(EOMd[i, i]) for i in range(n)], [lD, vD], dict=True
        )
        # compare squares: sqrt(alpha/6) vs sqrt(alpha)/sqrt(6) are not collapsed
        # automatically even for positive alpha.
        okd = (
            bool(sold)
            and sp.simplify(sold[0][lD] ** 2 - (Lg_**2).subs(dsym, dd)) == 0
            and sp.simplify(sold[0][vD] ** 2 - (Vg_**2).subs(dsym, dd)) == 0
        )
        check(
            f"[11] explicit Ricci at d={dd} matches the general-d fixed point",
            bool(okd),
        )

    # ---- Wong (arXiv:1307.7839) anchor: d = 3 <=> gamma > 3/4 -----------------
    # Wong's AdS4 conventions: S = (1/2k^2) int (R + 6/L^2) - (1/4e^2) int F^2,
    # gamma = 2 e^2 L^2 / kappa^2.  Our alpha = kappa^2/g_hat^2 is the same ratio
    # with the opposite orientation: alpha = 2/gamma at L = 1.
    gam = sp.Symbol("gamma", positive=True)
    astar3 = astar_d[0].subs(dsym, 3)
    gam_thresh = sp.solve(sp.Eq(2 / gam, astar3), gam)
    log(f"      d = 3: alpha_* = {astar3}  <=>  gamma_* = {gam_thresh}")
    check(
        "[LIT] d = 3 reproduces Wong arXiv:1307.7839 threshold gamma_* = 3/4",
        len(gam_thresh) == 1 and sp.simplify(gam_thresh[0] - sp.Rational(3, 4)) == 0,
    )

    # ---- [12] the horizon phase chi(nu) of the paper sec. 4.6 matching ------------------
    # arg Gamma(-i nu) = arg Gamma(1 - i nu) - arg(-i nu) = arg Gamma(1 - i nu) + pi/2
    # for nu > 0, and arg Gamma(x - i nu) = Im loggamma(x - i nu) on the branch
    # continuous from nu = 0, so chi is a pair of loggamma series in nu.
    log("[12] horizon phase chi(nu) = arg G(-i nu) - 2 arg G((1 - i nu)/2)")
    nu_s = sp.Symbol("nu", positive=True)

    def _im_loggamma_series(z, order=6):
        s = sp.series(sp.loggamma(z), nu_s, 0, order).removeO()
        return sp.expand(sp.expand_func(sp.im(sp.expand(s))))

    chi_series = sp.expand(
        sp.expand_func(
            sp.pi / 2
            + _im_loggamma_series(1 - sp.I * nu_s)
            - 2 * _im_loggamma_series((1 - sp.I * nu_s) / 2)
        )
    )
    chi_claim = sp.pi / 2 - 2 * nu_s * sp.log(2) + sp.zeta(3) / 4 * nu_s**3
    log(f"      chi = {sp.collect(chi_series, nu_s)} + O(nu^7)")
    check(
        "[12] chi = pi/2 - 2 nu ln 2 + zeta(3) nu^3/4 + O(nu^5)",
        is_zero(sp.series(chi_series - chi_claim, nu_s, 0, 5).removeO()),
    )
    check(
        "[12] chi - pi/2 is odd in nu (no nu^2, nu^4 terms)",
        all(is_zero(chi_series.coeff(nu_s, k)) for k in (2, 4)),
    )
    _chi_exact = [
        float(
            mpmath.arg(mpmath.gamma(-1j * n))
            - 2 * mpmath.arg(mpmath.gamma((1 - 1j * n) / 2))
        )
        for n in (0.05, 0.25, 0.6)
    ]
    # the series converges for nu < 1 (loggamma((1 - i nu)/2) has its pole at
    # nu = -i); through O(nu^15) the truncation error at nu = 0.6 is ~1e-5
    chi_long = sp.expand(
        sp.expand_func(
            sp.pi / 2
            + _im_loggamma_series(1 - sp.I * nu_s, 16)
            - 2 * _im_loggamma_series((1 - sp.I * nu_s) / 2, 16)
        )
    )
    _chi_ser = [float(chi_long.subs(nu_s, n)) for n in (0.05, 0.25, 0.6)]
    _chi_dev = [abs(e - s) for e, s in zip(_chi_exact, _chi_ser, strict=True)]
    log(f"      exact {[round(c, 8) for c in _chi_exact]}")
    log(f"      |exact - series to nu^15| = {[f'{d:.1e}' for d in _chi_dev]}")
    check(
        "[12] series matches the exact arg-Gamma phase at nu = 0.05, 0.25, 0.6",
        _chi_dev[0] < 1e-12 and _chi_dev[1] < 1e-9 and _chi_dev[2] < 1e-4,
    )

    # ===========================================================================
    log("")
    log("SUMMARY")
    log("  AdS3 x R^2 fixed point:  l3^2 = 1/3,  alpha B^2 e^(-4V) = 6")
    log("  throat invariant      :  b = B l3^2 e^(-2V) = sqrt(2/(3 alpha))")
    log("  T = 0 instability     :  b > 1  <=>  alpha < alpha_* = 2/3")
    log(
        f"  near-critical         :  alpha_* - alpha ~ {float(c_bkt):.3f}/(ln beta + d)^2"
    )
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"  {len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    for k, ok in CHECKS.items():
        if not ok:
            log(f"  FAILED: {k}")
    assert nfail == 0, f"{nfail} check(s) failed"
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
