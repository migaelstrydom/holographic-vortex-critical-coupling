"""Page-level comparison of the O(alpha B^2) G = 0 backreaction thermodynamics
(boundary stress, trace anomaly, free energy and entropy at fixed T) against
D'Hoker-Kraus magnetic branes (paper app. A.1).

Sources (LaTeX pulled from export.arxiv.org):
  DK1 = arXiv:0908.3875  'Magnetic Brane Solutions in AdS_5'
  DK2 = arXiv:0911.4518  'Charged Magnetic Brane Solutions in AdS_5 and the
                          Fate of the Third Law of Thermodynamics'

Their conventions (DK2 eqs 2.1-2.7; DK1 footnote 1):
  S = -(1/16 pi G5) Int sqrt(-g) (R_DK + F^2 - 12) + S_CS + S_bndy,
  R_DK computed with the OPPOSITE-sign Riemann convention
  (R^l_{mnk} = +d_k Gamma^l_{mn} - d_n Gamma^l_{mk} + GG - GG), i.e.
  Ricci_DK = -Ricci_standard.  Their EOM (DK2 eq 2.4):
      R_MN = 4 g_MN + (1/3) F^2 g_MN - 2 F_MP F_N^P          [their Ricci]
  Boundary action (DK2 eq 2.3):
      (1/8 pi G5) Int sqrt(-gamma) (K - 3 + (1/4)R(gamma)
                                    + (1/2)(ln r) F^{mu nu}F_{mu nu})
  Stress tensor (DK2 eq 2.7):
      8 pi G5 T^{mu nu} = r^6 ( -K^{mu nu} + K gamma^{mu nu} - 3 gamma^{mu nu}
                     - 2 (F^{mu a}F^nu_a - (1/4)F^2 gamma^{mu nu}) ln r )

Our conventions (paper sec. 2 and app. A):
  2 kappa^2 S = Int sqrt(-g)(R + 12) - (alpha/2) Int sqrt(-g) F^2 + GH + cts,
  standard Riemann/Ricci signs, mostly-plus signature, L = 1, F_xy = B.

DICTIONARY (each factor anchored independently, never tuned to the targets):
  [A] alpha = 2:  route (a) their printed EOM/horizon-constraint coefficients
      (DK2 2.4, 4.3; DK1 sec 2.2; DK2 5.1 extremal 6 - q^2 - 2b^2 = 0) are
      rederived here from OUR action with symbolic alpha; matching the q^2 and
      b^2 coefficients separately (overdetermined) forces alpha = 2.
  [B] 2 kappa^2 = 16 pi G5: anchored at B = 0 by the AdS5-Schwarzschild
      free energy: their (2.7) evaluated here gives E0 = 3M/(16 pi G5),
      s0 = r+^3/(4G5), F0 = -(pi T)^4/(16 pi G5); ours kappa^2 F0 = -(pi T)^4/2.
  [C] B_theirs = B_ours: F = B dx1^dx2 on a unit-normalised Minkowski boundary
      frame on both sides (their asymptotic frame, DK2 sec 4.4); field along
      x3 <-> our z.  (Their script-B = sqrt(3) B is only CFT flux counting.)

Stages
------
[1] SymPy: standard Ricci of their ansatz (DK2 3.2, C=0), q=0 and q!=0;
    verify our-action EOM  R_MN + 4 g_MN - alpha(F_MP F_N^P - (1/6)F^2 g_MN)=0
    reproduces DK2 (2.4) with their Ricci = -ours  iff alpha = 2 (all
    components), and reproduces DK2 (4.3)/(5.1) horizon data iff alpha = 2.
[2] SymPy: their perturbative uncharged solution (DK2 sec 6 at rho = 0):
    U2 = -(2/3)ln(r/rp)/r^2 - 1/(3r^2) - a3/(2r^2),
    T2' from (r^3 U0 T2')' = -2/r, asymptotics T2 = ln(r/rp)/(2r^4) + 1/(8r^4);
    verified to solve the alpha = 2 system of stage [1] at O(B^2), and to
    reproduce their printed asymptotic relations (DK2 4.5, 6.16).
[3] SymPy: boundary stress tensor from their (2.7) on that solution ->
    exact per-B^2 table, trace anomaly, energy.
[4] Thermodynamics at fixed T: horizon shift, T(r+,B,a3), s, F = E - Ts,
    a3-independence, first law, magnetisation, P_z = -F, P_perp = -F - MB.
[5] Numerics 1: solve the linearised T2/U2 ODEs by RK from the horizon and
    match the closed forms (independent of the hand integration).
[6] Numerics 2 (adjudicates their tau(0) = -4/3 bookkeeping): full NONLINEAR
    shooting of the uncharged system in THEIR coordinates/normalisations
    (DK1 sec 2.2 horizon data) at small b; measure physical (T, B, s) and
    test  4 G5 s - (pi T)^3  =  B^2/(2 pi T).
[7] Translate with [A]+[B]+[C] and compare per observable with our values: stress (1/2,0,0,0), anomaly -alpha B^2/2, kappa^2 F =
    -(pi T)^4/2 - (alpha B^2/2) ln(pi T/mubar), magnetisation.

Run:  uv run python -m backreaction.anchors.dhoker_kraus_compare
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp, quad

PASS = {True: "PASS", False: "FAIL"}
failures = []


def check(name, ok):
    print(f"  [{PASS[bool(ok)]}] {name}")
    if not ok:
        failures.append(name)


def main():
    # ----------------------------------------------------------------------------
    # Stage [1]: EOM dictionary anchor (alpha = 2), symbolic.
    # ----------------------------------------------------------------------------
    print(
        "== [1] dictionary route (a): their EOM/horizon constraints from our action =="
    )

    r, B, q, alpha, a3, rp, M0 = sp.symbols("r B q alpha a3 r_p M_0", positive=True)
    U = sp.Function("U")(r)
    V = sp.Function("V")(r)
    W = sp.Function("W")(r)
    A_t = sp.Function("At")(r)  # electric potential, E(r) = -At'

    coords = sp.symbols("t rr x1 x2 x3")
    t, rr, x1, x2, x3 = coords

    def ricci_standard(g, xs):
        """Standard-convention Ricci: R^l_{mnk} = d_n G^l_{mk} - d_k G^l_{mn} + ...,
        R_mk = R^l_{mlk}. Mostly-plus signature."""
        n = len(xs)
        ginv = g.inv()
        Gam = [
            [
                [
                    sp.together(
                        sum(
                            ginv[l, s]
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
        Ric = sp.zeros(n, n)
        for m in range(n):
            for k in range(n):
                expr = 0
                for l in range(n):
                    expr += sp.diff(Gam[l][m][k], xs[l]) - sp.diff(Gam[l][m][l], xs[k])
                    for s in range(n):
                        expr += (
                            Gam[l][l][s] * Gam[s][m][k] - Gam[l][k][s] * Gam[s][m][l]
                        )
                Ric[m, k] = sp.simplify(expr)
        return Ric

    # metric of DK2 (3.2) with C = 0, coordinates (t, r, x1, x2, x3), r as symbol rr -> r
    g = sp.diag(-U, 1 / U, sp.exp(2 * V), sp.exp(2 * V), sp.exp(2 * W))
    # replace r-dependence: treat functions of r with rr as the coordinate
    g = g.subs({U: U.subs(r, rr), V: V.subs(r, rr), W: W.subs(r, rr)})
    g = sp.Matrix(5, 5, lambda i, j: g[i, j])

    # Maxwell field: F = E(r) dr^dt + B dx1^dx2, E = -At'(r)
    F = sp.zeros(5, 5)
    E_r = -sp.diff(A_t.subs(r, rr), rr)
    F[1, 0] = E_r
    F[0, 1] = -E_r
    F[2, 3] = B
    F[3, 2] = -B

    ginv = g.inv()
    F2 = sp.simplify(
        sum(
            F[m, n] * ginv[m, a] * ginv[n, b] * F[a, b]
            for m in range(5)
            for n in range(5)
            for a in range(5)
            for b in range(5)
        )
    )

    FMPFNP = sp.zeros(5, 5)
    for m in range(5):
        for n in range(5):
            FMPFNP[m, n] = sp.simplify(
                sum(F[m, p] * ginv[p, s] * F[n, s] for p in range(5) for s in range(5))
            )

    Ric = ricci_standard(g, list(coords))

    # Our Einstein equation in Ricci form (standard conventions, L in 2kappa^2 units):
    #   R_MN + 4 g_MN - alpha (F_MP F_N^P - (1/6) F^2 g_MN) = 0
    OurEOM = sp.simplify(Ric + 4 * g - alpha * (FMPFNP - F2 * g / 6))

    # Their equation, DK2 (2.4), with THEIR Ricci = -standard Ricci:
    #   -R_MN(std) = 4 g + (1/3) F^2 g - 2 F_MP F_N^P
    TheirEOM = sp.simplify(-Ric - 4 * g - F2 * g / 3 + 2 * FMPFNP)

    # TheirEOM is 'their equation moved to one side'; OurEOM likewise. The two
    # describe the same physics iff TheirEOM == -OurEOM identically (both sides are
    # the SAME tensor equation, written with opposite overall sign conventions):
    diff_expr = sp.simplify((OurEOM + TheirEOM).subs(alpha, 2))
    check(
        "our EOM (alpha=2)  ==  DK2 (2.4) with their Ricci = -standard  (all 25 components)",
        all(sp.simplify(d) == 0 for d in diff_expr),
    )

    # alpha-sensitivity: the sum for general alpha is (2-alpha)*(matter source)
    diff_gen = sp.simplify(OurEOM + TheirEOM)
    matter = sp.simplify(FMPFNP - F2 * g / 6)
    check(
        "OurEOM + TheirEOM == (2 - alpha) * matter source (so alpha = 2 is forced)",
        all(
            sp.simplify(diff_gen[i, j] - (2 - alpha) * matter[i, j]) == 0
            for i in range(5)
            for j in range(5)
        ),
    )

    # Horizon constraints DK2 (4.3) at q, b: derive from OUR EOM with symbolic alpha.
    # At the horizon U(rp)=0, V,W regular, E(rp) = q e^{-2V-W}*...  In their horizon
    # frame V(rp)=W(rp)=0 and E(rp)=q (DK2 4.2). Evaluate the (x1x1) and (x3x3)
    # components of OurEOM at the horizon:
    Urp, Uprp, Vprp, Wprp = sp.symbols("U_h Up_h Vp_h Wp_h")
    hor = {
        U.subs(r, rr): 0,
        sp.diff(U.subs(r, rr), rr): Uprp,
        V.subs(r, rr): 0,
        sp.diff(V.subs(r, rr), rr): Vprp,
        W.subs(r, rr): 0,
        sp.diff(W.subs(r, rr), rr): Wprp,
        E_r: q,
    }
    # use l'Hopital-free route: multiply the (11) EOM by 1 (it is regular at U->0
    # only after using regularity; instead evaluate the combination that their
    # equations state: E2, E3 at horizon). The (x1x1) component of R_MN + ... has a
    # U * V'' term; at the horizon the second-derivative terms multiply U -> 0.
    Vpp, Wpp = sp.symbols("Vpp Wpp")
    hor2 = dict(hor)
    hor2[sp.diff(V.subs(r, rr), rr, 2)] = Vpp
    hor2[sp.diff(W.subs(r, rr), rr, 2)] = Wpp
    hor2[sp.diff(U.subs(r, rr), rr, 2)] = sp.Symbol("Upp")

    eq_x1x1 = sp.expand(OurEOM[2, 2].subs(hor2))
    eq_x3x3 = sp.expand(OurEOM[4, 4].subs(hor2))
    # these must reduce to  Up*Vp = 4 - (alpha/6)*2*q^2... print and check the form:
    solV = sp.solve(eq_x1x1, Uprp * Vprp)
    solW = sp.solve(eq_x3x3, Uprp * Wprp)
    UpVp = sp.expand(solV[0]) if solV else None
    UpWp = sp.expand(solW[0]) if solW else None
    print(f"    our horizon data (symbolic alpha):  U'V' = {UpVp},  U'W' = {UpWp}")
    check(
        "U'V'|hor = 4 - (2/3)q^2 - (4/3)b^2 at alpha=2  [DK2 (4.3); DK1 sec 2.2]",
        sp.simplify(
            UpVp.subs(alpha, 2)
            - (4 - sp.Rational(2, 3) * q**2 - sp.Rational(4, 3) * B**2)
        )
        == 0,
    )
    check(
        "U'W'|hor = 4 - (2/3)(q^2 - b^2) at alpha=2  [DK2 (4.3), C'=0]",
        sp.simplify(UpWp.subs(alpha, 2) - (4 - sp.Rational(2, 3) * (q**2 - B**2))) == 0,
    )
    check(
        "alpha appears in U'V' as 4 - (alpha/3)q^2 - (2alpha/3)b^2 (coefficient ratio anchors alpha uniquely)",
        sp.simplify(UpVp - (4 - alpha * q**2 / 3 - 2 * alpha * B**2 / 3)) == 0,
    )

    # Extremal condition DK2 (5.1): at an extremal horizon U = U' = 0, and V stays
    # regular (e^{2V} finite there even in the AdS3 x R^2 limit), so the x1x1 row
    # U'V' = 4 - (alpha/3) q^2 - (2 alpha/3) b^2 degenerates to
    #   4 - (alpha/3)(q^2 + 2 b^2) = 0,  i.e.  q^2 + 2 b^2 = 12/alpha.
    # (The W row cannot be used this way: e^{2W} ~ U at the extremal horizon.)
    ext_cond = sp.solve(sp.Eq(UpVp, 0), q**2)
    print(f"    extremal condition from x1x1 row: q^2 = {ext_cond[0]}")
    check(
        "extremal condition at alpha=2: q^2 + 2 b^2 = 6  [DK2 (5.1); DK1 sec 2.2 b_ext = sqrt(3)]",
        sp.simplify(ext_cond[0].subs(alpha, 2) - (6 - 2 * B**2)) == 0,
    )
    # The uncharged throat (paper app. A.1): b = B e^{-2V} is the field in the
    # frozen flux-plane frame, so q = 0 gives e^{2V} = B/b.  At alpha = 2 this is
    # DK1's e^{2V} = B/sqrt(3) (after their eq. 2.8), and in general the paper's
    # v = B sqrt(alpha/6) of eq. (4.1).
    b_throat = sp.sqrt(sp.solve(sp.Eq(ext_cond[0], 0), B**2)[0])
    Bfield = sp.Symbol("B_field", positive=True)
    v_throat = Bfield / b_throat
    check(
        "uncharged throat e^{2V} = B sqrt(alpha/6), = B/sqrt(3) at alpha=2  [DK1 after (2.8); paper (4.1)]",
        sp.simplify(v_throat - Bfield * sp.sqrt(alpha / 6)) == 0
        and sp.simplify(v_throat.subs(alpha, 2) - Bfield / sp.sqrt(3)) == 0,
    )

    # ----------------------------------------------------------------------------
    # Stage [2]: their uncharged perturbative solution solves the alpha=2 system
    # ----------------------------------------------------------------------------
    print(
        "== [2] their uncharged O(B^2) solution (DK2 sec 6, rho=0) vs the alpha=2 system =="
    )

    # background U0 with horizon rp: U0 = r^2 - rp^4/r^2
    U0 = rr**2 - rp**4 / rr**2
    U2f = (
        -sp.Rational(2, 3) * sp.log(rr / rp) / rr**2
        - 1 / (3 * rr**2)
        - a3 / (2 * rr**2)
    )
    T2 = sp.Function("T2")(rr)

    # Their linearised ODEs (DK2 6.8 with C1=P1=0, and 6.14/6.15 with X = 4/(3r)):
    ode_T2 = sp.Eq(sp.diff(rr**3 * U0 * sp.diff(T2, rr), rr), -2 / rr)
    ode_U2 = sp.simplify(sp.diff(rr**3 * sp.diff(U2f, rr), rr) - sp.Rational(4, 3) / rr)
    check(
        "U2 closed form satisfies (r^3 U2')' = 4/(3r)  [DK2 (6.14)-(6.15), rho=0]",
        sp.simplify(ode_U2) == 0,
    )

    # independent check: the full metric U0+B^2 U2, V=ln r + B^2 T2/3, W=ln r -2B^2 T2/3
    # solves OUR alpha=2 equations to O(B^2), with T2 satisfying ode_T2.
    Vfull = sp.log(rr) + B**2 * T2 / 3
    Wfull = sp.log(rr) - 2 * B**2 * T2 / 3
    Ufull = U0 + B**2 * U2f
    subs_full = {
        U.subs(r, rr): Ufull,
        V.subs(r, rr): Vfull,
        W.subs(r, rr): Wfull,
        A_t.subs(r, rr): 0,
    }

    def sub_derivs(expr):
        expr = expr.subs(
            {
                sp.diff(U.subs(r, rr), rr, 2): sp.diff(Ufull, rr, 2),
                sp.diff(V.subs(r, rr), rr, 2): sp.diff(Vfull, rr, 2),
                sp.diff(W.subs(r, rr), rr, 2): sp.diff(Wfull, rr, 2),
                sp.diff(A_t.subs(r, rr), rr, 2): 0,
                sp.diff(U.subs(r, rr), rr): sp.diff(Ufull, rr),
                sp.diff(V.subs(r, rr), rr): sp.diff(Vfull, rr),
                sp.diff(W.subs(r, rr), rr): sp.diff(Wfull, rr),
                sp.diff(A_t.subs(r, rr), rr): 0,
            }
        )
        return expr.subs(subs_full)

    ode_T2_solved = sp.solve(ode_T2.doit(), sp.diff(T2, rr, 2))[0]

    ok_all = True
    for i, j in [(0, 0), (1, 1), (2, 2), (4, 4)]:
        e = sub_derivs(OurEOM[i, j].subs(alpha, 2))
        e = e.subs(sp.diff(T2, rr, 2), ode_T2_solved)
        e2 = sp.series(sp.expand(e), B, 0, 3).removeO()
        e2 = sp.simplify(e2)
        if sp.simplify(e2) != 0:
            ok_all = False
            print(f"    residual [{i}{j}]: {e2}")
    check(
        "full perturbed metric solves our alpha=2 EOM at O(B^2) (T2 via its ODE)",
        ok_all,
    )

    # asymptotics of T2. Exact first integral (horizon regularity fixes the
    # constant): r^3 U0 T2' = -2 ln(r/rp), so T2' = -2 ln(r/rp)/(r^3 U0) exactly,
    # and T2_asym errs only at relative O(rp^4/r^4):
    T2p_exact = -2 * sp.log(rr / rp) / (rr**3 * U0)
    check(
        "exact first integral: (r^3 U0 T2')' = -2/r with r^3 U0 T2' -> 0 at the horizon",
        sp.simplify(sp.diff(rr**3 * U0 * T2p_exact, rr) + 2 / rr) == 0,
    )
    T2_asym = sp.log(rr / rp) / (2 * rr**4) + 1 / (8 * rr**4)
    res_asym = sp.simplify(sp.diff(rr**3 * U0 * sp.diff(T2_asym, rr), rr) + 2 / rr)
    check(
        "T2 asymptotics ln(r/rp)/(2r^4) + 1/(8r^4): ODE residual is O(ln r/r^5), i.e. r^4-suppressed",
        sp.limit(res_asym * rr**4, rr, sp.oo) == 0
        and sp.limit(res_asym * rr**6, rr, sp.oo) != 0,
    )

    # their printed asymptotic relations (DK2 4.5 / 6.16), v = w = 1:
    e2V = sp.exp(2 * Vfull)
    v2p = sp.limit(
        (e2V - rr**2).subs(T2, T2_asym).doit() * rr**2 / sp.log(rr), rr, sp.oo
    )
    check("v2' = B^2/3  [DK2 (4.5)]", sp.simplify(v2p - B**2 / 3) == 0)
    u2p = sp.limit(
        (Ufull - rr**2 + M0 / rr**2).subs(M0, rp**4) * rr**2 / sp.log(rr), rr, sp.oo
    )
    check("u2' = -2B^2/3  [DK2 (4.5)]", sp.simplify(u2p + 2 * B**2 / 3) == 0)
    u2_const = sp.simplify(
        (Ufull - rr**2 + rp**4 / rr**2 - u2p * sp.log(rr) / rr**2) * rr**2
    )
    u2_expect = B**2 * (sp.Rational(2, 3) * sp.log(rp) - sp.Rational(1, 3) - a3 / 2)
    check(
        "u2 = -M0 + B^2((2/3)ln rp - 1/3 - a3/2)  [DK2 (6.16)]",
        sp.simplify(sp.limit(u2_const, rr, sp.oo) - (u2_expect - 0)) == 0,
    )

    # ----------------------------------------------------------------------------
    # Stage [3]: boundary stress tensor from their (2.7)
    # ----------------------------------------------------------------------------
    print("== [3] their stress tensor (DK2 2.7) on the uncharged O(B^2) solution ==")

    # asymptotic metric through relative order r^-4 (u^4 data), logs exact:
    L_ = sp.log(rr)
    Uasy = (
        rr**2
        - M0 / rr**2
        + B**2
        * (
            -sp.Rational(2, 3) * (L_ - sp.log(rp)) / rr**2
            - 1 / (3 * rr**2)
            - a3 / (2 * rr**2)
        )
    )
    V2asy = T2_asym / 3
    W2asy = -2 * T2_asym / 3
    gam = sp.diag(
        -Uasy,
        sp.exp(2 * (sp.log(rr) + B**2 * V2asy)),
        sp.exp(2 * (sp.log(rr) + B**2 * V2asy)),
        sp.exp(2 * (sp.log(rr) + B**2 * W2asy)),
    )
    # boundary coordinates order: (t, x1, x2, x3)
    gam = sp.Matrix(4, 4, lambda i, j: sp.expand(gam[i, j]))

    grr = 1 / Uasy
    Kdd = sp.Matrix(4, 4, lambda i, j: sp.diff(gam[i, j], rr) / (2 * sp.sqrt(grr)))
    gaminv = sp.Matrix(4, 4, lambda i, j: 0)
    for i in range(4):
        gaminv[i, i] = 1 / gam[i, i]
    Kuu = gaminv * Kdd * gaminv
    Ktr = sum(gaminv[i, i] * Kdd[i, i] for i in range(4))

    # boundary Maxwell: F_12 = B (boundary indices), F^{mu a}F^nu_a with gamma:
    Fb = sp.zeros(4, 4)
    Fb[1, 2] = B
    Fb[2, 1] = -B
    Fbuu = gaminv * Fb * gaminv  # F^{mu nu}
    FFb = sp.zeros(4, 4)
    for m in range(4):
        for n in range(4):
            FFb[m, n] = sum(
                Fbuu[m, a] * Fb[n, s] * gaminv[a, s] * 0
                for a in range(4)
                for s in range(4)
            )
    # F^{mu a} F^{nu}{}_a = F^{mu a} F^{nu b} gamma_{ab}
    FFb = sp.Matrix(
        4,
        4,
        lambda m, n: sum(
            Fbuu[m, a] * Fbuu[n, b] * gam[a, b] for a in range(4) for b in range(4)
        ),
    )
    F2b = sum(Fb[m, n] * Fbuu[m, n] for m in range(4) for n in range(4))

    Tuu = sp.Matrix(
        4,
        4,
        lambda m, n: (
            rr**6
            * (
                -Kuu[m, n]
                + Ktr * gaminv[m, n]
                - 3 * gaminv[m, n]
                - 2 * (FFb[m, n] - F2b * gaminv[m, n] / 4) * L_
            )
        ),
    )

    eps = sp.Symbol("eps", positive=True)
    Tval = sp.zeros(4, 4)
    for i in range(4):
        ser = Tuu[i, i].subs(rr, 1 / eps)
        ser = sp.expand(sp.series(ser, eps, 0, 3).removeO())
        # keep only the eps^0, log-free limit; check logs cancel
        ser = sp.expand(ser)
        lim = ser.subs(sp.log(1 / eps), sp.Symbol("Leps"))
        lim = sp.expand(lim)
        val = sp.limit(
            sp.series(Tuu[i, i].subs(rr, 1 / eps), eps, 0, 2).removeO(), eps, 0
        )
        Tval[i, i] = sp.simplify(val)

    E_dk = sp.expand(Tval[0, 0])  # 8 pi G5 T^{tt}
    Pperp_dk = sp.expand(Tval[1, 1])  # 8 pi G5 T^{x1x1}
    Pz_dk = sp.expand(Tval[3, 3])  # 8 pi G5 T^{x3x3}
    trace_dk = sp.expand(-E_dk + 2 * Pperp_dk + Pz_dk)

    print(f"    8 pi G5 T^tt      = {E_dk}")
    print(f"    8 pi G5 T^x1x1    = {Pperp_dk}")
    print(f"    8 pi G5 T^x3x3    = {Pz_dk}")
    print(f"    8 pi G5 T^mu_mu   = {sp.simplify(trace_dk)}")

    check(
        "trace anomaly: 8 pi G5 T^mu_mu = -B^2 (scheme-free)",
        sp.simplify(trace_dk + B**2) == 0,
    )

    # ----------------------------------------------------------------------------
    # Stage [4]: thermodynamics at fixed T
    # ----------------------------------------------------------------------------
    print("== [4] their thermodynamics at fixed T (exact O(B^2), uncharged) ==")

    # horizon shift and temperature (hand result, now machine):
    U2_at = U2f.subs(rr, rp)
    U2p_at = sp.diff(U2f, rr).subs(rr, rp)
    U0p = sp.diff(U0, rr).subs(rr, rp)  # 4 rp
    U0pp = sp.diff(U0, rr, 2).subs(rr, rp)  # -4
    rh = rp - B**2 * U2_at / U0p
    T_full = sp.expand((U0p + B**2 * U2p_at + U0pp * (rh - rp)) / (4 * sp.pi))
    T_full = sp.simplify(T_full)
    print(f"    T(rp, B) = {T_full}")
    check(
        "4 pi T = 4 rp + B^2 (a3/2 - 1/3)/rp^3",
        sp.simplify(
            4 * sp.pi * T_full - (4 * rp + B**2 * (a3 / 2 - sp.Rational(1, 3)) / rp**3)
        )
        == 0,
    )

    # entropy density: s = (horizon area density)/(4 G5); e^{2V+W} = r^3 exactly
    # (S2 = 0), so 8 pi G5 s = 2 pi rh^3.
    s8_rp = sp.expand(sp.series(2 * sp.pi * rh**3, B, 0, 3).removeO())  # 8 pi G5 s

    # invert T(rp): rp(T) at O(B^2)
    rT = sp.Symbol("r_T", positive=True)  # rT = pi T
    rp_of_T = rT - B**2 * (a3 / 2 - sp.Rational(1, 3)) / (4 * rT**3)
    s8_T = sp.simplify(sp.expand(sp.series(s8_rp.subs(rp, rp_of_T), B, 0, 3).removeO()))
    print(f"    8 pi G5 s(T,B) = {s8_T}   (rT = pi T)")
    check(
        "8 pi G5 s(T,B) = 2 pi (pi T)^3 + pi B^2/(pi T), a3-independent",
        sp.simplify(s8_T - 2 * sp.pi * rT**3 - sp.pi * B**2 / rT) == 0,
    )

    # energy from stage [3]: E_dk contains M0 and B^2 pieces; express via rp then T
    E_rp = sp.simplify(E_dk.subs(M0, rp**4))
    E_T = sp.simplify(sp.expand(sp.series(E_rp.subs(rp, rp_of_T), B, 0, 3).removeO()))
    print(f"    8 pi G5 E(T,B) = {E_T}")
    check("E(T,B) is a3-independent", sp.diff(E_T, a3) == 0)

    # free energy F = E - Ts  (8 pi G5 units), T = rT/pi
    F_T = sp.simplify(sp.expand(E_T - (rT / sp.pi) * s8_T))
    print(f"    8 pi G5 F(T,B) = {F_T}")
    check("F(T,B) is a3-independent", sp.diff(F_T, a3) == 0)

    # B = 0 anchor (dictionary route [B], non-circular: E0 from their stress
    # tensor, s0 from the area law):
    F0_T = sp.simplify(F_T.subs(B, 0))
    check(
        "their 8 pi G5 F0 = -(pi T)^4/2  (so F0 = -(pi T)^4/(16 pi G5))",
        sp.simplify(F0_T + rT**4 / 2) == 0,
    )

    dF_B2 = sp.simplify((F_T - F0_T).coeff(B, 2))
    print(f"    8 pi G5 dF|_T per B^2 = {dF_B2}")
    check(
        "8 pi G5 dF|_T per B^2 = -ln(pi T)  (their scheme point: mubar_DK = 1/L)",
        sp.simplify(dF_B2 + sp.log(rT)) == 0,
    )
    # first law: dF/dT = -s
    check(
        "first law dF/dT = -s at O(B^2)",
        sp.simplify(sp.diff(F_T, rT) * sp.pi - (-s8_T)) == 0,
    )

    # magnetisation (definition m = -dF/dB, same definition used on both sides)
    m_dk = sp.simplify(-sp.diff(F_T, B))
    print(f"    8 pi G5 m = -d(8 pi G5 F)/dB = {m_dk}")
    check("8 pi G5 m = 2 B ln(pi T)", sp.simplify(m_dk - 2 * B * sp.log(rT)) == 0)

    # magnetic-system identities
    Pperp_T = sp.simplify(
        sp.expand(
            sp.series(
                sp.simplify(Pperp_dk.subs(M0, rp**4)).subs(rp, rp_of_T), B, 0, 3
            ).removeO()
        )
    )
    Pz_T = sp.simplify(
        sp.expand(
            sp.series(
                sp.simplify(Pz_dk.subs(M0, rp**4)).subs(rp, rp_of_T), B, 0, 3
            ).removeO()
        )
    )
    print(f"    8 pi G5 P_perp(T,B) = {Pperp_T}")
    print(f"    8 pi G5 P_z(T,B)    = {Pz_T}")
    check("P_z = -F (longitudinal pressure)", sp.simplify(Pz_T + F_T) == 0)
    check("P_perp = -F - mB", sp.simplify(Pperp_T + F_T + m_dk * B) == 0)

    # ----------------------------------------------------------------------------
    # Stage [5]: numeric check of the linearised closed forms
    # ----------------------------------------------------------------------------
    print("== [5] numeric ODE check of U2, T2 closed forms (rp = 1) ==")

    def rhs_T2(rr_, y):
        T2v, T2pv = y
        U0v = rr_**2 - 1 / rr_**2
        # (r^3 U0 T2')' = -2/r
        return [
            T2pv,
            (-2 / rr_ - (3 * rr_**2 * U0v + rr_**3 * (2 * rr_ + 2 / rr_**3)) * T2pv)
            / (rr_**3 * U0v),
        ]

    # regular start near horizon: from the ODE, at r->1: r^3 U0 T2' = -2 ln r + c;
    # smoothness requires c = 0 => T2'(r) ~ -2 ln r/(r^3 U0) finite? U0 ~ 4(r-1):
    # T2' ~ -2(r-1)/(4(r-1)) = -1/2 at r=1.
    r0n = 1 + 1e-8
    sol = solve_ivp(
        rhs_T2, [r0n, 1e4], [0.0, -0.5], rtol=1e-12, atol=1e-14, dense_output=True
    )
    # integrate T2 down from infinity: instead use closed-form comparison of T2'(r):
    rs = np.array([1.5, 2.0, 3.0, 5.0, 10.0, 100.0])
    T2p_num = sol.sol(rs)[1]
    T2p_closed = np.array(
        [-2 * np.log(r_) / (r_**3 * (r_**2 - 1 / r_**2)) for r_ in rs]
    )
    err = np.max(np.abs(T2p_num / T2p_closed - 1))
    check(
        f"T2'(r) numeric (RK from horizon) vs exact -2 ln r/(r^3 U0): max rel err = {err:.2e}",
        err < 1e-5,
    )

    # far-field: T2 (integrated from infinity, T2(inf)=0) vs asymptotic form;
    # the asymptotic form errs at relative O(r^-4), so test at moderate r with the
    # expected suppression
    T2_num_at = {}
    for r_ in [8.0, 16.0]:
        val, aerr = quad(
            lambda x: -2 * np.log(x) / (x**3 * (x**2 - 1 / x**2)),
            np.inf,
            r_,
            epsabs=1e-15,
            epsrel=1e-12,
            limit=400,
        )
        T2_num_at[r_] = val
    asym = lambda r_: np.log(r_) / (2 * r_**4) + 1 / (8 * r_**4)
    err8 = abs(T2_num_at[8.0] / asym(8.0) - 1)
    err16 = abs(T2_num_at[16.0] / asym(16.0) - 1)
    check(
        f"T2(r) far field vs ln r/(2r^4) + 1/(8r^4): rel err {err8:.1e} (r=8) -> {err16:.1e} (r=16), "
        f"suppression ratio {err8 / err16:.1f} (expect ~2^4 = 16 up to logs)",
        err8 < 2e-2 and err16 < 2e-3 and 6 < err8 / err16 < 40,
    )

    # ----------------------------------------------------------------------------
    # Stage [6]: full nonlinear shooting in their conventions (adjudicates tau(0))
    # ----------------------------------------------------------------------------
    print(
        "== [6] nonlinear uncharged magnetic brane, their coordinates (DK1 sec 2.2 data) =="
    )

    # derive the reduced ODEs from OurEOM (alpha=2, q=0) symbolically, solve for
    # U'', V'', W''
    Up_, Vp_, Wp_ = sp.symbols("Up Vp Wp")
    Upp_, Vpp_, Wpp_ = sp.symbols("Upp Vpp Wpp")
    Uv, Vv, Wv = sp.symbols("Uv Vv Wv", positive=True)
    rep = {
        sp.diff(U.subs(r, rr), rr, 2): Upp_,
        sp.diff(V.subs(r, rr), rr, 2): Vpp_,
        sp.diff(W.subs(r, rr), rr, 2): Wpp_,
        sp.diff(U.subs(r, rr), rr): Up_,
        sp.diff(V.subs(r, rr), rr): Vp_,
        sp.diff(W.subs(r, rr), rr): Wp_,
        U.subs(r, rr): Uv,
        V.subs(r, rr): sp.log(Vv) / 2,
        W.subs(r, rr): sp.log(Wv) / 2,
        A_t.subs(r, rr): 0,
    }
    # careful: keep V, W as V(r), W(r); use e^{2V} = Vv is messy -- instead keep Vv = V(r) value
    rep = {
        sp.diff(U.subs(r, rr), rr, 2): Upp_,
        sp.diff(V.subs(r, rr), rr, 2): Vpp_,
        sp.diff(W.subs(r, rr), rr, 2): Wpp_,
        sp.diff(U.subs(r, rr), rr): Up_,
        sp.diff(V.subs(r, rr), rr): Vp_,
        sp.diff(W.subs(r, rr), rr): Wp_,
        U.subs(r, rr): Uv,
        V.subs(r, rr): Vv,
        W.subs(r, rr): Wv,
        A_t.subs(r, rr): 0,
        sp.diff(A_t.subs(r, rr), rr): 0,
        sp.diff(A_t.subs(r, rr), rr, 2): 0,
    }
    eqs = [sp.simplify(OurEOM[i, i].subs(alpha, 2).subs(rep)) for i in [0, 1, 2, 4]]
    sol2 = sp.solve(eqs[0:1] + eqs[2:4], [Upp_, Vpp_, Wpp_], dict=True)
    assert sol2, "could not solve for second derivatives"
    sol2 = sol2[0]
    fUpp = sp.lambdify((rr, Uv, Vv, Wv, Up_, Vp_, Wp_, B), sol2[Upp_], "numpy")
    fVpp = sp.lambdify((rr, Uv, Vv, Wv, Up_, Vp_, Wp_, B), sol2[Vpp_], "numpy")
    fWpp = sp.lambdify((rr, Uv, Vv, Wv, Up_, Vp_, Wp_, B), sol2[Wpp_], "numpy")
    constraint = sp.lambdify((rr, Uv, Vv, Wv, Up_, Vp_, Wp_, B), eqs[1], "numpy")

    def run_brane(b, rmax=2000.0, u1=4.0):
        def rhs(r_, y):
            Uv_, Vv_, Wv_, Up__, Vp__, Wp__ = y
            return [
                Up__,
                Vp__,
                Wp__,
                fUpp(r_, Uv_, Vv_, Wv_, Up__, Vp__, Wp__, b),
                fVpp(r_, Uv_, Vv_, Wv_, Up__, Vp__, Wp__, b),
                fWpp(r_, Uv_, Vv_, Wv_, Up__, Vp__, Wp__, b),
            ]

        # horizon data (DK1 sec 2.2 with U'(1) = u1): U'V' = 4 - (4/3)b^2 etc.
        d = 1e-7
        Vp0 = (4 - 4 * b**2 / 3) / u1
        Wp0 = (4 + 2 * b**2 / 3) / u1
        y0 = [u1 * d, Vp0 * d, Wp0 * d, u1, Vp0, Wp0]
        soln = solve_ivp(
            rhs,
            [1 + d, rmax],
            y0,
            rtol=3e-13,
            atol=1e-16,
            dense_output=True,
            method="DOP853",
        )
        assert soln.success
        # constraint drift
        ys = soln.sol(rmax)
        cvi = constraint(rmax, *ys, b)

        # asymptotics: e^{2V} -> vinf (r - r0)^2, U -> uinf (r - r0u)^2, e^{2W} -> winf(...)
        def quad_fit(fvals, rsam):
            # fit f = a r^2 + b r + c
            Amat = np.vstack([rsam**2, rsam, np.ones_like(rsam)]).T
            coef = np.linalg.solve(Amat.T @ Amat, Amat.T @ fvals)
            return coef  # a, b, c

        rsam = np.array([rmax * 0.4, rmax * 0.55, rmax * 0.7, rmax * 0.85, rmax])
        ysam = soln.sol(rsam)
        aU, bU, cU = quad_fit(ysam[0], rsam)
        aV, bV, cV = quad_fit(np.exp(2 * ysam[1]), rsam)
        aW, bW, cW = quad_fit(np.exp(2 * ysam[2]), rsam)
        uinf, vinf, winf = aU, aV, aW
        Tphys = u1 / (4 * np.pi * np.sqrt(uinf))
        Bphys = b / vinf
        s4G = 1 / (vinf * np.sqrt(winf))  # 4 G5 s
        return Tphys, Bphys, s4G, cvi

    print(
        "      b     T_phys      B_phys      4G5*s       Delta/B^2   (target 1/(2 pi T))"
    )
    res = []
    for b in [0.05, 0.1, 0.2]:
        Tp_, Bp_, s4, cvi = run_brane(b)
        Delta = s4 - (np.pi * Tp_) ** 3
        ratio = Delta / Bp_**2
        res.append((b, Tp_, Bp_, s4, ratio))
        print(
            f"    {b:5.2f}  {Tp_:.8f}  {Bp_:.6f}  {s4:.8f}  {ratio:.6f}    "
            f"{1 / (2 * np.pi * Tp_):.6f}"
        )

    # Richardson in b^2 to remove O(B^4): ratio(b) = c + d b^2
    b2 = np.array([r_[0] ** 2 for r_ in res])
    rat = np.array([r_[4] for r_ in res])
    targ = np.array([1 / (2 * np.pi * r_[1]) for r_ in res])
    coef = np.polyfit(b2, rat - targ, 1)
    check(
        f"nonlinear numerics: 4G5 s - (pi T)^3 = B^2/(2 pi T); extrapolated offset = {coef[1]:.2e}",
        abs(coef[1]) < 5e-4,
    )
    print(
        "    (their (6.28) tau(0) = -4/3 with (6.21) S2(rp)=0 naively gives 1/(pi T), i.e. 2x;"
    )
    print("     the numerics decide between the two readings.)")

    # ----------------------------------------------------------------------------
    # Stage [7]: the dictionary translation and final comparison
    # ----------------------------------------------------------------------------
    print("== [7] translation to our conventions and verdicts ==")
    # Dictionary: 2 kappa^2 = 16 pi G5 (kappa^2 = 8 pi G5), alpha = 2, B = B.
    # So: kappa^2 X = 8 pi G5 X, and 'per alpha B^2' = 'per B^2' / 2.
    # Their scheme scale: ln(r/L), L = 1; ours mubar = 1/u_H = 1 (both at pi T = 1
    # when r_p = u_H = 1). A single identification mubar = 1/L must reconcile ALL
    # scheme-dependent constants below.

    trace_theirs = sp.simplify(trace_dk / B**2 / 2)
    check(
        f"TRACE ANOMALY (scheme-free): theirs kappa^2 T^mu_mu per alpha B^2 = {trace_theirs}"
        f" == ours -1/2",
        sp.simplify(trace_theirs + sp.Rational(1, 2)) == 0,
    )

    ds_theirs = sp.simplify(sp.expand(s8_T).coeff(B, 2) / 2)
    check(
        f"ENTROPY SHIFT (scheme-free): theirs kappa^2 ds|_T per alpha B^2 = {ds_theirs}"
        f" == ours alpha B^2/(2T) -> pi/(2 rT)",
        sp.simplify(ds_theirs - sp.pi / (2 * rT)) == 0,
    )

    dFdlnT = sp.simplify(sp.expand(rT * sp.diff(F_T - F0_T, rT)).coeff(B, 2) / 2)
    check(
        f"F LOG COEFFICIENT (scheme-free): theirs d(kappa^2 dF|_T)/d ln T per alpha B^2 = "
        f"{dFdlnT} == ours -1/2",
        sp.simplify(dFdlnT + sp.Rational(1, 2)) == 0,
    )

    # --- scheme-dependent constants, one identification mubar = 1/L = 1 ---
    print("    scheme-dependent items (single identification mubar_ours = 1/L_theirs):")
    dF_theirs = sp.simplify(dF_B2 / 2)
    check(
        f"FREE ENERGY: theirs kappa^2 dF|_T per alpha B^2 = {dF_theirs} == ours "
        f"-(1/2) ln(pi T/mubar) with mubar = 1  [dF = 0 at pi T = 1]",
        sp.simplify(dF_theirs + sp.log(rT) / 2) == 0,
    )

    E_theirs = sp.simplify(E_T.coeff(B, 2) / 2)
    check(
        f"ENERGY: theirs kappa^2 dE|_T per alpha B^2 = {E_theirs} == ours "
        f"1/2 - (1/2)ln(pi T/mubar), mubar = 1",
        sp.simplify(E_theirs - (sp.Rational(1, 2) - sp.log(rT) / 2)) == 0,
    )

    Pp_theirs = sp.simplify(Pperp_T.coeff(B, 2) / 2)
    Pz_theirs = sp.simplify(Pz_T.coeff(B, 2) / 2)
    check(
        f"STRESS TABLE: theirs per alpha B^2 (tt,xx,yy,zz) = ({E_theirs}, {Pp_theirs}, "
        f"{Pp_theirs}, {Pz_theirs}) == ours (1/2,0,0,0) + (ln mubar/pi T)(1/2)diag(1,1,1,-1) "
        f"[the a5-shift pattern], mubar = 1",
        sp.simplify(Pp_theirs + sp.log(rT) / 2) == 0
        and sp.simplify(Pz_theirs - sp.log(rT) / 2) == 0,
    )

    m_theirs = sp.simplify(m_dk / 2)
    check(
        f"MAGNETISATION: theirs kappa^2(-dF/dB) per alpha = {m_theirs} == ours "
        f"-d(kappa^2 F)/dB = alpha B ln(pi T/mubar), mubar = 1  "
        f"(kappa^2 M = alpha B ln(mubar/pi T) is the +dF/dB convention)",
        sp.simplify(m_theirs - B * sp.log(rT)) == 0,
    )

    print()
    print(
        "VERDICT TABLE (per alpha B^2, kappa^2 units, dictionary alpha=2, 2kappa^2=16piG5, B=B):"
    )
    print(
        "    observable            ours                theirs (translated)      scheme-free?"
    )
    print(
        f"    trace anomaly         -1/2                {trace_theirs}                     yes"
    )
    print(
        f"    ds|_T                 pi/(2 rT)           {ds_theirs}                yes"
    )
    print(
        f"    dF/dlnT               -1/2                {dFdlnT}                     yes"
    )
    print(
        f"    dF|_T                 -(1/2)ln(rT/mubar)  {dF_theirs}               no (mubar = 1/L)"
    )
    print(f"    stress tt             1/2 @ mubar=piT     {E_theirs}       no")
    print(
        f"    stress xx=yy          0   @ mubar=piT     {Pp_theirs}                no"
    )
    print(
        f"    stress zz             0   @ mubar=piT     {Pz_theirs}                 no"
    )
    print(
        f"    -dF/dB                aB ln(rT/mubar)     {m_theirs}                 no"
    )

    print()
    if failures:
        print("FAILURES:", *failures, sep="\n  - ")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
