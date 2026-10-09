"""The supersymmetric value of alpha: Romans' N = 4+ SU(2) x U(1) gauged
supergravity has kappa^2/ghat^2 = L^2/4 for its SU(2).

Question.  alpha_* = 2/3 is a bottom-up threshold.  Where does the top-down
Einstein-SU(2) theory sit -- the SU(2) of Romans' N = 4+ gauged supergravity,
the S^5 truncation with an SU(2) x U(1) subgroup of SO(6)?  Derived here from
three Lagrangians in three convention systems, each coded.

Our convention.  S = (1/2 kappa^2) int sqrt(-g)(R + 12/L^2)
                     - (1/4 ghat^2) int sqrt(-g) F^a F^a,
F^a = dA^a + eps^{abc} A^b A^c (unit structure constant), alpha = kappa^2/ghat^2,
mostly-plus signature, box = (1/sqrt-g) d_m (sqrt-g g^mn d_n).

[1] Trace reversal (D = 5): R_MN + (4/L^2) g_MN = alpha (F_MP F_N^P - F^2 g/6),
    checked symbolically AND as an explicit matrix identity on a random metric
    and a random antisymmetric F -- the component check of the same step.

[2] Lu-Pope-Tran hep-th/9909203 eqs (7), (8), (10):
      L = R*1 - 3X^{-2}*dX^dX - (1/2)X^4 *G^G - (1/2)X^{-2}(*F^i^F^i + ...)
          + 2 g2 (g2 X^2 + 2 sqrt2 g1 X^{-1}) *1,      F^i = dA^i + (1/2) g2 eps A^j^A^k,
      R_mn = ... - (4/3) g^2 (X^2 + 2X^{-1}) g_mn + (1/2)X^{-2}(F^i_m^r F^i_nr - g_mn (F^i)^2/6)
    with g1 = g, g2 = sqrt2 g.  The potential 4g^2(X^2 + 2/X) is extremal at
    X = 1 with value 12 g^2, so L = 1/g (and eq (10) gives R_mn = -4 g^2 g_mn,
    consistent).  *F^F = (1/2) F_mn F^mn vol, so the kinetic term is
    -(1/4) X^{-2} F_mn F^mn with unit Einstein-Hilbert coefficient: 2 kappa^2 = 1.
    Components of eq (8): F^i_mn = dA - dA + g2 eps A A (the 1/2 is eaten by the
    wedge); rescaling A = A~/g2 to the unit structure constant gives
    ghat^2 = g2^2 at X = 1.  Hence alpha = 1/(2 g2^2) = 1/(4 g^2) = L^2/4.
    Route (b), reading eq (10) directly: coefficient (1/2)(1/g2^2) = L^2/4.
    Invariance: for general (g1, g2) the extremum X0^3 = sqrt2 g1/g2 gives
    L^2 = 2/(g2^2 X0^2) and ghat^2 = g2^2 X0^2, so alpha/L^2 = 1/4 regardless
    of the g1/g2 split -- "g2 = sqrt2 g1, X = 1" is a normalisation, not an
    input.

[3] Two more Lagrangians, each run through the same generic machinery
    (vacuum radius from the explicit Ricci tensor of AdS_5 in that theory's
    own signature, kinetic coefficients, rescaling to unit structure
    constant):
    (a) Gauntlett-Varela 0712.3560 eqs (2.8), (2.14), the reduction from
        D = 11: mostly plus, unit EH, -(1/2) X^{-2} *F^i^F^i,
        F^i = dA^i - (1/sqrt2) m eps A^j^A^k (structure constant sqrt2 m),
        potential 4 m^2 (X^2 + 2/X): L = 1/m, alpha = L^2/4.
    (b) Romans' own conventions as reproduced in Ohl-Uhlemann 1011.3533
        eqs (1), (5) and the line below (2): signature (+,-,-,-,-),
        e^-1 L = -(1/4) R + (1/2)(d phi)^2 + P - (1/4) xi^2 F^I F^I + ...,
        xi = exp(sqrt(2/3) phi), P = (g^2/8)(xi^-2 + 2 xi), D v^I = dv^I +
        g eps^IJK A^J v^K (structure constant g2 = g): L^2 = 8/g^2,
        alpha = 2/g^2 = L^2/4.  The map to LPT (xi = 1/X, phi = phi_LPT/2,
        g^2 = 8 g_LPT^2) is checked, as stated in GV below (2.14).
    1907.12561 eqs (2.1), (2.6), at unit structure constant and L = 1 with
    Ricci coefficient (1/4) X^{-2}, is alpha = 1/4 by inspection; not coded.
    (Ammon et al 0912.3515 use our action with alpha_AEGKO = kappa/ghat
    unsquared; the SUSY point is 1/2 in their variable.)

[4] Consistent truncation: X == 1 with SU(2) flux only is NOT consistent.
    The scalar equation of each of the three Lagrangians is derived by an
    explicit Euler-Lagrange variation on Poincare AdS_5 (scalar depending on
    two coordinates, F_mn F^mn an external function), linearised about the
    vacuum, and compared with box computed from the metric.  All three give,
    in mostly-plus signature and in the unit-structure-constant field,
        (box + 4/L^2) dX = -(alpha/6) F^a_mn F^{a mn} = -(L^2/24) F^a F^a,
    i.e. -(1/12) F_LPT^2 with F_LPT = F/g2 (F_LPT^2 = 2 alpha F^2).  m^2 L^2 = -4,
    Delta = 2 (the BF-saturating 20' mode), sourced at O(F^2) -- the same order
    as the metric backreaction -- by the background B^2 and by the condensate
    |W|^2, and feeding back into the SU(2) equation at O(alpha F).  (In
    mostly-minus signature the same equation reads
    (box_- - 4/L^2) dX = +(L^2/24) F^2.)
    The U(1) is not sourced for the static, z-independent lattice
    (F^i^F^i needs four distinct indices from {x, y, r}), nor are the two-forms.
    So the top-down SU(2) sector is {g, X, A^i}, and alpha = 1/4 is the value of
    the coupling ratio in the supergravity Lagrangian, not a claim that the
    Einstein-SU(2) lattice lifts.

Consequences: 1/4 < 2/3, so the supergravity point sits INSIDE
the unstable window (consistent with Almuhairi-Polchinski / DGP finding
charged-mode instabilities in SO(6) throats); alpha_* = 2/3 is not a
supersymmetric value; and the X exchange channel would be an additional
attractive (scalar, Coulomb-like) piece of the lattice kernel in the top-down
theory -- a direction, not a correction to anything here.  The other SU(2)s
of SO(6), and the effective alpha of the most-charged W: n4_su2_embeddings.

Run:  uv run python -m backreaction.derivations.susy_coupling   (~10 s)
"""

import random
import time
import types

import sympy as sp
from sympy.calculus.euler import euler_equations

T0 = time.time()
CHECKS = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


D = 5
# the result of [4]: (box + 4/L^2) dX = -(alpha/6) F^a F^a  (mostly plus, unit structure constant)
SOURCE_PER_ALPHA = sp.Rational(1, 6)
MASS2_L2 = -4
# Poincare AdS_5, z the radial coordinate
COORDS = sp.symbols("t x y w z", positive=True)


def ricci(g, X):
    """Ricci tensor R_mn = d_r Gam^r_mn - d_n Gam^r_mr + Gam Gam - Gam Gam."""
    n = len(X)
    gi = g.inv()
    Gam = [
        [
            [
                sp.simplify(
                    sum(
                        gi[r, s]
                        * (
                            sp.diff(g[s, m], X[k])
                            + sp.diff(g[s, k], X[m])
                            - sp.diff(g[m, k], X[s])
                        )
                        for s in range(n)
                    )
                    / 2
                )
                for k in range(n)
            ]
            for m in range(n)
        ]
        for r in range(n)
    ]
    R = sp.zeros(n)
    for m in range(n):
        for k in range(n):
            R[m, k] = sp.simplify(
                sum(sp.diff(Gam[r][m][k], X[r]) for r in range(n))
                - sum(sp.diff(Gam[r][m][r], X[k]) for r in range(n))
                + sum(Gam[r][r][s] * Gam[s][m][k] for r in range(n) for s in range(n))
                - sum(Gam[r][k][s] * Gam[s][m][r] for r in range(n) for s in range(n))
            )
    return R


def theory(name, sig, c_EH, kin, gauge, sc, P, X_of, phi0):
    """One supergravity Lagrangian, e^-1 L = c_EH R + kin(phi, (dphi)^2) + P(phi)
    - gauge(phi)/4 F_th^2, in signature sig (+1 mostly plus, -1 mostly minus),
    with F_th the theory's field of structure constant sc, so F_th = F/sc in
    terms of the unit-structure-constant F.  Returns L^2, alpha and the
    linearised scalar equation in mostly-plus language."""
    phi0 = sp.sympify(phi0)
    Ls = sp.Symbol("L", positive=True)
    t, x, y, w, z = COORDS
    gplus = sp.diag(-1, 1, 1, 1, 1) * Ls**2 / z**2
    g = sig * gplus
    Ric = ricci(g, COORDS)
    # vacuum: c R_mn = -(1/3) P g_mn (trace of the Einstein equation, D = 5)
    p = sp.Symbol("p")
    dP = sp.diff(P(p), p).subs(p, phi0)
    vac = sp.simplify(c_EH * Ric + P(phi0) / 3 * g)
    L2 = sp.solve(vac[1, 1], Ls**2)
    L2 = [v for v in L2 if v.is_positive][0]
    check(
        f"[3] {name}: dP = 0 at the vacuum, Einstein equations solved by AdS_5 with L^2 = {L2}",
        sp.simplify(dP) == 0
        and vac.subs(Ls**2, L2).applyfunc(sp.simplify) == sp.zeros(D),
    )
    # alpha: EH coefficient in mostly-plus language c_EH^+ = sig c_EH = 1/(2 kappa^2);
    # -(gauge/4) F_th^2 = -(gauge/(4 sc^2)) F^2, F^2 signature-blind, so 1/ghat^2 = gauge/sc^2
    kappa2 = 1 / (2 * sig * sp.sympify(c_EH))
    alpha = sp.simplify(kappa2 * gauge(phi0) / sc**2)
    # Euler-Lagrange for phi(z, x) with F_mn F^mn = Phi(z, x) external (unit structure constant)
    ph = sp.Function("phi")(z, x)
    Phi = sp.Function("Phi")(z, x)
    gi = g.inv()
    sqrtg = sp.simplify(sp.sqrt(sp.Abs(g.det())))
    dphi2 = gi[4, 4] * sp.diff(ph, z) ** 2 + gi[1, 1] * sp.diff(ph, x) ** 2
    dens = sqrtg * (kin(ph, dphi2) + P(ph) - gauge(ph) / (4 * sc**2) * Phi)
    EL = euler_equations(dens, [ph], [z, x])[0].lhs
    eps_, d_ = sp.symbols("epsilon"), sp.Function("delta")(z, x)
    Phi1 = sp.Function("Phi1")(z, x)
    lin = sp.diff(EL.subs({ph: phi0 + eps_ * d_, Phi: eps_ * Phi1}).doit(), eps_).subs(
        eps_, 0
    )
    lin = sp.simplify(lin / sqrtg)  # the EL expression is a density
    # dX = X'(phi0) delta; box_+ computed from the mostly-plus metric
    dX = sp.diff(X_of(p), p).subs(p, phi0) * d_
    gpi = gplus.inv()
    sg = sp.sqrt(-gplus.det())
    box = (
        sp.diff(sg * gpi[4, 4] * sp.diff(dX, z), z)
        + sp.diff(sg * gpi[1, 1] * sp.diff(dX, x), x)
    ) / sg

    def tgt(m2L2, src):  # (box - m^2) dX + src alpha F^2
        return (
            (box - m2L2 / L2 * dX + src * alpha * Phi1)
            .subs(Ls**2, L2)
            .subs(Ls, sp.sqrt(L2))
        )

    target = tgt(MASS2_L2, SOURCE_PER_ALPHA)
    lin = lin.subs(Ls**2, L2).subs(Ls, sp.sqrt(L2))
    cst = sp.simplify(sp.expand(lin).coeff(Phi1) / sp.expand(target).coeff(Phi1))
    ok = cst != 0 and not cst.has(z, x) and sp.simplify(lin - cst * target) == 0
    check(
        f"[4] {name}: linearised scalar EL equation = const x [(box + 4/L^2) dX + (alpha/6) F^2]",
        ok,
        f"(const = {cst})",
    )
    # negative controls: a mass term of the wrong sign and the source
    # -(1/12) F^a F^a in the LPT normalisation, separately and together
    wrong = []
    # "-(1/12) F^a F^a" read in the unit-structure-constant F
    lpt_src = 1 / (12 * alpha)
    for m2L2, src in ((4, lpt_src), (4, SOURCE_PER_ALPHA), (MASS2_L2, lpt_src)):
        tw = tgt(m2L2, src)
        cw = sp.simplify(sp.expand(lin).coeff(Phi1) / sp.expand(tw).coeff(Phi1))
        wrong.append(sp.simplify(lin - cw * tw) != 0)
    check(
        f"[4] {name}: negative controls -- (box - 4/L^2) and a source -(1/12) F^a F^a both fail",
        all(wrong),
    )
    return types.SimpleNamespace(
        L2=L2, alpha=alpha, a_over_L2=sp.simplify(alpha / L2), lin=lin
    )


def main():
    # -----------------------------------------------------------------------
    # [1] trace reversal, abstract and by explicit components
    # -----------------------------------------------------------------------
    kappa2, ghat2, L = sp.symbols("kappa2 ghat2 L", positive=True)
    alpha = kappa2 / ghat2
    R, F2 = sp.symbols("R F2")
    # (1/2k^2)[R_MN - (R + 12/L^2) g/2] = (1/ghat^2)[Q_MN/2 - F2 g/8],  Q_MN = F_MP F_N^P
    Rsol = sp.solve(
        sp.Eq(
            R - sp.Rational(D, 2) * (R + 12 / L**2),
            alpha * (F2 - sp.Rational(D, 4) * F2),
        ),
        R,
    )[0]
    coef_g = sp.simplify(sp.Rational(1, 2) * (Rsol + 12 / L**2) - alpha * F2 / 4)
    c = sp.simplify(-(coef_g + 4 / L**2) / alpha / F2)
    check(
        "[1] R_MN + 4/L^2 g_MN = alpha (F_MP F_N^P - c F^2 g_MN) with c = 1/6",
        c == sp.Rational(1, 6),
    )

    random.seed(1)

    def randmat(n, sym=True):
        M = sp.Matrix(
            n, n, lambda i, j: sp.Rational(random.randint(-5, 5), random.randint(1, 4))
        )
        return (M + M.T) / 2 if sym else M

    g = randmat(D)
    while g.det() == 0:
        g = randmat(D)
    gi = g.inv()
    F = randmat(D, sym=False)
    F = F - F.T
    Q = F * gi * F.T
    F2s = sum(Q[i, j] * gi[i, j] for i in range(D) for j in range(D))
    k, Lv = sp.Rational(3, 7), sp.Rational(2, 3)
    S = k * (Q - F2s * g / 4) + 6 / Lv**2 * g  # R_MN - R g/2
    trS = sum(S[i, j] * gi[i, j] for i in range(D) for j in range(D))
    Rs = trS / (1 - sp.Rational(D, 2))
    Ric = S + Rs * g / 2
    resid = (Ric + 4 / Lv**2 * g) - k * (Q - F2s * g / 6)
    check(
        "[1] explicit 5x5 component identity on a random metric and F",
        resid.applyfunc(sp.simplify) == sp.zeros(D),
    )

    # -----------------------------------------------------------------------
    # [2] Lu-Pope-Tran normalisation
    # -----------------------------------------------------------------------
    g1, g2, X, gg = sp.symbols("g1 g2 X g", positive=True)
    V = 2 * g2 * (g2 * X**2 + 2 * sp.sqrt(2) * g1 / X)  # LPT eq (7), + sign in L
    Vs = V.subs({g1: gg, g2: sp.sqrt(2) * gg})
    X0 = sp.solve(sp.diff(V, X), X)
    check(
        "[2] potential extremum at X0^3 = sqrt2 g1/g2 (X0 = 1 for g2 = sqrt2 g1)",
        len(X0) == 1 and sp.simplify(X0[0] ** 3 - sp.sqrt(2) * g1 / g2) == 0,
    )
    check(
        "[2] value at X = 1 is 12 g^2  =>  L = 1/g",
        sp.simplify(Vs.subs(X, 1) - 12 * gg**2) == 0,
    )
    # Ricci coefficient of the vacuum: L = R + V gives R_MN - (1/2) R g_MN = (1/2) V g_MN,
    # so R_MN = c g_MN with c derived from the trace in D = 5.
    cR = sp.Symbol("c")
    c_vac = sp.solve(sp.Eq(cR - sp.Rational(5, 2) * cR, Vs / 2), cR)[0]
    check(
        "[2] eq (10) Ricci coefficient -(4/3)g^2(X^2 + 2/X) = -V/3 from the potential, "
        "-4 g^2 = -4/L^2 at X = 1",
        sp.simplify(c_vac + sp.Rational(4, 3) * gg**2 * (X**2 + 2 / X)) == 0
        and sp.simplify(c_vac.subs(X, 1) + 4 * gg**2) == 0,
    )
    # kinetic: -(1/2) X^-2 *F^F = -(1/4) X^-2 F_mn F^mn ; unit EH => 2 kappa^2 = 1 ; ghat^2 = g2^2 after A -> A/g2
    kappa2_LPT, ghat2_LPT = sp.Rational(1, 2), g2**2
    alpha_LPT = sp.simplify(
        (kappa2_LPT / ghat2_LPT).subs(g2, sp.sqrt(2) * gg).subs(gg, 1 / L)
    )
    check(
        "[2a] Lagrangian route: alpha = 1/(2 g2^2) = L^2/4",
        sp.simplify(alpha_LPT - L**2 / 4) == 0,
    )
    alpha_b = sp.simplify((1 / (2 * g2**2)).subs(g2, sp.sqrt(2) / L))
    check(
        "[2b] Einstein-equation route (eq 10 coefficient 1/2 -> 1/(2 g2^2)): L^2/4",
        sp.simplify(alpha_b - L**2 / 4) == 0,
    )
    # invariance under the g1/g2 split
    X0g = (sp.sqrt(2) * g1 / g2) ** sp.Rational(1, 3)
    V0 = sp.simplify(V.subs(X, X0g))  # = 12/L^2
    L2 = sp.simplify(12 / V0)
    ghat2_gen = g2**2 * X0g**2  # X^-2 F^2 kinetic at X0, then A -> A/g2
    alpha_gen = sp.simplify(sp.Rational(1, 2) / ghat2_gen / L2)
    check(
        "[2c] alpha/L^2 = 1/4 for ANY g1/g2 split (X = 1 is a normalisation, not an input)",
        sp.simplify(alpha_gen - sp.Rational(1, 4)) == 0,
        f"alpha/L^2 = {alpha_gen}",
    )
    # the rescaling to unit structure constant, by components: F_th = dA + s eps A A at A = A~/s
    s = sp.Symbol("s", positive=True)
    dA, AA = sp.symbols("dA AA")  # dA~ and eps A~ A~ components
    check(
        "[2] F_th(A~/s) = (dA~ + eps A~A~)/s, so F_th^2 = F^2/s^2 (LPT: F_LPT^2 = F^2 L^2/2 = 2 alpha F^2)",
        sp.simplify((dA / s + s * AA / s**2) - (dA + AA) / s) == 0
        and sp.simplify((1 / g2**2).subs(g2, sp.sqrt(2) / L) - 2 * L**2 / 4) == 0,
    )

    # -----------------------------------------------------------------------
    # [3] + [4] the three Lagrangians through one machinery
    # -----------------------------------------------------------------------
    m, gR = sp.symbols("m g_R", positive=True)
    LPT = theory(
        "LPT (g1 = g, g2 = sqrt2 g)",
        +1,
        1,
        lambda X_, d2: -3 * X_**-2 * d2,
        lambda X_: X_**-2,
        sp.sqrt(2) * gg,
        lambda X_: 4 * gg**2 * (X_**2 + 2 / X_),
        lambda X_: X_,
        1,
    )
    GV = theory(
        "Gauntlett-Varela (D = 11 reduction)",
        +1,
        1,
        lambda X_, d2: -3 * X_**-2 * d2,
        lambda X_: X_**-2,
        sp.sqrt(2) * m,
        lambda X_: 4 * m**2 * (X_**2 + 2 / X_),
        lambda X_: X_,
        1,
    )
    xi = lambda f: sp.exp(sp.sqrt(sp.Rational(2, 3)) * f)  # noqa: E731
    RO = theory(
        "Romans via Ohl-Uhlemann (mostly minus, EH -R/4)",
        -1,
        -sp.Rational(1, 4),
        lambda f, d2: sp.Rational(1, 2) * d2,
        lambda f: xi(f) ** 2,
        gR,
        lambda f: gR**2 / 8 * (xi(f) ** -2 + 2 * xi(f)),
        lambda f: 1 / xi(f),
        0,
    )
    check(
        "[3] LPT: L^2 = 1/g^2, alpha = L^2/4",
        sp.simplify(LPT.L2 - 1 / gg**2) == 0 and LPT.a_over_L2 == sp.Rational(1, 4),
    )
    check(
        "[3a] Gauntlett-Varela: L^2 = 1/m^2, alpha = 1/(4 m^2) = L^2/4",
        sp.simplify(GV.L2 - 1 / m**2) == 0 and GV.a_over_L2 == sp.Rational(1, 4),
    )
    check(
        "[3b] Romans / Ohl-Uhlemann: L^2 = 8/g^2, alpha = 2/g^2 = L^2/4",
        sp.simplify(RO.L2 - 8 / gR**2) == 0
        and sp.simplify(RO.alpha - 2 / gR**2) == 0
        and RO.a_over_L2 == sp.Rational(1, 4),
    )
    # GV below (2.14): Romans from GV by xi = 1/X, g2 = -2 sqrt2 m (and g1 = -2m):
    # our L^2 then agree, and the canonical dilatons phi_GV = 2 phi_Romans
    phiG = sp.Symbol("phi_G")
    check(
        "[3b] map Romans <-> GV/LPT: g_R^2 = 8 m^2 gives equal L^2; xi = 1/X with X = exp(-phi_G/sqrt6), phi_G = 2 phi_R",
        sp.simplify(RO.L2.subs(gR, 2 * sp.sqrt(2) * m) - GV.L2) == 0
        and sp.simplify(xi(phiG / 2) - 1 / sp.exp(-phiG / sp.sqrt(6))) == 0,
    )
    ALPHA_SOURCE = sp.Rational(1, 6)  # (box + 4/L^2) dX = -(alpha/6) F^a F^a
    SCALAR_MASS2_L2 = -4

    log("")
    log("SUMMARY")
    log(
        "  alpha_SUSY = kappa^2/ghat^2 = L^2/4 = 1/4  (L = 1) for Romans' SU(2), from LPT, GV and Romans/OU; inside the unstable window 1/4 < 2/3"
    )
    log(
        "  X == 1 is not a consistent truncation with SU(2) flux: (box + 4/L^2) dX = -(alpha/6) F^a F^a = -(1/24) F^a F^a at L = 1"
    )
    log(
        "  (mostly plus; unit structure constant; = -(1/12) F_LPT^2; m^2 L^2 = -4, Delta = 2)"
    )
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        raise SystemExit(1)
    log("ALL CHECKS PASSED")
    return types.SimpleNamespace(**locals())


if __name__ == "__main__":
    main()
