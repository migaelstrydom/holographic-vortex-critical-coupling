"""d = 3 anchor of the map C_J/C_T = (d-1)(d-2)/(2d(d+1)) alpha  (= alpha/12 at d = 3).

Bulk side:   minimal N = 2 gauged supergravity in D = 4 (Einstein-Maxwell + Lambda,
             graviphoton gauging U(1)_R).  alpha_SUSY read off in the project
             normalisation  S = (1/2kappa^2) int sqrt(-g)[R + 6/L^2 - (alpha/2) F^2]
             with A normalised so that the gravitino / supercurrent has R-charge 1.
             Two sources: Caldarelli-Klemm hep-th/9808097 eqs (25),(26);
             Gauntlett-Varela 0707.2315 eqs (2.3),(2.5).
Field side:  3d N = 2 superconformal Ward identity (Closset-Dumitrescu-Festuccia-
             Komargodski 1212.3388 eq (8.3); Nishioka-Yonekura 1303.1522 (2.11),(2.12))
             and an explicit Wick-contraction computation for the free chiral multiplet
             (complex scalar R = 1/2, 2-component Dirac fermion R = -1/2), compared with
             Osborn-Petkou hep-th/9307010 eqs (5.3),(5.5),(5.6).
Cross-check: Wong 1307.7839 eq (2.1): gamma = 2 e^2 L^2/kappa^2 = 2/alpha.

Every number is asserted.  Run from the project root:  uv run python -m backreaction.derivations.central_charge_anchor_d3
"""

import sympy as sp

PASS = []


def check(name, ok, detail=""):
    PASS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def main():
    # ---------------------------------------------------------------------------
    # [1] The two holographic normalisations at d = 3 (paper sec. 4.4) -> C_J/C_T = alpha/12
    # ---------------------------------------------------------------------------
    print("[1] holographic C_T, C_J at d = 3")
    d = sp.Symbol("d", positive=True)
    alpha, kappa2, g2, L, G4 = sp.symbols("alpha kappa2 g2 L G_4", positive=True)
    C_T_hol = (
        (d + 1)
        / (d - 1)
        * sp.gamma(d + 1)
        / (sp.pi ** (d / 2) * sp.gamma(d / 2))
        * L ** (d - 1)
        / kappa2
    )
    C_J_hol = (
        (d - 2)
        * sp.gamma(d)
        / (2 * sp.pi ** (d / 2) * sp.gamma(d / 2))
        * L ** (d - 3)
        / g2
    )
    ratio_general = sp.simplify((C_J_hol / C_T_hol).subs(g2, kappa2 / alpha))
    check(
        "general d: C_J/C_T = (d-1)(d-2)/(2d(d+1)) alpha / L^2",
        sp.simplify(
            ratio_general - (d - 1) * (d - 2) / (2 * d * (d + 1)) * alpha / L**2
        )
        == 0,
    )
    CT3 = sp.simplify(C_T_hol.subs(d, 3))
    CJ3 = sp.simplify(C_J_hol.subs(d, 3))
    check(
        "d = 3: C_T = 24 L^2/(pi^2 kappa^2) = 3 L^2/(pi^3 G_4)",
        sp.simplify(CT3 - 24 * L**2 / (sp.pi**2 * kappa2)) == 0
        and sp.simplify(CT3.subs(kappa2, 8 * sp.pi * G4) - 3 * L**2 / (sp.pi**3 * G4))
        == 0,
    )
    check("d = 3: C_J = 2/(pi^2 g^2)", sp.simplify(CJ3 - 2 / (sp.pi**2 * g2)) == 0)
    check(
        "d = 3: C_J/C_T = alpha/12 (L = 1)",
        sp.simplify((CJ3 / CT3).subs(g2, kappa2 / alpha).subs(L, 1) - alpha / 12) == 0,
    )
    # ABJM sanity for C_T: L^2/G_4 = (2 sqrt2/3) k^{1/2} N^{3/2}  ->  C_T/C_T^{real scalar} = 64 sqrt2/(3 pi) k^{1/2} N^{3/2}
    N, k = sp.symbols("N k", positive=True)
    CT_scalar_3 = sp.Rational(3, 32) / sp.pi**2  # derived in [2] below; OP (5.5)
    abjm = sp.simplify(
        (CT3 / CT_scalar_3)
        .subs(kappa2, 8 * sp.pi * G4)
        .subs(L**2, (2 * sp.sqrt(2) / 3) * sp.sqrt(k) * N ** sp.Rational(3, 2) * G4)
    )
    check(
        "sanity: ABJM C_T/C_T^scalar = 64 sqrt2/(3 pi) k^{1/2} N^{3/2} (Chester-Pufu et al value)",
        sp.simplify(
            abjm - 64 * sp.sqrt(2) / (3 * sp.pi) * sp.sqrt(k) * N ** sp.Rational(3, 2)
        )
        == 0,
    )

    # ---------------------------------------------------------------------------
    # [2] Free fields in d = 3 by explicit Wick contraction
    # ---------------------------------------------------------------------------
    print(
        "[2] free-field C_J, C_T in d = 3 by Wick contraction (Euclidean, OP conventions)"
    )
    X = sp.symbols("x1 x2 x3", real=True)
    Y = sp.symbols("y1 y2 y3", real=True)
    r2 = sum((xi - yi) ** 2 for xi, yi in zip(X, Y, strict=True))
    S3 = 4 * sp.pi  # OP: S_d = 2 pi^{d/2}/Gamma(d/2)
    G = 1 / (S3 * sp.sqrt(r2))  # OP (5.3): <phi phi> = 1/((d-2) S_d x^{d-2})
    sig = [
        sp.Matrix([[0, 1], [1, 0]]),
        sp.Matrix([[0, -sp.I], [sp.I, 0]]),
        sp.Matrix([[1, 0], [0, -1]]),
    ]
    Sf = sum((sig[i] * (X[i] - Y[i]) for i in range(3)), sp.zeros(2, 2)) / (
        S3 * r2 ** sp.Rational(3, 2)
    )  # OP (5.3)

    pt = {
        X[0]: 1,
        X[1]: 2,
        X[2]: 3,
        Y[0]: 0,
        Y[1]: 0,
        Y[2]: 0,
    }  # generic rational point, |x|^2 = 14
    x2 = sp.Integer(14)

    def I_mn(m, n):
        xv = [1, 2, 3]
        return sp.KroneckerDelta(m, n) - 2 * sp.Rational(xv[m] * xv[n], 14)

    def I_T(m, n, r, s):
        return sp.Rational(1, 2) * (
            I_mn(m, r) * I_mn(n, s) + I_mn(m, s) * I_mn(n, r)
        ) - sp.Rational(1, 3) * sp.KroneckerDelta(m, n) * sp.KroneckerDelta(r, s)

    def D(expr, var, idx):
        """derivative wrt component idx of point var ('x' or 'y')"""
        v = X[idx] if var == "x" else Y[idx]
        return sp.diff(expr, v)

    # --- complex scalar, charge q: J_m = i q (phi^* d_m phi - d_m phi^* phi)
    # <J_m(x) J_n(0)> = 2 q^2 [d_m G d_n G - G d_m d_n G]  (Wick, G = <phi phi^*>)
    def scalar_JJ(m, n):
        return sp.simplify(
            (2 * (D(G, "x", m) * D(G, "x", n) - G * D(D(G, "x", m), "x", n))).subs(pt)
        )

    CJ_s = {}
    for m, n in [(0, 1), (0, 0), (2, 2)]:
        CJ_s[(m, n)] = sp.nsimplify(scalar_JJ(m, n) / (I_mn(m, n) / x2**2))
    check(
        "complex scalar (charge 1): C_J = 1/(8 pi^2) = 2/((d-2) S_d^2), same from 3 components  [OP (5.5), N_phi = 2]",
        all(sp.simplify(v - 1 / (8 * sp.pi**2)) == 0 for v in CJ_s.values()),
        str(CJ_s[(0, 1)]),
    )
    CJ_scalar = sp.simplify(CJ_s[(0, 1)])  # the Wick-contraction value checked above

    # --- real scalar improved stress tensor, xi = (d-2)/(4(d-1)) = 1/8:
    # T_mn = d_m phi d_n phi - (1/2) delta (d phi)^2 - xi (d_m d_n - delta d^2) phi^2
    # represent T as sum of c * (D_a phi)(D_b phi), D_a a tuple of derivative indices
    def scalar_T_terms(m, n, xi):
        terms = []
        terms.append((1, (m,), (n,)))
        for l in range(3):
            terms.append((-sp.Rational(1, 2) * sp.KroneckerDelta(m, n), (l,), (l,)))
        # phi^2 -> d_m d_n (phi phi) = 2 (d_m d_n phi) phi + 2 d_m phi d_n phi
        terms.append((-xi * 2, (m, n), ()))
        terms.append((-xi * 2, (m,), (n,)))
        for l in range(3):
            terms.append((xi * sp.KroneckerDelta(m, n) * 2, (l, l), ()))
            terms.append((xi * sp.KroneckerDelta(m, n) * 2, (l,), (l,)))
        return terms

    def prop_derivs(a, b):
        """<D_a phi(x) D_b phi(y)> with G real-scalar propagator"""
        e = G
        for i in a:
            e = D(e, "x", i)
        for j in b:
            e = D(e, "y", j)
        return e

    def scalar_TT(m, n, r, s, xi):
        T1 = scalar_T_terms(m, n, xi)
        T2 = scalar_T_terms(r, s, xi)
        tot = 0
        for c1, a1, b1 in T1:
            for c2, a2, b2 in T2:
                # Wick: (a1-a2)(b1-b2) + (a1-b2)(b1-a2)
                tot += (
                    c1
                    * c2
                    * (
                        prop_derivs(a1, a2) * prop_derivs(b1, b2)
                        + prop_derivs(a1, b2) * prop_derivs(b1, a2)
                    )
                )
        return sp.nsimplify(sp.simplify(tot.subs(pt)))

    xi3 = sp.Rational(1, 8)
    CT_s = {}
    for comp in [(0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 2)]:
        CT_s[comp] = sp.nsimplify(scalar_TT(*comp, xi3) / (I_T(*comp) / x2**3))
    check(
        "real scalar (xi = 1/8): C_T = 3/(32 pi^2) = d/((d-1) S_d^2), same from 3 components  [OP (5.5), n_phi = 1]",
        all(sp.simplify(v - sp.Rational(3, 32) / sp.pi**2) == 0 for v in CT_s.values()),
        str(CT_s[(0, 1, 0, 1)]),
    )
    CT_unimp = [
        sp.nsimplify(scalar_TT(*comp, 0) / (I_T(*comp) / x2**3))
        for comp in [(0, 1, 0, 1), (0, 0, 1, 1)]
    ]
    check(
        "control: unimproved scalar (xi = 0) gives inconsistent values from two components",
        sp.simplify(CT_unimp[0] - CT_unimp[1]) != 0,
        f"{CT_unimp}",
    )
    CT_scalar = sp.simplify(CT_s[(0, 1, 0, 1)])

    # --- 2-component Dirac fermion.  <psi(x) psibar(y)> = S(x,y);  <(psibar M1 psi)(x) (psibar M2 psi)(y)>
    #     = - tr[M1 S(x,y) M2 S(y,x)]  with derivatives acting on the argument they belong to.
    def Sxy():
        return Sf

    def Syx():
        return Sf.subs(
            {X[i]: Y[i] for i in range(3)} | {Y[i]: X[i] for i in range(3)},
            simultaneous=True,
        )

    def fermion_2pt(terms1, terms2):
        """terms = list of (coef, matrix M, derivs on psibar (tuple), derivs on psi (tuple))."""
        tot = sp.zeros(1, 1)[0]
        for c1, M1, db1, dp1 in terms1:
            for c2, M2, db2, dp2 in terms2:
                A = Sxy()  # <psi(x) psibar(y)>: dp1 acts on x, db2 acts on y
                for i in dp1:
                    A = A.diff(X[i])
                for j in db2:
                    A = A.diff(Y[j])
                B = Syx()  # <psi(y) psibar(x)>: dp2 acts on y, db1 acts on x
                for j in dp2:
                    B = B.diff(Y[j])
                for i in db1:
                    B = B.diff(X[i])
                tot += -c1 * c2 * (M1 * A * M2 * B).trace()
        return sp.nsimplify(sp.simplify(tot.subs(pt)))

    # current J_m = i psibar gamma_m psi  (Euclidean i so that <JJ> is positive: hermitian gammas)
    def J_terms(m):
        return [(sp.I, sig[m], (), ())]

    CJ_f = {}
    for m, n in [(0, 1), (0, 0), (2, 2)]:
        CJ_f[(m, n)] = sp.nsimplify(
            fermion_2pt(J_terms(m), J_terms(n)) / (I_mn(m, n) / x2**2)
        )
    check(
        "Dirac fermion (2 comp., charge 1): C_J = 1/(8 pi^2) = tr(1)/S_d^2 with tr 1 = 2, same from 3 components  [OP (5.6)]",
        all(sp.simplify(v - 1 / (8 * sp.pi**2)) == 0 for v in CJ_f.values()),
        str(CJ_f[(0, 1)]),
    )
    CJ_dirac = sp.simplify(CJ_f[(0, 1)])

    # stress tensor T_mn = (1/4)[psibar g_m d_n psi - d_n psibar g_m psi + (m<->n)]
    # (Euclidean sign convention: J carries one factor i, T none, so that both C_J and C_T come out positive;
    #  the convention-independent content is the magnitude and the I_{mn,rs} tensor structure from 3 components)
    def T_terms(m, n):
        q = sp.Rational(1, 4)
        return [
            (q, sig[m], (), (n,)),
            (-q, sig[m], (n,), ()),
            (q, sig[n], (), (m,)),
            (-q, sig[n], (m,), ()),
        ]

    CT_f = {}
    for comp in [(0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 2)]:
        CT_f[comp] = sp.nsimplify(
            fermion_2pt(T_terms(comp[0], comp[1]), T_terms(comp[2], comp[3]))
            / (I_T(*comp) / x2**3)
        )
    check(
        "Dirac fermion: C_T = 3/(16 pi^2) = (d/2) tr(1)/S_d^2 with tr 1 = 2, same from 3 components  [OP (5.6)]",
        all(sp.simplify(v - sp.Rational(3, 16) / sp.pi**2) == 0 for v in CT_f.values()),
        str(CT_f[(0, 1, 0, 1)]),
    )
    CT_dirac = sp.simplify(CT_f[(0, 1, 0, 1)])
    check(
        "Dirac : real scalar = 2 : 1 in C_T (the standard 3d counting)",
        sp.simplify(CT_dirac / CT_scalar - 2) == 0,
    )
    # caution (a remark, not a check): OP (5.6) taken literally, 2^{d/2} at d = 3,
    # would give 2 sqrt2 rather than tr 1 = 2 for a two-component fermion.

    # ---------------------------------------------------------------------------
    # [3] Free N = 2 chiral multiplet: scalar R = 1/2, fermion R = -1/2
    # ---------------------------------------------------------------------------
    print("[3] free 3d N = 2 chiral multiplet")
    R_phi, R_psi = sp.Rational(1, 2), -sp.Rational(1, 2)
    C_R_chiral = R_phi**2 * CJ_scalar + R_psi**2 * CJ_dirac
    C_T_chiral = 2 * CT_scalar + CT_dirac
    check(
        "C_R(chiral) = 1/(32 pi^2) + 1/(32 pi^2) = 1/(16 pi^2)",
        sp.simplify(C_R_chiral - 1 / (16 * sp.pi**2)) == 0,
    )
    check(
        "C_T(chiral) = 2*3/(32 pi^2) + 3/(16 pi^2) = 3/(8 pi^2)",
        sp.simplify(C_T_chiral - 3 / (8 * sp.pi**2)) == 0,
    )
    ratio_chiral = sp.simplify(C_R_chiral / C_T_chiral)
    check(
        "C_R/C_T (free chiral) = 1/6",
        ratio_chiral == sp.Rational(1, 6),
        str(ratio_chiral),
    )
    # CDFKS normalisation: tau = 4 pi^2 C_J (derived in [4]); tau_ff = 1 for charge-1 chiral, tau_RR = 1/4
    check(
        "CDFKS: tau_ff(free chiral, charge 1) = 4 pi^2 (C_J^scalar + C_J^Dirac) = 1   [below (2.5)]",
        sp.simplify(4 * sp.pi**2 * (CJ_scalar + CJ_dirac) - 1) == 0,
    )
    check(
        "CDFK/NY: tau_RR(free chiral, R = 1/2) = 4 pi^2 C_R = 1/4   [1212.3388 below (8.3); 1303.1522 sec 2.2]",
        sp.simplify(4 * sp.pi**2 * C_R_chiral - sp.Rational(1, 4)) == 0,
    )
    # free vector multiplet: dual to a free chiral (sigma + dual photon = Phi, gaugino), same ratio 1/6;
    # the naive UV assignment R(gaugino) = 1 is not the superconformal one (dual photon carries R): noted, not a new number.
    # For comparison: if one (wrongly) used R = (1, 0) for a chiral, the ratio would not be universal:
    wrong = sp.simplify((1 * CJ_scalar + 0 * CJ_dirac) / C_T_chiral)
    check(
        "control: a non-superconformal assignment (R = 1, 0) gives 1/3, not 1/6 -- R = 1/2 is load-bearing",
        wrong == sp.Rational(1, 3),
    )

    # ---------------------------------------------------------------------------
    # [4] The Ward identity: CDFK 1212.3388 eq (8.3) -> C_J and C_T both in terms of tau_rr
    # ---------------------------------------------------------------------------
    print("[4] 3d N = 2 superconformal Ward identity, CDFK (8.3)")
    tau = sp.Symbol("tau_rr", positive=True)
    xs = X
    inv = 1 / sum(xi**2 for xi in xs)
    lap = lambda f: sum(sp.diff(f, xi, 2) for xi in xs)
    Pop = lambda m, n, f: sp.KroneckerDelta(m, n) * lap(f) - sp.diff(f, xs[m], xs[n])
    pt0 = {X[0]: 1, X[1]: 2, X[2]: 3}
    jj = {
        (m, n): sp.simplify(tau / (16 * sp.pi**2) * Pop(m, n, inv)).subs(pt0)
        for (m, n) in [(0, 1), (0, 0)]
    }
    CJ_from_tau = {kk: sp.nsimplify(v / (I_mn(*kk) / x2**2)) for kk, v in jj.items()}
    check(
        "<jj> of (8.3): C_J = tau_rr/(4 pi^2)",
        all(sp.simplify(v - tau / (4 * sp.pi**2)) == 0 for v in CJ_from_tau.values()),
    )

    def TT_cdfk(m, n, r, s):
        f = inv
        t1 = -tau / (64 * sp.pi**2) * Pop(m, n, Pop(r, s, f))
        t2 = tau / (64 * sp.pi**2) * (Pop(m, r, Pop(n, s, f)) + Pop(n, r, Pop(m, s, f)))
        return sp.simplify((t1 + t2).subs(pt0))

    CT_from_tau = {
        c: sp.nsimplify(TT_cdfk(*c) / (I_T(*c) / x2**3))
        for c in [(0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 2)]
    }
    check(
        "<TT> of (8.3): C_T = 3 tau_rr/(2 pi^2), same from 3 components",
        all(
            sp.simplify(v - 3 * tau / (2 * sp.pi**2)) == 0 for v in CT_from_tau.values()
        ),
        str(CT_from_tau[(0, 1, 0, 1)]),
    )
    ratio_ward = sp.simplify((tau / (4 * sp.pi**2)) / (3 * tau / (2 * sp.pi**2)))
    check(
        "Ward identity: C_R/C_T = 1/6 in EVERY 3d N = 2 SCFT",
        ratio_ward == sp.Rational(1, 6),
    )
    check(
        "free chiral reproduces (8.3): tau_rr = 1/4 gives C_T = 3/(8 pi^2) = the Wick value",
        sp.simplify(
            (3 * tau / (2 * sp.pi**2)).subs(tau, sp.Rational(1, 4)) - C_T_chiral
        )
        == 0,
    )
    # Nishioka-Yonekura (2.11) tau_RR = 4F/pi^2 and (2.12) C_T = 6F/pi^4  =>  C_T = 3 tau/(2 pi^2): same relation
    F1 = sp.Symbol("F")
    check(
        "NY (2.11)+(2.12): C_T/tau_RR = (6/pi^4)/(4/pi^2) = 3/(2 pi^2), identical to (8.3)",
        sp.simplify((6 * F1 / sp.pi**4) / (4 * F1 / sp.pi**2) - 3 / (2 * sp.pi**2))
        == 0,
    )

    # ---------------------------------------------------------------------------
    # [5] Bulk: alpha_SUSY(d = 3) from two sources
    # ---------------------------------------------------------------------------
    print("[5] alpha_SUSY in D = 4 minimal gauged supergravity")
    ell, n = sp.symbols("ell n", positive=True)
    # Caldarelli-Klemm (25): e^{-1} L = -(1/4) R + (1/4) F_mn F^mn - 3/(2 ell^2) + fermions, signature (-,+,+,+),
    # i.e. L = -(1/4)[R + 6/ell^2 - F^2]: overall sign convention, relative Maxwell coefficient -1 (geometrised F^2).
    # (26): D_m = nabla_m - i ell^{-1} A_m  => gravitino charge 1/ell on the dimensionless A.
    # Our form: (1/2kappa^2)[R + 6/L^2 - (alpha/2) F_can^2] with A_can = A/ell (unit gravitino charge).
    F2_coeff_CK = 1  # coefficient of -F^2 relative to R in (25)
    charge_CK = 1 / ell  # from (26)
    alpha_CK = sp.simplify(
        2 * F2_coeff_CK / (charge_CK * ell) ** 2 * ell**2
    )  # alpha/2 = c * (A/A_can)^2 = c/(q ell... )
    # generic: A = A_can/q  =>  -c F^2 = -(c/q^2) F_can^2  =>  alpha = 2c/q^2
    alpha_CK = sp.simplify(2 * F2_coeff_CK / charge_CK**2)
    check(
        "Caldarelli-Klemm (25),(26): alpha_SUSY = 2 ell^2 -> 2 at ell = L = 1",
        sp.simplify(alpha_CK - 2 * ell**2) == 0,
    )
    # holographic BPS consistency: RN-AdS with f = 1 - 2m/r + q^2/r^2 + r^2/ell^2, A_t = q/r (geometrised);
    # E = m/G, R-charge under A_can = (1/g^2) flux = ell q/(n G) for gravitino charge n/ell; BPS m = |q| (Romans; CK ref [14])
    # => Delta = E ell = n R.  CFT chiral primary Delta = R needs n = 1: CK's (26) has n = 1.
    m_, q_ = sp.symbols("m q", positive=True)
    inv_g2 = ell**2 / (
        4 * sp.pi * G4 * n**2
    )  # -(1/16 pi G) F^2 = -(1/4g^2) F_can^2, A_can = (n/ell) A
    Rcharge = (
        inv_g2 * 4 * sp.pi * (n / ell) * q_
    )  # Gauss law: (1/g^2) * flux of E_can = (n/ell) q/r^2
    Delta = (m_ / G4) * ell
    check(
        "BPS check: Delta/R = n for m = q; n = 1 <=> Delta = R (3d N = 2 chiral primary bound)",
        sp.simplify((Delta / Rcharge).subs(m_, q_) - n) == 0,
    )
    # Gauntlett-Varela (2.3): ds^2_11 = (1/4) ds^2_4 + (dpsi + sigma + A/4)^2 + ds^2(M6); (2.5): R_mn = -3 g_mn + (1/2) F F - (1/8) g F^2
    # Our Ricci form at d = 3 (trace-reversed, from [1] of dk_central_charges): R_mn + 3 g_mn = alpha (F_mp F_n^p - F^2 g_mn/4)
    R, F2, FF = sp.symbols("R F2 FF")
    Dd = 4
    d3 = 3
    Rsol = sp.solve(
        sp.Eq(R - Dd * R / 2 - Dd * d3 * (d3 - 1) / 2, alpha * (F2 - Dd * F2 / 4)), R
    )[0]
    g_coeff = sp.simplify(Rsol / 2 + sp.Rational(d3 * (d3 - 1), 2) - alpha * F2 / 4)
    check(
        "Ricci form at d = 3: R_mn + 3 g_mn = alpha (FF - F^2 g/4)",
        sp.simplify(g_coeff - (-3 - alpha * F2 / 4)) == 0,
    )
    alpha_GV_field = sp.Rational(
        1, 2
    )  # (2.5): coefficient of FF, and -(1/8) = -(1/2)(1/4) consistent
    check(
        "GV (2.5): FF coefficient 1/2 and g F^2 coefficient -1/8 = -(1/2)/4 -> alpha = 1/2 in GV's A",
        sp.Rational(-1, 8) == -alpha_GV_field / 4,
    )
    # charge normalisation: SE7 with Ric = 6g, KE6 Ric = 8g, dsigma = 2J: the holomorphic (4,0)-form has charge 4 under d/dpsi
    # (S^7 = cone over C^4: Omega = dz1..dz4, xi = sum d/dphi_i); Killing spinor = half of that, charge 2;
    # A enters as (dpsi + sigma + A/4), so the gravitino has charge 2 * (1/4) = 1/2 under A: A_can = A/2.
    q_grav_GV = sp.Rational(2, 4)
    alpha_GV = alpha_GV_field / q_grav_GV**2
    check(
        "GV: gravitino charge 1/2 under A -> A_can = A/2 -> alpha_SUSY = (1/2) * 4 = 2",
        alpha_GV == 2,
    )
    # consistency of the same charge map on a chiral primary: z_i (charge 1 under d/dpsi) -> R = 1/4 * 2 = 1/2 = Delta(z_i)
    check(
        "GV charge map on z_i: R = 1/2 = Delta of the ABJM/mesonic chiral primary",
        sp.Rational(1, 4) / q_grav_GV == sp.Rational(1, 2),
    )
    alpha_SUSY3 = 2
    check(
        "both bulk readings agree: alpha_SUSY(d = 3) = 2",
        alpha_CK.subs(ell, 1) == alpha_SUSY3 and alpha_GV == alpha_SUSY3,
    )

    # ---------------------------------------------------------------------------
    # [6] The comparison
    # ---------------------------------------------------------------------------
    print("[6] comparison")
    pred = sp.Rational(alpha_SUSY3, 12)
    check(
        "map: alpha_SUSY/12 = 1/6 = C_R/C_T (Ward identity) = C_R/C_T (free chiral Wick)",
        pred == ratio_ward == ratio_chiral,
        f"{pred} = {ratio_ward} = {ratio_chiral}",
    )
    # holographic tau_RR from our C_J with alpha = 2 vs localisation/large-N: F(1) = pi^2 tau_rr/4 (CDFK 8.14) with F = pi L^2/(2 G_4) (FP 6.19/6.21)
    tau_hol = (
        4
        * sp.pi**2
        * CJ3.subs(g2, kappa2 / alpha).subs({alpha: 2, kappa2: 8 * sp.pi * G4})
    )
    tau_loc = 4 / sp.pi**2 * (sp.pi * L**2 / (2 * G4))
    check(
        "tau_RR: holographic 4 pi^2 C_J|_{alpha=2} = 2 L^2/(pi G_4) = 4F/pi^2 with F = pi L^2/(2G_4)  (CDFK 8.14 + FP 6.21)",
        sp.simplify(tau_hol.subs(L, 1) - tau_loc.subs(L, 1)) == 0,
        str(tau_hol.subs(L, 1)),
    )
    check(
        "(C_J/C_T)_* = 8/(d^2(d+1)) = 2/9 at d = 3 and alpha_* = 8/3",
        sp.Rational(8, 36) == sp.Rational(2, 9)
        and sp.Rational(16, 6) == sp.Rational(8, 3),
    )
    check(
        "SUSY point inside the unstable window: alpha_SUSY = 2 < alpha_* = 8/3, i.e. C_R/C_T = 1/6 < 2/9",
        sp.Rational(1, 6) < sp.Rational(2, 9),
    )

    # ---------------------------------------------------------------------------
    # [7] Wong 1307.7839 cross-check of the alpha reading (not an anchor)
    # ---------------------------------------------------------------------------
    print("[7] Wong")
    e2 = sp.Symbol("e2", positive=True)
    # (2.1): S = (1/2kappa^2) int (R + 6/L^2) - (1/4e^2) int F^a F^a  => ghat^2 = e^2, alpha = kappa^2/e^2;  gamma = 2 e^2 L^2/kappa^2
    gamma = 2 * e2 * L**2 / kappa2
    alpha_w = kappa2 / e2
    check(
        "Wong (2.1): gamma = 2 L^2/alpha", sp.simplify(gamma - 2 * L**2 / alpha_w) == 0
    )
    check(
        "gamma_* = 3/4 (Wong 2.10) <-> alpha_* = 8/3",
        sp.Rational(2, 1) / sp.Rational(3, 4) == sp.Rational(8, 3),
    )
    check(
        "alpha_SUSY = 2 <-> gamma = 1, inside Wong's unstable range gamma > 3/4",
        sp.Rational(2, 2) == 1 and 1 > sp.Rational(3, 4),
    )

    print()
    print(f"{sum(PASS)}/{len(PASS)} checks passed")
    assert all(PASS)


if __name__ == "__main__":
    main()
