"""The N = 2 Ward-identity anchor for C_J/C_T: two independent field-theory
routes give C_J(SU(2)_R)/C_T = 3/80, exactly the holographic value at Romans'
alpha_SUSY = 1/4.

What is being anchored
----------------------
paper sec. 4.4 maps the bulk coupling alpha = kappa^2/ghat^2 to the boundary
ratio C_J/C_T = (d-1)(d-2)/(2d(d+1)) alpha using two QUOTED holographic
normalisations (C_T from Buchel et al 0911.4257 eq (3.15) with action (2.1);
C_J from Freedman-Mathur-Matusis-Rastelli hep-th/9804058 eqs (30), (54)).
paper app. A.2 derives alpha_SUSY = 1/4 for the SU(2) of Romans' N = 4^+
SU(2) x U(1) gauged supergravity.  Together they predict, in d = 4,

        (C_J/C_T)_SUSY = (3/20)(1/4) = 3/80

for the current dual to Romans' SU(2) gauge field.  Romans' theory is the
universal dual of 4d N = 2 SCFTs (Gauntlett-Varela 0712.3560; the boundary
multiplet is the N = 2 Weyl multiplet, Ohl-Uhlemann 1011.3533), so that
SU(2) is the N = 2 R-symmetry SU(2)_R, and C_J(SU(2)_R)/C_T is fixed by the
N = 2 superconformal Ward identities in terms of c alone.  It is therefore a
protected, model-independent number that the bulk normalisations must
reproduce.  Two field-theory routes compute it here without any holographic
input; the comparison is made only at the end.

Conventions (all routes are translated into these before comparison)
--------------------------------------------------------------------
Osborn-Petkou hep-th/9307010 eq (2.23): <T_mn(x) T_sr(0)> = C_T I_mn,sr/x^8,
<V_m(x) V_n(0)> = C_V I_mn/x^6, I_mn = delta_mn - 2 x_m x_n/x^2 (d = 4).
Currents are normalised by the algebra: SU(2) generators T^a = sigma^a/2 on a
doublet ([J^a, J^b] = i eps^abc J^c), so doublet constituents carry
T^3 = +-1/2 -- the CFT shadow of the bulk convention F = dA + eps A A.  For a
U(1) the charge is the one the current's own algebra assigns (R(Q) = -+1 for
R-currents).  c is the Weyl^2 anomaly coefficient with c = 1/120 per real
scalar, 1/40 per Weyl fermion, 1/10 per vector, so that C_T = 40 c/pi^4
(Osborn-Petkou (8.1), (8.12); also 0911.4257 below eq (3.3)).

Checks
------
[1] DERIVED (Wick contraction, SymPy, explicit d = 4 coordinates): the free
    two-point coefficients
        C_J(complex scalar, unit charge) = 1/(4 pi^4)
        C_J(Dirac fermion, unit charge)  = 1/pi^4   (Weyl: 1/(2 pi^4))
        C_T(real scalar) = 1/(3 pi^4), C_T(Dirac) = 2/pi^4, C_T(vector) = 4/pi^4,
    against Osborn-Petkou (5.5), (5.6), (5.16) [S_4 = 2 pi^2]; scalar : Weyl
    = 1 : 2 in C_J (the one-loop beta-function weights 1/3 : 2/3, OP (8.12)
    kappa = pi^2 C_V/6); and C_T = 40 c/pi^4 field by field.

[2] ROUTE 2 (free fields, N = 4 SYM): per adjoint index, 6 real scalars,
    4 Weyl fermions, 1 vector.  The N = 2 subalgebra picks SU(2)_R x U(1)_r
    x SU(2)_F inside SU(4)_R with 4 -> (2,1)_{+1} + (1,2)_{-1}, so the
    SU(2)_R Cartan is T^3 = diag(1/2, -1/2, 0, 0) on the 4 (embedding index
    1) and 6 = wedge^2(4) -> (2,2)_0 + (1,1)_{+2} + (1,1)_{-2}: one complex
    doublet of scalars plus the neutral complex vector-multiplet scalar; the
    gaugini are the fermion doublet.  Charge^2 sums give
        C_J = 3/(8 pi^4),  C_T = 10/pi^4  (= 40c/pi^4, c = 1/4)  =>  3/80.
    Cross-checked against FMMR eq (36): B = (N^2-1)/2 per SU(4) generator
    with tr_4(T^a T^b) = delta/2, C_J = 3B/(4 pi^4) [their eq (30) in d = 4].
    C_J is linear in the embedding index, so the identification is
    load-bearing: the index-2 SU(2) (4 -> 2 + 2) would give 3/40 instead.

[3] ROUTE 1 (superconformal Ward identities):
    (a) N = 1: Osborn hep-th/9808041 eqs (3.43), (11.12), (12.1): the
        R-current two-point coefficient is C_R = 4c/pi^4, i.e. C_R/C_T = 1/10.
        Verified on the free chiral (R = 2/3, -1/3) and vector (R = 1)
        multiplets with the [1] values; independently from Anselmi-Freedman-
        Grisaru-Johansen hep-th/9708042 eq (4.3), Theta ⊃ (c/6 pi^2) V^2,
        through OP (8.1) + (8.12).
    (b) N = 2 -> N = 1: R_{N=1} = r/3 + (4/3) I_3 (verified component by
        component on the free vector multiplet and hypermultiplet), and the
        N = 1 flavour combination F = r - 2 I_3 (the one under which Q^1 is
        neutral) has <R F> = 0 (different N = 1 superconformal multiplets), so
        C_r = 8 C_{I_3} and 4c/pi^4 = C_r/9 + 16 C_{I_3}/9 give
            C_J(SU(2)_R) = C_{I_3} = 3c/(2 pi^4),   C_r = 12 c/pi^4.
    (c) Directly from the N = 2 literature: Beem et al 1312.5344 eq (3.12),
        <J^{IJ} J^{KL}> = (3c/4 pi^4) eps^{K(I} eps^{J)L} I/x^6 with the
        Ward-identity term (2i/pi^2) x x x.J^{(K(I} eps^{J)L)}/x^6, and the
        flavour convention (3.19) (long roots of length sqrt 2, i.e.
        f = eps for SU(2)).  Writing J^{IJ} = n (sigma^a eps)^{IJ} J^a and using
        the two Fierz identities computed here, the 3-point term fixes n = 1/2
        and the 2-point term then gives C_J(SU(2)_R) = 3c/(2 pi^4) again.
    All give C_J(SU(2)_R)/C_T = (3c/2)/(40c) = 3/80 for EVERY N = 2 SCFT.

[4] COMPARISON with the holographic prediction: (3/20) alpha at alpha = 1/4
    is 3/80 -- both routes agree with it exactly.  Independently, FMMR's
    N = 4 match (their eq (55), g_SG = 4 pi/N for the SU(4) gauge fields)
    with L^3/kappa^2 fixed by C_T = 10 N^2/pi^4 gives alpha_{SO(6)} = 1/4 as
    well, so the N = 8 gauge fields and Romans' SU(2) share the ratio.

[5] The U(1)_r ratio test (a check on the SU(2) identification that is blind
    to the overall Einstein-Hilbert normalisation): Lu-Pope-Tran
    hep-th/9909203 eqs (7), (8), (11) -- equal kinetic terms for the SU(2) and
    U(1) fields at X = 1, structure constant g_2 = sqrt2 g_1, and the complex
    two-form charged with charge g_1 under the U(1); the two-form is dual to
    the Delta = 3 antisymmetric tensor of the N = 2 stress-tensor multiplet
    with |r| = 2.  This gives C_r/C_{I_3} = 8 in the bulk, matching (3b).

[6] Curiosity, not used: the N = 1 R-current ratio C_R/C_T = 1/10 coincides
    numerically with (C_J/C_T)_* = 8/(d^2(d+1)) at d = 4.

Run:  uv run python -m backreaction.derivations.central_charge_anchor
"""

from __future__ import annotations

import itertools
import time
import types
from fractions import Fraction

import sympy as sp

T0 = time.time()
CHECKS: dict[str, bool] = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Wick-contraction engine for free fields in d = 4 (Euclidean)
# ---------------------------------------------------------------------------
X = sp.symbols("x1:5", real=True)


def main():
    R2 = sum(xi**2 for xi in X)
    PI = sp.pi
    G_SCALAR = 1 / (4 * PI**2 * R2)  # <phi(x) phi(0)>, OP (5.3) with S_4 = 2 pi^2

    # Euclidean gamma matrices, {g_m, g_n} = 2 delta, Hermitian (chiral basis)
    _s = [
        sp.Matrix([[0, 1], [1, 0]]),
        sp.Matrix([[0, -sp.I], [sp.I, 0]]),
        sp.Matrix([[1, 0], [0, -1]]),
    ]
    _Z = sp.zeros(2)
    _I2 = sp.eye(2)
    GAMMA = [
        sp.BlockMatrix([[_Z, -sp.I * s], [sp.I * s, _Z]]).as_explicit() for s in _s
    ] + [sp.BlockMatrix([[_Z, _I2], [_I2, _Z]]).as_explicit()]
    S_FERMION = sum((GAMMA[m] * X[m] for m in range(4)), sp.zeros(4)) / (
        2 * PI**2 * R2**2
    )
    # <psi(x) psibar(0)> = gamma.x/(S_4 x^4), OP (5.3)

    # a fixed rational point at which everything is evaluated exactly
    POINT = {
        X[0]: sp.Rational(1, 3),
        X[1]: sp.Rational(-2, 5),
        X[2]: sp.Rational(3, 7),
        X[3]: sp.Rational(1, 2),
    }

    def dmulti(expr, idx):
        for m in idx:
            expr = sp.diff(expr, X[m])
        return expr

    def I_tensor(m, n):
        return (1 if m == n else 0) - 2 * X[m] * X[n] / R2

    def I_TT(m, n, s, r):
        return sp.Rational(1, 2) * (
            I_tensor(m, s) * I_tensor(n, r) + I_tensor(m, r) * I_tensor(n, s)
        ) - sp.Rational(1, 4) * (1 if m == n else 0) * (1 if s == r else 0)

    def two_point_scalar(op1, op2, ncomp):
        """<O1(x) O2(0)> for normal-ordered bilinears in ncomp real scalars with
        <phi_i(x) phi_j(0)> = delta_ij G.  A bilinear is a list of
        (coeff, (i, A), (j, B)) meaning coeff (d^A phi_i)(d^B phi_j)."""
        total = 0
        cache = {}

        def P(A, C):
            # <d^A phi(x) d^C phi(0)> = (-1)^|C| d^{A+C} G(x)
            key = tuple(sorted(A + C))
            if key not in cache:
                cache[key] = dmulti(G_SCALAR, key)
            return (-1) ** len(C) * cache[key]

        for c1, (i, A), (j, B) in op1:
            for c2, (k, C), (l, D) in op2:
                term = 0
                if i == k and j == l:
                    term += P(A, C) * P(B, D)
                if i == l and j == k:
                    term += P(A, D) * P(B, C)
                if term != 0:
                    total += c1 * c2 * term
        return total

    def two_point_fermion(op1, op2):
        """<O1(x) O2(0)> for Dirac bilinears (coeff, A, M, B) = coeff (d^A psibar) M (d^B psi):
        = - sum coeff coeff' Tr[M S_{BC}(x) N S_{DA}] with
        S_{BC}(x) = (-1)^|C| d^{B+C} S(x),  S_{DA} = (-1)^|A| (d^{A+D} S)(-x)."""
        total = sp.zeros(1)[0]
        cache = {}

        def dS(idx):
            key = tuple(sorted(idx))
            if key not in cache:
                cache[key] = S_FERMION.applyfunc(lambda e: dmulti(e, key))
            return cache[key]

        flip = {X[m]: -X[m] for m in range(4)}
        for c1, A, M, B in op1:
            for c2, C, N, D in op2:
                S1 = (-1) ** len(C) * dS(B + C)
                S2 = (-1) ** len(A) * dS(A + D).subs(flip, simultaneous=True)
                total += -c1 * c2 * (M * S1 * N * S2).trace()
        return total

    def scalar_current(charge_pairs):
        """J_m = i sum (phi^dag d phi - d phi^dag phi) written in real components:
        for a complex scalar phi = (phi_1 + i phi_2)/sqrt2 of charge q the current is
        q (phi_1 d phi_2 - phi_2 d phi_1)  [OP (5.1) with t antisymmetric real]."""
        ops = {}
        for m in range(4):
            terms = []
            for (i1, i2), q in charge_pairs:
                terms.append((q, (i1, ()), (i2, (m,))))
                terms.append((-q, (i2, ()), (i1, (m,))))
            ops[m] = terms
        return ops

    XI_CONFORMAL = sp.Rational(1, 6)

    def scalar_stress(ncomp, xi=XI_CONFORMAL):
        """T_mn = d_m phi d_n phi - 1/2 delta (d phi)^2 - xi (d_m d_n - delta box) phi^2,
        per real component, conformal improvement xi = 1/6 in d = 4."""
        ops = {}
        for m in range(4):
            for n in range(4):
                terms = []
                for i in range(ncomp):
                    terms.append((1, (i, (m,)), (i, (n,))))
                    if m == n:
                        for p in range(4):
                            terms.append((-sp.Rational(1, 2), (i, (p,)), (i, (p,))))
                    # -xi d_m d_n (phi phi) = -xi (2 phi d_m d_n phi + 2 d_m phi d_n phi)
                    terms.append((-2 * xi, (i, ()), (i, (m, n))))
                    terms.append((-2 * xi, (i, (m,)), (i, (n,))))
                    if m == n:
                        for p in range(4):
                            terms.append((2 * xi, (i, ()), (i, (p, p))))
                            terms.append((2 * xi, (i, (p,)), (i, (p,))))
                ops[(m, n)] = terms
        return ops

    def vector_stress():
        """T_mn = F_mr F_nr - 1/4 delta F^2 for a Maxwell field, A_m as 4 real
        components with the Feynman-gauge propagator delta_mn G (T is gauge
        invariant, so the gauge choice is immaterial in the free theory)."""

        def F_terms(a, b):
            # F_ab = d_a A_b - d_b A_a  as list of (coeff, (component, derivs))
            return [(1, (b, (a,))), (-1, (a, (b,)))]

        ops = {}
        for m in range(4):
            for n in range(4):
                terms = []
                for r in range(4):
                    for c1, f1 in F_terms(m, r):
                        for c2, f2 in F_terms(n, r):
                            terms.append((c1 * c2, f1, f2))
                if m == n:
                    for a in range(4):
                        for b in range(4):
                            for c1, f1 in F_terms(a, b):
                                for c2, f2 in F_terms(a, b):
                                    terms.append((-sp.Rational(1, 4) * c1 * c2, f1, f2))
                ops[(m, n)] = terms
        return ops

    def fermion_current(t=sp.I):
        """V_m = psibar t gamma_m psi with anti-Hermitian t (OP (5.2), tr t^2 = -1)."""
        return {m: [(t, (), GAMMA[m], ())] for m in range(4)}

    def fermion_stress():
        """T_mn = 1/2 psibar (gamma_m <-> d_n + gamma_n <-> d_m) psi, <->d = (d - d_left)/2,
        OP (5.2)."""
        ops = {}
        for m in range(4):
            for n in range(4):
                terms = []
                for g, dd in ((m, n), (n, m)):
                    terms.append((sp.Rational(1, 4), (), GAMMA[g], (dd,)))
                    terms.append((-sp.Rational(1, 4), (dd,), GAMMA[g], ()))
                ops[(m, n)] = terms
        return ops

    def extract_CJ(two_pt_fn, ops):
        """C_J from <J_1 J_2> = C_J I_12/x^6 and the diagonal component, exactly."""
        v12 = two_pt_fn(ops[0], ops[1])
        v11 = two_pt_fn(ops[0], ops[0])
        c12 = sp.nsimplify(sp.simplify((v12 * R2**3 / I_tensor(0, 1)).subs(POINT)))
        c11 = sp.nsimplify(sp.simplify((v11 * R2**3 / I_tensor(0, 0)).subs(POINT)))
        return c12, c11

    def extract_CT(two_pt_fn, ops):
        v = two_pt_fn(ops[(0, 1)], ops[(0, 1)])
        w = two_pt_fn(ops[(0, 0)], ops[(1, 1)])
        c1 = sp.nsimplify(sp.simplify((v * R2**4 / I_TT(0, 1, 0, 1)).subs(POINT)))
        c2 = sp.nsimplify(sp.simplify((w * R2**4 / I_TT(0, 0, 1, 1)).subs(POINT)))
        return c1, c2

    # ---------------------------------------------------------------------------
    # [1] free-field two-point coefficients by Wick contraction
    # ---------------------------------------------------------------------------
    log("[1] free-field C_J and C_T in d = 4 by explicit Wick contraction")
    inv_pi4 = 1 / PI**4

    cj12, cj11 = extract_CJ(
        lambda a, b: two_point_scalar(a, b, 2), scalar_current([((0, 1), 1)])
    )
    CJ_CSCALAR = cj12
    check(
        "[1] C_J(complex scalar, charge 1) = 1/(4 pi^4)  [OP (5.5), N_phi = 2]",
        cj12 == inv_pi4 / 4 and cj11 == cj12,
        f"C_J = {cj12} (off-diagonal and diagonal components agree)",
    )

    fj12, fj11 = extract_CJ(two_point_fermion, fermion_current())
    CJ_DIRAC = fj12
    check(
        "[1] C_J(Dirac fermion, charge 1) = 1/pi^4  [OP (5.6), N_psi = 1]",
        fj12 == inv_pi4 and fj11 == fj12,
        f"C_J = {fj12}",
    )
    CJ_WEYL = CJ_DIRAC / 2
    check(
        "[1] C_J ratio complex scalar : Weyl = 1 : 2 (beta-function weights 1/3 : 2/3)",
        sp.simplify(CJ_WEYL / CJ_CSCALAR) == 2,
    )

    ct1, ct2 = extract_CT(lambda a, b: two_point_scalar(a, b, 1), scalar_stress(1))
    CT_SCALAR = ct1
    check(
        "[1] C_T(real conformal scalar) = 1/(3 pi^4)  [OP (5.5), d/((d-1) S_d^2)]",
        ct1 == inv_pi4 / 3 and ct2 == ct1,
        f"C_T = {ct1} (two independent components agree)",
    )
    # the improvement term is load-bearing: xi = 0 does not give a conformal <TT>
    ct_bad, ct_bad2 = extract_CT(
        lambda a, b: two_point_scalar(a, b, 1), scalar_stress(1, xi=0)
    )
    check(
        "[1] control: the unimproved scalar stress tensor is NOT of the conformal form",
        ct_bad != ct_bad2,
        f"components would give {ct_bad} vs {ct_bad2}",
    )

    ft1, ft2 = extract_CT(two_point_fermion, fermion_stress())
    CT_DIRAC = sp.Abs(ft1)
    check(
        "[1] |C_T(Dirac fermion)| = 2/pi^4  [OP (5.6), (d/2) 2^{d/2}/S_d^2]",
        CT_DIRAC == 2 * inv_pi4 and ft2 == ft1,
        f"C_T = {ft1} (sign is the Euclidean-bilinear convention; magnitude is what C_T means)",
    )

    # calibration quoted in paper section 4.4: the ratio C_J/C_T of a fully charged
    # free field in the same conventions (complex scalar: C_T = 2 C_T(real scalar);
    # Weyl fermion: C_T = C_T(Dirac)/2)
    check(
        "[1] C_J/C_T(free complex scalar, unit charge) = 3/8",
        sp.simplify(CJ_CSCALAR / (2 * CT_SCALAR)) == sp.Rational(3, 8),
        f"ratio = {sp.simplify(CJ_CSCALAR / (2 * CT_SCALAR))}",
    )
    check(
        "[1] C_J/C_T(free Weyl fermion, unit charge) = 1/2",
        sp.simplify(CJ_WEYL / (CT_DIRAC / 2)) == sp.Rational(1, 2),
        f"ratio = {sp.simplify(CJ_WEYL / (CT_DIRAC / 2))}",
    )
    CT_WEYL = CT_DIRAC / 2

    vt1, vt2 = extract_CT(lambda a, b: two_point_scalar(a, b, 4), vector_stress())
    CT_VECTOR = vt1
    check(
        "[1] C_T(Maxwell field) = 4/pi^4  [OP (5.16), 16/S_4^2]",
        vt1 == 4 * inv_pi4 and vt2 == vt1,
        f"C_T = {vt1}",
    )

    def c_anomaly(n_s, n_w, n_v):
        return Fraction(n_s, 120) + Fraction(n_w, 40) + Fraction(n_v, 10)

    def CT_of(n_s, n_w, n_v):
        return n_s * CT_SCALAR + n_w * CT_WEYL + n_v * CT_VECTOR

    check(
        "[1] C_T = 40 c/pi^4 field by field (c = 1/120, 1/40, 1/10)  [OP (8.1)+(8.12)]",
        all(
            sp.simplify(
                CT_of(*n)
                - 40
                * sp.Rational(c_anomaly(*n).numerator, c_anomaly(*n).denominator)
                * inv_pi4
            )
            == 0
            for n in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        ),
    )

    # ---------------------------------------------------------------------------
    # [2] Route 2: free N = 4 SYM and the SU(2)_R subgroup of SU(4)_R
    # ---------------------------------------------------------------------------
    log("[2] Route 2: free-field N = 4 SYM, SU(2)_R subset SU(4)_R")

    def wedge2_charges(t3):
        """T^3 eigenvalues on the 6 = wedge^2(4) from those on the 4."""
        return [t3[i] + t3[j] for i, j in itertools.combinations(range(4), 2)]

    def CJ_from_charges(t3_on_4):
        """C_J of the Cartan current for one adjoint index of N = 4 SYM: fermions
        are the 4 (Weyl), scalars the real 6 = wedge^2(4).  A complex scalar of
        charge q is two real components with tr T^2 = 2 q^2, so the scalar sum is
        (1/2) tr_6(T^2) in units of C_J(complex scalar, charge 1)."""
        tr4 = sum(q**2 for q in t3_on_4)
        tr6 = sum(q**2 for q in wedge2_charges(t3_on_4))
        return tr4 * CJ_WEYL + sp.Rational(1, 2) * tr6 * CJ_CSCALAR, tr4, tr6

    half = sp.Rational(1, 2)
    T3_R = [half, -half, 0, 0]  # SU(2)_R Cartan on the 4: (2,1)_{+1} + (1,2)_{-1}
    six = wedge2_charges(T3_R)
    check(
        "[2] 6 = wedge^2(4) -> T^3 charges {+-1/2 x2, 0 x2}: one complex doublet + two singlets",
        sorted(six) == [-half, -half, 0, 0, half, half],
        f"charges on the 6: {six}",
    )
    # U(1)_r bookkeeping of the same branching, 4 -> (2,1)_{+1} + (1,2)_{-1}
    r_on_4 = [1, 1, -1, -1]
    r_on_6 = [r_on_4[i] + r_on_4[j] for i, j in itertools.combinations(range(4), 2)]
    check(
        "[2] U(1)_r on the 6: (2,2)_0 + (1,1)_{+2} + (1,1)_{-2}  (hyper scalars neutral, phi has |r| = 2)",
        sorted(r_on_6) == [-2, 0, 0, 0, 0, 2]
        and all((r == 0) == (abs(q) == half) for r, q in zip(r_on_6, six, strict=True)),
    )

    CJ_N4, tr4, tr6 = CJ_from_charges(T3_R)
    CT_N4 = CT_of(6, 4, 1)
    check(
        "[2] per adjoint index: tr_4 T3^2 = 1/2, tr_6 T3^2 = 1  =>  C_J = 3/(8 pi^4)",
        tr4 == half and tr6 == 1 and sp.simplify(CJ_N4 - 3 * inv_pi4 / 8) == 0,
        f"C_J = {CJ_N4}  (scalars 1/(8 pi^4) + fermions 1/(4 pi^4))",
    )
    check(
        "[2] C_T = 10/pi^4 per adjoint index  (= 40c/pi^4 with c = 1/4)",
        sp.simplify(CT_N4 - 10 * inv_pi4) == 0 and c_anomaly(6, 4, 1) == Fraction(1, 4),
    )
    RATIO_ROUTE2 = sp.nsimplify(sp.simplify(CJ_N4 / CT_N4))
    log(f"      ROUTE 2:  C_J(SU(2)_R)/C_T = {RATIO_ROUTE2}")
    # FMMR eq (36): B = (N^2-1)/2 for an SU(4) generator with tr_4(T^a T^b) = delta/2;
    # eq (30) in d = 4: C_J = B 2(d-1)(d-2)/(2 pi)^d = 3B/(4 pi^4)
    B_FMMR_per_adjoint = half
    CJ_FMMR = B_FMMR_per_adjoint * 2 * 3 * 2 / (2 * PI) ** 4
    check(
        "[2] FMMR eq (36) B = (N^2-1)/2 with eq (30) gives the same C_J = 3/(8 pi^4) per adjoint index",
        sp.simplify(CJ_FMMR - CJ_N4) == 0,
    )
    # the embedding index matters
    CJ_index2, _, _ = CJ_from_charges([half, -half, half, -half])
    check(
        "[2] control: the index-2 SU(2) (4 -> 2 + 2, 6 -> 3 + 1 + 1 + 1) gives twice the value, 3/(4 pi^4), ratio 3/40",
        sp.simplify(CJ_index2 - 2 * CJ_N4) == 0
        and sp.nsimplify(CJ_index2 / CT_N4) == sp.Rational(3, 40),
        "C_J is linear in the embedding index, so the identification of Romans' SU(2) is load-bearing",
    )

    # ---------------------------------------------------------------------------
    # [3] Route 1: the superconformal Ward identities
    # ---------------------------------------------------------------------------
    log("[3] Route 1: N = 1 and N = 2 superconformal Ward identities")
    c = sp.Symbol("c", positive=True)
    CT_c = 40 * c * inv_pi4
    CR_c = 4 * c * inv_pi4  # Osborn 9808041 (11.12) with (3.43), (12.1)

    # (a) free-field verification of C_R = 4c/pi^4 (R(scalar) = 2/3, R(chiral fermion) = -1/3, R(gaugino) = 1)
    CR_chiral = sp.Rational(4, 9) * CJ_CSCALAR + sp.Rational(1, 9) * CJ_WEYL
    CR_vector = 1 * CJ_WEYL
    check(
        "[3a] N = 1 chiral multiplet: C_R = 1/(6 pi^4) = 4c/pi^4 with c = 1/24",
        sp.simplify(CR_chiral - CR_c.subs(c, sp.Rational(1, 24))) == 0,
    )
    check(
        "[3a] N = 1 vector multiplet: C_R = 1/(2 pi^4) = 4c/pi^4 with c = 1/8",
        sp.simplify(CR_vector - CR_c.subs(c, sp.Rational(1, 8))) == 0,
    )
    check(
        "[3a] C_R/C_T = 1/10 in every N = 1 SCFT",
        sp.simplify(CR_c / CT_c) == sp.Rational(1, 10),
    )
    # AFGJ (4.3): Theta ⊃ (c/6 pi^2) V_mn^2; OP (8.1): Theta ⊃ -(kappa/4) F^2 with (8.12) kappa = pi^2 C_V/6
    kappa_R = 4 * c / (6 * PI**2)
    check(
        "[3a] AFGJ eq (4.3) c/(6 pi^2) V^2 through OP (8.1), (8.12) gives the same C_R = 4c/pi^4",
        sp.simplify(6 * kappa_R / PI**2 - CR_c) == 0,
    )

    # (b) N = 2 charges of the free multiplets: (r, I_3, kind) with kind in {scalar (complex), weyl}
    VECTOR_MULT = [
        ("lambda1", 1, half, "weyl"),
        ("lambda2", 1, -half, "weyl"),
        ("phi", 2, 0, "scalar"),
    ]
    HYPER_MULT = [
        ("q1", 0, half, "scalar"),
        ("q2", 0, -half, "scalar"),
        ("psi", -1, 0, "weyl"),
        ("psi~", -1, 0, "weyl"),
    ]
    N1_R = {
        "lambda1": 1,
        "lambda2": -sp.Rational(1, 3),
        "phi": sp.Rational(2, 3),
        "q1": sp.Rational(2, 3),
        "q2": -sp.Rational(2, 3),
        "psi": -sp.Rational(1, 3),
        "psi~": -sp.Rational(1, 3),
    }

    def R_N1(r, i3):
        return sp.Rational(1, 3) * r + sp.Rational(4, 3) * i3

    check(
        "[3b] R_{N=1} = r/3 + 4 I_3/3 reproduces the N = 1 R-charges of every free N = 2 component",
        all(R_N1(r, i3) == N1_R[name] for name, r, i3, _ in VECTOR_MULT + HYPER_MULT),
        "(lambda1 -> 1, lambda2 -> -1/3, phi -> 2/3, q1 -> 2/3, q2 = q~^dag -> -2/3, hyperini -> -1/3)",
    )

    def C_of_charge(mult, charge):
        unit = {"scalar": CJ_CSCALAR, "weyl": CJ_WEYL}
        return sum(charge(r, i3) ** 2 * unit[kind] for _, r, i3, kind in mult)

    def C_cross(mult, q1, q2):
        unit = {"scalar": CJ_CSCALAR, "weyl": CJ_WEYL}
        return sum(q1(r, i3) * q2(r, i3) * unit[kind] for _, r, i3, kind in mult)

    c_vec, c_hyp = (
        Fraction(1, 6),
        Fraction(1, 12),
    )  # KT 9907107 (4.19): c = (2 n_V + n_H)/12
    check(
        "[3b] c(vector multiplet) = 1/6, c(hyper) = 1/12 from the field content  [KT (4.19)]",
        c_anomaly(2, 2, 1) == c_vec and c_anomaly(4, 2, 0) == c_hyp,
    )
    results = {}
    for label, mult, cc in (
        ("vector", VECTOR_MULT, c_vec),
        ("hyper", HYPER_MULT, c_hyp),
    ):
        ccr = sp.Rational(cc.numerator, cc.denominator)
        C_I3 = C_of_charge(mult, lambda r, i3: i3)
        C_r = C_of_charge(mult, lambda r, i3: r)
        C_R = C_of_charge(mult, R_N1)
        C_RF = C_cross(mult, R_N1, lambda r, i3: r - 2 * i3)
        C_rI = C_cross(mult, lambda r, i3: r, lambda r, i3: i3)
        results[label] = (C_I3, C_r, C_R)
        check(
            f"[3b] free {label} multiplet: C_R = 4c/pi^4, <R F> = 0 for F = r - 2 I_3, <r I_3> = 0",
            sp.simplify(C_R - 4 * ccr * inv_pi4) == 0 and C_RF == 0 and C_rI == 0,
        )
        check(
            f"[3b] free {label} multiplet: C_{{I_3}} = 3c/(2 pi^4), C_r = 12 c/pi^4, C_r/C_{{I_3}} = 8",
            sp.simplify(C_I3 - 3 * ccr * inv_pi4 / 2) == 0
            and sp.simplify(C_r - 12 * ccr * inv_pi4) == 0,
            f"C_I3 = {C_I3}, C_r = {C_r}",
        )
    # the Ward-identity algebra itself: C_R = C_r/9 + 16 C_I3/9 with C_r = 8 C_I3 and C_R = 4c/pi^4
    CI3_sym, Cr_sym = sp.symbols("C_I3 C_r", positive=True)
    sol = sp.solve(
        [sp.Eq(Cr_sym / 9 + 16 * CI3_sym / 9, CR_c), sp.Eq(Cr_sym, 8 * CI3_sym)],
        [CI3_sym, Cr_sym],
    )
    check(
        "[3b] Ward identities  =>  C_J(SU(2)_R) = 3c/(2 pi^4), C_r = 12 c/pi^4 in every N = 2 SCFT",
        sp.simplify(sol[CI3_sym] - 3 * c * inv_pi4 / 2) == 0
        and sp.simplify(sol[Cr_sym] - 12 * c * inv_pi4) == 0,
    )
    RATIO_ROUTE1 = sp.nsimplify(sp.simplify(sol[CI3_sym] / CT_c))
    log(f"      ROUTE 1:  C_J(SU(2)_R)/C_T = {RATIO_ROUTE1}")
    check(
        "[3b] N = 4 SYM (vector + hyper, c = 1/4) reproduces Route 2's C_J = 3/(8 pi^4)",
        sp.simplify(results["vector"][0] + results["hyper"][0] - CJ_N4) == 0,
    )

    # (c) Beem et al 1312.5344 eq (3.12) translated with the Fierz identities
    eps = sp.Matrix([[0, 1], [-1, 0]])
    Ms = [s * eps for s in _s]  # (sigma^a eps)^{IJ}, symmetric

    def sym_pair(f):
        """symmetrise a 4-index function in (I,J) and in (K,L)"""
        return lambda I, J, K, L: (
            (f(I, J, K, L) + f(J, I, K, L) + f(I, J, L, K) + f(J, I, L, K)) / 4
        )

    fierz2_ok = all(
        sp.simplify(
            sum(Ms[a][I, J] * Ms[a][K, L] for a in range(3))
            - 2 * (eps[K, I] * eps[J, L] + eps[K, J] * eps[I, L]) / 2
        )
        == 0
        for I, J, K, L in itertools.product(range(2), repeat=4)
    )
    check(
        "[3c] Fierz: sum_a (sigma^a eps)^{IJ} (sigma^a eps)^{KL} = 2 eps^{K(I} eps^{J)L}",
        fierz2_ok,
    )
    lam = sp.Symbol("lambda")
    eqs = []
    for cidx in range(3):
        lhs = lambda I, J, K, L, cidx=cidx: sum(
            sp.LeviCivita(a, b, cidx) * Ms[a][I, J] * Ms[b][K, L]
            for a in range(3)
            for b in range(3)
        )
        rhs = sym_pair(lambda I, J, K, L, cidx=cidx: eps[K, I] * Ms[cidx][J, L])
        for I, J, K, L in itertools.product(range(2), repeat=4):
            eqs.append(sp.expand(lhs(I, J, K, L) - lam * rhs(I, J, K, L)))
    lam_sol = [s_[lam] for s_ in sp.solve([e for e in eqs if e != 0], lam, dict=True)]
    fierz3_ok = len(lam_sol) == 1 and all(
        sp.simplify(e.subs(lam, lam_sol[0])) == 0 for e in eqs
    )
    check(
        "[3c] Fierz: sum_ab eps^{abc} (sigma^a eps)^{IJ} (sigma^b eps)^{KL} = 2i eps^{(K(I} (sigma^c eps)^{J)L)}",
        fierz3_ok and lam_sol[0] == 2 * sp.I,
        f"lambda = {lam_sol}",
    )
    # J^{IJ} = n (sigma^a eps)^{IJ} J^a with <J^a J^b> = C_J delta^{ab} I/x^6 and the Ward term
    # (2/pi^2) eps^{abc} x x x.J^c/x^6 [Beem (3.19), f = eps]:  3-pt coefficient n * lambda * 2/pi^2 = 2i/pi^2
    n_beem = sp.solve(
        sp.Eq(sp.Symbol("n") * lam_sol[0] * 2 / PI**2, 2 * sp.I / PI**2), sp.Symbol("n")
    )[0]
    CJ_beem = sp.solve(
        sp.Eq(2 * n_beem**2 * sp.Symbol("CJ"), 3 * c / (4 * PI**4)), sp.Symbol("CJ")
    )[0]
    check(
        "[3c] Beem et al (3.12): the Ward term fixes n = 1/2, the 2-pt term then gives C_J(SU(2)_R) = 3c/(2 pi^4)",
        n_beem == half and sp.simplify(CJ_beem - 3 * c * inv_pi4 / 2) == 0,
        f"n = {n_beem}, C_J = {CJ_beem}",
    )

    # ---------------------------------------------------------------------------
    # [4] the comparison with the holographic prediction
    # ---------------------------------------------------------------------------
    log(
        "[4] comparison with the holographic C_J/C_T = (3/20) alpha at alpha_SUSY = 1/4"
    )
    d, alpha, kappa2, g2, L = sp.symbols("d alpha kappa2 g2 L", positive=True)
    # 0911.4257 (2.1) 1/(2 l_P^{d-1}) => l_P^{d-1} = kappa^2; (3.15) at lambda_GB = 0:
    C_T_holo = (
        (d + 1)
        / (d - 1)
        * sp.gamma(d + 1)
        / (PI ** (d / 2) * sp.gamma(d / 2))
        * L ** (d - 1)
        / kappa2
    )
    # FMMR (30) with (54): B = (1/g^2) 2^{d-2} pi^{d/2} Gamma(d)/((d-1) Gamma(d/2)), C_J = B 2(d-1)(d-2)/(2 pi)^d
    B_fmmr = (
        1
        / g2
        * 2 ** (d - 2)
        * PI ** (d / 2)
        * sp.gamma(d)
        / ((d - 1) * sp.gamma(d / 2))
    )
    C_J_holo = sp.simplify(B_fmmr * 2 * (d - 1) * (d - 2) / (2 * PI) ** d) * L ** (
        d - 3
    )
    check(
        "[4] FMMR (30)+(54) reproduce the C_J formula of paper sec. 4.4, (d-2) Gamma(d)/(2 pi^{d/2} Gamma(d/2)) L^{d-3}/g^2",
        sp.simplify(
            C_J_holo
            - (d - 2)
            * sp.gamma(d)
            / (2 * PI ** (d / 2) * sp.gamma(d / 2))
            * L ** (d - 3)
            / g2
        )
        == 0,
    )
    ratio_holo = sp.simplify(
        sp.gammasimp((C_J_holo / C_T_holo).subs({kappa2: alpha * g2, L: 1}))
    )
    check(
        "[4] holographic C_J/C_T = (d-1)(d-2)/(2d(d+1)) alpha; d = 4: (3/20) alpha",
        sp.simplify(ratio_holo - (d - 1) * (d - 2) / (2 * d * (d + 1)) * alpha) == 0
        and sp.simplify(ratio_holo.subs(d, 4) - sp.Rational(3, 20) * alpha) == 0,
    )
    ALPHA_SUSY = sp.Rational(1, 4)  # paper sec. 4.4, derived in app. A.2
    RATIO_HOLO = sp.nsimplify(ratio_holo.subs({d: 4, alpha: ALPHA_SUSY}))
    log(f"      HOLOGRAPHY at alpha_SUSY = 1/4:  C_J/C_T = {RATIO_HOLO}")
    check("[4] Route 2 (free N = 4 SYM) = 3/80", RATIO_ROUTE2 == sp.Rational(3, 80))
    check(
        "[4] Route 1 (N = 2 Ward identities) = 3/80", RATIO_ROUTE1 == sp.Rational(3, 80)
    )
    check(
        "[4] VERDICT: both field-theory routes equal the holographic value at alpha = 1/4 -- exactly",
        RATIO_ROUTE1 == RATIO_ROUTE2 == RATIO_HOLO,
        f"route 1 = {RATIO_ROUTE1}, route 2 = {RATIO_ROUTE2}, holography = {RATIO_HOLO}",
    )
    # the inverse reading: which alpha the field theory demands
    alpha_ft = sp.solve(sp.Eq(ratio_holo.subs(d, 4), RATIO_ROUTE1), alpha)[0]
    check(
        "[4] read backwards: the N = 2 Ward identities demand alpha = 1/4 for the SU(2)_R gauge field",
        alpha_ft == ALPHA_SUSY,
    )
    # FMMR (55): g_SG = 4 pi/N for the SU(4) gauge fields of N = 8 gauged supergravity (L = 1),
    # and kappa^2 from C_T = 10 N^2/pi^4:
    Nc = sp.Symbol("N", positive=True)
    kappa2_N4 = sp.solve(
        sp.Eq(C_T_holo.subs({d: 4, L: 1}), 10 * Nc**2 / PI**4), kappa2
    )[0]
    g2_fmmr = (4 * PI / Nc) ** 2
    check(
        "[4] FMMR (55) g_SG = 4 pi/N with kappa^2 = 4 pi^2/N^2 gives alpha_{SO(6)} = 1/4 for the N = 8 gauge fields too",
        sp.simplify(kappa2_N4 - 4 * PI**2 / Nc**2) == 0
        and sp.simplify(kappa2_N4 / g2_fmmr) == ALPHA_SUSY,
    )

    # ---------------------------------------------------------------------------
    # [5] the U(1)_r ratio test from the Lu-Pope-Tran Lagrangian
    # ---------------------------------------------------------------------------
    log("[5] U(1)_r / SU(2)_R ratio test from Lu-Pope-Tran eqs (7), (8), (11)")
    g1 = sp.Symbol("g1", positive=True)
    g2_LPT = sp.sqrt(2) * g1  # LPT: g_2 = sqrt2 g_1 at X = 1
    # equal kinetic terms -(1/2) X^{+-...} *F^F at X = 1, unit EH coefficient 2 kappa^2 = 1:
    # SU(2): unit structure constant after A -> A/g_2  =>  ghat_SU(2)^2 = g_2^2
    # U(1):  the complex two-form has D A_(2) = dA - i g_1 B_(1) A, so unit charge after B -> B/g_1
    #        =>  ghat_U(1)^2 = g_1^2 for the current J_B under which the two-form has charge 1
    ghat2_su2 = g2_LPT**2
    ghat2_u1 = g1**2
    CJ_su2 = C_J_holo.subs({d: 4, L: 1, g2: ghat2_su2})
    CJ_B = C_J_holo.subs({d: 4, L: 1, g2: ghat2_u1})
    r_two_form = 2  # |r| of the Delta = 3 antisymmetric tensor in the N = 2 stress-tensor multiplet
    C_r_holo = r_two_form**2 * CJ_B  # r = r_two_form * J_B
    check(
        "[5] bulk C_r/C_{I_3} = 4 * (g_2/g_1)^2 = 8 = field theory (12c)/(3c/2)",
        sp.simplify(C_r_holo / CJ_su2) == 8
        and sp.simplify(sol[Cr_sym] / sol[CI3_sym]) == 8,
        f"bulk ratio = {sp.simplify(C_r_holo / CJ_su2)}",
    )

    # ---------------------------------------------------------------------------
    # [6] curiosity
    # ---------------------------------------------------------------------------
    check(
        "[6] (not used) N = 1: C_R/C_T = 1/10 coincides numerically with (C_J/C_T)_* = 8/(d^2(d+1)) at d = 4",
        sp.simplify(CR_c / CT_c) == sp.Rational(8, 16 * 5),
    )

    log("")
    log("SUMMARY")
    log(
        "  free-field d = 4 (Wick, verified against Osborn-Petkou (5.5), (5.6), (5.16)):"
    )
    log(
        "    C_J: complex scalar 1/(4 pi^4), Weyl 1/(2 pi^4);  C_T: scalar 1/(3 pi^4), Weyl 1/pi^4, vector 4/pi^4 = 40c/pi^4"
    )
    log(
        f"  ROUTE 2 (free N = 4 SYM, SU(2)_R index 1):  C_J = 3/(8 pi^4), C_T = 10/pi^4  =>  C_J/C_T = {RATIO_ROUTE2}"
    )
    log(
        f"  ROUTE 1 (N = 2 Ward identities):  C_J(SU(2)_R) = 3c/(2 pi^4), C_r = 12c/pi^4, C_T = 40c/pi^4  =>  {RATIO_ROUTE1}"
    )
    log(
        f"  HOLOGRAPHY (C_J/C_T map x alpha_SUSY):  (3/20) alpha at alpha = 1/4  =>  {RATIO_HOLO}"
    )
    log(
        "  VERDICT: C_J(SU(2)_R)/C_T = 3/80 on all three sides; alpha_SUSY = 1/4 and the paper sec. 4.4 normalisations are anchored"
    )
    log(
        "  U(1)_r ratio test: C_r/C_{I_3} = 8 in the bulk (LPT) and in the field theory"
    )
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        raise SystemExit(1)
    log("ALL CHECKS PASSED")
    # The derivation's results, for callers that reuse them.
    return types.SimpleNamespace(**locals())


if __name__ == "__main__":
    main()
