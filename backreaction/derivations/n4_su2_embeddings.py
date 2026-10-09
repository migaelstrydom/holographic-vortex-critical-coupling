"""Where N = 4 SYM sits: alpha for every SU(2) inside SO(6)_R, and the
effective alpha seen by the most-charged W for a field along any Cartan
direction.

An extra: the paper quotes only the W bosons' alpha = 3/8 in the equal-charge
field (sec. 4.4, checked by anchors/dgp_charged_vector.py); the other
embeddings are not in the paper.  This module extends the map of sec. 4.4 and
app. A.2 to the SU(2) subgroups of SO(6)_R.

Question.  The paper maps alpha to C_J/C_T = (3/20) alpha in d = 4, with C_J
normalised by [J^a, J^b] = i eps^abc J^c.  For a global symmetry larger than
SU(2) this depends on which SU(2) one picks.  Large-N N = 4 SU(N) SYM has C_J
and C_T protected, so free-field counting is exact; per adjoint index the
fields are 6 real scalars in the 6 of SO(6) and 4 Weyl fermions in the 4 of
SU(4) (plus the R-neutral vector).

[1] FREE FIELDS, per SU(2) class.  The SU(2) subalgebras of su(4) = so(6) up to
    conjugacy are labelled by how the 4 splits: 2+1+1, 2+2, 3+1, 4.  Built
    explicitly as real antisymmetric 6x6 matrices, with the 4 obtained from
    the SAME matrices through the chiral Clifford algebra of SO(6) (not from
    the branching tables), and [t^a, t^b] = i eps^abc t^c checked in both
    representations:
      (a) SU(2)_L in SO(4) on scalars 1-4          (4 -> 2+1+1, 6 -> 2+2+1+1)
      (b) SO(3) rotating scalars 1-3 as a vector   (4 -> 2+2,   6 -> 3+1+1+1)
      (c) diagonal SO(3) in SO(3) x SO(3)          (4 -> 3+1,   6 -> 3+3)
      (d) principal SO(3) in SO(5), 5 = spin 2     (4 -> 4,     6 -> 5+1)
    C_J(T^3) = (1/2) tr_6(T3^2) C_J(complex scalar) + tr_4(T3^2) C_J(Weyl),
    C_T = 10/pi^4, with the Wick-contraction values derived in
    central_charge_anchor [1] (imported, not retyped).  Result:
        alpha = (20/3) C_J/C_T = j/4,  j = 1, 2, 4, 10 (embedding index),
    i.e. 1/4, 1/2, 1, 5/2.

[2] THE MOST-CHARGED W.  The bulk dual of the SO(6) currents is SO(6)
    Yang-Mills; truncated to metric + gauge fields (the scalars are discussed
    in [5]), a magnetic field along a Cartan generator Q is an abelian flux.
    Each charged vector W_lambda (a root of so(6)) satisfies, at linear order,
    exactly the SU(2) LLL equation with its own charge q = lambda(Q) -- the
    covariant derivative and the g = 2 moment both come from
    ad(Q) E_lambda = q E_lambda -- on a geometry that only knows the flux
    energy, proportional to C_J(Q).  Mapping onto unit-charge EYM:
        alpha_eff = alpha(Q) / q_max^2 = (20/3) C_J(Q) / (q_max^2 C_T),
    where q_max is the largest eigenvalue of ad(Q) on so(6), i.e. the largest
    Q-charge carried by a conserved current.  Checked on the AdS_3 x R^2
    throat: the trace-reversed Einstein equations with a weighted sum of
    commuting fluxes give L_3^2 = 1/3 and a W of charge q with
    -m^2 L_3^2 = q B L_3^2 = sqrt(2/(3 alpha_eff)), so BF violation
    <=> alpha_eff < 2/3.  alpha_eff is invariant under Q -> s Q (C_J ~ s^2,
    q ~ s): the criterion C_J/(q_max^2 C_T) < 1/10 needs no normalisation
    convention at all, and for an SU(2) with the paper's normalisation
    q_max = 1 (J^+- carry T^3 = +-1), so nothing changes there.
      (a) Q = (H1 + H2)/2, q_max = 1: alpha_eff = 1/4
      (b) Q = H1,          q_max = 1: alpha_eff = 1/2
      (c) Q = H1 + H2,     q_max = 2: alpha_eff = 1/4  (the root e1 + e2, which
          lies in SU(2)_L, not in the diagonal SO(3); its own W has 1)
      (d) Q = 2H1 + H2,    q_max = 3: alpha_eff = 5/18 (own W: 5/2, stable)
    with H_i the rotation generator of the scalar plane i (eigenvalues +-1).

[3] THREE-WAY NORMALISATION CHECK.  (i) free fields: alpha(Q) =
    (1/2)|b|^2 for Q = b.H; (ii) Romans/LPT (susy_coupling [2]): 1/4 for
    SU(2)_R, Q = (H1+H2)/2; (iii) the U(1)^3 truncation of DGP 1112.4195
    (2 kappa^2 = 1, -(1/4) sum F^i F^i, W in a_IJ of charge 1 under A^I and
    A^J, asserted in anchors/dgp_charged_vector [2]): alpha_eff =
    (1/2)|b|^2/(b_I + b_J)^2.  At b = (1,1,1) this is 3/8 -- the number that
    dgp_charged_vector [4] obtained from DGP's mass formula -L^2 m^2 = 4/3.
    All three agree on every direction.

[4] EVERY CARTAN DIRECTION.  q_max(b) = |b|_(1) + |b|_(2) (two largest),
    checked against ad(Q) eigenvalues on random directions, and
        1/4 <= alpha_eff(b) = |b|^2 / (2 q_max^2) <= 1/2,
    min at b ~ (1,1,0), max at b ~ (1,0,0) (proof in two lines, checked by
    sampling).  So in N = 4 SYM, in the metric + gauge-field truncation, every
    R-symmetry magnetic field is in the condensing window alpha_eff < 2/3; the
    directions near (1,1,0) are below alpha_L = 0.37218, b = (1,1,1) is at
    3/8 just above it, (1,0,0) at 1/2.

[5] SCALARS.  In the U(1)^3 truncation (the diagonal 20' scalars X_i, the
    only ones a Cartan flux can source at O(F^2), since (F F)_IJ in the 6 is
    diagonal) the X_i equation at X_i = 1 has source proportional to
    b_i^2 - b_3^2: the truncation to metric + gauge fields is consistent at
    O(F^2) iff |b1| = |b2| = |b3|.  So only the minimal-gauged-supergravity
    direction (1,1,1) -- alpha_eff = 3/8 -- is a top-down background without
    scalars; everywhere else (including Romans' SU(2), [4] of susy_coupling)
    the scalars are sourced at the order of the backreaction and the alpha
    regimes are those of a model, not of N = 4 SYM.  Beyond linear order the
    regimes of sec. 5 (alpha_L) are computed for a single SU(2): with several
    degenerate most-charged roots ((1,1,1) has three, (1,0,0) four) the
    condensate problem is different, so only the linear statement
    (alpha_eff < 2/3, instability) transfers.

Run:  uv run python -m backreaction.derivations.n4_su2_embeddings   (~5 s)
"""

from __future__ import annotations

import contextlib
import io
import itertools
import random
import time
import types

import sympy as sp

T0 = time.time()
CHECKS: dict[str, bool] = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def L(a, b, n=6):
    """Real antisymmetric generator of rotations in the (a, b) plane (1-based)."""
    M = sp.zeros(n, n)
    M[a - 1, b - 1], M[b - 1, a - 1] = 1, -1
    return M


def comm(A, B):
    return A * B - B * A


def spinor_rep():
    """omega (real antisymmetric 6x6) -> its action on the chiral 4 of Spin(6)."""
    s1 = sp.Matrix([[0, 1], [1, 0]])
    s2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    s3 = sp.Matrix([[1, 0], [0, -1]])
    one = sp.eye(2)

    def kron(*ms):
        out = ms[0]
        for m in ms[1:]:
            out = sp.kronecker_product(out, m)
        return out

    gam = [
        kron(s1, one, one),
        kron(s2, one, one),
        kron(s3, s1, one),
        kron(s3, s2, one),
        kron(s3, s3, s1),
        kron(s3, s3, s2),
    ]
    for i, j in itertools.product(range(6), repeat=2):
        assert gam[i] * gam[j] + gam[j] * gam[i] == (2 if i == j else 0) * sp.eye(8)
    g7 = -sp.I * gam[0] * gam[1] * gam[2] * gam[3] * gam[4] * gam[5]
    assert g7 * g7 == sp.eye(8)
    P = sp.Matrix.hstack(*[v for v in (g7 - sp.eye(8)).nullspace()])  # +1 eigenspace
    P = sp.Matrix.hstack(
        *[c / c.norm() for c in sp.GramSchmidt([P[:, k] for k in range(P.shape[1])])]
    )

    def rho(omega):
        S = sp.zeros(8, 8)
        for a, b in itertools.product(range(6), repeat=2):
            S += omega[a, b] * gam[a] * gam[b]
        S = S / 4
        return sp.simplify(P.H * S * P)

    return rho


def main():
    rho4 = spinor_rep()
    # the spin map is a Lie-algebra homomorphism (sign fixed by this check)
    random.seed(5)
    ok = True
    for (a, b), (c, d) in [((1, 2), (2, 3)), ((1, 4), (4, 6)), ((2, 5), (5, 3))]:
        w1, w2 = L(a, b), L(c, d)
        ok &= sp.simplify(comm(rho4(w1), rho4(w2)) - rho4(comm(w1, w2))) == sp.zeros(4)
    check(
        "[1] Clifford map so(6) -> 4 x 4 is a homomorphism (chiral 4 of Spin(6) = 4 of SU(4))",
        ok,
    )

    # free-field coefficients, derived by Wick contraction in central_charge_anchor [1]
    from backreaction.derivations import central_charge_anchor as cca_mod

    with contextlib.redirect_stdout(io.StringIO()):
        cca = cca_mod.main()
    CJ_CS, CJ_W, CT = cca.CJ_CSCALAR, cca.CJ_WEYL, cca.CT_N4
    pi4 = sp.pi**4
    check(
        "[1] imported free values: C_J(cplx scalar) = 1/(4pi^4), C_J(Weyl) = 1/(2pi^4), C_T(N=4, per adjoint) = 10/pi^4",
        sp.simplify(CJ_CS * pi4 - sp.Rational(1, 4)) == 0
        and sp.simplify(CJ_W * pi4 - sp.Rational(1, 2)) == 0
        and sp.simplify(CT * pi4 - 10) == 0,
    )

    # ---- the four SU(2) classes, as real antisymmetric omega^a; t^a = i omega^a ----
    h = sp.Rational(1, 2)
    eps = lambda i, j, k: sp.LeviCivita(i, j, k)  # noqa: E731

    def so3_vector(idx):
        """omega^a = -(1/2) eps_abc L_bc on the three scalars idx."""
        return [
            -h
            * sum(
                (eps(a, b, c) * L(idx[b], idx[c]) for b in range(3) for c in range(3)),
                sp.zeros(6),
            )
            for a in range(3)
        ]

    sl = [
        -h
        * (
            h
            * sum(
                (eps(a, b, c) * L(b + 1, c + 1) for b in range(3) for c in range(3)),
                sp.zeros(6),
            )
            + L(a + 1, 4)
        )
        for a in range(3)
    ]  # self-dual SU(2) in SO(4) on 1..4
    vb = so3_vector([1, 2, 3])
    vc = [
        x + y for x, y in zip(so3_vector([1, 3, 5]), so3_vector([2, 4, 6]), strict=True)
    ]

    # principal SO(3): spin 2 on symmetric traceless 3x3 matrices, embedded in scalars 1..5
    E = lambda i, j: sp.Matrix(3, 3, lambda r, s: 1 if (r, s) == (i, j) else 0)  # noqa: E731
    basis5 = [
        (E(0, 1) + E(1, 0)) / sp.sqrt(2),
        (E(0, 2) + E(2, 0)) / sp.sqrt(2),
        (E(1, 2) + E(2, 1)) / sp.sqrt(2),
        (E(0, 0) - E(1, 1)) / sp.sqrt(2),
        (E(0, 0) + E(1, 1) - 2 * E(2, 2)) / sp.sqrt(6),
    ]
    w3 = [
        -h
        * sum(
            (eps(a, b, c) * (E(b, c) - E(c, b)) for b in range(3) for c in range(3)),
            sp.zeros(3),
        )
        for a in range(3)
    ]
    vd = []
    for a in range(3):
        M = sp.zeros(6)
        for i, Bi in enumerate(basis5):
            img = comm(w3[a], Bi)
            for j, Bj in enumerate(basis5):
                M[j, i] = sp.simplify((Bj.T * img).trace())
        vd.append(M)

    classes = {
        "(a) SU(2)_L in SO(4)": (sl, "2+1+1"),
        "(b) SO(3) on 3 scalars": (vb, "2+2"),
        "(c) diagonal SO(3)": (vc, "3+1"),
        "(d) principal SO(3)": (vd, "4"),
    }

    def as_t(omegas, sign):
        return [sign * sp.I * w for w in omegas]

    # adjoint of so(6) on the 15 L_ab, for current charges
    basis15 = [L(a, b) for a, b in itertools.combinations(range(1, 7), 2)]

    def ad_eigs(t3):
        """Eigenvalues of ad(t3) on so(6): the t3-charges of the 15 currents."""
        omega = -sp.I * t3
        Mad = sp.zeros(15)
        for i, Bi in enumerate(basis15):
            img = comm(omega, Bi)
            for j, Bj in enumerate(basis15):
                Mad[j, i] = -(Bj * img).trace() / 2  # <L_ab, L_cd> = -tr/2 orthonormal
        ev = (sp.I * Mad).eigenvals()  # hermitian ad(t3) = i ad(omega)
        return sorted(
            [sp.nsimplify(sp.simplify(e)) for e, m in ev.items() for _ in range(m)]
        )

    results = {}
    for name, (omegas, split4) in classes.items():
        # pick the overall sign so that [t^a, t^b] = i eps t^c
        for sign in (1, -1):
            t6 = as_t(omegas, sign)
            if sp.simplify(comm(t6[0], t6[1]) - sp.I * t6[2]) == sp.zeros(6):
                break
        t4 = [sp.I * sign * rho4(w) for w in omegas]
        ok6 = all(
            sp.simplify(
                comm(t6[a], t6[b])
                - sp.I * sum((eps(a, b, c) * t6[c] for c in range(3)), sp.zeros(6))
            )
            == sp.zeros(6)
            for a in range(3)
            for b in range(3)
        )
        ok4 = all(
            sp.simplify(
                comm(t4[a], t4[b])
                - sp.I * sum((eps(a, b, c) * t4[c] for c in range(3)), sp.zeros(4))
            )
            == sp.zeros(4)
            for a in range(3)
            for b in range(3)
        )
        t3_6, t3_4 = t6[2], t4[2]
        ev4 = sorted(
            sp.nsimplify(e) for e, m in t3_4.eigenvals().items() for _ in range(m)
        )
        tr6 = sp.simplify((t3_6 * t3_6).trace())
        tr4 = sp.simplify((t3_4 * t3_4).trace())
        CJ = sp.simplify(h * tr6 * CJ_CS + tr4 * CJ_W)
        alpha_own = sp.simplify(sp.Rational(20, 3) * CJ / CT)
        j = sp.simplify(tr4 / h)
        q = ad_eigs(t3_6)
        qmax = max(q)
        alpha_eff = sp.simplify(alpha_own / qmax**2)
        check(
            f"[1] {name}: [t,t] = i eps t on the 6 and on the 4; T3 on the 4 = {ev4} ({split4})",
            ok6 and ok4,
        )
        check(
            f"[1] {name}: tr_6 T3^2 = 2 tr_4 T3^2 = {tr6}, C_J = {sp.nsimplify(CJ * pi4)}/pi^4, alpha = {alpha_own} = j/4 (j = {j})",
            sp.simplify(tr6 - 2 * tr4) == 0 and sp.simplify(alpha_own - j / 4) == 0,
        )
        check(
            f"[2] {name}: current charges max|ad T3| = {qmax} (own W: 1)  =>  alpha_eff = {alpha_eff}",
            1 in q and -qmax in q,
        )
        results[name] = types.SimpleNamespace(
            j=j, alpha=alpha_own, qmax=qmax, alpha_eff=alpha_eff, t3=t3_6, CJ=CJ
        )

    expect = {
        "(a) SU(2)_L in SO(4)": (1, sp.Rational(1, 4), 1, sp.Rational(1, 4)),
        "(b) SO(3) on 3 scalars": (2, sp.Rational(1, 2), 1, sp.Rational(1, 2)),
        "(c) diagonal SO(3)": (4, 1, 2, sp.Rational(1, 4)),
        "(d) principal SO(3)": (10, sp.Rational(5, 2), 3, sp.Rational(5, 18)),
    }
    check(
        "[1,2] table: (j, alpha, q_max, alpha_eff) = (1,1/4,1,1/4) (2,1/2,1,1/2) (4,1,2,1/4) (10,5/2,3,5/18)",
        all(
            (r.j, r.alpha, r.qmax, r.alpha_eff) == expect[k] for k, r in results.items()
        ),
    )

    # Cartan direction of each class in the basis H_i = rotation of plane (2i-1, 2i)
    H = [sp.I * L(1, 2), sp.I * L(3, 4), sp.I * L(5, 6)]

    def b_of(t3):
        """b with t3 conjugate to b.H: read from the spectrum of t3 on the 6."""
        ev = sorted(
            (sp.nsimplify(e) for e, m in t3.eigenvals().items() for _ in range(m)),
            reverse=True,
        )
        pos = [e for e in ev if e > 0]
        return sorted(pos + [0] * (3 - len(pos)), reverse=True)

    bdir = {k: b_of(r.t3) for k, r in results.items()}
    check(
        "[2] Cartan directions: (a) (1/2,1/2,0), (b) (1,0,0), (c) (1,1,0) = 2x(a), (d) (2,1,0)",
        bdir["(a) SU(2)_L in SO(4)"] == [h, h, 0]
        and bdir["(b) SO(3) on 3 scalars"] == [1, 0, 0]
        and bdir["(c) diagonal SO(3)"] == [1, 1, 0]
        and bdir["(d) principal SO(3)"] == [2, 1, 0],
        str(bdir),
    )

    # ---- [2] the throat: weighted commuting fluxes on AdS_3 x R^2 ----
    l2, B, w, alph, q = sp.symbols("ell2 B w alpha q", positive=True)
    t, x, y, z, rr = sp.symbols("t x y z r", real=True)
    coords = [t, z, rr, x, y]  # AdS_3 (t, z, r) Poincare, radius^2 = l2; flat (x, y)
    g = sp.diag(-l2 / rr**2, l2 / rr**2, l2 / rr**2, 1, 1)
    Ric = ricci(g, coords)
    gi = g.inv()
    # flux F_xy = B with energy weight w (w = |b|^2 in units where alpha(Q) = alpha w)
    Fm = sp.zeros(5)
    Fm[3, 4], Fm[4, 3] = B, -B
    Q2 = Fm * gi * Fm.T
    F2 = sum(Q2[i, j] * gi[i, j] for i in range(5) for j in range(5))
    E_ = Ric + 4 * g - alph * w * (Q2 - F2 * g / 6)
    sol = sp.solve([E_[0, 0], E_[3, 3]], [l2, B], dict=True)
    sol = [s for s in sol if s[l2].is_positive and s[B].is_positive][0]
    bW = sp.simplify(q * sol[B] * sol[l2])  # -m^2 L3^2 of the LLL W of charge q
    a_eff = sp.Symbol("alpha_eff", positive=True)
    check(
        "[2] throat: L3^2 = 1/3, all Einstein components solved, -m^2 L3^2 = q B L3^2 = sqrt(2/(3 alpha_eff)) with alpha_eff = alpha w/q^2",
        sol[l2] == sp.Rational(1, 3)
        and E_.subs(sol).applyfunc(sp.simplify) == sp.zeros(5)
        and sp.simplify(
            bW - sp.sqrt(sp.Rational(2, 3) / a_eff).subs(a_eff, alph * w / q**2)
        )
        == 0
        and sp.solve(sp.Eq(sp.sqrt(sp.Rational(2, 3) / a_eff), 1), a_eff)
        == [sp.Rational(2, 3)],
        f"B^2 = {sp.simplify(sol[B] ** 2)}",
    )

    # ---- [3] three-way normalisation ----
    b1, b2, b3 = sp.symbols("b1 b2 b3", real=True)
    Qgen = b1 * H[0] + b2 * H[1] + b3 * H[2]
    tr6 = sp.expand((Qgen * Qgen).trace())
    tr4 = sp.expand(
        (
            sum(
                (bb * rho4(-sp.I * Hk) for bb, Hk in zip((b1, b2, b3), H, strict=True)),
                sp.zeros(4),
            )
            * sp.I
        )
        ** 2
    ).trace()
    alpha_free = sp.expand(
        sp.Rational(20, 3) * (h * tr6 * CJ_CS + sp.expand(tr4) * CJ_W) / CT
    )
    bsq = b1**2 + b2**2 + b3**2
    alpha_dgp = bsq / 2  # 2 kappa^2 = 1, -(1/4) sum F^i F^i, unit charge per A^i
    check(
        "[3] free fields: alpha(b.H) = |b|^2/2 = DGP U(1)^3 normalisation, identically in b",
        sp.simplify(alpha_free - alpha_dgp) == 0,
        f"alpha_free = {sp.factor(alpha_free)}",
    )
    from backreaction.derivations import susy_coupling as sc_mod

    with contextlib.redirect_stdout(io.StringIO()):
        sc = sc_mod.main()
    a_R = sp.simplify(sc.alpha_gen)  # alpha/L^2 from LPT, any g1/g2 split
    check(
        "[3] Romans/LPT (susy_coupling [2c]) alpha = 1/4 = free alpha at b = (1/2,1/2,0)",
        a_R == sp.Rational(1, 4) and alpha_free.subs({b1: h, b2: h, b3: 0}) == a_R,
    )
    qmax_111 = 2
    a111 = alpha_dgp.subs({b1: 1, b2: 1, b3: 1}) / qmax_111**2
    bW111 = sp.sqrt(sp.Rational(2, 3) / a111)
    check(
        "[3] b = (1,1,1): alpha_eff = 3/8, -m^2 L3^2 = 4/3 = DGP eq (3.21) at the DK origin (dgp_charged_vector [4])",
        a111 == sp.Rational(3, 8) and bW111 == sp.Rational(4, 3),
    )

    # ---- [4] every Cartan direction ----
    rng = random.Random(54)
    ok_q, ok_rng = True, True
    samples = []
    for _ in range(12):
        bv = [sp.Rational(rng.randint(-9, 9), rng.randint(1, 4)) for _ in range(3)]
        if all(v == 0 for v in bv):
            continue
        Qn = sum((v * Hk for v, Hk in zip(bv, H, strict=True)), sp.zeros(6))
        qm = max(ad_eigs(Qn))
        s = sorted((abs(v) for v in bv), reverse=True)
        ok_q &= qm == s[0] + s[1]
        ae = sum(v**2 for v in bv) / (2 * qm**2)
        ok_rng &= sp.Rational(1, 4) <= ae <= sp.Rational(1, 2)
        samples.append(ae)
    check(
        "[4] q_max(b) = |b|_(1) + |b|_(2) on 12 random directions (ad eigenvalues)",
        ok_q,
    )
    # proof: s1 >= s2 >= s3 >= 0; |b|^2 >= s1^2 + s2^2 >= (s1+s2)^2/2  and
    # |b|^2 <= s1^2 + 2 s2^2 <= (s1 + s2)^2  (as s1 >= s2)
    s1, s2, s3 = sp.symbols("s1 s2 s3", nonnegative=True)
    lower = sp.expand(2 * (s1**2 + s2**2 + s3**2) - (s1 + s2) ** 2 / 2 * 2)
    upper = sp.expand((s1 + s2) ** 2 - (s1**2 + s2**2 + s3**2))
    check(
        "[4] 1/4 <= alpha_eff <= 1/2: 2|b|^2 - (s1+s2)^2 = (s1-s2)^2 + 2 s3^2 >= 0;"
        " (s1+s2)^2 - |b|^2 = 2 s1 s2 - s3^2 >= s2^2 + (s2^2 - s3^2) >= 0",
        sp.expand(lower - ((s1 - s2) ** 2 + 2 * s3**2)) == 0
        and sp.expand(upper - (2 * s1 * s2 - s3**2)) == 0
        and ok_rng,
        f"sampled range [{float(min(samples)):.4f}, {float(max(samples)):.4f}]",
    )
    ALPHA_L = 0.37217997
    regimes = {
        "(1,1,0) [SU(2)_L, diag SO(3)]": sp.Rational(1, 4),
        "(2,1,0) [principal]": sp.Rational(5, 18),
        "(1,1,1) [minimal sugra U(1)_R]": sp.Rational(3, 8),
        "(1,0,0) [SO(3) on 3 scalars]": sp.Rational(1, 2),
    }
    for k, v in regimes.items():
        reg = (
            "crystal (< alpha_L)"
            if v < ALPHA_L
            else ("alpha_L..alpha_*" if v < sp.Rational(2, 3) else "> alpha_*")
        )
        log(f"      b ~ {k}: alpha_eff = {v} = {float(v):.4f}  -> {reg}")

    # ---- [5] scalars in the U(1)^3 truncation ----
    X1, X2, Fs1, Fs2, Fs3 = sp.symbols("X1 X2 F1sq F2sq F3sq", positive=True)
    X3 = 1 / (X1 * X2)
    Lkin = -sp.Rational(1, 4) * (Fs1 / X1**2 + Fs2 / X2**2 + Fs3 / X3**2)
    Vpot = -4 * (1 / X1 + 1 / X2 + 1 / X3)
    src = [
        sp.simplify(sp.diff(Lkin - Vpot, Xi).subs({X1: 1, X2: 1})) for Xi in (X1, X2)
    ]
    check(
        "[5] U(1)^3 truncation: dL/dX_i at X = 1 = (F_i^2 - F_3^2)/2 (potential extremal) -> no scalar source iff |b1| = |b2| = |b3|",
        sp.simplify(src[0] - (Fs1 - Fs3) / 2) == 0
        and sp.simplify(src[1] - (Fs2 - Fs3) / 2) == 0,
        f"sources {src}",
    )

    log("")
    log("SUMMARY (N = 4 SU(N) SYM, large N; metric + SO(6) gauge-field truncation)")
    for k, r in results.items():
        log(
            f"  {k:26s} j = {str(r.j):>2}  alpha = {str(r.alpha):4s}  q_max = {r.qmax}  alpha_eff = {r.alpha_eff}"
        )
    log(
        "  alpha_eff = (20/3) C_J(Q)/(q_max^2 C_T) in [1/4, 1/2] for every Cartan direction; 3/8 at (1,1,1)"
    )
    log("  criterion, normalisation-free: C_J(Q)/(q_max^2 C_T) < 8/(d^2(d+1))")
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        raise SystemExit(1)
    log("ALL CHECKS PASSED")
    return types.SimpleNamespace(**locals())


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
    for m, k in itertools.product(range(n), repeat=2):
        R[m, k] = sp.simplify(
            sum(sp.diff(Gam[r][m][k], X[r]) for r in range(n))
            - sum(sp.diff(Gam[r][m][r], X[k]) for r in range(n))
            + sum(Gam[r][r][s] * Gam[s][m][k] for r in range(n) for s in range(n))
            - sum(Gam[r][k][s] * Gam[s][m][r] for r in range(n) for s in range(n))
        )
    return R


if __name__ == "__main__":
    main()
