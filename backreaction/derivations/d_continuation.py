"""The zero-temperature magnetic brane and its threshold mode with d a real
parameter: the reduced equations in two charts, the fixed point, its growing
deformation, and the Liouville normal form of the threshold mode.

Used by `backreaction/numerics/d3_crossover_why.py`.  An extra, not in the
paper: the paper treats only the integer dimensions d = 3, 4, 5, 6 (sec. 4.3,
app. D.2); the continuation in d explains why d = 3 differs.

Warped-product reduction.  For ds^2 = d rho^2 + sum_i m_i e^{2 a_i(rho)} dx_i^2
(m_i copies of each direction) the mixed Ricci components are

    R^i_i     = -(a_i'' + a_i' S),          S = sum_j m_j a_j',
    R^rho_rho = -sum_j m_j (a_j'' + a_j'^2),

affine in the multiplicities.  The brane at T = 0 has (d - 2) boost-invariant
directions (t, z_1..z_{d-3}) with warp A and the flux plane with warp C, so
d enters only as the multiplicity d - 2 and through the field equation
R_MN + d g_MN = alpha (F_MP F_N^P - F^2 g_MN / (2(d-1))).  The continuation
in d is therefore the one in which every coefficient is a polynomial in d
over (d - 1).

Checks
------
[1] Domain-wall gauge: the general-d equations (A'', C'', constraint) equal
    those derived from the Christoffel symbols by `brane_flows.domain_wall(d)`
    at d = 3, 4, 5, 6, 7, 8.  At fixed fields both sides are N(d)/(d - 1) with
    N of degree <= 3 (asserted for this side), so six integer points fix the
    continuation uniquely.
[2] r chart (g_tt g_rr = -1) at T = 0, e^{2Z} = U: the general-d (U'', V'')
    equal `brane_flows.r_chart(d)` at d = 4..8, and its Z'' equation is
    consistent with Z = (1/2) ln U (boost invariance).
[3] Fixed point and BF bound for real d: l^2 = (d-2)^2/(d(d-1)),
    v = B sqrt(alpha (d-2)/(d(d-1))), alpha_*(d) = 16/(d(d-1)(d-2)).
[4] The growing deformation for real d has the closed-form exponent
        p(d) = k l = (d-2)(sqrt((d-1)(d+15)) - (d-1)) / (2(d-1)),
    p(3) = 1, p(4) = sqrt(19/3) - 1 (brane_flows [3]); p(5), p(6) numerically
    equal to brane_flows.  Its A-amplitude a_k(d) is regular at d = 3.
    The same deformation in the r chart, U = (x/l)^2 (1 + u_p eps x^p),
    e^{2V} = v (1 + 2 eps x^p) (x = r - r0), with u_p(d) solved independently
    in that chart; u_p is consistent with a_k (change of chart).
[5] Liouville normal form of -(e^{nA} w')' = lam e^{nA-2C} w (n = d - 2 for the
    physical operator, n free for the test that replaces e^{(d-3)Z} by e^{eps Z}): with dy = e^{-C} d rho,
    psi = e^{(nA - C)/2} w,
        -psi_yy + V_L psi = lam psi,
        V_L = e^{2C} (s'' + s'^2 + C' s'),   s = (nA - C)/2.
    Throat (A = rho/l, C = C0): V_L = n^2 v/(4 l^2), = 1 at alpha_*(d) when
    n = d - 2 (the continuum edge).  AdS_{d+1} boundary (A = C = rho + c):
    V_L = (n^2 - 1)/(4 (y_b - y)^2), a centrifugal barrier with
    l_eff(l_eff + 1) = (n^2 - 1)/4, i.e. (d-1)(d-3)/4 for the physical
    operator, which vanishes exactly at d = 3.  Source-free <=> psi ~ (y_b - y)^{(n+1)/2}.

Run:  uv run python -m backreaction.derivations.d_continuation   (~1 min)
"""

import sys
import time

import sympy as sp

from backreaction.derivations import brane_flows as bf

T0 = time.time()
CHECKS = {}

d = sp.Symbol("d", positive=True)
B, alpha = bf.B, bf.alpha
a1, c1, cc = sp.symbols("a1 c1 cc")


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def is_zero(e):
    e = sp.expand(e)
    return e == 0 or sp.simplify(e) == 0


def warped_ricci(warps, mults, D):
    """Mixed Ricci components of d rho^2 + sum m_i e^{2 a_i} dx_i^2.
    `warps` are expressions, D is the proper-length derivative operator."""
    S = sum(m * D(a) for a, m in zip(warps, mults, strict=True))
    Rii = [-(D(D(a)) + D(a) * S) for a in warps]
    Rrr = -sum(m * (D(D(a)) + D(a) ** 2) for a, m in zip(warps, mults, strict=True))
    return Rii, Rrr


def source_terms(dd, C):
    """alpha(F F - F^2 g/(2(d-1))) mixed: (boost dirs, flux dirs, rho)."""
    Phi = B**2 * sp.exp(-4 * C)
    return (
        -alpha * Phi / (dd - 1),
        alpha * Phi * (dd - 2) / (dd - 1),
        -alpha * Phi / (dd - 1),
    )


# ---------------------------------------------------------------------------
# domain-wall gauge, real d
# ---------------------------------------------------------------------------
def domain_wall(dd=d):
    """(A'', C'', constraint) in terms of (a1, c1, cc) = (A', C', C)."""
    rho = sp.Symbol("rho")
    A, C = sp.Function("A")(rho), sp.Function("C")(rho)
    (RA, RC), Rrr = warped_ricci([A, C], [dd - 2, 2], lambda e: sp.diff(e, rho))
    sA, sC, sR = source_terms(dd, C)
    a2, c2 = sp.symbols("a2 c2")
    rep = {A.diff(rho, 2): a2, C.diff(rho, 2): c2, A.diff(rho): a1, C.diff(rho): c1}
    eA = (RA + dd - sA).subs(rep).subs(C, cc)
    eC = (RC + dd - sC).subs(rep).subs(C, cc)
    eR = (Rrr + dd - sR).subs(rep).subs(C, cc)
    sol = sp.solve([eA, eC], [a2, c2], dict=True)[0]
    return dict(
        App=sp.simplify(sol[a2]),
        Cpp=sp.simplify(sol[c2]),
        con=sp.simplify(eR.subs(sol)),
    )


# ---------------------------------------------------------------------------
# r chart, T = 0 (e^{2Z} = U), real d
# ---------------------------------------------------------------------------
def r_chart(dd=d):
    """(U'', V'', constraint) in terms of (U, V, U', V')."""
    r = sp.Symbol("r", positive=True)
    U, V = sp.Function("U")(r), sp.Function("V")(r)
    D = lambda e: sp.sqrt(U) * sp.diff(e, r)
    (RA, RC), Rrr = warped_ricci([sp.log(U) / 2, V], [dd - 2, 2], D)
    sA, sC, sR = source_terms(dd, V)
    Uv, Vv, Up, Vp, Upp, Vpp = sp.symbols("Uv Vv Up Vp Upp Vpp")
    rep = {U.diff(r, 2): Upp, V.diff(r, 2): Vpp, U.diff(r): Up, V.diff(r): Vp}
    fin = lambda e: sp.simplify(e.subs(rep).subs({U: Uv, V: Vv}))
    eA, eC, eR = fin(RA + dd - sA), fin(RC + dd - sC), fin(Rrr + dd - sR)
    sol = sp.solve([eA, eC], [Upp, Vpp], dict=True)[0]
    return dict(
        Upp=sp.simplify(sol[Upp]),
        Vpp=sp.simplify(sol[Vpp]),
        con=sp.simplify(eR.subs(sol)),
        syms=(Uv, Vv, Up, Vp),
    )


# ---------------------------------------------------------------------------
# fixed point, deformation
# ---------------------------------------------------------------------------
ELL = (d - 2) / sp.sqrt(d * (d - 1))
VFIX = B * sp.sqrt(alpha * (d - 2) / (d * (d - 1)))
ALPHA_STAR = 16 / (d * (d - 1) * (d - 2))
P_EXP = (d - 2) * (sp.sqrt((d - 1) * (d + 15)) - (d - 1)) / (2 * (d - 1))


def deformation_dw(dw):
    """Linearise the domain-wall system: C = C0 + e e^{k rho}, A = rho/l + a e e^{k rho}."""
    rho, e, k, a = sp.symbols("rho e k a")
    Cl = sp.log(VFIX) / 2 + e * sp.exp(k * rho)
    Al = rho / ELL + e * a * sp.exp(k * rho)
    sub = {a1: Al.diff(rho), c1: Cl.diff(rho), cc: Cl}

    def lin(expr, lhs):
        x = sp.diff(lhs - expr.subs(sub), e).subs(e, 0)
        return sp.simplify(x * sp.exp(-k * rho))

    return [
        lin(dw["App"], Al.diff(rho, 2)),
        lin(dw["Cpp"], Cl.diff(rho, 2)),
        lin(dw["con"], 0),
    ], (k, a)


def a_k_closed():
    """A-amplitude of the growing deformation (regular form, from the A'' row)."""
    S = sp.sqrt(d * (d - 1))
    k = P_EXP / ELL
    return sp.simplify(-(4 * d + 2 * k * S) / ((d - 2) * k * (k + 2 * S))), k


def deformation_r(rc):
    """U = (x/l)^2 (1 + u eps x^p), e^{2V} = v (1 + 2 eps x^p), x = r - r0 (r0 = 0)."""
    x, e, u = sp.symbols("x e u", positive=True)
    p = sp.Symbol("p", positive=True)
    Uv, Vv, Up, Vp = rc["syms"]
    Ue = (x / ELL) ** 2 * (1 + u * e * x**p)
    Ve = sp.log(VFIX) / 2 + e * x**p
    sub = {Uv: Ue, Vv: Ve, Up: Ue.diff(x), Vp: Ve.diff(x)}
    rows = []
    for lhs, key in ((Ue.diff(x, 2), "Upp"), (Ve.diff(x, 2), "Vpp")):
        r_ = sp.diff(lhs - rc[key].subs(sub), e).subs(e, 0)
        rows.append(sp.simplify(r_ * x ** (2 - p) if key == "Vpp" else r_ * x ** (-p)))
    return rows, (u, p)


# ---------------------------------------------------------------------------
# Liouville normal form
# ---------------------------------------------------------------------------
def liouville():
    rho, n, lam = sp.symbols("rho n lam")
    A, C, w = (sp.Function(s)(rho) for s in ("A", "C", "w"))
    s = (n * A - C) / 2
    VL = sp.exp(2 * C) * (s.diff(rho, 2) + s.diff(rho) ** 2 + C.diff(rho) * s.diff(rho))
    psi = sp.exp(s) * w
    Dy = lambda f: sp.exp(C) * f.diff(rho)
    schr = -Dy(Dy(psi)) + VL * psi - lam * psi
    mode = -(sp.exp(n * A) * w.diff(rho)).diff(rho) - lam * sp.exp(n * A - 2 * C) * w
    # schr = e^{s + 2C - nA} * mode
    ok = is_zero(sp.expand(schr - sp.exp(s + 2 * C - n * A) * mode))
    return VL, ok, (rho, n, A, C)


def main():
    log("[1] domain-wall gauge, general d, against brane_flows.domain_wall")
    dw = domain_wall()
    for key in ("App", "Cpp", "con"):
        num = sp.numer(sp.together(dw[key] * (d - 1)))
        deg = sp.degree(sp.expand(num), d)
        check(
            f"[1] {key}: (d-1) x expression is polynomial in d of degree <= 3",
            sp.denom(sp.together(dw[key] * (d - 1))).free_symbols.isdisjoint({d})
            and deg <= 3,
            f"degree {deg}",
        )
    for dd in (3, 4, 5, 6, 7, 8):
        ref = bf.domain_wall(dd)
        a1r, c1r, ccr = ref["syms"]
        m = {a1r: a1, c1r: c1, ccr: cc}
        okA = is_zero(dw["App"].subs(d, dd) - ref["App"].xreplace(m))
        okC = is_zero(dw["Cpp"].subs(d, dd) - ref["Cpp"].xreplace(m))
        ratio = sp.simplify(dw["con"].subs(d, dd) / ref["con"].xreplace(m))
        check(
            f"[1] d={dd}: A'', C'' and constraint equal brane_flows",
            okA and okC and ratio.is_number,
            f"constraint ratio {ratio}",
        )

    log("[2] r chart (T = 0, e^{2Z} = U), general d, against brane_flows.r_chart")
    rc = r_chart()
    Uv, Vv, Up, Vp = rc["syms"]
    for dd in (4, 5, 6, 7, 8):
        ref = bf.r_chart(dd)
        U2, V2, Z2, Up2, Vp2, Zp2 = ref["syms"]
        m = {U2: Uv, V2: Vv, Z2: sp.log(Uv) / 2, Up2: Up, Vp2: Vp, Zp2: Up / (2 * Uv)}
        okU = is_zero(rc["Upp"].subs(d, dd) - ref["Upp"].xreplace(m))
        okV = is_zero(rc["Vpp"].subs(d, dd) - ref["Vpp"].xreplace(m))
        Zpp_ans = rc["Upp"] / (2 * Uv) - Up**2 / (2 * Uv**2)
        okZ = is_zero(Zpp_ans.subs(d, dd) - ref["Zpp"].xreplace(m))
        check(
            f"[2] d={dd}: U'', V'' equal brane_flows; Z = ln(U)/2 consistent",
            okU and okV and okZ,
        )

    log("[3] fixed point and BF bound, real d")
    sub_fp = {a1: 1 / ELL, c1: 0, cc: sp.log(VFIX) / 2}
    check(
        "[3] AdS_{d-1} x R^2 solves all three equations for real d",
        all(is_zero(dw[k_].subs(sub_fp)) for k_ in ("App", "Cpp", "con")),
    )
    disc = sp.simplify(((d - 2) / ELL) ** 2 - 4 * B / VFIX)
    ast = sp.solve(sp.Eq(disc, 0), alpha)
    check(
        "[3] discriminant vanishes at alpha_*(d) = 16/(d(d-1)(d-2))",
        len(ast) == 1 and is_zero(ast[0] - ALPHA_STAR),
        f"{ast}",
    )

    log("[4] the growing deformation, real d")
    rows, (k, a) = deformation_dw(dw)
    kk = P_EXP / ELL
    ak, _ = a_k_closed()
    check("[4] k = p(d)/l solves the C row (closed form)", is_zero(rows[1].subs(k, kk)))
    res = [
        abs(complex(sp.N(r_.subs({k: kk, a: ak}).subs({d: dd, alpha: 1, B: 1}), 30)))
        for r_ in rows
        for dd in (3, sp.Rational(13, 4), sp.Rational(7, 2), 4, 5, 6)
    ]
    check(
        "[4] (k, a_k(d)) solves all three linearised rows (d = 3 ... 6, 30 digits)",
        max(res) < 1e-25,
        f"max residual {max(res):.1e}",
    )
    other = sp.solve(rows[1], k)
    check(
        "[4] the C row has exactly one positive root",
        sum(1 for o in other if sp.N(o.subs(d, sp.Rational(7, 2))) > 0) == 1,
        f"roots {other}",
    )
    pv = {dd: float(P_EXP.subs(d, dd)) for dd in (3, 4, 5, 6)}
    check(
        "[4] p(3) = 1, p(4) = sqrt(19/3) - 1",
        is_zero(P_EXP.subs(d, 3) - 1)
        and is_zero(sp.radsimp(P_EXP.subs(d, 4) - sp.sqrt(sp.Rational(19, 3)) + 1)),
    )
    check(
        "[4] p(5), p(6) = 1.854102, 2.098780 (as in brane_flows)",
        abs(pv[5] - 1.854102) < 1e-6 and abs(pv[6] - 2.098780) < 1e-6,
        f"p = {pv}",
    )
    check(
        "[4] a_k(d) finite at d = 3",
        sp.limit(ak, d, 3).is_finite,
        f"a_k(3) = {sp.nsimplify(sp.limit(ak, d, 3))}",
    )
    rows_r, (u, p) = deformation_r(rc)
    xs = sp.Symbol("x", positive=True)
    usol = sp.solve(rows_r[0].subs(p, P_EXP), u)
    resr = [
        abs(
            complex(
                sp.N(
                    r_.subs({p: P_EXP, u: usol[0]}).subs(
                        {d: dd, alpha: 1, B: 1, xs: sp.Rational(1, 3)}
                    ),
                    30,
                )
            )
        )
        for r_ in rows_r
        for dd in (3, sp.Rational(13, 4), sp.Rational(7, 2), 5)
    ]
    check(
        "[4] r chart: x^p solves the V row (same p), the U row fixes u_p",
        len(usol) == 1 and max(resr) < 1e-25,
        f"max residual {max(resr):.1e}",
    )
    # change of chart: x = r - r0 = l e^{rho/l}(1 + O(e)), U = e^{2A}
    # => u_p = 2 l^p (a_k - 1/(1 + k l))... checked numerically against a_k
    upd = sp.lambdify(d, usol[0])
    akd = sp.lambdify(d, ak)
    for dd in (3.25, 3.5, 4.0, 5.0):
        l_ = float(ELL.subs(d, dd))
        p_ = float(P_EXP.subs(d, dd))
        k_ = p_ / l_
        # A = rho/l + a_k e e^{k rho}, x = int e^A = l e^{rho/l} + a_k e e^{(k+1/l) rho}/(k+1/l)
        # U = e^{2A} = (x/l)^2 (1 + 2 e e^{k rho}(a_k - a_k/(1 + k l)))
        # and e e^{k rho} = e (x/l)^p = eps x^p with eps = e l^{-p} (the V row),
        # so u_p = 2 a_k k l/(1 + k l)
        pred = 2 * akd(dd) * (k_ * l_ / (1 + k_ * l_))
        CHECKS.setdefault(
            "[4] r-chart u_p = domain-wall a_k after the change of chart", True
        )
        if abs(upd(dd) / pred - 1) > 1e-10:
            CHECKS["[4] r-chart u_p = domain-wall a_k after the change of chart"] = (
                False
            )
        log(f"      d = {dd}: u_p = {upd(dd):.10f}, from a_k {pred:.10f}")
    check(
        "[4] r-chart u_p = domain-wall a_k after the change of chart",
        CHECKS["[4] r-chart u_p = domain-wall a_k after the change of chart"],
    )

    log("[5] Liouville normal form")
    VL, ok, (rho, n, A, C) = liouville()
    check("[5] -psi_yy + V_L psi - lam psi = e^{s+2C-nA} x (mode operator)", ok)
    l_, v_ = sp.symbols("l v", positive=True)
    throat = sp.simplify(VL.subs({A: rho / l_, C: sp.log(v_) / 2}).doit())
    check("[5] throat: V_L = n^2 v/(4 l^2)", is_zero(throat - n**2 * v_ / (4 * l_**2)))
    edge = (n**2 * VFIX / (4 * ELL**2)).subs({n: d - 2, alpha: ALPHA_STAR, B: 1})
    edge = sp.powdenest(sp.simplify(edge), force=True)
    dev = max(abs(float(edge.subs(d, dd)) - 1) for dd in (3, 3.3, 3.7, 4, 5, 6, 9.5))
    check(
        "[5] physical operator at alpha_*: throat V_L = 1 (continuum edge)",
        is_zero(edge - 1) or dev < 1e-14,
        f"{edge}, max |V_L - 1| = {dev:.0e}",
    )
    c0 = sp.Symbol("c0")
    uv = sp.simplify(VL.subs({A: rho + c0, C: rho + c0}).doit())
    yb_y = sp.exp(-rho - c0)  # y_b - y = int_rho^oo e^{-C}
    check(
        "[5] boundary: V_L = (n^2 - 1)/(4 (y_b - y)^2)",
        is_zero(uv - (n**2 - 1) / (4 * yb_y**2)),
    )
    check(
        "[5] physical barrier (n = d-2) = (d-1)(d-3)/4, zero only at d = 3",
        is_zero(((n**2 - 1) / 4).subs(n, d - 2) - (d - 1) * (d - 3) / 4),
    )

    failed = [k_ for k_, v in CHECKS.items() if not v]
    log(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
