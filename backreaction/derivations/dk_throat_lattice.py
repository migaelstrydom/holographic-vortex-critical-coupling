"""The lattice sector as alpha -> alpha_*: the throat limit is a fixed BTZ x R^2
problem (paper app. C.4).

Setting.  beta -> oo at fixed T drives the D'Hoker-Kraus brane to the AdS3 x R^2
fixed point and the coupling to alpha_* = 2/3 (paper sec. 4).  What the
LATTICE does there is decided not by the large-s tail of the kernel
K(s; beta) (paper sec. 5.1) at s = gamma B_c(beta) -> oo; the object that
decides the selection is

    lim_{beta -> oo}  K(gamma B_c(beta); beta) / (positive beta-dependent factor)

at FIXED gamma, and that limit is a fixed, beta-independent kernel on the BTZ
throat -- not the asymptotic tail of any single kernel.  The reason is that
every quartic integral is dominated by the BOTTOM of the throat, where the
background is exactly BTZ x R^2 in the fixed-T frame.

Results
-------
[1] The BTZ parameters are FIXED, not free.  derivations/dk_throat.py [5] shows
    U = ((r-r0)^2 - h^2)/l3^2, e^{2W} = c_W (r-r0)^2, e^{2V} = v solves the
    system for any (r0, h, c_W).  The fixed-T frame of `dk_background`
    (horizon at r_p = 1, U'(r_p) = 4 <=> T = 1/pi) plus l3^2 = 1/3 pins

        h = 2/3,   r0 = 1/3,   U -> (r - 1)(3r + 1),   e^{2V} = v = sqrt(beta/6),

    with c_W the one remaining (pure z-rescaling) frame constant.  So the
    beta -> oo near-horizon geometry is one definite metric, with no fitted
    parameter, and B_c -> 3 v = sqrt(3 beta / 2) as b -> 1.

[2] One radial operator for everything.  Both the condensate and the response
    live on the SAME throat operator.  With rho = r - r0,

        L_m y = [rho (rho^2 - h^2) y']' - m rho y ,

    the paper sec. 3 zero mode is L_{-b} w0 = 0 with b = B l3^2 e^{-2V} (the paper sec. 4 throat invariant), and the induced-field (photon) BVP of the paper sec. 5.1 and app. C kernel at reciprocal-lattice harmonic s = G^2 is

        L_lambda bhat = -lambda rho w0^2,   lambda = s l3^2 / v = gamma * b ,

    where gamma = s / B_c = G^2 / B is the B-INDEPENDENT lattice shape variable
    of `dk_selection`.  Since b -> 1 at the fixed point, lambda -> gamma
    exactly: the screening parameter of the harmonic IS the lattice harmonic.
    That is why the limit at fixed gamma exists and is beta-free.

[3] Exact solutions.  In z = h^2/rho^2 in (0, 1] (z = 1 horizon, z = 0 throat
    top) the operator becomes, with h cancelling identically (scale invariance),

        4 z^2 (1-z) y'' - 4 z^2 y' - m y = 0 ,

    a hypergeometric equation with exponents {0,0} at the horizon and
    {(1 -+ mu)/2} at the top, mu = sqrt(1 + m).  In AdS3 language m = m^2 l3^2
    and Delta = 1 + mu.  The two natural solutions are, with a = (1 + mu)/2,

        y_H = z^a 2F1(a, a; 1;      1 - z)     regular at the horizon,
        y_I = z^a 2F1(a, a; 1 + mu; z)         normalisable at the top.

[4] The marginal condensate in closed form.  At the fixed point b -> 1, i.e.
    m = -1 and mu = 0, so the horizon-regular zero mode is exactly

        R(rho) = sqrt(z) 2F1(1/2, 1/2; 1; 1-z) = (2/pi) sqrt(z) K(1-z),

    K the complete elliptic integral of the first kind, z = h^2/rho^2.  Its top
    behaviour R ~ (2h/pi rho) ln(4 rho/h) is the log of the degenerate
    (Delta = 1) case: the mode is NOT normalisable in the throat, which is what
    spreads it over the whole plateau and makes the overall amplitude vanish
    like a power of 1/ln beta.  That amplitude is gamma-independent, so it
    cancels from every lattice comparison.  At finite beta the same formula with
    a = (1 + i nu)/2, nu = sqrt(b_h - 1), gives the O(nu^2) correction.

[5] An exact tail identity, at EVERY beta.  For the paper sec. 3 operator
    (p2 w')' + B p1 w = 0 (p2 = U e^W, p1 = e^{W-2V}),

        d/dr [ p2 w^3 w' ] = 3 p2 w^2 w'^2 - B p1 w^4

    identically on solutions, and both endpoint terms vanish (p2 ~ (r - r_p) at
    the horizon; p2 w^3 w' ~ r^{-6} at the boundary).  Hence, with f = w0^2,

        <f, A f> = int p2 ((w0^2)')^2 dr = (4/3) B_c int p1 w0^4 dr = (4/3) B_c C4 ,

    and since the abelian kernel is (C4 - X)(s) = <u, T (T + s)^{-1} u> with
    u = M^{1/2} f, T = M^{-1/2} A M^{-1/2} (A = -d/dr p2 d/dr >= 0, M = p1 > 0),

        (C4 - X)(s) = (4/3) B_c C4 / s + O(1/s^2)   exactly, at every beta.

    So the abelian tail is c/s with c in closed form: no ln s, no log-periodic
    modulation.  The tail does not decide the lattice selection, which is set
    at fixed gamma (see above).

[6] Complete monotonicity survives in the limit.  A >= 0 and M > 0 give the
    positive spectral representation (C4 - X)(s) = int dmu(tau) tau/(tau + s),
    dmu >= 0, so the abelian kernel is completely monotone in s -- hence in
    gamma -- at every beta, and so is e^{-gamma/2} K(gamma B_c) (a product of CM
    functions).  Montgomery then forces the triangular lattice.  In the throat
    limit the discrete radial tower becomes the continuum tau >= 1: the
    marginal mode R is the threshold state (A R = M R, non-normalisable), so
    gamma = -1 (s = -B_c, where the probe tower starts) is the BRANCH POINT
    mu = 0 of the throat, not a pole.  The metric sector, which can spoil
    complete monotonicity, is not treated here (see dk_throat_channels.py).

CHECKS (all asserted)
---------------------
 [1] h = 2/3, r0 = 1/3 from r_p = 1, U'(r_p) = 4, l3^2 = 1/3; U = (r-1)(3r+1)
 [2] the z-normal form, with h cancelling identically
 [3] y_H and y_I are hypergeometric solutions (operator identities, symbolic mu)
 [4] the marginal case mu = 0 and its elliptic-K form
 [5] lambda = gamma * b exactly
 [6] the exact-derivative identity, and the 3 <-> 4/3 bookkeeping
 [7] the endpoint terms of [5] vanish, from the indicial data at both ends
"""

import sympy as sp

CHECKS = {}


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


r, rho, h, r0, m, z, w_, B, gam, b_sym = sp.symbols(
    "r rho h r_0 m z w B gamma b", positive=True
)
mu = sp.Symbol("mu")
L3SQ = sp.Rational(1, 3)


def main():
    print("[1] the BTZ parameters in the fixed-T frame")
    U = ((r - r0) ** 2 - h**2) / L3SQ
    sol = sp.solve([U.subs(r, 1), sp.diff(U, r).subs(r, 1) - 4], [r0, h], dict=True)
    sol = [s for s in sol if s[h] > 0][0]
    check(
        "[1] r_p = 1 and U'(r_p) = 4 give h = 2/3, r0 = 1/3",
        sol[h] == sp.Rational(2, 3) and sol[r0] == sp.Rational(1, 3),
        f"h = {sol[h]}, r0 = {sol[r0]}",
    )
    check(
        "[1] U -> (r-1)(3r+1)",
        sp.simplify(U.subs(sol) - (r - 1) * (3 * r + 1)) == 0,
        f"U = {sp.factor(sp.expand(U.subs(sol)))}",
    )
    # B_c = b v / l3^2 with b -> 1 and v = sqrt(beta/6)
    beta = sp.Symbol("beta", positive=True)
    v_fp = sp.sqrt(beta / 6)
    check(
        "[1] B_c -> sqrt(3 beta/2) at marginality (b = 1, v = sqrt(beta/6))",
        sp.simplify(v_fp / L3SQ - sp.sqrt(3 * beta / 2)) == 0,
    )

    print("[2] the z = h^2/rho^2 normal form (h cancels)")
    Y0, Y1, Y2 = sp.symbols("Y0 Y1 Y2")
    zr = h**2 / rho**2
    zp, zpp = sp.diff(zr, rho), sp.diff(zr, rho, 2)
    Lop = (
        rho * (rho**2 - h**2) * (Y2 * zp**2 + Y1 * zpp)
        + (3 * rho**2 - h**2) * (Y1 * zp)
        - m * rho * Y0
    )
    Lop = sp.simplify(sp.expand(Lop.subs(rho, h / sp.sqrt(z))))
    target = 4 * z**2 * (1 - z) * Y2 - 4 * z**2 * Y1 - m * Y0
    check(
        "[2] L_m / (h/sqrt z) = 4 z^2 (1-z) y'' - 4 z^2 y' - m y  (h-free)",
        sp.simplify(sp.expand(Lop / (h / sp.sqrt(z)) - target)) == 0,
    )

    print("[3] the two exact solutions are hypergeometric")
    a = (1 + mu) / 2
    mval = mu**2 - 1
    U_ = sp.Function("U_")
    y = z**a * U_(z)
    E = sp.expand(
        4 * z**2 * (1 - z) * sp.diff(y, z, 2) - 4 * z**2 * sp.diff(y, z) - mval * y
    )
    E = sp.expand(sp.simplify(E / (4 * z ** (a + 1))))
    H_top = (
        z * (1 - z) * sp.diff(U_(z), z, 2)
        + ((1 + mu) - (2 * a + 1) * z) * sp.diff(U_(z), z)
        - a**2 * U_(z)
    )
    check(
        "[3] y_I = z^a 2F1(a,a;1+mu;z) solves it (normalisable at the top)",
        sp.simplify(sp.expand(E - H_top)) == 0,
    )
    w_sym = sp.Symbol("w", positive=True)
    V_ = sp.Function("V_")
    yr = z**a * V_(1 - z)
    Er = sp.expand(
        4 * z**2 * (1 - z) * sp.diff(yr, z, 2) - 4 * z**2 * sp.diff(yr, z) - mval * yr
    )
    Er = sp.expand(sp.simplify(Er.subs(z, 1 - w_sym) / (4 * (1 - w_sym) ** (a + 1))))
    H_hor = (
        w_sym * (1 - w_sym) * sp.diff(V_(w_sym), w_sym, 2)
        + (1 - (2 * a + 1) * w_sym) * sp.diff(V_(w_sym), w_sym)
        - a**2 * V_(w_sym)
    )
    check(
        "[3] y_H = z^a 2F1(a,a;1;1-z) solves it (regular at the horizon)",
        sp.simplify(sp.expand(Er - H_hor)) == 0,
    )
    check(
        "[3] top exponents (1 -+ mu)/2, i.e. Delta = 1 + mu with mu = sqrt(1+m)",
        sp.simplify(
            sp.solve(
                sp.Eq(4 * sp.Symbol("p") * (sp.Symbol("p") - 1), m), sp.Symbol("p")
            )[1]
            - (1 + sp.sqrt(1 + m)) / 2
        )
        == 0,
    )

    print("[4] the marginal condensate")
    Fz = sp.hyper((sp.Rational(1, 2), sp.Rational(1, 2)), (1,), 1 - z)
    R = sp.sqrt(z) * Fz
    check(
        "[4] R = sqrt(z) 2F1(1/2,1/2;1;1-z) solves the m = -1 equation",
        sp.simplify(
            sp.hyperexpand(
                4 * z**2 * (1 - z) * sp.diff(R, z, 2) - 4 * z**2 * sp.diff(R, z) + R
            )
        )
        == 0,
    )
    x = sp.Symbol("x")
    check(
        "[4] 2F1(1/2,1/2;1;x) = (2/pi) K(x)",
        sp.simplify(
            sp.hyperexpand(sp.hyper((sp.Rational(1, 2), sp.Rational(1, 2)), (1,), x))
            - 2 * sp.elliptic_k(x) / sp.pi
        )
        == 0,
    )
    check(
        "[4] mu = 0 <=> m = -1 <=> b = 1 <=> alpha = alpha_* = 2/3",
        sp.simplify((mu**2 - 1).subs(mu, 0) + 1) == 0
        and sp.simplify(
            sp.solve(
                sp.Eq(sp.sqrt(2 / (3 * sp.Symbol("al", positive=True))), 1),
                sp.Symbol("al", positive=True),
            )[0]
            - sp.Rational(2, 3)
        )
        == 0,
    )

    print("[5] the screening parameter of the harmonic is the lattice harmonic")
    # lambda = s l3^2 / v,  b = B_c l3^2 / v,  s = gamma B_c   =>  lambda = gamma b
    Bc, vsym, s_sym = sp.symbols("B_c v s", positive=True)
    lam = s_sym * L3SQ / vsym
    check(
        "[5] lambda = gamma * b  (-> gamma as b -> 1)",
        sp.simplify(lam.subs(s_sym, gam * Bc) - gam * (Bc * L3SQ / vsym)) == 0,
    )

    print("[6] the exact tail identity")
    p1, p2 = sp.Function("p1"), sp.Function("p2")
    wf = sp.Function("w")
    # eliminate w'' by the SL equation (p2 w')' = -B p1 w
    wpp = (-B * p1(r) * wf(r) - sp.diff(p2(r), r) * sp.diff(wf(r), r)) / p2(r)
    lhs = sp.diff(p2(r) * wf(r) ** 3 * sp.diff(wf(r), r), r)
    lhs = lhs.subs(sp.Derivative(wf(r), (r, 2)), wpp).doit()
    rhs = 3 * p2(r) * wf(r) ** 2 * sp.diff(wf(r), r) ** 2 - B * p1(r) * wf(r) ** 4
    check(
        "[6] d/dr[p2 w^3 w'] = 3 p2 w^2 w'^2 - B p1 w^4  on solutions",
        sp.simplify(sp.expand(lhs - rhs)) == 0,
    )
    check(
        "[6] hence int p2 ((w^2)')^2 dr = (4/3) B int p1 w^4 dr   (4 w^2 w'^2 = ((w^2)')^2)",
        sp.simplify(
            sp.expand(
                sp.diff(wf(r) ** 2, r) ** 2 - 4 * wf(r) ** 2 * sp.diff(wf(r), r) ** 2
            )
        )
        == 0,
    )

    print("[7] the endpoint terms vanish")
    eps, kap, c1, c2 = sp.symbols("epsilon kappa c1 c2", positive=True)
    # horizon: p2 = kappa*(r-r_p)+..., w regular  =>  p2 w^3 w' ~ eps -> 0
    check(
        "[7] horizon: p2 ~ kappa (r - r_p), w and w' finite  =>  p2 w^3 w' -> 0",
        sp.limit(kap * eps * c1**3 * c2, eps, 0) == 0,
    )
    # boundary: p2 = U e^W ~ r^3, w ~ c/r^2 (source-free d = 4 falloff)
    check(
        "[7] boundary: p2 ~ r^3 and w ~ c/r^2  =>  p2 w^3 w' ~ r^{-6} -> 0",
        sp.limit(r**3 * (c1 / r**2) ** 3 * sp.diff(c1 / r**2, r), r, sp.oo) == 0,
    )

    print("")
    print("SUMMARY")
    print("  throat geometry :  U = (r-1)(3r+1), e^{2V} = sqrt(beta/6), l3^2 = 1/3")
    print(
        "  one operator    :  L_m,  m = -b (condensate), m = lambda = gamma b (photon)"
    )
    print("  marginal mode   :  R = (2/pi) sqrt(z) K(1-z),  z = h^2/(r-1/3)^2")
    print("  exact tail      :  (C4 - X)(s) -> (4/3) B_c C4 / s   at every beta")
    print("  CM              :  abelian kernel is completely monotone; gamma = -1 is")
    print("                     the throat BRANCH POINT mu = 0, not a pole")
    nfail = sum(1 for v in CHECKS.values() if not v)
    print(f"  {len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    assert nfail == 0, f"{nfail} check(s) failed"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
