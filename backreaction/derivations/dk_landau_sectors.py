"""Every static channel of the charged W sector on a general diagonal magnetic
brane, Landau level by Landau level: the polarised LLL is the ONLY unstable
one.

Question.  alpha_* = 2/3 (paper sec. 4) is the endpoint of the instability in
the polarised lowest-Landau-level channel W_y = -i W_x, psi_0 = e^{-Bx^2/2}.
Are the other channels of the charged sector -- higher Landau levels n >= 1,
the anti-aligned polarisation, the components W_t, W_z -- stable at every
coupling, field and temperature, so that alpha_* bounds the whole W sector?
This module proves that they are.

Setting (as in `dk_zeromode.py`): coordinates (t, r, x, y, z), a general
diagonal metric g_tt(r), g_rr(r), g_xx = g_yy = g_pp(r), g_zz(r), SU(2) with
the coupling absorbed, background A^3_y = B x.  Charged fluctuation
W_M = A^1_M + i A^2_M; with D = d + i A^3 the linearised field strength is
F^W = D W - D W and the Landau ladder on y-independent functions is

    D_+ = D_x + i D_y = d_x - B x,   D_- = D_x - i D_y = d_x + B x,
    D_+ psi_n = -sqrt(B) psi_{n+1},   D_- psi_n = 2 n sqrt(B) psi_{n-1},
    psi_n = H_n(sqrt(B) x) e^{-B x^2/2}   (psi_{-1} = psi_{-2} = 0).

Static, z- and y-independent fluctuations (k_y = 0 is the Landau gauge
representative; k_z != 0 and time dependence are treated in
`dk_charged_dynamics.py`, where k_z enters through two squares that mix W_z
with W_+-, not as a mass) in the gauge W_r = 0.  The Landau structure sorts the sector by a single integer n:

    W_x + i W_y = p(r) psi_n,   W_x - i W_y = m(r) psi_{n-2},
    W_t = a(r) psi_{n-1},       W_z = c(r) psi_{n-1},

because D_+, D_- shift the level by one and the metric depends on r only, so
the transverse structure at every r is the flat Landau problem with an overall
warp e^{-2V}.  n = 0 is the polarised LLL (m = a = c = 0 identically).

Results (all asserted below for n = 0..4, with the general-n pattern read off)
-------
[1] Consistency: no O(W) tadpole; the x-integration is exact at every n (the
    sector closes -- no residual x, so no leakage into other levels).
[2] W_t and W_z decouple from (p, m) and from each other; each obeys a
    decoupled Sturm-Liouville equation (P_c a')' - (2n-1) B rho_c a = 0 with
    P_c, rho_c > 0 (the Landau mass (2n-1)B of a spin-0 component is positive
    for n >= 1), so neither carries a static source-free zero mode.
[3] The (p, m) density is  -(P/4)(N_n p'^2 + N_{n-2} m'^2) + T_n(p, m) with P = sqrt(-g_tt
    g_zz/g_rr) > 0 the radial weight of paper app. B.2 and, with the single
    transverse weight rho = sqrt(-g_tt g_rr g_zz)/g_pp (zeta in paper app. B.2),

        T_n = -(B rho N_{n-1}/4) [ 2n(n-1) p^2 + 2n p m + n m^2/(2(n-1)) ]  (n >= 2)
        T_1 = 0,        T_0 = +(B rho N_0/4) p^2   (the LLL: N_0 Q w^2, w = p/2),

    in the DENSITY sign convention of paper app. B.2 (density = -P w'^2 + Q w^2,
    i.e. minus the energy).

    N_k = int psi_k^2 dx.  The n >= 2 matrix has determinant ZERO and negative
    trace: the density is negative semi-definite (the energy positive
    semi-definite), with null vector (p, m) = (-1, 2(n-1))
    -- which is EXACTLY the residual pure gauge D_M lambda with
    lambda = lambda_0 psi_{n-1}, lambda_0 constant (checked by construction).
[4] Gauge invariance, n = 1..4: with W_r = e(r) psi_{n-1} reinstated, the
    density is invariant under (p, m, e) -> (p, m, e) + (-sqrt B,
    2(n-1) sqrt B, d/dr) lambda(r) up to a total r-derivative -- the
    background is on shell.  The reductions [1]-[3] are in the gauge W_r = 0;
    only this check keeps W_r (the time-dependent reduction with all five
    components is [1] of `dk_charged_dynamics.py`, n = 0..3).
[5] The transverse form at symbolic n.  Pointwise in x, for arbitrary
    W_+-(x) = W_x +- i W_y: the linearised F^W_xy = F^1_xy + i F^2_xy equals
    (i/2)(D_+ W_- - D_- W_+), and the O(W^2) part of the (x, y) energy
    (minus the density) equals
        (rho/2) [ (1/4)|D_+ W_- - D_- W_+|^2 - (B/2)(|W_+|^2 - |W_-|^2) ],
    with the factor 1/2 in front.  The ladder relations and N_n = 2n N_{n-1}
    then integrate it over x to minus T_n of [3], identically in n.

Theorem.  In the gauge W_r = 0 the static quadratic action of the charged
sector in level n is a sum of positive radial terms and T_n; for n >= 1 it is
positive semi-definite with kernel the r-independent pure gauges, so a static
source-free zero mode is pure gauge: NO instability in any n >= 1 channel, at
any B, any T, and any alpha (only P > 0, rho > 0 were used).  For n = 0 the
transverse term is strictly negative and the competition with the radial term
is the paper sec. 3 eigenvalue problem.  Hence alpha_* is the endpoint of the
instability of the ENTIRE charged W sector, not of one mode.  (The neutral
sector -- A^3 and the metric -- is outside this statement.)

The same T_n in the gauge-fixed language: the level-n aligned polarisation
carries the Landau energy (2n+1)B and the moment coupling -2B, so its
transverse mass is (2n-1)B, non-negative for n >= 1; T_n is that statement
with the gauge mode projected out.

Run:  uv run python -m backreaction.derivations.dk_landau_sectors   (~1 min)
"""

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
    if isinstance(e, sp.MatrixBase):
        return all(is_zero(x) for x in e)
    e = sp.expand(e)
    return e == 0 or sp.simplify(e) == 0


# ---------------------------------------------------------------------------
# geometry and gauge field
# ---------------------------------------------------------------------------
t, r, x, y, z = sp.symbols("t r x y z", real=True)
coords = (t, r, x, y, z)
Nd = 5
B = sp.Symbol("B", positive=True)

gtt = sp.Function("g_tt")(r)
grr = sp.Function("g_rr")(r)
gpp = sp.Function("g_pp")(r)
gzz = sp.Function("g_zz")(r)
gup = sp.diag(1 / gtt, 1 / grr, 1 / gpp, 1 / gpp, 1 / gzz)
att, brr, cpp, dzz = sp.symbols("att brr cpp dzz", positive=True)
POS = {gtt: -att, grr: brr, gpp: cpp, gzz: dzz}
sqrtg = sp.sqrt(-gtt * grr * gpp * gpp * gzz)
P_cf = sp.sqrt(-gtt * gzz / grr)  # radial weight P of paper app. B.2
rho_cf = sp.sqrt(-gtt * grr * gzz) / gpp  # transverse weight zeta of paper app. B.2


def psi(n):
    """Landau function psi_n = H_n(sqrt(B) x) exp(-B x^2/2); zero for n < 0."""
    if n < 0:
        return sp.S.Zero
    return sp.hermite(n, sp.sqrt(B) * x) * sp.exp(-B * x**2 / 2)


def Dplus(f):
    return sp.diff(f, x) - B * x * f


def Dminus(f):
    return sp.diff(f, x) + B * x * f


def field_strength(Amat):
    F = [[[sp.S.Zero] * Nd for _ in range(Nd)] for _ in range(3)]
    for ai in range(3):
        for mu in range(Nd):
            for nu in range(Nd):
                e = sp.diff(Amat[ai][nu], coords[mu]) - sp.diff(
                    Amat[ai][mu], coords[nu]
                )
                for bi in range(3):
                    for ci in range(3):
                        lc = sp.LeviCivita(ai + 1, bi + 1, ci + 1)
                        if lc != 0:
                            e += lc * Amat[bi][mu] * Amat[ci][nu]
                F[ai][mu][nu] = sp.expand(e)
    return F


def ym_density(Amat):
    """-1/4 sqrt(-g) F^a_{mn} F^{a mn} on the diagonal metric."""
    F = field_strength(Amat)
    L = sp.S.Zero
    for ai in range(3):
        for mu in range(Nd):
            for nu in range(Nd):
                if F[ai][mu][nu] != 0:
                    L += (
                        -sp.Rational(1, 4)
                        * gup[mu, mu]
                        * gup[nu, nu]
                        * F[ai][mu][nu] ** 2
                    )
    return sp.expand(sqrtg * L)


def gaussian_x(expr):
    """int dx of  (polynomial in x) * exp(-B x^2): exact Gaussian moments."""
    expr = sp.expand(expr * sp.exp(B * x**2))
    poly = sp.Poly(expr, x)
    out = sp.S.Zero
    for (k,), coeff in poly.terms():
        if k % 2:
            continue
        out += coeff * sp.gamma(sp.Rational(k + 1, 2)) / B ** sp.Rational(k + 1, 2)
    return sp.simplify(out)


def Nnorm(n):
    return gaussian_x(psi(n) ** 2) if n >= 0 else sp.S.Zero


def main():
    # ---------------------------------------------------------------------------
    # ladder algebra sanity
    # ---------------------------------------------------------------------------
    log(
        "[0] Landau ladder: D_+ psi_n = -sqrt(B) psi_{n+1}, D_- psi_n = 2n sqrt(B) psi_{n-1}"
    )
    ok = True
    for n in range(0, 5):
        ok &= is_zero(Dplus(psi(n)) + sp.sqrt(B) * psi(n + 1))
        ok &= is_zero(Dminus(psi(n)) - 2 * n * sp.sqrt(B) * psi(n - 1))
    check("[0] ladder relations n = 0..4", ok)

    # ---------------------------------------------------------------------------
    # the sector-n quadratic density
    # ---------------------------------------------------------------------------
    fn = {
        k: sp.Function(k, real=True)(r)
        for k in ("p1", "p2", "m1", "m2", "a1", "a2", "c1", "c2", "e1", "e2")
    }
    eps = sp.Symbol("epsilon", positive=True)

    def sector_fields(n, with_r=False):
        """Colour components A^1, A^2 of W = A^1 + i A^2 for level n."""
        P = (fn["p1"] + sp.I * fn["p2"]) * psi(n)
        M = (fn["m1"] + sp.I * fn["m2"]) * psi(n - 2)
        Wx = (P + M) / 2
        Wy = (P - M) / (2 * sp.I)
        Wt = (fn["a1"] + sp.I * fn["a2"]) * psi(n - 1)
        Wz = (fn["c1"] + sp.I * fn["c2"]) * psi(n - 1)
        Wr = (fn["e1"] + sp.I * fn["e2"]) * psi(n - 1) if with_r else sp.S.Zero
        W = [Wt, Wr, Wx, Wy, Wz]
        A = [[sp.S.Zero] * Nd for _ in range(3)]
        for mu in range(Nd):
            A[0][mu] = eps * sp.expand(sp.re(sp.expand(W[mu])))
            A[1][mu] = eps * sp.expand(sp.im(sp.expand(W[mu])))
        A[2][3] = B * x
        return A

    def reduced_density(n, with_r=False):
        A = sector_fields(n, with_r)
        L = ym_density(A)
        L = sp.expand(L)
        L1 = L.coeff(eps, 1)
        L2 = L.coeff(eps, 2)
        assert is_zero(L1), f"O(W) tadpole at n = {n}"
        return gaussian_x(L2)

    def quad_matrix(E, vars_):
        """Symmetric matrix of a quadratic form E in the listed symbols."""
        M = sp.zeros(len(vars_))
        for i, u in enumerate(vars_):
            for j, v in enumerate(vars_):
                M[i, j] = (
                    sp.Rational(1, 2) * sp.diff(E, u, v)
                    if i != j
                    else sp.diff(E, u, u) / 2
                )
        return sp.simplify(M)

    NUM = {att: 1.7, brr: 0.9, cpp: 1.3, dzz: 1.1, B: 2.0}
    results = {}
    for n in range(0, 5):
        log(f"[n = {n}] reducing the static sector-{n} density")
        E = reduced_density(n)
        # replace derivatives and functions by symbols
        syms = {}
        for k in fn:
            syms[k] = sp.Symbol(k, real=True)
            syms[k + "p"] = sp.Symbol(k + "p", real=True)
        Es = E
        for k, f in fn.items():
            Es = Es.subs(sp.Derivative(f, r), syms[k + "p"]).subs(f, syms[k])
        Es = sp.expand(Es)
        assert is_zero(sp.diff(Es, x)), "residual x"
        # [2] decoupling of W_t, W_z
        tz = [syms[k] for k in ("a1", "a2", "c1", "c2", "a1p", "a2p", "c1p", "c2p")]
        pm = [syms[k] for k in ("p1", "p2", "m1", "m2", "p1p", "p2p", "m1p", "m2p")]
        cross = sp.S.Zero
        for u in tz:
            for v in pm:
                cross += sp.diff(Es, u, v) ** 2
        check(f"[2] n = {n}: W_t, W_z decouple from (p, m)", is_zero(cross))
        if n >= 1:
            for name, kin, comp in (("W_t", "a", 0), ("W_z", "c", 4)):
                sub = {s: 0 for s in pm + tz}
                for k in (kin + "1", kin + "2", kin + "1p", kin + "2p"):
                    sub.pop(syms[k])
                Ec = sp.expand(Es.subs(sub))
                # expected: radial weight sqrt(-g) g^{rr} g^{comp comp} * (a1'^2 + a2'^2)/2 * N_{n-1}
                #           + Landau (2n-1) B sqrt(-g) g^{pp} g^{comp comp} (a1^2 + a2^2)/2 * N_{n-1}
                gcc = gup[comp, comp]
                wrad = sqrtg * gup[1, 1] * gcc
                wlan = (2 * n - 1) * B * sqrtg * gup[2, 2] * gcc
                sgn = (
                    -1 if comp == 0 else 1
                )  # W_t enters with g^{tt} < 0: overall sign flip
                Eexp = (
                    sgn
                    * Nnorm(n - 1)
                    / 2
                    * (
                        wrad * (syms[kin + "1p"] ** 2 + syms[kin + "2p"] ** 2)
                        + wlan * (syms[kin + "1"] ** 2 + syms[kin + "2"] ** 2)
                    )
                )
                # sign convention: the density is -1/4 F^2 sqrt(-g); the r-kinetic
                # term of a magnetic component comes with a minus sign (as -P w'^2
                # in paper app. B.2).  Compare up to that overall sign.
                diff = sp.simplify(Ec + sgn * Eexp)
                check(
                    f"[2] n = {n}: {name} density = -(radial + (2n-1)B Landau) N_(n-1)/2, "
                    f"definite sign",
                    is_zero(diff),
                )
        # [3] the (p, m) block
        sub0 = {s: 0 for s in tz}
        Epm = sp.expand(Es.subs(sub0))
        # radial part
        rad = sp.simplify(sp.diff(Epm, syms["p1p"], 2) / 2)
        check(
            f"[3] n = {n}: radial weight of p is -P N_n/4 (P = sqrt(-g_tt g_zz/g_rr))",
            is_zero(sp.simplify(rad**2 - (P_cf * Nnorm(n) / 4) ** 2))
            and float(rad.subs(POS).subs(NUM)) < 0,
        )
        nocross = all(
            is_zero(sp.diff(Epm, syms[u], syms[v]))
            for u in ("p1p", "p2p", "m1p", "m2p")
            for v in ("p1", "p2", "m1", "m2", "p1p", "p2p", "m1p", "m2p")
            if u != v
        )
        check(f"[3] n = {n}: no p'-m' or p'-p cross terms", nocross)
        # transverse block on (p1, m1) and (p2, m2); Re/Im must not mix
        T = quad_matrix(
            Epm.subs({syms[k]: 0 for k in ("p1p", "p2p", "m1p", "m2p")}),
            [syms[k] for k in ("p1", "m1", "p2", "m2")],
        )
        check(
            f"[3] n = {n}: Re/Im blocks identical and decoupled",
            is_zero(T[0:2, 0:2] - T[2:4, 2:4]) and is_zero(T[0:2, 2:4]),
        )
        T2 = sp.simplify(T[0:2, 0:2] / (B * rho_cf))
        results[n] = T2
        log(f"      T_{n} / (B rho) = {T2.tolist()}")
        if n == 0:
            check(
                "[3] n = 0: T_0 = +B rho N_0/4 p^2 (N_0 Q w^2 with w = p/2)",
                is_zero((T2[0, 0] - Nnorm(0) / 4).subs(POS)) and is_zero(T2[1, 1]),
            )
        elif n == 1:
            check("[3] n = 1: T_1 = 0 identically", is_zero(T2))
        else:
            expected = (
                -Nnorm(n - 1)
                / 4
                * sp.Matrix([[2 * n * (n - 1), n], [n, sp.Rational(n, 2 * (n - 1))]])
            )
            check(
                f"[3] n = {n}: T_n = -(B rho N_(n-1)/4) [[2n(n-1), n],[n, n/(2(n-1))]]",
                is_zero((T2 - expected).subs(POS)),
            )
            check(
                f"[3] n = {n}: det T_n = 0, trace definite (negative-semidefinite density "
                f"= positive-semidefinite energy)",
                is_zero(T2.det()) and float(T2.trace().subs(POS).subs(NUM)) < 0,
            )
            null = sp.Matrix([-1, 2 * (n - 1)])
            check(
                f"[3] n = {n}: null vector (-1, 2(n-1)) = pure gauge D_+/D_- of psi_(n-1)",
                is_zero(T2 * null),
            )

    # ---------------------------------------------------------------------------
    # [4] gauge invariance with W_r reinstated (n = 2 and n = 3)
    # ---------------------------------------------------------------------------
    log("[4] gauge invariance of the density with W_r = e psi_{n-1}")
    lam = sp.Function("lambda", real=True)(r)
    for n in (1, 2, 3, 4):
        E = reduced_density(n, with_r=True)
        # pure gauge: W_M = D_M (lambda psi_{n-1}):  P = D_+ -> -sqrt(B) lambda psi_n,
        # M = D_- -> 2(n-1) sqrt(B) lambda psi_{n-2},  W_r = lambda' psi_{n-1}
        shift = {
            fn["p1"]: fn["p1"] - sp.sqrt(B) * lam,
            fn["m1"]: fn["m1"] + 2 * (n - 1) * sp.sqrt(B) * lam,
            fn["e1"]: fn["e1"] + sp.Derivative(lam, r),
        }
        Eg = E.subs(shift, simultaneous=True).doit()
        dE = sp.expand(sp.simplify(Eg - E))
        # must be a total derivative: its Euler-Lagrange derivative wrt lambda vanishes
        # (and also wrt every other field); test via the variational derivative
        ok = True
        for f in list(fn.values()) + [lam]:
            el = sp.diff(dE, f) - sp.diff(sp.diff(dE, sp.Derivative(f, r)), r)
            ok &= is_zero(sp.simplify(el.doit()))
        check(
            f"[4] n = {n}: density shifts by a total derivative under the pure gauge",
            ok,
        )
        # and the pure-gauge configuration itself has zero action density up to a total derivative
        pure = {
            fn["p1"]: -sp.sqrt(B) * lam,
            fn["m1"]: 2 * (n - 1) * sp.sqrt(B) * lam,
            fn["e1"]: sp.Derivative(lam, r),
        }
        Ep = sp.expand(
            E.subs({f: 0 for f in fn.values() if f not in pure})
            .subs(pure, simultaneous=True)
            .doit()
        )
        el = sp.simplify(
            (sp.diff(Ep, lam) - sp.diff(sp.diff(Ep, sp.Derivative(lam, r)), r)).doit()
        )
        check(
            f"[4] n = {n}: pure-gauge configuration is stationary (zero action mod boundary)",
            is_zero(el),
        )

    # ---------------------------------------------------------------------------
    # [5] the closed form at symbolic n, from the ladder relations alone
    # ---------------------------------------------------------------------------
    # [3] checks T_n case by case for n = 2..4.  [5a] derives the transverse energy
    # pointwise from the field strength, for arbitrary x-profiles W_+(x), W_-(x);
    # [5b] integrates it over x at symbolic n with D_+ psi_{n-2} = -sqrt(B)
    # psi_{n-1}, D_- psi_n = 2n sqrt(B) psi_{n-1} and N_n = 2n N_{n-1}, and finds
    # exactly minus the T_n of [3] (T is minus the energy).
    log("")
    log("[5a] transverse energy pointwise in x, arbitrary W_+-(x)")
    xf = {k: sp.Function(k, real=True)(x) for k in ("P1", "P2", "M1", "M2")}
    Wp = xf["P1"] + sp.I * xf["P2"]
    Wm = xf["M1"] + sp.I * xf["M2"]
    Wx_, Wy_ = (Wp + Wm) / 2, (Wp - Wm) / (2 * sp.I)
    Ax = [[sp.S.Zero] * Nd for _ in range(3)]
    Ax[0][2], Ax[0][3] = eps * sp.re(sp.expand(Wx_)), eps * sp.re(sp.expand(Wy_))
    Ax[1][2], Ax[1][3] = eps * sp.im(sp.expand(Wx_)), eps * sp.im(sp.expand(Wy_))
    Ax[2][3] = B * x
    Fx = field_strength(Ax)
    FW = sp.expand(Fx[0][2][3] + sp.I * Fx[1][2][3]).coeff(eps, 1)
    check(
        "[5a] F^W_xy = (i/2)(D_+ W_- - D_- W_+)",
        is_zero(FW - sp.I / 2 * (Dplus(Wm) - Dminus(Wp))),
    )
    dens_xy = sp.expand(
        -sp.Rational(1, 2)
        * sqrtg
        * gup[2, 2]
        * gup[3, 3]
        * sum(Fx[a][2][3] ** 2 for a in range(3))
    ).coeff(eps, 2)
    Fpm = Dplus(Wm) - Dminus(Wp)

    def cc(e):
        """Complex conjugate: every function and symbol here is real."""
        return e.subs(sp.I, -sp.I)

    E_expect = (
        rho_cf
        / 2
        * (sp.Rational(1, 4) * Fpm * cc(Fpm) - B / 2 * (Wp * cc(Wp) - Wm * cc(Wm)))
    )
    check(
        "[5a] (x, y) energy = (rho/2)[(1/4)|D_+W_- - D_-W_+|^2 - (B/2)(|W_+|^2 - |W_-|^2)]",
        is_zero((-dens_xy - E_expect).subs(POS)),
    )
    check(
        "[5a] negative control: without the 1/2 the identity fails",
        not is_zero((-dens_xy - 2 * E_expect).subs(POS)),
    )
    log("[5b] closed form at symbolic n")
    ns, Bs, Ns, ps, ms = sp.symbols("n B N p m", positive=True)
    # x-integral of (1/4)|D_+W_- - D_-W_+|^2 - (B/2)(|W_+|^2 - |W_-|^2) with
    # W_+ = p psi_n, W_- = m psi_{n-2}:  D_+W_- - D_-W_+ = -sqrt(B)(m + 2np) psi_{n-1}
    bracket = sp.Rational(1, 4) * Bs * Ns * (ms + 2 * ns * ps) ** 2 - Bs / 2 * (
        2 * ns * Ns * ps**2 - Ns * ms**2 / (2 * (ns - 1))
    )
    Tn_closed = (
        Bs
        * Ns
        / 4
        * (2 * ns * (ns - 1) * ps**2 + 2 * ns * ps * ms + ns / (2 * (ns - 1)) * ms**2)
    )
    check(
        "[5b] symbolic n: (1/2) x-integral of the bracket = -T_n of [3] (per rho), identically in n",
        is_zero(bracket / 2 - Tn_closed),
    )
    M5 = sp.Matrix([[2 * ns * (ns - 1), ns], [ns, ns / (2 * (ns - 1))]])
    check("[5b] symbolic n: det = 0 identically", is_zero(M5.det()))
    check(
        "[5b] symbolic n: null vector (-1, 2(n-1)) is the pure gauge",
        all(is_zero(e) for e in M5 * sp.Matrix([-1, 2 * (ns - 1)])),
    )

    # ---------------------------------------------------------------------------
    # summary
    # ---------------------------------------------------------------------------
    log("")
    log("SUMMARY (static, W_r = 0, general diagonal brane)")
    log(
        "  density = -(P/4) N_n p'^2 - (P/4) N_{n-2} m'^2 + T_n(p, m)  [W_t, W_z decoupled, definite]"
    )
    log(
        "  T_0 = +(B rho N_0/4) p^2 > 0    : the polarised LLL, N_0 Q w^2 with w = p/2 (density sign)"
    )
    log("  T_1 = 0                          : marginal = pure gauge only")
    log("  T_n = -(B rho N_{n-1}/4) [[2n(n-1), n],[n, n/(2(n-1))]], det = 0, n >= 2")
    log(
        "  => the energy functional is positive semi-definite for every n >= 1 with kernel"
    )
    log(
        "     the r-independent pure gauge; no static zero mode; alpha_* bounds the whole W sector"
    )
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        raise SystemExit(1)
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
