"""The linearised SU(2) W zero-mode operator on a GENERAL diagonal magnetic
brane.

Background (boundary at r -> oo, regular horizon at r_p with U(r_p) = 0):

    ds^2 = -U(r) dt^2 + dr^2/U(r) + e^{2V(r)}(dx^2 + dy^2) + e^{2W(r)} dz^2,

SU(2) Yang-Mills, coupling absorbed
(F^a_MN = d_M A^a_N - d_N A^a_M + eps^{abc} A^b_M A^c_N),
background A^3_y = B x, and the polarised lowest-Landau-level fluctuation

    W_x = w(r) psi(x),  W_y = -i W_x,  psi = exp(-B x^2 / 2),
    W_t = W_z = W_r = 0,  dA^3 = 0   (at O(w)).

The derivation is done for FOUR independent diagonal metric functions
(g_tt, g_rr, g_pp = g_xx = g_yy, g_zz) of a single radial coordinate, so the
same reduced quadratic form specialises to (i) the (U, V, W) brane [module,
horizon, boundary], (ii) AdS5-Schw in u-coordinates [CHECK 1], and (iii) the
O(alpha) perturbed u-metric of derivations/bc_shift.py [CHECK 2].

Results
-------
Quadratic action density (per unit t,y,z volume, after the exact Gaussian x
integral int psi^2 dx = sqrt(pi/B), overall colour/coupling constant stripped):

    E/psi^2  =  -P(r) w'^2 + Q(r) w^2,
    P = sqrt(-g_tt g_zz / g_rr),   Q = B * sqrt(-g_tt g_rr g_zz) / g_pp,

Euler-Lagrange (Sturm-Liouville) ODE:  (P w')' + Q w = 0, i.e.
    -(P w')' = B * rho(r) w,   rho(r) = sqrt(-g_tt g_rr g_zz)/g_pp = Q/B,
so B is the eigenvalue, P the SL weight and rho the natural norm density
(the analogue of J_2 = int rho w^2).  On the brane (g_rr = 1/U,
g_pp = e^{2V}, g_zz = e^{2W}):  P = U e^{W},  rho = e^{W - 2V},
Q = B e^{W - 2V}.

How B enters (kept separate then combined, block [1b]): on the polarised LLL
ansatz the transverse gluon field strength F^W_{xy} = 0 identically (metric
independent), so the transverse Landau kinetic energy drops out entirely; the
ONLY w^2 term is the non-abelian magnetic-moment / diamagnetic cross term
-2B w^2 psi^2 from (F^3_{xy})^2 = (B - w^2 psi^2)^2, carrying the raised
transverse metric g^{pp}g^{pp} = e^{-4V}; combined with sqrt(-g) this is the
single e^{-2V} weight rho.

CHECK 1  (exact limit): u-coordinate AdS5-Schw reduces P -> f/u, Q -> B/u and
the ODE to the probe-limit zero mode f w'' + (f' - f/u) w' + B w = 0; density to
-(f/u) w'^2 + (B/u) w^2.
CHECK 2  (first variation): the perturbed u-metric of derivations/bc_shift.py gives, to O(alpha),
dP = (f/2u)(H_tt + H_zz - H_uu), dQ = (B/2u)(H_tt + H_uu + H_zz - 2 H_xx),
matched symbolically to these closed forms AND numerically to dP/dQ of systems/bc_shift.py.
Consistent truncation: the linearised YM equations for the omitted components
(W_t, W_z, W_r colours 1,2 and all of dA^3 colour 3) carry NO O(w) source.
Horizon:  w'(r_p) = -(B e^{-2V(r_p)} / U'(r_p)) w(r_p).
Boundary:  w = c0(1 + (B/2)r^{-2} ln r + ...) + c2 r^{-2} + ... ; source c0 at
r^0 (log-free), source-free c0 = 0, source-log at r^{-2}; the DK O(r^{-4} ln r)
corrections to V, W do not touch the r^0, r^{-2} data.

Generates systems/dk_zeromode.py (SL coefficients of (r, U, U', V, V', W, W', B)).

Run:  uv run python -m backreaction.derivations.dk_zeromode   (~1 min)
"""

import sys
import time

import sympy as sp

sys.setrecursionlimit(100000)
T0_START = time.time()


def log(msg):
    print(f"[{time.time() - T0_START:7.1f}s] {msg}", flush=True)


def is_zero(e):
    e = sp.expand(e)
    if e == 0:
        return True
    return sp.simplify(sp.cancel(sp.together(e))) == 0


def eq_pos(e1, e2):
    """Equal after mapping the metric functions to manifestly positive symbols
    (g_tt = -att, g_rr = brr, g_pp = cpp, g_zz = dzz; all > 0).  Correct for
    quantities that are single-valued positive square roots on the physical
    domain, where sqrt(X^2) -> X unambiguously."""
    return is_zero(sp.simplify((e1 - e2).subs(POS)))


# ---------------------------------------------------------------------------
# General diagonal metric and the LLL ansatz
# ---------------------------------------------------------------------------
t, x, y, z = sp.symbols("t x y z", real=True)
r = sp.Symbol("r", positive=True)  # radial coordinate > 0
coords = [t, r, x, y, z]
Nd = 5
B = sp.Symbol("B", positive=True)

gtt = sp.Function("g_tt")(r)  # < 0
grr = sp.Function("g_rr")(r)
gpp = sp.Function("g_pp")(r)  # g_xx = g_yy (planar isotropy)
gzz = sp.Function("g_zz")(r)

att, brr, cpp, dzz = sp.symbols("att brr cpp dzz", positive=True)
POS = {gtt: -att, grr: brr, gpp: cpp, gzz: dzz}

gdn = sp.diag(gtt, grr, gpp, gpp, gzz)
gup = sp.diag(1 / gtt, 1 / grr, 1 / gpp, 1 / gpp, 1 / gzz)
sqrtg = sp.sqrt(-gtt * grr * gpp * gpp * gzz)

w = sp.Function("w", real=True)


def main():
    wp = sp.Derivative(w(r), r)
    psi = sp.exp(-B * x**2 / 2)

    # gauge potential A[a][mu], a = 1,2,3.  D = d + i A^3 (verified convention:
    # (D_x - i D_y) psi = 0 gives psi = exp(-B x^2/2); "D = d - i A^3" would
    # give the opposite sign).  Field assignment mirrors derivations/bc_shift.py.
    A = [[sp.S.Zero] * Nd for _ in range(3)]
    A[0][2] = w(r) * psi  # A^1_x = w psi          (W_x = w psi)
    A[1][3] = -w(r) * psi  # A^2_y = -w psi         (W_y = -i W_x)
    A[2][3] = B * x  # A^3_y = B x

    def field_strength(Amat):
        F = [[[sp.S.Zero] * Nd for _ in range(Nd)] for _ in range(3)]
        for ai in range(3):
            for m in range(Nd):
                for n in range(Nd):
                    e = sp.diff(Amat[ai][n], coords[m]) - sp.diff(
                        Amat[ai][m], coords[n]
                    )
                    for bi in range(3):
                        for ci in range(3):
                            lc = sp.LeviCivita(ai + 1, bi + 1, ci + 1)
                            if lc != 0:
                                e += lc * Amat[bi][m] * Amat[ci][n]
                    F[ai][m][n] = sp.expand(e)
        return F

    F = field_strength(A)

    # action density -1/4 sqrt(-g) F^a_{mn} F^{a mn} on the diagonal metric
    L = sp.S.Zero
    for ai in range(3):
        for m in range(Nd):
            for n in range(Nd):
                if F[ai][m][n] != 0:
                    L += -sp.Rational(1, 4) * gup[m, m] * gup[n, n] * F[ai][m][n] ** 2
    L = sp.expand(sqrtg * L)

    # truncate at O(w^2): O(w^1) must vanish identically (no tadpole)
    Wsym, Wpsym, ssym = sp.symbols("Wc Wpc s", real=True)
    Lflat = sp.expand(L.subs(wp, Wpsym).subs(w(r), Wsym))
    Lscaled = sp.expand(Lflat.subs({Wsym: ssym * Wsym, Wpsym: ssym * Wpsym}))
    assert is_zero(sp.expand(Lscaled.coeff(ssym, 1))), "unexpected O(w) action terms"
    L2 = sp.expand(Lscaled.coeff(ssym, 2)).subs({Wsym: w(r), Wpsym: wp})

    # all explicit x must cancel after stripping the LLL Gaussian factor psi^2
    L2_stripped = sp.simplify(L2 / psi**2)
    assert is_zero(sp.diff(L2_stripped, x)), "residual explicit x after LLL reduction"
    log("[0] O(w) tadpole absent; LLL x-reduction exact (no residual x)")

    Edens = sp.expand(sp.cancel(sp.together(L2_stripped)))  # = -P w'^2 + Q w^2
    P = sp.simplify(-Edens.coeff(wp, 2))
    Q = sp.simplify(Edens.coeff(w(r), 2))
    assert is_zero(sp.simplify(Edens - (-P * wp**2 + Q * w(r) ** 2))), "density form"

    P_cf = sp.sqrt(-gtt * gzz / grr)
    Q_cf = B * sp.sqrt(-gtt * grr * gzz) / gpp
    rho_cf = sp.sqrt(-gtt * grr * gzz) / gpp
    # identify via squared equality (both are positive square roots by construction)
    assert is_zero(sp.simplify(P**2 - P_cf**2)) and eq_pos(P, P_cf), "P mismatch"
    assert is_zero(sp.simplify(Q**2 - Q_cf**2)) and eq_pos(Q, Q_cf), "Q mismatch"
    # numeric sign anchor (positive metric point): P, Q > 0 and match the closed form
    num = {att: 1.7, brr: 0.9, cpp: 1.3, dzz: 1.1, B: 2.0}
    assert float(P.subs(POS).subs(num)) > 0 and float(Q.subs(POS).subs(num)) > 0
    assert abs(float((P - P_cf).subs(POS).subs(num))) < 1e-12
    assert abs(float((Q - Q_cf).subs(POS).subs(num))) < 1e-12
    log(
        "[1] density = -P w'^2 + Q w^2, "
        "P = sqrt(-g_tt g_zz/g_rr), Q = B sqrt(-g_tt g_rr g_zz)/g_pp"
    )

    # Euler-Lagrange => (P w')' + Q w = 0  (identity in P, Q; no sqrt simp needed)
    EL = sp.expand(sp.diff(sp.diff(Edens, wp), r).doit() - sp.diff(Edens, w(r)))
    SL = sp.expand(-2 * (sp.diff(P * wp, r).doit() + Q * w(r)))
    assert is_zero(sp.expand(EL - SL)), "EL != -2[(P w')' + Q w]"
    log(
        "[1] EL: (P w')' + Q w = 0;  norm density rho = Q/B = sqrt(-g_tt g_rr g_zz)/g_pp"
    )

    # ---------------------------------------------------------------------------
    # [1b] where B comes from: transverse Landau (= 0 here) vs magnetic moment
    # ---------------------------------------------------------------------------
    FW_xy = sp.expand(F[0][2][3] + sp.I * F[1][2][3])  # F^W_{xy} = F^1_{xy}+iF^2_{xy}
    assert is_zero(FW_xy), "F^W_{xy} != 0 on the polarised LLL ansatz"
    F3xy = sp.expand(F[2][2][3])
    assert is_zero(sp.expand(F3xy - (B - w(r) ** 2 * psi**2))), "F^3_{xy} form"
    mag_term = sp.expand(gup[2, 2] * gup[3, 3] * F3xy**2)
    mag_w2 = sp.expand(sp.expand(mag_term).coeff(w(r), 2))
    assert is_zero(sp.simplify(mag_w2 - (-2 * B * psi**2 / gpp**2))), "moment term"
    log(
        "[1b] F^W_{xy} = 0 (Landau kinetic drops); moment term -2B psi^2/g_pp^2 "
        "-> single e^{-2V} weight"
    )

    # ---------------------------------------------------------------------------
    # CHECK 1: AdS5-Schwarzschild in u-coordinates (r plays the role of u)
    # ---------------------------------------------------------------------------
    uH = sp.Symbol("u_H", positive=True)
    fp = sp.Symbol("fp", positive=True)  # blackening factor > 0 on domain
    f_uH = 1 - r**4 / uH**4
    schw = {gtt: -fp / r**2, grr: 1 / (r**2 * fp), gpp: 1 / r**2, gzz: 1 / r**2}
    # with fp, r > 0 the square roots resolve cleanly
    P1 = sp.simplify(P_cf.subs(schw))
    Q1 = sp.simplify(Q_cf.subs(schw))
    assert is_zero(P1 - fp / r), f"CHECK1 P: {P1}"
    assert is_zero(Q1 - B / r), f"CHECK1 Q: {Q1}"
    # density
    dens1 = -P1 * wp**2 + Q1 * w(r) ** 2
    assert is_zero(sp.simplify(dens1 - (-(fp / r) * wp**2 + (B / r) * w(r) ** 2)))
    # ODE with the actual f(r): substitute fp -> f_uH in P1, Q1 (linear, no fp')
    P1r, Q1r = f_uH / r, B / r
    ode1 = sp.expand((sp.diff(P1r * wp, r).doit() + Q1r * w(r)) * r)
    probe_ode = (
        f_uH * sp.Derivative(w(r), (r, 2))
        + (sp.diff(f_uH, r) - f_uH / r) * wp
        + B * w(r)
    )
    assert is_zero(sp.expand(ode1 - probe_ode)), "CHECK1 ODE != probe-limit zero mode"
    log(
        "[CHECK1] AdS5-Schw (u-coord): P=f/u, Q=B/u, ODE = probe-limit zero mode -- PASS"
    )

    # ---------------------------------------------------------------------------
    # CHECK 2: first variation vs the bc_shift dP, dQ (perturbed u-metric)
    # ---------------------------------------------------------------------------
    a = sp.Symbol("a", positive=True)
    Htt, Hxx, Hzz, Huu = (sp.Function(n) for n in ("H_tt", "H_xx", "H_zz", "H_uu"))
    pert = {
        gtt: -(fp / r**2) * (1 + a * Htt(r)),
        grr: (1 / (r**2 * fp)) * (1 + a * Huu(r)),
        gpp: (1 / r**2) * (1 + a * Hxx(r)),
        gzz: (1 / r**2) * (1 + a * Hzz(r)),
    }
    Edens_p = -P_cf.subs(pert) * wp**2 + Q_cf.subs(pert) * w(r) ** 2
    E_series = sp.series(sp.powsimp(sp.expand(Edens_p), force=True), a, 0, 2).removeO()
    E1 = sp.expand(E_series.coeff(a, 1))
    assert is_zero(sp.simplify(E1.coeff(w(r) * wp))), "unexpected w w' cross term"
    dP = sp.simplify((-E1.coeff(wp, 2)).subs(fp, f_uH))
    dQ = sp.simplify((E1.coeff(w(r), 2)).subs(fp, f_uH))
    dP_ref = (f_uH / (2 * r)) * (Htt(r) + Hzz(r) - Huu(r))
    dQ_ref = (B / (2 * r)) * (Htt(r) + Huu(r) + Hzz(r) - 2 * Hxx(r))
    assert is_zero(sp.simplify(dP - dP_ref)), f"CHECK2 dP: {dP}"
    assert is_zero(sp.simplify(dQ - dQ_ref)), f"CHECK2 dQ: {dQ}"
    log("[CHECK2] O(alpha) reduction = bc_shift dP, dQ (symbolic zero) -- PASS")

    # numeric cross-check against the generated dP/dQ of systems/bc_shift.py
    try:
        import random

        from backreaction.systems import bc_shift as bcs

        rng = random.Random(20260713)
        lam = sp.lambdify((r, B, uH, Htt(r), Hxx(r), Hzz(r), Huu(r)), (dP, dQ), "math")
        maxerr = 0.0
        for _ in range(12):
            uu = rng.uniform(0.15, 0.95)
            Bv = rng.uniform(1.0, 8.0)
            uHv = rng.uniform(0.8, 1.3)
            ht, hx, hz, hu = (rng.uniform(-1, 1) for _ in range(4))
            mine = lam(uu, Bv, uHv, ht, hx, hz, hu)
            theirs = (
                bcs.dP(uu, Bv, uHv, ht, hx, hz, hu, 0, 0, 0, 0),
                bcs.dQ(uu, Bv, uHv, ht, hx, hz, hu, 0, 0, 0, 0),
            )
            maxerr = max(maxerr, abs(mine[0] - theirs[0]), abs(mine[1] - theirs[1]))
        assert maxerr < 1e-12, f"systems/bc_shift.py mismatch {maxerr}"
        log(
            f"[CHECK2] numeric match to dP/dQ of systems/bc_shift.py (max {maxerr:.1e}) -- PASS"
        )
    except ImportError:
        log(
            "[CHECK2] systems/bc_shift.py not importable (is the project installed? "
            "`uv sync`); "
            "symbolic anchor still asserted"
        )

    # ---------------------------------------------------------------------------
    # Consistent truncation: no O(w) source in the omitted components
    # ---------------------------------------------------------------------------
    Fup = [
        [
            [sp.expand(gup[m, m] * gup[nn, nn] * F[ai][m][nn]) for nn in range(Nd)]
            for m in range(Nd)
        ]
        for ai in range(3)
    ]

    def eom_component(ai, N):
        e = sp.S.Zero
        for m in range(Nd):
            e += sp.diff(sqrtg * Fup[ai][m][N], coords[m])
            for bi in range(3):
                for ci in range(3):
                    lc = sp.LeviCivita(ai + 1, bi + 1, ci + 1)
                    if lc != 0:
                        e += lc * A[bi][m] * sqrtg * Fup[ci][m][N]
        return sp.expand(e)

    def worder(expr, k):
        ex = sp.expand(expr.subs(wp, Wpsym).subs(w(r), Wsym))
        ex = sp.expand(ex.subs({Wsym: ssym * Wsym, Wpsym: ssym * Wpsym}))
        return sp.expand(ex.coeff(ssym, k))

    for ai in (0, 1):  # colours 1,2, omitted comps t, r, z
        for N in (0, 1, 4):
            assert is_zero(eom_component(ai, N)), (
                f"colour {ai + 1} comp {coords[N]} EOM nonzero (truncation)"
            )
    log("[trunc] W_t, W_z, W_r (colours 1,2) equations vanish identically on ansatz")

    for N in range(Nd):  # colour 3 (dA^3): no O(w^0) bg, no O(w) source
        e = eom_component(2, N)
        assert is_zero(worder(e, 0)), f"colour-3 comp {coords[N]} O(w^0) nonzero"
        assert is_zero(worder(e, 1)), f"colour-3 comp {coords[N]} O(w^1) source!"
    log("[trunc] dA^3 (colour 3) carries no O(w) source; induced field is O(w^2)")

    # kept equations (colour 1 x, colour 2 y): the LINEAR part reduces to probe-limit
    # (the full equation carries an O(w^3) self-interaction from F^3_{xy}; extract
    # O(w) with an order tag that also scales w'')
    W0s, W1s, W2s = sp.symbols("W0s W1s W2s", real=True)
    wpp = sp.Derivative(w(r), (r, 2))

    def w_linear(expr):
        ex = expr.subs({wpp: W2s, wp: W1s, w(r): W0s})
        ex = ex.subs({W0s: ssym * W0s, W1s: ssym * W1s, W2s: ssym * W2s})
        lin = sp.expand(ex).coeff(ssym, 1)
        return lin.subs({W0s: w(r), W1s: wp, W2s: wpp})

    # substitute the FULL r-dependent Schwarzschild metric (not the constant-fp
    # proxy) so that metric derivatives inside eom_component evaluate f'(r)
    schw_full = {
        gtt: -f_uH / r**2,
        grr: 1 / (r**2 * f_uH),
        gpp: 1 / r**2,
        gzz: 1 / r**2,
    }
    for ai, N in ((0, 2), (1, 3)):
        e = eom_component(ai, N).subs(schw_full).doit()
        e1 = sp.simplify(w_linear(e) / psi)
        ratio = sp.simplify(sp.cancel(e1 / probe_ode))
        assert (
            not ratio.has(w(r))
            and not ratio.has(wp)
            and not ratio.has(wpp)
            and not is_zero(ratio)
        ), f"kept eq colour {ai + 1} not proportional to the probe-limit ODE"
    log(
        "[trunc] kept equations (colour 1 x, colour 2 y): linear part = the probe-limit ODE"
    )

    for m in range(Nd):  # stress tensor: no O(w) term
        for n in range(m, Nd):
            Tmn = sp.expand(
                sum(
                    Fup[ai][m][p] * gdn[p, p] * F[ai][n][p]
                    for ai in range(3)
                    for p in range(Nd)
                )
            )
            assert is_zero(worder(Tmn, 1)), f"stress T_{m}{n} has O(w) term"
    log("[trunc] stress tensor has no O(w) term (gravity sourced at O(w^2))")

    # ---------------------------------------------------------------------------
    # Specialise to the (U, V, W) brane
    # ---------------------------------------------------------------------------
    U = sp.Function("U")(r)
    Vf = sp.Function("V")(r)
    Wf = sp.Function("W")(r)
    brane = {gtt: -U, grr: 1 / U, gpp: sp.exp(2 * Vf), gzz: sp.exp(2 * Wf)}
    # closed forms on the brane; verify via squared equality (positive roots) with
    # U > 0 outside the horizon
    P_b, Q_b, rho_b = U * sp.exp(Wf), B * sp.exp(Wf - 2 * Vf), sp.exp(Wf - 2 * Vf)
    Upos = sp.Symbol("Upos", positive=True)
    subU = {U: Upos}
    assert is_zero(sp.simplify((P_cf.subs(brane) ** 2 - P_b**2).subs(subU))), "brane P"
    assert is_zero(sp.simplify((Q_cf.subs(brane) ** 2 - Q_b**2).subs(subU))), "brane Q"
    assert is_zero(sp.simplify((rho_cf.subs(brane) ** 2 - rho_b**2).subs(subU))), (
        "brane rho"
    )
    log("[brane] P = U e^{W}, Q = B e^{W-2V}, rho = e^{W-2V}")

    # --- horizon Frobenius at a regular horizon r_p (U(r_p)=0, U'(r_p)!=0) -------
    eps = sp.Symbol("epsilon", positive=True)
    Up, Vp, Wp = sp.symbols("Up Vp Wp", real=True)  # U'(r_p), V(r_p), W(r_p)
    w0h, w1h = sp.symbols("w0h w1h", real=True)
    Pser = (Up * eps) * sp.exp(Wp)  # P = U e^W ~ Up e^{Wp} eps
    Qser = B * sp.exp(Wp - 2 * Vp)  # Q ~ B e^{Wp-2Vp}
    wser = w0h + w1h * eps
    ode_h = sp.expand(sp.diff(Pser * sp.diff(wser, eps), eps) + Qser * wser)
    w1_sol = sp.solve(ode_h.coeff(eps, 0), w1h)[0]
    w1_expect = -B * sp.exp(-2 * Vp) / Up * w0h
    assert is_zero(sp.simplify(w1_sol - w1_expect)), f"horizon slope: {w1_sol}"
    log("[horizon] regularity:  w'(r_p) = -(B e^{-2V(r_p)}/U'(r_p)) w(r_p)")
    # AdS5-Schw (r-coord): U = r^2 - r^{-2} (u_H=1), U'(1)=4, e^{-2V(1)}=1
    schw_r = {U: r**2 - 1 / r**2, Vf: sp.log(r), Wf: sp.log(r)}
    Upv = sp.diff(schw_r[U], r).subs(r, 1)
    eVm = sp.exp(-2 * schw_r[Vf]).subs(r, 1)
    slope_schw = sp.simplify(-B * eVm / Upv)
    assert is_zero(sp.simplify(slope_schw + B / 4)), f"schw horizon slope: {slope_schw}"
    log(f"[horizon] AdS5-Schw check (r-coord): w'(1) = {slope_schw} w(1) = -B/4 w(1)")

    # --- boundary series in r with DK asymptotics -------------------------------
    # DK: U -> r^2 + ..., V, W -> ln r + O(r^{-4} ln r).  Keep generic subleading
    # placeholders at the DK orders and show the r^0, r^{-2} data are untouched.
    Uu4, Uul, v4, vd, w4, wd = sp.symbols("Uu4 Uul v4 vd w4 wd", real=True)
    c0, c2, cl = sp.symbols("c0 c2 cl", real=True)
    Ubdy = r**2 + Uu4 / r**2 + Uul * sp.log(r) / r**2
    e2V = r**2 * (1 + v4 * sp.log(r) / r**4 + vd / r**4)
    e2W = r**2 * (1 + w4 * sp.log(r) / r**4 + wd / r**4)
    Pr = Ubdy * sp.sqrt(e2W)  # U e^{W}   (sqrt(r^2 (...)) = r sqrt(...))
    Qr = B * sp.sqrt(e2W) / e2V  # B e^{W-2V}
    # w = c0 (1 + cl r^{-2} ln r) + c2 r^{-2} + higher; solve the ODE order by order
    e3, e4 = sp.symbols("e3 e4", real=True)
    wexp = c0 * (1 + cl * sp.log(r) / r**2) + c2 / r**2 + e3 / r**3 + e4 / r**4
    lhs = sp.diff(Pr * sp.diff(wexp, r), r) + Qr * wexp
    lhs = sp.expand(lhs.rewrite(sp.exp).doit())
    # leading balance sits at r^{-1}: the constant source B c0 must cancel against
    # the log term's kinetic -2 c0 cl, fixing cl (the source-log coefficient).  It
    # carries NO log itself (the derivative removes it).
    subl = [Uu4, Uul, v4, vd, w4, wd]
    lead = sp.expand(sp.limit(lhs * r, r, sp.oo))  # coefficient of r^{-1}
    assert is_zero(lead.coeff(sp.log(r), 1)), "unexpected log at r^{-1}"
    cl_sol = sp.solve(lead, cl)[0]  # per-c0 coefficient of ln r / r^2
    assert is_zero(sp.simplify(cl_sol - B / 2)), (
        f"boundary source-log: cl = {cl_sol} (expected +B/2)"
    )
    assert all(is_zero(sp.diff(cl_sol, s)) for s in subl), (
        "DK subleading corrections leak into the r^{-2} log coefficient"
    )
    # the ONLY log term in w is c0 cl / r^2 -- at r^{-2}, none at r^0
    logcoeff = sp.expand(wexp).coeff(sp.log(r))
    assert is_zero(logcoeff - c0 * cl / r**2), "log placement (should be r^{-2} only)"
    log(
        "[boundary] r->oo: source c0 at r^0 (log-free), source-log cl=+B/2 c0 at "
        "r^{-2}; DK O(r^{-4}ln r) corrections do not touch r^0, r^{-2}"
    )

    # ---------------------------------------------------------------------------
    # Code generation: systems/dk_zeromode.py
    # ---------------------------------------------------------------------------
    rs, Us, Ups, Vs, Vps, Ws, Wps = sp.symbols("r U Up V Vp W Wp", real=True)
    Bsym = sp.Symbol("B", positive=True)
    p_of = Us * sp.exp(Ws)  # p = U e^W
    p1_of = sp.exp(Ws) * (Ups + Us * Wps)  # p' = d/dr (U e^W)
    q_of = Bsym * sp.exp(Ws - 2 * Vs)  # q = B e^{W-2V}
    rho_of = sp.exp(Ws - 2 * Vs)  # weight
    whor = -Bsym * sp.exp(-2 * Vs) / Ups  # w'(r_p)/w(r_p) at U = 0
    # consistency: p1_of == d/dr (U e^W) with genuinely r-dependent functions,
    # mapped back to the value/derivative symbols afterwards
    Ur_, Wr_ = sp.Function("Ur_")(rs), sp.Function("Wr_")(rs)
    p1_check = sp.diff(Ur_ * sp.exp(Wr_), rs).subs(
        {sp.Derivative(Ur_, rs): Ups, sp.Derivative(Wr_, rs): Wps, Ur_: Us, Wr_: Ws}
    )
    assert is_zero(sp.simplify(p1_check - p1_of)), "generated p1 mismatch"
    assert is_zero(sp.simplify(P_b.subs({U: Us, Vf: Vs, Wf: Ws}) - p_of)), (
        "generated p mismatch"
    )

    from backreaction import paths  # noqa: E402

    gen_path = paths.SYSTEMS / "dk_zeromode.py"
    with open(gen_path, "w") as fh:
        fh.write(
            '"""AUTO-GENERATED by dk_zeromode.py -- do not edit.\n\n'
            "Linearised SU(2) W zero-mode (polarised LLL) radial operator on the\n"
            "general diagonal magnetic brane\n"
            "    ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2+dy^2) + e^{2W} dz^2,\n"
            "background A^3_y = B x.  Sturm-Liouville form\n"
            "    (p w')' + q w = 0   <=>   p w'' + p1 w' + q w = 0,\n"
            "    p = U e^{W},  p1 = e^{W}(U' + U W'),  q = B e^{W-2V},\n"
            "eigenvalue B, weight p, norm density rho = e^{W-2V}\n"
            "    (analogue of J_2 = int rho w^2 dr).\n"
            "Horizon regularity (U(r_p)=0):  w'(r_p) = horizon_slope * w(r_p).\n"
            "Boundary (r->oo): source c0 at r^0, source-free c0 = 0; vev at r^{-2}.\n"
            '"""\n\nimport math  # noqa: F401\n\n\n'
            "def sl_p(r, U, Up, V, Vp, W, Wp, B):\n"
            f"    return {sp.pycode(p_of)}\n\n\n"
            "def sl_p1(r, U, Up, V, Vp, W, Wp, B):\n"
            f"    return {sp.pycode(p1_of)}\n\n\n"
            "def sl_q(r, U, Up, V, Vp, W, Wp, B):\n"
            f"    return {sp.pycode(q_of)}\n\n\n"
            "def sl_rho(r, U, Up, V, Vp, W, Wp, B):\n"
            f"    return {sp.pycode(rho_of)}\n\n\n"
            "def horizon_slope(r, U, Up, V, Vp, W, Wp, B):\n"
            '    """w\'(r_p)/w(r_p) at a regular horizon (U -> 0)."""\n'
            f"    return {sp.pycode(whor)}\n"
        )
    log(f"[gen] wrote {gen_path}")
    log("done -- all checks passed")

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
