"""Exact reduction of the SU(2) Yang-Mills free energy on the GENERAL diagonal
magnetic brane, and the Landau (vortex-lattice) expansion on it.

This generalises the flat/AdS5-Schwarzschild reduction of
`scripts/nonlinear_reduction.py` to the D'Hoker-Kraus magnetic brane, keeping the
metric functions generic so that BOTH the flat u-coordinate limit (CHECK F) and
the (U, V, W) brane specialisation drop out of one derivation.

Background (boundary at r -> oo, regular horizon r_p, U(r_p) = 0):

    ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2 + dy^2) + e^{2W} dz^2,
    A^3_y = B x   (background abelian magnetic field, colour 3).

Static, planar, radial-gauge probe fields (A^a_t = A^a_z = A^a_r = 0):

    W_x = rho w0(r) psi(x,y),  W_y = -i W_x,   psi in the LLL at field B,
    A^3 = (xB + rho^2 a_y) dy + rho^2 a_x dx.

CONVENTION.  Only two metric contractions enter the static planar density; write
them as prefactors of a single radial coordinate r,

    p1 = sqrt(-g) g^{pp} g^{pp} = sqrt(-g) / g_pp^2      (F_xy channel),
    p2 = sqrt(-g) g^{pp} g^{rr} = sqrt(-g) / (g_pp g_rr) (radial channel),
    sqrt(-g) = sqrt(-g_tt g_rr) g_pp sqrt(g_zz).

Flat u-coordinate (g_tt=-f/u^2, g_rr=1/(u^2 f), g_pp=g_zz=1/u^2):  p1 = 1/u,
p2 = f/u -- the 1/(2u), f/(2u) of nonlinear_reduction.py.
(U,V,W) brane (g_tt=-U, g_rr=1/U, g_pp=e^{2V}, g_zz=e^{2W}):  p1 = e^{W-2V},
p2 = U e^{W}  -- exactly the Sturm-Liouville weight P = U e^{W} of paper
sec. 3.1 (derivations/dk_zeromode.py) and norm density
rho_SL = e^{W-2V}, with p1 = rho_SL and p2 = P.

Results / checks (all asserted)
-------------------------------
[1] EXACT reduction, all orders, generic 4-function diagonal metric:
    (1/4) sqrt(-g) F^a_{MN} F^{aMN} = (1/2)[ p1 (|F^W_xy|^2 + (F^3_xy)^2)
                                             + p2 (|W_x'|^2 + |W_y'|^2
                                                   + a_x'^2 + a_y'^2) ],
    with F^W_xy = D_x W_y - D_y W_x, F^3_xy = B + b + Im(Wbar_x W_y),
    b = d_x a_y - d_y a_x.  p1, p2 read off and identified with the closed forms.
[2] F^W_xy = 0 on the polarised LLL (metric independent).
[3] Landau expansion (W -> rho W, a -> rho^2 a):  O(rho^{1,3,5}) vanish;
        O(rho^2) = p2 w0'^2 |psi|^2 - p1 B w0^2 |psi|^2 + p1 B b   (last = total x-deriv),
        O(rho^4) = p1/2 (b - w0^2|psi|^2)^2 + p2/2 (a_x'^2 + a_y'^2),
    corrections start at O(rho^6).
[4] O(rho^2) form = the onset Sturm-Liouville operator (paper sec. 3.1):
    free-energy density / |psi|^2
    = P w0'^2 - Q w0^2 with P = p2, Q = B p1 (= -[action density]).
[5] Per-harmonic (G || x) induced-field ODE on the brane:
        -(p2 bhat')' + G^2 p1 bhat = G^2 p1 w0^2   (per rho^2 lambda_G),
    i.e. -(U e^W bhat')' + G^2 e^{W-2V}(bhat - w0^2) = 0 on the brane -- the
    e^{-2V} sits on BOTH the G^2 kinetic and the source (transverse g^{pp});
    longitudinal a_x is unsourced; flat limit = the probe-limit exchange
    equation (paper app. E.1).
[6] Contact integrand C4 = int p1 w0^4 dr = int e^{W-2V} w0^4 dr on the brane.
[7] On-shell quadratic-functional exchange identity: the on-shell per-harmonic
    quartic energy is  (C4 - X)/2  with X = int p1 bhat w0^2 dr, and the
    bhat-dependent (exchange) part is  -X/2 = -1/2 <source, bhat>  -- exchange is
    minus half the source pairing (an on-shell identity via the EL equation).
[F] Flat limit: p1 -> 1/u, p2 -> f/u reproduce nonlinear_reduction.py and the
    probe-limit densities of paper app. E.1 exactly.

Generates systems/dk_quartic.py (p1, p2, the induced-field ODE coefficients and
the C4/X integrand weights, of (r, U, U', V, V', W, W', B)).

Run:  uv run python -m backreaction.derivations.dk_quartic   (~1 min)
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


# ---------------------------------------------------------------------------
# General diagonal metric; static planar radial-gauge fields of (x, y, r)
# ---------------------------------------------------------------------------
x, y = sp.symbols("x y", real=True)
r = sp.Symbol("r", positive=True)
B, rho = sp.symbols("B rho", positive=True)
I = sp.I

gtt = sp.Function("g_tt")(r)  # < 0
grr = sp.Function("g_rr")(r)
gpp = sp.Function("g_pp")(r)  # g_xx = g_yy
gzz = sp.Function("g_zz")(r)
att, brr, cpp, dzz = sp.symbols("att brr cpp dzz", positive=True)
POS = {gtt: -att, grr: brr, gpp: cpp, gzz: dzz}

sqrtg = sp.sqrt(-gtt * grr * gpp * gpp * gzz)

# complex condensate + abelian fields (generic, for the exact identity)
Wx = sp.Function("W_x")(x, y, r)
Wy = sp.Function("W_y")(x, y, r)
Wxc = sp.Function("Wc_x")(x, y, r)  # formal independent conjugates
Wyc = sp.Function("Wc_y")(x, y, r)
ax = sp.Function("a_x")(x, y, r)
ay = sp.Function("a_y")(x, y, r)


def main():
    # real colour components  A[(colour, coord)], coords 0=x, 1=y, 2=r
    A = {}
    A[(0, 0)] = (Wx + Wxc) / 2
    A[(0, 1)] = (Wy + Wyc) / 2
    A[(1, 0)] = (Wx - Wxc) / (2 * I)
    A[(1, 1)] = (Wy - Wyc) / (2 * I)
    A[(2, 0)] = ax
    A[(2, 1)] = B * x + ay
    for a_ in range(3):
        A[(a_, 2)] = sp.S.Zero  # radial gauge A_r = 0

    coords = [x, y, r]
    LC = {
        (0, 1, 2): 1,
        (1, 2, 0): 1,
        (2, 0, 1): 1,
        (0, 2, 1): -1,
        (2, 1, 0): -1,
        (1, 0, 2): -1,
    }

    def Fld(a_, m, n):
        e = sp.diff(A[(a_, n)], coords[m]) - sp.diff(A[(a_, m)], coords[n])
        for b_ in range(3):
            for c_ in range(3):
                s = LC.get((a_, b_, c_), 0)
                if s:
                    e += s * A[(b_, m)] * A[(c_, n)]
        return e

    # metric prefactors of the (x,y,r) block; raised indices g^{pp}=1/gpp, g^{rr}=1/grr
    p1 = sqrtg / gpp**2  # F_xy channel
    p2 = sqrtg / (gpp * grr)  # radial channel

    # (1/4) sqrt(-g) F^a_{MN} F^{aMN}, only the (xy),(xr),(yr) planes are populated
    L_exact = sp.S.Zero
    for a_ in range(3):
        L_exact += p1 * Fld(a_, 0, 1) ** 2 + p2 * (
            Fld(a_, 0, 2) ** 2 + Fld(a_, 1, 2) ** 2
        )
    L_exact = sp.expand(L_exact / 2)

    # claimed reduced form
    b_ind = sp.diff(ay, x) - sp.diff(ax, y)
    A3x, A3y = ax, B * x + ay
    FWxy = (sp.diff(Wy, x) + I * A3x * Wy) - (sp.diff(Wx, y) + I * A3y * Wx)
    FWxyc = (sp.diff(Wyc, x) - I * A3x * Wyc) - (sp.diff(Wxc, y) - I * A3y * Wxc)
    ImWW = (Wxc * Wy - Wx * Wyc) / (2 * I)
    F3xy = B + b_ind + ImWW
    L_red = (
        p1 * (FWxy * FWxyc + F3xy**2)
        + p2
        * (
            sp.diff(Wx, r) * sp.diff(Wxc, r)
            + sp.diff(Wy, r) * sp.diff(Wyc, r)
            + sp.diff(ax, r) ** 2
            + sp.diff(ay, r) ** 2
        )
    ) / 2

    assert is_zero(L_exact - L_red), "exact reduction identity failed"
    log("[1] exact reduction (1/4)sqrt(-g)F^2 = (1/2)[p1(...)+p2(...)] (all orders)")

    # identify p1, p2 with closed forms and their brane/flat values
    U = sp.Function("U")(r)
    Vf = sp.Function("V")(r)
    Wf = sp.Function("W")(r)
    brane = {gtt: -U, grr: 1 / U, gpp: sp.exp(2 * Vf), gzz: sp.exp(2 * Wf)}
    Upos = sp.Symbol("Upos", positive=True)
    p1_brane = sp.exp(Wf - 2 * Vf)
    p2_brane = U * sp.exp(Wf)
    assert is_zero((p1.subs(brane) ** 2 - p1_brane**2).subs({U: Upos}))
    assert is_zero((p2.subs(brane) ** 2 - p2_brane**2).subs({U: Upos}))
    uH = sp.Symbol("u_H", positive=True)
    u = sp.Symbol("u", positive=True)
    fu = 1 - u**4 / uH**4
    flat = {gtt: -fu / u**2, grr: 1 / (u**2 * fu), gpp: 1 / u**2, gzz: 1 / u**2}
    assert is_zero(sp.simplify(p1.subs(flat) - 1 / u))
    assert is_zero(sp.simplify(p2.subs(flat) - fu / u))
    log(
        "[1] p1 = e^{W-2V} (=1/u flat), p2 = U e^{W} (=f/u flat); "
        "p2 = weight P, p1 = norm density"
    )

    # ---------------------------------------------------------------------------
    # [2] F^W_xy = 0 on the polarised LLL (metric-independent)
    # ---------------------------------------------------------------------------
    w0 = sp.Function("w_0", real=True)(r)
    p_1, p_2, c_1, c_2 = sp.symbols("p_1 p_2 c_1 c_2", real=True)
    psi_p = lambda p: sp.exp(I * p * y - B * (x + p / B) ** 2 / 2)
    psi = c_1 * psi_p(p_1) + c_2 * psi_p(p_2)  # two-mode LLL superposition
    WxL, WyL = w0 * psi, -I * w0 * psi
    FWxy_L = sp.diff(WyL, x) - (sp.diff(WxL, y) + I * B * x * WxL)
    assert sp.simplify(FWxy_L) == 0
    log("[2] F^W_xy = 0 on the polarised LLL (W_y = -i W_x, psi in LLL)")

    # ---------------------------------------------------------------------------
    # [3] Landau expansion in rho (W -> rho W, a -> rho^2 a)
    # ---------------------------------------------------------------------------
    psic = psi.conjugate()
    A2 = dict(A)
    A2[(0, 0)] = rho * (WxL + sp.conjugate(WxL)) / 2
    A2[(0, 1)] = rho * (WyL + sp.conjugate(WyL)) / 2
    A2[(1, 0)] = rho * (WxL - sp.conjugate(WxL)) / (2 * I)
    A2[(1, 1)] = rho * (WyL - sp.conjugate(WyL)) / (2 * I)
    A2[(2, 0)] = rho**2 * ax
    A2[(2, 1)] = B * x + rho**2 * ay
    A_save, A = A, A2

    L_rho = sp.S.Zero
    for a_ in range(3):
        L_rho += p1 * Fld(a_, 0, 1) ** 2 + p2 * (
            Fld(a_, 0, 2) ** 2 + Fld(a_, 1, 2) ** 2
        )
    L_rho = sp.expand(L_rho / 2)
    A = A_save

    realsubs = {sp.conjugate(s): s for s in (c_1, c_2, p_1, p_2)}
    L_rho = L_rho.subs(realsubs)
    poly = sp.Poly(L_rho, rho)
    orders = {n: sp.expand(poly.coeff_monomial(rho**n)) for n in range(7)}
    abspsi2 = sp.expand(psi * psic)
    w0p = sp.diff(w0, r)

    # O(rho^{1,3,5}) vanish; O(rho^6) present
    for k in (1, 3, 5):
        assert is_zero(orders[k]), f"O(rho^{k}) nonzero"
    assert not is_zero(orders[6])
    # O(rho^2)
    quad_expect = p2 * w0p**2 * abspsi2 - p1 * B * w0**2 * abspsi2 + p1 * B * b_ind
    assert is_zero(orders[2] - sp.expand(quad_expect)), "O(rho^2) mismatch"
    # O(rho^4)
    quart_expect = p1 / 2 * (b_ind - w0**2 * abspsi2) ** 2 + p2 / 2 * (
        sp.diff(ax, r) ** 2 + sp.diff(ay, r) ** 2
    )
    assert is_zero(sp.expand(orders[4] - sp.expand(quart_expect))), "O(rho^4) mismatch"
    log(
        "[3] Landau expansion: O(rho^{1,3,5})=0; O(rho^2), O(rho^4) match; "
        "corrections at O(rho^6)"
    )

    # ---------------------------------------------------------------------------
    # [4] O(rho^2) = SL operator (free-energy density = -[action density])
    # ---------------------------------------------------------------------------
    # strip |psi|^2 and the total-derivative p1 B b piece
    dens2 = sp.expand((orders[2] - p1 * B * b_ind) / abspsi2)
    P_wp1 = sp.sqrt(-gtt * gzz / grr)  # weight
    Q_wp1 = B * sp.sqrt(-gtt * grr * gzz) / gpp
    # free-energy density = P w0'^2 - Q w0^2 (action = -this)
    assert is_zero((dens2 - (P_wp1 * w0p**2 - Q_wp1 * w0**2)).subs(POS)), "SL mismatch"
    log(
        "[4] O(rho^2)/|psi|^2 = P w0'^2 - Q w0^2, P = p2 = weight, "
        "Q = B p1 (= -action density)"
    )

    # ---------------------------------------------------------------------------
    # [5] per-harmonic induced-field ODE on the brane (G || x)
    # ---------------------------------------------------------------------------
    # Per harmonic G || x the density (after cell average) is a function of the
    # transverse field bhat_G = b_G and the longitudinal a_x.  With a_y = -i bhat/G,
    # a_x = G ell:  b = d_x a_y = i G a_y = bhat.  Build the reduced functional.
    G = sp.Symbol("G", positive=True)
    bhat = sp.Function("bhat", real=True)(r)
    ell = sp.Function("ell", real=True)(r)
    lamG = sp.Symbol("lambda_G", real=True)
    # transverse + longitudinal harmonic amplitudes (as in the probe-limit stress derivation):
    #   a_y = -i bhat/G  =>  |a_y'|^2 = bhat'^2/G^2;   a_x = G ell  =>  |a_x'|^2 = G^2 ell'^2
    # induced-field quartic density per harmonic (b -> bhat, |psi|^2 harmonic -> lamG)
    q_dens = p1 / 2 * (bhat - w0**2 * lamG) ** 2 + p2 / 2 * (
        G**2 * sp.diff(ell, r) ** 2 + sp.diff(bhat, r) ** 2 / G**2
    )
    q_dens = sp.expand(q_dens)
    # EL for bhat and for ell
    EL_b = sp.expand(
        sp.diff(q_dens, bhat)
        - sp.diff(sp.diff(q_dens, sp.Derivative(bhat, r)), r).doit()
    )
    EL_ell = sp.expand(
        sp.diff(q_dens, ell) - sp.diff(sp.diff(q_dens, sp.Derivative(ell, r)), r).doit()
    )
    # longitudinal ell is unsourced: EL_ell is homogeneous (no w0, no lamG source)
    assert not EL_ell.has(w0) and not EL_ell.has(lamG), "longitudinal a_x sourced!"
    log("[5] longitudinal a_x (ell) is unsourced")
    # transverse EL, cleared: -(p2 bhat')' + G^2 p1 (bhat - w0^2 lamG) = 0
    ode_claim = -sp.diff(p2 * sp.diff(bhat, r), r).doit() + G**2 * p1 * (
        bhat - w0**2 * lamG
    )
    ratio = sp.simplify(sp.cancel(sp.expand(EL_b) / sp.expand(-ode_claim / G**2)))
    assert not ratio.has(bhat) and not ratio.has(w0) and ratio != 0, "induced ODE form"
    log(
        f"[5] induced-field ODE: -(p2 bhat')' + G^2 p1 (bhat - w0^2 lamG) = 0 "
        f"[EL/(-ode/G^2) = {ratio}]"
    )
    # flat limit = the probe-limit exchange equation (paper app. E.1)
    # substitute r->u carefully by rebuilding in u
    bhu = sp.Function("bhat", real=True)(u)
    w0u = sp.Function("w_0", real=True)(u)
    p1u, p2u = 1 / u, fu / u
    ode_u = -sp.diff(p2u * sp.diff(bhu, u), u).doit() + G**2 * p1u * (
        bhu - w0u**2 * lamG
    )
    target_u = (
        -sp.diff(fu / u * sp.diff(bhu, u), u).doit()
        + (G**2 / u) * bhu
        - (G**2 / u) * w0u**2 * lamG
    )
    assert is_zero(ode_u - target_u), "flat induced-field ODE mismatch"
    log(
        "[5] flat limit = -(f bhat'/u)' + (G^2/u) bhat = (G^2/u) w0^2 lambda_G "
        "(paper app. E.1)"
    )

    # ---------------------------------------------------------------------------
    # [6] contact integrand = e^{W-2V} w0^4 on the brane
    # ---------------------------------------------------------------------------
    # contact = the O(rho^4) density with the induced field switched off:
    # p1/2 (w0^2|psi|^2)^2, so per lattice the C4 weight integrand is p1 w0^4,
    # and p1 = e^{W-2V} on the brane ([1])
    contact = orders[4].subs({ax: 0, ay: 0}).doit()
    assert is_zero(sp.expand(contact - p1 / 2 * w0**4 * abspsi2**2)), "C4 integrand"
    log("[6] contact integrand C4 = int p1 w0^4 dr = int e^{W-2V} w0^4 dr (brane)")

    # ---------------------------------------------------------------------------
    # [7] on-shell exchange identity: exchange = -1/2 source pairing
    # ---------------------------------------------------------------------------
    # On-shell quartic energy per harmonic (lamG = 1):
    #   E4 = int [ p1/2 (bhat-w0^2)^2 + p2/2 bhat'^2/G^2 ] dr
    # Using the EL (-(p2 bhat')' + G^2 p1 (bhat-w0^2) = 0) and IBP:
    #   int p2/2 bhat'^2/G^2 = -1/2 int bhat (p2 bhat')'/G^2 + bdy
    #                        = -1/2 int bhat p1 (bhat-w0^2) + bdy.
    # => E4 = int p1/2 (bhat-w0^2)^2 - 1/2 int p1 bhat (bhat-w0^2) + bdy
    #       = 1/2 int p1 w0^2 (w0^2 - bhat) + bdy = (C4 - X)/2 + bdy,
    # with C4 = int p1 w0^4, X = int p1 bhat w0^2.  The bhat-dependent part is
    # -X/2 = -1/2 <p1 w0^2, bhat> = -1/2 <source, bhat>.  Verify the integrand
    # identity (pointwise, on the EL) that makes IBP exact:
    # EL as a rule for (p2 bhat')':
    pbp = sp.diff(p2 * sp.diff(bhat, r), r).doit()  # (p2 bhat')'
    el_rule = {pbp: G**2 * p1 * (bhat - w0**2)}
    # kinetic density 1/2 p2 bhat'^2/G^2 IBP: d/dr(1/2 bhat p2 bhat'/G^2) - 1/2 bhat (p2 bhat')'/G^2
    kin = p2 * sp.diff(bhat, r) ** 2 / (2 * G**2)
    ibp = sp.diff(bhat * p2 * sp.diff(bhat, r) / (2 * G**2), r).doit() - bhat * pbp / (
        2 * G**2
    )
    assert is_zero(kin - ibp), "kinetic IBP identity"
    # on the EL: -1/2 bhat (p2 bhat')'/G^2 = -1/2 bhat p1 (bhat - w0^2)
    onshell_kin_bulk = sp.expand((-bhat * pbp / (2 * G**2)).subs(el_rule))
    assert is_zero(onshell_kin_bulk - (-sp.Rational(1, 2) * p1 * bhat * (bhat - w0**2)))
    log(
        "[7] on-shell: E4 = (C4 - X)/2, X = int p1 bhat w0^2 = <src/G^2, bhat>; "
        "exchange part = -X/2 = -1/2 source pairing (IBP exact on EL)"
    )

    # ---------------------------------------------------------------------------
    # [F] flat limit reproduces nonlinear_reduction.py densities
    # ---------------------------------------------------------------------------
    o2_flat = sp.simplify(
        orders[2].subs(
            flat_sub := {
                gtt: -fu / u**2,
                grr: 1 / (u**2 * fu),
                gpp: 1 / u**2,
                gzz: 1 / u**2,
            }
        )
    )
    o2_target = (fu / u) * w0p**2 * abspsi2 - (B / u) * w0**2 * abspsi2 + B * b_ind / u
    assert is_zero(o2_flat - sp.expand(o2_target)), "flat O(rho^2) mismatch"
    o4_flat = sp.simplify(orders[4].subs(flat_sub))
    o4_target = (b_ind - w0**2 * abspsi2) ** 2 / (2 * u) + fu / (2 * u) * (
        sp.diff(ax, r) ** 2 + sp.diff(ay, r) ** 2
    )
    assert is_zero(sp.expand(o4_flat - sp.expand(o4_target))), "flat O(rho^4) mismatch"
    log("[F] flat limit reproduces nonlinear_reduction.py O(rho^2), O(rho^4)")

    # ---------------------------------------------------------------------------
    # code generation: systems/dk_quartic.py
    # ---------------------------------------------------------------------------
    rs, Us, Ups, Vs, Vps, Ws, Wps = sp.symbols("r U Up V Vp W Wp", real=True)
    p1_of = sp.exp(Ws - 2 * Vs)
    p2_of = Us * sp.exp(Ws)
    p2p_of = sp.exp(Ws) * (Ups + Us * Wps)  # (U e^W)'
    from backreaction import paths  # noqa: E402

    gen_path = paths.SYSTEMS / "dk_quartic.py"
    with open(gen_path, "w") as fh:
        fh.write(
            '"""AUTO-GENERATED by dk_quartic.py -- do not edit.\n\n'
            "Per-harmonic induced-field ODE and quartic-kernel weights on the\n"
            "general diagonal magnetic brane\n"
            "    ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2+dy^2) + e^{2W} dz^2.\n"
            "Induced field (per rho^2 lambda_G, G || x):\n"
            "    -(p2 bhat')' + G^2 p1 (bhat - w0^2) = 0,\n"
            "    p1 = e^{W-2V},  p2 = U e^{W},  p2' = e^{W}(U' + U W').\n"
            "Kernel weights (measure dr): contact C4 = int p1 w0^4,\n"
            "exchange X = int p1 bhat w0^2.\n"
            '"""\n\nimport math  # noqa: F401\n\n\n'
            "def p1(r, U, Up, V, Vp, W, Wp, B):\n"
            f"    return {sp.pycode(p1_of)}\n\n\n"
            "def p2(r, U, Up, V, Vp, W, Wp, B):\n"
            f"    return {sp.pycode(p2_of)}\n\n\n"
            "def p2p(r, U, Up, V, Vp, W, Wp, B):\n"
            f"    return {sp.pycode(p2p_of)}\n"
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
