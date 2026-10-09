"""The strict G = 0 (uniform) response of the D'Hoker-Kraus magnetic brane to
the condensate, at fixed temperature and fixed applied field.

The quartic Landau coefficient of a uniform lowest-Landau-level lattice is
F_4 = (1/2) K_{G=0} + C(tau), where C is the shell sum over G != 0 of the
kernel K(s; beta) and K_{G=0} is the self-interaction of the lattice-averaged
condensate density.  At G = 0 the abelian response vanishes identically: the
uniform part of F_xy is r-independent by the Bianchi identity and equals the
applied field at the boundary.  What remains is the uniform metric response,

    ds^2 -> ds^2 + h,   h = diag(-U H_tt, e^{2V} H_xx, e^{2V} H_xx,
                                  e^{2W} H_zz, H_rr / U)        (functions of r),

sourced by the lattice average of the condensate stress tensor, and

    K_{G=0}(beta) = C4 - (1/2) Pi_0 ,
    Pi_0 = int sqrt(-g) S^{MN}[w0] h_MN dr ,

with S the same Einstein-form condensate stress that sources the G != 0
response (systems/dk_stress.py at bhat = 0; it is G-independent), so that
K_{G=0} is in the same units and normalisation as K(s; beta).

Equations (Einstein form, exactly as for G != 0):
    delta(G_MN - 6 g_MN)[h] - alpha delta T^Maxwell_MN[h; Fbar] = alpha S_MN .
Here they are obtained by differentiating the FULL nonlinear Einstein-form
residual on g + eps h with respect to eps -- a route independent of the
Palatini machinery used for G != 0 in derivations/dk_response.py.

Checks (all asserted)
---------------------
[0] Background reduced ODEs (U'', V'', W'') and the (rr) constraint; horizon
    relations U'V' = 4 - (2/3) alpha B^2 e^{-4V}, U'W' = 4 + (1/3) alpha B^2 e^{-4V}.
[1] Sector closure: every off-diagonal row vanishes on the uniform ansatz, and
    the (yy) row equals the (xx) row (planar isotropy, H_yy = H_xx).
[2] Covariance: the homogeneous rows annihilate pure gauge h = L_xi g for
    xi = xi^r(r) d_r (the background potential A = Bx dy is inert under it).
[3] Source conservation: nabla_M S^M_r = 0 on the zero-mode shell
    (P w0')' + Q w0 = 0, P = U e^W, Q = B e^{W-2V}; the other components are
    identically zero.
[4] Gauge H_tt = 0 is reached by xi^r = -U H_tt / U', which vanishes at the
    horizon, so it is a regular gauge and keeps the horizon at r = 1.
[5] The (rr) row is algebraic in H_rr with coefficient
    (alpha B^2 e^{-4V} - 12)/(2U), non-vanishing while beta e^{-4V} < 12.
[6] Closed system: {rr, d/dr rr, xx, zz} determine (H_rr, H_rr', H_xx'',
    H_zz'') from (H_xx, H_xx', H_zz, H_zz'); the (tt) row then vanishes
    identically on the solved forms (the linearised Bianchi identity plus
    source conservation).
[7] Hellmann-Feynman identity, pointwise:
        sqrt(-g) S^{MN} h_MN = -2 (dP w0'^2 - dQ w0^2),
        dP = (P/2)(H_tt + H_zz - H_rr),  dQ = (Q/2)(H_tt + H_zz + H_rr - 2 H_xx),
    the first variation of the onset Sturm-Liouville form at fixed B.  Hence
    -(1/2) Pi_0 = int (dP w0'^2 - dQ w0^2) dr is the shift of the onset
    eigenvalue (in the norm J = int p1 w0^2 dr = 1) caused by the condensate's
    own uniform metric response: the amplitude-equation reading of the
    weight 1/2, which needs no on-shell action and no boundary terms.
[8] Horizon thermodynamics (surface gravity and area):
        delta T / T = (H_tt - H_rr)(r_h) / 2,  delta s / s = (2 H_xx + H_zz)(r_h) / 2.
    Fixed temperature in the gauge H_tt = 0 is H_rr(r_h) = 0.
[9] Horizon regularity: U x (xx), U x (zz) at U = 0 are first-order relations
    for the horizon data; the (rr) row gives H_rr(r_h) algebraically.  The two
    relations are proportional (the trace channel 2 H_xx + H_zz is regular by
    itself, as at beta = 0), which leaves exactly the one-parameter thermal
    family that fixed T removes; the rank is checked numerically in
    numerics/dk_uniform.py, where the relations are built as lim U x (row).
[10] beta -> 0: on AdS5-Schwarzschild (U = r^2 - 1/r^2, e^V = e^W = r) the
    rows reduce to the AdS5-Schwarzschild G = 0 system of derivations/g0.py under
    r = 1/u (same H normalisation); checked by the gauge-invariant channels:
    I_d = H_xx - H_zz obeys (f I_d'/u^3)' = S_d/u^3 with the derivations/g0.py per-rho^2
    source S_d = -2u^2 [f w0'^2 - 2B w0^2] (times alpha).

[11] x-diffeomorphism invariance of the G != 0 kernel pairing: under
    xi = xi^x(r) e^{iGx} d_x (any complex profile) delta a_y = B xi^x and
    delta h_xx = 2iG e^{2V} xi^x, and Re[delta X_c + (1/2) delta Pi_bare] = 0
    pointwise, because sqrt(-g) S^{xx} e^{2V} = -B p1 w0^2 (the x-x stress is
    the magnetic pressure of the current coupling to a_y).  Hence the s -> 0
    limit of K(s), computed in the gauge H_tt = H_xx = 0 where it needs
    xi^x ~ 1/G, equals K_{G=0}.

Generates systems/dk_uniform.py.

Run:  uv run python -m backreaction.derivations.dk_uniform   (~1 min)
"""

import sys
import time

import sympy as sp

sys.setrecursionlimit(100000)
T0_START = time.time()


def log(msg):
    print(f"[{time.time() - T0_START:7.1f}s] {msg}", flush=True)


t, x, y, z, r = coords = sp.symbols("t x y z r", real=True)
B = sp.Symbol("B", positive=True)
alpha = sp.Symbol("alpha", positive=True)
N = 5
idx = ("t", "x", "y", "z", "r")

U = sp.Function("U")(r)
Vf = sp.Function("V")(r)
Wf = sp.Function("W")(r)
gdn = sp.diag(-U, sp.exp(2 * Vf), sp.exp(2 * Vf), sp.exp(2 * Wf), 1 / U)
gup = sp.diag(-1 / U, sp.exp(-2 * Vf), sp.exp(-2 * Vf), sp.exp(-2 * Wf), U)
sqrtg = sp.exp(2 * Vf + Wf)


def main():
    Up, Vp, Wp = (sp.Derivative(F_, r) for F_ in (U, Vf, Wf))
    Upp, Vpp, Wpp = (sp.Derivative(F_, (r, 2)) for F_ in (U, Vf, Wf))

    # ring map: exp(cV V + cW W) -> EV^cV EW^cW so that cancel works in a polynomial
    # ring (the recorded performance trap of the brane derivations)
    EV, EW = sp.symbols("EV EW", positive=True)

    def to_ring(e):
        e = sp.sympify(e)
        for ex in list(e.atoms(sp.exp)):
            arg = sp.expand(ex.args[0])
            cV, cW = arg.coeff(Vf), arg.coeff(Wf)
            assert sp.expand(arg - cV * Vf - cW * Wf) == 0, f"exp arg not in V,W: {arg}"
            e = e.subs(ex, EV**cV * EW**cW)
        return e

    def is_zero(e):
        e = to_ring(sp.expand(sp.sympify(e).doit()))
        if e == 0:
            return True
        return sp.cancel(e) == 0

    def christoffel(gd, gu):
        """Christoffel symbols of a DIAGONAL metric (gu = inverse, diagonal)."""
        Gam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for a in range(N):
            for m in range(N):
                for n in range(m, N):
                    e = (
                        gu[a, a]
                        * (
                            sp.diff(gd[a, m], coords[n])
                            + sp.diff(gd[a, n], coords[m])
                            - sp.diff(gd[m, n], coords[a])
                        )
                        / 2
                    )
                    Gam[a][m][n] = Gam[a][n][m] = e
        return Gam

    def ricci(Gam):
        R = [[sp.S.Zero] * N for _ in range(N)]
        for m in range(N):
            for n in range(m, N):
                e = sp.S.Zero
                for a in range(N):
                    e += sp.diff(Gam[a][m][n], coords[a]) - sp.diff(
                        Gam[a][a][m], coords[n]
                    )
                    for l in range(N):
                        e += Gam[a][a][l] * Gam[l][m][n] - Gam[a][n][l] * Gam[l][a][m]
                R[m][n] = R[n][m] = e
        return R

    Abar = [sp.S.Zero, sp.S.Zero, B * x, sp.S.Zero, sp.S.Zero]  # A^3_y = B x
    Fb = [
        [sp.diff(Abar[n], coords[m]) - sp.diff(Abar[m], coords[n]) for n in range(N)]
        for m in range(N)
    ]

    def maxwell(gd, gu):
        """Einstein-form abelian stress F_MP F_N^P - (1/4) g_MN F^2 (diagonal gu)."""
        F2 = sum(
            Fb[m][n] * gu[m, m] * gu[n, n] * Fb[m][n]
            for m in range(N)
            for n in range(N)
        )
        T = [
            [
                sum(Fb[m][p] * gu[p, p] * Fb[n][p] for p in range(N))
                - gd[m, n] * F2 / 4
                for n in range(N)
            ]
            for m in range(N)
        ]
        return T

    # ---------------------------------------------------------------------------
    # [0] background
    # ---------------------------------------------------------------------------
    GAM = christoffel(gdn, gup)
    Rbg = [[sp.expand(e) for e in row] for row in ricci(GAM)]
    Tbg = maxwell(gdn, gup)
    Tbg_tr = sp.expand(sum(gup[a, a] * Tbg[a][a] for a in range(N)))
    EOM_bg = [
        [
            sp.expand(
                Rbg[m][n]
                + 4 * gdn[m, n]
                - alpha * (Tbg[m][n] - sp.Rational(1, 3) * gdn[m, n] * Tbg_tr)
            )
            for n in range(N)
        ]
        for m in range(N)
    ]
    sol_bg = sp.solve(
        [EOM_bg[0][0], EOM_bg[1][1], EOM_bg[3][3]], [Upp, Vpp, Wpp], dict=True
    )[0]
    bg_rule = {k: sol_bg[k] for k in (Upp, Vpp, Wpp)}
    constraint_bg = sp.expand(EOM_bg[4][4].subs(bg_rule))
    assert is_zero(EOM_bg[2][2] - EOM_bg[1][1])

    def on_shell_bg(e):
        e = sp.expand(sp.sympify(e).doit())
        while e.has(Upp) or e.has(Vpp) or e.has(Wpp):
            e = sp.expand(e.subs(bg_rule).doit())
        return e

    Uv, Vv, Wv, Upv, Vpv, Wpv = sp.symbols("Uv Vv Wv Upv Vpv Wpv", real=True)
    _hrep = {Up: Upv, Vp: Vpv, Wp: Wpv, U: Uv, Vf: Vv, Wf: Wv}
    # the raw (un-reduced) rows at U = 0, second derivatives as opaque symbols
    exx_h = sp.expand(
        EOM_bg[1][1]
        .subs({Upp: sp.Symbol("a1"), Vpp: sp.Symbol("a2"), Wpp: sp.Symbol("a3")})
        .subs(_hrep)
        .subs({Uv: 0})
    )
    ezz_h = sp.expand(
        EOM_bg[3][3]
        .subs({Upp: sp.Symbol("a1"), Vpp: sp.Symbol("a2"), Wpp: sp.Symbol("a3")})
        .subs(_hrep)
        .subs({Uv: 0})
    )
    assert is_zero(
        sp.solve(exx_h, Upv * Vpv)[0]
        - (4 - sp.Rational(2, 3) * alpha * B**2 * sp.exp(-4 * Vv))
    )
    assert is_zero(
        sp.solve(ezz_h, Upv * Wpv)[0]
        - (4 + sp.Rational(1, 3) * alpha * B**2 * sp.exp(-4 * Vv))
    )
    log("[0] background U'', V'', W'' rules, (rr) constraint and horizon relations")

    # ---------------------------------------------------------------------------
    # [1] linearised rows by eps-differentiation of the full nonlinear residual
    # ---------------------------------------------------------------------------
    Htt, Hxx, Hzz, Hrr = (sp.Function(n)(r) for n in ("H_tt", "H_xx", "H_zz", "H_rr"))
    W0 = sp.Function("w0")(r)
    W0p = sp.Derivative(W0, r)
    eps = sp.Symbol("epsilon")

    def uniform_h(Ht, Hx, Hz, Hr):
        h = [[sp.S.Zero] * N for _ in range(N)]
        h[0][0] = -U * Ht
        h[1][1] = sp.exp(2 * Vf) * Hx
        h[2][2] = sp.exp(2 * Vf) * Hx
        h[3][3] = sp.exp(2 * Wf) * Hz
        h[4][4] = Hr / U
        return h

    h = uniform_h(Htt, Hxx, Hzz, Hrr)

    # condensate source (Einstein form, bhat = 0; identical to dk_response Scond)
    S = [[sp.S.Zero] * N for _ in range(N)]
    S[0][0] = U * sp.exp(-4 * Vf) * (U * sp.exp(2 * Vf) * W0p**2 - B * W0**2)
    S[1][1] = -B * W0**2 * sp.exp(-2 * Vf)
    S[2][2] = -B * W0**2 * sp.exp(-2 * Vf)
    S[3][3] = sp.exp(-4 * Vf + 2 * Wf) * (B * W0**2 - U * sp.exp(2 * Vf) * W0p**2)
    S[4][4] = sp.exp(-4 * Vf) / U * (B * W0**2 + U * sp.exp(2 * Vf) * W0p**2)

    # cross-check against the generated per-harmonic stress at bhat = 0 (G-independent)
    from backreaction.systems import dk_stress as dks  # noqa: E402
    import random  # noqa: E402

    _rng = random.Random(20261001)
    _syms = (r, U, Up, Vf, Vp, Wf, Wp, B, sp.Symbol("G"), W0, W0p)
    _maxerr = 0.0
    for _k, _m in (("tt", 0), ("xx", 1), ("zz", 3), ("rr", 4)):
        _lam = sp.lambdify(_syms, S[_m][_m], "math")
        for _ in range(6):
            vv = [_rng.uniform(0.3, 2.0) for _ in range(11)]
            _gen = getattr(dks, f"S_{_k}")(*vv, 0, 0)
            _maxerr = max(_maxerr, abs(_lam(*vv) - _gen))
    assert _maxerr < 1e-10, f"S vs systems/dk_stress at bhat = 0: {_maxerr}"
    log(f"[1] condensate stress = systems/dk_stress at bhat = 0 (max {_maxerr:.1e})")

    def linearised_rows(hmat, with_source=True):
        gp = sp.Matrix(
            [[gdn[m, n] + eps * hmat[m][n] for n in range(N)] for m in range(N)]
        )
        gpi = sp.diag(*[1 / gp[i, i] for i in range(N)])
        Gp = christoffel(gp, gpi)
        Rp = ricci(Gp)
        Tp = maxwell(gp, gpi)
        Rs = sum(gpi[a, a] * Rp[a][a] for a in range(N))
        rows = {}
        for m in range(N):
            for n in range(m, N):
                full = (
                    Rp[m][n]
                    - sp.Rational(1, 2) * gp[m, n] * Rs
                    - 6 * gp[m, n]
                    - alpha * Tp[m][n]
                )
                if with_source:
                    full -= alpha * eps * S[m][n]
                rows[(m, n)] = on_shell_bg(sp.diff(full, eps).subs(eps, 0))
        return rows

    EE = linearised_rows(h)
    diag_pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    for (m, n), e in EE.items():
        if (m, n) not in diag_pairs:
            assert is_zero(e), f"off-diagonal row {idx[m]}{idx[n]} nonzero"
    assert is_zero(EE[(2, 2)] - EE[(1, 1)])
    log("[1] off-diagonal rows vanish; (yy) = (xx) with H_yy = H_xx")

    # ---------------------------------------------------------------------------
    # [2] covariance: pure gauge xi = xi^r(r) d_r is annihilated
    # ---------------------------------------------------------------------------
    xir = sp.Function("xi_r")(r)  # contravariant component xi^r
    hg = [[sp.S.Zero] * N for _ in range(N)]
    for m in range(N):  # (L_xi g)_MN = xi^r d_r g_MN for diagonal g, xi = xi^r d_r ...
        hg[m][m] = xir * sp.diff(gdn[m, m], r)
    hg[4][4] += 2 * gdn[4, 4] * sp.diff(xir, r)  # ... plus 2 g_rr d_r xi^r
    EEg = linearised_rows(hg, with_source=False)
    # the reduced rows also need the first-order (rr) background constraint; use it to
    # eliminate U' (it is linear in U' after the U'' rule)
    Up_sol = sp.solve(constraint_bg, Up)
    assert len(Up_sol) == 1
    for m, n in diag_pairs:
        e = on_shell_bg(EEg[(m, n)])
        e = sp.expand(e.subs(Up, Up_sol[0]))
        assert is_zero(e), f"pure gauge not annihilated in row {idx[m]}{idx[n]}"
    log("[2] homogeneous rows annihilate h = L_xi g, xi = xi^r(r) d_r")

    # ---------------------------------------------------------------------------
    # [3] source conservation nabla_M S^M_N = 0 on the zero-mode shell
    # ---------------------------------------------------------------------------
    P_sl = U * sp.exp(Wf)
    Q_sl = B * sp.exp(Wf - 2 * Vf)
    w0pp_rule = {
        sp.Derivative(W0, (r, 2)): sp.expand(
            sp.cancel(-(sp.diff(P_sl, r) * W0p + Q_sl * W0) / P_sl)
        )
    }

    def on_shell(e):
        e = on_shell_bg(e)
        while e.has(sp.Derivative(W0, (r, 2))):
            e = on_shell_bg(sp.expand(e.subs(w0pp_rule).doit()))
        return e

    Smix = [[gup[m, m] * S[m][n] for n in range(N)] for m in range(N)]  # S^M_N
    for nn in range(N):
        div = sum(sp.diff(Smix[m][nn], coords[m]) for m in range(N))
        for m in range(N):
            for l in range(N):
                div += GAM[m][m][l] * Smix[l][nn] - GAM[l][m][nn] * Smix[m][l]
        assert is_zero(on_shell(div)), f"nabla_M S^M_{idx[nn]} != 0"
    log("[3] nabla_M S^M_N = 0 on the zero-mode shell (all N)")

    # ---------------------------------------------------------------------------
    # [4], [5] gauge H_tt = 0 and the algebraic (rr) row
    # ---------------------------------------------------------------------------
    dHtt = sp.expand(-hg[0][0] / U)  # H_tt shift under xi^r: -(-U' xi^r)/U
    assert is_zero(dHtt - Up * xir / U)
    log("[4] delta H_tt = (U'/U) xi^r: H_tt = 0 via xi^r = -U H_tt/U' (zero at r_h)")

    hx, hx1, hx2, hz, hz1, hz2, hr, hr1 = sp.symbols("hx hx1 hx2 hz hz1 hz2 hr hr1")
    FLD = {
        sp.Derivative(Hxx, (r, 2)): hx2,
        sp.Derivative(Hzz, (r, 2)): hz2,
        sp.Derivative(Hxx, r): hx1,
        sp.Derivative(Hzz, r): hz1,
        sp.Derivative(Hrr, r): hr1,
        Hxx: hx,
        Hzz: hz,
        Hrr: hr,
    }
    gauge0 = {
        sp.Derivative(Htt, (r, 2)): 0,
        sp.Derivative(Htt, r): 0,
        Htt: 0,
    }

    def to_slots(e):
        e = sp.expand(sp.sympify(e).subs(gauge0))
        for a in (
            sp.Derivative(Hxx, (r, 2)),
            sp.Derivative(Hzz, (r, 2)),
            sp.Derivative(Hxx, r),
            sp.Derivative(Hzz, r),
            sp.Derivative(Hrr, r),
            Hxx,
            Hzz,
            Hrr,
        ):
            e = e.subs(a, FLD[a])
        return sp.expand(e)

    row_rr = to_slots(EE[(4, 4)])
    assert not row_rr.has(hr1) and not row_rr.has(hx2) and not row_rr.has(hz2)
    c_rr = row_rr.coeff(hr)
    assert is_zero(c_rr - (alpha * B**2 * sp.exp(-4 * Vf) - 12) / (2 * U))
    log("[5] (rr) row algebraic in H_rr, coefficient (alpha B^2 e^{-4V} - 12)/(2U)")

    # d/dr of the (rr) row on shell (background + zero mode), in slots
    row_rr_d = to_slots(on_shell(sp.diff(EE[(4, 4)].subs(gauge0), r)))
    row_xx = to_slots(EE[(1, 1)])
    row_zz = to_slots(EE[(3, 3)])
    row_tt = to_slots(EE[(0, 0)])
    log("[5] rows in slot form")

    # background/matter values as plain symbols (derivatives first, then functions)
    Us, Ups, Vs, Vps, Ws, Wps, ws, wps = sp.symbols("U Up V Vp W Wp W0 W0p")
    PLAIN = [
        (Up, Ups),
        (Vp, Vps),
        (Wp, Wps),
        (W0p, wps),
        (W0, ws),
        (U, Us),
        (Vf, Vs),
        (Wf, Ws),
    ]

    def plain(e):
        e = sp.sympify(e)
        for a, b in PLAIN:
            e = e.subs(a, b)
        return e

    # ---------------------------------------------------------------------------
    # [6] the closed system and the redundancy of (tt)
    # ---------------------------------------------------------------------------
    UNK = [hr, hr1, hx2, hz2]
    RHS_FLD = [hx, hx1, hz, hz1]
    EQS = [row_rr, row_rr_d, row_xx, row_zz]

    def cof(e, s_):
        return sp.expand(e).coeff(s_)

    def srcpart(e):
        return sp.expand(sp.expand(e).subs({s_: 0 for s_ in UNK + RHS_FLD}))

    for e in EQS:  # every row is linear in the slots (no products of slots)
        for s_ in UNK + RHS_FLD:
            assert not cof(e, s_).has(*(UNK + RHS_FLD))
    Mmat = sp.Matrix([[plain(cof(e, q)) for q in UNK] for e in EQS])
    Rmat = sp.Matrix(
        [[plain(-cof(e, f_)) for f_ in RHS_FLD] + [plain(-srcpart(e))] for e in EQS]
    )
    row_tt_p = plain(row_tt)
    log("[6] coefficient matrices built")
    # verification that (tt) is redundant: solve the 4x4 at random backgrounds in
    # 50-digit arithmetic (U' fixed by the (rr) background constraint, the only
    # first-order relation among U, V, W and their first derivatives)
    import mpmath as mp  # noqa: E402

    mp.mp.dps = 50
    _args = [r, Us, Ups, Vs, Vps, Ws, Wps, B, alpha, ws, wps]
    _fM = sp.lambdify(_args, Mmat, "mpmath")
    _fR = sp.lambdify(_args, Rmat, "mpmath")
    _ftt = sp.lambdify(_args + UNK + RHS_FLD, row_tt_p, "mpmath")
    _fUp = sp.lambdify([r, Us, Vs, Vps, Ws, Wps, B, alpha], plain(Up_sol[0]), "mpmath")
    worst_tt = mp.mpf(0)
    for _trial in range(5):
        v = {
            k: mp.mpf(_rng.uniform(0.3, 2.0))
            for k in ("r", "U", "V", "Vp", "W", "Wp", "B", "a", "w", "wp")
        }
        up = _fUp(v["r"], v["U"], v["V"], v["Vp"], v["W"], v["Wp"], v["B"], v["a"])
        a_ = [
            v["r"],
            v["U"],
            up,
            v["V"],
            v["Vp"],
            v["W"],
            v["Wp"],
            v["B"],
            v["a"],
            v["w"],
            v["wp"],
        ]
        Xv = mp.inverse(mp.matrix(_fM(*a_))) * mp.matrix(_fR(*a_))
        fld = [mp.mpf(_rng.uniform(-1, 1)) for _ in range(4)]
        vec = fld + [1]
        unk = [sum(Xv[i, j] * vec[j] for j in range(5)) for i in range(4)]
        val = _ftt(*a_, *unk, *fld)
        scale = max(abs(x_) for x_ in list(unk) + fld)
        worst_tt = max(worst_tt, abs(val) / scale)
    assert worst_tt < mp.mpf(10) ** -40, f"(tt) not redundant: {worst_tt}"
    log(
        f"[6] closed system {{rr, d/dr rr, xx, zz}}; (tt) redundant (residual {mp.nstr(worst_tt, 3)} at 50 digits)"
    )

    # ---------------------------------------------------------------------------
    # [7] Hellmann-Feynman identity for the pairing density
    # ---------------------------------------------------------------------------
    pi_dens = sp.expand(
        sqrtg * sum(gup[m, m] * gup[m, m] * S[m][m] * h[m][m] for m in range(N))
    )
    dP = P_sl * (Htt + Hzz - Hrr) / 2
    dQ = Q_sl * (Htt + Hzz + Hrr - 2 * Hxx) / 2
    assert is_zero(pi_dens + 2 * (dP * W0p**2 - dQ * W0**2))
    log("[7] sqrt(-g) S^{MN} h_MN = -2 (dP w0'^2 - dQ w0^2) pointwise")

    # ---------------------------------------------------------------------------
    # [8] horizon thermodynamics
    # ---------------------------------------------------------------------------
    # surface gravity of -A dt^2 + dr^2/C at a simple zero of A and C:
    #   kappa = (1/2) sqrt(A' C'),  A = U(1 + eps H_tt), C = U/(1 + eps H_rr)
    ht0, hr0, Up1 = sp.symbols("ht0 hr0 Up1")
    A_ = sp.Symbol("rho") * Up1 * (1 + eps * ht0)  # near-horizon, rho = r - r_h
    C_ = sp.Symbol("rho") * Up1 / (1 + eps * hr0)
    kap = sp.sqrt(sp.diff(A_, sp.Symbol("rho")) * sp.diff(C_, sp.Symbol("rho"))) / 2
    dlogT = sp.simplify(sp.diff(sp.log(kap), eps).subs(eps, 0))
    assert sp.simplify(dlogT - (ht0 - hr0) / 2) == 0
    log("[8] delta T/T = (H_tt - H_rr)(r_h)/2; delta s/s = (2 H_xx + H_zz)(r_h)/2")
    # (the area density is e^{2V+W} -> e^{2V+W}(1 + H_xx + H_zz/2) at fixed r_h;
    #  the horizon stays at r_h in any gauge with H_tt regular there)

    # ---------------------------------------------------------------------------
    # [9] horizon regularity
    # ---------------------------------------------------------------------------
    # The emitted 4x4 solve carries the 1/U poles of the (xx), (zz) rows; the
    # numerics build lim U x row by Richardson in the offset from r_h and verify the
    # rank-one structure (see the module docstring).

    # ---------------------------------------------------------------------------
    # [10] beta -> 0: the I_d channel reproduces derivations/g0.py
    # ---------------------------------------------------------------------------
    # I_d = H_xx - H_zz.  On the brane the mixed combination (xx) - (zz) still
    # carries H_rr through the B^2 stress; in the probe limit (alpha -> 0 at fixed
    # B, source per alpha) it must collapse to the g0.py massless-scalar channel
    #   (f I_d'/u^3)' = S_d/u^3,  S_d = -2 u^2 [f (dw0/du)^2 - 2 B w0^2],
    # on AdS5-Schwarzschild U = r^2 - 1/r^2, e^V = e^W = r, u = 1/r.
    diff_row = sp.expand(row_xx * sp.exp(-2 * Vf) - row_zz * sp.exp(-2 * Wf))
    uu = sp.Symbol("u", positive=True)
    Id = sp.Function("I_d")(uu)
    wu = sp.Function("w_u")(uu)  # w0 as a function of u
    flat_bg = {U: r**2 - 1 / r**2, Vf: sp.log(r), Wf: sp.log(r)}
    hom = sp.expand(diff_row.subs(alpha, 0))
    src_per_alpha = sp.expand(sp.diff(diff_row, alpha).subs(alpha, 0))
    src_per_alpha = sp.expand(src_per_alpha.subs({s_: 0 for s_ in UNK + RHS_FLD}))
    hom_flat = sp.expand(hom.subs(flat_bg).doit())
    assert is_zero(cof(hom_flat, hr)) and is_zero(cof(hom_flat, hr1))
    # slots in terms of I_d(u) (the hx + hz part must drop: check by isolating)
    cx_, cz_ = cof(hom_flat, hx2), cof(hom_flat, hz2)
    assert is_zero(cx_ + cz_), "probe (xx)-(zz) is not a pure I_d equation"
    assert is_zero(cof(hom_flat, hx1) + cof(hom_flat, hz1))
    assert is_zero(cof(hom_flat, hx)) and is_zero(cof(hom_flat, hz))
    # rewrite in u: I' = dI/dr = -u^2 dI/du, I'' = u^4 I_uu + 2u^3 I_u
    Iu1, Iu2 = sp.Derivative(Id, uu), sp.Derivative(Id, (uu, 2))
    eq_r = cx_ * (uu**4 * Iu2 + 2 * uu**3 * Iu1) + cof(hom_flat, hx1) * (-(uu**2) * Iu1)
    src_r = src_per_alpha.subs(flat_bg).doit()
    src_r = src_r.subs({W0p: -(uu**2) * sp.Derivative(wu, uu), W0: wu})
    eq_u = sp.expand((eq_r + src_r).subs(r, 1 / uu))
    f_g0 = 1 - uu**4
    Sd_g0 = -2 * uu**2 * (f_g0 * sp.Derivative(wu, uu) ** 2 - 2 * B * wu**2)
    ref_g0 = sp.expand(f_g0 * Iu2 + (sp.diff(f_g0, uu) - 3 * f_g0 / uu) * Iu1 - Sd_g0)
    ratio = sp.cancel(eq_u.coeff(Iu2) / ref_g0.coeff(Iu2))
    assert is_zero(sp.expand(eq_u - ratio * ref_g0)), (
        "probe I_d channel != derivations/g0.py"
    )
    log(
        f"[10] probe limit: (xx) - (zz) = ({sp.factor(ratio)}) x [g0.py I_d equation incl. its per-rho^2 source]"
    )

    # ---------------------------------------------------------------------------
    # [11] the s -> 0 limit: the kernel pairing is blind to x-diffeomorphisms
    # ---------------------------------------------------------------------------
    # At G != 0 the response is computed in the gauge H_tt = H_xx = 0; the G = 0
    # response has H_xx != 0.  Reaching H_xx = 0 at small G needs xi^x ~ 1/G, which
    # also shifts the photon: delta a_y = L_xi Abar_y = B xi^x, i.e. bhat = iG a_y
    # picks up an O(1) piece.  The kernel K = C4 - Re[X_c] - (1/2) Pi_bare
    # (X_c = int p1 bhat w0^2, Pi_bare = Re int sqrt(-g) S^{MN} conj(h_MN)) is
    # unchanged by xi = xi^x(r) e^{iGx} d_x for ANY complex profile, pointwise in r:
    # the x-x condensate stress is the magnetic pressure of the current that
    # couples to a_y.  So K(s -> 0) = K_{G=0}, and the O(1) photon screening seen
    # at s -> 0 in the G != 0 gauge is the gauge image of the uniform H_xx.
    Gs = sp.Symbol("G", positive=True)
    xiR, xiI = sp.Function("xi_R")(r), sp.Function("xi_I")(r)
    xi_up = (xiR + sp.I * xiI) * sp.exp(sp.I * Gs * x)  # contravariant xi^x
    dh_xx = 2 * gdn[1, 1] * sp.diff(xi_up, x)  # (L_xi g)_xx, g_xx independent of x
    dh_xr = gdn[1, 1] * sp.diff(xi_up, r)  # (L_xi g)_xr
    for m, n in [(0, 0), (2, 2), (3, 3), (4, 4)]:  # untouched components
        assert sp.simplify(xi_up * sp.diff(gdn[m, n], x)) == 0
    da_y = xi_up * sp.diff(Abar[2], x)  # (L_xi Abar)_y = xi^x d_x (B x) = B xi^x
    strip_E = sp.exp(sp.I * Gs * x)
    dX = sp.exp(Wf - 2 * Vf) * W0**2 * sp.I * Gs * sp.expand(da_y / strip_E)
    S_xr = 0  # bare stress has no xr component (it is pure bhat')
    dPi = sqrtg * (
        gup[1, 1] ** 2 * S[1][1] * sp.conjugate(sp.expand(dh_xx / strip_E))
        + 2 * gup[1, 1] * gup[4, 4] * S_xr * sp.conjugate(sp.expand(dh_xr / strip_E))
    )
    shift = sp.expand(dX + dPi / 2)
    assert (
        sp.simplify(
            sp.re(
                shift.subs(
                    {
                        xiR: sp.Symbol("a", real=True),
                        xiI: sp.Symbol("b", real=True),
                        W0: sp.Symbol("w", real=True),
                        Vf: sp.Symbol("v", real=True),
                        Wf: sp.Symbol("ww", real=True),
                    }
                )
            )
        )
        == 0
    )
    log(
        "[11] Re[delta X_c + delta Pi_bare/2] = 0 pointwise under xi^x(r) e^{iGx}: "
        "K is x-diffeomorphism invariant, so K(s -> 0) = K_{G=0}"
    )

    # ---------------------------------------------------------------------------
    # code generation: systems/dk_uniform.py
    # ---------------------------------------------------------------------------
    VC, WC = sp.symbols("V W")
    EMIT_SUBS = [
        (Up, sp.Symbol("Up")),
        (Vp, sp.Symbol("Vp")),
        (Wp, sp.Symbol("Wp")),
        (W0p, sp.Symbol("W0p")),
        (W0, sp.Symbol("W0")),
        (U, sp.Symbol("U")),
        (Vf, VC),
        (Wf, WC),
    ]

    def pyc(e):
        e = sp.sympify(e)
        for a, b in EMIT_SUBS:
            e = e.subs(a, b)
        return sp.pycode(e)

    from backreaction import paths  # noqa: E402

    gen_path = paths.SYSTEMS / "dk_uniform.py"
    ARGS = "r, U, Up, V, Vp, W, Wp, B, alpha, W0, W0p"
    with open(gen_path, "w") as fh:
        fh.write(
            '"""AUTO-GENERATED by dk_uniform.py -- do not edit.\n\n'
            "Uniform (G = 0) response of the D'Hoker-Kraus magnetic brane\n"
            "  ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2+dy^2) + e^{2W} dz^2,  A^3_y = Bx,\n"
            "to the lattice-averaged condensate stress, gauge H_tt = 0:\n"
            "  h = diag(0, e^{2V} H_xx, e^{2V} H_xx, e^{2W} H_zz, H_rr/U).\n"
            "Unknowns (H_rr, H_rr', H_xx'', H_zz'') from the rows {rr, d/dr rr, xx,\n"
            "zz}:  M . unk = R . (H_xx, H_xx', H_zz, H_zz', 1), solved numerically\n"
            "per radius (_solve).  The source carries the explicit factor alpha.\n"
            "Args: (" + ARGS + ").\n"
            '"""\n\nimport math  # noqa: F401\nimport numpy as np\n\n\n'
        )

        def emit_matrix(name, mat, ncol, doc):
            body = ",\n            ".join(
                "[" + ", ".join(pyc(mat[i, j]) for j in range(ncol)) + "]"
                for i in range(4)
            )
            fh.write(
                f'def {name}({ARGS}):\n    """{doc}"""\n'
                f"    return [\n            {body}]\n\n\n"
            )

        emit_matrix(
            "_Mmat", Mmat, 4, "4x4 coefficients of (H_rr, H_rr', H_xx'', H_zz'')."
        )
        emit_matrix("_Rmat", Rmat, 5, "4x5 RHS map onto (H_xx, H_xx', H_zz, H_zz', 1).")
        fh.write(
            f"def _solve({ARGS}):\n"
            '    """X = M^{-1} R (4x5); X[i] = coeffs of (H_xx, H_xx\', H_zz, H_zz\', 1)\n'
            "    in unknown i, order (H_rr, H_rr', H_xx'', H_zz'').\"\"\"\n"
            f"    M = np.array(_Mmat({ARGS}), dtype=float)\n"
            f"    R = np.array(_Rmat({ARGS}), dtype=float)\n"
            "    return np.linalg.solve(M, R)\n\n\n"
        )
        pd = sp.expand(pi_dens.subs(Htt, 0))
        fh.write(
            f"def pi_density({ARGS}, Hxx, Hzz, Hrr):\n"
            '    """sqrt(-g) S^{MN} h_MN in the gauge H_tt = 0 (= -2(dP w0\'^2 - dQ w0^2))."""\n'
            f"    return {pyc(pd.subs({Hxx: sp.Symbol('Hxx'), Hzz: sp.Symbol('Hzz'), Hrr: sp.Symbol('Hrr')}))}\n"
        )
    log(f"[gen] wrote {gen_path}")

    from backreaction.generated import stamp  # noqa: E402

    stamp(gen_path, __file__)
    log(f"[stamp] provenance written to {gen_path}")
    log("done -- all checks passed")


if __name__ == "__main__":
    main()
