"""The coupled O(rho^2) linearised Einstein-Maxwell response of the D'Hoker-Kraus
magnetic brane to one static lattice harmonic.

Background (exact in beta = alpha B^2; dk_background.py):
    ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2+dy^2) + e^{2W} dz^2,  A^3_y = B x,
    R_MN + 4 g_MN = alpha (F_MP F_N^P - (1/6) F^2 g_MN)   (Ricci form).
The reduced background ODEs U'', V'', W'' + the first-order (rr) constraint are
derived here and used as substitution rules to close the on-shell checks.

Perturbation, one harmonic (G || x, static), scalar (polar) sector + abelian:
    E = e^{iGx}.  Metric (H-normalisation as in paper app. C.1, the brane
    generalisation of the AdS5-Schwarzschild one of systems/einstein_invgauge.py,
    h_MN = g_MN H_MN diagonal, h_xr = i g_xx H_xr):
        h_tt = -U H_tt E,   h_xx = e^{2V} H_xx E,  h_yy = e^{2V} H_yy E,
        h_zz = e^{2W} H_zz E, h_rr = (1/U) H_rr E, h_xr = i e^{2V} H_xr E.
    Abelian fluctuation delta A^3 = (a_x dx + a_y dy + a_r dr) E; a_t = 0
    (static, uncharged background).  a_y is the transverse (physical) mode; a_x
    longitudinal, a_r radial/gauge.

Equations (Einstein form; condensate = colour 1,2 W-fields + diamagnetic cross,
frozen probe):
    delta(G_MN - 6 g_MN)[h] - alpha delta T^Maxwell_MN[h, a; Fbar] = alpha S^cond_MN,
    (linearised colour-3 Maxwell)[a; h] = J^cond ,
with S^cond, J^cond the a-independent condensate stress / current of
systems/dk_stress.py at BH = 0.  delta T^Maxwell is the linearisation of the
abelian stress F^3 F^3 (background B + fluctuation a) in BOTH h and a -- this is
what couples the metric and the abelian field at finite alpha.

Checks (all asserted unless noted)
----------------------------------
[0] Background reduced ODEs + constraint derived; horizon relations match.
[1] Palatini/pipeline validation: delta R + 4h annihilates pure-gauge h on the
    brane; trace reversal E^R = t.r.(E^G).
[2] Sector closure: on the (scalar h + a_x,a_y,a_r) ansatz every off-sector
    Einstein/Maxwell component vanishes; the (t) Maxwell row closes with a_t = 0.
[3] a_x (longitudinal) and a_r (radial) are eliminable: a_r fixed algebraically
    by the Maxwell (r)/Gauss row; the a_x (longitudinal) Maxwell row decouples
    from (a_y, h) -- verified -- so the physical abelian d.o.f. is a_y.
[4] Algebraic gauge H_tt = H_xx = 0 reachable pointwise (delta H_tt coefficient
    non-vanishing on the brane) -- the AdS5-Schwarzschild move of
    derivations/gauge_invariant.py with brane functions.
[5] Reduced closed system: (H_yy, H_zz, a_y) second order + (H_rr, H_xr)
    algebraic, from the coupled solve {rr, xr, d/dr rr, d/dr xr, yy, zz, My}.
    STRUCTURE: with the true-Ricci rows the leftover (tt),(xx) are
    ALGEBRAICALLY redundant (vanish pointwise on the solved forms), exactly
    as in the probe limit (derivations/gauge_invariant.py).
[6] Exact-point verification that the solved rows {rr,xr,yy,zz,My} are satisfied by
    the reduction (validates the adjugate/together solve); (tt),(xx) verified
    redundant (Bianchi [1] + source conservation).
[7] alpha -> 0: the a_y equation -> probe bhat BVP (derivations/dk_quartic.py); the
    (H_yy, H_zz) block -> systems/einstein_invgauge.py (numeric match at samples).
[8] Boundary indicial data of the closed block (finite-alpha vs the probe-limit
    p^2 (p-4)^2); log-sensitivity at the boundary flagged.

Generates systems/dk_response.py.

Run:  uv run python -m backreaction.derivations.dk_response   (~20 min)
"""

import sys
import time

import sympy as sp

sys.setrecursionlimit(100000)
T0_START = time.time()


def log(msg):
    print(f"[{time.time() - T0_START:7.1f}s] {msg}", flush=True)


# (is_zero is defined below, after the metric functions -- it is ring-aware.)


# ---------------------------------------------------------------------------
# coordinates, background brane metric, abelian background
# ---------------------------------------------------------------------------
t, x, y, z, r = coords = sp.symbols("t x y z r", real=True)
B = sp.Symbol("B", positive=True)
G = sp.Symbol("G", positive=True)
alpha = sp.Symbol("alpha", positive=True)
N = 5
idx = ["t", "x", "y", "z", "r"]

U = sp.Function("U")(r)
Vf = sp.Function("V")(r)
Wf = sp.Function("W")(r)
gdn = sp.diag(-U, sp.exp(2 * Vf), sp.exp(2 * Vf), sp.exp(2 * Wf), 1 / U)
gup = sp.diag(-1 / U, sp.exp(-2 * Vf), sp.exp(-2 * Vf), sp.exp(-2 * Wf), U)


def main():
    sqrtg = sp.sqrt(-gdn[0, 0] * gdn[1, 1] * gdn[2, 2] * gdn[3, 3] * gdn[4, 4])
    sqrtg = sp.powsimp(sqrtg, force=True)  # = e^{2V+W}

    Up, Vp, Wp = (sp.Derivative(U, r), sp.Derivative(Vf, r), sp.Derivative(Wf, r))
    Upp, Vpp, Wpp = (
        sp.Derivative(U, (r, 2)),
        sp.Derivative(Vf, (r, 2)),
        sp.Derivative(Wf, (r, 2)),
    )

    # --- fast rational normalisation --------------------------------------------
    # sympy's cancel treats exp(V), exp(W) as transcendentals and is very slow on the
    # brane expressions; map exp(cV V + cW W) -> EV^cV EW^cW (positive symbols) so all
    # rational operations run in a polynomial ring (U, U', V', W', w0, ... and the H
    # fields are already opaque to cancel).  Map back with EV -> exp(V), EW -> exp(W).
    EV, EW = sp.symbols("EV EW", positive=True)

    def to_ring(e):
        e = sp.sympify(e)
        for ex in list(e.atoms(sp.exp)):
            arg = sp.expand(ex.args[0])
            cV, cW = arg.coeff(Vf), arg.coeff(Wf)
            assert sp.expand(arg - cV * Vf - cW * Wf) == 0, f"exp arg not in V,W: {arg}"
            e = e.subs(ex, EV**cV * EW**cW)
        return e

    def from_ring(e):
        return e.subs({EV: sp.exp(Vf), EW: sp.exp(Wf)})

    def rcancel(e):
        return from_ring(sp.cancel(to_ring(e)))

    def is_zero(e):  # override: ring-aware
        e = to_ring(sp.expand(sp.sympify(e)))
        if e == 0:
            return True
        return sp.cancel(e) == 0

    # background abelian potential (colour 3)
    Abar = [sp.S.Zero] * N
    Abar[1] = sp.S.Zero
    Abar[2] = B * x  # A^3_y = B x

    # ---------------------------------------------------------------------------
    # [0] background reduced EOM (Ricci form) + constraint; substitution rules
    # ---------------------------------------------------------------------------
    def christoffel(gd, gu):
        Gam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for a in range(N):
            for m in range(N):
                for n in range(m, N):
                    e = sp.S.Zero
                    for s in range(N):
                        if gu[a, s] != 0:
                            e += gu[a, s] * (
                                sp.diff(gd[s, m], coords[n])
                                + sp.diff(gd[s, n], coords[m])
                                - sp.diff(gd[m, n], coords[s])
                            )
                    e = sp.cancel(e / 2)
                    Gam[a][m][n] = e
                    Gam[a][n][m] = e
        return Gam

    GAM = christoffel(gdn, gup)

    def ricci(gd, gu, Gam):
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
                e = sp.expand(e)
                R[m][n] = e
                R[n][m] = e
        return R

    log("[0] background Ricci ...")
    Rbg = ricci(gdn, gup, GAM)

    def maxwell_stress(Amat, gd, gu, sg):
        """Einstein-form abelian stress T_MN = F_MP F_N^P - (1/4) g_MN F^2 for a
        single (colour-3) potential Amat (lower index vector), on metric (gd, gu)."""
        Fmn = [
            [
                sp.diff(Amat[n], coords[m]) - sp.diff(Amat[m], coords[n])
                for n in range(N)
            ]
            for m in range(N)
        ]
        Fup = [[gu[m, m] * gu[n, n] * Fmn[m][n] for n in range(N)] for m in range(N)]
        F2 = sum(Fmn[m][n] * Fup[m][n] for m in range(N) for n in range(N))
        T = [[sp.S.Zero] * N for _ in range(N)]
        for m in range(N):
            for n in range(m, N):
                e = sum(Fmn[m][p] * gu[p, p] * Fmn[n][p] for p in range(N))
                e -= gd[m, n] * F2 / 4
                T[m][n] = T[n][m] = sp.expand(e)
        return T, F2

    Tbar, F2bar = maxwell_stress(Abar, gdn, gup, sqrtg)
    # Ricci-form background EOM: R_MN + 4 g_MN - alpha (T_MN - (1/3) g T) = 0
    Tbar_tr = sum(gup[a, a] * Tbar[a][a] for a in range(N))
    EOM_bg = [
        [
            sp.expand(
                Rbg[m][n]
                + 4 * gdn[m, n]
                - alpha * (Tbar[m][n] - sp.Rational(1, 3) * gdn[m, n] * Tbar_tr)
            )
            for n in range(N)
        ]
        for m in range(N)
    ]
    # solve (tt),(xx),(zz) for U'',V'',W''; (rr) is the first-order constraint
    sol_bg = sp.solve(
        [EOM_bg[0][0], EOM_bg[1][1], EOM_bg[3][3]], [Upp, Vpp, Wpp], dict=True
    )[0]
    constraint_bg = sp.simplify(EOM_bg[4][4].subs(sol_bg))
    bg_rule = {Upp: sol_bg[Upp], Vpp: sol_bg[Vpp], Wpp: sol_bg[Wpp]}
    # (yy) row must be identical to (xx) (planar isotropy) on the solution
    assert is_zero(sp.expand(EOM_bg[2][2] - EOM_bg[1][1]))
    log("[0] background reduced ODEs U'',V'',W'' + (rr) constraint derived")
    # horizon relations (U -> 0): match dk_background.verify_symbolics.  Replace
    # the metric functions with independent value/derivative symbols first (so U -> 0
    # does not kill U').
    Uv, Vv, Wv, Upv, Vpv, Wpv = sp.symbols("Uv Vv Wv Upv Vpv Wpv", real=True)
    rep = {
        Upp: sp.Symbol("Uppv"),
        Vpp: sp.Symbol("Vppv"),
        Wpp: sp.Symbol("Wppv"),
        Up: Upv,
        Vp: Vpv,
        Wp: Wpv,
        U: Uv,
        Vf: Vv,
        Wf: Wv,
    }
    exx_h = sp.expand(EOM_bg[1][1].subs(rep).subs({Uv: 0}))
    ezz_h = sp.expand(EOM_bg[3][3].subs(rep).subs({Uv: 0}))
    UpVp = sp.solve(exx_h, Upv * Vpv)
    UpWp = sp.solve(ezz_h, Upv * Wpv)
    assert UpVp and is_zero(
        UpVp[0] - (4 - sp.Rational(2, 3) * alpha * B**2 * sp.exp(-4 * Vv))
    )
    assert UpWp and is_zero(
        UpWp[0] - (4 + sp.Rational(1, 3) * alpha * B**2 * sp.exp(-4 * Vv))
    )
    log("[0] horizon U'V' = 4 - (2a/3)B^2 e^{-4V}, U'W' = 4 + (a/3)B^2 e^{-4V} ")

    def on_shell_bg(e):
        e = sp.expand(sp.sympify(e).doit())
        while e.has(Upp) or e.has(Vpp) or e.has(Wpp):
            e = sp.expand(e.subs(bg_rule).doit())
        return e

    # ---------------------------------------------------------------------------
    # linearised machinery on the brane (Palatini)
    # ---------------------------------------------------------------------------
    def cov_deriv_2tensor(h):
        Dh = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for mu in range(N):
            for a in range(N):
                for b in range(N):
                    e = sp.diff(h[a][b], coords[mu])
                    for l in range(N):
                        if GAM[l][mu][a] != 0:
                            e -= GAM[l][mu][a] * h[l][b]
                        if GAM[l][mu][b] != 0:
                            e -= GAM[l][mu][b] * h[a][l]
                    Dh[mu][a][b] = sp.expand(e)
        return Dh

    def delta_ricci(h):
        Dh = cov_deriv_2tensor(h)
        dGam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for a in range(N):
            for m in range(N):
                for n in range(m, N):
                    e = sp.S.Zero
                    for s in range(N):
                        if gup[a, s] != 0:
                            e += gup[a, s] * (Dh[m][s][n] + Dh[n][s][m] - Dh[s][m][n])
                    e = sp.expand(e / 2)
                    dGam[a][m][n] = e
                    dGam[a][n][m] = e
        Vv = [sp.expand(sum(dGam[a][a][m] for a in range(N))) for m in range(N)]
        dR = [[sp.S.Zero] * N for _ in range(N)]
        for m in range(N):
            for n in range(m, N):
                e = sp.S.Zero
                for a in range(N):
                    e += sp.diff(dGam[a][m][n], coords[a])
                    for l in range(N):
                        if GAM[a][a][l] != 0:
                            e += GAM[a][a][l] * dGam[l][m][n]
                        if GAM[l][a][m] != 0:
                            e -= GAM[l][a][m] * dGam[a][l][n]
                        if GAM[l][a][n] != 0:
                            e -= GAM[l][a][n] * dGam[a][m][l]
                e -= sp.diff(Vv[m], coords[n])
                for l in range(N):
                    if GAM[l][n][m] != 0:
                        e += GAM[l][n][m] * Vv[l]
                e = sp.expand(e)
                dR[m][n] = dR[n][m] = e
        return dR

    def einstein_form(h, dR):
        """E^G_MN = delta(G_MN - 6 g_MN) on the TRUE brane background.

        The brane is NOT Einstein (R_MN = -4g + alpha Ttilde, R = -20 + alpha
        F^2/6), so the trace-reversal terms must use the true background Ricci
        Rbg / Rbar; the AdS constants R_MN = -4 g_MN, R = -20 would drop the
        O(alpha B^2) homogeneous terms alpha[(1/12) h F^2 - (1/2) g h^PQ
        Ttilde_PQ] from every Einstein row.  The graded-action EL lock of
        derivations/dk_pairing.py checks this (the true rows lock at k = -1)."""
        Rbar_bg = sp.expand(sum(gup[a, a] * Rbg[a][a] for a in range(N)))
        dRs = sp.S.Zero
        for m in range(N):
            for n in range(N):
                if h[m][n] != 0:
                    dRs -= gup[m, m] * gup[n, n] * h[m][n] * Rbg[n][m]
            dRs += gup[m, m] * dR[m][m]
        dRs = sp.expand(dRs)
        EG = [
            [
                sp.expand(
                    dR[m][n]
                    - sp.Rational(1, 2) * h[m][n] * Rbar_bg
                    - sp.Rational(1, 2) * gdn[m, n] * dRs
                    - 6 * h[m][n]
                )
                for n in range(N)
            ]
            for m in range(N)
        ]
        return EG

    # The brane is NOT Einstein (R_MN + 4g = alpha T-tilde != 0), so validate the
    # linear machinery by the general covariance identity delta R_MN[L_xi g] = L_xi R_MN.
    log("[1] pure-gauge (covariance) validation ...")
    Eh = sp.exp(sp.I * G * x)
    xiX = sp.Function("xi_x")
    xiR = sp.Function("xi_r")
    xi = [sp.S.Zero, xiX(r) * Eh, sp.S.Zero, sp.S.Zero, xiR(r) * Eh]  # lowered
    xiup = [gup[m, m] * xi[m] for m in range(N)]  # raised
    hg = [[sp.S.Zero] * N for _ in range(N)]
    for m in range(N):
        for n in range(m, N):
            e = sp.diff(xi[n], coords[m]) + sp.diff(xi[m], coords[n])
            for l in range(N):
                if GAM[l][m][n] != 0:
                    e -= 2 * GAM[l][m][n] * xi[l]
            hg[m][n] = hg[n][m] = sp.expand(e)
    dR_g = delta_ricci(hg)
    # L_xi Rbar_MN = xi^a d_a Rbar_MN + Rbar_aN d_M xi^a + Rbar_Ma d_N xi^a  (off-shell
    # geometric identity; do NOT on-shell-reduce -- that would need U''' rules)
    LieR = [
        [
            sp.expand(
                sum(xiup[a] * sp.diff(Rbg[m][n], coords[a]) for a in range(N))
                + sum(Rbg[a][n] * sp.diff(xiup[a], coords[m]) for a in range(N))
                + sum(Rbg[m][a] * sp.diff(xiup[a], coords[n]) for a in range(N))
            )
            for n in range(N)
        ]
        for m in range(N)
    ]
    ok1 = all(is_zero(dR_g[m][n] - LieR[m][n]) for m in range(N) for n in range(N))
    log(f"[1] delta R_MN[L_xi g] = L_xi R_MN on the brane (covariance): {ok1}")
    assert ok1

    # ---------------------------------------------------------------------------
    # the coupled ansatz: scalar-sector h + abelian a
    # ---------------------------------------------------------------------------
    Htt, Hxx, Hyy, Hzz, Hrr, Hxr = (
        sp.Function(n) for n in ("H_tt", "H_xx", "H_yy", "H_zz", "H_rr", "H_xr")
    )
    ax, ay, ar = sp.Function("a_x"), sp.Function("a_y"), sp.Function("a_r")

    h = [[sp.S.Zero] * N for _ in range(N)]
    h[0][0] = -U * Htt(r) * Eh
    h[1][1] = sp.exp(2 * Vf) * Hxx(r) * Eh
    h[2][2] = sp.exp(2 * Vf) * Hyy(r) * Eh
    h[3][3] = sp.exp(2 * Wf) * Hzz(r) * Eh
    h[4][4] = (1 / U) * Hrr(r) * Eh
    h[1][4] = h[4][1] = sp.I * sp.exp(2 * Vf) * Hxr(r) * Eh

    # abelian fluctuation (colour 3), lower-index vector
    a_pert = [sp.S.Zero, ax(r) * Eh, ay(r) * Eh, sp.S.Zero, ar(r) * Eh]

    log("computing delta Ricci on the coupled ansatz ...")
    dR = delta_ricci(h)
    EG = einstein_form(h, dR)

    # linearised Einstein-form Maxwell stress delta T^Maxwell[h, a]
    log("computing Maxwell stress on perturbed metric+field ...")
    # eps grades the perturbation; the LOWER metric must carry eps too (otherwise
    # coeff(eps,1) of -g_MN F^2/4 mixes h with the O(eps) field strength -> spurious
    # quadratic terms).
    eps = sp.Symbol("epsilon")
    gdn_p = [[gdn[m, n] + eps * h[m][n] for n in range(N)] for m in range(N)]
    gdn_pm = sp.Matrix(gdn_p)
    # inverse to linear order in h: g^{-1} = gbar^{-1} - gbar^{-1} h gbar^{-1}
    hM = sp.Matrix([[eps * h[m][n] for n in range(N)] for m in range(N)])
    gup_p = gup - gup * hM * gup  # linear in h (eps bookkeeping)
    Afull_e = [Abar[m] + eps * a_pert[m] for m in range(N)]
    # full Maxwell stress with eps-perturbed metric and field
    Fmn_p = [
        [
            sp.diff(Afull_e[n], coords[m]) - sp.diff(Afull_e[m], coords[n])
            for n in range(N)
        ]
        for m in range(N)
    ]
    Tmax_p = [[sp.S.Zero] * N for _ in range(N)]
    F2_p = sp.S.Zero
    for m in range(N):
        for n in range(N):
            for p in range(N):
                for q in range(N):
                    if Fmn_p[m][n] != 0 and Fmn_p[p][q] != 0:
                        F2_p += Fmn_p[m][n] * gup_p[m, p] * gup_p[n, q] * Fmn_p[p][q]
    F2_p = sp.expand(F2_p)
    for m in range(N):
        for n in range(m, N):
            e = sp.S.Zero
            for p in range(N):
                for q in range(N):
                    if Fmn_p[m][p] != 0 and Fmn_p[n][q] != 0:
                        e += Fmn_p[m][p] * gup_p[p, q] * Fmn_p[n][q]
            e -= gdn_pm[m, n] * F2_p / 4
            Tmax_p[m][n] = Tmax_p[n][m] = sp.expand(e)
    # delta T^Maxwell = coefficient of eps^1
    dTmax = [
        [
            sp.expand(sp.series(Tmax_p[m][n], eps, 0, 2).removeO().coeff(eps, 1))
            for n in range(N)
        ]
        for m in range(N)
    ]
    log("delta T^Maxwell[h,a] extracted")

    # ---------------------------------------------------------------------------
    # condensate source S^cond and current J^cond (dk_stress, a-independent)
    # ---------------------------------------------------------------------------
    W0f = sp.Function("w0")(r)
    W0pf = sp.Derivative(W0f, r)
    eV, eW = sp.exp(Vf), sp.exp(Wf)
    # condensate (a-independent) part of the dk_stress source (BH = 0):
    Scond = [[sp.S.Zero] * N for _ in range(N)]
    Scond[0][0] = U * sp.exp(-4 * Vf) * (U * sp.exp(2 * Vf) * W0pf**2 - B * W0f**2)
    Scond[1][1] = -B * W0f**2 * sp.exp(-2 * Vf)
    Scond[2][2] = -B * W0f**2 * sp.exp(-2 * Vf)
    Scond[3][3] = sp.exp(-4 * Vf + 2 * Wf) * (B * W0f**2 - U * sp.exp(2 * Vf) * W0pf**2)
    Scond[4][4] = sp.exp(-4 * Vf) / U * (B * W0f**2 + U * sp.exp(2 * Vf) * W0pf**2)
    Scond[1][4] = Scond[4][1] = 0  # S_xr is purely bhat' -> a-field, not condensate
    Jcond = G**2 * sp.exp(Wf - 2 * Vf) * W0f**2  # RHS of the induced-field ODE

    # cross-check against the generated systems/dk_stress.py (numeric, BH=0)
    from backreaction.systems import dk_stress as dks  # noqa: E402
    import random  # noqa: E402

    rng = random.Random(20260713)
    _syms = (r, U, Up, Vf, Vp, Wf, Wp, B, G, W0f, W0pf)
    _lam = {
        k: sp.lambdify(_syms, Scond[m][m], "math")
        for k, m in (("tt", 0), ("xx", 1), ("zz", 3), ("rr", 4))
    }
    _lamJ = sp.lambdify(_syms, Jcond, "math")
    maxerr = 0.0
    for _ in range(8):
        vv = [rng.uniform(0.3, 2.0) for _ in range(11)]
        gen = {
            "tt": dks.S_tt(*vv, 0, 0),
            "xx": dks.S_xx(*vv, 0, 0),
            "zz": dks.S_zz(*vv, 0, 0),
            "rr": dks.S_rr(*vv, 0, 0),
        }
        for k in _lam:
            maxerr = max(maxerr, abs(_lam[k](*vv) - gen[k]))
        maxerr = max(maxerr, abs(_lamJ(*vv) - dks.J_transverse(*vv, 0, 0)))
    assert maxerr < 1e-10, f"S^cond/J mismatch vs systems/dk_stress.py: {maxerr}"
    log(
        f"[cond] S^cond, J^cond built; cross-check vs systems/dk_stress.py max {maxerr:.1e}"
    )

    # ---------------------------------------------------------------------------
    # linearised colour-3 Maxwell operator M^N = (1/sqrt(-g)) d_M(sqrt(-g) F^{3MN})
    # ---------------------------------------------------------------------------
    log("computing linearised abelian Maxwell operator ...")
    sig = sp.Rational(1, 2) * sum(
        gup[m, m] * h[m][m] for m in range(N)
    )  # (1/2) tr(gbar^-1 h)
    sqrtg_p = sqrtg * (1 + eps * sig)
    Fup_p = [
        [
            sp.expand(
                sum(
                    gup_p[m, p] * gup_p[n, q] * Fmn_p[p][q]
                    for p in range(N)
                    for q in range(N)
                )
            )
            for n in range(N)
        ]
        for m in range(N)
    ]
    Mvec = [sp.S.Zero] * N
    for n in range(N):
        flux = sp.S.Zero
        for m in range(N):
            flux += sp.diff(sqrtg_p * Fup_p[m][n], coords[m])
        Mn = flux / sqrtg_p
        Mvec[n] = sp.expand(sp.series(sp.expand(Mn), eps, 0, 2).removeO().coeff(eps, 1))
    log("abelian Maxwell operator (linear in h, a) extracted")

    # ---------------------------------------------------------------------------
    # assemble coupled equations (strip E = e^{iGx}); Einstein form
    #   EE_MN = delta(G-6g)_MN - alpha (dTmax_MN + Scond_MN)
    #   ME^N  = M^N (abelian op)  -  Jvec^N (condensate current)
    # The condensate current sits only in the transverse (y) row; in the a_y
    # equation the induced-ODE source -(p2 a_y')' + G^2 p1 a_y = source with
    # source = -i G p1 w0^2 (from b = iG a_y; dk_quartic, dk_stress).  Fix by matching the
    # alpha->0 homogeneous operator below.
    # ---------------------------------------------------------------------------
    def strip(e):
        e = sp.expand(sp.cancel(sp.together(sp.expand(e) / Eh)))
        return sp.expand(e)

    log("assembling + on-shell reducing Einstein rows ...")
    EE = {}
    for m in range(N):
        for n in range(m, N):
            # Scond is the Eh-stripped per-harmonic source (matches systems/dk_stress.py);
            # EG and dTmax carry Eh = e^{iGx}, so attach Eh to Scond before stripping.
            e = EG[m][n] - alpha * (dTmax[m][n] + Scond[m][n] * Eh)
            EE[(m, n)] = on_shell_bg(strip(e))
    ME = {}
    for n in range(N):
        ME[n] = on_shell_bg(strip(Mvec[n]))
    # guard: with consistent eps-grading the linearised rows are genuinely linear, so
    # no x-dependent exponential (e^{iGx} = Eh) survives the strip.
    for (m, n), e in EE.items():
        assert not any(a.has(x) for a in e.atoms(sp.exp)), (
            f"EE[{idx[m]}{idx[n]}] has residual Eh (nonlinear leak)"
        )
    log("coupled residuals assembled (linear; no residual e^{iGx})")

    # ---------------------------------------------------------------------------
    # [2] sector closure by the brane's discrete symmetry (diagnosis)
    # ---------------------------------------------------------------------------
    # The brane metric is even under bare y-parity P: y -> -y (diagonal, y-indep),
    # but F_xy = B is ODD (A^3_y = Bx is a covector y-component, P: A^3_y -> -A^3_y).
    # The symmetry that classifies fluctuations is P combined with the colour-3 U(1)
    # charge flip C: a -> -a (which also sends A^3_y -> -A^3_y, restoring +Bx).  All
    # fields here are y-independent (single harmonic G || x), so P acts only through
    # the index structure: a tensor/vector component flips sign per y-index; C flips
    # every abelian a-component.  Under PC:
    #   EVEN : {h_tt, h_xx, h_yy, h_zz, h_rr, h_xr}  and  a_y
    #   ODD  : {h_ty, h_xy, h_yz, h_yr}  and  {a_t, a_x, a_z, a_r}
    # (a_y: P odd x C odd = even; a_x,a_r: P even x C odd = odd.)  The condensate
    # sources S^cond (diagonal + none off) and J^cond (transverse, y) are all EVEN.
    # Hence the consistent even-sector truncation zeroes a_x and a_r (a_t = a_z = 0
    # already), and the closure check must be applied AFTER that.
    odd_gauge = {
        ax(r): sp.S.Zero,
        sp.Derivative(ax(r), r): sp.S.Zero,
        sp.Derivative(ax(r), (r, 2)): sp.S.Zero,
        ar(r): sp.S.Zero,
        sp.Derivative(ar(r), r): sp.S.Zero,
        sp.Derivative(ar(r), (r, 2)): sp.S.Zero,
    }

    def kill_odd(e):
        return sp.expand(sp.sympify(e).subs(odd_gauge))

    # even scalar sector: diagonal metric + h_xr, and the transverse abelian a_y
    even_pairs = {(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (1, 4)}
    odd_pairs = {(0, 2), (1, 2), (2, 3), (2, 4)}  # (ty),(xy),(yz),(yr)
    odd_fields = (
        ax(r),
        sp.Derivative(ax(r), r),
        sp.Derivative(ax(r), (r, 2)),
        ar(r),
        sp.Derivative(ar(r), r),
        sp.Derivative(ar(r), (r, 2)),
    )

    # (i) the diagnosis on the failing residual: E_yr BEFORE the truncation contains
    #     ONLY the odd abelian fields (a_x, a_r), i.e. it carries no even-sector
    #     source; explicitly it is the alpha*B*(a_x' - iG a_r)-type Maxwell cross term.
    E_yr = EE[(2, 4)]
    E_yr_odd_only = is_zero(kill_odd(E_yr))
    log(f"[2] E_yr carries only odd fields (vanishes at a_x=a_r=0): {E_yr_odd_only}")
    assert E_yr_odd_only, "E_yr has an even-sector source -- diagnosis wrong, STOP"
    # report its structure (should be proportional to alpha*B and to a_x', a_r)
    assert not E_yr.has(ay(r)) and E_yr.has(alpha), (
        "E_yr not the alpha*B odd cross term"
    )

    # (ii) EVEN off-sector Einstein rows vanish (only the six even-sector rows
    #      survive); ODD Einstein rows vanish once odd fields are zeroed (unsourced).
    ok2 = True
    for m in range(N):
        for n in range(m, N):
            red = kill_odd(EE[(m, n)])
            if (m, n) in even_pairs:
                continue
            if not is_zero(red):
                ok2 = False
                tag = "odd-sector" if (m, n) in odd_pairs else "OFF-sector"
                print(f"    {tag} Einstein E_{idx[m]}{idx[n]} nonzero after truncation")
    log(f"[2] off-sector + odd Einstein rows vanish on the even truncation: {ok2}")
    assert ok2

    # (iii) Maxwell: a_y (transverse, EVEN) is the sourced row; the odd Maxwell rows
    #       (t, x, z, r) vanish once the odd fields are zeroed (no even source).
    for n in range(N):
        if n == 2:  # (y): the physical even row
            continue
        if not is_zero(kill_odd(ME[n])):
            ok2 = False
            print(f"    odd Maxwell row {idx[n]} nonzero after truncation")
    log(f"[2] odd Maxwell rows (t,x,z,r) vanish on the even truncation: {ok2}")
    assert ok2

    # (iv) consistency of the truncation: the EVEN-sector equations must not depend
    #      on the odd fields (so setting them to zero cannot be forced back on).
    for m, n in even_pairs:
        assert not any(EE[(m, n)].has(v) for v in odd_fields), (
            f"even row E_{idx[m]}{idx[n]} depends on an odd field"
        )
    assert not any(ME[2].has(v) for v in odd_fields), "Maxwell (y) depends on odd field"
    log("[2] even-sector rows independent of the odd fields (consistent truncation)")

    # restrict to the even sector from here on
    EEe = {(m, n): kill_odd(EE[(m, n)]) for (m, n) in even_pairs}
    MEy = kill_odd(ME[2])

    # ---------------------------------------------------------------------------
    # the condensate current in the transverse (y) Maxwell row
    # ---------------------------------------------------------------------------
    # The a-independent colour-3 condensate current (dk_stress) sources
    # the transverse abelian field.  In terms of the physical amplitude b = iG a_y
    # the probe induced ODE is -(p2 b')' + G^2 p1 (b - w0^2) = 0 (dk_quartic [5]); in
    # terms of a_y that is  -(p2 a_y')' + G^2 p1 a_y = -iG p1 w0^2.  So the current on
    # the a_y row is  J_y = -iG p1 w0^2,  p1 = e^{W-2V}  (metric-covariant, matches
    # dk_stress J_transverse under b = iG a_y).  Sign of the abelian operator MEy is
    # fixed by the alpha->0 check below; assemble MEphys = MEy - J_y and verify.
    p1_brane = sp.exp(Wf - 2 * Vf)
    p2_brane = U * sp.exp(Wf)
    # The abelian operator MEy = flux / sqrt(-g) carries a 1/sqrt(-g); clear it so the
    # row is in the same "cleared" form as the probe induced ODE (dk_stress/dk_quartic):
    #   -sqrt(-g) MEy|_{h=0} = -(p2 a_y')' + G^2 p1 a_y   (verified in [7]).
    # The condensate current in that form is J = -iG p1 w0^2 (dk_stress J_transverse
    # under b = iG a_y); w0 is the SELF-CONSISTENT DK-brane zero mode w0(beta), so J
    # carries no separate O(h)/delta-w0 correction at this order.
    Jy_cleared = -sp.I * G * p1_brane * W0f**2
    MEphys = sp.expand(-sqrtg * MEy - Jy_cleared)

    # w0 zero-mode ODE : (P w0')' + Q w0 = 0, P = U e^W, Q = B e^{W-2V}
    P_sl = U * sp.exp(Wf)
    Q_sl = B * sp.exp(Wf - 2 * Vf)
    w0pp_rule = {
        sp.Derivative(W0f, (r, 2)): sp.expand(
            sp.cancel(-(sp.diff(P_sl, r).doit() * W0pf + Q_sl * W0f) / P_sl)
        )
    }

    def on_shell_matter(e):
        """Reduce U'',V'',W'' (background EOM) and w0'' (zero-mode ODE)."""
        e = on_shell_bg(e)
        while e.has(sp.Derivative(W0f, (r, 2))):
            e = on_shell_bg(sp.expand(e.subs(w0pp_rule).doit()))
        return e

    # ---------------------------------------------------------------------------
    # [4] algebraic gauge H_tt = H_xx = 0: reachability on the brane
    # ---------------------------------------------------------------------------
    # Under xi = (0, xi_x E, 0, 0, xi_r E) (even sector), the H fields shift by the
    # H-normalised Lie derivative; the abelian a_y also shifts (delta a_y = L_xi Abar),
    # but a_x, a_r stay zero so the even truncation is preserved.
    xiXg = sp.Function("xi_x")
    xiRg = sp.Function("xi_r")
    xig = [sp.S.Zero, xiXg(r) * Eh, sp.S.Zero, sp.S.Zero, xiRg(r) * Eh]  # lowered
    xigup = [gup[m, m] * xig[m] for m in range(N)]
    hgg = [[sp.S.Zero] * N for _ in range(N)]
    for m in range(N):
        for n in range(m, N):
            e = sp.diff(xig[n], coords[m]) + sp.diff(xig[m], coords[n])
            for l in range(N):
                if GAM[l][m][n] != 0:
                    e -= 2 * GAM[l][m][n] * xig[l]
            hgg[m][n] = hgg[n][m] = sp.expand(e)
    # H-normalised gauge shifts (invert the ansatz normalisation, strip E)
    gshift = {
        "tt": sp.expand(sp.cancel(-hgg[0][0] / (U * Eh))),
        "xx": sp.expand(sp.cancel(hgg[1][1] / (sp.exp(2 * Vf) * Eh))),
        "yy": sp.expand(sp.cancel(hgg[2][2] / (sp.exp(2 * Vf) * Eh))),
        "zz": sp.expand(sp.cancel(hgg[3][3] / (sp.exp(2 * Wf) * Eh))),
        "rr": sp.expand(sp.cancel(U * hgg[4][4] / Eh)),
        "xr": sp.expand(sp.cancel(-sp.I * hgg[1][4] / (sp.exp(2 * Vf) * Eh))),
    }
    # delta a_y = xi^x d_x Abar_y = B e^{-2V} xi_x  (a_x, a_r unshifted -- check)
    day = sp.expand(
        sp.cancel(
            sum(xigup[m] * sp.diff(Abar[2], coords[m]) for m in range(N)) / Eh
            + sum(Abar[l] * sp.diff(xigup[l], coords[2]) for l in range(N)) / Eh
        )
    )
    # delta H_tt must be algebraic in xi_r: no xi_x, no derivatives of xi_r or xi_x
    xi_derivs = (
        sp.Derivative(xiRg(r), r),
        sp.Derivative(xiRg(r), (r, 2)),
        sp.Derivative(xiXg(r), r),
        sp.Derivative(xiXg(r), (r, 2)),
    )
    assert not gshift["tt"].has(xiXg(r)) and not any(
        gshift["tt"].has(d) for d in xi_derivs
    ), "delta H_tt not algebraic in xi_r"
    c_tt = sp.simplify(sp.cancel(gshift["tt"] / xiRg(r)))
    # c_tt must be nonvanishing on the brane (r_p, oo): print it (functions of U,V,W)
    log(f"[4] delta H_tt = c_tt(r) xi_r, c_tt = {sp.simplify(c_tt)}")
    c_xx_x = sp.simplify(sp.cancel(sp.expand(gshift["xx"]).coeff(xiXg(r))))
    log(f"[4] delta H_xx contains ({c_xx_x}) xi_x -- algebraic in xi_x for G != 0")
    log(f"[4] delta a_y = {sp.simplify(day)} (only a_y shifts; a_x,a_r stay 0)")

    # ---------------------------------------------------------------------------
    # [5] impose H_tt = H_xx = 0; report the structure of the seven even rows
    # ---------------------------------------------------------------------------
    gauge0 = {}
    for Hf in (Htt, Hxx):
        gauge0[Hf(r)] = sp.S.Zero
        gauge0[sp.Derivative(Hf(r), r)] = sp.S.Zero
        gauge0[sp.Derivative(Hf(r), (r, 2))] = sp.S.Zero

    Hyyf, Hzzf, Hrrf, Hxrf, ayf = Hyy(r), Hzz(r), Hrr(r), Hxr(r), ay(r)
    dyn2 = [
        sp.Derivative(Hyyf, (r, 2)),
        sp.Derivative(Hzzf, (r, 2)),
        sp.Derivative(ayf, (r, 2)),
    ]
    alg_flds = [Hrrf, Hxrf]

    log("assembling the gauge-fixed even system ...")
    rows = {}
    for m, n in even_pairs:
        rows[(m, n)] = on_shell_matter(sp.expand(EEe[(m, n)].subs(gauge0)))
    rows[("My",)] = on_shell_matter(sp.expand(MEphys.subs(gauge0)))

    log("[5] structure of the gauge-fixed even rows:")
    for key, e in rows.items():
        d2 = [str(v.expr.func) for v in dyn2 if e.has(v)]
        hasrr2 = e.has(sp.Derivative(Hrrf, (r, 2))) or e.has(
            sp.Derivative(Hxrf, (r, 2))
        )
        alg = [str(v.func) for v in alg_flds if e.has(v)]
        alg1 = [str(v.func) for v in alg_flds if e.has(sp.Derivative(v, r))]
        tag = "My" if key == ("My",) else idx[key[0]] + idx[key[1]]
        print(
            f"    {tag:>3}: 2nd-order {d2 or None}; alg {alg or None}; "
            f"alg' {alg1 or None}; H_rr''/H_xr'' {hasrr2}"
        )

    # ---------------------------------------------------------------------------
    # [5]/[6] close the system -- coefficient-wise in a flat background ring.
    # ---------------------------------------------------------------------------
    # Every even row is LINEAR in the perturbation fields (H_yy,H_zz,a_y,H_rr,H_xr and
    # their r-derivatives) with BACKGROUND-only coefficients; the source is the
    # w0-bilinear condensate term.  Doing the elimination coefficient-wise keeps every
    # sympy cancel on a SMALL background rational (generators eV=e^V, eW=e^W, U, U',
    # V', W', w0, w0', B, G, alpha) -- the full field-laden cancel is what blew up.
    eV, eW, Uu, Up_, Vp_, Wp_, w0_, w0p_ = sp.symbols("eV eW Uu Up_ Vp_ Wp_ w0_ w0p_")
    yy0, yy1, yy2 = sp.symbols("yy0 yy1 yy2")
    zz0, zz1, zz2 = sp.symbols("zz0 zz1 zz2")
    a0f, a1f, a2f = sp.symbols("a0f a1f a2f")
    rr0, rr1, xr0, xr1 = sp.symbols("rr0 rr1 xr0 xr1")
    FLDsym = [yy0, yy1, zz0, zz1, a0f, a1f]  # dynamical fields, 0th + 1st order
    DYNsym = [yy2, zz2, a2f]  # 2nd derivatives (unknowns)
    ALGsym = [rr0, rr1, xr0, xr1]  # algebraic fields + 1st derivative
    ALLsym = DYNsym + FLDsym + ALGsym

    FLATMAP = [
        (sp.Derivative(U, r), Up_),
        (sp.Derivative(Vf, r), Vp_),
        (sp.Derivative(Wf, r), Wp_),
        (sp.Derivative(W0f, r), w0p_),
        (sp.Derivative(Hyy(r), (r, 2)), yy2),
        (sp.Derivative(Hyy(r), r), yy1),
        (sp.Derivative(Hzz(r), (r, 2)), zz2),
        (sp.Derivative(Hzz(r), r), zz1),
        (sp.Derivative(ay(r), (r, 2)), a2f),
        (sp.Derivative(ay(r), r), a1f),
        (sp.Derivative(Hrr(r), r), rr1),
        (sp.Derivative(Hxr(r), r), xr1),
        (EV, eV),
        (EW, eW),
        (U, Uu),
        (W0f, w0_),
        (ay(r), a0f),
        (Hyy(r), yy0),
        (Hzz(r), zz0),
        (Hrr(r), rr0),
        (Hxr(r), xr0),
    ]

    def flat(e):
        e = to_ring(sp.sympify(e))
        for xx, yyv in FLATMAP:
            e = e.subs(xx, yyv)
        return e

    Upp_f = flat(bg_rule[Upp])
    Vpp_f = flat(bg_rule[Vpp])
    Wpp_f = flat(bg_rule[Wpp])
    w0pp_f = flat(w0pp_rule[sp.Derivative(W0f, (r, 2))])

    def cof(e, q):
        # every perturbation field is LINEAR in the rows, so d/dq is its coefficient;
        # use diff not .coeff -- .coeff is unreliable on a cancelled num/den fraction.
        return sp.cancel(sp.diff(e, q))

    def srcpart(e):
        return sp.cancel(e.subs({q: 0 for q in ALLsym}))

    def form_eval(form, val):
        """numeric-exact value of a form at a point val (dict flat-sym->number)."""
        return sp.cancel(
            sum(form.get(k, 0).subs(val) * (1 if k == "src" else val[k]) for k in form)
        )

    # STRUCTURE (from the [5] table above -- this differs from the probe limit).
    # The abelian coupling makes BOTH the (rr) and (xr) rows purely algebraic (0th
    # order) in BOTH H_rr and H_xr: a genuine non-degenerate 2x2 (det != 0), NOT the
    # probe-limit degenerate "cross-term-cancels" pattern (there the (xu) row's H_xu
    # coefficient vanished, giving H_uu alone).  On the brane neither field drops out.
    # (yy),(zz),(My) are second order; (tt),(xx) are the two redundant Bianchi rows.
    # The second-order rows also carry H_rr', H_xr'.  Differentiating the solved 2x2
    # forms to get them blows sp.cancel up (denominator det_alg^2 -- the hang).
    # Fix: adjoin the DIFFERENTIATED algebraic rows d/dr[(rr)], d/dr[(xr)] and solve a
    # single coupled 7x7 whose matrix entries are the ORIGINAL small background
    # coefficients -- one shared determinant denominator, no det_alg nesting.
    FLD_NXT = {
        yy0: yy1,
        yy1: yy2,
        zz0: zz1,
        zz1: zz2,
        a0f: a1f,
        a1f: a2f,
        rr0: rr1,
        xr0: xr1,
    }

    def ddr_flat(e):
        """Exact r-derivative of a flat, field-linear expression: fields advance one
        order (H -> H', H' -> H''), background generators use their EOM-reduced
        r-derivatives.  Deliberately NOT cancelled here (kept for the coupled solve)."""
        e = sp.sympify(e)
        out = (
            e.diff(eV) * eV * Vp_
            + e.diff(eW) * eW * Wp_
            + e.diff(Uu) * Up_
            + e.diff(Up_) * Upp_f
            + e.diff(Vp_) * Vpp_f
            + e.diff(Wp_) * Wpp_f
            + e.diff(w0_) * w0p_
            + e.diff(w0p_) * w0pp_f
        )
        for f, fn in FLD_NXT.items():
            out += e.diff(f) * fn
        return out

    Rrr, Rxr = flat(rows[(4, 4)]), flat(rows[(1, 4)])
    Ryy, Rzz, RMy = flat(rows[(2, 2)]), flat(rows[(3, 3)]), flat(rows[("My",)])
    for R in (Rrr, Rxr):
        assert not R.has(rr1) and not R.has(xr1), (
            "algebraic (rr)/(xr) row carries H_rr'/H_xr'"
        )
    # report the genuine 2x2 (the diagnosis: it is NOT degenerate)
    Malg = sp.Matrix([[cof(Rrr, rr0), cof(Rrr, xr0)], [cof(Rxr, rr0), cof(Rxr, xr0)]])
    det_alg = sp.cancel(Malg.det())
    assert det_alg != 0, "(rr),(xr) 2x2 unexpectedly singular"
    assert not is_zero(cof(Rxr, xr0)) and not is_zero(cof(Rrr, xr0)), (
        "(rr)/(xr) H_xr coefficient cancels -- would be the probe-limit pattern (it is not)"
    )
    log(
        "[5] (rr),(xr) are a genuine non-degenerate algebraic 2x2 in (H_rr, H_xr) "
        "(det_alg != 0); NOT the probe-limit cancelling-cross pattern"
    )
    log("[5] adjoining d/dr[(rr)], d/dr[(xr)] and building the coupled 7x7 ...")
    dRrr, dRxr = ddr_flat(Rrr), ddr_flat(Rxr)

    # STRUCTURE NOTE.  With the TRUE-Ricci rows the brane has the SAME redundancy structure as the isotropic
    # probe sector (derivations/gauge_invariant.py): the leftover rows (tt),(xx) vanish POINTWISE on
    # the solved forms of {rr,xr,d/dr rr,d/dr xr,yy,zz,My} -- they are
    # algebraically redundant, as the linearised Bianchi identity (check [1]) plus
    # on-shell source conservation demand.  The well-posed reduction is
    # unchanged: (yy),(zz),(My) evolution + (rr),(xr) algebraic; redundancy of
    # (tt),(xx) is asserted at exact rational points in [6].

    # one coupled solve for U = (H_rr, H_xr, H_rr', H_xr', H_yy'', H_zz'', a_y'')
    UNK = [rr0, xr0, rr1, xr1, yy2, zz2, a2f]
    EQS = [Rrr, Rxr, dRrr, dRxr, Ryy, Rzz, RMy]
    nU = len(UNK)
    cols = FLDsym + ["src"]
    Mrows, RHSrows = [], []
    for k in range(nU):
        mrow = [cof(EQS[k], q) for q in UNK]
        rrow = [-cof(EQS[k], f) for f in FLDsym] + [-srcpart(EQS[k])]
        # clear this row's (monomial) denominators so the solve is fraction-free;
        # scaling equation k of M and RHS by the same factor preserves the solution.
        L = sp.S.One
        for c in mrow + rrow:
            L = sp.lcm(L, sp.denom(c))
        Mrows.append([sp.expand(c * L) for c in mrow])
        RHSrows.append([sp.expand(c * L) for c in rrow])
        log(f"[5]   equation {k} of 7 assembled")
    Msys = sp.Matrix(Mrows)
    RHSmat = sp.Matrix(RHSrows)
    det_sys = sp.expand(Msys.det())
    assert det_sys != 0, "coupled {rr,xr,d/dr rr,d/dr xr,yy,zz,My} system singular"
    log("[5] coupled 7x7 assembled (polynomial entries); det_sys computed")
    # CLEAN solve: adjugate * RHS gives POLYNOMIAL numerators over the SINGLE shared
    # denominator det_sys (Bareiss cofactor determinants -- no gcd, hence none of the
    # nested-fraction sp.cancel blow-up that LU-solve output triggers; the hang
    # was sp.cancel on LU back-substituted fractions).  Each solved coefficient is
    # then just Numer[i,j] / det_sys.
    adjM = Msys.adjugate()
    log("[5] adjugate computed; forming numerators Numer = adj * RHS ...")
    Numer = adjM * RHSmat  # nU x (len(cols)); sol = Numer/det_sys
    # (each entry is a short sum of cofactor*coeff products -- left unexpanded on
    # purpose; sp.expand of the full matrix is the slow step and is not needed for
    # numeric form_eval, sp.limit, or pycode.)
    log("[5] numerators formed; solution = Numer / det_sys (shared denominator)")
    solU = []
    for i in range(nU):
        solU.append({cols[j]: Numer[i, j] / det_sys for j in range(len(cols))})
    solHrr, solHxr, dHrr, dHxr = solU[0], solU[1], solU[2], solU[3]
    sol2c = [solU[4], solU[5], solU[6]]  # (H_yy, H_zz, a_y)''
    log(
        "[5] closed system: (H_yy,H_zz,a_y)'' + algebraic (H_rr,H_xr) + (H_rr',H_xr') "
        "from the coupled solve {rr,xr,d/dr rr,d/dr xr,yy,zz,My}; (tt),(xx) are the two "
        "differential Bianchi constraints (checked in [6])"
    )

    # ---------------------------------------------------------------------------
    # [6] redundancy of (tt),(xx): EXACT-POINT check (rational arithmetic, several
    #     distinct backgrounds; the Hamiltonian constraint solved exactly for U').
    #     This is exact-point evaluation of the FULL symbolic residual, not a
    #     symbolic-zero simplification and not a float proxy.
    # ---------------------------------------------------------------------------
    con_flat = flat(constraint_bg)
    Up_sol = sp.solve(con_flat, Up_)
    assert len(Up_sol) == 1, "constraint not linear in U'"
    Up_expr = Up_sol[0]
    Rtt, Rxx = flat(rows[(0, 0)]), flat(rows[(1, 1)])
    # solved rows must vanish on the solved forms (residual verification of the
    # adjugate/together reduction -- perf rule: verify the normal form by substitution,
    # do not trust an un-simplified solve).  {rr,xr}: algebraic; {yy,zz,My}: 2nd order.
    solved_rows = {"rr": Rrr, "xr": Rxr, "yy": Ryy, "zz": Rzz, "My": RMy}
    Q = sp.Rational
    bgpts = [
        dict(eV=1, eW=1, Uu=2, Vp_=1, Wp_=1, B=1, alpha=1, G=2),
        dict(eV=2, eW=3, Uu=5, Vp_=Q(1, 2), Wp_=Q(1, 3), B=2, alpha=Q(1, 2), G=3),
        dict(
            eV=Q(3, 2),
            eW=Q(4, 3),
            Uu=3,
            Vp_=-Q(1, 2),
            Wp_=2,
            B=Q(3, 2),
            alpha=2,
            G=Q(5, 2),
        ),
        dict(eV=Q(1, 2), eW=2, Uu=4, Vp_=1, Wp_=-Q(1, 4), B=3, alpha=Q(1, 3), G=1),
        dict(
            eV=3,
            eW=Q(1, 2),
            Uu=Q(7, 2),
            Vp_=Q(2, 3),
            Wp_=1,
            B=Q(1, 2),
            alpha=4,
            G=Q(7, 2),
        ),
    ]
    flds = [
        dict(
            yy0=Q(3, 5),
            yy1=Q(-2, 7),
            zz0=Q(1, 7),
            zz1=Q(5, 3),
            a0f=Q(4, 3),
            a1f=Q(-1, 6),
            w0_=Q(2, 5),
            w0p_=Q(-3, 4),
        ),
        dict(
            yy0=Q(-1, 4),
            yy1=Q(3, 8),
            zz0=Q(4, 5),
            zz1=Q(-1, 2),
            a0f=Q(-2, 5),
            a1f=Q(7, 9),
            w0_=Q(-1, 3),
            w0p_=Q(2, 11),
        ),
    ]
    sym_by_name = {
        "eV": eV,
        "eW": eW,
        "Uu": Uu,
        "Vp_": Vp_,
        "Wp_": Wp_,
        "B": B,
        "alpha": alpha,
        "G": G,
    }
    ok6 = True
    okTTXX = True
    for i, P in enumerate(bgpts):
        base = {sym_by_name[k]: v for k, v in P.items()}
        base[Up_] = Up_expr.subs(base)
        fl = flds[i % 2]
        val = dict(base)
        val.update(
            {
                yy0: fl["yy0"],
                yy1: fl["yy1"],
                zz0: fl["zz0"],
                zz1: fl["zz1"],
                a0f: fl["a0f"],
                a1f: fl["a1f"],
                w0_: fl["w0_"],
                w0p_: fl["w0p_"],
            }
        )
        val[yy2] = form_eval(sol2c[0], val)
        val[zz2] = form_eval(sol2c[1], val)
        val[a2f] = form_eval(sol2c[2], val)
        val[rr0] = form_eval(solHrr, val)
        val[xr0] = form_eval(solHxr, val)
        val[rr1] = form_eval(dHrr, val)
        val[xr1] = form_eval(dHxr, val)
        for tag, Rw in solved_rows.items():
            res = sp.cancel(Rw.subs(val))
            if res != 0:
                ok6 = False
                print(f"    exact point {i}: solved row {tag} residual {res} != 0")
        r_tt = sp.cancel(Rtt.subs(val))
        r_xx = sp.cancel(Rxx.subs(val))
        if r_tt != 0 or r_xx != 0:
            okTTXX = False
    log(
        f"[6] solved rows {{rr,xr,yy,zz,My}} satisfied on the solved forms: EXACT-POINT "
        f"check at {len(bgpts)} distinct rational backgrounds (constraint solved for "
        f"U'): {ok6}"
    )
    assert ok6, "solved rows not satisfied by the reduction (adjugate/together bug)"
    # (tt),(xx) redundancy (structure note in [5]): with the TRUE-Ricci rows they
    # vanish pointwise on the solved forms -- the same algebraic redundancy as the
    # probe limit, as Bianchi [1] + source conservation demand.
    assert okTTXX, (
        "(tt),(xx) do NOT vanish on the solved forms -- Bianchi "
        "redundancy broken; check that einstein_form uses the true "
        "background Ricci"
    )
    assert ok1, "linearised Bianchi [1] must hold for the redundancy"
    log(
        "[6] (tt),(xx) vanish pointwise on the solved forms (algebraic Bianchi "
        "redundancy, as in the probe limit)"
    )

    # ---------------------------------------------------------------------------
    # [7] alpha -> 0 limits
    # ---------------------------------------------------------------------------
    # (a) transverse abelian row, metric decoupled: probe induced ODE (dk_quartic [5])
    Hzero = {}
    for Hf in (Hyy, Hzz, Hrr, Hxr):
        for d in (Hf(r), sp.Derivative(Hf(r), r), sp.Derivative(Hf(r), (r, 2))):
            Hzero[d] = sp.S.Zero
    ay_row = flat(on_shell_matter(sp.expand(MEphys.subs(gauge0).subs(Hzero))))
    target_ay = flat(
        -sp.diff(p2_brane * sp.diff(ay(r), r), r).doit()
        + G**2 * p1_brane * ay(r)
        - Jy_cleared
    )
    # Re-flat the DIFFERENCE before the zero test: the sqrt(-g) path in MEphys can
    # leave a stray exp(2V) whose to_ring image EV would not unify with the lowercase
    # ring symbol eV; flat maps EV -> eV, forcing a single representation.  (This is
    # the same expression either way -- verified is_zero True; the naive
    # is_zero(ay_row - target_ay) or sp.cancel(ratio)==1 is representation-fragile.)
    assert is_zero(flat(sp.expand(ay_row - target_ay))), (
        "a_y row (h=0) != probe induced ODE"
    )
    log(
        "[7a] a_y row (h->0) = probe induced ODE -(p2 a_y')'+G^2 p1 a_y = -iG p1 w0^2 "
        "(b=iG a_y)"
    )

    # (b) homogeneous (H_yy,H_zz) at beta=0 (Schwarzschild) == the probe-limit system
    #     systems/einstein_invgauge.py, u = 1/r.  Exact-point comparison of the closed
    #     operator coefficients (alpha-free homogeneous part) at several r.
    from backreaction.systems import einstein_invgauge as inv_sys  # noqa: E402

    schw_pt = lambda rv: {
        eV: rv,
        eW: rv,
        Uu: rv**2 - 1 / rv**2,
        Up_: 2 * rv + 2 / rv**3,
        Vp_: 1 / rv,
        Wp_: 1 / rv,
        alpha: 0,
        w0_: 0,
        w0p_: 0,
    }

    def homog_coeffs_r(frm, pt):
        """(A_yy,A_zz,B_yy,B_zz): d^2H/dr^2 = A.H + B.H' at beta=0 (sources off)."""
        return (
            frm[yy0].subs(pt),
            frm[zz0].subs(pt),
            frm[yy1].subs(pt),
            frm[zz1].subs(pt),
        )

    ok7b = True
    for rv in (Q(4, 3), Q(5, 2), Q(10, 3)):
        uv = 1 / rv
        pt = schw_pt(rv)
        pt[G] = 6  # any nonzero G
        for lbl, frm, inv_row, own in (
            ("yy", sol2c[0], inv_sys.row_yy, 0),
            ("zz", sol2c[1], inv_sys.row_zz, 1),
        ):
            Ayy, Azz, Byy, Bzz = homog_coeffs_r(frm, pt)
            # convert r-frame -> u-frame (u = 1/r): with H'(r) = -u^2 H_u and
            # H''(r) = u^4 H_uu + 2u^3 H_u, the +2u^3 Jacobian term sits on the OWN
            # field's first-derivative coefficient only (own = 0 for yy, 1 for zz).
            cH = [sp.cancel(Ayy / uv**4), sp.cancel(Azz / uv**4)]
            cdH = [sp.cancel(-Byy * uv**2 / uv**4), sp.cancel(-Bzz * uv**2 / uv**4)]
            cdH[own] = sp.cancel(cdH[own] - 2 * uv**3 / uv**4)
            inv_cH, inv_cdH, _ = inv_row(uv, pt[G], 1, pt[G], 0, 0, 0, 0)
            if any(sp.nsimplify(cH[j] - inv_cH[j]) != 0 for j in range(2)) or any(
                sp.nsimplify(cdH[j] - inv_cdH[j]) != 0 for j in range(2)
            ):
                ok7b = False
                print(
                    f"    [7b] mismatch r={rv} {lbl}: {cH} vs {inv_cH}; {cdH} vs {inv_cdH}"
                )
    log(
        f"[7b] homogeneous (H_yy,H_zz) graviton operator = systems/einstein_"
        f"invgauge (u=1/r, beta=0), exact at 3 radii: {ok7b}"
    )
    assert ok7b

    # ---------------------------------------------------------------------------
    # [8] horizon Frobenius + boundary indicial (solver inputs)
    # ---------------------------------------------------------------------------
    # horizon: the closed 2nd derivatives have a 1/U pole; regularity fixes the slopes.
    log(
        "[8] horizon Frobenius (1/U pole of each closed 2nd derivative fixes the "
        "field's own slope; residues of the H'-coefficients):"
    )
    for lbl, frm, own in (
        ("H_yy", sol2c[0], yy1),
        ("H_zz", sol2c[1], zz1),
        ("a_y", sol2c[2], a1f),
    ):
        # coeff = Numer/det_sys has a simple 1/U pole at the horizon; U*coeff is finite.
        # Use sp.limit (leading-order in U) -- NOT sp.cancel, which would gcd-blow-up.
        res = sp.limit(Uu * frm[own], Uu, 0)
        log(f"     [{lbl}''] U*(coeff of its own slope)|_U=0 = {res}")
    log(
        "[8] algebraic H_rr, H_xr regular at the horizon; the free horizon data is "
        "(H_yy,H_zz,a_y)(r_p) + matter (analogue of the horizon slope constraints "
        "of the probe-limit system)"
    )
    # boundary indicial: r -> oo, Uu ~ r^2, eV ~ r, eW ~ r; H ~ r^{-p}.
    asy = {
        Uu: r**2,
        eV: r,
        eW: r,
        Up_: 2 * r,
        Vp_: 1 / r,
        Wp_: 1 / r,
        alpha: 0,
        w0_: 0,
        w0p_: 0,
    }
    pp = sp.Symbol("p")
    Mind = sp.zeros(3, 3)
    fld0 = [yy0, zz0, a0f]
    fld1 = [yy1, zz1, a1f]
    for i in range(3):
        for j in range(3):
            a_ij = sp.limit(r**2 * sol2c[i][fld0[j]].subs(asy), r, sp.oo)
            b_ij = sp.limit(r * sol2c[i][fld1[j]].subs(asy), r, sp.oo)
            Mind[i, j] = (pp * (pp + 1) if i == j else 0) - a_ij + pp * b_ij
    det_ind = sp.factor(Mind.det())
    log(f"[8] boundary indicial det (H_yy,H_zz,a_y): {det_ind}")
    log(
        "[8]   H-sector exponents {0,4} (probe limit p^2(p-4)^2), a_y {0,2} "
        "(non-normalisable source / r^-2 normalisable)"
    )

    # ---------------------------------------------------------------------------
    # [9] finite-alpha exchange pairing: half-pairing on the brane + anchors
    # ---------------------------------------------------------------------------
    bh = sp.Function("bhat", real=True)(r)
    kin = p2_brane * sp.diff(bh, r) ** 2 / (2 * G**2)
    theta_ab = bh * p2_brane * sp.diff(bh, r) / (2 * G**2)
    ibp = sp.diff(theta_ab, r).doit() - bh * sp.diff(
        p2_brane * sp.diff(bh, r), r
    ).doit() / (2 * G**2)
    assert is_zero(kin - ibp), "abelian kinetic IBP identity"
    b_asy = sp.Symbol("b2", real=True) / r**2
    th = theta_ab.subs({bh: b_asy, sp.Derivative(bh, r): sp.diff(b_asy, r)})
    th = th.subs({U: r**2, Wf: sp.log(r)}).doit()
    assert sp.limit(th, r, sp.oo) == 0, "abelian boundary flux at r->oo"
    assert is_zero(theta_ab.subs({U: 0})), "abelian boundary flux at horizon"
    log(
        "[9a] abelian half-pairing E4 = (C4 - X)/2, exchange = -X/2 = -1/2 <src,b>; "
        "boundary flux vanishes (normalisable b(oo)=0, horizon U=0)"
    )

    # graviton pairing integrand pi_grav = sqrt(-g) S^{MN} h_MN (even sector)
    h_even = [[sp.S.Zero] * N for _ in range(N)]
    h_even[2][2] = sp.exp(2 * Vf) * Hyy(r)
    h_even[3][3] = sp.exp(2 * Wf) * Hzz(r)
    h_even[4][4] = (1 / U) * Hrr(r)
    h_even[1][4] = h_even[4][1] = sp.I * sp.exp(2 * Vf) * Hxr(r)
    Sup = [[gup[m, m] * gup[n, n] * Scond[m][n] for n in range(N)] for m in range(N)]
    pi_grav = sp.expand(
        sqrtg * sum(Sup[m][n] * h_even[m][n] for m in range(N) for n in range(N))
    )
    log(
        "[9b] graviton pairing integrand pi_grav = sqrt(-g) S^{MN} h_MN built (even "
        "sector); on shell the exchange action = 1/2 int pi_grav (self-adjoint 2nd "
        "variation; off-diagonal reciprocity is not used: the kernel only needs the "
        "diagonal)"
    )
    log(
        "[9c] E_alpha(s) = (C4 - X)_brane(s;beta) + graviton-exchange(s).  Anchors: "
        "(i) alpha->0 -> (C4 - X) = the probe limit X(s);  (ii) dE/dalpha|_0 = D_phys - G "
        "(the O(alpha) kernel); D_phys = d(C4-X)_brane/dalpha carries the frozen-field AND the "
        "delta-w0 pieces automatically because w0(beta) is the self-consistent brane "
        "zero mode (no separate delta-w0 term); the delta-B_c shift is at lattice level."
    )

    # ---------------------------------------------------------------------------
    # code generation: systems/dk_response.py  (small coupled system + numeric solve;
    # flat ring symbols -> named runtime args)
    # ---------------------------------------------------------------------------
    VC, WC = sp.symbols("V W")
    EMIT = {
        eV: sp.exp(VC),
        eW: sp.exp(WC),
        Uu: sp.Symbol("U"),
        Up_: sp.Symbol("Up"),
        Vp_: sp.Symbol("Vp"),
        Wp_: sp.Symbol("Wp"),
        w0_: sp.Symbol("W0"),
        w0p_: sp.Symbol("W0p"),
    }

    def pyc(e):
        # emit small background matrix/rhs entries (and pairing integrands) as-is; no
        # sp.cancel -- gcd on the full solved forms is the blow-up, and the small
        # entries are already compact.
        return sp.pycode(sp.sympify(e).subs(EMIT))

    from backreaction import paths  # noqa: E402

    gen_path = paths.SYSTEMS / "dk_response.py"
    ARGS = "r, U, Up, V, Vp, W, Wp, B, G, alpha, W0, W0p"
    with open(gen_path, "w") as fh:
        fh.write(
            '"""AUTO-GENERATED by dk_response.py -- do not edit.\n\n'
            "Coupled O(rho^2) lattice response on the D'Hoker-Kraus magnetic brane\n"
            "  ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2+dy^2) + e^{2W} dz^2,  A^3_y=Bx,\n"
            "per harmonic G || x, even sector (y-parity o U(1) charge flip).\n"
            "Algebraic gauge H_tt=H_xx=0: (H_yy,H_zz,a_y) second order, (H_rr,H_xr)\n"
            "algebraic; a_y = transverse abelian mode, b = iG a_y.  Each row:\n"
            "  field'' = coeff_H . (H_yy,H_zz,a_y) + coeff_dH . (H_yy',H_zz',a_y') + src\n"
            "src = w0-bilinear condensate source (W0=w0, W0p=w0').  Complex (a_y and\n"
            "its couplings carry i) -- evaluate with complex arithmetic.\n"
            "The closed-form solved coefficients are ~MB and numerically stiff to\n"
            "emit, so this module instead carries the SMALL background coupled system\n"
            "(_Mmat, _Rmat) and solves the 7x7 numerically per radius (_solve); the\n"
            "row_*/alg_* wrappers return the same (coeff_H, coeff_dH, src) triples.\n"
            "Args: (" + ARGS + ").\n"
            '"""\n\nimport math  # noqa: F401\nimport numpy as np\n\n\n'
        )
        # Compact emission.  Unknowns U = (H_rr, H_xr, H_rr', H_xr', H_yy'', H_zz'',
        # a_y''); the seven equations are {rr, xr, d/dr rr, d/dr xr, yy, zz, My}.  With
        #   M . U = R . (H_yy, H_yy', H_zz, H_zz', a_y, a_y', 1),
        # every response coefficient is a row of X = M^{-1} R, so we emit only the SMALL
        # background M (7x7) and R (7x8) and solve numerically per radius.
        Mmat = [[cof(EQS[k], q) for q in UNK] for k in range(nU)]
        Rmat = [
            [-cof(EQS[k], f) for f in FLDsym] + [-srcpart(EQS[k])] for k in range(nU)
        ]

        def emit_matrix(name, mat, ncol, doc):
            body = ",\n            ".join(
                "[" + ", ".join(pyc(mat[i][j]) for j in range(ncol)) + "]"
                for i in range(nU)
            )
            fh.write(
                f'def {name}({ARGS}):\n    """{doc}"""\n'
                f"    return [\n            {body}]\n\n\n"
            )

        emit_matrix(
            "_Mmat",
            Mmat,
            nU,
            "7x7 coefficient matrix of the unknowns "
            "(H_rr,H_xr,H_rr',H_xr',H_yy'',H_zz'',a_y'').",
        )
        emit_matrix(
            "_Rmat",
            Rmat,
            len(FLDsym) + 1,
            "7x8 RHS map onto (H_yy,H_yy',H_zz,H_zz',a_y,a_y',1).",
        )
        fh.write(
            f"def _solve({ARGS}):\n"
            "    \"\"\"X = M^{-1} R (7x8); X[i] = coeffs of (H_yy,H_yy',H_zz,H_zz',a_y,a_y',1)\n"
            "    in unknown i, order (H_rr,H_xr,H_rr',H_xr',H_yy'',H_zz'',a_y'').\"\"\"\n"
            f"    M = np.array(_Mmat({ARGS}), dtype=complex)\n"
            f"    R = np.array(_Rmat({ARGS}), dtype=complex)\n"
            "    return np.linalg.solve(M, R)\n\n\n"
        )
        for name, ui in (
            ("row_hyy", 4),
            ("row_hzz", 5),
            ("row_ay", 6),
            ("alg_Hrr", 0),
            ("alg_Hxr", 1),
        ):
            fh.write(
                f"def {name}({ARGS}):\n"
                f"    row = _solve({ARGS})[{ui}]\n"
                "    return ([row[0], row[2], row[4]], [row[1], row[3], row[5]], row[6])\n\n\n"
            )
        # pairing integrands
        c4_int = flat(p1_brane * W0f**4)
        fh.write(
            f"def c4_integrand({ARGS}):\n"
            '    """C4 = int p1 w0^4 dr, p1 = e^{W-2V}."""\n'
            f"    return {pyc(c4_int)}\n\n\n"
        )
        fh.write(
            f"def x_integrand({ARGS}, ay):\n"
            '    """X = int p1 bhat w0^2 dr, bhat = iG a_y (transverse)."""\n'
            f"    return {pyc(flat(sp.I * G * p1_brane * W0f**2))} * ay\n\n\n"
        )
        pig = flat(pi_grav)
        for a, b in [
            (yy0, sp.Symbol("hyy")),
            (zz0, sp.Symbol("hzz")),
            (a0f, sp.Symbol("ay")),
            (rr0, sp.Symbol("hrr")),
            (xr0, sp.Symbol("hxr")),
        ]:
            pig = pig.subs(a, b)
        fh.write(
            "def pi_grav(" + ARGS + ", hyy, hzz, ay, hrr, hxr):\n"
            '    """Graviton pairing density sqrt(-g) S^{MN} h_MN (even sector);\n'
            '    graviton exchange = 1/2 int pi_grav dr (plug solved fields)."""\n'
            f"    return {sp.pycode(sp.cancel(pig).subs(EMIT))}\n"
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
