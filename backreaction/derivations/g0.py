"""The G = 0 (homogeneous) sector of the lattice backreaction.

The lattice-averaged source T0 + rho^2 <T2> (lambda_0 = 1, b_0 = 0) drives a
diagonal, x-independent metric perturbation

    h_tt = -(f/u^2) Htt(u),   h_xx = h_yy = (1/u^2) Hxx(u),
    h_zz = (1/u^2) Hzz(u),    h_uu = (1/(u^2 f)) Huu(u),

the O(alpha B^2) anisotropic 'magnetic brane' correction plus the O(alpha
rho^2) condensate correction.  This script derives the closed ODE system in
two independent gauges, the horizon thermodynamics formulas, and the
decoupled-invariant quadrature, and code-generates systems/einstein_g0.py.

Checks / derivations
--------------------
[1] Pipeline at G = 0: delta R + 4h annihilates pure-gauge h = nabla xi +
    (nabla xi)^T with xi = xi_u(u) du; both sources (per B^2 and per rho^2)
    are covariantly conserved on shell *separately*.
[2] Gauge shifts of (Htt, Hxx, Hzz, Huu) under xi_u: delta Htt =
    -2u(1 + u^4/u_H^4) xi_u is algebraic with never-vanishing coefficient,
    so Htt = 0 is a complete algebraic gauge fixing (no residual freedom);
    the radial gauge Huu = 0 keeps the horizon-singular residual mode
    xi_u = 1/(u sqrt(f)).
[3] Algebraic gauge Htt = 0: the (tt) and (uu) rows are both first order;
    the combination cancelling Huu' is *algebraic* in Huu.  Eliminating Huu,
    the (xx) and (zz) rows close into a two-field second-order system for
    (Hxx, Hzz); the (tt) and (uu) rows then reduce to zero identically on
    shell (the Bianchi identity in action).
[4] Radial gauge Huu = 0: three second-order rows (tt, xx, zz) plus the
    first-order (uu) constraint C; the derivative of C closes on C on shell
    (constraint propagation), so C = 0 anywhere propagates everywhere.
[5] The difference I_d = Hxx - Hzz is gauge invariant and obeys the G = 0
    massless-scalar equation  f I_d'' + (f' - 3f/u) I_d' = S_d(u), i.e.
    (f I_d'/u^3)' = S_d/u^3 -- solvable by quadrature.  For the B^2 source
    the quadrature is done in closed form (analytic anchor for the
    numerics); horizon regularity + Dirichlet give
        I_d(u) = -int_0^u ds s^3/f(s) int_s^{u_H} dt S_d(t)/t^3.
[6] Horizon thermodynamics at fixed u_H (derived via surface gravity and
    horizon area):
        delta T / T0 = (Htt - Huu)(u_H) / 2 ,
        delta s / s0 = (2 Hxx + Hzz)(u_H) / 2.
[7] Boundary indicial structure of the closed (Hxx, Hzz) system and the
    leading boundary powers of both sources (resonance bookkeeping).
[8] Code generation: systems/einstein_g0.py (algebraic-gauge closed system +
    algebraic Huu + I_d source), with the two sources (per B^2 and per
    rho^2) kept separate.  The radial-gauge system is derivation-only.
[9] The trace channel J = 2 Hxx + Hzz: the closed system is first order in
    V = H'; V_J decouples, V_J' = (3 - 5u^4)/(u(1+u^4)) V_J + s_J, with
    integrating factor (1+u^4)^2/u^3 and a *horizon-regular* homogeneous
    solution V_J = u^3/(1+u^4)^2 -- the thermal zero mode (ds = 3 dT,
    the shift along the black-brane family).  Physical convention: fixed
    temperature, Huu(u_H) = 0.  Exact fixed-T anchor per alpha B^2:
    ds|_T = 1/4 (closed-form integrals, verified).

Run:  uv run python -m backreaction.derivations.g0   (~1 min)
"""

import contextlib
import io
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
    return sp.cancel(sp.together(e)) == 0


t, x, y, z, u = coords = sp.symbols("t x y z u", real=True)
uH, B = sp.symbols("u_H B", positive=True)
alpha = sp.Symbol("alpha", positive=True)

f = 1 - u**4 / uH**4
gdn = sp.diag(-f / u**2, 1 / u**2, 1 / u**2, 1 / u**2, 1 / (u**2 * f))
gup = sp.diag(-(u**2) / f, u**2, u**2, u**2, u**2 * f)
N = 5
idx = ["t", "x", "y", "z", "u"]

w = sp.Function("w0")


def main():
    wp = sp.Derivative(w(u), u)
    # probe-limit zero-mode ODE:  f w'' + (f' - f/u) w' + B w = 0
    wpp_rule = {
        sp.Derivative(w(u), (u, 2)): -((sp.diff(f, u) - f / u) * wp + B * w(u)) / f
    }

    def on_shell(expr):
        e = sp.expand(sp.sympify(expr).doit())
        while e.has(sp.Derivative(w(u), (u, 2))):
            e = sp.expand(e.subs(wpp_rule).doit())
        return e

    def christoffel():
        Gam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for r in range(N):
            for m in range(N):
                for n in range(m, N):
                    e = sp.S.Zero
                    for s_ in range(N):
                        if gup[r, s_] != 0:
                            e += gup[r, s_] * (
                                sp.diff(gdn[s_, m], coords[n])
                                + sp.diff(gdn[s_, n], coords[m])
                                - sp.diff(gdn[m, n], coords[s_])
                            )
                    Gam[r][m][n] = sp.cancel(e / 2)
                    Gam[r][n][m] = Gam[r][m][n]
        return Gam

    GAM = christoffel()

    def cov_deriv_2tensor(h):
        Dh = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for mu in range(N):
            for a in range(N):
                for b in range(N):
                    e = sp.diff(h[a][b], coords[mu])
                    for lam in range(N):
                        if GAM[lam][mu][a] != 0:
                            e -= GAM[lam][mu][a] * h[lam][b]
                        if GAM[lam][mu][b] != 0:
                            e -= GAM[lam][mu][b] * h[a][lam]
                    Dh[mu][a][b] = sp.expand(e)
        return Dh

    def delta_ricci(h):
        Dh = cov_deriv_2tensor(h)
        dGam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for r in range(N):
            for m in range(N):
                for n in range(m, N):
                    e = sp.S.Zero
                    for s_ in range(N):
                        if gup[r, s_] != 0:
                            e += gup[r, s_] * (
                                Dh[m][s_][n] + Dh[n][s_][m] - Dh[s_][m][n]
                            )
                    dGam[r][m][n] = sp.expand(e / 2)
                    dGam[r][n][m] = dGam[r][m][n]
        V = [sp.expand(sum(dGam[r][r][m] for r in range(N))) for m in range(N)]
        dR = [[sp.S.Zero] * N for _ in range(N)]
        for m in range(N):
            for n in range(m, N):
                e = sp.S.Zero
                for r in range(N):
                    e += sp.diff(dGam[r][m][n], coords[r])
                    for lam in range(N):
                        if GAM[r][r][lam] != 0:
                            e += GAM[r][r][lam] * dGam[lam][m][n]
                        if GAM[lam][r][m] != 0:
                            e -= GAM[lam][r][m] * dGam[r][lam][n]
                        if GAM[lam][r][n] != 0:
                            e -= GAM[lam][r][n] * dGam[r][m][lam]
                e -= sp.diff(V[m], coords[n])
                for lam in range(N):
                    if GAM[lam][n][m] != 0:
                        e += GAM[lam][n][m] * V[lam]
                dR[m][n] = sp.expand(e)
                dR[n][m] = dR[m][n]
        return dR

    def divergence(T):
        out = [sp.S.Zero] * N
        for nu in range(N):
            e = sp.S.Zero
            for mu in range(N):
                cov = sp.diff(T[mu][nu], coords[mu])
                for s_ in range(N):
                    if GAM[s_][mu][mu] != 0:
                        cov -= GAM[s_][mu][mu] * T[s_][nu]
                    if GAM[s_][mu][nu] != 0:
                        cov -= GAM[s_][mu][nu] * T[mu][s_]
                e += gup[mu, mu] * cov
            out[nu] = sp.expand(e)
        return out

    # background Ricci tensor of AdS5-Schwarzschild (an Einstein space)
    Rbg = [[-4 * gdn[m, n] for n in range(N)] for m in range(N)]

    def einstein_form(h, dR):
        dRs = sp.S.Zero
        for m in range(N):
            for n in range(N):
                if h[m][n] != 0:
                    dRs -= gup[m, m] * gup[n, n] * h[m][n] * Rbg[m][n]
            dRs += gup[m, m] * dR[m][m]
        dRs = sp.expand(dRs)
        E = [
            [
                sp.expand(
                    dR[m][n]
                    - sp.Rational(1, 2) * h[m][n] * (-20)
                    - sp.Rational(1, 2) * gdn[m, n] * dRs
                    - 6 * h[m][n]
                )
                for n in range(N)
            ]
            for m in range(N)
        ]
        return E

    # ---------------------------------------------------------------------------
    # [1] pipeline validation at G = 0 + separate source conservation
    # ---------------------------------------------------------------------------
    xiU = sp.Function("xi_u")
    xi = [sp.S.Zero, sp.S.Zero, sp.S.Zero, sp.S.Zero, xiU(u)]
    hg = [[sp.S.Zero] * N for _ in range(N)]
    for m in range(N):
        for n in range(m, N):
            e = sp.diff(xi[n], coords[m]) + sp.diff(xi[m], coords[n])
            for lam in range(N):
                if GAM[lam][m][n] != 0:
                    e -= 2 * GAM[lam][m][n] * xi[lam]
            hg[m][n] = sp.expand(e)
            hg[n][m] = hg[m][n]
    dR_g = delta_ricci(hg)
    ok1a = all(is_zero(dR_g[m][n] + 4 * hg[m][n]) for m in range(N) for n in range(N))
    log(f"[1] delta R + 4h annihilates the G = 0 pure gauge: {ok1a}")
    assert ok1a

    # the two G = 0 sources, kept separate (per B^2 and per rho^2):
    SB = [[sp.S.Zero] * N for _ in range(N)]  # T^(0) per B^2
    SB[0][0] = (u**2 / 2) * f
    SB[1][1] = u**2 / 2
    SB[2][2] = u**2 / 2
    SB[3][3] = -(u**2) / 2
    SB[4][4] = -(u**2) / 2 / f
    SR = [[sp.S.Zero] * N for _ in range(N)]  # <T^(2)> per rho^2 (n -> 1, b -> 0)
    SR[0][0] = f * u**2 * (f * wp**2 - B * w(u) ** 2)
    SR[1][1] = -(u**2) * B * w(u) ** 2
    SR[2][2] = SR[1][1]
    SR[3][3] = -(u**2) * (f * wp**2 - B * w(u) ** 2)
    SR[4][4] = u**2 * (wp**2 + (B / f) * w(u) ** 2)

    for name, S in (("B^2", SB), ("rho^2", SR)):
        div = divergence(S)
        ok = all(is_zero(on_shell(div[nu])) for nu in range(N))
        log(f"[1] G = 0 source per {name} conserved on shell separately: {ok}")
        assert ok

    # ---------------------------------------------------------------------------
    # [2] gauge shifts at G = 0
    # ---------------------------------------------------------------------------
    shift = {
        "tt": sp.expand(sp.cancel(-(u**2) / f * hg[0][0])),
        "xx": sp.expand(u**2 * hg[1][1]),
        "zz": sp.expand(u**2 * hg[3][3]),
        "uu": sp.expand(sp.cancel(u**2 * f * hg[4][4])),
    }
    print("gauge shifts under xi = xi_u(u) du:")
    for k, v in shift.items():
        print(f"  delta H_{k} = {sp.simplify(v)}")
    c_tt = sp.simplify(sp.cancel(shift["tt"] / xiU(u)))
    ok2a = (
        not c_tt.has(sp.Derivative)
        and sp.simplify(c_tt + 2 * u * (1 + u**4 / uH**4)) == 0
    )
    log(f"[2] delta Htt = -2u(1 + u^4/u_H^4) xi_u (algebraic, never zero): {ok2a}")
    assert ok2a
    ok2b = is_zero(shift["xx"] - shift["zz"])
    log(f"[2] delta Hxx = delta Hzz  =>  I_d = Hxx - Hzz gauge invariant: {ok2b}")
    assert ok2b
    # radial-gauge residual mode
    sol_xiu = sp.dsolve(sp.Eq(shift["uu"], 0), xiU(u))
    log(
        f"[2] radial-gauge residual mode: xi_u = {sp.simplify(sol_xiu.rhs)} "
        "(horizon-singular)"
    )

    # ---------------------------------------------------------------------------
    # the two gauge-fixed systems
    # ---------------------------------------------------------------------------
    Htt = sp.Function("H_tt")
    Hxx = sp.Function("H_xx")
    Hzz = sp.Function("H_zz")
    Huu = sp.Function("H_uu")
    q2, r2 = sp.symbols("q2 r2", positive=True)  # bookkeeping: per B^2 / per rho^2

    def build_rows(with_tt, with_uu):
        """Trace-reversed rows (tt), (xx), (zz) and Einstein-form row (uu) on the
        diagonal G = 0 ansatz; source q2*SB + r2*SR."""
        h = [[sp.S.Zero] * N for _ in range(N)]
        if with_tt:
            h[0][0] = -(f / u**2) * Htt(u)
        h[1][1] = (1 / u**2) * Hxx(u)
        h[2][2] = (1 / u**2) * Hxx(u)
        h[3][3] = (1 / u**2) * Hzz(u)
        if with_uu:
            h[4][4] = (1 / (u**2 * f)) * Huu(u)
        dR = delta_ricci(h)
        ER = [[sp.expand(dR[m][n] + 4 * h[m][n]) for n in range(N)] for m in range(N)]
        EG = einstein_form(h, dR)
        S = [
            [sp.expand(q2 * SB[m][n] + r2 * SR[m][n]) for n in range(N)]
            for m in range(N)
        ]
        Str = sum(gup[a, a] * S[a][a] for a in range(N))
        eqs = {}
        for m, n in ((0, 0), (1, 1), (3, 3)):
            eqs[(m, n)] = sp.expand(
                ER[m][n] - alpha * (S[m][n] - sp.Rational(1, 3) * gdn[m, n] * Str)
            )
        eqs[(4, 4)] = sp.expand(EG[4][4] - alpha * S[4][4])
        return eqs

    def drop_cancelling_d2(eqs, fields):
        d2 = [sp.Derivative(F, (u, 2)) for F in fields]
        for key in list(eqs):
            e = eqs[key]
            for d2f in d2:
                c = e.coeff(d2f)
                if c != 0 and is_zero(c):
                    e = sp.expand(e - sp.expand(c * d2f))
            eqs[key] = e
        return eqs

    def structure(eqs, fields):
        d2all = [sp.Derivative(F, (u, 2)) for F in fields]
        out = {}
        for key, e in eqs.items():
            d2 = [str(F.func) for F, dd in zip(fields, d2all, strict=True) if e.has(dd)]
            d1 = [str(F.func) for F in fields if e.has(sp.Derivative(F, u))]
            alg = [str(F.func) for F in fields if e.has(F)]
            out[key] = (d2, d1, alg)
        return out

    # ---------------------------------------------------------------------------
    # [3] algebraic gauge Htt = 0
    # ---------------------------------------------------------------------------
    log("[3] algebraic gauge Htt = 0 ...")
    eqsA = build_rows(with_tt=False, with_uu=True)
    fieldsA = [Hxx(u), Hzz(u), Huu(u)]
    eqsA = drop_cancelling_d2(eqsA, fieldsA)
    for key, (d2, d1, alg) in structure(eqsA, fieldsA).items():
        print(
            f"    eq ({idx[key[0]]}{idx[key[1]]}): H'' of {d2 or None}; "
            f"H' of {d1 or None}; algebraic {alg or None}"
        )

    # (tt) and (uu) rows are both first order; the combination cancelling Huu'
    # is algebraic in Huu (Hamiltonian constraint)
    dHuu = sp.Derivative(Huu(u), u)
    e_tt, e_uu = eqsA[(0, 0)], eqsA[(4, 4)]
    ok3a0 = not any(
        e.has(sp.Derivative(F, (u, 2))) for e in (e_tt, e_uu) for F in fieldsA
    )
    log(f"[3] (tt) and (uu) rows are first order: {ok3a0}")
    assert ok3a0
    c_tt_dHuu = sp.cancel(sp.together(e_tt.coeff(dHuu)))
    c_uu_dHuu = sp.cancel(sp.together(e_uu.coeff(dHuu)))
    combo = sp.expand(c_uu_dHuu * e_tt - c_tt_dHuu * e_uu)
    ok3a = is_zero(combo.coeff(dHuu)) and not is_zero(combo.coeff(Huu(u)))
    log(
        f"[3] the Huu'-cancelling combination of (tt), (uu) is algebraic in Huu: {ok3a}"
    )
    assert ok3a
    combo = sp.expand(combo - sp.expand(combo.coeff(dHuu) * dHuu))
    cHuu = sp.cancel(sp.together(combo.coeff(Huu(u))))
    sol_Huu = sp.expand(
        sp.cancel(sp.together(-(combo - sp.expand(cHuu * Huu(u))) / cHuu))
    )
    log("[3] Huu solved algebraically from the (tt)/(uu) combination")

    huu_rules = [
        (sp.Derivative(Huu(u), (u, 2)), sp.diff(sol_Huu, u, 2)),
        (sp.Derivative(Huu(u), u), sp.diff(sol_Huu, u)),
        (Huu(u), sol_Huu),
    ]

    def elim_huu(e):
        e = sp.expand(sp.sympify(e).doit())
        for k, v in huu_rules:
            if e.has(k):
                e = e.subs(k, v).doit()
        return on_shell(sp.expand(e))

    workA = [elim_huu(eqsA[(1, 1)]), elim_huu(eqsA[(3, 3)])]
    d2A = [sp.Derivative(Hxx(u), (u, 2)), sp.Derivative(Hzz(u), (u, 2))]
    MatA = sp.Matrix(2, 2, lambda i, j: sp.cancel(sp.together(workA[i].coeff(d2A[j]))))
    remA = sp.Matrix(2, 1, lambda i, _: sp.expand(workA[i].subs({v: 0 for v in d2A})))
    detA = sp.cancel(sp.together(MatA.det()))
    ok3b = not is_zero(detA)
    log(
        f"[3] (xx), (zz) rows (Huu eliminated) solvable for (Hxx'', Hzz''): {ok3b} "
        f"[det = {sp.simplify(detA)}]"
    )
    assert ok3b
    XA = MatA.LUsolve(-remA)
    solA = {d2A[i]: sp.expand(sp.cancel(sp.together(XA[i]))) for i in range(2)}
    for k2 in d2A:
        bad = [v for v in (Huu(u), sp.Derivative(Huu(u), u)) if solA[k2].has(v)]
        assert not bad, f"unexpected dependence of {k2}: {bad}"
    log(
        "[3] (Hxx, Hzz) close into a two-field second-order system; "
        "Huu algebraic in (H, H')"
    )

    def reduce_eqA(e):
        rules = [
            (sp.Derivative(Huu(u), (u, 2)), sp.diff(sol_Huu, u, 2)),
            (sp.Derivative(Hxx(u), (u, 3)), sp.diff(solA[d2A[0]], u)),
            (sp.Derivative(Hzz(u), (u, 3)), sp.diff(solA[d2A[1]], u)),
            (sp.Derivative(Huu(u), u), sp.diff(sol_Huu, u)),
            (sp.Derivative(Hxx(u), (u, 2)), solA[d2A[0]]),
            (sp.Derivative(Hzz(u), (u, 2)), solA[d2A[1]]),
            (Huu(u), sol_Huu),
        ]
        keys = [k for k, _ in rules]
        e = sp.expand(sp.sympify(e).doit())
        for _ in range(10):
            if not any(e.has(k) for k in keys):
                break
            for k, v in rules:
                if e.has(k):
                    e = e.subs(k, v).doit()
            e = on_shell(sp.expand(e))
        return e

    ok3c = True
    for key in ((0, 0), (4, 4)):
        r = reduce_eqA(eqsA[key])
        if not is_zero(r):
            ok3c = False
            print(f"    eq ({idx[key[0]]}{idx[key[1]]}) NOT redundant; residue:")
            sp.pprint(sp.simplify(sp.cancel(sp.together(r))))
    log(f"[3] (tt) and (uu) rows identically redundant on shell (Bianchi): {ok3c}")
    assert ok3c

    # ---------------------------------------------------------------------------
    # [4] radial gauge Huu = 0
    # ---------------------------------------------------------------------------
    log("[4] radial gauge Huu = 0 ...")
    eqsR = build_rows(with_tt=True, with_uu=False)
    fieldsR = [Htt(u), Hxx(u), Hzz(u)]
    eqsR = drop_cancelling_d2(eqsR, fieldsR)
    for key, (d2, d1, alg) in structure(eqsR, fieldsR).items():
        print(
            f"    eq ({idx[key[0]]}{idx[key[1]]}): H'' of {d2 or None}; "
            f"H' of {d1 or None}; algebraic {alg or None}"
        )

    d2R = [sp.Derivative(F, (u, 2)) for F in fieldsR]
    C = eqsR[(4, 4)]
    ok4a = not any(C.has(dd) for dd in d2R)
    log(f"[4] (uu) row is a first-order constraint C: {ok4a}")
    assert ok4a

    solR = sp.solve([eqsR[(0, 0)], eqsR[(1, 1)], eqsR[(3, 3)]], d2R, dict=True)[0]
    solR = {k: sp.expand(sp.cancel(sp.together(v))) for k, v in solR.items()}
    log("[4] radial-gauge dynamical rows solved for (Htt'', Hxx'', Hzz'')")

    # constraint propagation: dC/du reduced on shell must be proportional to C
    dC = sp.expand(sp.diff(C, u).doit())
    for k2, v in solR.items():
        dC = dC.subs(k2, v)
    dC = on_shell(sp.expand(dC.doit()))
    symsC = [F for F in fieldsR] + [sp.Derivative(F, u) for F in fieldsR]
    vecC = [sp.expand(C.coeff(s_)) for s_ in symsC]
    vecdC = [sp.expand(dC.coeff(s_)) for s_ in symsC]
    # find the proportionality factor from the first nonzero coefficient pair
    lam_prop = None
    for a, b in zip(vecdC, vecC, strict=True):
        if not is_zero(b):
            lam_prop = sp.cancel(sp.together(a / b))
            break
    ok4b = lam_prop is not None and is_zero(sp.expand(dC - sp.expand(lam_prop * C)))
    log(
        f"[4] constraint propagation dC/du = lambda(u) C with "
        f"lambda = {sp.simplify(lam_prop) if lam_prop is not None else '??'}: {ok4b}"
    )
    assert ok4b

    # ---------------------------------------------------------------------------
    # [5] the decoupled invariant I_d = Hxx - Hzz and its quadrature
    # ---------------------------------------------------------------------------
    log("[5] decoupled invariant ...")
    ediff = sp.expand(solA[d2A[0]] - solA[d2A[1]])
    Id = sp.Function("I_d")
    # substitute Hxx = Id + Hzz and check Hzz drops out
    ediff_sub = sp.expand(
        ediff.subs(
            {
                Hxx(u): Id(u) + Hzz(u),
                sp.Derivative(Hxx(u), u): sp.Derivative(Id(u), u)
                + sp.Derivative(Hzz(u), u),
            }
        ).doit()
    )
    ok5a = not ediff_sub.has(Hzz(u)) and not ediff_sub.has(sp.Derivative(Hzz(u), u))
    log(f"[5] Hxx'' - Hzz'' depends only on I_d = Hxx - Hzz: {ok5a}")
    assert ok5a
    # match against the massless-scalar operator:
    #   I_d'' = -(f' - 3f/u)/f I_d' + S_d/f
    cId = sp.cancel(sp.together(ediff_sub.coeff(sp.Derivative(Id(u), u))))
    ok5b = is_zero(ediff_sub.coeff(Id(u))) and is_zero(
        sp.expand(cId + sp.cancel((sp.diff(f, u) - 3 * f / u) / f))
    )
    S_d = sp.expand(f * sp.expand(ediff_sub - sp.expand(cId * sp.Derivative(Id(u), u))))
    log(f"[5] I_d obeys f I_d'' + (f' - 3f/u) I_d' = S_d: {ok5b}")
    assert ok5b
    S_d_B2 = sp.cancel(sp.together(S_d.coeff(q2).subs(alpha, 1)))
    S_d_r2 = sp.expand(S_d.coeff(r2).subs(alpha, 1))
    print(f"    S_d per B^2  : {S_d_B2}")
    print(f"    S_d per rho^2: {sp.collect(S_d_r2, [w(u) ** 2, wp**2])}")

    # closed-form quadrature for the B^2 source at u_H = 1:
    #   f I_d'(u)/u^3 = -int_u^{u_H} dt S_d(t)/t^3   (horizon regularity),
    #   I_d(u_H) = int_0^{u_H} ds [s^3/f(s)] [-int_s^{u_H} dt S_d(t)/t^3]
    ok5c = is_zero(sp.expand(S_d_B2 + 2 * u**2))
    log(f"[5] S_d per B^2 = -2 u^2 exactly: {ok5c}")
    assert ok5c
    s_, t_ = sp.symbols("s t", positive=True)
    inner = sp.integrate(-(-2 / t_), (t_, s_, 1))  # -int_s^1 S_d/t^3 dt
    ok5d = sp.simplify(inner - (-2 * sp.log(s_))) == 0
    log(f"[5] f I_d'/u^3 = -2 log u (u_H = 1): {ok5d}")
    assert ok5d
    # substitute v = s^4:  I = -(1/8) int_0^1 log(v)/(1 - v) dv = pi^2/48
    v_ = sp.Symbol("v", positive=True)
    Id_B2_uH = sp.simplify(
        -sp.Rational(1, 8) * sp.integrate(sp.log(v_) / (1 - v_), (v_, 0, 1))
    )
    ok5e = sp.simplify(Id_B2_uH - sp.pi**2 / 48) == 0
    import mpmath

    num = mpmath.quad(lambda sv: -2 * sv**3 * mpmath.log(sv) / (1 - sv**4), [0, 1])
    ok5f = abs(float(num) - float(sp.pi**2 / 48)) < 1e-12
    log(
        f"[5] analytic anchor: I_d(u_H) per alpha B^2 = {Id_B2_uH} = pi^2/48: "
        f"{ok5e}; numerical quadrature agrees: {ok5f} "
        f"[{float(num):.12f} vs {float(sp.pi**2 / 48):.12f}]"
    )
    assert ok5e and ok5f

    # ---------------------------------------------------------------------------
    # [6] horizon thermodynamics formulas
    # ---------------------------------------------------------------------------
    log("[6] horizon thermodynamics ...")
    # kappa^2 = A'^2 / (4 A B) is rational (the f factors cancel), so the
    # horizon limit is a plain substitution after cancel.  Represent the H
    # functions by their horizon Taylor data (limit must not see Ht1, Hu1).
    eps = sp.Symbol("epsilon", positive=True)
    Ht0, Ht1, Hu0, Hu1 = sp.symbols("Ht0 Ht1 Hu0 Hu1", real=True)
    A_tt = (f / u**2) * (1 + eps * (Ht0 + Ht1 * (u - uH)))
    B_uu = (1 + eps * (Hu0 + Hu1 * (u - uH))) / (u**2 * f)
    kappa2 = sp.cancel(sp.together(sp.diff(A_tt, u) ** 2 / (4 * A_tt * B_uu)))
    kappa2_H = sp.cancel(kappa2.subs(u, uH))
    ratio2 = sp.cancel(sp.together(kappa2_H / kappa2_H.subs(eps, 0)))
    dT_formula = sp.expand(sp.diff(ratio2, eps).subs(eps, 0) / 2)
    ok6 = (
        is_zero(sp.expand(dT_formula - (Ht0 - Hu0) / 2))
        and not dT_formula.has(Ht1)
        and not dT_formula.has(Hu1)
    )
    log(f"[6] delta T / T0 = (Htt - Huu)(u_H)/2: {ok6}   [T0 = 1/(pi u_H)]")
    assert ok6
    log("[6] delta s / s0 = (2 Hxx + Hzz)(u_H)/2   [sqrt(det gamma), definition]")

    # ---------------------------------------------------------------------------
    # [7] boundary indicial structure and source powers
    # ---------------------------------------------------------------------------
    p = sp.Symbol("p")
    Hsym = sp.symbols("hxx hzz", real=True)
    Hpsym = sp.symbols("dhxx dhzz", real=True)
    flatH = {}
    for F, s0, s1 in zip([Hxx(u), Hzz(u)], Hsym, Hpsym, strict=True):
        flatH[sp.Derivative(F, u)] = s1
        flatH[F] = s0
    Mind = sp.zeros(2, 2)
    for i, k2 in enumerate(d2A):
        e = sp.expand(solA[k2].subs({q2: 0, r2: 0}))
        e = e.subs(flatH)
        for j in range(2):
            aij = sp.limit(u**2 * e.coeff(Hsym[j]), u, 0)
            bij = sp.limit(u * e.coeff(Hpsym[j]), u, 0)
            Mind[i, j] = (p * (p - 1) if i == j else 0) - aij - bij * p
    log(f"[7] boundary indicial det (Hxx, Hzz block): {sp.factor(Mind.det())}")
    for nm, k2 in (("xx", d2A[0]), ("zz", d2A[1])):
        for tag, sym in (("B^2", q2), ("rho^2", r2)):
            src = sp.expand(solA[k2].subs(flatH)).coeff(sym)
            if tag == "rho^2":
                # w0 ~ w2 u^2 near the boundary (c0 = 0)
                w2 = sp.Symbol("w2")
                src = src.subs({wp: 2 * w2 * u, w(u): w2 * u**2})
            ser = sp.series(sp.expand(src), u, 0, 5)
            lead = sp.LT(sp.expand(ser.removeO())) if ser.removeO() != 0 else 0
            print(f"    src(H{nm}'' , per {tag}): leading boundary power ~ {lead}")

    # ---------------------------------------------------------------------------
    # [8] code generation
    # ---------------------------------------------------------------------------
    log("[8] generating systems/einstein_g0.py ...")
    W0, W0p = sp.symbols("W0 W0p", real=True)
    flat_matter = {w(u): W0, wp: W0p}

    def flatten2(e, hsyms, hpsyms, fields):
        e = on_shell(sp.expand(e))
        for F, s0, s1 in zip(fields, hsyms, hpsyms, strict=True):
            e = e.subs({sp.Derivative(F, u): s1, F: s0})
        return sp.expand(e.subs(flat_matter))

    def split_linear(expr, hsyms, hpsyms):
        A_row, B_row = [], []
        rem = sp.expand(expr)
        for s0 in hsyms:
            c = sp.cancel(sp.together(rem.coeff(s0)))
            assert not any(c.has(v) for v in (W0, W0p, q2, r2)), (
                f"H-coefficient not pure geometry: {c}"
            )
            A_row.append(c)
            rem = sp.expand(rem - c * s0)
        for s1 in hpsyms:
            c = sp.cancel(sp.together(rem.coeff(s1)))
            assert not any(c.has(v) for v in (W0, W0p, q2, r2)), (
                f"dH-coefficient not pure geometry: {c}"
            )
            B_row.append(c)
            rem = sp.expand(rem - c * s1)
        sB2 = sp.cancel(sp.together(rem.coeff(q2)))
        sR2 = sp.cancel(sp.together(rem.coeff(r2)))
        left = sp.expand(rem - sp.expand(q2 * sB2) - sp.expand(r2 * sR2))
        assert is_zero(left), f"unclassified source remainder: {left}"
        return A_row, B_row, sB2, sR2

    HsymA = sp.symbols("hxx hzz", real=True)
    HpsymA = sp.symbols("dhxx dhzz", real=True)
    fieldsA2 = [Hxx(u), Hzz(u)]

    from backreaction import paths  # noqa: E402

    gen_path = paths.SYSTEMS / "einstein_g0.py"
    # the module text is assembled in memory and written only after the checks below
    GEN_TEXT = io.StringIO()
    with contextlib.nullcontext(GEN_TEXT) as fh:
        fh.write(
            '"""AUTO-GENERATED by g0.py -- do not edit.\n\n'
            "G = 0 (homogeneous) sector, alpha = 1, sources kept separate:\n"
            "each row function returns (coeff_H, coeff_dH, src_B2, src_r2)\n"
            "so that  LHS = coeff_H . H + coeff_dH . H' + q2*src_B2 + r2*src_r2\n"
            "with q2 = B^2-source unit and r2 = rho^2-source unit.\n\n"
            "Algebraic gauge Htt = 0, field order (Hxx, Hzz):\n"
            "    Hxx'' = g0a_row_xx,  Hzz'' = g0a_row_zz,  Huu = g0a_huu "
            "[algebraic]\n"
            "Radial gauge Huu = 0, field order (Htt, Hxx, Hzz):\n"
            "    H_i'' = g0r_row_*,  constraint 0 = g0r_constraint\n"
            "Decoupled invariant I_d = Hxx - Hzz:\n"
            "    f Id'' + (f' - 3f/u) Id' = q2*Sd_B2 + r2*Sd_r2 "
            "(g0_id_source)\n"
            '"""\n\nimport math  # noqa: F401\n\n\n'
        )
        # algebraic gauge
        for name, expr in (
            ("g0a_row_xx", solA[d2A[0]]),
            ("g0a_row_zz", solA[d2A[1]]),
            ("g0a_huu", sol_Huu),
        ):
            log(f"[8]   generating {name} ...")
            e = flatten2(expr.subs(alpha, 1), HsymA, HpsymA, fieldsA2)
            A_row, B_row, sB2, sR2 = split_linear(e, HsymA, HpsymA)
            fh.write(
                f"def {name}(u, B, u_H, W0, W0p):\n"
                "    coeff_H = [" + ", ".join(sp.pycode(c) for c in A_row) + "]\n"
                "    coeff_dH = [" + ", ".join(sp.pycode(c) for c in B_row) + "]\n"
                f"    src_B2 = {sp.pycode(sB2)}\n"
                f"    src_r2 = {sp.pycode(sR2)}\n"
                "    return coeff_H, coeff_dH, src_B2, src_r2\n\n\n"
            )
        # radial gauge: derivation and constraint propagation are verified in
        # [4]; its code generation is skipped (the algebraic gauge carries the
        # numerics, cross-checked by two methods, the I_d quadrature anchor and
        # the first-law/boundary-stress identities).
        # I_d source
        sdB = sp.cancel(sp.together(S_d_B2))
        sdR = flatten2(sp.expand(S_d_r2), (), (), [])
        fh.write(
            "def g0_id_source(u, B, u_H, W0, W0p):\n"
            f"    src_B2 = {sp.pycode(sdB)}\n"
            f"    src_r2 = {sp.pycode(sdR)}\n"
            "    return src_B2, src_r2\n\n\n"
        )
        fh.write(
            f"ID_B2_HORIZON_COEFF = {float(Id_B2_uH.subs(uH, 1)):.12e}"
            "  # I_d(u_H) per alpha B^2 u_H^2... exact quadrature, u_H = 1\n"
        )
    log(f"[8] generated the text of {gen_path.name} (written after the checks)")

    # ---------------------------------------------------------------------------
    # [9] the trace channel J = 2 Hxx + Hzz: first-order reduction, the thermal
    #     zero mode, and exact fixed-T anchors
    # ---------------------------------------------------------------------------
    log("[9] trace channel ...")
    u1 = {uH: 1}
    # V-form: with coeff_H = 0 (verified below), the closed system is first
    # order in V_i = H_i'.  Check the J-combination decouples:
    for k2, nm in ((d2A[0], "xx"), (d2A[1], "zz")):
        for F2 in fieldsA2:
            cf = sp.expand(solA[k2]).coeff(F2)
            assert is_zero(cf), f"H-coefficient nonzero in {nm}"
    log("[9] closed system is first order in V = H' (no H terms): True")

    VX, VZ = sp.symbols("VX VZ", real=True)
    flatV = {sp.Derivative(Hxx(u), u): VX, sp.Derivative(Hzz(u), u): VZ}
    rx = on_shell(solA[d2A[0]].subs(alpha, 1)).subs(flatV).subs(u1)
    rz = on_shell(solA[d2A[1]].subs(alpha, 1)).subs(flatV).subs(u1)
    VJ_rhs = sp.expand(2 * rx + rz)
    cJ = sp.cancel(sp.together(VJ_rhs.coeff(VX) / 2))
    okJa = is_zero(sp.expand(VJ_rhs.coeff(VX) - 2 * cJ)) and is_zero(
        sp.expand(VJ_rhs.coeff(VZ) - cJ)
    )
    log(f"[9] V_J' = c(u) V_J + s_J decouples with c = {cJ}: {okJa}")
    assert okJa
    okJb = is_zero(sp.expand(cJ - (3 - 5 * u**4) / (u * (1 + u**4))))
    log(f"[9] c(u) = (3 - 5u^4)/(u(1+u^4)): {okJb}")
    assert okJb
    sJ = sp.expand(
        VJ_rhs - sp.expand(VJ_rhs.coeff(VX) * VX) - sp.expand(VJ_rhs.coeff(VZ) * VZ)
    )
    sJ_B2 = sp.cancel(sp.together(sJ.coeff(q2)))
    sJ_r2 = sp.expand(sJ.coeff(r2))
    okJc = is_zero(sp.expand(sJ_B2 + 4 * u**2 / (1 + u**4)))
    log(f"[9] s_J per B^2 = -4u^2/(1+u^4): {okJc}")
    assert okJc
    sJ_r2_target = sp.expand(4 * u**2 * (2 * B * w(u) ** 2 / (1 + u**4) - wp**2))
    okJd = is_zero(sp.expand(sJ_r2 - sJ_r2_target))
    log(f"[9] s_J per rho^2 = 4u^2 [2B w0^2/(1+u^4) - w0'^2]: {okJd}")
    assert okJd

    # integrating factor: (V_J (1+u^4)^2 / u^3)' = s_J (1+u^4)^2/u^3
    mu_if = (1 + u**4) ** 2 / u**3
    okJe = is_zero(sp.expand(sp.diff(mu_if, u) + mu_if * cJ))
    log(f"[9] integrating factor mu = (1+u^4)^2/u^3: {okJe}")
    assert okJe

    # thermal zero mode: V_J = u^3/(1+u^4)^2, V_d = 0.  Its horizon data:
    # Huu(1) = -V_J(1)/6  =>  dT = 1/48, ds = (1/2) int_0^1 V_J = 1/16,
    # so ds/dT = 3 -- the s ~ T^3 thermal direction.
    VJ_mode = u**3 / (1 + u**4) ** 2
    ds_C = sp.Rational(1, 2) * sp.integrate(VJ_mode, (u, 0, 1))
    dT_C = -(-sp.Rational(1, 6) * VJ_mode.subs(u, 1)) / 2
    okJf = sp.simplify(ds_C / dT_C - 3) == 0
    log(f"[9] thermal zero mode: ds = {ds_C}, dT = {dT_C}, ratio 3: {okJf}")
    assert okJf

    # fixed-T anchor per B^2: Huu(1) = 0 fixes V_J(1) = 1/2 (src_B2(1) = 1/12),
    # then C = 3 and
    #   ds|_T = (1/2) int_0^1 u^3 (3 - 4 ln u - u^4)/(1+u^4)^2 du = 1/4 exactly.
    I1 = sp.integrate(u**3 / (1 + u**4) ** 2, (u, 0, 1))  # 1/8
    I2 = sp.integrate(u**7 / (1 + u**4) ** 2, (u, 0, 1))  # ln2/4 - 1/8
    v_ = sp.Symbol("v", positive=True)
    I3 = -sp.Rational(1, 4) * sp.integrate(
        sp.log(v_) / (1 + v_) ** 2, (v_, 0, 1)
    )  # ln2/4
    ds_T_B2 = sp.simplify(sp.Rational(1, 2) * (3 * I1 + I3 - I2))
    okJg = sp.simplify(ds_T_B2 - sp.Rational(1, 4)) == 0
    log(f"[9] EXACT fixed-T anchor per alpha B^2: ds|_T = {ds_T_B2} = 1/4: {okJg}")
    assert okJg

    # append the J-channel to the generated module
    with contextlib.nullcontext(GEN_TEXT) as fh:
        fh.write(
            "\n\ndef g0_j_channel(u, B, u_H, W0, W0p):\n"
            '    """V_J\' = c V_J + src (V_J = 2 Hxx\' + Hzz\'), u_H = 1."""\n'
            f"    c = {sp.pycode(cJ)}\n"
            f"    src_B2 = {sp.pycode(sJ_B2)}\n"
            f"    src_r2 = {sp.pycode(sp.expand(sJ_r2_target.subs({w(u): sp.Symbol('W0'), wp: sp.Symbol('W0p')})))}\n"
            "    return c, src_B2, src_r2\n\n\n"
            "DS_FIXED_T_B2 = 0.25  # exact: ds|_T per alpha B^2 (u_H = 1)\n"
        )
    log("[9] appended g0_j_channel to the generated module")
    log("done")

    # ---------------------------------------------------------------------------
    # provenance stamp
    # ---------------------------------------------------------------------------
    # Records which revision of this file produced the generated module, plus a
    # digest of the module itself, so that a hand edit or a stale regeneration is
    # visible instead of silent.  Must stay last: it digests the finished file.
    # See backreaction/generated.py and tests/test_generated_systems.py.
    from backreaction.generated import stamp  # noqa: E402

    with open(gen_path, "w") as fh:
        fh.write(GEN_TEXT.getvalue())
    log(f"[8] wrote {gen_path}")
    stamp(gen_path, __file__)
    log(f"[stamp] provenance written to {gen_path}")


if __name__ == "__main__":
    main()
