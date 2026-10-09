"""G = 0 thermodynamic shift of the black brane.

Solves the homogeneous (G = 0) sector of the linearised Einstein equations
derived in derivations/g0.py (systems/einstein_g0.py), per unit source:
  - the O(alpha B^2) magnetic-brane correction (source T^(0) per B^2), and
  - the O(alpha rho^2) condensate correction (source <T^(2)> per rho^2,
    lambda_0 = 1, b_0 = 0, at B = B_c),
at u_H = 1, alpha = 1.

The G = 0 system has a one-parameter family of normalisable solutions:
the *thermal zero mode* (V_J = u^3/(1+u^4)^2 in the trace channel, the
shift along the black-brane family; derivations/g0.py [9]).  The physical
convention adopted here is FIXED TEMPERATURE: Huu(1) = 0 (delta T = 0), so
all shifts are responses at fixed T.  Convention-independent invariants:
I_d(1) = (Hxx - Hzz)(1) and ds - 3 dT.

Methods:
  * algebraic gauge Htt = 0: closed (Hxx, Hzz) system + algebraic Huu +
    the fixed-T row, solved by (A) Chebyshev collocation and
    (B) second-order finite differences (N = 2400, no Richardson
    extrapolation), both checked against the exact ds|_T to 1e-3 relative;
  * (C) quadratures from the two decoupled channels (derivations/g0.py [5],
    [9]): I_d(1) = (1/4) int_0^1 S_d(t) log(1-t^4)/t^3 dt and the
    J = 2 Hxx + Hzz channel via its integrating factor.
Exact anchors per alpha B^2: I_d(1) = pi^2/48, ds|_T = 1/4.

Observables (per alpha x source unit, fixed u_H = 1, fixed T):
    delta s / s0 |_T = (2 Hxx + Hzz)(1) / 2,   I_d(1) ,
plus the near-boundary u^4 / u^4 log u data of (Hxx, Hzz) for the
boundary-stress read-off (same convention).  These are not fitted: the u^4 log u
coefficients are the closed forms of the near-boundary expansion (the trace
channel's log coefficient a = -4 per B^2 and 0 per rho^2, and the per-B^2 I_d
data u^4/8 - (u^4/2) log u), entered by hand, and only the u^4 coefficient of
the trace channel is computed.  The checks on the log coefficients are
therefore consistency checks of those closed forms, not measurements.

Run from the project root:
    uv run python -m backreaction.numerics.g0_thermo
"""

import numpy as np
from numpy.linalg import lstsq
from scipy.integrate import quad

from spectral_check import cheb
from exchange_kernel import zero_mode, clencurt_weights
from backreaction import paths
from backreaction.systems import einstein_g0 as eq

U_H = 1.0
EPS = 1e-7


# ---------------------------------------------------------------------------
# matter profiles (probe-limit zero mode at B_c, J_2 = 1 normalisation)
# ---------------------------------------------------------------------------
class Matter:
    def __init__(self, N=240):
        D, xs = cheb(N)
        u = (1.0 - xs) / 2.0
        D = -2.0 * D
        self.u, self.D = u, D
        self.wcc = clencurt_weights(N) / 2.0
        Bc, w0 = zero_mode(u, D, D @ D)
        one_over_u = np.zeros_like(u)
        one_over_u[1:] = 1.0 / u[1:]
        J2 = float(self.wcc @ (w0**2 * one_over_u))
        w0 /= np.sqrt(J2)
        self.Bc, self.w0, self.w0p = Bc, w0, D @ w0

    def interp(self, vals, ug):
        xs = 1.0 - 2.0 * self.u
        xg = 1.0 - 2.0 * np.atleast_1d(np.asarray(ug, dtype=float))
        n = len(xs) - 1
        wts = np.ones(n + 1)
        wts[1::2] = -1.0
        wts[0] *= 0.5
        wts[-1] *= 0.5
        out = np.empty(len(xg))
        for i, xv in enumerate(xg):
            diff = xv - xs
            hit = np.where(np.abs(diff) < 1e-14)[0]
            if len(hit):
                out[i] = vals[hit[0]]
            else:
                tmp = wts / diff
                out[i] = np.dot(tmp, vals) / tmp.sum()
        return out


MAT = Matter()
BC = MAT.Bc


def matter_at(uk):
    return MAT.interp(MAT.w0, [uk])[0], MAT.interp(MAT.w0p, [uk])[0]


# ---------------------------------------------------------------------------
# generic collocation solver (rows of mixed order, source selected by unit)
# ---------------------------------------------------------------------------
def solve_system(u, D, D2, rows, orders, unit, fixed_T=True):
    """rows: list of generated row functions; orders: LHS derivative order
    per row; unit: 'B2' or 'r2' selects the source.  fixed_T appends the
    convention row Huu(1) = 0 that removes the thermal zero mode."""
    nfield = len(rows[0](0.5, BC, U_H, 1.0, 1.0)[0])
    npts = len(u)
    nun = nfield * npts
    nrows = len(rows) * npts + (1 if fixed_T else 0)
    M = np.zeros((nrows, nun))
    rhs = np.zeros(nrows)

    def regfun(uk, order):
        fk = 1.0 - uk**4
        return uk**2 * fk if order == 2 else uk * fk

    def rowdata(fn, uk):
        W0, W0p = matter_at(uk)
        cH, cdH, sB2, sR2 = fn(uk, BC, U_H, W0, W0p)
        return cH, cdH, (sB2 if unit == "B2" else sR2)

    for i, (fn, order) in enumerate(zip(rows, orders, strict=True)):
        base = i * npts
        for k, uk in enumerate(u):
            row = base + k
            uk_eval = min(max(uk, EPS), 1.0 - EPS)
            cH, cdH, src = rowdata(fn, uk_eval)
            if uk <= EPS or uk >= 1.0 - EPS:
                u2 = min(max(uk, 2 * EPS), 1.0 - 2 * EPS)
                cH2, cdH2, src2 = rowdata(fn, u2)
                r1, r2_ = regfun(uk_eval, order), regfun(u2, order)
                cH = [2 * r1 * a - r2_ * b for a, b in zip(cH, cH2, strict=True)]
                cdH = [2 * r1 * a - r2_ * b for a, b in zip(cdH, cdH2, strict=True)]
                src = 2 * r1 * src - r2_ * src2
                r = 1.0
            else:
                r = regfun(uk, order)
            if order == 2:
                # LHS is H_i'' for the i-th *field* (row i <-> field i)
                M[row, base : base + npts] += r * D2[k]
            # subtract RHS coefficients (constraint rows: LHS = 0, all terms
            # are 'RHS')
            for j in range(nfield):
                M[row, j * npts : (j + 1) * npts] -= (r * cdH[j]) * D[k]
                M[row, j * npts + k] -= r * cH[j]
            rhs[row] = r * src
        # Dirichlet at the u = 0 node for second-order rows
        if order == 2:
            k0 = int(np.argmin(np.abs(u)))
            M[base + k0, :] = 0.0
            M[base + k0, base + k0] = 1.0
            rhs[base + k0] = 0.0
    if fixed_T:
        # Huu(1) = 0: coeff_dH . V(1) + src(1) = 0 (V = D @ H per field)
        W0, W0p = matter_at(1.0 - 1e-12)
        cH, cdH, sB2, sR2 = eq.g0a_huu(1.0, BC, U_H, W0, W0p)
        src = sB2 if unit == "B2" else sR2
        row = len(rows) * npts
        klast = int(np.argmin(np.abs(u - 1.0)))
        for j in range(nfield):
            M[row, j * npts : (j + 1) * npts] += cdH[j] * D[klast]
            M[row, j * npts + klast] += cH[j]
        rhs[row] = -src
    sol, res, rank, sv = lstsq(M, rhs, rcond=None)
    H = sol.reshape(nfield, npts)
    dH = np.array([D @ H[i] for i in range(nfield)])
    lin_res = float(np.max(np.abs(M @ sol - rhs)))
    return H, dH, sv, lin_res


def grids(N_cheb=180, N_fd=2400):
    D, xs = cheb(N_cheb)
    uc = (1.0 - xs) / 2.0
    Dc = -2.0 * D
    ufd = np.linspace(0.0, 1.0, N_fd + 1)
    du = ufd[1] - ufd[0]
    Nf = N_fd
    Df = np.zeros((Nf + 1, Nf + 1))
    D2f = np.zeros((Nf + 1, Nf + 1))
    for k in range(1, Nf):
        Df[k, k - 1], Df[k, k + 1] = -0.5 / du, 0.5 / du
        D2f[k, k - 1], D2f[k, k], D2f[k, k + 1] = 1 / du**2, -2 / du**2, 1 / du**2
    Df[0, :3] = np.array([-1.5, 2.0, -0.5]) / du
    Df[Nf, -3:] = np.array([0.5, -2.0, 1.5]) / du
    D2f[0, :4] = np.array([2.0, -5.0, 4.0, -1.0]) / du**2
    D2f[Nf, -4:] = np.array([-1.0, 4.0, -5.0, 2.0]) / du**2
    return (uc, Dc, Dc @ Dc), (ufd, Df, D2f)


# ---------------------------------------------------------------------------
# gauge A: algebraic Htt = 0
# ---------------------------------------------------------------------------
def solve_alg(unit, grid):
    u, D, D2 = grid
    H, dH, sv, lres = solve_system(
        u, D, D2, [eq.g0a_row_xx, eq.g0a_row_zz], [2, 2], unit
    )
    # algebraic Huu on the solution
    Huu = np.zeros(len(u))
    for k, uk in enumerate(u):
        uk = min(max(uk, 1e-9), 1.0 - 1e-9)
        W0, W0p = matter_at(uk)
        cH, cdH, sB2, sR2 = eq.g0a_huu(uk, BC, U_H, W0, W0p)
        src = sB2 if unit == "B2" else sR2
        Huu[k] = np.dot(cH, H[:, k]) + np.dot(cdH, dH[:, k]) + src
    return dict(
        u=u,
        Hxx=H[0],
        Hzz=H[1],
        Huu=Huu,
        dH=dH,
        sv=sv,
        lres=lres,
        dT=-0.5 * Huu[-1],
        ds=H[0][-1] + 0.5 * H[1][-1],
        Id=H[0][-1] - H[1][-1],
    )


# ---------------------------------------------------------------------------
# method C: quadratures from the decoupled channels (derivations/g0.py [5], [9])
# ---------------------------------------------------------------------------
def id_quadrature(unit):
    """I_d(1) = (1/4) int_0^1 S_d(t) log(1 - t^4)/t^3 dt (order-swapped
    double quadrature; log-integrable endpoint)."""

    def igrand(t):
        W0, W0p = matter_at(t)
        sB2, sR2 = eq.g0_id_source(t, BC, U_H, W0, W0p)
        Sd = sB2 if unit == "B2" else sR2
        return 0.25 * Sd * np.log(max(1.0 - t**4, 1e-300)) / t**3

    val, err = quad(igrand, 1e-9, 1.0, limit=200)
    return val, err


# ---------------------------------------------------------------------------
# method A': full profiles from the two decoupled channels (primary method;
# machine-precision, no collocation).  V_J = 2Hxx' + Hzz', V_d = Hxx' - Hzz'
# => Hxx' = (V_J + V_d)/3, Hzz' = (V_J - 2 V_d)/3, H by antiderivative,
# Huu algebraic.  Fixed-T convention throughout.
# ---------------------------------------------------------------------------
def antideriv(ug, Dg, vals, at_one=False):
    """F' = vals with F(0) = 0 (or F(1) = 0 for at_one)."""
    M = Dg.copy()
    k0 = int(np.argmin(np.abs(ug - (1.0 if at_one else 0.0))))
    M[k0, :] = 0.0
    M[k0, k0] = 1.0
    rhs = vals.copy()
    rhs[k0] = 0.0
    return np.linalg.solve(M, rhs)


def solve_quad(unit, N=400):
    D, xs = cheb(N)
    ug = (1.0 - xs) / 2.0
    Dg = -2.0 * D

    # --- V_d channel: f V_d / u^3 = gtil(u) = -int_u^1 S_d/t^3 dt
    if unit == "B2":
        gtil = np.array([-2.0 * np.log(max(uk, 1e-300)) for uk in ug])
        Sd1 = -2.0
    else:
        integ = np.zeros(N + 1)
        for k, tk in enumerate(ug):
            tk_e = min(max(tk, 1e-10), 1.0 - 1e-12)
            W0, W0p = matter_at(tk_e)
            _, sR2 = eq.g0_id_source(tk_e, BC, U_H, W0, W0p)
            integ[k] = sR2 / tk_e**3
        # antideriv gives F(u) with F' = integ, F(1) = 0, i.e.
        # F(u) = -int_u^1 integ = gtil
        gtil = antideriv(ug, Dg, integ, at_one=True)
        W1, W1p = matter_at(1.0 - 1e-12)
        Sd1 = eq.g0_id_source(1.0 - 1e-12, BC, U_H, W1, W1p)[1]
    Vd = np.zeros(N + 1)
    for k, uk in enumerate(ug):
        if uk < 1e-12:
            Vd[k] = 0.0
        elif uk > 1.0 - 1e-10:
            Vd[k] = -Sd1 / 4.0  # l'Hopital at the horizon
        else:
            Vd[k] = uk**3 * gtil[k] / (1.0 - uk**4)

    # --- V_J channel (fixed T): V_J = u^3/(1+u^4)^2 [4 V_J(1) + a log u
    #     + R(u)],  R' = mu s_J - a/u, R(1) = 0
    W1, W1p = matter_at(1.0 - 1e-12)
    _, _, sB2_uu, sR2_uu = eq.g0a_huu(1.0, BC, U_H, W1, W1p)
    VJ1 = 6.0 * (sB2_uu if unit == "B2" else sR2_uu)
    a_log = -4.0 if unit == "B2" else 0.0
    reg = np.zeros(N + 1)
    for k, tk in enumerate(ug):
        tk_e = min(max(tk, 1e-10), 1.0)
        W0, W0p = matter_at(tk_e)
        _, sB2j, sR2j = eq.g0_j_channel(tk_e, BC, U_H, W0, W0p)
        sJ = sB2j if unit == "B2" else sR2j
        reg[k] = sJ * (1.0 + tk_e**4) ** 2 / tk_e**3 - a_log / tk_e
    R = antideriv(ug, Dg, reg, at_one=True)
    VJ = np.zeros(N + 1)
    for k, uk in enumerate(ug):
        if uk < 1e-12:
            VJ[k] = 0.0
        else:
            VJ[k] = uk**3 / (1.0 + uk**4) ** 2 * (4.0 * VJ1 + a_log * np.log(uk) + R[k])

    V1 = (VJ + Vd) / 3.0
    V2 = (VJ - 2.0 * Vd) / 3.0
    Hxx = antideriv(ug, Dg, V1)
    Hzz = antideriv(ug, Dg, V2)
    Huu = np.zeros(N + 1)
    for k, uk in enumerate(ug):
        uk_e = min(max(uk, 1e-9), 1.0)
        W0, W0p = matter_at(uk_e)
        cH, cdH, sB2u, sR2u = eq.g0a_huu(uk_e, BC, U_H, W0, W0p)
        src = sB2u if unit == "B2" else sR2u
        Huu[k] = cdH[0] * V1[k] + cdH[1] * V2[k] + src

    # exact boundary data: t4 of the two channels
    # V_J = u^3 [T4 + a log u] + O(u^7 log) with T4 = 4 VJ1 + R(0);
    # integrating, the u^4 data of J = 2Hxx + Hzz are
    #   t4J = T4/4 - a/16,  sl4J = a/4  (the -a/16 from int u^3 log u).
    R0 = R[int(np.argmin(np.abs(ug)))]
    t4J = (4.0 * VJ1 + R0) / 4.0 - a_log / 16.0
    sl4J = a_log / 4.0
    if unit == "B2":
        t4d, sl4d = 0.125, -0.5  # exact: I_d = u^4/8 - (1/2) u^4 log u
    else:
        t4d = float(
            -0.25
            * quad(
                lambda tk: eq.g0_id_source(tk, BC, U_H, *matter_at(tk))[1] / tk**3,
                1e-9,
                1.0,
                limit=200,
            )[0]
        )
        sl4d = 0.0
    s4x = (t4J + t4d) / 3.0
    s4z = (t4J - 2.0 * t4d) / 3.0
    sl4x = (sl4J + sl4d) / 3.0
    sl4z = (sl4J - 2.0 * sl4d) / 3.0
    return dict(
        u=ug,
        D=Dg,
        Hxx=Hxx,
        Hzz=Hzz,
        Huu=Huu,
        V1=V1,
        V2=V2,
        dT=-0.5 * Huu[-1] if abs(ug[-1] - 1) < 1e-12 else None,
        ds=Hxx[-1] + 0.5 * Hzz[-1],
        Id=Hxx[-1] - Hzz[-1],
        s4x=s4x,
        s4z=s4z,
        sl4x=sl4x,
        sl4z=sl4z,
    )


CHECKS = {}


def check(name, ok):
    CHECKS[name] = bool(ok)


def report(unit):
    print(
        f"===== source unit: per alpha {'B^2' if unit == 'B2' else 'rho^2'}"
        f" (u_H = 1, fixed-T convention"
        f"{'' if unit == 'B2' else ', B = B_c'}) ====="
    )
    Q = solve_quad(unit, N=400)
    Q2 = solve_quad(unit, N=600)
    gc, gf = grids()
    A_c = solve_alg(unit, gc)
    A_f = solve_alg(unit, gf)
    Id_q, Id_err = id_quadrature(unit)

    print(
        f"  fixed-T check: delta T/T0 = {Q2['dT']:.2e} (should be 0; "
        f"collocation {A_c['dT']:.2e})"
    )
    print(f"  ds|_T: channel N=400/600 {Q['ds']:.10f} / {Q2['ds']:.10f}")
    print(f"         collocation cheb {A_c['ds']:.6f} / fd {A_f['ds']:.6f}")
    if unit == "B2":
        print("         EXACT anchor 1/4 = 0.2500000000")
    else:
        print(f"         -B_c = {-BC:.10f}")
    print(
        f"  I_d(1): channel {Q2['Id']:.10f}  adaptive quadrature "
        f"{Id_q:.10f} (err {Id_err:.1e})"
    )
    if unit == "B2":
        print(f"          analytic anchor pi^2/48 = {np.pi**2 / 48:.10f}")
    print(
        f"  boundary u^4 data (exact/quadrature): s4x = {Q2['s4x']:.8f}, "
        f"s4z = {Q2['s4z']:.8f}"
    )
    print(f"  u^4 log u coefficients: sl4x = {Q2['sl4x']:.6f}, sl4z = {Q2['sl4z']:.6f}")
    if unit == "B2":
        print("  (exact per B^2: s4x = 3/8, s4z = 1/4, sl4x = -1/2, sl4z = 0)")
    # profile-level agreement of collocation vs channel construction
    Hxx_i = np.interp(A_c["u"][1:-1], Q2["u"], Q2["Hxx"])
    dev = np.max(np.abs(Hxx_i - A_c["Hxx"][1:-1]))
    print(f"  collocation vs channel profiles: max |dHxx| = {dev:.2e}")
    print()

    # Checks.  Exact values: per alpha B^2, ds|_T = 1/4, I_d(1) = pi^2/48 and
    # the boundary data (3/8, 1/4, -1/2, 0) (closed forms of the near-boundary
    # expansion of the G = 0 system, not derived in this package); per alpha
    # rho^2, ds|_T = -B_c and s4x = -3B_c/2, s4z = -B_c (the G = 0 identities).
    ds_exact = 0.25 if unit == "B2" else -BC
    check(f"[{unit}] fixed T: delta T/T0 < 1e-10", abs(Q2["dT"]) < 1e-10)
    check(f"[{unit}] ds|_T exact to 1e-9", abs(Q2["ds"] - ds_exact) < 1e-9)
    check(f"[{unit}] ds|_T N = 400 vs 600 to 1e-9", abs(Q["ds"] - Q2["ds"]) < 1e-9)
    check(
        f"[{unit}] second method (collocation, fd) within 1e-3 relative",
        max(abs(A_c["ds"] - ds_exact), abs(A_f["ds"] - ds_exact))
        < 1e-3 * abs(ds_exact),
    )
    check(f"[{unit}] I_d channel vs adaptive quadrature", abs(Q2["Id"] - Id_q) < 1e-8)
    if unit == "B2":
        check("[B2] I_d(1) = pi^2/48", abs(Q2["Id"] - np.pi**2 / 48) < 1e-9)
        check(
            "[B2] boundary data (3/8, 1/4, -1/2, 0)",
            max(
                abs(Q2["s4x"] - 0.375),
                abs(Q2["s4z"] - 0.25),
                abs(Q2["sl4x"] + 0.5),
                abs(Q2["sl4z"]),
            )
            < 1e-6,
        )
    else:
        check(
            "[r2] boundary data s4x = -3B_c/2, s4z = -B_c",
            max(abs(Q2["s4x"] + 1.5 * BC), abs(Q2["s4z"] + BC)) < 1e-7,
        )
    return dict(Q=Q2)


def main():
    print(f"zero mode: B_c = {BC:.8f} (u_H = 1), J2 = 1 normalisation")
    print(
        "convention: fixed temperature (thermal zero mode removed; "
        "derivations/g0.py [9])\n"
    )
    out_B2 = report("B2")
    out_r2 = report("r2")
    QB, QR = out_B2["Q"], out_r2["Q"]
    print(f"headline: ds|_T per alpha B^2   = {QB['ds']:.10f}  (exact 1/4)")
    print(
        f"          ds|_T per alpha rho^2 = {QR['ds']:.10f}  "
        f"(-B_c = {-BC:.10f}, rel diff "
        f"{abs(QR['ds'] + BC) / BC:.2e})"
    )
    failed = [k for k, v in CHECKS.items() if not v]
    print(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
        print("cache not written: a check failed")
        raise SystemExit(1)
    np.savez(
        paths.data("g0_thermo.npz"),
        Bc=BC,
        ds_B2=QB["ds"],
        Id_B2=QB["Id"],
        ds_r2=QR["ds"],
        Id_r2=QR["Id"],
        s4_B2=[QB["s4x"], QB["s4z"]],
        s4_r2=[QR["s4x"], QR["s4z"]],
        sl4_B2=[QB["sl4x"], QB["sl4z"]],
        sl4_r2=[QR["sl4x"], QR["sl4z"]],
    )
    print("saved backreaction/data/g0_thermo.npz")


if __name__ == "__main__":
    main()
