"""Exact uncharged magnetic black-brane background (D'Hoker-Kraus) in OUR
conventions, at fixed temperature -- a reusable solver for downstream radial
BVPs (condensate zero mode, per-harmonic response).

PHYSICS
-------
Einstein-Maxwell in Ricci form (L = 1, uncharged, our action normalisation):

    R_MN + 4 g_MN = alpha (F_MP F_N^P - (1/6) F^2 g_MN),     F_xy = B (const).

Metric ansatz (boundary r -> inf, non-extremal horizon at r_p):

    ds^2 = -U(r) dt^2 + dr^2/U(r) + e^{2V(r)}(dx^2 + dy^2) + e^{2W(r)} dz^2.

The background depends on (alpha, B) ONLY through beta = alpha B^2 (proved
symbolically below: F~ = sqrt(alpha/2) F maps to the alpha = 2 D'Hoker-Kraus
normalisation with B~ = sqrt(alpha/2) B).  The one-parameter family is labelled
by beta.

FRAME  (also reported at runtime by solve_report)
-----
Horizon at r_p = 1.  Boundary-normalised: U/r^2 -> 1, e^{2V}/r^2 -> 1,
e^{2W}/r^2 -> 1 as r -> inf.  The leading coefficient of U is 1 automatically
(the "4" in the EOM fixes the AdS radius L = 1; verified to 1e-13); e^{2V},
e^{2W} are normalised by shifting V, W by constants (== rescaling x,y,z), which
also makes B the PHYSICAL boundary flux density B_phys.  Fixed temperature:
U'(r_p) = 4  <=>  T = U'(r_p)/(4 pi) = 1/pi exactly (matching the probe's
u_H = 1, T = 1/pi at d = 4).  So B here IS the physical field and beta =
alpha B^2 with T pinned at 1/pi.

There is a genuine O(beta) linear (r^1) term in the U asymptotics and B^2 log r
terms at order r^-2 log r (the conformal anomaly); the leading-coefficient
normalisation above is unaffected (the logs live at r^-2 relative to r^2), and
far-field frame extraction is not corrupted by them (they are r^-4 relative).

COMPACTIFIED COORDINATE / REDUCED FIELDS (downstream interface)
-----
sigma = r_p / r = 1/r in (0, 1];  sigma = 1 horizon, sigma = 0 boundary.
    A(sigma) = sigma^2 U      (A(0) = 1, A(1) = 0)     U   = A / sigma^2
    M(sigma) = V + ln sigma   (M(0) = 0)               V   = M - ln sigma
    N(sigma) = W + ln sigma   (N(0) = 0)               W   = N - ln sigma
A, M, N are represented in a Chebyshev basis on sigma in [0,1].  Their
coefficients decay algebraically ~ n^-9 (the sigma^4 log sigma anomaly term),
reaching a ~1e-13 tail by N ~ 64 -- effectively spectral; demonstrated in
main.

METHODS (two methods: shooting vs Chebyshev spectral collocation)
-----
  (1) solve_shoot: DOP853 from the horizon (like dhoker_kraus_compare.run_brane),
      root-finding the raw horizon field to hit a target beta, then extracting
      and imposing the boundary-normalised fixed-T frame.
  (2) solve_colloc: nonlinear Chebyshev collocation (Newton) of the BVP in sigma
      on (A, M, N), directly in the normalised frame.  Reuses the symbolic
      reduced ODE (same equations; independent discretisation).

Run:  uv run python -m backreaction.numerics.dk_background
"""

import time

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from spectral_check import cheb
from backreaction import paths

PI = np.pi

CHECKS = {}


def check(name, ok):
    CHECKS[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")


# ===========================================================================
# SYMBOLIC CORE: reduce the Ricci-form EOM to ODEs for U, V, W; verify.
# ===========================================================================
def _build_symbolics():
    rr, B, alpha = sp.symbols("rr B alpha", positive=True)
    coords = sp.symbols("t rr x y z")
    t, rr_, x, y, z = coords
    U = sp.Function("U")(rr_)
    V = sp.Function("V")(rr_)
    W = sp.Function("W")(rr_)
    g = sp.diag(-U, 1 / U, sp.exp(2 * V), sp.exp(2 * V), sp.exp(2 * W))
    ginv = g.inv()

    def ricci(g, xs):
        n = len(xs)
        gi = g.inv()
        Gam = [
            [
                [
                    sp.together(
                        sum(
                            gi[l, s]
                            * (
                                sp.diff(g[s, m], xs[k])
                                + sp.diff(g[s, k], xs[m])
                                - sp.diff(g[m, k], xs[s])
                            )
                            for s in range(n)
                        )
                        / 2
                    )
                    for k in range(n)
                ]
                for m in range(n)
            ]
            for l in range(n)
        ]
        Ric = sp.zeros(n, n)
        for m in range(n):
            for k in range(n):
                e = 0
                for l in range(n):
                    e += sp.diff(Gam[l][m][k], xs[l]) - sp.diff(Gam[l][m][l], xs[k])
                    for s in range(n):
                        e += Gam[l][l][s] * Gam[s][m][k] - Gam[l][k][s] * Gam[s][m][l]
                Ric[m, k] = sp.simplify(e)
        return Ric

    F = sp.zeros(5, 5)
    F[2, 3] = B
    F[3, 2] = -B
    F2 = sum(
        F[m, n] * ginv[m, a] * ginv[n, b] * F[a, b]
        for m in range(5)
        for n in range(5)
        for a in range(5)
        for b in range(5)
    )
    FMPFNP = sp.Matrix(
        5,
        5,
        lambda m, n: sum(
            F[m, p] * ginv[p, s] * F[n, s] for p in range(5) for s in range(5)
        ),
    )
    Ric = ricci(g, list(coords))
    EOM = sp.simplify(Ric + 4 * g - alpha * (FMPFNP - F2 * g / 6))

    Up, Vp, Wp = sp.symbols("Up Vp Wp")
    Upp, Vpp, Wpp = sp.symbols("Upp Vpp Wpp")
    Uv, Vv, Wv = sp.symbols("Uv Vv Wv")
    rep = {
        sp.diff(U, rr_, 2): Upp,
        sp.diff(V, rr_, 2): Vpp,
        sp.diff(W, rr_, 2): Wpp,
        sp.diff(U, rr_): Up,
        sp.diff(V, rr_): Vp,
        sp.diff(W, rr_): Wp,
        U: Uv,
        V: Vv,
        W: Wv,
    }
    eom_tt = EOM[0, 0].subs(rep)
    eom_rr = EOM[1, 1].subs(rep)
    eom_xx = EOM[2, 2].subs(rep)
    eom_zz = EOM[4, 4].subs(rep)
    solved = sp.solve([eom_tt, eom_xx, eom_zz], [Upp, Vpp, Wpp], dict=True)[0]
    # first-order Hamiltonian constraint: rr-equation with 2nd derivatives removed
    constraint = sp.simplify(eom_rr.subs(solved))

    args = (rr_, Uv, Vv, Wv, Up, Vp, Wp, B, alpha)
    lam = lambda e: sp.lambdify(args, e, "numpy")
    out = dict(
        Upp=lam(solved[Upp]),
        Vpp=lam(solved[Vpp]),
        Wpp=lam(solved[Wpp]),
        con=lam(constraint),
        # symbolic pieces kept for the verification block
        _sym=dict(
            EOM=EOM,
            solved=solved,
            constraint=constraint,
            rep=rep,
            Upp=solved[Upp],
            Vpp=solved[Vpp],
            Wpp=solved[Wpp],
            symbols=(rr_, Uv, Vv, Wv, Up, Vp, Wp, B, alpha, Upp, Vpp, Wpp),
        ),
    )

    # analytic partials of the reduced RHS w.r.t. (U,V,W,U',V',W') for the
    # collocation Jacobian
    def partials(expr):
        return {v: lam(sp.diff(expr, v)) for v in (Uv, Vv, Wv, Up, Vp, Wp)}

    out["dUpp"] = partials(solved[Upp])
    out["dVpp"] = partials(solved[Vpp])
    out["dWpp"] = partials(solved[Wpp])
    return out


_SYM = _build_symbolics()
fUpp, fVpp, fWpp, fcon = _SYM["Upp"], _SYM["Vpp"], _SYM["Wpp"], _SYM["con"]


def verify_symbolics(verbose=True):
    """Symbolic checks: (i) the reduced ODE follows from the Ricci-form EOM;
    (ii) alpha,B enter only via beta = alpha B^2; (iii) horizon-constraint
    coefficients."""
    S = _SYM["_sym"]
    rr_, Uv, Vv, Wv, Up, Vp, Wp, B, alpha, Upp, Vpp, Wpp = S["symbols"]
    ok = {}
    # (i) the components NOT used to build the reduced system: every
    #     off-diagonal component vanishes identically and (yy) equals (xx).
    #     (Substituting the solved second derivatives back into (tt),(xx),(zz)
    #     would be a tautology, so it is not done.)
    EOM = S["EOM"]
    rep = S["rep"]
    offdiag = all(
        sp.simplify(EOM[i, j]) == 0 for i in range(5) for j in range(5) if i != j
    )
    ok["off-diagonal Ricci-form components vanish on the ansatz"] = offdiag
    ok["(yy) component equals (xx)"] = sp.simplify(EOM[3, 3] - EOM[2, 2]) == 0
    # (i') the Hamiltonian constraint is the one printed in the paper (app. D.1),
    #      U(2V'^2 + 4V'W') + U'(2V' + W') + beta e^{-4V} = 12, up to a factor.
    con = S["constraint"]
    printed = (
        Uv * (2 * Vp**2 + 4 * Vp * Wp)
        + Up * (2 * Vp + Wp)
        + alpha * B**2 * sp.exp(-4 * Vv)
        - 12
    )
    ratio = sp.simplify(con / printed)
    ok["constraint = (printed app. D.1 form) x a nonzero factor"] = ratio != 0 and not (
        ratio.free_symbols & {Up, Vp, Wp, Upp, Vpp, Wpp}
    )
    # (i'') the Bianchi identity: the constraint is propagated by the reduced
    #      system.  Its total r-derivative along the flow vanishes wherever the
    #      constraint itself does; tested at random points on the constraint
    #      surface (the constraint is linear in U', so solve it for U').
    dcon = (
        sp.diff(con, rr_)
        + sp.diff(con, Uv) * Up
        + sp.diff(con, Vv) * Vp
        + sp.diff(con, Wv) * Wp
        + sp.diff(con, Up) * S["Upp"]
        + sp.diff(con, Vp) * S["Vpp"]
        + sp.diff(con, Wp) * S["Wpp"]
    )
    up_on_shell = sp.solve(con, Up)
    rng = np.random.default_rng(7)
    prop = []
    for _ in range(6):
        pt = {
            rr_: sp.Rational(int(rng.integers(12, 40)), 10),
            Uv: sp.Rational(int(rng.integers(5, 30)), 10),
            Vv: sp.Rational(int(rng.integers(-5, 5)), 10),
            Wv: sp.Rational(int(rng.integers(-5, 5)), 10),
            Vp: sp.Rational(int(rng.integers(1, 20)), 10),
            Wp: sp.Rational(int(rng.integers(1, 20)), 10),
            B: sp.Rational(int(rng.integers(1, 20)), 10),
            alpha: sp.Rational(int(rng.integers(1, 30)), 10),
        }
        upv = up_on_shell[0].subs(pt)
        prop.append(abs(sp.N(dcon.subs(pt).subs(Up, upv), 30)))
    ok["constraint propagated by the reduced system (Bianchi), 6 random points"] = (
        len(up_on_shell) == 1 and max(prop) < 1e-25
    )
    # (ii) beta-only dependence
    beta = sp.symbols("beta", positive=True)
    sub = {B: sp.sqrt(beta / alpha)}
    beta_only = all(
        alpha not in sp.simplify(e.subs(sub)).free_symbols
        for e in (S["Upp"], S["Vpp"], S["Wpp"], S["constraint"])
    )
    ok["beta = alpha B^2 is the only coupling (RHS + constraint)"] = beta_only
    # (iii) horizon-regularity coefficients: at U=0 the (xx),(zz) EOM give
    #   U'V' = 4 - (2 alpha/3)B^2 e^{-4V},  U'W' = 4 + (alpha/3)B^2 e^{-4V}.
    # at U = 0 the (xx),(zz) EOM carry no second derivatives (checked); keep
    # V(1) = Vv, W(1) = Wv general.
    ex1 = sp.expand(EOM[2, 2].subs(rep).subs({Uv: 0}))
    ex3 = sp.expand(EOM[4, 4].subs(rep).subs({Uv: 0}))
    s1 = sp.solve(ex1, Up * Vp)
    s3 = sp.solve(ex3, Up * Wp)
    UpVp = sp.simplify(s1[0]) if s1 else None
    UpWp = sp.simplify(s3[0]) if s3 else None
    # V(1) = Vv enters via e^{-4V} (F^2 ~ B^2 e^{-4V}); at Vv=0 the raw-frame
    # coefficients 4 -/+ (2/3, 1/3) alpha B^2 are recovered.
    ok["horizon U'V' = 4 - (2 alpha/3) B^2 e^{-4V}"] = (
        UpVp is not None
        and sp.simplify(UpVp - (4 - sp.Rational(2, 3) * alpha * B**2 * sp.exp(-4 * Vv)))
        == 0
    )
    ok["horizon U'W' = 4 + (alpha/3) B^2 e^{-4V}"] = (
        UpWp is not None
        and sp.simplify(UpWp - (4 + sp.Rational(1, 3) * alpha * B**2 * sp.exp(-4 * Vv)))
        == 0
    )
    if verbose:
        for k, v in ok.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return ok


# ===========================================================================
# Frame extraction helpers (far-field fit; logs are r^-4 relative -> negligible)
# ===========================================================================
def _leading_coeffs(sol, rmax, nfit=60):
    rs = np.linspace(rmax * 0.5, rmax, nfit)
    ys = sol.sol(rs)
    Amat = np.stack([np.ones_like(rs), 1 / rs, 1 / rs**2, 1 / rs**4], axis=1)

    def coeff(y):
        c, *_ = np.linalg.lstsq(Amat, y, rcond=None)
        return c[0]

    uinf = coeff(ys[0] / rs**2)
    vinf = coeff(np.exp(2 * ys[1]) / rs**2)
    winf = coeff(np.exp(2 * ys[2]) / rs**2)
    return uinf, vinf, winf


# ===========================================================================
# METHOD 1: shooting
# ===========================================================================
def _shoot_raw(b, alpha=2.0, u1=4.0, rmax=3000.0):
    """Integrate the ansatz from just outside the horizon (V(1)=W(1)=0 frame)."""
    Vp0 = (4 - 2 * alpha * b**2 / 3) / u1
    Wp0 = (4 + alpha * b**2 / 3) / u1
    d = 1e-9
    y0 = [u1 * d, Vp0 * d, Wp0 * d, u1, Vp0, Wp0]

    def rhs(r_, yv):
        Uv_, Vv_, Wv_, Up_, Vp_, Wp_ = yv
        return [
            Up_,
            Vp_,
            Wp_,
            fUpp(r_, Uv_, Vv_, Wv_, Up_, Vp_, Wp_, b, alpha),
            fVpp(r_, Uv_, Vv_, Wv_, Up_, Vp_, Wp_, b, alpha),
            fWpp(r_, Uv_, Vv_, Wv_, Up_, Vp_, Wp_, b, alpha),
        ]

    return solve_ivp(
        rhs,
        [1 + d, rmax],
        y0,
        rtol=3e-13,
        atol=1e-16,
        dense_output=True,
        method="DOP853",
    )


def _rmax_for_beta(beta):
    """Asymptotic region recedes as the throat lengthens with beta
    (an empirical calibration of where the shooting reaches it)."""
    return float(max(3000.0, 300.0 * beta + 3000.0))


def _grid_rmax(N):
    """Smallest positive sigma node maps to r = 1/sigma_min; the shooting
    integration must reach at least this far to fill the Chebyshev rep."""
    sig_min = (np.cos(np.pi * (N - 1) / N) + 1) / 2.0
    return 1.05 / sig_min


def solve_shoot(beta, alpha=2.0, N=96, rmax=None, return_raw=False):
    """Shooting solver.  Returns a Background in the boundary-normalised,
    fixed-T frame for the target beta = alpha B^2."""
    if rmax is None:
        rmax = _rmax_for_beta(beta)
    # cover both the throat (allowing overshoot during the b-search) and the
    # boundary-clustered Chebyshev grid
    rmax = max(rmax, _rmax_for_beta(1.6 * beta), _grid_rmax(N))

    # relation:  B_phys = b/vinf, beta = alpha B_phys^2.  Root-find b to hit beta.
    def beta_of_b(b):
        sol = _shoot_raw(b, alpha=alpha, rmax=rmax)
        if not sol.success:
            return np.nan
        uinf, vinf, winf = _leading_coeffs(sol, rmax)
        if not np.isfinite(vinf) or vinf <= 0:
            return np.nan
        return alpha * (b / vinf) ** 2

    if beta == 0.0:
        b_star = 0.0
    else:
        # beta(b) increases monotonically with b; scan upward for a bracket,
        # backing off if b overshoots into the divergent (unreachable-rmax) regime
        b_lo = 1e-4
        b_hi = max(1e-3, np.sqrt(beta / alpha) * 0.5)
        f_hi = beta_of_b(b_hi) - beta
        it = 0
        while (not np.isfinite(f_hi) or f_hi < 0) and it < 80:
            if not np.isfinite(f_hi):
                b_hi = 0.5 * (b_lo + b_hi)
            else:
                b_lo = b_hi
                b_hi *= 1.15
            f_hi = beta_of_b(b_hi) - beta
            it += 1
        b_star = brentq(
            lambda b: beta_of_b(b) - beta, b_lo, b_hi, xtol=1e-14, rtol=1e-13
        )
    sol = _shoot_raw(b_star, alpha=alpha, rmax=rmax)
    uinf, vinf, winf = _leading_coeffs(sol, rmax)
    Bphys = b_star / vinf
    # normalised reduced fields on Chebyshev sigma-grid
    D, xs = cheb(N)
    sig = (xs + 1) / 2.0  # sigma in [0,1], sig[0]=1 horizon
    A = np.empty(N + 1)
    M = np.empty(N + 1)
    Nn = np.empty(N + 1)
    lnv = 0.5 * np.log(vinf)
    lnw = 0.5 * np.log(winf)
    for j, sg in enumerate(sig):
        if sg < 1e-13:
            A[j], M[j], Nn[j] = 1.0, 0.0, 0.0
            continue
        r_ = 1.0 / sg
        y = sol.sol(min(r_, rmax))
        A[j] = sg**2 * y[0]
        M[j] = (y[1] - lnv) + np.log(sg)  # V_norm = V_raw - lnv
        Nn[j] = (y[2] - lnw) + np.log(sg)
    bg = Background(
        beta,
        alpha,
        N,
        A,
        M,
        Nn,
        Bphys=Bphys,
        uinf=uinf,
        vinf=vinf,
        winf=winf,
        method="shoot",
        b_raw=b_star,
        rmax=rmax,
    )
    if return_raw:
        bg._raw = sol
    return bg


# ===========================================================================
# METHOD 2: nonlinear Chebyshev collocation (Newton) in sigma
# ===========================================================================
def _fields_from_AMN(sig, A, M, Nn, D):
    """Return U,V,W and first derivatives (in r), plus the sigma-derivatives,
    at every node, given nodal (A,M,N)."""
    As = D @ A
    Ms = D @ M
    Ns = D @ Nn
    with np.errstate(divide="ignore", invalid="ignore"):
        U = A / sig**2
        Up = 2 * A / sig - As
        V = M - np.log(sig)
        Vp = -(sig**2) * Ms + sig
        W = Nn - np.log(sig)
        Wp = -(sig**2) * Ns + sig
    return U, V, W, Up, Vp, Wp, As, Ms, Ns


RES_FLOOR_C = (
    10.0  # Newton residual floor = RES_FLOOR_C * N^4 * eps (see residual_floor)
)


def residual_floor(N):
    """Round-off floor of the collocation residual: the second-derivative
    matrix D2 has entries ~N^4, so |F| cannot be driven below ~N^4 eps.  The
    floor used is RES_FLOOR_C = 10 times that, 2e-6..7e-5 at the ladder sizes
    (N = 176..416) -- far above the nominal `tol`.  The factor 10 is a margin
    that is used: the top rung (beta = 1e11, N = 416) stops at 1.07e-5 =
    1.6 N^4 eps and passes only because of it; the effect of that residual on
    B_c has not been measured separately.  A solve is ACCEPTED if its residual
    is below max(tol, residual_floor(N))."""
    return RES_FLOOR_C * N**4 * np.finfo(float).eps


class NotConverged(RuntimeError):
    """A collocation solve whose final residual missed max(tol, residual_floor(N))."""


def solve_colloc(
    beta,
    alpha=2.0,
    N=64,
    guess=None,
    tol=1e-12,
    maxit=60,
    verbose=False,
    require=False,
):
    """Newton collocation in sigma on (A,M,N).  Direct normalised frame.
    Returns a Background whose `accepted` attribute records whether the final
    residual met max(tol, residual_floor(N)).  With require=True an unaccepted
    solve raises NotConverged; with the default require=False it is returned
    and the caller must check `bg.accepted` itself (the retry loops do)."""
    Bphys = np.sqrt(beta / alpha)
    D, xs = cheb(N)
    sig = (xs + 1) / 2.0
    D = 2.0 * D  # d/dsigma on [0,1] (xs=2 sigma-1)
    D2 = D @ D
    ihor = 0  # sig[0] = 1
    ibnd = N  # sig[N] = 0
    interior = np.arange(1, N)  # 0 < sigma < 1

    # horizon regularity (general alpha), U'(1)=4.  In the normalised frame
    # V(1)=M(1) != 0, so the F^2 ~ B^2 e^{-4V} factor enters:
    #   U'V'|_h = 4 - (2 alpha/3) B^2 e^{-4 V(1)},  U'W'|_h = 4 + (alpha/3) B^2 e^{-4 V(1)}
    #   => M'(1) = (beta/6) e^{-4 M(1)},  N'(1) = -(beta/12) e^{-4 M(1)}
    As_h = -4.0  # A'(1) = -4  <=>  U'(1)=4 (T=1/pi)

    if guess is None:
        A = 1.0 - sig**4  # AdS-Schw
        M = np.zeros(N + 1)
        Nn = np.zeros(N + 1)
    else:
        A, M, Nn = guess

    def residual(vec):
        A, M, Nn = vec[: N + 1], vec[N + 1 : 2 * (N + 1)], vec[2 * (N + 1) :]
        U, V, W, Up, Vp, Wp, As, Ms, Ns = _fields_from_AMN(sig, A, M, Nn, D)
        Ass = D2 @ A
        Mss = D2 @ M
        Nss = D2 @ Nn
        # ODE residuals (interior): U''(fields) - fUpp(fields) etc.
        Upp_f = sig**2 * Ass - 2 * sig * As + 2 * A
        Vpp_f = sig**4 * Mss + 2 * sig**3 * Ms - sig**2
        Wpp_f = sig**4 * Nss + 2 * sig**3 * Ns - sig**2
        rA = np.empty(N + 1)
        rM = np.empty(N + 1)
        rN = np.empty(N + 1)
        for k in interior:
            r_ = 1.0 / sig[k]
            rA[k] = Upp_f[k] - fUpp(
                r_, U[k], V[k], W[k], Up[k], Vp[k], Wp[k], Bphys, alpha
            )
            rM[k] = Vpp_f[k] - fVpp(
                r_, U[k], V[k], W[k], Up[k], Vp[k], Wp[k], Bphys, alpha
            )
            rN[k] = Wpp_f[k] - fWpp(
                r_, U[k], V[k], W[k], Up[k], Vp[k], Wp[k], Bphys, alpha
            )
        # BCs
        efac = np.exp(-4.0 * M[ihor])  # e^{-4 V(1)}, V(1)=M(1)
        rA[ihor] = A[ihor]  # A(1) = 0
        rM[ihor] = Ms[ihor] - (beta / 6.0) * efac  # M'(1) = (beta/6) e^{-4M(1)}
        rN[ihor] = Ns[ihor] + (beta / 12.0) * efac  # N'(1) = -(beta/12) e^{-4M(1)}
        rA[ibnd] = As[ihor] - As_h  # A'(1) = -4 (place at bnd row)
        rM[ibnd] = M[ibnd]  # M(0) = 0
        rN[ibnd] = Nn[ibnd]  # N(0) = 0
        return np.concatenate([rA, rM, rN])

    vec = np.concatenate([A, M, Nn])
    n = len(vec)
    last = np.max(np.abs(residual(vec)))
    for it in range(maxit):
        F = residual(vec)
        nF = np.max(np.abs(F))
        if verbose:
            print(f"    newton it={it} |F|={nF:.3e}")
        if nF < tol or nF > 0.999 * last and it > 2:  # converged or stalled
            last = nF
            break
        last = nF
        # numerical Jacobian (columns); step scaled per-variable
        J = np.empty((n, n))
        h = 1e-7
        for j in range(n):
            dv = np.zeros(n)
            dv[j] = h * (1.0 + abs(vec[j]))
            J[:, j] = (residual(vec + dv) - F) / dv[j]
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -F, rcond=None)[0]
        # damped update
        lam = 1.0
        while lam > 1e-4:
            trial = vec + lam * step
            if np.max(np.abs(residual(trial))) < nF:
                break
            lam *= 0.5
        vec = vec + lam * step
    A, M, Nn = vec[: N + 1], vec[N + 1 : 2 * (N + 1)], vec[2 * (N + 1) :]
    bg = Background(
        beta,
        alpha,
        N,
        A,
        M,
        Nn,
        Bphys=Bphys,
        uinf=None,
        vinf=1.0,
        winf=1.0,
        method="colloc",
        newton_res=last,
    )
    bg.accepted = bool(last < max(tol, residual_floor(N)))
    bg.res_floor = residual_floor(N)
    if require and not bg.accepted:
        raise NotConverged(
            f"solve_colloc(beta={beta:g}, N={N}): residual {last:.2e} above "
            f"max(tol={tol:g}, floor={bg.res_floor:.1e})"
        )
    return bg


# ===========================================================================
# Background object: Chebyshev representation + callables + thermodynamics
# ===========================================================================
class Background:
    """Boundary-normalised fixed-T magnetic brane.  Canonical downstream
    interface: profiles U,V,W and derivatives as callables of r (or sigma),
    from a Chebyshev representation of (A,M,N) on sigma in [0,1]."""

    def __init__(
        self, beta, alpha, N, A, M, Nn, Bphys, uinf, vinf, winf, method, **extra
    ):
        self.beta = beta
        self.alpha = alpha
        self.N = N
        self.Bphys = Bphys
        self.uinf, self.vinf, self.winf = uinf, vinf, winf
        self.method = method
        self.extra = extra
        D, xs = cheb(N)
        self.sig = (xs + 1) / 2.0
        self.D = 2.0 * D
        self.A = np.asarray(A)
        self.M = np.asarray(M)
        self.Nn = np.asarray(Nn)
        self.As = self.D @ self.A
        self.Ms = self.D @ self.M
        self.Ns = self.D @ self.Nn

    # ---- barycentric evaluation on the Chebyshev-Lobatto sigma nodes ----
    def _bary(self, vals, sg):
        xs = 2.0 * self.sig - 1.0
        xq = 2.0 * np.atleast_1d(np.asarray(sg, float)) - 1.0
        n = len(xs) - 1
        w = np.ones(n + 1)
        w[1::2] = -1.0
        w[0] *= 0.5
        w[-1] *= 0.5
        out = np.empty(len(xq))
        for i, xv in enumerate(xq):
            diff = xv - xs
            hit = np.where(np.abs(diff) < 1e-14)[0]
            if len(hit):
                out[i] = vals[hit[0]]
            else:
                tmp = w / diff
                out[i] = tmp @ vals / tmp.sum()
        return out

    def fields_sigma(self, sg):
        """Return dict of U,V,W,Up,Vp,Wp (derivatives in r) at sigma=sg."""
        sg = np.atleast_1d(np.asarray(sg, float))
        A = self._bary(self.A, sg)
        M = self._bary(self.M, sg)
        Nn = self._bary(self.Nn, sg)
        As = self._bary(self.As, sg)
        Ms = self._bary(self.Ms, sg)
        Ns = self._bary(self.Ns, sg)
        U = A / sg**2
        Up = 2 * A / sg - As
        V = M - np.log(sg)
        Vp = -(sg**2) * Ms + sg
        W = Nn - np.log(sg)
        Wp = -(sg**2) * Ns + sg
        return dict(
            U=U, V=V, W=W, Up=Up, Vp=Vp, Wp=Wp, A=A, M=M, Nn=Nn, As=As, Ms=Ms, Ns=Ns
        )

    def fields_r(self, r):
        return self.fields_sigma(1.0 / np.atleast_1d(np.asarray(r, float)))

    def cheb_coeffs(self, which="A"):
        vals = getattr(self, {"A": "A", "M": "M", "N": "Nn"}[which])
        N = self.N
        v = np.concatenate([vals, vals[-2:0:-1]])
        a = np.real(np.fft.fft(v)) / N
        return np.concatenate([[a[0] / 2], a[1:N], [a[N] / 2]])

    # ---- thermodynamics (fixed-T frame) ----
    def temperature(self):
        # U'(1) with sig=1: Up = 2A - As at sig=1; A(1)=0 => Up(1) = -As(1)
        ih = int(np.argmin(np.abs(self.sig - 1.0)))
        Up1 = 2 * self.A[ih] / self.sig[ih] - self.As[ih]
        return Up1 / (4 * PI)

    def entropy_4G5(self):
        """4 G5 s = e^{2V(1)+W(1)} in the normalised frame (horizon area
        density per unit proper boundary volume)."""
        ih = int(np.argmin(np.abs(self.sig - 1.0)))
        V1 = self.M[ih] - np.log(self.sig[ih])
        W1 = self.Nn[ih] - np.log(self.sig[ih])
        return np.exp(2 * V1 + W1)

    def entropy_shift(self):
        """delta s / s0 |_T = 4G5 s / (4G5 s0) - 1, s0 = (pi T)^3 area."""
        s = self.entropy_4G5()
        s0 = (PI * self.temperature()) ** 3
        return s / s0 - 1.0

    def constraint_drift(self):
        """Max |Hamiltonian constraint| over interior sigma nodes."""
        f = self.fields_sigma(self.sig[1:-1])
        r = 1.0 / self.sig[1:-1]
        vals = fcon(
            r, f["U"], f["V"], f["W"], f["Up"], f["Vp"], f["Wp"], self.Bphys, self.alpha
        )
        return np.max(np.abs(vals))

    def ode_residual(self):
        """Max reduced-ODE residual (U''-fUpp etc.) over interior nodes."""
        sig = self.sig[1:-1]
        f = self.fields_sigma(sig)
        As = f["As"]
        Ms = f["Ms"]
        Ns = f["Ns"]
        # need 2nd sigma-derivatives -> use dense D2 on stored nodal values
        D2 = self.D @ self.D
        Ass = (D2 @ self.A)[1:-1]
        Mss = (D2 @ self.M)[1:-1]
        Nss = (D2 @ self.Nn)[1:-1]
        Upp_f = sig**2 * Ass - 2 * sig * As + 2 * f["A"]
        Vpp_f = sig**4 * Mss + 2 * sig**3 * Ms - sig**2
        Wpp_f = sig**4 * Nss + 2 * sig**3 * Ns - sig**2
        r = 1.0 / sig
        rA = Upp_f - fUpp(
            r, f["U"], f["V"], f["W"], f["Up"], f["Vp"], f["Wp"], self.Bphys, self.alpha
        )
        rM = Vpp_f - fVpp(
            r, f["U"], f["V"], f["W"], f["Up"], f["Vp"], f["Wp"], self.Bphys, self.alpha
        )
        rN = Wpp_f - fWpp(
            r, f["U"], f["V"], f["W"], f["Up"], f["Vp"], f["Wp"], self.Bphys, self.alpha
        )
        return max(np.max(np.abs(rA)), np.max(np.abs(rM)), np.max(np.abs(rN)))

    def Id_horizon(self):
        """Gauge-invariant I_d(u_H) = (H_xx - H_zz)(horizon) = 2(V-W)(1)."""
        ih = int(np.argmin(np.abs(self.sig - 1.0)))
        return 2 * (self.M[ih] - self.Nn[ih])

    @classmethod
    def from_nodal(cls, beta, alpha, N, A, M, Nn, Bphys):
        """Reconstruct a Background from stored Chebyshev-Lobatto nodal values
        (downstream loader).  See load_grid / dk_background_grid.npz."""
        return cls(
            beta,
            alpha,
            N,
            A,
            M,
            Nn,
            Bphys=Bphys,
            uinf=None,
            vinf=None,
            winf=None,
            method="grid",
        )


def load_grid(path=None):
    """Load the saved grid.  Returns (data_dict, get) where get(i) returns the
    i-th Background and betas = data['betas'].  Downstream: for an arbitrary
    beta not on the grid, call solve_colloc(beta) directly (spectrally exact);
    the grid is a convenience cache of pre-solved backgrounds."""
    path = paths.data("dk_background_grid.npz") if path is None else path
    d = dict(np.load(path, allow_pickle=True))
    N = int(d["N"])
    alpha = float(d["alpha"])

    def get(i):
        return Background.from_nodal(
            float(d["betas"][i]),
            alpha,
            N,
            d["A"][i],
            d["M"][i],
            d["Nn"][i],
            float(d["Bphys"][i]),
        )

    return d, get


def profile_maxdiff(bg1, bg2, ng=400):
    """Max abs difference of U,V,W and derivatives between two backgrounds,
    sampled on a common sigma grid (interior)."""
    sg = np.linspace(0.02, 0.999, ng)
    f1 = bg1.fields_sigma(sg)
    f2 = bg2.fields_sigma(sg)
    out = {}
    for key in ("U", "V", "W", "Up", "Vp", "Wp"):
        # U blows up near boundary; compare A=sig^2 U scale for U
        if key == "U":
            d = np.max(np.abs((f1[key] - f2[key]) * sg**2))
        elif key == "Up":
            d = np.max(np.abs((f1[key] - f2[key]) * sg))
        else:
            d = np.max(np.abs(f1[key] - f2[key]))
        out[key] = d
    return out


# ===========================================================================
# ANCHORS
# ===========================================================================
def anchor_ads_schwarzschild():
    """beta -> 0: exact AdS5-Schwarzschild U = r^2(1 - 1/r^4), V = W = ln r,
    i.e. A = 1 - sigma^4, M = N = 0.  Quantify both methods at beta = 0."""
    print("== anchor [beta->0]: AdS5-Schwarzschild ==")
    bgs = solve_shoot(0.0, N=80)
    bgc = solve_colloc(0.0, N=64, tol=1e-13, require=True)
    sg = np.linspace(0.0, 1.0, 400)
    exactA = 1.0 - sg**4
    for name, bg in [("shoot", bgs), ("colloc", bgc)]:
        A = bg._bary(bg.A, sg)
        eA = np.max(np.abs(A - exactA))
        eM = np.max(np.abs(bg._bary(bg.M, sg)))
        eN = np.max(np.abs(bg._bary(bg.Nn, sg)))
        print(
            f"  {name}: max|A-(1-sig^4)|={eA:.2e}  max|M|={eM:.2e}  max|N|={eN:.2e}  "
            f"T={bg.temperature():.12f}"
        )
        tol = 1e-12 if name == "colloc" else 3e-9
        check(
            f"beta = 0 {name} reproduces AdS5-Schwarzschild to {tol:g}",
            max(eA, eM, eN) < tol and abs(bg.temperature() - 1 / PI) < 1e-9,
        )
    return bgs, bgc


def anchor_small_beta_slopes(betas=(0.01, 0.02, 0.04, 0.06, 0.08)):
    """delta s/s0|_T -> (1/4) beta and I_d(horizon) -> (pi^2/48) beta as
    beta -> 0 (g0_thermo exact anchors 1/4 and pi^2/48 per alpha B^2).
    Quadratic extrapolation in beta removes the O(beta) curvature of the ratio."""
    print("== anchor [small beta]: entropy slope 1/4, I_d slope pi^2/48 ==")
    ds, idh = [], []
    for b in betas:
        bg = solve_colloc(b, N=64, tol=1e-13, require=True)
        ds.append(bg.entropy_shift() / b)
        idh.append(bg.Id_horizon() / b)
    bb = np.array(betas)
    A = np.stack([np.ones_like(bb), bb, bb**2], axis=1)  # quadratic in beta
    ds0 = np.linalg.lstsq(A, np.array(ds), rcond=None)[0][0]
    id0 = np.linalg.lstsq(A, np.array(idh), rcond=None)[0][0]
    print(
        f"  ds/s0 per beta -> {ds0:.9f}   (exact 1/4 = 0.250000000, err {abs(ds0 - 0.25):.1e})"
    )
    print(
        f"  I_d(hor) per beta -> {id0:.9f}   (exact pi^2/48 = {np.pi**2 / 48:.9f}, "
        f"err {abs(id0 - np.pi**2 / 48):.1e})"
    )
    # the quadratic extrapolation leaves ~1e-6 (the O(beta^3) term)
    check("entropy slope 1/4 to 3e-6", abs(ds0 - 0.25) < 3e-6)
    check("I_d slope pi^2/48 to 3e-6", abs(id0 - np.pi**2 / 48) < 3e-6)
    return ds0, id0


def anchor_dhoker_kraus():
    """Reproduce dhoker_kraus_compare.py stage [6]: the nonlinear relation
    4 G5 s - (pi T)^3 = B^2/(2 pi T), Richardson-extrapolated in b^2, at its
    b = 0.05, 0.1, 0.2 points (their alpha = 2).  Both our methods.  Paper app. A.1
    quotes the same identity, from a nonlinear solve in D'Hoker-Kraus's coordinates
    (arXiv:0908.3875), to a residual of 9e-5."""
    print(
        "== anchor [D'Hoker-Kraus]: 4G5 s - (piT)^3 = B^2/(2 pi T) at b=0.05,0.1,0.2 =="
    )
    res = []
    for b in (0.05, 0.1, 0.2):
        rmax = _rmax_for_beta(2 * b**2)
        sol = _shoot_raw(b, alpha=2.0, rmax=max(rmax, 4000.0))
        uinf, vinf, winf = _leading_coeffs(sol, max(rmax, 4000.0))
        T = 4.0 / (4 * PI * np.sqrt(uinf))
        Bphys = b / vinf
        s4G = 1.0 / (vinf * np.sqrt(winf))
        beta = 2 * Bphys**2
        # cross-method: collocation at the same beta
        bc = solve_colloc(beta, N=64, tol=1e-12, require=True)
        s4G_c = bc.entropy_4G5()
        ratio = (s4G - (PI * T) ** 3) / Bphys**2
        res.append((b, T, Bphys, s4G, ratio, beta, s4G_c))
        print(
            f"  b={b}: T={T:.8f} B={Bphys:.6f} 4G5s(shoot)={s4G:.8f} "
            f"4G5s(colloc)={s4G_c:.8f} diff={abs(s4G - s4G_c):.1e} "
            f"[s-(piT)^3]/B^2={ratio:.6f} (target 1/(2piT)={1 / (2 * PI * T):.6f})"
        )
    b2 = np.array([r[0] ** 2 for r in res])
    rat = np.array([r[4] for r in res])
    targ = np.array([1 / (2 * PI * r[1]) for r in res])
    off = np.polyfit(b2, rat - targ, 1)[1]
    print(f"  Richardson offset (should be ~0): {off:.2e}")
    check(
        "D'Hoker-Kraus entropy: shooting vs collocation to 1e-8",
        max(abs(r[3] - r[6]) for r in res) < 1e-8,
    )
    check(
        "D'Hoker-Kraus relation 4G5 s - (pi T)^3 = B^2/(2 pi T): offset < 1e-5",
        abs(off) < 1e-5,
    )
    return res, off


# ===========================================================================
# TWO-METHOD AGREEMENT + RANGE STUDY + GRID SAVE
# ===========================================================================
def two_method_agreement(
    betas=(0.1, 1.0, 5.0, 15.0, 30.0, 40.0, 50.0), N_sh=80, N_co=64
):
    """Shooting vs collocation, beta <= 50.  The profiles (U as sigma^2 U, V, W)
    and T, I_d agree to 1e-9; their r-derivatives (sigma U', V', W') to 5e-9 and
    the entropy shift, a ratio of horizon data, to 5e-9 (measured: 3.8e-9 and
    3.5e-9 at beta = 50, growing with beta).  The Hamiltonian constraint, which
    neither solver imposes in the interior, is evaluated on the collocation
    solution."""
    print("== two-method agreement (shoot vs colloc) ==")
    print(
        f"  {'beta':>7} {'|dU|A':>10} {'|dV|':>10} {'|dW|':>10} "
        f"{'dT':>10} {'d(ds/s0)':>11} {'dId':>11} {'constraint':>11}"
    )
    rows = []
    for b in betas:
        bs = solve_shoot(b, N=N_sh)
        bc = solve_colloc(b, N=N_co, tol=1e-12, maxit=40, require=True)
        d = profile_maxdiff(bs, bc)
        dT = abs(bs.temperature() - bc.temperature())
        dds = abs(bs.entropy_shift() - bc.entropy_shift())
        dId = abs(bs.Id_horizon() - bc.Id_horizon())
        con = bc.constraint_drift()
        print(
            f"  {b:7.2f} {d['U']:10.2e} {d['V']:10.2e} {d['W']:10.2e} "
            f"{dT:10.2e} {dds:11.2e} {dId:11.2e} {con:11.2e}"
        )
        rows.append((b, d, dT, dds, dId, con))
    prof = max(max(r[1][k] for k in ("U", "V", "W")) for r in rows)
    dprof = max(max(r[1][k] for k in ("Up", "Vp", "Wp")) for r in rows)
    thermo = max(max(r[2], r[4]) for r in rows)
    ent = max(r[3] for r in rows)
    con = max(r[5] for r in rows)
    print(
        f"  worst to beta = {max(betas):g}: profiles {prof:.1e}, derivatives "
        f"{dprof:.1e}, T and I_d {thermo:.1e}, ds/s0 {ent:.1e}, constraint {con:.1e}"
    )
    check(
        f"two methods: profile derivatives agree to 5e-9 for beta <= {max(betas):g}",
        dprof < 5e-9,
    )
    check(
        f"two methods: profiles agree to 1e-9 for beta <= {max(betas):g}", prof < 1e-9
    )
    check(
        f"two methods: T and I_d agree to 1e-9 for beta <= {max(betas):g}",
        thermo < 1e-9,
    )
    check(f"two methods: ds/s0 agrees to 5e-9 for beta <= {max(betas):g}", ent < 5e-9)
    check(f"Hamiltonian constraint below 1e-8 for beta <= {max(betas):g}", con < 1e-8)
    return rows


def cheb_decay_demo(beta=5.0, N=96):
    print(f"== Chebyshev coefficient decay (beta={beta}, colloc N={N}) ==")
    bg = solve_colloc(beta, N=N, tol=1e-12, maxit=40, require=True)
    for which in ("A", "M", "N"):
        c = np.abs(bg.cheb_coeffs(which))
        # report magnitude at a few n
        idx = [8, 16, 32, 48, min(N, 64)]
        s = "  ".join(f"c[{i}]={c[i]:.1e}" for i in idx if i <= N)
        print(f"  |coeff({which})|: {s}   tail(last4)={np.max(c[-4:]):.1e}")
        check(
            f"Chebyshev tail of {which} below 1e-14 at N = {N}", np.max(c[-4:]) < 1e-14
        )
    return bg


def range_study(betas):
    print("== range study ==")
    out = []
    for b in betas:
        t0 = time.time()
        bc = solve_colloc(b, N=72, tol=1e-11, maxit=50)
        nr = bc.extra["newton_res"]
        con = bc.constraint_drift()
        ok = (
            bc.accepted
            and (bc.ode_residual() < 1e-8)
            and abs(bc.A[-1] - 1) < 1e-6
            and con < 1e-8
        )
        out.append(
            (b, ok, nr, bc.ode_residual(), bc.temperature(), bc.entropy_shift(), con)
        )
        print(
            f"  beta={b:7.2f}: colloc newton={nr:.1e} (floor {bc.res_floor:.0e}) "
            f"ode_res={bc.ode_residual():.1e} constraint={con:.1e} "
            f"A(0)-1={bc.A[-1] - 1:.1e} ds/s0={bc.entropy_shift():.5f} "
            f"[{'OK' if ok else 'DEGRADED'}] {time.time() - t0:.1f}s"
        )
    check(
        f"range study: every beta in [{min(betas):g}, {max(betas):g}] converged",
        all(r[1] for r in out),
    )
    return out


def build_grid(betas, N=72):
    """Solve a grid of collocation backgrounds for downstream consumers.
    Returns the arrays; write_grid saves them (main does so only after every
    check has passed)."""
    print(f"== building grid ({len(betas)} backgrounds, N={N}) ==")
    D, xs = cheb(N)
    sig = (xs + 1) / 2.0
    As, Ms, Ns, Bph, Temp, S4G, DS, IDH = [], [], [], [], [], [], [], []
    for b in betas:
        bc = solve_colloc(b, N=N, tol=1e-11, maxit=50, require=True)
        As.append(bc.A)
        Ms.append(bc.M)
        Ns.append(bc.Nn)
        Bph.append(bc.Bphys)
        Temp.append(bc.temperature())
        S4G.append(bc.entropy_4G5())
        DS.append(bc.entropy_shift())
        IDH.append(bc.Id_horizon())
    return dict(
        betas=np.array(betas),
        sigma=sig,
        N=N,
        A=np.array(As),
        M=np.array(Ms),
        Nn=np.array(Ns),
        Bphys=np.array(Bph),
        T=np.array(Temp),
        s4G=np.array(S4G),
        ds_over_s0=np.array(DS),
        Id_horizon=np.array(IDH),
        alpha=2.0,
        readme=(
            "A=sigma^2 U, M=V+ln sigma, N=W+ln sigma on Chebyshev-Lobatto "
            "sigma=(cos(pi j/N)+1)/2 in [0,1]; sigma=1 horizon, sigma=0 boundary; "
            "U=A/sigma^2, V=M-ln sigma, W=N-ln sigma; ds^2=-U dt^2+dr^2/U"
            "+e^{2V}(dx^2+dy^2)+e^{2W}dz^2, r=1/sigma, r_p=1, T=1/pi, "
            "F_xy=Bphys, beta=alpha*Bphys^2 (alpha=2)."
        ),
    )


def write_grid(grid, path=None):
    path = paths.data("dk_background_grid.npz") if path is None else path
    np.savez(path, **grid)
    print(f"  saved {path}")


def save_grid(betas, path=None, N=72):
    """Solve and save a grid in one step (for downstream use; main itself
    builds first and writes only after its checks pass)."""
    write_grid(build_grid(betas, N=N), path)


def main():
    t0 = time.time()
    print("#" * 74)
    print("# D'Hoker-Kraus magnetic black-brane background solver ")
    print("#" * 74)
    print(
        "\nFRAME: r_p=1, T=1/pi (U'(1)=4), boundary-normalised (U,e^{2V},e^{2W}~r^2),"
    )
    print("       F_xy=B_phys, family labelled by beta=alpha*B_phys^2.\n")
    print("== symbolic verification ==")
    for name, ok in verify_symbolics(verbose=False).items():
        check(name, ok)
    print()
    anchor_ads_schwarzschild()
    print()
    anchor_small_beta_slopes()
    print()
    anchor_dhoker_kraus()
    print()
    two_method_agreement()
    print()
    cheb_decay_demo()
    print()
    range_study([0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0])
    print()
    # a parameter grid: one value per line would hide its shape
    # fmt: off
    grid_betas = [0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                  8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 30.0]
    # fmt: on
    grid = build_grid(grid_betas)
    print(f"\nTOTAL runtime {time.time() - t0:.1f}s")
    nfail = sum(1 for v in CHECKS.values() if not v)
    print(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    for k, v in CHECKS.items():
        if not v:
            print(f"  FAILED: {k}")
    if nfail:
        print("cache not written: a check failed")
        raise SystemExit(1)
    write_grid(grid)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
