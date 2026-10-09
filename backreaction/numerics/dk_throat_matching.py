"""Matched asymptotics for the approach to alpha_* = 2/3: the marginality
coefficient of the throat ladder is DERIVED, not fitted (paper sec. 4.6).

A fit of nu = sqrt(b_h - 1) = A/(ln beta + d) along the beta ladder gives
A = 7.45, while counting half-oscillations of the marginal mode in the throat
predicts A = 4 pi.  Both are right about different things, and this module
shows why.

THE MATCHING.  For h << rho << r_c the marginal mode is the AdS3 log-oscillation
rho^{-1} cos(nu ln rho + const).  Its phase is fixed at both ends:

  * horizon end -- EXACT, from the paper app. C.4 hypergeometric y_H = z^a 2F1(a,a;1;1-z),
    a = (1 + i nu)/2, z = h^2/rho^2:  rho w ~ cos(nu ln(rho/h) - chi(nu)),
        chi(nu) = arg Gamma(-i nu) - 2 arg Gamma((1 - i nu)/2)  ->  pi/2 - nu ln 4 ;
  * UV end -- from the T = 0 brane (the beta -> oo outer region in scaled
    variables, built here by integrating out of the AdS3 x R^2 fixed point along
    its irrelevant deformation, exponent p = sqrt(19/3) - 1 = 1.517, which is
    mu_+ - 1 = sqrt(1 + m_+) - 1 at lambda = 0, m_+(0) = 16/3, of paper app. C.4):  the c_0 = 0 solution reads, in the
    throat,  rho w ~ cos(nu ln(rho/rho_*) + phi_*(nu)),  rho_* the point where
    e^{2V} has doubled from its throat value -- a beta-independent function.

The ground state matches them with no extra node,

        nu ln(rho_*/h)  =  chi(nu) + phi_*(nu) ,                          (M)

and rho_*(beta) is read off each rung (ln(rho_*/h) -> (1/4) ln(beta/6) + 0.031).
(M) has NO free parameter.  It fixes the throat exponent, 1 + nu^2 =
sqrt(alpha_*/alpha); converted to the ladder's nu = sqrt(b_h - 1) with the exact
relation b_h = (1 + nu^2) sqrt(1 - eps/6), eps = 6 - beta e^{-4V} at the horizon,
it reproduces the ladder to 6e-7 at beta = 1e11 (error falling monotonically up
the ladder, from 5e-3 at beta = 1e4).

WHY THE FIT GIVES 7.45.  At nu = 0 the outer solution is rho^{-1}(a + b ln(rho/rho_*))
with an anomalously SMALL log coefficient, b/a = -0.035.  So phi_*(nu) =
pi/2 - arctan(nu |a/b|) with |a/b| ~ 29: it sits near 0.24 for every nu the
ladder reaches (0.25 <= nu <= 0.6) and only turns over to pi/2 for
nu << 0.035, i.e. ln beta >> 400.  Hence

    accessible regime:   nu ln(rho_*/h) -> pi/2 + 0.24 - nu ln 4   =>  A_eff ~ 7.3-7.6
    true asymptote:      nu ln(rho_*/h) -> pi                       =>  A = 4 pi.

The fitted A = 7.45 is therefore an effective slope over the accessible window,
not the asymptotic one: asymptotically the throat holds exactly one
half-oscillation of the marginal mode (A -> 4 pi), the standard holographic-BKT
statement, reached logarithmically slowly (paper sec. 4.6).  At every FINITE
beta the throat holds less than one half-oscillation; that is a statement about
each rung, not about the limit.

CHECKS
------
[1] T = 0 brane: e^{2V} -> 1/3 (= sqrt(alpha/6) at B = 1) at the fixed point,
    U = 3 e^{2W} to roundoff along the whole flow (zero horizon flux), AdS5 at
    the far end (approached as O(r_c/rho), a radial shift);  U x constraint < 1e-8.
[2] chi(nu) -> pi/2 - nu ln 4 as nu -> 0 (matches the elliptic-K log of paper app. C.4).
[3] phi_*(nu) from the T = 0 outer solution is fitted by cos to < 1e-9; its
    small-nu form pi/2 - arctan(nu |a/b|) with a, b read off the nu = 0 solution.
[4] (M) against the throat ladder (nu, eps and rho_* from the ladder
    backgrounds, rebuilt here), compared in one frame through
    b_h = (1 + nu^2) sqrt(1 - eps/6): relative error < 1e-6 at the top rung,
    decreasing monotonically.
[5] the effective slope d ln beta / d(1/nu) from (M) equals the paper sec. 4.6 fit
    (7.45) at ln beta ~ 25 and tends to 4 pi.

Run:  uv run python -m backreaction.numerics.dk_throat_matching   (~5 min: rebuilds
      the ladder;  --quick stops at beta = 1e7, ~2 min)
"""

import sys
import time

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, curve_fit

from backreaction import paths
from backreaction.numerics import dk_background as dk
from backreaction.numerics import dk_throat as dt

T0 = time.time()
CHECKS = {}
CACHE = paths.data("dk_throat_matching.npz")
H_LADDER = 2.0 / 3.0  # BTZ horizon radius in the fixed-T frame (paper app. C.4)


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# the T = 0 brane at alpha = 2/3, B = 1 (so b = 1 exactly): integrate out of the
# fixed point along its irrelevant deformation
# ---------------------------------------------------------------------------
_S = dk._SYM["_sym"]
_rr, _U, _V, _W, _Up, _Vp, _Wp, _B, _al, _Upp, _Vpp, _Wpp = _S["symbols"]
_args = (_U, _V, _W, _Up, _Vp, _Wp, _B, _al)
fU = sp.lambdify(_args, _S["Upp"], "numpy")
fV = sp.lambdify(_args, _S["Vpp"], "numpy")
fW = sp.lambdify(_args, _S["Wpp"], "numpy")
fcon = sp.lambdify(_args, _S["constraint"], "numpy")
ALPHA, BFIELD = 2.0 / 3.0, 1.0


def irrelevant_mode():
    """Linearise the reduced system about the fixed point U = 3 rho^2,
    e^{2V} = 1/3, e^{2W} = rho^2: V = V0 + eps rho^p, W = ln rho + eps c rho^p,
    U = 3 rho^2 (1 + eps cu rho^p).  Returns (p, c, cu) for the growing mode."""
    rho, eps, p, c, cu = sp.symbols("rho epsilon p c cu")
    al, Bs = sp.Rational(2, 3), sp.Integer(1)
    Vl = sp.log(sp.sqrt(al / 6)) / 2 + eps * rho**p
    Wl = sp.log(rho) + eps * c * rho**p
    Ul = 3 * rho**2 * (1 + eps * cu * rho**p)
    sub = {
        _B: Bs,
        _al: al,
        _U: Ul,
        _V: Vl,
        _W: Wl,
        _Up: sp.diff(Ul, rho),
        _Vp: sp.diff(Vl, rho),
        _Wp: sp.diff(Wl, rho),
    }

    def lin(e):
        return sp.simplify(sp.diff(e.subs(sub), eps).subs(eps, 0))

    eqs = [
        sp.simplify(
            (lin(_S["Vpp"]) - sp.diff(Vl, rho, 2).diff(eps).subs(eps, 0))
            / rho ** (p - 2)
        ),
        sp.simplify(
            (lin(_S["Wpp"]) - sp.diff(Wl, rho, 2).diff(eps).subs(eps, 0))
            / rho ** (p - 2)
        ),
        sp.simplify(
            (lin(_S["Upp"]) - sp.diff(Ul, rho, 2).diff(eps).subs(eps, 0)) / rho**p
        ),
        sp.simplify(lin(_S["constraint"]) / rho ** (p - 2)),
    ]
    sols = sp.solve(eqs, [p, c, cu], dict=True)
    grow = [s for s in sols if float(s[p]) > 0]
    assert len(grow) == 1, sols
    s = grow[0]
    return float(s[p]), float(s[c]), float(s[cu]), s[p]


class _IRExtended:
    """Dense output of the integrated brane for rho >= rho0, and below rho0 the
    fixed point plus the linear irrelevant mode, whose neglected O(e^2 rho^2p)
    terms are below 1e-15 there.  Same .sol(r) interface as solve_ivp."""

    def __init__(self, ivp, rho0, ir):
        self.ivp, self.rho0, self.ir = ivp, rho0, ir

    def sol(self, r):
        r_arr = np.asarray(r, dtype=float)
        if r_arr.ndim == 0:
            return self.ir(r_arr) if r_arr < self.rho0 else self.ivp.sol(r_arr)
        out = np.empty((6, r_arr.size))
        lo = r_arr < self.rho0
        if lo.any():
            out[:, lo] = self.ir(r_arr[lo])
        if (~lo).any():
            out[:, ~lo] = self.ivp.sol(r_arr[~lo])
        return out


def t0_brane(amp=1e-7, delta0=1e-8, rho_max=1e8, rtol=1e-12):
    """The brane V = V0 + amp rho^p + ... out of the fixed point.  The seed sits
    where the deformation amp rho0^p equals delta0, well above the integrator's
    resolution relative to V0 = -0.549 (seeding at a fixed rho0 = 1e-9 put it at
    2e-14, below that resolution, so the brane actually built was seeded by
    integrator error and rho_* moved by 15% with rtol).  amp only sets the scale:
    rho_* = 0.6039 amp^(-1/p), here ~2.6e4, so the throat window 1e-8 <= rho <= 1e-4
    of [3] lies 8-12 decades below rho_*; below the seed the brane is the
    linearised series (`_IRExtended`)."""
    p, c, cu, p_sym = irrelevant_mode()
    v0 = np.sqrt(ALPHA / 6)
    e = amp
    rho0 = (delta0 / e) ** (1 / p)

    def ir(r):
        d = e * r**p
        return np.array(
            [
                3 * r**2 * (1 + cu * d),
                np.log(v0) / 2 + d,
                np.log(r) + c * d,
                6 * r * (1 + cu * d) + 3 * r * cu * p * d,
                p * d / r,
                1 / r + c * p * d / r,
            ]
        )

    y0 = list(ir(rho0))

    def rhs(r, y):
        U, V, W, a, b_, cc = y
        return [
            a,
            b_,
            cc,
            fU(U, V, W, a, b_, cc, BFIELD, ALPHA),
            fV(U, V, W, a, b_, cc, BFIELD, ALPHA),
            fW(U, V, W, a, b_, cc, BFIELD, ALPHA),
        ]

    sol = solve_ivp(
        rhs,
        [rho0, rho_max],
        y0,
        method="DOP853",
        rtol=rtol,
        atol=1e-16,
        dense_output=True,
    )
    return _IRExtended(sol, rho0, ir), p, p_sym


def brane_rho_star(sol):
    """rho_* of the T = 0 brane: e^{2V} has doubled from its throat value v."""
    v0 = np.sqrt(ALPHA / 6)
    return brentq(lambda r: np.exp(2 * sol.sol(r)[1]) - 2 * v0, 1e-6, 1e6)


def outer_mode(sol, Bm, rho_max=1e8, rho_min=1e-9):
    """c_0 = 0 solution of (U e^W w')' + Bm e^{W-2V} w = 0 on the T = 0 brane,
    integrated inward from the AdS5 end (w = rho^-2).  The start is ~4000 rho_*
    out: the extraction error is O(rho_*/rho_max), and starting at 1e7 instead
    moves b/a of [3] by 1.3e-6 relative."""

    def rhs(r, y):
        w, Pwp = y
        U, V, W = sol.sol(r)[:3]
        return [Pwp / (U * np.exp(W)), -Bm * np.exp(W - 2 * V) * w]

    U, V, W = sol.sol(rho_max)[:3]
    return solve_ivp(
        rhs,
        [rho_max, rho_min],
        [rho_max**-2, U * np.exp(W) * (-2 * rho_max**-3)],
        method="DOP853",
        rtol=1e-11,
        atol=1e-30,
        dense_output=True,
    )


XI_THROAT = np.linspace(np.log(1e-8), np.log(1e-4), 400)


def phi_star(sol, rho_star, nu, xi_thr=XI_THROAT):
    """phi_*(nu) of the T = 0 brane `sol`: the c_0 = 0 outer solution fitted in
    the throat window to cos(nu ln(rho/rho_*) + phi_*).  Returns (phi_*, the
    relative fit error)."""
    m = outer_mode(sol, 1 + nu**2)
    y = np.exp(xi_thr) * m.sol(np.exp(xi_thr))[0]

    def f(x, A, ph):
        return A * np.cos(nu * (x - np.log(rho_star)) + ph)

    pp, _ = curve_fit(f, xi_thr, y, p0=[y[0], 0.0])
    A, ph = pp
    if A < 0:
        ph = ph + np.pi
    ph = (ph + np.pi) % (2 * np.pi) - np.pi
    return ph, np.max(np.abs(f(xi_thr, *pp) - y)) / np.max(np.abs(y))


def nu_match(lnbeta, sol, rho_star, offset, bracket=(1e-3, 0.89)):
    """nu from the matching condition (M), nu L = chi(nu) + phi_*(nu) with
    L = (1/4) ln(beta/6) + offset, phi_* evaluated directly on the T = 0 brane
    (no interpolation: the cubic spline of the 45-node table is off by 2e-4
    at nu = 0.055)."""
    L = 0.25 * (lnbeta - np.log(6.0)) + offset
    return brentq(
        lambda n: n * L - chi(n) - phi_star(sol, rho_star, n)[0],
        *bracket,
        xtol=1e-14,
        rtol=1e-14,
    )


def dw_brane_phase(nus):
    """Second construction of the matching inputs (app. D.3, table 4): the T = 0 brane in the
    domain-wall gauge of t0_threshold.Flow (its own derivation of the flow
    equations in derivations/brane_flows and its own seed; DOP853 in both), with
    the throat coordinate R = r - r0 = int e^A d rho integrated as an ODE (a
    trapezoid rule here biases ln R by ~2e-5 in the throat, i.e. phi_* by
    2e-5 nu).  Returns (phi_* at each nu, b/a at nu = 0, R_*, c_V), with
    phi_* referenced at R_* (e^{2C} = 2v) and c_V = lim e^{2C}/R^2 =
    e^{2(c_inf - a_inf)}, A -> rho + a_inf, C -> rho + c_inf."""
    from backreaction.numerics import t0_threshold as T0T

    fl = T0T.Flow(4)
    rho_s = fl.rho_star()
    rho_uv = min(fl.rho_end, rho_s + 45.0)
    # below rho = 0 the flow is the fixed point plus the linear deformation,
    # e^A = e^{rho/l}(1 + O(eps)), so R(RHO_IR) = l e^{RHO_IR/l} to 1e-36
    Rsol = solve_ivp(
        lambda r, y: [np.exp(fl.fields(r)[0][0])],
        [T0T.RHO_IR, rho_uv],
        [fl.l * np.exp(T0T.RHO_IR / fl.l)],
        method="DOP853",
        rtol=1e-13,
        atol=1e-300,
        dense_output=True,
    )
    R_star = float(Rsol.sol(rho_s)[0])
    grid = np.linspace(-45.0, -12.0, 4000)  # throat: ln(R/R_*) in [-92, -35]
    Rg = Rsol.sol(grid)[0]
    xg = np.log(Rg / R_star)

    def threshold(Bm):
        th = T0T.mode(
            fl, Bm, rho_uv, T0T.RHO_IR, np.exp(-2 * rho_uv), -2 * np.exp(-2 * rho_uv)
        )
        return Rg * th.sol(grid)[0]

    bco, aco = np.polyfit(xg, threshold(1.0), 1)
    phis = []
    for nu in nus:
        y = threshold(1.0 + nu**2)

        def f(x, A, ph, nu=nu):
            return A * np.cos(nu * x + ph)

        (A, ph), _ = curve_fit(f, xg, y, p0=[y[0], 0.0])
        if A < 0:
            ph += np.pi
        phis.append((ph + np.pi) % (2 * np.pi) - np.pi)
    A_end, C_end = fl.fields(rho_uv)[:2, 0]
    c_V = float(np.exp(2 * ((C_end - rho_uv) - (A_end - rho_uv))))
    return np.array(phis), bco / aco, R_star, c_V


def offset_t0(rho_star, c_V, beta0=ALPHA * BFIELD**2):
    """The beta -> oo limit of ln(rho_*/h) - (1/4) ln(beta/6) from the T = 0
    brane alone (app. D.3).  Normalising the brane at the boundary (e^{2V}/rho^2 ->
    1) shifts V by -(1/2) ln c_V and beta_0 -> beta_0/c_V^2; the scaling
    rho -> lam rho, U -> lam^2 U, e^{2V}, e^{2W} -> lam^2 (.) keeps that
    normalisation and maps beta -> lam^4 beta, so the T = 0 brane at beta has
    ln rho_* = ln rho_*^{(0)} + (1/2) ln c_V + (1/4) ln(beta/beta_0)."""
    return (
        np.log(rho_star)
        + 0.5 * np.log(c_V)
        + 0.25 * np.log(6.0 / beta0)
        - np.log(H_LADDER)
    )


def chi(nu):
    """Exact horizon phase: rho w ~ cos(nu ln(rho/h) - chi)."""
    nu = mp.mpf(nu)
    return float(mp.arg(mp.gamma(-1j * nu) / mp.gamma((1 - 1j * nu) / 2) ** 2))


def main():
    quick = "--quick" in sys.argv

    log("[1] the T = 0 brane at alpha = 2/3 (B = 1, b = 1)")
    sol, p, p_sym = t0_brane()
    check(
        "[1] irrelevant exponent p = sqrt(19/3) - 1 (= mu_+ - 1 at lambda = 0, paper app. C.4)",
        sp.simplify(p_sym - (sp.sqrt(sp.Rational(19, 3)) - 1)) == 0,
        f"p = {p:.6f}",
    )
    rs = np.logspace(np.log10(sol.rho0), 7, 300)  # the integrated part only
    U, V, W, Up, Vp, Wp = sol.sol(rs)
    con = np.max(
        np.abs(U * fcon(U, V, W, Up, Vp, Wp, BFIELD, ALPHA))
    )  # U x constraint: O(1) terms
    flux = np.max(np.abs(U / np.exp(2 * W) - 3.0))
    check(
        "[1] constraint residual and U = 3 e^{2W} (zero horizon flux) along the flow",
        con < 1e-8 and flux < 1e-8,
        f"U*constraint {con:.1e}, |U/e^2W - 3| {flux:.1e}",
    )
    # the approach to AdS5 is O(r_c/rho) (a radial shift: U = rho^2 + c rho + ...), so test the
    # log-derivatives at the far end rather than the coefficients
    Uf, Vf, Wf, Upf, Vpf, Wpf = sol.sol(1e7)
    check(
        "[1] AdS5 at the far end: rho V', rho W' -> 1, U/rho^2 -> 1 to O(r_c/rho)",
        abs(1e7 * Vpf - 1) < 1e-2
        and abs(1e7 * Wpf - 1) < 1e-2
        and abs(Uf / 1e14 - 1) < 1e-2,
        f"rho V' = {1e7 * Vpf:.5f}, rho W' = {1e7 * Wpf:.5f}, U/rho^2 = {Uf / 1e14:.5f}",
    )
    v0 = np.sqrt(ALPHA / 6)
    rho_star = brane_rho_star(sol)
    log(f"      rho_* (e^2V = 2 v) = {rho_star:.6f}   (seed rho0 = {sol.rho0:.4f})")
    # the seeded deformation is what the integrator carries: (V - V0)/(amp rho^p)
    # stays 1 up to the O(amp rho^p) nonlinear correction
    rl = sol.rho0 * np.array([1.5, 3.0, 10.0])
    real_amp = (sol.sol(rl)[1] - np.log(v0) / 2) / (1e-7 * rl**p)
    check(
        "[1] the seeded deformation is resolved: (V - V0)/(amp rho^p) = 1 near the seed",
        np.max(np.abs(real_amp - 1)) < 1e-5,
        f"{', '.join(f'{x:.8f}' for x in real_amp)}",
    )
    # rho_* is a property of the brane, not of the integration: rebuild with
    # rtol x 10 and with the seed moved one decade in deformation
    rs_alt = [
        brane_rho_star(t0_brane(rtol=1e-11)[0]),
        brane_rho_star(t0_brane(delta0=1e-7)[0]),
    ]
    drs = max(abs(x / rho_star - 1) for x in rs_alt)
    check(
        "[1] rho_* stable under rtol 1e-12 -> 1e-11 and seed delta0 1e-8 -> 1e-7",
        drs < 1e-5,
        f"rho_* = {rho_star:.4f}, {rs_alt[0]:.4f}, {rs_alt[1]:.4f}; max rel. change {drs:.1e}",
    )

    log("[2] the horizon phase chi(nu)")
    nus = np.array([0.01, 0.05, 0.1])
    check(
        "[2] chi(nu) -> pi/2 - nu ln 4",
        np.max(np.abs([(chi(n) - (np.pi / 2 - n * np.log(4))) / n**3 for n in nus]))
        < 5,
        f"residual/nu^3 = {[(chi(n) - (np.pi / 2 - n * np.log(4))) / n**3 for n in nus]}",
    )

    log("[3] the UV phase phi_*(nu) from the T = 0 outer solution")
    xi_thr = XI_THROAT
    m0 = outer_mode(sol, 1.0)
    y0 = np.exp(xi_thr) * m0.sol(np.exp(xi_thr))[0]
    bcoef, acoef = np.polyfit(xi_thr - np.log(rho_star), y0, 1)
    lin_err = np.max(
        np.abs(np.polyval([bcoef, acoef], xi_thr - np.log(rho_star)) - y0)
    ) / abs(acoef)
    ratio = bcoef / acoef
    log(
        f"      nu = 0: rho w = a + b ln(rho/rho_*),  b/a = {ratio:.5f},  linear fit error {lin_err:.1e}"
    )
    check(
        "[3] nu = 0 outer solution is a + b ln(rho/rho_*) in the throat, "
        "b/a = -0.0350004 (rho_* where e^{2V} = 2 v)",
        lin_err < 1e-4 and round(ratio, 7) == -0.0350004,
        f"b/a = {ratio:.9f}",
    )
    # the same fit referenced at rho = 1 instead: the ratio is scale-dependent,
    # b/a -> b/(a - b ln rho_*), so the reference point is part of the number
    log(
        f"      referenced at rho = 1 (B = 1 chart) instead: b/a = "
        f"{ratio / (1 - ratio * np.log(rho_star)):.5f}"
    )

    nu_tab = np.concatenate([np.linspace(0.02, 0.2, 10), np.linspace(0.22, 0.9, 35)])
    phi_tab, fit_err = [], []
    for nu in nu_tab:
        ph, er = phi_star(sol, rho_star, nu, xi_thr)
        phi_tab.append(ph)
        fit_err.append(er)
    phi_tab = np.array(phi_tab)
    check(
        "[3] cos fit of the outer solution in the throat, all nu",
        max(fit_err) < 1e-8,
        f"worst {max(fit_err):.1e}",
    )
    small = nu_tab < 0.12
    pred_small = np.pi / 2 - np.arctan(nu_tab[small] * abs(acoef / bcoef))
    check(
        "[3] small-nu form phi_* = pi/2 - arctan(nu |a/b|)",
        np.max(np.abs(phi_tab[small] - pred_small)) < 0.05,
        f"|a/b| = {abs(acoef / bcoef):.2f}, worst dev {np.max(np.abs(phi_tab[small] - pred_small)):.3f}",
    )

    # The next order.  In the throat rho w = a cos(nu L) + b sin(nu L)/nu with
    # L = ln(rho/rho_*), a basis entire in nu^2, so a(nu^2), b(nu^2) are analytic
    # and tan phi_* = -(b/a)/nu = |b/a|_0/nu + k1 nu + O(nu^3), with
    # k1 = -d(b/a)/d(nu^2) at 0.  Method 1: central difference in nu^2 = +-h
    # (b_h below and above 1, linear least squares for a, b).  Method 2: the
    # cos-fitted phi_tab above at its two smallest nu, (tan phi - |b/a|/nu)/nu
    # extrapolated linearly in nu^2.
    def b_over_a(n2):
        mm = outer_mode(sol, 1 + n2)
        yy = np.exp(xi_thr) * mm.sol(np.exp(xi_thr))[0]
        nc = np.sqrt(complex(n2))
        L = xi_thr - np.log(rho_star)
        basis = np.vstack([np.real(np.cos(nc * L)), np.real(np.sin(nc * L) / nc)]).T
        (aa, bb), *_ = np.linalg.lstsq(basis, yy, rcond=None)
        return bb / aa

    k1_h = [-(b_over_a(h) - b_over_a(-h)) / (2 * h) for h in (3e-3, 1e-3)]
    k1 = k1_h[1] + (k1_h[1] - k1_h[0]) / 8  # Richardson, O(h^2) error
    kk = (np.tan(phi_tab[:2]) - abs(ratio) / nu_tab[:2]) / nu_tab[:2]
    k1_tab = kk[0] - (kk[1] - kk[0]) * nu_tab[0] ** 2 / (
        nu_tab[1] ** 2 - nu_tab[0] ** 2
    )
    check(
        "[3] O(nu) term of tan phi_*: k1 = -d(b/a)/d(nu^2) = 0.4988, two methods",
        abs(k1 - 0.49875) < 1e-4 and abs(k1_tab / k1 - 1) < 1e-3,
        f"finite difference {k1:.6f} (h = 3e-3, 1e-3: {k1_h[0]:.6f}, {k1_h[1]:.6f}); "
        f"from phi_tab {k1_tab:.6f}",
    )
    lead_rel = np.arctan(abs(ratio) / nu_tab) / phi_tab - 1
    two_rel = np.arctan(abs(ratio) / nu_tab + k1 * nu_tab) / phi_tab - 1
    nu10_lead = nu_tab[np.argmax(np.abs(lead_rel) > 0.1)]
    nu10_two = nu_tab[np.argmax(np.abs(two_rel) > 0.1)]
    log(
        f"      leading term arctan(|b/a|/nu) first misses phi_* by > 10% at "
        f"nu = {nu10_lead:.2f}; with the k1 nu term at nu = {nu10_two:.2f}"
    )
    check(
        "[3] the small-nu forms do not explain the plateau 0.25 <= nu <= 0.6",
        0.08 < nu10_lead < 0.11 and 0.3 < nu10_two <= 0.34,
        f"leading-term error at nu = 0.6: {lead_rel[np.argmin(np.abs(nu_tab - 0.6))]:+.2f}",
    )
    log(
        "      nu, phi_*: "
        + ", ".join(
            f"({n:.2f}, {p_:.4f})"
            for n, p_ in zip(nu_tab[::5], phi_tab[::5], strict=True)
        )
    )
    win = (nu_tab >= 0.25) & (nu_tab <= 0.6)
    log(
        f"      phi_* over the ladder window 0.25 <= nu <= 0.6: "
        f"{phi_tab[win].min():.3f} to {phi_tab[win].max():.3f}"
    )
    phi = CubicSpline(nu_tab, phi_tab)

    log("[3'] second construction: phi_* on the domain-wall-gauge T = 0 brane")
    win_nu = nu_tab[win]
    phi_dw, ratio_dw, R_star_dw, cV_dw = dw_brane_phase(win_nu)
    dphi = phi_dw - phi_tab[win]
    for n_, a_, b_ in zip(win_nu[::4], phi_tab[win][::4], phi_dw[::4], strict=True):
        log(
            f"      nu = {n_:.3f}: r chart {a_:+.9f}, DW gauge {b_:+.9f}, diff {b_ - a_:+.1e}"
        )
    check(
        "[3'] phi_* by the r chart and the DW gauge agree over 0.25 <= nu <= 0.6, "
        "and so does b/a",
        np.max(np.abs(dphi)) < 1e-8 and abs(ratio_dw / ratio - 1) < 1e-7,
        f"max |diff| = {np.max(np.abs(dphi)):.1e} over {win_nu.size} nodes; "
        f"b/a = {ratio_dw:.9f} (DW) vs {ratio:.9f}",
    )

    log("[4] the ladder: nu ln(rho_*/h) = chi(nu) + phi_*(nu), no free parameter")
    ladder = [(b_, n) for b_, n in dt.LADDER if b_ <= (1e7 if quick else np.inf)]
    from backreaction.numerics import bc_alpha as bca

    bg, rows, rec = None, [], []
    for beta_, N in ladder:
        for Ntry in (N, N + 48, N + 96):
            try:
                bg = dk.solve_colloc(
                    beta_,
                    N=Ntry,
                    tol=1e-10,
                    maxit=200,
                    guess=dt._regrid(bg, Ntry) if bg else None,
                    require=True,
                )
                break
            except dk.NotConverged:
                log(f"    beta={beta_:g} N={Ntry}: not accepted, retrying")
        else:
            raise dk.NotConverged(f"ladder background at beta = {beta_:g}")
        vals, _ = bca.bc_spectral_grid(bg, n_keep=2)
        fh = bg.fields_r(np.array([1.0 + 1e-9]))
        rows.append(
            dict(
                beta=beta_,
                b_h=float(vals[0] * dt.L3SQ * np.exp(-2 * fh["V"][0])),
                eps=float(6.0 - beta_ * np.exp(-4 * fh["V"][0])),
            )
        )
        r = np.exp(np.linspace(1e-3, np.log(60 * (beta_ / 6) ** 0.25), 6000))
        e2V = np.exp(2 * bg.fields_r(r)["V"])
        if e2V.max() < 2 * np.sqrt(beta_ / 6):
            rec.append(np.nan)
            continue
        rec.append(
            np.log((np.interp(2 * np.sqrt(beta_ / 6), e2V, r) - 1.0 / 3.0) / H_LADDER)
        )
    Lst = np.array(rec)
    beta = np.array([r_["beta"] for r_ in rows])
    nu_m = np.sqrt(np.array([r_["b_h"] for r_ in rows]) - 1)
    eps_l = np.array([r_["eps"] for r_ in rows])
    # (M) fixes the THROAT exponent, 1 + nu^2 = sqrt(alpha_*/alpha); the ladder's
    # nu = sqrt(b_h - 1) is the horizon value.  On the onset curve the two are
    # tied exactly by b_h = (1 + nu^2) sqrt(1 - eps/6), eps = 6 - beta e^{-4V}
    # at the horizon, so the prediction is converted with the rung's eps before
    # it is compared (the unconverted error, printed as "raw", is that O(eps)
    # frame difference, not a residual of (M)).
    nu_p, nu_ph = np.full_like(nu_m, np.nan), np.full_like(nu_m, np.nan)
    err, err_raw = np.full_like(nu_m, np.nan), np.full_like(nu_m, np.nan)
    log("      beta        ln(rho_*/h)  eps        nu_meas    nu_pred    rel.err   raw")
    for i, (b_, L) in enumerate(zip(beta, Lst, strict=True)):
        if b_ < 1e3 or not np.isfinite(L):
            continue
        nu_p[i] = brentq(lambda n, L=L: n * L - chi(n) - phi(n), 0.03, 0.89)
        nu_ph[i] = np.sqrt((1 + nu_p[i] ** 2) * np.sqrt(1 - eps_l[i] / 6) - 1)
        err[i] = abs(nu_ph[i] / nu_m[i] - 1)
        err_raw[i] = abs(nu_p[i] / nu_m[i] - 1)
        log(
            f"      {b_:9.3g}   {L:8.4f}    {eps_l[i]:.2e}   {nu_m[i]:.5f}   "
            f"{nu_ph[i]:.5f}   {err[i]:.1e}   {err_raw[i]:.1e}"
        )
    sel = np.isfinite(err)
    check(
        "[4] prediction error (one frame) decreases monotonically up the ladder",
        bool(np.all(np.diff(err[sel]) < 0)),
        f"{err[sel][0]:.1e} at beta = {beta[sel][0]:g} -> {err[sel][-1]:.1e} at "
        f"beta = {beta[sel][-1]:g}",
    )
    check(
        f"[4] top-rung error < {1e-6 if not quick else 2e-4:g}",
        err[sel][-1] < (1e-6 if not quick else 2e-4),
        f"{err[sel][-1]:.1e}",
    )
    off = Lst[sel] - 0.25 * np.log(beta[sel] / 6)
    log(f"      ln(rho_*/h) - (1/4) ln(beta/6) -> {off[-1]:.4f}")

    log("[4'] the offset from the T = 0 brane alone (scaling symmetries, no ladder)")
    # c_V = lim e^{2V}/rho^2; e^V = sqrt(c_V)(rho + s + O(1/rho)), so (e^V)' =
    # e^V V' -> sqrt(c_V) free of the radial shift s that rho V' -> 1 carries
    cV_r = [float((np.exp(sol.sol(x)[1]) * sol.sol(x)[4]) ** 2) for x in (1e6, 1e7)]
    off_t0 = offset_t0(rho_star, cV_r[1])
    off_t0_dw = offset_t0(R_star_dw, cV_dw)
    off_big = off[beta[sel] >= 1e6]
    log(
        f"      r chart: rho_* = {rho_star:.6f}, c_V = {cV_r[1]:.6e} (rho = 1e6: "
        f"{cV_r[0]:.6e}) -> offset {off_t0:.7f}"
    )
    log(
        f"      DW gauge: R_* = {R_star_dw:.6f}, c_V = {cV_dw:.6e} -> offset {off_t0_dw:.7f}"
    )
    log(
        "      ladder, beta >= 1e6: "
        + ", ".join(
            f"{b_:.0e}: {o_:.6f}"
            for b_, o_ in zip(beta[sel][beta[sel] >= 1e6], off_big, strict=True)
        )
    )
    check(
        "[4'] the T = 0 offset by both constructions, and the ladder decreases "
        "monotonically onto it from above (beta >= 1e6)",
        abs(off_t0 - off_t0_dw) < 1e-7
        and bool(np.all(np.diff(off_big) < 0))
        and bool(np.all(off_big > off_t0))
        and (quick or abs(off[-1] - off_t0) < 2e-5),
        f"offset_T=0 = {off_t0:.7f} (r chart), {off_t0_dw:.7f} (DW); top rung "
        f"{off[-1]:.7f} ({off[-1] - off_t0:+.1e})",
    )

    log("[5] the effective slope A_eff = d ln beta / d(1/nu) of (M), and its limit")
    a_over_b = abs(acoef / bcoef)

    def phi_ext(n):
        return float(phi(n)) if n > 0.02 else np.pi / 2 - np.arctan(n * a_over_b)

    def nu_of_lnbeta(lb):
        L = 0.25 * (lb - np.log(6)) + off[-1]
        return brentq(lambda n: n * L - chi(n) - phi_ext(n), 1e-3, 0.89)

    A_eff = {}
    for lb in (20, 25, 30, 50, 100, 200, 400, 800, 1600, 3200):
        n1, n2 = nu_of_lnbeta(lb), nu_of_lnbeta(lb + 1.0)
        A_eff[lb] = 1.0 / (1 / n2 - 1 / n1)
    log("      A_eff: " + ", ".join(f"ln beta={k}: {v:.3f}" for k, v in A_eff.items()))
    A_fit = float(np.load(paths.data("dk_throat.npz"))["fit_A"])
    check(
        "[5] A_eff at ln beta = 25 reproduces the paper sec. 4.6 fit A = 7.45",
        abs(A_eff[25] / A_fit - 1) < 0.02,
        f"{A_eff[25]:.3f} vs {A_fit:.3f}",
    )
    check(
        "[5] A_eff -> 4 pi",
        abs(A_eff[3200] / (4 * np.pi) - 1) < 0.01,
        f"{A_eff[3200]:.3f} vs {4 * np.pi:.3f}",
    )
    # nu^2 = (alpha_* - alpha)/(2 alpha_*) turns nu = A/ln(beta) into
    # T_c ~ exp(-(A/(2 sqrt 3))/sqrt(alpha_* - alpha)); the paper quotes both.
    log(
        f"      T_c exponent: asymptotic A/(2 sqrt 3) = 4 pi/(2 sqrt 3) = "
        f"{2 * np.pi / np.sqrt(3):.2f}; on the ladder read the exponent from the "
        f"direct fit in alpha (dk_throat.py [T3]: 2.25), not from A_fit/(2 sqrt 3) "
        f"= {A_fit / (2 * np.sqrt(3)):.2f}, which uses nu^2 ~ (alpha_*-alpha)/(2 alpha_*)"
    )

    failed = [k for k, v in CHECKS.items() if not v]
    log(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    if failed:
        log("cache not written")
        sys.exit(1)
    if quick:
        log("--quick: partial ladder, cache not written")
        return
    np.savez(
        CACHE,
        nu_tab=nu_tab,
        phi_tab=phi_tab,
        chi_tab=np.array([chi(n) for n in nu_tab]),
        rho_star_t0=rho_star,
        log_ratio_b_over_a=ratio,
        beta=beta,
        L=Lst,
        nu_meas=nu_m,
        nu_pred=nu_p,
        eps_ladder=eps_l,
        nu_pred_h=nu_ph,
        A_eff_lnbeta=np.array(list(A_eff.keys())),
        A_eff=np.array(list(A_eff.values())),
        phi_tab_dw=np.where(win, np.interp(nu_tab, win_nu, phi_dw), np.nan),
        log_ratio_dw=ratio_dw,
        offset_ladder=np.where(np.isfinite(Lst), Lst - 0.25 * np.log(beta / 6), np.nan),
        offset_t0=off_t0,
        offset_t0_dw=off_t0_dw,
        readme="Matched asymptotics for nu(beta) near alpha_*: nu ln(rho_*/h) = chi(nu) + phi_*(nu). "
        "chi = exact BTZ horizon phase, phi_* = T=0 outer phase referenced at rho_* (e^2V = 2v). "
        "A_eff -> 4 pi; the ladder's 7.45 is the pre-asymptotic slope.  log_ratio_b_over_a = b/a of "
        "the nu = 0 outer solution rho w = a + b ln(rho/rho_*), referenced at rho_*.  "
        "Second construction: phi_tab_dw (on the window 0.25 <= nu <= 0.6, nan "
        "elsewhere) and log_ratio_dw on the domain-wall-gauge T = 0 brane; offset_t0, "
        "offset_t0_dw = lim ln(rho_*/h) - (1/4) ln(beta/6) from the T = 0 brane alone; "
        "offset_ladder the same per rung.  nu_pred is the throat exponent from the "
        "matching condition, nu_pred_h = sqrt((1 + nu_pred^2) sqrt(1 - eps/6) - 1) the "
        "same in the frame of nu_meas = sqrt(b_h - 1), with eps_ladder = 6 - beta "
        "e^{-4V} at the horizon of each rung.",
    )
    log(f"cache written: {CACHE}")


if __name__ == "__main__":
    main()
