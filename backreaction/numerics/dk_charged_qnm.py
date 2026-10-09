"""Quasinormal modes of the polarised lowest-Landau-level channel on the
magnetic brane: the instability is a single purely imaginary mode that leaves
omega = 0 exactly at the static onset (paper app. B.2).

`derivations/dk_charged_dynamics.py` proves that on a static magnetic brane
with no A_t every growing charged mode has omega = i Gamma, and that in the
n = 0 channel the number of growing modes equals #{k : B_k < B}, the number of
static eigenvalues below B.  This module checks that numerically on the brane
in the channel n = 0, k_z = 0, and computes the slowest decaying modes.  It
does not compute the n >= 1 or k_z != 0 channels: that those have no growing
mode is the theorem, not a numerical statement.

Equation ([4] of the derivation; ingoing Eddington-Finkelstein, w =
e^{-i omega r_*} phi, phi regular at the horizon), in sigma = 1/r with the
frame functions A = sigma^2 U, M = V + ln sigma, N = W + ln sigma of
`dk_background.py` and Ph = A e^N:

    sigma Ph phi'' + (sigma Ph' - Ph) phi' + B sigma e^{N - 2M} phi
        + i omega e^N [ 2 sigma phi' + (sigma N' - 1) phi ] = 0,

source-free at the boundary (phi(0) = 0), regular at the horizon sigma = 1.

Path through the onset.  The background is held fixed at each beta and B is
scanned through the static eigenvalue B_0(beta) of `bc_alpha.bc_spectral_grid`;
along this path the coupling alpha = beta / B^2 varies, and B = B_0 is the
onset at alpha_c = beta / B_0^2.  The slope dGamma/dB along it is the
fixed-beta slope.  At fixed coupling alpha (fixed T) the background moves with
B, beta = alpha B^2, and since Gamma vanishes on the onset curve the chain rule
gives

    dGamma/dB |_alpha = dGamma/dB |_beta * d ln alpha_c / d ln beta,

which [Q9] checks against a direct fixed-alpha scan.  In the probe limit
d ln alpha_c / d ln beta -> 1 and the two slopes coincide.

Methods:
  (1) Chebyshev collocation of the equation as a generalised eigenvalue
      problem, linear in omega, at N and N + 32.  Only the low-lying modes
      converge: the imaginary mode and the first one or two decaying modes
      agree between the two resolutions to 1e-8 or better, the next ones move
      by 1e-5 to 1e-2, and everything below Im omega ~ -6 moves by O(1).  The
      growing-mode counts [Q1]/[Q2] are therefore taken on the raw spectra,
      not on a filtered list; the decaying modes quoted are the converged ones.
  (2) shooting from the horizon in sigma with the source c_0(omega) read off by
      a fit near sigma = 0, and a complex secant root search seeded by (1); on
      the imaginary axis the equation is real and the root is bracketed.
  (3) the argument principle for c_0(omega) from (2) around a rectangle in the
      upper half plane: an independent count of the growing modes that does
      not use the collocation matrix or the theorem.

Checks (asserted; non-zero exit on failure):
  [Q1] raw spectra at N and N + 32: no eigenvalue with Im omega > 1e-8 below
       B_0 (B = 0.9, 0.98, 0.999 B_0);
  [Q2] raw spectra at N and N + 32: above B_0 the eigenvalues with
       Im omega > 1e-8 number exactly #{k : B_k < B} (1 at 1.001, 1.02,
       1.1 B_0; 2 at (B_1 + B_2)/2; 3 at 1.05 B_2), are purely imaginary
       (|Re omega| < 1e-8) and agree between the resolutions to 1e-9;
  [Q3] Gamma -> 0 linearly as B -> B_0 from above; at B = B_0 a mode sits at
       omega = 0;
  [Q4] the converged decaying modes have Im omega < 0 and come in pairs
       (-conj(omega), omega), the T o (y -> -y) symmetry of the equation;
  [Q5] shooting agrees with collocation for the unstable mode and the first
       decaying mode;
  [Q6] the fixed-beta onset slope dGamma/dB at B_0 equals
       J_2 / (e^W w_0^2)|_horizon, J_2 = int rho w_0^2 dr, from first-order
       perturbation theory about the static zero mode w_0 (the boundary terms
       of (P w')' vanish; only -i omega int (e^W w_0^2)' dr survives, a pure
       horizon term) -- a third, semi-analytic determination;
  [Q7] anchor: at beta -> 0 the slope is the probe growth-rate slope
       J_2 / w_0(u_H)^2 on AdS5-Schwarzschild, computed here in the u chart
       (`probe_slope`; the same ratio is the probe-limit TDGL relaxation
       coefficient);
  [Q8] argument principle: the winding number of c_0(omega) around
       |Re omega| <= R, 1e-5 <= Im omega <= R equals #{k : B_k < B} at the six
       fields of [Q1]/[Q2], for R = 1.2 G + 1 and 2 G + 1, where
       G = (B max U e^{-2V})^{1/2} bounds the growth rate of a real-omega^2
       mode; the largest phase step on the contour is below 0.5;
  [Q9] the fixed-alpha onset slope: d ln alpha_c / d ln beta by centred
       differences of B_0(beta), and the slope of a direct fixed-alpha scan
       (background re-solved at beta = alpha_c B^2) equal to the fixed-beta
       slope times d ln alpha_c / d ln beta.

Units: horizon at r = 1, T = 1/pi, so omega is in units of pi T and B in
units of u_H^{-2}.

Run:  uv run python -m backreaction.numerics.dk_charged_qnm   (~4 min)
"""

import sys
import time
import warnings

import numpy as np
from numpy.polynomial import chebyshev as Ch
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.optimize import brentq

from spectral_check import cheb
from backreaction.numerics import bc_alpha as bca
from backreaction.numerics import dk_background as dk
from backreaction.numerics import dk_throat as dkt

T0 = time.time()
CHECKS = {}
UP_TOL = 1e-8  # Im omega above this counts as growing


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# probe limit: the growth-rate slope J_2 / w_0(u_H)^2 in the u chart
# ---------------------------------------------------------------------------
def probe_slope(N):
    """dGamma/dB at B_c on AdS5-Schwarzschild (u_H = 1, T = 1/pi).

    Static zero mode of u f w'' + (u f' - f) w' + B u w = 0, f = 1 - u^4,
    source-free (w(0) = 0) and regular at u = 1, by Chebyshev collocation as
    a generalised eigenproblem in B.  The slope is J_2 / w_0(1)^2 with
    J_2 = int_0^1 w_0^2 du / u: the u-chart form of [Q6], where the
    Eddington-Finkelstein weight sqrt(P R) = 1/u equals 1 at the horizon."""
    D, xs = cheb(N)
    u = (1 - xs) / 2  # u[0] = 0 (boundary), u[N] = 1 (horizon)
    Du = -2 * D
    f, fp = 1 - u**4, -4 * u**3
    L = np.diag(u * f) @ Du @ Du + np.diag(u * fp - f) @ Du
    Mm = -np.diag(u)
    L[0, :] = 0.0
    L[0, 0] = 1.0
    Mm[0, :] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vals, vecs = eig(L, Mm)
    ok = np.isfinite(vals) & (np.abs(vals.imag) < 1e-8) & (vals.real > 0)
    j = np.flatnonzero(ok)[np.argmin(vals[ok].real)]
    Bc, w = vals[j].real, vecs[:, j].real
    integrand = np.zeros_like(u)
    integrand[1:] = w[1:] ** 2 / u[1:]
    c = Ch.chebint(Ch.chebfit(xs, integrand, N))
    J2 = 0.5 * (Ch.chebval(1.0, c) - Ch.chebval(-1.0, c))  # du = -dx/2
    return Bc, J2 / w[N] ** 2


# ---------------------------------------------------------------------------
# backgrounds along the ladder
# ---------------------------------------------------------------------------
def accepted_solve(beta, N, guess=None, tol=1e-10):
    """solve_colloc at N, retried at N + 48 and N + 96 if the residual does
    not reach max(tol, residual_floor(N)); exits non-zero if none does."""
    for Ntry in (N, N + 48, N + 96):
        g = guess(Ntry) if callable(guess) else guess
        bg = dk.solve_colloc(beta, N=Ntry, tol=tol, maxit=200, guess=g)
        if bg.accepted:
            return bg
        log(
            f"    beta = {beta:g}, N = {Ntry}: residual {bg.extra['newton_res']:.1e}"
            f" above the floor {bg.res_floor:.1e}"
        )
    log(f"background at beta = {beta:g} not accepted at any N")
    sys.exit(1)


def ladder_backgrounds(targets):
    """Newton-continue the brane up dk_throat.LADDER, keeping the target rungs."""
    out, bg = {}, None
    for beta, N in dkt.LADDER:
        if beta > max(targets):
            break
        prev = bg
        bg = accepted_solve(
            beta, N, guess=(lambda n, prev=prev: dkt._regrid(prev, n)) if prev else None
        )
        if beta in targets:
            out[beta] = bg
    return out


def neighbour(bg, beta):
    """The brane at a nearby beta, Newton-started from bg on the same grid."""
    return accepted_solve(beta, bg.N, guess=lambda n: dkt._regrid(bg, n))


class Frame:
    """Background functions of sigma from the Chebyshev series of (A, M, N)."""

    def __init__(self, bg):
        self.cA = bg.cheb_coeffs("A")
        self.cM = bg.cheb_coeffs("M")
        self.cN = bg.cheb_coeffs("N")
        self.dA = 2 * Ch.chebder(self.cA)
        self.dN = 2 * Ch.chebder(self.cN)

    def at(self, s):
        xx = 2 * np.asarray(s, float) - 1
        A, M, Nn = (
            Ch.chebval(xx, self.cA),
            Ch.chebval(xx, self.cM),
            Ch.chebval(xx, self.cN),
        )
        As, Ns = Ch.chebval(xx, self.dA), Ch.chebval(xx, self.dN)
        eN = np.exp(Nn)
        return dict(
            Ph=A * eN, Phs=(As + A * Ns) * eN, eN=eN, mass=np.exp(Nn - 2 * M), Ns=Ns
        )

    def growth_bound(self, B):
        """G = (B max_sigma U e^{-2V})^{1/2} = (B max A e^{-2M})^{1/2}."""
        xx = np.linspace(-1, 1, 4001)
        return float(
            np.sqrt(
                B
                * np.max(Ch.chebval(xx, self.cA) * np.exp(-2 * Ch.chebval(xx, self.cM)))
            )
        )


# ---------------------------------------------------------------------------
# METHOD 1: collocation, generalised eigenproblem L phi = omega (-i K) phi
# ---------------------------------------------------------------------------
def qnm_colloc(fr, B, N, cap=60.0):
    """All finite eigenvalues (with |omega| < cap unless cap is None)."""
    D, xs = cheb(N)
    s = (xs + 1) / 2
    Ds = 2 * D
    D2 = Ds @ Ds
    f = fr.at(s)
    L = (s * f["Ph"])[:, None] * D2 + (s * f["Phs"] - f["Ph"])[:, None] * Ds
    L += np.diag(B * s * f["mass"])
    K = f["eN"][:, None] * (2 * s[:, None] * Ds + np.diag(s * f["Ns"] - 1))
    ib = N  # sigma = 0
    L[ib, :] = 0.0
    L[ib, ib] = 1.0
    K[ib, :] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        w = eig(L, -1j * K, right=False)
    w = w[np.isfinite(w)]
    return w if cap is None else w[np.abs(w) < cap]


def matched(wa, wb, tol):
    """Modes of wa that reappear in wb to within tol (relative + absolute)."""
    return np.array([z for z in wa if np.min(np.abs(wb - z)) < tol * (1 + abs(z))])


def spectrum(fr, B, N, tol=1e-5):
    """Modes converged between N and N + 32 to tol, slowest first."""
    wa = qnm_colloc(fr, B, N)
    wb = qnm_colloc(fr, B, N + 32)
    good = matched(wb, wa, tol)
    return good[np.argsort(-good.imag)]


def raw_growing(fr, B, N):
    """Raw (unfiltered, uncapped) eigenvalues with Im omega > UP_TOL."""
    w = qnm_colloc(fr, B, N, cap=None)
    up = w[w.imag > UP_TOL]
    return up[np.argsort(-up.imag)]


# ---------------------------------------------------------------------------
# METHOD 2: shooting from the horizon in sigma (vectorised in omega)
# ---------------------------------------------------------------------------
SMIN, SFIT = 1e-4, 0.02


def horizon_series(fr, B, om, h=0.05, deg=14, K=40, delta=0.02):
    """phi and phi' at sigma = 1 - delta on the regular (ingoing) branch, for
    an array of omega.

    The decaying modes need this: the outgoing branch behaves as
    (1 - sigma)^{-Im omega / 2}, so any error eps in the initial data is
    amplified by delta^{Im omega / 2}.  The coefficients a, b, c of
    a phi'' + b phi' + c phi = 0 are fitted by polynomials in
    tau = (sigma - 1)/h on [1 - h, 1] (analytic there), the Frobenius
    recursion gives the regular series, and it is summed at delta.  The
    normalisation phi(1) = 1 has poles only at omega = -2i(k + 1), in the
    lower half plane."""
    M = 48
    sg = 1 - h * (1 - np.cos(np.pi * np.arange(M + 1) / M)) / 2
    tau = (sg - 1) / h
    g = fr.at(sg)
    io = 1j * np.atleast_1d(np.asarray(om, complex))[None, :]
    a = sg * g["Ph"]
    b0, b1 = sg * g["Phs"] - g["Ph"], 2 * sg * g["eN"]
    c0, c1 = B * sg * g["mass"], g["eN"] * (sg * g["Ns"] - 1)
    V = np.vander(tau, deg + 1, increasing=True)

    def fit(y):
        co, *_ = np.linalg.lstsq(V, y, rcond=None)
        return np.concatenate([co, np.zeros(K + 2 - len(co))])[:, None]

    al = fit(a)
    al[0] = 0.0  # A(1) = 0: the horizon
    be = h * (fit(b0) + io * fit(b1))
    ga = h * h * (fit(c0) + io * fit(c1))
    f = np.zeros((K + 2, io.shape[1]), dtype=complex)
    f[0] = 1.0
    for kk in range(K):
        rest = sum(
            al[j] * (kk - j + 2) * (kk - j + 1) * f[kk - j + 2]
            for j in range(2, kk + 2)
        )
        rest = rest + sum(
            be[j] * (kk - j + 1) * f[kk - j + 1] for j in range(1, kk + 1)
        )
        rest = rest + sum(ga[j] * f[kk - j] for j in range(0, kk + 1))
        f[kk + 1] = -rest / ((kk + 1) * (kk * al[1] + be[0]))
    t0 = -delta / h
    pw = (t0 ** np.arange(K + 2))[:, None]
    phi0 = np.sum(f * pw, axis=0)
    dphi0 = np.sum(np.arange(1, K + 2)[:, None] * f[1:] * pw[:-1], axis=0) / h
    return phi0, dphi0


def source(fr, B, om, delta=0.02):
    """Coefficient c0 of the non-normalisable branch phi -> c0 at sigma -> 0
    (scalar omega in, scalar out; array in, array out)."""
    oms = np.atleast_1d(np.asarray(om, complex))
    n = len(oms)
    io = 1j * oms

    def rhs(s, yv):
        g = fr.at(np.array([s]))
        Ph, Phs, eN, ma, Ns = (g[k][0] for k in ("Ph", "Phs", "eN", "mass", "Ns"))
        ph, dph = yv[:n], yv[n:]
        dd = -(
            (s * Phs - Ph) * dph
            + io * eN * (2 * s * dph + (s * Ns - 1) * ph)
            + B * s * ma * ph
        ) / (s * Ph)
        return np.concatenate([dph, dd])

    p0, d0 = horizon_series(fr, B, oms, delta=delta)
    sol = solve_ivp(
        rhs,
        [1.0 - delta, SMIN],
        np.concatenate([p0, d0]),
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
    )
    ss = np.geomspace(SMIN, SFIT, 60)
    ph = sol.sol(ss)[:n].T
    lg = np.log(ss)
    basis = np.stack(
        [np.ones_like(ss), ss, ss**2, ss**2 * lg, ss**3, ss**3 * lg, ss**4, ss**4 * lg],
        axis=1,
    )
    c, *_ = np.linalg.lstsq(basis.astype(complex), ph, rcond=None)
    return c[0] if np.ndim(om) else c[0, 0]


def root_complex(fr, B, z0, tol=1e-11, maxit=40):
    """Secant search for c0(omega) = 0 from z0."""
    z1 = z0 * (1 + 1e-4) + 1e-5j
    f0, f1 = source(fr, B, z0), source(fr, B, z1)
    for _ in range(maxit):
        z2 = z1 - f1 * (z1 - z0) / (f1 - f0)
        if abs(z2 - z1) < tol * (1 + abs(z2)):
            return z2
        z0, f0, z1, f1 = z1, f1, z2, source(fr, B, z2)
    return np.nan


def gamma_shoot(fr, B, g_lo, g_hi):
    """Unstable root omega = i Gamma: the equation is real on this axis."""
    return brentq(lambda g: source(fr, B, 1j * g).real, g_lo, g_hi, xtol=1e-13)


# ---------------------------------------------------------------------------
# METHOD 3: argument principle for c0(omega) in the upper half plane
# ---------------------------------------------------------------------------
def winding(fr, B, R, d=1e-5, n_edge=400, max_step=0.4):
    """Winding number of c0 around |Re omega| <= R, d <= Im omega <= R
    (counter-clockwise), with the contour bisected wherever the phase step
    exceeds max_step.  Returns (winding, largest phase step, points)."""
    corners = [complex(-R, d), complex(R, d), complex(R, R), complex(-R, R)]
    z = np.concatenate(
        [
            a + (b - a) * np.linspace(0, 1, n_edge, endpoint=False)
            for a, b in zip(corners, corners[1:] + corners[:1], strict=True)
        ]
        + [np.array([corners[0]])]
    )
    f = source(fr, B, z)
    for _ in range(8):
        bad = np.flatnonzero(np.abs(np.angle(f[1:] / f[:-1])) > max_step)
        if len(bad) == 0:
            break
        zm = 0.5 * (z[bad] + z[bad + 1])
        z = np.insert(z, bad + 1, zm)
        f = np.insert(f, bad + 1, source(fr, B, zm))
    dphi = np.angle(f[1:] / f[:-1])
    return float(np.sum(dphi) / (2 * np.pi)), float(np.max(np.abs(dphi))), len(z)


# ---------------------------------------------------------------------------
def main():
    targets = (1.0, 45.0, 1e4)
    PROBE_BETA = 1e-6

    # [Q7] reference: the probe slope in the u chart, at two resolutions
    (_, ps1), (bc2, ps2) = probe_slope(48), probe_slope(64)
    PROBE_SLOPE = ps2
    log(
        f"probe limit (u chart): B_c = {bc2:.10f}, J_2/w_0(u_H)^2 = {ps2:.12f}"
        f"  (N = 48 vs 64: {abs(ps1 - ps2):.1e})"
    )
    check(
        "[Q7] probe slope J_2/w_0(u_H)^2 converged in N, B_c = 5.13126764",
        abs(ps1 - ps2) < 1e-11 and abs(bc2 - 5.13126764) < 1e-8,
        f"slope = {ps2:.12f}",
    )

    log(f"continuing the brane to beta = {targets} ...")
    bgs = ladder_backgrounds(set(targets))
    bgs[PROBE_BETA] = accepted_solve(PROBE_BETA, 64, tol=1e-12)
    rows, slopes, apcount = [], [], []
    worst_re = 0.0  # largest |Re omega| of a growing mode, any beta, field, N
    for beta in (PROBE_BETA,) + targets:
        bg = bgs[beta]
        fr = Frame(bg)
        Bk = [float(v) for v in bca.bc_spectral_grid(bg, n_keep=4)[0]]
        B0, B1, B2 = Bk[:3]
        N = bg.N + 16
        log(
            f"beta = {beta:g}: N_bg = {bg.N}, T = {bg.temperature():.10f}, "
            f"B_0 = {B0:.8f} (alpha_c = {beta / B0**2:.5f}), B_1 = {B1:.6f}, "
            f"B_2 = {B2:.6f}"
        )
        fields = [(f"{c} B_0", c * B0) for c in (0.9, 0.98, 0.999, 1.001, 1.02, 1.1)]
        fields += [("(B_1+B_2)/2", 0.5 * (B1 + B2)), ("1.05 B_2", 1.05 * B2)]
        # ---- [Q1]/[Q2]: growing modes counted on the raw spectra --------------
        for lab, B in fields:
            expect = sum(1 for b in Bk if b < B)
            ua, ub = raw_growing(fr, B, N), raw_growing(fr, B, N + 32)
            ok = len(ua) == expect and len(ub) == expect
            if ok and expect:
                worst_re = max(
                    worst_re,
                    float(np.max(np.abs(ua.real))),
                    float(np.max(np.abs(ub.real))),
                )
                ok &= bool(np.all(np.abs(ua.real) < UP_TOL))
                ok &= bool(np.all(np.abs(ub.real) < UP_TOL))
                ok &= bool(np.max(np.abs(ua - ub) / (1 + np.abs(ub))) < 1e-9)
            tag = "[Q1]" if expect == 0 else "[Q2]"
            check(
                f"{tag} beta = {beta:g}, B = {lab}: raw spectra at N = {N}, {N + 32}"
                f" have exactly #{{B_k < B}} = {expect} modes with Im omega > 0,"
                f" all imaginary",
                ok,
                "Gamma = " + ", ".join(f"{z.imag:.9f}" for z in ub) if len(ub) else "",
            )
        # ---- [Q8] argument principle ------------------------------------------
        for lab, B in [fields[i] for i in (0, 2, 3, 5, 6, 7)]:
            expect = sum(1 for b in Bk if b < B)
            G = fr.growth_bound(B)
            res = [winding(fr, B, c * G + 1) for c in (1.2, 2.0)]
            ok = all(abs(wn - expect) < 1e-2 and st < 0.5 for wn, st, _ in res)
            check(
                f"[Q8] beta = {beta:g}, B = {lab}: winding of c_0 in the upper half "
                f"plane = #{{B_k < B}} = {expect}",
                ok,
                "windings "
                + ", ".join(f"{round(wn, 4) + 0.0:+.4f}" for wn, _, _ in res)
                + f"; max phase step {max(st for _, st, _ in res):.2f}",
            )
            apcount.append(
                (beta, lab, expect, [round(wn, 4) + 0.0 for wn, _, _ in res])
            )
        if beta == PROBE_BETA:
            continue
        # ---- converged spectra: decaying modes, shooting ----------------------
        gam = {}
        for fac in (0.9, 0.98, 1.0, 1.02, 1.1):
            B = B0 * fac
            w = spectrum(fr, B, N)
            up = w[w.imag > UP_TOL]
            down = w[w.imag < -UP_TOL]
            log(
                f"  B/B_0 = {fac:5.2f} (alpha = {beta / B**2:.5f}), "
                f"{len(w)} modes converged to 1e-5: "
                + "  ".join(f"{z.real:+.6f}{z.imag:+.6f}i" for z in w[:5])
            )
            if fac > 1 and len(up) == 1:
                g = up[0].imag
                gs = gamma_shoot(fr, B, 0.5 * g, 1.5 * g)
                gam[fac] = g
                check(
                    f"[Q5] beta = {beta:g}, B = {fac} B_0: shooting Gamma = collocation",
                    abs(gs - g) < 1e-6 * (1 + g),
                    f"{gs:.10f} vs {g:.10f}",
                )
            if fac == 1.0:
                wr = qnm_colloc(fr, B, N, cap=None)
                check(
                    f"[Q3] beta = {beta:g}: at B = B_0 a mode sits at omega = 0",
                    np.min(np.abs(wr)) < 1e-7,
                    f"min |omega| = {np.min(np.abs(wr)):.1e}",
                )
            pair_ok = all(
                np.min(np.abs(down + np.conj(z))) < 1e-6 * (1 + abs(z)) for z in down
            )
            check(
                f"[Q4] beta = {beta:g}, B = {fac} B_0: converged decaying modes in "
                f"pairs (-conj w, w)",
                pair_ok and len(down) >= 1,
            )
            if fac == 1.1:
                z = down[np.argmax(down.imag)]
                if z.real < 0:
                    z = -np.conj(z)
                if abs(z.real) < 1e-8:  # overdamped: real equation on the axis
                    g = z.imag
                    zs = 1j * gamma_shoot(fr, B, 1.02 * g, 0.98 * g)
                else:
                    zs = root_complex(fr, B, z)
                check(
                    f"[Q5] beta = {beta:g}: first decaying mode, shooting = collocation",
                    abs(zs - z) < 1e-6 * (1 + abs(z)),
                    f"{zs:.8f} vs {z:.8f}",
                )
                rows.append((beta, B0, z))
        if 1.02 in gam and 1.1 in gam:
            ratio = gam[1.02] / gam[1.1]
            check(
                f"[Q3] beta = {beta:g}: Gamma vanishes linearly at B_0",
                0.15 < ratio < 0.25,
                f"Gamma(1.02)/Gamma(1.1) = {ratio:.4f}",
            )
        else:
            check(f"[Q3] beta = {beta:g}: Gamma vanishes linearly at B_0", False)
        # ---- [Q6] fixed-beta onset slope vs first-order perturbation theory -----
        _, w0 = bca.bc_spectral_grid(bg, n_keep=1)
        wn, _ = bca.normalise_mode(bg, w0)
        pred = 1.0 / (np.exp(bg.Nn[0]) * wn[0] ** 2)
        sl = [spectrum(fr, B0 * (1 + e_), N)[0].imag / (e_ * B0) for e_ in (2e-3, 1e-3)]
        meas = 2 * sl[1] - sl[0]  # Richardson in the step
        check(
            f"[Q6] beta = {beta:g}: dGamma/dB|_beta at B_0 = J_2 / (e^W w_0^2)|_horizon",
            abs(meas / pred - 1) < 1e-5,
            f"{meas:.8f} vs {pred:.8f}",
        )
        # ---- [Q9] the fixed-alpha slope -----------------------------------------
        dl = []
        for hh in (2e-3, 1e-3):
            Bp = float(bca.bc_spectral_grid(neighbour(bg, beta * np.exp(hh)), 1)[0][0])
            Bm = float(bca.bc_spectral_grid(neighbour(bg, beta * np.exp(-hh)), 1)[0][0])
            dl.append(1 - (np.log(Bp) - np.log(Bm)) / hh)
        dla = (4 * dl[1] - dl[0]) / 3
        sa = []
        for e_ in (2e-3, 1e-3):
            fre = Frame(neighbour(bg, beta * (1 + e_) ** 2))
            sa.append(spectrum(fre, B0 * (1 + e_), N)[0].imag / (e_ * B0))
        meas_a = 2 * sa[1] - sa[0]
        check(
            f"[Q9] beta = {beta:g}: fixed-alpha slope = fixed-beta slope x "
            f"d ln alpha_c/d ln beta",
            abs(meas_a / (meas * dla) - 1) < 1e-5 and abs(dl[0] - dl[1]) < 1e-6,
            f"{meas_a:.8f} vs {meas:.8f} x {dla:.6f} = {meas * dla:.8f}",
        )
        slopes.append((beta, B0, meas, pred, dla, meas_a))

    # [Q7] the brane at beta = 1e-6 against the u-chart probe slope
    bg = bgs[PROBE_BETA]
    fr = Frame(bg)
    B0 = float(bca.bc_spectral_grid(bg, n_keep=1)[0][0])
    sl = [spectrum(fr, B0 * (1 + e_), 80)[0].imag / (e_ * B0) for e_ in (2e-3, 1e-3)]
    meas = 2 * sl[1] - sl[0]
    check(
        f"[Q7] beta = 1e-6: brane onset slope = probe slope {PROBE_SLOPE:.8f}",
        abs(meas / PROBE_SLOPE - 1) < 1e-5 and abs(B0 / 5.13126764 - 1) < 1e-6,
        f"B_0 = {B0:.8f}, slope = {meas:.8f}",
    )
    # paper app. B.2: "purely imaginary to 2e-11" (measured 1.2e-11)
    check(
        "[Q2] every growing mode, both resolutions, all beta: |Re omega| < 2e-11",
        0.0 < worst_re < 2e-11,
        f"worst |Re omega| = {worst_re:.2e}",
    )

    log("")
    log("first decaying mode at B = 1.1 B_0 (units pi T):")
    for beta, B0, z in rows:
        log(
            f"  beta = {beta:8g}  B_0 = {B0:12.6f}  omega = {z.real:+.6f} {z.imag:+.6f} i"
        )
    log("onset slope dGamma/dB (units pi T per u_H^-2):")
    log(
        "  fixed beta: measured vs J_2/(e^W w_0^2)_h; d ln alpha_c/d ln beta; fixed alpha"
    )
    for beta, B0, meas, pred, dla, meas_a in slopes:
        log(
            f"  beta = {beta:8g}  B_0 = {B0:12.6f}  {meas:.8f}  {pred:.8f}"
            f"  {dla:.6f}  {meas_a:.8f}"
        )
    log("growing-mode count by the argument principle (two contours):")
    for beta, lab, expect, wns in apcount:
        log(
            f"  beta = {beta:8g}  B = {lab:12s}  #{{B_k < B}} = {expect}  winding {wns}"
        )
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        sys.exit(1)
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
