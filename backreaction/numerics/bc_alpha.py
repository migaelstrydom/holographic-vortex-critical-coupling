"""Self-consistent critical field B_c(alpha) of the holographic vortex lattice
with the gravitational backreaction included to all orders in beta.

The normal phase at physical field B is the exact D'Hoker-Kraus magnetic brane
with beta = alpha B^2 (dk_background.py, fixed T = 1/pi).  The onset of the
Nielsen-Olesen instability is where the polarised SU(2) W zero mode
(systems/dk_zeromode.py)

    (p w')' + q w = 0,   p = U e^{W},  q = B e^{W - 2V},   rho = e^{W - 2V},
    horizon:  w'(r_p) = -(B e^{-2V(r_p)}/U'(r_p)) w(r_p),
    boundary: source c0 (r^0) -> 0 (source-free / normalisable)

first admits a source-free solution ON THAT SAME background.  Since the field B
enters both the eigenproblem and (through beta) the background it lives on, the
critical field is the self-consistent root

    B_c(alpha):   c0(B; background[beta = alpha B^2]) = 0.

At alpha = 0 the background is fixed AdS5-Schwarzschild for every B, so this
reduces to the probe eigenvalue B_c(0) = 5.13126764.

Two independent methods for the source / eigenvalue:
  (1) shoot the SL ODE from the horizon (in ln(r - r_p)), and read the source
      off exactly far out, c0 = w + (r/2) w' at r = 1e4 (`source_coefficient`;
      the residual falls as r_max^-3: 4e-9, 4e-12, 2e-13 relative in B_c at
      r_max = 1e3, 1e4, 1e5 on the beta = 45 brane);
  (2) Chebyshev collocation of the SL as a generalised eigenvalue problem
      A w = B M w (the B in the horizon Robin BC goes into M).

Anchors:
  * B_c(0) = 5.13126764 (both methods);
  * O(alpha) slope  dB_c/B_c = +0.20970443 alpha B_c^2, the first-order
    perturbative shift computed independently by numerics/bc_shift.py,
    i.e. dB_c/dalpha|_0 = +28.332285.

Run:  uv run python -m backreaction.numerics.bc_alpha
"""

import os
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig
from scipy.optimize import brentq

from spectral_check import cheb
from backreaction import paths
from backreaction.systems import dk_zeromode as zm
from backreaction.numerics import dk_background as dk

PI = np.pi
CHECKS = {}


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


BC0_TARGET = 5.131267637682236  # probe value to 30 digits: numerics/probe_bc_precise.py
SLOPE_TARGET = (
    0.20970443  # dB_c/B_c per alpha B_c^2  (perturbative, numerics/bc_shift.py)
)


# ---------------------------------------------------------------------------
# background provider: physical field B, coupling alpha -> Background(beta)
# ---------------------------------------------------------------------------
def background_for(B, alpha, N=64, method="colloc", guess=None):
    beta = alpha * B**2
    if method == "colloc":
        return dk.solve_colloc(
            beta,
            alpha=alpha if alpha > 0 else 2.0,
            N=N,
            tol=1e-12,
            maxit=40,
            guess=guess,
            require=True,
        )
    return dk.solve_shoot(beta, alpha=alpha if alpha > 0 else 2.0, N=max(N, 80))


# ---------------------------------------------------------------------------
# METHOD 1: SL shooting from the horizon; exact source extraction far out
# ---------------------------------------------------------------------------
def source_coefficient(r, w, wp):
    """c0 from (w, w') at large r, without a fit.

    Near the boundary w = c0 [1 + (B/2) r^-2 ln r + ...] + c2 r^-2 [1 + O(1/r)],
    and in the fixed-T frame (U = r^2 + 2 a r + ...) the O(1/r) corrections
    are genuine: r^-3 and r^-3 ln r terms.  The combination w + (r/2) w' kills
    the c2 r^-2 solution exactly, leaves c0 times (1 + B/(4 r^2) + ...), which
    cannot move the ROOT c0(B) = 0, and leaves an additive O(c2 / r^3).  At the
    r_max used below that is far below round-off, so no fit columns (and no
    fit bias) are involved.  A fit that omits the r^-3 terms would be biased
    at O(r_max^-3)."""
    return w + 0.5 * r * wp


def sl_c0_shoot(B, bg, rmax=1e4, rhor=1.0, d=1e-8):
    """Source coefficient c0(B) of the zero mode shot from the horizon.

    Integrated in s = ln(r - r_p), so the step tracks the geometry from the
    horizon through any throat to r_max; with y = dw/ds = (r - r_p) w' the
    equation p w'' + p1 w' + q w = 0 is y_s = y - x (p1/p) y - x^2 (q/p) w,
    x = r - r_p.  The source is read off by `source_coefficient`."""
    keys = ("U", "Up", "V", "Vp", "W", "Wp")
    f = bg.fields_r(np.array([rhor + d]))
    slope = zm.horizon_slope(rhor + d, *[f[k][0] for k in keys], B)

    def rhs(s, Y):
        w, y = Y
        x = np.exp(s)
        r = rhor + x
        ff = bg.fields_r(np.array([r]))
        a = [ff[k][0] for k in keys]
        p, p1, q = zm.sl_p(r, *a, B), zm.sl_p1(r, *a, B), zm.sl_q(r, *a, B)
        return [y, y - x * p1 * y / p - x * x * q * w / p]

    sol = solve_ivp(
        rhs,
        [np.log(d), np.log(rmax - rhor)],
        [1.0, d * slope],
        rtol=1e-12,
        atol=1e-14,
        method="DOP853",
    )
    if not sol.success:
        return np.nan
    w, y = sol.y[:, -1]
    return source_coefficient(rmax, w, y / (rmax - rhor))


# ---------------------------------------------------------------------------
# METHOD 2: SL Chebyshev-collocation generalised eigenvalue problem
# ---------------------------------------------------------------------------
def sl_eig_colloc(bg, N=80, nret=1):
    """Smallest real positive SL eigenvalue B on the fixed background bg.
    A w = B M w with the B-dependent horizon Robin BC folded into M."""
    D, xs = cheb(N)
    sig = (xs + 1) / 2.0
    Ds = 2.0 * D
    D2s = Ds @ Ds
    ih, ib = 0, N  # sigma=1 horizon, sigma=0 boundary
    # evaluate the SL coefficients per node (scalar zm.* calls; never at
    # sigma=0 where r=inf).  p,p1,rho are B-independent.
    nodes = list(range(0, N))  # interior + horizon (skip ib)
    fs = bg.fields_sigma(sig[nodes])
    A = np.zeros((N + 1, N + 1))
    M = np.zeros((N + 1, N + 1))
    for idx, k in enumerate(nodes):
        r_ = 1.0 / sig[k]
        U, Up = fs["U"][idx], fs["Up"][idx]
        V, Vp = fs["V"][idx], fs["Vp"][idx]
        W, Wp = fs["W"][idx], fs["Wp"][idx]
        if k == ih:  # horizon sigma=1 Robin BC
            # w_s(1) = B (e^{-2V(1)}/U'(1)) w(1)
            A[ih, :] = Ds[ih, :]
            M[ih, ih] = np.exp(-2 * V) / Up
            continue
        p = zm.sl_p(r_, U, Up, V, Vp, W, Wp, 1.0)
        p1 = zm.sl_p1(r_, U, Up, V, Vp, W, Wp, 1.0)
        rho = zm.sl_rho(r_, U, Up, V, Vp, W, Wp, 1.0)
        A[k, :] = -(
            p * sig[k] ** 4 * D2s[k, :]
            + (2 * p * sig[k] ** 3 - p1 * sig[k] ** 2) * Ds[k, :]
        )
        M[k, k] = rho
    # boundary sigma=0: Dirichlet w = 0 (source-free)
    A[ib, ib] = 1.0
    vals, vecs = eig(A, M)
    vals = vals[np.isfinite(vals)]
    real = vals[np.abs(vals.imag) < 1e-7 * (1 + np.abs(vals.real))].real
    pos = np.sort(real[real > 1e-6])
    return pos[:nret] if nret > 1 else (pos[0] if len(pos) else np.nan)


# ---------------------------------------------------------------------------
# self-consistent B_c(alpha)
# ---------------------------------------------------------------------------
def bc_shoot(alpha, N=64, blo=3.0, bhi=8.0):
    """Method 1: root of H(B) = c0(B; background[alpha B^2])."""

    def H(B):
        bg = background_for(B, alpha, N=N, method="colloc")
        return sl_c0_shoot(B, bg)

    # widen bracket if needed
    flo, fhi = H(blo), H(bhi)
    it = 0
    while flo * fhi > 0 and it < 8:
        bhi += 2.0
        fhi = H(bhi)
        it += 1
    return brentq(H, blo, bhi, xtol=1e-10, rtol=1e-12)


def bc_colloc(alpha, N=64, Nsl=80, B0=5.13):
    """Method 2: fixed point B = smallest SL eigenvalue on background[alpha B^2]."""

    def D_(B):
        bg = background_for(B, alpha, N=N, method="colloc")
        return sl_eig_colloc(bg, N=Nsl) - B

    blo, bhi = 3.0, 8.0
    flo, fhi = D_(blo), D_(bhi)
    it = 0
    while flo * fhi > 0 and it < 8:
        bhi += 2.0
        fhi = D_(bhi)
        it += 1
    return brentq(D_, blo, bhi, xtol=1e-10, rtol=1e-12)


# ---------------------------------------------------------------------------
# grid-aligned spectral eigenvalue + eigenvector (for the saved zero mode)
# ---------------------------------------------------------------------------
def bc_spectral_grid(bg, n_keep=4):
    """Lowest SL eigenvalue(s) and ground-state eigenvector of the operator
    on background bg, on bg's OWN sigma-grid (so w0 aligns with the saved A,M,N).

    Cleared form (x sigma^3, Phat = A e^{N} vanishes at the horizon):
        -Phat sigma w'' - (Phat' sigma - Phat) w' = B (sigma e^{N-2M}) w ,
    boundary row = Dirichlet w(0)=0 (source-free); horizon row = the degenerate
    (Phat(1)=0) equation (regularity)."""
    sig, D = bg.sig, bg.D
    D2 = D @ D
    A, M, Nn = bg.A, bg.M, bg.Nn
    Phat = A * np.exp(Nn)
    Phat_s = D @ Phat
    mass = sig * np.exp(Nn - 2 * M)
    L = -(Phat * sig)[:, None] * D2 - (Phat_s * sig - Phat)[:, None] * D
    Mm = np.diag(mass)
    ib = bg.N
    L[ib, :] = 0.0
    L[ib, ib] = 1.0
    Mm[ib, :] = 0.0
    vals, vecs = eig(L, Mm)
    vals = vals.real
    pos = np.where(np.isfinite(vals) & (vals > 1e-6) & (vals < 1e6))[0]
    order = pos[np.argsort(vals[pos])]
    v0 = vecs[:, order[0]].real
    if v0[0] < 0:  # w0 > 0 at the horizon
        v0 = -v0
    return vals[order][:n_keep], v0


def _cc_weights_01(N):
    """Clenshaw-Curtis weights for int_0^1 on sigma = (cos(pi j/N)+1)/2."""
    th = PI * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    for j in range(N + 1):
        s = sum(
            (1.0 if 2 * k == N else 2.0) / (4 * k * k - 1) * np.cos(2 * k * th[j])
            for k in range(1, N // 2 + 1)
        )
        w[j] = 1.0 - s
    w *= 2.0 / N
    w[0] /= 2.0
    w[-1] /= 2.0
    return w * 0.5


def normalise_mode(bg, w_nodal):
    """Scale w so J = int e^{W-2V} w^2 dr = 1 (brane measure).  dr=-dsigma/sigma^2,
    e^{W-2V}=sigma e^{N-2M} => integrand = e^{N-2M} w^2 / sigma."""
    ccw = _cc_weights_01(bg.N)
    integ = np.zeros_like(bg.sig)
    m = bg.sig > 1e-12
    integ[m] = np.exp(bg.Nn - 2 * bg.M)[m] * w_nodal[m] ** 2 / bg.sig[m]
    J = float(np.sum(ccw * integ))
    return w_nodal / np.sqrt(J), J


# ---------------------------------------------------------------------------
# full-range beta-sweep (B_eig depends on beta ONLY -> no root-find needed)
# ---------------------------------------------------------------------------
def beta_sweep(betas, N=64, cross_every=5):
    """Self-consistent curve by sweeping beta: B_c=B_eig(beta) (metric-only
    eigenvalue), alpha=beta/B_c^2, beta_c=alpha B_c^2=beta.  Returns arrays and
    per-beta normalised zero modes + background nodal (A,M,N) for rebuild."""
    alpha, Bc, w0s, As, Ms, Ns, twomethod = [], [], [], [], [], [], []
    for i, b in enumerate(betas):
        bg = dk.solve_colloc(b, N=N, tol=1e-12, maxit=60, require=True)
        tower, v0 = bc_spectral_grid(bg)
        Bc_b = tower[0]
        w0n, J = normalise_mode(bg, v0)
        assert abs(J) > 0 and np.isfinite(Bc_b)
        a = 0.0 if b == 0.0 else b / Bc_b**2
        alpha.append(a)
        Bc.append(Bc_b)
        w0s.append(w0n)
        As.append(bg.A)
        Ms.append(bg.M)
        Ns.append(bg.Nn)
        tag = ""
        if i % cross_every == 0:  # shooting cross-check
            Bsh = bc_shoot_bg(bg, Bc_b)
            twomethod.append((b, Bc_b, Bsh))
            tag = f"  [shoot {Bsh:.7f}, d={abs(Bsh - Bc_b):.1e}]"
        print(f"  beta={b:6.2f}  alpha={a:.6f}  B_c={Bc_b:.8f}  J->{J:.2e}{tag}")
    return (
        np.array(alpha),
        np.array(Bc),
        np.array(betas, float),
        np.array(w0s),
        np.array(As),
        np.array(Ms),
        np.array(Ns),
        twomethod,
        N,
    )


def bc_shoot_bg(bg, B_guess):
    """Shooting eigenvalue on a FIXED background (root of c0(B)); cross-check."""
    lo, hi = B_guess * 0.94, B_guess * 1.06
    flo, fhi = sl_c0_shoot(lo, bg), sl_c0_shoot(hi, bg)
    it = 0
    while (not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0) and it < 30:
        lo *= 0.97
        hi *= 1.03
        flo, fhi = sl_c0_shoot(lo, bg), sl_c0_shoot(hi, bg)
        it += 1
    return brentq(lambda B: sl_c0_shoot(B, bg), lo, hi, xtol=1e-9, rtol=1e-12)


def save_curve(alpha, Bc, beta, w0s, As, Ms, Ns, N, path=None, **extra):
    path = paths.data("bc_alpha.npz") if path is None else path
    D, xs = cheb(N)
    sig = (xs + 1) / 2.0
    np.savez(
        path,
        alpha=alpha,
        B_c=Bc,
        beta=beta,
        beta_c=beta,
        N=N,
        sigma=sig,
        w0=w0s,
        A=As,
        M=Ms,
        Nn=Ns,
        B_c0=BC0_TARGET,
        **extra,
        readme=(
            "Self-consistent B_c(alpha) on the DK brane, T=1/pi. "
            "Parametrised by beta: alpha=beta/B_c^2, beta_c=alpha B_c^2=beta. "
            "Per-beta zero mode w0 on Chebyshev-Lobatto sigma=(cos(pi j/N)+1)/2 "
            "in [0,1] (sigma=1 horizon, sigma=0 boundary), NORMALISED so "
            "J=int e^{W-2V} w0^2 dr=1 (brane measure, dr=-dsigma/sigma^2). "
            "Background reduced fields A=sigma^2 U, M=V+ln sigma, N=W+ln sigma "
            "(nodal): U=A/sigma^2, V=M-ln sigma, W=N-ln sigma. Physical field "
            "B=B_c. Rebuild any background via dk_background.solve_colloc(beta, N)."
        ),
    )
    print(f"  saved {path}")


def tangent_ten_percent(alpha, Bc):
    """alpha at which the alpha = 0 tangent of B_c is 10% off, |B_c - tangent|
    = 0.1 B_c, by a cubic spline through the sweep (paper sec. 3.3: 0.0787;
    the first sweep point past it is 0.0839).  None if never within the sweep."""
    from scipy.interpolate import CubicSpline  # noqa: PLC0415

    o = np.argsort(alpha)
    a, B = np.asarray(alpha)[o], np.asarray(Bc)[o]
    lin = BC0_TARGET + SLOPE_TARGET * BC0_TARGET**3 * a
    dev = np.abs((B - lin) / B) - 0.10
    i10 = np.where(dev > 0)[0]
    if not len(i10) or i10[0] == 0:
        return None
    sp_ = CubicSpline(a, B)
    return brentq(
        lambda x: (
            abs(1 - (BC0_TARGET + SLOPE_TARGET * BC0_TARGET**3 * x) / sp_(x)) - 0.10
        ),
        a[i10[0] - 1],
        a[i10[0]],
    )


def make_figure(alpha, Bc, beta, path=None):
    path = paths.figure("bc_alpha.png") if path is None else path
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    lin = BC0_TARGET + SLOPE_TARGET * BC0_TARGET**3 * alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = (Bc - lin) / Bc
    a10 = tangent_ten_percent(alpha, Bc)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # left: B_c(alpha) vs its alpha = 0 tangent, with the 10% deviation marker (paper sec. 3.3)
    ax[0].plot(
        alpha, Bc, "o-", ms=3, lw=1.3, label=r"$B_c(\alpha)$ (all orders in $\beta$)"
    )
    aline = np.linspace(0, alpha.max(), 100)
    ax[0].plot(
        aline,
        BC0_TARGET + SLOPE_TARGET * BC0_TARGET**3 * aline,
        "--",
        lw=1,
        color="crimson",
        label=r"tangent $+28.33\,\alpha$ (perturbative slope)",
    )
    if a10 is not None:
        ax[0].axvline(
            a10,
            color="gray",
            ls=":",
            lw=1,
            label=rf"10% dev. at $\alpha\!\approx\!{a10:.3f}$",
        )
    ax[0].set_xlabel(r"$\alpha$")
    ax[0].set_ylabel(r"$B_c\,u_H^2$")
    ax[0].set_title("Self-consistent critical field")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    # deviation inset
    axin = ax[0].inset_axes([0.14, 0.55, 0.4, 0.4])
    axin.plot(alpha, dev * 100, "-", lw=1)
    axin.axhline(-10, color="crimson", ls=":", lw=0.8)
    axin.set_title(r"$(B_c-B_c^{\rm lin})/B_c$ [%]", fontsize=7)
    axin.tick_params(labelsize=6)
    # right: beta_c(alpha) = alpha B_c^2 (the self-consistent throat depth)
    ax[1].plot(alpha, beta, "o-", ms=3, lw=1.3, color="teal")
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_ylabel(r"$\beta_c=\alpha B_c^2$")
    ax[1].set_title("Backreaction strength at criticality")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# the degenerate-equation spectral solve (beta-parametrised root-find) and the
# three-solver cross-check
# ---------------------------------------------------------------------------
def degenerate_bc_of_alpha(alpha, N=64):
    """B_c(alpha) as the root of B - B_eig(alpha B^2), with B_eig from the
    degenerate-equation spectral solver on the background's own grid (a
    discretisation distinct from the Robin-BC GEP and from the shooting)."""
    if alpha == 0.0:
        return bc_spectral_grid(dk.solve_colloc(0.0, N=N, tol=1e-13, require=True))[0][
            0
        ]

    def g(B):
        tower, _ = bc_spectral_grid(
            dk.solve_colloc(alpha * B**2, N=N, tol=1e-12, maxit=60, require=True)
        )
        return tower[0] - B

    return brentq(g, 3.0, 20.0, xtol=1e-9, rtol=1e-12)


def independent_crosscheck(alphas):
    """Compare three onset solvers at the SAME alpha (no interpolation): the
    horizon shooting with exact source extraction and the Robin-BC collocation
    GEP (both per-alpha root-finds in B), and the degenerate-equation spectral
    GEP on the background's own grid (beta-parametrised root-find).

    What is shared: all three solve the zero mode on the SAME collocation
    background (dk_background.solve_colloc); the brane itself is checked
    against shooting separately (dk_background).  The two GEPs also share the
    Sturm-Liouville coefficients and differ only in discretisation and in how
    the horizon condition is imposed, so they agree to ~1e-9; the shooting is
    the independent method for the eigenvalue."""
    print("\n== cross-check: 3 onset solvers at the same alpha ==")
    print(
        f"  {'alpha':>8} {'shoot':>14} {'Robin GEP':>14} "
        f"{'degenerate GEP':>15} {'max|diff|':>11}"
    )
    rows = []
    for a in alphas:
        bsh = bc_shoot(a)  # shooting + root-find
        bcc = bc_colloc(a)  # Robin-BC GEP + root-find
        bdg = degenerate_bc_of_alpha(a)  # degenerate-eq spectral + root-find
        d = max(abs(bsh - bcc), abs(bsh - bdg), abs(bcc - bdg))
        rows.append((a, bsh, bcc, bdg, d))
        print(f"  {a:8.6f} {bsh:14.10f} {bcc:14.10f} {bdg:15.10f} {d:11.1e}")
    return rows


def n_convergence(betas=(10.0, 45.0), Ns=(48, 64, 96)):
    """Combined background + eigenvalue N-refinement at large beta (spectral)."""
    print("\n== N-convergence (spectral, background+eigenvalue) at large beta ==")
    out = {}
    for b in betas:
        vals = []
        for N in Ns:
            t, _ = bc_spectral_grid(
                dk.solve_colloc(b, N=N, tol=1e-12, maxit=60, require=True)
            )
            vals.append(t[0])
        out[b] = list(zip(Ns, vals, strict=True))
        s = "  ".join(f"N={N}:{v:.8f}" for N, v in zip(Ns, vals, strict=True))
        print(
            f"  beta={b:5.1f}: {s}   |N{Ns[-2]}-N{Ns[-1]}|={abs(vals[-2] - vals[-1]):.1e}"
        )
    return out


# ---------------------------------------------------------------------------
# report / anchors
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("#" * 74)
    print("# Self-consistent critical field B_c(alpha) on the DK background ")
    print("#" * 74)

    # --- anchor: B_c(0) both methods, N-convergence of method 2 ---
    print("\n== anchor B_c(0) = 5.13126764 ==")
    bg0 = dk.solve_colloc(0.0, N=64, tol=1e-13, require=True)
    c0_sh = brentq(lambda B: sl_c0_shoot(B, bg0), 4.5, 6.0, xtol=1e-12)
    print(
        f"  method 1 (shoot):   B_c(0) = {c0_sh:.13f}  err {abs(c0_sh - BC0_TARGET):.1e}"
    )
    # table 4, "collocation at beta = 0": 5e-12; worst of the five below is 4.5e-12 (N = 120)
    check("B_c(0) by shooting to 1.5e-11", abs(c0_sh - BC0_TARGET) < 1.5e-11)
    for Nsl in (40, 60, 80, 120):
        e = sl_eig_colloc(bg0, N=Nsl)
        print(
            f"  method 2 (colloc N={Nsl:3d}): B_c(0) = {e:.13f}  "
            f"err {abs(e - BC0_TARGET):.1e}"
        )
        check(
            f"B_c(0) by collocation (N = {Nsl}) to 1.5e-11",
            abs(e - BC0_TARGET) < 1.5e-11,
        )

    # --- B_c(alpha) table, both methods ---
    print("\n== B_c(alpha), two methods ==")
    print(
        f"  {'alpha':>7} {'B_c(shoot)':>13} {'B_c(colloc)':>13} {'diff':>10} "
        f"{'dBc/Bc/(aBc^2)':>15}"
    )
    alphas = [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    rows = []
    for a in alphas:
        bcs = bc_shoot(a)
        bcc = bc_colloc(a)
        slope = (
            (bcs - BC0_TARGET) / BC0_TARGET / (a * BC0_TARGET**2)
            if a > 0
            else float("nan")
        )
        rows.append((a, bcs, bcc, slope))
        print(
            f"  {a:7.4f} {bcs:13.10f} {bcc:13.10f} {abs(bcs - bcc):10.1e} {slope:15.8f}"
        )
    worst_tab = max(abs(r[1] - r[2]) / r[2] for r in rows)
    check(
        "B_c(alpha): shooting vs Robin GEP agree to 1e-9 (relative) for alpha <= 0.1",
        worst_tab < 1e-9,
        f"worst {worst_tab:.1e}",
    )

    # --- slope anchor: the reduced slope s(alpha) = dB_c/B_c/(alpha B_c^2) is
    # linear in alpha at small alpha; fit the small-alpha subset (curvature only
    # matters for alpha >~ 0.01) and cross-check with a 2-point Richardson. ---
    small = [r for r in rows[1:] if r[0] <= 0.005]
    aa = np.array([r[0] for r in small])
    sl = np.array(
        [(r[1] - BC0_TARGET) / BC0_TARGET / (r[0] * BC0_TARGET**2) for r in small]
    )
    A = np.stack([np.ones_like(aa), aa], axis=1)
    slope0 = np.linalg.lstsq(A, sl, rcond=None)[0][0]
    # 2-point Richardson from the two smallest alpha
    a1, a2 = aa[0], aa[1]
    rich = (sl[0] * a2 - sl[1] * a1) / (a2 - a1)
    print("\n  O(alpha) slope dB_c/B_c per alpha B_c^2:")
    print(
        f"    linear fit (alpha<=0.005) -> {slope0:.8f}   "
        f"(target {SLOPE_TARGET:.8f}, err {abs(slope0 - SLOPE_TARGET):.1e})"
    )
    print(
        f"    2-point Richardson        -> {rich:.8f}   err {abs(rich - SLOPE_TARGET):.1e}"
    )
    print(
        f"    => dB_c/dalpha|_0 = {slope0 * BC0_TARGET**3:.6f}  (perturbative target +28.332285, bc_shift.py)"
    )
    check(
        "O(alpha) slope: Richardson reproduces the perturbative slope to 1e-6",
        abs(rich - SLOPE_TARGET) < 1e-6,
    )
    check(
        "O(alpha) slope: linear fit reproduces the perturbative slope to 2e-5",
        abs(slope0 - SLOPE_TARGET) < 2e-5,
    )

    # --- full-range beta-sweep: B_c(alpha) to beta_c <= 45, with saved modes ---
    print("\n== full-range beta-sweep (B_eig is metric-only -> no root-find) ==")
    # a parameter grid: one value per line would hide its shape
    # fmt: off
    betas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             8.0, 10.0, 13.0, 16.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]
    # fmt: on
    (alpha_s, Bc_s, beta_s, w0s, As, Ms, Ns, twomethod, Ncol) = beta_sweep(betas)

    # cross-check by three independent solvers + N-convergence
    # up to the top of the grid: alpha(beta = 45) = 0.17445
    ind = independent_crosscheck(
        [0.0, 0.005, 0.02, 0.05, 0.1, 0.15, float(alpha_s[-1])]
    )
    nconv = n_convergence()

    # One npz, written only after every check has passed: sweep +
    # J-normalised modes + rebuild metadata + the dense small-alpha scan + the
    # cross-check table.
    curve = (
        alpha_s,
        Bc_s,
        beta_s,
        w0s,
        As,
        Ms,
        Ns,
        Ncol,
    )
    curve_extra = dict(
        small_alpha=np.array([r[0] for r in rows]),
        small_Bc=np.array([r[1] for r in rows]),
        slope0=slope0,
        ind_alpha=np.array([r[0] for r in ind]),
        ind_shoot=np.array([r[1] for r in ind]),
        ind_colloc=np.array([r[2] for r in ind]),
        ind_spectral=np.array([r[3] for r in ind]),
    )

    # --- shape / range report ---
    a10 = tangent_ten_percent(alpha_s, Bc_s)
    d2 = np.gradient(np.gradient(Bc_s, alpha_s), alpha_s)
    dmeth = max((abs(b - s) for (_, b, s) in twomethod), default=float("nan"))
    ind_worst = max(r[4] for r in ind)
    print("\n== SHAPE / RANGE ==")
    print(
        f"  alpha range: {alpha_s.min():.4f} .. {alpha_s.max():.4f} "
        f"(beta_c up to {beta_s.max():.0f})"
    )
    print(f"  B_c range:   {Bc_s.min():.5f} .. {Bc_s.max():.5f}")
    print(f"  monotone increasing in alpha: {bool(np.all(np.diff(Bc_s) > 0))}")
    med2 = float(np.median(d2[2:-2]))
    print(
        f"  d2 B_c/dalpha2 (interior median) = {med2:+.3f} "
        f"({'convex' if med2 > 0 else 'concave'})"
    )
    print(
        f"  linear-in-alpha approx deviates 10% at alpha ~ "
        f"{f'{a10:.4f}' if a10 is not None else '> range (never within scan)'}"
    )
    print(f"  worst two-method |B_spec-B_shoot| over sweep: {dmeth:.1e}")
    print(f"  worst 3-solver |diff| over alpha <= {ind[-1][0]:.5f}: {ind_worst:.1e}")
    check(
        "B_c(alpha) monotone increasing on the sweep", bool(np.all(np.diff(Bc_s) > 0))
    )
    check("B_c(alpha) convex on the sweep", med2 > 0)
    check(
        "sweep: spectral vs shooting on the same brane to 1e-9 (absolute), beta <= 45",
        dmeth < 1e-9,
        f"{dmeth:.1e}",
    )
    check(
        "three onset solvers agree to 1e-8 (absolute) up to beta = 45",
        ind_worst < 1e-8,
        f"{ind_worst:.1e}",
    )
    nworst = max(abs(v[-2][1] - v[-1][1]) for v in nconv.values())
    check("N-convergence of the spectral onset at beta = 10, 45 to 1e-9", nworst < 1e-9)
    print(f"\nTOTAL runtime {time.time() - t0:.1f}s")
    nfail = sum(1 for v in CHECKS.values() if not v)
    print(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    for k, v in CHECKS.items():
        if not v:
            print(f"  FAILED: {k}")
    if nfail:
        print("cache and figure not written: a check failed")
        raise SystemExit(1)
    save_curve(*curve, **curve_extra)
    make_figure(alpha_s, Bc_s, beta_s)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
