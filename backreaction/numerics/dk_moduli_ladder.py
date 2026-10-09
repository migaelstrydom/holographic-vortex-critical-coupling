"""The one-flux-quantum moduli scan on the ladder rungs: is the triangular
lattice still the minimum of the Bravais functional above alpha = 0.174, and
is the LLL functional bounded below on the modular strip?

dk_moduli.py scans C(tau; beta) over the whole modular fundamental domain on
the background grid beta <= 45 (alpha <= 0.174); this module repeats the scan
on the ladder rungs, and the two together are the scan of paper sec. 5.2 and
app. D.5.  dk_long_wavelength.py continues the kernel up the beta ladder (paper
sec. 5.1) and locates the zero of the long-wavelength kernel, K_0 = 0 at
beta = 3305.4 (alpha = 0.3506, paper sec. 5.3; beta_K0, alpha_K0 in the code),
where the long-range part of the vortex interaction turns attractive.  (It is
not an instability: the long-wavelength density stiffness of the lattice is
K_0 + 2 C_tri, and the uniform lattice holds until K_0/2 + C_tri = 0 at
alpha_L = 0.37218, paper sec. 5.3 and app. D.5; the beta_K0 rung is kept below
only as a marker.)  This module scans the
moduli space at every ladder rung (beta = 1 ... 1e8, the same `walk` as
dk_long_wavelength.py) plus the beta_K0 rung; at each it

  [1] tabulates the kernel K(gamma B_c; beta) of paper sec. 5.1 and app. C on a dense
      gamma grid, splines it, and certifies the spline against exact calls;
  [2] scans C(tau) = (1/2) sum_{(m,n) != 0} e^{-gamma/2} K(gamma B_c) over the
      fundamental domain with dk_moduli's grid, refines the minimum, and
      reports tau*, |tau* - rho|, the Hessian at rho (raw and divided by
      C_tri) and the relative margin (C_sq - C_tri)/C_tri with EXACT calls,
      which must reproduce dk_long_wavelength.npz at the same rungs;
  [3] follows the stripe edge tau = i tau2 out to tau2 = 50.  There the sum
      is dominated by the n = 0 row, gamma_m = 2 pi m^2/tau2, and Poisson
      summation gives the exact asymptotics

          C(i tau2) = (1/2) sqrt(tau2) I(beta) - (1/2) K_0 + O(e^{-c sqrt(tau2)}),
          I(beta)   = int_{-inf}^{inf} dx e^{-pi x^2} K(2 pi x^2 B_c; beta) ,

      with an oscillating remainder set by the singularity of K(s) on the
      negative s axis nearest the origin (c = sqrt(2 pi) for a pole at
      s = -B_c; the n != 0 rows add only O(e^{-pi tau2})).  So the functional
      is bounded below on the Bravais moduli space iff the Gaussian average
      I(beta) of the kernel is positive -- NOT iff K_0 > 0: a stretched
      one-flux-quantum cell samples the kernel at all s = 2 pi x^2 B_c with
      Gaussian weight, and the positive bulk of K at s ~ B_c can outweigh the
      negative lobe at s -> 0.  The script reports I(beta) at every rung and
      locates its zero on the ladder if there is one.

CHECKS
------
[M1] The wide shell generator (needed for tau2 up to 50) reproduces
     dk_moduli's multiset on the scanned domain and is modular invariant.
[M2] Spline vs exact kernel at random gammas: relative error < 1e-6 at every
     rung; C(tau*), C_tri and C_sq recomputed with exact calls agree with the
     spline values to 1e-7.
[M3] Exact-call C_tri and C_sq reproduce dk_long_wavelength.npz at every ladder rung
     to 1e-8, and the beta_K0 rung reproduces alpha_K0, B_c(alpha_K0).  Cache
     consistency through the same kernel solver, not a second method.
[M4] |tau* - rho| < 1e-4, both Hessian eigenvalues at rho positive, and the
     Z3 isotropy of the Hessian (dk_moduli.py anchor [4]; paper app. D.5) at
     every rung, to 1.4e-6: the finite-difference floor of the h = 1e-3
     Hessian is a constant 1.33e-6 relative anisotropy.
[M5] The stripe-edge asymptotics: C(i tau2)/sqrt(tau2) at tau2 = 50 agrees
     with I(beta)/2 - K_0/(2 sqrt(50)) to 5e-5 at every rung (Poisson
     summation vs direct shell sum -- two methods for the edge).
[M5b] Along both domain edges (tau1 = 0 and 1/2, tau2 to 50) the minimum of
     C is C_tri itself (at the corner rho) at every rung with I > 0, and C
     falls below C_tri only at rungs with I < 0.
[M6] beta = 45 reproduces the grid scan of dk_moduli.npz, read from the cache:
     |tau* - rho| < 1e-6, smallest Hessian eigenvalue (0.143856) to 2e-5,
     C_sq - C_tri (0.0046197) to 1e-6.
[M7] The zero of I(beta) is bracketed on the ladder and located by a secant
     in ln beta (fresh background solve per step) to |I/C_4| < 1e-6.  This
     stopping test leaves the secant point a few 1e-6 from the zero in alpha
     (the cached alpha_I_zero = 0.4939766 has |I/C_4| = 4e-7); it is a
     starting point, not the quoted alpha_I.  alpha_i_uncertainty.py refines
     it with fresh solves to alpha_I = 0.4939747 +/- 2e-7 (paper app. D.5).

Run:  uv run python -m backreaction.numerics.dk_moduli_ladder        (~7 min)
      uv run python -m backreaction.numerics.dk_moduli_ladder --quick (to 1e4; writes nothing)
      uv run python -m backreaction.numerics.dk_moduli_ladder --figure
"""

import sys
import time

import matplotlib
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from backreaction import paths  # noqa: E402
from backreaction.numerics import dk_long_wavelength as CL  # noqa: E402
from backreaction.numerics import dk_moduli as M  # noqa: E402

T0 = time.time()
CHECKS = {}
OUT = paths.data("dk_moduli_ladder.npz")
FIG = paths.figure("dk_moduli_ladder.png")
LONGWAVE = paths.data("dk_long_wavelength.npz")
MODULI = paths.data("dk_moduli.npz")

RHO, TAU_SQ, FLOOR = M.RHO, M.TAU_SQ, M.WEIGHT_FLOOR
GAM_MIN, GAM_MAX, NGRID = 0.02, 44.0, 320  # spline grid (geometric)
STRIPE_TAU2 = np.geomspace(np.sqrt(3.0) / 2.0, 50.0, 48)
WIDE_BOX = 32  # enumeration half-width for tau2 <= 50


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# shells, wide enough for the stripe edge
# ---------------------------------------------------------------------------
_a = np.arange(-WIDE_BOX, WIDE_BOX + 1)
_m, _n = (x.ravel().astype(float) for x in np.meshgrid(_a, _a, indexing="ij"))
_keep = (_m != 0) | (_n != 0)
_MW, _NW = _m[_keep], _n[_keep]


def gammas_wide(tau, floor=FLOOR):
    """Same multiset as dk_moduli.gammas but with a box that holds tau2 = 50."""
    t1, t2 = tau.real, tau.imag
    g = (2.0 * np.pi / t2) * ((_MW + _NW * t1) ** 2 + (_NW * t2) ** 2)
    keep = g <= -2.0 * np.log(floor)
    edge = (np.abs(_MW) == WIDE_BOX) | (np.abs(_NW) == WIDE_BOX)
    assert not (keep & edge).any(), "wide enumeration box too small"
    return np.sort(g[keep])


def C_wide(tau, kfun):
    g = gammas_wide(tau)
    return 0.5 * float(np.sum(np.exp(-0.5 * g) * kfun(g)))


# ---------------------------------------------------------------------------
# per-rung kernel spline and the stripe asymptotics
# ---------------------------------------------------------------------------
def kernel_spline(rung):
    grid = np.geomspace(GAM_MIN, GAM_MAX, NGRID)
    vals = np.array([rung.K(float(g)) for g in grid])
    return CubicSpline(grid, vals), grid, vals


def exact_kfun(rung):
    return lambda g: np.array([rung.K(float(x)) for x in np.atleast_1d(g)])


def stripe_coefficient(spl):
    """I = int dx e^{-pi x^2} K(2 pi x^2 B_c) = int_0^inf dg e^{-g/2} K(g B_c)
    / sqrt(2 pi g), evaluated in the x variable (no endpoint singularity).
    Below gamma = 0.02 (x < 0.0564) the spline extrapolates.  That interval
    carries 11.3% of the Gaussian weight, but K is analytic in s at fixed beta
    (its nearest singularity is on the negative s axis) and the cubic
    extrapolation follows it: `analyse_rung`
    measures the effect on I against exact kernel calls there (`I_small_dev`,
    asserted below 1e-5 C4 in [M2])."""
    val, err = quad(
        lambda x: np.exp(-np.pi * x * x) * float(spl(2.0 * np.pi * x * x)),
        0.0,
        np.sqrt(GAM_MAX / (2.0 * np.pi)),
        limit=200,
        epsabs=1e-12,
        epsrel=1e-10,
    )
    return 2.0 * val, 2.0 * err


def stripe_profile(spl, tau1=0.0):
    return np.array([C_wide(complex(tau1, t2), spl) for t2 in STRIPE_TAU2])


# ---------------------------------------------------------------------------
def analyse_rung(rung, rng):
    spl, ggrid, gvals = kernel_spline(rung)
    K0 = rung.K0()

    # spline certification at random gammas (exact calls)
    gt = rng.uniform(GAM_MIN, 20.0, 6)
    ex = exact_kfun(rung)(gt)
    serr = float(np.abs(spl(gt) - ex).max() / max(np.abs(ex).max(), 1e-30))

    # the scan (dk_moduli's grid, refinement and Hessian machinery)
    _t1s, _t2s, _Cg, (cbest, pbest) = M.scan_domain(spl)
    tstar, cstar = M.refine(spl, pbest)
    tau_star = complex(tstar[0], tstar[1])
    c_tri = M.C_of_tau(RHO, spl)
    c_sq = M.C_of_tau(TAU_SQ, spl)
    H = M.hessian(spl, RHO)
    ev = np.linalg.eigvalsh(H)

    # exact-call recomputation at the minimum and both anchor lattices
    kex = exact_kfun(rung)
    c_tri_ex = M.C_of_tau(RHO, kex)
    c_sq_ex = M.C_of_tau(TAU_SQ, kex)
    c_star_ex = M.C_of_tau(tau_star, kex)
    exact_dev = max(abs(c_tri_ex - c_tri), abs(c_sq_ex - c_sq), abs(c_star_ex - cstar))

    # the stripe edge, both domain edges, and its Poisson asymptotics
    prof0 = stripe_profile(spl, 0.0)
    prof1 = stripe_profile(spl, 0.5)
    I, Ierr = stripe_coefficient(spl)
    # the spline-extrapolated region gamma < GAM_MIN of the stripe integral:
    # bound its effect on I with exact kernel calls (Simpson on 5 points in x)
    x0 = np.sqrt(GAM_MIN / (2.0 * np.pi))
    xs = np.linspace(x0 / 5, x0, 5)
    dK = np.abs(spl(2.0 * np.pi * xs**2) - kex(2.0 * np.pi * xs**2))
    I_small_dev = float(x0 * np.max(np.exp(-np.pi * xs**2) * dK))
    t2max = STRIPE_TAU2[-1]
    asym = 0.5 * np.sqrt(t2max) * I - 0.5 * K0
    stripe_dev = abs(prof0[-1] - asym) / max(abs(asym), 1e-12)

    return dict(
        beta=rung.beta,
        alpha=rung.alpha,
        Bc=rung.Bc,
        C4=rung.C4,
        K0=K0,
        I_small_dev=I_small_dev,
        tau1=tstar[0],
        tau2=tstar[1],
        dtau=abs(tau_star - RHO),
        Cmin=cstar,
        C_tri=c_tri_ex,
        C_sq=c_sq_ex,
        ev0=ev[0],
        ev1=ev[1],
        serr=serr,
        exact_dev=exact_dev,
        I=I,
        Ierr=Ierr,
        stripe_dev=stripe_dev,
        prof0=prof0,
        prof1=prof1,
        ggrid=ggrid,
        gvals=gvals,
    )


def figure(path=OUT):
    d = np.load(path)
    beta, alpha = d["beta"], d["alpha"]
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.3))

    # (a) stripe edge: C(i tau2)/sqrt(tau2) -> I(beta)/2
    t2 = d["stripe_tau2"]
    cmap = plt.get_cmap("viridis")
    for i in range(len(beta)):
        if beta[i] not in (1.0, 45.0, 1e2, 1e3, 3e3, 1e4, 1e5, 1e6, 1e8):
            continue
        col = cmap(np.log10(beta[i]) / 8.0)
        ax[0].plot(
            t2,
            d["stripe_C0"][i] / np.sqrt(t2),
            "-",
            color=col,
            lw=1.3,
            label=rf"$\beta=10^{{{np.log10(beta[i]):.1f}}}$, $\alpha={alpha[i]:.3f}$",
        )
        ax[0].axhline(0.5 * d["I_stripe"][i], color=col, lw=0.6, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_xlabel(r"$\tau_2$  (stripe edge $\tau = i\tau_2$)")
    ax[0].set_ylabel(r"$C(i\tau_2)/\sqrt{\tau_2}$")
    ax[0].set_title(r"stripe edge, dotted: $I(\beta)/2$ (Poisson asymptote)")
    ax[0].legend(fontsize=7, loc="upper right")
    ax[0].grid(alpha=0.25, lw=0.5)

    # (b) the well at rho and the gap to the square, normalised by C_tri
    ax[1].plot(
        alpha,
        d["hess_ev"][:, 0] / d["C_tri"],
        "o-",
        ms=4,
        color="#c1272d",
        label=r"$\lambda_{\min}(\nabla^2C|_\rho)/C_{\rm tri}$",
    )
    ax[1].plot(
        alpha,
        (d["C_sq"] - d["C_tri"]) / d["C_tri"],
        "s-",
        ms=4,
        color="#1f6feb",
        label=r"$(C_{\rm sq}-C_{\rm tri})/C_{\rm tri}$",
    )
    try:
        m = np.load(MODULI)
        ax[1].plot(
            m["alpha"],
            m["hess_ev"][:, 0] / m["C_tri"],
            ".",
            ms=3,
            color="#c1272d",
            alpha=0.5,
            label=r"grid scan ($\beta\leq45$)",
        )
        ax[1].plot(
            m["alpha"],
            (m["C_sq"] - m["C_tri"]) / m["C_tri"],
            ".",
            ms=3,
            color="#1f6feb",
            alpha=0.5,
        )
    except FileNotFoundError:
        pass
    ax[1].axvline(float(d["alpha_K0zero"]), color="0.4", lw=0.8, ls="--")
    ax[1].text(
        float(d["alpha_K0zero"]), 0.02, r" $\alpha_{K_0}$", fontsize=8, color="0.3"
    )
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_title(r"the triangular well along the ladder")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25, lw=0.5)

    # (c) K_0/C4 vs I/C4: long-range attraction vs Bravais boundedness
    ax[2].plot(
        alpha,
        d["K0"] / d["C4"],
        "o-",
        ms=4,
        color="#c1272d",
        label=r"$K_0/C_4$ (long range)",
    )
    ax[2].plot(
        alpha,
        d["I_stripe"] / d["C4"],
        "s-",
        ms=4,
        color="#1f6feb",
        label=r"$I(\beta)/C_4$ (Bravais stripe edge)",
    )
    ax[2].axhline(0.0, color="k", lw=0.6)
    ax[2].axvline(float(d["alpha_K0zero"]), color="0.4", lw=0.8, ls="--")
    ax[2].set_xlabel(r"$\alpha$")
    ax[2].set_title(r"two positivity criteria")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.25, lw=0.5)
    fig.suptitle(
        rf"one-flux-quantum moduli scan on the $\beta$ ladder: "
        rf"$\tau^*=\rho$ at every rung to $\alpha={alpha[-1]:.3f}$; "
        rf"$C$ bounded below on the Bravais strip for $\alpha<\alpha_I="
        rf"{float(d['alpha_I_zero']):.3f}$ ($\alpha_{{K_0}}={float(d['alpha_K0zero']):.3f}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=160)
    log(f"saved {FIG}")


def main():
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(11)
    lwc = np.load(LONGWAVE)
    beta_K0, alpha_K0, Bc_K0 = (
        float(lwc["beta_K0zero"]),
        float(lwc["alpha_K0zero"]),
        float(lwc["Bc_K0zero"]),
    )

    # [M1] the wide shell generator
    worst = 0.0
    for _ in range(12):
        tau = complex(rng.uniform(-0.5, 0.5), rng.uniform(0.9, 4.5))
        a, b = M.gammas(tau), gammas_wide(tau)
        worst = max(worst, abs(len(a) - len(b)), float(np.abs(a - b).max()))
    worst_mod = 0.0
    for _ in range(8):
        tau = complex(rng.uniform(-0.5, 0.5), rng.uniform(5.0, 40.0))
        base = gammas_wide(tau)
        for img in (tau + 1.0, -1.0 / tau):
            g = gammas_wide(img)
            k = min(len(base), len(g))
            worst_mod = max(
                worst_mod,
                float(np.abs(base[:k] - g[:k]).max()),
                abs(len(base) - len(g)),
            )
    check(
        "[M1] wide shell generator = dk_moduli on tau2 <= 4.5, modular to tau2 = 40",
        worst < 1e-10 and worst_mod < 1e-10,
        f"max |d gamma| = {worst:.1e}, modular {worst_mod:.1e}",
    )

    # the ladder, plus the beta_K0 rung
    rungs = CL.walk(max_beta=1e4 if quick else 1e8)
    rK = CL.rung_at(beta_K0, rungs)
    check(
        "[M3] beta_K0 rung reproduces alpha_K0 and B_c(alpha_K0) of dk_long_wavelength.npz",
        abs(rK.alpha - alpha_K0) < 1e-6 and abs(rK.Bc - Bc_K0) / Bc_K0 < 1e-6,
        f"alpha = {rK.alpha:.6f} vs {alpha_K0:.6f}, B_c = {rK.Bc:.4f} vs {Bc_K0:.4f}",
    )
    allr = sorted(rungs + [rK], key=lambda r: r.beta)

    log("scan over the fundamental domain at every rung")
    log(
        "       beta     alpha     B_c     |tau*-rho|   ev_min     ev_min/C_tri  "
        "(C_sq-C_tri)/C_tri   K_0/C4     I/C4      spline    stripe"
    )
    rows = []
    for r in allr:
        d = analyse_rung(r, rng)
        rows.append(d)
        log(
            f"  {d['beta']:9.4g}  {d['alpha']:.5f}  {d['Bc']:8.3f}  {d['dtau']:.2e}  "
            f"{d['ev0']:+.6f}  {d['ev0'] / d['C_tri']:+.5f}  "
            f"{(d['C_sq'] - d['C_tri']) / d['C_tri']:+.6f}  "
            f"{d['K0'] / d['C4']:+.6f}  {d['I'] / d['C4']:+.6f}  "
            f"{d['serr']:.1e}  {d['stripe_dev']:.1e}"
        )

    beta = np.array([d["beta"] for d in rows])
    alpha = np.array([d["alpha"] for d in rows])
    dtau = np.array([d["dtau"] for d in rows])
    ev0 = np.array([d["ev0"] for d in rows])
    ev1 = np.array([d["ev1"] for d in rows])
    ctri = np.array([d["C_tri"] for d in rows])
    csq = np.array([d["C_sq"] for d in rows])
    K0 = np.array([d["K0"] for d in rows])
    C4 = np.array([d["C4"] for d in rows])
    Istr = np.array([d["I"] for d in rows])
    serr = np.array([d["serr"] for d in rows])
    exdev = np.array([d["exact_dev"] for d in rows])
    sdev = np.array([d["stripe_dev"] for d in rows])
    prof0 = np.array([d["prof0"] for d in rows])
    prof1 = np.array([d["prof1"] for d in rows])

    # [M2] spline certification
    check(
        "[M2] spline vs exact kernel at random gammas < 1e-6, C at tau*, rho, i "
        "recomputed with exact calls < 1e-7",
        serr.max() < 1e-6 and exdev.max() < 1e-7,
        f"max rel spline error {serr.max():.1e}, max |dC| {exdev.max():.1e}",
    )
    ismall = np.array([d["I_small_dev"] for d in rows]) / C4
    check(
        "[M2] the extrapolated gamma < 0.02 part of I is accurate to 1e-5 C4",
        ismall.max() < 1e-5,
        f"max effect on I/C4 {ismall.max():.1e}",
    )

    # [M3] anchor to dk_long_wavelength.npz at the common rungs
    dev = 0.0
    for i, b in enumerate(lwc["beta"]):
        j = int(np.argmin(np.abs(beta - b)))
        if abs(beta[j] - b) > 1e-9:
            continue
        dev = max(dev, abs(ctri[j] - lwc["C_tri"][i]), abs(csq[j] - lwc["C_sq"][i]))
    check(
        "[M3] exact-call C_tri, C_sq reproduce dk_long_wavelength.npz at every ladder rung",
        dev < 1e-8,
        f"max |d| = {dev:.1e}",
    )

    # [M4] rho is the minimum of the compact-cell scan (tau2 <= 5), well convex
    # and Z3-isotropic.  Where I < 0 the functional is unbounded towards the
    # stripe edge, so there rho is a local minimum, not a global one.
    # relative anisotropy at each rung, worst rung
    iso = float(np.max(np.abs(ev1 - ev0) / ev0)) if ev0.min() > 0 else np.inf
    check(
        "[M4] |tau* - rho| < 1e-4, Hessian at rho positive and Z3-isotropic "
        "to 1.4e-6 at every rung",
        dtau.max() < 1e-4 and ev0.min() > 0.0 and iso < 1.4e-6,
        f"max |tau*-rho| = {dtau.max():.1e}, min ev = {ev0.min():+.6f}, "
        f"isotropy {iso:.1e}",
    )

    # [M5] stripe asymptotics: direct shell sum at tau2 = 50 vs Poisson
    check(
        "[M5] stripe edge: C(i 50)/sqrt(50) = I/2 - K_0/(2 sqrt 50) to 5e-5 at every rung",
        sdev.max() < 5e-5,
        f"max rel dev {sdev.max():.1e}",
    )
    # both domain edges: monotone beyond tau2 = 1.2 when I > 0, and the edge
    # minimum is C_tri (the corner rho) unless I < 0
    mono = True
    for i in range(len(rows)):
        if Istr[i] > 0:
            k = np.searchsorted(STRIPE_TAU2, 1.2)
            mono &= bool(np.all(np.diff(prof0[i][k:]) > 0)) and bool(
                np.all(np.diff(prof1[i][k:]) > 0)
            )
    edge_min = np.array(
        [min(p0.min(), p1.min()) for p0, p1 in zip(prof0, prof1, strict=True)]
    )
    below = edge_min < ctri - 1e-8  # spline-level slack at the corner
    check(
        "[M5b] edge minimum = C_tri at every rung with I > 0; below C_tri only where I < 0",
        mono and bool(np.all(~below | (Istr < 0))),
        f"monotone beyond tau2 = 1.2: {mono}; rungs with edge min < C_tri: "
        f"{[f'{b:.0e}' for b in beta[below]]} (I < 0 at "
        f"{[f'{b:.0e}' for b in beta[Istr < 0]]})",
    )

    # boundedness verdict
    ipos = Istr > 0
    if ipos.all():
        bverdict = (
            f"I(beta) > 0 at every rung to beta = {beta[-1]:.0e} "
            f"(alpha = {alpha[-1]:.4f}): C is bounded below on the Bravais "
            f"moduli space at every rung; min I/C4 = {(Istr / C4).min():+.5f} "
            f"at beta = {beta[int(np.argmin(Istr / C4))]:.3g}"
        )
        beta_I0 = np.nan
    else:
        k = int(np.argmax(~ipos))
        x0, x1 = np.log(beta[k - 1]), np.log(beta[k])
        f0, f1 = Istr[k - 1], Istr[k]
        beta_I0 = float(np.exp(x0 - f0 * (x1 - x0) / (f1 - f0)))
        bverdict = (
            f"I(beta) changes sign between beta = {beta[k - 1]:.3g} and "
            f"{beta[k]:.3g}; log-interpolated zero at beta = {beta_I0:.4g}"
        )
    log(f"  BOUNDEDNESS: {bverdict}")

    # [M7] secant refinement of the I(beta) zero (fresh solves, as dk_long_wavelength)
    alpha_I0, Bc_I0 = np.nan, np.nan
    if not np.isnan(beta_I0):
        x0, f0 = np.log(beta[k - 1]), Istr[k - 1] / C4[k - 1]
        x1, f1 = np.log(beta[k]), Istr[k] / C4[k]
        best, fbest = None, np.inf
        for _ in range(12):
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x2 = min(max(x2, min(x0, x1)), max(x0, x1))
            rung = CL.rung_at(float(np.exp(x2)), rungs)
            f2 = stripe_coefficient(kernel_spline(rung)[0])[0] / rung.C4
            log(
                f"    beta = {rung.beta:.6g}  alpha = {rung.alpha:.6f}  "
                f"I/C4 = {f2:+.3e}  (N = {rung.bg.N})"
            )
            best, fbest = rung, f2
            if abs(f2) < 1e-6:
                break
            if f2 > 0:
                x0, f0 = x2, f2
            else:
                x1, f1 = x2, f2
        check(
            "[M7] secant converged for I(beta)/C4 = 0",
            best is not None and abs(fbest) < 1e-6,
            f"beta_I = {best.beta:.6g}, alpha_I = {best.alpha:.6f}, "
            f"B_c = {best.Bc:.3f}, residual {fbest:+.2e}",
        )
        beta_I0, alpha_I0, Bc_I0 = best.beta, best.alpha, best.Bc
        bverdict += (
            f"; secant zero at beta_I = {beta_I0:.1f}, alpha_I = {alpha_I0:.4f}, "
            f"B_c = {Bc_I0:.2f}"
        )

    # [M6] beta = 45 reproduces the dk_moduli.npz grid scan (read from the cache)
    j45 = int(np.argmin(np.abs(beta - 45.0)))
    grid = np.load(paths.data("dk_moduli.npz"))
    k45 = int(np.argmin(np.abs(grid["beta"] - 45.0)))
    ev_ref = float(np.min(grid["hess_ev"][k45]))
    gap_ref = float(grid["C_sq"][k45] - grid["C_tri"][k45])
    check(
        "[M6] beta = 45 reproduces dk_moduli.npz (|tau*-rho|, ev_min, gap)",
        grid["beta"][k45] == 45.0
        and dtau[j45] < 1e-6
        and abs(ev0[j45] - ev_ref) < 2e-5
        and abs((csq[j45] - ctri[j45]) - gap_ref) < 1e-6,
        f"|tau*-rho| = {dtau[j45]:.1e}, ev_min = {ev0[j45]:.6f} vs {ev_ref:.6f}, "
        f"gap = {csq[j45] - ctri[j45]:.7f} vs {gap_ref:.7f}",
    )

    jK = int(np.argmin(np.abs(beta - beta_K0)))
    j1e4 = int(np.argmin(np.abs(beta - 1e4)))
    log(
        f"SUMMARY: tau* = rho at all {len(rows)} rungs (max |tau*-rho| = "
        f"{dtau.max():.1e}); ev_min/C_tri from {ev0[0] / ctri[0]:.4f} (beta = 1) "
        f"to {ev0[jK] / ctri[jK]:.4f} (beta_K0 = {beta_K0:.1f}, alpha_K0 = "
        f"{alpha_K0:.4f}) to {ev0[-1] / ctri[-1]:.4f} (beta = {beta[-1]:.0e}); "
        f"relative margin (C_sq-C_tri)/C_tri = {(csq[0] - ctri[0]) / ctri[0]:.4f} "
        f"(beta = 1), {(csq[jK] - ctri[jK]) / ctri[jK]:.4f} (beta_K0), "
        f"{(csq[j1e4] - ctri[j1e4]) / ctri[j1e4]:.4f} (beta = 1e4); "
        f"I/C4 = {Istr[jK] / C4[jK]:+.4f} at beta_K0 while K_0/C4 = "
        f"{K0[jK] / C4[jK]:+.1e}; {bverdict}"
    )

    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        sys.exit(1)
    if quick:
        log("--quick: partial ladder, cache and figure not written")
        return
    np.savez(
        OUT,
        beta=beta,
        alpha=alpha,
        B_c=np.array([d["Bc"] for d in rows]),
        C4=C4,
        K0=K0,
        tau1=np.array([d["tau1"] for d in rows]),
        tau2=np.array([d["tau2"] for d in rows]),
        dtau=dtau,
        Cmin=np.array([d["Cmin"] for d in rows]),
        C_tri=ctri,
        C_sq=csq,
        hess_ev=np.array([[d["ev0"], d["ev1"]] for d in rows]),
        spline_err=serr,
        exact_dev=exdev,
        I_stripe=Istr,
        I_stripe_err=np.array([d["Ierr"] for d in rows]),
        stripe_dev=sdev,
        stripe_tau2=STRIPE_TAU2,
        stripe_C0=prof0,
        stripe_C_half=prof1,
        kernel_gamma=rows[0]["ggrid"],
        kernel_K=np.array([d["gvals"] for d in rows]),
        beta_K0zero=beta_K0,
        alpha_K0zero=alpha_K0,
        beta_I_zero=beta_I0,
        alpha_I_zero=alpha_I0,
        Bc_I_zero=Bc_I0,
        readme=(
            "the one-flux-quantum moduli scan (paper sec. 5.2) "
            "C(tau) = 1/2 sum_{mn!=0} e^{-gamma/2} K(gamma B_c; beta), gamma = "
            "(2pi/tau2)|m+n tau|^2, on the dk_long_wavelength beta ladder "
            "plus the rung beta_K0 where K0 = 0.  tau1, tau2: the refined global "
            "minimiser over the fundamental domain; hess_ev: Hessian at rho = "
            "e^{i pi/3}; C_tri, C_sq with exact kernel calls; K0 the s->0 "
            "kernel = K_{G=0} (numerics/dk_uniform.py); I_stripe = int dx e^{-pi x^2} K(2 pi x^2 "
            "B_c) is the Poisson coefficient of the stripe edge, C(i tau2) -> "
            "1/2 sqrt(tau2) I - K0/2: bounded below on the Bravais moduli "
            "space iff I > 0; beta_I_zero/alpha_I_zero/Bc_I_zero the secant "
            "point at |I/C4| < 1e-6, a starting estimate a few 1e-6 from the "
            "zero in alpha (refined by numerics/alpha_i_uncertainty.py).  stripe_C0 / stripe_C_half: C along tau = i tau2 "
            "and 1/2 + i tau2 on stripe_tau2.  kernel_K: K on kernel_gamma per "
            "rung (rows ordered as beta)."
        ),
    )
    log(f"saved {OUT}")
    figure()
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
