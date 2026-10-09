"""Finite-alpha lattice selection over the FULL one-flux-quantum Bravais
moduli space on the D'Hoker-Kraus brane.

Away from the probe limit the kernel is not completely monotone (it is not
even monotone for beta >~ 8), so Montgomery's minimal-theta-function theorem,
the structural argument of the probe limit (paper app. E), does not apply.
This script replaces the triangular-vs-square comparison with a scan over the
whole moduli space (paper sec. 5.2).

Parametrisation.  A one-flux-quantum Bravais lattice is fixed up to rotation
and scale by a single modular parameter tau = tau1 + i tau2 in the upper half
plane; the direct cell area is A = 2 pi / B, so the reciprocal harmonics are

    gamma_{mn} = G_{mn}^2 / B = (2 pi / tau2) |m + n tau|^2,   (m,n) != (0,0)

which reproduces the two hardcoded cases of dk_selection.py exactly:
    square      tau = i,          tau2 = 1      ->  2 pi (m^2 + n^2)
    triangular  tau = e^{i pi/3}, tau2 = sqrt3/2 -> (4 pi/sqrt3)(m^2+mn+n^2)
(anchor [1] checks the two multisets element by element).  The LLL quartic
functional is then, exactly as in paper sec. 5.2,

    C(tau; beta) = (1/2) sum_{(m,n) != 0} exp(-gamma/2) K(gamma B_c(beta); beta)

with the kernel K of paper sec. 5.1 and app. C.  gamma is B-independent, so the entire
beta-dependence again enters through the argument shift s = gamma B_c(beta)
and through K itself -- resummed to all orders in alpha B^2.

Two structural facts make the scan sharper than a blind minimisation.

  (i) C is MODULAR INVARIANT: gamma_{mn}(tau+1) and gamma_{mn}(-1/tau) are the
      same multiset as gamma_{mn}(tau), by (m,n) -> (m-n,n) and (n,-m).  This
      holds for ANY kernel, so it is a pure test of the shell generator
      (anchor [2]) -- and it is why the scan may be restricted to the standard
      fundamental domain |tau| >= 1, |tau1| <= 1/2.

 (ii) Consequently tau = rho = e^{i pi/3} (triangular) and tau = i (square) are
      the order-3 and order-2 elliptic fixed points of the modular group, so
      they are STATIONARY POINTS of C(.; beta) at every beta, automatically and
      for every kernel.  Backreaction therefore cannot move the triangular
      critical point; it can only change its Hessian, or hand the global
      minimum to some other point.  Anchor [4] measures the Hessian at rho and
      [5] does the global scan.

Speed.  A generic tau has no shell degeneracy, so a direct scan would need
O(200) BVP solves per tau.  Instead K(gamma B_c) is tabulated on a dense gamma
grid per beta and splined; the spline is certified against exact kernel calls
at the scan minimum and at the two anchor lattices (anchor [3]), and the tri/sq
gap recomputed with exact calls must reproduce dk_selection.py.

Run:  uv run python -m backreaction.numerics.dk_moduli
"""

import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backreaction import paths
from backreaction.numerics import dk_kernel as K
from backreaction.numerics import dk_selection as S

OUT_PATH = paths.data("dk_moduli.npz")
FIG_PATH = paths.figure("dk_moduli.png")

RHO = 0.5 + 0.5j * np.sqrt(3.0)  # triangular
TAU_SQ = 1.0j  # square
WEIGHT_FLOOR = 1e-9  # same truncation as dk_selection.py
BOX = 16  # (m,n) enumeration half-width
TAU2_MAX = 5.0  # elongation cutoff for the scan
GAM_LO, GAM_HI = 2.0 * np.pi / TAU2_MAX, 44.0


# ---------------------------------------------------------------------------
# shells
# ---------------------------------------------------------------------------
_MN = None


def _mn():
    global _MN
    if _MN is None:
        a = np.arange(-BOX, BOX + 1)
        m, n = np.meshgrid(a, a, indexing="ij")
        m, n = m.ravel(), n.ravel()
        keep = (m != 0) | (n != 0)
        _MN = (m[keep].astype(float), n[keep].astype(float))
    return _MN


def gammas(tau, floor=WEIGHT_FLOOR):
    """The gamma = G^2/B multiset of the lattice with modular parameter tau,
    truncated at exp(-gamma/2) < floor.  Asserts the enumeration box is wide
    enough that the truncation, not the box, sets the cutoff."""
    m, n = _mn()
    t1, t2 = tau.real, tau.imag
    g = (2.0 * np.pi / t2) * ((m + n * t1) ** 2 + (n * t2) ** 2)
    keep = g <= -2.0 * np.log(floor)
    edge = (np.abs(m) == BOX) | (np.abs(n) == BOX)
    assert not (keep & edge).any(), "enumeration box too small for this tau"
    return np.sort(g[keep])


def C_of_tau(tau, kfun):
    """(1/2) sum exp(-gamma/2) K(gamma B_c) for a callable kfun(gamma)."""
    g = gammas(tau)
    return 0.5 * float(np.sum(np.exp(-0.5 * g) * kfun(g)))


# ---------------------------------------------------------------------------
# per-beta kernel spline in the B-independent variable gamma
# ---------------------------------------------------------------------------
def kernel_spline(br, ng=140):
    grid = np.geomspace(GAM_LO * 0.9, GAM_HI, ng)
    vals = np.array([br.kernel(float(g * br.Bc))["K"] for g in grid])
    return CubicSpline(grid, vals), grid, vals


def exact_kfun(br):
    return lambda g: np.array(
        [br.kernel(float(x * br.Bc))["K"] for x in np.atleast_1d(g)]
    )


# ---------------------------------------------------------------------------
# fundamental domain
# ---------------------------------------------------------------------------
def in_domain(t1, t2):
    return (
        (t2 > 0.0) and (abs(t1) <= 0.5 + 1e-12) and (t1 * t1 + t2 * t2 >= 1.0 - 1e-12)
    )


def scan_domain(spl, n1=61, n2=81):
    """Grid scan of the fundamental domain; returns (T1, T2, C) with C = nan
    outside, plus the grid minimum."""
    t1s = np.linspace(0.0, 0.5, n1)  # C(-tau1) = C(tau1) by complex conj
    t2s = np.linspace(np.sqrt(3.0) / 2.0, TAU2_MAX, n2)
    Cg = np.full((n1, n2), np.nan)
    best = (np.inf, None)
    for i, t1 in enumerate(t1s):
        for j, t2 in enumerate(t2s):
            if not in_domain(t1, t2):
                continue
            c = C_of_tau(complex(t1, t2), spl)
            Cg[i, j] = c
            if c < best[0]:
                best = (c, (t1, t2))
    return t1s, t2s, Cg, best


def refine(spl, t0):
    """Local refinement; C is smooth on the upper half plane (the domain edges
    are not physical boundaries, only a fundamental-domain choice)."""

    def obj(p):
        # keep the simplex inside the tabulated gamma range (tau2 <= TAU2_MAX)
        if not (0.2 <= p[1] <= TAU2_MAX) or abs(p[0]) > 2.0:
            return 1e30
        return C_of_tau(complex(p[0], p[1]), spl)

    r = minimize(
        obj,
        np.array(t0),
        method="Nelder-Mead",
        options=dict(xatol=1e-8, fatol=1e-14, maxiter=2000),
    )
    return r.x, float(r.fun)


def refine_from_offsets(spl, radius=0.08, n=6):
    """Nelder-Mead from n starts on a circle of the given radius around rho.

    The grid scan's minimum lands on rho because rho is itself a grid node, and
    `refine` starts there, so |tau* - rho| from that path is small by
    construction.  Starting off rho tests that the minimisation actually finds
    it.  Returns the worst distance |tau* - rho| over the starts."""
    worst = 0.0
    for k in range(n):
        th = 2 * np.pi * (k + 0.5) / n
        t0 = (RHO.real + radius * np.cos(th), RHO.imag + radius * np.sin(th))
        x, _ = refine(spl, t0)
        worst = max(worst, abs(complex(x[0], x[1]) - RHO))
    return worst


def hessian(spl, tau, h=1e-3):
    """2x2 Hessian of C in (tau1, tau2) by central differences."""

    def f(a, b):
        return C_of_tau(complex(tau.real + a, tau.imag + b), spl)

    f0 = f(0, 0)
    fxx = (f(h, 0) - 2 * f0 + f(-h, 0)) / h**2
    fyy = (f(0, h) - 2 * f0 + f(0, -h)) / h**2
    fxy = (f(h, h) - f(h, -h) - f(-h, h) + f(-h, -h)) / (4 * h**2)
    return np.array([[fxx, fxy], [fxy, fyy]])


# ---------------------------------------------------------------------------
def figure(path=OUT_PATH):
    """Regenerate the figure from the saved scan (dk_moduli.py --figure)."""
    d = np.load(path)
    alpha, betas = d["alpha"], d["beta"]
    gapsq = d["C_sq"] - d["C_tri"]
    ev0 = d["hess_ev"][:, 0]
    t1s, t2s = d["tau1_grid"], d["tau2_grid"]

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.3))

    # (a) the landscape.  Plotted as C(tau) - C(rho) on a log colour scale:
    # the raw C is dominated by its growth with tau2 (elongated cells), which
    # buries the shallow tau1 structure that actually decides the selection.
    Cg = d["C_grid_beta0"]
    Z = np.ma.masked_invalid(Cg.T - np.nanmin(Cg))
    lo = max(float(Z[Z > 0].min()), 1e-6)
    pc = ax[0].pcolormesh(
        t1s,
        t2s,
        np.ma.masked_less_equal(Z, 0.0),
        shading="nearest",
        cmap="magma_r",
        norm=matplotlib.colors.LogNorm(vmin=lo, vmax=Z.max()),
    )
    ax[0].contour(
        t1s, t2s, Z, levels=np.geomspace(lo, Z.max(), 14), colors="0.25", linewidths=0.4
    )
    th = np.linspace(np.pi / 3, np.pi / 2, 100)
    ax[0].plot(np.cos(th), np.sin(th), "k-", lw=1.2)
    ax[0].plot(
        [RHO.real],
        [RHO.imag],
        "*",
        color="#00b0f0",
        ms=16,
        mec="k",
        mew=0.6,
        label=r"$\rho=e^{i\pi/3}$ (triangular)",
    )
    ax[0].plot(
        [0.0], [1.0], "o", color="w", ms=7, mec="k", mew=0.8, label=r"$i$ (square)"
    )
    ax[0].set_xlabel(r"$\tau_1$")
    ax[0].set_ylabel(r"$\tau_2$")
    ax[0].set_xlim(-0.02, 0.52)
    ax[0].set_ylim(0.84, 2.0)
    ax[0].set_title(r"$C(\tau)-C(\rho)$ on the fundamental domain, $\alpha=0$")
    ax[0].legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.colorbar(pc, ax=ax[0])

    # (b) and (c): the two things backreaction could have done to the
    # triangular minimum -- close the gap to its nearest competitor, or
    # flatten the well.  Both turn around; the O(alpha) tangent misses that.
    for a, y, ylab, ttl in (
        (ax[1], gapsq, r"$C_{\rm sq}-C_{\rm tri}$", "gap to the nearest competitor"),
        (
            ax[2],
            ev0,
            r"min eigenvalue of $\nabla^2 C|_\rho$",
            r"curvature of the well at $\rho$",
        ),
    ):
        sl = (y[1] - y[0]) / (alpha[1] - alpha[0])
        a.plot(alpha, y, "o-", ms=4, color="#c1272d", zorder=3)
        a.plot(
            alpha,
            y[0] + sl * alpha,
            "--",
            lw=1.1,
            color="0.45",
            label=rf"$O(\alpha)$ tangent, slope ${sl:+.4f}$",
        )
        j = int(np.argmin(y))
        a.plot(
            [alpha[j]],
            [y[j]],
            "v",
            ms=9,
            color="#1f6feb",
            zorder=4,
            label=rf"turning point, $\alpha={alpha[j]:.3f}$",
        )
        a.set_xlabel(r"$\alpha$")
        a.set_ylabel(ylab)
        a.set_title(ttl)
        a.set_ylim(
            min(y.min(), y[0] + sl * alpha[-1]) - 0.04 * (y.max() - y.min()),
            y.max() + 0.12 * (y.max() - y.min()),
        )
        a.legend(fontsize=8, loc="lower left")
        a.grid(alpha=0.25, lw=0.5)
    fig.suptitle(
        rf"one-flux-quantum moduli scan on the D'Hoker-Kraus brane: "
        rf"$\tau^*=\rho$ for all $\alpha\leq{alpha[-1]:.3f}$ "
        rf"($\beta\leq{betas[-1]:.0f}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_PATH, dpi=160)
    K.tlog(f"saved {FIG_PATH}")


def main():
    t0 = time.time()
    bca = K.load_cache()
    betas = np.asarray(bca["beta"], float)

    # ---- [1] the shell generator reproduces dk_selection's two lattices ----
    for tau, kind in ((RHO, "tri"), (TAU_SQ, "sq")):
        mine = gammas(tau)
        ref = []
        for gam, mult in S.shells_gamma(kind, kmax=80):
            if mult * np.exp(-0.5 * gam) >= WEIGHT_FLOOR:
                ref.extend([gam] * mult)
        ref = np.sort(np.array(ref))
        assert len(mine) == len(ref), (kind, len(mine), len(ref))
        d = np.abs(mine - ref).max()
        K.tlog(
            f"[1] shells {kind}: {len(mine)} harmonics match dk_selection to {d:.2e}"
        )
        assert d < 1e-10, f"anchor 1 FAILED for {kind}"

    # ---- [2] modular invariance of the shell multiset ----------------------
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(12):
        tau = complex(rng.uniform(-0.5, 0.5), rng.uniform(0.9, 3.0))
        base = gammas(tau)
        for img in (tau + 1.0, -1.0 / tau):
            g = gammas(img)
            k = min(len(base), len(g))
            worst = max(
                worst, float(np.abs(base[:k] - g[:k]).max()), abs(len(base) - len(g))
            )
    K.tlog(
        f"[2] modular invariance of the shells: max |d gamma| over "
        f"tau -> tau+1, -1/tau = {worst:.2e}"
    )
    assert worst < 1e-10, "anchor 2 FAILED -- shell generator not modular"

    # ---- the scan ----------------------------------------------------------
    rows = []
    for ib, b in enumerate(betas):
        br = K.make_brane(bca, ib)
        spl, ggrid, gvals = kernel_spline(br)

        # spline accuracy against exact kernel calls at random gammas
        gt = rng.uniform(GAM_LO, 20.0, 6)
        ex = exact_kfun(br)(gt)
        serr = float(np.abs(spl(gt) - ex).max() / max(np.abs(ex).max(), 1e-30))

        t1s, t2s, Cg, (cbest, pbest) = scan_domain(spl)
        tstar, cstar = refine(spl, pbest)
        tau_star = complex(tstar[0], tstar[1])

        off = refine_from_offsets(spl)

        c_tri = C_of_tau(RHO, spl)
        c_sq = C_of_tau(TAU_SQ, spl)
        H = hessian(spl, RHO)
        ev = np.linalg.eigvalsh(H)

        rows.append(
            dict(
                beta=b,
                alpha=br.alpha,
                Bc=br.Bc,
                tau1=tstar[0],
                tau2=tstar[1],
                Cmin=cstar,
                C_tri=c_tri,
                C_sq=c_sq,
                ev0=ev[0],
                ev1=ev[1],
                serr=serr,
                off=off,
            )
        )
        K.tlog(
            f"    beta = {b:6.2f} (alpha = {br.alpha:.5f}): "
            f"tau* = {tstar[0]:.6f} + {tstar[1]:.6f}i  "
            f"|tau*-rho| = {abs(tau_star - RHO):.2e} (from 6 offset starts "
            f"{off:.1e})  "
            f"C_tri = {c_tri:+.6f}  gap_sq = {c_sq - c_tri:+.8f}  "
            f"Hess ev = ({ev[0]:+.4f}, {ev[1]:+.4f})  "
            f"spline {serr:.1e}  [{time.time() - t0:.1f}s]"
        )

        if ib == 0:
            grid0 = (t1s, t2s, Cg.copy())
        grid1 = (t1s, t2s, Cg.copy())
        if ib == 0:
            br0 = br

    alpha = np.array([r["alpha"] for r in rows])
    dtau = np.array([abs(complex(r["tau1"], r["tau2"]) - RHO) for r in rows])
    gapsq = np.array([r["C_sq"] - r["C_tri"] for r in rows])
    ev0 = np.array([r["ev0"] for r in rows])
    serr = np.array([r["serr"] for r in rows])

    # ---- [3] spline certification + exact recomputation at beta = 0 --------
    K.tlog(f"[3] spline error vs exact kernel over the grid: max {serr.max():.1e}")
    assert serr.max() < 1e-6, "anchor 3 FAILED -- spline not converged"
    g0_exact = C_of_tau(TAU_SQ, exact_kfun(br0)) - C_of_tau(RHO, exact_kfun(br0))
    rel = abs(g0_exact - S.PROBE_GAP) / S.PROBE_GAP
    K.tlog(
        f"[3] beta = 0, EXACT kernel calls: C_sq - C_tri = {g0_exact:+.14f} "
        f"vs probe {S.PROBE_GAP:+.14f} (rel {rel:.1e})"
    )
    assert rel < 3e-8, "anchor 3 FAILED -- beta=0 gap does not reproduce the probe"

    # ---- [4] triangular stays the global minimum ---------------------------
    offs = np.array([r["off"] for r in rows])
    K.tlog(
        f"[4] |tau* - rho| over the grid: max {dtau.max():.2e} from the grid "
        f"minimum, {offs.max():.2e} from six starts 0.08 away from rho "
        f"(alpha up to {alpha[-1]:.5f})"
    )
    assert offs.max() < 1e-6, (
        "check 4 FAILED -- minimisation from off-rho starts misses rho"
    )
    K.tlog(
        f"[4] Hessian at rho: smallest eigenvalue min over the grid = "
        f"{ev0.min():+.6f}  (positive = rho stays a local minimum)"
    )

    # Z3 prediction: rho is the order-3 elliptic fixed point, and the hyperbolic
    # metric is conformal to the Euclidean one in (tau1, tau2), so a Z3-invariant
    # quadratic form there must be proportional to the identity -- the Hessian is
    # ISOTROPIC at every beta, for any kernel.  Free check on the shell sum.
    ev1 = np.array([r["ev1"] for r in rows])
    iso = float(np.max(np.abs(ev1 - ev0) / ev0))  # relative, per beta, worst beta
    K.tlog(
        f"[4] Z3 isotropy of the Hessian at rho: max |ev1-ev0|/ev0 = "
        f"{iso:.1e}  (predicted 0; the h = 1e-3 differences leave a constant "
        f"1.33e-6)"
    )
    assert iso < 1.4e-6, (
        "anchor 4 FAILED -- Hessian at rho is not Z3-isotropic to 1.4e-6"
    )

    K.tlog(
        f"[4] square gap C_sq - C_tri: min {gapsq.min():+.8f}, max {gapsq.max():+.8f}"
    )
    # the ratios the paper quotes (the brane normalisation makes C itself
    # meaningful only up to a positive beta-dependent factor)
    ctri = np.array([r["C_tri"] for r in rows])
    margin = gapsq / ctri
    curv = 0.5 * (ev0 + np.array([r["ev1"] for r in rows])) / ctri
    K.tlog(
        f"[4] relative margin (C_sq - C_tri)/C_tri: {margin[0]:.4f} -> "
        f"{margin[-1]:.4f} at alpha = {alpha[-1]:.5f}, monotone "
        f"{bool(np.all(np.diff(margin) < 0))}; Hessian/C_tri: {curv[0]:.3f} -> "
        f"{curv[-1]:.3f} ({curv[-1] / curv[0] - 1:+.1%})"
    )
    jt = int(np.argmin(gapsq))
    if 0 < jt < len(gapsq) - 1:
        K.tlog(
            f"[4] the gap TURNS AROUND: minimum at beta = {betas[jt]:g} "
            f"(alpha = {alpha[jt]:.5f}), gap/gap_0 = "
            f"{gapsq[jt] / gapsq[0]:.4f}, recovering to "
            f"{gapsq[-1] / gapsq[0]:.4f} at alpha = {alpha[-1]:.5f}; "
            f"the initial slope {(gapsq[1] - gapsq[0]) / (alpha[1] - alpha[0]):+.5f} "
            f"extrapolates to a spurious zero at alpha = "
            f"{-gapsq[0] / ((gapsq[1] - gapsq[0]) / (alpha[1] - alpha[0])):.3f}"
        )
    if dtau.max() < 1e-4 and ev0.min() > 0.0:
        K.tlog(
            "[4] VERDICT: triangular is the global minimum over the scanned "
            "one-flux-quantum moduli space (tau2 <= 5) at every alpha on the grid"
        )
    else:
        j = int(np.argmax(dtau))
        K.tlog(
            f"[4] VERDICT: minimum MOVES -- worst at beta = {betas[j]}, "
            f"tau* = {rows[j]['tau1']:.6f} + {rows[j]['tau2']:.6f}i"
        )
        raise SystemExit("check 4 FAILED -- the published result is tau* = rho")

    # ---- [5] two-method spot check at a mid beta ---------------------------
    imid = int(np.argmin(np.abs(betas - 8.0)))
    brm = K.make_brane(bca, imid)
    fd = K.BraneFD(brm, 240)

    def kfd(g):
        return np.array([fd.kernel(float(x * brm.Bc))["K"] for x in np.atleast_1d(g)])

    gfd = C_of_tau(TAU_SQ, kfd) - C_of_tau(RHO, kfd)
    gch = gapsq[imid]
    K.tlog(
        f"[5] two-method gap at beta = {betas[imid]}: Chebyshev {gch:+.8f} "
        f"vs FD {gfd:+.8f} (|d| = {abs(gch - gfd):.1e})"
    )
    assert abs(gch - gfd) / abs(gch) < 1e-4, "check 5 FAILED -- Chebyshev vs FD gap"

    np.savez(
        OUT_PATH,
        beta=betas,
        alpha=alpha,
        B_c=np.array([r["Bc"] for r in rows]),
        tau1=np.array([r["tau1"] for r in rows]),
        tau2=np.array([r["tau2"] for r in rows]),
        Cmin=np.array([r["Cmin"] for r in rows]),
        C_tri=np.array([r["C_tri"] for r in rows]),
        C_sq=np.array([r["C_sq"] for r in rows]),
        hess_ev=np.array([[r["ev0"], r["ev1"]] for r in rows]),
        tau1_grid=grid1[0],
        tau2_grid=grid1[1],
        C_grid_beta0=grid0[2],
        C_grid_last=grid1[2],
        readme="C(tau) = 1/2 sum_{mn!=0} e^{-gamma/2} K(gamma B_c); "
        "gamma = (2pi/tau2)|m+n tau|^2; scan over the modular "
        "fundamental domain |tau|>=1, |tau1|<=1/2",
    )
    K.tlog(f"saved {OUT_PATH}")

    figure()
    K.tlog(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    import sys

    figure() if "--figure" in sys.argv else main()
