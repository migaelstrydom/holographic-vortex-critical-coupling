"""Paper figure 5: the quartic kernel family and the alpha_* throat kernel.

Paper section 5.1.  The left panel and the throat curve's cached nodes are
read from committed caches; the throat curve is densified at small gamma by
direct solves (`dk_throat_channels.S_full`, ~1 min), because the cached
nodes are linear in gamma below 1 and so sparse on the log axis.  The interior
maximum is located by direct solves as well (bounded scalar search, ~30 s), and
checked against a cubic spline through the cached nodes.

    dk_kernel.npz           the kernel K(s;beta) of paper sec. 5.1 (derived in app. C.3) on the grid
                            23 beta in [0, 45] x 41 s/B_c in [0.02, 45]
    dk_throat_channels.npz  the full kernel on the AdS3 x R^2 throat at
                            alpha_* = 2/3 (paper app. C.4), S_full = K/C4, with
                            its photon (C4 - X) and metric (Pi) sectors; X is
                            the photon response of the coupled solve (paper
                            sec. 5.1), not the frozen-metric X_ab

The inset shows the gauge-invariant split of the throat kernel, from the
dressing identity of paper app. C.3, K = (C4 - X_ab) - Pi_dress/2.  X and Pi
of the coupled solve shift separately under a diffeomorphism along x (only
X + Pi/2 is invariant), so their split is a property of the gauge
H_tt = H_xx = 0, not of the kernel.  X_ab, the screening by the photon with
the metric frozen, is the decoupled photon BVP of paper app. C.4: its
(C4 - X_ab)/C4 = S(gamma)/S(0) of `dk_throat_lattice` (finite differences +
Richardson), checked against that module's hypergeometric Green's function at
three harmonics.  The metric part is then -Pi_dress/2C4 = S_full - (C4 - X_ab)/C4.

Left panel: K(s;beta)/C_4 against s/B_c = gamma for a subset of the beta
grid.  The figure encodes (brane norm J = int p1 w0^2 dr = 1):
  * K > 0 at every (beta, s) on the grid;
  * K(s = 0.02 B_c) suppressed, 1.552 -> 0.432 (K/C_4: 0.982 -> 0.180);
  * min_s K at the largest harmonic, rising 0.0448 -> 0.0537;
  * the first triangular shell gamma_1 = 4 pi / sqrt 3 stiffening.
Right panel: S_full(gamma) on the throat -- the negative lobe below
gamma_0 = 0.8465, saturating near -0.057 as gamma -> 0, the interior
maximum 0.0509 at gamma = 4.23 (a flat peak: S_full varies by < 1e-4 over
gamma in [3.9, 4.6]), and (inset) the near-cancellation of the photon and
metric sectors that produces it, in the gauge-invariant split.

Full JHEP text width (poster_style.DOUBLE_IN), included at natural size;
greyscale-safe, mathtext only.  Output:
backreaction/figures/kernel_family.{png,pdf}.

Run:  uv run python -m backreaction.numerics.poster_kernel       (~2 min)
"""

import numpy as np
from scipy.interpolate import CubicSpline

from backreaction import paths
from backreaction.numerics import poster_style

GAMMA_TRI = 4.0 * np.pi / np.sqrt(3.0)  # first shell, triangular lattice
GAMMA_SQ = 2.0 * np.pi  # first shell, square lattice
BETA_SHOWN = [0.0, 1.0, 4.0, 8.0, 16.0, 45.0]


# extra throat harmonics below gamma = 1.2, log-spaced, solved directly
GAMMA_DENSE = np.geomspace(0.02, 1.2, 19)


def densified_throat(throat):
    """Merge the cached throat nodes with direct solves at GAMMA_DENSE.  The
    solver must reproduce the cached nodes it shares with the dense set."""
    from backreaction.numerics import dk_throat_channels as T

    g0, S0 = throat["gamma"], throat["S"]
    new = {}
    for g in GAMMA_DENSE:
        j = int(np.argmin(np.abs(g0 - g)))
        s_new, c4, xx, pp = T.S_full(float(g))
        if abs(g0[j] - g) < 1e-12:
            assert abs(s_new - S0[j]) < 1e-6, (g, s_new, S0[j])
        new[float(g)] = (s_new, c4, xx, pp)
    cols = {k: list(throat[k]) for k in ("gamma", "S", "C4", "X", "Pi")}
    for g, (s_new, c4, xx, pp) in new.items():
        if np.min(np.abs(np.asarray(cols["gamma"]) - g)) < 1e-12:
            continue
        for k, v in zip(
            ("gamma", "S", "C4", "X", "Pi"), (g, s_new, c4, xx, pp), strict=True
        ):
            cols[k].append(v)
    order = np.argsort(cols["gamma"])
    out = {k: np.asarray(v)[order] for k, v in cols.items()}
    out["gamma0"] = throat["gamma0"]
    return out, len(new)


GAMMA_GREEN = (0.02, 0.8465, 20.0)  # where the photon split is checked


def frozen_photon(g):
    """(C4 - X_ab)/C4 on the throat at every gamma of g: the decoupled photon
    of paper app. C.4 by finite differences, normalised by its gamma -> 0
    value, against the hypergeometric Green's function (Richardson in n) at
    GAMMA_GREEN."""
    from backreaction.numerics import dk_throat_lattice as TL

    s0 = TL.S_richardson(1e-12)
    ph = np.array([TL.S_richardson(float(x)) for x in g]) / s0
    g0 = TL.S_green(1e-12, n=3000)
    dev = 0.0
    for x in GAMMA_GREEN:
        green = (4 * TL.S_green(x, n=3000) - TL.S_green(x, n=1500)) / 3 / g0
        dev = max(dev, abs(green - TL.S_richardson(x) / s0))
    assert dev < 1e-6, dev
    return ph, dev


def throat_peak(g, S, imax):
    """The interior maximum of S_full by two methods: a bounded scalar search
    with direct solves, and a cubic spline through the cached nodes.  They must
    agree; the direct value is the one plotted and printed."""
    from scipy.optimize import minimize_scalar

    from backreaction.numerics import dk_throat_channels as T

    lo, hi = float(g[imax - 1]), float(g[imax + 1])
    res = minimize_scalar(
        lambda x: -T.S_full(float(x))[0],
        bounds=(lo, hi),
        method="bounded",
        options=dict(xatol=1e-3),
    )
    assert res.success, res.message
    g_pk, S_pk = float(res.x), float(-res.fun)
    fine = np.linspace(lo, hi, 4001)
    spl = CubicSpline(g, S)(fine)
    g_sp, S_sp = float(fine[np.argmax(spl)]), float(spl.max())
    assert lo + 1e-3 < g_pk < hi - 1e-3, "the maximum should be interior"
    assert S_pk >= S[imax] - 1e-7, (S_pk, S[imax])
    assert abs(g_pk - g_sp) < 0.15 and abs(S_pk - S_sp) < 1e-4, (g_pk, g_sp)
    return g_pk, S_pk, g_sp, S_sp, int(res.nfev)


def shell_lines(ax, ymin, ymax, ytext, ink):
    """The two first reciprocal-lattice shells as thin vertical lines."""
    for x, txt, dx in [
        (GAMMA_SQ, r"$2\pi$", 0.93),
        (GAMMA_TRI, r"$4\pi/\sqrt{3}$", 1.07),
    ]:
        ax.plot([x, x], [ymin, ymax], "-", color="0.6", lw=0.5, zorder=1)
        ax.text(
            x * dx,
            ytext,
            txt,
            fontsize=7,
            color="0.35",
            ha="right" if dx < 1 else "left",
            va="top",
        )


def make_figure(grid, throat, photon, peak, path_stem):
    plt = poster_style.pyplot()
    ink = "black"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(poster_style.DOUBLE_IN, 2.6))

    # --- left: the kernel family on the beta <= 45 grid ---------------------
    beta, gamma = grid["beta"], grid["gamma"]
    ratio = grid["K"] / grid["C4"]
    shades = np.linspace(0.62, 0.0, len(BETA_SHOWN))
    # labels sit on their own curves, white-boxed and darker than the line,
    # alternating between two x positions so neighbours never stack
    xlab = (0.032, 0.17)
    for k, (b, shade) in enumerate(zip(BETA_SHOWN, shades, strict=True)):
        i = int(np.argmin(np.abs(beta - b)))
        assert abs(beta[i] - b) < 1e-12
        ax.plot(gamma, ratio[i], "-", color=str(shade), lw=1.0, zorder=3)
        x = xlab[k % 2]
        ax.text(
            x,
            float(np.interp(np.log(x), np.log(gamma), ratio[i])),
            rf"$\beta = {b:g}$",
            fontsize=7.5,
            color=str(0.55 * shade),
            ha="left",
            va="center",
            zorder=4,
            bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"),
        )
    shell_lines(ax, 0.0, 1.05, 1.01, ink)
    ax.set_xscale("log")
    ax.set_xlim(gamma[0] * 0.9, gamma[-1] * 1.1)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$s/B_c$")
    ax.set_ylabel(r"$K(s;\beta)/C_4$")
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(direction="in", length=2.5, right=True, top=True, which="both")
    ax.text(0.03, 0.04, r"$\alpha \leq 0.1745$", transform=ax.transAxes, fontsize=8)

    # --- right: the throat kernel at alpha_* ---------------------------------
    g, S = throat["gamma"], throat["S"]
    metric = S - photon  # -Pi_dress/2C4, by the dressing identity
    gamma0 = float(throat["gamma0"])

    ylo, yhi = -0.13, 0.09  # room below the axis for the inset
    bx.axhline(0.0, color="0.6", lw=0.5, zorder=1)
    bx.fill_between(g, S, 0.0, where=S < 0, color="0.85", lw=0, zorder=2)
    bx.plot(g, S, "-", color=ink, lw=1.1, zorder=3)
    bx.plot([gamma0, gamma0], [ylo, 0.0], ":", color=ink, lw=0.8, zorder=2)
    bx.text(
        gamma0 * 0.92,
        ylo + 0.004,
        rf"$\gamma_0 = {gamma0:.4f}$",
        fontsize=7.5,
        ha="right",
        va="bottom",
    )
    g_pk, S_pk = peak
    bx.plot([g_pk], [S_pk], "o", ms=3.2, color=ink, zorder=5)
    # the maximum (gamma = 4.2) is labelled from the left, clear of the shell
    # labels, which sit above the curve at the top of the panel
    bx.annotate(
        rf"max ${S_pk:.4f}$ at $\gamma = {g_pk:.1f}$",
        xy=(g_pk, S_pk),
        xytext=(0.03, 0.066),
        fontsize=7.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=ink, shrinkA=0, shrinkB=2),
    )
    shell_lines(bx, 0.0, 0.078, 0.0855, ink)
    bx.set_xscale("log")
    bx.set_xlim(g[0] * 0.9, g[-1] * 1.1)
    bx.set_ylim(ylo, yhi)
    bx.set_xlabel(r"$\gamma$")
    bx.set_ylabel(r"$K/C_4$")
    bx.tick_params(direction="in", length=2.5, right=True, top=True, which="both")
    bx.text(0.03, 0.70, r"$\alpha = \alpha_\star$", transform=bx.transAxes, fontsize=8)

    # inset: the two gauge-invariant sectors that nearly cancel
    ins = bx.inset_axes([0.57, 0.12, 0.41, 0.40])
    ins.axhline(0.0, color="0.6", lw=0.4)
    ins.plot(g, photon, "--", color=ink, lw=0.8)
    ins.plot(g, metric, "-.", color="0.45", lw=0.8)
    ins.plot(g, S, "-", color=ink, lw=0.9)
    ins.set_xscale("log")
    ins.set_xlim(g[0] * 0.9, g[-1] * 1.1)
    ins.set_ylim(-1.6, 1.6)  # headroom: the labels sit above and below the curves
    ins.set_yticks([-1, 0, 1])
    ins.set_xticks([0.1, 1, 10])
    ins.tick_params(direction="in", length=1.5, labelsize=6.5, pad=1.5, which="both")
    # both sectors decay to zero at large gamma, leaving the right-hand
    # corners free for their labels
    ins.text(
        0.97,
        0.95,
        r"$(C_4 - X_{\rm ab})/C_4$",
        transform=ins.transAxes,
        fontsize=7,
        ha="right",
        va="top",
    )
    ins.text(
        0.97,
        0.05,
        r"$-\mathcal{P}_{\rm dress}/2C_4$",
        transform=ins.transAxes,
        fontsize=7,
        color="0.35",
        ha="right",
        va="bottom",
    )

    fig.tight_layout(pad=0.3, w_pad=1.2)
    poster_style.save(fig, path_stem)


def main():
    grid = np.load(paths.data("dk_kernel.npz"))
    throat = np.load(paths.data("dk_throat_channels.npz"))

    beta, gamma, K, C4 = grid["beta"], grid["gamma"], grid["K"], grid["C4"]
    assert np.allclose(grid["sgrid"] / grid["B_c"][:, None], gamma[None, :])
    assert bool(np.all(K > 0)), "the grid kernel should be positive everywhere"
    kmin = K.min(axis=1)
    assert bool(np.all(np.argmin(K, axis=1) == len(gamma) - 1))
    k_tri = np.array([CubicSpline(gamma, row)(GAMMA_TRI) for row in K])
    k_sq = np.array([CubicSpline(gamma, row)(GAMMA_SQ) for row in K])
    # the interior rise at small s that sets in from beta = 8 (paper sec. 5.1)
    rises = [beta[i] for i in range(len(beta)) if np.any(np.diff(K[i]) > 0)]

    throat, n_dense = densified_throat(throat)
    S, g = throat["S"], throat["gamma"]
    assert np.allclose(
        S, (throat["C4"] - throat["X"] - throat["Pi"] / 2) / throat["C4"]
    )
    sign_changes = int(np.sum(np.diff(np.sign(S)) != 0))
    imax = int(np.argmax(S))
    gamma0 = float(throat["gamma0"])
    S_tri = float(CubicSpline(g, S)(GAMMA_TRI))
    S_sq = float(CubicSpline(g, S)(GAMMA_SQ))

    print("paper figure 5: the quartic kernel, grid and throat")
    print(
        f"  grid: {len(beta)} beta in [{beta[0]:g}, {beta[-1]:g}] "
        f"(alpha <= {grid['alpha'][-1]:.4f}) x {len(gamma)} harmonics "
        f"s/B_c in [{gamma[0]:g}, {gamma[-1]:g}]"
    )
    print(f"  K > 0 everywhere: min K = {K.min():.6f}")
    print(
        f"  K(s = {gamma[0]:g} B_c): {K[0, 0]:.6f} (beta = 0) -> {K[-1, 0]:.6f} (beta = 45);  "
        f"K/C_4: {K[0, 0] / C4[0, 0]:.4f} -> {K[-1, 0] / C4[-1, 0]:.4f}"
    )
    print(f"  min_s K at the largest harmonic: {kmin[0]:.6f} -> {kmin[-1]:.6f}")
    print(
        f"  first triangular shell gamma_1 = {GAMMA_TRI:.4f}: "
        f"K = {k_tri[0]:.5f} -> {k_tri[-1]:.5f}   (grid point s/B_c = 7.5: "
        f"{K[0, list(gamma).index(7.5)]:.5f} -> {K[-1, list(gamma).index(7.5)]:.5f})"
    )
    print(
        f"  first square shell gamma_1 = {GAMMA_SQ:.4f}: K = {k_sq[0]:.5f} -> {k_sq[-1]:.5f}"
    )
    print(
        f"  monotone in s for beta < {min(rises):g}; from beta = {min(rises):g} a shallow "
        f"interior maximum at s/B_c < 1 (largest rise {np.diff(K, axis=1).max():.4f})"
    )
    print(
        f"  throat (alpha_* = 2/3): {len(g)} harmonics gamma in [{g[0]:g}, {g[-1]:g}]"
    )
    print(
        f"  densified with {n_dense} direct solves on [0.02, 1.2] "
        f"(shared cached nodes reproduced to 1e-6)"
    )
    print(
        f"  S_full(gamma -> 0.02) = {S[0]:+.4f}, S_full(0.1) = "
        f"{float(np.interp(0.1, g, S)):+.4f}, sign changes: {sign_changes}"
    )
    assert bool(np.all(np.diff(S[g < 1.2]) > 0)), (
        "S_full should rise smoothly below gamma = 1.2"
    )
    print(f"  gamma_0 = {gamma0:.4f}   (zero of S_full)")
    g_pk, S_pk, g_sp, S_sp, nfev = throat_peak(g, S, imax)
    print(
        f"  interior maximum S_full = {S_pk:.4f} at gamma = {g_pk:.2f}   "
        f"({nfev} direct solves; cached-node spline: {S_sp:.4f} at {g_sp:.2f}; "
        f"best cached node {S[imax]:.4f} at {g[imax]:g})"
    )
    print(f"  S_full at the shells: square {S_sq:.4f}, triangular {S_tri:.4f}")
    photon, dev = frozen_photon(g)
    print(
        f"  gauge-invariant split (dressing identity): frozen-metric photon "
        f"(C_4 - X_ab)/C_4 by FD, against the Green's function at "
        f"gamma = {', '.join(f'{x:g}' for x in GAMMA_GREEN)}: {dev:.1e}"
    )
    for x in (0.02, gamma0, GAMMA_TRI):
        ph = float(np.interp(np.log(x), np.log(g), photon))
        sx = float(np.interp(np.log(x), np.log(g), S))
        print(
            f"    gamma = {x:.4f}: photon {ph:+.4f}, metric -P_dress/2C_4 "
            f"{sx - ph:+.4f}"
        )
    print(
        f"  (gauge H_tt = H_xx = 0 split at gamma = 0.02, not invariant: "
        f"(C_4 - X)/C_4 = {(throat['C4'][0] - throat['X'][0]) / throat['C4'][0]:+.4f}, "
        f"-P/2C_4 = {-0.5 * throat['Pi'][0] / throat['C4'][0]:+.4f})"
    )

    paths.FIGURES.mkdir(exist_ok=True)
    make_figure(grid, throat, photon, (g_pk, S_pk), str(paths.figure("kernel_family")))


if __name__ == "__main__":
    main()
