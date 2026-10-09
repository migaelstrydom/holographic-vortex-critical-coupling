"""Paper figure 1: the onset field B_c(alpha), with T_c/sqrt(B) on the right axis.

Everything here is read from committed caches; nothing is solved.

    bc_alpha.npz     the dense beta <= 45 grid, alpha <= 0.1745
    dk_throat.npz    the collocation ladder to beta = 1e11, alpha -> 0.591
    onset_scan.npz   horizon shooting to beta ~ 1e66; drawn only beyond the
                     ladder, where it continues the curve towards alpha_*
    dk_uniform.npz   alpha_L, the zero of the uniform quartic coefficient
                     K_G0/2 + C_tri (K_G0 = K_0 exactly): above it the
                     uniform lattice is unstable to long-wavelength density
                     modulation and the transition is first order

The figure encodes, at fixed T = 1/pi (u_H = 1):
  * B_c(0) = 5.13126764, the probe value;
  * B_c(alpha) monotone and convex, with a vertical asymptote at alpha_* = 2/3;
  * the three regimes: vortex crystal with a second-order onset
    (alpha < alpha_L), first order with B_c a spinodal (alpha_L < alpha <
    alpha_*), no linear charged instability (alpha > alpha_*);
  * on the right axis, the same curve read as the onset temperature at fixed
    field.  At fixed alpha the only scale is T, so B_c = f(alpha) T^2 with
    f = pi^2 B_c u_H^2, and T_c/sqrt(B) = 1/(pi sqrt(B_c u_H^2)) exactly
    (0.1405 in the probe limit, zero at alpha_*).

0.6 x JHEP text width (poster_style.SINGLE_IN), included at natural size;
greyscale-safe, mathtext only.  Output:
backreaction/figures/poster_bc_alpha.{png,pdf}.

Run:  uv run backreaction/numerics/poster_bc_alpha.py       (~2 s)
"""

import numpy as np

from backreaction import paths
from backreaction.numerics import poster_style

ALPHA_STAR = 2.0 / 3.0


def merged_curve(grid, ladder, tol=1e-9):
    """Union of the dense grid and the ladder, keyed by beta; the shared rungs
    must agree.  Returns (alpha, B_c, is_ladder) sorted by alpha."""
    gb, gB, ga = grid["beta"], grid["B_c"], grid["alpha"]
    lb, lB, la = ladder["beta"], ladder["B_c"], ladder["alpha"]
    worst, shared = 0.0, 0
    keep = np.ones(len(lb), bool)
    for i, b in enumerate(lb):
        j = np.argmin(np.abs(gb - b))
        if abs(gb[j] - b) <= tol * max(b, 1.0):
            shared += 1
            worst = max(worst, abs(lB[i] - gB[j]) / gB[j])
            keep[i] = False  # keep the grid copy, mark it as a rung below
    assert shared >= 2, "expected shared rungs between the grid and the ladder"
    assert worst < 1e-9, f"shared rungs disagree: rel {worst:.1e}"
    alpha = np.concatenate([ga, la[keep]])
    Bc = np.concatenate([gB, lB[keep]])
    on_ladder = np.concatenate(
        [
            np.array([np.any(np.abs(lb - b) <= tol * max(b, 1.0)) for b in gb]),
            keep[keep],
        ]
    )
    order = np.argsort(alpha)
    return alpha[order], Bc[order], on_ladder[order], shared, worst


def make_figure(alpha, Bc, on_ladder, ext_a, ext_B, Bc0, alpha_L, Bc_c, path_stem):
    plt = poster_style.pyplot()
    ink = "black"
    fig, ax = plt.subplots(figsize=(poster_style.SINGLE_IN, 2.95))

    # --- regimes: three grey levels, labelled at the top -------------------
    xmax = 0.89
    ax.axvspan(0.0, alpha_L, color="0.97", lw=0, zorder=0)
    ax.axvspan(alpha_L, ALPHA_STAR, color="0.90", lw=0, zorder=0)
    ax.axvspan(ALPHA_STAR, xmax, color="0.78", lw=0, zorder=0)
    ytop = 1e10
    for x0, x1, txt in [
        (0.0, alpha_L, "vortex crystal"),
        (alpha_L, ALPHA_STAR, "first\norder"),
        (ALPHA_STAR, xmax, "no\ncharged\ninstability"),
    ]:
        ax.text(
            0.5 * (x0 + x1),
            1.5e9,
            txt,
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.25",
            style="italic",
            linespacing=1.0,
        )

    # --- the curve ---------------------------------------------------------
    ax.plot(alpha, Bc, "-", color=ink, lw=1.1, zorder=3)
    # continuation beyond the top rung by horizon shooting (a direct solve,
    # not an extrapolation), drawn lighter to separate it from the ladder
    ax.plot(
        np.concatenate([[alpha[-1]], ext_a]),
        np.concatenate([[Bc[-1]], ext_B]),
        "-",
        color="0.45",
        lw=1.1,
        zorder=3,
    )
    ax.plot(
        alpha[on_ladder],
        Bc[on_ladder],
        "o",
        ms=3.2,
        mfc="white",
        mec=ink,
        mew=0.8,
        zorder=4,
        label="throat ladder",
    )

    # --- the asymptote and the marked couplings ---------------------------
    ax.axvline(ALPHA_STAR, color=ink, ls="--", lw=0.8, zorder=2)
    ax.text(
        ALPHA_STAR - 0.012,
        3.0e5,
        r"$\alpha_\star = 2/3$",
        rotation=90,
        ha="right",
        va="center",
        fontsize=8,
    )
    ax.plot([alpha_L], [Bc_c], "s", ms=4.5, color=ink, zorder=5)
    ax.annotate(
        r"$\alpha_L$",
        xy=(alpha_L, Bc_c),
        xytext=(alpha_L + 0.035, Bc_c * 0.12),
        fontsize=8.5,
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=ink, shrinkA=0, shrinkB=2),
    )
    ax.plot([0.0], [Bc0], "o", ms=3.2, color=ink, zorder=5)
    ax.annotate(
        r"$B_c(0)=5.131$",
        xy=(0.0, Bc0),
        xytext=(0.03, 2.3),
        fontsize=7.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=ink, shrinkA=0, shrinkB=2),
    )

    ax.set_yscale("log")
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(1.0, ytop)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$B_c\,u_H^2$")
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_yticks([1e1, 1e3, 1e5, 1e7, 1e9])
    ax.tick_params(direction="in", length=2.5, right=False, top=True)
    ax.grid(False)

    # T_c/sqrt(B) = 1/(pi sqrt(B_c u_H^2)): a fixed monotone relabelling of
    # the same axis, so its ticks sit at B_c u_H^2 = 1/(pi t)^2
    tax = ax.twinx()
    tax.set_yscale("log")
    tax.set_ylim(ax.get_ylim())
    t_ticks = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    tax.set_yticks([1.0 / (np.pi * t) ** 2 for t in t_ticks])
    tax.set_yticklabels([rf"$10^{{{int(round(np.log10(t)))}}}$" for t in t_ticks])
    tax.minorticks_off()
    tax.set_ylabel(r"$T_c/\sqrt{B}$")
    tax.tick_params(direction="in", length=2.5)

    fig.tight_layout(pad=0.3)
    poster_style.save(fig, path_stem)


def main():
    grid = np.load(paths.data("bc_alpha.npz"))
    ladder = np.load(paths.data("dk_throat.npz"))
    scan = np.load(paths.data("onset_scan.npz"))
    uni = np.load(paths.data("dk_uniform.npz"))

    alpha, Bc, on_ladder, shared, worst = merged_curve(grid, ladder)
    Bc0 = float(grid["B_c"][0])
    alpha_L = float(uni["alpha_fo"])
    Bc_c = float(uni["Bc_fo"])
    a_star = float(ladder["alpha_star"])
    assert abs(a_star - ALPHA_STAR) < 1e-15
    assert bool(np.all(np.diff(Bc) > 0)), "B_c(alpha) should be monotone"

    # the shooting scan beyond the top rung; it must sit on the ladder curve
    sa, sB = scan["alpha"], scan["B_c"]
    on = (sa > alpha[0]) & (sa < alpha[-1])
    dev = np.max(np.abs(np.interp(sa[on], alpha, np.log(Bc)) - np.log(sB[on])))
    assert dev < 2e-2, f"scan and ladder curves disagree: {dev:.1e} in ln B_c"
    beyond = sa > alpha[-1]
    ext_a, ext_B = sa[beyond], sB[beyond]
    assert bool(np.all(np.diff(ext_a) > 0)) and bool(np.all(ext_a < ALPHA_STAR))

    print("paper figure 1: B_c(alpha) at fixed T = 1/pi")
    print(
        f"  merged points: {len(alpha)} ({shared} shared rungs, "
        f"max rel. disagreement {worst:.1e})"
    )
    print(
        f"  shooting scan vs ladder curve (interpolated, alpha <= {alpha[-1]:.3f}): "
        f"max |d ln B_c| = {dev:.1e}"
    )
    print(f"  B_c(0) = {Bc0:.7f}   (probe value)")
    tc0 = 1.0 / (np.pi * np.sqrt(Bc0))
    assert abs(tc0 - 0.1405) < 5e-5, tc0
    tcL = 1.0 / (np.pi * np.sqrt(Bc_c))
    print(
        f"  right axis: T_c/sqrt(B) = {tc0:.4f} (probe), {tcL:.4f} at alpha_L "
        f"(ratio {tcL / tc0:.3f})"
    )
    print(f"  alpha_star = {a_star:.4f} = 2/3   (vertical asymptote)")
    print(
        f"  alpha_L = {alpha_L:.5f}   B_c(alpha_L) = {Bc_c:.2f}   "
        f"(beta = {float(uni['beta_fo']):.0f})"
    )
    print(
        f"  ladder endpoint: beta = {float(ladder['beta'][-1]):.0e}, "
        f"alpha = {float(ladder['alpha'][-1]):.4f}, "
        f"B_c = {float(ladder['B_c'][-1]):.4e}"
    )
    i_top = int(np.searchsorted(ext_B, 1e10))
    print(
        f"  shooting continuation drawn to B_c = 1e10, alpha = "
        f"{float(ext_a[min(i_top, len(ext_a) - 1)]):.4f}"
    )
    print(
        "  regimes: vortex crystal (alpha < alpha_L), first order (alpha_L < "
        "alpha < alpha_star), no charged instability (alpha > alpha_star)"
    )

    paths.FIGURES.mkdir(exist_ok=True)
    make_figure(
        alpha,
        Bc,
        on_ladder,
        ext_a,
        ext_B,
        Bc0,
        alpha_L,
        Bc_c,
        str(paths.figure("poster_bc_alpha")),
    )


if __name__ == "__main__":
    main()
