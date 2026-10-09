"""Paper figure 6: the long-wavelength kernel along the ladder and the
stability threshold alpha_L of the uniform lattice.

Everything here is read from committed caches; nothing is solved.

    dk_long_wavelength.npz  K_0 = K(s -> 0; beta), C_4 and the one-flux-quantum
                            cell functionals on the throat ladder beta = 1 ... 1e8
    dk_uniform.npz          alpha_L, the zero of the uniform quartic coefficient
                            K_G0/2 + C_tri with the directly computed K_G0 (= K_0)

The figure encodes, at fixed T = 1/pi (u_H = 1):
  * K_0/C_4, the long-range part of the vortex interaction, falling
    monotonically along the ladder, +0.72 (beta = 1) through +0.18
    (beta = 45) and +0.027 (beta = 1e3), saturating near -0.06; its zero
    (alpha = 0.3506, beta = 3305; alpha_K0 in the code, "the zero of K_0" in
    the paper) marks only where that part turns attractive, and is drawn as a
    light marker;
  * the uniform quartic coefficient (K_0/2 + C_tri)/C_4, which is also half
    the long-wavelength density stiffness (K_0 + 2 C_tri)/C_4 of the
    uniform lattice; its zero alpha_L = 0.37218 (beta_L = 6510) is the one
    regime boundary below alpha_*: there the uniform lattice becomes
    unstable to long-wavelength density modulation and the transition
    turns first order;
  * the three regimes: vortex crystal (alpha < alpha_L), first order
    (alpha_L < alpha < alpha_*), no linear charged instability
    (alpha > alpha_* = 2/3).
K_0, C_tri and C_4 are all in the brane norm J = int p1 w0^2 dr = 1, so both
curves are drawn divided by C_4.

The top axis labels ln(beta) at the alpha of the ladder, interpolated in
ln(beta) between rungs and drawn only inside the ladder's range (a
secondary axis built from np.interp clamps outside it and misplaces ticks).

0.6 x JHEP text width (poster_style.SINGLE_IN), included at natural size;
greyscale-safe, mathtext only.  Output:
backreaction/figures/density_threshold.{png,pdf}.

Run:  uv run python -m backreaction.numerics.poster_density       (~2 s)
"""

import numpy as np

from backreaction import paths
from backreaction.numerics import poster_style

ALPHA_STAR = 2.0 / 3.0


def make_figure(alpha, lnbeta, k0, ctot, alpha_K0, alpha_L, path_stem):
    plt = poster_style.pyplot()
    ink = "black"
    fig, ax = plt.subplots(figsize=(poster_style.SINGLE_IN, 2.95))

    # --- regimes: three grey levels, labelled at the top -------------------
    xmax = 0.84
    ylo, yhi = -0.17, 0.85
    ax.axvspan(0.0, alpha_L, color="0.97", lw=0, zorder=0)
    ax.axvspan(alpha_L, ALPHA_STAR, color="0.90", lw=0, zorder=0)
    ax.axvspan(ALPHA_STAR, xmax, color="0.78", lw=0, zorder=0)
    for x0, x1, txt in [
        (0.0, alpha_L, "vortex crystal"),
        (alpha_L, ALPHA_STAR, "first\norder"),
        (ALPHA_STAR, xmax, "no\ncharged\ninstability"),
    ]:
        ax.text(
            0.5 * (x0 + x1),
            0.76,
            txt,
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.25",
            style="italic",
            linespacing=1.0,
        )

    # --- the two curves on the ladder rungs --------------------------------
    ax.axhline(0.0, color="0.6", lw=0.5, zorder=1)
    ax.plot(
        alpha,
        k0,
        "-",
        color=ink,
        lw=1.1,
        zorder=3,
        label=r"$K_0/C_4$ (long range)",
    )
    ax.plot(alpha, k0, "o", ms=3.0, mfc="white", mec=ink, mew=0.8, zorder=4)
    ax.plot(
        alpha,
        ctot,
        "--",
        color=ink,
        lw=0.9,
        zorder=3,
        label=r"$(K_0/2 + C_\triangle)/C_4$ (quartic)",
    )
    ax.plot(alpha, ctot, "s", ms=2.6, mfc="white", mec=ink, mew=0.7, zorder=4)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.10, 0.82),
        fontsize=7,
        frameon=False,
        handlelength=2.2,
    )

    # --- the asymptote, the threshold and the light alpha_K0 marker ---------
    ax.axvline(ALPHA_STAR, color=ink, ls="--", lw=0.8, zorder=2)
    ax.text(
        ALPHA_STAR - 0.012,
        0.42,
        r"$\alpha_\star = 2/3$",
        rotation=90,
        ha="right",
        va="center",
        fontsize=8,
    )
    ax.plot([alpha_L], [0.0], "s", ms=4.5, color=ink, zorder=5)
    ax.annotate(
        r"$\alpha_L$",
        xy=(alpha_L, 0.0),
        xytext=(alpha_L + 0.006, -0.13),
        fontsize=8.5,
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.6, color=ink, shrinkA=0, shrinkB=2),
    )
    ax.plot([alpha_K0], [0.0], "o", ms=3.6, mfc="white", mec="0.45", mew=0.8, zorder=5)
    ax.annotate(
        r"$K_0 = 0$",
        xy=(alpha_K0, 0.0),
        xytext=(alpha_K0 - 0.08, 0.20),
        fontsize=8,
        color="0.45",
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45", shrinkA=0, shrinkB=2),
    )

    # --- top axis: ln beta at the ladder's alpha, inside the ladder only ----
    lb_ticks = np.array([0.0, 4.0, 8.0, 12.0, 16.0])
    assert lb_ticks.min() >= lnbeta[0] and lb_ticks.max() <= lnbeta[-1]
    a_ticks = np.interp(lb_ticks, lnbeta, alpha)
    top = ax.secondary_xaxis("top")
    top.set_xticks(a_ticks, [f"{t:g}" for t in lb_ticks])
    top.set_xlabel(r"$\ln\beta$ (ladder)", fontsize=8, labelpad=2)
    top.tick_params(direction="in", length=2.5, labelsize=7)

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$K_0/C_4$,  $(K_0/2 + C_\triangle)/C_4$")
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(direction="in", length=2.5, right=True)
    ax.grid(False)

    fig.tight_layout(pad=0.3)
    poster_style.save(fig, path_stem)
    return a_ticks


def main():
    lwc = np.load(paths.data("dk_long_wavelength.npz"))
    uni = np.load(paths.data("dk_uniform.npz"))
    beta, alpha, C4 = lwc["beta"], lwc["alpha"], lwc["C4"]
    k0 = lwc["K0"] / C4
    ctot = lwc["C_tot"] / C4
    assert np.allclose(lwc["C_tot"], 0.5 * lwc["K0"] + lwc["C_tri"])
    assert bool(np.all(np.diff(alpha) > 0))
    assert bool(np.all(np.diff(k0) < 0)), "K_0/C_4 should fall along the ladder"
    assert int(np.sum(np.diff(np.sign(k0)) != 0)) == 1
    assert int(np.sum(np.diff(np.sign(ctot)) != 0)) == 1
    assert bool(np.all(lwc["C_tri"] > 0)) and bool(np.all(lwc["C_sq"] > lwc["C_tri"]))
    alpha_K0 = float(lwc["alpha_K0zero"])
    alpha_L = float(uni["alpha_fo"])
    # Cache consistency, not a second method: both caches take K_0 = K_G0 from
    # dk_uniform's collocation solve (dk_long_wavelength calls it), so the two
    # copies of K_0 on the shared rungs and the two zeros of K_0/2 + C_tri must
    # agree to rounding.  A larger gap means one cache is stale.  The
    # independent second methods (shooting, small-gamma extrapolation) are
    # checked where the caches are written.
    assert abs(alpha_L - float(lwc["alpha_first_order"])) < 1e-9
    stencil = np.abs(lwc["K0"] - lwc["K0_alt"]) / C4
    shared = [int(np.argmin(np.abs(beta - b))) for b in uni["lad_beta"]]
    dk = np.max(np.abs(lwc["K0"][shared] - uni["lad_KG0"]) / C4[shared])
    assert dk < 1e-9

    def at(b):
        i = int(np.argmin(np.abs(beta - b)))
        assert abs(beta[i] - b) < 1e-9 * b
        return i

    print("paper figure 6: the long-wavelength kernel along the ladder")
    print(
        f"  ladder: {len(beta)} rungs, beta = {beta[0]:g} ... {beta[-1]:.0e}, alpha <= {alpha[-1]:.4f}"
    )
    print(
        f"  K_0/C_4 = {k0[at(1)]:+.4f} (beta = 1), {k0[at(45)]:+.4f} (beta = 45), "
        f"{k0[at(1e3)]:+.4f} (beta = 1e3), {k0[-1]:+.4f} (beta = 1e8); monotone, one sign change"
    )
    print(
        "  K_0/C_4 increments over the last decades: "
        + ", ".join(f"{d:+.4f}" for d in np.diff(k0[-5::2]))
        + "  (shrinking: saturates near -0.06)"
    )
    print(f"  extrapolation check: max |K_0 - K_0_ext|/C_4 = {stencil.max():.1e}")
    print(
        f"  cache consistency: K_0 in the two caches on {len(shared)} shared rungs "
        f"agrees to {dk:.1e} C_4"
    )
    print(
        f"  alpha_K0 = {alpha_K0:.4f}   (zero of K_0: long-range part turns attractive; light marker)"
    )
    print(
        f"  (K_0/2 + C_tri)/C_4 = {ctot[at(1)]:+.4f} (beta = 1), {ctot[-1]:+.4f} (beta = 1e8); one sign change"
    )
    print(
        f"  alpha_L = {alpha_L:.5f}   beta_L = {float(uni['beta_fo']):.0f}   "
        f"B_c u_H^2 = {float(uni['Bc_fo']):.2f}   (dk_uniform.npz; dk_long_wavelength.npz "
        f"has {float(lwc['alpha_first_order']):.6f})"
    )
    print(
        "  regimes: vortex crystal (alpha < alpha_L), first order (alpha_L < "
        "alpha < alpha_star = 2/3), no charged instability (alpha > alpha_star)"
    )

    paths.FIGURES.mkdir(exist_ok=True)
    a_ticks = make_figure(
        alpha,
        np.log(beta),
        k0,
        ctot,
        alpha_K0,
        alpha_L,
        str(paths.figure("density_threshold")),
    )
    print(
        "  top-axis ticks: "
        + ", ".join(
            f"ln beta = {t:g} at alpha = {a:.3f}"
            for t, a in zip([0, 4, 8, 12, 16], a_ticks, strict=True)
        )
    )
    assert abs(a_ticks[2] - 0.347) < 2e-3  # ln beta = 8 is the 3e3 rung region


if __name__ == "__main__":
    main()
