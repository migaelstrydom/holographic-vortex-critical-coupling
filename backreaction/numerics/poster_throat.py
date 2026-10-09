"""Paper figure 2 (section 4.1): the magnetic brane forms its AdS_3 x R^2
throat as beta = alpha B^2 grows.

The throat is described in paper sec. 4.1 and its fixed-T-frame limit in
app. C.4.  Two stacked panels against ln r in
the fixed-T frame (horizon r_p = 1, T = 1/pi, boundary U, e^{2V}, e^{2Z} -> r^2):

  (a) the proper size e^{2V} of the flux plane: r^2 outside, frozen at
      v = sqrt(beta/6) (= B sqrt(alpha/6)) inside the throat;
  (b) the invariant beta e^{-4V}: 0 at the boundary, 6 at the AdS_3 x R^2
      fixed point (l_3^2 = 1/3), on a plateau that lengthens like
      ln r_c = (1/4) ln(beta/6).

The paper calls the third metric function Z; the code (dk_background) calls it
W.  Only V enters the figure.

Sources.  beta = 0, 1, 45 are read from the committed nodal profiles in
`bc_alpha.npz` (the dense grid of paper sec. 3.3, collocation with N = 64).  No cache stores profiles deeper than
beta = 45, so beta = 1e4 and 1e8 are obtained exactly as `poster_matching.py`
obtains its rung: Newton continuation of `dk_background.solve_colloc` up the
coarse subset of `dk_throat.LADDER` (~90 s).  Cross-checks: the ladder's
beta = 1 profile against the cached one, and every horizon invariant against
`dk_throat.npz` (the ladder of paper sec. 4.6 and app. D.2).

0.6 x JHEP text width (poster_style.SINGLE_IN), included at natural size;
greyscale, mathtext only.  Output:
backreaction/figures/throat_geometry.{png,pdf}.

Run:  uv run python -m backreaction.numerics.poster_throat    (~2 min)
"""

import time

import numpy as np

from backreaction import paths
from backreaction.numerics import dk_background as dk
from backreaction.numerics import dk_throat as dt
from backreaction.numerics import poster_style

BETAS = (0.0, 1.0, 45.0, 1e4, 1e8)
CACHED = (0.0, 1.0, 45.0)  # nodal profiles in bc_alpha.npz
RUNGS = (1.0, 20.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 3e7, 1e8)  # as poster_matching
FIXED_POINT = 6.0  # beta e^{-4V} on AdS_3 x R^2 (derivations/dk_throat.py [2])
PAPER_INV = {
    45.0: 3.58,
    1e4: 5.613,
    1e8: 5.9877,
}  # horizon values quoted in the paper (figure 2 caption, app. D.2)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# backgrounds
# ---------------------------------------------------------------------------
def cached_backgrounds():
    d = np.load(paths.data("bc_alpha.npz"))
    N = int(d["N"])
    out = {}
    for beta in CACHED:
        i = int(np.argmin(np.abs(d["beta"] - beta)))
        assert d["beta"][i] == beta, (beta, d["beta"][i])
        out[beta] = dk.Background.from_nodal(
            beta, 2.0, N, d["A"][i], d["M"][i], d["Nn"][i], np.sqrt(beta / 2.0)
        )
    return out


def ladder_backgrounds(keep):
    """Continue up the coarse ladder; return the rungs in `keep`."""
    bg, out = None, {}
    for beta, N in dt.LADDER:
        if beta not in RUNGS:
            continue
        bg = dk.solve_colloc(
            beta, N=N, tol=1e-10, maxit=200, guess=dt._regrid(bg, N) if bg else None
        )
        log(f"    beta = {beta:8.3g}  N = {N}  residual {bg.ode_residual():.1e}")
        if beta in keep:
            out[beta] = bg
    return out


def inv_of_r(bg, r):
    return bg.beta * np.exp(-4 * bg.fields_r(r)["V"])


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def make_figure(curves, path_stem):
    plt = poster_style.pyplot()
    ink = "black"
    xmax = 7.0
    # sequential greyscale: light -> dark with beta; beta = 0 is the dashed
    # AdS_5-Schwarzschild reference
    style = {
        0.0: dict(color="0.35", ls=(0, (4, 2.5)), lw=0.9),
        1.0: dict(color="0.72", ls="-", lw=1.1),
        45.0: dict(color="0.52", ls="-", lw=1.1),
        1e4: dict(color="0.28", ls="-", lw=1.2),
        1e8: dict(color=ink, ls="-", lw=1.3),
    }
    name = {0.0: "0", 1.0: "1", 45.0: "45", 1e4: "10^{4}", 1e8: "10^{8}"}

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(poster_style.SINGLE_IN, 4.2), sharex=True
    )

    # --- (a) flux-plane size -------------------------------------------------
    for beta, c in curves.items():
        ax1.plot(c["x"], c["e2V"], zorder=3, **style[beta])
        if beta > 6.0:  # r_c > r_p = 1: on the plot
            ax1.plot(
                [c["xc"]],
                [c["e2V_c"]],
                "o",
                ms=3.2,
                mfc="white",
                mec=ink,
                mew=0.8,
                zorder=4,
            )
    for beta in (1e4, 1e8):
        v = np.sqrt(beta / 6)
        ax1.text(
            0.12,
            v * 1.35,
            rf"$\beta={name[beta]}$",
            ha="left",
            va="bottom",
            fontsize=7,
            color="0.2",
        )
    ax1.annotate(
        r"$e^{2V}\to\sqrt{\beta/6}$",
        xy=(2.3, np.sqrt(1e8 / 6)),
        xytext=(2.3, 2.2e5),
        ha="center",
        va="bottom",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="-", lw=0.6, color=ink, shrinkA=0, shrinkB=2),
    )
    ax1.set_yscale("log")
    ax1.set_ylim(0.6, 2e6)
    ax1.set_yticks([1, 1e2, 1e4, 1e6])
    ax1.minorticks_off()
    handles = [
        ax1.plot([], [], **style[b], label=rf"$\beta={name[b]}$")[0] for b in BETAS
    ]
    ax1.legend(
        handles=handles,
        loc="lower right",
        fontsize=7,
        frameon=False,
        handlelength=2.4,
        borderaxespad=0.3,
        labelspacing=0.3,
    )
    ax1.set_ylabel(r"$e^{2V}$")
    ax1.tick_params(direction="in", length=2.5, right=True, top=True)
    ax1.text(0.02, 0.96, "(a)", transform=ax1.transAxes, ha="left", va="top")

    # --- (b) the invariant ---------------------------------------------------
    ax2.axhline(FIXED_POINT, color=ink, lw=0.5, ls=":", zorder=1)
    ax2.text(
        xmax - 0.08,
        FIXED_POINT + 0.12,
        r"AdS$_3\times\mathbb{R}^2$",
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.25",
        style="italic",
    )
    for beta, c in curves.items():
        if beta == 0.0:
            continue
        ax2.plot(c["x"], c["inv"], zorder=3, **style[beta])
        if beta > 6.0:  # r_c > r_p = 1: on the plot
            ax2.plot(
                [c["xc"]],
                [c["inv_c"]],
                "o",
                ms=3.2,
                mfc="white",
                mec=ink,
                mew=0.8,
                zorder=4,
            )
    # the r_c key lives in (b), where there is room to say why beta = 1 has
    # no marker: r_c lies outside the horizon r_p = 1 only for beta > 6
    rc = ax2.plot(
        [],
        [],
        "o",
        ms=3.2,
        mfc="white",
        mec=ink,
        mew=0.8,
        label="$r_c=(\\beta/6)^{1/4}$\n(inside the horizon\nfor $\\beta < 6$)",
    )
    ax2.legend(
        handles=rc,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.80),
        fontsize=7,
        frameon=False,
        handlelength=1.2,
        borderaxespad=0.3,
    )
    ax2.set_xlim(0.0, xmax)
    ax2.set_ylim(0.0, 7.2)
    ax2.set_yticks([0, 2, 4, 6])
    ax2.set_xticks(range(0, 8))
    ax2.set_xlabel(r"$\ln r$")
    ax2.set_ylabel(r"$\beta\,e^{-4V}$")
    ax2.tick_params(direction="in", length=2.5, right=True, top=True)
    ax2.text(0.02, 0.96, "(b)", transform=ax2.transAxes, ha="left", va="top")

    fig.subplots_adjust(left=0.15, right=0.97, top=0.985, bottom=0.095, hspace=0.07)
    poster_style.save(fig, path_stem)


def main():
    log("cached profiles (bc_alpha.npz): beta = 0, 1, 45")
    bgs = cached_backgrounds()
    log("ladder continuation to beta = 1e8 (as poster_matching)")
    lad = ladder_backgrounds(keep=(1.0, 1e4, 1e8))
    bgs[1e4], bgs[1e8] = lad[1e4], lad[1e8]
    thr = np.load(paths.data("dk_throat.npz"))

    print("throat geometry figure: fixed T = 1/pi, horizon r_p = 1")
    print("  [1] two sources agree at beta = 1 (cached N = 64 vs ladder N = 64):")
    dmax = dk.profile_maxdiff(bgs[1.0], lad[1.0])
    print(
        "      max profile difference "
        + ", ".join(f"{k} {v:.1e}" for k, v in dmax.items())
    )

    print("  [2] horizon invariant beta e^{-4V(r_p)} -> 6")
    print("      beta      value      6 - value   dk_throat.npz   paper")
    for beta in BETAS:
        bg = bgs[beta]
        val = float(inv_of_r(bg, np.array([1.0 + 1e-12]))[0])
        j = np.where(thr["beta"] == beta)[0]
        ref = f"{thr['inv'][j[0]]:.8f}" if len(j) else "     --    "
        pq = f"{PAPER_INV[beta]}" if beta in PAPER_INV else "--"
        print(
            f"      {beta:8.3g}  {val:.8f}  {FIXED_POINT - val:.3e}   {ref}      {pq}"
        )
        if len(j):
            assert abs(val - thr["inv"][j[0]]) < 1e-6, (beta, val, thr["inv"][j[0]])
        if beta in PAPER_INV:
            nd = len(str(PAPER_INV[beta]).split(".")[1])
            assert round(val, nd) == PAPER_INV[beta], (beta, val)

    print("  [3] boundary normalisation e^{2V}/r^2 -> 1 (also U/r^2, e^{2Z}/r^2)")
    print(
        "      beta      e^{2V}/r^2 at r=1e4   at boundary node   U/r^2 bnd   e^{2Z}/r^2 bnd"
    )
    for beta in BETAS:
        bg = bgs[beta]
        f = bg.fields_r(np.array([1e4]))
        ib = int(np.argmin(bg.sig))  # sigma = 0 node
        print(
            f"      {beta:8.3g}  {np.exp(2 * f['V'][0]) / 1e8:.10f}         "
            f"{np.exp(2 * bg.M[ib]):.10f}       {bg.A[ib]:.10f}  {np.exp(2 * bg.Nn[ib]):.10f}"
        )
        assert abs(np.exp(2 * bg.M[ib]) - 1) < 1e-8 and abs(bg.A[ib] - 1) < 1e-8
        assert abs(np.exp(2 * bg.Nn[ib]) - 1) < 1e-8

    print("  [4] r_c = (beta/6)^{1/4} against the fall of beta e^{-4V}")
    print(
        "      beta      ln r_c   inv(r_c)   inv(r_c)/6   inv(r_c)/inv(r_p)   "
        "e^{2V}(r_c)/v   ln r_half   plateau width (inv > 5.4)"
    )
    curves = {}
    x = np.linspace(0.0, 7.0, 1400)
    r = np.exp(x)
    r[0] = 1.0 + 1e-12
    for beta in BETAS:
        bg = bgs[beta]
        e2V = np.exp(2 * bg.fields_r(r)["V"])
        inv = beta / e2V**2
        c = dict(x=x, e2V=e2V, inv=inv)
        if beta > 0 and beta < 6:
            print(
                f"      {beta:8.3g}  {0.25 * np.log(beta / 6):6.3f}   "
                "(r_c < r_p: no throat outside the horizon)"
            )
        elif beta > 0:
            rc = (beta / 6) ** 0.25
            xc = np.log(rc)
            inv_c = float(inv_of_r(bg, np.array([rc]))[0])
            e2V_c = float(np.sqrt(beta / inv_c))
            c.update(xc=xc, inv_c=inv_c, e2V_c=e2V_c)
            if beta >= 1e4:
                xh = float(np.interp(-0.5 * inv[0], -inv, x))  # inv decreasing in x
                wide = x[inv > 0.9 * FIXED_POINT]
                width = f"{wide[-1] - wide[0]:.2f}" if len(wide) else "none"
                print(
                    f"      {beta:8.3g}  {xc:6.3f}   {inv_c:7.4f}   {inv_c / 6:8.4f}     "
                    f"{inv_c / inv[0]:8.4f}           {e2V_c / np.sqrt(beta / 6):6.3f}      "
                    f"{xh:6.3f}      {width}"
                )
            else:
                print(
                    f"      {beta:8.3g}  {xc:6.3f}   {inv_c:7.4f}   {inv_c / 6:8.4f}     "
                    f"{inv_c / inv[0]:8.4f}           {e2V_c / np.sqrt(beta / 6):6.3f}      "
                    f"   --       (no plateau)"
                )
        curves[beta] = c
    print(
        "      (1/4) ln(beta/6) = ln r_c; e^{2V}(r_c)/v = 2 would give inv(r_c)/6 = 1/4"
    )

    paths.FIGURES.mkdir(exist_ok=True)
    make_figure(curves, str(paths.figure("throat_geometry")))


if __name__ == "__main__":
    main()
