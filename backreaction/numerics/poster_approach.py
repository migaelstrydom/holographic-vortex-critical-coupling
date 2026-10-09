"""Paper figure 4 (section 4.6): the approach law nu(beta) near alpha_* and the
two phases that make it.

See paper sec. 4.6.  Everything
is read from committed caches; the only computation is the exact horizon phase
chi(nu) (a Gamma-function ratio) and the root of the matching condition

    nu ln(rho_*/h) = chi(nu) + phi_*(nu),    ln(rho_*/h) = (1/4) ln(beta/6) + off,

with `off` read off the top rung of `dk_throat_matching.npz` (0.0313).

    dk_throat_matching.npz   nu_tab, phi_tab (the T = 0 outer phase, nu >= 0.02),
                             the ladder (beta, L, nu_meas), A_eff at ten ln beta
    dk_throat.npz            fit_A = 7.450, fit_d = 4.535 (the ladder fit; paper sec. 4.6 quotes A = 7.45)

Below nu = 0.02 the outer phase is its derived small-nu form
phi_* = pi/2 - arctan(nu |c_0/c_1|), |c_1/c_0| = 0.035, exactly as
`dk_throat_matching` [5] extends it; that stretch is drawn dashed.

Left panel: 1/nu against ln beta -- the ladder (open circles), the matching
prediction (solid), the ladder fit 1/nu = (ln beta + d)/A (dotted), and, in the
inset that reaches ln beta = 2000, the asymptotic slope 1/(4 pi) (dashed).
Right panel: chi, phi_* and their sum against nu, with the pi limit, the
turnover scale nu = 0.035 and the numerically accessible band 0.25 <= nu <= 0.6.

Full JHEP text width (poster_style.DOUBLE_IN), included at natural size;
greyscale-safe, mathtext only.  Output:
backreaction/figures/approach_law.{png,pdf}.

Run:  uv run backreaction/numerics/poster_approach.py       (~3 s)
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from backreaction import paths
from backreaction.numerics import poster_style
from backreaction.numerics.dk_throat_matching import chi

NU_LADDER = (0.25, 0.60)  # the range the ladder reaches (paper sec. 4.6)
LNBETA_REPORT = (20, 25, 30, 100, 400, 1600)


def phases(match):
    """phi_*(nu) on [1e-3, 1]: the cached table above nu = 0.02, its small-nu
    form below.  Returns (phi, ratio, nu_switch)."""
    nu_tab, phi_tab = match["nu_tab"], match["phi_tab"]
    ratio = abs(float(match["log_ratio_b_over_a"]))  # |c_1/c_0| = 0.035
    spline = CubicSpline(nu_tab, phi_tab)
    nu_switch = float(nu_tab[0])

    def phi(n):
        n = np.asarray(n, float)
        return np.where(
            n >= nu_switch,
            spline(np.clip(n, nu_switch, None)),
            np.pi / 2 - np.arctan(n / ratio),
        )

    return phi, ratio, nu_switch


def matching_curve(lnbeta, phi, off):
    """nu(ln beta) from the matching condition, one Brent root per point."""
    out = np.empty_like(lnbeta)
    for i, lb in enumerate(lnbeta):
        L = 0.25 * (lb - np.log(6.0)) + off
        out[i] = brentq(lambda n, L=L: n * L - chi(n) - float(phi(n)), 5e-4, 0.89)
    return out


def make_figure(d, path_stem):
    plt = poster_style.pyplot()
    ink = "black"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(poster_style.DOUBLE_IN, 2.6))

    # ------------------------------------------------------------------ left
    lb_main = d["lb_main"]
    ax.plot(
        lb_main,
        1 / d["nu_main"],
        "-",
        color=ink,
        lw=1.1,
        zorder=3,
        label="matching condition",
    )
    ax.plot(
        lb_main,
        (lb_main + d["fit_d"]) / d["fit_A"],
        ":",
        color=ink,
        lw=1.2,
        zorder=2,
        label=rf"ladder fit, $A={d['fit_A']:.2f}$",
    )
    ax.plot(
        d["lb_ladder"],
        1 / d["nu_ladder"],
        "o",
        ms=3.4,
        mfc="white",
        mec=ink,
        mew=0.8,
        zorder=4,
        label="throat ladder",
    )
    ax.set_xlim(0, 30)
    ax.set_ylim(1.0, 4.6)
    ax.set_xlabel(r"$\ln\beta$")
    ax.set_ylabel(r"$1/\nu$")
    ax.set_xticks([0, 10, 20, 30])
    ax.tick_params(direction="in", length=2.5, right=True, top=True)
    ax.legend(loc="lower right", fontsize=7, frameon=False, handlelength=2.4)

    # the far view: the same three lines to ln beta = 2000, plus the asymptote
    ins = ax.inset_axes([0.115, 0.56, 0.35, 0.41])
    lb_far = d["lb_far"]
    ins.plot(lb_far, 1 / d["nu_far"], "-", color=ink, lw=1.0)
    ins.plot(lb_far, (lb_far + d["fit_d"]) / d["fit_A"], ":", color=ink, lw=1.0)
    end = lb_far[-1]
    ins.plot(
        lb_far,
        1 / d["nu_far"][-1] + (lb_far - end) / (4 * np.pi),
        "--",
        color=ink,
        lw=0.8,
        dashes=(4, 2.5),
    )
    ins.plot(
        d["lb_ladder"], 1 / d["nu_ladder"], "o", ms=1.6, mfc="white", mec=ink, mew=0.5
    )
    ins.set_xlim(0, end)
    ins.set_ylim(0, 1.05 * end / d["fit_A"])
    ins.set_xticks([0, 1000, 2000])
    ins.set_yticks([0, 100, 200])
    ins.tick_params(direction="in", length=2, labelsize=6, pad=1.5)
    ins.text(
        0.50 * end,
        0.86 * end / d["fit_A"],
        r"$1/A$",
        fontsize=6.5,
        ha="center",
        va="center",
    )
    ins.text(
        0.80 * end,
        0.80 * end / (4 * np.pi) - 8,
        r"$1/4\pi$",
        fontsize=6.5,
        ha="center",
        va="top",
    )

    # ----------------------------------------------------------------- right
    nu = d["nu_grid"]
    bx.axvspan(*NU_LADDER, color="0.90", lw=0, zorder=0)
    bx.text(
        np.sqrt(NU_LADDER[0] * NU_LADDER[1]),
        3.36,
        "ladder",
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.25",
        style="italic",
    )
    bx.axhline(np.pi, color=ink, lw=0.6, ls="--", dashes=(4, 2.5), zorder=1)
    bx.text(1.3e-3, np.pi + 0.06, r"$\pi$", fontsize=8, ha="left", va="bottom")
    bx.axvline(d["ratio"], color="0.45", lw=0.7, ls=":", zorder=1)
    bx.text(
        d["ratio"] * 1.12,
        2.78,
        r"$\nu=|c_1/c_0|$",
        fontsize=7,
        ha="left",
        va="center",
        color="0.25",
    )
    tab = nu >= d["nu_switch"]
    bx.plot(nu, d["chi_grid"] + d["phi_grid"], "-", color=ink, lw=1.3, zorder=3)
    bx.plot(nu, d["chi_grid"], "-", color=ink, lw=0.8, zorder=3)
    bx.plot(nu[tab], d["phi_grid"][tab], "-", color="0.5", lw=1.1, zorder=3)
    bx.plot(
        nu[~tab],
        d["phi_grid"][~tab],
        "--",
        color="0.5",
        lw=1.1,
        dashes=(3, 2),
        zorder=3,
    )
    bx.plot(
        nu[~tab],
        (d["chi_grid"] + d["phi_grid"])[~tab],
        "--",
        color=ink,
        lw=1.3,
        dashes=(3, 2),
        zorder=3,
    )
    bx.plot(nu[~tab], d["chi_grid"][~tab], "-", color=ink, lw=0.8, zorder=3)
    # below nu_switch phi_* is the asymptotic small-nu form, not a solve: say so
    bx.axvline(d["nu_switch"], color="0.6", lw=0.5, ls="-", zorder=1, ymax=0.08)
    bx.text(
        1.15e-3,
        0.30,
        "dashed: small-$\\nu$ form\n" + r"$\phi_\star = \pi/2 - \arctan(\nu|c_0/c_1|)$",
        fontsize=6.5,
        ha="left",
        va="center",
        color="0.3",
        zorder=2,
        bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"),
    )
    for x, key, txt, dy, ha in [
        (0.10, "sum", r"$\chi+\phi_\star$", 0.06, "left"),
        (0.11, "chi", r"$\chi$", 0.10, "center"),
        (0.11, "phi", r"$\phi_\star$", 0.10, "center"),
    ]:
        y = float(np.interp(x, nu, d[f"{key}_grid"]))
        bx.text(x, y + dy, txt, fontsize=8, ha=ha, va="bottom")

    bx.set_xscale("log")
    bx.set_xlim(1e-3, 1.0)
    bx.set_ylim(0, 3.5)
    bx.set_xlabel(r"$\nu$")
    bx.set_ylabel("phase")
    bx.set_yticks([0, 1, 2, 3])
    bx.tick_params(direction="in", length=2.5, right=True, top=True, which="both")

    fig.tight_layout(pad=0.3, w_pad=1.5)
    poster_style.save(fig, path_stem)


def main():
    match = np.load(paths.data("dk_throat_matching.npz"))
    throat = np.load(paths.data("dk_throat.npz"))
    fit_A, fit_d = float(throat["fit_A"]), float(throat["fit_d"])
    phi, ratio, nu_switch = phases(match)

    beta, L = match["beta"], match["L"]
    off = float(L[-1] - 0.25 * np.log(beta[-1] / 6.0))

    lb_main = np.linspace(np.log(1e3), 30.0, 120)
    lb_far = np.concatenate(
        [np.linspace(np.log(1e3), 40, 40), np.linspace(41, 2000, 200)]
    )
    nu_main = matching_curve(lb_main, phi, off)
    nu_far = matching_curve(lb_far, phi, off)

    nu_grid = np.logspace(-3, 0, 400)
    chi_grid = np.array([chi(n) for n in nu_grid])
    phi_grid = phi(nu_grid)

    print(
        "approach law: nu ln(rho_*/h) = chi(nu) + phi_*(nu),  L = (1/4) ln(beta/6) + off"
    )
    print(f"  off = {off:.4f}  (top rung, beta = {beta[-1]:.0e})")
    print(
        f"  |c_1/c_0| = {ratio:.4f}   (phi_* turnover scale; table used for nu >= {nu_switch:.2f})"
    )
    print(f"  ladder fit: 1/nu = (ln beta + {fit_d:.3f})/{fit_A:.3f}")
    print("  A_eff = d ln beta / d(1/nu) of the matching condition:")
    A_eff = {}
    for lb in LNBETA_REPORT:
        n1, n2 = matching_curve(np.array([lb, lb + 1.0]), phi, off)
        A_eff[lb] = 1.0 / (1 / n2 - 1 / n1)
        j = np.where(match["A_eff_lnbeta"] == lb)[0]
        cached = f"  (cache {float(match['A_eff'][j[0]]):.3f})" if len(j) else ""
        print(f"    ln beta = {lb:5d}:  A_eff = {A_eff[lb]:.3f}{cached}")
    print(f"    4 pi = {4 * np.pi:.3f}")
    assert abs(A_eff[25] / fit_A - 1) < 0.02
    assert abs(A_eff[1600] / (4 * np.pi) - 1) < 0.01
    lim = float(chi_grid[0] + phi_grid[0])
    print(f"  chi + phi_* at nu = 1e-3: {lim:.4f}  (pi = {np.pi:.4f})")
    print(
        f"  phi_* on the ladder range {NU_LADDER}: "
        f"{float(np.min(phi(np.array(NU_LADDER)))):.3f} .. {float(np.max(phi(np.array(NU_LADDER)))):.3f}"
    )

    d = dict(
        lb_main=lb_main,
        nu_main=nu_main,
        lb_far=lb_far,
        nu_far=nu_far,
        lb_ladder=np.log(beta),
        nu_ladder=match["nu_meas"],
        fit_A=fit_A,
        fit_d=fit_d,
        nu_grid=nu_grid,
        chi_grid=chi_grid,
        phi_grid=phi_grid,
        sum_grid=chi_grid + phi_grid,
        ratio=ratio,
        nu_switch=nu_switch,
    )
    paths.FIGURES.mkdir(exist_ok=True)
    make_figure(d, str(paths.figure("approach_law")))


if __name__ == "__main__":
    main()
