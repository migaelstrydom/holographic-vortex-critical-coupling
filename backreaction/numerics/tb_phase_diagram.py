"""The (T, B) phase diagram at fixed gravitational coupling alpha.

The onset met by cooling at fixed field.  This script is the source of the T_c
ratios quoted in section 5.3 of the paper (T_c/T_c(0) at alpha_L and at
alpha = 0.55); its figure is a diagnostic and is not in the paper, where the
same curve is the right axis of figure 1 (poster_bc_alpha.py).  Everything
here is read from committed caches; nothing is solved.

    bc_alpha.npz            the dense beta <= 45 grid
    dk_throat.npz           the collocation ladder to beta = 1e11
    onset_scan.npz          horizon shooting to beta ~ 1e66: the onset beyond
                            the top rung (alpha > 0.5908), including the
                            alpha = 0.62 line, is read from here -- a direct
                            solve, not a continuation
    dk_uniform.npz          alpha_L and B_c(alpha_L), the zero of the uniform
                            quartic coefficient K_G0/2 + C_tri: the vortex
                            crystal forms with a second-order onset below it,
                            and above it the transition is first order with
                            B_c a spinodal
    dk_throat_matching.npz  the matching condition nu ln(rho_*/h) = chi + phi_*,
                            kept only as a cross-check of the scan beyond the
                            top rung

Why B_c = f(alpha) T^2 exactly.  alpha is dimensionless (alpha = kappa^2/g^2,
paper sec. 2.1) and the boundary theory is a CFT, so at
fixed alpha the only scales are T and B.  The onset condition is a
dimensionless relation between them, B_c/T^2 = F(alpha).  The project computes
it in the fixed-T frame u_H = 1, T = 1/pi, where B_c u_H^2 is a pure function
of alpha; restoring units, f(alpha) = B_c/T^2 = pi^2 [B_c u_H^2](alpha).  So
each fixed-alpha phase boundary is the parabola B = f T^2, a straight line of
slope 2 on log-log axes, and T_c(B) = sqrt(B / f(alpha)).  No numerical check
is needed for this; it is dimensional analysis.

The onset data are parametrised by beta = alpha B_c^2 (u_H = 1).  f(alpha) is
built by a cubic spline of alpha against ln beta through the merged grid and
ladder, root-found at each target alpha, with B_c = sqrt(beta/alpha).

Checks printed:
  [1] grid and ladder agree on their shared rungs (beta = 1, 20, 45)
  [2] f(0) = pi^2 * 5.13126764 (the probe value)
  [3] alpha = beta/B_c^2 holds in both caches
  [4] interpolation error: cubic-in-ln-beta vs PCHIP in ln B_c(alpha)
  [5] the interpolated B_c(alpha_L) against the fresh rung of dk_uniform
  [6] the matching-condition continuation reproduces the ladder's own top
      rungs (beta >= 3e7)
  [7] the shooting scan reproduces the ladder's top rungs, and agrees with
      the matching continuation beyond them

0.6 x JHEP text width (poster_style.SINGLE_IN); greyscale, mathtext only.  Output:
backreaction/figures/tb_phase_diagram.png, plus the .pdf with --pdf.  The figure
is not in the paper (the T_c ratios printed here are, in the table of sec. 5.3),
so by default only the PNG, which git ignores, is written and a full
reproduction leaves the tree clean.

Run:  uv run python -m backreaction.numerics.tb_phase_diagram [--pdf]   (~3 s)
"""

import sys

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq

from backreaction import paths
from backreaction.numerics import poster_style
from backreaction.numerics.poster_bc_alpha import merged_curve

ALPHA_STAR = 2.0 / 3.0
BC0_PROBE = 5.13126764  # probe-limit, B_c u_H^2 at alpha = 0
CHECKS = {}


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else "")
    )


class Onset:
    """[B_c u_H^2](alpha) from the merged caches, the shooting scan past the
    top rung, and the matching continuation as a cross-check."""

    def __init__(self):
        grid = np.load(paths.data("bc_alpha.npz"))
        ladder = np.load(paths.data("dk_throat.npz"))
        self.grid, self.ladder = grid, ladder
        alpha, Bc, on_ladder, shared, worst = merged_curve(grid, ladder)
        self.alpha, self.Bc, self.on_ladder = alpha, Bc, on_ladder
        self.shared, self.worst = shared, worst
        beta = alpha * Bc**2
        pos = beta > 0
        self.lnbeta = np.log(beta[pos])
        self.spl = CubicSpline(self.lnbeta, alpha[pos])
        self.pchip = PchipInterpolator(alpha, np.log(Bc))
        self.alpha_top = float(alpha[-1])

        scan = np.load(paths.data("onset_scan.npz"))
        self.scan_alpha, self.scan_Bc = scan["alpha"], scan["B_c"]
        assert bool(np.all(np.diff(self.scan_alpha) > 0))
        self.scan_pchip = PchipInterpolator(self.scan_alpha, np.log(self.scan_Bc))

        m = np.load(paths.data("dk_throat_matching.npz"))
        self.chi = CubicSpline(m["nu_tab"], m["chi_tab"])
        self.phi = CubicSpline(m["nu_tab"], m["phi_tab"])
        self.nu_min = float(m["nu_tab"][0])
        # ln(rho_*/h) - (1/4) ln(beta/6), read off the top rung (paper app. D.3)
        self.L_off = float(m["L"][-1] - 0.25 * np.log(m["beta"][-1] / 6))

    def bc(self, a):
        """B_c u_H^2 at coupling a, inside the computed range."""
        if a == 0.0:
            return float(self.Bc[0])
        assert 0 < a <= self.alpha_top
        x = brentq(lambda x: self.spl(x) - a, self.lnbeta[0], self.lnbeta[-1])
        return float(np.sqrt(np.exp(x) / a))

    def bc_matching(self, a):
        """B_c u_H^2 from the paper sec. 4.6 matching condition, with the
        fixed-point map nu^2 = sqrt(alpha_*/alpha) - 1."""
        nu = np.sqrt(np.sqrt(ALPHA_STAR / a) - 1.0)
        assert nu > self.nu_min
        L = (self.chi(nu) + self.phi(nu)) / nu
        lnbeta = 4.0 * (L - self.L_off) + np.log(6.0)
        return float(np.sqrt(np.exp(lnbeta) / a))

    def bc_scan(self, a):
        """B_c u_H^2 from the horizon-shooting scan (any alpha it covers)."""
        assert self.scan_alpha[0] <= a <= self.scan_alpha[-1]
        return float(np.exp(self.scan_pchip(a)))

    def bc_any(self, a):
        return self.bc(a) if a <= self.alpha_top else self.bc_scan(a)


def make_figure(on, rows, alpha_L, path_stem, pdf=False):
    plt = poster_style.pyplot()
    ink = "black"
    fig, (ax, bx) = plt.subplots(
        2,
        1,
        figsize=(poster_style.SINGLE_IN, 5.9),
        gridspec_kw=dict(height_ratios=[1.25, 1.0]),
    )

    # ---------------- (a) the (T, B) plane, B_0 an arbitrary reference field
    xlo, xhi = 1e-5, 1.0
    ylo, yhi = 1e-2, 1e4
    T = np.geomspace(xlo, xhi, 400)
    for r in rows:
        f, a = r["f"], r["alpha"]
        if a > alpha_L + 1e-12:
            style = dict(color=ink, ls=(0, (4, 2)), lw=1.0)
        elif abs(a - alpha_L) < 1e-12:
            style = dict(color=ink, ls="-", lw=1.8)
        else:
            style = dict(color=ink, ls="-", lw=0.9)
        ax.plot(T, f * T**2, zorder=3, **style)
        # direct label where the line leaves the frame (right or top edge)
        lab = r"$\alpha_L$" if abs(a - alpha_L) < 1e-12 else f"{a:g}"
        y_exit = f * xhi**2
        col = ink
        if y_exit <= yhi:
            ax.text(
                xhi * 1.15, y_exit, lab, ha="left", va="center", fontsize=7, color=col
            )
        else:
            ax.text(
                np.sqrt(yhi / f),
                yhi * 1.4,
                lab,
                ha="center",
                va="bottom",
                fontsize=7,
                color=col,
            )
        # crossing of the fixed-field cooling path B = B_0
        ax.plot([1 / np.sqrt(f)], [1.0], "o", ms=2.6, color=col, zorder=5)

    # the cooling path at fixed B = B_0
    ax.annotate(
        "",
        xy=(1.6e-5, 1.0),
        xytext=(0.9, 1.0),
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35", mutation_scale=7),
        zorder=2,
    )
    ax.text(
        2.5e-4,
        0.6,
        r"cool at $B = B_0$",
        ha="center",
        va="top",
        fontsize=7,
        color="0.25",
        style="italic",
    )
    ax.text(
        1.3e-5,
        7e3,
        "normal phase unstable\n" + r"($B > B_c(T)$)",
        ha="left",
        va="top",
        fontsize=7.5,
        color="0.25",
        style="italic",
    )
    ax.text(
        0.8,
        0.3,
        "normal",
        ha="right",
        va="center",
        fontsize=7.5,
        color="0.25",
        style="italic",
    )
    ax.text(
        0.8,
        3e-2,
        r"$\alpha > \alpha_\star$: no line;"
        + "\nnormal phase linearly\nstable everywhere",
        ha="right",
        va="center",
        fontsize=6.8,
        color="0.25",
        bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="0.6", lw=0.5),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$T/\sqrt{B_0}$", labelpad=1)
    ax.set_ylabel(r"$B/B_0$")
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticks([1e-2, 1, 1e2, 1e4])
    ax.tick_params(direction="in", length=2.5, right=True, top=True)
    ax.tick_params(which="minor", length=0)
    ax.text(
        0.97, 0.97, "(a)", transform=ax.transAxes, ha="right", va="top", fontsize=8.5
    )
    # ---------------- (b) T_c / sqrt(B) = 1/sqrt(f) against alpha
    xmax = 0.79
    bx.axvspan(0.0, alpha_L, color="0.97", lw=0, zorder=0)
    bx.axvspan(alpha_L, ALPHA_STAR, color="0.90", lw=0, zorder=0)
    bx.axvspan(ALPHA_STAR, xmax, color="0.78", lw=0, zorder=0)
    for x0, x1, txt in [
        (0.0, alpha_L, "vortex crystal"),
        (alpha_L, ALPHA_STAR, "first\norder"),
        (ALPHA_STAR, xmax, "linearly\nstable"),
    ]:
        bx.text(
            0.5 * (x0 + x1),
            3.5e-6,
            txt,
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.25",
            style="italic",
            linespacing=1.0,
        )
    a_cr = np.concatenate([[0.0], np.linspace(on.alpha[1], alpha_L, 200)])
    a_fo = np.linspace(alpha_L, on.alpha_top, 200)
    for a_seg, ls_ in ((a_cr, "-"), (a_fo, (0, (4, 2)))):
        y_seg = [1 / (np.pi * np.sqrt(on.bc(a))) for a in a_seg]
        bx.plot(a_seg, y_seg, color=ink, ls=ls_, lw=1.1, zorder=3)
    # beyond the top rung: the horizon-shooting scan, a direct solve
    a_out = np.linspace(on.alpha_top, float(on.scan_alpha[-1]), 300)
    y_out = [1 / (np.pi * np.sqrt(on.bc_scan(a))) for a in a_out]
    bx.plot(a_out, y_out, color=ink, ls=(0, (4, 2)), lw=1.1, zorder=3)
    for r in rows:
        mk = "s" if abs(r["alpha"] - alpha_L) < 1e-12 else "o"
        bx.plot([r["alpha"]], [r["Tc_over_sqrtB"]], mk, ms=3.4, color=ink, zorder=5)
    bx.axvline(ALPHA_STAR, color=ink, ls="--", lw=0.8, zorder=2)
    bx.text(
        ALPHA_STAR - 0.012,
        3e-3,
        r"$\alpha_\star = 2/3$",
        rotation=90,
        ha="right",
        va="center",
        fontsize=8,
    )
    # line-style key
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=ink, lw=0.9, label=r"second order, $\alpha<\alpha_L$"),
        Line2D([], [], color=ink, lw=1.8, label=rf"$\alpha_L = {alpha_L:.4f}$"),
        Line2D(
            [],
            [],
            color=ink,
            lw=1.0,
            ls=(0, (4, 2)),
            label=r"first order (spinodal), $\alpha>\alpha_L$",
        ),
    ]
    bx.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.12),
        fontsize=6.3,
        frameon=False,
        handlelength=2.4,
        borderaxespad=0.2,
        labelspacing=0.3,
    )

    bx.set_yscale("log")
    bx.set_xlim(0.0, xmax)
    bx.set_ylim(1.5e-6, 0.3)
    bx.set_xlabel(r"$\alpha$", labelpad=1)
    bx.set_ylabel(r"$T_c/\sqrt{B}$")
    bx.set_xticks([0, 0.2, 0.4, 0.6])
    bx.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    bx.tick_params(direction="in", length=2.5, right=True, top=True)
    bx.tick_params(which="minor", length=0)
    bx.text(
        0.97, 0.96, "(b)", transform=bx.transAxes, ha="right", va="top", fontsize=8.5
    )

    fig.tight_layout(pad=0.3, h_pad=0.8)
    poster_style.save(fig, path_stem, exts=("png", "pdf") if pdf else ("png",))


def main():
    on = Onset()
    uni = np.load(paths.data("dk_uniform.npz"))
    alpha_L = float(uni["alpha_fo"])
    Bc_c_rung = float(uni["Bc_fo"])
    a_star = float(on.ladder["alpha_star"])

    print(
        "(T, B) phase diagram at fixed alpha: B_c(T) = f(alpha) T^2, f = pi^2 [B_c u_H^2]"
    )
    print(
        f"  merged onset points: {len(on.alpha)}, computed range alpha <= {on.alpha_top:.4f}"
        f" (ladder top beta = {float(on.ladder['beta'][-1]):.0e})"
    )
    print(
        f"  alpha_L = {alpha_L:.5f} (dk_uniform.npz)   alpha_star = {a_star:.4f} (dk_throat.npz)"
    )

    print("checks")
    g, lad = on.grid, on.ladder
    for b in (1.0, 20.0, 45.0):
        i = int(np.argmin(np.abs(g["beta"] - b)))
        j = int(np.argmin(np.abs(lad["beta"] - b)))
        print(
            f"        beta = {b:4.0f}:  grid B_c = {g['B_c'][i]:.10f}   ladder B_c = "
            f"{lad['B_c'][j]:.10f}   rel {abs(lad['B_c'][j] / g['B_c'][i] - 1):.1e}"
        )
    check(
        "[1] grid and ladder agree on shared rungs",
        on.worst < 1e-9,
        f"{on.shared} shared, max rel {on.worst:.1e}",
    )
    f0 = np.pi**2 * on.bc(0.0)
    f0_probe = np.pi**2 * BC0_PROBE
    check(
        "[2] f(0) = pi^2 * 5.13126764",
        abs(f0 / f0_probe - 1) < 1e-8,
        f"f(0) = {f0:.6f} vs {f0_probe:.6f}",
    )
    r3 = max(
        np.max(np.abs(g["alpha"][1:] * g["B_c"][1:] ** 2 / g["beta"][1:] - 1)),
        np.max(np.abs(lad["alpha"] * lad["B_c"] ** 2 / lad["beta"] - 1)),
    )
    check("[3] alpha = beta / B_c^2 in both caches", r3 < 1e-6, f"max rel {r3:.1e}")

    targets = [0.0, 0.1, 0.2, 0.3, alpha_L, 0.45, 0.55, 0.62]
    rows, interp_err = [], 0.0
    for a in targets:
        if a <= on.alpha_top:
            bc, src = on.bc(a), ("probe" if a == 0 else "interp")
            if a > 0:
                interp_err = max(interp_err, abs(np.exp(on.pchip(a)) / bc - 1))
        else:
            bc, src = on.bc_scan(a), "shooting"
        f = np.pi**2 * bc
        rows.append(dict(alpha=a, Bc=bc, f=f, Tc_over_sqrtB=1 / np.sqrt(f), source=src))
    check(
        "[4] interpolation: cubic-in-ln-beta vs PCHIP-in-alpha",
        interp_err < 5e-3,
        f"max rel {interp_err:.1e} at the plotted alpha",
    )
    bcc = on.bc(alpha_L)
    check(
        "[5] interpolated B_c(alpha_L) vs fresh rung",
        abs(bcc / Bc_c_rung - 1) < 1e-3,
        f"{bcc:.4f} vs {Bc_c_rung:.4f}, rel {abs(bcc / Bc_c_rung - 1):.1e}",
    )
    sel = lad["beta"] >= 3e7
    dev = [
        abs(on.bc_matching(float(a)) / float(b) - 1)
        for a, b in zip(lad["alpha"][sel], lad["B_c"][sel], strict=True)
    ]
    check(
        "[6] matching continuation reproduces ladder rungs beta >= 3e7",
        max(dev) < 1e-3,
        f"{int(sel.sum())} rungs, max rel {max(dev):.1e}",
    )

    sel = lad["beta"] >= 1e9
    dev_l = [
        abs(on.bc_scan(float(a)) / float(b) - 1)
        for a, b in zip(lad["alpha"][sel], lad["B_c"][sel], strict=True)
    ]
    a_beyond = np.linspace(on.alpha_top + 1e-3, 0.64, 12)
    dev_m = [abs(on.bc_scan(a) / on.bc_matching(a) - 1) for a in a_beyond]
    check(
        "[7] shooting scan vs ladder top rungs, and vs matching beyond them",
        max(dev_l) < 1e-3 and max(dev_m) < 1e-2,
        f"ladder {int(sel.sum())} rungs max rel {max(dev_l):.1e}; matching on "
        f"[{a_beyond[0]:.3f}, 0.64] max rel {max(dev_m):.1e}",
    )

    print("table  (T_c(B) = sqrt(B/f);  T_c/sqrt(B) = 1/sqrt(f))")
    print("  alpha    regime     B_c u_H^2      f(alpha)       T_c/sqrt(B)   source")
    for r in rows:
        a = r["alpha"]
        reg = (
            "crystal"
            if a < alpha_L - 1e-12
            else ("alpha_L" if abs(a - alpha_L) < 1e-12 else "1st order")
        )
        print(
            f"  {a:6.4f}   {reg:9s}  {r['Bc']:12.6g}   {r['f']:12.6g}   "
            f"{r['Tc_over_sqrtB']:.4e}    {r['source']}"
        )
    print(
        f"  alpha >= {ALPHA_STAR:.4f}: no onset at any (T, B); normal phase linearly stable"
    )
    t0 = rows[0]["Tc_over_sqrtB"]
    for r in rows[1:]:
        q = r["Tc_over_sqrtB"] / t0
        print(
            f"    T_c(alpha={r['alpha']:.4f}) / T_c(0) at equal B = {q:.4g}"
            f"   (T_c falls by a factor {1 / q:.3g})"
        )

    failed = [k for k, v in CHECKS.items() if not v]
    print(
        f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed"
        + (f"; FAILED: {failed}" if failed else "")
    )
    assert not failed, failed

    paths.FIGURES.mkdir(exist_ok=True)
    make_figure(
        on,
        rows,
        alpha_L,
        str(paths.figure("tb_phase_diagram")),
        pdf="--pdf" in sys.argv,
    )


if __name__ == "__main__":
    main()
