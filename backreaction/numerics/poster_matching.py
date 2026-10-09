"""Paper figure 3 (section 4.6): the matched-asymptotics diagram of the
marginal mode on one deep rung of the throat ladder.

See paper sec. 4.6.  The figure
shows rho w against ln(rho/h) at beta = 1e8, rho = r - r_0 the BTZ radial
coordinate of the fixed-T frame (paper app. C.4: r_0 = 1/3, h = 2/3), and lays
the two asymptotic solutions over the full numerical zero mode:

  * the exact BTZ horizon solution y_H = Re[z^a 2F1(a,a;1;1-z)], a = (1+i nu)/2,
    z = h^2/rho^2 (paper app. C.4), which in the throat is cos(nu ln(rho/h) - chi);
  * the T = 0 brane's source-free mode at alpha_* (paper sec. 4.6), rescaled by
    rho_* so that it reads cos(nu ln(rho/rho_*) + phi_*) in the throat.

They overlap across the throat, which is the matching condition
nu ln(rho_*/h) = chi + phi_*, and the throat holds less than one half-
oscillation (nu ln(rho_*/h) = 1.36 < pi at this rung).

Solves: the background ladder to beta = 1e8 (the continuation of paper sec. 4.6, the
coarse subset of `dk_throat.LADDER` that still converges) and the onset
eigenproblem on the top rung, in this process; the T = 0 brane of
`dk_throat_matching.t0_brane` in a child process (the ladder takes ~100 s, the brane ~1 s;
neither depends on the other).  Everything else comes from the caches.

0.6 x JHEP text width (poster_style.SINGLE_IN), included at natural size;
greyscale-safe, mathtext only.  Output:
backreaction/figures/matching_diagram.{png,pdf}.

Run:  uv run backreaction/numerics/poster_matching.py    (~2 min)
"""

import multiprocessing as mp_
import time

import mpmath as mp
import numpy as np
from scipy.optimize import brentq, curve_fit

from backreaction import paths
from backreaction.numerics import bc_alpha as bca
from backreaction.numerics import dk_background as dk
from backreaction.numerics import dk_throat as dt
from backreaction.numerics import dk_throat_matching as dtm
from backreaction.numerics import poster_style

BETA = 1e8
R0, H = (
    1.0 / 3.0,
    2.0 / 3.0,
)  # BTZ centre and horizon in the fixed-T frame (paper app. C.4)
RUNGS = (1.0, 20.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 3e7, 1e8)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# the T = 0 outer solution (child process)
# ---------------------------------------------------------------------------
def t0_outer(nu):
    """rho w of the c_0 = 0 mode on the T = 0 brane at alpha_*, in the brane's
    own units, plus rho_* of that brane.  Runs `dk_throat_matching` unchanged."""
    sol, _, _ = dtm.t0_brane()
    v0 = np.sqrt(dtm.ALPHA / 6)
    rho_star = brentq(lambda r: np.exp(2 * sol.sol(r)[1]) - 2 * v0, 1e-6, 1e6)
    mode = dtm.outer_mode(sol, 1 + nu**2)
    rho = np.logspace(-8, 7, 6000)
    return rho, rho * mode.sol(rho)[0], rho_star


# ---------------------------------------------------------------------------
# the rung
# ---------------------------------------------------------------------------
def solve_rung():
    """Continue the background up the coarse ladder and solve the onset."""
    bg = None
    for beta_, N in dt.LADDER:
        if beta_ not in RUNGS:
            continue
        bg = dk.solve_colloc(
            beta_, N=N, tol=1e-10, maxit=200, guess=dt._regrid(bg, N) if bg else None
        )
        log(f"    beta = {beta_:8.3g}  N = {N}  residual {bg.ode_residual():.1e}")
    assert bg.beta == BETA
    vals, v0 = bca.bc_spectral_grid(bg, n_keep=2)
    Bc = float(vals[0])
    fh = bg.fields_r(np.array([1.0 + 1e-9]))
    b_h = Bc * dt.L3SQ * np.exp(-2 * fh["V"][0])
    return bg, Bc, float(b_h), v0


def rung_rho_star(bg):
    """rho_* on the rung: e^{2V} has doubled from its throat value sqrt(beta/6)
    (the paper sec. 4.6 definition, same code path as `dk_throat_matching` [4])."""
    beta_ = bg.beta
    r = np.exp(np.linspace(1e-3, np.log(60 * (beta_ / 6) ** 0.25), 6000))
    e2V = np.exp(2 * bg.fields_r(r)["V"])
    return float(np.interp(2 * np.sqrt(beta_ / 6), e2V, r)) - R0


def horizon_exact(nu, rho):
    """(rho/h) Re y_H, the BTZ solution of paper app. C.4 that is regular at the
    horizon; ~ cos(nu ln(rho/h) - chi) for rho >> h."""
    a = (1 + 1j * nu) / 2
    z = (H / rho) ** 2
    y = [complex(mp.hyp2f1(a, a, 1, 1 - zz) * mp.power(zz, a)) for zz in z]
    return (rho / H) * np.real(y)


def fit_amplitude(x, y, mask):
    """Least-squares scale factor c with y ~ c x on the mask."""
    return float(np.dot(x[mask], y[mask]) / np.dot(x[mask], x[mask]))


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def make_figure(xi, y_num, xi_H, y_H, xi_T, y_T, marks, labels, path_stem):
    plt = poster_style.pyplot()
    ink = "black"
    xi_h3, xi_star, xi_c = marks["h3"], marks["star"], marks["c"]
    xmax = 8.0
    fig, ax = plt.subplots(figsize=(poster_style.SINGLE_IN, 2.75))

    # --- the three regions ------------------------------------------------
    ax.axvspan(0.0, xi_h3, color="0.80", lw=0, zorder=0)
    ax.axvspan(xi_h3, xi_star, color="0.92", lw=0, zorder=0)
    ax.axvspan(xi_star, xmax, color="0.97", lw=0, zorder=0)
    ytop = 1.55
    for x0, x1, txt in [
        (0.0, xi_h3, "BTZ\nhorizon"),
        (xi_h3, xi_star, "throat"),
        (xi_star, xmax, "outer AdS$_5$"),
    ]:
        ax.text(
            0.5 * (x0 + x1),
            1.36,
            txt,
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.25",
            style="italic",
            linespacing=0.95,
        )

    # --- curves -----------------------------------------------------------
    ax.plot(xi, y_num, "-", color=ink, lw=1.3, zorder=3, label="full mode")
    ax.plot(
        xi_H,
        y_H,
        "--",
        color="0.45",
        lw=0.9,
        dashes=(4, 2.5),
        zorder=4,
        label="BTZ horizon, exact",
    )
    ax.plot(
        xi_T,
        y_T,
        ":",
        color="0.45",
        lw=1.3,
        zorder=5,
        label=r"$T=0$ outer mode",
    )

    # --- the throat phase and the UV phase -------------------------------
    ax.annotate(
        r"$\cos(\nu\ln(\rho/h)-\chi)$",
        xy=(
            0.55 * (xi_h3 + xi_star),
            float(np.interp(0.55 * (xi_h3 + xi_star), xi, y_num)),
        ),
        xytext=(0.5 * (xi_h3 + xi_star) - 0.15, 0.56),
        ha="center",
        va="center",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="-", lw=0.6, color=ink, shrinkA=0, shrinkB=2),
    )
    ax.annotate(
        r"$\cos(\nu\ln(\rho/\rho_\star)+\phi_\star)$",
        xy=(xi_star + 1.0, float(np.interp(xi_star + 1.0, xi, y_num))),
        xytext=(xi_star + 2.2, 1.08),
        ha="center",
        va="center",
        fontsize=7.5,
        arrowprops=dict(arrowstyle="-", lw=0.6, color=ink, shrinkA=0, shrinkB=2),
    )
    ax.text(
        0.5 * (xi_h3 + xi_star) - 0.15,
        0.415,
        labels["phase_run"],
        ha="center",
        va="center",
        fontsize=7,
        color="0.25",
    )

    # --- the marked radii on a top axis ------------------------------------
    for x in (xi_star, xi_c):
        ax.axvline(x, color=ink, lw=0.5, ls="-", alpha=0.35, zorder=1)
    top = ax.secondary_xaxis("top")
    top.set_xticks([0.0, xi_star])
    top.set_xticklabels([r"$h$", r"$\rho_\star$"], fontsize=8)
    top.tick_params(direction="in", length=2.5, width=0.6, pad=2)
    # rho_* and r_c sit close together: r_c gets its own top axis whose
    # outward tick is a leader line lifting its label clear of rho_*'s
    top_c = ax.secondary_xaxis("top")
    top_c.set_xticks([xi_c])
    top_c.set_xticklabels([r"$r_c$"], fontsize=8)
    top_c.tick_params(direction="out", length=9.0, width=0.5, pad=1)
    top_c.spines["top"].set_visible(False)

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(-0.12, ytop)
    ax.set_xlabel(r"$\ln(\rho/h)$")
    ax.set_ylabel(r"$\rho\,w$  (normalised)")
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(direction="in", length=2.5, right=True)
    # lower left is the one corner the curves leave empty; the white frame
    # keeps the grey handles readable over the dark BTZ band
    leg = ax.legend(
        loc="lower left",
        fontsize=7,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        handlelength=2.4,
        borderaxespad=0.3,
        borderpad=0.3,
    )
    leg.get_frame().set(facecolor="white", edgecolor="0.6", linewidth=0.4)
    ax.text(
        0.985,
        0.79,
        labels["rung"],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.25",
    )

    fig.tight_layout(pad=0.3)
    poster_style.save(fig, path_stem)


def main():
    cache = np.load(paths.data("dk_throat.npz"))
    i = int(np.argmin(np.abs(cache["beta"] - BETA)))
    nu_cache = float(np.sqrt(cache["b_h"][i] - 1))
    match = np.load(paths.data("dk_throat_matching.npz"))
    L_cache = float(match["L"][int(np.argmin(np.abs(match["beta"] - BETA)))])

    log(f"T = 0 brane at alpha_* (child process, nu = {nu_cache:.5f} from the cache)")
    ctx = mp_.get_context("spawn")
    with ctx.Pool(1) as pool:
        job = pool.apply_async(t0_outer, (nu_cache,))
        log(f"ladder to beta = {BETA:g} (this process)")
        bg, Bc, b_h, v0 = solve_rung()
        rho_t0, y_t0, rho_star_t0 = job.get()
    nu = float(np.sqrt(b_h - 1))
    assert abs(nu - nu_cache) < 1e-5, (nu, nu_cache)

    rho_star = rung_rho_star(bg)
    L = np.log(rho_star / H)
    assert abs(L - L_cache) < 1e-4, (L, L_cache)
    r_c = (BETA / 6) ** 0.25
    chi = dtm.chi(nu)
    phi_star = float(np.interp(nu, match["nu_tab"], match["phi_tab"]))
    log(f"  ratio rho_*(rung)/rho_*(T=0 brane) = {rho_star / rho_star_t0:.6e}")

    # the full numerical mode, rho w against xi = ln(rho/h)
    xi = np.linspace(0.0, 8.0, 1600)
    rho = H * np.exp(xi)
    w = bg._bary(v0, 1.0 / (rho + R0))
    y_num = rho * w

    # throat fit: rho w = A cos(nu xi - chi_fit) on h e^1.5 < rho < rho_* e^-1
    thr = (xi > 1.5) & (xi < L - 1.0)

    def f(x, A, ph):
        return A * np.cos(nu * x - ph)

    (A_thr, chi_fit), _ = curve_fit(f, xi[thr], y_num[thr], p0=[y_num[thr][0], chi])
    if A_thr < 0:
        A_thr, chi_fit = -A_thr, chi_fit + np.pi
    chi_fit = (chi_fit + np.pi) % (2 * np.pi) - np.pi
    fit_err = float(np.max(np.abs(f(xi[thr], A_thr, chi_fit) - y_num[thr])) / A_thr)
    y_num = y_num / A_thr

    # exact BTZ horizon solution, amplitude fixed at the horizon end
    xi_H = xi[xi <= 6.0]
    y_H = horizon_exact(nu, H * np.exp(xi_H))
    y_H *= fit_amplitude(y_H, y_num[xi <= 6.0], xi_H < 1.0)

    # T = 0 outer mode, rescaled onto the rung by rho_*, amplitude fixed outside
    xi_T = np.log(rho_t0 * (rho_star / rho_star_t0) / H)
    keep = (xi_T > 1.2) & (xi_T <= 8.0)
    xi_T, y_T = xi_T[keep], y_t0[keep]
    y_T *= fit_amplitude(y_T, np.interp(xi_T, xi, y_num), xi_T > L)

    dev_H = float(np.max(np.abs(y_H - y_num[xi <= 6.0])[xi_H < L - 1.0]))
    dev_T = float(np.max(np.abs(y_T - np.interp(xi_T, xi, y_num))[xi_T > 2.0]))

    print(f"matching diagram at beta = {BETA:g} (alpha = {BETA / Bc**2:.4f})")
    print(f"  B_c = {Bc:.4f}   b_h = {b_h:.6f}   nu = {nu:.5f}")
    print(f"  h = {H:.4f}   rho_* = {rho_star:.4f}   r_c = (beta/6)^(1/4) = {r_c:.4f}")
    print(f"  ln(rho_*/h) = {L:.4f}   ln(r_c/h) = {np.log(r_c / H):.4f}")
    print(
        f"  chi(nu) = {chi:.4f}   phi_*(nu) = {phi_star:.4f}   chi + phi_* = {chi + phi_star:.4f}"
    )
    print(f"  nu ln(rho_*/h) = {nu * L:.4f}  (< pi: less than one half-oscillation)")
    print(
        f"  throat fit of the full mode: chi_fit = {chi_fit:.4f} (exact {chi:.4f}), residual {fit_err:.1e}"
    )
    print(
        f"  overlay deviations: horizon {dev_H:.1e} (throat), T=0 outer {dev_T:.1e} (rho > 7h)"
    )

    paths.FIGURES.mkdir(exist_ok=True)
    make_figure(
        xi,
        y_num,
        xi_H,
        y_H,
        xi_T,
        y_T,
        dict(h3=np.log(3.0), star=L, c=np.log(r_c / H)),
        dict(
            phase_run=rf"$\nu\ln(\rho_\star/h)={nu * L:.2f}<\pi$",
            rung=rf"$\beta=10^{{8}},\ \nu={nu:.3f}$",
        ),
        str(paths.figure("matching_diagram")),
    )


if __name__ == "__main__":
    main()
