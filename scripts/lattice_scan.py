"""Free energy over the moduli space of vortex lattices; the triangular minimum.

Near the transition the free energy per unit boundary volume is

    F(rho, tau) = -J2 (B - B_c) rho^2 + C(tau) rho^4,
    C(tau) = (1/2) [ C4 + sum_{G in Lambda*(tau), G != 0}
                          e^{-G^2/(2 B_c)} ( C4 - X(|G|^2) ) ],

with rho^2 = <|psi|^2>, lattice cell area 2 pi / B_c (one flux quantum), and
the kernel C4, X from exchange_kernel.py.  Minimising over rho:

    F_min(tau) = - J2^2 (B - B_c)^2 / (4 C(tau)),

so the preferred lattice MINIMISES C(tau).  This script scans the fundamental
domain of the modular parameter tau, refines the minimum, and produces the
figures.  Expected: unique minimum at tau = 1/2 + i sqrt(3)/2 (triangular).
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import script_paths
from lll_identity import make_lattice, reciprocal, psi_on_cell

dat = np.load(script_paths.data("kernel.npz"))
G2grid, X, C4, J2, B_c = dat["G2grid"], dat["X"], float(dat["C4"]), \
    float(dat["J2"]), float(dat["B_c"])
Xs = CubicSpline(G2grid, X)
G2MAX = G2grid[-1]


def quartic_coefficient(tau_re, tau_im, nmax=14):
    """C(tau): half the lattice sum of e^{-G^2/2B}(C4 - X)."""
    A1, A2, _, _ = make_lattice(tau_re + 1j * tau_im)
    b1, b2 = reciprocal(A1, A2)
    m, n = np.meshgrid(np.arange(-nmax, nmax + 1), np.arange(-nmax, nmax + 1),
                       indexing="ij")
    Gx = m * b1[0] + n * b2[0]
    Gy = m * b1[1] + n * b2[1]
    G2 = Gx**2 + Gy**2
    # harmonics with G^2 >= G2MAX are dropped: their weight e^{-G^2/2B_c}
    # is < e^{-30}, far below double precision
    mask = (G2 > 1e-12) & (G2 < G2MAX)
    ker = C4 - Xs(G2[mask])
    return 0.5 * (C4 + np.sum(np.exp(-G2[mask] / (2 * B_c)) * ker))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- special points -----------------------------------------------------
    C_hex = quartic_coefficient(0.5, np.sqrt(3) / 2)
    C_sq = quartic_coefficient(0.0, 1.0)
    print(f"C(triangular) = {C_hex:.8f}")
    print(f"C(square)     = {C_sq:.8f}")
    print(f"relative quartic-coefficient difference (sq - tri)/tri = "
          f"{(C_sq - C_hex) / C_hex:.4%}")
    print(f"free-energy gain of triangular over square: "
          f"F_tri/F_sq = C_sq/C_hex = {C_sq / C_hex:.6f}")

    # 'beta_eff': effective Abrikosov ratio of the holographic functional
    print(f"2 C(tau)/C4: triangular {2 * C_hex / C4:.6f}, "
          f"square {2 * C_sq / C4:.6f}  "
          f"(pure-contact values would be 1.159595, 1.180341)")

    # the paper (appendix E.4) drops the tau-independent G = 0 term: C - C4/2
    Ct, Cs = C_hex - C4 / 2, C_sq - C4 / 2
    print(f"G != 0 part: C(triangular) = {Ct:.7f}, C(square) = {Cs:.7f}, "
          f"(sq - tri)/tri = {(Cs - Ct) / Ct:+.4f}")

    # --- scan of the fundamental domain --------------------------------------
    re_grid = np.linspace(0.0, 0.5, 81)
    im_grid = np.linspace(0.75, 1.35, 97)
    Cgrid = np.array([[quartic_coefficient(r, i) for r in re_grid]
                      for i in im_grid])

    opt = minimize(lambda p: quartic_coefficient(p[0], p[1]),
                   x0=[0.45, 0.9], method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-14})
    print(f"\nminimum of C(tau) at tau = {opt.x[0]:.8f} + {opt.x[1]:.8f} i")
    print(f"triangular point:        tau = {0.5:.8f} + {np.sqrt(3)/2:.8f} i")
    dev = np.hypot(opt.x[0] - 0.5, opt.x[1] - np.sqrt(3) / 2)
    print(f"|tau_min - tau_triangular| = {dev:.2e}")

    # grid minimum over the fundamental domain |tau| >= 1, 0 <= Re tau <= 1/2
    R, I = np.meshgrid(re_grid, im_grid)
    inside = np.hypot(R, I) >= 1.0
    k = np.argmin(np.where(inside, Cgrid, np.inf))
    tau_grid = complex(R.flat[k], I.flat[k])
    step = np.hypot(re_grid[1] - re_grid[0], im_grid[1] - im_grid[0])
    print(f"grid minimum at tau = {tau_grid.real:.4f} + {tau_grid.imag:.4f} i")

    # Checks.  The G != 0 values and the margin are those of the table in
    # appendix E.4 of the paper; the Nelder-Mead start
    # (0.45 + 0.9i) is not the triangular point, so the minimiser check is real.
    checks = {
        "C_tri (G != 0) = 0.0184443": abs(Ct - 0.0184443) < 5e-8,
        "C_sq (G != 0) = 0.0229818": abs(Cs - 0.0229818) < 5e-8,
        "margin (C_sq - C_tri)/C_tri = +0.2460": abs((Cs - Ct) / Ct - 0.2460) < 5e-5,
        "C(tri) = 0.8087409, C(sq) = 0.8132783":
            abs(C_hex - 0.8087409) < 5e-8 and abs(C_sq - 0.8132783) < 5e-8,
        "Nelder-Mead minimiser at the triangular point": dev < 1e-6,
        "grid minimum within one cell of the triangular point":
            abs(tau_grid - complex(0.5, np.sqrt(3) / 2)) <= step,
    }
    failed = [key for key, v in checks.items() if not v]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
        raise SystemExit(1)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    levels = np.linspace(Cgrid.min(), Cgrid.max(), 28)
    cs = ax.contourf(re_grid, im_grid, Cgrid, levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax, label=r"quartic coefficient $C(\tau)$")
    th = np.linspace(np.pi / 3, np.pi / 2, 50)
    ax.plot(np.cos(th), np.sin(th), "w--", lw=0.8)   # |tau| = 1 boundary
    ax.plot([0.5, 0.5], [np.sqrt(3) / 2, im_grid[-1]], "w--", lw=0.8)
    ax.plot(0.5, np.sqrt(3) / 2, "r*", ms=16, label=r"triangular $\tau=e^{i\pi/3}$")
    ax.plot(0.0, 1.0, "ws", ms=8, label=r"square $\tau=i$")
    ax.plot(opt.x[0], opt.x[1], "c+", ms=14, mew=2, label="numerical minimum")
    ax.set_xlabel(r"$\mathrm{Re}\,\tau$")
    ax.set_ylabel(r"$\mathrm{Im}\,\tau$")
    ax.set_title("Quartic coefficient over lattice moduli space")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(script_paths.figure("lattice_scan.png"), dpi=160)

    # --- kernel figure --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(G2grid / B_c, X / C4, lw=1.4, label=r"$X(G)/C_4$")
    ax[0].plot(G2grid / B_c, 1 - X / C4, lw=1.4, label=r"$1 - X(G)/C_4$")
    ax[0].set_xlabel(r"$G^2/B_c$")
    ax[0].set_title("Exchange kernel (poles at the $B_n$ tower)")
    ax[0].legend()
    h = np.exp(-G2grid / (2 * B_c)) * (C4 - X)
    ax[1].plot(G2grid / B_c, h / C4, lw=1.4)
    G2_1 = 4 * np.pi * B_c / np.sqrt(3)
    ax[1].axvline(G2_1 / B_c, color="crimson", ls=":", lw=1.0,
                  label="first shell, triangular")
    ax[1].axvline(4 * np.pi, color="grey", ls=":", lw=1.0,
                  label="first shell, square")
    ax[1].set_xlabel(r"$G^2/B_c$")
    ax[1].set_title(r"summand $e^{-G^2/2B_c}\,(C_4 - X)/C_4$")
    ax[1].set_xlim(0, 30)
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(script_paths.figure("exchange_kernel.png"), dpi=160)

    # --- condensate amplitude and free energy vs B ----------------------------
    Bs = np.linspace(B_c, 1.2 * B_c, 200)
    rho2 = J2 * (Bs - B_c) / (2 * C_hex)
    Fmin = -J2**2 * (Bs - B_c)**2 / (4 * C_hex)
    Fsq = -J2**2 * (Bs - B_c)**2 / (4 * C_sq)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(Bs / B_c, rho2, lw=1.4)
    ax[0].set_xlabel(r"$B/B_c$")
    ax[0].set_ylabel(r"$\rho^2 = \langle|\psi|^2\rangle$")
    ax[0].set_title("Condensate turns on linearly (second order)")
    ax[1].plot(Bs / B_c, Fmin, lw=1.4, label="triangular")
    ax[1].plot(Bs / B_c, Fsq, lw=1.2, ls="--", label="square")
    ax[1].set_xlabel(r"$B/B_c$")
    ax[1].set_ylabel(r"$\Delta F$ per unit volume")
    ax[1].set_title("Condensation energy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(script_paths.figure("condensate.png"), dpi=160)

    # --- vortex-lattice visualisation ----------------------------------------
    tau = 0.5 + 1j * np.sqrt(3) / 2
    A1, A2, k, phases = make_lattice(tau)
    b1, b2 = reciprocal(A1, A2)
    # density on the cell + Fourier coefficients with phases
    psi = psi_on_cell(A1, A2, k, phases, Ngrid=96)
    dens = np.abs(psi)**2
    lam = np.fft.fft2(dens) / dens.size / dens.mean()
    bprof = CubicSpline(G2grid, dat["b_horizon"])
    w0H = float(dat["w0"][-1])

    # cartesian window, a few cells wide
    Lwin = 2.6 * np.linalg.norm(A1)
    xs = np.linspace(-Lwin / 2, Lwin / 2, 260)
    Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
    psiw = np.zeros_like(Xg, dtype=complex)
    for n in range(-40, 41):
        psiw += phases(n) * np.exp(1j * n * k * Yg - B_c * (Xg + n * k / B_c)**2 / 2)
    densw = np.abs(psiw)**2 / dens.mean()

    nmax = 10
    deltaF = np.zeros_like(Xg, dtype=complex)
    for m in range(-nmax, nmax + 1):
        for n in range(-nmax, nmax + 1):
            G = m * b1 + n * b2
            G2 = G @ G
            if G2 < 1e-12 or G2 > G2MAX:
                continue
            lmn = lam[m % lam.shape[0], n % lam.shape[1]]
            deltaF += lmn * bprof(G2) * np.exp(1j * (G[0] * Xg + G[1] * Yg))
    deltaF = deltaF.real - w0H**2 * densw

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    im0 = ax[0].pcolormesh(Xg, Yg, densw, cmap="magma", shading="auto")
    fig.colorbar(im0, ax=ax[0], label=r"$|\psi|^2/\langle|\psi|^2\rangle$")
    ax[0].set_title("Condensate density (triangular lattice)")
    im1 = ax[1].pcolormesh(Xg, Yg, deltaF, cmap="RdBu_r", shading="auto")
    fig.colorbar(im1, ax=ax[1],
                 label=r"$\delta F^3_{xy}(u_H)/\rho^2$ (arb. units)")
    ax[1].set_title("Induced field at the horizon: flux at vortex cores")
    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("x")
        a.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(script_paths.figure("vortex_lattice.png"), dpi=160)
    print("\nFigures written: lattice_scan.png, exchange_kernel.png, "
          "condensate.png, vortex_lattice.png")


if __name__ == "__main__":
    main()
