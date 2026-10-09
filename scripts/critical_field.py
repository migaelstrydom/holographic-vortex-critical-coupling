"""Numerical solution of the linearised radial equation and the critical field B_c.

Equation (u_H = 1, so B is measured in units of 1/u_H^2):

    f w'' + (f' - f/u) w' + B w = 0,        f(u) = 1 - u^4,

on u in (0, 1), with
  * regularity at the horizon u = 1 (Frobenius series start),
  * vanishing source at the boundary u -> 0, i.e. w ~ c2 u^2.

Near the boundary the two behaviours are
    w = c0 [1 - (B/2) u^2 ln u + ...] + c2 u^2 + ...
and the combination  c0 = w - (u/2) w'  is log-free up to O(u^2) corrections,
so we use it as the shooting target.  The eigenvalues B_n are the zeros of
c0(B); the smallest one is the critical field B_c.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import script_paths

U_EPS = 1e-4    # boundary cutoff where c0 is read off
DELTA = 1e-6    # distance from horizon where integration starts


def rhs(u, Y, B):
    w, wp = Y
    f = 1.0 - u**4
    fp = -4.0 * u**3
    wpp = -((fp - f / u) * wp + B * w) / f
    return [wp, wpp]


def horizon_start(B, delta=DELTA):
    """Regular solution near u = 1:  w = 1 + a1 v + a2 v^2,  v = 1 - u."""
    a1 = -B / 4.0
    a2 = -B * (8.0 - B) / 64.0
    v = delta
    w = 1.0 + a1 * v + a2 * v**2
    dw_du = -(a1 + 2.0 * a2 * v)        # d/du = -d/dv
    return [w, dw_du]


def source_coefficient(B):
    """Integrate from the horizon to the boundary; return c0 = w - (u/2) w'."""
    sol = solve_ivp(rhs, [1.0 - DELTA, U_EPS], horizon_start(B), args=(B,),
                    method="DOP853", rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    w, wp = sol.y[:, -1]
    return w - 0.5 * U_EPS * wp


def find_eigenvalues(b_max=100.0, n_scan=4000):
    """Scan c0(B) for sign changes and refine each with Brent's method."""
    bs = np.linspace(0.05, b_max, n_scan)
    c0s = np.array([source_coefficient(b) for b in bs])
    eigs = []
    for i in range(len(bs) - 1):
        if c0s[i] * c0s[i + 1] < 0:
            eigs.append(brentq(source_coefficient, bs[i], bs[i + 1],
                               xtol=1e-12, rtol=1e-12))
    return eigs, bs, c0s


def eigenfunction(B, n_pts=2000):
    us = np.linspace(1.0 - DELTA, U_EPS, n_pts)
    sol = solve_ivp(rhs, [us[0], us[-1]], horizon_start(B), args=(B,),
                    method="DOP853", rtol=1e-12, atol=1e-14, t_eval=us)
    return sol.t[::-1], sol.y[0][::-1]


def main():
    eigs, bs, c0s = find_eigenvalues()
    print("Eigenvalues B_n u_H^2 (zeros of the source coefficient):")
    for n, b in enumerate(eigs):
        print(f"  n = {n}:  B u_H^2 = {b:.8f}")
    Bc = eigs[0]
    print(f"\nCritical field:  B_c u_H^2 = {Bc:.8f}")
    print(f"With T = 1/(pi u_H):  B_c = {Bc:.6f} (pi T)^2 = {Bc * np.pi**2:.4f} T^2")

    # convergence check in the cutoffs
    global U_EPS, DELTA
    cutoff_bc = []
    for ueps, delta in ((1e-3, 1e-5), (1e-4, 1e-6), (1e-5, 1e-7)):
        U_EPS, DELTA = ueps, delta
        b = brentq(source_coefficient, 0.9 * Bc, 1.1 * Bc, xtol=1e-12)
        print(f"  cutoff check: u_eps={ueps:g}, delta={delta:g}  ->  B_c = {b:.8f}")
        cutoff_bc.append(b)
    U_EPS, DELTA = 1e-4, 1e-6
    cutoff_bc.append(Bc)

    # The first four condensation fields B_n u_H^2 (the paper quotes B_0 =
    # 5.13126764, section 3.1); the collocation of spectral_check.py
    # reproduces all four.
    tower = [5.13126764, 22.48156859, 51.20975989, 91.41056318]
    failed = [f"n = {n}: {eigs[n]:.10f} vs {ref}"
              for n, ref in enumerate(tower) if abs(eigs[n] - ref) > 5e-9]
    if max(cutoff_bc) - min(cutoff_bc) > 1e-8:
        failed.append(f"cutoff dependence {max(cutoff_bc) - min(cutoff_bc):.1e}")
    if failed:
        print("FAILED: " + "; ".join(failed))
        raise SystemExit(1)
    print("tower and cutoff independence checked: ALL CHECKS PASSED")

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(bs, c0s, lw=1.2)
    axes[0].axhline(0, color="k", lw=0.6)
    for b in eigs:
        axes[0].axvline(b, color="crimson", ls=":", lw=0.9)
    axes[0].set_xlabel(r"$B\,u_H^2$")
    axes[0].set_ylabel(r"source coefficient $c_0(B)$")
    axes[0].set_title("Shooting target: zeros = normalisable modes")

    for n, b in enumerate(eigs[:3]):
        us, ws = eigenfunction(b)
        axes[1].plot(us, ws / np.max(np.abs(ws)), lw=1.4,
                     label=rf"$n={n}$, $B u_H^2 = {b:.4f}$")
    axes[1].set_xlabel(r"$u/u_H$")
    axes[1].set_ylabel(r"$w(u)$ (normalised)")
    axes[1].set_title("Radial zero-mode profiles")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(script_paths.figure("critical_field.png"), dpi=160)
    print("\nFigure written to figures/critical_field.png")


if __name__ == "__main__":
    main()
