"""Uncertainty on the coupling alpha = 0.3506 at which K_0 changes sign along
the ladder (alpha_K0 in the code), from the cache.

This coupling (paper sec. 5.3) is the zero of K_0(beta)/C_4 along the throat ladder, where the
long-range part of the vortex interaction turns attractive.  It is not an
instability threshold: the lattice destabilises at K_0/2 + C_tri = 0,
alpha_L = 0.3721800 (numerics/dk_uniform.py, numerics/dk_bogoliubov.py).

K_0 = K_{G=0} = lim_{s->0} K(s; beta) is computed on every rung by collocation
of the G = 0 system, and cached beside two other determinations
(dk_long_wavelength.py): shooting of the same system, and the cubic s -> 0
extrapolation of the G != 0 kernel from gamma = 1e-3..8e-3.  The zero itself
is a secant with fresh solves, converged to |K_0/C_4| < 1e-10.  The
uncertainty is the shift of the zero under the change of method, i.e. the
method spread near the zero divided by the slope d(K_0/C_4)/d alpha.
A quadratic extrapolation from gamma >= 0.02 (K0_stencil) is shown for
comparison: it moves the zero by ~1e-6.  Reads backreaction/data/dk_long_wavelength.npz only.

Run:  uv run python -m backreaction.numerics.k0_zero_uncertainty
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from backreaction.paths import DATA


def main():
    d = np.load(DATA / "dk_long_wavelength.npz")
    lb, alpha, C4 = np.log(d["beta"]), d["alpha"], d["C4"]
    a_of_lb = PchipInterpolator(lb, alpha)

    def zero(ratio):
        f = PchipInterpolator(lb, ratio)
        x = brentq(f, lb[0], lb[-1])
        return x, float(a_of_lb(x)), f

    x1, a1, f1 = zero(d["K0"] / C4)
    i = int(np.argmin(abs(lb - x1)))
    win = slice(i - 1, i + 2)
    slope = float(f1.derivative()(x1) / a_of_lb.derivative()(x1))
    a_secant = float(d["alpha_K0zero"])
    spread = {
        "K_G0 collocation vs shooting": float(
            np.abs((d["K0_shoot"] - d["K0"]) / C4)[win].max()
        ),
        "K_G0 vs s->0 limit of the G != 0 kernel": float(
            np.abs((d["K0_ext"] - d["K0"]) / C4)[win].max()
        ),
    }
    budget = {k: v / abs(slope) for k, v in spread.items()}
    budget["secant residual (< 1e-10 C4)"] = 1e-10 / abs(slope)
    old_shift = abs(zero(d["K0_stencil"] / C4)[1] - a1)

    print(
        f"alpha_K0 from the secant with fresh solves (dk_long_wavelength.py, cached): {a_secant:.8f}"
    )
    print(
        f"alpha_K0 from PCHIP in ln beta on the cached rungs:                   {a1:.6f}  (interpolation only)"
    )
    print(
        f"slope d(K_0/C_4)/d alpha at the zero:                               {slope:+.4f}"
    )
    for k, v in budget.items():
        print(f"  delta alpha_K0 from {k:<42s} {v:.1e}")
    unc = float(np.sqrt(sum(v * v for v in budget.values())))
    print(
        f"  (a quadratic stencil from gamma >= 0.02 would move the cached-rung zero by {old_shift:.1e})"
    )
    print(
        f"RESULT  alpha_K0 = {a_secant:.7f} +/- {unc:.0e}; beta_K0 = {float(d['beta_K0zero']):.2f}, "
        f"B_c u_H^2 = {float(d['Bc_K0zero']):.3f}"
    )
    ok = unc < 1e-7 and abs(a1 - a_secant) < 5e-5 and abs(a_secant - 0.3505596) < 1e-7
    print(
        ("PASS" if ok else "FAIL"),
        "[1] alpha_K0 = 0.3505596 with uncertainty below 1e-7; cached-rung interpolation within 5e-5",
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
