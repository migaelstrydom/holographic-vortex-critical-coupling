"""The O(alpha) shift of B_c from the magnetic-brane backreaction.

Uses the O(alpha B^2) G = 0 metric correction (g0_thermo.py, fixed-T
convention -- so the result is the shift of the dimensionless B_c/T^2) and
the perturbed zero-mode operator derived in derivations/bc_shift.py
(systems/bc_shift.py):

    delta B_c = alpha int_0^{u_H} [dP w0'^2 - dQ w0^2] du     (J2 = 1)

Methods:
  1. the perturbation integral on the channel-constructed profiles
     (gauge invariance of the integral is *proven symbolically* in
     derivations/bc_shift.py [4]);
  2. direct re-solve of the perturbed generalised eigenvalue problem at
     small alpha (two values, linear-in-alpha extrapolation of the slope).

The metric profiles are per unit alpha B^2, so H = B_c^2 x (unit profile)
and the result is reported as  delta B_c / B_c = k alpha B_c^2 u_H^4.

CHECKS (each can fail at the precision printed)
[1] the perturbation integral is converged in N: N = 240 vs 300 to 1e-9
    relative (measured 1e-10);
[2] the direct problem at alpha = 0 reproduces the probe B_c to 3e-7 (the
    clipped endpoints of the direct discretisation leave 1.6e-7 at N = 240;
    this offset cancels in the slope);
[3] the extrapolated direct slope agrees with the perturbation integral to
    1e-5 relative (measured 4.5e-6);
[4] the direct slope is stable under N = 200, 240, 300 to 1e-5 relative
    (measured spread 5e-6);
[5] a third, independent method: the nonlinear brane solves of bc_alpha.py
    (cached in bc_alpha.npz, full D'Hoker-Kraus background at alpha = 2e-4
    and 5e-4), Richardson-extrapolated, give the same reduced slope
    dB_c/B_c/(alpha B_c^2) to 1e-6, the tolerance bc_alpha.py itself uses
    (measured 2.3e-7).

Run from the project root:
    uv run python -m backreaction.numerics.bc_shift
"""

import numpy as np
from scipy.linalg import eig

from exchange_kernel import clencurt_weights
from backreaction import paths
from backreaction.numerics import g0_thermo as g0
from backreaction.systems import bc_shift as pert

BC = g0.BC
U_H = 1.0


def profiles(N=240):
    """Metric correction per unit alpha B^2 (source unit 'B2') on the
    Chebyshev grid, as (ht, hx, hz, hu) plus derivatives (algebraic gauge:
    ht = 0; gauge invariance of delta B_c is proven symbolically in
    derivations/bc_shift.py [4])."""
    Q = g0.solve_quad("B2", N=N)
    u, D = Q["u"], Q["D"]
    ht = np.zeros_like(u)
    hx, hz, hu = Q["Hxx"], Q["Hzz"], Q["Huu"]
    dH = (np.zeros_like(u), Q["V1"], Q["V2"], D @ hu)
    return u, D, (ht, hx, hz, hu), dH


def pt_integral(N=240):
    """delta B_c per alpha, with H per unit B^2 scaled by B_c^2."""
    u, D, H, dH = profiles(N)
    wcc = clencurt_weights(N) / 2.0
    w0 = g0.MAT.interp(g0.MAT.w0, u)
    w0p = g0.MAT.interp(g0.MAT.w0p, u)
    vals = np.zeros_like(u)
    for k, uk in enumerate(u):
        if uk < 1e-9 or uk > 1 - 1e-9:
            # integrand ~ u^3 at the boundary and finite at the horizon;
            # extrapolate the horizon endpoint
            uk2 = min(max(uk, 1e-5), 1 - 1e-5)
            args = [BC**2 * v[k] for v in H] + [BC**2 * v[k] for v in dH]
            dPv = pert.dP(uk2, BC, U_H, *args)
            dQv = pert.dQ(uk2, BC, U_H, *args)
            vals[k] = dPv * w0p[k] ** 2 - dQv * w0[k] ** 2
            continue
        args = [BC**2 * v[k] for v in H] + [BC**2 * v[k] for v in dH]
        dPv = pert.dP(uk, BC, U_H, *args)
        dQv = pert.dQ(uk, BC, U_H, *args)
        vals[k] = dPv * w0p[k] ** 2 - dQv * w0[k] ** 2
    return float(wcc @ vals)


def direct_eig(alpha, N=240):
    """Smallest positive eigenvalue B of the perturbed problem
    ((Pb + a dP) w')' + (Qb0 + a dQ0) w + B (1/u + a dQB) w = 0
    where dQ = B dQB + dQ0 is split numerically (linearity in B checked)."""
    u, D, H, dH = profiles(N)
    D2 = D @ D

    def coeffs(uk, k):
        args = [BC**2 * v[k] for v in H] + [BC**2 * v[k] for v in dH]
        dP1 = pert.dP(uk, 1.0, U_H, *args)
        dP2 = pert.dP(uk, 2.0, U_H, *args)
        dP3 = pert.dP(uk, 3.0, U_H, *args)
        dQ1 = pert.dQ(uk, 1.0, U_H, *args)
        dQ2 = pert.dQ(uk, 2.0, U_H, *args)
        dQ3 = pert.dQ(uk, 3.0, U_H, *args)
        qB = dQ2 - dQ1
        q0 = dQ1 - qB
        # linearity checks in the explicit B
        assert abs(dQ3 - (q0 + 3 * qB)) < 1e-10 * max(1.0, abs(dQ3))
        assert abs(dP2 - dP1) < 1e-10 * max(1.0, abs(dP1)) and abs(
            dP3 - dP1
        ) < 1e-10 * max(1.0, abs(dP1))
        return dP1, q0, qB

    n = len(u)
    # node-consistent coefficient arrays; dP' spectrally (the H(u)-chain
    # rule is then included automatically)
    dP_arr = np.zeros(n)
    q0_arr = np.zeros(n)
    qB_arr = np.zeros(n)
    for k, uk in enumerate(u):
        uk_e = min(max(uk, 1e-7), 1 - 1e-7)
        dP_arr[k], q0_arr[k], qB_arr[k] = coeffs(uk_e, k)
    ddP_arr = D @ dP_arr
    A = np.zeros((n, n))
    Nmat = np.zeros((n, n))
    for k, uk in enumerate(u):
        uk_e = min(max(uk, 1e-7), 1 - 1e-7)
        fk = 1.0 - uk_e**4
        fpk = -4.0 * uk_e**3
        P = fk / uk_e + alpha * dP_arr[k]
        Pp = (fpk / uk_e - fk / uk_e**2) + alpha * ddP_arr[k]
        # row: u * [ (P w')' + q0-part w + B (1/u + qB-part) w ] = 0
        reg = uk_e * fk  # regulator clears 1/u and horizon poles
        A[k, :] = reg * (P * D2[k] + Pp * D[k])
        A[k, k] += reg * alpha * q0_arr[k]
        Nmat[k, k] = -reg * (1.0 / uk_e + alpha * qB_arr[k])
    # Dirichlet at u = 0 (grid node 0)
    k0 = int(np.argmin(np.abs(u)))
    A[k0, :] = 0.0
    A[k0, k0] = 1.0
    Nmat[k0, :] = 0.0
    vals, vecs = eig(A, Nmat)
    sel = np.isfinite(vals) & (np.abs(vals.imag) < 1e-6) & (vals.real > 0.1)
    idx = np.where(sel)[0]
    j = idx[np.argmin(np.abs(vals[idx].real - BC))]
    return float(vals[j].real)


CHECKS = {}


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")
    )


def direct_slope(N=240, alphas=(1e-3, 3e-4)):
    """Linear-in-alpha extrapolation of (B(alpha) - B(0))/alpha to alpha -> 0."""
    B0 = direct_eig(0.0, N)
    slopes, Bs = [], []
    for al in alphas:
        Ba = direct_eig(al, N)
        Bs.append(Ba)
        slopes.append((Ba - B0) / al)
    a1_, a2_ = alphas
    k = slopes[1] - a2_ * (slopes[0] - slopes[1]) / (a1_ - a2_)
    return k, B0, Bs, slopes


def bc_alpha_richardson():
    """Reduced slope from the cached nonlinear brane solves of bc_alpha.py:
    2-point Richardson of s(alpha) = (B_c - B_c0)/B_c0/(alpha B_c0^2) from the
    two smallest alpha > 0."""
    d = np.load(paths.data("bc_alpha.npz"))
    a, b, b0 = d["small_alpha"], d["small_Bc"], float(d["B_c0"])
    m = a > 0
    a, b = a[m][:2], b[m][:2]
    sl = (b - b0) / b0 / (a * b0**2)
    return float((sl[0] * a[1] - sl[1] * a[0]) / (a[1] - a[0]))


def main():
    print(f"B_c (probe) = {BC:.8f}\n")
    k_alg = pt_integral()
    k_alg2 = pt_integral(N=300)
    print(
        f"PT integral (algebraic gauge): delta B_c per alpha = "
        f"{k_alg:.8f} (N=240) / {k_alg2:.8f} (N=300)"
    )
    print(
        "(gauge invariance of the integral is proven symbolically in "
        "derivations/bc_shift.py [4])"
    )
    dN = abs(k_alg - k_alg2) / abs(k_alg2)
    check(
        "[1] PT integral converged: N = 240 vs 300 to 1e-9", dN < 1e-9, f"rel {dN:.1e}"
    )
    print()

    k_direct, B0, Bs, slopes = direct_slope(240)
    print(
        f"direct eigenvalue at alpha = 0: {B0:.8f} "
        f"(vs B_c {BC:.8f}, diff {B0 - BC:.2e})"
    )
    check("[2] direct problem at alpha = 0 reproduces B_c to 3e-7", abs(B0 - BC) < 3e-7)
    for al, Ba, sl in zip((1e-3, 3e-4), Bs, slopes, strict=True):
        print(f"direct eigenvalue at alpha = {al:g}: {Ba:.10f}  slope {sl:.8f}")
    print(f"extrapolated direct slope (alpha -> 0): {k_direct:.6f}")
    dev = abs(k_direct - k_alg2) / abs(k_alg2)
    print(
        f"\nsummary: delta B_c = k alpha with k = {k_alg2:.6f} (PT)"
        f" vs {k_direct:.6f} (direct, extrapolated)"
    )
    print(f"  relative deviation {dev:.2e}")
    check("[3] direct slope = PT integral to 1e-5 (relative)", dev < 1e-5, f"{dev:.1e}")

    ks = [direct_slope(N)[0] for N in (200, 300)] + [k_direct]
    spread = (max(ks) - min(ks)) / abs(k_alg2)
    print("direct slope at N = 200, 300, 240: " + ", ".join(f"{k:.6f}" for k in ks))
    check(
        "[4] direct slope stable under N = 200, 240, 300 to 1e-5",
        spread < 1e-5,
        f"spread {spread:.1e}",
    )

    red = k_alg2 / BC / BC**2
    print(f"  delta B_c / B_c = {k_alg2 / BC:.6f} alpha = {red:.8f} alpha B_c^2 u_H^4")
    rich = bc_alpha_richardson()
    print(f"nonlinear brane solves (bc_alpha.npz), Richardson: {rich:.8f} alpha B_c^2")
    check(
        "[5] PT reduced slope = nonlinear brane Richardson slope to 1e-6",
        abs(rich - red) < 1e-6,
        f"diff {abs(rich - red):.1e}",
    )

    nfail = sum(1 for v in CHECKS.values() if not v)
    print(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
