"""Independent check of B_c via Chebyshev spectral collocation.

Substituting w(u) = u^2 q(u) (which removes the source mode and builds in the
normalisable boundary behaviour) into

    f w'' + (f' - f/u) w' + B w = 0,   f = 1 - u^4,   u in [0, 1],

and dividing by u gives the everywhere-regular equation

    (u - u^5) q'' + (3 - 7u^4) q' - 8 u^3 q  +  B u q = 0.

Collocating at Chebyshev points on [0, 1] yields a generalised eigenvalue
problem  L q = -B M q  with M = diag(u).  Horizon regularity is imposed
automatically by collocating the (degenerate) equation at u = 1.
"""

import numpy as np
from scipy.linalg import eig


def cheb(N):
    """Chebyshev differentiation matrix on [-1, 1] (Trefethen)."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    xs = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(xs, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, xs


def critical_fields(N=80, n_keep=5):
    D, xs = cheb(N)
    u = (1.0 - xs) / 2.0        # map [-1,1] -> [0,1], u[0]=0 ... u[-1]=1
    D = -2.0 * D                # d/du
    D2 = D @ D

    L = (np.diag(u - u**5) @ D2
         + np.diag(3.0 - 7.0 * u**4) @ D
         - 8.0 * np.diag(u**3))
    M = np.diag(u)

    vals, _ = eig(L, -M)
    vals = vals[np.isfinite(vals)]
    vals = np.real(vals[np.abs(vals.imag) < 1e-8 * np.maximum(1, np.abs(vals.real))])
    vals = np.sort(vals[vals > 0])
    return vals[:n_keep]


# The first four condensation fields B_n u_H^2 (the paper quotes only B_0 =
# 5.13126764, section 3.1): the shooting of critical_field.py and this
# collocation must both reproduce them to 8 decimals.
B_TOWER = [5.13126764, 22.48156859, 51.20975989, 91.41056318]


if __name__ == "__main__":
    runs = {}
    for N in (40, 60, 80, 120):
        vals = critical_fields(N)
        runs[N] = vals
        print(f"N = {N:4d}:  B_n u_H^2 = " + "  ".join(f"{v:.8f}" for v in vals))
    failed = []
    for N, vals in runs.items():
        for n, ref in enumerate(B_TOWER):
            if abs(vals[n] - ref) > 5e-9:
                failed.append(f"N = {N}, n = {n}: {vals[n]:.10f} vs {ref}")
    spread = max(abs(runs[N][0] - runs[120][0]) for N in runs)
    print(f"tower reproduced to 5e-9 at every N; B_c spread over N = {spread:.1e}")
    if failed:
        print("FAILED: " + "; ".join(failed))
        raise SystemExit(1)
    print("ALL CHECKS PASSED")
