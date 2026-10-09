"""Quartic coefficients of the Landau expansion: contact term and exchange kernel.

Quantities (u_H = 1, w0 = zero mode at B_c normalised by J2 = int w0^2/u du = 1):

    C4   = int_0^1 du  w0(u)^4 / u                     (contact)
    X(G) = int_0^1 du  b_G(u) w0(u)^2 / u              (exchange)

where the induced-field profile b_G solves the BVP

    -( f b_G' / u )' + (G^2/u) b_G = (G^2/u) w0^2,
    b_G(0) = 0  (external field held fixed),  b_G regular at u = 1.

In operator form, with T = -u (f (.)'/u)' self-adjoint and positive in the
inner product <p,q> = int p q du/u:

    b_G = G^2 (T + G^2)^{-1} w0^2 ,
    C4 - X(G) = < w0^2, T (T + G^2)^{-1} w0^2 >  >= 0 ,

and s = G^2 -> C4 - X(s) is completely monotone (a Laplace transform of a
positive measure).  This script computes X(G) spectrally, verifies the
operator representation against an explicit eigenfunction expansion, checks
positivity/monotonicity/limits, and saves the kernel for the lattice scan.
"""

import numpy as np
from scipy.linalg import eig, solve

import script_paths
from spectral_check import cheb

B_C = 5.13126764


def grid_and_operators(N=240):
    D, xs = cheb(N)
    u = (1.0 - xs) / 2.0           # u[0] = 0 (boundary), u[-1] = 1 (horizon)
    D = -2.0 * D                   # d/du
    return u, D, D @ D


def clencurt_weights(N):
    """Clenshaw-Curtis weights on [-1,1] for Chebyshev extreme points (Trefethen)."""
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    v = np.ones(N - 1)
    for k in range(1, N // 2 + 1):
        factor = 2.0 if 2 * k < N else 1.0
        v -= factor * np.cos(2 * k * theta[1:-1]) / (4 * k * k - 1)
    w[0] = w[N] = 1.0 / (N * N - 1.0) if N % 2 == 0 else 1.0 / (N * N)
    w[1:-1] = 2.0 * v / N
    return w


def zero_mode(u, D, D2):
    """w0 = u^2 q from the generalised eigenproblem of spectral_check.py."""
    L = (np.diag(u - u**5) @ D2 + np.diag(3.0 - 7.0 * u**4) @ D
         - 8.0 * np.diag(u**3))
    M = np.diag(u)
    vals, vecs = eig(L, -M)
    sel = np.isfinite(vals) & (np.abs(vals.imag) < 1e-8) & (vals.real > 0)
    idx = np.where(sel)[0]
    j = idx[np.argmin(np.abs(vals[idx].real - B_C))]
    q = vecs[:, j].real
    w0 = u**2 * q
    if w0[len(w0) // 2] < 0:
        w0 = -w0
    return vals[j].real, w0


def radial_matrix(u, D, D2):
    """T acting on nodal values:  (T b)(u) = -u (f b'/u)' = -f b'' + (f/u - f') b'.
    Row 0 (u = 0) is replaced by the Dirichlet condition b(0) = 0;
    the degenerate equation at u = 1 imposes horizon regularity."""
    f = 1.0 - u**4
    fp = -4.0 * u**3
    finv_u = np.zeros_like(u)
    finv_u[1:] = f[1:] / u[1:]
    T = -np.diag(f) @ D2 + np.diag(finv_u - fp) @ D
    return T


def solve_bG(T, u, w0, G2):
    A = T + G2 * np.eye(len(u))
    rhs = G2 * w0**2
    A[0, :] = 0.0
    A[0, 0] = 1.0                  # b(0) = 0
    rhs = rhs.copy()
    rhs[0] = 0.0
    return solve(A, rhs)


def main():
    N = 240
    u, D, D2 = grid_and_operators(N)
    wcc = clencurt_weights(N) / 2.0          # quadrature weights on [0,1]

    Bc, w0 = zero_mode(u, D, D2)
    print(f"zero mode eigenvalue: B_c = {Bc:.8f}")

    # integrals with weight 1/u: integrands vanish fast at u=0 (w0 ~ u^2)
    integ = lambda vals: float(wcc @ vals)
    one_over_u = np.zeros_like(u)
    one_over_u[1:] = 1.0 / u[1:]
    J2 = integ(w0**2 * one_over_u)
    w0 /= np.sqrt(J2)                        # normalise J2 = 1
    J2 = integ(w0**2 * one_over_u)
    C4 = integ(w0**4 * one_over_u)
    print(f"J2 (normalised) = {J2:.10f},  C4 = {C4:.10f}")

    T = radial_matrix(u, D, D2)

    # --- kernel on a grid of G^2 -------------------------------------------
    G2max = 60.0 * B_C
    G2grid = np.linspace(0.0, G2max, 481)
    X = np.zeros_like(G2grid)
    b_horizon = np.zeros_like(G2grid)
    for i, G2 in enumerate(G2grid):
        if G2 == 0.0:
            continue
        b = solve_bG(T, u, w0, G2)
        X[i] = integ(b * w0**2 * one_over_u)
        b_horizon[i] = b[-1]

    ker = C4 - X
    # The G = 0 grid point is not solved: there b_G = 0 identically (the
    # source is proportional to G^2), and the stored X[0] = 0 is the limit
    # G -> 0.  What can fail is that limit, so check it: X(s) = s X'(0) + O(s^2)
    # with X'(0) = <w0^2, T^{-1} w0^2> from a separate solve with the same
    # boundary rows.
    Tdir = T.copy()
    Tdir[0, :] = 0.0
    Tdir[0, 0] = 1.0
    rhs0 = w0**2
    rhs0[0] = 0.0
    dX0 = integ(solve(Tdir, rhs0) * w0**2 * one_over_u)
    s_small = 1e-6 * B_C
    X_small = integ(solve_bG(T, u, w0, s_small) * w0**2 * one_over_u)
    lim_err = abs(X_small / s_small / dX0 - 1)
    print(f"\nX(s)/s at s = {s_small:.1e}: {X_small / s_small:.10f}, "
          f"X'(0) = <w0^2, T^-1 w0^2> = {dX0:.10f}  (rel {lim_err:.1e})")
    print(f"X -> C4 at large G:  X(G2={G2grid[-1]:.0f}) / C4 = {X[-1]/C4:.6f}")
    mono = np.all(np.diff(X[1:]) > 0)
    pos = np.all(ker > -1e-12) and np.all(X >= -1e-12)
    print(f"0 <= X(G) <= C4 everywhere: {pos};  X monotonically increasing: {mono}")

    # numerical complete-monotonicity check on C4 - X(s), s = G^2
    h = ker[1:]
    ds = G2grid[1] - G2grid[0]
    signs_ok = True
    deriv = h.copy()
    for k in range(1, 5):
        deriv = np.diff(deriv) / ds
        want = (-1) ** k
        if not np.all(want * deriv[:-5] > -1e-10):
            signs_ok = False
    print(f"finite-difference complete monotonicity (k<=4): {signs_ok}")

    # --- spectral representation check -------------------------------------
    # T phi_n = kappa_n^2 phi_n with phi(0) = 0; restrict to nodes 1..N
    Tred = T[1:, 1:]
    kap2, V = eig(Tred)
    sel = np.isfinite(kap2) & (np.abs(kap2.imag) < 1e-6 * np.abs(kap2.real))
    kap2, V = kap2[sel].real, V[:, sel].real
    order = np.argsort(kap2)
    kap2, V = kap2[order], V[:, order]
    nkeep = 25
    cn = np.zeros(nkeep)
    for n in range(nkeep):
        phi = np.concatenate([[0.0], V[:, n]])
        norm = integ(phi**2 * one_over_u)
        phi /= np.sqrt(norm)
        cn[n] = integ(w0**2 * phi * one_over_u)
    print(f"\nlowest kappa_n^2: {np.array2string(kap2[:5], precision=4)}")
    print(f"sum c_n^2 (n<{nkeep}) = {np.sum(cn**2):.8f}  vs  C4 = {C4:.8f}")
    spec_err = 0.0
    for G2 in (10.0, 50.0, 200.0):
        Xspec = np.sum(cn**2 * G2 / (kap2[:nkeep] + G2))
        Xdir = integ(solve_bG(T, u, w0, G2) * w0**2 * one_over_u)
        spec_err = max(spec_err, abs(Xspec - Xdir))
        print(f"  G^2 = {G2:6.1f}:  X direct = {Xdir:.8f},  "
              f"spectral sum = {Xspec:.8f}")

    # Checks.  C4 (appendix E.4 of the paper) and B_c (section 3.1) are the
    # published values; the spectral (Stieltjes) form and the
    # complete monotonicity are the inputs of the triangular-lattice argument.
    checks = {
        "B_c = 5.13126764": abs(Bc - 5.13126764) < 5e-9,
        "C4 = 1.5805931": abs(C4 - 1.5805931) < 5e-8,
        "X(s) -> 0 as s -> 0, with slope <w0^2, T^-1 w0^2>": lim_err < 1e-5 and X[0] == 0.0,
        "0 <= X <= C4": bool(pos),
        "X increasing": bool(mono),
        "complete monotonicity, k <= 4": bool(signs_ok),
        "sum c_n^2 = C4": abs(np.sum(cn**2) - C4) < 1e-8,
        "spectral sum = direct X": spec_err < 1e-8,
        "overtones = B_n tower": bool(np.all(np.abs(kap2[:4] - np.array(
            [5.13126764, 22.48156859, 51.20975989, 91.41056318])) < 5e-8)),
    }
    failed = [k for k, v in checks.items() if not v]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + "; ".join(failed))
        raise SystemExit(1)

    np.savez(script_paths.data("kernel.npz"), G2grid=G2grid, X=X, C4=C4, J2=J2,
             B_c=Bc, u=u, w0=w0, b_horizon=b_horizon,
             kap2=kap2[:nkeep], cn=cn)
    print("\nKernel saved to scripts/kernel.npz")


if __name__ == "__main__":
    main()
