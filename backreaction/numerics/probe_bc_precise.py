"""The probe critical field B_c u_H^2 (d = 4) to 30 digits, by two methods.

With w(u) = u^2 q(u) the probe zero-mode equation on AdS5-Schwarzschild
(scripts/spectral_check.py) is the everywhere-regular

    (u - u^5) q'' + (3 - 7u^4) q' - 8 u^3 q + B u q = 0,   u in [0, 1],

with q analytic at both ends: at u = 0 that is the source-free condition
(the other Frobenius solution is u^-2 + log), at u = 1 horizon regularity
(the other solution is log(1 - u)).  The singular points nearest to the
interval are u = -1, +-i.

  (1) Frobenius/Wronskian: the analytic Frobenius series about u = 0 and about
      u = 1 (each of radius 1) are summed at u = 1/2 and B_c is the root of
      their Wronskian, in mpmath at 50 digits; the error falls like 2^-K in
      the series order K (9e-19, 9e-25, 8e-31 at K = 60, 80, 100).
  (2) Chebyshev collocation in mpmath: B_c is the root of det(L + B M) for
      the collocated equation (no boundary rows; regularity at both ends is
      enforced by collocating the degenerate equation there, as in
      spectral_check.py); the error falls like e^-0.7N (4e-16, 6e-22, 7e-29
      at N = 20, 30, 40).

Result: B_c u_H^2 = 5.131267637682236116498479102784, both methods agreeing
to 2e-49 relative at K = 180, N = 90.

The two share only the equation.  The value fixes BC0_TARGET in
numerics/bc_alpha.py, the reference against which table 4's collocation
error (5e-12) of the probe B_c is measured.  The paper quotes B_c u_H^2 to
eight decimals only; the 30-digit value is an extra, not in the paper.

Run:  uv run python -m backreaction.numerics.probe_bc_precise    (~20 s)
"""

import time

import mpmath as mp

CHECKS = {}


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# the equation P2 q'' + P1 q' + P0 q = 0 as polynomial coefficient lists in u
# (lowest power first); P0 = -8u^3 + B u is split into P0A + B * P0B
P2 = [0, 1, 0, 0, 0, -1]
P1 = [3, 0, 0, 0, -7]
P0A = [0, 0, 0, -8]
P0B = [0, 1]


def shift(poly, u0):
    """Coefficients of poly(u0 + t) in t."""
    n = len(poly)
    out = [mp.mpf(0)] * n
    for k, c in enumerate(poly):
        for j in range(k + 1):
            out[j] += c * mp.binomial(k, j) * mp.mpf(u0) ** (k - j)
    return out


def frobenius(B, u0, K):
    """Taylor coefficients a_0..a_K (a_0 = 1) of the analytic solution about the
    regular singular point u0 (P2(u0) = 0).  The coefficient of t^m reads
    sum_k [p2_k (m-k+2)(m-k+1) a_{m-k+2} + p1_k (m-k+1) a_{m-k+1}
    + p0_k a_{m-k}] = 0, solved for a_{m+1}, which enters with
    (m+1)(m p2_1 + p1_0) != 0 at both ends."""
    p2 = shift(P2, u0)
    p1 = shift(P1, u0)
    p0 = [
        x + B * y for x, y in zip(shift(P0A, u0), shift(P0B, u0) + [0, 0], strict=False)
    ]
    assert p2[0] == 0
    a = [mp.mpf(1)]
    for m in range(K):
        s = mp.mpf(0)
        for k in range(2, len(p2)):
            n = m - k + 2
            if n >= 0:
                s += p2[k] * n * (n - 1) * a[n]
        for k in range(1, len(p1)):
            n = m - k + 1
            if n >= 0:
                s += p1[k] * n * a[n]
        for k in range(len(p0)):
            n = m - k
            if n >= 0:
                s += p0[k] * a[n]
        a.append(-s / ((m + 1) * (m * p2[1] + p1[0])))
    return a


def series_at(a, t):
    v = d = mp.mpf(0)
    for n in range(len(a) - 1, -1, -1):
        v = v * t + a[n]
    for n in range(len(a) - 1, 0, -1):
        d = d * t + n * a[n]
    return v, d


def wronskian(B, K):
    um = mp.mpf(1) / 2
    q0, q0p = series_at(frobenius(B, 0, K), um)
    q1, q1p = series_at(frobenius(B, 1, K), um - 1)
    return q0 * q1p - q0p * q1


def bc_frobenius(K, B0="5.1312676377"):
    return mp.findroot(lambda B: wronskian(B, K), mp.mpf(B0), solver="secant")


def colloc_mats(N):
    """L, M of the collocated equation on Chebyshev points mapped to [0, 1]."""
    xs = [mp.cos(mp.pi * j / N) for j in range(N + 1)]
    c = [(2 if j in (0, N) else 1) * (-1) ** j for j in range(N + 1)]
    D = mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = mp.mpf(c[i]) / c[j] / (xs[i] - xs[j])
        D[i, i] = -sum(D[i, j] for j in range(N + 1) if j != i)
    u = [(1 - x) / 2 for x in xs]
    D = -2 * D
    D2 = D * D
    L = mp.matrix(N + 1, N + 1)
    for i in range(N + 1):
        ui = u[i]
        for j in range(N + 1):
            L[i, j] = (ui - ui**5) * D2[i, j] + (3 - 7 * ui**4) * D[i, j]
        L[i, i] += -8 * ui**3
    return L, u


def bc_colloc(N, B0="5.1312676377"):
    L, u = colloc_mats(N)

    def det(B):
        A = L.copy()
        for i in range(N + 1):
            A[i, i] += B * u[i]
        return mp.det(A)

    # det is O(10^180) in scale, so findroot's |f| test is useless: iterate
    # the secant until the step is below the working precision
    x0, x1 = mp.mpf(B0), mp.mpf(B0) + mp.mpf("1e-8")
    f0, f1 = det(x0), det(x1)
    for _ in range(40):
        x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f0, f1 = f1, det(x1)
        if abs(x1 - x0) < mp.mpf(10) ** (5 - mp.mp.dps):
            return x1
    raise RuntimeError("collocation secant did not converge")


def main():
    t0 = time.time()
    mp.mp.dps = 50
    print("probe B_c u_H^2 (d = 4) at 50 digits")

    print("\n(1) Frobenius series about u = 0 and u = 1, Wronskian at u = 1/2")
    fro = {}
    for K in (60, 100, 140, 180):
        fro[K] = bc_frobenius(K)
        print(f"  K = {K:3d}:  B_c = {mp.nstr(fro[K], 35)}")
    d_fro = abs(fro[180] - fro[140])
    print(f"  |K=180 - K=140| = {mp.nstr(d_fro, 3)}")
    check("(1) converged in K to 1e-30", d_fro < mp.mpf("1e-30"))

    print("\n(2) Chebyshev collocation, root of det(L + B M)")
    col = {}
    for N in (30, 50, 70, 90):
        col[N] = bc_colloc(N)
        print(f"  N = {N:3d}:  B_c = {mp.nstr(col[N], 35)}")
    d_col = abs(col[90] - col[70])
    print(f"  |N=90 - N=70| = {mp.nstr(d_col, 3)}")
    check("(2) converged in N to 1e-25", d_col < mp.mpf("1e-25"))

    diff = abs(fro[180] - col[90]) / fro[180]
    check(
        "the two methods agree to 1e-25 relative",
        diff < mp.mpf("1e-25"),
        mp.nstr(diff, 3),
    )
    print(f"\n  B_c u_H^2 = {mp.nstr(fro[180], 30)}")

    print(f"\nTOTAL runtime {time.time() - t0:.1f}s")
    nfail = sum(1 for v in CHECKS.values() if not v)
    print(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
