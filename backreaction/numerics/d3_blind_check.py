"""Blind check: is the T = 0 critical coupling in d = 3 the BF value alpha_* = 8/3?

Einstein-SU(2) Yang-Mills in AdS4, in the project normalisation
(paper sec. 2.1, generalised to d + 1 dimensions):

    S = (1/2 kappa^2) int sqrt(-g) [R + d(d-1) - (alpha/2) F^a F^a],
    R_MN + d g_MN = alpha (F_MP F_N^P - F^2 g_MN / (2(d-1))),     [ST1]

background the magnetic brane  ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2 + dy^2),
A^3 = B x dy.  Charged mode: the polarised lowest Landau level W_y = -i W_x of
W = A^1 + i A^2, static, source-free at the boundary.

WHAT IS DONE (an independent implementation that shares no code or data with
brane_flows.py, t0_threshold.py or onset_scan.py)
-------------
 [S1] Ricci form [ST1] from the action by trace reversal, general d.
 [S2] magnetic RN-AdS4,  U = r^2 - m/r + alpha B^2 / (2 r^2),  e^{2V} = r^2,
      solves [ST1] for d = 3 (and a wrong charge coefficient does not).
 [S3] linearised YM for the polarised LLL  W_x = e^{-B x^2/2} w(r),
      W_y = -i W_x  reduces EXACTLY to  (U w')' + B e^{-2V} w = 0  (all four
      components; the opposite-sign Gaussian is inconsistent).
 [S4] units r_h = 1, z = r_h/r:  (f w_z)_z + bh w = 0,  w(0) = 0 (source-free),
      f = 1 - (1+q) z^3 + q z^4,  q = alpha bh^2 / 2,  bh = B/r_h^2,
      T/sqrt(B) = (3 - q)/(4 pi sqrt(bh)).
      Extremal q = 3: f = (1-z)^2 (1 + 2z + 3z^2), AdS2 x R^2 throat with
      f ~ 6 x^2 (x = 1 - z, L2^2 = 1/6), exponents 6 s(s+1) + bh = 0, so
      BF violation <=> bh > 3/2.  At T = 0 the coupling is the single
      eigenvalue parameter bh = sqrt(6/alpha) (B sets the only scale):
          alpha_BF(3) = 6/(3/2)^2 = 8/3.
 [N1] T = 0 node count: the source-free solution at threshold bh = 3/2,
      integrated from the boundary into the throat (log variable).
 [N2] T = 0 ground state below threshold, three ways: shooting (coefficient
      of the non-normalisable throat branch), Chebyshev collocation of the
      analytic Frobenius factor  w = x^{s+} g(x), and the convergent Frobenius
      series itself (radius sqrt 2 > 1, so c0 = sum a_k exactly).
 [N3] rigorous variational upper bound: Rayleigh-Ritz on the basis
      z^{k+1} (1-z)^{-p}, p = 3/8 (finite norm and energy for p < 1/2), whose
      matrix elements are Beta functions, evaluated exactly (mpmath, 40
      digits); the Rayleigh quotient of the Ritz vector is recomputed in the
      same precision, so it is an upper bound on bh_0 whatever the accuracy of
      the floating-point eigensolver.  Two terms already give a quotient
      < 3/2 (a bound state below the BF threshold) and alpha_c(3) > 3.03;
      eight give alpha_c(3) > 3.0370.
 [N4] finite T: onset bh_c(q) by Chebyshev GEP and by horizon shooting;
      alpha_c(T/sqrt B) = 2q/bh_c^2 and its T -> 0 limit; the temperature at
      which alpha_c crosses 8/3, root-found by both methods.
 [N5] d = 4 anchor: the T = 0 D'Hoker-Kraus brane built here from scratch
      (AdS3 x R^2 throat + its irrelevant mode r^gamma,
      gamma = -1 + sqrt(57)/3, integrated out to AdS5) and the same
      node count / source scan; expected answer alpha_c(4) = 2/3.

Run:  uv run python -m backreaction.numerics.d3_blind_check
"""

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import eig, eigh
from scipy.optimize import brentq


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    assert ok, name


# ===========================================================================
# symbolic
# ===========================================================================
def ricci_diag_metric(g, X):
    n = len(X)
    gi = g.inv()
    Gam = [
        [
            [
                sp.together(
                    sum(
                        gi[a, e]
                        * (
                            sp.diff(g[e, b], X[c])
                            + sp.diff(g[e, c], X[b])
                            - sp.diff(g[b, c], X[e])
                        )
                        for e in range(n)
                    )
                    / 2
                )
                for c in range(n)
            ]
            for b in range(n)
        ]
        for a in range(n)
    ]

    def R(b, c):
        return sp.simplify(
            sum(
                sp.diff(Gam[a][b][c], X[a])
                - sp.diff(Gam[a][b][a], X[c])
                + sum(
                    Gam[a][a][e] * Gam[e][b][c] - Gam[a][c][e] * Gam[e][b][a]
                    for e in range(n)
                )
                for a in range(n)
            )
        )

    return R, gi


def symbolic():
    print("[S1] Ricci form from the action, general d")
    d, al, R, F2, FF, g = sp.symbols("d alpha R F2 FF g")
    D = d + 1
    # Einstein eq: R_MN - R g/2 - d(d-1) g/2 = alpha (FF_MN - g F2/4); trace with g^MN g_MN = D
    Rs = sp.solve(
        sp.Eq(R - R * D / 2 - d * (d - 1) * D / 2, al * (F2 - D * F2 / 4)), R
    )[0]
    ricci = sp.simplify(Rs * g / 2 + d * (d - 1) * g / 2 + al * (FF - g * F2 / 4))
    target = -d * g + al * (FF - F2 * g / (2 * (d - 1)))
    check("R_MN + d g = alpha(FF - F^2 g/(2(d-1)))", sp.simplify(ricci - target) == 0)

    print("[S2] magnetic RN-AdS4 solves the d = 3 equations")
    t, x, y, r = sp.symbols("t x y r", real=True)
    B, m = sp.symbols("B m", positive=True)
    al = sp.Symbol("alpha", positive=True)
    U = sp.Function("U")(r)
    V = sp.Function("V")(r)
    X = [t, x, y, r]
    g = sp.diag(-U, sp.exp(2 * V), sp.exp(2 * V), 1 / U)
    Ric, gi = ricci_diag_metric(g, X)
    F = sp.zeros(4)
    F[1, 2], F[2, 1] = B, -B
    F2 = sum(
        F[a, b] * F[c, e] * gi[a, c] * gi[b, e]
        for a in range(4)
        for b in range(4)
        for c in range(4)
        for e in range(4)
    )

    def eom(coef):
        sub = {U: r**2 - m / r + coef * al * B**2 / r**2, V: sp.log(r)}
        out = []
        for a in range(4):
            FFa = sum(F[a, p] * F[a, q] * gi[p, q] for p in range(4) for q in range(4))
            E = Ric(a, a) + 3 * g[a, a] - al * (FFa - g[a, a] * F2 / 4)
            out.append(sp.simplify(E.subs(sub).doit()))
        return out

    check(
        "U = r^2 - m/r + alpha B^2/(2 r^2)", all(e == 0 for e in eom(sp.Rational(1, 2)))
    )
    check("negative control: coefficient 1 fails", any(e != 0 for e in eom(1)))

    print("[S3] polarised LLL reduces to (U w')' + B e^{-2V} w = 0")
    eps = sp.Symbol("epsilon", positive=True)
    w = sp.Function("w")(r)
    sq = sp.exp(2 * V)
    lc = sp.LeviCivita
    sl = sp.diff(U * sp.diff(w, r), r) + B * sp.exp(-2 * V) * w
    for sgn, expect in ((1, True), (-1, False)):
        psi = sp.exp(-sgn * B * x**2 / 2)
        Wd = [0, psi * w, -sp.I * psi * w, 0]
        Wc = [0, psi * w, sp.I * psi * w, 0]
        A = [
            [eps * (Wd[M] + Wc[M]) / 2 for M in range(4)],
            [eps * (Wd[M] - Wc[M]) / (2 * sp.I) for M in range(4)],
            [0, 0, B * x, 0],
        ]
        Fa = [
            [
                [
                    sp.diff(A[a][N], X[M])
                    - sp.diff(A[a][M], X[N])
                    + sum(
                        lc(a, b, c) * A[b][M] * A[c][N]
                        for b in range(3)
                        for c in range(3)
                    )
                    for N in range(4)
                ]
                for M in range(4)
            ]
            for a in range(3)
        ]
        Fu = [
            [
                [
                    sum(
                        gi[M, P] * gi[N, Q] * Fa[a][P][Q]
                        for P in range(4)
                        for Q in range(4)
                    )
                    for N in range(4)
                ]
                for M in range(4)
            ]
            for a in range(3)
        ]

        def E(a, N, Fu=Fu, A=A):
            return sum(
                sp.diff(sq * Fu[a][M][N], X[M])
                + sum(
                    lc(a, b, c) * A[b][M] * sq * Fu[c][M][N]
                    for b in range(3)
                    for c in range(3)
                )
                for M in range(4)
            )

        lin = [
            sp.simplify(sp.diff(E(0, N) + sp.I * E(1, N), eps).subs(eps, 0) / psi)
            for N in range(4)
        ]
        ok = (
            lin[0] == 0
            and lin[3] == 0
            and sp.simplify(lin[1] - sl) == 0
            and sp.simplify(lin[2] + sp.I * sl) == 0
        )
        check(
            f"Gaussian sign {sgn:+d} {'reduces' if expect else 'is inconsistent'}",
            ok == expect,
        )

    print("[S4] extremal throat and the BF value")
    z, q, bh, s = sp.symbols("z q bh s")
    f = 1 - (1 + q) * z**3 + q * z**4
    check(
        "f(q=3) = (1-z)^2 (1+2z+3z^2)",
        sp.expand(f.subs(q, 3) - (1 - z) ** 2 * (1 + 2 * z + 3 * z**2)) == 0,
    )
    # z-form: with u = r^2 f(z), (U w_r)_r + (B/r^2) w = z^2 [(f w_z)_z + bh w]
    rho = sp.Symbol("rho", positive=True)
    wf = sp.Function("w")
    lhs = sp.diff(
        (rho**2 * f.subs(z, 1 / rho)) * sp.diff(wf(1 / rho), rho), rho
    ) + bh / rho**2 * wf(1 / rho)
    zz = sp.Symbol("zz", positive=True)
    rhs = zz**2 * (sp.diff(f.subs(z, zz) * sp.diff(wf(zz), zz), zz) + bh * wf(zz))
    check(
        "radial eq in z = 1/rho is (f w_z)_z + bh w = 0",
        sp.simplify(lhs.subs(rho, 1 / zz).doit() - rhs) == 0,
    )
    T_over = sp.diff(rho**2 * f.subs(z, 1 / rho), rho).subs(rho, 1) / (4 * sp.pi)
    check("T r_h^-1 = (3 - q)/(4 pi)", sp.simplify(T_over - (3 - q) / (4 * sp.pi)) == 0)
    xx = sp.Symbol("x")
    ind = sp.expand(sp.diff((6 * xx**2) * sp.diff(xx**s, xx), xx) / xx**s + bh)
    sBF = sp.solve(sp.diff(ind, s), s)[0]
    bBF = sp.solve(ind.subs(s, sBF), bh)[0]
    aBF = 2 * 3 / bBF**2  # q = alpha bh^2/2 = 3
    check(
        "alpha_BF(3) = 8/3",
        sp.simplify(aBF - sp.Rational(8, 3)) == 0,
        f"(bh_BF = {bBF})",
    )
    check(
        "paper's alpha_*(d) = 16/(d(d-1)(d-2)) at d=3",
        sp.Rational(16, 3 * 2 * 1) == aBF,
    )


# ===========================================================================
# d = 3 numerics
# ===========================================================================
B_BF = 1.5


def s_plus(b):
    return (-1 + np.sqrt(1 - 2 * b / 3)) / 2


def h_ext(z):
    return 1 + 2 * z + 3 * z**2  # f/(1-z)^2 at q = 3


def shoot_from_boundary(b, xi_end=40.0, events=False):
    """xi = -ln(1-z); Y = (w, Pi), Pi = h w_xi.  w(0)=0, w_z(0)=1."""

    def rhs(xi, Y):
        return [Y[1] / h_ext(-np.expm1(-xi)), Y[1] - b * Y[0]]

    ev = (lambda xi, Y: Y[0]) if events else None
    return solve_ivp(
        rhs, [0, xi_end], [0.0, 1.0], method="DOP853", rtol=1e-13, atol=1e-30, events=ev
    )


def growing_coeff(b, xi_end=40.0):
    sol = shoot_from_boundary(b, xi_end)
    w, Pi = sol.y[:, -1]
    wp = Pi / h_ext(-np.expm1(-xi_end))
    rt = np.sqrt(1 - 2 * b / 3)
    lp, lm = (1 + rt) / 2, (1 - rt) / 2
    return (wp - lm * w) * np.exp(-lp * xi_end)


def cheb(N):
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.r_[2, np.ones(N - 1), 2] * (-1) ** np.arange(N + 1)
    Xm = np.tile(x, (N + 1, 1)).T
    D = np.outer(c, 1 / c) / (Xm - Xm.T + np.eye(N + 1))
    return D - np.diag(D.sum(1)), x


def c0_cheb(b, N):
    """w = x^{s+} g(x), x = 1 - z; g analytic, g(0) = 1; returns c0 = g(1)."""
    D, t = cheb(N)
    x = (1 - t) / 2
    D = -2 * D
    s = s_plus(b)
    h = 6 - 8 * x + 3 * x**2
    hp = -8 + 6 * x
    L = (
        np.diag(x * h) @ D @ D
        + np.diag(2 * h + x * hp + 2 * s * h) @ D
        + np.diag(s * (s + 1) * (-8 + 3 * x) + s * hp)
    )
    L[-1, :] = 0
    L[-1, 0] = 1
    rhs = np.zeros(N + 1)
    rhs[-1] = 1
    return np.linalg.solve(L, rhs)[-1]


def c0_series(b, K=200):
    s = s_plus(b)
    a = [1.0]
    for n in range(1, K):
        a2 = a[n - 2] if n >= 2 else 0.0
        a.append(
            (
                8 * a[n - 1] * (n - 1 + s) * (n + 1 + s)
                - 3 * a2 * (n - 2 + s) * (n + 1 + s)
            )
            / (6 * (n + s) * (n + s + 1) + b)
        )
    return sum(a)


def ritz_matrices(K, p):
    """Energy and norm matrices of int f w'^2 / int w^2 on the basis
    w_k = z^{k+1} (1-z)^{-p}, k < K, with f = (1-z)^2 (1 + 2z + 3z^2), exactly:
    every entry is a sum of Beta functions int z^a (1-z)^b = B(a+1, b+1)."""
    p = mp.mpf(p)

    def beta(a, b):
        return mp.beta(a + 1, b + 1)

    Km, Mm = mp.matrix(K, K), mp.matrix(K, K)
    for j in range(K):
        for k in range(K):
            e = 0
            for m, hm in enumerate((1, 2, 3)):
                n = j + k + m
                e += hm * (
                    (j + 1) * (k + 1) * beta(n, 2 - 2 * p)
                    + p * (j + k + 2) * beta(n + 1, 1 - 2 * p)
                    + p * p * beta(n + 2, -2 * p)
                )
            Km[j, k] = e
            Mm[j, k] = beta(j + k + 2, -2 * p)
    return Km, Mm


def ritz_bound(K, p="0.375"):
    """A rigorous upper bound on bh_0: the exact Rayleigh quotient (40 digits)
    of the lowest Ritz vector on K basis functions."""
    with mp.workdps(40):
        Km, Mm = ritz_matrices(K, p)
        _, vec = eigh(np.array(Km.tolist(), float), np.array(Mm.tolist(), float))
        c = mp.matrix([mp.mpf(float(x)) for x in vec[:, 0]])
        return (c.T * Km * c)[0] / (c.T * Mm * c)[0]


def bc_colloc(q, N):
    D, t = cheb(N)
    z = (1 + t) / 2
    D = 2 * D
    f = 1 - (1 + q) * z**3 + q * z**4
    fp = -3 * (1 + q) * z**2 + 4 * q * z**3
    L = np.diag(f) @ D @ D + np.diag(fp) @ D
    ev = eig(-L[:-1, :-1], np.eye(N), right=False)
    ev = ev[np.isfinite(ev) & (np.abs(ev.imag) < 1e-8)].real
    return np.min(ev[ev > 0])


def w0_from_horizon(b, q, ifac=1e-8):
    eps = 3 - q
    x0 = ifac * eps

    def hh(x):
        return (1 + (1 - x) + (1 - x) ** 2 - q * (1 - x) ** 3) / x

    rhs = lambda xi, Y: [Y[1] / hh(np.exp(-xi)), Y[1] - b * Y[0]]  # noqa: E731
    w = 1 - b * x0 / eps
    sol = solve_ivp(
        rhs, [-np.log(x0), 0], [w, b * w], method="DOP853", rtol=1e-12, atol=1e-14
    )
    return sol.y[0, -1]


def d3_numerics():
    print("\n[N1] T = 0 node count at threshold bh = 3/2 (alpha = 8/3)")
    sol = shoot_from_boundary(B_BF, 60.0, events=True)
    zs = sol.t_events[0][sol.t_events[0] > 1e-9]
    print(
        f"  zeros of the source-free solution in xi = -ln(1-z): {zs}  (x = 1-z = {np.exp(-zs)})"
    )
    check("threshold solution has exactly one node", len(zs) == 1)
    for b in (1.0, 1.4):
        n = sol2 = shoot_from_boundary(b, 60.0, events=True)
        k = len(sol2.t_events[0][sol2.t_events[0] > 1e-9])
        print(f"  bh = {b}: {k} nodes")
        del n

    print("\n[N2] T = 0 ground state below the BF threshold")
    for b in (0.5, 1.0, 1.3, 1.4, 1.42, 1.45, 1.49):
        print(
            f"  bh = {b:5.2f}  c0 series = {c0_series(b):+.6e}   c0 cheb(40) = {c0_cheb(b, 40):+.6e}"
            f"   growing coeff (shoot) = {growing_coeff(b):+.4e}"
        )
    b_ser = brentq(c0_series, 1.0, 1.4999, xtol=1e-15)
    b_ch = {
        N: brentq(lambda b, N=N: c0_cheb(b, N), 1.3, 1.45, xtol=1e-15)
        for N in (10, 20, 30, 40, 60)
    }
    b_sh = {
        xe: brentq(lambda b, xe=xe: growing_coeff(b, xe), 1.3, 1.45, xtol=1e-15)
        for xe in (20, 30, 40, 50)
    }
    for N, v in b_ch.items():
        print(f"  Chebyshev N = {N:3d}: bh_0 = {v:.15f}")
    for xe, v in b_sh.items():
        print(f"  shooting xi_end = {xe}: bh_0 = {v:.15f}")
    print(f"  Frobenius series:      bh_0 = {b_ser:.15f}")
    b0 = b_ser
    spread = max(b_ch[40], b_sh[40], b0) - min(b_ch[40], b_sh[40], b0)
    check(
        "Chebyshev, shooting and series agree to 3e-14",
        spread < 3e-14,
        f"(spread {spread:.1e})",
    )
    # table 4 quotes the agreement for alpha_c(3) = 6/bh_0^2 itself (relative):
    # twice the relative spread of bh_0, measured 3.0e-14
    a3 = [6 / b**2 for b in (b_ch[40], b_sh[40], b0)]
    a_spread = (max(a3) - min(a3)) / min(a3)
    check(
        "the three methods give alpha_c(3) to 4e-14 (relative)",
        a_spread < 4e-14,
        f"(relative spread {a_spread:.1e})",
    )
    a_c = 6 / b0**2
    print(
        f"  ==> alpha_c(3) = 6/bh_0^2 = {a_c:.12f}   vs alpha_BF = 8/3 = {8 / 3:.12f}"
        f"   (2/alpha_c = {2 / a_c:.10f}; BF: 3/4)"
    )
    print(
        f"      throat exponent at bh_0: s+ = {s_plus(b0):.6f}  (normalisable: s+ > -1/2)"
    )

    print("\n[N3] rigorous variational bound (Rayleigh-Ritz, exact Beta integrals)")
    bounds = {K: ritz_bound(K) for K in (1, 2, 4, 8)}
    for K, R in bounds.items():
        print(
            f"  K = {K}: Rayleigh quotient {mp.nstr(R, 13)}  =>  alpha_c(3) > "
            f"{mp.nstr(6 / R**2, 10)}"
        )
    check(
        "two terms: quotient < 3/2 (bound state below BF) and alpha_c(3) > 3.03",
        bounds[2] < 1.5 and 6 / bounds[2] ** 2 > 3.03,
    )
    check("eight terms: alpha_c(3) > 3.0370", 6 / bounds[8] ** 2 > 3.0370)
    check(
        "quotients decrease with K and stay >= bh_0 (variational consistency)",
        all(bounds[a] > bounds[b] for a, b in ((1, 2), (2, 4), (4, 8)))
        and min(bounds.values()) >= b0,
        f"(K = 8 exceeds bh_0 by {float(bounds[8] - b0):.1e})",
    )

    print("\n[N4] finite T: alpha at onset vs T/sqrt(B)")
    rows = []
    for q in (
        0.0,
        1.0,
        2.0,
        2.5,
        2.9,
        2.99,
        2.999,
        3 - 1e-4,
        3 - 1e-5,
        3 - 1e-6,
        3 - 1e-8,
        3 - 1e-10,
    ):
        hi = (
            2.4 if q < 2.995 else 1.5
        )  # near T = 0 the BF-violating tower enters (1.5, 2.4)
        bs = brentq(lambda b, q=q: w0_from_horizon(b, q), 1.3, hi, xtol=1e-14)
        bc = bc_colloc(q, 120) if q <= 2.9 else np.nan
        rows.append((q, bs))
        print(
            f"  q = {q:.10f}  T/sqrtB = {(3 - q) / (4 * np.pi * np.sqrt(bs)):.3e}  bh_c shoot = {bs:.12f}"
            f"  colloc = {bc:.12f}  alpha_c = {2 * q / bs**2:.8f}"
        )
        if q <= 2.9:
            check("  shoot vs colloc", abs(bs - bc) < 1e-9)
    al = [2 * q / b**2 for q, b in rows]
    check("alpha_c(T) monotone increasing as T -> 0", all(np.diff(al) > 0))
    # local slope from the two coldest points (wider windows carry O(T^{4 nu}) drift:
    # fitting the last four gives 0.2625)
    e = np.array([3 - q for q, _ in rows[-2:]])
    db = np.array([b for _, b in rows[-2:]]) - b0
    p = np.polyfit(np.log(e), np.log(db), 1)[0]
    nu2 = np.sqrt(1 - 2 * b0 / 3)  # = 2 nu_AdS2
    print(
        f"  bh_c - bh_0 ~ (3-q)^p with p = {p:.4f};  2 nu_AdS2 = sqrt(1 - 2 bh_0/3) = {nu2:.4f}"
    )
    check("finite-T approach exponent = 2 nu_AdS2 (to 3%)", abs(p - nu2) / nu2 < 0.03)

    # where alpha_c(T) crosses the BF value 8/3, by both methods
    def cross(bh_of_q):
        q = brentq(lambda q: 2 * q / bh_of_q(q) ** 2 - 8 / 3, 2.95, 2.994, xtol=1e-13)
        b = bh_of_q(q)
        return q, (3 - q) / (4 * np.pi * np.sqrt(b))

    q_s, t_s = cross(
        lambda q: brentq(lambda b: w0_from_horizon(b, q), 1.3, 2.4, xtol=1e-14)
    )
    q_c, t_c = cross(lambda q: bc_colloc(q, 300))
    print(
        f"  alpha_c = 8/3 at q = {q_s:.10f}, T/sqrtB = {t_s:.6e} (shooting); "
        f"q = {q_c:.10f}, T/sqrtB = {t_c:.6e} (collocation, N = 300)"
    )
    check(
        "alpha_c(T) crosses 8/3 at T/sqrtB = 7.42e-4 (shooting and collocation)",
        abs(t_s - t_c) < 1e-9 and round(t_s, 6) == 7.42e-4,
    )
    return b0


# ===========================================================================
# d = 4 anchor: T = 0 D'Hoker-Kraus brane built from the throat outwards
# ===========================================================================
def d4_symbolic():
    """Reduced Ricci-form equations (alpha B^2 -> beta) and the throat irrelevant mode."""
    t, x, y, zc, r = sp.symbols("t x y z r", real=True)
    beta, gm, a, b, c, eps = sp.symbols("beta gamma a b c epsilon")
    U, V, W = (sp.Function(n)(r) for n in "UVW")
    X = [t, x, y, zc, r]
    g = sp.diag(-U, sp.exp(2 * V), sp.exp(2 * V), sp.exp(2 * W), 1 / U)
    Ric, gi = ricci_diag_metric(g, X)
    F = sp.zeros(5)
    F[1, 2], F[2, 1] = 1, -1  # B factored into beta = alpha B^2
    F2 = sum(
        F[i, j] * F[k, l] * gi[i, k] * gi[j, l]
        for i in range(5)
        for j in range(5)
        for k in range(5)
        for l in range(5)
    )
    E = []
    for i in range(5):
        FFi = sum(F[i, p] * F[i, q] * gi[p, q] for p in range(5) for q in range(5))
        E.append(
            sp.simplify(
                (Ric(i, i) + 4 * g[i, i] - beta * (FFi - g[i, i] * F2 / 6)) / g[i, i]
            )
        )
    th = {U: 3 * r**2, V: 0, W: sp.log(r)}
    check(
        "d=4 throat AdS3 x R^2: U=3r^2, e^{2V}=1, beta=6",
        all(sp.simplify(e.subs(th).doit().subs(beta, 6)) == 0 for e in E),
    )
    sub = {
        U: 3 * r**2 * (1 + eps * a * r**gm),
        V: eps * b * r**gm,
        W: sp.log(r) + eps * c * r**gm,
    }
    rows = []
    for e in (E[0], E[1], E[3]):
        le = sp.expand(
            sp.simplify(
                sp.diff(e.subs(sub).doit().subs(beta, 6), eps).subs(eps, 0) / r**gm
            )
        )
        rows.append([le.coeff(v) for v in (a, b, c)])
    det = sp.factor(sp.Matrix(rows).det())
    g0 = -1 + sp.sqrt(57) / 3
    check(
        "irrelevant exponent gamma = -1 + sqrt(57)/3 is a root",
        sp.simplify(det.subs(gm, g0)) == 0,
    )
    sol = sp.solve(
        [
            sum(rw[i] * v for i, v in enumerate((a, 1, c))).subs(gm, g0)
            for rw in rows[::2]
        ],
        [a, c],
    )
    allrows = [
        sp.simplify(
            sum(rw[i] * v for i, v in enumerate((a, 1, c))).subs(gm, g0).subs(sol)
        )
        for rw in rows
    ]
    check(
        "eigenvector satisfies all three linearised equations",
        all(v == 0 for v in allrows),
    )
    # second-order ODEs, solved for U'', V'', W''
    Upp, Vpp, Wpp = sp.symbols("Upp Vpp Wpp")
    reps = {
        sp.Derivative(U, (r, 2)): Upp,
        sp.Derivative(V, (r, 2)): Vpp,
        sp.Derivative(W, (r, 2)): Wpp,
    }
    sec = sp.solve(
        [E[0].subs(reps), E[1].subs(reps), E[3].subs(reps)], [Upp, Vpp, Wpp], dict=True
    )[0]
    Us, Ups, Vs, Vps, Ws, Wps = sp.symbols("U Up V Vp W Wp")
    rep1 = {
        sp.Derivative(U, r): Ups,
        sp.Derivative(V, r): Vps,
        sp.Derivative(W, r): Wps,
    }
    rep0 = {U: Us, V: Vs, W: Ws}
    args = (Us, Ups, Vs, Vps, Ws, Wps, beta)
    acc = [sp.lambdify(args, sec[k].subs(rep1).subs(rep0)) for k in (Upp, Vpp, Wpp)]
    cons = sp.lambdify(args, E[4].subs(reps).subs(sec).subs(rep1).subs(rep0))
    return float(g0), float(sol[a]), float(sol[c]), acc, cons


def d4_anchor():
    print("\n[N5] d = 4 anchor on the T = 0 D'Hoker-Kraus brane")
    GAM, A_, C_, acc, cons = d4_symbolic()
    BETA = 6.0  # raw frame: throat e^{2V} = 1, so alpha B_raw^2 = 6
    S0, S1 = np.log(1e-5), np.log(1e7)

    def bg_rhs(s, Y):
        r = np.exp(s)
        return [
            r * Y[1],
            r * acc[0](*Y, BETA),
            r * Y[3],
            r * acc[1](*Y, BETA),
            r * Y[5],
            r * acc[2](*Y, BETA),
        ]

    r0 = np.exp(S0)
    dd = r0**GAM
    Y0 = [
        3 * r0**2 * (1 + A_ * dd),
        6 * r0 * (1 + A_ * dd) + 3 * r0 * A_ * GAM * dd,
        dd,
        GAM * dd / r0,
        np.log(r0) + C_ * dd,
        1 / r0 + C_ * GAM * dd / r0,
    ]
    bg = solve_ivp(
        bg_rhs, [S0, S1], Y0, method="DOP853", rtol=1e-13, atol=1e-16, dense_output=True
    )
    Yend = bg.sol(S1)
    rend = np.exp(S1)
    cV = np.exp(2 * Yend[2]) / rend**2
    drift = max(abs(cons(*bg.sol(s), BETA)) for s in np.linspace(S0, S1, 200))
    print(
        f"  U/r^2 -> {Yend[0] / rend**2:.9f},  e^2V/r^2 -> {cV:.6f},  e^2W/r^2 -> "
        f"{np.exp(2 * Yend[4]) / rend**2:.6f},  max |rr constraint| = {drift:.1e}"
    )
    check("brane flows to AdS5 (U/r^2 -> 1)", abs(Yend[0] / rend**2 - 1) < 1e-6)
    check("constraint preserved", drift < 1e-10)

    def rhs(lam):
        def f(s, y):
            r = np.exp(s)
            U, _, V, _, W, _ = bg.sol(s)
            return [r * y[1] / (U * np.exp(W)), -r * lam * np.exp(W - 2 * V) * y[0]]

        return f

    def c0(lam):
        p = -1 + np.sqrt(1 - lam / 3)
        r = np.exp(S0)
        so = solve_ivp(
            rhs(lam),
            [S0, S1],
            [r**p, 3 * r**3 * p * r ** (p - 1)],
            method="DOP853",
            rtol=1e-12,
            atol=1e-40,
        )
        w, Pi = so.y[:, -1]
        U, _, V, _, W, _ = Yend
        wr = Pi / (U * np.exp(W))
        return (w + rend * wr / 2) / (1 + lam / (4 * cV * rend**2)) / abs(w)

    def nodes(lam):
        U, _, V, _, W, _ = Yend
        so = solve_ivp(
            rhs(lam),
            [S1, S0],
            [1 / rend**2, U * np.exp(W) * (-2 / rend**3)],
            method="DOP853",
            rtol=1e-12,
            atol=1e-40,
            events=lambda s, y: y[0],
        )
        return np.exp(so.t_events[0])

    # BF threshold: b = lam l3^2 e^{-2V} = lam/3 = 1;  alpha = 6/lam^2 -> 2/3
    n_thr = nodes(3.0)
    print(
        f"  nodes of the source-free solution at threshold (lam = 3, alpha = 2/3): {len(n_thr)}"
    )
    for lam in (3.2, 4.0, 6.0):
        print(
            f"  positive control lam = {lam} (BF violated): nodes at r = {nodes(lam)}"
        )
    c0s = [c0(l) for l in (0.5, 1.0, 2.0, 2.5, 2.9, 2.99, 2.999)]
    print(
        f"  normalised source c0(lam) for lam in (0.5 .. 2.999): min = {min(c0s):.6f}"
    )
    check("d=4: no node at threshold", len(n_thr) == 0)
    check(
        "d=4: no source-free zero mode below threshold (alpha_c(4) = 2/3)", min(c0s) > 0
    )
    check("d=4 positive control: BF-violating lam has nodes", len(nodes(4.0)) >= 1)


def main():
    symbolic()
    b0 = d3_numerics()
    d4_anchor()
    print(
        f"\nSUMMARY  alpha_BF(3) = 8/3 = {8 / 3:.10f};  alpha_c(3) = {6 / b0**2:.10f} "
        f"(bh_0 = {b0:.13f});  alpha_c(3) != alpha_BF(3).  d = 4 anchor: alpha_c(4) = 2/3."
    )


if __name__ == "__main__":
    main()
