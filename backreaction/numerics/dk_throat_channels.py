"""The FULL lattice kernel on the AdS3 x R^2 throat: the metric sector at
alpha_* (paper app. C.4).

The abelian piece C4 - X of the kernel (paper app. C.2) is completely
monotone in the alpha -> alpha_* limit, so Montgomery forces triangular there
(numerics/dk_throat_lattice.py).  This module adds the metric sector, where
complete monotonicity can fail.  `derivations/dk_throat_channels.py` settles its analytic structure
in closed form; this module evaluates the resulting finite, beta-free problem.

The throat problem.  At the fixed point every coefficient of the coupled
(H_yy, H_zz, a_y) system is v-, c_W- and h-free once written in the paper app. C.4
screening invariant lambda = G^2 l3^2 / v = gamma b, and b = 1 at alpha_*.  So
the FULL kernel limit

    S_full(gamma) = lim_{beta -> oo} K(gamma B_c(beta); beta) / N(beta)

is one fixed BVP per gamma on the BTZ throat, solved here on xi = ln(rho/h) in
[0, Xi] with horizon regularity and Dirichlet at the far end.  The response rows
come from the generated `backreaction.systems.dk_response` -- i.e. from the
production system, not from the derivation module, so the two are independent.

CHECKS
------
[C1] w0 = (2/pi) sqrt(z) K(1-z) solves the b = 1 throat zero-mode equation.
[C2] The (H_yy, a_y) block read off the PRODUCTION rows is exactly the constant
     mass matrix of the derivation: M11 = M22 = lambda + 8/3, M12 M21 =
     (16/9)(3 lambda + 4), H_zz columns identically zero, first-derivative
     coefficient -(3 rho^2 - h^2)/(rho (rho^2 - h^2)) -- at several rho, gamma,
     and at h, v, c_W, b away from their default values.
[C3] Eigenvalues m_+- = (sqrt(3 lambda + 4) +- 2)^2/3 and mu_+- = sqrt(1 + m_+-).
[C4] Abelian anchor: the decoupled photon BVP of paper app. C.4, solved with
     this module's xi discretisation, reproduces S(gamma) of
     numerics/dk_throat_lattice.py (an independent solver of the same BVP).
[C5] Grid refinement of S_full: the FD + Richardson values at N and N/2 agree
     (one discretisation at two resolutions, not two methods).
[C6] The far cutoff Xi (20 vs 26) at the same gamma points and the same N.  A
     weak check: the generated throat sources are already roundoff beyond
     xi ~ 12 (noise/value 1e-4 at xi = 12, O(1) at 20), so both cutoffs lie in
     the roundoff region, and with N fixed the ~1e-7 difference mostly
     measures the change of grid step (Xi = 14 vs 20 differs by 5e-8 as well).
     The far end is tested by [C10], which needs no wall.
[C7] Complete monotonicity of S_full: signs of (-1)^k d^k/d gamma^k, k <= 4.
[C8] Lattice selection at alpha_* with the FULL kernel: C_square - C_triangle
     summed over direct solves at the exact shells (relative gap +0.176562),
     the PCHIP interpolant on the 41-node grid as an interpolation check
     (+0.17652; the same nodes, not a second method), and the paper sec. 5.2
     moduli scan on the interpolant.
[C10] The second method (table 4): S_shoot, shooting from both ends (a
     Frobenius series at the horizon, the exact decaying solutions at the far
     end, no wall), against the FD S_full at six gamma; gamma_0 by direct
     roots of both; the exact-shell gap by both.  Independent discretisations
     of the same generated rows, not independent derivations.
[C9] (--ladder) The production brane kernel of numerics/dk_kernel.py, continued
     up the beta ladder at fixed gamma, converges monotonically onto S_full.

Run:  uv run python -m backreaction.numerics.dk_throat_channels
"""

import sys
import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ellipe, ellipkm1

from backreaction import paths
from backreaction.numerics import dk_throat_lattice as TL
from backreaction.systems import dk_response as RS

T0 = time.time()
CHECKS = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# the throat background and the marginal condensate
# ---------------------------------------------------------------------------
H_DEF, V_DEF, CW_DEF, B_DEF = 2.0 / 3.0, 1.0, 1.0, 1.0


def w0_and_dw0(rho, h=H_DEF):
    """Marginal throat condensate w0 = (2/pi) sqrt(z) K(1-z), z = h^2/rho^2,
    and dw0/drho.  d/dm K(m) = (E(m) - (1-m) K(m)) / (2 m (1-m)); here
    m = 1 - z, so dK/dz = -dK/dm.  Asymptotics used near both ends."""
    rho = np.asarray(rho, dtype=float)
    z = (h / rho) ** 2
    out = np.empty_like(rho)
    dout = np.empty_like(rho)
    # K(1-z) and dK/dz
    near1 = (1.0 - z) < 1e-8
    small = z < 1e-12
    zs = np.where(small, 1e-12, z)
    K = ellipkm1(zs)  # = K(1 - z), accurate as z -> 0 where ellipk(1 - z) is not
    Ee = ellipe(1.0 - zs)
    den = 2.0 * (1.0 - zs) * zs
    dKdz = -(Ee - zs * K) / np.where(np.abs(den) < 1e-300, 1.0, den)
    w = (2.0 / np.pi) * np.sqrt(zs) * K
    dwdz = (2.0 / np.pi) * (0.5 * K / np.sqrt(zs) + np.sqrt(zs) * dKdz)
    if np.any(near1):  # z -> 1: K(m) = (pi/2)(1 + m/4), dK/dz = -pi/8
        m = 1.0 - z
        Kn = (np.pi / 2) * (1.0 + m / 4.0 + 9.0 * m**2 / 64.0)
        dKn = -(np.pi / 2) * (0.25 + 9.0 * m / 32.0)
        zz1 = np.where(near1, z, 1.0)
        w = np.where(near1, (2.0 / np.pi) * np.sqrt(zz1) * Kn, w)
        dwdz = np.where(
            near1, (2.0 / np.pi) * (0.5 * Kn / np.sqrt(zz1) + np.sqrt(zz1) * dKn), dwdz
        )
    if np.any(small):  # z -> 0: K = (1/2) ln(16/z), dK/dz = -1/(2z)
        Ks = 0.5 * np.log(16.0 / np.where(small, z, 1.0))
        ws = (2.0 / np.pi) * np.sqrt(np.where(small, z, 1.0)) * Ks
        dws = (2.0 / np.pi) * (0.5 * Ks - 0.5) / np.sqrt(np.where(small, z, 1.0))
        w = np.where(small, ws, w)
        dwdz = np.where(small, dws, dwdz)
    out = w
    dout = dwdz * (-2.0 * h**2 / rho**3)
    return out, dout


def bg_fields(rho, h=H_DEF, v=V_DEF, cW=CW_DEF, b=B_DEF, gam=1.0):
    """(r, U, U', V, V', W, W', B, G, alpha) of the throat at radius rho."""
    rho = np.asarray(rho, dtype=float)
    U = 3.0 * (rho**2 - h**2)
    Up = 6.0 * rho
    V = 0.5 * np.log(v) * np.ones_like(rho)
    Vp = np.zeros_like(rho)
    W = 0.5 * np.log(cW * rho**2)
    Wp = 1.0 / rho
    Bfld = 3.0 * v * b
    alpha = 2.0 / (3.0 * b**2)
    G = np.sqrt(3.0 * v * b * gam)  # lambda = G^2 l3^2 / v = gamma b
    return rho, U, Up, V, Vp, W, Wp, Bfld, G, alpha


def row_triples(rho, h=H_DEF, v=V_DEF, cW=CW_DEF, b=B_DEF, gam=1.0, src=True):
    """(cH, cdH, s) for the three evolution rows and the two algebraic rows at
    one radius, straight from the generated production system."""
    r, U, Up, V, Vp, W, Wp, Bf, G, al = bg_fields(float(rho), h, v, cW, b, gam)
    if src:
        W0, W0p = w0_and_dw0(np.array([float(rho)]), h)
        W0, W0p = float(W0[0]), float(W0p[0])
    else:
        W0 = W0p = 0.0
    a = (
        float(r),
        float(U),
        float(Up),
        float(V),
        float(Vp),
        float(W),
        float(Wp),
        float(Bf),
        float(G),
        float(al),
        W0,
        W0p,
    )
    return {
        "yy": RS.row_hyy(*a),
        "zz": RS.row_hzz(*a),
        "ay": RS.row_ay(*a),
        "Hrr": RS.alg_Hrr(*a),
        "Hxr": RS.alg_Hxr(*a),
        "args": a,
    }


def mass_matrix(rho, h=H_DEF, v=V_DEF, cW=CW_DEF, b=B_DEF, gam=1.0):
    """The 2x2 mass matrix in the (H_yy, bhat = i G a_y) basis, read off the
    production rows: coefficient of field j in row i, times (rho^2 - h^2)."""
    tr = row_triples(rho, h, v, cW, b, gam, src=False)
    _, _, _, _, _, _, _, _, G, _ = bg_fields(float(rho), h, v, cW, b, gam)
    f = rho**2 - h**2
    cHy, cdHy, _ = tr["yy"]
    cHa, cdHa, _ = tr["ay"]
    M = np.array(
        [[cHy[0] * f, cHy[2] * f / (1j * G)], [1j * G * cHa[0] * f, cHa[2] * f]],
        dtype=complex,
    )
    zz_cols = np.array([cHy[1], cdHy[1], cHa[1], cdHa[1]], dtype=complex)
    dcoef = np.array([cdHy[0], cdHa[2]], dtype=complex)
    return M, zz_cols, dcoef


def channel_masses(lam):
    """m_+- = (sqrt(3 lambda + 4) +- 2)^2 / 3   and   mu_+- = sqrt(1 + m_+-)."""
    u = np.sqrt(3.0 * np.asarray(lam, dtype=float) + 4.0)
    mm = (u - 2.0) ** 2 / 3.0
    mp = (u + 2.0) ** 2 / 3.0
    return mm, mp, np.sqrt(1.0 + mm), np.sqrt(1.0 + mp)


def part_structure():
    """[C1]-[C3]: the condensate, the mass matrix, the channel masses."""
    log("[C1] the marginal throat condensate")
    # [rho (rho^2 - h^2) w0']' + rho w0 = 0   (b = 1)
    for h in (2.0 / 3.0, 1.0, 0.4):
        rr = np.linspace(1.02 * h, 40.0 * h, 400000)
        w, dw = w0_and_dw0(rr, h)
        flux = rr * (rr**2 - h**2) * dw
        d = np.gradient(flux, rr)
        res = d + rr * w
        sc = np.max(np.abs(rr * w))
        rel = np.max(np.abs(res[200:-200])) / sc
        check(
            f"[C1] w0 solves L_{{-1}} w0 = 0 at h = {h:g}",
            rel < 3e-6,
            f"max |residual|/scale = {rel:.1e}",
        )

    log("[C2] the (H_yy, bhat) mass matrix from the production rows")
    worst_const = 0.0
    worst_zz = 0.0
    worst_d = 0.0
    worst_form = 0.0
    cases = [
        (2 / 3, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
        (0.4, 2.7, 1.0, 1.0),
        (2 / 3, 0.35, 3.1, 1.0),
        (0.9, 1.6, 0.4, 1.7),
        (2 / 3, 1.0, 1.0, 0.6),
    ]
    off_neg = True
    for h, v, cW, b in cases:
        for gam in (0.3, 1.0, 7.2552, 25.0):
            lam = b * gam
            ref = None
            for rho in (1.05 * h, 1.7 * h, 4.0 * h, 30.0 * h):
                M, zzc, dc = mass_matrix(rho, h, v, cW, b, gam)
                scale = max(abs(M[0, 0]), abs(dc[0]))
                worst_zz = max(worst_zz, float(np.max(np.abs(zzc))) / scale)
                dref = -(3 * rho**2 - h**2) / (rho * (rho**2 - h**2))
                worst_d = max(worst_d, float(np.max(np.abs(dc - dref))) / abs(dref))
                if ref is None:
                    ref = M
                else:
                    worst_const = max(
                        worst_const,
                        float(np.max(np.abs(M - ref)))
                        / max(1.0, float(np.max(np.abs(ref)))),
                    )
                pred_diag = lam + 8.0 / 3.0
                pred_off = (16.0 / 9.0) * (3.0 * lam + 4.0)
                worst_form = max(
                    worst_form,
                    abs(M[0, 0] - pred_diag) / pred_diag,
                    abs(M[1, 1] - pred_diag) / pred_diag,
                    abs(M[0, 1] * M[1, 0] - pred_off) / pred_off,
                )
                off_neg = off_neg and M[0, 1].real < 0 and M[1, 0].real < 0
    check(
        "[C2] H_zz columns of the (yy) and (My) rows vanish identically",
        worst_zz < 1e-12,
        f"max |coefficient| / scale = {worst_zz:.1e}",
    )
    check(
        "[C2] the mass entries are rho-independent constants",
        worst_const < 1e-12,
        f"max relative spread over rho = {worst_const:.1e}",
    )
    check(
        "[C2] the first-derivative coefficient is the L_m form",
        worst_d < 1e-12,
        f"max relative deviation = {worst_d:.1e}",
    )
    check(
        "[C2] M11 = M22 = lambda + 8/3 and M12 M21 = (16/9)(3 lambda + 4)",
        worst_form < 1e-12,
        f"max relative deviation = {worst_form:.1e}",
    )
    check("[C2] M12 < 0 and M21 < 0 (real positive symmetrising rescaling)", off_neg)

    log("[C3] channel masses and exponents")
    lam = np.array([0.0, 0.3, 1.0, 7.2552, 25.0, 100.0])
    lam_solve = np.array([1e-3, 0.3, 1.0, 7.2552, 25.0, 100.0])
    mm, mp, mum, mup = channel_masses(lam)
    mms, mps, _, _ = channel_masses(lam_solve)
    worst = 0.0
    for i, la in enumerate(lam_solve):
        M, _, _ = mass_matrix(2.0, 2 / 3, 1.0, 1.0, 1.0, la)
        ev = np.sort(np.linalg.eigvals(M).real)
        worst = max(
            worst, abs(ev[0] - mms[i]) / max(mms[i], 1e-3), abs(ev[1] - mps[i]) / mps[i]
        )
    check(
        "[C3] eigenvalues of M are (sqrt(3 lambda + 4) +- 2)^2/3",
        worst < 1e-10,
        f"max relative |diff| = {worst:.1e}",
    )
    check(
        "[C3] m_+- >= 0, so mu_+- >= 1 everywhere on lambda >= 0",
        bool(np.all(mm >= -1e-14) and np.all(mum >= 1.0 - 1e-14)),
        f"m_-(0) = {mm[0]:.3e}, min mu_- = {mum.min():.6f}",
    )
    print("")
    print("     lambda      m_-        m_+       mu_-      mu_+     Delta_-   Delta_+")
    for i, la in enumerate(lam):
        print(
            f"    {la:8.4f}  {mm[i]:9.5f}  {mp[i]:9.5f}  {mum[i]:8.5f}  "
            f"{mup[i]:8.5f}  {1 + mum[i]:8.5f}  {1 + mup[i]:8.5f}"
        )
    print("")


# ---------------------------------------------------------------------------
# the coupled throat BVP on xi = ln(rho/h)
# ---------------------------------------------------------------------------
FIELDS = ("yy", "zz", "ay")


def _row_arrays(xi, h, v, cW, b, gam):
    """cH[i,j,k], cdH[i,j,k], src[i,k] of the three evolution rows over the grid
    (i = row, j = field, k = node), from the production system."""
    n = xi.size
    cH = np.zeros((3, 3, n), dtype=complex)
    cdH = np.zeros((3, 3, n), dtype=complex)
    src = np.zeros((3, n), dtype=complex)
    for k in range(n):
        if xi[k] <= 0.0:  # horizon node: rows are degenerate, handled separately
            continue
        tr = row_triples(h * np.exp(xi[k]), h, v, cW, b, gam)
        for i, key in enumerate(FIELDS):
            a, d, sq = tr[key]
            cH[i, :, k] = a
            cdH[i, :, k] = d
            src[i, k] = sq
    return cH, cdH, src


def _horizon_row(h, v, cW, b, gam, eps=1e-5):
    """The degenerate horizon row: multiply each evolution row by (rho^2 - h^2)
    and let rho -> h.  Extrapolated in eps from two radii (Richardson)."""
    out = []
    for e in (eps, 2 * eps):
        xi = np.array([np.log1p(e)])
        cH, cdH, src = _row_arrays(xi, h, v, cW, b, gam)
        rho = h * np.exp(xi[0])
        f = rho**2 - h**2
        out.append((cH[:, :, 0] * f, cdH[:, :, 0] * f / rho, src[:, 0] * f))
    a, c = out
    return tuple(2 * x - y for x, y in zip(a, c, strict=True))


def solve_throat(gam, Xi=22.0, N=4000, h=H_DEF, v=V_DEF, cW=CW_DEF, b=B_DEF):
    """Solve the coupled (H_yy, H_zz, a_y) system on the throat.

    In xi = ln(rho/h) the rows Phi'' = cH.Phi + cdH.Phi' + src become
        Phi_xixi = Phi_xi + rho^2 cH.Phi + rho cdH.Phi_xi + rho^2 src.
    Horizon (xi = 0): the degenerate row (rho^2 - h^2) x (row) -> 0.
    Far end (xi = Xi): Dirichlet, every channel decaying at least as rho^{-2}.
    """
    xi = np.linspace(0.0, Xi, N + 1)
    dx = xi[1] - xi[0]
    rho = h * np.exp(xi)
    cH, cdH, src = _row_arrays(xi, h, v, cW, b, gam)
    nn = 3 * (N + 1)
    nl = nu = 8
    AB = np.zeros((nl + nu + 1, nn), dtype=complex)
    rhs = np.zeros(nn, dtype=complex)

    def ix(i, k):
        return 3 * k + i

    def put(row, col, val):
        AB[nu + row - col, col] += val

    for k in range(1, N):
        r2 = rho[k] ** 2
        for i in range(3):
            row = ix(i, k)
            put(row, ix(i, k - 1), 1.0 / dx**2 + 0.5 / dx)
            put(row, ix(i, k), -2.0 / dx**2)
            put(row, ix(i, k + 1), 1.0 / dx**2 - 0.5 / dx)
            for j in range(3):
                put(row, ix(j, k), -r2 * cH[i, j, k])
                put(row, ix(j, k - 1), rho[k] * cdH[i, j, k] * 0.5 / dx)
                put(row, ix(j, k + 1), -rho[k] * cdH[i, j, k] * 0.5 / dx)
            rhs[row] = r2 * src[i, k]
    cHh, cdHh, srch = _horizon_row(h, v, cW, b, gam)
    fwd = np.array([-1.5, 2.0, -0.5]) / dx
    for i in range(3):
        row = ix(i, 0)
        for j in range(3):
            put(row, ix(j, 0), cHh[i, j])
            for q in range(3):
                put(row, ix(j, q), cdHh[i, j] * fwd[q])
        rhs[row] = -srch[i]
    for i in range(3):
        put(ix(i, N), ix(i, N), 1.0)
        rhs[ix(i, N)] = 0.0
    solv = solve_banded((nl, nu), AB, rhs)
    fld = np.array([solv[i::3] for i in range(3)])
    dfld = np.gradient(fld, dx, axis=1, edge_order=2) / rho
    return xi, rho, fld, dfld


def kernel_pieces(gam, Xi=22.0, N=4000, h=H_DEF, v=V_DEF, cW=CW_DEF, b=B_DEF):
    """(C4, X, Pi) of the paper sec. 5.1 and app. C kernel on the throat, in the fixed
    normalisation w0 = (2/pi) sqrt(z) K(1-z).  Integration measure: dr = rho dxi."""
    xi, rho, fld, dfld = solve_throat(gam, Xi, N, h, v, cW, b)
    Hyy, Hzz, ay = fld[0], fld[1], fld[2]
    W0, W0p = w0_and_dw0(rho, h)
    _, U, Up, V, Vp, W, Wp, Bf, G, al = bg_fields(rho, h, v, cW, b, gam)
    c4 = W0**4 * np.exp(-2 * V) * np.exp(W)
    xd = 1j * G * W0**2 * np.exp(-2 * V) * np.exp(W) * ay
    Hrr = np.zeros_like(ay)
    Hxr = np.zeros_like(ay)
    for k in range(1, xi.size):
        tr = row_triples(rho[k], h, v, cW, b, gam)
        for nm, tgt in (("Hrr", Hrr), ("Hxr", Hxr)):
            a, d, sq = tr[nm]
            tgt[k] = (
                a[0] * Hyy[k]
                + a[1] * Hzz[k]
                + a[2] * ay[k]
                + d[0] * dfld[0, k]
                + d[1] * dfld[1, k]
                + d[2] * dfld[2, k]
                + sq
            )
    Hrr[0] = 2 * Hrr[1] - Hrr[2]
    Hxr[0] = 2 * Hxr[1] - Hxr[2]
    pig = np.array(
        [
            RS.pi_grav(
                rho[k],
                U[k],
                Up[k],
                V[k],
                Vp[k],
                W[k],
                Wp[k],
                Bf,
                float(G),
                al,
                W0[k],
                W0p[k],
                Hyy[k],
                Hzz[k],
                ay[k],
                Hrr[k],
                Hxr[k],
            )
            for k in range(xi.size)
        ]
    )
    w = rho  # dr = rho dxi
    C4 = float(np.real(np.trapezoid(c4 * w, xi)))
    X = float(np.real(np.trapezoid(xd * w, xi)))
    Pi = float(np.real(np.trapezoid(pig * w, xi)))
    return C4, X, Pi


def S_full(gam, Xi=20.0, N=3000, **kw):
    """K/C4 on the throat with Richardson in N (the scheme is 2nd order)."""
    a = kernel_pieces(gam, Xi=Xi, N=N, **kw)
    q = kernel_pieces(gam, Xi=Xi, N=2 * N, **kw)
    C4, X, Pi = ((4 * y - x) / 3 for x, y in zip(a, q, strict=True))
    return (C4 - X - 0.5 * Pi) / C4, C4, X, Pi


def abelian_probe(gam, Xi=20.0, N=200000, h=H_DEF):
    """the paper app. C.4 decoupled photon BVP L_lambda bhat = -lambda rho w0^2,
    solved with THIS module's xi discretisation: an anchor for the scheme.
    Returns (C4 - X)/C4, which should be S(gamma)/S(0) of paper app. C.4."""
    xi = np.linspace(0.0, Xi, N + 1)
    dx = xi[1] - xi[0]
    rho = h * np.exp(xi)
    w0, _ = w0_and_dw0(rho, h)
    f = w0**2
    ph = np.expm1(2 * (xi[:-1] + dx / 2))
    wgt = np.exp(2 * xi)
    lo, di, up = np.zeros(N), np.zeros(N + 1), np.zeros(N)
    rhs = np.zeros(N + 1)
    di[1:N] = -(ph[1:N] + ph[0 : N - 1]) / dx**2 - gam * wgt[1:N]
    up[1:N] = ph[1:N] / dx**2
    lo[0 : N - 1] = ph[0 : N - 1] / dx**2
    rhs[1:N] = -gam * wgt[1:N] * f[1:N]
    di[0], up[0], rhs[0] = -1.5 / dx - gam / 2.0, 2.0 / dx, -gam * f[0] / 2.0
    di[N], rhs[N] = 1.0, 0.0
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    Mtx = diags([lo, di, up], [-1, 0, 1], format="lil")
    Mtx[0, 2] = -0.5 / dx
    bh = spsolve(Mtx.tocsr(), rhs)
    num = float(np.trapezoid(wgt * (f - bh) * f, xi))
    den = float(np.trapezoid(wgt * f * f, xi))
    return num, den


def abelian_probe_R(gam, Xi=20.0, N=100000):
    """Richardson pair; returns (S(gamma), S(0)) in the paper app. C.4 normalisation."""
    a, da = abelian_probe(gam, Xi, N)
    c, dc = abelian_probe(gam, Xi, 2 * N)
    return (4 * c - a) / 3.0, (4 * dc - da) / 3.0


# ---------------------------------------------------------------------------
# the second method: shooting from both ends (table 4)
# ---------------------------------------------------------------------------
# S_shoot shares with S_full only the generated production rows (row_triples),
# the condensate w0 and the pairing density RS.pi_grav -- i.e. the equations,
# not the discretisation.  No grid, no far wall, no degenerate horizon row:
#   * horizon: the four regular solutions (three homogeneous, one particular)
#     from a Frobenius series of order HORIZON_ORDER in x = rho - h, whose
#     coefficients come from the Taylor data of (rho^2 - h^2) x (rows) by
#     Chebyshev interpolation on [0, 0.1 h]; integrated outward with DOP853;
#   * far end: the three decaying solutions of the asymptotic constant-
#     coefficient system (companion eigenproblem at Xi; no Dirichlet wall),
#     each integrated inward with its exponential factored out (otherwise
#     roundoff in the exactly-zero couplings, times |Y| ~ 1e12, stalls the
#     integrator); the far particular solution integrated inward in unit
#     segments, projecting out the decaying basis at each segment start
#     (otherwise it grows like e^{3.8 (Xi - xi)} and the superposition
#     cancels ~9 digits);
#   * values and xi-derivatives matched at xi_m; C4, X, Pi by composite
#     Gauss-Legendre on the dense output.
# The rows have a pole just inside the horizon: the determinant of the
# algebraic (H_rr, H_xr) block vanishes where rho^2 - h^2 = -(gamma/4)
# (2 rho - h^2/rho)^2, i.e. at x = rho - h ~ -gamma h/8 at small gamma.  That
# sets the radius of convergence of the horizon series, so the Taylor fit
# interval and the start point scale with gamma (_horizon_scales).  The FD
# solver's degenerate horizon row feels the same pole: it is the less accurate
# of the two at small gamma, and its N-refinement moves it onto S_shoot.
HORIZON_ORDER = 6


def _horizon_scales(gam):
    """(Taylor-fit interval, start point) in units of h: 1/2 and 1/50 of the
    distance gamma/8 to the pole, capped at 0.1 and 5e-3."""
    return min(0.1, gam / 16.0), min(5e-3, gam / 400.0)


def _rows3(rho, gam):
    """(A, D, s, triples) with Phi_rhorho = A Phi + D Phi_rho + s."""
    tr = row_triples(float(rho), gam=gam)
    A = np.array([tr[k][0] for k in FIELDS], dtype=complex)
    D = np.array([tr[k][1] for k in FIELDS], dtype=complex)
    s = np.array([tr[k][2] for k in FIELDS], dtype=complex)
    return A, D, s, tr


def _horizon_taylor(gam, order, h=H_DEF, X=0.1, n=16):
    """Taylor coefficients in x = rho - h, k = 0..order, of f A, f D and f s,
    f = rho^2 - h^2 (each row has a simple pole at the horizon, so these are
    analytic; the nearest singularity is rho = 0).  Chebyshev interpolation at
    n Gauss nodes on [0, X h], then the k-th derivative at the left end,
    T_m^(k)(-1) = (-1)^(m+k) prod_{i<k} (m^2 - i^2)/(2i + 1)."""
    th = np.pi * (np.arange(n) + 0.5) / n
    t = np.cos(th)
    vals = []
    for tk in t:
        x = 0.5 * X * h * (tk + 1.0)
        A, D, s, _ = _rows3(h + x, gam)
        f = x * (2.0 * h + x)
        vals.append(np.concatenate([(A * f).ravel(), (D * f).ravel(), s * f]))
    vals = np.array(vals)
    m = np.arange(n)
    c = (2.0 / n) * np.cos(np.outer(m, th)) @ vals
    c[0] *= 0.5
    out = []
    fact = 1.0
    for k in range(order + 1):
        if k:
            fact *= k
        Tk = (-1.0) ** (m + k) * np.prod(
            [(m**2 - i**2) / (2.0 * i + 1.0) for i in range(k)] or [np.ones(n)], axis=0
        )
        a = (Tk @ c) * (2.0 / (X * h)) ** k / fact
        out.append((a[:9].reshape(3, 3), a[9:18].reshape(3, 3), a[18:]))
    return out


def _frobenius(gam, order=HORIZON_ORDER, h=H_DEF, X=0.1):
    """Coefficients P[n] (n = 0..order) of the four regular horizon solutions
    Phi = sum_n P[n] x^n: three homogeneous (P[0] = e_i) and one particular.
    From f Phi'' = (f A) Phi + (f D) Phi' + f s with f = 2 h x + x^2, order x^k:
      (2h(k+1)k - (k+1) D0) P[k+1] = sum_{j<=k} A_j P[k-j]
          + sum_{1<=j<=k} (k-j+1) D_j P[k-j+1] + S_k - k(k-1) P[k]."""
    tay = _horizon_taylor(gam, order, h, X)
    Aj = [a for a, _, _ in tay]
    Dj = [d for _, d, _ in tay]
    Sj = [s for _, _, s in tay]
    P = np.zeros((order + 1, 3, 4), dtype=complex)  # [power, field, solution]
    P[0, :, :3] = np.eye(3)
    src = np.array([0.0, 0.0, 0.0, 1.0])
    for k in range(order):
        rhs = np.outer(Sj[k], src) - k * (k - 1) * P[k]
        for j in range(k + 1):
            rhs += Aj[j] @ P[k - j]
        for j in range(1, k + 1):
            rhs += (k - j + 1) * Dj[j] @ P[k - j + 1]
        lhs = 2.0 * h * (k + 1) * k * np.eye(3) - (k + 1) * Dj[0]
        P[k + 1] = np.linalg.solve(lhs, rhs)
    return P


def _series(P, x, h=H_DEF):
    """(Phi, Phi_xi) of the four horizon solutions at x = rho - h: arrays of
    shape (3, 4, len(x)); Phi_xi = rho Phi_rho."""
    x = np.atleast_1d(x)
    n = np.arange(P.shape[0])
    xp = x[None, :] ** n[:, None]
    dxp = np.where(
        n[:, None] > 0, n[:, None] * x[None, :] ** np.maximum(n - 1, 0)[:, None], 0.0
    )
    Phi = np.einsum("nfs,nx->fsx", P, xp)
    dPhi = np.einsum("nfs,nx->fsx", P, dxp) * (h + x)[None, None, :]
    return Phi, dPhi


def _make_rhs(gam, src_mask):
    """y = [Phi (3), Phi_xi (3)] per solution, stacked; xi = ln(rho/h)."""
    nsol = len(src_mask)
    src_mask = np.asarray(src_mask, dtype=float)

    def rhs(xi, y):
        rho = H_DEF * np.exp(xi)
        A, D, s, _ = _rows3(rho, gam)
        Y = y.reshape(nsol, 6)
        out = np.empty_like(Y)
        Phi, Px = Y[:, :3], Y[:, 3:]
        out[:, :3] = Px
        # Phi_xixi = Phi_xi + rho^2 A Phi + rho D Phi_xi + rho^2 s
        out[:, 3:] = Px + rho**2 * Phi @ A.T + rho * Px @ D.T
        out[:, 3:] += np.outer(src_mask, rho**2 * s)
        return out.ravel()

    return rhs


def S_shoot(
    gam,
    Xi=14.0,
    xim=2.0,
    x0rel=None,
    order=HORIZON_ORDER,
    rtol=1e-10,
    atol=1e-11,
    nGL=8,
    npan=(60, 120),
):
    """K/C4 on the throat by shooting from both ends (see the block comment
    above).  Returns (S, C4, X, Pi) like S_full."""
    from scipy.integrate import solve_ivp

    h = H_DEF
    ivp = dict(method="DOP853", rtol=rtol, atol=atol, dense_output=True)
    # ---- horizon side --------------------------------------------------------
    Xfit, x0def = _horizon_scales(gam)
    x0rel = x0def if x0rel is None else x0rel
    P = _frobenius(gam, order, X=Xfit)
    x0 = x0rel * h
    xi0 = np.log1p(x0rel)
    Phi0, dPhi0 = _series(P, np.array([x0]))
    y0 = np.concatenate([np.r_[Phi0[:, k, 0], dPhi0[:, k, 0]] for k in range(4)])
    solh = solve_ivp(_make_rhs(gam, [0.0, 0.0, 0.0, 1.0]), [xi0, xim], y0, **ivp)
    if solh.status != 0:
        raise RuntimeError(f"S_shoot horizon side: {solh.message}")
    # ---- far side: the decaying basis of the asymptotic system ----------------
    A, D, _, _ = _rows3(h * np.exp(Xi), gam)
    rhoX = h * np.exp(Xi)
    Ainf, Dinf = rhoX**2 * A, rhoX * D
    # kappa^2 v = kappa (1 + Dinf) v + Ainf v  (companion linearisation)
    Cmp = np.block([[np.zeros((3, 3)), np.eye(3)], [Ainf, np.eye(3) + Dinf]])
    kap, vec = np.linalg.eig(Cmp)
    order_k = np.argsort(kap.real)[:3]  # the three most negative: decaying
    base = _make_rhs(gam, [0.0])
    fsols = []
    for j in order_k:
        v = vec[:3, j] / np.max(np.abs(vec[:3, j]))
        kj = kap[j]

        def rhs_s(xi, y, kj=kj):
            return base(xi, y) - kj * y

        sj = solve_ivp(rhs_s, [Xi, xim], np.r_[v, kj * v], first_step=1e-3, **ivp)
        if sj.status != 0:
            raise RuntimeError(f"S_shoot far basis: {sj.message}")
        fsols.append((sj, kj))

    def Fat(x):
        return np.column_stack([sj.sol(x) * np.exp(kj * (x - xim)) for sj, kj in fsols])

    # the far particular solution, inward in unit segments with the decaying
    # basis projected out at each segment start.  On segment k the field is
    # P_k + F (b - sum_{i > k} c_i); cum[k] stores sum_{i > k} c_i.
    edges = np.r_[np.arange(Xi, xim, -1.0), xim]
    segs, cs = [], []
    yp = np.zeros(6, complex)
    prhs = _make_rhs(gam, [1.0])
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if segs:
            Fa = Fat(a)
            c, *_ = np.linalg.lstsq(Fa, yp, rcond=None)
            yp = yp - Fa @ c
            cs.append(c)
        sp_ = solve_ivp(prhs, [a, b], yp, first_step=1e-3, **ivp)
        if sp_.status != 0:
            raise RuntimeError(f"S_shoot far particular: {sp_.message}")
        segs.append((b, a, sp_))
        yp = sp_.y[:, -1]
    cum = [np.zeros(3, complex) for _ in segs]
    for k in range(len(segs) - 2, -1, -1):
        cum[k] = cum[k + 1] + cs[k]

    def far(xi, shift=False):
        """Stacked far basis (3 decaying, 1 particular), (4, 6, len(xi))."""
        xi = np.atleast_1d(xi)
        parts = [sj.sol(xi) * np.exp(kj * (xi - xim)) for sj, kj in fsols]
        Pp = np.zeros((6, xi.size), complex)
        for k, (lo_, hi_, sp_) in enumerate(segs):
            last = k == len(segs) - 1
            msk = (xi >= lo_) & (xi <= hi_) if last else (xi > lo_) & (xi <= hi_)
            if msk.any():
                Pp[:, msk] = sp_.sol(xi[msk])
                if shift:  # express in the innermost segment's particular solution
                    Pp[:, msk] -= np.stack(parts, axis=-1)[:, msk, :] @ cum[k]
        return np.stack(parts + [Pp], axis=0)

    # ---- match at xi_m ----------------------------------------------------------
    Yh = solh.y[:, -1].reshape(4, 6)
    Yf = far(np.array([xim]))[:, :, 0]
    Mat = np.column_stack([Yh[0], Yh[1], Yh[2], -Yf[0], -Yf[1], -Yf[2]])
    coef = np.linalg.solve(Mat, Yf[3] - Yh[3])
    ch, cf = np.r_[coef[:3], 1.0], np.r_[coef[3:], 1.0]

    def field(xi):
        out = np.empty((6, xi.size), complex)
        tay = xi < xi0
        if tay.any():
            Ph, dPh = _series(P, h * np.expm1(xi[tay]))
            out[:3, tay] = np.einsum("fsx,s->fx", Ph, ch)
            out[3:, tay] = np.einsum("fsx,s->fx", dPh, ch)
        lo = (xi >= xi0) & (xi <= xim)
        if lo.any():
            out[:, lo] = np.einsum("k,kif->if", ch, solh.sol(xi[lo]).reshape(4, 6, -1))
        hi = xi > xim
        if hi.any():
            out[:, hi] = np.einsum("k,kif->if", cf, far(xi[hi], shift=True))
        return out

    # ---- quadrature: composite GL, graded (u^2 map) on [0, xi_m] --------------
    t, w = np.polynomial.legendre.leggauss(nGL)

    def nodes(a, b, n, grade=False):
        e = a + (b - a) * np.linspace(0, 1, n + 1) ** (2 if grade else 1)
        X_ = np.concatenate(
            [0.5 * (q - p) * (t + 1) + p for p, q in zip(e[:-1], e[1:], strict=True)]
        )
        W_ = np.concatenate(
            [0.5 * (q - p) * w for p, q in zip(e[:-1], e[1:], strict=True)]
        )
        return X_, W_

    x1, w1 = nodes(0.0, xim, npan[0], grade=True)
    x2, w2 = nodes(xim, Xi, npan[1])
    xs, ws = np.r_[x1, x2], np.r_[w1, w2]
    F = field(xs)
    rho = h * np.exp(xs)
    W0, W0p = w0_and_dw0(rho, h)
    _, U, Up, V, Vp, W, Wp, Bf, G, al = bg_fields(rho, gam=gam)
    Hyy, Hzz, ay = F[0], F[1], F[2]
    dPhi = F[3:] / rho  # d/drho
    c4 = W0**4 * np.exp(-2 * V) * np.exp(W)
    xd = 1j * G * W0**2 * np.exp(-2 * V) * np.exp(W) * ay
    pig = np.empty(xs.size)
    for k in range(xs.size):
        tr = row_triples(rho[k], gam=gam)
        vec = np.array([Hyy[k], Hzz[k], ay[k]])
        al_r, al_x = tr["Hrr"], tr["Hxr"]
        Hrr = np.dot(al_r[0], vec) + np.dot(al_r[1], dPhi[:, k]) + al_r[2]
        Hxr = np.dot(al_x[0], vec) + np.dot(al_x[1], dPhi[:, k]) + al_x[2]
        pig[k] = np.real(
            RS.pi_grav(
                rho[k],
                U[k],
                Up[k],
                V[k],
                Vp[k],
                W[k],
                Wp[k],
                Bf,
                float(G),
                al,
                W0[k],
                W0p[k],
                Hyy[k],
                Hzz[k],
                ay[k],
                Hrr,
                Hxr,
            )
        )
    C4 = float(np.real(np.sum(ws * c4 * rho)))
    X = float(np.real(np.sum(ws * xd * rho)))
    Pi = float(np.sum(ws * pig * rho))
    return (C4 - X - 0.5 * Pi) / C4, C4, X, Pi


# ---------------------------------------------------------------------------
# the kernel curve, selection, and the beta-ladder anchor
# ---------------------------------------------------------------------------
# stability of S_full under refinement of the grid and of the far cutoff
# (paper app. D.5 quotes 2e-6)
TOL_REFINE = 2e-6

GAM_GRID = np.concatenate(
    [np.linspace(0.02, 1.0, 15), np.linspace(1.2, 4.0, 10), np.geomspace(4.5, 60.0, 16)]
)


def kernel_curve(Xi=20.0, N=3000, grid=GAM_GRID):
    S = np.empty(grid.size)
    rows = []
    for i, g in enumerate(grid):
        s, c4, xx, pp = S_full(g, Xi=Xi, N=N)
        S[i] = s
        rows.append((c4, xx, pp))
    arr = np.array(rows)
    return S, arr[:, 0], arr[:, 1], arr[:, 2]


def _interp(grid, S):
    from scipy.interpolate import PchipInterpolator

    p = PchipInterpolator(grid, S)
    gmax, tail = grid[-1], float(PchipInterpolator(grid, S)(grid[-1]))

    def f(x):
        x = np.asarray(x, float)
        return np.where(
            x < gmax, p(np.clip(x, grid[0], gmax)), tail * gmax / np.maximum(x, 1e-12)
        )

    return f


def cell_energy(Sfun, tau1, tau2, kmax=200.0, R=25):
    """the paper sec. 5.2 one-flux-quantum cell functional at modulus tau."""
    m, n = np.meshgrid(np.arange(-R, R + 1), np.arange(-R, R + 1), indexing="ij")
    q = (m + n * tau1) ** 2 + (n * tau2) ** 2
    g = 2 * np.pi * q / tau2
    g = g[(g > 0) & (g < kmax)]
    return 0.5 * float(np.sum(np.exp(-0.5 * g) * Sfun(g)))


def part_kernel(Xi=20.0, N=3000):
    """[C4]-[C8]: the throat kernel, its sign structure and the selection."""
    log("[C4] abelian anchor: this scheme against the paper app. C.4 S(gamma)")
    worst = worst0 = 0.0
    for g in (0.5, 2.0, 7.2552, 20.0):
        mine, S0 = abelian_probe_R(g, 26.0, 50000)
        ref = TL.S_richardson(g, 26.0, 100000)
        worst = max(worst, abs(mine - ref) / abs(ref))
        worst0 = abs(S0 - TL.S0_EXACT) / TL.S0_EXACT
    check(
        "[C4] the decoupled photon BVP reproduces the paper app. C.4 S(gamma)",
        worst < 1e-6,
        f"max relative deviation = {worst:.1e}",
    )
    check(
        "[C4] its S(0) matches the closed form 1.89804004507405",
        worst0 < 1e-6,
        f"S(0) = {S0:.10f} (rel {worst0:.1e})",
    )

    log("[C5]/[C6] the full throat kernel S_full(gamma) = K/C4")
    S, C4, X, Pi = kernel_curve(Xi=Xi, N=N)
    # refinement tests at fixed gamma points, against direct solves there
    gtest = np.array([0.3, 0.9, 4.0, 7.2552, 20.0])
    ref, _, _, _ = kernel_curve(Xi=Xi, N=N, grid=gtest)
    # weak: both cutoffs are in the roundoff region of the sources (docstring [C6])
    S2, _, _, _ = kernel_curve(Xi=26.0, N=N, grid=gtest)
    dXi = float(np.max(np.abs(S2 - ref)))
    check(
        "[C6] convergence in the far cutoff Xi (20 vs 26)",
        dXi < TOL_REFINE,
        f"max |difference| = {dXi:.1e}",
    )
    Scoarse, _, _, _ = kernel_curve(Xi=Xi, N=N // 2, grid=gtest)
    dN = float(np.max(np.abs(Scoarse - ref)))
    check(
        "[C5] two resolutions after Richardson agree",
        dN < TOL_REFINE,
        f"max |difference| = {dN:.1e}",
    )

    log("[C10] second method: shooting from both ends against the FD solver")
    # S_shoot shares the rows, w0 and pi_grav with S_full, not the
    # discretisation.  The FD at the production N = 3000 is the less accurate
    # of the two; at small gamma its N-refinement moves it onto S_shoot.
    gC10 = np.array([0.02, 0.3, 0.8465, 4.0, 7.2552, 20.0])
    S_sh = np.array([S_shoot(g)[0] for g in gC10])
    S_fd = np.array([S_full(g, Xi=Xi, N=N)[0] for g in gC10])
    dev = S_sh - S_fd
    for g, a, b, d_ in zip(gC10, S_sh, S_fd, dev, strict=True):
        log(
            f"    gamma = {g:7.4f}:  S_shoot = {a:+.12f}  S_FD = {b:+.12f}  diff {d_:+.1e}"
        )
    S_fd2 = np.array([S_full(g, Xi=Xi, N=2 * N)[0] for g in gC10[:2]])
    closer = bool(np.all(np.abs(S_sh[:2] - S_fd2) < np.abs(dev[:2])))
    check(
        "[C10] S_shoot vs S_FD: < 1e-6 at gamma = 0.02, < 1e-7 for gamma >= 0.3, "
        "< 3e-8 at the lattice shells (gamma >= 7.2552)",
        abs(dev[0]) < 1e-6
        and np.max(np.abs(dev[1:])) < 1e-7
        and np.max(np.abs(dev[4:])) < 3e-8,
        f"|diff| = {abs(dev[0]):.1e} (0.02), max {np.max(np.abs(dev[1:])):.1e} (>= 0.3), "
        f"max {np.max(np.abs(dev[4:])):.1e} (>= 7.2552)",
    )
    check(
        "[C10] the FD error is the FD's: doubling N moves it towards S_shoot (gamma = 0.02, 0.3)",
        closer,
        "N = 6000: diff "
        + ", ".join(f"{a - b:+.1e}" for a, b in zip(S_sh[:2], S_fd2, strict=True)),
    )

    log("[C7] complete monotonicity")
    fS = _interp(GAM_GRID, S)
    gm = np.linspace(0.02, 40.0, 4000)
    vals = fS(gm)
    neg = gm[vals < 0]
    from scipy.optimize import brentq

    # gamma_0 by direct roots of both solvers; the PCHIP root of the
    # 41-node interpolant is kept as an interpolation check only
    g0 = brentq(lambda x: S_shoot(x)[0], 0.8, 0.9, xtol=1e-12)
    g0_fd = brentq(lambda x: S_full(x, Xi=Xi, N=N)[0], 0.8, 0.9, xtol=1e-12)
    g0_i = brentq(lambda x: float(fS(x)), 0.5, 1.5)
    check(
        "[C7] S_full is NEGATIVE at long wavelength => NOT completely monotone",
        bool(neg.size) and S_sh[0] < 0 and float(fS(0.05)) < 0,
        f"S_shoot(0.02) = {S_sh[0]:+.6f}, S_full(0.05) = {float(fS(0.05)):+.5f}",
    )
    check(
        "[C10] sign change gamma_0: direct roots by shooting and by FD agree to 1e-7",
        abs(g0 - g0_fd) < 1e-7,
        f"gamma_0 = {g0:.9f} (shooting), {g0_fd:.9f} (FD); "
        f"PCHIP on the 41 nodes {g0_i:.9f} ({g0_i - g0:+.1e})",
    )
    d1 = np.gradient(vals, gm)
    check(
        "[C7] and S_full is non-monotone: an interior MAXIMUM in gamma",
        bool(np.any(d1 > 0) and np.any(d1 < 0)),
        f"argmax gamma = {gm[np.argmax(vals)]:.3f}, S_max = {vals.max():.6f}",
    )

    log("[C8] lattice selection with the full kernel")
    # Direct solves at the exact shells gamma_mn of the two lattices (four
    # shells each above the 1e-10 weight floor), by both solvers.  The PCHIP
    # interpolant on GAM_GRID is an interpolation check, not a second method:
    # its interpolation error moves the relative gap by ~4e-5.
    Ct = TL.lattice_sum("tri", lambda g: S_full(g, Xi=Xi, N=N)[0])
    Cs = TL.lattice_sum("sq", lambda g: S_full(g, Xi=Xi, N=N)[0])
    Ct_s = TL.lattice_sum("tri", lambda g: S_shoot(g)[0])
    Cs_s = TL.lattice_sum("sq", lambda g: S_shoot(g)[0])
    Ct_i = TL.lattice_sum("tri", lambda g: float(fS(g)))
    Cs_i = TL.lattice_sum("sq", lambda g: float(fS(g)))
    gap = (Cs - Ct) / abs(Ct)
    gap_s = (Cs_s - Ct_s) / abs(Ct_s)
    gap_i = (Cs_i - Ct_i) / abs(Ct_i)
    check(
        "[C8] triangular still beats square (exact shells, FD), relative gap +0.1766",
        Cs > Ct and abs(gap - 0.176562) < 5e-6,
        f"C_tri = {Ct:.9f}, C_sq = {Cs:.9f}, gap = {Cs - Ct:+.9f} "
        f"(relative {gap:+.7f})",
    )
    check(
        "[C10] the same by shooting: relative gap agrees with the FD to 1e-6",
        Cs_s > Ct_s and abs(gap_s - gap) < 1e-6,
        f"C_tri = {Ct_s:.9f}, C_sq = {Cs_s:.9f}, relative gap {gap_s:+.7f} "
        f"({gap_s - gap:+.1e})",
    )
    check(
        "[C8] interpolation check: the PCHIP interpolant on the 41-node grid",
        abs(gap_i - gap) < 1e-4,
        f"C_tri = {Ct_i:.9f}, C_sq = {Cs_i:.9f}, relative gap {gap_i:+.6f} "
        f"(vs exact shells {gap_i - gap:+.1e})",
    )
    grid_t1 = np.linspace(0.0, 0.5, 26)
    grid_t2 = np.geomspace(0.55, 40.0, 140)
    best = (np.inf, 0.0, 0.0)
    loc = (np.inf, 0.0, 0.0)
    for t1 in grid_t1:
        for t2 in grid_t2:
            C = cell_energy(fS, t1, t2)
            if C < best[0]:
                best = (C, t1, t2)
            if t2 < 2.0 and C < loc[0]:
                loc = (C, t1, t2)
    check(
        "[C8] triangular is still the minimum among COMPACT cells (tau2 < 2)",
        abs(loc[1] - 0.5) < 0.03 and abs(loc[2] - np.sqrt(3) / 2) < 0.03,
        f"argmin (tau1, tau2) = ({loc[1]:.3f}, {loc[2]:.3f}) vs "
        f"(0.5, {np.sqrt(3) / 2:.3f})",
    )
    check(
        "[C8] but the functional is UNBOUNDED toward the stripe edge",
        best[2] > 20.0 and best[0] < 0.0,
        f"scan argmin (tau1, tau2) = ({best[1]:.3f}, {best[2]:.2f}), "
        f"C = {best[0]:+.6f} at the tau2 edge",
    )
    return dict(
        gamma=GAM_GRID,
        S=S,
        C4=C4,
        X=X,
        Pi=Pi,
        gamma0=g0,
        gamma0_fd=g0_fd,
        gamma0_pchip=g0_i,
        C_tri=Ct,
        C_sq=Cs,
        C_tri_shoot=Ct_s,
        C_sq_shoot=Cs_s,
        gamma_C10=gC10,
        S_shoot_C10=S_sh,
        S_fd_C10=S_fd,
        loc=loc,
        best=best,
    )


def part_ladder(res, gams=(0.3, 0.9, 7.5, 20.0)):
    """[C9] the decisive anchor: continue the PRODUCTION brane kernel up the
    beta ladder at fixed gamma and watch it converge onto the throat values."""
    import contextlib

    from backreaction.numerics import bc_alpha as BCA
    from backreaction.numerics import dk_background as DK
    from backreaction.numerics import dk_kernel as KER
    from backreaction.numerics import dk_throat as TH

    @contextlib.contextmanager
    def pinned(bg):
        orig = KER.dk.solve_colloc
        KER.dk.solve_colloc = lambda *a, **k: bg
        try:
            yield
        finally:
            KER.dk.solve_colloc = orig

    ladder = [
        (1.0, 64),
        (5.0, 64),
        (20.0, 80),
        (45.0, 96),
        (1e2, 112),
        (3e2, 128),
        (1e3, 144),
        (3e3, 160),
        (1e4, 176),
        (3e4, 192),
        (1e5, 208),
        (3e5, 224),
        (1e6, 240),
        (3e6, 256),
        (1e7, 288),
        (3e7, 304),
        (1e8, 320),
    ]
    keep = {1e4, 1e5, 1e6, 1e7, 1e8}
    fS = _interp(res["gamma"], res["S"])
    target = np.array([float(fS(g)) for g in gams])
    log("[C9] the beta ladder: the brane kernel against the throat limit")
    log("       beta      nu^2 " + "".join(f"   K/C4({g:g})" for g in gams))
    bg, rows = None, []
    for beta, N in ladder:
        for Ntry in (N, N + 48, N + 96):
            try:
                bg = DK.solve_colloc(
                    beta,
                    N=Ntry,
                    tol=1e-10,
                    maxit=200,
                    guess=TH._regrid(bg, Ntry) if bg else None,
                    require=True,
                )
                break
            except DK.NotConverged:
                continue
        else:
            raise DK.NotConverged(f"ladder background at beta = {beta:g}")
        if beta not in keep:
            continue
        vals, w0 = BCA.bc_spectral_grid(bg, n_keep=1)
        Bc = float(vals[0])
        fh = bg.fields_r(np.array([1.0 + 1e-9]))
        nu2 = float(Bc * np.exp(-2 * fh["V"][0]) / 3.0) - 1.0
        with pinned(bg):
            br = KER.Brane(beta, Bc, w0, N=bg.N)
        kk = np.array([br.kernel(g * Bc)["K"] / br.kernel(g * Bc)["C4"] for g in gams])
        rows.append((beta, nu2, kk))
        log(f"    {beta:9.1e}  {nu2:6.4f} " + "".join(f"  {x:+11.6f}" for x in kk))
    log("       throat       -- " + "".join(f"  {x:+11.6f}" for x in target))
    devs = np.array([np.abs(kk - target) for _, _, kk in rows])
    shrinking = bool(np.all(devs[-1] < devs[0]) and np.all(np.diff(devs, axis=0) < 0))
    last = devs[-1] / np.array([abs(t) for t in target])
    check(
        "[C9] the brane kernel converges monotonically onto the throat kernel",
        shrinking,
        f"max |deviation| at beta = 1e8: {devs[-1].max():.4f} absolute, "
        f"{last.max():.3f} relative (nu^2 = {rows[-1][1]:.4f})",
    )
    check(
        "[C9] and it changes sign at small gamma on the way up",
        bool(rows[-1][2][0] < 0),
        f"K/C4(gamma = {gams[0]:g}) = {rows[0][2][0]:+.5f} at beta = 1e4 -> "
        f"{rows[-1][2][0]:+.5f} at beta = 1e8",
    )
    return rows


def figure(res, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g, S = res["gamma"], res["S"]
    fS = _interp(g, S)
    gg = np.geomspace(0.02, 60.0, 600)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].axhline(0.0, color="0.7", lw=0.8)
    ax[0].semilogx(gg, fS(gg), "-", label=r"full $K/C_4$")
    ab = np.array([TL.S_richardson(x, 20.0, 20000) for x in gg[::12]])
    ab /= TL.S_richardson(1e-12, 20.0, 20000)
    ax[0].semilogx(gg[::12], ab, "--", label="abelian only (paper app. C.4)")
    ax[0].axvline(res["gamma0"], color="C3", lw=0.8)
    ax[0].axvline(4 * np.pi / np.sqrt(3), color="0.5", lw=0.8, ls=":")
    ax[0].set_xlabel(r"$\gamma = G^2/B$")
    ax[0].set_ylabel(r"$\mathcal{S}(\gamma)$")
    ax[0].set_title(
        r"throat kernel at $\alpha_\star$: sign change at "
        rf"$\gamma_0 = {res['gamma0']:.3f}$"
    )
    ax[0].legend(fontsize=8)
    lam = np.geomspace(1e-3, 100.0, 400)
    mm, mp, mum, mup = channel_masses(lam)
    ax[1].loglog(lam, 1 + mum, label=r"$\Delta_-$")
    ax[1].loglog(lam, 1 + mup, label=r"$\Delta_+$")
    ax[1].loglog(lam, 1 + np.sqrt(1 + lam), ":", label=r"$\Delta_{\rm photon}$ (probe)")
    ax[1].set_xlabel(r"$\lambda = \gamma b$")
    ax[1].set_ylabel(r"$\Delta = 1 + \mu$")
    ax[1].set_title("AdS$_3$ channel dimensions")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _previous_ladder(path, keys):
    """The ladder arrays of an existing cache, or empty arrays."""
    try:
        with np.load(path) as old:
            return tuple(old[k] for k in keys)
    except (OSError, KeyError):
        return tuple(np.zeros(0) for _ in keys)


def main():
    ladder = "--ladder" in sys.argv
    part_structure()
    res = part_kernel()
    rows = part_ladder(res) if ladder else None
    if rows:
        lad_beta = np.array([r[0] for r in rows])
        lad_K = np.array([r[2] for r in rows])
    else:
        # without --ladder, keep the ladder arrays of the committed cache
        lad_beta, lad_K = _previous_ladder(
            paths.data("dk_throat_channels.npz"), ("ladder_beta", "ladder_K")
        )
    nfail = sum(1 for val in CHECKS.values() if not val)
    if nfail:
        for k, val in CHECKS.items():
            if not val:
                print(f"  FAILED: {k}")
        print("cache not written: a check failed")
        raise SystemExit(1)
    np.savez(
        paths.data("dk_throat_channels.npz"),
        gamma=res["gamma"],
        S=res["S"],
        C4=res["C4"],
        X=res["X"],
        Pi=res["Pi"],
        gamma0=res["gamma0"],
        gamma0_fd=res["gamma0_fd"],
        gamma0_pchip=res["gamma0_pchip"],
        C_tri=res["C_tri"],
        C_sq=res["C_sq"],
        C_tri_shoot=res["C_tri_shoot"],
        C_sq_shoot=res["C_sq_shoot"],
        gamma_C10=res["gamma_C10"],
        S_shoot_C10=res["S_shoot_C10"],
        S_fd_C10=res["S_fd_C10"],
        ladder_beta=lad_beta,
        ladder_K=lad_K,
        readme=(
            "The full quartic kernel (paper sec. 5.1, app. C.4) on the "
            "AdS3 x R^2 throat at alpha_* = 2/3.  S = K/C4 at fixed gamma; the "
            "metric sector triangularises into two AdS3 scalars of mass "
            "m_+- = (sqrt(3 lambda + 4) +- 2)^2/3."
        ),
    )
    log(f"saved {paths.data('dk_throat_channels.npz')}")
    figure(res, paths.figure("dk_throat_channels.png"))
    log(f"saved {paths.figure('dk_throat_channels.png')}")
    print("")
    print("SUMMARY")
    print(
        f"  sign change     :  gamma_0 = {res['gamma0']:.4f}  (K < 0 below it; "
        f"{res['gamma0']:.9f} shooting, {res['gamma0_fd']:.9f} FD)"
    )
    print(
        f"  triangular gap  :  C_sq - C_tri = {res['C_sq'] - res['C_tri']:+.8f}  "
        f"(relative {(res['C_sq'] - res['C_tri']) / abs(res['C_tri']):+.4f})"
    )
    print("  complete monotonicity FAILS at alpha_*, and not marginally: the")
    print("  kernel itself changes sign, so the Montgomery route stays broken.")
    print(f"  {len(CHECKS)}/{len(CHECKS)} checks passed")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
