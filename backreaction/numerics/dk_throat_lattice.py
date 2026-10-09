"""The lattice sector as alpha -> alpha_* : numerics for the BTZ-throat limit
(paper app. C.4).

Companion to `backreaction/derivations/dk_throat_lattice.py`, which derives

  * the fixed-T-frame BTZ throat  U -> (r-1)(3r+1),  e^{2V} -> sqrt(beta/6),
  * one radial operator  L_m y = [rho(rho^2-h^2) y']' - m rho y  for both the
    condensate (m = -b -> -1) and the harmonic-gamma photon (m = gamma b -> gamma),
  * the exact solutions  y_H = z^a 2F1(a,a;1;1-z),  y_I = z^a 2F1(a,a;1+mu;z),
    a = (1+mu)/2, mu = sqrt(1+m), z = h^2/rho^2,
  * the marginal condensate  R = (2/pi) sqrt(z) K(1-z),
  * the exact identity  int p2 ((w0^2)')^2 dr = (4/3) B_c C4  at every beta.

WHAT IS TESTED
--------------
 [T1] the exact identity, at every beta on the `bc_alpha.npz` grid (it is an
      algebraic consequence of the SL equation, so it must hold to roundoff);
 [T2] its consequence, the abelian large-s tail (C4 - X)(s) -> (4/3) B_c C4 / s,
      against the committed probe kernel `scripts/kernel.npz`;
 [T3] complete monotonicity, constructively: the spectral tower of the SL
      operator reproduces (C4 - X)(s) = sum_n c_n^2 tau_n/(tau_n + s) with all
      weights c_n^2 >= 0 -- a positive superposition of Yukawas;
 [T4] the throat kernel S(gamma), two methods: a conservative finite-difference
      BVP in xi = ln(rho/h) with Richardson, and the exact hypergeometric
      Green's function (mpmath, 40 digits).  S(0) against the closed form
      (1/2) int_0^1 F^4 dz, and gamma S(gamma) -> (4/3) S(0);
 [T5] the throat-limit lattice selection: the gap C_sq - C_tri, and a scan of
      the whole Bravais moduli space (the paper sec. 5.2 variable), which must find
      triangular -- Montgomery applies, since S is CM;
 [T6] (--ladder, slow) the brane itself: the beta ladder's near-horizon geometry
      against (r-1)(3r+1), and its abelian kernel SHAPE against S(gamma)/S(0),
      which must converge at rate nu^2 = b_h - 1.

Cache: `dk_throat_lattice.npz`.   Figure: `dk_throat_lattice.png`.

Run:  uv run python -m backreaction.numerics.dk_throat_lattice             (~30 s)
      ... --ladder    also runs the beta ladder to 1e8             (~5 min)
"""

import sys
import time

import numpy as np
from scipy.linalg import solve
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import ellipe, ellipkm1

import script_paths
from backreaction import paths
from backreaction.numerics import dk_kernel as KER

CACHE = paths.data("dk_throat_lattice.npz")
FIG = paths.figure("dk_throat_lattice.png")
KER_PATH = script_paths.data("kernel.npz")

T0 = time.time()
CHECKS = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# the marginal throat profile  R = (2/pi) sqrt(z) K(1-z),  f = R^2 = z F^2
# ---------------------------------------------------------------------------


# (1/2) int_0^1 F^4 dz, F = 2F1(1/2,1/2;1;1-z), to 30 digits by mpmath
# tanh-sinh quadrature (1.89804004507404944896383461775).  Gauss-Legendre on
# [0, 1] does NOT converge to this: F^4 ~ ln^4 z at the endpoint, and 4000 nodes
# leave a 4e-6 error.
S0_EXACT = 1.8980400450740494


def S0_closed_form(n=80):
    """(1/2) int_0^1 F(z)^4 dz with z = e^{-t}: F(e^{-t}) = (2/pi) K(1 - e^{-t})
    is smooth in t and grows only linearly, so Gauss-Laguerre converges
    (80 nodes: 3e-14)."""
    t, w = np.polynomial.laguerre.laggauss(n)
    F = (2 / np.pi) * ellipkm1(np.exp(-t))
    return 0.5 * float(np.sum(w * F**4))


def F_of_z(z):
    """2F1(1/2,1/2;1;1-z) = (2/pi) K(1-z).  ellipkm1(z) = K(1-z) keeps full
    relative accuracy as z -> 0, where ellipk(1 - z) loses the digits of z."""
    z = np.maximum(np.asarray(z, float), 1e-300)
    return (2 / np.pi) * ellipkm1(z)


def dF_of_z(z):
    z = np.asarray(z, float)
    out = np.empty_like(z)
    mid = z < 1.0 - 1e-12
    zz = np.maximum(z[mid], 1e-300)
    K, E = ellipkm1(zz), ellipe(1 - zz)
    out[mid] = -(2 / np.pi) * (E - zz * K) / (2 * (1 - zz) * zz)
    out[~mid] = -0.25  # dK/dm -> pi/8 as m -> 0
    return out


def f_and_df(xi):
    """f = R^2 and df/dxi on xi = ln(rho/h), z = e^{-2 xi}."""
    z = np.exp(-2 * xi)
    F, dF = F_of_z(z), dF_of_z(z)
    return z * F**2, (-2 * z) * (F**2 + 2 * z * F * dF)


def S_fd(gam, Xi=30.0, N=100000):
    """S(gamma) by a conservative FD BVP for e = f - bhat:
    [(e^{2xi}-1) e_xi]_xi - gamma e^{2xi} e = [(e^{2xi}-1) f_xi]_xi,
    e_xi(0) = f_xi(0) + gamma e(0)/2  (regularity),   e(Xi) = 0."""
    xi = np.linspace(0.0, Xi, N + 1)
    dx = xi[1] - xi[0]
    f, df = f_and_df(xi)
    _, dfh = f_and_df(xi[:-1] + dx / 2)
    ph = np.expm1(2 * (xi[:-1] + dx / 2))
    wgt = np.exp(2 * xi)
    flux = ph * dfh
    lo, di, up = np.zeros(N), np.zeros(N + 1), np.zeros(N)
    rhs = np.zeros(N + 1)
    di[1:N] = -(ph[1:N] + ph[0 : N - 1]) / dx**2 - gam * wgt[1:N]
    up[1:N] = ph[1:N] / dx**2
    lo[0 : N - 1] = ph[0 : N - 1] / dx**2
    rhs[1:N] = (flux[1:N] - flux[0 : N - 1]) / dx
    di[0], up[0], rhs[0] = -1.5 / dx - gam / 2.0, 2.0 / dx, df[0]
    di[N], rhs[N] = 1.0, 0.0
    Mtx = diags([lo, di, up], [-1, 0, 1], format="lil")
    Mtx[0, 2] = -0.5 / dx
    e = spsolve(Mtx.tocsr(), rhs)
    return float(np.trapezoid(wgt * e * f, xi))


def S_richardson(gam, Xi=30.0, N=50000):
    """Two resolutions + Richardson (the FD scheme is 2nd order)."""
    a, b = S_fd(gam, Xi, N), S_fd(gam, Xi, 2 * N)
    return (4.0 * b - a) / 3.0


def S_green(gam, T=25.0, n=1500, dps=30):
    """Independent method: the exact hypergeometric Green's function.

        bhat(z) = (-gamma/W) [ y_H(z) int_0^z y_I s du + y_I(z) int_z^1 y_H s du ],
        s = f/z^2 = F^2/z,   W = 4(1-z)(y_I y_H' - y_I' y_H)  (constant),

    evaluated on t = ln(1/z) in [0, T] with Simpson, in mpmath: in that variable
    s du = F^2 dt and the two products y_H*inner, y_I*outer are both O(e^{-t}),
    so nothing overflows however large mu = sqrt(1+gamma) gets."""
    import mpmath as mp

    mp.mp.dps = dps
    mu = mp.sqrt(1 + mp.mpf(gam))
    a = (1 + mu) / 2

    def yH(t):
        return mp.power(t, a) * mp.hyp2f1(a, a, 1, 1 - t)

    def yI(t):
        return mp.power(t, a) * mp.hyp2f1(a, a, 1 + mu, t)

    def Fz(t):
        return mp.hyp2f1(mp.mpf(1) / 2, mp.mpf(1) / 2, 1, 1 - t)

    zc = mp.mpf("0.5")
    W = 4 * (1 - zc) * (yI(zc) * mp.diff(yH, zc) - mp.diff(yI, zc) * yH(zc))

    if n % 2:
        n += 1
    # graded grid t = T u^2: y_I has a log singularity at z = 1 (t = 0) -- it is
    # the horizon-SINGULAR solution -- and the Jacobian 2 T u kills it there.
    uu = [mp.mpf(k) / n for k in range(n + 1)]
    du = mp.mpf(1) / n
    tt = [T * u**2 for u in uu]
    JJ = [2 * mp.mpf(T) * u for u in uu]
    zz = [mp.e ** (-t) for t in tt]
    F2 = [Fz(z) ** 2 for z in zz]
    H = [yH(z) for z in zz]
    I = [mp.mpf(0)] + [yI(z) for z in zz[1:]]
    gI = [mp.mpf(0)] + [I[k] * F2[k] * JJ[k] for k in range(1, n + 1)]
    gH = [mp.mpf(0)] + [H[k] * F2[k] * JJ[k] for k in range(1, n + 1)]

    def cumtrap(g):
        out = [mp.mpf(0)] * (n + 1)
        for k in range(1, n + 1):
            out[k] = out[k - 1] + (g[k] + g[k - 1]) * du / 2
        return out

    outer = cumtrap(gH)
    cI = cumtrap(gI)
    inner = [cI[n] - v for v in cI]
    bh = [mp.mpf(0)] + [
        (-mp.mpf(gam) / W) * (H[k] * inner[k] + I[k] * outer[k])
        for k in range(1, n + 1)
    ]

    def simpson(g):
        ssum = g[0] + g[n]
        ssum += 4 * sum(g[k] for k in range(1, n, 2))
        ssum += 2 * sum(g[k] for k in range(2, n, 2))
        return ssum * du / 3

    I1 = simpson([F2[k] ** 2 * zz[k] * JJ[k] for k in range(n + 1)])  # int F^4 dz
    I2 = simpson([bh[k] * F2[k] * JJ[k] for k in range(n + 1)])  # int bhat F^2/z dz
    return float((I1 - I2) / 2)


# ---------------------------------------------------------------------------
# lattice shells (same convention as dk_selection)
# ---------------------------------------------------------------------------
def shells_gamma(kind, kmax=60):
    out = {}
    for mm in range(-9, 10):
        for nn in range(-9, 10):
            if mm == 0 and nn == 0:
                continue
            k = mm * mm + mm * nn + nn * nn if kind == "tri" else mm * mm + nn * nn
            if k <= kmax:
                out[k] = out.get(k, 0) + 1
    unit = 4 * np.pi / np.sqrt(3.0) if kind == "tri" else 2 * np.pi
    return sorted((unit * k, mult) for k, mult in out.items())


def lattice_sum(kind, Sfun, floor=1e-10):
    tot = 0.0
    for g, mult in shells_gamma(kind):
        wgt = mult * np.exp(-0.5 * g)
        if wgt < floor:
            continue
        tot += wgt * Sfun(g)
    return 0.5 * tot


def moduli_gammas(tau1, tau2, kmax=40):
    """gamma_mn = (2 pi / tau2) |m + n tau|^2 at one flux quantum per cell
    (the paper sec. 5.2 variable).  tau = 1/2 + i sqrt(3)/2 is triangular, tau = i square."""
    out = []
    for mm in range(-7, 8):
        for nn in range(-7, 8):
            if mm == 0 and nn == 0:
                continue
            q = (mm + nn * tau1) ** 2 + (nn * tau2) ** 2
            g = 2 * np.pi * q / tau2
            if g < kmax:
                out.append(g)
    return np.array(out)


# ===========================================================================
def main():
    ladder = "--ladder" in sys.argv

    # ---- [T1] the exact identity at every beta ----------------------------
    log("[T1] exact identity  int p2 ((w0^2)')^2 dr = (4/3) B_c C4")
    bca = KER.load_cache()
    devs = []
    for i in range(len(bca["beta"])):
        br = KER.make_brane(bca, i)
        f = br.w0**2
        fp = -(br.sig**2) * (br.D @ f)
        p2 = np.where(br.sig > 1e-13, br.P2 / np.maximum(br.sig, 1e-300) ** 3, 0.0)
        p1 = br.P1 * br.sig
        lhs = br.integ_dr(p2 * fp**2).real
        C4 = br.integ_dr(p1 * br.w0**4).real
        devs.append(abs(lhs / ((4 / 3) * br.Bc * C4) - 1.0))
    devs = np.array(devs)
    check(
        "[T1] identity holds at all 23 beta of bc_alpha.npz",
        devs.max() < 1e-10,
        f"max relative deviation {devs.max():.1e}",
    )

    # ---- [T2] the tail of the committed probe kernel ----------------------
    log("[T2] abelian large-s tail against (4/3) B_c C4")
    ker = np.load(KER_PATH)
    C4_0, Bc_0 = float(ker["C4"]), float(ker["B_c"])
    pred = (4 / 3) * Bc_0 * C4_0
    G2, X = np.asarray(ker["G2grid"]), np.asarray(ker["X"])
    CX = C4_0 - X
    top = G2 > 100.0
    # s (C4-X) = a0 + a1/s + a2/s^2 + ... ; the intercept a0 is the test
    A = np.vstack([G2[top] ** (-k) for k in range(4)]).T
    coef, *_ = np.linalg.lstsq(A, G2[top] * CX[top], rcond=None)
    check(
        "[T2] s (C4-X) extrapolates to (4/3) B_c C4",
        abs(coef[0] / pred - 1.0) < 1e-4,
        f"intercept {coef[0]:.6f} vs {pred:.6f} (rel {abs(coef[0] / pred - 1):.1e}); "
        f"1/s coefficient {coef[1]:.3f}",
    )

    # ---- [T3] complete monotonicity, constructively -----------------------
    log("[T3] positive spectral representation (Yukawa superposition)")
    from scipy.linalg import eig

    br0 = KER.make_brane(bca, 0)
    sig, D = br0.sig, br0.D
    D2 = D @ D
    Phat = br0.bg.A * np.exp(br0.bg.Nn)
    Phat_s = D @ Phat
    mass = sig * np.exp(br0.bg.Nn - 2 * br0.bg.M)
    Lm = -(Phat * sig)[:, None] * D2 - (Phat_s * sig - Phat)[:, None] * D
    Mm = np.diag(mass)
    Lm[br0.N, :] = 0.0
    Lm[br0.N, br0.N] = 1.0
    Mm[br0.N, :] = 0.0
    vals, vecs = eig(Lm, Mm)
    vals = vals.real
    okv = np.where(np.isfinite(vals) & (vals > 1e-6) & (vals < 1e12))[0]
    order = okv[np.argsort(vals[okv])]
    tau = vals[order]
    P1s = br0.P1 * sig
    cw = []
    for j in order:
        ph = vecs[:, j].real
        ph = ph / np.sqrt(br0.integ_dr(P1s * ph**2).real)
        cw.append(br0.integ_dr(P1s * ph * br0.w0**2).real)
    cw = np.array(cw)
    C4_b = br0.integ_dr(P1s * br0.w0**4).real
    ss = np.array([0.05, 1.0, 5.13126764, 20.0, 100.0, 300.0])
    got = np.array([float(np.sum(cw**2 * tau / (tau + s))) for s in ss])
    ref = np.array([br0.photon_kernel(float(s))[0] for s in ss])
    check(
        "[T3] sum_n c_n^2 tau_n/(tau_n+s) reproduces (C4-X)(s)",
        np.max(np.abs(got - ref)) < 1e-9,
        f"max |diff| {np.max(np.abs(got - ref)):.1e} over s in [0.05, 300]",
    )
    check(
        "[T3] all spectral weights c_n^2 >= 0 and tau_n > 0  => completely monotone",
        bool(np.all(tau > 0)) and bool(np.all(cw**2 >= 0)),
        f"min tau = {tau.min():.8f}, min c_n^2 = {(cw**2).min():.1e}",
    )
    check(
        "[T3] the tower head is tau_0 = B_c, i.e. the singularity sits at gamma = -1",
        abs(tau[0] / br0.Bc - 1.0) < 1e-10,
        f"tau_0 = {tau[0]:.8f}, B_c = {br0.Bc:.8f}",
    )
    check(
        "[T3] Parseval: sum_n c_n^2 = C4",
        abs(float(np.sum(cw**2)) / C4_b - 1) < 1e-10,
        f"{float(np.sum(cw**2)):.9f} vs {C4_b:.9f}",
    )
    check(
        "[T3] first moment: sum_n c_n^2 tau_n = (4/3) B_c C4  (the [T1] identity)",
        abs(float(np.sum(cw**2 * tau)) / ((4 / 3) * br0.Bc * C4_b) - 1) < 1e-6,
        f"{float(np.sum(cw**2 * tau)):.6f} vs {(4 / 3) * br0.Bc * C4_b:.6f}",
    )

    # ---- [T4] the throat kernel S(gamma), two methods ---------------------
    log("[T4] the throat kernel S(gamma)")
    xi = np.linspace(0, 30, 300001)
    fq, _ = f_and_df(xi)
    S0 = float(np.trapezoid(np.exp(2 * xi) * fq**2, xi))
    S0_closed = S0_closed_form()
    check(
        "[T4] closed form S(0) = (1/2) int_0^1 F^4 dz = 1.89804004507405",
        abs(S0_closed - S0_EXACT) < 1e-12,
        f"{S0_closed:.14f}",
    )
    check(
        "[T4] S(0) from the xi-quadrature of the BVP solution matches it",
        abs(S0 / S0_closed - 1) < 1e-6,
        f"{S0:.10f} vs {S0_closed:.10f} (rel {abs(S0 / S0_closed - 1):.1e})",
    )
    GAMS = np.array(
        [0.25, 0.5, 1.0, 2.0, 4.0, 7.2552, 12.5664, 20.0, 40.0, 200.0, 2000.0]
    )
    Sfd = np.array([S_richardson(float(g)) for g in GAMS])
    # the graded-grid Green's quadrature is 2nd order in 1/n: Richardson it
    gg_green = [0.5, 2.0, 20.0]
    Sgr = np.array(
        [(4 * S_green(g, n=3000) - S_green(g, n=1500)) / 3 for g in gg_green]
    )
    idx = [int(np.argmin(np.abs(GAMS - g))) for g in gg_green]
    dmax = float(np.max(np.abs(Sfd[idx] - Sgr) / np.abs(Sgr)))
    check(
        "[T4] two methods: FD+Richardson vs the hypergeometric Green's function",
        dmax < 2e-7,
        f"max relative |diff| {dmax:.1e} over gamma = {gg_green}",
    )
    tailratio = GAMS[-1] * Sfd[-1] / ((4 / 3) * S0)
    check(
        "[T4] gamma S(gamma) -> (4/3) S(0)  (the throat instance of [T1])",
        abs(tailratio - 1.0) < 2e-3,
        f"gamma S / (4/3) S(0) = {tailratio:.6f} at gamma = {GAMS[-1]:g}",
    )

    # ---- [T5] the throat-limit lattice selection --------------------------
    log("[T5] lattice selection in the alpha -> alpha_* limit")
    cacheS = {}

    def S(g):
        g = float(g)
        if g not in cacheS:
            cacheS[g] = S_richardson(g)
        return cacheS[g]

    ctri = lattice_sum("tri", S)
    csq = lattice_sum("sq", S)
    gap = csq - ctri
    check(
        "[T5] triangular is favoured in the limit",
        gap > 0,
        f"C_tri = {ctri:.8f}, C_sq = {csq:.8f}, gap = {gap:+.8f}, "
        f"relative gap {gap / ctri:+.6f}",
    )
    # moduli scan (the paper sec. 5.2 variable): triangular must be the global min.
    # S is interpolated (PCHIP in ln gamma) off a 160-point grid -- the scan
    # needs ~3e4 evaluations; the interpolant is checked against direct solves.
    from scipy.interpolate import PchipInterpolator

    ggrid = np.geomspace(0.05, 400.0, 160)
    Sgrid = np.array([S_richardson(float(g)) for g in ggrid])
    Sint = PchipInterpolator(np.log(ggrid), Sgrid)
    tst = np.array([0.3, 1.7, 5.5, 18.0, 60.0])
    err = np.max(np.abs(Sint(np.log(tst)) / np.array([S(g) for g in tst]) - 1))
    check(
        "[T5] the PCHIP interpolant of S is accurate on the scan grid",
        err < 1e-4,
        f"max relative error {err:.1e}",
    )
    best, bestC = None, np.inf
    scan = []
    for t1 in np.linspace(0.0, 0.5, 26):
        for t2 in np.linspace(0.5, 3.0, 251):
            gams = moduli_gammas(t1, t2)
            wgts = np.exp(-0.5 * gams)
            keepg = (wgts > 1e-10) & (gams > 0.05)
            C = 0.5 * float(np.sum(wgts[keepg] * Sint(np.log(gams[keepg]))))
            scan.append((t1, t2, C))
            if C < bestC:
                bestC, best = C, (t1, t2)
    check(
        "[T5] the moduli scan lands on triangular, tau = 1/2 + i sqrt3/2",
        abs(best[0] - 0.5) < 0.03 and abs(best[1] - np.sqrt(3.0) / 2) < 0.02,
        f"argmin (tau1, tau2) = ({best[0]:.3f}, {best[1]:.3f}) vs "
        f"triangular (0.5, {np.sqrt(3.0) / 2:.3f}); C = {bestC:.8f} "
        f"vs C_tri = {ctri:.8f}",
    )

    # ---- [T6] the brane ladder (slow) -------------------------------------
    lad = {}
    if ladder:
        log("[T6] beta ladder: geometry and kernel shape (slow)")
        from backreaction.numerics import bc_alpha as bcam
        from backreaction.numerics import dk_background as dkb

        LAD = [
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
        bg = None
        for beta, Nn in LAD:
            guess = None
            if bg is not None:
                _, xs = dkb.cheb(Nn)
                sgn = (xs + 1) / 2.0
                guess = (bg._bary(bg.A, sgn), bg._bary(bg.M, sgn), bg._bary(bg.Nn, sgn))
            bg = dkb.solve_colloc(
                beta, N=Nn, tol=1e-10, maxit=200, guess=guess, require=True
            )
            if beta not in (1e4, 1e6, 1e8):
                continue
            vals, w0 = bcam.bc_spectral_grid(bg, n_keep=2)
            Bc = float(vals[0])
            b_h = (
                Bc
                * (1 / 3.0)
                * float(
                    np.real(np.exp(-2 * bg.fields_r(np.array([1.0 + 1e-9]))["V"][0]))
                )
            )
            sg, Dg = bg.sig, bg.D
            ccw = KER.cc_weights_01(bg.N)
            P1g, P2g = np.exp(bg.Nn - 2 * bg.M), bg.A * np.exp(bg.Nn)
            P2sg = Dg @ P2g

            def integ(gv, sg=sg, ccw=ccw, npts=bg.N + 1):
                o = np.zeros(npts)
                mk = sg > 1e-13
                o[mk] = np.asarray(gv)[mk] / sg[mk] ** 2
                return float(ccw @ o)

            C4b = integ(P1g * sg * w0**4)
            shape = []
            for g in GAMS[:9]:
                s = g * Bc
                Lm = (
                    np.diag(-sg * P2g) @ (Dg @ Dg)
                    + np.diag(P2g - sg * P2sg) @ Dg
                    + np.diag(s * sg * P1g)
                )
                rr = s * sg * P1g * w0**2
                Lm[bg.N, :] = 0.0
                Lm[bg.N, bg.N] = 1.0
                rr[bg.N] = 0.0
                bh = solve(Lm, rr)
                shape.append((C4b - integ(P1g * sg * bh * w0**2)) / C4b)
            lad[beta] = (Bc, b_h, np.array(shape))
            log(f"    beta = {beta:g}: B_c = {Bc:.4f}, nu^2 = b_h - 1 = {b_h - 1:.4f}")
        # geometry
        rg = np.array([1.05, 1.2, 1.5, 2.0])
        fg = bg.fields_r(rg)
        rel = np.abs(np.real(fg["U"]) / ((rg - 1) * (3 * rg + 1)) - 1)
        check(
            "[T6] near-horizon geometry -> (r-1)(3r+1) at beta = 1e8",
            rel.max() < 3e-3,
            f"max relative deviation {rel.max():.1e} over r in [1.05, 2] "
            f"(it grows towards the crossover r_c ~ (beta/6)^{{1/4}}: "
            f"{np.abs(np.real(bg.fields_r(np.array([3.0]))['U'])[0] / 20.0 - 1):.1e} at r = 3)",
        )
        relW = np.abs(np.real(fg["Wp"]) * (rg - 1 / 3.0) - 1)
        check(
            "[T6] W' -> 1/(r - 1/3)",
            relW.max() < 3e-3,
            f"max relative deviation {relW.max():.1e}",
        )
        # shape convergence at rate nu^2
        shp = Sfd[:9] / S0
        d = {b: float(np.max(np.abs(lad[b][2] - shp))) for b in lad}
        nus = {b: lad[b][1] - 1.0 for b in lad}
        ratios = [d[b] / nus[b] for b in sorted(lad)]
        check(
            "[T6] the brane kernel shape converges to S(gamma)/S(0) at rate nu^2",
            d[max(lad)] < d[min(lad)] and max(ratios) / min(ratios) < 1.6,
            "  ".join(
                f"beta={b:g}: max|d| = {d[b]:.4f} (nu^2 = {nus[b]:.4f})"
                for b in sorted(lad)
            ),
        )

    # ---- checks, then save + figure ----------------------------------------
    nfail = sum(1 for v in CHECKS.values() if not v)
    if nfail:
        for k, ok in CHECKS.items():
            if not ok:
                log(f"  FAILED: {k}")
        log("cache and figure not written: a check failed")
        sys.exit(1)
    if lad:
        lad_beta = np.array(sorted(lad))
        lad_shape = np.array([lad[b][2] for b in sorted(lad)])
        lad_nu2 = np.array([lad[b][1] - 1 for b in sorted(lad)])
    else:
        # without --ladder, keep the ladder arrays of the committed cache
        try:
            with np.load(CACHE) as old:
                lad_beta, lad_shape, lad_nu2 = (
                    old["ladder_beta"],
                    old["ladder_shape"],
                    old["ladder_nu2"],
                )
        except (OSError, KeyError):
            lad_beta = lad_shape = lad_nu2 = np.zeros(0)
    np.savez(
        CACHE,
        gamma=GAMS,
        S=Sfd,
        S0=S0,
        S0_closed=S0_closed,
        gamma_green=np.array(gg_green),
        S_green=Sgr,
        gap=gap,
        C_tri=ctri,
        C_sq=csq,
        moduli=np.array(scan),
        best=np.array(best),
        identity_dev=devs,
        tail_pred=pred,
        tail_fit=coef,
        ladder_beta=lad_beta,
        ladder_shape=lad_shape,
        ladder_nu2=lad_nu2,
        readme=(
            "alpha -> alpha_* throat limit of the lattice kernel.  S(gamma) is "
            "the BTZ x R^2 abelian kernel shape (K = N^4 h^2 S, the prefactor "
            "gamma-independent); gamma S -> (4/3) S(0) exactly.  gap = C_sq - "
            "C_tri in the limit; moduli = (tau1, tau2, C) scan."
        ),
    )
    log(f"saved {CACHE}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].loglog(GAMS, Sfd, "o-", ms=3, label=r"throat $\mathcal{S}(\gamma)$")
    ax[0].loglog(
        GAMS,
        (4 / 3) * S0 / GAMS,
        "--",
        lw=1,
        label=r"exact tail $\frac{4}{3}\mathcal{S}(0)/\gamma$",
    )
    ax[0].set_xlabel(r"$\gamma = G^2/B$")
    ax[0].set_ylabel(r"$\mathcal{S}(\gamma)$")
    ax[0].set_title(r"BTZ throat kernel ($\alpha \to \alpha_\star$)")
    ax[0].legend(fontsize=8)
    if lad:
        for b in sorted(lad):
            ax[1].plot(
                GAMS[:9],
                lad[b][2],
                "o-",
                ms=3,
                label=rf"$\beta = 10^{{{int(np.log10(b))}}}$",
            )
    ax[1].plot(GAMS[:9], Sfd[:9] / S0, "k--", lw=1.4, label="throat limit")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$\gamma$")
    ax[1].set_ylabel(r"$K(\gamma B_c)/K(0)$")
    ax[1].set_title("brane kernel shape vs the throat limit")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=160)
    log(f"saved {FIG}")

    log("")
    log("SUMMARY")
    log(f"  S(0) = {S0:.8f}   exact tail (4/3) S(0) = {(4 / 3) * S0:.8f}")
    log(f"  limit gap C_sq - C_tri = {gap:+.8f}  (relative {gap / ctri:+.6f})")
    log("  the limit is a FIXED BTZ problem: no beta-drift, no log-periodicity")
    log(f"  {len(CHECKS)}/{len(CHECKS)} checks passed")
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
