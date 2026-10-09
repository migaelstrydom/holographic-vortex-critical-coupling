"""Stability of the uniform triangular lattice against every lowest-Landau-level
perturbation on the D'Hoker-Kraus brane, and the coupling at which it is lost.

The quartic LLL functional with the exact kernel K(s; beta) is

    F[psi] = -eps <|psi|^2> + (1/2) sum_k |rho_k|^2 kappa(k^2),
    kappa(gamma) = K(gamma B_c; beta),   gamma = k^2/B  (B = 1 below).

Its second variation about the triangular lattice is derived in
`backreaction.derivations.lll_second_variation`: in the magnetic Bloch basis
the Hessian is block diagonal with eigenvalues

    lambda_pm(p) = A(p) + S_0(p) +- |S_2(p)|,

which do not involve the strict uniform response kappa_{G=0} for p != 0.  As
p -> 0, lambda_- -> 0 (phase mode) and lambda_+ -> 2 (K_0 + 2 C_tri), with
K_0 = K(s -> 0) and C_tri = (1/2) sum_{G != 0} e^{-gamma_G/2} K(gamma_G B_c).
So the long-wavelength density stiffness is K_0 + 2 C_tri, and the lattice is
lost to long-wavelength density modulation where K_0/2 + C_tri = 0.

This module
  [1] builds, on each rung, a spline of the kernel in gamma from direct solves
      anchored at gamma = 0 by K_0 = K_{G=0} (collocation of the G = 0 system,
      numerics/dk_uniform.py) and checks it against fresh off-grid solves;
  [2] method A -- real space: constructs exact LLL states on a rectangular
      magnetic torus (Landau gauge, Gaussian rows), checks the Abrikosov
      triangular state (beta_A, |omega_G|^2 = e^{-gamma_G/2}), assembles the
      full real Hessian of F by FFT, and compares its spectrum with the
      Bogoliubov formula at every torus momentum;
  [3] method A' -- the exactly modulated Abrikosov state
      C_n -> C_n (1 + delta cos(2 pi n/M)) on an M-row torus, whose second
      variation Q_M is evaluated by FFT and extrapolated in M -> oo to the
      density stiffness, compared with K_0 + 2 C_tri;
  [4] method B -- the Bogoliubov spectrum over the magnetic Brillouin zone
      (an N_BZ x N_BZ grid plus the Gamma-M-K-Gamma path) from the probe limit
      beta = 0 through the small-beta rungs to both sides of the threshold,
      reporting the lowest eigenvalue and where it sits; in the probe limit the
      zone scan is repeated as a direct diagonalisation of the Hessian on a
      36-flux-quantum torus whose momenta include M and K;
  [5] locates the zero of K_0/2 + C_tri by secant iteration in ln beta with
      fresh solves, with an error budget (K_0 by collocation vs shooting vs the
      s -> 0 extrapolation of the G != 0 kernel, collocation resolution, shell
      truncation, interpolation on the cached ladder), and checks it against
      alpha_L of dk_uniform.npz.

CHECKS
------
[B1] spline of kappa(gamma) reproduces fresh off-grid solves to < 1e-7 C4
[B2] Abrikosov state: beta_A = 1.1595953 (to 1e-7), |omega_G|^2 = e^{-gamma_G/2}
     (to 1e-10), |psi|^2 periodic under a2 (to 1e-10)
[B3] 24-flux-quantum torus at beta = 5000 (alpha = 0.36397): Hessian spectrum =
     Bogoliubov multiset {lambda_pm(p)} (to 1e-13 C4; measured 1.6e-15),
     with the Gamma amplitude eigenvalue 2 (kappa_{G=0} + 2C) the only one that
     moves when kappa_{G=0} is changed
[B4] Q_M extrapolates to K_0 + 2 C_tri (to 2e-4 relative to C4), and Q_M does
     not depend on kappa_{G=0}
[B5] {lambda_+, lambda_-}(p -> 0) -> {0, 2 (K_0 + 2 C_tri)}
[B6] below the threshold no eigenvalue anywhere in the zone is negative, from
     beta = 0 up; above it the unstable set is a neighbourhood of p = 0 only
[B7] the zero of K_0/2 + C_tri is bracketed, the secant converges to 1e-10,
     the error budget is below 1e-7 in alpha, and the zero agrees with
     alpha_L of dk_uniform.npz within that budget
[B8] C_tri on the cached ladder (dk_long_wavelength.npz) is reproduced on fresh rungs
     (cache consistency through the same kernel solver, not a second method)
[B9] probe limit, 36-flux-quantum torus: Hessian = Bogoliubov multiset, and
     its lowest non-zero eigenvalue is positive and equals the zone-formula
     minimum over the torus momenta; lambda_-(M) from the torus = zone formula

Run:  uv run python -m backreaction.numerics.dk_bogoliubov        (~4 min)
      ... --quick   half the Brillouin-zone grid (nbz = 120); writes nothing
"""

import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from backreaction import paths
from backreaction.numerics import dk_long_wavelength as CL

T0 = time.time()
CHECKS = {}
OUT = paths.data("dk_bogoliubov.npz")
LONGWAVE = paths.data("dk_long_wavelength.npz")
UNIFORM = paths.data("dk_uniform.npz")


def zr_beta_guess(x0, x1):
    return float(np.exp(0.5 * (x0 + x1)))


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# --------------------------------------------------------------------------
# geometry of the one-flux-quantum triangular lattice, B = 1
# --------------------------------------------------------------------------
ALAT = np.sqrt(4 * np.pi / np.sqrt(3))  # cell area 2 pi
A1 = np.array([ALAT, 0.0])
A2 = np.array([ALAT / 2, ALAT * np.sqrt(3) / 2])
B1 = 2 * np.pi * np.array([A2[1], -A2[0]]) / (A1[0] * A2[1] - A1[1] * A2[0])
B2 = 2 * np.pi * np.array([-A1[1], A1[0]]) / (A1[0] * A2[1] - A1[1] * A2[0])
ROW = 2 * np.pi / ALAT  # Landau-gauge row spacing = a sqrt(3)/2
GAMMA_CUT = 90.0  # e^{-45} below double precision


def recip(nmax=12):
    m, n = np.meshgrid(
        np.arange(-nmax, nmax + 1), np.arange(-nmax, nmax + 1), indexing="ij"
    )
    G = m.reshape(-1, 1) * B1 + n.reshape(-1, 1) * B2
    return G[np.sum(G * G, 1) <= GAMMA_CUT + 40]


GVEC = recip()


def wedge(u, v):
    return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]


# --------------------------------------------------------------------------
# [1] the kernel on a rung, as a spline in gamma
# --------------------------------------------------------------------------
class Kernel:
    def __init__(self, rung):
        self.rung = rung
        self.C4 = rung.C4
        self.K0 = rung.K0()  # K_{G=0}, collocation
        self.K0_alt = rung.K0_ext()  # s -> 0 extrapolation of the G != 0 kernel
        g = np.unique(
            np.concatenate(
                [
                    np.geomspace(1e-3, 1.0, 60),
                    np.linspace(1.0, 20.0, 140),
                    np.linspace(20.0, GAMMA_CUT + 40, 90),
                ]
            )
        )
        k = np.array([rung.K(x) for x in g])
        self.spl = CubicSpline(
            np.concatenate([[0.0], g]), np.concatenate([[self.K0], k])
        )
        self.gmax = g[-1]

    def __call__(self, gam):
        gam = np.asarray(gam, float)
        return np.where(
            gam <= self.gmax, self.spl(np.minimum(gam, self.gmax)), self.spl(self.gmax)
        )

    def spline_error(self, n=25, seed=1):
        rng = np.random.default_rng(seed)
        g = np.concatenate(
            [rng.uniform(1e-3, 1.0, n // 2), rng.uniform(1.0, 60.0, n - n // 2)]
        )
        return (
            max(abs(self(x) - self.rung.br.kernel(x * self.rung.Bc)["K"]) for x in g)
            / self.C4
        )

    def C_tri(self):
        G2 = np.sum(GVEC * GVEC, 1)
        G2 = G2[G2 > 1e-12]
        return 0.5 * float(np.sum(np.exp(-G2 / 2) * self(G2)))


# --------------------------------------------------------------------------
# [4] the Bogoliubov spectrum
# --------------------------------------------------------------------------
def bogoliubov(kap, P, kG0=None):
    """lambda_pm at momenta P (n, 2); kap(gamma) the kernel in LLL units.
    kG0 replaces kappa at exactly zero momentum transfer (only reached at p = 0)."""
    P = np.atleast_2d(P)
    G = GVEC
    G2 = np.sum(G * G, 1)
    nz = G2 > 1e-12
    kGG = np.exp(-G2[nz] / 2) * kap(G2[nz])
    lam_p, lam_m = np.empty(len(P)), np.empty(len(P))
    for i0 in range(0, len(P), 2000):
        Pc = P[i0 : i0 + 2000]
        wGp = wedge(G[None, :, :], Pc[:, None, :])  # G ^ p
        A = np.sum(kGG[None, :] * (np.cos(wGp[:, nz]) - 1.0), 1)
        D = G[None, :, :] - Pc[:, None, :]
        D2 = np.sum(D * D, 2)
        kv = kap(D2)
        if kG0 is not None:
            kv = np.where(D2 < 1e-20, kG0, kv)
        w = np.exp(-D2 / 2) * kv
        S0 = np.sum(w, 1)
        S2 = np.abs(np.sum(w * np.exp(-1j * wGp), 1))
        lam_p[i0 : i0 + 2000] = A + S0 + S2
        lam_m[i0 : i0 + 2000] = A + S0 - S2
    return lam_p, lam_m


def zone_grid(nbz):
    f = (np.arange(nbz) + 0.5) / nbz - 0.5  # offset grid avoids p = 0 exactly
    u, v = np.meshgrid(f, f, indexing="ij")
    P = u.reshape(-1, 1) * B1 + v.reshape(-1, 1) * B2
    # fold into the Wigner-Seitz cell for reporting |p|
    best = P.copy()
    for m in (-1, 0, 1):
        for n in (-1, 0, 1):
            Q = P + m * B1 + n * B2
            sel = np.sum(Q * Q, 1) < np.sum(best * best, 1)
            best[sel] = Q[sel]
    return best


def path_GMKG(npts=400):
    Mp = B1 / 2
    Kp = (2 * B1 + B2) / 3  # b1, b2 at 120 degrees: |K| = |b|/sqrt(3)
    assert abs(np.linalg.norm(Kp) - np.linalg.norm(B1) / np.sqrt(3)) < 1e-12
    seg = []
    for a, b in ((np.zeros(2), Mp), (Mp, Kp), (Kp, np.zeros(2))):
        t = np.linspace(0, 1, npts, endpoint=False)[:, None]
        seg.append(a + t * (b - a))
    P = np.concatenate(seg)[1:]  # drop Gamma
    return P, Mp, Kp


# --------------------------------------------------------------------------
# [2] real-space LLL states on a rectangular magnetic torus
# --------------------------------------------------------------------------
class Torus:
    """Lx = n1 a, Ly = R * ROW (R rows of the Landau gauge, A = (-y, 0)).
    Basis psi_j = sum_m exp(i k_{j+mN} x) exp(-(y + k_{j+mN})^2/2), k_j = 2 pi j/Lx,
    N = Lx Ly / 2 pi flux quanta.  Rows of the lattice state sit at j = n1 n."""

    def __init__(self, n1, R, h=0.2):
        self.n1, self.R = n1, R
        self.Lx, self.Ly = n1 * ALAT, R * ROW
        self.N = n1 * R
        self.nx = int(2 * np.ceil(self.Lx / h / 2))
        self.ny = int(2 * np.ceil(self.Ly / h / 2))
        x = np.arange(self.nx) * self.Lx / self.nx
        y = np.arange(self.ny) * self.Ly / self.ny
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.Lx / self.nx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.Ly / self.ny)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.basis = np.array([self._psi(j) for j in range(self.N)])
        nrm = np.mean(np.abs(self.basis) ** 2, axis=(1, 2))
        self.basis /= np.sqrt(nrm)[:, None, None]

    def _psi(self, j):
        out = np.zeros_like(self.X, dtype=complex)
        mmax = int(np.ceil(12 / self.Ly)) + 2
        for m in range(-mmax - 1, mmax + 2):
            kk = 2 * np.pi * (j + m * self.N) / self.Lx
            out += np.exp(1j * kk * self.X) * np.exp(-((self.Y + kk) ** 2) / 2)
        return out

    def state(self, coef):
        return np.tensordot(coef, self.basis, axes=1)

    def lattice_coef(self, mod=None):
        c = np.zeros(self.N, complex)
        for nrow in range(self.R):
            c[self.n1 * nrow] = np.exp(1j * np.pi * nrow**2 / 2) * (
                1.0 if mod is None else mod(nrow)
            )
        return c / np.sqrt(self.R)

    def ft(self, f):
        return np.fft.fft2(f) / f.size  # <f e^{-ik.x}>

    def kap_grid(self, kap, kG0):
        kv = kap(self.K2)
        kv[0, 0] = kG0
        return kv

    def F(self, psi, kv, eps):
        rho = np.abs(psi) ** 2
        rk = self.ft(rho)
        return -eps * np.mean(rho) + 0.5 * np.sum(np.abs(rk) ** 2 * kv)

    def eps_of(self, psiL, kv):
        rk = self.ft(np.abs(psiL) ** 2)
        return float(np.sum(np.abs(rk) ** 2 * kv))  # stationarity at eta = 1

    def quad(self, psiL, kv, eps, U, Uv=None):
        """Bilinear second variation Q(u, v) for stacks of fields U, Uv."""
        Uv = U if Uv is None else Uv
        V = np.real(
            np.fft.ifft2(self.ft(np.abs(psiL) ** 2) * kv) * kv.size
        )  # (K * rho_L)(x)
        M12 = (
            np.real(np.einsum("axy,bxy->ab", np.conj(U), (V - eps)[None] * Uv)) / V.size
        )
        dU = np.array([self.ft(2 * np.real(np.conj(psiL) * u)) for u in U])
        dV = (
            dU
            if Uv is U
            else np.array([self.ft(2 * np.real(np.conj(psiL) * u)) for u in Uv])
        )
        M3 = 0.5 * np.real(np.einsum("axy,bxy->ab", dU * kv[None], np.conj(dV)))
        return M12 + M3

    def momenta(self):
        """Torus momenta reduced modulo the reciprocal lattice (one per flux quantum)."""
        out = []
        seen = set()
        for i in range(-self.N, self.N + 1):
            for j in range(-self.N, self.N + 1):
                p = np.array([2 * np.pi * i / self.Lx, 2 * np.pi * j / self.Ly])
                f = np.array([p @ A1, p @ A2]) / (2 * np.pi)
                f = f - np.floor(f + 1e-9)
                key = (
                    round(f[0] * 4 * self.N) % (4 * self.N),
                    round(f[1] * 4 * self.N) % (4 * self.N),
                )
                if key not in seen:
                    seen.add(key)
                    out.append(f[0] * B1 + f[1] * B2)
        return np.array(out)


def torus_validation(ker, n1=4, R=6):
    tor = Torus(n1, R)
    psiL = tor.state(tor.lattice_coef())
    rho = np.abs(psiL) ** 2
    betaA = np.mean(rho**2) / np.mean(rho) ** 2
    rk = tor.ft(rho)
    # |omega_G|^2 = e^{-G^2/2} at the reciprocal vectors present on the torus
    errs = []
    for G in GVEC:
        if G @ G > 30:
            continue
        i = int(round(G[0] * tor.Lx / (2 * np.pi))) % tor.nx
        j = int(round(G[1] * tor.Ly / (2 * np.pi))) % tor.ny
        errs.append(abs(abs(rk[i, j]) ** 2 - np.exp(-(G @ G) / 2)))
    # every other torus Fourier component of rho must vanish (rho has the lattice periodicity)
    mask = np.ones_like(rho, bool)
    for G in GVEC:
        i = int(round(G[0] * tor.Lx / (2 * np.pi))) % tor.nx
        j = int(round(G[1] * tor.Ly / (2 * np.pi))) % tor.ny
        mask[i, j] = False
    stray = float(np.max(np.abs(rk[mask])))
    check(
        "[B2] Abrikosov state on the torus",
        abs(betaA - 1.1595953) < 1e-7 and max(errs) < 1e-10 and stray < 1e-10,
        f"beta_A = {betaA:.10f}, max ||omega_G|^2 - e^(-G^2/2)| = {max(errs):.1e}, "
        f"stray harmonics {stray:.1e}, N = {tor.N} flux quanta, grid {tor.nx}x{tor.ny}",
    )

    out = {}
    for label, kG0 in (("K0", ker.K0), ("K0+C4", ker.K0 + ker.C4)):
        kv = tor.kap_grid(ker, kG0)
        eps = tor.eps_of(psiL, kv)
        U = np.concatenate([tor.basis, 1j * tor.basis])
        M = tor.quad(psiL, kv, eps, U)
        M = 0.5 * (M + M.T)
        ev = np.sort(np.linalg.eigvalsh(M))
        P = tor.momenta()
        assert len(P) == tor.N, (len(P), tor.N)
        isG = np.sum(P * P, 1) < 1e-20
        lp, lm = bogoliubov(ker, P[~isG])
        C = ker.C_tri()
        pred = np.sort(np.concatenate([lp, lm, [0.0, 2 * (kG0 + 2 * C)]]))
        out[label] = (ev, pred)
    dev = max(np.max(np.abs(ev - pr)) for ev, pr in out.values()) / ker.C4
    # multiset difference: which eigenvalues of the second spectrum have no partner in the first
    e1, e2 = list(out["K0"][0] / ker.C4), out["K0+C4"][0] / ker.C4
    new_vals = []
    for x in e2:
        j = int(np.argmin(np.abs(np.array(e1) - x)))
        if abs(e1[j] - x) < 1e-9:
            e1.pop(j)
        else:
            new_vals.append(x)
    shift = (new_vals[0] - e1[0]) if (len(new_vals) == 1 and len(e1) == 1) else np.nan
    check(
        "[B3] torus Hessian = Bogoliubov at every torus momentum; kappa_(G=0) moves one eigenvalue",
        dev < 1e-13 and len(new_vals) == 1 and abs(shift - 2.0) < 1e-9,
        f"max |dev| = {dev:.1e} C4 over {2 * tor.N} eigenvalues; kappa_(G=0) -> kappa_(G=0) + C4 "
        f"changes {len(new_vals)} eigenvalue(s), by {shift:.9f} C4 (expected 2: the Gamma amplitude mode)",
    )
    return dev


def torus_spectrum(ker, n1=6, R=6):
    """Direct diagonalisation of the Hessian on an n1 x R torus (default 36
    flux quanta: its momenta include Gamma, one M = b2/2 and one K point).
    Returns the lowest non-zero eigenvalue, the Bogoliubov minimum over the
    torus momenta, lambda_-(M) from the zone formula, the multiset deviation
    and the torus Abrikosov ratio (all in C4)."""
    tor = Torus(n1, R)
    psiL = tor.state(tor.lattice_coef())
    rho = np.abs(psiL) ** 2
    betaA = float(np.mean(rho**2) / np.mean(rho) ** 2)
    kv = tor.kap_grid(ker, ker.K0)
    eps = tor.eps_of(psiL, kv)
    U = np.concatenate([tor.basis, 1j * tor.basis])
    Mh = tor.quad(psiL, kv, eps, U)
    ev = np.sort(np.linalg.eigvalsh(0.5 * (Mh + Mh.T))) / ker.C4
    P = tor.momenta()
    assert len(P) == tor.N, (len(P), tor.N)
    isG = np.sum(P * P, 1) < 1e-20
    lp, lm = bogoliubov(ker, P[~isG])
    C = ker.C_tri()
    pred = np.sort(np.concatenate([lp, lm, [0.0, 2 * (ker.K0 + 2 * C)]])) / ker.C4
    Mp = B2 / 2
    hasM = bool(np.any(np.sum((P - Mp) ** 2, 1) < 1e-16))
    Kp = (B1 + 2 * B2) / 3
    folded = [P - m * B1 - n * B2 for m in (-1, 0, 1) for n in (-1, 0, 1)]
    hasK = any(
        bool(np.any(np.sum((Q - Kp) ** 2, 1) < 1e-16))
        or bool(np.any(np.sum((Q + Kp) ** 2, 1) < 1e-16))
        for Q in folded
    )
    lmM = float(bogoliubov(ker, Mp[None])[1][0] / ker.C4)
    # the eigenvalue of the torus Hessian closest to the zone-formula lambda_-(M)
    evM = float(ev[np.argmin(np.abs(ev - lmM))])
    return dict(
        betaA=betaA,
        zero=float(ev[0]),
        ev_min=float(ev[1]),
        bogo_min=float(min(lm.min(), lp.min()) / ker.C4),
        dev=float(np.max(np.abs(ev - pred))),
        lmM=lmM,
        evM=evM,
        hasM=hasM,
        hasK=hasK,
        N=tor.N,
    )


def modulation(ker, Ms=(8, 16, 32, 64, 128)):
    """Second variation of F along the exactly modulated Abrikosov state."""
    C = ker.C_tri()
    target = (ker.K0 + 2 * C) / ker.C4
    rows = []
    for M in Ms:
        tor = Torus(1, M, h=0.25)
        psiL = tor.state(tor.lattice_coef())
        u = tor.state(tor.lattice_coef(mod=lambda n, M=M: np.cos(2 * np.pi * n / M)))
        qs = []
        for kG0 in (ker.K0, ker.K0 + ker.C4):
            kv = tor.kap_grid(ker, kG0)
            eps = tor.eps_of(psiL, kv)
            qs.append(float(tor.quad(psiL, kv, eps, u[None])[0, 0]))
            # cross-check by finite differences of F itself
        kv = tor.kap_grid(ker, ker.K0)
        eps = tor.eps_of(psiL, kv)
        d = 1e-3
        fd = (
            tor.F(psiL + d * u, kv, eps)
            + tor.F(psiL - d * u, kv, eps)
            - 2 * tor.F(psiL, kv, eps)
        ) / (2 * d * d)
        rows.append((M, qs[0] / ker.C4, qs[1] / ker.C4, fd / ker.C4))
        log(
            f"      M = {M:4d}: Q_M/C4 = {qs[0] / ker.C4:+.8f}  (kappa_G0 -> +C4: {qs[1] / ker.C4:+.8f}; "
            f"finite-difference {fd / ker.C4:+.8f})"
        )
    Mv = np.array([r[0] for r in rows], float)
    Qv = np.array([r[1] for r in rows])
    # Q_M = Q_oo + c1/M^2 + c2/M^4 (analytic in p^2, p ~ 1/M)
    Vm = np.vstack([np.ones_like(Mv), Mv**-2.0, Mv**-4.0]).T
    coef = np.linalg.lstsq(Vm[-3:], Qv[-3:], rcond=None)[0]
    coef2 = np.linalg.lstsq(Vm[-4:-1], Qv[-4:-1], rcond=None)[0]
    indep = max(abs(r[1] - r[2]) for r in rows)
    fdev = max(abs(r[1] - r[3]) for r in rows)
    return coef[0], abs(coef[0] - coef2[0]), target, indep, fdev, rows


def scan(ker, nbz):
    P = zone_grid(nbz)
    lp, lm = bogoliubov(ker, P)
    pr, pm, pk = path_GMKG()
    lpp, lmp = bogoliubov(ker, pr)
    C = ker.C_tri()
    pn = np.linalg.norm(P, axis=1)
    i = int(np.argmin(lm))
    res = dict(
        min_lm=float(lm.min() / ker.C4),
        at=float(pn[i] / np.linalg.norm(pk)),
        min_lp=float(lp.min() / ker.C4),
        min_path=float(min(lmp.min(), lpp.min()) / ker.C4),
        M=float(bogoliubov(ker, pm[None])[1][0] / ker.C4),
        K=float(bogoliubov(ker, pk[None])[1][0] / ker.C4),
        stiff=float((ker.K0 + 2 * C) / ker.C4),
    )
    # small-p limits along a ray
    t = np.array([1e-3, 2e-3, 4e-3])[:, None] * np.linalg.norm(pk)
    ray = t * np.array([[np.cos(0.3), np.sin(0.3)]])
    a, b = bogoliubov(ker, ray)
    res["lp0"] = float(a[0] / ker.C4)
    res["lm0"] = float(b[0] / ker.C4)
    # shear branch: lambda_- / |p|^4 at moderate p, in two directions
    t2 = np.array([0.05, 0.1])[:, None] * np.linalg.norm(pk)
    c4 = []
    for th in (0.0, np.pi / 6):
        _, b2 = bogoliubov(ker, t2 * np.array([[np.cos(th), np.sin(th)]]))
        c4.append(b2 / ker.C4 / (t2[:, 0] ** 4))
    res["shear_c"] = [float(x) for x in np.concatenate(c4)]
    # unstable set
    away = pn > 0.1 * np.linalg.norm(pk)
    res["min_away"] = float(min(lm[away].min(), lp[away].min()) / ker.C4)
    res["at_away"] = float(pn[away][int(np.argmin(lm[away]))] / np.linalg.norm(pk))
    neg = lm < -1e-12 * ker.C4
    res["unstable_radius"] = (
        float(pn[neg].max() / np.linalg.norm(pk)) if neg.any() else 0.0
    )
    return res


def main():
    quick = "--quick" in sys.argv
    nbz = 120 if quick else 240
    lwc = np.load(LONGWAVE)

    # the ladder by continuation, then fresh rungs around the threshold
    rungs = CL.walk(max_beta=1e4)
    log(f"ladder solved to beta = {rungs[-1].beta:g}")

    # ---------------- [5] the threshold ----------------
    log("[5] zero of K_0/2 + C_tri (secant in ln beta, fresh solves)")
    kers = {}

    def kern(r):
        if r.beta not in kers:
            kers[r.beta] = Kernel(r)
        return kers[r.beta]

    def f_tot(r, k0=None):
        k = kern(r)
        return (0.5 * (r.K0() if k0 is None else k0) + k.C_tri()) / r.C4

    vals = [(r, f_tot(r)) for r in rungs]
    lo = max((r for r, v in vals if v > 0), key=lambda r: r.beta)
    hi = min((r for r, v in vals if v < 0), key=lambda r: r.beta)
    check(
        "[B7a] zero of K_0/2 + C_tri bracketed",
        lo.beta < hi.beta,
        f"beta in [{lo.beta:g}, {hi.beta:g}]",
    )
    x0, f0, x1, f1 = np.log(lo.beta), f_tot(lo), np.log(hi.beta), f_tot(hi)
    dfdlb = (f1 - f0) / (x1 - x0)  # bracket slope, for the error conversion
    near = sorted(rungs, key=lambda r: abs(np.log(r.beta / zr_beta_guess(x0, x1))))[:2]
    dalpha_dlb = (near[0].alpha - near[1].alpha) / np.log(near[0].beta / near[1].beta)
    zr, fz, side = None, np.inf, 0
    for _ in range(16):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        r = CL.rung_at(float(np.exp(x2)), rungs)
        f2 = f_tot(r)
        log(
            f"    beta = {r.beta:.8f}  alpha = {r.alpha:.9f}  (K_0/2 + C_tri)/C4 = {f2:+.3e}  N = {r.bg.N}"
        )
        if abs(f2) < abs(fz):
            zr, fz = r, f2
        if abs(f2) < 1e-10:
            break
        if f2 > 0:
            x0, f0 = x2, f2
            if side == +1:
                f1 *= 0.5
            side = +1
        else:
            x1, f1 = x2, f2
            if side == -1:
                f0 *= 0.5
            side = -1
    check(
        "[B7b] secant converged",
        abs(fz) < 1e-10,
        f"beta_c = {zr.beta:.4f}, alpha_c = {zr.alpha:.9f}, residual {fz:+.1e}",
    )

    # error budget, each term a shift of alpha
    conv = abs(dalpha_dlb) / abs(dfdlb)
    d_shoot = abs(f_tot(zr, zr.KG0("shoot")) - fz) * conv
    d_st = abs(f_tot(zr, zr.K0_ext()) - fz) * conv
    # (ii) collocation resolution: same beta at N + 32 and N + 64
    d_res = 0.0
    for dN in (32, 64):
        bg = CL.solve_bg(zr.beta, zr.bg.N + dN, zr.bg)
        rr = CL.Rung(zr.beta, bg)
        fr = f_tot(rr)
        d_res = max(
            d_res, abs(fr) / abs(dfdlb) * abs(dalpha_dlb), abs(rr.alpha - zr.alpha)
        )
        log(
            f"    resolution N = {bg.N}: (K_0/2 + C_tri)/C4 = {fr:+.2e}, alpha = {rr.alpha:.8f}"
        )
    # (iii) shell truncation: spline sum (all shells to gamma ~ 130) vs dk_selection (weight floor 1e-9)
    ct_sel, _ = zr.cells()
    ct_full = kern(zr).C_tri()
    d_sh = abs(ct_full - ct_sel) / zr.C4 / abs(dfdlb) * abs(dalpha_dlb)
    # (iv) interpolation on the cached ladder vs the secant
    lb = np.log(lwc["beta"])
    tot = (0.5 * lwc["K0"] + lwc["C_tri"]) / lwc["C4"]
    a_of = PchipInterpolator(lb, lwc["alpha"])
    from scipy.optimize import brentq

    xz = brentq(PchipInterpolator(lb, tot), lb[0], lb[-1])
    d_int = abs(float(a_of(xz)) - zr.alpha)
    budget = {
        "K_G0 collocation vs shooting": d_shoot,
        "K_G0 vs s->0 limit of the G != 0 kernel": d_st,
        "resolution N + 32, N + 64": d_res,
        "shell truncation": d_sh,
        "secant residual": abs(fz) * conv,
    }
    unc = float(np.sqrt(sum(v * v for v in budget.values())))
    log(
        "    error budget on alpha_c: "
        + ", ".join(f"{k} {v:.1e}" for k, v in budget.items())
        + f"; PCHIP on the cached ladder (not in the budget) {d_int:.1e}"
    )
    check(
        "[B7c] error budget below 1e-7",
        unc < 1e-7,
        f"alpha_c = {zr.alpha:.8f} +/- {unc:.0e}  (beta_c = {zr.beta:.3f}); cached-ladder zero alpha = {float(a_of(xz)):.6f}",
    )
    uni = np.load(UNIFORM)
    check(
        "[B7d] agrees with alpha_L of dk_uniform.npz",
        abs(float(uni["alpha_fo"]) - zr.alpha) < max(unc, float(uni["alpha_fo_unc"])),
        f"dk_uniform alpha_L = {float(uni['alpha_fo']):.9f} +/- {float(uni['alpha_fo_unc']):.0e}, "
        f"here {zr.alpha:.9f}",
    )
    log(
        f"    for comparison: zero of K_0 alone (dk_long_wavelength) at alpha = {float(lwc['alpha_K0zero']):.7f}"
    )

    # [B8] C_tri: spline shell sum vs dk_selection shell sum vs cache, on the cached ladder rungs
    worst = 0.0
    for r in rungs:
        i = int(np.argmin(abs(lwc["beta"] - r.beta)))
        if abs(lwc["beta"][i] / r.beta - 1) > 1e-9:
            continue
        worst = max(worst, abs(kern(r).C_tri() - lwc["C_tri"][i]) / r.C4)
    check(
        "[B8] C_tri reproduced on fresh rungs (vs dk_long_wavelength.npz)",
        worst < 1e-7,
        f"worst {worst:.1e} C4",
    )

    # ---------------- [1], [2], [3] at a rung inside the disputed window ----------------
    r5 = CL.rung_at(5000.0, rungs)
    k5 = kern(r5)
    err = k5.spline_error()
    check(
        "[B1] kernel spline vs fresh off-grid solves",
        err < 1e-7,
        f"max {err:.1e} C4 at beta = 5000",
    )
    log(
        f"[2] torus Hessian at beta = 5000 (alpha = {r5.alpha:.5f}): K_0/C4 = {k5.K0 / k5.C4:+.6f}, "
        f"C_tri/C4 = {k5.C_tri() / k5.C4:+.6f}"
    )
    torus_dev_5000 = torus_validation(k5)  # 24 flux quanta, beta = 5000
    log("[3] exactly modulated Abrikosov state at beta = 5000")
    qinf, qerr, target, indep, fdev, mrows = modulation(k5)
    check(
        "[B4] Q_M -> K_0 + 2 C_tri, independent of kappa_(G=0)",
        abs(qinf - target) < 2e-4 and indep < 1e-12 and fdev < 1e-6,
        f"extrapolated {qinf:+.6f} (+/- {qerr:.0e}) vs (K_0 + 2C_tri)/C4 = {target:+.6f}; "
        f"K_0/C4 alone = {k5.K0 / k5.C4:+.6f}; kappa_(G=0) dependence {indep:.0e}; finite-difference dev {fdev:.0e}",
    )

    # ---------------- [4] zone scans on both sides ----------------
    log(f"[4] Bogoliubov spectrum over the zone ({nbz}x{nbz} + Gamma-M-K-Gamma)")
    r0 = CL.Rung(0.0, CL.solve_bg(0.0, 64, None, tol=1e-12))
    rK = CL.rung_at(float(lwc["beta_K0zero"]), rungs)
    scan_betas = [
        0.0,
        1.0,
        5.0,
        20.0,
        45.0,
        1e2,
        3e2,
        1e3,
        3e3,
        rK.beta,
        5e3,
        6e3,
        zr.beta,
        7e3,
        1e4,
    ]
    extra = {0.0: r0, rK.beta: rK}
    srows = []
    log(
        "    eigenvalues lambda in the normalisation Q = lambda (|a|^2 + |b|^2); the density branch is lambda_+(0+) = 2(K_0 + 2C_tri)"
    )
    log(
        "        beta     alpha    (K0+2C)/C4  min l/C4   min l/C4 off |p|<0.1|K| (at |p|/|K|)  l-(M)/C4   l-(K)/C4   unstable |p|/|K|"
    )
    for b in scan_betas:
        r = (
            zr
            if b == zr.beta
            else extra.get(b)
            or next((x for x in rungs if abs(x.beta / b - 1) < 1e-9), None)
            or CL.rung_at(b, rungs)
        )
        k = kern(r)
        s = scan(k, nbz)
        s.update(beta=r.beta, alpha=r.alpha)
        srows.append(s)
        log(
            f"    {r.beta:9.2f}  {r.alpha:.5f}  {s['stiff']:+.6f}  {min(s['min_lm'], s['min_path']):+.2e}  "
            f"{s['min_away']:+.6f} ({s['at_away']:.3f})                {s['M']:+.6f}  {s['K']:+.6f}  "
            f"{s['unstable_radius']:.3f}"
        )
        log(
            "        shear lambda_-/(C4 |p|^4) at |p| = 0.05, 0.1 |K| (two directions): "
            + ", ".join(f"{c:.4f}" for c in s["shear_c"])
        )
    # as p -> 0 the pair {lambda_+, lambda_-} -> {0, 2 (K_0 + 2C_tri)} (labels swap where the stiffness < 0);
    # at |p| = 1e-3 |K| the O(p^2) drift of kappa(p^2) is ~1e-5 C4
    lim_dev = max(
        max(
            abs(min(s["lp0"], s["lm0"]) - min(0, 2 * s["stiff"])),
            abs(max(s["lp0"], s["lm0"]) - max(0, 2 * s["stiff"])),
        )
        for s in srows
    )
    check(
        "[B5] {lambda_+, lambda_-}(p -> 0) -> {0, 2(K_0 + 2C_tri)}",
        lim_dev < 1e-4,
        f"max deviation {lim_dev:.1e} C4 at |p| = 1e-3 |K|",
    )
    below = [s for s in srows if s["beta"] < zr.beta * (1 - 1e-6)]
    above = [s for s in srows if s["beta"] > zr.beta * (1 + 1e-6)]
    ok6 = all(s["min_lm"] > -1e-12 and s["min_path"] > -1e-12 for s in below)
    ok6 &= all(s["unstable_radius"] > 0 and s["unstable_radius"] < 0.5 for s in above)
    check(
        "[B6] stable everywhere in the zone below the threshold; above it only near p = 0",
        ok6,
        "below: min lambda/C4 = "
        + ", ".join(f"{min(s['min_lm'], s['min_path']):+.1e}" for s in below)
        + "; above: unstable |p|/|K| < "
        + ", ".join(f"{s['unstable_radius']:.3f}" for s in above),
    )

    log("[B9] probe limit: direct diagonalisation on a 36-flux-quantum torus")
    tp = torus_spectrum(kern(r0))
    log(
        f"    beta = 0: beta_A = {tp['betaA']:.10f}, momenta include M: {tp['hasM']}, K: {tp['hasK']}; "
        f"Hessian vs Bogoliubov {tp['dev']:.1e} C4; zero mode {tp['zero']:+.1e}; lowest non-zero "
        f"eigenvalue {tp['ev_min']:+.6f} C4 (zone formula over the torus momenta {tp['bogo_min']:+.6f}); "
        f"lambda_-(M): torus {tp['evM']:+.6f}, zone formula {tp['lmM']:+.6f}"
    )
    check(
        "[B9] probe-limit torus Hessian: = Bogoliubov, positive apart from the phase mode, lambda_-(M) agrees",
        tp["dev"] < 1e-13
        and abs(tp["zero"]) < 1e-12
        and tp["ev_min"] > 0
        and abs(tp["ev_min"] - tp["bogo_min"]) < 1e-9
        and tp["hasM"]
        and tp["hasK"]
        and abs(tp["evM"] - tp["lmM"]) < 1e-9
        and abs(tp["betaA"] - 1.1595953) < 1e-7,
    )

    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        sys.exit(1)
    if quick:
        log("--quick: cache not written")
        return
    np.savez(
        OUT,
        torus_probe_min=tp["ev_min"],
        torus_probe_M=tp["evM"],
        torus_probe_dev=tp["dev"],
        torus_5000_dev=torus_dev_5000,
        torus_5000_alpha=r5.alpha,
        beta_c=zr.beta,
        alpha_c=zr.alpha,
        alpha_c_unc=unc,
        budget_keys=np.array(list(budget.keys())),
        budget_vals=np.array(list(budget.values())),
        scan_beta=np.array([s["beta"] for s in srows]),
        scan_alpha=np.array([s["alpha"] for s in srows]),
        scan_stiff=np.array([s["stiff"] for s in srows]),
        scan_min_lm=np.array([s["min_lm"] for s in srows]),
        scan_min_lp=np.array([s["min_lp"] for s in srows]),
        scan_M=np.array([s["M"] for s in srows]),
        scan_K=np.array([s["K"] for s in srows]),
        scan_min_away=np.array([s["min_away"] for s in srows]),
        scan_unstable=np.array([s["unstable_radius"] for s in srows]),
        mod_M=np.array([r[0] for r in mrows]),
        mod_Q=np.array([r[1] for r in mrows]),
        mod_Qinf=qinf,
        mod_target=target,
        nbz=nbz,
        readme=(
            "Second variation of the quartic LLL functional about the triangular lattice with the "
            "exact kernel K(s; beta).  beta_c/alpha_c: zero of K_0/2 + C_tri (long-wavelength "
            "density instability), with error budget; scan_*: Bogoliubov spectrum over the magnetic "
            "Brillouin zone in units of C4, from the probe limit beta = 0 up (min lambda_-, its "
            "|p|/|K|, min lambda_+, lambda_- at M and K, density stiffness (K_0 + 2C_tri)/C4, radius "
            "of the unstable set); torus_probe_*: beta = 0 Hessian on a 36-flux-quantum torus "
            "(lowest non-zero eigenvalue, the eigenvalue at M, deviation from the block formula); "
            "torus_5000_*: the same deviation on a 24-flux-quantum torus at beta = 5000 "
            "(alpha = torus_5000_alpha); "
            "K_0 = K_{G=0} throughout; alpha_c_unc is the quadrature sum of budget_vals; mod_*: "
            "second variation along the modulated Abrikosov state on M-row tori at beta = 5000, "
            "and its M -> oo extrapolation."
        ),
    )
    log(f"saved {OUT}")
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
