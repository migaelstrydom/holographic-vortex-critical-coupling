"""Stability of every one-flux-quantum Bravais lattice on the D'Hoker-Kraus
brane: density stiffness K_0 + 2 C(tau), the stationary points of C, and the
full lowest-Landau-level Bogoliubov spectrum off the triangular point.

derivations/lll_second_variation.py shows that the second variation of the
quartic LLL functional about the uniform lattice of ANY shape tau has the
Bogoliubov blocks

    lambda_pm(p) = A(p) + S_0(p) +- |S_2(p)|,   lambda_+(p -> 0) = 2 [K_0 + 2 C(tau)],

so a lattice tau is stable against long-wavelength density modulation iff
D(tau) = K_0/2 + C(tau) > 0, and that C is stationary at tau = i and
tau = rho = e^{i pi/3} for every kernel.  Since C(tau) > C_tri for every
tau != rho below alpha_I, D(tau) > D(rho) and non-triangular lattices keep
their density stiffness beyond alpha_L.  This module quantifies that and asks
what it is worth:

  [1] the general-tau Bogoliubov formula against the triangular code of
      numerics/dk_bogoliubov.py, and C(tau) by shell sums against the exact
      cell functionals of numerics/dk_long_wavelength.py;
  [2] real space: the full Hessian of F on a rectangular magnetic torus
      (square and tau = 2i lattice states built from Landau-gauge rows) against
      the Bogoliubov multiset at every torus momentum, in the probe limit and
      on a ladder rung;
  [3] the zero of K_0/2 + C_sq along the ladder (secant in ln beta, fresh
      solves), with K_0 = K_{G=0} by collocation and C_sq by the exact shell
      sum, against K_0 by shooting / by the s -> 0 extrapolation of the G != 0
      kernel and the all-shell sum; resolution N -> N + 48;
  [4] at every rung: the critical points of C over the fundamental domain
      (only rho and i), the Hessian at rho (positive, isotropic) and at i
      (one negative eigenvalue: a saddle), and the Bogoliubov spectrum of the
      square lattice over its zone (negative at the zone-boundary point
      X = b1/2 at every rung, including the probe limit: the square lattice is
      not a stable state at any coupling);
  [5] the density thresholds of elongated lattices: the coupling at which
      D(i tau2) and D(1/2 + i tau2) change sign, for tau2 up to 50; they
      grow with tau2 towards alpha_I, as the stripe asymptotics
      D(i tau2) -> (1/2) sqrt(tau2) I(beta) requires;
  [6] whether any Bravais lattice survives above alpha_L: at couplings from
      just above alpha_L up the ladder, the largest over a grid of the compact
      fundamental domain (tau2 <= 3) of the lowest Bogoliubov eigenvalue
      (zone grid, small-p ring, and the density branch 4 D(tau)); and at
      alpha_L itself the long-wavelength shear coefficient lambda_-/p^4 of
      slightly deformed lattices, negative while the triangle's is positive
      (the deformed lattice's shear mode couples to its density mode, whose
      stiffness D vanishes at the triangular point at alpha_L).

CHECKS
------
[L1] general-tau formula = triangular code at rho (1e-12 C4); C(rho), C(i)
     from the spline = exact cell functionals (2e-9 C4)
[L2] square and tau = 2i torus states: Abrikosov ratio (square 1.1803406),
     |omega_G|^2 = e^{-gamma_G/2}; Hessian spectrum = Bogoliubov multiset
     (1e-9 C4); the most negative torus eigenvalue = lambda_-(X)
[L3] square zero bracketed, secant converged (|f| < 1e-10), methods agree,
     error budget below 1e-6 in alpha
[L4] at every rung: critical points of C on the domain only at rho and i;
     Hessian at rho positive; Hessian at i indefinite; lambda_-(X) < 0 for the
     square lattice
[L5] density thresholds increase with tau2 to tau2 = 5 and sit within 1e-4
     of alpha_I beyond; the tau2 = 1 threshold reproduces [3]; D(i 50) matches
     (1/2) sqrt(50) I
[L6] above alpha_L no lattice on the grid is stable (largest lowest
     eigenvalue < 0 at every coupling sampled); at alpha_L the deformed
     lattices have negative long-wavelength shear coefficient

Run:  uv run python -m backreaction.numerics.dk_lattice_stiffness   (~8 min)
"""

import sys
import time

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from backreaction import paths
from backreaction.numerics import dk_bogoliubov as BG
from backreaction.numerics import dk_long_wavelength as CL
from backreaction.numerics import dk_moduli as M
from backreaction.numerics import dk_moduli_ladder as ML

T0 = time.time()
CHECKS = {}
OUT = paths.data("dk_lattice_stiffness.npz")
RHO, TAU_SQ = M.RHO, M.TAU_SQ
EDGE_TAU2 = np.concatenate([[1.0], np.geomspace(1.02, 50.0, 60)])
THRESH_TAU2 = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0)
ALPHA_I = 0.4939747  # zero of I(beta), numerics/alpha_i_uncertainty.py
ALPHA_L_BETA = 6510.485393  # beta at alpha_L, numerics/dk_uniform.py


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# general-tau lattice geometry, B = 1, cell area 2 pi
# ---------------------------------------------------------------------------
def basis(tau):
    t1, t2 = tau.real, tau.imag
    ell = np.sqrt(2 * np.pi / t2)
    A = np.array([[ell, 0.0], [ell * t1, ell * t2]])  # rows a1, a2
    Bm = 2 * np.pi * np.linalg.inv(A)  # columns b1, b2
    return A[0], A[1], Bm[:, 0], Bm[:, 1]


def recip(tau, gcut=BG.GAMMA_CUT + 40):
    _, _, b1, b2 = basis(tau)
    nmax = (
        int(np.ceil(np.sqrt(gcut) / min(np.linalg.norm(b1), np.linalg.norm(b2)) * 2))
        + 2
    )
    m, n = np.meshgrid(
        np.arange(-nmax, nmax + 1), np.arange(-nmax, nmax + 1), indexing="ij"
    )
    G = m.reshape(-1, 1) * b1 + n.reshape(-1, 1) * b2
    return G[np.sum(G * G, 1) <= gcut]


def bogoliubov_tau(kap, tau, P, kG0=None):
    """lambda_pm at momenta P for the lattice tau (derivation [4], [7])."""
    P = np.atleast_2d(P)
    G = recip(tau)
    G2 = np.sum(G * G, 1)
    nz = G2 > 1e-12
    kGG = np.exp(-G2[nz] / 2) * kap(G2[nz])
    lp, lm = np.empty(len(P)), np.empty(len(P))
    for i0 in range(0, len(P), 1500):
        Pc = P[i0 : i0 + 1500]
        wGp = BG.wedge(G[None, :, :], Pc[:, None, :])
        A = np.sum(kGG[None, :] * (np.cos(wGp[:, nz]) - 1.0), 1)
        D = G[None, :, :] - Pc[:, None, :]
        D2 = np.sum(D * D, 2)
        kv = kap(D2)
        if kG0 is not None:
            kv = np.where(D2 < 1e-20, kG0, kv)
        w = np.exp(-D2 / 2) * kv
        S0 = np.sum(w, 1)
        S2 = np.abs(np.sum(w * np.exp(-1j * wGp), 1))
        lp[i0 : i0 + 1500] = A + S0 + S2
        lm[i0 : i0 + 1500] = A + S0 - S2
    return lp, lm


def C_tau(kap, tau):
    G = recip(tau)
    G2 = np.sum(G * G, 1)
    G2 = G2[G2 > 1e-12]
    return 0.5 * float(np.sum(np.exp(-G2 / 2) * kap(G2)))


def zone(tau, nbz):
    _, _, b1, b2 = basis(tau)
    f = (np.arange(nbz) + 0.5) / nbz - 0.5
    u, v = np.meshgrid(f, f, indexing="ij")
    return u.reshape(-1, 1) * b1 + v.reshape(-1, 1) * b2


def square_scan(ker, nbz=160):
    """Bogoliubov spectrum of the square lattice: zone minimum, lambda_-(X),
    lambda_-(M), and the small-p shear branch along x."""
    _, _, b1, b2 = basis(TAU_SQ)
    lp, lm = bogoliubov_tau(ker, TAU_SQ, zone(TAU_SQ, nbz))
    X, Mp = b1 / 2, (b1 + b2) / 2
    _, lX = bogoliubov_tau(ker, TAU_SQ, X[None])
    _, lM = bogoliubov_tau(ker, TAU_SQ, Mp[None])
    t = np.array([0.02, 0.04]) * np.linalg.norm(b1)
    _, ls = bogoliubov_tau(ker, TAU_SQ, np.c_[t, 0 * t])
    return dict(
        min=float(min(lm.min(), lp.min()) / ker.C4),
        X=float(lX[0] / ker.C4),
        M=float(lM[0] / ker.C4),
        shear=float(ls[0] / ker.C4 / t[0] ** 4),
    )


def best_lattice(ker, n1=11, n2=23, t2max=3.0, nbz=48):
    """max over a tau grid of the compact fundamental domain of the lowest
    Bogoliubov eigenvalue (zone grid, a small-p ring, density branch 4D)."""
    th = np.linspace(0.0, np.pi, 91)
    ring = 0.03 * np.c_[np.cos(th), np.sin(th)]
    best = (-np.inf, None)
    for a in np.linspace(0.0, 0.5, n1):
        for b in np.linspace(np.sqrt(3) / 2, t2max, n2):
            if a * a + b * b < 1.0 - 1e-12:
                continue
            tau = complex(a, b)
            lp, lm = bogoliubov_tau(ker, tau, np.concatenate([zone(tau, nbz), ring]))
            D = 0.5 * ker.K0 + C_tau(ker, tau)
            low = min(lm.min(), lp.min(), 4 * D) / ker.C4
            if low > best[0]:
                best = (low, tau)
    return best


def shear_coefficient(ker, tau, p=1e-3):
    """min over directions of lambda_-(p)/(C4 p^4) at small p."""
    th = np.linspace(0.0, np.pi, 361)
    _, lm = bogoliubov_tau(ker, tau, p * np.c_[np.cos(th), np.sin(th)])
    return float(lm.min() / ker.C4 / p**4)


# ---------------------------------------------------------------------------
# [2] rectangular magnetic torus (tau = i tau2)
# ---------------------------------------------------------------------------
class RectTorus(BG.Torus):
    """n1 cells of width ell in x, R rows of spacing ell tau2 = 2 pi/ell;
    the rectangular lattice state has the same coefficient on every row."""

    def __init__(self, tau2, n1, R, h=0.2):
        self.tau = 1j * tau2
        self.ell = np.sqrt(2 * np.pi / tau2)
        self.n1, self.R = n1, R
        self.Lx, self.Ly = n1 * self.ell, R * 2 * np.pi / self.ell
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

    def lattice_coef(self, mod=None):
        c = np.zeros(self.N, complex)
        for nrow in range(self.R):
            c[self.n1 * nrow] = 1.0
        return c / np.sqrt(self.R)

    def momenta(self):
        a1, a2, b1, b2 = basis(self.tau)
        out, seen = [], set()
        for i in range(self.n1):
            for j in range(self.R):
                f = np.array([i / self.n1, j / self.R])
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    out.append(f[0] * b1 + f[1] * b2)
        return np.array(out)


def torus_check(ker, tau2, n1=4, R=4):
    tor = RectTorus(tau2, n1, R)
    psiL = tor.state(tor.lattice_coef())
    rho = np.abs(psiL) ** 2
    betaA = float(np.mean(rho**2) / np.mean(rho) ** 2)
    rk = tor.ft(rho)
    errs = []
    for G in recip(tor.tau, 30.0):
        i = int(round(G[0] * tor.Lx / (2 * np.pi))) % tor.nx
        j = int(round(G[1] * tor.Ly / (2 * np.pi))) % tor.ny
        errs.append(abs(abs(rk[i, j]) ** 2 - np.exp(-(G @ G) / 2)))
    kv = tor.kap_grid(ker, ker.K0)
    eps = tor.eps_of(psiL, kv)
    U = np.concatenate([tor.basis, 1j * tor.basis])
    Mh = tor.quad(psiL, kv, eps, U)
    ev = np.sort(np.linalg.eigvalsh(0.5 * (Mh + Mh.T)))
    P = tor.momenta()
    isG = np.sum(P * P, 1) < 1e-20
    lp, lm = bogoliubov_tau(ker, tor.tau, P[~isG])
    C = C_tau(ker, tor.tau)
    pred = np.sort(np.concatenate([lp, lm, [0.0, 2 * (ker.K0 + 2 * C)]]))
    _, _, b1, _ = basis(tor.tau)
    _, lX = bogoliubov_tau(ker, tor.tau, (b1 / 2)[None])
    return dict(
        betaA=betaA,
        omega=max(errs),
        dev=float(np.max(np.abs(ev - pred)) / ker.C4),
        ev_min=float(ev[0] / ker.C4),
        lX=float(lX[0] / ker.C4),
        lG=float(2 * (ker.K0 + 2 * C) / ker.C4),
        N=tor.N,
    )


# ---------------------------------------------------------------------------
# [4] critical points of C over the fundamental domain
# ---------------------------------------------------------------------------
def critical_points(ker, n1=26, n2=44, t2max=3.0):
    """Critical points of C over tau1 in [0, 1/2], tau2 <= t2max inside the
    fundamental domain.  Every grid local minimum of |grad C|^2 is followed
    by a Newton iteration on grad C (finite-difference Hessian), kept inside
    0.6 < tau2 < 6.  Returns (converged points with their |grad C|/|C|,
    worst relative gradient among the candidates that did not converge)."""
    h = 1e-4

    def Cf(a, b):
        return C_tau(ker, complex(a, b))

    def grad(a, b):
        return np.array(
            [
                (Cf(a + h, b) - Cf(a - h, b)) / (2 * h),
                (Cf(a, b + h) - Cf(a, b - h)) / (2 * h),
            ]
        )

    t1s = np.linspace(0.0, 0.5, n1)
    t2s = np.linspace(np.sqrt(3) / 2, t2max, n2)
    g2 = np.full((n1, n2), np.nan)
    for i, a in enumerate(t1s):
        for j, b in enumerate(t2s):
            if a * a + b * b >= 1.0 - 1e-12:
                g2[i, j] = float(np.sum(grad(a, b) ** 2))
    found, unconv = [], np.inf
    for i in range(n1):
        for j in range(n2):
            if np.isnan(g2[i, j]):
                continue
            nb = g2[max(i - 1, 0) : i + 2, max(j - 1, 0) : j + 2]
            if g2[i, j] > np.nanmin(nb):
                continue
            x = np.array([t1s[i], t2s[j]])
            ok = False
            for _ in range(40):
                g = grad(*x)
                if np.linalg.norm(g) / abs(Cf(*x)) < 1e-10:
                    ok = True
                    break
                H = np.column_stack(
                    [
                        (grad(x[0] + h, x[1]) - grad(x[0] - h, x[1])) / (2 * h),
                        (grad(x[0], x[1] + h) - grad(x[0], x[1] - h)) / (2 * h),
                    ]
                )
                step = np.linalg.lstsq(0.5 * (H + H.T), -g, rcond=1e-12)[0]
                if not np.all(np.isfinite(step)):
                    break
                lam = min(1.0, 0.1 / max(np.linalg.norm(step), 1e-300))
                x = x + lam * step
                if not (0.6 < x[1] < 6.0) or abs(x[0]) > 3.0:
                    break
            rel = float(np.linalg.norm(grad(*x)) / abs(Cf(*x)))
            if ok:
                found.append((complex(x[0], x[1]), rel))
            else:
                unconv = min(unconv, float(np.sqrt(g2[i, j]) / abs(Cf(t1s[i], t2s[j]))))
    return found, unconv


def fold(t):
    """Map tau to the fundamental domain with tau1 >= 0."""
    for _ in range(50):
        t = complex(t.real - np.round(t.real), t.imag)
        if abs(t) < 1 - 1e-12:
            t = -1 / t
        else:
            break
    return complex(abs(t.real), t.imag)


# ---------------------------------------------------------------------------
def main():
    log("ladder (as numerics/dk_long_wavelength), plus the probe limit beta = 0")
    rungs = CL.walk(max_beta=1e8)
    r0 = CL.Rung(0.0, CL.solve_bg(0.0, 64, None, tol=1e-12))
    kers = {}

    def kern(r):
        if r.beta not in kers:
            kers[r.beta] = BG.Kernel(r)
        return kers[r.beta]

    # ---------------- [1] formula and shell sums ----------------
    k45 = kern(next(r for r in rungs if r.beta == 45.0))
    P = BG.zone_grid(40)
    a = BG.bogoliubov(k45, P)
    b = bogoliubov_tau(k45, RHO, P)
    dform = max(np.max(np.abs(a[0] - b[0])), np.max(np.abs(a[1] - b[1]))) / k45.C4
    r45 = next(r for r in rungs if r.beta == 45.0)
    ct, cs = r45.cells()
    dcell = max(abs(C_tau(k45, RHO) - ct), abs(C_tau(k45, TAU_SQ) - cs)) / k45.C4
    check(
        "[L1] general-tau Bogoliubov = triangular code at rho; spline shell sums = exact cells",
        dform < 1e-12 and dcell < 2e-9,
        f"formula {dform:.1e} C4, C(rho), C(i) {dcell:.1e} C4 (beta = 45)",
    )

    # ---------------- [2] rectangular torus ----------------
    log("[2] rectangular-torus Hessian (16 flux quanta)")
    tor_rows = []
    for r in (r0, CL.rung_at(1e4, rungs)):
        for t2 in (1.0, 2.0):
            d = torus_check(kern(r), t2)
            d.update(beta=r.beta, tau2=t2)
            tor_rows.append(d)
            log(
                f"    beta = {r.beta:7.1f}  tau = {t2:.0f}i: beta_A = {d['betaA']:.10f}, "
                f"max ||omega|^2 - e^(-G^2/2)| = {d['omega']:.1e}, Hessian vs Bogoliubov "
                f"{d['dev']:.1e} C4, lowest eigenvalue {d['ev_min']:+.6f} C4, lambda_-(X) = {d['lX']:+.6f} C4, "
                f"Gamma density mode 2(K_0 + 2C) = {d['lG']:+.6f} C4"
            )
    sq = [d for d in tor_rows if d["tau2"] == 1.0]
    check(
        "[L2] torus: square beta_A = 1.1803406, omega_G exact, Hessian = Bogoliubov, "
        "lowest eigenvalue = min(lambda_-(X), Gamma density mode) < 0",
        all(abs(d["betaA"] - 1.1803406) < 1e-7 for d in sq)
        and max(d["omega"] for d in tor_rows) < 1e-10
        and max(d["dev"] for d in tor_rows) < 1e-9
        and all(
            abs(d["ev_min"] - min(d["lX"], d["lG"])) < 1e-9 and d["lX"] < 0 for d in sq
        ),
    )

    # ---------------- [3] the square-lattice density zero ----------------
    log("[3] zero of K_0/2 + C_sq (density stiffness of the square lattice)")

    def f_sq(r):
        return (0.5 * r.K0() + r.cells()[1]) / r.C4

    zs, slope = CL.find_zero(f_sq, rungs, "(K_0/2 + C_sq)/C4")
    nb = [CL.rung_at(zs.beta * np.exp(e), rungs) for e in (-0.01, 0.01)]
    dF_da = (f_sq(nb[1]) - f_sq(nb[0])) / (nb[1].alpha - nb[0].alpha)
    k = kern(zs)
    budget = {
        "K_G0 collocation vs shooting": abs(0.5 * (zs.KG0("shoot") - zs.K0()) / zs.C4),
        "K_G0 vs s->0 limit of the G != 0 kernel": abs(
            0.5 * (zs.K0_ext() - zs.K0()) / zs.C4
        ),
        "exact shell sum vs spline over all shells": abs(
            C_tau(k, TAU_SQ) - zs.cells()[1]
        )
        / zs.C4,
        "secant residual": abs(f_sq(zs)),
    }
    budget = {kk: v / abs(dF_da) for kk, v in budget.items()}
    rN = CL.Rung(zs.beta, CL.solve_bg(zs.beta, zs.bg.N + 48, zs.bg))
    budget["radial resolution N -> N + 48"] = abs(f_sq(rN)) / abs(dF_da) + abs(
        rN.alpha - zs.alpha
    )
    unc_sq = float(np.sqrt(sum(v * v for v in budget.values())))
    for kk, v in budget.items():
        log(f"    delta alpha_sq from {kk:<45s} {v:.1e}")
    check(
        "[L3] square-lattice density threshold located with error below 1e-6",
        unc_sq < 1e-6,
        f"alpha_sq = {zs.alpha:.7f} +/- {unc_sq:.0e}, beta = {zs.beta:.2f}, "
        f"B_c u_H^2 = {zs.Bc:.3f}; dF/dalpha = {dF_da:+.4f}",
    )

    # ---------------- [4], [5] every rung ----------------
    log(
        "[4] moduli structure, square Bogoliubov spectrum and edge profiles at every rung"
    )
    allr = [r0] + rungs + [zs]
    allr = sorted({r.beta: r for r in allr}.values(), key=lambda r: r.beta)
    rows = []
    log(
        "      beta    alpha    D_tri/C4   D_sq/C4   Hess(rho)/C   Hess(i)/C (two)   "
        "crit. pts   sq: min/C4   l-(X)/C4   l-(M)/C4   shear c"
    )
    for r in allr:
        ker = kern(r)
        spl = ker
        Hr = np.linalg.eigvalsh(M.hessian(spl, RHO))
        Hi = np.linalg.eigvalsh(M.hessian(spl, TAU_SQ))
        cps, unconv = critical_points(ker)
        cps = [(fold(t), g) for t, g in cps]
        dist = [min(abs(t - RHO), abs(t - TAU_SQ)) for t, _ in cps]
        extra = [t for t, _ in cps if min(abs(t - RHO), abs(t - TAU_SQ)) > 1e-4]
        sqs = square_scan(ker)
        prof_r = np.array([C_tau(ker, 1j * t2) for t2 in EDGE_TAU2])
        prof_h = np.array(
            [C_tau(ker, 0.5 + 1j * t2) for t2 in np.maximum(EDGE_TAU2, np.sqrt(3) / 2)]
        )
        Ival = ML.stripe_coefficient(ker.spl)[0]
        ctri, csq = C_tau(ker, RHO), C_tau(ker, TAU_SQ)
        row = dict(
            beta=r.beta,
            alpha=r.alpha,
            C4=r.C4,
            K0=ker.K0,
            C_tri=ctri,
            C_sq=csq,
            hess_rho=Hr,
            hess_i=Hi,
            ncrit=len(cps),
            crit_dist=max(dist) if dist else np.nan,
            crit_near_rho=any(abs(t - RHO) < 1e-4 for t, _ in cps),
            crit_near_i=any(abs(t - TAU_SQ) < 1e-4 for t, _ in cps),
            unconv=unconv,
            extra=extra,
            crit_dist_rho_i=max([d for d in dist if d < 1e-4], default=np.nan),
            sq=sqs,
            D_rect=(0.5 * ker.K0 + prof_r) / r.C4,
            D_rhomb=(0.5 * ker.K0 + prof_h) / r.C4,
            I=Ival,
        )
        rows.append(row)
        log(
            f"    {r.beta:8.3g}  {r.alpha:.5f}  {(0.5 * ker.K0 + ctri) / r.C4:+.6f}  "
            f"{(0.5 * ker.K0 + csq) / r.C4:+.6f}  {Hr[0] / ctri:+.4f} {Hr[1] / ctri:+.4f}  "
            f"{Hi[0] / csq:+.4f} {Hi[1] / csq:+.4f}  {len(cps)} (max dist {row['crit_dist']:.0e})  "
            f"{sqs['min']:+.5f}  {sqs['X']:+.6f}  {sqs['M']:+.6f}  {sqs['shear']:+.4f}"
        )
    for x in rows:
        if x["extra"]:
            log(
                f"    beta = {x['beta']:.3g} (I/C4 = {x['I'] / x['C4']:+.5f}): further critical "
                "point(s) at tau = " + ", ".join(f"{t:.4f}" for t in x["extra"])
            )
    check(
        "[L4] C has critical points only at rho and i at every rung with I > 0; with I < 0 "
        "the others lie on the ridge tau2 > 3.5 before the unbounded stripe (flat in tau1)",
        all(x["crit_near_rho"] and x["crit_near_i"] for x in rows)
        and all(not x["extra"] for x in rows if x["I"] > 0)
        and all(t.imag > 3.5 for x in rows for t in x["extra"])
        and min(x["unconv"] for x in rows) > 1e-3,
        f"max distance of the rho, i critical points from rho, i: "
        f"{np.nanmax([x['crit_dist_rho_i'] for x in rows]):.1e}; "
        f"smallest |grad C|/C at a non-converging grid minimum: {min(x['unconv'] for x in rows):.1e}",
    )
    check(
        "[L4] Hessian of C: positive at rho (isotropic), indefinite at i (saddle), every rung",
        all(x["hess_rho"][0] > 0 for x in rows)
        and all(
            abs(x["hess_rho"][1] - x["hess_rho"][0]) / x["hess_rho"][0] < 1e-4
            for x in rows
        )
        and all(x["hess_i"][0] < 0 < x["hess_i"][1] for x in rows),
        f"Hess(i) eigenvalues / C_sq from ({rows[0]['hess_i'][0] / rows[0]['C_sq']:+.3f}, "
        f"{rows[0]['hess_i'][1] / rows[0]['C_sq']:+.3f}) at beta = 0 to "
        f"({rows[-1]['hess_i'][0] / rows[-1]['C_sq']:+.3f}, {rows[-1]['hess_i'][1] / rows[-1]['C_sq']:+.3f}) at 1e8",
    )
    check(
        "[L4] the square lattice is Bogoliubov-unstable at X at every rung (probe included); "
        "below its density threshold X is the zone minimum",
        all(x["sq"]["X"] < 0 for x in rows)
        and all(
            x["sq"]["min"] >= x["sq"]["X"] - 1e-9
            for x in rows
            if x["alpha"] < zs.alpha - 1e-6
        ),
        "lambda_-(X)/C4 from "
        f"{rows[0]['sq']['X']:+.5f} (beta = 0) to {rows[-1]['sq']['X']:+.5f} (beta = 1e8)",
    )

    # [5] density thresholds of rectangular and rhombic lattices
    lad = [x for x in rows if x["beta"] > 0 and x["beta"] in {r.beta for r in rungs}]
    lb = np.log([x["beta"] for x in lad])
    al = np.array([x["alpha"] for x in lad])
    a_of = PchipInterpolator(lb, al)
    thr = {}
    for name, key in (("rect", "D_rect"), ("rhomb", "D_rhomb")):
        Dm = np.array([x[key] for x in lad])  # rungs x tau2
        out = []
        for t2 in THRESH_TAU2:
            if name == "rhomb" and t2 < np.sqrt(3) / 2:
                continue
            col = np.array(
                [PchipInterpolator(EDGE_TAU2, Dm[i])(t2) for i in range(len(lad))]
            )
            sgn = np.where(np.diff(np.sign(col)) != 0)[0]
            if len(sgn) == 0:
                out.append((t2, np.nan))
                continue
            fz = PchipInterpolator(lb, col)
            x0 = brentq(fz, lb[sgn[0]], lb[sgn[0] + 1])
            out.append((t2, float(a_of(x0))))
        thr[name] = out
        log(
            f"    density threshold alpha(tau2) on the {name} edge: "
            + ", ".join(f"{t:g}: {a:.6f}" for t, a in out)
        )
    rect = np.array([a for _, a in thr["rect"]])
    # monotone in tau2 up to the interpolation error of the rung grid, and -> alpha_I
    # (beyond tau2 ~ 5 the threshold oscillates about alpha_I with the Poisson remainder)
    t2a = np.array(THRESH_TAU2)
    ok5 = bool(np.all(np.diff(rect[t2a <= 5.0]) > 0))
    ok5 &= bool(np.all(np.abs(rect[t2a >= 10.0] - ALPHA_I) < 1e-4))
    ok5 &= abs(rect[0] - zs.alpha) < 2e-4
    # asymptotics: D(i tau2) / sqrt(tau2) -> I/2 at tau2 = 50
    asy = max(
        abs(x["D_rect"][-1] - (0.5 * np.sqrt(EDGE_TAU2[-1]) * x["I"] / x["C4"]))
        / abs(0.5 * np.sqrt(EDGE_TAU2[-1]) * x["I"] / x["C4"])
        for x in lad
        if abs(x["I"]) / x["C4"] > 1e-3
    )
    ok5 &= asy < 1e-3
    check(
        "[L5] density thresholds grow with elongation to tau2 = 5 and sit within 1e-4 of alpha_I beyond, "
        "tau2 = 1 matches [3], stripe asymptotics",
        ok5,
        f"rectangular: tau2 = 1 -> {rect[0]:.5f} (secant {zs.alpha:.5f}), tau2 = 50 -> {rect[-1]:.4f}; "
        f"D(50i) vs sqrt(50) I/2: {asy:.1e}",
    )

    # [6] does any lattice survive above alpha_L?
    log(
        "[6] the best Bravais lattice above alpha_L, and the shear coefficient at alpha_L"
    )
    above = [CL.rung_at(b, rungs) for b in (6600.0, 7000.0, 8000.0)]
    above += [r for r in rungs if r.beta >= 1e4]
    best_rows = []
    for r in above:
        low, tau = best_lattice(kern(r))
        best_rows.append((r.beta, r.alpha, low, tau))
        log(
            f"    beta = {r.beta:9.4g}  alpha = {r.alpha:.5f}: largest lowest eigenvalue "
            f"{low:+.2e} C4 (at tau = {tau:.3f})"
        )
    rL = CL.rung_at(ALPHA_L_BETA, rungs)
    kL = kern(rL)
    c_tri = shear_coefficient(kL, RHO, p=0.05)
    c_def = {
        f"rho + {d:.2f}{'i' if im else ''}": shear_coefficient(
            kL, RHO + (1j if im else 1) * d
        )
        for im in (True, False)
        for d in (0.03, 0.1)
    }
    log(
        f"    at alpha_L (beta = {rL.beta:.3f}): triangle lambda_-/(C4 p^4) = {c_tri:+.4f}; "
        "deformed lattices: " + ", ".join(f"{k}: {v:+.4f}" for k, v in c_def.items())
    )
    check(
        "[L6] above alpha_L no Bravais lattice on the grid is stable; at alpha_L deformed "
        "lattices have negative shear coefficient",
        all(b[2] < 0 for b in best_rows)
        and c_tri > 0
        and all(v < 0 for v in c_def.values()),
        "largest lowest eigenvalue from "
        f"{best_rows[0][2]:+.1e} (alpha = {best_rows[0][1]:.5f}) to {best_rows[-1][2]:+.1e} "
        f"(alpha = {best_rows[-1][1]:.5f})",
    )

    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        sys.exit(1)
    np.savez(
        OUT,
        beta=np.array([x["beta"] for x in rows]),
        alpha=np.array([x["alpha"] for x in rows]),
        C4=np.array([x["C4"] for x in rows]),
        K0=np.array([x["K0"] for x in rows]),
        C_tri=np.array([x["C_tri"] for x in rows]),
        C_sq=np.array([x["C_sq"] for x in rows]),
        I=np.array([x["I"] for x in rows]),
        hess_rho=np.array([x["hess_rho"] for x in rows]),
        hess_i=np.array([x["hess_i"] for x in rows]),
        sq_min=np.array([x["sq"]["min"] for x in rows]),
        sq_X=np.array([x["sq"]["X"] for x in rows]),
        sq_M=np.array([x["sq"]["M"] for x in rows]),
        edge_tau2=EDGE_TAU2,
        D_rect=np.array([x["D_rect"] for x in rows]),
        D_rhomb=np.array([x["D_rhomb"] for x in rows]),
        thresh_tau2=np.array(THRESH_TAU2),
        thresh_rect=rect,
        thresh_rhomb=np.array([a for _, a in thr["rhomb"]]),
        alpha_sq=zs.alpha,
        beta_sq=zs.beta,
        Bc_sq=zs.Bc,
        alpha_sq_unc=unc_sq,
        torus_dev=max(d["dev"] for d in tor_rows),
        best_beta=np.array([b[0] for b in best_rows]),
        best_alpha=np.array([b[1] for b in best_rows]),
        best_lowest=np.array([b[2] for b in best_rows]),
        shear_tri_at_alpha_L=c_tri,
        shear_deformed_at_alpha_L=np.array(list(c_def.values())),
        readme=(
            "Stability of one-flux-quantum Bravais lattices tau with the exact kernel. "
            "Per row (beta = 0 and the ladder): C_tri, C_sq shell sums, I the Gaussian "
            "average of the kernel, hess_rho / hess_i the eigenvalues of the Hessian of "
            "C(tau) in (tau1, tau2) at rho and i, sq_min / sq_X / sq_M the square-lattice "
            "Bogoliubov minimum over the zone and lambda_- at X = b1/2 and M = (b1+b2)/2 "
            "(units of C4, normalisation Q = lambda(|a|^2+|b|^2)), D_rect / D_rhomb the "
            "density stiffness (K0/2 + C)/C4 along tau = i tau2 and 1/2 + i tau2 on "
            "edge_tau2.  thresh_*: the coupling at which D changes sign at thresh_tau2.  "
            "alpha_sq/beta_sq/Bc_sq: zero of K0/2 + C_sq, with error alpha_sq_unc.  "
            "best_*: at couplings above alpha_L, the largest over the compact tau grid of "
            "the lowest Bogoliubov eigenvalue (C4); shear_*: lambda_-/(C4 p^4) at alpha_L "
            "for the triangle and for rho + 0.03i, rho + 0.1i, rho + 0.03, rho + 0.1."
        ),
    )
    log(f"saved {OUT}")
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
