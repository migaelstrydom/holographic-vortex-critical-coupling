"""The long-wavelength kernel K_0(beta) along the ladder, its zero, and the
quartic coefficient of the uniform one-flux-quantum lattices.

The zero of the full kernel at one harmonic, gamma = 0.3, is a proxy only, and
the throat limit of the kernel is not uniform down to gamma -> 0 (the minus
channel is massless there).  The physical long-wavelength quantity at finite
beta is

    K_0(beta) = lim_{s -> 0} K(s; beta) = K_{G=0}(beta),

the second equality being the identity of numerics/dk_uniform.py: the strict
uniform (G = 0) response at fixed temperature equals the s -> 0 limit of the
G != 0 kernel.  K_0 is therefore computed on every rung directly, as K_{G=0},
by Chebyshev collocation of the G = 0 system (dk_uniform.Uniform.colloc), and
cross-checked by shooting (Uniform.shoot) and by a cubic extrapolation of the
production kernel from gamma = s/B_c in {1e-3, 2e-3, 4e-3, 8e-3}.  A quadratic
extrapolation from gamma >= 0.02 carries a bias of 0.5-4e-6 C4; it is kept in
the cache for reference only (K0_stencil).

This module
  [1] continues the production `dk_kernel` coupled solve up the `dk_throat`
      beta ladder and evaluates K_0(beta) three ways (above);
  [2] locates the zero of K_0(beta) by secant iteration in ln beta between the
      bracketing rungs, each step a fresh background solve, and converts it
      to alpha_K0 = beta/B_c(beta)^2: there the long-range part of the vortex
      interaction turns attractive (not an instability: the long-wavelength
      density stiffness of a lattice tau is K_0 + 2 C(tau));
  [3] does the same for K(gamma = 0.3; beta), the one-harmonic proxy, so the
      two definitions can be compared;
  [4] evaluates the one-flux-quantum cell functionals C_tri(beta), C_sq(beta)
      (shell sums with the full kernel) along the ladder;
  [5] locates the zero of K_0/2 + C_tri, which is both the density-stability
      threshold of the triangular lattice and the point where the transition
      into it turns first order (alpha_L; the error budget is in
      numerics/dk_uniform.py).

CHECKS
------
[K1] K_0 by collocation and by shooting agree to 5e-10 C4 at every rung, and
     the small-gamma extrapolation of the G != 0 kernel agrees with both to
     2e-8 C4 (the gamma >= 0.02 stencil is off by up to 2e-5 C4).
[K2] Each zero is bracketed by consecutive rungs and the secant converges to
     |f| < 1e-10 (in units of C4).
[K3] beta = 45 reproduces the dk_kernel.npz grid value K(gamma = 0.02) =
     0.432163 (K/C4 = 0.18, the value quoted in paper sec. 5.1) and
     K_0 = 0.4254779, held here as literals: a regression check through the
     same kernel code, not a second method.
[K4] C_tri(beta) > 0 at every rung up to 1e8, and C_sq - C_tri > 0 at every rung.
[K5] The zero of K_0/2 + C_tri lies above the K_0 zero.
[K6] Every background solve is accepted by its own residual test.
[K7] (table 4) C_tri and C_sq on every rung by a second discretisation: the
     finite-difference kernel of dk_kernel.BraneFD (uniform sigma grid,
     Richardson M = 1600/3200) at the same shells, against collocation, to
     5e-9.  It shares the background, w0 and the generated rows with
     collocation: a second discretisation, not a second derivation.

Run:  uv run python -m backreaction.numerics.dk_long_wavelength        (~4 min)
      uv run python -m backreaction.numerics.dk_long_wavelength --quick (to 1e5; writes nothing)
"""

import contextlib
import sys
import time

import numpy as np

from backreaction import paths
from backreaction.numerics import bc_alpha as BCA
from backreaction.numerics import dk_background as DK
from backreaction.numerics import dk_kernel as KER
from backreaction.numerics import dk_selection as SEL
from backreaction.numerics import dk_throat as TH
from backreaction.numerics import dk_uniform as UNI

T0 = time.time()
CHECKS = {}
OUT = paths.data("dk_long_wavelength.npz")

# small-gamma stencil for the s -> 0 extrapolation of the G != 0 kernel.  Below
# gamma ~ 5e-4 the kernel evaluation itself loses ~1e-7 relative precision (the
# photon and metric responses become nearly gauge-degenerate), so going lower
# makes the limit worse, not better.
EXT_STENCIL = (1e-3, 2e-3, 4e-3, 8e-3)
EXT_STENCIL_ALT = (1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2)  # quartic, for the error estimate
OLD_STENCIL = (0.02, 0.04, 0.08)
# second discretisation of the G != 0 kernel at the lattice shells (table 4):
# dk_kernel.BraneFD, finite differences on a uniform sigma grid, Richardson
# between M and 2M.  Same background, w0 and generated rows as collocation.
FD_M = 1600


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


LADDER = [
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


@contextlib.contextmanager
def pinned(bg):
    orig = KER.dk.solve_colloc
    KER.dk.solve_colloc = lambda *a, **k: bg
    try:
        yield
    finally:
        KER.dk.solve_colloc = orig


def solve_bg(beta, N, guess, tol=1e-10):
    """Background at beta, continued from `guess`; the first resolution whose
    solve passes its own residual test (bg.accepted) is returned."""
    for Ntry in (N, N + 48, N + 96):
        bg = DK.solve_colloc(
            beta,
            N=Ntry,
            tol=tol,
            maxit=200,
            guess=TH._regrid(guess, Ntry) if guess is not None else None,
        )
        if bg.accepted:
            return bg
    raise RuntimeError(f"background not accepted at beta = {beta} (N up to {N + 96})")


class Rung:
    """One solved brane with its kernel evaluator (kernel values cached by s)."""

    def __init__(self, beta, bg):
        if not getattr(bg, "accepted", False):
            raise RuntimeError(f"Rung given an unaccepted background at beta = {beta}")
        self.beta, self.bg = float(beta), bg
        vals, w0 = BCA.bc_spectral_grid(bg, n_keep=1)
        w0, _ = BCA.normalise_mode(bg, w0)  # brane norm J = int p1 w0^2 dr = 1
        self.Bc = float(vals[0])
        self.alpha = self.beta / self.Bc**2
        self.w0 = w0
        with pinned(bg):
            self.br = KER.Brane(self.beta, self.Bc, w0, N=bg.N)
        self._cache = {}
        self._fd = {}
        self._fd_cache = {}
        self._uni = None
        self._kg0 = {}
        k = self.br.kernel(0.02 * self.Bc)
        self.C4 = k["C4"]
        self._cache[0.02] = k["K"]

    def K(self, gam):
        gam = float(gam)
        if gam not in self._cache:
            self._cache[gam] = self.br.kernel(gam * self.Bc)["K"]
        return self._cache[gam]

    def K_fd(self, gam, M=FD_M):
        """The G != 0 kernel by finite differences, Richardson in (M, 2M)."""
        key = (float(gam), M)
        if key not in self._fd_cache:
            for m in (M, 2 * M):
                if m not in self._fd:
                    with pinned(self.bg):
                        self._fd[m] = KER.BraneFD(self.br, m)
            s = float(gam) * self.Bc
            k1 = self._fd[M].kernel(s)["K"]
            k2 = self._fd[2 * M].kernel(s)["K"]
            self._fd_cache[key] = (4.0 * k2 - k1) / 3.0
        return self._fd_cache[key]

    def KG0(self, method="colloc"):
        """Strict uniform response K_{G=0} = lim_{s->0} K: 'colloc' or 'shoot'."""
        if method not in self._kg0:
            if self._uni is None:
                self._uni = UNI.Uniform(self.bg, self.Bc, self.w0)
            res = (
                self._uni.colloc() if method == "colloc" else self._uni.shoot(r_max=1e6)
            )
            self._kg0[method] = float(res["KG0"])
        return self._kg0[method]

    def K0(self, stencil=None):
        """K_0 = lim_{s->0} K(s).  Default: the exact value K_{G=0} by
        collocation.  With a stencil of gamma values: polynomial extrapolation
        of the G != 0 kernel through them (degree = number of points - 1)."""
        if stencil is None:
            return self.KG0("colloc")
        g = np.array(stencil, float)
        k = np.array([self.K(x) for x in g])
        return float(np.polyfit(g, k, len(g) - 1)[-1])

    def K0_ext(self, stencil=EXT_STENCIL):
        """s -> 0 extrapolation of the G != 0 kernel (second method for K_0)."""
        return self.K0(stencil)

    def cells(self):
        with pinned(self.bg):
            gap, ctri, csq, _ = SEL.gap_at(
                self.br, kernel_of_s=lambda s: self.K(s / self.Bc)
            )
        return ctri, csq

    def cells_fd(self, M=FD_M):
        """C_tri, C_sq with the finite-difference kernel at the same shells."""
        with pinned(self.bg):
            gap, ctri, csq, _ = SEL.gap_at(
                self.br, kernel_of_s=lambda s: self.K_fd(s / self.Bc, M)
            )
        return ctri, csq


def walk(max_beta=1e8):
    """Solve the ladder once (continuation), returning the rungs."""
    rungs, bg = [], None
    for beta, N in LADDER:
        if beta > max_beta:
            break
        bg = solve_bg(beta, N, bg)
        rung = Rung(beta, bg)
        rungs.append(rung)
        log(
            f"rung beta = {beta:9.1e}  N = {bg.N}  alpha = {rung.alpha:.5f}  "
            f"B_c = {rung.Bc:.4f}  K(0.02)/C4 = {rung.K(0.02) / rung.C4:+.5f}"
        )
    return rungs


def rung_at(beta, rungs):
    """Solve a new background at beta, continuing from the nearest rung."""
    near = min(rungs, key=lambda r: abs(np.log(r.beta / beta)))
    N = near.bg.N + (16 if beta > near.beta else 0)
    bg = solve_bg(beta, N, near.bg)
    return Rung(beta, bg)


def find_zero(fun, rungs, label, tol=1e-10, maxit=16):
    """Secant (regula falsi with the Illinois modification) in ln beta for
    fun(rung) = 0, bracketed by consecutive rungs; each step is a fresh
    background solve.  Returns (rung, slope d fun / d ln beta) or (None, nan)."""
    vals = [fun(r) for r in rungs]
    idx = None
    for i in range(len(rungs) - 1):
        if vals[i] > 0 and vals[i + 1] < 0:
            idx = i
            break
    if idx is None:
        check(f"[K2] zero of {label} bracketed on the ladder", False)
        return None, np.nan
    lo, hi = rungs[idx], rungs[idx + 1]
    x0, f0 = np.log(lo.beta), vals[idx]
    x1, f1 = np.log(hi.beta), vals[idx + 1]
    log(
        f"  {label}: bracket beta in [{lo.beta:.3g}, {hi.beta:.3g}], "
        f"values {f0:+.5f}, {f1:+.5f}"
    )
    best, fbest, side = None, np.inf, 0
    for _ in range(maxit):
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x2 = min(max(x2, min(x0, x1)), max(x0, x1))  # stay in the bracket
        rung = rung_at(float(np.exp(x2)), rungs)
        f2 = fun(rung)
        log(
            f"    beta = {rung.beta:.10g}  alpha = {rung.alpha:.9f}  "
            f"{label} = {f2:+.3e}  (N = {rung.bg.N})"
        )
        if abs(f2) < abs(fbest):
            best, fbest = rung, f2
        if abs(f2) < tol:
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
    slope = (vals[idx + 1] - vals[idx]) / np.log(hi.beta / lo.beta)
    check(
        f"[K2] secant converged for {label}",
        best is not None and abs(fbest) < tol,
        f"beta = {best.beta:.6f}, alpha = {best.alpha:.8f}, residual {fbest:+.2e}",
    )
    return best, slope


def main():
    quick = "--quick" in sys.argv
    rungs = walk(max_beta=1e5 if quick else 1e8)
    check(
        "[K6] every ladder background accepted",
        all(r.bg.accepted for r in rungs),
        f"{len(rungs)} rungs",
    )

    # [1] K_0(beta) along the ladder, three ways
    log("[1] K_0(beta) = K_{G=0} = lim_{s->0} K(s; beta) along the ladder")
    log(
        "       beta      alpha      B_c        K_0/C4      shoot-coll   ext-coll    "
        "0.02st-coll K(0.3)/C4    C_tri      C_sq-C_tri"
    )
    rows = []
    for r in rungs:
        k0 = r.K0()
        k0s = r.KG0("shoot")
        k0e = r.K0_ext()
        k0e2 = r.K0(EXT_STENCIL_ALT)
        k0o = r.K0(OLD_STENCIL)
        k03 = r.K(0.3)
        ctri, csq = r.cells()
        ctri_fd, csq_fd = r.cells_fd()
        rows.append(
            dict(
                beta=r.beta,
                alpha=r.alpha,
                Bc=r.Bc,
                K0=k0,
                K0_shoot=k0s,
                K0_ext=k0e,
                K0_ext_alt=k0e2,
                K0_stencil=k0o,
                K03=k03,
                C4=r.C4,
                C_tri=ctri,
                C_sq=csq,
                C_tri_fd=ctri_fd,
                C_sq_fd=csq_fd,
            )
        )
        log(
            f"  {r.beta:9.1e}  {r.alpha:.6f}  {r.Bc:9.4f}  {k0 / r.C4:+.8f}  "
            f"{(k0s - k0) / r.C4:+.1e}  {(k0e - k0) / r.C4:+.1e}  {(k0o - k0) / r.C4:+.1e}  "
            f"{k03 / r.C4:+.6f}  {ctri:+.6f}  {csq - ctri:+.6f}"
        )
    col = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    d_shoot = np.max(np.abs(col["K0_shoot"] - col["K0"]) / col["C4"])
    d_ext = np.max(np.abs(col["K0_ext"] - col["K0"]) / col["C4"])
    d_ext_alt = np.max(np.abs(col["K0_ext_alt"] - col["K0_ext"]) / col["C4"])
    d_old = np.max(np.abs(col["K0_stencil"] - col["K0"]) / col["C4"])
    check(
        "[K1] K_0: collocation, shooting and small-gamma extrapolation agree",
        d_shoot < 5e-10 and d_ext < 2e-8,
        f"max |shoot - colloc| = {d_shoot:.1e} C4, max |extrapolated - colloc| = "
        f"{d_ext:.1e} C4 (cubic vs quartic extrapolation {d_ext_alt:.1e}); "
        f"gamma >= 0.02 stencil off by up to {d_old:.1e} C4",
    )
    r45 = [r for r in rungs if r.beta == 45.0][0]
    check(
        "[K3] beta = 45 reproduces K(0.02) = 0.432163 and K_0 = 0.4254779 (kernel grid)",
        abs(r45.K(0.02) - 0.432163) < 1e-6 and abs(r45.K0() - 0.4254779323) < 1e-9,
        f"K(gamma = 0.02) = {r45.K(0.02):.6f}, K_0 = {r45.K0():.10f}",
    )

    # [2], [3] zeros
    log("[2] locating the zero of K_0(beta)")
    z0, s0 = find_zero(lambda r: r.K0() / r.C4, rungs, "K_0/C4")
    log("[3] locating the zero of K(gamma = 0.3; beta) (one-harmonic proxy)")
    z3, _ = find_zero(lambda r: r.K(0.3) / r.C4, rungs, "K(0.3)/C4", tol=1e-8)

    # [4] cell functionals
    ctri, csq = col["C_tri"], col["C_sq"]
    gaps = csq - ctri
    check(
        "[K4] C_tri > 0 and C_sq - C_tri > 0 at every rung",
        bool(np.all(ctri > 0) and np.all(gaps > 0)),
        f"min C_tri = {ctri.min():.6f} at beta = {col['beta'][int(np.argmin(ctri))]:.3g}; "
        f"min gap = {gaps.min():.6f}",
    )
    # [K7] the second discretisation at the shells
    d_tri = np.abs(col["C_tri_fd"] - ctri)
    d_sq = np.abs(col["C_sq_fd"] - csq)
    gap_c = (csq[-1] - ctri[-1]) / ctri[-1]
    gap_f = (col["C_sq_fd"][-1] - col["C_tri_fd"][-1]) / col["C_tri_fd"][-1]
    log(
        "      C_tri, C_sq: finite differences (Richardson M = "
        f"{FD_M}/{2 * FD_M}) - collocation, per rung:"
    )
    for b_, a_, c_ in zip(
        col["beta"], col["C_tri_fd"] - ctri, col["C_sq_fd"] - csq, strict=True
    ):
        log(f"        {b_:9.1e}  {a_:+.2e}  {c_:+.2e}")
    check(
        "[K7] C_tri and C_sq by finite differences agree with collocation at every rung",
        d_tri.max() < 5e-9 and d_sq.max() < 5e-9,
        f"max |dC_tri| = {d_tri.max():.1e}, max |dC_sq| = {d_sq.max():.1e} (at beta = "
        f"{col['beta'][int(np.argmax(d_sq))]:.0e}); top-rung relative margin "
        f"{gap_c:.8f} (collocation), {gap_f:.8f} (FD)",
    )

    ct0 = np.nan
    if z0 is not None:
        ct0, cs0 = z0.cells()
        log(f"  at the K_0 zero: C_tri = {ct0:+.6f}, C_sq - C_tri = {cs0 - ct0:+.6f}")

    # [5] the uniform-lattice quartic coefficient K_0/2 + C_tri
    log(
        "[5] locating the zero of K_0/2 + C_tri (density stiffness of the triangular lattice)"
    )
    ztot, _ = find_zero(
        lambda r: (0.5 * r.K0() + r.cells()[0]) / r.C4, rungs, "(K_0/2 + C_tri)/C4"
    )
    check(
        "[K5] the zero of K_0/2 + C_tri lies above the zero of K_0",
        z0 is not None and ztot is not None and ztot.alpha > z0.alpha,
        f"alpha_K0 = {z0.alpha:.7f} (beta = {z0.beta:.2f}), alpha_L = {ztot.alpha:.7f} "
        f"(beta = {ztot.beta:.2f}, B_c = {ztot.Bc:.3f})"
        if (z0 and ztot)
        else "",
    )
    check(
        "[K6] every secant background accepted",
        all(x is None or x.bg.accepted for x in (z0, z3, ztot)),
    )

    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        sys.exit(1)
    if quick:
        log("--quick: nothing written")
        log("ALL CHECKS PASSED")
        return
    np.savez(
        OUT,
        beta=col["beta"],
        alpha=col["alpha"],
        B_c=col["Bc"],
        K0=col["K0"],
        K0_shoot=col["K0_shoot"],
        K0_ext=col["K0_ext"],
        K0_ext_alt=col["K0_ext_alt"],
        K0_alt=col[
            "K0_ext"
        ],  # alias read by poster_density (second determination of K0)
        K0_stencil=col["K0_stencil"],
        K03=col["K03"],
        C4=col["C4"],
        C_tri=ctri,
        C_sq=csq,
        C_tri_fd=col["C_tri_fd"],
        C_sq_fd=col["C_sq_fd"],
        beta_K0zero=z0.beta,
        alpha_K0zero=z0.alpha,
        Bc_K0zero=z0.Bc,
        slope_K0_lnbeta=s0,
        C_tri_at_K0_zero=ct0,
        beta_K03zero=z3.beta if z3 else np.nan,
        alpha_K03zero=z3.alpha if z3 else np.nan,
        beta_first_order=ztot.beta,
        alpha_first_order=ztot.alpha,
        Bc_first_order=ztot.Bc,
        C_tot=0.5 * col["K0"] + ctri,
        readme=(
            "Long-wavelength kernel K0 = K_{G=0} = lim_{s->0} K(s;beta) along the "
            "dk_throat ladder: K0 by collocation of the G = 0 system "
            "(numerics/dk_uniform.py), K0_shoot by shooting, K0_ext by cubic "
            "extrapolation of the G != 0 kernel from gamma = 1e-3..8e-3 "
            "(K0_ext_alt: quartic, to 1.6e-2; K0_alt = K0_ext), K0_stencil a quadratic "
            "extrapolation from gamma = 0.02, 0.04, 0.08 (biased, reference only).  "
            "K03 = K at gamma = 0.3.  C_tri, C_sq: one-flux-quantum shell sums with "
            "the full kernel; C_tri_fd, C_sq_fd the same with the finite-difference "
            "kernel (dk_kernel.BraneFD, Richardson M = 1600/3200).  beta_K0zero/alpha_K0zero/Bc_K0zero: the zero of K0 "
            "(alpha_K0, where the long-range interaction turns attractive); "
            "beta_K03zero/alpha_K03zero: the zero of K(gamma=0.3; beta); beta_first_order/"
            "alpha_first_order/Bc_first_order: the zero of C_tot = K0/2 + C_tri "
            "(alpha_L, density instability of the triangular lattice and "
            "first-order point; error budget in dk_uniform.npz)."
        ),
    )
    log(f"saved {OUT}")
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
