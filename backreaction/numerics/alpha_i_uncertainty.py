"""Uncertainty on alpha_I, the zero of the stripe-edge Gaussian average I(beta).

alpha_I = 0.494 (paper sec. 5.2 and app. D.5) is the coupling
on the brane ladder at which

    I(beta) = int dx e^{-pi x^2} K(2 pi x^2 B_c; beta)

changes sign, i.e. where the quartic LLL functional stops being bounded below
along the stripe edge of the Bravais moduli space.  dk_moduli_ladder.py located
it by a secant in ln beta with fresh solves (N = 256), evaluating I on each rung
from a cubic spline of K on 320 geometric points gamma = s/B_c in [0.02, 44]
(extrapolated below 0.02) integrated by adaptive quadrature in x.  The steps
that are not converged solves are therefore: the spline in gamma (and its
extrapolation below gamma = 0.02, 11% of the Gaussian weight), the quadrature
(cutoff gamma = 44), the secant's residual, and the radial resolution N.  This
script re-locates the zero in independent ways and budgets each step.

  [A] cache only (dk_moduli_ladder.npz): zero of the cached I/C4 on the 18
      rungs by three interpolants -- secant (linear in ln beta on the bracket),
      PCHIP and cubic spline in ln beta, and PCHIP directly in alpha.
  [B] cache only: I/C4 on the two bracket rungs recomputed from the cached
      kernel tables with a different gamma interpolant (PCHIP, with the exact
      s -> 0 value K0 as an anchor point) and a different quadrature
      (Gauss-Legendre in x); converted to a shift of alpha_I with the slope.
  [C] fresh solves (~3 min): the rung at the cached secant zero, solved again,
      and I/C4 there by (i) the original spline + adaptive quad and (ii) exact
      kernel calls at Gauss-Legendre nodes in x with no spline at all, with
      node number and cutoff varied; two neighbouring rungs give the local
      slope dI/dalpha and a quadratic zero from exact calls alone; a Newton
      step from the cached point with the original method is the second
      location; the same rung at N + 32 gives the resolution shift.  The
      cached secant stopped at |I/C4| = 4e-7, i.e. 2e-6 above the zero in
      alpha; the fresh run corrects that rather than budgeting it.  The same
      three rungs with the finite-difference kernel (dk_kernel.BraneFD,
      Richardson M = 800/1600 and 1600/3200) give the zero by a second
      discretisation (table 4; same background, w0 and rows).

Run:  uv run python -m backreaction.numerics.alpha_i_uncertainty           (~5 min)
      uv run python -m backreaction.numerics.alpha_i_uncertainty --cache-only
"""

import sys
import time
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq

from backreaction.paths import DATA

warnings.filterwarnings("ignore", message="The occurrence of roundoff error")
T0 = time.time()
PAPER = 0.494
PAPER_PRECISE = 0.4939747  # paper app. D.5


def main():
    d = np.load(DATA / "dk_moduli_ladder.npz")
    beta, alpha, C4 = d["beta"], d["alpha"], d["C4"]
    lb, f = np.log(beta), d["I_stripe"] / d["C4"]
    ggrid, Ktab, K0 = d["kernel_gamma"], d["kernel_K"], d["K0"]
    a_ref = float(d["alpha_I_zero"])
    b_ref = float(d["beta_I_zero"])
    GAM_MAX = float(ggrid[-1])
    budget = {}

    def log(msg):
        print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)

    # ---------------------------------------------------------------------------
    # [A] interpolation of the cached I/C4 between rungs
    # ---------------------------------------------------------------------------
    k = int(np.argmax(f < 0))  # first rung with I < 0
    lo, hi = k - 1, k
    a_of_lb = PchipInterpolator(lb, alpha)
    x_sec = lb[lo] - f[lo] * (lb[hi] - lb[lo]) / (f[hi] - f[lo])
    zA = {
        "secant in ln beta (bracket)": float(a_of_lb(x_sec)),
        "PCHIP in ln beta": float(
            a_of_lb(brentq(PchipInterpolator(lb, f), lb[lo], lb[hi]))
        ),
        "cubic spline in ln beta": float(
            a_of_lb(brentq(CubicSpline(lb, f), lb[lo], lb[hi]))
        ),
        "PCHIP in alpha": float(
            brentq(PchipInterpolator(alpha, f), alpha[lo], alpha[hi])
        ),
    }
    log(
        f"[A] bracket beta = {beta[lo]:.0e} (I/C4 = {f[lo]:+.5f}) .. "
        f"{beta[hi]:.0e} ({f[hi]:+.5f}); cached secant zero alpha_I = {a_ref:.6f}"
    )
    for name, a in zA.items():
        log(f"    {name:30s} alpha_I = {a:.6f}   (vs secant: {a - a_ref:+.1e})")
    slope_cache = float(
        PchipInterpolator(alpha, f).derivative()(a_ref)
    )  # d(I/C4)/d alpha
    log(
        f"    slope d(I/C4)/d alpha at alpha_I (PCHIP on the ladder) = {slope_cache:.4f}"
    )
    interp_spread = max(abs(a - a_ref) for a in zA.values())
    log(
        f"    largest interpolant miss of the fresh-solve secant: {interp_spread:.1e} "
        "(error of a cache-only estimate, not of alpha_I itself)"
    )

    # ---------------------------------------------------------------------------
    # I from a kernel function: adaptive quad and Gauss-Legendre in x
    # ---------------------------------------------------------------------------
    def I_quad(kfun, gmax=GAM_MAX):
        val, _ = quad(
            lambda x: np.exp(-np.pi * x * x) * float(kfun(2.0 * np.pi * x * x)),
            0.0,
            np.sqrt(gmax / (2.0 * np.pi)),
            limit=200,
            epsabs=1e-12,
            epsrel=1e-10,
        )
        return 2.0 * val

    def I_gl(kfun, n, gmax=GAM_MAX):
        xm = np.sqrt(gmax / (2.0 * np.pi))
        t, w = np.polynomial.legendre.leggauss(n)
        x = 0.5 * xm * (t + 1.0)
        kv = np.asarray(kfun(2.0 * np.pi * x * x), dtype=float)
        return float(xm * np.sum(w * np.exp(-np.pi * x * x) * kv))  # 2 * (xm/2) * sum

    def variants(gam, Kv, K0v):
        """Kernel interpolants on a tabulated grid: the original cubic spline and
        PCHIP with the s -> 0 value K0 as an anchor (no extrapolation)."""
        cs = CubicSpline(gam, Kv)
        pc = PchipInterpolator(np.r_[0.0, gam], np.r_[K0v, Kv])
        return cs, pc

    # ---------------------------------------------------------------------------
    # [B] interpolant and quadrature on the cached bracket rungs
    # ---------------------------------------------------------------------------
    log("[B] cached bracket rungs: kernel interpolant and quadrature")
    dB_interp, dB_quad = 0.0, 0.0
    for j in (lo, hi):
        cs, pc = variants(ggrid, Ktab[j], K0[j])
        i_cs = I_quad(cs) / C4[j]
        i_pc = I_quad(pc) / C4[j]
        i_gl = I_gl(cs, 200) / C4[j]
        log(
            f"    beta = {beta[j]:.0e}: I/C4 cubic+quad {i_cs:+.8f} (cached "
            f"{f[j]:+.8f}), PCHIP+K0 {i_pc:+.8f}, cubic+GL200 {i_gl:+.8f}"
        )
        dB_interp = max(dB_interp, abs(i_pc - i_cs))
        dB_quad = max(dB_quad, abs(i_gl - i_cs), abs(i_cs - f[j]))
    log(
        f"    interpolant shift in I/C4 {dB_interp:.1e} -> alpha {dB_interp / abs(slope_cache):.1e}; "
        f"quadrature shift {dB_quad:.1e} -> alpha {dB_quad / abs(slope_cache):.1e}"
    )
    budget["[B] gamma interpolant (cubic vs PCHIP+K0, bracket rungs)"] = (
        dB_interp / abs(slope_cache)
    )
    budget["[B] quadrature (adaptive vs GL200, bracket rungs)"] = dB_quad / abs(
        slope_cache
    )

    # ---------------------------------------------------------------------------
    # [C] fresh solves at the zero
    # ---------------------------------------------------------------------------
    fresh = "--cache-only" not in sys.argv
    if fresh:
        from backreaction.numerics import dk_long_wavelength as CL
        from backreaction.numerics import dk_moduli_ladder as ML

        log("[C] fresh solves: walking the ladder to 3e6 for continuation")
        rungs = CL.walk(max_beta=3e6)
        r0 = CL.rung_at(b_ref, rungs)
        log(
            f"    rung at beta_I = {r0.beta:.6g}: N = {r0.bg.N}, alpha = {r0.alpha:.7f}"
        )

        def kex(rung):
            return lambda g: np.array([rung.K(float(x)) for x in np.atleast_1d(g)])

        spl0 = ML.kernel_spline(r0)[0]
        f_spl = ML.stripe_coefficient(spl0)[0] / r0.C4
        f_gl = {n: I_gl(kex(r0), n) / r0.C4 for n in (48, 64, 96)}
        f_cut = {g: I_gl(kex(r0), 96, gmax=g) / r0.C4 for g in (30.0, 60.0)}
        g320 = np.geomspace(ML.GAM_MIN, ML.GAM_MAX, ML.NGRID)
        _, pc0 = variants(g320, spl0(g320), r0.K0())
        f_pc = I_quad(pc0) / r0.C4
        log(f"    I/C4 spline + adaptive quad (original method): {f_spl:+.3e}")
        for n, v in f_gl.items():
            log(f"    I/C4 exact kernel, GL{n:<3d} in x, cutoff 44:    {v:+.3e}")
        for g, v in f_cut.items():
            log(f"    I/C4 exact kernel, GL96, cutoff gamma = {g:4.0f}:  {v:+.3e}")
        log(f"    I/C4 PCHIP + K0 anchor + adaptive quad:       {f_pc:+.3e}")

        # local slope from two neighbouring fresh rungs
        rp = CL.rung_at(b_ref * np.exp(0.1), rungs)
        rm = CL.rung_at(b_ref * np.exp(-0.1), rungs)
        fp = I_gl(kex(rp), 96) / rp.C4
        fm = I_gl(kex(rm), 96) / rm.C4
        slope = (fp - fm) / (rp.alpha - rm.alpha)
        log(
            f"    neighbours alpha = {rm.alpha:.6f}, {rp.alpha:.6f}: I/C4 = {fm:+.6f}, "
            f"{fp:+.6f}; slope = {slope:.4f} (ladder PCHIP: {slope_cache:.4f})"
        )
        # three-point (quadratic) zero through the fresh rungs, exact-call method
        fr = np.array([fm, f_gl[96], fp])
        ar = np.array([rm.alpha, r0.alpha, rp.alpha])
        c = np.polyfit(ar - r0.alpha, fr, 2)
        roots = np.roots(c).real + r0.alpha
        a_exact = float(roots[np.argmin(abs(roots - r0.alpha))])
        log(
            f"    zero of the exact-call I/C4 through the three fresh rungs: alpha_I = {a_exact:.9f}"
        )

        # resolution: the same beta at N + 32
        bgN = CL.solve_bg(b_ref, r0.bg.N + 32, r0.bg)
        rN = CL.Rung(b_ref, bgN)
        fN = I_gl(kex(rN), 96) / rN.C4
        dres = abs((rN.alpha - r0.alpha) - (fN - f_gl[96]) / slope)
        log(
            f"    N = {rN.bg.N}: alpha(beta_I) = {rN.alpha:.8f} (shift {rN.alpha - r0.alpha:+.1e}), "
            f"I/C4 = {fN:+.3e} -> alpha_I shift {dres:.1e}"
        )

        # two zero locations at the fresh rungs: Newton step from the cached
        # secant point with the original method, and the quadratic through the
        # three exact-call values.  The cached secant stopped at |I/C4| = 4e-7,
        # so both lie ~2e-6 below it; that residual is corrected, not budgeted.
        a_newton = r0.alpha - f_spl / slope
        log(
            f"    Newton step from the cached secant (spline + quad): alpha_I = {a_newton:.9f} "
            f"(cached secant residual moves the zero by {a_newton - a_ref:+.1e})"
        )

        # The second discretisation (table 4).  The same three rungs with the
        # finite-difference kernel (dk_kernel.BraneFD, Richardson M/2M) at the
        # same GL96 nodes; the zero from the quadratic through them.
        def kfd(rung, M):
            return lambda g: np.array(
                [rung.K_fd(float(x), M) for x in np.atleast_1d(g)]
            )

        f_fd = {}
        for M in (800, 1600):
            f_fd[M] = np.array([I_gl(kfd(r, M), 96) / r.C4 for r in (rm, r0, rp)])
            cf = np.polyfit(ar - r0.alpha, f_fd[M], 2)
            rf = np.roots(cf).real + r0.alpha
            a_fd = float(rf[np.argmin(abs(rf - r0.alpha))])
            log(
                f"    finite differences, Richardson M = {M}/{2 * M}: I/C4 at the zero "
                f"{f_fd[M][1]:+.3e} (collocation {f_gl[96]:+.3e}); quadratic zero "
                f"alpha_I = {a_fd:.9f} ({a_fd - a_exact:+.1e})"
            )
        a_fd_shift = a_fd - a_exact
        a_best = a_exact
        budget["[C] two zero locations (Newton/spline vs quadratic/exact calls)"] = abs(
            a_newton - a_exact
        )
        budget["[C] spline vs exact-call GL96 at the zero"] = abs(
            f_spl - f_gl[96]
        ) / abs(slope)
        budget["[C] GL nodes 48/64/96"] = (
            max(f_gl.values()) - min(f_gl.values())
        ) / abs(slope)
        budget["[C] cutoff gamma 30/44/60"] = max(
            abs(v - f_gl[96]) for v in f_cut.values()
        ) / abs(slope)
        budget["[C] PCHIP+K0 vs cubic spline at the zero"] = abs(f_pc - f_spl) / abs(
            slope
        )
        budget["[C] radial resolution N -> N+32"] = dres
        budget["[C] collocation vs finite-difference kernel (quadratic zero)"] = abs(
            a_fd_shift
        )
    else:
        log("[C] skipped (--cache-only): central value is the cached secant point,")
        log(
            "    whose recorded residual |I/C4| = 4e-7 (dk_moduli_ladder.py check [M7]) enters the budget"
        )
        a_best = a_ref
        budget["[A] cached secant residual 4e-7 / |slope|"] = 4e-7 / abs(slope_cache)

    # ---------------------------------------------------------------------------
    log("ERROR BUDGET (shift of alpha_I):")
    for name, v in sorted(budget.items(), key=lambda kv: -kv[1]):
        log(f"    {v:8.1e}  {name}")
    dom = max(budget, key=budget.get)
    unc = max(budget.values())
    # quote one significant figure, rounded up
    e = 10.0 ** np.floor(np.log10(unc))
    unc_q = float(np.ceil(unc / e) * e)
    log(f"dominant contribution: {dom}")
    digits = 7 if fresh else 6
    print(f"RESULT  alpha_I = {a_best:.{digits}f} +/- {unc_q:.0e}  (paper: {PAPER})")
    checks = {
        "[1] alpha_I agrees with the paper's 0.494 to its last digit": abs(
            a_best - PAPER
        )
        < 5e-4,
        "[2] uncertainty below half a unit in the third decimal": unc_q < 5e-4,
        "[3] every cache-only interpolant within 1e-3 of the secant": interp_spread
        < 1e-3,
    }
    if fresh:
        # app. D.5 quotes alpha_I = 0.4939747 +- 2e-7
        checks["[1'] fresh zero = 0.4939747 to 1e-7 (paper app. D.5)"] = (
            abs(a_best - PAPER_PRECISE) < 1e-7
        )
        checks["[2'] budget at most 2e-7 (paper app. D.5)"] = unc_q <= 2e-7
        checks["[4] fresh rung reproduces the cached alpha_I to 1e-9"] = (
            abs(r0.alpha - a_ref) < 1e-9
        )
        checks["[5] cached secant point within its residual of the corrected zero"] = (
            abs(a_ref - a_exact) < 5e-6
        )
        checks["[6] alpha_I with the finite-difference kernel within 1e-7"] = (
            abs(a_fd_shift) < 1e-7
        )
    ok = all(checks.values())
    for name, v in checks.items():
        print(("PASS" if v else "FAIL"), name)
    log(f"runtime {time.time() - T0:.0f} s")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
