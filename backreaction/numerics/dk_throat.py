"""Numerical verification of the AdS3 x R^2 throat and the critical coupling
alpha_* = 2/3.

Companion to derivations/dk_throat.py, which derives (all steps asserted):

    fixed point   l3^2 = 1/3,   alpha B^2 e^{-4V} = 6,
    invariant     b = B l3^2 e^{-2V} = sqrt(2/(3 alpha))       [B cancels]
    criterion     BF violation in the throat  <=>  b > 1  <=>  alpha < 2/3,
    general d     alpha_*(d) = 16 / (d (d-1) (d-2))   [d=3 <=> Wong's gamma>3/4]

This module tests those statements against the solved brane, by continuing the
`dk_background` collocation solver up a beta ladder to beta = 1e11 and solving
the paper sec. 3 onset eigenproblem on each background.

WHAT IS TESTED
--------------
 [T1] the deep interior reaches the fixed point:  beta e^{-4V(r_p)} -> 6
      (the frame-invariant form of v = B sqrt(alpha/6));
 [T2] the throat is a plateau, not a point: V' -> 0 and U'' -> 2/l3^2 = 6 over
      a growing window of ln r, whose width grows like (1/4) ln(beta/6): the
      slope of the width against ln beta across the deep rungs is 1/4;
 [T3] the self-consistent onset drives the throat to marginality:
      b_h(beta) = B_c l3^2 e^{-2V(r_p)} decreases monotonically towards 1, with
      nu = sqrt(b_h - 1) ~ A/(ln beta + d) -- the holographic-BKT form.  The
      local slope A = d ln beta / d(1/nu) is NOT constant: it has a minimum
      near ln beta = 21 and then rises towards 4 pi (paper sec. 4.6), which the
      ladder resolves and the dense horizon-shooting scan (onset_scan.npz)
      confirms rung pair by rung pair.  alpha -> 2/3 follows from [T1] +
      b_h -> 1 via alpha = 2/(3 b_h^2).
 [T4] the excited tower.  The WKB phase Phi_n = int sqrt(Omega^2) d(ln r)
      (Omega = the Liouville normal form of the SL operator in xi = ln r) was
      expected to give Phi_{n+1} - Phi_n -> pi, the signature of a
      throat-controlled tower.  It does not, and `dk_tower.py` shows why:
      the phase is truncated at r_start > r_p, so it
      measures a TRUNCATED Liouville length and its plateau is exactly

          (Phi_{n+1} - Phi_n)/pi  ->  I[r_start, r_max] / I  <  1 ,
          I = int_{r_p}^oo e^{-V}/sqrt(U) dr.

      Nothing spurious is added; a fixed piece is missing.  The tower itself is
      an ordinary Sturm-Liouville spectrum, B_n = (n pi/I)^2 (1 + O(1/n)), with
      no throat index in it.  ASSERTED here against that prediction.
 [T5] two methods: collocation vs shooting for the background,
      and the spectral GEP vs the horizon-shooting c_0 root for B_c, at
      beta = 1e4 and again on the top rung, where the shooting runs out to a
      multiple of r_c = (beta/6)^{1/4}.  The source is read off exactly,
      c_0 = w + (r/2) w' (`bc_alpha.source_coefficient`), whose remainder is
      O((r_c/r_max)^3) and is pushed below 1e-9 by shooting (cheaply, in
      ln r) to 1e4 r_c on the top rung.  A log-aware FIT that omits the r^-3,
      r^-3 ln r terms of the fixed-T frame and stops at a few hundred r_c only
      measures this r^-3 remainder, not agreement between methods.  The
      background residual and the Hamiltonian constraint are checked against
      the collocation round-off floor 10 N^4 eps (`dk_background.residual_floor`;
      the top rung's Newton residual is 1.6 N^4 eps).
 [T6] anchors: beta -> 0 reproduces B_c(0) = 5.13126764 and the paper sec. 3
      `bc_alpha.npz` grid.

NOTE ON THE LIOUVILLE NORMAL FORM ([T4]).  For (P w')' + Q w = 0 the naive WKB
phase int sqrt(Q/P) dr is NOT the right counter: in the throat it gives
sqrt(b) instead of the exact sqrt(b - 1), the classic Langer-type error.  The
fix is to change variable to xi = ln r FIRST -- the SL pair becomes
Pt = P/r, Qt = r Q -- and only then Liouville-normalise:

    Omega^2 = Qt/Pt - Pt_xixi/(2 Pt) + Pt_xi^2/(4 Pt^2),

which is exact (= b - 1) on the fixed point.  Recorded because the r-variable
version silently mis-counts by ~30% at the couplings of interest.

Cache: `dk_throat.npz` (ladder + per-beta throat diagnostics).
Figure: `backreaction/figures/dk_throat.png`.

Run:  uv run python -m backreaction.numerics.dk_throat            (~7 min)
      ... --quick   stops the ladder at beta = 1e6                    (~1 min)
"""

import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

from backreaction import paths
from backreaction.numerics import bc_alpha as bca
from backreaction.numerics import dk_background as dk

PI = np.pi
L3SQ = 1.0 / 3.0  # AdS3 radius^2 (derivations/dk_throat.py [2])
ALPHA_STAR = 2.0 / 3.0  # derivations/dk_throat.py [6]
CACHE = paths.data("dk_throat.npz")
FIGDIR = paths.FIGURES

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


CHECKS = {}


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# [T5] the second onset method on deep rungs: shooting in s = ln(r - r_p)
# ---------------------------------------------------------------------------
def c0_shoot_log(B, bg, rmax, rp=1.0, d=1e-8):
    """Source coefficient c_0(B) of the zero mode shot from the horizon in
    s = ln(r - r_p) out to r_max, read off exactly (`bc_alpha.sl_c0_shoot`)."""
    return bca.sl_c0_shoot(B, bg, rmax=rmax, rhor=rp, d=d)


def bc_shoot_log(bg, B_guess, rmax):
    """Root of c_0(B) near B_guess."""
    lo, hi = B_guess * 0.999, B_guess * 1.001
    flo, fhi = c0_shoot_log(lo, bg, rmax), c0_shoot_log(hi, bg, rmax)
    it = 0
    while flo * fhi > 0 and it < 20:
        lo *= 0.995
        hi *= 1.005
        flo, fhi = c0_shoot_log(lo, bg, rmax), c0_shoot_log(hi, bg, rmax)
        it += 1
    return brentq(lambda B: c0_shoot_log(B, bg, rmax), lo, hi, xtol=1e-12, rtol=1e-13)


# ---------------------------------------------------------------------------
# the beta ladder (Newton continuation: a cold start at large beta lands on the
# wrong branch -- recorded trap, see main())
# ---------------------------------------------------------------------------
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
    (3e8, 336),
    (1e9, 352),
    (3e9, 368),
    (1e10, 384),
    (3e10, 400),
    (1e11, 416),
]


def _sig(N):
    _, xs = dk.cheb(N)
    return (xs + 1) / 2.0


def _regrid(bg, N):
    sg = _sig(N)
    return (bg._bary(bg.A, sg), bg._bary(bg.M, sg), bg._bary(bg.Nn, sg))


def throat_diagnostics(bg, B, beta, nsamp=4000, rmax=None):
    """Local throat data on a log-r grid just outside the horizon."""
    if rmax is None:
        rmax = max(50.0, 3.0 * (beta / 6.0) ** 0.25)
    r = np.exp(np.linspace(np.log(1.0 + 1e-9), np.log(rmax), nsamp))
    f = bg.fields_r(r)
    out = dict(r=r, V=f["V"], Vp=f["Vp"], U=f["U"], W=f["W"], Wp=f["Wp"])
    out["b"] = B * L3SQ * np.exp(-2 * f["V"])  # local throat invariant
    out["inv"] = beta * np.exp(-4 * f["V"])  # -> 6 at the fixed point
    return out


def plateau_width(bg, B, beta):
    """Width in ln r of the window where the fixed-point invariant
    beta e^{-4V} holds to 5% (|inv - 6| < 0.3)."""
    diag = throat_diagnostics(bg, B, beta)
    flat = np.abs(diag["inv"] - 6.0) < 0.3
    r = diag["r"]
    return float(np.log(r[flat].max()) - np.log(r[flat].min())) if flat.any() else 0.0


def wkb_phase(bg, Bn, nsamp=40000, rmax=1e5, r_start=1.05):
    """Phi = int sqrt(Omega^2) dxi over Omega^2 > 0, xi = ln r, with Omega the
    Liouville normal form of (P w')' + Q w = 0 rewritten in xi (see module
    docstring).  Exact (= sqrt(b-1)) on the fixed point.

    The lower limit matters: as r -> r_p, P -> 0 and Omega^2 diverges, so the
    absolute phase depends on r_start.  That contribution is n-INDEPENDENT
    (checked in main() by repeating at a second r_start), so only the SPACING
    Phi_{n+1} - Phi_n is quoted -- which is what the tower test needs."""
    xi = np.linspace(np.log(r_start), np.log(rmax), nsamp)
    r = np.exp(xi)
    f = bg.fields_r(r)
    P = f["U"] * np.exp(f["W"])
    Q = Bn * np.exp(f["W"] - 2 * f["V"])
    Pt, Qt = P / r, r * Q
    d1 = np.gradient(Pt, xi)
    d2 = np.gradient(d1, xi)
    om2 = Qt / Pt - d2 / (2 * Pt) + d1**2 / (4 * Pt**2)
    m = om2 > 0
    if not m.any():
        return 0.0, (np.nan, np.nan)
    return float(np.trapezoid(np.sqrt(om2[m]), xi[m])), (
        float(xi[m].min()),
        float(xi[m].max()),
    )


def run_ladder(ladder=LADDER, n_keep=2, verbose=True):
    """Continue the background up the ladder, solving the onset on each rung."""
    rows, bg = [], None
    for beta, N in ladder:
        t0 = time.time()
        # Acceptance is explicit: the residual must reach max(tol,
        # residual_floor(N)).  A rung that does not is retried at higher N,
        # and the run fails if none does.
        ok = False
        for Ntry in (N, N + 48, N + 96):
            bg2 = dk.solve_colloc(
                beta,
                N=Ntry,
                tol=1e-10,
                maxit=200,
                guess=_regrid(bg, Ntry) if bg else None,
            )
            if bg2.accepted:
                bg, N, ok = bg2, Ntry, True
                break
            log(
                f"    beta={beta:g} N={Ntry}: residual {bg2.extra['newton_res']:.1e} "
                f"above the floor {bg2.res_floor:.1e}"
            )
        if not ok:
            raise dk.NotConverged(f"ladder background at beta = {beta:g}")
        vals, _ = bca.bc_spectral_grid(bg, n_keep=n_keep)
        B = float(vals[0])
        fh = bg.fields_r(np.array([1.0 + 1e-9]))
        b_h = float(B * L3SQ * np.exp(-2 * fh["V"][0]))
        rows.append(
            dict(
                beta=beta,
                N=N,
                B=B,
                alpha=beta / B**2,
                b_h=b_h,
                inv=float(beta * np.exp(-4 * fh["V"][0])),
                res=float(bg.ode_residual()),
                newton=float(bg.extra["newton_res"]),
                floor=float(bg.res_floor),
                con=float(bg.constraint_drift()),
                width=plateau_width(bg, B, beta),
            )
        )
        if verbose:
            r = rows[-1]
            log(
                f"    beta={beta:9.3g} N={N:4d} B_c={B:13.4f} alpha={r['alpha']:.6f} "
                f"b_h={b_h:.6f} beta e^-4V={r['inv']:.5f} res={r['res']:.1e} "
                f"({time.time() - t0:.0f}s)"
            )
    return rows, bg


# ---------------------------------------------------------------------------
# [T3] the near-critical fit  nu = A / (ln beta + d)
# ---------------------------------------------------------------------------
def fit_nu(rows, beta_min=1e7):
    sel = [r for r in rows if r["beta"] >= beta_min and r["b_h"] > 1]
    lb = np.array([np.log(r["beta"]) for r in sel])
    nu = np.sqrt(np.array([r["b_h"] - 1 for r in sel]))
    # 1/nu = (ln beta + d)/A  is linear in ln beta
    p = np.polyfit(lb, 1.0 / nu, 1)
    A = 1.0 / p[0]
    d = p[1] * A
    resid = np.max(np.abs(np.polyval(p, lb) - 1.0 / nu) * nu)  # relative in nu
    return A, d, resid, sel


def main():
    quick = "--quick" in sys.argv
    ladder = [(b, n) for b, n in LADDER if b <= (1e6 if quick else np.inf)]

    log(
        "beta ladder (Newton continuation; a cold start at large beta lands on "
        "a spurious branch -- see the trap note below)"
    )
    rows, bg_last = run_ladder(ladder)
    assert len(rows) >= 8, "ladder too short to test anything"

    beta = np.array([r["beta"] for r in rows])
    alpha = np.array([r["alpha"] for r in rows])
    b_h = np.array([r["b_h"] for r in rows])
    inv = np.array([r["inv"] for r in rows])
    Bc = np.array([r["B"] for r in rows])

    log("")
    log("[T1] the deep interior reaches the AdS3 x R^2 fixed point")
    log("      beta        alpha      b_h        beta e^{-4V(r_p)}   (-> 6)")
    for r in rows[-8:]:
        log(
            f"      {r['beta']:9.3g}  {r['alpha']:.6f}  {r['b_h']:.6f}   {r['inv']:.6f}"
        )
    # tolerances scale with how deep the ladder actually went (--quick stops at
    # 1e6, where the throat is only ~1% developed)
    deep = beta[-1] >= 1e10
    tol_fp = 1e-3 if deep else 0.1
    check(
        "[T1] beta e^{-4V} -> 6 monotonically",
        bool(np.all(np.diff(np.abs(inv[-8:] - 6.0)) < 0)),
        f"|last - 6| = {abs(inv[-1] - 6.0):.2e}",
    )
    check(
        f"[T1] fixed point reached to better than {tol_fp:g} at the top rung",
        abs(inv[-1] - 6.0) < tol_fp,
        f"{abs(inv[-1] - 6.0):.2e}",
    )
    # alpha = 2/(3 b_h^2) is NOT independent evidence: algebraically
    # 2/(3 b_h^2) = 6 alpha / (beta e^{-4V}), so it is the SAME statement as
    # [T1] rewritten.  Asserted here only as a bookkeeping identity, to
    # machine precision, so the equivalence is on the record.
    ident = np.max(np.abs(2.0 / (3 * b_h**2) - 6 * alpha / inv))
    check(
        "[T1] bookkeeping: 2/(3 b_h^2) == 6 alpha/(beta e^{-4V}) exactly",
        ident < 1e-12,
        f"max deviation {ident:.2e}",
    )
    log(
        f"      => alpha_infty read off the throat: 2/(3 b_h^2) = "
        f"{2.0 / (3 * b_h[-1] ** 2):.6f} at the top rung (-> 2/3 as b_h -> 1)"
    )

    log("")
    log("[T2] the throat is an extended plateau")
    diag = throat_diagnostics(bg_last, Bc[-1], beta[-1])
    # the plateau (where beta e^{-4V} holds to 5%) must GROW like
    # (1/4) ln(beta/6): its offset depends on the 5% threshold, its slope
    # against ln beta does not.
    widths = np.array([r["width"] for r in rows])
    deep_w = (beta >= 1e6) & (widths > 0)
    lbw, ww = np.log(beta[deep_w]), widths[deep_w]
    loc = np.diff(ww) / np.diff(lbw)  # local slope between rungs
    log(
        "      plateau width (e-folds) at beta = "
        + ", ".join(f"{b:.0e}: {w:.2f}" for b, w in zip(beta[deep_w], ww, strict=True))
    )
    log("      local slope d(width)/d(ln beta): " + ", ".join(f"{x:.3f}" for x in loc))
    if deep:
        tail = lbw >= np.log(1e9)
        wslope = np.polyfit(lbw[tail], ww[tail], 1)[0]
        half = len(loc) // 2
        check(
            "[T2] plateau width grows like (1/4) ln beta: slope -> 1/4 from below",
            abs(wslope - 0.25) < 0.015 and loc[half:].mean() > loc[:half].mean(),
            f"slope over beta >= 1e9 = {wslope:.4f}; local slopes "
            f"{loc[0]:.3f} -> {loc[-1]:.3f}",
        )
    else:
        check("[T2] plateau width grows with beta", ww[-1] > ww[0])
    # U'' -> 2/l3^2 = 6 in the throat (sampled well inside it)
    r_c = (beta[-1] / 6) ** 0.25
    rr = np.exp(np.linspace(np.log(1.3), np.log(max(1.6, 0.3 * r_c)), 600))
    ff = bg_last.fields_r(rr)
    Upp = np.gradient(ff["Up"], rr)
    upp_med = float(np.median(Upp))
    check(
        "[T2] U'' -> 2/l3^2 = 6 across the throat",
        abs(upp_med - 6.0) < (0.15 if deep else 1.5),
        f"median U'' = {upp_med:.4f}",
    )

    log("")
    log("[T3] the onset drives the throat to marginality:  b_h -> 1")
    A, dfit, resid, sel = fit_nu(rows, beta_min=1e7 if not quick else 1e4)
    log(
        f"      fit  nu = A/(ln beta + d):  A = {A:.4f},  d = {dfit:.4f},  "
        f"max rel. residual {resid:.2e}  ({len(sel)} rungs, "
        f"window beta >= {sel[0]['beta']:.0e} .. {sel[-1]['beta']:.0e})"
    )
    log(
        "      A by window: "
        + ", ".join(
            f"beta >= {bm:.0e}: {fit_nu(rows, bm)[0]:.4f}"
            for bm in (1e3, 1e4, 1e5, 1e6, 1e8, 1e9)
            if len(fit_nu(rows, bm)[3]) >= 3
        )
    )
    # The T_c exponent read off DIRECTLY in alpha (paper eq. for the effective
    # law): ln beta = s / sqrt(alpha_* - alpha) + const on the same rungs, and
    # T_c ~ sqrt(B) exp(-(s/4)/sqrt(alpha_* - alpha)) since beta ~ B^2/T^4.
    # A/(2 sqrt 3) is the same coefficient only if nu^2 = (alpha_*-alpha)/(2
    # alpha_*); on these rungs nu^2 exceeds that by 9-18%.
    al_sel = np.array([s["alpha"] for s in sel])
    lb_sel = np.log(np.array([s["beta"] for s in sel]))
    s_dir = np.polyfit(1.0 / np.sqrt(ALPHA_STAR - al_sel), lb_sel, 1)[0]
    nu_sel = np.sqrt(np.array([s["b_h"] - 1 for s in sel]))
    nu_lin = np.sqrt((ALPHA_STAR - al_sel) / (2 * ALPHA_STAR))
    c_asym = 2 * PI / np.sqrt(3.0)
    log(
        f"      direct fit ln beta vs (alpha_*-alpha)^(-1/2): slope {s_dir:.4f}, "
        f"T_c exponent {s_dir / 4:.4f} (via A/2sqrt3: {A / (2 * np.sqrt(3)):.4f}); "
        f"asymptotic 2pi/sqrt3 = {c_asym:.4f}, deficit "
        f"{1 - s_dir / 4 / c_asym:.1%}"
    )
    log(
        f"      nu / sqrt((alpha_*-alpha)/(2 alpha_*)) on the window: "
        f"{(nu_sel / nu_lin).min():.4f} .. {(nu_sel / nu_lin).max():.4f}"
    )
    if deep:
        check(
            "[T3] effective T_c exponent from the direct fit = 2.25 (window 1e7..1e11)",
            len(sel) == 9 and abs(s_dir / 4 - 2.251) < 2e-3 and abs(A - 7.450) < 2e-3,
            f"s/4 = {s_dir / 4:.4f}, A = {A:.4f}",
        )
    check(
        "[T3] b_h decreases monotonically towards 1",
        bool(np.all(np.diff(b_h) < 0)) and b_h[-1] > 1.0,
        f"b_h(top) = {b_h[-1]:.6f}",
    )
    check(
        "[T3] nu = A/(ln beta + d) fits the tail to < 1%", resid < 1e-2, f"{resid:.2e}"
    )
    # The local slope A = d ln beta / d(1/nu) between neighbouring rungs is not
    # constant: it has a minimum near ln beta = 21 and then turns up (towards
    # 4 pi, paper sec. 4.6).  Checked against the dense horizon-shooting scan
    # (onset_scan.npz, a different solver), rung pair by rung pair.
    lb = np.log(np.array([s["beta"] for s in sel]))
    inv_nu = 1.0 / np.sqrt(np.array([s["b_h"] - 1 for s in sel]))
    A_loc = np.diff(lb) / np.diff(inv_nu)
    log("      local slope A between rungs: " + ", ".join(f"{a:.3f}" for a in A_loc))
    if deep:
        k = int(np.argmin(A_loc))
        check(
            "[T3] the local slope has an interior minimum near 7.43 and then rises",
            0 < k < len(A_loc) - 1
            and abs(A_loc[k] - 7.43) < 0.02
            and A_loc[-1] > A_loc[k] + 0.02,
            f"min {A_loc[k]:.4f} at ln beta ~ {0.5 * (lb[k] + lb[k + 1]):.1f}, "
            f"last {A_loc[-1]:.4f}",
        )
        scan_path = paths.data("onset_scan.npz")
        if os.path.exists(scan_path):
            from scipy.interpolate import CubicSpline  # noqa: PLC0415

            sc = np.load(scan_path)
            o = np.argsort(sc["beta"])
            spl = CubicSpline(np.log(sc["beta"][o]), 1.0 / sc["nu"][o])
            A_scan = np.diff(lb) / np.diff(spl(lb))
            dA = float(np.max(np.abs(A_scan / A_loc - 1)))
            check(
                "[T3] local slopes agree with the horizon-shooting scan",
                dA < 1e-3,
                f"max relative difference {dA:.1e}",
            )
            # the direct T_c-exponent fit, repeated on the scan's own points in
            # the same beta window (a different solver and different nodes)
            win = (sc["beta"] >= 0.999 * sel[0]["beta"]) & (
                sc["beta"] <= 1.001 * sel[-1]["beta"]
            )
            s_scan = np.polyfit(
                1.0 / np.sqrt(ALPHA_STAR - sc["alpha"][win]), np.log(sc["beta"][win]), 1
            )[0]
            check(
                "[T3] direct T_c-exponent fit agrees with the scan's own points",
                abs(s_scan / s_dir - 1) < 1e-3,
                f"scan ({int(win.sum())} points): s/4 = {s_scan / 4:.4f} vs "
                f"ladder {s_dir / 4:.4f}",
            )
        else:
            log("      (onset_scan.npz absent -- scan comparison skipped)")
    a_extrap = 2.0 / (3 * (1 + 0.0) ** 2)
    log(
        f"      => alpha(beta -> oo) = 2/(3(1+nu^2)^2) -> {a_extrap:.6f} "
        f"= alpha_* (nu -> 0)"
    )

    log("")
    log(
        "[T4] the excited tower: WKB spacing = the TRUNCATED Liouville length "
        "ratio I[r_start, r_max]/I (dk_tower.py)"
    )
    from backreaction.numerics import dk_tower as tw  # noqa: PLC0415

    I_tot = tw.I_quad(bg_last)[0]
    # NB not bca.bc_spectral_grid: its (1e-6, 1e6) eigenvalue window sits BELOW
    # B_c itself at beta >= 1e9 and returns only three modes there, so a
    # level read from it there is mislabelled.
    vals = tw.raw_tower(bg_last, 14)
    assert len(vals) == 14, f"only {len(vals)} modes at beta = {beta[-1]:g}"
    phis, phis_alt = [], []
    for n, Bn in enumerate(vals):
        ph, rng = wkb_phase(bg_last, float(Bn), r_start=1.05)
        ph2, _ = wkb_phase(bg_last, float(Bn), r_start=1.20)
        phis.append(ph)
        phis_alt.append(ph2)
        if n < 6:
            log(
                f"      n={n} B_n={Bn:12.3f}  Phi/pi = {ph / PI:8.4f}  "
                f"xi in [{rng[0]:.2f}, {rng[1]:.2f}]"
            )
    sp_ = np.diff(np.array(phis)) / PI
    sp_alt = np.diff(np.array(phis_alt)) / PI
    pred = tw.I_segment(bg_last, 1.05, 1e5) / I_tot
    pred_alt = tw.I_segment(bg_last, 1.20, 1e5) / I_tot
    log(
        f"      spacing/pi (r_start=1.05): {sp_[0]:.4f} -> {sp_[-1]:.4f};  "
        f"predicted I_trunc/I = {pred:.4f}"
    )
    log(
        f"      spacing/pi (r_start=1.20): {sp_alt[0]:.4f} -> {sp_alt[-1]:.4f};  "
        f"predicted I_trunc/I = {pred_alt:.4f}"
    )
    check(
        "[T4] the WKB spacing rises monotonically in n",
        bool(np.all(np.diff(sp_) > -1e-3)),
        f"spacing/pi runs {sp_[0]:.4f} -> {sp_[-1]:.4f}",
    )
    # The dk_tower.py prediction, at both start radii.  The residual is the
    # O(1/n) tail of the tower law, so the tolerance is set by n = 13.
    err = max(abs(sp_[-1] - pred), abs(sp_alt[-1] - pred_alt))
    check(
        "[T4] the plateau equals I[r_start, r_max]/I (a truncation of the phase integral)",
        err < 6e-3,
        f"worst |measured - predicted| = {err:.2e} at n = 13",
    )

    log("")
    log("[T5] two methods")
    b_x = 1e4
    bg_c = None
    for bb, NN in [(bb, NN) for bb, NN in LADDER if bb <= b_x]:
        bg_c = dk.solve_colloc(
            bb,
            N=NN,
            tol=1e-11,
            maxit=200,
            guess=_regrid(bg_c, NN) if bg_c else None,
            require=True,
        )
    bg_s = dk.solve_shoot(b_x, N=176)
    dmax = max(dk.profile_maxdiff(bg_c, bg_s).values())
    check(
        "[T5] background: collocation vs shooting at beta = 1e4",
        dmax < 1e-6,
        f"max profile difference {dmax:.2e}",
    )
    B_gep = float(bca.bc_spectral_grid(bg_c, n_keep=1)[0][0])
    B_sho = float(bca.bc_shoot_bg(bg_c, B_gep))
    check(
        "[T5] onset: spectral GEP vs horizon-shooting c_0 root at beta = 1e4",
        abs(B_gep - B_sho) / B_gep < 1e-9,
        f"{B_gep:.10f} vs {B_sho:.10f}  (rel {abs(B_gep - B_sho) / B_gep:.1e})",
    )

    rc = (beta[-1] / 6.0) ** 0.25
    rels = []
    for k in (100, 1000, 10000):
        Bs = bc_shoot_log(bg_last, Bc[-1], k * rc)
        rels.append(abs(Bs - Bc[-1]) / Bc[-1])
        log(
            f"      top rung beta = {beta[-1]:g}, r_max = {k} r_c: GEP "
            f"{Bc[-1]:.6f} vs log-shooting {Bs:.6f}  (rel {rels[-1]:.1e})"
        )
    # the exact extraction leaves O((r_c/r_max)^3): 1e3 per decade of r_max
    # until round-off
    check(
        "[T5] onset on the top rung: GEP vs log-shooting, converging as r_max^-3",
        rels[0] / rels[1] > 300 and rels[-1] < 1e-9,
        f"rel {rels[0]:.1e} -> {rels[1]:.1e} -> {rels[2]:.1e} at 100, 1e3, 1e4 r_c",
    )
    newt = np.array([r["newton"] for r in rows])
    floor = np.array([r["floor"] for r in rows])
    con = np.array([r["con"] for r in rows])
    i45 = int(np.argmin(np.abs(beta - 45.0)))
    log(
        f"      Newton residual / floor 10 N^4 eps: worst {np.max(newt / floor):.2f}; "
        f"Hamiltonian constraint {con[i45]:.1e} at beta = 45, {con[-1]:.1e} on the top rung"
    )
    check(
        "[T5] every rung's Newton residual is at or below the round-off floor",
        bool(np.all(newt <= np.maximum(floor, 1e-10))),
    )
    check(
        "[T5] Hamiltonian constraint below the round-off floor on every rung",
        bool(np.all(con <= np.maximum(floor, 1e-8))),
        f"max constraint/floor {np.max(con / np.maximum(floor, 1e-8)):.2f}",
    )

    log("")
    log("[T6] anchors against paper sec. 3")
    bg0 = dk.solve_colloc(0.0, N=64, tol=1e-13, require=True)
    B0 = float(bca.bc_spectral_grid(bg0, n_keep=1)[0][0])
    check(
        "[T6] beta = 0 reproduces B_c(0) = 5.13126764",
        abs(B0 - 5.13126764) < 1e-7,
        f"{B0:.8f}",
    )
    try:
        grid = np.load(paths.data("bc_alpha.npz"))
        gb, gB = grid["beta"], grid["B_c"]
        errs = []
        for bb, BB in zip(beta, Bc, strict=True):
            j = np.argmin(np.abs(gb - bb))
            if abs(gb[j] - bb) < 1e-9 * max(bb, 1.0):
                errs.append(abs(BB - gB[j]) / gB[j])
        check(
            "[T6] shared rungs agree with the bc_alpha.npz grid",
            bool(errs) and max(errs) < 1e-6,
            f"{len(errs)} shared rungs, max rel. difference {max(errs):.1e}"
            if errs
            else "no shared rungs",
        )
    except FileNotFoundError:
        log("      (bc_alpha.npz absent -- grid anchor skipped)")

    log("")
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"SUMMARY: {len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    log(f"  alpha_* = 2/3 = {ALPHA_STAR:.6f} (exact, derivations/dk_throat.py)")
    log(
        f"  ladder top: beta = {beta[-1]:.3g}, alpha = {alpha[-1]:.6f}, "
        f"b_h = {b_h[-1]:.6f}"
    )
    for k, ok in CHECKS.items():
        if not ok:
            log(f"  FAILED: {k}")
    if nfail:
        log("  cache not written")
        sys.exit(1)
    if quick:
        log("  --quick: partial ladder, cache and figure not written")
        log("ALL CHECKS PASSED")
        return
    np.savez(
        CACHE,
        beta=beta,
        alpha=alpha,
        B_c=Bc,
        b_h=b_h,
        inv=inv,
        N=np.array([r["N"] for r in rows]),
        res=np.array([r["res"] for r in rows]),
        fit_A=A,
        fit_d=dfit,
        alpha_star=ALPHA_STAR,
        wkb_phi=np.array(phis),
        wkb_B=np.array(vals, dtype=float),
        readme=(
            "Throat ladder.  b_h = B_c l3^2 "
            "e^{-2V(r_p)} (l3^2=1/3); inv = beta e^{-4V(r_p)} -> 6 at "
            "the AdS3xR^2 fixed point; alpha = 2/(3 b_h^2) there.  Fit "
            "nu = sqrt(b_h-1) = A/(ln beta + d) on beta >= 1e7."
        ),
    )
    log(f"      cache written: {CACHE}")
    try:
        make_figure(beta, alpha, b_h, inv, A, dfit, diag, np.array(phis))
    except Exception as exc:  # noqa: BLE001
        log(f"      figure skipped ({type(exc).__name__}: {exc})")
    log("ALL CHECKS PASSED")


def make_figure(beta, alpha, b_h, inv, A, dfit, diag, phis):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    a = ax[0, 0]
    a.semilogx(beta, alpha, "o-", ms=3, lw=1)
    a.axhline(2 / 3, color="crimson", ls="--", lw=1)
    a.text(beta[0] * 3, 2 / 3 - 0.035, r"$\alpha_*=2/3$", color="crimson")
    a.set_xlabel(r"$\beta=\alpha B^2$")
    a.set_ylabel(r"$\alpha(\beta)=\beta/B_c^2$")
    a.set_title("the onset curve saturates at the throat criterion")

    a = ax[0, 1]
    a.loglog(beta, np.abs(inv - 6.0), "o-", ms=3, lw=1)
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"$|\beta e^{-4V(r_p)}-6|$")
    a.set_title(r"the interior reaches the AdS$_3\times\mathbb{R}^2$ fixed point")

    a = ax[1, 0]
    lb = np.log(beta)
    m = b_h > 1
    a.plot(lb[m], 1 / np.sqrt(b_h[m] - 1), "o", ms=4)
    xs = np.linspace(lb[m].min(), lb[m].max(), 50)
    a.plot(
        xs,
        (xs + dfit) / A,
        "-",
        lw=1,
        color="crimson",
        label=rf"$\nu=A/(\ln\beta+d)$, $A={A:.2f}$",
    )
    a.set_xlabel(r"$\ln\beta$")
    a.set_ylabel(r"$1/\nu$,  $\nu=\sqrt{b_h-1}$")
    a.legend(fontsize=8)
    a.set_title("holographic-BKT approach to marginality")

    a = ax[1, 1]
    r, b = diag["r"], diag["b"]
    a.semilogx(r, b, lw=1.2)
    a.axhline(1.0, color="crimson", ls="--", lw=1)
    a.set_xlabel(r"$r$")
    a.set_ylabel(r"$b(r)=B_c\,\ell_3^2 e^{-2V}$")
    a.set_ylim(0, max(1.5, float(np.nanmax(b[:10])) * 1.1))
    a.set_title("the throat plateau and its crossover")

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, "dk_throat.png")
    fig.savefig(path, dpi=140)
    log(f"      figure written: {path}")


if __name__ == "__main__":
    main()
