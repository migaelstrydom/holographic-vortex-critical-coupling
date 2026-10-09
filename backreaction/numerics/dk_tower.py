"""The excited tower of the onset operator on the D'Hoker-Kraus brane:
an ordinary Sturm-Liouville spectrum.

The tower law tested here follows from the Liouville transform of the onset
operator:

    (P w')' + B rho w = 0,  P = U e^W, rho = e^{W-2V}
      --Liouville-->  y_zz + (B - q) y = 0,   z = int e^{-V}/sqrt(U) dr,
    endpoint indices (mu_hor, mu_bnd) = (0, 1)  =>  ZERO constant offset,
    tower law      sqrt(B_n) I = n pi + O(1/n),   I = int_{r_p}^oo e^{-V}/sqrt(U) dr.

This module tests that law against the solved brane.  A WKB phase spacing
computed from a finite start radius plateaus below pi because it measures a
TRUNCATED I: the plateau equals I[r_start, r_max]/I.

WHAT IS TESTED
--------------
 [W1] I is finite and two independent quadratures agree (adaptive quad after
      sigma = 1 - t^2 vs Gauss-Jacobi with the exact (1-sigma)^{-1/2} weight);
 [W2] I is converged in the background resolution N;
 [W3] the tower law: d_n = sqrt(B_n) I/pi - n has NO constant term (the sharp
      test of the endpoint indices) and falls as C(beta)/n;
 [W4] two methods for B_n: the spectral GEP vs the horizon
      shooting c_0 root, for n = 1..6;
 [W5] the tower is NOT throat-controlled.  The throat does dominate I -- once
      b_h -> 1 the horizon sits inside it -- but through its LOGARITHMIC
      measure, I = (l3 ln r_c + O(1))/r_c with l3 = 1/sqrt3 and
      r_c = (beta/6)^{1/4} (asserted as a fitted slope).  The index nu enters
      nowhere in that.  What nu does control is N_throat = nu ln(r_c)/pi, the
      number of oscillations the throat holds AT THE MARGINAL eigenvalue, and
      that is below 1 at every rung reached -- so there is no throat tower to
      see, quite apart from its spacing law;
 [W6] no Efimov structure: the geometric ratio exp(2 pi/nu) demanded by a
      throat-controlled tower exceeds 1e5 at nu = 0.522 (beta = 1e4), while
      the measured ratios start at 2.45 and fall towards 1 as (n+1)^2/n^2;
 [W7] the WKB deficit of `dk_throat.py` [T4] is exactly I[r_start, r_max]/I,
      at two start radii and two beta.  This is what turns `dk_throat.py` [T4]
      into an assert.

NOTE.  The law is an n -> oo statement about the tower, NOT about the ground
state: B_1 misses it by ~40%, and B_1 = B_c is precisely the throat-controlled
quantity that carries the critical coupling alpha_* = 2/3 (paper sec. 4.2).
The two results are complementary, not in tension.

Cache: `dk_tower.npz`.

Run:  uv run python -m backreaction.numerics.dk_tower           (~9 min)
      ... --quick   ladder stops at beta = 1e6                  (~2 min)
"""

import sys
import time

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eig
from scipy.special import roots_jacobi

from backreaction import paths
from backreaction.numerics import bc_alpha as bca
from backreaction.numerics import dk_background as dk
from backreaction.numerics import dk_throat as th

PI = np.pi
L3SQ = 1.0 / 3.0
CACHE = paths.data("dk_tower.npz")

# beta at which the full per-beta diagnostics are run
PROBES = (1.0, 45.0, 1e4, 1e6, 1e9, 1e11)
NKEEP = 40  # tower depth for the law
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


CHECKS = {}


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# the Liouville length I = int_{r_p}^oo e^{-V}/sqrt(U) dr = int_0^1 e^{-M}/sqrt(A) dsigma
# (sigma = 1/r, V = M - ln sigma, U = A/sigma^2; the sigma^2 Jacobian cancels)
# ---------------------------------------------------------------------------
def I_quad(bg):
    """Method 1: adaptive quadrature after sigma = 1 - t^2, which resolves the
    (1-sigma)^{-1/2} horizon endpoint by hand."""

    def g(t):
        s = min(max(1.0 - t * t, 1e-14), 1.0 - 1e-14)
        f = bg.fields_sigma(np.array([s]))
        return 2.0 * t * np.exp(-f["M"][0]) / np.sqrt(f["A"][0])

    val, err = quad(g, 0.0, 1.0, limit=500, epsabs=1e-13, epsrel=1e-12)
    return val, err


def I_gaussjacobi(bg, n=200):
    """Method 2: Gauss-Jacobi with the exact endpoint weight.  A vanishes
    linearly at the horizon, so h(sigma) = e^{-M} sqrt((1-sigma)/A) is smooth
    and  I = int_0^1 h(sigma) (1-sigma)^{-1/2} dsigma  is a Jacobi quadrature
    with (alpha, beta) = (-1/2, 0) after sigma = (1+x)/2."""
    x, wt = roots_jacobi(n, -0.5, 0.0)
    s = (1.0 + x) / 2.0
    s = np.clip(s, 1e-14, 1.0 - 1e-14)
    f = bg.fields_sigma(s)
    h = np.exp(-f["M"]) * np.sqrt((1.0 - s) / f["A"])
    return float(np.sqrt(0.5) * np.sum(wt * h))


def I_segment(bg, r0, r1):
    """I restricted to r in [r0, r1] (r0 > r_p, so no endpoint singularity)."""
    val, _ = quad(
        lambda r: (
            float(np.exp(-bg.fields_r(np.array([r]))["V"][0]))
            / np.sqrt(float(bg.fields_r(np.array([r]))["U"][0]))
        ),
        r0,
        r1,
        limit=500,
    )
    return val


# ---------------------------------------------------------------------------
# the tower.  `bc_alpha.bc_spectral_grid` filters eigenvalues to (1e-6, 1e6),
# a window that is BELOW B_c itself once beta >~ 1e9, and a raw GEP on a
# singular mass matrix also emits spurious modes.  So the tower is built here
# with its own window and, more importantly, resolution-verified: only the
# leading run of eigenvalues that two background resolutions agree on is kept.
# ---------------------------------------------------------------------------
def raw_tower(bg, nmax):
    """Positive finite eigenvalues of the onset operator on bg's grid, in
    ascending order.  Same discretisation as bc_alpha.bc_spectral_grid."""
    sig, D = bg.sig, bg.D
    D2 = D @ D
    Phat = bg.A * np.exp(bg.Nn)
    Phat_s = D @ Phat
    L = -(Phat * sig)[:, None] * D2 - (Phat_s * sig - Phat)[:, None] * D
    Mm = np.diag(sig * np.exp(bg.Nn - 2 * bg.M))
    ib = bg.N
    L[ib, :] = 0.0
    L[ib, ib] = 1.0
    Mm[ib, :] = 0.0
    vals = eig(L, Mm)[0]
    v = vals.real[
        np.isfinite(vals.real) & (np.abs(vals.imag) < 1e-8 * (1 + np.abs(vals.real)))
    ]
    v = np.sort(v[v > 1e-6])
    return v[:nmax]


def resolved_tower(bg, beta, N, nmax=NKEEP, tol=1e-7):
    """The leading run of eigenvalues that two background resolutions agree on.
    Returns (B_n, n_resolved) -- this doubles as the [W3] convergence check."""
    bg2 = dk.solve_colloc(
        beta,
        N=N + 64,
        tol=1e-11,
        maxit=200,
        guess=th._regrid(bg, N + 64),
        require=True,
    )
    a, b2 = raw_tower(bg, nmax), raw_tower(bg2, nmax)
    m = min(len(a), len(b2))
    rel = np.abs(a[:m] - b2[:m]) / a[:m]
    bad = np.where(rel > tol)[0]
    k = int(bad[0]) if len(bad) else m
    return a[:k], k


def tower_law(bg, I, vals):
    """d_n = sqrt(B_n) I/pi - n, and the fit d_n = c0 + C/n over the top half."""
    n = np.arange(1, len(vals) + 1)
    d = np.sqrt(vals) * I / PI - n
    k = slice(len(vals) // 2, len(vals))
    # least squares for (c0, C) in d = c0 + C/n
    Amat = np.stack([np.ones(n[k].shape), 1.0 / n[k]], axis=1)
    c0, C = np.linalg.lstsq(Amat, d[k], rcond=None)[0]
    return vals, d, float(c0), float(C)


def main():
    quick = "--quick" in sys.argv
    top = 1e6 if quick else np.inf
    ladder = [(b, N) for b, N in th.LADDER if b <= top]
    probes = [b for b in PROBES if b <= top]

    log("continuing the brane up the beta ladder (Newton continuation)")
    bgs, Ns, bg = {}, {}, None
    for b, N in ladder:
        bg = dk.solve_colloc(
            b,
            N=N,
            tol=1e-11,
            maxit=200,
            guess=th._regrid(bg, N) if bg else None,
            require=True,
        )
        if b in probes:
            bgs[b], Ns[b] = bg, N
            log(f"    kept beta = {b:g} (N = {N})")
    assert len(bgs) == len(probes), "ladder did not reach every probe"

    # -----------------------------------------------------------------------
    log("")
    log("[W1] the Liouville length I -- two independent quadratures")
    log("       beta          I (adaptive)      I (Gauss-Jacobi)   |diff|")
    Ivals, worst_I = {}, 0.0
    for b in probes:
        Iq, eq = I_quad(bgs[b])
        Ig = I_gaussjacobi(bgs[b])
        Ivals[b] = Iq
        worst_I = max(worst_I, abs(Iq - Ig) / Iq)
        log(f"      {b:9.3g}   {Iq:.12f}    {Ig:.12f}   {abs(Iq - Ig):.2e}")
    check(
        "[W1] the two quadratures for I agree",
        worst_I < 1e-9,
        f"worst relative difference {worst_I:.2e}",
    )

    # -----------------------------------------------------------------------
    log("")
    log("[W2] I is converged in the background resolution")
    bref = probes[-2] if len(probes) > 1 else probes[0]
    Nref = dict(ladder)[bref]
    spread = []
    for dN in (0, 48, 96):
        bgN = dk.solve_colloc(
            bref,
            N=Nref + dN,
            tol=1e-11,
            maxit=200,
            guess=th._regrid(bgs[bref], Nref + dN),
            require=True,
        )
        spread.append(I_quad(bgN)[0])
        log(f"      beta = {bref:g}, N = {Nref + dN:4d}:  I = {spread[-1]:.12f}")
    rel = (max(spread) - min(spread)) / spread[0]
    check("[W2] I converged in N", rel < 1e-9, f"spread {rel:.2e}")

    # -----------------------------------------------------------------------
    log("")
    log("[W3] the tower law   sqrt(B_n) I / pi = n + C/n   (no constant term)")
    log("      the tower is resolution-verified first: only the leading run of")
    log("      eigenvalues that N and N+64 agree on to 1e-7 is used.")
    log("       beta      n_res    I          C(beta)      c0 (must vanish)   d_1")
    towers = {}
    worst_c0, worst_res = 0.0, 0.0
    for b in probes:
        raw, nres = resolved_tower(bgs[b], b, Ns[b])
        assert nres >= 8, f"only {nres} resolved modes at beta = {b:g}"
        vals, d, c0, C = tower_law(bgs[b], Ivals[b], raw)
        towers[b] = (vals, d, c0, C)
        n = np.arange(1, len(vals) + 1)
        k = slice(len(vals) // 2, len(vals))
        res = float(np.max(np.abs(d[k] - (c0 + C / n[k]))))
        worst_c0 = max(worst_c0, abs(c0))
        worst_res = max(worst_res, res)
        log(
            f"      {b:9.3g}   {len(vals):3d}   {Ivals[b]:.8f}  {C:+.6f}   "
            f"{c0:+.2e}        {d[0]:+.4f}   (fit residual {res:.1e})"
        )
    check(
        "[W3] the constant offset of the tower vanishes  (mu_hor, mu_bnd) = (0, 1)",
        worst_c0 < 1e-3,
        f"worst |c0| = {worst_c0:.2e} in n units",
    )
    check(
        "[W3] d_n = C/n fits the upper half of the tower",
        worst_res < 3e-4,
        f"worst residual {worst_res:.2e} in n units",
    )
    b0 = probes[-1]
    vals, d, c0, C = towers[b0]
    log(f"      at beta = {b0:g}:  n * d_n over the tower =")
    log(
        "        "
        + "  ".join(
            f"n={k + 1}:{(k + 1) * d[k]:+.5f}"
            for k in sorted({0, 1, len(d) // 4, len(d) // 2, len(d) - 1})
        )
    )
    check(
        "[W3] the ground state does NOT obey the law (it is the throat-controlled one)",
        abs(d[0]) > 0.1,
        f"d_1 = {d[0]:+.4f}, i.e. B_1 = {(1 + d[0]) ** 2:.2f} x the law's value",
    )

    # -----------------------------------------------------------------------
    log("")
    log("[W4] two methods for B_n: spectral GEP vs c_0 shooting")
    bx = 1e4 if 1e4 in probes else probes[-1]
    valsx = towers[bx][0]
    worst = 0.0
    for i in range(6):
        Bs = bca.bc_shoot_bg(bgs[bx], float(valsx[i]))
        rel = abs(Bs - valsx[i]) / valsx[i]
        worst = max(worst, rel)
        log(
            f"      n={i + 1}  GEP {valsx[i]:13.6f}  shoot {Bs:13.6f}  reldiff {rel:.2e}"
        )
    check(
        "[W4] GEP vs shooting agree to 1e-4 for the whole low tower",
        worst < 1e-4,
        f"worst relative difference {worst:.2e} at beta = {bx:g}",
    )

    # -----------------------------------------------------------------------
    log("")
    log("[W5] the tower is not throat-controlled")
    log("       beta      nu       r_c      I * r_c    throat share of I   N_throat")
    rc_l, Irc_l, nthr = [], [], []
    for b in probes:
        bgb = bgs[b]
        Bc = float(towers[b][0][0])
        fh = bgb.fields_r(np.array([1.0 + 1e-9]))
        b_h = Bc * L3SQ * np.exp(-2 * fh["V"][0])
        nu = np.sqrt(max(b_h - 1.0, 0.0))
        rc = (b / 6.0) ** 0.25
        if rc <= 1.5:
            log(f"      {b:9.3g}   {nu:.4f}   {rc:7.3f}   (no throat yet)")
            continue
        share = I_segment(bgb, 1.0 + 1e-6, rc) / Ivals[b]
        Nt = nu * np.log(rc) / PI
        rc_l.append(rc)
        Irc_l.append(Ivals[b] * rc)
        nthr.append(Nt)
        log(
            f"      {b:9.3g}   {nu:.4f}   {rc:7.3f}   {Ivals[b] * rc:.4f}     "
            f"{share:.4f}              {Nt:.4f}"
        )
    check(
        "[W5] the throat holds fewer than one oscillation at the marginal eigenvalue",
        max(nthr) < 1.0,
        f"max N_throat = {max(nthr):.4f} at beta = {probes[-1]:g} -- no throat tower exists",
    )
    # Beyond the rungs: by the matching condition of paper section 4.6 the
    # throat holds the phase chi + phi_* of the marginal mode, so
    # N_throat -> (chi + phi_*)/pi, which is below 1 (chi < pi/2, phi_* < pi/2)
    # and tends to 1 only as beta -> infinity, where A -> 4 pi.
    check(
        "[W5] N_throat rises monotonically along the rungs",
        bool(np.all(np.diff(np.array(nthr)) > 0)),
        "N_throat = " + ", ".join(f"{n:.4f}" for n in nthr),
    )
    log(
        "      by the matching condition N_throat -> (chi + phi_*)/pi < 1, "
        "reaching 1 only as beta -> infinity (A -> 4 pi)"
    )
    # I * r_c = l3 ln r_c + const, l3 = 1/sqrt3: the throat's LOGARITHMIC
    # measure sets the phase scale, and nu appears nowhere in it.
    if len(rc_l) >= 3:
        slope = float(
            np.polyfit(np.log(np.array(rc_l[-3:])), np.array(Irc_l[-3:]), 1)[0]
        )
        log(
            f"      fitted d(I r_c)/d(ln r_c) over the top three rungs = {slope:.5f} "
            f"vs l3 = 1/sqrt3 = {np.sqrt(L3SQ):.5f}"
        )
        check(
            "[W5] I = (l3 ln r_c + O(1))/r_c -- the throat feeds I through ln r, not nu",
            abs(slope - np.sqrt(L3SQ)) < 0.03,
            f"slope {slope:.5f}, relative error {abs(slope / np.sqrt(L3SQ) - 1):.2%}",
        )

    # -----------------------------------------------------------------------
    log("")
    log("[W6] no Efimov structure in the tower")
    ok6 = True
    for b in probes[1:]:
        vals = towers[b][0]
        Bc = float(vals[0])
        fh = bgs[b].fields_r(np.array([1.0 + 1e-9]))
        nu = np.sqrt(max(Bc * L3SQ * np.exp(-2 * fh["V"][0]) - 1.0, 0.0))
        ratios = [float(vals[i + 1] / vals[i]) for i in range(5)]
        efim = np.exp(2 * PI / nu) if nu > 0 else np.inf
        log(
            f"      beta={b:9.3g}  nu={nu:.4f}  needs e^(2pi/nu) = {efim:.3e};  "
            f"measured {', '.join(f'{x:.3f}' for x in ratios)}"
        )
        ok6 &= max(ratios) < 1e-3 * efim
    check(
        "[W6] measured level ratios fall short of the Efimov ratio by >1000x",
        ok6,
        "the tower is quadratic, (n+1)^2/n^2, not geometric",
    )

    # -----------------------------------------------------------------------
    log("")
    log("[W8] the two ends of the spectrum, and the level gap")
    log(
        "       beta      B_2/B_1   sqrt(Bc) I   sqrt(b_h) ln r_c   nu ln r_c   sqrt(b/(b-1))"
    )
    gaps = []
    for b in probes:
        vals = towers[b][0]
        Bc = float(vals[0])
        fh = bgs[b].fields_r(np.array([1.0 + 1e-9]))
        b_h = Bc * L3SQ * np.exp(-2 * fh["V"][0])
        nu = np.sqrt(max(b_h - 1.0, 0.0))
        lrc = np.log((b / 6.0) ** 0.25)
        gaps.append(float(vals[1] / vals[0]))
        log(
            f"      {b:9.3g}   {gaps[-1]:7.4f}   {np.sqrt(Bc) * Ivals[b]:9.4f}   "
            f"{np.sqrt(b_h) * lrc:12.4f}     {nu * lrc:8.4f}   "
            f"{np.sqrt(b_h / (b_h - 1)):.4f}"
        )
    # The level gap does NOT
    # stay bounded away from 1.  1 + d_1 = sqrt(B_c) I / pi grows like
    # ln(r_c)/pi (I ~ l3 ln r_c / r_c with B_c -> sqrt(3 beta/2)), so the whole
    # low tower crowds towards B_c logarithmically.  It is still 1.38 at the
    # top rung, but it is a trend, not a floor.
    check(
        "[W8] the level gap B_2/B_1 falls monotonically towards 1 (NOT a floor)",
        bool(np.all(np.diff(np.array(gaps)) < 0)) and gaps[-1] > 1.0,
        f"B_2/B_1 runs {gaps[0]:.4f} -> {gaps[-1]:.4f}; 1 + d_1 = sqrt(Bc) I/pi "
        f"grows like ln(r_c)/pi, so the gap closes logarithmically",
    )
    # The structural statement, exact on the fixed point and free of any
    # beta -> oo limit: the throat enters the
    # bottom of the spectrum through sqrt(b - 1) = nu (the BF subtraction) and
    # the top through sqrt(b).  Their ratio diverges as b_h -> 1, so no single
    # phase count can cover both ends of the spectrum.
    fh = bgs[probes[-1]].fields_r(np.array([1.0 + 1e-9]))
    b_top = float(towers[probes[-1]][0][0]) * L3SQ * np.exp(-2 * fh["V"][0])
    check(
        "[W8] the two phase counts are already 4x apart at the top rung and "
        "separate without bound as b_h -> 1",
        np.sqrt(b_top / (b_top - 1)) > 4.0,
        f"sqrt(b_h/(b_h - 1)) = {np.sqrt(b_top / (b_top - 1)):.4f} at "
        f"b_h = {b_top:.6f}",
    )

    # -----------------------------------------------------------------------
    log("")
    log("[W7] the dk_throat.py [T4] WKB deficit = I[r_start, r_max] / I")
    log("       beta      r_start   predicted   measured (n=13)   |diff|")
    worst7 = 0.0
    for b in [x for x in probes if x >= 1e4]:
        bgb = bgs[b]
        vals = towers[b][0][:14]
        for rs in (1.05, 1.20):
            pred = I_segment(bgb, rs, 1e5) / Ivals[b]
            ph = [th.wkb_phase(bgb, float(Bn), r_start=rs)[0] for Bn in vals]
            meas = float(np.diff(ph)[-1] / PI)
            worst7 = max(worst7, abs(pred - meas))
            log(
                f"      {b:9.3g}   {rs:.2f}     {pred:.4f}      {meas:.4f}"
                f"            {abs(pred - meas):.4f}"
            )
    check(
        "[W7] the WKB plateau equals the truncated Liouville length ratio",
        worst7 < 5e-3,
        f"worst |predicted - measured| = {worst7:.2e} (residual is the O(1/n) tail)",
    )

    # -----------------------------------------------------------------------
    log("")
    log("SUMMARY")
    log("  the tower is an ordinary SL spectrum: B_n = (n pi / I)^2 (1 + O(1/n))")
    log("  it is NOT throat-controlled and shows no Efimov structure")
    log("  the WKB deficit is I_trunc / I (a truncation, not a near-horizon term)")
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"  {len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    for k, ok in CHECKS.items():
        if not ok:
            log(f"  FAILED: {k}")
    if nfail:
        log("  cache not written")
        sys.exit(1)
    if quick:
        log("  --quick: partial ladder, cache not written")
        log("ALL CHECKS PASSED")
        return
    np.savez(
        CACHE,
        beta=np.array(probes),
        I=np.array([Ivals[b] for b in probes]),
        C=np.array([towers[b][3] for b in probes]),
        c0=np.array([towers[b][2] for b in probes]),
        n_resolved=np.array([len(towers[b][0]) for b in probes]),
        **{f"B_n_{i}": towers[b][0] for i, b in enumerate(probes)},
        **{f"d_n_{i}": towers[b][1] for i, b in enumerate(probes)},
        readme=(
            "The excited tower of the DK-brane onset "
            "operator.  I = int_{r_p}^oo e^{-V}/sqrt(U) dr is the Liouville "
            "length; the tower obeys sqrt(B_n) I/pi = n + C/n with NO constant "
            "term (endpoint indices mu_hor = 0, mu_bnd = 1).  B_n_<i> and "
            "d_n_<i> = the resolution-verified tower and its d_n = "
            "sqrt(B_n) I/pi - n at beta[i]; n_resolved[i] = its length."
        ),
    )
    log(f"  saved {CACHE}")
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
