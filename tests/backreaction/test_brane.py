"""The lattice on the D'Hoker-Kraus brane.

Here alpha B^2 is treated exactly and only rho^2 is perturbative, so the
background is the nonlinear magnetic brane and the whole family is labelled by
the single coupling beta = alpha B^2.  The results tested: the exact onset
curve B_c(alpha) (paper sec. 2.2, sec. 3.1), the finite-alpha lattice selection (paper sec. 5.2, app. D.5), the full
modular moduli scan (paper sec. 5.2, app. D.5), and the AdS_3 throat with its critical coupling
alpha_* = 2/3 (paper sec. 4.1, sec. 4.2), and the excited tower, which the paper uses only for the
statement that higher overtones condense at larger fields (paper app. D.2).

Three of these are statements about the shape of a curve, not a single
value, so the tests pin the shape: the gap dips and recovers rather than
crossing zero, the Hessian well dips and recovers at the same beta, and
B_c(alpha) is convex rather than linear.  A test that only checked the final
value would also pass on a monotone or linear curve.

The exact kernel and selection caches are already covered in
test_committed_caches.py; this file covers the caches that were not, and adds
the live brane solves that are cheap enough for the fast tier.
"""

import numpy as np
import pytest

from backreaction.numerics import dk_kernel as K
from backreaction.numerics import dk_moduli as md
from backreaction.numerics import dk_selection as sel
from published_values import (
    ALPHA_STAR,
    ALPHA_STAR_BY_D,
    B_C,
    BC_ALPHA_SLOPE0,
    BC_AT_BETA_MAX,
    BC_OF_ALPHA,
    BC_TANGENT_TEN_PERCENT_ALPHA,
    DBC_SLOPE,
    DK_BETA_MAX,
    DK_GAP_AT_ZERO_BETA,
    DK_ID_SLOPE,
    DK_ENTROPY_SLOPE,
    MODULI_BETA_TURNING,
    MODULI_CURVATURE_RATIO_AT_ZERO,
    MODULI_CURVATURE_RATIO_FINAL,
    MODULI_MARGIN_FINAL,
    MODULI_GAP_DIP_RATIO,
    MODULI_GAP_FINAL_RATIO,
    MODULI_HESS_AT_ZERO,
    MODULI_HESS_ISOTROPY,
    MODULI_HESS_MIN,
    MODULI_TAU_TOL,
    PROBE_RELATIVE_MARGIN,
    THROAT_BKT_A,
    THROAT_BKT_D,
    THROAT_INVARIANT,
    THROAT_L3SQ,
    THROAT_LADDER,
    TOWER_C,
    TOWER_C0_MAX,
    TOWER_D1,
    TOWER_FIT_RESID_MAX,
    TOWER_I,
    TOWER_I_RC_SLOPE,
    TOWER_LEVEL_GAP,
    TOWER_N_THROAT,
    MATCHING_ERR,
    MATCHING_A_EFF,
    MATCHING_LOG_RATIO,
    MATCHING_OFFSET_T0,
    OFFSET_TWO_CHARTS_DEV,
    MATCHING_OFFSET_TOP_RUNG,
    MATCHING_RHOSTAR_OFFSET,
    MATCHING_IRRELEVANT_P,
    TOWER_PHASE_RATIO_TOP,
    WONG_GAMMA_STAR,
    K0_ZERO_ALPHA,
    K0_ZERO_BETA,
    K03_ZERO_ALPHA,
    LADDER_K0_OVER_C4,
    LADDER_C_TRI_MIN,
    LADDER_GAP_MIN,
    ALPHA_L_FIRST_ORDER,
)


@pytest.fixture(scope="module")
def bca():
    return K.load_cache()


@pytest.fixture(scope="module")
def branes(bca):
    """The beta = 0, beta = 4 (the turning point) and beta = 45 branes."""
    betas = np.asarray(bca["beta"], float)
    out = {}
    for b in (0.0, MODULI_BETA_TURNING, DK_BETA_MAX):
        i = int(np.argmin(np.abs(betas - b)))
        assert abs(betas[i] - b) < 1e-12
        out[b] = K.make_brane(bca, i)
    return out


# -------------------------------------------------------------------- paper sec. 2.2, sec. 3.1


def test_the_onset_curve_starts_at_the_probe_value(bca):
    """At zero coupling the cached curve sits on the probe B_c = 5.13126764.

    B_c[0] is the brane's Sturm-Liouville eigenvalue at beta = 0, solved on
    the nonlinear background; it is compared with the probe value of
    scripts/critical_field.py.  B_c0 is the stored reference
    (bc_alpha.BC0_TARGET, from probe_bc_precise.py), so its assert is only a
    regression check of the constant.  The 1.5e-11 lock of both methods to it
    runs in bc_alpha.py itself (slow tier)."""
    assert float(bca["B_c0"]) == pytest.approx(B_C, abs=5e-9)
    assert float(bca["B_c"][0]) == pytest.approx(B_C, abs=5e-9)
    assert float(bca["beta"][0]) == 0.0
    assert float(bca["alpha"][0]) == 0.0


def test_the_tangent_reproduces_the_perturbative_shift(bca):
    """dB_c/B_c = +0.20970 alpha B_c^2 -- the finite-alpha machinery must
    reduce to the independently derived O(alpha) shift (paper sec. 3).  This is the
    anchor that ties the exact-in-beta onset back to perturbation theory."""
    assert float(bca["slope0"]) == pytest.approx(BC_ALPHA_SLOPE0, abs=5e-6)
    assert float(bca["slope0"]) == pytest.approx(DBC_SLOPE, abs=1e-5)


@pytest.mark.parametrize("alpha,Bc", sorted(BC_OF_ALPHA.items()))
def test_the_onset_curve_values(bca, alpha, Bc):
    """The quoted curve, from the small-alpha sweep (T = 1/pi units) (paper sec. 2.2, sec. 3.1)."""
    a, B = np.asarray(bca["small_alpha"]), np.asarray(bca["small_Bc"])
    i = int(np.argmin(np.abs(a - alpha)))
    assert a[i] == pytest.approx(alpha, abs=1e-12)
    assert B[i] == pytest.approx(Bc, abs=5e-4)


def test_the_curve_reaches_the_quoted_endpoint(bca):
    assert float(bca["beta"][-1]) == pytest.approx(DK_BETA_MAX, abs=1e-12)
    assert float(bca["B_c"][-1]) == pytest.approx(BC_AT_BETA_MAX, abs=5e-4)
    assert float(bca["alpha"][-1]) == pytest.approx(0.17445, abs=5e-6)


def test_the_curve_is_monotone_and_convex(bca):
    """Why the O(alpha) shift cannot be continued to large coupling (paper sec. 2.2, sec. 3.1): the
    perturbative stabilisation *strengthens* at finite coupling, so the
    linear approximation is already 10% low (relative to B_c) at alpha =
    0.0787 (sec. 3.3; the spline root is tested in
    test_critical_coupling_results.py).  Here, on the raw sweep: the relative
    deviation grows monotonically and crosses 10% between the two sweep points
    that bracket 0.0787 (0.0839 is the first point past it, 11.2% of B_c there,
    i.e. 12.6% above the tangent)."""
    alpha, Bc = np.asarray(bca["alpha"]), np.asarray(bca["B_c"])
    assert np.all(np.diff(Bc) > 0)
    d1 = np.diff(Bc) / np.diff(alpha)
    assert np.all(np.diff(d1) > 0)  # convex throughout
    off = 1.0 - B_C * (1 + BC_ALPHA_SLOPE0 * B_C**2 * alpha) / Bc
    assert np.all(np.diff(off) > 0)
    k = int(np.argmax(off > 0.10))  # first sweep point past 10%
    assert alpha[k - 1] < BC_TANGENT_TEN_PERCENT_ALPHA < alpha[k]
    assert off[k] == pytest.approx(0.112, abs=1e-3)


def test_three_solvers_agree_across_the_sweep(bca):
    """The two-method requirement, met three ways on the grid to beta = 45:
    horizon shooting with the exact source extraction c_0 = w + (r/2) w',
    a boundary-row collocation generalised eigenvalue problem, and a
    metric-only spectral GEP.  All three share the solve_colloc background,
    and the two GEPs share the Sturm-Liouville coefficients.  Measured
    agreement 2.8e-10, with the source extraction keeping the r^-3 terms of
    the fixed-T frame (paper sec. 2.2, sec. 3.1)."""
    sh, co, sp = bca["ind_shoot"], bca["ind_colloc"], bca["ind_spectral"]
    assert len(sh) >= 7
    assert np.max(np.abs(sh - co) / co) < 1e-10
    assert np.max(np.abs(co - sp) / co) < 1e-9  # these two share a path


def test_the_background_grid_reduces_to_the_G_zero_sector():
    """The brane background must reproduce the two exact per-alpha-B^2
    anchors of the G = 0 sector in the beta -> 0 limit: delta s/s_0|_T -> beta/4 and I_d ->
    pi^2/48.  Both are *slopes*, so they are extracted by extrapolating the two
    smallest beta on the grid rather than read off a single point."""
    d = np.load("backreaction/data/dk_background_grid.npz", allow_pickle=True)
    betas = np.asarray(d["betas"], float)
    assert float(d["alpha"]) == 2.0  # the D'Hoker-Kraus dictionary
    assert np.allclose(d["T"], 1.0 / np.pi)  # boundary-normalised fixed T

    i1, i2 = 1, 2  # the two smallest nonzero
    for key, want in (("ds_over_s0", DK_ENTROPY_SLOPE), ("Id_horizon", DK_ID_SLOPE)):
        r1 = float(d[key][i1]) / betas[i1]
        r2 = float(d[key][i2]) / betas[i2]
        slope = r1 + (r1 - r2) * betas[i1] / (betas[i2] - betas[i1])
        assert slope == pytest.approx(want, rel=2e-3), key


# ------------------------------------------------------------------- paper sec. 5.2, app. D.5


def test_the_selection_gap_from_a_live_brane_solve(branes):
    """The beta = 0 anchor recomputed rather than read from the cache: the
    brane pipeline reproduces the probe gap C_sq - C_tri."""
    gap, c_tri, c_sq, nshells = sel.gap_at(branes[0.0])
    assert gap == pytest.approx(DK_GAP_AT_ZERO_BETA, abs=5e-9)
    assert c_sq > c_tri > 0
    assert nshells == (3, 4)


def test_the_gap_dips_and_recovers_rather_than_crossing(branes):
    """The selection gap, recomputed at the three branes that matter (paper sec. 5.2, app. D.5).  The
    exact kernel dips 0.6% at beta = 4 and comes back 1.8% *above* the probe
    value, without crossing zero.  The dip and recovery are the result -- a test
    of the endpoint alone would also pass on a monotone decrease."""
    g0 = sel.gap_at(branes[0.0])[0]
    g4 = sel.gap_at(branes[MODULI_BETA_TURNING])[0]
    g45 = sel.gap_at(branes[DK_BETA_MAX])[0]
    assert g4 < g0 < g45
    assert g4 / g0 == pytest.approx(MODULI_GAP_DIP_RATIO, abs=5e-4)
    assert g45 / g0 == pytest.approx(MODULI_GAP_FINAL_RATIO, abs=5e-4)


def test_the_exact_kernel_anchor_at_zero_coupling(branes):
    """dk_kernel anchor [1]: at beta = 0 the brane kernel equals the tabulated
    probe-limit kernel over the whole grid.  The script asserts this itself; it
    is repeated here so the fast tier catches a break without the 2.5-minute
    scan."""
    assert K.anchor1(branes[0.0]) < 1e-11  # table 4 quotes 3e-12


# ------------------------------------------------------------------- paper sec. 5.2, app. D.5


@pytest.fixture(scope="module")
def moduli():
    return np.load("backreaction/data/dk_moduli.npz", allow_pickle=True)


def test_the_shell_generator_is_modular_invariant():
    """Structural fact (i) of the moduli scan (paper sec. 5.2, app. D.5), and a pure test of the generator: the
    multiset {gamma_mn} is invariant under tau -> tau + 1 and tau -> -1/tau,
    for any kernel whatsoever.  It is what licenses restricting the scan to the
    fundamental domain."""
    rng = np.random.default_rng(20260802)
    for _ in range(12):
        tau = complex(rng.uniform(-0.5, 0.5), rng.uniform(0.9, 2.5))
        base = md.gammas(tau)
        for image in (tau + 1.0, -1.0 / tau):
            other = md.gammas(image)
            n = min(len(base), len(other))
            assert np.max(np.abs(base[:n] - other[:n])) < 1e-12


def test_the_generator_reproduces_the_two_hardcoded_lattices():
    """Anchor [1]: tau = i gives 2 pi (m^2 + n^2) and tau = rho gives
    (4 pi / sqrt 3)(m^2 + mn + n^2) -- the shells the two-lattice selection
    hardcodes (paper sec. 5.2, app. D.5)."""
    for tau, unit, form in (
        (md.TAU_SQ, 2 * np.pi, lambda m, n: m * m + n * n),
        (md.RHO, 4 * np.pi / np.sqrt(3.0), lambda m, n: m * m + m * n + n * n),
    ):
        want = sorted(
            unit * form(m, n)
            for m in range(-8, 9)
            for n in range(-8, 9)
            if not (m == 0 and n == 0)
        )
        got = md.gammas(tau)
        assert np.max(np.abs(np.array(want)[: len(got)] - got)) < 1e-12


def test_triangular_is_the_global_minimiser_at_every_coupling(moduli):
    """The moduli-scan result (paper sec. 5.2, app. D.5): over the whole beta grid the minimiser never
    leaves tau = rho = exp(i pi/3), to the optimiser's tolerance.  The probe
    limit's theorem needs a completely monotone kernel, which the brane kernel
    is not, so this is a numerical statement over Bravais lattices at one flux
    quantum, not a proof."""
    tau = moduli["tau1"] + 1j * moduli["tau2"]
    assert np.max(np.abs(tau - md.RHO)) < MODULI_TAU_TOL
    assert moduli["Cmin"] == pytest.approx(moduli["C_tri"], abs=1e-12)
    assert np.all(moduli["C_sq"] > moduli["C_tri"])


def test_the_hessian_at_the_triangular_point_is_isotropic(moduli):
    """Structural fact (ii) and anchor [4]: rho is the order-3 elliptic fixed
    point of the modular group and the hyperbolic metric is conformal to the
    Euclidean one in (tau1, tau2), so a Z3-invariant quadratic form there must
    be proportional to the identity -- at every beta, for any kernel.  This is
    the check number the "by symmetry" step needs (third clause)."""
    ev = moduli["hess_ev"]
    rel = np.abs(ev[:, 1] - ev[:, 0]) / np.abs(ev[:, 0])
    # limited by the h = 1e-3 differences; the value app. D.5 quotes,
    # MODULI_HESS_ISOTROPY, is asserted in test_critical_coupling_results.py
    assert rel.max() < 5 * MODULI_HESS_ISOTROPY


def test_the_well_stays_strictly_convex_and_turns_where_the_gap_does(moduli):
    """The second half of the moduli-scan result (paper sec. 5.2, app. D.5): the curvature of the well dips and
    recovers, with its minimum at the *same* beta as the gap's.  Two
    independent quantities sharing a turning point is the evidence that this is
    one feature of the kernel rather than two coincidences."""
    ev, beta = moduli["hess_ev"], np.asarray(moduli["beta"], float)
    lo = ev.min(axis=1)
    assert lo.min() > 0
    assert lo.min() == pytest.approx(MODULI_HESS_MIN, abs=5e-6)
    assert lo[0] == pytest.approx(MODULI_HESS_AT_ZERO, abs=5e-6)
    assert lo[-1] > lo[0]

    gap = moduli["C_sq"] - moduli["C_tri"]
    assert beta[int(np.argmin(lo))] == MODULI_BETA_TURNING
    assert beta[int(np.argmin(gap))] == MODULI_BETA_TURNING


def test_the_normalisation_free_ratios(moduli):
    """The normalisation-free ratios of the selection, since the brane
    normalisation makes C itself meaningful only up to a positive factor: the
    margin over the square falls monotonically from the probe 0.246 to 0.227
    at alpha = 0.174 (paper section 5.2), and the curvature of the well over
    C_tri by 8% (a regression value; the paper says only that the Hessian stays
    positive and isotropic)."""
    ctri = moduli["C_tri"]
    margin = (moduli["C_sq"] - ctri) / ctri
    assert np.all(np.diff(margin) < 0)
    assert margin[0] == pytest.approx(PROBE_RELATIVE_MARGIN, abs=5e-5)
    assert margin[-1] == pytest.approx(MODULI_MARGIN_FINAL, abs=5e-5)
    curv = moduli["hess_ev"].mean(axis=1) / ctri
    assert curv[0] == pytest.approx(MODULI_CURVATURE_RATIO_AT_ZERO, abs=5e-4)
    assert curv[-1] == pytest.approx(MODULI_CURVATURE_RATIO_FINAL, abs=5e-4)


def test_the_moduli_scan_covers_the_fundamental_domain(moduli):
    """Scope, asserted rather than assumed: |tau1| <= 1/2, |tau| >= 1, and the
    elongation cutoff tau2 <= 5 that leaves the stripe limit unproved."""
    t1, t2 = moduli["tau1_grid"], moduli["tau2_grid"]
    assert t1.min() == 0.0 and t1.max() == pytest.approx(0.5)
    assert t2.min() == pytest.approx(np.sqrt(3.0) / 2)
    assert t2.max() == pytest.approx(md.TAU2_MAX)
    for grid in (moduli["C_grid_beta0"], moduli["C_grid_last"]):
        # C rises steeply towards the elongation cutoff: the stripe limit is
        # strongly disfavoured, though not proved so.
        col = grid[0][~np.isnan(grid[0])]
        assert col[-1] / col[0] > 10


def test_the_moduli_scan_agrees_with_a_live_kernel_at_the_two_lattices(moduli, branes):
    """The spline the scan uses is certified against exact kernel calls in the
    script; here the endpoint that matters -- C at the two named lattices -- is
    recomputed from the exact kernel and compared to the stored scan."""
    _, c_tri, c_sq, _ = sel.gap_at(branes[0.0])
    assert float(moduli["C_tri"][0]) == pytest.approx(c_tri, rel=2e-4)
    assert float(moduli["C_sq"][0]) == pytest.approx(c_sq, rel=2e-4)


# ------------------------------------------------------------------- paper sec. 4.1, sec. 4.2


@pytest.fixture(scope="module")
def throat():
    return np.load("backreaction/data/dk_throat.npz", allow_pickle=True)


_PRODUCTION_THROAT = {}


def _throat_from_production_equations():
    """(l^2, alpha_*) from the PRODUCTION reduced equations of dk_background
    (the ones the brane solver integrates): the AdS_3 x R^2 ansatz
    U = rho^2/l^2, e^{2V} = v, e^{2Z} = rho^2 solves them only at l^2 = 1/3,
    v = B sqrt(alpha/6); the lowest-Landau-level mass in the throat is then
    b = B l^2/v = sqrt(2/(3 alpha)), and the AdS_3 BF bound b = 1 gives
    alpha_*.  Cached, since three tests use it."""
    if _PRODUCTION_THROAT:
        return _PRODUCTION_THROAT["l2"], _PRODUCTION_THROAT["alpha_star"]
    import sympy as sp

    from backreaction.numerics import dk_background as dkb

    S = dkb._SYM["_sym"]
    rr, Uv, Vv, Wv, Up, Vp, Wp, B, alpha, *_ = S["symbols"]
    l2, v, rho = sp.symbols("l2 v rho", positive=True)
    U = rho**2 / l2
    sub = {Uv: U, Vv: sp.log(v) / 2, Wv: sp.log(rho), Up: sp.diff(U, rho), Vp: 0, Wp: 1 / rho}
    eqs = [
        (S["Upp"] - sp.diff(U, rho, 2)).subs(sub).subs(rr, rho),
        S["Vpp"].subs(sub).subs(rr, rho),
        (S["Wpp"] + 1 / rho**2).subs(sub).subs(rr, rho),
        S["constraint"].subs(sub).subs(rr, rho),
    ]
    sol = sp.solve([sp.simplify(e) for e in eqs], [l2, v], dict=True)
    assert len(sol) == 1
    b = sp.simplify(B * sol[0][l2] / sol[0][v])
    a_star = sp.solve(sp.Eq(b, 1), alpha)
    assert len(a_star) == 1
    _PRODUCTION_THROAT.update(l2=sol[0][l2], alpha_star=a_star[0])
    return sol[0][l2], a_star[0]


def test_the_critical_coupling(throat):
    """alpha_* = 2/3, computed rather than read back from the production
    equations (see _throat_from_production_equations).  The cache value is
    checked against the computed one."""
    import sympy as sp

    l2, a_star = _throat_from_production_equations()
    assert l2 == sp.Rational(1, 3)
    assert a_star == sp.Rational(2, 3)
    assert float(a_star) == pytest.approx(ALPHA_STAR, abs=1e-15)
    assert float(throat["alpha_star"]) == pytest.approx(float(a_star), abs=1e-15)


def test_the_general_d_formula_and_its_external_anchor():
    """alpha_*(d) = 16 / (d(d-1)(d-2)), derived here from the throat data
    l^2 = (d-2)^2/(d(d-1)) and B l^2/v = (d-2)^{3/2}/sqrt(alpha d(d-1)) by
    setting the discriminant of Delta(Delta - (d-2)) = -B l^2/v to zero.  The
    flows into the throat and the node count that make it the critical coupling
    for d = 4, 5, 6 are derivations/brane_flows.py and numerics/t0_threshold.py
    (slow tier).  The d = 3 entry is the BF value only: it equals Wong's
    gamma_* = 3/4 (arXiv:1307.7839) via alpha = 2/gamma, but the true T = 0
    threshold there is alpha_c(3) = 3.0370 (paper sec. 4.2, sec. 4.3), so this is an anchor of the
    bound, not of the critical coupling."""
    import sympy as sp

    a = sp.Symbol("alpha", positive=True)
    derived = {}
    for d, want in ALPHA_STAR_BY_D.items():
        mass = (d - 2) ** sp.Rational(3, 2) / sp.sqrt(a * d * (d - 1))  # B l^2 / v
        # zero discriminant of Delta(Delta - (d-2)) + B l^2/v = 0
        sol = sp.solve(sp.Eq((d - 2) ** 2 - 4 * mass, 0), a)
        assert len(sol) == 1
        assert sol[0] == sp.Rational(16, d * (d - 1) * (d - 2))
        assert float(sol[0]) == pytest.approx(want, rel=1e-14)
        derived[d] = sol[0]
    # the derived d = 3 value against Wong's published gamma_* (external), and
    # the derived d = 4 value against the production equations' fixed point
    assert float(derived[3]) == pytest.approx(2.0 / WONG_GAMMA_STAR, rel=1e-14)
    assert derived[4] == _throat_from_production_equations()[1]


@pytest.mark.parametrize("beta", sorted(THROAT_LADDER))
def test_the_throat_ladder(throat, beta):
    """The throat table: alpha, the throat invariant b_h and beta e^{-4V(r_p)}
    at each rung (paper sec. 4.1, sec. 4.2)."""
    betas = np.asarray(throat["beta"], float)
    i = int(np.argmin(np.abs(betas - beta)))
    assert betas[i] == pytest.approx(beta, rel=1e-12)
    alpha, b_h, inv = THROAT_LADDER[beta]
    assert float(throat["alpha"][i]) == pytest.approx(alpha, abs=5e-6)
    assert float(throat["b_h"][i]) == pytest.approx(b_h, abs=5e-6)
    assert float(throat["inv"][i]) == pytest.approx(inv, rel=2e-5)


def test_the_deep_interior_reaches_the_fixed_point(throat):
    """Statement (i): beta e^{-4V(r_p)} -> 6 monotonically, to 8.9e-4 at the
    top rung.  That is the AdS_3 x R^2 invariant alpha B^2 e^{-4V} = 6."""
    inv = np.asarray(throat["inv"], float)
    assert np.all(np.diff(inv) > 0)
    assert inv[-1] == pytest.approx(THROAT_INVARIANT, rel=1e-3)
    assert inv[-1] < THROAT_INVARIANT  # approached from below


def test_the_onset_drives_the_throat_to_marginality(throat):
    """Statement (ii), the one that carries the physics: b_h decreases towards
    1, the Breitenlohner-Freedman bound of the AdS_3 throat.  Combined with (i)
    this *is* alpha -> 2/3, since 2/(3 b_h^2) = 6 alpha / (beta e^{-4V})
    identically -- so the identity is checked too, not just the limits."""
    b_h = np.asarray(throat["b_h"], float)
    alpha = np.asarray(throat["alpha"], float)
    inv = np.asarray(throat["inv"], float)
    assert np.all(np.diff(b_h) < 0)
    assert np.all(b_h > 1.0)
    assert b_h[-1] == pytest.approx(1.0, abs=0.07)
    assert alpha[-1] < ALPHA_STAR
    assert np.max(np.abs(2.0 / (3.0 * b_h**2) - 6.0 * alpha / inv)) < 1e-12


def test_the_bc_curve_diverges_at_finite_coupling(throat):
    """B_c(alpha) has a vertical asymptote at alpha_* = 2/3 (paper sec. 4.1, sec. 4.2): B_c runs over
    five decades while alpha moves only from 0.17 to 0.59.  The approach is
    logarithmically slow, so the small-alpha sweep (beta <= 45) shows no sign
    of it."""
    Bc, alpha = np.asarray(throat["B_c"]), np.asarray(throat["alpha"])
    assert np.all(np.diff(Bc) > 0)
    assert Bc[-1] / Bc[0] > 1e4
    assert alpha[-1] < ALPHA_STAR


def test_the_holographic_bkt_law(throat):
    """Statement (iii): nu = sqrt(b_h - 1) = A/(ln beta + d), fitted over the
    ladder.  The fitted A is an effective coefficient at these rungs; its
    asymptotic value 4 pi is derived from the matching and tested with it (paper sec. 4.6, app. D.3).  Here the test pins the fit and its residual."""
    A, dd = float(throat["fit_A"]), float(throat["fit_d"])
    assert A == pytest.approx(THROAT_BKT_A, abs=5e-3)
    assert dd == pytest.approx(THROAT_BKT_D, abs=5e-3)

    beta = np.asarray(throat["beta"], float)
    b_h = np.asarray(throat["b_h"], float)
    m = beta >= 1e7
    nu = np.sqrt(b_h[m] - 1.0)
    resid = np.abs(nu - A / (np.log(beta[m]) + dd)) / nu
    assert m.sum() == 9
    assert resid.max() < 6e-4


def test_the_throat_ladder_shares_rungs_with_the_bc_alpha_sweep(throat, bca):
    """The two independent solvers overlap at beta = 1, 5, 20, 45 and agree
    there to 1.5e-12 (paper sec. 4.1, sec. 4.2)."""
    bt, Bt = np.asarray(throat["beta"]), np.asarray(throat["B_c"])
    bb, Bb = np.asarray(bca["beta"]), np.asarray(bca["B_c"])
    shared = [b for b in bt if np.min(np.abs(bb - b)) < 1e-12]
    assert len(shared) >= 3
    for b in shared:
        assert Bt[np.argmin(np.abs(bt - b))] == pytest.approx(
            Bb[np.argmin(np.abs(bb - b))], rel=1e-10
        )


def test_the_ads3_radius_is_the_dhoker_kraus_value():
    """l_3^2 = 1/3, i.e. l_3 = L/sqrt 3 -- D'Hoker-Kraus's AdS_3 radius.  The
    constants the numerics build b_h and alpha_* from are checked against the
    fixed point of the production equations, not against stored copies."""
    from backreaction.numerics import dk_throat as dkt

    l2, a_star = _throat_from_production_equations()
    assert dkt.L3SQ == pytest.approx(float(l2), rel=1e-15)
    assert dkt.ALPHA_STAR == pytest.approx(float(a_star), rel=1e-15)
    assert float(l2) == pytest.approx(THROAT_L3SQ, rel=1e-15)


# -- the excited tower (paper app. D.2) ---------------------------------


@pytest.fixture(scope="module")
def tower():
    return np.load("backreaction/data/dk_tower.npz", allow_pickle=True)


def _tower_at(tw, beta):
    """(B_n, d_n) at the requested beta rung of the dk_tower cache."""
    betas = np.asarray(tw["beta"], float)
    i = int(np.argmin(np.abs(betas - beta)))
    assert betas[i] == pytest.approx(beta, rel=1e-12)
    return i, np.asarray(tw[f"B_n_{i}"], float), np.asarray(tw[f"d_n_{i}"], float)


@pytest.mark.parametrize("beta", sorted(TOWER_I))
def test_the_liouville_length(tower, beta):
    """I = int_{r_p}^oo e^{-V}/sqrt(U) dr, the phase scale of the whole tower.
    Two quadratures agree on it to 3e-11 in the script; here the published
    value is pinned."""
    i, _, _ = _tower_at(tower, beta)
    assert float(tower["I"][i]) == pytest.approx(TOWER_I[beta], abs=5e-9)


@pytest.mark.parametrize("beta", sorted(TOWER_I))
def test_the_tower_law_has_no_constant_term(tower, beta):
    """The sharp test of the derivation.  The endpoint indices of the Liouville
    potential are mu_hor = 0 (the Y_0 partner is the log-singular solution
    excluded by horizon regularity) and mu_bnd = 1 (the Y_1 partner is
    w -> const, excluded by c_0 = 0), and two Bessel ends quantise as
    sqrt(B_n) I = pi(n + mu_0/2 + mu_1/2 - 1/2) -- which for (0, 1) is exactly
    n pi.  A wrong index assignment would show up as a constant offset of order
    1/4 or 1/2 in n units; the measured offset is below 2e-4."""
    i, _, d = _tower_at(tower, beta)
    n = np.arange(1, len(d) + 1)
    k = slice(len(d) // 2, len(d))
    A = np.stack([np.ones(n[k].shape), 1.0 / n[k]], axis=1)
    c0, C = np.linalg.lstsq(A, d[k], rcond=None)[0]
    assert abs(c0) < TOWER_C0_MAX
    assert C == pytest.approx(TOWER_C[beta], abs=5e-6)
    assert np.max(np.abs(d[k] - (c0 + C / n[k]))) < TOWER_FIT_RESID_MAX


@pytest.mark.parametrize("beta", sorted(TOWER_D1))
def test_the_ground_state_is_not_in_the_tower_law(tower, beta):
    """B_1 = B_c is the throat-controlled quantity (paper sec. 4.1, sec. 4.2) and misses the SL law
    badly -- by a factor 6.6 at beta = 1e11.  The two results are
    complementary: the throat sets the bottom of the spectrum, the geometry as
    a whole sets its asymptotics.  If d_1 ever came out small, the tower law
    would be claiming B_c, which it must not."""
    _, _, d = _tower_at(tower, beta)
    assert d[0] == pytest.approx(TOWER_D1[beta], abs=1e-3)
    if beta >= 1e4:
        assert abs(d[0]) > 0.3


def test_the_tower_is_quadratic_not_geometric(tower):
    """No Efimov structure.  A throat-controlled tower would be geometric with
    ratio exp(2 pi/nu); at nu = 0.522 (beta = 1e4) that is 1.7e5, and the
    measured ratios start at 2.45 and fall towards 1 as (n+1)^2/n^2."""
    for beta in (1e4, 1e11):
        _, B, _ = _tower_at(tower, beta)
        ratios = B[1:6] / B[0:5]
        assert ratios.max() < 3.0
        assert np.all(np.diff(ratios[1:]) < 0)  # falling towards 1
        assert ratios[-1] < 1.5
        # and three orders of magnitude short of the geometric alternative,
        # which needs exp(2 pi/nu) >= 1.7e5 everywhere on this ladder
        assert ratios.max() < 1e-3 * 1.6e5


def test_the_throat_capacity_and_its_supremum(tower, throat):
    """N_throat = nu ln(r_c)/pi is the number of oscillations the throat holds
    AT the marginal eigenvalue.  It tends to exactly ONE half-oscillation (the
    ground state) as A -> 4 pi (paper sec. 4.6, app. D.3).  What is tested is the per-rung
    statement -- below 1 at every rung reached, and rising along the ladder --
    which is what licenses the single-LLL-mode reduction at finite beta.  (A
    comparison with the fitted law nu = A/(ln beta + d) would only restate the
    fit, which was made to these same b_h, so it is not repeated here.)"""
    bt = np.asarray(throat["beta"], float)
    b_h = np.asarray(throat["b_h"], float)
    got = []
    for beta, want in TOWER_N_THROAT.items():
        j = int(np.argmin(np.abs(bt - beta)))
        nu = np.sqrt(b_h[j] - 1.0)
        rc = (beta / 6.0) ** 0.25
        got.append(nu * np.log(rc) / np.pi)
        assert got[-1] == pytest.approx(want, abs=5e-4)
    assert max(got) < 1.0
    assert np.all(np.diff(got) > 0)


# ------------------------------------------------------------------- paper sec. 4.6, app. D.3


@pytest.fixture(scope="module")
def matching():
    return np.load("backreaction/data/dk_throat_matching.npz", allow_pickle=True)


def test_the_marginality_coefficient_is_derived(matching, throat):
    """Paper sec. 4.6, app. D.3: nu ln(rho_*/h) = chi(nu) + phi_*(nu) with no free parameter
    reproduces the ladder once its throat exponent is converted to the
    ladder's nu = sqrt(b_h - 1); its effective slope equals the ladder fit of
    test_the_holographic_bkt_law at ln beta ~ 25 and tends to 4 pi."""
    beta = np.asarray(matching["beta"], float)
    # the conversion uses the horizon eps of each rung, the dk_throat ladder's
    np.testing.assert_allclose(
        np.asarray(matching["eps_ladder"]), 6.0 - np.asarray(throat["inv"]), atol=1e-9
    )
    err = np.abs(np.asarray(matching["nu_pred_h"]) / np.asarray(matching["nu_meas"]) - 1)
    for b, want in MATCHING_ERR.items():
        i = int(np.argmin(np.abs(beta - b)))
        assert err[i] == pytest.approx(want, rel=0.1)
    ok = np.isfinite(err)
    assert np.all(np.diff(err[ok]) < 0)
    lb = np.asarray(matching["A_eff_lnbeta"], float)
    A = np.asarray(matching["A_eff"], float)
    for l, want in MATCHING_A_EFF.items():
        assert A[int(np.argmin(np.abs(lb - l)))] == pytest.approx(want, abs=2e-3)
    assert A[int(np.argmin(np.abs(lb - 25)))] == pytest.approx(
        float(throat["fit_A"]), rel=3e-3
    )
    assert A[-1] == pytest.approx(4 * np.pi, rel=1e-3)
    assert float(matching["log_ratio_b_over_a"]) == pytest.approx(
        MATCHING_LOG_RATIO, abs=2e-4
    )
    off = np.asarray(matching["L"], float) - 0.25 * np.log(beta / 6.0)
    assert off[np.isfinite(off)][-1] == pytest.approx(MATCHING_RHOSTAR_OFFSET, abs=2e-4)
    # [4'] (app. D.3): the offset as a limit, from the T = 0 brane alone in two charts;
    # the ladder decreases onto it from above
    t0, t0dw = float(matching["offset_t0"]), float(matching["offset_t0_dw"])
    assert t0 == pytest.approx(MATCHING_OFFSET_T0, abs=5e-8)
    assert abs(t0 - t0dw) <= OFFSET_TWO_CHARTS_DEV
    assert t0 == pytest.approx(MATCHING_RHOSTAR_OFFSET, abs=5e-4)
    big = off[np.isfinite(off) & (beta >= 1e6)]
    assert np.all(np.diff(big) < 0) and np.all(big > t0)
    assert big[-1] == pytest.approx(MATCHING_OFFSET_TOP_RUNG, abs=5e-8)


def test_the_irrelevant_exponent_is_sqrt_19_over_3_minus_1():
    """The growing perturbation of the AdS_3 x R^2 fixed point has exponent
    sqrt(19/3) - 1, i.e. the throat dimension Delta_+ = 1 + sqrt(19/3) at
    lambda = 0 (paper app. D.3), seen from the background's own linearisation."""
    from backreaction.numerics import dk_throat_matching as M

    p, _, _, p_sym = M.irrelevant_mode()
    assert p == pytest.approx(MATCHING_IRRELEVANT_P, rel=1e-12)


def test_the_throat_feeds_I_through_its_logarithmic_measure(tower):
    """I = (l3 ln r_c + O(1))/r_c: the throat DOES dominate I -- once b_h -> 1
    the horizon sits inside it -- but through the log measure of an AdS_3
    plateau, which carries no nu.  That, not any smallness of the throat, is
    why the tower is not throat-controlled.  The slope is a derived constant,
    l3 = 1/sqrt3, matched to 0.02%."""
    beta = np.asarray(tower["beta"], float)
    I = np.asarray(tower["I"], float)
    m = beta >= 1e6
    rc = (beta[m] / 6.0) ** 0.25
    slope = np.polyfit(np.log(rc), I[m] * rc, 1)[0]
    assert slope == pytest.approx(TOWER_I_RC_SLOPE, abs=5e-4)
    assert slope == pytest.approx(np.sqrt(THROAT_L3SQ), rel=3e-3)


def test_the_radial_level_gap_closes(tower):
    """B_2/B_1 falls monotonically towards 1 -- it is NOT bounded away from it.
    The mechanism is the tower law itself: 1 + d_1 = sqrt(B_c) I/pi grows like
    ln(r_c)/pi, so the whole low tower crowds towards B_c as the optical length
    of the geometry grows, so any single rung's gap is a value on a trend,
    not a floor.  The crowding is algebraic, though --
    ((2+d_2)/(1+d_1))^2 -- not the geometric accumulation of an Efimov tower."""
    gaps, d1 = [], []
    for beta in sorted(TOWER_LEVEL_GAP):
        i, B, d = _tower_at(tower, beta)
        gaps.append(B[1] / B[0])
        d1.append(1.0 + d[0])
        assert B[1] / B[0] == pytest.approx(TOWER_LEVEL_GAP[beta], abs=1e-3)
    assert np.all(np.diff(gaps) < 0)  # monotone, all the way down
    assert np.all(np.diff(d1) > 0)  # because sqrt(B_c) I/pi grows
    assert gaps[-1] > 1.0  # but has not reached 1 yet


def test_the_two_phase_functionals_separate_at_marginality(tower, throat):
    """Why no single phase count fixes both ends of the spectrum, and the
    one statement here that needs no limit in beta at all.  The throat enters
    the bottom of the spectrum through sqrt(b-1) = nu -- the AdS_3 BF
    subtraction -- and the top through sqrt(b).  Their ratio sqrt(b/(b-1))
    diverges as b_h -> 1, i.e. exactly at alpha_*, so no single phase count
    covers both ends: using sqrt(b) where sqrt(b-1) belongs is ~30% off at the
    couplings of the throat ladder."""
    bt = np.asarray(throat["beta"], float)
    b_h = np.asarray(throat["b_h"], float)
    ratios = np.sqrt(b_h / (b_h - 1.0))
    assert np.all(np.diff(ratios) > 0)  # grows all the way up the ladder
    j = int(np.argmin(np.abs(bt - 1e11)))
    assert ratios[j] == pytest.approx(TOWER_PHASE_RATIO_TOP, abs=1e-3)
    assert ratios[j] > 4.0


# ------------------------------------------------------------------- paper sec. 5.1, sec. 5.3


@pytest.fixture(scope="module")
def long_wavelength():
    return np.load("backreaction/data/dk_long_wavelength.npz", allow_pickle=True)


def test_the_long_wavelength_kernel_changes_sign_once_on_the_ladder(long_wavelength):
    """Paper sec. 5.1, sec. 5.3: K_0(beta) = lim_{s->0} K = K_{G=0} changes sign once on the
    ladder, at alpha = 0.3505596 (beta = 3305.42); the gamma = 0.3 proxy kernel
    crosses at 0.4272; the triangular cell functional and the selection gap stay positive
    at every rung, so the homogeneous lattice is still selected while the
    long-wavelength direction has already gone soft."""
    beta = np.asarray(long_wavelength["beta"], float)
    k0 = np.asarray(long_wavelength["K0"], float) / np.asarray(long_wavelength["C4"], float)
    for b, want in LADDER_K0_OVER_C4.items():
        i = int(np.argmin(np.abs(beta - b)))
        assert k0[i] == pytest.approx(want, abs=5e-7)
    assert np.sum(np.diff(np.sign(k0)) != 0) == 1
    assert float(long_wavelength["alpha_K0zero"]) == pytest.approx(K0_ZERO_ALPHA, abs=5e-8)
    assert float(long_wavelength["beta_K0zero"]) == pytest.approx(K0_ZERO_BETA, abs=5e-3)
    assert float(long_wavelength["alpha_K03zero"]) == pytest.approx(K03_ZERO_ALPHA, abs=2e-5)
    ctri = np.asarray(long_wavelength["C_tri"], float)
    gap = np.asarray(long_wavelength["C_sq"], float) - ctri
    assert np.all(ctri > 0) and np.all(gap > 0)
    assert ctri.min() == pytest.approx(LADDER_C_TRI_MIN, abs=2e-5)
    assert gap.min() == pytest.approx(LADDER_GAP_MIN, abs=2e-5)
    # K_0 three ways (paper sec. 5.1, app. C.3): collocation, shooting, s -> 0 extrapolation of the G != 0 kernel
    c4 = np.asarray(long_wavelength["C4"], float)
    for key, tol in (("K0_shoot", 5e-10), ("K0_ext", 2e-8)):
        other = np.asarray(long_wavelength[key], float) / c4
        assert np.max(np.abs(other - k0)) < tol  # observed 1.1e-10, 5.1e-9


def test_the_homogeneous_quartic_turns_negative_above_the_zero_of_K0(long_wavelength):
    """Paper sec. 5.1, sec. 5.3: C_tot = K_0/2 + C_tri > 0 at the zero of K_0 and crosses zero
    6% later in alpha -- above that the LLL transition is not second order."""
    assert float(long_wavelength["alpha_first_order"]) == pytest.approx(ALPHA_L_FIRST_ORDER, abs=5e-8)
    assert float(long_wavelength["alpha_first_order"]) > float(long_wavelength["alpha_K0zero"])
    ctot = np.asarray(long_wavelength["C_tot"], float) / np.asarray(long_wavelength["C4"], float)
    assert np.sum(np.diff(np.sign(ctot)) != 0) == 1

