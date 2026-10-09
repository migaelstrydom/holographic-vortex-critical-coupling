"""Published numbers of the uniform-response, second-variation, T = 0 threshold
and onset-scan results, and the other cached numbers the paper quotes.

Most of these are cache checks: they catch a corrupted or silently regenerated
cache, not a regression in the code.  The code paths are exercised by the
slow-tier scrape in `test_scripts.py`, which runs every script below and
requires a zero exit code (each one asserts its own results).  The last
section recomputes in the fast tier what is cheap to recompute: the
probe-limit Bogoliubov spectrum, from the independent probe kernel of
`scripts/exchange_kernel.py`, and two points of the dense onset scan by fresh
horizon shooting.
"""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from backreaction import paths
from published_values import (
    ALPHA_I_PRECISE,
    ALPHA_I_PRECISE_UNC,
    ALPHA_L,
    ALPHA_L_BUDGET,
    ALPHA_L_BUDGET_REST_MAX,
    ALPHA_L_SECANT_F4,
    ALPHA_L_UNC,
    BC_AT_ALPHA_I,
    BC_AT_ALPHA_L,
    BC_TANGENT_TEN_PERCENT_ALPHA,
    BETA_I,
    BETA_L,
    BETA_ONE_ALPHA,
    BOGO_MODULATED_5000,
    BOGO_PROBE_K,
    BOGO_PROBE_M,
    BOGO_PROBE_TORUS_DEV,
    BOGO_PROBE_TORUS_MIN,
    BOGO_STIFFNESS_5000,
    BOGO_TORUS_AGREEMENT,
    C4,
    K0_ZERO_BC,
    LADDER_K0_OVER_C4,
    D3_ALPHA_C,
    D3_CROSS_Q,
    D3_CROSS_T,
    D3_RITZ_ALPHA_BOUND,
    D_CRIT,
    D_CRIT_CHART_DIFF,
    DK_C4_AT_45,
    DK_PI_OVER_C4_S0_AT_45,
    F4_AT_TOP_RUNG,
    FLOW_EXPONENTS,
    KERNEL_FD_AGREEMENT,
    KERNEL_LONGWAVE_OVER_C4,
    KG0_AT_45,
    KG0_LADDER_EXT_DEV,
    KG0_OALPHA_REFERENCE,
    KG0_OALPHA_SLOPE,
    KG0_PROBE,
    KG0_TWO_METHOD,
    K0_AT_45,
    LADDER_SLOPE_SPREAD_DIRECT,
    LATTICE_TORUS_DEV,
    MATCHING_A_EFF,
    MATCHING_ERR,
    MATCHING_LOG_RATIO,
    MODULI_HESS_ISOTROPY,
    ONSET_ALPHA_AT_BETA,
    ONSET_D_CHEB_AGREEMENT,
    ONSET_D_D3_ANCHOR_DEV,
    ONSET_D_D4_DEV,
    ONSET_LADDER_TWO_METHOD,
    ONSET_SCAN_ALPHA_END,
    ONSET_SCAN_BH_MIN,
    ONSET_SCAN_BETA_END,
    ONSET_SCAN_BKT_MIN,
    ONSET_SCAN_BKT_MIN_LNBETA,
    ONSET_SLOPE_DIRECT,
    PHI_STAR_K1,
    PHI_STAR_PLATEAU,
    PHI_STAR_TWO_CONSTRUCTIONS,
    ALPHA_L_BUDGET_FD,
    ONSET_MATCH_ALPHA_DEV,
    ONSET_MATCH_SLOPE_DEV,
    PROBE_BC_PRECISE,
    QNM_DLN_ALPHAC_DLN_BETA,
    T0_THRESHOLD_BA,
    T0_THRESHOLD_BA_D3,
    T0_THRESHOLD_BA_D5,
    T0_THRESHOLD_BA_D6,
    TC_RATIO_AT_055,
    THROAT_FULL_MARGIN,
    THROAT_LADDER,
    THROAT_S_GREEN_AGREEMENT,
    THROAT_TAIL_IDENTITY_DEV,
    THROAT_TC_EXPONENT,
    X_MINUS_XAB_MAX,
)


def _load(name):
    return np.load(paths.data(name), allow_pickle=False)


@pytest.fixture(scope="module")
def uniform():
    return _load("dk_uniform.npz")


@pytest.fixture(scope="module")
def bogo():
    return _load("dk_bogoliubov.npz")


@pytest.fixture(scope="module")
def t0():
    return _load("t0_threshold.npz")


@pytest.fixture(scope="module")
def scan():
    return _load("onset_scan.npz")


# -- the strict uniform response K_{G=0} (paper sec. 5.1, sec. 5.3) -------------------------------------------------------------


def test_kg0_probe_limit_is_c4(uniform):
    i = int(np.argmin(uniform["beta"]))
    assert uniform["beta"][i] == 0.0
    assert uniform["KG0"][i] == pytest.approx(KG0_PROBE, abs=1e-9)
    assert KG0_PROBE == pytest.approx(C4, abs=1e-7)


def test_kg0_two_methods_agree(uniform):
    for a, b, c4 in ((uniform["KG0"], uniform["KG0_shoot"], uniform["C4"]),
                     (uniform["lad_KG0"], uniform["lad_KG0_shoot"], uniform["lad_C4"])):
        assert np.max(np.abs(a - b) / c4) < 2e-10


def test_kg0_oalpha_slope_matches_independent_reference(uniform):
    assert float(uniform["slope_brane"]) == pytest.approx(KG0_OALPHA_SLOPE, abs=1e-9)
    assert float(uniform["slope_flat"]) == pytest.approx(KG0_OALPHA_REFERENCE, abs=1e-9)
    # the two methods against each other, from the cache (table 4: 2e-8)
    assert abs(float(uniform["slope_brane"]) / float(uniform["slope_flat"]) - 1) < 3e-8


def test_kg0_at_45(uniform):
    i = int(np.argmin(np.abs(uniform["beta"] - 45.0)))
    assert uniform["KG0"][i] == pytest.approx(KG0_AT_45, rel=1e-9)


def test_kg0_equals_ladder_k0(uniform):
    # Paper sec. 5.1, app. C.3: K_{G=0} against the cubic s -> 0 extrapolation of the G != 0 kernel on
    # every rung to 1e8 (7.8e-9 C4)
    rel = np.abs(uniform["lad_KG0"] - uniform["lad_K0_ext"]) / uniform["lad_C4"]
    assert np.max(rel) < 1e-8
    assert np.max(rel) == pytest.approx(KG0_LADDER_EXT_DEV, abs=1e-9)
    compared = 0
    for beta, k0 in LADDER_K0_OVER_C4.items():
        hit = np.isclose(uniform["lad_beta"], beta)
        if hit.any():
            j = int(np.argmax(hit))
            assert uniform["lad_KG0"][j] / uniform["lad_C4"][j] == pytest.approx(k0, abs=5e-7)
            compared += 1
    assert compared >= 5


def test_alpha_l(uniform):
    # alpha_L = 0.37217997 +- 1.1e-8, the quadrature sum of the named budget
    # (paper sec. 5.3, app. D.5, which rounds it up to 2e-8)
    assert float(uniform["alpha_fo"]) == pytest.approx(ALPHA_L, abs=5e-9)
    assert float(uniform["alpha_fo_unc"]) == pytest.approx(ALPHA_L_UNC, abs=5e-9)
    vals = dict(zip(uniform["budget_keys"], uniform["budget_vals"], strict=True))
    assert np.sqrt(np.sum(np.square(list(vals.values())))) == pytest.approx(
        float(uniform["alpha_fo_unc"]), rel=1e-12
    )
    assert abs(float(uniform["F4_fo"])) < 1e-10
    assert float(uniform["beta_fo"]) == pytest.approx(BETA_L, abs=5e-3)
    assert float(uniform["Bc_fo"]) == pytest.approx(BC_AT_ALPHA_L, abs=5e-5)


def test_quartic_coefficient_negative_at_top_rung(uniform):
    j = int(np.argmax(uniform["lad_beta"]))
    f4 = (0.5 * uniform["lad_KG0"][j] + uniform["lad_Ctri"][j]) / uniform["lad_C4"][j]
    assert f4 == pytest.approx(F4_AT_TOP_RUNG, abs=5e-6)


# -- second variation and Bogoliubov spectrum (paper sec. 5.3, app. C.5) ------------------------------


def test_bogoliubov_threshold_agrees_with_exact_k0(bogo):
    # dk_bogoliubov locates the same zero with K_0 = K_{G=0} (paper sec. 5.3, app. D.5)
    assert float(bogo["alpha_c"]) == pytest.approx(ALPHA_L, abs=max(float(bogo["alpha_c_unc"]), ALPHA_L_UNC))
    assert float(bogo["alpha_c_unc"]) < 1e-7


def test_modulated_state_costs_energy(bogo):
    assert float(bogo["mod_Qinf"]) == pytest.approx(BOGO_MODULATED_5000, abs=1e-6)
    assert float(bogo["mod_target"]) == pytest.approx(BOGO_STIFFNESS_5000, abs=1e-6)
    assert float(bogo["mod_Qinf"]) > 0


def test_no_unstable_mode_below_threshold(bogo):
    below = bogo["scan_alpha"] < ALPHA_L - 1e-4
    assert below.sum() >= 5
    assert not bogo["scan_unstable"][below].any()
    above = bogo["scan_alpha"] > ALPHA_L + 1e-4
    assert bogo["scan_unstable"][above].all()


# -- T = 0 threshold and the dense scan (paper sec. 4.2, sec. 4.3) ------------------------------------


def test_d3_has_a_crossover_bound_state(t0):
    i = int(np.argmax(t0["d"] == 3))
    assert int(t0["nodes"][i]) == 1
    assert float(t0["alpha_c_d3"]) == pytest.approx(D3_ALPHA_C, abs=1e-6)
    assert float(t0["alpha_c_d3"]) > 8 / 3
    assert np.all(np.diff(t0["d3_finite_t_alpha"]) > 0)


def test_t0_threshold_has_no_node_for_d_4_5_6(t0):
    for d, ba in ((4, T0_THRESHOLD_BA), (5, T0_THRESHOLD_BA_D5), (6, T0_THRESHOLD_BA_D6)):
        i = int(np.argmax(t0["d"] == d))
        assert int(t0["nodes"][i]) == 0
        assert float(t0["b_over_a"][i]) == pytest.approx(ba, abs=1e-6)
        assert float(t0["p"][i]) == pytest.approx(FLOW_EXPONENTS[d], abs=1e-6)
        assert not t0["nodes_below"][i].any()


def test_onset_scan_monotone_below_alpha_star(scan):
    order = np.argsort(scan["beta"])
    alpha = scan["alpha"][order]
    assert np.all(np.diff(alpha) > 0)
    assert alpha[-1] < 2 / 3
    assert alpha[-1] == pytest.approx(ONSET_SCAN_ALPHA_END, abs=1e-5)


def test_onset_scan_b_h_stays_above_one(scan):
    """Paper sec. 4.2 and app. D.2: b_h >= 1.003 along the scan, the minimum at
    its top."""
    order = np.argsort(scan["beta"])
    b_h = scan["b_h"][order]
    assert b_h.min() == pytest.approx(ONSET_SCAN_BH_MIN, abs=1e-6)
    assert int(np.argmin(b_h)) == len(b_h) - 1


def test_onset_scan_reproduces_ladder(scan):
    # The scan grid does not land on the rungs (the script itself checks the
    # rungs by direct solves), so compare through a spline in ln beta.
    from scipy.interpolate import CubicSpline

    order = np.argsort(scan["beta"])
    spline = CubicSpline(np.log(scan["beta"][order]), scan["alpha"][order])
    assert len(THROAT_LADDER) >= 6
    for b, row in THROAT_LADDER.items():
        # spline error: 9e-6 at beta = 45 (coarse there), <= 5e-7 above
        assert float(spline(np.log(b))) == pytest.approx(row[0], abs=2e-5)


def test_onset_scan_quoted_alpha_and_slopes(scan):
    # alpha at exactly the quoted beta (app. D.2), and the direct local slopes
    for b, a in zip(scan["alpha_quoted_beta"], scan["alpha_quoted"], strict=True):
        assert float(a) == pytest.approx(ONSET_ALPHA_AT_BETA[float(b)], abs=1e-7)
    marks = [float(m) for m in scan["lnbeta_marks"]]
    for m, want in ONSET_SLOPE_DIRECT.items():
        assert float(scan["A_local"][marks.index(m)]) == pytest.approx(want, abs=1e-5)
    # round-off level; only the bound is stable
    assert float(np.max(scan["refine_max_move"])) < 1e-13
    # [7] (app. D.2): the matching condition beyond the ladder
    dev = np.abs(scan["alpha_match"] - scan["alpha_quoted"])
    assert float(dev.max()) == pytest.approx(ONSET_MATCH_ALPHA_DEV, rel=0.3)
    for b, a in zip(scan["alpha_quoted_beta"], scan["alpha_match"], strict=True):
        assert float(a) == pytest.approx(ONSET_ALPHA_AT_BETA[float(b)], abs=1e-7)
    assert float(np.max(np.abs(scan["b_h_match"] - scan["b_h_quoted"]))) < 1e-10
    # the local slopes by matching, nu converted to sqrt(b_h - 1) (app. D.3);
    # unconverted, ln beta = 40 is off by the O(eps) frame difference
    for m, a_m in zip(scan["lnbeta_slope_match"], scan["A_local_match"], strict=True):
        a_s = float(scan["A_local"][marks.index(float(m))])
        want = ONSET_MATCH_SLOPE_DEV[float(m)]
        assert abs(float(a_m) / a_s - 1) == pytest.approx(want, rel=0.2, abs=2e-10)
    i40 = [float(m) for m in scan["lnbeta_slope_match"]].index(40.0)
    a40 = float(scan["A_local"][marks.index(40.0)])
    assert abs(float(scan["A_local_match_throat"][i40]) / a40 - 1) > 5e-5


def test_d3_finite_t_crossing(t0):
    assert float(t0["d3_cross_q"]) == pytest.approx(D3_CROSS_Q, abs=1e-8)
    assert float(t0["d3_cross_T_over_sqrtB"]) == pytest.approx(D3_CROSS_T, rel=1e-4)


# -- other cached numbers the paper quotes ------------------------------------


@pytest.fixture(scope="module")
def kernel():
    return _load("dk_kernel.npz")


def test_long_wavelength_kernel_and_its_pairing(kernel):
    # sec. 5.1: K/C4 at s = 0.02 B_c, 0.98 -> 0.18; app. C.3: P(s -> 0)/C4;
    # table 4: collocation vs finite differences; max(X - X_ab) is a
    # regression value, not in the paper
    beta = kernel["beta"]
    assert kernel["gamma"][0] == pytest.approx(0.02)
    for b, want in KERNEL_LONGWAVE_OVER_C4.items():
        i = int(np.argmin(np.abs(beta - b)))
        assert beta[i] == b
        assert kernel["K"][i, 0] / kernel["C4"][i, 0] == pytest.approx(want, abs=1e-6)
    i = int(np.argmin(np.abs(beta - 45.0)))
    g, r = kernel["gamma"][:4], kernel["Pi_bare"][i, :4] / kernel["C4"][i, :4]
    assert np.polyval(np.polyfit(g, r, 3), 0.0) == pytest.approx(DK_PI_OVER_C4_S0_AT_45, abs=1e-5)
    assert float(np.max(kernel["X_coupled"] - kernel["X"])) == pytest.approx(X_MINUS_XAB_MAX, abs=1e-4)
    assert float(np.max(np.abs(kernel["K"] - kernel["K_fd"]))) == pytest.approx(KERNEL_FD_AGREEMENT, rel=0.05)


def test_uniform_response_numbers(uniform):
    # app. C.3 / D.5: brane C4 at beta = 45, the two methods for K_{G=0}, the
    # alpha_L budget and the secant residual
    i = int(np.argmin(np.abs(uniform["beta"] - 45.0)))
    assert uniform["C4"][i] == pytest.approx(DK_C4_AT_45, abs=1e-7)
    two = max(np.max(np.abs(uniform["KG0"] - uniform["KG0_shoot"]) / uniform["C4"]),
              np.max(np.abs(uniform["lad_KG0"] - uniform["lad_KG0_shoot"]) / uniform["lad_C4"]))
    assert two == pytest.approx(KG0_TWO_METHOD, rel=0.05)
    budget = dict(zip(uniform["budget_keys"], uniform["budget_vals"], strict=True))
    for k, v in ALPHA_L_BUDGET.items():
        assert budget[k] == pytest.approx(v, rel=0.05)
    # C_tri by the finite-difference kernel moves alpha_L by 6e-12
    assert budget["C_tri: collocation vs finite differences"] == pytest.approx(ALPHA_L_BUDGET_FD, rel=0.1)
    rest = [v for k, v in budget.items() if k not in ALPHA_L_BUDGET]
    assert max(rest) == pytest.approx(ALPHA_L_BUDGET_REST_MAX, rel=0.05)
    assert abs(float(uniform["F4_fo"])) == pytest.approx(ALPHA_L_SECANT_F4, rel=0.05)


def test_lattice_table_couplings():
    # table 3: B_c at the zero of K_0; B_c at alpha_I is a regression value,
    # not in the paper
    lwc = _load("dk_long_wavelength.npz")
    assert float(lwc["Bc_K0zero"]) == pytest.approx(K0_ZERO_BC, abs=5e-4)
    ladder = _load("dk_moduli_ladder.npz")
    assert float(ladder["Bc_I_zero"]) == pytest.approx(BC_AT_ALPHA_I, abs=0.05)
    assert float(ladder["beta_I_zero"]) == pytest.approx(BETA_I, rel=1e-3)
    h = _load("dk_moduli.npz")["hess_ev"]
    assert float(np.abs(h[:, 1] - h[:, 0]).max() / h[:, 0].min()) == pytest.approx(
        MODULI_HESS_ISOTROPY, rel=0.05)


def test_torus_checks_of_the_block_formula():
    # table 4 and app. C.5: the Bogoliubov block formula against direct tori
    assert float(_load("dk_bogoliubov.npz")["torus_probe_dev"]) == pytest.approx(BOGO_PROBE_TORUS_DEV, rel=0.05)
    assert float(_load("dk_lattice_stiffness.npz")["torus_dev"]) == pytest.approx(LATTICE_TORUS_DEV, rel=0.05)


def test_throat_kernel_checks():
    # table 4: tail identity; app. D.5, table 4: finite differences vs Green's function
    t = _load("dk_throat_lattice.npz")
    assert float(np.max(t["identity_dev"])) == pytest.approx(THROAT_TAIL_IDENTITY_DEV, rel=0.05)
    g = t["gamma"]
    idx = [int(np.argmin(np.abs(g - x))) for x in t["gamma_green"]]
    dev = np.max(np.abs(t["S"][idx] - t["S_green"]) / np.abs(t["S_green"]))
    assert float(dev) == pytest.approx(THROAT_S_GREEN_AGREEMENT, rel=0.05)


def test_matching_plateau_and_errors():
    # sec. 4.6 and app. D.3: phi_* plateau over the ladder window, the matching
    # error along the ladder, the effective slope A_eff
    m = _load("dk_throat_matching.npz")
    win = (m["nu_tab"] >= 0.25) & (m["nu_tab"] <= 0.6)
    assert float(m["phi_tab"][win].min()) == pytest.approx(PHI_STAR_PLATEAU[0], abs=1e-6)
    assert float(m["phi_tab"][win].max()) == pytest.approx(PHI_STAR_PLATEAU[1], abs=1e-6)
    # the throat exponent of (M) in the frame of nu_meas = sqrt(b_h - 1)
    nu_h = np.sqrt((1 + m["nu_pred"] ** 2) * np.sqrt(1 - m["eps_ladder"] / 6) - 1)
    np.testing.assert_allclose(m["nu_pred_h"], nu_h, rtol=1e-14)
    rel = np.abs(m["nu_pred_h"] / m["nu_meas"] - 1)
    for b, e in MATCHING_ERR.items():
        j = int(np.argmin(np.abs(m["beta"] - b)))
        assert m["beta"][j] == b
        assert float(rel[j]) == pytest.approx(e, rel=0.05)
    lb = [int(x) for x in m["A_eff_lnbeta"]]
    for x, a in MATCHING_A_EFF.items():
        assert float(m["A_eff"][lb.index(x)]) == pytest.approx(a, abs=5e-4)
    # [3'] (app. D.3): the same phi_* on the domain-wall-gauge T = 0 brane
    dw = np.abs(m["phi_tab_dw"][win] - m["phi_tab"][win])
    assert np.all(np.isfinite(dw))
    assert float(dw.max()) < 2 * PHI_STAR_TWO_CONSTRUCTIONS
    assert float(m["log_ratio_dw"]) == pytest.approx(float(m["log_ratio_b_over_a"]), rel=1e-7)


def test_d_continuation_and_d3_table_entry(t0):
    # the real-d zero of c_1/c_0 in two charts is an extra, not in the paper;
    # table 2 (sec. 4.4): the d = 3 entry
    w = _load("d3_crossover_why.npz")
    assert float(w["d_crit"]) == pytest.approx(D_CRIT, abs=1e-7)
    assert float(np.ptp(w["d_crit_methods"])) == pytest.approx(D_CRIT_CHART_DIFF, rel=0.05)
    i = int(np.argmax(t0["d"] == 3))
    assert float(t0["b_over_a"][i]) == pytest.approx(T0_THRESHOLD_BA_D3, abs=1e-6)


def test_general_d_onset_agreements():
    # app. D.2: shooting vs Chebyshev for d = 5, 6; the d = 4 ladder and the
    # exact d = 3 RN-AdS4 onset reproduced by the general-d code
    d = _load("onset_scan_d.npz")
    cheb = max(float(np.max(d[k][:, 4])) for k in ("m2_d5", "m2_d6"))
    assert cheb == pytest.approx(ONSET_D_CHEB_AGREEMENT, rel=0.05)
    assert float(np.max(d["d4_ladder_dev"])) == pytest.approx(ONSET_D_D4_DEV, rel=0.05)
    a = d["d3_anchor"]
    assert float(np.max(np.abs(a[:, 1] - a[:, 2]) / a[:, 2])) == pytest.approx(ONSET_D_D3_ANCHOR_DEV, rel=0.05)


def test_onset_scan_slope_minimum_and_spread(scan):
    # sec. 4.6 / app. D.3: the minimum of the local slope and where it sits;
    # its spread over the ladder's last four and three decades
    marks = [float(m) for m in scan["lnbeta_marks"]]
    j = int(np.argmin(scan["A_local"]))
    assert marks[j] == ONSET_SCAN_BKT_MIN_LNBETA
    assert float(scan["A_local"][j]) == pytest.approx(ONSET_SCAN_BKT_MIN, abs=5e-4)
    assert float(np.max(scan["beta"])) == pytest.approx(ONSET_SCAN_BETA_END, rel=1e-3)
    lnb = np.log(scan["beta"])
    order = np.argsort(lnb)
    inv_nu = CubicSpline(lnb[order], 1 / scan["nu"][order])
    for lo, want in LADDER_SLOPE_SPREAD_DIRECT.items():
        x = np.linspace(np.log(lo), np.log(1e11), 400)
        a = 1 / inv_nu(x, 1)
        assert float((a.max() - a.min()) / a.mean()) == pytest.approx(want, abs=1.5e-3)


def test_tangent_of_the_onset_curve_ten_percent_off():
    # sec. 3.3: where the alpha = 0 tangent of B_c(alpha) is ten percent off
    b = _load("bc_alpha.npz")
    o = np.argsort(b["alpha"])
    s = CubicSpline(b["alpha"][o], b["B_c"][o])
    b0, sl = float(b["B_c0"]), float(b["slope0"])
    x = np.linspace(0.02, 0.17, 15001)
    off = (s(x) - b0 * (1 + sl * x * b0**2)) / s(x)
    assert float(x[np.argmin(np.abs(np.abs(off) - 0.1))]) == pytest.approx(BC_TANGENT_TEN_PERCENT_ALPHA, abs=5e-4)


# -- recomputed in the fast tier ----------------------------------------------


class _ProbeKernel:
    """The probe-limit kernel K = C4 - X(s) of scripts/exchange_kernel.py, in
    the interface dk_bogoliubov expects (gamma = s/B_c, K_0 = C4 at beta = 0).
    An independent code from the brane rung dk_bogoliubov used for its cache."""

    def __init__(self):
        z = np.load("scripts/kernel.npz")
        self.C4 = float(z["C4"])
        self.K0 = self.C4
        g = z["G2grid"] / float(z["B_c"])
        self.spl = CubicSpline(g, self.C4 - z["X"])
        self.gmax = g[-1]

    def __call__(self, gam):
        gam = np.asarray(gam, float)
        return np.where(gam <= self.gmax, self.spl(np.minimum(gam, self.gmax)), self.spl(self.gmax))

    def C_tri(self):
        from backreaction.numerics import dk_bogoliubov as bg

        G2 = np.sum(bg.GVEC * bg.GVEC, 1)
        G2 = G2[G2 > 1e-12]
        return 0.5 * float(np.sum(np.exp(-G2 / 2) * self(G2)))


def test_probe_bogoliubov_spectrum_recomputed():
    """sec. 5.3: lambda_-(M) = 0.0186 C4 and the 36-flux-quantum torus, from
    scratch on the probe kernel rather than read from dk_bogoliubov.npz."""
    from backreaction.numerics import dk_bogoliubov as bg

    ker = _ProbeKernel()
    lm_M = bg.bogoliubov(ker, (bg.B2 / 2)[None])[1][0] / ker.C4
    lm_K = bg.bogoliubov(ker, ((bg.B1 + 2 * bg.B2) / 3)[None])[1][0] / ker.C4
    assert lm_M == pytest.approx(BOGO_PROBE_M, abs=1e-6)
    assert lm_K == pytest.approx(BOGO_PROBE_K, abs=1e-6)
    tp = bg.torus_spectrum(ker)
    assert tp["ev_min"] == pytest.approx(BOGO_PROBE_TORUS_MIN, abs=1e-6)
    assert tp["evM"] == pytest.approx(lm_M, abs=1e-12)
    assert tp["dev"] < 10 * BOGO_PROBE_TORUS_DEV
    assert abs(tp["zero"]) < 1e-12 and tp["ev_min"] > 0


@pytest.mark.parametrize("i", [60, 184])
def test_onset_scan_points_recomputed(scan, i):
    """Two points of the dense onset scan (one inside the ladder's range, and the
    top at eps = 1e-24) solved again by horizon shooting, ~2 s each."""
    from backreaction.numerics import onset_scan

    order = np.argsort(scan["beta"])
    eps, nu = float(scan["eps"][order][i]), float(scan["nu"][order][i])
    r = onset_scan.onset(eps, nu)
    assert r["nodes"] == 0
    assert r["alpha"] == pytest.approx(float(scan["alpha"][order][i]), rel=1e-10)
    assert r["Bc"] == pytest.approx(float(scan["B_c"][order][i]), rel=1e-10)
    assert r["beta"] == pytest.approx(float(scan["beta"][order][i]), rel=1e-10)
    if i == len(order) - 1:
        assert r["alpha"] == pytest.approx(ONSET_SCAN_ALPHA_END, abs=1e-5)


def test_alpha_where_beta_reaches_one():
    # sec. 2.2: beta = alpha B_c^2 reaches one at alpha = 0.028
    t = _load("dk_throat.npz")
    i = int(np.argmin(np.abs(t["beta"] - 1.0)))
    assert t["beta"][i] == 1.0
    assert float(t["alpha"][i]) == pytest.approx(BETA_ONE_ALPHA, abs=1e-7)


# -- direct fits and second scripts behind quoted numbers -----------------


def test_tc_exponent_read_directly_in_alpha(scan):
    # sec. 4.6: ln beta = s/sqrt(alpha_* - alpha) + c on the nine rungs beta >=
    # 1e7, T_c ~ sqrt(B) exp(-(s/4)/sqrt(alpha_* - alpha)); the ladder and the
    # scan's own points in the same window
    t = _load("dk_throat.npz")
    w = t["beta"] >= 1e7
    assert w.sum() == 9
    s_lad = np.polyfit(1 / np.sqrt(2 / 3 - t["alpha"][w]), np.log(t["beta"][w]), 1)[0]
    assert s_lad / 4 == pytest.approx(THROAT_TC_EXPONENT, abs=5e-5)
    ws = (scan["beta"] >= 0.999e7) & (scan["beta"] <= 1.001e11)
    s_scan = np.polyfit(1 / np.sqrt(2 / 3 - scan["alpha"][ws]), np.log(scan["beta"][ws]), 1)[0]
    assert s_scan / s_lad == pytest.approx(1.0, abs=1e-3)


def test_phi_star_next_order_from_the_table():
    # sec. 4.6: tan phi_* = |b/a|/nu + k1 nu + O(nu^3); k1 from the cos-fitted
    # phi_* at the two smallest nu (the script's finite difference in nu^2 is
    # the other method), and the reference ratio b/a itself
    m = _load("dk_throat_matching.npz")
    r = -float(m["log_ratio_b_over_a"])
    assert -r == pytest.approx(MATCHING_LOG_RATIO, abs=5e-8)
    nu, ph = m["nu_tab"][:2], m["phi_tab"][:2]
    kk = (np.tan(ph) - r / nu) / nu
    k1 = kk[0] - (kk[1] - kk[0]) * nu[0] ** 2 / (nu[1] ** 2 - nu[0] ** 2)
    assert k1 == pytest.approx(PHI_STAR_K1, abs=2e-5)


def test_throat_margin_exact_shells():
    # sec. 5.2, app. E.4: direct solves at the exact shells
    d = _load("dk_throat_channels.npz")
    margin = (float(d["C_sq"]) - float(d["C_tri"])) / float(d["C_tri"])
    assert margin == pytest.approx(THROAT_FULL_MARGIN, abs=2e-6)


def test_k0_at_45_from_a_second_script():
    # app. C.3: K(s -> 0) at beta = 45 from dk_uniform.py [5] against the same
    # extrapolation in dk_long_wavelength.py; the two scripts differ by up to
    # 1.0e-8 C4 on the ladder, the floor of the extrapolation
    lw = _load("dk_long_wavelength.npz")
    i = int(np.argmin(np.abs(lw["beta"] - 45.0)))
    assert lw["beta"][i] == 45.0
    assert abs(float(lw["K0_ext"][i]) - K0_AT_45) / DK_C4_AT_45 < 1.1e-8
    u = _load("dk_uniform.npz")
    floor = 0.0
    for j, b in enumerate(u["lad_beta"]):
        k = int(np.argmin(np.abs(lw["beta"] - b)))
        if np.isclose(lw["beta"][k], b):
            floor = max(floor, abs(u["lad_K0_ext"][j] - lw["K0_ext"][k]) / u["lad_C4"][j])
    assert floor == pytest.approx(1.0e-8, abs=1e-9)


def test_qnm_factor_is_the_slope_of_the_onset_curve():
    # app. B.2: d ln alpha_c / d ln beta from the QNM script against the
    # derivative of the onset curve itself (bc_alpha.npz grid; onset_scan at 1e4)
    b = _load("bc_alpha.npz")
    o = np.argsort(b["beta"])
    m = b["beta"][o] > 0
    sp = CubicSpline(np.log(b["beta"][o][m]), np.log(b["alpha"][o][m]))
    for beta in (1.0, 45.0):
        assert float(sp(np.log(beta), 1)) == pytest.approx(QNM_DLN_ALPHAC_DLN_BETA[beta], rel=1e-4)
    sc = _load("onset_scan.npz")
    o = np.argsort(sc["beta"])
    sp = CubicSpline(np.log(sc["beta"][o]), np.log(sc["alpha"][o]))
    assert float(sp(np.log(1e4), 1)) == pytest.approx(QNM_DLN_ALPHAC_DLN_BETA[1e4], rel=1e-4)


def test_tc_ratio_at_alpha_055():
    # sec. 5.3: T_c(0.55)/T_c(0) at equal B = (B_c(0)/B_c(0.55))^(1/2)
    sc = _load("onset_scan.npz")
    o = np.argsort(sc["alpha"])
    bc = float(CubicSpline(sc["alpha"][o], sc["B_c"][o])(0.55))
    assert (PROBE_BC_PRECISE / bc) ** 0.5 == pytest.approx(TC_RATIO_AT_055, abs=5e-6)


def test_bogoliubov_24_quantum_torus():
    # app. C.5: block formula vs the 24-flux-quantum torus Hessian at beta = 5000
    b = _load("dk_bogoliubov.npz")
    assert float(b["torus_5000_dev"]) == pytest.approx(BOGO_TORUS_AGREEMENT, rel=0.05)
    assert float(b["torus_5000_alpha"]) == pytest.approx(0.363965, abs=1e-6)


def test_alpha_i_precise_against_the_cached_secant():
    # app. D.5: the cached secant stopped at |I/C4| = 4e-7, i.e. 1.8e-6 above the
    # zero that the fresh solves of alpha_i_uncertainty.py locate (slow test below)
    lad = _load("dk_moduli_ladder.npz")
    assert 1.5e-6 < float(lad["alpha_I_zero"]) - ALPHA_I_PRECISE < 2.1e-6


def test_d3_ritz_bound_recomputed():
    # app. D.2: alpha_c(3) > 3.0370 from the exact Rayleigh quotient on 8 terms
    from backreaction.numerics import d3_blind_check as d3

    for K, bound in D3_RITZ_ALPHA_BOUND.items():
        assert float(6 / d3.ritz_bound(K) ** 2) == pytest.approx(bound, abs=5e-8)
    assert D3_RITZ_ALPHA_BOUND[8] < D3_ALPHA_C


def test_probe_bc_two_methods_recomputed():
    # an extra: the probe B_c u_H^2 by Frobenius/Wronskian and by mpmath
    # Chebyshev collocation (probe_bc_precise.py runs them to 30 digits); the
    # paper has only table 4's reference value
    import mpmath as mp

    from backreaction.numerics import probe_bc_precise as pb

    with mp.workdps(30):
        a, b = float(pb.bc_frobenius(60)), float(pb.bc_colloc(30))
    assert a == pytest.approx(PROBE_BC_PRECISE, abs=2e-15)
    assert b == pytest.approx(PROBE_BC_PRECISE, abs=2e-15)


@pytest.mark.slow
def test_onset_ladder_two_methods():
    # app. D.2: horizon shooting at every rung of the collocation ladder
    from backreaction.numerics import onset_scan

    sc, t = _load("onset_scan.npz"), _load("dk_throat.npz")
    da = db = 0.0
    for bL, aL, BL in zip(t["beta"], t["alpha"], t["B_c"], strict=True):
        r = onset_scan.at_beta(float(bL), sc["eps"], sc["beta"], sc["nu"])
        da, db = max(da, abs(r["alpha"] / aL - 1)), max(db, abs(r["Bc"] / BL - 1))
    assert da == pytest.approx(ONSET_LADDER_TWO_METHOD[0], rel=0.1)
    assert db == pytest.approx(ONSET_LADDER_TWO_METHOD[1], rel=0.1)


@pytest.mark.slow
def test_alpha_i_fresh_solves(tmp_path):
    # app. D.5: alpha_I = 0.4939747 +- 2e-7 by fresh solves (~3 min)
    import re
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "backreaction.numerics.alpha_i_uncertainty"],
        capture_output=True, text=True, check=True,
    ).stdout
    m = re.search(r"RESULT  alpha_I = ([0-9.]+) \+/- ([0-9.e-]+)", out)
    assert m, out[-2000:]
    assert float(m.group(1)) == pytest.approx(ALPHA_I_PRECISE, abs=1e-7)
    assert float(m.group(2)) == pytest.approx(ALPHA_I_PRECISE_UNC, rel=0.01)
