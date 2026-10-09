"""The committed .npz caches still agree with the published numbers.

These files are build artefacts that are nevertheless checked in, because
regenerating them costs minutes (dk_kernel) or is a prerequisite of other
scripts (kernel.npz feeds lattice_scan and the finite-alpha anchors).  That
makes a stale or corrupted cache a silent-failure route: downstream scripts
would keep running and keep printing numbers.  These tests close it.

They check the caches as data.  The slow tier regenerates most of the
backreaction caches and `kernel.npz` in a scratch copy and compares them with
these committed files (the last test in tests/backreaction/test_scripts.py).
"""

import numpy as np
import pytest

from published_values import (
    ALPHA_I, BETA_I, KERNEL_NONMONOTONE_BETA, LADDER_FD_AGREEMENT, LADDER_TOP_MARGIN,
    WEIGHTED_KERNEL_CONCAVE_BETA, WEIGHTED_KERNEL_D2_AT_4, WEIGHTED_KERNEL_NONMONOTONE_BETA,
    THROAT_ABELIAN_MARGIN, THROAT_FULL_MARGIN, THROAT_FULL_MARGIN_SHOOT, THROAT_GAMMA0,
    THROAT_GAMMA0_FD, THROAT_GAMMA0_SHOOT, THROAT_SHOOT_FD_DEV,
    THROAT_KERNEL_PEAK, THROAT_LADDER_WORST_AT_1E8, THROAT_S0,
    B_C, C4, DK_BETA_MAX, DK_C4_AT_ZERO_BETA, DK_GAP_AT_ZERO_BETA,
    DK_GAP_FINAL, DK_GAP_MINIMUM, DK_PROBE_GAP,
)


@pytest.fixture(scope="module")
def kernel():
    return np.load("scripts/kernel.npz")


@pytest.fixture(scope="module")
def dk_kernel():
    return np.load("backreaction/data/dk_kernel.npz", allow_pickle=True)


@pytest.fixture(scope="module")
def dk_selection():
    return np.load("backreaction/data/dk_selection.npz", allow_pickle=True)


# ---------------------------------------------------- probe-limit kernel cache -----

def test_kernel_cache_holds_the_published_constants(kernel):
    assert float(kernel["C4"]) == pytest.approx(C4, abs=5e-8)
    assert float(kernel["B_c"]) == pytest.approx(B_C, abs=5e-9)


def test_kernel_cache_normalisation(kernel):
    """J2 = 1 is the normalisation every quoted number is expressed in."""
    assert float(kernel["J2"]) == pytest.approx(1.0, abs=1e-9)


def test_kernel_cache_is_internally_consistent(kernel):
    """X(0) = 0, X is monotone up in G^2, and never reaches C4."""
    G2, X = kernel["G2grid"], kernel["X"]
    assert X[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(G2) > 0)
    assert np.all(np.diff(X) > 0)
    assert np.all(X < float(kernel["C4"]))


def test_kernel_cache_pole_completeness(kernel):
    """sum_n c_n^2 = C4: the spectral representation is complete."""
    assert np.sum(kernel["cn"] ** 2) / float(kernel["C4"]) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------- finite-alpha brane caches

def test_dk_kernel_reduces_to_the_probe_kernel_at_zero_coupling(dk_kernel):
    """Anchor [1] (paper sec. 5.1, app. C.3): beta -> 0 must give back the probe-limit kernel."""
    beta = dk_kernel["beta"]
    i0 = int(np.argmin(np.abs(beta)))
    assert beta[i0] == pytest.approx(0.0, abs=1e-12)
    assert dk_kernel["C4"][i0].max() == pytest.approx(DK_C4_AT_ZERO_BETA, abs=5e-8)


def test_dk_kernel_is_positive_everywhere(dk_kernel):
    """Paper sec. 5.1, app. C.3: the exact kernel is positive at every harmonic out to
    beta = 45, asserted over the whole grid rather than spot-checked."""
    assert np.all(dk_kernel["K"] > 0)
    assert dk_kernel["beta"].max() == pytest.approx(DK_BETA_MAX, abs=1e-9)


def test_dk_kernel_two_methods_agree(dk_kernel):
    """Chebyshev collocation against uniform-sigma finite differences with
    Richardson extrapolation -- the two-method check, on the grid."""
    rel = np.abs(dk_kernel["K"] - dk_kernel["K_fd"]) / np.abs(dk_kernel["K"])
    assert rel.max() < 1e-4


def test_the_kernel_turns_non_monotone_from_beta_8(dk_kernel):
    """Paper secs. 5.1 and 5.2: K(s) decreases in s on every brane below
    beta = 8, and from beta = 8 on the grid it has an interior maximum."""
    K, beta = dk_kernel["K"], dk_kernel["beta"]
    mono = np.all(np.diff(K, axis=1) < 0, axis=1)
    peaked = np.argmax(K, axis=1) > 0
    assert np.all(mono[beta < KERNEL_NONMONOTONE_BETA])
    assert np.all(peaked[beta >= KERNEL_NONMONOTONE_BETA])


def test_the_weighted_kernel_is_not_completely_monotone_from_beta_4(dk_kernel):
    """App. E: Montgomery's theorem needs f(gamma) = e^{-gamma/2} K(gamma B_c)
    completely monotone, so in particular convex and decreasing.  On the grid f
    is convex for beta <= 3 and concave at small gamma from beta = 4, by both
    discretisations; it stays decreasing up to beta = 25 and rises at small
    gamma from beta = 30, well after K itself turns non-monotone (beta = 8)."""
    g, beta = dk_kernel["gamma"], dk_kernel["beta"]
    h = np.diff(g)
    for key in ("K", "K_fd"):
        f = np.exp(-g / 2) * dk_kernel[key] / dk_kernel["C4"]
        df = np.diff(f, axis=1) / h
        d2 = 2 * (df[:, 1:] - df[:, :-1]) / (h[1:] + h[:-1])
        convex = np.all(d2 > 0, axis=1)
        assert np.all(convex[beta < WEIGHTED_KERNEL_CONCAVE_BETA])
        assert not np.any(convex[beta >= WEIGHTED_KERNEL_CONCAVE_BETA])
        i = int(np.argmin(np.abs(beta - WEIGHTED_KERNEL_CONCAVE_BETA)))
        for gam, want in WEIGHTED_KERNEL_D2_AT_4.items():
            j = int(np.argmin(np.abs(g[1:-1] - gam)))
            assert d2[i, j] == pytest.approx(want, abs=2e-3)
        assert np.all(d2[i, g[1:-1] > 0.08] > 0)  # concave only at gamma <= 0.075
        decreasing = np.all(df < 0, axis=1)
        assert np.all(decreasing[beta < WEIGHTED_KERNEL_NONMONOTONE_BETA])
        assert not np.any(decreasing[beta >= WEIGHTED_KERNEL_NONMONOTONE_BETA])


def test_gravity_lowers_the_kernel_below_the_photon_only_kernel(dk_kernel):
    """Paper sec. 5.1: the dressed metric pairing P_dress is positive at every
    beta > 0 on the grid, so K = C4 - X_ab - P_dress/2 lies below the
    photon-only kernel C4 - X_ab (the cache's X is X_ab, its P is P_dress)."""
    K, C4, X, P, beta = (dk_kernel[k] for k in ("K", "C4", "X", "P", "beta"))
    np.testing.assert_allclose(K, C4 - X - P / 2, atol=1e-10)
    assert np.all(P[beta > 0] > 0)
    assert np.all(P[beta == 0] == 0)


def test_dk_kernel_is_real(dk_kernel):
    """The kernel is a real quantity; the stored imaginary residue is a
    numerics monitor and must stay negligible."""
    assert np.abs(dk_kernel["imag"]).max() < 1e-8


def test_finite_alpha_selection_stays_triangular(dk_selection):
    """The selection verdict (paper sec. 5.2, app. D.5): the gap C_sq - C_tri never changes sign on
    the grid, although the O(alpha) slope alone would reach zero near
    alpha ~ 0.11."""
    gap = dk_selection["gap"]
    assert np.all(gap > 0)


def test_finite_alpha_gap_matches_the_probe_value_at_zero_coupling(dk_selection):
    """Anchor [1] (paper sec. 5.2, app. D.5).  DK_PROBE_GAP is an external reference value, from a
    separate perturbative code that is not part of this repository."""
    gap, beta = dk_selection["gap"], dk_selection["beta"]
    i0 = int(np.argmin(np.abs(beta)))
    assert gap[i0] == pytest.approx(DK_GAP_AT_ZERO_BETA, abs=5e-9)
    assert gap[i0] == pytest.approx(DK_PROBE_GAP, abs=1e-10)  # measured 9.8e-12
    # The stored probe_gap is dk_selection's own reference constant, which the
    # script asserts against the probe lattice code at run time; comparing it
    # with DK_PROBE_GAP here would compare two literals, so it is not repeated.


def test_finite_alpha_gap_dips_then_recovers(dk_selection):
    """The shape of the gap (paper sec. 5.2, app. D.5): a 0.6% dip near alpha = 0.068, then a rise to
    1.8% above the probe value.  The non-monotonicity is the physical content --
    the resummation reverses the linear trend rather than merely slowing it."""
    gap = dk_selection["gap"]
    assert gap.min() == pytest.approx(DK_GAP_MINIMUM, abs=5e-8)
    assert gap[-1] == pytest.approx(DK_GAP_FINAL, abs=5e-8)
    assert 0 < int(np.argmin(gap)) < len(gap) - 1      # an interior minimum
    assert gap[-1] > gap[0]                            # and it ends above probe


# ------------------------------------------------- the lattice at alpha_*

@pytest.fixture(scope="module")
def throat_lattice():
    return np.load("backreaction/data/dk_throat_lattice.npz")


@pytest.fixture(scope="module")
def throat_channels():
    return np.load("backreaction/data/dk_throat_channels.npz")


def test_throat_abelian_kernel_and_margin(throat_lattice):
    """S(0) of the BTZ x R^2 abelian kernel, and the relative margin of the
    square over the triangle in the abelian sector alone (paper apps. C.4
    and E.4)."""
    d = throat_lattice
    assert float(d["S0"]) == pytest.approx(THROAT_S0, abs=5e-6)
    margin = (float(d["C_sq"]) - float(d["C_tri"])) / float(d["C_tri"])
    assert margin == pytest.approx(THROAT_ABELIAN_MARGIN, abs=5e-4)


def test_throat_full_kernel_sign_change_and_margin(throat_channels):
    """With the metric sector the kernel is negative below gamma_0, peaks at
    0.051 (a regression value), and the triangular margin drops to +0.1766
    (paper secs. 5.1 and 5.2)."""
    d = throat_channels
    assert float(d["gamma0"]) == pytest.approx(THROAT_GAMMA0, abs=5e-5)
    assert float(np.max(d["S"])) == pytest.approx(THROAT_KERNEL_PEAK, abs=5e-4)
    margin = (float(d["C_sq"]) - float(d["C_tri"])) / float(d["C_tri"])
    assert margin == pytest.approx(THROAT_FULL_MARGIN, abs=5e-5)


def test_throat_full_kernel_second_method(throat_channels):
    """Check [C10] (table 4): shooting from both ends against the FD solver --
    S at six gamma, gamma_0 by direct roots of both, the exact-shell margin."""
    d = throat_channels
    g, sh, fd = d["gamma_C10"], d["S_shoot_C10"], d["S_fd_C10"]
    dev = np.abs(sh - fd)
    assert dev[0] == pytest.approx(THROAT_SHOOT_FD_DEV[0.02], abs=5e-8)
    assert dev[1] == pytest.approx(THROAT_SHOOT_FD_DEV[0.3], abs=5e-9)
    assert dev[g >= 7.0].max() < THROAT_SHOOT_FD_DEV["shells"] + 2e-9
    assert float(d["gamma0"]) == pytest.approx(THROAT_GAMMA0_SHOOT, abs=2e-9)
    assert float(d["gamma0_fd"]) == pytest.approx(THROAT_GAMMA0_FD, abs=2e-9)
    assert abs(float(d["gamma0"]) - float(d["gamma0_fd"])) < 1e-7
    assert float(d["gamma0_pchip"]) - float(d["gamma0"]) == pytest.approx(-2.7e-6, abs=1e-7)
    m_sh = (float(d["C_sq_shoot"]) - float(d["C_tri_shoot"])) / float(d["C_tri_shoot"])
    m_fd = (float(d["C_sq"]) - float(d["C_tri"])) / float(d["C_tri"])
    assert m_sh == pytest.approx(THROAT_FULL_MARGIN_SHOOT, abs=5e-8)
    assert abs(m_sh - m_fd) < 1e-7
    assert m_sh == pytest.approx(THROAT_FULL_MARGIN, abs=5e-6)


def test_the_brane_ladder_converges_onto_the_throat_kernel(throat_channels):
    """Check [C9] (paper app. D.5): the brane kernel at fixed gamma, rung by
    rung up the ladder, against the throat kernel -- the worst deviation
    shrinks at every rung and is 0.009 at beta = 1e8.  Also guards the cache
    itself: a run without --ladder would leave these arrays empty."""
    from backreaction.numerics.dk_throat_channels import _interp

    d = throat_channels
    assert list(d["ladder_beta"]) == [1e4, 1e5, 1e6, 1e7, 1e8]
    f = _interp(d["gamma"], d["S"])
    target = np.array([float(f(g)) for g in (0.3, 0.9, 7.5, 20.0)])
    worst = np.abs(d["ladder_K"] - target).max(axis=1)
    assert np.all(np.diff(worst) < 0)
    assert worst[-1] == pytest.approx(THROAT_LADDER_WORST_AT_1E8, abs=5e-4)


def test_ladder_cells_two_methods_agree():
    """Check [K7] (table 4): C_tri and C_sq on every ladder rung by finite
    differences (Richardson M = 1600/3200) against collocation."""
    d = np.load("backreaction/data/dk_long_wavelength.npz")
    dt = np.abs(d["C_tri_fd"] - d["C_tri"])
    ds = np.abs(d["C_sq_fd"] - d["C_sq"])
    assert dt.max() == pytest.approx(LADDER_FD_AGREEMENT[0], rel=0.05)
    assert ds.max() == pytest.approx(LADDER_FD_AGREEMENT[1], rel=0.05)
    assert d["beta"][int(np.argmax(ds))] == 1e8
    assert dt[d["beta"] <= 1e6].max() < 5e-11
    m_c = (d["C_sq"][-1] - d["C_tri"][-1]) / d["C_tri"][-1]
    m_f = (d["C_sq_fd"][-1] - d["C_tri_fd"][-1]) / d["C_tri_fd"][-1]
    assert abs(m_f - m_c) < 1e-8
    assert m_f == pytest.approx(LADDER_TOP_MARGIN, abs=5e-4)


def test_the_bravais_boundedness_threshold_alpha_I():
    """Paper sec. 5.2, app. D.5: the Gaussian average I(beta) of the kernel changes sign at
    alpha_I = 0.494 (beta_I = 1.569e6), where the quartic functional stops being
    bounded below on Bravais lattices; triangular stays the minimiser at every
    rung, and the top-rung margin over the square is 0.183."""
    d = np.load("backreaction/data/dk_moduli_ladder.npz", allow_pickle=True)
    assert float(d["alpha_I_zero"]) == pytest.approx(ALPHA_I, abs=5e-5)
    assert float(d["beta_I_zero"]) == pytest.approx(BETA_I, rel=5e-4)
    I = d["I_stripe"]
    assert I[0] > 0 and I[-1] < 0 and np.sum(np.diff(np.sign(I)) != 0) == 1
    assert np.max(d["dtau"]) < 1e-7
    margin = (d["C_sq"][-1] - d["C_tri"][-1]) / d["C_tri"][-1]
    assert d["beta"][-1] == 1e8
    assert margin == pytest.approx(LADDER_TOP_MARGIN, abs=5e-4)


# ------------------------------------------------- figures the paper includes
def test_every_figure_the_paper_includes_is_tracked_by_git():
    """A figure that exists locally but is not committed (e.g. caught by a
    .gitignore rule) builds fine here and breaks every fresh clone."""
    import pathlib
    import re
    import subprocess

    root = pathlib.Path(__file__).resolve().parent.parent
    tex = "".join(p.read_text() for p in sorted((root / "paper").glob("**/*.tex")))
    names = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    assert names, "no \\includegraphics found in paper/"
    try:
        out = subprocess.run(
            ["git", "ls-files", "backreaction/figures"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    missing = [n for n in names if f"backreaction/figures/{n}.pdf" not in out]
    assert not missing, f"figures included by the paper but not tracked by git: {missing}"
