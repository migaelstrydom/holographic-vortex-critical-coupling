"""The probe-limit lattice: the triangular lattice wins (paper app. E).

The structural claim is that
the exchange kernel C4 - X(G^2) is completely monotone and positive, so
Montgomery's theorem selects the triangular lattice over the whole
one-flux-quantum Bravais family; these tests pin the numbers that claim rests
on, and check the moduli-space minimisation that confirms it for this kernel.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

import lattice_scan
import lll_identity
from published_values import (
    ABRIKOSOV_BETA_SQUARE, ABRIKOSOV_BETA_TRIANGULAR, B_C, C4,
    CONDENSATION_ENERGY_RATIO, C_SQUARE, C_TRIANGULAR,
    EFFECTIVE_RATIO_SQUARE, EFFECTIVE_RATIO_TRIANGULAR,
    LATTICE_GAP_PERCENT, SCREENING_AT_FIRST_SHELL,
    C_TRIANGULAR_G_NONZERO, C_SQUARE_G_NONZERO, PROBE_RELATIVE_MARGIN,
)

TAU_TRIANGULAR = (0.5, np.sqrt(3) / 2)
TAU_SQUARE = (0.0, 1.0)
SIGMA_SHELL = 4 * np.pi / np.sqrt(3)   # G_1^2/B_c, triangular first shell


@pytest.fixture(scope="module")
def C_tri():
    return lattice_scan.quartic_coefficient(*TAU_TRIANGULAR)


@pytest.fixture(scope="module")
def C_sq():
    return lattice_scan.quartic_coefficient(*TAU_SQUARE)


def test_contact_coefficient(C_tri):
    """C4 = 1.5805931 (app. E.4)."""
    assert lattice_scan.C4 == pytest.approx(C4, abs=5e-8)


def test_quartic_coefficient_at_the_two_lattices(C_tri, C_sq):
    """C4/2 + C(tau) at the two lattices, to 7 decimals (regression values: app.
    E.4 quotes C(tau) without the G = 0 term, as in the next test)."""
    assert C_tri == pytest.approx(C_TRIANGULAR, abs=5e-8)
    assert C_sq == pytest.approx(C_SQUARE, abs=5e-8)


def test_the_g_nonzero_convention(C_tri, C_sq):
    """The same numbers with the G = 0 term C4/2 dropped (the convention of
    the finite-alpha work and the paper), and the +0.2460 margin that
    every finite-alpha margin is compared against."""
    Ct, Cs = C_tri - lattice_scan.C4 / 2, C_sq - lattice_scan.C4 / 2
    assert Ct == pytest.approx(C_TRIANGULAR_G_NONZERO, abs=5e-8)
    assert Cs == pytest.approx(C_SQUARE_G_NONZERO, abs=5e-8)
    assert (Cs - Ct) / Ct == pytest.approx(PROBE_RELATIVE_MARGIN, abs=5e-5)


def test_triangular_beats_square(C_tri, C_sq):
    """The probe-limit result (app. E): lower C means lower free energy."""
    assert C_tri < C_sq


def test_lattice_gap_and_condensation_energy_ratio(C_tri, C_sq):
    """The 0.561% gap and the F_tri/F_sq = 1.005611 it implies (F = -a^2/4C)."""
    assert 100 * (C_sq - C_tri) / C_tri == pytest.approx(LATTICE_GAP_PERCENT, abs=5e-4)
    assert C_sq / C_tri == pytest.approx(CONDENSATION_ENERGY_RATIO, abs=5e-7)


def test_effective_abrikosov_ratios(C_tri, C_sq):
    """2C/C4, the holographic analogue of the Abrikosov ratio (a regression
    value; not quoted in the paper)."""
    assert 2 * C_tri / lattice_scan.C4 == pytest.approx(EFFECTIVE_RATIO_TRIANGULAR, abs=5e-5)
    assert 2 * C_sq / lattice_scan.C4 == pytest.approx(EFFECTIVE_RATIO_SQUARE, abs=5e-5)


def test_bulk_exchange_screens_most_of_the_lattice_discrimination(C_tri, C_sq):
    """The bulk exchange screens ~70% of the bare Ginzburg-Landau lattice
    discrimination (a regression check; not quoted in the paper).

    The bare Ginzburg-Landau gap is (beta_sq - beta_tri)/beta_tri; the
    holographic one is the 2C/C4 gap.  The claim is a ratio, so it is checked
    as one rather than by re-quoting the rounded 70%.
    """
    bare = (ABRIKOSOV_BETA_SQUARE - ABRIKOSOV_BETA_TRIANGULAR) / ABRIKOSOV_BETA_TRIANGULAR
    holographic = (C_sq - C_tri) / C_tri
    assert 0.65 < 1 - holographic / bare < 0.75


def test_screening_fraction_at_the_first_reciprocal_shell():
    """X/C4 = 0.854 at sigma_1 = 4 pi/sqrt 3, the shell the lattice lives on."""
    X_shell = float(lattice_scan.Xs(SIGMA_SHELL * B_C))
    assert X_shell / lattice_scan.C4 == pytest.approx(SCREENING_AT_FIRST_SHELL, abs=5e-4)


def test_exchange_never_overcomes_contact():
    """Positivity of the screened repulsion (app. E.3): C4 - X(G^2) > 0 for
    every harmonic.

    This is what forbids the exchange from reversing the lattice ordering, so
    it is asserted over the whole tabulated kernel, not just at the shell.
    """
    kernel = lattice_scan.C4 - lattice_scan.X
    assert np.all(kernel > 0)


def test_kernel_is_monotone_decreasing_in_G2():
    """Complete monotonicity is the hypothesis Montgomery's theorem needs; its
    first consequence, a decreasing kernel, is checked directly."""
    kernel = lattice_scan.C4 - lattice_scan.X
    assert np.all(np.diff(kernel) < 0)


def test_moduli_minimum_is_the_triangular_point():
    """Unconstrained minimisation lands on tau = e^{i pi/3} (app. E.4).

    Started away from the answer, a Nelder-Mead refinement alone reaches the
    triangular point to 1e-7.
    """
    res = minimize(lambda p: lattice_scan.quartic_coefficient(*p),
                   x0=[0.35, 1.05], method="Nelder-Mead",
                   options=dict(xatol=1e-10, fatol=1e-14, maxiter=2000))
    assert res.success
    offset = np.hypot(res.x[0] - TAU_TRIANGULAR[0], res.x[1] - TAU_TRIANGULAR[1])
    assert offset < 1e-7


def test_lll_identity_and_bare_abrikosov_ratios():
    """The LLL identity of app. E.2: |lambda_G| = e^{-G^2/4B} exactly, and the
    theta-sum beta it implies reproduces the textbook Abrikosov ratios.

    check_lattice asserts the identity internally (to 1e-10) and returns beta.
    """
    beta_tri = lll_identity.check_lattice("triangular", 0.5 + 1j * np.sqrt(3) / 2)
    beta_sq = lll_identity.check_lattice("square", 1j)
    assert beta_tri == pytest.approx(ABRIKOSOV_BETA_TRIANGULAR, abs=5e-8)
    assert beta_sq == pytest.approx(ABRIKOSOV_BETA_SQUARE, abs=5e-8)
    assert beta_tri < beta_sq
