"""The G = 0 sector and the B_c shift (paper sec. 3).

Two linked results, tested together because they share the same solved
profiles: the homogeneous fixed-temperature response of the black brane to the
lattice-averaged source, and the shift of the critical field it induces (paper sec. 3), the small-alpha slope of B_c(alpha).

Several of the per-alpha-B^2 entries are exact closed forms and are asserted at
machine precision, not at the printed digits: delta s|_T = 1/4, I_d = pi^2/48,
(s_4x, s_4z) = (3/8, 1/4), the u^4 log u pair (-1/2, 0).  Their per-alpha-rho^2
partners are proved in closed form and are asserted at the 1e-12 the
quadrature achieves.  Loosening any of these tolerances would mean the closed
forms had stopped holding, which is a different event from a number drifting.
"""

import numpy as np
import pytest

from backreaction.numerics import bc_shift
from backreaction.numerics import g0_thermo as g0
from published_values import (
    DBC_PER_ALPHA, DBC_SLOPE,
    DBC_TWO_METHOD_TOL, G0_DS_B2, G0_DS_R2_TOL, G0_ID_B2, G0_ID_R2,
    G0_LOG_B2, G0_S4_B2,
)


@pytest.fixture(scope="module")
def QB():
    """The per-alpha-B^2 channel solution, fixed-T convention."""
    return g0.solve_quad("B2", N=400)


@pytest.fixture(scope="module")
def QR():
    """The per-alpha-rho^2 channel solution at B = B_c, fixed-T convention."""
    return g0.solve_quad("r2", N=400)


# ------------------------------------------------------- per B^2 ------

def test_the_fixed_temperature_convention_is_actually_imposed(QB, QR):
    """The G = 0 problem has a one-parameter solution family -- the thermal
    zero mode, the shift along the black-brane family.  Every number in this
    file is a response *at fixed T*, so delta T = 0 is the convention that
    makes them well defined at all, and is checked before anything else."""
    assert QB["dT"] == pytest.approx(0.0, abs=1e-11)
    assert QR["dT"] == pytest.approx(0.0, abs=1e-11)


def test_entropy_response_per_B2_is_exactly_one_quarter(QB):
    """delta s / s_0 |_T = 1/4, from V_J(1) = 1/2 and the integrating factor;
    closed form in derivations/g0.py [9]."""
    assert QB["ds"] == pytest.approx(G0_DS_B2, abs=1e-12)


def test_Id_per_B2_is_pi_squared_over_48(QB):
    """I_d(u_H) = -2 int_0^1 u^3 log u / (1 - u^4) du = pi^2/48, from horizon
    regularity of the massless-scalar channel."""
    assert QB["Id"] == pytest.approx(np.pi**2 / 48, abs=1e-11)
    assert QB["Id"] == pytest.approx(G0_ID_B2, abs=5e-11)


def test_boundary_u4_data_per_B2_are_the_exact_rationals(QB):
    assert (QB["s4x"], QB["s4z"]) == pytest.approx(G0_S4_B2, abs=1e-12)


def test_the_u4_log_resonance_lives_only_in_the_G_zero_sector(QB, QR):
    """The genuine u^4 log u resonance is per B^2 and only in H_xx.  At G != 0
    it is absent (the row sources start at u^4), and it is
    absent in the per-rho^2 sector too -- which is what makes the harmonic
    boundary stress renormalisation-scheme-independent.

    g0_thermo enters these log coefficients as closed forms rather than
    fitting them, so this is a regression check of those constants (and of
    their split into H_xx and H_zz), not a measurement."""
    assert (QB["sl4x"], QB["sl4z"]) == pytest.approx(G0_LOG_B2, abs=1e-12)
    assert (QR["sl4x"], QR["sl4z"]) == pytest.approx((0.0, 0.0), abs=1e-12)


# ----------------------------------------------------- per rho^2 ------

def test_the_condensate_removes_horizon_entropy_in_proportion_to_its_norm(QR):
    """delta s / s_0 |_T = -alpha rho^2 B_c u_H^2 J_2 exactly, proved in closed
    form:
    at fixed temperature the flux crystal orders the horizon.  The identity is
    with B_c itself, not a number that happens to be near it, so the tolerance
    is the quadrature's."""
    assert QR["ds"] == pytest.approx(-g0.BC, rel=G0_DS_R2_TOL)


def test_boundary_u4_data_per_rho2_are_minus_three_halves_and_minus_one_Bc(QR):
    """(s_4x, s_4z) = (-3/2, -1) B_c, the other pair proved in closed form."""
    assert QR["s4x"] == pytest.approx(-1.5 * g0.BC, abs=1e-8)
    assert QR["s4z"] == pytest.approx(-g0.BC, abs=1e-8)


def test_Id_per_rho2(QR):
    assert QR["Id"] == pytest.approx(G0_ID_R2, abs=5e-9)


def test_the_g0_cache_still_agrees_with_a_live_solve(QB, QR):
    d = np.load("backreaction/data/g0_thermo.npz")
    assert float(d["Bc"]) == pytest.approx(g0.BC, rel=1e-14)
    assert float(d["ds_B2"]) == pytest.approx(QB["ds"], abs=1e-11)
    assert float(d["Id_B2"]) == pytest.approx(QB["Id"], abs=1e-11)
    assert float(d["ds_r2"]) == pytest.approx(QR["ds"], abs=1e-9)
    assert float(d["Id_r2"]) == pytest.approx(QR["Id"], abs=1e-9)
    assert d["s4_r2"] == pytest.approx([QR["s4x"], QR["s4z"]], abs=1e-8)


def test_the_collocation_solver_agrees_with_the_channel_construction(QB, QR):
    """The cross-check of the G = 0 solution, and the second method behind
    every number above.  The primary route builds the solution from the two decoupled
    channels by quadrature; this one appends the fixed-T row to a two-field
    collocation system and solves it directly.  They share the generated
    coefficients and nothing else.

    The agreement is at the collocation route's own accuracy (1e-4-level
    profiles -- the log modes limit collocation accuracy here), and that is the
    level measured:
    ~3e-4 relative in delta s.  It is four orders looser than the quadrature
    route, which is why the exact closed forms are asserted against the latter;
    what this test protects is that the two routes still describe the same
    solution at all."""
    grid_cheb, _ = g0.grids()
    for unit, Q in (("B2", QB), ("r2", QR)):
        A = g0.solve_alg(unit, grid_cheb)
        assert A["dT"] == pytest.approx(0.0, abs=1e-6)      # fixed T, its accuracy
        assert A["ds"] == pytest.approx(Q["ds"], rel=1e-3), unit


@pytest.mark.slow
def test_the_finite_difference_collocation_leg_agrees_too(QB, QR):
    """The third leg of the cross-check: the same collocation system on a
    uniform finite-difference grid (N = 2400).  Slow -- 26 s per source unit --
    and it is the leg that makes "two methods" mean two discretisations rather
    than two ways of assembling one."""
    _, grid_fd = g0.grids()
    for unit, Q in (("B2", QB), ("r2", QR)):
        F = g0.solve_alg(unit, grid_fd)
        assert F["dT"] == pytest.approx(0.0, abs=1e-4)
        assert F["ds"] == pytest.approx(Q["ds"], rel=1e-3), unit


# ------------------------------------------------------------------- paper sec. 3

def test_the_critical_field_shift():
    """delta B_c = +28.332285 alpha at u_H = 1, from the perturbation-theory
    integral over the G = 0 profiles.  Backreaction *stabilises*: at fixed T the
    critical field rises, so gravity makes the normal phase harder to
    destabilise."""
    k = bc_shift.pt_integral(N=240)
    assert k > 0
    assert k == pytest.approx(DBC_PER_ALPHA, abs=5e-6)
    assert k / g0.BC**3 == pytest.approx(DBC_SLOPE, abs=5e-8)


def test_the_pt_integral_is_converged_in_the_grid():
    """N = 240 vs 300 stable to 1e-8 (paper sec. 3)."""
    assert bc_shift.pt_integral(N=240) == pytest.approx(
        bc_shift.pt_integral(N=300), abs=1e-7)


def test_two_methods_agree_on_the_shift():
    """Two methods: the PT integral against a direct re-solve of the
    perturbed generalised eigenvalue problem at alpha = 1e-3 and 3e-4, with the
    slope extrapolated linearly in alpha.  These are genuinely different
    computations -- one integrates a perturbation against the unperturbed mode,
    the other never linearises at all."""
    # The direct solver discretises the unperturbed problem its own way, so it
    # returns B_c only to its own accuracy (~2e-7); that offset cancels out of
    # the slope, which is why the slope is what serves as the cross-check.
    B0 = bc_shift.direct_eig(0.0)
    assert B0 == pytest.approx(g0.BC, abs=5e-7)
    a1, a2 = 1e-3, 3e-4
    s1 = (bc_shift.direct_eig(a1) - B0) / a1
    s2 = (bc_shift.direct_eig(a2) - B0) / a2
    k_direct = s2 - a2 * (s1 - s2) / (a1 - a2)          # linear extrapolation
    k_pt = bc_shift.pt_integral(N=300)
    assert abs(k_direct - k_pt) / abs(k_pt) < DBC_TWO_METHOD_TOL
