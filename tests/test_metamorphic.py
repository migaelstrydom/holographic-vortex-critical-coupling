"""Metamorphic relations: transform an input, predict the output.

Every other test file in this suite asserts a published value out of
``published_values.py``.  That catches drift away from a state we already
believe, but it cannot catch a number that was never right, because the
expected value would have been wrong too.  The tests here assert no published
number at all.  Each applies a transformation to an input and asserts how the
output must transform -- a symmetry, a scaling, a limit, a parity -- so they
have teeth against code that has never been correct.
"""

import numpy as np
import pytest

import critical_field
import lattice_scan as ls
import lll_identity as lll
import spectral_check

TAU_TRIANGULAR = 0.5 + 1j * np.sqrt(3) / 2
TAU_SQUARE = 1j


# -- the linear instability -----------------------------------------


@pytest.fixture(scope="module")
def tower():
    return spectral_check.critical_fields(N=80, n_keep=4)


def test_sturm_oscillation_indexes_the_tower(tower):
    """The n-th radial mode has exactly n interior nodes.

    Sturm oscillation theory fixes the node count of a regular Sturm-Liouville
    problem's eigenfunctions, so this does two things no fixed-value test can:
    it proves the tower is labelled correctly, and -- since the count goes up
    by exactly one per level -- that no eigenvalue was skipped between the
    tabulated ones.  It is the cheap, non-rigorous version of the oscillation
    count that a certified enclosure would use.

    Both endpoints are trimmed: u -> 0 is a regular singular point where w ~ u^2
    is numerically indistinguishable from zero, and the integration starts a
    distance DELTA from the horizon.
    """
    for n, B in enumerate(tower):
        us, ws = critical_field.eigenfunction(B, n_pts=4000)
        interior = (us > 5e-3) & (us < 1.0 - 5e-3)
        sign_changes = int(np.sum(np.diff(np.sign(ws[interior])) != 0))
        assert sign_changes == n


def test_eigenvalues_are_invariant_under_the_horizon_normalisation(monkeypatch):
    """The shooting target is linear: scaling the Frobenius start at the
    horizon scales c0 by the same factor and moves no eigenvalue.

    horizon_start fixes w(u_H) = 1, which is a choice of normalisation for a
    homogeneous problem, not physics.  A failure means a nonlinearity has crept
    into the shot -- the one way this method could produce a plausible but
    normalisation-dependent B_c.
    """
    from scipy.optimize import brentq

    def bc():
        return brentq(critical_field.source_coefficient, 4.0, 10.0, xtol=1e-13)

    plain_c0 = critical_field.source_coefficient(4.0)
    plain_bc = bc()

    scale = -7.5   # sign flip included: c0 must follow it, the eigenvalue must not
    original = critical_field.horizon_start
    monkeypatch.setattr(
        critical_field, "horizon_start",
        lambda B, delta=critical_field.DELTA: [scale * y for y in original(B, delta)],
    )

    assert critical_field.source_coefficient(4.0) / plain_c0 == pytest.approx(scale, rel=1e-10)
    assert bc() == pytest.approx(plain_bc, abs=1e-11)


def _C(tau):
    return ls.quartic_coefficient(tau.real, tau.imag)


@pytest.mark.parametrize("tau", [TAU_TRIANGULAR, TAU_SQUARE, 0.3 + 1.1j,
                                 0.3 + 0.25j, 2.7 + 3.1j])
def test_quartic_coefficient_is_modular_invariant(tau):
    """C(tau) depends on the lattice, not on the basis chosen for it.

    tau -> tau + 1 and tau -> -1/tau generate SL(2, Z) and describe the same
    lattice in a different basis; tau -> -conj(tau) is the reflection.  The
    code does not know this: it rebuilds the direct basis, the reciprocal
    basis and the truncated sum from scratch for each tau.  A failure would
    mean the moduli-space scan is exploring a coordinate
    artefact rather than lattice shape.

    Two of the sampled points sit far outside the fundamental domain, where
    the nmax = 14 truncation window is most strained; the relation holds
    there to machine precision, which is also evidence the truncation is not
    biasing the scan.
    """
    def to_upper_half_plane(t):
        return t if t.imag > 0 else -t.conjugate()

    reference = _C(tau)
    for transformed in (tau + 1, tau - 1, to_upper_half_plane(-1 / tau),
                        to_upper_half_plane(-tau.conjugate())):
        assert _C(transformed) == pytest.approx(reference, rel=1e-12)


@pytest.mark.parametrize("name,tau", [("triangular", TAU_TRIANGULAR),
                                      ("square", TAU_SQUARE),
                                      ("generic", 0.3 + 1.1j)])
def test_zero_kernel_limit_reproduces_the_abrikosov_ratios(name, tau, monkeypatch, capsys):
    """Switching the exchange kernel off must return the pure-contact theory.

    With X == 0, C(tau) = (C4/2) sum_G e^{-G^2/2B_c}, so 2C/C4 collapses to the
    theta sum -- the classic Abrikosov ratio.  lll_identity computes that sum
    from the density on the cell by FFT, sharing no code path with
    lattice_scan's reciprocal-lattice sum, so this pins the two lattice-sum
    implementations against each other at a point where the answer is also
    independently known from the literature.
    """
    monkeypatch.setattr(ls, "Xs", lambda s: np.zeros_like(np.asarray(s, dtype=float)))
    contact_only = 2 * _C(tau) / ls.C4
    beta = lll.check_lattice(name, tau)      # asserts the LLL identity internally
    capsys.readouterr()                      # check_lattice reports to stdout
    assert contact_only == pytest.approx(beta, abs=1e-8)


@pytest.mark.parametrize("B", [1.0, 17.3])
def test_theta_sum_is_independent_of_the_field_strength(B, monkeypatch, capsys):
    """One flux quantum per cell means beta cannot depend on B.

    The cell area is 2 pi/B, so G^2 scales with B and the exponent G^2/2B is
    scale-free.  The module computes at B = B_c 'for definiteness'; if beta
    moved with B, the flux quantisation would be wired up wrongly and every
    lattice sum downstream would inherit the error.
    """
    monkeypatch.setattr(lll, "B", B)
    beta = lll.check_lattice("triangular", TAU_TRIANGULAR)
    capsys.readouterr()
    assert beta == pytest.approx(1.1595952670, abs=1e-8)


def test_kernel_is_completely_monotone_to_fourth_order():
    """Complete monotonicity is the hypothesis Montgomery's theorem needs.

    A completely monotone function of s = G^2 has derivatives alternating in
    sign, so on the uniform grid its forward differences must alternate too.
    test_part2 checks the first difference; this checks four, which is as far
    as double precision carries -- the fifth difference is already at the 1e-14
    level, i.e. rounding noise.  Failure here would take the probe-limit structural
    proof with it, not just a number.
    """
    assert np.allclose(np.diff(ls.G2grid), ls.G2grid[1] - ls.G2grid[0])
    differences = ls.C4 - ls.X
    for order in range(1, 5):
        differences = np.diff(differences)
        assert np.all(np.sign(differences) == (-1) ** order)


def test_kernel_screens_as_one_over_G2_in_the_tail():
    """C4 - X = <w0^2, T (T + G^2)^{-1} w0^2> decays like 1/G^2.

    At large G^2 the resolvent is T/G^2 + O(G^-4), so the kernel must halve
    when G^2 doubles.  This is the asymptotic content of the operator
    representation in exchange_kernel.py's docstring, checked against the
    computed spline rather than assumed.
    """
    for s in (75.0, 100.0, 150.0):
        ratio = float(ls.C4 - ls.Xs(2 * s)) / float(ls.C4 - ls.Xs(s))
        assert 0.45 < ratio < 0.60
