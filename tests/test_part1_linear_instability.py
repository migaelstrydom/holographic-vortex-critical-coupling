"""The linear instability and the critical field (paper sec. 3.1, app. D.1).

The verification standard is two
independent numerical methods per number; here that is horizon-to-boundary
shooting (scripts/critical_field.py) against Chebyshev spectral collocation
(scripts/spectral_check.py), which share no code, no discretisation and no
boundary treatment.
"""

import numpy as np
import pytest

import critical_field
import spectral_check
from published_values import B_C, B_TOWER

# find_eigenvalues returns (eigenvalues, scan_grid, scan_values); a coarse scan
# below B_1 = 22.48 is enough to bracket B_c alone, and is ~20x cheaper than the
# default scan that resolves the whole tower.
def _shoot_bc(**kw):
    return critical_field.find_eigenvalues(b_max=10.0, n_scan=200, **kw)[0][0]


@pytest.fixture(scope="module")
def spectral_tower():
    return spectral_check.critical_fields(N=80, n_keep=len(B_TOWER))


def test_spectral_tower_matches_the_published_tower(spectral_tower):
    """B_c and its first overtones to 8 decimals (the paper quotes B_c; the
    overtones are regression values)."""
    assert spectral_tower == pytest.approx(B_TOWER, abs=5e-9)


def test_two_methods_agree_on_bc(spectral_tower):
    """The verification standard itself: shooting vs collocation on B_c."""
    shooting = _shoot_bc()
    assert shooting == pytest.approx(spectral_tower[0], abs=1e-8)
    assert shooting == pytest.approx(B_C, abs=5e-9)


@pytest.mark.slow
def test_shooting_reproduces_the_whole_tower():
    """The independent method on every level of the tabulated tower."""
    eigs = critical_field.find_eigenvalues(b_max=100.0)[0]
    assert eigs[:len(B_TOWER)] == pytest.approx(B_TOWER, abs=5e-9)


@pytest.mark.parametrize("N", [40, 60, 120])
def test_spectral_method_is_converged_by_40_points(N, spectral_tower):
    """Collocation is converged to 5e-9 by 40 grid points."""
    assert spectral_check.critical_fields(N, n_keep=len(B_TOWER)) == \
        pytest.approx(spectral_tower, abs=5e-9)


@pytest.mark.parametrize("u_eps,delta", [(1e-3, 1e-5), (1e-4, 1e-6), (1e-5, 1e-7)])
def test_shooting_is_insensitive_to_both_cutoffs(u_eps, delta, monkeypatch):
    """Shooting is unchanged to 5e-9 when either cutoff is varied.

    Both cutoffs are module-level constants read inside source_coefficient, so
    they are patched rather than passed.
    """
    monkeypatch.setattr(critical_field, "U_EPS", u_eps)
    monkeypatch.setattr(critical_field, "DELTA", delta)
    assert _shoot_bc() == pytest.approx(B_C, abs=5e-9)


def test_bc_is_the_lowest_mode_and_the_tower_is_ordered(spectral_tower):
    """The instability sets in at B_c: nothing condenses below it."""
    assert spectral_tower[0] == min(spectral_tower)
    assert np.all(np.diff(spectral_tower) > 0)
