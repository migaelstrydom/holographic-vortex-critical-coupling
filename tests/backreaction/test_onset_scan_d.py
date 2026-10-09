"""Published numbers of the general-d finite-temperature onset scan (paper sec. 4.3, app. D.2), read
from the committed cache `onset_scan_d.npz`.

Cache checks only: they catch a corrupted or silently regenerated cache.  The
code path is exercised by the slow-tier scrape in `test_scripts.py`, which runs
`numerics/onset_scan_d.py --quick` and requires every check to pass.
"""

import numpy as np
import pytest

from backreaction import paths
from published_values import (
    ONSET_D_ALPHA_MAX_OVER_STAR,
    ONSET_D_BETA_MAX,
    ONSET_D_REFINE_MAX,
    ONSET_D_TWO_METHOD_MAX,
)


@pytest.fixture(scope="module")
def cache():
    return np.load(paths.data("onset_scan_d.npz"), allow_pickle=False)


@pytest.mark.parametrize("d", [5, 6])
def test_onset_monotone_below_alpha_star(cache, d):
    beta, alpha = cache[f"beta_d{d}"], cache[f"alpha_d{d}"]
    ast = float(cache[f"alpha_star_d{d}"])
    assert ast == pytest.approx(16 / (d * (d - 1) * (d - 2)), rel=1e-15)
    assert np.all(np.diff(beta) > 0)
    assert np.all(np.diff(alpha) > 0)
    assert np.all(cache[f"s_d{d}"] > 0)
    assert np.all(cache[f"nodes_d{d}"] == 0)
    assert alpha.max() / ast == pytest.approx(ONSET_D_ALPHA_MAX_OVER_STAR[d], abs=5e-9)
    assert beta.max() == pytest.approx(ONSET_D_BETA_MAX[d], rel=1e-3)
    # the identity alpha/alpha_* = (1 - eps/beta_ext)/(1 + s)^2
    be = d * (d - 1) / (d - 2)
    rhs = (1 - cache[f"eps_d{d}"] / be) / (1 + cache[f"s_d{d}"]) ** 2
    assert np.max(np.abs(alpha / ast - rhs)) < 1e-13


@pytest.mark.parametrize("d", [5, 6])
def test_two_methods_and_refinement(cache, d):
    m2 = cache[f"m2_d{d}"]
    assert np.max(m2[:, 4]) < ONSET_D_TWO_METHOD_MAX  # B_m, shooting vs Chebyshev
    assert np.max(m2[:, 5]) < ONSET_D_TWO_METHOD_MAX  # v_inf
    assert np.all(m2[:, 7] == 0)
    assert cache[f"refine_d{d}"].max() < ONSET_D_REFINE_MAX


def test_anchors(cache):
    assert cache["d4_dev"].max() < 1e-9
    assert cache["d4_ladder_dev"].max() < 1e-8
    d3 = cache["d3_anchor"]
    assert np.max(np.abs(d3[:, 1] / d3[:, 2] - 1)) < 1e-9
    assert np.max(np.abs(d3[:, 3] / d3[:, 4] - 1)) < 1e-9
    # d = 3 is the counter-example: the onset passes alpha_*(3) = 8/3 (s < 0)
    assert d3[:, 1].max() > 8 / 3 and d3[:, 5].min() < 0
