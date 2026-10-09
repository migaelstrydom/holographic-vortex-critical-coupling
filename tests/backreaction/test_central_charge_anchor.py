"""The N = 2 Ward-identity anchor for the boundary reading of alpha (paper sec. 4.4).

`central_charge_anchor` is a self-asserting derivation module: its `main()`
records every numbered check in `CHECKS`, exits non-zero if one fails, and
returns its namespace, which these tests read.  It is cheap enough (about 2 s: five Wick contractions in explicit d = 4
coordinates plus rational bookkeeping) to live in the fast tier, and it is
worth having there because it is the check that turns the quoted holographic
normalisations behind alpha = C_J/C_T (paper sec. 4.4, app. A.2) into an anchored statement.  The slow scrape in
`test_scripts.py` covers the same module as a subprocess.

The values are exact rationals and are asserted exactly: the whole content of
the anchor is that two protected field-theory numbers and one holographic number are
the same fraction, so an approximate agreement would be a failure.
"""

import functools
import types
from fractions import Fraction

import sympy as sp

from backreaction.derivations import central_charge_anchor
from published_values import (
    ALPHA_SUSY, CJ_CT_SUSY, CJ_N4_PER_ADJOINT_PI4, CJ_SU2R_OVER_C_PI4,
    CR_N1_OVER_CT, CR_U1R_OVER_C_PI4, CT_N4_PER_ADJOINT_PI4,
)


@functools.cache
def _results(module):
    """A derivation module's definitions plus whatever its main() computed."""
    ns = dict(vars(module))
    if hasattr(module, "main"):
        ns.update(vars(module.main()))
    return types.SimpleNamespace(**ns)


def _cca():
    return _results(central_charge_anchor)


def _frac(x):
    x = sp.nsimplify(sp.simplify(x))
    return Fraction(int(x.p), int(x.q))


def test_every_numbered_check_passed():
    cca = _cca()
    failed = [name for name, ok in cca.CHECKS.items() if not ok]
    assert not failed, failed
    assert len(cca.CHECKS) >= 38


def test_free_field_coefficients_match_osborn_petkou():
    """The Wick engine against OP (5.5), (5.6), (5.16) in d = 4."""
    cca = _cca()
    pi4 = sp.pi**4
    assert sp.simplify(cca.CJ_CSCALAR * pi4) == sp.Rational(1, 4)
    assert sp.simplify(cca.CJ_DIRAC * pi4) == 1
    assert sp.simplify(cca.CT_SCALAR * pi4) == sp.Rational(1, 3)
    assert sp.simplify(cca.CT_DIRAC * pi4) == 2
    assert sp.simplify(cca.CT_VECTOR * pi4) == 4


def test_route_2_free_n4_sym():
    cca = _cca()
    assert _frac(cca.CJ_N4 * sp.pi**4) == CJ_N4_PER_ADJOINT_PI4
    assert _frac(cca.CT_N4 * sp.pi**4) == CT_N4_PER_ADJOINT_PI4
    assert _frac(cca.RATIO_ROUTE2) == CJ_CT_SUSY


def test_route_1_ward_identities():
    cca = _cca()
    c = sp.Symbol("c", positive=True)
    assert _frac(cca.sol[cca.CI3_sym] * sp.pi**4 / c) == CJ_SU2R_OVER_C_PI4
    assert _frac(cca.sol[cca.Cr_sym] * sp.pi**4 / c) == CR_U1R_OVER_C_PI4
    assert _frac(cca.RATIO_ROUTE1) == CJ_CT_SUSY
    assert _frac(cca.CR_c / cca.CT_c) == CR_N1_OVER_CT


def test_holography_at_the_romans_point_matches_both_routes():
    cca = _cca()
    # alpha_SUSY is DERIVED in susy_coupling (from the Romans / LPT action, for
    # any g1/g2 split); central_charge_anchor takes it as an input, so check
    # the input against the derivation rather than against itself.
    import sympy as sp

    from backreaction.derivations import susy_coupling

    sc = _results(susy_coupling)
    assert all(sc.CHECKS.values())
    assert sp.simplify(sc.alpha_gen - cca.ALPHA_SUSY) == 0
    assert float(cca.ALPHA_SUSY) == ALPHA_SUSY
    assert _frac(cca.RATIO_HOLO) == CJ_CT_SUSY
    assert cca.RATIO_ROUTE1 == cca.RATIO_ROUTE2 == cca.RATIO_HOLO
