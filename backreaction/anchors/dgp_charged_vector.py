"""Donos-Gauntlett-Pantelidou (arXiv:1112.4195) charged-vector modes on the
magnetic AdS_3 x R^2 family, re-derived where cheap and evaluated at the
D'Hoker-Kraus point (paper sec. 1).

Their family (DGP eqs 2.2-2.6, U(1)^3 truncation of SO(6) gauged SUGRA):
    L = (R - V) - (1/2) sum_i X_i^{-2} *F^i ^ F^i,   V = -4 sum_i X_i^{-1},
(their Lagrangian also has the Chern-Simons term F^1 ^ F^2 ^ A^3, omitted
here: F^1 ^ F^2 = 0 on these purely magnetic backgrounds, so check [1] does
not need it; the vector masses below are DGP's),
    ds^2 = L^2 ds^2(AdS_3) + dx_1^2 + dx_2^2,   F^i = 2 q_i dx_1 ^ dx_2,
    L^{-2} = sum_i X_i^{-1},   q_i^2 = X_i   (X_1 X_2 X_3 = 1, constants).
Romans' line: X_1 = X_2 = X = exp(-f_1/sqrt6), X_3 = X^{-2}, q_1 = q_2.
The origin f_1 = f_2 = 0 (X_i = 1, q_i^2 = 1, all three fields equal) is the
minimal-gauged-supergravity throat, i.e. the D'Hoker-Kraus AdS_3 x R^2 with
L^2 = 1/3 (DGP intro, and Almuhairi 1011.1266 eq 2.25).

Their charged-vector tower (DGP eq 3.21, appendix B eq B.22), n = 0, vectors
only, for the SO(6) components a_{IJ}, I != J:
    m^2_{0,m} = -2|omega_IJ| - 2 W_IJ V_JI + V_IJ^2,
    omega_IJ = q_I + q_J,  W_IJ = sign(omega)(1/q_I - 1/q_J),  V_IJ = X_I - X_J.
On Romans' line with IJ = 12: W = V = 0 and a_12 is the SU(2) W boson, so
    m^2 = -4 |q_1| = -4 sqrt(X),   L^2 m^2 = -4 sqrt(X) / (2/X + X^2).
DGP section 3.4: BF-violating (L^2 m^2 < -1) for -2.00 <~ f_1 <~ 0.87.

What this script checks, all asserted:
  [1] the background (2.5) solves the trace-reversed Einstein equations and the
      constant-scalar equation of the Lagrangian above, for symbolic X_i;
  [2] the n = 0 vector mass on Romans' line is the g = 2 lowest-Landau-level
      (Nielsen-Olesen) value -B_W for a unit-charge W seeing the proper field
      B_W = 2(2 q_1) of the two U(1)s it couples to -- i.e. eq (3.21) with
      W = V = 0 is the same LLL mode as ours, and b = B_W L^2 is its -m^2 L^2;
  [3] the window: L^2 m^2 = -1 at f_1 = 0.8733 and f_1 = -2.0052 (DGP quote
      0.87 and -2.00), unstable in between;
  [4] at the D'Hoker-Kraus origin L^2 m^2 = -4/3 < -1: the S^5-uplifted DK
      throat is unstable to the W-boson LLL mode (AP 1108.1213 v2 section
      6.2, v1 section 6.3:
      'the simplest case q_1 = q_2 = q_3 is unstable' -- this is the mode);
      the flux the W sees is 2/3 of the total, and reading -m^2 L^2 = 4/3
      through our pure-SU(2) criterion b = sqrt(2/(3 alpha)) would put the
      origin at alpha_eff = 3/8, inside the unstable window alpha < 2/3,
      consistent with alpha_SUSY = 1/4 for the same SU(2) (paper sec. 4.4) --
      the two numbers differ because the SUGRA throat is supported by three
      equal U(1) fluxes, not by the SU(2) Cartan alone (paper sec. 4.4);
  [5] the same 3/8 from the boundary side (paper sec. 4.4): the N = 1
      R-current has C_R/C_T = 4c/40c = 1/10 (paper app. A.2), which the d = 4
      map C_J/C_T = 3 alpha/20 reads as alpha = 2/3 for a vector of unit
      R-charge; with R = (2/3)(Q_1 + Q_2 + Q_3) the W boson a_12, of charges
      (1, 1, 0), has R-charge 4/3, and alpha scales as 1/charge^2 because b
      is linear in the charge, so it sees (2/3)(3/4)^2 = 3/8.  Two
      independent routes to one number: [4] is the gravity side's mass, [5]
      the boundary central charges.

Run:  uv run python -m backreaction.anchors.dgp_charged_vector
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.optimize import brentq


def einstein_check() -> None:
    """[1] DGP eqs (2.4)-(2.5) from R - V - (1/4) sum X_i^{-2} F^i_{mn}F^{i mn}."""
    X1, X2 = sp.symbols("X1 X2", positive=True)
    X3 = 1 / (X1 * X2)
    X = [X1, X2, X3]
    q2 = [Xi for Xi in X]  # q_i^2 = X_i   (2.5)
    V = -4 * sum(1 / Xi for Xi in X)
    # Trace-reversed Einstein equation in D = 5 for this Lagrangian:
    #   R_mn = (V/3) g_mn + (1/2) sum_i X_i^{-2} (F^i_mr F^i_n^r - g_mn F^i^2/6)
    # with F^i = 2 q_i dx1^dx2 on the flat R^2 (proper coordinates):
    #   F_xr F_x^r = 4 q_i^2,  F^2 = 8 q_i^2.
    rhs_xx = V / 3 + sp.Rational(1, 2) * sum(
        (4 * q2[i] - sp.Rational(8, 6) * q2[i]) / X[i] ** 2 for i in range(3)
    )
    rhs_ab = V / 3 + sp.Rational(1, 2) * sum(
        (0 - sp.Rational(8, 6) * q2[i]) / X[i] ** 2 for i in range(3)
    )  # coefficient of g_ab on the AdS_3 block, whose Ricci is -(2/L^2) g_ab
    assert sp.simplify(rhs_xx) == 0, rhs_xx
    Linv2 = sp.simplify(-rhs_ab / 2)
    assert sp.simplify(Linv2 - sum(1 / Xi for Xi in X)) == 0, Linv2
    # Constant-scalar equation: stationarity of V + (1/4) sum X_i^{-2} F^i^2
    # = V + 2 sum q_i^2 X_i^{-2} at fixed q_i, with respect to the two
    # independent scalars (X_3 = 1/(X_1 X_2)).
    qs = sp.symbols("qs1 qs2 qs3", positive=True)  # q_i^2 as free constants
    Xf = [X1, X2, 1 / (X1 * X2)]
    Veff = -4 * sum(1 / Xi for Xi in Xf) + 2 * sum(qs[i] / Xf[i] ** 2 for i in range(3))
    onshell = {qs[i]: Xf[i] for i in range(3)}
    for s in (X1, X2):
        assert sp.simplify(sp.diff(Veff, s).subs(onshell)) == 0
    print(
        "[1] DGP (2.5): L^-2 = sum X_i^-1, q_i^2 = X_i solves Einstein + scalar equations"
    )


def dgp_mass2(q: list[float], X: list[float], I: int, J: int) -> float:
    """DGP eq (3.21), n = 0 vector a_IJ, in full (no W = V = 0 shortcut)."""
    omega = q[I] + q[J]
    W_IJ = np.sign(omega) * (1.0 / q[I] - 1.0 / q[J])
    V_IJ, V_JI = X[I] - X[J], X[J] - X[I]
    return -2 * abs(omega) - 2 * W_IJ * V_JI + V_IJ**2


def romans_point(f1: float) -> tuple[list[float], list[float], float]:
    """(q_i, X_i, L^2) on Romans' line, from DGP (2.5) with q_1 = q_2 > 0."""
    Xr = np.exp(-f1 / np.sqrt(6.0))
    X = [Xr, Xr, Xr**-2]
    q = [np.sqrt(Xi) for Xi in X]  # q_i^2 = X_i
    L2 = 1.0 / sum(1.0 / Xi for Xi in X)
    return q, X, L2


def romans_line(f1: float) -> tuple[float, float, float]:
    """Return (X, L^2, L^2 m^2) for the SU(2) W boson (IJ = 12) on Romans' line."""
    q, X, L2 = romans_point(f1)
    return X[0], L2, L2 * dgp_mass2(q, X, 0, 1)


def main() -> None:
    einstein_check()

    # [2] LLL identification.  Unit-charge W in the SU(2) with F = dA + eps A A:
    # A^{12} = A^{34} = A^3/2 in the SO(6) matrix basis, so the W couples to the
    # sum of the two U(1) fields, proper strength B_W = |F^1_x1x2| + |F^2_x1x2|
    # = 2 q_1 + 2 q_2, and a g = 2 vector in its n-th Landau level has
    # m^2 = (2n + 1) B_W - 2 B_W.  The other side is DGP (3.21) in full, whose
    # W_12 V_21 and V_12^2 terms vanish on Romans' line only because X_1 = X_2.
    for f1 in (0.0, 0.5, -1.0, 2.0):
        q, X, L2 = romans_point(f1)
        B_W = 2 * q[0] + 2 * q[1]
        n = 0
        m2_landau = (2 * n + 1) * B_W - 2 * B_W
        m2_dgp = dgp_mass2(q, X, 0, 1)
        assert np.isclose(m2_dgp, m2_landau), (f1, m2_dgp, m2_landau)
    # control: off Romans' line (X_1 != X_2) the W, V terms do not vanish and
    # the two sides differ, so the agreement above is not built in
    Xg = [0.7, 1.3, 1 / (0.7 * 1.3)]
    qg = [np.sqrt(Xi) for Xi in Xg]
    assert not np.isclose(dgp_mass2(qg, Xg, 0, 1), -(2 * qg[0] + 2 * qg[1]))
    print("[2] DGP (3.21) on Romans' line == g = 2 LLL value -B_W, b = B_W L^2")

    # [3] the window.
    g = lambda f1: romans_line(f1)[2] + 1.0  # noqa: E731
    hi = brentq(g, 0.5, 1.5)
    lo = brentq(g, -3.0, -1.0)
    print(
        f"[3] BF-marginal endpoints on Romans' line: f1 = {lo:.4f}, {hi:.4f}  (DGP: -2.00, 0.87)"
    )
    assert abs(hi - 0.87) < 0.006 and abs(lo + 2.00) < 0.006, (
        lo,
        hi,
    )  # their two decimals
    assert romans_line(0.5 * (lo + hi))[2] < -1  # unstable inside
    assert (
        romans_line(hi + 0.5)[2] > -1 and romans_line(lo - 0.5)[2] > -1
    )  # stable outside
    # supersymmetric point on this line, DGP section 2.3: phi_1 = (2 sqrt6/3) ln 2
    f_susy = 2 * np.sqrt(6) / 3 * np.log(2)
    assert lo < f_susy and f_susy > hi  # SUSY point (1.13) sits in the stable range
    print(
        f"    supersymmetric point f1 = {f_susy:.4f} lies outside the window (stable), as it must"
    )

    # [4] the D'Hoker-Kraus origin.
    X, L2, L2m2 = romans_line(0.0)
    assert np.isclose(L2, 1.0 / 3.0) and np.isclose(L2m2, -4.0 / 3.0)
    b = -L2m2
    alpha_eff = 2.0 / (3.0 * b**2)  # b = sqrt(2/(3 alpha)) inverted
    assert np.isclose(alpha_eff, 3.0 / 8.0)
    print(
        f"[4] DK origin: L^2 = {L2:.4f}, L^2 m^2 = {L2m2:.4f} < -1 (unstable); b = 4/3,"
    )
    print(
        f"    read through b = sqrt(2/(3 alpha)): alpha_eff = {alpha_eff:.4f} = 3/8 < 2/3;"
    )
    print(
        f"    pure-SU(2) DK throat at alpha_SUSY = 1/4 would have b = {np.sqrt(8 / 3):.4f}"
    )

    # [5] the boundary route: C_R/C_T = 1/10, R-charge 4/3.
    r = sp.Rational
    alpha_unit = (r(1, 10)) / r(3, 20)  # invert C_J/C_T = 3 alpha / 20
    q_R = r(2, 3) * (1 + 1 + 0)  # R = (2/3) sum Q_i on a_12
    alpha_W = alpha_unit / q_R**2
    assert alpha_unit == r(2, 3) and q_R == r(4, 3) and alpha_W == r(3, 8)
    assert np.isclose(float(alpha_W), alpha_eff)
    print(
        f"[5] boundary route: C_R/C_T = 1/10 -> alpha = {alpha_unit} at unit R-charge;"
        f" W R-charge {q_R} -> alpha = {alpha_W}, equal to [4]"
    )
    print("all asserted")


if __name__ == "__main__":
    main()
