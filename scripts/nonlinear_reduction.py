"""Symbolic verification of the reduced (Ginzburg-Landau-like) free energy.

For static, z- and t-independent configurations with A^a_t = A^a_z = A^a_u = 0,
the SU(2) Yang-Mills free energy density on the black brane reduces to

  F = int du d^2x  L,
  L = (1/2u) [ |F^W_xy|^2 + ( B + b + Im(Wbar_x W_y) )^2 ]
      + (f/2u) [ |W_x'|^2 + |W_y'|^2 + a_x'^2 + a_y'^2 ],

with W_mu = A^1_mu + i A^2_mu,  a_i the A^3 fluctuation (A^3_y = xB + a_y,
A^3_x = a_x),  b = d_x a_y - d_y a_x,  ' = d/du, and
F^W_xy = D_x W_y - D_y W_x,  D_i = d_i + i A^3_i  (full A^3, fluctuation incl.).

Checks:
[1] L equals (1/4) sqrt(-g) F^a_{mu nu} F^{a mu nu} EXACTLY (all orders) for
    generic complex W_x, W_y and real a_x, a_y.
[2] F^W_xy = 0 (at a = 0) for the polarised lowest-Landau-level ansatz
    W_x = w0(u) psi(x,y),  W_y = -i W_x,  psi any LLL superposition.
[3] Landau expansion W -> rho W, a -> rho^2 a:  the O(rho^2) and O(rho^4)
    terms of L match the quadratic and quartic functionals of paper
    app. E.1, the O(rho^2) piece linear in a is a total derivative, the
    O(rho^3, rho^5) terms vanish, and corrections start at O(rho^6).
"""

import sympy as sp

u, x, y = sp.symbols("u x y", real=True, positive=True)
B, rho = sp.symbols("B rho", positive=True)
I = sp.I
f = sp.Function("f")(u)          # generic blackening function

# ---------------------------------------------------------------------------
# [1] Exact reduction identity, generic fields
# ---------------------------------------------------------------------------
Wx = sp.Function("W_x")(u, x, y)
Wy = sp.Function("W_y")(u, x, y)
Wxc = sp.Function("Wc_x")(u, x, y)   # formal conjugates, independent
Wyc = sp.Function("Wc_y")(u, x, y)
ax = sp.Function("a_x")(u, x, y)
ay = sp.Function("a_y")(u, x, y)

# real colour components
A = {}   # A[(colour, coord)] ; coords 0=x, 1=y, 2=u
A[(0, 0)] = (Wx + Wxc) / 2
A[(0, 1)] = (Wy + Wyc) / 2
A[(1, 0)] = (Wx - Wxc) / (2 * I)
A[(1, 1)] = (Wy - Wyc) / (2 * I)
A[(2, 0)] = ax
A[(2, 1)] = B * x + ay
for a_ in range(3):
    A[(a_, 2)] = sp.S.Zero       # radial gauge A_u = 0

coords = [x, y, u]
LC = {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1,
      (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}


def F(a_, m, n):
    expr = sp.diff(A[(a_, n)], coords[m]) - sp.diff(A[(a_, m)], coords[n])
    for b_ in range(3):
        for c_ in range(3):
            e = LC.get((a_, b_, c_), 0)
            if e:
                expr += e * A[(b_, m)] * A[(c_, n)]
    return expr


# L_exact = (1/4) sqrt(-g) F^a_{mu nu} F^{a mu nu} for static planar configs
L_exact = sp.S.Zero
for a_ in range(3):
    L_exact += (F(a_, 0, 1)**2 + f * (F(a_, 0, 2)**2 + F(a_, 1, 2)**2))
L_exact = L_exact / (2 * u)

b_ind = sp.diff(ay, x) - sp.diff(ax, y)
A3x, A3y = ax, B * x + ay
FWxy = (sp.diff(Wy, x) + I * A3x * Wy) - (sp.diff(Wx, y) + I * A3y * Wx)
FWxyc = (sp.diff(Wyc, x) - I * A3x * Wyc) - (sp.diff(Wxc, y) - I * A3y * Wxc)
ImWW = (Wxc * Wy - Wx * Wyc) / (2 * I)

L_red = ((FWxy * FWxyc + (B + b_ind + ImWW)**2) / (2 * u)
         + f / (2 * u) * (sp.diff(Wx, u) * sp.diff(Wxc, u)
                          + sp.diff(Wy, u) * sp.diff(Wyc, u)
                          + sp.diff(ax, u)**2 + sp.diff(ay, u)**2))

diff1 = sp.simplify(sp.expand(L_exact - L_red))
print(f"[1] L_exact == L_reduced for generic fields: {diff1 == 0}")
assert diff1 == 0

# ---------------------------------------------------------------------------
# [2] F^W_xy = 0 on the polarised LLL (a = 0); a two-mode superposition
# ---------------------------------------------------------------------------
w0 = sp.Function("w_0", real=True)(u)
p1, p2, c1, c2 = sp.symbols("p_1 p_2 c_1 c_2", real=True)
psi_p = lambda p: sp.exp(I * p * y - B * (x + p / B)**2 / 2)
psi = c1 * psi_p(p1) + c2 * psi_p(p2)
psic = psi.conjugate()

WxL, WyL = w0 * psi, -I * w0 * psi
FWxy_L = (sp.diff(WyL, x)) - (sp.diff(WxL, y) + I * B * x * WxL)
print(f"[2] F^W_xy = 0 on LLL with W_y = -i W_x: {sp.simplify(FWxy_L) == 0}")
assert sp.simplify(FWxy_L) == 0

# ---------------------------------------------------------------------------
# [3] Landau expansion in rho
# ---------------------------------------------------------------------------
subs_rho = {Wx: rho * WxL, Wy: rho * WyL,
            Wxc: rho * WxL.conjugate(), Wyc: rho * WyL.conjugate(),
            ax: rho**2 * ax, ay: rho**2 * ay}
# build L with the scaled fields directly (substitute before differentiating)
A2 = dict(A)
A2[(0, 0)] = rho * (WxL + sp.conjugate(WxL)) / 2
A2[(0, 1)] = rho * (WyL + sp.conjugate(WyL)) / 2
A2[(1, 0)] = rho * (WxL - sp.conjugate(WxL)) / (2 * I)
A2[(1, 1)] = rho * (WyL - sp.conjugate(WyL)) / (2 * I)
A2[(2, 0)] = rho**2 * ax
A2[(2, 1)] = B * x + rho**2 * ay
A_save, A = A, A2

L_rho = sp.S.Zero
for a_ in range(3):
    L_rho += (F(a_, 0, 1)**2 + f * (F(a_, 0, 2)**2 + F(a_, 1, 2)**2))
L_rho = sp.expand(L_rho / (2 * u))
A = A_save

# treat conjugate() of the symbols as themselves (all parameters real)
realsubs = {sp.conjugate(s): s for s in (c1, c2, p1, p2)}
L_rho = L_rho.subs(realsubs)
poly = sp.Poly(L_rho, rho)

abspsi2 = sp.expand(psi * psic)
w0p = sp.diff(w0, u)

orders = {n: sp.expand(poly.coeff_monomial(rho**n)) for n in range(7)}

# O(rho^0): pure background
ok0 = sp.simplify(orders[0] - B**2 / (2 * u)) == 0
# O(rho^1): absent
ok1 = sp.simplify(orders[1]) == 0
# O(rho^2): quadratic functional + total-derivative piece B*b/u
quad_expect = (f / u) * w0p**2 * abspsi2 - (B / u) * w0**2 * abspsi2 + B * b_ind / u
ok2 = sp.simplify(orders[2] - sp.expand(quad_expect)) == 0
# O(rho^3): absent
ok3 = sp.simplify(orders[3]) == 0
# O(rho^4): (b - w0^2 |psi|^2)^2/2u + radial gradient of a
quart_expect = ((b_ind - w0**2 * abspsi2)**2 / (2 * u)
                + f / (2 * u) * (sp.diff(ax, u)**2 + sp.diff(ay, u)**2))
ok4 = sp.simplify(sp.expand(orders[4] - sp.expand(quart_expect))) == 0
# O(rho^5): absent
ok5 = sp.simplify(orders[5]) == 0
# O(rho^6): the |a x W|^2 remainder, nonzero but subleading
ok6 = sp.simplify(orders[6]) != 0

print(f"[3] O(rho^0) = B^2/2u (background):                    {ok0}")
print(f"    O(rho^1) absent:                                   {ok1}")
print(f"    O(rho^2) = (f w0'^2 - B w0^2)|psi|^2/u + d(B a/u): {ok2}")
print(f"    O(rho^3) absent:                                   {ok3}")
print(f"    O(rho^4) = (b - w0^2|psi|^2)^2/2u + f a'^2/2u:     {ok4}")
print(f"    O(rho^5) absent:                                   {ok5}")
print(f"    corrections start at O(rho^6):                     {ok6}")
assert all([ok0, ok1, ok2, ok3, ok4, ok5, ok6])
print("\nAll reduction checks passed.")
