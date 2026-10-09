"""Symbolic verification of the SU(2) Yang-Mills equations on AdS5-Schwarzschild.

Conventions
-----------
* Metric (L = 1, boundary at u -> 0, horizon at u = uH):
      ds^2 = (1/u^2) ( -f dt^2 + dx^2 + dy^2 + dz^2 ) + du^2/(u^2 f),
      f(u) = 1 - u^4/uH^4.
* SU(2) field strength (gauge coupling absorbed into A):
      F^a_{mu nu} = d_mu A^a_nu - d_nu A^a_mu + eps^{abc} A^b_mu A^c_nu.
* Yang-Mills equations:
      (1/sqrt(-g)) d_mu ( sqrt(-g) F^{a mu nu} ) + eps^{abc} A^b_mu F^{c mu nu} = 0.

Checks performed
----------------
1. The background A^3_y = x B solves the full nonlinear equations exactly.
2. Linear fluctuations
       a^1_x = eps * w(u) h(x),   a^2_y = eps * s * w(u) h(x),   s = +/- 1,
   i.e. W_x = w h, W_y = i s w h for W_mu = A^1_mu + i A^2_mu.
   We extract the O(eps) equations, find the consistent polarisation s and the
   lowest-Landau-level profile h(x) = exp(-B x^2 / 2), and print the resulting
   radial ODE for w(u).
"""

import sympy as sp

t, x, y, z, u = coords = sp.symbols("t x y z u", real=True)
uH, B, eps, s = sp.symbols("u_H B epsilon s", positive=True, real=True)
# s will be substituted with +1 / -1 explicitly below
w = sp.Function("w")
h = sp.Function("h")

f = 1 - u**4 / uH**4

g = sp.diag(-f / u**2, 1 / u**2, 1 / u**2, 1 / u**2, 1 / (u**2 * f))
ginv = g.inv()
sqrtg = sp.sqrt(-g.det())
sqrtg = sp.simplify(sqrtg)  # = 1/u^5

N = 5
LC = {}  # SU(2) structure constants eps^{abc}, a,b,c in {0,1,2} ~ colours 1,2,3
for perm, sign in (((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1),
                   ((0, 2, 1), -1), ((2, 1, 0), -1), ((1, 0, 2), -1)):
    LC[perm] = sign


def levi(a, b, c):
    return LC.get((a, b, c), 0)


def field_strength(A):
    """F^a_{mu nu} for a 3 x 5 array of sympy expressions A[a][mu]."""
    F = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(3)]
    for a in range(3):
        for mu in range(N):
            for nu in range(N):
                # F^a_{mu nu} = d_mu A_nu - d_nu A_mu
                expr = sp.diff(A[a][nu], coords[mu]) - sp.diff(A[a][mu], coords[nu])
                for b in range(3):
                    for c in range(3):
                        e = levi(a, b, c)
                        if e:
                            expr += e * A[b][mu] * A[c][nu]
                F[a][mu][nu] = expr
    return F


def raise_indices(F):
    """F^{a mu nu} = g^{mu rho} g^{nu sig} F^a_{rho sig}."""
    Fup = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(3)]
    for a in range(3):
        for mu in range(N):
            for nu in range(N):
                expr = sp.S.Zero
                for rho in range(N):
                    for sig in range(N):
                        if F[a][rho][sig] != 0:
                            expr += ginv[mu, rho] * ginv[nu, sig] * F[a][rho][sig]
                Fup[a][mu][nu] = sp.simplify(expr)
    return Fup


def ym_equations(A):
    """E^{a nu} = (1/sqrt(-g)) d_mu( sqrt(-g) F^{a mu nu} ) + eps^{abc} A^b_mu F^{c mu nu}."""
    F = field_strength(A)
    Fup = raise_indices(F)
    E = [[sp.S.Zero] * N for _ in range(3)]
    for a in range(3):
        for nu in range(N):
            expr = sp.S.Zero
            for mu in range(N):
                expr += sp.diff(sqrtg * Fup[a][mu][nu], coords[mu]) / sqrtg
                for b in range(3):
                    for c in range(3):
                        e = levi(a, b, c)
                        if e:
                            expr += e * A[b][mu] * Fup[c][mu][nu]
            E[a][nu] = sp.simplify(sp.expand(expr))
    return E


# ----------------------------------------------------------------------------
# 1. Background check: A^3_y = x B
# ----------------------------------------------------------------------------
A_bg = [[sp.S.Zero] * N for _ in range(3)]
A_bg[2][2] = x * B  # colour 3, coordinate y

E_bg = ym_equations(A_bg)
bg_ok = all(E_bg[a][nu] == 0 for a in range(3) for nu in range(N))
print(f"[1] Background A^3_y = xB solves the full YM equations: {bg_ok}")
assert bg_ok

# ----------------------------------------------------------------------------
# 2. Linear fluctuation with generic profile h(x) and polarisation s
#    W_x = w(u) h(x),  W_y = i s w(u) h(x)
#    => a^1_x = w h, a^2_x = 0, a^1_y = 0, a^2_y = s w h
# ----------------------------------------------------------------------------
A = [[sp.S.Zero] * N for _ in range(3)]
A[2][2] = x * B
A[0][1] = eps * w(u) * h(x)        # a^1_x
A[1][2] = eps * s * w(u) * h(x)    # a^2_y

E = ym_equations(A)

print("\n[2] O(eps) equations of motion (generic h(x), generic s):")
lin = {}
for a in range(3):
    for nu in range(N):
        expr = sp.expand(sp.diff(E[a][nu], eps).subs(eps, 0))
        expr = sp.simplify(expr)
        if expr != 0:
            lin[(a, nu)] = expr
            name = f"colour {a+1}, nu = {['t','x','y','z','u'][nu]}"
            print(f"  E^({name}):")
            sp.pprint(expr)
            print()

# Each nonzero linear equation should be of the form
#   u^2 * [ radial-operator(w) * h  +  w * landau-operator(h) ] = 0  (up to factors)
# Identify the Landau operator by collecting h, h', h''.

# ----------------------------------------------------------------------------
# 3. Impose the lowest-Landau-level profile h = exp(-B x^2/2) and find which
#    polarisation s makes everything collapse to a single radial ODE.
# ----------------------------------------------------------------------------
hLLL = sp.exp(-B * x**2 / 2)
print("[3] Substituting h(x) = exp(-B x^2/2):")
for sval in (+1, -1):
    print(f"\n  --- polarisation s = {sval:+d} ---")
    eqs = set()
    for (a, nu), expr in lin.items():
        e2 = expr.subs(s, sval)
        e2 = e2.subs({h(x): hLLL,
                      sp.Derivative(h(x), x): sp.diff(hLLL, x),
                      sp.Derivative(h(x), x, 2): sp.diff(hLLL, x, 2)})
        e2 = sp.simplify(e2 / hLLL)
        name = f"colour {a+1}, nu = {['t','x','y','z','u'][nu]}"
        if e2 == 0:
            print(f"    E^({name}) = 0 identically")
        else:
            # strip overall u- and sign factors to compare equations
            e2 = sp.simplify(sp.expand(e2 * u**2))
            print(f"    E^({name}) * u^2 :")
            sp.pprint(e2)
            eqs.add(sp.simplify(e2))
    # check all surviving equations are proportional
    eqlist = [e for e in eqs if e != 0]
    if eqlist:
        base = eqlist[0]
        prop = all(sp.simplify(sp.expand(e / base)).is_constant() for e in eqlist)
        print(f"    surviving equations mutually proportional: {prop}")

# ----------------------------------------------------------------------------
# 4. Print the final radial ODE in a clean form for the consistent choice
# ----------------------------------------------------------------------------
print("\n[4] Final radial ODE (consistent LLL ansatz, s = -1):")
expr = lin[(0, 1)].subs(s, -1)
expr = expr.subs({h(x): hLLL,
                  sp.Derivative(h(x), x): sp.diff(hLLL, x),
                  sp.Derivative(h(x), x, 2): sp.diff(hLLL, x, 2)})
expr = sp.simplify(expr / hLLL)
ode = sp.collect(sp.expand(expr * u**2 / f), w(u))
print("  0 =")
sp.pprint(sp.nsimplify(sp.simplify(expr * u**2)))

# compare with the expected form  f w'' + (f' - f/u) w' + B w = 0
# (the raw equation carries an irrelevant overall power of u)
expected = f * sp.diff(w(u), u, 2) + (sp.diff(f, u) - f / u) * sp.diff(w(u), u) + B * w(u)
ratio = sp.cancel(sp.together(expr * u**2) / sp.together(expected))
is_prop = w(u) not in ratio.atoms(sp.Function) and not ratio.has(sp.Derivative)
print(f"\n  equation = ({ratio}) * [ f w'' + (f' - f/u) w' + B w ]")
print(f"  matches  f w'' + (f' - f/u) w' + B w = 0  up to overall factor: {is_prop}")
assert is_prop
