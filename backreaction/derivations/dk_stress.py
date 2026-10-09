"""O(rho^2) condensate stress tensor and colour-3 current on the GENERAL diagonal
magnetic brane, per reciprocal-lattice harmonic.

Generalises the AdS5-Schwarzschild (probe-limit) condensate-stress derivation
to the D'Hoker-Kraus brane

    ds^2 = -U dt^2 + dr^2/U + e^{2V}(dx^2+dy^2) + e^{2W} dz^2,   A^3_y = B x,

with the frozen probe condensate  W_x = rho w0(r) psi,  W_y = -i W_x,  psi in the
LLL, and the induced abelian field  A^3 = (xB + rho^2 a_y)dy + rho^2 a_x dx.

The metric is kept as four generic functions of a single radial coordinate so
that both the (U,V,W) brane specialisation and the flat u-coordinate limit
(the AdS5-Schwarzschild source table, hard-coded in check [6]) fall out; NO
background Einstein equation is used here (the
stress/current are pure Yang-Mills tensor algebra; only the matter w0 and bhat
ODEs are imposed for conservation).

Conventions (as in the probe-limit stress derivation):
    T_{MN} = F^a_{M lam} F^a_{N sig} g^{lam sig} - (1/4) g_{MN} F^2 ,
    E^{a nu} = (1/sqrt(-g)) d_M(sqrt(-g) F^{a M nu}) + eps^{abc} A^b_M F^{c M nu}
      (the Yang-Mills equation expression; E^a = 0 on shell).
The colour-3 current sourcing the abelian field is J^nu = -E^3^nu|_{a=0, O(rho^2)}
(the condensate contribution), so the induced-field equation is
    (abelian operator on a)^nu = J^nu.

Checks (all asserted)
---------------------
[1] No O(rho), O(rho^3) terms; background T^(0) = magnetic-brane stress.
[2] O(rho^2) closes on the gauge invariants n = |psi|^2, b = d_x a_y - d_y a_x
    alone -- all explicit x cancel -- with the closed form generalising the AdS5-Schwarzschild one.
[3] nabla^M T_{MN} = F^a_{N lam} E^{a lam} identically at O(rho^2) (generic).
[4] On shell: E^{1,2}|_O(rho) = 0 iff LLL + brane zero-mode ODE ; the
    colour-3 O(rho^2) equation per harmonic = the brane induced-field ODE of
    derivations/dk_quartic.py, with the longitudinal a_x unsourced and the Gauss (r)
    constraint identically satisfied.
[5] Per-harmonic source table S_{MN}(r) e^{iGx} and current J_nu(r) e^{iGx}
    (G || x), conserved given only the w0 and bhat ODEs.
[6] Flat limit (g_tt=-f/u^2 etc.) reproduces the AdS5-Schwarzschild source table exactly.

Generates systems/dk_stress.py (the per-harmonic S_{MN}(r) and J_nu(r) as
functions of (r, U, U', V, V', W, W', B, G, w0, w0', bhat, bhat')).

Run:  uv run python -m backreaction.derivations.dk_stress   (~3 min)
"""

import sys
import time

import sympy as sp

sys.setrecursionlimit(100000)
T0_START = time.time()


def log(msg):
    print(f"[{time.time() - T0_START:7.1f}s] {msg}", flush=True)


def is_zero(e):
    e = sp.expand(e)
    if e == 0:
        return True
    return sp.simplify(sp.cancel(sp.together(e))) == 0


# ---------------------------------------------------------------------------
# General diagonal metric; coordinates (t, x, y, z, r)
# ---------------------------------------------------------------------------
t, x, y, z = sp.symbols("t x y z", real=True)
r = sp.Symbol("r", positive=True)
coords = [t, x, y, z, r]
N = 5
B, rho = sp.symbols("B rho", positive=True)
Gx, Gy = sp.symbols("G_x G_y", real=True)
G2 = Gx**2 + Gy**2

gtt = sp.Function("g_tt")(r)
grr = sp.Function("g_rr")(r)
gpp = sp.Function("g_pp")(r)
gzz = sp.Function("g_zz")(r)
att, brr, cpp, dzz = sp.symbols("att brr cpp dzz", positive=True)
POS = {gtt: -att, grr: brr, gpp: cpp, gzz: dzz}

gdn = sp.diag(gtt, gpp, gpp, gzz, grr)  # order (t, x, y, z, r)
gup = sp.diag(1 / gtt, 1 / gpp, 1 / gpp, 1 / gzz, 1 / grr)
sqrtg = sp.sqrt(-gtt * gpp * gpp * gzz * grr)

# (U,V,W) brane specialisation (resolves sqrt(-g) cleanly for on-shell checks)
Uf, Vf, Wf = sp.Function("U")(r), sp.Function("V")(r), sp.Function("W")(r)
brane = {gtt: -Uf, grr: 1 / Uf, gpp: sp.exp(2 * Vf), gzz: sp.exp(2 * Wf)}
p1s = sp.exp(Wf - 2 * Vf)  # brane norm density
p2s = Uf * sp.exp(Wf)  # brane SL weight


def to_brane(e):
    """Substitute the (U,V,W) brane metric; resolve sqrt (U>0, exp>0)."""
    e = sp.sympify(e).subs(brane).doit()
    e = sp.powsimp(e, force=True)
    return sp.expand(sp.radsimp(e))


w = sp.Function("w0")
P = sp.Function("P")  # Re psi
Q = sp.Function("Q")  # Im psi
aX = sp.Function("a_x")  # A^3_x / rho^2
aY = sp.Function("a_y")  # (A^3_y - xB) / rho^2

LC = {}


def main():
    for perm, sign in (
        ((0, 1, 2), 1),
        ((1, 2, 0), 1),
        ((2, 0, 1), 1),
        ((0, 2, 1), -1),
        ((2, 1, 0), -1),
        ((1, 0, 2), -1),
    ):
        LC[perm] = sign

    def levi(a, b, c):
        return LC.get((a, b, c), 0)

    def field_strength(A):
        F = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(3)]
        for a in range(3):
            for m in range(N):
                for n in range(N):
                    e = sp.diff(A[a][n], coords[m]) - sp.diff(A[a][m], coords[n])
                    for b in range(3):
                        for c in range(3):
                            s = levi(a, b, c)
                            if s:
                                e += s * A[b][m] * A[c][n]
                    F[a][m][n] = sp.expand(e)
        return F

    def raise_indices(F):
        Fup = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(3)]
        for a in range(3):
            for m in range(N):
                for n in range(N):
                    if F[a][m][n] != 0:
                        Fup[a][m][n] = sp.expand(gup[m, m] * gup[n, n] * F[a][m][n])
        return Fup

    def ym_equations(A, F, Fup):
        E = [[sp.S.Zero] * N for _ in range(3)]
        for a in range(3):
            for nu in range(N):
                e = sp.S.Zero
                for m in range(N):
                    if Fup[a][m][nu] != 0:
                        e += sp.diff(sqrtg * Fup[a][m][nu], coords[m]) / sqrtg
                    for b in range(3):
                        for c in range(3):
                            s = levi(a, b, c)
                            if s and A[b][m] != 0 and Fup[c][m][nu] != 0:
                                e += s * A[b][m] * Fup[c][m][nu]
                E[a][nu] = sp.expand(e)
        return E

    def stress(F, Fup):
        F2 = sp.S.Zero
        for a in range(3):
            for m in range(N):
                for n in range(N):
                    if F[a][m][n] != 0:
                        F2 += F[a][m][n] * Fup[a][m][n]
        F2 = sp.expand(F2)
        T = [[sp.S.Zero] * N for _ in range(N)]
        for m in range(N):
            for n in range(m, N):
                e = sp.S.Zero
                for a in range(3):
                    for lam in range(N):
                        if F[a][m][lam] != 0 and F[a][n][lam] != 0:
                            e += F[a][m][lam] * gup[lam, lam] * F[a][n][lam]
                e -= gdn[m, n] * F2 / 4
                e = sp.expand(e)
                T[m][n] = e
                T[n][m] = e
        return T, F2

    def christoffel(gd, gu):
        Gam = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(N)]
        for a in range(N):
            for m in range(N):
                for n in range(m, N):
                    e = sp.S.Zero
                    for s_ in range(N):
                        if gu[a, s_] != 0:
                            e += gu[a, s_] * (
                                sp.diff(gd[s_, m], coords[n])
                                + sp.diff(gd[s_, n], coords[m])
                                - sp.diff(gd[m, n], coords[s_])
                            )
                    e = sp.cancel(e / 2)
                    Gam[a][m][n] = e
                    Gam[a][n][m] = e
        return Gam

    GAM = christoffel(gdn, gup)

    def divergence(T, Gam=None, gu=None):
        Gam = GAM if Gam is None else Gam
        gu = gup if gu is None else gu
        out = [sp.S.Zero] * N
        for nu in range(N):
            e = sp.S.Zero
            for rho_ in range(N):
                mu = rho_
                cov = sp.diff(T[mu][nu], coords[rho_])
                for s_ in range(N):
                    if Gam[s_][rho_][mu] != 0:
                        cov -= Gam[s_][rho_][mu] * T[s_][nu]
                    if Gam[s_][rho_][nu] != 0:
                        cov -= Gam[s_][rho_][nu] * T[mu][s_]
                e += gu[rho_, mu] * cov
            out[nu] = sp.expand(e)
        return out

    def lll_reduce(expr):
        """Eliminate x-derivatives of P, Q via (d_x - i d_y + Bx)(P+iQ) = 0."""
        Pf, Qf = P(x, y), Q(x, y)
        expr = sp.expand(sp.sympify(expr).doit())
        while True:
            cands = [
                d
                for d in expr.atoms(sp.Derivative)
                if d.expr in (Pf, Qf) and x in d.variables
            ]
            if not cands:
                return sp.expand(expr)
            target = sorted(cands, key=lambda d: (len(d.variables), str(d)))[-1]
            vars_ = list(target.variables)
            vars_.remove(x)
            first = (
                (-sp.Derivative(Qf, y) - B * x * Pf)
                if target.expr == Pf
                else (sp.Derivative(Pf, y) - B * x * Qf)
            )
            newd = sp.diff(first, *vars_) if vars_ else first
            expr = sp.expand(expr.subs(target, newd).doit())

    # ---------------------------------------------------------------------------
    # ansatz (coords order t, x, y, z, r)
    # ---------------------------------------------------------------------------
    A = [[sp.S.Zero] * N for _ in range(3)]
    A[0][1] = rho * w(r) * P(x, y)  # A^1_x
    A[1][1] = rho * w(r) * Q(x, y)  # A^2_x
    A[0][2] = rho * w(r) * Q(x, y)  # A^1_y
    A[1][2] = -rho * w(r) * P(x, y)  # A^2_y
    A[2][1] = rho**2 * aX(x, y, r)  # A^3_x
    A[2][2] = x * B + rho**2 * aY(x, y, r)  # A^3_y

    log("building field strength and stress tensor ...")
    F = field_strength(A)
    Fup = raise_indices(F)
    T, F2 = stress(F, Fup)

    # ---------------------------------------------------------------------------
    # [1] order counting + background
    # ---------------------------------------------------------------------------
    assert all(
        T[m][n].coeff(rho, k) == 0 for m in range(N) for n in range(N) for k in (1, 3)
    ), "odd orders"
    log("[1] no O(rho), O(rho^3) terms in T")
    # background magnetic-brane stress (colour 3, F_xy = B): T0_MN
    T0 = [[sp.expand(T[m][n].coeff(rho, 0)) for n in range(N)] for m in range(N)]
    # closed form: T0 = B^2/(2) [ diag over the metric ] -- the F_xy^2 stress.
    # F^3_xy = B, F^{3xy} = B g^xx g^yy; T0_MN = B^2 g^pp (delta-structure) - (1/4) g F^2
    F2_0 = sp.expand(F2.coeff(rho, 0))
    T0_claim = [[sp.S.Zero] * N for _ in range(N)]
    for m in range(N):
        T0_claim[m][m] = sp.expand(sp.S(0))
    # explicit: F2_0 = 2 B^2 g^{pp}g^{pp}; T0_tt = -gdn_tt F2_0/4 etc; T0_xx = B^2 g^pp g^pp gdn_xx - gdn_xx F2_0/4
    for m in range(N):
        e = -gdn[m, m] * F2_0 / 4
        if m in (1, 2):  # x, y carry F_xy^2
            e += B**2 * gup[1, 1] * gup[2, 2] * gdn[m, m]
        T0_claim[m][m] = sp.expand(e)
    assert all(
        is_zero(T0[m][n] - (T0_claim[m][n] if m == n else 0))
        for m in range(N)
        for n in range(N)
    ), "background stress form"
    log("[1] background T^(0) = magnetic-brane stress (F_xy = B), closed form")

    # ---------------------------------------------------------------------------
    # [2] O(rho^2) closes on (n, b) after LLL reduction; harmonic transcription
    # ---------------------------------------------------------------------------
    wp = sp.diff(w(r), r)

    Pf, Qf = P(x, y), Q(x, y)
    Py, Qy = sp.Derivative(Pf, y), sp.Derivative(Qf, y)
    Eh = sp.exp(sp.I * (Gx * x + Gy * y))
    bh = sp.Function("bhat")
    ell = sp.Function("ell")
    axh = sp.I * Gy * bh(r) / G2 + Gx * ell(r)
    ayh = -sp.I * Gx * bh(r) / G2 + Gy * ell(r)
    assert sp.simplify(sp.I * (Gx * ayh - Gy * axh) - bh(r)) == 0
    sP, sQ, sPy, sQy = sp.symbols("sP sQ sPy sQy")

    def harm_reduce(expr, want_transverse=True):
        """LLL-reduce a psi-bilinear + a-field expression, verify it organises into
        the gauge invariants (n, grad n, a-fields), and return the per-harmonic
        r-function with e^{i G.r} stripped (a_i replaced by the transverse+long split
        if want_transverse else kept symbolic).  Mirrors the probe-limit stress
        derivation."""
        e = lll_reduce(sp.expand(expr))
        if e == 0:
            return sp.S.Zero, True
        bad = [
            d
            for d in e.atoms(sp.Derivative)
            if d.expr in (Pf, Qf) and len(d.variables) > 1
        ]
        assert not bad, f"higher psi-derivatives: {bad}"
        es = e.subs({Py: sPy, Qy: sQy}).subs({Pf: sP, Qf: sQ})
        pol = sp.Poly(es, sP, sQ, sPy, sQy)
        rest = sp.S.Zero
        cP2 = cQ2 = cPPy = cQQy = cQPy = cPQy = sp.S.Zero
        ok = True
        for mono, coeff_ in pol.terms():
            key = tuple(mono)
            if key == (0, 0, 0, 0):
                rest += coeff_
            elif key == (2, 0, 0, 0):
                cP2 += coeff_
            elif key == (0, 2, 0, 0):
                cQ2 += coeff_
            elif key == (1, 0, 1, 0):
                cPPy += coeff_
            elif key == (0, 1, 0, 1):
                cQQy += coeff_
            elif key == (0, 1, 1, 0):
                cQPy += coeff_
            elif key == (1, 0, 0, 1):
                cPQy += coeff_
            else:
                ok = False
        if (
            not is_zero(cP2 - cQ2)
            or not is_zero(cPPy - cQQy)
            or not is_zero(cQPy + cPQy)
        ):
            ok = False
        e_h = (
            (cP2 + B * x * cQPy) * Eh
            + cPPy * (sp.I * Gy / 2) * Eh
            + cQPy * (sp.I * Gx / 2) * Eh
            + rest
        )
        e_h = sp.expand(e_h.subs({aX(x, y, r): axh * Eh, aY(x, y, r): ayh * Eh}).doit())
        e_h = sp.expand(sp.cancel(e_h / Eh))
        if e_h.has(x) or e_h.has(y):
            ok = False
        return e_h, ok

    log("[2] LLL reduction + harmonic transcription of T2 ...")
    T2 = [[T[m][n].coeff(rho, 2) for n in range(N)] for m in range(N)]
    Sharm = [[sp.S.Zero] * N for _ in range(N)]  # brane per-harmonic S_MN(r)
    Sharm_gen = [
        [sp.S.Zero] * N for _ in range(N)
    ]  # generic-metric version (flat check)
    ok2 = True
    for m in range(N):
        for n in range(m, N):
            e_h, ok = harm_reduce(T2[m][n])  # closure (generic metric)
            if not ok:
                ok2 = False
                print(f"    T2[{m}][{n}] does not close on (n, b)")
            Sharm_gen[m][n] = Sharm_gen[n][m] = e_h
            e_hb, _ = harm_reduce(to_brane(T2[m][n]))  # brane table
            Sharm[m][n] = Sharm[n][m] = e_hb
    assert ok2, "T2 closure on (n, b)"
    log("[2] every T2 component closes on the gauge invariants (n, b); x cancels")

    # ---------------------------------------------------------------------------
    # [3] divergence identity at O(rho^2), generic fields
    # ---------------------------------------------------------------------------
    log("[3] YM equations + divergence identity ...")
    E = ym_equations(A, F, Fup)
    divT = divergence(T)
    ok3 = True
    for nu in range(N):
        rhs = sp.S.Zero
        for a in range(3):
            for lam in range(N):
                if F[a][nu][lam] != 0 and E[a][lam] != 0:
                    rhs += F[a][nu][lam] * E[a][lam]
        resid = sp.expand(divT[nu].coeff(rho, 2) - sp.expand(rhs).coeff(rho, 2))
        if not is_zero(resid):
            ok3 = False
            print(f"    identity fails nu={coords[nu]}")
    assert ok3, "divergence identity"
    log("[3] nabla^M T_MN = F^a_N_lam E^a^lam at O(rho^2), generic fields")

    # ---------------------------------------------------------------------------
    # [4] on-shell (brane form so sqrt(-g) resolves): E^{1,2}|_O(rho) via
    #     zero-mode ODE; E^3|_O(rho^2) = brane induced-field ODE
    # ---------------------------------------------------------------------------
    # zero-mode ODE operator (p2 w0')' + B p1 w0 (brane, sqrt-free)
    wpp = sp.Derivative(w(r), (r, 2))
    op_sl = sp.expand(sp.diff(p2s * wp, r).doit() + B * p1s * w(r))
    w0pp_expr = sp.solve(op_sl, wpp)[0]
    wpp_rule = {wpp: w0pp_expr}

    ok4a = True
    for a in (0, 1):
        for nu in range(N):
            e1 = to_brane(E[a][nu].coeff(rho, 1))
            e1 = lll_reduce(sp.expand(e1.subs(wpp_rule).doit()))
            if not is_zero(e1):
                ok4a = False
                print(f"    O(rho) YM survives colour {a + 1}, nu={coords[nu]}")
    assert ok4a, "E^{1,2} O(rho)"
    log("[4] E^(1,2)|_O(rho) = 0 given LLL + brane zero-mode ODE")

    # E^3 at O(rho^2): harmonic decomposition on the brane
    log("[4] reducing E^3 at O(rho^2) ...")
    harm = {}
    ok_bilin = True
    for nu in range(N):
        e_h, ok = harm_reduce(to_brane(E[2][nu].coeff(rho, 2)))
        if not ok:
            ok_bilin = False
            print(f"    E^3 nu={nu} does not close on (n, grad n)")
        harm[nu] = e_h
    assert ok_bilin, "E^3 bilinears"
    log("[4] E^3|_O(rho^2) bilinears reduce to (n, grad n); harmonics well-defined")

    assert harm[0] == 0 and harm[3] == 0, "E^3 t,z nonzero"
    # Gauss (r) constraint vanishes for transverse a_i (ell -> 0)
    assert is_zero(harm[4].subs(ell(r), 0)), "Gauss (r) constraint"
    # longitudinal unsourced
    long_proj = sp.expand(Gx * harm[1] + Gy * harm[2])
    assert is_zero(
        long_proj.subs(ell(r), sp.S.Zero)
        .subs(sp.Derivative(ell(r), r), 0)
        .subs(sp.Derivative(ell(r), (r, 2)), 0)
    ), "longitudinal sourced"
    log("[4] Gauss (r) constraint identically satisfied; longitudinal a_x unsourced")

    # transverse projection = the dk_quartic brane induced-field ODE:
    #   -(p2 bhat')' + G^2 p1 (bhat - w0^2) = 0
    trans_proj = sp.expand((Gy * harm[1] - Gx * harm[2]).subs(ell(r), 0))
    target = -sp.diff(p2s * sp.Derivative(bh(r), r), r).doit() + G2 * p1s * (
        bh(r) - w(r) ** 2
    )
    ratio = sp.simplify(sp.cancel(trans_proj / sp.expand(target)))
    assert not ratio.has(bh(r)) and not ratio.has(sp.Derivative) and ratio != 0, (
        "transverse != brane induced ODE"
    )
    log(
        f"[4] transverse E^3 = ({ratio}) x [brane induced-field ODE] -- as in dk_quartic"
    )

    # ---------------------------------------------------------------------------
    # [5] per-harmonic source table S and current J (G || x), conservation
    # ---------------------------------------------------------------------------
    log("[5] per-harmonic source table + current ...")
    Gs = sp.Symbol("G", positive=True)
    subGx = {Gx: Gs, Gy: 0}
    # S_MN = the per-harmonic transcription Sharm with G || x and transverse a (ell->0)
    S = [[sp.S.Zero] * N for _ in range(N)]
    for m in range(N):
        for n in range(N):
            e = Sharm[m][n].subs(ell(r), 0).doit()
            S[m][n] = sp.expand(sp.cancel(e.subs(subGx)))

    # current J^nu = -E^3^nu at O(rho^2), a=0 (condensate source); lower it: J_nu.
    # From harm[nu] = (abelian op on a)_nu - J_nu (harm=0 on shell), the source is
    # the a-independent part: J-tilde_nu = -harm[nu]|_{a=0}.  The physical transverse
    # current is J = G^2 p1 w0^2 (the RHS of the induced ODE).
    Jsrc = sp.expand((G2 * p1s * w(r) ** 2).subs(subGx))
    log("[5] transverse condensate current J = G^2 p1 w0^2 (RHS of induced ODE)")

    # conservation of S given only the w0  and bhat ODEs; use BRANE Christoffels
    gdn_b = sp.diag(-Uf, sp.exp(2 * Vf), sp.exp(2 * Vf), sp.exp(2 * Wf), 1 / Uf)
    gup_b = sp.diag(-1 / Uf, sp.exp(-2 * Vf), sp.exp(-2 * Vf), sp.exp(-2 * Wf), Uf)
    GAM_b = christoffel(gdn_b, gup_b)
    target_G = target.subs(subGx)  # induced ODE with G || x (G2 -> Gs^2)
    bhpp_expr = sp.solve(target_G, sp.Derivative(bh(r), (r, 2)))[0]
    bhpp_rule = {sp.Derivative(bh(r), (r, 2)): bhpp_expr}
    # re-attach the e^{iGx} factor for the divergence (x-derivative acts on it)
    Sfull = [
        [sp.expand(S[m][n] * sp.exp(sp.I * Gs * x)) for n in range(N)] for m in range(N)
    ]
    divS = divergence(Sfull, GAM_b, gup_b)
    ok5 = True
    for nu in range(N):
        e = divS[nu].subs(wpp_rule).subs(bhpp_rule).doit()
        e = sp.expand(e.subs(wpp_rule).subs(bhpp_rule).doit())
        if not is_zero(e):
            ok5 = False
            print(f"    div S nonzero nu={coords[nu]}:")
            sp.pprint(sp.simplify(e))
    assert ok5, "source conservation"
    log("[5] per-harmonic source S conserved on shell (w0 ODE + bhat ODE)")

    # printed table (brane form, e^{iGx} stripped)
    names = ["t", "x", "y", "z", "r"]
    print("\nPer-harmonic source S_MN (G || x, per rho^2 lambda_G, factor e^{iGx}):")
    for m in range(N):
        for n in range(m, N):
            if S[m][n] != 0:  # S is already the coefficient (Eh stripped)
                print(f"  S_{names[m]}{names[n]} = {sp.simplify(S[m][n])}")

    # ---------------------------------------------------------------------------
    # [6] flat limit reproduces the AdS5-Schwarzschild source table
    # ---------------------------------------------------------------------------
    log("[6] flat-limit check against the AdS5-Schwarzschild source table ...")
    u = sp.Symbol("u", positive=True)
    uH = sp.Symbol("u_H", positive=True)
    fu = 1 - u**4 / uH**4
    # map coordinate r -> u by substituting the metric AND relabelling functions;
    # work with explicit metric so the check is a literal comparison
    flat = {gtt: -fu / u**2, grr: 1 / (u**2 * fu), gpp: 1 / u**2, gzz: 1 / u**2}
    # rebuild the flat source table in (u, w0(u), bhat(u)); compare S (as functions) at u=r
    wu = sp.Function("w0")(u)
    bhu = sp.Function("bhat")(u)
    wpu = sp.Derivative(wu, u)
    S_flat = {
        (0, 0): fu * u**2 * (fu * wpu**2 + B * (bhu - wu**2)),
        (1, 1): u**2 * B * (bhu - wu**2),
        (2, 2): u**2 * B * (bhu - wu**2),
        (3, 3): -(u**2) * (fu * wpu**2 + B * (bhu - wu**2)),
        (4, 4): u**2 * (wpu**2 - (B / fu) * (bhu - wu**2)),
        (1, 4): -sp.I * (B * u**2 / Gs) * sp.Derivative(bhu, u),
    }
    # our generic-metric S in the flat metric, then relabel r -> u
    ok6 = True
    mrel = {
        w(r): wu,
        bh(r): bhu,
        sp.Derivative(w(r), r): wpu,
        sp.Derivative(bh(r), r): sp.Derivative(bhu, u),
    }
    for (m, n), val in S_flat.items():
        ours = Sharm_gen[m][n].subs(subGx)  # e^{iGx} already stripped
        ours = ours.subs(flat).subs(mrel).subs(r, u)
        if not is_zero(sp.simplify(sp.cancel(ours - val))):
            ok6 = False
            print(
                f"    flat mismatch S_{names[m]}{names[n]}: {sp.simplify(ours - val)}"
            )
    assert ok6, "flat source table"
    log("[6] flat limit reproduces the AdS5-Schwarzschild source table exactly")

    # ---------------------------------------------------------------------------
    # code generation: systems/dk_stress.py
    # ---------------------------------------------------------------------------
    rs, Us, Ups, Vs, Vps, Ws, Wps = sp.symbols("r U Up V Vp W Wp", real=True)
    W0, W0p, BH, BHp = sp.symbols("W0 W0p BH BHp", real=True)
    # flatten the derivative-symbol map for pycode; S is already in (U,V,W) brane form
    branevals = {
        Uf: Us,
        Vf: Vs,
        Wf: Ws,
        sp.Derivative(Uf, r): Ups,
        sp.Derivative(Vf, r): Vps,
        sp.Derivative(Wf, r): Wps,
    }
    matter = {
        w(r): W0,
        sp.Derivative(w(r), r): W0p,
        bh(r): BH,
        sp.Derivative(bh(r), r): BHp,
    }

    def brane_flat(e):
        e = sp.expand(e.subs(branevals).subs(matter))
        return sp.simplify(e)

    from backreaction import paths  # noqa: E402

    gen_path = paths.SYSTEMS / "dk_stress.py"
    Sexpr = {}
    for m, n in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (1, 4)]:
        Sexpr[(m, n)] = brane_flat(S[m][n])  # S already brane, Eh stripped
    Jexpr = brane_flat(Jsrc)
    with open(gen_path, "w") as fh:
        fh.write(
            '"""AUTO-GENERATED by dk_stress.py -- do not edit.\n\n'
            "Per-harmonic O(rho^2) condensate stress S_MN(r) e^{iGx} and\n"
            "transverse colour-3 current J(r) on the (U,V,W) magnetic brane\n"
            "(G || x, per rho^2 lambda_G).  Coord order (t,x,y,z,r).\n"
            "Args: (r, U, Up, V, Vp, W, Wp, B, G, W0, W0p, BH, BHp) with\n"
            "W0=w0, W0p=w0', BH=bhat, BHp=bhat'.\n"
            '"""\n\nimport math  # noqa: F401\n\n\n'
        )
        args = "r, U, Up, V, Vp, W, Wp, B, G, W0, W0p, BH, BHp"
        for (m, n), key in [
            ((0, 0), "S_tt"),
            ((1, 1), "S_xx"),
            ((2, 2), "S_yy"),
            ((3, 3), "S_zz"),
            ((4, 4), "S_rr"),
            ((1, 4), "S_xr"),
        ]:
            fh.write(f"def {key}({args}):\n    return {sp.pycode(Sexpr[(m, n)])}\n\n\n")
        fh.write(f"def J_transverse({args}):\n    return {sp.pycode(Jexpr)}\n")
    log(f"[gen] wrote {gen_path}")
    log("done -- all checks passed")

    # ---------------------------------------------------------------------------
    # provenance stamp
    # ---------------------------------------------------------------------------
    # Records which revision of this file produced the generated module, plus a
    # digest of the module itself, so that a hand edit or a stale regeneration is
    # visible instead of silent.  Must stay last: it digests the finished file.
    # See backreaction/generated.py and tests/test_generated_systems.py.
    from backreaction.generated import stamp  # noqa: E402

    stamp(gen_path, __file__)
    log(f"[stamp] provenance written to {gen_path}")


if __name__ == "__main__":
    main()
