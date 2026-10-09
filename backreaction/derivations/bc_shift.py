"""The O(alpha) shift of B_c: perturbed zero-mode operator.

The G = 0, O(alpha B^2) metric correction (magnetic-brane deformation, from
derivations/g0.py / g0_thermo.py) feeds back into the probe-limit linear-instability
eigenvalue problem.  This script derives the perturbed radial operator from
the *full non-abelian* Yang-Mills action on the general diagonal metric

    ds^2 = (f/u^2)(1 + a Ht) (-dt^2) + (1/u^2)(1 + a Hx)(dx^2 + dy^2)
         + (1/u^2)(1 + a Hz) dz^2 + (1/(u^2 f))(1 + a Hu) du^2 ,

with the LLL ansatz W_x = w(u) psi(x), W_y = -i W_x, psi = exp(-B x^2/2),
A^3_y = B x (k = 0 slice; the harmonic label is immaterial at G = 0).

Checks / derivations
--------------------
[1] Background reduction: the quadratic action density reduces to
        E0 = N(B) [ (f/u) w'^2 - (B/u) w^2 ]   per unit y,z,t volume,
    whose Euler-Lagrange equation is exactly the probe-limit zero-mode ODE
    f w'' + (f' - f/u) w' + B w = 0 -- including the magnetic-moment and
    diamagnetic F^3-cross terms (the eigenvalue +B is the net LLL+moment
    value; a machine check of the dictionary).
[2] O(alpha) reduction: E1 = N(B) [ dP w'^2 - dQ w^2 ] with
        dP = (f/u) pi_P(H),   dQ = (B/u) pi_Q(H) + cross terms,
    linear in (Ht, Hx, Hz, Hu) with *no* H' terms after integrating the
    single w w' cross term by parts (boundary terms vanish for w ~ u^2).
[3] Perturbation theory: delta B_c = int_0^{u_H} (dP w0'^2 - dQ w0^2) du
    with the J_2 = int w0^2/u = 1 normalisation (denominator = J_2).
[4] Gauge invariance of delta B_c under xi_u diffeomorphisms: for a
    pure-gauge metric perturbation the coefficient shifts are exactly the
    reparametrisation of the background quadratic form,
        dP[delta_xi H] = xi Pb' - xi' Pb,  dQ[delta_xi H] = xi Qb' + xi' Qb
    with xi = xi^u = u^2 f xi_u, Pb = f/u, Qb = B/u -- an algebraic
    identity (verified).  The PT integral of such a shift vanishes on shell
    after one integration by parts (integrand = 2 w0' [(Pb w0')' + Qb w0]),
    with boundary terms vanishing for w0 ~ u^2 and regular xi.
[5] Code generation: systems/bc_shift.py with the coefficient functions of
    (Ht, Hx, Hz, Hu) in dP and dQ.

Run:  uv run python -m backreaction.derivations.bc_shift   (~1 min)
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
    return sp.cancel(sp.together(e)) == 0


t, x, y, z = sp.symbols("t x y z", real=True)
u = sp.Symbol("u", positive=True)
coords = [t, x, y, z, u]
uH, B = sp.symbols("u_H B", positive=True)
a = sp.Symbol("a", positive=True)  # alpha bookkeeping

f = 1 - u**4 / uH**4
w = sp.Function("w0", real=True)


def main():
    wp = sp.Derivative(w(u), u)
    wpp_rule = {
        sp.Derivative(w(u), (u, 2)): -((sp.diff(f, u) - f / u) * wp + B * w(u)) / f
    }

    def on_shell(expr):
        e = sp.expand(sp.sympify(expr).doit())
        while e.has(sp.Derivative(w(u), (u, 2))):
            e = sp.expand(e.subs(wpp_rule).doit())
        return e

    Ht = sp.Function("H_tt")
    Hx = sp.Function("H_xx")
    Hz = sp.Function("H_zz")
    Hu = sp.Function("H_uu")

    gdn = sp.diag(
        -(f / u**2) * (1 + a * Ht(u)),
        (1 / u**2) * (1 + a * Hx(u)),
        (1 / u**2) * (1 + a * Hx(u)),
        (1 / u**2) * (1 + a * Hz(u)),
        (1 / (u**2 * f)) * (1 + a * Hu(u)),
    )
    N = 5

    # inverse metric and sqrt(-g) to O(a)
    gup = sp.diag(
        *[sp.expand(sp.series(1 / gdn[i, i], a, 0, 2).removeO()) for i in range(N)]
    )
    sqrtg = sp.sqrt(sp.expand(-sp.prod([gdn[i, i] for i in range(N)])))
    sqrtg = sp.expand(sp.series(sp.powsimp(sqrtg, force=True), a, 0, 2).removeO())

    # ---------------------------------------------------------------------------
    # gauge fields: LLL ansatz + background
    # ---------------------------------------------------------------------------
    psi = sp.exp(-B * x**2 / 2)
    A = [[sp.S.Zero] * N for _ in range(3)]  # A[a][mu], a = 1,2,3
    A[0][1] = w(u) * psi  # A^1_x = w psi        (W_x = w psi)
    A[1][2] = -w(u) * psi  # A^2_y = -w psi       (W_y = -i w psi)
    A[2][2] = B * x  # A^3_y = B x

    eps3 = sp.LeviCivita
    F = [[[sp.S.Zero] * N for _ in range(N)] for _ in range(3)]
    for ai in range(3):
        for m in range(N):
            for n in range(N):
                e = sp.diff(A[ai][n], coords[m]) - sp.diff(A[ai][m], coords[n])
                for bi in range(3):
                    for ci in range(3):
                        if eps3(ai + 1, bi + 1, ci + 1) != 0:
                            e += eps3(ai + 1, bi + 1, ci + 1) * A[bi][m] * A[ci][n]
                F[ai][m][n] = sp.expand(e)

    # Lagrangian density -1/4 sqrt(-g) F^a_{mu nu} F^{a mu nu}, diag metric
    L = sp.S.Zero
    for ai in range(3):
        for m in range(N):
            for n in range(N):
                if F[ai][m][n] != 0:
                    L += -sp.Rational(1, 4) * gup[m, m] * gup[n, n] * F[ai][m][n] ** 2
    L = sp.expand(sqrtg * L)

    # truncate at O(w^2) via a scaling parameter (w -> s w); the O(s^1) part
    # must vanish identically (no terms linear in W exist)
    Wsym, Wpsym, ssym = sp.symbols("W Wp s", real=True)
    Lflat = sp.expand(L.subs(wp, Wpsym).subs(w(u), Wsym))
    Lscaled = sp.expand(Lflat.subs({Wsym: ssym * Wsym, Wpsym: ssym * Wpsym}))
    lin_part = sp.expand(Lscaled.coeff(ssym, 1))
    assert is_zero(lin_part), "unexpected O(w) terms in the action"
    L2 = sp.expand(Lscaled.coeff(ssym, 2)).subs({Wsym: w(u), Wpsym: wp})
    L2 = sp.expand(L2)

    # integrate over x (Gaussian, B > 0); y, z, t volume factors out
    L2x = sp.integrate(L2, (x, -sp.oo, sp.oo))
    L2x = sp.expand(sp.powsimp(L2x, force=True))

    # split into O(a^0) and O(a^1)
    E0 = sp.expand(L2x.coeff(a, 0))
    E1 = sp.expand(L2x.coeff(a, 1))

    # ---------------------------------------------------------------------------
    # [1] background reduction
    # ---------------------------------------------------------------------------
    pref = sp.sqrt(sp.pi) / sp.sqrt(B)  # int psi^2 dx
    E0n = sp.expand(sp.cancel(sp.together(E0 / pref)))
    # expected: -(f/u) w'^2 + (B/u) w^2 up to overall normalisation c0
    cP = sp.cancel(sp.together(E0n.coeff(wp**2)))
    c0 = sp.cancel(sp.together(-cP / (f / u)))
    resid0 = sp.expand(E0n - sp.expand(c0 * (-(f / u) * wp**2 + (B / u) * w(u) ** 2)))
    ok1a = is_zero(resid0)
    log(f"[1] E0 = c0 [-(f/u) w'^2 + (B/u) w^2] with c0 = {sp.simplify(c0)}: {ok1a}")
    assert ok1a
    # Euler-Lagrange => zero-mode ODE
    EL0 = sp.expand(sp.diff(sp.diff(E0n, wp), u).doit() - sp.diff(E0n, w(u)))
    target0 = f * sp.Derivative(w(u), (u, 2)) + (sp.diff(f, u) - f / u) * wp + B * w(u)
    ratio = sp.cancel(sp.together(EL0 / target0))
    ok1b = not any(ratio.has(v) for v in (w(u), wp)) and not is_zero(ratio)
    log(f"[1] EL equation = [{sp.simplify(ratio)}] x (zero-mode ODE): {ok1b}")
    assert ok1b

    # ---------------------------------------------------------------------------
    # [2] O(alpha) reduction
    # ---------------------------------------------------------------------------
    E1n = sp.expand(sp.cancel(sp.together(E1 / pref / c0)))
    # structure: terms in w'^2, w^2, w w' (and H, H')
    cross = sp.expand(E1n.coeff(w(u) * wp))
    ok2a = is_zero(cross)
    if not ok2a:
        # integrate the w w' cross term by parts:  c(u) w w' -> -(c'/2) w^2
        E1n = sp.expand(
            E1n
            - sp.expand(cross * w(u) * wp)
            + sp.expand(-sp.diff(cross, u) / 2 * w(u) ** 2).doit()
        )
    log(f"[2] w w' cross term {'absent' if ok2a else 'removed by parts'}")
    dP = sp.expand(E1n.coeff(wp**2)) * (-1)  # E1n = -(dP w'^2 - dQ w^2)
    dQ = sp.expand(E1n.coeff(w(u) ** 2))
    resid1 = sp.expand(E1n + sp.expand(dP * wp**2) - sp.expand(dQ * w(u) ** 2))
    ok2b = is_zero(resid1)
    log(f"[2] E1 = -(dP w'^2 - dQ w^2) with dP, dQ linear in H: {ok2b}")
    assert ok2b
    Hfns = [Ht(u), Hx(u), Hz(u), Hu(u)]
    ok2c = all(
        is_zero(sp.diff(dP, F2, 2)) and is_zero(sp.diff(dQ, F2, 2)) for F2 in Hfns
    )
    log(f"[2] dP, dQ linear in the H's: {ok2c}")
    assert ok2c
    print("    dP =", sp.collect(sp.expand(dP), Hfns))
    print("    dQ =", sp.collect(sp.expand(dQ), Hfns))

    # ---------------------------------------------------------------------------
    # [4] gauge invariance of delta B_c under xi_u
    # ---------------------------------------------------------------------------
    log("[4] gauge invariance of the perturbation integral ...")
    xiU = sp.Function("xi_u")
    # gauge shifts of the H's under xi = xi_u(u) du (derived in derivations/g0.py
    # [2]; delta H_uu recomputed from first principles here):
    Gam_uuu = sp.cancel(sp.diff(1 / (u**2 * f), u) / (2 / (u**2 * f)))
    gauge = {
        Ht(u): -2 * u * (1 + u**4 / uH**4) * xiU(u),
        Hx(u): -2 * u * f * xiU(u),
        Hz(u): -2 * u * f * xiU(u),
        Hu(u): sp.expand(
            u**2 * f * (2 * sp.Derivative(xiU(u), u) - 2 * Gam_uuu * xiU(u))
        ),
    }
    dP_g = sp.expand(dP.subs(gauge).doit())
    dQ_g = sp.expand(dQ.subs(gauge).doit())
    # expected: pure reparametrisation of the background form by
    # xi = xi^u = u^2 f xi_u (contravariant component)
    xi_up = u**2 * f * xiU(u)
    Pb, Qb = f / u, B / u
    dP_expect = sp.expand(
        (xi_up * sp.diff(Pb, u) - sp.diff(xi_up, u).doit() * Pb).doit()
    )
    dQ_expect = sp.expand(
        (xi_up * sp.diff(Qb, u) + sp.diff(xi_up, u).doit() * Qb).doit()
    )
    ok4a = is_zero(sp.expand(dP_g - dP_expect))
    ok4b = is_zero(sp.expand(dQ_g - dQ_expect))
    log(
        f"[4] dP[delta_xi H] = xi Pb' - xi' Pb: {ok4a};  "
        f"dQ[delta_xi H] = xi Qb' + xi' Qb: {ok4b}"
    )
    assert ok4a and ok4b
    # the PT integral of such a shift is int 2 w0' [(Pb w0')' + Qb w0] xi du
    # after integrating by parts, which vanishes on the zero-mode shell:
    integrand_var = sp.expand(dP_expect * wp**2 - dQ_expect * w(u) ** 2)
    onshell_piece = on_shell(
        sp.expand(
            integrand_var + sp.diff(xi_up * (Pb * wp**2 + Qb * w(u) ** 2), u).doit()
        )
    )
    # what remains under the integral after removing the total derivative:
    ok4d = is_zero(
        sp.expand(
            onshell_piece
            - on_shell(
                sp.expand(2 * xi_up * wp * (sp.diff(Pb * wp, u).doit() + Qb * w(u)))
            )
        )
    )
    ok4e = is_zero(
        on_shell(sp.expand(2 * xi_up * wp * (sp.diff(Pb * wp, u).doit() + Qb * w(u))))
    )
    log(
        f"[4] variation = total derivative + 2 xi w0' [(Pb w0')' + Qb w0]: "
        f"{ok4d}; on-shell remainder vanishes: {ok4e}"
    )
    assert ok4d and ok4e

    # ---------------------------------------------------------------------------
    # [5] code generation
    # ---------------------------------------------------------------------------
    from backreaction import paths  # noqa: E402

    gen_path = paths.SYSTEMS / "bc_shift.py"
    Hsym = sp.symbols("ht hx hz hu", real=True)
    Hpsym = sp.symbols("dht dhx dhz dhu", real=True)
    flatH = {}
    for F2, s0, s1 in zip(Hfns, Hsym, Hpsym, strict=True):
        flatH[sp.Derivative(F2, u)] = s1
        flatH[F2] = s0

    def flatten(e):
        e = sp.expand(sp.sympify(e).doit()).subs(flatH)
        return sp.expand(e)

    dPf = flatten(dP)
    dQf = flatten(dQ)
    with open(gen_path, "w") as fh:
        fh.write(
            '"""AUTO-GENERATED by bc_shift.py -- do not edit.\n\n'
            "O(alpha) perturbation of the probe-limit zero-mode operator by a\n"
            "diagonal G = 0 metric correction (ht, hx, hz, hu) =\n"
            "(Htt, Hxx, Hzz, Huu), per unit alpha:\n"
            "    delta B_c = int_0^{u_H} [dP w0'^2 - dQ w0^2] du   (J2 = 1)\n"
            '"""\n\nimport math  # noqa: F401\n\n\n'
            "def dP(u, B, u_H, ht, hx, hz, hu, dht, dhx, dhz, dhu):\n"
            f"    return {sp.pycode(dPf)}\n\n\n"
            "def dQ(u, B, u_H, ht, hx, hz, hu, dht, dhx, dhz, dhu):\n"
            f"    return {sp.pycode(dQf)}\n"
        )
    log(f"[5] wrote {gen_path}")
    log("done")

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
