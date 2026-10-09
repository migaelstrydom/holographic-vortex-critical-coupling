"""Time-dependent charged sector on a general static magnetic brane: every
growing mode is purely imaginary, so the charged instability sets in through
the static zero mode at omega = 0 (paper app. B.2).

Question.  The onset B_c and the critical coupling alpha_* = 2/3 are found
from static zero modes.  With ingoing conditions at the horizon the
quasinormal problem is not self-adjoint, and the magnetic field breaks time
reversal, so could a pair of complex frequencies cross into the upper half
plane at omega != 0, before (or without) any static zero mode?

Setting.  As in `dk_landau_sectors.py`: coordinates (t, r, x, y, z), a general
diagonal static metric g_tt(r) < 0, g_rr(r), g_xx = g_yy = g_pp(r), g_zz(r),
SU(2) with the coupling absorbed, background A^3_y = B x and NOTHING else --
no A_t (no charge density) and no g_ti (no rotation).  Charged fluctuations
W_M = A^1_M + i A^2_M sort by Landau level n,

    W_x + i W_y = p psi_n,  W_x - i W_y = m psi_{n-2},
    W_t = a psi_{n-1},  W_r = e psi_{n-1},  W_z = c psi_{n-1},

now with every amplitude a function of (t, r) and an optional e^{i k z}.

Results (asserted below)
-------
[1] Structure of the quadratic action, n = 0..3, all five components
    including W_t and W_r, general (t, r) dependence.  Splitting
    -1/4 sqrt(-g) F^2 by index pair,
        L = K - V_rad - V_perp,
        K      = (t i) pairs: +1/2 sqrt(-g)(-g^tt) g^ii |F_ti|^2,
        V_rad  = (r i) pairs, i = x, y, z: +1/2 sqrt(-g) g^rr g^ii |F_ri|^2,
        V_perp = pairs within (x, y, z), including the moment term.
    That d/dt and W_t sit only in K, that V_perp has no W_r and no r- or
    t-derivative, and that K and V_rad are sums of squares with positive
    weights hold by construction on a diagonal metric with A^3_t = 0, and are
    not asserted.  What is asserted is the reduction:
      (a) K after the x integral equals the closed form built from the ladder
          relations, F_t+ = (p_t + sqrt(B) a) psi_n,
          F_t- = (m_t - 2(n-1) sqrt(B) a) psi_{n-2}, F_tr = (e_t - a_r) psi_{n-1},
          F_tz = c_t psi_{n-1}, with |F_x|^2 + |F_y|^2 = (|F_+|^2 + |F_-|^2)/2
          (negative control: a flipped ladder sign fails);
      (b) V_rad likewise, F_r+ = (p_r + sqrt(B) e) psi_n,
          F_r- = (m_r - 2(n-1) sqrt(B) e) psi_{n-2}, F_rz = c_r psi_{n-1}:
          W_r enters the static energy only through these squares;
      (c) V_perp >= 0 for every n >= 1 (the (p, m) block has zero determinant
          and positive trace as an energy; W_z mass (2n-1) B > 0), and
          V_perp = -(B rho N_0/4) |p|^2 < 0 for n = 0;
      (d) the real and imaginary parts carry identical, decoupled blocks.
[2] With k_z != 0.  Pointwise in x and z, for arbitrary x-profiles times
    e^{ikz}, the (x, z) and (y, z) pairs give the energy
        (1/4) sqrt(-g) g^xx g^zz ( |D_+ W_z - i k W_+|^2 + |D_- W_z - i k W_-|^2 ),
    two squares that mix W_z with W_+-, and the (x, y) pair is independent of
    k and z.  So V_perp stays >= 0 for n >= 1 (the (x, y) form is the static
    T_n, the rest is squares), confirmed at 12 random rational points of
    (g, B, k), with a one-complex-dimensional kernel, the pure gauge
    D(lambda psi_{n-1} e^{ikz}); for n = 0, where W_- = W_z = 0, the k_z term
    is the positive mass k^2 sqrt(-g)/(g_pp g_zz) N_0/4 on p.
[3] The polarised lowest Landau level n = 0 is the single amplitude p(t, r):
        L_0 = (N_0/4)[ R |p_t|^2 - P |p_r|^2 + B rho |p|^2 ],
        R = sqrt(-g)/(-g_tt g_pp),  P = sqrt(-g_tt g_zz/g_rr),
        rho = sqrt(-g_tt g_rr g_zz)/g_pp,
    so with p = 2 w e^{-i omega t}:  (P w')' + B rho w + omega^2 R w = 0, real
    coefficients, omega^2 only.  On the brane P = U e^W, rho = e^{W-2V},
    R = e^W / U.
[4] Ingoing Eddington-Finkelstein form: w = e^{-i omega r_*} phi with
    dr_*/dr = 1/U.  The omega^2 terms cancel identically (R U^2 = P), leaving
        (P phi')' - i omega [ (e^W phi)' + e^W phi' ] + B rho phi = 0,
    linear in omega, phi regular at the horizon: the form solved numerically
    in `numerics/dk_charged_qnm.py`.
[5] Horizon exponents of [3] at a regular horizon U ~ U_1 (r - r_p) are
    +- i omega / U_1; the ingoing branch (r - r_p)^{-i omega/U_1} has modulus
    (r - r_p)^{Im omega / U_1} -> 0 when Im omega > 0, and the flux
    P w-bar w' = U_1 e^{W_0} s (r - r_p)^{2 Im omega / U_1}(1 + ...) vanishes.
[5b] Every component sector at once: a mode regular in ingoing coordinates,
    W = e^{-i omega v}(alpha dv + beta dr + gamma_i dx^i), brought to W_t = 0
    by lambda = W_t/(i omega), has W_r and W_i equal to e^{-i omega v} times
    regular functions, so they vanish like (r - r_p)^{Im omega/U_1}; F_ti and
    F_tr do too.  F_ri does NOT: U F_ri / e^{-i omega v} -> -i omega gamma_i,
    so F_ri ~ (r - r_p)^{Im omega/U_1 - 1}.  The integrands of K and V_rad go
    like |e^{-i omega v}|^2 / U ~ (r - r_p)^{2 Im omega/U_1 - 1}, integrable
    for Im omega > 0, and the surface term sqrt(-g) g^rr g^ii W-bar_i F_ri like
    (r - r_p)^{2 Im omega/U_1} -> 0.
[6] Positive control: a chemical potential A^3_t = mu(r) puts a term
    mu (p1 p2_t - p2 p1_t) into the n = 0 action, so omega enters as
    (omega - mu)^2 and the argument below fails; it is the absence of A_t
    (and of g_ti), not time-reversal symmetry, that it needs.

The theorem these assemble (paper app. B.2).  Let a mode e^{-i omega t} have
Im omega > 0, ingoing at the horizon and source-free at the boundary.  Gauge
W_t away (lambda = W_t / (i omega)), so that F_ti = -i omega W_i.  The spatial
equations are omega^2 KK W = HH W with KK the positive kinetic weight of K and
HH the Hermitian static operator of V_rad + V_perp.  Contract with W-bar and
integrate over r: by [5] and the normalisable boundary fall-off, the surface
terms vanish (by [5b] the components of W in this gauge vanish like
(r - r_p)^{Im omega/U_1}, the integrands like (r - r_p)^{2 Im omega/U_1 - 1}),
so

    (omega^2 / |omega|^2) <F_t, KK F_t> = <W, HH W>,

whose right side is real.  Hence omega^2 is real, and with Im omega > 0,
omega = i Gamma, and the static energy <W, HH W> = -<F_t, KK F_t> < 0.
  * n >= 1: HH >= 0, also with W_r != 0 and k_z != 0: W_r enters only
    through the squares |F_ri|^2 of V_rad ([1b]), and V_perp >= 0 by [1c],
    [2].  No growing mode at all.
  * n = 0: growing modes are the negative eigenvalues -Gamma^2 of the
    self-adjoint problem -(P w')' - B rho w = -Gamma^2 R w, whose
    continuum starts at 0 (the horizon is at r_* -> -oo).  By Sturm
    oscillation their number equals the number of nodes of the static,
    horizon-regular solution, which is #{k : B_k < B}, and they appear by
    leaving Gamma = 0.
So the charged sector destabilises only through a static zero mode, at the
B_c of the static problem; no oscillatory instability exists in it.  The
neutral sector (A^3 and the metric) is not covered.

Run:  uv run python -m backreaction.derivations.dk_charged_dynamics   (~10 s)
"""

import random
import sys
import time

import mpmath
import sympy as sp

T0 = time.time()
CHECKS = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


def is_zero(e):
    if isinstance(e, sp.MatrixBase):
        return all(is_zero(x) for x in e)
    e = sp.expand(e)
    return e == 0 or sp.simplify(e) == 0


# ---------------------------------------------------------------------------
# geometry: a general static diagonal metric
# ---------------------------------------------------------------------------
t, r, x, y, z = sp.symbols("t r x y z", real=True)
coords = (t, r, x, y, z)
ND = 5
IT, IR, IX, IY, IZ = range(ND)
B = sp.Symbol("B", positive=True)
k = sp.Symbol("k", positive=True)

gtt = sp.Function("g_tt")(r)
grr = sp.Function("g_rr")(r)
gpp = sp.Function("g_pp")(r)
gzz = sp.Function("g_zz")(r)
ginv = sp.diag(1 / gtt, 1 / grr, 1 / gpp, 1 / gpp, 1 / gzz)
att, brr, cpp, dzz = sp.symbols("att brr cpp dzz", positive=True)
POS = {gtt: -att, grr: brr, gpp: cpp, gzz: dzz}
SAMPLE = {
    att: sp.Rational(17, 10),
    brr: sp.Rational(9, 10),
    cpp: sp.Rational(13, 10),
    dzz: sp.Rational(11, 10),
    B: 2,
}
sqrtg = sp.sqrt(-gtt * grr * gpp * gpp * gzz)
P_cf = sp.sqrt(-gtt * gzz / grr)
rho_cf = sp.sqrt(-gtt * grr * gzz) / gpp
R_cf = sqrtg / (-gtt * gpp)


def psi(n):
    """Landau function psi_n = H_n(sqrt(B) x) exp(-B x^2/2); zero for n < 0."""
    if n < 0:
        return sp.S.Zero
    return sp.hermite(n, sp.sqrt(B) * x) * sp.exp(-B * x**2 / 2)


def gaussian_x(expr):
    """int dx of (polynomial in x) * exp(-B x^2): exact Gaussian moments."""
    expr = sp.expand(expr * sp.exp(B * x**2))
    poly = sp.Poly(expr, x)
    out = sp.S.Zero
    for (kk,), coeff in poly.terms():
        if kk % 2:
            continue
        out += coeff * sp.gamma(sp.Rational(kk + 1, 2)) / B ** sp.Rational(kk + 1, 2)
    return sp.expand(out)


def z_average(expr):
    """Average over one period of e^{ikz} (trigonometric polynomial in kz)."""
    expr = sp.expand(sp.expand_trig(expr))
    return sp.expand(k / (2 * sp.pi) * sp.integrate(expr, (z, 0, 2 * sp.pi / k)))


def Nnorm(n):
    return gaussian_x(psi(n) ** 2) if n >= 0 else sp.S.Zero


def field_strength(A):
    F = [[[sp.S.Zero] * ND for _ in range(ND)] for _ in range(3)]
    for ai in range(3):
        for mu in range(ND):
            for nu in range(mu + 1, ND):
                e = sp.diff(A[ai][nu], coords[mu]) - sp.diff(A[ai][mu], coords[nu])
                for bi in range(3):
                    for ci in range(3):
                        lc = sp.LeviCivita(ai + 1, bi + 1, ci + 1)
                        if lc != 0:
                            e += lc * A[bi][mu] * A[ci][nu]
                F[ai][mu][nu] = sp.expand(e)
    return F


def pair_density(F, pairs):
    """-1/4 sqrt(-g) F^2 restricted to the listed index pairs (mu < nu),
    each pair counted twice (F_{mu nu} and F_{nu mu})."""
    L = sp.S.Zero
    for ai in range(3):
        for mu, nu in pairs:
            if F[ai][mu][nu] != 0:
                L += (
                    -sp.Rational(1, 2)
                    * ginv[mu, mu]
                    * ginv[nu, nu]
                    * F[ai][mu][nu] ** 2
                )
    return sp.expand(sqrtg * L)


PAIRS_E = [(IT, i) for i in (IR, IX, IY, IZ)]
PAIRS_R = [(IR, i) for i in (IX, IY, IZ)]
PAIRS_P = [(IX, IY), (IX, IZ), (IY, IZ)]

eps = sp.Symbol("epsilon", positive=True)
NAMES = ("p", "m", "a", "e", "c")


def amplitudes(time_dep):
    args = (t, r) if time_dep else (r,)
    return {nm + s: sp.Function(nm + s, real=True)(*args) for nm in NAMES for s in "12"}


def sector_fields(n, amp, kz=False, a3t=None):
    """Colour components of W = A^1 + i A^2 for level n (optionally e^{ikz})."""
    ph = sp.exp(sp.I * k * z) if kz else sp.S.One

    def cplx(nm):
        return amp[nm + "1"] + sp.I * amp[nm + "2"]

    Pp = cplx("p") * psi(n) * ph
    Mm = cplx("m") * psi(n - 2) * ph
    Wx = (Pp + Mm) / 2
    Wy = (Pp - Mm) / (2 * sp.I)
    Wt = cplx("a") * psi(n - 1) * ph
    Wr = cplx("e") * psi(n - 1) * ph
    Wz = cplx("c") * psi(n - 1) * ph
    W = [Wt, Wr, Wx, Wy, Wz]
    A = [[sp.S.Zero] * ND for _ in range(3)]
    for mu in range(ND):
        A[0][mu] = eps * sp.expand(sp.re(sp.expand(W[mu])))
        A[1][mu] = eps * sp.expand(sp.im(sp.expand(W[mu])))
    A[2][IY] = B * x
    if a3t is not None:
        A[2][IT] = a3t
    return A


def reduce_sector(n, amp, pairs, kz=False):
    """O(eps^2) part of the pair density, integrated over x (and z)."""
    F = field_strength(sector_fields(n, amp, kz))
    L = pair_density(F, pairs)
    assert is_zero(L.coeff(eps, 1)), f"O(W) tadpole at n = {n}"
    L2 = L.coeff(eps, 2)
    if kz:
        L2 = z_average(L2)
    return gaussian_x(L2)


def jet_symbols(expr, amp, time_dep):
    """Replace amplitudes and their first derivatives by plain symbols."""
    out = expr
    rep = {}
    for nm, fobj in amp.items():
        s0 = sp.Symbol(nm, real=True)
        sr = sp.Symbol(nm + "_r", real=True)
        rep[nm], rep[nm + "_r"] = s0, sr
        out = out.subs(sp.Derivative(fobj, r), sr)
        if time_dep:
            st = sp.Symbol(nm + "_t", real=True)
            rep[nm + "_t"] = st
            out = out.subs(sp.Derivative(fobj, t), st)
        out = out.subs(fobj, s0)
    return sp.expand(out), rep


def eigs(Msym):
    """Eigenvalues of a real symmetric numeric sympy matrix, 40 digits."""
    mpmath.mp.dps = 40
    A = mpmath.matrix([[mpmath.mpf(sp.N(v, 45)) for v in row] for row in Msym.tolist()])
    return [
        float(v) if abs(v) > 1e-300 else 0.0 for v in mpmath.eigsy(A, eigvals_only=True)
    ]


def quad_matrix(E, vars_):
    M = sp.zeros(len(vars_))
    for i, u in enumerate(vars_):
        for j, v in enumerate(vars_):
            M[i, j] = sp.diff(E, u, v) / 2
    return sp.simplify(M)


def main():
    # ---------------------------------------------------------------------------
    # [1] structure of the time-dependent quadratic action, n = 0..3
    # ---------------------------------------------------------------------------
    log("[1] time-dependent sectors n = 0..3: L = K - V_rad - V_perp")
    ampT = amplitudes(True)
    for n in range(0, 4):
        parts = {}
        for name, pairs in (("E", PAIRS_E), ("R", PAIRS_R), ("P", PAIRS_P)):
            Ls, rep = jet_symbols(reduce_sector(n, ampT, pairs), ampT, True)
            assert is_zero(sp.diff(Ls, x)), "residual x"
            parts[name] = Ls

        # K and V_rad in closed form from the ladder relations
        # D_+ psi_{n-1} = -sqrt(B) psi_n, D_- psi_{n-1} = 2(n-1) sqrt(B) psi_{n-2}:
        #   F_t+ = (p_t + sqrt(B) a) psi_n,  F_t- = (m_t - 2(n-1) sqrt(B) a) psi_{n-2},
        #   F_tr = (e_t - a_r) psi_{n-1},    F_tz = c_t psi_{n-1},
        #   F_r+ = (p_r + sqrt(B) e) psi_n,  F_r- = (m_r - 2(n-1) sqrt(B) e) psi_{n-2},
        #   F_rz = c_r psi_{n-1},
        # with |F_x|^2 + |F_y|^2 = (|F_+|^2 + |F_-|^2)/2.
        def sq(nm1, nm2, extra1=0, extra2=0, rep=rep):
            return (rep[nm1] + extra1) ** 2 + (rep[nm2] + extra2) ** 2

        sB, g2 = sp.sqrt(B), 2 * (n - 1) * sp.sqrt(B)
        Kexp = (
            sp.Rational(1, 2)
            * sqrtg
            * (-ginv[IT, IT])
            * (
                ginv[IR, IR]
                * Nnorm(n - 1)
                * sq("e1_t", "e2_t", -rep["a1_r"], -rep["a2_r"])
                + ginv[IX, IX]
                / 2
                * (
                    Nnorm(n) * sq("p1_t", "p2_t", sB * rep["a1"], sB * rep["a2"])
                    + Nnorm(n - 2)
                    * sq("m1_t", "m2_t", -g2 * rep["a1"], -g2 * rep["a2"])
                )
                + ginv[IZ, IZ] * Nnorm(n - 1) * sq("c1_t", "c2_t")
            )
        )
        Vexp = (
            sp.Rational(1, 2)
            * sqrtg
            * ginv[IR, IR]
            * (
                ginv[IX, IX]
                / 2
                * (
                    Nnorm(n) * sq("p1_r", "p2_r", sB * rep["e1"], sB * rep["e2"])
                    + Nnorm(n - 2)
                    * sq("m1_r", "m2_r", -g2 * rep["e1"], -g2 * rep["e2"])
                )
                + ginv[IZ, IZ] * Nnorm(n - 1) * sq("c1_r", "c2_r")
            )
        )
        if n == 0:  # W_t = a psi_{-1} and W_r = e psi_{-1} vanish identically
            absent = {rep[q]: 0 for q in ("a1", "a2", "e1", "e2", "a1_r", "a2_r")}
            Kexp, Vexp = Kexp.subs(absent), Vexp.subs(absent)
        check(
            f"[1a] n = {n}: K = (t i) pairs = (1/2) sqrt(-g)(-g^tt) g^ii |F_ti|^2 in "
            f"closed form (W_t only through F_ti)",
            is_zero((parts["E"] - Kexp).subs(POS)),
        )
        check(
            f"[1b] n = {n}: V_rad = (r i) pairs = (1/2) sqrt(-g) g^rr g^ii |F_ri|^2 in "
            f"closed form (W_r only through F_ri)",
            is_zero((-parts["R"] - Vexp).subs(POS)),
        )
        if n == 2:
            check(
                "[1a] negative control: the closed form with D_+ psi_1 = +sqrt(B) psi_2 fails",
                not is_zero((parts["E"] - Kexp.subs(sB, -sB)).subs(POS)),
            )
        # transverse block
        pm = [rep[s] for s in ("p1", "m1", "c1", "p2", "m2", "c2")]
        T = quad_matrix(parts["P"], pm)
        check(
            f"[1d] n = {n}: Re/Im blocks identical and decoupled",
            is_zero(T[0:3, 0:3] - T[3:6, 3:6]) and is_zero(T[0:3, 3:6]),
        )
        T3 = sp.simplify(T[0:3, 0:3] / (B * rho_cf))
        if n == 0:
            check(
                "[1c] n = 0: V_perp density = +(B rho N_0/4) p^2 (destabilising)",
                is_zero((T3[0, 0] - Nnorm(0) / 4).subs(POS))
                and is_zero(T3[1:, :])
                and is_zero(T3[:, 1:]),
            )
        else:
            Tpm = T3[0:2, 0:2]
            cc = T3[2, 2]
            ok = is_zero(T3[0:2, 2]) and is_zero(Tpm.det())
            ok &= float(Tpm.trace().subs(POS).subs({B: 2.0})) <= 0
            ok &= float(cc.subs(POS).subs(SAMPLE)) < 0
            check(
                f"[1c] n = {n}: V_perp >= 0 (det 0, trace and W_z mass of definite sign)",
                ok,
            )

    # ---------------------------------------------------------------------------
    # [2] k_z != 0: the transverse form stays non-negative for n >= 1
    # ---------------------------------------------------------------------------
    log("[2] k_z != 0: static transverse form, n = 0..3")
    # pointwise in x and z, arbitrary W_+(x), W_-(x), W_z(x) times e^{ikz}
    xf = {
        nm: sp.Function(nm, real=True)(x) for nm in ("P1", "P2", "M1", "M2", "C1", "C2")
    }
    phz = sp.exp(sp.I * k * z)
    Wp_ = (xf["P1"] + sp.I * xf["P2"]) * phz
    Wm_ = (xf["M1"] + sp.I * xf["M2"]) * phz
    Wz_ = (xf["C1"] + sp.I * xf["C2"]) * phz
    Wxyz = [sp.S.Zero, sp.S.Zero, (Wp_ + Wm_) / 2, (Wp_ - Wm_) / (2 * sp.I), Wz_]
    Az = [[sp.S.Zero] * ND for _ in range(3)]
    for mu in range(ND):
        Az[0][mu] = eps * sp.expand(sp.re(sp.expand(Wxyz[mu])))
        Az[1][mu] = eps * sp.expand(sp.im(sp.expand(Wxyz[mu])))
    Az[2][IY] = B * x
    Fz = field_strength(Az)
    dens_z = pair_density(Fz, [(IX, IZ), (IY, IZ)]).coeff(eps, 2)
    dens_xy = pair_density(Fz, [(IX, IY)]).coeff(eps, 2)

    def cc(e):
        """Complex conjugate: every function and symbol here is real."""
        return e.subs(sp.I, -sp.I)

    def Dpm(f, sgn):
        return sp.diff(f, x) - sgn * B * x * f

    Fpz = Dpm(Wz_, +1) - sp.I * k * Wp_
    Fmz = Dpm(Wz_, -1) - sp.I * k * Wm_
    two_sq = (
        sp.Rational(1, 4)
        * sqrtg
        * ginv[IX, IX]
        * ginv[IZ, IZ]
        * (Fpz * cc(Fpz) + Fmz * cc(Fmz))
    )
    check(
        "[2] k_z: (x,z)+(y,z) energy = (1/4) sqrt(-g) g^xx g^zz (|D_+W_z - ik W_+|^2 + "
        "|D_-W_z - ik W_-|^2), pointwise",
        is_zero(sp.expand(sp.powsimp(sp.expand((-dens_z - two_sq).subs(POS))))),
    )
    check(
        "[2] k_z: the (x,y) energy contains neither k nor z",
        is_zero(sp.diff(dens_xy, k))
        and is_zero(sp.diff(sp.expand(sp.powsimp(dens_xy)), z)),
    )
    ampS = amplitudes(False)
    rng = random.Random(7)
    for n in range(0, 4):
        Ls, rep = jet_symbols(reduce_sector(n, ampS, PAIRS_P, kz=True), ampS, False)
        present = ["p"] + (["c"] if n >= 1 else []) + (["m"] if n >= 2 else [])
        vars_ = [rep[nm + s] for s in "12" for nm in present]
        T = quad_matrix(Ls, vars_).subs(POS)
        if n == 0:
            expect = (
                B * rho_cf * Nnorm(0) / 4 - k**2 * sqrtg / (gpp * gzz) * Nnorm(0) / 4
            ).subs(POS)
            check(
                "[2] n = 0: k_z adds only a positive mass k^2 to p",
                is_zero(T[0, 0] - expect) and is_zero(T[1, 1] - expect),
            )
            continue
        worst, nulls = 0.0, set()
        for _ in range(12):
            pt = {
                att: sp.Rational(rng.randint(1, 30), 7),
                brr: sp.Rational(rng.randint(1, 30), 7),
                cpp: sp.Rational(rng.randint(1, 30), 7),
                dzz: sp.Rational(rng.randint(1, 30), 7),
                B: sp.Rational(rng.randint(1, 30), 7),
                k: sp.Rational(rng.randint(1, 30), 7),
            }
            evs = eigs(T.subs(pt))
            scale = max(abs(v) for v in evs)
            worst = max(worst, max(evs) / scale)
            nulls.add(sum(1 for v in evs if abs(v) < 1e-25 * scale))
        check(
            f"[2] n = {n}: k_z != 0 transverse density negative semi-definite "
            f"(12 random points), kernel = pure gauge",
            worst < 1e-25 and nulls == {2},
            f"max ev/|ev| = {worst:.1e}, null dims {sorted(nulls)}",
        )

    # ---------------------------------------------------------------------------
    # [3] the n = 0 channel: one amplitude, omega^2 only
    # ---------------------------------------------------------------------------
    log("[3] the polarised lowest Landau level with time dependence")
    F0 = field_strength(sector_fields(0, ampT))
    L0 = pair_density(F0, PAIRS_E + PAIRS_R + PAIRS_P)
    L0 = gaussian_x(L0.coeff(eps, 2))
    L0s, rep = jet_symbols(L0, ampT, True)
    p1, p2 = rep["p1"], rep["p2"]
    expect = (
        Nnorm(0)
        / 4
        * (
            R_cf * (rep["p1_t"] ** 2 + rep["p2_t"] ** 2)
            - P_cf * (rep["p1_r"] ** 2 + rep["p2_r"] ** 2)
            + B * rho_cf * (p1**2 + p2**2)
        )
    )
    check(
        "[3] n = 0: L_0 = (N_0/4)[R |p_t|^2 - P |p_r|^2 + B rho |p|^2]",
        is_zero((L0s - expect).subs(POS)),
    )
    Ub = sp.Function("U", real=True)(r)
    Vb = sp.Function("V", real=True)(r)
    Wb = sp.Function("W", real=True)(r)
    DK = {gtt: -Ub, grr: 1 / Ub, gpp: sp.exp(2 * Vb), gzz: sp.exp(2 * Wb)}
    POSDK = {Ub: sp.Symbol("Upos", positive=True)}

    def on_brane(e):
        return sp.simplify(e.subs(DK).subs(POSDK))

    Upos = POSDK[Ub]
    check(
        "[3] brane: P = U e^W, rho = e^{W-2V}, R = e^W / U",
        is_zero(on_brane(P_cf) - Upos * sp.exp(Wb))
        and is_zero(on_brane(rho_cf) - sp.exp(Wb - 2 * Vb))
        and is_zero(on_brane(R_cf) - sp.exp(Wb) / Upos),
    )

    # ---------------------------------------------------------------------------
    # [4] ingoing Eddington-Finkelstein form: linear in omega
    # ---------------------------------------------------------------------------
    log("[4] ingoing EF reduction w = e^{-i omega r_*} phi")
    om = sp.Symbol("omega")
    phi = sp.Function("phi")(r)
    rstar = sp.Function("r_star")(r)
    P_b, rho_b, R_b = Ub * sp.exp(Wb), sp.exp(Wb - 2 * Vb), sp.exp(Wb) / Ub
    w = sp.exp(-sp.I * om * rstar) * phi
    ode_w = sp.diff(P_b * sp.diff(w, r), r) + B * rho_b * w + om**2 * R_b * w
    ode_w = ode_w.subs(sp.Derivative(rstar, (r, 2)), sp.diff(1 / Ub, r))
    ode_w = ode_w.subs(sp.Derivative(rstar, r), 1 / Ub)
    reduced = sp.expand(sp.simplify(ode_w * sp.exp(sp.I * om * rstar)))
    target = (
        sp.diff(P_b * sp.diff(phi, r), r)
        - sp.I * om * (sp.diff(sp.exp(Wb) * phi, r) + sp.exp(Wb) * sp.diff(phi, r))
        + B * rho_b * phi
    )
    check(
        "[4] omega^2 cancels: (P phi')' - i omega[(e^W phi)' + e^W phi'] + B rho phi = 0",
        is_zero(reduced - sp.expand(target)),
    )

    # ---------------------------------------------------------------------------
    # [5] horizon exponents and the vanishing flux for Im omega > 0
    # ---------------------------------------------------------------------------
    log("[5] horizon exponents of the n = 0 equation")
    s_ = sp.Symbol("s")
    rho_h = sp.Symbol("varrho", positive=True)  # r - r_p
    U2, V0, V1, W0, W1 = sp.symbols("U2 V0 V1 W0 W1", real=True)
    U1 = sp.Symbol("U1", positive=True)
    loc = {
        Ub: U1 * rho_h + U2 * rho_h**2,
        Vb: V0 + V1 * rho_h,
        Wb: W0 + W1 * rho_h,
    }
    wloc = rho_h**s_
    Pl = (Ub * sp.exp(Wb)).subs(loc)
    Rl = (sp.exp(Wb) / Ub).subs(loc)
    rhol = sp.exp(Wb - 2 * Vb).subs(loc)
    ode_l = sp.diff(Pl * sp.diff(wloc, rho_h), rho_h) + (B * rhol + om**2 * Rl) * wloc
    lead = sp.limit(sp.simplify(ode_l * rho_h ** (1 - s_)), rho_h, 0)
    roots = sp.solve(sp.simplify(lead / sp.exp(W0)), s_)
    check(
        "[5] horizon exponents s = +- i omega / U_1",
        set(sp.simplify(rt) for rt in roots) == {sp.I * om / U1, -sp.I * om / U1},
        f"roots {roots}",
    )
    s_in = -sp.I * om / U1
    a_, b_ = sp.symbols("a b", real=True)
    mod_exp = sp.re(s_in.subs(om, a_ + sp.I * b_))
    check("[5] ingoing |w| ~ (r - r_p)^{Im omega/U_1}", is_zero(mod_exp - b_ / U1))
    s_ab = s_in.subs(om, a_ + sp.I * b_)
    w_in = rho_h**s_ab
    flux = Pl * cc(w_in) * sp.diff(w_in, rho_h)  # P w-bar w', rho_h > 0
    flux_lead = sp.limit(
        sp.simplify(sp.powsimp(flux * rho_h ** (-2 * b_ / U1), force=True)), rho_h, 0
    )
    check(
        "[5] flux P w-bar w' = U_1 e^{W_0} s (r - r_p)^{2 Im omega/U_1} (1 + O(r - r_p))",
        is_zero(flux_lead - U1 * sp.exp(W0) * s_ab),
        f"leading coefficient {sp.simplify(flux_lead)}",
    )

    # [5b] the general ingoing mode, regular in ingoing Eddington-Finkelstein
    # coordinates (v = t + r_*, r): W = e^{-i omega v} (alpha dv + beta dr + gamma_i dx^i)
    # with alpha, beta, gamma regular.  In (t, r) components W_t = W_v,
    # W_r = W_r^EF + W_v / U.  The gauge transformation lambda = W_t/(i omega)
    # removes W_t; the claim is that every remaining component of W is
    # e^{-i omega v} times a regular function (no 1/U), so it vanishes like
    # (r - r_p)^{Im omega/U_1}, while F_ri = d_r W_i - D_i W_r carries a 1/U and
    # grows like (r - r_p)^{Im omega/U_1 - 1}; the integrands of K and V_rad
    # then go like |e^{-i omega v}|^2 / U, integrable for Im omega > 0, and the
    # surface term sqrt(-g) g^rr g^ii W-bar_i F_ri like |e^{-i omega v}|^2 -> 0.
    log("[5b] ingoing modes in the gauge W_t = 0")
    al_, be_, ga_, de_ = (
        sp.Function(nm)(r) for nm in ("alpha", "beta", "gamma", "delta")
    )
    ee = sp.exp(-sp.I * om * (t + rstar))

    def d_r(e_):
        return sp.diff(e_, r).subs(sp.Derivative(rstar, r), 1 / Ub)

    Wt_S, Wr_S, Wi_S = ee * al_, ee * be_ + ee * al_ / Ub, ee * ga_
    lam_ = Wt_S / (sp.I * om)
    Wt_new = sp.simplify(Wt_S + sp.diff(lam_, t))
    Wr_new = sp.simplify(Wr_S + d_r(lam_))
    Wi_new = Wi_S + ee * de_ / (
        sp.I * om
    )  # D_i lambda = e^{-i omega v} delta / (i omega), delta = D_i alpha regular
    Fri_new = d_r(Wi_new) - ee * sp.Function("epsilon_i")(
        r
    )  # D_i W_r = e^{-i omega v} x regular
    check(
        "[5b] gauge W_t = 0: W_t -> 0 and W_r = e^{-i omega v}(beta + alpha'/(i omega)), no 1/U",
        is_zero(Wt_new)
        and is_zero(sp.expand(Wr_new / ee - be_ - sp.diff(al_, r) / (sp.I * om))),
    )
    ga_t = ga_ + de_ / (sp.I * om)
    check(
        "[5b] U F_ri / e^{-i omega v} -> -i omega (gamma_i + D_i alpha/(i omega)) at U = 0: "
        "F_ri ~ (r - r_p)^{Im omega/U_1 - 1}, not vanishing",
        is_zero(
            sp.expand(sp.simplify(Ub * Fri_new / ee).subs(Ub, 0) + sp.I * om * ga_t)
        ),
    )
    rr_ = sp.Symbol("varrho", positive=True)
    bpos = sp.Symbol("b", positive=True)
    conv = sp.integrate(rr_ ** (2 * bpos / U1 - 1), (rr_, 0, 1))
    check(
        "[5b] |e^{-i omega v}|^2 / U ~ (r - r_p)^{2 Im omega/U_1 - 1} is integrable for Im omega > 0",
        is_zero(sp.simplify(conv - U1 / (2 * bpos))),
        f"int_0^1 = {sp.simplify(conv)}",
    )

    # ---------------------------------------------------------------------------
    # [6] positive control: a chemical potential A^3_t = mu(r) breaks the omega^2
    #     structure (a term linear in d/dt appears), so the theorem uses A_t = 0
    # ---------------------------------------------------------------------------
    log("[6] positive control: with A^3_t = mu(r) the n = 0 action is gyroscopic")
    mu = sp.Function("mu", real=True)(r)
    Fmu = field_strength(sector_fields(0, ampT, a3t=mu))
    Lmu = gaussian_x(pair_density(Fmu, PAIRS_E + PAIRS_R + PAIRS_P).coeff(eps, 2))
    Lmus, rep = jet_symbols(Lmu, ampT, True)
    gyro = sp.diff(Lmus, rep["p1"], rep["p2_t"])
    check(
        "[6] A^3_t = mu: L_0 acquires mu (p1 p2_t - p2 p1_t), i.e. (omega - mu)^2",
        not is_zero(gyro) and is_zero(gyro.subs(mu, 0).doit()),
        f"coefficient {sp.simplify(gyro.subs(POS))}",
    )

    # ---------------------------------------------------------------------------
    log("")
    log("SUMMARY")
    log("  L = K(F_ti) - V_rad(F_ri) - V_perp, time only in K, V_perp >= 0 for n >= 1")
    log("  n = 0: (P w')' + B rho w + omega^2 R w = 0, real, omega^2 only")
    log("  => Im omega > 0 forces omega = i Gamma; onset only through omega = 0")
    nfail = sum(1 for v in CHECKS.values() if not v)
    log(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        sys.exit(1)
    log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
