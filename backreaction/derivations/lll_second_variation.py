"""Second variation of the quartic lowest-Landau-level functional about the
uniform one-flux-quantum lattice: the long-wavelength density stiffness and
the Bogoliubov blocks over the magnetic Brillouin zone.

The functional (per unit area, B = 1 so that gamma = k^2/B = k^2) is

    F[psi] = -eps <|psi|^2> + (1/2) sum_k |rho_k|^2 kappa(k^2),
    rho = |psi|^2,  rho_k = <rho e^{-ik.x}>,  kappa(gamma) = K(gamma B_c; beta),

with psi in the lowest Landau level.  At the lattice state psi_L = eta phi_L,
<|phi_L|^2> = 1, the Fourier coefficients are omega_G = e^{-G^2/4} chi_G with
|chi_G| = 1, so F = -eps eta^2 + [kappa_{G=0}/2 + C] eta^4 with

    C = (1/2) sum_{G != 0} e^{-G^2/2} kappa(G^2)

(the shell sum C(tau; beta) of the paper, section 5.1).  kappa_{G=0} is the
strict uniform response, kept as an independent symbol throughout.

Results
-------
[1] Magnetic translations.  In the LLL, e^{-ik.x} projects to e^{-k^2/4} tau(k)
    with tau(k) = exp(-i k.R) and [R_x, R_y] = i sigma (sigma = +-1, the sign
    of the charge).  Derived here by BCH on an explicit representation:
        tau(k) tau(k') = exp(i sigma (k ^ k')/2) tau(k + k').
[2] One flux quantum per cell: b1 ^ b2 = 2 pi, and chi_{m b1 + n b2} = (-1)^{mn}
    is a consistent set of eigenvalues tau(G) phi_L = chi_G phi_L (for either
    sigma); chi_{-G} = chi_G.
[3] Bloch states psi_p = tau(p) phi_L.  From [1] alone:
      (conj(phi_L) psi_p)_k       = e^{-k^2/4} e^{ i sigma (k^p)/2} chi_{k+p}, k = G - p,
      (phi_L conj(psi_{-p}))_k    = e^{-k^2/4} e^{-i sigma (k^p)/2} chi_{k+p}, k = G - p,
      (|psi_p|^2)_{-G} omega_G    = e^{-G^2/2} e^{-i sigma (G^p)}.
[4] The quadratic part of F[psi_L + a psi_p + b psi_{-p}] (eta = 1, eps fixed
    by stationarity) is, identically in a, b,
      Q = A(p) (|a|^2 + |b|^2) + sum_G w_G(p) |a + conj(b) e^{-i sigma G^p}|^2,
      A(p)   = sum_{G != 0} e^{-G^2/2} kappa(G^2) [cos(G^p) - 1],
      w_G(p) = e^{-|G-p|^2/2} kappa(|G-p|^2).
    kappa_{G=0} cancels against eps: the p != 0 spectrum does not depend on it.
    Verified on a symbolic set of shells with symbolic kappa values.
[5] Bogoliubov eigenvalues in the (a, conj b) block:
      lambda_pm(p) = A(p) + S_0(p) +- |S_2(p)|,
      S_0 = sum_G w_G,  S_2 = sum_G w_G e^{-i sigma G^p}.
    As p -> 0: lambda_- -> 0 (the Goldstone/phase mode) and
      lambda_+ -> 2 S_0(0+) = 2 [kappa(0+) + 2C] = 2 [K_0 + 2C],
    with K_0 = lim_{s->0} K(s).  At p = 0 exactly the amplitude eigenvalue is
    2 [kappa_{G=0} + 2C] instead.
[6] The slow density modulation of the lattice, rho -> |phi_L|^2 (1 + delta cos q.x),
    raises the quartic energy by (delta^2/4) [kappa(q^2) + 2C] + O(q^2): the
    long-wavelength density stiffness is K_0 + 2C, not K_0.  The envelope
    functional int [-eps eta^2 + (1/2) eta^2 K eta^2] keeps only the G = 0 band
    of the modulated density and misses the 2C carried by the satellites G +- q.

[7] Every step above uses only b1 ^ b2 = 2 pi and chi_G = (-1)^{mn}, so it
    holds for every one-flux-quantum Bravais lattice tau = tau1 + i tau2, not
    only the triangular one.  Made explicit:
    [7a] a1 = l (1, 0), a2 = l (tau1, tau2), l^2 = 2 pi/tau2: cell area 2 pi
         and b1 ^ b2 = 2 pi identically in tau;
    [7b] |m b1 + n b2|^2 = (2 pi/tau2) |n - m tau|^2, the multiset
         gamma_mn = (2 pi/tau2) |m + n tau|^2 of the moduli scan;
    [7c] on an explicit shell set of the tau lattice, lambda_+(0+) =
         2 [K_0 + 2 C(tau)] and the slow modulation costs (delta^2/4)
         [K_0 + 2 C(tau)]: the long-wavelength density stiffness of lattice
         tau is K_0 + 2 C(tau), direction independent;
    [7d] gamma_{m,n}(tau + 1) = gamma_{m+n,n}(tau), gamma_{m,n}(-1/tau) =
         gamma_{-n,m}(tau): C is modular invariant;
    [7e] at tau = i and tau = rho = e^{i pi/3} the gradient of every shell
         orbit of the stabiliser vanishes, so dC = 0 there for EVERY kernel
         (the only symmetry-forced stationary points; any other critical
         point would be accidental and is excluded numerically);
    [7f] the Hessian of every orbit is proportional to the identity at rho
         (C is isotropic there) and diagonal in (tau1, tau2) at i, with
         unequal entries in general: whether i is a minimum or a saddle is a
         property of the kernel;
    [7g] stripe edge: by Poisson summation over the n = 0 row,
             K_0/2 + C(i tau2) = (1/2) sqrt(tau2) I + R,
             I = int dx e^{-pi x^2} kappa(2 pi x^2),
         with K_0 cancelled exactly; verified as an exact theta-function
         identity, remainder included, for kappa = e^{-c gamma}.  So the
         density stiffness of a strongly elongated lattice has the sign of
         I, not of K_0/2 + C_tri.

Hence the uniform lattice is stable against long-wavelength density
modulation iff K_0/2 + C > 0, and against a uniform amplitude change iff
kappa_{G=0}/2 + C > 0 (the second-order condition), for every tau.

Run:  uv run python -m backreaction.derivations.lll_second_variation      (~3 s)
"""

import functools
import sys

import sympy as sp

print = functools.partial(print, flush=True)  # noqa: A001

CHECKS = {}


def check(name, ok, detail=""):
    CHECKS[name] = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")


I = sp.I


def wedge(k, q):
    return k[0] * q[1] - k[1] * q[0]


def main():
    # ---------------------------------------------------------------------------
    # [1] tau(k) tau(k') by BCH on R_x = X, R_y = -i sigma d/dX  ([R_x, R_y] = i sigma)
    # ---------------------------------------------------------------------------
    print("[1] magnetic translation algebra", flush=True)
    X = sp.Symbol("X", real=True)
    f = sp.Function("f")
    kx, ky, qx, qy = sp.symbols("k_x k_y q_x q_y", real=True)
    SIGMAS = (1, -1)

    def tau(k, expr, s):
        """tau(k) = exp(-i(k_x X + k_y R_y)); for central [A, B]:
        e^{A+B} = e^A e^B e^{-[A,B]/2}, A = -i k_x X, B = -s k_y d/dX,
        [A, B] = (-i k_x)(-i k_y)(i s) = -i s k_x k_y."""
        return (
            sp.exp(-I * k[0] * X)
            * expr.subs(X, X - s * k[1])
            * sp.exp(I * s * k[0] * k[1] / 2)
        )

    for s in SIGMAS:
        lhs = tau((kx, ky), tau((-kx, -ky), f(X), s), s)
        check(
            f"[1a] tau(k) tau(-k) = 1  (sigma = {s:+d})", sp.simplify(lhs - f(X)) == 0
        )
        ratio = sp.simplify(
            tau((kx, ky), tau((qx, qy), f(X), s), s) / tau((kx + qx, ky + qy), f(X), s)
        )
        check(
            f"[1b] tau(k) tau(q) = exp(-i sigma (k^q)/2) tau(k+q)  (sigma = {s:+d})",
            sp.simplify(ratio - sp.exp(-I * s * wedge((kx, ky), (qx, qy)) / 2)) == 0,
            f"ratio = {ratio}",
        )
    # Below, s denotes the sign in tau(k) tau(q) = exp(i s (k^q)/2) tau(k+q), i.e. s = -sigma.
    # Every later step is carried out for both signs, so nothing depends on the charge.

    # ---------------------------------------------------------------------------
    # [2] one flux quantum: chi_G = (-1)^{mn} is consistent
    # ---------------------------------------------------------------------------
    print(
        "[2] reciprocal-lattice eigenvalues of the one-flux-quantum lattice state",
        flush=True,
    )
    m, n, m2, n2 = sp.symbols("m n m2 n2", integer=True)
    # b1 ^ b2 = (2 pi)^2 / (cell area 2 pi) = 2 pi, so G ^ G' = 2 pi (m n2 - n m2) and
    # exp(i sigma G^G'/2) = (-1)^{sigma (m n2 - n m2)}.  The composition law requires
    # mn + m2 n2 - (m+m2)(n+n2) - sigma (m n2 - n m2) to be even:
    for s in SIGMAS:
        E = sp.expand(m * n + m2 * n2 - (m + m2) * (n + n2) - s * (m * n2 - n * m2))
        half = sp.Poly(sp.expand(E / 2), m, n, m2, n2)
        check(
            f"[2a] chi = (-1)^(mn) obeys the composition law (sigma = {s:+d})",
            all(c.is_integer for c in half.coeffs()),
            f"exponent = {E}",
        )
    check("[2b] chi_{-G} = chi_G", sp.expand((-m) * (-n) - m * n) == 0)

    # ---------------------------------------------------------------------------
    # [3] matrix elements from the algebra (abstract: <phi| tau(G) |phi> = chi_G)
    # ---------------------------------------------------------------------------
    print("[3] Fourier coefficients of products of Bloch states", flush=True)

    class Word:
        """phase * tau(total momentum) acting on phi_L."""

        def __init__(self, phase, k, s):
            self.phase, self.k, self.s = phase, k, s

        def left(self, q):  # tau(q) * self
            return Word(
                self.phase * sp.exp(I * self.s * wedge(q, self.k) / 2),
                (sp.expand(q[0] + self.k[0]), sp.expand(q[1] + self.k[1])),
                self.s,
            )

    Gx, Gy, px, py = sp.symbols("G_x G_y p_x p_y", real=True)
    G, p = (Gx, Gy), (px, py)
    k = (Gx - px, Gy - py)  # k = G - p
    for s in SIGMAS:
        # (conj(phi) psi_p)_k = <phi| e^{-ik.x} |psi_p> = e^{-k^2/4} <phi| tau(k) tau(p) |phi>
        w1 = Word(1, p, s).left(k)
        # (phi conj(psi_{-p}))_k = conj( (conj(phi) psi_{-p})_{-k} ),  -k - p = -G
        w2 = Word(1, (-px, -py), s).left((-k[0], -k[1]))
        # (|psi_p|^2)_{-G} e^{G^2/4} = <phi| tau(-p) tau(-G) tau(p) |phi>
        w3 = Word(1, p, s).left((-Gx, -Gy)).left((-px, -py))
        ok = w1.k == (Gx, Gy) and w2.k == (-Gx, -Gy) and w3.k == (-Gx, -Gy)
        ok &= sp.simplify(w1.phase - sp.exp(I * s * wedge(G, p) / 2)) == 0
        ok &= (
            sp.simplify(sp.conjugate(w2.phase) - sp.exp(-I * s * wedge(G, p) / 2)) == 0
        )
        ok &= sp.simplify(w3.phase - sp.exp(-I * s * wedge(G, p))) == 0
        check(
            f"[3] phases e^(+i s G^p/2), e^(-i s G^p/2), e^(-i s G^p)  (sigma = {s:+d})",
            ok,
        )

    # ---------------------------------------------------------------------------
    # [4] the quadratic form on a symbolic set of shells
    # ---------------------------------------------------------------------------
    print("[4] quadratic part of F on a symbolic shell set", flush=True)
    a_r, a_i, b_r, b_i = sp.symbols("a_r a_i b_r b_i", real=True)
    a = a_r + I * a_i
    b = b_r + I * b_i
    nG = 3  # G = 0 plus two generic harmonics and their negatives
    Gs = [(0, 0)]
    for j in range(1, nG):
        Gs.append((sp.Symbol(f"g{j}x", real=True), sp.Symbol(f"g{j}y", real=True)))
    Gs += [(-g[0], -g[1]) for g in Gs[1:]]
    nh = nG - 1
    kG = [sp.Symbol("kappa_G0", real=True)] + [
        sp.Symbol(f"kG{(j - 1) % nh + 1}", real=True) for j in range(1, len(Gs))
    ]
    kGp = [sp.Symbol(f"kGp{j}", real=True) for j in range(len(Gs))]  # kappa(|G - p|^2)
    eGp = [
        sp.Symbol(f"eGp{j}", positive=True) for j in range(len(Gs))
    ]  # e^{-|G-p|^2/4}
    eG = [sp.Integer(1)] + [
        sp.Symbol(f"eG{(j - 1) % nh + 1}", positive=True) for j in range(1, len(Gs))
    ]  # e^{-G^2/4}

    def absq(z):
        return sp.expand(z * sp.conjugate(z))

    norm2 = absq(a) + absq(b)
    eps = sum(
        eG[j] ** 2 * kG[j] for j in range(len(Gs))
    )  # stationarity at eta = 1 (chi^2 = 1)
    Csym = sp.Rational(1, 2) * sum(eG[j] ** 2 * kG[j] for j in range(1, len(Gs)))
    A_by_s, S0, S2_by_s = {}, sum(eGp[j] ** 2 * kGp[j] for j in range(len(Gs))), {}
    for s in SIGMAS:
        T1 = -eps * norm2
        T2 = sum(
            kG[j] * eG[j] ** 2 * sp.exp(-I * s * wedge(g, pp)) * absq(amp)
            for j, g in enumerate(Gs)
            for pp, amp in ((p, a), ((-px, -py), b))
        )
        T3 = sum(
            kGp[j]
            * absq(
                eGp[j]
                * (
                    a * sp.exp(I * s * wedge(g, p) / 2)
                    + sp.conjugate(b) * sp.exp(-I * s * wedge(g, p) / 2)
                )
            )
            for j, g in enumerate(Gs)
        )
        Q = T1 + T2 + T3
        A = sum(
            eG[j] ** 2 * kG[j] * (sp.cos(wedge(g, p)) - 1)
            for j, g in enumerate(Gs)
            if j
        )
        Qc = A * norm2 + sum(
            kGp[j]
            * eGp[j] ** 2
            * absq(a + sp.conjugate(b) * sp.exp(-I * s * wedge(g, p)))
            for j, g in enumerate(Gs)
        )
        diff = sp.expand((Q - Qc).rewrite(sp.exp))
        diff = sp.simplify(sp.powsimp(diff, combine="exp"))
        check(
            f"[4a] Q = A(|a|^2+|b|^2) + sum_G w_G |a + conj(b) e^(-i s G^p)|^2 (sigma = {s:+d})",
            diff == 0,
        )
        check(
            f"[4b] kappa_(G=0) absent from Q (sigma = {s:+d})",
            kG[0] not in sp.expand(Q.rewrite(sp.exp)).free_symbols,
        )
        A_by_s[s] = A
        S2_by_s[s] = sum(
            kGp[j] * eGp[j] ** 2 * sp.exp(-I * s * wedge(g, p))
            for j, g in enumerate(Gs)
        )

    # ---------------------------------------------------------------------------
    # [5] eigenvalues and the p -> 0 limits
    # ---------------------------------------------------------------------------
    print("[5] Bogoliubov block and its p -> 0 limits", flush=True)
    Asym, S0s, S2r, S2i = sp.symbols("A S_0 S2r S2i", real=True)
    H = sp.Matrix([[Asym + S0s, S2r + I * S2i], [S2r - I * S2i, Asym + S0s]])
    ev = {sp.simplify(e) for e in H.eigenvals()}
    r = sp.sqrt(S2r**2 + S2i**2)
    check(
        "[5a] lambda_pm = A + S_0 +- |S_2|",
        ev == {sp.simplify(Asym + S0s + r), sp.simplify(Asym + S0s - r)},
    )
    # p -> 0: kappa(|G-p|^2) -> kappa(G^2) for G != 0 and -> K_0 = kappa(0+) for G = 0;
    # e^{-|G-p|^2/4} -> e^{-G^2/4}
    K0 = sp.Symbol("K_0", real=True)
    lim = {kGp[j]: (K0 if j == 0 else kG[j]) for j in range(len(Gs))}
    lim.update({eGp[j]: eG[j] for j in range(len(Gs))})
    for s in SIGMAS:
        A0 = A_by_s[s].subs({px: 0, py: 0})
        S20 = S2_by_s[s].subs({px: 0, py: 0}).subs(lim)
        S00 = S0.subs(lim)
        check(
            f"[5b] A -> 0, S_2 -> S_0 as p -> 0 (sigma = {s:+d})",
            sp.simplify(A0) == 0 and sp.simplify(S20 - S00) == 0,
        )
    check(
        "[5c] lambda_-(0+) = 0, lambda_+(0+) = 2 (K_0 + 2C)",
        sp.simplify(2 * S0.subs(lim) - 2 * (K0 + 2 * Csym)) == 0,
    )
    # at p = 0 exactly the G = 0 band carries kappa_{G=0}: delta psi = c phi_L,
    # delta rho = 2 Re(c) |phi_L|^2, Q = -eps |c|^2 + eps |c|^2 + (1/2) sum_G kappa_G |omega_G|^2 4 Re(c)^2
    cr = sp.Symbol("c_r", real=True)
    Qgamma = (
        sp.Rational(1, 2) * sum(kG[j] * eG[j] ** 2 for j in range(len(Gs))) * 4 * cr**2
    )
    check(
        "[5d] p = 0 amplitude eigenvalue = 2 (kappa_(G=0) + 2C)",
        sp.simplify(Qgamma / cr**2 - 2 * (kG[0] + 2 * Csym)) == 0,
    )

    # ---------------------------------------------------------------------------
    # [6] the slow density modulation, directly in Fourier space
    # ---------------------------------------------------------------------------
    print(
        "[6] slow density modulation rho -> |phi_L|^2 (1 + delta cos q.x)", flush=True
    )
    delta = sp.Symbol("delta", real=True)
    # rho_k = omega_G at k = G and (delta/2) omega_G at k = G +- q.  F4 = (1/2) sum_k kappa |rho_k|^2.
    # kappa(|G +- q|^2) -> kappa(G^2) for G != 0 and -> K_0 for G = 0 as q -> 0.
    kq = [K0] + kG[1:]
    F4_0 = sp.Rational(1, 2) * sum(kG[j] * eG[j] ** 2 for j in range(len(Gs)))
    F4_mod = F4_0 + sp.Rational(1, 2) * sum(
        2 * kq[j] * (delta / 2) ** 2 * eG[j] ** 2 for j in range(len(Gs))
    )
    dF = sp.expand(F4_mod - F4_0)
    check(
        "[6a] Delta F4 -> (delta^2/4) (K_0 + 2C) as q -> 0",
        sp.simplify(dF - delta**2 / 4 * (K0 + 2 * Csym)) == 0,
    )
    env = sp.Rational(1, 2) * 2 * (delta / 2) ** 2 * K0  # only the G = 0 band
    check(
        "[6b] the envelope functional gives (delta^2/4) K_0: it drops exactly the satellites, (delta^2/2) C",
        sp.simplify(dF - env - delta**2 / 2 * Csym) == 0,
    )
    # <|psi|^2> = rho_0 is unchanged by the modulation, so -eps<|psi|^2> contributes nothing at O(delta^2)

    # ---------------------------------------------------------------------------
    # [7] general one-flux-quantum Bravais lattice tau = tau1 + i tau2
    # ---------------------------------------------------------------------------
    print("[7] general Bravais lattice tau", flush=True)
    t1 = sp.Symbol("tau1", real=True)
    t2 = sp.Symbol("tau2", positive=True)
    ell = sp.sqrt(2 * sp.pi / t2)
    a1v = sp.Matrix([ell, 0])
    a2v = sp.Matrix([ell * t1, ell * t2])
    area = sp.simplify(a1v[0] * a2v[1] - a1v[1] * a2v[0])
    # reciprocal vectors: b_i . a_j = 2 pi delta_ij
    Amat = sp.Matrix([[a1v[0], a1v[1]], [a2v[0], a2v[1]]])
    Bmat = sp.simplify(2 * sp.pi * Amat.inv())  # columns b_j: A B = 2 pi 1
    b1v, b2v = Bmat[:, 0], Bmat[:, 1]
    dual_ok = all(
        sp.simplify((Bmat[:, i].T * v)[0] - (2 * sp.pi if i == j else 0)) == 0
        for i in range(2)
        for j, v in enumerate((a1v, a2v))
    )
    check(
        "[7a] cell area 2 pi and b1 ^ b2 = 2 pi for every tau",
        sp.simplify(area - 2 * sp.pi) == 0
        and dual_ok
        and sp.simplify(wedge(tuple(b1v), tuple(b2v)) - 2 * sp.pi) == 0,
    )
    Gmn = lambda mm, nn: mm * b1v + nn * b2v  # noqa: E731
    mi, ni = sp.symbols("m n", integer=True)
    g_mn = sp.expand((Gmn(mi, ni).T * Gmn(mi, ni))[0])

    def gam(mm, nn, tt1=t1, tt2=t2):
        """(2 pi/tau2) |m + n tau|^2."""
        return 2 * sp.pi / tt2 * ((mm + nn * tt1) ** 2 + (nn * tt2) ** 2)

    check(
        "[7b] |m b1 + n b2|^2 = (2 pi/tau2) |n - m tau|^2",
        sp.simplify(g_mn - gam(ni, -mi)) == 0,
    )

    # [7c] explicit tau-lattice shells: stiffness K_0 + 2 C(tau)
    shells = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)]
    Gt = (
        [(0, 0)]
        + [tuple(Gmn(mm, nn)) for mm, nn in shells]
        + [tuple(-Gmn(mm, nn)) for mm, nn in shells]
    )
    kt = [sp.Symbol(f"kt{j}", real=True) for j in range(len(shells))]  # kappa(gamma_mn)
    et = [
        sp.exp(-gam(mm, nn) / 2) for mm, nn in shells
    ]  # e^{-gamma_mn/2} = |omega_G|^2
    w_t = [K0] + kt + kt  # p -> 0 weights kappa(|G - p|^2) -> kappa(G^2), K_0 at G = 0
    e_t = [sp.Integer(1)] + et + et
    C_tau = sp.Rational(1, 2) * sum(e_t[j] * w_t[j] for j in range(1, len(Gt)))
    S0_t = sum(e_t[j] * w_t[j] for j in range(len(Gt)))
    # S_2(p -> 0) along an arbitrary direction: e^{-i s G^p} -> 1, A(p) -> 0
    th = sp.Symbol("theta", real=True)
    eps_p = sp.Symbol("epsilon_p", positive=True)
    pp = (eps_p * sp.cos(th), eps_p * sp.sin(th))
    ok7c = True
    for s_ in SIGMAS:
        S2_t = sum(
            e_t[j] * w_t[j] * sp.exp(-I * s_ * wedge(g, pp)) for j, g in enumerate(Gt)
        )
        A_t = sum(
            e_t[j] * w_t[j] * (sp.cos(wedge(g, pp)) - 1) for j, g in enumerate(Gt) if j
        )
        # weights already at their p -> 0 values; the phases are continuous in p
        A0 = sp.simplify(A_t.subs(eps_p, 0))
        S20 = sp.simplify(S2_t.subs(eps_p, 0))
        lam_p0 = A0 + S0_t + S20  # |S_2| = S_2 at p = 0 (real, = S_0)
        ok7c &= A0 == 0 and sp.simplify(S20 - S0_t) == 0
        ok7c &= sp.simplify(lam_p0 - 2 * (K0 + 2 * C_tau)) == 0
    dF_t = sp.Rational(1, 2) * sum(
        2 * w_t[j] * (delta / 2) ** 2 * e_t[j] for j in range(len(Gt))
    )
    ok7c &= sp.simplify(dF_t - delta**2 / 4 * (K0 + 2 * C_tau)) == 0
    check(
        "[7c] tau lattice: lambda_+(0+) = 2(K_0 + 2C(tau)) in every direction, "
        "slow modulation (delta^2/4)(K_0 + 2C(tau))",
        ok7c,
    )

    # [7d] modular invariance of the multiset
    check(
        "[7d] gamma_(m,n)(tau+1) = gamma_(m+n,n)(tau), gamma_(m,n)(-1/tau) = gamma_(-n,m)(tau)",
        sp.simplify(gam(mi, ni, t1 + 1, t2) - gam(mi + ni, ni)) == 0
        and sp.simplify(
            gam(mi, ni, sp.re(-1 / (t1 + I * t2)), sp.im(-1 / (t1 + I * t2)))
            - gam(-ni, mi)
        )
        == 0,
    )

    # [7e] stabiliser orbits: gradients vanish at i and rho for any (m, n)
    def grad(mm, nn):
        g = gam(mm, nn)
        return sp.Matrix([sp.diff(g, t1), sp.diff(g, t2)])

    def hess(mm, nn):
        g = gam(mm, nn)
        return sp.hessian(g, (t1, t2))

    at_i = {t1: 0, t2: 1}
    at_rho = {t1: sp.Rational(1, 2), t2: sp.sqrt(3) / 2}
    orb_i = [(mi, ni), (-ni, mi)]  # S fixes i: gamma_(m,n)(i) = gamma_(-n,m)(i)
    orb_rho = [(mi, ni), (-ni, mi + ni), (-mi - ni, mi)]  # tau -> 1 - 1/tau fixes rho
    same_i = all(sp.simplify((gam(*o) - gam(mi, ni)).subs(at_i)) == 0 for o in orb_i)
    same_rho = all(
        sp.simplify((gam(*o) - gam(mi, ni)).subs(at_rho)) == 0 for o in orb_rho
    )
    gi = sp.simplify(sum((grad(*o).subs(at_i) for o in orb_i), sp.zeros(2, 1)))
    grho = sp.simplify(sum((grad(*o).subs(at_rho) for o in orb_rho), sp.zeros(2, 1)))
    check(
        "[7e] each stabiliser orbit has one gamma and zero total gradient at i and rho",
        same_i and same_rho and gi == sp.zeros(2, 1) and grho == sp.zeros(2, 1),
        "so dC/dtau = sum_orbits f'(gamma) sum_orbit dgamma = 0 at i and rho for any kernel",
    )
    # a generic point is not stationary for a generic orbit: dC != 0 off the fixed points
    gen = {t1: sp.Rational(1, 5), t2: sp.Rational(13, 10)}
    check(
        "[7e'] the orbit gradients do not vanish at a generic tau (no symmetry-forced stationarity)",
        sp.simplify(sum((grad(*o).subs(gen) for o in orb_i), sp.zeros(2, 1)))
        != sp.zeros(2, 1),
    )

    # [7f] Hessian structure at the fixed points: orbit sums of grad grad^T and of hess
    GGr = sp.simplify(
        sum((grad(*o) * grad(*o).T for o in orb_rho), sp.zeros(2, 2)).subs(at_rho)
    )
    HHr = sp.simplify(sum((hess(*o) for o in orb_rho), sp.zeros(2, 2)).subs(at_rho))
    iso = all(
        sp.simplify(M[0, 1]) == 0 and sp.simplify(M[0, 0] - M[1, 1]) == 0
        for M in (GGr, HHr)
    )
    # at i: the reflection tau -> -conj(tau) maps (m, n) -> (m, -n) and fixes i
    orb_i_full = orb_i + [(mi, -ni), (ni, mi)]
    GGi = sp.simplify(
        sum((grad(*o) * grad(*o).T for o in orb_i_full), sp.zeros(2, 2)).subs(at_i)
    )
    HHi = sp.simplify(sum((hess(*o) for o in orb_i_full), sp.zeros(2, 2)).subs(at_i))
    diag_i = sp.simplify(GGi[0, 1]) == 0 and sp.simplify(HHi[0, 1]) == 0
    uneq_i = sp.simplify(GGi[0, 0] - GGi[1, 1]) != 0
    check(
        "[7f] orbit Hessians: proportional to the identity at rho, diagonal with unequal "
        "entries at i",
        iso and diag_i and uneq_i,
        f"at i, sum grad grad^T = diag({sp.factor(GGi[0, 0])}, {sp.factor(GGi[1, 1])})",
    )

    # [7g] the stripe edge: Poisson summation, exact for kappa = e^{-c gamma}
    import mpmath as mp  # noqa: E402

    c_, xx = sp.symbols("c x", positive=True)
    I_c = sp.integrate(
        sp.exp(-sp.pi * xx**2) * sp.exp(-c_ * 2 * sp.pi * xx**2), (xx, -sp.oo, sp.oo)
    )
    ok7g = sp.simplify(I_c - 1 / sp.sqrt(1 + 2 * c_)) == 0
    # the n = 0 row of C(i tau2) for this kernel: (1/2) sum_{m != 0} e^{-(1+2c) pi m^2/tau2}
    mp.mp.dps = 50
    worst7g = mp.mpf(0)
    for cv, t2v in (
        (mp.mpf("0.3"), mp.mpf(4)),
        (mp.mpf("1.7"), mp.mpf(25)),
        (mp.mpf("0.05"), mp.mpf(60)),
    ):
        a_ = (1 + 2 * cv) / t2v
        row = mp.nsum(
            lambda m, a_=a_: mp.exp(-mp.pi * a_ * m * m), [1, mp.inf]
        )  # half of sum_{m != 0}
        K0v = mp.mpf(1)  # kappa(0)
        Iv = 1 / mp.sqrt(1 + 2 * cv)
        # exact Jacobi identity: sum_m e^{-pi a m^2} = a^{-1/2} sum_k e^{-pi k^2/a}
        R = (
            mp.sqrt(t2v)
            * Iv
            * mp.nsum(lambda k, a_=a_: mp.exp(-mp.pi * k * k / a_), [1, mp.inf])
        )
        lhs = K0v / 2 + row  # K_0/2 + (n = 0 row of C)
        rhs = mp.sqrt(t2v) * Iv / 2 + R
        worst7g = max(worst7g, abs(lhs - rhs))
    check(
        "[7g] K_0/2 + C(i tau2)|_(n=0 row) = (1/2) sqrt(tau2) I + R exactly (theta identity), "
        "K_0 cancelled",
        ok7g and worst7g < mp.mpf(10) ** -45,
        f"I = 1/sqrt(1+2c) for kappa = e^(-c gamma); worst |lhs - rhs| = {mp.nstr(worst7g, 3)}",
    )

    nfail = sum(1 for v in CHECKS.values() if not v)
    print(f"{len(CHECKS) - nfail}/{len(CHECKS)} checks passed")
    if nfail:
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
