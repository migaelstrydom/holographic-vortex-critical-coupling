"""Gauge independence of the finite-coupling quartic kernel K(s; beta).

The kernel is computed in the algebraic gauge H_tt = H_xx = 0.  Under an
even-sector diffeomorphism xi = (xi^x, xi^r) e^{iGx} the response shifts by
h -> h + L_xi g and a_y -> a_y + B xi^x, and the shift of the pairing
Re X + (1/2) Re P is the total derivative of

    Q_xi = e^{Z-2V} (B w0^2 + U e^{2V} w0'^2) Re xi^r

(derived symbolically in derivations/paper_response_display.py [6], where the
xi^x part is also shown to cancel only at the weight 1/2).  So K is the same in
every gauge reached by a xi that keeps the horizon in place (xi^r(r_p) = 0)
and grows slower than r^3 at the boundary.  This script checks that on actual
solutions, at beta = 3 and 45 and two harmonics s = gamma B_c, gamma = 1, 8:

  [1] radial gauge: xi^r and xi^x chosen to remove H_rr and H_xr,
          (xi^r / sqrt U)' = -H_rr / (2 sqrt U),   xi^r(r_p) = 0,
          psi' = -H_xr - G xi^r e^{-2V}/U,  xi^x = i psi,  psi(oo) = 0,
      so H_tt and H_xx become non-zero and the boundary metric picks up a Weyl
      factor; K re-evaluated with the bare stress paired against all metric
      components equals the algebraic-gauge K (to 6e-10, as paper app. C.3
      quotes);
  [2] a generic smooth gauge (analytic xi profiles with xi^r(r_p) = 0): same,
      to 7e-9;
  [3] control: with a weight w = 0.4 on P the radial-gauge value moves by
      O(0.1 |w - 1/2| ...) -- the invariance is specific to w = 1/2;
  [4] the 7x7 coupled matrix that solves for (H_rr, H_xr, H_rr', H_xr',
      H_yy'', H_zz'', a_y'') is non-singular at every node of both branes;
  [5] on the solutions H_rr(r_p) = 0 (the temperature is not modulated) and
      the boundary fall-offs are H_yy, H_zz, H_rr ~ r^-4, H_xr ~ r^-5,
      a_y ~ r^-2.

This is not an independent solve of the radial-gauge boundary value problem:
the radial-gauge configuration is the gauge transform of the algebraic-gauge
solution, which solves the linearised equations by covariance.  What it tests
is that the kernel formula, with every metric component and both ends of the
radial integral included, does not depend on the gauge.

Run:  uv run python -m backreaction.numerics.dk_kernel_gauge_check   (~5 s; the cached
      zero modes fix N = 64)
"""

import sys
import time
import warnings

import numpy as np
import sympy as sp
from scipy.integrate import IntegrationWarning, quad

from backreaction.numerics import dk_kernel as dkk
from backreaction.systems import dk_response as dkr

T0 = time.time()
CHECKS = {}

# paper app. C.3 quotes the kernel unchanged to 6e-10 (radial gauge) and
# 7e-9 (generic gauge); measured 5.7e-10 and 7.0e-9.  Asserted at the quoted
# digit.
TOL_RADIAL = 6.5e-10
TOL_GENERIC = 7.5e-9


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(tag, ok, detail=""):
    CHECKS[tag] = bool(ok)
    log(f"{tag}: {'PASS' if ok else 'FAIL'} {detail}")


# ---------------------------------------------------------------------------
# H-normalised shifts under xi^x = i psi e^{iGx}, xi^r = rho e^{iGx}
# ---------------------------------------------------------------------------
def _shift_functions():
    r, x = sp.symbols("r x", real=True)
    G = sp.Symbol("G", positive=True)
    co = [sp.Symbol("t"), x, sp.Symbol("y"), sp.Symbol("z"), r]
    N = 5
    U, V, Z, rho, psi = (sp.Function(n)(r) for n in ("U", "V", "Z", "rho", "psi"))
    g = sp.diag(-U, sp.exp(2 * V), sp.exp(2 * V), sp.exp(2 * Z), 1 / U)
    gi = g.inv()
    Gam = [
        [
            [
                sum(
                    gi[l, k]
                    * (
                        sp.diff(g[k, m], co[n])
                        + sp.diff(g[k, n], co[m])
                        - sp.diff(g[m, n], co[k])
                    )
                    for k in range(N)
                )
                / 2
                for n in range(N)
            ]
            for m in range(N)
        ]
        for l in range(N)
    ]
    E = sp.exp(sp.I * G * x)
    xl = [0, sp.exp(2 * V) * sp.I * psi * E, 0, 0, rho / U * E]  # lowered
    L = [
        [
            sp.simplify(
                (
                    sp.diff(xl[n], co[m])
                    + sp.diff(xl[m], co[n])
                    - 2 * sum(Gam[l][m][n] * xl[l] for l in range(N))
                )
                / E
            )
            for n in range(N)
        ]
        for m in range(N)
    ]
    shifts = {
        "tt": -L[0][0] / U,
        "xx": L[1][1] * sp.exp(-2 * V),
        "yy": L[2][2] * sp.exp(-2 * V),
        "zz": L[3][3] * sp.exp(-2 * Z),
        "rr": U * L[4][4],
        "xr": L[1][4] / (sp.I * sp.exp(2 * V)),
    }
    Us, Ups, Vs, Vps, Zs, Zps, R, Rp, P, Pp = sp.symbols("U Up V Vp Z Zp R Rp P Pp")
    rep = {
        sp.Derivative(U, r): Ups,
        sp.Derivative(V, r): Vps,
        sp.Derivative(Z, r): Zps,
        sp.Derivative(rho, r): Rp,
        sp.Derivative(psi, r): Pp,
    }
    rep0 = {U: Us, V: Vs, Z: Zs, rho: R, psi: P}
    out = {}
    for k, e in shifts.items():
        e = sp.simplify(sp.expand(e).subs(rep).subs(rep0))
        out[k] = sp.lambdify((Us, Ups, Vs, Vps, Zs, Zps, R, Rp, P, Pp, G), e, "numpy")
        if k == "rr":
            assert sp.simplify(e - (2 * Rp - Ups / Us * R)) == 0, e
        if k == "xr":
            assert sp.simplify(e - (Pp + G * R * sp.exp(-2 * Vs) / Us)) == 0, e
    return out


SHIFT = _shift_functions()


def kernel_any_gauge(br, s, w0, w0p, H, w=0.5):
    """K = C4 - Re X - w Re P with the BARE stress paired against all six
    metric components (S_xr = 0 for the bare stress).  H: dict of nodal
    arrays tt, xx, yy, zz, rr, xr (H-normalised) and bh."""
    f = br.bg.fields_sigma(br.sig[:-1])
    Uu, V, Z = (np.real(f[k]) for k in ("U", "V", "W"))
    B = br.Bc
    e2V, e2Z = np.exp(2 * V), np.exp(2 * Z)
    a, ap = w0[:-1], w0p[:-1]
    # S^{MM}... g^MM g^MM S_MM conj(h_MM), h_tt = -U Htt etc.
    Stt_over_U = np.exp(-4 * V) * (Uu * e2V * ap**2 - B * a**2)  # S_tt / U
    Sxx = -B * a**2 / e2V
    Szz = np.exp(-4 * V) * e2Z * (B * a**2 - Uu * e2V * ap**2)
    USrr = np.exp(-4 * V) * (B * a**2 + Uu * e2V * ap**2)
    sq = np.exp(2 * V + Z)
    cj = {k: np.conj(v[:-1]) for k, v in H.items()}
    dens = sq * (
        # g^tt g^tt S_tt h_tt = U^-2 S_tt (-U Htt) = -(S_tt/U) Htt
        -Stt_over_U * cj["tt"]
        + Sxx / e2V * (cj["xx"] + cj["yy"])
        + Szz / e2Z * cj["zz"]
        + USrr * cj["rr"]
    )
    full = np.zeros(br.N + 1, dtype=complex)
    full[:-1] = dens
    P = br.integ_dr(full).real
    X = br.integ_dr(br.P1 * br.sig * H["bh"] * w0**2).real
    C4 = br.integ_dr(br.P1 * br.sig * w0**4).real
    return C4 - X - w * P


def _shift_at(br, sg, k, G, prof):
    """Shift of component k at sigma = sg; prof = nodal (rho, rho', psi, psi')
    interpolated by barycentric Chebyshev (real profiles)."""
    f = br.bg.fields_sigma([sg])
    args = [float(np.real(f[q][0])) for q in ("U", "Up", "V", "Vp", "W", "Wp")]
    vals = [float(br.bg._bary(np.real(p), [sg])[0]) for p in prof]
    return complex(SHIFT[k](*args, *vals, G))


def transformed(br, s, sol, rho, rhop, psi, psip):
    G = np.sqrt(s)
    f = br.bg.fields_sigma(br.sig[1:-1])
    args = [np.real(f[k]) for k in ("U", "Up", "V", "Vp", "W", "Wp")]
    Hyy, Hzz, ay = sol["H"]
    H0 = dict(
        tt=np.zeros(br.N + 1, complex),
        xx=np.zeros(br.N + 1, complex),
        yy=Hyy,
        zz=Hzz,
        rr=sol["Hrr"],
        xr=sol["Hxr"],
    )
    prof = (rho, rhop, psi, psip)
    eps = 1e-6
    Hn = {}
    for k, v in H0.items():
        sh = np.zeros(br.N + 1, dtype=complex)
        sh[1:-1] = SHIFT[k](*args, rho[1:-1], rhop[1:-1], psi[1:-1], psip[1:-1], G)
        # horizon node: Richardson limit from inside (the shifts carry 1/U)
        sh[0] = 2 * _shift_at(br, 1 - eps, k, G, prof) - _shift_at(
            br, 1 - 2 * eps, k, G, prof
        )
        Hn[k] = v + sh  # boundary node is masked by integ_dr
    bh = 1j * G * ay
    Hn["bh"] = bh + 1j * G * br.Bc * (1j * psi)  # delta a_y = B xi^x, xi^x = i psi
    return Hn


def radial_gauge_xi(br, s, sol):
    """xi^r = rho, xi^x = i psi removing H_rr and H_xr (algebraic -> radial)."""
    G = np.sqrt(s)
    sig = br.sig
    bg = br.bg
    Hrr = np.real(sol["Hrr"])
    Hxr = np.real(sol["Hxr"])

    def U_of(sg):
        return float(np.real(bg.fields_sigma([sg])["U"][0]))

    def Hrr_of(sg):
        return float(bg._bary(Hrr, [sg])[0])

    # I(sigma) = int_1^r Hrr/sqrt(U) dr' = int_sigma^1 Hrr/sqrt(U) dsigma'/sigma'^2,
    # accumulated node to node from the horizon; on the segment touching the
    # horizon the weight (1 - sigma)^(-1/2) carries the simple zero of U
    def seg(a, b):
        if b > 1 - 1e-13:
            v, _ = quad(
                lambda q: Hrr_of(q) / q**2 * np.sqrt((1 - q) / U_of(q)),
                a,
                b,
                weight="alg",
                wvar=(0.0, -0.5),
                epsabs=1e-15,
                epsrel=1e-13,
                limit=200,
            )
        else:
            v, _ = quad(
                lambda q: Hrr_of(q) / q**2 / np.sqrt(U_of(q)),
                a,
                b,
                epsabs=1e-15,
                epsrel=1e-13,
                limit=200,
            )
        return v

    Icum = np.zeros_like(sig)  # sig[0] = 1 (horizon) ... sig[N] = 0 (boundary)
    for k in range(1, len(sig) - 1):
        Icum[k] = Icum[k - 1] + seg(sig[k], sig[k - 1])
    rho = np.zeros_like(sig)
    for k in range(1, len(sig) - 1):
        rho[k] = -0.5 * np.sqrt(U_of(sig[k])) * Icum[k]
    f = bg.fields_sigma(sig[:-1])
    Uu, Up, V = (np.real(f[k]) for k in ("U", "Up", "V"))
    rhop = np.zeros_like(sig)
    m = (sig > 1e-13) & (sig < 1 - 1e-13)
    rhop[:-1][m[:-1]] = (Up / (2 * Uu) * rho[:-1] - Hrr[:-1] / 2)[m[:-1]]
    # horizon node: rho ~ c (r-1)^2 since H_rr(r_p) = 0, so rho = rho' = 0
    # psi' = -Hxr - G rho e^{-2V}/U, psi(oo) = 0
    rho_s = rho * sig  # smooth through the boundary (rho ~ r)
    tail, _ = quad(
        lambda q: Hrr_of(q) / q**2 / np.sqrt(U_of(q)) if q > 0 else 0.0,
        0.0,
        sig[-2],
        epsabs=1e-15,
        epsrel=1e-13,
        limit=200,
    )
    rho_s[-1] = -0.5 * (Icum[-2] + tail)  # sqrt(U) sigma -> 1 at the boundary
    rhs = np.zeros_like(sig)
    rhs[:-1][m[:-1]] = (-Hxr[:-1] - G * rho[:-1] * np.exp(-2 * V) / Uu)[m[:-1]]
    rhs[0] = -Hxr[0]  # rho/U -> 0 at the horizon
    psi = np.zeros_like(sig)

    def rhs_of(q):
        if q > 1 - 1e-13:
            return float(-bg._bary(Hxr, [q])[0])
        ff = bg.fields_sigma([q])
        Uq = float(np.real(ff["U"][0]))
        Vq = float(np.real(ff["V"][0]))
        rq = float(bg._bary(rho_s, [q])[0]) / q
        return float(-bg._bary(Hxr, [q])[0]) - G * rq * np.exp(-2 * Vq) / Uq

    # psi(r) = int_r^oo (Hxr + G rho e^{-2V}/U) dr' = -int_0^sigma rhs dsigma'/sigma'^2,
    # accumulated node to node from the boundary
    # (the last segment, at the boundary, can exhaust the subdivision limit
    # at these tolerances without losing accuracy: the gauge agreement below
    # is the test)
    acc = 0.0
    for k in range(len(sig) - 2, -1, -1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            v, _ = quad(
                lambda q: rhs_of(q) / q**2 if q > 0 else 0.0,
                sig[k + 1],
                sig[k],
                epsabs=1e-15,
                epsrel=1e-13,
                limit=200,
            )
        acc += v
        psi[k] = -acc
    return rho, rhop, psi, rhs


def generic_xi(br):
    sig = br.sig
    # xi^r = (1 - sigma)^2 (0.3 + 0.2 sigma)/sigma (vanishes at the horizon, ~ r at
    # the boundary); xi^x = i psi, psi = 0.1 + 0.3 sigma - 0.2 sigma^2
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(sig > 0, (1 - sig) ** 2 * (0.3 + 0.2 * sig) / sig, 0.0)
        drho_dsig = np.where(
            sig > 0,
            (-2 * (1 - sig) * (0.3 + 0.2 * sig) + 0.2 * (1 - sig) ** 2) / sig
            - (1 - sig) ** 2 * (0.3 + 0.2 * sig) / sig**2,
            0.0,
        )
    psi = 0.1 + 0.3 * sig - 0.2 * sig**2
    dpsi_dsig = 0.3 - 0.4 * sig
    rhop = -(sig**2) * drho_dsig  # d/dr = -sigma^2 d/dsigma
    psip = -(sig**2) * dpsi_dsig
    return rho, rhop, psi, psip


def main(N=64):
    bca = dkk.load_cache()
    worst = 0.0
    for beta in (3.0, 45.0):
        i = int(np.argmin(np.abs(bca["beta"] - beta)))
        br = dkk.make_brane(bca, i, N=N)
        # [4] the coupled 7x7 matrix is non-singular at every interior node
        f = br.bg.fields_sigma(br.sig[1:-1])
        dets = []
        for gam in (1.0, 8.0):
            G = np.sqrt(gam * br.Bc)
            alpha = br.alpha
            for k in range(len(br.sig) - 2):
                M = np.array(
                    dkr._Mmat(
                        1 / br.sig[k + 1],
                        *(
                            float(np.real(f[q][k]))
                            for q in ("U", "Up", "V", "Vp", "W", "Wp")
                        ),
                        br.Bc,
                        G,
                        alpha,
                        float(br.w0[k + 1]),
                        float(br.w0p[k + 1]),
                    ),
                    dtype=complex,
                )
                # equilibrate rows and columns (the raw entries span many decades
                # between horizon and boundary), then the inverse condition number
                M = M / np.max(np.abs(M), axis=1)[:, None]
                M = M / np.max(np.abs(M), axis=0)[None, :]
                dets.append(1.0 / np.linalg.cond(M))
        check(
            f"[4] beta = {br.beta:g}: coupled 7x7 matrix non-singular at all nodes",
            min(dets) > 1e-8,
            f"(min equilibrated 1/cond = {min(dets):.1e})",
        )
        for gam in (1.0, 8.0):
            s = gam * br.Bc
            G = np.sqrt(s)
            sol = br.coupled_solve(s)
            Kref = br.kernel(s)["K"]
            Hyy, Hzz, ay = sol["H"]
            # [5] horizon value and fall-offs
            sig = br.sig
            msk = (sig > 2e-3) & (sig < 4e-2)
            slopes = {
                nm: np.polyfit(np.log(sig[msk]), np.log(np.abs(v[msk])), 1)[0]
                for nm, v in (
                    ("Hyy", Hyy),
                    ("Hzz", Hzz),
                    ("Hrr", sol["Hrr"]),
                    ("Hxr", sol["Hxr"]),
                    ("ay", ay),
                )
            }
            ok5 = (
                abs(sol["Hrr"][0]) < 1e-9
                and min(slopes["Hyy"], slopes["Hzz"], slopes["Hrr"]) > 3.5
                and slopes["Hxr"] > 4.4
                and slopes["ay"] > 1.9
            )
            check(
                f"[5] beta = {br.beta:g}, gamma = {gam:g}: H_rr(r_p) = 0 and fall-offs",
                ok5,
                f"|H_rr(r_p)| = {abs(sol['Hrr'][0]):.1e}; sigma^p: "
                + ", ".join(f"{k} {v:.2f}" for k, v in slopes.items()),
            )
            # algebraic gauge through the general-gauge evaluator (sanity)
            H0 = dict(
                tt=0 * Hyy,
                xx=0 * Hyy,
                yy=Hyy,
                zz=Hzz,
                rr=sol["Hrr"],
                xr=sol["Hxr"],
                bh=1j * G * ay,
            )
            K0 = kernel_any_gauge(br, s, sol["w0"], sol["w0p"], H0)
            # [1] radial gauge
            rho, rhop, psi, psip = radial_gauge_xi(br, s, sol)
            Hn = transformed(br, s, sol, rho, rhop, psi, psip)
            resid_rr = np.max(np.abs(Hn["rr"][1:-1]))
            resid_xr = np.max(np.abs(Hn["xr"][1:-1]))
            Krad = kernel_any_gauge(br, s, sol["w0"], sol["w0p"], Hn)
            d1 = abs(Krad - Kref)
            worst = max(worst, d1)
            check(
                f"[1] beta = {br.beta:g}, gamma = {gam:g}: radial gauge K = algebraic K",
                d1 < TOL_RADIAL
                and abs(K0 - Kref) < 1e-12
                and max(resid_rr, resid_xr) < 1e-8,
                f"K = {Kref:.10f}, |dK| = {d1:.1e}; residual H_rr, H_xr = {resid_rr:.1e}, "
                f"{resid_xr:.1e}; max |H_tt|, |H_xx| = {np.max(np.abs(Hn['tt'])):.3f}, "
                f"{np.max(np.abs(Hn['xx'])):.3f}",
            )
            # [3] control: weight 0.4
            Kw = kernel_any_gauge(br, s, sol["w0"], sol["w0p"], H0, w=0.4)
            Kwr = kernel_any_gauge(br, s, sol["w0"], sol["w0p"], Hn, w=0.4)
            check(
                f"[3] beta = {br.beta:g}, gamma = {gam:g}: weight 0.4 is NOT invariant",
                abs(Kwr - Kw) > 1e3 * max(d1, 1e-12),
                f"|dK(w = 0.4)| = {abs(Kwr - Kw):.2e}",
            )
            # [2] generic gauge
            rho, rhop, psi, psip = generic_xi(br)
            Hg = transformed(br, s, sol, rho, rhop, psi, psip)
            Kgen = kernel_any_gauge(br, s, sol["w0"], sol["w0p"], Hg)
            d2 = abs(Kgen - Kref)
            worst = max(worst, d2)
            check(
                f"[2] beta = {br.beta:g}, gamma = {gam:g}: generic gauge K = algebraic K",
                d2 < TOL_GENERIC,
                f"|dK| = {d2:.1e}",
            )
    log(f"worst gauge discrepancy {worst:.1e}")
    failed = [k for k, v in CHECKS.items() if not v]
    log(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for k in failed:
            log(f"FAILED: {k}")
        sys.exit(1)


if __name__ == "__main__":
    main()
