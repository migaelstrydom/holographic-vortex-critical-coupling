"""Finite-alpha lattice selection on the D'Hoker-Kraus brane: the gap
C_sq - C_tri from the exact kernel on the grid beta <= 45 (paper sec. 5.2).

The LLL quartic functional per lattice, at one flux quantum per cell, is

    C_lattice(beta) = (1/2) sum_shells mult_k * exp(-gamma_k/2) * K(gamma_k; beta)

with gamma = G^2/B fixed by the LATTICE SHAPE alone (gamma_k = u_L * k,
u_L = 4 pi/sqrt3 triangular, 2 pi square; k = m^2+mn+n^2 resp. m^2+n^2), so the
Boltzmann-like weights exp(-gamma/2) are B-independent and the whole
beta-dependence enters through the exact kernel K evaluated at s = gamma*B_c(beta)
on the brane.  This automatically includes the shift of the sampled arguments
(they move with B_c(beta)) and the metric-deformation and
exchange pieces, all resummed to all orders in alpha B^2 -- that is the point
of doing it here rather than extrapolating.

K is recomputed at the exact shell arguments (no interpolation of the scan
grid).  Normalisation: K is in the brane SL norm J = int p1 w0^2 dr = 1, which
differs from the flat J2 = 1 norm of the probe limit by a positive s-INDEPENDENT factor
at each beta; a positive rescaling cannot move the zero of the gap, so the
crossing location is norm-invariant.  At beta = 0 the two norms coincide (anchor
[1] of dk_kernel), so the gap there must reproduce the probe value
computed independently by the probe lattice code `scripts/lattice_scan.py`
(AdS5-Schwarzschild in the u chart, the tabulated probe kernel of
`exchange_kernel.py`): +0.0045374436, agreement ~1e-8 relative, set by the
spline of that kernel table.

Run:  uv run python -m backreaction.numerics.dk_selection
"""

import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backreaction import paths
from backreaction.numerics import dk_kernel as K

OUT_PATH = paths.data("dk_selection.npz")
FIG_PATH = paths.figure("dk_selection.png")
# C_sq - C_tri at alpha = 0 from the independent probe lattice code
# (scripts/lattice_scan.py, spline of the tabulated probe kernel); recomputed
# in main() and checked against this literal.
PROBE_GAP = 0.0045374435605
WEIGHT_FLOOR = 1e-9


def shells_gamma(kind, kmax=40):
    """(gamma, multiplicity) shells in the B-independent variable gamma = G^2/B."""
    out = {}
    for m in range(-8, 9):
        for n in range(-8, 9):
            if m == 0 and n == 0:
                continue
            k = m * m + m * n + n * n if kind == "tri" else m * m + n * n
            if k <= kmax:
                out[k] = out.get(k, 0) + 1
    unit = 4 * np.pi / np.sqrt(3.0) if kind == "tri" else 2 * np.pi
    return sorted((unit * k, mult) for k, mult in out.items())


def lattice_sum(br, kind, kernel_of_s):
    """(1/2) sum mult exp(-gamma/2) K(gamma B_c), truncated at WEIGHT_FLOOR."""
    tot = 0.0
    used = 0
    for gam, mult in shells_gamma(kind):
        w = mult * np.exp(-0.5 * gam)
        if w < WEIGHT_FLOOR:
            continue
        tot += w * kernel_of_s(gam * br.Bc)
        used += 1
    return 0.5 * tot, used


def gap_at(br, kernel_of_s=None):
    if kernel_of_s is None:

        def kernel_of_s(s):
            return br.kernel(s)["K"]

    ctri, ntri = lattice_sum(br, "tri", kernel_of_s)
    csq, nsq = lattice_sum(br, "sq", kernel_of_s)
    return csq - ctri, ctri, csq, (ntri, nsq)


def main():
    t0 = time.time()
    bca = K.load_cache()
    betas = np.asarray(bca["beta"], float)

    # ---- anchor: beta = 0 must reproduce the probe gap -------------------
    i0 = int(np.argmin(np.abs(betas)))
    br0 = K.make_brane(bca, i0)
    g0, c_tri0, c_sq0, nsh = gap_at(br0)
    import lattice_scan

    probe = lattice_scan.quartic_coefficient(
        0.0, 1.0
    ) - lattice_scan.quartic_coefficient(0.5, np.sqrt(3) / 2)
    assert abs(probe - PROBE_GAP) < 1e-12, "probe lattice gap changed"
    rel = abs(g0 - probe) / abs(probe)
    K.tlog(
        f"[1] beta = 0 gap: C_sq - C_tri = {g0:+.14f} vs probe {probe:+.14f} "
        f"(rel {rel:.1e}); C_tri = {c_tri0:.6f}, C_sq = {c_sq0:.6f}; "
        f"shells used (tri, sq) = {nsh}"
    )
    assert rel < 3e-8, "anchor 1 FAILED -- beta=0 gap does not reproduce the probe"

    # ---- the scan --------------------------------------------------------
    gaps, ctris, csqs, alphas, Bcs = [], [], [], [], []
    for i, b in enumerate(betas):
        br = K.make_brane(bca, i)
        g, ct, cs, _ = gap_at(br)
        gaps.append(g)
        ctris.append(ct)
        csqs.append(cs)
        alphas.append(br.alpha)
        Bcs.append(br.Bc)
        K.tlog(
            f"    beta = {b:6.2f} (alpha = {br.alpha:.5f}): "
            f"C_tri = {ct:+.6f}  C_sq = {cs:+.6f}  gap = {g:+.8f}  "
            f"[{time.time() - t0:.1f}s]"
        )
    gaps = np.array(gaps)
    alphas = np.array(alphas)

    # ---- two-method spot check at a mid beta -----------------------------
    imid = int(np.argmin(np.abs(betas - 8.0)))
    brm = K.make_brane(bca, imid)
    fd = K.BraneFD(brm, 240)
    gfd, _, _, _ = gap_at(brm, kernel_of_s=lambda s: fd.kernel(s)["K"])
    dgap = abs(gaps[imid] - gfd) / abs(gaps[imid])
    K.tlog(
        f"[3] two-method gap at beta = {brm.beta}: Chebyshev {gaps[imid]:+.8f} "
        f"vs FD {gfd:+.8f} (rel {dgap:.1e})"
    )
    assert dgap < 1e-4, "check 3 FAILED -- Chebyshev and FD gaps disagree"

    # ---- verdict ---------------------------------------------------------
    K.tlog(
        f"[4] gap over the grid: min {gaps.min():+.8f} at alpha = "
        f"{alphas[int(np.argmin(gaps))]:.5f}; max {gaps.max():+.8f}"
    )
    if (gaps < 0).any():
        j = int(np.argmax(gaps < 0))
        K.tlog(
            f"[4] SQUARE CROSSING between alpha = {alphas[j - 1]:.5f} and "
            f"{alphas[j]:.5f}"
        )
        raise SystemExit("check 4 FAILED -- the published result is no crossing")
    else:
        slope0 = (gaps[1] - gaps[0]) / (alphas[1] - alphas[0])
        K.tlog(
            f"[4] NO CROSSING on the grid: triangular stays favoured to "
            f"alpha = {alphas[-1]:.5f} (beta = {betas[-1]:.0f}); "
            f"gap/gap_0 = {gaps[-1] / gaps[0]:.4f}; "
            f"initial slope d(gap)/d(alpha) = {slope0:+.5f}"
        )

    np.savez(
        OUT_PATH,
        beta=betas,
        alpha=alphas,
        B_c=np.array(Bcs),
        gap=gaps,
        C_tri=np.array(ctris),
        C_sq=np.array(csqs),
        probe_gap=PROBE_GAP,
        readme="C_lat = 1/2 sum mult e^{-gamma/2} K(gamma B_c); "
        "gap = C_sq - C_tri; brane SL norm (positive rescaling, "
        "zero of gap is norm-invariant)",
    )
    K.tlog(f"saved {OUT_PATH}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].axhline(0.0, color="0.6", lw=0.8)
    ax[0].plot(alphas, gaps, "o-", ms=3, label=r"exact brane $K$")
    ax[0].set_xlabel(r"$\alpha$")
    ax[0].set_ylabel(r"$C_{\rm sq} - C_{\rm tri}$")
    ax[0].set_title("lattice selection gap")
    ax[0].legend(fontsize=8)
    ax[1].plot(alphas, gaps / gaps[0], "o-", ms=3)
    ax[1].axhline(0.0, color="0.6", lw=0.8)
    ax[1].set_xlabel(r"$\alpha$")
    ax[1].set_ylabel("gap / gap$(0)$")
    ax[1].set_title("gap, normalised to the probe value")
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=160)
    K.tlog(f"saved {FIG_PATH}")
    K.tlog(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
