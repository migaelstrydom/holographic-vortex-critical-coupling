"""Numeric verification of the lowest-Landau-level lattice identities.

For a vortex-lattice LLL function with one flux quantum per unit cell
(cell area 2 pi / B),

    psi(x, y) = sum_n  c_n  exp(i n k y)  exp(-B (x + n k/B)^2 / 2),

|psi|^2 is periodic on the lattice spanned by  A1 = (0, 2 pi/k)  and
A2 = (k/B, dy),  provided c_n = gamma^n exp(i pi n(n-1) dy k / (2 pi));
the choice of gamma only translates the lattice.  The Fourier coefficients
of |psi|^2 on the reciprocal lattice satisfy the LLL identity

    lambda_G := <|psi|^2 e^{-i G.r}> / <|psi|^2>,   |lambda_G| = e^{-G^2/(4B)},

independently of the lattice shape, so the Abrikosov ratio is the theta sum

    beta = <|psi|^4> / <|psi|^2>^2 = sum_G |lambda_G|^2 = sum_G e^{-G^2/(2B)}.

This script verifies both statements numerically for the square and the
triangular lattice, reproducing the classic values
beta(square) = 1.1803406, beta(triangular) = 1.1595953.
"""

import numpy as np

B = 5.13126764   # any positive value works; use B_c for definiteness


def make_lattice(tau):
    """Direct basis (A1 along y) with cell area 2 pi/B and modulus tau,
    plus the matching LLL phase sequence c_n."""
    Ly = np.sqrt(2 * np.pi / (B * tau.imag))     # |A1|
    A1 = np.array([0.0, Ly])
    A2 = np.array([Ly * tau.imag, Ly * tau.real])  # area = Ly^2 tau_im = 2 pi/B
    k = 2 * np.pi / Ly
    assert abs(k / B - A2[0]) < 1e-12            # centres march with A2_x
    dy = A2[1]
    phases = lambda n: np.exp(1j * np.pi * n * (n - 1) * dy * k / (2 * np.pi))
    return A1, A2, k, phases


def reciprocal(A1, A2):
    area = A1[0] * A2[1] - A1[1] * A2[0]
    b1 = 2 * np.pi * np.array([A2[1], -A2[0]]) / area
    b2 = 2 * np.pi * np.array([-A1[1], A1[0]]) / area
    return b1, b2


def psi_on_cell(A1, A2, k, phases, Ngrid=128, nmax=40):
    s1, s2 = np.meshgrid(np.arange(Ngrid) / Ngrid, np.arange(Ngrid) / Ngrid,
                         indexing="ij")
    X = s1 * A1[0] + s2 * A2[0]
    Y = s1 * A1[1] + s2 * A2[1]
    psi = np.zeros_like(X, dtype=complex)
    for n in range(-nmax, nmax + 1):
        psi += phases(n) * np.exp(1j * n * k * Y - B * (X + n * k / B)**2 / 2)
    return psi


def check_lattice(name, tau):
    print(f"--- {name} lattice (tau = {tau:.4f}) ---")
    A1, A2, k, phases = make_lattice(tau)
    psi = psi_on_cell(A1, A2, k, phases)
    dens = np.abs(psi)**2
    mean2 = dens.mean()
    beta_direct = (dens**2).mean() / mean2**2

    lam = np.fft.fft2(dens) / dens.size / mean2
    b1, b2 = reciprocal(A1, A2)
    print("   (m,n)    |lambda_G|      e^{-G^2/4B}")
    max_err = 0.0
    for m, n in [(0, 1), (1, 0), (1, 1), (1, -1), (2, 0), (2, 1)]:
        G = m * b1 + n * b2
        pred = np.exp(-np.dot(G, G) / (4 * B))
        got = np.abs(lam[m % lam.shape[0], n % lam.shape[1]])
        max_err = max(max_err, abs(got - pred))
        print(f"   ({m:2d},{n:2d})   {got:.10f}   {pred:.10f}")

    mm, nn = np.meshgrid(np.arange(-25, 26), np.arange(-25, 26), indexing="ij")
    G2 = ((mm[..., None] * b1 + nn[..., None] * b2)**2).sum(-1)
    beta_theta = np.exp(-G2 / (2 * B)).sum()
    print(f"   beta direct = {beta_direct:.10f}   theta sum = {beta_theta:.10f}")
    print(f"   max |lambda| error = {max_err:.2e},  "
          f"|beta mismatch| = {abs(beta_direct - beta_theta):.2e}\n")
    assert max_err < 1e-10 and abs(beta_direct - beta_theta) < 1e-10
    return beta_theta


if __name__ == "__main__":
    beta_sq = check_lattice("square", 1j)
    beta_tri = check_lattice("triangular", 0.5 + 1j * np.sqrt(3) / 2)

    print(f"beta(square)     = {beta_sq:.7f}   (classic value 1.1803406)")
    print(f"beta(triangular) = {beta_tri:.7f}   (classic value 1.1595953)")
    assert abs(beta_sq - 1.1803406) < 1e-6 and abs(beta_tri - 1.1595953) < 1e-6
    print("LLL identity and Abrikosov ratios verified.")
