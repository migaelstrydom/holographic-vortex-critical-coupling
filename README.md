# A critical coupling for the magnetic instability of holographic SU(2) plasmas

Code and LaTeX source for the paper

> Migael Strydom, *A critical coupling for the magnetic instability of holographic SU(2) plasmas* (2026).

A strongly magnetised holographic plasma with an SU(2) global symmetry condenses
into a triangular lattice of vortices. The paper switches on the gravitational
backreaction of the magnetic field exactly, on the D'Hoker–Kraus magnetic brane of
Einstein–SU(2) Yang–Mills theory in AdS₅. The critical field then diverges at the
critical coupling α★ = 2/3, and above that coupling the normal phase has no linear
instability in the charged sector at any field or temperature. Below α★, the paper
follows the lattice as the coupling grows: the triangular lattice forms in a
second-order transition up to α_L = 0.372, where the transition turns first order.

Every number and figure in the paper is produced by a script in this repository.
The section-by-section map is below.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). It installs the pinned interpreter and
packages (Python 3.14, NumPy 2.4.6, SciPy 1.17.1, SymPy 1.14.0, Matplotlib 3.10.9).
These are the versions the published numbers were computed with; the code also
runs on Python 3.13.

```bash
uv sync --frozen         # exact environment from uv.lock, without re-locking
uv run pytest            # fast tests, ~30 s (what they recompute: see below)
bash reproduce.sh        # every figure and number in the paper, ~2.5 h
bash paper/build.sh      # the PDF (needs a TeX installation: pdflatex + bibtex)
```

Without uv, run `pip install -r requirements.txt` from the repository root,
into a Python 3.13 or later environment (plus `pip install pytest` for the tests); it
installs this repository in editable mode. Only an editable install is
supported: the probe-limit scripts are flat top-level modules that read
`scripts/kernel.npz` next to them, so a regular wheel install would scatter
them into site-packages.

Run everything from the repository root.

## Checking the headline numbers in five minutes

```bash
uv run pytest                                                     # ~30 s, every cache against the published values
uv run python -m backreaction.derivations.dk_throat               # alpha_* = 2/3 from the AdS3 x R2 throat (~20 s)
uv run python -m backreaction.numerics.t0_threshold               # alpha_* is the exact T = 0 threshold for d = 4, 5, 6;
                                                                  #   the d = 3 exception alpha_c(3) = 3.037 (~1 min)
uv run python -m backreaction.numerics.dk_background              # the brane, two methods (~1 min)
uv run python scripts/lattice_scan.py                             # probe limit: C_tri = 0.0184443, C_sq = 0.0229818 (~2 s)
uv run python -m backreaction.numerics.dk_uniform --quick         # K_{G=0} = K_0 and its anchors (~30 s; writes nothing)
```

The onset curve B_c(β) itself is `numerics/bc_alpha.py` (~5 min) and the
lattice threshold α_L = 0.372 is the full `numerics/dk_uniform.py` (~4 min).
Each script prints its checks and exits non-zero if one fails. Two of the checks
above rewrite a tracked cache: `t0_threshold` writes
`backreaction/data/t0_threshold.npz` and `dk_background` writes
`backreaction/data/dk_background_grid.npz`. On the reference platform (macOS
arm64) they come back byte-identical; elsewhere expect differences at the
rounding level (`git checkout` restores the committed files).

The fast tests recompute the probe-limit suite, the G = 0 sector, brane solves
at β = 0, 4 and 45, the symbolic α★, the central-charge Wick checks, the probe
Bogoliubov spectrum and two onset-scan points. The other headline numbers are
read from the committed `.npz` caches and compared with the published values;
`reproduce.sh` regenerates those caches. `uv run pytest -m slow` regenerates
most of them, but runs `numerics/dk_kernel.py` with `--anchors` and
`numerics/onset_scan_d.py` with `--quick`, so it does not regenerate
`dk_kernel.npz` or `onset_scan_d.npz`, and it does not run the extra
`numerics/dk_tower.py`.

## What `reproduce.sh` does

It runs every derivation and numerical script the paper relies on, in dependency
order, and writes one log per script to `logs/<timestamp>/` plus a PASS/FAIL
summary, then runs the fast test tier. Every script asserts the numbers it is
responsible for and exits non-zero if a check fails, so a PASS means its checks
passed, and `reproduce.sh` itself exits non-zero if any step failed. The printed
numbers are the ones the paper quotes.

The expensive intermediate results are committed as `.npz` caches in
`backreaction/data/` and `scripts/kernel.npz`, so the figures can be rebuilt
without the long solves. Three of them do some solving of their own: figures 2
and 3 re-solve part of the background ladder, and figure 5 re-solves points of
the throat kernel and the frozen-metric photon problems of its inset (one to two
minutes each):

```bash
bash reproduce.sh --figures    # only the six paper figures, from the caches (~5 min)
```

A full run regenerates the caches as well, including `dk_tower.npz` of the
extra `numerics/dk_tower.py` (see "Not in the paper" below). Checked on a clean clone on the
reference platform (macOS arm64): the generated modules in `backreaction/systems/`,
every cache and the paper figures come back byte-identical. On another platform
expect rounding-level differences in the caches; the slow test tier compares
regenerated caches numerically rather than byte for byte. The probe-limit scripts also write diagnostic plots to an
untracked `figures/` directory.

## From the paper to the code

Paths below are relative to `backreaction/` unless they start with `scripts/`.

| paper | result | script |
|---|---|---|
| §2.2, app. A.1, app. D.1 | D'Hoker–Kraus brane at finite β = αB²: two-method solve, entropy and Schwarzschild checks | `numerics/dk_background.py` |
| app. A.1 | the dictionary to D'Hoker–Kraus (α = 2, 2κ² = 16πG₅); our fixed-temperature entropy shift reproduces theirs, and their throat e^{2V} = B/√3 is (4.1) at α = 2 | `anchors/dhoker_kraus_compare.py` |
| §3.1–3.2, app. B.1, app. D.2 | onset B_c(α) to all orders in αB², three solvers | `numerics/bc_alpha.py`; symbolic `derivations/dk_zeromode.py` |
| §3.3, app. D.5 | small-β limit anchors: the tangent of B_c(α) at α = 0 and the G = 0 sector on AdS₅–Schwarzschild | `numerics/bc_shift.py`, `numerics/g0_thermo.py`; symbolic `derivations/bc_shift.py`, `derivations/g0.py` |
| §3, probe limit | B_c u_H² = 5.13126764 | `scripts/critical_field.py`, `scripts/spectral_check.py`; symbolic `scripts/symbolic_check.py` |
| fig. 1 | the onset curve B_c(α), with T_c/√B on the right axis | `numerics/poster_bc_alpha.py` |
| §4.1–4.3 | AdS₃×ℝ² throat, BF bound, α★ = 2/3, the bound in general d | `derivations/dk_throat.py`, `derivations/brane_flows.py` |
| §4.2–4.3, app. D.2 | α★ is the exact T = 0 threshold (no node, c₁/c₀ = −0.035) for d = 4, 5, 6; d = 3 exception α_c(3) = 3.037 | `numerics/t0_threshold.py`, `numerics/d3_blind_check.py` |
| §4.3, app. D.2 | at finite T the onset coupling for d = 5, 6 stays below α★(d) | `numerics/onset_scan_d.py`; symbolic `derivations/onset_scan_d.py` |
| §4.2, §4.6, app. D.2 | onset by horizon shooting to β ≈ 10⁶⁶; local BKT slope | `numerics/onset_scan.py` |
| fig. 2 | the throat forming along the ladder | `numerics/poster_throat.py` |
| §4.4, app. A.2 | α ↔ C_J/C_T, threshold 8/(d²(d+1)) | `derivations/dk_central_charges.py` |
| §4.4 | supersymmetric couplings α = 1/4 (d = 4) and α = 2 (d = 3); Ward-identity anchors | `derivations/susy_coupling.py`, `derivations/central_charge_anchor.py`, `derivations/central_charge_anchor_d3.py` |
| §1, §4.4 | prior charged-vector BF criterion on the throat; in N = 4 SYM the SO(6) W bosons see α = 3/8, from their throat mass and from C_R/C_T = 1/10 with R-charge 4/3 | `anchors/dgp_charged_vector.py` |
| §4.5, app. B.2 | every other charged channel is stable; the onset is static (ω² real) | `derivations/dk_landau_sectors.py`, `derivations/dk_charged_dynamics.py`, `numerics/dk_charged_qnm.py` |
| §4.6, app. D.2 | ladder to β = 10¹¹, BKT approach law | `numerics/dk_throat.py` |
| §4.6, app. D.3, figs. 3–4 | matched asymptotics, A → 4π, the φ★ plateau | `numerics/dk_throat_matching.py`; figures `numerics/poster_matching.py`, `numerics/poster_approach.py` |
| §5.1, app. C, app. D.4 | the exact quartic kernel on the brane; pairing identity; displayed response equations; gauge independence | `numerics/dk_kernel.py`, `derivations/paper_response_display.py`, `numerics/dk_kernel_gauge_check.py`; symbolic `derivations/dk_stress.py` (the condensate stress, app. C.1), `derivations/dk_response.py`, `derivations/dk_pairing.py`, `derivations/dk_quartic.py`, `derivations/gauge_invariant.py` |
| §5.1, app. D.5 | the strict uniform response K_{G=0} = K₀ | `numerics/dk_uniform.py`; symbolic `derivations/dk_uniform.py` |
| fig. 5 | the kernel family | `numerics/poster_kernel.py` |
| §5.2, app. D.5 | triangular selection, margin over the square | `numerics/dk_selection.py`, `numerics/dk_moduli.py` |
| §5.2–5.3, app. D.5 | moduli scan on the ladder rungs, top-rung margin 0.183; the stripe-edge coupling α_I = 0.494 and its uncertainty | `numerics/dk_moduli_ladder.py`, `numerics/alpha_i_uncertainty.py` |
| §5.2, app. C.4, app. D.5 | the lattice at α★ (BTZ×ℝ²), metric channels, margin +0.1766; the ladder converging onto it (`--ladder`) | `numerics/dk_throat_lattice.py`, `numerics/dk_throat_channels.py`; symbolic `derivations/dk_throat_lattice.py`, `derivations/dk_throat_channels.py` |
| §5.3, fig. 6, app. D.5 | the lattice threshold α_L = 0.3721800 (the uniform lattice destabilises and the transition turns first order); the long-range kernel K₀ changes sign at α = 0.3506 | `numerics/dk_uniform.py`, `numerics/dk_long_wavelength.py`, `numerics/k0_zero_uncertainty.py`; figure `numerics/poster_density.py` |
| §5.3, app. C.5 | second variation of the lattice and its Bogoliubov spectrum over the zone, from the probe limit; the density stiffness of every Bravais lattice | `numerics/dk_bogoliubov.py`, `numerics/dk_lattice_stiffness.py`; symbolic `derivations/lll_second_variation.py` |
| §5.3 | the T_c ratios at fixed field (its plot is a diagnostic, not in the paper) | `numerics/tb_phase_diagram.py` |
| table 4 | the second methods: shooting from both ends for the throat kernel ([C10]); C△, C□ by finite differences on the ladder ([K7]); φ★ and the T = 0 offset by a second construction ([3′], [4′]) and the ladder in one frame ([4]); the onset beyond the ladder and the local slopes by matching ([7]); α_I by the finite-difference kernel ([6]); α_L's finite-difference entry in the error budget | `numerics/dk_throat_channels.py`, `numerics/dk_long_wavelength.py`, `numerics/dk_throat_matching.py`, `numerics/onset_scan.py`, `numerics/alpha_i_uncertainty.py`, `numerics/dk_uniform.py` |
| app. E | triangular selection in the probe limit: exchange kernel, LLL identity, lattice scan, margin +0.2460 | `scripts/exchange_kernel.py`, `scripts/lll_identity.py`, `scripts/lattice_scan.py`, `scripts/nonlinear_reduction.py`; the LLL identity on a generic lattice (τ = 0.3 + 1.1i, app. E.2) is checked in `tests/test_metamorphic.py` |

### Not in the paper

These scripts ship with the code and run in `reproduce.sh`, but the paper does
not quote their results:

- `derivations/d_continuation.py` and `numerics/d3_crossover_why.py`: the
  boundary dimension d as a real parameter, and why d = 3 binds a crossover
  state while d = 4, 5, 6 do not. The paper treats only d = 3, 4, 5, 6.
- `derivations/n4_su2_embeddings.py`: α for every SU(2) inside the SO(6)
  R-symmetry of N = 4 SYM. The paper quotes only the W bosons' α = 3/8 of the
  equal-charge field (§4.4, checked in `anchors/dgp_charged_vector.py`).
- `numerics/probe_bc_precise.py`: the probe B_c u_H² to 30 digits, by two
  methods. It is the reference against which the collocation error of table 4
  is measured; the paper quotes eight decimals.
- `numerics/dk_tower.py`: the excited tower of the onset operator is an
  ordinary Sturm–Liouville spectrum, not controlled by the throat. It writes
  `dk_tower.npz`; `numerics/dk_throat.py` imports it for its check [T4], and the
  tests pin its results.

### Reading the docstrings

Each script's docstring states its method, checks and expected output, and
points to the paper section it supports ("paper sec. 4.6", "app. C.3"). A few
docstrings mention earlier derivations that are not included here; the paper
does not rely on them. One name differs from the paper: the code calls the
metric function of the z direction W where the paper writes Z (the paper keeps
W for the charged vector).

## Layout

- `paper/`: the LaTeX source (JHEP class, unmodified). It reads `references.bib`
  and the figures in `backreaction/figures/` directly; `bash paper/build.sh`
  builds the PDF.
- `backreaction/`: the finite-coupling computation, an installed package.
  - `derivations/`: SymPy derivations. Several write generated modules into
    `systems/`.
  - `systems/`: generated code. Never edit these files by hand. Each ends with a
    provenance stamp, and `tests/test_generated_systems.py` fails if a file was
    edited or its generator has changed since it was written.
  - `numerics/`: the solvers and scans.
  - `anchors/`: comparisons against independent results.
  - `data/`: the committed caches.
  - `figures/`: output.
- `scripts/`: the probe-limit computation, which the finite-coupling code uses at
  β = 0. These are flat modules installed at top level, so they import each other
  by bare name.
- `tests/`: pytest. `tests/published_values.py` holds the published numbers, and the
  tests assert that the code still produces them. `uv run pytest -m slow` reruns
  the derivations, the output-scraping checks and most of the cache
  regeneration (not `dk_kernel.npz`, `onset_scan_d.npz` or `dk_tower.npz`), and
  compares the regenerated caches with the committed ones (several hours: the
  per-script budgets add up to about four).

## Conventions

The conventions are those of section 2 of the paper. The brane metric is
ds² = −U dt² + dr²/U + e^{2V}(dx² + dy²) + e^{2Z}dz², with the horizon at r = 1
and T = 1/π. The coupling is α = κ²/ĝ², and β = αB². Each derivation restates the
conventions it uses in its docstring.

## Licence

Code (everything outside `paper/`, `references.bib`, `backreaction/figures/` and
this README): MIT, see `LICENSE`. Paper text, bibliography, figures and this
README: CC BY 4.0, see `LICENSE-docs`. `NOTICE` states the split. `paper/jheppub.sty` and `paper/JHEP.bst` are the JHEP
class files, © SISSA Medialab, distributed under the LaTeX Project Public
License (see their headers).

## Citation and contact

If you use this code, please cite the paper; `CITATION.cff` has the details
(the arXiv identifier will be added on submission). Questions and corrections:
Migael Strydom, migael@strydom.me.uk, or an issue on this repository.
