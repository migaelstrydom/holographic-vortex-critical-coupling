#!/usr/bin/env bash
# Reproduce every number and figure in the paper.
#
# Usage (from the repository root):
#   bash reproduce.sh              # everything, in dependency order (~2.5 h),
#                                  # then the fast test tier
#   bash reproduce.sh --figures    # only the six paper figures, from the
#                                  # committed caches (~5 min)
#   bash reproduce.sh --help       # this text
#
# Every script asserts the numbers it is responsible for and exits non-zero if
# any check fails, so a step marked PASS means its checks passed; the printed
# numbers are the ones the paper quotes (see README.md for the map from paper
# sections to scripts). The full run ends with the fast test tier (pytest),
# which compares the regenerated caches with the published values; the slow
# tier (`uv run pytest -m slow`, several hours) is separate. One log per step goes
# to logs/<timestamp>/, with a PASS/FAIL summary in summary.tsv, and the script
# exits non-zero if any step failed.
#
# Steps marked "extra" support results the paper does not quote (README.md,
# "Not in the paper").
#
# A full run regenerates the committed .npz caches, the generated modules in
# backreaction/systems/ and the figures. Run it on a clean checkout and inspect
# `git status` afterwards: on the reference platform (macOS arm64) the
# generated modules, the caches and the six paper figures (PDF) come back
# byte-identical; on another platform expect rounding-level differences in the
# caches.
# Runs against the locked environment (uv.lock, --frozen).

set -u
cd "$(dirname "$0")"

case "${1:-}" in
    "" | --figures) ;;
    -h | --help)
        sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
        exit 0 ;;
    *)
        echo "error: unknown argument '$1' (try --help)" >&2
        exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    echo "error: at most one argument (try --help)" >&2
    exit 2
fi

if ! command -v uv >/dev/null; then
    echo "error: uv not found -- see https://docs.astral.sh/uv/" >&2
    exit 1
fi
uv sync --frozen --quiet || { echo "error: uv sync failed" >&2; exit 1; }

LOGDIR="logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOGDIR" figures   # figures/: the probe scripts' own plots (untracked)
SUMMARY="$LOGDIR/summary.tsv"
printf 'status\tseconds\tscript\n' > "$SUMMARY"
FAILED=0
export MPLBACKEND=Agg

run() {
    # run <label> <command...>
    local label=$1; shift
    local log="$LOGDIR/${label//[\/ ]/_}.log"
    printf '==> %s\n' "$label"
    local t0=$SECONDS status
    if "$@" >"$log" 2>&1; then status=PASS; else status=FAIL; FAILED=$((FAILED + 1)); fi
    local dt=$((SECONDS - t0))
    printf '%s\t%d\t%s\n' "$status" "$dt" "$label" >> "$SUMMARY"
    printf '    %s (%ds)  log: %s\n' "$status" "$dt" "$log"
}

py() { uv run --no-sync python "$@"; }
mod() { local m=$1; shift; run "$m${*:+ $*}" py -m "backreaction.$m" "$@"; }

figures() {
    # ---- the six paper figures (from the caches; figures 2-3 re-solve the ladder)
    mod numerics.poster_bc_alpha        # figure 1
    mod numerics.poster_throat          # figure 2
    mod numerics.poster_matching        # figure 3
    mod numerics.poster_approach        # figure 4
    mod numerics.poster_kernel          # figure 5
    mod numerics.poster_density         # figure 6
}

if [ "${1:-}" = "--figures" ]; then
    figures
else
    # ---- probe limit (appendix E; the beta = 0 anchor of everything else) ---
    run symbolic_check          py scripts/symbolic_check.py
    run critical_field          py scripts/critical_field.py
    run spectral_check          py scripts/spectral_check.py
    run nonlinear_reduction     py scripts/nonlinear_reduction.py
    run exchange_kernel         py scripts/exchange_kernel.py      # writes scripts/kernel.npz
    run lll_identity            py scripts/lll_identity.py
    run lattice_scan            py scripts/lattice_scan.py

    # ---- small-beta limit anchors: the G = 0 sector on AdS5-Schwarzschild and
    # ---- the tangent of B_c(alpha) at beta = 0 (section 3.3; appendix D)
    mod derivations.gauge_invariant     # writes systems/einstein_invgauge.py
    mod derivations.g0                  # writes systems/einstein_g0.py
    mod derivations.bc_shift            # writes systems/bc_shift.py
    mod numerics.g0_thermo
    mod numerics.bc_shift

    # ---- the brane and the onset (sections 2.2, 3; appendices A, B, D.1-D.2)
    mod derivations.dk_zeromode         # writes systems/dk_zeromode.py
    mod numerics.dk_background
    mod numerics.probe_bc_precise       # extra: the probe B_c to 30 digits (table 4's reference)
    mod numerics.bc_alpha
    mod anchors.dhoker_kraus_compare

    # ---- the response system and the kernel (section 5.1; appendix C) ------
    mod derivations.dk_quartic          # writes systems/dk_quartic.py
    mod derivations.dk_stress           # writes systems/dk_stress.py
    mod derivations.dk_response         # writes systems/dk_response.py (~5 min)
    mod derivations.dk_pairing          # writes systems/dk_pairing.py
    mod numerics.dk_kernel              # ~3 min
    mod derivations.paper_response_display  # app. C displays vs the generated rows
    mod numerics.dk_kernel_gauge_check  # K is gauge independent; the weight 1/2
    mod numerics.dk_selection
    mod numerics.dk_moduli

    # ---- the throat and alpha_* (section 4; appendix D.3) -------------------
    mod derivations.dk_throat
    # dk_throat and onset_scan cross-check each other, so whichever runs first
    # reads the other's committed cache; both regenerate byte-identically.
    mod numerics.dk_tower               # extra, ~9 min: the excited tower (read by dk_throat [T4])
    mod numerics.dk_throat              # ~6 min, ladder to beta = 1e11
    mod numerics.dk_throat_matching     # ~7 min
    mod numerics.onset_scan             # ~11 min, horizon shooting to beta ~ 1e66
    mod derivations.brane_flows         # general d: fixed points and flows
    mod numerics.t0_threshold           # ~1 min, T = 0 node count, d = 3..6
    mod derivations.onset_scan_d        # ~2 min, general-d horizon data and zero mode
    mod numerics.onset_scan_d           # ~24 min, finite-T onset for d = 5, 6
    mod numerics.d3_blind_check         # ~8 min, independent d = 3 check
    mod derivations.d_continuation      # extra, ~1 min: d as a real parameter
    mod numerics.d3_crossover_why       # extra: why d = 3 binds a crossover state
    mod derivations.dk_landau_sectors
    mod derivations.dk_charged_dynamics # the onset is static (omega^2 real)
    mod numerics.dk_charged_qnm         # ~1 min, quasinormal modes cross at omega = 0
    mod derivations.dk_central_charges
    mod derivations.susy_coupling
    mod derivations.central_charge_anchor
    mod derivations.central_charge_anchor_d3
    mod derivations.n4_su2_embeddings   # extra: SU(2) embeddings in N = 4 SYM
    mod anchors.dgp_charged_vector

    # ---- the lattice at alpha_* and along the ladder (sections 5.2-5.3) -----
    mod derivations.dk_throat_lattice
    mod numerics.dk_throat_lattice --ladder
    mod derivations.dk_throat_channels  # ~30 min
    mod numerics.dk_throat_channels --ladder
    mod numerics.dk_long_wavelength     # ~4 min, K_0 along the ladder and its zero
    mod numerics.k0_zero_uncertainty
    mod derivations.dk_uniform          # writes systems/dk_uniform.py
    mod numerics.dk_uniform             # ~4 min, K_{G=0} and alpha_L
    mod derivations.lll_second_variation
    mod numerics.dk_bogoliubov          # ~4 min, Bogoliubov spectrum over the zone
    mod numerics.dk_lattice_stiffness   # ~8 min, every Bravais lattice
    mod numerics.dk_moduli_ladder       # ~7 min
    mod numerics.alpha_i_uncertainty    # ~3 min, the stripe-edge coupling alpha_I
    mod numerics.tb_phase_diagram       # the T_c ratios of section 5.3

    figures

    # ---- the fast test tier: regenerated caches against the published values
    run pytest uv run --no-sync pytest -q
fi

echo
echo "summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
if [ "$FAILED" -ne 0 ]; then
    echo "$FAILED script(s) FAILED -- see the logs above" >&2
    exit 1
fi
echo "all scripts passed"
