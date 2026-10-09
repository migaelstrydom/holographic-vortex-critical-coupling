"""The self-asserting derivations and anchors still run and still say so.

This is the output-scraping tier, and it is honest about being that: it runs
each script as a subprocess, requires a zero exit code, and requires the
published numbers to appear in its stdout.  It tests no API and proves nothing
about internal structure.  What it buys is coverage of the parts of the
backreaction tree that have no cache and no return value -- the SymPy
derivations, whose whole output is a sequence of `assert`s, and the anchors,
several of which used only to *print* PASS/FAIL.  Every script exits non-zero
when a check fails, and the scrape also guards the printed conclusions, not
only the exit code.  The probe-limit mains under `scripts/` are run the same
way (their rows are written `../scripts/<name>.py`).

It deliberately does not run the generators under `derivations/`
that write into `systems/`: `tests/test_generated_systems.py` re-runs those,
and re-running them here would duplicate that work for no extra signal.  It
does run most of the `numerics/` scripts that rewrite a committed `.npz`, each
into the scratch copy; the last test then compares every cache they rewrote
with the committed one (numerically: byte identity holds only on the reference
platform, macOS arm64).

Everything runs against a **scratch copy** of the package and of `scripts/`,
so a run can never touch the committed caches or figures.  The isolation depends on invoking each
script by file path rather than with `-m`: Python then puts the script's own
directory on `sys.path` first, and the `backreaction` import resolves through
PYTHONPATH into the scratch tree; the `scripts/` modules resolve their data
and figure paths from their own location, so the scratch copy writes into
itself.  With `-m`, `sys.path[0]` is the working directory and the real
package wins instead.

The per-script estimates below are seconds on a laptop; the rows run one
after another and their estimates add up to about four hours.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PACKAGE = PROJECT_ROOT / "backreaction"

# (module path under backreaction/, argv, approximate seconds, required stdout
# patterns).  The patterns are regexes over the whole captured stdout, and are
# chosen to be the *conclusions* -- a quoted number or the verdict
# line -- rather than incidental formatting.
SCRIPTS = [
    # --- symbolic derivations (paper sec. 4.1, sec. 4.2) -----------------------------------------
    ("derivations/dk_throat.py", [], 20, [
        # Paper sec. 4.1, sec. 4.2: including the Wong d = 3 anchor, the only fully external one.
        r"l3\^2 = 1/3",
        r"alpha_\* = 2/3",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    # --- anchors (paper sec. 2.2, app. A.1) -------------------------------------------------
    ("anchors/dhoker_kraus_compare.py", [], 480, [
        # Paper sec. 2.2, app. A.1: the page-level coefficient mapping onto D'Hoker-Kraus.  An
        # external comparison, so a silent failure here would be expensive --
        # and it is what caught our inverted magnetisation log.
        r"VERDICT TABLE",
        r"trace anomaly\s+-1/2\s+-1/2\s+yes",
        r"ALL CHECKS PASSED",
    ]),
    ("anchors/dgp_charged_vector.py", [], 60, [
        # Paper sec. 1, sec. 4: Donos-Gauntlett-Pantelidou's charged-vector window on Romans'
        # line and the D'Hoker-Kraus origin value -- the top-down anchor of
        # "the supersymmetric point is inside the unstable window".
        r"f1 = -2\.0052, 0\.8733",
        r"L\^2 m\^2 = -1\.3333 < -1",
        r"alpha_eff = 0\.3750 = 3/8",
        # paper sec. 4.4: the same 3/8 from C_R/C_T = 1/10 and R-charge 4/3.
        r"W R-charge 4/3 -> alpha = 3/8, equal to \[4\]",
        r"all asserted",
    ]),
    # --- numerics with no cache of their own (paper sec. 5.1, app. C.3) -----------------
    ("numerics/dk_kernel.py", ["--anchors"], 60, [
        # Paper sec. 5: anchor [3], the polarised two-source identity.
        r"\[3\] worst of \(a\),\(b\),\(c\) = \d\.\de-1[0-9]",
        r"anchors done",
    ]),
    ("derivations/dk_landau_sectors.py", [], 10, [
        # Paper sec. 4.5, app. B.2: every static channel of the charged sector other than the
        # polarised LLL is positive semi-definite; T_1 = 0, det T_n = 0 with the
        # pure-gauge null vector, gauge invariance of the reduction.
        r"T_1 = 0 identically",
        r"n = 4: det T_n = 0",
        r"n = 4: density shifts by a total derivative",
        r"\[5a\] F\^W_xy = \(i/2\)\(D_\+ W_- - D_- W_\+\)",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("derivations/dk_central_charges.py", [], 5, [
        # Paper sec. 4.4, app. A.2: alpha as C_J/C_T and the critical ratio 8/(d^2(d+1)).
        r"\(C_J/C_T\)_\* = 8/\(d\*\*2\*\(d \+ 1\)\)",
        r"d = 4: \(C_J/C_T\)_\* = 1/10",
        r"(\d+)/\1 checks passed",
    ]),
    ("derivations/susy_coupling.py", [], 5, [
        # Paper sec. 4.4, app. A.2: alpha_SUSY = L^2/4 three ways, and the X truncation caveat.
        r"alpha_SUSY = kappa\^2/ghat\^2 = L\^2/4 = 1/4",
        r"(\d+)/\1 checks passed",
    ]),
    ("derivations/n4_su2_embeddings.py", [], 10, [
        # An extra: alpha_eff of every SU(2) embedding in N = 4 SYM; the paper
        # quotes only the W bosons' 3/8 (sec. 4.4, anchors/dgp_charged_vector.py).
        r"23/23 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("derivations/central_charge_anchor.py", [], 5, [
        # C_J(SU(2)_R)/C_T = 3/80 from the N = 2 Ward identities and from
        # (3/20) alpha at alpha_SUSY = 1/4 (paper sec. 4.4, app. A.2); the
        # free N = 4 SYM route is an extra, not in the paper.
        r"ROUTE 2 \(free N = 4 SYM, SU\(2\)_R index 1\):.*=>  C_J/C_T = 3/80",
        r"ROUTE 1 \(N = 2 Ward identities\):.*=>  3/80",
        r"HOLOGRAPHY \(C_J/C_T map x alpha_SUSY\):.*=>  3/80",
        r"VERDICT: C_J\(SU\(2\)_R\)/C_T = 3/80 on all three sides",
        r"C_J/C_T\(free complex scalar, unit charge\) = 3/8",
        r"C_J/C_T\(free Weyl fermion, unit charge\) = 1/2",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("derivations/central_charge_anchor_d3.py", [], 5, [
        # Paper sec. 4.4, app. A.2: the d = 3 anchor, alpha_SUSY = 2 against the 3d N = 2 Ward
        # identity C_R/C_T = 1/6, and alpha/12 = 1/6 exactly.
        r"both bulk readings agree: alpha_SUSY\(d = 3\) = 2",
        r"Ward identity: C_R/C_T = 1/6 in EVERY 3d N = 2 SCFT",
        r"map: alpha_SUSY/12 = 1/6 = C_R/C_T \(Ward identity\) = C_R/C_T \(free chiral Wick\)",
        r"(\d+)/\1 checks passed",
    ]),
    ("numerics/poster_bc_alpha.py", [], 5, [
        # Paper figure 1: the onset curve B_c(alpha), drawn from the caches.
        # No solves; the scrape checks the numbers the figure encodes.
        r"B_c\(0\) = 5\.1312676",
        r"alpha_star = 0\.6667",
        r"alpha_L = 0\.37218   B_c\(alpha_L\) = 132\.26",
        r"ladder endpoint: beta = 1e\+11, alpha = 0\.5908, B_c = 4\.1143e\+05",
    ]),
    ("numerics/poster_matching.py", [], 180, [
        # Paper section 4.6, figure 3: the matching diagram on the beta = 1e8
        # rung.  Solves the ladder and the T = 0 brane (in parallel); the
        # scrape checks the rung's phases against the matching condition (paper sec. 4.6, app. D.3).
        r"nu = 0\.32464",
        r"rho_\* = 43\.9576",
        r"r_c = \(beta/6\)\^\(1/4\) = 63\.8943",
        r"chi\(nu\) = 1\.1304   phi_\*\(nu\) = 0\.2390",
        r"nu ln\(rho_\*/h\) = 1\.3598",
    ]),
    ("numerics/poster_approach.py", [], 5, [
        # Paper section 4.6, figure 4: the approach law from the caches.  The
        # effective slope of the matching condition at the ladder's ln beta
        # and at its asymptote (paper sec. 4.6, app. D.3; check [5]).
        r"off = 0\.0313",
        r"ln beta =    25:  A_eff = 7\.436",
        r"ln beta =  1600:  A_eff = 12\.543",
        r"4 pi = 12\.566",
    ]),
    ("numerics/poster_kernel.py", [], 120, [
        # Paper section 5.1, figure 5: the quartic kernel family (paper sec. 5.1, app. C.3;
        # dk_kernel.npz) and the alpha_* throat kernel (paper sec. 5.1, sec. 5.2;
        # dk_throat_channels.npz), drawn from the caches, densified and
        # refined by direct solves; the inset's gauge-invariant split:
        # the frozen-metric photon by FD against the Green's function.
        r"K\(s = 0\.02 B_c\): 1\.551947 \(beta = 0\) -> 0\.432163 \(beta = 45\)",
        r"min_s K at the largest harmonic: 0\.044797 -> 0\.053665",
        r"grid point s/B_c = 7\.5: 0\.22506 -> 0\.24941",
        r"gamma_0 = 0\.8465",
        r"interior maximum S_full = 0\.0509 at gamma = 4\.23",
        r"gamma = 0\.0200: photon \+0\.9845, metric -P_dress/2C_4 -1\.0418",
        r"gamma = 0\.8465: photon \+0\.6042, metric -P_dress/2C_4 -0\.6042",
    ]),
    ("numerics/poster_density.py", [], 5, [
        # Paper section 5.3, figure 6: K_0/C_4 and C_tot/C_4 along the
        # ladder from dk_long_wavelength.npz.  No solves.
        r"K_0/C_4 = \+0\.7239 \(beta = 1\), \+0\.1777 \(beta = 45\), "
        r"\+0\.0267 \(beta = 1e3\), -0\.0575 \(beta = 1e8\)",
        r"alpha_K0 = 0\.3506   \(zero of K_0: long-range part turns attractive",
        r"alpha_L = 0\.37218   beta_L = 6510   B_c u_H\^2 = 132\.26",
    ]),
    ("numerics/dk_moduli.py", [], 60, [
        # Paper sec. 5.2, app. D.5: the paper quotes the margin over the square,
        # 0.246 -> 0.227; the well curvature over C_tri (down 8%) is a
        # regression value.
        r"relative margin \(C_sq - C_tri\)/C_tri: 0\.2460 -> 0\.2268 at alpha = 0\.17445, "
        r"monotone True; Hessian/C_tri: 7\.708 -> 7\.063 \(-8\.4%\)",
        r"\[4\] VERDICT: triangular is the global minimum",
    ]),
    ("numerics/dk_throat.py", [], 360, [
        # Paper sec. 4.1, sec. 4.2: the two-method checks of the kind appendix D.1-D.2 describes,
        # at beta = 1e4 (background 5e-9, onset 2e-10 with the exact source
        # extraction; these particular numbers are not quoted in the paper), the
        # onset on the top rung converging as r_max^-3 to 2e-10, and the probe
        # anchor B_c(0) = 5.13126764.
        r"collocation vs shooting at beta = 1e4  max profile difference 5\.\d+e-09",
        r"horizon-shooting c_0 root at beta = 1e4  161\.1663538\d+ vs 161\.1663538\d+  \(rel \d\.\de-10\)",
        r"beta = 0 reproduces B_c\(0\) = 5\.13126764",
        r"converging as r_max\^-3  rel 2\.\de-07 -> 2\.\de-10 -> 2\.\de-10 at 100, 1e3, 1e4 r_c",
        r"s/4 = 2\.251",
        r"SUMMARY: 20/20 checks passed",
    ]),
    ("numerics/dk_throat_matching.py", [], 420, [
        # Paper sec. 4.6, app. D.3: the phi_* plateau over the
        # ladder window and the asymptotic T_c exponent 3.63 (A -> 4 pi); the
        # ladder exponent is read from dk_throat.py [T3].
        r"phi_\* over the ladder window 0\.25 <= nu <= 0\.6: 0\.216 to 0\.245",
        r"A_eff -> 4 pi  12\.56\d vs 12\.566",
        r"T_c exponent: asymptotic A/\(2 sqrt 3\) = 4 pi/\(2 sqrt 3\) = 3\.63",
        r"k1 = -d\(b/a\)/d\(nu\^2\) = 0\.4988",
        # [4] (app. D.3): the ladder's nu in one frame, 5e-3 at 1e4 to 6e-7 at 1e11
        r"\[PASS\] \[4\] prediction error \(one frame\) decreases monotonically",
        r"1e\+04     1\.8959    3\.87e-01   0\.52224   0\.51973   4\.8e-03",
        r"\[PASS\] \[4\] top-rung error < 1e-06  6\.2e-07",
        r"17/17 checks passed",
    ]),
    ("numerics/tb_phase_diagram.py", [], 5, [
        # Paper section 5.3 and its table: T_c at fixed B falls to 0.20
        # of its probe value at alpha_L and to 0.017 at alpha = 0.55; the probe
        # T_c/sqrt(B) = 0.141; the ladder top is alpha = 0.591.
        r"7/7 checks passed",
        r"0\.0000\s+crystal\s+5\.13127\s+50\.6436\s+1\.4052e-01",
        r"computed range alpha <= 0\.5908",
        r"T_c\(alpha=0\.3722\) / T_c\(0\) at equal B = 0\.197",
        r"T_c\(alpha=0\.5500\) / T_c\(0\) at equal B = 0\.01678",
    ]),
    ("numerics/dk_moduli_ladder.py", [], 600, [
        # Paper sec. 5.2, app. D.5: the moduli scan repeated on the beta ladder -- tau* = rho at
        # every rung, and the Bravais strip is bounded below iff the Gaussian
        # average I(beta) of the kernel is positive, which fails at alpha_I =
        # 0.4940, well above the zero of K_0, alpha = 0.3506.
        r"tau\* = rho at all 18 rungs \(max \|tau\*-rho\| = \d\.\de-0[89]\)",
        r"ev_min/C_tri from 7\.6050 \(beta = 1\) to 6\.379\d \(beta_K0 = 3305\.4",
        r"relative margin \(C_sq-C_tri\)/C_tri = 0\.2429 \(beta = 1\), 0\.2064 \(beta_K0\), 0\.2021 \(beta = 1e4\)",
        r"secant zero at beta_I = 15690\d\d\.\d, alpha_I = 0\.4940",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/dk_throat_channels.py", [], 420, [
        # Paper sec. 5.1, sec. 5.2: the metric sector at alpha_*.  The channel masses and exponents
        # come from the closed form, the sign change and the selection gap from
        # the throat BVP built on the PRODUCTION dk_response rows.
        r"0\.0000\s+0\.00000\s+5\.33333\s+1\.00000\s+2\.51661",
        r"gamma_0 = 0\.8465",
        r"C_sq - C_tri = \+0\.00063668",
        r"relative \+0\.176562",
        # [C10] (table 4): the second method, shooting from both ends
        r"0\.84652653\d shooting, 0\.8465266\d\d FD",
        r"relative gap \+0\.1765622 \(",
        r"\[PASS\] \[C10\] S_shoot vs S_FD",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    # --- uniform response, second variation, threshold, dynamics ----------
    ("derivations/lll_second_variation.py", [], 5, [
        # Paper sec. 5.3, app. C.5: the long-wavelength density stiffness is K_0 + 2 C_tri, and the
        # envelope functional drops exactly the satellites.
        r"Delta F4 -> \(delta\^2/4\) \(K_0 \+ 2C\) as q -> 0",
        # Paper sec. 5.3, app. D.5: the same for every Bravais lattice tau; stationarity at i, rho;
        # the stripe-edge cancellation of K_0.
        r"\[PASS\] \[7c\] tau lattice: lambda_\+\(0\+\) = 2\(K_0 \+ 2C\(tau\)\)",
        r"\[PASS\] \[7e\] each stabiliser orbit has one gamma and zero total gradient",
        r"\[PASS\] \[7g\] K_0/2 \+ C\(i tau2\)",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/dk_lattice_stiffness.py", [], 1200, [
        # Paper sec. 5.3, app. D.5: every Bravais lattice -- square density zero, critical points of
        # C only at rho and i, the square lattice unstable at X at every rung.
        r"alpha_sq = 0\.37666\d\d \+/- 8e-09",
        r"\[PASS\] \[L4\] C has critical points only at rho and i at every rung with I > 0",
        r"\[PASS\] \[L4\] the square lattice is Bogoliubov-unstable at X",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/dk_bogoliubov.py", [], 900, [
        # Paper sec. 5.3, app. C.5: triangular stable over the zone from beta = 0 to alpha_L;
        # probe-limit torus; alpha_L with K_0 = K_{G=0}.
        r"lambda_-\(M\): torus \+0\.01858\d, zone formula \+0\.01858\d",
        r"alpha_c = 0\.3721799\d \+/- 9e-09",
        r"1\.5e-15 C4 over 48",
        r"12/12 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/dk_uniform.py", [], 420, [
        # The full run -- ladder, K_{G=0} = K(s -> 0) on every rung, and
        # alpha_L with its named error budget (paper sec. 5.3, app. D.5).
        r"\[PASS\] \[7'\] K_G0 = lim_\{s->0\} K\(s\) of the G != 0 kernel on every rung",
        r"alpha_L = 0\.3721800 \+/- 1e-08, beta_L = 6510\.48\d",
        # C_tri by finite differences in the budget (table 4)
        r"delta alpha_L from C_tri: collocation vs finite differences\s+6\.3e-12",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("derivations/brane_flows.py", [], 10, [
        # Paper sec. 4.2, sec. 4.3: general-d flows, alpha_*(d), irrelevant exponents; the
        # horizon series start of the LLL mode used by onset_scan.
        r"w2 = B_m\*\(3\*B_m - 2\*epsilon \+ 36\)/192",
        r"(\d+)/\1 checks passed",
    ]),
    ("derivations/dk_charged_dynamics.py", [], 15, [
        # Paper sec. 4.5, app. B.2: omega^2 is real; onset only through omega = 0.
        r"onset only through omega = 0",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/t0_threshold.py", [], 60, [
        # Paper sec. 4.2, sec. 4.3: the T = 0 threshold mode has no zero for d = 4, 5, 6.
        r"5\s+1\.854102\s+0\s+-0\.350252",
        r"6\s+2\.098780\s+0\s+-0\.996391",
        # app. D.2: d = 3 finite-T onset crosses 8/3 at T/sqrt(B) = 7.4e-4;
        # the d = 4 throat ratio at rho_*, against the r chart
        r"crosses 8/3 at T/sqrt\(B\) = 7\.42e-4  q = 2\.98858887\d+, T/sqrt\(B\) = 7\.4214\d+e-04",
        r"-0\.03500036\d vs -0\.03500038\d",
        r"(\d+)/\1 checks passed",
    ]),
    ("numerics/dk_charged_qnm.py", [], 200, [
        # Paper sec. 4.5, app. B.2: one mode crosses at omega = 0; QNM slope vs perturbation theory
        # (fixed beta), d ln alpha_c/d ln beta and the fixed-alpha slope; the
        # probe slope from its formula; the argument-principle count.
        r"J_2/w_0\(u_H\)\^2 = 0\.335197668773",
        r"beta =\s+1\s+B_0 =\s+6\.002974\s+0\.30664094\s+0\.30664119\s+0\.756767\s+0\.23205572",
        r"beta =\s+45\s+B_0 =\s+16\.060791\s+0\.17359969\s+0\.17359988\s+0\.267793\s+0\.04648886",
        r"beta =\s+10000\s+B_0 =\s+161\.166354\s+0\.04190540\s+0\.04190545\s+0\.075539\s+0\.00316548",
        r"beta =\s+1e-06\s+B = 1\.05 B_2\s+#\{B_k < B\} = 3\s+winding \[3\.0, 3\.0\]",
        r"beta =\s+10000\s+B = 1\.05 B_2\s+#\{B_k < B\} = 3\s+winding \[3\.0, 3\.0\]",
        r"worst \|Re omega\| = 1\.2\de-11",
        r"95/95 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    # The scratch copy holds the package only, so the display check is pointed
    # at the real paper source.
    ("derivations/paper_response_display.py",
     [str(PROJECT_ROOT / "paper" / "appendices" / "C-response-system.tex")], 10, [
        # Paper app. C.3, app. D.4: every displayed equation of paper app. C against the generated
        # rows and an independent linearisation; the dressing identity.
        r"X_c - X_ab = \(P_dress - P\)/2 on dk_kernel\.npz: PASS",
        r"(\d+)/\1 checks passed",
    ]),
    ("numerics/dk_kernel_gauge_check.py", [], 10, [
        # Paper app. C.3, app. D.4: K is gauge independent; the weight 1/2 is what gauge invariance requires.
        r"(\d+)/\1 checks passed",
    ]),
    # --- scripts whose checks now fail loudly ------------------------------
    ("numerics/bc_alpha.py", [], 330, [
        # Paper sec. 2.2, sec. 3.1: the onset curve by three solvers to beta = 45, exact source
        # extraction (agreement ~3e-10).
        r"13/13 checks passed",
        r"deviates 10% at alpha ~ 0\.0787",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/g0_thermo.py", [], 60, [
        # the G = 0 fixed-T sector, the small-beta anchor of the brane code.
        r"13/13 checks passed",
    ]),
    ("numerics/dk_background.py", [], 60, [
        # Paper sec. 2.2, sec. 3.1: two methods to beta = 50, constraint on the solutions, the
        # small-beta limits delta s/s -> beta/4 and I_d -> pi^2/48.
        r"22/22 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/dk_selection.py", [], 40, [
        # Paper sec. 5.2, app. D.5: beta = 0 gap against the probe lattice code; no crossing to 45.
        r"\[1\] beta = 0 gap: C_sq - C_tri = \+0\.004537443\d+ vs probe \+0\.004537443\d+ \(rel 1\.4e-08\)",
        r"NO CROSSING on the grid",
    ]),
    ("numerics/dk_long_wavelength.py", [], 360, [
        # Paper sec. 5.1, sec. 5.3: the numbers on the ladder (zero of K_0, quartic coefficient).
        # [K7] (table 4): the cells by finite differences on every rung.
        r"\[PASS\] \[K7\] .* max \|dC_tri\| = 1\.1e-09, max \|dC_sq\| = 1\.5e-09",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/dk_throat_lattice.py", [], 90, [
        # The throat abelian kernel; S(0) exact (paper app. C.4).
        r"S\(0\) = 1\.89804005",
        r"(\d+)/\1 checks passed",
    ]),
    ("derivations/dk_throat_lattice.py", [], 10, [r"(\d+)/\1 checks passed"]),
    ("numerics/k0_zero_uncertainty.py", [], 10, [
        # alpha_K0 from the exact K_0 = K_{G=0}, method spread as the error
        # (paper sec. 5.3, table 3).
        r"RESULT  alpha_K0 = 0\.3505596 \+/- 9e-09",
    ]),
    ("numerics/alpha_i_uncertainty.py", [], 600, [
        # Paper app. D.5: the stripe-edge coupling relocated by fresh solves (~5 min);
        # app. D.5 quotes 0.4939747 +/- 2e-7.
        r"RESULT  alpha_I = 0\.4939747 \+/- 2e-07",
        r"PASS \[5\]",
        # The zero with the finite-difference kernel (table 4), R(1600, 3200)
        r"M = 1600/3200: .* quadratic zero alpha_I = 0\.4939747\d\d \(-8\.9e-09\)",
        r"PASS \[6\]",
    ]),
    ("derivations/d_continuation.py", [], 60, [
        # the reduced equations, fixed point and deformation for real d.
        r"(\d+)/\1 checks passed",
    ]),
    ("numerics/d3_crossover_why.py", [], 120, [
        # the crossover bound state leaves at d_crit, two charts.
        r"d_crit = 3\.8374197",
        r"no eps in \[0, 1\] changes the sign",
        r"15/15 checks passed",
    ]),
    ("numerics/probe_bc_precise.py", [], 20, [
        # The reference probe B_c u_H^2 behind table 4's collocation error,
        # by two methods (the paper quotes 8 decimals; the 30 digits are extra).
        r"B_c u_H\^2 = 5\.1312676376822361164",
        r"3/3 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/d3_blind_check.py", [], 900, [
        # Paper sec. 4.2, sec. 4.3: the independent d = 3 check -- the crossover bound state moves
        # the T = 0 threshold above the AdS2 BF value; its own d = 4 anchor.
        r"alpha_c\(3\) = 3\.0370100956",
        r"d = 4 anchor: alpha_c\(4\) = 2/3",
        # app. D.2: table 4: three methods to 3e-14; app. D.2: Rayleigh-Ritz bounds 3.03, 3.0370;
        # the finite-T crossing of 8/3
        r"agree to 3e-14 \(spread \d\.\de-14\)",
        r"K = 2: Rayleigh quotient 1\.40591940995\d  =>  alpha_c\(3\) > 3\.035501\d+",
        r"K = 8: Rayleigh quotient 1\.40557053\d+  =>  alpha_c\(3\) > 3\.0370081\d+",
        r"T/sqrtB = 7\.421418e-04 \(shooting\)",
    ]),
    ("numerics/onset_scan.py", [], 1200, [
        # Paper sec. 4.2, sec. 4.3: ladder vs horizon shooting (app. D.2-D.3); alpha at exactly
        # 1e24, 1e45, 1e66; direct local slopes; matching secants; refinement.
        r"alpha 1\.2e-10, B_c 5\.8e-11, b_h 1\.8e-09",
        r"beta = 1e\+24: alpha = 0\.64529318\d+",
        r"beta = 1e\+45: alpha = 0\.65909097\d+",
        r"beta = 1e\+66: alpha = 0\.66260829\d+",
        r"spread of A over beta in \[1e7, 1e11\]: 2\.02%; over \[1e8, 1e11\]: 0\.60%",
        r"150\.000   10\.27714\d",
        r"ln beta = 100: direct secant A = 9\.4356\d, matching A_eff = 9\.4357\d",
        # [7] (app. D.2): the matching condition against shooting beyond the ladder
        r"\[PASS\] \[7\] .* max \|d alpha\| = 1\.\de-11",
        # the local slopes in one frame (app. D.3): 1.5e-8 at 40, < 1e-9 beyond
        r"relative slope differences 1\.5e-08, [1-9]\.\de-10, [1-9]\.\de-10",
        r"(\d+)/\1 checks passed",
    ]),
    ("derivations/onset_scan_d.py", [], 110, [
        # Paper sec. 4.3, app. D.2: general-d raw-frame horizon data, V = O(eps), the zero mode from
        # Yang-Mills, the source c0 = w + r w'/(d-2), the frame map.
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/onset_scan_d.py", ["--quick"], 740, [
        # Paper sec. 4.3, app. D.2: finite-T onset for d = 5, 6 below alpha_* to eps = 1e-60, two
        # methods, d = 4 and d = 3 anchors (the full run, ~25 min, writes the
        # cache checked in test_onset_scan_d.py).
        r"alpha/alpha_\* 3\.254e-04 \.\.\. 0\.99861272",
        r"alpha/alpha_\* 3\.274e-04 \.\.\. 0\.99895529",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/dk_uniform.py", ["--quick"], 45, [
        # Paper sec. 5.1, sec. 5.3: K_{G=0} grid + anchors (the full run above adds the ladder and
        # the secant for alpha_L; its cache is checked in
        # test_critical_coupling_results.py).  K_{G=0} against K(s -> 0) at
        # beta = 45: paper app. C.3 (6e-9 C4).
        r"beta= 45\.0: K_G0 = 0\.4254779323, K\(s->0\) = 0\.4254779467",
        r"(\d+)/\1 checks passed",
    ]),
    # --- slow rows not covered elsewhere -----------------------------------
    ("derivations/dk_throat_channels.py", [], 1800, [
        # Paper sec. 5.1, sec. 5.2: the metric sector on the throat in closed form -- the H_zz
        # decoupling, the constant (H_yy, b) mass matrix and m_pm.  Slow:
        # SymPy on the full response rows (well over the ~5 min its
        # docstring says).
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/poster_throat.py", [], 120, [
        # Paper figure 2: the throat geometry; re-solves four rungs (~100 s).
        # The beta = 1e8 row: ln r_c, beta e^{-4V} at r_c and its ratio to 6.
        r"1e\+08   4\.157    0\.7538     0\.1256",
        r"saved .*throat_geometry\.pdf",
    ]),
    ("numerics/dk_throat_channels.py", ["--ladder"], 900, [
        # Check [C9] (paper sec. 5.1, sec. 5.2): the production brane kernel continued up the ladder at
        # fixed gamma approaches the throat kernel from above (app. D.5).
        r"\[PASS\] \[C9\]",
        r"(\d+)/\1 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    ("numerics/dk_throat_lattice.py", ["--ladder"], 300, [
        # Check [T6] (paper app. C.4): the brane ladder's near-horizon geometry approaches BTZ.
        r"S\(0\) = 1\.89804005",
        r"\[PASS\] \[T6\]",
        r"(\d+)/\1 checks passed",
    ]),
    ("numerics/bc_shift.py", [], 30, [
        # Paper sec. 3: the O(alpha) shift of B_c, perturbation theory vs direct slope.
        r"5/5 checks passed",
        r"ALL CHECKS PASSED",
    ]),
    # --- the probe-limit mains under scripts/ -------------------------------
    ("../scripts/critical_field.py", [], 45, [
        # B_c u_H^2 = 5.13126764 (paper sec. 3.1, table 4) by shooting, the tower,
        # and B_c/T^2 = 50.64 (sec. 2.2: B_c = 50 T^2).
        r"Critical field:  B_c u_H\^2 = 5\.13126764",
        r"= 50\.6436 T\^2",
        r"ALL CHECKS PASSED",
    ]),
    ("../scripts/spectral_check.py", [], 5, [
        r"N =  120:  B_n u_H\^2 = 5\.13126764  22\.48156859  51\.20975989  91\.41056318",
        r"ALL CHECKS PASSED",
    ]),
    ("../scripts/exchange_kernel.py", [], 5, [
        # C4 (app. E.4), the Stieltjes form and complete monotonicity
        r"J2 \(normalised\) = 1\.0000000000,  C4 = 1\.5805931\d*",
        r"X\(s\)/s at s = .*\(rel \d\.\de-0[6-9]\)",
        r"(\d+)/\1 checks passed",
    ]),
    ("../scripts/lattice_scan.py", [], 10, [
        # app. E.4 table: C(tau) without the G = 0 term and the margin +0.2460
        r"G != 0 part: C\(triangular\) = 0\.0184443, C\(square\) = 0\.0229818, \(sq - tri\)/tri = \+0\.2460",
        r"(\d+)/\1 checks passed",
    ]),
    ("../scripts/lll_identity.py", [], 5, [
        # app. E.2: the LLL identity and the Abrikosov ratios
        r"beta\(square\)     = 1\.1803406",
        r"beta\(triangular\) = 1\.1595953",
        r"LLL identity and Abrikosov ratios verified",
    ]),
    ("../scripts/nonlinear_reduction.py", [], 10, [r"All reduction checks passed"]),
    ("../scripts/symbolic_check.py", [], 10, [
        r"matches  f w'' \+ \(f' - f/u\) w' \+ B w = 0  up to overall factor: True",
    ]),
]

IDS = [s[0] + (" " + " ".join(s[1]) if s[1] else "") for s in SCRIPTS]


@pytest.fixture(scope="session")
def scratch_package(tmp_path_factory):
    """A throwaway copy of the package, built once for the whole session.

    `paths.py` resolves data and figures from the package's own location, so a
    script run against this copy reads the committed caches (they are copied
    in) and writes its output here, leaving the working tree clean.  `figures/`
    is created empty rather than copied: several scripts save into it and would
    fail on a missing directory, but none of them reads a figure back.
    """
    root = tmp_path_factory.mktemp("package")
    shutil.copytree(PACKAGE, root / "backreaction",
                    ignore=shutil.ignore_patterns("__pycache__", "figures"))
    (root / "backreaction" / "figures").mkdir(exist_ok=True)
    # scripts/ resolves its caches and figures/ from its own location
    # (script_paths.py), so a copy writes into the scratch tree
    shutil.copytree(PROJECT_ROOT / "scripts", root / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"))
    _CACHE_MTIMES[root] = {p: p.stat().st_mtime_ns for p in _caches(root)}
    return root


# scratch root -> {cache path: mtime right after the copy}
_CACHE_MTIMES = {}


def _caches(root):
    return sorted([*(root / "backreaction" / "data").glob("*.npz"),
                   *(root / "scripts").glob("*.npz")])


def _env(scratch_package):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(scratch_package), str(scratch_package / "scripts"),
         env.get("PYTHONPATH", "")])
    env["MPLBACKEND"] = "Agg"
    return env


@pytest.mark.slow
@pytest.mark.parametrize("module,argv,seconds,patterns", SCRIPTS, ids=IDS)
def test_script_runs_and_prints_its_published_numbers(
        scratch_package, module, argv, seconds, patterns):
    script = (scratch_package / "backreaction" / module).resolve()
    assert script.exists(), f"{module} is missing from the scratch copy"

    proc = subprocess.run([sys.executable, str(script), *argv],
                          capture_output=True, text=True, cwd=PROJECT_ROOT,
                          env=_env(scratch_package), timeout=seconds * 6,
                          check=False)
    assert proc.returncode == 0, (
        f"{module} exited {proc.returncode}\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr tail ---\n{proc.stderr[-3000:]}")

    missing = [p for p in patterns if not re.search(p, proc.stdout)]
    assert not missing, (
        f"{module} ran clean but no longer prints: {missing}\n"
        f"--- stdout tail ---\n{proc.stdout[-3000:]}")


@pytest.mark.slow
def test_a_run_leaves_the_committed_tree_untouched(scratch_package):
    """The isolation itself, asserted rather than assumed.  If the PYTHONPATH
    trick ever stopped working, every script above would silently start
    rewriting the committed .npz caches and figures, and the tests would still
    pass -- the worst possible failure mode for this file.  So: run the
    cheapest script that saves a figure, and the cheapest main-tree script that
    rewrites a cache, and check that the mtimes of the real caches, the
    committed figure PDFs and the PNGs did not move."""
    def snapshot():
        return {p: p.stat().st_mtime_ns
                for p in sorted([*(PACKAGE / "data").glob("*.npz"),
                                 *(PACKAGE / "figures").glob("*.png"),
                                 *(PACKAGE / "figures").glob("*.pdf"),
                                 *(PROJECT_ROOT / "scripts").glob("*.npz"),
                                 *(PROJECT_ROOT / "figures").glob("*.png")])}

    before = snapshot()
    kinds = {p.suffix for p in before}
    assert {".npz", ".png", ".pdf"} <= kinds, f"nothing to watch of {kinds} -- vacuous guard"

    for script, wrote in (
        (scratch_package / "backreaction" / "numerics" / "tb_phase_diagram.py",
         scratch_package / "backreaction" / "figures" / "tb_phase_diagram.png"),
        (scratch_package / "scripts" / "exchange_kernel.py",
         scratch_package / "scripts" / "kernel.npz"),
    ):
        mtime = wrote.stat().st_mtime_ns if wrote.exists() else None
        proc = subprocess.run([sys.executable, str(script)], capture_output=True,
                              text=True, cwd=PROJECT_ROOT, env=_env(scratch_package),
                              timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr[-2000:]
        # It wrote its output into the scratch tree, not the package.
        assert wrote.exists() and wrote.stat().st_mtime_ns != mtime, wrote
    assert snapshot() == before, "a scratch run wrote into the committed tree"


def _regenerated_caches(scratch_package):
    """(scratch, committed) pairs for every .npz a scratch run rewrote: those
    whose mtime moved since the scratch copy was made."""
    copied = _CACHE_MTIMES[scratch_package]
    return [(new, PROJECT_ROOT / new.relative_to(scratch_package))
            for new in _caches(scratch_package)
            if copied.get(new) != new.stat().st_mtime_ns
            and (PROJECT_ROOT / new.relative_to(scratch_package)).exists()]


@pytest.mark.slow
def test_regenerated_caches_match_the_committed_ones(scratch_package):
    """Every cache the scripts above rewrote in the scratch tree agrees with
    the committed one.  Runs last (pytest keeps file order), so it sees all the
    rewrites; with a `-k` selection it compares whatever that selection wrote.

    Byte identity holds only on the reference platform (macOS arm64), so the
    comparison is numerical: each numeric array to 1e-7 of its own largest
    entry, plus 1e-10 absolute so that residual arrays at rounding level do not
    fail on noise.  Text entries (readmes) are not compared.
    """
    pairs = _regenerated_caches(scratch_package)
    bad = []
    for new, old in pairs:
        with np.load(new, allow_pickle=True) as a, np.load(old, allow_pickle=True) as b:
            if set(a.files) != set(b.files):
                bad.append(f"{old.name}: keys {sorted(set(a.files) ^ set(b.files))} differ")
                continue
            for key in b.files:
                x, y = a[key], b[key]
                if not (np.issubdtype(y.dtype, np.number) or y.dtype == bool):
                    continue
                if x.shape != y.shape:
                    bad.append(f"{old.name}[{key}]: shape {x.shape} != {y.shape}")
                    continue
                if y.size == 0:
                    continue
                x, y = x.astype(complex), y.astype(complex)
                finite = np.isfinite(y)
                if not (np.isfinite(x) == finite).all() or not np.array_equal(
                        x[~finite], y[~finite], equal_nan=True):
                    bad.append(f"{old.name}[{key}]: non-finite entries differ")
                    continue
                if not finite.any():
                    continue
                scale = float(np.max(np.abs(y[finite])))
                diff = np.abs(x[finite] - y[finite])
                if np.max(diff) > 1e-7 * scale + 1e-10:
                    bad.append(f"{old.name}[{key}]: max |diff| {np.max(diff):.2e} "
                               f"at scale {scale:.2e}")
    assert not bad, "regenerated caches differ from the committed ones:\n" + "\n".join(bad)
