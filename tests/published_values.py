"""The published numbers, in one place.

The entries indexed in PAPER_QUOTED below are the values a reader of the paper
will see, with where they appear; the tests assert that the code still
produces them.  Changing a number here changes a published result.

Tolerances are written per-assertion at the site of use, matched to the number
of digits actually quoted: asserting more precision than is published would
make the suite fragile without protecting anything.

The entries outside PAPER_QUOTED are regression values: intermediate results,
cross-checks and a few first-order estimates the exact computation goes
beyond, kept so that a change in the code that would alter them is still
caught.  They are not claims of the paper.  Comments name the paper section
that states a result where there is one.
"""

# -- linear instability -------------------------------------------
# The condensation tower B_n u_H^2, d = 4, u_H = 1.  Both the shooting and the
# spectral method reproduce these to all quoted digits.
B_TOWER = [5.13126764, 22.48156859, 51.20975989, 91.41056318]
B_C = B_TOWER[0]

# -- the triangular lattice --------------------------------------
C4 = 1.5805931  # contact coefficient, J2 = 1
C_TRIANGULAR = 0.8087409  # C(tau) at the triangular point
C_SQUARE = 0.8132783  # C(tau) at the square point
LATTICE_GAP_PERCENT = 0.561  # (C_sq - C_tri)/C_tri, per cent
CONDENSATION_ENERGY_RATIO = 1.005611  # F_tri/F_sq = C_sq/C_tri
EFFECTIVE_RATIO_TRIANGULAR = 1.0233  # 2 C_tri / C4
EFFECTIVE_RATIO_SQUARE = 1.0291  # 2 C_sq  / C4
SCREENING_AT_FIRST_SHELL = 0.854  # X/C4 at sigma_1 = 4 pi/sqrt 3
# The same two lattices without the tau-independent G = 0 term, C - C4/2: the
# convention of the finite-alpha kernel and of the paper's
# appendix E table.
C_TRIANGULAR_G_NONZERO = 0.0184443
C_SQUARE_G_NONZERO = 0.0229818
PROBE_RELATIVE_MARGIN = 0.2460  # (C_sq - C_tri)/C_tri in that convention

# Pure Ginzburg-Landau reference (bare Abrikosov ratios), quoted for contrast
#: the holographic discrimination is ~70% screened relative to these.
ABRIKOSOV_BETA_TRIANGULAR = 1.1595953  # paper app. E.2 quotes 7 digits
ABRIKOSOV_BETA_SQUARE = 1.1803406

# -- the locked Ginzburg parameter ------------------------------
# Per spacetime dimension d (AdS_{d+1}).
UNIVERSALITY = {
    #     B_c u_H^2,    B_c/T^2,  C4 (J2=1), c_0^2/C4, chi(sigma_1), kappa_eff
    3: dict(
        B_c=2.22339814,
        B_over_T2=39.01,
        C4=1.6667,
        lowest_pole_weight=0.946,
        chi_shell=0.859,
        kappa_eff=0.7629,
    ),
    4: dict(
        B_c=5.13126764,
        B_over_T2=50.64,
        C4=1.5806,
        lowest_pole_weight=0.901,
        chi_shell=0.854,
        kappa_eff=0.7653,
    ),
    6: dict(
        B_c=13.11031859,
        B_over_T2=57.51,
        C4=1.4738,
        lowest_pole_weight=0.844,
        chi_shell=0.851,
        kappa_eff=0.7666,
    ),
}

# -- the lattice at finite alpha -------------------------------------
# The exact kernel on the D'Hoker-Kraus magnetic brane, and the lattice
# selection it implies.  beta = alpha B^2 runs to 45 (alpha to 0.17445).
DK_C4_AT_ZERO_BETA = 1.58059313  # anchor [1]: beta -> 0 reproduces the probe limit
DK_PROBE_GAP = 0.004537443622742641  # external reference value, computed by no script here: C_sq - C_tri at alpha = 0 from a separate perturbative code
DK_GAP_AT_ZERO_BETA = 0.00453744  # the brane computation of the same gap
DK_GAP_MINIMUM = 0.00450822  # the dip, at alpha = 0.068 (beta = 4)
DK_GAP_FINAL = 0.00461964  # at alpha = 0.174, 1.8% above the probe value
DK_BETA_MAX = 45.0

# ===========================================================================
# The O(alpha) gravitational backreaction of the lattice.
# ===========================================================================

# -- the G = 0 sector at fixed temperature ------------------------
# Per alpha B^2 the first four are closed forms proved in derivations/g0.py [9];
# the per-alpha-rho^2 entries are proved in closed form.
G0_DS_B2 = 0.25  # delta s / s_0 |_T, exact
G0_ID_B2 = 0.2056167584  # I_d(u_H) = pi^2 / 48, exact
G0_S4_B2 = (0.375, 0.25)  # (s_4x, s_4z) = (3/8, 1/4), exact
G0_LOG_B2 = (-0.5, 0.0)  # u^4 log u coefficients: a genuine resonance
G0_ID_R2 = -3.9119463021
G0_DS_R2_TOL = 2e-12  # delta s/s_0|_T = -B_c to 1.7e-12 relative

# -- the O(alpha) shift of the critical field ---------------------
DBC_PER_ALPHA = 28.332285  # delta B_c = this * alpha, u_H = 1
DBC_SLOPE = 0.20970443  # delta B_c / B_c per alpha B_c^2 u_H^4
DBC_TWO_METHOD_TOL = 1e-5  # PT integral vs direct perturbed eigenvalue

# -- the exact onset B_c(alpha) on the brane ----------------------
# T = 1/pi units; the curve is monotone and convex, so the linear-in-alpha
# approximation is already 10% off (relative to B_c) at alpha = 0.0787
# (BC_TANGENT_TEN_PERCENT_ALPHA, a spline root; the first sweep point past it
# is 0.084).
BC_OF_ALPHA = {0.02: 5.741, 0.05: 6.846, 0.10: 9.393}
BC_AT_BETA_MAX = 16.061  # alpha = 0.17445, beta = 45
BC_ALPHA_SLOPE0 = 0.20970  # tangent; the O(alpha) shift gives 0.20970443
DK_ENTROPY_SLOPE = 0.25  # delta s/s_0|_T -> beta/4 as beta -> 0
DK_ID_SLOPE = G0_ID_B2  # I_d -> pi^2/48 as beta -> 0

# -- the full one-flux-quantum moduli scan ------------------------
# tau = rho = exp(i pi/3) is the minimiser at every beta, to the optimiser's
# tolerance; the well stays strictly convex and both diagnostics turn at the
# same alpha.
MODULI_TAU_TOL = 1e-8  # |tau* - rho| over the whole grid
MODULI_HESS_MIN = 0.140927  # smallest Hessian eigenvalue over the grid
MODULI_HESS_AT_ZERO = 0.142173  # its probe value
# max |lam2 - lam1| / min lam1 over the grid (the Z3 prediction is 0; limited by
# the h = 1e-3 differences), as dk_moduli.py prints it.  Paper table 4: 1.4e-6.
MODULI_HESS_ISOTROPY = 1.36e-6  # regression value; app. D.5 says only "to the floor of its finite-difference evaluation"
MODULI_GAP_DIP_RATIO = 0.9936  # gap at the beta = 4 dip, over the probe gap
MODULI_GAP_FINAL_RATIO = 1.0181  # and at beta = 45
MODULI_BETA_TURNING = 4.0
# The normalisation-free ratios (beta = 45 row; the probe end is
# PROBE_RELATIVE_MARGIN): the margin over the square falls monotonically, and
# the Hessian over C_tri by 8%.
MODULI_MARGIN_FINAL = 0.2268
MODULI_CURVATURE_RATIO_AT_ZERO = 7.708
MODULI_CURVATURE_RATIO_FINAL = 7.063

# -- the AdS_3 throat and the critical coupling -------------------
ALPHA_STAR = 2.0 / 3.0  # instability survives T -> 0 iff alpha < this
THROAT_L3SQ = 1.0 / 3.0  # the AdS_3 radius^2, D'Hoker-Kraus
THROAT_INVARIANT = 6.0  # beta e^{-4V} at the fixed point
# alpha_*(d) = 16 / (d(d-1)(d-2)); the d = 3 entry is Wong's published
# gamma_* = 3/4 (arXiv:1307.7839) via alpha = 2/gamma -- an external anchor.
ALPHA_STAR_BY_D = {3: 8 / 3, 4: 2 / 3, 5: 4 / 15, 6: 2 / 15}
WONG_GAMMA_STAR = 0.75
# The throat ladder: beta -> (alpha, b_h, beta e^{-4V(r_p)}).
THROAT_LADDER = {
    45.0: (0.174453, 1.510320, 3.5815),
    1e4: (0.384992, 1.272739, 5.6127),
    1e6: (0.486821, 1.163399, 5.9302),
    1e8: (0.544490, 1.105389, 5.98774),
    1e10: (0.578775, 1.073055, 5.99786),
    1e11: (0.590759, 1.062226, 5.999106),
}
THROAT_BKT_A = (
    7.450  # nu = A / (ln beta + d): the LADDER fit (a plateau; A -> 4 pi)
)
THROAT_BKT_D = 4.535
# the T_c exponent read directly, ln beta = s/sqrt(alpha_* - alpha) + c on
# the same nine rungs (beta >= 1e7), T_c ~ sqrt(B) e^{-(s/4)/sqrt(alpha_*-alpha)}.
# A/(2 sqrt 3) = 2.15 assumed nu^2 = (alpha_*-alpha)/(2 alpha_*), which on these
# rungs undershoots nu by 4.5-8.5%.  dk_throat.py [T3]; the scan's own points
# in the same window give 2.2503 (onset_scan.npz).
THROAT_TC_EXPONENT = 2.2511

# -- the excited tower is an ordinary SL spectrum -----------------
# I = int_{r_p}^oo e^{-V}/sqrt(U) dr, the Liouville length; the tower obeys
# sqrt(B_n) I / pi = n + C/n with NO constant term (endpoint indices
# mu_hor = 0, mu_bnd = 1 -- the sharp test of the derivation).
TOWER_I = {
    1.0: 1.23545994,
    45.0: 0.84908624,
    1e4: 0.34465378,
    1e6: 0.14188847,
    1e9: 0.03400431,
    1e11: 0.01260298,
}
TOWER_C = {  # the measured 1/n coefficient
    1.0: -0.027827,
    45.0: +0.058886,
    1e4: +0.347611,
    1e6: +0.739965,
    1e9: +1.579512,
    1e11: +2.304872,
}
TOWER_D1 = {  # d_1: the ground state is NOT in the law
    1.0: -0.0365,
    45.0: +0.0831,
    1e4: +0.3927,
    1e6: +0.7098,
    1e9: +1.2214,
    1e11: +1.5732,
}
TOWER_C0_MAX = 1.8e-4  # worst |constant offset|, in n units
TOWER_FIT_RESID_MAX = 2.2e-5  # worst residual of d_n = C/n, in n units
TOWER_I_RC_SLOPE = 0.57723  # d(I r_c)/d(ln r_c), against l3 = 1/sqrt3
# N_throat = nu ln(r_c)/pi, the throat's capacity at the MARGINAL eigenvalue,
# < 1 at every rung.  The fitted-law supremum A/(4 pi) = 0.593 below is what the
# ladder fit extrapolates to; the true limit is 1, since A -> 4 pi.
TOWER_N_THROAT = {45.0: 0.1145, 1e4: 0.3083, 1e6: 0.3868, 1e9: 0.4444, 1e11: 0.4672}

# -- the marginality coefficient derived -------------------------
# Matching condition nu ln(rho_*/h) = chi(nu) + phi_*(nu); chi exact (BTZ
# hypergeometric), phi_* from the T = 0 brane; no free parameter.
# b/a of the nu = 0 outer solution rho w = a + b ln(rho/rho_*), referenced at
# rho_* (e^{2V} doubled); the ratio depends on the reference point (-0.0257 at
# rho = 1 in the B = 1 chart).  Quoted to seven digits.  With the brane's
# deformation resolved by the integrator it is -0.035000389, = the domain-wall
# chart's -0.035000360 to 8e-7.
MATCHING_LOG_RATIO = -0.0350004
# tan phi_* = |b/a|/nu + k1 nu + O(nu^3); k1 = -d(b/a)/d(nu^2) by central
# differences in nu^2 and from the cos-fitted phi_* table (dk_throat_matching.py [3])
PHI_STAR_K1 = 0.49875
MATCHING_RHOSTAR_OFFSET = 0.0313  # ln(rho_*/h) - (1/4) ln(beta/6) at large beta
# |nu_pred/nu_meas - 1| on the ladder, with (M)'s throat exponent converted to
# the ladder's nu = sqrt(b_h - 1) by b_h = (1 + nu^2) sqrt(1 - eps/6) (dk_throat_matching
# [4]); falls like eps^1.5.  Unconverted, the O(eps) frame difference dominates:
# 7.2e-2, 2.0e-2, 5.3e-3, 1.3e-3, 6.4e-4 at the same rungs.
MATCHING_ERR = {1e4: 4.81e-3, 1e6: 3.73e-4, 1e8: 2.79e-5, 1e10: 2.00e-6, 1e11: 6.15e-7}
MATCHING_A_EFF = {25: 7.436, 100: 9.436, 400: 11.950, 3200: 12.563}  # -> 4 pi
MATCHING_IRRELEVANT_P = (19.0 / 3.0) ** 0.5 - 1.0  # = Delta_+ - 2 at lambda = 0

# [W8] the level gap CLOSES: B_2/B_1 falls monotonically towards 1, because
# 1 + d_1 = sqrt(B_c) I/pi grows like ln(r_c)/pi.  The 1.38 at beta = 1e11 is
# a value on a trend, not a floor.
TOWER_LEVEL_GAP = {
    1.0: 4.2438,
    45.0: 3.5137,
    1e4: 2.4511,
    1e6: 1.9265,
    1e9: 1.5231,
    1e11: 1.3799,
}
# The two phase functionals: the throat enters the bottom of the spectrum
# through sqrt(b-1) = nu (the AdS_3 BF subtraction) and the top through
# sqrt(b).  Their ratio diverges as b_h -> 1, i.e. exactly at alpha_*.
TOWER_PHASE_RATIO_TOP = 4.1316  # sqrt(b_h/(b_h-1)) at beta = 1e11

# -- the lattice at alpha_* (the throat limit) ----------
# Paper sec. 5.2 and apps. C.4, D.5 and E.4.  Abelian sector alone:
# the BTZ x R^2 kernel shape S(0) (C.4) and the square/triangle margin +0.250
# (E.4).  Full kernel with the metric sector: sign change at gamma_0 and margin
# +0.1766 (5.2), from direct solves at the exact shells, +0.176562; a PCHIP
# interpolant on 41 nodes gives +0.17652; the interior maximum 0.051 is a regression value, not in the
# paper.  The brane ladder converges onto the throat kernel
# at fixed gamma, worst deviation 0.009 at beta = 1e8.
THROAT_S0 = 1.89804
THROAT_ABELIAN_MARGIN = 0.250
THROAT_GAMMA0 = 0.8465
THROAT_FULL_MARGIN = 0.176562
THROAT_KERNEL_PEAK = 0.051
THROAT_LADDER_WORST_AT_1E8 = 0.009

# -- second methods for the single-route numbers ------------------
# The second methods of table 4.  The full throat kernel by shooting from both ends (dk_throat_channels.py
# S_shoot, check [C10]) against the FD solver.  gamma_0 by direct roots of both
# (the PCHIP root of the 41-node interpolant, 0.846523842, is 2.7e-6 low); the
# exact-shell margin by both.  S_shoot is the more accurate: its start, cutoff,
# matching point, tolerance and quadrature each move it by < 5e-10; the FD
# (N = 3000) error is the deviation below, and doubling N moves it onto S_shoot.
THROAT_GAMMA0_SHOOT = 0.846526531
THROAT_GAMMA0_FD = 0.846526597
THROAT_FULL_MARGIN_SHOOT = 0.1765622
THROAT_SHOOT_FD_DEV = {0.02: 6.5e-7, 0.3: 8.4e-8, "shells": 1.5e-8, "shell_sums": 1.45e-9}  # max |S_shoot - S_FD|; shell sums |C_tri, C_sq shoot - FD|
# The matching inputs from a second T = 0 brane (dk_throat_matching.py [3'],
# [4']): phi_* on the domain-wall-gauge brane (t0_threshold.Flow, R = int e^A as
# an ODE) against the r chart over 0.25 <= nu <= 0.6, and the beta -> oo limit
# of ln(rho_*/h) - (1/4) ln(beta/6) from the T = 0 brane alone (scaling
# symmetries), in both charts; the ladder (0.031461 at 1e8 ... 0.031328 at
# 1e11) decreases monotonically onto it.  MATCHING_RHOSTAR_OFFSET above is the
# paper's 0.031, i.e. this limit.
PHI_STAR_TWO_CONSTRUCTIONS = 2.6e-10  # max |phi_dw - phi_r| on the window
MATCHING_OFFSET_T0 = 0.0313232  # r chart 0.031323212, DW 0.031323214
OFFSET_TWO_CHARTS_DEV = 1.4e-9  # |offset_t0 - offset_t0_dw| = 1.386e-9 (table 4)
MATCHING_OFFSET_TOP_RUNG = 0.0313278  # beta = 1e11
# The G != 0 kernel on the ladder by a second discretisation: dk_kernel.BraneFD
# (sparse finite differences on a uniform sigma grid), Richardson M = 1600/3200,
# at the shells of C_tri, C_sq on every rung (dk_long_wavelength.py [K7]); max
# |C_FD - C_colloc| (absolute), worst at beta = 1e8, <= 6e-11 up to 1e6.  It
# shares the background, w0 and the generated rows with collocation.
LADDER_FD_AGREEMENT = (1.12e-9, 1.50e-9)  # (C_tri, C_sq)
ALPHA_L_BUDGET_FD = 6.3e-12  # dk_uniform.py [8]: alpha_L shift from C_tri by FD (alpha_L 0.372179975 unchanged)
ALPHA_I_FD_SHIFT = {800: -5.3e-8, 1600: -8.9e-9}  # alpha_I(FD R(M,2M)) - alpha_I(colloc), alpha_i_uncertainty.py [6]
# The onset beyond the ladder by the matching condition (onset_scan.py [7]):
# nu L = chi(nu) + phi_*(nu), phi_* evaluated on the T = 0 brane, L with the T = 0
# offset; alpha = (2/3)/(1 + nu^2)^2 against horizon shooting at 1e24, 1e45, 1e66
# (a cubic spline of the phi_* table is off by 2.3e-4 at nu = 0.055, so [7]
# evaluates phi_* on the T = 0 brane directly).  The local slope by matching:
# relative difference at ln beta = 40, 77.5, 150, with the matching's throat
# exponent converted to the horizon nu = sqrt(b_h - 1) of the shooting slope
# through b_h = (1 + nu^2) sqrt(1 - eps/6).
ONSET_MATCH_ALPHA_DEV = 1.2e-11
ONSET_MATCH_SLOPE_DEV = {40.0: 1.5e-8, 77.5: 4.1e-10, 150.0: 9.5e-10}  # unconverted: 7.6e-5, 8.1e-11, 9.5e-10

# -- the moduli scan on the ladder rungs -------------------------
# The functional is bounded below on Bravais lattices iff the Gaussian average
# I(beta) of the kernel is positive; I changes sign at alpha_I.  Top-rung
# (beta = 1e8) margin over the square: 0.183.
ALPHA_I = 0.4940
BETA_I = 1.569e6
LADDER_TOP_MARGIN = 0.183

# -- the zero of the long-wavelength kernel K_0 (alpha = 0.351) --
# K_0(beta) = lim_{s->0} K(s; beta) of the exact kernel along the dk_throat
# ladder, computed as K_{G=0} by collocation of the G = 0 system.  A
# Richardson stencil in gamma = 0.02, 0.04, 0.08 is biased by up to 1.2e-5 C4
# and puts the zero at beta = 3305.7, alpha_K0 = 0.350563.  Its zero is where the
# long-range part of the vortex interaction turns attractive.  It is not an
# instability: the density stiffness is K_0 + 2 C_tri, so the lattice
# holds to ALPHA_L below.
# The gamma = 0.3 proxy kernel crosses later; the two are different numbers.
K0_ZERO_ALPHA = 0.3505596  # zero of K_0(beta): beta = 3305.42, B_c = 97.103, +-9e-9
K0_ZERO_BETA = 3305.42
K03_ZERO_ALPHA = 0.427175  # zero of K(gamma = 0.3; beta): beta = 49464
LADDER_K0_OVER_C4 = {  # K_0/C4 along the ladder; sign change between 3e3 and 1e4
    1.0: +0.723908,  # K_{G=0}/C4; the stencil reads up to 1.2e-5 high
    45.0: +0.177654,
    1e3: +0.026732,
    3e3: +0.001803,
    1e4: -0.017112,
    1e6: -0.049317,
    1e8: -0.057517,
}
LADDER_C_TRI_MIN = 0.018598  # C_tri > 0 at every rung (min at beta = 1); grows up the ladder
LADDER_GAP_MIN = 0.004509  # C_sq - C_tri > 0 at every rung (min at beta = 5)
ALPHA_L_FIRST_ORDER = 0.37217997  # zero of K_0/2 + C_tri on the dk_long_wavelength ladder (= ALPHA_L);

# -- the density stiffness and the Bogoliubov spectrum ---------
# Long-wavelength density stiffness K_0 + 2 C_tri; Bogoliubov spectrum over the
# magnetic BZ; the lattice is stable against every LLL perturbation below ALPHA_L.
BOGO_STIFFNESS_5000 = 0.004279  # (K_0 + 2 C_tri)/C4 at beta = 5000 (alpha = 0.36397); 0.004280 with the stencil K_0
BOGO_MODULATED_5000 = 0.004279  # modulated Abrikosov state, M -> infinity, +-4e-7
BOGO_TORUS_AGREEMENT = 1.53e-15  # 48-eigenvalue (24-flux-quantum) torus Hessian vs block formula at beta = 5000, alpha = 0.363965 (dk_bogoliubov.npz torus_5000_dev; app. C.5)

# -- the strict uniform response ---------------------------------
KG0_PROBE = 1.580593127  # beta = 0: K_{G=0} = C4
KG0_OALPHA_SLOPE = -19.847153392  # -Pi_0/(2 alpha) as alpha -> 0, per alpha rho^2
KG0_OALPHA_REFERENCE = -19.847153839  # independent AdS5-Schwarzschild (g0_thermo + bc_shift)
KG0_AT_45 = 0.4254779323  # K_{G=0}(beta = 45); K(s->0) = 0.4254779467; regression value, not in the paper
ALPHA_L = 0.37217997  # zero of K_{G=0}/2 + C_tri = K_0/2 + C_tri: lattice instability AND first-order point
ALPHA_L_UNC = 1.1e-8  # error budget: collocation vs shooting 1.5e-10, vs s->0 extrapolation 1.1e-8, rest < 1e-10.
# The paper quotes the rounded-up bound +-2e-8 (PAPER_BOUNDS).
BETA_L = 6510.485
BC_AT_ALPHA_L = 132.2605  # B_c u_H^2 there
KG0_LADDER_EXT_DEV = 7.8e-9  # max |K_{G=0} - K(s->0)|/C4 on the 13 rungs 1e2..1e8 (cubic, gamma = 1e-3..8e-3)
F4_AT_TOP_RUNG = -0.024849  # (K_0/2 + C_tri)/C4 at beta = 1e8 (alpha = 0.54449)

# -- every Bravais lattice, and the probe-limit spectrum ---------
# Density stiffness of lattice tau is K_0 + 2 C(tau) (derivation [7]).  The
# square lattice keeps it to ALPHA_SQ, elongated lattices further, tending to
# alpha_I; but C is stationary only at rho and i, i is a saddle, the square
# lattice is Bogoliubov-unstable at X at every coupling, and above ALPHA_L no
# Bravais lattice is stable against all LLL perturbations.
ALPHA_SQ = 0.3766629  # zero of K_0/2 + C_sq: beta = 7546.37, B_c u_H^2 = 141.544, +-8e-9
BOGO_PROBE_M = 0.018581  # lambda_-(M)/C4 of the triangular lattice at beta = 0 (zone and torus)
BOGO_PROBE_K = 0.260773  # lambda_-(K)/C4 at beta = 0
BOGO_PROBE_TORUS_MIN = 0.000839  # lowest non-zero eigenvalue, 36-flux-quantum torus, beta = 0

# -- alpha_* exact at T = 0; dense onset scan ----------------------
T0_THRESHOLD_BA = -0.0350004  # throat b/a of the d = 4 threshold mode, no node
T0_THRESHOLD_BA_D5 = -0.350252
T0_THRESHOLD_BA_D6 = -0.996391
FLOW_EXPONENTS = {4: 1.516611, 5: 1.854102, 6: 2.098780}
ONSET_SCAN_ALPHA_END = 0.66263  # alpha at beta ~ 1.8e66
# minimum local slope d ln beta / d(1/nu), at ln beta ~ 21.4: the direct
# derivative (central differences in ln eps about a solve, Richardson).  The
# derivative of the scan grid (spacing 0.95 in ln beta) gives 7.427 at 21.2.
ONSET_SCAN_BKT_MIN = 7.4257
# direct local slopes A(ln beta) (onset_scan.py [4]; paper app. D.3)
ONSET_SLOPE_DIRECT = {
    16.118: 7.57546,  # beta = 1e7
    18.421: 7.46504,  # beta = 1e8
    21.5: 7.42577,
    25.328: 7.47062,  # beta = 1e11
    40.0: 7.88995,
    77.5: 8.92544,
    100.0: 9.42552,
    150.0: 10.27714,  # 18% below 4 pi
}
LADDER_SLOPE_SPREAD_DIRECT = {1e7: 0.0202, 1e8: 0.0060}
# alpha at exactly these beta (app. D.2; the dense scan's nearest point,
# beta = 1.1e24, gives 0.6454).  At 1e24 onset_scan gives 0.6452931814; an
# external reference integration (LSODA, not shipped) gave 0.6452931813.
ONSET_ALPHA_AT_BETA = {1e24: 0.6452932, 1e45: 0.6590910, 1e66: 0.6626083}
# d = 3: alpha_c(T) = 8/3 at q = alpha B^2/2 = D3_CROSS_Q (horizon units),
# T/sqrt(B) = D3_CROSS_T (three methods; app. D.2)
D3_CROSS_Q = 2.98858887
D3_CROSS_T = 7.4214e-4
# d = 3: Rayleigh-Ritz lower bounds on alpha_c(3), basis z^{k+1}(1-z)^{-3/8},
# K terms, exact Beta-function integrals
D3_RITZ_ALPHA_BOUND = {2: 3.0355011, 8: 3.0370082}
D3_ALPHA_C = 3.0370100956  # d = 3 true T = 0 critical coupling; BF value 8/3 is only sufficient

# -- finite-T onset coupling for d = 5, 6 -----------------------------
# onset_scan_d.npz: alpha(beta) strictly increasing, s = B_m/B_thr - 1 > 0, no node,
# from the probe limit to eps = 1e-60 (beta = alpha (B u_H^2)^2, u_H = d/(4 pi T))
ONSET_D_ALPHA_MAX_OVER_STAR = {5: 0.99861272, 6: 0.99895529}  # at the top of the scan
ONSET_D_BETA_MAX = {5: 1.510e132, 6: 1.046e117}
ONSET_D_TWO_METHOD_MAX = 2e-9  # |dB_m|/B_m and |dv_inf|/v_inf, shooting vs Chebyshev
ONSET_D_REFINE_MAX = 1e-11  # largest relative movement of alpha, beta under refinement

# -- the onset is static ----------------------------------------------
QNM_DLN_ALPHAC_DLN_BETA = {1.0: 0.756767, 45.0: 0.267793, 1e4: 0.075539}

# -- the supersymmetric coupling ----------------------------------
ALPHA_SUSY = 0.25  # Romans N=4+ SU(2): kappa^2/ghat^2 = L^2/4, inside the window

# -- the N = 2 Ward-identity anchor for C_J/C_T ------------------
# Exact rationals (Fraction), asserted exactly: they are convention bookkeeping
# on protected quantities, not numerics.
from fractions import Fraction as _F  # noqa: E402

CJ_CT_SUSY = _F(3, 80)  # C_J(SU(2)_R)/C_T: free N=4 SYM, N=2 Ward identities, and (3/20)*alpha_SUSY
CJ_N4_PER_ADJOINT_PI4 = _F(3, 8)  # C_J(SU(2)_R) * pi^4 per adjoint index of N=4 SYM
CT_N4_PER_ADJOINT_PI4 = _F(10, 1)  # C_T * pi^4 per adjoint index (= 40c, c = 1/4)
CJ_SU2R_OVER_C_PI4 = _F(3, 2)  # C_J(SU(2)_R) = (3/2) c/pi^4 in every N=2 SCFT
CR_U1R_OVER_C_PI4 = _F(12, 1)  # C_r(U(1)_r) = 12 c/pi^4;  C_r/C_{I_3} = 8, also from the LPT Lagrangian
CR_N1_OVER_CT = _F(1, 10)  # N=1 R-current: C_R/C_T = 1/10 (Osborn 9808041 eq 11.12 over C_T = 40c/pi^4)

# -- numbers the paper quotes that had no entry ---------------------------
# Each names the cache or script it comes from; the cached ones are asserted in
# tests/backreaction/test_critical_coupling_results.py, the printed-only ones
# by the slow-tier scrape (tests/backreaction/test_scripts.py).
KERNEL_LONGWAVE_OVER_C4 = {0.0: 0.981876, 45.0: 0.180445}  # K/C4 at s = 0.02 B_c (dk_kernel.npz; sec. 5.1)
T0_THRESHOLD_BA_D3 = 0.1226413  # d = 3 threshold c_1/c_0, one node (t0_threshold.npz; table 2)
D_CRIT = 3.8374197  # real-d zero of the threshold ratio (d3_crossover_why.npz; not quoted in the paper)
D_CRIT_CHART_DIFF = 2.8e-10  # its two charts (d3_crossover_why.npz d_crit_methods)
BC_TANGENT_TEN_PERCENT_ALPHA = 0.0787  # where the alpha = 0 tangent of B_c is 10% off (bc_alpha.npz; sec. 3.3)
PHI_STAR_PLATEAU = (0.215622, 0.245466)  # phi_* over the table nodes 0.26 <= nu <= 0.6, i.e. those of the window 0.25 <= nu <= 0.6 (phi_* = 0.2472 at nu = 0.25; dk_throat_matching.npz; sec. 4.6, app. D.3)
K0_ZERO_BC = 97.103  # B_c u_H^2 at the zero of K_0 (dk_long_wavelength.npz; table 3)
BC_AT_ALPHA_I = 1782.26  # B_c u_H^2 at alpha_I (dk_moduli_ladder.npz; not quoted in the paper)
QNM_WORST_RE = 1.22e-11  # worst |Re omega| of a growing mode, dk_charged_qnm.py [Q2] (app. B.2)
# the probe B_c u_H^2 to 16 digits (probe_bc_precise.py: Frobenius/Wronskian
# and mpmath Chebyshev, agreeing to 2e-49); bc_alpha's beta = 0 anchors (shooting,
# collocation N = 40..120) recover it to 4.5e-12 (worst, N = 120)
PROBE_BC_PRECISE = 5.131267637682236
PROBE_BC_RECOVERED_DEV = 4.5e-12
ALPHA_I_PRECISE = 0.4939747  # fresh solves, alpha_i_uncertainty.py (app. D.5 quotes 0.4939747 +- 2e-7)
ALPHA_I_PRECISE_UNC = 2e-7  # its budget after the switch to K_0 = K_{G=0}
ONSET_SCAN_BETA_END = 1.814e66  # top of the dense onset scan (onset_scan.npz)
ONSET_SCAN_BH_MIN = 1.003038  # min of b_h over the onset scan, at its top (onset_scan.npz)
KERNEL_NONMONOTONE_BETA = 8.0  # first grid beta where K(s) has an interior maximum in s (dk_kernel.npz)
# The function Montgomery's theorem needs, f(gamma) = e^{-gamma/2} K(gamma B_c; beta)
# (app. E), on the dk_kernel.npz grid, by collocation (K) and finite differences
# (K_fd) alike: convex on every rung beta <= 3, concave at small gamma from
# beta = 4, so not completely monotone there; still decreasing up to beta = 25,
# rising at gamma <= 0.075 from beta = 30.  Second divided differences of f/C4 at
# beta = 4 (collocation; finite differences -0.0836, -0.0585, -0.0319):
WEIGHTED_KERNEL_CONCAVE_BETA = 4.0
WEIGHTED_KERNEL_D2_AT_4 = {0.035: -0.0826, 0.05: -0.0581, 0.075: -0.0317}
WEIGHTED_KERNEL_NONMONOTONE_BETA = 30.0
ONSET_LADDER_TWO_METHOD = (1.2e-10, 5.8e-11)  # ladder vs horizon shooting: alpha, B_c (onset_scan.py stdout; table 4)
ONSET_D_CHEB_AGREEMENT = 8.0e-10  # d = 5, 6: max |dB_m|/B_m, shooting vs Chebyshev (onset_scan_d.npz m2_d5/m2_d6)
ONSET_D_D4_DEV = 1.16e-10  # the general-d code against the d = 4 ladder (onset_scan_d.npz d4_ladder_dev)
ONSET_D_D3_ANCHOR_DEV = 5.5e-12  # ... against exact RN-AdS4 at d = 3 (onset_scan_d.npz d3_anchor)
KERNEL_FD_AGREEMENT = 6.2e-6  # max |K - K_fd|, collocation vs finite differences (dk_kernel.npz; table 4)
THROAT_TAIL_IDENTITY_DEV = 1.95e-13  # app. C.2 tail identity (dk_throat_lattice.npz identity_dev; table 4)
THROAT_S_GREEN_AGREEMENT = 6.5e-8  # S(gamma): finite differences vs Green's function (dk_throat_lattice.npz; app. D.5, table 4)
X_MINUS_XAB_MAX = 0.6118  # max of X - X_ab over the scan (dk_kernel.npz X_coupled - X); regression value, not quoted in the paper
KG0_TWO_METHOD = 1.11e-10  # max |K_G0 collocation - shooting|/C4, grid and ladder at shooting start delta = 1e-10 (dk_uniform.npz; table 4)
K0_AT_45 = 0.4254779467  # K(s -> 0) of the G != 0 kernel at beta = 45, printed by dk_uniform.py [5]; regression value
DK_C4_AT_45 = 2.3949845  # brane-normalised C4 at beta = 45 (dk_uniform.npz); regression value
DK_PI_OVER_C4_S0_AT_45 = 1.10507  # P(s -> 0)/C4 at beta = 45, cubic in gamma (dk_kernel.npz Pi_bare; app. C.3)
ALPHA_L_BUDGET = {  # dk_uniform.npz budget_keys/budget_vals (app. D.5)
    "K_G0 collocation vs shooting": 1.49e-10,  # shooting start delta = 1e-10
    "K_G0 vs s->0 limit of the G != 0 kernel": 1.08e-8,
}
ALPHA_L_BUDGET_REST_MAX = 7.8e-11  # largest of the remaining entries (the secant residual)
ALPHA_L_SECANT_F4 = 2.0e-11  # |K_0/2 + C_tri|/C4 at the secant's end (dk_uniform.npz F4_fo)
LATTICE_TORUS_DEV = 1.83e-15  # block formula vs rectangular-torus Hessian (dk_lattice_stiffness.npz); regression value
BOGO_PROBE_TORUS_DEV = 5.3e-15  # block formula vs 36-flux-quantum torus, beta = 0 (dk_bogoliubov.npz; app. C.5)
TC_RATIO_AT_055 = 0.01678  # T_c(alpha = 0.55)/T_c(0) at equal B, printed by tb_phase_diagram.py (sec. 5.3)
BETA_ONE_ALPHA = 0.0277503  # alpha on the beta = 1 rung (dk_throat.npz; sec. 2.2)
ONSET_SCAN_BKT_MIN_LNBETA = 21.5  # the scan mark where the local slope is smallest (onset_scan.npz)

# -- the index of numbers the paper quotes ---------------------------------
# name -> where in the paper.  tests/test_published_values_index.py checks that
# every name exists above.
PAPER_QUOTED = {
    "B_C": "section 3.1, table 4",
    "UNIVERSALITY": "section 2.2 (B_c = 50 T^2 for d = 4)",
    "C4": "appendix E.4",
    "C_TRIANGULAR_G_NONZERO": "appendix E.4",
    "C_SQUARE_G_NONZERO": "appendix E.4",
    "PROBE_RELATIVE_MARGIN": "appendix E.4, section 5.2",
    "ABRIKOSOV_BETA_TRIANGULAR": "appendix E.2",
    "ABRIKOSOV_BETA_SQUARE": "appendix E.2",
    "DK_GAP_AT_ZERO_BETA": "appendix D.5",
    "BC_AT_BETA_MAX": "section 3.3, appendix D.2",
    "MODULI_MARGIN_FINAL": "section 5.2",
    "ALPHA_STAR": "abstract, sections 1 and 4",
    "ALPHA_STAR_BY_D": "section 4.3",
    "WONG_GAMMA_STAR": "appendix D.2 (section 4.3 gives it as alpha_*(3) = 8/3)",
    "THROAT_L3SQ": "section 4.1",
    "THROAT_LADDER": "sections 1, 3.3, 4.1, 4.6, 5.1, 5.2, appendix D.2, figures 1 and 2",
    "THROAT_BKT_A": "appendix D.3",
    "THROAT_TC_EXPONENT": "section 4.6 (eq. for the effective T_c law), appendix D.3",
    "MATCHING_LOG_RATIO": "section 4.6, appendix D.2",
    "MATCHING_RHOSTAR_OFFSET": "appendix D.3",
    "MATCHING_ERR": "appendix D.3",
    "ALPHA_I": "section 5.2, appendix D.5",
    "ALPHA_I_PRECISE": "appendix D.5",
    "ALPHA_I_PRECISE_UNC": "appendix D.5",
    "QNM_WORST_RE": "appendix B.2",
    "PROBE_BC_RECOVERED_DEV": "table 4",
    "LADDER_TOP_MARGIN": "section 5.2",
    "THROAT_S0": "appendix C.4",
    "THROAT_ABELIAN_MARGIN": "appendix E.4",
    "THROAT_GAMMA0": "sections 5.1, 5.2, figure 5, appendix D.5",
    "THROAT_FULL_MARGIN": "section 5.2, appendix E.4",
    "THROAT_LADDER_WORST_AT_1E8": "appendix D.5",
    "K0_ZERO_ALPHA": "section 5.3, table 3, figure 6",
    "K0_ZERO_BETA": "section 5.3, table 3",
    "K0_ZERO_BC": "table 3",
    "LADDER_K0_OVER_C4": "section 5.3 (K_0/C_4 saturates near -0.06)",
    "ALPHA_L": "abstract, sections 1, 5, 5.3, table 3, figure 6, appendix D.5",
    "ALPHA_L_UNC": "appendix D.5",
    "ALPHA_L_BUDGET": "appendix D.5",
    "ALPHA_L_BUDGET_REST_MAX": "appendix D.5",
    "ALPHA_L_SECANT_F4": "appendix D.5",
    "BETA_L": "appendix D.5, table 3",
    "BC_AT_ALPHA_L": "section 5.3, table 3, appendix D.5",
    "F4_AT_TOP_RUNG": "section 5.3",
    "ALPHA_SQ": "section 5.3",
    "BOGO_PROBE_M": "section 5.3",
    "BOGO_PROBE_TORUS_DEV": "table 4",
    "BOGO_TORUS_AGREEMENT": "table 4",
    "KERNEL_LONGWAVE_OVER_C4": "section 5.1",
    "KG0_LADDER_EXT_DEV": "section 5.1, table 4",
    "KG0_OALPHA_SLOPE": "table 4 (its ratio to the reference)",
    "KG0_OALPHA_REFERENCE": "table 4 (its ratio to the slope)",
    "KG0_TWO_METHOD": "table 4",
    "DK_PI_OVER_C4_S0_AT_45": "appendix C.3",
    "KERNEL_FD_AGREEMENT": "table 4",
    "THROAT_TAIL_IDENTITY_DEV": "table 4",
    "THROAT_S_GREEN_AGREEMENT": "table 4, appendix D.5",
    "T0_THRESHOLD_BA": "sections 4.2, 4.4, 4.6, table 2, appendices D.2, D.3",
    "T0_THRESHOLD_BA_D3": "table 2",
    "T0_THRESHOLD_BA_D5": "table 2, appendix D.2",
    "T0_THRESHOLD_BA_D6": "table 2, appendix D.2",
    "ONSET_SCAN_ALPHA_END": "section 4.2, appendix D.2",
    "ONSET_SCAN_BETA_END": "sections 1, 4.2, appendix D.2",
    "ONSET_SCAN_BH_MIN": "section 4.2, appendix D.2",
    "KERNEL_NONMONOTONE_BETA": "sections 5.1, 5.2",
    "WEIGHTED_KERNEL_CONCAVE_BETA": "sections 5.2, 6",
    "ONSET_SCAN_BKT_MIN": "appendix D.3",
    "ONSET_SCAN_BKT_MIN_LNBETA": "appendix D.3",
    "ONSET_SLOPE_DIRECT": "appendix D.3",
    "ONSET_ALPHA_AT_BETA": "appendix D.2",
    "ONSET_LADDER_TWO_METHOD": "table 4",
    "ONSET_D_ALPHA_MAX_OVER_STAR": "appendix D.2",
    "ONSET_D_BETA_MAX": "appendix D.2",
    "ONSET_D_CHEB_AGREEMENT": "table 4",
    "ONSET_D_D4_DEV": "table 4",
    "ONSET_D_D3_ANCHOR_DEV": "table 4",
    "D3_ALPHA_C": "sections 1, 4.3, 4.4, tables 1 and 2, appendix D.2",
    "D3_RITZ_ALPHA_BOUND": "appendix D.2",
    "D3_CROSS_T": "section 4.3, appendix D.2",
    "BC_TANGENT_TEN_PERCENT_ALPHA": "section 3.3",
    "BETA_ONE_ALPHA": "section 2.2",
    "PHI_STAR_PLATEAU": "section 4.6, appendix D.3",
    "TC_RATIO_AT_055": "section 5.3",
    "ALPHA_SUSY": "section 4.4, appendix A.2",
    "CJ_CT_SUSY": "section 4.4, appendix A.2",
}

# -- the strings the paper typesets ------------------------------------------
# name -> (string as it appears in paper/*.tex, value it must round from).
# tests/test_published_values_index.py checks that each string is in the paper
# source and that the stored value rounds to it at the printed precision, so
# a number changed on one side only fails.  Values that are a function of a
# stored entry (ratios, conversions) are given as that function.
_PI = 3.141592653589793

def _tc_ratio(bc):
    """T_c/T_c^probe at fixed B, (B_c(0)/B_c)^(1/2) (table in sec. 5.3)."""
    return (B_C / bc) ** 0.5

def _ladder_bc(beta):
    """B_c u_H^2 on a rung, sqrt(beta/alpha) (app. D.2 table)."""
    return (beta / THROAT_LADDER[beta][0]) ** 0.5

PAPER_STRINGS = {
    "THROAT_FULL_MARGIN": ("0.1766", THROAT_FULL_MARGIN),  # sec. 5.2, app. E.4; exact shells
    "TC_EXPONENT_LADDER": ("2.25", THROAT_TC_EXPONENT),  # sec. 4.6; direct fit in alpha
    "B_C": ("5.13126764", B_C),
    "B_C_SHORT": ("5.13", B_C),  # sec. 2.2, table 3
    "C4": ("1.5805931", C4),
    "C_TRIANGULAR_G_NONZERO": ("0.0184443", C_TRIANGULAR_G_NONZERO),
    "C_SQUARE_G_NONZERO": ("0.0229818", C_SQUARE_G_NONZERO),
    "PROBE_RELATIVE_MARGIN": ("0.2460", PROBE_RELATIVE_MARGIN),  # app. E.4
    "PROBE_RELATIVE_MARGIN_SHORT": ("0.246", PROBE_RELATIVE_MARGIN),  # sec. 5.2, app. E.4
    "ABRIKOSOV_BETA_TRIANGULAR": ("1.1595953", ABRIKOSOV_BETA_TRIANGULAR),
    "ABRIKOSOV_BETA_SQUARE": ("1.1803406", ABRIKOSOV_BETA_SQUARE),
    "DK_GAP_AT_ZERO_BETA": ("0.00453744", DK_GAP_AT_ZERO_BETA),
    "THROAT_BKT_A": ("7.45", THROAT_BKT_A),
    "MATCHING_RHOSTAR_OFFSET": ("0.031", MATCHING_RHOSTAR_OFFSET),
    "ALPHA_I": ("0.494", ALPHA_I),
    "ALPHA_I_PRECISE": ("0.4939747", ALPHA_I_PRECISE),  # app. D.5
    "THROAT_S0": ("1.89804", THROAT_S0),
    "THROAT_GAMMA0": ("0.8465", THROAT_GAMMA0),
    "THROAT_ABELIAN_MARGIN": ("0.250", THROAT_ABELIAN_MARGIN),  # app. E.4
    "THROAT_LADDER_WORST_AT_1E8": ("0.009", THROAT_LADDER_WORST_AT_1E8),  # app. D.5
    "K0_ZERO_ALPHA": ("0.3506", K0_ZERO_ALPHA),
    "K0_ZERO_ALPHA_SHORT": ("0.351", K0_ZERO_ALPHA),  # sec. 5.3, table 3, figure 6
    "K0_ZERO_BETA": ("3305", K0_ZERO_BETA),
    "K0_ZERO_BETA_TABLE": ("3.31\\times10^3", K0_ZERO_BETA),  # table 3
    "K0_ZERO_BC": ("97.1", K0_ZERO_BC),  # table 3
    "TC_RATIO_AT_K0_ZERO": ("0.23", _tc_ratio(K0_ZERO_BC)),  # table 3
    "LADDER_K0_TOP": ("0.06", -LADDER_K0_OVER_C4[1e8]),  # sec. 5.3; "near -0.06"
    "ALPHA_L": ("0.3721800", ALPHA_L),  # app. D.5; with the error budget
    "ALPHA_L_SHORT": ("0.3722", ALPHA_L),
    "ALPHA_L_THREE": ("0.372", ALPHA_L),  # abstract, sec. 1, sec. 5.3, table 3, figure 6
    "BC_AT_ALPHA_L": ("132.3", BC_AT_ALPHA_L),
    "BC_AT_ALPHA_L_D5": ("132.26", BC_AT_ALPHA_L),  # app. D.5
    "BETA_L": ("6510.5", BETA_L),  # app. D.5
    "BETA_L_TABLE": ("6.51\\times10^3", BETA_L),  # table 3
    "TC_RATIO_AT_ALPHA_L": ("0.20", _tc_ratio(BC_AT_ALPHA_L)),  # table 3
    "TC_PROBE_OVER_SQRT_B": ("0.141", 1 / (_PI * B_C**0.5)),  # figure 1; the right axis
    "F4_AT_TOP_RUNG": ("0.025", -F4_AT_TOP_RUNG),  # sec. 5.3
    "ALPHA_SQ": ("0.3767", ALPHA_SQ),  # sec. 5.3
    "BOGO_PROBE_M": ("0.0186", BOGO_PROBE_M),  # sec. 5.3
    "KERNEL_LONGWAVE_PROBE": ("0.98", KERNEL_LONGWAVE_OVER_C4[0.0]),  # sec. 5.1
    "KERNEL_LONGWAVE_45": ("0.18", KERNEL_LONGWAVE_OVER_C4[45.0]),  # sec. 5.1
    "MODULI_MARGIN_FINAL": ("0.227", MODULI_MARGIN_FINAL),  # sec. 5.2
    "LADDER_TOP_MARGIN": ("0.18", LADDER_TOP_MARGIN),  # sec. 5.2
    "DK_PI_OVER_C4_S0_AT_45": ("1.105", DK_PI_OVER_C4_S0_AT_45),  # app. C.3
    "T0_THRESHOLD_BA": ("0.0350004", -T0_THRESHOLD_BA),
    "T0_THRESHOLD_BA_SHORT": ("0.035", -T0_THRESHOLD_BA),  # sec. 4.2, sec. 4.6, app. D.3
    "T0_THRESHOLD_BA_D3": ("0.123", T0_THRESHOLD_BA_D3),  # table 2
    "T0_THRESHOLD_BA_D5": ("0.350", -T0_THRESHOLD_BA_D5),  # table 2, app. D.2
    "T0_THRESHOLD_BA_D6": ("0.996", -T0_THRESHOLD_BA_D6),  # table 2, app. D.2
    "MATCHING_LOG_RATIO": ("0.0350004", -MATCHING_LOG_RATIO),  # app. D.2; both constructions
    "MATCHING_ERR_1E4": ("5\\times10^{-3}", MATCHING_ERR[1e4]),  # app. D.3
    "MATCHING_ERR_1E11": ("6\\times10^{-7}", MATCHING_ERR[1e11]),  # app. D.3
    "TC_EXPONENT_ASYMPTOTIC": ("3.63", 4 * _PI / (2 * 3**0.5)),  # sec. 4.6
    "PLATEAU_SLOPE": ("7.2", 4 * (_PI / 2 + 0.23)),  # app. D.3; 4(pi/2 + 0.23)
    "PHI_STAR_PLATEAU_LOW": ("0.22", PHI_STAR_PLATEAU[0]),  # sec. 4.6, app. D.3
    "PHI_STAR_PLATEAU_HIGH": ("0.25", PHI_STAR_PLATEAU[1]),  # sec. 4.6, app. D.3
    "ONSET_SCAN_BKT_MIN": ("7.43", ONSET_SCAN_BKT_MIN),
    "ONSET_SLOPE_40": ("7.89", ONSET_SLOPE_DIRECT[40.0]),
    "ONSET_SLOPE_775": ("8.93", ONSET_SLOPE_DIRECT[77.5]),
    "ONSET_SLOPE_150": ("10.3", ONSET_SLOPE_DIRECT[150.0]),
    "ONSET_ALPHA_1E24": ("0.6453", ONSET_ALPHA_AT_BETA[1e24]),
    "ONSET_ALPHA_1E45": ("0.6591", ONSET_ALPHA_AT_BETA[1e45]),
    "ONSET_ALPHA_1E66": ("0.6626", ONSET_ALPHA_AT_BETA[1e66]),
    "ONSET_SCAN_ALPHA_END": ("0.6626", ONSET_SCAN_ALPHA_END),
    "ONSET_SCAN_BH_MIN": ("1.003", ONSET_SCAN_BH_MIN),  # sec. 4.2, app. D.2
    "KERNEL_NONMONOTONE_BETA": ("8", KERNEL_NONMONOTONE_BETA),  # sec. 5.1, sec. 5.2
    "WEIGHTED_KERNEL_CONCAVE_BETA": ("4", WEIGHTED_KERNEL_CONCAVE_BETA),  # sec. 5.2, sec. 6
    "ONSET_D5_ALPHA_MAX": ("0.99861", ONSET_D_ALPHA_MAX_OVER_STAR[5]),  # app. D.2
    "ONSET_D6_ALPHA_MAX": ("0.99896", ONSET_D_ALPHA_MAX_OVER_STAR[6]),  # app. D.2
    "ONSET_D5_BETA_MAX": ("1.5\\times10^{132}", ONSET_D_BETA_MAX[5]),  # app. D.2
    "ONSET_D6_BETA_MAX": ("1.0\\times10^{117}", ONSET_D_BETA_MAX[6]),  # app. D.2
    # the ladder: secs. 1, 3.3, 4.1, 4.6, 5.1, 5.2, figures 1 and 2, app. D.2
    "LADDER_ALPHA_45": ("0.1745", THROAT_LADDER[45.0][0]),
    "LADDER_ALPHA_45_SHORT": ("0.1745", THROAT_LADDER[45.0][0]),  # sec. 3.3, sec. 5.1, sec. 5.2, app. D.2
    "LADDER_ALPHA_1E4": ("0.3850", THROAT_LADDER[1e4][0]),
    "LADDER_ALPHA_1E4_SHORT": ("0.4", THROAT_LADDER[1e4][0]),  # sec. 3.3; "from 0.4 to 0.59"
    "LADDER_ALPHA_1E6": ("0.4868", THROAT_LADDER[1e6][0]),
    "LADDER_ALPHA_1E8": ("0.5445", THROAT_LADDER[1e8][0]),
    "LADDER_ALPHA_1E8_SHORT": ("0.544", THROAT_LADDER[1e8][0]),
    "LADDER_ALPHA_1E10": ("0.5788", THROAT_LADDER[1e10][0]),
    "LADDER_ALPHA_1E11": ("0.5908", THROAT_LADDER[1e11][0]),
    "LADDER_ALPHA_1E11_THREE": ("0.591", THROAT_LADDER[1e11][0]),
    "LADDER_ALPHA_1E11_TWO": ("0.59", THROAT_LADDER[1e11][0]),
    "LADDER_BH_45": ("1.5103", THROAT_LADDER[45.0][1]),
    "LADDER_BH_1E4": ("1.2727", THROAT_LADDER[1e4][1]),
    "LADDER_BH_1E6": ("1.1634", THROAT_LADDER[1e6][1]),
    "LADDER_BH_1E8": ("1.1054", THROAT_LADDER[1e8][1]),
    "LADDER_BH_1E10": ("1.0731", THROAT_LADDER[1e10][1]),
    "LADDER_BH_1E11": ("1.0622", THROAT_LADDER[1e11][1]),
    "LADDER_INV_45": ("3.58", THROAT_LADDER[45.0][2]),
    "LADDER_INV_1E4": ("5.613", THROAT_LADDER[1e4][2]),
    "LADDER_INV_1E6": ("5.930", THROAT_LADDER[1e6][2]),
    "LADDER_INV_1E8": ("5.9877", THROAT_LADDER[1e8][2]),
    "LADDER_INV_1E10": ("5.9979", THROAT_LADDER[1e10][2]),
    "LADDER_INV_1E11": ("5.9991", THROAT_LADDER[1e11][2]),
    "BC_AT_BETA_MAX": ("16.06", BC_AT_BETA_MAX),
    "LADDER_BC_1E4": ("161.2", _ladder_bc(1e4)),
    "LADDER_BC_1E6": ("1433", _ladder_bc(1e6)),
    "LADDER_BC_1E8": ("1.36\\times10^4", _ladder_bc(1e8)),
    "LADDER_BC_1E10": ("1.31\\times10^5", _ladder_bc(1e10)),
    "LADDER_BC_1E11": ("4.11\\times10^5", _ladder_bc(1e11)),
    "LADDER_BC_1E11_SHORT": ("4.1 \\times 10^5", _ladder_bc(1e11)),  # sec. 4.6
    "D3_ALPHA_C_FULL": ("3.0370100956", D3_ALPHA_C),
    "D3_ALPHA_C": ("3.0370", D3_ALPHA_C),
    "D3_ALPHA_C_SHORT": ("3.037", D3_ALPHA_C),  # sec. 1, table 1, table 2
    "D3_RITZ_BOUND": ("3.0370", D3_RITZ_ALPHA_BOUND[8]),  # app. D.2; alpha_c(3) > 3.0370
    "D3_WONG_VARIABLE": ("0.6585", 2 / D3_ALPHA_C),
    "D3_CJ_CT": ("0.253", D3_ALPHA_C / 12),
    "D3_CROSS_T": ("7.4\\times10^{-4}", D3_CROSS_T),
}

# Corrected numbers whose paper edit has not landed yet.
# The index test checks that each value rounds to its string;
# once the string is in the paper source, move the entry to PAPER_STRINGS.
PAPER_STRINGS_PENDING = {}

# -- precisions the paper states ("agree to", "stable to") ---------------------
# name -> (string in the paper, measured value).  The test requires the measured
# value not to exceed the stated one: for a mantissa (6\times10^{-9}) after
# rounding to the printed digits, for a bare power (10^{-14}) to within half a
# decade.  The string includes enough context to find the right occurrence.
PAPER_BOUNDS = {
    "QNM_IMAGINARY": ("purely imaginary to\n$2\\times10^{-11}$", QNM_WORST_RE),  # app. B.2
    "PROBE_BC_RECOVERED": ("collocation at $\\beta = 0$ & $5\\times10^{-12}$", PROBE_BC_RECOVERED_DEV),  # table 4
    "KG0_VS_K0_LADDER": ("$10^{-8}C_4$", KG0_LADDER_EXT_DEV),  # sec. 5.1, table 4
    # app. C.5: the 36-quantum probe torus and the 24-quantum torus at beta =
    # 5000 "reproduce the block spectrum to 10^{-14}" (measured 5.3e-15, 1.5e-15)
    "BOGO_PROBE_TORUS": ("36 and 24 flux quanta & $10^{-14}$", BOGO_PROBE_TORUS_DEV),  # table 4
    "BOGO_TORUS": ("tori of 36 and 24 flux quanta & $10^{-14}$", BOGO_TORUS_AGREEMENT),  # table 4
    "ONSET_LADDER_ALPHA": ("$1.2\\times10^{-10}$ in $\\alpha$", ONSET_LADDER_TWO_METHOD[0]),  # table 4
    "ONSET_LADDER_BC": ("$6\\times10^{-11}$ in $B_c u_H^2$", ONSET_LADDER_TWO_METHOD[1]),  # table 4
    "ONSET_D_CHEB": ("$8\\times10^{-10}$", ONSET_D_CHEB_AGREEMENT),  # table 4
    "ONSET_D_D4": ("exact magnetic RN--AdS$_4$ & $10^{-10}$", ONSET_D_D4_DEV),  # table 4
    "ONSET_D_D3": ("$6\\times10^{-12}$", ONSET_D_D3_ANCHOR_DEV),  # table 4
    "KERNEL_FD": ("collocation vs finite differences & $6\\times10^{-6}$", KERNEL_FD_AGREEMENT),  # table 4
    # second methods
    "THROAT_SHOOT_FD_02": ("both ends & $7\\times10^{-7}$ at $\\gamma = 0.02$", THROAT_SHOOT_FD_DEV[0.02]),  # table 4
    "THROAT_SHOOT_FD": ("$10^{-7}$ for $\\gamma \\ge 0.3$; shell", THROAT_SHOOT_FD_DEV[0.3]),  # table 4
    "THROAT_SHOOT_FD_SHELLS": ("shell sums $1.5\\times10^{-9}$", THROAT_SHOOT_FD_DEV["shell_sums"]),  # table 4
    "LADDER_FD_CELLS": ("$\\beta = 10^8$ & collocation vs finite differences & $1.5\\times10^{-9}$", max(LADDER_FD_AGREEMENT)),  # table 4
    "LADDER_FD_ALPHA_L": ("$\\alpha_L$ & collocation vs finite differences & $6\\times10^{-12}$", ALPHA_L_BUDGET_FD),  # table 4
    "LADDER_FD_ALPHA_I": ("$\\alpha_I$ & collocation vs finite differences & $9\\times10^{-9}$", abs(ALPHA_I_FD_SHIFT[1600])),  # table 4
    "ONSET_MATCH_TABLE": ("horizon shooting vs matching condition & $1.2\\times10^{-11}$", ONSET_MATCH_ALPHA_DEV),  # table 4
    "ONSET_MATCH_TEXT": ("reproduces these to $1.2\\times10^{-11}$", ONSET_MATCH_ALPHA_DEV),  # app. D.2
    "ONSET_MATCH_D3": ("horizon-shooting $\\alpha$ to $1.2\\times10^{-11}$", ONSET_MATCH_ALPHA_DEV),  # app. D.3
    "ONSET_MATCH_SLOPE_40": ("reproduces to $2\\times10^{-8}$ at\n$\\ln\\beta = 40$", ONSET_MATCH_SLOPE_DEV[40.0]),  # app. D.3
    "ONSET_MATCH_SLOPE_BEYOND": ("$\\ln\\beta = 40$ and $10^{-9}$ beyond", max(ONSET_MATCH_SLOPE_DEV[77.5], ONSET_MATCH_SLOPE_DEV[150.0])),  # app. D.3
    "PHI_STAR_DW": ("$r$ chart vs domain-wall gauge & $3\\times10^{-10}$", PHI_STAR_TWO_CONSTRUCTIONS),  # table 4
    "OFFSET_TWO_CHARTS": ("two charts & $1.4\\times10^{-9}$", OFFSET_TWO_CHARTS_DEV),  # table 4
    "THROAT_TAIL_IDENTITY": ("$2\\times10^{-13}$", THROAT_TAIL_IDENTITY_DEV),  # table 4
    "THROAT_S_GREEN": ("$7\\times10^{-8}$", THROAT_S_GREEN_AGREEMENT),  # app. D.5, table 4
    "KG0_TWO_METHOD": ("$1.1\\times10^{-10}C_4$", KG0_TWO_METHOD),  # table 4
    "ALPHA_L_UNC": ("\\alpha_L = 0.3721800 \\pm 2\\times10^{-8}$", ALPHA_L_UNC),  # app. D.5; the budget rounded up
    "KG0_SLOPE_REFERENCE": ("independent chart and gauge & $2\\times10^{-8}$", abs(KG0_OALPHA_SLOPE / KG0_OALPHA_REFERENCE - 1)),  # table 4
    "ALPHA_L_SECANT": ("< 10^{-10}C_4", ALPHA_L_SECANT_F4),  # app. D.5
    "ALPHA_L_BUDGET_SHOOT": ("$1.5\\times10^{-10}$ in $\\alpha$", ALPHA_L_BUDGET["K_G0 collocation vs shooting"]),  # app. D.5
    "ALPHA_L_BUDGET_EXT": ("$1.1\\times10^{-8}$, which dominates", ALPHA_L_BUDGET["K_G0 vs s->0 limit of the G != 0 kernel"]),  # app. D.5
    "ALPHA_L_BUDGET_REST": ("less than $10^{-10}$", ALPHA_L_BUDGET_REST_MAX),  # app. D.5
    "ALPHA_I_UNC": ("$2\\times10^{-7}$, so\n$\\alpha_I", ALPHA_I_PRECISE_UNC),  # app. D.5
}

# -- approximate statements -----------------------------------------------------
# name -> (string in the paper, value, low, high): the value must lie in
# [low, high], the range the wording ("about", "two percent") allows.
PAPER_APPROX = {
    # the coefficient in T_c is THROAT_TC_EXPONENT vs 2 pi/sqrt 3, a 38% deficit
    # (the 41% deficit of A against 4 pi is SLOPE_MIN_DEFICIT's, sec. 4.6)
    "LADDER_COEFF_DEFICIT": ("a coefficient $38\\%$ below", 100 * (1 - THROAT_TC_EXPONENT / (2 * _PI / 3**0.5)), 37.5, 38.5),  # sec. 4.6
    "SLOPE_MIN_LOCATION": ("minimum $7.43$ at $\\ln\\beta \\approx 21$", ONSET_SCAN_BKT_MIN_LNBETA, 20.5, 21.5),  # app. D.3
    "SCAN_END_BETA": ("\\beta \\approx 10^{66}", ONSET_SCAN_BETA_END, 10**65.5, 10**66.5),  # sec. 4.2, app. D.2
    "BC_OVER_T2": ("B_c \\simeq 50\\,T^2", UNIVERSALITY[4]["B_over_T2"], 47.5, 52.5),  # sec. 2.2
    "TANGENT_TEN_PERCENT": ("off at $\\alpha \\approx 0.08$", BC_TANGENT_TEN_PERCENT_ALPHA, 0.075, 0.085),  # sec. 3.3
    "LADDER_BC_DECADES": ("rises by more than three orders", _ladder_bc(1e11) / _ladder_bc(1e4), 1e3, 1e4),  # sec. 3.3
    "PLATEAU_REPRESENTATIVE": ("4(\\pi/2 + 0.23)", 0.23, *PHI_STAR_PLATEAU),  # app. D.3
    "TC_FIFTH": ("to a fifth of its probe value", _tc_ratio(BC_AT_ALPHA_L), 0.18, 0.22),  # sec. 5.3
    "BETA_ONE": ("reaches one already at $\\alpha \\approx 0.03$", BETA_ONE_ALPHA, 0.025, 0.035),  # sec. 2.2
    "TC_AT_055": ("under two percent of it at", 100 * TC_RATIO_AT_055, 0.0, 2.0),  # sec. 5.3
}

# -- quoted numbers with no stored value ----------------------------------------
# name -> (string in the paper, why it is not covered).  The test only checks
# that each string is still in the paper, so the list cannot silently go stale.
# Not listed at all: equation coefficients and exact rationals, section and
# equation numbers, cited literature values, integers that count things (23
# rungs, 41 harmonics, 24 and 36 flux quanta), and the parameters of a
# computation (w = 0.4, alpha = 0.31 and 0.37, tau = 0.3 + 1.1i, eps = 1e-24,
# tau_2 <= 5, gamma = 1e-3 to 8e-3, the plotted alpha = 0.55 and 0.62 and
# nu = 0.02, the window 0.25 <= nu <= 0.6, ln beta = 40, 77.5, 150).
PAPER_UNCOVERED = {
    "GAUGE_RADIAL": ("kernel in radial; generic gauge (app.~\\ref{app:response-metric}) & transformed solutions & $6\\times10^{-10}$", "printed by numerics/dk_kernel_gauge_check.py; no stored value"),  # table 4
    "GAUGE_GENERIC": ("transformed solutions & $6\\times10^{-10}$; $7\\times10^{-9}$", "printed by numerics/dk_kernel_gauge_check.py; no stored value"),  # table 4
    "DRESSING_IDENTITY_D4": ("$8\\times10^{-13}$", "printed by derivations/paper_response_display.py; no stored value"),  # table 4
    "CONSTRAINT_45": ("$4\\times10^{-9}$ at $\\beta = 45$", "printed by numerics/dk_throat.py [T5]; no stored value"),  # table 4
    "CONSTRAINT_TOP": ("$2\\times10^{-5}$ on the top rung", "printed by numerics/dk_throat.py; no stored value"),  # table 4
    "BACKGROUND_PROFILES": ("collocation vs shooting & $10^{-9}$", "printed by numerics/dk_background.py; no stored value"),  # table 4
    "ENTROPY_SLOPE": ("$\\delta S/S = \\beta/4 + O(\\beta^2)$ & exact & $10^{-6}$", "printed by numerics/dk_background.py; no stored value"),  # table 4
    "D3_THREE_METHODS": ("collocation, shooting, series & $3\\times10^{-14}$", "asserted by numerics/d3_blind_check.py (relative spread of alpha_c(3) 3.0e-14, asserted < 4e-14); no stored value"),  # table 4
    "D3_TWO_CONSTRUCTIONS": ("domain-wall flow vs exact background & $2\\times10^{-11}$", "asserted by numerics/t0_threshold.py [5]; no stored value"),  # table 4
    "HORIZON_FLUX": ("relative to the kernel & below $10^{-9}$", "printed by numerics/dk_kernel.py --anchors; no stored value"),  # table 4
    "PROBE_KERNEL": ("probe kernel & $3\\times10^{-12}$", "printed by numerics/dk_kernel.py --anchors; no stored value"),  # table 4
    "ALPHA_I_TWO_LOCATIONS": ("agree to $4\\times10^{-9}$, and changing", "printed by numerics/alpha_i_uncertainty.py; no stored value"),  # app. D.5
    "LLL_IDENTITY": ("three lattices & $10^{-10}$", "checked by scripts/lll_identity.py; no stored value"),  # table 4
    "PWAVE_FIRST_ORDER": ("$\\alpha \\approx 0.13$ in our variable", "a literature value (Ammon et al.) converted to alpha"),  # sec. 5.3; a footnote
    "PLATEAU_ONSET": ("$\\ln\\beta \\gg 400$", "qualitative: where nu << |c_1/c_0|; A_eff(ln beta = 400) = 11.95 in dk_throat_matching.npz"),  # sec. 4.6
}
