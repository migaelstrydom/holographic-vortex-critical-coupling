"""Machine check of the displayed equations of the paper's response appendix.

Appendix C of the paper displays the source, the five linearised rows of the
coupled metric-photon response in the gauge H_tt = H_xx = 0, the horizon
value of the by-parts flux, and the gauge-variation flux of the kernel.  This
module reads those displays straight out of the LaTeX (every equation
environment preceded by a line ``%% check: <id>``), converts them to SymPy and
compares them with independently held references:

  [1] the five rows against the generated system systems/dk_response.py
      (its 7x7 coupled matrix, read as SymPy), after the background
      constraint is imposed on both sides;
  [2] the same five rows against an independent rebuild from the field
      equations, delta(G - 6g) - alpha delta T = alpha S and the y component
      of the linearised Maxwell equation, by epsilon-differentiation of the
      full nonlinear tensors (no use of the response derivation);
  [3] the source display against systems/dk_stress.py at bhat = 0;
  [4] the horizon value of the flux Theta of systems/dk_pairing.py, and that
      it vanishes with H_rr(r_p);
  [5] the constants of the kernel formula as the paper prints them -- the two
      source couplings of the L_2 display, the conversion factor -1/2alpha
      of the text, the weight of P in the K display -- read from the LaTeX and
      compared with the constants of systems/dk_pairing.py; and the arithmetic
      that turns them, with the displayed definitions of X and P, into the
      displayed K = C4 - X - P/2 (with a negative control);
  [6] gauge invariance of the pairing: under xi = (xi^x, xi^r) e^{iGx} the
      shift of Re X + w Re P has a pointwise xi^x part proportional to
      (w - 1/2), and at w = 1/2 it is the total derivative of the displayed
      flux Q_xi;
  [7] the dressing identity X_c - X_ab = (P_dress - P)/2 and the kernel
      formula on every brane and harmonic of the committed kernel cache;
  [8] the source terms of the L_2 display, contracted with the response in
      the normalisation of the h_MN display, against the source Lagrangian
      src_h + src_k of systems/dk_pairing.py (with a negative control);
  [9] the displayed by-parts flux formula Q, applied to the generated
      quadratic Lagrangian l2q of systems/dk_pairing.py, against the
      generated flux theta (with a negative control).

The displays of [1]-[4] and [6] are located by their ``%% check:`` markers;
those of [5], [8] and [9] (L_2, Q, K, X and P) carry no marker and are located
by their left-hand sides.  A missing or reworded display exits non-zero.

Every comparison is exact (SymPy) or evaluated at random points to 30
digits; a failure exits non-zero.

Run:  uv run python -m backreaction.derivations.paper_response_display [tex]
      (default tex: paper/appendices/C-response-system.tex; ~1 min)
"""

import inspect
import pathlib
import random
import re
import sys
import time

import mpmath as mp
import numpy as np
import sympy as sp

from backreaction import paths
from backreaction.systems import dk_pairing as dps
from backreaction.systems import dk_response as dkr
from backreaction.systems import dk_stress as dks

T0 = time.time()
ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TEX = ROOT / "paper" / "appendices" / "C-response-system.tex"
CHECKS = {}


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def check(tag, ok, detail=""):
    CHECKS[tag] = bool(ok)
    log(f"{tag}: {'PASS' if ok else 'FAIL'} {detail}")


# ---------------------------------------------------------------------------
# symbols: background functions of r, response fields, sources
# ---------------------------------------------------------------------------
r = sp.Symbol("r", positive=True)
B, G, alpha = sp.symbols("B G alpha", positive=True)
Uf, Vf, Zf = (sp.Function(n)(r) for n in ("U", "V", "Z"))
w0f = sp.Function("w0")(r)
FIELD_NAMES = ("Htt", "Hxx", "Hyy", "Hzz", "Hrr", "Hxr", "bh")
FLD = {n: sp.Function(n)(r) for n in FIELD_NAMES}
CONJ = {n: sp.Function("K" + n[1:] if n.startswith("H") else "bhc")(r) for n in FLD}
xir = sp.Function("xir", real=True)(r)
OBJ = {
    n: sp.Symbol(n)
    for n in ("Stt", "Sxx", "Syy", "Szz", "Srr", "Qhor", "Qxi", "deltaXP", "Qbr")
}

# flat jet symbols used for all comparisons
U, Up, V, Vp, Z, Zp, w0, w0p = sp.symbols("U Up V Vp Z Zp w0 w0p")


def flat(e):
    """Functions of r -> jet symbols (background to first order, fields to 2nd)."""
    e = sp.sympify(e).doit()
    funcs = [(f, f.func.__name__) for f in list(FLD.values()) + list(CONJ.values())]
    funcs.append((xir, "xir"))
    rep2, rep1, rep0 = {}, {}, {}
    for f, nm in funcs:
        rep2[sp.Derivative(f, (r, 2))] = sp.Symbol(nm + "pp")
        rep1[sp.Derivative(f, r)] = sp.Symbol(nm + "p")
        rep0[f] = sp.Symbol(nm)
    for f, s0, s1 in ((Uf, U, Up), (Vf, V, Vp), (Zf, Z, Zp), (w0f, w0, w0p)):
        rep1[sp.Derivative(f, r)] = s1
        rep0[f] = s0
    for rep in (rep2, rep1, rep0):
        e = e.subs(rep)
    return e


# The displays are these multiples of the generated rows; the Einstein rows
# are the raw components delta(G - 6g)_MN - alpha delta T_MN - alpha S_MN, and
# the generated Maxwell row is -sqrt(-g) delta(nabla_M F^{My})/(iG) + source,
# so the Maxwell display is iG sqrt(-g) delta(nabla_M F^{My}) - (iG) source.
FACTOR = {
    "rr": U,
    "xr": -sp.I * G * U,
    "yy": sp.exp(-2 * V),
    "zz": sp.exp(-2 * Z),
    "My": -sp.I * G,
}

# background constraint (rr Einstein equation of the brane), solved for Z'
CONSTRAINT = (
    Up * (2 * Vp + Zp)
    + 2 * U * Vp**2
    + 4 * U * Vp * Zp
    - 12
    + alpha * B**2 * sp.exp(-4 * V)
)
ZP_RULE = {Zp: sp.solve(CONSTRAINT, Zp)[0]}


# ---------------------------------------------------------------------------
# LaTeX -> SymPy for the restricted vocabulary of the checked displays
# ---------------------------------------------------------------------------
TOKENS = [
    (r"\\mathcal\{Q\}_\\xi", "Qxi"),
    (r"\\mathcal\{Q\}\(r_p\)", "Qhor"),
    (r"\\bar\{\\hat b\}", "bhc"),
    (r"\\bar H_\{(tt|xx|yy|zz|rr|xr)\}", lambda m: "K" + m.group(1)),
    (r"H_\{(tt|xx|yy|zz|rr|xr)\}", lambda m: "H" + m.group(1)),
    (r"S_\{(tt|xx|yy|zz|rr)\}", lambda m: "S" + m.group(1)),
    (r"\\hat b", "bh"),
    (r"w_0", "w0"),
    (r"p_1", "p1"),
    (r"p_2", "p2"),
    (r"\\alpha", "alpha"),
    (r"\\mathrm\{Re\}\s*\\xi\^r", "xir"),
    (r"\\xi\^r", "xir"),
]
STRIP = [
    r"\\label\{[^}]*\}",
    r"\\nonumber",
    r"\\notag",
    r"\\left",
    r"\\right",
    r"\\[bB]igg?[lr]?",
    r"\\,",
    r"\\;",
    r"\\!",
    r"~",
    r"&",
    r"\\begin\{split\}",
    r"\\end\{split\}",
    r"\\quad(?!q)",
]


def _match_brace(s, i):
    """s[i] == '{'; return index of the matching '}'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    raise ValueError(f"unbalanced braces in {s!r}")


def _fracs(s):
    s = re.sub(r"\\[td]?frac(\d)(\d)", r"((\1)/(\2))", s)
    while True:
        m = re.search(r"\\[td]?frac\{", s)
        if not m:
            return s
        i = m.end() - 1
        j = _match_brace(s, i)
        assert s[j + 1] == "{", f"frac without denominator: {s[m.start() :]}"
        k = _match_brace(s, j + 1)
        s = (
            s[: m.start()]
            + "(("
            + s[i + 1 : j]
            + ")/("
            + s[j + 2 : k]
            + "))"
            + s[k + 1 :]
        )


def _exps_and_powers(s):
    while True:
        m = re.search(r"(?<!\\)e\^\{", s)
        if not m:
            break
        i = m.end() - 1
        j = _match_brace(s, i)
        s = s[: m.start()] + "⟨exp⟩(" + s[i + 1 : j] + ")" + s[j + 1 :]
    while "^{" in s:
        i = s.index("^{") + 1
        j = _match_brace(s, i)
        s = s[: i - 1] + "**(" + s[i + 1 : j] + ")" + s[j + 1 :]
    return re.sub(r"\^(\d)", r"**\1", s)


def _primes(s):
    """X' and X'' on a token or a parenthesised group -> D(...)."""
    while True:
        m = re.search(r"(⟨[^⟩]+⟩|\))('+)", s)
        if not m:
            return s
        n = len(m.group(2))
        if m.group(1) == ")":
            depth, j = 0, m.start(1)
            while j >= 0:
                depth += {")": 1, "(": -1}.get(s[j], 0)
                if depth == 0:
                    break
                j -= 1
            inner = s[j : m.start(1) + 1]
            s = s[:j] + "⟨D⟩(" * n + inner + ")" * n + s[m.end(2) :]
        else:
            s = s[: m.start()] + "⟨D⟩(" * n + m.group(1) + ")" * n + s[m.end(2) :]


def latex_to_sympy(body, ns):
    s = body
    for pat in STRIP:
        s = re.sub(pat, " ", s)
    s = s.replace("\\\\", " ")
    s = re.sub(r"[,.]\s*$", "", s.strip())
    s = _fracs(s)
    for pat, rep in TOKENS:
        s = re.sub(
            pat, (lambda m, rep=rep: "⟨" + (rep(m) if callable(rep) else rep) + "⟩"), s
        )
    s = _exps_and_powers(s)
    # remaining bare letters are single-letter symbols
    parts = re.split(r"(⟨[^⟩]*⟩)", s)
    for i, part in enumerate(parts):
        if not part.startswith("⟨"):
            if re.search(r"[A-Za-z]", re.sub(r"[UVZBGsr]", "", part)):
                raise ValueError(f"unknown letter in {part!r} of {body!r}")
            parts[i] = re.sub(r"([UVZBGsr])", r"⟨\1⟩", part)
    s = "".join(parts)
    if "\\" in s:
        raise ValueError(f"unconverted LaTeX command in: {s}")
    s = s.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    s = _primes(s)
    # implicit multiplication between adjacent operands
    s = re.sub(r"(⟩|\)|\d)\s*(?=⟨|\()", r"\1*", s)
    s = s.replace("*⟨D⟩*(", "*⟨D⟩(").replace("⟨D⟩*(", "⟨D⟩(")
    s = s.replace("⟨exp⟩*(", "⟨exp⟩(")
    s = re.sub(r"⟨([^⟩]+)⟩", r"\1", s)
    if re.search(
        r"[A-Za-z]{2,}", re.sub(r"\b(" + "|".join(map(re.escape, ns)) + r")\b", "", s)
    ):
        raise ValueError(f"unknown name in converted display: {s}")
    return sp.sympify(s, locals=ns)


def namespace():
    ns = {k: v for k, v in FLD.items()}
    ns.update({"K" + k[1:]: CONJ[k] for k in FLD if k.startswith("H")})
    ns["bhc"] = CONJ["bh"]
    ns.update(OBJ)
    ns.update(
        dict(U=Uf, V=Vf, Z=Zf, w0=w0f, B=B, G=G, s=G**2, alpha=alpha, r=r, xir=xir)
    )
    ns["p1"] = sp.exp(Zf - 2 * Vf)
    ns["p2"] = Uf * sp.exp(Zf)
    ns["exp"] = sp.exp
    ns["D"] = lambda e: sp.diff(e, r)
    return ns


def read_displays(path):
    """{id: [ (lhs, rhs), ... ]} for every %% check: block."""
    text = pathlib.Path(path).read_text()
    out = {}
    for m in re.finditer(
        r"%% check: (\w+)\s*\n\\begin\{equation\}(.*?)\\end\{equation\}", text, re.S
    ):
        tag, body = m.group(1), m.group(2)
        body = re.sub(r"(?m)%.*$", "", body)
        pieces = re.split(r"\\qquad", body)
        rels = []
        for p in pieces:
            p = p.strip().rstrip(",.").strip()
            p = re.sub(r"[,.]\s*$", "", p)
            if not p:
                continue
            sides = p.split("=")
            for a, b in zip(sides[:-1], sides[1:], strict=True):
                rels.append((a, b))
        out[tag] = rels
    return out


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------
class _M:
    exp = staticmethod(sp.exp)


def sym_of(fn, names):
    """Evaluate a generated numeric function on SymPy symbols."""
    src = inspect.getsource(fn).replace("1j", "sp.I")
    g = {"math": _M, "sp": sp, "np": None}
    exec(src, g)
    return g[fn.__name__](*names)


def generated_rows():
    """rows {rr, xr, yy, zz, My} as flat expressions in (H.., bh) jets."""
    args = (r, U, Up, V, Vp, Z, Zp, B, G, alpha, w0, w0p)
    M = sp.Matrix(sym_of(dkr._Mmat, args))
    Rm = sp.Matrix(sym_of(dkr._Rmat, args))
    X = sp.Matrix(sp.symbols("Hrr Hxr Hrrp Hxrp Hyypp Hzzpp aypp"))
    Y = sp.Matrix(list(sp.symbols("Hyy Hyyp Hzz Hzzp ay ayp")) + [1])
    rows = M * X - Rm * Y
    bh, bhp, bhpp = sp.symbols("bh bhp bhpp")
    ay, ayp, aypp = sp.symbols("ay ayp aypp")
    sub = {ay: bh / (sp.I * G), ayp: bhp / (sp.I * G), aypp: bhpp / (sp.I * G)}
    return {
        k: sp.expand(rows[i].subs(sub))
        for k, i in (("rr", 0), ("xr", 1), ("yy", 4), ("zz", 5), ("My", 6))
    }


def _rand_point(rng, extra=()):
    pt = {
        s: mp.mpf(rng.uniform(0.3, 1.7))
        for s in (U, Up, V, Vp, Z, w0, w0p, B, G, alpha)
    }
    for s in extra:
        pt[s] = mp.mpf(rng.uniform(-1.5, 1.5))
    return pt


def proportional(a, b, syms, npts=8, seed=1):
    """a == c * b for a constant c?  Returns (ok, c) on random points (Z' from the
    constraint), with c compared across points to 25 digits."""
    a = sp.expand(a.subs(ZP_RULE))
    b = sp.expand(b.subs(ZP_RULE))
    syms = jet_syms(a, b)
    fa = sp.lambdify(list(syms), a, "mpmath")
    fb = sp.lambdify(list(syms), b, "mpmath")
    rng = random.Random(seed)
    cs = []
    mp.mp.dps = 40
    for _ in range(npts):
        pt = _rand_point(rng)
        vals = [pt.get(s, mp.mpf(rng.uniform(-1.5, 1.5))) for s in syms]
        vb = fb(*vals)
        if abs(vb) < mp.mpf(10) ** -20:
            continue
        cs.append(fa(*vals) / vb)
    if not cs:
        return False, None
    c0 = cs[0]
    ok = all(abs(c - c0) < mp.mpf(10) ** -25 * max(1, abs(c0)) for c in cs)
    return ok, complex(c0)


def jet_syms(*exprs):
    out = set()
    for e in exprs:
        out |= {s for s in sp.sympify(e).free_symbols}
    return sorted(out, key=str)


# ---------------------------------------------------------------------------
# [2] independent rebuild of the five rows from the field equations
# ---------------------------------------------------------------------------
def rebuild_rows():
    """delta(G_MN - 6 g_MN) - alpha delta T_MN - alpha S_MN e^{iGx}, (MN) in
    {rr, xr, yy, zz}, and iG * sqrt(-g) * delta(nabla_M F^{My}) per e^{iGx}, in
    the gauge H_tt = H_xx = 0, from the full nonlinear tensors."""
    t, x, y, z = sp.symbols("t x y z", real=True)
    co = [t, x, y, z, r]
    N = 5
    eps = sp.Symbol("epsilon")
    E = sp.exp(sp.I * G * x)
    H = FLD
    gb = sp.diag(-Uf, sp.exp(2 * Vf), sp.exp(2 * Vf), sp.exp(2 * Zf), 1 / Uf)
    h = sp.zeros(N)
    h[2, 2] = sp.exp(2 * Vf) * H["Hyy"]
    h[3, 3] = sp.exp(2 * Zf) * H["Hzz"]
    h[4, 4] = H["Hrr"] / Uf
    h[1, 4] = h[4, 1] = sp.I * sp.exp(2 * Vf) * H["Hxr"]
    g = gb + eps * E * h
    gbi = gb.inv()
    gi = gbi - eps * E * gbi * h * gbi
    A = [0, 0, B * x + eps * E * H["bh"] / (sp.I * G), 0, 0]

    def lin(e):
        return sp.expand(sp.diff(e, eps).subs(eps, 0) / E)

    def bg(e):
        return sp.expand(e.subs(eps, 0))

    Gam = [
        [
            [
                sp.expand(
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
                )
                for n in range(N)
            ]
            for m in range(N)
        ]
        for l in range(N)
    ]
    # keep only O(eps^1)
    Gam = [
        [
            [sp.series(Gam[l][m][n], eps, 0, 2).removeO() for n in range(N)]
            for m in range(N)
        ]
        for l in range(N)
    ]

    def ric(m, n):
        e = sum(sp.diff(Gam[l][m][n], co[l]) for l in range(N))
        e -= sum(sp.diff(Gam[l][m][l], co[n]) for l in range(N))
        e += sum(
            Gam[l][l][k] * Gam[k][m][n] - Gam[l][n][k] * Gam[k][m][l]
            for l in range(N)
            for k in range(N)
        )
        return sp.series(sp.expand(e), eps, 0, 2).removeO()

    Ric = {(m, n): ric(m, n) for m in range(N) for n in range(m, N)}
    Rs = sp.series(
        sp.expand(
            sum(
                gi[m, n] * Ric[(min(m, n), max(m, n))]
                for m in range(N)
                for n in range(N)
            )
        ),
        eps,
        0,
        2,
    ).removeO()
    F = [
        [sp.diff(A[n], co[m]) - sp.diff(A[m], co[n]) for n in range(N)]
        for m in range(N)
    ]
    F2 = sp.expand(
        sum(
            F[m][n] * gi[m, p] * gi[n, q] * F[p][q]
            for m in range(N)
            for n in range(N)
            for p in range(N)
            for q in range(N)
        )
    )

    def T(m, n):
        return sp.expand(
            sum(F[m][p] * gi[p, q] * F[n][q] for p in range(N) for q in range(N))
            - g[m, n] * F2 / 4
        )

    S = {
        (4, 4): sp.exp(-4 * Vf)
        / Uf
        * (B * w0f**2 + Uf * sp.exp(2 * Vf) * sp.diff(w0f, r) ** 2),
        (2, 2): -B * w0f**2 * sp.exp(-2 * Vf),
        (3, 3): sp.exp(-4 * Vf + 2 * Zf)
        * (B * w0f**2 - Uf * sp.exp(2 * Vf) * sp.diff(w0f, r) ** 2),
        (1, 4): 0,
    }
    rows, bgE = {}, {}
    for (m, n), tag in (
        ((4, 4), "rr"),
        ((1, 4), "xr"),
        ((2, 2), "yy"),
        ((3, 3), "zz"),
        ((0, 0), "tt"),
        ((1, 1), "xx"),
    ):
        full = Ric[(m, n)] - Rs * g[m, n] / 2 - 6 * g[m, n] - alpha * T(m, n)
        bgE[tag] = bg(full)
        if tag in ("tt", "xx"):
            continue
        rows[tag] = lin(full) - alpha * S[(m, n)]
    # Maxwell (y): sqrt(-g) nabla_M F^{My} = d_M (sqrt(-g) F^{My})
    tr = sum(gbi[m, m] * h[m, m] for m in range(N))
    sq = sp.exp(2 * Vf + Zf) * (1 + eps * E * tr / 2)  # sqrt(-g) to O(eps)
    Fup = [
        [
            sum(gi[m, p] * gi[n, q] * F[p][q] for p in range(N) for q in range(N))
            for n in range(N)
        ]
        for m in range(N)
    ]
    MY = sum(
        sp.diff(sp.series(sq * Fup[m][2], eps, 0, 2).removeO(), co[m]) for m in range(N)
    )
    rows["My"] = sp.expand(sp.I * G * lin(MY))
    # background second derivatives from tt, xx, zz; constraint is rr
    U2, V2, Z2 = (sp.Derivative(f, (r, 2)) for f in (Uf, Vf, Zf))
    sol = sp.solve([bgE["tt"], bgE["xx"], bgE["zz"]], [U2, V2, Z2], dict=True)[0]
    P = Uf * sp.exp(Zf)
    w0pp = sp.solve(
        sp.diff(P * sp.diff(w0f, r), r) + B * sp.exp(Zf - 2 * Vf) * w0f,
        sp.Derivative(w0f, (r, 2)),
    )[0]
    out = {}
    for k, e in rows.items():
        e = e.subs(sol).subs(sp.Derivative(w0f, (r, 2)), w0pp)
        out[k] = sp.expand(flat(sp.expand(e.doit())))
        assert not out[k].has(sp.Derivative), f"unreduced derivative in rebuilt row {k}"
    return out


# ---------------------------------------------------------------------------
# [5], [8], [9]: displays located by content, and the generated pairing objects
# ---------------------------------------------------------------------------
def find_display(text, needle):
    """Body of the first equation environment whose body contains `needle`."""
    for m in re.finditer(r"\\begin\{equation\}(.*?)\\end\{equation\}", text, re.S):
        body = re.sub(r"(?m)%.*$", "", m.group(1))
        if needle in body:
            return body
    raise LookupError(f"no display containing {needle!r}")


def simple_parse(body, tokens, names):
    """LaTeX -> SymPy for a display whose every name is listed in `tokens`
    (pattern -> name); `names` maps names to SymPy objects."""
    s = body
    for pat, name in tokens:
        s = re.sub(pat, lambda m, n=name: " ⟨" + n + "⟩ ", s)
    for pat in STRIP + [r"\\!", r"\;", r"\\qquad"]:
        s = re.sub(pat, " ", s)
    s = re.sub(r"[,.]\s*$", "", s.strip())
    s = _fracs(s)
    s = _exps_and_powers(s)
    s = s.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    s = _primes(s)
    s = re.sub(r"(⟩|\)|\d)\s*(?=⟨|\()", r"\1*", s)
    s = s.replace("⟨D⟩*(", "⟨D⟩(").replace("⟨Re⟩*(", "⟨Re⟩(")
    s = re.sub(r"⟨([^⟩]+)⟩", r"\1", s)
    rest = re.sub(r"\b(" + "|".join(map(re.escape, names)) + r")\b", "", s)
    if re.search(r"\\|[A-Za-z]", rest):
        raise ValueError(f"unconverted token in {body!r} -> {s!r}")
    return sp.sympify(s, locals=names)


def gen_sym(fn):
    """A generated systems/dk_pairing.py function on symbols named as its
    arguments; returns (expr, {name: symbol})."""
    names = list(inspect.signature(fn).parameters)
    S = {n: sp.Symbol(n) for n in names}
    return sp.expand(sym_of(fn, [S[n] for n in names])), S


PAIR_FIELDS = ("Hyy", "Hzz", "Hrr", "Hxr", "ay", "Kyy", "Kzz", "Krr", "Kxr", "Ac")


def total_d(e, S):
    """d/dr of a jet expression of the generated pairing objects (background
    to first order: second background derivatives must not appear)."""
    for bad in ("Up", "Vp", "Wp", "W0p"):
        assert sp.diff(e, S[bad]) == 0, f"total_d needs the derivative of {bad}"
    out = sp.diff(e, S["r"])
    for f, fp in (("U", "Up"), ("V", "Vp"), ("W", "Wp"), ("W0", "W0p")):
        out += sp.diff(e, S[f]) * S[fp]
    for f in PAIR_FIELDS:
        assert sp.diff(e, S[f + "pp"]) == 0, (
            f"total_d needs the third derivative of {f}"
        )
        out += sp.diff(e, S[f]) * S[f + "p"] + sp.diff(e, S[f + "p"]) * S[f + "pp"]
    return sp.expand(out)


L2_TOKENS = [
    (r"\\mathcal\{L\}_2", "L2"),
    (r"\\mathcal\{L\}_q", "Lq"),
    (r"\\sqrt\{-g\}", "sqrtg"),
    (r"S\^\{MN\}", "SMN"),
    (r"\\bar h_\{MN\}", "hbMN"),
    (r"h_\{MN\}", "hMN"),
    (r"\\bar\{\\hat b\}", "bhc"),
    (r"\\hat b", "bh"),
    (r"p_1", "p1"),
    (r"w_0", "w0"),
    (r"\\alpha", "alpha"),
]


def check_L2_sources(text):
    """[8] the L_2 source terms against src_h + src_k of systems/dk_pairing."""
    body = find_display(text, r"\mathcal{L}_2 =")
    names = {n: sp.Symbol(n) for _, n in L2_TOKENS}
    lhs, rhs = body.split("=")
    assert simple_parse(lhs, L2_TOKENS, names) == names["L2"]
    e = sp.expand(simple_parse(rhs, L2_TOKENS, names).subs(names["Lq"], 0))
    sq, SM, hM, hbM = (names[k] for k in ("sqrtg", "SMN", "hMN", "hbMN"))
    SH, SHc = sp.symbols("SH SHc")
    e = sp.expand(e.subs({hM: SH / (sq * SM), hbM: SHc / (sq * SM)}))
    assert not e.has(sq, SM), (
        f"L_2 source terms not of the form sqrt(-g) S^MN h_MN: {e}"
    )
    # the generated source Lagrangian, in its own symbols (W = Z, Ac = conj a_y)
    src, S = gen_sym(dps.src_h)
    srck, _ = gen_sym(dps.src_k)
    src = sp.expand(src + srck)
    eV, eW = sp.exp(S["V"]), sp.exp(S["W"])
    args0 = (
        S["r"],
        S["U"],
        S["Up"],
        S["V"],
        S["Vp"],
        S["W"],
        S["Wp"],
        S["B"],
        S["G"],
        S["W0"],
        S["W0p"],
        0,
        0,
    )
    Sl = {
        c: sym_of(getattr(dks, "S_" + c), args0) for c in ("tt", "xx", "yy", "zz", "rr")
    }
    sqrtg = eV**2 * eW
    ginv = {"yy": eV**-2, "zz": eW**-2, "rr": S["U"]}

    def contract(pref):
        """sqrt(-g) S^MN h_MN, h_MN as in the h display, gauge H_tt = H_xx = 0."""
        h = {
            "yy": eV**2 * S[pref + "yy"],
            "zz": eW**2 * S[pref + "zz"],
            "rr": S[pref + "rr"] / S["U"],
        }
        return sqrtg * sum(ginv[c] ** 2 * Sl[c] * h[c] for c in h)

    sub = {
        SH: contract("H"),
        SHc: contract("K"),
        names["bh"]: sp.I * S["G"] * S["ay"],
        names["bhc"]: -sp.I * S["G"] * S["Ac"],
        names["p1"]: eW / eV**2,
        names["w0"]: S["W0"],
        names["alpha"]: S["alpha"],
    }

    def matches(expr):
        return sp.simplify(sp.expand(expr.subs(sub)) - src) == 0

    ok = matches(e)
    # negative control: the same display with the two couplings interchanged
    p1w2 = names["p1"] * names["w0"] ** 2
    cS = e.coeff(SH)
    cJ = sp.expand(e.coeff(names["bh"]) / p1w2)
    swapped = sp.expand(cJ * (SH + SHc) + cS * p1w2 * (names["bh"] + names["bhc"]))
    return ok, not matches(swapped), e, names, (SH, SHc)


def check_kernel_constants(text, l2e, names, SHs):
    """[5] constants of the kernel formula as printed vs systems/dk_pairing."""
    a = names["alpha"]
    SH, SHc = SHs
    cS = sp.simplify(l2e.coeff(SH) / a)
    cJ = sp.simplify(l2e.coeff(names["bh"]) / (a * names["p1"] * names["w0"] ** 2))
    m = re.search(
        r"on-shell\s+action\s+is\s+multiplied\s+by\s+\$-1/(\d+)\\alpha\$", text
    )
    assert m, "conversion factor '-1/n\\alpha' not found in the text"
    factor = sp.Integer(-1) / (int(m.group(1)) * a)
    half = re.search(r"on shell \$\\mathcal\{L\}_2\$ is half its source terms", text)
    Kb = find_display(text, r"K(s;\beta) =")
    Kt = [
        (r"K\(s;\\beta\)", "K"),
        (r"C_4", "C4"),
        (r"X\(s;\\beta\)", "Xs"),
        (r"\\mathcal\{P\}\(s;\\beta\)", "Ps"),
    ]
    kn = {n: sp.Symbol(n) for _, n in Kt}
    Kl, Kr = Kb.split("=")
    assert simple_parse(Kl, Kt, kn) == kn["K"]
    Krhs = sp.expand(simple_parse(Kr, Kt, kn))
    wP = Krhs.coeff(kn["Ps"])
    XPb = find_display(text, r"X = \mathrm{Re}")
    # complex integrands: z2 = p1 w0^2 bhat, z1 = sqrt(-g) S^MN hbar_MN
    z1, z1c, z2, z2c = sp.symbols("z1 z1c z2 z2c")
    re_int = r"\\mathrm\{Re\}\s*(\\!)?\s*\\int\s*"
    XPt = [
        (re_int + r"p_1\s*w_0\^2\s*(\\,)?\s*\\hat b\s*(\\,)?\s*dr", "ReZ2"),
        (
            re_int + r"\\sqrt\{-g\}\s*(\\;)?\s*S\^\{MN\}\s*(\\,)?\s*"
            r"\\bar h_\{MN\}\s*(\\,)?\s*dr",
            "ReZ1",
        ),
        (r"\\mathcal\{P\}", "P"),
        (r"\bX\b", "X"),
    ]
    xn = {n: sp.Symbol(n) for _, n in XPt}
    re_of = {xn["ReZ1"]: (z1 + z1c) / 2, xn["ReZ2"]: (z2 + z2c) / 2}
    defs = {}
    for piece in re.split(r"\\qquad", XPb):
        piece = piece.strip().rstrip("~,. ").strip()
        if piece:
            lhs, rhs = piece.split("=")
            defs[simple_parse(lhs, XPt, xn)] = sp.expand(
                simple_parse(rhs, XPt, xn).subs(re_of)
            )
    Xd, Pd = defs[xn["X"]], defs[xn["P"]]

    def arithmetic(fac):
        # on shell L_2 = (1/2)(source terms); S^MN real, so h + hbar pairs give
        # z1 + conj(z1), and bhat + conj(bhat) gives z2 + conj(z2)
        srcint = cS * a * (z1 + z1c) + cJ * a * (z2 + z2c)
        lhs = sp.expand(fac * srcint / 2)
        return (
            sp.expand(lhs - (Krhs - kn["C4"]).subs({kn["Xs"]: Xd, kn["Ps"]: Pd})) == 0
        )

    consts = (
        float(cS) == dps.C_SRC
        and float(cJ) == dps.CJ_PER_ALPHA
        and float(sp.simplify(factor * a)) == dps.NORM_TIMES_ALPHA
        and float(wP) == dps.W_BARE
    )
    return dict(
        consts=bool(consts and half),
        arith=arithmetic(factor),
        control=not arithmetic(2 * factor),
        detail=f"c = {cS}, cJ/alpha = {cJ}, factor = {factor}, weight on P = {wP}",
    )


Q_TOKENS = [
    (r"\\frac\{\\partial\\mathcal\{L\}_q\}\{\\partial\\phi''\}", "d2"),
    (r"\\frac\{\\partial\\mathcal\{L\}_q\}\{\\partial\\phi'\}", "d1"),
    (r"\\mathcal\{Q\}", "Q"),
    (r"\\sum_\\phi", "SUM"),
    (r"\\phi'", "phip"),
    (r"\\phi", "phi"),
]


def check_Q_formula(text):
    """[9] the displayed Q formula on the generated l2q equals generated theta."""
    body = find_display(text, r"\mathcal{Q} = \sum_\phi")
    qn = {n: sp.Symbol(n) for _, n in Q_TOKENS}
    qn["D"] = sp.Function("D")
    lhs, rhs = body.split("=")
    assert simple_parse(lhs, Q_TOKENS, qn) == qn["Q"]
    form = sp.expand(simple_parse(rhs, Q_TOKENS, qn))
    assert form.coeff(qn["SUM"]) != 0, "no sum over the fields"
    form = form.coeff(qn["SUM"])
    L, S = gen_sym(dps.l2q)
    theta, _ = gen_sym(dps.theta)
    Q = 0
    for f in PAIR_FIELDS:
        d1 = sp.diff(L, S[f + "p"])
        d2 = sp.diff(L, S[f + "pp"])
        e = form.subs(
            {qn["d1"]: d1, qn["d2"]: d2, qn["phi"]: S[f], qn["phip"]: S[f + "p"]}
        )
        e = e.replace(qn["D"], lambda arg: total_d(sp.expand(arg), S))
        Q += sp.expand(e)
    ok = sp.simplify(sp.expand(Q - theta)) == 0
    # negative control: the same formula without the (dL/dphi'')' term
    Qc = sum(
        sp.expand(S[f] * sp.diff(L, S[f + "p"]) + S[f + "p"] * sp.diff(L, S[f + "pp"]))
        for f in PAIR_FIELDS
    )
    ctrl = sp.simplify(sp.expand(Qc - theta)) != 0
    return ok, ctrl


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(tex):
    log(f"reading displays from {tex}")
    disp = read_displays(tex)
    need = {"source", "rr", "xr", "yy", "zz", "My", "Qhor", "Qxi"}
    missing = need - set(disp)
    if missing:
        log(f"no '%% check:' display for {sorted(missing)} in {tex}")
        sys.exit(1)
    ns = namespace()
    parsed = {}
    for k, v in disp.items():
        if k == "Qxi":  # only the definition of the flux is checked
            v = [(a, b) for a, b in v if a.strip().startswith("\\mathcal{Q}_\\xi")]
        parsed[k] = [(latex_to_sympy(a, ns), latex_to_sympy(b, ns)) for a, b in v]
    log(
        f"parsed {sum(len(v) for v in parsed.values())} relations in {len(parsed)} displays"
    )

    gen = generated_rows()
    log("generated rows read from systems/dk_response.py")
    t1 = time.time()
    reb = rebuild_rows()
    log(f"independent rebuild from the field equations done ({time.time() - t1:.0f}s)")

    # [1]/[2] rows: each display is (factor) x (row), the factor stated in the text
    for tag in ("rr", "xr", "yy", "zz", "My"):
        ((lhs, rhs),) = parsed[tag]
        d = sp.expand(flat(lhs - rhs))
        syms = jet_syms(d, gen[tag], reb[tag], FACTOR[tag])
        ok1, c1 = proportional(d, FACTOR[tag] * gen[tag], syms)
        if tag == "My":
            # the rebuild carries the Maxwell operator only; its source, the
            # colour-3 current, is checked against systems/dk_stress in [3]
            d_h = d.subs({w0: 0, w0p: 0})
            ok2, c2 = proportional(d_h, reb[tag], syms)
        else:
            ok2, c2 = proportional(d, FACTOR[tag] * reb[tag], syms)
        ok1 = ok1 and abs(c1 - 1) < 1e-12
        ok2 = ok2 and abs(abs(c2) - 1) < 1e-12
        check(f"[1] display ({tag}) = c x generated row", ok1, f"c = {c1}")
        check(f"[2] display ({tag}) = c x field-equation rebuild", ok2, f"c = {c2}")

    # [3] sources
    args0 = (r, U, Up, V, Vp, Z, Zp, B, G, w0, w0p, 0, 0)
    ref = {
        OBJ["S" + c]: sym_of(getattr(dks, "S_" + c), args0)
        for c in ("tt", "xx", "yy", "zz", "rr")
    }
    ok = True
    for lhs, rhs in parsed["source"]:
        lhs_f = flat(lhs).subs(ref)
        rhs_f = flat(rhs).subs(ref)
        ok &= sp.simplify(sp.expand(lhs_f - rhs_f)) == 0
    check(
        "[3] source display = systems/dk_stress at bhat = 0",
        ok,
        f"({len(parsed['source'])} relations)",
    )
    Jref = sym_of(dks.J_transverse, args0)
    check(
        "[3] current J = s p1 w0^2",
        sp.simplify(Jref - G**2 * sp.exp(Z - 2 * V) * w0**2) == 0,
    )

    # [4] flux at the horizon
    th_args = list(inspect.signature(dps.theta).parameters)
    TS = {a: sp.Symbol(a) for a in th_args}
    theta = sp.expand(sym_of(dps.theta, [TS[a] for a in th_args]))
    rename = {
        TS["W"]: Z,
        TS["Wp"]: Zp,
        TS["U"]: U,
        TS["Up"]: Up,
        TS["V"]: V,
        TS["Vp"]: Vp,
    }
    th0 = sp.expand(theta.subs(TS["U"], 0).subs(rename))
    ((lhs, rhs),) = parsed["Qhor"]
    disp_q = sp.expand(flat(rhs))
    check("[4] displayed Q(r_p) = Theta at U = 0", sp.simplify(th0 - disp_q) == 0)
    check(
        "[4] Q(r_p) vanishes with H_rr(r_p)",
        sp.expand(disp_q.subs({sp.Symbol("Hrr"): 0, sp.Symbol("Krr"): 0})) == 0,
    )

    # [5] the constants of the kernel formula as printed, and its arithmetic
    text = pathlib.Path(tex).read_text()
    ok8, ctrl8, l2e, l2names, SHs = check_L2_sources(text)
    kc = check_kernel_constants(text, l2e, l2names, SHs)
    check(
        "[5] printed couplings (L_2), conversion factor and weight of P = "
        "systems/dk_pairing constants",
        kc["consts"],
        kc["detail"],
    )
    check(
        "[5] printed factor x (1/2)(L_2 source terms), with the displayed X and P, "
        "= the displayed K - C4",
        kc["arith"],
    )
    check(
        "[5] negative control: twice the factor does not give the displayed K",
        kc["control"],
    )

    # [6] gauge variation of Re X + w Re P under xi = (xi^x, xi^r) e^{iGx}
    t1 = time.time()
    ok6a, ok6b, flux = gauge_variation()
    check("[6] xi^x shift of X + w P is proportional to (w - 1/2), nonzero", ok6a)
    check(
        "[6] at w = 1/2 the shift is d/dr of the flux Q_xi",
        ok6b,
        f"({time.time() - t1:.0f}s)",
    )
    rel = parsed["Qxi"]
    assert len(rel) == 1, "no 'Q_xi = ...' relation in the Qxi display"
    check("[6] displayed Q_xi = derived flux", sp.simplify(flat(rel[0][1]) - flux) == 0)

    # [7] dressing identity and kernel formula on the committed cache
    d = np.load(paths.data("dk_kernel.npz"), allow_pickle=True)
    Xc, X, P, Pib, K, C4 = (d[k] for k in ("X_coupled", "X", "P", "Pi_bare", "K", "C4"))
    dres = np.max(np.abs((Xc - X) - 0.5 * (P - Pib)))
    kres = np.max(np.abs(K - (C4 - Xc - 0.5 * Pib)))
    check(
        "[7] X_c - X_ab = (P_dress - P)/2 on dk_kernel.npz",
        # 8.4e-13 on the reference platform (macOS arm64), quoted in app. D.4
        # as 8e-13; the tolerance leaves ~4x for other BLAS/CPU rounding.
        dres < 3e-12,
        f"max |res| = {dres:.1e}, max |X_c - X_ab| = {np.max(np.abs(Xc - X)):.4f}",
    )
    # dk_kernel.py stores K by this very formula, so this guards the cache's
    # internal consistency (no stale K beside fresh X_c, P); it is not an
    # independent check of the kernel formula, which [5] checks symbolically.
    check(
        "[7] cache consistency: stored K = C4 - X_c - P/2 in dk_kernel.npz",
        kres < 1e-12,
        f"{kres:.1e}",
    )

    # [8] the L_2 source terms against the generated source Lagrangian
    check("[8] L_2 source terms = src_h + src_k of systems/dk_pairing", ok8)
    check("[8] negative control: couplings interchanged do not match", ctrl8)

    # [9] the displayed flux formula against the generated flux
    t1 = time.time()
    ok9, ctrl9 = check_Q_formula(text)
    check(
        "[9] displayed Q formula applied to the generated l2q = generated theta",
        ok9,
        f"({time.time() - t1:.0f}s)",
    )
    check("[9] negative control: dropping the (dL/dphi'')' term breaks it", ctrl9)

    failed = [k for k, v in CHECKS.items() if not v]
    log(f"{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        for k in failed:
            log(f"FAILED: {k}")
        sys.exit(1)


def gauge_variation():
    """Shift of Re X + w Re P under the diffeomorphism xi_M = (0, xi_x, 0, 0,
    xi_r) e^{iGx} (lowered, general complex profiles), with delta a_y = B xi^x."""
    t, x, y, z = sp.symbols("t x y z", real=True)
    co = [t, x, y, z, r]
    N = 5
    Ur, Vr, Zr, wr = (sp.Function(n, real=True)(r) for n in ("U", "V", "Z", "w0"))
    g = sp.diag(-Ur, sp.exp(2 * Vr), sp.exp(2 * Vr), sp.exp(2 * Zr), 1 / Ur)
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
    a1, a2, b1, b2 = (
        sp.Function(n, real=True)(r) for n in ("xa1", "xa2", "xb1", "xb2")
    )
    E = sp.exp(sp.I * G * x)
    xl = [0, (a1 + sp.I * a2) * E, 0, 0, (b1 + sp.I * b2) * E]
    h = sp.zeros(N)
    for m in range(N):
        for n in range(N):
            h[m, n] = sp.expand(
                (
                    sp.diff(xl[n], co[m])
                    + sp.diff(xl[m], co[n])
                    - 2 * sum(Gam[l][m][n] * xl[l] for l in range(N))
                )
                / E
            )
    wp = sp.diff(wr, r)
    Sl = sp.zeros(N)
    Sl[0, 0] = Ur * sp.exp(-4 * Vr) * (Ur * sp.exp(2 * Vr) * wp**2 - B * wr**2)
    Sl[1, 1] = Sl[2, 2] = -B * wr**2 * sp.exp(-2 * Vr)
    Sl[3, 3] = sp.exp(-4 * Vr + 2 * Zr) * (B * wr**2 - Ur * sp.exp(2 * Vr) * wp**2)
    Sl[4, 4] = sp.exp(-4 * Vr) / Ur * (B * wr**2 + Ur * sp.exp(2 * Vr) * wp**2)

    def conj(e):
        return sp.expand(e).subs(sp.I, -sp.I)

    def re_(e):
        return (sp.expand(e) + conj(e)) / 2

    sq = sp.exp(2 * Vr + Zr)
    dP = re_(
        sq
        * sum(
            gi[m, m] * gi[n, n] * Sl[m, n] * conj(h[m, n])
            for m in range(N)
            for n in range(N)
        )
    )
    dX = re_(
        sp.exp(Zr - 2 * Vr) * wr**2 * sp.I * G * B * sp.exp(-2 * Vr) * (a1 + sp.I * a2)
    )
    w = sp.Symbol("w")
    dens = sp.expand(dX + w * dP)
    P_ = Ur * sp.exp(Zr)
    rule = {
        sp.Derivative(wr, (r, 2)): sp.solve(
            sp.diff(P_ * wp, r) + B * sp.exp(Zr - 2 * Vr) * wr,
            sp.Derivative(wr, (r, 2)),
        )[0]
    }

    def el(e, f):
        c1 = e.coeff(sp.Derivative(f, r))
        c0 = sp.expand(e - c1 * sp.Derivative(f, r)).coeff(f)
        return sp.simplify(sp.expand((c0 - sp.diff(c1, r)).subs(rule)))

    ela = [el(dens, f) for f in (a1, a2)]
    # proportional to (w - 1/2) and nonzero
    ok_a = all(sp.simplify(e.subs(w, sp.Rational(1, 2))) == 0 for e in ela) and any(
        sp.simplify(e) != 0 for e in ela
    )
    half = dens.subs(w, sp.Rational(1, 2))
    ok_b = all(el(half, f) == 0 for f in (a1, a2, b1, b2))
    # flux: coefficient of the first derivatives
    flux = sp.expand(sum(half.coeff(sp.Derivative(f, r)) * f for f in (a1, a2, b1, b2)))
    # in terms of xi^r = U xi_r (real part) -> flat symbols
    flux_up = sp.simplify(flux.subs(b1, xir / Ur).subs(b2, 0))
    rep = {
        sp.Derivative(wr, r): w0p,
        wr: w0,
        Ur: U,
        Vr: V,
        Zr: Z,
        xir: sp.Symbol("xir"),
    }
    flux_flat = sp.expand(flux_up.subs(rep))
    return ok_a, ok_b, flux_flat


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEX)
