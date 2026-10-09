"""The index of paper-quoted numbers in published_values.py points at real entries,
and every number it lists is still typeset in the paper the way it is stored."""

import math
import pathlib
import re

import published_values as nv

# '7.4\times10^{-4}', '4.1 \times 10^5', or a bare power '10^{-14}'
_SCI = re.compile(r"^([0-9.]+)\s*\\times\s*10\^\{?(-?[0-9]+)\}?$")
_POW = re.compile(r"^10\^\{?(-?[0-9]+)\}?$")
_NUM = re.compile(r"10\^\{?-?[0-9]+\}?|[0-9.]+(?:\s*\\times\s*10\^\{?-?[0-9]+\}?)?")


def _parse(text):
    """(mantissa string, scale) of a typeset number."""
    m = _SCI.match(text)
    if m:
        return m.group(1), 10.0 ** int(m.group(2))
    m = _POW.match(text)
    if m:
        return None, 10.0 ** int(m.group(1))
    return text, 1.0


def _rounds_to(text, value):
    """Whether value rounds to the typeset string, '1.23' or '7.4\\times10^{-4}'."""
    mant, scale = _parse(text)
    assert mant is not None, f"a bare power of ten is not a point value: {text}"
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return round(float(value) / scale, decimals) == float(mant)


def _within(text, value):
    """Whether a measured value respects a stated precision: at most the stated
    number after rounding to its digits, or within half a decade of a bare power."""
    mant, scale = _parse(text)
    if mant is None:
        return math.log10(abs(value)) <= math.log10(scale) + 0.5
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return round(abs(value) / scale, decimals) <= float(mant)


def _tex():
    root = pathlib.Path(__file__).resolve().parent.parent / "paper"
    return "".join(p.read_text() for p in sorted(root.glob("**/*.tex")))


def test_every_paper_quoted_name_exists():
    missing = [k for k in nv.PAPER_QUOTED if not hasattr(nv, k)]
    assert not missing, f"PAPER_QUOTED names entries that do not exist: {missing}"


# A string of one or two significant digits ('0.18', '0.4', '7\times10^{-2}')
# occurs in the paper for many quantities, so finding it proves nothing.  Each
# such entry names the text around it instead: `{n}` stands for the number,
# whitespace matches any run of whitespace (line breaks in the source), and
# everything else is literal.
CONTEXT = {
    "TC_EXPONENT_LADDER": "e^{-{n}/\\sqrt",
    "MATCHING_RHOSTAR_OFFSET": r"\tfrac14\ln(\beta/6) + {n}",
    "THROAT_LADDER_WORST_AT_1E8": "worst deviation ${n}$ at $\\beta = 10^8$",
    "TC_RATIO_AT_K0_ZERO": "$97.1$ & ${n}$ &",
    "LADDER_K0_TOP": "$K_0/C_4$ saturates near $-{n}$",
    "TC_RATIO_AT_ALPHA_L": "$132.3$ & ${n}$ &",
    "F4_AT_TOP_RUNG": "C_\\triangle)/C_4 = -{n}$ at $\\beta = 10^8$",
    "KERNEL_LONGWAVE_PROBE": "falls from ${n}$ at $\\beta = 0$",
    "KERNEL_LONGWAVE_45": "to ${n}$ at $\\beta = 45$",
    "LADDER_TOP_MARGIN": "and ${n}$ at $\\beta = 10^8$",
    "T0_THRESHOLD_BA_SHORT": "$c_1/c_0 = -{n}$",
    "MATCHING_ERR_1E4": "from ${n}$ at $\\beta = 10^4$",
    "MATCHING_ERR_1E11": "to ${n}$ at $\\beta = 10^{11}$",
    "PLATEAU_SLOPE": "\\approx {n}$",
    "PHI_STAR_PLATEAU_LOW": "plateau of about ${n}$--",
    "PHI_STAR_PLATEAU_HIGH": "plateau of about $0.22$--${n}$",
    "ONSET_D5_BETA_MAX": "to $\\beta = {n}$",
    "ONSET_D6_BETA_MAX": "and ${n}$ respectively",
    "LADDER_ALPHA_1E4_SHORT": "$\\alpha$ moves from ${n}$ to",
    "LADDER_ALPHA_1E11_TWO": "moves from $0.4$ to ${n}$",
    "LADDER_BC_1E11_SHORT": "$B_c u_H^2 = {n}$",
    "D3_CROSS_T": "$T/\\sqrt B \\approx {n}$",
    "KERNEL_NONMONOTONE_BETA": "from $\\beta \\approx {n}$ the kernel",
    "WEIGHTED_KERNEL_CONCAVE_BETA": "not completely monotone from $\\beta \\approx {n}$",
}


def _significant_digits(text):
    mant = re.match(r"^[0-9.]*", text).group(0)
    return len(mant.replace(".", "").lstrip("0"))


def _number(text):
    """The typeset number as a regex that cannot match inside a longer number."""
    return r"(?<![0-9.])" + re.escape(text) + r"(?![0-9])"


def _context_regex(context, text):
    parts = [r"\s+".join(re.escape(w) for w in re.split(r"\s+", chunk))
             for chunk in context.split("{n}")]
    return _number(text).join(parts)


def test_every_short_paper_string_has_a_context():
    short = [name for name, (text, _) in nv.PAPER_STRINGS.items()
             if _significant_digits(text) <= 2 and name not in CONTEXT]
    assert not short, (f"{short}: one or two significant digits match unrelated "
                       f"numbers in the paper; add a CONTEXT entry in this file")
    stale = [name for name in CONTEXT if name not in nv.PAPER_STRINGS]
    assert not stale, f"CONTEXT names entries not in PAPER_STRINGS: {stale}"


def test_every_typeset_number_is_in_the_paper_and_matches():
    """Each PAPER_STRINGS entry is in the paper source as a whole number (not
    the head of a longer one), in its stated context where it has one, and the
    stored value rounds to it at the printed number of decimals."""
    tex = _tex()
    for name, (text, value) in nv.PAPER_STRINGS.items():
        assert _rounds_to(text, value), (
            f"{name}: stored {value} does not round to the typeset {text}"
        )
        if name in CONTEXT:
            assert re.search(_context_regex(CONTEXT[name], text), tex), (
                f"{name}: '{text}' not found in its context {CONTEXT[name]!r}"
            )
        else:
            assert re.search(_number(text), tex), (
                f"{name}: '{text}' not found in the paper as a whole number"
            )


def test_every_stated_precision_holds():
    """Each 'agree to X' / 'stable to X' in PAPER_BOUNDS: the string is in the
    paper and the stored measurement does not exceed X."""
    tex = _tex()
    for name, (text, measured) in nv.PAPER_BOUNDS.items():
        assert text in tex, f"{name}: '{text}' not found in the paper"
        numbers = _NUM.findall(text.replace("$", " "))
        stated = next(n for n in numbers if "10^" in n)
        assert _within(stated, measured), f"{name}: measured {measured:.2e} exceeds the stated {stated}"


def test_every_approximate_statement_holds():
    tex = _tex()
    for name, (text, value, low, high) in nv.PAPER_APPROX.items():
        assert text in tex, f"{name}: '{text}' not found in the paper"
        assert low <= value <= high, f"{name}: {value} outside [{low}, {high}] for '{text}'"


def test_uncovered_numbers_are_still_in_the_paper():
    """The list of quoted numbers with no stored value names real text, so a
    paper edit that changes one of them shows up here."""
    tex = _tex()
    gone = [k for k, (text, _) in nv.PAPER_UNCOVERED.items() if text not in tex]
    assert not gone, f"PAPER_UNCOVERED entries no longer in the paper: {gone}"


def test_pending_corrections_round_to_their_strings():
    """Corrections not yet typeset: the stored value already rounds to the
    string the paper will carry."""
    for name, (text, value) in getattr(nv, "PAPER_STRINGS_PENDING", {}).items():
        assert _rounds_to(text, value), f"{name}: stored {value} does not round to {text}"


# -- where each string is ------------------------------------------------------
# A trailing comment on a PAPER_STRINGS, PAPER_BOUNDS, PAPER_APPROX or
# PAPER_UNCOVERED entry names where the paper typesets it: a comma-separated
# list of 'abstract', 'sec. 4', 'sec. 4.6', 'app. D', 'app. D.5', 'table 4' or
# 'figure 1', optionally followed by '; free text'.  Every location listed must
# contain the string.
_LOC = r"(?:abstract|(?:sec|app)\. [0-9A-Z](?:\.[0-9]+)?|table [0-9]|figure [0-9])"
_LOCS = re.compile(rf"^({_LOC}(?:, {_LOC})*)(?:;|$)")


def _paper_by_location():
    """Paper text by location.  A section includes its subsections; floats are
    numbered in source order, which is the order LaTeX numbers them here."""
    root = pathlib.Path(__file__).resolve().parent.parent / "paper"
    main = (root / "main.tex").read_text()
    out = {}
    sec = sub = 0
    count = {"table": 0, "figure": 0}
    appendix = False
    for m in re.finditer(r"\\appendix\b|\\input\{([^}]*)\}", main):
        if m.group(0) == "\\appendix":
            appendix, sec = True, 0
            continue
        path = root / (m.group(1) + ".tex")
        if not path.exists():
            continue
        text = path.read_text()
        if m.group(1) == "abstract":
            out["abstract"] = text
            continue
        floating = None
        pos = 0
        marks = r"\\(sub)?section\{|\\begin\{(table|figure)\}|\\end\{(?:table|figure)\}"
        for tok in list(re.finditer(marks, text)) + [None]:
            end = tok.start() if tok else len(text)
            name = (chr(64 + sec) if appendix else str(sec)) if sec else None
            keys = [f"{'app' if appendix else 'sec'}. {name}"] if name else []
            if keys and sub:
                keys.append(f"{keys[0]}.{sub}")
            if floating:
                keys.append(floating)
            for k in keys:
                out[k] = out.get(k, "") + text[pos:end]
            if tok is None:
                break
            pos = tok.start()
            if tok.group(0) == "\\section{":
                sec, sub = sec + 1, 0
            elif tok.group(0) == "\\subsection{":
                sub += 1
            elif tok.group(2):
                count[tok.group(2)] += 1
                floating = f"{tok.group(2)} {count[tok.group(2)]}"
            else:
                floating = None
    return out


def _located_entries():
    """(dict name, entry name, locations or None if malformed) per commented entry."""
    src = pathlib.Path(nv.__file__).read_text()
    block = None
    for line in src.split("\n"):
        m = re.match(r"^(PAPER_[A-Z_]+) = \{", line)
        if m:
            block = m.group(1)
            continue
        if line.startswith("}"):
            block = None
        m = re.match(r'^    "([A-Z0-9_]+)": \(.*\),\s+#\s*(.*)$', line)
        if block in ("PAPER_STRINGS", "PAPER_BOUNDS", "PAPER_APPROX", "PAPER_UNCOVERED") and m:
            locs = _LOCS.match(m.group(2))
            yield block, m.group(1), locs.group(1).split(", ") if locs else None


def test_every_stated_location_contains_its_string():
    where = _paper_by_location()
    bad = []
    for block, name, locs in _located_entries():
        if locs is None:
            bad.append(f"{name}: comment does not start with a location list")
            continue
        text = getattr(nv, block)[name][0]
        for loc in locs:
            body = where.get(loc)
            if body is None:
                bad.append(f"{name}: no {loc} in the paper")
            elif block == "PAPER_STRINGS":
                pattern = _context_regex(CONTEXT[name], text) if name in CONTEXT else _number(text)
                if not re.search(pattern, body):
                    bad.append(f"{name}: '{text}' not in {loc}")
            elif text not in body:
                bad.append(f"{name}: '{text}' not in {loc}")
    assert not bad, "\n".join(bad)
