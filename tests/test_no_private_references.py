"""The shipped code and tests cite the paper, and nothing else.

Results are referred to by the paper's sections, appendices, tables and
figures.  This guard fails on a reference to anything a reader of the paper
cannot see: a development tag such as B27 or §B44 in a comment or docstring, a
mention of notebooks or reviews anywhere, or such a tag in an identifier.  The
string arrays inside the shipped .npz caches (their `readme` and key lists) are
prose too and are held to the same rule.
"""

import io
import pathlib
import re
import tokenize

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCANNED = ("backreaction", "scripts", "tests")

# In prose (comments and docstrings): a tag such as B27, §B44 or (R7), or a
# mention of notebooks, reviews or development directions.  Check labels such
# as [B3] and quoted names such as 'B2' are not tags.
PROSE = re.compile(
    r"(?<![\['])\bB\d{1,2}\b(?![\]'])|§B\d|\(R\d{1,2}[);,]"
    r"|[Nn]otebook|[Rr]eviewer|mock review|[Dd]irection \d"
)
# In string literals that are not docstrings, B2 and the like are ordinary
# names, so only the unambiguous mentions count.
CODE = re.compile(r"§B\d|[Nn]otebook|[Rr]eviewer|mock review|[Dd]irection \d")
# In identifiers: a lower-case tag (test_..._b30, ..._on_part_i_...) or a
# notebook mention.  Upper-case names such as per_B2 or B2 are physics.
NAME = re.compile(r"(?:^|_)(?:b\d{2}|part_[ivx]+)(?:_|$)|[Nn]otebook")


def _files():
    me = pathlib.Path(__file__).resolve()
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            if path.resolve() != me and "__pycache__" not in path.parts:
                yield path


def _caches():
    for top in SCANNED:
        yield from sorted((ROOT / top).rglob("*.npz"))


def _cache_leaks():
    """file[key]: text for every private reference in a cache's string arrays."""
    found = []
    for path in _caches():
        rel = path.relative_to(ROOT)
        try:
            with np.load(path, allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
        except ValueError:
            found.append(f"{rel}: holds an object array, which this guard cannot read")
            continue
        for key, arr in arrays.items():
            if arr.dtype.kind not in "US":
                continue
            for item in arr.ravel():
                text = item.decode() if isinstance(item, bytes) else str(item)
                m = PROSE.search(text)
                if m:
                    found.append(f"{rel}[{key}]: {m.group(0)!r}")
    return found


def leaks():
    """file:line: text for every private reference in the shipped Python and caches."""
    found = []
    for path in _files():
        rel = path.relative_to(ROOT)
        prev = tokenize.NEWLINE
        for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
            if tok.type == tokenize.NAME:
                pattern = NAME
            elif tok.type == tokenize.COMMENT or (
                tok.type == tokenize.STRING
                and prev in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)
            ):
                pattern = PROSE
            elif tok.type == tokenize.STRING:
                pattern = CODE
            else:
                pattern = None
            m = pattern.search(tok.string) if pattern else None
            if m:
                line = tok.start[0] + tok.string.count("\n", 0, m.start())
                found.append(f"{rel}:{line}: {m.group(0)!r}")
            if tok.type not in (tokenize.COMMENT, tokenize.NL):
                prev = tok.type
        if "notebook" in path.name:
            found.append(f"{rel}: file name")
    return found + _cache_leaks()


def test_no_private_references():
    found = leaks()
    assert not found, "private references in the shipped code:\n" + "\n".join(found)
