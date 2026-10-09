"""Provenance stamping for the auto-generated modules in `systems/`.

Those modules are SymPy output written to disk by the derivations in
`derivations/` and imported by the numerics.  They must never be edited by
hand, and nothing enforced that; nor did anything record *which* version of a
derivation produced the committed file.  The seam was invisible in both
directions -- a hand edit and a stale regeneration looked identical to a clean
tree.

Each generated file therefore ends with a provenance block:

    # generator:             dk_response.py
    # generator_code_sha256: <digest of the generator's parsed AST>
    # content_sha256:        <sha256 of every byte above the block>
    # python:                <major.minor of the interpreter that stamped it>

which supports two checks of different strength and cost (see
`tests/test_generated_systems.py`):

* **content_sha256** -- cheap and exact.  Detects any hand edit of the generated
  file.  Runs in the fast tier.
* **generator_code_sha256** -- cheap.  Detects that the generator's *code*
  changed since the file was last generated.  It digests the parsed AST rather
  than the raw bytes, so comment, whitespace and docstring edits do not trip it
  (see `code_digest`); any real code change does.  The fix is to re-run the
  generator, which re-stamps.
* re-running the generator and diffing the output is the real check, and is the
  only one that proves the committed file is current.  It costs minutes, so it
  lives in the slow tier.

Limits of the cheap checks, accepted deliberately:

* `ast.dump` is not guaranteed stable across Python minor versions, so the
  generator digest is only comparable under the interpreter that stamped it
  (recorded in the `python` field; the pinned `.python-version` is 3.14, and
  stamps written before the field existed are all 3.14).  Under another
  version `check_generator` reports the digest as not comparable instead of
  as changed, and the test skips; re-running the generator settles it.
* The digest covers the generator file only, not the modules it imports
  (for example a shared conventions module or another generated module in
  `systems/`) or the SymPy version (pinned in `uv.lock`).  A change there is caught by the slow-tier re-run,
  not by the fast tier.

Usage from a generator, as the last statement after writing its output:

    from backreaction.generated import stamp
    stamp(gen_path, __file__)
"""

import ast
import hashlib
import pathlib
import re
import sys

MARKER = "# --- provenance " + "-" * 58

# Interpreter that wrote the stamps that predate the `python` field.
LEGACY_STAMP_PYTHON = "3.14"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


_FIELD = re.compile(
    r"^#\s*(generator|generator_code_sha256|content_sha256|python):\s*(\S+)\s*$"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_digest(source: str) -> str:
    """Digest of a generator's *code*, blind to comments, docstrings and layout.

    Hashing the raw bytes would make every comment edit look like a reason to
    re-run the generator -- and the slowest generators take several
    minutes, so that cost is not academic.  Hashing the parsed AST instead
    (with docstrings stripped) is invariant under comments, whitespace and
    docstring rewording, while still changing on any real edit: swapping
    `sp.cancel` for `sp.simplify`, reordering a solve, changing a coefficient.

    The trade is deliberate and documented: a change confined to a docstring or
    a comment will not be flagged.  That is acceptable because the *output* is
    still pinned byte-for-byte by `content_sha256`, and true currency is
    established only by re-running the generator (the slow-tier test).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body:
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                body.pop(0)
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def _split(path):
    """Return (body_bytes, fields) for a generated file.

    `body_bytes` is everything above the provenance block -- exactly the bytes
    `content_sha256` covers.  `fields` is empty if the file carries no block.
    """
    raw = pathlib.Path(path).read_bytes()
    text = raw.decode("utf-8")
    # First occurrence: the block is opening-marker .. fields .. closing-marker,
    # so rfind would land on the closing line and parse no fields at all.
    marker_at = text.find(MARKER)
    if marker_at == -1:
        return raw, {}
    # The block is the LAST thing in the file: opening marker, fields, closing
    # marker, nothing else.  Requiring that means a marker-like line inside the
    # generated body (SymPy is free to emit anything) cannot be mistaken for the
    # start of the block and silently truncate what `content_sha256` covers.
    tail = text[marker_at:]
    if tail.count(MARKER) != 2 or not tail.rstrip().endswith(MARKER):
        raise ValueError(
            f"{path}: malformed provenance block -- expected exactly one block "
            f"at end of file, found {tail.count(MARKER)} marker line(s)"
        )
    body = text[:marker_at].encode("utf-8")
    fields = {}
    for line in text[marker_at:].splitlines():
        m = _FIELD.match(line)
        if m:
            fields[m.group(1)] = m.group(2)
    return body, fields


def stamp(path, generator_file):
    """Append (or refresh) the provenance block on a just-written file."""
    path = pathlib.Path(path)
    generator = pathlib.Path(generator_file)
    body, _ = _split(path)  # drop any previous block
    # Normalise the separator to exactly one blank line, so that the bytes we
    # hash here are byte-identical to the bytes `_split` will hand back on
    # verification, and so that re-stamping is idempotent rather than growing a
    # newline (and a stale hash) on every call.
    body = body.rstrip(b"\n") + b"\n\n"
    block = (
        f"{MARKER}\n"
        f"# Written by {generator.name}; do not edit by hand.  Verified by\n"
        f"# tests/test_generated_systems.py -- see backreaction/generated.py.\n"
        f"# generator:             {generator.name}\n"
        f"# generator_code_sha256: {code_digest(generator.read_text())}\n"
        f"# content_sha256:        {_sha256(body)}\n"
        f"# python:                {_python_version()}\n"
        f"{MARKER}\n"
    )
    path.write_bytes(body + block.encode("utf-8"))
    return path


def read_stamp(path):
    """Provenance fields of a generated file; `{}` if it carries no block."""
    return _split(path)[1]


def check_content(path):
    """(ok, detail) -- does content_sha256 still match the file's own body?"""
    body, fields = _split(path)
    if not fields:
        return False, "no provenance block"
    want = fields.get("content_sha256")
    got = _sha256(body)
    if want is None:
        return False, "provenance block has no content_sha256"
    if got != want:
        return False, f"content_sha256 {got[:12]} != stamped {want[:12]}"
    return True, "content matches stamp"


def stamped_python(path):
    """major.minor of the interpreter that stamped `path`."""
    return read_stamp(path).get("python", LEGACY_STAMP_PYTHON)


def check_generator(path, generator_dir=None):
    """(ok, detail) -- is the generator's source still the stamped version?

    `ok` is None when the digest is not comparable: the file was stamped under
    a different Python minor version, whose `ast.dump` may differ.
    """
    path = pathlib.Path(path)
    fields = read_stamp(path)
    if not fields:
        return False, "no provenance block"
    if stamped_python(path) != _python_version():
        return None, (
            f"stamped under Python {stamped_python(path)}, running "
            f"{_python_version()}: generator digests are not comparable"
        )
    name = fields.get("generator")
    if name is None:
        return False, "provenance block names no generator"
    gen = pathlib.Path(generator_dir or path.parent) / name
    if not gen.exists():
        return False, f"generator {name} not found"
    got = code_digest(gen.read_text())
    want = fields.get("generator_code_sha256")
    if got != want:
        return False, (
            f"{name} changed since {path.name} was generated "
            f"({got[:12]} != stamped {want[:12]}); re-run it"
        )
    return True, "generator unchanged"
