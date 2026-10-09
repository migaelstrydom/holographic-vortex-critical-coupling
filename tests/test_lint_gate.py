"""Ruff stays clean on the package -- both `check` and `format`.

`backreaction/` was brought to a clean `ruff check` (the
`F`/`B`/`SIM`/`UP` rule set selected in pyproject.toml).  Nothing enforced that
afterwards, so the next `zip()` without `strict=` or the next dead binding would
land unnoticed and the bucket would grow back.  This is a cheap gate: it runs
the same command the developer would.

The formatter was run over the package and this gate keeps it
that way, so that a later diff shows the change rather than the reflow.

The full rule set and the formatter cover `backreaction/` only:
`scripts/` and `tests/` are in ruff's `extend-exclude` and keep their own
layout.  They are held to the pyflakes rules (`F`: unused imports, dead
bindings, undefined names) by a third gate.  `backreaction/systems/`
is excluded from the *formatter* alone (`[tool.ruff.format]` in pyproject.toml):
its layout is chosen by the generators and pinned byte-for-byte by
`generated.stamp()`, so reformatting it would invalidate every content_sha256.
No test is needed for that exclusion -- `test_generated_systems.py`'s
`test_content_matches_stamp` fails immediately if the formatter is ever let in.
"""

import importlib.util
import subprocess
import sys

import pytest

# The ruff pinned in uv.lock (dev group), run through this interpreter rather
# than whatever `ruff` is first on PATH: format output is version-dependent.
HAVE_RUFF = importlib.util.find_spec("ruff") is not None
RUFF = [sys.executable, "-m", "ruff"]


@pytest.mark.skipif(not HAVE_RUFF, reason="ruff not installed (dev dependency)")
def test_ruff_check_is_clean_on_the_backreaction_package(project_root):
    proc = subprocess.run([*RUFF, "check", "backreaction"],
                          cwd=project_root, capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0, (
        "ruff check backreaction is no longer clean:\n"
        + proc.stdout + proc.stderr)


@pytest.mark.skipif(not HAVE_RUFF, reason="ruff not installed (dev dependency)")
def test_ruff_format_is_clean_on_the_backreaction_package(project_root):
    proc = subprocess.run([*RUFF, "format", "--check", "backreaction"],
                          cwd=project_root, capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0, (
        "ruff format would reformat these files; run `uv run ruff format "
        "backreaction`:\n" + proc.stdout + proc.stderr)


@pytest.mark.skipif(not HAVE_RUFF, reason="ruff not installed (dev dependency)")
def test_pyflakes_rules_are_clean_on_scripts_and_tests(project_root):
    proc = subprocess.run([*RUFF, "check", "--select", "F", "scripts", "tests"],
                          cwd=project_root, capture_output=True, text=True,
                          check=False)
    assert proc.returncode == 0, (
        "ruff check --select F scripts tests is no longer clean:\n"
        + proc.stdout + proc.stderr)
