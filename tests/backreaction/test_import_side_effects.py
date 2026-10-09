"""Importing a package module does no work: no output, no files, no waiting.

Every script under `derivations/`, `anchors/` and `numerics/` does its work in
`main()` behind an `if __name__ == "__main__"` guard, so that importing one for
its definitions (as `numerics/dk_throat.py` imports `numerics/dk_tower.py`, or a
test imports a module to read its helpers) neither runs a derivation nor
rewrites a generated module in `systems/` or a cache in `data/`.

All modules are imported in one fresh subprocess, so the test sees a cold
import of each.  It checks that the import prints nothing, that no file under
`systems/`, `data/` or `figures/` changes, and that the whole sweep stays well
inside the fast-tier budget.
"""

import json
import pathlib
import subprocess
import sys

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "backreaction"
SUBPACKAGES = ("derivations", "anchors", "numerics")
OUTPUT_DIRS = ("systems", "data", "figures")
BUDGET_S = 60.0

MODULES = sorted(
    f"backreaction.{sub}.{p.stem}"
    for sub in SUBPACKAGES
    for p in (PACKAGE / sub).glob("*.py")
    if p.stem != "__init__"
)

_PROBE = r"""
import contextlib, importlib, io, json, sys, time
report = {}
t_all = time.perf_counter()
for name in sys.argv[1:]:
    buf = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        importlib.import_module(name)
    report[name] = {"seconds": time.perf_counter() - t, "stdout": buf.getvalue()}
report["__total__"] = {"seconds": time.perf_counter() - t_all, "stdout": ""}
print(json.dumps(report))
"""


def _snapshot():
    return {
        str(p.relative_to(PACKAGE)): (p.stat().st_mtime_ns, p.stat().st_size)
        for d in OUTPUT_DIRS
        for p in (PACKAGE / d).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }


def test_the_module_list_is_not_empty():
    assert len(MODULES) > 40, MODULES


def test_importing_every_module_is_side_effect_free(project_root):
    before = _snapshot()
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, *MODULES],
        cwd=project_root, capture_output=True, text=True,
        timeout=3 * BUDGET_S, check=False,
    )
    after = _snapshot()
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = json.loads(proc.stdout.strip().splitlines()[-1])

    printed = {m: r["stdout"][:200] for m, r in report.items() if r["stdout"]}
    assert not printed, f"modules print at import (work outside main()): {printed}"

    changed = sorted(k for k in before.keys() | after.keys()
                     if before.get(k) != after.get(k))
    assert not changed, f"importing the package modules wrote files: {changed}"

    total = report["__total__"]["seconds"]
    slowest = max((r["seconds"], m) for m, r in report.items() if m != "__total__")
    assert total < BUDGET_S, (
        f"importing every module took {total:.1f} s (slowest {slowest[1]}: "
        f"{slowest[0]:.1f} s); some module does real work at import")
