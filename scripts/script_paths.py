"""Filesystem locations for the main tree, resolved from the file rather than
the cwd.  The counterpart of `backreaction/paths.py`.

Named `script_paths` and not `paths` deliberately.  `pyproject.toml` maps
`scripts/` onto the wheel root, so this module's importable name is top-level;
calling it `paths` would collide with `backreaction.paths` at any call site that
wants both -- which is exactly the call site this module was written for
(`backreaction/numerics/dk_kernel.py` reads the main tree's `kernel.npz`).

The main-tree caches sit beside the scripts that write them (`kernel.npz` from
`exchange_kernel.py`), where the backreaction package keeps its own under
`backreaction/data/`.  Diagnostic plots go to a top-level `figures/`, created
on first use.
"""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS.parent
DATA = SCRIPTS
FIGURES = PROJECT_ROOT / "figures"


def data(name):
    """Absolute path of a main-tree .npz cache."""
    return DATA / name


def figure(name):
    """Absolute path of a generated figure (creating figures/ if needed)."""
    FIGURES.mkdir(exist_ok=True)
    return FIGURES / name
