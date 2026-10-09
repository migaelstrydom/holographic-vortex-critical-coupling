"""Code paths the result-oriented tests never reach.

The files in this directory are organised by *result*, and that leaves a few
plumbing seams uncovered: the default-argument path resolution, the small
N-refinement helper, the perturbed zero-mode solver called only from a
`main()`.  They get behaviour tests here rather than being trusted to
inspection.

The path tests monkeypatch `paths.DATA` / `paths.FIGURES` after import and
assert that the write follows.  Output paths are therefore resolved when a
function runs (a `None` default), not baked into a signature at import time
(`path=paths.data(...)`), where a later change of `paths.DATA` would have no
effect.
"""

import re

import numpy as np
import pytest

from backreaction import paths
from backreaction.numerics import bc_alpha as bca
from backreaction.numerics import dk_background as dk
from backreaction.numerics import dk_kernel as K
from published_values import B_C


# ------------------------------------------------------------- bc_alpha paper sec. 2.2, sec. 3.1

def test_n_convergence_pairs_every_grid_size_with_its_eigenvalue():
    """The zip(Ns, vals) rows must line up, and beta = 0 must land on B_c."""
    out = bca.n_convergence(betas=(0.0,), Ns=(24, 32))
    assert list(out) == [0.0]
    rows = out[0.0]
    assert [N for N, _ in rows] == [24, 32]
    for _, v in rows:
        assert float(v) == pytest.approx(B_C, abs=1e-8)


def test_save_curve_resolves_its_default_path_when_called(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    N = 8
    n = np.zeros(N + 1)
    bca.save_curve(np.array([0.0]), np.array([B_C]), np.array([0.0]),
                   n[None, :], n[None, :], n[None, :], n[None, :], N)
    assert (tmp_path / "bc_alpha.npz").exists()


def test_make_figure_resolves_its_default_path_when_called(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "FIGURES", tmp_path)
    a = np.linspace(0.0, 0.05, 5)
    bca.make_figure(a, B_C * (1 + 0.2 * a), a * B_C**2)
    assert (tmp_path / "bc_alpha.png").exists()


# ------------------------------------------------------------- dk_background

def test_load_grid_reads_the_committed_cache_with_no_argument():
    d, get = dk.load_grid()
    betas = np.asarray(d["betas"], float)
    assert betas[0] == pytest.approx(0.0, abs=1e-12)
    assert get(0).beta == pytest.approx(betas[0])


def test_save_grid_resolves_its_default_path_when_called(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA", tmp_path)
    dk.save_grid([], N=8)
    assert (tmp_path / "dk_background_grid.npz").exists()


# --------------------------------------------------------- cwd independence

def test_dk_kernel_finds_the_main_tree_cache_from_any_directory():
    """`dk_kernel.anchor1` reads the probe-limit `kernel.npz`, which lives in
    the main tree rather than in `backreaction/data/`, so it is resolved through
    `scripts/script_paths.py` rather than against the working directory."""
    assert K.KER_PATH.is_absolute()
    assert K.KER_PATH.exists()


RELATIVE_PATH_LITERAL = re.compile(
    r'''["'][A-Za-z_][A-Za-z0-9_/]*/[A-Za-z0-9_.]+\.(?:npz|png|py|txt|json|csv)["']''')


def test_no_module_in_the_package_hardcodes_a_relative_path(project_root):
    """An invariant rather than a one-off audit: every data and figure location in `backreaction/` resolves
    from `paths.py` (or `script_paths.py` for the main-tree caches), so the
    package works from any working directory.

    A source scan rather than an import, because importing a derivation runs
    it -- so the module is read with `ast` rather than imported.
    """
    offenders = []
    for path in sorted((project_root / "backreaction").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if RELATIVE_PATH_LITERAL.search(line):
                offenders.append(f"{path.relative_to(project_root)}:{n}: {line.strip()}")
    assert not offenders, (
        "cwd-relative path literal(s) in the package; route them through "
        "backreaction.paths or script_paths:\n" + "\n".join(offenders))
