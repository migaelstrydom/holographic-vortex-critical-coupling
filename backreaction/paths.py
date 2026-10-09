"""Filesystem locations, resolved from the package rather than the cwd."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA = PACKAGE_ROOT / "data"
FIGURES = PACKAGE_ROOT / "figures"
SYSTEMS = PACKAGE_ROOT / "systems"


def data(name):
    """Absolute path of a committed .npz cache."""
    return DATA / name


def figure(name):
    """Absolute path of a generated figure."""
    return FIGURES / name
