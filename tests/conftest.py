"""Shared fixtures and the working-directory contract.

The numerics resolve data and figure paths relative to the working directory
(``scripts/kernel.npz``, ``figures/...``), and some modules read those files at
import time -- ``lattice_scan`` loads the kernel cache into module globals.
pytest imports test modules during collection, before any fixture runs, so the
chdir has to happen here at conftest import rather than in a fixture.

Importing the numerics themselves needs no path setup: both script directories
are installed as top-level modules (see pyproject.toml).

References such as "paper sec. 4.2" or "app. D.5" in the test docstrings and
comments are to sections of the paper.
"""

import os
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT
