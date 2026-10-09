"""Tests for the `backreaction` package, mirroring its place in the tree.

One constraint on this directory, and it is load-bearing: **do not add an
`__init__.py` here.**  `pyproject.toml` puts `tests/` on `sys.path` (so the
tests can `import published_values`), so a directory called `backreaction`
underneath it sits on the import path next to the real package.  Without an
`__init__.py` it is only a namespace-package *portion*, and the import machinery
keeps scanning and finds the real regular package -- whatever the path order.
With one it would be a regular package itself and would shadow the code under
test, silently, with every import failing far from the cause.

The files here are organised by result rather than by module, because a
result cuts across `derivations/`, `numerics/` and `anchors/`: it is typically
a symbolic derivation, a generated system and a numeric solve, and testing them
apart would test none of the seams between them.

The working-directory contract and the shared fixtures come from `tests/conftest.py`.
"""
