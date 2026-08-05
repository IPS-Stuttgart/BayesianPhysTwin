# Development source preflight

## Purpose

BayesianPhysTwin has specialized workflows for GPU experiments, rendering,
cross-repository installation, and frozen evidence. Those workflows should not
be the first place a Python import-order or formatting defect is discovered.

The changed-Python preflight provides a lightweight first-line check without
changing any specialized scientific or compatibility gate.

## Local use

Install the development environment and enable the repository hooks once:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

The hooks use the repository's Ruff configuration and run, in order:

1. `ruff-check --fix` on Python and stub files;
2. `ruff-format` on the resulting source.

Run them explicitly across the repository with:

```bash
pre-commit run --all-files
```

The hooks modify local files when an automatic fix is available. Review and
commit those changes normally; pull-request workflows remain read-only.

## Pull-request workflow

`.github/workflows/changed-python-preflight.yml` checks the exact base and head
commit IDs from the pull request. It obtains the changed file set through a
NUL-delimited Git diff with deletion filtering and then runs only on ordinary,
repository-local Python files that still exist at the reviewed head.

For each changed Python file it runs:

```text
ruff check --output-format=github
ruff format --check --diff
python -m py_compile
```

The workflow also validates `.pre-commit-config.yaml` and exercises the
preflight script against temporary Git histories. It rejects unavailable
revisions, paths escaping the repository, non-files, and Python symlinks.

A manual exact-diff run is available through:

```bash
python scripts/ci/check_changed_python.py \
  --base <base-commit> \
  --head <head-commit>
```

## Boundary

This is an inexpensive source gate, not a substitute for:

- strict module-level mypy checks;
- unit or integration tests;
- wheel and source-distribution installation;
- the three-repository golden path;
- GPU/runtime validation; or
- claim-bearing experiment workflows.

Specialized jobs should depend on an equivalent lightweight source check when
that dependency prevents expensive runner allocation. The generic workflow is
still useful independently because it gives every Python pull request one
consistent exact-head lint, format, and compile result.
