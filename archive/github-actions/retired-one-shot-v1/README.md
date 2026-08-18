# Retired one-shot GitHub Actions v1

This directory preserves the exact Git blobs of twelve historical one-shot
GitHub Actions workflows removed from `.github/workflows` after their registered
executions and terminal evidence records were complete.

Files below this directory are **not active GitHub Actions entry points**.
`manifest.json` binds every original path, archived path, Git blob SHA-1, and
byte count. It also binds the exact historical contract-test sources that still
exercise these workflow bytes from the normal test suite.

Validate the archive and the inactive-path boundary with:

```bash
python tools/quality/check_retired_workflow_archive.py
pytest -q tests/test_retired_workflow_archive.py
```

The archived workflow files reuse the original Git blobs; they were not
reformatted or regenerated during retirement. Restoring an archived launcher to
`.github/workflows` requires a separately reviewed workflow and protocol
decision. Copying a historical file back into the active directory is not an
authorized rerun.

This archive changes workflow activation only. It does not alter any historical
run, artifact, protocol, estimator, metric, claim, target-access state, or
scientific result.
