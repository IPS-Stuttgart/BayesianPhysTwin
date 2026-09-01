# Retired one-shot GitHub Actions v1

This directory preserves the exact Git blobs of eighteen historical one-shot
GitHub Actions workflows removed from `.github/workflows` after the corresponding
executions reached a terminal state and their evidence boundaries were recorded.

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

The DLO4/DLO5 frozen-transfer launcher is preserved after its protected
execution lineage and evidence were retained and later superseded by completed
recovery and follow-on workflows. Its Python experiment and contract tests
remain active; only the historical Actions entry point is retired.

The DLO4/DLO5 decision-identifiability launcher is preserved after one-shot run
`33473378340` completed successfully, uploaded sealed evidence, and committed the
compact result under `results/science/deform_dlo45_decision_identifiability_v1`.
The request, protocol, implementation, tests, provenance, seals, and result stay
active or committed; only the consumed launcher is retired.

The Tracking Cloth hosted Zenodo mirror launcher is preserved after run
`33518742754` completed successfully and uploaded checksum-bound compact evidence
as artifact `9804820476`. The evaluated source, request, protocol, result, and
historical run remain unchanged; only the consumed transport/execution entry
point is retired to make room for the permanent DLO4/DLO5 terminal-evidence
router.

The visual-production failure diagnosis and its coupled reporter are preserved
here after all of their trusted-main runs failed without a diagnosis artifact or
published report. Their exact terminal history and later supersession are bound
by the machine-readable retirement record under `results/diagnostics`.

The Deform360 v6 frozen-upstream history locator is preserved after it completed
its target-blind purpose: the accepted locator report identified the unique
byte-identical historical physical-source snapshot subsequently materialized by
the protected source workflow. Its reusable Python locator and unit tests remain
active; only the now-completed manual GitHub Actions launcher is retired.

This archive changes workflow activation only. It does not alter any historical
run, artifact, protocol, estimator, metric, claim, target-access state, or
scientific result.
