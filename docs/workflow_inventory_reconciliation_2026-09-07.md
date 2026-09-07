# Workflow inventory reconciliation (2026-09-07)

The frozen code-review base `9d7383ea56a0a9e3ad6753d1c42fe653cd7e615d`
already contains 90 active workflow files, while the historical v1 inventory
contract expects 88. The inherited full-suite failure is the exact count check,
not a GP numerical failure. PR #952's retained Python 3.10 receipt (run
34054043990, artifact 9995469028) reports this as its only failed test.

This maintenance change explicitly adopts the already-merged 90-file baseline
in `.github/quality/workflow-inventory-budget-v2.json`. It adds no active
workflow, deletes no existing execution route, and does not silently relax v1.
The v1 bytes and all historical rule tests remain. The operational test selects
v2 explicitly and continues to reject growth or unrecorded shrinkage. The
retirement target remains 81, and the temporary-looking allowlist remains empty.

This is an infrastructure-policy successor, not a scientific protocol amendment.
The twelve imported GP/DP PRs propose fifteen workflow files; all fifteen are
kept inactive in `archive/research-workflows/residual-models-2026-09-07/` instead.
The existing changed-source preflight owns the one maintained CPU-only test job.

Current inventory check:

```bash
python tools/quality/check_workflow_inventory_budget.py \
  --contract .github/quality/workflow-inventory-budget-v2.json
```

The generic validator still defaults to v1 for compatibility with its historical
fixtures. Current automation passes v2 explicitly. No numerical gate, frozen
result, dataset split, or scientific continuation threshold is changed.
