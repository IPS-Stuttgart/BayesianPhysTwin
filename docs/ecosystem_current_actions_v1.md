# Ecosystem current actions v1

## Purpose

`api/ecosystem-current-actions-v1.json` is the fail-closed operational
navigation record for BayesianPhysTwin, Prob4D, and Causal4D. It names the
small set of work that can currently change the scientific position and records
which target or confirmation boundaries must remain closed.

The registry is deliberately separate from claim records and run manifests.
It does not bind a mutable repository-head table, and a green validation result
is not empirical evidence. Closed owning issues are not retained as current
actions; their terminal evidence remains in the corresponding result records.

## Current ordering

1. Complete the source decision for the covariance-only independent
   confirmation. Confirmation remains closed until that source gate passes.
2. Resolve the independent-verifier governance blocker before any Causal4D
   confirmatory acquisition.
3. Freeze the retained CUT3R recurrent-online run/cohort bundle and execute its
   ordered source qualification. No additional provider, graph, covariance, or
   guard architecture may be added before a retained source result localizes a
   missing capability.

Each entry names its owning issue, next gate, blockers, target-access state, and
forbidden post-outcome actions. Priorities must be unique, ordered, and
contiguous.

## Recent terminal transition

The material-backend qualification action is no longer current. BayesianPhysTwin
issue #664 closed on 2026-08-20 after both registered candidates produced valid
bounded negative source-value decisions:

- `docs/genesis_mpm_zebra_source_value_v1_result.md` records a source-physics
  pass followed by a source-value failure for the frozen Genesis MPM arm; and
- `docs/jax_fem_zebra_source_value_v1_result.md` records a source-physics pass
  followed by a full-horizon physical-gate failure for the frozen JAX-FEM arm.

Neither candidate is source-value-qualified or eligible to replace the
incumbent. A future backend attempt requires a genuinely new physical mechanism
and separately frozen source protocol; it must not be represented as unfinished
work under issue #664.

## Validation

Run the local structural and boundary checks with:

```bash
python tools/quality/check_ecosystem_current_actions.py
pytest -q tests/test_ecosystem_current_actions.py
```

CI additionally runs:

```bash
python tools/quality/check_ecosystem_current_actions.py --check-github
```

The online audit reads issue metadata only. It rejects an owning issue or blocker
that is closed, missing, inaccessible, or actually a pull request. The current
repository token is used only for its owning repository; cross-repository public
issues are read without sending that repository-scoped token.

The complete checker rejects:

- omission, replacement, or reintroduction of a noncurrent action;
- noncontiguous priority changes;
- opening a target boundary without changing the registered protocol;
- loss or closure of the Causal4D independent-verifier blocker;
- substitution of the selected CUT3R recurrent-online source candidate;
- removal of any fail-closed prohibition;
- stale closed owning issues or PR numbers used as issues; and
- malformed repositories, issue references, blockers, or list fields.

A priority update must change the machine-readable registry and its validator
together. Exact scientific revisions remain in the corresponding protocol,
evidence decision, run manifest, or claim bundle.

## Scientific boundary

This artifact improves coordination and prevents contradictory roadmaps. It
does not establish estimator accuracy, uncertainty calibration, provider
competence, fresh-object transfer, Causal4D intervention benefit, deployment
safety, or state of the art.
