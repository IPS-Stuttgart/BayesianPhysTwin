# Ecosystem current actions v1

## Purpose

`api/ecosystem-current-actions-v1.json` is the fail-closed operational
navigation record for BayesianPhysTwin, Prob4D, and Causal4D. It names the
small set of work that can currently change the scientific position and records
which target or confirmation boundaries must remain closed.

The registry is deliberately separate from claim records and run manifests.
It does not bind a mutable repository-head table, and a green validation result
is not empirical evidence.

## Current ordering

1. Complete the source decision for the covariance-only independent
   confirmation. Confirmation remains closed until that source gate passes.
2. Resolve the independent-verifier governance blocker before any Causal4D
   confirmatory acquisition.
3. Freeze support feasibility for a separately versioned real Prob4D provider
   before source residuals are opened.
4. Advance Genesis MPM or JAX-FEM beyond their retained source-physics passes
   and failed first source-value arms. No new backend family may enter the
   canonical registry before one active candidate reaches source value.

Each entry names its owning issue, next gate, blockers, target-access state, and
forbidden post-outcome actions. Priorities must be unique, ordered, and
contiguous.

## Validation

Run:

```bash
python tools/quality/check_ecosystem_current_actions.py
pytest -q tests/test_ecosystem_current_actions.py
```

The checker rejects:

- omission or replacement of a required action;
- noncontiguous priority changes;
- opening a target boundary without changing the registered protocol;
- loss of the Causal4D independent-verifier blocker;
- changes to the active Genesis MPM/JAX-FEM qualification roster;
- removal of any fail-closed prohibition; and
- malformed repositories, issue references, blockers, or list fields.

A priority update must change the machine-readable registry and its validator
together. Exact scientific revisions remain in the corresponding protocol,
evidence decision, run manifest, or claim bundle.

## Scientific boundary

This artifact improves coordination and prevents contradictory roadmaps. It
does not establish estimator accuracy, uncertainty calibration, provider
competence, fresh-object transfer, Causal4D intervention benefit, deployment
safety, or state of the art.
