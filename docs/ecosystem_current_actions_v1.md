# Ecosystem current actions v1

## Purpose

`api/ecosystem-current-actions-v1.json` is the generated fail-closed
operational navigation record for BayesianPhysTwin, Prob4D, and Causal4D. It
names the small set of work that can currently change the scientific position
and records which target or confirmation boundaries must remain closed.

The lifecycle source of truth is
`api/ecosystem-action-records-v1.json`. Current records are rendered into the
public snapshot; completed records remain in the source ledger with their
terminal evidence and cannot silently re-enter the current priority list.

The registry is deliberately separate from claim records and run manifests. A
green validation result proves synchronization and lifecycle consistency, not
empirical accuracy or scientific promotion.

## Current ordering

1. Complete the source decision for the covariance-only independent
   confirmation. Confirmation remains closed until that source gate passes.
2. Resolve the independent-verifier governance blocker before any Causal4D
   confirmatory acquisition.
3. Freeze support feasibility for a separately versioned real Prob4D provider
   before source residuals are opened.

Each current entry names its owning issue, next gate, blockers, target-access
state, and forbidden post-outcome actions. Priorities must be unique and
contiguous.

## Completed backend qualification

The former fourth action, material-backend qualification under issue `#664`, is
terminal rather than active:

- Genesis MPM passed source physics and failed the frozen source-value gate;
- JAX-FEM passed its short source-physics gate and failed the outcome-blind
  full-horizon physical gate.

The lifecycle record binds both canonical result documents. Neither candidate
is source-value-qualified, and neither opened target, confirmation, Causal4D,
DEFORM, or held-v8 outcomes. A future backend attempt requires a genuinely new
physical mechanism and a separately frozen source protocol.

## Updating the registry

Edit only the lifecycle source, then regenerate:

```bash
python tools/quality/render_ecosystem_current_actions.py
python tools/quality/check_ecosystem_current_actions.py
pytest -q tests/test_ecosystem_current_actions.py
```

CI also runs:

```bash
python tools/quality/render_ecosystem_current_actions.py --check
python tools/quality/check_ecosystem_action_issue_states.py
```

The first command rejects manual drift in the generated snapshot. The second
queries each owning GitHub issue and rejects a `current` record whose issue is
closed or a `terminal` record whose issue is open. The scheduled audit catches
issue-state changes that occur without a repository edit.

The validators additionally reject:

- omission or replacement of a required current or terminal action;
- noncontiguous current priorities;
- opening a target boundary without a separately versioned protocol;
- loss of the Causal4D independent-verifier blocker;
- loss of either backend terminal result;
- removal of any fail-closed prohibition;
- source snapshots older than a bound terminal transition; and
- malformed repositories, issue references, paths, blockers, or list fields.

Exact scientific revisions remain in the corresponding protocol, evidence
decision, run manifest, terminal result, or claim bundle.

## Scientific boundary

These artifacts improve coordination and prevent contradictory roadmaps. They
do not establish estimator accuracy, uncertainty calibration, provider
competence, fresh-object transfer, Causal4D intervention benefit, deployment
safety, or state of the art.
