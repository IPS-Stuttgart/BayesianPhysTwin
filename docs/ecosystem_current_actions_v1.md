# Ecosystem current actions v1

## Purpose

`api/ecosystem-current-actions-v1.json` is the fail-closed operational
navigation record for BayesianPhysTwin, Prob4D, and Causal4D. It names the
small set of work that can currently change the scientific position, retains
terminal bounded outcomes that constrain later work, and records which target
or confirmation boundaries must remain closed.

The registry is deliberately separate from claim records and run manifests.
It does not bind a mutable repository-head table, and a green validation result
is not empirical evidence.

## Current ordering

1. Complete the source decision for the covariance-only independent
   confirmation. Confirmation remains closed until that source gate passes.
2. Resolve the independent-verifier governance blocker before any Causal4D
   confirmatory acquisition.
3. Freeze the retained CUT3R recurrent-online source run and complete
   object/session cohort bundle before executing the ordered real-provider
   source gates.
4. Retain the completed bounded-negative material-backend result. Genesis MPM
   passed source physics but failed its frozen source-value gate; JAX-FEM passed
   its short source-physics checks but failed the full-horizon physical gate
   before prefix outcomes were opened. Neither candidate remains active, and
   both preserve the registered incumbent fallback.

The fourth entry is a terminal constraint rather than active method work. It
remains in the registry so a later roadmap cannot silently reinterpret a
source-physics pass as source value, rerun either candidate under the same
identity, or open a fresh target from those rejected candidates.

Each entry names its owning issue, current lifecycle state, next gate or
retention action, blockers, target-access state, and forbidden post-outcome
actions. Priorities must be unique, ordered, and contiguous.

## State-aware validation

Run:

```bash
python tools/quality/check_ecosystem_current_actions.py
pytest -q tests/test_ecosystem_current_actions.py
```

The checker separates immutable policy from mutable progress. For each action it
binds ownership, the allowed lifecycle state machine, target-access semantics,
required blockers, and fail-closed prohibitions. A normal registered transition,
such as CUT3R moving from bundle preparation to active source gates, does not
require rewriting the validator. An unknown transition, premature target
opening, lost blocker, or reactivation of a terminal backend still fails closed.

The checker rejects:

- omission or replacement of a required action;
- noncontiguous priority changes;
- a status outside the registered lifecycle state machine;
- target access inconsistent with the current status;
- loss of the Causal4D independent-verifier blocker while the action is blocked;
- blockers retained after an action leaves its blocked state;
- active candidates attached to the completed backend action;
- an active-looking next step on a completed action;
- removal of any fail-closed prohibition; and
- malformed repositories, issue references, blockers, or list fields.

A new scientific protocol or genuinely new action changes the policy and
registry together. Ordinary progress within an already registered lifecycle
changes the registry snapshot only. Exact scientific revisions remain in the
corresponding protocol, evidence decision, run manifest, pull request, or claim
bundle.

## Scientific boundary

This artifact improves coordination and prevents contradictory roadmaps. It
does not establish estimator accuracy, uncertainty calibration, provider
competence, fresh-object transfer, Causal4D intervention benefit, deployment
safety, or state of the art.
