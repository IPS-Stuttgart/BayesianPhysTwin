# Prospective study lifecycle v1

## Purpose

`bayesian_phystwin.prospective_study_lifecycle_v1` provides one reusable,
content-addressed state machine for future source/target studies. It records the
information order and artifact handoffs that were previously reimplemented in
study-specific orchestration code.

The contract is additive. It does not alter any frozen Deform360, PokeFlex,
PhysTwin, Prob4D, Causal4D, or paper protocol, and it does not reinterpret a
completed result.

## Protocol lock

`ProspectiveStudyProtocolV1` binds:

- the method-set, decision-rule, fallback, and information-boundary identities;
- the independent statistical unit;
- pairwise-disjoint development, calibration, and target group rosters; and
- recursively immutable finite JSON metadata.

The protocol receives a canonical SHA-256 content identity. Group identifiers
are sorted and unique, and any overlap between source and protected target
rosters fails closed.

## Lifecycle

A normal positive or negative target study follows this sequence:

```text
design-locked
  -> source-predictions-sealed
  -> source-scored
  -> target-authorized
  -> target-predictions-sealed
  -> target-scored
  -> terminal-positive | terminal-negative
```

A source gate may instead produce `terminal-source-negative` before target
access. Any nonterminal state may produce `terminal-technical` while retaining
an explicit record of whether target payloads or outcomes had already opened.
Terminal states cannot advance.

Every transition binds exactly one new SHA-256 artifact identity and the exact
predecessor state. Existing prediction, score, authorization, or terminal
artifacts cannot be replaced. `validate_prospective_study_chain` recomputes the
complete chain and rejects changed ancestry, artifact substitution, illegal
stage skipping, or inconsistent target-access flags.

## Target-access boundary

The normal path enforces the following semantics:

- target payload and outcomes remain closed through `target-authorized`;
- `target-predictions-sealed` records an opened target payload but sealed target
  outcomes;
- target outcomes may be recorded only at `target-scored`; and
- source-negative termination keeps all protected target data closed.

A technical terminal can truthfully record premature payload or outcome access,
but it cannot turn that access into a valid scientific transition or close data
that were already recorded as open.

## Claim boundary

`terminal-positive` means only that the external terminal decision artifact
recorded a positive result under the bound protocol. Lifecycle states always
set `claim_authorized=false`. Paper promotion still requires the repository's
separate evidence-decision, claim-bundle, paper-handoff, and review boundaries.

Green tests for this module establish deterministic state construction,
content-addressed lineage, target-access ordering, and fail-closed transition
validation. They establish no provider competence, physical benefit,
calibration, intervention benefit, deployment safety, benchmark parity, or
state-of-the-art result.

## Example

```python
from bayesian_phystwin.prospective_study_lifecycle_v1 import (
    ProspectiveStudyProtocolV1,
    advance_prospective_study,
    lock_prospective_study,
)

protocol = ProspectiveStudyProtocolV1(
    protocol_id="example-v1",
    method_set_id=method_set_sha256,
    decision_rule_id=decision_rule_sha256,
    fallback_identity_id=fallback_sha256,
    information_boundary_id=boundary_sha256,
    statistical_unit="physical object session",
    development_group_ids=("source-01",),
    calibration_group_ids=("calibration-01",),
    target_group_ids=("target-01",),
)

state = lock_prospective_study(protocol)
state = advance_prospective_study(
    state,
    next_stage="source-predictions-sealed",
    artifact_id=source_prediction_bundle_sha256,
)
```
