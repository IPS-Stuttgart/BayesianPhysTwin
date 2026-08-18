# Material-backend evidence promotion v1

## Purpose

The canonical material-backend registry records transport and integration
maturity. It deliberately does **not** imply physical accuracy. A tested adapter,
a valid portable rollout, and even a pinned native simulator execution are
necessary engineering milestones, but none establishes that the backend is a
useful physical prior on real objects.

`bayesian_phystwin.material_backend_evidence_v1` adds a separate, executable
promotion ladder. Each status is content-addressed, applies to one canonical
family and exact producer profile, and becomes runtime-specific as soon as a
native execution is claimed.

## Evidence stages

| Code | Stage | Meaning |
| --- | --- | --- |
| T0 | `transport-registered` | The family and producer transport are registered. |
| T1 | `adapter-tested` | Dependency-free adapter behavior has a bound evidence digest. |
| T2 | `native-runtime-replayed` | One exact native runtime produced a replayable artifact. |
| T3 | `numerically-qualified` | A passing `MaterialBackendQualificationV1` binds the runtime, source groups, numerical gates, information order, and exact fallback. |
| T4 | `source-competent` | A passing source-only `EvidenceDecisionV1` binds the qualification and exact runtime. |
| T5 | `fresh-object-validated` | An authorized confirmatory decision binds disjoint target object/session groups to the source-selected runtime. |
| T6 | `downstream-query-benefit` | A separately authorized physical-query or Causal4D decision binds the fresh-object result. |

Stages are contiguous. A backend cannot skip from adapter coverage to source or
target competence, and one runtime cannot borrow another runtime's qualification
or decision.

The canonical machine-readable table is available through:

```python
from bayesian_phystwin.material_backend_evidence_v1 import (
    describe_material_backend_evidence_stages,
)

print(describe_material_backend_evidence_stages())
```

## Building a status

The builder accepts only exact evidence objects for claim-bearing stages:

```python
from bayesian_phystwin.material_backend_evidence_v1 import (
    build_material_backend_evidence_status_v1,
)

status = build_material_backend_evidence_status_v1(
    canonical_profile_id="jax-fem-quasistatic-v1",
    producer_profile_id="jax-fem-quasistatic-v1",
    adapter_evidence_id=adapter_digest,
    runtime_id=runtime_digest,
    native_replay_evidence_id=native_replay_digest,
    qualification=qualification,
    source_decision=source_decision,
    target_decision=target_decision,
    target_group_ids=target_object_session_ids,
)
```

The builder verifies that:

- the producer belongs to the canonical family and uses the registered
  transport;
- adapter, native-replay, qualification, source, target, and downstream bindings
  are contiguous;
- `runtime_id` and native replay are supplied together;
- the qualification is passing for the exact runtime and verifies exact
  fallback;
- the status source roster equals the qualification source roster;
- source and target object/session identifiers are disjoint;
- source competence is scientific source evidence and does not authorize a
  target-facing claim;
- target and downstream decisions are passing, confirmatory, and explicitly
  authorize their bounded claims; and
- each decision metadata record binds the expected parent decision or
  qualification identity.

## Required decision metadata

A source-competence decision must contain:

```json
{
  "evidence_role": "source-competence",
  "canonical_profile_id": "<family>",
  "producer_profile_id": "<producer>",
  "runtime_id": "<sha256>",
  "qualification_artifact_id": "<sha256>"
}
```

A fresh-object decision replaces the last field with
`source_decision_id`. A downstream-query decision uses
`target_decision_id`. This prevents a numerically valid but unrelated decision
from being substituted into the promotion chain.

## Target-facing admission

Loading a status validates its schema, content address, contiguous fields, and
roster separation. Target-facing code must additionally replay the external
evidence objects:

```python
from bayesian_phystwin.material_backend_evidence_v1 import (
    require_material_backend_evidence_stage,
)

require_material_backend_evidence_stage(
    status,
    "fresh-object-validated",
    qualification=qualification,
    source_decision=source_decision,
    target_decision=target_decision,
)
```

Supplying only the status JSON is insufficient for T3 and above. This prevents a
self-asserted stage string from becoming an admission certificate.

## Negative results and changed runtimes

A failed qualification or decision is complete evidence for its exact runtime
and opened cohort. It must not be converted to a passing stage by deleting
objects, changing thresholds, replacing streams, or retuning on the same
outcomes. A materially changed solver, scene, topology, parameterization,
producer, or runtime receives a new runtime identity and begins a new evidence
chain.

## Claim boundary

T0 through T2 are compatibility evidence. T3 is numerical and information-order
evidence. T4 is source competence. T5 is the first stage that can support a
bounded fresh-object or fresh-session claim. T6 is separate downstream evidence;
it cannot be inferred from T5. No stage by itself establishes deployment safety
or overall state of the art.
