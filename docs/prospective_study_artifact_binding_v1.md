# Prospective study artifact binding v1

## Purpose

`bayesian_phystwin.prospective_study_artifact_binding_v1` adds explicit artifact
roles to the content-addressed prospective-study lifecycle. The base lifecycle
stores one opaque SHA-256 identity per transition. This additive contract keeps
the digest of the underlying bytes separate from the identity used by the
lifecycle state.

The separation prevents the same byte-identical object from being silently
reinterpreted as, for example, source predictions, target authorization, target
scores, or a terminal decision.

## Domain-separated identity

`ProspectiveStudyArtifactBindingV1` records:

- the exact protocol ID and protocol content identity;
- the destination lifecycle stage and its single permitted artifact role;
- the SHA-256 digest of the underlying artifact bytes;
- the artifact schema name and positive schema version; and
- recursively immutable finite JSON metadata.

Its `binding_id` is the SHA-256 content identity of a canonical descriptor that
also contains the fixed domain separator
`bayesian-phystwin/prospective-study-artifact-binding/v1`.

Consequently, the same raw bytes bound to two different roles have different
binding identities. `validate_role_bound_prospective_study_chain` additionally
rejects reuse of the same raw content digest anywhere in one lifecycle chain,
even though the domain-separated binding IDs differ.

## Stage-to-role registry

The contract admits exactly these bindings:

| Lifecycle stage | Artifact role |
| --- | --- |
| `source-predictions-sealed` | `source-prediction-bundle` |
| `source-scored` | `source-score-bundle` |
| `target-authorized` | `target-authorization` |
| `target-predictions-sealed` | `target-prediction-bundle` |
| `target-scored` | `target-score-bundle` |
| Any terminal stage | `terminal-decision` |

`design-locked` has no transition artifact and therefore cannot receive an
artifact binding. A caller cannot override the registry through metadata or by
supplying another role.

## Advancing a study

Use `advance_role_bound_prospective_study` for new claim-bearing studies. It
constructs the canonical binding, stores its `binding_id` in the lifecycle
state, and returns both immutable records:

```python
from bayesian_phystwin.prospective_study_artifact_binding_v1 import (
    advance_role_bound_prospective_study,
    validate_role_bound_prospective_study_chain,
)
from bayesian_phystwin.prospective_study_lifecycle_v1 import (
    lock_prospective_study,
)

states = [lock_prospective_study(protocol)]
bindings = []

state, binding = advance_role_bound_prospective_study(
    states[-1],
    next_stage="source-predictions-sealed",
    artifact_content_id=source_prediction_bytes_sha256,
    artifact_schema_name="example.source-prediction-bundle-v1",
    artifact_schema_version=1,
)
states.append(state)
bindings.append(binding)

validate_role_bound_prospective_study_chain(protocol, states, bindings)
```

The validator first recomputes the ordinary lifecycle chain, then requires
exactly one external binding per transition and verifies protocol, stage, role,
binding identity, and raw-content uniqueness.

## Compatibility boundary

The contract does not change `ProspectiveStudyStateV1` or reinterpret existing
frozen evidence. Historical chains remain valid under
`validate_prospective_study_chain`. They pass the stronger role-bound validator
only when the corresponding canonical binding records exist and the lifecycle
states store those binding identities.

Green tests establish deterministic role binding and fail-closed lifecycle
accounting. They do not establish provider competence, calibration, physical
benefit, intervention benefit, deployment safety, or a paper claim.
