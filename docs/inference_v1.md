# Guarded inference API v1

`bayesian_phystwin.inference.v1` is the supported inference-facing namespace for
new integrations. It composes existing strict contracts; it does not introduce a
new estimator, guard, covariance interpretation, or scientific claim.

## Choose the supported namespace

Use the smallest versioned namespace that owns the operation:

| Task | Supported namespace |
| --- | --- |
| Load and write portable observation, query, run, decision, and claim artifacts | `bayesian_phystwin.v1` |
| Construct a strict Prob4D candidate and finalize a guarded complete-belief decision | `bayesian_phystwin.inference.v1` |
| Maintain a historical pre-versioned integration | Package-root compatibility shim or the owning module in the migration map |

New code should not discover inference helpers through the historical package
root. The root remains a lazy compatibility surface for existing integrations.

## Why this namespace exists

The broad package root is a compatibility surface, while
`bayesian_phystwin.v1` intentionally concentrates on portable artifacts. External
inference consumers previously had to discover several research-oriented modules
to perform the supported sequence:

```text
strict observation and linearization admission
    -> covariance-typed candidate inference
    -> independently frozen nonlinear-closure and regret guard
    -> complete candidate selection or exact baseline fallback
```

The inference namespace provides one stable import location for the first and
last steps while preserving the explicit guard boundary between them.

## Candidate inference

Use `infer_prob4d_candidate` for a claim-bearing Prob4D observation. It delegates
to the existing strict provider-v2 admission and prior-aware inference path and
returns `ClaimBearingProb4DCandidateV1`. The candidate contains typed raw
posterior-covariance semantics but is not a deployment decision.

```python
from bayesian_phystwin.inference.v1 import infer_prob4d_candidate

inference = infer_prob4d_candidate(
    observation,
    linearization,
    physical_prediction_xyz_m=physical_prediction,
    config=frozen_solver_config,
)
```

`config` must be `None` or an actual `PriorAwareGaugeConfigV1` instance.
`covariance_semantics` must likewise be `None` or an actual
`PosteriorCovarianceSemanticsV1`. Falsey objects such as `0` are rejected rather
than interpreted as omitted configuration.

## Complete-belief selection

The caller remains responsible for converting the candidate result into one
complete belief and evaluating the separately frozen nonlinear-closure and
regret guard. Finalize that decision with `finalize_guarded_update`:

```python
from bayesian_phystwin.inference.v1 import finalize_guarded_update

result = finalize_guarded_update(
    inference,
    baseline_belief,
    candidate_belief,
    guard_decision,
    metadata={"protocol_id": protocol_id},
)
```

The finalizer independently checks that:

- the candidate inference and guard agree on numerical admissibility;
- the guard binds the exact baseline and candidate artifact identities;
- the selection binds the exact guard decision;
- accepted routing reuses the exact candidate object; and
- rejected routing reuses the exact baseline object.

`GuardedUpdateResultV1.artifact_id` binds the candidate-inference identity, guard
decision, complete-belief selection, selected artifact, exact-fallback flag, and
immutable metadata. `to_record()` produces the portable identifier-only record;
it does not serialize arbitrary application-owned belief payloads.

## Keep point-mean and covariance decisions separate

A valid candidate may contain raw posterior-covariance semantics, but that does
not establish calibration or authorize a mean update. A registered experiment
should state which complete candidate belief is being guarded:

- a point-mean and uncertainty update;
- a covariance-only candidate that preserves the exact registered mean object; or
- the unchanged physical baseline.

`finalize_guarded_update` routes between the two complete beliefs supplied by the
caller. It does not silently construct a covariance-only candidate, relabel raw
covariance as calibrated, or allow a covariance result to imply improved point
prediction.

## Integration checklist

Before protected target outcomes are opened:

1. validate the versioned observation and physical-linearization artifacts;
2. freeze solver configuration and covariance semantics;
3. construct the complete candidate belief without interpreting it as accepted;
4. evaluate the separately registered closure and regret guard;
5. finalize the complete-belief route and retain its identifier-only record;
6. verify that rejection returned the exact baseline object; and
7. keep provider competence, physical-query benefit, and downstream Causal4D
   benefit as separate decisions.

## Executable accepted and fallback example

Run the bundled deterministic demonstration:

```bash
python examples/guarded_inference_v1.py
```

It emits one JSON object with accepted and rejected records. The accepted record
shows that the selected belief is the exact candidate object. The rejected record
shows `exact_fallback=true`, reason `regret-guard-rejected`, and that the selected
belief is the exact baseline Python object rather than a reconstructed numerical
copy. Repository tests execute this example and lock those identity guarantees.

## Separate mean and covariance admission

When point-mean and covariance evidence have separate frozen decisions, use the
explicit `bayesian_phystwin.inference.components_v1` module. It composes the
existing point regret guard and query covariance decision into a five-arm policy:
physical fallback, deterministic reference, mean-only, covariance-only, or full
belief. A positive covariance result cannot rescue a rejected mean, and a
positive mean cannot silently authorize covariance.

The default policy permits covariance-only routing and retains the deterministic
reference when only the mean passes. Common-domain or reference-support failure
returns the exact physical fallback object. See
[`component_admission_v1.md`](component_admission_v1.md) for the complete decision
matrix, artifact bindings, and routing example.

This explicit submodule is not re-exported by `bayesian_phystwin.inference.v1`;
the existing exact 12-symbol namespace remains unchanged.

## Compatibility contract

The exact ordered export surface is recorded in
`api/inference-public-api-v1.json` and validated by
`tools/quality/check_public_api.py`:

```bash
python tools/quality/check_public_api.py \
  --manifest api/inference-public-api-v1.json
```

The namespace is included in wheel/sdist installation and the external strict
MyPy consumer. Importing it requires only the base NumPy dependency; it must not
load Deform360, PhysTwin/Warp, vision, graph, or experiment-only modules.

## Scientific boundary

A valid candidate, guard record, exact fallback, content address, or green API
check is implementation and provenance evidence only. It does not establish
provider competence, calibrated covariance, unseen-object transfer, deployment
safety, Causal4D benefit, or state of the art. Candidate construction and the
guard must be frozen from source/calibration evidence before protected target
outcomes are opened.
