# Guarded inference API v1

`bayesian_phystwin.inference.v1` is the supported inference-facing namespace for
new integrations. It composes existing strict contracts; it does not introduce a
new estimator, guard, covariance interpretation, or scientific claim.

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

The new namespace provides one stable import location for the first and last
steps while preserving the explicit guard boundary between them.

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

Run the bundled deterministic demonstration:

```bash
python examples/guarded_inference_v1.py
```

It exercises one accepted candidate and one regret-guard rejection. The rejected
case returns the same baseline Python object, not a reconstructed numerical copy.

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
