# Recursive Prob4D factor-stream consumption

BayesianPhysTwin can consume Prob4D's append-only `ObservationFactorStreamV1`
without importing Prob4D. The consumer independently revalidates the portable
stream manifest, referenced schema-v4 bundle and payload bytes, causal frame
chain, persistent identities, source revision, and complete joint cross-window
gauge order before an update may enter a recursive belief run.

This interface is additive. The frozen one-shot provider-v1 and provider-v2
experiments retain their original behavior and artifact identities.

## Information flow

```text
Prob4D ObservationFactorStreamV1
        |
        |  manifest, member and row-identity validation
        v
ObservationBeliefV1 + PhysicalLinearizationV1
        |
        |  existing claim-bearing grouped update
        v
candidate complete belief
        |
        |  numerical admissibility + frozen regret decision
        v
candidate belief or exact prior-belief object
        |
        v
ClaimBearingProb4DStreamRunV1
```

Every stream interval is consumed at most once. The next physical linearization
must bind the exact complete belief selected by the preceding interval. A
rejected update is recorded as an exact fallback and returns the same baseline
object; BayesianPhysTwin does not reconstruct an approximate baseline from a
zero correction vector.

## Loading a stream

```python
from bayesian_phystwin.prob4d_factor_stream import (
    load_prob4d_observation_factor_stream,
)

stream = load_prob4d_observation_factor_stream(
    "outputs/case-a/prob4d-factor-stream.json"
)
```

Member verification is enabled by default. For every update the loader checks:

- the path-independent update ID and complete stream ID;
- contiguous update indices, frame intervals, and predecessor hashes;
- exact sequence, case, stream, repository, and source revision;
- the referenced bundle-manifest and payload SHA-256 values;
- schema-v4 observation factors and disabled pickle loading;
- preserved `joint-cross-window` gauge covariance and gauge order; and
- ordinary, confined files without symlink traversal.

`verify_member_files=False` is intended only for metadata inspection. A
claim-bearing recursive update still requires the corresponding
`ObservationBeliefV1` row identity to match the stream update exactly.

## Starting and extending a run

```python
from bayesian_phystwin.prob4d_factor_stream import (
    RecursiveNuisancePolicyV1,
    apply_claim_bearing_prob4d_stream_update,
    start_claim_bearing_prob4d_stream_run,
)

nuisance_policy = RecursiveNuisancePolicyV1(
    mode="persistent_explicit_state",
    state_domain_id=common_complete_belief_domain_id,
    nuisance_family_ids=(
        "prob4d-window-gauge",
        "shared-camera-bias",
        "material-identity",
    ),
)
run = start_claim_bearing_prob4d_stream_run(
    stream,
    initial_belief,
    nuisance_policy=nuisance_policy,
    metadata={"protocol": "prob4d-bpt-recursive-prefix-v1"},
)

selected_belief, run, step = apply_claim_bearing_prob4d_stream_update(
    stream,
    run,
    baseline=initial_belief,
    candidate=candidate_belief,
    observation=observation_belief,
    linearization=physical_linearization,
    claim_update=claim_bearing_update,
    decision=complete_belief_guard_decision,
    nuisance_policy=nuisance_policy,
)
```

The run begins with a content-addressed recursive nuisance policy. Every
physical linearization must bind that exact policy ID, and the policy state
domain must match the complete-belief guard domain. The first accepted or
rejected step also locks the provider manifest, calibration artifact set,
independently verified runtime-revision source, and posterior covariance
interpretation policy. Later steps fail closed if any locked identity changes.

The persisted step binds:

- stream, update, observation-binding, and source identities;
- prior, candidate, selected, and physical-linearization belief identities;
- the one-shot claim-bearing update and complete-belief guard decision;
- exact accept/fallback routing and predecessor step;
- provider, calibration, and runtime evidence; and
- explicit posterior covariance semantics.

Runs can be atomically persisted and revalidated:

```python
from bayesian_phystwin.prob4d_factor_stream import (
    load_claim_bearing_prob4d_stream_run,
    write_claim_bearing_prob4d_stream_run,
)

write_claim_bearing_prob4d_stream_run(run, "outputs/case-a/bpt-stream-run.json")
restored = load_claim_bearing_prob4d_stream_run(
    "outputs/case-a/bpt-stream-run.json"
)
```

Writes are non-overwriting by default.

## Posterior covariance semantics

`GaugeAwareBeliefResult.posterior_covariance` is currently a working
Gauss--Newton/IRLS covariance. It is not silently relabeled as an exact mixture
posterior or a calibrated predictive interval.

`PosteriorCovarianceSemanticsV1` records that distinction in a content-addressed
artifact. The contract supports the following method labels:

- `irls_working`: current working solver covariance;
- `laplace_observed_information`: reserved for a separately implemented exact
  local mixture-curvature calculation; and
- `group_sandwich`: reserved for a separately implemented independent-group
  score correction.

The method label is checked against its curvature and group-score flags. A
calibrated label additionally requires an independent calibration-artifact ID.
The default recursive path creates an uncalibrated `irls_working` record and
locks its dimension-independent policy ID across all updates.

## Persistent nuisance boundary

A valid stream proves causal ordering and byte/identity integrity. It does not
make different updates conditionally independent. `RecursiveNuisancePolicyV1`
therefore permits only two explicit modes: persistent nuisance variables carried
in the complete belief, or conditionally independent increments backed by a
content-addressed evidence artifact. Every physical linearization binds the
chosen policy ID. Repeatedly marginalizing the same nuisance as though it were new
without one of these policies is rejected before complete-belief routing.

## Claim boundary

This implementation provides recursive orchestration, provenance, explicit
uncertainty semantics, and exact fallback. It does not establish:

- real MotionCrafter or Prob4D provider competence;
- calibrated deployment uncertainty;
- physical-state identifiability beyond the supplied linearization and query;
- improvement on a fresh physical object or acquisition session; or
- Causal4D intervention benefit.

Those claims still require a frozen object/session-level calibration and target
protocol with retained technical failures and no target-informed retuning.
