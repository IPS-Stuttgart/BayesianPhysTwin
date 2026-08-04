# Prospective belief updates v1

This document defines two additive development interfaces. Neither interface
changes the frozen fixed-anchor provider, released PhysTwin reproduction, or any
existing empirical claim.

## Evidence-weighted endpoint uncertainty

`bayesian_phystwin.endpoint_model_average` keeps the fixed robust random-walk
endpoint semantics but replaces winner-take-all process/observation-noise
selection with a finite Bayesian model average. Each component is a complete
`FixedBayesianAnchorConfigV1`. For each tracked identity the filter accumulates
causal-prefix predictive log evidence and obtains

```text
component weight ∝ frozen component prior × prefix predictive evidence.
```

The returned covariance applies the law of total covariance:

```text
within-component covariance + between-component mean disagreement.
```

Future covariance is propagated without future observations by adding each
component's random-walk process variance before recomputing the mixture. The
mean remains the robust causal endpoint. A single-component configuration is
numerically equivalent to the fixed provider-v1 endpoint.

```python
from bayesian_phystwin import (
    ModelAveragedEndpointConfigV1,
    infer_model_averaged_endpoint,
    predict_model_averaged_endpoint,
)

posterior = infer_model_averaged_endpoint(
    residual_m,
    valid,
    end_frame=train_end,
    config=ModelAveragedEndpointConfigV1(),
)
prediction = predict_model_averaged_endpoint(
    posterior,
    horizon_steps=20,
)
```

The raw covariance is model-based predictive uncertainty. It is not a
frequentist calibration claim. Any coverage statement still requires frozen
components and priors, independent object/session calibration groups, retained
technical failures, and one prospective target evaluation. Grouped conformal
bounds remain a separate layer.

Causal4D consumers may use the additive
`bayesian_phystwin.causal4d_belief_provider_v2` module. Provider v1 remains
unchanged for frozen experiments and exact historical semantics.

## Strict Prob4D update composition

`update_claim_bearing_prob4d_from_artifacts` is the supported one-call
composition for a new prospective Prob4D-to-Bayesian-PhysTwin experiment. It:

1. validates the strict provider-v2 attestation and stream-v2 joint gauge
   covariance before forming an innovation;
2. verifies the `ObservationBeliefV1` and `PhysicalLinearizationV1` row and
   content identities;
3. preserves full vector-valued state, query, gauge, shared-bias, view-bias,
   anchor, covariance, and correlation-group semantics;
4. invokes the prior-aware grouped nominal/outlier mixture solver; and
5. returns a typed record binding the observation, physical linearization,
   provider manifest, covariance calibration artifacts, and independently
   verified runtime revision.

```python
from bayesian_phystwin import update_claim_bearing_prob4d_from_artifacts

update = update_claim_bearing_prob4d_from_artifacts(
    observation_belief,
    physical_linearization,
    physical_prediction_xyz_m=physical_prediction,
)
```

This path deliberately does not consume the older trace-reduced exploratory
Prob4D prefix interface. It also does not perform candidate deployment by
itself. The resulting complete candidate belief must still pass nonlinear
closure, the source-frozen baseline-relative regret guard, and complete-belief
selection. Rejection must return the exact baseline belief object.

## Gap-aware reliability

The historical `MarkovReliabilityConfig` uses `time_values` only for ordering.
That remains the default under `time_delta_mode="order-only"`. New irregularly
sampled streams may opt into `time_delta_mode="integer-steps"`; each transition
is then raised to the positive integer number of elapsed `time_step` intervals.
Unit-spaced inputs remain exactly equivalent to the historical behavior.

Sequence identities are validated as nonempty strings or genuine integers
without lossy coercion. Integer identities retain the historical string keys in
`sequence_log_evidence`, but a mixed pair such as `1` and `"1"` now fails closed
instead of silently merging two independent tracks. Prior reliability must lie
in `[0, 1]`, and returned posterior arrays and sequence-evidence mappings are
defensively owned and immutable.

## Fail-closed drift-bias evidence

The random-walk nuisance-bias filter follows the same typed sequence-identity
boundary. A string and an integer that serialize to the same evidence key are
rejected instead of sharing one bias trajectory and evidence accumulator.
Valid integer identities retain their historical string evidence keys.

Prior reliability and optional bias probabilities must lie in `[0, 1]` rather
than being repaired by clipping. Numeric timestamps must be finite, empty
particle or measurement batches are rejected, and non-finite likelihood or
state-update numerics raise an explicit error. Returned bias trajectories,
variances, inlier responsibilities, and scalar or batched evidence are
defensively owned and immutable. These checks do not change the random-walk or
robust-mixture equations for valid inputs.

## Validation boundary

The `Prospective belief validation` workflow exercises these interfaces on a
GitHub-hosted CPU runner by default, records its environment identity, runs
focused numerical and contract tests, and publishes a compact stress-test
artifact. Manual runs can select the registered `workstation2` self-hosted
runner when an independent host comparison is useful. Passing that workflow is
implementation evidence only. It does not establish target accuracy,
calibration, or Causal4D intervention benefit.
