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

## Validation boundary

The self-hosted workflow `Prospective belief validation` exercises these
interfaces on the registered `workstation2` runner, records GPU/environment
identity, runs focused numerical and contract tests, and publishes a compact
stress-test artifact. Passing that workflow is implementation evidence only.
It does not establish target accuracy, calibration, or Causal4D intervention
benefit.
