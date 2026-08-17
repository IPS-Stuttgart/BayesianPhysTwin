# Horizon-conditioned discrepancy prediction

The historical model-averaged endpoint predictor holds the inferred discrepancy
mean fixed and adds random-walk variance linearly with forecast horizon. That
behavior remains unchanged for frozen experiments. The additive
`horizon_conditioned_discrepancy` interface supports a separately calibrated
alternative for new prospective studies:

```text
endpoint discrepancy posterior
          |
          v
source-calibrated mean retention rho(h)
+ stationary discrepancy floor
+ strictly positive process-growth floor
          |
          v
horizon-conditioned predictive belief
```

## Model

For forecast horizon `h`, the endpoint component means are multiplied by

```text
rho(h) = rho_min + (1 - rho_min) * 2^(-h / half_life).
```

A no-reversion arm is represented explicitly by an infinite half-life and
`rho_min = 1`. The component endpoint covariance is multiplied by `rho(h)^2`.
The prediction then adds:

```text
(1 - rho(h)^2) * stationary_axis_variance
+ h * additional_axis_process_variance
+ h * scaled_component_process_variance.
```

The final covariance uses the law of total covariance, retaining both
within-component uncertainty and between-component disagreement. At horizon
zero, the mean and covariance are numerically identical to the input
`ModelAveragedEndpointPosteriorV1`.

## Source-only calibration

`fit_horizon_discrepancy_calibration` accepts exactly one endpoint vector and one
future-summary vector per independent physical object or acquisition session.
It does not accept frame rows, views, tracks, points, or tactile taxels as extra
calibration groups.

```python
from bayesian_phystwin.horizon_conditioned_discrepancy import (
    fit_horizon_discrepancy_calibration,
)

calibration = fit_horizon_discrepancy_calibration(
    source_group_ids=("object-a", "object-b", "object-c"),
    endpoint_mean_m=endpoint_by_object,
    future_mean_m=future_by_object_and_horizon,
    horizon_steps=(4, 8, 16, 32),
)
```

The compact candidate grid selects a half-life and minimum-retention floor by
equal-group mean prediction error. Axiswise residual second moments are then
split into a nonnegative stationary term and a nonnegative linear process-growth
term. The process-growth term has a strictly positive registered floor so a
source fit cannot recreate the zero-process-noise undercoverage failure by
accident.

The resulting `HorizonDiscrepancyCalibrationV1` is content addressed, immutable,
and records that only source outcomes were used. Construction fails if interval
calibration, confirmation, or target outcomes are declared as selection inputs.

## Prediction

```python
from bayesian_phystwin.horizon_conditioned_discrepancy import (
    predict_horizon_conditioned_endpoint,
)

prediction = predict_horizon_conditioned_endpoint(
    endpoint_posterior,
    calibration,
    horizon_steps=20,
)
```

The additive Causal4D belief-provider v2 manifest advertises the calibration and
prediction schemas. Provider v1 and the historical
`predict_model_averaged_endpoint` function remain unchanged.

## Deform360 Stage-1 binding

The official-Hub Deform360 calibration execution is already sealed through:

```text
bpt experiment run seal-deform360-calibration
```

A horizon calibration used by that deployed predictor must be selected on an
external or already-open source cohort before the ten registered interval-
calibration objects are scored. Persist its JSON artifact and include its exact
artifact ID and file digest in the `physical_response_and_closure` calibration
artifact supplied to the sealer. The existing execution seal then binds those
bytes, the clean implementation revision, the complete calibration bundle, and
the confirmation-opening token without changing the frozen Stage-1 contract.

Do not select the half-life, retention floor, process-growth scale, guard, or
endpoint set from the same ten objects used for ordinary split-conformal
interval calibration. A different information order would require a separately
versioned CV+ or jackknife+ protocol.

## Claim boundary

This interface models horizon structure and preserves its source-selection
lineage. It does not by itself establish calibrated predictive intervals,
physical-state validity, transfer to new objects, safe deployment, or lower
error than a last-residual fallback. A claim-bearing experiment still requires:

- source-only policy selection;
- independent object/session interval calibration;
- a frozen baseline-relative regret guard;
- exact fallback for every rejection; and
- one prospective confirmation opening.
