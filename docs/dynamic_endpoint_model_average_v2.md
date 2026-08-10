# Dynamic endpoint model average v2

`bayesian_phystwin.dynamic_endpoint_model_average` is an additive endpoint
belief for prospective PhysTwin and Causal4D experiments. It leaves the frozen
`endpoint_model_average` v1 implementation and every historical result
unchanged.

The v2 family addresses two limitations of the historical finite random-walk
grid:

1. the strongest simple deterministic comparator, the last observed residual,
   was not part of the Bayesian family; and
2. every component had a constant forecast mean, so persistent discrepancy
   velocity could only appear as widening covariance.

## Component family

A `DynamicEndpointModelAverageConfigV2` contains any finite, source-frozen
combination of three component types.

### Exact persistence

`PersistenceEndpointComponentV2` uses the newest valid residual as its level
mean exactly. Between valid observations it follows sample-and-hold dynamics.
Its robust nominal/outlier predictive density still supplies evidence and
uncertainty, but its point prediction is the exact last-residual comparator.

### Robust local level

A historical `FixedBayesianAnchorConfigV1` remains a valid v2 component. In a
single-component v2 configuration, its filtered mean, variance, nominal
responsibility, update count, and predictive evidence are numerically equivalent
to the historical robust random-walk endpoint.

### Robust damped trend

`DampedTrendEndpointComponentV2` uses the state

```text
[level, velocity]
```

with transition

```text
level[t + 1]    = level[t] + velocity[t] + level_noise
velocity[t + 1] = retention * velocity[t] + velocity_noise.
```

The retention lies in `[0, 1]`. Values below one prevent an estimated short
prefix trend from producing an unbounded constant-velocity extrapolation. Both
level and velocity are updated by the same robust three-dimensional
nominal/outlier observation mixture used by the local-level components.

## Causal evidence weighting

Every component accumulates predictive log evidence from frames strictly before
`end_frame`. The component prior is frozen in the configuration. Unless explicit
probabilities are supplied, the default prior assigns equal mass to persistence,
local-level, and damped-trend families, then divides each family mass over its
grid entries. This prevents adding another grid point from silently increasing a
dynamics family's prior probability. Component-uniform priors remain available
with `balance_component_families=False`.

The default evidence policy computes one weight vector per tracked identity:

```text
weight(track, component)
  proportional to
prior(component) * prefix_predictive_evidence(track, component).
```

`evidence_pooling="object"` is an explicit alternative. It sums evidence over
tracks that received at least one observation and applies one common component
weight vector to the object. This partially pools short or noisy tracks without
using future frames or target outcomes. The pooling choice belongs in the
prospective run manifest.

## Posterior and forecast moments

`DynamicEndpointPosteriorV2` stores the complete two-state component posterior,
not only the moment-matched endpoint. Forecasting can therefore propagate every
component's mean and covariance at the requested horizon before recomputing the
mixture.

For a horizon `h`, each component supplies

```text
component mean[h]
component covariance[h]
```

and the returned covariance uses the law of total covariance:

```text
weighted within-component covariance
+ weighted between-component mean disagreement.
```

The transition and accumulated process covariance are exponentiated by repeated
squaring. Very long horizons therefore require logarithmically many transition
compositions rather than one loop per future step. Component updates use
Joseph-form covariance expressions. Non-finite or materially indefinite
covariance fails closed; the implementation does not add jitter, clip
eigenvalues, or substitute a pseudoinverse.

## Python use

```python
import numpy as np

from bayesian_phystwin.dynamic_endpoint_model_average import (
    DynamicEndpointModelAverageConfigV2,
    infer_dynamic_endpoint_model_average,
    predict_dynamic_endpoint_model_average,
)

posterior = infer_dynamic_endpoint_model_average(
    residual_m,
    valid,
    end_frame=causal_prefix_stop,
    config=DynamicEndpointModelAverageConfigV2(
        evidence_pooling="object",
    ),
)

prediction = predict_dynamic_endpoint_model_average(
    posterior,
    horizon_steps=20,
)
```

The default family contains one exact persistence component, three robust
local-level components, and three damped-trend components.

## Causal4D provider

`bayesian_phystwin.causal4d_belief_provider_v3` exposes the dynamic endpoint and
retains the provider-v2 recursive Prob4D stream symbols unchanged:

```python
from bayesian_phystwin.causal4d_belief_provider_v3 import (
    causal4d_belief_provider_v3_manifest,
    infer_dynamic_bayesian_anchor_endpoint,
    predict_dynamic_endpoint_model_average,
)
```

Provider v1 and provider v2 remain available for frozen experiments. A Causal4D
consumer must bind the provider revision, schema versions, component family,
component priors, evidence-pooling policy, and causal cutoff in its run
provenance.

## Scientific and calibration boundary

This module adds a stronger candidate family; it does not establish that the
new family improves a physical query. Component selection uses only causal
prefix evidence, but the component grid and priors must still be frozen on
source data before confirmation or target evaluation.

The returned covariance is a model-based predictive covariance. It includes
within-component uncertainty and between-component disagreement, but it is not
a frequentist coverage claim. Coverage, interval width, negative log predictive
density, energy score, selective risk, and deterministic error against exact
persistence should be evaluated on independent object or acquisition-session
groups. Rejected downstream physical updates must continue to return the exact
physical baseline.
