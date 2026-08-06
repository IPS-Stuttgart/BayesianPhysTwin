# Observation timing as an explicit nuisance

A clock offset between camera/provider observations and the actuator or physical
simulator creates a coherent residual

\[
r_i \approx H_{x,i}\Delta x + H_{b,i}b + \dot h_i\,\delta t + \epsilon_i.
\]

`bayesian_phystwin.observation_timing_nuisance` builds the `dh/dt` columns,
keeps clock-domain priors explicit, diagnoses timing/state/bias confounding, and
supports a source-only Gaussian synchronization calibration.

## Producer-consumer sign convention

Portable timing-prior payloads use the fixed convention

```text
aligned_observation_time_s = observation_time_s + offset_s
```

A negative offset therefore shifts an observation timestamp earlier. The
interchange boundary validates this text before admitting the scalar mean, so a
producer's fitted offset cannot be silently reversed.

```python
from bayesian_phystwin.observation_timing_interchange import (
    observation_timing_prior_from_payload,
)

prior = observation_timing_prior_from_payload(causal4d_timing_prior_payload)
```

The payload is closed-schema and contains the clock domain, signed mean,
predictive standard deviation, exact source-artifact identity, and convention.
`observation_timing_prior_payload(prior)` exports the same fixed convention for
a subsequent portable boundary.

## Nuisance design and identifiability

```python
import numpy as np

from bayesian_phystwin.observation_timing_nuisance import (
    append_timing_nuisance,
    assess_timing_identifiability,
    build_timing_jacobian,
)

# One scalar derivative for every row of the observation Jacobian.
timing = build_timing_jacobian(dh_dt_per_observation_row)
result = assess_timing_identifiability(
    timing,
    competing_design=np.column_stack([state_jacobian, spatial_bias_jacobian]),
    independent_timing_jacobian=sync_pulse_jacobian,
)
if not result.identifiable:
    raise ValueError(result.reason)  # retain exact physical fallback

joint_nuisance_jacobian = append_timing_nuisance(
    spatial_bias_jacobian,
    timing,
)
```

The source-only timing posterior should be exported with exact clock-domain,
source-artifact, timestamp-lineage, and revision identities. When the timing
latent is retained explicitly, do **not** add the same timing-induced covariance
to every local point covariance; doing both double counts uncertainty.

Hardware clock offset and physical relaxation lag are distinct latent causes.
If the declared observation design cannot distinguish timing from state,
spatial bias, gauge, or material lag, the claim-bearing update must fail closed.
