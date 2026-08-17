# Nuisance-aware observability diagnostics

`bayesian_phystwin.observability_diagnostics` measures whether an additional
observation family contributes information to the **physical query** after all
declared nuisance variables have been marginalized. It is intended for
calibration-only mechanism checks such as comparing a visual explicit-gauge
belief with the otherwise identical visual-plus-contact belief.

The diagnostic does not select a deployable method and does not use target
outcomes. A positive information gain is necessary for a sensor family to break
an ambiguity, but it is not evidence of prediction accuracy, calibration,
transfer, or downstream Causal4D benefit.

## Marginal information

For a joint precision over physical coefficients `x` and nuisance coefficients
`n`,

```text
I = [[I_xx, I_xn],
     [I_nx, I_nn]],
```

the physical-state precision after nuisance marginalization is the Schur
complement

```text
I_x|n = I_xx - I_xn I_nn^-1 I_nx.
```

`NuisanceAwareInformationState.marginal_state_precision()` already implements
this boundary. The observability diagnostic converts that precision to a state
covariance, projects it through an optional full-row-rank physical-query
Jacobian `Q`, and obtains

```text
Sigma_q = Q I_x|n^-1 Q.T,
I_q = Sigma_q^-1.
```

Using a query Jacobian prevents well-observed but irrelevant state directions
from hiding an ambiguity in the actual endpoint, readout, contact, or physical
response being evaluated.

## Example

```python
import json
import numpy as np

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
)
from bayesian_phystwin.observability_diagnostics import (
    compare_marginal_observability,
)

prior = NuisanceAwareInformationState.from_independent_priors(
    state_precision=np.eye(2),
    nuisance_precision=np.eye(1),
)

# Camera displacement is confounded with one shared camera-bias coefficient.
visual = prior.add_observation(
    state_jacobian=np.array([[1.0, 0.0]]),
    nuisance_jacobian=np.array([[1.0]]),
    observation_covariance=np.array([[0.1]]),
)

# The independent contact factor constrains the same physical direction but
# contains no camera-gauge or camera-bias coefficient.
visuotactile = visual.add_observation(
    state_jacobian=np.array([[1.0, 0.0]]),
    nuisance_jacobian=np.array([[0.0]]),
    observation_covariance=np.array([[0.1]]),
)

report = compare_marginal_observability(
    visual,
    visuotactile,
    query_jacobian=np.array([[1.0, 0.0]]),
)
print(json.dumps(report.to_record(), indent=2, sort_keys=True))
```

The comparison reports:

- the complete reference and candidate query spectra;
- numerical and entropy effective rank;
- minimum and maximum precision eigenvalues;
- condition number and weakest-direction variance;
- log-determinant and Gaussian mutual-information gain;
- positive-semidefinite information-increment eigenvalues;
- trace-precision gain;
- weakest-direction precision ratio; and
- mean and maximum marginal-variance reduction.

All arrays are defensively copied and read-only. `to_record()` returns only
JSON-compatible values for inclusion in a content-addressed calibration bundle.

## Fail-closed comparison

The comparison rejects a candidate when, beyond the declared numerical
tolerance, it has:

- lower nuisance-marginalized query information;
- lower log-determinant or trace precision;
- larger marginal query variance; or
- lower weakest-direction precision.

Such a failure usually means the two states do not share the same prior,
nuisance domain, query definition, or evidence order. It must not be described
as negative information supplied by a sensor.

## Deform360 use

For the official-Hub Deform360 calibration objects, produce the report for each
object and for each frozen physical query using the same state prior and visual
rows in both arms:

```text
reference = visual explicit-gauge belief
candidate = same visual belief + independent contact anchor
```

The primary mechanism check should use the exact physical-query Jacobian rather
than the full simulator state. Aggregate only at the independent-object level.
Taxels, frames, views, tracks, and repeated timestamps remain within-object
observations and do not increase the calibration-group count.

The diagnostic should be archived even when it shows negligible contact gain.
That outcome localizes a later negative result to contact informativeness or its
bias model without opening or retuning on the confirmation cohort.
