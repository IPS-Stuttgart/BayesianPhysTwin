# Causal4D fixed-anchor belief provider v1

`bayesian_phystwin.causal4d_belief_provider_v1` is the NumPy-only boundary for
Causal4D's robust endpoint discrepancy inference. It replaces downstream imports
from Bayesian-PhysTwin experiment modules without changing the historical
random-walk mixture semantics.

## Scope

The provider exposes:

- `FixedBayesianAnchorConfigV1`, an immutable fully specified configuration;
- `RobustEndpointPosteriorV1`, an immutable per-track endpoint posterior;
- `infer_fixed_bayesian_anchor_endpoint`, which reads only frames strictly
  before the exclusive `end_frame` cutoff; and
- `causal4d_belief_provider_manifest`, which declares the versioned capability
  and artifact schemas.

The default configuration is the fixed additional-cohort anchor:

| Quantity | Value |
| --- | ---: |
| Process standard deviation | 0.005 m |
| Observation standard deviation | 0.001 m |
| Initial standard deviation | 0.010 m |
| Nominal-component prior | 0.95 |
| Outlier variance multiplier | 100 |

Callers may supply another immutable configuration when reproducing an existing
recorded diagnostic. New confirmatory protocols must freeze that configuration
before target outcomes are opened.

## Example

```python
from bayesian_phystwin.causal4d_belief_provider_v1 import (
    infer_fixed_bayesian_anchor_endpoint,
)

posterior = infer_fixed_bayesian_anchor_endpoint(
    residual_m,
    valid,
    end_frame=causal_frame_stop,
)
print(posterior.mean_m, posterior.variance_m2)
```

The residual must have shape `(T, N, 3)`, the validity mask must have shape
`(T, N)`, and every residual value must be finite. The provider rejects malformed
or future-inconsistent input before invoking the historical inference kernel.

## Posterior semantics

The endpoint posterior contains:

- a 3-D discrepancy mean per tracked material point;
- one isotropic scalar variance per point;
- the final nominal-mixture responsibility; and
- the number of valid recursive updates.

Every returned array is a defensive, C-contiguous, read-only copy. The historical
`mean`, `variance`, and `final_inlier_probability` names remain read-only
properties for compatibility; new consumers should use the unit-bearing v1
field names.

This posterior is a readout/model-discrepancy belief. It is not automatically a
physically admissible position/velocity state correction and does not bypass
Bayesian-PhysTwin's nonlinear-closure or prospective-regret guards.

## Causal4D migration

Causal4D should construct `FixedBayesianAnchorConfigV1` from its already recorded
belief-export settings, call `infer_fixed_bayesian_anchor_endpoint`, and consume
`mean_m`, `variance_m2`, `final_nominal_probability`, and `update_count`. It
should validate the provider manifest's capability and artifact-schema versions
at runtime and bind that manifest into the exported belief provenance.

Frozen historical runs may retain an exact Bayesian-PhysTwin revision and their
original direct-import stack. New development must use this provider boundary so
experiment-module names and file layout are not part of the downstream API.

## Compatibility policy

The provider depends only on NumPy at import and execution time. It does not
load Torch, Warp, OpenCV, or SciPy. Its manifest is validated separately from
the replay provider and graph-provider manifests so a frozen experiment can
identify the exact endpoint-inference contract independently of the simulator
backend.

Provider v1 preserves the numerical semantics of
`phystwin_bayesian_anchor.robust_random_walk_endpoint` for valid inputs. Invalid
shapes, non-finite residuals, invalid configuration values, and inconsistent
cutoffs fail closed. Causal4D should consume this module rather than importing
`phystwin_additional_bayesian_confirmation` or `phystwin_bayesian_anchor`
directly.
