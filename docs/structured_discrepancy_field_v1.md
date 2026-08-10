# Structured discrepancy field v1

`bayesian_phystwin.structured_discrepancy` is an additive development interface
for endpoint discrepancy beliefs whose uncertainty is correlated across tracked
object locations. It does not replace the fixed Bayesian anchor, the existing
model-averaged endpoint, or any frozen provider contract.

## Motivation

The historical endpoint filters estimate every tracked 3-D residual separately.
That is useful and lightweight, but it cannot represent a shared translation,
bending mode, stretch mode, or another coherent object-level discrepancy as one
uncertain latent quantity. It also assigns model-family weights independently to
every track.

The structured interface instead uses a frozen spatial basis and one set of
component weights for the complete object-level field.

## Model

For each Cartesian coordinate, the discrepancy field is

```text
d_t = B a_t + e_t,
```

where:

- `B` has shape `(track_count, rank)` and orthonormal columns;
- `a_t` is a robust random-walk coefficient belief shared across tracks; and
- `e_t` is a diagonal local remainder that preserves omitted marginal variance.

Each endpoint component reuses a complete `FixedBayesianAnchorConfigV1`. The
coefficient prior and process covariance are isotropic in basis coordinates.
For track `i`, the projector leverage is

```text
h_i = sum_r B[i, r] ** 2.
```

The unresolved diagonal remainder receives the complementary fraction `1 - h_i`
of the component's initial and process variance. Consequently, omitted spatial
directions retain marginal uncertainty rather than being reported as exact zero.
For any complete orthonormal basis with binary reliability, a single component is
numerically equivalent to the existing independent endpoint filter after the
corresponding coefficient reparameterization.

## Robust observation update

The interface consumes:

- residuals with shape `(frames, tracks, 3)`;
- an exact Boolean validity mask;
- optional residual-independent prior reliability in `[0, 1]`; and
- an exclusive causal `end_frame`.

For a non-identity basis, every frame uses one order-invariant information-form
update in coefficient space. A per-track nominal/outlier Gaussian mixture
computes the robust responsibility. Prior reliability scales both the update
precision and the component score. Zero-reliability rows have no effect.

The component score has semantics
`mean-cumulative-track-marginal-mixture-log-score-v1`: cumulative marginal
mixture scores are averaged over supported tracks before combination with the
frozen component prior. This deliberately prevents the number of retained
vertices from acting as an unbounded independent evidence multiplier. It is a
versioned generalized score, not a claim of exact marginal likelihood under a
fully correlated mixture model.

## Covariance representation

For component `k`, the within-component covariance for one Cartesian coordinate
is

```text
B P_k B.T + diag(v_local,k).
```

The complete mixture adds between-component disagreement of the 3-D field. The
stored representation therefore contains:

- coefficient means with shape `(components, rank, 3)`;
- coefficient covariances with shape `(components, rank, rank)`;
- local diagonal variances with shape `(components, tracks)`;
- one object-level component-weight vector; and
- derived per-track `3 x 3` marginal covariance.

`structured_discrepancy_query_moments` evaluates an arbitrary linear query
without materializing the complete `(3 * tracks) x (3 * tracks)` covariance. It
includes shared-basis covariance, unresolved local variance, and model-family
disagreement exactly under the stored representation.

## Example

```python
import numpy as np

from bayesian_phystwin.structured_discrepancy import (
    infer_structured_discrepancy,
    predict_structured_discrepancy,
    structured_discrepancy_query_moments,
)

# A normalized common-translation mode for N tracks.
track_count = residual_m.shape[1]
basis = np.ones((track_count, 1)) / np.sqrt(track_count)

posterior = infer_structured_discrepancy(
    residual_m,
    valid,
    basis,
    prior_reliability=prior_reliability,
    end_frame=train_end,
)
prediction = predict_structured_discrepancy(
    posterior,
    horizon_steps=20,
)

# Mean x displacement of all tracks.
query = np.zeros((1, track_count, 3))
query[0, :, 0] = 1.0 / track_count
moments = structured_discrepancy_query_moments(prediction, query)
```

The basis must be selected and frozen without target outcomes. Appropriate
candidates include source-only residual modes, normalized spring-graph modes, or
simulator sensitivity modes. A target-selected basis is not admissible evidence.

## Validation boundary

The contracts fail closed on nonnumeric arrays, nonfinite values, non-Boolean
validity, out-of-range reliability, non-orthonormal bases, malformed covariance,
inconsistent component identities, and tampered component weights. Returned
arrays are defensively owned and read-only.

The current interface establishes numerical structure and exact query algebra.
It does not establish:

- calibrated predictive uncertainty;
- physical-state or material-parameter identification;
- energy- or contact-consistent simulator assimilation;
- improvement over the last-residual comparator;
- independent-object transfer;
- Causal4D intervention benefit; or
- deployment safety.

A claim-bearing successor requires a source-frozen basis, independent
object/session calibration, nonlinear PhysTwin closure, the common
baseline-relative guard, exact complete-belief fallback, and prospective
object/session-level comparison against both physical fallback and last residual.
