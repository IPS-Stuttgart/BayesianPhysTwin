# Same-mean predictive query mixture v1

## Purpose

`bayesian_phystwin.predictive_query_mixture` adds a prospective uncertainty
candidate for experiments in which the point prediction must remain fixed.
It preserves the caller-owned query mean by Python object identity and attaches
a two-component Gaussian scale mixture:

\[
p(q\mid\mathcal D)
=
\pi\,\mathcal N(q;\mu,P_{\mathrm{nom}})
+
(1-\pi)\,\mathcal N(q;\mu,P_{\mathrm{tail}}),
\]

with

\[
P_{\mathrm{tail}}\succeq P_{\mathrm{nom}}.
\]

The two components have the exact same mean. Consequently, changing the mixture
cannot change Chamfer distance, track error, or any other metric that depends
only on the point prediction. Its value must be established through a proper
predictive score and a separately reported coverage--sharpness trade-off.

This method is motivated by the retrospective full-22 covariance-only result:
a Gaussian covariance improved negative log score and marginal coverage while
increasing mean full interval width by `3.10x`. A broad low-probability tail can,
in principle, represent rare large errors without widening the nominal component
for every prediction. The implementation is prospective method infrastructure;
it does not establish that the mixture improves that trade-off.

## Exact-mean construction

```python
import numpy as np

from bayesian_phystwin.predictive_query_mixture import (
    compose_same_mean_gaussian_mixture,
)

reference_mean_m = np.ascontiguousarray(reference_mean_m, dtype=np.float64)
mixture = compose_same_mean_gaussian_mixture(
    reference_mean_m,
    nominal_covariance_m2,
    tail_covariance_m2,
    reference_predictor_id="last-residual-policy-v1",
    nominal_covariance_id="structured-query-covariance-v1",
    tail_covariance_id="source-frozen-tail-v1",
    nominal_probability=0.90,
)

assert mixture.mean_m is reference_mean_m
assert mixture.record.point_prediction_changed is False
```

The composition contract requires:

- a finite, C-contiguous `float64` NumPy mean so object identity can be retained;
- finite, symmetric, positive-definite component covariances;
- a tail covariance that dominates the nominal covariance in positive-semidefinite
  order;
- nominal probabilities strictly inside `(0, 1)`;
- an explicit density floor when a positive-definite observation model is needed;
- immutable covariance and probability arrays; and
- a content-addressed record binding the exact arrays and information lineage.

No jitter, eigenvalue clipping, pseudoinverse, or probability clipping is added
implicitly. A caller with a positive-semidefinite model covariance must freeze an
observation variance or other density floor before scoring and pass it through
`density_floor_variance_m2`.

## Source-frozen scalar candidates

`SameMeanGaussianMixtureCandidateV1` supplies a deliberately low-dimensional
candidate family:

\[
P_{\mathrm{tail}}
=
\kappa P_{\mathrm{nom}}+\sigma_{\mathrm{tail}}^2I,
\qquad
\kappa\geq1.
\]

```python
from bayesian_phystwin.predictive_query_mixture import (
    SameMeanGaussianMixtureCandidateV1,
    compose_candidate_same_mean_gaussian_mixture,
)

candidate = SameMeanGaussianMixtureCandidateV1(
    nominal_probability=0.90,
    tail_covariance_scale=9.0,
    tail_isotropic_variance_m2=1.0e-6,
)

prediction = compose_candidate_same_mean_gaussian_mixture(
    reference_mean_m,
    nominal_covariance_m2,
    candidate,
    reference_predictor_id=reference_policy_id,
    nominal_covariance_id=nominal_covariance_artifact_id,
)
```

A candidate with `tail_covariance_scale=1` and zero tail nugget is exactly the
single-Gaussian reference, independent of its nominal probability. This gives
source selection an exact statistical fallback rather than requiring a separate
implementation path.

## Group-balanced source selection

`select_same_mean_gaussian_mixture` evaluates a finite candidate grid on complete
source/development physical objects or acquisition sessions. Each group
contributes:

- one mean negative log score; and
- one root-mean-square marginal standard deviation computed from the mixture's
  moment covariance.

The reference candidate is always eligible. A non-reference candidate is
eligible only if:

1. its worst development-group log-score regret relative to the Gaussian
   reference is no larger than the frozen threshold; and
2. its moment-width ratio is no larger than the frozen limit in every group.

Eligible candidates are ordered by mean group log score, worst group log score,
median group log score, mean width, and finally content identity. Candidate and
group order are canonicalized in the resulting artifact.

```python
from bayesian_phystwin.predictive_query_mixture import (
    SameMeanGaussianMixtureCandidateV1,
    select_same_mean_gaussian_mixture,
)

reference = SameMeanGaussianMixtureCandidateV1(
    nominal_probability=0.5,
    tail_covariance_scale=1.0,
)
candidates = [
    reference,
    SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.95,
        tail_covariance_scale=4.0,
    ),
    SameMeanGaussianMixtureCandidateV1(
        nominal_probability=0.90,
        tail_covariance_scale=9.0,
    ),
]

selection = select_same_mean_gaussian_mixture(
    development_group_ids=source_object_ids,
    residual_groups=source_query_residuals,
    nominal_covariance_groups=source_query_covariances,
    candidates=candidates,
    predictor_id=point_predictor_id,
    query_set_id=query_set_id,
    grouping_rule_id=grouping_rule_id,
    development_evidence_id=source_manifest_id,
    reference_candidate_id=reference.candidate_id,
    maximum_worst_group_regret=0.0,
    maximum_width_ratio=2.0,
    grid_frozen_before_development_scores=True,
    target_outcomes_used=False,
)
```

This is development selection, not calibration or confirmation. The selected
candidate must be frozen into the complete predictor identity before a disjoint
calibration cohort is scored.

## Proper scoring

The module scores the actual mixture density rather than replacing it with a
moment-matched Gaussian:

```python
from bayesian_phystwin.predictive_query_mixture import (
    gaussian_mixture_negative_log_density,
    group_gaussian_mixture_negative_log_score,
)

endpoint_nll = gaussian_mixture_negative_log_density(residual_m, prediction)
group_nll = group_gaussian_mixture_negative_log_score(residual_m, prediction)
```

For a non-logarithmic proper-score diagnostic,
`group_gaussian_mixture_energy_score` accepts two caller-owned standard-normal
and component-uniform draw banks. The caller must generate and freeze those banks
without using scored outcomes. Reusing identical draws across candidates gives a
deterministic paired Monte Carlo comparison and avoids hidden random-number
state inside the estimator.

The moment covariance is available only as a sharpness diagnostic:

```python
from bayesian_phystwin.predictive_query_mixture import (
    gaussian_mixture_moment_covariance,
)

moment_covariance = gaussian_mixture_moment_covariance(prediction)
```

It must not be used to claim that the predictive density was Gaussian.

## Density-level calibration

`bayesian_phystwin.query_density_calibration` calibrates a density superlevel set
without collapsing the mixture. For endpoint `j` in independent calibration
group `g`, define

\[
s_{g,j}=-\log p(q_{g,j}\mid\mathcal D_g),
\qquad
s_g=\max_j s_{g,j}.
\]

For `n` independent calibration groups and nominal coverage `c`, the finite rank
is

\[
k=\lceil(n+1)c\rceil.
\]

The fit fails before retaining a calibration artifact when `k>n`. Otherwise the
calibrated region for a future endpoint is

\[
\mathcal C=\{q:-\log p(q\mid\mathcal D)\leq s_{(k)}\}.
\]

Taking the maximum within each group targets simultaneous coverage of every
registered endpoint in one future physical object/session. Frames, coordinates,
tracks, cameras, and points do not increase the calibration sample size.

```python
from bayesian_phystwin.query_density_calibration import (
    fit_query_density_calibration,
    group_density_region_covered,
)

calibration = fit_query_density_calibration(
    calibration_group_ids=calibration_object_ids,
    residual_groups=calibration_query_residuals,
    prediction_groups=calibration_mixture_predictions,
    nominal_coverage=0.90,
    predictor_id=complete_predictor_id,
    query_set_id=query_set_id,
    grouping_rule_id=grouping_rule_id,
    guard_id=guard_id,
    calibration_evidence_id=calibration_manifest_id,
    predictor_frozen_before_scores=True,
    calibration_outcomes_used_for_selection=False,
    calibration_groups_independent=True,
)

covered = group_density_region_covered(
    confirmation_residual_m,
    confirmation_prediction,
    calibration,
    predictor_id=complete_predictor_id,
)
```

The calibration artifact has strict content-addressed JSON save/load helpers.
Publication is atomic and idempotent for the same artifact and refuses symbolic
links or replacement by different content.

## Required future experiment

A claim-bearing same-mean comparison should keep the following data roles
separate:

1. **Development/source groups:** choose the nominal structured covariance and,
   if used, one mixture candidate.
2. **Calibration groups:** fit only the density threshold for the already frozen
   complete predictor.
3. **Confirmation groups:** report proper score, calibrated coverage, sharpness,
   worst-group behavior, exact point identity, and exact fallback.

The minimum locked arms are:

1. deterministic `last_residual`;
2. the same mean with the Gaussian reference covariance;
3. the same mean with the selected heavy-tailed mixture;
4. each accepted uncertainty policy with its disjoint calibration; and
5. the unchanged physical fallback for every unsupported or rejected unit.

Report the mixture log score and a paired energy score together with 50%, 90%,
and 95% coverage, density-region volume or a preregistered Monte Carlo proxy,
moment width, worst-object regret, acceptance, and fallback frequency. A tie
retains the Gaussian reference.

## Scientific boundary

This implementation establishes:

- exact point-mean identity;
- a valid same-mean broad-tail density family;
- deterministic group-balanced source selection with a Gaussian reference;
- exact mixture log scoring and deterministic paired energy scoring;
- finite-group density-level calibration; and
- content-addressed, fail-closed artifacts.

It does **not** establish:

- improved proper score or coverage--sharpness on unseen objects;
- calibration of the raw Gaussian or mixture density;
- a physical-state correction or unique causal explanation;
- real Prob4D provider competence;
- Causal4D counterfactual benefit;
- deployment safety; or
- state of the art.

No source, calibration, confirmation, target, DLO3, held-v8, or Causal4D physical
outcome is opened by this change. Existing frozen predictors, protocols,
calibrations, claims, and exact-fallback behavior remain unchanged.
