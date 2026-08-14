# Cross-fitted structured query covariance V1

`query_covariance_crossfit` selects a low-dimensional covariance correction on
complete source/development objects or acquisition sessions. It is additive to
the full-covariance dynamic endpoint model and to the existing raw covariance
portfolio. It never changes a point prediction and it does not replace
`QueryCalibrationV1`.

## Transform family

For one raw query covariance `P`, a candidate applies

```text
P_base = scale * ((1 - shrinkage) * P + shrinkage * diag(diag(P)))
         + isotropic_variance * I
P_structured = P_base + L
```

where `scale > 0`, `0 <= shrinkage <= 1`, `isotropic_variance >= 0`, and `L` is a
positive-semidefinite low-rank excess covariance. Rank-zero candidates recover
scalar/diagonal/nugget transforms. Retaining `shrinkage < 1` preserves some or
all cross-axis covariance rather than silently diagonalizing the full-covariance
predictor.

For a candidate with positive maximum rank, `L` is fitted only from the training
groups of each leave-one-group-out fold. Every training group contributes equal
weight:

```text
R_g = mean_j e[g,j] e[g,j]^T
B_g = mean_j P_base[g,j]
E   = mean_g (R_g - B_g)
L   = fraction * top_positive_rank(E)
```

Thus a long source sequence cannot dominate merely because it has more frames,
tracks, or points. Tiny negative eigenvalues consistent with floating-point
roundoff are clipped; materially non-PSD inputs fail closed.

## Frozen selection

The hyperparameter grid and the reference candidate must be fixed before any
development residual is scored. Each candidate receives one mean Gaussian NLL
per held-out physical group. Selection minimizes, in order:

1. mean group NLL;
2. worst group NLL;
3. median group NLL; and
4. the content-addressed candidate ID.

A candidate is eligible only when its worst group NLL regret relative to the
registered reference does not exceed `maximum_worst_group_regret`. Setting this
to zero requires no held-out development group to be harmed. The reference is
always eligible, so the procedure has an exact statistical fallback even when
no structured correction passes the guard.

After selection, the low-rank term is refitted on all development groups. The
result binds the predictor, query set, grouping rule, development evidence,
canonical group roster, complete candidate grid, cross-validated group-score
matrix, reference, selected transform, and information-order declarations.
Candidate and group order are canonical, and all arrays are retained in
bytes-backed read-only storage.

## Evaluation diagnostics

`score_query_covariance_group(...)` reports, for one independent confirmation
group:

- mean Gaussian negative log likelihood;
- mean squared Mahalanobis error;
- empirical ellipsoid coverage at a caller-frozen squared radius;
- mean log square-root determinant as a sharpness measure;
- mean effective covariance rank; and
- maximum condition number.

`group_gaussian_energy_score(...)` provides a deterministic paired Monte Carlo
energy score. Its standard-normal sample pairs must be generated and frozen
independently of the scored outcomes and should be bound by the surrounding
evaluation manifest.

## Use with finite-group calibration

Structured selection and finite-group calibration require separate data roles:

1. use source/development groups to select the transform;
2. apply the selected transform to covariances from a disjoint calibration
   cohort;
3. call `fit_query_calibration(...)` with `covariance_scale=1` and
   `isotropic_variance=0` on those already transformed covariances; and
4. evaluate proper score, coverage, width, worst-group regret, and exact fallback
   on a separately retained confirmation or target cohort.

The selected transform must be included in the complete deployed predictor ID.
Do not tune the transform on the calibration cohort that determines the
conformal quantile.

## Example

```python
from bayesian_phystwin.query_covariance_crossfit import (
    StructuredQueryCovarianceCandidateV1,
    apply_structured_query_covariance,
    fit_cross_fitted_query_covariance,
)

raw = StructuredQueryCovarianceCandidateV1()
structured = StructuredQueryCovarianceCandidateV1(
    covariance_scale=1.0,
    diagonal_shrinkage=0.25,
    isotropic_variance=1e-6,
    low_rank_rank=1,
    low_rank_fraction=1.0,
)

selection = fit_cross_fitted_query_covariance(
    development_group_ids=source_object_ids,
    residual_groups=source_residuals,
    covariance_groups=source_covariances,
    candidates=[raw, structured],
    predictor_id=predictor_id,
    query_set_id=query_set_id,
    grouping_rule_id=grouping_rule_id,
    development_evidence_id=source_manifest_id,
    reference_candidate_id=raw.candidate_id,
    maximum_worst_group_regret=0.0,
    hyperparameter_grid_frozen_before_scores=True,
    target_outcomes_used=False,
)

calibration_covariances = [
    apply_structured_query_covariance(group, selection.selected_transform)
    for group in raw_calibration_covariances
]
```

## Claim boundary

This artifact establishes deterministic, group-balanced source-development
selection inside the declared candidate family. It does not establish raw or
post-transform calibration, finite-sample coverage, provider competence,
physical-query improvement on unseen objects, causal identifiability,
intervention benefit, deployment safety, or state of the art. Those claims
require the separate frozen calibration and independent physical confirmation
protocols, with exact physical fallback preserved whenever a prospective guard
rejects the update.
