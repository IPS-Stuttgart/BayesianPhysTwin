# Calibration-frozen domain covariance calibration v1

## Purpose

`bayesian_phystwin.domain_covariance_calibration` calibrates predictive
covariance separately for declared physical regimes without tuning on
application outcomes. It targets the failure mode in which posterior means are
useful but raw covariance is severely under-dispersed, while preserving the
existing exact-fallback boundary.

The module is experimental and is not exported from the stable package root.

## Conservative transform

For one domain, the fitted covariance is

```text
Sigma_calibrated = scale * Sigma_raw + floor_variance * I.
```

The constraints are deliberately one-sided:

- `scale >= 1`, so calibration never shrinks the supplied covariance;
- `floor_variance >= 0`, so unresolved isotropic discrepancy can be added; and
- the identity transform is always present in the frozen search grid.

The isotropic floor is parameterized relative to a robust reference variance.
For each training group, the median `trace(Sigma_raw) / dimension` is computed;
the median of those group medians is the reference. This prevents a long trial
or a dense frame sequence from dominating the floor scale.

The default scale grid spans `1` through `4096`, covering the large inflation
needed by strongly under-dispersed posteriors. The default floor grid contains
zero plus positive ratios from `1e-6` through `4`. Candidate transforms are
ranked by group-balanced Gaussian negative log likelihood per dimension. A
candidate must improve the score by more than the frozen `1e-3` tolerance to
displace an earlier, simpler grid point; because zero floor is enumerated first,
a practically tied scale-only transform is preferred over an additive floor.

## Independent statistical units

The input is organized as independent calibration groups. Each group may
contain many causally valid residual/covariance samples, but samples within one
group are averaged before groups are averaged. Frames, vertices, pixels, and
tracks therefore do not become artificial independent replicates.

Both group identifiers and sample identifiers are required. The fitter sorts
both levels before fitting and content addressing, making equivalent input
permutations identical. Residual and covariance arrays are converted to
canonical little-endian float64 and bound by shape and SHA-256 digest.

All calibration covariances must be finite, symmetric, and positive definite.
Every domain needs at least two groups to construct a leave-one-group-out
record. The default domain guard still requires at least three groups before a
domain can be deployment-supported.

## Prediction-first domain decision

For each held-out group:

1. fit `scale` and `floor_variance` using all other groups in the same domain;
2. score the held-out group under raw and calibrated covariance;
3. record normalized energy and coordinate-wise nominal-90% coverage as
   diagnostics; and
4. convert the Gaussian NLL difference into a positive geometric loss ratio.

For held-out per-dimension scores `L_calibrated` and `L_raw`, the guard receives

```text
candidate_loss = exp(clip(L_calibrated - L_raw, -50, 50))
fallback_loss = 1.
```

Thus the existing relative-improvement rule is

```text
1 - exp(L_calibrated - L_raw),
```

up to the frozen numerical clip. A positive value means the calibrated
covariance assigned greater geometric predictive density to the held-out group.
The default domain rule requires at least three independent groups, at least 5%
mean improvement, wins on at least two thirds of groups, and no group regression
greater than 5%.

Only after the leave-one-group-out records and support decision are fixed is one
final transform fitted on all calibration groups in the domain.

## Information boundary and exact fallback

The certificate embeds `CalibrationDomainGuardCertificateV1`. Deployment is
admissible only when:

- calibration was frozen before application outcomes;
- application outcomes were not used for selection; and
- calibration groups are independent.

`apply_domain_covariance_calibration` accepts no application residual, loss, or
target outcome. For an unsupported, unknown, or retrospective domain, it returns
the exact input NumPy covariance object. For an authorized domain, it returns a
new bytes-backed immutable covariance array and a content-addressed application
record.

The covariance decision and complete-belief decision can use the same embedded
guard:

```python
from bayesian_phystwin.calibration_domain_guard import (
    select_calibration_domain_guarded_belief,
)
from bayesian_phystwin.domain_covariance_calibration import (
    apply_domain_covariance_calibration,
    fit_domain_covariance_calibration,
)

certificate = fit_domain_covariance_calibration(
    calibration_partition_id=calibration_partition_id,
    statistical_unit="independent-physical-trial",
    residual_definition="prediction-minus-observation-m",
    covariance_definition="raw-predictive-covariance-m2",
    group_ids=calibration_trial_ids,
    domain_ids=calibration_regimes,
    sample_ids=calibration_sample_ids,
    residuals=calibration_residual_batches,
    covariances=calibration_covariance_batches,
    guard_frozen_before_application_outcomes=True,
    application_outcomes_used_for_guard_selection=False,
    calibration_groups_independent=True,
)

calibrated_covariance, covariance_application = (
    apply_domain_covariance_calibration(
        candidate_covariance,
        certificate,
        domain_id=current_regime,
    )
)

# Construct the candidate belief with calibrated_covariance when authorized.
# The same domain guard then selects the complete belief, so rejection retains
# state, parameters, particles, discrepancy, covariance, and provenance together.
selected_belief, routing = select_calibration_domain_guarded_belief(
    physical_baseline_belief,
    candidate_belief,
    certificate.guard_certificate,
    domain_id=current_regime,
    common_domain_id=physical_query_domain_id,
    inference_admissible=(
        inference_result.accepted
        and covariance_application.calibration_applied
    ),
)
```

## Recorded diagnostics

Every cross-fitted fold records:

- the held-out and exact training-group rosters;
- fitted scale, floor, and reference variance;
- raw and calibrated Gaussian NLL per dimension;
- log and geometric loss ratios;
- raw and calibrated normalized energy; and
- raw and calibrated coordinate-wise nominal-90% coverage.

Every final domain transform records group-balanced NLL and normalized energy
before and after calibration. Coverage and normalized energy are diagnostics;
selection and domain support remain based on the proper Gaussian log score.

## Scientific boundary

This is calibration and guard infrastructure, not fresh empirical evidence. A
supported domain does not establish unseen-object transfer, general provider
competence, physical-state identifiability, Causal4D intervention benefit,
deployment safety, or state of the art. The transform is a conservative
scale-plus-isotropic-floor model; it does not claim that all remaining
miscalibration is isotropic or that a low-rank discrepancy term is unnecessary.
A future low-rank extension should be separately versioned and must earn its
complexity through independent group-level proper scores.
