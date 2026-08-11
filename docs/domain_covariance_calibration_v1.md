# Calibration-frozen domain covariance calibration v1

## Purpose

The full-22 discrepancy tournament retained `last_residual`: every structured
challenger widened predictive intervals toward nominal coverage, but none
improved the registered point loss or proper score sufficiently to pass the
source-only gate. That result does not justify another outcome-tuned mean model.
It does identify a narrower infrastructure gap: predictive covariance needs a
prospectively frozen, domain-aware calibration layer that can abstain without
changing the point predictor.

`bayesian_phystwin.domain_covariance_calibration` provides that layer. It keeps
the predictive mean fixed and considers only transforms of the form

```text
Sigma_calibrated = scale * Sigma_raw + isotropic_variance * I.
```

Both parameters come from a finite grid frozen before calibration outcomes are
opened. The raw transform `(scale=1, isotropic_variance=0)` is mandatory, so the
selection procedure never has to manufacture a covariance change.

The module is explicit experimental infrastructure. It is not exported from the
stable package root and does not alter an existing predictor automatically.

## Statistical unit and scoring

The fitting function receives event-level residuals and covariances together
with unique event identifiers, independent group identifiers, and declared
domain identifiers. Multiple events may belong to one group, but every group
must belong to exactly one domain.

For each transform, Gaussian negative log likelihood is averaged in two stages:

1. average events within each independent group;
2. average the resulting group scores equally.

Consequently, a long sequence cannot dominate a short physical object or
session merely by contributing more frames or query endpoints. Event records are
sorted by their identifiers before content addressing, so input order does not
change the calibration identity. Validated transform and held-group rows are
normalized to typed tuple records before arithmetic or serialization.

The scoring-only eigenvalue floor is separate from the selected isotropic
variance. It makes Gaussian scoring finite for positive-semidefinite raw
covariances without silently changing the covariance returned to an application.
All inputs must be finite, symmetric, and positive semidefinite under the frozen
tolerance.

## Leave-one-group-out support

A low in-sample calibration score is not sufficient. For every domain, the
implementation repeats the complete transform selection while holding out one
independent group. It then compares the selected transform with the raw
covariance on that held group.

The default support rule requires:

- at least four independent calibration groups;
- nonnegative mean held-group NLL improvement;
- no held group with positive NLL regression; and
- support from the companion calibration-domain guard.

These thresholds and the transform grid are part of the content-addressed
certificate. The full-data transform remains visible even when cross-fitted
support fails, but an unsupported decision cannot be applied.

## Domain-guard binding

Fitting requires a
[`CalibrationDomainGuardCertificateV1`](calibration_domain_guard_v1.md) from the
same calibration partition and statistical unit. For every shared domain, the
independent-group roster must match exactly. This prevents a covariance
calibrator from borrowing a more favorable subset than the point-loss domain
guard.

The domain guard and the covariance calibration answer different questions:

- the domain guard asks whether the candidate continuation is supported for the
  declared point-loss domain;
- covariance calibration asks whether a frozen scale-plus-floor transform
  improves held-group Gaussian scoring without harmful group regressions.

Both must pass before application.

## Information boundary

The certificate records whether:

- the predictor was frozen before calibration outcomes;
- the transform grid was frozen before calibration outcomes;
- application outcomes were used for calibration selection;
- calibration groups are independent; and
- the bound domain-guard certificate is deployment-admissible.

A certificate is application-admissible only when both predictor and grid were
frozen first, no application outcome informed selection, calibration groups are
independent, and the domain guard satisfies its own prospective boundary.
Retrospective calculations remain inspectable but cannot route a transformed
covariance.

## Exact fallback

`apply_domain_covariance_calibration` accepts only a covariance object, a domain
identifier, and an inference-admissibility flag. It has no residual, target
loss, or application-outcome argument.

Unknown domains, unsupported domains, rejected inference, nonprospective
certificates, and nonadmissible domain guards return the exact caller-owned raw
NumPy array object. An admitted domain whose selected transform is exactly the
mandatory raw transform is handled identically: it retains the caller-owned
object and records `calibration-identity-transform-retained` rather than claiming
that a calibration was applied. No copy, zero correction, or reconstructed
covariance is substituted.

Successful nonidentity application returns a new immutable float64 array and a
content-addressed record binding the selected transform and the actual input and
output shapes and digests. Application records validate their internal gate,
decision, transform, reason, and exact-fallback consistency before an artifact
identifier can be accepted.

## Example

```python
from bayesian_phystwin.domain_covariance_calibration import (
    DomainCovarianceCalibrationConfigV1,
    apply_domain_covariance_calibration,
    fit_domain_covariance_calibration,
)

config = DomainCovarianceCalibrationConfigV1(
    covariance_scales=(0.5, 1.0, 2.0, 4.0, 8.0),
    isotropic_variances=(0.0, 1e-8, 1e-6, 1e-4),
)

certificate = fit_domain_covariance_calibration(
    predictor_id=frozen_predictor_id,
    calibration_partition_id=calibration_partition_id,
    statistical_unit="independent-physical-session",
    residual_semantics="prediction-error-m",
    covariance_semantics="raw-predictive-covariance-m2",
    event_ids=calibration_event_ids,
    group_ids=calibration_session_ids,
    domain_ids=calibration_domains,
    residuals=calibration_residuals,
    covariances=calibration_covariances,
    domain_guard=domain_guard_certificate,
    predictor_frozen_before_calibration_outcomes=True,
    transform_grid_frozen_before_calibration_outcomes=True,
    application_outcomes_used_for_calibration_selection=False,
    calibration_groups_independent=True,
    config=config,
)

covariance, application = apply_domain_covariance_calibration(
    raw_covariance,
    certificate,
    domain_id=current_domain,
    inference_admissible=inference_result.accepted,
)
```

Isotropic variances use the same squared physical unit as the supplied
covariances. A grid therefore belongs to a named covariance semantics and must
be frozen for that semantics rather than copied blindly between tasks.

## Scientific boundary

This module calibrates a declared covariance representation under a frozen
finite grid and independent calibration groups. It does not identify simulator
state, improve the predictive mean, establish fresh-object transfer, validate a
provider, prove Causal4D intervention benefit, authorize deployment, or establish
state of the art. A certificate and every application record remain
non-claim-bearing infrastructure until a separately registered independent study
supports the intended use domain.
