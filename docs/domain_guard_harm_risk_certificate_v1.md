# Simultaneous domain harm-risk certification

`bayesian_phystwin.guard_harm_risk_domains` closes a gap between the
calibration-domain guard and the finite-group harmful-update certificate.

The existing domain guard can authorize a candidate in several declared
physical regimes. The existing harm-risk certificate can certify a pooled set
of accepted groups. Those statements are not interchangeable: a pooled upper
bound can pass while a smaller regime has an unacceptable harmful-update rate.

This module therefore requires one finite-group harm certificate for every
domain authorized by a calibration-frozen domain guard.

## Contract

A `DomainGuardHarmRiskCertificateV1` binds:

- the exact `CalibrationDomainGuardCertificateV1` and each supported domain's
  content-addressed decision;
- one `GuardHarmRiskCertificateV1` for every supported domain;
- the common threshold source, certification partition, statistical unit,
  metric, threshold, harm margin, target harmful-update probability, and
  minimum accepted-group count;
- globally disjoint calibration, threshold-selection, and certification
  groups; and
- a family confidence level covering the complete supported-domain roster.

The composite record owns only the cross-domain policy statement. Each
underlying finite-group certificate retains its existing independent artifact
identity and can still be audited under the original harm-risk contract.

The per-domain confidence level uses Bonferroni control:

```text
per_domain_confidence = 1 - (1 - family_confidence) / supported_domain_count
```

Every supported domain must satisfy its own support requirement and exact
one-sided Clopper--Pearson bound at that adjusted confidence level. The
composite certificate is deployment-admissible only when:

1. the calibration-domain guard is prospective and deployment-admissible; and
2. every supported domain has a valid, certified harm-risk record.

A pooled certificate is deliberately not accepted as a substitute.

## Exact fallback

Application routing uses `select_domain_guard_harm_risk_belief`. The candidate
complete belief is selected only when inference is admissible, the requested
domain was supported during calibration, the requested domain is certified,
and all other supported domains are also certified.

Any unknown, unsupported, under-supported, harmful, retrospective, malformed,
or incomplete state routes through `select_complete_belief` and returns the
exact registered baseline belief object. The rejected path does not reconstruct
a nominal fallback from zero corrections.

The all-domain requirement is intentional. Once a single content-addressed
policy is described as deployable over a declared domain roster, a failure in
one supported regime invalidates that policy-level deployment statement. A
future version may define separately versioned, independently deployed
single-domain policies instead of weakening this certificate.

## Usage

```python
from bayesian_phystwin.guard_harm_risk_domains import (
    certify_domain_guard_harm_risk,
    select_domain_guard_harm_risk_belief,
)

certificate = certify_domain_guard_harm_risk(
    domain_guard_certificate=domain_guard,
    threshold_source_artifact_id=threshold_source_id,
    certification_partition_id=certification_partition_id,
    threshold_selection_group_ids=threshold_selection_group_ids,
    group_ids=certification_group_ids,
    domain_ids=certification_domain_ids,
    risk_scores=risk_scores,
    candidate_losses=candidate_losses,
    fallback_losses=fallback_losses,
    fallback_identity_verified=fallback_identity_verified,
    threshold=threshold,
    harm_margin=harm_margin,
    target_harm_probability=0.10,
    family_confidence_level=0.95,
    minimum_accepted_group_count=minimum_accepted_groups,
    threshold_frozen_before_certification_outcomes=True,
    certification_outcomes_used_for_threshold_selection=False,
    certification_groups_independent=True,
)

selected, receipt = select_domain_guard_harm_risk_belief(
    baseline,
    candidate,
    certificate,
    domain_id=application_domain,
    common_domain_id=common_domain_id,
    inference_admissible=True,
)
```

The function receives no application loss and cannot use an application
outcome to change the domain decision, threshold, or certification result.

## Finite-group consequence

Multiplicity control increases the required independent support. With two
supported domains and 95% family confidence, each domain is evaluated at 97.5%
confidence. Even with zero harmful accepted updates, each domain then requires
at least:

- 17 accepted independent groups for a 20% upper risk target;
- 36 accepted independent groups for a 10% upper risk target; and
- 72 accepted independent groups for a 5% upper risk target.

These counts apply per domain, not to pooled frames, points, windows, tracks, or
taxels. Insufficient support remains a valid negative result and cannot be
rescued by pooling another regime's groups.

## Compatibility and claim boundary

The module is additive and is not exported from the frozen package-root API.
It changes no existing estimator, domain certificate, harm certificate,
threshold, result, or artifact identity. Existing single-domain or pooled
analyses remain readable under their original contracts; they simply do not
establish this stronger simultaneous-domain statement.

Passing this certificate establishes a finite-group upper bound for one frozen
policy over the declared, independently sampled certification domains. It does
not establish provider competence, calibrated uncertainty under arbitrary
shift, causal sufficiency, physical-query benefit, Causal4D intervention
benefit, deployment safety, or state of the art.
