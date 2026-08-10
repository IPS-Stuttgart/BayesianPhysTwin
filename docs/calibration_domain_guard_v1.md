# Calibration-frozen domain guard v1

## Purpose

One Bayesian correction can help dynamic continuation and still harm a different
regime, such as contact-rich quasi-static evolution. A global average therefore
is not a safe deployment rule. `bayesian_phystwin.calibration_domain_guard`
turns independent calibration-group losses into a content-addressed domain
certificate and routes every unsupported application to the exact registered
baseline belief.

The module is explicit and experimental. It is not exported from the stable
package root.

## Calibration rule

For each declared domain, the default rule requires all of the following:

- at least three independent calibration groups;
- mean relative improvement of at least 5%;
- wins on at least two thirds of calibration groups; and
- no single-group relative regression greater than 5%.

For lower-is-better losses, group-level relative improvement is

```text
(fallback_loss - candidate_loss) / fallback_loss.
```

Threshold comparisons are inclusive up to the frozen numerical tolerance. Group
identifiers are unique statistical units, and records are sorted before content
addressing, so input order does not change the certificate identity.

Unknown domains, insufficiently supported domains, and malformed inputs fail
closed.

## Information boundary

The calibration factory records three boundary facts:

- whether the guard was frozen before application outcomes;
- whether application outcomes were used for guard selection; and
- whether calibration groups are independent.

A certificate is deployment-admissible only when the guard was frozen first,
application outcomes were not used, and the calibration groups are independent.
A retrospective diagnostic can still preserve its calibration calculations, but
its certificate cannot select the candidate belief.

The application function accepts a domain identifier and an inference
admissibility flag. It has no application-loss or target-outcome argument.

## Exact complete-belief fallback

Selection delegates to `select_complete_belief`. Rejection returns the baseline
object itself rather than reconstructing a zero correction. This preserves state,
parameters, uncertainty, particles, discrepancy, nuisance moments, and
provenance together.

Candidate selection requires all three conditions:

1. inference is admissible;
2. the certificate information boundary is prospective; and
3. calibration supports the declared domain.

Failure of any condition, including an unseen domain, returns the exact baseline
belief.

## Example

```python
from bayesian_phystwin.calibration_domain_guard import (
    fit_calibration_domain_guard,
    select_calibration_domain_guarded_belief,
)

certificate = fit_calibration_domain_guard(
    calibration_partition_id=calibration_partition_id,
    statistical_unit="independent-physical-cloth-trial",
    metric="symmetric-l1-chamfer-m",
    group_ids=calibration_trial_ids,
    domain_ids=calibration_regimes,
    candidate_losses=calibration_candidate_losses,
    fallback_losses=calibration_physical_losses,
    guard_frozen_before_application_outcomes=True,
    application_outcomes_used_for_guard_selection=False,
    calibration_groups_independent=True,
)

selected, routing = select_calibration_domain_guarded_belief(
    physical_belief,
    candidate_belief,
    certificate,
    domain_id=current_regime,
    common_domain_id=physical_query_domain_id,
    inference_admissible=inference_result.accepted,
)
```

The certificate binds the calibration partition, statistical unit, metric,
absolute calibration losses, per-domain relative improvements, threshold
configuration, decisions, information-boundary flags, and metadata.

## Relation to the real-Cloth diagnostic

The retrospective real-Cloth covariance study motivated this interface. Its
calibration-repeat diagnostic supported dynamic continuation and rejected
quasi-static continuation under the same 5% / two-thirds / 5% rule. Because that
rule was specified after the target outcomes were already open, the recorded
study remains diagnostic rather than prospective evidence.

This module prevents such a retrospective certificate from routing the candidate.
A fresh calibration-frozen cohort is still required for a claim-bearing result.

## Scientific boundary

Domain authorization is not evidence of general uncertainty calibration,
provider competence, fresh-object transfer, Causal4D intervention benefit,
deployment safety, or state of the art. It controls only the declared loss and
domains under the recorded calibration assumptions. Unknown or shifted regimes
must retain the physical fallback.
