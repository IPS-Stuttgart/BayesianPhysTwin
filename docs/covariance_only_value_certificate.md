# Covariance-only value certificate

## Purpose

`bayesian_phystwin.covariance_only_value` supplies a prospective admission lane
for a candidate that changes predictive covariance but must not change its point
prediction. It is deliberately separate from the baseline-relative point-regret
guard.

The certificate answers one bounded question on independent physical objects or
acquisition sessions:

> Does one already frozen covariance-only policy retain proper-score value,
> remain within its registered interval-width budget, and keep the probability
> of a harmful group below its registered limit while preserving the reference
> mean exactly?

It does not select a covariance donor, horizon scale, score, width budget,
threshold, query, or cohort from the certification outcomes. It does not replace
provider-competence, identifiability, nonlinear-closure, calibration, or exact-
fallback gates.

## Exact-mean boundary

Every certification group supplies a SHA-256 digest for the candidate point
prediction and the matched reference point prediction. The contract derives

```text
mean_identity[g] = candidate_mean_sha256[g] == reference_mean_sha256[g].
```

A mismatch is retained as a valid negative certificate rather than hidden by a
numerical tolerance. The certificate can pass only when every independent group
has exact digest identity.

This supports experiments such as the frozen `last_residual` mean with an
`independent_endpoint_v1` covariance donor. It does not establish that either
mean is physically correct.

## Three simultaneous gates

For each independent group `g`, define the proper-score difference

```text
Delta[g] = candidate_score[g] - reference_score[g],
```

where lower is better. The score metric, registered query set, aggregation
within a physical group, and finite bounds `[L, U]` must be frozen before the
certification outcomes are opened. Values outside `[L, U]` fail closed; they are
not clipped after inspection.

The certificate evaluates three one-sided gates:

1. expected proper-score regret;
2. expected candidate full interval width; and
3. harmful-group probability, where `Delta[g] > harm_margin`.

The first two use bounded-mean Hoeffding upper confidence bounds. The third uses
the exact one-sided Clopper--Pearson binomial upper bound implemented by
`guard_harm_risk`. The requested familywise confidence is divided across the
three gates by Bonferroni:

```text
per_gate_confidence = 1 - (1 - familywise_confidence) / 3.
```

The certificate passes only when all of the following hold:

```text
all point-mean digests are identical
number of independent groups >= minimum_group_count
score upper bound <= maximum_expected_score_regret
width upper bound <= maximum_expected_full_width
harm upper bound <= target_harm_probability
```

The bounded-mean result is conservative when the registered ranges are wide.
That weakness is visible evidence rather than a reason to tighten the bounds
after outcomes are inspected.

## Statistical unit and information order

One row must represent one independent physical object or acquisition session.
Frames, points, tracks, views, pixels, tactile taxels, horizons, and query
coordinates remain nested observations. Their aggregation into the group score,
width, and harm event must be frozen in the policy and query-set identities.

The contract requires:

- an exact candidate-policy identity;
- an exact reference-policy identity;
- an exact registered-query identity;
- an exact policy-freeze artifact;
- an exact certification-partition identity;
- disjoint policy-selection and certification group IDs;
- thresholds frozen before certification outcomes;
- no use of certification outcomes for policy selection; and
- independent physical certification groups.

Scanning donors, scale schedules, width limits, harm margins, score bounds, or
queries on the same certification outcomes invalidates this certificate.

## Example

```python
import numpy as np

from bayesian_phystwin.covariance_only_value import (
    certify_covariance_only_value,
    save_covariance_only_value_certificate,
)

certificate = certify_covariance_only_value(
    candidate_policy_id=candidate_policy_id,
    reference_policy_id=last_residual_policy_id,
    query_set_id=query_set_id,
    policy_freeze_artifact_id=policy_freeze_id,
    certification_partition_id=certification_partition_id,
    statistical_unit="independent-physical-object-v1",
    score_metric="group-gaussian-nll-v1",
    width_metric="group-mean-full-width-m-v1",
    selection_group_ids=source_object_ids,
    group_ids=certification_object_ids,
    candidate_mean_sha256=candidate_mean_digests,
    reference_mean_sha256=reference_mean_digests,
    candidate_scores=np.asarray(candidate_scores),
    reference_scores=np.asarray(reference_scores),
    candidate_full_widths=np.asarray(candidate_widths_m),
    reference_full_widths=np.asarray(reference_widths_m),
    score_difference_lower_bound=-20.0,
    score_difference_upper_bound=20.0,
    full_width_upper_bound=0.100,
    maximum_expected_score_regret=0.0,
    maximum_expected_full_width=0.060,
    harm_margin=0.0,
    target_harm_probability=0.20,
    familywise_confidence_level=0.95,
    minimum_group_count=20,
    thresholds_frozen_before_certification_outcomes=True,
    certification_outcomes_used_for_policy_selection=False,
    certification_groups_independent=True,
    metadata={"protocol": "registered-covariance-only-confirmation-v1"},
)

save_covariance_only_value_certificate(
    certificate,
    "covariance-only-value-certificate.json",
)
```

Saving is atomic and non-overwriting by default. Loading rejects duplicate JSON
keys, schema drift, altered derived masks or bounds, changed content identity,
non-real arrays, non-finite values, and post-hoc violations of the registered
bounded ranges. Retained arrays use immutable bytes-backed NumPy storage.

## Relationship to the other gates

The intended prospective sequence is:

```text
provider support and source competence
        -> registered physical-query identifiability
        -> exact-mean covariance candidate
        -> this score/width/harm certificate
        -> query calibration on disjoint calibration groups, when registered
        -> one target evaluation with exact fallback
```

A mean-changing candidate still requires the ordinary point-regret and nonlinear-
closure route. A covariance-only certificate must not be used to authorize a
point update merely because its score improved.

## Claim boundary

A passing artifact supports only the registered finite-group statement for the
fixed policy, bounded score, width metric, harm definition, and exchangeable
physical-group population. It does not establish:

- calibrated raw posterior covariance;
- conditional coverage for every horizon, object type, or observability level;
- provider competence or material-identity correctness;
- physical-state identifiability or a unique causal explanation;
- safety or deployment authorization;
- Causal4D intervention benefit; or
- state of the art.

A separate target result must still report empirical coverage, width, point
identity, exact fallback, worst-group behavior, technical failures, and any
predeclared subgroup effects. A failed or underpowered certificate is complete
evidence and must not be rescued by retuning on the same groups.
