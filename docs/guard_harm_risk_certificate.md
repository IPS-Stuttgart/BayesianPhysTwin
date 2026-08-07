# Finite-group harmful-update risk certificate

## Purpose

`decisive_evidence.py` reports how many accepted updates were harmful and draws
risk–coverage curves. Those are essential diagnostics, but an observed fraction
is not an upper confidence bound on the future harmful-update probability.

`bayesian_phystwin.guard_harm_risk` adds a separate prospective certificate for
one already frozen baseline-relative guard threshold. It answers:

> On independent physical objects or acquisition sessions accepted by this
> fixed policy, how large can the harmful-update probability still be at the
> declared confidence level?

The certificate does not select a threshold, modify an update, or replace exact
fallback.

## Statistical unit and harm definition

The input contains one record per independent physical unit, normally one
complete object or acquisition session. Frames, views, tracks, points, pixels,
and tactile taxels do not increase the group count.

For certification group `i`, the frozen policy accepts when

```text
risk_score_i <= threshold.
```

A candidate is harmful when

```text
candidate_loss_i > fallback_loss_i + harm_margin.
```

The metric, margin, inclusive threshold semantics, guard-policy identity,
threshold-source identity, and certification partition are all content-bound.
Every rejected group must independently verify that deployment returned the
exact fallback.

When several horizons, endpoints, or metrics belong to one physical group, a
claim-bearing protocol must either:

- aggregate them into one predeclared group-level harm event, such as the maximum
  registered regression; or
- register separate certificates and an appropriate multiplicity policy before
  outcomes are opened.

It must not count correlated endpoints as additional independent trials.

## Exact upper bound

For `k` harmful accepted groups among `n` accepted independent groups, the
certificate reports the exact one-sided Clopper–Pearson upper bound `p_upper`
at confidence `1 - delta`:

```text
P[Binomial(n, p_upper) <= k] = delta.
```

The zero-harm case has the closed form

```text
p_upper = 1 - delta ** (1 / n).
```

No accepted groups and an all-harmful accepted set both have upper bound one.
General cases use deterministic inversion of the binomial CDF without SciPy.
The implementation does not replace the exact bound with a Wald approximation.

The certificate passes only if both hold:

```text
accepted_count >= minimum_accepted_group_count
p_upper <= target_harm_probability.
```

Insufficient support is retained as a valid negative certificate rather than an
exception or a silently weakened target.

## Information-order boundary

The guarantee applies to one threshold frozen independently of the certification
outcomes. The contract therefore requires all of the following:

- the threshold is frozen before certification losses are opened;
- certification outcomes did not select the threshold;
- threshold-selection group IDs are content-bound;
- threshold-selection and certification groups are disjoint; and
- certification groups are declared independent physical units.

An empty threshold-selection group list is allowed for a threshold fixed from
source-only reasoning or a prior external experiment. A nonempty list is sorted
and checked against every certification group ID.

Scanning several thresholds on the certification outcomes and publishing the
best one invalidates this certificate. That operation requires a separately
versioned simultaneous or split-sample procedure.

## Finite-group planning consequence

Even zero observed harm can support only a limited claim. At 95% confidence,
the minimum accepted independent-group counts are:

| Target harmful-update probability | Zero-harm groups required |
| ---: | ---: |
| 30% | 9 |
| 25% | 11 |
| 20% | 14 |
| 10% | 29 |
| 5% | 59 |

For example, zero harmful updates among ten accepted objects gives an upper
95% bound of approximately `0.2589`. Ten objects therefore cannot certify a
20% or 10% harmful-update cap at 95% confidence, even under the most favorable
zero-harm outcome.

This is relevant to the ten-object Deform360 calibration stage. Those ten
objects can diagnose guard behavior and may support a looser bounded statement,
but they cannot by themselves authorize a strict 10% harmful-update claim. If
some objects select the threshold, the disjoint certification support is smaller
still.

Use the planning helper before target access:

```python
from bayesian_phystwin.guard_harm_risk import (
    minimum_zero_harm_groups_for_certificate,
)

assert minimum_zero_harm_groups_for_certificate(0.10, 0.95) == 29
assert minimum_zero_harm_groups_for_certificate(0.20, 0.95) == 14
```

## Building a certificate

```python
import numpy as np

from bayesian_phystwin.guard_harm_risk import (
    certify_guard_harm_risk,
    save_guard_harm_risk_certificate,
)

certificate = certify_guard_harm_risk(
    guard_policy_id=guard_policy_id,
    threshold_source_artifact_id=threshold_source_artifact_id,
    certification_partition_id=certification_partition_id,
    statistical_unit="independent-physical-object-v1",
    metric="endpoint-rmse-m",
    threshold_selection_group_ids=source_or_selection_group_ids,
    group_ids=certification_object_ids,
    risk_scores=np.asarray(risk_scores),
    candidate_losses=np.asarray(candidate_losses),
    fallback_losses=np.asarray(fallback_losses),
    fallback_identity_verified=np.asarray(fallback_verified, dtype=bool),
    threshold=frozen_threshold,
    harm_margin=0.0,
    target_harm_probability=0.20,
    confidence_level=0.95,
    minimum_accepted_group_count=10,
    threshold_frozen_before_certification_outcomes=True,
    certification_outcomes_used_for_threshold_selection=False,
    certification_groups_independent=True,
    metadata={"protocol": "registered-guard-certificate-v1"},
)

save_guard_harm_risk_certificate(
    certificate,
    "guard-harm-risk-certificate.json",
)
```

Saving is atomic and non-overwriting by default. Loading rejects duplicate JSON
keys, unknown or missing fields, non-finite values, changed derived masks or
counts, altered exact bounds, inconsistent certification decisions, and content
identity tampering.

## Retained evidence

`GuardHarmRiskCertificateV1` retains:

- the exact guard, threshold-source, and certification-partition identities;
- the statistical unit, metric, harm margin, and risk-score semantics;
- disjoint threshold-selection and certification group IDs;
- risk scores, candidate losses, fallback losses, and fallback verification;
- derived accepted and harmful masks;
- accepted and harmful accepted counts;
- observed harmful-update frequency when support exists;
- exact one-sided upper confidence bound;
- minimum zero-harm support needed for the declared target;
- information-order and independence declarations; and
- an order-invariant content identity.

Arrays are defensively owned and read-only. Reordering the same physical groups
does not change the artifact ID.

## Claim boundary

The certificate controls only the stated conditional harmful-update probability
for the fixed policy under exchangeable independent certification groups. It is
not:

- evidence that the observation provider is competent;
- a guarantee under object, sensor, or acquisition distribution shift;
- a physical-query accuracy or coverage certificate;
- a simultaneous guarantee over adaptively selected thresholds or metrics;
- a substitute for accepted-update coverage and interval-width reporting;
- a safety or deployment authorization; or
- evidence of Causal4D intervention benefit or state of the art.

The target evaluation must still report unconditional deployed loss, acceptance
rate, exact fallback, worst-object regression, uncertainty coverage and width,
and every technical failure. A negative or underpowered certificate is complete
evidence and must not be rescued by selecting another threshold on the same
certification outcomes.
