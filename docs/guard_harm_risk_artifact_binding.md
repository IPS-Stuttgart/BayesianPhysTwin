# Artifact-bound exact fallback for guard-risk certification

## Purpose

The finite-group harmful-update certificate accepts a boolean vector stating
whether deployment returned the exact fallback on each group. That vector is a
useful lower-level contract input, but a claim-bearing result should not trust a
caller-supplied declaration when selected and fallback artifacts have stable
content identities.

`bayesian_phystwin.guard_harm_risk_artifacts` supplies the stronger boundary. It
binds, for every independent certification group:

- the selected deployment artifact identity;
- the registered physical-fallback artifact identity; and
- the equality result derived from those two SHA-256 identities.

The claim-bearing helper has no fallback-verification boolean argument. It
recomputes the mask from artifact identities and passes that immutable mask into
the finite-group risk certificate.

## Rejection invariant

For an accepted group, the selected candidate is normally different from the
fallback and no equality is required. For every rejected group, the invariant is

```text
selected_artifact_id == fallback_artifact_id.
```

A rejected group with unequal identities fails before a risk certificate can be
constructed. This is stronger than checking a zero-valued correction vector or
a caller-provided `fallback_verified=true` field: the complete selected artifact
must be the exact registered fallback artifact.

## Compound certificate

`GuardFallbackArtifactBindingV1` content-addresses the ordered group identities,
selected identities, fallback identities, derived equality mask, and metadata.
Group order is canonicalized, so reordering the same records does not change the
artifact ID.

`GuardHarmRiskArtifactCertificateV1` then binds:

- the fallback-artifact binding ID;
- the underlying finite-group risk-certificate ID;
- exact equality between the risk certificate's verification mask and the mask
  recomputed from artifact IDs; and
- the final certification decision.

Directly combining a forged boolean mask with valid-looking artifact IDs is
rejected by the compound constructor.

## Usage

```python
from bayesian_phystwin.guard_harm_risk_artifacts import (
    certify_guard_harm_risk_from_artifacts,
)

certificate = certify_guard_harm_risk_from_artifacts(
    guard_policy_id=guard_policy_id,
    threshold_source_artifact_id=threshold_source_artifact_id,
    certification_partition_id=certification_partition_id,
    statistical_unit="independent-physical-object-v1",
    metric="endpoint-rmse-m",
    threshold_selection_group_ids=threshold_selection_group_ids,
    group_ids=certification_group_ids,
    risk_scores=risk_scores,
    candidate_losses=candidate_losses,
    fallback_losses=fallback_losses,
    selected_artifact_ids=selected_artifact_ids,
    fallback_artifact_ids=fallback_artifact_ids,
    threshold=frozen_threshold,
    harm_margin=0.0,
    target_harm_probability=0.20,
    confidence_level=0.95,
    minimum_accepted_group_count=14,
    threshold_frozen_before_certification_outcomes=True,
    certification_outcomes_used_for_threshold_selection=False,
    certification_groups_independent=True,
)
```

Use this artifact-bound helper for a claim-bearing exact-fallback statement. The
lower-level finite-group constructor remains useful when fallback equality was
already verified by an independently content-addressed execution contract.

## Claim boundary

Artifact equality proves exact routing, not method quality. The compound
certificate still controls only the declared conditional harmful-update rate for
one frozen guard under exchangeable independent certification groups. It does
not establish provider competence, uncertainty coverage, physical-query
accuracy, robustness under distribution shift, deployment safety, Causal4D
benefit, or state of the art.
