# Mapping-bound claim-bearing Prob4D candidate

`mapped_prob4d_candidate_v1` closes the gap between the provider-to-physical
mapping audit and the existing claim-bearing Prob4D inference path.

The historical candidate remains unchanged. The new composition adds a strict
preflight and a content-addressed wrapper:

1. the mapping audit must be a valid
   `ProviderPhysicalMappingAuditV1` artifact;
2. the audit must identify the exact `ObservationBeliefV1` artifact supplied to
   inference;
3. the audit must identify the exact physical-query digest supplied by the
   caller;
4. the audit must be admissible; and
5. only after those checks may the existing claim-bearing Prob4D candidate
   inference run.

A failed or mismatched audit raises before the observation is adapted and before
the physical solver is called. It therefore cannot produce a physical candidate
that merely carries a warning in metadata.

## Binding an existing candidate

```python
from bayesian_phystwin.mapped_prob4d_candidate_v1 import (
    bind_provider_mapping_to_prob4d_candidate,
)

mapped = bind_provider_mapping_to_prob4d_candidate(
    candidate,
    mapping_audit,
    physical_query_id=physical_query_id,
)
```

The resulting `MappedClaimBearingProb4DCandidateV1` binds:

- the complete claim-bearing candidate identity;
- the complete provider-to-physical mapping-audit identity;
- the provider observation artifact;
- the physical query;
- the mapping protocol;
- the covariance semantics; and
- the underlying inference admission result.

## One-call guarded inference

```python
from bayesian_phystwin.mapped_prob4d_candidate_v1 import (
    infer_mapped_claim_bearing_prob4d_candidate_from_artifacts,
)

mapped = infer_mapped_claim_bearing_prob4d_candidate_from_artifacts(
    observation_belief,
    linearization,
    mapping_audit=mapping_audit,
    physical_query_id=physical_query_id,
    physical_prediction_xyz_m=physical_prediction_xyz_m,
)
```

The function performs the mapping checks before delegating to
`infer_claim_bearing_prob4d_candidate_from_artifacts`. All historical solver,
covariance, fallback, and provider-attestation semantics are preserved.

## Failure behavior

The composition fails closed for:

- an inadmissible mapping audit;
- a provider-artifact mismatch;
- a physical-query mismatch;
- a malformed audit identity; or
- a forged mapped-candidate identity.

A valid mapping does not force the Bayesian solve to succeed. If the existing
solver rejects its update, the wrapper preserves the exact-prior-fallback
covariance semantics and the original rejection reason.

## Scientific boundary

This composition is software lineage and fail-closed mapping evidence only. It
does not establish provider competence, covariance calibration, fresh-object
physical benefit, Causal4D intervention benefit, deployment safety, or state of
the art. It changes no registered target roster, experiment threshold, frozen
candidate, or target-access authorization.
