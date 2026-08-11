# Claim-bearing domain covariance calibration v2

`domain_covariance_calibration_v2` is an additive admission layer around the
experimental version-1 scale-plus-floor fitter. Version 1 and its existing
content identities remain unchanged.

The version-2 layer addresses four boundaries that are required before a fitted
covariance transform can be used in a claim-bearing run:

1. **Conservative default.** The default transform grid contains only scales
   greater than or equal to one and nonnegative isotropic variance. A custom
   grid containing covariance shrinkage is rejected unless the frozen policy
   explicitly allows it.
2. **Physical semantics.** The certificate binds covariance dimension,
   coordinate frame, physical unit, query type, and horizon semantics.
   Application requires an independently supplied descriptor matching all five
   fields. Any mismatch returns the exact caller-owned covariance object.
3. **Finite-group support.** Version 2 independently requires a minimum number
   of groups, a practical mean leave-one-group-out Gaussian-NLL improvement, a
   minimum held-group win fraction, and a worst-group harm limit. These checks
   supplement rather than replace the version-1 calibration-domain guard.
4. **Content-addressed admission.** Application accepts both an explicit
   `CovarianceSemanticsV2` descriptor and an `EvidenceDecisionV1` bound to the
   registered claim, protocol, and exact version-2 certificate. The decision's
   metadata must contain the certificate artifact ID under
   `domain_covariance_calibration_v2_certificate_id`; a decision for another
   certificate cannot be replayed even when claim and protocol IDs match. The
   complete semantics descriptor must also match the certificate, and a naked
   Boolean cannot authorize the version-2 path. The decision must be passing,
   confirmatory, and meet the frozen evidence-level policy. A deployment policy
   may additionally require `claim_authorized=true`.

## Fitting

```python
from bayesian_phystwin.domain_covariance_calibration_v2 import (
    fit_domain_covariance_calibration_v2,
)

certificate = fit_domain_covariance_calibration_v2(
    predictor_id=predictor_id,
    calibration_partition_id=partition_id,
    statistical_unit="independent-physical-session",
    residual_semantics="endpoint-position-error-m",
    covariance_semantics="raw-endpoint-position-covariance-m2",
    coordinate_frame="phystwin-world",
    physical_unit="m2",
    query_type="endpoint-position",
    horizon_semantics="fixed-endpoint-frame",
    admission_claim_id="claim/domain-covariance-calibration-v2",
    admission_protocol_id="protocol/domain-covariance-calibration-v2",
    event_ids=event_ids,
    group_ids=group_ids,
    domain_ids=domain_ids,
    residuals=residuals,
    covariances=covariances,
    domain_guard=domain_guard,
    predictor_frozen_before_calibration_outcomes=True,
    transform_grid_frozen_before_calibration_outcomes=True,
    application_outcomes_used_for_calibration_selection=False,
    calibration_groups_independent=True,
)
```

The returned certificate embeds the exact version-1 certificate ID, the bound
semantics, the complete policy, per-domain held-group win fractions, and all
version-2 rejection reasons.

## Application

The admission decision must be built after the certificate exists and bind that
exact artifact in its content-addressed metadata. The implementation does not
infer certificate identity from a matching claim ID, protocol ID, or status:

```python
from bayesian_phystwin.domain_covariance_calibration_v2 import (
    EVIDENCE_CERTIFICATE_ID_METADATA_KEY,
    apply_domain_covariance_calibration_v2,
)

admission_decision = build_evidence_decision(
    manifest=manifest,
    evidence_summary_path=evidence_summary_path,
    claim_id=certificate.admission_claim_id,
    status="pass",
    claim_authorized=False,
    evidence_level=3,
    metric=decision_metric,
    metadata={
        EVIDENCE_CERTIFICATE_ID_METADATA_KEY: certificate.artifact_id,
    },
)

covariance, application = apply_domain_covariance_calibration_v2(
    raw_covariance,
    certificate,
    domain_id="dynamic",
    application_semantics=application_semantics,
    evidence_decision=admission_decision,
)
```

Application returns the exact input NumPy object when any boundary fails,
including unknown domain, semantic mismatch, nonprospective source certificate,
policy rejection, missing or mismatched certificate binding, unmatched evidence
decision, or an identity transform. An accepted nonidentity transform returns a
new immutable float64 array. The application record binds the version-2
certificate, both the certified and application-declared semantics, the evidence
decision, the underlying version-1 application, canonical float64 numerical
digests, and exact shape/dtype/byte digests of the actual input and returned
arrays. Invalid, nonsymmetric, or indefinite covariance matrices are rejected
before fallback routing.

## Scientific boundary

This module is infrastructure for a future independently frozen calibration and
evaluation study. Unit tests or a successful fit do not establish calibrated
uncertainty, physical-query benefit, fresh-object transfer, Causal4D benefit,
deployment safety, or state of the art. The transform and its admission decision
must be evaluated on independent physical object or session groups with the
claim and protocol fixed before application outcomes are opened.
