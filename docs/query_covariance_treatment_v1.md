# Query-space covariance treatment v1

This document describes an additive, experimental interface for deciding whether
shared covariance matters in a registered physical query and whether one
covariance-only BayesianPhysTwin candidate may replace the exact deterministic
baseline belief.

The interface changes no frozen estimator, provider, cohort, calibration result,
or scientific claim. It is infrastructure for a separately registered fresh
object/session study.

## Motivation

Observation-space covariance can be large while being almost irrelevant to the
physical quantity consumed by BayesianPhysTwin or Causal4D. Conversely, a
low-rank shared mode can dominate a registered physical query even when its
marginal point variances appear small.

A covariance-only candidate also needs a stronger admission boundary than a
single numerical-success Boolean. The useful current candidate preserves the
exact `last_residual` mean and changes only its uncertainty treatment. Admission
therefore needs to bind all of the following without reconstructing a rejected
belief:

- exact predictive-mean identity;
- the calibrated covariance application;
- query-space relevance of shared covariance;
- proper-score evidence on independent object/session units;
- simultaneous query coverage and interval width;
- harmful accepted-update certification; and
- exact complete-belief fallback.

## Query-space relevance certificate

For a registered query Jacobian `J`, local covariance `D`, shared factor `U`, and
optional query-noise covariance `R`, the interface computes

```text
local query covariance  = J D J^T
shared query covariance = (J U) (J U)^T
reference covariance    = J D J^T + R
```

`QueryCovarianceRelevanceCertificateV1` records:

- the shared fraction of total query-space covariance trace;
- the effective rank of `J U`;
- the fraction of declared shared modes that are null in the query;
- the maximum generalized eigenvalue of the shared covariance relative to the
  local-plus-noise covariance;
- source/calibration partition and statistical-unit identities; and
- whether the rule was frozen before target outcomes and fitted on independent
  groups.

Materiality and evidence admissibility are deliberately separate. A shared mode
can be physically material even when the certificate is inadmissible because its
rule was selected using target outcomes. That case must reject deployment rather
than relabel the shared mode as irrelevant.

The frozen policy selects one of two modes:

- `marginal` when shared covariance is not material in the registered query;
- `explicit-joint` when shared covariance is material.

The certificate does not choose the query, provider, target cohort, or promotion
threshold.

## Covariance-only treatment decision

`CovarianceOnlyTreatmentDecisionV1` composes the following immutable inputs:

1. `CovarianceOnlyHybridRecordV1`, proving that the caller-owned reference mean
   is unchanged;
2. `DomainCovarianceCalibrationApplicationV2`, including exact input covariance
   identity, exact calibrated output identity, and whether calibrated covariance
   was actually applied;
3. `GuardHarmRiskArtifactCertificateV1`, including selected-versus-fallback
   artifact equality for rejected groups;
4. `QueryCovarianceRelevanceCertificateV1`;
5. one confirmatory `EvidenceDecisionV1` for the registered proper-score result,
   simultaneous query coverage, and mean interval-width ratio; and
6. the frozen covariance-mode, proper-score, coverage, width, evidence-level,
   and information-boundary policy.

The confirmatory evidence decision must bind the exact baseline, candidate,
hybrid record, calibrated candidate covariance bytes, calibration application,
harm-risk certificate, query-relevance certificate, registered query,
calibration partition, and statistical unit in its metadata. The decision metric
must match the frozen proper-score name, comparison, rule, and threshold.

Authorization requires all gates to pass together:

- candidate inference is numerically admissible;
- the calibration application consumed the exact hybrid covariance bytes;
- the proposed covariance identity equals the exact calibrated output bytes;
- calibration was applied under admissible evidence rather than falling back;
- the harmful-update certificate passes;
- the query-relevance certificate is deployment-admissible;
- the selected marginal or explicit-joint mode matches query materiality;
- the confirmatory evidence decision binds the exact candidate and passes the
  registered proper-score threshold;
- simultaneous query coverage reaches its frozen minimum;
- mean full width remains below its frozen ratio limit;
- the treatment was frozen before target outcomes;
- target outcomes did not select the treatment; and
- complete physical object/session units are the independent statistical units.

A failed gate returns a content-addressed rejection reason. It does not trigger
threshold selection, covariance substitution, camera deletion, or cohort
replacement.

## Complete-belief selection

`select_covariance_only_belief` converts the treatment decision into the existing
complete-belief guard. Authorization returns the caller-owned complete candidate
belief. Any rejection returns the exact caller-owned baseline object, preserving
state, parameters, particles, discrepancy moments, nuisance beliefs, covariance,
dtype, and provenance together.

```python
from bayesian_phystwin.query_covariance_treatment import (
    CovarianceOnlyTreatmentPolicyV1,
    QueryCovarianceRelevancePolicyV1,
    certify_query_covariance_relevance,
    decide_covariance_only_treatment,
    select_covariance_only_belief,
)

relevance = certify_query_covariance_relevance(
    query_id=query_id,
    covariance_artifact_id=calibration_application.output_array_sha256,
    jacobian_artifact_id=query_jacobian_id,
    calibration_partition_id=calibration_partition_id,
    statistical_unit="physical-object-session",
    local_covariance=local_covariance,
    shared_factor=shared_factor,
    query_jacobian=query_jacobian,
    query_noise_covariance=query_noise,
    policy=QueryCovarianceRelevancePolicyV1(
        minimum_shared_trace_fraction=0.05,
        minimum_maximum_generalized_eigenvalue=0.10,
    ),
    frozen_before_target_outcomes=True,
    target_outcomes_used_for_selection=False,
    calibration_groups_independent=True,
)

# The confirmatory EvidenceDecisionV1 is produced by the registered evaluator
# and binds all dependency IDs plus coverage and width in its metadata.
decision = decide_covariance_only_treatment(
    baseline_belief_id=baseline.artifact_id,
    candidate_belief_id=candidate.artifact_id,
    common_domain_id=common_domain_id,
    candidate_covariance_artifact_id=(
        calibration_application.output_array_sha256
    ),
    query_id=query_id,
    calibration_partition_id=calibration_partition_id,
    statistical_unit="physical-object-session",
    hybrid_record=hybrid.record,
    calibration_application=calibration_application,
    harm_risk_certificate=harm_certificate,
    query_relevance=relevance,
    evaluation_decision=evaluation_decision,
    candidate_inference_admissible=True,
    treatment_frozen_before_target_outcomes=True,
    target_outcomes_used_for_treatment_selection=False,
    independent_statistical_units=True,
    policy=CovarianceOnlyTreatmentPolicyV1(
        covariance_mode="explicit-joint",
        evaluation_claim_id="fresh-query-covariance-v1",
        evaluation_protocol_id="fresh-query-covariance-protocol-v1",
        proper_score_metric_name="query-gaussian-nll-upper-confidence-bound",
        proper_score_comparison="candidate-minus-last-residual",
        proper_score_rule="upper-confidence-bound-at-most",
        maximum_proper_score_value=0.0,
        minimum_simultaneous_coverage=0.90,
        maximum_mean_full_width_ratio=4.0,
    ),
)

selected, selection = select_covariance_only_belief(
    baseline,
    candidate,
    decision,
)
```

## Evidence boundary

Passing unit tests or constructing these artifacts is software and contract
evidence only. It does not establish:

- real Prob4D provider competence;
- fresh-object BayesianPhysTwin benefit;
- calibrated deployment uncertainty;
- Causal4D interventional benefit;
- overall state of the art; or
- authorization to open a sealed target cohort.

Those statements require the separately registered object/session experiment and
its exact source, calibration, target-access, scoring, fallback, and reporting
artifacts.
