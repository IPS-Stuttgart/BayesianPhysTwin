from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.covariance_only_value import (
    CovarianceOnlyValueCertificateV1,
)
from bayesian_phystwin.physical_query_v1 import (
    COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
    MARGINAL_GAUGE_COVARIANCE,
    PhysicalQueryBootstrapV1,
    PhysicalQueryDecisionMarginsV1,
    PhysicalQueryV1,
)
from bayesian_phystwin.probabilistic_scoring import GAUSSIAN_NLL_PER_DIMENSION
from bayesian_phystwin.query_covariance_decision_v1 import (
    PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY,
    PROB4D_QUERY_COVARIANCE_SCHEMA,
    PROB4D_QUERY_COVARIANCE_VERSION,
    TRACE_RELEVANCE_DIAGNOSTIC,
    TRACE_RELEVANCE_SELECTION_RULE,
    compose_query_covariance_treatment,
    load_query_covariance_treatment_decision,
    write_query_covariance_treatment_decision,
)
from bayesian_phystwin.repository_provenance import RepositoryState


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _query(
    *,
    principal: str = COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
    practical_equivalence: float = 0.3,
) -> PhysicalQueryV1:
    return PhysicalQueryV1(
        query_name="fresh-provider-endpoint-displacement",
        dimension=2,
        component_order=("early-displacement", "late-displacement"),
        physical_unit="m",
        coordinate_frame="registered-world-frame",
        horizon_values=(0.08, 0.20),
        horizon_unit="s",
        jacobian_provider_id=_sha256("jacobian-provider"),
        baseline_physical_belief_id=_sha256("physical-belief"),
        exact_fallback_id=_sha256("fallback-bytes"),
        covariance_treatments=(
            MARGINAL_GAUGE_COVARIANCE,
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
        ),
        principal_covariance_treatment=principal,
        primary_proper_score=GAUSSIAN_NLL_PER_DIMENSION,
        decision_margins=PhysicalQueryDecisionMarginsV1(
            practical_equivalence_score=practical_equivalence,
            maximum_harmful_score_increase=0.05,
            minimum_accepted_coverage=0.85,
            maximum_mean_width=0.3,
            maximum_worst_group_score_regret=0.1,
            minimum_shared_covariance_relevance=0.05,
            width_unit="m",
        ),
        shared_covariance_diagnostic=TRACE_RELEVANCE_DIAGNOSTIC,
        computational_selection_rule=TRACE_RELEVANCE_SELECTION_RULE,
        bootstrap=PhysicalQueryBootstrapV1(
            independent_group_definition="complete physical object/session",
            method="paired-stratified-group-bootstrap",
            resamples=10_000,
            seed=1731,
            confidence_level=0.95,
            stratification_keys=("object-stratum",),
        ),
        package_artifact_ids={
            "bayesian-phystwin": _sha256("bpt-wheel"),
            "prob4d": _sha256("prob4d-wheel"),
            "causal4d": _sha256("causal4d-wheel"),
        },
        provider_manifest_id=_sha256("provider-manifest"),
        evidence_decision_ids={
            "source-provider-gate": _sha256("source-decision"),
        },
        repositories=(
            RepositoryState(
                repository="IPS-Stuttgart/BayesianPhysTwin",
                revision="a" * 40,
                dirty=False,
                role="primary",
            ),
            RepositoryState(
                repository="IPS-Stuttgart/Prob4D",
                revision="b" * 40,
                dirty=False,
                role="observation",
            ),
            RepositoryState(
                repository="IPS-Stuttgart/Causal4D",
                revision="c" * 40,
                dirty=False,
                role="downstream",
            ),
        ),
        metadata={"target_opened": False},
    )


def _certificate(
    query: PhysicalQueryV1,
    *,
    candidate_score: float = 0.0,
) -> CovarianceOnlyValueCertificateV1:
    count = 128
    groups = tuple(f"object-session-{index:03d}" for index in range(count))
    digests = tuple(_sha256(group) for group in groups)
    return CovarianceOnlyValueCertificateV1(
        candidate_policy_id=_sha256("covariance-candidate"),
        reference_policy_id=_sha256("last-residual-reference"),
        query_set_id=query.query_id,
        policy_freeze_artifact_id=query.query_id,
        certification_partition_id=_sha256("certification-partition"),
        statistical_unit=query.bootstrap.independent_group_definition,
        score_metric=query.primary_proper_score,
        width_metric="mean-full-width-m",
        selection_group_ids=("selection-a", "selection-b"),
        group_ids=groups,
        candidate_mean_sha256=digests,
        reference_mean_sha256=digests,
        candidate_scores=np.full(count, candidate_score, dtype=np.float64),
        reference_scores=np.zeros(count, dtype=np.float64),
        candidate_full_widths=np.full(count, 0.1, dtype=np.float64),
        reference_full_widths=np.full(count, 0.05, dtype=np.float64),
        score_difference_lower_bound=-1.0,
        score_difference_upper_bound=1.0,
        full_width_upper_bound=1.0,
        maximum_expected_score_regret=(
            query.decision_margins.practical_equivalence_score
        ),
        maximum_expected_full_width=query.decision_margins.maximum_mean_width,
        harm_margin=query.decision_margins.maximum_harmful_score_increase,
        target_harm_probability=0.05,
        familywise_confidence_level=query.bootstrap.confidence_level,
        minimum_group_count=100,
        thresholds_frozen_before_certification_outcomes=True,
        certification_outcomes_used_for_policy_selection=False,
        certification_groups_independent=True,
    )


def _summary(shared_fraction: float | None = 0.2) -> dict[str, object]:
    if shared_fraction is None:
        conditional_trace = 0.0
        shared_trace = 0.0
        total_trace = 0.0
        active_dimension = 0
        total_rank = 0
        shared_rank = 0
        shared_columns = 0
        coordinate_fractions: list[float | None] = [None, None]
        minimum = mean = maximum = None
        frobenius = None
    else:
        total_trace = 1.0
        shared_trace = shared_fraction
        conditional_trace = total_trace - shared_trace
        active_dimension = 2
        total_rank = 2
        shared_rank = 1
        shared_columns = 1
        coordinate_fractions = [shared_fraction, shared_fraction]
        minimum = max(shared_fraction / 2.0, 0.0)
        mean = shared_fraction
        maximum = min(shared_fraction * 1.5, 1.0)
        frobenius = shared_fraction
    return {
        "schema": PROB4D_QUERY_COVARIANCE_SCHEMA,
        "version": PROB4D_QUERY_COVARIANCE_VERSION,
        "observation_count": 20,
        "query_dimension": 2,
        "shared_rank_column_count": shared_columns,
        "total_effective_rank": total_rank,
        "shared_effective_rank": shared_rank,
        "active_query_dimension": active_dimension,
        "conditional_trace": conditional_trace,
        "shared_trace": shared_trace,
        "total_trace": total_trace,
        "shared_trace_fraction": shared_fraction,
        "shared_frobenius_fraction": frobenius,
        "coordinate_shared_fractions": coordinate_fractions,
        "minimum_directional_shared_fraction": minimum,
        "mean_directional_shared_fraction": mean,
        "maximum_directional_shared_fraction": maximum,
        "relative_rank_tolerance": 1e-10,
        "claim_boundary": PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY,
    }


def test_composition_authorizes_frozen_explicit_joint_treatment() -> None:
    query = _query()
    certificate = _certificate(query)

    decision = compose_query_covariance_treatment(
        query,
        _summary(0.2),
        certificate,
        source_observation_artifact_id=_sha256("observation"),
    )

    assert certificate.certified
    assert decision.authorized
    assert decision.selected_covariance_treatment == (
        COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE
    )
    assert decision.reasons == ("covariance-treatment-authorized",)
    assert decision.exact_fallback_id == query.exact_fallback_id


def test_low_relevance_selects_marginal_treatment() -> None:
    query = _query(principal=MARGINAL_GAUGE_COVARIANCE)
    decision = compose_query_covariance_treatment(
        query,
        _summary(0.01),
        _certificate(query),
        source_observation_artifact_id=_sha256("observation"),
    )

    assert decision.authorized
    assert decision.selected_covariance_treatment == MARGINAL_GAUGE_COVARIANCE


def test_principal_treatment_mismatch_fails_closed() -> None:
    query = _query()
    decision = compose_query_covariance_treatment(
        query,
        _summary(0.01),
        _certificate(query),
        source_observation_artifact_id=_sha256("observation"),
    )

    assert not decision.authorized
    assert decision.selected_covariance_treatment == MARGINAL_GAUGE_COVARIANCE
    assert decision.reasons == ("principal-covariance-treatment-mismatch",)
    assert decision.exact_fallback_id == query.exact_fallback_id


def test_undefined_relevance_fails_closed() -> None:
    query = _query()
    decision = compose_query_covariance_treatment(
        query,
        _summary(None),
        _certificate(query),
        source_observation_artifact_id=_sha256("observation"),
    )

    assert not decision.authorized
    assert decision.selected_covariance_treatment is None
    assert decision.reasons == (
        "principal-covariance-treatment-mismatch",
        "shared-covariance-relevance-undefined",
    )


def test_rejected_value_certificate_fails_closed() -> None:
    query = _query()
    certificate = _certificate(query, candidate_score=0.5)
    decision = compose_query_covariance_treatment(
        query,
        _summary(0.2),
        certificate,
        source_observation_artifact_id=_sha256("observation"),
    )

    assert not certificate.certified
    assert not decision.authorized
    assert decision.reasons == ("covariance-value-certificate-rejected",)


def test_structural_binding_mismatch_is_rejected() -> None:
    query = _query()
    certificate = replace(
        _certificate(query),
        policy_freeze_artifact_id=_sha256("different-freeze"),
        artifact_id=None,
    )

    with pytest.raises(ValueError, match="not frozen by PhysicalQueryV1"):
        compose_query_covariance_treatment(
            query,
            _summary(0.2),
            certificate,
            source_observation_artifact_id=_sha256("observation"),
        )


def test_prob4d_summary_tampering_is_rejected() -> None:
    query = _query()
    summary = _summary(0.2)
    summary["shared_trace_fraction"] = 0.8

    with pytest.raises(ValueError, match="disagrees with query traces"):
        compose_query_covariance_treatment(
            query,
            summary,
            _certificate(query),
            source_observation_artifact_id=_sha256("observation"),
        )


def test_decision_roundtrip_and_no_clobber(tmp_path: Path) -> None:
    query = _query()
    decision = compose_query_covariance_treatment(
        query,
        _summary(0.2),
        _certificate(query),
        source_observation_artifact_id=_sha256("observation"),
    )
    path = tmp_path / "query-covariance-decision.json"

    write_query_covariance_treatment_decision(decision, path)
    loaded = load_query_covariance_treatment_decision(path)

    assert loaded.to_record() == decision.to_record()
    with pytest.raises(FileExistsError):
        write_query_covariance_treatment_decision(decision, path)


def test_content_identity_binds_source_observation() -> None:
    query = _query()
    certificate = _certificate(query)
    first = compose_query_covariance_treatment(
        query,
        _summary(0.2),
        certificate,
        source_observation_artifact_id=_sha256("observation-a"),
    )
    second = compose_query_covariance_treatment(
        query,
        _summary(0.2),
        certificate,
        source_observation_artifact_id=_sha256("observation-b"),
    )

    assert first.projection_summary_id != second.projection_summary_id
    assert first.artifact_id != second.artifact_id


def test_content_addressed_boolean_tampering_is_rejected() -> None:
    query = _query()
    decision = compose_query_covariance_treatment(
        query,
        _summary(0.2),
        _certificate(query),
        source_observation_artifact_id=_sha256("observation"),
    )

    with pytest.raises(ValueError, match="authorized"):
        replace(decision, authorized=False)
