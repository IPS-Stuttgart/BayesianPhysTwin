from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.bound_query_covariance_decision_v1 import (
    BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY,
    BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA,
    BOUND_QUERY_COVARIANCE_PROJECTION_VERSION,
    compose_bound_query_covariance_treatment,
    validate_bound_query_covariance_projection,
)
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
)
from bayesian_phystwin.query_jacobian_binding_v1 import (
    QueryJacobianBindingV1,
    build_query_jacobian_binding,
)
from bayesian_phystwin.repository_provenance import RepositoryState


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding() -> QueryJacobianBindingV1:
    jacobian = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    return build_query_jacobian_binding(
        query_name="fresh-provider-endpoint-displacement",
        component_order=("early-displacement", "late-displacement"),
        physical_unit="m",
        coordinate_frame="registered-world-frame",
        source_observation_artifact_id=_sha256("observation"),
        provider_manifest_id=_sha256("provider-manifest"),
        causal_frame_stop=18,
        query_jacobian=jacobian,
        row_ids=("factor-a/point-0", "factor-b/point-4"),
    )


def _query(binding: QueryJacobianBindingV1) -> PhysicalQueryV1:
    return PhysicalQueryV1(
        query_name=binding.query_name,
        dimension=2,
        component_order=binding.component_order,
        physical_unit=binding.physical_unit,
        coordinate_frame=binding.coordinate_frame,
        horizon_values=(0.08, 0.20),
        horizon_unit="s",
        jacobian_provider_id=binding.artifact_id,
        baseline_physical_belief_id=_sha256("physical-belief"),
        exact_fallback_id=_sha256("fallback-bytes"),
        covariance_treatments=(
            MARGINAL_GAUGE_COVARIANCE,
            COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE,
        ),
        principal_covariance_treatment=(COMPLETE_EXPLICIT_JOINT_GAUGE_COVARIANCE),
        primary_proper_score=GAUSSIAN_NLL_PER_DIMENSION,
        decision_margins=PhysicalQueryDecisionMarginsV1(
            practical_equivalence_score=0.3,
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
        },
        provider_manifest_id=binding.provider_manifest_id,
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
        ),
        metadata={"target_opened": False},
    )


def _certificate(query: PhysicalQueryV1) -> CovarianceOnlyValueCertificateV1:
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
        candidate_scores=np.zeros(count, dtype=np.float64),
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


def _summary() -> dict[str, object]:
    return {
        "schema": PROB4D_QUERY_COVARIANCE_SCHEMA,
        "version": PROB4D_QUERY_COVARIANCE_VERSION,
        "observation_count": 2,
        "query_dimension": 2,
        "shared_rank_column_count": 1,
        "total_effective_rank": 2,
        "shared_effective_rank": 1,
        "active_query_dimension": 2,
        "conditional_trace": 0.8,
        "shared_trace": 0.2,
        "total_trace": 1.0,
        "shared_trace_fraction": 0.2,
        "shared_frobenius_fraction": 0.2,
        "coordinate_shared_fractions": [0.2, 0.2],
        "minimum_directional_shared_fraction": 0.1,
        "mean_directional_shared_fraction": 0.2,
        "maximum_directional_shared_fraction": 0.3,
        "relative_rank_tolerance": 1e-10,
        "claim_boundary": PROB4D_QUERY_COVARIANCE_CLAIM_BOUNDARY,
    }


def _bound_projection(binding: QueryJacobianBindingV1) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema": BOUND_QUERY_COVARIANCE_PROJECTION_SCHEMA,
        "schema_version": BOUND_QUERY_COVARIANCE_PROJECTION_VERSION,
        "query_jacobian_binding_id": binding.artifact_id,
        "source_observation_artifact_id": (binding.source_observation_artifact_id),
        "provider_manifest_id": binding.provider_manifest_id,
        "query_jacobian_sha256": binding.query_jacobian_sha256,
        "row_ids_sha256": binding.row_ids_sha256,
        "local_covariance_m2": {
            "dtype": "<f8",
            "shape": [2, 3, 3],
            "sha256": _sha256("local-covariance-bytes"),
        },
        "low_rank_factor_m": {
            "dtype": "<f8",
            "shape": [2, 3, 1],
            "sha256": _sha256("low-rank-factor-bytes"),
        },
        "projection_summary": _summary(),
        "claim_boundary": BOUND_QUERY_COVARIANCE_PROJECTION_CLAIM_BOUNDARY,
    }
    return {"artifact_id": content_id(unsigned), **unsigned}


def test_bound_composition_authorizes_and_binds_projection_artifact() -> None:
    binding = _binding()
    query = _query(binding)
    projection = _bound_projection(binding)

    decision = compose_bound_query_covariance_treatment(
        query,
        binding,
        projection,
        _certificate(query),
    )

    assert decision.authorized
    assert decision.projection_summary_id == projection["artifact_id"]
    assert decision.metadata["query_jacobian_binding_id"] == binding.artifact_id
    assert decision.metadata["query_jacobian_sha256"] == (binding.query_jacobian_sha256)
    assert decision.metadata["row_ids_sha256"] == binding.row_ids_sha256


def test_caller_metadata_cannot_override_bound_lineage() -> None:
    binding = _binding()
    query = _query(binding)

    decision = compose_bound_query_covariance_treatment(
        query,
        binding,
        _bound_projection(binding),
        _certificate(query),
        metadata={"query_jacobian_binding_id": _sha256("spoofed")},
    )

    assert decision.metadata["query_jacobian_binding_id"] == binding.artifact_id


def test_projection_binding_mismatches_fail_closed() -> None:
    binding = _binding()
    projection = _bound_projection(binding)
    projection["row_ids_sha256"] = _sha256("different-rows")
    unsigned = dict(projection)
    unsigned.pop("artifact_id")
    projection["artifact_id"] = content_id(unsigned)

    with pytest.raises(ValueError, match="row order"):
        validate_bound_query_covariance_projection(projection, binding=binding)


def test_projection_content_identity_tampering_fails_closed() -> None:
    binding = _binding()
    projection = _bound_projection(binding)
    projection["artifact_id"] = _sha256("tampered")

    with pytest.raises(ValueError, match="artifact_id"):
        validate_bound_query_covariance_projection(projection, binding=binding)


def test_physical_query_must_bind_exact_jacobian_artifact() -> None:
    binding = _binding()
    query = _query(binding)
    mismatched = replace(
        query,
        jacobian_provider_id=_sha256("different-binding"),
        query_id=None,
    )

    with pytest.raises(ValueError, match="does not bind"):
        compose_bound_query_covariance_treatment(
            mismatched,
            binding,
            _bound_projection(binding),
            _certificate(mismatched),
        )


def test_projection_summary_dimensions_must_match_binding() -> None:
    binding = _binding()
    projection = _bound_projection(binding)
    projection["projection_summary"] = {
        **_summary(),
        "observation_count": 3,
    }
    unsigned = dict(projection)
    unsigned.pop("artifact_id")
    projection["artifact_id"] = content_id(unsigned)

    with pytest.raises(ValueError, match="observation count"):
        validate_bound_query_covariance_projection(projection, binding=binding)
