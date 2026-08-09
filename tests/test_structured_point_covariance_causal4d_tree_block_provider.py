from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_structured_point_covariance_tree_block_operator import _covariance

from bayesian_phystwin.causal4d_tree_block_provider_v1 import (
    CAUSAL4D_TREE_BLOCK_PROVIDER_API_VERSION,
    CAUSAL4D_TREE_BLOCK_PROVIDER_ARTIFACT_SCHEMA_VERSIONS,
    CAUSAL4D_TREE_BLOCK_PROVIDER_BOUNDARY,
    CAUSAL4D_TREE_BLOCK_PROVIDER_CAPABILITIES,
    CAUSAL4D_TREE_BLOCK_PROVIDER_COMPATIBILITY,
    CAUSAL4D_TREE_BLOCK_PROVIDER_INFERENCE_ROLE,
    CAUSAL4D_TREE_BLOCK_PROVIDER_RAW_COVARIANCE_CLAIM,
    CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA,
    CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_VERSION,
    causal4d_tree_block_provider_manifest,
    evaluate_claim_bearing_tree_block_query,
)
from bayesian_phystwin.tree_block_sparse_gauge_belief import (
    TreeBlockGaugeAwareBeliefResultV1,
    TreeBlockPosteriorCovarianceV1,
)
from bayesian_phystwin.tree_block_sparse_prob4d import (
    ClaimBearingTreeBlockProb4DUpdateV1,
)

_IDS = tuple(character * 64 for character in "abcdef")


def _lineage() -> dict[str, object]:
    return {
        "observation_artifact_id": _IDS[0],
        "linearization_artifact_id": _IDS[1],
        "prob4d_claim_bearing_provider_manifest_id": _IDS[2],
        "prob4d_claim_bearing_calibration_artifact_ids": {
            "gauge_artifact_id": _IDS[3],
            "point_artifact_id": _IDS[4],
        },
        "prob4d_claim_bearing_runtime_revision_source": "source_checkout",
        "prob4d_claim_bearing_runtime_revision_independently_verified": True,
    }


def _update(*, accepted: bool = True) -> ClaimBearingTreeBlockProb4DUpdateV1:
    covariance = _covariance()
    retained = covariance.retained_state_count
    result = TreeBlockGaugeAwareBeliefResultV1(
        inference_admissible=accepted,
        reason="inference-admissible" if accepted else "strict-rejection",
        state_coefficients=np.zeros(covariance.state_count),
        gauge_delta=np.zeros(covariance.gauge_parameter_count),
        shared_bias_coefficients=np.zeros(1),
        view_bias_coefficients=np.zeros(1),
        anchor_bias_coefficients=np.zeros(0),
        covariance=covariance,
        identifiable_state_transform=covariance.state_mapping,
        identifiable_fractions=np.ones(retained),
        query_sensitivity_fractions=np.ones(retained),
        robust_weights=np.ones(1),
        anchor_robust_weights=np.zeros(0),
        diagnostics={
            "implementation_id": "tree-block-group-mixture-strict-admission-v2",
            "strict_admission_version": 2,
            "strict_admission_passed": accepted,
        },
        input_lineage=_lineage(),
    )
    return ClaimBearingTreeBlockProb4DUpdateV1(
        result=result,
        observation_artifact_id=_IDS[0],
        linearization_artifact_id=_IDS[1],
        provider_manifest_id=_IDS[2],
        calibration_artifact_ids={
            "gauge_artifact_id": _IDS[3],
            "point_artifact_id": _IDS[4],
        },
        runtime_revision_source="source_checkout",
        runtime_revision_independently_verified=True,
    )


def test_provider_manifest_binds_additive_query_contract() -> None:
    manifest = causal4d_tree_block_provider_manifest(provider_revision="revision")

    assert manifest["provider_name"] == "bayesian-phystwin"
    assert manifest["provider_revision"] == "revision"
    assert manifest["schema_version"] == CAUSAL4D_TREE_BLOCK_PROVIDER_API_VERSION
    assert tuple(manifest["capabilities"]) == CAUSAL4D_TREE_BLOCK_PROVIDER_CAPABILITIES
    assert manifest["artifact_schema_versions"] == (
        CAUSAL4D_TREE_BLOCK_PROVIDER_ARTIFACT_SCHEMA_VERSIONS
    )
    metadata = manifest["metadata"]
    assert metadata["provider_api"] == (
        "bayesian_phystwin.causal4d_tree_block_provider_v1"
    )
    assert metadata["inference_role"] == CAUSAL4D_TREE_BLOCK_PROVIDER_INFERENCE_ROLE
    assert metadata["compatibility"] == CAUSAL4D_TREE_BLOCK_PROVIDER_COMPATIBILITY
    assert metadata["raw_covariance_claim"] == (
        CAUSAL4D_TREE_BLOCK_PROVIDER_RAW_COVARIANCE_CLAIM
    )
    assert metadata["claim_boundary"] == CAUSAL4D_TREE_BLOCK_PROVIDER_BOUNDARY


def test_query_matches_dense_reference_and_binds_lineage() -> None:
    update = _update()
    covariance = update.result.covariance
    query = np.random.default_rng(41).normal(size=(4, covariance.dimension))

    result = evaluate_claim_bearing_tree_block_query(
        update,
        query,
        query_id=_IDS[5],
    )

    np.testing.assert_allclose(
        result.covariance,
        query @ covariance.materialize() @ query.T,
        rtol=3e-11,
        atol=3e-12,
    )
    assert result.schema == CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_SCHEMA
    assert result.schema_version == CAUSAL4D_TREE_BLOCK_QUERY_COVARIANCE_VERSION
    assert result.update_id == update.update_id
    assert result.tree_block_result_id == update.tree_block_result_id
    assert result.query_id == _IDS[5]
    assert result.coefficient_dimension == covariance.dimension
    assert result.query_row_count == len(query)
    assert result.inference_admissible
    assert len(result.query_matrix_sha256) == 64
    assert len(result.result_id) == 64
    assert not result.covariance.flags.writeable
    with pytest.raises(ValueError):
        result.covariance.setflags(write=True)


def test_rejected_update_exposes_its_exact_fallback_covariance() -> None:
    update = _update(accepted=False)
    query = np.eye(update.result.covariance.dimension)[:3]

    result = evaluate_claim_bearing_tree_block_query(
        update,
        query,
        query_id=_IDS[5],
    )

    np.testing.assert_allclose(
        result.covariance,
        query @ update.result.covariance.materialize() @ query.T,
        rtol=3e-11,
        atol=3e-12,
    )
    assert not result.inference_admissible
    assert result.inference_reason == "strict-rejection"


def test_provider_never_calls_complete_covariance_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    update = _update()
    query = np.eye(update.result.covariance.dimension)[:2]

    def fail_materialization(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("dense covariance path used")

    monkeypatch.setattr(
        TreeBlockPosteriorCovarianceV1, "materialize", fail_materialization
    )
    result = evaluate_claim_bearing_tree_block_query(
        update,
        query,
        query_id=_IDS[5],
    )
    assert result.covariance.shape == (2, 2)


@pytest.mark.parametrize(
    "query",
    [
        np.asarray([[True]]),
        np.asarray([[1.0 + 1.0j]]),
        np.zeros(3),
    ],
)
def test_provider_rejects_non_real_or_non_matrix_queries(query: object) -> None:
    with pytest.raises(ValueError):
        evaluate_claim_bearing_tree_block_query(
            _update(),
            query,
            query_id=_IDS[5],
        )


def test_provider_rejects_wrong_shape_empty_nonfinite_and_bad_identity() -> None:
    update = _update()
    dimension = update.result.covariance.dimension
    with pytest.raises(ValueError, match="coefficient dimension"):
        evaluate_claim_bearing_tree_block_query(
            update,
            np.zeros((1, dimension - 1)),
            query_id=_IDS[5],
        )
    with pytest.raises(ValueError, match="at least one row"):
        evaluate_claim_bearing_tree_block_query(
            update,
            np.zeros((0, dimension)),
            query_id=_IDS[5],
        )
    nonfinite = np.zeros((1, dimension))
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        evaluate_claim_bearing_tree_block_query(
            update,
            nonfinite,
            query_id=_IDS[5],
        )
    with pytest.raises(ValueError, match="query_id"):
        evaluate_claim_bearing_tree_block_query(
            update,
            np.zeros((1, dimension)),
            query_id="short",
        )
    with pytest.raises(TypeError, match="ClaimBearingTreeBlockProb4DUpdateV1"):
        evaluate_claim_bearing_tree_block_query(
            object(),  # type: ignore[arg-type]
            np.zeros((1, dimension)),
            query_id=_IDS[5],
        )


def test_query_result_rejects_invalid_covariance_fields() -> None:
    update = _update()
    valid = evaluate_claim_bearing_tree_block_query(
        update,
        np.zeros((1, update.result.covariance.dimension)),
        query_id=_IDS[5],
    )

    with pytest.raises(ValueError, match="coefficient_dimension"):
        replace(valid, coefficient_dimension=0)
    with pytest.raises(ValueError, match="nonempty square"):
        replace(valid, covariance=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="symmetric"):
        replace(valid, covariance=np.asarray([[1.0, 1.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="positive semidefinite"):
        replace(valid, covariance=np.asarray([[-1.0]]))
    with pytest.raises(ValueError, match="finite"):
        replace(valid, covariance=np.asarray([[np.nan]]))


def test_provider_manifest_rejects_invalid_revision() -> None:
    with pytest.raises(ValueError, match="provider_revision"):
        causal4d_tree_block_provider_manifest(provider_revision="")
