from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_causal_response_tracker import (
    CausalResponseTrackerPrediction,
    CrossPanelProviderConfig,
    birth_associations_from_adaptive_query,
    corroborate_disjoint_panels,
    validate_causal_response_tracker_artifacts,
    write_causal_response_tracker_artifacts,
)
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY,
    DynamicMultiviewConfig,
    DynamicMultiviewResult,
)
from bayesian_phystwin.tapnextpp_dynamic_runtime import (
    DynamicTAPNextPPRuntimeResult,
)


def _query() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    cameras = [f"cam{index}" for index in range(8)]
    query_count = 16
    arrays = {
        "entity_ids": np.arange(query_count, dtype=np.int64),
        "query_points_world_m": np.column_stack(
            (
                np.linspace(0.0, 0.15, query_count),
                np.zeros(query_count),
                np.ones(query_count),
            )
        ),
        "association_query_points_xy": np.zeros(
            (8, query_count, 2),
            dtype=np.float64,
        ),
        "association_valid": np.ones((8, query_count), dtype=bool),
        "association_probability": np.full((8, query_count), 0.8),
        "association_entropy": np.full((8, query_count), 0.1),
        "association_candidate_count": np.ones(
            (8, query_count),
            dtype=np.int64,
        ),
        "association_covariance_px2": np.repeat(
            np.eye(2)[None, None],
            8 * query_count,
            axis=0,
        ).reshape(8, query_count, 2, 2),
        "selected_complete_camera_indices": np.arange(
            8,
            dtype=np.int64,
        ),
    }
    report: dict[str, object] = {
        "artifact_kind": "Deform360CausalResponseAdaptiveQueryV13",
        "case": "synthetic-source",
        "status": "admitted",
        "result_sha256": "1" * 64,
        "schedule": {
            "admitted": True,
            "query_schedule": {
                "admitted": True,
                "camera_ids": cameras,
            },
        },
    }
    return report, arrays


def _panel(
    *,
    offset_m: float = 0.0,
    accepted: bool = True,
) -> DynamicMultiviewResult:
    frame_count = 58
    query_count = 16
    trajectory = np.zeros((frame_count, query_count, 3))
    trajectory[..., 0] = offset_m
    scalar = (frame_count, query_count)
    support = np.full(scalar, accepted, dtype=bool)
    covariance = np.repeat(
        (np.eye(3) * 1e-5)[None, None],
        frame_count * query_count,
        axis=0,
    ).reshape(frame_count, query_count, 3, 3)
    return DynamicMultiviewResult(
        trajectory_world_m=trajectory,
        proposal_available=support,
        accepted_support=support,
        prior_reliability=np.where(support, 0.8, 0.0),
        association_probability=np.where(support, 0.9, 0.0),
        local_covariance_m2=covariance,
        naive_independent_covariance_m2=covariance * 0.5,
        assignment_mixture_spread_m2=np.zeros_like(covariance),
        independent_support_count=np.where(support, 3, 0),
        raw_support_count=np.where(support, 3, 0),
        reprojection_rmse_px=np.where(support, 1.0, np.nan),
        depth_residual_rmse_m=np.where(support, 0.001, np.nan),
        inlier_camera_mask=np.repeat(
            support[None],
            4,
            axis=0,
        ),
        camera_cluster_ids=np.arange(4, dtype=np.int64),
        shared_bias_standard_deviation_m=0.005,
        config=DynamicMultiviewConfig(
            minimum_claim_view_count=3,
            assignment_uncertainty_mode=(
                COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY
            ),
        ),
    )


def _runtime() -> DynamicTAPNextPPRuntimeResult:
    tracks = np.zeros((8, 58, 16, 2), dtype=np.float64)
    return DynamicTAPNextPPRuntimeResult(
        tracks_xy=tracks,
        visibility_probability=np.full((8, 58, 16), 0.8),
        active=np.ones((8, 58, 16), dtype=bool),
        rollout_count=8,
        model_frame_count=8 * 58,
        elapsed_seconds=1.0,
    )


def test_adaptive_query_converts_without_recomputing_association() -> None:
    report, arrays = _query()
    associations = birth_associations_from_adaptive_query(report, arrays)

    assert associations.camera_names == tuple(f"cam{i}" for i in range(8))
    assert np.array_equal(
        associations.query_points_xy,
        arrays["association_query_points_xy"],
    )
    assert np.array_equal(
        associations.candidate_pixel_covariance_px2,
        arrays["association_covariance_px2"],
    )


def test_cross_panel_corroboration_adds_disagreement_uncertainty() -> None:
    proposal = _panel()
    validation = _panel(offset_m=0.003)
    result = corroborate_disjoint_panels(proposal, validation)

    assert np.all(result.accepted_support)
    assert np.all(result.panel_disagreement_m == pytest.approx(0.003))
    assert np.all(
        np.linalg.eigvalsh(
            result.local_covariance_m2
            - proposal.local_covariance_m2
        )
        >= -1e-12
    )
    assert np.all(result.prior_reliability < proposal.prior_reliability)


def test_cross_panel_disagreement_forces_exact_abstention() -> None:
    result = corroborate_disjoint_panels(
        _panel(),
        _panel(offset_m=0.020),
        config=CrossPanelProviderConfig(
            maximum_disagreement_m=0.015,
            disagreement_scale_m=0.005,
        ),
    )

    assert not np.any(result.accepted_support)
    assert np.array_equal(
        result.prior_reliability,
        np.zeros_like(result.prior_reliability),
    )


def test_tracker_artifact_round_trip_and_tamper_detection(
    tmp_path: Path,
) -> None:
    query_report, query_arrays = _query()
    query_dir = tmp_path / "query"
    query_dir.mkdir()
    (query_dir / "causal_response_adaptive_query_v13.json").write_text(
        json.dumps(query_report),
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    proposal = _panel()
    validation = _panel(offset_m=0.003)
    prediction = corroborate_disjoint_panels(proposal, validation)
    output = tmp_path / "provider"

    written = write_causal_response_tracker_artifacts(
        output,
        query_report,
        query_arrays,
        _runtime(),
        proposal,
        validation,
        prediction,
        case_id="synthetic-source",
        repository_revision="2" * 40,
        protocol_path=protocol,
        query_output_dir=query_dir,
        runtime_provenance={"device": "synthetic"},
        causal_input_sha256={"prefix": "3" * 64},
        update_frame=57,
    )
    validated, arrays = validate_causal_response_tracker_artifacts(output)

    assert written["result_sha256"] == validated["result_sha256"]
    assert arrays["accepted_support"].shape == (58, 16)
    assert written["information_boundary"][
        "physical_innovation_used_for_prior_reliability"
    ] is False

    report_path = output / "causal_response_tracker_v13.json"
    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    tampered["accepted_endpoint_count"] += 1
    report_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="report is invalid"):
        validate_causal_response_tracker_artifacts(output)


def test_prediction_rejects_reliability_without_support() -> None:
    proposal = _panel()
    validation = _panel(accepted=False)
    prediction = corroborate_disjoint_panels(proposal, validation)

    with pytest.raises(ValueError, match="admission rule changed"):
        CausalResponseTrackerPrediction(
            **{
                **prediction.__dict__,
                "prior_reliability": np.ones((58, 16)),
            }
        )
