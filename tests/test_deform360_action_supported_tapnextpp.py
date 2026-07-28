from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_action_supported_tapnextpp import (
    build_action_supported_query_schedule,
    query_schedule_from_artifacts,
    validate_action_supported_provider_artifacts,
    validate_action_supported_query_artifacts,
    write_action_supported_provider_artifacts,
    write_action_supported_query_artifacts,
)
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY,
    DynamicMultiviewConfig,
    DynamicMultiviewResult,
    conservative_triangulation_covariance_m2,
)
from bayesian_phystwin.tapnextpp_dynamic_runtime import (
    DynamicTAPNextPPRuntimeResult,
)


def _canonical(payload: dict[str, object], key: str) -> str:
    value = dict(payload)
    value.pop(key, None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _v10_carrier(
    *,
    node_count: int = 12,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    camera_count = 4
    query = np.zeros((camera_count, node_count, 2), dtype=np.float64)
    for camera in range(camera_count):
        query[camera, :, 0] = np.arange(node_count) + camera
        query[camera, :, 1] = np.arange(node_count) + 2 * camera
    valid = np.ones((camera_count, node_count), dtype=bool)
    probability = np.full((camera_count, node_count), 0.8)
    arrays = {
        "candidate_entity_ids": np.arange(node_count, dtype=np.int64),
        "candidate_support_count": np.full(node_count, camera_count),
        "association_query_points_xy": query,
        "association_valid": valid,
        "association_probability": probability,
        "association_entropy": np.full_like(probability, 0.1),
        "association_candidate_count": np.ones_like(
            probability,
            dtype=np.int64,
        ),
        "association_covariance_px2": np.repeat(
            np.eye(2)[None, None],
            camera_count * node_count,
            axis=0,
        ).reshape(camera_count, node_count, 2, 2),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ActiveQueryFeasibilityAudit",
        "protocol_id": "deform360-active-query-feasibility-v10-source",
        "case": "synthetic-source",
        "status": "admitted",
        "repository_revision": "1" * 40,
        "audit": {
            "camera_panel": {
                "camera_indices": [0, 1, 2, 3],
                "camera_names": ["cam0", "cam1", "cam2", "cam3"],
                "frame_zero_coverage": [1.0, 1.0, 1.0, 1.0],
                "selection_scores": [1.0, 0.9, 0.8, 0.7],
            }
        },
        "inputs_sha256": {},
        "archive": {
            "filename": "active_query_feasibility.npz",
            "file_sha256": "2" * 64,
            "array_sha256": {},
        },
        "information_boundary": {},
    }
    report["result_sha256"] = _canonical(report, "result_sha256")
    return report, arrays


def _physical_inputs(
    node_count: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_zero = np.column_stack(
        (
            np.linspace(0.0, 0.11, node_count),
            np.linspace(0.0, 0.05, node_count),
            np.ones(node_count),
        )
    )
    basis = np.zeros((3 * node_count, 8), dtype=np.float64)
    for rank in range(8):
        basis[3 * rank : 3 * rank + 3, rank] = 1.0
    support = np.full(node_count, 0.2)
    support[-2:] = 0.05
    return frame_zero, basis, support


def test_action_supported_queries_ignore_predicted_displacement() -> None:
    report, arrays = _v10_carrier()
    frame_zero, basis, support = _physical_inputs()
    schedule = build_action_supported_query_schedule(
        report,
        arrays,
        frame_zero,
        basis,
        support,
    )

    assert schedule.admitted
    assert len(schedule.plan.node_ids) == 8
    assert np.all(schedule.selected_action_support >= 0.1)
    assert np.all(schedule.plan.motion_score == 0.0)
    assert (
        schedule.descriptor()["information_boundary"][
            "predicted_displacement_used"
        ]
        is False
    )


def test_action_support_is_not_folded_into_association_probability() -> None:
    report, arrays = _v10_carrier()
    frame_zero, basis, support = _physical_inputs()
    expected = arrays["association_probability"].copy()
    schedule = build_action_supported_query_schedule(
        report,
        arrays,
        frame_zero,
        basis,
        support,
    )

    assert np.array_equal(
        schedule.association_probability,
        expected[:, schedule.plan.node_ids],
    )


def test_query_artifact_round_trip(tmp_path: Path) -> None:
    report, arrays = _v10_carrier()
    frame_zero, basis, support = _physical_inputs()
    schedule = build_action_supported_query_schedule(
        report,
        arrays,
        frame_zero,
        basis,
        support,
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    physical_manifest = tmp_path / "physical.json"
    physical_manifest.write_text("{}\n")
    physical_archive = tmp_path / "physical.npz"
    physical_archive.write_bytes(b"physical")
    v10 = tmp_path / "v10"
    v10.mkdir()
    (v10 / "active_query_feasibility.json").write_text(
        json.dumps(report, sort_keys=True)
    )
    output = tmp_path / "query"

    written = write_action_supported_query_artifacts(
        output,
        schedule,
        case_id="synthetic-source",
        repository_revision="1" * 40,
        protocol_path=protocol,
        v10_output_dir=v10,
        physical_manifest_path=physical_manifest,
        physical_archive_path=physical_archive,
    )
    validated, stored = validate_action_supported_query_artifacts(output)
    restored = query_schedule_from_artifacts(output)

    assert written["result_sha256"] == validated["result_sha256"]
    assert np.array_equal(stored["entity_ids"], schedule.plan.node_ids)
    assert restored.artifact_sha256 == schedule.artifact_sha256
    assert np.array_equal(restored.plan.node_ids, schedule.plan.node_ids)


def test_provider_artifact_round_trip(tmp_path: Path) -> None:
    report, arrays = _v10_carrier()
    frame_zero, basis, support = _physical_inputs()
    schedule = build_action_supported_query_schedule(
        report,
        arrays,
        frame_zero,
        basis,
        support,
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n")
    physical_manifest = tmp_path / "physical.json"
    physical_manifest.write_text("{}\n")
    physical_archive = tmp_path / "physical.npz"
    physical_archive.write_bytes(b"physical")
    v10 = tmp_path / "v10"
    v10.mkdir()
    (v10 / "active_query_feasibility.json").write_text(
        json.dumps(report, sort_keys=True)
    )
    query_output = tmp_path / "query"
    write_action_supported_query_artifacts(
        query_output,
        schedule,
        case_id="synthetic-source",
        repository_revision="1" * 40,
        protocol_path=protocol,
        v10_output_dir=v10,
        physical_manifest_path=physical_manifest,
        physical_archive_path=physical_archive,
    )
    camera_count = 4
    frame_count = 58
    query_count = 8
    tracks = np.zeros((camera_count, frame_count, query_count, 2))
    visibility = np.full(tracks.shape[:-1], 0.8)
    active = np.ones_like(visibility, dtype=bool)
    runtime = DynamicTAPNextPPRuntimeResult(
        tracks_xy=tracks,
        visibility_probability=visibility,
        active=active,
        rollout_count=camera_count,
        model_frame_count=camera_count * frame_count,
        elapsed_seconds=1.0,
    )
    scalar = (frame_count, query_count)
    covariance = np.repeat(
        (np.eye(3) * 1e-4)[None, None],
        frame_count * query_count,
        axis=0,
    ).reshape(*scalar, 3, 3)
    inliers = np.zeros((camera_count, *scalar), dtype=bool)
    inliers[:2] = True
    provider = DynamicMultiviewResult(
        trajectory_world_m=np.repeat(
            schedule.query_points_world_m[None],
            frame_count,
            axis=0,
        ),
        proposal_available=np.ones(scalar, dtype=bool),
        accepted_support=np.ones(scalar, dtype=bool),
        prior_reliability=np.full(scalar, 0.5),
        association_probability=np.full(scalar, 0.8),
        local_covariance_m2=covariance,
        naive_independent_covariance_m2=covariance * 0.5,
        assignment_mixture_spread_m2=np.zeros_like(covariance),
        independent_support_count=np.full(scalar, 2),
        raw_support_count=np.full(scalar, 2),
        reprojection_rmse_px=np.ones(scalar),
        depth_residual_rmse_m=np.full(scalar, 0.001),
        inlier_camera_mask=inliers,
        camera_cluster_ids=np.arange(camera_count),
        shared_bias_standard_deviation_m=0.005,
        config=DynamicMultiviewConfig(
            minimum_claim_view_count=2,
            two_view_covariance_inflation=4.0,
            assignment_uncertainty_mode=(
                COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY
            ),
        ),
    )
    output = tmp_path / "provider"
    written = write_action_supported_provider_artifacts(
        output,
        schedule,
        runtime,
        provider,
        case_id="synthetic-source",
        repository_revision="1" * 40,
        protocol_path=protocol,
        query_output_dir=query_output,
        runtime_provenance={"device": "test"},
        causal_input_sha256={"rgb": "2" * 64},
    )
    validated, stored = validate_action_supported_provider_artifacts(output)

    assert written["result_sha256"] == validated["result_sha256"]
    assert stored["trajectory_world_m"].shape == (58, 8, 3)
    assert np.all(stored["accepted_support"])


def test_two_view_claim_requires_fourfold_inflation() -> None:
    with pytest.raises(ValueError, match="fourfold"):
        DynamicMultiviewConfig(
            minimum_claim_view_count=2,
            two_view_covariance_inflation=2.0,
        )


def test_two_view_unknown_correlation_is_more_conservative() -> None:
    point = np.asarray([0.0, 0.0, 1.0])
    intrinsics = np.asarray(
        [
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        ]
    )
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 0.2
    visibility = np.asarray([0.9, 0.9])
    assignment = np.repeat(np.eye(2)[None] * 0.25, 2, axis=0)
    conservative, naive, _ = conservative_triangulation_covariance_m2(
        point,
        intrinsics,
        poses,
        visibility,
        assignment,
        config=DynamicMultiviewConfig(
            minimum_claim_view_count=2,
            two_view_covariance_inflation=4.0,
        ),
    )

    assert np.all(
        np.linalg.eigvalsh(conservative - naive) >= -1e-12
    )
    assert np.trace(conservative) > np.trace(naive)
