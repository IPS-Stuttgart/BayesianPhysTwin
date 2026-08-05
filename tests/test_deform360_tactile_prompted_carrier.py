from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.deform360_tactile_metric_gauge import SimilarityTransform
from bayesian_phystwin.deform360_tactile_prompted_carrier import (
    DEFORM360_TACTILE_PROMPTED_CARRIER_VALIDATION_LOCK_SCHEMA,
    TACTILE_PROMPTED_CARRIER_POLICY,
    TACTILE_PROMPTED_CARRIER_VALIDATION_INFORMATION_BOUNDARY,
    TACTILE_PROMPTED_CARRIER_VALIDATION_STAGE_GATES,
    CrossViewCandidatePair,
    ProjectedPromptPair,
    PromptedCandidateGeometry,
    PromptMaskDiagnostics,
    build_bias_aware_metric_carrier,
    build_dense_point_candidates,
    build_tactile_prompt_assignments,
    evaluate_prompted_mask,
    load_tactile_prompted_carrier_validation_lock,
    object_facing_finger_normal_world,
    select_crossview_candidate_pair,
    validate_tactile_prompted_carrier_validation_lock,
)


def _validation_lock() -> dict[str, object]:
    descriptor: dict[str, object] = {
        "schema": DEFORM360_TACTILE_PROMPTED_CARRIER_VALIDATION_LOCK_SCHEMA,
        "schema_version": 1,
        "status": "locked-independent-calibration-pre-payload",
        "source_case": {
            "object_id": "036-napkin-cloth",
            "source_episode_id": 9,
            "processing_episode_index": 0,
            "bimanual": True,
            "stratum": "sheet",
            "camera_panel": [
                "brics-odroid-010_cam0",
                "brics-odroid-019_cam1",
                "brics-odroid-022_cam1",
            ],
            "causal_window": {
                "source_frame_start": 78,
                "contact_start_frame": 114,
                "causal_frame_stop": 120,
                "untouched_future_frame_start": 120,
                "untouched_future_frame_stop_exclusive": 144,
            },
            "bound_input_files_sha256": "a" * 64,
            "source_output_tree_sha256": "b" * 64,
        },
        "selection": {
            "rule": (
                "first-unopened-bimanual-calibration-object-after-development-"
                "in-stratum"
            ),
            "development_object_id": "026-sock-cloth",
            "candidate_audit": [
                {
                    "bimanual": False,
                    "metadata_sha256": (
                        "c2c5d30efc85fea22b52d1b5317c3a8084f272ef79154f313441a79c35adaa08"
                    ),
                    "object_id": "031-cotton-cloth",
                },
                {
                    "bimanual": True,
                    "metadata_sha256": (
                        "3eca16d7f72ce1d83828d60fdc98fa942843d12cd1aa20846f036a3e882547a6"
                    ),
                    "object_id": "036-napkin-cloth",
                },
            ],
        },
        "bindings": {
            "selection_lock_file_sha256": "c" * 64,
            "causal_window_manifest_file_sha256": "d" * 64,
            "causal_window_manifest_id": "e" * 64,
            "motioncrafter_job_manifest_file_sha256": "f" * 64,
            "motioncrafter_job_manifest_id": "1" * 64,
            "motioncrafter_stage1_run_report_sha256": "2" * 64,
            "motioncrafter_jobs": [
                {
                    "camera": camera,
                    "job_id": character * 64,
                    "source_frame_start": 78,
                    "source_frame_stop_exclusive": 120,
                    "video_sha256": character * 64,
                }
                for camera, character in zip(
                    [
                        "brics-odroid-010_cam0",
                        "brics-odroid-019_cam1",
                        "brics-odroid-022_cam1",
                    ],
                    "345",
                    strict=True,
                )
            ],
        },
        "implementation": {
            "revision": "6" * 40,
            "module_source_sha256": "7" * 64,
            "validation_runner_source_sha256": "8" * 64,
        },
        "stage_gates": TACTILE_PROMPTED_CARRIER_VALIDATION_STAGE_GATES,
        "information_boundary": (
            TACTILE_PROMPTED_CARRIER_VALIDATION_INFORMATION_BOUNDARY
        ),
    }
    return {"artifact_id": content_id(descriptor), **descriptor}


def test_validation_lock_is_content_addressed_and_loadable(tmp_path: Path) -> None:
    lock = _validation_lock()
    assert validate_tactile_prompted_carrier_validation_lock(lock) == lock["artifact_id"]
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    assert load_tactile_prompted_carrier_validation_lock(path) == lock


def test_validation_lock_rejects_policy_change() -> None:
    lock = _validation_lock()
    gates = dict(TACTILE_PROMPTED_CARRIER_VALIDATION_STAGE_GATES)
    gates["carrier"] = {
        **TACTILE_PROMPTED_CARRIER_POLICY,
        "shared_bias_floor_m": 0.0,
    }
    lock["stage_gates"] = gates
    descriptor = dict(lock)
    descriptor.pop("artifact_id")
    lock["artifact_id"] = content_id(descriptor)
    try:
        validate_tactile_prompted_carrier_validation_lock(lock)
    except ValueError as error:
        assert "stage gates changed" in str(error)
    else:
        raise AssertionError("changed carrier policy was accepted")


def test_object_facing_normals_point_to_opposite_gripper_sides() -> None:
    pose = np.eye(4)
    assert np.array_equal(object_facing_finger_normal_world(0, pose), [-1.0, 0.0, 0.0])
    assert np.array_equal(object_facing_finger_normal_world(1, pose), [1.0, 0.0, 0.0])


def test_tactile_prompts_preserve_assignments_and_use_latest_frame() -> None:
    tactile_frames = np.asarray([4, 5, 5, 5])
    sides = np.asarray([0, 0, 1, 0])
    values = np.asarray([100.0, 1.0, 3.0, 2.0])
    world = np.zeros((4, 2, 3))
    world[:, 0, 0] = [100.0, 0.0, 0.02, 0.04]
    world[:, 1, 0] = [100.0, 1.0, 1.02, 1.04]
    grippers = np.asarray([[0, 1], [0, 1], [0, 1], [1, 0]])
    poses = np.repeat(np.eye(4)[None, None], 4, axis=0)
    poses = np.repeat(poses, 2, axis=1)
    result = build_tactile_prompt_assignments(
        tactile_source_frame_ids=tactile_frames,
        tactile_values=values,
        finger_side_indices=sides,
        world_points_hypotheses_m=world,
        gripper_indices_hypotheses=grippers,
        robot_source_frame_ids=np.arange(3, 7),
        robot_world_from_gripper=poses,
        offset_m=0.01,
    )
    assert len(result) == 2
    assert all(item.source_frame_id == 5 for item in result)
    assert result[0].assignment_index == 0
    assert result[1].assignment_index == 1
    side_zero = result[0].finger_side_indices == 0
    assert np.all(
        result[0].positive_world_m[side_zero, 0]
        < result[0].negative_world_m[side_zero, 0]
    )
    assert np.all(
        result[0].positive_world_m[~side_zero, 0]
        > result[0].negative_world_m[~side_zero, 0]
    )
    assert not np.array_equal(result[0].positive_world_m, result[1].positive_world_m)


def test_prompt_reliability_has_no_state_residual_input() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    prompts = ProjectedPromptPair(
        positive_pixel_xy=np.asarray([[4.0, 4.0], [5.0, 5.0]]),
        negative_pixel_xy=np.asarray([[0.0, 0.0]]),
        positive_visible=np.ones(2, dtype=bool),
        negative_visible=np.ones(1, dtype=bool),
    )
    result = evaluate_prompted_mask(
        mask,
        prompts,
        predicted_iou=0.8,
        stability_score=0.9,
        minimum_positive_hits=1,
        maximum_negative_hits=0,
        minimum_area_fraction=0.1,
        maximum_area_fraction=0.5,
    )
    assert result.eligible
    assert result.positive_hits == 2
    assert result.negative_hits == 0


def _dense(offset: np.ndarray):
    height = width = 32
    yy, xx = np.mgrid[:height, :width]
    point_map = np.stack((xx, yy, np.zeros_like(xx)), axis=2).astype(float) * 0.001
    point_map += offset
    return build_dense_point_candidates(
        point_map,
        np.ones((height, width), dtype=bool),
        np.ones((height, width), dtype=bool),
        np.ones((height, width), dtype=bool),
        transform=SimilarityTransform(1.0, np.eye(3), np.zeros(3)),
        gauge_covariance_m2=np.eye(3) * 1e-4,
        block_size_px=8,
        minimum_mask_pixels=8,
        minimum_valid_fraction=0.5,
        full_reliability_deform_fraction=0.5,
        covariance_floor_m=0.005,
    )


def _candidate(index: int, offset: np.ndarray) -> PromptedCandidateGeometry:
    return PromptedCandidateGeometry(
        candidate_index=index,
        predicted_iou=0.9,
        stability_score=0.9,
        prompt=PromptMaskDiagnostics(True, 2, 2, 0, 2, 1024, 1.0, 1.2),
        dense=_dense(offset),
    )


def test_dense_resampling_does_not_create_independent_information() -> None:
    reference = _candidate(1, np.zeros(3))
    support = _candidate(2, np.asarray([0.001, 0.0, 0.0]))
    pair = CrossViewCandidatePair(
        assignment_index=0,
        reference_camera="a",
        support_camera="b",
        reference=reference,
        support=support,
        mutual_block_match_count=16,
        median_block_distance_m=0.001,
        percentile_90_block_distance_m=0.001,
    )
    carrier = build_bias_aware_metric_carrier(
        pair,
        node_count=128,
        maximum_distance_m=0.01,
        shared_bias_floor_m=0.005,
        unsupported_node_floor_m=0.015,
        unsupported_reliability_scale=0.5,
    )
    assert carrier.points_world_m.shape == (128, 3)
    assert len(np.unique(carrier.information_cluster_id)) <= 16
    assert carrier.mutual_block_match_count == 16


def test_unknown_crossview_correlation_never_increases_confidence() -> None:
    reference = _candidate(1, np.zeros(3))
    support = _candidate(2, np.asarray([0.002, -0.001, 0.0]))
    pair = CrossViewCandidatePair(
        assignment_index=0,
        reference_camera="a",
        support_camera="b",
        reference=reference,
        support=support,
        mutual_block_match_count=16,
        median_block_distance_m=0.002,
        percentile_90_block_distance_m=0.002,
    )
    carrier = build_bias_aware_metric_carrier(
        pair,
        node_count=64,
        maximum_distance_m=0.01,
        shared_bias_floor_m=0.005,
        unsupported_node_floor_m=0.015,
        unsupported_reliability_scale=0.5,
    )
    selected_reference_covariance = reference.dense.covariance_m2[
        np.asarray(
            [
                np.flatnonzero(np.all(reference.dense.pixel_xy == pixel, axis=1))[0]
                for pixel in carrier.reference_pixel_xy
            ]
        )
    ]
    assert np.all(
        np.linalg.eigvalsh(
            carrier.marginal_covariance_m2 - selected_reference_covariance
        )
        >= -1e-12
    )
    assert np.all(np.diag(carrier.shared_bias_covariance_m2) >= 0.005**2)


def test_crossview_selection_uses_geometry_not_candidate_order() -> None:
    reference = _candidate(7, np.zeros(3))
    wrong = _candidate(1, np.asarray([0.2, 0.0, 0.0]))
    supported = _candidate(9, np.asarray([0.001, 0.0, 0.0]))
    pair = select_crossview_candidate_pair(
        {"a": [reference], "b": [wrong, supported]},
        assignment_index=1,
        camera_order=["a", "b"],
        maximum_distance_m=0.01,
        minimum_mutual_matches=8,
        maximum_percentile_90_m=0.005,
    )
    assert pair.assignment_index == 1
    assert {pair.reference.candidate_index, pair.support.candidate_index} == {7, 9}
    assert pair.mutual_block_match_count == 16
