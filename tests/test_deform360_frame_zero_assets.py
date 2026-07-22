from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from deform360_held_test_helpers import (
    default_frame_zero_config,
    dummy_immutable_bindings,
)

import bayesian_phystwin.deform360_frame_zero_assets as frame_zero_assets
from bayesian_phystwin.deform360_frame_zero_assets import (
    APPROVED_CALIBRATION_SMOKE_CASE,
    FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
    FRAME_ZERO_CAMERA_SELECTION_RULE,
    FRAME_ZERO_INFORMATION_BOUNDARY,
    FrameZeroAssetConfig,
    HELD_TARGET_CASES_V1,
    artifact_sha256,
    authorize_frame_zero_case,
    decode_exact_frame_zero,
    frame_zero_view_diagnostics_sha256,
    load_generic_held_lock,
    reject_future_derived_input,
    run_frame_zero_asset_builder,
    segment_frame_zero_views,
    select_action_only_window,
    validate_frame_zero_bundle_manifest,
    validate_generic_held_lock,
    _verify_clean_pinned_git_runtime,
)
from bayesian_phystwin.deform360_held_protocol import (
    create_held_protocol_lock,
    validate_frame_zero_bundle_manifest as validate_held_frame_zero_manifest,
)
from bayesian_phystwin.deform360_robot_kinematics import (
    validate_robot_kinematics_arrays,
)

CALIBRATION_CASES = [
    "002-rope-silk-ep0003",
    "002-rope-silk-ep0004",
    "002-rope-silk-ep0008",
    "083-blanket-cloth-ep0000",
    "083-blanket-cloth-ep0003",
    "083-blanket-cloth-ep0006",
    "085-scarf-cloth-ep0000",
    "085-scarf-cloth-ep0005",
    "085-scarf-cloth-ep0007",
    "092-squirrel-ep0002",
    "092-squirrel-ep0003",
    "092-squirrel-ep0006",
    "170-spider-ep0002",
    "170-spider-ep0004",
    "170-spider-ep0007",
]


def _selected_view_diagnostic(camera: str) -> dict[str, object]:
    return {
        "camera": camera,
        "automatic_candidate_count": 1,
        "eligible_candidate_count": 1,
        "rejected_candidate_count": 0,
        "rejection_counts": {
            "mask_threshold": 0,
            "reference_appearance_threshold": 0,
            "total": 0,
        },
        "maximum_reference_appearance_similarity": 1.0,
        "view_selected": True,
        "abstained": False,
        "abstention_reason": None,
        "selected": {"candidate_index": 0},
    }


def _lock(*, stage: str = "calibration") -> dict:
    promoted = stage == "confirmation"
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldOnlineBeliefLock",
        "protocol_id": "deform360-held-online-belief-v7",
        "stage": stage,
        "confirmation_access_authorized": promoted,
        "parent_calibration_lock": ({"sha256": "b" * 64} if promoted else None),
        "calibration_gate_evidence": ({"sha256": "c" * 64} if promoted else None),
        "cohort": [
            {
                "case_name": case_name,
                "object_id": case_name.rsplit("-ep", 1)[0],
                "episode_id": int(case_name.rsplit("ep", 1)[1]),
            }
            for case_name in HELD_TARGET_CASES_V1
        ],
        "case_whitelist": list(HELD_TARGET_CASES_V1),
        "calibration_case_whitelist": CALIBRATION_CASES,
        "update_frames": [19, 38, 57],
        "frame_count": 76,
        "immutable_bindings": dummy_immutable_bindings(),
    }
    payload["artifact_sha256"] = artifact_sha256(payload)
    return payload


def _robot_archive_arrays(frame_count: int = 120) -> dict[str, np.ndarray]:
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    actions = np.zeros((frame_count, 5, 3), dtype=np.float64)
    actions[:, 1:4] = transforms[:, :3, :3]
    openings = np.zeros(frame_count, dtype=np.float64)
    return {
        "format_version": np.asarray(1, dtype=np.uint16),
        "actions": actions,
        "T_worlds": transforms,
        "openings": openings,
        "bimanual": np.asarray(False, dtype=np.bool_),
    }


def _write_robot_archive(path: Path, frame_count: int = 120) -> dict[str, np.ndarray]:
    arrays = _robot_archive_arrays(frame_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def _write_file(path: Path, payload: bytes = b"fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _manifest(tmp_path: Path) -> dict:
    bundle = tmp_path / "frame_zero_bundle.npz"
    episode = tmp_path / "aligned" / "083-blanket-cloth" / "episode_0000"
    robot = episode / "robot" / "robot.npz"
    metadata = episode / "robot" / "robot.meta.json"
    action = tmp_path / "known_action_76.npz"
    _write_robot_archive(robot)
    metadata.write_text("{}\n", encoding="utf-8")
    intrinsics_path = episode / "undistorted_intrinsics.npy"
    extrinsics_path = episode / "extrinsics.npy"
    _, action_alignment = frame_zero_assets._slice_known_action(
        robot,
        action,
        config=FrameZeroAssetConfig(),
    )
    selected_start = action_alignment["selected_raw_frame_range_half_open"][0]
    reference_camera = FrameZeroAssetConfig().reference_camera
    cameras = sorted([reference_camera, *(f"camera-{index:02d}" for index in range(7))])
    np.save(
        intrinsics_path,
        {camera: np.eye(3, dtype=np.float64) for camera in cameras},
    )
    np.save(
        extrinsics_path,
        {camera: np.eye(4, dtype=np.float64) for camera in cameras},
    )
    for camera in cameras:
        _write_file(episode / camera / "undistorted.mp4")
    diagnostics = [_selected_view_diagnostic(camera) for camera in cameras]
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldFrameZeroBundle",
        "protocol_id": "deform360-held-online-belief-v7",
        "case_name": APPROVED_CALIBRATION_SMOKE_CASE,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "role": "calibration",
        "lock_sha256": "0" * 64,
        "lock_artifact_sha256": "a" * 64,
        "frame_indices": [0],
        "config": default_frame_zero_config(),
        "bundle": {
            "path": str(bundle.resolve()),
            "sha256": "1" * 64,
            "size_bytes": 10,
        },
        "action_inputs": {
            "robot_trajectory": _record_existing(robot),
            "robot_metadata": _record_existing(metadata),
        },
        "calibration_inputs": {
            "intrinsics": _record_existing(intrinsics_path),
            "extrinsics": _record_existing(extrinsics_path),
        },
        "action_alignment": action_alignment,
        "camera_policy": {
            "policy_id": FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
            "rule": FRAME_ZERO_CAMERA_SELECTION_RULE,
            "reference_camera": reference_camera,
            "minimum_selected_camera_count": 8,
            "candidate_cameras": cameras,
            "candidate_camera_count": len(cameras),
            "selected_cameras": cameras,
            "selected_camera_count": len(cameras),
            "abstained_cameras": [],
            "abstained_camera_count": 0,
        },
        "sam2": {
            "view_diagnostics": diagnostics,
            "view_diagnostics_sha256": frame_zero_view_diagnostics_sha256(diagnostics),
        },
        "camera_frame_zero_access": [
            {
                "camera": camera,
                "path": str((episode / camera / "undistorted.mp4").resolve()),
                "decoded_frame_count": 1,
                "maximum_rgb_frame_read": 0,
                "action_window_frame_index": 0,
                "source_aligned_frame_index": selected_start,
                "decoded_rgb_sha256": "d" * 64,
                "whole_file_hashed_or_read": False,
            }
            for camera in cameras
        ],
        "information_boundary": deepcopy(FRAME_ZERO_INFORMATION_BOUNDARY),
    }
    payload["artifact_sha256"] = artifact_sha256(payload)
    return payload


def _manifest_with_common_fallback(
    tmp_path: Path,
    masks: dict[str, np.ndarray],
    diagnostics: list[dict[str, object]],
    common_audit: dict[str, object],
) -> dict:
    payload = _manifest(tmp_path)
    candidate_cameras = [str(record["camera"]) for record in diagnostics]
    selected_cameras = sorted(masks)
    abstained_cameras = sorted(set(candidate_cameras) - set(selected_cameras))
    payload["camera_policy"].update(
        {
            "candidate_cameras": candidate_cameras,
            "candidate_camera_count": len(candidate_cameras),
            "selected_cameras": selected_cameras,
            "selected_camera_count": len(selected_cameras),
            "abstained_cameras": abstained_cameras,
            "abstained_camera_count": len(abstained_cameras),
        }
    )
    payload["sam2"]["view_diagnostics"] = deepcopy(diagnostics)
    payload["sam2"]["view_diagnostics_sha256"] = frame_zero_view_diagnostics_sha256(
        diagnostics
    )
    selected_start = payload["action_alignment"]["selected_raw_frame_range_half_open"][
        0
    ]
    raw_robot = Path(payload["action_inputs"]["robot_trajectory"]["path"])
    episode = raw_robot.parent.parent
    for camera in candidate_cameras:
        _write_file(episode / camera / "undistorted.mp4")
    payload["camera_frame_zero_access"] = [
        {
            "camera": camera,
            "path": str((episode / camera / "undistorted.mp4").resolve()),
            "decoded_frame_count": 1,
            "maximum_rgb_frame_read": 0,
            "action_window_frame_index": 0,
            "source_aligned_frame_index": selected_start,
            "decoded_rgb_sha256": "d" * 64,
            "whole_file_hashed_or_read": False,
        }
        for camera in candidate_cameras
    ]
    geometry_qa = {
        "strategy": "common-voxel-assignment-projected-footprint",
        "fallback_policy_id": (
            frame_zero_assets.FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID
        ),
        "acceptance_gates": {
            name: True for name in frame_zero_assets._FALLBACK_ACCEPTANCE_GATE_NAMES
        },
        "geometry_qa_passed": True,
    }
    geometry_sha256 = hashlib.sha256(
        frame_zero_assets._canonical_bytes(geometry_qa)
    ).hexdigest()
    final_proposals = frame_zero_assets._selected_proposal_audit(diagnostics, masks)
    final_mask_set_sha256 = frame_zero_assets._mask_set_sha256(masks)
    legacy_proposals = []
    for diagnostic in diagnostics:
        inlier = diagnostic["geometry_inlier_selection"]
        legacy_proposals.append(
            {
                "camera": diagnostic["camera"],
                "candidate_index": int(inlier["candidate_index"]),
                "automatic_candidate_count": int(
                    diagnostic["automatic_candidate_count"]
                ),
                "eligible_candidate_count": max(
                    1, int(diagnostic["eligible_candidate_count"])
                ),
                "mask_sha256": inlier["mask_sha256"],
            }
        )
    legacy_mask_set_sha256 = hashlib.sha256(
        frame_zero_assets._canonical_bytes(
            {record["camera"]: record["mask_sha256"] for record in legacy_proposals}
        )
    ).hexdigest()
    fallback = {
        "policy_id": frame_zero_assets.FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID,
        "ordered_strategies": [
            "legacy",
            "same-masks-projected-footprint",
            "common-voxel-assignment-projected-footprint",
        ],
        "strict_consensus_vote_count": 8,
        "reference_seed_policy": "top-eight-frozen-local-mask-score",
        "reference_seed_limit": 8,
        "common_grid_axis_count": 64,
        "common_local_requested_voxel_size_m": 0.004,
        "minimum_coarse_component_point_count": 64,
        "local_requested_voxel_size_m": 0.002,
        "stability_requested_voxel_size_m": 0.0025,
        "maximum_local_grid_point_count": 8_000_000,
        "minimum_scale_stability": 0.70,
        "selected_strategy": "common-voxel-assignment-projected-footprint",
        "attempts": [
            {
                "strategy": "legacy",
                "status": "failed",
                "error_type": "FrameZeroGeometryQAError",
                "reason": "frame-zero visual hull is too small",
            },
            {
                "strategy": "same-masks-projected-footprint",
                "status": "failed",
                "error_type": "FrameZeroGeometryQAError",
                "reason": "projected-footprint coarse hull is too small",
            },
            {
                "strategy": "common-voxel-assignment-projected-footprint",
                "status": "passed",
                "selected_camera_count": 8,
                "selected_mask_set_sha256": final_mask_set_sha256,
                "coarse_peak_vote_count": 8,
                "coarse_required_vote_count": 8,
                "coarse_connected_core_point_count": 64,
                "refined_surface_point_count": 128,
                "refined_required_vote_count": 8,
                "stability_required_vote_count": 8,
                "refined_grid_coarsened_for_cap": False,
                "stability_grid_coarsened_for_cap": False,
                "refined_effective_axis_spacing_m": [0.002, 0.002, 0.002],
                "stability_effective_axis_spacing_m": [0.0025, 0.0025, 0.0025],
                "raw_median_hull_mask_containment": 0.60,
                "footprint_median_hull_mask_containment": 0.90,
                "median_depth_mask_coverage": 0.10,
                "local_scale_stability": 0.70,
                "stability_component_count": 1,
                "stability_largest_component_fraction": 0.50,
                "projected_footprint_diagnostics_sha256": "7" * 64,
                "geometry_qa_sha256": geometry_sha256,
            },
        ],
        "legacy_selected_proposals": legacy_proposals,
        "legacy_selected_mask_set_sha256": legacy_mask_set_sha256,
        "common_assignment": deepcopy(common_audit),
        "final_selected_proposals": final_proposals,
        "final_selected_mask_set_sha256": final_mask_set_sha256,
    }
    fallback["artifact_sha256"] = artifact_sha256(fallback)
    payload["geometry_qa"] = geometry_qa
    payload["geometry_fallback"] = fallback
    payload["artifact_sha256"] = artifact_sha256(payload)
    return payload


def _record_existing(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def test_calibration_lock_authorizes_only_calibration_cases() -> None:
    lock = _lock()

    assert validate_generic_held_lock(lock)["passed"] is True
    authorization = authorize_frame_zero_case(
        lock, APPROVED_CALIBRATION_SMOKE_CASE, role="calibration"
    )
    non_smoke = authorize_frame_zero_case(
        lock, "002-rope-silk-ep0003", role="calibration"
    )

    assert authorization["role"] == "calibration"
    assert non_smoke == {
        "case_name": "002-rope-silk-ep0003",
        "object_id": "002-rope-silk",
        "episode_id": 3,
        "role": "calibration",
        "lock_artifact_sha256": lock["artifact_sha256"],
    }
    with pytest.raises(ValueError, match="not authorized for calibration"):
        authorize_frame_zero_case(lock, "001-rope-ep0000", role="calibration")
    with pytest.raises(ValueError, match="not authorized for calibration"):
        authorize_frame_zero_case(lock, HELD_TARGET_CASES_V1[0], role="calibration")
    with pytest.raises(ValueError, match="promoted held lock"):
        authorize_frame_zero_case(lock, HELD_TARGET_CASES_V1[0], role="confirmation")


def test_promoted_lock_is_required_for_confirmation() -> None:
    lock = _lock(stage="confirmation")

    authorization = authorize_frame_zero_case(
        lock, HELD_TARGET_CASES_V1[0], role="confirmation"
    )

    assert authorization["role"] == "confirmation"
    with pytest.raises(ValueError, match="initial lock"):
        authorize_frame_zero_case(
            lock, APPROVED_CALIBRATION_SMOKE_CASE, role="calibration"
        )


def test_generic_loader_rejects_self_promoted_confirmation_lock(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration-lock.json"
    lock = create_held_protocol_lock(
        calibration_path, immutable_bindings=dummy_immutable_bindings()
    )
    forged = deepcopy(lock)
    forged.pop("artifact_sha256")
    forged["stage"] = "confirmation"
    forged["confirmation_access_authorized"] = True
    forged["parent_calibration_lock"] = {}
    forged["calibration_gate_evidence"] = {}
    forged["information_boundary"]["calibration_outcomes_read_before_lock"] = True
    forged["artifact_sha256"] = artifact_sha256(forged)
    forged_path = tmp_path / "forged-confirmation-lock.json"
    forged_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(ValueError, match="parent calibration lock path is missing"):
        load_generic_held_lock(forged_path)


def test_pinned_runtime_rejects_dirty_tracked_source(tmp_path: Path) -> None:
    repository = tmp_path / "runtime"
    repository.mkdir()

    def git(*arguments: str) -> bytes:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
        ).stdout

    git("init", "-q")
    git("config", "user.email", "held-test@example.invalid")
    git("config", "user.name", "Held Test")
    source = repository / "runtime.py"
    source.write_text("PINNED = True\n", encoding="utf-8")
    git("add", "runtime.py")
    git("commit", "-qm", "pinned runtime")
    revision = git("rev-parse", "HEAD").decode().strip()
    tree_lines = git("ls-tree", "-r", "--full-tree", "HEAD").splitlines()
    bindings = {
        "sam2_revision_literal": hashlib.sha256(revision.encode()).hexdigest(),
        "sam2_commit_object": hashlib.sha256(
            git("cat-file", "commit", "HEAD")
        ).hexdigest(),
        "sam2_git_tree_manifest": hashlib.sha256(
            b"".join(line + b"\n" for line in sorted(tree_lines))
        ).hexdigest(),
    }
    observed = _verify_clean_pinned_git_runtime(
        repository,
        bindings,
        prefix="sam2",
        expected_revision=revision,
    )
    assert observed["revision"] == revision

    source.write_text("PINNED = False\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty tracked files"):
        _verify_clean_pinned_git_runtime(
            repository,
            bindings,
            prefix="sam2",
            expected_revision=revision,
        )


def test_builder_rejects_effective_config_change_before_episode_access(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "calibration-lock.json"
    create_held_protocol_lock(lock_path, immutable_bindings=dummy_immutable_bindings())
    changed = replace(FrameZeroAssetConfig(), rng_seed=1)

    with pytest.raises(ValueError, match="configuration differs"):
        run_frame_zero_asset_builder(
            tmp_path / "deliberately-missing-episode",
            APPROVED_CALIBRATION_SMOKE_CASE,
            lock_path,
            tmp_path / "output-must-not-be-created",
            SimpleNamespace(),
            role="calibration",
            config=changed,
        )

    assert not (tmp_path / "output-must-not-be-created").exists()


@pytest.mark.parametrize(
    "name, message",
    [
        ("mask_refined.h5", "HDF5"),
        ("rendered_depth.hdf5", "HDF5"),
        ("sam2_propagated_mask.npz", "future-derived"),
        ("target_data.npz", "future-derived"),
    ],
)
def test_future_derived_inputs_are_rejected(
    tmp_path: Path, name: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reject_future_derived_input(tmp_path / name, purpose="frame-zero mask")


def test_nonreference_zero_eligible_abstains_without_stopping_later_cameras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_camera = FrameZeroAssetConfig().reference_camera
    cameras = [reference_camera, *(f"camera-{index:02d}" for index in range(1, 10))]
    rgb_by_camera = {
        camera: np.full((4, 4, 3), index, dtype=np.uint8)
        for index, camera in enumerate(cameras)
    }
    calls: list[int] = []

    class Runtime:
        model_id = "test-only"

        def generate(self, rgb: np.ndarray) -> list[dict[str, object]]:
            camera_index = int(rgb[0, 0, 0])
            calls.append(camera_index)
            return [
                {
                    "segmentation": np.ones((4, 4), dtype=bool),
                    "predicted_iou": 0.69 if camera_index == 4 else 0.95,
                    "stability_score": 0.95,
                }
            ]

    monkeypatch.setattr(
        frame_zero_assets,
        "deformable_object_mask_candidate_diagnostics",
        lambda _rgb, _mask, _config: {
            "eligible": True,
            "score": 1.0,
            "area_pixels": 100,
            "area_fraction": 0.25,
            "foreground_contrast": 0.25,
        },
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "mask_appearance_descriptor",
        lambda rgb, _mask: {"camera_index": int(rgb[0, 0, 0])},
    )

    def similarity(
        _reference: dict[str, object], candidate: dict[str, object]
    ) -> dict[str, float]:
        combined = 0.349 if candidate["camera_index"] == 3 else 0.90
        return {
            "combined": combined,
            "hs_histogram_intersection": combined,
            "lab_similarity": combined,
            "shape_similarity": 0.90,
        }

    monkeypatch.setattr(frame_zero_assets, "mask_appearance_similarity", similarity)

    masks, diagnostics = segment_frame_zero_views(
        rgb_by_camera,
        Runtime(),
        reference_camera=reference_camera,
        config=FrameZeroAssetConfig().sam2,
    )

    assert calls == list(range(10))
    assert len(diagnostics) == 10
    assert len(masks) == 8
    assert reference_camera in masks
    assert "camera-03" not in masks
    assert "camera-04" not in masks
    appearance_abstention = next(
        record for record in diagnostics if record["camera"] == "camera-03"
    )
    assert appearance_abstention["eligible_candidate_count"] == 0
    assert appearance_abstention["rejected_candidate_count"] == 1
    assert appearance_abstention["maximum_reference_appearance_similarity"] == 0.349
    assert (
        appearance_abstention["abstention_reason"]
        == "no-candidate-met-frozen-reference-appearance-threshold"
    )
    quality_abstention = next(
        record for record in diagnostics if record["camera"] == "camera-04"
    )
    assert quality_abstention["rejection_counts"] == {
        "mask_threshold": 1,
        "reference_appearance_threshold": 0,
        "total": 1,
    }
    assert quality_abstention["abstention_reason"] == (
        "no-candidate-met-frozen-mask-thresholds"
    )


def test_fixed_reference_camera_cannot_abstain() -> None:
    reference_camera = FrameZeroAssetConfig().reference_camera
    rgb_by_camera = {
        reference_camera: np.zeros((4, 4, 3), dtype=np.uint8),
        "camera-01": np.ones((4, 4, 3), dtype=np.uint8),
    }
    runtime = SimpleNamespace(model_id="test-only", generate=lambda _rgb: [])

    with pytest.raises(ValueError, match="no eligible reference mask"):
        segment_frame_zero_views(
            rgb_by_camera,
            runtime,
            reference_camera=reference_camera,
            config=FrameZeroAssetConfig().sam2,
        )


def test_projected_footprint_uses_half_voxel_jacobian_and_pixel_quantization() -> None:
    points = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64)
    intrinsics = np.asarray(
        [[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    camera_to_world = np.eye(4, dtype=np.float64)
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 6] = True

    _pixels, _depth, radius_u, radius_v = frame_zero_assets._projected_half_voxel_radii(
        points,
        intrinsics,
        camera_to_world,
        axis_spacing_m=[0.02, 0.02, 0.02],
    )
    hits, in_bounds, diagnostics = frame_zero_assets._projected_footprint_hits(
        points,
        mask,
        intrinsics,
        camera_to_world,
        axis_spacing_m=[0.02, 0.02, 0.02],
    )

    assert radius_u.tolist() == pytest.approx([1.0])
    assert radius_v.tolist() == pytest.approx([1.0])
    assert in_bounds.tolist() == [True]
    assert hits.tolist() == [True]
    assert diagnostics["projected_half_voxel_radius_u_pixels"]["median"] == 1.0


def test_off_center_footprint_does_not_double_expand_quantization_support() -> None:
    intrinsics = np.asarray(
        [[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    points = np.asarray([[0.004, 0.0, 2.0]], dtype=np.float64)
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 4] = True

    hits, in_bounds, _diagnostics = frame_zero_assets._projected_footprint_hits(
        points,
        mask,
        intrinsics,
        np.eye(4),
        axis_spacing_m=[0.02, 0.02, 0.02],
    )
    found, selected = frame_zero_assets._nearest_projected_footprint_mask_pixels(
        points,
        mask,
        intrinsics,
        np.eye(4),
        axis_spacing_m=[0.02, 0.02, 0.02],
    )

    assert in_bounds.tolist() == [True]
    assert hits.tolist() == [False]
    assert found.tolist() == [False]
    assert selected.tolist() == [[-1, -1]]


def test_footprint_integral_hits_match_direct_inclusive_rectangles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(12)
    count = 200
    mask = rng.random((17, 19)) < 0.2
    pixels = np.column_stack(
        (rng.uniform(-3.0, 22.0, count), rng.uniform(-3.0, 20.0, count))
    )
    depth = np.ones(count, dtype=np.float64)
    radius_u = rng.uniform(0.5, 3.5, count)
    radius_v = rng.uniform(0.5, 3.5, count)
    monkeypatch.setattr(
        frame_zero_assets,
        "_projected_half_voxel_radii",
        lambda *_args, **_kwargs: (pixels, depth, radius_u, radius_v),
    )

    hits, in_bounds, _diagnostics = frame_zero_assets._projected_footprint_hits(
        np.zeros((count, 3)),
        mask,
        np.eye(3),
        np.eye(4),
        axis_spacing_m=[0.01, 0.01, 0.01],
    )
    expected_hits = np.zeros(count, dtype=bool)
    expected_in_bounds = np.zeros(count, dtype=bool)
    for index in range(count):
        left = int(np.ceil(pixels[index, 0] - radius_u[index]))
        right = int(np.floor(pixels[index, 0] + radius_u[index]))
        top = int(np.ceil(pixels[index, 1] - radius_v[index]))
        bottom = int(np.floor(pixels[index, 1] + radius_v[index]))
        valid = (
            left <= right
            and top <= bottom
            and right >= 0
            and left < mask.shape[1]
            and bottom >= 0
            and top < mask.shape[0]
        )
        expected_in_bounds[index] = valid
        if valid:
            expected_hits[index] = bool(
                np.any(
                    mask[
                        max(0, top) : min(mask.shape[0] - 1, bottom) + 1,
                        max(0, left) : min(mask.shape[1] - 1, right) + 1,
                    ]
                )
            )

    assert np.array_equal(in_bounds, expected_in_bounds)
    assert np.array_equal(hits, expected_hits)


def test_projected_footprint_carving_never_relaxes_eight_vote_quorum() -> None:
    points = np.asarray([[0.0, 0.0, 2.0], [0.08, 0.0, 2.0]], dtype=np.float64)
    intrinsics = np.asarray(
        [[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cameras = [f"camera-{index:02d}" for index in range(8)]
    intrinsics_by_camera = {camera: intrinsics for camera in cameras}
    extrinsics_by_camera = {camera: np.eye(4) for camera in cameras}
    masks_by_camera = {}
    for camera in cameras:
        mask = np.zeros((11, 11), dtype=bool)
        mask[5, 5] = True
        masks_by_camera[camera] = mask

    hull, accepted, diagnostics = (
        frame_zero_assets._carve_candidate_points_with_footprints(
            points,
            masks_by_camera,
            intrinsics_by_camera,
            extrinsics_by_camera,
            axis_spacing_m=[0.002, 0.002, 0.002],
        )
    )

    assert diagnostics["required_vote_count"] == 8
    assert diagnostics["peak_vote_count"] == 8
    assert accepted.tolist() == [True, False]
    assert hull.tolist() == [points[0].tolist()]

    masks_by_camera[cameras[-1]][5, 5] = False
    empty, accepted, diagnostics = (
        frame_zero_assets._carve_candidate_points_with_footprints(
            points,
            masks_by_camera,
            intrinsics_by_camera,
            extrinsics_by_camera,
            axis_spacing_m=[0.002, 0.002, 0.002],
        )
    )
    assert diagnostics["peak_vote_count"] == 7
    assert not np.any(accepted)
    assert len(empty) == 0


def test_footprint_rgb_pixel_uses_distance_then_row_then_column_tie_break() -> None:
    points = np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64)
    intrinsics = np.asarray(
        [[100.0, 0.0, 5.0], [0.0, 100.0, 5.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    mask = np.zeros((11, 11), dtype=bool)
    mask[4, 5] = True
    mask[5, 4] = True

    raw_hits, _visible = frame_zero_assets._point_mask_hits(
        points, mask, intrinsics, np.eye(4)
    )
    found, selected = frame_zero_assets._nearest_projected_footprint_mask_pixels(
        points,
        mask,
        intrinsics,
        np.eye(4),
        axis_spacing_m=[0.02, 0.02, 0.02],
    )

    assert raw_hits.tolist() == [False]
    assert found.tolist() == [True]
    assert selected.tolist() == [[5, 4]]


def test_local_refinement_bounds_are_clipped_and_margin_is_source_fixed() -> None:
    component = np.asarray(
        [[-0.49, -0.10, -0.02], [-0.45, 0.12, 0.04]], dtype=np.float64
    )
    minimum, maximum, diagnostics = frame_zero_assets._local_refinement_bounds(
        component,
        global_minimum_world_m=np.asarray([-0.5, -0.5, -0.5]),
        global_maximum_world_m=np.asarray([0.5, 0.5, 0.5]),
        coarse_axis_spacing_m=[0.01, 0.02, 0.03],
    )

    assert minimum.tolist() == pytest.approx([-0.5, -0.16, -0.11])
    assert maximum.tolist() == pytest.approx([-0.42, 0.18, 0.13])
    assert diagnostics["coarse_margin_cell_count"] == 3
    assert diagnostics["margin_m"] == pytest.approx([0.03, 0.06, 0.09])


def test_bounded_exact_eight_subset_audit_matches_legacy_canonical_stream() -> None:
    cameras = [f"camera-{index:02d}" for index in range(9)]
    records = []
    for index, subset in enumerate(itertools.combinations(cameras, 8)):
        records.append(
            {
                "cameras": list(subset),
                "largest_exact_component_voxel_count": index + 1,
                "exact_common_voxel_count": index + 2,
                "exact_component_count": index % 3,
                "raw_component_coverage_sum": float(index) / 10.0,
                "semantic_score_sum": float(index) / 20.0,
                "exact_common_mask_sha256": f"{index + 1:064x}",
            }
        )

    legacy = frame_zero_assets._ExactEightSubsetAuditAccumulator(bounded=False)
    bounded = frame_zero_assets._ExactEightSubsetAuditAccumulator(bounded=True)
    for record in records:
        legacy.add(record)
        bounded.add(record)
    legacy_value = legacy.materialize(selected_record=records[-1])
    bounded_value = bounded.materialize(selected_record=records[-1])

    assert legacy_value == records
    assert bounded_value["record_count"] == len(records)
    assert (
        bounded_value["contract_sha256"]
        == hashlib.sha256(
            json.dumps(
                frame_zero_assets.EXACT_EIGHT_SUBSET_BOUNDED_AUDIT_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert (
        bounded_value["canonical_json_array_sha256"]
        == hashlib.sha256(
            json.dumps(
                records,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    assert bounded_value["selected_record"] == records[-1]
    assert len(json.dumps(bounded_value)) < len(json.dumps(records))
    frame_zero_assets._validate_bounded_exact_eight_subset_audit(
        bounded_value,
        expected_record_count=len(records),
        expected_first_cameras=records[0]["cameras"],
        expected_last_cameras=records[-1]["cameras"],
        selected_cameras=sorted(records[-1]["cameras"]),
        candidate_cameras=cameras,
        fixed_first_camera=None,
    )

    tampered = deepcopy(bounded_value)
    tampered["metric_extrema"]["largest_exact_component_voxel_count"]["maximum"] += 1
    with pytest.raises(ValueError, match="winner/extrema"):
        frame_zero_assets._validate_bounded_exact_eight_subset_audit(
            tampered,
            expected_record_count=len(records),
            expected_first_cameras=records[0]["cameras"],
            expected_last_cameras=records[-1]["cameras"],
            selected_cameras=sorted(records[-1]["cameras"]),
            candidate_cameras=cameras,
            fixed_first_camera=None,
        )


def test_common_voxel_assignment_is_strict_top_seeded_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_camera = FrameZeroAssetConfig().reference_camera
    cameras = sorted([reference_camera, *(f"camera-{index:02d}" for index in range(8))])
    rgb_by_camera = {
        camera: np.zeros((64, 64, 3), dtype=np.uint8) for camera in cameras
    }
    left = np.zeros((64, 64), dtype=bool)
    left[28:37, 9:14] = True
    common = np.zeros((64, 64), dtype=bool)
    common[28:37, 30:35] = True
    proposals = {
        camera: [
            {
                "segmentation": common.copy(),
                "predicted_iou": 0.95,
                "stability_score": 0.95,
            }
        ]
        for camera in cameras
    }
    outlier_camera = cameras[-1]
    proposals[outlier_camera][0]["segmentation"] = left.copy()
    proposals[reference_camera] = [
        {
            "segmentation": (left if index < 7 else common).copy(),
            "predicted_iou": 0.95,
            "stability_score": 0.95,
        }
        for index in range(9)
    ]
    intrinsics = np.asarray(
        [[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsics_by_camera = {camera: intrinsics for camera in cameras}
    extrinsics_by_camera = {camera: np.eye(4) for camera in cameras}

    def diagnostic(
        _rgb: np.ndarray, mask: np.ndarray, _config: object
    ) -> dict[str, object]:
        rows, columns = np.nonzero(mask)
        score = 2.0 if float(np.mean(columns)) < 20.0 else 1.0
        return {
            "eligible": True,
            "score": score,
            "area_pixels": int(np.count_nonzero(mask)),
            "area_fraction": float(np.mean(mask)),
            "centroid_xy": [float(np.mean(columns)), float(np.mean(rows))],
            "normalized_center_distance": 0.1,
            "bounding_box_wh": [5, 9],
            "bounding_box_fill_fraction": 1.0,
            "border_side_count": 0,
            "foreground_contrast": 0.5,
            "border_background_rgb": [0.0, 0.0, 0.0],
        }

    monkeypatch.setattr(
        frame_zero_assets,
        "deformable_object_mask_candidate_diagnostics",
        diagnostic,
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "mask_appearance_descriptor",
        lambda _rgb, mask: {"centroid": np.mean(np.nonzero(mask)[1]).item()},
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "mask_appearance_similarity",
        lambda _reference, _candidate: {
            "combined": 1.0,
            "hs_histogram_intersection": 1.0,
            "lab_similarity": 1.0,
            "shape_similarity": 1.0,
        },
    )

    first_masks, first_diagnostics, first_audit = (
        frame_zero_assets._common_voxel_mask_assignment(
            rgb_by_camera,
            proposals,
            intrinsics_by_camera,
            extrinsics_by_camera,
            reference_camera=reference_camera,
            config=FrameZeroAssetConfig(),
        )
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "HELD_PROTOCOL_ID",
        "deform360-held-online-belief-v8.1",
    )
    second_masks, second_diagnostics, second_audit = (
        frame_zero_assets._common_voxel_mask_assignment(
            rgb_by_camera,
            proposals,
            intrinsics_by_camera,
            extrinsics_by_camera,
            reference_camera=reference_camera,
            config=FrameZeroAssetConfig(),
        )
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "HELD_PROTOCOL_ID",
        "deform360-held-online-belief-v7",
    )

    assert first_audit["selected_reference_candidate_index"] == 7
    assert first_audit["evaluated_reference_seed_count"] == 8
    assert len(first_audit["reference_candidate_ranking"]) == 9
    assert (
        first_audit["reference_candidate_ranking"][8]["selected_for_seed_evaluation"]
        is False
    )
    assert first_audit["selected_common_voxel_support_count"] == 8
    assert first_audit["strict_consensus_vote_count"] == 8
    assert isinstance(first_audit["exact_eight_subset_evaluations"], list)
    bounded_subset_audit = second_audit["exact_eight_subset_evaluations"]
    assert isinstance(bounded_subset_audit, dict)
    assert (
        bounded_subset_audit["canonical_json_array_sha256"]
        == hashlib.sha256(
            json.dumps(
                first_audit["exact_eight_subset_evaluations"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    first_without_representation = deepcopy(first_audit)
    second_without_representation = deepcopy(second_audit)
    first_without_representation.pop("exact_eight_subset_evaluations")
    second_without_representation.pop("exact_eight_subset_evaluations")
    first_without_representation.pop("artifact_sha256")
    second_without_representation.pop("artifact_sha256")
    assert first_without_representation == second_without_representation
    assert (
        sorted(bounded_subset_audit["selected_record"]["cameras"])
        == (second_audit["selected_exact_eight_cameras"])
    )
    assert len(first_masks) == 8
    assert reference_camera in first_masks
    assert outlier_camera not in first_masks
    outlier_diagnostic = next(
        record for record in first_diagnostics if record["camera"] == outlier_camera
    )
    assert outlier_diagnostic["abstention_reason"] == (
        frame_zero_assets._COMMON_INLIER_ABSTENTION_REASON
    )
    assert outlier_diagnostic["eligible_candidate_count"] == 1
    assert outlier_diagnostic["geometry_inlier_selection"]["retained"] is False
    assert first_audit["grid"]["grid_shape"] == [64, 64, 64]
    assert first_diagnostics == second_diagnostics
    assert all(
        np.array_equal(first_masks[camera], second_masks[camera])
        for camera in sorted(first_masks)
    )

    payload = _manifest_with_common_fallback(
        tmp_path, first_masks, first_diagnostics, first_audit
    )
    assert validate_frame_zero_bundle_manifest(payload)["passed"] is True
    bounded_payload = _manifest_with_common_fallback(
        tmp_path / "bounded",
        second_masks,
        second_diagnostics,
        second_audit,
    )
    assert (
        validate_frame_zero_bundle_manifest(
            bounded_payload,
            require_bounded_subset_audit=True,
        )["passed"]
        is True
    )
    with pytest.raises(ValueError, match="common-voxel assignment audit"):
        validate_frame_zero_bundle_manifest(bounded_payload)

    seed_tamper = deepcopy(payload)
    common_tamper = seed_tamper["geometry_fallback"]["common_assignment"]
    common_tamper["selected_reference_seed_rank"] = 0
    common_tamper["selected_reference_candidate_index"] = common_tamper[
        "reference_seed_evaluations"
    ][0]["reference_candidate_index"]
    common_tamper["selected_reference_mask_sha256"] = common_tamper[
        "reference_seed_evaluations"
    ][0]["reference_mask_sha256"]
    common_tamper["artifact_sha256"] = artifact_sha256(common_tamper)
    seed_tamper["geometry_fallback"]["artifact_sha256"] = artifact_sha256(
        seed_tamper["geometry_fallback"]
    )
    seed_tamper["artifact_sha256"] = artifact_sha256(seed_tamper)
    with pytest.raises(ValueError, match="common-voxel assignment audit"):
        validate_frame_zero_bundle_manifest(seed_tamper)

    subset_tamper = deepcopy(payload)
    common_tamper = subset_tamper["geometry_fallback"]["common_assignment"]
    common_tamper["exact_eight_subset_evaluations"][1] = deepcopy(
        common_tamper["exact_eight_subset_evaluations"][0]
    )
    common_tamper["artifact_sha256"] = artifact_sha256(common_tamper)
    subset_tamper["geometry_fallback"]["artifact_sha256"] = artifact_sha256(
        subset_tamper["geometry_fallback"]
    )
    subset_tamper["artifact_sha256"] = artifact_sha256(subset_tamper)
    with pytest.raises(ValueError, match="common-voxel assignment audit"):
        validate_frame_zero_bundle_manifest(subset_tamper)


def test_common_voxel_assignment_rejects_support_without_reference_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_camera = FrameZeroAssetConfig().reference_camera
    cameras = sorted([reference_camera, *(f"camera-{index:02d}" for index in range(8))])
    rgb_by_camera = {
        camera: np.zeros((64, 64, 3), dtype=np.uint8) for camera in cameras
    }
    reference_only = np.zeros((64, 64), dtype=bool)
    reference_only[28:37, 9:14] = True
    nonreference_common = np.zeros((64, 64), dtype=bool)
    nonreference_common[28:37, 30:35] = True
    proposals = {
        camera: [
            {
                "segmentation": (
                    reference_only
                    if camera == reference_camera
                    else nonreference_common
                ).copy(),
                "predicted_iou": 0.95,
                "stability_score": 0.95,
            }
        ]
        for camera in cameras
    }

    def diagnostic(
        _rgb: np.ndarray, mask: np.ndarray, _config: object
    ) -> dict[str, object]:
        rows, columns = np.nonzero(mask)
        return {
            "eligible": True,
            "score": 1.0,
            "area_pixels": int(np.count_nonzero(mask)),
            "area_fraction": float(np.mean(mask)),
            "centroid_xy": [float(np.mean(columns)), float(np.mean(rows))],
            "normalized_center_distance": 0.1,
            "bounding_box_wh": [5, 9],
            "bounding_box_fill_fraction": 1.0,
            "border_side_count": 0,
            "foreground_contrast": 0.5,
            "border_background_rgb": [0.0, 0.0, 0.0],
        }

    monkeypatch.setattr(
        frame_zero_assets,
        "deformable_object_mask_candidate_diagnostics",
        diagnostic,
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "mask_appearance_descriptor",
        lambda _rgb, mask: {"centroid": np.mean(np.nonzero(mask)[1]).item()},
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "mask_appearance_similarity",
        lambda _reference, _candidate: {
            "combined": 1.0,
            "hs_histogram_intersection": 1.0,
            "lab_similarity": 1.0,
            "shape_similarity": 1.0,
        },
    )
    intrinsics = np.asarray(
        [[40.0, 0.0, 32.0], [0.0, 40.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    with pytest.raises(
        frame_zero_assets.FrameZeroGeometryQAError,
        match="reference-anchored strict-eight component",
    ):
        frame_zero_assets._common_voxel_mask_assignment(
            rgb_by_camera,
            proposals,
            {camera: intrinsics for camera in cameras},
            {camera: np.eye(4) for camera in cameras},
            reference_camera=reference_camera,
            config=FrameZeroAssetConfig(),
        )


def test_fallback_geometry_refines_sub_thousand_coarse_core_without_relaxing_qa() -> (
    None
):
    cameras = [f"camera-{index:02d}" for index in range(8)]
    rgb_by_camera = {
        camera: np.full((16, 16, 3), [100, 120, 140], dtype=np.uint8)
        for camera in cameras
    }
    masks_by_camera = {camera: np.ones((16, 16), dtype=bool) for camera in cameras}
    intrinsics = np.asarray(
        [[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsics_by_camera = {camera: intrinsics for camera in cameras}
    extrinsics_by_camera = {camera: np.eye(4) for camera in cameras}
    config = replace(
        FrameZeroAssetConfig(),
        reference_camera=cameras[0],
        cube_half_extent_m=0.05,
        requested_voxel_size_m=0.01,
        object_point_count=1_000,
    )

    arrays, diagnostics = frame_zero_assets._build_frame_zero_fallback_geometry(
        rgb_by_camera,
        masks_by_camera,
        intrinsics_by_camera,
        extrinsics_by_camera,
        config=config,
        strategy="same-masks-projected-footprint",
    )

    assert diagnostics["geometry_qa_passed"] is True
    assert diagnostics["coarse_carving"]["required_vote_count"] == 8
    assert diagnostics["refined_carving"]["required_vote_count"] == 8
    assert diagnostics["stability_carving"]["required_vote_count"] == 8
    assert diagnostics["refined_grid"]["coarsened_for_grid_cap"] is False
    assert diagnostics["stability_grid"]["coarsened_for_grid_cap"] is False
    assert (
        64
        <= diagnostics["coarse_components"]["largest_component_point_count"]
        < config.minimum_hull_point_count
    )
    assert diagnostics["refined_surface_point_count"] >= 128
    assert diagnostics["raw_hull_mask_containment"]["median"] >= 0.60
    assert diagnostics["footprint_hull_mask_containment"]["median"] >= 0.60
    assert diagnostics["depth_mask_coverage"]["median"] >= 0.10
    assert diagnostics["local_resolution_stability"]["symmetric_volume_ratio"] >= 0.70
    assert diagnostics["stability_components"]["largest_component_fraction"] >= 0.5
    assert np.all(arrays["object_color_support_count"] == 8)
    assert diagnostics["raw_center_object_color_support_count"]["minimum"] == 0


def test_manifest_rejects_modified_or_incomplete_camera_abstention_diagnostics(
    tmp_path: Path,
) -> None:
    payload = _manifest(tmp_path)
    payload["sam2"]["view_diagnostics"][1][
        "maximum_reference_appearance_similarity"
    ] = 0.5
    payload["artifact_sha256"] = artifact_sha256(payload)

    with pytest.raises(ValueError, match="diagnostics checksum"):
        validate_frame_zero_bundle_manifest(payload)

    below_minimum = _manifest(tmp_path)
    removed = below_minimum["camera_policy"]["selected_cameras"][-1]
    below_minimum["camera_policy"]["selected_cameras"] = below_minimum["camera_policy"][
        "selected_cameras"
    ][:-1]
    below_minimum["camera_policy"]["selected_camera_count"] = 7
    below_minimum["camera_policy"]["abstained_cameras"] = [removed]
    below_minimum["camera_policy"]["abstained_camera_count"] = 1
    record = next(
        item
        for item in below_minimum["sam2"]["view_diagnostics"]
        if item["camera"] == removed
    )
    record.update(
        {
            "eligible_candidate_count": 0,
            "rejected_candidate_count": 1,
            "rejection_counts": {
                "mask_threshold": 0,
                "reference_appearance_threshold": 1,
                "total": 1,
            },
            "view_selected": False,
            "abstained": True,
            "abstention_reason": (
                "no-candidate-met-frozen-reference-appearance-threshold"
            ),
            "selected": None,
        }
    )
    below_minimum["sam2"]["view_diagnostics_sha256"] = (
        frame_zero_view_diagnostics_sha256(below_minimum["sam2"]["view_diagnostics"])
    )
    below_minimum["artifact_sha256"] = artifact_sha256(below_minimum)

    with pytest.raises(ValueError, match="selected camera requirement"):
        validate_frame_zero_bundle_manifest(below_minimum)


def test_manifest_binds_geometry_fallback_audit_independently(
    tmp_path: Path,
) -> None:
    payload = _manifest(tmp_path)
    cameras = payload["camera_policy"]["selected_cameras"]
    geometry_qa = {
        "strategy": "same-masks-projected-footprint",
        "fallback_policy_id": (
            frame_zero_assets.FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID
        ),
        "acceptance_gates": {
            name: True for name in frame_zero_assets._FALLBACK_ACCEPTANCE_GATE_NAMES
        },
        "geometry_qa_passed": True,
    }
    geometry_sha256 = hashlib.sha256(
        frame_zero_assets._canonical_bytes(geometry_qa)
    ).hexdigest()
    proposal_audit = [
        {
            "camera": camera,
            "candidate_index": 0,
            "automatic_candidate_count": 1,
            "eligible_candidate_count": 1,
            "mask_sha256": "6" * 64,
        }
        for camera in cameras
    ]
    mask_set_sha256 = hashlib.sha256(
        frame_zero_assets._canonical_bytes(
            {record["camera"]: record["mask_sha256"] for record in proposal_audit}
        )
    ).hexdigest()
    fallback = {
        "policy_id": frame_zero_assets.FRAME_ZERO_GEOMETRY_FALLBACK_POLICY_ID,
        "ordered_strategies": [
            "legacy",
            "same-masks-projected-footprint",
            "common-voxel-assignment-projected-footprint",
        ],
        "strict_consensus_vote_count": 8,
        "reference_seed_policy": "top-eight-frozen-local-mask-score",
        "reference_seed_limit": 8,
        "common_grid_axis_count": 64,
        "common_local_requested_voxel_size_m": 0.004,
        "minimum_coarse_component_point_count": 64,
        "local_requested_voxel_size_m": 0.002,
        "stability_requested_voxel_size_m": 0.0025,
        "maximum_local_grid_point_count": 8_000_000,
        "minimum_scale_stability": 0.70,
        "selected_strategy": "same-masks-projected-footprint",
        "attempts": [
            {
                "strategy": "legacy",
                "status": "failed",
                "error_type": "FrameZeroGeometryQAError",
                "reason": "frame-zero visual hull is too small",
            },
            {
                "strategy": "same-masks-projected-footprint",
                "status": "passed",
                "selected_camera_count": len(cameras),
                "selected_mask_set_sha256": mask_set_sha256,
                "coarse_peak_vote_count": 8,
                "coarse_required_vote_count": 8,
                "coarse_connected_core_point_count": 64,
                "refined_surface_point_count": 128,
                "refined_required_vote_count": 8,
                "stability_required_vote_count": 8,
                "refined_grid_coarsened_for_cap": False,
                "stability_grid_coarsened_for_cap": False,
                "refined_effective_axis_spacing_m": [0.002, 0.002, 0.002],
                "stability_effective_axis_spacing_m": [0.0025, 0.0025, 0.0025],
                "raw_median_hull_mask_containment": 0.60,
                "footprint_median_hull_mask_containment": 0.90,
                "median_depth_mask_coverage": 0.10,
                "local_scale_stability": 0.70,
                "stability_component_count": 1,
                "stability_largest_component_fraction": 0.50,
                "projected_footprint_diagnostics_sha256": "7" * 64,
                "geometry_qa_sha256": geometry_sha256,
            },
        ],
        "legacy_selected_proposals": proposal_audit,
        "legacy_selected_mask_set_sha256": mask_set_sha256,
        "common_assignment": None,
        "final_selected_proposals": deepcopy(proposal_audit),
        "final_selected_mask_set_sha256": mask_set_sha256,
    }
    fallback["artifact_sha256"] = artifact_sha256(fallback)
    payload["geometry_qa"] = geometry_qa
    payload["geometry_fallback"] = fallback
    payload["artifact_sha256"] = artifact_sha256(payload)

    assert validate_frame_zero_bundle_manifest(payload)["passed"] is True

    shallow_tamper = deepcopy(payload)
    shallow_tamper["geometry_fallback"]["local_requested_voxel_size_m"] = 0.004
    shallow_tamper["artifact_sha256"] = artifact_sha256(shallow_tamper)
    with pytest.raises(ValueError, match="fallback checksum"):
        validate_frame_zero_bundle_manifest(shallow_tamper)

    deep_tamper = deepcopy(payload)
    deep_tamper["geometry_fallback"]["attempts"][-1][
        "coarse_connected_core_point_count"
    ] = 63
    deep_tamper["geometry_fallback"]["artifact_sha256"] = artifact_sha256(
        deep_tamper["geometry_fallback"]
    )
    deep_tamper["artifact_sha256"] = artifact_sha256(deep_tamper)
    with pytest.raises(ValueError, match="passed geometry fallback"):
        validate_frame_zero_bundle_manifest(deep_tamper)

    proposal_tamper = deepcopy(payload)
    proposal_tamper["geometry_fallback"]["final_selected_proposals"][0][
        "candidate_index"
    ] = 1
    proposal_tamper["geometry_fallback"]["artifact_sha256"] = artifact_sha256(
        proposal_tamper["geometry_fallback"]
    )
    proposal_tamper["artifact_sha256"] = artifact_sha256(proposal_tamper)
    with pytest.raises(ValueError, match="unexpectedly changed proposals"):
        validate_frame_zero_bundle_manifest(proposal_tamper)

    mask_digest_tamper = deepcopy(payload)
    mask_digest_tamper["geometry_fallback"]["final_selected_proposals"][0][
        "mask_sha256"
    ] = "8" * 64
    mask_digest_tamper["geometry_fallback"]["artifact_sha256"] = artifact_sha256(
        mask_digest_tamper["geometry_fallback"]
    )
    mask_digest_tamper["artifact_sha256"] = artifact_sha256(mask_digest_tamper)
    with pytest.raises(ValueError, match="fallback mask checksum"):
        validate_frame_zero_bundle_manifest(mask_digest_tamper)

    omitted_gate = deepcopy(payload)
    omitted_gate["geometry_qa"]["acceptance_gates"].pop(
        frame_zero_assets._FALLBACK_ACCEPTANCE_GATE_NAMES[-1]
    )
    omitted_gate["geometry_fallback"]["attempts"][-1]["geometry_qa_sha256"] = (
        hashlib.sha256(
            frame_zero_assets._canonical_bytes(omitted_gate["geometry_qa"])
        ).hexdigest()
    )
    omitted_gate["geometry_fallback"]["artifact_sha256"] = artifact_sha256(
        omitted_gate["geometry_fallback"]
    )
    omitted_gate["artifact_sha256"] = artifact_sha256(omitted_gate)
    with pytest.raises(ValueError, match="geometry fallback differs from geometry QA"):
        validate_frame_zero_bundle_manifest(omitted_gate)

    failed_gate = deepcopy(payload)
    failed_gate["geometry_qa"]["acceptance_gates"][
        frame_zero_assets._FALLBACK_ACCEPTANCE_GATE_NAMES[-1]
    ] = False
    failed_gate["geometry_fallback"]["attempts"][-1]["geometry_qa_sha256"] = (
        hashlib.sha256(
            frame_zero_assets._canonical_bytes(failed_gate["geometry_qa"])
        ).hexdigest()
    )
    failed_gate["geometry_fallback"]["artifact_sha256"] = artifact_sha256(
        failed_gate["geometry_fallback"]
    )
    failed_gate["artifact_sha256"] = artifact_sha256(failed_gate)
    with pytest.raises(ValueError, match="geometry fallback differs from geometry QA"):
        validate_frame_zero_bundle_manifest(failed_gate)


def test_builder_slices_rgb_and_calibration_to_nonabstaining_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "calibration-lock.json"
    create_held_protocol_lock(lock_path, immutable_bindings=dummy_immutable_bindings())
    episode = tmp_path / "083-blanket-cloth" / "episode_0000"
    reference_camera = FrameZeroAssetConfig().reference_camera
    cameras = tuple(
        sorted([reference_camera, *(f"camera-{index:02d}" for index in range(8))])
    )
    selected_cameras = cameras[:-1]
    abstained_camera = cameras[-1]
    for camera in cameras:
        video = episode / camera / "undistorted.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"frame-zero-only-test")
    intrinsics = {camera: np.eye(3) for camera in cameras}
    extrinsics = {camera: np.eye(4) for camera in cameras}
    np.save(episode / "undistorted_intrinsics.npy", intrinsics)
    np.save(episode / "extrinsics.npy", extrinsics)
    robot_path = episode / "robot" / "robot.npz"
    _write_robot_archive(robot_path)
    metadata_path = robot_path.with_name("robot.meta.json")
    metadata_path.write_bytes(b"{}")
    monkeypatch.setattr(
        frame_zero_assets,
        "_action_inputs",
        lambda _episode: (
            {
                "robot_trajectory": _record_existing(robot_path),
                "robot_metadata": _record_existing(metadata_path),
            },
            robot_path,
        ),
    )

    monkeypatch.setattr(
        frame_zero_assets,
        "decode_exact_frame_zero",
        lambda _path, *, source_aligned_frame_index: (
            np.zeros((4, 4, 3), dtype=np.uint8),
            {
                "path": str(Path(_path).resolve()),
                "decoded_frame_count": 1,
                "maximum_rgb_frame_read": 0,
                "action_window_frame_index": 0,
                "source_aligned_frame_index": source_aligned_frame_index,
                "decoded_rgb_sha256": "d" * 64,
                "whole_file_hashed_or_read": False,
            },
        ),
    )
    diagnostics = [_selected_view_diagnostic(camera) for camera in cameras]
    diagnostics[-1].update(
        {
            "eligible_candidate_count": 0,
            "rejected_candidate_count": 1,
            "rejection_counts": {
                "mask_threshold": 0,
                "reference_appearance_threshold": 1,
                "total": 1,
            },
            "maximum_reference_appearance_similarity": 0.2,
            "view_selected": False,
            "abstained": True,
            "abstention_reason": (
                "no-candidate-met-frozen-reference-appearance-threshold"
            ),
            "selected": None,
        }
    )
    masks = {camera: np.ones((4, 4), dtype=bool) for camera in selected_cameras}
    monkeypatch.setattr(
        frame_zero_assets,
        "segment_frame_zero_views",
        lambda rgb, *_args, **_kwargs: (
            (
                masks,
                diagnostics,
            )
            if tuple(sorted(rgb)) == cameras
            else pytest.fail("builder did not process every candidate camera")
        ),
    )
    received: dict[str, tuple[str, ...]] = {}

    def fake_geometry(
        rgb: dict[str, np.ndarray],
        selected_masks: dict[str, np.ndarray],
        selected_intrinsics: dict[str, np.ndarray],
        selected_extrinsics: dict[str, np.ndarray],
        *,
        config: FrameZeroAssetConfig,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del config
        received.update(
            {
                "rgb": tuple(sorted(rgb)),
                "masks": tuple(sorted(selected_masks)),
                "intrinsics": tuple(sorted(selected_intrinsics)),
                "extrinsics": tuple(sorted(selected_extrinsics)),
            }
        )
        return {
            "frame_indices": np.asarray([0], dtype=np.int64),
            "camera_names": np.asarray(selected_cameras),
        }, {"geometry_qa_passed": True}

    monkeypatch.setattr(frame_zero_assets, "build_frame_zero_geometry", fake_geometry)

    manifest = run_frame_zero_asset_builder(
        episode,
        APPROVED_CALIBRATION_SMOKE_CASE,
        lock_path,
        tmp_path / "frame-zero-output",
        SimpleNamespace(model_id="test-only"),
        role="calibration",
    )

    assert received == {
        "rgb": selected_cameras,
        "masks": selected_cameras,
        "intrinsics": selected_cameras,
        "extrinsics": selected_cameras,
    }
    assert manifest["camera_policy"]["candidate_cameras"] == list(cameras)
    assert manifest["camera_policy"]["selected_cameras"] == list(selected_cameras)
    assert manifest["camera_policy"]["abstained_cameras"] == [abstained_camera]
    assert len(manifest["camera_frame_zero_access"]) == len(cameras)
    assert "geometry_fallback" not in manifest


def test_squirrel2_frame224_runs_common_assignment_after_too_few_legacy_masks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "calibration-lock.json"
    create_held_protocol_lock(lock_path, immutable_bindings=dummy_immutable_bindings())
    episode = tmp_path / "092-squirrel" / "episode_0002"
    reference_camera = FrameZeroAssetConfig().reference_camera
    cameras = tuple(
        sorted([reference_camera, *(f"camera-{index:02d}" for index in range(8))])
    )
    for camera in cameras:
        video = episode / camera / "undistorted.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"source-only-frame")
    intrinsics = {camera: np.eye(3, dtype=np.float64) for camera in cameras}
    extrinsics = {camera: np.eye(4, dtype=np.float64) for camera in cameras}
    monkeypatch.setattr(
        frame_zero_assets,
        "_load_calibration",
        lambda _episode: (intrinsics, extrinsics, {"test_only": True}),
    )
    robot_path = episode / "robot" / "robot.npz"
    _write_robot_archive(robot_path, frame_count=320)
    metadata_path = robot_path.with_name("robot.meta.json")
    metadata_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        frame_zero_assets,
        "_action_inputs",
        lambda _episode: (
            {
                "robot_trajectory": _record_existing(robot_path),
                "robot_metadata": _record_existing(metadata_path),
            },
            robot_path,
        ),
    )

    def fake_slice(
        _robot_path: Path,
        output_path: Path,
        *,
        config: FrameZeroAssetConfig,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        assert config.prediction_frame_count == 76
        selected = _robot_archive_arrays(76)
        np.savez_compressed(output_path, **selected)
        record = _record_existing(output_path)
        return selected, {
            "selected_raw_frame_range_half_open": [224, 305],
            "prediction_raw_frame_range_half_open": [224, 300],
            "selected_action_bundle": record,
        }

    monkeypatch.setattr(frame_zero_assets, "_slice_known_action", fake_slice)
    monkeypatch.setattr(
        frame_zero_assets,
        "decode_exact_frame_zero",
        lambda _path, *, source_aligned_frame_index: (
            np.zeros((4, 4, 3), dtype=np.uint8),
            {
                "path": str(Path(_path).resolve()),
                "decoded_frame_count": 1,
                "maximum_rgb_frame_read": 0,
                "action_window_frame_index": 0,
                "source_aligned_frame_index": source_aligned_frame_index,
                "decoded_rgb_sha256": "d" * 64,
                "whole_file_hashed_or_read": False,
            },
        ),
    )
    legacy_cameras = cameras[:7]
    common_cameras = cameras[:8]

    def diagnostics_for(selected: tuple[str, ...]) -> list[dict[str, object]]:
        diagnostics = []
        for camera in cameras:
            record = _selected_view_diagnostic(camera)
            if camera not in selected:
                record.update(
                    {
                        "eligible_candidate_count": 0,
                        "rejected_candidate_count": 1,
                        "rejection_counts": {
                            "mask_threshold": 0,
                            "reference_appearance_threshold": 1,
                            "total": 1,
                        },
                        "view_selected": False,
                        "abstained": True,
                        "abstention_reason": (
                            "no-candidate-met-frozen-reference-appearance-threshold"
                        ),
                        "selected": None,
                    }
                )
            diagnostics.append(record)
        return diagnostics

    def fake_segment(
        rgb_by_camera: dict[str, np.ndarray],
        _runtime: object,
        *,
        reference_camera: str,
        config: object,
        proposal_sink: dict[str, list[dict[str, object]]],
    ) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
        del config
        assert reference_camera in legacy_cameras
        assert tuple(sorted(rgb_by_camera)) == cameras
        for camera in cameras:
            proposal_sink[camera] = [
                {
                    "segmentation": np.ones((4, 4), dtype=bool),
                    "predicted_iou": 1.0,
                    "stability_score": 1.0,
                }
            ]
        return (
            {camera: np.ones((4, 4), dtype=bool) for camera in legacy_cameras},
            diagnostics_for(legacy_cameras),
        )

    monkeypatch.setattr(frame_zero_assets, "segment_frame_zero_views", fake_segment)
    monkeypatch.setattr(
        frame_zero_assets,
        "build_frame_zero_geometry",
        lambda *_args, **_kwargs: pytest.fail(
            "legacy geometry must not run below its camera quorum"
        ),
    )
    call_log: list[str] = []

    def fake_common(*_args: object, **_kwargs: object) -> tuple[dict, list, dict]:
        call_log.append("common")
        return (
            {camera: np.ones((4, 4), dtype=bool) for camera in common_cameras},
            diagnostics_for(common_cameras),
            {"test_only": True},
        )

    monkeypatch.setattr(
        frame_zero_assets,
        "_common_voxel_mask_assignment",
        fake_common,
    )

    def fake_fallback_geometry(
        _rgb: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        _intrinsics: dict[str, np.ndarray],
        _extrinsics: dict[str, np.ndarray],
        *,
        config: FrameZeroAssetConfig,
        strategy: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del config
        call_log.append(strategy)
        assert tuple(sorted(masks)) == common_cameras
        return (
            {"frame_indices": np.asarray([0], dtype=np.int64)},
            {"geometry_qa_passed": True},
        )

    monkeypatch.setattr(
        frame_zero_assets,
        "_build_frame_zero_fallback_geometry",
        fake_fallback_geometry,
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "_fallback_attempt_pass_record",
        lambda strategy, _qa, _masks: {"strategy": strategy, "status": "passed"},
    )
    monkeypatch.setattr(
        frame_zero_assets,
        "validate_frame_zero_bundle_manifest",
        lambda _payload: {"passed": True},
    )

    manifest = run_frame_zero_asset_builder(
        episode,
        "092-squirrel-ep0002",
        lock_path,
        tmp_path / "squirrel2-frame224-output",
        SimpleNamespace(model_id="test-only"),
        role="calibration",
    )

    assert call_log == [
        "common",
        "common-voxel-assignment-projected-footprint",
    ]
    assert manifest["camera_policy"]["selected_cameras"] == list(common_cameras)
    assert all(
        record["source_aligned_frame_index"] == 224
        for record in manifest["camera_frame_zero_access"]
    )
    fallback = manifest["geometry_fallback"]
    assert fallback["selected_strategy"] == (
        "common-voxel-assignment-projected-footprint"
    )
    assert [attempt["status"] for attempt in fallback["attempts"]] == [
        "failed",
        "failed",
        "passed",
    ]
    assert "too few cameras" in fallback["attempts"][0]["reason"]


def test_decode_reads_once_and_labels_action_window_frame_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "undistorted.mp4"
    video.write_bytes(b"fixture")
    bgr = np.asarray([[[1, 2, 3]]], dtype=np.uint8)

    class Capture:
        def __init__(self) -> None:
            self.read_count = 0
            self.set_calls: list[tuple[int, int]] = []

        def set(self, key: int, value: int) -> bool:
            self.set_calls.append((key, value))
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            self.read_count += 1
            return True, bgr.copy()

        def release(self) -> None:
            return None

    capture = Capture()
    fake_cv2 = SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        COLOR_BGR2RGB=2,
        VideoCapture=lambda _: capture,
        cvtColor=lambda value, _: value[..., ::-1],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    rgb, audit = decode_exact_frame_zero(video, source_aligned_frame_index=62)

    assert rgb.tolist() == [[[3, 2, 1]]]
    assert capture.read_count == 1
    assert capture.set_calls == [(1, 62)]
    assert audit["decoded_frame_count"] == 1
    assert audit["maximum_rgb_frame_read"] == 0
    assert audit["action_window_frame_index"] == 0
    assert audit["source_aligned_frame_index"] == 62
    assert audit["whole_file_hashed_or_read"] is False


def test_robot_window_is_selected_from_realized_kinematics_with_earliest_tie() -> None:
    arrays = _robot_archive_arrays()
    state = validate_robot_kinematics_arrays(**arrays)

    selected = select_action_only_window(
        state,
        window_length_frames=81,
        prediction_frame_count=76,
        candidate_first_frame=8,
        candidate_stride_frames=6,
    )

    assert selected["selected_raw_frame_range_half_open"] == [8, 89]
    assert selected["prediction_raw_frame_range_half_open"] == [8, 84]
    assert selected["tracking_tail_frame_count"] == 5
    assert selected["object_geometry_used_for_selection"] is False
    assert selected["tactile_used_for_selection"] is False
    assert selected["policy_id"] == ("deform360-tworld-eef-translation-closed-path-v1")


def test_known_robot_slice_has_exact_five_fields_and_preserves_scalars(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "robot.npz"
    source = _write_robot_archive(source_path)
    selected_path = tmp_path / "known_robot_76.npz"

    arrays, alignment = frame_zero_assets._slice_known_action(
        source_path,
        selected_path,
        config=FrameZeroAssetConfig(),
    )

    assert set(arrays) == {
        "format_version",
        "actions",
        "T_worlds",
        "openings",
        "bimanual",
    }
    start, stop = alignment["prediction_raw_frame_range_half_open"]
    assert stop - start == 76
    with np.load(selected_path, allow_pickle=False) as stored:
        assert set(stored.files) == set(arrays)
        for name in ("actions", "T_worlds", "openings"):
            assert np.array_equal(stored[name], source[name][start:stop])
        for name in ("format_version", "bimanual"):
            assert stored[name].shape == ()
            assert np.array_equal(stored[name], source[name])
    assert alignment["selected_bundle_exact_slice_audit"]["exact_source_slice"] is True
    assert alignment["selected_action_bundle_is_compatibility_alias"] is True
    assert (
        alignment["selected_robot_kinematics_bundle"]
        == alignment["selected_action_bundle"]
    )


def test_known_robot_slice_rejects_redundant_representation_mismatch(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "robot.npz"
    arrays = _robot_archive_arrays()
    arrays["actions"][0, 0, 0] = 0.25
    np.savez_compressed(source_path, **arrays)

    with pytest.raises(ValueError, match="row 0 does not match T_worlds"):
        frame_zero_assets._slice_known_action(
            source_path,
            tmp_path / "selected.npz",
            config=FrameZeroAssetConfig(),
        )


def test_manifest_recomputes_robot_selection_and_exact_slice(tmp_path: Path) -> None:
    payload = _manifest(tmp_path)

    assert validate_frame_zero_bundle_manifest(payload)["passed"] is True

    tampered = deepcopy(payload)
    audit = tampered["action_alignment"]["selection_audit"]
    audit["selected_candidate_index"] = 1
    audit["artifact_sha256"] = artifact_sha256(audit)
    tampered["artifact_sha256"] = artifact_sha256(tampered)
    with pytest.raises(ValueError, match="selection audit changed"):
        validate_frame_zero_bundle_manifest(tampered)

    proof_tamper = deepcopy(payload)
    proof_tamper["action_alignment"]["selected_bundle_exact_slice_audit"][
        "exact_source_slice"
    ] = False
    proof_tamper["artifact_sha256"] = artifact_sha256(proof_tamper)
    with pytest.raises(ValueError, match="exact-slice proof changed"):
        validate_frame_zero_bundle_manifest(proof_tamper)


@pytest.mark.parametrize("linked_component", ["robot", "camera"])
def test_manifest_rejects_symlinked_dataset_ancestor(
    tmp_path: Path, linked_component: str
) -> None:
    payload = _manifest(tmp_path)
    raw_robot = Path(payload["action_inputs"]["robot_trajectory"]["path"])
    episode = raw_robot.parent.parent
    if linked_component == "robot":
        source = episode / "robot"
    else:
        camera = payload["camera_policy"]["candidate_cameras"][0]
        source = episode / camera
    outside = tmp_path / f"outside-{linked_component}"
    source.rename(outside)
    source.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        validate_frame_zero_bundle_manifest(payload)


def test_manifest_rejects_camera_source_index_different_from_robot_start(
    tmp_path: Path,
) -> None:
    payload = _manifest(tmp_path)
    payload["camera_frame_zero_access"][0]["source_aligned_frame_index"] += 1
    payload["artifact_sha256"] = artifact_sha256(payload)

    with pytest.raises(ValueError, match="selected raw start"):
        validate_frame_zero_bundle_manifest(payload)


def test_manifest_rejects_multiframe_or_future_geometry(tmp_path: Path) -> None:
    payload = _manifest(tmp_path)

    assert validate_frame_zero_bundle_manifest(payload)["passed"] is True

    multiframe = deepcopy(payload)
    multiframe["frame_indices"] = [0, 1]
    multiframe["artifact_sha256"] = artifact_sha256(multiframe)
    with pytest.raises(ValueError, match="multi-frame"):
        validate_frame_zero_bundle_manifest(multiframe)

    future = deepcopy(payload)
    future["information_boundary"]["future_object_geometry_read"] = True
    future["artifact_sha256"] = artifact_sha256(future)
    with pytest.raises(ValueError, match="information boundary"):
        validate_frame_zero_bundle_manifest(future)


def test_manifest_rejects_unsliced_robot_action(tmp_path: Path) -> None:
    payload = _manifest(tmp_path)
    payload["action_alignment"]["prediction_raw_frame_range_half_open"] = [62, 143]
    payload["artifact_sha256"] = artifact_sha256(payload)

    with pytest.raises(ValueError, match="76-frame"):
        validate_frame_zero_bundle_manifest(payload)


def test_manifest_matches_the_independent_held_validator(tmp_path: Path) -> None:
    lock_path = tmp_path / "calibration-lock.json"
    lock = create_held_protocol_lock(
        lock_path,
        immutable_bindings=dummy_immutable_bindings(),
    )
    payload = _manifest(tmp_path)
    bundle_path = Path(payload["bundle"]["path"])
    np.savez_compressed(
        bundle_path,
        camera_names=np.asarray(payload["camera_policy"]["selected_cameras"]),
    )
    payload["bundle"] = _record_existing(bundle_path)
    payload["lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    payload["lock_artifact_sha256"] = lock["artifact_sha256"]
    payload["artifact_sha256"] = artifact_sha256(payload)
    manifest_path = tmp_path / "frame-zero-bundle.manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    validated = validate_held_frame_zero_manifest(
        manifest_path,
        lock_path,
        expected_case_name=APPROVED_CALIBRATION_SMOKE_CASE,
        expected_role="calibration",
    )

    assert validated["artifact_sha256"] == payload["artifact_sha256"]
