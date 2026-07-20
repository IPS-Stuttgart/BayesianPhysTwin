from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
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
        "protocol_id": "deform360-held-online-belief-v2",
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


def _manifest(tmp_path: Path) -> dict:
    bundle = tmp_path / "frame_zero_bundle.npz"
    robot = tmp_path / "robot.npz"
    metadata = tmp_path / "robot.meta.json"
    action = tmp_path / "known_action_76.npz"
    reference_camera = FrameZeroAssetConfig().reference_camera
    cameras = sorted([reference_camera, *(f"camera-{index:02d}" for index in range(7))])
    diagnostics = [_selected_view_diagnostic(camera) for camera in cameras]
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldFrameZeroBundle",
        "protocol_id": "deform360-held-online-belief-v2",
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
            "robot_trajectory": {
                "path": str(robot.resolve()),
                "sha256": "2" * 64,
                "size_bytes": 20,
            },
            "robot_metadata": {
                "path": str(metadata.resolve()),
                "sha256": "3" * 64,
                "size_bytes": 30,
            },
        },
        "action_alignment": {
            "selected_raw_frame_range_half_open": [62, 143],
            "prediction_raw_frame_range_half_open": [62, 138],
            "selected_action_bundle": {
                "path": str(action.resolve()),
                "sha256": "4" * 64,
                "size_bytes": 40,
            },
        },
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
        "information_boundary": deepcopy(FRAME_ZERO_INFORMATION_BOUNDARY),
    }
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
    monkeypatch.setattr(
        frame_zero_assets,
        "_load_calibration",
        lambda _episode: (intrinsics, extrinsics, {"test_only": True}),
    )
    robot_path = episode / "robot" / "robot.npz"
    robot_path.parent.mkdir(parents=True)
    robot_path.write_bytes(b"robot")
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

    def fake_slice(
        _robot_path: Path,
        output_path: Path,
        *,
        config: FrameZeroAssetConfig,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        del config
        np.savez_compressed(output_path, actions=np.zeros((76, 1, 3)))
        return {}, {
            "selected_raw_frame_range_half_open": [0, 81],
            "prediction_raw_frame_range_half_open": [0, 76],
            "selected_action_bundle": _record_existing(output_path),
        }

    monkeypatch.setattr(frame_zero_assets, "_slice_known_action", fake_slice)
    monkeypatch.setattr(
        frame_zero_assets,
        "decode_exact_frame_zero",
        lambda _path, *, source_aligned_frame_index: (
            np.zeros((4, 4, 3), dtype=np.uint8),
            {
                "decoded_frame_count": 1,
                "maximum_rgb_frame_read": 0,
                "source_aligned_frame_index": source_aligned_frame_index,
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


def test_action_window_is_selected_from_action_only_with_earliest_tie() -> None:
    actions = np.zeros((120, 5, 3), dtype=np.float64)
    openings = np.ones(120, dtype=np.float64)

    selected = select_action_only_window(
        actions,
        openings,
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
    for path in (
        Path(payload["bundle"]["path"]),
        Path(payload["action_inputs"]["robot_trajectory"]["path"]),
        Path(payload["action_inputs"]["robot_metadata"]["path"]),
        Path(payload["action_alignment"]["selected_action_bundle"]["path"]),
    ):
        path.write_bytes(path.name.encode("utf-8"))
    payload["bundle"] = _record_existing(Path(payload["bundle"]["path"]))
    for key in ("robot_trajectory", "robot_metadata"):
        payload["action_inputs"][key] = _record_existing(
            Path(payload["action_inputs"][key]["path"])
        )
    payload["action_alignment"]["selected_action_bundle"] = _record_existing(
        Path(payload["action_alignment"]["selected_action_bundle"]["path"])
    )
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
