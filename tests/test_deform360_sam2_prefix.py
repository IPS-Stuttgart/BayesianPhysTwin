from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from causal4d_public.deform360 import (
    DEFORM360_PREFLIGHT_SCHEMA_VERSION,
    load_deform360_protocol_config,
    preflight_result_sha256,
)
from causal4d_public.deform360_contact import contact_artifact_sha256
from causal4d_public.deform360_sam2 import RopeSam2MaskConfig
from causal4d_public.deform360_sam2_prefix import (
    build_sam2_prefix_mask_audit,
    decode_video_frame_window,
    select_source_locked_prefix_cameras,
    target_prefix_bounds,
    validate_sam2_prefix_mask_artifact,
)
from causal4d_public.deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    build_sam2_view_audit,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_001_rope_v1.json"
    )


def _contact_seal(*, start: int = 103) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360TargetContactPredictionSeal",
        "protocol_id": "causal4d-deform360-001-rope-v1",
        "contact_model_sha256": "a" * 64,
        "target_episode_id": "001-rope/episode_0006",
        "information_boundary": {
            "target_tactile_oracle_read": False,
        },
        "target_prefix": {
            "start_frame": start,
            "stop_frame_exclusive": start + 6,
            "frame_count": 6,
        },
    }
    payload["result_sha256"] = contact_artifact_sha256(payload)
    return payload


def _source_view_audit() -> dict[str, object]:
    consistency = {
        "accepted_cameras": ["cam0", "cam1", "cam2"],
        "rejected_cameras": [],
    }
    return build_sam2_view_audit(
        protocol_id="causal4d-deform360-001-rope-v1",
        episode_access={
            "episode_index": 0,
            "split": "source",
            "target_future_annotation_unlocked": False,
            "held_out_prediction_seal_sha256": None,
        },
        automatic_view_diagnostics=[],
        consistency=consistency,
        reliability_config=CrossViewMaskReliabilityConfig(voxel_resolution=16),
    )


def _preflight() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": DEFORM360_PREFLIGHT_SCHEMA_VERSION,
        "artifact_kind": "Deform360001RopePreflight",
        "protocol_id": "causal4d-deform360-001-rope-v1",
        "information_boundary": {
            "prediction_metrics_computed": False,
            "model_parameters_fitted": False,
            "split_uses_metadata_only": True,
        },
        "preflight_passed": True,
        "split": {"held_out_action": "move both edges"},
        "processed_episodes": [
            {
                "episode_id": "001-rope/episode_0000",
                "alignment": {
                    "quality": {
                        "cameras": [
                            {"camera": "cam0", "synchronization_reliability": 0.96},
                            {"camera": "cam1", "synchronization_reliability": 0.91},
                            {"camera": "cam2", "synchronization_reliability": 0.70},
                        ]
                    }
                },
            }
        ],
    }
    payload["result_sha256"] = preflight_result_sha256(payload)
    return payload


def test_target_prefix_bounds_are_derived_from_the_contact_seal() -> None:
    config = load_deform360_protocol_config(_config_path())
    seal = _contact_seal()

    assert target_prefix_bounds(config, seal) == (103, 109)

    seal["target_prefix"]["stop_frame_exclusive"] = 110  # type: ignore[index]
    seal["result_sha256"] = contact_artifact_sha256(seal)
    with pytest.raises(ValueError, match="length differs"):
        target_prefix_bounds(config, seal)


def test_source_locked_camera_policy_filters_only_by_source_timing() -> None:
    policy = select_source_locked_prefix_cameras(
        _source_view_audit(),
        _preflight(),
        minimum_synchronization_reliability=0.85,
        minimum_camera_count=2,
    )

    assert policy["selection_scope"] == "source-only"
    assert policy["selected_cameras"] == ["cam0", "cam1"]
    assert policy["rejected_for_synchronization"] == [
        {"camera": "cam2", "reliability": 0.70}
    ]


def test_video_window_decoder_never_reads_the_suffix() -> None:
    cv2 = pytest.importorskip("cv2")

    class FakeCapture:
        def __init__(self) -> None:
            self.next_frame = 0
            self.read_count = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def set(self, prop: int, value: float) -> bool:
            assert prop == cv2.CAP_PROP_POS_FRAMES
            self.next_frame = int(value)
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            value = self.next_frame
            self.next_frame += 1
            self.read_count += 1
            return True, np.full((2, 3, 3), value, dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    frames = decode_video_frame_window(
        "target.mp4",
        103,
        109,
        capture_factory=lambda _: capture,
    )

    assert frames.shape == (6, 2, 3, 3)
    assert capture.read_count == 6
    assert capture.next_frame == 109
    assert capture.released is True
    assert frames[:, 0, 0, 0].tolist() == list(range(103, 109))


def test_prefix_mask_audit_is_checksummed_and_future_locked(tmp_path: Path) -> None:
    config = load_deform360_protocol_config(_config_path())
    seal = _contact_seal()
    mask = tmp_path / "cam0.npy"
    np.save(mask, np.ones((6, 3, 4), dtype=np.uint8), allow_pickle=False)
    predictor = SimpleNamespace(
        config=RopeSam2MaskConfig(),
        model_id="prefix-fixture",
    )
    policy = {
        "selection_scope": "source-only",
        "selected_cameras": ["cam0"],
    }
    outputs = [
        {
            "camera": "cam0",
            "source_video_fully_hashed": False,
            "decoded_frame_indices": list(range(103, 109)),
            "mask_path": str(mask),
            "mask_sha256": "b" * 64,
        }
    ]

    artifact = build_sam2_prefix_mask_audit(
        config=config,
        contact_prediction_seal=seal,
        camera_policy=policy,
        predictor=predictor,
        camera_outputs=outputs,
        minimum_camera_count=1,
    )

    assert validate_sam2_prefix_mask_artifact(artifact)["passed"] is True
    assert artifact["target_prefix"] == {
        "start_frame": 103,
        "stop_frame_exclusive": 109,
        "frame_count": 6,
    }
    assert artifact["information_boundary"]["target_future_visual_frames_read"] is False
    artifact["information_boundary"]["target_future_visual_frames_read"] = True
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_sam2_prefix_mask_artifact(artifact)


def test_prefix_mask_audit_records_deterministic_camera_failure(
    tmp_path: Path,
) -> None:
    config = load_deform360_protocol_config(_config_path())
    mask = tmp_path / "cam0.npy"
    np.save(mask, np.ones((6, 3, 4), dtype=np.uint8), allow_pickle=False)
    predictor = SimpleNamespace(config=RopeSam2MaskConfig(), model_id="fixture")
    artifact = build_sam2_prefix_mask_audit(
        config=config,
        contact_prediction_seal=_contact_seal(),
        camera_policy={
            "selection_scope": "source-only",
            "selected_cameras": ["cam0", "cam1"],
        },
        predictor=predictor,
        camera_outputs=[
            {
                "camera": "cam0",
                "source_video_fully_hashed": False,
                "decoded_frame_indices": list(range(103, 109)),
                "mask_path": str(mask),
                "mask_sha256": "b" * 64,
            }
        ],
        camera_failures=[
            {"camera": "cam1", "reason": "RuntimeError", "message": "no mask"}
        ],
        minimum_camera_count=1,
    )

    assert artifact["camera_policy"]["selected_cameras"] == ["cam0"]
    assert artifact["camera_policy"]["source_locked_selected_cameras"] == [
        "cam0",
        "cam1",
    ]
    assert (
        artifact["information_boundary"][
            "camera_selection_used_target_prefix_measurement_availability"
        ]
        is True
    )
