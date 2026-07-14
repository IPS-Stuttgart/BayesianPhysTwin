from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from causal4d_public.deform360 import load_deform360_protocol_config
from causal4d_public.deform360_sam2 import (
    PINNED_SAM2_CHECKPOINT_SHA256,
    PINNED_SAM2_COMMIT,
    RopeSam2MaskConfig,
    build_sam2_mask_audit,
    rope_mask_candidate_diagnostics,
    validate_sam2_episode_access,
    validate_sam2_mask_artifact,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_001_rope_v1.json"
    )


def test_source_masks_do_not_require_a_prediction_seal() -> None:
    config = load_deform360_protocol_config(_config_path())

    access = validate_sam2_episode_access(
        0,
        config,
        held_out_prediction_seal_sha256=None,
    )

    assert access["split"] == "source"
    assert access["target_future_annotation_unlocked"] is False


def test_target_masks_require_a_prediction_seal() -> None:
    config = load_deform360_protocol_config(_config_path())

    with pytest.raises(ValueError, match="full target masks"):
        validate_sam2_episode_access(
            6,
            config,
            held_out_prediction_seal_sha256=None,
        )

    access = validate_sam2_episode_access(
        6,
        config,
        held_out_prediction_seal_sha256="a" * 64,
    )
    assert access["split"] == "target"
    assert access["target_future_annotation_unlocked"] is True


def test_source_masks_reject_a_target_prediction_seal() -> None:
    config = load_deform360_protocol_config(_config_path())

    with pytest.raises(ValueError, match="only valid for a target"):
        validate_sam2_episode_access(
            1,
            config,
            held_out_prediction_seal_sha256="b" * 64,
        )


def test_elongated_two_color_candidate_beats_compact_candidate() -> None:
    pytest.importorskip("cv2")
    config = RopeSam2MaskConfig(minimum_colored_pixels_per_family=20)
    rgb = np.zeros((80, 140, 3), dtype=np.uint8)
    rgb[34:46, 10:70] = (255, 0, 180)
    rgb[34:46, 70:130] = (0, 255, 0)
    elongated = np.zeros(rgb.shape[:2], dtype=bool)
    elongated[30:50, 5:135] = True
    compact = np.zeros_like(elongated)
    compact[25:55, 50:90] = True

    elongated_result = rope_mask_candidate_diagnostics(rgb, elongated, config)
    compact_result = rope_mask_candidate_diagnostics(rgb, compact, config)

    assert elongated_result["eligible"] is True
    assert compact_result["eligible"] is True
    assert elongated_result["elongation"] > compact_result["elongation"]
    assert elongated_result["score"] > compact_result["score"]


def test_mask_audit_is_pinned_and_checksummed(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.h5"
    mask_path.write_bytes(b"source-only-mask-fixture")
    predictor = SimpleNamespace(
        config=RopeSam2MaskConfig(),
        model_id=f"fixture@{PINNED_SAM2_COMMIT[:12]}",
        diagnostics=[{"camera": "fixture_cam0"}],
    )
    access = {
        "episode_index": 0,
        "split": "source",
        "target_future_annotation_unlocked": False,
        "held_out_prediction_seal_sha256": None,
    }

    artifact = build_sam2_mask_audit(
        protocol_id="fixture",
        episode_access=access,
        predictor=predictor,
        output_paths={"fixture_cam0": mask_path},
        view_audit_result_sha256="c" * 64,
    )

    assert validate_sam2_mask_artifact(artifact)["passed"] is True
    assert artifact["upstream"]["commit"] == PINNED_SAM2_COMMIT
    assert artifact["upstream"]["checkpoint_sha256"] == PINNED_SAM2_CHECKPOINT_SHA256
    assert artifact["view_selection"]["cross_view_gate_applied"] is True
    artifact["model_id"] = "tampered"
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_sam2_mask_artifact(artifact)
