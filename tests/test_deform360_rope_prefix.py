from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import causal4d_public.deform360_rope_prefix as module
from causal4d_public.deform360_rope_prefix import (
    rope_prefix_geometry_artifact_sha256,
    validate_target_prefix_rope_geometry,
)


def test_target_prefix_geometry_artifact_binds_archive(tmp_path: Path) -> None:
    archive = tmp_path / "prefix.npz"
    centerlines = np.zeros((6, 11, 3), dtype=np.float64)
    frame_indices = np.arange(103, 109, dtype=np.int32)
    np.savez_compressed(archive, frame_indices=frame_indices, centerlines_m=centerlines)
    payload = {
        "schema_version": 2,
        "artifact_kind": "Deform360TargetPrefixRopeGeometry",
        "frame_indices": frame_indices.tolist(),
        "quality": {"passed": True},
        "archive": {
            "path": str(archive),
            "sha256": module._sha256_file(archive),
            "centerlines_sha256": module._sha256_array(centerlines),
        },
        "information_boundary": {
            "target_future_visual_frames_read": False,
            "target_tactile_oracle_read": False,
        },
    }
    payload["result_sha256"] = rope_prefix_geometry_artifact_sha256(payload)

    assert validate_target_prefix_rope_geometry(payload)["prefix_end_frame"] == 108
    payload["information_boundary"]["target_future_visual_frames_read"] = True
    payload["result_sha256"] = rope_prefix_geometry_artifact_sha256(payload)
    with pytest.raises(ValueError, match="future visual"):
        validate_target_prefix_rope_geometry(payload, verify_archive=False)
