from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import causal4d_public.deform360_rope_future as module
from causal4d_public.deform360_rope_future import (
    rope_future_geometry_artifact_sha256,
    validate_target_future_rope_geometry,
)


def test_future_geometry_artifact_requires_prior_seal(tmp_path: Path) -> None:
    archive = tmp_path / "future.npz"
    centerlines = np.zeros((8, 11, 3), dtype=np.float64)
    np.savez_compressed(
        archive,
        frame_indices=np.arange(109, 117, dtype=np.int32),
        centerlines_m=centerlines,
    )
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360TargetFutureRopeGeometry",
        "frame_indices": list(range(109, 117)),
        "quality": {"passed": True},
        "archive": {
            "path": str(archive),
            "sha256": module._sha256_file(archive),
            "centerlines_sha256": module._sha256_array(centerlines),
        },
        "information_boundary": {
            "deployable_predictions_previously_sealed": True,
            "target_future_geometry_used_for_fitting": False,
        },
    }
    payload["result_sha256"] = rope_future_geometry_artifact_sha256(payload)

    assert validate_target_future_rope_geometry(payload)["frame_count"] == 8
    payload["information_boundary"]["deployable_predictions_previously_sealed"] = False
    payload["result_sha256"] = rope_future_geometry_artifact_sha256(payload)
    with pytest.raises(ValueError, match="before prediction sealing"):
        validate_target_future_rope_geometry(payload, verify_archive=False)
