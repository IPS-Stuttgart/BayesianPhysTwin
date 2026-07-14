from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_rope_sequence import (
    RopeCenterlineSequenceConfig,
    rope_sequence_artifact_sha256,
    validate_rope_sequence_artifact,
)


def test_sequence_config_rejects_an_empty_interval() -> None:
    with pytest.raises(ValueError, match="stop must exceed"):
        RopeCenterlineSequenceConfig(frame_start=10, frame_stop_exclusive=10)


def test_source_sequence_artifact_verifies_its_archive(tmp_path: Path) -> None:
    archive = tmp_path / "sequence.npz"
    frame_indices = np.asarray((0, 2, 4), dtype=np.int32)
    centerlines = np.zeros((3, 7, 3), dtype=np.float64)
    np.savez_compressed(
        archive,
        frame_indices=frame_indices,
        centerlines_m=centerlines,
    )
    import causal4d_public.deform360_rope_sequence as module

    payload = {
        "schema_version": 5,
        "artifact_kind": "Deform360SourceRopeCenterlineSequence",
        "split": "source",
        "frame_indices": frame_indices.tolist(),
        "archive": {
            "path": str(archive),
            "sha256": module._sha256_file(archive),
            "centerlines_sha256": module._sha256_array(centerlines),
        },
        "information_boundary": {"target_files_read": False},
        "quality": {"passed": True},
    }
    payload["result_sha256"] = rope_sequence_artifact_sha256(payload)

    assert validate_rope_sequence_artifact(payload)["frame_count"] == 3
    payload["split"] = "target"
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_rope_sequence_artifact(payload, verify_archive=False)
