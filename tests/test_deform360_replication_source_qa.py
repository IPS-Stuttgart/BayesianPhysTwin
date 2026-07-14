from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_replication_source_qa import (
    load_source_qa_policy,
    select_diverse_cameras,
    source_qa_artifact_sha256,
    validate_source_qa_artifact,
)


def _transform(x: float, y: float, z: float) -> np.ndarray:
    value = np.eye(4)
    value[:3, 3] = (x, y, z)
    return value


def test_canonical_source_qa_policy_is_locked() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_replication_source_qa_v1.json"
    )

    policy = load_source_qa_policy(path)

    assert policy["config"]["selected_camera_count"] == 12


def test_diverse_camera_selection_keeps_reference_and_is_deterministic() -> None:
    extrinsics = {
        "a": _transform(0.0, 0.0, 0.0),
        "b": _transform(1.0, 0.0, 0.0),
        "c": _transform(0.0, 1.0, 0.0),
        "d": _transform(0.0, 0.0, 1.0),
        "e": _transform(0.1, 0.1, 0.1),
    }

    selected = select_diverse_cameras(
        ["e", "d", "c", "b", "a"],
        extrinsics,
        reference_camera="a",
        selected_count=4,
    )

    assert selected == ["a", "d", "c", "b"]


def test_source_qa_artifact_rejects_tampering() -> None:
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReplicationSourceGeometryQa",
        "objects": [{"object_id": "fixture"}],
        "passed": True,
        "information_boundary": {
            "source_first_frames_only": True,
            "target_media_read": False,
            "target_metrics_computed": False,
        },
    }
    payload["result_sha256"] = source_qa_artifact_sha256(payload)

    assert validate_source_qa_artifact(payload)["passed"] is True
    payload["passed"] = False
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source_qa_artifact(payload)
