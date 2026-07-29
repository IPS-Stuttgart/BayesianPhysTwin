from __future__ import annotations

import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    REQUIRED_SOURCE_ROLES,
    CausalResponseSourceCameraRecord,
    evaluate_causal_response_source_preflight,
    load_causal_response_source_preflight,
    write_causal_response_source_preflight,
)


def _camera_records() -> tuple[CausalResponseSourceCameraRecord, ...]:
    return tuple(
        CausalResponseSourceCameraRecord(
            camera_id=camera,
            depth_frame_count=76,
            mask_frame_count=76,
            calibration_valid=True,
            frame_zero_projected_support_count=32,
        )
        for camera in REGISTERED_CAMERA_IDS
    )


def _source_sha256() -> dict[str, str]:
    return {
        role: f"{index + 1:064x}"
        for index, role in enumerate(sorted(REQUIRED_SOURCE_ROLES))
    }


def _preflight(**overrides):
    arguments = {
        "object_id": "001-source-object",
        "episode_id": 3,
        "category": "cloth",
        "bimanual_value": "no",
        "episode_frame_count": 76,
        "robot_frame_count": 76,
        "tactile_frame_count": 76,
        "physical_node_count": 256,
        "camera_records": _camera_records(),
        "source_sha256": _source_sha256(),
    }
    arguments.update(overrides)
    return evaluate_causal_response_source_preflight(**arguments)


def test_admitted_source_preflight_is_hash_only_and_round_trips(
    tmp_path: Path,
) -> None:
    artifact = _preflight()
    path = tmp_path / "source_preflight.json"

    write_causal_response_source_preflight(path, artifact)
    loaded = load_causal_response_source_preflight(path)
    encoded = json.dumps(loaded.descriptor(), sort_keys=True)

    assert loaded.admitted
    assert loaded.descriptor() == artifact.descriptor()
    assert "001-source-object" not in encoded
    assert '"episode_id"' not in encoded
    assert (
        loaded.descriptor()["information_boundary"][
            "future_object_payload_deserialized"
        ]
        is False
    )


def test_preflight_rejects_known_source_and_backend_contract_failures() -> None:
    records = list(_camera_records())
    records[0] = CausalResponseSourceCameraRecord(
        camera_id=records[0].camera_id,
        depth_frame_count=75,
        mask_frame_count=76,
        calibration_valid=True,
        frame_zero_projected_support_count=32,
    )

    artifact = _preflight(
        bimanual_value="yess",
        robot_frame_count=75,
        physical_node_count=54,
        camera_records=tuple(records),
    )

    assert not artifact.admitted
    assert set(artifact.rejection_reasons) >= {
        "invalid-bimanual-enum",
        "robot-frame-count-mismatch",
        "physical-backend-node-count",
        "proposal-camera-panel-inadmissible",
    }


def test_preflight_rejects_an_incomplete_source_checksum_set() -> None:
    sources = _source_sha256()
    del sources[f"depth/{REGISTERED_CAMERA_IDS[0]}"]

    artifact = _preflight(source_sha256=sources)

    assert not artifact.admitted
    assert "required-source-checksum-missing" in artifact.rejection_reasons


def test_preflight_detects_persisted_tampering(tmp_path: Path) -> None:
    artifact = _preflight()
    path = tmp_path / "source_preflight.json"
    write_causal_response_source_preflight(path, artifact)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["physical_node_count"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_causal_response_source_preflight(path)
