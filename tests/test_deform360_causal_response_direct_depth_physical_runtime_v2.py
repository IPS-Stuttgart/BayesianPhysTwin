from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bayesian_phystwin.deform360_causal_response_direct_depth_physical_runtime_v2 import (
    RUNTIME_V2_CONTRACT,
    RUNTIME_V2_KIND,
    RUNTIME_V2_PROTOCOL_ID,
    load_v14_physical_runtime_v2,
    validate_v14_physical_action_v2,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_protocol(
    path: Path,
    *,
    parent_path: Path,
    stage_path: Path,
    action_path: Path,
) -> dict:
    stage = json.loads(stage_path.read_text())
    payload = {
        "schema_version": 1,
        "artifact_kind": RUNTIME_V2_KIND,
        "contract": RUNTIME_V2_CONTRACT,
        "protocol_id": RUNTIME_V2_PROTOCOL_ID,
        "status": "locked_before_first_physical_carrier_execution",
        "parent_physical_prelock": {
            "config_sha256": json.loads(parent_path.read_text())["config_sha256"],
            "file_sha256": file_sha256(parent_path),
        },
        "trigger": {
            "prefix_geometry_robot_frame_count": 58,
            "required_physical_frame_count": 76,
            "physical_carrier_executed_before_fix": False,
            "method_or_gate_changed": False,
        },
        "action_contract": {
            "known_action_source": "exact_action_only_staged_window_robot",
            "accepted_staged_frame_counts": [76, 81],
            "physical_frame_count": 76,
            "object_observation_source": "frame_zero_only",
        },
        "action_cases": [
            {
                "queue_rank": rank,
                "object_hash": _digest(f"object-{rank}"),
                "case_hash": _digest(f"case-{rank}"),
                "window_stage_artifact_sha256": (
                    stage["artifact_sha256"]
                    if rank == 3
                    else _digest(f"stage-artifact-{rank}")
                ),
                "window_stage_file_sha256": (
                    file_sha256(stage_path)
                    if rank == 3
                    else _digest(f"stage-file-{rank}")
                ),
                "known_action_file_sha256": (
                    file_sha256(action_path)
                    if rank == 3
                    else _digest(f"action-{rank}")
                ),
                "staged_frame_count": 81,
            }
            for rank in range(3, 15)
        ],
        "implementation": {
            "parent_commit": "a" * 40,
            "file_sha256": {
                "physical_runner": "b" * 64,
                "runtime_module": "c" * 64,
            },
        },
        "information_boundary": {
            "known_future_robot_action_read": True,
            "future_object_observation_read": False,
            "future_tactile_read": False,
            "future_identity_or_metric_read": False,
            "target_object_or_outcome_read": False,
            "held_v8_artifact_or_process_access": False,
        },
    }
    canonical = dict(payload)
    payload["config_sha256"] = hashlib.sha256(
        b"deform360-causal-response-direct-depth-physical-runtime-v14-v2\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_physical_runtime_v2_binds_the_full_staged_action(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"config_sha256": "d" * 64}))
    action = tmp_path / "robot.npz"
    action.write_bytes(b"known 81-frame action")
    stage = tmp_path / "stage.json"
    stage.write_text(
        json.dumps(
            {
                "artifact_sha256": "e" * 64,
                "status": "staged",
                "queue_rank": 3,
                "object_hash": _digest("object-3"),
                "case_hash": _digest("case-3"),
            }
        )
    )
    protocol_path = tmp_path / "runtime.json"
    expected = _write_protocol(
        protocol_path,
        parent_path=parent,
        stage_path=stage,
        action_path=action,
    )

    protocol = load_v14_physical_runtime_v2(
        protocol_path,
        parent_prelock_path=parent,
    )
    record = validate_v14_physical_action_v2(
        protocol,
        queue_rank=3,
        object_hash=_digest("object-3"),
        case_hash=_digest("case-3"),
        window_stage_result_path=stage,
        known_action_path=action,
        staged_frame_count=81,
    )

    assert protocol == expected
    assert record["known_action_file_sha256"] == file_sha256(action)


def test_physical_runtime_v2_rejects_the_prefix_robot(tmp_path: Path) -> None:
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"config_sha256": "d" * 64}))
    action = tmp_path / "robot.npz"
    action.write_bytes(b"known 81-frame action")
    prefix_action = tmp_path / "prefix_robot.npz"
    prefix_action.write_bytes(b"only 58 frames")
    stage = tmp_path / "stage.json"
    stage.write_text(
        json.dumps(
            {
                "artifact_sha256": "e" * 64,
                "status": "staged",
                "queue_rank": 3,
                "object_hash": _digest("object-3"),
                "case_hash": _digest("case-3"),
            }
        )
    )
    protocol_path = tmp_path / "runtime.json"
    _write_protocol(
        protocol_path,
        parent_path=parent,
        stage_path=stage,
        action_path=action,
    )
    protocol = load_v14_physical_runtime_v2(protocol_path)

    with pytest.raises(ValueError, match="action differs"):
        validate_v14_physical_action_v2(
            protocol,
            queue_rank=3,
            object_hash=_digest("object-3"),
            case_hash=_digest("case-3"),
            window_stage_result_path=stage,
            known_action_path=prefix_action,
            staged_frame_count=58,
        )
