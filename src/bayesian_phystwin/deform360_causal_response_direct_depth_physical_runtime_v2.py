"""Runtime custody for the 76-frame V14 physical action.

The prefix-geometry bundle intentionally contains only the 58 observable
camera-prefix frames. The physical rollout still receives the already staged
76-frame known robot action. This child lock binds that action without
changing the frozen frame-zero geometry or reading future object evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .deform360_object_exclusion import file_sha256

RUNTIME_V2_KIND = "Deform360CausalResponseDirectDepthPhysicalRuntimeV14V2"
RUNTIME_V2_CONTRACT = (
    "deform360-causal-response-direct-depth-physical-runtime-v14-v2"
)
RUNTIME_V2_PROTOCOL_ID = (
    "deform360-causal-response-direct-depth-v14-physical-runtime-v2"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-physical-runtime-v14-v2\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_v14_physical_runtime_v2(
    path: str | Path,
    *,
    parent_prelock_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the child lock that repairs the physical action source."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    _require(
        isinstance(payload, dict)
        and payload.get("schema_version") == 1
        and payload.get("artifact_kind") == RUNTIME_V2_KIND
        and payload.get("contract") == RUNTIME_V2_CONTRACT
        and payload.get("protocol_id") == RUNTIME_V2_PROTOCOL_ID
        and payload.get("status")
        == "locked_before_first_physical_carrier_execution",
        "V14 physical runtime-v2 identity changed",
    )
    _require(
        payload.get("config_sha256") == _canonical_sha256(payload),
        "V14 physical runtime-v2 checksum changed",
    )
    parent = payload.get("parent_physical_prelock")
    _require(
        isinstance(parent, Mapping)
        and _valid_digest(parent.get("config_sha256"))
        and _valid_digest(parent.get("file_sha256")),
        "V14 physical runtime-v2 parent binding is invalid",
    )
    if parent_prelock_path is not None:
        parent_path = Path(parent_prelock_path)
        parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        _require(
            parent_payload.get("config_sha256") == parent["config_sha256"]
            and file_sha256(parent_path) == parent["file_sha256"],
            "V14 physical runtime-v2 uses another physical pre-lock",
        )
    trigger = payload.get("trigger")
    _require(
        isinstance(trigger, Mapping)
        and trigger.get("prefix_geometry_robot_frame_count") == 58
        and trigger.get("required_physical_frame_count") == 76
        and trigger.get("physical_carrier_executed_before_fix") is False
        and trigger.get("method_or_gate_changed") is False,
        "V14 physical runtime-v2 trigger changed",
    )
    contract = payload.get("action_contract")
    _require(
        isinstance(contract, Mapping)
        and contract.get("known_action_source")
        == "exact_action_only_staged_window_robot"
        and contract.get("accepted_staged_frame_counts") == [76, 81]
        and contract.get("physical_frame_count") == 76
        and contract.get("object_observation_source") == "frame_zero_only",
        "V14 physical runtime-v2 action contract changed",
    )
    cases = payload.get("action_cases")
    _require(
        isinstance(cases, list)
        and len(cases) == 12
        and [record.get("queue_rank") for record in cases] == list(range(3, 15)),
        "V14 physical runtime-v2 action ledger changed",
    )
    for record in cases:
        _require(
            all(
                _valid_digest(record.get(key))
                for key in (
                    "object_hash",
                    "case_hash",
                    "window_stage_artifact_sha256",
                    "window_stage_file_sha256",
                    "known_action_file_sha256",
                )
            )
            and record.get("staged_frame_count") in {76, 81},
            "V14 physical runtime-v2 action record is invalid",
        )
    implementation = payload.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and isinstance(implementation.get("parent_commit"), str)
        and len(implementation["parent_commit"]) == 40
        and set(implementation.get("file_sha256", {}))
        == {"physical_runner", "runtime_module"}
        and all(
            _valid_digest(value)
            for value in implementation["file_sha256"].values()
        ),
        "V14 physical runtime-v2 implementation binding changed",
    )
    boundary = payload.get("information_boundary")
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("known_future_robot_action_read") is True
        and boundary.get("future_object_observation_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("future_identity_or_metric_read") is False
        and boundary.get("target_object_or_outcome_read") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 physical runtime-v2 crossed its information boundary",
    )
    return payload


def validate_v14_physical_action_v2(
    protocol: Mapping[str, Any],
    *,
    queue_rank: int,
    object_hash: str,
    case_hash: str,
    window_stage_result_path: str | Path,
    known_action_path: str | Path,
    staged_frame_count: int,
) -> dict[str, Any]:
    """Bind one full known-action archive to its hash-only ledger row."""

    record = next(
        (
            item
            for item in protocol["action_cases"]
            if int(item["queue_rank"]) == int(queue_rank)
        ),
        None,
    )
    _require(record is not None, "V14 physical action rank is outside the ledger")
    _require(
        record["object_hash"] == object_hash
        and record["case_hash"] == case_hash
        and record["window_stage_file_sha256"]
        == file_sha256(window_stage_result_path)
        and record["known_action_file_sha256"] == file_sha256(known_action_path)
        and record["staged_frame_count"] == int(staged_frame_count),
        "V14 physical action differs from its runtime-v2 ledger",
    )
    stage = json.loads(Path(window_stage_result_path).read_text(encoding="utf-8"))
    _require(
        stage.get("artifact_sha256") == record["window_stage_artifact_sha256"]
        and stage.get("status") == "staged"
        and stage.get("queue_rank") == int(queue_rank)
        and stage.get("object_hash") == object_hash
        and stage.get("case_hash") == case_hash,
        "V14 physical action uses another staged source window",
    )
    return dict(record)


__all__ = [
    "RUNTIME_V2_CONTRACT",
    "RUNTIME_V2_KIND",
    "RUNTIME_V2_PROTOCOL_ID",
    "load_v14_physical_runtime_v2",
    "validate_v14_physical_action_v2",
]
