"""Outcome-blind source admissibility for V14 causal direct depth."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .deform360_causal_response_adaptive_query import (
    ABSTAINED_ARM,
    AdaptiveCausalResponseQuerySchedule,
)
from .deform360_causal_response_preflight import (
    REGISTERED_CAMERA_IDS,
    CausalResponseSourceCameraRecord,
    deform360_object_hash,
)

CONTRACT = "deform360-causal-response-direct-depth-preflight-v14"
CASE_HASH_NAMESPACE = b"deform360-causal-response-direct-depth-case-v14\0"
BASE_SOURCE_ROLES = frozenset(
    {"metadata", "robot", "physical_geometry", "tactile"}
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
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-preflight-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def deform360_v14_case_hash(object_id: str, episode_id: int) -> str:
    """Hash one V14 source episode without retaining its plaintext identity."""

    _require(bool(str(object_id).strip()), "object ID is empty")
    _require(
        isinstance(episode_id, int)
        and not isinstance(episode_id, bool)
        and episode_id >= 0,
        "episode ID is invalid",
    )
    return hashlib.sha256(
        CASE_HASH_NAMESPACE + f"{object_id}\0{episode_id}".encode()
    ).hexdigest()


@dataclass(frozen=True)
class AdaptiveDirectDepthSourcePreflightConfigV14:
    """Frozen source and carrier contracts checked before source locking."""

    required_frame_count: int = 76
    prefix_frame_count: int = 58
    minimum_physical_node_count: int = 128
    maximum_physical_node_count: int = 10_000
    minimum_complete_camera_count: int = 8
    registered_camera_ids: tuple[str, ...] = REGISTERED_CAMERA_IDS

    def __post_init__(self) -> None:
        _require(
            self.required_frame_count > self.prefix_frame_count >= 2,
            "V14 source frame contract is invalid",
        )
        _require(
            self.minimum_physical_node_count >= 3
            and self.maximum_physical_node_count >= self.minimum_physical_node_count,
            "V14 physical node contract is invalid",
        )
        _require(
            self.minimum_complete_camera_count == 8
            and len(self.registered_camera_ids)
            >= self.minimum_complete_camera_count
            and len(set(self.registered_camera_ids))
            == len(self.registered_camera_ids),
            "V14 registered camera contract is invalid",
        )


@dataclass(frozen=True)
class AdaptiveDirectDepthSourcePreflightV14:
    """Checksummed hash-only V14 source disposition."""

    config: AdaptiveDirectDepthSourcePreflightConfigV14
    object_hash: str
    case_hash: str
    category: str
    bimanual_value: str
    episode_frame_count: int
    robot_frame_count: int
    tactile_frame_count: int
    physical_node_count: int
    camera_records: tuple[CausalResponseSourceCameraRecord, ...]
    complete_camera_ids: tuple[str, ...]
    carrier_artifact_sha256: str
    carrier_arm: str
    source_sha256: dict[str, str]
    admitted: bool
    rejection_reasons: tuple[str, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(
            _valid_digest(self.object_hash)
            and _valid_digest(self.case_hash)
            and _valid_digest(self.carrier_artifact_sha256)
            and _valid_digest(self.artifact_sha256),
            "V14 preflight digest is invalid",
        )
        _require(bool(self.category.strip()), "V14 category is empty")
        _require(
            self.episode_frame_count >= 0
            and self.robot_frame_count >= 0
            and self.tactile_frame_count >= 0
            and self.physical_node_count >= 0,
            "V14 source count is negative",
        )
        record_ids = tuple(record.camera_id for record in self.camera_records)
        _require(
            record_ids == tuple(sorted(set(record_ids))),
            "V14 camera records are duplicated or unsorted",
        )
        _require(
            self.complete_camera_ids
            == tuple(sorted(set(self.complete_camera_ids)))
            and set(self.complete_camera_ids).issubset(record_ids),
            "V14 complete camera set is invalid",
        )
        sources = dict(sorted(self.source_sha256.items()))
        _require(
            sources
            and all(
                bool(role) and _valid_digest(digest)
                for role, digest in sources.items()
            ),
            "V14 source provenance is invalid",
        )
        _require(
            self.rejection_reasons
            == tuple(sorted(set(self.rejection_reasons))),
            "V14 rejection reasons are duplicated or unsorted",
        )
        _require(
            self.admitted == (not self.rejection_reasons),
            "V14 preflight decision differs from its reasons",
        )
        object.__setattr__(self, "source_sha256", sources)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360AdaptiveDirectDepthSourcePreflightV14",
            "contract": CONTRACT,
            "config": asdict(self.config),
            "object_hash": self.object_hash,
            "case_hash": self.case_hash,
            "category": self.category,
            "bimanual_value": self.bimanual_value,
            "episode_frame_count": self.episode_frame_count,
            "robot_frame_count": self.robot_frame_count,
            "tactile_frame_count": self.tactile_frame_count,
            "physical_node_count": self.physical_node_count,
            "camera_records": [asdict(record) for record in self.camera_records],
            "complete_camera_ids": list(self.complete_camera_ids),
            "complete_camera_count": len(self.complete_camera_ids),
            "carrier_artifact_sha256": self.carrier_artifact_sha256,
            "carrier_arm": self.carrier_arm,
            "source_sha256": dict(self.source_sha256),
            "admitted": self.admitted,
            "rejection_reasons": list(self.rejection_reasons),
            "information_boundary": {
                "metadata_enums_read": True,
                "stream_lengths_read": True,
                "frame_zero_geometry_count_read": True,
                "frame_zero_camera_support_read": True,
                "frame_zero_adaptive_carrier_read": True,
                "prefix_or_future_object_payload_deserialized": False,
                "future_identity_or_metric_read": False,
                "plaintext_object_or_episode_identity_retained": False,
                "held_v8_artifact_or_process_access": False,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def evaluate_adaptive_direct_depth_source_preflight_v14(
    *,
    object_id: str,
    episode_id: int,
    category: str,
    bimanual_value: str,
    episode_frame_count: int,
    robot_frame_count: int,
    tactile_frame_count: int,
    physical_node_count: int,
    camera_records: Iterable[CausalResponseSourceCameraRecord],
    carrier: AdaptiveCausalResponseQuerySchedule,
    source_sha256: Mapping[str, str],
    config: AdaptiveDirectDepthSourcePreflightConfigV14 | None = None,
) -> AdaptiveDirectDepthSourcePreflightV14:
    """Check source and adaptive-carrier contracts without outcome access."""

    cfg = config or AdaptiveDirectDepthSourcePreflightConfigV14()
    records = tuple(sorted(camera_records, key=lambda record: record.camera_id))
    sources = dict(sorted(source_sha256.items()))
    _require(
        sources
        and all(
            bool(role) and _valid_digest(digest)
            for role, digest in sources.items()
        ),
        "V14 source checksums are malformed",
    )
    reasons: list[str] = []
    if bimanual_value not in {"yes", "no"}:
        reasons.append("invalid-bimanual-enum")
    for name, count in (
        ("episode", episode_frame_count),
        ("robot", robot_frame_count),
        ("tactile", tactile_frame_count),
    ):
        if count != cfg.required_frame_count:
            reasons.append(f"{name}-frame-count-mismatch")
    if not (
        cfg.minimum_physical_node_count
        <= physical_node_count
        <= cfg.maximum_physical_node_count
    ):
        reasons.append("physical-backend-node-count")
    if not BASE_SOURCE_ROLES.issubset(sources):
        reasons.append("required-base-source-checksum-missing")

    record_by_camera = {record.camera_id: record for record in records}
    if set(record_by_camera) != set(cfg.registered_camera_ids):
        reasons.append("registered-camera-record-set-mismatch")
    complete: list[str] = []
    for camera in cfg.registered_camera_ids:
        record = record_by_camera.get(camera)
        if record is None:
            continue
        camera_roles = {
            f"depth/{camera}",
            f"mask/{camera}",
            f"calibration/{camera}",
        }
        if (
            record.depth_frame_count == cfg.required_frame_count
            and record.mask_frame_count == cfg.required_frame_count
            and record.calibration_valid
            and record.frame_zero_projected_support_count > 0
            and camera_roles.issubset(sources)
        ):
            complete.append(camera)
    complete_ids = tuple(sorted(complete))
    if len(complete_ids) < cfg.minimum_complete_camera_count:
        reasons.append("insufficient-complete-camera-count")
    if set(carrier.available_camera_ids) != set(complete_ids):
        reasons.append("carrier-available-camera-set-mismatch")
    if not carrier.admitted or carrier.arm == ABSTAINED_ARM:
        reasons.append("adaptive-carrier-abstained")
    if not set(carrier.selected_camera_ids).issubset(complete_ids):
        reasons.append("adaptive-carrier-uses-incomplete-camera")

    provisional = AdaptiveDirectDepthSourcePreflightV14(
        config=cfg,
        object_hash=deform360_object_hash(object_id),
        case_hash=deform360_v14_case_hash(object_id, episode_id),
        category=str(category),
        bimanual_value=str(bimanual_value),
        episode_frame_count=int(episode_frame_count),
        robot_frame_count=int(robot_frame_count),
        tactile_frame_count=int(tactile_frame_count),
        physical_node_count=int(physical_node_count),
        camera_records=records,
        complete_camera_ids=complete_ids,
        carrier_artifact_sha256=carrier.artifact_sha256,
        carrier_arm=carrier.arm,
        source_sha256=sources,
        admitted=not reasons,
        rejection_reasons=tuple(sorted(set(reasons))),
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor())
    result = AdaptiveDirectDepthSourcePreflightV14(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )
    validate_adaptive_direct_depth_source_preflight_v14(result)
    return result


def validate_adaptive_direct_depth_source_preflight_v14(
    artifact: AdaptiveDirectDepthSourcePreflightV14,
) -> None:
    """Validate one sealed V14 source disposition."""

    _require(
        _canonical_sha256(artifact.descriptor()) == artifact.artifact_sha256,
        "V14 source preflight checksum changed",
    )
    boundary = artifact.descriptor()["information_boundary"]
    _require(
        boundary["prefix_or_future_object_payload_deserialized"] is False
        and boundary["future_identity_or_metric_read"] is False
        and boundary["plaintext_object_or_episode_identity_retained"] is False
        and boundary["held_v8_artifact_or_process_access"] is False,
        "V14 source preflight crossed its information boundary",
    )


def write_adaptive_direct_depth_source_preflight_v14(
    path: str | Path,
    artifact: AdaptiveDirectDepthSourcePreflightV14,
) -> None:
    """Write one immutable V14 source disposition."""

    validate_adaptive_direct_depth_source_preflight_v14(artifact)
    output = Path(path)
    _require(not output.exists(), "V14 source preflight already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            artifact.descriptor(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BASE_SOURCE_ROLES",
    "CASE_HASH_NAMESPACE",
    "CONTRACT",
    "AdaptiveDirectDepthSourcePreflightConfigV14",
    "AdaptiveDirectDepthSourcePreflightV14",
    "deform360_v14_case_hash",
    "evaluate_adaptive_direct_depth_source_preflight_v14",
    "validate_adaptive_direct_depth_source_preflight_v14",
    "write_adaptive_direct_depth_source_preflight_v14",
]
