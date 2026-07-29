"""Outcome-blind source admissibility for the V12 Deform360 experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTRACT = "deform360-causal-response-source-preflight-v12"
OBJECT_HASH_NAMESPACE = b"deform360-fresh-object-exclusion-v1\0"
CASE_HASH_NAMESPACE = b"deform360-causal-response-case-v12\0"
REGISTERED_CAMERA_IDS = (
    "brics-odroid-001_cam0",
    "brics-odroid-006_cam0",
    "brics-odroid-007_cam0",
    "brics-odroid-008_cam0",
    "brics-odroid-010_cam0",
    "brics-odroid-013_cam0",
    "brics-odroid-014_cam1",
    "brics-odroid-015_cam1",
    "brics-odroid-019_cam1",
    "brics-odroid-021_cam1",
    "brics-odroid-024_cam1",
    "brics-odroid-027_cam0",
)
PROPOSAL_CAMERA_IDS = REGISTERED_CAMERA_IDS[::2]
VALIDATION_CAMERA_IDS = REGISTERED_CAMERA_IDS[1::2]
REQUIRED_SOURCE_ROLES = frozenset(
    {
        "metadata",
        "robot",
        "physical_geometry",
        "tactile",
    }
    | {
        f"{modality}/{camera}"
        for camera in REGISTERED_CAMERA_IDS
        for modality in ("depth", "mask", "calibration")
    }
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
        b"deform360-causal-response-source-preflight-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def deform360_object_hash(object_id: str) -> str:
    """Hash a physical object in the shared fresh-object namespace."""

    _require(bool(str(object_id).strip()), "object ID is empty")
    return hashlib.sha256(
        OBJECT_HASH_NAMESPACE + str(object_id).encode("utf-8")
    ).hexdigest()


def deform360_case_hash(object_id: str, episode_id: int) -> str:
    """Hash one episode without retaining its plaintext identity."""

    _require(bool(str(object_id).strip()), "object ID is empty")
    _require(
        isinstance(episode_id, int)
        and not isinstance(episode_id, bool)
        and episode_id >= 0,
        "episode ID is invalid",
    )
    identity = f"{object_id}\0{episode_id}".encode()
    return hashlib.sha256(CASE_HASH_NAMESPACE + identity).hexdigest()


@dataclass(frozen=True)
class CausalResponseSourcePreflightConfig:
    """Frozen source contracts checked before V12 prediction."""

    required_frame_count: int = 76
    prefix_frame_count: int = 58
    minimum_physical_node_count: int = 128
    maximum_physical_node_count: int = 10_000
    registered_camera_ids: tuple[str, ...] = REGISTERED_CAMERA_IDS
    proposal_camera_ids: tuple[str, ...] = PROPOSAL_CAMERA_IDS
    validation_camera_ids: tuple[str, ...] = VALIDATION_CAMERA_IDS
    require_full_registered_camera_panel: bool = True

    def __post_init__(self) -> None:
        registered_ids = tuple(self.registered_camera_ids)
        proposal_ids = tuple(self.proposal_camera_ids)
        validation_ids = tuple(self.validation_camera_ids)
        _require(
            self.required_frame_count > self.prefix_frame_count >= 2,
            "source frame contract is invalid",
        )
        _require(
            self.minimum_physical_node_count >= 3
            and self.maximum_physical_node_count >= self.minimum_physical_node_count,
            "physical node contract is invalid",
        )
        registered = set(registered_ids)
        proposal = set(proposal_ids)
        validation = set(validation_ids)
        _require(
            len(registered) == len(registered_ids)
            and proposal
            and validation
            and not proposal.intersection(validation)
            and proposal.union(validation) == registered,
            "registered camera panels are invalid",
        )
        _require(
            len(proposal) >= 3 and len(validation) >= 3,
            "each registered panel needs at least three cameras",
        )
        object.__setattr__(self, "registered_camera_ids", registered_ids)
        object.__setattr__(self, "proposal_camera_ids", proposal_ids)
        object.__setattr__(self, "validation_camera_ids", validation_ids)


@dataclass(frozen=True)
class CausalResponseSourceCameraRecord:
    """Target-free stream and support metadata for one camera."""

    camera_id: str
    depth_frame_count: int
    mask_frame_count: int
    calibration_valid: bool
    frame_zero_projected_support_count: int

    def __post_init__(self) -> None:
        _require(bool(self.camera_id.strip()), "camera ID is empty")
        _require(
            self.depth_frame_count >= 0
            and self.mask_frame_count >= 0
            and self.frame_zero_projected_support_count >= 0,
            "camera source counts are negative",
        )


@dataclass(frozen=True)
class CausalResponseSourcePreflight:
    """Checksummed hash-only source disposition for one candidate episode."""

    config: CausalResponseSourcePreflightConfig
    object_hash: str
    case_hash: str
    category: str
    bimanual_value: str
    episode_frame_count: int
    robot_frame_count: int
    tactile_frame_count: int
    physical_node_count: int
    camera_records: tuple[CausalResponseSourceCameraRecord, ...]
    source_sha256: dict[str, str]
    admitted: bool
    rejection_reasons: tuple[str, ...]
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(
            _valid_digest(self.object_hash)
            and _valid_digest(self.case_hash)
            and _valid_digest(self.artifact_sha256),
            "preflight identity or artifact digest is invalid",
        )
        _require(bool(self.category.strip()), "category is empty")
        _require(
            self.episode_frame_count >= 0
            and self.robot_frame_count >= 0
            and self.tactile_frame_count >= 0
            and self.physical_node_count >= 0,
            "source contract counts are negative",
        )
        camera_ids = [record.camera_id for record in self.camera_records]
        _require(
            len(camera_ids) == len(set(camera_ids)),
            "preflight camera records are duplicated",
        )
        sources = dict(sorted(self.source_sha256.items()))
        _require(
            all(
                bool(name) and _valid_digest(digest) for name, digest in sources.items()
            ),
            "preflight source digest is invalid",
        )
        _require(
            tuple(sorted(set(self.rejection_reasons))) == self.rejection_reasons,
            "preflight rejection reasons are duplicated or unsorted",
        )
        _require(
            self.admitted == (not self.rejection_reasons),
            "preflight decision differs from its reasons",
        )
        object.__setattr__(self, "source_sha256", sources)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalResponseSourcePreflight",
            "contract": CONTRACT,
            "config": json.loads(json.dumps(asdict(self.config), allow_nan=False)),
            "object_hash": self.object_hash,
            "case_hash": self.case_hash,
            "category": self.category,
            "bimanual_value": self.bimanual_value,
            "episode_frame_count": self.episode_frame_count,
            "robot_frame_count": self.robot_frame_count,
            "tactile_frame_count": self.tactile_frame_count,
            "physical_node_count": self.physical_node_count,
            "camera_records": [asdict(record) for record in self.camera_records],
            "source_sha256": dict(self.source_sha256),
            "admitted": self.admitted,
            "rejection_reasons": list(self.rejection_reasons),
            "information_boundary": {
                "metadata_enums_read": True,
                "stream_lengths_read": True,
                "frame_zero_geometry_count_read": True,
                "frame_zero_camera_support_count_read": True,
                "future_object_payload_deserialized": False,
                "future_identity_or_metric_read": False,
                "plaintext_object_or_episode_identity_retained": False,
                "held_v8_artifact_or_process_access": False,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def evaluate_causal_response_source_preflight(
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
    source_sha256: Mapping[str, str],
    config: CausalResponseSourcePreflightConfig | None = None,
) -> CausalResponseSourcePreflight:
    """Evaluate source contracts without opening a future observation or metric."""

    cfg = config or CausalResponseSourcePreflightConfig()
    records = tuple(sorted(camera_records, key=lambda record: record.camera_id))
    sources = dict(sorted(source_sha256.items()))
    _require(
        all(bool(name) and _valid_digest(digest) for name, digest in sources.items()),
        "source checksums are malformed",
    )
    reasons: list[str] = []
    if bimanual_value not in {"yes", "no"}:
        reasons.append("invalid-bimanual-enum")
    for name, value in (
        ("episode", episode_frame_count),
        ("robot", robot_frame_count),
        ("tactile", tactile_frame_count),
    ):
        if value != cfg.required_frame_count:
            reasons.append(f"{name}-frame-count-mismatch")
    if not (
        cfg.minimum_physical_node_count
        <= physical_node_count
        <= cfg.maximum_physical_node_count
    ):
        reasons.append("physical-backend-node-count")
    if not REQUIRED_SOURCE_ROLES.issubset(sources):
        reasons.append("required-source-checksum-missing")

    record_by_camera = {record.camera_id: record for record in records}
    registered = set(cfg.registered_camera_ids)
    if cfg.require_full_registered_camera_panel and set(record_by_camera) != registered:
        reasons.append("registered-camera-panel-mismatch")
    eligible: set[str] = set()
    for camera in cfg.registered_camera_ids:
        record = record_by_camera.get(camera)
        if record is None:
            continue
        if (
            record.depth_frame_count == cfg.required_frame_count
            and record.mask_frame_count == cfg.required_frame_count
            and record.calibration_valid
            and record.frame_zero_projected_support_count > 0
        ):
            eligible.add(camera)
    if not set(cfg.proposal_camera_ids).issubset(eligible):
        reasons.append("proposal-camera-panel-inadmissible")
    if not set(cfg.validation_camera_ids).issubset(eligible):
        reasons.append("validation-camera-panel-inadmissible")

    provisional = CausalResponseSourcePreflight(
        config=cfg,
        object_hash=deform360_object_hash(object_id),
        case_hash=deform360_case_hash(object_id, episode_id),
        category=str(category),
        bimanual_value=str(bimanual_value),
        episode_frame_count=int(episode_frame_count),
        robot_frame_count=int(robot_frame_count),
        tactile_frame_count=int(tactile_frame_count),
        physical_node_count=int(physical_node_count),
        camera_records=records,
        source_sha256=sources,
        admitted=not reasons,
        rejection_reasons=tuple(sorted(set(reasons))),
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor())
    result = CausalResponseSourcePreflight(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )
    validate_causal_response_source_preflight(result)
    return result


def validate_causal_response_source_preflight(
    artifact: CausalResponseSourcePreflight,
) -> None:
    """Validate the sealed V12 source disposition."""

    _require(
        _canonical_sha256(artifact.descriptor()) == artifact.artifact_sha256,
        "source preflight checksum changed",
    )
    boundary = artifact.descriptor()["information_boundary"]
    _require(
        boundary["future_object_payload_deserialized"] is False
        and boundary["future_identity_or_metric_read"] is False
        and boundary["plaintext_object_or_episode_identity_retained"] is False
        and boundary["held_v8_artifact_or_process_access"] is False,
        "source preflight crossed its information boundary",
    )


def write_causal_response_source_preflight(
    path: str | Path,
    artifact: CausalResponseSourcePreflight,
) -> None:
    """Write one immutable source disposition."""

    validate_causal_response_source_preflight(artifact)
    output = Path(path)
    _require(not output.exists(), "source preflight output already exists")
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


def load_causal_response_source_preflight(
    path: str | Path,
) -> CausalResponseSourcePreflight:
    """Load and validate one persisted source disposition."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == "Deform360CausalResponseSourcePreflight"
        and payload.get("contract") == CONTRACT,
        "source preflight kind or contract changed",
    )
    artifact = CausalResponseSourcePreflight(
        config=CausalResponseSourcePreflightConfig(**payload["config"]),
        object_hash=payload["object_hash"],
        case_hash=payload["case_hash"],
        category=payload["category"],
        bimanual_value=payload["bimanual_value"],
        episode_frame_count=payload["episode_frame_count"],
        robot_frame_count=payload["robot_frame_count"],
        tactile_frame_count=payload["tactile_frame_count"],
        physical_node_count=payload["physical_node_count"],
        camera_records=tuple(
            CausalResponseSourceCameraRecord(**record)
            for record in payload["camera_records"]
        ),
        source_sha256=payload["source_sha256"],
        admitted=payload["admitted"],
        rejection_reasons=tuple(payload["rejection_reasons"]),
        artifact_sha256=payload["artifact_sha256"],
    )
    _require(
        artifact.descriptor() == payload,
        "source preflight descriptor changed",
    )
    validate_causal_response_source_preflight(artifact)
    return artifact


__all__ = [
    "CASE_HASH_NAMESPACE",
    "CONTRACT",
    "OBJECT_HASH_NAMESPACE",
    "PROPOSAL_CAMERA_IDS",
    "REGISTERED_CAMERA_IDS",
    "REQUIRED_SOURCE_ROLES",
    "VALIDATION_CAMERA_IDS",
    "CausalResponseSourceCameraRecord",
    "CausalResponseSourcePreflight",
    "CausalResponseSourcePreflightConfig",
    "deform360_case_hash",
    "deform360_object_hash",
    "evaluate_causal_response_source_preflight",
    "load_causal_response_source_preflight",
    "validate_causal_response_source_preflight",
    "write_causal_response_source_preflight",
]
