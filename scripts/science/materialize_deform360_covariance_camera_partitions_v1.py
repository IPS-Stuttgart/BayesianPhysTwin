#!/usr/bin/env python3
"""Materialize the locked Deform360 camera split without decoding payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from bayesian_phystwin.deform360_covariance_provider_v1 import (
    CAMERA_PARTITION_NAMESPACE_V1,
    plan_deform360_camera_partition_v1,
)

_REPOSITORY: Final = Path(__file__).resolve().parents[2]
_PROTOCOL: Final = (
    _REPOSITORY
    / "protocols"
    / "locks"
    / "deform360_covariance_only_target_v1_5.json"
)
_EXACT_PLAN: Final = (
    _REPOSITORY
    / "results"
    / "science"
    / "deform360_covariance_only_target_v1"
    / "exact_file_plan_v1.json"
)
_OUTPUT: Final = (
    _REPOSITORY
    / "results"
    / "science"
    / "deform360_covariance_only_target_v1"
    / "camera_partitions_v1_5.json"
)
_SCHEMA: Final = "bayesian-phystwin/deform360-covariance-camera-partitions-v1"
_SESSION_NAMESPACE: Final = "deform360-covariance-object-session-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_digest(value: Mapping[str, Any], *, digest_key: str) -> str:
    payload = dict(value)
    payload.pop(digest_key, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mapping(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _digest(value: object, *, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return cast(str, value)


def _git_revision(value: object) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value),
        "implementation_revision must be a lowercase Git SHA-1",
    )
    return cast(str, value)


def _object_session_hash(*, object_hash: str, episode_id: int) -> str:
    _digest(object_hash, name="object_hash")
    _require(
        type(episode_id) is int and episode_id >= 0,
        "episode_id must be a nonnegative integer",
    )
    return hashlib.sha256(
        f"{_SESSION_NAMESPACE}\0{object_hash}\0{episode_id}".encode()
    ).hexdigest()


def _camera_ids(value: object) -> tuple[str, ...]:
    _require(
        isinstance(value, list) and len(value) >= 4,
        "camera_streams must contain at least four names",
    )
    raw_cameras = cast(list[object], value)
    cameras = tuple(cast(str, camera) for camera in raw_cameras)
    _require(
        all(
            isinstance(camera, str)
            and camera == camera.strip()
            and camera
            for camera in cameras
        ),
        "camera stream names must be canonical nonempty strings",
    )
    _require(len(cameras) == len(set(cameras)), "camera stream names repeat")
    return cameras


def build_partition_artifact(
    *,
    protocol_path: Path,
    exact_plan_path: Path,
    implementation_revision: str,
) -> dict[str, Any]:
    """Build a names-only partition artifact under the locked v1.5 contract."""

    protocol = _load_mapping(protocol_path, name="protocol")
    _require(
        protocol.get("protocol_id")
        == "deform360-covariance-only-target-v1.5",
        "unexpected protocol",
    )
    _require(
        protocol.get("protocol_sha256")
        == _canonical_digest(protocol, digest_key="protocol_sha256"),
        "protocol digest changed",
    )
    boundary = protocol.get("information_boundary")
    _require(isinstance(boundary, dict), "protocol information boundary missing")
    boundary_map = cast(dict[str, Any], boundary)
    _require(
        boundary_map.get("camera_media_decoded") is False
        and boundary_map.get("geometry_or_track_annotations_opened") is False
        and boundary_map.get("target_outcomes_opened") is False,
        "protocol no longer authorizes a names-only stage",
    )
    revision = _git_revision(implementation_revision)

    exact_plan = _load_mapping(exact_plan_path, name="exact file plan")
    registered = protocol.get("selection_and_acquisition")
    _require(isinstance(registered, dict), "selection/acquisition record missing")
    registered_map = cast(dict[str, Any], registered)
    _require(
        _file_digest(exact_plan_path)
        == registered_map.get("exact_file_plan_file_sha256"),
        "exact file plan bytes changed",
    )
    _require(
        exact_plan.get("plan_sha256")
        == registered_map.get("exact_file_plan_sha256"),
        "exact file plan digest differs from the protocol",
    )
    plan_payload = dict(exact_plan)
    supplied_plan_digest = plan_payload.pop("plan_sha256", None)
    _require(
        supplied_plan_digest
        == hashlib.sha256(_canonical_bytes(plan_payload)).hexdigest(),
        "exact file plan canonical digest changed",
    )
    objects = exact_plan.get("objects")
    target_count = registered_map.get("target_count")
    _require(
        isinstance(objects, list)
        and type(target_count) is int
        and len(objects) == target_count == 24,
        "exact file plan must retain the complete 24-session denominator",
    )
    object_rows = cast(list[object], objects)

    rows: list[dict[str, Any]] = []
    session_hashes: set[str] = set()
    status_counts: dict[str, int] = {}
    for index, raw in enumerate(object_rows):
        _require(isinstance(raw, dict), f"objects[{index}] must be an object")
        raw_map = cast(dict[str, Any], raw)
        object_hash = _digest(raw_map.get("object_hash"), name="object_hash")
        episode_id = raw_map.get("episode_id")
        _require(type(episode_id) is int, "episode_id must be an integer")
        episode = cast(int, episode_id)
        session_hash = _object_session_hash(
            object_hash=object_hash,
            episode_id=episode,
        )
        _require(session_hash not in session_hashes, "object-session repeats")
        session_hashes.add(session_hash)
        cameras = _camera_ids(raw_map.get("camera_streams"))
        provider, scoring = plan_deform360_camera_partition_v1(
            camera_ids=cameras,
            object_session_hash=session_hash,
        )
        _require(
            set(provider).isdisjoint(scoring)
            and set(provider) | set(scoring) == set(cameras),
            "camera partition is not a disjoint cover",
        )
        status = raw_map.get("status")
        _require(
            status in {"planned", "unsupported_without_replacement"},
            "unexpected exact-plan status",
        )
        status_text = cast(str, status)
        status_counts[status_text] = status_counts.get(status_text, 0) + 1
        rows.append(
            {
                "camera_count": len(cameras),
                "episode_id": episode,
                "object_hash": object_hash,
                "object_session_hash": session_hash,
                "provider_camera_ids": list(provider),
                "provider_camera_count": len(provider),
                "scoring_camera_ids": list(scoring),
                "scoring_camera_count": len(scoring),
                "source_plan_status": status_text,
            }
        )

    rows.sort(key=lambda row: row["object_session_hash"])
    provider_counts = [row["provider_camera_count"] for row in rows]
    scoring_counts = [row["scoring_camera_count"] for row in rows]
    artifact: dict[str, Any] = {
        "schema": _SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_file_sha256": _file_digest(protocol_path),
        "exact_file_plan_sha256": exact_plan["plan_sha256"],
        "exact_file_plan_file_sha256": _file_digest(exact_plan_path),
        "implementation_revision": revision,
        "implementation_file": (
            "scripts/science/"
            "materialize_deform360_covariance_camera_partitions_v1.py"
        ),
        "implementation_file_sha256": _file_digest(Path(__file__)),
        "camera_partition_namespace": CAMERA_PARTITION_NAMESPACE_V1,
        "object_session_hash_namespace": _SESSION_NAMESPACE,
        "target_count": len(rows),
        "rows": rows,
        "summary": {
            "provider_camera_count_min": min(provider_counts),
            "provider_camera_count_max": max(provider_counts),
            "scoring_camera_count_min": min(scoring_counts),
            "scoring_camera_count_max": max(scoring_counts),
            "source_plan_status_counts": dict(sorted(status_counts.items())),
        },
        "information_boundary": {
            "names_only_exact_plan_read": True,
            "payload_path_opened": False,
            "camera_media_decoded": False,
            "sensor_arrays_loaded": False,
            "geometry_or_tracks_opened": False,
            "reconstructions_built": False,
            "predictions_run": False,
            "target_outcomes_opened": False,
            "scoring_run": False,
            "replacement_allowed": False,
        },
        "next_stage": (
            "build provider and scoring reconstructions independently from their "
            "registered disjoint camera panels, then seal predictions before scoring"
        ),
    }
    artifact["partition_sha256"] = _canonical_digest(
        artifact,
        digest_key="partition_sha256",
    )
    return artifact


def _write_once(path: Path, artifact: Mapping[str, Any]) -> None:
    encoded = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if path.exists():
        _require(path.read_text(encoding="utf-8") == encoded, "output differs")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=_PROTOCOL)
    parser.add_argument("--exact-plan", type=Path, default=_EXACT_PLAN)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    parser.add_argument("--implementation-revision", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = build_partition_artifact(
        protocol_path=args.protocol,
        exact_plan_path=args.exact_plan,
        implementation_revision=args.implementation_revision,
    )
    _write_once(args.output, artifact)
    print(
        json.dumps(
            {
                "partition_sha256": artifact["partition_sha256"],
                "target_count": artifact["target_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
