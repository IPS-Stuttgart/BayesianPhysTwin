#!/usr/bin/env python3
"""Plan and verify a retrospective Deform360 DLO cross-episode pilot.

The planner reads only directory names, file names, sizes, and one small
``metadata.json`` file per registered object. It freezes source/target episode
pairs before the workflow decodes any video or loads any numerical array.

The verifier inspects only generated file contracts. It does not load robot,
tactile, calibration, or image arrays and it never computes a model score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin/deform360-dlo-cross-episode-pilot-protocol"
)
PLAN_SCHEMA: Final = "bayesian-phystwin/deform360-dlo-cross-episode-plan"
VERIFICATION_SCHEMA: Final = (
    "bayesian-phystwin/deform360-dlo-cross-episode-preprocessing-verification"
)
REQUEST_SCHEMA: Final = (
    "bayesian-phystwin/deform360-dlo-cross-episode-pilot-request"
)
OBJECT_RE: Final = re.compile(r"^\d{3}-.+$")
CAMERA_RE: Final = re.compile(r"^brics-odroid-\d+_cam\d+$")
TACTILE_RE: Final = re.compile(r"^brics-odroid_tactile.+$")
EPISODE_KEY_RE: Final = re.compile(r"(?:episode[_ -]*)?(\d+)", re.IGNORECASE)
ACTION_KEY_FRAGMENTS: Final = (
    "action",
    "interaction",
    "manipulation",
    "motion",
    "primitive",
    "task",
)
ACTION_FAMILIES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "shape-change",
        (
            "bend",
            "compress",
            "crumple",
            "fold",
            "knot",
            "roll",
            "scrunch",
            "squeeze",
            "twist",
            "wrap",
        ),
    ),
    (
        "elevation",
        (
            "hang",
            "lift",
            "pick",
            "raise",
            "shake",
            "swing",
            "wave",
        ),
    ),
    (
        "planar-contact",
        (
            "drag",
            "move",
            "pull",
            "push",
            "slide",
            "stretch",
            "tug",
        ),
    ),
)


class PilotError(ValueError):
    """Raised when a frozen pilot contract cannot be satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def _pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PilotError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream, object_pairs_hook=_pairs_hook)
    except PilotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PilotError(f"cannot read JSON: {path}") from error
    _require(type(value) is dict, f"JSON root must be an object: {path}")
    return value


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(value: Mapping[str, object], identity_field: str) -> str:
    payload = dict(value)
    payload.pop(identity_field, None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)
        stream.flush()
        os.fsync(stream.fileno())


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = _read_json(path.resolve())
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    _require(protocol.get("schema_version") == 1, "unsupported protocol version")
    _require(
        protocol.get("status") == "retrospective-development-frozen-before-run",
        "protocol status changed",
    )
    raw_root = protocol.get("raw_root")
    _require(
        type(raw_root) is str and raw_root.startswith("/"),
        "raw_root must be an absolute path",
    )
    objects = protocol.get("objects")
    _require(
        type(objects) is list
        and len(objects) == 4
        and objects == sorted(set(objects))
        and all(type(item) is str and OBJECT_RE.fullmatch(item) for item in objects),
        "objects must be four sorted canonical Deform360 IDs",
    )
    _require(
        protocol.get("primary_object") in objects,
        "primary_object must be in the registered roster",
    )
    thresholds = protocol.get("thresholds")
    _require(type(thresholds) is dict, "thresholds must be an object")
    for key in (
        "minimum_camera_pairs_per_episode",
        "minimum_tactile_pairs_per_episode",
        "minimum_episodes_per_object",
        "minimum_ready_objects",
    ):
        value = thresholds.get(key)
        _require(
            type(value) is int and value > 0,
            f"threshold {key} must be a positive integer",
        )
    _require(
        thresholds["minimum_ready_objects"] == len(objects),
        "the pilot must require all four registered objects",
    )
    upstream = protocol.get("official_processing")
    _require(type(upstream) is dict, "official_processing must be an object")
    _require(
        upstream.get("repository") == "lhy0807/deform360",
        "unexpected official processing repository",
    )
    revision = upstream.get("revision")
    _require(
        type(revision) is str
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "official processing revision must be a full lowercase SHA",
    )
    boundary = protocol.get("information_boundary")
    _require(type(boundary) is dict, "information_boundary must be an object")
    expected_boundary = {
        "fresh_confirmation": False,
        "planner_decodes_media": False,
        "planner_loads_numeric_arrays": False,
        "pairing_frozen_before_media_decode": True,
        "preprocessing_may_decode_selected_media": True,
        "model_scoring_authorized": False,
        "paper_claim_authorized": False,
    }
    _require(boundary == expected_boundary, "information boundary changed")
    return protocol


def load_request(path: Path, protocol: Mapping[str, object]) -> dict[str, Any]:
    request = _read_json(path.resolve())
    _require(request.get("schema") == REQUEST_SCHEMA, "unexpected request schema")
    _require(request.get("schema_version") == 1, "unsupported request version")
    _require(
        request.get("protocol_id") == protocol.get("protocol_id"),
        "request protocol binding changed",
    )
    _require(
        request.get("authorization")
        == "retrospective-four-object-preprocessing-only",
        "request authorization changed",
    )
    _require(request.get("execute") is True, "request is not executable")
    _require(request.get("model_scoring") is False, "model scoring is forbidden")
    _require(request.get("paper_claim") is False, "paper claim is forbidden")
    _require(
        request.get("objects") == protocol.get("objects"),
        "request object roster changed",
    )
    return request


def _paired_stems(
    directory: Path,
    *,
    data_suffix: str,
    exclude_prefix: str | None = None,
) -> set[str]:
    data: set[str] = set()
    timestamps: set[str] = set()
    try:
        entries = tuple(directory.iterdir())
    except OSError as error:
        raise PilotError(f"cannot enumerate stream directory: {directory}") from error
    for path in entries:
        if path.is_symlink() or not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == data_suffix:
            if exclude_prefix is None or not path.stem.lower().startswith(exclude_prefix):
                data.add(path.stem)
        elif suffix == ".txt":
            timestamps.add(path.stem)
    return data & timestamps


def _sequence_index(raw_key: object, ordinal: int) -> int:
    if type(raw_key) is int and raw_key >= 0:
        return raw_key
    if type(raw_key) is str:
        match = EPISODE_KEY_RE.fullmatch(raw_key.strip())
        if match is not None:
            return int(match.group(1))
    return ordinal


def _sequence_records(metadata: Mapping[str, object]) -> dict[int, Mapping[str, object]]:
    candidate: object | None = None
    for key in ("sequences", "episodes", "interactions"):
        if key in metadata:
            candidate = metadata[key]
            break
    if type(candidate) is list:
        result: dict[int, Mapping[str, object]] = {}
        for index, value in enumerate(candidate):
            if type(value) is dict:
                result[index] = value
        return result
    if type(candidate) is dict:
        result = {}
        ordered = sorted(candidate.items(), key=lambda item: str(item[0]))
        for ordinal, (raw_key, value) in enumerate(ordered):
            if type(value) is dict:
                index = _sequence_index(raw_key, ordinal)
                _require(index not in result, "duplicate sequence index")
                result[index] = value
        return result
    return {}


def _scalar_text(value: object) -> str | None:
    if type(value) is str:
        rendered = " ".join(value.split())
        return rendered if rendered else None
    if type(value) in {int, float}:
        return str(value)
    return None


def _action_values(
    value: object,
    *,
    path: tuple[str, ...] = (),
    action_context: bool = False,
) -> list[tuple[int, str, str]]:
    found: list[tuple[int, str, str]] = []
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                continue
            lowered = key.lower()
            direct = any(fragment in lowered for fragment in ACTION_KEY_FRAGMENTS)
            active = action_context or direct
            scalar = _scalar_text(child)
            child_path = (*path, key)
            if active and scalar is not None:
                priority = 0 if direct else 1
                found.append((priority, ".".join(child_path), scalar))
            found.extend(
                _action_values(
                    child,
                    path=child_path,
                    action_context=active,
                )
            )
    elif type(value) is list:
        for index, child in enumerate(value):
            found.extend(
                _action_values(
                    child,
                    path=(*path, str(index)),
                    action_context=action_context,
                )
            )
    return found


def _action_signature(record: Mapping[str, object] | None) -> str | None:
    if record is None:
        return None
    unique: dict[str, tuple[int, str, str]] = {}
    for priority, path, value in _action_values(record):
        normalized = value.casefold()
        current = unique.get(normalized)
        row = (priority, path, value)
        if current is None or row < current:
            unique[normalized] = row
    if not unique:
        return None
    ordered = sorted(unique.values())
    return " | ".join(row[2] for row in ordered[:4])


def _action_family(signature: str | None) -> str | None:
    if signature is None:
        return None
    lowered = signature.casefold()
    for family, tokens in ACTION_FAMILIES:
        if any(token in lowered for token in tokens):
            return family
    return "other"


def _bimanual(record: Mapping[str, object] | None) -> bool | None:
    if record is None:
        return None
    stack: list[object] = [record]
    while stack:
        current = stack.pop()
        if type(current) is dict:
            for key, value in current.items():
                if type(key) is str and key.casefold() == "bimanual":
                    if type(value) is bool:
                        return value
                    if type(value) is int and value in {0, 1}:
                        return bool(value)
                stack.append(value)
        elif type(current) is list:
            stack.extend(current)
    return None


def _metadata_identity(path: Path, maximum_bytes: int) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"missing metadata: {path}")
    size = path.stat().st_size
    _require(0 < size <= maximum_bytes, f"metadata size outside policy: {path}")
    payload = _read_json(path)
    return payload, _sha256_file(path)


def _episode_rows(
    object_dir: Path,
    metadata: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    camera_streams: dict[str, set[str]] = {}
    tactile_streams: dict[str, set[str]] = {}
    try:
        entries = sorted(object_dir.iterdir(), key=lambda path: path.name)
    except OSError as error:
        raise PilotError(f"cannot enumerate object directory: {object_dir}") from error
    for path in entries:
        if path.is_symlink() or not path.is_dir():
            continue
        if CAMERA_RE.fullmatch(path.name):
            camera_streams[path.name] = _paired_stems(path, data_suffix=".mp4")
        elif TACTILE_RE.fullmatch(path.name):
            tactile_streams[path.name] = _paired_stems(
                path,
                data_suffix=".npy",
                exclude_prefix="median_",
            )
    stems: set[str] = set()
    for values in (*camera_streams.values(), *tactile_streams.values()):
        stems.update(values)
    ordered_stems = sorted(stems)
    sequences = _sequence_records(metadata)
    minimum_camera = int(thresholds["minimum_camera_pairs_per_episode"])
    minimum_tactile = int(thresholds["minimum_tactile_pairs_per_episode"])
    rows: list[dict[str, object]] = []
    for episode_index, stem in enumerate(ordered_stems):
        sequence = sequences.get(episode_index)
        signature = _action_signature(sequence)
        camera_pairs = sum(stem in values for values in camera_streams.values())
        tactile_pairs = sum(stem in values for values in tactile_streams.values())
        rows.append(
            {
                "episode_index": episode_index,
                "raw_stem": stem,
                "camera_pairs": camera_pairs,
                "tactile_pairs": tactile_pairs,
                "action_signature": signature,
                "action_family": _action_family(signature),
                "bimanual": _bimanual(sequence),
                "usable": camera_pairs >= minimum_camera
                and tactile_pairs >= minimum_tactile,
            }
        )
    return rows, {
        "camera_stream_directory_count": len(camera_streams),
        "tactile_stream_directory_count": len(tactile_streams),
        "metadata_sequence_count": len(sequences),
    }


def _pair_score(
    source: Mapping[str, object], target: Mapping[str, object]
) -> tuple[int, int, int, int, int, int]:
    source_signature = source.get("action_signature")
    target_signature = target.get("action_signature")
    signatures_known = type(source_signature) is str and type(target_signature) is str
    different_signature = signatures_known and source_signature != target_signature
    source_family = source.get("action_family")
    target_family = target.get("action_family")
    different_family = (
        type(source_family) is str
        and type(target_family) is str
        and source_family != target_family
    )
    return (
        int(different_family),
        int(different_signature),
        min(int(source["camera_pairs"]), int(target["camera_pairs"])),
        min(int(source["tactile_pairs"]), int(target["tactile_pairs"])),
        -int(source["episode_index"]),
        -int(target["episode_index"]),
    )


def _choose_pair(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    usable = [row for row in rows if row.get("usable") is True]
    if len(usable) < 2:
        return None
    candidates = [
        (source, target)
        for source in usable
        for target in usable
        if source["episode_index"] != target["episode_index"]
    ]
    source, target = max(candidates, key=lambda pair: _pair_score(*pair))
    source_signature = source.get("action_signature")
    target_signature = target.get("action_signature")
    signature_resolved = (
        type(source_signature) is str
        and type(target_signature) is str
        and source_signature != target_signature
    )
    family_resolved = (
        type(source.get("action_family")) is str
        and type(target.get("action_family")) is str
        and source["action_family"] != target["action_family"]
    )
    return {
        "source_episode_index": source["episode_index"],
        "target_episode_index": target["episode_index"],
        "source_raw_stem": source["raw_stem"],
        "target_raw_stem": target["raw_stem"],
        "source_action_signature": source_signature,
        "target_action_signature": target_signature,
        "source_action_family": source.get("action_family"),
        "target_action_family": target.get("action_family"),
        "source_bimanual": source.get("bimanual"),
        "target_bimanual": target.get("bimanual"),
        "different_action_signature": signature_resolved,
        "different_action_family": family_resolved,
        "pairing_tier": (
            "cross-action-family"
            if family_resolved
            else "different-action-signature"
            if signature_resolved
            else "episode-only-action-unresolved"
        ),
    }


def build_plan(
    protocol: Mapping[str, Any],
    *,
    protocol_path: Path,
    repository_revision: str,
) -> dict[str, object]:
    _require(
        len(repository_revision) == 40
        and all(character in "0123456789abcdef" for character in repository_revision),
        "repository_revision must be a full lowercase SHA",
    )
    raw_root = Path(protocol["raw_root"])
    _require(raw_root.is_dir() and not raw_root.is_symlink(), "raw root unavailable")
    resolved_root = raw_root.resolve(strict=True)
    _require(str(resolved_root) == str(raw_root), "raw root must be canonical")
    thresholds = protocol["thresholds"]
    maximum_metadata_bytes = int(protocol["maximum_metadata_json_bytes"])
    object_records: list[dict[str, object]] = []
    for object_id in protocol["objects"]:
        object_dir = resolved_root / object_id
        _require(
            object_dir.is_dir() and not object_dir.is_symlink(),
            f"registered object unavailable: {object_id}",
        )
        calibration_dir = object_dir / "calibration_refined"
        calibration_files = {
            name: (calibration_dir / name).is_file()
            for name in ("intrinsics.npy", "extrinsics.npy", "dist.npy")
        }
        metadata, metadata_sha256 = _metadata_identity(
            object_dir / "metadata.json",
            maximum_metadata_bytes,
        )
        episodes, stream_counts = _episode_rows(object_dir, metadata, thresholds)
        pair = _choose_pair(episodes)
        object_records.append(
            {
                "object_id": object_id,
                "object_dir": str(object_dir),
                "metadata_sha256": metadata_sha256,
                "calibration_files_present": calibration_files,
                "calibration_ready": all(calibration_files.values()),
                **stream_counts,
                "episode_count": len(episodes),
                "usable_episode_count": sum(
                    row["usable"] is True for row in episodes
                ),
                "episodes": episodes,
                "selected_pair": pair,
                "processing_ready": pair is not None
                and all(calibration_files.values()),
                "cross_action_ready": pair is not None
                and pair["different_action_signature"] is True,
            }
        )
    ready_count = sum(row["processing_ready"] is True for row in object_records)
    cross_action_count = sum(
        row["cross_action_ready"] is True for row in object_records
    )
    minimum_ready = int(thresholds["minimum_ready_objects"])
    if ready_count >= minimum_ready:
        decision = (
            "four-object-cross-action-preprocessing-ready"
            if cross_action_count >= minimum_ready
            else "four-object-preprocessing-ready-action-resolution-partial"
        )
    else:
        decision = "insufficient-four-object-preprocessing-roster"
    protocol_source = protocol_path.resolve()
    plan: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": _sha256_file(protocol_source),
        "repository_revision": repository_revision,
        "runner_label": protocol["runner_label"],
        "raw_root": str(resolved_root),
        "object_records": object_records,
        "summary": {
            "registered_object_count": len(object_records),
            "processing_ready_object_count": ready_count,
            "cross_action_ready_object_count": cross_action_count,
            "decision": decision,
        },
        "information_boundary": protocol["information_boundary"],
        "claim_boundary": (
            "Retrospective preprocessing design only. This plan freezes episode "
            "pairs before this workflow decodes selected media. It is not a model "
            "evaluation, a fresh confirmation, a physical-transport result, or a "
            "paper claim."
        ),
    }
    plan["plan_id"] = _content_id(plan, "plan_id")
    return plan


def _load_plan(path: Path) -> dict[str, Any]:
    plan = _read_json(path.resolve())
    _require(plan.get("schema") == PLAN_SCHEMA, "unexpected plan schema")
    _require(plan.get("schema_version") == 1, "unsupported plan version")
    _require(plan.get("plan_id") == _content_id(plan, "plan_id"), "plan ID mismatch")
    records = plan.get("object_records")
    _require(type(records) is list and records, "plan object_records are invalid")
    return plan


def emit_worklist(plan: Mapping[str, Any]) -> str:
    lines: list[str] = []
    for record in plan["object_records"]:
        _require(type(record) is dict, "plan object record is invalid")
        pair = record.get("selected_pair")
        _require(type(pair) is dict, f"no pair for {record.get('object_id')}")
        source_bimanual = pair.get("source_bimanual")
        target_bimanual = pair.get("target_bimanual")
        _require(
            type(source_bimanual) is bool and type(target_bimanual) is bool,
            f"bimanual state unresolved for {record.get('object_id')}",
        )
        fields = (
            record["object_id"],
            pair["source_episode_index"],
            pair["target_episode_index"],
            int(source_bimanual),
            int(target_bimanual),
            pair["pairing_tier"],
        )
        lines.append("\t".join(str(field) for field in fields))
    return "\n".join(lines) + "\n"


def verify_outputs(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    processed_root: Path,
) -> dict[str, object]:
    _require(processed_root.is_dir(), f"processed root unavailable: {processed_root}")
    resolved_root = processed_root.resolve(strict=True)
    minimum_camera = int(protocol["thresholds"]["minimum_camera_pairs_per_episode"])
    minimum_tactile = int(
        protocol["thresholds"]["minimum_tactile_pairs_per_episode"]
    )
    records: list[dict[str, object]] = []
    for object_record in plan["object_records"]:
        pair = object_record["selected_pair"]
        object_id = object_record["object_id"]
        object_dir = resolved_root / object_id
        for role in ("source", "target"):
            episode_index = int(pair[f"{role}_episode_index"])
            episode_dir = object_dir / f"episode_{episode_index:04d}"
            camera_count = 0
            tactile_count = 0
            manifest_hashes: dict[str, str] = {}
            if episode_dir.is_dir():
                for path in sorted(episode_dir.iterdir(), key=lambda item: item.name):
                    if not path.is_dir() or path.is_symlink():
                        continue
                    if CAMERA_RE.fullmatch(path.name):
                        required = (
                            path / "undistorted.mp4",
                            path / "aligned_timestamps.txt",
                            path / "alignment.json",
                            path / "metadata.json",
                        )
                        if all(
                            item.is_file() and item.stat().st_size > 0
                            for item in required
                        ):
                            camera_count += 1
                            for item in required[2:]:
                                manifest_hashes[
                                    item.relative_to(episode_dir).as_posix()
                                ] = _sha256_file(item)
                    elif TACTILE_RE.fullmatch(path.name):
                        required = (
                            path / "synced_tactile.npy",
                            path / "alignment.json",
                            path / "metadata.json",
                        )
                        if all(
                            item.is_file() and item.stat().st_size > 0
                            for item in required
                        ):
                            tactile_count += 1
                            for item in required[1:]:
                                manifest_hashes[
                                    item.relative_to(episode_dir).as_posix()
                                ] = _sha256_file(item)
            episode_contracts = (
                episode_dir / "alignment.json",
                episode_dir / "undistorted_intrinsics.npy",
                episode_dir / "extrinsics.npy",
            )
            robot_contracts = (
                episode_dir / "robot" / "robot.npz",
                episode_dir / "robot" / "robot.meta.json",
            )
            for item in (*episode_contracts[:1], *robot_contracts[1:]):
                if item.is_file() and item.stat().st_size > 0:
                    manifest_hashes[item.relative_to(episode_dir).as_posix()] = (
                        _sha256_file(item)
                    )
            ready = (
                all(
                    item.is_file() and item.stat().st_size > 0
                    for item in episode_contracts
                )
                and camera_count >= minimum_camera
                and tactile_count >= minimum_tactile
                and all(
                    item.is_file() and item.stat().st_size > 0
                    for item in robot_contracts
                )
            )
            records.append(
                {
                    "object_id": object_id,
                    "role": role,
                    "episode_index": episode_index,
                    "episode_dir": str(episode_dir),
                    "camera_contract_count": camera_count,
                    "tactile_contract_count": tactile_count,
                    "episode_contracts_present": {
                        item.name: item.is_file() and item.stat().st_size > 0
                        for item in episode_contracts
                    },
                    "robot_contracts_present": {
                        item.name: item.is_file() and item.stat().st_size > 0
                        for item in robot_contracts
                    },
                    "small_manifest_sha256": manifest_hashes,
                    "ready": ready,
                }
            )
    ready_count = sum(record["ready"] is True for record in records)
    verification: dict[str, object] = {
        "schema": VERIFICATION_SCHEMA,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "plan_id": plan["plan_id"],
        "repository_revision": plan["repository_revision"],
        "processed_root": str(resolved_root),
        "episode_record_count": len(records),
        "ready_episode_count": ready_count,
        "all_selected_episodes_ready": ready_count == len(records),
        "episode_records": records,
        "information_boundary": {
            "generated_video_payloads_loaded_by_verifier": False,
            "generated_numeric_arrays_loaded_by_verifier": False,
            "model_scores_computed": False,
            "target_outcomes_used_for_model_selection": False,
        },
        "claim_boundary": (
            "This verifies synchronized RGB, tactile, and robot-state file "
            "contracts for the frozen retrospective pairs. It does not verify "
            "object geometry, model accuracy, uncertainty calibration, physical "
            "parameter transport, or paper-level evidence."
        ),
    }
    verification["verification_id"] = _content_id(
        verification, "verification_id"
    )
    return verification


def _fixture_protocol(root: Path, objects: Sequence[str]) -> dict[str, object]:
    return {
        "schema": PROTOCOL_SCHEMA,
        "schema_version": 1,
        "protocol_id": "fixture",
        "status": "retrospective-development-frozen-before-run",
        "runner_label": "fixture",
        "raw_root": str(root),
        "objects": list(objects),
        "primary_object": objects[0],
        "maximum_metadata_json_bytes": 1048576,
        "thresholds": {
            "minimum_camera_pairs_per_episode": 2,
            "minimum_tactile_pairs_per_episode": 1,
            "minimum_episodes_per_object": 2,
            "minimum_ready_objects": 4,
        },
        "official_processing": {
            "repository": "lhy0807/deform360",
            "revision": "d8522a4403b766aeb387510c04e89032a56fdf35",
        },
        "information_boundary": {
            "fresh_confirmation": False,
            "planner_decodes_media": False,
            "planner_loads_numeric_arrays": False,
            "pairing_frozen_before_media_decode": True,
            "preprocessing_may_decode_selected_media": True,
            "model_scoring_authorized": False,
            "paper_claim_authorized": False,
        },
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "raw"
        objects = (
            "001-rope",
            "002-rope-silk",
            "003-cable",
            "081-stripe-rope",
        )
        for object_id in objects:
            object_dir = root / object_id
            calibration = object_dir / "calibration_refined"
            calibration.mkdir(parents=True)
            for name in ("intrinsics.npy", "extrinsics.npy", "dist.npy"):
                (calibration / name).write_bytes(b"fixture")
            metadata = {
                "sequences": [
                    {"action": "drag edge", "bimanual": False},
                    {"action": "lift edge", "bimanual": True},
                ]
            }
            (object_dir / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            for camera in range(2):
                stream = object_dir / f"brics-odroid-{camera + 1:03d}_cam0"
                stream.mkdir()
                for stem in ("capture_a", "capture_b"):
                    (stream / f"{stem}.mp4").write_bytes(b"video")
                    (stream / f"{stem}.txt").write_text("0\n", encoding="utf-8")
            tactile = object_dir / "brics-odroid_tactile_left"
            tactile.mkdir()
            for stem in ("capture_a", "capture_b"):
                (tactile / f"{stem}.npy").write_bytes(b"array")
                (tactile / f"{stem}.txt").write_text("0\n", encoding="utf-8")
        protocol = _fixture_protocol(root, objects)
        protocol_path = Path(temporary) / "protocol.json"
        protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
        loaded = load_protocol(protocol_path)
        plan = build_plan(
            loaded,
            protocol_path=protocol_path,
            repository_revision="1" * 40,
        )
        _require(
            plan["summary"]["processing_ready_object_count"] == 4,
            "fixture processing-ready count changed",
        )
        _require(
            plan["summary"]["cross_action_ready_object_count"] == 4,
            "fixture cross-action count changed",
        )
        lines = emit_worklist(plan).splitlines()
        _require(len(lines) == 4, "fixture worklist cardinality changed")
        _require(all(len(line.split("\t")) == 6 for line in lines), "bad worklist")
    print("self-test passed")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--protocol", type=Path, required=True)
    plan_parser.add_argument("--request", type=Path, required=True)
    plan_parser.add_argument("--repository-revision", required=True)
    plan_parser.add_argument("--output", type=Path, required=True)

    worklist_parser = subparsers.add_parser("worklist")
    worklist_parser.add_argument("--plan", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--protocol", type=Path, required=True)
    verify_parser.add_argument("--plan", type=Path, required=True)
    verify_parser.add_argument("--processed-root", type=Path, required=True)
    verify_parser.add_argument("--output", type=Path, required=True)

    subparsers.add_parser("self-test")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    if arguments.command == "self-test":
        self_test()
        return 0
    if arguments.command == "worklist":
        sys.stdout.write(emit_worklist(_load_plan(arguments.plan)))
        return 0
    if arguments.command == "plan":
        protocol = load_protocol(arguments.protocol)
        load_request(arguments.request, protocol)
        plan = build_plan(
            protocol,
            protocol_path=arguments.protocol,
            repository_revision=arguments.repository_revision,
        )
        _write_json(arguments.output, plan)
        print(json.dumps(plan["summary"], sort_keys=True))
        minimum = int(protocol["thresholds"]["minimum_ready_objects"])
        return (
            0
            if plan["summary"]["processing_ready_object_count"] >= minimum
            else 3
        )
    if arguments.command == "verify":
        protocol = load_protocol(arguments.protocol)
        plan = _load_plan(arguments.plan)
        verification = verify_outputs(protocol, plan, arguments.processed_root)
        _write_json(arguments.output, verification)
        print(
            json.dumps(
                {
                    "all_selected_episodes_ready": verification[
                        "all_selected_episodes_ready"
                    ],
                    "ready_episode_count": verification["ready_episode_count"],
                    "verification_id": verification["verification_id"],
                },
                sort_keys=True,
            )
        )
        return 0 if verification["all_selected_episodes_ready"] else 4
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PilotError as error:
        print(f"Deform360 DLO pilot failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
