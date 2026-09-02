"""Verify and replay the exact Deform360 readiness-bound carrier subset.

The original untouched-object readiness artifact bound metadata bytes, carrier paths,
and carrier sizes before target access.  Later dataset processing may add carriers for
previously incomplete episodes.  Such additive files must not silently change the
precommitted target episode.  This module verifies every bound file plus every
sampled fingerprint retained in the immutable confirmation result, records any
unbound additions, and constructs descriptors from only the original bound subset.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "bayesian-phystwin/deform360-bound-carrier-replay-v8"
TACTILE_RE = re.compile(r"tactile", re.IGNORECASE)


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe bound relative path: {value}")
    return path


def _bound_path(root: Path, value: str) -> Path:
    return root / _safe_relative(value)


def _sampled_fingerprint(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(1024 * 1024))
        if size > 1024 * 1024:
            stream.seek(max(size - 1024 * 1024, 0))
            digest.update(stream.read(1024 * 1024))
    return {
        "size_bytes": int(size),
        "sampled_sha256": digest.hexdigest(),
        "rule": "sha256(size || first_1MiB || last_1MiB)",
    }


def _verify_identity(root: Path, record: Mapping[str, Any]) -> Path:
    path = _bound_path(root, str(record["relative_path"]))
    if not path.is_file():
        raise ValueError(f"bound carrier is missing: {record['relative_path']}")
    if path.name != str(record["name"]):
        raise ValueError(f"bound carrier name changed: {record['relative_path']}")
    observed_size = int(path.stat().st_size)
    expected_size = int(record["size_bytes"])
    if observed_size != expected_size:
        raise ValueError(
            f"bound carrier size changed: {record['relative_path']} "
            f"({observed_size} != {expected_size})"
        )
    return path


def _retained_fingerprints(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    source = row.get("source_fingerprints")
    if not isinstance(source, list):
        raise ValueError(f"source fingerprints missing for {row.get('object_id')}")
    for episode in source:
        if not isinstance(episode, Mapping) or not isinstance(
            episode.get("files"), list
        ):
            raise ValueError(
                f"invalid source fingerprint row for {row.get('object_id')}"
            )
        result.extend(episode["files"])
    target = row.get("target_fingerprint")
    if not isinstance(target, Mapping) or not isinstance(target.get("files"), list):
        raise ValueError(f"target fingerprint missing for {row.get('object_id')}")
    result.extend(target["files"])
    return result


def _verify_retained_fingerprints(
    data_root: Path,
    original_root: Path,
    row: Mapping[str, Any],
) -> int:
    checked: set[str] = set()
    for record in _retained_fingerprints(row):
        original_path = Path(str(record["path"]))
        try:
            relative = original_path.relative_to(original_root)
        except ValueError as error:
            raise ValueError(
                f"retained fingerprint escaped original root: {original_path}"
            ) from error
        relative_text = relative.as_posix()
        if relative_text in checked:
            continue
        checked.add(relative_text)
        path = _bound_path(data_root, relative_text)
        if not path.is_file():
            raise ValueError(
                f"retained confirmation carrier is missing: {relative_text}"
            )
        observed = _sampled_fingerprint(path)
        if int(record["size_bytes"]) != observed["size_bytes"]:
            raise ValueError(f"retained carrier size changed: {relative_text}")
        if str(record["sampled_sha256"]) != observed["sampled_sha256"]:
            raise ValueError(f"retained carrier fingerprint changed: {relative_text}")
        if str(record["rule"]) != observed["rule"]:
            raise ValueError(f"retained fingerprint rule changed: {relative_text}")
    return len(checked)


def _recognized_current_files(
    data_root: Path,
    object_id: str,
) -> tuple[set[str], set[str]]:
    raw = data_root / "raw-repository" / "raw" / object_id
    processed = data_root / "processed-repository" / "processed" / object_id
    tactile: set[str] = set()
    if raw.is_dir():
        for directory in sorted(
            (path for path in raw.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        ):
            if not TACTILE_RE.search(directory.name):
                continue
            for path in directory.glob("*.npy"):
                if path.name.lower().startswith("median_") or path.stat().st_size <= 0:
                    continue
                tactile.add(path.relative_to(data_root).as_posix())
    robot: set[str] = set()
    if processed.is_dir():
        for name in ("robot.npy", "robot.npz"):
            for path in processed.rglob(name):
                if path.is_file() and path.stat().st_size > 0:
                    robot.add(path.relative_to(data_root).as_posix())
    return robot, tactile


def verify_bound_replay(
    data_root: Path,
    protocol: Mapping[str, Any],
    readiness: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify that the original carrier subset is intact and changes are additive."""

    data_root = data_root.resolve(strict=True)
    original_root = Path(str(protocol["dataset_root"]))
    manifest = readiness.get("selection_manifest")
    if not isinstance(manifest, list):
        raise ValueError("readiness selection manifest is absent")
    reference_rows = {
        str(row["object_id"]): row for row in reference.get("objects", [])
    }
    records: list[dict[str, Any]] = []
    total_bound_identities = 0
    total_fingerprints = 0
    total_additions = 0
    for expected in manifest:
        if not isinstance(expected, Mapping):
            raise ValueError("invalid readiness manifest row")
        object_id = str(expected["object_id"])
        raw = data_root / "raw-repository" / "raw" / object_id
        metadata_path = raw / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"bound metadata is missing: {object_id}")
        metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
        if metadata_sha256 != str(expected["metadata_sha256"]):
            raise ValueError(f"bound metadata changed: {object_id}")

        expected_robot: set[str] = set()
        for identity in expected["robot_files"]:
            _verify_identity(data_root, identity)
            expected_robot.add(str(identity["relative_path"]))
            total_bound_identities += 1
        expected_tactile: set[str] = set()
        for group in expected["tactile_groups"]:
            if int(group["recording_count"]) != len(group["recordings"]):
                raise ValueError(f"bound tactile count is inconsistent: {object_id}")
            for identity in group["recordings"]:
                _verify_identity(data_root, identity)
                expected_tactile.add(str(identity["relative_path"]))
                total_bound_identities += 1

        reference_row = reference_rows.get(object_id)
        if reference_row is None:
            raise ValueError(f"immutable confirmation row is missing: {object_id}")
        fingerprint_count = _verify_retained_fingerprints(
            data_root, original_root, reference_row
        )
        total_fingerprints += fingerprint_count

        current_robot, current_tactile = _recognized_current_files(data_root, object_id)
        missing_robot = sorted(expected_robot - current_robot)
        missing_tactile = sorted(expected_tactile - current_tactile)
        if missing_robot or missing_tactile:
            raise ValueError(
                f"bound carrier disappeared from current inventory: {object_id}"
            )
        added_robot = sorted(current_robot - expected_robot)
        added_tactile = sorted(current_tactile - expected_tactile)
        total_additions += len(added_robot) + len(added_tactile)
        records.append(
            {
                "object_id": object_id,
                "metadata_sha256_matches": True,
                "bound_identity_count": len(expected_robot) + len(expected_tactile),
                "retained_fingerprint_count": fingerprint_count,
                "added_robot_files": added_robot,
                "added_tactile_recordings": added_tactile,
                "bound_files_missing": False,
                "bound_files_changed": False,
            }
        )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 8,
        "status": "verified-bound-subset-with-additive-unbound-files-allowed",
        "object_count": len(records),
        "bound_identity_count": total_bound_identities,
        "retained_fingerprint_count": total_fingerprints,
        "additive_unbound_file_count": total_additions,
        "objects_with_additions": sum(
            bool(row["added_robot_files"] or row["added_tactile_recordings"])
            for row in records
        ),
        "objects": records,
        "contract": {
            "metadata_sha256_must_match": True,
            "readiness_bound_path_name_and_size_must_match": True,
            "immutable_result_sampled_fingerprints_must_match": True,
            "only_unbound_additions_may_be_ignored": True,
            "target_episode_may_not_change": True,
            "exact_scientific_result_reproduction_still_required": True,
        },
        "paper_claim_authorized": False,
    }
    result["result_sha256"] = canonical_digest(result)
    return result


def inspection_from_bound_manifest(expected: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact readiness projection expected by the immutable control."""

    return {
        "object_id": str(expected["object_id"]),
        "eligible": True,
        "reason": None,
        "metadata_sha256": str(expected["metadata_sha256"]),
        "complete_episode_ids": list(expected["complete_episode_ids"]),
        "target_episode_id": int(expected["target_episode_id"]),
        "target_action": expected["target_action"],
        "robot_files": list(expected["robot_files"]),
        "tactile_groups": list(expected["tactile_groups"]),
    }


def build_bound_descriptors(
    base: Any,
    data_root: Path,
    expected: Mapping[str, Any],
    minimum_episodes: int,
) -> list[Any]:
    """Construct the same descriptors as the original precommitted subset."""

    object_id = str(expected["object_id"])
    metadata_path = data_root / "raw-repository" / "raw" / object_id / "metadata.json"
    episodes = base.episode_records(base.read_json(metadata_path))
    action_by_id = {int(row["episode_id"]): row["action"] for row in episodes}
    robot_by_id = {
        int(record["episode_id"]): _bound_path(data_root, str(record["relative_path"]))
        for record in expected["robot_files"]
    }
    groups = list(expected["tactile_groups"])
    descriptors: list[Any] = []
    for episode_id in map(int, expected["complete_episode_ids"]):
        if episode_id not in robot_by_id or episode_id not in action_by_id:
            raise ValueError(
                f"bound descriptor inputs are missing: {object_id}/{episode_id}"
            )
        tactile_paths = tuple(
            _bound_path(
                data_root,
                str(group["recordings"][episode_id]["relative_path"]),
            )
            for group in groups
        )
        descriptors.append(
            base.EpisodeDescriptor(
                object_id=object_id,
                episode_id=episode_id,
                action=action_by_id[episode_id],
                robot_path=robot_by_id[episode_id],
                tactile_paths=tactile_paths,
                median_paths=tuple(
                    base.median_path_for(path) for path in tactile_paths
                ),
            )
        )
    if len(descriptors) < minimum_episodes:
        raise ValueError(f"bound descriptor roster is too short: {object_id}")
    if [item.episode_id for item in descriptors] != list(
        map(int, expected["complete_episode_ids"])
    ):
        raise ValueError(f"bound descriptor order changed: {object_id}")
    return descriptors


def self_test() -> None:
    value = {"b": 2, "a": 1}
    if canonical_digest(value) != canonical_digest({"a": 1, "b": 2}):
        raise AssertionError("canonical digest is order dependent")
    try:
        _safe_relative("../escape")
    except ValueError:
        pass
    else:
        raise AssertionError("unsafe path was accepted")
    print("Deform360 bound carrier replay v8 self-test passed")
