"""Fail-closed Stage-1 acquisition for the official Deform360 Hub cohort.

The committed cohort lock permits payload access for calibration objects only.
This module turns an official Hub tree snapshot into an exact file allowlist before
bulk transfer.  It deliberately mirrors the pinned Deform360 processing code's
lexical, exact-stem episode indexing while retaining only one temporally preceding
tactile baseline in the selective local tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_SCHEMA = "bayesian-phystwin/deform360-official-hub-visuotactile-protocol"
SELECTION_SCHEMA = "bayesian-phystwin/deform360-official-hub-selection-v1"
PREFLIGHT_SCHEMA = "bayesian-phystwin/deform360-official-hub-stage1-preflight-v1"
DOWNLOAD_SCHEMA = "bayesian-phystwin/deform360-official-hub-stage1-download-v1"
PROCESSING_VIEW_SCHEMA = (
    "bayesian-phystwin/deform360-official-hub-stage1-processing-view-v1"
)

EXPECTED_PROTOCOL_ID = "deform360-official-hub-visuotactile-v1"
EXPECTED_PROTOCOL_SHA256 = (
    "55534067fb0b3d7965eb66438cbec2ac5b85bcf5378abd1a73785479a5cdbeab"
)
EXPECTED_CONTENT_SELECTION_SHA256 = (
    "f3d3ac25020ec85cad3fadf097259930437baae2b50b4c7f21f61d4823fc649b"
)
EXPECTED_SELECTION_ARTIFACT_SHA256 = (
    "dc1c2d192fbb841d2f0e290d77f21d697983b3f8bfbcae476e71fe902309cd82"
)
EXPECTED_DATASET_REPOSITORY = "brownu/deform360"
EXPECTED_DATASET_REVISION = "f804696d7a133908c7497ffdab43819d879b5cbc"
EXPECTED_PROCESSING_REPOSITORY = "lhy0807/deform360"
EXPECTED_PROCESSING_REVISION = "d8522a4403b766aeb387510c04e89032a56fdf35"

MINIMUM_CAMERA_COUNT = 3
EXPECTED_TACTILE_SENSOR_COUNT = 4
PHYSICAL_BACKEND_MINIMUM_NODE_COUNT = 128

_CAMERA_RE = re.compile(r"^brics-odroid-\d+_cam\d+$")
_TACTILE_RE = re.compile(r"^brics-odroid_tactile[^/]+$")
_TIMESTAMP_RE = re.compile(r".*_(\d+)$")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# These are released directory/metadata differences already recorded by the
# existing official-Hub downloader.  Any new mismatch still fails closed.
RELEASED_METADATA_OBJECT_ALIASES = {
    "026-sock-cloth": "026-sock",
    "112-wristband-cloth": "112-wristband",
    "163-bear": "teddy bear",
    "164-sheep": "white sheep",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON object: {path}") from error
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    result = str(value)
    _require(_HEX64_RE.fullmatch(result) is not None, f"{name} is not SHA-256")
    return result


def _safe_hub_path(value: object, *, prefix: str) -> str:
    path = str(value)
    parsed = PurePosixPath(path)
    _require(
        path == parsed.as_posix()
        and not parsed.is_absolute()
        and ".." not in parsed.parts
        and path.startswith(prefix),
        f"Hub path escaped {prefix}: {path}",
    )
    return path


@dataclass(frozen=True, slots=True)
class SelectedEpisode:
    """One immutable calibration episode from the Stage-0 selection."""

    object_id: str
    stratum: str
    episode_id: int
    metadata_path: str
    metadata_sha256: str


@dataclass(frozen=True, slots=True)
class OfficialHubStage1Lock:
    """Validated lock required before calibration payload access."""

    protocol_id: str
    protocol_sha256: str
    selection_artifact_sha256: str
    selection_file_sha256: str
    dataset_repository: str
    dataset_revision: str
    processing_repository: str
    processing_revision: str
    calibration: tuple[SelectedEpisode, ...]
    confirmation_object_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HubFileRecord:
    """Content identity advertised by one exact official-Hub tree entry."""

    path: str
    size: int
    blob_id: str
    lfs_sha256: str | None = None

    def __post_init__(self) -> None:
        _require(self.size >= 0, f"negative Hub file size: {self.path}")
        _require(
            _HEX40_RE.fullmatch(self.blob_id) is not None,
            f"invalid Git blob identity: {self.path}",
        )
        if self.lfs_sha256 is not None:
            _require_sha256(self.lfs_sha256, name=f"{self.path} LFS SHA-256")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "blob_id": self.blob_id,
            "lfs_sha256": self.lfs_sha256,
        }


def load_official_hub_stage1_lock(
    repository: str | Path,
    protocol_path: str | Path,
    selection_path: str | Path,
) -> OfficialHubStage1Lock:
    """Validate the exact v1 protocol and selection before payload access."""

    root = Path(repository).resolve()
    protocol_file = (root / protocol_path).resolve()
    selection_file = (root / selection_path).resolve()
    protocol = _load_json(protocol_file)
    selection = _load_json(selection_file)

    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "unexpected protocol schema")
    _require(protocol.get("schema_version") == 1, "unsupported protocol version")
    _require(
        protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID,
        "unexpected protocol identity",
    )
    _require(
        protocol.get("status") == "locked-before-official-raw-payload-access",
        "protocol was not locked before payload access",
    )
    protocol_sha256 = _sha256_json(protocol)
    _require(
        protocol_sha256 == EXPECTED_PROTOCOL_SHA256,
        "official-Hub protocol content changed",
    )

    _require(selection.get("schema") == SELECTION_SCHEMA, "unexpected selection schema")
    _require(selection.get("schema_version") == 1, "unsupported selection version")
    _require(
        selection.get("protocol_id") == EXPECTED_PROTOCOL_ID,
        "selection protocol changed",
    )
    _require(
        selection.get("protocol_sha256") == protocol_sha256,
        "selection does not bind the protocol",
    )
    selection_content = dict(selection)
    content_digest = selection_content.pop("content_selection_sha256", None)
    selection_content.pop("implementation_revision", None)
    selection_content.pop("selection_artifact_sha256", None)
    _require(
        content_digest
        == _sha256_json(selection_content)
        == EXPECTED_CONTENT_SELECTION_SHA256,
        "selection content digest changed",
    )
    selection_without_artifact = dict(selection)
    artifact_digest = selection_without_artifact.pop("selection_artifact_sha256", None)
    _require(
        artifact_digest
        == _sha256_json(selection_without_artifact)
        == EXPECTED_SELECTION_ARTIFACT_SHA256,
        "selection artifact digest changed",
    )
    bound_selection = selection.get("selection")
    _require(isinstance(bound_selection, Mapping), "selection cohorts are missing")
    _require(
        selection.get("selection_sha256") == _sha256_json(bound_selection),
        "selected cohort digest changed",
    )

    dataset = selection.get("dataset")
    _require(isinstance(dataset, Mapping), "selection dataset binding is missing")
    _require(
        dataset.get("repo_id") == EXPECTED_DATASET_REPOSITORY
        and dataset.get("resolved_revision") == EXPECTED_DATASET_REVISION,
        "official dataset identity or revision changed",
    )
    processing = selection.get("official_processing")
    _require(isinstance(processing, Mapping), "processing binding is missing")
    _require(
        processing.get("repository") == EXPECTED_PROCESSING_REPOSITORY
        and processing.get("revision") == EXPECTED_PROCESSING_REVISION,
        "official processing identity or revision changed",
    )

    def parse_records(role: str) -> tuple[SelectedEpisode, ...]:
        raw = bound_selection.get(role)
        _require(isinstance(raw, list), f"selection role is missing: {role}")
        records: list[SelectedEpisode] = []
        for value in raw:
            _require(isinstance(value, Mapping), f"invalid {role} selection row")
            object_id = str(value.get("object_id", ""))
            episode_id = value.get("episode_id")
            _require(object_id and "/" not in object_id, f"invalid {role} object ID")
            _require(
                isinstance(episode_id, int)
                and not isinstance(episode_id, bool)
                and episode_id >= 0,
                f"invalid episode for {object_id}",
            )
            metadata_path = str(value.get("metadata_path", ""))
            _require(
                metadata_path == f"raw/{object_id}/metadata.json",
                f"metadata path changed for {object_id}",
            )
            records.append(
                SelectedEpisode(
                    object_id=object_id,
                    stratum=str(value.get("stratum", "")),
                    episode_id=episode_id,
                    metadata_path=metadata_path,
                    metadata_sha256=_require_sha256(
                        value.get("metadata_sha256"),
                        name=f"{object_id} metadata SHA-256",
                    ),
                )
            )
        _require(
            len({record.object_id for record in records}) == len(records),
            f"duplicate object in {role}",
        )
        return tuple(records)

    calibration = parse_records("calibration")
    confirmation = parse_records("confirmation")
    _require(len(calibration) == 10, "calibration cohort is not the locked 10 objects")
    _require(
        len(confirmation) == 12, "confirmation cohort is not the locked 12 objects"
    )
    _require(
        not (
            {item.object_id for item in calibration}
            & {item.object_id for item in confirmation}
        ),
        "calibration and confirmation objects overlap",
    )
    boundary = selection.get("information_boundary")
    _require(isinstance(boundary, Mapping), "selection information boundary missing")
    _require(
        boundary.get("camera_media_opened") is False
        and boundary.get("tactile_arrays_opened") is False
        and boundary.get("target_outcomes_opened") is False,
        "Stage-0 payload boundary changed",
    )
    return OfficialHubStage1Lock(
        protocol_id=EXPECTED_PROTOCOL_ID,
        protocol_sha256=protocol_sha256,
        selection_artifact_sha256=EXPECTED_SELECTION_ARTIFACT_SHA256,
        selection_file_sha256=_file_sha256(selection_file),
        dataset_repository=EXPECTED_DATASET_REPOSITORY,
        dataset_revision=EXPECTED_DATASET_REVISION,
        processing_repository=EXPECTED_PROCESSING_REPOSITORY,
        processing_revision=EXPECTED_PROCESSING_REVISION,
        calibration=calibration,
        confirmation_object_ids=tuple(item.object_id for item in confirmation),
    )


def _validate_metadata(raw: bytes, selected: SelectedEpisode) -> dict[str, Any]:
    _require(
        hashlib.sha256(raw).hexdigest() == selected.metadata_sha256,
        f"metadata checksum changed for {selected.object_id}",
    )
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid metadata JSON for {selected.object_id}") from error
    _require(
        isinstance(metadata, dict), f"metadata is not an object: {selected.object_id}"
    )
    expected_name = RELEASED_METADATA_OBJECT_ALIASES.get(
        selected.object_id, selected.object_id
    )
    _require(
        metadata.get("object") == expected_name,
        f"metadata object identity changed: {selected.object_id}",
    )
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), f"sequences missing: {selected.object_id}")
    expected_ids = {str(index) for index in range(10)}
    _require(
        set(map(str, sequences)) == expected_ids,
        f"episode inventory changed: {selected.object_id}",
    )
    anomalies: list[dict[str, object]] = []
    selected_episode_key = str(selected.episode_id)
    for episode_id in sorted(expected_ids, key=int):
        row = sequences.get(episode_id)
        _require(
            isinstance(row, Mapping),
            f"invalid sequence {episode_id}: {selected.object_id}",
        )
        _require(
            isinstance(row.get("action"), str) and bool(row["action"].strip()),
            f"action is missing for {selected.object_id} episode {episode_id}",
        )
        for name in ("bimanual", "nonprehensile"):
            value = row.get(name)
            if value not in {"yes", "no"}:
                _require(
                    episode_id != selected_episode_key,
                    f"invalid {name} enum for {selected.object_id} episode "
                    f"{episode_id}",
                )
                anomalies.append(
                    {
                        "episode_id": int(episode_id),
                        "field": name,
                        "released_value": value,
                    }
                )
    selected_row = sequences[selected_episode_key]
    return {
        "released_object": metadata["object"],
        "action": selected_row["action"],
        "bimanual": selected_row["bimanual"],
        "nonprehensile": selected_row["nonprehensile"],
        "nonselected_sequence_anomalies": anomalies,
    }


def _records_by_path(
    records: Sequence[HubFileRecord], *, object_id: str
) -> dict[str, HubFileRecord]:
    prefix = f"raw/{object_id}/"
    result: dict[str, HubFileRecord] = {}
    for record in records:
        path = _safe_hub_path(record.path, prefix=prefix)
        _require(path not in result, f"duplicate Hub path: {path}")
        result[path] = record
    _require(result, f"empty Hub tree: {object_id}")
    return result


def _stream_pairs(
    paths: Sequence[str],
    *,
    data_suffix: str,
    exclude_prefix: str | None = None,
    allow_orphan_timestamps: bool = False,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    data: dict[str, str] = {}
    timestamps: dict[str, str] = {}
    for path in paths:
        pure = PurePosixPath(path)
        stem = pure.stem
        if pure.suffix.lower() == data_suffix:
            if exclude_prefix is not None and stem.startswith(exclude_prefix):
                continue
            _require(stem not in data, f"ambiguous data stem {stem}")
            data[stem] = path
        elif pure.suffix.lower() == ".txt":
            _require(stem not in timestamps, f"ambiguous timestamp stem {stem}")
            timestamps[stem] = path
    missing = set(data) - set(timestamps)
    orphan = set(timestamps) - set(data)
    _require(not missing, f"stream has missing timestamp sidecars: {sorted(missing)}")
    _require(
        allow_orphan_timestamps or not orphan,
        f"stream has orphan timestamp sidecars: {sorted(orphan)}",
    )
    return (
        tuple((data[stem], timestamps[stem]) for stem in sorted(data)),
        tuple(sorted(orphan)),
    )


def _terminal_timestamp(path: str) -> int:
    match = _TIMESTAMP_RE.fullmatch(PurePosixPath(path).stem)
    _require(match is not None, f"file has no terminal integer timestamp: {path}")
    return int(match.group(1))


def _select_object_files(
    selected: SelectedEpisode,
    records: Sequence[HubFileRecord],
) -> tuple[tuple[HubFileRecord, ...], dict[str, Any]]:
    by_path = _records_by_path(records, object_id=selected.object_id)
    prefix = f"raw/{selected.object_id}/"
    required_static = (
        selected.metadata_path,
        f"{prefix}calibration_refined/intrinsics.npy",
        f"{prefix}calibration_refined/extrinsics.npy",
        f"{prefix}calibration_refined/dist.npy",
    )
    for path in required_static:
        _require(path in by_path, f"required official file is missing: {path}")

    grouped: dict[str, list[str]] = {}
    for path in by_path:
        relative = PurePosixPath(path).relative_to(prefix)
        first = relative.parts[0]
        grouped.setdefault(first, []).append(path)
    cameras = sorted(name for name in grouped if _CAMERA_RE.fullmatch(name))
    tactile = sorted(name for name in grouped if _TACTILE_RE.fullmatch(name))
    _require(
        len(cameras) >= MINIMUM_CAMERA_COUNT,
        f"{selected.object_id} has fewer than {MINIMUM_CAMERA_COUNT} camera streams",
    )
    _require(
        len(tactile) == EXPECTED_TACTILE_SENSOR_COUNT,
        f"{selected.object_id} does not have exactly four tactile streams",
    )

    chosen = list(required_static)
    camera_stems: dict[str, str] = {}
    camera_orphan_timestamps: dict[str, list[str]] = {}
    for camera in cameras:
        pairs, orphan_timestamps = _stream_pairs(
            grouped[camera],
            data_suffix=".mp4",
            allow_orphan_timestamps=True,
        )
        _require(
            selected.episode_id < len(pairs),
            f"camera {camera} lacks selected episode {selected.episode_id}",
        )
        data_path, timestamp_path = pairs[selected.episode_id]
        chosen.extend((data_path, timestamp_path))
        camera_stems[camera] = PurePosixPath(data_path).stem
        if orphan_timestamps:
            camera_orphan_timestamps[camera] = list(orphan_timestamps)

    tactile_stems: dict[str, str] = {}
    tactile_baselines: dict[str, str] = {}
    for sensor in tactile:
        pairs, orphan_timestamps = _stream_pairs(
            grouped[sensor],
            data_suffix=".npy",
            exclude_prefix="median_",
        )
        _require(
            not orphan_timestamps,
            f"tactile sensor {sensor} has orphan timestamp sidecars",
        )
        _require(
            selected.episode_id < len(pairs),
            f"tactile sensor {sensor} lacks selected episode {selected.episode_id}",
        )
        data_path, timestamp_path = pairs[selected.episode_id]
        recording_time = _terminal_timestamp(data_path)
        baselines = sorted(
            path
            for path in grouped[sensor]
            if PurePosixPath(path).suffix.lower() == ".npy"
            and PurePosixPath(path).stem.startswith("median_")
            and _terminal_timestamp(path) < recording_time
        )
        _require(
            bool(baselines),
            f"tactile sensor {sensor} has no preceding baseline for episode "
            f"{selected.episode_id}",
        )
        baseline = max(baselines, key=_terminal_timestamp)
        chosen.extend((data_path, timestamp_path, baseline))
        tactile_stems[sensor] = PurePosixPath(data_path).stem
        tactile_baselines[sensor] = PurePosixPath(baseline).name

    _require(
        len(chosen) == len(set(chosen)),
        f"duplicate selected path: {selected.object_id}",
    )
    selected_records = tuple(by_path[path] for path in sorted(chosen))
    return selected_records, {
        "camera_count": len(cameras),
        "camera_recording_stems": camera_stems,
        "camera_ignored_orphan_timestamp_stems": camera_orphan_timestamps,
        "tactile_sensor_count": len(tactile),
        "tactile_recording_stems": tactile_stems,
        "tactile_baselines": tactile_baselines,
    }


def build_official_hub_stage1_preflight(
    lock: OfficialHubStage1Lock,
    *,
    tree_by_object: Mapping[str, Sequence[HubFileRecord]],
    metadata_bytes_by_object: Mapping[str, bytes],
) -> dict[str, Any]:
    """Build an exact calibration-only payload plan from target-free Hub trees."""

    calibration_ids = {record.object_id for record in lock.calibration}
    _require(
        set(tree_by_object) == calibration_ids,
        "tree snapshots must contain exactly the locked calibration objects",
    )
    _require(
        set(metadata_bytes_by_object) == calibration_ids,
        "metadata payloads must contain exactly the locked calibration objects",
    )
    _require(
        not (set(tree_by_object) & set(lock.confirmation_object_ids)),
        "confirmation object tree was opened during Stage 1",
    )

    objects: list[dict[str, Any]] = []
    all_paths: set[str] = set()
    for selected in lock.calibration:
        metadata = _validate_metadata(
            metadata_bytes_by_object[selected.object_id], selected
        )
        files, stream_summary = _select_object_files(
            selected, tree_by_object[selected.object_id]
        )
        paths = {record.path for record in files}
        _require(not (paths & all_paths), "selected Hub paths overlap across objects")
        all_paths.update(paths)
        objects.append(
            {
                "object_id": selected.object_id,
                "stratum": selected.stratum,
                "episode_id": selected.episode_id,
                "metadata": metadata,
                **stream_summary,
                "file_count": len(files),
                "total_bytes": sum(record.size for record in files),
                "files": [record.to_dict() for record in files],
            }
        )

    payload: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "schema_version": 1,
        "protocol_id": lock.protocol_id,
        "protocol_sha256": lock.protocol_sha256,
        "selection_artifact_sha256": lock.selection_artifact_sha256,
        "selection_file_sha256": lock.selection_file_sha256,
        "dataset": {
            "repo_id": lock.dataset_repository,
            "resolved_revision": lock.dataset_revision,
        },
        "official_processing": {
            "repository": lock.processing_repository,
            "revision": lock.processing_revision,
            "episode_pairing": "exact-stem pairs sorted lexically by data filename",
            "tactile_baseline": "latest median timestamp strictly before recording",
        },
        "role": "calibration",
        "object_count": len(objects),
        "file_count": sum(int(item["file_count"]) for item in objects),
        "total_bytes": sum(int(item["total_bytes"]) for item in objects),
        "objects": objects,
        "physical_backend_contract": {
            "minimum_node_count": PHYSICAL_BACKEND_MINIMUM_NODE_COUNT,
            "status": "must-pass-after-reconstruction-before-calibration-scoring",
        },
        "information_boundary": {
            "calibration_payload_authorized": True,
            "confirmation_tree_opened": False,
            "confirmation_payload_opened": False,
            "future_target_opened": False,
            "replacement_allowed": False,
            "technical_failures_retained": True,
        },
    }
    payload["preflight_sha256"] = _sha256_json(payload)
    return payload


def validate_official_hub_stage1_preflight(
    value: Mapping[str, Any],
    *,
    lock: OfficialHubStage1Lock | None = None,
) -> None:
    """Validate a serialized preflight before any transfer is attempted."""

    _require(value.get("schema") == PREFLIGHT_SCHEMA, "unexpected preflight schema")
    _require(value.get("schema_version") == 1, "unsupported preflight version")
    _require(value.get("role") == "calibration", "Stage 1 role is not calibration")
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, Mapping), "preflight boundary is missing")
    _require(
        boundary.get("confirmation_tree_opened") is False
        and boundary.get("confirmation_payload_opened") is False
        and boundary.get("future_target_opened") is False,
        "preflight crossed the confirmation boundary",
    )
    canonical = dict(value)
    digest = canonical.pop("preflight_sha256", None)
    _require(digest == _sha256_json(canonical), "preflight digest changed")
    if lock is None:
        return

    _require(value.get("protocol_id") == lock.protocol_id, "preflight protocol changed")
    _require(
        value.get("protocol_sha256") == lock.protocol_sha256,
        "preflight protocol digest changed",
    )
    _require(
        value.get("selection_artifact_sha256") == lock.selection_artifact_sha256,
        "preflight selection digest changed",
    )
    dataset = value.get("dataset")
    _require(isinstance(dataset, Mapping), "preflight dataset binding is missing")
    _require(
        dataset.get("repo_id") == lock.dataset_repository
        and dataset.get("resolved_revision") == lock.dataset_revision,
        "preflight dataset binding changed",
    )
    objects = value.get("objects")
    _require(isinstance(objects, list), "preflight objects are missing")
    expected = {record.object_id: record for record in lock.calibration}
    actual: dict[str, Mapping[str, Any]] = {}
    for row in objects:
        _require(isinstance(row, Mapping), "invalid preflight object row")
        object_id = str(row.get("object_id", ""))
        _require(object_id and "/" not in object_id, "invalid preflight object ID")
        _require(object_id not in actual, f"duplicate preflight object: {object_id}")
        actual[object_id] = row
    _require(
        set(actual) == set(expected),
        "preflight objects do not match the locked calibration cohort",
    )
    _require(
        not (set(actual) & set(lock.confirmation_object_ids)),
        "preflight contains a locked confirmation object",
    )
    for object_id, selected in expected.items():
        row = actual[object_id]
        _require(
            row.get("episode_id") == selected.episode_id
            and row.get("stratum") == selected.stratum,
            f"preflight selected episode changed: {object_id}",
        )
    _require(
        value.get("object_count") == len(objects),
        "preflight object count changed",
    )


def _iter_preflight_files(value: Mapping[str, Any]) -> tuple[HubFileRecord, ...]:
    objects = value.get("objects")
    _require(isinstance(objects, list), "preflight objects are missing")
    result: list[HubFileRecord] = []
    for obj in objects:
        _require(isinstance(obj, Mapping), "invalid preflight object row")
        object_id = str(obj.get("object_id", ""))
        _require(object_id and "/" not in object_id, "invalid preflight object ID")
        files = obj.get("files")
        _require(isinstance(files, list), f"preflight files missing: {object_id}")
        for item in files:
            _require(isinstance(item, Mapping), f"invalid file row: {object_id}")
            result.append(
                HubFileRecord(
                    path=_safe_hub_path(item.get("path"), prefix=f"raw/{object_id}/"),
                    size=int(item.get("size", -1)),
                    blob_id=str(item.get("blob_id", "")),
                    lfs_sha256=(
                        None
                        if item.get("lfs_sha256") is None
                        else str(item["lfs_sha256"])
                    ),
                )
            )
    _require(
        len({item.path for item in result}) == len(result), "duplicate preflight file"
    )
    return tuple(result)


def _validate_download_root(root: Path, allowed: set[str]) -> None:
    if not root.exists():
        return
    _require(root.is_dir(), f"download root is not a directory: {root}")
    entries = tuple(root.rglob("*"))
    _require(
        not any(path.is_symlink() for path in entries),
        "download root contains a symbolic link",
    )
    present = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    _require(
        not (present - allowed),
        f"download root contains unauthorized files: {sorted(present - allowed)[:5]}",
    )


def _verify_download(path: Path, record: HubFileRecord) -> str:
    _require(path.is_file(), f"downloaded file is missing: {record.path}")
    _require(
        path.stat().st_size == record.size, f"downloaded size changed: {record.path}"
    )
    sha256 = _file_sha256(path)
    if record.lfs_sha256 is not None:
        _require(sha256 == record.lfs_sha256, f"LFS content changed: {record.path}")
    else:
        _require(
            _git_blob_sha1(path) == record.blob_id, f"Git blob changed: {record.path}"
        )
    return sha256


HubDownload = Callable[[str], str | Path]


def download_official_hub_stage1(
    preflight: Mapping[str, Any],
    output_root: str | Path,
    *,
    lock: OfficialHubStage1Lock,
    hub_download: HubDownload,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Transfer and verify exactly the files admitted by a Stage-1 preflight."""

    validate_official_hub_stage1_preflight(preflight, lock=lock)
    _require(max_workers >= 1, "download workers must be positive")
    records = _iter_preflight_files(preflight)
    root = Path(output_root).resolve()
    allowed = {record.path for record in records}
    _validate_download_root(root, allowed)
    root.mkdir(parents=True, exist_ok=True)

    def transfer(record: HubFileRecord) -> dict[str, object]:
        destination = root / PurePosixPath(record.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            sha256 = _verify_download(destination, record)
            return {"path": record.path, "sha256": sha256, "reused": True}
        source = Path(hub_download(record.path)).resolve()
        sha256 = _verify_download(source, record)
        temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
        _require(not temporary.exists(), f"stale partial download: {temporary}")
        try:
            shutil.copyfile(source, temporary)
            _verify_download(temporary, record)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return {"path": record.path, "sha256": sha256, "reused": False}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        downloaded = tuple(executor.map(transfer, records))
    _validate_download_root(root, allowed)
    payload: dict[str, Any] = {
        "schema": DOWNLOAD_SCHEMA,
        "schema_version": 1,
        "protocol_id": preflight["protocol_id"],
        "preflight_sha256": preflight["preflight_sha256"],
        "dataset": preflight["dataset"],
        "role": "calibration",
        "file_count": len(downloaded),
        "total_bytes": sum(record.size for record in records),
        "files": sorted(downloaded, key=lambda item: str(item["path"])),
        "information_boundary": {
            "calibration_payload_opened": True,
            "confirmation_payload_opened": False,
            "future_target_opened": False,
        },
    }
    payload["download_sha256"] = _sha256_json(payload)
    return payload


def validate_official_hub_stage1_download(
    value: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    lock: OfficialHubStage1Lock,
) -> dict[str, str]:
    """Validate a download manifest and return its exact per-file digests."""

    validate_official_hub_stage1_preflight(preflight, lock=lock)
    _require(value.get("schema") == DOWNLOAD_SCHEMA, "unexpected download schema")
    _require(value.get("schema_version") == 1, "unsupported download version")
    _require(value.get("role") == "calibration", "download role is not calibration")
    _require(
        value.get("protocol_id") == preflight.get("protocol_id"),
        "download protocol changed",
    )
    _require(
        value.get("preflight_sha256") == preflight.get("preflight_sha256"),
        "download does not bind the preflight",
    )
    _require(value.get("dataset") == preflight.get("dataset"), "dataset changed")
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, Mapping), "download boundary is missing")
    _require(
        boundary.get("calibration_payload_opened") is True
        and boundary.get("confirmation_payload_opened") is False
        and boundary.get("future_target_opened") is False,
        "download crossed the confirmation boundary",
    )
    canonical = dict(value)
    digest = canonical.pop("download_sha256", None)
    _require(digest == _sha256_json(canonical), "download digest changed")

    expected_records = {
        record.path: record for record in _iter_preflight_files(preflight)
    }
    files = value.get("files")
    _require(isinstance(files, list), "download file inventory is missing")
    actual: dict[str, str] = {}
    for item in files:
        _require(isinstance(item, Mapping), "invalid download file row")
        path = str(item.get("path", ""))
        _require(
            path in expected_records, f"download contains unauthorized path: {path}"
        )
        _require(path not in actual, f"duplicate download path: {path}")
        _require(
            isinstance(item.get("reused"), bool),
            f"download reuse flag is invalid: {path}",
        )
        actual[path] = _require_sha256(
            item.get("sha256"), name=f"{path} downloaded SHA-256"
        )
    _require(
        set(actual) == set(expected_records),
        "download inventory does not match the preflight",
    )
    _require(value.get("file_count") == len(actual), "download file count changed")
    _require(
        value.get("total_bytes")
        == sum(record.size for record in expected_records.values()),
        "download byte count changed",
    )
    return actual


def materialize_official_hub_stage1_processing_view(
    preflight: Mapping[str, Any],
    download: Mapping[str, Any],
    payload_root: str | Path,
    output_root: str | Path,
    *,
    lock: OfficialHubStage1Lock,
) -> dict[str, Any]:
    """Create an immutable one-episode view for the pinned official processor.

    The selective payload contains one recording per stream.  The official
    processor therefore assigns that recording local episode index zero.  This
    view records the original-to-local mapping and rewrites only ``metadata.json``
    so downstream action labels follow the selected source episode.  All media,
    timestamps, calibration arrays, and tactile baselines remain symlinks to the
    hash-verified payload.
    """

    file_sha256 = validate_official_hub_stage1_download(
        download, preflight=preflight, lock=lock
    )
    records = {record.path: record for record in _iter_preflight_files(preflight)}
    source_root = Path(payload_root).resolve()
    destination_root = Path(output_root).resolve()
    _require(source_root != destination_root, "processing view equals payload root")
    _require(
        not destination_root.is_relative_to(source_root),
        "processing view may not be nested inside the payload root",
    )
    _validate_download_root(source_root, set(records))
    _require(not destination_root.exists(), "processing view already exists")
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = destination_root.with_name(
        f".{destination_root.name}.partial-{os.getpid()}"
    )
    _require(not temporary_root.exists(), "stale processing-view partial exists")

    preflight_objects = preflight.get("objects")
    _require(isinstance(preflight_objects, list), "preflight objects are missing")
    by_object = {
        str(item.get("object_id")): item
        for item in preflight_objects
        if isinstance(item, Mapping)
    }
    _require(
        len(by_object) == len(preflight_objects), "invalid preflight object inventory"
    )

    object_rows: list[dict[str, Any]] = []
    link_count = 0
    try:
        for selected in lock.calibration:
            row = by_object[selected.object_id]
            source_metadata = source_root / selected.metadata_path
            _require(
                _file_sha256(source_metadata) == file_sha256[selected.metadata_path],
                f"payload metadata changed: {selected.object_id}",
            )
            metadata = _load_json(source_metadata)
            sequences = metadata.get("sequences")
            _require(
                isinstance(sequences, Mapping),
                f"payload sequences missing: {selected.object_id}",
            )
            selected_sequence = sequences.get(str(selected.episode_id))
            _require(
                isinstance(selected_sequence, Mapping),
                f"selected payload sequence missing: {selected.object_id}",
            )
            derived_metadata = dict(metadata)
            derived_metadata["sequences"] = {"0": dict(selected_sequence)}

            object_files = row.get("files")
            _require(
                isinstance(object_files, list),
                f"preflight files missing: {selected.object_id}",
            )
            linked_paths: list[str] = []
            for item in object_files:
                _require(isinstance(item, Mapping), "invalid preflight file row")
                relative = _safe_hub_path(
                    item.get("path"), prefix=f"raw/{selected.object_id}/"
                )
                if relative == selected.metadata_path:
                    continue
                source = source_root / PurePosixPath(relative)
                record = records[relative]
                _require(
                    not source.is_symlink(), f"payload link is forbidden: {relative}"
                )
                _require(source.is_file(), f"payload file is missing: {relative}")
                _require(
                    source.stat().st_size == record.size,
                    f"payload size changed: {relative}",
                )
                _require(
                    _file_sha256(source) == file_sha256[relative],
                    f"payload SHA-256 changed: {relative}",
                )
                destination = temporary_root / PurePosixPath(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source)
                linked_paths.append(relative)
                link_count += 1

            derived_path = temporary_root / selected.metadata_path
            derived_path.parent.mkdir(parents=True, exist_ok=True)
            derived_path.write_text(
                json.dumps(
                    derived_metadata,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            object_rows.append(
                {
                    "object_id": selected.object_id,
                    "stratum": selected.stratum,
                    "source_episode_id": selected.episode_id,
                    "processing_episode_index": 0,
                    "source_metadata_path": selected.metadata_path,
                    "source_metadata_sha256": file_sha256[selected.metadata_path],
                    "derived_metadata_sha256": _file_sha256(derived_path),
                    "action": selected_sequence.get("action"),
                    "bimanual": selected_sequence.get("bimanual"),
                    "nonprehensile": selected_sequence.get("nonprehensile"),
                    "camera_recording_stems": row.get("camera_recording_stems"),
                    "tactile_recording_stems": row.get("tactile_recording_stems"),
                    "tactile_baselines": row.get("tactile_baselines"),
                    "linked_file_count": len(linked_paths),
                    "linked_paths": sorted(linked_paths),
                }
            )

        manifest: dict[str, Any] = {
            "schema": PROCESSING_VIEW_SCHEMA,
            "schema_version": 1,
            "protocol_id": preflight["protocol_id"],
            "preflight_sha256": preflight["preflight_sha256"],
            "download_sha256": download["download_sha256"],
            "dataset": preflight["dataset"],
            "official_processing": preflight["official_processing"],
            "role": "calibration",
            "mapping_rule": (
                "the sole exact-stem recording in each admitted stream is local "
                "episode 0; metadata sequence 0 is copied from the locked source "
                "episode"
            ),
            "object_count": len(object_rows),
            "linked_file_count": link_count,
            "objects": object_rows,
            "information_boundary": {
                "calibration_payload_opened": True,
                "confirmation_payload_opened": False,
                "future_target_opened": False,
                "media_content_modified": False,
                "metadata_derived": True,
            },
        }
        manifest["processing_view_sha256"] = _sha256_json(manifest)
        write_official_hub_stage1_manifest(
            temporary_root / "stage1_processing_view.json", manifest
        )
        os.replace(temporary_root, destination_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return manifest


def validate_official_hub_stage1_processing_view(
    value: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    download: Mapping[str, Any],
    view_root: str | Path,
    payload_root: str | Path,
    lock: OfficialHubStage1Lock,
) -> None:
    """Validate the mapping manifest and every file in a materialized view."""

    file_sha256 = validate_official_hub_stage1_download(
        download, preflight=preflight, lock=lock
    )
    _require(
        value.get("schema") == PROCESSING_VIEW_SCHEMA,
        "unexpected processing-view schema",
    )
    _require(value.get("schema_version") == 1, "unsupported processing-view version")
    _require(value.get("role") == "calibration", "processing-view role changed")
    _require(
        value.get("protocol_id") == preflight.get("protocol_id")
        and value.get("preflight_sha256") == preflight.get("preflight_sha256")
        and value.get("download_sha256") == download.get("download_sha256"),
        "processing view does not bind its inputs",
    )
    _require(value.get("dataset") == preflight.get("dataset"), "dataset changed")
    canonical = dict(value)
    digest = canonical.pop("processing_view_sha256", None)
    _require(digest == _sha256_json(canonical), "processing-view digest changed")
    boundary = value.get("information_boundary")
    _require(isinstance(boundary, Mapping), "processing-view boundary is missing")
    _require(
        boundary.get("confirmation_payload_opened") is False
        and boundary.get("future_target_opened") is False
        and boundary.get("media_content_modified") is False,
        "processing view crossed its information boundary",
    )

    rows = value.get("objects")
    _require(isinstance(rows, list), "processing-view objects are missing")
    by_object: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), "invalid processing-view object row")
        object_id = str(row.get("object_id", ""))
        _require(object_id and "/" not in object_id, "invalid processing object ID")
        _require(
            object_id not in by_object, f"duplicate processing object: {object_id}"
        )
        by_object[object_id] = row
    expected_objects = {item.object_id: item for item in lock.calibration}
    _require(
        set(by_object) == set(expected_objects),
        "processing-view objects do not match calibration lock",
    )
    _require(value.get("object_count") == len(rows), "processing object count changed")

    root = Path(view_root).resolve()
    source_root = Path(payload_root).resolve()
    _require(root.is_dir(), "processing-view root is missing")
    stored_manifest = root / "stage1_processing_view.json"
    _require(stored_manifest.is_file(), "stored processing-view manifest is missing")
    _require(_load_json(stored_manifest) == dict(value), "stored view manifest changed")
    expected_paths = {"stage1_processing_view.json"}
    total_links = 0
    for object_id, selected in expected_objects.items():
        row = by_object[object_id]
        _require(
            row.get("source_episode_id") == selected.episode_id
            and row.get("processing_episode_index") == 0,
            f"processing episode mapping changed: {object_id}",
        )
        metadata_path = selected.metadata_path
        metadata = root / metadata_path
        _require(
            metadata.is_file() and not metadata.is_symlink(), "derived metadata missing"
        )
        _require(
            _file_sha256(metadata) == row.get("derived_metadata_sha256"),
            f"derived metadata changed: {object_id}",
        )
        decoded = _load_json(metadata)
        _require(
            decoded.get("sequences")
            == {
                "0": {
                    "action": row.get("action"),
                    "bimanual": row.get("bimanual"),
                    "nonprehensile": row.get("nonprehensile"),
                }
            },
            f"derived episode metadata changed: {object_id}",
        )
        expected_paths.add(metadata_path)
        linked_paths = row.get("linked_paths")
        _require(isinstance(linked_paths, list), f"linked paths missing: {object_id}")
        _require(
            len(linked_paths) == row.get("linked_file_count"),
            f"linked path count changed: {object_id}",
        )
        for relative_value in linked_paths:
            relative = _safe_hub_path(relative_value, prefix=f"raw/{object_id}/")
            _require(
                relative in file_sha256, f"view link was not downloaded: {relative}"
            )
            path = root / PurePosixPath(relative)
            _require(path.is_symlink(), f"view media is not a link: {relative}")
            source = source_root / PurePosixPath(relative)
            _require(path.resolve() == source, f"view link target changed: {relative}")
            _require(
                _file_sha256(source) == file_sha256[relative],
                f"view source content changed: {relative}",
            )
            expected_paths.add(relative)
            total_links += 1
    _require(
        value.get("linked_file_count") == total_links,
        "processing-view linked file count changed",
    )
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _require(actual_paths == expected_paths, "processing-view file inventory changed")


def write_official_hub_stage1_manifest(
    path: str | Path, value: Mapping[str, Any]
) -> None:
    """Atomically write a canonical, human-readable Stage-1 manifest."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DOWNLOAD_SCHEMA",
    "EXPECTED_DATASET_REVISION",
    "HubFileRecord",
    "OfficialHubStage1Lock",
    "PREFLIGHT_SCHEMA",
    "PROCESSING_VIEW_SCHEMA",
    "SelectedEpisode",
    "build_official_hub_stage1_preflight",
    "download_official_hub_stage1",
    "load_official_hub_stage1_lock",
    "materialize_official_hub_stage1_processing_view",
    "validate_official_hub_stage1_download",
    "validate_official_hub_stage1_processing_view",
    "validate_official_hub_stage1_preflight",
    "write_official_hub_stage1_manifest",
]
