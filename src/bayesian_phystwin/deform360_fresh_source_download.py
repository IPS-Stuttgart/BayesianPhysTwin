"""Queue-bound, outcome-safe download boundary for fresh Deform360 sources."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_REPOSITORY = "brownu/deform360"
DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
QUEUE_KIND = "Deform360FreshSourceStagingQueue"
DOWNLOAD_KIND = "Deform360FreshSourceDownload"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_OBJECT_ID = re.compile(r"^[0-9]{3}-[a-z0-9][a-z0-9-]*$")
_CAMERA_DIR = re.compile(r"^brics-odroid-\d+_cam\d+$")
_TACTILE_DIR = re.compile(r"^brics-odroid_tactile[^/]+$")
_TIMESTAMP_SUFFIX = re.compile(r"_([0-9]+)$")


SnapshotDownload = Callable[..., str]
ListRepoTree = Callable[..., Any]
HubDownload = Callable[..., str]
ListObjectFiles = Callable[[str], list[str]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(
    payload: Mapping[str, Any], *, digest_key: str
) -> str:
    canonical = dict(payload)
    canonical.pop(digest_key, None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


@dataclass(frozen=True)
class FreshSourceDownloadPlan:
    """Exact public dataset request derived from the frozen source queue."""

    repository: str
    revision: str
    queue_sha256: str
    queue_file_sha256: str
    candidates: tuple[tuple[str, int, str], ...]
    allow_patterns: tuple[str, ...]
    ignore_patterns: tuple[str, ...]

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(object_id for object_id, _, _ in self.candidates)


def fresh_source_download_plan(
    queue_path: str | Path,
) -> FreshSourceDownloadPlan:
    """Load and validate the immutable source-only queue."""

    path = Path(queue_path).resolve()
    queue = _load_json(path)
    _require(queue.get("schema_version") == 1, "fresh source queue schema changed")
    _require(queue.get("artifact_kind") == QUEUE_KIND, "wrong queue artifact kind")
    _require(
        queue.get("status") == "source_only_locked_before_payload",
        "fresh source queue is not locked",
    )
    queue_sha256 = queue.get("queue_sha256")
    _require(
        isinstance(queue_sha256, str) and _HEX64.fullmatch(queue_sha256) is not None,
        "fresh source queue digest is malformed",
    )
    if queue.get("contract") == (
        "deform360-causal-response-direct-depth-staging-v14"
    ):
        from .deform360_causal_response_direct_depth_cohort import (
            validate_v14_staging_queue,
        )

        validate_v14_staging_queue(queue)
    else:
        _require(
            queue_sha256 == _canonical_sha256(queue, digest_key="queue_sha256"),
            "fresh source queue checksum changed",
        )
    candidates_raw = queue.get("candidates")
    _require(
        isinstance(candidates_raw, list) and len(candidates_raw) >= 12,
        "fresh source queue is incomplete",
    )
    candidates: list[tuple[str, int, str]] = []
    ranks: list[int] = []
    for row in candidates_raw:
        _require(isinstance(row, Mapping), "fresh source candidate is malformed")
        rank = row.get("queue_rank")
        object_id = row.get("object_id")
        episode_id = row.get("episode_id")
        category = row.get("category")
        _require(
            isinstance(rank, int) and not isinstance(rank, bool),
            "fresh source queue rank is malformed",
        )
        _require(
            isinstance(object_id, str) and _OBJECT_ID.fullmatch(object_id) is not None,
            "fresh source object identity is malformed",
        )
        _require(
            isinstance(episode_id, int)
            and not isinstance(episode_id, bool)
            and episode_id >= 0,
            "fresh source episode identity is malformed",
        )
        _require(
            isinstance(category, str) and bool(category),
            "fresh source category is malformed",
        )
        ranks.append(rank)
        candidates.append((object_id, episode_id, category))
    _require(
        ranks == list(range(1, len(candidates) + 1)),
        "fresh source queue ranks changed",
    )
    _require(
        len({object_id for object_id, _, _ in candidates}) == len(candidates),
        "fresh source queue repeats a physical object",
    )
    source_lock = queue.get("source_lock")
    _require(isinstance(source_lock, Mapping), "source-lock binding is missing")
    _require(
        isinstance(source_lock.get("implementation_commit"), str)
        and _HEX40.fullmatch(str(source_lock["implementation_commit"])) is not None,
        "source-lock implementation commit is malformed",
    )
    boundary = queue.get("information_boundary")
    _require(isinstance(boundary, Mapping), "queue information boundary is missing")
    for key in (
        "episode_media_read_before_queue_lock",
        "processed_geometry_read_before_queue_lock",
        "future_object_positions_deserialized",
        "outcome_or_metric_read",
        "held_v8_target_query_score_barrier_or_outcome_access",
    ):
        _require(boundary.get(key) is False, f"queue crossed information boundary: {key}")
    return FreshSourceDownloadPlan(
        repository=DATASET_REPOSITORY,
        revision=DATASET_REVISION,
        queue_sha256=queue_sha256,
        queue_file_sha256=_file_sha256(path),
        candidates=tuple(candidates),
        allow_patterns=tuple(
            f"raw/{object_id}/*" for object_id, _, _ in candidates
        ),
        ignore_patterns=("*.flac", "*.wav"),
    )


def validate_fresh_download_root(
    output_root: str | Path,
    *,
    plan: FreshSourceDownloadPlan,
    require_complete: bool,
) -> None:
    """Reject contamination by objects outside the frozen queue."""

    raw_root = Path(output_root).resolve() / "raw"
    if not raw_root.exists():
        _require(not require_complete, "fresh source raw root is missing")
        return
    _require(raw_root.is_dir(), "fresh source raw path is not a directory")
    present = {path.name for path in raw_root.iterdir() if path.is_dir()}
    expected = set(plan.object_ids)
    unexpected = sorted(present - expected)
    missing = sorted(expected - present)
    _require(not unexpected, f"unlocked objects exist in download root: {unexpected}")
    if require_complete:
        _require(not missing, f"queued objects are missing from download root: {missing}")


def _validate_metadata(
    metadata_path: Path,
    *,
    object_id: str,
    episode_id: int,
) -> str:
    _require(metadata_path.is_file(), f"object metadata is missing: {object_id}")
    metadata = _load_json(metadata_path)
    _require(
        isinstance(metadata.get("object"), str) and bool(metadata["object"]),
        f"descriptive metadata label is missing: {object_id}",
    )
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), f"metadata sequences are missing: {object_id}")
    _require(
        str(episode_id) in sequences,
        f"queued metadata episode is missing: {object_id}",
    )
    _require(
        all(
            isinstance(row, Mapping) and row.get("bimanual") in {"yes", "no"}
            for row in sequences.values()
        ),
        f"metadata enum domain changed: {object_id}",
    )
    return _file_sha256(metadata_path)


def build_fresh_download_manifest(
    output_root: str | Path,
    *,
    plan: FreshSourceDownloadPlan,
    selected_files_by_object: Mapping[str, tuple[str, ...]] | None = None,
    selected_candidate_ranks: tuple[int, ...] | None = None,
    download_scope: str | None = None,
    tactile_included: bool | None = None,
) -> dict[str, Any]:
    """Inventory the exact download without opening episode payloads."""

    root = Path(output_root).resolve()
    partial = selected_candidate_ranks is not None
    validate_fresh_download_root(root, plan=plan, require_complete=not partial)
    if selected_candidate_ranks is None:
        selected_candidates = tuple(enumerate(plan.candidates, start=1))
    else:
        _require(
            bool(selected_candidate_ranks)
            and tuple(sorted(set(selected_candidate_ranks)))
            == selected_candidate_ranks
            and selected_candidate_ranks[-1] <= len(plan.candidates),
            "selected candidate ranks are invalid",
        )
        selected_candidates = tuple(
            (rank, plan.candidates[rank - 1]) for rank in selected_candidate_ranks
        )
    rows: list[dict[str, Any]] = []
    for queue_rank, (object_id, episode_id, category) in selected_candidates:
        object_root = root / "raw" / object_id
        files = sorted(path for path in object_root.rglob("*") if path.is_file())
        if selected_files_by_object is not None:
            expected = set(selected_files_by_object[object_id])
            observed = {
                path.relative_to(root).as_posix()
                for path in files
            }
            _require(
                observed == expected,
                f"episode-source inventory changed: {object_id}",
            )
        row = {
            "object_id": object_id,
            "episode_id": episode_id,
            "category": category,
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "metadata_sha256": _validate_metadata(
                object_root / "metadata.json",
                object_id=object_id,
                episode_id=episode_id,
            ),
        }
        if partial:
            row["queue_rank"] = queue_rank
        rows.append(row)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": DOWNLOAD_KIND,
        "repository": plan.repository,
        "revision": plan.revision,
        "queue_sha256": plan.queue_sha256,
        "queue_file_sha256": plan.queue_file_sha256,
        "download_scope": (
            download_scope
            if download_scope is not None
            else (
                "full_queued_objects_without_audio"
                if selected_files_by_object is None
                else "queued_episode_camera_source_only"
            )
        ),
        "audio_included": False,
        "tactile_included": (
            selected_files_by_object is None
            if tactile_included is None
            else tactile_included
        ),
        "object_count": len(rows),
        "objects": rows,
        "information_boundary": {
            "queued_objects_only": True,
            "episode_payload_deserialized": False,
            "future_object_positions_deserialized": False,
            "target_metrics_opened": False,
        },
    }
    if partial:
        payload["queue_candidate_count"] = len(plan.candidates)
        payload["selected_candidate_ranks"] = [
            queue_rank for queue_rank, _ in selected_candidates
        ]
    payload["manifest_sha256"] = _canonical_sha256(
        payload, digest_key="manifest_sha256"
    )
    return payload


def select_episode_camera_source_files(
    paths: list[str],
    *,
    object_id: str,
    episode_id: int,
) -> tuple[str, ...]:
    """Select one exact-stem recording per camera plus source calibration."""

    prefix = f"raw/{object_id}/"
    _require(
        bool(paths) and all(path.startswith(prefix) for path in paths),
        f"file index escaped queued object: {object_id}",
    )
    relative = [path[len(prefix) :] for path in paths]
    cameras = sorted(
        {
            Path(path).parts[0]
            for path in relative
            if len(Path(path).parts) == 2
            and _CAMERA_DIR.fullmatch(Path(path).parts[0]) is not None
        }
    )
    _require(bool(cameras), f"queued object has no camera streams: {object_id}")
    selected = {
        f"{prefix}metadata.json",
        *(
            path
            for path in paths
            if path.startswith(f"{prefix}calibration_refined/")
        ),
    }
    for camera in cameras:
        camera_prefix = f"{prefix}{camera}/"
        camera_files = [path for path in paths if path.startswith(camera_prefix)]
        videos = {
            Path(path).stem: path
            for path in camera_files
            if Path(path).suffix.lower() == ".mp4"
        }
        timestamps = {
            Path(path).stem: path
            for path in camera_files
            if Path(path).suffix.lower() == ".txt"
        }
        paired = sorted(set(videos) & set(timestamps))
        _require(
            episode_id < len(paired),
            f"queued episode is missing from camera: {object_id}/{camera}",
        )
        stem = paired[episode_id]
        selected.add(videos[stem])
        selected.add(timestamps[stem])
    _require(
        f"{prefix}metadata.json" in paths,
        f"queued object metadata is absent: {object_id}",
    )
    _require(
        any(path.startswith(f"{prefix}calibration_refined/") for path in selected),
        f"queued object calibration is absent: {object_id}",
    )
    return tuple(sorted(selected))


def _timestamp_from_stem(stem: str) -> int:
    match = _TIMESTAMP_SUFFIX.search(stem)
    _require(match is not None, f"recording stem lacks timestamp: {stem}")
    return int(match.group(1))


def _selected_recording_pair(
    paths: list[str],
    *,
    directory_prefix: str,
    episode_id: int,
    data_suffix: str,
) -> tuple[str, str, str]:
    files = [path for path in paths if path.startswith(directory_prefix)]
    data = {
        Path(path).stem: path
        for path in files
        if Path(path).suffix.lower() == data_suffix
        and not Path(path).stem.startswith("median_")
    }
    timestamps = {
        Path(path).stem: path
        for path in files
        if Path(path).suffix.lower() == ".txt"
    }
    paired = sorted(set(data) & set(timestamps), key=_timestamp_from_stem)
    _require(
        episode_id < len(paired),
        f"queued episode is missing from stream: {directory_prefix}",
    )
    stem = paired[episode_id]
    return stem, data[stem], timestamps[stem]


def select_episode_causal_source_files(
    paths: list[str],
    *,
    object_id: str,
    episode_id: int,
    required_camera_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Select exact camera, tactile, metadata, and calibration inputs for V14."""

    prefix = f"raw/{object_id}/"
    _require(
        bool(paths) and all(path.startswith(prefix) for path in paths),
        f"file index escaped queued object: {object_id}",
    )
    _require(
        required_camera_ids
        and len(set(required_camera_ids)) == len(required_camera_ids)
        and all(_CAMERA_DIR.fullmatch(camera) is not None for camera in required_camera_ids),
        "required camera panel is invalid",
    )
    selected = {
        f"{prefix}metadata.json",
        *(
            path
            for path in paths
            if path.startswith(f"{prefix}calibration_refined/")
        ),
    }
    for camera in required_camera_ids:
        directory = f"{prefix}{camera}/"
        _, video, timestamps = _selected_recording_pair(
            paths,
            directory_prefix=directory,
            episode_id=episode_id,
            data_suffix=".mp4",
        )
        selected.update((video, timestamps))

    tactile_directories = sorted(
        {
            Path(path[len(prefix) :]).parts[0]
            for path in paths
            if len(Path(path[len(prefix) :]).parts) == 2
            and _TACTILE_DIR.fullmatch(
                Path(path[len(prefix) :]).parts[0]
            )
            is not None
        }
    )
    _require(
        bool(tactile_directories),
        f"queued object has no tactile streams: {object_id}",
    )
    for sensor in tactile_directories:
        directory = f"{prefix}{sensor}/"
        stem, data, timestamps = _selected_recording_pair(
            paths,
            directory_prefix=directory,
            episode_id=episode_id,
            data_suffix=".npy",
        )
        recording_timestamp = _timestamp_from_stem(stem)
        baselines = sorted(
            (
                path
                for path in paths
                if path.startswith(directory)
                and Path(path).suffix.lower() == ".npy"
                and Path(path).stem.startswith("median_")
            ),
            key=lambda path: _timestamp_from_stem(Path(path).stem),
        )
        _require(
            bool(baselines),
            f"queued tactile stream lacks a baseline: {object_id}/{sensor}",
        )
        preceding = [
            path
            for path in baselines
            if _timestamp_from_stem(Path(path).stem) <= recording_timestamp
        ]
        baseline = preceding[-1] if preceding else baselines[0]
        selected.update((data, timestamps, baseline))

    _require(
        f"{prefix}metadata.json" in paths,
        f"queued object metadata is absent: {object_id}",
    )
    _require(
        any(path.startswith(f"{prefix}calibration_refined/") for path in selected),
        f"queued object calibration is absent: {object_id}",
    )
    return tuple(sorted(selected))


def download_fresh_source_queue(
    queue_path: str | Path,
    output_root: str | Path,
    *,
    max_workers: int,
    snapshot_download: SnapshotDownload,
) -> dict[str, Any]:
    """Download exactly the queued objects and return a sealed inventory."""

    _require(max_workers >= 1, "download workers must be positive")
    plan = fresh_source_download_plan(queue_path)
    validate_fresh_download_root(output_root, plan=plan, require_complete=False)
    snapshot_download(
        repo_id=plan.repository,
        repo_type="dataset",
        revision=plan.revision,
        local_dir=str(Path(output_root).resolve()),
        allow_patterns=list(plan.allow_patterns),
        ignore_patterns=list(plan.ignore_patterns),
        max_workers=max_workers,
    )
    return build_fresh_download_manifest(output_root, plan=plan)


def download_fresh_source_queue_by_object(
    queue_path: str | Path,
    output_root: str | Path,
    *,
    max_workers: int,
    object_delay_seconds: float,
    list_repo_tree: ListRepoTree,
    hub_download: HubDownload,
) -> dict[str, Any]:
    """Download one queued object subtree at a time to avoid global enumeration."""

    _require(max_workers >= 1, "download workers must be positive")
    _require(object_delay_seconds >= 0.0, "object delay must be non-negative")
    plan = fresh_source_download_plan(queue_path)
    root = Path(output_root).resolve()
    validate_fresh_download_root(root, plan=plan, require_complete=False)
    for object_index, object_id in enumerate(plan.object_ids):
        prefix = f"raw/{object_id}/"
        entries = list(
            list_repo_tree(
                repo_id=plan.repository,
                path_in_repo=f"raw/{object_id}",
                recursive=True,
                expand=False,
                revision=plan.revision,
                repo_type="dataset",
            )
        )
        files = sorted(
            str(entry.path)
            for entry in entries
            if getattr(entry, "blob_id", None) is not None
            and str(entry.path).startswith(prefix)
            and Path(str(entry.path)).suffix.lower() not in {".flac", ".wav"}
        )
        _require(files, f"queued object subtree is empty: {object_id}")
        _require(
            f"raw/{object_id}/metadata.json" in files,
            f"queued object metadata is absent: {object_id}",
        )
        _require(
            all(path.startswith(prefix) for path in files),
            f"object listing escaped its queued subtree: {object_id}",
        )

        def download_one(filename: str) -> str:
            return hub_download(
                repo_id=plan.repository,
                filename=filename,
                repo_type="dataset",
                revision=plan.revision,
                local_dir=str(root),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tuple(executor.map(download_one, files))
        if object_index + 1 < len(plan.object_ids) and object_delay_seconds:
            time.sleep(object_delay_seconds)
    return build_fresh_download_manifest(root, plan=plan)


def download_fresh_episode_sources_from_index(
    queue_path: str | Path,
    output_root: str | Path,
    *,
    max_workers: int,
    object_delay_seconds: float,
    list_object_files: ListObjectFiles,
    hub_download: HubDownload,
) -> dict[str, Any]:
    """Download only the queued episode's camera sources from a pinned file index."""

    _require(max_workers >= 1, "download workers must be positive")
    _require(object_delay_seconds >= 0.0, "object delay must be non-negative")
    plan = fresh_source_download_plan(queue_path)
    root = Path(output_root).resolve()
    validate_fresh_download_root(root, plan=plan, require_complete=False)
    selected_files_by_object: dict[str, tuple[str, ...]] = {}
    for object_index, (object_id, episode_id, _) in enumerate(plan.candidates):
        files = select_episode_camera_source_files(
            list_object_files(object_id),
            object_id=object_id,
            episode_id=episode_id,
        )
        selected_files_by_object[object_id] = files

        def download_one(filename: str) -> str:
            destination = root / filename
            if destination.is_file():
                return str(destination)
            return hub_download(
                repo_id=plan.repository,
                filename=filename,
                repo_type="dataset",
                revision=plan.revision,
                local_dir=str(root),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tuple(executor.map(download_one, files))
        if object_index + 1 < len(plan.object_ids) and object_delay_seconds:
            time.sleep(object_delay_seconds)
    return build_fresh_download_manifest(
        root,
        plan=plan,
        selected_files_by_object=selected_files_by_object,
    )


def download_fresh_causal_episode_sources_from_index(
    queue_path: str | Path,
    output_root: str | Path,
    *,
    candidate_ranks: tuple[int, ...],
    required_camera_ids: tuple[str, ...],
    max_workers: int,
    object_delay_seconds: float,
    list_object_files: ListObjectFiles,
    hub_download: HubDownload,
) -> dict[str, Any]:
    """Download ranked V14 source inputs without deserializing episode payloads."""

    _require(max_workers >= 1, "download workers must be positive")
    _require(object_delay_seconds >= 0.0, "object delay must be non-negative")
    plan = fresh_source_download_plan(queue_path)
    _require(
        bool(candidate_ranks)
        and tuple(sorted(set(candidate_ranks))) == candidate_ranks
        and candidate_ranks[-1] <= len(plan.candidates),
        "candidate ranks are invalid",
    )
    root = Path(output_root).resolve()
    validate_fresh_download_root(root, plan=plan, require_complete=False)
    selected_files_by_object: dict[str, tuple[str, ...]] = {}
    for selected_index, rank in enumerate(candidate_ranks):
        object_id, episode_id, _ = plan.candidates[rank - 1]
        files = select_episode_causal_source_files(
            list_object_files(object_id),
            object_id=object_id,
            episode_id=episode_id,
            required_camera_ids=required_camera_ids,
        )
        selected_files_by_object[object_id] = files

        def download_one(filename: str) -> str:
            destination = root / filename
            if destination.is_file():
                return str(destination)
            return hub_download(
                repo_id=plan.repository,
                filename=filename,
                repo_type="dataset",
                revision=plan.revision,
                local_dir=str(root),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tuple(executor.map(download_one, files))
        if selected_index + 1 < len(candidate_ranks) and object_delay_seconds:
            time.sleep(object_delay_seconds)
    return build_fresh_download_manifest(
        root,
        plan=plan,
        selected_files_by_object=selected_files_by_object,
        selected_candidate_ranks=candidate_ranks,
        download_scope="ranked_queued_episode_causal_source",
        tactile_included=True,
    )


def write_fresh_download_manifest(
    path: str | Path, payload: Mapping[str, Any]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DATASET_REPOSITORY",
    "DATASET_REVISION",
    "FreshSourceDownloadPlan",
    "build_fresh_download_manifest",
    "download_fresh_causal_episode_sources_from_index",
    "download_fresh_episode_sources_from_index",
    "download_fresh_source_queue",
    "download_fresh_source_queue_by_object",
    "fresh_source_download_plan",
    "select_episode_camera_source_files",
    "select_episode_causal_source_files",
    "validate_fresh_download_root",
    "write_fresh_download_manifest",
]
