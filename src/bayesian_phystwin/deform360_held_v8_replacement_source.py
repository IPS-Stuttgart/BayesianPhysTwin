"""Leakage-safe acquisition of the fresh Deform360 v8 replacement source.

The v8 calibration panel replaces the retired ``002-rope-silk-ep0003`` case
with public episode three of ``072-cotton-clohesline``.  This module is the
only operator that may acquire that replacement.  It intentionally downloads
neither the rest of object 072 nor any tactile/audio stream:

* enumerate the pinned Hugging Face revision;
* prove that every one of the 37 camera streams has ten exact-stem
  MP4/timestamp pairs;
* select pair index three independently in every sorted stream;
* add only the three refined-calibration files and ``metadata.json``;
* pass those exact 78 paths as ``allow_patterns`` to ``snapshot_download``;
* re-hash all downloaded bytes against the independently frozen content pin;
* run the pinned public Deform360 undistortion at local subset index zero,
  relabel its directory to canonical ``episode_0003``, and recover the
  monomanual robot trajectory with seed zero.

The source capability is deliberately injected.  It is consumed before any
remote inventory request, download, or processing action.  Formal protocol
code can therefore own a process-local, single-use permit without this module
learning how the permit is represented.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Protocol

from .deform360_robot_kinematics import (
    load_robot_kinematics_archive,
    robot_kinematics_array_records,
)


PROTOCOL_ID = "deform360-held-online-belief-v8"
SCHEMA_VERSION = 1

HF_REPO_ID = "brownu/deform360"
HF_REPO_TYPE = "dataset"
HF_DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
PROCESSING_CODE_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"

REPLACEMENT_OBJECT_ID = "072-cotton-clohesline"
REPLACEMENT_EPISODE_ID = 3
REPLACEMENT_EPISODE_LABEL = "0003"
REPLACEMENT_CASE_NAME = "072-cotton-clohesline-ep0003"
REMOTE_OBJECT_ROOT = f"raw/{REPLACEMENT_OBJECT_ID}"
LOCAL_SUBSET_EPISODE_INDEX = 0
SOURCE_OPERATION = "acquire-aligned-replacement-source-v1"

EXPECTED_CAMERA_COUNT = 37
EXPECTED_PAIR_COUNT_PER_CAMERA = 10
EXPECTED_SELECTED_RECORD_COUNT = 78
EXPECTED_DOWNLOAD_SIZE_BYTES = 95_165_257
REFERENCE_CAMERA = "brics-odroid-001_cam0"
REFERENCE_SELECTED_STEM = "brics-odroid-001_cam0_1769121824929789"

UNCALIBRATED_CAMERAS = (
    "brics-odroid-004_cam0",
    "brics-odroid-014_cam0",
    "brics-odroid-018_cam0",
    "brics-odroid-018_cam1",
    "brics-odroid-019_cam0",
)
EXPECTED_ALIGNED_CAMERA_COUNT = 32

INVENTORY_RECORDS_SHA256 = (
    "9a3c4755cc635bdd1702d6739f540401c532dae3c394f85416da54e81161a839"
)
DOWNLOADED_CONTENT_RECORDS_SHA256 = (
    "874c31f58f9f6679e6d625621e37a9d2591c708c551c0fd592b101c770650df2"
)

SEMANTICS = {
    "semantic_label": "rope",
    "action": "drag",
    "action_location": "center",
    "bimanual": False,
    "prehensile": True,
}

PUBLIC_METADATA_EVIDENCE = {
    "object": REPLACEMENT_OBJECT_ID,
    "sequences.3.action": "drag center",
    "sequences.3.bimanual": "no",
    "sequences.3.nonprehensile": "no",
}

REPLACEMENT_SOURCE_INVENTORY_CONTRACT = {
    "contract_id": "deform360-held-v8-replacement-source-inventory-v1",
    "case_name": REPLACEMENT_CASE_NAME,
    "object_id": REPLACEMENT_OBJECT_ID,
    "episode_id": REPLACEMENT_EPISODE_LABEL,
    "hf_repo_id": HF_REPO_ID,
    "hf_repo_type": HF_REPO_TYPE,
    "hf_dataset_revision": HF_DATASET_REVISION,
    "remote_object_root": REMOTE_OBJECT_ROOT,
    "selection": {
        "camera_count": EXPECTED_CAMERA_COUNT,
        "pairs_per_camera": EXPECTED_PAIR_COUNT_PER_CAMERA,
        "zero_based_sorted_pair_index": REPLACEMENT_EPISODE_ID,
        "selected_record_count": EXPECTED_SELECTED_RECORD_COUNT,
        "selected_total_size_bytes": EXPECTED_DOWNLOAD_SIZE_BYTES,
        "inventory_records_sha256": INVENTORY_RECORDS_SHA256,
        "downloaded_content_records_sha256": DOWNLOADED_CONTENT_RECORDS_SHA256,
        "tactile_audio_or_other_episode_permitted": False,
    },
    "processing": {
        "code_revision": PROCESSING_CODE_REVISION,
        "isolated_subset_local_episode_index": LOCAL_SUBSET_EPISODE_INDEX,
        "canonical_episode_id": REPLACEMENT_EPISODE_LABEL,
        "robot_bimanual": False,
        "robot_seed": 0,
        "downloaded_camera_count": EXPECTED_CAMERA_COUNT,
        "aligned_camera_count": EXPECTED_ALIGNED_CAMERA_COUNT,
        "uncalibrated_skipped_cameras": list(UNCALIBRATED_CAMERAS),
    },
    "semantics": SEMANTICS,
}

REMOTE_INVENTORY_MANIFEST_KIND = "Deform360HeldV8ReplacementRemoteInventoryManifest"
DOWNLOADED_CONTENT_MANIFEST_KIND = "Deform360HeldV8ReplacementDownloadedContentManifest"
ALIGNED_SOURCE_MANIFEST_KIND = "Deform360HeldV8AlignedReplacementSourceManifest"

_CALIBRATION_RELATIVE_PATHS = (
    "calibration_refined/dist.npy",
    "calibration_refined/extrinsics.npy",
    "calibration_refined/intrinsics.npy",
)
_CAMERA_RE = re.compile(r"^brics-odroid-\d+_cam\d+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40}$")

_CAMERA_OUTPUT_FILENAMES = frozenset(
    {
        "undistorted.mp4",
        "undistorted_000000.png",
        "aligned_timestamps.txt",
        "alignment.json",
        "metadata.json",
    }
)
_EPISODE_OUTPUT_FILENAMES = frozenset(
    {"undistorted_intrinsics.npy", "extrinsics.npy", "alignment.json"}
)
_ROBOT_OUTPUT_FILENAMES = frozenset({"robot.npz", "robot.meta.json"})

_UNDISTORT_SCRIPT = """
import json
import sys
from deform360.undistort import undistort_episode

undistort_episode(
    object_dir=sys.argv[1],
    output_dir=sys.argv[2],
    episode_index=0,
    cameras=json.loads(sys.argv[3]),
    tol_units=100000,
    overwrite=True,
    rebuild_timeline=False,
)
""".strip()

_ROBOT_SCRIPT = """
import json
import sys
from deform360.processing.robot_stage import process_robot_episode

process_robot_episode(
    aligned_dir=sys.argv[1],
    episode_index=3,
    bimanual=False,
    cameras=json.loads(sys.argv[2]),
    seed=0,
    overwrite=True,
    plot=False,
)
""".strip()


class SourcePermitConsumer(Protocol):
    """Protocol-owned validator for a process-local, single-use capability."""

    def __call__(
        self,
        permit: object,
        *,
        case_name: str,
        operation: str,
    ) -> Mapping[str, Any]: ...


class InventoryProvider(Protocol):
    def __call__(self, **kwargs: Any) -> Iterable[object]: ...


class SnapshotDownloader(Protocol):
    def __call__(self, **kwargs: Any) -> str | os.PathLike[str]: ...


class CommandRunner(Protocol):
    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> object: ...


@dataclass(frozen=True)
class ReplacementSourcePaths:
    """Fresh destinations for one formal source-acquisition operation."""

    download_root: Path
    aligned_root: Path
    inventory_manifest: Path
    content_manifest: Path
    aligned_source_manifest: Path
    processing_code_root: Path
    python_executable: Path


@dataclass(frozen=True)
class SelectedInventory:
    records: tuple[dict[str, Any], ...]
    allow_patterns: tuple[str, ...]
    camera_names: tuple[str, ...]
    selected_stems: Mapping[str, str]
    total_size_bytes: int
    records_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _artifact_sha256(artifact: Mapping[str, Any]) -> str:
    value = dict(artifact)
    value.pop("artifact_sha256", None)
    return _canonical_sha256(value)


def _json_value(value: object, *, label: str) -> Any:
    """Round-trip one capability record through strict canonical JSON."""

    try:
        payload = _canonical_bytes(value)
        result = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a finite JSON value") from error
    return result


def _absolute_lexical(path: str | os.PathLike[str], *, label: str) -> Path:
    raw = os.fspath(path)
    value = Path(raw)
    _require(value.is_absolute(), f"{label} must be absolute")
    _require(".." not in value.parts, f"{label} contains parent traversal")
    _require(
        os.path.normpath(raw) == raw,
        f"{label} is not lexically canonical",
    )
    return value


def _existing_canonical_directory(path: str | os.PathLike[str], *, label: str) -> Path:
    value = _absolute_lexical(path, label=label)
    try:
        observed = os.lstat(value)
    except OSError as error:
        raise ValueError(f"{label} is missing: {value}") from error
    _require(stat.S_ISDIR(observed.st_mode), f"{label} is not a directory")
    _require(not stat.S_ISLNK(observed.st_mode), f"{label} is a symlink")
    _require(value.resolve(strict=True) == value, f"{label} has a symlink ancestor")
    return value


def _fresh_root(path: str | os.PathLike[str], *, label: str) -> Path:
    value = _absolute_lexical(path, label=label)
    _require(not os.path.lexists(value), f"{label} is not fresh: {value}")
    _existing_canonical_directory(value.parent, label=f"{label} parent")
    return value


def _fresh_manifest_path(path: str | os.PathLike[str], *, label: str) -> Path:
    value = _absolute_lexical(path, label=label)
    _require(not os.path.lexists(value), f"{label} already exists: {value}")
    _existing_canonical_directory(value.parent, label=f"{label} parent")
    return value


def _validated_executable(path: str | os.PathLike[str], *, label: str) -> Path:
    """Validate a pinned interpreter path, permitting a virtualenv symlink.

    The formal held runtime's ``bin/python`` is a deliberate symlink to the
    system interpreter; invoking it through that lexical path is what activates
    the adjacent ``pyvenv.cfg``.  Dataset files and manifests remain strictly
    no-follow.  This narrowly scoped validator resolves the executable chain,
    requires a regular executable target, and preserves the pinned lexical path
    for invocation and provenance.
    """

    value = _absolute_lexical(path, label=label)
    try:
        observed = os.lstat(value)
        resolved = value.resolve(strict=True)
        target = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} is missing: {value}") from error
    _require(
        stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode),
        f"{label} is neither a regular file nor a symlink",
    )
    _require(stat.S_ISREG(target.st_mode), f"{label} target is not a regular file")
    _require(os.access(value, os.X_OK), f"{label} is not executable")
    return value


def _open_regular_snapshot(
    path: str | os.PathLike[str], *, label: str, exact_mode: int | None = None
) -> tuple[Path, int, os.stat_result]:
    source = _absolute_lexical(path, label=label)
    try:
        before = os.lstat(source)
    except OSError as error:
        raise ValueError(f"{label} is missing: {source}") from error
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    _require(not stat.S_ISLNK(before.st_mode), f"{label} is a symlink")
    _require(source.resolve(strict=True) == source, f"{label} has a symlink ancestor")
    if exact_mode is not None:
        _require(
            stat.S_IMODE(before.st_mode) == exact_mode,
            f"{label} mode must be {oct(exact_mode)}",
        )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{label} changed while opening",
        )
    except BaseException:
        os.close(descriptor)
        raise
    return source, descriptor, opened


def _unchanged_open_file(
    source: Path,
    descriptor: int,
    opened: os.stat_result,
    *,
    label: str,
) -> os.stat_result:
    after = os.fstat(descriptor)
    current = os.lstat(source)
    identity = (opened.st_dev, opened.st_ino)
    _require(
        (after.st_dev, after.st_ino) == identity
        and (current.st_dev, current.st_ino) == identity
        and after.st_size == opened.st_size
        and after.st_mtime_ns == opened.st_mtime_ns
        and after.st_ctime_ns == opened.st_ctime_ns,
        f"{label} changed while reading",
    )
    return after


def _file_record(
    path: str | os.PathLike[str],
    *,
    label: str,
    record_path: str | None = None,
    exact_mode: int | None = None,
) -> dict[str, Any]:
    source, descriptor, opened = _open_regular_snapshot(
        path, label=label, exact_mode=exact_mode
    )
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = _unchanged_open_file(source, descriptor, opened, label=label)
    finally:
        os.close(descriptor)
    return {
        "path": str(source) if record_path is None else record_path,
        "size_bytes": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def _load_json(
    path: str | os.PathLike[str], *, label: str, exact_mode: int | None = None
) -> dict[str, Any]:
    source, descriptor, opened = _open_regular_snapshot(
        path, label=label, exact_mode=exact_mode
    )
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as stream:
            value = json.load(stream)
        _unchanged_open_file(source, descriptor, opened, label=label)
    finally:
        os.close(descriptor)
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    return value


def _write_sealed_json(path: Path, artifact: Mapping[str, Any]) -> Path:
    payload = (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    try:
        os.fchmod(descriptor, 0o400)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o400,
        "sealed manifest was not created as an exact mode-0400 regular file",
    )
    return path


def _field(item: object, *names: str) -> object:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return None


def _inventory_file_record(item: object) -> dict[str, Any] | None:
    item_type = _field(item, "type")
    if item_type not in (None, "file"):
        return None
    path = _field(item, "path", "rfilename")
    size = _field(item, "size")
    blob_id = _field(item, "blob_id", "oid")
    lfs = _field(item, "lfs")
    lfs_sha256: object = None
    if lfs is not None:
        lfs_sha256 = _field(lfs, "sha256", "oid")
    _require(isinstance(path, str), "remote inventory file path is invalid")
    _require(type(size) is int and size >= 0, f"remote size is invalid: {path}")
    _require(
        isinstance(blob_id, str) and _GIT_BLOB_RE.fullmatch(blob_id) is not None,
        f"remote blob id is invalid: {path}",
    )
    _require(
        lfs_sha256 is None
        or (
            isinstance(lfs_sha256, str) and _SHA256_RE.fullmatch(lfs_sha256) is not None
        ),
        f"remote LFS digest is invalid: {path}",
    )
    return {
        "path": path,
        "size_bytes": size,
        "blob_id": blob_id,
        "lfs_sha256": lfs_sha256,
    }


def derive_selected_inventory(
    entries: Iterable[object],
    *,
    expected_records_sha256: str | None = None,
    expected_total_size_bytes: int | None = None,
) -> SelectedInventory:
    """Derive the exact public episode-three allowlist from a remote tree."""

    if expected_records_sha256 is None:
        expected_records_sha256 = INVENTORY_RECORDS_SHA256
    if expected_total_size_bytes is None:
        expected_total_size_bytes = EXPECTED_DOWNLOAD_SIZE_BYTES

    by_path: dict[str, dict[str, Any]] = {}
    for item in entries:
        record = _inventory_file_record(item)
        if record is None:
            continue
        path = record["path"]
        _require(path not in by_path, f"duplicate remote inventory path: {path}")
        by_path[path] = record

    camera_files: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    prefix = f"{REMOTE_OBJECT_ROOT}/"
    for path, record in by_path.items():
        if not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        parts = relative.split("/")
        if len(parts) == 2 and _CAMERA_RE.fullmatch(parts[0]) is not None:
            camera_files[parts[0]].append(record)

    camera_names = tuple(sorted(camera_files))
    _require(
        len(camera_names) == EXPECTED_CAMERA_COUNT,
        f"expected {EXPECTED_CAMERA_COUNT} camera streams, found {len(camera_names)}",
    )
    selected: list[dict[str, Any]] = []
    selected_stems: dict[str, str] = {}
    for camera in camera_names:
        records = camera_files[camera]
        mp4 = sorted(
            (record for record in records if str(record["path"]).endswith(".mp4")),
            key=lambda value: str(value["path"]),
        )
        timestamps = sorted(
            (record for record in records if str(record["path"]).endswith(".txt")),
            key=lambda value: str(value["path"]),
        )
        _require(
            len(records) == 2 * EXPECTED_PAIR_COUNT_PER_CAMERA
            and len(mp4) == EXPECTED_PAIR_COUNT_PER_CAMERA
            and len(timestamps) == EXPECTED_PAIR_COUNT_PER_CAMERA,
            f"{camera} does not contain exactly ten MP4/TXT pairs",
        )
        mp4_stems = [str(record["path"]).rsplit(".", 1)[0] for record in mp4]
        timestamp_stems = [
            str(record["path"]).rsplit(".", 1)[0] for record in timestamps
        ]
        _require(
            mp4_stems == timestamp_stems,
            f"{camera} MP4 and timestamp stems are not exact pairs",
        )
        selected.extend(
            (mp4[REPLACEMENT_EPISODE_ID], timestamps[REPLACEMENT_EPISODE_ID])
        )
        selected_stems[camera] = Path(mp4_stems[REPLACEMENT_EPISODE_ID]).name

    for relative in (*_CALIBRATION_RELATIVE_PATHS, "metadata.json"):
        path = f"{REMOTE_OBJECT_ROOT}/{relative}"
        _require(path in by_path, f"required remote source is missing: {path}")
        selected.append(by_path[path])

    selected.sort(key=lambda value: str(value["path"]))
    paths = tuple(str(record["path"]) for record in selected)
    _require(
        len(selected) == EXPECTED_SELECTED_RECORD_COUNT
        and len(set(paths)) == EXPECTED_SELECTED_RECORD_COUNT,
        "selected remote source set is not exactly 78 unique records",
    )
    _require(
        selected_stems.get(REFERENCE_CAMERA) == REFERENCE_SELECTED_STEM,
        "reference-camera episode-three stem changed",
    )
    _require(
        not any(
            "tactile" in path.lower() or path.lower().endswith(".wav") for path in paths
        ),
        "tactile or audio payload entered the source allowlist",
    )
    total_size = sum(int(record["size_bytes"]) for record in selected)
    digest = _canonical_sha256(selected)
    _require(
        total_size == expected_total_size_bytes,
        f"selected source size changed: {total_size}",
    )
    _require(
        digest == expected_records_sha256,
        f"selected remote inventory digest changed: {digest}",
    )
    return SelectedInventory(
        records=tuple(selected),
        allow_patterns=paths,
        camera_names=camera_names,
        selected_stems=dict(sorted(selected_stems.items())),
        total_size_bytes=total_size,
        records_sha256=digest,
    )


def _default_inventory_provider(**kwargs: Any) -> Iterable[object]:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "huggingface_hub is required for formal source acquisition"
        ) from error
    api = HfApi()
    return api.list_repo_tree(**kwargs)


def _default_snapshot_downloader(**kwargs: Any) -> str | os.PathLike[str]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "huggingface_hub is required for formal source acquisition"
        ) from error
    return snapshot_download(**kwargs)


def _default_command_runner(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> object:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env),
        check=True,
    )


def _git_revision(code_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(code_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _scan_exact_download(
    download_root: Path,
    selected: SelectedInventory,
    *,
    expected_records_sha256: str | None = None,
) -> tuple[dict[str, Any], ...]:
    if expected_records_sha256 is None:
        expected_records_sha256 = DOWNLOADED_CONTENT_RECORDS_SHA256
    raw_repo_root = download_root / "raw"
    raw_object = raw_repo_root / REPLACEMENT_OBJECT_ID
    _existing_canonical_directory(raw_repo_root, label="downloaded raw root")
    _existing_canonical_directory(raw_object, label="downloaded object root")

    all_entries = sorted(raw_repo_root.rglob("*"))
    _require(
        all(not entry.is_symlink() for entry in all_entries),
        "downloaded raw source contains a symlink",
    )
    files = [entry for entry in all_entries if entry.is_file()]
    _require(
        all(entry.is_file() or entry.is_dir() for entry in all_entries),
        "downloaded raw source contains a non-file, non-directory entry",
    )
    relative_files = tuple(
        sorted(entry.relative_to(download_root).as_posix() for entry in files)
    )
    _require(
        relative_files == selected.allow_patterns,
        "downloaded raw tree differs from the exact 78-path allowlist",
    )
    observed_directory_set = {
        entry.relative_to(download_root).as_posix()
        for entry in all_entries
        if entry.is_dir()
    }
    observed_directory_set.add(raw_repo_root.relative_to(download_root).as_posix())
    observed_directories = frozenset(observed_directory_set)
    expected_directories: set[str] = set()
    for relative in selected.allow_patterns:
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    _require(
        observed_directories == frozenset(expected_directories),
        "downloaded raw tree contains an unexpected directory",
    )

    records = tuple(
        _file_record(
            download_root / path,
            label=f"downloaded source {path}",
            record_path=path,
        )
        for path in selected.allow_patterns
    )
    _require(
        sum(int(record["size_bytes"]) for record in records)
        == EXPECTED_DOWNLOAD_SIZE_BYTES,
        "downloaded source byte count changed",
    )
    digest = _canonical_sha256(records)
    _require(
        digest == expected_records_sha256,
        f"downloaded content digest changed: {digest}",
    )
    return records


def _validate_public_metadata(raw_object: Path) -> dict[str, Any]:
    metadata = _load_json(raw_object / "metadata.json", label="public metadata")
    _require(
        metadata.get("object") == REPLACEMENT_OBJECT_ID,
        "public metadata object changed",
    )
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), "public metadata sequences are invalid")
    sequence = sequences.get(str(REPLACEMENT_EPISODE_ID))
    _require(isinstance(sequence, Mapping), "public episode-three metadata is missing")
    _require(sequence.get("action") == "drag center", "public action changed")
    _require(sequence.get("bimanual") == "no", "public bimanual flag changed")
    _require(
        sequence.get("nonprehensile") == "no",
        "public prehensile flag changed",
    )
    return metadata


def _validate_episode_layout(
    episode_dir: Path,
    *,
    expected_cameras: tuple[str, ...],
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], int]:
    _existing_canonical_directory(episode_dir, label="aligned replacement episode")
    entries = sorted(episode_dir.rglob("*"))
    _require(
        all(not entry.is_symlink() for entry in entries),
        "aligned replacement source contains a symlink",
    )
    _require(
        all(entry.is_file() or entry.is_dir() for entry in entries),
        "aligned replacement source contains a non-file, non-directory entry",
    )

    observed_cameras = tuple(
        sorted(
            entry.name
            for entry in episode_dir.iterdir()
            if entry.is_dir() and _CAMERA_RE.fullmatch(entry.name) is not None
        )
    )
    _require(observed_cameras == expected_cameras, "aligned camera set changed")
    _require(
        len(observed_cameras) == EXPECTED_ALIGNED_CAMERA_COUNT,
        "aligned camera count changed",
    )
    for camera in expected_cameras:
        camera_dir = _existing_canonical_directory(
            episode_dir / camera, label=f"aligned camera {camera}"
        )
        names = frozenset(entry.name for entry in camera_dir.iterdir())
        _require(
            names == _CAMERA_OUTPUT_FILENAMES
            and all(entry.is_file() for entry in camera_dir.iterdir()),
            f"aligned camera output set changed: {camera}",
        )

    root_files = frozenset(
        entry.name for entry in episode_dir.iterdir() if entry.is_file()
    )
    root_dirs = frozenset(
        entry.name for entry in episode_dir.iterdir() if entry.is_dir()
    )
    _require(root_files == _EPISODE_OUTPUT_FILENAMES, "episode root files changed")
    _require(
        root_dirs == frozenset((*expected_cameras, "robot")),
        "episode root directories changed",
    )
    robot_dir = _existing_canonical_directory(
        episode_dir / "robot", label="robot output directory"
    )
    _require(
        frozenset(entry.name for entry in robot_dir.iterdir())
        == _ROBOT_OUTPUT_FILENAMES
        and all(entry.is_file() for entry in robot_dir.iterdir()),
        "robot output set changed",
    )

    alignment = _load_json(episode_dir / "alignment.json", label="alignment metadata")
    _require(
        alignment.get("episode_index") == LOCAL_SUBSET_EPISODE_INDEX,
        "embedded alignment index is not the isolated local index zero",
    )
    _require(
        alignment.get("cameras") == list(expected_cameras),
        "alignment metadata camera set changed",
    )
    frame_count = alignment.get("frame_count")
    _require(
        type(frame_count) is int and frame_count > 0, "invalid aligned frame count"
    )

    robot_meta = _load_json(
        robot_dir / "robot.meta.json", label="robot provenance metadata"
    )
    parameters = robot_meta.get("parameters")
    outputs = robot_meta.get("outputs")
    _require(isinstance(parameters, Mapping), "robot parameters are invalid")
    _require(isinstance(outputs, Mapping), "robot outputs are invalid")
    _require(parameters.get("seed") == 0, "robot recovery seed changed")
    _require(parameters.get("bimanual") is False, "robot recovery is not monomanual")
    _require(
        parameters.get("cameras") == list(expected_cameras),
        "robot recovery camera set changed",
    )
    _require(outputs.get("bimanual") is False, "robot output bimanual flag changed")
    _require(
        outputs.get("num_frames") == frame_count,
        "robot and aligned frame counts differ",
    )
    state = load_robot_kinematics_archive(
        robot_dir / "robot.npz", expected_frame_count=frame_count
    )
    _require(not state.bimanual, "robot archive is not monomanual")

    records = tuple(
        _file_record(
            entry,
            label=f"aligned source {entry.relative_to(episode_dir).as_posix()}",
            record_path=(
                f"{REPLACEMENT_OBJECT_ID}/episode_{REPLACEMENT_EPISODE_ID:04d}/"
                f"{entry.relative_to(episode_dir).as_posix()}"
            ),
        )
        for entry in sorted(path for path in episode_dir.rglob("*") if path.is_file())
    )
    _require(
        len(records)
        == EXPECTED_ALIGNED_CAMERA_COUNT * len(_CAMERA_OUTPUT_FILENAMES)
        + len(_EPISODE_OUTPUT_FILENAMES)
        + len(_ROBOT_OUTPUT_FILENAMES),
        "aligned source regular-file count changed",
    )
    return robot_meta, records, frame_count


def _manifest_artifact(*, kind: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    artifact = {
        "kind": kind,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        **dict(fields),
    }
    artifact["artifact_sha256"] = _artifact_sha256(artifact)
    return artifact


def _validate_artifact_common(
    artifact: Mapping[str, Any], *, kind: str, label: str
) -> None:
    _require(artifact.get("kind") == kind, f"{label} kind changed")
    _require(
        artifact.get("schema_version") == SCHEMA_VERSION,
        f"{label} schema changed",
    )
    _require(artifact.get("protocol_id") == PROTOCOL_ID, f"{label} protocol changed")
    _require(
        artifact.get("artifact_sha256") == _artifact_sha256(artifact),
        f"{label} artifact checksum changed",
    )


def acquire_and_align_replacement_source(
    paths: ReplacementSourcePaths,
    *,
    source_permit: object,
    consume_source_permit: SourcePermitConsumer,
    expected_source_permit: Mapping[str, Any] | None = None,
    inventory_provider: InventoryProvider | None = None,
    snapshot_downloader: SnapshotDownloader | None = None,
    command_runner: CommandRunner | None = None,
    revision_reader: Callable[[Path], str] | None = None,
) -> Path:
    """Acquire, process, and seal the exact held-v8 replacement source.

    All injected I/O callbacks are present solely to make the operator testable
    without a network or video codec.  They cannot change any formal source pin
    or processing parameter.
    """

    _require(callable(consume_source_permit), "source permit consumer is missing")
    permit_record = consume_source_permit(
        source_permit,
        case_name=REPLACEMENT_CASE_NAME,
        operation=SOURCE_OPERATION,
    )
    _require(isinstance(permit_record, Mapping), "source permit record is invalid")
    permit_json = _json_value(dict(permit_record), label="source permit record")
    _require(isinstance(permit_json, dict), "source permit record is not an object")
    if expected_source_permit is not None:
        expected_json = _json_value(
            dict(expected_source_permit), label="expected source permit"
        )
        _require(permit_json == expected_json, "source permit lock binding changed")

    download_root = _fresh_root(paths.download_root, label="download root")
    aligned_root = _fresh_root(paths.aligned_root, label="aligned root")
    inventory_manifest_path = _fresh_manifest_path(
        paths.inventory_manifest, label="inventory manifest"
    )
    content_manifest_path = _fresh_manifest_path(
        paths.content_manifest, label="content manifest"
    )
    aligned_manifest_path = _fresh_manifest_path(
        paths.aligned_source_manifest, label="aligned-source manifest"
    )
    code_root = _existing_canonical_directory(
        paths.processing_code_root, label="processing code root"
    )
    python_executable = _validated_executable(
        paths.python_executable, label="processing Python executable"
    )

    read_revision = _git_revision if revision_reader is None else revision_reader
    revision = read_revision(code_root)
    _require(revision == PROCESSING_CODE_REVISION, "processing code revision changed")

    provide_inventory = (
        _default_inventory_provider
        if inventory_provider is None
        else inventory_provider
    )
    selected = derive_selected_inventory(
        provide_inventory(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            revision=HF_DATASET_REVISION,
            path_in_repo=REMOTE_OBJECT_ROOT,
            recursive=True,
            expand=True,
        )
    )
    inventory_artifact = _manifest_artifact(
        kind=REMOTE_INVENTORY_MANIFEST_KIND,
        fields={
            "case_name": REPLACEMENT_CASE_NAME,
            "object_id": REPLACEMENT_OBJECT_ID,
            "episode_id": REPLACEMENT_EPISODE_LABEL,
            "repository": {
                "repo_id": HF_REPO_ID,
                "repo_type": HF_REPO_TYPE,
                "revision": HF_DATASET_REVISION,
                "remote_object_root": REMOTE_OBJECT_ROOT,
            },
            "selection": {
                "rule": "fourth-lexicographically-sorted-exact-stem-mp4-txt-pair-per-camera",
                "zero_based_pair_index": REPLACEMENT_EPISODE_ID,
                "camera_count": len(selected.camera_names),
                "pairs_per_camera": EXPECTED_PAIR_COUNT_PER_CAMERA,
                "selected_record_count": len(selected.records),
                "selected_total_size_bytes": selected.total_size_bytes,
                "camera_names": list(selected.camera_names),
                "selected_stems": dict(selected.selected_stems),
                "tactile_audio_or_other_episode_selected": False,
            },
            "allow_patterns": list(selected.allow_patterns),
            "records": list(selected.records),
            "records_sha256": selected.records_sha256,
            "source_permit": permit_json,
        },
    )
    _write_sealed_json(inventory_manifest_path, inventory_artifact)

    download = (
        _default_snapshot_downloader
        if snapshot_downloader is None
        else snapshot_downloader
    )
    returned_root = Path(
        download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            revision=HF_DATASET_REVISION,
            local_dir=str(download_root),
            allow_patterns=list(selected.allow_patterns),
        )
    )
    _require(
        returned_root.resolve(strict=True) == download_root,
        "snapshot downloader returned a different local root",
    )
    content_records = _scan_exact_download(download_root, selected)
    raw_object = download_root / REMOTE_OBJECT_ROOT
    _validate_public_metadata(raw_object)

    content_artifact = _manifest_artifact(
        kind=DOWNLOADED_CONTENT_MANIFEST_KIND,
        fields={
            "case_name": REPLACEMENT_CASE_NAME,
            "object_id": REPLACEMENT_OBJECT_ID,
            "episode_id": REPLACEMENT_EPISODE_LABEL,
            "repository": inventory_artifact["repository"],
            "download_root": str(download_root),
            "raw_object_dir": str(raw_object),
            "allow_patterns": list(selected.allow_patterns),
            "record_count": len(content_records),
            "total_size_bytes": sum(
                int(record["size_bytes"]) for record in content_records
            ),
            "records": list(content_records),
            "records_sha256": _canonical_sha256(content_records),
            "remote_inventory_manifest": _file_record(
                inventory_manifest_path,
                label="sealed inventory manifest",
                exact_mode=0o400,
            ),
            "source_permit": permit_json,
        },
    )
    _write_sealed_json(content_manifest_path, content_artifact)

    aligned_root.mkdir(mode=0o755)
    aligned_object_root = aligned_root / REPLACEMENT_OBJECT_ID
    aligned_object_root.mkdir(mode=0o755)
    local_episode = aligned_object_root / f"episode_{LOCAL_SUBSET_EPISODE_INDEX:04d}"
    canonical_episode = aligned_object_root / f"episode_{REPLACEMENT_EPISODE_ID:04d}"
    _require(
        not os.path.lexists(local_episode) and not os.path.lexists(canonical_episode),
        "aligned episode destination is not fresh",
    )

    run_command = _default_command_runner if command_runner is None else command_runner
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(code_root)
    environment["PYTHONNOUSERSITE"] = "1"
    run_command(
        (
            str(python_executable),
            "-c",
            _UNDISTORT_SCRIPT,
            str(raw_object),
            str(aligned_object_root),
            json.dumps(list(selected.camera_names), separators=(",", ":")),
        ),
        cwd=code_root,
        env=environment,
    )
    _existing_canonical_directory(local_episode, label="local aligned episode zero")
    _require(not os.path.lexists(canonical_episode), "canonical episode already exists")
    local_episode.rename(canonical_episode)

    expected_aligned_cameras = tuple(
        camera for camera in selected.camera_names if camera not in UNCALIBRATED_CAMERAS
    )
    _require(
        len(expected_aligned_cameras) == EXPECTED_ALIGNED_CAMERA_COUNT,
        "frozen calibrated-camera census changed",
    )
    run_command(
        (
            str(python_executable),
            "-c",
            _ROBOT_SCRIPT,
            str(aligned_object_root),
            json.dumps(list(expected_aligned_cameras), separators=(",", ":")),
        ),
        cwd=code_root,
        env=environment,
    )
    _require(
        not os.path.lexists(local_episode), "local episode zero was not relabelled"
    )
    _require(
        frozenset(entry.name for entry in aligned_object_root.iterdir())
        == {canonical_episode.name},
        "aligned object root contains an unexpected episode",
    )

    robot_meta, aligned_records, frame_count = _validate_episode_layout(
        canonical_episode, expected_cameras=expected_aligned_cameras
    )
    state = load_robot_kinematics_archive(
        canonical_episode / "robot" / "robot.npz",
        expected_frame_count=frame_count,
    )
    aligned_artifact = _manifest_artifact(
        kind=ALIGNED_SOURCE_MANIFEST_KIND,
        fields={
            "case_name": REPLACEMENT_CASE_NAME,
            "object_id": REPLACEMENT_OBJECT_ID,
            "episode_id": REPLACEMENT_EPISODE_LABEL,
            "semantics": SEMANTICS,
            "public_metadata_evidence": PUBLIC_METADATA_EVIDENCE,
            "source_permit": permit_json,
            "repository": inventory_artifact["repository"],
            "processing": {
                "code_root": str(code_root),
                "code_revision": revision,
                "python_executable": str(python_executable),
                "source_episode_id": REPLACEMENT_EPISODE_ID,
                "isolated_subset_local_episode_index": LOCAL_SUBSET_EPISODE_INDEX,
                "canonical_episode_id": REPLACEMENT_EPISODE_ID,
                "embedded_alignment_episode_index": LOCAL_SUBSET_EPISODE_INDEX,
                "undistortion_tolerance_us": 100_000,
                "robot_bimanual": False,
                "robot_seed": 0,
            },
            "download_root": str(download_root),
            "raw_object_dir": str(raw_object),
            "aligned_root": str(aligned_root),
            "aligned_episode_dir": str(canonical_episode),
            "camera_census": {
                "downloaded_camera_count": len(selected.camera_names),
                "downloaded_cameras": list(selected.camera_names),
                "uncalibrated_skipped_camera_count": len(UNCALIBRATED_CAMERAS),
                "uncalibrated_skipped_cameras": list(UNCALIBRATED_CAMERAS),
                "aligned_camera_count": len(expected_aligned_cameras),
                "aligned_cameras": list(expected_aligned_cameras),
            },
            "frame_count": frame_count,
            "robot_arrays": robot_kinematics_array_records(state),
            "robot_metadata_artifact_sha256": robot_meta.get("artifact_sha256"),
            "aligned_record_count": len(aligned_records),
            "aligned_records": list(aligned_records),
            "aligned_records_sha256": _canonical_sha256(aligned_records),
            "remote_inventory_manifest": _file_record(
                inventory_manifest_path,
                label="sealed inventory manifest",
                exact_mode=0o400,
            ),
            "downloaded_content_manifest": _file_record(
                content_manifest_path,
                label="sealed content manifest",
                exact_mode=0o400,
            ),
            "no_tactile_audio_or_other_episode_downloaded": True,
            "no_prior_072_aligned_source_reused": True,
        },
    )
    _write_sealed_json(aligned_manifest_path, aligned_artifact)
    validate_aligned_source_manifest(
        aligned_manifest_path,
        expected_source_permit=permit_json,
    )
    return aligned_manifest_path


def _validate_record_list(
    value: object,
    *,
    root: Path,
    expected_paths: Sequence[str],
    label: str,
) -> tuple[dict[str, Any], ...]:
    _require(isinstance(value, list), f"{label} records are invalid")
    records = tuple(value)
    _require(
        all(isinstance(record, dict) for record in records),
        f"{label} record is invalid",
    )
    _require(
        tuple(record.get("path") for record in records) == tuple(expected_paths),
        f"{label} record paths changed",
    )
    observed = tuple(
        _file_record(
            root / path,
            label=f"{label} {path}",
            record_path=path,
        )
        for path in expected_paths
    )
    _require(observed == records, f"{label} bytes changed")
    return records


def validate_aligned_source_manifest(
    path: str | os.PathLike[str],
    *,
    expected_source_permit: Mapping[str, Any] | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Revalidate a sealed replacement-source manifest and every bound byte."""

    manifest_path = _absolute_lexical(path, label="aligned-source manifest")
    if expected_manifest_sha256 is not None:
        _require(
            _SHA256_RE.fullmatch(expected_manifest_sha256) is not None,
            "expected aligned-source manifest SHA-256 is invalid",
        )
        observed_manifest = _file_record(
            manifest_path,
            label="aligned-source manifest",
            exact_mode=0o400,
        )
        _require(
            observed_manifest["sha256"] == expected_manifest_sha256,
            "aligned-source manifest lock binding changed",
        )
    artifact = _load_json(
        manifest_path, label="aligned-source manifest", exact_mode=0o400
    )
    _validate_artifact_common(
        artifact, kind=ALIGNED_SOURCE_MANIFEST_KIND, label="aligned-source manifest"
    )
    _require(artifact.get("case_name") == REPLACEMENT_CASE_NAME, "case name changed")
    _require(artifact.get("object_id") == REPLACEMENT_OBJECT_ID, "object id changed")
    _require(
        artifact.get("episode_id") == REPLACEMENT_EPISODE_LABEL,
        "episode id changed",
    )
    _require(artifact.get("semantics") == SEMANTICS, "replacement semantics changed")
    _require(
        artifact.get("public_metadata_evidence") == PUBLIC_METADATA_EVIDENCE,
        "public metadata evidence changed",
    )
    permit = artifact.get("source_permit")
    _require(isinstance(permit, dict), "source permit binding is invalid")
    if expected_source_permit is not None:
        _require(
            permit
            == _json_value(
                dict(expected_source_permit), label="expected source permit"
            ),
            "source permit lock binding changed",
        )

    repository = artifact.get("repository")
    _require(
        repository
        == {
            "repo_id": HF_REPO_ID,
            "repo_type": HF_REPO_TYPE,
            "revision": HF_DATASET_REVISION,
            "remote_object_root": REMOTE_OBJECT_ROOT,
        },
        "repository binding changed",
    )
    processing = artifact.get("processing")
    _require(isinstance(processing, dict), "processing binding is invalid")
    _require(
        processing.get("code_revision") == PROCESSING_CODE_REVISION
        and processing.get("source_episode_id") == REPLACEMENT_EPISODE_ID
        and processing.get("isolated_subset_local_episode_index") == 0
        and processing.get("canonical_episode_id") == REPLACEMENT_EPISODE_ID
        and processing.get("embedded_alignment_episode_index") == 0
        and processing.get("undistortion_tolerance_us") == 100_000
        and processing.get("robot_bimanual") is False
        and processing.get("robot_seed") == 0,
        "processing contract changed",
    )

    download_root = _existing_canonical_directory(
        artifact.get("download_root"), label="manifest download root"
    )
    raw_object = _existing_canonical_directory(
        artifact.get("raw_object_dir"), label="manifest raw object"
    )
    _require(
        raw_object == download_root / REMOTE_OBJECT_ROOT,
        "manifest raw object path changed",
    )
    aligned_root = _existing_canonical_directory(
        artifact.get("aligned_root"), label="manifest aligned root"
    )
    episode_dir = _existing_canonical_directory(
        artifact.get("aligned_episode_dir"), label="manifest aligned episode"
    )
    _require(
        episode_dir
        == aligned_root
        / REPLACEMENT_OBJECT_ID
        / f"episode_{REPLACEMENT_EPISODE_ID:04d}",
        "manifest aligned episode path changed",
    )
    aligned_object = episode_dir.parent
    _require(
        frozenset(entry.name for entry in aligned_root.iterdir())
        == {REPLACEMENT_OBJECT_ID}
        and all(
            entry.is_dir() and not entry.is_symlink()
            for entry in aligned_root.iterdir()
        ),
        "aligned root contains an unexpected object",
    )
    _require(
        frozenset(entry.name for entry in aligned_object.iterdir())
        == {episode_dir.name}
        and all(
            entry.is_dir() and not entry.is_symlink()
            for entry in aligned_object.iterdir()
        ),
        "aligned object contains an unexpected episode",
    )

    inventory_bound = artifact.get("remote_inventory_manifest")
    content_bound = artifact.get("downloaded_content_manifest")
    _require(isinstance(inventory_bound, dict), "inventory manifest binding is invalid")
    _require(isinstance(content_bound, dict), "content manifest binding is invalid")
    inventory_path = inventory_bound.get("path")
    content_path = content_bound.get("path")
    observed_inventory_bound = _file_record(
        inventory_path, label="sealed inventory manifest", exact_mode=0o400
    )
    observed_content_bound = _file_record(
        content_path, label="sealed content manifest", exact_mode=0o400
    )
    _require(
        observed_inventory_bound == inventory_bound,
        "sealed inventory manifest bytes changed",
    )
    _require(
        observed_content_bound == content_bound,
        "sealed content manifest bytes changed",
    )
    inventory_artifact = _load_json(
        inventory_path, label="sealed inventory manifest", exact_mode=0o400
    )
    content_artifact = _load_json(
        content_path, label="sealed content manifest", exact_mode=0o400
    )
    _validate_artifact_common(
        inventory_artifact,
        kind=REMOTE_INVENTORY_MANIFEST_KIND,
        label="inventory manifest",
    )
    _validate_artifact_common(
        content_artifact,
        kind=DOWNLOADED_CONTENT_MANIFEST_KIND,
        label="content manifest",
    )
    _require(
        inventory_artifact.get("source_permit") == permit
        and content_artifact.get("source_permit") == permit,
        "source permit differs across manifests",
    )
    _require(
        inventory_artifact.get("case_name") == REPLACEMENT_CASE_NAME
        and inventory_artifact.get("object_id") == REPLACEMENT_OBJECT_ID
        and inventory_artifact.get("episode_id") == REPLACEMENT_EPISODE_LABEL
        and content_artifact.get("case_name") == REPLACEMENT_CASE_NAME
        and content_artifact.get("object_id") == REPLACEMENT_OBJECT_ID
        and content_artifact.get("episode_id") == REPLACEMENT_EPISODE_LABEL,
        "source identity differs across manifests",
    )
    _require(
        inventory_artifact.get("repository") == repository
        and content_artifact.get("repository") == repository,
        "repository differs across manifests",
    )
    allow_patterns = inventory_artifact.get("allow_patterns")
    inventory_records = inventory_artifact.get("records")
    _require(
        isinstance(allow_patterns, list)
        and len(allow_patterns) == EXPECTED_SELECTED_RECORD_COUNT
        and allow_patterns == sorted(allow_patterns),
        "inventory allowlist changed",
    )
    _require(
        isinstance(inventory_records, list)
        and all(
            isinstance(record, dict)
            and set(record) == {"path", "size_bytes", "blob_id", "lfs_sha256"}
            for record in inventory_records
        )
        and allow_patterns == [record.get("path") for record in inventory_records]
        and inventory_artifact.get("records_sha256") == INVENTORY_RECORDS_SHA256
        and _canonical_sha256(inventory_records) == INVENTORY_RECORDS_SHA256,
        "inventory record digest changed",
    )
    selection = inventory_artifact.get("selection")
    _require(isinstance(selection, dict), "inventory selection is invalid")
    selected_stems = selection.get("selected_stems")
    downloaded_camera_names = selection.get("camera_names")
    _require(
        selection.get("zero_based_pair_index") == REPLACEMENT_EPISODE_ID
        and selection.get("camera_count") == EXPECTED_CAMERA_COUNT
        and selection.get("pairs_per_camera") == EXPECTED_PAIR_COUNT_PER_CAMERA
        and selection.get("selected_record_count") == EXPECTED_SELECTED_RECORD_COUNT
        and selection.get("selected_total_size_bytes") == EXPECTED_DOWNLOAD_SIZE_BYTES
        and selection.get("tactile_audio_or_other_episode_selected") is False
        and isinstance(selected_stems, dict)
        and isinstance(downloaded_camera_names, list)
        and downloaded_camera_names == sorted(downloaded_camera_names)
        and len(downloaded_camera_names) == EXPECTED_CAMERA_COUNT
        and selected_stems.get(REFERENCE_CAMERA) == REFERENCE_SELECTED_STEM,
        "inventory selection contract changed",
    )
    _require(
        content_artifact.get("allow_patterns") == allow_patterns
        and content_artifact.get("download_root") == str(download_root)
        and content_artifact.get("raw_object_dir") == str(raw_object)
        and content_artifact.get("remote_inventory_manifest") == inventory_bound,
        "downloaded content provenance changed",
    )
    selected = SelectedInventory(
        records=tuple(inventory_records),
        allow_patterns=tuple(allow_patterns),
        camera_names=tuple(downloaded_camera_names),
        selected_stems=selected_stems,
        total_size_bytes=EXPECTED_DOWNLOAD_SIZE_BYTES,
        records_sha256=INVENTORY_RECORDS_SHA256,
    )
    stored_content_records = content_artifact.get("records")
    _require(isinstance(stored_content_records, list), "content records are invalid")
    scanned_content_records = _scan_exact_download(download_root, selected)
    _require(
        scanned_content_records == tuple(stored_content_records),
        "downloaded content scan changed",
    )
    content_records = _validate_record_list(
        stored_content_records,
        root=download_root,
        expected_paths=allow_patterns,
        label="downloaded source",
    )
    _require(
        _canonical_sha256(content_records) == DOWNLOADED_CONTENT_RECORDS_SHA256
        and content_artifact.get("records_sha256") == DOWNLOADED_CONTENT_RECORDS_SHA256
        and content_artifact.get("record_count") == EXPECTED_SELECTED_RECORD_COUNT
        and content_artifact.get("total_size_bytes") == EXPECTED_DOWNLOAD_SIZE_BYTES,
        "downloaded content record contract changed",
    )
    _validate_public_metadata(raw_object)

    census = artifact.get("camera_census")
    _require(isinstance(census, dict), "camera census is invalid")
    downloaded_cameras = census.get("downloaded_cameras")
    aligned_cameras = census.get("aligned_cameras")
    _require(
        isinstance(downloaded_cameras, list)
        and len(downloaded_cameras) == EXPECTED_CAMERA_COUNT
        and downloaded_cameras == sorted(downloaded_cameras),
        "downloaded camera census changed",
    )
    expected_aligned = [
        camera for camera in downloaded_cameras if camera not in UNCALIBRATED_CAMERAS
    ]
    _require(
        census
        == {
            "downloaded_camera_count": EXPECTED_CAMERA_COUNT,
            "downloaded_cameras": downloaded_cameras,
            "uncalibrated_skipped_camera_count": len(UNCALIBRATED_CAMERAS),
            "uncalibrated_skipped_cameras": list(UNCALIBRATED_CAMERAS),
            "aligned_camera_count": EXPECTED_ALIGNED_CAMERA_COUNT,
            "aligned_cameras": expected_aligned,
        },
        "camera census contract changed",
    )
    _require(aligned_cameras == expected_aligned, "aligned cameras changed")
    _robot_meta, observed_aligned_records, frame_count = _validate_episode_layout(
        episode_dir, expected_cameras=tuple(expected_aligned)
    )
    _require(artifact.get("frame_count") == frame_count, "frame count changed")
    aligned_paths = [record["path"] for record in observed_aligned_records]
    aligned_records = _validate_record_list(
        artifact.get("aligned_records"),
        root=aligned_root,
        expected_paths=aligned_paths,
        label="aligned source",
    )
    _require(
        aligned_records == observed_aligned_records
        and artifact.get("aligned_record_count") == len(aligned_records)
        and artifact.get("aligned_records_sha256")
        == _canonical_sha256(aligned_records),
        "aligned source record contract changed",
    )
    state = load_robot_kinematics_archive(
        episode_dir / "robot" / "robot.npz", expected_frame_count=frame_count
    )
    _require(
        artifact.get("robot_arrays") == robot_kinematics_array_records(state),
        "robot arrays changed",
    )
    _require(
        artifact.get("no_tactile_audio_or_other_episode_downloaded") is True
        and artifact.get("no_prior_072_aligned_source_reused") is True,
        "replacement source isolation claim changed",
    )
    return artifact


__all__ = [
    "ALIGNED_SOURCE_MANIFEST_KIND",
    "DOWNLOADED_CONTENT_MANIFEST_KIND",
    "DOWNLOADED_CONTENT_RECORDS_SHA256",
    "EXPECTED_DOWNLOAD_SIZE_BYTES",
    "HF_DATASET_REVISION",
    "HF_REPO_ID",
    "INVENTORY_RECORDS_SHA256",
    "PROCESSING_CODE_REVISION",
    "REPLACEMENT_SOURCE_INVENTORY_CONTRACT",
    "REMOTE_INVENTORY_MANIFEST_KIND",
    "REPLACEMENT_CASE_NAME",
    "REPLACEMENT_EPISODE_ID",
    "REPLACEMENT_EPISODE_LABEL",
    "REPLACEMENT_OBJECT_ID",
    "ReplacementSourcePaths",
    "SEMANTICS",
    "SOURCE_OPERATION",
    "UNCALIBRATED_CAMERAS",
    "acquire_and_align_replacement_source",
    "derive_selected_inventory",
    "validate_aligned_source_manifest",
]
