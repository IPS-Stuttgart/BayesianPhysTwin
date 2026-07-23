"""Write-once, pre-barrier source custody for one confirmation case.

The prediction-prefix and frame-zero stages intentionally run before any
future outcome is authorized.  This module closes a remaining provenance gap:
it inventories the complete aligned source episode and the exact staged case
tree, binds both trees to one H2-locked case and their absolute paths, and
records the canonical raw-RGB digest of every staged 58-frame prefix.

The source inventory includes every source-preparation output.  In particular,
it binds the full aligned video and timestamps for every source camera, rather
than relying on the older source-preparation manifest (which did not hash those
two files).  The staged inventory is an exact allowlist: prefix video,
timestamps, mask and depth, the one-frame reconstruction inputs and outputs,
the frame-zero Splat, all three stage manifests, calibration, robot slices,
and the known action.  Symlinks, hard links, special files, and extra entries
fail closed.

No function accepts an outcome, target, score, or future-geometry path.
``replay=True`` never rewrites the seal; it re-hashes and re-decodes the
currently bound inputs and validates the existing write-once artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any

from .deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    validate_confirmation_cohort_lock,
)


SCHEMA_VERSION = 1
ARTIFACT_KIND = "Deform360AdaptiveCovarianceConfirmationSourceCustodyV1"
SOURCE_PREPARATION_FILENAME = "bias_aware_source_preparation_manifest.json"
PREDICTION_PREFIX_MANIFEST_FILENAME = "prediction_prefix_manifest.json"
FRAME_ZERO_MANIFEST_FILENAME = "frame_zero_reconstruction_manifest.json"
FRAME_ZERO_ARCHIVE_FILENAME = "frame_zero_points.npz"
PREFIX_FRAME_COUNT = 58
FRAME_ZERO_FRAME_COUNT = 1
KNOWN_ACTION_FRAME_COUNT = 76
STAGING_FRAME_COUNT = 81

SOURCE_CAMERA_FILES = frozenset(
    {
        "aligned_timestamps.txt",
        "alignment.json",
        "metadata.json",
        "undistorted.mp4",
        "undistorted_000000.png",
    }
)
SOURCE_EPISODE_FILES = frozenset(
    {
        "alignment.json",
        "extrinsics.npy",
        SOURCE_PREPARATION_FILENAME,
        "undistorted_intrinsics.npy",
    }
)
SOURCE_ROBOT_FILES = frozenset({"robot.meta.json", "robot.npz"})
STAGED_PREFIX_CAMERA_FILES = frozenset(
    {
        "aligned_timestamps.txt",
        "mask_refined.h5",
        "rendered_depth.h5",
        "rendered_depth.meta.json",
        "undistorted.mp4",
    }
)
STAGED_FRAME_ZERO_CAMERA_FILES = frozenset(
    {
        "aligned_timestamps.txt",
        "mask_refined.h5",
        "rendered_depth.h5",
        "rendered_depth.meta.json",
        "rendered_urdf.h5",
        "rendered_urdf.meta.json",
        "undistorted.mp4",
    }
)
STAGED_SPLAT_FILES = frozenset({"splat_0.ply", "splatfacto.meta.json"})
STAGED_CASE_ROOT_FILES = frozenset(
    {
        FRAME_ZERO_ARCHIVE_FILENAME,
        FRAME_ZERO_MANIFEST_FILENAME,
        PREDICTION_PREFIX_MANIFEST_FILENAME,
    }
)
RAW_RGB24_PREFIX_ALGORITHM = "ffmpeg-rgb24-first-58-frames-concatenated-byte-sha256"
INFORMATION_BOUNDARY: Mapping[str, Any] = {
    "outcome_or_target_argument_accepted": False,
    "future_object_geometry_read": False,
    "future_object_track_read": False,
    "future_tactile_read": False,
    "metric_or_score_computed": False,
    "target_access_authorization_absent": True,
    "full_aligned_source_videos_hashed_not_decoded": True,
    "staged_prefix_videos_decoded_for_custody_sha256": True,
    "known_future_robot_action_hashed_as_conditioning_input": True,
    "seal_must_exist_before_complete_cohort_prediction_barrier": True,
}
CAMERA_CUSTODY_KEYS = frozenset(
    {
        "source_full_video_file_sha256",
        "source_full_timestamps_file_sha256",
        "source_prefix_frame_range_half_open",
        "staged_prefix_video_file_sha256",
        "staged_prefix_timestamps_file_sha256",
        "staged_prefix_mask_file_sha256",
        "staged_prefix_depth_file_sha256",
        "staged_frame_zero_video_file_sha256",
        "staged_frame_zero_timestamps_file_sha256",
        "staged_frame_zero_mask_file_sha256",
        "staged_frame_zero_depth_file_sha256",
        "staged_frame_zero_gripper_mask_file_sha256",
        "decoded_rgb24_prefix_sha256",
        "timestamp_prefix_exact_source_slice",
        "timestamp_frame_zero_exact_source_slice",
    }
)

_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")
_FULL_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class _FileSnapshot:
    path: Path
    payload: bytes
    size_bytes: int
    sha256: str
    state: tuple[int, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_full_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and _FULL_SHA1.fullmatch(value) is not None
        and value != "0" * 40
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _FULL_SHA256.fullmatch(value) is not None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    """Return the self-digest of a source-custody artifact."""

    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _result_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _file_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _canonical_directory(path: str | Path, *, label: str) -> Path:
    root = Path(path).absolute()
    try:
        observed = os.lstat(root)
    except OSError as error:
        raise ValueError(f"{label} is missing: {root}") from error
    _require(
        stat.S_ISDIR(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and root.resolve(strict=True) == root,
        f"{label} is linked, noncanonical, or not a directory: {root}",
    )
    return root


def _stable_regular_file(
    path: str | Path,
    *,
    label: str,
    capture_payload: bool,
) -> _FileSnapshot:
    source = Path(path).absolute()
    try:
        observed = os.lstat(source)
    except OSError as error:
        raise ValueError(f"{label} is missing: {source}") from error
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1
        and source.resolve(strict=True) == source,
        f"{label} is linked, noncanonical, or not a single-link regular file: {source}",
    )
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        _require(
            _file_state(opened) == _file_state(observed),
            f"{label} changed while opening: {source}",
        )
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            if capture_payload:
                chunks.append(block)
        after = os.fstat(descriptor)
        current = os.lstat(source)
        _require(
            _file_state(after) == _file_state(opened) == _file_state(current),
            f"{label} changed while hashing: {source}",
        )
    finally:
        os.close(descriptor)
    return _FileSnapshot(
        path=source,
        payload=b"".join(chunks),
        size_bytes=opened.st_size,
        sha256=digest.hexdigest(),
        state=_file_state(opened),
    )


def _load_json_snapshot(
    path: str | Path, *, label: str
) -> tuple[_FileSnapshot, dict[str, Any]]:
    snapshot = _stable_regular_file(path, label=label, capture_payload=True)
    try:
        value = json.loads(
            snapshot.payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return snapshot, value


def _inventory_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_bytes({"records": list(records)})).hexdigest()


def _inventory_exact_tree(
    root: Path,
    *,
    expected_directories: set[str],
    expected_files: set[str],
    label: str,
) -> dict[str, Any]:
    """Hash one exact tree without following links or accepting extra entries."""

    _require("" not in expected_directories, "root must not be an inventory child")
    _require(
        not (expected_directories & expected_files),
        f"{label} expected inventory overlaps",
    )
    observed_directories: set[str] = set()
    observed_files: set[str] = set()
    records: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        relative_directory = directory.relative_to(root).as_posix()
        if relative_directory == ".":
            relative_directory = ""
        before = os.lstat(directory)
        _require(
            stat.S_ISDIR(before.st_mode) and not stat.S_ISLNK(before.st_mode),
            f"{label} contains a linked or special directory: {directory}",
        )
        with os.scandir(directory) as entries:
            children = sorted(entries, key=lambda entry: entry.name)
        for entry in children:
            path = Path(entry.path)
            observed = entry.stat(follow_symlinks=False)
            relative = path.relative_to(root).as_posix()
            _require(
                not stat.S_ISLNK(observed.st_mode),
                f"{label} contains a symlink: {relative}",
            )
            if stat.S_ISDIR(observed.st_mode):
                observed_directories.add(relative)
                records.append({"path": relative, "type": "directory"})
                pending.append(path)
                continue
            _require(
                stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1,
                f"{label} contains a special or hard-linked file: {relative}",
            )
            snapshot = _stable_regular_file(
                path,
                label=f"{label} file {relative}",
                capture_payload=False,
            )
            observed_files.add(relative)
            records.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": snapshot.size_bytes,
                    "sha256": snapshot.sha256,
                }
            )
        after = os.lstat(directory)
        _require(
            _file_state(before) == _file_state(after),
            f"{label} directory changed during inventory: {relative_directory or '.'}",
        )
    _require(
        observed_directories == expected_directories,
        f"{label} directory inventory changed; "
        f"missing={sorted(expected_directories - observed_directories)}, "
        f"extra={sorted(observed_directories - expected_directories)}",
    )
    _require(
        observed_files == expected_files,
        f"{label} file inventory changed; "
        f"missing={sorted(expected_files - observed_files)}, "
        f"extra={sorted(observed_files - expected_files)}",
    )
    records.sort(key=lambda record: (str(record["path"]), str(record["type"])))
    return {
        "root": str(root),
        "directory_count": len(observed_directories),
        "regular_file_count": len(observed_files),
        "regular_file_bytes": sum(
            int(record["size_bytes"]) for record in records if record["type"] == "file"
        ),
        "records": records,
        "inventory_sha256": _inventory_sha256(records),
        "all_files_single_link_regular": True,
        "all_directories_real": True,
        "symlinks_special_files_and_extras_accepted": False,
    }


def _inventory_files(inventory: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(record["path"]): record
        for record in inventory["records"]
        if record["type"] == "file"
    }


def _validate_inventory_envelope(
    value: object,
    *,
    expected_root: Path,
    expected_directories: set[str],
    expected_files: set[str],
    label: str,
) -> dict[str, Mapping[str, Any]]:
    """Validate inventory metadata without opening any inventoried path."""

    _require(
        isinstance(value, Mapping)
        and set(value)
        == {
            "root",
            "directory_count",
            "regular_file_count",
            "regular_file_bytes",
            "records",
            "inventory_sha256",
            "all_files_single_link_regular",
            "all_directories_real",
            "symlinks_special_files_and_extras_accepted",
        }
        and value.get("root") == str(expected_root)
        and value.get("all_files_single_link_regular") is True
        and value.get("all_directories_real") is True
        and value.get("symlinks_special_files_and_extras_accepted") is False,
        f"{label} inventory envelope changed",
    )
    records = value.get("records")
    _require(isinstance(records, list), f"{label} records are absent")
    normalized: list[dict[str, Any]] = []
    directories: set[str] = set()
    files: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _require(
            isinstance(record, Mapping)
            and isinstance(record.get("path"), str)
            and bool(record["path"])
            and not Path(record["path"]).is_absolute()
            and ".." not in Path(record["path"]).parts,
            f"{label} has an invalid relative inventory path",
        )
        path = str(record["path"])
        if record.get("type") == "directory":
            _require(
                set(record) == {"path", "type"} and path not in directories,
                f"{label} has an invalid or duplicate directory record",
            )
            directories.add(path)
        else:
            _require(
                record.get("type") == "file"
                and set(record) == {"path", "type", "size_bytes", "sha256"}
                and type(record.get("size_bytes")) is int
                and record["size_bytes"] >= 0
                and _is_sha256(record.get("sha256"))
                and path not in files,
                f"{label} has an invalid or duplicate file record",
            )
            files[path] = record
        normalized.append(dict(record))
    _require(
        normalized
        == sorted(
            normalized,
            key=lambda record: (str(record["path"]), str(record["type"])),
        )
        and not (directories & set(files))
        and directories == expected_directories
        and set(files) == expected_files
        and value.get("directory_count") == len(directories)
        and value.get("regular_file_count") == len(files)
        and value.get("regular_file_bytes")
        == sum(int(record["size_bytes"]) for record in files.values())
        and value.get("inventory_sha256") == _inventory_sha256(normalized),
        f"{label} exact inventory envelope changed",
    )
    return files


def _external_case_identity(
    lock: Mapping[str, Any],
    case_id: str,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for stratum, records in lock["cohort"].items():
        for record in records:
            object_id = record["object_id"]
            for episode in record["episodes"]:
                if episode["case_id"] == case_id:
                    episode_id = int(episode["episode_id"])
                    matches.append(
                        {
                            "case": case_id,
                            "object_id": object_id,
                            "episode_id": episode_id,
                            "episode_key": f"{object_id}/{episode_id}",
                            "stratum": stratum,
                            "role": "calibration",
                        }
                    )
    _require(len(matches) == 1, "case is outside the exact H2-locked cohort")
    return matches[0]


def _validate_stage_manifest(
    manifest: Mapping[str, Any],
    *,
    kind: str,
    lock: Mapping[str, Any],
    identity: Mapping[str, Any],
    label: str,
) -> None:
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("artifact_kind") == kind
        and manifest.get("protocol_id") == PROTOCOL_ID
        and manifest.get("protocol_config_sha256") == lock["artifact_sha256"]
        and manifest.get("result_sha256") == _result_sha256(manifest)
        and all(manifest.get(key) == value for key, value in identity.items()),
        f"{label} binds another protocol, lock, or case",
    )


def _source_expected_paths(cameras: Sequence[str]) -> tuple[set[str], set[str]]:
    directories = {"robot", *cameras}
    files = set(SOURCE_EPISODE_FILES)
    files.update(f"robot/{name}" for name in SOURCE_ROBOT_FILES)
    for camera in cameras:
        files.update(f"{camera}/{name}" for name in SOURCE_CAMERA_FILES)
    return directories, files


def _staged_expected_paths(cameras: Sequence[str]) -> tuple[set[str], set[str]]:
    prefix = "prefix/episode_0000"
    frame_zero = "frame-zero/episode_0000"
    directories = {
        "prefix",
        prefix,
        f"{prefix}/robot",
        "frame-zero",
        frame_zero,
        f"{frame_zero}/robot",
        f"{frame_zero}/splatfacto",
        "known-action",
    }
    directories.update(f"{prefix}/{camera}" for camera in cameras)
    directories.update(f"{frame_zero}/{camera}" for camera in cameras)
    files = set(STAGED_CASE_ROOT_FILES)
    files.update(
        {
            f"{prefix}/undistorted_intrinsics.npy",
            f"{prefix}/extrinsics.npy",
            f"{prefix}/robot/robot.npz",
            f"{frame_zero}/undistorted_intrinsics.npy",
            f"{frame_zero}/extrinsics.npy",
            f"{frame_zero}/robot/robot.npz",
            "known-action/robot.npz",
        }
    )
    files.update(f"{frame_zero}/splatfacto/{name}" for name in STAGED_SPLAT_FILES)
    for camera in cameras:
        files.update(f"{prefix}/{camera}/{name}" for name in STAGED_PREFIX_CAMERA_FILES)
        files.update(
            f"{frame_zero}/{camera}/{name}" for name in STAGED_FRAME_ZERO_CAMERA_FILES
        )
    return directories, files


def _read_text_lines(path: Path, *, label: str) -> tuple[_FileSnapshot, list[str]]:
    snapshot = _stable_regular_file(path, label=label, capture_payload=True)
    try:
        value = snapshot.payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    _require(
        value.endswith("\n"),
        f"{label} lacks a terminating newline",
    )
    return snapshot, value.splitlines()


def _decoded_rgb24_prefix_sha256(
    video_path: Path,
    *,
    frame_count: int = PREFIX_FRAME_COUNT,
) -> str:
    """Mirror the frozen authorized-future raw-RGB prefix digest exactly."""

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _require(process.stdout is not None, "ffmpeg output pipe is unavailable")
    digest = hashlib.sha256()
    decoded_bytes = 0
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
        decoded_bytes += len(chunk)
    stderr = b"" if process.stderr is None else process.stderr.read()
    return_code = process.wait()
    _require(
        return_code == 0 and decoded_bytes > 0,
        "cannot decode staged RGB prefix with ffmpeg: "
        f"{video_path}: {stderr.decode('utf-8', errors='replace').strip()}",
    )
    return digest.hexdigest()


def _stable_decoded_prefix_sha256(video_path: Path) -> str:
    before = os.lstat(video_path)
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1,
        f"staged prefix video is linked or special: {video_path}",
    )
    digest = _decoded_rgb24_prefix_sha256(video_path)
    after = os.lstat(video_path)
    _require(
        _file_state(before) == _file_state(after),
        f"staged prefix video changed while decoding: {video_path}",
    )
    _require(_is_sha256(digest), "decoded staged-prefix digest is invalid")
    return digest


def _sha_at(files: Mapping[str, Mapping[str, Any]], path: str) -> str:
    record = files.get(path)
    _require(record is not None, f"inventory omitted required file: {path}")
    digest = record.get("sha256")
    _require(_is_sha256(digest), f"inventory digest is invalid: {path}")
    return str(digest)


def _manifest_record(
    snapshot: _FileSnapshot,
    manifest: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    return {
        "relative_path": snapshot.path.relative_to(root).as_posix(),
        "file_sha256": snapshot.sha256,
        "result_sha256": manifest["result_sha256"],
    }


def _capture_payload(
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    source_episode_dir: str | Path,
    staged_case_dir: str | Path,
    seal_path: str | Path,
    *,
    expected_h1: str | None,
) -> dict[str, Any]:
    _require(_is_full_sha1(h2_commit), "H2 must be a full lowercase nonzero SHA-1")
    _require(isinstance(case_id, str) and bool(case_id), "case id is empty")
    source = _canonical_directory(source_episode_dir, label="source episode")
    staged = _canonical_directory(staged_case_dir, label="staged case")
    _require(
        source != staged
        and source not in staged.parents
        and staged not in source.parents,
        "source episode and staged case paths overlap",
    )
    lock_snapshot, lock = _load_json_snapshot(lock_path, label="H2 cohort lock")
    validate_confirmation_cohort_lock(
        lock,
        expected_implementation_commit_h1=expected_h1,
    )
    identity = _external_case_identity(lock, case_id)
    h1_commit = lock["two_commit_freeze"]["implementation_commit_h1"]
    _require(
        staged.name == case_id
        and source.name == f"episode_{identity['episode_id']:04d}"
        and source.parent.name == identity["object_id"],
        "source or staged path binds another H2 case",
    )

    source_manifest_snapshot, source_manifest = _load_json_snapshot(
        source / SOURCE_PREPARATION_FILENAME,
        label="source-preparation manifest",
    )
    prefix_manifest_snapshot, prefix_manifest = _load_json_snapshot(
        staged / PREDICTION_PREFIX_MANIFEST_FILENAME,
        label="prediction-prefix manifest",
    )
    frame_manifest_snapshot, frame_manifest = _load_json_snapshot(
        staged / FRAME_ZERO_MANIFEST_FILENAME,
        label="frame-zero manifest",
    )
    _validate_stage_manifest(
        source_manifest,
        kind="Deform360BiasAwareSourcePreparation",
        lock=lock,
        identity=identity,
        label="source-preparation manifest",
    )
    _validate_stage_manifest(
        prefix_manifest,
        kind="Deform360BiasAwarePredictionPrefix",
        lock=lock,
        identity=identity,
        label="prediction-prefix manifest",
    )
    _validate_stage_manifest(
        frame_manifest,
        kind="Deform360BiasAwareFrameZeroReconstruction",
        lock=lock,
        identity=identity,
        label="frame-zero manifest",
    )
    _require(
        source_manifest.get("target_access_authorization") is None
        and prefix_manifest.get("target_access_authorization") is None,
        "target access was authorized during target-free source custody",
    )

    source_cameras = source_manifest.get("cameras")
    _require(
        isinstance(source_cameras, list)
        and len(source_cameras) >= 8
        and source_cameras == sorted(source_cameras)
        and len(source_cameras) == len(set(source_cameras))
        and all(isinstance(camera, str) and bool(camera) for camera in source_cameras)
        and source_manifest.get("camera_count") == len(source_cameras),
        "source-preparation camera panel changed",
    )
    camera_records = prefix_manifest.get("camera_records")
    _require(
        isinstance(camera_records, list)
        and len(camera_records) >= 8
        and prefix_manifest.get("camera_count") == len(camera_records),
        "prediction-prefix camera panel changed",
    )
    prefix_by_camera: dict[str, Mapping[str, Any]] = {}
    for record in camera_records:
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {
                "camera",
                "prefix_video_sha256",
                "frame_zero_video_sha256",
                "frame_zero_mask_sha256",
            }
            and isinstance(record.get("camera"), str)
            and bool(record["camera"])
            and record["camera"] not in prefix_by_camera
            and all(
                _is_sha256(record.get(key))
                for key in (
                    "prefix_video_sha256",
                    "frame_zero_video_sha256",
                    "frame_zero_mask_sha256",
                )
            ),
            "prediction-prefix camera record changed",
        )
        prefix_by_camera[str(record["camera"])] = record
    cameras = tuple(sorted(prefix_by_camera))
    _require(
        tuple(prefix_by_camera) == cameras
        and set(cameras) <= set(source_cameras)
        and frame_manifest.get("cameras") == list(cameras)
        and frame_manifest.get("camera_count") == len(cameras),
        "source, prefix, and frame-zero camera panels differ",
    )
    _require(
        prefix_manifest.get("staged_prefix_frame_count") == PREFIX_FRAME_COUNT
        and prefix_manifest.get("staged_frame_zero_frame_count")
        == FRAME_ZERO_FRAME_COUNT
        and prefix_manifest.get("known_action_frame_count") == KNOWN_ACTION_FRAME_COUNT,
        "staged temporal contract changed",
    )
    action_window = prefix_manifest.get("action_window")
    _require(isinstance(action_window, Mapping), "action window is absent")
    prefix_range = action_window.get("prefix_raw_frame_range_half_open")
    prediction_range = action_window.get("prediction_raw_frame_range_half_open")
    staging_range = action_window.get("selected_raw_frame_range_half_open")
    _require(
        isinstance(prefix_range, list)
        and len(prefix_range) == 2
        and all(type(value) is int for value in prefix_range)
        and prefix_range[1] - prefix_range[0] == PREFIX_FRAME_COUNT
        and prediction_range
        == [prefix_range[0], prefix_range[0] + KNOWN_ACTION_FRAME_COUNT]
        and staging_range == [prefix_range[0], prefix_range[0] + STAGING_FRAME_COUNT],
        "action-window frame ranges changed",
    )
    source_start = int(prefix_range[0])

    source_directories, source_files_expected = _source_expected_paths(source_cameras)
    staged_directories, staged_files_expected = _staged_expected_paths(cameras)
    source_inventory = _inventory_exact_tree(
        source,
        expected_directories=source_directories,
        expected_files=source_files_expected,
        label="aligned source episode",
    )
    staged_inventory = _inventory_exact_tree(
        staged,
        expected_directories=staged_directories,
        expected_files=staged_files_expected,
        label="staged prediction case",
    )
    source_files = _inventory_files(source_inventory)
    staged_files = _inventory_files(staged_inventory)

    _require(
        source_manifest_snapshot.sha256
        == _sha_at(source_files, SOURCE_PREPARATION_FILENAME)
        and prefix_manifest_snapshot.sha256
        == _sha_at(staged_files, PREDICTION_PREFIX_MANIFEST_FILENAME)
        and frame_manifest_snapshot.sha256
        == _sha_at(staged_files, FRAME_ZERO_MANIFEST_FILENAME),
        "stage manifest changed during custody inventory",
    )
    source_inputs = source_manifest.get("inputs_sha256")
    prefix_inputs = prefix_manifest.get("inputs_sha256")
    frame_inputs = frame_manifest.get("inputs_sha256")
    _require(
        isinstance(source_inputs, Mapping)
        and source_inputs.get("protocol") == lock_snapshot.sha256
        and source_inputs.get("calibration_gate") is None
        and isinstance(prefix_inputs, Mapping)
        and prefix_inputs.get("protocol") == lock_snapshot.sha256
        and prefix_inputs.get("source_preparation_manifest")
        == source_manifest_snapshot.sha256
        and prefix_inputs.get("calibration_gate") is None
        and isinstance(frame_inputs, Mapping)
        and frame_inputs.get("prediction_prefix_manifest")
        == prefix_manifest_snapshot.sha256,
        "source-stage manifest lineage changed",
    )
    source_outputs = source_manifest.get("outputs_sha256")
    _require(
        isinstance(source_outputs, Mapping)
        and source_outputs.get("alignment") == _sha_at(source_files, "alignment.json")
        and source_outputs.get("undistorted_intrinsics")
        == _sha_at(source_files, "undistorted_intrinsics.npy")
        and source_outputs.get("extrinsics") == _sha_at(source_files, "extrinsics.npy")
        and source_outputs.get("robot") == _sha_at(source_files, "robot/robot.npz")
        and source_outputs.get("robot_metadata")
        == _sha_at(source_files, "robot/robot.meta.json")
        and isinstance(source_outputs.get("camera_metadata"), Mapping)
        and set(source_outputs["camera_metadata"]) == set(source_cameras)
        and all(
            source_outputs["camera_metadata"][camera]
            == _sha_at(source_files, f"{camera}/metadata.json")
            for camera in source_cameras
        ),
        "source-preparation output hashes changed",
    )
    _require(
        prefix_inputs.get("source_robot") == _sha_at(source_files, "robot/robot.npz")
        and prefix_inputs.get("source_intrinsics")
        == _sha_at(source_files, "undistorted_intrinsics.npy")
        and prefix_inputs.get("source_extrinsics")
        == _sha_at(source_files, "extrinsics.npy"),
        "prediction prefix binds another source robot or calibration",
    )

    prefix_episode = "prefix/episode_0000"
    frame_zero_episode = "frame-zero/episode_0000"
    staged_robots = prefix_manifest.get("staged_robot_sha256")
    _require(
        isinstance(staged_robots, Mapping)
        and staged_robots.get("prefix")
        == _sha_at(staged_files, f"{prefix_episode}/robot/robot.npz")
        and staged_robots.get("frame_zero")
        == _sha_at(staged_files, f"{frame_zero_episode}/robot/robot.npz")
        and staged_robots.get("known_action")
        == _sha_at(staged_files, "known-action/robot.npz"),
        "staged robot or known-action hash changed",
    )
    _require(
        _sha_at(staged_files, f"{prefix_episode}/undistorted_intrinsics.npy")
        == _sha_at(staged_files, f"{frame_zero_episode}/undistorted_intrinsics.npy")
        and _sha_at(staged_files, f"{prefix_episode}/extrinsics.npy")
        == _sha_at(staged_files, f"{frame_zero_episode}/extrinsics.npy"),
        "prefix and frame-zero calibration differ",
    )

    frame_outputs = frame_manifest.get("outputs_sha256")
    depth_by_camera = (
        frame_outputs.get("depth_by_camera")
        if isinstance(frame_outputs, Mapping)
        else None
    )
    gripper_by_camera = (
        frame_outputs.get("gripper_mask_by_camera")
        if isinstance(frame_outputs, Mapping)
        else None
    )
    _require(
        isinstance(frame_outputs, Mapping)
        and frame_outputs.get("frame_zero_splat")
        == _sha_at(staged_files, f"{frame_zero_episode}/splatfacto/splat_0.ply")
        and frame_outputs.get("frame_zero_points")
        == _sha_at(staged_files, FRAME_ZERO_ARCHIVE_FILENAME)
        and isinstance(depth_by_camera, Mapping)
        and set(depth_by_camera) == set(cameras)
        and isinstance(gripper_by_camera, Mapping)
        and set(gripper_by_camera) == set(cameras),
        "frame-zero output panel changed",
    )

    camera_custody: dict[str, dict[str, Any]] = {}
    for camera in cameras:
        source_video = f"{camera}/undistorted.mp4"
        source_timestamps = f"{camera}/aligned_timestamps.txt"
        prefix_video = f"{prefix_episode}/{camera}/undistorted.mp4"
        prefix_timestamps = f"{prefix_episode}/{camera}/aligned_timestamps.txt"
        prefix_mask = f"{prefix_episode}/{camera}/mask_refined.h5"
        prefix_depth = f"{prefix_episode}/{camera}/rendered_depth.h5"
        frame_video = f"{frame_zero_episode}/{camera}/undistorted.mp4"
        frame_timestamps = f"{frame_zero_episode}/{camera}/aligned_timestamps.txt"
        frame_mask = f"{frame_zero_episode}/{camera}/mask_refined.h5"
        frame_depth = f"{frame_zero_episode}/{camera}/rendered_depth.h5"
        frame_gripper = f"{frame_zero_episode}/{camera}/rendered_urdf.h5"
        record = prefix_by_camera[camera]
        _require(
            record["prefix_video_sha256"] == _sha_at(staged_files, prefix_video)
            and record["frame_zero_video_sha256"] == _sha_at(staged_files, frame_video)
            and record["frame_zero_mask_sha256"] == _sha_at(staged_files, prefix_mask)
            and depth_by_camera[camera]
            == _sha_at(staged_files, prefix_depth)
            == _sha_at(staged_files, frame_depth)
            and gripper_by_camera[camera] == _sha_at(staged_files, frame_gripper),
            f"staged camera hashes changed: {camera}",
        )
        _source_timestamp_snapshot, source_lines = _read_text_lines(
            source / source_timestamps,
            label=f"{camera} full source timestamps",
        )
        prefix_timestamp_snapshot, prefix_lines = _read_text_lines(
            staged / prefix_timestamps,
            label=f"{camera} staged prefix timestamps",
        )
        frame_timestamp_snapshot, frame_lines = _read_text_lines(
            staged / frame_timestamps,
            label=f"{camera} staged frame-zero timestamps",
        )
        _require(
            len(source_lines) >= source_start + STAGING_FRAME_COUNT
            and prefix_lines
            == source_lines[source_start : source_start + PREFIX_FRAME_COUNT]
            and frame_lines == source_lines[source_start : source_start + 1],
            f"staged timestamps do not replay the source window: {camera}",
        )
        _require(
            prefix_timestamp_snapshot.sha256 == _sha_at(staged_files, prefix_timestamps)
            and frame_timestamp_snapshot.sha256
            == _sha_at(staged_files, frame_timestamps),
            f"staged timestamp changed during lineage replay: {camera}",
        )
        decoded_digest = _stable_decoded_prefix_sha256(staged / prefix_video)
        camera_custody[camera] = {
            "source_full_video_file_sha256": _sha_at(source_files, source_video),
            "source_full_timestamps_file_sha256": _sha_at(
                source_files,
                source_timestamps,
            ),
            "source_prefix_frame_range_half_open": list(prefix_range),
            "staged_prefix_video_file_sha256": _sha_at(
                staged_files,
                prefix_video,
            ),
            "staged_prefix_timestamps_file_sha256": _sha_at(
                staged_files,
                prefix_timestamps,
            ),
            "staged_prefix_mask_file_sha256": _sha_at(
                staged_files,
                prefix_mask,
            ),
            "staged_prefix_depth_file_sha256": _sha_at(
                staged_files,
                prefix_depth,
            ),
            "staged_frame_zero_video_file_sha256": _sha_at(
                staged_files,
                frame_video,
            ),
            "staged_frame_zero_timestamps_file_sha256": _sha_at(
                staged_files,
                frame_timestamps,
            ),
            "staged_frame_zero_mask_file_sha256": _sha_at(
                staged_files,
                frame_mask,
            ),
            "staged_frame_zero_depth_file_sha256": _sha_at(
                staged_files,
                frame_depth,
            ),
            "staged_frame_zero_gripper_mask_file_sha256": _sha_at(
                staged_files,
                frame_gripper,
            ),
            "decoded_rgb24_prefix_sha256": decoded_digest,
            "timestamp_prefix_exact_source_slice": True,
            "timestamp_frame_zero_exact_source_slice": True,
        }

    seal = Path(seal_path).absolute()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_identity": dict(identity),
        "lock_binding": {
            "implementation_commit_h1": h1_commit,
            "cohort_lock_commit_h2": h2_commit,
            "cohort_lock_artifact_sha256": lock["artifact_sha256"],
            "cohort_lock_file_sha256": lock_snapshot.sha256,
        },
        "path_binding": {
            "cohort_lock_path": str(lock_snapshot.path),
            "source_episode_dir": str(source),
            "staged_case_dir": str(staged),
            "custody_seal_path": str(seal),
        },
        "manifests": {
            "source_preparation": _manifest_record(
                source_manifest_snapshot,
                source_manifest,
                root=source,
            ),
            "prediction_prefix": _manifest_record(
                prefix_manifest_snapshot,
                prefix_manifest,
                root=staged,
            ),
            "frame_zero": _manifest_record(
                frame_manifest_snapshot,
                frame_manifest,
                root=staged,
            ),
        },
        "source_camera_panel": list(source_cameras),
        "camera_panel": list(cameras),
        "camera_count": len(cameras),
        "raw_rgb24_prefix": {
            "algorithm": RAW_RGB24_PREFIX_ALGORITHM,
            "frame_count": PREFIX_FRAME_COUNT,
            "by_camera": {
                camera: camera_custody[camera]["decoded_rgb24_prefix_sha256"]
                for camera in cameras
            },
            "direct_source_vs_reencoded_prefix_equality_required": False,
            "authorized_future_must_reuse_this_digest": True,
        },
        "camera_custody": camera_custody,
        "known_action": {
            "relative_path": "known-action/robot.npz",
            "frame_count": KNOWN_ACTION_FRAME_COUNT,
            "file_sha256": _sha_at(staged_files, "known-action/robot.npz"),
            "conditioning_input_not_object_outcome": True,
        },
        "frame_zero_custody": {
            "archive_relative_path": FRAME_ZERO_ARCHIVE_FILENAME,
            "archive_file_sha256": _sha_at(
                staged_files,
                FRAME_ZERO_ARCHIVE_FILENAME,
            ),
            "splat_relative_path": (f"{frame_zero_episode}/splatfacto/splat_0.ply"),
            "splat_file_sha256": _sha_at(
                staged_files,
                f"{frame_zero_episode}/splatfacto/splat_0.ply",
            ),
            "all_prefix_panel_depth_files_bound": True,
        },
        "inventories": {
            "aligned_source_episode": source_inventory,
            "staged_prediction_case": staged_inventory,
        },
        "information_boundary": dict(INFORMATION_BOUNDARY),
    }
    payload["artifact_sha256"] = artifact_sha256(payload)
    return payload


def _canonical_absent_output(path: str | Path) -> Path:
    destination = Path(path).absolute()
    _require(
        not destination.exists() and not destination.is_symlink(),
        f"source-custody seal already exists: {destination}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    parent = destination.parent.resolve(strict=True)
    _require(
        parent == destination.parent and not parent.is_symlink(),
        "source-custody seal parent is noncanonical",
    )
    return parent / destination.name


def _write_once_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"source-custody seal already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    snapshot = _stable_regular_file(
        path,
        label="published source-custody seal",
        capture_payload=True,
    )
    _require(snapshot.payload == serialized, "published source-custody seal changed")


def build_confirmation_source_custody_seal(
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    source_episode_dir: str | Path,
    staged_case_dir: str | Path,
    output_path: str | Path,
    *,
    expected_h1: str | None = None,
    replay: bool = False,
) -> dict[str, Any]:
    """Build one absent custody seal, or validate it without rewriting in replay mode."""

    if replay:
        return validate_confirmation_source_custody_seal(
            output_path,
            lock_path,
            h2_commit,
            case_id,
            source_episode_dir,
            staged_case_dir,
            expected_h1=expected_h1,
        )
    output = _canonical_absent_output(output_path)
    source = Path(source_episode_dir).absolute()
    staged = Path(staged_case_dir).absolute()
    _require(
        output != source
        and output != staged
        and output not in source.parents
        and output not in staged.parents
        and source not in output.parents
        and staged not in output.parents,
        "source-custody seal must be outside both input trees",
    )
    payload = _capture_payload(
        lock_path,
        h2_commit,
        case_id,
        source_episode_dir,
        staged_case_dir,
        output,
        expected_h1=expected_h1,
    )
    _write_once_json(output, payload)
    return payload


def _envelope_absolute_path(value: object, *, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} path is absent")
    path = Path(value)
    _require(
        path.is_absolute()
        and str(path) == os.path.normpath(value)
        and ".." not in path.parts,
        f"{label} path is not canonical absolute syntax",
    )
    return path


def validate_confirmation_source_custody_envelope(
    seal_path: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    *,
    expected_h1: str | None = None,
    expected_source_episode_dir: str | Path | None = None,
    expected_staged_case_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate only the lock and custody JSON envelopes.

    This validator is safe to call inside the prediction boundary.  It opens
    exactly two files: ``seal_path`` and ``lock_path``.  Source and staged paths
    are checked lexically and against the sealed inventory records, but no
    source video, timestamp, mask, depth, Splat, robot, or calibration file is
    statted or opened.  Full input replay remains the responsibility of
    :func:`validate_confirmation_source_custody_seal`.
    """

    _require(_is_full_sha1(h2_commit), "H2 must be a full lowercase nonzero SHA-1")
    lock_snapshot, lock = _load_json_snapshot(lock_path, label="H2 cohort lock")
    validate_confirmation_cohort_lock(
        lock,
        expected_implementation_commit_h1=expected_h1,
    )
    seal_snapshot, observed = _load_json_snapshot(
        seal_path,
        label="source-custody seal",
    )
    _require(
        set(observed)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "case_identity",
            "lock_binding",
            "path_binding",
            "manifests",
            "source_camera_panel",
            "camera_panel",
            "camera_count",
            "raw_rgb24_prefix",
            "camera_custody",
            "known_action",
            "frame_zero_custody",
            "inventories",
            "information_boundary",
            "artifact_sha256",
        }
        and observed.get("schema_version") == SCHEMA_VERSION
        and observed.get("artifact_kind") == ARTIFACT_KIND
        and observed.get("protocol_id") == PROTOCOL_ID
        and observed.get("artifact_sha256") == artifact_sha256(observed),
        "source-custody artifact envelope changed",
    )
    identity = _external_case_identity(lock, case_id)
    _require(
        observed.get("case_identity") == identity,
        "source-custody case identity changed",
    )
    binding = observed.get("lock_binding")
    _require(
        isinstance(binding, Mapping)
        and set(binding)
        == {
            "implementation_commit_h1",
            "cohort_lock_commit_h2",
            "cohort_lock_artifact_sha256",
            "cohort_lock_file_sha256",
        }
        and binding.get("implementation_commit_h1")
        == lock["two_commit_freeze"]["implementation_commit_h1"]
        and binding.get("cohort_lock_commit_h2") == h2_commit
        and binding.get("cohort_lock_artifact_sha256") == lock["artifact_sha256"]
        and binding.get("cohort_lock_file_sha256") == lock_snapshot.sha256,
        "source-custody lock binding changed",
    )
    path_binding = observed.get("path_binding")
    _require(
        isinstance(path_binding, Mapping)
        and set(path_binding)
        == {
            "cohort_lock_path",
            "source_episode_dir",
            "staged_case_dir",
            "custody_seal_path",
        },
        "source-custody path binding changed",
    )
    bound_lock = _envelope_absolute_path(
        path_binding["cohort_lock_path"],
        label="bound lock",
    )
    source = _envelope_absolute_path(
        path_binding["source_episode_dir"],
        label="bound source episode",
    )
    staged = _envelope_absolute_path(
        path_binding["staged_case_dir"],
        label="bound staged case",
    )
    bound_seal = _envelope_absolute_path(
        path_binding["custody_seal_path"],
        label="bound source-custody seal",
    )
    _require(
        bound_lock == lock_snapshot.path
        and bound_seal == seal_snapshot.path
        and staged.name == case_id
        and source.name == f"episode_{identity['episode_id']:04d}"
        and source.parent.name == identity["object_id"]
        and source != staged
        and source not in staged.parents
        and staged not in source.parents,
        "source-custody absolute path binding changed",
    )
    _require(
        bound_seal != source
        and bound_seal != staged
        and bound_seal not in source.parents
        and bound_seal not in staged.parents
        and source not in bound_seal.parents
        and staged not in bound_seal.parents,
        "source-custody seal path overlaps an input tree",
    )
    if expected_source_episode_dir is not None:
        _require(
            source == Path(expected_source_episode_dir).absolute(),
            "source-custody seal binds another expected source episode",
        )
    if expected_staged_case_dir is not None:
        _require(
            staged == Path(expected_staged_case_dir).absolute(),
            "source-custody seal binds another expected staged case",
        )

    manifests = observed.get("manifests")
    expected_manifest_paths = {
        "source_preparation": SOURCE_PREPARATION_FILENAME,
        "prediction_prefix": PREDICTION_PREFIX_MANIFEST_FILENAME,
        "frame_zero": FRAME_ZERO_MANIFEST_FILENAME,
    }
    _require(
        isinstance(manifests, Mapping)
        and set(manifests) == set(expected_manifest_paths),
        "source-custody manifest envelope changed",
    )
    for role, relative in expected_manifest_paths.items():
        record = manifests[role]
        _require(
            isinstance(record, Mapping)
            and set(record) == {"relative_path", "file_sha256", "result_sha256"}
            and record.get("relative_path") == relative
            and _is_sha256(record.get("file_sha256"))
            and _is_sha256(record.get("result_sha256")),
            f"source-custody {role} manifest record changed",
        )

    source_cameras = observed.get("source_camera_panel")
    cameras = observed.get("camera_panel")
    _require(
        isinstance(source_cameras, list)
        and len(source_cameras) >= 8
        and source_cameras == sorted(source_cameras)
        and len(source_cameras) == len(set(source_cameras))
        and all(isinstance(camera, str) and bool(camera) for camera in source_cameras)
        and isinstance(cameras, list)
        and len(cameras) >= 8
        and cameras == sorted(cameras)
        and len(cameras) == len(set(cameras))
        and all(isinstance(camera, str) and bool(camera) for camera in cameras)
        and observed.get("camera_count") == len(cameras),
        "source-custody camera panel changed",
    )
    _require(
        set(cameras) <= set(source_cameras),
        "source camera panel omits a prefix-panel camera",
    )
    inventories = observed.get("inventories")
    _require(
        isinstance(inventories, Mapping)
        and set(inventories) == {"aligned_source_episode", "staged_prediction_case"},
        "source-custody inventories envelope changed",
    )
    source_inventory = inventories["aligned_source_episode"]
    _require(
        isinstance(source_inventory, Mapping)
        and isinstance(source_inventory.get("records"), list),
        "source inventory records are absent",
    )
    source_directories, source_expected_files = _source_expected_paths(source_cameras)
    staged_directories, staged_expected_files = _staged_expected_paths(cameras)
    source_files = _validate_inventory_envelope(
        source_inventory,
        expected_root=source,
        expected_directories=source_directories,
        expected_files=source_expected_files,
        label="aligned source episode",
    )
    staged_files = _validate_inventory_envelope(
        inventories["staged_prediction_case"],
        expected_root=staged,
        expected_directories=staged_directories,
        expected_files=staged_expected_files,
        label="staged prediction case",
    )
    _require(
        manifests["source_preparation"]["file_sha256"]
        == _sha_at(source_files, SOURCE_PREPARATION_FILENAME)
        and manifests["prediction_prefix"]["file_sha256"]
        == _sha_at(staged_files, PREDICTION_PREFIX_MANIFEST_FILENAME)
        and manifests["frame_zero"]["file_sha256"]
        == _sha_at(staged_files, FRAME_ZERO_MANIFEST_FILENAME),
        "manifest records differ from their custody inventories",
    )

    raw_rgb = observed.get("raw_rgb24_prefix")
    camera_custody = observed.get("camera_custody")
    _require(
        isinstance(raw_rgb, Mapping)
        and set(raw_rgb)
        == {
            "algorithm",
            "frame_count",
            "by_camera",
            "direct_source_vs_reencoded_prefix_equality_required",
            "authorized_future_must_reuse_this_digest",
        }
        and raw_rgb.get("algorithm") == RAW_RGB24_PREFIX_ALGORITHM
        and raw_rgb.get("frame_count") == PREFIX_FRAME_COUNT
        and raw_rgb.get("direct_source_vs_reencoded_prefix_equality_required") is False
        and raw_rgb.get("authorized_future_must_reuse_this_digest") is True
        and isinstance(raw_rgb.get("by_camera"), Mapping)
        and set(raw_rgb["by_camera"]) == set(cameras)
        and all(_is_sha256(value) for value in raw_rgb["by_camera"].values())
        and isinstance(camera_custody, Mapping)
        and set(camera_custody) == set(cameras),
        "raw-RGB prefix or camera-custody envelope changed",
    )
    prefix_episode = "prefix/episode_0000"
    frame_episode = "frame-zero/episode_0000"
    for camera in cameras:
        record = camera_custody[camera]
        source_prefix_range = (
            record.get("source_prefix_frame_range_half_open")
            if isinstance(record, Mapping)
            else None
        )
        _require(
            isinstance(record, Mapping)
            and set(record) == CAMERA_CUSTODY_KEYS
            and all(
                _is_sha256(record.get(key))
                for key in CAMERA_CUSTODY_KEYS
                if key.endswith("_sha256")
            )
            and isinstance(source_prefix_range, list)
            and len(source_prefix_range) == 2
            and all(type(value) is int for value in source_prefix_range)
            and source_prefix_range[0] >= 0
            and source_prefix_range[1] - source_prefix_range[0] == PREFIX_FRAME_COUNT
            and record.get("timestamp_prefix_exact_source_slice") is True
            and record.get("timestamp_frame_zero_exact_source_slice") is True
            and raw_rgb["by_camera"][camera] == record["decoded_rgb24_prefix_sha256"],
            f"camera-custody record changed: {camera}",
        )
        source_prefix = f"{camera}"
        staged_prefix = f"{prefix_episode}/{camera}"
        staged_frame = f"{frame_episode}/{camera}"
        expected_hashes = {
            "source_full_video_file_sha256": _sha_at(
                source_files,
                f"{source_prefix}/undistorted.mp4",
            ),
            "source_full_timestamps_file_sha256": _sha_at(
                source_files,
                f"{source_prefix}/aligned_timestamps.txt",
            ),
            "staged_prefix_video_file_sha256": _sha_at(
                staged_files,
                f"{staged_prefix}/undistorted.mp4",
            ),
            "staged_prefix_timestamps_file_sha256": _sha_at(
                staged_files,
                f"{staged_prefix}/aligned_timestamps.txt",
            ),
            "staged_prefix_mask_file_sha256": _sha_at(
                staged_files,
                f"{staged_prefix}/mask_refined.h5",
            ),
            "staged_prefix_depth_file_sha256": _sha_at(
                staged_files,
                f"{staged_prefix}/rendered_depth.h5",
            ),
            "staged_frame_zero_video_file_sha256": _sha_at(
                staged_files,
                f"{staged_frame}/undistorted.mp4",
            ),
            "staged_frame_zero_timestamps_file_sha256": _sha_at(
                staged_files,
                f"{staged_frame}/aligned_timestamps.txt",
            ),
            "staged_frame_zero_mask_file_sha256": _sha_at(
                staged_files,
                f"{staged_frame}/mask_refined.h5",
            ),
            "staged_frame_zero_depth_file_sha256": _sha_at(
                staged_files,
                f"{staged_frame}/rendered_depth.h5",
            ),
            "staged_frame_zero_gripper_mask_file_sha256": _sha_at(
                staged_files,
                f"{staged_frame}/rendered_urdf.h5",
            ),
        }
        _require(
            all(record[key] == value for key, value in expected_hashes.items()),
            f"camera custody differs from inventories: {camera}",
        )

    known_action = observed.get("known_action")
    _require(
        isinstance(known_action, Mapping)
        and set(known_action)
        == {
            "relative_path",
            "frame_count",
            "file_sha256",
            "conditioning_input_not_object_outcome",
        }
        and known_action.get("relative_path") == "known-action/robot.npz"
        and known_action.get("frame_count") == KNOWN_ACTION_FRAME_COUNT
        and known_action.get("file_sha256")
        == _sha_at(staged_files, "known-action/robot.npz")
        and known_action.get("conditioning_input_not_object_outcome") is True,
        "known-action custody envelope changed",
    )
    frame_zero = observed.get("frame_zero_custody")
    frame_splat = f"{frame_episode}/splatfacto/splat_0.ply"
    _require(
        isinstance(frame_zero, Mapping)
        and set(frame_zero)
        == {
            "archive_relative_path",
            "archive_file_sha256",
            "splat_relative_path",
            "splat_file_sha256",
            "all_prefix_panel_depth_files_bound",
        }
        and frame_zero.get("archive_relative_path") == FRAME_ZERO_ARCHIVE_FILENAME
        and frame_zero.get("archive_file_sha256")
        == _sha_at(staged_files, FRAME_ZERO_ARCHIVE_FILENAME)
        and frame_zero.get("splat_relative_path") == frame_splat
        and frame_zero.get("splat_file_sha256") == _sha_at(staged_files, frame_splat)
        and frame_zero.get("all_prefix_panel_depth_files_bound") is True,
        "frame-zero custody envelope changed",
    )
    _require(
        observed.get("information_boundary") == INFORMATION_BOUNDARY,
        "source-custody information boundary changed",
    )
    return observed


def validate_confirmation_source_custody_seal(
    seal_path: str | Path,
    lock_path: str | Path,
    h2_commit: str,
    case_id: str,
    source_episode_dir: str | Path,
    staged_case_dir: str | Path,
    *,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Replay every custody check and validate one existing write-once seal."""

    observed = validate_confirmation_source_custody_envelope(
        seal_path,
        lock_path,
        h2_commit,
        case_id,
        expected_h1=expected_h1,
        expected_source_episode_dir=source_episode_dir,
        expected_staged_case_dir=staged_case_dir,
    )
    snapshot = _stable_regular_file(
        seal_path,
        label="source-custody seal",
        capture_payload=False,
    )
    expected = _capture_payload(
        lock_path,
        h2_commit,
        case_id,
        source_episode_dir,
        staged_case_dir,
        snapshot.path,
        expected_h1=expected_h1,
    )
    _require(
        observed.get("artifact_sha256") == artifact_sha256(observed),
        "source-custody artifact checksum changed",
    )
    _require(
        observed == expected,
        "source-custody replay differs from the write-once artifact",
    )
    return observed


__all__ = [
    "ARTIFACT_KIND",
    "FRAME_ZERO_ARCHIVE_FILENAME",
    "FRAME_ZERO_FRAME_COUNT",
    "FRAME_ZERO_MANIFEST_FILENAME",
    "INFORMATION_BOUNDARY",
    "KNOWN_ACTION_FRAME_COUNT",
    "PREFIX_FRAME_COUNT",
    "PREDICTION_PREFIX_MANIFEST_FILENAME",
    "RAW_RGB24_PREFIX_ALGORITHM",
    "SCHEMA_VERSION",
    "SOURCE_PREPARATION_FILENAME",
    "STAGING_FRAME_COUNT",
    "artifact_sha256",
    "build_confirmation_source_custody_seal",
    "validate_confirmation_source_custody_envelope",
    "validate_confirmation_source_custody_seal",
]
