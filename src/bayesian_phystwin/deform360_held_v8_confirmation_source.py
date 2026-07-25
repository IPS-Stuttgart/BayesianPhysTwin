"""Post-GO materialization of the exact held-v8.1 confirmation source cohort.

This module deliberately knows nothing about how a protocol capability is
represented.  Its first operation is to consume the injected, process-local
capability.  No destination, provider, repository, or payload path is even
statted before that call succeeds.

The legacy prospective inventory contains 32 metadata records per case: the
12 source-QA camera MP4/timestamp pairs and four tactile NPY/timestamp pairs.
The tactile records are used only to reproduce the already-locked remote
inventory digest.  Tactile payloads are never requested, downloaded, or read.
The processing allowlist contains the 24 camera files plus the three shared
calibration arrays and metadata.json.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Any, Protocol

from .deform360_robot_kinematics import load_robot_kinematics_archive


PROTOCOL_ID = "deform360-held-online-belief-v8.1"
SCHEMA_VERSION = 1
SOURCE_OPERATION = "materialize-confirmation-source-cohort-v1"
COHORT_MANIFEST_KIND = "Deform360HeldV8ConfirmationAlignedSourceCohort"
COHORT_STATUS = "confirmation-source-cohort-complete"
HF_REPO_ID = "brownu/deform360"
HF_REPO_TYPE = "dataset"
HF_DATASET_REVISION = "7fea8e20231a47641d1d2bc8791920ec4e62ec5e"
PROCESSING_CODE_REVISION = "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317"
PROCESSING_CODE_TREE = "c566ed29db7e0fd6a4cb768d840a4aa662864680"
PINNED_PYTHON_LINK_TARGET = "/usr/bin/python3"
PINNED_PYTHON_RESOLVED = Path("/usr/bin/python3.12")
PINNED_PYTHON_TARGET_SHA256 = (
    "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7"
)
SOURCE_GEOMETRY_QA_FILE_SHA256 = (
    "5cba6655ba3714f949a0342813e94c2c72fee0112ab9a0f6734b1d5b44b0501c"
)
REPLICATION_PREREGISTRATION_FILE_SHA256 = (
    "7660e0313edccc603d5f0e3f8e6b3c1268dd472c06da304172d19ef1d461d200"
)
LOCAL_SUBSET_EPISODE_INDEX = 0
UNDISTORTION_TOLERANCE_US = 100_000

TACTILE_STREAMS = (
    "brics-odroid_tactilel_left",
    "brics-odroid_tactilel_right",
    "brics-odroid_tactiler_left",
    "brics-odroid_tactiler_right",
)
SHARED_RELATIVE_PATHS = (
    "calibration_refined/dist.npy",
    "calibration_refined/extrinsics.npy",
    "calibration_refined/intrinsics.npy",
    "metadata.json",
)


@dataclass(frozen=True)
class ConfirmationSourceCase:
    case_name: str
    object_id: str
    episode_id: int
    remote_inventory_sha256: str
    remote_file_count: int
    remote_total_bytes: int
    processing_inventory_sha256: str
    processing_total_bytes: int
    bimanual: bool
    cameras: tuple[str, ...]


CONFIRMATION_SOURCE_CASES = (
    ConfirmationSourceCase(
        "002-rope-silk-ep0001",
        "002-rope-silk",
        1,
        "b33791f6faa8d05717408d7b77cf1405083b614fe42ecef3a3538a0dc2008858",
        32,
        37_863_432,
        "e22d487e67bdfd9c80ce2fe17948034705cd8919bb3f3f06093f92f7dc04f821",
        35_071_469,
        False,
        (
            "brics-odroid-001_cam0",
            "brics-odroid-013_cam0",
            "brics-odroid-021_cam1",
            "brics-odroid-007_cam0",
            "brics-odroid-028_cam0",
            "brics-odroid-019_cam1",
            "brics-odroid-010_cam0",
            "brics-odroid-014_cam1",
            "brics-odroid-025_cam1",
            "brics-odroid-006_cam0",
            "brics-odroid-015_cam1",
            "brics-odroid-027_cam0",
        ),
    ),
    ConfirmationSourceCase(
        "081-stripe-rope-ep0005",
        "081-stripe-rope",
        5,
        "6055375fb66ea1e0732e808d855e4eecb66687f14dfd6a6a604d5d9a39a194e0",
        32,
        61_222_868,
        "b4e3233b7b099a095598964ac908e7b31f22059671080152baf6adaa764cc523",
        57_750_695,
        True,
        (
            "brics-odroid-001_cam0",
            "brics-odroid-027_cam0",
            "brics-odroid-011_cam0",
            "brics-odroid-008_cam1",
            "brics-odroid-021_cam1",
            "brics-odroid-019_cam1",
            "brics-odroid-024_cam1",
            "brics-odroid-015_cam0",
            "brics-odroid-010_cam1",
            "brics-odroid-017_cam1",
            "brics-odroid-023_cam0",
            "brics-odroid-014_cam1",
        ),
    ),
    ConfirmationSourceCase(
        "085-scarf-cloth-ep0002",
        "085-scarf-cloth",
        2,
        "cb9ee9be4c99244e94f676a329b31ecb629c0afef9b7ffbe6060a6b061b81249",
        32,
        31_710_094,
        "9e6ad1ce079d56764b085a9a67525d14020319a67806d106dfd755d25abc3329",
        29_598_061,
        False,
        (
            "brics-odroid-001_cam0",
            "brics-odroid-027_cam0",
            "brics-odroid-011_cam0",
            "brics-odroid-028_cam0",
            "brics-odroid-021_cam1",
            "brics-odroid-019_cam1",
            "brics-odroid-010_cam0",
            "brics-odroid-024_cam1",
            "brics-odroid-015_cam0",
            "brics-odroid-006_cam0",
            "brics-odroid-017_cam1",
            "brics-odroid-023_cam0",
        ),
    ),
    ConfirmationSourceCase(
        "083-blanket-cloth-ep0007",
        "083-blanket-cloth",
        7,
        "102f9edd98b6d3703c3d98625a358c7588d87c79024c795d233771e76b10be84",
        32,
        53_947_570,
        "4c0416e1218d6a6579d2ea3f1c7b06e810e0d55230fa08b5530f7060ea71fe4f",
        50_787_356,
        True,
        (
            "brics-odroid-001_cam0",
            "brics-odroid-013_cam0",
            "brics-odroid-021_cam1",
            "brics-odroid-007_cam0",
            "brics-odroid-028_cam0",
            "brics-odroid-019_cam1",
            "brics-odroid-010_cam0",
            "brics-odroid-014_cam1",
            "brics-odroid-025_cam1",
            "brics-odroid-006_cam0",
            "brics-odroid-015_cam1",
            "brics-odroid-012_cam1",
        ),
    ),
    ConfirmationSourceCase(
        "092-squirrel-ep0001",
        "092-squirrel",
        1,
        "6f02afc8e8101fdc0e30ee171435162d1d6a4d648f5ee910070f711313d2b960",
        32,
        38_161_504,
        "79fec8a5ff5cccae59af499129d5c572368e8aeea91003530b3e0d31e86d0bfa",
        35_012_467,
        False,
        (
            "brics-odroid-001_cam0",
            "brics-odroid-013_cam0",
            "brics-odroid-021_cam1",
            "brics-odroid-007_cam0",
            "brics-odroid-008_cam0",
            "brics-odroid-019_cam1",
            "brics-odroid-006_cam0",
            "brics-odroid-014_cam1",
            "brics-odroid-010_cam0",
            "brics-odroid-024_cam1",
            "brics-odroid-015_cam1",
            "brics-odroid-027_cam0",
        ),
    ),
    ConfirmationSourceCase(
        "170-spider-ep0006",
        "170-spider",
        6,
        "c19cb57b087aa98c5e792e8dfcb2e889cb4b2a52653a78a2cba6591a0fdc80a7",
        32,
        47_269_453,
        "babf0e8e6e4555442e923ae5f0059480c940fcf8e859ed1e80b1bcc7f1227527",
        44_778_111,
        True,
        (
            "brics-odroid-001_cam0",
            "brics-odroid-027_cam0",
            "brics-odroid-011_cam0",
            "brics-odroid-008_cam1",
            "brics-odroid-021_cam1",
            "brics-odroid-019_cam1",
            "brics-odroid-024_cam1",
            "brics-odroid-015_cam0",
            "brics-odroid-010_cam1",
            "brics-odroid-017_cam1",
            "brics-odroid-023_cam0",
            "brics-odroid-028_cam0",
        ),
    ),
)

CONFIRMATION_SOURCE_CASE_NAMES = tuple(
    case.case_name for case in CONFIRMATION_SOURCE_CASES
)
CONFIRMATION_SOURCE_CONTRACT = {
    "contract_id": "deform360-held-v8-confirmation-source-cohort-v1",
    "repository": {
        "repo_id": HF_REPO_ID,
        "repo_type": HF_REPO_TYPE,
        "revision": HF_DATASET_REVISION,
    },
    "operation": SOURCE_OPERATION,
    "ordered_case_names": list(CONFIRMATION_SOURCE_CASE_NAMES),
    "selection_lineage": {
        "camera_lists_source": (
            "milestones/deform360-replication-source-qa-v1/artifacts/"
            "source_geometry_qa.json"
        ),
        "camera_lists_source_sha256": SOURCE_GEOMETRY_QA_FILE_SHA256,
        "bimanual_flags_source": (
            "configs/sota/deform360_replication_v1.json"
        ),
        "bimanual_flags_source_sha256": REPLICATION_PREREGISTRATION_FILE_SHA256,
    },
    "legacy_inventory": {
        "record_keys": ["blob_id", "path", "sha256", "size"],
        "canonical_json": "sort_keys-compact-separators-no-newline",
        "camera_stream_count": 12,
        "camera_extensions": [".mp4", ".txt"],
        "tactile_streams": list(TACTILE_STREAMS),
        "tactile_extensions": [".npy", ".txt"],
        "tactile_payload_downloaded_or_read": False,
    },
    "processing_download": {
        "camera_stream_count": 12,
        "camera_file_count": 24,
        "shared_relative_paths": list(SHARED_RELATIVE_PATHS),
        "record_count_per_case": 28,
        "tactile_path_permitted": False,
    },
    "processing": {
        "code_revision": PROCESSING_CODE_REVISION,
        "code_tree": PROCESSING_CODE_TREE,
        "isolated_subset_episode_index": LOCAL_SUBSET_EPISODE_INDEX,
        "undistortion_tolerance_us": UNDISTORTION_TOLERANCE_US,
        "rebuild_timeline": False,
        "robot_seed": 0,
        "overwrite": True,
        "plot": False,
    },
}

_CAMERA_OUTPUT_FILES = frozenset(
    {
        "undistorted.mp4",
        "undistorted_000000.png",
        "aligned_timestamps.txt",
        "alignment.json",
        "metadata.json",
    }
)
_EPISODE_OUTPUT_FILES = frozenset(
    {"undistorted_intrinsics.npy", "extrinsics.npy", "alignment.json"}
)
_ROBOT_OUTPUT_FILES = frozenset({"robot.npz", "robot.meta.json"})


class PermitConsumer(Protocol):
    def __call__(
        self,
        permit: object,
        *,
        operation: str,
        ordered_case_names: Sequence[str],
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
class ConfirmationSourcePaths:
    source_root: Path
    processing_code_root: Path
    python_executable: Path


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


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    unsigned = deepcopy(dict(value))
    unsigned.pop("artifact_sha256", None)
    return _digest(unsigned)


def confirmation_source_contract_sha256() -> str:
    return _digest(CONFIRMATION_SOURCE_CONTRACT)


def _field(item: object, name: str) -> object:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _remote_record(item: object) -> dict[str, Any] | None:
    path = _field(item, "path")
    size = _field(item, "size")
    blob_id = _field(item, "blob_id")
    if not isinstance(path, str):
        return None
    if size is None and blob_id is None:
        # huggingface_hub RepoFolder entries have a path but no file fields.
        return None
    _require(type(size) is int and size >= 0, "remote record size is invalid")
    _require(
        isinstance(blob_id, str)
        and len(blob_id) == 40
        and all(character in "0123456789abcdef" for character in blob_id),
        "remote record blob id is invalid",
    )
    lfs = _field(item, "lfs")
    if isinstance(lfs, Mapping):
        sha256 = lfs.get("sha256")
    else:
        sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
    _require(
        sha256 is None
        or (
            isinstance(sha256, str)
            and len(sha256) == 64
            and all(character in "0123456789abcdef" for character in sha256)
        ),
        "remote LFS SHA-256 is invalid",
    )
    return {"path": path, "size": size, "sha256": sha256, "blob_id": blob_id}


def _valid_hex(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_remote_records(
    value: object,
    *,
    case: ConfirmationSourceCase,
    expected_count: int,
    processing_only: bool,
) -> list[Mapping[str, Any]]:
    _require(
        isinstance(value, list) and len(value) == expected_count,
        f"remote record count changed: {case.case_name}",
    )
    records: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for item in value:
        _require(
            isinstance(item, Mapping)
            and set(item) == {"path", "size", "sha256", "blob_id"}
            and isinstance(item.get("path"), str)
            and type(item.get("size")) is int
            and item["size"] >= 0
            and (
                item.get("sha256") is None or _valid_hex(item.get("sha256"), length=64)
            )
            and _valid_hex(item.get("blob_id"), length=40),
            f"remote record fields changed: {case.case_name}",
        )
        path = str(item["path"])
        parts = Path(path).parts
        _require(
            not path.startswith("/")
            and ".." not in parts
            and path.startswith(f"raw/{case.object_id}/"),
            f"remote record path escaped its case: {case.case_name}",
        )
        records.append(item)
        paths.append(path)
    _require(
        paths == sorted(paths) and len(set(paths)) == len(paths),
        f"remote record ordering changed: {case.case_name}",
    )
    tactile_paths = [
        path
        for path in paths
        if len(Path(path).parts) == 4 and Path(path).parts[2] in TACTILE_STREAMS
    ]
    if processing_only:
        _require(
            not tactile_paths,
            f"processing inventory contains tactile payloads: {case.case_name}",
        )
        expected_shared = {
            f"raw/{case.object_id}/{relative}" for relative in SHARED_RELATIVE_PATHS
        }
        _require(
            expected_shared.issubset(paths),
            f"processing inventory lacks shared inputs: {case.case_name}",
        )
    else:
        _require(
            len(tactile_paths) == 2 * len(TACTILE_STREAMS),
            f"legacy tactile metadata inventory changed: {case.case_name}",
        )
    return records


def _validate_content_record_metadata(
    value: object,
    *,
    label: str,
) -> list[Mapping[str, Any]]:
    _require(isinstance(value, list), f"{label} records are not a list")
    records: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for item in value:
        _require(
            isinstance(item, Mapping)
            and set(item) == {"path", "size_bytes", "sha256"}
            and isinstance(item.get("path"), str)
            and type(item.get("size_bytes")) is int
            and item["size_bytes"] >= 0
            and _valid_hex(item.get("sha256"), length=64),
            f"{label} record fields changed",
        )
        relative = str(item["path"])
        _require(
            relative
            and not relative.startswith("/")
            and ".." not in Path(relative).parts,
            f"{label} record path is unsafe",
        )
        records.append(item)
        paths.append(relative)
    _require(
        paths == sorted(paths) and len(set(paths)) == len(paths),
        f"{label} record ordering changed",
    )
    return records


def _selected_records(
    items: Iterable[object],
    case: ConfirmationSourceCase,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    root = f"raw/{case.object_id}"
    records: dict[str, dict[str, Any]] = {}
    for item in items:
        record = _remote_record(item)
        if record is None:
            continue
        path = record["path"]
        _require(path not in records, f"duplicate remote path: {path}")
        records[path] = record
    streams: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    accepted_streams = set(case.cameras) | set(TACTILE_STREAMS)
    for record in records.values():
        parts = record["path"].split("/")
        if len(parts) != 4 or "/".join(parts[:2]) != root:
            continue
        stream = parts[2]
        if stream not in accepted_streams:
            continue
        filename = parts[3]
        if "." not in filename:
            continue
        stem, extension = filename.rsplit(".", 1)
        streams[stream][stem][extension] = record

    legacy: list[dict[str, Any]] = []
    camera_only: list[dict[str, Any]] = []
    for stream in (*case.cameras, *TACTILE_STREAMS):
        extensions = {"npy", "txt"} if stream in TACTILE_STREAMS else {"mp4", "txt"}
        pairs = [
            (stem, values)
            for stem, values in sorted(streams[stream].items())
            if extensions.issubset(values) and not stem.startswith("median_")
        ]
        _require(len(pairs) == 10, f"remote stream episode count changed: {stream}")
        _stem, selected = pairs[case.episode_id]
        pair = [selected[extension] for extension in sorted(extensions)]
        legacy.extend(pair)
        if stream not in TACTILE_STREAMS:
            camera_only.extend(pair)
    legacy.sort(key=lambda record: record["path"])
    _require(
        len(legacy) == case.remote_file_count
        and sum(record["size"] for record in legacy) == case.remote_total_bytes
        and _digest(legacy) == case.remote_inventory_sha256,
        f"locked remote inventory changed for {case.case_name}",
    )

    shared: list[dict[str, Any]] = []
    for relative in SHARED_RELATIVE_PATHS:
        path = f"{root}/{relative}"
        _require(path in records, f"shared processing record is absent: {path}")
        shared.append(records[path])
    processing = sorted((*camera_only, *shared), key=lambda record: record["path"])
    replacement_schema = [
        {
            "path": record["path"],
            "size_bytes": record["size"],
            "blob_id": record["blob_id"],
            "lfs_sha256": record["sha256"],
        }
        for record in processing
    ]
    _require(
        len(processing) == 28
        and sum(record["size"] for record in processing) == case.processing_total_bytes
        and _digest(replacement_schema) == case.processing_inventory_sha256,
        f"processing-only remote inventory changed for {case.case_name}",
    )
    return tuple(legacy), tuple(processing)


def _stable_file_state(value: os.stat_result) -> tuple[int, ...]:
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


def _sha256_file(path: Path) -> str:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"not a regular file: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_file_state(opened) == _stable_file_state(observed),
            f"file changed while opening: {path}",
        )
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"file changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _read_regular_bytes(path: Path) -> bytes:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1,
        f"not a single-link regular file: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    payload = bytearray()
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_file_state(opened) == _stable_file_state(observed),
            f"file changed while opening: {path}",
        )
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"file changed while reading: {path}",
        )
    finally:
        os.close(descriptor)
    return bytes(payload)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _git_blob_sha1_file(path: Path) -> str:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        f"not a regular Git blob: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha1(usedforsecurity=False)
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_file_state(opened) == _stable_file_state(observed),
            f"Git blob changed while opening: {path}",
        )
        digest.update(f"blob {opened.st_size}\0".encode("ascii"))
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"Git blob changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _content_records(
    root: Path,
    remote: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    expected = {str(record["path"]): record for record in remote}
    observed_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(observed_paths == set(expected), "downloaded raw file allowlist changed")
    records: list[dict[str, Any]] = []
    for relative in sorted(expected):
        path = root / relative
        _require(path.resolve(strict=True) == path, "downloaded raw path has a link")
        remote_record = expected[relative]
        digest = _sha256_file(path)
        _require(
            path.stat().st_size == remote_record["size"],
            f"downloaded size changed: {relative}",
        )
        if remote_record["sha256"] is not None:
            _require(
                digest == remote_record["sha256"],
                f"downloaded LFS content changed: {relative}",
            )
        else:
            _require(
                _git_blob_sha1_file(path) == remote_record["blob_id"],
                f"downloaded Git blob changed: {relative}",
            )
        records.append(
            {"path": relative, "size_bytes": path.stat().st_size, "sha256": digest}
        )
    return tuple(records)


def _tree_records(root: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        observed = os.lstat(path)
        _require(not stat.S_ISLNK(observed.st_mode), f"source tree has a link: {path}")
        if stat.S_ISDIR(observed.st_mode):
            continue
        _require(
            stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1,
            f"source tree has a nonregular or hard-linked file: {path}",
        )
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": observed.st_size,
                "sha256": _sha256_file(path),
            }
        )
    return tuple(records)


_UNDISTORT_SCRIPT = """
import json
import sys
from pathlib import Path
processing = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(processing))
import deform360.undistort as module
if processing not in Path(module.__file__).resolve(strict=True).parents:
    raise RuntimeError("undistort module resolved outside the pinned source")
module.undistort_episode(
    object_dir=sys.argv[2],
    output_dir=sys.argv[3],
    episode_index=0,
    cameras=json.loads(sys.argv[4]),
    tol_units=100000,
    overwrite=True,
    rebuild_timeline=False,
)
""".strip()

_ROBOT_SCRIPT = """
import json
import sys
from pathlib import Path
processing = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(processing))
import deform360.processing.robot_stage as module
if processing not in Path(module.__file__).resolve(strict=True).parents:
    raise RuntimeError("robot-stage module resolved outside the pinned source")
module.process_robot_episode(
    aligned_dir=sys.argv[2],
    episode_index=int(sys.argv[3]),
    bimanual=(sys.argv[4] == "true"),
    cameras=json.loads(sys.argv[5]),
    seed=0,
    overwrite=True,
    plot=False,
)
""".strip()


def _default_inventory_provider(**kwargs: Any) -> Iterable[object]:
    from huggingface_hub import HfApi

    return HfApi().list_repo_tree(**kwargs)


def _default_snapshot_downloader(**kwargs: Any) -> str | os.PathLike[str]:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _default_command_runner(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> object:
    return subprocess.run(
        list(command),
        check=True,
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
    )


def _git_value(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "HOME": "/nonexistent",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    return completed.stdout.decode("ascii").strip()


def _validated_python(path: Path) -> dict[str, str]:
    value = Path(os.path.abspath(path))
    try:
        lexical = os.lstat(value)
        resolved = value.resolve(strict=True)
        target = os.stat(resolved, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"pinned processing Python is missing: {value}") from error
    _require(
        stat.S_ISLNK(lexical.st_mode)
        and os.readlink(value) == PINNED_PYTHON_LINK_TARGET
        and resolved == PINNED_PYTHON_RESOLVED
        and stat.S_ISREG(target.st_mode)
        and os.access(value, os.X_OK)
        and _sha256_file(resolved) == PINNED_PYTHON_TARGET_SHA256,
        "pinned processing Python identity changed",
    )
    return {
        "python_executable": str(value),
        "python_link_target": PINNED_PYTHON_LINK_TARGET,
        "python_resolved": str(resolved),
        "python_resolved_sha256": PINNED_PYTHON_TARGET_SHA256,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, "manifest write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_tree(root: Path) -> None:
    for path in sorted(
        root.rglob("*"), key=lambda value: len(value.parts), reverse=True
    ):
        observed = os.lstat(path)
        _require(not stat.S_ISLNK(observed.st_mode), f"source tree has a link: {path}")
        if stat.S_ISDIR(observed.st_mode):
            os.chmod(path, 0o500, follow_symlinks=False)
        else:
            _require(
                stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1,
                f"source tree has a nonregular or hard-linked file: {path}",
            )
            os.chmod(path, 0o400, follow_symlinks=False)
    os.chmod(root, 0o500, follow_symlinks=False)


def _validate_episode_layout(
    episode: Path, *, case: ConfirmationSourceCase
) -> tuple[dict[str, Any], ...]:
    expected_top = set(case.cameras) | {"robot"} | set(_EPISODE_OUTPUT_FILES)
    _require(
        {entry.name for entry in episode.iterdir()} == expected_top,
        f"aligned episode top-level allowlist changed: {case.case_name}",
    )
    for camera in case.cameras:
        directory = episode / camera
        _require(
            directory.is_dir()
            and not directory.is_symlink()
            and {entry.name for entry in directory.iterdir()} == _CAMERA_OUTPUT_FILES,
            f"aligned camera allowlist changed: {case.case_name}/{camera}",
        )
    robot = episode / "robot"
    _require(
        robot.is_dir()
        and not robot.is_symlink()
        and {entry.name for entry in robot.iterdir()} == _ROBOT_OUTPUT_FILES,
        f"aligned robot allowlist changed: {case.case_name}",
    )
    alignment_path = episode / "alignment.json"
    alignment = _read_json_object(
        alignment_path, label=f"episode alignment {case.case_name}"
    )
    _require(
        isinstance(alignment, Mapping)
        and alignment.get("episode_index") == LOCAL_SUBSET_EPISODE_INDEX
        and alignment.get("cameras") == list(case.cameras)
        and type(alignment.get("frame_count")) is int
        and alignment["frame_count"] > 0,
        f"aligned episode metadata changed: {case.case_name}",
    )
    frame_count = alignment["frame_count"]
    robot_meta = _read_json_object(
        robot / "robot.meta.json", label=f"robot metadata {case.case_name}"
    )
    parameters = robot_meta.get("parameters")
    outputs = robot_meta.get("outputs")
    _require(
        isinstance(parameters, Mapping)
        and isinstance(outputs, Mapping)
        and parameters.get("seed") == 0
        and parameters.get("bimanual") is case.bimanual
        and parameters.get("cameras") == list(case.cameras)
        and outputs.get("bimanual") is case.bimanual
        and outputs.get("num_frames") == frame_count,
        f"robot provenance changed: {case.case_name}",
    )
    state = load_robot_kinematics_archive(
        robot / "robot.npz", expected_frame_count=frame_count
    )
    _require(
        state.bimanual is case.bimanual,
        f"robot archive bimanual state changed: {case.case_name}",
    )
    return _tree_records(episode)


def _validate_source_directory_layout(source_root: Path) -> None:
    _require(
        {entry.name for entry in source_root.iterdir()}
        == {"download", "aligned", "manifests"},
        "confirmation source top-level allowlist changed",
    )
    download = source_root / "download"
    raw = download / "raw"
    aligned = source_root / "aligned"
    manifests = source_root / "manifests"
    _require(
        {entry.name for entry in download.iterdir()} == {"raw"}
        and {entry.name for entry in raw.iterdir()}
        == {case.object_id for case in CONFIRMATION_SOURCE_CASES}
        and {entry.name for entry in aligned.iterdir()}
        == {case.object_id for case in CONFIRMATION_SOURCE_CASES}
        and {entry.name for entry in manifests.iterdir()}
        == {"aligned-source-cohort.json"},
        "confirmation source cohort directory allowlist changed",
    )
    for case in CONFIRMATION_SOURCE_CASES:
        raw_object = raw / case.object_id
        _require(
            {entry.name for entry in raw_object.iterdir()}
            == {*case.cameras, "calibration_refined", "metadata.json"},
            f"raw object directory allowlist changed: {case.case_name}",
        )
        for camera in case.cameras:
            directory = raw_object / camera
            _require(
                directory.is_dir()
                and not directory.is_symlink()
                and len(tuple(directory.iterdir())) == 2
                and {entry.suffix for entry in directory.iterdir()} == {".mp4", ".txt"},
                f"raw camera directory allowlist changed: {case.case_name}/{camera}",
            )
        calibration = raw_object / "calibration_refined"
        _require(
            {entry.name for entry in calibration.iterdir()}
            == {"dist.npy", "extrinsics.npy", "intrinsics.npy"},
            f"raw calibration allowlist changed: {case.case_name}",
        )
        aligned_object = aligned / case.object_id
        episode = aligned_object / f"episode_{case.episode_id:04d}"
        _require(
            {entry.name for entry in aligned_object.iterdir()} == {episode.name},
            f"aligned object directory allowlist changed: {case.case_name}",
        )


def _remove_owned_tree(root: Path, *, expected_identity: tuple[int, int]) -> None:
    """Remove only the just-published tree after a failed final validation."""

    observed_root = os.lstat(root)
    _require(
        stat.S_ISDIR(observed_root.st_mode)
        and not stat.S_ISLNK(observed_root.st_mode)
        and (observed_root.st_dev, observed_root.st_ino) == expected_identity,
        "refusing to clean up a replaced confirmation source root",
    )
    directories: list[Path] = []
    for current, names, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        observed_current = os.lstat(current_path)
        _require(
            stat.S_ISDIR(observed_current.st_mode)
            and not stat.S_ISLNK(observed_current.st_mode),
            "confirmation source cleanup encountered a non-directory",
        )
        os.chmod(current_path, 0o700, follow_symlinks=False)
        retained: list[str] = []
        for name in sorted(names):
            child = current_path / name
            observed = os.lstat(child)
            if stat.S_ISLNK(observed.st_mode):
                child.unlink()
                continue
            _require(
                stat.S_ISDIR(observed.st_mode),
                "confirmation source cleanup encountered a non-directory child",
            )
            os.chmod(child, 0o700, follow_symlinks=False)
            retained.append(name)
            directories.append(child)
        names[:] = retained
        for name in sorted(files):
            child = current_path / name
            observed = os.lstat(child)
            _require(
                not stat.S_ISDIR(observed.st_mode),
                "confirmation source cleanup encountered an unexpected directory",
            )
            child.unlink()
    for directory in sorted(
        directories, key=lambda value: len(value.parts), reverse=True
    ):
        directory.rmdir()
    current_root = os.lstat(root)
    _require(
        (current_root.st_dev, current_root.st_ino) == expected_identity,
        "confirmation source root changed during cleanup",
    )
    root.rmdir()


def materialize_confirmation_source_cohort(
    paths: ConfirmationSourcePaths,
    *,
    source_permit: object,
    consume_source_permit: PermitConsumer,
    expected_source_permit: Mapping[str, Any],
    inventory_provider: InventoryProvider | None = None,
    snapshot_downloader: SnapshotDownloader | None = None,
    command_runner: CommandRunner | None = None,
) -> Path:
    """Build, validate, atomically publish, and freeze the exact six cases."""

    # This must remain the first externally observable operation.
    consumed = consume_source_permit(
        source_permit,
        operation=SOURCE_OPERATION,
        ordered_case_names=CONFIRMATION_SOURCE_CASE_NAMES,
    )
    _require(
        dict(consumed) == dict(expected_source_permit),
        "confirmation source permit evidence changed",
    )

    source_root = Path(os.path.abspath(paths.source_root))
    processing = Path(os.path.abspath(paths.processing_code_root))
    python = Path(os.path.abspath(paths.python_executable))
    _require(
        not os.path.lexists(source_root),
        "canonical confirmation source root is not fresh",
    )
    _require(
        source_root.parent.is_dir()
        and not source_root.parent.is_symlink()
        and source_root.parent.resolve() == source_root.parent,
        "confirmation source parent is unsafe",
    )
    partial = source_root.parent / f".{source_root.name}.partial.{os.getpid()}"
    _require(not os.path.lexists(partial), "confirmation source partial root exists")
    _require(
        processing.is_dir()
        and not processing.is_symlink()
        and processing.resolve() == processing
        and _git_value(processing, "rev-parse", "HEAD") == PROCESSING_CODE_REVISION
        and _git_value(processing, "rev-parse", "HEAD^{tree}") == PROCESSING_CODE_TREE
        and not _git_value(
            processing, "status", "--porcelain=v1", "--untracked-files=all"
        ),
        "pinned Deform360 processing source changed",
    )
    _require(
        not _git_value(
            processing, "ls-files", "--others", "--ignored", "--exclude-standard"
        ),
        "pinned Deform360 processing source contains ignored files",
    )
    python_identity = _validated_python(python)
    provide = inventory_provider or _default_inventory_provider
    download = snapshot_downloader or _default_snapshot_downloader
    run = command_runner or _default_command_runner
    partial_identity: tuple[int, int] | None = None
    published_identity: tuple[int, int] | None = None
    try:
        partial.mkdir(mode=0o700)
        partial_state = os.lstat(partial)
        partial_identity = (partial_state.st_dev, partial_state.st_ino)
        download_root = partial / "download"
        aligned_root = partial / "aligned"
        manifests = partial / "manifests"
        cache = partial / "provider-cache"
        for directory in (download_root, aligned_root, manifests, cache):
            directory.mkdir(mode=0o700)

        legacy_by_case: dict[str, tuple[dict[str, Any], ...]] = {}
        processing_by_case: dict[str, tuple[dict[str, Any], ...]] = {}
        allow_patterns: list[str] = []
        for case in CONFIRMATION_SOURCE_CASES:
            items = provide(
                repo_id=HF_REPO_ID,
                path_in_repo=f"raw/{case.object_id}",
                recursive=True,
                expand=True,
                repo_type=HF_REPO_TYPE,
                revision=HF_DATASET_REVISION,
            )
            legacy, processing_records = _selected_records(items, case)
            legacy_by_case[case.case_name] = legacy
            processing_by_case[case.case_name] = processing_records
            allow_patterns.extend(record["path"] for record in processing_records)
        _require(
            len(allow_patterns) == 168
            and len(set(allow_patterns)) == 168
            and not any("tactile" in path for path in allow_patterns),
            "confirmation processing allowlist changed",
        )
        downloaded = Path(
            download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                revision=HF_DATASET_REVISION,
                allow_patterns=sorted(allow_patterns),
                local_dir=str(download_root),
                local_dir_use_symlinks=False,
            )
        ).resolve()
        _require(downloaded == download_root, "provider changed download root")
        cache_sidecar = download_root / ".cache"
        if cache_sidecar.exists():
            shutil.rmtree(cache_sidecar)
        processing_records_flat = tuple(
            record
            for case in CONFIRMATION_SOURCE_CASES
            for record in processing_by_case[case.case_name]
        )
        downloaded_records = _content_records(download_root, processing_records_flat)

        environment = {
            "HOME": "/home/florianpfaff",
            "USER": "florianpfaff",
            "LOGNAME": "florianpfaff",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "TMPDIR": str(cache),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": str(processing),
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
        }
        aligned_by_case: dict[str, tuple[dict[str, Any], ...]] = {}
        for case in CONFIRMATION_SOURCE_CASES:
            object_output = aligned_root / case.object_id
            object_output.mkdir(mode=0o700)
            run(
                (
                    str(python),
                    "-I",
                    "-B",
                    "-c",
                    _UNDISTORT_SCRIPT,
                    str(processing),
                    str(download_root / "raw" / case.object_id),
                    str(object_output),
                    json.dumps(list(case.cameras), separators=(",", ":")),
                ),
                cwd=partial,
                env=environment,
            )
            local_episode = object_output / "episode_0000"
            canonical_episode = object_output / f"episode_{case.episode_id:04d}"
            _require(
                local_episode.is_dir() and not os.path.lexists(canonical_episode),
                f"isolated aligned episode was not created: {case.case_name}",
            )
            local_episode.rename(canonical_episode)
            run(
                (
                    str(python),
                    "-I",
                    "-B",
                    "-c",
                    _ROBOT_SCRIPT,
                    str(processing),
                    str(object_output),
                    str(case.episode_id),
                    "true" if case.bimanual else "false",
                    json.dumps(list(case.cameras), separators=(",", ":")),
                ),
                cwd=partial,
                env=environment,
            )
            aligned_by_case[case.case_name] = _validate_episode_layout(
                canonical_episode, case=case
            )
        shutil.rmtree(cache)

        cases: list[dict[str, Any]] = []
        downloaded_index = {record["path"]: record for record in downloaded_records}
        for case in CONFIRMATION_SOURCE_CASES:
            remote_processing = processing_by_case[case.case_name]
            cases.append(
                {
                    "case_name": case.case_name,
                    "object_id": case.object_id,
                    "episode_id": case.episode_id,
                    "bimanual": case.bimanual,
                    "cameras": list(case.cameras),
                    "legacy_remote_inventory": {
                        "records": list(legacy_by_case[case.case_name]),
                        "records_sha256": case.remote_inventory_sha256,
                        "record_count": case.remote_file_count,
                        "total_size_bytes": case.remote_total_bytes,
                        "tactile_records_metadata_only": True,
                        "tactile_payload_downloaded_or_read": False,
                    },
                    "processing_remote_inventory": {
                        "records": list(remote_processing),
                        "replacement_schema_records_sha256": (
                            case.processing_inventory_sha256
                        ),
                        "record_count": 28,
                        "total_size_bytes": case.processing_total_bytes,
                    },
                    "downloaded_raw_records": [
                        downloaded_index[record["path"]] for record in remote_processing
                    ],
                    "aligned_episode_relative_path": (
                        f"aligned/{case.object_id}/episode_{case.episode_id:04d}"
                    ),
                    "aligned_records": list(aligned_by_case[case.case_name]),
                }
            )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": COHORT_MANIFEST_KIND,
            "protocol_id": PROTOCOL_ID,
            "status": COHORT_STATUS,
            "role": "confirmation",
            "source_root": str(source_root),
            "ordered_case_names": list(CONFIRMATION_SOURCE_CASE_NAMES),
            "confirmation_lock_and_capability": dict(consumed),
            "source_contract": deepcopy(CONFIRMATION_SOURCE_CONTRACT),
            "source_contract_sha256": confirmation_source_contract_sha256(),
            "repository": {
                "repo_id": HF_REPO_ID,
                "repo_type": HF_REPO_TYPE,
                "revision": HF_DATASET_REVISION,
            },
            "processing": {
                "code_root": str(processing),
                "code_revision": PROCESSING_CODE_REVISION,
                "code_tree": PROCESSING_CODE_TREE,
                **python_identity,
                "isolated_subset_episode_index": 0,
                "undistortion_tolerance_us": UNDISTORTION_TOLERANCE_US,
                "rebuild_timeline": False,
                "robot_seed": 0,
                "overwrite": True,
                "plot": False,
            },
            "cases": cases,
            "information_boundary": {
                "calibration_go_recursively_validated_before_provider_touch": True,
                "process_local_single_use_capability_consumed": True,
                "full_six_case_cohort_published_atomically": True,
                "partial_cohort_never_published": True,
                "tactile_metadata_used_only_for_legacy_inventory": True,
                "tactile_payload_downloaded_or_read": False,
                "shared_aligned_dataset_mutated": False,
            },
            "artifact_sha256": "",
        }
        manifest["artifact_sha256"] = _artifact_sha256(manifest)
        manifest_path = manifests / "aligned-source-cohort.json"
        _write_json(manifest_path, manifest)
        _validate_source_directory_layout(partial)
        _freeze_tree(partial)
        _validate_confirmation_source_cohort_manifest_at_root(
            manifest_path,
            actual_source_root=partial,
            declared_source_root=source_root,
            expected_source_permit=expected_source_permit,
            verify_content=True,
        )
        partial_state = os.lstat(partial)
        published_identity = (partial_state.st_dev, partial_state.st_ino)
        os.rename(partial, source_root)
        published_state = os.lstat(source_root)
        _require(
            (published_state.st_dev, published_state.st_ino) == published_identity,
            "confirmation source root changed while publishing",
        )
        parent_fd = os.open(source_root.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        published = source_root / "manifests" / "aligned-source-cohort.json"
        validate_confirmation_source_cohort_manifest(
            published,
            expected_source_permit=expected_source_permit,
            verify_content=True,
        )
        return published
    except BaseException:
        if os.path.lexists(partial):
            _require(
                partial_identity is not None,
                "refusing to clean up a partial root not created by this operation",
            )
            _remove_owned_tree(
                partial,
                expected_identity=partial_identity,
            )
        elif published_identity is not None and os.path.lexists(source_root):
            _remove_owned_tree(source_root, expected_identity=published_identity)
        raise


def _validate_confirmation_source_cohort_manifest_at_root(
    manifest_path: str | os.PathLike[str],
    *,
    actual_source_root: Path,
    declared_source_root: Path,
    expected_source_permit: Mapping[str, Any],
    verify_content: bool = True,
) -> dict[str, Any]:
    path = Path(os.path.abspath(os.fspath(manifest_path)))
    source_root = Path(os.path.abspath(actual_source_root))
    declared_root = Path(os.path.abspath(declared_source_root))
    _require(
        path == source_root / "manifests" / "aligned-source-cohort.json"
        and declared_root.name == "confirmation-source"
        and declared_root.resolve(strict=False) == declared_root,
        "confirmation source manifest path is non-canonical",
    )
    _require(
        source_root.is_dir()
        and not source_root.is_symlink()
        and source_root.resolve() == source_root
        and stat.S_IMODE(os.lstat(source_root).st_mode) == 0o500,
        "confirmation source root is not sealed",
    )
    _require(
        stat.S_IMODE(os.lstat(path).st_mode) == 0o400,
        "confirmation source manifest is not sealed",
    )
    artifact = _read_json_object(path, label="confirmation source manifest")
    _require(
        isinstance(artifact, dict)
        and set(artifact)
        == {
            "schema_version",
            "artifact_kind",
            "protocol_id",
            "status",
            "role",
            "source_root",
            "ordered_case_names",
            "confirmation_lock_and_capability",
            "source_contract",
            "source_contract_sha256",
            "repository",
            "processing",
            "cases",
            "information_boundary",
            "artifact_sha256",
        },
        "confirmation source manifest fields changed",
    )
    _require(
        artifact.get("schema_version") == SCHEMA_VERSION
        and artifact.get("artifact_kind") == COHORT_MANIFEST_KIND
        and artifact.get("protocol_id") == PROTOCOL_ID
        and artifact.get("status") == COHORT_STATUS
        and artifact.get("role") == "confirmation"
        and artifact.get("source_root") == str(declared_root)
        and artifact.get("ordered_case_names") == list(CONFIRMATION_SOURCE_CASE_NAMES)
        and artifact.get("confirmation_lock_and_capability")
        == dict(expected_source_permit)
        and artifact.get("source_contract") == CONFIRMATION_SOURCE_CONTRACT
        and artifact.get("source_contract_sha256")
        == confirmation_source_contract_sha256()
        and artifact.get("artifact_sha256") == _artifact_sha256(artifact),
        "confirmation source manifest identity or self-hash changed",
    )
    _require(
        artifact.get("repository")
        == {
            "repo_id": HF_REPO_ID,
            "repo_type": HF_REPO_TYPE,
            "revision": HF_DATASET_REVISION,
        },
        "confirmation source repository changed",
    )
    processing = artifact.get("processing")
    _require(
        isinstance(processing, Mapping)
        and set(processing)
        == {
            "code_root",
            "code_revision",
            "code_tree",
            "python_executable",
            "python_link_target",
            "python_resolved",
            "python_resolved_sha256",
            "isolated_subset_episode_index",
            "undistortion_tolerance_us",
            "rebuild_timeline",
            "robot_seed",
            "overwrite",
            "plot",
        }
        and processing.get("code_revision") == PROCESSING_CODE_REVISION
        and processing.get("code_tree") == PROCESSING_CODE_TREE
        and processing.get("isolated_subset_episode_index") == 0
        and processing.get("undistortion_tolerance_us") == UNDISTORTION_TOLERANCE_US
        and processing.get("rebuild_timeline") is False
        and processing.get("robot_seed") == 0
        and processing.get("overwrite") is True
        and processing.get("plot") is False,
        "confirmation source processing contract changed",
    )
    processing_root = Path(str(processing["code_root"]))
    _require(
        processing_root.is_dir()
        and not processing_root.is_symlink()
        and processing_root.resolve() == processing_root
        and _git_value(processing_root, "rev-parse", "HEAD") == PROCESSING_CODE_REVISION
        and _git_value(processing_root, "rev-parse", "HEAD^{tree}")
        == PROCESSING_CODE_TREE
        and not _git_value(
            processing_root, "status", "--porcelain=v1", "--untracked-files=all"
        )
        and not _git_value(
            processing_root,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
        ),
        "live confirmation processing source changed",
    )
    _require(
        _validated_python(Path(str(processing["python_executable"])))
        == {
            key: processing[key]
            for key in (
                "python_executable",
                "python_link_target",
                "python_resolved",
                "python_resolved_sha256",
            )
        },
        "live confirmation processing Python changed",
    )
    _require(
        artifact.get("information_boundary")
        == {
            "calibration_go_recursively_validated_before_provider_touch": True,
            "process_local_single_use_capability_consumed": True,
            "full_six_case_cohort_published_atomically": True,
            "partial_cohort_never_published": True,
            "tactile_metadata_used_only_for_legacy_inventory": True,
            "tactile_payload_downloaded_or_read": False,
            "shared_aligned_dataset_mutated": False,
        },
        "confirmation source information boundary changed",
    )
    cases = artifact.get("cases")
    _require(
        isinstance(cases, list)
        and [case.get("case_name") for case in cases if isinstance(case, Mapping)]
        == list(CONFIRMATION_SOURCE_CASE_NAMES),
        "confirmation source manifest cohort changed",
    )
    _validate_source_directory_layout(source_root)
    case_specs = {case.case_name: case for case in CONFIRMATION_SOURCE_CASES}
    for record in cases:
        _require(
            isinstance(record, Mapping)
            and set(record)
            == {
                "case_name",
                "object_id",
                "episode_id",
                "bimanual",
                "cameras",
                "legacy_remote_inventory",
                "processing_remote_inventory",
                "downloaded_raw_records",
                "aligned_episode_relative_path",
                "aligned_records",
            },
            "source case record is invalid",
        )
        case = case_specs[str(record["case_name"])]
        _require(
            record.get("object_id") == case.object_id
            and record.get("episode_id") == case.episode_id
            and record.get("bimanual") is case.bimanual
            and record.get("cameras") == list(case.cameras),
            f"confirmation source case identity changed: {case.case_name}",
        )
        legacy = record.get("legacy_remote_inventory")
        processing_remote = record.get("processing_remote_inventory")
        _require(
            isinstance(legacy, Mapping)
            and set(legacy)
            == {
                "records",
                "records_sha256",
                "record_count",
                "total_size_bytes",
                "tactile_records_metadata_only",
                "tactile_payload_downloaded_or_read",
            }
            and isinstance(processing_remote, Mapping)
            and set(processing_remote)
            == {
                "records",
                "replacement_schema_records_sha256",
                "record_count",
                "total_size_bytes",
            },
            f"remote inventory fields changed: {case.case_name}",
        )
        legacy_records = _validate_remote_records(
            legacy.get("records"),
            case=case,
            expected_count=case.remote_file_count,
            processing_only=False,
        )
        processing_records = _validate_remote_records(
            processing_remote.get("records"),
            case=case,
            expected_count=28,
            processing_only=True,
        )
        downloaded_records = _validate_content_record_metadata(
            record.get("downloaded_raw_records"),
            label=f"downloaded raw {case.case_name}",
        )
        aligned_records = _validate_content_record_metadata(
            record.get("aligned_records"),
            label=f"aligned episode {case.case_name}",
        )
        _require(
            len(downloaded_records) == 28
            and legacy.get("record_count") == case.remote_file_count
            and legacy.get("total_size_bytes") == case.remote_total_bytes
            and legacy.get("records_sha256") == case.remote_inventory_sha256
            and _digest(legacy_records) == case.remote_inventory_sha256
            and legacy.get("tactile_records_metadata_only") is True
            and legacy.get("tactile_payload_downloaded_or_read") is False,
            f"legacy remote inventory changed: {case.case_name}",
        )
        replacement_records = [
            {
                "path": item["path"],
                "size_bytes": item["size"],
                "blob_id": item["blob_id"],
                "lfs_sha256": item["sha256"],
            }
            for item in processing_records
        ]
        _require(
            processing_remote.get("record_count") == 28
            and processing_remote.get("total_size_bytes") == case.processing_total_bytes
            and processing_remote.get("replacement_schema_records_sha256")
            == case.processing_inventory_sha256
            and _digest(replacement_records) == case.processing_inventory_sha256,
            f"processing remote inventory changed: {case.case_name}",
        )
        aligned_relative = record.get("aligned_episode_relative_path")
        _require(
            aligned_relative
            == f"aligned/{case.object_id}/episode_{case.episode_id:04d}",
            f"aligned path changed: {case.case_name}",
        )
        if verify_content:
            _require(
                [item.get("path") for item in downloaded_records]
                == [item.get("path") for item in processing_records],
                f"downloaded raw ordering changed: {case.case_name}",
            )
            raw_expected = {item["path"]: item for item in downloaded_records}
            for remote_record in processing_records:
                relative = remote_record["path"]
                raw_record = raw_expected.get(relative)
                raw_path = source_root / "download" / relative
                observed_sha256 = _sha256_file(raw_path)
                _require(
                    isinstance(raw_record, Mapping)
                    and set(raw_record) == {"path", "size_bytes", "sha256"}
                    and raw_record
                    == {
                        "path": relative,
                        "size_bytes": raw_path.stat().st_size,
                        "sha256": observed_sha256,
                    },
                    f"downloaded raw byte changed: {relative}",
                )
                _require(
                    raw_path.stat().st_size == remote_record["size"],
                    f"downloaded raw size differs from remote: {relative}",
                )
                if remote_record["sha256"] is not None:
                    _require(
                        observed_sha256 == remote_record["sha256"],
                        f"downloaded LFS byte differs from remote: {relative}",
                    )
                else:
                    _require(
                        _git_blob_sha1_file(raw_path) == remote_record["blob_id"],
                        f"downloaded Git blob differs from remote: {relative}",
                    )
            aligned_path = source_root / str(aligned_relative)
            observed_aligned = _validate_episode_layout(aligned_path, case=case)
            _require(
                list(observed_aligned) == aligned_records,
                f"aligned source byte changed: {case.case_name}",
            )
    if verify_content:
        for entry in source_root.rglob("*"):
            observed = os.lstat(entry)
            _require(
                not stat.S_ISLNK(observed.st_mode)
                and (
                    (
                        stat.S_ISDIR(observed.st_mode)
                        and stat.S_IMODE(observed.st_mode) == 0o500
                    )
                    or (
                        stat.S_ISREG(observed.st_mode)
                        and observed.st_nlink == 1
                        and stat.S_IMODE(observed.st_mode) == 0o400
                    )
                ),
                f"confirmation source tree is not recursively sealed: {entry}",
            )
    return artifact


def validate_confirmation_source_cohort_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_source_permit: Mapping[str, Any],
    verify_content: bool = True,
) -> dict[str, Any]:
    """Validate the canonical, atomically published confirmation cohort."""

    path = Path(os.path.abspath(os.fspath(manifest_path)))
    _require(
        path.name == "aligned-source-cohort.json"
        and path.parent.name == "manifests"
        and path.parent.parent.name == "confirmation-source",
        "confirmation source manifest path is non-canonical",
    )
    source_root = path.parent.parent
    return _validate_confirmation_source_cohort_manifest_at_root(
        path,
        actual_source_root=source_root,
        declared_source_root=source_root,
        expected_source_permit=expected_source_permit,
        verify_content=verify_content,
    )


__all__ = [
    "COHORT_MANIFEST_KIND",
    "COHORT_STATUS",
    "CONFIRMATION_SOURCE_CASES",
    "CONFIRMATION_SOURCE_CASE_NAMES",
    "CONFIRMATION_SOURCE_CONTRACT",
    "ConfirmationSourcePaths",
    "SOURCE_OPERATION",
    "confirmation_source_contract_sha256",
    "materialize_confirmation_source_cohort",
    "validate_confirmation_source_cohort_manifest",
]
