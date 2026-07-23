"""Object-scoped, metadata-only-bound download for H2 Deform360 confirmation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from .deform360_adaptive_covariance_confirmation_external_runtime import (
    DOWNLOAD_ARTIFACT_KIND,
)
from .deform360_adaptive_covariance_confirmation_lock import (
    DATASET_REPOSITORY,
    DATASET_REVISION,
    PROTOCOL_ID,
    load_confirmation_cohort_lock,
)


ListRepoTree = Callable[..., Any]
HubDownload = Callable[..., str]
RemoteInventory = Mapping[str, Sequence[Mapping[str, Any]]]
_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _regular_file_record(path: Path) -> dict[str, Any]:
    """Hash one immutable, single-link regular file without following links."""

    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1,
        f"downloaded path is not a single-link regular file: {path}",
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    try:
        opened = os.fstat(descriptor)
        _require(
            _stable_file_state(opened) == _stable_file_state(observed),
            f"downloaded file changed while opening: {path}",
        )
        git_blob.update(f"blob {opened.st_size}\0".encode("ascii"))
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            git_blob.update(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"downloaded file changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    return {
        "size_bytes": opened.st_size,
        "sha256": digest.hexdigest(),
        "git_blob_id": git_blob.hexdigest(),
    }


def _read_regular_bytes(path: Path) -> bytes:
    observed = os.lstat(path)
    _require(
        stat.S_ISREG(observed.st_mode)
        and not stat.S_ISLNK(observed.st_mode)
        and observed.st_nlink == 1,
        f"downloaded path is not a single-link regular file: {path}",
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
            f"downloaded file changed while opening: {path}",
        )
        while block := os.read(descriptor, 1024 * 1024):
            payload.extend(block)
        after = os.fstat(descriptor)
        current = os.lstat(path)
        _require(
            _stable_file_state(after)
            == _stable_file_state(opened)
            == _stable_file_state(current),
            f"downloaded file changed while reading: {path}",
        )
    finally:
        os.close(descriptor)
    return bytes(payload)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ConfirmationDownloadPlan:
    repository: str
    revision: str
    object_ids: tuple[str, ...]
    selected_episodes_by_object: tuple[tuple[str, tuple[int, ...]], ...]
    implementation_commit_h1: str
    cohort_lock_commit_h2: str
    cohort_lock_artifact_sha256: str


def confirmation_download_plan(
    lock_path: str | Path,
    h2_commit: str,
    *,
    expected_h1: str | None = None,
) -> ConfirmationDownloadPlan:
    lock = load_confirmation_cohort_lock(
        lock_path,
        expected_implementation_commit_h1=expected_h1,
    )
    h1 = lock["two_commit_freeze"]["implementation_commit_h1"]
    _require(
        isinstance(h2_commit, str)
        and len(h2_commit) == 40
        and h2_commit != h1
        and all(character in "0123456789abcdef" for character in h2_commit),
        "H2 commit is invalid",
    )
    rows: list[tuple[str, tuple[int, ...]]] = []
    for records in lock["cohort"].values():
        for record in records:
            rows.append(
                (
                    record["object_id"],
                    tuple(int(episode["episode_id"]) for episode in record["episodes"]),
                )
            )
    _require(
        len(rows) == 17
        and len({object_id for object_id, _episodes in rows}) == 17
        and all(len(episodes) == 2 for _object_id, episodes in rows),
        "H2 download plan is incomplete",
    )
    return ConfirmationDownloadPlan(
        repository=DATASET_REPOSITORY,
        revision=DATASET_REVISION,
        object_ids=tuple(object_id for object_id, _episodes in rows),
        selected_episodes_by_object=tuple(rows),
        implementation_commit_h1=h1,
        cohort_lock_commit_h2=h2_commit,
        cohort_lock_artifact_sha256=lock["artifact_sha256"],
    )


def validate_confirmation_download_root(
    output_root: str | Path,
    *,
    plan: ConfirmationDownloadPlan,
    require_complete: bool,
) -> None:
    root = Path(output_root).absolute()
    if root.exists():
        _require(
            root.is_dir()
            and not root.is_symlink()
            and root.resolve(strict=True) == root,
            "confirmation download root is noncanonical",
        )
    raw = root / "raw"
    if not raw.exists():
        _require(not require_complete, "confirmation raw download root is missing")
        return
    _require(
        raw.is_dir() and not raw.is_symlink() and raw.resolve(strict=True) == raw,
        "raw download root is invalid",
    )
    present: set[str] = set()
    invalid: list[str] = []
    for path in raw.iterdir():
        observed = os.lstat(path)
        if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            present.add(path.name)
        else:
            invalid.append(path.name)
    expected = set(plan.object_ids)
    _require(
        not invalid and not present - expected,
        "unlocked objects or non-object entries exist in download root: "
        f"{sorted((*invalid, *(present - expected)))}",
    )
    if require_complete:
        _require(
            present == expected,
            f"locked objects are missing: {sorted(expected - present)}",
        )


def _normalize_remote_inventory(
    plan: ConfirmationDownloadPlan,
    remote_inventory_by_object: RemoteInventory,
) -> dict[str, tuple[dict[str, Any], ...]]:
    _require(
        isinstance(remote_inventory_by_object, Mapping),
        "pinned remote inventory is required",
    )
    expected_objects = set(plan.object_ids)
    _require(
        set(remote_inventory_by_object) == expected_objects,
        "pinned remote inventory object panel changed",
    )
    normalized: dict[str, tuple[dict[str, Any], ...]] = {}
    for object_id in plan.object_ids:
        raw_records = remote_inventory_by_object[object_id]
        _require(
            isinstance(raw_records, Sequence)
            and not isinstance(raw_records, (str, bytes))
            and bool(raw_records),
            f"pinned remote inventory is empty: {object_id}",
        )
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_records:
            _require(
                isinstance(item, Mapping)
                and set(item) == {"path", "blob_id", "lfs_sha256"},
                f"pinned remote inventory record changed: {object_id}",
            )
            path = item.get("path")
            blob_id = item.get("blob_id")
            lfs_sha256 = item.get("lfs_sha256")
            _require(
                isinstance(path, str)
                and isinstance(blob_id, str)
                and len(blob_id) == 40
                and all(character in "0123456789abcdef" for character in blob_id),
                f"pinned remote inventory record is invalid: {object_id}",
            )
            _require(
                lfs_sha256 is None
                or (
                    isinstance(lfs_sha256, str)
                    and len(lfs_sha256) == 64
                    and all(character in "0123456789abcdef" for character in lfs_sha256)
                ),
                f"pinned remote LFS identity is invalid: {object_id}",
            )
            candidate = PurePosixPath(path)
            _require(
                not candidate.is_absolute()
                and ".." not in candidate.parts
                and "\\" not in path
                and len(candidate.parts) >= 3
                and candidate.parts[:2] == ("raw", object_id),
                f"pinned remote path escaped its object: {object_id}",
            )
            _require(
                candidate.suffix.lower() not in _AUDIO_SUFFIXES,
                f"audio is forbidden in pinned remote inventory: {path}",
            )
            _require(path not in seen, f"duplicate pinned remote path: {path}")
            seen.add(path)
            records.append(
                {
                    "path": path,
                    "blob_id": blob_id,
                    "lfs_sha256": lfs_sha256,
                }
            )
        records.sort(key=lambda record: record["path"])
        _require(
            f"raw/{object_id}/metadata.json" in seen,
            f"pinned remote metadata is absent: {object_id}",
        )
        normalized[object_id] = tuple(records)
    return normalized


def _scan_object_tree(
    object_root: Path,
    *,
    download_root: Path,
) -> tuple[dict[str, Path], set[str]]:
    observed_root = os.lstat(object_root)
    _require(
        stat.S_ISDIR(observed_root.st_mode) and not stat.S_ISLNK(observed_root.st_mode),
        f"downloaded object root is invalid: {object_root}",
    )
    files: dict[str, Path] = {}
    directories: set[str] = set()
    pending = [object_root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                observed = entry.stat(follow_symlinks=False)
                relative = path.relative_to(download_root).as_posix()
                _require(
                    not stat.S_ISLNK(observed.st_mode),
                    f"downloaded object tree contains a symlink: {relative}",
                )
                if stat.S_ISDIR(observed.st_mode):
                    directories.add(relative)
                    pending.append(path)
                    continue
                _require(
                    stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1,
                    f"downloaded object tree contains a special or linked file: "
                    f"{relative}",
                )
                _require(
                    PurePosixPath(relative).suffix.lower() not in _AUDIO_SUFFIXES,
                    f"downloaded object tree contains forbidden audio: {relative}",
                )
                files[relative] = path
    return files, directories


def _metadata_record(
    root: Path,
    object_id: str,
    selected_episode_ids: tuple[int, ...],
    remote_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    object_root = root / "raw" / object_id
    metadata_path = object_root / "metadata.json"
    _require(
        os.path.lexists(metadata_path),
        f"metadata is missing: {object_id}",
    )
    local_files, local_directories = _scan_object_tree(
        object_root,
        download_root=root,
    )
    expected = {str(record["path"]): record for record in remote_records}
    observed_paths = set(local_files)
    missing = sorted(set(expected) - observed_paths)
    extras = sorted(observed_paths - set(expected))
    _require(not missing, f"downloaded files are missing: {object_id}: {missing}")
    _require(not extras, f"downloaded object has extra files: {object_id}: {extras}")
    expected_directories: set[str] = set()
    object_prefix = PurePosixPath("raw") / object_id
    for relative in expected:
        parent = PurePosixPath(relative).parent
        while parent != object_prefix:
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    _require(
        local_directories <= expected_directories,
        f"downloaded object has extra directories: {object_id}: "
        f"{sorted(local_directories - expected_directories)}",
    )

    file_records: list[dict[str, Any]] = []
    for relative in sorted(expected):
        content = _regular_file_record(local_files[relative])
        remote = expected[relative]
        if remote["lfs_sha256"] is not None:
            _require(
                content["sha256"] == remote["lfs_sha256"],
                f"downloaded LFS content differs from pinned remote: {relative}",
            )
        else:
            _require(
                content["git_blob_id"] == remote["blob_id"],
                f"downloaded Git blob differs from pinned remote: {relative}",
            )
        file_records.append(
            {
                "path": relative,
                "remote_blob_id": remote["blob_id"],
                "remote_lfs_sha256": remote["lfs_sha256"],
                "size_bytes": content["size_bytes"],
                "sha256": content["sha256"],
            }
        )
    metadata_bytes = _read_regular_bytes(local_files[f"raw/{object_id}/metadata.json"])
    metadata_content_sha256 = hashlib.sha256(metadata_bytes).hexdigest()
    _require(
        metadata_content_sha256
        == next(
            record["sha256"]
            for record in file_records
            if record["path"] == f"raw/{object_id}/metadata.json"
        ),
        f"metadata changed after content binding: {object_id}",
    )
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    _require(isinstance(metadata, Mapping), f"metadata is invalid: {object_id}")
    released_label = metadata.get("object")
    _require(
        isinstance(released_label, str) and bool(released_label.strip()),
        f"released metadata label is empty: {object_id}",
    )
    sequences = metadata.get("sequences")
    _require(isinstance(sequences, Mapping), f"sequences are missing: {object_id}")
    _require(
        set(sequences) == {str(index) for index in range(10)},
        f"episode inventory changed: {object_id}",
    )
    _require(
        all(str(episode) in sequences for episode in selected_episode_ids),
        f"selected episode is missing: {object_id}",
    )
    remote_binding = [
        {
            "path": record["path"],
            "blob_id": record["blob_id"],
            "lfs_sha256": record["lfs_sha256"],
        }
        for record in remote_records
    ]
    return {
        "object_id": object_id,
        "selected_episode_ids": list(selected_episode_ids),
        "released_metadata_object_label": released_label,
        "directory_id_is_identity": True,
        "file_count": len(file_records),
        "total_bytes": sum(record["size_bytes"] for record in file_records),
        "metadata_sha256": metadata_content_sha256,
        "remote_inventory_sha256": _value_sha256(remote_binding),
        "content_inventory_sha256": _value_sha256(file_records),
        "files": file_records,
    }


def build_confirmation_download_manifest(
    output_root: str | Path,
    *,
    plan: ConfirmationDownloadPlan,
    remote_inventory_by_object: RemoteInventory,
) -> dict[str, Any]:
    root = Path(output_root).absolute()
    validate_confirmation_download_root(root, plan=plan, require_complete=True)
    remote_inventory = _normalize_remote_inventory(
        plan,
        remote_inventory_by_object,
    )
    rows = [
        _metadata_record(
            root,
            object_id,
            episodes,
            remote_inventory[object_id],
        )
        for object_id, episodes in plan.selected_episodes_by_object
    ]
    payload: dict[str, Any] = {
        "schema_version": 2,
        "artifact_kind": DOWNLOAD_ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "dataset_repository": plan.repository,
        "dataset_revision": plan.revision,
        "implementation_commit_h1": plan.implementation_commit_h1,
        "cohort_lock_commit_h2": plan.cohort_lock_commit_h2,
        "cohort_lock_artifact_sha256": plan.cohort_lock_artifact_sha256,
        "audio_included": False,
        "object_count": len(rows),
        "objects": rows,
        "information_boundary": {
            "locked_object_directories_only": True,
            "directory_id_used_as_identity": True,
            "released_metadata_label_used_for_identity": False,
            "target_or_outcome_opened": False,
            "metric_computed": False,
            "pinned_remote_inventory_captured": True,
            "every_non_audio_file_content_hashed": True,
        },
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def download_confirmation_panel_by_object(
    lock_path: str | Path,
    h2_commit: str,
    output_root: str | Path,
    *,
    max_workers: int,
    object_delay_seconds: float,
    list_repo_tree: ListRepoTree,
    hub_download: HubDownload,
    expected_h1: str | None = None,
) -> dict[str, Any]:
    """Download only exact H2 object subtrees at the pinned dataset revision."""

    _require(max_workers >= 1, "download workers must be positive")
    _require(object_delay_seconds >= 0.0, "object delay must be non-negative")
    plan = confirmation_download_plan(
        lock_path,
        h2_commit,
        expected_h1=expected_h1,
    )
    root = Path(output_root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    _require(
        not root.is_symlink() and root.resolve(strict=True) == root,
        "download root is noncanonical",
    )
    validate_confirmation_download_root(root, plan=plan, require_complete=False)
    remote_inventory_by_object: dict[str, tuple[dict[str, Any], ...]] = {}
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
        remote_records: list[dict[str, Any]] = []
        for entry in entries:
            filename = str(getattr(entry, "path", ""))
            candidate = PurePosixPath(filename)
            _require(
                bool(filename)
                and not candidate.is_absolute()
                and ".." not in candidate.parts
                and "\\" not in filename
                and candidate.parts[:2] == ("raw", object_id),
                f"object listing escaped its locked subtree: {object_id}",
            )
            blob_id = getattr(entry, "blob_id", None)
            if blob_id is None:
                _require(
                    getattr(entry, "tree_id", None) is not None,
                    f"remote listing entry is neither a file nor directory: "
                    f"{object_id}",
                )
                continue
            if candidate.suffix.lower() not in _AUDIO_SUFFIXES:
                lfs = getattr(entry, "lfs", None)
                if isinstance(lfs, Mapping):
                    lfs_sha256 = lfs.get("sha256")
                else:
                    lfs_sha256 = getattr(lfs, "sha256", None)
                remote_records.append(
                    {
                        "path": filename,
                        "blob_id": str(blob_id),
                        "lfs_sha256": lfs_sha256,
                    }
                )
        normalized = _normalize_remote_inventory(
            ConfirmationDownloadPlan(
                repository=plan.repository,
                revision=plan.revision,
                object_ids=(object_id,),
                selected_episodes_by_object=(
                    (
                        object_id,
                        dict(plan.selected_episodes_by_object)[object_id],
                    ),
                ),
                implementation_commit_h1=plan.implementation_commit_h1,
                cohort_lock_commit_h2=plan.cohort_lock_commit_h2,
                cohort_lock_artifact_sha256=plan.cohort_lock_artifact_sha256,
            ),
            {object_id: remote_records},
        )[object_id]
        remote_inventory_by_object[object_id] = normalized
        files = [record["path"] for record in normalized]
        _require(files, f"locked object subtree is empty: {object_id}")
        _require(
            f"raw/{object_id}/metadata.json" in files,
            f"locked object metadata is absent: {object_id}",
        )
        _require(all(path.startswith(prefix) for path in files), "invalid file prefix")

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
    return build_confirmation_download_manifest(
        root,
        plan=plan,
        remote_inventory_by_object=remote_inventory_by_object,
    )


def write_confirmation_download_manifest(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not destination.exists() and not destination.is_symlink(),
        "download manifest already exists",
    )
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise ValueError("download manifest already exists") from error
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "ConfirmationDownloadPlan",
    "RemoteInventory",
    "build_confirmation_download_manifest",
    "confirmation_download_plan",
    "download_confirmation_panel_by_object",
    "validate_confirmation_download_root",
    "write_confirmation_download_manifest",
]
