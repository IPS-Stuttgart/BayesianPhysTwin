"""Exact selected-episode file planning for Deform360 Stage-1 calibration."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Protocol

DATASET_REPOSITORY = "brownu/deform360"
CAMERA_RE = re.compile(r"^brics-odroid-\d+_cam\d+$")
TACTILE_RE = re.compile(r"^brics-odroid_tactile[^/]+$")
CALIBRATION_FILENAMES = frozenset({"intrinsics.npy", "extrinsics.npy", "dist.npy"})


class HubTreeMember(Protocol):
    """Minimal Hugging Face tree-member surface used by the planner."""

    path: str
    size: int | None
    lfs: object | None


class HubApi(Protocol):
    """Minimal immutable dataset-listing surface used by the planner."""

    def repo_info(self, **arguments: Any) -> Any: ...

    def list_repo_tree(self, **arguments: Any) -> Iterable[HubTreeMember]: ...


@dataclass(frozen=True, order=True)
class HubFile:
    """Portable metadata for one regular file in an immutable Hub tree."""

    path: str
    size: int | None = None
    lfs_oid: str | None = None

    def __post_init__(self) -> None:
        path = canonical_hub_path(self.path)
        size = self.size
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError("Hub file size must be a nonnegative integer or null")
        lfs_oid = self.lfs_oid
        if lfs_oid is not None:
            lfs_oid = str(lfs_oid)
            if lfs_oid.startswith("sha256:"):
                lfs_oid = lfs_oid.removeprefix("sha256:")
            require_sha256(lfs_oid, name="Hub LFS oid")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "lfs_oid", lfs_oid)

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "lfs_oid": self.lfs_oid,
        }


@dataclass(frozen=True)
class UnitPlan:
    """Exact selected-episode access plan for one Stage-0 calibration object."""

    object_id: str
    episode_id: int
    stratum: str
    metadata_path: str
    expected_metadata_sha256: str
    calibration_files: tuple[HubFile, ...]
    camera_recordings: tuple[tuple[str, HubFile, HubFile], ...]
    tactile_recordings: tuple[tuple[str, HubFile, HubFile, HubFile], ...]
    technical_failures: tuple[str, ...]

    @property
    def materialization_paths(self) -> tuple[str, ...]:
        paths = {
            self.metadata_path,
            *(item.path for item in self.calibration_files),
            *(timestamp.path for _, _, timestamp in self.camera_recordings),
        }
        for _, data, timestamp, baseline in self.tactile_recordings:
            paths.update((data.path, timestamp.path, baseline.path))
        return tuple(sorted(paths))

    @property
    def planned_camera_media_paths(self) -> tuple[str, ...]:
        return tuple(sorted(data.path for _, data, _ in self.camera_recordings))

    @property
    def status(self) -> str:
        return "ready" if not self.technical_failures else "technical_failure"

    def to_record(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "episode_id": self.episode_id,
            "stratum": self.stratum,
            "metadata_path": self.metadata_path,
            "expected_metadata_sha256": self.expected_metadata_sha256,
            "status": self.status,
            "technical_failures": list(self.technical_failures),
            "calibration_files": [item.to_record() for item in self.calibration_files],
            "camera_recordings": [
                {
                    "camera": camera,
                    "media": media.to_record(),
                    "timestamp": timestamp.to_record(),
                }
                for camera, media, timestamp in self.camera_recordings
            ],
            "tactile_recordings": [
                {
                    "sensor": sensor,
                    "data": data.to_record(),
                    "timestamp": timestamp.to_record(),
                    "baseline": baseline.to_record(),
                }
                for sensor, data, timestamp, baseline in self.tactile_recordings
            ],
            "materialization_paths": list(self.materialization_paths),
            "planned_camera_media_paths": list(self.planned_camera_media_paths),
        }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_hub_path(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("Hub path must be a nonempty literal string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"Hub path is not canonical and confined: {value!r}")
    return value


def require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _lfs_oid(value: object) -> str | None:
    if isinstance(value, Mapping):
        oid = value.get("oid", value.get("sha256"))
    elif hasattr(value, "oid"):
        oid = value.oid
    elif hasattr(value, "sha256"):
        oid = value.sha256
    else:
        oid = None
    return None if oid is None else str(oid)


def hub_file_from_member(member: HubTreeMember) -> HubFile | None:
    """Convert a Hugging Face RepoFile-like object; folders return ``None``."""

    if not hasattr(member, "size"):
        return None
    size = member.size
    if size is not None:
        size = int(size)
    lfs = member.lfs if hasattr(member, "lfs") else None
    return HubFile(
        path=member.path,
        size=size,
        lfs_oid=_lfs_oid(lfs),
    )


def _pair_recordings(
    files: Sequence[HubFile],
    *,
    data_suffix: str,
    exclude_data_prefixes: Sequence[str] = (),
) -> tuple[tuple[tuple[HubFile, HubFile], ...], tuple[str, ...]]:
    data_by_stem: dict[str, HubFile] = {}
    timestamps_by_stem: dict[str, HubFile] = {}
    for item in files:
        name = PurePosixPath(item.path).name
        suffix = PurePosixPath(name).suffix.lower()
        stem = PurePosixPath(name).stem
        if suffix == data_suffix and not any(
            stem.startswith(prefix) for prefix in exclude_data_prefixes
        ):
            require(stem not in data_by_stem, f"duplicate data stem {stem!r}")
            data_by_stem[stem] = item
        elif suffix == ".txt":
            require(
                stem not in timestamps_by_stem,
                f"duplicate timestamp stem {stem!r}",
            )
            timestamps_by_stem[stem] = item
    data_stems = set(data_by_stem)
    timestamp_stems = set(timestamps_by_stem)
    common = sorted(data_stems & timestamp_stems)
    issues = (
        *(
            f"missing_timestamp:{stem}"
            for stem in sorted(data_stems - timestamp_stems)
        ),
        *(
            f"orphan_timestamp:{stem}"
            for stem in sorted(timestamp_stems - data_stems)
        ),
    )
    pairs = tuple((data_by_stem[stem], timestamps_by_stem[stem]) for stem in common)
    return pairs, tuple(issues)


def build_unit_plan(unit: Any, files: Sequence[HubFile]) -> UnitPlan:
    """Reproduce official sorted-file episode indexing without opening payloads."""

    prefix = f"raw/{unit.object_id}/"
    by_path: dict[str, HubFile] = {}
    for item in files:
        require(
            item.path.startswith(prefix),
            "Hub listing escaped the selected object",
        )
        require(item.path not in by_path, f"duplicate Hub file path: {item.path}")
        by_path[item.path] = item

    failures: list[str] = []
    metadata_path = str(unit.metadata_path)
    if metadata_path not in by_path:
        failures.append("metadata_missing")

    calibration_prefix = f"{prefix}calibration_refined/"
    calibration_files = tuple(
        sorted(
            (
                item
                for item in by_path.values()
                if item.path.startswith(calibration_prefix)
                and PurePosixPath(item.path).name in CALIBRATION_FILENAMES
            ),
            key=lambda item: item.path,
        )
    )
    observed_calibration = {
        PurePosixPath(item.path).name for item in calibration_files
    }
    if observed_calibration != CALIBRATION_FILENAMES:
        failures.append("calibration_files_incomplete")

    streams: dict[str, list[HubFile]] = {}
    for item in by_path.values():
        relative = PurePosixPath(item.path).relative_to(PurePosixPath(prefix))
        if len(relative.parts) < 2:
            continue
        streams.setdefault(relative.parts[0], []).append(item)

    camera_records: list[tuple[str, HubFile, HubFile]] = []
    camera_names = sorted(name for name in streams if CAMERA_RE.fullmatch(name))
    if not camera_names:
        failures.append("camera_streams_missing")
    for camera in camera_names:
        pairs, pairing_issues = _pair_recordings(
            streams[camera],
            data_suffix=".mp4",
        )
        failures.extend(
            f"camera_pairing:{camera}:{issue}" for issue in pairing_issues
        )
        if pairing_issues:
            continue
        if unit.episode_id >= len(pairs):
            failures.append(f"camera_episode_missing:{camera}")
            continue
        data, timestamp = pairs[unit.episode_id]
        camera_records.append((camera, data, timestamp))

    tactile_records: list[tuple[str, HubFile, HubFile, HubFile]] = []
    tactile_names = sorted(name for name in streams if TACTILE_RE.fullmatch(name))
    if not tactile_names:
        failures.append("tactile_streams_missing")
    for sensor in tactile_names:
        pairs, pairing_issues = _pair_recordings(
            streams[sensor],
            data_suffix=".npy",
            exclude_data_prefixes=("median_",),
        )
        failures.extend(
            f"tactile_pairing:{sensor}:{issue}" for issue in pairing_issues
        )
        if pairing_issues:
            continue
        baselines = sorted(
            (
                item
                for item in streams[sensor]
                if PurePosixPath(item.path).name.startswith("median_")
                and PurePosixPath(item.path).suffix.lower() == ".npy"
            ),
            key=lambda item: item.path,
        )
        if len(baselines) != 1:
            failures.append(f"tactile_baseline_count:{sensor}:{len(baselines)}")
            continue
        if unit.episode_id >= len(pairs):
            failures.append(f"tactile_episode_missing:{sensor}")
            continue
        data, timestamp = pairs[unit.episode_id]
        tactile_records.append((sensor, data, timestamp, baselines[0]))

    return UnitPlan(
        object_id=str(unit.object_id),
        episode_id=int(unit.episode_id),
        stratum=str(unit.stratum),
        metadata_path=metadata_path,
        expected_metadata_sha256=str(unit.metadata_sha256),
        calibration_files=calibration_files,
        camera_recordings=tuple(camera_records),
        tactile_recordings=tuple(tactile_records),
        technical_failures=tuple(sorted(set(failures))),
    )


def list_unit_files(
    api: HubApi,
    *,
    object_id: str,
    revision: str,
    token: str | None,
) -> tuple[HubFile, ...]:
    members = api.list_repo_tree(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=revision,
        path_in_repo=f"raw/{object_id}",
        recursive=True,
        expand=True,
        token=token,
    )
    files = [item for member in members if (item := hub_file_from_member(member))]
    return tuple(sorted(files, key=lambda item: item.path))


__all__ = [
    "CALIBRATION_FILENAMES",
    "CAMERA_RE",
    "DATASET_REPOSITORY",
    "HubApi",
    "HubFile",
    "HubTreeMember",
    "TACTILE_RE",
    "UnitPlan",
    "build_unit_plan",
    "canonical_hub_path",
    "hub_file_from_member",
    "list_unit_files",
    "require",
    "require_sha256",
]
