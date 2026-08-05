# ruff: noqa: F403, F405
"""Internal implementation slice for Deform360 calibration acquisition."""

from __future__ import annotations

from ._deform360_calibration_acquisition_common import *

def _paired_episode_paths(
    paths: Sequence[str],
    *,
    directory: str,
    data_suffix: str,
    episode_id: int,
    exclude_prefixes: Sequence[str] = (),
) -> tuple[str, str] | None:
    prefix = f"{directory}/"
    direct = [
        path
        for path in paths
        if path.startswith(prefix) and "/" not in path[len(prefix) :]
    ]
    data = {
        Path(path).stem: path
        for path in direct
        if Path(path).suffix.lower() == data_suffix
        and not any(Path(path).stem.startswith(item) for item in exclude_prefixes)
    }
    timestamps = {
        Path(path).stem: path for path in direct if Path(path).suffix.lower() == ".txt"
    }
    stems = sorted(set(data) & set(timestamps))
    if episode_id >= len(stems):
        return None
    stem = stems[episode_id]
    return data[stem], timestamps[stem]


def select_calibration_object_paths(
    repository_paths: Sequence[str],
    *,
    object_id: str,
    episode_id: int,
) -> tuple[str, ...]:
    """Select only one locked object's selected-episode source files.

    Repository tree listing is target blind. The returned paths include object
    metadata, refined camera calibration, one exact MP4/timestamp pair per camera,
    one exact NPY/timestamp pair plus baselines per tactile sensor, and no audio,
    geometry, reconstruction, depth, tracking, point-cloud, or control-point file.
    """

    object_id = nonempty_string(object_id, name="object_id")
    episode_id = genuine_integer(episode_id, name="episode_id", minimum=0)
    object_prefix = f"raw/{object_id}"
    paths = tuple(
        sorted(
            path
            for path in repository_paths
            if isinstance(path, str)
            and (path == object_prefix or path.startswith(f"{object_prefix}/"))
            and Path(path).suffix.lower() not in _AUDIO_SUFFIXES
        )
    )
    if not paths:
        raise ValueError(f"dataset tree has no object path {object_prefix}")

    selected: set[str] = {f"{object_prefix}/metadata.json"}
    selected.update(f"{object_prefix}/{name}" for name in _CALIBRATION_FILES)
    relative_paths = tuple(
        path[len(object_prefix) + 1 :]
        for path in paths
        if path.startswith(f"{object_prefix}/")
    )
    stream_names = sorted(
        {
            path.split("/", 1)[0]
            for path in relative_paths
            if "/" in path
        }
    )
    cameras = [name for name in stream_names if _CAMERA_RE.fullmatch(name)]
    tactile = [name for name in stream_names if _TACTILE_RE.fullmatch(name)]
    if not cameras:
        raise ValueError(f"object {object_id} has no official camera streams")

    for camera in cameras:
        pair = _paired_episode_paths(
            relative_paths,
            directory=camera,
            data_suffix=".mp4",
            episode_id=episode_id,
        )
        if pair is None:
            raise ValueError(
                f"camera {camera} has no paired recording for episode {episode_id}"
            )
        selected.update(f"{object_prefix}/{path}" for path in pair)

    for sensor in tactile:
        sensor_prefix = f"{sensor}/"
        selected.update(
            f"{object_prefix}/{path}"
            for path in relative_paths
            if path.startswith(sensor_prefix)
            and "/" not in path[len(sensor_prefix) :]
            and Path(path).suffix.lower() == ".npy"
            and Path(path).stem.startswith("median_")
        )
        pair = _paired_episode_paths(
            relative_paths,
            directory=sensor,
            data_suffix=".npy",
            episode_id=episode_id,
            exclude_prefixes=("median_",),
        )
        if pair is not None:
            selected.update(f"{object_prefix}/{path}" for path in pair)

    missing = sorted(path for path in selected if path not in paths)
    if missing:
        raise ValueError(f"object {object_id} lacks required source paths: {missing}")
    return tuple(sorted(selected))


def validate_calibration_download_root(
    root: str | Path,
    plan: Deform360CalibrationAcquisitionPlanV1,
    *,
    require_complete: bool,
    expected_paths_by_object: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Reject confirmation, unexpected, missing, or symlinked payload files."""

    if not isinstance(plan, Deform360CalibrationAcquisitionPlanV1):
        raise TypeError("plan must be a Deform360CalibrationAcquisitionPlanV1")
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("calibration download root must be an ordinary directory")
    raw_root = root_path / "raw"
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("calibration download root has no ordinary raw directory")
    invalid_entries = sorted(
        path.name
        for path in raw_root.iterdir()
        if path.is_symlink() or not path.is_dir()
    )
    if invalid_entries:
        raise ValueError(
            "calibration raw root contains non-directory entries: "
            f"{invalid_entries}"
        )
    present = tuple(sorted(path.name for path in raw_root.iterdir()))
    allowed = {unit.object_id for unit in plan.calibration_units}
    forbidden = set(plan.forbidden_confirmation_object_ids)
    unexpected = set(present) - allowed
    if unexpected & forbidden:
        raise ValueError("calibration root contains a confirmation-object subtree")
    if unexpected:
        raise ValueError(
            "calibration root contains unselected objects: "
            f"{sorted(unexpected)}"
        )
    if any(path.is_symlink() for path in raw_root.rglob("*")):
        raise ValueError("calibration root contains a symlink")

    allowed_top_level = {".cache", "processed", "raw"}
    unexpected_top_level = sorted(
        path.name for path in root_path.iterdir() if path.name not in allowed_top_level
    )
    if unexpected_top_level:
        raise ValueError(
            "calibration data root contains unexpected top-level entries: "
            f"{unexpected_top_level}"
        )

    processed_root = root_path / "processed"
    if processed_root.exists():
        if processed_root.is_symlink() or not processed_root.is_dir():
            raise ValueError("processed calibration root must be an ordinary directory")
        processed_entries = sorted(processed_root.iterdir())
        invalid_processed_entries = [
            path.name
            for path in processed_entries
            if path.is_symlink() or not path.is_dir()
        ]
        if invalid_processed_entries:
            raise ValueError(
                "processed calibration root contains non-directory entries: "
                f"{invalid_processed_entries}"
            )
        processed_ids = {path.name for path in processed_entries}
        unexpected_processed = processed_ids - allowed
        if unexpected_processed & forbidden:
            raise ValueError("processed root contains a confirmation-object subtree")
        if unexpected_processed:
            raise ValueError(
                "processed root contains unselected objects: "
                f"{sorted(unexpected_processed)}"
            )
        episode_by_object = {
            unit.object_id: f"source_episode_{unit.episode_id:04d}"
            for unit in plan.calibration_units
        }
        for object_root in processed_entries:
            expected_episode = episode_by_object[object_root.name]
            unexpected_children = sorted(
                child.name
                for child in object_root.iterdir()
                if child.is_symlink()
                or not child.is_dir()
                or child.name != expected_episode
            )
            if unexpected_children:
                raise ValueError(
                    f"processed calibration object {object_root.name} contains "
                    f"unselected episode entries: {unexpected_children}"
                )
            if any(path.is_symlink() for path in object_root.rglob("*")):
                raise ValueError("processed calibration root contains a symlink")

    expected: dict[str, frozenset[str]] | None = None
    if expected_paths_by_object is not None:
        if not isinstance(expected_paths_by_object, Mapping):
            raise ValueError("expected_paths_by_object must be a mapping")
        unexpected_keys = set(expected_paths_by_object) - allowed
        if unexpected_keys & forbidden:
            raise ValueError("payload allowlist contains a confirmation object")
        if unexpected_keys:
            raise ValueError(
                "payload allowlist contains unselected objects: "
                f"{sorted(unexpected_keys)}"
            )
        if require_complete and set(expected_paths_by_object) != allowed:
            missing_keys = allowed - set(expected_paths_by_object)
            raise ValueError(
                "payload allowlist is incomplete for calibration objects: "
                f"{sorted(missing_keys)}"
            )
        expected = {}
        for object_id, raw_paths in expected_paths_by_object.items():
            if isinstance(raw_paths, (str, bytes)):
                raise ValueError("payload allowlist paths must be a sequence")
            canonical = canonical_sorted_strings(
                raw_paths,
                name=f"payload allowlist for {object_id}",
            )
            prefix = PurePosixPath("raw") / object_id
            for raw_path in canonical:
                path = PurePosixPath(raw_path)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.as_posix() != raw_path
                    or path == prefix
                    or not path.is_relative_to(prefix)
                ):
                    raise ValueError(
                        f"payload allowlist path escapes raw/{object_id}: {raw_path}"
                    )
            expected[object_id] = frozenset(canonical)

        for object_id in present:
            if object_id not in expected:
                raise ValueError(
                    f"calibration object has no payload allowlist: {object_id}"
                )
            object_root = raw_root / object_id
            observed_files = {
                path.relative_to(root_path).as_posix()
                for path in object_root.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            extra_files = observed_files - expected[object_id]
            if extra_files:
                raise ValueError(
                    f"calibration object {object_id} contains unselected payloads: "
                    f"{sorted(extra_files)}"
                )
            if require_complete:
                missing_files = expected[object_id] - observed_files
                if missing_files:
                    raise ValueError(
                        f"calibration object {object_id} is missing selected payloads: "
                        f"{sorted(missing_files)}"
                    )
    if require_complete:
        missing = allowed - set(present)
        if missing:
            raise ValueError(f"calibration root is incomplete: {sorted(missing)}")
    return present
