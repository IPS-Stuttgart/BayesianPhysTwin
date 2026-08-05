"""Names-only official-Hub plan for the locked calibration objects."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import (
    CAMERA_RE,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    MINIMUM_CAMERA_STREAMS,
    PARENT_PROTOCOL_ID,
    PLAN_SCHEMA,
    PROCESSING_REPOSITORY,
    PROCESSING_REVISION,
    PROTOCOL_ID,
    SHA256_RE,
    TACTILE_RE,
    CalibrationUnit,
    RepositoryFile,
    canonical_sha256,
    file_sha256,
    load_json,
    load_protocol,
    load_units,
    require,
    summary_gate,
    validate_provider_lock,
    write_json,
)


def _entry_lfs_sha256(entry: object) -> str | None:
    lfs = getattr(entry, "lfs", None)
    if lfs is None:
        return None
    oid = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
    if isinstance(oid, str):
        oid = oid.removeprefix("sha256:")
        if SHA256_RE.fullmatch(oid):
            return oid
    if isinstance(lfs, Mapping):
        value = lfs.get("sha256") or lfs.get("oid")
        if isinstance(value, str):
            value = value.removeprefix("sha256:")
            if SHA256_RE.fullmatch(value):
                return value
    return None


def repository_files(
    entries: Iterable[object],
    *,
    prefix: str,
) -> tuple[RepositoryFile, ...]:
    files: list[RepositoryFile] = []
    for entry in entries:
        path = getattr(entry, "path", None)
        if not isinstance(path, str) or not path.startswith(prefix):
            continue
        entry_type = getattr(entry, "type", None)
        blob_id = getattr(entry, "blob_id", None)
        if entry_type not in {None, "file"} and blob_id is None:
            continue
        if blob_id is None and entry_type != "file":
            continue
        size = getattr(entry, "size", None)
        if isinstance(size, bool) or not isinstance(size, int):
            size = None
        files.append(
            RepositoryFile(
                path=path,
                size=size,
                blob_id=blob_id if isinstance(blob_id, str) else None,
                lfs_sha256=_entry_lfs_sha256(entry),
            )
        )
    result = tuple(sorted(files, key=lambda item: item.path))
    require(result, f"official object subtree is empty: {prefix}")
    return result


def _paired_recordings(
    files: Mapping[str, RepositoryFile],
    *,
    directory: str,
    data_suffix: str,
    exclude_prefix: str | None = None,
) -> tuple[tuple[RepositoryFile, RepositoryFile], ...]:
    prefix = f"{directory}/"
    data: dict[str, RepositoryFile] = {}
    timestamps: dict[str, RepositoryFile] = {}
    for path, record in files.items():
        if not path.startswith(prefix):
            continue
        suffix = PurePosixPath(path).suffix.lower()
        stem = PurePosixPath(path).stem
        if exclude_prefix and stem.startswith(exclude_prefix):
            continue
        if suffix == data_suffix:
            data[stem] = record
        elif suffix == ".txt":
            timestamps[stem] = record
    stems = sorted(set(data) & set(timestamps))
    return tuple((data[stem], timestamps[stem]) for stem in stems)


def select_object_files(
    records: Sequence[RepositoryFile],
    *,
    unit: CalibrationUnit,
) -> dict[str, Any]:
    prefix = f"raw/{unit.object_id}/"
    files = {record.path: record for record in records}
    metadata_path = f"raw/{unit.object_id}/metadata.json"
    require(metadata_path in files, f"metadata is missing: {unit.object_id}")
    calibration = tuple(
        record
        for record in records
        if record.path.startswith(f"{prefix}calibration_refined/")
    )
    require(calibration, f"refined calibration is missing: {unit.object_id}")

    stream_names = sorted(
        {
            PurePosixPath(record.path[len(prefix) :]).parts[0]
            for record in records
            if len(PurePosixPath(record.path[len(prefix) :]).parts) >= 2
        }
    )
    cameras: dict[str, tuple[RepositoryFile, RepositoryFile]] = {}
    tactile: dict[
        str,
        tuple[RepositoryFile, RepositoryFile, RepositoryFile],
    ] = {}
    errors: list[str] = []
    for stream in stream_names:
        directory = f"{prefix}{stream}"
        if CAMERA_RE.fullmatch(stream):
            pairs = _paired_recordings(
                files,
                directory=directory,
                data_suffix=".mp4",
            )
            if unit.episode_id < len(pairs):
                cameras[stream] = pairs[unit.episode_id]
            continue
        if TACTILE_RE.fullmatch(stream):
            pairs = _paired_recordings(
                files,
                directory=directory,
                data_suffix=".npy",
                exclude_prefix="median_",
            )
            baselines = tuple(
                record
                for record in records
                if record.path.startswith(f"{directory}/median_")
                and record.path.endswith(".npy")
            )
            if unit.episode_id >= len(pairs):
                errors.append(f"{stream}: selected tactile episode is absent")
            elif len(baselines) != 1:
                errors.append(
                    f"{stream}: expected exactly one tactile baseline"
                )
            else:
                tactile[stream] = (*pairs[unit.episode_id], baselines[0])

    if len(cameras) < MINIMUM_CAMERA_STREAMS:
        errors.append(
            f"only {len(cameras)} camera streams expose episode "
            f"{unit.episode_id}"
        )
    if not tactile:
        errors.append("no exact tactile stream exposes the selected episode")

    selected: dict[str, RepositoryFile] = {metadata_path: files[metadata_path]}
    selected.update({record.path: record for record in calibration})
    for pair in cameras.values():
        selected.update({record.path: record for record in pair})
    for triple in tactile.values():
        selected.update({record.path: record for record in triple})
    return {
        "object_id": unit.object_id,
        "episode_id": unit.episode_id,
        "stratum": unit.stratum,
        "metadata_path": unit.metadata_path,
        "metadata_sha256": unit.metadata_sha256,
        "status": (
            "planned" if not errors else "unsupported_without_replacement"
        ),
        "errors": errors,
        "camera_streams": sorted(cameras),
        "tactile_streams": sorted(tactile),
        "selected_files": [
            selected[path].to_record() for path in sorted(selected)
        ],
    }


def build_plan(
    *,
    protocol_path: Path,
    selection_path: Path,
    provider_path: Path,
    output_path: Path,
    api: object,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    units, confirmations = load_units(selection_path)
    provider = validate_provider_lock(provider_path)
    rows: list[dict[str, Any]] = []
    for unit in units:
        prefix = f"raw/{unit.object_id}/"
        entries = api.list_repo_tree(
            repo_id=DATASET_REPOSITORY,
            repo_type="dataset",
            revision=DATASET_REVISION,
            path_in_repo=f"raw/{unit.object_id}",
            recursive=True,
            expand=True,
        )
        records = repository_files(entries, prefix=prefix)
        rows.append(select_object_files(records, unit=unit))
    selected_paths = {
        file["path"]
        for row in rows
        for file in row["selected_files"]
        if isinstance(file, Mapping)
    }
    require(
        not any(
            path.startswith(f"raw/{object_id}/")
            for path in selected_paths
            for object_id in confirmations
        ),
        "plan admitted a confirmation object",
    )
    gate = summary_gate(rows, status="planned")
    payload: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "parent_protocol_id": PARENT_PROTOCOL_ID,
        "selection_source_sha256": file_sha256(selection_path),
        "visual_provider_lock_id": provider["artifact_id"],
        "visual_provider_source_sha256": file_sha256(provider_path),
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "processing_repository": PROCESSING_REPOSITORY,
        "processing_revision": PROCESSING_REVISION,
        "objects": rows,
        "gate": gate,
        "information_boundary": {
            "repository_names_opened": True,
            "calibration_payloads_opened": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "replacement_allowed": False,
        },
    }
    payload["plan_sha256"] = canonical_sha256(
        payload,
        digest_key="plan_sha256",
    )
    write_json(output_path, payload)
    return payload


def verify_plan(
    path: Path,
    *,
    protocol_path: Path,
    selection_path: Path,
    provider_path: Path,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    plan = load_json(path)
    protocol = load_protocol(protocol_path)
    _units, confirmations = load_units(selection_path)
    provider = validate_provider_lock(provider_path)
    require(plan.get("schema") == PLAN_SCHEMA, "plan schema changed")
    require(
        plan.get("protocol_sha256") == protocol["protocol_sha256"],
        "plan protocol changed",
    )
    require(
        plan.get("selection_source_sha256") == file_sha256(selection_path),
        "plan selection changed",
    )
    require(
        plan.get("visual_provider_lock_id") == provider["artifact_id"],
        "plan provider changed",
    )
    require(
        plan.get("dataset_revision") == DATASET_REVISION,
        "plan dataset changed",
    )
    require(
        plan.get("processing_revision") == PROCESSING_REVISION,
        "plan processing changed",
    )
    supplied = plan.get("plan_sha256")
    require(isinstance(supplied, str), "plan digest is missing")
    require(
        supplied == canonical_sha256(plan, digest_key="plan_sha256"),
        "plan digest changed",
    )
    boundary = plan.get("information_boundary")
    require(
        isinstance(boundary, Mapping)
        and boundary.get("calibration_payloads_opened") is False
        and boundary.get("confirmation_payloads_opened") is False
        and boundary.get("target_outcomes_used") is False,
        "plan information boundary changed",
    )
    return plan, confirmations
