#!/usr/bin/env python3
"""Validate and inventory retained Deform360 calibration-only prepared source.

The successful calibration-source workflow intentionally publishes compact
provenance while retaining the larger aligned RGB, tactile, and robot products
on the protected self-hosted runner. This command verifies that the retained
bytes still agree with the exact successful plan/download/result/terminal-record
chain, rejects every confirmation object, records portable array/media contracts,
and writes one content-addressed inventory for the next calibration-only stage.

No target metric, geometry annotation, confirmation payload, or target outcome is
opened by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from bayesian_phystwin._deform360_calibration_artifact_chain import (
    download_summary,
    plan_summary,
    result_summary,
    source_lock_summary,
)
from bayesian_phystwin._deform360_calibration_run_common import load_json_object
from bayesian_phystwin._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    sha256_digest,
    write_atomic_json,
)
from bayesian_phystwin.deform360_calibration_source_run_record import (
    validate_deform360_calibration_source_run_record,
)

INVENTORY_SCHEMA = "bayesian-phystwin.deform360-calibration-prepared-source-inventory"
INVENTORY_VERSION = 1
INVENTORY_SEMANTICS = "exact-retained-calibration-rgb-tactile-robot-inventory-v1"
INVENTORY_STATUS = "complete-calibration-only-prepared-source"
INVENTORY_CLAIM_BOUNDARY = (
    "Calibration-only retained-source custody and portable array/media contracts. "
    "A valid inventory does not establish visual-provider competence, contact "
    "calibration, physical-query observability, uncertainty calibration, "
    "confirmation accuracy, Causal4D benefit, deployment safety, or state of the art."
)

_BOUNDARY = {
    "calibration_camera_payloads_opened": True,
    "calibration_tactile_payloads_opened": True,
    "calibration_robot_state_opened": True,
    "calibration_target_metrics_computed": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}
_REQUIRED_ROBOT_ARRAYS = frozenset({"actions", "T_worlds", "openings", "bimanual"})
_SOURCE_KEYS = (
    "source_locks_available",
    "source_locks_valid",
    "source_locks_error",
    "source_protocol_file_sha256",
    "source_protocol_sha256",
    "stage0_protocol_file_sha256",
    "stage0_protocol_sha256",
    "selection_lock_file_sha256",
    "selection_artifact_sha256",
    "content_selection_sha256",
    "visual_provider_lock_file_sha256",
    "visual_provider_lock_id",
)
_PLAN_KEYS = (
    "plan_available",
    "plan_valid",
    "plan_error",
    "plan_file_sha256",
    "plan_sha256",
    "plan_support_gate",
)
_DOWNLOAD_KEYS = (
    "download_available",
    "download_valid",
    "download_error",
    "download_file_sha256",
    "download_sha256",
)
_RESULT_KEYS = (
    "result_available",
    "result_valid",
    "result_error",
    "result_file_sha256",
    "result_sha256",
    "support_gate",
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    absolute = Path(path).absolute()
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError(f"{name} path must not contain symbolic links: {path}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an ordinary directory: {path}")
    return resolved


def _ordinary_file(path: Path, *, root: Path, name: str) -> Path:
    candidate = path.absolute()
    current = candidate
    while True:
        if current.is_symlink():
            raise ValueError(f"{name} path must not contain symbolic links: {path}")
        if current == root or current.parent == current:
            break
        current = current.parent
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist: {path}") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes prepared root: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must be an ordinary file: {path}")
    return resolved


def _portable_path(path: Path, *, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"prepared path escapes root: {path}") from error
    text = relative.as_posix()
    pure = PurePosixPath(text)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"prepared path is not canonical: {text}")
    return pure.as_posix()


def _file_record(path: Path, *, root: Path) -> dict[str, object]:
    ordinary = _ordinary_file(path, root=root, name=_portable_path(path, root=root))
    return {
        "path": _portable_path(ordinary, root=root),
        "sha256": _sha256_file(ordinary),
        "byte_count": ordinary.stat().st_size,
    }


def _numeric_contract(value: np.ndarray, *, name: str) -> dict[str, object]:
    array = np.asarray(value)
    if array.dtype.kind in "iufcb":
        finite = bool(np.all(np.isfinite(array)))
        _require(finite, f"{name} contains non-finite values")
    else:
        finite = None
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "finite": finite,
    }


def _npy_record(path: Path, *, root: Path, expected_sha256: str) -> dict[str, object]:
    record = _file_record(path, root=root)
    _require(record["sha256"] == expected_sha256, f"prepared array changed: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    try:
        contract = _numeric_contract(array, name=str(path))
    finally:
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    return {**record, **contract}


def _npz_record(path: Path, *, root: Path, expected_sha256: str) -> dict[str, object]:
    record = _file_record(path, root=root)
    _require(record["sha256"] == expected_sha256, f"prepared archive changed: {path}")
    with np.load(path, allow_pickle=False) as stored:
        names = tuple(sorted(stored.files))
        missing = sorted(_REQUIRED_ROBOT_ARRAYS - set(names))
        if missing:
            raise ValueError(f"robot archive is missing arrays: {missing}")
        arrays = {
            name: _numeric_contract(stored[name], name=f"{path}:{name}")
            for name in names
        }
    return {**record, "arrays": arrays}


def _load_camera_metadata(path: Path) -> Mapping[str, Any]:
    metadata = load_strict_json_object(path, label="prepared camera metadata")
    if metadata.get("schema") != "deform360.camera-alignment/v1":
        raise ValueError(f"prepared camera metadata schema changed: {path}")
    return metadata


def _camera_record(
    episode_dir: Path,
    camera: str,
    *,
    root: Path,
    frame_count: int,
) -> dict[str, object]:
    camera_dir = episode_dir / camera
    _ordinary_directory(camera_dir, name=f"camera {camera}")
    video = _file_record(camera_dir / "undistorted.mp4", root=root)
    preview = _file_record(camera_dir / "undistorted_000000.png", root=root)
    timestamps = _file_record(camera_dir / "aligned_timestamps.txt", root=root)
    alignment = _file_record(camera_dir / "alignment.json", root=root)
    metadata_file = _file_record(camera_dir / "metadata.json", root=root)
    metadata = _load_camera_metadata(camera_dir / "metadata.json")
    output = metadata.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(f"prepared camera metadata lacks output contract: {camera}")
    expected = {
        "video_sha256": video["sha256"],
        "preview_sha256": preview["sha256"],
        "timestamp_sha256": timestamps["sha256"],
        "alignment_sha256": alignment["sha256"],
    }
    for key, digest in expected.items():
        if output.get(key) != digest:
            raise ValueError(f"prepared camera metadata changed {key}: {camera}")
    if output.get("frame_count") != frame_count:
        raise ValueError(f"prepared camera frame count changed: {camera}")
    target = metadata.get("target_timeline")
    if not isinstance(target, Mapping) or target.get("count") != frame_count:
        raise ValueError(f"prepared camera target timeline changed: {camera}")
    for dimension in ("width", "height"):
        value = output.get(dimension)
        if type(value) is not int or value <= 0:
            raise ValueError(f"prepared camera {dimension} is invalid: {camera}")
    fps = output.get("fps")
    if type(fps) not in {int, float} or not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError(f"prepared camera fps is invalid: {camera}")
    return {
        "camera": camera,
        "video": video,
        "preview": preview,
        "timestamps": timestamps,
        "alignment": alignment,
        "metadata": metadata_file,
        "frame_count": frame_count,
        "width": output["width"],
        "height": output["height"],
        "fps": float(fps),
        "timeline_sha256": target.get("sha256"),
    }


def _compare_summary(
    observed: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    keys: Sequence[str],
    name: str,
) -> None:
    for key in keys:
        if observed.get(key) != record.get(key):
            raise ValueError(f"{name} differs from terminal record: {key}")


def _successful_record(value: Mapping[str, Any]) -> Mapping[str, Any]:
    record = validate_deform360_calibration_source_run_record(value)
    if record.get("status") != "succeeded" or record.get("exit_code") != 0:
        raise ValueError("calibration-source terminal record did not succeed")
    if record.get("confirmation_boundary_verified") is not True:
        raise ValueError("calibration-source confirmation boundary is unverified")
    if record.get("confirmation_payloads_opened") is not False:
        raise ValueError(
            "calibration-source terminal record reports confirmation access"
        )
    gate = record.get("support_gate")
    if not isinstance(gate, Mapping) or gate.get("support_passed") is not True:
        raise ValueError("calibration-source support gate did not pass")
    return record


def _result_rows(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = value.get("objects")
    if not isinstance(rows, list):
        raise ValueError("calibration-source result rows are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"calibration-source result row {index} is invalid")
        object_id = row.get("object_id")
        if type(object_id) is not str or not object_id:
            raise ValueError(f"calibration-source result row {index} lacks object_id")
        if object_id in result:
            raise ValueError(f"calibration-source result repeats object {object_id}")
        result[object_id] = row
    return result


def build_inventory(
    *,
    source_protocol_path: Path,
    stage0_protocol_path: Path,
    selection_lock_path: Path,
    visual_provider_lock_path: Path,
    plan_path: Path,
    download_path: Path,
    result_path: Path,
    run_record_path: Path,
    processed_root: Path,
    implementation_revision: str,
) -> dict[str, object]:
    implementation = exact_revision(
        implementation_revision,
        name="implementation_revision",
    )
    raw_record, run_record_file_sha256 = load_json_object(run_record_path)
    record = _successful_record(raw_record)
    processing_revision = exact_revision(
        record.get("processing_revision"),
        name="processing_revision",
    )
    source_locks, expected_units, confirmation_ids = source_lock_summary(
        source_protocol_json=source_protocol_path,
        stage0_protocol_json=stage0_protocol_path,
        selection_lock=selection_lock_path,
        visual_provider_lock=visual_provider_lock_path,
        processing_revision=processing_revision,
    )
    if source_locks.get("source_locks_valid") is not True:
        raise ValueError("prepared-source locks are invalid")
    _compare_summary(source_locks, record, keys=_SOURCE_KEYS, name="source locks")
    plan, expected_identities, planned_ids = plan_summary(
        plan_path,
        processing_revision=processing_revision,
        source_locks=source_locks,
        expected_units=expected_units,
        confirmation_ids=confirmation_ids,
    )
    if plan.get("plan_valid") is not True:
        raise ValueError("prepared-source plan is invalid")
    _compare_summary(plan, record, keys=_PLAN_KEYS, name="plan")
    plan_sha256 = sha256_digest(plan.get("plan_sha256"), name="plan_sha256")
    download = download_summary(
        download_path,
        plan_sha256=plan_sha256,
        planned_ids=planned_ids,
        confirmation_ids=confirmation_ids,
    )
    if download.get("download_valid") is not True:
        raise ValueError("prepared-source download manifest is invalid")
    _compare_summary(download, record, keys=_DOWNLOAD_KEYS, name="download")
    download_sha256 = sha256_digest(
        download.get("download_sha256"),
        name="download_sha256",
    )
    result = result_summary(
        result_path,
        processing_revision=processing_revision,
        plan_sha256=plan_sha256,
        download_sha256=download_sha256,
        expected_identities=expected_identities,
        planned_ids=planned_ids,
    )
    if result.get("result_valid") is not True:
        raise ValueError("prepared-source result is invalid")
    _compare_summary(result, record, keys=_RESULT_KEYS, name="result")
    result_value, result_file_sha256 = load_json_object(result_path)
    if result_file_sha256 != result.get("result_file_sha256"):
        raise ValueError("prepared-source result changed after validation")
    rows = _result_rows(result_value)

    root = _ordinary_directory(processed_root, name="prepared processed root")
    present = {path.name for path in root.iterdir() if path.is_dir()}
    forbidden = sorted(present & set(confirmation_ids))
    if forbidden:
        raise ValueError(f"confirmation objects appear in prepared root: {forbidden}")
    expected_object_ids = tuple(sorted(expected_units))
    if tuple(sorted(rows)) != expected_object_ids:
        raise ValueError(
            "prepared-source result does not cover the exact calibration cohort"
        )

    objects: list[dict[str, object]] = []
    for object_id in expected_object_ids:
        episode_id, stratum, _metadata_path, _metadata_sha256 = expected_units[
            object_id
        ]
        row = rows[object_id]
        if (
            row.get("status") != "source_prepared"
            or row.get("episode_id") != episode_id
            or row.get("stratum") != stratum
            or row.get("synthetic_episode_index") != 0
        ):
            raise ValueError(f"prepared-source object identity changed: {object_id}")
        frame_count = row.get("aligned_frame_count")
        if type(frame_count) is not int or frame_count <= 0:
            raise ValueError(f"prepared-source frame count is invalid: {object_id}")
        cameras = row.get("cameras")
        tactile_sensors = row.get("tactile_sensors")
        outputs = row.get("outputs_sha256")
        action_window = row.get("action_window")
        if (
            not isinstance(cameras, list)
            or not cameras
            or cameras != sorted(set(cameras))
            or row.get("camera_count") != len(cameras)
        ):
            raise ValueError(f"prepared-source cameras are invalid: {object_id}")
        if (
            not isinstance(tactile_sensors, list)
            or not tactile_sensors
            or tactile_sensors != sorted(set(tactile_sensors))
            or row.get("tactile_sensor_count") != len(tactile_sensors)
        ):
            raise ValueError(
                f"prepared-source tactile sensors are invalid: {object_id}"
            )
        if not isinstance(outputs, Mapping) or not isinstance(action_window, Mapping):
            raise ValueError(f"prepared-source outputs are invalid: {object_id}")

        object_root = _ordinary_directory(root / object_id, name=f"object {object_id}")
        episode_dir = _ordinary_directory(
            object_root / "episode_0000",
            name=f"object {object_id} episode",
        )
        alignment = _file_record(episode_dir / "alignment.json", root=root)
        intrinsics = _file_record(
            episode_dir / "undistorted_intrinsics.npy",
            root=root,
        )
        extrinsics = _file_record(episode_dir / "extrinsics.npy", root=root)
        for name, file_record in (
            ("alignment", alignment),
            ("undistorted_intrinsics", intrinsics),
            ("extrinsics", extrinsics),
        ):
            if outputs.get(name) != file_record["sha256"]:
                raise ValueError(f"prepared-source {name} changed: {object_id}")

        robot_sha256 = sha256_digest(outputs.get("robot"), name=f"{object_id} robot")
        robot = _npz_record(
            episode_dir / "robot" / "robot.npz",
            root=root,
            expected_sha256=robot_sha256,
        )
        tactile_hashes = outputs.get("tactile")
        if not isinstance(tactile_hashes, Mapping) or set(tactile_hashes) != set(
            tactile_sensors
        ):
            raise ValueError(f"prepared-source tactile identities changed: {object_id}")
        tactile = [
            {
                "sensor": sensor,
                **_npy_record(
                    episode_dir / sensor / "synced_tactile.npy",
                    root=root,
                    expected_sha256=sha256_digest(
                        tactile_hashes[sensor],
                        name=f"{object_id} tactile {sensor}",
                    ),
                ),
            }
            for sensor in tactile_sensors
        ]
        camera_records = [
            _camera_record(
                episode_dir,
                camera,
                root=root,
                frame_count=frame_count,
            )
            for camera in cameras
        ]
        window = dict(action_window)
        selected_range = window.get("selected_raw_frame_range_half_open")
        if (
            not isinstance(selected_range, list)
            or len(selected_range) != 2
            or any(type(value) is not int for value in selected_range)
            or selected_range[0] < 0
            or selected_range[1] - selected_range[0] != 81
            or selected_range[1] > frame_count
        ):
            raise ValueError(f"prepared-source action window changed: {object_id}")
        objects.append(
            {
                "object_id": object_id,
                "episode_id": episode_id,
                "stratum": stratum,
                "synthetic_episode_index": 0,
                "aligned_frame_count": frame_count,
                "action_window": window,
                "episode_files": {
                    "alignment": alignment,
                    "undistorted_intrinsics": intrinsics,
                    "extrinsics": extrinsics,
                    "robot": robot,
                },
                "cameras": camera_records,
                "tactile": tactile,
            }
        )

    source_artifacts = {
        "sources/calibration-source/protocol.json": sha256_digest(
            source_locks.get("source_protocol_file_sha256"),
            name="source protocol file SHA-256",
        ),
        "sources/stage0/protocol.json": sha256_digest(
            source_locks.get("stage0_protocol_file_sha256"),
            name="Stage-0 protocol file SHA-256",
        ),
        "sources/stage0/selection.json": sha256_digest(
            source_locks.get("selection_lock_file_sha256"),
            name="Stage-0 selection file SHA-256",
        ),
        "sources/locks/visual-provider-lock.json": sha256_digest(
            source_locks.get("visual_provider_lock_file_sha256"),
            name="visual-provider lock file SHA-256",
        ),
        "sources/calibration-source/plan.json": sha256_digest(
            plan.get("plan_file_sha256"),
            name="plan file SHA-256",
        ),
        "sources/calibration-source/download.json": sha256_digest(
            download.get("download_file_sha256"),
            name="download file SHA-256",
        ),
        "sources/calibration-source/result.json": result_file_sha256,
        "sources/calibration-source/execution-manifest.json": run_record_file_sha256,
    }
    identity: dict[str, object] = {
        "schema": INVENTORY_SCHEMA,
        "schema_version": INVENTORY_VERSION,
        "semantics": INVENTORY_SEMANTICS,
        "status": INVENTORY_STATUS,
        "implementation_revision": implementation,
        "calibration_source_revision": exact_revision(
            record.get("source_revision"),
            name="calibration_source_revision",
        ),
        "processing_revision": processing_revision,
        "selection_artifact_sha256": sha256_digest(
            source_locks.get("selection_artifact_sha256"),
            name="selection_artifact_sha256",
        ),
        "visual_provider_lock_id": sha256_digest(
            source_locks.get("visual_provider_lock_id"),
            name="visual_provider_lock_id",
        ),
        "calibration_source_run_record_sha256": sha256_digest(
            record.get("record_sha256"),
            name="calibration source record SHA-256",
        ),
        "object_count": len(objects),
        "objects": objects,
        "source_artifacts": source_artifacts,
        "information_boundary": dict(_BOUNDARY),
        "claim_boundary": INVENTORY_CLAIM_BOUNDARY,
    }
    return {**identity, "inventory_id": content_id(identity)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, required=True)
    parser.add_argument("--stage0-protocol", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--visual-provider-lock", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inventory = build_inventory(
            source_protocol_path=args.source_protocol,
            stage0_protocol_path=args.stage0_protocol,
            selection_lock_path=args.selection_lock,
            visual_provider_lock_path=args.visual_provider_lock,
            plan_path=args.plan,
            download_path=args.download_manifest,
            result_path=args.result,
            run_record_path=args.run_record,
            processed_root=args.processed_root,
            implementation_revision=args.implementation_revision,
        )
        write_atomic_json(inventory, args.output, overwrite=False)
    except (OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {"complete": False, "error": str(error)},
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "complete": True,
                "inventory_id": inventory["inventory_id"],
                "object_count": inventory["object_count"],
                "output": str(args.output),
                "information_boundary": inventory["information_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
