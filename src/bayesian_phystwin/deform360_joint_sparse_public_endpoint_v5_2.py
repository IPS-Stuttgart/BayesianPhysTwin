"""Post-seal public Deform360 endpoint archive materialization for v5.2."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    require_exact_fields,
    write_atomic_json,
)
from .deform360_joint_sparse_endpoint_v5 import select_reserved_endpoint_views_v5
from .deform360_joint_sparse_source_evidence_v5 import (
    validate_deform360_joint_sparse_source_prediction_batch_v5,
)
from .deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from .deform360_joint_sparse_source_runner_v5 import _ordinary_root, _sha256_file
from .deform360_joint_sparse_source_runner_v5_2 import (
    validate_deform360_joint_sparse_source_prediction_plan_v5_2,
    validate_deform360_joint_sparse_source_prediction_receipt_v5_2,
)

PUBLIC_ENDPOINT_MANIFEST_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-public-endpoint-materialization"
)
PUBLIC_ENDPOINT_MANIFEST_VERSION: Final = 1
PUBLIC_ENDPOINT_MANIFEST_SEMANTICS: Final = (
    "post-seal-public-rgb-derived-reserved-view-endpoint-archives-v1"
)
LOCAL_ENDPOINT_FRAME_RANGE_HALF_OPEN: Final = (58, 76)
PUBLIC_ENDPOINT_PROCESSING_LOCK_SCHEMA: Final = (
    "bayesian-phystwin.deform360-joint-sparse-public-endpoint-processing-lock"
)
PUBLIC_ENDPOINT_PROCESSING_LOCK_VERSION: Final = 1
PUBLIC_ENDPOINT_PROCESSING_LOCK_SEMANTICS: Final = (
    "sealed-public-rgb-endpoint-geometry-processing-v1"
)
PUBLIC_ENDPOINT_BASE_CAMERA_PANEL: Final = [
    "brics-odroid-001_cam0",
    "brics-odroid-006_cam0",
    "brics-odroid-007_cam0",
    "brics-odroid-008_cam0",
    "brics-odroid-010_cam0",
    "brics-odroid-013_cam0",
    "brics-odroid-014_cam1",
    "brics-odroid-015_cam1",
    "brics-odroid-019_cam1",
    "brics-odroid-021_cam1",
    "brics-odroid-024_cam1",
    "brics-odroid-027_cam0",
]
PUBLIC_ENDPOINT_PROCESSING_CONTRACT: Final = {
    "windowing": {
        "raw_start": "source-plan-raw-prefix-start",
        "raw_frame_count": 81,
        "local_prefix_range_half_open": [0, 58],
        "local_endpoint_range_half_open": [58, 76],
        "local_unscored_tail_range_half_open": [76, 81],
        "video_materialization": "exact-ffmpeg-libx264-crf12-30hz",
        "timestamps_and_robot_state_sliced_to_same_raw_range": True,
    },
    "camera_policy": {
        "base_camera_panel": PUBLIC_ENDPOINT_BASE_CAMERA_PANEL,
        "reserved_endpoint_camera_count": 2,
        "reserved_endpoint_cameras_must_succeed": True,
        "support_camera_order": "lexical-order-of-successful-fixed-panel-union-reserved",
        "minimum_successful_camera_count": 8,
        "camera_substitution_allowed": False,
    },
    "masking": {
        "method": "object-agnostic-automatic-sam2-candidate-on-local-frame-zero",
        "propagation_frame_count": 81,
        "manual_prompting_or_mask_selection": False,
        "selector_base_source_sha256": (
            "419be2e98ab2b01627ea188c8658b43b39d8b3d4e34e8b33559f32ccdcd04184"
        ),
        "selector_object_source_sha256": (
            "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
        ),
        "sam2_revision": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "sam2_checkpoint_sha256": (
            "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
        ),
    },
    "reconstruction": {
        "deform360_revision": "d8522a4403b766aeb387510c04e89032a56fdf35",
        "reconstruct_stage_sha256": (
            "53a1e8b73e56a1c68a0c4344b279c2817ed4b3ed93e8f5ea792def26d5099c7c"
        ),
        "urdf_render_stage_sha256": (
            "c4d6a10e980ed4952f974d2e8a991c6fb819a3e6fdc6c121d3ce6925c94c2467"
        ),
        "depth_stage_sha256": (
            "34befb732107b805f1e1924699f1e26fc2ca5d3041561b920d8c23d8e85feef0"
        ),
        "first_frame_iterations": 500,
        "warm_start_iterations": 250,
        "cube_half_extent_m": 0.5,
        "voxel_resolution": 120,
        "minimum_visual_hull_points": 512,
        "reuse_previous_geometry": False,
    },
    "depth": {
        "expected_depth": True,
        "object_mask_applied": True,
        "gripper_urdf_mask_applied": True,
        "storage_dtype": "uint16",
        "storage_unit": "millimetre",
        "frame_count": 81,
        "preview_video": False,
    },
    "runtime": {
        "python_major_minor": "3.10",
        "torch_version": "2.4.0+cu121",
        "torch_cuda_version": "12.1",
        "gsplat_version": "1.4.0",
        "gsplat_extension_sha256": (
            "152153a9cde346203f32a9792f2e2345450324cbc7652668e76574e8f3a490f0"
        ),
        "gsplat_build_ninja_sha256": (
            "a51f8d2b746d7e4a41846e196c01333a537c4c6342ae9e7ee58a22b881f0b5e4"
        ),
        "gsplat_backend_probe": "CameraModelType.PINHOLE",
        "torch_cuda_arch_list": "8.9",
    },
    "failure_policy": {
        "technical_failure_is_terminal": True,
        "retry_after_inspecting_endpoint_geometry": False,
        "implicit_replacement": False,
    },
}
PUBLIC_ENDPOINT_PROCESSING_BOUNDARY: Final = {
    "source_predictions_sealed_before_endpoint_processing": True,
    "development_suffix_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "held_v8_artifacts_accessed": False,
    "human_approval_required": False,
    "human_selection_allowed": False,
    "new_measurements_required": False,
    "released_real_world_recordings_only": True,
}
_PROCESSING_LOCK_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "semantics",
        "execution_lock_id",
        "source_prediction_plan_id",
        "source_prediction_receipt_id",
        "prediction_batch_id",
        "dataset_repository",
        "dataset_revision",
        "processing",
        "information_boundary",
        "lock_id",
    }
)
_ARCHIVE_MEMBERS: Final = frozenset(
    {
        "camera_to_world",
        "depth_m",
        "frame_indices",
        "intrinsics",
        "object_mask",
        "raw_frame_indices",
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _ordinary_file(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    _require(
        not any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)),
        f"{name} path must not contain symlinks",
    )
    resolved = absolute.resolve(strict=True)
    _require(
        resolved.is_file(),
        f"{name} must be an ordinary file",
    )
    return resolved


def validate_public_endpoint_processing_lock_v5_2(
    value: Mapping[str, Any],
    *,
    execution_lock: Mapping[str, Any],
    source_plan: Mapping[str, Any],
    prediction_batch: Mapping[str, Any],
    prediction_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen, outcome-blind public endpoint recipe."""

    require_exact_fields(
        value,
        expected=_PROCESSING_LOCK_FIELDS,
        name="public endpoint processing lock",
    )
    identity = {key: item for key, item in value.items() if key != "lock_id"}
    _require(value.get("lock_id") == content_id(identity), "processing lock ID changed")
    _require(
        value.get("schema") == PUBLIC_ENDPOINT_PROCESSING_LOCK_SCHEMA
        and value.get("schema_version") == PUBLIC_ENDPOINT_PROCESSING_LOCK_VERSION
        and value.get("semantics") == PUBLIC_ENDPOINT_PROCESSING_LOCK_SEMANTICS,
        "processing lock schema changed",
    )
    _require(
        value.get("execution_lock_id") == execution_lock.get("execution_lock_id")
        and value.get("source_prediction_plan_id") == source_plan.get("plan_id")
        and value.get("prediction_batch_id")
        == prediction_batch.get("prediction_batch_id")
        and value.get("source_prediction_receipt_id")
        == prediction_receipt.get("receipt_id"),
        "processing lock binds another sealed source panel",
    )
    measurements = cast(Mapping[str, Any], execution_lock["public_measurements"])
    _require(
        value.get("dataset_repository") == measurements.get("dataset_repository")
        and value.get("dataset_revision") == measurements.get("dataset_revision"),
        "processing lock binds another public dataset",
    )
    _require(
        value.get("processing") == PUBLIC_ENDPOINT_PROCESSING_CONTRACT,
        "public endpoint processing contract changed",
    )
    _require(
        value.get("information_boundary") == PUBLIC_ENDPOINT_PROCESSING_BOUNDARY,
        "public endpoint processing boundary changed",
    )
    return dict(value)


def _camera_dictionary(path: Path, *, name: str) -> dict[str, np.ndarray]:
    source = _ordinary_file(path, name=name)
    try:
        value = np.load(source, allow_pickle=True)
        _require(value.shape == () and value.dtype == object, f"{name} changed format")
        raw = value.item()
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {name}") from error
    _require(isinstance(raw, Mapping), f"{name} must contain a camera mapping")
    result: dict[str, np.ndarray] = {}
    for key, array in raw.items():
        _require(type(key) is str, f"{name} camera IDs must be strings")
        values = np.asarray(array, dtype=np.float64)
        _require(np.all(np.isfinite(values)), f"{name} must be finite")
        result[key] = values
    return result


def _h5_data(path: Path, *, name: str) -> np.ndarray:
    source = _ordinary_file(path, name=name)
    try:
        import h5py  # noqa: PLC0415

        with h5py.File(source, "r") as stream:
            _require(set(stream) == {"data"}, f"{name} member roster changed")
            values = np.asarray(stream["data"])
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load {name}") from error
    return cast(np.ndarray[Any, Any], values)


def _write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    _require(set(arrays) == _ARCHIVE_MEMBERS, "endpoint archive roster changed")
    _require(not path.exists(), f"endpoint archive already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            camera_to_world=arrays["camera_to_world"],
            depth_m=arrays["depth_m"],
            frame_indices=arrays["frame_indices"],
            intrinsics=arrays["intrinsics"],
            object_mask=arrays["object_mask"],
            raw_frame_indices=arrays["raw_frame_indices"],
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_public_endpoint_archive_v5_2(
    *,
    object_id: str,
    camera_id: str,
    raw_endpoint_range_half_open: tuple[int, int],
    episode_directory: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Convert pinned Deform360 depth products into the exact endpoint NPZ."""

    episode = _ordinary_root(episode_directory)
    camera = episode / camera_id
    depth_path = _ordinary_file(camera / "rendered_depth.h5", name="rendered depth")
    mask_path = _ordinary_file(camera / "mask_refined.h5", name="object mask")
    intrinsics_path = _ordinary_file(
        episode / "undistorted_intrinsics.npy", name="intrinsics"
    )
    extrinsics_path = _ordinary_file(episode / "extrinsics.npy", name="extrinsics")
    depth_mm = _h5_data(depth_path, name="rendered depth")
    mask = _h5_data(mask_path, name="object mask")
    start, stop = LOCAL_ENDPOINT_FRAME_RANGE_HALF_OPEN
    _require(
        depth_mm.ndim == 3
        and depth_mm.dtype == np.uint16
        and depth_mm.shape[0] >= stop,
        "rendered depth must be uint16 millimetres with all endpoint frames",
    )
    _require(
        mask.ndim == 3
        and mask.dtype in (np.dtype(np.uint8), np.dtype(np.bool_))
        and mask.shape == depth_mm.shape,
        "object mask must match rendered depth",
    )
    intrinsics = _camera_dictionary(intrinsics_path, name="intrinsics")
    extrinsics = _camera_dictionary(extrinsics_path, name="extrinsics")
    _require(
        camera_id in intrinsics and camera_id in extrinsics, "camera is uncalibrated"
    )
    camera_intrinsics = intrinsics[camera_id]
    camera_to_world = extrinsics[camera_id]
    _require(camera_intrinsics.shape == (3, 3), "intrinsics shape changed")
    _require(camera_to_world.shape == (4, 4), "extrinsics shape changed")
    raw_start, raw_stop = raw_endpoint_range_half_open
    _require(raw_stop - raw_start == stop - start, "raw endpoint length changed")
    destination = Path(output_path).absolute()
    _write_npz_once(
        destination,
        {
            "frame_indices": np.arange(start, stop, dtype=np.int64),
            "raw_frame_indices": np.arange(raw_start, raw_stop, dtype=np.int64),
            "depth_m": np.asarray(depth_mm[start:stop], dtype=np.float32) / 1000.0,
            "object_mask": np.asarray(mask[start:stop] > 0, dtype=np.bool_),
            "intrinsics": np.asarray(camera_intrinsics, dtype=np.float64),
            "camera_to_world": np.asarray(camera_to_world, dtype=np.float64),
        },
    )
    with np.load(destination, allow_pickle=False) as archive:
        _require(set(archive.files) == _ARCHIVE_MEMBERS, "written archive changed")
    return {
        "object_id": object_id,
        "camera_id": camera_id,
        "archive_sha256": _sha256_file(destination),
        "source_files_sha256": {
            "rendered_depth.h5": _sha256_file(depth_path),
            "mask_refined.h5": _sha256_file(mask_path),
            "undistorted_intrinsics.npy": _sha256_file(intrinsics_path),
            "extrinsics.npy": _sha256_file(extrinsics_path),
        },
    }


def materialize_public_endpoint_inputs_v5_2(
    *,
    execution_lock_path: str | Path,
    source_prediction_plan_path: str | Path,
    source_prediction_root: str | Path,
    processing_lock_path: str | Path,
    processed_root: str | Path,
    output_root: str | Path,
    objects_output_path: str | Path,
    manifest_output_path: str | Path,
) -> dict[str, Any]:
    """Validate the sealed panel, then materialize all reserved public views."""

    lock = load_deform360_joint_sparse_source_execution_lock_v5(execution_lock_path)
    plan_path = Path(source_prediction_plan_path).resolve(strict=True)
    plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        load_strict_json_object(plan_path, label="v5.2 source prediction plan"),
        lock=lock,
    )
    prediction_root = _ordinary_root(source_prediction_root)
    batch_path = prediction_root / "source-prediction-batch.json"
    receipt_path = prediction_root / "source-prediction-receipt.json"
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        load_strict_json_object(batch_path, label="source prediction batch"), lock
    )
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5_2(
        load_strict_json_object(receipt_path, label="v5.2 source prediction receipt"),
        lock=lock,
        plan=plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=_sha256_file(batch_path),
    )
    processing_lock_path_resolved = _ordinary_file(
        Path(processing_lock_path), name="public endpoint processing lock"
    )
    processing_lock = validate_public_endpoint_processing_lock_v5_2(
        load_strict_json_object(
            processing_lock_path_resolved, label="public endpoint processing lock"
        ),
        execution_lock=lock,
        source_plan=plan,
        prediction_batch=batch,
        prediction_receipt=receipt,
    )

    processing = _ordinary_root(processed_root)
    output = Path(output_root).absolute()
    output.mkdir(parents=True, exist_ok=True)
    _require(
        output.is_dir()
        and not output.is_symlink()
        and not any(parent.is_symlink() for parent in output.parents),
        "endpoint output root is invalid",
    )
    objects: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for row in cast(Sequence[Mapping[str, Any]], plan["objects"]):
        object_id = cast(str, row["object_id"])
        cameras = cast(Sequence[str], row["all_camera_ids"])
        reserved = select_reserved_endpoint_views_v5(object_id, cameras, count=2)
        raw_prefix = cast(Sequence[int], row["raw_prefix_range_half_open"])
        raw_endpoint = (int(raw_prefix[1]), int(raw_prefix[1]) + 18)
        episode = processing / object_id / "episode_0000"
        views: list[dict[str, Any]] = []
        for camera_id in reserved:
            relative = Path("endpoint") / object_id / f"{camera_id}.npz"
            record = materialize_public_endpoint_archive_v5_2(
                object_id=object_id,
                camera_id=camera_id,
                raw_endpoint_range_half_open=raw_endpoint,
                episode_directory=episode,
                output_path=output / relative,
            )
            source_records.append({**record, "archive_path": relative.as_posix()})
            views.append(
                {
                    "camera_id": camera_id,
                    "endpoint_archive": {
                        "path": relative.as_posix(),
                        "sha256": record["archive_sha256"],
                    },
                }
            )
        objects.append(
            {
                "object_id": object_id,
                "episode_id": row["episode_id"],
                "stratum": row["stratum"],
                "all_camera_ids": list(cameras),
                "raw_endpoint_range_half_open": list(raw_endpoint),
                "reserved_views": views,
            }
        )
    objects.sort(key=lambda item: cast(str, item["object_id"]))
    source_records.sort(
        key=lambda item: (cast(str, item["object_id"]), cast(str, item["camera_id"]))
    )
    objects_destination = Path(objects_output_path).absolute()
    manifest_destination = Path(manifest_output_path).absolute()
    objects_payload = {"objects": objects}
    write_atomic_json(objects_payload, objects_destination, overwrite=False)
    public_measurements = cast(Mapping[str, Any], lock["public_measurements"])
    identity: dict[str, Any] = {
        "schema": PUBLIC_ENDPOINT_MANIFEST_SCHEMA,
        "schema_version": PUBLIC_ENDPOINT_MANIFEST_VERSION,
        "semantics": PUBLIC_ENDPOINT_MANIFEST_SEMANTICS,
        "execution_lock_id": lock["execution_lock_id"],
        "source_prediction_plan_id": plan["plan_id"],
        "source_prediction_receipt_id": receipt["receipt_id"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "processing_lock_id": processing_lock["lock_id"],
        "processing_lock_file_sha256": _sha256_file(processing_lock_path_resolved),
        "dataset_repository": public_measurements["dataset_repository"],
        "dataset_revision": public_measurements["dataset_revision"],
        "endpoint_geometry_derivation_repository": public_measurements[
            "endpoint_geometry_derivation_repository"
        ],
        "endpoint_geometry_derivation_revision": public_measurements[
            "endpoint_geometry_derivation_revision"
        ],
        "local_endpoint_frame_range_half_open": list(
            LOCAL_ENDPOINT_FRAME_RANGE_HALF_OPEN
        ),
        "objects_file_sha256": _sha256_file(objects_destination),
        "source_records": source_records,
        "information_boundary": {
            "development_suffix_opened_after_prediction_receipt": True,
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
            "human_approval_required": False,
            "new_measurements_required": False,
        },
    }
    manifest = {**identity, "manifest_id": content_id(identity)}
    write_atomic_json(manifest, manifest_destination, overwrite=False)
    return manifest


__all__ = [
    "LOCAL_ENDPOINT_FRAME_RANGE_HALF_OPEN",
    "PUBLIC_ENDPOINT_MANIFEST_SCHEMA",
    "PUBLIC_ENDPOINT_PROCESSING_BOUNDARY",
    "PUBLIC_ENDPOINT_PROCESSING_CONTRACT",
    "PUBLIC_ENDPOINT_PROCESSING_LOCK_SCHEMA",
    "materialize_public_endpoint_archive_v5_2",
    "materialize_public_endpoint_inputs_v5_2",
    "validate_public_endpoint_processing_lock_v5_2",
]
