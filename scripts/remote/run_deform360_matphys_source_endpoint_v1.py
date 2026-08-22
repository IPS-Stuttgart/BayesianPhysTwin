#!/usr/bin/env python3
"""Build a disjoint-camera source endpoint for MatPhys surface UQ."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin._portable_contracts import content_id, write_atomic_json
from bayesian_phystwin.matphys_surface_uq_v1 import deterministic_camera_partition

SCHEMA = "bayesian-phystwin.deform360-matphys-source-endpoint-v1"
SOURCE_SELECTED_FRAME_COUNT = 81
RAW_FRAME_COUNT = 76
PREFIX_FRAME_COUNT = 58
EVALUATION_STOP = 76
MINIMUM_SUPPORT_CAMERAS = 3
MANIFEST_FILENAME = "matphys_source_endpoint.json"
EXPECTED_RUNTIME_IDENTITY: dict[str, Any] = {
    "python_version": "3.10.20",
    "numpy_version": "1.26.4",
    "torch_version": "2.4.0+cu121",
    "torchvision_version": "0.19.0+cu121",
    "torch_cuda_version": "12.1",
    "gsplat_version": "1.4.0+pt24cu121",
    "nerfstudio_version": "1.1.5",
    "opencv_python_headless_version": "4.10.0.84",
    "opencv_contrib_python_version": "4.10.0.84",
    "decord_version": "0.6.0",
    "cuda_available": True,
    "cuda_device_capability": [8, 9],
    "gsplat_cuda_backend_available": True,
    "gsplat_camera_model_available": True,
    "nerfstudio_splatfacto_available": True,
    "nerfstudio_gaussian_exporter_available": True,
}
EXPECTED_WARP_REPLAY_RUNTIME: dict[str, str] = {
    "python_version": "3.10.20",
    "numpy_version": "1.26.4",
    "torch_version": "2.4.0+cu121",
    "torch_cuda_version": "12.1",
    "warp_version": "1.16.0",
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _ordinary_file(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_file()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} is invalid",
    )
    return source.resolve(strict=True)


def _ordinary_directory(path: str | Path, *, name: str) -> Path:
    source = Path(path).absolute()
    _require(
        source.is_dir()
        and not source.is_symlink()
        and not any(parent.is_symlink() for parent in source.parents),
        f"{name} is invalid",
    )
    return source.resolve(strict=True)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean_revision(repository: Path, *, include_untracked: bool = True) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            f"--untracked-files={'normal' if include_untracked else 'no'}",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _validate_runtime_identity(observed: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        set(observed) == {*EXPECTED_RUNTIME_IDENTITY, "cuda_device_name"},
        "scoring reconstruction runtime fields changed",
    )
    for key, expected in EXPECTED_RUNTIME_IDENTITY.items():
        _require(observed.get(key) == expected, f"scoring runtime {key} changed")
    device_name = observed.get("cuda_device_name")
    _require(
        isinstance(device_name, str) and bool(device_name),
        "CUDA device name is missing",
    )
    return dict(observed)


def _runtime_identity(device: str) -> dict[str, Any]:
    import torch
    from gsplat.cuda._backend import _C
    from nerfstudio.configs import method_configs
    from nerfstudio.scripts import exporter

    torch_device = torch.device(device)
    observed = {
        "python_version": platform.python_version(),
        "numpy_version": importlib.metadata.version("numpy"),
        "torch_version": importlib.metadata.version("torch"),
        "torchvision_version": importlib.metadata.version("torchvision"),
        "torch_cuda_version": torch.version.cuda,
        "gsplat_version": importlib.metadata.version("gsplat"),
        "nerfstudio_version": importlib.metadata.version("nerfstudio"),
        "opencv_python_headless_version": importlib.metadata.version(
            "opencv-python-headless"
        ),
        "opencv_contrib_python_version": importlib.metadata.version(
            "opencv-contrib-python"
        ),
        "decord_version": importlib.metadata.version("decord"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_capability": list(torch.cuda.get_device_capability(torch_device)),
        "cuda_device_name": torch.cuda.get_device_name(torch_device),
        "gsplat_cuda_backend_available": _C is not None,
        "gsplat_camera_model_available": _C is not None
        and hasattr(_C, "CameraModelType"),
        "nerfstudio_splatfacto_available": (
            "splatfacto" in method_configs.method_configs
        ),
        "nerfstudio_gaussian_exporter_available": hasattr(
            exporter, "ExportGaussianSplat"
        ),
    }
    return _validate_runtime_identity(observed)


def _trim_video(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    *,
    start: int,
    count: int,
) -> None:
    from bayesian_phystwin.deform360_exact_video_cadence import decoded_frame_count

    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select='between(n,{start},{start + count - 1})',setpts=N/(30*TB)",
            "-frames:v",
            str(count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-vsync",
            "cfr",
            str(destination),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    _require(decoded_frame_count(destination) == count, "trimmed frame count changed")


def _trim_timestamps(
    source: Path, destination: Path, *, start: int, count: int
) -> None:
    selected = source.read_text(encoding="utf-8").splitlines()[start : start + count]
    _require(len(selected) == count, "timestamp stream is too short")
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _subset_calibration(
    source: Path, destination: Path, *, cameras: Sequence[str]
) -> None:
    values = np.load(source, allow_pickle=True).item()
    _require(isinstance(values, Mapping), "camera calibration changed format")
    _require(set(cameras) <= set(values), "camera calibration is incomplete")
    np.save(destination, {camera: values[camera] for camera in cameras})


def _write_partial_masks(path: Path, masks: Sequence[np.ndarray]) -> np.ndarray:
    import h5py

    values = np.asarray(masks, dtype=np.uint8)
    _require(
        values.ndim == 3 and values.shape[0] == RAW_FRAME_COUNT,
        "SAM2 returned an incomplete mask tensor",
    )
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )
    return np.count_nonzero(values, axis=(1, 2)) > 0


def _camera_dictionary(path: Path, *, name: str) -> dict[str, np.ndarray]:
    value = np.load(path, allow_pickle=True)
    _require(value.shape == () and value.dtype == object, f"{name} format changed")
    raw = value.item()
    _require(isinstance(raw, Mapping), f"{name} must contain a camera mapping")
    return {str(key): np.asarray(array) for key, array in raw.items()}


def _h5_data(path: Path, *, name: str) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as stream:
        _require(set(stream) == {"data"}, f"{name} member roster changed")
        return np.asarray(stream["data"])


def _write_endpoint_archive(
    *,
    episode: Path,
    camera_id: str,
    raw_endpoint_range: tuple[int, int],
    destination: Path,
) -> dict[str, Any]:
    depth_path = episode / camera_id / "rendered_depth.h5"
    mask_path = episode / camera_id / "mask_refined.h5"
    intrinsics_path = episode / "undistorted_intrinsics.npy"
    extrinsics_path = episode / "extrinsics.npy"
    depth_mm = _h5_data(depth_path, name="rendered depth")
    mask = _h5_data(mask_path, name="object mask")
    _require(
        depth_mm.shape[0] == RAW_FRAME_COUNT
        and depth_mm.dtype == np.uint16
        and mask.shape == depth_mm.shape,
        "endpoint depth or mask contract changed",
    )
    intrinsics = _camera_dictionary(intrinsics_path, name="intrinsics")
    extrinsics = _camera_dictionary(extrinsics_path, name="extrinsics")
    _require(
        camera_id in intrinsics and camera_id in extrinsics,
        "endpoint calibration changed",
    )
    arrays = {
        "frame_indices": np.arange(PREFIX_FRAME_COUNT, EVALUATION_STOP, dtype=np.int64),
        "raw_frame_indices": np.arange(*raw_endpoint_range, dtype=np.int64),
        "depth_m": np.asarray(
            depth_mm[PREFIX_FRAME_COUNT:EVALUATION_STOP], dtype=np.float32
        )
        / 1000.0,
        "object_mask": np.asarray(
            mask[PREFIX_FRAME_COUNT:EVALUATION_STOP] > 0, dtype=np.bool_
        ),
        "intrinsics": np.asarray(intrinsics[camera_id], dtype=np.float64),
        "camera_to_world": np.asarray(extrinsics[camera_id], dtype=np.float64),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"archive_sha256": _sha256_file(destination)}


def _json(path: Path, *, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{name} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--physical-source-episode", type=Path, required=True)
    parser.add_argument("--prefix-manifest", type=Path, required=True)
    parser.add_argument("--deform-prediction-manifest", type=Path, required=True)
    parser.add_argument("--part-feature-manifest", type=Path, required=True)
    parser.add_argument("--matphys-warp-manifest", type=Path, required=True)
    parser.add_argument("--deform360-repository", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--causal4d-repository", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--scoring-camera-id", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _validate_prediction_seals(
    *,
    prefix_path: Path,
    deform_path: Path,
    part_path: Path,
    warp_path: Path,
    scoring_cameras: tuple[str, ...],
) -> dict[str, Any]:
    prefix = _json(prefix_path, name="prefix manifest")
    case_id = prefix.get("case")
    object_id = prefix.get("object_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("prefix case ID is missing")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("prefix object ID is missing")
    boundary_value = prefix.get("information_boundary")
    if not isinstance(boundary_value, Mapping):
        raise ValueError("prefix boundary is missing")
    boundary = cast(Mapping[str, Any], boundary_value)
    _require(
        boundary.get("source_object_frames_after_prefix_read") is False
        and boundary.get("future_dense_reconstruction_read") is False
        and boundary.get("target_metric_read") is False,
        "prefix crossed its prediction boundary",
    )
    action_value = prefix.get("action_window")
    if not isinstance(action_value, Mapping):
        raise ValueError("prefix action window is missing")
    action = cast(Mapping[str, Any], action_value)
    selected = action.get("selected_raw_frame_range_half_open")
    prediction = action.get("prediction_raw_frame_range_half_open")
    prefix_range = action.get("prefix_raw_frame_range_half_open")
    if not (
        isinstance(selected, list)
        and len(selected) == 2
        and all(type(value) is int for value in selected)
        and selected[1] - selected[0] == SOURCE_SELECTED_FRAME_COUNT
    ):
        raise ValueError("selected source window changed")
    selected_range = cast(list[int], selected)
    _require(
        isinstance(prediction, list)
        and prediction == [selected_range[0], selected_range[0] + EVALUATION_STOP],
        "prediction source window changed",
    )
    _require(
        isinstance(prefix_range, list)
        and prefix_range == [selected_range[0], selected_range[0] + PREFIX_FRAME_COUNT],
        "prefix source window changed",
    )

    deform = _json(deform_path, name="DEFORM prediction manifest")
    deform_boundary = deform.get("information_boundary")
    _require(
        deform.get("case") == case_id
        and deform.get("object_id") == object_id
        and deform.get("passed") is True
        and isinstance(deform_boundary, Mapping)
        and deform_boundary.get("outcome_read") is False
        and deform_boundary.get("prediction_hashed_before_future_outcome_scoring")
        is True,
        "DEFORM prediction is not a sealed causal mean",
    )

    part = _json(part_path, name="part-feature manifest")
    part_boundary = part.get("information_boundary")
    selection = part.get("camera_selection")
    _require(
        part.get("case_id") == case_id
        and part.get("target_object_id") == object_id
        and isinstance(part_boundary, Mapping)
        and part_boundary.get("future_object_observations_read") is False
        and part_boundary.get("changes_frozen_deform_mean") is False,
        "part features crossed their causal boundary",
    )
    _require(
        isinstance(selection, Mapping)
        and selection.get("mode") == "explicit-disjoint-provider-panel"
        and isinstance(selection.get("camera_ids"), list),
        "part features do not bind an explicit provider panel",
    )
    selection_mapping = cast(Mapping[str, Any], selection)
    provider = tuple(str(value) for value in selection_mapping["camera_ids"])
    camera_records = prefix.get("camera_records")
    if not isinstance(camera_records, list):
        raise ValueError("prefix camera roster is missing")
    typed_camera_records = cast(list[Mapping[str, Any]], camera_records)
    all_cameras = tuple(
        sorted(
            {
                *(str(row["camera"]) for row in typed_camera_records),
                *scoring_cameras,
            }
        )
    )
    expected_provider, expected_scoring = deterministic_camera_partition(
        all_cameras,
        scoring_camera_ids=scoring_cameras,
    )
    _require(
        tuple(sorted(provider)) == expected_provider
        and scoring_cameras == expected_scoring,
        "provider and scoring camera panels are not the registered partition",
    )

    warp = _json(warp_path, name="MatPhys Warp manifest")
    warp_boundary = warp.get("information_boundary")
    warp_runtime = warp.get("runtime")
    _require(
        warp.get("case_id") == case_id
        and warp.get("target_object_id") == object_id
        and warp.get("passed") is True
        and isinstance(warp_boundary, Mapping)
        and isinstance(warp_runtime, Mapping)
        and all(
            warp_runtime.get(name) == expected
            for name, expected in EXPECTED_WARP_REPLAY_RUNTIME.items()
        )
        and warp_boundary.get("target_future_observations_used") is False
        and warp_boundary.get("target_future_outcomes_opened") is False,
        "MatPhys Warp ensemble is not sealed before scoring",
    )
    return {
        "case_id": case_id,
        "object_id": object_id,
        "raw_start": selected_range[0],
        "provider_camera_ids": list(expected_provider),
        "scoring_camera_ids": list(expected_scoring),
        "prediction_seals": {
            "prefix_manifest_sha256": _sha256_file(prefix_path),
            "deform_prediction_manifest_sha256": _sha256_file(deform_path),
            "part_feature_manifest_sha256": _sha256_file(part_path),
            "matphys_warp_manifest_sha256": _sha256_file(warp_path),
        },
    }


def _validate_protocol(
    path: Path,
    *,
    case_id: str,
    scoring_cameras: tuple[str, ...],
) -> dict[str, Any]:
    protocol = _json(path, name="MatPhys source protocol")
    _require(
        protocol.get("schema")
        == "bayesian-phystwin.matphys-surface-uq-source-protocol-v1"
        and protocol.get("schema_version") == 1
        and protocol.get("protocol_name") == "matphys-surface-uq-source-v1",
        "MatPhys source protocol identity changed",
    )
    source_panel = protocol.get("source_panel")
    camera_partition = protocol.get("camera_partition")
    window = protocol.get("window")
    outcome = protocol.get("outcome")
    covariance = protocol.get("covariance")
    runtime = protocol.get("scoring_reconstruction_runtime")
    boundary = protocol.get("information_boundary")
    _require(
        isinstance(source_panel, Mapping)
        and isinstance(source_panel.get("case_ids"), list)
        and case_id in source_panel["case_ids"]
        and source_panel.get("replacement_allowed") is False,
        "case is absent from the frozen source denominator",
    )
    _require(
        isinstance(camera_partition, Mapping)
        and tuple(camera_partition.get("scoring_camera_ids", ())) == scoring_cameras
        and camera_partition.get("minimum_successful_scoring_cameras")
        == MINIMUM_SUPPORT_CAMERAS,
        "source camera partition changed",
    )
    _require(
        isinstance(window, Mapping)
        and window.get("frame_count") == RAW_FRAME_COUNT
        and window.get("source_selected_frame_count") == SOURCE_SELECTED_FRAME_COUNT
        and window.get("prefix_frame_count") == PREFIX_FRAME_COUNT
        and window.get("prediction_frame_count") == EVALUATION_STOP
        and window.get("evaluation_frame_range_half_open")
        == [PREFIX_FRAME_COUNT, EVALUATION_STOP],
        "source evaluation window changed",
    )
    _require(
        isinstance(outcome, Mapping)
        and outcome.get("robot_state_required_for_scoring_reconstruction") is False
        and outcome.get("urdf_gripper_mask_used") is False
        and outcome.get("gripper_pixels_excluded") is False,
        "source outcome reconstruction changed",
    )
    _require(
        isinstance(covariance, Mapping)
        and covariance.get("official_warp_version") == "1.16.0",
        "source covariance runtime changed",
    )
    typed_covariance = cast(Mapping[str, Any], covariance)
    _require(
        typed_covariance.get("replay_runtime") == EXPECTED_WARP_REPLAY_RUNTIME,
        "source covariance replay identity changed",
    )
    _require(
        isinstance(runtime, Mapping)
        and dict(runtime) == EXPECTED_RUNTIME_IDENTITY,
        "source scoring runtime changed",
    )
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("provider_and_scoring_cameras_disjoint") is True
        and boundary.get("robot_state_read_for_scoring_reconstruction") is False
        and boundary.get("target_or_confirmation_data_read") is False
        and boundary.get("held_v8_artifacts_accessed") is False
        and boundary.get("frozen_deform_results_changed") is False,
        "source protocol boundary changed",
    )
    return protocol


def _copy_source_window(
    *,
    source_episode: Path,
    destination_episode: Path,
    cameras: tuple[str, ...],
    raw_start: int,
    ffmpeg: Path,
) -> None:
    destination_episode.mkdir(parents=True)
    _subset_calibration(
        source_episode / "undistorted_intrinsics.npy",
        destination_episode / "undistorted_intrinsics.npy",
        cameras=cameras,
    )
    _subset_calibration(
        source_episode / "extrinsics.npy",
        destination_episode / "extrinsics.npy",
        cameras=cameras,
    )
    for camera in cameras:
        source_camera = source_episode / camera
        target_camera = destination_episode / camera
        target_camera.mkdir()
        _trim_video(
            ffmpeg,
            _ordinary_file(source_camera / "undistorted.mp4", name=f"{camera} RGB"),
            target_camera / "undistorted.mp4",
            start=raw_start,
            count=RAW_FRAME_COUNT,
        )
        _trim_timestamps(
            _ordinary_file(
                source_camera / "aligned_timestamps.txt",
                name=f"{camera} timestamps",
            ),
            target_camera / "aligned_timestamps.txt",
            start=raw_start,
            count=RAW_FRAME_COUNT,
        )
        metadata = source_camera / "metadata.json"
        if metadata.is_file():
            shutil.copy2(metadata, target_camera / "metadata.json")
    alignment = source_episode / "alignment.json"
    if alignment.is_file():
        shutil.copy2(alignment, destination_episode / "alignment.json")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = _ordinary_directory(args.repo, name="BayesianPhysTwin repository")
    code_revision = _clean_revision(repository, include_untracked=False)
    protocol_path = _ordinary_file(args.protocol, name="MatPhys source protocol")
    source_episode = _ordinary_directory(
        args.physical_source_episode, name="physical source episode"
    )
    prefix_path = _ordinary_file(args.prefix_manifest, name="prefix manifest")
    deform_path = _ordinary_file(
        args.deform_prediction_manifest, name="DEFORM prediction manifest"
    )
    part_path = _ordinary_file(args.part_feature_manifest, name="part-feature manifest")
    warp_path = _ordinary_file(args.matphys_warp_manifest, name="MatPhys Warp manifest")
    scoring = tuple(sorted(str(value) for value in args.scoring_camera_id))
    _require(
        len(scoring) == len(set(scoring)) and len(scoring) >= MINIMUM_SUPPORT_CAMERAS,
        "scoring camera panel is invalid",
    )
    identity = _validate_prediction_seals(
        prefix_path=prefix_path,
        deform_path=deform_path,
        part_path=part_path,
        warp_path=warp_path,
        scoring_cameras=scoring,
    )
    _validate_protocol(
        protocol_path,
        case_id=str(identity["case_id"]),
        scoring_cameras=scoring,
    )
    deform360_repository = _ordinary_directory(
        args.deform360_repository, name="Deform360 repository"
    )
    sam2_repository = _ordinary_directory(args.sam2_repository, name="SAM2 repository")
    causal4d_repository = _ordinary_directory(
        args.causal4d_repository, name="Causal4D repository"
    )
    checkpoint = _ordinary_file(args.sam2_checkpoint, name="SAM2 checkpoint")
    _require(
        _clean_revision(deform360_repository, include_untracked=False)
        == "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317",
        "Deform360 revision changed",
    )
    _require(
        _clean_revision(sam2_repository, include_untracked=False)
        == "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        "SAM2 revision changed",
    )
    _require(
        _clean_revision(causal4d_repository, include_untracked=False)
        == "50e3682a5dbf976b20cc9115b6e7a975d0144ea5",
        "Causal4D selector revision changed",
    )
    _require(
        _sha256_file(checkpoint)
        == "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        "SAM2 checkpoint changed",
    )
    runtime_identity = _runtime_identity(str(args.device))
    run_identity = {
        "schema": SCHEMA,
        "schema_version": 1,
        **identity,
        "implementation_revision": code_revision,
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256_file(protocol_path),
        },
        "dependencies": {
            "deform360_revision": "0fe36f0b7a7a917ba62b5f8cee707299a9a4a317",
            "sam2_revision": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
            "causal4d_revision": "50e3682a5dbf976b20cc9115b6e7a975d0144ea5",
            "sam2_checkpoint_sha256": _sha256_file(checkpoint),
        },
        "scoring_reconstruction_runtime": runtime_identity,
        "window": {
            "frame_count": RAW_FRAME_COUNT,
            "prefix_frame_count": PREFIX_FRAME_COUNT,
            "evaluation_frame_range_half_open": [PREFIX_FRAME_COUNT, EVALUATION_STOP],
        },
        "information_boundary": {
            "prediction_seals_verified_before_future_decode": True,
            "provider_scoring_camera_overlap": False,
            "robot_state_read_for_scoring_reconstruction": False,
            "urdf_gripper_mask_used": False,
            "source_suffix_opened": not args.preflight_only,
            "target_or_confirmation_data_read": False,
            "held_v8_artifacts_accessed": False,
            "deform_mean_changed": False,
            "replacement_allowed": False,
        },
    }
    if args.preflight_only:
        print(json.dumps({**run_identity, "artifact_id": content_id(run_identity)}))
        return 0

    output = Path(args.output_dir).absolute()
    _require(not output.exists(), "source endpoint output already exists")
    scratch = output.parent / f".{output.name}.incomplete"
    _require(not scratch.exists(), "source endpoint scratch already exists")
    scratch.mkdir(parents=True)
    failure_stage = "copy-source-window"
    camera_records: list[dict[str, Any]] = []
    support_by_camera: dict[str, np.ndarray] = {}
    try:
        episode = scratch / "episode_0000"
        _copy_source_window(
            source_episode=source_episode,
            destination_episode=episode,
            cameras=scoring,
            raw_start=int(identity["raw_start"]),
            ffmpeg=Path("/usr/bin/ffmpeg").resolve(strict=True),
        )
        sys.path.insert(0, str(causal4d_repository / "src"))
        sys.path.insert(0, str(sam2_repository))
        from causal4d_public.deform360_object_sam2 import (
            DeformableObjectSam2VideoPredictor,
        )

        predictor = DeformableObjectSam2VideoPredictor(
            sam2_repository, checkpoint, device=args.device
        )
        failure_stage = "sam2-mask-generation"
        try:
            for camera in scoring:
                video = episode / camera / "undistorted.mp4"
                try:
                    initial_mask, initialization = predictor.select_initial_mask(video)
                    propagated = list(
                        predictor.segment_from_initial_mask(
                            video,
                            initial_mask,
                            initialization={
                                "policy": "matphys-uq-disjoint-source-frame-zero-v1",
                                "source_frame_index": 0,
                                "selection": initialization,
                            },
                        )
                    )
                    _require(
                        [index for index, _mask in propagated]
                        == list(range(RAW_FRAME_COUNT)),
                        "SAM2 frame roster changed",
                    )
                    mask_path = episode / camera / "mask_refined.h5"
                    support = _write_partial_masks(
                        mask_path, [mask for _index, mask in propagated]
                    )
                    support_by_camera[camera] = support
                    camera_records.append(
                        {
                            "camera_id": camera,
                            "status": "success",
                            "mask_sha256": _sha256_file(mask_path),
                            "nonempty_frame_count": int(np.sum(support)),
                            "initialization": initialization,
                        }
                    )
                except BaseException as error:
                    camera_records.append(
                        {
                            "camera_id": camera,
                            "status": "technical_failure",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        finally:
            predictor.close()
        _require(
            len(support_by_camera) >= MINIMUM_SUPPORT_CAMERAS,
            "fewer than three scoring cameras produced masks",
        )
        frame_support = np.sum(np.stack(list(support_by_camera.values())), axis=0)
        _require(
            np.all(frame_support >= MINIMUM_SUPPORT_CAMERAS),
            "fewer than three scoring cameras support at least one frame",
        )
        successful = tuple(sorted(support_by_camera))

        failure_stage = "splatfacto-reconstruction"
        sys.path.insert(0, str(deform360_repository))
        from deform360.processing import depth_stage, reconstruct_stage

        original_visual_hull = reconstruct_stage.visual_hull_points

        def locked_visual_hull(*call_args: object, **call_kwargs: object) -> Any:
            call_kwargs["min_points"] = 512
            return original_visual_hull(*call_args, **call_kwargs)

        reconstruct_stage.visual_hull_points = locked_visual_hull
        try:
            splats = reconstruct_stage.process_reconstruction_episode(
                scratch,
                0,
                cameras=successful,
                first_frame_iterations=500,
                warm_start_iterations=250,
                cube_half_extent_m=0.5,
                voxel_resolution=120,
                overwrite=True,
                keep_scratch=False,
            )
        finally:
            reconstruct_stage.visual_hull_points = original_visual_hull
        _require(set(splats) == set(range(RAW_FRAME_COUNT)), "splats are incomplete")
        failure_stage = "depth-reconstruction"
        depths = depth_stage.process_depth_episode(
            scratch, 0, cameras=successful, overwrite=True, preview=False
        )
        _require(set(depths) == set(successful), "endpoint depth panel changed")
        failure_stage = "endpoint-archive"
        endpoint_records: list[dict[str, Any]] = []
        raw_start = int(identity["raw_start"])
        raw_endpoint = (raw_start + PREFIX_FRAME_COUNT, raw_start + EVALUATION_STOP)
        for camera in successful:
            relative = Path("endpoint_archives") / f"{camera}.npz"
            record = _write_endpoint_archive(
                episode=episode,
                camera_id=camera,
                raw_endpoint_range=raw_endpoint,
                destination=scratch / relative,
            )
            endpoint_records.append(
                {
                    "camera_id": camera,
                    "path": relative.as_posix(),
                    "sha256": record["archive_sha256"],
                }
            )
        result = {
            **run_identity,
            "status": "success",
            "successful_scoring_camera_ids": list(successful),
            "robot_state_read_for_scoring_reconstruction": False,
            "urdf_gripper_mask_used": False,
            "gripper_pixels_excluded": False,
            "frame_support_camera_count": frame_support.astype(int).tolist(),
            "camera_records": camera_records,
            "endpoint_archives": endpoint_records,
        }
        result["artifact_id"] = content_id(result)
        write_atomic_json(result, scratch / MANIFEST_FILENAME, overwrite=False)
        scratch.rename(output)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        shutil.rmtree(scratch, ignore_errors=True)
        output.mkdir(parents=True, exist_ok=False)
        failure = {
            **run_identity,
            "status": "retained-source-technical-failure",
            "failure_stage": failure_stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "camera_records": camera_records,
            "successful_mask_camera_count": len(support_by_camera),
        }
        failure["artifact_id"] = content_id(failure)
        write_atomic_json(failure, output / MANIFEST_FILENAME, overwrite=False)
        print(json.dumps(failure, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
