#!/usr/bin/env python3
"""Build one prefix-only V14 Deform360 geometry and physical preflight."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from bayesian_phystwin.deform360_causal_response_direct_depth_assets import (
    PREFIX_FRAME_COUNT,
    canonical_sha256,
    load_v14_asset_protocol,
    validate_v14_staged_window,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_prefix_geometry import (
    GEOMETRY_CONTRACT,
    GEOMETRY_MANIFEST_KIND,
    GEOMETRY_PROTOCOL_ID,
    GEOMETRY_RESULT_KIND,
    load_v14_prefix_geometry_protocol,
    projected_seed_support,
    validate_v14_geometry_mask_input,
)
from bayesian_phystwin.deform360_exact_video_cadence import decoded_frame_count
from bayesian_phystwin.deform360_object_exclusion import file_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact: {path}") from error
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository has uncommitted files: {repository}")
    return revision


def _trim_lines(source: Path, destination: Path, count: int) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    _require(len(lines) >= count, f"timestamp stream is too short: {source}")
    destination.write_text(
        "\n".join(lines[:count]) + "\n",
        encoding="utf-8",
    )


def _subset_calibration(
    source: Path,
    destination: Path,
    cameras: tuple[str, ...],
) -> dict[str, np.ndarray]:
    values = np.load(source, allow_pickle=True).item()
    _require(
        isinstance(values, dict) and set(cameras).issubset(values),
        f"calibration lacks cameras: {source}",
    )
    selected = {camera: np.asarray(values[camera]) for camera in cameras}
    np.save(destination, selected)
    return selected


def _validate_stage_inputs(
    stage: Mapping[str, Any],
    stage_episode: Path,
) -> None:
    outputs = stage.get("outputs_sha256")
    _require(isinstance(outputs, Mapping), "V14 staged outputs are missing")
    fixed = {
        "intrinsics": stage_episode / "undistorted_intrinsics.npy",
        "extrinsics": stage_episode / "extrinsics.npy",
        "robot": stage_episode / "robot" / "robot.npz",
    }
    _require(
        all(
            path.is_file() and file_sha256(path) == outputs.get(role)
            for role, path in fixed.items()
        ),
        "V14 staged fixed inputs changed",
    )
    camera_records = stage.get("camera_records")
    _require(isinstance(camera_records, list), "V14 staged cameras are missing")
    for row in camera_records:
        camera = str(row["camera"])
        root = stage_episode / camera
        timestamps = root / "aligned_timestamps.txt"
        metadata = root / "metadata.json"
        _require(
            timestamps.is_file()
            and file_sha256(timestamps) == row["timestamps_sha256"]
            and (
                row.get("metadata_sha256") is None
                or (
                    metadata.is_file()
                    and file_sha256(metadata) == row["metadata_sha256"]
                )
            ),
            f"V14 staged camera sidecars changed: {camera}",
        )
    tactile_records = stage.get("tactile_records")
    _require(
        isinstance(tactile_records, list) and tactile_records,
        "V14 staged tactile panel is missing",
    )
    for row in tactile_records:
        sensor = str(row["sensor"])
        array = stage_episode / sensor / "synced_tactile.npy"
        _require(
            row.get("frame_count") == 81
            and array.is_file()
            and file_sha256(array) == row["array_sha256"],
            f"V14 staged tactile input changed: {sensor}",
        )
        for name in ("metadata.json", "alignment.json"):
            _require(
                (stage_episode / sensor / name).is_file(),
                f"V14 staged tactile sidecar is missing: {sensor}/{name}",
            )


def _validate_deform360_sources(
    repository: Path,
    runtime: Mapping[str, Any],
) -> None:
    _require_clean_repository(repository)
    _require(
        _git_revision(repository) == runtime["deform360_revision"],
        "Deform360 revision changed",
    )
    sources = runtime["deform360_source_sha256"]
    expected = {
        "reconstruct_stage": (
            repository / "deform360" / "processing" / "reconstruct_stage.py"
        ),
        "urdf_render": (
            repository / "deform360" / "processing" / "urdf_render.py"
        ),
        "depth_stage": (
            repository / "deform360" / "processing" / "depth_stage.py"
        ),
        "pcd_stage": repository / "deform360" / "processing" / "pcd_stage.py",
    }
    _require(
        set(sources) == set(expected)
        and all(file_sha256(path) == sources[name] for name, path in expected.items()),
        "Deform360 prefix geometry source changed",
    )


def _validate_runtime(runtime: Mapping[str, Any]) -> dict[str, str]:
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    _require(
        bool(path_entries)
        and Path(path_entries[0]).resolve()
        == Path(runtime["required_path_prefix"]).resolve(),
        "CUDA toolkit is not the first runtime PATH entry",
    )
    nvcc = shutil.which("nvcc")
    _require(
        nvcc is not None
        and Path(nvcc).resolve() == Path(runtime["nvcc_path"]).resolve(),
        "frozen nvcc is unavailable",
    )
    nvcc_output = subprocess.run(
        [nvcc, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(
        runtime["nvcc_version_line"] in nvcc_output,
        "nvcc version changed",
    )
    import gsplat  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from gsplat.cuda._backend import _C  # noqa: PLC0415

    _require(
        gsplat.__version__ == runtime["gsplat_version"]
        and torch.__version__ == runtime["torch_version"]
        and torch.version.cuda == runtime["torch_cuda_version"]
        and sys.version.split()[0] == runtime["python_version"],
        "prefix geometry Python, torch, CUDA, or gsplat runtime changed",
    )
    _require(_C is not None, "gsplat CUDA backend is disabled")
    extension = Path(_C.__file__).resolve()
    _require(
        str(extension) == runtime["gsplat_extension_path"]
        and file_sha256(extension) == runtime["gsplat_extension_sha256"],
        "gsplat CUDA extension changed",
    )
    probe = str(_C.CameraModelType.PINHOLE)
    _require(
        probe == runtime["required_backend_probe"],
        "gsplat camera-model probe changed",
    )
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torch_cuda_version": str(torch.version.cuda),
        "gsplat_version": gsplat.__version__,
        "nvcc_path": str(Path(nvcc).resolve()),
        "gsplat_extension_path": str(extension),
        "gsplat_extension_sha256": file_sha256(extension),
        "backend_probe": probe,
    }


def _prepare_prefix_episode(
    *,
    stage: Mapping[str, Any],
    stage_episode: Path,
    mask: Mapping[str, Any],
    mask_episode: Path,
    output_episode: Path,
    robot_module: Any,
) -> tuple[tuple[str, ...], dict[str, np.ndarray], dict[str, np.ndarray]]:
    cameras = tuple(
        sorted(
            str(row["camera"])
            for row in mask["camera_records"]
            if row.get("status") == "success"
        )
    )
    _require(len(cameras) >= 8, "too few V14 prefix geometry cameras")
    output_episode.mkdir(parents=True)
    intrinsics = _subset_calibration(
        stage_episode / "undistorted_intrinsics.npy",
        output_episode / "undistorted_intrinsics.npy",
        cameras,
    )
    extrinsics = _subset_calibration(
        stage_episode / "extrinsics.npy",
        output_episode / "extrinsics.npy",
        cameras,
    )
    robot = robot_module.load_robot_state(stage_episode / "robot" / "robot.npz")
    _require(
        len(robot.actions) >= PREFIX_FRAME_COUNT
        and len(robot.T_worlds) >= PREFIX_FRAME_COUNT
        and len(robot.openings) >= PREFIX_FRAME_COUNT,
        "V14 staged robot stream is too short",
    )
    prefix_robot = robot_module.RobotState(
        actions=robot.actions[:PREFIX_FRAME_COUNT],
        T_worlds=robot.T_worlds[:PREFIX_FRAME_COUNT],
        openings=robot.openings[:PREFIX_FRAME_COUNT],
        bimanual=robot.bimanual,
    )
    robot_module.save_robot_state(
        output_episode / "robot" / "robot.npz",
        prefix_robot,
    )
    stage_by_camera = {
        str(row["camera"]): row for row in stage["camera_records"]
    }
    for camera in cameras:
        source_mask = mask_episode / camera
        source_stage = stage_episode / camera
        destination = output_episode / camera
        destination.mkdir()
        shutil.copy2(source_mask / "prefix.mp4", destination / "undistorted.mp4")
        shutil.copy2(source_mask / "mask_refined.h5", destination / "mask_refined.h5")
        _trim_lines(
            source_stage / "aligned_timestamps.txt",
            destination / "aligned_timestamps.txt",
            PREFIX_FRAME_COUNT,
        )
        metadata_sha256 = stage_by_camera[camera].get("metadata_sha256")
        if metadata_sha256 is not None:
            metadata = source_stage / "metadata.json"
            _require(
                file_sha256(metadata) == metadata_sha256,
                f"V14 staged camera metadata changed: {camera}",
            )
            shutil.copy2(metadata, destination / "metadata.json")
        _require(
            decoded_frame_count(destination / "undistorted.mp4")
            == PREFIX_FRAME_COUNT,
            f"V14 prefix video cadence changed: {camera}",
        )
    for row in stage["tactile_records"]:
        sensor = str(row["sensor"])
        source = stage_episode / sensor
        destination = output_episode / sensor
        destination.mkdir()
        values = np.load(source / "synced_tactile.npy", allow_pickle=False)
        _require(
            values.ndim == 3 and len(values) >= PREFIX_FRAME_COUNT,
            f"V14 staged tactile stream is too short: {sensor}",
        )
        np.save(destination / "synced_tactile.npy", values[:PREFIX_FRAME_COUNT])
        for name in ("metadata.json", "alignment.json"):
            shutil.copy2(source / name, destination / name)
    return cameras, intrinsics, extrinsics


def _frame_zero_depth(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as stream:
        _require("data" in stream, f"depth archive lacks data: {path}")
        values = stream["data"]
        _require(
            values.shape[0] == PREFIX_FRAME_COUNT
            and values.dtype == np.dtype(np.uint16),
            f"depth archive violates the V14 metric contract: {path}",
        )
        return np.asarray(values[0], dtype=np.uint16)


def _h5_frame_count(path: Path, *, dtype: np.dtype[Any] | None = None) -> int:
    with h5py.File(path, "r") as stream:
        _require("data" in stream, f"HDF5 archive lacks data: {path}")
        values = stream["data"]
        _require(
            len(values.shape) >= 1
            and (dtype is None or values.dtype == dtype),
            f"HDF5 archive violates the V14 contract: {path}",
        )
        return int(values.shape[0])


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    namespace: bytes,
) -> str:
    _require(not path.exists(), f"refusing to replace V14 geometry artifact: {path}")
    value = dict(payload)
    value["artifact_sha256"] = canonical_sha256(
        value,
        namespace=namespace,
        digest_key="artifact_sha256",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _require(not temporary.exists(), f"temporary V14 geometry artifact exists: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return str(value["artifact_sha256"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--method-protocol", type=Path, required=True)
    parser.add_argument("--asset-protocol", type=Path, required=True)
    parser.add_argument("--geometry-protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--stage-result", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--mask-result", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--candidate-rank", type=int, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository = args.repo.resolve()
    code_revision = _require_clean_repository(repository)
    method_path = args.method_protocol.resolve()
    method = _read_json(method_path)
    asset_path = args.asset_protocol.resolve()
    asset = load_v14_asset_protocol(asset_path)
    geometry_path = args.geometry_protocol.resolve()
    geometry_protocol = load_v14_prefix_geometry_protocol(geometry_path)
    _require(
        method.get("protocol_id") == asset["parent_method"]["protocol_id"]
        and method.get("config_sha256")
        == asset["parent_method"]["config_sha256"]
        and file_sha256(method_path) == asset["parent_method"]["file_sha256"],
        "V14 geometry builder binds another method",
    )
    parent = geometry_protocol["parent_prefix_assets"]
    _require(
        parent["protocol_id"] == asset["protocol_id"]
        and parent["config_sha256"] == asset["config_sha256"]
        and parent["file_sha256"] == file_sha256(asset_path),
        "V14 geometry protocol binds another prefix-asset lock",
    )
    implementation = geometry_protocol["implementation_file_sha256"]
    _require(
        file_sha256(
            repository
            / "src"
            / "bayesian_phystwin"
            / "deform360_causal_response_prefix_geometry.py"
        )
        == implementation["geometry_module"]
        and file_sha256(
            repository
            / "scripts"
            / "remote"
            / "build_deform360_causal_response_direct_depth_v14_prefix_geometry.py"
        )
        == implementation["geometry_builder"],
        "V14 prefix geometry implementation changed",
    )
    queue_path = args.queue.resolve()
    queue = validate_v14_staging_queue(queue_path)
    _require(
        queue["queue_sha256"] == asset["staging"]["queue_sha256"]
        and file_sha256(queue_path) == asset["staging"]["queue_file_sha256"],
        "V14 geometry protocol binds another staging queue",
    )
    rank = args.candidate_rank
    _require(
        1 <= rank <= len(queue["candidates"]),
        "V14 geometry rank is outside the frozen queue",
    )
    candidate = queue["candidates"][rank - 1]
    object_id = str(candidate["object_id"])
    episode_id = int(candidate["episode_id"])
    stage_episode = (
        args.stage_root.resolve()
        / object_id
        / f"episode_{episode_id:04d}"
    )
    stage_path = args.stage_result.resolve()
    stage, _ = validate_v14_staged_window(
        stage_path,
        protocol=method,
        asset_protocol=asset,
        queue=queue,
        queue_rank=rank,
        stage_episode=stage_episode,
    )
    _validate_stage_inputs(stage, stage_episode)
    mask_episode = (
        args.mask_root.resolve()
        / object_id
        / f"episode_{episode_id:04d}"
    )
    mask_path = args.mask_result.resolve()
    mask = validate_v14_geometry_mask_input(
        mask_path,
        protocol=geometry_protocol,
        asset_protocol=asset,
        mask_episode=mask_episode,
        queue_rank=rank,
    )
    _require(
        mask["object_hash"] == stage["object_hash"]
        and mask["case_hash"] == stage["case_hash"],
        "V14 geometry mask and staged case disagree",
    )
    deform360_repository = args.deform360_repo.resolve()
    runtime_contract = geometry_protocol["runtime"]
    _validate_deform360_sources(deform360_repository, runtime_contract)
    runtime_probe = _validate_runtime(runtime_contract)
    sys.path.insert(0, str(deform360_repository))
    from deform360 import robot as robot_module  # noqa: PLC0415
    from deform360.processing import (  # noqa: PLC0415
        depth_stage,
        pcd_stage,
        reconstruct_stage,
        urdf_render,
    )

    geometry = geometry_protocol["geometry"]
    _require(
        pcd_stage.SEED_POINT_COUNT == geometry["seed_point_count"]
        and pcd_stage.CROP_HALF_EXTENT_M
        == geometry["seed_crop_half_extent_m"],
        "Deform360 frame-zero seed constants changed",
    )
    output_object = args.output_root.resolve() / object_id
    result_path = args.result.resolve()
    _require(
        not output_object.exists() and not result_path.exists(),
        "V14 prefix geometry output or result already exists",
    )
    scratch_object = output_object.with_name(
        f".{output_object.name}.incomplete-{os.getpid()}"
    )
    _require(
        not scratch_object.exists(),
        f"V14 prefix geometry scratch exists: {scratch_object}",
    )
    scratch_episode = scratch_object / f"episode_{episode_id:04d}"
    result_base: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": GEOMETRY_RESULT_KIND,
        "contract": GEOMETRY_CONTRACT,
        "protocol_id": GEOMETRY_PROTOCOL_ID,
        "geometry_protocol_config_sha256": geometry_protocol["config_sha256"],
        "geometry_protocol_file_sha256": file_sha256(geometry_path),
        "queue_sha256": queue["queue_sha256"],
        "queue_rank": rank,
        "object_hash": stage["object_hash"],
        "case_hash": stage["case_hash"],
        "window_stage_artifact_sha256": stage["artifact_sha256"],
        "window_stage_file_sha256": file_sha256(stage_path),
        "prefix_mask_artifact_sha256": mask["artifact_sha256"],
        "prefix_mask_file_sha256": file_sha256(mask_path),
        "code_revision": code_revision,
    }
    try:
        cameras, intrinsics, extrinsics = _prepare_prefix_episode(
            stage=stage,
            stage_episode=stage_episode,
            mask=mask,
            mask_episode=mask_episode,
            output_episode=scratch_episode,
            robot_module=robot_module,
        )
        original_visual_hull = reconstruct_stage.visual_hull_points

        def strict_visual_hull(*call_args: object, **call_kwargs: object) -> Any:
            call_kwargs["min_points"] = int(
                geometry["minimum_visual_hull_points"]
            )
            return original_visual_hull(*call_args, **call_kwargs)

        reconstruct_stage.visual_hull_points = strict_visual_hull
        try:
            splats = reconstruct_stage.process_reconstruction_episode(
                scratch_object,
                episode_id,
                cameras=cameras,
                first_frame_iterations=int(geometry["first_frame_iterations"]),
                warm_start_iterations=int(geometry["warm_start_iterations"]),
                cube_half_extent_m=float(geometry["cube_half_extent_m"]),
                voxel_resolution=int(geometry["voxel_resolution"]),
                overwrite=True,
                keep_scratch=False,
            )
        finally:
            reconstruct_stage.visual_hull_points = original_visual_hull
        _require(
            set(splats) == set(range(PREFIX_FRAME_COUNT)),
            "V14 prefix reconstruction is incomplete",
        )
        gripper_masks = urdf_render.process_gripper_masks_episode(
            scratch_object,
            episode_id,
            cameras=cameras,
            overwrite=True,
        )
        depths = depth_stage.process_depth_episode(
            scratch_object,
            episode_id,
            cameras=cameras,
            overwrite=True,
            preview=False,
        )
        _require(
            set(gripper_masks) == set(depths) == set(cameras),
            "V14 prefix geometry stages used different camera panels",
        )
        splat_path = Path(splats[0]).resolve()
        points, colors = pcd_stage.seed_points_from_splat(
            splat_path,
            crop_half_extent_m=float(geometry["seed_crop_half_extent_m"]),
            seed_count=int(geometry["seed_point_count"]),
            rng_seed=int(geometry["seed_rng"]),
        )
        node_count = len(points)
        _require(
            int(geometry["minimum_physical_node_count"])
            <= node_count
            <= int(geometry["maximum_physical_node_count"])
            and np.all(np.isfinite(points))
            and np.all(np.isfinite(colors)),
            "V14 frame-zero geometry is physically inadmissible",
        )
        start_ply = reconstruct_stage.write_seed_ply(
            scratch_episode / "start_obj_pcd.ply",
            points,
            colors,
        )
        depth_zero = {
            camera: _frame_zero_depth(
                scratch_episode / camera / "rendered_depth.h5"
            )
            for camera in cameras
        }
        support_count, support_by_camera = projected_seed_support(
            points,
            intrinsics_by_camera=intrinsics,
            camera_to_world_by_camera=extrinsics,
            depth_mm_by_camera=depth_zero,
            depth_tolerance_m=float(geometry["support_depth_tolerance_m"]),
        )
        camera_records = [
            {
                "camera": camera,
                "rgb_frame_count": decoded_frame_count(
                    scratch_episode / camera / "undistorted.mp4"
                ),
                "mask_frame_count": _h5_frame_count(
                    scratch_episode / camera / "mask_refined.h5",
                    dtype=np.dtype(np.uint8),
                ),
                "depth_frame_count": _h5_frame_count(
                    scratch_episode / camera / "rendered_depth.h5",
                    dtype=np.dtype(np.uint16),
                ),
                "gripper_mask_frame_count": _h5_frame_count(
                    scratch_episode / camera / "rendered_urdf.h5"
                ),
                "frame_zero_projected_support_count": support_by_camera[camera],
            }
            for camera in cameras
        ]
        _require(
            all(
                row["rgb_frame_count"]
                == row["mask_frame_count"]
                == row["depth_frame_count"]
                == row["gripper_mask_frame_count"]
                == PREFIX_FRAME_COUNT
                for row in camera_records
            ),
            "V14 prefix geometry camera stages have inconsistent lengths",
        )
        outputs_sha256: dict[str, Any] = {
            "intrinsics": file_sha256(
                scratch_episode / "undistorted_intrinsics.npy"
            ),
            "extrinsics": file_sha256(scratch_episode / "extrinsics.npy"),
            "robot": file_sha256(scratch_episode / "robot" / "robot.npz"),
            "frame_zero_splat": file_sha256(splat_path),
            "frame_zero_points": file_sha256(start_ply),
            "video_by_camera": {
                camera: file_sha256(
                    scratch_episode / camera / "undistorted.mp4"
                )
                for camera in cameras
            },
            "mask_by_camera": {
                camera: file_sha256(
                    scratch_episode / camera / "mask_refined.h5"
                )
                for camera in cameras
            },
            "depth_by_camera": {
                camera: file_sha256(
                    scratch_episode / camera / "rendered_depth.h5"
                )
                for camera in cameras
            },
            "depth_metadata_by_camera": {
                camera: file_sha256(
                    scratch_episode / camera / "rendered_depth.meta.json"
                )
                for camera in cameras
            },
            "gripper_mask_by_camera": {
                camera: file_sha256(
                    scratch_episode / camera / "rendered_urdf.h5"
                )
                for camera in cameras
            },
            "tactile_by_sensor": {
                str(row["sensor"]): file_sha256(
                    scratch_episode
                    / str(row["sensor"])
                    / "synced_tactile.npy"
                )
                for row in stage["tactile_records"]
            },
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": GEOMETRY_MANIFEST_KIND,
            "contract": GEOMETRY_CONTRACT,
            "protocol_id": GEOMETRY_PROTOCOL_ID,
            "geometry_protocol_config_sha256": geometry_protocol["config_sha256"],
            "geometry_protocol_file_sha256": file_sha256(geometry_path),
            "queue_sha256": queue["queue_sha256"],
            "queue_rank": rank,
            "object_hash": stage["object_hash"],
            "case_hash": stage["case_hash"],
            "status": "ready_for_physical_preflight",
            "code_revision": code_revision,
            "deform360_revision": runtime_contract["deform360_revision"],
            "runtime": runtime_probe,
            "prefix_frame_count": PREFIX_FRAME_COUNT,
            "maximum_object_observation_frame": PREFIX_FRAME_COUNT - 1,
            "cameras": list(cameras),
            "camera_count": len(cameras),
            "camera_records": camera_records,
            "calibration_valid": True,
            "physical_node_count": node_count,
            "frame_zero_projected_support_count": support_count,
            "frame_zero_projected_support_by_camera": support_by_camera,
            "window_stage_artifact_sha256": stage["artifact_sha256"],
            "window_stage_file_sha256": file_sha256(stage_path),
            "prefix_mask_artifact_sha256": mask["artifact_sha256"],
            "prefix_mask_file_sha256": file_sha256(mask_path),
            "outputs_sha256": outputs_sha256,
            "information_boundary": {
                "object_observation_frame_range_inclusive": [
                    0,
                    PREFIX_FRAME_COUNT - 1,
                ],
                "future_object_observation_read": False,
                "future_identity_or_metric_read": False,
                "target_object_or_outcome_read": False,
                "held_v8_artifact_or_process_access": False,
                "frame_zero_geometry_is_a_model_prediction": False,
            },
        }
        manifest_path = scratch_object / "prefix_geometry_manifest.json"
        manifest_sha256 = _write_json_atomic(
            manifest_path,
            manifest,
            namespace=(
                b"deform360-causal-response-direct-depth-prefix-geometry-v14\0"
            ),
        )
        manifest_file_sha256 = file_sha256(manifest_path)
        output_object.parent.mkdir(parents=True, exist_ok=True)
        scratch_object.rename(output_object)
        result = {
            **result_base,
            "status": "ready_for_source_lock",
            "geometry_manifest_artifact_sha256": manifest_sha256,
            "geometry_manifest_file_sha256": manifest_file_sha256,
            "successful_camera_count": len(cameras),
            "physical_node_count": node_count,
            "frame_zero_projected_support_count": support_count,
            "information_boundary": {
                "maximum_object_observation_frame": PREFIX_FRAME_COUNT - 1,
                "future_object_observation_read": False,
                "future_identity_or_metric_read": False,
                "target_object_or_outcome_read": False,
                "held_v8_artifact_or_process_access": False,
            },
        }
    except Exception as error:
        shutil.rmtree(scratch_object, ignore_errors=True)
        result = {
            **result_base,
            "status": "technical_prelock_failure",
            "failure": {
                "type": type(error).__name__,
                "reason": "prefix-geometry-or-physical-preflight-failed",
            },
            "information_boundary": {
                "maximum_object_observation_frame": PREFIX_FRAME_COUNT - 1,
                "future_object_observation_read": False,
                "future_identity_or_metric_read": False,
                "target_object_or_outcome_read": False,
                "held_v8_artifact_or_process_access": False,
                "technical_failure_is_a_model_prediction": False,
            },
        }
    _write_json_atomic(
        result_path,
        result,
        namespace=(
            b"deform360-causal-response-direct-depth-prefix-geometry-result-v14\0"
        ),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "queue_rank": rank,
                "object_hash": stage["object_hash"],
                "case_hash": stage["case_hash"],
                "result": str(result_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
