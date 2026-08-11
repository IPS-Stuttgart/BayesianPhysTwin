#!/usr/bin/env python3
"""Build frozen public Deform360 endpoint geometry for one sealed v5.2 object."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import h5py
import numpy as np

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_exact_video_cadence import (
    decoded_frame_count,
    trim_video_exact_30hz,
)
from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    select_reserved_endpoint_views_v5,
)
from bayesian_phystwin.deform360_joint_sparse_public_endpoint_v5_2 import (
    PUBLIC_ENDPOINT_BASE_CAMERA_PANEL,
    validate_public_endpoint_processing_lock_v5_2,
)
from bayesian_phystwin.deform360_joint_sparse_source_evidence_v5 import (
    validate_deform360_joint_sparse_source_prediction_batch_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_gate_v5 import (
    load_deform360_joint_sparse_source_execution_lock_v5,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5 import (
    _ordinary_root,
    _sha256_file,
)
from bayesian_phystwin.deform360_joint_sparse_source_runner_v5_2 import (
    validate_deform360_joint_sparse_source_prediction_plan_v5_2,
    validate_deform360_joint_sparse_source_prediction_receipt_v5_2,
)

RAW_FRAME_COUNT = 81
MINIMUM_SUCCESSFUL_CAMERAS = 8
MANIFEST_FILENAME = "joint_sparse_public_endpoint_processing_v5_2.json"
FAILURE_DIRNAME = "terminal-failures"
SELECTOR_BASE_SHA256 = (
    "419be2e98ab2b01627ea188c8658b43b39d8b3d4e34e8b33559f32ccdcd04184"
)
SELECTOR_OBJECT_SHA256 = (
    "79b161fa66489f75b5b078c7ae409387feed74c51a38b86e89800d0aa578b1df"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_revision(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _ffmpeg_version_first_line(ffmpeg: Path) -> str:
    output = subprocess.run(
        [str(ffmpeg), "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    _require(bool(output), "FFmpeg version probe is empty")
    return output[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--source-prediction-plan", type=Path, required=True)
    parser.add_argument("--source-prediction-root", type=Path, required=True)
    parser.add_argument("--processing-lock", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--selector-source-root", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--deform360-repository", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _sealed_context(
    args: argparse.Namespace,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    lock = load_deform360_joint_sparse_source_execution_lock_v5(args.execution_lock)
    plan = validate_deform360_joint_sparse_source_prediction_plan_v5_2(
        load_strict_json_object(
            args.source_prediction_plan, label="v5.2 source prediction plan"
        ),
        lock=lock,
    )
    prediction_root = _ordinary_root(args.source_prediction_root)
    batch_path = prediction_root / "source-prediction-batch.json"
    batch = validate_deform360_joint_sparse_source_prediction_batch_v5(
        load_strict_json_object(batch_path, label="source prediction batch"), lock
    )
    receipt = validate_deform360_joint_sparse_source_prediction_receipt_v5_2(
        load_strict_json_object(
            prediction_root / "source-prediction-receipt.json",
            label="v5.2 source prediction receipt",
        ),
        lock=lock,
        plan=plan,
        prediction_batch=batch,
        prediction_batch_file_sha256=_sha256_file(batch_path),
    )
    processing_lock = validate_public_endpoint_processing_lock_v5_2(
        load_strict_json_object(
            args.processing_lock, label="public endpoint processing lock"
        ),
        execution_lock=lock,
        source_plan=plan,
        prediction_batch=batch,
        prediction_receipt=receipt,
    )
    rows = [
        row
        for row in cast(Sequence[Mapping[str, Any]], plan["objects"])
        if row.get("object_id") == args.object_id
    ]
    _require(len(rows) == 1, "object is absent from the sealed source plan")
    return (
        lock,
        plan,
        batch,
        receipt,
        {"processing_lock": processing_lock, "row": rows[0]},
    )


def _source_paths(repository: Path) -> dict[str, Path]:
    root = repository / "deform360" / "processing"
    return {
        "reconstruct_stage": root / "reconstruct_stage.py",
        "urdf_render_stage": root / "urdf_render.py",
        "depth_stage": root / "depth_stage.py",
    }


def _validate_dependencies(
    *,
    processing_lock: Mapping[str, Any],
    selector_root: Path,
    sam2_repository: Path,
    checkpoint: Path,
    deform360_repository: Path,
    ffmpeg: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    processing = cast(Mapping[str, Any], processing_lock["processing"])
    masking = cast(Mapping[str, Any], processing["masking"])
    reconstruction = cast(Mapping[str, Any], processing["reconstruction"])
    runtime = cast(Mapping[str, Any], processing["runtime"])
    selector_sources = {
        "selector_base": selector_root / "causal4d_public" / "deform360_sam2.py",
        "selector_object": (
            selector_root / "causal4d_public" / "deform360_object_sam2.py"
        ),
    }
    _require(
        _sha256_file(selector_sources["selector_base"])
        == SELECTOR_BASE_SHA256
        == masking["selector_base_source_sha256"],
        "base automatic-mask selector changed",
    )
    _require(
        _sha256_file(selector_sources["selector_object"])
        == SELECTOR_OBJECT_SHA256
        == masking["selector_object_source_sha256"],
        "object automatic-mask selector changed",
    )
    _require(
        _git_revision(sam2_repository) == masking["sam2_revision"],
        "SAM2 revision changed",
    )
    _require(
        checkpoint.is_file()
        and _sha256_file(checkpoint) == masking["sam2_checkpoint_sha256"],
        "SAM2 checkpoint changed",
    )
    _require(
        _clean_revision(deform360_repository) == reconstruction["deform360_revision"],
        "Deform360 revision changed",
    )
    deform_sources = _source_paths(deform360_repository)
    _require(
        _sha256_file(deform_sources["reconstruct_stage"])
        == reconstruction["reconstruct_stage_sha256"]
        and _sha256_file(deform_sources["urdf_render_stage"])
        == reconstruction["urdf_render_stage_sha256"]
        and _sha256_file(deform_sources["depth_stage"])
        == reconstruction["depth_stage_sha256"],
        "Deform360 endpoint source changed",
    )
    import gsplat  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from gsplat.cuda._backend import _C  # noqa: PLC0415

    _require(
        f"{sys.version_info.major}.{sys.version_info.minor}"
        == runtime["python_major_minor"],
        "Python runtime changed",
    )
    _require(
        torch.__version__ == runtime["torch_version"]
        and torch.version.cuda == runtime["torch_cuda_version"]
        and torch.cuda.is_available(),
        "Torch/CUDA runtime changed",
    )
    _require(gsplat.__version__ == runtime["gsplat_version"], "gsplat changed")
    _require(
        os.environ.get("TORCH_CUDA_ARCH_LIST") == runtime["torch_cuda_arch_list"],
        "TORCH_CUDA_ARCH_LIST changed",
    )
    _require(
        _sha256_file(ffmpeg) == runtime["ffmpeg_sha256"]
        and _ffmpeg_version_first_line(ffmpeg)
        == runtime["ffmpeg_version_first_line"],
        "FFmpeg runtime changed",
    )
    _require(_C is not None, "gsplat CUDA backend is unavailable")
    extension = Path(_C.__file__).resolve()
    build_ninja = extension.parent / "build.ninja"
    _require(
        _sha256_file(extension) == runtime["gsplat_extension_sha256"]
        and _sha256_file(build_ninja) == runtime["gsplat_build_ninja_sha256"]
        and str(_C.CameraModelType.PINHOLE) == runtime["gsplat_backend_probe"],
        "gsplat backend changed",
    )
    hashes = {
        **{name: _sha256_file(path) for name, path in selector_sources.items()},
        **{name: _sha256_file(path) for name, path in deform_sources.items()},
        "sam2_checkpoint": _sha256_file(checkpoint),
        "ffmpeg": _sha256_file(ffmpeg),
        "gsplat_extension": _sha256_file(extension),
        "gsplat_build_ninja": _sha256_file(build_ninja),
    }
    return deform_sources, hashes


def _frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    try:
        _require(capture.isOpened(), f"cannot open public RGB video: {video}")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _trim_timestamps(
    source: Path, destination: Path, *, start: int, count: int
) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    selected = lines[start : start + count]
    _require(len(selected) == count, f"timestamp stream is too short: {source}")
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _subset_calibration(
    source: Path, destination: Path, *, cameras: Sequence[str]
) -> None:
    values = np.load(source, allow_pickle=True).item()
    _require(isinstance(values, Mapping), "camera calibration changed format")
    _require(set(cameras) <= set(values), f"camera calibration is incomplete: {source}")
    np.save(destination, {camera: values[camera] for camera in cameras})


def _write_masks(path: Path, masks: Sequence[np.ndarray]) -> None:
    values = np.asarray(masks, dtype=np.uint8)
    _require(
        values.ndim == 3
        and values.shape[0] == RAW_FRAME_COUNT
        and np.all(np.count_nonzero(values, axis=(1, 2)) > 0),
        "SAM2 returned incomplete or empty masks",
    )
    _require(not path.exists(), "mask output already exists")
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data", data=values, dtype=np.uint8, compression="gzip", compression_opts=4
        )


def _write_terminal_failure(
    output_root: Path,
    *,
    identity: Mapping[str, Any],
    error: BaseException,
    camera_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        **identity,
        "status": "technical_failure",
        "camera_records": list(camera_records),
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    failure_path = output_root / FAILURE_DIRNAME / f"{identity['object_id']}.json"
    write_atomic_json(payload, failure_path, overwrite=False)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = _ordinary_root(args.repo)
    code_revision = _clean_revision(repository)
    lock, plan, batch, receipt, context = _sealed_context(args)
    processing_lock = cast(Mapping[str, Any], context["processing_lock"])
    row = cast(Mapping[str, Any], context["row"])
    all_cameras = tuple(cast(Sequence[str], row["all_camera_ids"]))
    reserved = select_reserved_endpoint_views_v5(args.object_id, all_cameras, count=2)
    candidates = tuple(
        sorted(
            (set(PUBLIC_ENDPOINT_BASE_CAMERA_PANEL) | set(reserved)) & set(all_cameras)
        )
    )
    _require(set(reserved) <= set(candidates), "reserved cameras are unavailable")
    aligned_root = _ordinary_root(args.aligned_root)
    source_object = aligned_root / args.object_id
    source_episode = source_object / "episode_0000"
    _require(source_episode.is_dir(), "public aligned object is unavailable")
    raw_prefix = tuple(
        int(value) for value in cast(Sequence[int], row["raw_prefix_range_half_open"])
    )
    _require(
        len(raw_prefix) == 2 and raw_prefix[1] - raw_prefix[0] == 58,
        "sealed raw prefix range changed",
    )
    raw_start = raw_prefix[0]
    raw_stop = raw_start + RAW_FRAME_COUNT
    for camera in candidates:
        video = source_episode / camera / "undistorted.mp4"
        timestamps = source_episode / camera / "aligned_timestamps.txt"
        _require(video.is_file(), f"public RGB video is unavailable: {camera}")
        _require(
            _frame_count(video) >= raw_stop,
            "public video does not cover the sealed 81-frame window",
        )
        _require(
            timestamps.is_file()
            and len(timestamps.read_text(encoding="utf-8").splitlines()) >= raw_stop,
            "public timestamps do not cover the sealed 81-frame window",
        )
    selector_root = _ordinary_root(args.selector_source_root)
    sam2_repository = _ordinary_root(args.sam2_repository)
    checkpoint = args.sam2_checkpoint.resolve(strict=True)
    deform360_repository = _ordinary_root(args.deform360_repository)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    _, dependency_hashes = _validate_dependencies(
        processing_lock=processing_lock,
        selector_root=selector_root,
        sam2_repository=sam2_repository,
        checkpoint=checkpoint,
        deform360_repository=deform360_repository,
        ffmpeg=ffmpeg,
    )
    preflight = {
        "object_id": args.object_id,
        "episode_id": row["episode_id"],
        "candidate_cameras": list(candidates),
        "reserved_endpoint_cameras": list(reserved),
        "raw_window_range_half_open": [raw_start, raw_stop],
        "code_revision": code_revision,
        "execution_lock_id": lock["execution_lock_id"],
        "source_prediction_plan_id": plan["plan_id"],
        "prediction_batch_id": batch["prediction_batch_id"],
        "source_prediction_receipt_id": receipt["receipt_id"],
        "processing_lock_id": processing_lock["lock_id"],
        "dependency_hashes": dependency_hashes,
        "information_boundary": processing_lock["information_boundary"],
    }
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
        return 0

    output_root = args.output_root.absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    _require(
        output_root.is_dir()
        and not output_root.is_symlink()
        and not any(parent.is_symlink() for parent in output_root.parents),
        "output root is invalid",
    )
    destination = output_root / args.object_id
    failure_path = output_root / FAILURE_DIRNAME / f"{args.object_id}.json"
    _require(not destination.exists(), "endpoint processing output already exists")
    _require(not failure_path.exists(), "terminal failure already exists")
    scratch = output_root / f".{args.object_id}.incomplete-{os.getpid()}"
    _require(not scratch.exists(), "endpoint processing scratch already exists")
    scratch.mkdir()
    episode = scratch / "episode_0000"
    episode.mkdir()
    _subset_calibration(
        source_episode / "undistorted_intrinsics.npy",
        episode / "undistorted_intrinsics.npy",
        cameras=candidates,
    )
    _subset_calibration(
        source_episode / "extrinsics.npy",
        episode / "extrinsics.npy",
        cameras=candidates,
    )
    for camera in candidates:
        source_camera = source_episode / camera
        output_camera = episode / camera
        output_camera.mkdir()
        output_video = output_camera / "undistorted.mp4"
        trim_video_exact_30hz(
            ffmpeg,
            source_camera / "undistorted.mp4",
            output_video,
            raw_start,
            RAW_FRAME_COUNT,
            output_sync_mode="legacy-vsync",
        )
        _require(decoded_frame_count(output_video) == RAW_FRAME_COUNT, "trim changed")
        _trim_timestamps(
            source_camera / "aligned_timestamps.txt",
            output_camera / "aligned_timestamps.txt",
            start=raw_start,
            count=RAW_FRAME_COUNT,
        )
        metadata = source_camera / "metadata.json"
        if metadata.is_file():
            shutil.copy2(metadata, output_camera / "metadata.json")
    alignment = source_episode / "alignment.json"
    if alignment.is_file():
        shutil.copy2(alignment, episode / "alignment.json")
    sys.path.insert(0, str(deform360_repository))
    from deform360.robot import (  # noqa: PLC0415
        RobotState,
        load_robot_state,
        save_robot_state,
    )

    source_robot = load_robot_state(source_episode / "robot" / "robot.npz")
    _require(len(source_robot.actions) >= raw_stop, "public robot stream is too short")
    save_robot_state(
        episode / "robot" / "robot.npz",
        RobotState(
            actions=source_robot.actions[raw_start:raw_stop],
            T_worlds=source_robot.T_worlds[raw_start:raw_stop],
            openings=source_robot.openings[raw_start:raw_stop],
            bimanual=source_robot.bimanual,
        ),
    )
    camera_records: list[dict[str, Any]] = []
    identity = {
        **preflight,
        "schema": "bayesian-phystwin.deform360-joint-sparse-public-endpoint-processing",
        "schema_version": 1,
        "local_episode_id": 0,
    }
    try:
        sys.path.insert(0, str(selector_root))
        from causal4d_public.deform360_object_sam2 import (  # noqa: PLC0415
            DeformableObjectSam2VideoPredictor,
        )

        predictor = DeformableObjectSam2VideoPredictor(
            sam2_repository, checkpoint, device=args.device
        )
        try:
            for camera in candidates:
                video = episode / camera / "undistorted.mp4"
                try:
                    initial_mask, initialization = predictor.select_initial_mask(video)
                    propagated = list(
                        predictor.segment_from_initial_mask(
                            video,
                            initial_mask,
                            initialization={
                                "policy": "frozen-generic-exact-local-frame-zero",
                                "source_frame_index": 0,
                                "future_object_observations_used_for_scoring_only": True,
                                "selection": initialization,
                            },
                        )
                    )
                    _require(
                        [index for index, _ in propagated]
                        == list(range(RAW_FRAME_COUNT)),
                        "SAM2 frame roster changed",
                    )
                    mask_path = episode / camera / "mask_refined.h5"
                    _write_masks(mask_path, [mask for _, mask in propagated])
                    camera_records.append(
                        {
                            "camera_id": camera,
                            "status": "success",
                            "video_sha256": _sha256_file(video),
                            "mask_sha256": _sha256_file(mask_path),
                            "initialization": initialization,
                        }
                    )
                except BaseException as error:
                    camera_records.append(
                        {
                            "camera_id": camera,
                            "status": "technical_failure",
                            "video_sha256": _sha256_file(video),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        finally:
            predictor.close()
        successful = tuple(
            sorted(
                record["camera_id"]
                for record in camera_records
                if record["status"] == "success"
            )
        )
        _require(
            len(successful) >= MINIMUM_SUCCESSFUL_CAMERAS,
            "fewer than eight automatic camera masks succeeded",
        )
        _require(
            set(reserved) <= set(successful),
            "a reserved endpoint camera mask failed",
        )

        from deform360.processing import (  # noqa: PLC0415
            depth_stage,
            reconstruct_stage,
            urdf_render,
        )

        reconstruction = cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], processing_lock["processing"])["reconstruction"],
        )
        original_visual_hull = reconstruct_stage.visual_hull_points

        def locked_visual_hull(*call_args: object, **call_kwargs: object) -> Any:
            call_kwargs["min_points"] = reconstruction["minimum_visual_hull_points"]
            return original_visual_hull(*call_args, **call_kwargs)

        reconstruct_stage.visual_hull_points = locked_visual_hull
        try:
            splats = reconstruct_stage.process_reconstruction_episode(
                scratch,
                0,
                cameras=successful,
                first_frame_iterations=reconstruction["first_frame_iterations"],
                warm_start_iterations=reconstruction["warm_start_iterations"],
                cube_half_extent_m=reconstruction["cube_half_extent_m"],
                voxel_resolution=reconstruction["voxel_resolution"],
                overwrite=True,
                keep_scratch=False,
            )
        finally:
            reconstruct_stage.visual_hull_points = original_visual_hull
        _require(set(splats) == set(range(RAW_FRAME_COUNT)), "splats are incomplete")
        gripper_masks = urdf_render.process_gripper_masks_episode(
            scratch, 0, cameras=reserved, overwrite=True
        )
        depths = depth_stage.process_depth_episode(
            scratch, 0, cameras=reserved, overwrite=True, preview=False
        )
        _require(
            set(gripper_masks) == set(depths) == set(reserved),
            "reserved endpoint depth panel changed",
        )
        outputs = {}
        for camera in reserved:
            mask_path = episode / camera / "mask_refined.h5"
            depth_path = episode / camera / "rendered_depth.h5"
            with h5py.File(depth_path, "r") as stream:
                values = stream["data"]
                _require(
                    values.shape[0] == RAW_FRAME_COUNT and values.dtype == np.uint16,
                    "rendered endpoint depth contract changed",
                )
            outputs[camera] = {
                "mask_sha256": _sha256_file(mask_path),
                "depth_sha256": _sha256_file(depth_path),
                "urdf_sha256": _sha256_file(episode / camera / "rendered_urdf.h5"),
            }
        manifest = {
            **identity,
            "status": "success",
            "successful_support_cameras": list(successful),
            "camera_records": camera_records,
            "outputs_sha256": outputs,
            "calibration_sha256": {
                "intrinsics": _sha256_file(episode / "undistorted_intrinsics.npy"),
                "extrinsics": _sha256_file(episode / "extrinsics.npy"),
                "splatfacto_metadata": _sha256_file(
                    episode / "splatfacto" / "splatfacto.meta.json"
                ),
            },
        }
        write_atomic_json(manifest, episode / MANIFEST_FILENAME, overwrite=False)
        scratch.rename(destination)
        print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        shutil.rmtree(scratch, ignore_errors=True)
        failure = _write_terminal_failure(
            output_root,
            identity=identity,
            error=error,
            camera_records=camera_records,
        )
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
