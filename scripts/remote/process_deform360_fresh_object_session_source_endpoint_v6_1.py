#!/usr/bin/env python3
"""Materialize one authorized Deform360 v6.1 public-source endpoint carrier."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from bayesian_phystwin._portable_contracts import (
    load_strict_json_object,
    write_atomic_json,
)
from bayesian_phystwin.deform360_exact_video_cadence import decoded_frame_count
from bayesian_phystwin.deform360_fresh_object_session_source_scorer_v6_1 import (
    ENDPOINT_ARCHIVE_MEMBERS,
    SOURCE_SCORING_AMENDMENT_ID,
    load_deform360_v61_source_scoring_amendment,
    validate_deform360_v61_source_plan,
    validate_deform360_v61_source_suffix_authorization,
)

RAW_FRAME_COUNT = 81
FRAME_RATE_HZ = 30
LOCAL_ENDPOINT_RANGE = (58, 76)
MANIFEST_FILENAME = "source-endpoint-processing-v6-1.json"
OBJECT_RECORD_FILENAME = "endpoint-object.json"
FAILURE_DIRNAME = "terminal-failures"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _sha256_file(path: str | Path) -> str:
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _clean_revision(repository: Path, *, include_untracked: bool = True) -> str:
    revision = _git_revision(repository)
    untracked = "normal" if include_untracked else "no"
    status = subprocess.run(
        ["git", "status", "--porcelain", f"--untracked-files={untracked}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _ordinary_root(path: str | Path, *, name: str) -> Path:
    requested = Path(path).absolute()
    _require(
        requested.is_dir()
        and not requested.is_symlink()
        and not any(parent.is_symlink() for parent in requested.parents),
        f"{name} is invalid",
    )
    return requested.resolve(strict=True)


def _ffmpeg_version_first_line(ffmpeg: Path) -> str:
    lines = subprocess.run(
        [str(ffmpeg), "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    _require(bool(lines), "FFmpeg version probe is empty")
    return lines[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scoring-amendment", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--selector-source-root", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--deform360-repository", type=Path, required=True)
    parser.add_argument("--gsplat-wheel", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/usr/bin/ffmpeg"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def _source_paths(repository: Path) -> dict[str, Path]:
    root = repository / "deform360" / "processing"
    return {
        "reconstruct_stage": root / "reconstruct_stage.py",
        "urdf_render_stage": root / "urdf_render.py",
        "depth_stage": root / "depth_stage.py",
    }


def _validate_dependencies(
    *,
    amendment: Mapping[str, Any],
    repository: Path,
    selector_root: Path,
    sam2_repository: Path,
    checkpoint: Path,
    deform360_repository: Path,
    gsplat_wheel: Path,
    ffmpeg: Path,
) -> dict[str, str]:
    processing = cast(Mapping[str, Any], amendment["source_processing"])
    selector = cast(Mapping[str, Any], processing["selector"])
    reconstruction = cast(Mapping[str, Any], processing["reconstruction"])
    runtime = cast(Mapping[str, Any], processing["runtime"])
    endpoint = cast(Mapping[str, Any], amendment["endpoint_carrier"])
    runtime_repair = cast(
        Mapping[str, Any], runtime["source_independent_runtime_repair"]
    )
    selector_sources = {
        "selector_base": selector_root / str(selector["base_path"]),
        "selector_object": (selector_root / str(selector["object_path"])),
    }
    _require(
        selector.get("repository") == "IPS-Stuttgart/Causal4D"
        and _git_revision(selector_root) == selector["repository_revision"],
        "automatic-mask selector repository changed",
    )
    _require(
        _sha256_file(selector_sources["selector_base"])
        == selector["base_source_sha256"],
        "base automatic-mask selector changed",
    )
    _require(
        _sha256_file(selector_sources["selector_object"])
        == selector["object_source_sha256"],
        "object automatic-mask selector changed",
    )
    _require(
        _git_revision(sam2_repository) == endpoint["sam2_revision"],
        "SAM2 revision changed",
    )
    _require(
        checkpoint.is_file()
        and _sha256_file(checkpoint) == endpoint["sam2_checkpoint_sha256"],
        "SAM2 checkpoint changed",
    )
    _require(
        _clean_revision(deform360_repository) == reconstruction["deform360_revision"],
        "Deform360 revision changed",
    )
    deform_sources = _source_paths(deform360_repository)
    for key in ("reconstruct_stage", "urdf_render_stage", "depth_stage"):
        _require(
            _sha256_file(deform_sources[key]) == reconstruction[f"{key}_sha256"],
            f"Deform360 {key} source changed",
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
    _require(
        gsplat.__version__ == runtime["gsplat_base_version"]
        and importlib.metadata.version("gsplat")
        == runtime["gsplat_distribution_version"],
        "gsplat distribution changed",
    )
    _require(
        os.environ.get("TORCH_CUDA_ARCH_LIST") == runtime["torch_cuda_arch_list"],
        "TORCH_CUDA_ARCH_LIST changed",
    )
    _require(
        _sha256_file(ffmpeg) == runtime["ffmpeg_sha256"]
        and _ffmpeg_version_first_line(ffmpeg) == runtime["ffmpeg_version_first_line"],
        "FFmpeg runtime changed",
    )
    _require(_C is not None, "gsplat CUDA backend is unavailable")
    extension = Path(_C.__file__).resolve()
    installed_extension = Path(
        importlib.metadata.distribution("gsplat").locate_file(
            runtime["gsplat_extension_relative_path"]
        )
    ).resolve()
    repair_path = repository / str(runtime_repair["path"])
    _require(
        gsplat_wheel.is_file()
        and gsplat_wheel.stat().st_size == runtime["gsplat_wheel_byte_count"]
        and _sha256_file(gsplat_wheel) == runtime["gsplat_wheel_sha256"]
        and extension == installed_extension
        and _sha256_file(extension) == runtime["gsplat_extension_sha256"]
        and str(_C.CameraModelType.PINHOLE) == runtime["gsplat_backend_probe"],
        "gsplat backend changed",
    )
    _require(
        runtime["jit_compilation_used"] is False
        and runtime["nvcc_required"] is False
        and repair_path.is_file()
        and _sha256_file(repair_path) == runtime_repair["file_sha256"],
        "source-independent CUDA runtime repair changed",
    )
    return {
        **{name: _sha256_file(path) for name, path in selector_sources.items()},
        **{name: _sha256_file(path) for name, path in deform_sources.items()},
        "sam2_checkpoint": _sha256_file(checkpoint),
        "ffmpeg": _sha256_file(ffmpeg),
        "gsplat_wheel": _sha256_file(gsplat_wheel),
        "gsplat_extension": _sha256_file(extension),
        "source_independent_runtime_repair": _sha256_file(repair_path),
    }


def _frame_count(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    try:
        _require(capture.isOpened(), f"cannot open public RGB video: {video}")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def _trim_video(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    *,
    start: int,
    count: int,
) -> None:
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
            (
                f"select='between(n,{start},{start + count - 1})',"
                f"setpts=N/({FRAME_RATE_HZ}*TB)"
            ),
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
            str(FRAME_RATE_HZ),
            "-vsync",
            "cfr",
            str(destination),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    _require(
        decoded_frame_count(destination) == count,
        "exact video trim changed frame count",
    )


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
    _require(set(cameras) <= set(values), f"camera calibration incomplete: {source}")
    np.save(destination, {camera: values[camera] for camera in cameras})


def _write_partial_masks(path: Path, masks: Sequence[np.ndarray]) -> np.ndarray:
    """Write all 81 masks while retaining genuine empty-frame evidence."""

    import h5py  # noqa: PLC0415

    values = np.asarray(masks, dtype=np.uint8)
    _require(
        values.ndim == 3 and values.shape[0] == RAW_FRAME_COUNT,
        "SAM2 returned an incomplete mask tensor",
    )
    _require(not path.exists(), "mask output already exists")
    with h5py.File(path, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )
    return np.count_nonzero(values, axis=(1, 2)) > 0


def _validate_frame_support(
    masks_nonempty_by_camera: Mapping[str, np.ndarray], *, minimum: int = 2
) -> np.ndarray:
    _require(
        type(minimum) is int and minimum == 2,
        "minimum reconstruction support changed",
    )
    _require(bool(masks_nonempty_by_camera), "no automatic camera mask succeeded")
    values = list(masks_nonempty_by_camera.values())
    _require(
        all(value.shape == (RAW_FRAME_COUNT,) for value in values),
        "camera mask support shape changed",
    )
    counts = np.sum(np.stack(values, axis=0), axis=0)
    _require(
        np.all(counts >= minimum),
        "fewer than two non-empty support masks occur in at least one frame",
    )
    return np.asarray(counts, dtype=np.int64)


def _camera_dictionary(path: Path, *, name: str) -> dict[str, np.ndarray]:
    value = np.load(path, allow_pickle=True)
    _require(value.shape == () and value.dtype == object, f"{name} format changed")
    raw = value.item()
    _require(isinstance(raw, Mapping), f"{name} must contain a camera mapping")
    return {str(key): np.asarray(array) for key, array in raw.items()}


def _h5_data(path: Path, *, name: str) -> np.ndarray:
    import h5py  # noqa: PLC0415

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
    start, stop = LOCAL_ENDPOINT_RANGE
    _require(
        depth_mm.shape[0] == RAW_FRAME_COUNT
        and depth_mm.dtype == np.uint16
        and mask.shape == depth_mm.shape
        and mask.dtype in {np.dtype(np.uint8), np.dtype(np.bool_)},
        "endpoint depth or mask contract changed",
    )
    intrinsics = _camera_dictionary(intrinsics_path, name="intrinsics")
    extrinsics = _camera_dictionary(extrinsics_path, name="extrinsics")
    raw_start, raw_stop = raw_endpoint_range
    _require(
        raw_stop - raw_start == stop - start
        and camera_id in intrinsics
        and camera_id in extrinsics,
        "endpoint calibration or raw frame range changed",
    )
    arrays = {
        "frame_indices": np.arange(start, stop, dtype=np.int64),
        "raw_frame_indices": np.arange(raw_start, raw_stop, dtype=np.int64),
        "depth_m": np.asarray(depth_mm[start:stop], dtype=np.float32) / 1000.0,
        "object_mask": np.asarray(mask[start:stop] > 0, dtype=np.bool_),
        "intrinsics": np.asarray(intrinsics[camera_id], dtype=np.float64),
        "camera_to_world": np.asarray(extrinsics[camera_id], dtype=np.float64),
    }
    _require(set(arrays) == ENDPOINT_ARCHIVE_MEMBERS, "archive roster changed")
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
    return {
        "archive_sha256": _sha256_file(destination),
        "source_files_sha256": {
            "rendered_depth.h5": _sha256_file(depth_path),
            "mask_refined.h5": _sha256_file(mask_path),
            "undistorted_intrinsics.npy": _sha256_file(intrinsics_path),
            "extrinsics.npy": _sha256_file(extrinsics_path),
        },
    }


def _terminal_failure(
    output_root: Path,
    *,
    identity: Mapping[str, Any],
    error: BaseException,
    camera_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        **identity,
        "status": "terminal-source-endpoint-technical-failure",
        "camera_records": list(camera_records),
        "error": {"type": type(error).__name__, "message": str(error)},
        "source_gate_evaluated": False,
        "replacement_allowed": False,
        "confirmation_payloads_opened": False,
        "claim_authorized": False,
    }
    path = output_root / FAILURE_DIRNAME / f"{identity['object_id']}.json"
    write_atomic_json(payload, path, overwrite=False)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository = _ordinary_root(args.repo, name="BayesianPhysTwin repository")
    # Actions checks out pinned third-party repositories beneath this checkout.
    # They are untracked here, while every tracked BayesianPhysTwin file must
    # still match the authorized revision exactly.
    code_revision = _clean_revision(repository, include_untracked=False)
    amendment = load_deform360_v61_source_scoring_amendment(args.scoring_amendment)
    authorization = validate_deform360_v61_source_suffix_authorization(
        load_strict_json_object(args.authorization, label="source authorization")
    )
    _require(
        authorization["source_scoring_amendment_id"] == SOURCE_SCORING_AMENDMENT_ID
        and authorization["scorer_revision"] == code_revision,
        "authorization binds another scorer revision",
    )
    source_plan = validate_deform360_v61_source_plan(
        load_strict_json_object(args.source_plan, label="source plan")
    )
    rows = [
        row
        for row in cast(Sequence[Mapping[str, Any]], source_plan["objects"])
        if row.get("object_id") == args.object_id
    ]
    _require(len(rows) == 1, "object is absent from the sealed source plan")
    row = rows[0]
    all_cameras = tuple(cast(Sequence[str], row["all_camera_ids"]))
    reserved = tuple(cast(Sequence[str], row["reserved_endpoint_camera_ids"]))
    endpoint_contract = cast(Mapping[str, Any], amendment["endpoint_carrier"])
    fixed_panel = set(
        cast(Sequence[str], endpoint_contract["fixed_support_camera_panel"])
    )
    candidates = tuple(sorted((fixed_panel | set(reserved)) & set(all_cameras)))
    _require(set(reserved) <= set(candidates), "reserved cameras are unavailable")
    aligned_root = _ordinary_root(args.aligned_root, name="aligned public root")
    source_episode = aligned_root / args.object_id / "episode_0000"
    _require(source_episode.is_dir(), "public aligned object is unavailable")
    raw_prefix = tuple(cast(Sequence[int], row["raw_prefix_range_half_open"]))
    _require(
        len(raw_prefix) == 2 and raw_prefix[1] - raw_prefix[0] == 58,
        "sealed raw prefix range changed",
    )
    raw_start = int(raw_prefix[0])
    raw_stop = raw_start + RAW_FRAME_COUNT
    for camera in candidates:
        video = source_episode / camera / "undistorted.mp4"
        timestamps = source_episode / camera / "aligned_timestamps.txt"
        _require(
            video.is_file() and _frame_count(video) >= raw_stop,
            f"public video does not cover the sealed window: {camera}",
        )
        _require(
            timestamps.is_file()
            and len(timestamps.read_text(encoding="utf-8").splitlines()) >= raw_stop,
            f"public timestamps do not cover the sealed window: {camera}",
        )
    selector_root = _ordinary_root(args.selector_source_root, name="selector root")
    sam2_repository = _ordinary_root(args.sam2_repository, name="SAM2 repository")
    deform360_repository = _ordinary_root(
        args.deform360_repository, name="Deform360 repository"
    )
    checkpoint = args.sam2_checkpoint.resolve(strict=True)
    gsplat_wheel = args.gsplat_wheel.resolve(strict=True)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    dependency_hashes = _validate_dependencies(
        amendment=amendment,
        repository=repository,
        selector_root=selector_root,
        sam2_repository=sam2_repository,
        checkpoint=checkpoint,
        deform360_repository=deform360_repository,
        gsplat_wheel=gsplat_wheel,
        ffmpeg=ffmpeg,
    )
    identity = {
        "schema": (
            "bayesian-phystwin.deform360-fresh-object-session-v6-"
            "source-endpoint-processing"
        ),
        "schema_version": 1,
        "source_scoring_amendment_id": SOURCE_SCORING_AMENDMENT_ID,
        "authorization_id": authorization["authorization_id"],
        "processor_revision": code_revision,
        "upstream_source_plan_id": source_plan["plan_id"],
        "object_id": args.object_id,
        "episode_id": row["episode_id"],
        "stratum": row["stratum"],
        "candidate_cameras": list(candidates),
        "reserved_endpoint_cameras": list(reserved),
        "raw_window_range_half_open": [raw_start, raw_stop],
        "dependency_hashes": dependency_hashes,
        "information_boundary": {
            "candidate_predictions_sealed_before_suffix_open": True,
            "development_source_suffix_opened": True,
            "future_geometry_used_for_prediction": False,
            "confirmation_payloads_opened": False,
            "target_outcomes_opened": False,
            "held_v8_artifacts_accessed": False,
            "human_approval_required": False,
            "human_selection_used": False,
            "replacement_allowed": False,
            "released_real_world_recordings_only": True,
        },
    }
    if args.preflight_only:
        print(json.dumps(identity, indent=2, sort_keys=True, allow_nan=False))
        return 0

    output_root = args.output_root.absolute()
    output_root.mkdir(parents=True, exist_ok=True)
    _require(
        output_root.is_dir()
        and not output_root.is_symlink()
        and not any(parent.is_symlink() for parent in output_root.parents),
        "output root is invalid",
    )
    destination = output_root / "objects" / args.object_id
    failure_path = output_root / FAILURE_DIRNAME / f"{args.object_id}.json"
    _require(
        not destination.exists() and not failure_path.exists(),
        "source endpoint object already has a terminal disposition",
    )
    scratch = destination.parent / f".{args.object_id}.incomplete-{os.getpid()}"
    _require(not scratch.exists(), "endpoint scratch already exists")
    scratch.mkdir(parents=True)
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
        target_camera = episode / camera
        target_camera.mkdir()
        _trim_video(
            ffmpeg,
            source_camera / "undistorted.mp4",
            target_camera / "undistorted.mp4",
            start=raw_start,
            count=RAW_FRAME_COUNT,
        )
        _trim_timestamps(
            source_camera / "aligned_timestamps.txt",
            target_camera / "aligned_timestamps.txt",
            start=raw_start,
            count=RAW_FRAME_COUNT,
        )
        metadata = source_camera / "metadata.json"
        if metadata.is_file():
            shutil.copy2(metadata, target_camera / "metadata.json")
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
    try:
        selector_package_root = selector_root / "src"
        _require(
            selector_package_root.is_dir(),
            "Causal4D source package root is unavailable",
        )
        sys.path.insert(0, str(selector_package_root))
        from causal4d_public.deform360_object_sam2 import (  # noqa: PLC0415
            DeformableObjectSam2VideoPredictor,
        )

        predictor = DeformableObjectSam2VideoPredictor(
            sam2_repository, checkpoint, device=args.device
        )
        support_by_camera: dict[str, np.ndarray] = {}
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
                            "video_sha256": _sha256_file(video),
                            "mask_sha256": _sha256_file(mask_path),
                            "nonempty_mask_frame_count": int(np.sum(support)),
                            "empty_mask_frame_indices": np.nonzero(~support)[
                                0
                            ].tolist(),
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
        _require(
            set(reserved) <= set(support_by_camera),
            "a reserved camera failed automatic mask materialization",
        )
        frame_support = _validate_frame_support(support_by_camera)
        successful = tuple(sorted(support_by_camera))

        from deform360.processing import (  # noqa: PLC0415
            depth_stage,
            reconstruct_stage,
            urdf_render,
        )

        reconstruction = cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], amendment["source_processing"])["reconstruction"],
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
        raw_endpoint = (int(raw_prefix[1]), int(raw_prefix[1]) + 18)
        archives: list[dict[str, Any]] = []
        output_hashes: dict[str, Any] = {}
        for camera in reserved:
            archive_relative = Path("endpoint-archives") / f"{camera}.npz"
            archive_path = scratch / archive_relative
            record = _write_endpoint_archive(
                episode=episode,
                camera_id=camera,
                raw_endpoint_range=raw_endpoint,
                destination=archive_path,
            )
            archives.append(
                {
                    "camera_id": camera,
                    "endpoint_archive": {
                        "path": (
                            Path("objects") / args.object_id / archive_relative
                        ).as_posix(),
                        "sha256": record["archive_sha256"],
                    },
                }
            )
            output_hashes[camera] = record
        object_record = {
            "object_id": args.object_id,
            "episode_id": row["episode_id"],
            "stratum": row["stratum"],
            "all_camera_ids": list(all_cameras),
            "raw_endpoint_range_half_open": list(raw_endpoint),
            "reserved_views": sorted(archives, key=lambda item: item["camera_id"]),
        }
        write_atomic_json(
            object_record,
            scratch / OBJECT_RECORD_FILENAME,
            overwrite=False,
        )
        manifest = {
            **identity,
            "status": "success",
            "successful_support_cameras": list(successful),
            "frame_support_camera_count": frame_support.tolist(),
            "camera_records": camera_records,
            "endpoint_outputs": output_hashes,
            "object_record_sha256": _sha256_file(scratch / OBJECT_RECORD_FILENAME),
            "calibration_sha256": {
                "intrinsics": _sha256_file(episode / "undistorted_intrinsics.npy"),
                "extrinsics": _sha256_file(episode / "extrinsics.npy"),
                "splatfacto_metadata": _sha256_file(
                    episode / "splatfacto" / "splatfacto.meta.json"
                ),
            },
        }
        write_atomic_json(
            manifest,
            episode / MANIFEST_FILENAME,
            overwrite=False,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        scratch.rename(destination)
        print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        shutil.rmtree(scratch, ignore_errors=True)
        failure = _terminal_failure(
            output_root,
            identity=identity,
            error=error,
            camera_records=camera_records,
        )
        print(json.dumps(failure, indent=2, sort_keys=True, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
