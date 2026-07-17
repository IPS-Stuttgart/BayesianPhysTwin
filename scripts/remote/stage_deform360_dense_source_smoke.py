#!/usr/bin/env python3
"""Stage a short, source-only Deform360 episode slice for the dense pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np

from causal4d_public.deform360_dense_source import (
    require_source_episode,
    sha256_file,
    unpack_sampled_mask,
    write_dense_source_manifest,
)
from causal4d_public.deform360_action_audit import summarize_robot_action
from causal4d_public.deform360_dense_reusable_panel import (
    authorize_dense_panel_episode,
    load_dense_reusable_panel_config,
)
from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_reusable_trust_protocol import (
    authorize_reusable_trust_episode,
    load_reusable_trust_protocol,
)
from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_reusable_sota_window import (
    authorize_development_fit_window,
    load_reusable_sota_window,
    select_reusable_sota_action_window,
)
from causal4d_public.deform360_sota_processing import (
    authorize_development_processing,
    write_development_action_window_stage,
)
from deform360.robot import RobotState, load_robot_state, save_robot_state


def _trim_video(source: Path, destination: Path, start: int, count: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"select='between(n,{start},{start + count - 1})',setpts=N/FRAME_RATE/TB",
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
            str(destination),
        ],
        check=True,
    )


def _trim_timestamps(source: Path, destination: Path, start: int, count: int) -> None:
    lines = source.read_text(encoding="utf-8").splitlines()
    selected = lines[start : start + count]
    if len(selected) != count:
        raise ValueError(f"requested {count} timestamps but found {len(selected)}")
    destination.write_text("\n".join(selected) + "\n", encoding="utf-8")


def _subset_calibration(source: Path, destination: Path, cameras: list[str]) -> None:
    payload = np.load(source, allow_pickle=True).item()
    missing = sorted(set(cameras) - set(payload))
    if missing:
        raise ValueError(f"calibration {source.name} lacks cameras {missing}")
    np.save(destination, {camera: payload[camera] for camera in cameras})


def _write_masks(
    destination: Path,
    masks: list[np.ndarray],
) -> None:
    values = np.asarray(masks, dtype=np.uint8)
    with h5py.File(destination, "w") as stream:
        stream.create_dataset(
            "data",
            data=values,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=4,
        )


def _write_sampled_mask_archive(
    destination: Path,
    *,
    cameras: list[str],
    frame_index: int,
    masks: dict[str, np.ndarray],
) -> None:
    values = np.stack([np.asarray(masks[camera], dtype=bool) for camera in cameras])
    if values.ndim != 3 or len({tuple(mask.shape) for mask in values}) != 1:
        raise ValueError("automatic initial masks must share one image shape")
    packed = np.packbits(values[:, None], axis=-1)
    np.savez_compressed(
        destination,
        frame_indices=np.asarray([frame_index], dtype=np.int64),
        cameras=np.asarray(cameras),
        packed_masks=packed,
        image_shape=np.asarray(values.shape[1:], dtype=np.int64),
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _trim_robot(
    source_episode: Path, output_episode: Path, start: int, count: int
) -> Path:
    source = load_robot_state(source_episode / "robot" / "robot.npz")
    stop = start + count
    if stop > source.num_frames:
        raise ValueError("robot state is shorter than the requested frame slice")
    trimmed = RobotState(
        actions=source.actions[start:stop],
        T_worlds=source.T_worlds[start:stop],
        openings=source.openings[start:stop],
        bimanual=source.bimanual,
    )
    return save_robot_state(output_episode / "robot" / "robot.npz", trimmed)


def _trim_tactile_streams(
    source_episode: Path,
    output_episode: Path,
    start: int,
    count: int,
) -> dict[str, str]:
    stop = start + count
    outputs: dict[str, str] = {}
    for source_dir in sorted(source_episode.glob("*tactile*")):
        source = source_dir / "synced_tactile.npy"
        if not source.exists():
            continue
        values = np.load(source, allow_pickle=False)
        trimmed = values[start:stop]
        if len(trimmed) != count:
            raise ValueError(f"tactile stream {source_dir.name} is too short")
        output_dir = output_episode / source_dir.name
        output_dir.mkdir()
        destination = output_dir / "synced_tactile.npy"
        np.save(destination, trimmed)
        for filename in ("metadata.json", "alignment.json"):
            if (source_dir / filename).exists():
                shutil.copy2(source_dir / filename, output_dir / filename)
        outputs[source_dir.name] = sha256_file(destination)
    if not outputs:
        raise FileNotFoundError(f"no tactile streams found in {source_episode}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_aligned_root")
    parser.add_argument("sampled_masks_npz")
    parser.add_argument("output_aligned_root")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--dense-panel-config")
    parser.add_argument(
        "--sota-window-addendum",
        help=(
            "Opt into the locked reusable-SOTA action window. The --protocol "
            "argument must name its parent protocol. Development fit episodes only."
        ),
    )
    parser.add_argument("--fresh-parent-lock")
    parser.add_argument("--physics-addendum")
    parser.add_argument("--execution-lock")
    parser.add_argument("--fresh-operation", choices=("fit", "held-prediction"))
    parser.add_argument(
        "--action-aligned",
        action="store_true",
        help="Select the source window using the locked known-action-only rule.",
    )
    parser.add_argument(
        "--prediction-only-staging",
        action="store_true",
        help=(
            "Read and segment only the window-start object frame while retaining "
            "the complete known robot-action window. Required by the fresh "
            "outcome-blind prediction path."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sam2-repository", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-automatic-initial-mask-fallback",
        action="store_true",
        help=(
            "Use the frozen generic SAM2 selector on the exact window-start frame "
            "when its source-QA mask is absent; never reuse a mask from another frame."
        ),
    )
    args = parser.parse_args()

    fresh_values = (
        args.fresh_parent_lock,
        args.physics_addendum,
        args.execution_lock,
        args.fresh_operation,
    )
    if any(value is not None for value in fresh_values) and not all(
        value is not None for value in fresh_values
    ):
        raise ValueError(
            "fresh parent, physics, execution, and operation locks are required together"
        )
    fresh_authorization = None
    sota_authorization = None
    sota_processing_authorization = None
    if args.sota_window_addendum is not None:
        if args.fresh_parent_lock is not None or args.dense_panel_config is not None:
            raise ValueError(
                "reusable-SOTA staging cannot combine fresh or dense-panel locks"
            )
        if not args.action_aligned or args.start_frame is not None:
            raise ValueError(
                "reusable-SOTA staging requires action alignment without a fixed start"
            )
        if args.prediction_only_staging:
            raise ValueError(
                "this runner authorizes reusable-SOTA development fit staging only"
            )
        parent = load_reusable_sota_config(args.protocol)
        sota_window = load_reusable_sota_window(args.sota_window_addendum)
        sota_processing_authorization = authorize_development_processing(
            parent,
            object_id=args.object_id,
            episode_id=args.episode,
            role="fit",
        )
        sota_authorization = authorize_development_fit_window(
            parent,
            sota_window,
            object_id=args.object_id,
            episode_id=args.episode,
        )
    elif args.fresh_parent_lock is None:
        require_source_episode(args.protocol, args.object_id, args.episode)
    else:
        if not args.prediction_only_staging:
            raise ValueError(
                "fresh prediction staging must not read post-initial object frames"
            )
        fresh_protocol = load_reusable_trust_protocol(
            args.fresh_parent_lock, args.physics_addendum, args.execution_lock
        )
        fresh_authorization = authorize_reusable_trust_episode(
            fresh_protocol,
            object_id=args.object_id,
            episode_id=args.episode,
            operation=args.fresh_operation,
        )
    source_episode = (
        Path(args.source_aligned_root) / args.object_id / f"episode_{args.episode:04d}"
    )
    action_alignment: dict[str, object] | None = None
    if args.action_aligned:
        if args.dense_panel_config is None and sota_authorization is None:
            raise ValueError("action alignment requires --dense-panel-config")
        if args.start_frame is not None:
            raise ValueError("action alignment cannot also set --start-frame")
        if sota_authorization is not None:
            with np.load(
                source_episode / "robot" / "robot.npz", allow_pickle=False
            ) as robot:
                selected_action = select_reusable_sota_action_window(
                    robot["actions"], robot["openings"], sota_window
                )
            action_summary = selected_action["action_summary"]
            selected_range = selected_action["selected_raw_frame_range_half_open"]
            selection = selected_action["selection_rule"]
            authorization = sota_authorization
            authorization_config_sha256 = sota_authorization["window_config_sha256"]
        else:
            panel = load_dense_reusable_panel_config(args.dense_panel_config)
        if sota_authorization is None and fresh_authorization is None:
            authorization = authorize_dense_panel_episode(
                panel,
                object_id=args.object_id,
                episode_id=args.episode,
                phase="source",
                source_admission_passed=False,
            )
            authorization_config_sha256 = authorization["config_sha256"]
        elif sota_authorization is None:
            authorization = fresh_authorization
            authorization_config_sha256 = authorization["addendum_file_sha256"]
        if sota_authorization is None:
            selection = panel["config"]["frame_protocol"]["window_selection"]
            old_start, old_stop = panel["config"]["frame_protocol"][
                "superseded_fixed_raw_aligned_range_half_open"
            ]
            with np.load(
                source_episode / "robot" / "robot.npz", allow_pickle=False
            ) as robot:
                action_summary = summarize_robot_action(
                    robot["actions"],
                    robot["openings"],
                    locked_start=int(old_start),
                    locked_stop=int(old_stop),
                    candidate_start_frame=int(selection["candidate_starts"]["first"]),
                    candidate_stride_frames=int(
                        selection["candidate_starts"]["stride"]
                    ),
                )
            selected_range = action_summary["best_contact_conditioned_path_window"][
                "frame_range_half_open"
            ]
        args.start_frame = int(selected_range[0])
        args.frame_count = int(selection["window_length_frames"])
        action_alignment = {
            "schema_version": 1,
            "artifact_kind": "Deform360ActionAlignedSourceStaging",
            "protocol_id": authorization["protocol_id"],
            "config_sha256": authorization_config_sha256,
            "object_id": args.object_id,
            "episode_id": int(args.episode),
            "selected_raw_frame_range_half_open": selected_range,
            "selection_rule": selection,
            "action_summary": action_summary,
            "robot_sha256": sha256_file(source_episode / "robot" / "robot.npz"),
            "source_only": True,
            "target_action_read": False,
            "target_observation_read": False,
            "target_future_read": False,
        }
        if fresh_authorization is not None:
            action_alignment["fresh_authorization"] = fresh_authorization
        if sota_authorization is not None:
            action_alignment["reusable_sota_authorization"] = sota_authorization
        action_alignment["result_sha256"] = _canonical_sha256(action_alignment)
    elif args.start_frame is None:
        raise ValueError("fixed staging requires --start-frame")
    if args.dense_panel_config is not None and not args.action_aligned:
        raise ValueError("--dense-panel-config requires --action-aligned")

    output_episode_name = (
        f"episode_{args.episode:04d}"
        if sota_authorization is not None
        else "episode_0000"
    )
    output_episode = Path(args.output_aligned_root) / output_episode_name
    if output_episode.exists():
        if not args.overwrite:
            raise FileExistsError(f"source staging already exists: {output_episode}")
        shutil.rmtree(output_episode)
    output_episode.mkdir(parents=True)
    object_frame_count = 1 if args.prediction_only_staging else args.frame_count

    sampled_masks_path = Path(args.sampled_masks_npz)
    initial_masks: dict[str, np.ndarray] = {}
    sampled_masks_exact = False
    if sampled_masks_path.is_file():
        with np.load(sampled_masks_path, allow_pickle=False) as archive:
            cameras = [str(value) for value in archive["cameras"]]
            frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            sampled_masks_exact = (
                int(np.count_nonzero(frame_indices == args.start_frame)) == 1
            )
            if sampled_masks_exact:
                initial_masks = {
                    camera: unpack_sampled_mask(archive, camera, args.start_frame)
                    for camera in cameras
                }
    else:
        calibration = np.load(
            source_episode / "undistorted_intrinsics.npy", allow_pickle=True
        ).item()
        cameras = sorted(str(camera) for camera in calibration)
    if not sampled_masks_exact and not args.allow_automatic_initial_mask_fallback:
        reason = (
            "source-QA mask archive is absent"
            if not sampled_masks_path.is_file()
            else f"source-QA archive has no unique mask at frame {args.start_frame}"
        )
        raise ValueError(
            f"{reason}; pass --allow-automatic-initial-mask-fallback for the "
            "prospectively declared exact-frame generic-SAM2 fallback"
        )
    initial_mask_policy = (
        "exact_source_qa_mask"
        if sampled_masks_exact
        else "exact_frame_generic_sam2_fallback"
    )

    _subset_calibration(
        source_episode / "undistorted_intrinsics.npy",
        output_episode / "undistorted_intrinsics.npy",
        cameras,
    )
    _subset_calibration(
        source_episode / "extrinsics.npy",
        output_episode / "extrinsics.npy",
        cameras,
    )
    robot_path = _trim_robot(
        source_episode,
        output_episode,
        args.start_frame,
        args.frame_count,
    )
    tactile_hashes = (
        _trim_tactile_streams(
            source_episode,
            output_episode,
            args.start_frame,
            args.frame_count,
        )
        if fresh_authorization is None
        else {}
    )

    predictor = DeformableObjectSam2VideoPredictor(
        args.sam2_repository,
        args.checkpoint,
        device=args.device,
    )
    diagnostics: dict[str, object] = {}
    initialization_diagnostics: dict[str, object] = {}
    try:
        for camera in cameras:
            source_camera = source_episode / camera
            output_camera = output_episode / camera
            output_camera.mkdir()
            _trim_video(
                source_camera / "undistorted.mp4",
                output_camera / "undistorted.mp4",
                args.start_frame,
                object_frame_count,
            )
            _trim_timestamps(
                source_camera / "aligned_timestamps.txt",
                output_camera / "aligned_timestamps.txt",
                args.start_frame,
                object_frame_count,
            )
            metadata_path = source_camera / "metadata.json"
            if metadata_path.exists():
                shutil.copy2(metadata_path, output_camera / "metadata.json")
            if sampled_masks_exact:
                initial_mask = initial_masks[camera]
                initialization = {
                    "policy": initial_mask_policy,
                    "source_archive_sha256": sha256_file(sampled_masks_path),
                    "source_frame_index": args.start_frame,
                    "object_observation_frames_used": [0],
                }
            else:
                initial_mask, automatic_diagnostics = predictor.select_initial_mask(
                    output_camera / "undistorted.mp4"
                )
                initial_masks[camera] = initial_mask
                initialization = {
                    "policy": initial_mask_policy,
                    "source_frame_index": args.start_frame,
                    "staged_frame_index": 0,
                    "object_observation_frames_used": [0],
                    "future_object_observations_used": False,
                    "automatic_selection": automatic_diagnostics,
                }
            initialization_diagnostics[camera] = initialization
            masks = list(
                predictor.segment_from_initial_mask(
                    output_camera / "undistorted.mp4",
                    initial_mask,
                    initialization=initialization,
                )
            )
            if [index for index, _ in masks] != list(range(object_frame_count)):
                raise ValueError(f"SAM2 returned incomplete frames for {camera}")
            _write_masks(
                output_camera / "mask_refined.h5",
                [mask for _, mask in masks],
            )
            diagnostics[camera] = predictor.diagnostics[-1]
    finally:
        predictor.close()

    diagnostics_path = output_episode / "sam2_source_masks.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    initialization_path = output_episode / "sam2_initial_masks.json"
    initialization_path.write_text(
        json.dumps(
            {
                "policy": initial_mask_policy,
                "locked_window_start_frame": args.start_frame,
                "future_object_observations_used": False,
                "cameras": initialization_diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_masks_path = sampled_masks_path
    if not sampled_masks_exact:
        manifest_masks_path = output_episode / "automatic_initial_masks.npz"
        _write_sampled_mask_archive(
            manifest_masks_path,
            cameras=cameras,
            frame_index=args.start_frame,
            masks=initial_masks,
        )
    write_dense_source_manifest(
        output_episode / "dense_source_smoke.manifest.json",
        protocol_path=(
            args.sota_window_addendum
            if args.sota_window_addendum is not None
            else (
                args.protocol
                if args.physics_addendum is None
                else args.physics_addendum
            )
        ),
        object_id=args.object_id,
        episode_index=args.episode,
        source_episode_dir=source_episode,
        sampled_masks_path=manifest_masks_path,
        start_frame=args.start_frame,
        frame_count=object_frame_count,
        cameras=cameras,
        outputs={
            "episode_dir": str(output_episode.resolve()),
            "sam2_diagnostics_sha256": sha256_file(diagnostics_path),
            "initial_mask_policy": initial_mask_policy,
            "initial_mask_diagnostics_sha256": sha256_file(initialization_path),
            "automatic_initial_mask_fallback_used": not sampled_masks_exact,
            "robot_sha256": sha256_file(robot_path),
            "known_robot_action_frame_count": args.frame_count,
            "object_observation_frame_count": object_frame_count,
            "tactile_sha256": tactile_hashes,
            "tactile_copied": fresh_authorization is None,
        },
    )
    if action_alignment is not None:
        alignment_path = output_episode / "action_aligned_source_staging.json"
        alignment_path.write_text(
            json.dumps(action_alignment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if sota_authorization is not None:
        assert sota_processing_authorization is not None
        write_development_action_window_stage(
            output_episode / "development_staging.json",
            authorization=sota_processing_authorization,
            window_authorization=sota_authorization,
            selected_raw_frame_range_half_open=selected_range,
            camera_count=len(cameras),
            frame_count=object_frame_count,
            window_config_sha256=sota_authorization["window_config_sha256"],
            mask_diagnostics_sha256=sha256_file(diagnostics_path),
            initialization_diagnostics_sha256=sha256_file(initialization_path),
        )
    print(
        json.dumps(
            {
                "passed": True,
                "source_only": True,
                "episode_dir": str(output_episode),
                "camera_count": len(cameras),
                "frame_count": object_frame_count,
                "known_robot_action_frame_count": args.frame_count,
                "start_frame": args.start_frame,
                "action_aligned": args.action_aligned,
                "initial_mask_policy": initial_mask_policy,
                "action_alignment_result_sha256": (
                    action_alignment["result_sha256"]
                    if action_alignment is not None
                    else None
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
