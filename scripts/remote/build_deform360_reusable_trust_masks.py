#!/usr/bin/env python3
"""Build one exact-frame, source-referenced multiview mask artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from causal4d_public.deform360_action_audit import summarize_robot_action
from causal4d_public.deform360_contact_conditioned_action import (
    geometry_latched_contact_schedule,
)
from causal4d_public.deform360_dense_reusable_panel import (
    load_dense_reusable_panel_config,
)
from causal4d_public.deform360_grounded_sam2 import (
    GroundedSam2ImagePredictor,
    GroundedSam2MaskConfig,
)
from causal4d_public.deform360_object_sam2 import (
    DeformableObjectSam2MaskConfig,
    DeformableObjectSam2VideoPredictor,
)
from causal4d_public.deform360_reusable_trust_masks import (
    GEOMETRY_CONTACT_MASK_ADDENDUM_ID,
    GROUNDED_MASK_ADDENDUM_ID,
    SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID,
    authorize_reusable_trust_mask_episode,
    load_reusable_trust_mask_addendum,
    sha256_file,
    write_sampled_mask_archive,
)
from causal4d_public.deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    JointMultiviewMaskSelectionConfig,
    select_joint_multiview_masks,
)
from deform360.processing.control_points_stage import _frame_controller_points
from deform360.processing.episode import load_episode_calibration
from deform360.processing.reconstruct_stage import visual_hull_points
from deform360.robot import load_robot_state


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _read_rgb_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"cannot read frame {frame_index} from {video_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _action_window_start(
    episode_dir: Path, panel: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    selection = panel["config"]["frame_protocol"]["window_selection"]
    old_start, old_stop = panel["config"]["frame_protocol"][
        "superseded_fixed_raw_aligned_range_half_open"
    ]
    state = load_robot_state(episode_dir / "robot" / "robot.npz")
    summary = summarize_robot_action(
        state.actions,
        state.openings,
        locked_start=int(old_start),
        locked_stop=int(old_stop),
        candidate_start_frame=int(selection["candidate_starts"]["first"]),
        candidate_stride_frames=int(selection["candidate_starts"]["stride"]),
    )
    start, stop = summary["best_contact_conditioned_path_window"][
        "frame_range_half_open"
    ]
    if int(stop) - int(start) != int(selection["window_length_frames"]):
        raise ValueError("selected action window has the wrong length")
    return int(start), summary


def _load_reference(
    path: Path,
    *,
    object_id: str,
    protocol: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    summary_path = path.with_suffix(".json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("artifact_kind") != "Deform360ReusableTrustMaskReference"
        or summary.get("object_id") != object_id
        or int(summary.get("episode_id", -1)) != 1
        or summary.get("mask_addendum_file_sha256")
        != protocol["mask_addendum_file_sha256"]
        or summary.get("archive_sha256") != sha256_file(path)
    ):
        raise ValueError("source-reference mask artifact is incompatible")
    with np.load(path, allow_pickle=False) as archive:
        rgb = np.asarray(archive["rgb"], dtype=np.uint8)
        mask = np.asarray(archive["mask"], dtype=bool)
    if mask.shape != rgb.shape[:2]:
        raise ValueError("source-reference RGB and mask shapes differ")
    return rgb, mask, summary


def _write_reference(
    output: Path,
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    object_id: str,
    frame_index: int,
    camera: str,
    diagnostics: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    np.savez_compressed(output, rgb=rgb, mask=np.asarray(mask, dtype=np.uint8))
    summary = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTrustMaskReference",
        "protocol_id": protocol["mask_addendum"]["protocol_id"],
        "mask_addendum_file_sha256": protocol["mask_addendum_file_sha256"],
        "object_id": object_id,
        "episode_id": 1,
        "camera": camera,
        "raw_frame_index": frame_index,
        "bootstrap_diagnostics": diagnostics,
        "rgb_sha256": _array_sha256(rgb),
        "mask_sha256": _array_sha256(np.asarray(mask, dtype=bool)),
        "archive_sha256": sha256_file(output),
        "information_boundary": {
            "object_observation_frames_used": [frame_index],
            "future_object_observation_used": False,
            "manual_mask_or_box_used": False,
            "simulator_residual_used": False,
        },
    }
    summary["result_sha256"] = _canonical_sha256(summary)
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument(
        "--operation", choices=("fit", "held-prediction"), required=True
    )
    parser.add_argument("--fresh-parent-lock", type=Path, required=True)
    parser.add_argument("--physics-addendum", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--mask-addendum", type=Path, required=True)
    parser.add_argument("--dense-panel-config", type=Path, required=True)
    parser.add_argument("--sam2-repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path)
    parser.add_argument("--grounding-dino-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_trust_mask_addendum(
        args.fresh_parent_lock,
        args.physics_addendum,
        args.execution_lock,
        args.mask_addendum,
    )
    authorization = authorize_reusable_trust_mask_episode(
        protocol,
        object_id=args.object_id,
        episode_id=args.episode_id,
        operation=args.operation,
    )
    if (
        sha256_file(args.checkpoint)
        != protocol["mask_addendum"]["sam2"]["checkpoint_sha256"]
    ):
        raise ValueError("SAM2 checkpoint hash changed")
    panel = load_dense_reusable_panel_config(args.dense_panel_config)
    episode_dir = args.aligned_root / args.object_id / f"episode_{args.episode_id:04d}"
    frame_index, action_summary = _action_window_start(episode_dir, panel)
    object_policy = protocol["mask_addendum"]["objects"][args.object_id]
    cameras = [str(value) for value in object_policy["cameras"]]
    intrinsics_all, extrinsics_all = load_episode_calibration(episode_dir)
    if not set(cameras).issubset(intrinsics_all) or not set(cameras).issubset(
        extrinsics_all
    ):
        raise ValueError("frozen camera set is unavailable")
    intrinsics = {camera: intrinsics_all[camera] for camera in cameras}
    extrinsics = {camera: extrinsics_all[camera] for camera in cameras}
    rgb = {
        camera: _read_rgb_frame(episode_dir / camera / "undistorted.mp4", frame_index)
        for camera in cameras
    }

    sam2_policy = protocol["mask_addendum"]["sam2"]
    object_config = DeformableObjectSam2MaskConfig(**sam2_policy["candidate_config"])
    mask_protocol_id = protocol["mask_addendum"]["protocol_id"]
    grounded = mask_protocol_id in {
        GROUNDED_MASK_ADDENDUM_ID,
        GEOMETRY_CONTACT_MASK_ADDENDUM_ID,
        SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID,
    }
    if grounded:
        if args.reference_artifact is not None:
            raise ValueError("grounded masks do not accept a source-reference artifact")
        grounded_config = GroundedSam2MaskConfig(
            **protocol["mask_addendum"]["observation_initializer"]["grounding_dino"]
        )
        predictor = GroundedSam2ImagePredictor(
            args.sam2_repository,
            args.checkpoint,
            device=args.device,
            object_config=object_config,
            grounded_config=grounded_config,
            cache_dir=args.grounding_dino_cache,
        )
        reference_policy = None
        reference_camera = None
    else:
        predictor = DeformableObjectSam2VideoPredictor(
            args.sam2_repository,
            args.checkpoint,
            device=args.device,
            config=object_config,
        )
        reference_policy = protocol["mask_addendum"]["source_reference"]
        reference_camera = str(reference_policy["camera"])
    bootstrap_diagnostics = None
    reference_summary = None
    reference_artifact = None
    try:
        if grounded:
            reference_rgb = None
            reference_mask = None
        elif args.reference_artifact is None:
            assert reference_policy is not None and reference_camera is not None
            if args.episode_id != int(reference_policy["episode_id"]):
                raise ValueError("non-reference episodes require a source reference")
            if args.operation != "fit":
                raise ValueError("a held episode cannot bootstrap the reference")
            reference_rgb = rgb[reference_camera]
            reference_mask, bootstrap_diagnostics = (
                predictor.select_initial_mask_from_rgb(
                    reference_rgb,
                    camera=reference_camera,
                    video_name=f"raw-frame-{frame_index:06d}",
                )
            )
        else:
            reference_rgb, reference_mask, reference_summary = _load_reference(
                args.reference_artifact,
                object_id=args.object_id,
                protocol=protocol,
            )
        candidates: dict[str, list[dict[str, Any]]] = {}
        candidate_diagnostics = []
        maximum = int(
            protocol["mask_addendum"]["joint_multiview_selection"][
                "maximum_candidates_per_camera"
            ]
        )
        include_basic = bool(
            protocol["mask_addendum"]["joint_multiview_selection"].get(
                "include_basic_candidates_below_appearance_threshold", False
            )
        )
        for camera in cameras:
            if grounded:
                try:
                    records, summary = predictor.candidates_from_rgb(
                        rgb[camera],
                        prompt=str(object_policy["text_prompt"]),
                        camera=camera,
                        video_name=f"raw-frame-{frame_index:06d}",
                    )
                except ValueError as error:
                    if "text grounding found no box" not in str(error):
                        raise
                    candidate_diagnostics.append(
                        {
                            "camera": camera,
                            "candidate_count": 0,
                            "status": "rejected-no-grounding-box",
                            "error": str(error),
                            "candidates": [],
                        }
                    )
                    continue
            else:
                records, summary = (
                    predictor.initial_mask_candidates_from_rgb_with_reference(
                        rgb[camera],
                        camera=camera,
                        video_name=f"raw-frame-{frame_index:06d}",
                        reference_rgb=reference_rgb,
                        reference_mask=reference_mask,
                        reference_camera=reference_camera,
                        maximum_candidates=maximum,
                        include_below_appearance_threshold=include_basic,
                    )
                )
            candidates[camera] = records
            candidate_diagnostics.append(
                {
                    **summary,
                    "candidates": [record["diagnostic"] for record in records],
                }
            )
    finally:
        predictor.close()

    if len(candidates) < 3:
        raise ValueError("fewer than three cameras produced mask candidates")

    selection = protocol["mask_addendum"]["joint_multiview_selection"]
    candidate_cameras = sorted(candidates)
    candidate_intrinsics = {camera: intrinsics[camera] for camera in candidate_cameras}
    candidate_extrinsics = {camera: extrinsics[camera] for camera in candidate_cameras}
    candidate_rgb = {camera: rgb[camera] for camera in candidate_cameras}
    selected_masks, joint = select_joint_multiview_masks(
        candidates,
        candidate_intrinsics,
        candidate_extrinsics,
        CrossViewMaskReliabilityConfig(**selection["cross_view_config"]),
        JointMultiviewMaskSelectionConfig(
            maximum_candidates_per_camera=int(
                selection["maximum_candidates_per_camera"]
            ),
            voxel_resolution=int(selection["voxel_resolution"]),
            coordinate_descent_passes=int(selection["coordinate_descent_passes"]),
            appearance_weight=float(selection["appearance_weight"]),
            projected_volume_penalty=float(selection["projected_volume_penalty"]),
        ),
    )
    hull, _ = visual_hull_points(
        selected_masks,
        candidate_rgb,
        candidate_intrinsics,
        candidate_extrinsics,
        voxel_resolution=int(selection["cross_view_config"]["voxel_resolution"]),
        min_points=int(
            protocol["mask_addendum"]["source_qa_gates"][
                "minimum_visual_hull_point_count"
            ]
        ),
    )
    state = load_robot_state(episode_dir / "robot" / "robot.npz")
    controller = _frame_controller_points(state, frame_index)
    frame_zero_hull_to_controller_m = float(
        cKDTree(controller).query(hull, k=1)[0].min()
    )
    gates_policy = protocol["mask_addendum"]["source_qa_gates"]
    geometry_contact = mask_protocol_id in {
        GEOMETRY_CONTACT_MASK_ADDENDUM_ID,
        SOURCE_TRAINED_CAMERA_MASK_ADDENDUM_ID,
    }
    contact_onset_frames: list[int | None] = []
    minimum_group_distance_m: list[float] = []
    retained_contact_group_count = 0
    if geometry_contact:
        contact_policy = protocol["mask_addendum"]["geometry_contact_policy"]
        start, stop = action_summary["best_contact_conditioned_path_window"][
            "frame_range_half_open"
        ]
        controller_trajectory = np.stack(
            [_frame_controller_points(state, raw) for raw in range(start, stop)]
        )
        contact_schedule, contact_distances = geometry_latched_contact_schedule(
            controller_trajectory,
            hull,
            controller_group_size=int(contact_policy["controller_group_size"]),
            maximum_contact_distance_m=float(
                contact_policy["maximum_contact_distance_m"]
            ),
            confirmation_frames=int(contact_policy["confirmation_frames"]),
        )
        for group in range(contact_schedule.shape[1]):
            active = np.flatnonzero(contact_schedule[:, group])
            contact_onset_frames.append(int(active[0]) if len(active) else None)
            minimum_group_distance_m.append(float(np.min(contact_distances[:, group])))
        retained_contact_group_count = sum(
            onset is not None for onset in contact_onset_frames
        )
        hull_to_controller_m = float(np.min(contact_distances))
    else:
        hull_to_controller_m = frame_zero_hull_to_controller_m
    gates = {
        "accepted_camera_count": int(
            joint["cross_view_consistency"]["accepted_camera_count"]
        )
        >= int(gates_policy["minimum_accepted_camera_count"]),
        "visual_hull_point_count": len(hull)
        >= int(gates_policy["minimum_visual_hull_point_count"]),
        "hull_to_controller_distance": hull_to_controller_m
        <= float(gates_policy["maximum_hull_to_controller_distance_m"]),
    }
    if geometry_contact:
        gates["geometry_contact_group_count"] = retained_contact_group_count >= int(
            gates_policy["minimum_geometry_contact_group_count"]
        )
    passed = all(gates.values())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    mask_archive = write_sampled_mask_archive(
        args.output_dir / "sampled_masks.npz",
        cameras=candidate_cameras,
        frame_index=frame_index,
        masks=selected_masks,
    )
    np.savez_compressed(args.output_dir / "frame_zero_hull.npz", points=hull)
    if grounded:
        source_reference_payload = {
            "mode": "locked-text-grounding",
            "text_prompt": str(object_policy["text_prompt"]),
            "model_id": grounded_config.model_id,
            "model_revision": grounded_config.model_revision,
            "transformers_version": grounded_config.transformers_version,
        }
    elif args.reference_artifact is None:
        reference_artifact = args.output_dir / "source_reference.npz"
        reference_summary = _write_reference(
            reference_artifact,
            rgb=reference_rgb,
            mask=reference_mask,
            object_id=args.object_id,
            frame_index=frame_index,
            camera=reference_camera,
            diagnostics=bootstrap_diagnostics,
            protocol=protocol,
        )
    else:
        reference_artifact = args.reference_artifact
    if not grounded:
        assert reference_artifact is not None and reference_summary is not None
        source_reference_payload = {
            "path": str(reference_artifact.resolve()),
            "archive_sha256": sha256_file(reference_artifact),
            "result_sha256": reference_summary["result_sha256"],
        }

    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTrustMultiviewMasks",
        "protocol_id": protocol["mask_addendum"]["protocol_id"],
        "mask_addendum_file_sha256": protocol["mask_addendum_file_sha256"],
        "authorization": authorization,
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "operation": args.operation,
        "raw_frame_index": frame_index,
        "requested_cameras": cameras,
        "selected_cameras": candidate_cameras,
        "unavailable_camera_count": len(cameras) - len(candidate_cameras),
        "source_reference": source_reference_payload,
        "action_summary": action_summary,
        "candidate_diagnostics": candidate_diagnostics,
        "joint_selection": joint,
        "geometry": {
            "visual_hull_point_count": len(hull),
            "hull_to_controller_distance_m": hull_to_controller_m,
            "frame_zero_hull_to_controller_distance_m": (
                frame_zero_hull_to_controller_m
            ),
            "minimum_group_distance_m": minimum_group_distance_m,
            "geometry_contact_onset_frames": contact_onset_frames,
            "retained_contact_group_count": retained_contact_group_count,
        },
        "thresholds": gates_policy,
        "gates": gates,
        "passed": passed,
        "input_sha256": {
            "robot": sha256_file(episode_dir / "robot" / "robot.npz"),
            "intrinsics": sha256_file(episode_dir / "undistorted_intrinsics.npy"),
            "extrinsics": sha256_file(episode_dir / "extrinsics.npy"),
            "rgb": {camera: _array_sha256(rgb[camera]) for camera in cameras},
            "sam2_checkpoint": sha256_file(args.checkpoint),
        },
        "output_sha256": {
            "sampled_masks": sha256_file(mask_archive),
            "frame_zero_hull": sha256_file(args.output_dir / "frame_zero_hull.npz"),
        },
        "information_boundary": {
            "object_observation_frames_used": [frame_index],
            "known_robot_action_used_for_window_and_contact_qa": True,
            "contact_qa_uses_initial_object_geometry_only": geometry_contact,
            "contact_qa_release_inferred_from_initial_geometry": False,
            "post_initial_object_observation_used": False,
            "tactile_used": False,
            "simulator_residual_used": False,
            "held_object_outcome_used": False,
        },
        "claim_boundary": (
            "frame-zero source-referenced segmentation and contact QA only; "
            "failure returns persistence without changing physics"
        ),
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    (args.output_dir / "mask_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "object_id": args.object_id,
                "episode_id": args.episode_id,
                "raw_frame_index": frame_index,
                "accepted_camera_count": joint["cross_view_consistency"][
                    "accepted_camera_count"
                ],
                "visual_hull_point_count": len(hull),
                "hull_to_controller_distance_m": hull_to_controller_m,
                "passed": passed,
                "result_sha256": payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
