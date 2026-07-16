#!/usr/bin/env python3
"""Build a source-only PhysTwin bundle with sparse automatic contact points."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_dense_source import (
    associate_controller_material_patch,
    fit_source_controller_patch,
    fit_phystwin_support_frame,
    select_sparse_controller_patch,
    sha256_file,
)
from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    reusable_dynamics_result_sha256,
    validate_reusable_dynamics_calibration_request,
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--points-per-group", type=int, default=1)
    parser.add_argument("--input-group-size", type=int, default=768)
    parser.add_argument("--minimum-separation-m", type=float, default=0.004)
    parser.add_argument(
        "--association-mode",
        choices=(
            "frame-zero-nearest",
            "source-fit-comotion",
            "source-transfer-material",
        ),
        default="frame-zero-nearest",
    )
    parser.add_argument("--fit-stop-frame", type=int)
    parser.add_argument("--source-patch-meta", type=Path)
    parser.add_argument("--maximum-initial-distance-m", type=float, default=0.02)
    parser.add_argument("--proximity-weight", type=float, default=0.25)
    parser.add_argument("--support-normal-axis", choices=("x", "y", "z"))
    parser.add_argument("--support-normal-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--support-quantile", type=float, default=0.01)
    parser.add_argument("--support-clearance-m", type=float, default=0.002)
    parser.add_argument("--reusable-dynamics-repo", type=Path)
    parser.add_argument("--reusable-observation-artifact", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    reusable = args.reusable_dynamics_repo is not None
    if reusable != (args.reusable_observation_artifact is not None):
        raise ValueError("reusable dynamics needs both repo and observation artifact")
    if reusable:
        if args.association_mode != "frame-zero-nearest":
            raise ValueError("reusable dynamics permits frame-zero association only")
        assert args.reusable_dynamics_repo is not None
        assert args.reusable_observation_artifact is not None
        protocol_path = (
            args.reusable_dynamics_repo
            / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
        )
        protocol = load_reusable_dynamics_config(protocol_path)
        observation = json.loads(
            args.reusable_observation_artifact.read_text(encoding="utf-8")
        )
        if observation.get("artifact_kind") != "Deform360ReusableDynamicsObservations":
            raise ValueError("unexpected reusable observation artifact")
        if observation.get("parent_config_sha256") != protocol["config_sha256"]:
            raise ValueError("reusable observations use another protocol")
        if observation.get("result_sha256") != reusable_dynamics_result_sha256(
            observation
        ):
            raise ValueError("reusable observation checksum mismatch")
        request = validate_reusable_dynamics_calibration_request(
            protocol,
            object_id=str(observation["object_id"]),
            episode_id=int(observation["episode_id"]),
            operation="one-shot-scoring",
        )
        boundary_manifest = args.reusable_observation_artifact
        boundary = observation
        source_only = False
        claim_boundary = (
            "independent calibration preparation under the frozen reusable-"
            "dynamics protocol; prediction metrics not yet computed"
        )
    else:
        source_manifest = args.episode_dir / "dense_source_smoke.manifest.json"
        boundary = json.loads(source_manifest.read_text(encoding="utf-8"))
        if not boundary.get("source_only"):
            raise ValueError("sparse association accepts only source-only data")
        boundary_manifest = source_manifest
        source_only = True
        request = None
        claim_boundary = (
            "exploratory source-only automatic association; no calibration or "
            "target episode was read"
        )
    source_path = args.episode_dir / "final_data.pkl"
    with source_path.open("rb") as stream:
        data = pickle.load(stream)  # noqa: S301 - trusted local research artifact
    object_points = np.asarray(data["object_points"])
    controller_points = np.asarray(data["controller_points"])
    if controller_points.ndim != 3 or object_points.ndim != 3:
        raise ValueError("PhysTwin bundle trajectories have invalid dimensions")
    if len(controller_points) != len(object_points):
        raise ValueError("object and controller trajectories differ in length")
    total_points = controller_points.shape[1]
    if total_points % args.input_group_size:
        raise ValueError("controller points do not divide into fixed gripper groups")
    group_count = total_points // args.input_group_size
    if args.association_mode == "source-fit-comotion":
        if args.fit_stop_frame is None or not 2 <= args.fit_stop_frame <= len(
            object_points
        ):
            raise ValueError("source-fit association requires a valid fit stop frame")
    elif args.fit_stop_frame is not None:
        raise ValueError("fit stop frame is only valid for source-fit association")
    source_patch_payload = None
    source_patch_sha256 = None
    if args.association_mode == "source-transfer-material":
        if args.source_patch_meta is None:
            raise ValueError("source-transfer association requires patch metadata")
        source_patch_payload = json.loads(
            args.source_patch_meta.read_text(encoding="utf-8")
        )
        source_patch_canonical = dict(source_patch_payload)
        source_patch_result_sha256 = source_patch_canonical.pop(
            "result_sha256", None
        )
        if (
            source_patch_payload.get("source_only") is not True
            or source_patch_payload.get("association_mode")
            != "source-fit-comotion"
            or source_patch_payload.get("held_out_object_motion_read_for_association")
            is not False
            or source_patch_result_sha256
            != hashlib.sha256(_canonical_bytes(source_patch_canonical)).hexdigest()
        ):
            raise ValueError("source patch metadata crosses the information boundary")
        if (
            source_patch_payload.get("group_count") != group_count
            or len(source_patch_payload.get("groups", ())) != group_count
        ):
            raise ValueError("source and target controller group counts differ")
        source_patch_sha256 = sha256_file(args.source_patch_meta)
    elif args.source_patch_meta is not None:
        raise ValueError("source patch metadata is only valid for transfer association")
    selected_global: list[int] = []
    groups = []
    for group in range(group_count):
        start = group * args.input_group_size
        stop = start + args.input_group_size
        association_diagnostics = None
        if args.association_mode == "source-fit-comotion":
            patch, association_diagnostics = fit_source_controller_patch(
                object_points[: args.fit_stop_frame],
                controller_points[: args.fit_stop_frame, start:stop],
                count=args.points_per_group,
                maximum_initial_distance_m=args.maximum_initial_distance_m,
                proximity_weight=args.proximity_weight,
                minimum_separation_m=args.minimum_separation_m,
            )
        elif args.association_mode == "frame-zero-nearest":
            patch = select_sparse_controller_patch(
                object_points[0],
                controller_points[0, start:stop],
                count=args.points_per_group,
                minimum_separation_m=args.minimum_separation_m,
            )
        else:
            assert source_patch_payload is not None
            source_group = source_patch_payload["groups"][group]
            if source_group.get("group") != group:
                raise ValueError("source controller groups are not canonical")
            source_start, source_stop = source_group["input_index_range"]
            if source_start != start or source_stop != stop:
                raise ValueError("source and target controller group layouts differ")
            local_indices = np.asarray(
                source_group["selected_global_indices"], dtype=np.int64
            ) - start
            if len(local_indices) != args.points_per_group:
                raise ValueError("source patch size differs from requested patch size")
            patch = associate_controller_material_patch(
                object_points[0],
                controller_points[0, start:stop],
                local_indices,
            )
            association_diagnostics = {
                "source_patch_meta_sha256": source_patch_sha256,
                "source_patch_result_sha256": source_patch_payload.get(
                    "result_sha256"
                ),
                "selection_rule": (
                    "reuse source-learned gripper material indices; attach to "
                    "nearest target frame-zero object nodes"
                ),
                "target_future_motion_read": False,
            }
        global_indices = patch.controller_indices + start
        selected_global.extend(global_indices.tolist())
        groups.append(
            {
                "group": group,
                "input_index_range": [start, stop],
                "selected_global_indices": global_indices.astype(int).tolist(),
                "nearest_object_indices": (
                    patch.nearest_object_indices.astype(int).tolist()
                ),
                "initial_distances_m": patch.initial_distances_m.tolist(),
                "source_fit": association_diagnostics,
            }
        )
    selected = np.asarray(selected_global, dtype=np.int32)
    sparse = dict(data)
    sparse["controller_points"] = controller_points[:, selected].copy()
    support_frame_payload = None
    frame_suffix = "native-frame"
    if args.support_normal_axis is not None:
        support_axis = {"x": 0, "y": 1, "z": 2}[args.support_normal_axis]
        support_frame = fit_phystwin_support_frame(
            object_points[0],
            support_axis=support_axis,
            free_space_sign=args.support_normal_sign,
            support_quantile=args.support_quantile,
            clearance_m=args.support_clearance_m,
        )
        for key in ("object_points", "surface_points", "interior_points"):
            sparse[key] = support_frame.transform(np.asarray(sparse[key])).astype(
                np.asarray(sparse[key]).dtype,
                copy=False,
            )
        sparse["controller_points"] = support_frame.transform(
            sparse["controller_points"]
        ).astype(controller_points.dtype, copy=False)
        support_frame_payload = {
            "rotation_world_to_sim": (
                support_frame.rotation_world_to_sim.tolist()
            ),
            "translation_sim_m": support_frame.translation_sim_m.tolist(),
            "support_axis": support_frame.support_axis,
            "free_space_sign": support_frame.free_space_sign,
            "support_location_world_m": support_frame.support_location_world_m,
            "clearance_m": support_frame.clearance_m,
            "fit_information_scope": "frame-zero-object-geometry-only",
        }
        sign_label = "positive" if args.support_normal_sign > 0 else "negative"
        frame_suffix = f"support-{sign_label}-{args.support_normal_axis}-to-positive-z"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.association_mode == "frame-zero-nearest":
        filename = (
            f"final_data_sparse_controller_k{args.points_per_group}_per_group_"
            f"{frame_suffix}.pkl"
        )
    elif args.association_mode == "source-fit-comotion":
        filename = (
            f"final_data_sparse_controller_k{args.points_per_group}_per_group_"
            f"source-comotion-f{args.fit_stop_frame}_{frame_suffix}.pkl"
        )
    else:
        assert source_patch_sha256 is not None
        filename = (
            f"final_data_sparse_controller_k{args.points_per_group}_per_group_"
            f"source-transfer-{source_patch_sha256[:12]}_{frame_suffix}.pkl"
        )
    output_path = args.output_dir / filename
    with output_path.open("wb") as stream:
        pickle.dump(sparse, stream, protocol=4)
    payload: dict[str, Any] = {
        "schema": "bayesian-phystwin/deform360-sparse-controller-bundle/v1",
        "source_only": source_only,
        "source_manifest_sha256": sha256_file(boundary_manifest),
        "source_final_data_sha256": sha256_file(source_path),
        "output_final_data": str(output_path.resolve()),
        "output_final_data_sha256": sha256_file(output_path),
        "frame_count": len(controller_points),
        "object_point_count": object_points.shape[1],
        "input_controller_point_count": total_points,
        "output_controller_point_count": len(selected),
        "input_group_size": args.input_group_size,
        "group_count": group_count,
        "points_per_group": args.points_per_group,
        "minimum_separation_m": args.minimum_separation_m,
        "association_mode": args.association_mode,
        "fit_stop_frame": args.fit_stop_frame,
        "source_patch_meta": (
            str(args.source_patch_meta.resolve())
            if args.source_patch_meta is not None
            else None
        ),
        "source_patch_meta_sha256": source_patch_sha256,
        "maximum_initial_distance_m": args.maximum_initial_distance_m,
        "proximity_weight": args.proximity_weight,
        "groups": groups,
        "support_frame": support_frame_payload,
        "association_information_scope": (
            "frame-zero-geometry-only"
            if args.association_mode == "frame-zero-nearest"
            else (
                f"source-train-frames-[0,{args.fit_stop_frame})-only"
                if args.association_mode == "source-fit-comotion"
                else "source-learned-gripper-patch-plus-target-frame-zero-geometry"
            )
        ),
        "object_motion_used_for_association": (
            args.association_mode == "source-fit-comotion"
        ),
        "source_object_motion_used_to_fit_controller_patch": (
            args.association_mode == "source-transfer-material"
        ),
        "held_out_object_motion_read_for_association": False,
        "claim_boundary": claim_boundary,
    }
    if reusable:
        payload["reusable_dynamics_request"] = request
    payload["result_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
