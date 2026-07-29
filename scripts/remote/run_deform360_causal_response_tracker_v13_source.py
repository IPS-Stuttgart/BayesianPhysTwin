#!/usr/bin/env python3
"""Seal one V13 disjoint-panel TAPNext++ source prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_causal_response_adaptive_query import (
    validate_adaptive_causal_response_query_artifacts,
)
from bayesian_phystwin.deform360_causal_response_tracker import (
    PROTOCOL_ID,
    CrossPanelProviderConfig,
    birth_associations_from_adaptive_query,
    corroborate_disjoint_panels,
    write_causal_response_tracker_artifacts,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    load_complete_camera_geometry,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_provider import (
    _decode_rgb_prefix,
    _read_h5_prefix,
)
from bayesian_phystwin.observation_belief import array_sha256, file_sha256
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    TAPNEXT_CHECKPOINT_SHA256,
    TAPNEXT_REVISION,
    DynamicMultiviewConfig,
    fuse_dynamic_tapnextpp_multiview,
)
from bayesian_phystwin.tapnextpp_dynamic_runtime import (
    DynamicTAPNextPPRuntimeConfig,
    run_dynamic_tapnextpp_births,
)

CONFIG_RELATIVE_PATH = Path(
    "configs/sota/deform360_causal_response_tracker_v13.json"
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _git_output(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--adaptive-query-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tapnet-root", type=Path, required=True)
    parser.add_argument("--tapnextpp-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_protocol(
    repo: Path,
    case_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = repo / CONFIG_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    records = {str(record["case"]): record for record in payload["cases"]}
    _require(
        case_id in records,
        "case is outside the frozen V13 source panel",
    )
    DynamicTAPNextPPRuntimeConfig(**payload["tracker_runtime"])
    DynamicMultiviewConfig(**payload["multiview"])
    CrossPanelProviderConfig(**payload["cross_panel"])
    return path, payload, records[case_id]


def _load_model(
    tapnet_root: Path,
    checkpoint: Path,
    device_name: str,
    tracker_config: Mapping[str, Any],
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    revision = _git_output(tapnet_root, "rev-parse", "HEAD")
    _require(revision == TAPNEXT_REVISION, "TAPNet revision changed")
    _require(
        file_sha256(checkpoint) == TAPNEXT_CHECKPOINT_SHA256,
        "TAPNext++ checkpoint checksum changed",
    )
    if str(tapnet_root) not in sys.path:
        sys.path.insert(0, str(tapnet_root))
    from tapnet.tapnextpp.votsp2026 import utils as tapnext_utils
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    device = torch.device(device_name)
    _require(device.type == "cuda", "V13 source prediction requires CUDA")
    torch.manual_seed(73)
    torch.cuda.manual_seed_all(73)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = TAPNextPP.from_checkpoint(
        checkpoint,
        device=device,
        half_precision=False,
        compile_model=False,
        input_resolution=int(tracker_config["input_resolution"]),
    )
    return (
        model,
        tapnext_utils,
        {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "tapnet_revision": revision,
            "tapnextpp_checkpoint_sha256": TAPNEXT_CHECKPOINT_SHA256,
            "random_seed": 73,
        },
    )


def _load_camera_prefix(
    processed: Path,
    camera_names: tuple[str, ...],
    *,
    frame_count: int,
    depth_scale_to_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rgbs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    hashes: dict[str, Any] = {}
    for camera in camera_names:
        directory = processed / camera
        rgb = _decode_rgb_prefix(
            directory / "undistorted.mp4",
            frame_count,
        )
        encoded_depth = _read_h5_prefix(
            directory / "rendered_depth.h5",
            frame_count,
        )
        mask = _read_h5_prefix(
            directory / "mask_refined.h5",
            frame_count,
        ).astype(bool, copy=False)
        depth = encoded_depth.astype(np.float32) * depth_scale_to_m
        _require(
            rgb.shape[:-1] == depth.shape == mask.shape,
            f"causal camera arrays differ: {camera}",
        )
        rgbs.append(rgb)
        depths.append(depth)
        masks.append(mask)
        hashes[camera] = {
            "decoded_rgb_sha256": array_sha256(rgb),
            "decoded_depth_m_sha256": array_sha256(depth),
            "decoded_mask_sha256": array_sha256(mask),
        }
    return np.stack(rgbs), np.stack(depths), np.stack(masks), hashes


def _panel_result(
    panel: np.ndarray,
    runtime_tracks: np.ndarray,
    runtime_visibility: np.ndarray,
    depths: np.ndarray,
    masks: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    associations: Any,
    query_points_world_m: np.ndarray,
    *,
    config: DynamicMultiviewConfig,
) -> Any:
    return fuse_dynamic_tapnextpp_multiview(
        runtime_tracks[panel],
        runtime_visibility[panel],
        depths[panel],
        masks[panel],
        intrinsics[panel],
        camera_to_world[panel],
        query_points_world_m,
        association_valid=associations.valid[panel],
        association_probability=associations.association_probability[
            panel
        ],
        association_entropy=associations.association_entropy[panel],
        assignment_pixel_covariance_px2=(
            associations.candidate_pixel_covariance_px2[panel]
        ),
        config=config,
    )


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    revision = _git_output(repo, "rev-parse", "HEAD")
    _require(not _git_output(repo, "status", "--porcelain"), "repository is dirty")
    protocol_path, protocol, case_record = _load_protocol(repo, args.case)
    output = args.output_dir.resolve()
    _require(not output.exists(), "V13 tracker case output already exists")
    output.mkdir(parents=True)

    query_dir = args.adaptive_query_dir.resolve()
    query_report, query_arrays = (
        validate_adaptive_causal_response_query_artifacts(query_dir)
    )
    _require(
        query_report.get("case") == args.case
        and query_report.get("result_sha256")
        == case_record["adaptive_query_result_sha256"],
        "adaptive V13 carrier differs from the tracker lock",
    )
    if query_report["status"] == "abstained":
        disposition = {
            "case": args.case,
            "status": "exact_query_abstention",
            "adaptive_query_result_sha256": query_report["result_sha256"],
            "tracker_executed": False,
            "identity_target_read": False,
            "state_or_readout_update_constructed": False,
            "future_prediction_metric_read": False,
            "held_v8_artifact_or_process_access": False,
        }
        (output / "disposition.json").write_text(
            json.dumps(disposition, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(disposition, sort_keys=True))
        return 0

    schedule = query_report["schedule"]
    query = schedule["query_schedule"]
    frame_count = int(protocol["prefix"]["frame_count"])
    update_frame = int(protocol["prefix"]["update_frame"])
    _require(
        frame_count == 58
        and update_frame == 57
        and query["config"]["prefix_frame_count"] == frame_count,
        "prefix boundary changed",
    )
    processed = args.processed_episode_dir.resolve()
    geometry = load_complete_camera_geometry(
        processed,
        minimum_complete_camera_count=int(
            protocol["minimum_complete_camera_count"]
        ),
        frame_count=frame_count,
    )
    geometry_by_name = {
        name: index for index, name in enumerate(geometry.camera_names)
    }
    camera_names = tuple(map(str, query["camera_ids"]))
    _require(
        len(camera_names) == 8
        and all(name in geometry_by_name for name in camera_names),
        "a V13-selected camera lacks a complete causal prefix",
    )
    geometry_indices = np.asarray(
        [geometry_by_name[name] for name in camera_names],
        dtype=np.int64,
    )
    rgbs, depths, masks, causal_hashes = _load_camera_prefix(
        processed,
        camera_names,
        frame_count=frame_count,
        depth_scale_to_m=float(protocol["depth_scale_to_m"]),
    )
    intrinsics = geometry.intrinsics[geometry_indices]
    camera_to_world = geometry.camera_to_world[geometry_indices]
    associations = birth_associations_from_adaptive_query(
        query_report,
        query_arrays,
    )
    model, tapnext_utils, runtime_provenance = _load_model(
        args.tapnet_root.resolve(),
        args.tapnextpp_checkpoint.resolve(),
        args.device,
        protocol["tracker_runtime"],
    )
    runtime = run_dynamic_tapnextpp_births(
        model,
        rgbs,
        associations,
        np.zeros(16, dtype=np.int64),
        np.full(16, update_frame, dtype=np.int64),
        tapnext_utils,
        config=DynamicTAPNextPPRuntimeConfig(
            **protocol["tracker_runtime"]
        ),
    )
    import torch

    torch.cuda.synchronize(model.device)
    runtime_provenance["peak_gpu_memory_gib"] = (
        torch.cuda.max_memory_allocated(model.device) / (1024**3)
    )
    proposal_panel = np.asarray(
        query["proposal_camera_indices"],
        dtype=np.int64,
    )
    validation_panel = np.asarray(
        query["validation_camera_indices"],
        dtype=np.int64,
    )
    arm = str(schedule["arm"])
    minimum_claim_views = int(
        protocol["arm_settings"][arm]["minimum_claim_view_count"]
    )
    multiview = DynamicMultiviewConfig(
        **{
            **protocol["multiview"],
            "minimum_claim_view_count": minimum_claim_views,
        }
    )
    proposal = _panel_result(
        proposal_panel,
        runtime.tracks_xy,
        runtime.visibility_probability,
        depths,
        masks,
        intrinsics,
        camera_to_world,
        associations,
        query_arrays["query_points_world_m"],
        config=multiview,
    )
    validation = _panel_result(
        validation_panel,
        runtime.tracks_xy,
        runtime.visibility_probability,
        depths,
        masks,
        intrinsics,
        camera_to_world,
        associations,
        query_arrays["query_points_world_m"],
        config=multiview,
    )
    prediction = corroborate_disjoint_panels(
        proposal,
        validation,
        config=CrossPanelProviderConfig(**protocol["cross_panel"]),
    )
    provider_report = write_causal_response_tracker_artifacts(
        output / "provider",
        query_report,
        query_arrays,
        runtime,
        proposal,
        validation,
        prediction,
        case_id=args.case,
        repository_revision=revision,
        protocol_path=protocol_path,
        query_output_dir=query_dir,
        runtime_provenance=runtime_provenance,
        causal_input_sha256={
            "camera_certificate": geometry.artifact_sha256,
            "camera_prefixes": causal_hashes,
            "intrinsics": array_sha256(intrinsics),
            "camera_to_world": array_sha256(camera_to_world),
        },
        update_frame=update_frame,
    )
    disposition = {
        "case": args.case,
        "status": "tracker_prediction_sealed",
        "adaptive_query_result_sha256": query_report["result_sha256"],
        "provider_result_sha256": provider_report["result_sha256"],
        "tracker_executed": True,
        "identity_target_read": False,
        "state_or_readout_update_constructed": False,
        "future_prediction_metric_read": False,
        "held_v8_artifact_or_process_access": False,
    }
    (output / "disposition.json").write_text(
        json.dumps(disposition, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                **disposition,
                "accepted_endpoint_count": provider_report[
                    "accepted_endpoint_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
