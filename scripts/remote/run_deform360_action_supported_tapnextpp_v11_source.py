#!/usr/bin/env python3
"""Run one sealed action-supported TAPNext++ source prediction."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_action_supported_tapnextpp import (
    PROTOCOL_ID,
    ActionSupportedQueryConfig,
    build_action_supported_query_schedule,
    write_action_supported_provider_artifacts,
    write_action_supported_query_artifacts,
)
from bayesian_phystwin.deform360_active_query_feasibility import (
    validate_active_query_feasibility_artifacts,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    load_complete_camera_geometry,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    PHYSICAL_ARCHIVE_FILENAME,
    PHYSICAL_MANIFEST_FILENAME,
    validate_dynamic_physical_artifacts,
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
    "configs/sota/deform360_action_supported_tapnextpp_source_v11.json"
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


def _require_clean_repository(repo: Path) -> str:
    revision = _git_output(repo, "rev-parse", "HEAD")
    _require(not _git_output(repo, "status", "--porcelain"), "repository is dirty")
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--v10-output-dir", type=Path, required=True)
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
    _require(case_id in records, "case is outside the frozen V11 source panel")
    ActionSupportedQueryConfig(**payload["query"])
    DynamicTAPNextPPRuntimeConfig(**payload["tracker_runtime"])
    DynamicMultiviewConfig(**payload["multiview"])
    return path, payload, records[case_id]


def _load_model(
    tapnet_root: Path,
    checkpoint: Path,
    device_name: str,
    tracker_config: MappingLike,
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
    _require(device.type == "cuda", "V11 source prediction requires CUDA")
    torch.manual_seed(72)
    torch.cuda.manual_seed_all(72)
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
        },
    )


MappingLike = dict[str, Any]


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


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    revision = _require_clean_repository(repo)
    protocol_path, protocol, case_record = _load_protocol(repo, args.case)
    output = args.output_dir.resolve()
    _require(not output.exists(), "V11 case output already exists")
    output.mkdir(parents=True)

    physical_dir = args.physical_dir.resolve()
    physical_manifest, physical = validate_dynamic_physical_artifacts(
        physical_dir
    )
    _require(
        physical_manifest.get("partition") == "source"
        and physical_manifest.get("case") == args.case,
        "physical artifact is not the frozen source case",
    )
    physical_archive = physical_dir / PHYSICAL_ARCHIVE_FILENAME
    _require(
        file_sha256(physical_archive)
        == case_record["physical_archive_sha256"],
        "physical archive differs from the V11 lock",
    )
    v10_output = args.v10_output_dir.resolve()
    v10_report, v10_arrays = validate_active_query_feasibility_artifacts(
        v10_output
    )
    _require(
        v10_report.get("case") == args.case
        and v10_report.get("result_sha256")
        == case_record["v10_result_sha256"],
        "V10 carrier differs from the V11 lock",
    )
    schedule = build_action_supported_query_schedule(
        v10_report,
        v10_arrays,
        physical["physical_prediction_m"][0],
        physical["graph_basis"],
        physical["action_support"],
        config=ActionSupportedQueryConfig(**protocol["query"]),
    )
    query_output = output / "query"
    query_report = write_action_supported_query_artifacts(
        query_output,
        schedule,
        case_id=args.case,
        repository_revision=revision,
        protocol_path=protocol_path,
        v10_output_dir=v10_output,
        physical_manifest_path=physical_dir / PHYSICAL_MANIFEST_FILENAME,
        physical_archive_path=physical_archive,
    )
    if not schedule.admitted:
        (output / "disposition.json").write_text(
            json.dumps(
                {
                    "case": args.case,
                    "status": "query_budget_abstention",
                    "query_result_sha256": query_report["result_sha256"],
                    "tracker_executed": False,
                    "identity_target_read": False,
                    "state_update_constructed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 0

    processed = args.processed_episode_dir.resolve()
    geometry = load_complete_camera_geometry(
        processed,
        minimum_complete_camera_count=protocol[
            "minimum_complete_camera_count"
        ],
        frame_count=schedule.config.prefix_frame_count,
    )
    geometry_by_name = {
        name: index for index, name in enumerate(geometry.camera_names)
    }
    _require(
        all(name in geometry_by_name for name in schedule.camera_panel.camera_names),
        "a V10-selected camera lacks a complete V11 causal prefix",
    )
    local_indices = np.asarray(
        [
            geometry_by_name[name]
            for name in schedule.camera_panel.camera_names
        ],
        dtype=np.int64,
    )
    rgbs, depths, masks, causal_hashes = _load_camera_prefix(
        processed,
        schedule.camera_panel.camera_names,
        frame_count=schedule.config.prefix_frame_count,
        depth_scale_to_m=float(protocol["depth_scale_to_m"]),
    )
    intrinsics = geometry.intrinsics[local_indices]
    camera_to_world = geometry.camera_to_world[local_indices]
    associations = schedule.birth_associations()
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
        schedule.plan.seed_frames,
        np.full(
            len(schedule.plan.node_ids),
            schedule.config.update_frame,
            dtype=np.int64,
        ),
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
    provider = fuse_dynamic_tapnextpp_multiview(
        runtime.tracks_xy,
        runtime.visibility_probability,
        depths,
        masks,
        intrinsics,
        camera_to_world,
        schedule.query_points_world_m,
        association_valid=associations.valid,
        association_probability=associations.association_probability,
        association_entropy=associations.association_entropy,
        assignment_pixel_covariance_px2=(
            associations.candidate_pixel_covariance_px2
        ),
        config=DynamicMultiviewConfig(**protocol["multiview"]),
    )
    provider_report = write_action_supported_provider_artifacts(
        output / "provider",
        schedule,
        runtime,
        provider,
        case_id=args.case,
        repository_revision=revision,
        protocol_path=protocol_path,
        query_output_dir=query_output,
        runtime_provenance=runtime_provenance,
        causal_input_sha256={
            "camera_certificate": geometry.artifact_sha256,
            "camera_prefixes": causal_hashes,
            "intrinsics": array_sha256(intrinsics),
            "camera_to_world": array_sha256(camera_to_world),
        },
    )
    (output / "disposition.json").write_text(
        json.dumps(
            {
                "case": args.case,
                "status": "provider_prediction_sealed",
                "query_result_sha256": query_report["result_sha256"],
                "provider_result_sha256": provider_report["result_sha256"],
                "tracker_executed": True,
                "identity_target_read": False,
                "state_update_constructed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case": args.case,
                "status": "provider_prediction_sealed",
                "accepted_endpoint_count": int(
                    np.sum(provider.accepted_support[-1])
                ),
                "result_sha256": provider_report["result_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
