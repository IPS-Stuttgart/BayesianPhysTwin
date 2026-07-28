#!/usr/bin/env python3
"""Run one source-only TAPNext++ sentinel-debias development case."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    load_complete_camera_geometry,
    load_selected_complete_causal_inputs,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    SET_VALUED_MIXTURE_ASSIMILATION,
    build_birth_anchored_measurements,
    predict_dynamic_tapnextpp_candidate,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    PHYSICAL_ARCHIVE_FILENAME,
    PHYSICAL_MANIFEST_FILENAME,
    validate_dynamic_physical_artifacts,
)
from bayesian_phystwin.deform360_sentinel_assimilation import (
    build_sentinel_debiased_measurements,
)
from bayesian_phystwin.deform360_sentinel_query_schedule import (
    PREFIX_END_FRAME,
    build_deform360_sentinel_query_schedule,
)
from bayesian_phystwin.observation_belief import array_sha256, file_sha256
from bayesian_phystwin.tapnextpp_birth_association import (
    SET_VALUED_COVARIANCE_ASSOCIATION,
    BirthAssociationConfig,
)
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY,
    TAPNEXT_CHECKPOINT_SHA256,
    TAPNEXT_REVISION,
    DynamicMultiviewConfig,
    dynamic_multiview_result_sha256,
    fuse_dynamic_tapnextpp_multiview,
)
from bayesian_phystwin.tapnextpp_dynamic_runtime import (
    build_dynamic_birth_associations,
    run_dynamic_tapnextpp_births,
)

PROTOCOL_ID = "deform360-dynamic-tapnextpp-sentinel-v5-source-development"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tapnet-root", type=Path, required=True)
    parser.add_argument("--tapnextpp-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_model(
    tapnet_root: Path,
    checkpoint: Path,
    device_name: str,
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
    _require(device.type == "cuda", "source development requires CUDA")
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
        input_resolution=512,
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


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        **{
            name: np.ascontiguousarray(np.asarray(values))
            for name, values in arrays.items()
        },
    )
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    revision = _require_clean_repository(repo)
    output = args.output_dir.resolve()
    _require(not output.exists(), "source-development output already exists")
    output.mkdir(parents=True)
    physical_dir = args.physical_dir.resolve()
    if not (physical_dir / PHYSICAL_MANIFEST_FILENAME).is_file():
        physical_dir = physical_dir / "sealed_physical"
    manifest, physical = validate_dynamic_physical_artifacts(physical_dir)
    _require(manifest["case"] == args.case, "physical case changed")

    processed = args.processed_episode_dir.resolve()
    geometry = load_complete_camera_geometry(processed)
    schedule = build_deform360_sentinel_query_schedule(
        physical["physical_prediction_m"],
        physical["graph_basis"],
        geometry.intrinsics,
        geometry.camera_to_world,
        geometry.image_shapes_hw,
        geometry.camera_names,
    )
    camera_inputs = load_selected_complete_causal_inputs(
        processed,
        geometry,
        schedule.camera_panel.camera_indices,
    )
    associations = build_dynamic_birth_associations(
        schedule,
        physical["physical_prediction_m"],
        camera_inputs.intrinsics,
        camera_inputs.camera_to_world,
        camera_inputs.depths_m,
        camera_inputs.object_masks,
        input_camera_indices=camera_inputs.camera_indices,
        config=BirthAssociationConfig(
            association_mode=SET_VALUED_COVARIANCE_ASSOCIATION,
        ),
    )

    model, tapnext_utils, runtime_provenance = _load_model(
        args.tapnet_root.resolve(),
        args.tapnextpp_checkpoint.resolve(),
        args.device,
    )
    runtime = run_dynamic_tapnextpp_births(
        model,
        camera_inputs.rgbs,
        associations,
        schedule.birth_frames,
        schedule.update_frames,
        tapnext_utils,
    )
    import torch

    torch.cuda.synchronize(model.device)
    runtime_provenance.update(
        {
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated(model.device) / (1024**3)
            ),
            "rollout_count": runtime.rollout_count,
            "model_frame_count": runtime.model_frame_count,
            "elapsed_seconds": runtime.elapsed_seconds,
        }
    )
    provider = fuse_dynamic_tapnextpp_multiview(
        runtime.tracks_xy,
        runtime.visibility_probability,
        camera_inputs.depths_m,
        camera_inputs.object_masks,
        camera_inputs.intrinsics,
        camera_inputs.camera_to_world,
        associations.query_points_world_m,
        association_valid=associations.valid,
        association_probability=associations.association_probability,
        association_entropy=associations.association_entropy,
        assignment_pixel_covariance_px2=(
            associations.candidate_pixel_covariance_px2
        ),
        config=DynamicMultiviewConfig(
            assignment_uncertainty_mode=(
                COVARIANCE_ONLY_ASSIGNMENT_UNCERTAINTY
            )
        ),
    )
    raw_measurements = build_birth_anchored_measurements(
        provider,
        schedule,
        physical["physical_prediction_m"],
    )
    debias = build_sentinel_debiased_measurements(
        raw_measurements,
        schedule,
        physical["physical_prediction_m"],
    )
    assimilation_report, assimilation_arrays = (
        predict_dynamic_tapnextpp_candidate(
            physical["physical_prediction_m"],
            physical["persistence_prediction_m"],
            debias.measurements,
            assimilation_mode=SET_VALUED_MIXTURE_ASSIMILATION,
        )
    )
    if not debias.applied:
        _require(
            np.array_equal(
                assimilation_arrays[CANDIDATE_ARM][
                    PREFIX_END_FRAME + 1 :
                ],
                assimilation_arrays[PERSISTENCE_ARM][
                    PREFIX_END_FRAME + 1 :
                ],
            ),
            "sentinel rejection did not retain bit-exact future persistence",
        )

    provider_arrays = {
        "trajectory_world_m": provider.trajectory_world_m,
        "accepted_support": provider.accepted_support,
        "proposal_available": provider.proposal_available,
        "prior_reliability": provider.prior_reliability,
        "association_probability": provider.association_probability,
        "local_covariance_m2": provider.local_covariance_m2,
        "naive_independent_covariance_m2": (
            provider.naive_independent_covariance_m2
        ),
        "assignment_mixture_spread_m2": (
            provider.assignment_mixture_spread_m2
        ),
        "independent_support_count": provider.independent_support_count,
        "raw_support_count": provider.raw_support_count,
        "reprojection_rmse_px": provider.reprojection_rmse_px,
        "depth_residual_rmse_m": provider.depth_residual_rmse_m,
        "entity_ids": schedule.entity_ids,
        "birth_frames": schedule.birth_frames,
        "update_frames": schedule.update_frames,
        "query_roles": schedule.query_roles,
    }
    measurement_arrays = {
        "measurement_m": debias.measurements.measurement_m,
        "covariance_m2": debias.measurements.covariance_m2,
        "prior_reliability": debias.measurements.prior_reliability,
        "association_probability": (
            debias.measurements.association_probability
        ),
        "available": debias.measurements.available,
        "entity_ids": debias.measurements.entity_ids,
    }
    provider_path = output / "provider_arrays.npz"
    measurement_path = output / "sentinel_measurements.npz"
    assimilation_path = output / "assimilation_arrays.npz"
    _write_npz(provider_path, provider_arrays)
    _write_npz(measurement_path, measurement_arrays)
    _write_npz(assimilation_path, dict(assimilation_arrays))

    schedule_payload = schedule.descriptor()
    schedule_payload["schedule_sha256"] = schedule.artifact_sha256
    (output / "query_schedule.json").write_text(
        json.dumps(
            schedule_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    endpoint_supported = (
        provider.accepted_support[
            schedule.birth_frames,
            np.arange(len(schedule.birth_frames)),
        ]
        & provider.accepted_support[
            schedule.update_frames,
            np.arange(len(schedule.update_frames)),
        ]
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": (
            "Deform360DynamicTAPNextPPSentinelV5SourceDevelopment"
        ),
        "protocol_id": PROTOCOL_ID,
        "status": "post_open_source_development_not_confirmation",
        "case": args.case,
        "case_hash": manifest["case_hash"],
        "repository_revision": revision,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            **runtime_provenance,
        },
        "inputs_sha256": {
            "physical_manifest": file_sha256(
                physical_dir / PHYSICAL_MANIFEST_FILENAME
            ),
            "physical_archive": file_sha256(
                physical_dir / PHYSICAL_ARCHIVE_FILENAME
            ),
            "camera_certificate": geometry.artifact_sha256,
            "query_schedule": schedule.artifact_sha256,
        },
        "support": {
            "complete_camera_count": len(geometry.camera_names),
            "selected_camera_count": len(camera_inputs.camera_names),
            "scheduled_identity_count": len(schedule.entity_ids),
            "birth_and_update_supported_count": int(
                np.sum(endpoint_supported)
            ),
            "birth_and_update_supported_fraction": float(
                np.mean(endpoint_supported)
            ),
        },
        "sentinel_debias": debias.report(),
        "provider_result_sha256": dynamic_multiview_result_sha256(provider),
        "assimilation_report": assimilation_report,
        "outputs": {
            "provider_arrays_file_sha256": file_sha256(provider_path),
            "provider_array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(provider_arrays.items())
            },
            "sentinel_measurements_file_sha256": file_sha256(
                measurement_path
            ),
            "sentinel_measurement_array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(measurement_arrays.items())
            },
            "assimilation_arrays_file_sha256": file_sha256(
                assimilation_path
            ),
            "assimilation_array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(assimilation_arrays.items())
            },
        },
        "information_boundary": {
            "maximum_rgb_depth_mask_frame_read": PREFIX_END_FRAME,
            "maximum_physical_frame_read": PREFIX_END_FRAME,
            "future_identity_read": False,
            "future_object_observation_read": False,
            "target_metric_read": False,
            "v1_sealed_target_cohort_read": False,
            "held_v8_artifact_read": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(report)
    (output / "source_development_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
