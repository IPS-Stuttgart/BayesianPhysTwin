#!/usr/bin/env python3
"""Run the source-only direct RGB-D sentinel update."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
    build_direct_depth_birth_anchored_measurements,
    build_direct_depth_endpoint_observations,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    load_complete_camera_geometry,
    load_selected_complete_causal_inputs,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    SET_VALUED_MIXTURE_ASSIMILATION,
    predict_dynamic_tapnextpp_candidate,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    PHYSICAL_ARCHIVE_FILENAME,
    PHYSICAL_MANIFEST_FILENAME,
    validate_dynamic_physical_artifacts,
)
from bayesian_phystwin.deform360_prefix_support_screen import (
    PrefixAssociationSupportConfig,
    build_prefix_association_support_screen,
)
from bayesian_phystwin.deform360_sentinel_assimilation import (
    build_sentinel_debiased_measurements,
)
from bayesian_phystwin.deform360_sentinel_query_schedule import (
    DIRECT_DEPTH_PROTOCOL_ID,
    Deform360SentinelQueryConfig,
    build_deform360_sentinel_query_schedule,
)
from bayesian_phystwin.observation_belief import array_sha256, file_sha256
from bayesian_phystwin.phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
)


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
    _require(
        not _git_output(repo, "status", "--porcelain"),
        "repository is dirty",
    )
    return revision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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
    schedule_config = Deform360SentinelQueryConfig(
        query_birth_frame=51,
        protocol_id=DIRECT_DEPTH_PROTOCOL_ID,
    )
    provisional_schedule = build_deform360_sentinel_query_schedule(
        physical["physical_prediction_m"],
        physical["graph_basis"],
        geometry.intrinsics,
        geometry.camera_to_world,
        geometry.image_shapes_hw,
        geometry.camera_names,
        config=schedule_config,
    )
    camera_inputs = load_selected_complete_causal_inputs(
        processed,
        geometry,
        provisional_schedule.camera_panel.camera_indices,
    )
    prefix_support = build_prefix_association_support_screen(
        physical["physical_prediction_m"],
        camera_inputs.intrinsics,
        camera_inputs.camera_to_world,
        camera_inputs.depths_m,
        camera_inputs.object_masks,
        config=PrefixAssociationSupportConfig(
            birth_frame=schedule_config.query_birth_frame,
            update_frame=schedule_config.query_update_frame,
            minimum_camera_support=schedule_config.minimum_camera_support,
        ),
    )
    schedule = build_deform360_sentinel_query_schedule(
        physical["physical_prediction_m"],
        physical["graph_basis"],
        geometry.intrinsics,
        geometry.camera_to_world,
        geometry.image_shapes_hw,
        geometry.camera_names,
        candidate_entity_ids=prefix_support.eligible_entity_ids,
        config=schedule_config,
    )
    _require(
        np.array_equal(
            schedule.camera_panel.camera_indices,
            provisional_schedule.camera_panel.camera_indices,
        ),
        "prefix support changed the target-free camera panel",
    )

    provider = build_direct_depth_endpoint_observations(
        physical["physical_prediction_m"],
        schedule,
        camera_inputs.intrinsics,
        camera_inputs.camera_to_world,
        camera_inputs.depths_m,
        camera_inputs.object_masks,
        config=DirectDepthEndpointConfig(
            minimum_camera_support=schedule.config.minimum_camera_support,
        ),
    )
    raw_measurements = build_direct_depth_birth_anchored_measurements(
        provider,
        physical["physical_prediction_m"],
    )
    debias = build_sentinel_debiased_measurements(
        raw_measurements,
        schedule,
        physical["physical_prediction_m"],
    )
    assimilation_report, assimilation_arrays = predict_dynamic_tapnextpp_candidate(
        physical["physical_prediction_m"],
        physical["persistence_prediction_m"],
        debias.measurements,
        assimilation_mode=SET_VALUED_MIXTURE_ASSIMILATION,
    )
    if not debias.applied:
        _require(
            np.array_equal(
                assimilation_arrays[CANDIDATE_ARM][
                    schedule.config.query_update_frame + 1 :
                ],
                assimilation_arrays[PERSISTENCE_ARM][
                    schedule.config.query_update_frame + 1 :
                ],
            ),
            "sentinel rejection did not retain bit-exact future persistence",
        )

    provider_arrays = {
        "endpoint_frames": provider.endpoint_frames,
        "entity_ids": provider.entity_ids,
        "point_world_m": provider.point_world_m,
        "covariance_m2": provider.covariance_m2,
        "accepted_support": provider.accepted_support,
        "association_probability": provider.association_probability,
        "support_count": provider.support_count,
        "maximum_view_scatter_m": provider.maximum_view_scatter_m,
        "birth_frames": schedule.birth_frames,
        "update_frames": schedule.update_frames,
        "query_roles": schedule.query_roles,
    }
    measurement_arrays = {
        "measurement_m": debias.measurements.measurement_m,
        "covariance_m2": debias.measurements.covariance_m2,
        "prior_reliability": debias.measurements.prior_reliability,
        "association_probability": (debias.measurements.association_probability),
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
    schedule_path = output / "query_schedule.json"
    schedule_path.write_text(
        json.dumps(
            schedule_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    prefix_support_payload = prefix_support.descriptor()
    prefix_support_payload["artifact_sha256"] = prefix_support.artifact_sha256
    prefix_support_path = output / "prefix_support_screen.json"
    prefix_support_path.write_text(
        json.dumps(
            prefix_support_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    endpoint_supported = np.all(provider.accepted_support, axis=0)
    active_mask = schedule.query_roles == ACTIVE_QUERY_ROLE
    sentinel_mask = schedule.query_roles == SENTINEL_QUERY_ROLE
    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DirectDepthSentinelV8SourceDevelopment",
        "protocol_id": DIRECT_DEPTH_PROTOCOL_ID,
        "status": "post_open_source_development_not_confirmation",
        "case": args.case,
        "case_hash": manifest["case_hash"],
        "repository_revision": revision,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "inputs_sha256": {
            "physical_manifest": file_sha256(physical_dir / PHYSICAL_MANIFEST_FILENAME),
            "physical_archive": file_sha256(physical_dir / PHYSICAL_ARCHIVE_FILENAME),
            "camera_certificate": geometry.artifact_sha256,
            "query_schedule": schedule.artifact_sha256,
            "prefix_support_screen": prefix_support.artifact_sha256,
        },
        "support": {
            "complete_camera_count": len(geometry.camera_names),
            "selected_camera_count": len(camera_inputs.camera_names),
            "prefix_screen_candidate_count": len(prefix_support.entity_ids),
            "prefix_screen_eligible_count": int(np.sum(prefix_support.eligible)),
            "scheduled_identity_count": len(schedule.entity_ids),
            "active_identity_count": int(np.sum(active_mask)),
            "sentinel_identity_count": int(np.sum(sentinel_mask)),
            "birth_and_update_supported_count": int(np.sum(endpoint_supported)),
            "birth_and_update_supported_fraction": float(np.mean(endpoint_supported)),
            "active_endpoint_supported_count": int(
                np.sum(endpoint_supported & active_mask)
            ),
            "sentinel_endpoint_supported_count": int(
                np.sum(endpoint_supported & sentinel_mask)
            ),
            "minimum_camera_support": (provider.config.minimum_camera_support),
        },
        "sentinel_debias": debias.report(),
        "assimilation_report": assimilation_report,
        "outputs": {
            "provider_arrays_file_sha256": file_sha256(provider_path),
            "provider_array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(provider_arrays.items())
            },
            "sentinel_measurements_file_sha256": file_sha256(measurement_path),
            "sentinel_measurement_array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(measurement_arrays.items())
            },
            "assimilation_arrays_file_sha256": file_sha256(assimilation_path),
            "assimilation_array_sha256": {
                name: array_sha256(values)
                for name, values in sorted(assimilation_arrays.items())
            },
            "query_schedule_file_sha256": file_sha256(schedule_path),
            "prefix_support_screen_file_sha256": file_sha256(prefix_support_path),
        },
        "method_contract": {
            "learned_material_identity_carrier": False,
            "direct_depth_mask_endpoint_association": True,
            "association_probability_separate_from_reliability": True,
            "prior_reliability_uses_physical_innovation": False,
            "unknown_cross_view_correlation": "covariance-intersection",
            "between_view_scatter_added_to_covariance": True,
            "birth_anchored_displacement": True,
            "sentinel_common_bias_removed_before_state_update": True,
            "innovation_robustified_once": True,
            "rejection": "bit-exact selected physical-or-persistence backbone",
        },
        "information_boundary": {
            "provider_depth_mask_frames_consumed": [
                schedule.config.query_birth_frame,
                schedule.config.query_update_frame,
            ],
            "camera_admission_prefix_range_half_open": [0, 58],
            "maximum_rgb_depth_mask_frame_read": (schedule.config.query_update_frame),
            "maximum_physical_frame_read": (schedule.config.query_update_frame),
            "future_identity_read": False,
            "future_object_observation_read": False,
            "target_metric_read": False,
            "v1_sealed_target_cohort_read": False,
            "held_v8_artifact_read": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(report)
    report_path = output / "source_development_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
