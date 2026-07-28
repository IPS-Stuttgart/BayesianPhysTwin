#!/usr/bin/env python3
"""Run sequential direct-depth action-response admission on one source case."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_direct_depth_action_response import (
    DirectDepthActionResponseConfig,
    evaluate_direct_depth_action_response,
)
from bayesian_phystwin.deform360_direct_depth_provider import (
    DirectDepthEndpointConfig,
    build_direct_depth_birth_anchored_measurements,
    build_direct_depth_endpoint_observations,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_admission_v2 import (
    load_complete_camera_geometry,
    load_selected_complete_causal_inputs,
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
    DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
    DYNAMIC_DEPTH_ENDPOINT_PAIRS,
    Deform360SentinelQueryConfig,
    build_deform360_sentinel_query_schedule,
)
from bayesian_phystwin.observation_belief import array_sha256, file_sha256
from bayesian_phystwin.phystwin_sentinel_queries import (
    ACTIVE_QUERY_ROLE,
    SENTINEL_QUERY_ROLE,
)

CONFIG_RELATIVE_PATH = Path(
    "configs/sota/deform360_dynamic_direct_depth_admission_source_v9.json"
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
    _require(not _git_output(repo, "status", "--porcelain"), "repository is dirty")
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


def _controller_centroid(controller_points_m: np.ndarray) -> np.ndarray:
    controller = np.asarray(controller_points_m, dtype=np.float64)
    _require(
        controller.ndim == 3 and controller.shape[2] == 3,
        "controller points must have shape (T, A, 3)",
    )
    finite = np.all(np.isfinite(controller), axis=2)
    centroid = np.full((len(controller), 3), np.nan, dtype=np.float64)
    for frame in range(len(controller)):
        _require(np.any(finite[frame]), "controller frame is empty")
        centroid[frame] = np.median(controller[frame, finite[frame]], axis=0)
    return centroid


def _actuator_displacement(
    controller_centroid_m: np.ndarray,
    birth_frame: int,
    update_frame: int,
) -> float:
    relative = (
        controller_centroid_m[birth_frame : update_frame + 1]
        - controller_centroid_m[birth_frame]
    )
    return float(np.max(np.linalg.norm(relative, axis=1)))


def _load_protocol(repo: Path, case: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = repo / CONFIG_RELATIVE_PATH
    protocol = json.loads(path.read_text(encoding="utf-8"))
    _require(
        protocol["protocol_id"] == DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
        "protocol ID changed",
    )
    _require(
        tuple(tuple(map(int, pair)) for pair in protocol["endpoint_pairs"])
        == DYNAMIC_DEPTH_ENDPOINT_PAIRS,
        "registered endpoint pairs changed",
    )
    records = {str(record["case"]): record for record in protocol["cases"]}
    _require(case in records, "case is outside the locked source panel")
    return protocol, records[case]


def _branch_report(
    *,
    case_id: str,
    branch_dir: Path,
    physical: dict[str, np.ndarray],
    processed: Path,
    protocol: dict[str, Any],
    birth_frame: int,
    update_frame: int,
    actuator_displacement_m: float,
) -> dict[str, Any]:
    geometry = load_complete_camera_geometry(
        processed,
        frame_count=update_frame + 1,
    )
    schedule_config = Deform360SentinelQueryConfig(
        **protocol["query_schedule"],
        query_birth_frame=birth_frame,
        query_update_frame=update_frame,
        protocol_id=DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
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
        frame_count=update_frame + 1,
    )
    prefix_support = build_prefix_association_support_screen(
        physical["physical_prediction_m"],
        camera_inputs.intrinsics,
        camera_inputs.camera_to_world,
        camera_inputs.depths_m,
        camera_inputs.object_masks,
        config=PrefixAssociationSupportConfig(
            birth_frame=birth_frame,
            update_frame=update_frame,
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
            **{
                key: value
                for key, value in protocol["direct_depth"].items()
                if key != "unknown_cross_view_correlation"
            }
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
    admission = evaluate_direct_depth_action_response(
        case_id,
        physical["physical_prediction_m"],
        debias.measurements,
        schedule,
        sentinel_applied=debias.applied,
        actuator_displacement_m=actuator_displacement_m,
        config=DirectDepthActionResponseConfig(**protocol["admission"]),
    )

    branch_dir.mkdir(parents=True)
    provider_arrays = {
        "endpoint_frames": provider.endpoint_frames,
        "entity_ids": provider.entity_ids,
        "point_world_m": provider.point_world_m,
        "covariance_m2": provider.covariance_m2,
        "accepted_support": provider.accepted_support,
        "association_probability": provider.association_probability,
        "support_count": provider.support_count,
        "maximum_view_scatter_m": provider.maximum_view_scatter_m,
    }
    measurement_arrays = {
        "measurement_m": debias.measurements.measurement_m,
        "covariance_m2": debias.measurements.covariance_m2,
        "prior_reliability": debias.measurements.prior_reliability,
        "association_probability": debias.measurements.association_probability,
        "available": debias.measurements.available,
        "entity_ids": debias.measurements.entity_ids,
    }
    provider_path = branch_dir / "provider_arrays.npz"
    measurement_path = branch_dir / "sentinel_measurements.npz"
    _write_npz(provider_path, provider_arrays)
    _write_npz(measurement_path, measurement_arrays)
    schedule_path = branch_dir / "query_schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                **schedule.descriptor(),
                "schedule_sha256": schedule.artifact_sha256,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    support_path = branch_dir / "prefix_support_screen.json"
    support_path.write_text(
        json.dumps(
            {
                **prefix_support.descriptor(),
                "artifact_sha256": prefix_support.artifact_sha256,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    admission_path = branch_dir / "action_response_admission.json"
    admission_path.write_text(
        json.dumps(
            admission.descriptor(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    endpoint_supported = np.all(provider.accepted_support, axis=0)
    active = schedule.query_roles == ACTIVE_QUERY_ROLE
    sentinel = schedule.query_roles == SENTINEL_QUERY_ROLE
    return {
        "birth_frame": birth_frame,
        "update_frame": update_frame,
        "status": "admitted" if admission.admitted else "rejected",
        "admitted": admission.admitted,
        "reason": admission.reason,
        "actuator_displacement_m": actuator_displacement_m,
        "support": {
            "complete_camera_count": len(geometry.camera_names),
            "selected_camera_count": len(camera_inputs.camera_names),
            "prefix_screen_eligible_count": int(np.sum(prefix_support.eligible)),
            "scheduled_identity_count": len(schedule.entity_ids),
            "active_endpoint_supported_count": int(
                np.sum(endpoint_supported & active)
            ),
            "sentinel_endpoint_supported_count": int(
                np.sum(endpoint_supported & sentinel)
            ),
        },
        "sentinel_debias": debias.report(),
        "action_response": admission.descriptor(),
        "inputs_sha256": {
            "camera_certificate": geometry.artifact_sha256,
            "query_schedule": schedule.artifact_sha256,
            "prefix_support_screen": prefix_support.artifact_sha256,
        },
        "outputs_sha256": {
            "provider_arrays_file": file_sha256(provider_path),
            "provider_arrays": {
                name: array_sha256(values)
                for name, values in sorted(provider_arrays.items())
            },
            "sentinel_measurements_file": file_sha256(measurement_path),
            "sentinel_measurements": {
                name: array_sha256(values)
                for name, values in sorted(measurement_arrays.items())
            },
            "query_schedule_file": file_sha256(schedule_path),
            "prefix_support_screen_file": file_sha256(support_path),
            "action_response_admission_file": file_sha256(admission_path),
        },
    }


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    revision = _require_clean_repository(repo)
    protocol, case_record = _load_protocol(repo, args.case)
    config_path = repo / CONFIG_RELATIVE_PATH
    output = args.output_dir.resolve()
    _require(not output.exists(), "source-admission output already exists")
    output.mkdir(parents=True)

    physical_root = args.physical_dir.resolve()
    physical_dir = physical_root
    if not (physical_dir / PHYSICAL_MANIFEST_FILENAME).is_file():
        physical_dir = physical_dir / "sealed_physical"
    manifest, physical = validate_dynamic_physical_artifacts(physical_dir)
    _require(manifest["case"] == args.case, "physical case changed")
    _require(
        file_sha256(physical_dir / PHYSICAL_ARCHIVE_FILENAME)
        == case_record["physical_archive_sha256"],
        "physical archive differs from the locked source panel",
    )
    prediction_only_path = physical_root / "prediction_only_input.pkl"
    with prediction_only_path.open("rb") as stream:
        prediction_only = pickle.load(stream)
    controller = _controller_centroid(prediction_only["controller_points"])
    _require(
        len(controller) == len(physical["physical_prediction_m"]),
        "controller and physical frame counts differ",
    )

    branches: list[dict[str, Any]] = []
    selected_pair: list[int] | None = None
    maximum_observation_frame = -1
    for birth_frame, update_frame in DYNAMIC_DEPTH_ENDPOINT_PAIRS:
        maximum_observation_frame = update_frame
        try:
            branch = _branch_report(
                case_id=args.case,
                branch_dir=output / f"branch_{birth_frame:02d}_{update_frame:02d}",
                physical=physical,
                processed=args.processed_episode_dir.resolve(),
                protocol=protocol,
                birth_frame=birth_frame,
                update_frame=update_frame,
                actuator_displacement_m=_actuator_displacement(
                    controller,
                    birth_frame,
                    update_frame,
                ),
            )
        except (OSError, RuntimeError, ValueError) as error:
            branch = {
                "birth_frame": birth_frame,
                "update_frame": update_frame,
                "status": "target-free-prefix-rejection",
                "admitted": False,
                "reason": f"{type(error).__name__}:{error}",
            }
        branches.append(branch)
        if branch["admitted"]:
            selected_pair = [birth_frame, update_frame]
            break

    report: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicDirectDepthAdmissionV9Source",
        "protocol_id": DYNAMIC_DEPTH_ADMISSION_PROTOCOL_ID,
        "status": "source_admitted" if selected_pair is not None else "source_rejected",
        "case": args.case,
        "case_hash": manifest["case_hash"],
        "repository_revision": revision,
        "protocol_config_sha256": file_sha256(config_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "selected_endpoint_pair": selected_pair,
        "branches": branches,
        "inputs_sha256": {
            "physical_manifest": file_sha256(
                physical_dir / PHYSICAL_MANIFEST_FILENAME
            ),
            "physical_archive": file_sha256(
                physical_dir / PHYSICAL_ARCHIVE_FILENAME
            ),
            "prediction_only_input": file_sha256(prediction_only_path),
        },
        "method_contract": {
            "sequential_stopping_rule": True,
            "direct_depth_mask_endpoint_association": True,
            "sentinel_common_bias_removed_before_admission": True,
            "action_aligned_spatial_group_gate": True,
            "unknown_cross_view_correlation": "covariance_intersection",
            "candidate_state_update_constructed": False,
            "rejection": "no candidate; unchanged physical baseline required",
        },
        "information_boundary": {
            "maximum_rgb_depth_mask_frame_read": maximum_observation_frame,
            "future_identity_read": False,
            "future_object_observation_read": False,
            "future_metric_read": False,
            "candidate_state_update_read": False,
            "v1_sealed_target_read": False,
            "held_v8_artifact_read": False,
        },
    }
    report["result_sha256"] = _canonical_sha256(report)
    report_path = output / "source_admission_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
