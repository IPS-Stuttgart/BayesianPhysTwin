#!/usr/bin/env python3
"""Run the sealed source-only PokeFlex prior-aware belief smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repository_root() / "src"))

from run_pokeflex_bayesian_registration_smoke import (  # noqa: E402
    _cd_ul1_mm,
    _load_mesh,
    _surface_sample,
    _template_frame,
    _view_points,
)
from run_pokeflex_checkpoint_registration_independent_depth import (  # noqa: E402
    _correction_field_variants,
    _load_official_template,
    _realsense_parameters,
    _realsense_world_points,
)

from bayesian_phystwin.observation_belief import (  # noqa: E402
    array_sha256,
    save_observation_belief,
)
from bayesian_phystwin.physical_linearization import (  # noqa: E402
    save_physical_linearization,
)
from bayesian_phystwin.pokeflex_bayesian_registration import (  # noqa: E402
    PokeFlexBayesianRegistrationConfig,
    register_pokeflex_graph_posterior,
)
from bayesian_phystwin.pokeflex_independent_depth import (  # noqa: E402
    PokeFlexDepthCalibration,
    PokeFlexIndependentDepthAnchor,
    build_independent_depth_anchor,
    calibrate_depth_translation,
    crop_points_to_geometry,
    select_points_near_geometry,
)
from bayesian_phystwin.pokeflex_prior_aware_belief import (  # noqa: E402
    PokeFlexPriorAwareConfigV1,
    build_pokeflex_prior_aware_frame_artifacts,
    infer_pokeflex_prior_aware_frame,
)
from bayesian_phystwin.pokeflex_registration_protocol import (  # noqa: E402
    load_pokeflex_registration_protocol,
)
from bayesian_phystwin.pokeflex_released_checkpoint import (  # noqa: E402
    PokeFlexReleasedCheckpoint,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_bundle_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(np.asarray(value)).encode("ascii"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rendered = (
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _load_smoke_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("artifact_kind") != "PokeFlexPriorAwareBeliefSourceSmokeProtocol":
        raise ValueError("unexpected prior-aware smoke protocol kind")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported prior-aware smoke protocol version")
    return protocol


def _select_source_frame(
    robot_by_frame: dict[int, dict[str, object]],
    protocol: dict[str, object],
) -> int:
    selection = protocol["source_frame_selection"]
    minimum = int(selection["minimum_source_frame"])
    threshold = float(selection["force_y_threshold_n"])
    offset = int(selection["target_offset_frames"])
    for frame, record in sorted(robot_by_frame.items()):
        if (
            frame >= minimum
            and frame + offset in robot_by_frame
            and float(record["forces"][1]) > threshold
        ):
            return frame
    raise ValueError("no frame satisfies the locked source-frame rule")


def _build_source_anchor(
    take_root: Path,
    *,
    source_frame: int,
    template_frame: int,
    template_vertices_m: np.ndarray,
    protocol: dict[str, object],
) -> tuple[
    PokeFlexIndependentDepthAnchor,
    tuple[PokeFlexDepthCalibration, ...],
    dict[str, object],
]:
    depth_lock = protocol["independent_depth"]
    parameters_and_hashes = tuple(
        _realsense_parameters(take_root, camera) for camera in (0, 1)
    )
    calibrations = []
    sensor_points = []
    source_files: dict[str, str] = {}
    maximum_distance_m = float(depth_lock["static_template_support_radius_mm"]) / 1000.0
    for camera, (parameters, parameters_sha256) in enumerate(parameters_and_hashes):
        template_points = _realsense_world_points(
            take_root,
            template_frame,
            camera,
            parameters,
        )
        calibration = calibrate_depth_translation(
            template_points,
            template_vertices_m,
        )
        calibrations.append(calibration)
        points = _realsense_world_points(
            take_root,
            source_frame,
            camera,
            parameters,
        )
        points = points + calibration.translation_m
        points = crop_points_to_geometry(
            points,
            template_vertices_m,
            padding_m=0.05,
        )
        points = select_points_near_geometry(
            points,
            template_vertices_m,
            maximum_distance_m=maximum_distance_m,
        )
        sensor_points.append(points)
        source_files[f"realsense-{camera}-parameters"] = parameters_sha256
        source_files[f"realsense-{camera}-source-depth"] = _sha256(
            take_root / "realsense" / str(camera) / "depth" / f"{source_frame:05d}.png"
        )
        source_files[f"realsense-{camera}-template-depth"] = _sha256(
            take_root
            / "realsense"
            / str(camera)
            / "depth"
            / f"{template_frame:05d}.png"
        )
    anchor = build_independent_depth_anchor(
        take_id=take_root.name,
        frame_id=source_frame,
        causal_cutoff_frame=source_frame,
        sensor_points_m=tuple(sensor_points),
        sensor_names=("realsense0", "realsense1"),
        calibration_sha256=tuple(
            source_files[f"realsense-{camera}-parameters"] for camera in (0, 1)
        ),
        sensor_variance_m2=tuple(
            float(depth_lock["sensor_variance_mm2"]) * 1e-6 for _ in (0, 1)
        ),
        voxel_size_m=float(depth_lock["voxel_size_mm"]) / 1000.0,
        maximum_clusters_per_sensor=int(depth_lock["maximum_clusters_per_sensor"]),
        metadata={
            "template_frame": template_frame,
            "calibration_median_residual_m": [
                item.median_residual_m for item in calibrations
            ],
            "calibration_p90_residual_m": [
                item.p90_residual_m for item in calibrations
            ],
            "maximum_template_distance_m": maximum_distance_m,
        },
    )
    source_manifest = {
        "source_files": source_files,
        "anchor_arrays_sha256": _array_bundle_sha256(
            {
                "points_m": anchor.points_m,
                "variance_m2": anchor.variance_m2,
                "sensor_index": anchor.sensor_index,
            }
        ),
        "anchor_metadata": anchor.metadata_dict(),
    }
    return anchor, tuple(calibrations), source_manifest


def _checkpoint_prediction_pair(
    take_root: Path,
    template_vertices_m: np.ndarray,
    *,
    source_frame: int,
    upstream_checkout: Path,
    checkpoint_root: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], dict[str, object]]:
    checkpoint = PokeFlexReleasedCheckpoint.load(
        template_vertices_m,
        upstream_checkout=upstream_checkout,
        checkpoint_root=checkpoint_root,
        device=device,
    )
    encoded: dict[int, object] = {}
    preprocessing: dict[int, object] = {}
    views: dict[int, tuple[np.ndarray, ...]] = {}
    for frame in range(source_frame - checkpoint.history_frame_count, source_frame + 1):
        current_views = tuple(
            _view_points(take_root, frame, camera, template_vertices_m)
            for camera in (0, 1)
        )
        feature, record = checkpoint.encode_frame(current_views)
        encoded[frame] = feature
        preprocessing[frame] = record
        views[frame] = current_views
    source_history = range(source_frame - checkpoint.history_frame_count, source_frame)
    target_history = range(
        source_frame - checkpoint.history_frame_count + 1,
        source_frame + 1,
    )
    source = checkpoint.predict_from_encoded_history(
        [encoded[frame] for frame in source_history],
        [preprocessing[frame] for frame in source_history],
    )
    target = checkpoint.predict_from_encoded_history(
        [encoded[frame] for frame in target_history],
        [preprocessing[frame] for frame in target_history],
    )
    checkpoint_hashes = {
        name: _sha256(checkpoint_root / name)
        for name in (
            "pointcloud_encoder.pth",
            "attention_model.pth",
            "decoder.pth",
        )
    }
    return (
        source.vertices_m,
        target.vertices_m,
        views[source_frame],
        {
            "checkpoint_sha256": checkpoint_hashes,
            "source_history": list(source_history),
            "target_history": list(target_history),
            "source_history_retained_point_counts": list(
                source.history_retained_point_counts
            ),
            "target_history_retained_point_counts": list(
                target.history_retained_point_counts
            ),
        },
    )


def _force_action_fields(
    source_prior_m: np.ndarray,
    target_prior_m: np.ndarray,
    correction_m: np.ndarray,
    field_names: tuple[str, ...],
    robot_by_frame: dict[int, dict[str, object]],
    source_frame: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    history = range(max(1, source_frame - 3), source_frame + 1)
    tool_positions = np.asarray(
        [robot_by_frame[frame]["T_WT"] for frame in history],
        dtype=np.float64,
    )[:, :3, 3]
    end_effector_positions = np.asarray(
        [robot_by_frame[frame]["T_WE"] for frame in history],
        dtype=np.float64,
    )[:, :3, 3]
    force_vectors = np.asarray(
        [robot_by_frame[frame]["forces"][:3] for frame in history],
        dtype=np.float64,
    )
    common = {
        "names": field_names,
        "previous_correction": None,
        "tool_positions": tool_positions,
        "end_effector_positions": end_effector_positions,
        "force_vectors": force_vectors,
    }
    source_fields = _correction_field_variants(
        source_prior_m,
        source_prior_m,
        correction_m,
        **common,
    )
    target_fields = _correction_field_variants(
        source_prior_m,
        target_prior_m,
        correction_m,
        **common,
    )
    return source_fields, target_fields


def _write_prediction_archive(
    path: Path,
    *,
    baseline_vertices_m: np.ndarray,
    candidate_vertices_m: np.ndarray,
    selected_vertices_m: np.ndarray,
    candidate_covariance_m2: np.ndarray,
    query_update_m: np.ndarray,
    inference: Any,
) -> None:
    np.savez_compressed(
        path,
        baseline_vertices_m=baseline_vertices_m,
        candidate_vertices_m=candidate_vertices_m,
        selected_vertices_m=selected_vertices_m,
        candidate_covariance_m2=candidate_covariance_m2,
        query_update_m=query_update_m,
        state_coefficients=inference.result.state_coefficients,
        shared_bias_coefficients=inference.result.shared_bias_coefficients,
        view_bias_coefficients=inference.result.view_bias_coefficients,
        posterior_covariance=inference.result.posterior_covariance,
        robust_weights=inference.result.robust_weights,
    )


def run_smoke(
    *,
    take_root: Path,
    output_root: Path,
    protocol_path: Path,
    parent_protocol_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
    source_revision: str,
    device: str,
) -> dict[str, object]:
    protocol = _load_smoke_protocol(protocol_path)
    parent = load_pokeflex_registration_protocol(parent_protocol_path)
    if parent["protocol_sha256"] != protocol["parent_protocol"]["protocol_sha256"]:
        raise ValueError("parent PokeFlex protocol checksum changed")
    if take_root.name != protocol["source_take"]:
        raise ValueError("take differs from the locked source smoke")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("smoke output root is not empty; use a new run root")
    output_root.mkdir(parents=True, exist_ok=True)

    robot_path = take_root / "robot_data.json"
    robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(record["frame"]): record for record in robot_records}
    source_frame = _select_source_frame(robot_by_frame, protocol)
    target_frame = source_frame + int(
        protocol["source_frame_selection"]["target_offset_frames"]
    )
    active = [
        frame
        for frame, record in sorted(robot_by_frame.items())
        if float(record["forces"][1])
        > float(protocol["source_frame_selection"]["force_y_threshold_n"])
    ]
    template_frame = _template_frame(active)
    template_path = take_root / "meshes" / f"mesh-f{template_frame:05d}.obj"
    template_vertices, template_faces, template_preprocessing = _load_official_template(
        template_path
    )

    source_prior, target_prior, source_views, checkpoint_record = (
        _checkpoint_prediction_pair(
            take_root,
            template_vertices,
            source_frame=source_frame,
            upstream_checkout=upstream_checkout,
            checkpoint_root=checkpoint_root,
            device=device,
        )
    )
    action_supported = float(robot_by_frame[source_frame]["forces"][1]) > float(
        protocol["source_frame_selection"]["force_y_threshold_n"]
    )
    registration = register_pokeflex_graph_posterior(
        source_prior,
        source_views,
        action_supported=action_supported,
        prior_faces=template_faces,
        config=PokeFlexBayesianRegistrationConfig(residual_geometry="point_to_point"),
    )
    if not registration.accepted:
        raise ValueError(
            f"locked smoke source registration is inadmissible: {registration.reason}"
        )
    correction = registration.posterior_vertices_m - source_prior
    field_names = tuple(protocol["physical_state_span"]["correction_fields"])
    source_fields, target_fields = _force_action_fields(
        source_prior,
        target_prior,
        correction,
        field_names,
        robot_by_frame,
        source_frame,
    )
    anchor, calibrations, anchor_source_manifest = _build_source_anchor(
        take_root,
        source_frame=source_frame,
        template_frame=template_frame,
        template_vertices_m=template_vertices,
        protocol=protocol,
    )
    source_artifact_sha256 = _canonical_sha256(anchor_source_manifest)
    baseline_belief_id = _array_bundle_sha256(
        {
            "source_prior_m": source_prior,
            "target_prior_m": target_prior,
        }
    )
    action_prefix_id = _canonical_sha256(
        [record for record in robot_records if int(record["frame"]) <= source_frame]
    )
    config = PokeFlexPriorAwareConfigV1(**protocol["belief_config"])
    artifacts = build_pokeflex_prior_aware_frame_artifacts(
        anchor=anchor,
        baseline_source_vertices_m=source_prior,
        baseline_target_vertices_m=target_prior,
        source_correction_fields_m=source_fields,
        target_correction_fields_m=target_fields,
        baseline_belief_id=baseline_belief_id,
        action_prefix_id=action_prefix_id,
        simulator_revision=parent["payload"]["upstream"]["code_commit"],
        source_revision=source_revision,
        source_artifact_sha256=source_artifact_sha256,
        config=config,
    )
    save_observation_belief(
        output_root / "observation_belief.npz",
        artifacts.observation_belief,
    )
    save_physical_linearization(
        output_root / "physical_linearization.npz",
        artifacts.linearization,
    )
    np.savez_compressed(
        output_root / "independent_depth_anchor.npz",
        points_m=anchor.points_m,
        variance_m2=anchor.variance_m2,
        sensor_index=anchor.sensor_index,
        metadata_json=np.asarray(json.dumps(anchor.metadata_dict(), sort_keys=True)),
    )
    inference = infer_pokeflex_prior_aware_frame(
        artifacts,
        target_prior,
        config=config,
    )
    selected = inference.select_or_exact_fallback(target_prior)
    prediction_path = output_root / "prediction.npz"
    _write_prediction_archive(
        prediction_path,
        baseline_vertices_m=target_prior,
        candidate_vertices_m=inference.candidate_vertices_m,
        selected_vertices_m=selected,
        candidate_covariance_m2=inference.candidate_covariance_m2,
        query_update_m=inference.query_update_m,
        inference=inference,
    )
    seal = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexPriorAwareBeliefPredictionSeal",
        "protocol_path": str(protocol_path.resolve()),
        "protocol_file_sha256": _sha256(protocol_path),
        "source_revision": source_revision,
        "take_id": take_root.name,
        "template_frame": template_frame,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "causal_frame_stop": target_frame,
        "future_observation_used": False,
        "target_mesh_opened": False,
        "baseline_belief_id": baseline_belief_id,
        "action_prefix_id": action_prefix_id,
        "source_artifact_sha256": source_artifact_sha256,
        "observation_artifact_id": artifacts.observation_belief.artifact_id,
        "linearization_artifact_id": artifacts.linearization.artifact_id,
        "prediction_npz_sha256": _sha256(prediction_path),
        "inference_admissible": inference.result.inference_admissible,
        "inference_reason": inference.result.reason,
        "selected_candidate": inference.result.inference_admissible,
        "rejected_prediction_is_exact_baseline_object": (
            inference.result.inference_admissible or selected is target_prior
        ),
    }
    seal_path = output_root / "prediction_seal.json"
    _atomic_json(seal_path, seal)

    # The target geometry is first touched after the immutable prediction seal.
    target_mesh = _load_mesh(take_root / "meshes" / f"mesh-f{target_frame:05d}.obj")
    sample_count = int(protocol["evaluation"]["surface_points"])
    seed = int(protocol["evaluation"]["seed"]) + target_frame
    target_sample = _surface_sample(
        np.asarray(target_mesh.vertices, dtype=np.float64) / 1000.0,
        np.asarray(target_mesh.faces, dtype=np.int64),
        sample_count,
        seed,
    )
    baseline_sample = _surface_sample(
        target_prior,
        template_faces,
        sample_count,
        seed,
    )
    candidate_sample = _surface_sample(
        selected,
        template_faces,
        sample_count,
        seed,
    )
    baseline_error = _cd_ul1_mm(baseline_sample, target_sample)
    candidate_error = _cd_ul1_mm(candidate_sample, target_sample)
    result = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexPriorAwareBeliefSourceSmokeResult",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_file_sha256": _sha256(protocol_path),
        "prediction_seal_sha256": _sha256(seal_path),
        "prediction_npz_sha256": _sha256(prediction_path),
        "take_id": take_root.name,
        "template_frame": template_frame,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "future_observation_used": False,
        "prediction_sealed_before_target_mesh": True,
        "source_revision": source_revision,
        "template_sha256": _sha256(template_path),
        "template_preprocessing": template_preprocessing,
        "robot_prefix_sha256": action_prefix_id,
        "checkpoint": checkpoint_record,
        "source_registration": {
            "accepted": registration.accepted,
            "reason": registration.reason,
            "diagnostics": registration.diagnostics,
        },
        "independent_depth": {
            "anchor": anchor.metadata_dict(),
            "source_manifest": anchor_source_manifest,
            "calibration_translation_m": [
                item.translation_m.tolist() for item in calibrations
            ],
            "calibration_median_residual_mm": [
                item.median_residual_m * 1000.0 for item in calibrations
            ],
            "calibration_p90_residual_mm": [
                item.p90_residual_m * 1000.0 for item in calibrations
            ],
        },
        "belief": {
            "config": protocol["belief_config"],
            "state_mode_names": list(artifacts.state_mode_names),
            "field_names": list(field_names),
            "observation_count": artifacts.observation_belief.observation_count,
            "group_count": len(artifacts.observation_belief.group_ids),
            "prior_reliability_by_group": {
                str(group): float(
                    np.mean(
                        artifacts.observation_belief.prior_reliability[
                            artifacts.observation_belief.correlation_group_ids == group
                        ]
                    )
                )
                for group in artifacts.observation_belief.group_ids
            },
            "inference_admissible": inference.result.inference_admissible,
            "reason": inference.result.reason,
            "state_coefficients_m": inference.result.state_coefficients,
            "shared_bias_coefficients_m": inference.result.shared_bias_coefficients,
            "view_bias_coefficients_m": inference.result.view_bias_coefficients,
            "maximum_query_update_mm": float(
                1000.0 * np.max(np.linalg.norm(inference.query_update_m, axis=1))
            ),
            "mean_query_posterior_std_mm": float(
                1000.0
                * np.mean(
                    np.sqrt(
                        np.maximum(
                            np.trace(
                                inference.candidate_covariance_m2, axis1=1, axis2=2
                            )
                            / 3.0,
                            0.0,
                        )
                    )
                )
            ),
            "robust_weight_mean": float(np.mean(inference.result.robust_weights)),
            "robust_weight_minimum": float(np.min(inference.result.robust_weights)),
            "diagnostics": inference.result.diagnostics,
        },
        "score": {
            "metric": protocol["evaluation"]["metric"],
            "released_checkpoint_mm": baseline_error,
            "prior_aware_selected_mm": candidate_error,
            "absolute_change_mm": candidate_error - baseline_error,
            "relative_change_percent": 100.0
            * (candidate_error - baseline_error)
            / baseline_error,
        },
    }
    _atomic_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("take_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_prior_aware_belief_source_smoke_v1.json"
        ),
    )
    parser.add_argument(
        "--parent-protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_bayesian_registration_v1.json"
        ),
    )
    args = parser.parse_args()
    result = run_smoke(
        take_root=args.take_root.resolve(),
        output_root=args.output_root.resolve(),
        protocol_path=args.protocol.resolve(),
        parent_protocol_path=args.parent_protocol.resolve(),
        upstream_checkout=args.upstream_checkout.resolve(),
        checkpoint_root=args.checkpoint_root.resolve(),
        source_revision=args.source_revision,
        device=args.device,
    )
    print(json.dumps(_json_safe(result["score"]), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
