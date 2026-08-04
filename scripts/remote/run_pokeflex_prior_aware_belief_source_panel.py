#!/usr/bin/env python3
"""Run the fixed PokeFlex prior-aware belief on the opened source panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


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
    _load_official_template,
    _realsense_parameters,
    _realsense_world_points,
)
from run_pokeflex_prior_aware_belief_smoke import (  # noqa: E402
    _array_bundle_sha256,
    _atomic_json,
    _canonical_sha256,
    _force_action_fields,
    _load_smoke_protocol,
    _sha256,
    _write_prediction_archive,
)

from bayesian_phystwin.observation_belief import (  # noqa: E402
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

CHI_SQUARE_3D_90 = 6.251388631170325
OUTCOME_READOUT_STD_M = 0.004


def _load_panel_protocol(path: Path) -> dict[str, object]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("artifact_kind") != "PokeFlexPriorAwareBeliefSourcePanelProtocol":
        raise ValueError("unexpected prior-aware source-panel protocol kind")
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported prior-aware source-panel protocol version")
    return protocol


def _calibrate_d405(
    take_root: Path,
    *,
    template_frame: int,
    template_vertices_m: np.ndarray,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[str, ...],
    tuple[PokeFlexDepthCalibration, ...],
]:
    parameters_and_hashes = tuple(
        _realsense_parameters(take_root, camera) for camera in (0, 1)
    )
    calibrations = []
    for camera, (parameters, _) in enumerate(parameters_and_hashes):
        template_points = _realsense_world_points(
            take_root,
            template_frame,
            camera,
            parameters,
        )
        calibrations.append(
            calibrate_depth_translation(template_points, template_vertices_m)
        )
    return (
        tuple(item[0] for item in parameters_and_hashes),
        tuple(item[1] for item in parameters_and_hashes),
        tuple(calibrations),
    )


def _source_anchor(
    take_root: Path,
    *,
    source_frame: int,
    template_frame: int,
    template_vertices_m: np.ndarray,
    parameters: tuple[dict[str, object], ...],
    parameter_hashes: tuple[str, ...],
    calibrations: tuple[PokeFlexDepthCalibration, ...],
    smoke_protocol: dict[str, object],
) -> tuple[PokeFlexIndependentDepthAnchor, str]:
    depth_lock = smoke_protocol["independent_depth"]
    maximum_distance_m = float(depth_lock["static_template_support_radius_mm"]) / 1000.0
    sensor_points = []
    depth_hashes = []
    for camera in (0, 1):
        points = _realsense_world_points(
            take_root,
            source_frame,
            camera,
            parameters[camera],
        )
        points = points + calibrations[camera].translation_m
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
        depth_hashes.append(
            _sha256(
                take_root
                / "realsense"
                / str(camera)
                / "depth"
                / f"{source_frame:05d}.png"
            )
        )
    anchor = build_independent_depth_anchor(
        take_id=take_root.name,
        frame_id=source_frame,
        causal_cutoff_frame=source_frame,
        sensor_points_m=tuple(sensor_points),
        sensor_names=("realsense0", "realsense1"),
        calibration_sha256=parameter_hashes,
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
    digest = _canonical_sha256(
        {
            "take_id": take_root.name,
            "source_frame": source_frame,
            "parameter_sha256": list(parameter_hashes),
            "depth_sha256": depth_hashes,
            "anchor_array_sha256": _array_bundle_sha256(
                {
                    "points_m": anchor.points_m,
                    "variance_m2": anchor.variance_m2,
                    "sensor_index": anchor.sensor_index,
                }
            ),
        }
    )
    return anchor, digest


def _surface_posterior_diagnostic(
    selected_vertices_m: np.ndarray,
    covariance_m2: np.ndarray,
    target_surface_m: np.ndarray,
) -> tuple[float, float]:
    nearest = cKDTree(target_surface_m).query(selected_vertices_m, k=1)[1]
    residual = target_surface_m[nearest] - selected_vertices_m
    covariance = np.asarray(covariance_m2, dtype=np.float64).copy()
    covariance += OUTCOME_READOUT_STD_M**2 * np.eye(3)[None]
    nees = np.einsum(
        "ni,nij,nj->n",
        residual,
        np.linalg.inv(covariance),
        residual,
    )
    return float(np.mean(nees <= CHI_SQUARE_3D_90)), float(np.mean(nees))


def _write_fallback_prediction(
    path: Path,
    baseline_vertices_m: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        baseline_vertices_m=baseline_vertices_m,
        candidate_vertices_m=baseline_vertices_m,
        selected_vertices_m=baseline_vertices_m,
        candidate_covariance_m2=np.zeros(
            (len(baseline_vertices_m), 3, 3),
            dtype=np.float64,
        ),
        query_update_m=np.zeros_like(baseline_vertices_m),
        state_coefficients=np.empty(0, dtype=np.float64),
        shared_bias_coefficients=np.empty(0, dtype=np.float64),
        view_bias_coefficients=np.empty(0, dtype=np.float64),
        posterior_covariance=np.empty((0, 0), dtype=np.float64),
        robust_weights=np.empty(0, dtype=np.float64),
    )


def _take_suffix(take_id: str) -> str:
    _, separator, suffix = take_id.rpartition("_")
    if not separator or not suffix.startswith("T") or not suffix[1:].isdigit():
        raise ValueError(f"invalid PokeFlex take id: {take_id}")
    return suffix


def run_take(
    *,
    take_root: Path,
    output_root: Path,
    panel_protocol_path: Path,
    smoke_protocol_path: Path,
    parent_protocol_path: Path,
    upstream_checkout: Path,
    checkpoint_root: Path,
    source_revision: str,
    device: str,
) -> dict[str, object]:
    panel = _load_panel_protocol(panel_protocol_path)
    smoke = _load_smoke_protocol(smoke_protocol_path)
    parent = load_pokeflex_registration_protocol(parent_protocol_path)
    if (
        _sha256(smoke_protocol_path)
        != panel["predecessor"]["smoke_protocol_file_sha256"]
    ):
        raise ValueError("smoke protocol bytes changed")
    belief_module_path = (
        _repository_root()
        / "src"
        / "bayesian_phystwin"
        / "pokeflex_prior_aware_belief.py"
    )
    if _sha256(belief_module_path) != panel["method_lock"]["belief_module_sha256"]:
        raise ValueError("locked prior-aware belief implementation changed")
    expected = {
        f"{object_name}_{suffix}"
        for object_name in panel["cohort"]["development_objects"]
        for suffix in panel["cohort"]["take_suffixes"]
    }
    if take_root.name not in expected:
        raise ValueError("take lies outside the locked source panel")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("take output root is not empty")
    output_root.mkdir(parents=True, exist_ok=True)

    robot_path = take_root / "robot_data.json"
    robot_records = json.loads(robot_path.read_text(encoding="utf-8"))
    robot_by_frame = {int(record["frame"]): record for record in robot_records}
    force_threshold = float(smoke["source_frame_selection"]["force_y_threshold_n"])
    active_targets = {
        frame
        for frame, record in robot_by_frame.items()
        if float(record["forces"][1]) > force_threshold
    }
    template_frame = _template_frame(sorted(active_targets))
    template_path = take_root / "meshes" / f"mesh-f{template_frame:05d}.obj"
    template_vertices, template_faces, template_preprocessing = _load_official_template(
        template_path
    )
    maximum_frame = panel["cohort"]["maximum_frame_by_take_suffix"][
        _take_suffix(take_root.name)
    ]
    frame_limit = max(robot_by_frame) if maximum_frame is None else int(maximum_frame)
    checkpoint = PokeFlexReleasedCheckpoint.load(
        template_vertices,
        upstream_checkout=upstream_checkout,
        checkpoint_root=checkpoint_root,
        device=device,
    )
    parameters, parameter_hashes, calibrations = _calibrate_d405(
        take_root,
        template_frame=template_frame,
        template_vertices_m=template_vertices,
    )
    config = PokeFlexPriorAwareConfigV1(**smoke["belief_config"])
    field_names = tuple(panel["method_lock"]["correction_fields"])
    registration_config = PokeFlexBayesianRegistrationConfig(
        residual_geometry="point_to_point"
    )
    sample_count = int(panel["evaluation"]["surface_points"])
    base_seed = int(panel["evaluation"]["seed"])

    encoded: dict[int, object] = {}
    preprocessing: dict[int, object] = {}
    views_by_frame: dict[int, tuple[np.ndarray, ...]] = {}
    predictions: dict[int, np.ndarray] = {}
    records = []
    for target_frame in range(6, frame_limit + 1):
        history = range(target_frame - checkpoint.history_frame_count, target_frame)
        for observed_frame in history:
            if observed_frame in encoded:
                continue
            views = tuple(
                _view_points(take_root, observed_frame, camera, template_vertices)
                for camera in (0, 1)
            )
            feature, prepared = checkpoint.encode_frame(views)
            encoded[observed_frame] = feature
            preprocessing[observed_frame] = prepared
            views_by_frame[observed_frame] = views
        predictions[target_frame] = checkpoint.predict_from_encoded_history(
            [encoded[frame] for frame in history],
            [preprocessing[frame] for frame in history],
        ).vertices_m
        if target_frame not in active_targets or target_frame < 7:
            continue

        source_frame = target_frame - 1
        baseline = predictions[target_frame]
        frame_root = output_root / "frames" / f"f{target_frame:05d}"
        frame_root.mkdir(parents=True, exist_ok=False)
        selected = baseline
        covariance = np.zeros((len(baseline), 3, 3), dtype=np.float64)
        update_m = np.zeros_like(baseline)
        inference_admissible = False
        inference_reason = "exact-baseline-fallback"
        observation_artifact_id = None
        linearization_artifact_id = None
        observation_count = 0
        robust_weight_mean = None
        maximum_query_update_mm = 0.0
        registration_diagnostics: dict[str, object] = {}
        action_supported = (
            float(robot_by_frame[source_frame]["forces"][1]) > force_threshold
        )
        try:
            source_prior = predictions[source_frame]
            registration = register_pokeflex_graph_posterior(
                source_prior,
                views_by_frame[source_frame],
                action_supported=action_supported,
                prior_faces=template_faces,
                config=registration_config,
            )
            registration_diagnostics = dict(registration.diagnostics)
            if not registration.accepted:
                raise ValueError(f"kinect-registration:{registration.reason}")
            correction = registration.posterior_vertices_m - source_prior
            source_fields, target_fields = _force_action_fields(
                source_prior,
                baseline,
                correction,
                field_names,
                robot_by_frame,
                source_frame,
            )
            anchor, source_artifact_sha256 = _source_anchor(
                take_root,
                source_frame=source_frame,
                template_frame=template_frame,
                template_vertices_m=template_vertices,
                parameters=parameters,
                parameter_hashes=parameter_hashes,
                calibrations=calibrations,
                smoke_protocol=smoke,
            )
            baseline_belief_id = _array_bundle_sha256(
                {
                    "source_prior_m": source_prior,
                    "target_prior_m": baseline,
                }
            )
            action_prefix_id = _canonical_sha256(
                [
                    record
                    for record in robot_records
                    if int(record["frame"]) <= source_frame
                ]
            )
            artifacts = build_pokeflex_prior_aware_frame_artifacts(
                anchor=anchor,
                baseline_source_vertices_m=source_prior,
                baseline_target_vertices_m=baseline,
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
                frame_root / "observation_belief.npz",
                artifacts.observation_belief,
            )
            save_physical_linearization(
                frame_root / "physical_linearization.npz",
                artifacts.linearization,
            )
            inference = infer_pokeflex_prior_aware_frame(
                artifacts,
                baseline,
                config=config,
            )
            selected = inference.select_or_exact_fallback(baseline)
            covariance = inference.candidate_covariance_m2
            update_m = inference.query_update_m
            inference_admissible = inference.result.inference_admissible
            inference_reason = inference.result.reason
            observation_artifact_id = artifacts.observation_belief.artifact_id
            linearization_artifact_id = artifacts.linearization.artifact_id
            observation_count = artifacts.observation_belief.observation_count
            robust_weight_mean = float(np.mean(inference.result.robust_weights))
            maximum_query_update_mm = float(
                1000.0 * np.max(np.linalg.norm(update_m, axis=1))
            )
            prediction_path = frame_root / "prediction.npz"
            _write_prediction_archive(
                prediction_path,
                baseline_vertices_m=baseline,
                candidate_vertices_m=inference.candidate_vertices_m,
                selected_vertices_m=selected,
                candidate_covariance_m2=covariance,
                query_update_m=update_m,
                inference=inference,
            )
        except ValueError as error:
            inference_reason = str(error)
            prediction_path = frame_root / "prediction.npz"
            _write_fallback_prediction(prediction_path, baseline)

        if not inference_admissible and selected is not baseline:
            raise AssertionError("rejected frame did not preserve the baseline object")
        seal = {
            "schema_version": 1,
            "artifact_kind": "PokeFlexPriorAwareBeliefSourcePanelPredictionSeal",
            "panel_protocol_file_sha256": _sha256(panel_protocol_path),
            "smoke_protocol_file_sha256": _sha256(smoke_protocol_path),
            "source_revision": source_revision,
            "take_id": take_root.name,
            "source_frame": source_frame,
            "target_frame": target_frame,
            "future_observation_used": False,
            "target_mesh_opened": False,
            "prediction_npz_sha256": _sha256(prediction_path),
            "inference_admissible": inference_admissible,
            "reason": inference_reason,
            "observation_artifact_id": observation_artifact_id,
            "linearization_artifact_id": linearization_artifact_id,
        }
        seal_path = frame_root / "prediction_seal.json"
        _atomic_json(seal_path, seal)

        target_mesh = _load_mesh(take_root / "meshes" / f"mesh-f{target_frame:05d}.obj")
        target_sample = _surface_sample(
            np.asarray(target_mesh.vertices, dtype=np.float64) / 1000.0,
            np.asarray(target_mesh.faces, dtype=np.int64),
            sample_count,
            base_seed + target_frame,
        )
        baseline_sample = _surface_sample(
            baseline,
            template_faces,
            sample_count,
            base_seed + target_frame,
        )
        selected_sample = _surface_sample(
            selected,
            template_faces,
            sample_count,
            base_seed + target_frame,
        )
        baseline_error = _cd_ul1_mm(baseline_sample, target_sample)
        selected_error = _cd_ul1_mm(selected_sample, target_sample)
        coverage, nees = _surface_posterior_diagnostic(
            selected,
            covariance,
            target_sample,
        )
        records.append(
            {
                "source_frame": source_frame,
                "target_frame": target_frame,
                "prediction_seal_sha256": _sha256(seal_path),
                "action_supported": action_supported,
                "inference_admissible": inference_admissible,
                "reason": inference_reason,
                "observation_count": observation_count,
                "robust_weight_mean": robust_weight_mean,
                "maximum_query_update_mm": maximum_query_update_mm,
                "registration_diagnostics": registration_diagnostics,
                "released_checkpoint_CD_UL1_mm": baseline_error,
                "prior_aware_selected_CD_UL1_mm": selected_error,
                "absolute_change_mm": selected_error - baseline_error,
                "surface_proxy_coverage_90": coverage,
                "surface_proxy_NEES": nees,
            }
        )

    baseline_values = np.asarray(
        [record["released_checkpoint_CD_UL1_mm"] for record in records]
    )
    selected_values = np.asarray(
        [record["prior_aware_selected_CD_UL1_mm"] for record in records]
    )
    admitted = np.asarray(
        [record["inference_admissible"] for record in records],
        dtype=bool,
    )
    result = {
        "schema_version": 1,
        "artifact_kind": "PokeFlexPriorAwareBeliefSourcePanelTakeResult",
        "claim_boundary": panel["claim_boundary"],
        "panel_protocol_file_sha256": _sha256(panel_protocol_path),
        "smoke_protocol_file_sha256": _sha256(smoke_protocol_path),
        "source_revision": source_revision,
        "take_id": take_root.name,
        "template_frame": template_frame,
        "template_sha256": _sha256(template_path),
        "template_preprocessing": template_preprocessing,
        "maximum_frame": frame_limit,
        "future_observation_used": False,
        "all_predictions_sealed_before_target_mesh": True,
        "calibration_median_residual_mm": [
            item.median_residual_m * 1000.0 for item in calibrations
        ],
        "calibration_p90_residual_mm": [
            item.p90_residual_m * 1000.0 for item in calibrations
        ],
        "aggregate": {
            "target_frame_count": len(records),
            "released_checkpoint_CD_UL1_mm": float(np.mean(baseline_values)),
            "prior_aware_selected_CD_UL1_mm": float(np.mean(selected_values)),
            "relative_change_percent": float(
                100.0
                * (np.mean(selected_values) - np.mean(baseline_values))
                / np.mean(baseline_values)
            ),
            "admitted_frame_count": int(np.sum(admitted)),
            "admitted_frame_fraction": float(np.mean(admitted)),
            "admitted_win_count": int(
                np.sum(admitted & (selected_values < baseline_values))
            ),
            "admitted_loss_count": int(
                np.sum(admitted & (selected_values > baseline_values))
            ),
            "false_safe_rate": (
                float(np.mean(selected_values[admitted] > baseline_values[admitted]))
                if np.any(admitted)
                else 0.0
            ),
            "mean_surface_proxy_coverage_90": float(
                np.mean([record["surface_proxy_coverage_90"] for record in records])
            ),
            "mean_surface_proxy_NEES": float(
                np.mean([record["surface_proxy_NEES"] for record in records])
            ),
        },
        "frames": records,
    }
    _atomic_json(output_root / "result.json", result)
    return result


def _summarize_panel(
    protocol: dict[str, object],
    take_results: list[dict[str, object]],
) -> dict[str, object]:
    by_object: dict[str, list[dict[str, object]]] = {}
    for result in take_results:
        object_name, _, _ = result["take_id"].rpartition("_T")
        by_object.setdefault(object_name, []).append(result)
    objects = {}
    for object_name, results in sorted(by_object.items()):
        baseline = float(
            np.mean(
                [item["aggregate"]["released_checkpoint_CD_UL1_mm"] for item in results]
            )
        )
        selected = float(
            np.mean(
                [
                    item["aggregate"]["prior_aware_selected_CD_UL1_mm"]
                    for item in results
                ]
            )
        )
        objects[object_name] = {
            "take_count": len(results),
            "released_checkpoint_CD_UL1_mm": baseline,
            "prior_aware_selected_CD_UL1_mm": selected,
            "relative_improvement": (baseline - selected) / baseline,
        }
    expected_count = len(protocol["cohort"]["development_objects"]) * len(
        protocol["cohort"]["take_suffixes"]
    )
    complete = len(take_results) == expected_count and set(objects) == set(
        protocol["cohort"]["development_objects"]
    )
    baseline = float(
        np.mean([item["released_checkpoint_CD_UL1_mm"] for item in objects.values()])
    )
    selected = float(
        np.mean([item["prior_aware_selected_CD_UL1_mm"] for item in objects.values()])
    )
    admitted_frames = [
        frame
        for result in take_results
        for frame in result["frames"]
        if frame["inference_admissible"]
    ]
    all_frames = [frame for result in take_results for frame in result["frames"]]
    gate = protocol["source_advancement_gate"]
    object_wins = sum(item["relative_improvement"] > 0.0 for item in objects.values())
    object_losses = sum(item["relative_improvement"] < 0.0 for item in objects.values())
    maximum_regression = max(
        0.0,
        *[-item["relative_improvement"] for item in objects.values()],
    )
    false_safe = (
        float(
            np.mean(
                [
                    frame["prior_aware_selected_CD_UL1_mm"]
                    > frame["released_checkpoint_CD_UL1_mm"]
                    for frame in admitted_frames
                ]
            )
        )
        if admitted_frames
        else 0.0
    )
    admitted_fraction = len(admitted_frames) / max(len(all_frames), 1)
    checks = {
        "cohort_complete": complete,
        "minimum_object_balanced_relative_improvement": (
            (baseline - selected) / baseline
            >= float(gate["minimum_object_balanced_relative_improvement"])
        ),
        "minimum_object_wins": object_wins >= int(gate["minimum_object_wins"]),
        "maximum_object_losses": object_losses <= int(gate["maximum_object_losses"]),
        "maximum_per_object_relative_regression": maximum_regression
        <= float(gate["maximum_per_object_relative_regression"]),
        "maximum_false_safe_rate_among_admitted_frames": false_safe
        <= float(gate["maximum_false_safe_rate_among_admitted_frames"]),
        "minimum_admitted_frame_fraction": admitted_fraction
        >= float(gate["minimum_admitted_frame_fraction"]),
        "finite_posterior_diagnostics": all(
            np.isfinite(frame["surface_proxy_coverage_90"])
            and np.isfinite(frame["surface_proxy_NEES"])
            for frame in all_frames
        ),
    }
    return {
        "schema_version": 1,
        "artifact_kind": "PokeFlexPriorAwareBeliefSourcePanelSummary",
        "cohort_complete": complete,
        "take_count": len(take_results),
        "object_count": len(objects),
        "object_balanced": {
            "released_checkpoint_CD_UL1_mm": baseline,
            "prior_aware_selected_CD_UL1_mm": selected,
            "relative_improvement": (baseline - selected) / baseline,
            "object_wins": object_wins,
            "object_losses": object_losses,
            "maximum_object_regression": maximum_regression,
            "admitted_frame_count": len(admitted_frames),
            "admitted_frame_fraction": admitted_fraction,
            "false_safe_rate": false_safe,
        },
        "objects": objects,
        "gate_checks": checks,
        "gate_passed": complete and all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--take-id", action="append")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=(
            _repository_root()
            / "configs"
            / "sota"
            / "pokeflex_prior_aware_belief_source_panel_v1.json"
        ),
    )
    parser.add_argument(
        "--smoke-protocol",
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
    protocol = _load_panel_protocol(args.protocol)
    expected = [
        f"{object_name}_{suffix}"
        for object_name in protocol["cohort"]["development_objects"]
        for suffix in protocol["cohort"]["take_suffixes"]
    ]
    selected = args.take_id or expected
    unknown = sorted(set(selected) - set(expected))
    if unknown:
        raise ValueError(f"takes lie outside the locked source panel: {unknown}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_records = []
    take_results = []
    for take_id in selected:
        take_output = args.output_root / take_id
        result_path = take_output / "result.json"
        if args.skip_existing and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("panel_protocol_file_sha256") != _sha256(args.protocol):
                raise ValueError(f"existing take uses another protocol: {take_id}")
            status = "existing"
        else:
            try:
                result = run_take(
                    take_root=(args.dataset_root / take_id).resolve(),
                    output_root=take_output.resolve(),
                    panel_protocol_path=args.protocol.resolve(),
                    smoke_protocol_path=args.smoke_protocol.resolve(),
                    parent_protocol_path=args.parent_protocol.resolve(),
                    upstream_checkout=args.upstream_checkout.resolve(),
                    checkpoint_root=args.checkpoint_root.resolve(),
                    source_revision=args.source_revision,
                    device=args.device,
                )
                status = "completed"
            except Exception as error:
                progress_records.append(
                    {
                        "take_id": take_id,
                        "status": "failed-no-replacement",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                _atomic_json(
                    args.output_root / "progress.json",
                    {
                        "schema_version": 1,
                        "artifact_kind": "PokeFlexPriorAwareBeliefSourcePanelProgress",
                        "replacement_allowed": False,
                        "records": progress_records,
                    },
                )
                print(json.dumps(progress_records[-1], sort_keys=True), flush=True)
                continue
        take_results.append(result)
        progress_records.append(
            {
                "take_id": take_id,
                "status": status,
                "result_sha256": _sha256(result_path),
            }
        )
        _atomic_json(
            args.output_root / "progress.json",
            {
                "schema_version": 1,
                "artifact_kind": "PokeFlexPriorAwareBeliefSourcePanelProgress",
                "replacement_allowed": False,
                "records": progress_records,
            },
        )
        print(json.dumps(progress_records[-1], sort_keys=True), flush=True)
    summary = _summarize_panel(protocol, take_results)
    _atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary["object_balanced"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
