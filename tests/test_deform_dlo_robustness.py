import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deform_causal_inputs,
    fit_deform_local_residual,
    predict_deform_local_residual,
    serialize_deform_local_residual_model,
)
from bayesian_phystwin_experiments.deform_dlo_pyelastica import (
    deform_pyelastica_directors,
    deform_pyelastica_kinematic_sample,
    deform_pyelastica_parameter_bank,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS,
    assign_deform_dlo3_source_partitions,
    augment_deform_local_residual_full_covariance,
    build_deform_bayesian_covariance_ablation_v1,
    build_deform_dlo3_source_manifest,
    calibrate_deform_full_covariance,
    deform_bayesian_covariance_archive_key,
    deform_local_feature_indices,
    evaluate_deform_backend_portability_report,
    evaluate_deform_backend_source_gate,
    evaluate_deform_compute_matched_report,
    evaluate_deform_dlo3_source_gate,
    evaluate_deform_dlo3_stability_gate,
    evaluate_deform_dlo3_target_gate,
    evaluate_deform_predictive_distribution,
    fit_deform_local_residual_variant,
    load_deform_dlo3_method_seal_recovery_v1,
    load_deform_dlo_robustness_v1_protocol,
    predict_deform_local_residual_full_covariance,
    predict_deform_local_residual_variant,
    scale_deform_coordinate_covariance,
    validate_deform_bayesian_audit_v1,
    validate_deform_compute_matched_report_v1,
    validate_deform_dlo3_alltrain_compute_match_v1,
    validate_deform_dlo3_backend_result_v1,
    validate_deform_dlo3_sensitivity_result_v1,
    validate_deform_dlo3_source_manifest,
    verify_deform_dlo3_backend_artifacts_v1,
    verify_deform_dlo3_evaluator_bayesian_artifacts_v1,
    verify_deform_dlo3_evaluator_compute_matched_artifacts_v1,
    verify_deform_dlo3_seed_bayesian_artifacts_v1,
    verify_deform_dlo3_seed_diagnostic_artifacts_v1,
    verify_deform_dlo3_sensitivity_artifacts_v1,
    verify_deform_dlo3_stability_artifacts_v1,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "sota" / "deform_dlo_robustness_v1.json"
RECOVERY_LOCK = ROOT / "configs" / "sota" / "deform_dlo3_method_seal_recovery_v1.json"
RECOVERY_AUTHORIZATION = (
    ROOT
    / "configs"
    / "sota"
    / "deform_dlo3_method_seal_completion_authorization_v1.json"
)
RECOVERY_VALIDATION = (
    ROOT
    / "results"
    / "sota"
    / "deform_dlo3_robustness_v2"
    / "method_seal_recovery_validation.json"
)
SEED_RUNNER = ROOT / "scripts" / "remote" / "run_deform_dlo3_robustness_seed_v1.py"
RECOVERY_RUNNER = (
    ROOT / "scripts" / "remote" / "run_deform_dlo3_method_seal_recovery_v1.py"
)
STABILITY_RUNNER = (
    ROOT / "scripts" / "remote" / "evaluate_deform_dlo3_stability_gate_v1.py"
)
SENSITIVITY_RUNNER = ROOT / "scripts" / "remote" / "run_deform_dlo3_sensitivity_v1.py"
PYELASTICA_RUNNER = (
    ROOT / "scripts" / "remote" / "run_deform_dlo3_pyelastica_source_v1.py"
)
ALLTRAIN_RUNNER = (
    ROOT / "scripts" / "remote" / "run_deform_dlo3_robustness_alltrain_v1.py"
)
EVALUATOR_RUNNER = (
    ROOT / "scripts" / "remote" / "run_deform_dlo3_robustness_evaluator_v1.py"
)
READINESS_RUNNER = (
    ROOT / "scripts" / "remote" / "attest_deform_dlo3_robustness_readiness_v1.py"
)


def _payload() -> dict[str, object]:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _residual_problem(count: int = 12) -> tuple[np.ndarray, ...]:
    frames = 14
    nodes = 7
    trajectories = np.zeros((count, frames, nodes, 3), dtype=np.float64)
    arc = np.linspace(-1.0, 1.0, nodes)
    for case in range(count):
        amplitude = 0.01 + 0.001 * case
        for frame in range(frames):
            phase = frame / (frames - 1)
            trajectories[case, frame, :, 0] = arc
            trajectories[case, frame, :, 1] = amplitude * phase * (1.0 - arc**2)
            trajectories[case, frame, :, 2] = 0.002 * case * phase
            trajectories[case, frame, :2, 1] += amplitude * phase
            trajectories[case, frame, -2:, 1] -= 0.5 * amplitude * phase
    initial, action = deform_causal_inputs(trajectories)
    targets = trajectories[:, 2:].copy()
    baseline = targets.copy()
    time = np.linspace(0.0, 1.0, targets.shape[1])
    bias = (0.004 + 0.0005 * np.arange(count))[:, None, None]
    baseline[:, :, 2:-2, 1] -= bias * time[None, :, None]
    baseline[:, :, 2:-2, 2] += 0.25 * bias * np.square(time)[None, :, None]
    names = np.asarray([f"case-{index:02d}" for index in range(count)])
    return initial, action, baseline, targets, names


def _seed_result(
    seed: int, *, ratio: float = 0.90, passed: bool = True
) -> dict[str, object]:
    cases = [
        {
            "name": f"case-{index}",
            "candidate_to_baseline_ratio": ratio,
        }
        for index in range(8)
    ]
    distributions = {
        name: {
            "schema_version": 1,
            "contract": "deform-dlo-predictive-distribution-metrics-v1",
            "mean_coordinate_l1_m": 0.006,
            "gaussian_nll": -1.0,
            "coordinate_nees": 1.0,
            "multivariate_nees": 1.0,
            "coordinate_coverage_90": 0.9,
            "interval_width_m": 0.01,
            "energy_score": 0.005,
            "energy_score_sample_count": 32,
            "energy_score_seed": 0,
        }
        for name in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
    }
    return {
        "contract": "deform-dlo3-robustness-seed-result-v1",
        "seed": seed,
        "protocol": {"sha256": "a" * 64},
        "source_manifest": {"sha256": "b" * 64},
        "primary_source_gate": {
            "contract": "deform-dlo3-robustness-source-gate-v1",
            "case_count": 8,
            "candidate_mean_l1_m": ratio * 0.008,
            "baseline_mean_l1_m": 0.008,
            "maximum_case_ratio": ratio,
            "passed": passed,
            "cases": cases,
        },
        "bayesian_audit": {
            "calibration": {
                "schema_version": 1,
                "contract": "deform-dlo-full-covariance-calibration-v1",
                "trajectory_scores": [2.0] * 9,
                "rank": 9,
                "order_statistic": "maximum-of-nine",
                "nominal_coordinate_coverage": 0.9,
                "standardized_radius": 2.0,
                "gaussian_radius": 1.6448536269514722,
                "variance_scale": 4.0,
                "confidence_increase_forbidden": True,
                "source_test_opened": False,
                "official_eval_read": False,
            },
            "uncalibrated": distributions[
                "trajectory-clustered-full-coordinate-covariance-v1"
            ],
            "calibrated": distributions["calibrated-full-coordinate-covariance-v1"],
            "distributions": distributions,
            "point_mean_unchanged": True,
            "distribution_selection": "none",
            "source_test_outcomes_used_for_covariance_construction": False,
        },
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _backend_result_with_artifacts(tmp_path: Path) -> dict[str, object]:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    feature_count = 92
    internal_count = 8
    dimension = feature_count + 1
    model = {
        "schema_version": 1,
        "contract": "deform-dlo-local-residual-model-v1",
        "node_count": 12,
        "prediction_horizon": 498,
        "feature_count": feature_count,
        "trajectory_clusters": tuple((f"fit-{index}",) for index in range(39)),
        "feature_location": np.zeros((internal_count, feature_count)),
        "feature_scale": np.ones((internal_count, feature_count)),
        "coefficients": np.zeros((internal_count, dimension, 3)),
        "coefficient_covariance": np.zeros((internal_count, 3, dimension, dimension)),
        "residual_variance": np.ones((internal_count, 3)) * 1e-6,
        "ridge": 1.0,
        "variance_floor_m2": 1e-6,
    }
    model_payload = serialize_deform_local_residual_model(model)
    model_payload["coefficient_covariance_full"] = np.zeros(
        (internal_count, 3, 3, dimension, dimension)
    )
    model_payload["residual_covariance_full"] = np.repeat(
        np.eye(3, dtype=np.float64)[None] * 1e-6,
        internal_count,
        axis=0,
    )
    model_path = tmp_path / "full_covariance_model.npz"
    np.savez_compressed(model_path, **model_payload)
    calibration = {
        "schema_version": 1,
        "contract": "deform-dlo-full-covariance-calibration-v1",
        "trajectory_scores": [2.0] * 9,
        "rank": 9,
        "order_statistic": "maximum-of-nine",
        "nominal_coordinate_coverage": 0.9,
        "standardized_radius": 2.0,
        "gaussian_radius": 1.6448536269514722,
        "variance_scale": 4.0,
        "confidence_increase_forbidden": True,
        "source_test_opened": False,
        "primary_eval_read": False,
    }
    calibration_path = tmp_path / "covariance_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, sort_keys=True) + "\n", encoding="utf-8"
    )
    protocol_identity = _file_identity(PROTOCOL)
    method = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-method-seal-v1",
        "protocol": protocol_identity,
        "full_covariance_model": _file_identity(model_path),
        "covariance_calibration": _file_identity(calibration_path),
        "selected_parameters": deform_pyelastica_parameter_bank(protocol)[
            0
        ].to_record(),
        "ridge": 1.0,
        "shrinkage": 0.25,
        "source_test_opened": False,
        "primary_eval_read": False,
        "selection_effect_after_fit": "none",
    }
    method_path = tmp_path / "method_seal.json"
    method_path.write_text(json.dumps(method, sort_keys=True) + "\n", encoding="utf-8")
    backend = np.ones((8, 498, 12, 3), dtype=np.float64) * 0.02
    candidate = backend.copy()
    candidate[:, :, 2:-2] = 0.018
    raw_covariance = np.ones((*backend.shape, 3), dtype=np.float64) * 1e-6
    predictions_path = tmp_path / "source_predictions.npz"
    np.savez_compressed(
        predictions_path,
        names=np.asarray([f"case-{index}" for index in range(8)]),
        backend=backend,
        candidate=candidate,
        coordinate_covariance_m2=raw_covariance,
        calibrated_coordinate_covariance_m2=raw_covariance * 4.0,
    )
    method_identity = _file_identity(method_path)
    prediction_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-prediction-seal-v1",
        "method_seal": method_identity,
        "predictions": _file_identity(predictions_path),
        "source_outcomes_scored": False,
        "primary_eval_read": False,
    }
    prediction_seal_path = tmp_path / "prediction_seal.json"
    prediction_seal_path.write_text(
        json.dumps(prediction_seal, sort_keys=True) + "\n", encoding="utf-8"
    )
    targets = np.zeros_like(backend)
    gate = evaluate_deform_backend_source_gate(
        candidate,
        backend,
        targets,
        [f"case-{index}" for index in range(8)],
        protocol,
    )
    distributions = _seed_result(42)["bayesian_audit"]
    assert isinstance(distributions, dict)
    return {
        "contract": "deform-dlo3-pyelastica-source-result-v1",
        "protocol": protocol_identity,
        "method_seal": method_identity,
        "prediction_seal": _file_identity(prediction_seal_path),
        "source_gate": gate,
        "bayesian_audit": {
            "uncalibrated": distributions["uncalibrated"],
            "calibrated": distributions["calibrated"],
            "point_mean_unchanged_by_calibration": True,
        },
        "backend_target_arm_authorized": True,
        "primary_target_authorized": False,
        "selection_effect": "none-after-fit",
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }


def _bayesian_prediction_archive(
    path: Path,
    *,
    case_count: int,
    include_source_alias: bool,
    include_compute_control: bool = False,
    omit_arm: str | None = None,
) -> None:
    candidate = np.zeros((case_count, 3, 6, 3), dtype=np.float64)
    raw = np.zeros((*candidate.shape, 3), dtype=np.float64)
    raw[:, :, 2:-2] = np.eye(3, dtype=np.float64) * 1e-6
    payload: dict[str, np.ndarray] = {
        "names": np.asarray([f"case-{index}" for index in range(case_count)]),
        "candidate": candidate,
        "calibrated_coordinate_covariance_m2": raw * 4.0,
    }
    if include_compute_control:
        payload["baseline"] = np.full_like(candidate, 0.008)
        payload["compute_matched_physical"] = np.full_like(candidate, 0.007)
    if include_source_alias:
        payload["coordinate_covariance_m2"] = raw
    for index, label in enumerate(DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS):
        if label == omit_arm:
            continue
        covariance = raw * float(index + 1)
        if label == "trajectory-clustered-full-coordinate-covariance-v1":
            covariance = raw
        elif label == "calibrated-full-coordinate-covariance-v1":
            covariance = raw * 4.0
        payload[deform_bayesian_covariance_archive_key(label)] = covariance
    np.savez_compressed(path, **payload)


def _seed_result_with_artifacts(
    tmp_path: Path,
    *,
    omit_arm: str | None = None,
) -> dict[str, object]:
    result = _seed_result(42)
    method_path = tmp_path / "method_seal.json"
    method_path.write_text('{"contract":"method"}\n', encoding="utf-8")
    predictions_path = tmp_path / "source_predictions.npz"
    _bayesian_prediction_archive(
        predictions_path,
        case_count=8,
        include_source_alias=True,
        omit_arm=omit_arm,
    )
    method_identity = _file_identity(method_path)
    seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-prediction-seal-v1",
        "seed": 42,
        "method_seal": method_identity,
        "predictions": _file_identity(predictions_path),
        "source_test_case_count": 8,
        "bayesian_ablation_distributions": list(
            DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        ),
        "bayesian_covariance_archive_keys": {
            label: deform_bayesian_covariance_archive_key(label)
            for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        },
        "bayesian_point_means_identical": True,
        "source_outcomes_scored": False,
        "official_eval_read": False,
    }
    seal_path = tmp_path / "prediction_seal.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8")
    result["method_seal"] = method_identity
    result["prediction_seal"] = _file_identity(seal_path)
    return result


def _seed_diagnostic_result_with_artifacts(
    tmp_path: Path,
    *,
    seed: int = 42,
) -> dict[str, object]:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    result = _seed_result(seed)
    result["protocol"] = _file_identity(PROTOCOL)
    calibration = result["bayesian_audit"]["calibration"]
    calibration_path = tmp_path / "diagnostic_calibration.json"
    calibration_path.write_text(
        json.dumps(calibration, sort_keys=True) + "\n", encoding="utf-8"
    )
    physical_path = tmp_path / "physical.pt"
    physical_path.write_bytes(b"physical-checkpoint")
    compute_path = tmp_path / "compute.pt"
    compute_path.write_bytes(b"compute-checkpoint")
    local_path = tmp_path / "local.npz"
    np.savez_compressed(local_path, value=np.asarray([1.0]))
    full_path = tmp_path / "full.npz"
    np.savez_compressed(full_path, value=np.asarray([1.0]))
    physical_identity = {
        **_file_identity(physical_path),
        "update": 6400,
        "label": "registered",
    }
    compute_identity = {**_file_identity(compute_path), "update": 6401}

    model_specs = {
        "persistence-plus-full-local": ("full-local", "initial-action-local"),
        "physical-plus-intercept-only": (
            "intercept-only",
            "initial-action-local",
        ),
        "physical-plus-full-no-action": (
            "full-no-action",
            "initial-action-local",
        ),
        "physical-plus-full-global-frame": (
            "full-global",
            "action-centered-global",
        ),
    }
    model_identities: dict[str, object] = {}
    for label, (arm, frame) in model_specs.items():
        feature_indices = deform_local_feature_indices(arm)
        path = tmp_path / f"{label}.npz"
        np.savez_compressed(
            path,
            arm=np.asarray([arm]),
            coordinate_frame=np.asarray([frame]),
            node_count=np.asarray([12], dtype=np.int64),
            prediction_horizon=np.asarray([498], dtype=np.int64),
            feature_indices=np.asarray(feature_indices, dtype=np.int64),
            feature_location=np.zeros((8, len(feature_indices))),
            feature_scale=np.ones((8, len(feature_indices))),
            coefficients=np.zeros((8, len(feature_indices) + 1, 3)),
            ridge=np.asarray([1.0]),
        )
        model_identities[label] = _file_identity(path)
    method = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-method-seal-v1",
        "seed": seed,
        "protocol": result["protocol"],
        "source_manifest": result["source_manifest"],
        "physical_checkpoint": physical_identity,
        "compute_matched_checkpoint": compute_identity,
        "local_residual_model": _file_identity(local_path),
        "full_covariance_model": _file_identity(full_path),
        "covariance_calibration": _file_identity(calibration_path),
        "mechanism_models": model_identities,
        "ridge": 1.0,
        "shrinkage": 0.25,
        "source_test_opened": False,
        "official_eval_read": False,
        "target_selection": False,
    }
    method_path = tmp_path / "diagnostic_method_seal.json"
    method_path.write_text(json.dumps(method, sort_keys=True) + "\n", encoding="utf-8")

    shape = (8, 498, 12, 3)
    targets = np.zeros(shape, dtype=np.float64)
    physical = np.full(shape, 0.008, dtype=np.float64)
    candidate = np.full(shape, 0.006, dtype=np.float64)
    compute_prediction = np.full(shape, 0.007, dtype=np.float64)
    names = [f"case-{index}" for index in range(8)]
    primary_gate = evaluate_deform_dlo3_source_gate(
        candidate, physical, targets, names, protocol
    )
    physical_gate = evaluate_deform_dlo3_source_gate(
        physical, physical, targets, names, protocol
    )
    arms = protocol["mechanism_ablation"]["arms"]
    mechanism = {
        label: physical_gate if label == "physical-only" else primary_gate
        for label in arms
    }
    predictions_path = tmp_path / "diagnostic_source_predictions.npz"
    payload = {
        "names": np.asarray(names),
        "physical": physical,
        "compute_matched_physical": compute_prediction,
        "candidate": candidate,
    }
    payload.update(
        {
            f"mechanism_{label}": physical if label == "physical-only" else candidate
            for label in arms
        }
    )
    raw_covariance = np.zeros((*shape, 3), dtype=np.float64)
    raw_covariance[:, :, 2:-2] = np.eye(3, dtype=np.float64) * 1e-6
    payload["coordinate_covariance_m2"] = raw_covariance
    payload["calibrated_coordinate_covariance_m2"] = raw_covariance * 4.0
    for index, label in enumerate(DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS):
        covariance = raw_covariance * float(index + 1)
        if label == "trajectory-clustered-full-coordinate-covariance-v1":
            covariance = raw_covariance
        elif label == "calibrated-full-coordinate-covariance-v1":
            covariance = raw_covariance * 4.0
        payload[deform_bayesian_covariance_archive_key(label)] = covariance
    np.savez_compressed(predictions_path, **payload)
    method_identity = _file_identity(method_path)
    seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-source-prediction-seal-v1",
        "seed": seed,
        "method_seal": method_identity,
        "predictions": _file_identity(predictions_path),
        "source_test_case_count": 8,
        "bayesian_ablation_distributions": list(
            DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        ),
        "bayesian_covariance_archive_keys": {
            label: deform_bayesian_covariance_archive_key(label)
            for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        },
        "bayesian_point_means_identical": True,
        "source_outcomes_scored": False,
        "official_eval_read": False,
    }
    seal_path = tmp_path / "diagnostic_prediction_seal.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8")
    result.update(
        {
            "method_seal": method_identity,
            "prediction_seal": _file_identity(seal_path),
            "physical_checkpoint": physical_identity,
            "compute_match": {
                "schema_version": 1,
                "contract": "deform-dlo3-compute-match-v1",
                "seed": seed,
                "local_residual_wall_seconds": 0.5,
                "median_update_seconds_6301_6400": 1.0,
                "additional_updates": 1,
                "start_update": 6400,
                "end_update": 6401,
                "source_test_opened": False,
                "official_eval_read": False,
                "checkpoint": compute_identity,
                "source_mean_l1_m": 0.007,
            },
            "primary_source_gate": primary_gate,
            "mechanism_ablation": mechanism,
        }
    )
    return result


def _stability_result_with_artifacts(tmp_path: Path) -> dict[str, object]:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    seed_results: list[dict[str, object]] = []
    seed_paths: list[Path] = []
    for seed in (42, 43, 44):
        seed_root = tmp_path / f"seed-{seed}"
        seed_root.mkdir()
        seed_result = _seed_diagnostic_result_with_artifacts(seed_root, seed=seed)
        seed_path = seed_root / "source_result.json"
        seed_path.write_text(
            json.dumps(seed_result, sort_keys=True) + "\n", encoding="utf-8"
        )
        seed_results.append(seed_result)
        seed_paths.append(seed_path)
    gate = evaluate_deform_dlo3_stability_gate(seed_results, protocol)
    bayesian = [
        verify_deform_dlo3_seed_bayesian_artifacts_v1(result) for result in seed_results
    ]
    diagnostics = [
        verify_deform_dlo3_seed_diagnostic_artifacts_v1(result, protocol)
        for result in seed_results
    ]
    return {
        **gate,
        "bayesian_artifacts_verified": True,
        "bayesian_artifact_verifications": bayesian,
        "diagnostic_artifacts_verified": True,
        "diagnostic_artifact_verifications": diagnostics,
        "diagnostic_seed_count": 3,
        "protocol": _file_identity(PROTOCOL),
        "seed_results": [_file_identity(path) for path in seed_paths],
    }


def _sensitivity_result_with_artifacts(tmp_path: Path) -> dict[str, object]:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    parent_root = tmp_path / "parent"
    parent_root.mkdir()
    parent = _seed_diagnostic_result_with_artifacts(parent_root)
    parent_path = tmp_path / "seed_result.json"
    parent_path.write_text(json.dumps(parent, sort_keys=True) + "\n", encoding="utf-8")
    parent_seal = json.loads(
        Path(str(parent["prediction_seal"]["path"])).read_text(encoding="utf-8")
    )
    with np.load(
        Path(str(parent_seal["predictions"]["path"])), allow_pickle=False
    ) as archive:
        names = np.asarray(archive["names"])
        physical = np.asarray(archive["physical"])
        candidate = np.asarray(archive["candidate"])
    labels = (
        "pbd-5",
        "pbd-10",
        "pbd-20",
        "stiffness-0.9",
        "stiffness-1.0",
        "stiffness-1.1",
    )
    predictions_path = tmp_path / "sensitivity_predictions.npz"
    payload = {"names": names}
    payload.update({f"physical_{label}": physical for label in labels})
    payload.update({f"candidate_{label}": candidate for label in labels})
    np.savez_compressed(predictions_path, **payload)
    seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-sensitivity-prediction-seal-v1",
        "predictions": _file_identity(predictions_path),
        "variant_count": 6,
        "source_outcomes_scored": False,
        "primary_eval_read": False,
    }
    seal_path = tmp_path / "sensitivity_prediction_seal.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8")
    targets = np.zeros_like(physical)
    gate = evaluate_deform_dlo3_source_gate(
        candidate, physical, targets, names.tolist(), protocol
    )
    return {
        "contract": "deform-dlo3-physics-solver-sensitivity-result-v1",
        "protocol": parent["protocol"],
        "source_manifest": parent["source_manifest"],
        "seed_result": _file_identity(parent_path),
        "prediction_seal": _file_identity(seal_path),
        "variants": {label: json.loads(json.dumps(gate)) for label in labels},
        "selection_effect": "none",
        "nominal_replay_exact": True,
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }


def _evaluator_result_with_artifacts(tmp_path: Path) -> dict[str, object]:
    source_audit = _seed_result(42)["bayesian_audit"]
    assert isinstance(source_audit, dict)
    predictions_path = tmp_path / "predictions.npz"
    _bayesian_prediction_archive(
        predictions_path,
        case_count=8,
        include_source_alias=False,
        include_compute_control=True,
    )
    compute_checkpoint = tmp_path / "alltrain_compute.pt"
    compute_checkpoint.write_bytes(b"compute")
    compute_match = tmp_path / "alltrain_compute_match.json"
    compute_match.write_text('{"contract":"compute"}\n', encoding="utf-8")
    compute_verification = {
        "contract": "deform-dlo3-alltrain-compute-match-verification-v1",
        "verified": True,
    }
    sealed_compute = {
        "status": "sealed",
        "checkpoint": _file_identity(compute_checkpoint),
        "compute_match": _file_identity(compute_match),
        "compute_match_verification": compute_verification,
        "selection_effect": "none",
        "retry_authorized": False,
    }
    seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-robustness-evaluator-prediction-seal-v1",
        "mode": "dry-run",
        "predictions": _file_identity(predictions_path),
        "bayesian_ablation_distributions": list(
            DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        ),
        "bayesian_covariance_archive_keys": {
            label: deform_bayesian_covariance_archive_key(label)
            for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        },
        "bayesian_point_means_identical": True,
        "compute_matched_control": sealed_compute,
        "outcomes_scored": False,
        "target_retries": False,
    }
    seal_path = tmp_path / "evaluator_prediction_seal.json"
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8")
    targets = np.zeros((8, 3, 6, 3), dtype=np.float64)
    candidate = np.zeros_like(targets)
    baseline = np.full_like(targets, 0.008)
    compute = np.full_like(targets, 0.007)
    return {
        "contract": "deform-dlo3-robustness-evaluator-dry-run-v1",
        "prediction_seal": _file_identity(seal_path),
        "bayesian_audit": {
            "primary_distribution": "calibrated-full-coordinate-covariance-v1",
            "distributions": source_audit["distributions"],
            "point_mean_unchanged": True,
            "distribution_selection": "none",
            "target_outcomes_used_for_distribution_construction": False,
            "target_outcomes_used_for_distribution_selection": False,
        },
        "compute_matched_control": {
            "status": "scored",
            "checkpoint": sealed_compute["checkpoint"],
            "compute_match": sealed_compute["compute_match"],
            "compute_match_verification": compute_verification,
            "selection_effect": "none",
            "retry_authorized": False,
            "report": evaluate_deform_compute_matched_report(
                candidate,
                baseline,
                compute,
                targets,
                [f"case-{index}" for index in range(8)],
            ),
        },
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "held_v8_access": False,
    }


def test_loads_locked_dlo_robustness_protocol() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)

    assert protocol["prob4d_used"] is False
    assert protocol["freshness"]["primary_dlo"] == "DLO3"
    assert protocol["custody"]["held_v8_access"] is False


def test_pyelastica_bank_and_geometry_are_frozen() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    bank = deform_pyelastica_parameter_bank(protocol)
    assert len(bank) == 36
    assert bank[0].to_record() == {
        "youngs_modulus_pa": 1e5,
        "density_kg_m3": 900.0,
        "damping_constant": 0.1,
        "integration_substeps": 2,
    }
    assert bank[-1].integration_substeps == 8

    parameter = np.linspace(0.0, 1.0, 12)
    positions = np.column_stack((parameter, 0.1 * parameter**2, 0.05 * parameter))
    directors = deform_pyelastica_directors(positions)
    assert directors.shape == (3, 3, 11)
    assert np.allclose(
        np.einsum("ain,bin->abn", directors, directors),
        np.repeat(np.eye(3)[:, :, None], 11, axis=2),
        atol=1e-12,
    )


def test_pyelastica_kinematic_interpolation_is_causal_and_metric() -> None:
    series = np.zeros((3, 4, 3), dtype=np.float64)
    series[1, :, 0] = 0.01
    series[2, :, 0] = 0.03

    position, velocity = deform_pyelastica_kinematic_sample(series, 0.015)

    assert np.allclose(position[:, 0], 0.02)
    assert np.allclose(velocity[:, 0], 2.0)
    assert np.allclose(position[:, 1:], 0.0)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("freshness", "primary_dlo"), "DLO4", "data boundary"),
        (("physical_training", "primary_seed"), 43, "fixed recipe"),
        (("local_residual", "shrinkage"), 0.5, "fixed recipe"),
        (("source_gate", "minimum_case_wins"), 5, "source gates"),
        (("backend_portability", "version"), "latest", "backend contract"),
        (("target_evaluation", "target_retries"), True, "Bayesian or target"),
        (("custody", "held_v8_access"), True, "Bayesian or target"),
    ],
)
def test_rejects_protocol_mutation(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    match: str,
) -> None:
    payload = _payload()
    payload[path[0]][path[1]] = value
    mutated = tmp_path / "protocol.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_deform_dlo_robustness_v1_protocol(mutated)


def test_source_assignment_is_order_independent_and_disjoint() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    names = [f"trajectory_{index:02d}.pkl" for index in range(56)]

    forward = assign_deform_dlo3_source_partitions(names, protocol)
    reverse = assign_deform_dlo3_source_partitions(list(reversed(names)), protocol)

    assert forward == reverse
    assert forward["payload_read"] is False
    fit = set(forward["fit"])
    calibration = set(forward["calibration"])
    source_test = set(forward["source_test"])
    assert (len(fit), len(calibration), len(source_test)) == (39, 9, 8)
    assert fit.isdisjoint(calibration | source_test)
    assert calibration.isdisjoint(source_test)
    assert fit | calibration | source_test == set(names)


def test_source_assignment_rejects_non_basename_or_wrong_count() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    names = [f"trajectory_{index:02d}.pkl" for index in range(56)]

    with pytest.raises(ValueError, match="incomplete"):
        assign_deform_dlo3_source_partitions(names[:-1], protocol)
    names[0] = "nested/trajectory_00.pkl"
    with pytest.raises(ValueError, match="basename"):
        assign_deform_dlo3_source_partitions(names, protocol)


def test_builds_and_revalidates_source_manifest_without_deserialization(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data_set"
    train_root = data_root / "DLO3" / "train"
    train_root.mkdir(parents=True)
    for index in range(56):
        (train_root / f"trajectory_{index:02d}.pkl").write_bytes(
            f"opaque-{index}".encode()
        )

    manifest = build_deform_dlo3_source_manifest(PROTOCOL, data_root)
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(PROTOCOL),
        verify_files=True,
    )

    assert tuple(len(partitions[name]) for name in partitions) == (39, 9, 8)
    assert manifest["trajectory_deserialized"] is False
    assert manifest["primary_eval_enumerated"] is False
    assert not (data_root / "DLO3" / "eval").exists()


def test_source_manifest_detects_byte_or_partition_change(tmp_path: Path) -> None:
    data_root = tmp_path / "data_set"
    train_root = data_root / "DLO3" / "train"
    train_root.mkdir(parents=True)
    for index in range(56):
        (train_root / f"trajectory_{index:02d}.pkl").write_bytes(
            f"opaque-{index}".encode()
        )
    manifest = build_deform_dlo3_source_manifest(PROTOCOL, data_root)
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    protocol_sha256 = sha256_file(PROTOCOL)

    manifest["split"]["fit"], manifest["split"]["source_test"] = (
        manifest["split"]["source_test"],
        manifest["split"]["fit"],
    )
    with pytest.raises(ValueError, match="partition differs"):
        validate_deform_dlo3_source_manifest(
            manifest,
            protocol,
            protocol_sha256=protocol_sha256,
            verify_files=False,
        )

    manifest = build_deform_dlo3_source_manifest(PROTOCOL, data_root)
    changed = Path(next(iter(manifest["trajectories"].values()))["path"])
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity changed"):
        validate_deform_dlo3_source_manifest(
            manifest,
            protocol,
            protocol_sha256=protocol_sha256,
            verify_files=True,
        )


def test_mechanism_feature_subsets_are_fixed() -> None:
    assert len(deform_local_feature_indices("full-local")) == 92
    assert deform_local_feature_indices("intercept-only") == ()
    no_action = deform_local_feature_indices("full-no-action")
    assert len(no_action) == 36
    assert set(no_action).isdisjoint(set(range(24, 66)) | {69, 70} | set(range(80, 92)))

    with pytest.raises(ValueError, match="mechanism arm"):
        deform_local_feature_indices("selected-from-target")


def test_intercept_only_variant_supports_zero_feature_matrices() -> None:
    initial, action, baseline, targets, names = _residual_problem()
    variant = fit_deform_local_residual_variant(
        initial[:9],
        action[:9],
        baseline[:9],
        targets[:9],
        names[:9].tolist(),
        ridge=1.0,
        arm="intercept-only",
    )

    internal_count = baseline.shape[2] - 4
    assert np.asarray(variant["feature_location"]).shape == (internal_count, 0)
    assert np.asarray(variant["feature_scale"]).shape == (internal_count, 0)
    prediction = predict_deform_local_residual_variant(
        variant,
        initial[9:],
        action[9:],
        baseline[9:],
        shrinkage=0.25,
    )["predictions"]

    assert prediction.shape == baseline[9:].shape
    assert np.isfinite(prediction).all()
    assert np.array_equal(
        prediction[:, :, (0, 1, -2, -1)],
        baseline[9:, :, (0, 1, -2, -1)],
    )


def test_full_variant_preserves_frozen_point_operator() -> None:
    initial, action, baseline, targets, names = _residual_problem()
    frozen = fit_deform_local_residual(
        initial[:9],
        action[:9],
        baseline[:9],
        targets[:9],
        names[:9].tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    variant = fit_deform_local_residual_variant(
        initial[:9],
        action[:9],
        baseline[:9],
        targets[:9],
        names[:9].tolist(),
        ridge=1.0,
        arm="full-local",
    )

    expected = predict_deform_local_residual(
        frozen,
        initial[9:],
        action[9:],
        baseline[9:],
        shrinkage=0.25,
    )
    actual = predict_deform_local_residual_variant(
        variant,
        initial[9:],
        action[9:],
        baseline[9:],
        shrinkage=0.25,
    )

    assert np.allclose(
        actual["predictions"], expected["predictions"], rtol=0.0, atol=1e-14
    )
    assert np.array_equal(
        actual["predictions"][:, :, (0, 1, -2, -1)],
        baseline[9:, :, (0, 1, -2, -1)],
    )


def test_full_covariance_preserves_mean_and_supports_calibration() -> None:
    initial, action, baseline, targets, names = _residual_problem(count=9)
    diagonal = fit_deform_local_residual(
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    full_model = augment_deform_local_residual_full_covariance(
        diagonal,
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
    )
    expected = predict_deform_local_residual(
        diagonal,
        initial,
        action,
        baseline,
        shrinkage=0.25,
    )
    full = predict_deform_local_residual_full_covariance(
        full_model,
        initial,
        action,
        baseline,
        shrinkage=0.25,
    )

    assert np.array_equal(full["predictions"], expected["predictions"])
    covariance = full["coordinate_covariance_m2"][:, :, 2:-2]
    assert np.allclose(covariance, covariance.swapaxes(-1, -2), atol=1e-12)
    assert np.min(np.linalg.eigvalsh(covariance)) > 0.0
    assert np.allclose(
        np.diagonal(covariance, axis1=-2, axis2=-1),
        expected["coordinate_variance_m2"][:, :, 2:-2],
        rtol=1e-7,
        atol=1e-12,
    )

    calibration = calibrate_deform_full_covariance(
        full["predictions"],
        targets,
        full["coordinate_covariance_m2"],
    )
    assert calibration["rank"] == 9
    assert calibration["variance_scale"] >= 1.0
    scaled = scale_deform_coordinate_covariance(
        full["coordinate_covariance_m2"],
        float(calibration["variance_scale"]),
    )
    raw_metrics = evaluate_deform_predictive_distribution(
        full["predictions"],
        targets,
        full["coordinate_covariance_m2"],
        sample_count=4,
    )
    scaled_metrics = evaluate_deform_predictive_distribution(
        full["predictions"],
        targets,
        scaled,
        sample_count=4,
    )
    assert all(
        math.isfinite(float(raw_metrics[key]))
        for key in (
            "gaussian_nll",
            "coordinate_nees",
            "multivariate_nees",
            "energy_score",
        )
    )
    assert (
        scaled_metrics["coordinate_coverage_90"]
        >= raw_metrics["coordinate_coverage_90"]
    )
    assert np.array_equal(full["predictions"], expected["predictions"])


def test_bayesian_covariance_ablation_is_complete_and_mean_preserving() -> None:
    initial, action, baseline, targets, names = _residual_problem(count=9)
    diagonal = fit_deform_local_residual(
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    full_model = augment_deform_local_residual_full_covariance(
        diagonal,
        initial,
        action,
        baseline,
        targets,
        names.tolist(),
    )
    point = predict_deform_local_residual(
        diagonal,
        initial,
        action,
        baseline,
        shrinkage=0.25,
    )["predictions"]

    arms = build_deform_bayesian_covariance_ablation_v1(
        full_model,
        initial,
        action,
        baseline,
        shrinkage=0.25,
        variance_scale=4.0,
    )

    assert tuple(arms) == DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
    assert len(
        {
            deform_bayesian_covariance_archive_key(label)
            for label in DEFORM_DLO_BAYESIAN_ABLATION_DISTRIBUTIONS
        }
    ) == len(arms)
    for values in arms.values():
        assert np.array_equal(values["predictions"], point)
        covariance = values["coordinate_covariance_m2"]
        assert covariance.shape == (*point.shape, 3)
        assert np.count_nonzero(covariance[:, :, :2]) == 0
        assert np.count_nonzero(covariance[:, :, -2:]) == 0
        assert np.min(np.linalg.eigvalsh(covariance[:, :, 2:-2])) > 0.0

    current = arms["current-diagonal-conservative-v1"]["coordinate_covariance_m2"]
    propagated = arms["shrinkage-propagated-diagonal"]["coordinate_covariance_m2"]
    current_variance = np.diagonal(current[:, :, 2:-2], axis1=-2, axis2=-1)
    propagated_variance = np.diagonal(propagated[:, :, 2:-2], axis1=-2, axis2=-1)
    assert np.all(propagated_variance <= current_variance + 1e-15)
    assert np.any(propagated_variance < current_variance)

    coefficient = arms["coefficient-only"]["coordinate_covariance_m2"]
    residual = arms["residual-only"]["coordinate_covariance_m2"]
    assert not np.array_equal(coefficient, residual)

    pooled = arms["pooled-isotropic"]["coordinate_covariance_m2"][:, :, 2:-2]
    pooled_diagonal = np.diagonal(pooled, axis1=-2, axis2=-1)
    assert np.allclose(pooled_diagonal, pooled_diagonal.reshape(-1)[0])
    assert (
        np.count_nonzero(
            pooled - np.eye(3, dtype=np.float64) * pooled_diagonal[..., None]
        )
        == 0
    )

    raw = arms["trajectory-clustered-full-coordinate-covariance-v1"][
        "coordinate_covariance_m2"
    ]
    calibrated = arms["calibrated-full-coordinate-covariance-v1"][
        "coordinate_covariance_m2"
    ]
    assert np.array_equal(calibrated, raw * 4.0)


def test_bayesian_covariance_ablation_rejects_unknown_archive_label() -> None:
    with pytest.raises(ValueError, match="covariance arm"):
        deform_bayesian_covariance_archive_key("selected-from-target")


def test_bayesian_audit_validator_requires_every_frozen_arm() -> None:
    complete = _seed_result(42)
    verification = validate_deform_bayesian_audit_v1(complete, context="source")
    assert verification["distribution_count"] == 7
    assert verification["point_mean_unchanged"] is True

    missing = json.loads(json.dumps(complete))
    del missing["bayesian_audit"]["distributions"]["coefficient-only"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_deform_bayesian_audit_v1(missing, context="source")

    selected = json.loads(json.dumps(complete))
    selected["bayesian_audit"]["distribution_selection"] = "best-source-test"
    with pytest.raises(ValueError, match="incomplete"):
        validate_deform_bayesian_audit_v1(selected, context="source")

    shifted = json.loads(json.dumps(complete))
    shifted["bayesian_audit"]["distributions"]["residual-only"][
        "mean_coordinate_l1_m"
    ] = 0.007
    with pytest.raises(ValueError, match="point means differ"):
        validate_deform_bayesian_audit_v1(shifted, context="source")


def test_seed_bayesian_artifact_verifier_rehashes_all_seven_arms(
    tmp_path: Path,
) -> None:
    result = _seed_result_with_artifacts(tmp_path)

    verification = verify_deform_dlo3_seed_bayesian_artifacts_v1(result)

    assert verification["verified"] is True
    assert verification["archive"]["distribution_count"] == 7


def test_seed_bayesian_artifact_verifier_rejects_missing_archive_arm(
    tmp_path: Path,
) -> None:
    result = _seed_result_with_artifacts(tmp_path, omit_arm="coefficient-only")

    with pytest.raises(ValueError, match="archive is incomplete"):
        verify_deform_dlo3_seed_bayesian_artifacts_v1(result)


def test_evaluator_bayesian_artifact_verifier_checks_dry_run(
    tmp_path: Path,
) -> None:
    result = _evaluator_result_with_artifacts(tmp_path)

    verification = verify_deform_dlo3_evaluator_bayesian_artifacts_v1(
        result, expected_mode="dry-run"
    )

    assert verification["verified"] is True
    assert verification["archive"]["case_count"] == 8


def test_evaluator_compute_matched_artifacts_are_sealed_before_scoring(
    tmp_path: Path,
) -> None:
    result = _evaluator_result_with_artifacts(tmp_path)

    verification = verify_deform_dlo3_evaluator_compute_matched_artifacts_v1(
        result, expected_mode="dry-run"
    )

    assert verification["verified"] is True
    assert verification["status"] == "scored"
    assert verification["report"]["case_count"] == 8

    changed = json.loads(json.dumps(result))
    changed["compute_matched_control"]["selection_effect"] = "best-target"
    with pytest.raises(ValueError, match="scored arm differs"):
        verify_deform_dlo3_evaluator_compute_matched_artifacts_v1(
            changed, expected_mode="dry-run"
        )


def test_source_gate_uses_fixed_casewise_arithmetic() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    targets = np.zeros((8, 2, 5, 3), dtype=np.float64)
    baseline = np.full_like(targets, 0.007)
    candidate = np.full_like(targets, 0.006)
    names = [f"case-{index}" for index in range(8)]

    gate = evaluate_deform_dlo3_source_gate(
        candidate, baseline, targets, names, protocol
    )

    assert gate["passed"] is True
    assert gate["wins"] == 8
    assert gate["candidate_mean_l1_m"] == pytest.approx(0.006)
    assert gate["relative_improvement"] == pytest.approx(1.0 - 6.0 / 7.0)

    candidate[0] = 0.008
    changed = evaluate_deform_dlo3_source_gate(
        candidate, baseline, targets, names, protocol
    )
    assert changed["maximum_case_ratio"] == pytest.approx(8.0 / 7.0)
    assert changed["maximum_case_ratio_passed"] is False
    assert changed["passed"] is False


def test_backend_source_gate_has_no_published_reference_shortcut() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    targets = np.zeros((8, 2, 5, 3), dtype=np.float64)
    backend = np.full_like(targets, 0.020)
    candidate = np.full_like(targets, 0.018)
    names = [f"case-{index}" for index in range(8)]

    gate = evaluate_deform_backend_source_gate(
        candidate, backend, targets, names, protocol
    )

    assert gate["passed"] is True
    assert gate["relative_improvement"] == pytest.approx(0.10)
    assert "published_reference_passed" not in gate


def test_backend_portability_report_is_descriptive_and_casewise() -> None:
    targets = np.zeros((14, 2, 5, 3), dtype=np.float64)
    backend = np.full_like(targets, 0.020)
    candidate = np.full_like(targets, 0.018)
    names = [f"case-{index}" for index in range(14)]

    report = evaluate_deform_backend_portability_report(
        candidate, backend, targets, names
    )

    assert report["contract"] == "deform-dlo3-backend-portability-report-v1"
    assert report["case_count"] == 14
    assert report["relative_improvement"] == pytest.approx(0.10)
    assert report["wins"] == 14
    assert report["selection_effect"] == "none"
    assert "passed" not in report

    with pytest.raises(ValueError, match="arrays do not align"):
        evaluate_deform_backend_portability_report(
            candidate, backend, targets, ["duplicate"] * 14
        )


def test_compute_matched_report_is_descriptive_and_self_consistent() -> None:
    targets = np.zeros((8, 2, 5, 3), dtype=np.float64)
    registered = np.full_like(targets, 0.008)
    compute = np.full_like(targets, 0.007)
    candidate = np.full_like(targets, 0.006)
    names = [f"case-{index}" for index in range(8)]

    report = evaluate_deform_compute_matched_report(
        candidate, registered, compute, targets, names
    )
    verification = validate_deform_compute_matched_report_v1(
        report, expected_case_count=8
    )

    assert report["candidate_relative_improvement_over_compute_matched"] == (
        pytest.approx(1.0 - 6.0 / 7.0)
    )
    assert report["compute_matched_relative_improvement_over_registered"] == (
        pytest.approx(1.0 - 7.0 / 8.0)
    )
    assert report["selection_effect"] == "none"
    assert "passed" not in report
    assert verification["verified"] is True

    changed = json.loads(json.dumps(report))
    changed["cases"][0]["candidate_to_compute_matched_ratio"] = 0.5
    with pytest.raises(ValueError, match="report case differs"):
        validate_deform_compute_matched_report_v1(changed, expected_case_count=8)


def test_alltrain_compute_match_requires_exact_timing_rule() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    record = {
        "contract": "deform-dlo3-alltrain-compute-match-v1",
        "seed": 42,
        "local_residual_wall_seconds": 2.1,
        "median_update_seconds_6301_6400": 1.0,
        "additional_updates": 3,
        "start_update": 6400,
        "end_update": 6403,
        "selection_effect": "none",
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "primary_eval_read": False,
    }

    verification = validate_deform_dlo3_alltrain_compute_match_v1(record, protocol)

    assert verification["additional_updates"] == 3
    assert verification["verified"] is True

    record["additional_updates"] = 2
    with pytest.raises(ValueError, match="compute match differs"):
        validate_deform_dlo3_alltrain_compute_match_v1(record, protocol)


def test_backend_artifacts_are_rehashed_for_target_carryover(tmp_path: Path) -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    result = _backend_result_with_artifacts(tmp_path)

    verification = verify_deform_dlo3_backend_artifacts_v1(result, protocol)

    assert verification["verified"] is True
    assert verification["backend_target_arm_authorized"] is True
    assert verification["variance_scale"] == pytest.approx(4.0)

    calibration_path = Path(
        str(
            json.loads(
                Path(str(result["method_seal"]["path"])).read_text(encoding="utf-8")
            )["covariance_calibration"]["path"]
        )
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["standardized_radius"] = 1.0
    calibration_path.write_text(
        json.dumps(calibration, sort_keys=True) + "\n", encoding="utf-8"
    )
    method_path = Path(str(result["method_seal"]["path"]))
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["covariance_calibration"] = _file_identity(calibration_path)
    method_path.write_text(json.dumps(method, sort_keys=True) + "\n", encoding="utf-8")
    result["method_seal"] = _file_identity(method_path)
    prediction_seal_path = Path(str(result["prediction_seal"]["path"]))
    prediction_seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    prediction_seal["method_seal"] = result["method_seal"]
    prediction_seal_path.write_text(
        json.dumps(prediction_seal, sort_keys=True) + "\n", encoding="utf-8"
    )
    result["prediction_seal"] = _file_identity(prediction_seal_path)
    with pytest.raises(ValueError, match="covariance calibration differs"):
        verify_deform_dlo3_backend_artifacts_v1(result, protocol)

    method_path = Path(str(result["method_seal"]["path"]))
    method_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        verify_deform_dlo3_backend_artifacts_v1(result, protocol)


def test_seed_diagnostic_artifacts_bind_mechanisms_and_compute_control(
    tmp_path: Path,
) -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    result = _seed_diagnostic_result_with_artifacts(tmp_path)

    verification = verify_deform_dlo3_seed_diagnostic_artifacts_v1(result, protocol)

    assert verification["verified"] is True
    assert verification["mechanism_arm_count"] == 7
    assert verification["compute_matched_additional_updates"] == 1

    method = json.loads(
        Path(str(result["method_seal"]["path"])).read_text(encoding="utf-8")
    )
    compute_path = Path(str(method["compute_matched_checkpoint"]["path"]))
    compute_path.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="identity changed"):
        verify_deform_dlo3_seed_diagnostic_artifacts_v1(result, protocol)


def test_stability_artifacts_replay_all_three_seed_bundles(tmp_path: Path) -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    result = _stability_result_with_artifacts(tmp_path)

    verification = verify_deform_dlo3_stability_artifacts_v1(result, protocol)

    assert verification["verified"] is True
    assert verification["seed_count"] == 3
    assert verification["gate_passed"] is True
    assert set(verification["seed_result_sha256_by_seed"]) == {"42", "43", "44"}

    seed_path = Path(str(result["seed_results"][1]["path"]))
    seed_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity changed"):
        verify_deform_dlo3_stability_artifacts_v1(result, protocol)


def test_sensitivity_artifacts_bind_full_matrix_and_parent(tmp_path: Path) -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    result = _sensitivity_result_with_artifacts(tmp_path)

    verification = verify_deform_dlo3_sensitivity_artifacts_v1(result, protocol)

    assert verification["artifact_matrix_verified"] is True
    assert verification["variant_count"] == 6

    seal_path = Path(str(result["prediction_seal"]["path"]))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    predictions_path = Path(str(seal["predictions"]["path"]))
    with np.load(predictions_path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]) for key in archive.files}
    payload["candidate_stiffness-1.0"] = payload["candidate_stiffness-1.0"] + 1e-4
    np.savez_compressed(predictions_path, **payload)
    seal["predictions"] = _file_identity(predictions_path)
    seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n", encoding="utf-8")
    result["prediction_seal"] = _file_identity(seal_path)
    with pytest.raises(ValueError, match="nominal replay artifact differs"):
        verify_deform_dlo3_sensitivity_artifacts_v1(result, protocol)


def test_sensitivity_result_validator_requires_complete_fixed_matrix() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    targets = np.zeros((8, 2, 5, 3), dtype=np.float64)
    baseline = np.full_like(targets, 0.007)
    candidate = np.full_like(targets, 0.006)
    names = [f"case-{index}" for index in range(8)]
    gate = evaluate_deform_dlo3_source_gate(
        candidate, baseline, targets, names, protocol
    )
    labels = (
        "pbd-5",
        "pbd-10",
        "pbd-20",
        "stiffness-0.9",
        "stiffness-1.0",
        "stiffness-1.1",
    )
    result = {
        "contract": "deform-dlo3-physics-solver-sensitivity-result-v1",
        "variants": {label: json.loads(json.dumps(gate)) for label in labels},
        "selection_effect": "none",
        "nominal_replay_exact": True,
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }

    verification = validate_deform_dlo3_sensitivity_result_v1(result, protocol)

    assert verification["verified"] is True
    assert verification["variant_count"] == 6

    missing = json.loads(json.dumps(result))
    del missing["variants"]["pbd-20"]
    with pytest.raises(ValueError, match="sensitivity result differs"):
        validate_deform_dlo3_sensitivity_result_v1(missing, protocol)

    selected = json.loads(json.dumps(result))
    selected["selection_effect"] = "best-source-test"
    with pytest.raises(ValueError, match="sensitivity result differs"):
        validate_deform_dlo3_sensitivity_result_v1(selected, protocol)


def test_backend_result_validator_binds_gate_and_authorization() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    targets = np.zeros((8, 2, 5, 3), dtype=np.float64)
    backend = np.full_like(targets, 0.020)
    candidate = np.full_like(targets, 0.018)
    names = [f"case-{index}" for index in range(8)]
    gate = evaluate_deform_backend_source_gate(
        candidate, backend, targets, names, protocol
    )
    distributions = _seed_result(42)["bayesian_audit"]
    assert isinstance(distributions, dict)
    result = {
        "contract": "deform-dlo3-pyelastica-source-result-v1",
        "source_gate": gate,
        "bayesian_audit": {
            "uncalibrated": distributions["uncalibrated"],
            "calibrated": distributions["calibrated"],
            "point_mean_unchanged_by_calibration": True,
        },
        "backend_target_arm_authorized": True,
        "primary_target_authorized": False,
        "selection_effect": "none-after-fit",
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }

    verification = validate_deform_dlo3_backend_result_v1(result, protocol)

    assert verification["verified"] is True
    assert verification["backend_target_arm_authorized"] is True

    changed = json.loads(json.dumps(result))
    changed["backend_target_arm_authorized"] = False
    with pytest.raises(ValueError, match="target authorization differs"):
        validate_deform_dlo3_backend_result_v1(changed, protocol)


def test_target_gate_reports_unique_and_canonical_reference_operators() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    targets = np.zeros((14, 2, 5, 3), dtype=np.float64)
    baseline = np.full_like(targets, 0.008)
    candidate = np.full_like(targets, 0.007)
    names = [f"case-{index}" for index in range(14)]

    gate = evaluate_deform_dlo3_target_gate(
        candidate, baseline, targets, names, protocol
    )

    assert gate["passed"] is True
    assert gate["candidate_mean_l1_m"] == pytest.approx(0.007)
    assert gate["canonical_reference_draw_mean_l1_m"] == pytest.approx(0.007)
    assert gate["all_unique_below_published_reference"] is True
    assert gate["canonical_draw_below_published_reference"] is True


def test_stability_gate_requires_primary_and_two_of_three_seeds() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    results = [_seed_result(seed) for seed in (42, 43, 44)]

    gate = evaluate_deform_dlo3_stability_gate(results, protocol)

    assert gate["passed"] is True
    assert gate["alltrain_fit_authorized"] is True
    assert gate["target_authorized"] is False
    assert gate["seed_source_passes"] == 3
    assert gate["seed_selection"] is False
    assert gate["bayesian_audit_complete"] is True
    assert gate["bayesian_distribution_count"] == 7

    primary_failed = [_seed_result(seed, passed=seed != 42) for seed in (42, 43, 44)]
    rejected = evaluate_deform_dlo3_stability_gate(primary_failed, protocol)
    assert rejected["seed_source_passes"] == 2
    assert rejected["primary_seed_passed"] is False
    assert rejected["passed"] is False


def test_stability_gate_rejects_instability_or_custody_change() -> None:
    protocol = load_deform_dlo_robustness_v1_protocol(PROTOCOL)
    unstable = [
        _seed_result(42),
        _seed_result(43),
        _seed_result(44, ratio=1.20, passed=False),
    ]

    gate = evaluate_deform_dlo3_stability_gate(unstable, protocol)

    assert gate["maximum_seed_mean_ratio"] == pytest.approx(1.20)
    assert gate["seed_mean_ratio_requirement"] is False
    assert gate["passed"] is False

    changed = [_seed_result(seed) for seed in (42, 43, 44)]
    changed[2]["primary_eval_read"] = True
    with pytest.raises(ValueError, match="custody"):
        evaluate_deform_dlo3_stability_gate(changed, protocol)


def test_stability_runner_cannot_authorize_target() -> None:
    source = STABILITY_RUNNER.read_text(encoding="utf-8")

    assert "evaluate_deform_dlo3_stability_gate" in source
    assert "verify_deform_dlo3_seed_bayesian_artifacts_v1" in source
    assert "verify_deform_dlo3_seed_diagnostic_artifacts_v1" in source
    assert '"diagnostic_artifacts_verified": True' in source
    assert "DLO3" not in source or "/eval" not in source
    assert "target_authorized" not in source


def test_sensitivity_runner_seals_before_scoring_and_never_selects() -> None:
    source = SENSITIVITY_RUNNER.read_text(encoding="utf-8")

    source_open = source.index("trajectories = source_runtime._load_named_trajectories")
    prediction_seal = source.index('seal_path = output_root / "prediction_seal.json"')
    scoring = source.index("scores = {")
    assert source_open < prediction_seal < scoring
    assert '"selection_effect": "none"' in source
    assert '"target_authorized": False' in source
    assert (
        'source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")' in source
    )


def test_pyelastica_runner_seals_fit_and_predictions_before_source_scoring() -> None:
    source = PYELASTICA_RUNNER.read_text(encoding="utf-8")

    method_seal = source.index('method_seal_path = output_root / "method_seal.json"')
    source_open = source.index("source_panel = source_runtime._load_named_trajectories")
    prediction_seal = source.index(
        'prediction_seal_path = output_root / "prediction_seal.json"'
    )
    scoring = source.index("gate = evaluate_deform_backend_source_gate")
    assert method_seal < source_open < prediction_seal < scoring
    assert '"primary_target_authorized": False' in source
    assert '"retry_authorized": False' in source
    assert 'full_model_path = output_root / "full_covariance_model.npz"' in source
    assert 'full_payload["coefficient_covariance_full"]' in source
    assert 'full_payload["residual_covariance_full"]' in source


def test_alltrain_runner_requires_every_source_audit_and_guards_eval() -> None:
    source = ALLTRAIN_RUNNER.read_text(encoding="utf-8")

    assert '"deform-dlo3-training-stability-gate-v1"' in source
    assert '"deform-dlo3-physics-solver-sensitivity-result-v1"' in source
    assert '"deform-dlo3-pyelastica-source-result-v1"' in source
    assert '"deform-dlo3-count-only-custody-deviation-v1"' in source
    assert 'stability.get("bayesian_audit_complete") is not True' in source
    assert 'stability.get("bayesian_artifacts_verified") is not True' in source
    assert 'stability.get("diagnostic_artifacts_verified") is not True' in source
    assert "validate_deform_dlo3_sensitivity_result_v1" in source
    assert "verify_deform_dlo3_seed_diagnostic_artifacts_v1" in source
    assert "verify_deform_dlo3_stability_artifacts_v1" in source
    assert "verify_deform_dlo3_sensitivity_artifacts_v1" in source
    assert "validate_deform_dlo3_alltrain_compute_match_v1" in source
    assert "schedule_updates = registered_updates + maximum_extra" in source
    assert '"selection_effect": "none"' in source
    assert '"target_selection": False' in source
    assert 'sensitivity_artifacts.get("parent_seed_result_sha256")' in source
    assert "validate_deform_dlo3_backend_result_v1" in source
    assert "verify_deform_dlo3_backend_artifacts_v1" in source
    assert '"backend_target_arm": authorization["backend_artifacts"]' in source
    assert (
        'source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")' in source
    )
    assert '"target_authorized": False' in source


def test_evaluator_authorizes_before_target_manifest_and_seals_before_score() -> None:
    source = EVALUATOR_RUNNER.read_text(encoding="utf-8")

    authorization = source.index(
        'authorization_path = output_root / "authorization.json"'
    )
    target_manifest = source.index('stage = "target-manifest"')
    compute_rollout = source.index('stage = "compute-matched-rollout"')
    bayesian_construction = source.index(
        "bayesian_predictions = build_deform_bayesian_covariance_ablation_v1"
    )
    covariance_archive = source.index("deform_bayesian_covariance_archive_key(label):")
    prediction_seal = source.index(
        'prediction_seal_path = output_root / "prediction_seal.json"'
    )
    distribution_scoring = source.index("bayesian_distributions = {")
    target_score = source.index("gate = evaluate_deform_dlo3_target_gate")
    backend_rollout = source.index('stage = "backend-portability-rollout"')
    backend_score = source.index('"report": evaluate_deform_backend_portability_report')
    compute_score = source.index('"report": evaluate_deform_compute_matched_report')
    assert (
        authorization
        < target_manifest
        < compute_rollout
        < bayesian_construction
        < backend_rollout
        < covariance_archive
        < prediction_seal
        < compute_score
        < backend_score
        < distribution_scoring
        < target_score
    )
    assert '"distribution_selection": "none"' in source
    assert '"target_outcomes_used_for_distribution_selection": False' in source
    assert '"retry_authorized": False' in source
    assert '"case_replacement": False' in source
    assert '"backend_target_arm_authorized": backend_authorized' in source
    assert '"status": "technical-failure"' in source
    assert 'readiness.get("compute_matched_control_verified") is not True' in source


def test_readiness_requires_dry_run_and_discloses_count_deviation() -> None:
    source = READINESS_RUNNER.read_text(encoding="utf-8")

    assert '"deform-dlo3-robustness-evaluator-dry-run-v1"' in source
    assert '"deform-dlo3-count-only-custody-deviation-v1"' in source
    assert '"count_only_custody_deviation_acknowledged": True' in source
    assert '"bayesian_audit_complete": True' in source
    assert 'final_method.get("source_diagnostics_verified") is not True' in source
    assert "verify_deform_dlo3_evaluator_bayesian_artifacts_v1" in source
    assert "verify_deform_dlo3_evaluator_compute_matched_artifacts_v1" in source
    assert '"compute_matched_control_verified": True' in source
    assert "verify_deform_dlo3_backend_artifacts_v1" in source
    assert 'expected_backend_status = "scored" if backend_authorized' in source
    assert '"target_authorized": True' in source


def test_seed_runner_seals_models_and_predictions_before_scoring() -> None:
    source = SEED_RUNNER.read_text(encoding="utf-8")

    mechanism_preflight = source.index("calibration_baselines = {")
    method_seal = source.index('method_seal_path = output_root / "method_seal.json"')
    source_open = source.index("source_test_trajectories =")
    bayesian_construction = source.index(
        "bayesian_predictions = build_deform_bayesian_covariance_ablation_v1"
    )
    covariance_archive = source.index("deform_bayesian_covariance_archive_key(label):")
    prediction_seal = source.index(
        'prediction_seal_path = output_root / "prediction_seal.json"'
    )
    distribution_scoring = source.index("bayesian_distributions = {")
    scoring = source.index("primary_gate = evaluate_deform_dlo3_source_gate")
    assert (
        mechanism_preflight
        < method_seal
        < source_open
        < bayesian_construction
        < covariance_archive
        < prediction_seal
        < scoring
        < distribution_scoring
    )
    assert '"distribution_selection": "none"' in source
    assert '"source_test_outcomes_used_for_covariance_construction": False' in source
    assert (
        'source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")' in source
    )
    assert 'source_runtime._install_eval_read_guard(data_root / "DLO4")' in source
    assert 'source_runtime._install_eval_read_guard(data_root / "DLO5")' in source
    assert '"target_authorized": False' in source


def test_method_seal_recovery_lock_is_pending_and_exact() -> None:
    recovery = load_deform_dlo3_method_seal_recovery_v1(RECOVERY_LOCK)

    decision = recovery["decision"]
    assert isinstance(decision, dict)
    assert decision["status"] == "pending"
    assert decision["source_completion_authorized"] is False
    assert decision["permitted_operation"] == "artifact-validation-only"
    assert decision["implementation_source_revision"] is None
    assert decision["implementation_archive_sha256"] is None
    assert decision["seed_44_authorized"] is False
    policy = recovery["recovery_policy"]
    assert isinstance(policy, dict)
    assert policy["eligible_seeds"] == [42, 43]
    assert policy["retraining"] is False
    assert policy["refitting"] is False
    assert policy["checkpoint_continuation"] is False
    assert policy["maximum_completions_per_seed"] == 1


def test_method_seal_completion_authorization_is_exact_and_source_only() -> None:
    authorization = load_deform_dlo3_method_seal_recovery_v1(RECOVERY_AUTHORIZATION)

    decision = authorization["decision"]
    assert isinstance(decision, dict)
    assert decision["status"] == "authorized"
    assert decision["source_completion_authorized"] is True
    assert decision["permitted_operation"] == ("complete-source-from-exact-method-seal")
    assert decision["implementation_source_revision"] == (
        "68feea8ae5852b7713f498cfaba81cdac744a000"
    )
    assert decision["implementation_archive_sha256"] == (
        "111ac9b2c2d74976277a8aba1b52663788e109ec67b796e98e619c83919e56f7"
    )
    assert decision["seed_44_authorized"] is False
    policy = authorization["recovery_policy"]
    assert isinstance(policy, dict)
    assert policy["eligible_seeds"] == [42, 43]
    assert policy["maximum_completions_per_seed"] == 1
    assert policy["retraining"] is False
    assert policy["refitting"] is False
    assert policy["checkpoint_continuation"] is False
    custody = authorization["custody"]
    assert isinstance(custody, dict)
    assert custody["official_eval_read"] is False
    assert custody["dlo4_dlo5_reserve_access"] is False
    assert custody["held_v8_access"] is False
    assert custody["target_retry"] is False


def test_method_seal_recovery_validation_is_target_blind() -> None:
    receipt = json.loads(RECOVERY_VALIDATION.read_text(encoding="utf-8"))

    assert receipt["contract"] == (
        "deform-dlo3-method-seal-recovery-validation-receipt-v1"
    )
    assert receipt["recovery_lock_sha256"] == sha256_file(RECOVERY_LOCK)
    assert set(receipt["seeds"]) == {"42", "43"}
    assert all(record["verified"] is True for record in receipt["seeds"].values())
    assert receipt["source_completion_authorized"] is False
    assert receipt["source_payload_deserialized"] is False
    assert receipt["source_test_opened"] is False
    assert receipt["source_test_scored"] is False
    assert receipt["official_eval_read"] is False
    assert receipt["dlo4_dlo5_reserve_access"] is False
    assert receipt["held_v8_access"] is False


def test_method_seal_recovery_lock_rejects_incoherent_authorization(
    tmp_path: Path,
) -> None:
    payload = json.loads(RECOVERY_LOCK.read_text(encoding="utf-8"))
    payload["decision"]["source_completion_authorized"] = True
    path = tmp_path / "incoherent.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="recovery decision"):
        load_deform_dlo3_method_seal_recovery_v1(path)

    payload["decision"]["status"] = "authorized"
    payload["decision"]["permitted_operation"] = (
        "complete-source-from-exact-method-seal"
    )
    payload["decision"]["implementation_source_revision"] = "1" * 40
    payload["decision"]["implementation_archive_sha256"] = "2" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    authorized = load_deform_dlo3_method_seal_recovery_v1(path)
    assert authorized["decision"]["source_completion_authorized"] is True


def test_method_seal_recovery_runner_cannot_train_or_open_source_early() -> None:
    source = RECOVERY_RUNNER.read_text(encoding="utf-8")

    authorization = source.index(
        'decision.get("source_completion_authorized") is not True'
    )
    output_creation = source.index("output_root.mkdir(parents=True, exist_ok=True)")
    file_verification = source.index("verify_files=True")
    recovery_method_seal = source.index(
        'recovery_method_seal_path = output_root / "recovery_method_seal.json"'
    )
    source_open = source.index("source_test_trajectories =")
    prediction_seal = source.index(
        'prediction_seal_path = output_root / "prediction_seal.json"'
    )
    scoring = source.index("primary_gate = evaluate_deform_dlo3_source_gate")
    assert (
        authorization
        < output_creation
        < file_verification
        < recovery_method_seal
        < source_open
        < prediction_seal
        < scoring
    )
    assert "_train_update" not in source
    assert "fit_deform_local_residual" not in source
    assert "fit_deform_local_residual_variant" not in source
    assert '"retraining": False' in source
    assert '"refitting": False' in source
    assert '"checkpoint_continuation": False' in source
    assert (
        'source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")' in source
    )
    assert 'source_runtime._install_eval_read_guard(data_root / "DLO4")' in source
    assert 'source_runtime._install_eval_read_guard(data_root / "DLO5")' in source
    assert '"retry_authorized": False' in source
    assert '"held_v8_access": False' in source


def test_pending_method_seal_recovery_complete_mode_writes_nothing(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "seed-42-method-seal-completion-v1"
    missing = tmp_path / "intentionally-missing"
    completed = subprocess.run(
        (
            sys.executable,
            str(RECOVERY_RUNNER),
            "--mode",
            "complete",
            "--recovery-lock",
            str(RECOVERY_LOCK),
            "--failure-receipt",
            str(missing),
            "--calibration-smoke",
            str(missing),
            "--protocol",
            str(missing),
            "--source-manifest",
            str(missing),
            "--failed-root",
            str(missing),
            "--failure-log",
            str(missing),
            "--output-root",
            str(output_root),
            "--seed",
            "42",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "source completion is not authorized" in completed.stderr
    assert not output_root.exists()
