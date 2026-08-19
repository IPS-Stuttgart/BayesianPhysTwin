#!/usr/bin/env python3
"""Run the frozen PyElastica portability gate on DLO3 train partitions."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    fit_deform_local_residual,
    serialize_deform_local_residual_model,
)
from bayesian_phystwin_experiments.deform_dlo_pyelastica import (
    PyElasticaParameters,
    deform_pyelastica_parameter_bank,
    simulate_deform_pyelastica,
)
from bayesian_phystwin_experiments.deform_dlo_robustness import (
    augment_deform_local_residual_full_covariance,
    calibrate_deform_full_covariance,
    evaluate_deform_backend_source_gate,
    evaluate_deform_predictive_distribution,
    load_deform_dlo_robustness_v1_protocol,
    predict_deform_local_residual_full_covariance,
    scale_deform_coordinate_covariance,
    validate_deform_dlo3_source_manifest,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

Array = np.ndarray[Any, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--pyelastica-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "run"), default="run")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(
    path: Path, payload: dict[str, object], *, immutable: bool = True
) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and immutable:
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked PyElastica output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _load_pyelastica(root: Path, expected_commit: str, expected_version: str) -> Any:
    source = root.resolve()
    if not (source / "elastica" / "__init__.py").is_file():
        raise FileNotFoundError("PyElastica source package is missing")
    completed = subprocess.run(
        ("git", "-C", str(source), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip() != expected_commit:
        raise ValueError("PyElastica commit differs")
    status = subprocess.run(
        ("git", "-C", str(source), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise ValueError("PyElastica checkout is dirty")
    sys.path.insert(0, str(source))
    elastica = importlib.import_module("elastica")
    version = importlib.import_module("elastica.version")
    if str(getattr(version, "VERSION", "")) != expected_version:
        raise ValueError("PyElastica version differs")
    return elastica


def _stack_targets(trajectories: Mapping[str, Array], names: list[str]) -> Array:
    return cast(
        Array,
        np.stack([np.asarray(trajectories[name][2:]) for name in names]),
    )


def _simulate_panel(
    trajectories: Mapping[str, Array],
    names: list[str],
    parameters: PyElasticaParameters,
    *,
    elastica: Any,
    poisson_ratio: float,
    radius_ratio: float,
) -> Array:
    return cast(
        Array,
        np.stack(
            [
                simulate_deform_pyelastica(
                    trajectories[name],
                    parameters,
                    elastica=elastica,
                    poisson_ratio=poisson_ratio,
                    radius_to_mean_edge_ratio=radius_ratio,
                )
                for name in names
            ]
        ),
    )


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    manifest_path = args.source_manifest.resolve()
    protocol = load_deform_dlo_robustness_v1_protocol(protocol_path)
    manifest = _read_json(manifest_path)
    partitions = validate_deform_dlo3_source_manifest(
        manifest,
        protocol,
        protocol_sha256=sha256_file(protocol_path),
        verify_files=True,
    )
    backend = _mapping(protocol.get("backend_portability"), label="backend")
    raw_bank = _mapping(backend.get("parameter_bank"), label="parameter bank")
    elastica = _load_pyelastica(
        args.pyelastica_root,
        str(backend["commit"]),
        str(backend["version"]),
    )
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"PyElastica output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    upstream = source_runtime._assert_upstream(
        args.upstream_root,
        str(_mapping(protocol["upstream"], label="upstream")["commit"]),
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO3" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO4")
    source_runtime._install_eval_read_guard(data_root / "DLO5")
    bank = deform_pyelastica_parameter_bank(protocol)
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-preflight-v1",
        "mode": args.mode,
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "upstream": upstream,
        "pyelastica": {
            "root": str(args.pyelastica_root.resolve()),
            "commit": backend["commit"],
            "version": backend["version"],
        },
        "bank_size": len(bank),
        "source_test_opened": False,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "target_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    _write_json(output_root / "preflight.json", preflight)
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0

    fit_names = list(partitions["fit"])
    calibration_names = list(partitions["calibration"])
    development_names = fit_names + calibration_names
    development = source_runtime._load_named_trajectories(
        manifest,
        development_names,
        frame_count=500,
        node_count=12,
    )
    poisson_ratio = float(cast(Any, raw_bank["poisson_ratio"]))
    radius_ratio = float(cast(Any, raw_bank["radius_to_mean_edge_ratio"]))
    if args.mode == "smoke":
        shortened = np.asarray(development[fit_names[0]][:6])
        started = time.perf_counter()
        prediction = simulate_deform_pyelastica(
            shortened,
            bank[0],
            elastica=elastica,
            poisson_ratio=poisson_ratio,
            radius_to_mean_edge_ratio=radius_ratio,
        )
        smoke = {
            "schema_version": 1,
            "contract": "deform-dlo3-pyelastica-smoke-v1",
            "parameter": bank[0].to_record(),
            "prediction_shape": list(prediction.shape),
            "elapsed_seconds": time.perf_counter() - started,
            "source_test_opened": False,
            "primary_eval_read": False,
            "target_authorized": False,
        }
        _write_json(output_root / "smoke_result.json", smoke)
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0

    fit = {name: development[name] for name in fit_names}
    calibration_panel = {name: development[name] for name in calibration_names}
    fit_targets = _stack_targets(fit, fit_names)
    bank_records: list[dict[str, object]] = []
    selected_index = -1
    selected_error = float("inf")
    selected_fit_predictions: Array | None = None
    bank_started = time.perf_counter()
    for index, parameters in enumerate(bank):
        prediction = _simulate_panel(
            fit,
            fit_names,
            parameters,
            elastica=elastica,
            poisson_ratio=poisson_ratio,
            radius_ratio=radius_ratio,
        )
        error = float(np.mean(np.abs(prediction - fit_targets)))
        if not np.isfinite(error):
            raise RuntimeError("PyElastica bank member produced an invalid fit error")
        bank_records.append(
            {
                "index": index,
                "parameters": parameters.to_record(),
                "fit_mean_l1_m": error,
            }
        )
        if error < selected_error:
            selected_index = index
            selected_error = error
            selected_fit_predictions = prediction
        progress: dict[str, object] = {
            "completed_bank_members": index + 1,
            "bank_size": len(bank),
            "elapsed_seconds": time.perf_counter() - bank_started,
            "source_test_opened": False,
            "primary_eval_read": False,
        }
        _write_json(output_root / "progress.json", progress, immutable=False)
        print(json.dumps(progress, sort_keys=True), flush=True)
    if selected_index < 0 or selected_fit_predictions is None:
        raise RuntimeError("PyElastica bank did not select a finite member")
    selected = bank[selected_index]
    selection = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-fit-selection-v1",
        "selection_partition": "fit-only",
        "selected_index": selected_index,
        "selected_parameters": selected.to_record(),
        "selected_fit_mean_l1_m": selected_error,
        "tie_break": "first-frozen-bank-index",
        "members": bank_records,
        "source_test_opened": False,
        "primary_eval_read": False,
    }
    selection_path = output_root / "fit_selection.json"
    _write_json(selection_path, selection)

    fit_initial, fit_action = local_runtime._causal_inputs(fit, fit_names)
    residual = _mapping(protocol.get("local_residual"), label="local residual")
    local_model = fit_deform_local_residual(
        fit_initial,
        fit_action,
        selected_fit_predictions,
        fit_targets,
        fit_names,
        ridge=float(cast(Any, residual["ridge"])),
        variance_floor_m2=float(cast(Any, residual["coordinate_variance_floor_m2"])),
    )
    full_model = augment_deform_local_residual_full_covariance(
        local_model,
        fit_initial,
        fit_action,
        selected_fit_predictions,
        fit_targets,
        fit_names,
    )
    full_model_path = output_root / "full_covariance_model.npz"
    full_payload = serialize_deform_local_residual_model(full_model)
    full_payload["coefficient_covariance_full"] = np.asarray(
        full_model["coefficient_covariance_full"]
    )
    full_payload["residual_covariance_full"] = np.asarray(
        full_model["residual_covariance_full"]
    )
    np.savez_compressed(full_model_path, **full_payload)
    calibration_baseline = _simulate_panel(
        calibration_panel,
        calibration_names,
        selected,
        elastica=elastica,
        poisson_ratio=poisson_ratio,
        radius_ratio=radius_ratio,
    )
    calibration_targets = _stack_targets(calibration_panel, calibration_names)
    calibration_initial, calibration_action = local_runtime._causal_inputs(
        calibration_panel, calibration_names
    )
    calibration_prediction = predict_deform_local_residual_full_covariance(
        full_model,
        calibration_initial,
        calibration_action,
        calibration_baseline,
        shrinkage=float(cast(Any, residual["shrinkage"])),
    )
    calibration = calibrate_deform_full_covariance(
        calibration_prediction["predictions"],
        calibration_targets,
        calibration_prediction["coordinate_covariance_m2"],
    )
    calibration_record = {
        **calibration,
        "trajectory_scores": [
            float(value) for value in np.asarray(calibration["trajectory_scores"])
        ],
        "source_test_opened": False,
        "primary_eval_read": False,
    }
    calibration_path = output_root / "covariance_calibration.json"
    _write_json(calibration_path, calibration_record)
    method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-method-seal-v1",
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "fit_selection": _identity(selection_path),
        "full_covariance_model": _identity(full_model_path),
        "covariance_calibration": _identity(calibration_path),
        "selected_parameters": selected.to_record(),
        "ridge": float(cast(Any, residual["ridge"])),
        "shrinkage": float(cast(Any, residual["shrinkage"])),
        "source_test_opened": False,
        "primary_eval_read": False,
        "selection_effect_after_fit": "none",
    }
    method_seal_path = output_root / "method_seal.json"
    _write_json(method_seal_path, method_seal)

    source_names = list(partitions["source_test"])
    source_panel = source_runtime._load_named_trajectories(
        manifest,
        source_names,
        frame_count=500,
        node_count=12,
    )
    source_baseline = _simulate_panel(
        source_panel,
        source_names,
        selected,
        elastica=elastica,
        poisson_ratio=poisson_ratio,
        radius_ratio=radius_ratio,
    )
    source_initial, source_action = local_runtime._causal_inputs(
        source_panel, source_names
    )
    source_prediction = predict_deform_local_residual_full_covariance(
        full_model,
        source_initial,
        source_action,
        source_baseline,
        shrinkage=float(cast(Any, residual["shrinkage"])),
    )
    calibrated_covariance = scale_deform_coordinate_covariance(
        source_prediction["coordinate_covariance_m2"],
        float(cast(Any, calibration["variance_scale"])),
    )
    predictions_path = output_root / "source_predictions.npz"
    np.savez_compressed(
        predictions_path,
        names=np.asarray(source_names),
        backend=source_baseline,
        candidate=source_prediction["predictions"],
        coordinate_covariance_m2=source_prediction["coordinate_covariance_m2"],
        calibrated_coordinate_covariance_m2=calibrated_covariance,
    )
    prediction_seal = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-prediction-seal-v1",
        "method_seal": _identity(method_seal_path),
        "predictions": _identity(predictions_path),
        "source_outcomes_scored": False,
        "primary_eval_read": False,
    }
    prediction_seal_path = output_root / "prediction_seal.json"
    _write_json(prediction_seal_path, prediction_seal)

    source_targets = _stack_targets(source_panel, source_names)
    gate = evaluate_deform_backend_source_gate(
        source_prediction["predictions"],
        source_baseline,
        source_targets,
        source_names,
        protocol,
    )
    raw_distribution = evaluate_deform_predictive_distribution(
        source_prediction["predictions"],
        source_targets,
        source_prediction["coordinate_covariance_m2"],
    )
    calibrated_distribution = evaluate_deform_predictive_distribution(
        source_prediction["predictions"],
        source_targets,
        calibrated_covariance,
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo3-pyelastica-source-result-v1",
        "claim_boundary": "DLO3 train source partitions only; official evaluation unopened.",
        "protocol": _identity(protocol_path),
        "source_manifest": _identity(manifest_path),
        "method_seal": _identity(method_seal_path),
        "prediction_seal": _identity(prediction_seal_path),
        "source_gate": gate,
        "bayesian_audit": {
            "calibration": calibration_record,
            "uncalibrated": raw_distribution,
            "calibrated": calibrated_distribution,
            "point_mean_unchanged_by_calibration": True,
        },
        "backend_target_arm_authorized": gate["passed"],
        "primary_target_authorized": False,
        "selection_effect": "none-after-fit",
        "source_test_opened": True,
        "primary_eval_enumerated": False,
        "primary_eval_read": False,
        "retry_authorized": False,
        "prob4d_used": False,
        "held_v8_access": False,
    }
    result_path = output_root / "source_result.json"
    _write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
