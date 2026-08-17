#!/usr/bin/env python3
"""Run the frozen DLO1-only local residual source study."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import run_deform_dlo_action_residual as common_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin_experiments.deform_dlo_action_residual import (
    deform_action_residual_records,
    select_deform_action_residual_arm,
    summarize_deform_action_residual_records,
)
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    deform_causal_inputs,
    fit_deform_local_residual,
    load_deform_local_residual_protocol,
    predict_deform_local_residual,
    serialize_deform_local_residual_model,
)
from bayesian_phystwin_experiments.deform_dlo_longrun import (
    load_deform_dlo_longrun_protocol,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--longrun-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _token(value: float) -> str:
    return format(value, ".8g").replace(".", "p").replace("+", "").replace("-", "m")


def _ridge_name(ridge: float) -> str:
    return f"ridge_{_token(ridge)}"


def _arm_name(ridge: float, shrinkage: float) -> str:
    return f"r{_token(ridge)}_s{_token(shrinkage)}"


def _arm_specs(protocol: dict[str, object]) -> dict[str, dict[str, float]]:
    bank = protocol["candidate_bank"]
    return {
        _arm_name(float(ridge), float(shrinkage)): {
            "ridge": float(ridge),
            "shrinkage": float(shrinkage),
        }
        for ridge in bank["ridges"]
        for shrinkage in bank["shrinkages"]
    }


def _causal_inputs(
    trajectories: dict[str, np.ndarray], names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    full = common_runtime._stack_trajectories(trajectories, names)
    return deform_causal_inputs(full)


def _prediction_diagnostics(
    prediction: dict[str, np.ndarray], targets: object
) -> dict[str, object]:
    predicted = np.asarray(prediction["predictions"], dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    variance = np.asarray(prediction["coordinate_variance_m2"], dtype=np.float64)
    if predicted.shape != target.shape or variance.shape != target.shape:
        raise ValueError("local-residual diagnostic arrays do not align")
    internal_error = predicted[:, :, 2:-2] - target[:, :, 2:-2]
    internal_variance = variance[:, :, 2:-2]
    if np.any(internal_variance <= 0.0):
        raise ValueError("local-residual predictive variance is non-positive")
    standardized_square = np.square(internal_error) / internal_variance
    covered = np.abs(internal_error) <= 1.6448536269514722 * np.sqrt(internal_variance)
    return {
        "correction_l2_m": np.asarray(prediction["correction_l2_m"]).tolist(),
        "mean_coordinate_standard_deviation_m": float(
            np.mean(np.sqrt(internal_variance))
        ),
        "mean_coordinate_nees": float(np.mean(standardized_square)),
        "nominal_90_coordinate_coverage": float(np.mean(covered)),
        "trajectory_mean_nees": np.mean(standardized_square, axis=(1, 2, 3)).tolist(),
        "trajectory_coverage": np.mean(covered, axis=(1, 2, 3)).tolist(),
    }


def main() -> int:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol.resolve()
    protocol = load_deform_local_residual_protocol(protocol_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    longrun_protocol_identity = protocol["longrun_protocol"]
    longrun_protocol_path = common_runtime._identity_path(
        longrun_protocol_identity,
        repository_root=repository_root,
    )
    common_runtime._verify_identity(
        longrun_protocol_path,
        longrun_protocol_identity,
        label="long-run protocol",
    )
    longrun_protocol = load_deform_dlo_longrun_protocol(longrun_protocol_path)
    longrun_result_path = args.longrun_result.resolve()
    common_runtime._verify_identity(
        longrun_result_path,
        protocol["longrun_result"],
        label="long-run result",
    )
    longrun_result = common_runtime._read_json(longrun_result_path)
    posterior_runtime._validate_longrun_result(
        longrun_result,
        protocol_sha256=sha256_file(longrun_protocol_path),
    )
    selected = longrun_result["selected_checkpoint"]
    baseline = protocol["baseline"]
    if (
        int(selected.get("update", -1)) != int(baseline["selected_update"])
        or selected.get("checkpoint", {}).get("sha256") != baseline["checkpoint_sha256"]
    ):
        raise ValueError("local-residual baseline differs from locked checkpoint")

    source_manifest_path = args.source_manifest.resolve()
    common_runtime._verify_identity(
        source_manifest_path,
        protocol["source_manifest"],
        label="source manifest",
    )
    manifest = common_runtime._read_json(source_manifest_path)
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO1"
        or manifest.get("partition") != "train"
        or manifest.get("official_eval_read") is not False
    ):
        raise ValueError("local-residual source manifest is invalid")

    source_runtime._assert_upstream(
        args.upstream_root,
        longrun_result["upstream"]["commit"],
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO2")
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo-local-residual-preflight-v4",
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "longrun_protocol": {
            "path": str(longrun_protocol_path),
            "sha256": sha256_file(longrun_protocol_path),
        },
        "longrun_result": {
            "path": str(longrun_result_path),
            "sha256": sha256_file(longrun_result_path),
        },
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
        },
        "selected_checkpoint": selected["checkpoint"],
        "dlo1_source_test_opened": False,
        "dlo2_read": False,
        "official_eval_read": False,
    }
    common_runtime._write_json(output_root / "preflight.json", preflight)

    cublas_config = str(longrun_protocol["training"]["cublas_workspace_config"])
    existing_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas_config not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    started = time.perf_counter()
    source_runtime._seed_everything(torch, 42)
    modules = source_runtime._load_upstream(args.upstream_root)
    state = posterior_runtime._checkpoint_states(
        longrun_result,
        {int(baseline["selected_update"])},
        torch=torch,
    )[int(baseline["selected_update"])]
    fit_names = list(manifest["split"]["fit"])
    validation_names = list(manifest["split"]["validation"])
    source_names = list(manifest["split"]["source_test"])
    development_names = fit_names + validation_names
    development = source_runtime._load_named_trajectories(
        manifest,
        development_names,
        frame_count=500,
        node_count=13,
    )
    development_rollout = common_runtime._rollout(
        state,
        development,
        modules=modules,
        torch=torch,
        device=args.device,
    )
    fit_rollout = common_runtime._split_rollout(development_rollout, 0, len(fit_names))
    validation_rollout = common_runtime._split_rollout(
        development_rollout,
        len(fit_names),
        len(development_names),
    )
    reproduction_tolerance = float(baseline["reproduction_tolerance_m"])
    validation_baseline_l1_m = common_runtime._mean_l1(
        validation_rollout["predictions"], validation_rollout["targets"]
    )
    common_runtime._require_baseline_reproduction(
        validation_baseline_l1_m,
        expected=float(baseline["validation_l1_m"]),
        tolerance=reproduction_tolerance,
        stage="validation",
    )

    fit_initial, fit_action = _causal_inputs(development, fit_names)
    validation_initial, validation_action = _causal_inputs(
        development, validation_names
    )
    floor = float(protocol["posterior"]["coordinate_variance_floor_m2"])
    bank = protocol["candidate_bank"]
    models: dict[float, dict[str, object]] = {}
    model_identities = {}
    model_root = output_root / "models"
    for ridge_value in bank["ridges"]:
        ridge = float(ridge_value)
        model = fit_deform_local_residual(
            fit_initial,
            fit_action,
            np.asarray(fit_rollout["predictions"]),
            np.asarray(fit_rollout["targets"]),
            fit_names,
            ridge=ridge,
            variance_floor_m2=floor,
        )
        model_path = model_root / f"{_ridge_name(ridge)}.npz"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            model_path,
            **serialize_deform_local_residual_model(model),
        )
        models[ridge] = model
        model_identities[_ridge_name(ridge)] = {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        }

    specs = _arm_specs(protocol)
    validation_records = {}
    validation_diagnostics = {}
    for name, spec in specs.items():
        prediction = predict_deform_local_residual(
            models[spec["ridge"]],
            validation_initial,
            validation_action,
            np.asarray(validation_rollout["predictions"]),
            shrinkage=spec["shrinkage"],
        )
        validation_records[name] = deform_action_residual_records(
            prediction["predictions"],
            validation_rollout["targets"],
            validation_rollout["predictions"],
            validation_names,
        )
        validation_diagnostics[name] = _prediction_diagnostics(
            prediction, validation_rollout["targets"]
        )
    validation_gate = protocol["gates"]["validation"]
    selection = select_deform_action_residual_arm(
        validation_records,
        minimum_relative_improvement=float(
            validation_gate["minimum_relative_improvement"]
        ),
        minimum_case_wins=int(validation_gate["minimum_case_wins"]),
        maximum_case_ratio=float(validation_gate["maximum_case_ratio"]),
    )
    selected_name = str(selection["selected_arm"])
    selection_seal = {
        "schema_version": 1,
        "contract": "deform-dlo-local-residual-validation-selection-v4",
        "protocol_sha256": sha256_file(protocol_path),
        "longrun_result_sha256": sha256_file(longrun_result_path),
        "model_identities": model_identities,
        "selected_arm": selected_name,
        "selected_spec": None if selection["fallback_used"] else specs[selected_name],
        "selected_diagnostics": (
            None
            if selection["fallback_used"]
            else validation_diagnostics[selected_name]
        ),
        "fallback_used": bool(selection["fallback_used"]),
        "validation": selection,
        "validation_baseline_l1_m": validation_baseline_l1_m,
        "source_test_opened": False,
        "dlo2_read": False,
        "official_eval_read": False,
    }
    selection_path = output_root / "validation_selection_seal.json"
    common_runtime._write_json(selection_path, selection_seal)

    if bool(selection["fallback_used"]):
        result = {
            "schema_version": 1,
            "contract": "deform-dlo-local-residual-result-v4",
            "claim_boundary": protocol["claim_boundary"],
            "protocol_sha256": sha256_file(protocol_path),
            "validation_selection_sha256": sha256_file(selection_path),
            "selected_arm": "baseline_exact",
            "validation": selection,
            "validation_baseline_l1_m": validation_baseline_l1_m,
            "source_test_opened": False,
            "source_gate": {"passed": False, "reason": "validation-gate-failed"},
            "fresh_dlo2_local_residual_authorized": False,
            "dlo2_read": False,
            "official_eval_read": False,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "elapsed_seconds": time.perf_counter() - started,
            },
        }
        common_runtime._write_json(output_root / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    source_trajectories = source_runtime._load_named_trajectories(
        manifest,
        source_names,
        frame_count=500,
        node_count=13,
    )
    source_rollout = common_runtime._rollout(
        state,
        source_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
    )
    source_baseline_l1_m = common_runtime._mean_l1(
        source_rollout["predictions"], source_rollout["targets"]
    )
    common_runtime._require_baseline_reproduction(
        source_baseline_l1_m,
        expected=float(baseline["source_test_l1_m"]),
        tolerance=reproduction_tolerance,
        stage="source-test",
    )
    selected_spec = specs[selected_name]
    source_initial, source_action = _causal_inputs(source_trajectories, source_names)
    source_prediction = predict_deform_local_residual(
        models[selected_spec["ridge"]],
        source_initial,
        source_action,
        np.asarray(source_rollout["predictions"]),
        shrinkage=selected_spec["shrinkage"],
    )
    source_records = deform_action_residual_records(
        source_prediction["predictions"],
        source_rollout["targets"],
        source_rollout["predictions"],
        source_names,
    )
    source_summary = summarize_deform_action_residual_records(source_records)
    source_gate_config = protocol["gates"]["source_test"]
    source_gate = {
        **source_summary,
        "minimum_relative_improvement": float(
            source_gate_config["minimum_relative_improvement"]
        ),
        "minimum_case_wins": int(source_gate_config["minimum_case_wins"]),
        "maximum_allowed_case_ratio": float(source_gate_config["maximum_case_ratio"]),
        "maximum_candidate_l1_m": float(source_gate_config["maximum_candidate_l1_m"]),
    }
    source_gate["passed"] = bool(
        float(source_summary["relative_improvement"])
        >= source_gate["minimum_relative_improvement"]
        and int(source_summary["wins"]) >= source_gate["minimum_case_wins"]
        and float(source_summary["maximum_case_ratio"])
        <= source_gate["maximum_allowed_case_ratio"]
        and float(source_summary["candidate_mean_l1_m"])
        <= source_gate["maximum_candidate_l1_m"]
    )
    source_prediction_path = output_root / "source_prediction.npz"
    np.savez_compressed(
        source_prediction_path,
        names=np.asarray(source_names),
        baseline_predictions=np.asarray(source_rollout["predictions"]),
        candidate_predictions=np.asarray(source_prediction["predictions"]),
        coordinate_variance_m2=np.asarray(source_prediction["coordinate_variance_m2"]),
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo-local-residual-result-v4",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": sha256_file(protocol_path),
        "validation_selection_sha256": sha256_file(selection_path),
        "source_prediction_sha256": sha256_file(source_prediction_path),
        "selected_arm": selected_name,
        "selected_spec": selected_spec,
        "selected_model": model_identities[_ridge_name(selected_spec["ridge"])],
        "validation": selection,
        "validation_baseline_l1_m": validation_baseline_l1_m,
        "source_test_opened": True,
        "source_baseline_l1_m": source_baseline_l1_m,
        "source_records": source_records,
        "source_diagnostics": _prediction_diagnostics(
            source_prediction, source_rollout["targets"]
        ),
        "source_gate": source_gate,
        "fresh_dlo2_local_residual_authorized": bool(source_gate["passed"]),
        "dlo2_read": False,
        "official_eval_read": False,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    common_runtime._write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
