#!/usr/bin/env python3
"""Run the sealed DLO2 shrinkage-0.25 source transfer once."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import run_deform_dlo2_local_residual as v5_runtime
import run_deform_dlo_action_residual as common_runtime
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.experiments.deform_dlo_action_residual import (
    deform_action_residual_records,
    summarize_deform_action_residual_records,
)
from bayesian_phystwin.experiments.deform_dlo_local_residual import (
    fit_deform_local_residual,
    load_deform_dlo2_local_residual_protocol,
    load_deform_dlo2_local_residual_v6_protocol,
    predict_deform_local_residual,
    serialize_deform_local_residual_model,
    validate_deform_dlo2_local_residual_v6_parents,
)
from bayesian_phystwin.experiments.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--development-selection", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("preflight", "evaluate"), default="evaluate")
    return parser.parse_args()


def _verify_identity(
    path: Path,
    identity: dict[str, object],
    *,
    label: str,
) -> None:
    if not path.is_file() or sha256_file(path) != identity["sha256"]:
        raise ValueError(f"{label} identity does not verify")


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo2_local_residual_v6_protocol(protocol_path)
    parent_protocol_path = args.parent_protocol.resolve()
    parent_result_path = args.parent_result.resolve()
    development_path = args.development_selection.resolve()
    training_path = args.training_result.resolve()
    _verify_identity(
        parent_protocol_path,
        protocol["parent_protocol"],
        label="v5 parent protocol",
    )
    _verify_identity(
        parent_result_path,
        protocol["parent_result"],
        label="v5 parent result",
    )
    _verify_identity(
        development_path,
        protocol["development_selection"],
        label="v6 development selection",
    )
    _verify_identity(
        training_path,
        protocol["training_result"],
        label="DLO2 training result",
    )
    parent_result = common_runtime._read_json(parent_result_path)
    development = common_runtime._read_json(development_path)
    parent_summary = validate_deform_dlo2_local_residual_v6_parents(
        protocol,
        parent_result,
        development,
    )
    parent_protocol = load_deform_dlo2_local_residual_protocol(parent_protocol_path)
    training, manifest, manifest_path = v5_runtime._verify_training_result(
        training_path,
        protocol=parent_protocol,
        protocol_path=parent_protocol_path,
    )
    _verify_identity(
        manifest_path,
        protocol["source_manifest"],
        label="DLO2 source manifest",
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO2" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    source_runtime._assert_upstream(args.upstream_root, protocol["upstream"]["commit"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        parent_protocol["training"]["cublas_workspace_config"]
    )
    import torch

    started = time.perf_counter()
    source_runtime._seed_everything(torch, 42)
    modules = source_runtime._load_upstream(args.upstream_root)
    state, checkpoint_path = v5_runtime._checkpoint_state(training, torch=torch)
    _verify_identity(
        checkpoint_path,
        protocol["selected_checkpoint"],
        label="DLO2 selected checkpoint",
    )
    fit_names = list(manifest["split"]["fit"])
    fit_trajectories = source_runtime._load_named_trajectories(
        manifest,
        fit_names,
        frame_count=500,
        node_count=12,
    )
    fit_rollout = v5_runtime._rollout(
        state,
        fit_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
    )
    fit_initial, fit_action = local_runtime._causal_inputs(
        fit_trajectories,
        fit_names,
    )
    local = protocol["local_residual"]
    fixed = local["fixed_arm"]
    model = fit_deform_local_residual(
        fit_initial,
        fit_action,
        np.asarray(fit_rollout["predictions"]),
        np.asarray(fit_rollout["targets"]),
        fit_names,
        ridge=float(fixed["ridge"]),
        variance_floor_m2=float(local["coordinate_variance_floor_m2"]),
    )
    model_path = output_root / "local_residual_model.npz"
    np.savez_compressed(model_path, **serialize_deform_local_residual_model(model))
    model_sha256 = sha256_file(model_path)
    if model_sha256 != local["expected_fitted_model_sha256"]:
        raise ValueError("DLO2 fitted local-residual model differs from v5")

    runner_path = Path(__file__).resolve()
    module_path = (
        runner_path.parents[2]
        / "src"
        / "bayesian_phystwin"
        / "experiments"
        / "deform_dlo_local_residual.py"
    )
    source_opening = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-source-opening-v6",
        "mode": str(args.mode),
        "protocol_sha256": sha256_file(protocol_path),
        "parent_authorization": parent_summary,
        "parent_protocol_sha256": sha256_file(parent_protocol_path),
        "parent_result_sha256": sha256_file(parent_result_path),
        "development_selection_sha256": sha256_file(development_path),
        "training_result_sha256": sha256_file(training_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "fitted_model_sha256": model_sha256,
        "fixed_arm": fixed,
        "implementation": {
            "runner_sha256": sha256_file(runner_path),
            "local_residual_module_sha256": sha256_file(module_path),
        },
        "source_trajectory_count": int(
            parent_protocol["source_split"]["source_test_count"]
        ),
        "source_test_opened": False,
        "official_eval_read": False,
    }
    source_opening_path = output_root / "source_opening_seal.json"
    common_runtime._write_json(source_opening_path, source_opening)
    if args.mode == "preflight":
        result = {
            "schema_version": 1,
            "contract": "deform-dlo2-local-residual-preflight-v6",
            "protocol_sha256": sha256_file(protocol_path),
            "source_opening_sha256": sha256_file(source_opening_path),
            "fitted_model_sha256": model_sha256,
            "fixed_arm": fixed,
            "source_ready": True,
            "source_test_opened": False,
            "official_eval_read": False,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "elapsed_seconds": time.perf_counter() - started,
            },
        }
        common_runtime._write_json(output_root / "preflight_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    source_names = list(manifest["split"]["source_test"])
    source_trajectories = source_runtime._load_named_trajectories(
        manifest,
        source_names,
        frame_count=500,
        node_count=12,
    )
    source_rollout = v5_runtime._rollout(
        state,
        source_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
    )
    source_initial, source_action = local_runtime._causal_inputs(
        source_trajectories,
        source_names,
    )
    source_prediction = predict_deform_local_residual(
        model,
        source_initial,
        source_action,
        np.asarray(source_rollout["predictions"]),
        shrinkage=float(fixed["shrinkage"]),
    )
    source_records = deform_action_residual_records(
        source_prediction["predictions"],
        source_rollout["targets"],
        source_rollout["predictions"],
        source_names,
    )
    source_summary = summarize_deform_action_residual_records(source_records)
    source_gate = v5_runtime._gate(
        source_summary,
        protocol["source_transfer_gate"],
        require_published_reference=True,
    )
    passed = bool(source_gate["passed"])
    baseline_predictions = np.asarray(source_rollout["predictions"])
    candidate_predictions = np.asarray(source_prediction["predictions"])
    selected_predictions = (
        candidate_predictions.copy() if passed else baseline_predictions.copy()
    )
    fallback_byte_exact = bool(
        passed or selected_predictions.tobytes() == baseline_predictions.tobytes()
    )
    if not fallback_byte_exact:
        raise RuntimeError("rejected DLO2 correction did not preserve exact fallback")
    prediction_path = output_root / "source_prediction.npz"
    np.savez_compressed(
        prediction_path,
        names=np.asarray(source_names),
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        selected_predictions=selected_predictions,
        coordinate_variance_m2=np.asarray(source_prediction["coordinate_variance_m2"]),
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-result-v6",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": sha256_file(protocol_path),
        "source_opening_sha256": sha256_file(source_opening_path),
        "source_prediction_sha256": sha256_file(prediction_path),
        "fitted_model_sha256": model_sha256,
        "fixed_arm": fixed,
        "source_test_opened": True,
        "source_records": source_records,
        "source_gate": source_gate,
        "source_diagnostics": local_runtime._prediction_diagnostics(
            source_prediction,
            source_rollout["targets"],
        ),
        "fallback_used": not passed,
        "fallback_byte_exact": fallback_byte_exact,
        "selected_method": (
            "dlo2-baseline-plus-validation-selected-local-residual"
            if passed
            else "selected-dlo2-checkpoint-exact"
        ),
        "alltrain_and_official_evaluation_authorized": passed,
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
