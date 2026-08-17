#!/usr/bin/env python3
"""Run fixed-arm fresh DLO2 local-residual transfer without official eval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_dlo_action_residual as common_runtime
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_action_residual import (
    deform_action_residual_records,
    summarize_deform_action_residual_records,
)
from bayesian_phystwin.deform_dlo_local_residual import (
    fit_deform_local_residual,
    load_deform_dlo2_local_residual_protocol,
    predict_deform_local_residual,
    serialize_deform_local_residual_model,
    validate_deform_dlo2_local_residual_parent,
)
from bayesian_phystwin.deform_dlo_source import (
    sha256_file,
    validate_deform_dlo2_stage_authorization,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-result", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode",
        choices=("preflight", "smoke", "train-validation", "evaluate"),
        default="evaluate",
    )
    return parser.parse_args()


def _parent_authorization(
    protocol: dict[str, object],
    parent_path: Path,
) -> dict[str, object]:
    expected = protocol["local_residual"]["parent_result"]
    common_runtime._verify_identity(
        parent_path,
        expected,
        label="DLO1 local-residual result",
    )
    parent = common_runtime._read_json(parent_path)
    return {
        "path": str(parent_path),
        "sha256": sha256_file(parent_path),
        **validate_deform_dlo2_local_residual_parent(protocol, parent),
    }


def _write_authorization(
    output_root: Path,
    *,
    mode: str,
    protocol_path: Path,
    parent: dict[str, object],
) -> Path:
    authorization = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-authorization-v1",
        "mode": mode,
        "official_eval_read": False,
        "source_test_opened": False,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "parent_local_residual_result": parent,
    }
    path = output_root / "authorization.json"
    common_runtime._write_json(path, authorization)
    return path


def _run_training_stage(
    args: argparse.Namespace,
    *,
    authorization_path: Path,
) -> int:
    mode = str(args.mode)
    runner_mode = mode
    runner = Path(__file__).resolve().with_name("run_deform_dlo_source.py")
    command = [
        sys.executable,
        str(runner),
        "--protocol",
        str(args.protocol.resolve()),
        "--upstream-root",
        str(args.upstream_root.resolve()),
        "--output-root",
        str(args.output_root.resolve() / "training_run"),
        "--stage-authorization",
        str(authorization_path),
        "--dlo-type",
        "DLO2",
        "--device",
        args.device,
        "--mode",
        runner_mode,
    ]
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(command, env=environment, check=False)
    return int(completed.returncode)


def _verify_training_result(
    path: Path,
    *,
    protocol: dict[str, object],
    protocol_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path]:
    result = common_runtime._read_json(path)
    if (
        result.get("contract") != "deform-dlo-training-validation-result-v1"
        or result.get("source_test_opened") is not False
        or result.get("official_eval_read") is not False
    ):
        raise ValueError("DLO2 training result crossed its source boundary")
    stage_identity = result.get("stage_authorization")
    if not isinstance(stage_identity, dict):
        raise ValueError("DLO2 training result omits stage authorization")
    authorization_path = Path(str(stage_identity.get("path", ""))).resolve()
    if not authorization_path.is_file() or sha256_file(
        authorization_path
    ) != stage_identity.get("sha256"):
        raise ValueError("DLO2 training stage authorization does not verify")
    authorization = common_runtime._read_json(authorization_path)
    validate_deform_dlo2_stage_authorization(
        protocol,
        authorization,
        protocol_sha256=sha256_file(protocol_path),
    )
    manifest_identity = result.get("source_manifest")
    if not isinstance(manifest_identity, dict):
        raise ValueError("DLO2 training result omits its source manifest")
    manifest_path = Path(str(manifest_identity.get("path", ""))).resolve()
    if not manifest_path.is_file() or sha256_file(
        manifest_path
    ) != manifest_identity.get("sha256"):
        raise ValueError("DLO2 source manifest does not verify")
    manifest = common_runtime._read_json(manifest_path)
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO2"
        or manifest.get("partition") != "train"
        or manifest.get("official_eval_read") is not False
        or not isinstance(manifest.get("protocol"), dict)
        or manifest["protocol"].get("sha256") != sha256_file(protocol_path)
    ):
        raise ValueError("DLO2 source manifest differs from the frozen protocol")
    return result, manifest, manifest_path


def _checkpoint_state(result: dict[str, object], *, torch: Any) -> tuple[dict, Path]:
    selected = result.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise ValueError("DLO2 training result omits its selected checkpoint")
    identity = selected.get("checkpoint")
    if not isinstance(identity, dict):
        raise ValueError("DLO2 selected checkpoint identity is invalid")
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file() or sha256_file(path) != identity.get("sha256"):
        raise ValueError("DLO2 selected checkpoint does not verify")
    bundle = torch.load(path, map_location="cpu")
    state = bundle.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("DLO2 selected checkpoint omits model state")
    return state, path


def _rollout(
    state: dict[str, Any],
    trajectories: dict[str, np.ndarray],
    *,
    modules: Any,
    torch: Any,
    device: str,
) -> dict[str, object]:
    return posterior_runtime._evaluate_state(
        state,
        trajectories,
        modules=modules,
        torch=torch,
        device=device,
        dlo_type="DLO2",
        node_count=12,
    )


def _gate(
    summary: dict[str, object],
    config: dict[str, object],
    *,
    require_published_reference: bool,
) -> dict[str, object]:
    gate = {
        **summary,
        "minimum_relative_improvement": float(config["minimum_relative_improvement"]),
        "minimum_case_wins": int(config["minimum_case_wins"]),
        "maximum_allowed_case_ratio": float(config["maximum_case_ratio"]),
    }
    passed = bool(
        float(summary["relative_improvement"]) >= gate["minimum_relative_improvement"]
        and int(summary["wins"]) >= gate["minimum_case_wins"]
        and float(summary["maximum_case_ratio"]) <= gate["maximum_allowed_case_ratio"]
    )
    if require_published_reference:
        maximum = float(config["maximum_candidate_l1_m"])
        gate["maximum_candidate_l1_m"] = maximum
        passed = passed and float(summary["candidate_mean_l1_m"]) < maximum
    gate["passed"] = passed
    return gate


def _evaluate(args: argparse.Namespace, protocol: dict[str, object]) -> int:
    if args.training_result is None:
        raise ValueError("evaluate mode requires --training-result")
    protocol_path = args.protocol.resolve()
    training_path = args.training_result.resolve()
    training, manifest, manifest_path = _verify_training_result(
        training_path,
        protocol=protocol,
        protocol_path=protocol_path,
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO2" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    source_runtime._assert_upstream(args.upstream_root, protocol["upstream"]["commit"])
    cublas_config = str(protocol["training"]["cublas_workspace_config"])
    existing_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas_config not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    started = time.perf_counter()
    source_runtime._seed_everything(torch, 42)
    modules = source_runtime._load_upstream(args.upstream_root)
    state, checkpoint_path = _checkpoint_state(training, torch=torch)
    fit_names = list(manifest["split"]["fit"])
    validation_names = list(manifest["split"]["validation"])
    source_names = list(manifest["split"]["source_test"])
    development_names = fit_names + validation_names
    development = source_runtime._load_named_trajectories(
        manifest,
        development_names,
        frame_count=500,
        node_count=12,
    )
    development_rollout = _rollout(
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
    selected = training["selected_checkpoint"]
    validation_baseline_l1_m = common_runtime._mean_l1(
        validation_rollout["predictions"], validation_rollout["targets"]
    )
    common_runtime._require_baseline_reproduction(
        validation_baseline_l1_m,
        expected=float(selected["validation_l1_m"]),
        tolerance=1e-7,
        stage="DLO2-validation",
    )
    local = protocol["local_residual"]
    fixed = local["fixed_arm"]
    fit_initial, fit_action = local_runtime._causal_inputs(development, fit_names)
    model = fit_deform_local_residual(
        fit_initial,
        fit_action,
        np.asarray(fit_rollout["predictions"]),
        np.asarray(fit_rollout["targets"]),
        fit_names,
        ridge=float(fixed["ridge"]),
        variance_floor_m2=float(local["coordinate_variance_floor_m2"]),
    )
    output_root = args.output_root.resolve()
    model_path = output_root / "local_residual_model.npz"
    np.savez_compressed(model_path, **serialize_deform_local_residual_model(model))
    validation_initial, validation_action = local_runtime._causal_inputs(
        development, validation_names
    )
    validation_prediction = predict_deform_local_residual(
        model,
        validation_initial,
        validation_action,
        np.asarray(validation_rollout["predictions"]),
        shrinkage=float(fixed["shrinkage"]),
    )
    validation_records = deform_action_residual_records(
        validation_prediction["predictions"],
        validation_rollout["targets"],
        validation_rollout["predictions"],
        validation_names,
    )
    validation_summary = summarize_deform_action_residual_records(validation_records)
    validation_gate = _gate(
        validation_summary,
        local["validation_gate"],
        require_published_reference=False,
    )
    selection = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-validation-v5",
        "protocol_sha256": sha256_file(protocol_path),
        "training_result": {
            "path": str(training_path),
            "sha256": sha256_file(training_path),
        },
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "selected_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "model_sha256": sha256_file(model_path),
        "fixed_arm": fixed,
        "validation_records": validation_records,
        "validation_gate": validation_gate,
        "validation_diagnostics": local_runtime._prediction_diagnostics(
            validation_prediction, validation_rollout["targets"]
        ),
        "source_test_opened": False,
        "official_eval_read": False,
    }
    selection_path = output_root / "validation_transfer_seal.json"
    common_runtime._write_json(selection_path, selection)
    if not bool(validation_gate["passed"]):
        result = {
            "schema_version": 1,
            "contract": "deform-dlo2-local-residual-result-v5",
            "claim_boundary": protocol["claim_boundary"],
            "protocol_sha256": sha256_file(protocol_path),
            "validation_transfer_sha256": sha256_file(selection_path),
            "fallback_used": True,
            "selected_method": "selected-dlo2-checkpoint-exact",
            "validation_gate": validation_gate,
            "source_test_opened": False,
            "source_gate": {"passed": False, "reason": "validation-gate-failed"},
            "alltrain_and_official_evaluation_authorized": False,
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
        node_count=12,
    )
    source_rollout = _rollout(
        state,
        source_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
    )
    source_initial, source_action = local_runtime._causal_inputs(
        source_trajectories, source_names
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
    source_gate = _gate(
        source_summary,
        local["source_transfer_gate"],
        require_published_reference=True,
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
        "contract": "deform-dlo2-local-residual-result-v5",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": sha256_file(protocol_path),
        "training_result_sha256": sha256_file(training_path),
        "validation_transfer_sha256": sha256_file(selection_path),
        "model_sha256": sha256_file(model_path),
        "source_prediction_sha256": sha256_file(source_prediction_path),
        "fallback_used": False,
        "selected_method": "dlo2-baseline-plus-fixed-dlo1-local-residual",
        "fixed_arm": fixed,
        "validation_gate": validation_gate,
        "source_test_opened": True,
        "source_records": source_records,
        "source_gate": source_gate,
        "source_diagnostics": local_runtime._prediction_diagnostics(
            source_prediction, source_rollout["targets"]
        ),
        "alltrain_and_official_evaluation_authorized": bool(source_gate["passed"]),
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


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo2_local_residual_protocol(protocol_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    parent = _parent_authorization(protocol, args.parent_result.resolve())
    authorization_path = _write_authorization(
        output_root,
        mode=str(args.mode),
        protocol_path=protocol_path,
        parent=parent,
    )
    if args.mode != "evaluate":
        return _run_training_stage(args, authorization_path=authorization_path)
    return _evaluate(args, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
