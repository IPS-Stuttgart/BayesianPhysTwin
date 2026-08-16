#!/usr/bin/env python3
"""Run the single authorized official DLO2 local-residual evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import run_deform_dlo2_official as official_runtime
import run_deform_dlo_local_residual as local_runtime
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_local_residual import (
    deserialize_deform_local_residual_model,
    load_deform_dlo2_local_residual_alltrain_v7_protocol,
    load_deform_dlo2_local_residual_official_v7_protocol,
    predict_deform_local_residual,
    validate_deform_dlo2_local_residual_official_v7_authorization,
)
from bayesian_phystwin.deform_dlo_official import (
    evaluate_deform_dlo2_official_uncertainty,
    summarize_deform_dlo2_official_records,
)
from bayesian_phystwin.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--alltrain-protocol", type=Path, required=True)
    parser.add_argument("--alltrain-result", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise RuntimeError(f"locked output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verified_json_identity(
    identity: Mapping[str, object], *, label: str
) -> tuple[Path, dict[str, object]]:
    path = Path(str(identity.get("path", ""))).resolve()
    expected_size = int(str(identity.get("size_bytes", -1)))
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity does not verify")
    return path, _read_json(path)


def _verified_file_identity(identity: Mapping[str, object], *, label: str) -> Path:
    path = Path(str(identity.get("path", ""))).resolve()
    expected_size = int(str(identity.get("size_bytes", -1)))
    if (
        not path.is_file()
        or path.stat().st_size != expected_size
        or sha256_file(path) != identity.get("sha256")
    ):
        raise ValueError(f"{label} identity does not verify")
    return path


def _failure_payload(*, stage: str, error: BaseException) -> dict[str, object]:
    return {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-official-failure-v7",
        "official_eval_read": True,
        "retry_authorized": False,
        "stage": stage,
        "exception_type": type(error).__name__,
        "message": str(error),
    }


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve()
    alltrain_protocol_path = args.alltrain_protocol.resolve()
    alltrain_result_path = args.alltrain_result.resolve()
    protocol = load_deform_dlo2_local_residual_official_v7_protocol(protocol_path)
    alltrain_protocol = load_deform_dlo2_local_residual_alltrain_v7_protocol(
        alltrain_protocol_path
    )
    alltrain_result = _read_json(alltrain_result_path)
    final_identity = _mapping(
        alltrain_result.get("final_method"), label="final-method identity"
    )
    final_method_path, final_method = _verified_json_identity(
        final_identity, label="final method"
    )
    selected = validate_deform_dlo2_local_residual_official_v7_authorization(
        protocol,
        alltrain_protocol,
        alltrain_result,
        final_method,
        alltrain_protocol_sha256=sha256_file(alltrain_protocol_path),
        alltrain_result_sha256=sha256_file(alltrain_result_path),
        final_method_sha256=sha256_file(final_method_path),
    )
    checkpoint_identity = _mapping(
        selected["physical_checkpoint"], label="physical checkpoint"
    )
    local_model_identity = _mapping(
        selected["local_residual_model"], label="local residual model"
    )
    method_spec_identity = _mapping(
        final_method.get("method_spec"), label="method specification"
    )
    schedule_identity = _mapping(
        final_method.get("window_schedule"), label="window schedule"
    )
    method_spec_path = _verified_file_identity(
        method_spec_identity, label="method specification"
    )
    schedule_path = _verified_file_identity(schedule_identity, label="window schedule")
    local_model_path = _verified_file_identity(
        local_model_identity, label="local residual model"
    )
    with np.load(local_model_path, allow_pickle=False) as archive:
        local_model = deserialize_deform_local_residual_model(archive)

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"one-shot output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    upstream = source_runtime._assert_upstream(
        args.upstream_root, str(alltrain_protocol["upstream"]["commit"])
    )
    cublas_config = str(alltrain_protocol["training"]["cublas_workspace_config"])
    existing_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_cublas not in (None, cublas_config):
        raise RuntimeError("existing cuBLAS workspace configuration differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_config
    import torch

    runtime = _mapping(selected["runtime"], label="all-train runtime")
    if torch.__version__ != runtime.get("torch") or torch.version.cuda != runtime.get(
        "cuda"
    ):
        raise RuntimeError("official evaluator runtime differs from all-train refit")
    checkpoint_bundle = official_runtime._verified_checkpoint_bundle(
        checkpoint_identity,
        torch=torch,
        alltrain_protocol_sha256=sha256_file(alltrain_protocol_path),
        schedule_sha256=sha256_file(schedule_path),
        method_spec_sha256=sha256_file(method_spec_path),
        expected_update=6400,
    )
    authorization = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-official-authorization-v7",
        "official_eval_read": False,
        "one_shot_execution_authorized": True,
        "protocol": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "alltrain_protocol": {
            "path": str(alltrain_protocol_path),
            "sha256": sha256_file(alltrain_protocol_path),
        },
        "alltrain_result": {
            "path": str(alltrain_result_path),
            "sha256": sha256_file(alltrain_result_path),
        },
        "final_method": {
            "path": str(final_method_path),
            "sha256": sha256_file(final_method_path),
        },
        "physical_checkpoint": checkpoint_identity,
        "local_residual_model": local_model_identity,
        "fixed_arm": selected["fixed_arm"],
        "target_selection": False,
        "target_calibration": False,
        "target_retries": False,
        "case_replacement": False,
        "upstream": upstream,
    }
    authorization_path = output_root / "authorization.json"
    _write_json(authorization_path, authorization)

    evaluation = _mapping(protocol["evaluation"], label="evaluation")
    reference = _mapping(
        evaluation["published_reference_operator"],
        label="published reference operator",
    )
    reference_draw = [int(value) for value in reference["canonical_eval_indices"]]
    eval_root = args.upstream_root.resolve() / "data_set" / "DLO2" / "eval"
    modules = source_runtime._load_upstream(args.upstream_root)
    source_runtime._seed_everything(
        torch, int(alltrain_protocol["training"]["random_seed"])
    )
    stage = "target-manifest"
    started = time.perf_counter()
    try:
        manifest = official_runtime._build_eval_manifest(
            eval_root,
            expected_count=int(evaluation["expected_trajectory_count"]),
            canonical_reference_draw_indices=reference_draw,
            protocol_path=protocol_path,
            alltrain_result_path=alltrain_result_path,
        )
        manifest_path = output_root / "evaluation_manifest.json"
        _write_json(manifest_path, manifest)
        stage = "target-load"
        names = list(manifest["ordered_names"])
        trajectories = source_runtime._load_named_trajectories(
            manifest,
            names,
            frame_count=int(evaluation["expected_frame_count"]),
            node_count=int(evaluation["expected_node_count"]),
        )
        stage = "fixed-rollout"
        baseline_rollout = posterior_runtime._evaluate_state(
            checkpoint_bundle["model_state_dict"],
            trajectories,
            modules=modules,
            torch=torch,
            device=args.device,
            dlo_type="DLO2",
            node_count=int(evaluation["expected_node_count"]),
        )
        initial, action = local_runtime._causal_inputs(trajectories, names)
        prediction = predict_deform_local_residual(
            local_model,
            initial,
            action,
            np.asarray(baseline_rollout["predictions"]),
            shrinkage=float(selected["fixed_arm"]["shrinkage"]),
        )
        candidate_rollout = {
            "names": baseline_rollout["names"],
            "predictions": prediction["predictions"],
            "targets": baseline_rollout["targets"],
            "persistence": baseline_rollout["persistence"],
        }
        candidate_records = posterior_runtime._records(candidate_rollout)
        baseline_records = posterior_runtime._records(baseline_rollout)
        gate = _mapping(protocol["claim_gate"], label="claim gate")
        summary = summarize_deform_dlo2_official_records(
            candidate_records,
            baseline_records,
            expected_case_count=int(evaluation["expected_trajectory_count"]),
            published_reference_l1_m=float(evaluation["published_reference_l1_m"]),
            minimum_relative_improvement=float(
                gate["candidate_relative_improvement_min"]
            ),
            minimum_case_wins=int(gate["candidate_minimum_case_wins"]),
            canonical_reference_draw_indices=reference_draw,
        )
        candidate_by_name = {
            str(record["name"]): float(record["model_l1_m"])
            for record in candidate_records
        }
        baseline_by_name = {
            str(record["name"]): float(record["model_l1_m"])
            for record in baseline_records
        }
        ratios = [
            candidate_by_name[name] / baseline_by_name[name]
            if baseline_by_name[name] > 0.0
            else float("inf")
            for name in sorted(candidate_by_name)
        ]
        maximum_case_ratio = float(max(ratios))
        ratio_passed = maximum_case_ratio <= float(gate["maximum_case_ratio"])
        summary["maximum_case_ratio"] = maximum_case_ratio
        summary["claim_gate"]["maximum_case_ratio_passed"] = ratio_passed
        summary["claim_gate"]["passed"] = bool(
            summary["claim_gate"]["passed"] and ratio_passed
        )
        uncertainty_config = _mapping(protocol["uncertainty"], label="uncertainty")
        uncertainty = evaluate_deform_dlo2_official_uncertainty(
            np.asarray(prediction["predictions"]),
            np.asarray(baseline_rollout["targets"]),
            np.asarray(prediction["coordinate_variance_m2"]),
            variance_floor_m2=float(uncertainty_config["variance_floor_m2"]),
            variance_scale=float(uncertainty_config["variance_scale"]),
            nominal_coverage=float(
                uncertainty_config["nominal_coordinate_coverage"]
            ),
        )
        prediction_path = output_root / "official_prediction.npz"
        np.savez_compressed(
            prediction_path,
            names=np.asarray(names),
            baseline_predictions=np.asarray(baseline_rollout["predictions"]),
            candidate_predictions=np.asarray(prediction["predictions"]),
            targets=np.asarray(baseline_rollout["targets"]),
            persistence=np.asarray(baseline_rollout["persistence"]),
            coordinate_variance_m2=np.asarray(
                prediction["coordinate_variance_m2"]
            ),
        )
        result = {
            "schema_version": 1,
            "contract": "deform-dlo2-local-residual-official-result-v7",
            "claim_boundary": protocol["claim_boundary"],
            "official_eval_read": True,
            "target_selection_performed": False,
            "target_calibration_performed": False,
            "target_retry_performed": False,
            "case_replacement_performed": False,
            "all_expected_cases_evaluated_once": True,
            "protocol": authorization["protocol"],
            "authorization": {
                "path": str(authorization_path),
                "sha256": sha256_file(authorization_path),
            },
            "evaluation_manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            },
            "prediction": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
                "size_bytes": prediction_path.stat().st_size,
            },
            "fixed_arm": selected["fixed_arm"],
            "comparison": summary,
            "candidate_cases": candidate_records,
            "comparison_baseline_cases": baseline_records,
            "uncertainty": {
                "alltrain_fit_covariance_reused_unchanged": True,
                "variance_scale": uncertainty_config["variance_scale"],
                "variance_floor_m2": uncertainty_config["variance_floor_m2"],
                "nominal_coordinate_coverage": uncertainty_config[
                    "nominal_coordinate_coverage"
                ],
                "metrics": uncertainty,
            },
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "elapsed_seconds": time.perf_counter() - started,
            },
        }
        _write_json(output_root / "official_result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        _write_json(
            output_root / "official_failure.json",
            _failure_payload(stage=stage, error=error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
