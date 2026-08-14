#!/usr/bin/env python3
"""Run the DLO1-only action-conditioned residual source study."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_dlo_longrun_posterior as posterior_runtime
import run_deform_dlo_source as source_runtime

from bayesian_phystwin.deform_dlo_action_residual import (
    deform_action_residual_records,
    fit_deform_action_residual,
    load_deform_action_residual_protocol,
    predict_deform_action_residual,
    select_deform_action_residual_arm,
    serialize_deform_action_residual_model,
    summarize_deform_action_residual_records,
)
from bayesian_phystwin.deform_dlo_longrun import load_deform_dlo_longrun_protocol
from bayesian_phystwin.deform_dlo_source import sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--longrun-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
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


def _identity_path(
    identity: dict[str, object],
    *,
    repository_root: Path,
) -> Path:
    raw = identity.get("repository_path", identity.get("path"))
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _verify_identity(
    path: Path,
    identity: dict[str, object],
    *,
    label: str,
) -> None:
    if not path.is_file() or sha256_file(path) != identity.get("sha256"):
        raise ValueError(f"{label} identity does not verify")


def _arm_name(neighbors: int, length_scale: float, shrinkage: float) -> str:
    def token(value: float) -> str:
        return format(value, ".8g").replace(".", "p")

    return f"k{neighbors:02d}_l{token(length_scale)}_s{token(shrinkage)}"


def _arm_specs(protocol: dict[str, object]) -> dict[str, dict[str, float | int]]:
    bank = protocol["candidate_bank"]
    specs = {}
    for neighbors in bank["neighbor_counts"]:
        for length_scale in bank["length_scale_multipliers"]:
            for shrinkage in bank["shrinkages"]:
                name = _arm_name(
                    int(neighbors),
                    float(length_scale),
                    float(shrinkage),
                )
                specs[name] = {
                    "neighbor_count": int(neighbors),
                    "length_scale_multiplier": float(length_scale),
                    "shrinkage": float(shrinkage),
                }
    return specs


def _stack_trajectories(
    trajectories: dict[str, np.ndarray], names: list[str]
) -> np.ndarray:
    return np.stack([trajectories[name] for name in names])


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
        dlo_type="DLO1",
        node_count=13,
    )


def _split_rollout(
    rollout: dict[str, object], start: int, stop: int
) -> dict[str, object]:
    return {
        "names": list(rollout["names"])[start:stop],
        "predictions": np.asarray(rollout["predictions"])[start:stop],
        "targets": np.asarray(rollout["targets"])[start:stop],
        "persistence": np.asarray(rollout["persistence"])[start:stop],
    }


def _json_safe_prediction_diagnostics(result: dict[str, object]) -> dict[str, object]:
    return {
        "neighbor_indices": np.asarray(result["neighbor_indices"]).tolist(),
        "neighbor_distances": np.asarray(result["neighbor_distances"]).tolist(),
        "weights": np.asarray(result["weights"]).tolist(),
        "effective_sample_size": np.asarray(result["effective_sample_size"]).tolist(),
        "correction_l2_m": np.asarray(result["correction_l2_m"]).tolist(),
    }


def _mean_l1(predictions: object, targets: object) -> float:
    predicted = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    if (
        predicted.shape != observed.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("baseline reproduction arrays are invalid")
    return float(np.mean(np.abs(predicted - observed)))


def _require_baseline_reproduction(
    actual: float,
    *,
    expected: float,
    tolerance: float,
    stage: str,
) -> None:
    if abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"{stage} baseline drifted: {actual:.12g} != {expected:.12g}"
        )


def main() -> int:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    protocol_path = args.protocol.resolve()
    protocol = load_deform_action_residual_protocol(protocol_path)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    longrun_protocol_identity = protocol["longrun_protocol"]
    longrun_protocol_path = _identity_path(
        longrun_protocol_identity,
        repository_root=repository_root,
    )
    _verify_identity(
        longrun_protocol_path,
        longrun_protocol_identity,
        label="long-run protocol",
    )
    longrun_protocol = load_deform_dlo_longrun_protocol(longrun_protocol_path)
    longrun_result_path = args.longrun_result.resolve()
    _verify_identity(
        longrun_result_path,
        protocol["longrun_result"],
        label="long-run result",
    )
    longrun_result = _read_json(longrun_result_path)
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
        raise ValueError("action-residual baseline differs from the locked checkpoint")

    source_manifest_path = args.source_manifest.resolve()
    _verify_identity(
        source_manifest_path,
        protocol["source_manifest"],
        label="source manifest",
    )
    manifest = _read_json(source_manifest_path)
    if (
        manifest.get("contract") != "deform-dlo-source-reproduction-v1"
        or manifest.get("dlo_type") != "DLO1"
        or manifest.get("partition") != "train"
        or manifest.get("official_eval_read") is not False
    ):
        raise ValueError("action-residual source manifest is invalid")

    source_runtime._assert_upstream(
        args.upstream_root,
        longrun_result["upstream"]["commit"],
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO2")
    preflight = {
        "schema_version": 1,
        "contract": "deform-dlo-action-residual-preflight-v3",
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
    _write_json(output_root / "preflight.json", preflight)

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
    development_rollout = _rollout(
        state,
        development,
        modules=modules,
        torch=torch,
        device=args.device,
    )
    fit_rollout = _split_rollout(development_rollout, 0, len(fit_names))
    validation_rollout = _split_rollout(
        development_rollout,
        len(fit_names),
        len(development_names),
    )
    reproduction_tolerance = float(baseline["reproduction_tolerance_m"])
    validation_baseline_l1_m = _mean_l1(
        validation_rollout["predictions"], validation_rollout["targets"]
    )
    _require_baseline_reproduction(
        validation_baseline_l1_m,
        expected=float(baseline["validation_l1_m"]),
        tolerance=reproduction_tolerance,
        stage="validation",
    )
    descriptor = protocol["descriptor"]
    posterior = protocol["posterior"]
    model = fit_deform_action_residual(
        _stack_trajectories(development, fit_names),
        fit_rollout["predictions"],
        fit_rollout["targets"],
        fit_names,
        sample_count=int(descriptor["sample_count"]),
        variance_floor_m2=float(posterior["coordinate_variance_floor_m2"]),
    )
    model_path = output_root / "action_residual_model.npz"
    np.savez_compressed(model_path, **serialize_deform_action_residual_model(model))

    validation_trajectories = _stack_trajectories(development, validation_names)
    specs = _arm_specs(protocol)
    validation_records = {}
    validation_diagnostics = {}
    for name, spec in specs.items():
        prediction = predict_deform_action_residual(
            model,
            validation_trajectories,
            validation_rollout["predictions"],
            neighbor_count=int(spec["neighbor_count"]),
            length_scale_multiplier=float(spec["length_scale_multiplier"]),
            shrinkage=float(spec["shrinkage"]),
        )
        validation_records[name] = deform_action_residual_records(
            prediction["predictions"],
            validation_rollout["targets"],
            validation_rollout["predictions"],
            validation_names,
        )
        validation_diagnostics[name] = _json_safe_prediction_diagnostics(prediction)
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
        "contract": "deform-dlo-action-residual-validation-selection-v3",
        "protocol_sha256": sha256_file(protocol_path),
        "model_sha256": sha256_file(model_path),
        "longrun_result_sha256": sha256_file(longrun_result_path),
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
    _write_json(selection_path, selection_seal)

    if bool(selection["fallback_used"]):
        result = {
            "schema_version": 1,
            "contract": "deform-dlo-action-residual-result-v3",
            "claim_boundary": protocol["claim_boundary"],
            "protocol_sha256": sha256_file(protocol_path),
            "model_sha256": sha256_file(model_path),
            "validation_selection_sha256": sha256_file(selection_path),
            "selected_arm": "baseline_exact",
            "validation": selection,
            "validation_baseline_l1_m": validation_baseline_l1_m,
            "source_test_opened": False,
            "source_gate": {"passed": False, "reason": "validation-gate-failed"},
            "fresh_dlo2_action_residual_authorized": False,
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
        _write_json(output_root / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    source_trajectories = source_runtime._load_named_trajectories(
        manifest,
        source_names,
        frame_count=500,
        node_count=13,
    )
    source_rollout = _rollout(
        state,
        source_trajectories,
        modules=modules,
        torch=torch,
        device=args.device,
    )
    source_baseline_l1_m = _mean_l1(
        source_rollout["predictions"], source_rollout["targets"]
    )
    _require_baseline_reproduction(
        source_baseline_l1_m,
        expected=float(baseline["source_test_l1_m"]),
        tolerance=reproduction_tolerance,
        stage="source-test",
    )
    selected_spec = specs[selected_name]
    source_prediction = predict_deform_action_residual(
        model,
        _stack_trajectories(source_trajectories, source_names),
        source_rollout["predictions"],
        neighbor_count=int(selected_spec["neighbor_count"]),
        length_scale_multiplier=float(selected_spec["length_scale_multiplier"]),
        shrinkage=float(selected_spec["shrinkage"]),
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
        "contract": "deform-dlo-action-residual-result-v3",
        "claim_boundary": protocol["claim_boundary"],
        "protocol_sha256": sha256_file(protocol_path),
        "model_sha256": sha256_file(model_path),
        "validation_selection_sha256": sha256_file(selection_path),
        "source_prediction_sha256": sha256_file(source_prediction_path),
        "selected_arm": selected_name,
        "selected_spec": selected_spec,
        "validation": selection,
        "validation_baseline_l1_m": validation_baseline_l1_m,
        "source_test_opened": True,
        "source_baseline_l1_m": source_baseline_l1_m,
        "source_records": source_records,
        "source_diagnostics": _json_safe_prediction_diagnostics(source_prediction),
        "source_gate": source_gate,
        "fresh_dlo2_action_residual_authorized": bool(source_gate["passed"]),
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
    _write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
