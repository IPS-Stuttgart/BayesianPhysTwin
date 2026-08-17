#!/usr/bin/env python3
"""Select a conservative DLO2 shrinkage using fit and validation only."""

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

from bayesian_phystwin_experiments.deform_dlo_action_residual import (
    deform_action_residual_records,
    summarize_deform_action_residual_records,
)
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    fit_deform_local_residual,
    load_deform_dlo2_local_residual_protocol,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file

SHRINKAGE_BANK = (0.125, 0.25, 0.375, 0.5)
MINIMUM_RELATIVE_IMPROVEMENT = 0.01
MINIMUM_CASE_WINS = 6
MAXIMUM_CASE_RATIO = 1.05


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _passes(summary: dict[str, object]) -> bool:
    return bool(
        float(summary["relative_improvement"]) >= MINIMUM_RELATIVE_IMPROVEMENT
        and int(summary["wins"]) >= MINIMUM_CASE_WINS
        and float(summary["maximum_case_ratio"]) <= MAXIMUM_CASE_RATIO
    )


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    protocol_path = args.protocol.resolve()
    protocol = load_deform_dlo2_local_residual_protocol(protocol_path)
    training_path = args.training_result.resolve()
    training, manifest, _ = v5_runtime._verify_training_result(
        training_path,
        protocol=protocol,
        protocol_path=protocol_path,
    )
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO2" / "eval")
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    source_runtime._assert_upstream(args.upstream_root, protocol["upstream"]["commit"])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(
        protocol["training"]["cublas_workspace_config"]
    )
    import torch

    started = time.perf_counter()
    source_runtime._seed_everything(torch, 42)
    modules = source_runtime._load_upstream(args.upstream_root)
    state, checkpoint_path = v5_runtime._checkpoint_state(training, torch=torch)
    fit_names = list(manifest["split"]["fit"])
    validation_names = list(manifest["split"]["validation"])
    development_names = fit_names + validation_names
    development = source_runtime._load_named_trajectories(
        manifest,
        development_names,
        frame_count=500,
        node_count=12,
    )
    rollout = v5_runtime._rollout(
        state,
        development,
        modules=modules,
        torch=torch,
        device=args.device,
    )
    fit_rollout = common_runtime._split_rollout(rollout, 0, len(fit_names))
    validation_rollout = common_runtime._split_rollout(
        rollout,
        len(fit_names),
        len(development_names),
    )
    selected_checkpoint = training["selected_checkpoint"]
    baseline_l1_m = common_runtime._mean_l1(
        validation_rollout["predictions"], validation_rollout["targets"]
    )
    common_runtime._require_baseline_reproduction(
        baseline_l1_m,
        expected=float(selected_checkpoint["validation_l1_m"]),
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
    validation_initial, validation_action = local_runtime._causal_inputs(
        development, validation_names
    )
    bank = []
    for shrinkage in SHRINKAGE_BANK:
        prediction = predict_deform_local_residual(
            model,
            validation_initial,
            validation_action,
            np.asarray(validation_rollout["predictions"]),
            shrinkage=shrinkage,
        )
        records = deform_action_residual_records(
            prediction["predictions"],
            validation_rollout["targets"],
            validation_rollout["predictions"],
            validation_names,
        )
        summary = summarize_deform_action_residual_records(records)
        bank.append(
            {
                "name": f"shrinkage-{shrinkage:g}",
                "shrinkage": shrinkage,
                "records": records,
                "summary": summary,
                "passes_validation_gate": _passes(summary),
                "diagnostics": local_runtime._prediction_diagnostics(
                    prediction, validation_rollout["targets"]
                ),
            }
        )
    eligible = [entry for entry in bank if entry["passes_validation_gate"]]
    selected = (
        min(
            eligible,
            key=lambda entry: (
                float(entry["summary"]["candidate_mean_l1_m"]),
                float(entry["shrinkage"]),
            ),
        )
        if eligible
        else None
    )
    result = {
        "schema_version": 1,
        "contract": "deform-dlo2-local-residual-development-v6",
        "claim_boundary": (
            "DLO2 fit/validation development only; source and official outcomes "
            "remain unopened."
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "training_result_sha256": sha256_file(training_path),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "selection_rule": (
            "minimum validation L1 among finite arms passing locked mean, win, "
            "and worst-case gates"
        ),
        "gates": {
            "minimum_relative_improvement": MINIMUM_RELATIVE_IMPROVEMENT,
            "minimum_case_wins": MINIMUM_CASE_WINS,
            "maximum_case_ratio": MAXIMUM_CASE_RATIO,
        },
        "bank": bank,
        "selected_arm": (
            None
            if selected is None
            else {
                "name": selected["name"],
                "shrinkage": selected["shrinkage"],
                "summary": selected["summary"],
            }
        ),
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
    common_runtime._write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
