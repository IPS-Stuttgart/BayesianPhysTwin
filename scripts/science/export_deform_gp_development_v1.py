#!/usr/bin/env python3
"""Export DLO1/train native hybrid and ridge forecasts for GP development.

Uses the exact parent checkpoint and existing per-node residual implementation.
No retraining of DEFORM, official evaluation, new hardware, or active sensing.
Execute only through reviewed source-data compute, not pull-request CI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/remote"))

from bayesian_phystwin_experiments.deform_dlo_local_residual import (  # noqa: E402
    build_deform_local_residual_features,
    deform_causal_inputs,
    fit_deform_local_residual,
    load_deform_local_residual_protocol,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file  # noqa: E402
from bayesian_phystwin_experiments.gp_discrepancy_v1 import chain_modes  # noqa: E402
import run_deform_dlo_action_residual as common  # noqa: E402
import run_deform_dlo_longrun_posterior as posterior  # noqa: E402
import run_deform_dlo_source as source_runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--longrun-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = load_deform_local_residual_protocol(args.parent_protocol)
    common._verify_identity(args.longrun_result, protocol["longrun_result"], label="parent longrun result")
    common._verify_identity(args.source_manifest, protocol["source_manifest"], label="DLO1 source manifest")
    result = common._read_json(args.longrun_result)
    manifest = common._read_json(args.source_manifest)
    if (manifest.get("contract") != "deform-dlo-source-reproduction-v1"
            or manifest.get("dlo_type") != "DLO1" or manifest.get("partition") != "train"
            or manifest.get("official_eval_read") is not False):
        raise ValueError("only the pinned DLO1/train source partition is allowed")
    selected = result["selected_checkpoint"]
    if selected["checkpoint"]["sha256"] != protocol["baseline"]["checkpoint_sha256"]:
        raise ValueError("parent checkpoint changed")
    source_runtime._assert_upstream(args.upstream_root, result["upstream"]["commit"])
    data_root = args.upstream_root.resolve() / "data_set"
    source_runtime._install_eval_read_guard(data_root / "DLO1" / "eval")
    for number in (2, 3, 4, 5):
        source_runtime._install_eval_read_guard(data_root / f"DLO{number}")
    fit_names = list(manifest["split"]["fit"])
    validation = sorted(manifest["split"]["validation"])
    if len(validation) < 4:
        raise ValueError("not enough validation recordings for separate selection and calibration")
    select_names, calibration_names = validation[::2], validation[1::2]
    score_names = list(manifest["split"]["source_test"])
    groups = [fit_names, select_names, calibration_names, score_names]
    names = [name for group in groups for name in group]
    if len(names) != len(set(names)):
        raise ValueError("parent recording splits overlap")
    preflight = {"scope": "DLO1/train", "official_evaluation_opened": False,
                 "parent_protocol_sha256": sha256_file(args.parent_protocol),
                 "parent_result_sha256": sha256_file(args.longrun_result),
                 "source_manifest_sha256": sha256_file(args.source_manifest),
                 "checkpoint_sha256": selected["checkpoint"]["sha256"],
                 "upstream_commit": result["upstream"]["commit"],
                 "roles": dict(zip(("fit", "select", "calibrate", "score"), groups, strict=True)),
                 "score_status": "previously inspected DLO1 source-test; retrospective development only"}
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    source_runtime._seed_everything(torch, 42)
    modules = source_runtime._load_upstream(args.upstream_root)
    update = int(selected["update"])
    state = posterior._checkpoint_states(result, {update}, torch=torch)[update]
    trajectories = source_runtime._load_named_trajectories(manifest, names, frame_count=500, node_count=13)
    rollout = common._rollout(state, trajectories, modules=modules, torch=torch, device=args.device)
    if list(rollout["names"]) != names:
        raise ValueError("native rollout reordered recordings")
    baseline, truth = np.asarray(rollout["predictions"]), np.asarray(rollout["targets"])
    initial, action = deform_causal_inputs(common._stack_trajectories(trajectories, names))
    fit_count, select_count = len(fit_names), len(select_names)
    fit_slice, select_slice = slice(0, fit_count), slice(fit_count, fit_count + select_count)
    choices = []
    for ridge in protocol["candidate_bank"]["ridges"]:
        fitted = fit_deform_local_residual(initial[fit_slice], action[fit_slice], baseline[fit_slice],
                                           truth[fit_slice], fit_names, ridge=float(ridge),
                                           variance_floor_m2=float(protocol["posterior"]["coordinate_variance_floor_m2"]))
        for shrinkage in protocol["candidate_bank"]["shrinkages"]:
            candidate = predict_deform_local_residual(fitted, initial[select_slice], action[select_slice],
                                                      baseline[select_slice], shrinkage=float(shrinkage))["predictions"]
            choices.append((float(np.mean(np.abs(candidate - truth[select_slice]))), float(ridge), float(shrinkage), fitted))
    best = min(choices, key=lambda item: item[0])
    ridge_predictions = predict_deform_local_residual(best[3], initial, action, baseline, shrinkage=best[2])["predictions"]
    features, frames = build_deform_local_residual_features(initial, action, baseline)
    times = np.unique(np.linspace(0, baseline.shape[1] - 1, 24, dtype=int))
    role_labels = np.concatenate([np.repeat(role, len(group)) for role, group in zip(("fit", "select", "calibrate", "score"), groups, strict=True)])
    archive = output / "development.npz"
    np.savez_compressed(archive, features=features[:, times].reshape(len(names), len(times), -1),
                        times=times / (baseline.shape[1] - 1), ids=np.asarray(names), split=role_labels,
                        baseline=baseline[:, times], ridge=ridge_predictions[:, times], truth=truth[:, times],
                        basis=chain_modes(13, 4), frames=frames)
    metadata = {**preflight, "schema": "gp-discrepancy-development-archive-v1",
                "archive_sha256": sha256_file(archive), "forecast_time_indices": times.tolist(),
                "native_ridge_selection": {"ridge": best[1], "shrinkage": best[2], "selection_l1_m": best[0]},
                "native_ridge_candidates": [{"selection_l1_m": item[0], "ridge": item[1], "shrinkage": item[2]} for item in choices],
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                "claim": "sampled-horizon development comparison, not the full official operator"}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
