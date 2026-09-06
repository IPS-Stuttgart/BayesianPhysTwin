#!/usr/bin/env python3
"""Issue #946: matched offline DLO1 GP diagnostic, all final data forbidden."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import run_deform_dlo_action_residual as common
import run_deform_dlo_longrun_posterior as posterior
import run_deform_dlo_source as source

from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    _collapse_duplicate_queries,
    build_deform_local_residual_features,
    deform_causal_inputs,
    fit_deform_local_residual,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.deform_dlo_source import sha256_file
from bayesian_phystwin_experiments.deform_gp_residual_v1 import (
    add_residual,
    balanced_anchor_indices,
    fit_gp_bank,
)


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def errors(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid prediction")
    return np.mean(np.abs(prediction - truth), axis=(1, 2, 3))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("protocols/deform_gp_residual_development_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    if protocol["contract"] != "deform-gp-residual-development-v1" or protocol["allowed_dataset"] != "DLO1/train":
        raise ValueError("wrong development protocol")
    root = Path(__file__).resolve().parents[2]
    parent_path = root / protocol["parent_protocol"]
    parent = json.loads(parent_path.read_text())
    manifest_path = root / parent["source_manifest"]["repository_path"]
    common._verify_identity(manifest_path, parent["source_manifest"], label="manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest["dlo_type"] != "DLO1" or manifest["partition"] != "train" or manifest["official_eval_read"]:
        raise ValueError("forbidden dataset")
    split = manifest["split"]
    names = split["fit"] + split["validation"] + split["source_test"]
    if [len(split[k]) for k in ("fit", "validation", "source_test")] != [40, 8, 8] or len(set(names)) != 56:
        raise ValueError("historical split changed")
    # Strict raw-data allowlist, including symlink resolution, with source closed until selection.
    phase = {"source_open": False}
    allowed = {}
    for stage in ("fit", "validation", "source_test"):
        for name in split[stage]:
            p = Path(manifest["trajectories"][name]["path"]).resolve()
            if p.name != name or p.parent.name != "train" or p.parent.parent.name != "DLO1":
                raise ValueError("manifest points outside DLO1/train")
            allowed[str(p)] = stage
    accessed: set[str] = set()

    def audit(event: str, arguments: tuple) -> None:
        if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        p = os.path.realpath(os.fsdecode(arguments[0]))
        low = p.lower()
        if ("/data_set/dlo" in low or "/datasets/deform/" in low) and low.endswith(".pkl"):
            stage = allowed.get(p)
            if stage is None or (stage == "source_test" and not phase["source_open"]):
                raise PermissionError(f"GP diagnostic raw-data boundary: {p}")
            accessed.add(p)
        if "pokeflex" in low or "/data_set/dlo2/" in low or "/dlo1/eval/" in low:
            raise PermissionError("forbidden data path")

    sys.addaudithook(audit)
    longrun_path = Path(parent["longrun_result"]["path"])
    common._verify_identity(longrun_path, parent["longrun_result"], label="longrun result")
    longrun = json.loads(longrun_path.read_text())
    upstream = Path(protocol["upstream_root"])
    source._assert_upstream(upstream, longrun["upstream"]["commit"])
    write(out / "preflight.json", {
        "protocol": protocol, "protocol_sha256": sha256_file(args.protocol),
        "parent_protocol_sha256": sha256_file(parent_path),
        "manifest_sha256": sha256_file(manifest_path),
        "implementation_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "longrun_result_sha256": sha256_file(longrun_path),
        "source_test_opened": False, "dlo2_read": False, "official_eval_read": False,
        "python": sys.version, "numpy": np.__version__, "host": platform.node(),
        "split": split,
    })
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    source._seed_everything(torch, 42)
    modules = source._load_upstream(upstream)
    state = posterior._checkpoint_states(longrun, {6400}, torch=torch)[6400]
    if longrun["selected_checkpoint"]["checkpoint"]["sha256"] != parent["baseline"]["checkpoint_sha256"]:
        raise ValueError("baseline checkpoint changed")

    def load(stage: str) -> tuple:
        trajectories = source._load_named_trajectories(manifest, split[stage], frame_count=500, node_count=13)
        rollout = common._rollout(state, trajectories, modules=modules, torch=torch, device="cuda:0")
        initial, action = deform_causal_inputs(np.stack([trajectories[n] for n in split[stage]]))
        return initial, action, np.asarray(rollout["predictions"]), np.asarray(rollout["targets"])

    print("GP_PHASE fit-validation baseline replay", flush=True)
    fi, fa, fb, fy = load("fit")
    vi, va, vb, vy = load("validation")
    common._require_baseline_reproduction(float(errors(vb, vy).mean()), expected=parent["baseline"]["validation_l1_m"],
                                         tolerance=parent["baseline"]["reproduction_tolerance_m"], stage="validation")
    fi, fa, fb, fy, groups = _collapse_duplicate_queries(fi, fa, fb, fy, split["fit"])
    fit_names = [group[0] for group in groups]
    features, frames = build_deform_local_residual_features(fi, fa, fb)
    vf, vframes = build_deform_local_residual_features(vi, va, vb)
    floor = parent["posterior"]["coordinate_variance_floor_m2"]
    ridge_models, validation = {}, {}
    specs, predictions = {}, {}
    for ridge in protocol["ridge_ridges"]:
        model = fit_deform_local_residual(fi, fa, fb, fy, fit_names, ridge=ridge, variance_floor_m2=floor)
        ridge_models[ridge] = model
        for shrinkage in protocol["ridge_shrinkages"]:
            key = f"ridge-r{ridge:g}-s{shrinkage:g}"
            p = predict_deform_local_residual(model, vi, va, vb, shrinkage=shrinkage)["predictions"]
            validation[key] = float(errors(p, vy).mean())
            specs[key] = {"family": "ridge", "ridge": ridge, "shrinkage": shrinkage}
    tuned_ridge = min(validation, key=lambda k: (validation[k], k))
    fixed_model = ridge_models[protocol["fixed_ridge"]]
    fixed_fit = predict_deform_local_residual(fixed_model, fi, fa, fb, shrinkage=protocol["fixed_shrinkage"])["predictions"]
    fixed_val = predict_deform_local_residual(fixed_model, vi, va, vb, shrinkage=protocol["fixed_shrinkage"])["predictions"]
    raw = np.einsum("ntvi,nij->ntvj", fy - fb, frames)[:, :, 2:-2]
    remaining = np.einsum("ntvi,nij->ntvj", fy - fixed_fit, frames)[:, :, 2:-2]
    response = np.concatenate((raw, remaining), axis=-1)
    n, horizon, nodes, dim = features.shape
    arc = np.linspace(-1, 1, fb.shape[2])[2:-2]
    gp_models = {}
    for coupled in (False, True):
        family = "spatial" if coupled else "independent"
        for multiplier in protocol["length_multipliers"]:
            print(f"GP_PHASE fit {family} length={multiplier}", flush=True)
            banks = []
            for node in ([None] if coupled else range(nodes)):
                x = features.reshape(-1, dim) if coupled else features[:, :, node].reshape(-1, dim)
                y = response.reshape(-1, 6) if coupled else response[:, :, node].reshape(-1, 6)
                a = np.broadcast_to(arc, (n, horizon, nodes)).reshape(-1) if coupled else None
                count = protocol["spatial_anchors"] if coupled else protocol["independent_anchors"]
                indices = balanced_anchor_indices(n, horizon, nodes if coupled else 1, count)
                bank = fit_gp_bank(x, y, arc=a, anchor_indices=indices, length_multiplier=multiplier,
                                   noises=protocol["noise_levels"], row_weight=1.0 / horizon,
                                   nugget=protocol["explicit_nystrom_nugget"], material_length=protocol["material_length"])
                banks.append(bank)
            for noise in protocol["noise_levels"]:
                models = [bank[noise] for bank in banks]
                model_key = f"{family}-l{multiplier:g}-n{noise:g}"
                gp_models[model_key] = models
                if coupled:
                    a = np.broadcast_to(arc, vf.shape[:3]).reshape(-1)
                    pred = models[0].predict(vf.reshape(-1, dim), a).reshape(*vf.shape[:3], 6)
                else:
                    pred = np.stack([m.predict(vf[:, :, j].reshape(-1, dim)).reshape(*vf.shape[:2], 6)
                                     for j, m in enumerate(models)], axis=2)
                for augmented in (False, True):
                    residual = pred[..., 3:] if augmented else pred[..., :3]
                    base = fixed_val if augmented else vb
                    label = "ridge_plus_" + family if augmented else family
                    for shrinkage in protocol["shrinkages"]:
                        key = f"{label}-l{multiplier:g}-n{noise:g}-s{shrinkage:g}"
                        p = add_residual(base, residual, vframes, shrinkage)
                        validation[key] = float(errors(p, vy).mean())
                        specs[key] = {"family": label, "model": model_key, "shrinkage": shrinkage,
                                      "augmented": augmented, "coupled": coupled}
    families = ["independent", "spatial", "ridge_plus_independent", "ridge_plus_spatial"]
    selected = {family: min((k for k in specs if specs[k]["family"] == family), key=lambda k: (validation[k], k))
                for family in families}
    champion = min(selected.values(), key=lambda k: (validation[k], k))
    selection = {"selected": selected, "champion": champion, "tuned_ridge": tuned_ridge,
                 "validation_mean_l1_m": validation, "specifications": specs,
                 "fit_causal_query_clusters": [list(g) for g in groups], "source_test_opened": False}
    write(out / "selection.json", selection)
    for key in set(specs[k]["model"] for k in selected.values()):
        for index, model in enumerate(gp_models[key]):
            np.savez_compressed(out / f"model-{key}-{index}.npz", **model.archive())
    print("GP_SELECTION " + json.dumps({"champion": champion, "selected": selected, "tuned_ridge": tuned_ridge}), flush=True)
    phase["source_open"] = True
    si, sa, sb, sy = load("source_test")
    common._require_baseline_reproduction(float(errors(sb, sy).mean()), expected=parent["baseline"]["source_test_l1_m"],
                                         tolerance=parent["baseline"]["reproduction_tolerance_m"], stage="historical-source")
    sf, sframes = build_deform_local_residual_features(si, sa, sb)
    fixed_source = predict_deform_local_residual(fixed_model, si, sa, sb, shrinkage=protocol["fixed_shrinkage"])["predictions"]
    spec = specs[tuned_ridge]
    tuned_source = predict_deform_local_residual(ridge_models[spec["ridge"]], si, sa, sb, shrinkage=spec["shrinkage"])["predictions"]
    predictions.update(baseline=sb, fixed_ridge=fixed_source, tuned_ridge=tuned_source)
    for family, key in selected.items():
        spec = specs[key]
        models = gp_models[spec["model"]]
        if spec["coupled"]:
            a = np.broadcast_to(arc, sf.shape[:3]).reshape(-1)
            pred = models[0].predict(sf.reshape(-1, dim), a).reshape(*sf.shape[:3], 6)
        else:
            pred = np.stack([m.predict(sf[:, :, j].reshape(-1, dim)).reshape(*sf.shape[:2], 6)
                             for j, m in enumerate(models)], axis=2)
        residual = pred[..., 3:] if spec["augmented"] else pred[..., :3]
        predictions[family] = add_residual(fixed_source if spec["augmented"] else sb, residual, sframes, spec["shrinkage"])
    baseline_errors, ridge_errors = errors(sb, sy), errors(tuned_source, sy)
    metrics = {}
    for key, p in predictions.items():
        if not np.array_equal(p[:, :, [0, 1, -2, -1]], sb[:, :, [0, 1, -2, -1]]):
            raise RuntimeError("clamped node correction")
        case_errors = errors(p, sy)
        metrics[key] = {
            "mean_coordinate_l1_mm": float(1000 * case_errors.mean()),
            "relative_gain_vs_baseline": float(1 - case_errors.mean() / baseline_errors.mean()),
            "relative_gain_vs_tuned_ridge": float(1 - case_errors.mean() / ridge_errors.mean()),
            "wins_vs_tuned_ridge": int(np.sum(case_errors < ridge_errors)),
            "maximum_case_ratio_vs_tuned_ridge": float(np.max(case_errors / ridge_errors)),
            "case_l1_mm": dict(zip(split["source_test"], (1000 * case_errors).tolist(), strict=True)),
            "horizon_bin_l1_mm": [float(1000 * np.abs(p[:, indices] - sy[:, indices]).mean())
                                  for indices in np.array_split(np.arange(horizon), 5)],
        }
    best = metrics[specs[champion]["family"]]
    worthwhile = (best["relative_gain_vs_tuned_ridge"] >= protocol["continuation_minimum_relative_gain_vs_tuned_ridge"]
                  and best["wins_vs_tuned_ridge"] >= protocol["continuation_minimum_paired_wins"]
                  and best["maximum_case_ratio_vs_tuned_ridge"] <= protocol["continuation_maximum_case_ratio"])
    np.savez_compressed(out / "source_predictions.npz", names=np.asarray(split["source_test"]), targets=sy, **predictions)
    result = {
        "contract": protocol["contract"], "claim_boundary": protocol["claim_boundary"],
        "validation_selected_gp": champion, "historical_source_metrics": metrics,
        "continuation_gate_passed": bool(worthwhile), "source_trajectories": len(sb),
        "preflight_sha256": sha256_file(out / "preflight.json"),
        "selection_sha256": sha256_file(out / "selection.json"),
        "source_predictions_sha256": sha256_file(out / "source_predictions.npz"),
        "raw_data_files_read": sorted(accessed), "dlo2_read": False,
        "official_eval_read": False, "pokeflex_read": False,
        "paper_claim_authorized": False, "uncertainty_calibration_tested": False,
        "elapsed_seconds": time.monotonic() - start, "torch": torch.__version__,
        "implementation_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    write(out / "result.json", result)
    print("GP_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("GP_TECHNICAL_FAILURE " + repr(error), flush=True)
        raise
