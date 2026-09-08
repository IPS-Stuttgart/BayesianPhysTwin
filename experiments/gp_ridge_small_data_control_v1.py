#!/usr/bin/env python3
"""Retrospective control for the eight-recording GP signal from PR #955.

Only checksum-pinned DLO2 fit/validation rollout caches are read. No raw data,
source-test, official evaluation, physical retraining, or production change.
All three seeds are selected on inner recordings before any outer scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.remote.run_gp_residual_development_v1 import build_splits, metrics
from bayesian_phystwin_experiments.deform_dlo_local_residual import (
    build_deform_local_residual_features,
    fit_deform_local_residual,
    predict_deform_local_residual,
)
from bayesian_phystwin_experiments.gaussian_residual_v1 import (
    GaussianResidual,
    apply_canonical_correction,
)

CACHE_HASHES = {
    "fit": "c3641d203354d0a28b87d43fb3a0d17f1de85f860b8989ccf226219efec49379",
    "validation": "2f3c6e245f6246c805850fdaef9867025a9942e488b6e819a669b3f58f312ab4",
}
CHECKPOINT = "b64affff638c9d47ca51f17bb7124cc4bd224facd1f7137b0042b7fa9037ea65"
MANIFEST = "7c5501997e6bab7b0537ef9cda932ec19312e40618f03a4fad80ffc1622a6d98"
PLAN = {
    "contract": "gp-ridge-small-data-control-v1",
    "classification": "post-result retrospective development control, not fresh confirmation",
    "seeds": [7, 19, 41],
    "fit_recordings": 8,
    "inner_recordings": 8,
    "outer_recordings": 8,
    "ridge_penalties": [0.01, 0.1, 1.0, 10.0, 100.0],
    "strengths": [0.0, 0.125, 0.25, 0.5, 1.0],
    "gp_lengths": [0.5, 1.0, 2.0],
    "gp_noise_variances": [0.03, 0.3],
    "gp_support_rows_per_recording": 8,
    "selection": "minimum inner-validation coordinate L1, first candidate breaks ties",
    "primary_comparator": "ridge with inner-selected penalty and strength",
    "row_budget_boundary": "GP uses eight frames per fit recording; ridge uses all permitted fit frames",
    "new_data_collection": False,
    "source_test_read": False,
    "official_eval_read": False,
    "backbone_retrained": False,
    "prior_gp_grid_changed": False,
    "cache_hashes": CACHE_HASHES,
    "checkpoint_sha256": CHECKPOINT,
    "manifest_sha256": MANIFEST,
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def choose(candidates: list, inner_truth: np.ndarray) -> tuple:
    if not candidates or not np.isfinite(inner_truth).all():
        raise ValueError("Invalid inner selection data")
    count = len(inner_truth)
    bank = []
    for specification, prediction in candidates:
        if prediction.shape[1:] != inner_truth.shape[1:] or len(prediction) < count:
            raise ValueError("Unaligned candidate")
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite candidate")
        loss = float(np.abs(prediction[:count] - inner_truth).mean() * 1000)
        bank.append({"parameters": specification, "inner_l1_mm": loss})
    index = min(range(len(bank)), key=lambda i: bank[i]["inner_l1_mm"])
    return bank[index], candidates[index][1], bank


def self_test() -> None:
    truth = np.zeros((2, 3, 4, 3))
    first = np.concatenate((truth + 1, truth + 1000))
    second = np.concatenate((truth + 2, truth))
    selected = choose([({"id": 1}, first), ({"id": 2}, second)], truth)[0]
    assert selected["parameters"]["id"] == 1
    first[2:] = -1e6
    second[2:] = 1e6
    assert choose([({"id": 1}, first), ({"id": 2}, second)], truth)[0] == selected
    try:
        choose([({}, np.full_like(first, np.nan))], truth)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite candidate accepted")
    print("PASS: outer values do not select parameters; nonfinite values fail")


def load_cache(cache: Path) -> tuple:
    parts = []
    receipts = {}
    for split, count in (("fit", 40), ("validation", 8)):
        path = cache / f"{split}.npz"
        receipt_path = cache / f"{split}.json"
        receipt = json.loads(receipt_path.read_text())
        if sha(path) != CACHE_HASHES[split]:
            raise ValueError(f"Unrecognized cache bytes: {split}")
        if (receipt["sha256"] != CACHE_HASHES[split]
                or receipt["checkpoint_sha256"] != CHECKPOINT
                or receipt["manifest_sha256"] != MANIFEST
                or receipt["official_eval_read"] is not False):
            raise ValueError(f"Cache provenance mismatch: {split}")
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in (
                "initial", "action", "predictions", "targets", "names"
            )}
        if data["names"].tolist() != receipt["names"] or len(data["names"]) != count:
            raise ValueError("Cache roster mismatch")
        expected_shapes = {
            "initial": (count, 2, 12, 3), "action": (count, 498, 4, 3),
            "predictions": (count, 498, 12, 3), "targets": (count, 498, 12, 3),
        }
        for key, shape in expected_shapes.items():
            data[key] = np.asarray(data[key], dtype=np.float64)
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError(f"Invalid cache geometry: {key}")
        baseline_error = float(np.abs(data["predictions"] - data["targets"]).mean())
        if abs(baseline_error - receipt["baseline_l1_m"]) > 1e-8:
            raise ValueError("Cache baseline parity failed")
        parts.append(data)
        receipts[split] = {"receipt": receipt, "receipt_sha256": sha(receipt_path)}
    arrays = {
        key: np.concatenate([part[key] for part in parts])
        for key in parts[0]
    }
    names = arrays["names"].tolist()
    if len(set(names)) != 48:
        raise ValueError("Non-disjoint recording roster")
    queries = [hashlib.sha256(b"".join(arrays[key][i].tobytes() for key in (
        "initial", "action", "predictions"
    ))).hexdigest() for i in range(48)]
    if len(set(queries)) != 48:
        raise ValueError("Duplicated causal query")
    return arrays, receipts


def run(cache: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(output / "plan_before_data.json", dict(PLAN, source_revision=os.getenv("EXPERIMENT_REVISION")))
    arrays, receipts = load_cache(cache)
    initial, action = arrays["initial"], arrays["action"]
    baseline, targets, names = arrays["predictions"], arrays["targets"], arrays["names"].tolist()
    features, frames = build_deform_local_residual_features(initial, action, baseline)
    residual = np.einsum("ntvi,nij->ntvj", targets[:40] - baseline[:40], frames[:40])[:, :, 2:-2]
    selections, predictions = [], {}
    for seed in PLAN["seeds"]:
        train, inner, outer = build_splits(names, seed, 8)
        query = np.concatenate((inner, outer))
        candidates = {"ridge_strength_only": [], "ridge_penalty_and_strength": [], "gp": []}
        print(f"FIT seed={seed}; fixed eight-recording budget", flush=True)
        for penalty in PLAN["ridge_penalties"]:
            model = fit_deform_local_residual(
                initial[train], action[train], baseline[train], targets[train],
                [names[i] for i in train], ridge=penalty, variance_floor_m2=1e-6,
            )
            full = predict_deform_local_residual(
                model, initial[query], action[query], baseline[query], shrinkage=1.0,
            )["predictions"]
            for strength in PLAN["strengths"]:
                pred = baseline[query] + strength * (full - baseline[query])
                item = ({"ridge": penalty, "strength": strength}, pred)
                candidates["ridge_penalty_and_strength"].append(item)
                if penalty == 1.0:
                    candidates["ridge_strength_only"].append(item)
        groups = np.repeat(np.asarray([names[i] for i in train]), 498)
        for length in PLAN["gp_lengths"]:
            for noise in PLAN["gp_noise_variances"]:
                corrections = []
                for node in range(features.shape[2]):
                    model = GaussianResidual.fit(
                        features[train, :, node].reshape(-1, features.shape[-1]),
                        residual[train, :, node].reshape(-1, 3), groups,
                        length_scale=length, noise_variance=noise, per_group=8,
                    )
                    mean = model.predict_mean(features[query, :, node].reshape(-1, features.shape[-1]))
                    corrections.append(mean.reshape(len(query), 498, 3))
                correction = np.stack(corrections, axis=2)
                for strength in PLAN["strengths"]:
                    pred = apply_canonical_correction(baseline[query], correction, frames[query], strength)
                    candidates["gp"].append(({
                        "length_scale": length, "noise_variance": noise, "strength": strength,
                    }, pred))
        record = {"seed": seed, "train_names": [names[i] for i in train],
                  "inner_names": [names[i] for i in inner], "outer_names": [names[i] for i in outer],
                  "selection": {}}
        for family, bank in candidates.items():
            selected, pred, inner_bank = choose(bank, targets[inner])
            if not np.array_equal(pred[:, :, [0, 1, 10, 11]], baseline[query][:, :, [0, 1, 10, 11]]):
                raise ValueError("Clamped node changed")
            predictions[f"seed{seed}_{family}"] = pred[len(inner):]
            record["selection"][family] = {"selected": selected, "inner_bank": inner_bank}
        selections.append(record)
    # Seal ALL seeds before outer scoring. The cache parity check above reads
    # historical validation truth, but it is never supplied to parameter selection.
    np.savez_compressed(output / "selected_predictions.npz", **predictions)
    write_json(output / "selection_before_scoring.json", {
        "seeds": selections, "prediction_sha256": sha(output / "selected_predictions.npz"),
        "outer_used_for_selection": False,
    })
    scored = []
    for record in selections:
        seed = record["seed"]
        scored.append({"seed": seed, "scores": {
            family: metrics(predictions[f"seed{seed}_{family}"], targets[40:])
            for family in record["selection"]
        }})
    np.savez_compressed(output / "evaluation_targets.npz", targets=targets[40:], baseline=baseline[40:], names=arrays["names"][40:])
    means = {family: float(np.mean([record["scores"][family]["coordinate_l1_mm"] for record in scored]))
             for family in ("ridge_strength_only", "ridge_penalty_and_strength", "gp")}
    result = {
        "contract": PLAN["contract"], "status": "completed-retrospective-control",
        "source_revision": os.getenv("EXPERIMENT_REVISION"),
        "workflow_run_id": os.getenv("GITHUB_RUN_ID"), "numpy": np.__version__,
        "plan_sha256": sha(output / "plan_before_data.json"),
        "selection_sha256": sha(output / "selection_before_scoring.json"),
        "prediction_sha256": sha(output / "selected_predictions.npz"),
        "targets_sha256": sha(output / "evaluation_targets.npz"),
        "receipts": receipts, "seed_results": scored, "mean_errors_mm": means,
        "gp_gain_vs_strength_only_percent": 100 * (1 - means["gp"] / means["ridge_strength_only"]),
        "gp_gain_vs_penalty_selected_percent": 100 * (1 - means["gp"] / means["ridge_penalty_and_strength"]),
        "source_test_read": False, "official_eval_read": False, "backbone_retrained": False,
        "fresh_confirmation": False, "paper_claim_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("mean_errors_mm", "gp_gain_vs_strength_only_percent", "gp_gain_vs_penalty_selected_percent")}, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.cache is None or args.output is None:
        parser.error("--cache and --output are required")
    run(args.cache, args.output)


if __name__ == "__main__":
    main()
