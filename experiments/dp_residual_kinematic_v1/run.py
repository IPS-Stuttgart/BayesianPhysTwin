"""Exploratory held-trajectory comparison; no hardware or new target opening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.dp_residual_kinematic_v1.model import (  # noqa: E402 -- source-tree bootstrap
    GateSpec,
    ResidualExperts,
)

SEEDS = (7, 23, 61)
RIDGES = (1.0, 100.0)
SHRINKAGES = (0.0, 0.125, 0.25, 0.5, 1.0)
SPECS = (
    GateSpec("single", 1),
    *(GateSpec("finite", k) for k in (2, 4, 8)),
    *(GateSpec("finite_bayes", k) for k in (2, 4, 8)),
    *(GateSpec("dp", 8, a) for a in (0.1, 1.0, 10.0)),
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def correction_prediction(baseline, canonical, frames, shrinkage):
    result = baseline.copy()
    if shrinkage:
        result[:, :, 2:-2] += shrinkage * np.einsum("ntvj,nij->ntvi", canonical, frames)
    np.testing.assert_array_equal(
        result[:, :, [0, 1, -2, -1]], baseline[:, :, [0, 1, -2, -1]]
    )
    return result


def case_error(prediction, truth):
    return np.mean(np.abs(prediction - truth), axis=(1, 2, 3)) * 1000


def grouping(initial, action, baseline):
    groups = {}
    for index in range(len(initial)):
        h = hashlib.sha256()
        for array in (initial[index], action[index], baseline[index]):
            h.update(np.ascontiguousarray(array).tobytes())
        groups.setdefault(h.hexdigest(), []).append(index)
    return groups


def bootstrap_difference(candidate, reference):
    difference = np.asarray(candidate) - np.asarray(reference)
    rng = np.random.default_rng(1943)
    samples = difference[
        rng.integers(0, len(difference), size=(5000, len(difference)))
    ].mean(1)
    return {
        "mean_difference_mm": float(difference.mean()),
        "paired_bootstrap_95_percent_mm": np.quantile(samples, [0.025, 0.975]).tolist(),
        "wins": int(np.sum(difference < -1e-10)),
        "ties": int(np.sum(abs(difference) <= 1e-10)),
        "cases": len(difference),
        "interpretation": "Descriptive pilot interval; old validation trajectories, one object",
    }


def fit_predict(spec, ridge, train_x, train_y, query_x, *, hard=False):
    predictions, diagnostics = [], []
    for seed in SEEDS if spec.components > 1 else SEEDS[:1]:
        model = ResidualExperts(spec, ridge, seed).fit(train_x, train_y)
        predictions.append(model.predict(query_x, hard=hard))
        diagnostics.append(model.diagnostics())
    return np.mean(predictions, axis=0), diagnostics, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    metadata = json.loads(args.preparation.read_text())
    if digest(args.bundle) != metadata["bundle_sha256"]:
        raise RuntimeError("Prepared data hash mismatch")
    if metadata["contract"] != "dp-residual-dlo1-development-bundle-v1":
        raise RuntimeError("Unrecognized development data contract")
    with np.load(args.bundle, allow_pickle=False) as z:
        names, split = z["names"].astype(str), z["split"].astype(str)
        initial, action = (
            z["initial"].astype(np.float64),
            z["action"].astype(np.float64),
        )
        baseline = z["baseline"].astype(np.float64)
        truth = z["targets"].astype(np.float64)
        persistence = z["persistence"].astype(np.float64)
    if list(split) != ["fit"] * 40 + ["validation"] * 8:
        raise RuntimeError("Unexpected data split")
    for array in (initial, action, baseline, truth, persistence):
        if not np.isfinite(array).all():
            raise RuntimeError("Nonfinite input bundle")
    groups = grouping(initial, action, baseline)
    overlap = [
        v for v in groups.values() if any(i < 40 for i in v) and any(i >= 40 for i in v)
    ]
    if overlap:
        write_json(
            out / "blocked.json",
            {
                "reason": "Exact causal queries cross historical fit/validation",
                "groups": overlap,
            },
        )
        raise RuntimeError(
            "Cross-split duplicate causal queries; cannot score a held-trajectory pilot"
        )
    fit_groups = [(k, v) for k, v in sorted(groups.items()) if v[0] < 40]
    held_groups = [(k, v) for k, v in sorted(groups.items()) if v[0] >= 40]
    fit_ids = [v[0] for _, v in fit_groups]
    held_ids = [v[0] for _, v in held_groups]
    if len(fit_ids) < 20:
        raise RuntimeError("Fewer than twenty distinct fit trajectory groups")
    n_tune = max(4, len(fit_ids) // 5)
    tune_ids, subfit_ids = fit_ids[:n_tune], fit_ids[n_tune:]
    truth = truth.copy()
    for _, indices in fit_groups:
        truth[indices[0]] = truth[indices].mean(axis=0)
    from bayesian_phystwin_experiments.deform_dlo_local_residual import (
        build_deform_local_residual_features,
        fit_deform_local_residual,
        predict_deform_local_residual,
    )

    features, frames = build_deform_local_residual_features(initial, action, baseline)
    residual = np.einsum("ntvi,nij->ntvj", truth - baseline, frames)[:, :, 2:-2]
    protocol = {
        "contract": "dp-residual-dlo1-pilot-v1",
        "seeds": SEEDS,
        "ridges": RIDGES,
        "shrinkages": SHRINKAGES,
        "gate_specs": [asdict(s) for s in SPECS],
        "gate_training": "16 equally spaced causal kinematic feature summaries per fit trajectory; shared across nodes",
        "gate_covariance": "full",
        "gate_pca_dimension_cap": 8,
        "expert": "nodewise ridge corrections partially pooled toward global ridge",
        "selection": "mean coordinate L1 on hash-assigned tuning groups inside original fit split",
        "subfit_names": names[subfit_ids].tolist(),
        "tuning_names": names[tune_ids].tolist(),
        "refit_names": names[fit_ids].tolist(),
        "held_validation_names": names[held_ids].tolist(),
        "full_fit_count_before_deduplication": 40,
        "full_validation_count_before_deduplication": 8,
        "bundle_sha256": metadata["bundle_sha256"],
        "commit": os.environ.get("GITHUB_SHA"),
        "claim_boundary": "Exploratory reuse of historical DLO1 development; no fresh confirmation or full sticky-HDP claim",
        "official_eval_read": False,
        "source_test_read": False,
        "held_validation_used_for_selection": False,
    }
    write_json(out / "protocol.json", protocol)
    bank = []
    for spec in SPECS:
        for ridge in RIDGES:
            pred, diagnostics, _ = fit_predict(
                spec,
                ridge,
                features[subfit_ids],
                residual[subfit_ids],
                features[tune_ids],
            )
            for shrinkage in SHRINKAGES:
                prediction = correction_prediction(
                    baseline[tune_ids], pred, frames[tune_ids], shrinkage
                )
                error = case_error(prediction, truth[tune_ids])
                bank.append(
                    {
                        "spec": asdict(spec),
                        "key": spec.key,
                        "ridge": ridge,
                        "shrinkage": shrinkage,
                        "tuning_mean_l1_mm": float(error.mean()),
                        "tuning_case_l1_mm": error.tolist(),
                        "diagnostics": diagnostics,
                    }
                )
            print(f"TUNING {spec.key} ridge={ridge:g} complete", flush=True)
    selection = {}
    for family in ("single", "finite", "finite_bayes", "dp"):
        eligible = [a for a in bank if a["spec"]["family"] == family]
        selection[family] = min(
            eligible,
            key=lambda a: (
                a["tuning_mean_l1_mm"],
                a["spec"]["components"],
                a["shrinkage"],
                a["ridge"],
                a["key"],
            ),
        )
    write_json(
        out / "selection_seal.json",
        {
            "selection": selection,
            "bank": bank,
            "protocol_sha256": digest(out / "protocol.json"),
            "validation_scored": False,
        },
    )
    predictions = {
        "hybrid": baseline[held_ids].copy(),
        "action_persistence": persistence[held_ids].copy(),
    }
    # At this initial-state cutoff the permitted residual is zero: do not use a
    # forbidden future innovation to make last-residual a stronger comparator.
    predictions["last_residual_initial_zero"] = baseline[held_ids].copy()
    fitted = {}
    seed_predictions = {}
    for family, chosen in selection.items():
        spec = GateSpec(**chosen["spec"])
        pred, diagnostics, individual = fit_predict(
            spec,
            chosen["ridge"],
            features[fit_ids],
            residual[fit_ids],
            features[held_ids],
        )
        predictions[family] = correction_prediction(
            baseline[held_ids], pred, frames[held_ids], chosen["shrinkage"]
        )
        fitted[family] = diagnostics
        seed_predictions[family] = [
            correction_prediction(
                baseline[held_ids], p, frames[held_ids], chosen["shrinkage"]
            )
            for p in individual
        ]
    old = fit_deform_local_residual(
        initial[fit_ids],
        action[fit_ids],
        baseline[fit_ids],
        truth[fit_ids],
        names[fit_ids].tolist(),
        ridge=1.0,
        variance_floor_m2=1e-6,
    )
    predictions["existing_local_r1_s0p5"] = predict_deform_local_residual(
        old, initial[held_ids], action[held_ids], baseline[held_ids], shrinkage=0.5
    )["predictions"]
    single_check = ResidualExperts(GateSpec("single", 1), 1).fit(
        features[fit_ids], residual[fit_ids]
    )
    equivalent = correction_prediction(
        baseline[held_ids],
        single_check.predict(features[held_ids]),
        frames[held_ids],
        0.5,
    )
    parity = float(np.max(np.abs(equivalent - predictions["existing_local_r1_s0p5"])))
    if parity > 1e-7:
        raise RuntimeError(f"Single expert differs from production baseline: {parity}")
    prediction_path = out / "sealed_predictions.npz"
    np.savez_compressed(prediction_path, names=names[held_ids], **predictions)
    write_json(
        out / "prediction_seal.json",
        {
            "sha256": digest(prediction_path),
            "selection_sha256": digest(out / "selection_seal.json"),
            "validation_scored": False,
        },
    )
    scores = {}
    for key, prediction in predictions.items():
        error = case_error(prediction, truth[held_ids])
        scores[key] = {
            "mean_l1_mm": float(error.mean()),
            "case_l1_mm": error.tolist(),
            "prefix_horizon_mean_l1_mm": {
                str(h): float(case_error(prediction[:, :h], truth[held_ids, :h]).mean())
                for h in (50, 150, 498)
            },
        }
    differences = {
        ref: bootstrap_difference(scores["dp"]["case_l1_mm"], scores[ref]["case_l1_mm"])
        for ref in (
            "hybrid",
            "existing_local_r1_s0p5",
            "single",
            "finite",
            "finite_bayes",
        )
    }
    references = [
        scores[k]["mean_l1_mm"]
        for k in ("existing_local_r1_s0p5", "single", "finite", "finite_bayes")
    ]
    dp_better = scores["dp"]["mean_l1_mm"] < 0.99 * min(references)
    result = {
        "contract": "dp-residual-dlo1-pilot-result-v1",
        "preparation": metadata,
        "protocol": protocol,
        "selection": selection,
        "fitted_gate_diagnostics": fitted,
        "scores": scores,
        "dp_paired_differences": differences,
        "seed_case_errors_mm": {
            f: [case_error(p, truth[held_ids]).tolist() for p in pp]
            for f, pp in seed_predictions.items()
        },
        "single_expert_production_parity_max_abs_m": parity,
        "dp_improves_all_strong_references_by_one_percent": dp_better,
        "prediction_sha256": digest(prediction_path),
        "interpretation": (
            "Promising exploratory DP accuracy result; not confirmation"
            if dp_better
            else "This pilot does not establish an incremental DP accuracy benefit"
        ),
        "uncertainty_calibration_tested": False,
        "sticky_hdp_tested": False,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "result.json", result)
    lines = [
        "# DP residual DLO1 development pilot",
        "",
        protocol["claim_boundary"],
        "",
        "| Model | Mean coordinate L1 (mm) |",
        "|---|---:|",
    ]
    lines += [f"| {key} | {value['mean_l1_mm']:.6f} |" for key, value in scores.items()]
    lines += [
        "",
        result["interpretation"],
        "",
        "Tuning used only whole trajectory groups inside the original fit split.",
        "All methods use two initial states, supplied clamp trajectories and the same hybrid rollout.",
        "At this initial-state cutoff residual persistence is the unchanged hybrid (zero initial discrepancy).",
        "This is a truncated DP gate with ridge experts, not a full sticky HDP, physical regime discovery, or calibration evidence.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "scores": scores,
                "dp_paired_differences": differences,
                "dp_improves_all_strong_references_by_one_percent": dp_better,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
