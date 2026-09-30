"""Five-outer/four-inner recording cross-fit, prediction seals, and source gate.

Only fit_queries.npz is opened in Study One. The original validation archive,
raw recordings and official evaluation are not arguments to this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .conditioning import (
    GRID,
    calibrate,
    covariance_shapes,
    fit_models,
    forecast_pair,
    predict_open,
    prefix_indices,
    sandwich_operator,
    world_forecast,
)
from .controls import (
    INDEPENDENT,
    apply_probability_calibration,
    choose_settings,
    context_features,
    fit_probability_calibration,
    predict_empirical,
)
from .risk import TemporalBlocks, deploy, forecast_risk, ranked_acceptance

SEED = 20260907
ENDPOINTS = (50, 150, 250)
HORIZONS = (100, 25, 200)
MANIFEST_SHA256 = "ff24ae708753febd0282c77c3f310a7ead21e981f2dd8464ee35ed0bb8d8c2dc"
FIT_SHA256 = "b99f34a577f8616c4873df846866a01edd44dc9afd37cda68eeee3330a18cfa8"
METHODS = ("posterior", "matched_distribution", *INDEPENDENT)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )


def source_identity(root):
    files = sorted(Path(root).glob("*.py")) + [Path(root) / "PROTOCOL.md"]
    return {str(p.name): sha256(p) for p in files}


def load_source(cache):
    cache = Path(cache)
    if sha256(cache / "cache_manifest.json") != MANIFEST_SHA256:
        raise ValueError("source manifest checksum differs")
    manifest = json.loads((cache / "cache_manifest.json").read_text())
    if (
        manifest["contract"] != "deform-gp-v2-development-query-cache-v1"
        or manifest["source_test_opened"] is not False
        or manifest["official_eval_read"] is not False
    ):
        raise ValueError("unapproved source cache")
    (entry,) = [r for r in manifest["files"] if r["label"] == "fit"]
    if entry["file"] != "fit_queries.npz" or entry["sha256"] != FIT_SHA256:
        raise ValueError("fit-cache mapping differs")
    if sha256(cache / "fit_queries.npz") != FIT_SHA256:
        raise ValueError("fit-cache checksum differs")
    with np.load(cache / "fit_queries.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if data["names"].tolist() != entry["names"] or len(set(entry["names"])) != 40:
        raise ValueError("source membership differs")
    if data["features"].shape != (40, 498, 8, 92):
        raise ValueError("source feature dimensions differ")
    for key, array in data.items():
        if key != "names" and not np.isfinite(array).all():
            raise ValueError(f"nonfinite source array: {key}")
    keys = [
        hashlib.sha256(
            b"".join(
                np.ascontiguousarray(data[k][i]).tobytes()
                for k in ("initial", "action", "baseline")
            )
        ).hexdigest()
        for i in range(40)
    ]
    if len(set(keys)) != 40:
        raise ValueError("duplicate recording inputs require a new grouped protocol")
    frames = data["frames"]
    if not np.allclose(frames.swapaxes(1, 2) @ frames, np.eye(3), atol=1e-10):
        raise ValueError("local frames are not orthonormal")
    # The cached future clamped nodes are input boundaries, not observations.
    if data["action"].shape != data["baseline"][:, :, [0, 1, -2, -1]].shape:
        raise ValueError("clamped-node/action dimensions differ")
    return data


def folds(indices, count, salt):
    indices = np.asarray(indices, dtype=int)
    if len(indices) % count:
        raise ValueError("recording folds must have equal size")
    shuffled = np.random.default_rng(SEED + salt).permutation(indices)
    return [np.sort(a) for a in np.split(shuffled, count)]


class ModelBank:
    def __init__(self, data):
        self.data = data
        self.models = {}
        self.lineage = []

    def get(self, indices):
        key = tuple(sorted(map(int, indices)))
        if key not in self.models:
            d = self.data
            self.models[key] = fit_models(
                d["features"][list(key)],
                d["local_target"][list(key)],
                d["names"][list(key)],
            )
            self.lineage.append(
                {
                    "fit_recordings": d["names"][list(key)].tolist(),
                    "rows_per_recording": 498,
                    "anchors_per_node": 128,
                }
            )
            print("FIT", len(key), len(self.models), flush=True)
        return self.models[key]

    def crossfit(self, indices, count, salt):
        data = self.data
        error = np.empty((len(indices), 498, 8, 3))
        shapes = np.empty((len(indices), 8, 3, len(GRID), len(GRID)))
        robust = np.empty_like(shapes)
        model_by_recording = {}
        positions = {int(v): k for k, v in enumerate(indices)}
        for held in folds(indices, count, salt):
            train = np.setdiff1d(indices, held)
            models = self.get(train)
            for i in held:
                k = positions[int(i)]
                error[k] = data["local_target"][i] - predict_open(
                    models, data["features"][i]
                )
                shapes[k] = covariance_shapes(models, data["features"][i], GRID)
                robust[k] = covariance_shapes(
                    models, data["features"][i], GRID, robust=True
                )
                model_by_recording[int(i)] = models
        return (
            error,
            calibrate(error, shapes),
            calibrate(error, robust),
            model_by_recording,
        )


def make_contexts(data, indices, models, parameters, sandwich_parameters, *, outcomes):
    result = []
    for i in indices:
        current_models = models[int(i)] if isinstance(models, dict) else models
        for endpoint in ENDPOINTS:
            prefix = data["local_target"][i, prefix_indices(endpoint)].copy()
            for horizon in HORIZONS:
                row = {
                    "recording": str(data["names"][i]),
                    "endpoint": endpoint,
                    "horizon": horizon,
                    "key": f"{data['names'][i]}:{endpoint}:{horizon}",
                }
                future = slice(endpoint, endpoint + horizon)
                try:
                    b, c, innovation, covariance, raw, robust = forecast_pair(
                        current_models,
                        parameters,
                        data["features"][i],
                        prefix,
                        endpoint,
                        horizon,
                    )
                    posterior = forecast_risk(b, c, c, covariance)
                    if not posterior.valid:
                        raise ValueError(posterior.failure)
                    raw_risk = forecast_risk(b, c, c, raw)
                    matched = forecast_risk(
                        b.copy(),
                        c.copy(),
                        c.copy(),
                        TemporalBlocks(covariance.blocks.copy()),
                    )
                    if matched != posterior:
                        raise AssertionError(
                            "equivalent predictive distribution changed risk"
                        )
                    sandwich = forecast_risk(
                        b,
                        c,
                        c,
                        sandwich_operator(robust, sandwich_parameters, innovation),
                    )
                    q = np.asarray([p["q_m2"] for p in parameters])
                    constant = TemporalBlocks(q[:, :, None, None] * np.eye(horizon))
                    constant_risk = forecast_risk(b, c, c, constant)
                    if (
                        not raw_risk.valid
                        or not sandwich.valid
                        or not constant_risk.valid
                    ):
                        raise ValueError("invalid comparator covariance")
                    expected = float(posterior.regret_mean_mm2)
                    row.update(
                        {
                            "b": b,
                            "c": c,
                            "valid": True,
                            "raw": {
                                "posterior": posterior.harm_probability,
                                "matched_distribution": matched.harm_probability,
                                "constant_diagonal": constant_risk.harm_probability,
                                "sandwich": sandwich.harm_probability,
                                # Expected-regret control has the identical mean.
                                "expected_regret": float(expected > 1.0),
                            },
                            "expected_score": expected,
                            "posterior_uncalibrated": raw_risk.harm_probability,
                            "risk": posterior.as_dict(),
                            "variance_diag_m2": np.diagonal(
                                covariance.blocks, axis1=-2, axis2=-1
                            ).transpose(2, 0, 1),
                            "features": context_features(
                                b,
                                c,
                                innovation,
                                data["action"][i, future],
                                endpoint,
                                horizon,
                            ),
                        }
                    )
                except (ValueError, np.linalg.LinAlgError, RuntimeError) as failure:
                    # Retain technical failures with unchanged GP forecast; never replace.
                    b = predict_open(current_models, data["features"][i])[future]
                    row.update(
                        {
                            "b": b,
                            "c": b.copy(),
                            "valid": False,
                            "failure": str(failure),
                            "raw": dict.fromkeys(METHODS, 1.0),
                            "expected_score": 0.0,
                            "posterior_uncalibrated": 1.0,
                            "risk": {"valid": False, "failure": str(failure)},
                            "variance_diag_m2": np.full_like(b, 1e-10),
                            "features": context_features(
                                b,
                                b,
                                prefix
                                - predict_open(current_models, data["features"][i])[
                                    prefix_indices(endpoint)
                                ],
                                data["action"][i, future],
                                endpoint,
                                horizon,
                            ),
                        }
                    )
                if outcomes:
                    attach_outcome(row, data["local_target"][i, future])
                # Exact clamped-node handoff is tested on every generated pair.
                for forecast in (row["b"], row["c"]):
                    world_forecast(
                        data["baseline"][i, future], forecast, data["frames"][i]
                    )
                result.append(row)
    return result


def attach_outcome(row, truth):
    row["open_error"] = truth - row["b"]
    row["mse_b_mm2"] = float(np.mean((truth - row["b"]) ** 2) * 1e6)
    row["mse_c_mm2"] = float(np.mean((truth - row["c"]) ** 2) * 1e6)
    row["regret_mm2"] = row["mse_c_mm2"] - row["mse_b_mm2"]
    variance = row["variance_diag_m2"]
    z2 = (truth - row["c"]) ** 2 / variance
    row["uncertainty_scores"] = {
        "coordinate_coverage90": float(np.mean(z2 <= 1.6448536269514722**2)),
        "coordinate_nees": float(z2.mean()),
        "interval90_width_mm": float(
            np.mean(2 * 1.6448536269514722 * np.sqrt(variance)) * 1000
        ),
    }


def fit_selectors(training, panels):
    settings, validation = choose_settings(panels)
    empirical = predict_empirical(training, training, settings)
    # Calibration uses recording-OOF empirical predictions, not resubstitution.
    calibration_predictions = {m: [] for m in METHODS if m != "matched_distribution"}
    calibration_truth = []
    for train, held in panels:
        probabilities = predict_empirical(train, held, settings)
        for method in calibration_predictions:
            values = (
                probabilities[method]
                if method in probabilities
                else [r["raw"][method] for r in held]
            )
            calibration_predictions[method].extend(values)
        calibration_truth.extend(r["regret_mm2"] > 1 for r in held)
    fitted = {
        m: fit_probability_calibration(p, calibration_truth)
        for m, p in calibration_predictions.items()
    }
    # Calibrated distribution-match control remains bit-for-bit identical.
    fitted["matched_distribution"] = fitted["posterior"]
    return settings, fitted, validation, empirical


def apply_selectors(training, queries, settings, calibrators):
    empirical = predict_empirical(training, queries, settings)
    for method in METHODS:
        raw = (
            empirical[method]
            if method in empirical
            else np.asarray([r["raw"][method] for r in queries])
        )
        calibrated = apply_probability_calibration(raw, calibrators[method])
        for k, row in enumerate(queries):
            row.setdefault("probability", {})[method] = float(calibrated[k])
            row["raw"][method] = float(raw[k])
    for row in queries:
        # Rank the covariance-free control by its continuous expected regret.
        row["score"] = dict(row["probability"])
        row["score"]["expected_regret"] = row["expected_score"]
        if (
            row["probability"]["posterior"]
            != row["probability"]["matched_distribution"]
        ):
            raise AssertionError("matched-distribution parity failed")


def selections(rows, fraction):
    return {
        m: ranked_acceptance(
            [r["score"][m] for r in rows],
            [r["key"] for r in rows],
            [r["valid"] for r in rows],
            fraction,
        )
        for m in METHODS
    }


def summarize(rows, fraction=0.5, *, use_global_selection=False):
    accepted = (
        {m: np.asarray([r["accepted50"][m] for r in rows]) for m in METHODS}
        if use_global_selection
        else selections(rows, fraction)
    )
    truth = np.asarray([r["regret_mm2"] > 1 for r in rows])
    report = {}
    for m in METHODS:
        select = accepted[m]
        deployed = [
            r["mse_c_mm2"] if take else r["mse_b_mm2"]
            for r, take in zip(rows, select, strict=True)
        ]
        probabilities = np.asarray([r["probability"][m] for r in rows])
        raw = np.asarray([r["raw"][m] for r in rows])
        report[m] = {
            "accepted": int(select.sum()),
            "actual_acceptance": float(select.mean()),
            "harmful_accepted": int(np.sum(select & truth)),
            "brier": float(np.mean((probabilities - truth) ** 2)),
            "raw_brier": float(np.mean((raw - truth) ** 2)),
            "deployed_rmse_mm": float(np.sqrt(np.mean(deployed))),
            "fallback_count": int(np.sum(~select)),
        }
    return report


def source_gate(rows):
    primary = [r for r in rows if r["horizon"] == 100]
    report = summarize(primary)
    control = min(
        INDEPENDENT,
        key=lambda m: (report[m]["harmful_accepted"], report[m]["brier"], m),
    )
    reference, posterior = report[control], report["posterior"]
    beneficial = sum(r["regret_mm2"] < 0 for r in primary)
    harmful = sum(r["regret_mm2"] > 1 for r in primary)
    reduction = (
        1 - posterior["harmful_accepted"] / reference["harmful_accepted"]
        if reference["harmful_accepted"]
        else None
    )
    brier_improvement = reference["brier"] - posterior["brier"]
    passed = (
        beneficial >= 10
        and harmful >= 10
        and reduction is not None
        and reduction >= 0.2
        and brier_improvement >= 0.005
        and all(r["valid"] for r in primary)
    )
    return {
        "passed": bool(passed),
        "discovery_not_significance": True,
        "beneficial_contexts": beneficial,
        "materially_harmful_contexts": harmful,
        "strongest_independent_control": control,
        "harm_reduction_fraction": reduction,
        "brier_improvement": brier_improvement,
        "primary_comparators": report,
        "replication_authorized": bool(passed),
        "stop_reason": None if passed else "registered_source_gate_failed",
    }


def compact_row(row):
    return {
        k: v
        for k, v in row.items()
        if k not in ("b", "c", "features", "open_error", "variance_diag_m2")
    }


def run_source(cache, output):
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = load_source(cache)
    identity = source_identity(Path(__file__).parent)
    outer_folds = folds(np.arange(40), 5, 0)
    all_rows, all_models = [], []
    for outer, held in enumerate(outer_folds):
        print("OUTER", outer + 1, "of 5", flush=True)
        indices = np.setdiff1d(np.arange(40), held)
        bank = ModelBank(data)
        _, parameters, sandwich, oof_models = bank.crossfit(indices, 4, 100 + outer)
        training = make_contexts(
            data, indices, oof_models, parameters, sandwich, outcomes=True
        )
        # Rebuild inner training risk contexts without the inner held recordings.
        # The inner held cohort contributes neither GP/covariance fitting nor controls.
        panels = []
        for inner, inner_held in enumerate(folds(indices, 4, 100 + outer)):
            inner_fit = np.setdiff1d(indices, inner_held)
            _, ip, isp, inner_oof = bank.crossfit(
                inner_fit, 3, 1000 + 10 * outer + inner
            )
            inner_training = make_contexts(
                data, inner_fit, inner_oof, ip, isp, outcomes=True
            )
            inner_queries = make_contexts(
                data, inner_held, bank.get(inner_fit), ip, isp, outcomes=True
            )
            panels.append((inner_training, inner_queries))
        settings, calibrators, validation, _ = fit_selectors(training, panels)
        queries = make_contexts(
            data, held, bank.get(indices), parameters, sandwich, outcomes=False
        )
        apply_selectors(training, queries, settings, calibrators)
        fold_dir = output / f"outer-{outer}"
        fold_dir.mkdir()
        write_json(
            fold_dir / "fit.json",
            {
                "fit_recordings": data["names"][indices].tolist(),
                "held_recordings": data["names"][held].tolist(),
                "covariance": parameters,
                "sandwich_covariance": sandwich,
                "settings": settings,
                "calibrators": calibrators,
                "inner_validation_brier": validation,
                "model_lineage": bank.lineage,
            },
        )
        np.savez_compressed(
            fold_dir / "forecast_pair.npz",
            **{
                f"{i}_{which}": r[which]
                for i, r in enumerate(queries)
                for which in ("b", "c")
            },
        )
        write_json(fold_dir / "predicted_risks.json", [compact_row(r) for r in queries])
        write_json(
            fold_dir / "prediction_seal.json",
            {
                "source": identity,
                "forecast_pair_sha256": sha256(fold_dir / "forecast_pair.npz"),
                "risk_sha256": sha256(fold_dir / "predicted_risks.json"),
                "fit_sha256": sha256(fold_dir / "fit.json"),
                "held_suffix_observations_used": False,
            },
        )
        all_rows.extend(queries)
        all_models.extend(bank.lineage)
        del bank
    # Fixed-coverage decisions are sealed using predictions alone before aggregate scoring.
    for horizon in HORIZONS:
        rows = [r for r in all_rows if r["horizon"] == horizon]
        chosen = selections(rows, 0.5)
        for k, row in enumerate(rows):
            row["accepted50"] = {m: bool(chosen[m][k]) for m in METHODS}
            for m in METHODS:
                actual = deploy(row["b"], row["c"], row["accepted50"][m])
                expected = row["c"] if row["accepted50"][m] else row["b"]
                if actual.tobytes() != expected.tobytes():
                    raise AssertionError("fallback byte parity failed")
    write_json(
        output / "selection_seal.json",
        {
            "source": identity,
            "selections": [
                {"key": r["key"], "accepted50": r["accepted50"]} for r in all_rows
            ],
            "selection_uses_outcomes": False,
        },
    )
    # All pairs, risks and selections exist before any held-fold scoring.
    for row in all_rows:
        i = int(np.flatnonzero(data["names"] == row["recording"])[0])
        attach_outcome(
            row,
            data["local_target"][i, row["endpoint"] : row["endpoint"] + row["horizon"]],
        )
    gate = source_gate(all_rows)
    report = {
        "contract": "deform-posterior-regret-source-result-v1",
        "status": "completed",
        "source_gate": gate,
        "replication_gate": {
            "status": "not_run" if not gate["passed"] else "authorized_not_yet_run"
        },
        "source": identity,
        "cache_manifest_sha256": MANIFEST_SHA256,
        "fit_cache_sha256": FIT_SHA256,
        "source_recordings": 40,
        "contexts": len(all_rows),
        "technical_failures": sum(not r["valid"] for r in all_rows),
        "historical_validation_read": False,
        "source_test_read": False,
        "official_evaluation_read": False,
        "held_v8_accessed": False,
        "simulator_training_exposure": "historically trained on source; only residual/risk learners cross-fitted",
        "statistical_unit": "complete_recording",
        "elapsed_seconds": time.perf_counter() - start,
        "comparators_by_horizon": {
            str(h): summarize([r for r in all_rows if r["horizon"] == h])
            for h in HORIZONS
        },
        "risk_coverage_curves": {
            str(h): {
                str(f): summarize([r for r in all_rows if r["horizon"] == h], f)
                for f in (0.1, 0.25, 0.5, 0.75, 1.0)
            }
            for h in HORIZONS
        },
        "implementation_controls": {
            "matched_distribution_exact": True,
            "clamped_nodes_exact": True,
            "fallback_bit_exact": True,
            "all_selectors_same_forecasts": True,
        },
    }
    write_json(output / "per_context.json", [compact_row(r) for r in all_rows])
    write_json(
        output / "per_recording.json",
        [
            {
                "recording": str(name),
                "primary": summarize(
                    [
                        r
                        for r in all_rows
                        if r["recording"] == name and r["horizon"] == 100
                    ],
                    use_global_selection=True,
                ),
                "contexts": [
                    compact_row(r) for r in all_rows if r["recording"] == name
                ],
            }
            for name in data["names"]
        ],
    )
    write_json(output / "report.json", report)
    print(json.dumps(gate, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if head != args.freeze_commit or dirty:
        raise RuntimeError("run requires the exact clean pre-outcome commit")
    run_source(args.cache, args.output)


if __name__ == "__main__":
    main()
