"""Fixed-forecast heavy-tail controls for the completed real DEFORM study.

Source calibration may fit controls; target scores never select a control.
Parent predictions/models and source membership are content-addressed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln

PARENT_ZIP_SHA = "e7e182e009ac234fb611afb6ca03eb1006ee65da2688ec2a084105667fa4e591"
PARENT_SCRIPT_BLOB = "57f9aff414f39f52a4ef346e525f40342554e75e"
NEW_ARMS = (
    "empirical_student_global",
    "empirical_student_conditional",
    "sandwich_student_conditional",
    "matched_plugin_student",
)
METRICS = ("nll", "crps_m", "coverage90", "width90_m", "brier")
CONFIG = {
    "contract": "deform-conditional-query-heavy-tail-controls-v1",
    "parent_run_id": 33985128557,
    "parent_artifact_id": 9974939619,
    "parent_artifact_sha256": PARENT_ZIP_SHA,
    "parent_script_git_blob": PARENT_SCRIPT_BLOB,
    "df_grid": [3.0, 4.0, 5.0, 8.0, 11.0, 16.0, 19.0, 35.0, 64.0, 128.0, None],
    "temperature_grid": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
    "shrinkage_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
    "new_arms": list(NEW_ARMS),
    "primary_source_size": 32,
    "secondary_source_sizes": [8, 16],
    "bootstrap_replicates": 10000,
    "bootstrap_seed": 20260906,
    "primary_controls": ["empirical_student_conditional", "sandwich_student_conditional"],
    "source_calibration_objective": "equal-recording mean NLL over the original six development queries",
    "test_informed_control_selection": False,
    "parent_model_refit": False,
    "official_eval_access": False,
    "new_physical_acquisition": False,
    "analysis_status": "post-result retrospective matched-control follow-up; not independent confirmation",
}


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def load_parent_module(path: Path) -> Any:
    if blob_sha(path.read_bytes()) != PARENT_SCRIPT_BLOB:
        raise ValueError("Parent evaluator does not match its frozen Git blob")
    spec = importlib.util.spec_from_file_location("frozen_parent_query", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unpack(archive: Path, destination: Path) -> None:
    if sha(archive) != PARENT_ZIP_SHA:
        raise ValueError("Parent archive hash mismatch")
    destination.mkdir()
    with zipfile.ZipFile(archive) as z:
        names = [i for i in z.infolist() if not i.is_dir()]
        bases = [Path(i.filename).name for i in names]
        if len(bases) != len(set(bases)):
            raise ValueError("Duplicate basenames in parent archive")
        for item, name in zip(names, bases, strict=True):
            if item.file_size > 100_000_000:
                raise ValueError("Unexpectedly large parent member")
            (destination / name).write_bytes(z.read(item))
    seal = read(destination / "prediction_seal.json")
    for field, name in (
        ("protocol_sha256", "protocol.json"),
        ("models_sha256", "frozen_models.json"),
        ("predictions_sha256", "sealed_predictions.json"),
        ("input_manifest_sha256", "input_manifest.json"),
    ):
        if sha(destination / name) != seal[field]:
            raise ValueError(f"Parent seal mismatch: {field}")


def model_arrays(raw: dict) -> dict:
    return {k: np.asarray(v, dtype=float) if isinstance(v, list) else v for k, v in raw.items()}


def empirical_covariance(residuals: np.ndarray, shrinkage: float) -> np.ndarray:
    # Second moment about the fixed zero residual mean, not a recentered mean.
    covariance = residuals.T @ residuals / len(residuals)
    return (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance)) + 1e-10 * np.eye(covariance.shape[0])


def covariance_for(model: dict, arm: str, shrinkage: float) -> np.ndarray:
    if arm == "matched_plugin_student":
        return model["psi"] / (model["df"] - 2)
    key = "raw_residuals" if arm == "empirical_student_global" else "normalized_residuals"
    return empirical_covariance(model[key], shrinkage)


def multiplier(parent: Any, model: dict, feature: np.ndarray, arm: str) -> float:
    x, s2 = parent.transform(model, feature)
    if arm == "empirical_student_global":
        return 1.0
    if arm == "sandwich_student_conditional":
        g = model["x"][:, :len(model["beta"])]
        fit_s2 = np.exp(np.clip(g @ model["beta"] - model["log_center"], -3, 3))
        xw = model["x"] / np.sqrt(fit_s2[:, None])
        # Frequentist covariance of the penalized estimate: V Xw' Xw V, not V.
        projection = xw @ model["v"] @ x
        s2 += float(projection @ projection)
    return s2


def distribution(variance: np.ndarray, df: float | None, temperature: float) -> dict:
    if not np.isfinite(variance).all() or np.any(variance <= 0) or temperature <= 0:
        raise ValueError("Invalid predictive variance")
    if df is None:
        return {"kind": "normal", "scale": np.sqrt(temperature * variance)}
    if df <= 2:
        raise ValueError("Student-t variance requires df > 2")
    # Calibrating variance, not confusing Student scale with standard deviation.
    return {"kind": "student", "df": df, "scale": np.sqrt(temperature * variance * (df - 2) / df)}


def candidate_losses(variances: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, list]:
    """All candidates use the same calibration recordings and query weights."""
    specs, losses = [], []
    for df in CONFIG["df_grid"]:
        temperatures = np.asarray(CONFIG["temperature_grid"])
        scale2 = variances[:, None] * temperatures[None, :, None, None]
        if df is None:
            value = .5 * (error[None, None] ** 2 / scale2 + np.log(scale2) + np.log(2 * np.pi))
        else:
            scale2 = scale2 * (df - 2) / df
            value = (.5 * np.log(df * np.pi * scale2) + gammaln(df / 2) - gammaln((df + 1) / 2)
                     + .5 * (df + 1) * np.log1p(error[None, None] ** 2 / (df * scale2)))
        losses.append(value.mean(axis=(-1, -2)).reshape(-1))
        specs.extend((df, s, temp) for s in range(len(variances)) for temp in CONFIG["temperature_grid"])
    result = np.concatenate(losses)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite calibration objective")
    return result, specs


def calibrate(parent: Any, model: dict, features: np.ndarray, residuals: np.ndarray, queries: np.ndarray, arm: str) -> dict:
    shrinkages = [0.0] if arm == "matched_plugin_student" else CONFIG["shrinkage_grid"]
    multipliers = np.array([multiplier(parent, model, f, arm) for f in features])
    variances = np.stack([
        multipliers[:, None] * np.einsum("qi,ij,qj->q", queries, covariance_for(model, arm, s), queries)
        for s in shrinkages
    ])
    losses, specs = candidate_losses(variances, residuals @ queries.T)
    best = int(np.argmin(losses))
    df, index, temperature = specs[best]
    return {"df": df, "shrinkage": shrinkages[index], "temperature": temperature,
            "calibration_nll": float(losses[best]), "candidate_count": len(losses)}


def predict(parent: Any, model: dict, feature: np.ndarray, queries: np.ndarray, arm: str, settings: dict) -> dict:
    covariance = covariance_for(model, arm, settings["shrinkage"])
    variance = multiplier(parent, model, feature, arm) * np.einsum("qi,ij,qj->q", queries, covariance, queries)
    return distribution(variance, settings["df"], settings["temperature"])


def checked_records(parent: Any, root: Path, manifest: dict, dlo: str, partition: str) -> dict:
    if partition not in ("calibration", "source_test"):
        raise ValueError("Only parent calibration and source-test records are authorized")
    records = {}
    for item in manifest[dlo][partition]:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Unexpected dataset filename")
        path = root / dlo / "train" / item["name"]
        if sha(path) != item["sha256"]:
            raise ValueError(f"Dataset identity changed: {path}")
        records[item["name"]] = parent.load(path)
    if len(records) != 12:
        raise ValueError("Expected the twelve frozen source records")
    return records


def summarize(rows: list[dict], parent_arms: tuple) -> dict:
    arms = parent_arms + NEW_ARMS
    summaries, differences, object_means, supported = {}, {}, {}, {}
    for size in (8, 16, 32):
        panel = [r for r in rows if r["source_size"] == size]
        keys = sorted({(r["dlo"], r["trajectory"]) for r in panel})
        if len(keys) != 24:
            raise ValueError("Missing trajectory")
        grouped = {(arm, key): [] for arm in arms for key in keys}
        for row in panel:
            grouped[(row["arm"], (row["dlo"], row["trajectory"]))].append([row[m] for m in METRICS])
        if any(len(v) != 15 for v in grouped.values()):
            raise ValueError("Missing or duplicate object/origin/horizon score")
        blocks = {arm: np.array([np.mean(grouped[arm, key], axis=0) for key in keys]) for arm in arms}
        summaries[str(size)] = {a: dict(zip(METRICS, b.mean(0).tolist(), strict=True)) for a, b in blocks.items()}
        objects = np.array([key[0] for key in keys])
        object_means[str(size)] = {dlo: {a: dict(zip(METRICS, b[objects == dlo].mean(0).tolist(), strict=True))
                                       for a, b in blocks.items()} for dlo in ("DLO4", "DLO5")}
        rng = np.random.default_rng(CONFIG["bootstrap_seed"])
        indices = np.column_stack([rng.choice(np.flatnonzero(objects == dlo), size=(CONFIG["bootstrap_replicates"], 12), replace=True)
                                   for dlo in ("DLO4", "DLO5")])
        differences[str(size)] = {}
        for arm in arms[1:]:
            delta = blocks["posterior_student"] - blocks[arm]
            limits = np.quantile(delta[indices].mean(1), [.025, .975], axis=0)
            differences[str(size)][arm] = {m: {"difference": float(delta[:, j].mean()),
                "pointwise_trajectory_bootstrap_95_ci": limits[:, j].tolist(),
                "posterior_trajectory_wins": int(np.sum(delta[:, j] < 0))} for j, m in enumerate(METRICS)}
        supported[str(size)] = all(differences[str(size)][a][m]["pointwise_trajectory_bootstrap_95_ci"][1] < 0
                                  for a in CONFIG["primary_controls"] for m in ("nll", "crps_m"))
    return {"aggregate": summaries, "per_object": object_means, "posterior_minus_comparator": differences,
            "superiority_condition_by_source_size": supported,
            "primary_decision": "supported-in-this-retrospective-control-analysis" if supported["32"] else "not-established-against-stronger-heavy-tail-controls",
            "inference_boundary": "Pointwise descriptive intervals on 24 complete recordings in two fixed objects; post-result follow-up, not independent confirmation."}


def report(result: dict) -> str:
    lines = ["# Heavy-tail controls on frozen real DEFORM forecasts", "", result["primary_decision"], "", CONFIG["analysis_status"], ""]
    for size in (8, 16, 32):
        lines += [f"## {size} fitting + 12 calibration trajectories per object", "",
                  "| Distribution | NLL | CRPS (mm) | Coverage90 | Width (mm) | Brier |", "|---|---:|---:|---:|---:|---:|"]
        for arm, v in result["aggregate"][str(size)].items():
            lines.append(f"| {arm} | {v['nll']:.6f} | {1000*v['crps_m']:.5f} | {100*v['coverage90']:.3f}% | {1000*v['width90_m']:.3f} | {v['brier']:.6f} |")
        lines += ["", "Posterior minus new control, pointwise trajectory-bootstrap 95% intervals:", ""]
        for arm in NEW_ARMS:
            d = result["posterior_minus_comparator"][str(size)][arm]
            c = d["crps_m"]
            lines.append(f"- {arm}: NLL {d['nll']}; CRPS difference {1000*c['difference']:.6f} mm, interval {np.array(c['pointwise_trajectory_bootstrap_95_ci'])*1000} mm.")
    lines += ["", "Original means and Bayesian forecasts are unchanged. No new objects, official eval files, or physical data were used.",
              "The first three new controls use plug-in empirical residual covariance. matched_plugin_student reuses the parent noise-covariance estimate and is a distribution-family control, not an independently estimated covariance baseline.",
              "Source calibration selects degrees of freedom, variance temperature, and empirical shrinkage using only the original development queries. None of the new controls averages predictions over a parameter posterior."]
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    out: Path = args.output
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "protocol.json", CONFIG)
    parent_dir = out / "parent"
    unpack(args.parent_zip, parent_dir)
    parent = load_parent_module(parent_dir / "run.py")
    manifest = read(parent_dir / "input_manifest.json")
    raw_models = read(parent_dir / "frozen_models.json")
    entries = {k: {**v, "model": model_arrays(v["model"])} for k, v in raw_models.items()}
    old_predictions = read(parent_dir / "sealed_predictions.json")
    devq, heldq = parent.queries()
    controls = {}
    for dlo in ("DLO4", "DLO5"):
        cal = checked_records(parent, args.dataset_root, manifest, dlo, "calibration")
        for model_id, e in entries.items():
            if e["dlo"] != dlo:
                continue
            features, errors = [], []
            for trajectory in cal.values():
                f, base, _ = parent.inputs(trajectory, e["origin"], e["horizon"])
                mean = base + parent.transform(e["model"], f)[0] @ e["model"]["w"]
                features.append(f)
                errors.append(trajectory[e["origin"] + e["horizon"], 2:10].ravel() - mean)
            controls[model_id] = {a: calibrate(parent, e["model"], np.array(features), np.array(errors), devq, a) for a in NEW_ARMS}
        print(f"All new controls calibrated without source-test outcomes: {dlo}", flush=True)
    dump(out / "source_fitted_controls.json", controls)
    controls_hash = sha(out / "source_fitted_controls.json")
    predictions, mean_max_error = {}, 0.0
    for dlo in ("DLO4", "DLO5"):
        targets = checked_records(parent, args.dataset_root, manifest, dlo, "source_test")
        for index, p in old_predictions.items():
            if p["dlo"] != dlo:
                continue
            e = entries[str(p["model_id"])]
            f, base, _ = parent.inputs(targets[p["trajectory"]], p["origin"], p["horizon"])
            reconstructed = base + parent.transform(e["model"], f)[0] @ e["model"]["w"]
            mean_max_error = max(mean_max_error, float(np.max(np.abs(reconstructed - np.array(p["mean"])))))
            distributions = {a: parent.serialize(predict(parent, e["model"], f, heldq, a, controls[str(p["model_id"])][a])) for a in NEW_ARMS}
            predictions[index] = {"parent_index": index, "mean": p["mean"], "distributions": distributions}
    if mean_max_error > 1e-12:
        raise ValueError("Parent mean reconstruction drift")
    dump(out / "sealed_control_predictions.json", predictions)
    seal = {"source_revision": args.revision, "protocol_sha256": sha(out / "protocol.json"),
            "fitted_controls_sha256": controls_hash, "predictions_sha256": sha(out / "sealed_control_predictions.json"),
            "parent_zip_sha256": PARENT_ZIP_SHA, "parent_prediction_seal": read(parent_dir / "prediction_seal.json"),
            "max_parent_mean_reconstruction_error_m": mean_max_error, "target_outcomes_used_for_control_fitting": False}
    dump(out / "prediction_seal.json", seal)
    print("All control predictions sealed. Scoring unchanged real trajectory outcomes.", flush=True)
    original_rows = [json.loads(line) for line in (parent_dir / "trajectory_panel_scores.jsonl").read_text().splitlines()]
    original_index = {(r["dlo"], r["trajectory"], r["source_size"], r["origin"], r["horizon"], r["arm"]): r for r in original_rows}
    rows, score_max_error = list(original_rows), 0.0
    for dlo in ("DLO4", "DLO5"):
        targets = checked_records(parent, args.dataset_root, manifest, dlo, "source_test")
        for index, p in old_predictions.items():
            if p["dlo"] != dlo:
                continue
            error = (targets[p["trajectory"]][p["origin"] + p["horizon"], 2:10].ravel() - np.array(p["mean"])) @ heldq.T
            columns = {k: p[k] for k in ("dlo", "trajectory", "source_size", "origin", "horizon")}
            old_score = parent.scores(parent.deserialize_distribution(p["distributions"]["posterior_student"]), error)
            retained = original_index[tuple(columns.values()) + ("posterior_student",)]
            score_max_error = max(score_max_error, max(abs(float(old_score[m].mean()) - retained[m]) for m in METRICS))
            for arm, raw in predictions[index]["distributions"].items():
                metrics = parent.scores(parent.deserialize_distribution(raw), error)
                rows.append({**columns, "arm": arm, **{k: float(v.mean()) for k, v in metrics.items()}})
    if score_max_error > 1e-10 or sha(out / "source_fitted_controls.json") != controls_hash:
        raise ValueError("Original scores drifted or controls changed after sealing")
    result = {**summarize(rows, parent.ARMS), "config": CONFIG, "seal": seal,
              "accounting": {"physical_objects": 2, "source_test_trajectories": 24, "contexts": len(predictions),
                             "score_rows": len(rows), "original_bayesian_score_replay_error": score_max_error,
                             "official_eval_files_opened": 0, "new_physical_data": False,
                             "predictive_means_copied_exactly": True, "existing_bayesian_models_refitted": False},
              "runtime": {"runner": os.environ.get("RUNNER_NAME", "local"), "python": platform.python_version(), "numpy": np.__version__}}
    dump(out / "result.json", result)
    with (out / "trajectory_panel_scores.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (out / "SUMMARY.md").write_text(report(result))
    print(report(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-zip", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    run(parser.parse_args())
