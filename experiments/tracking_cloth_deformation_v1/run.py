"""Maintained CLI for read-only inventory, source fit, prediction seal and scoring."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .data import (
    Case, Inputs, audit_dataset, digest, infer_source_scale, input_view,
    object_digest, read_prefix, scoring_view, write_json,
)
from .model import ARMS, complete_beliefs, fit_specimen, predict, score

HERE = Path(__file__).resolve().parent
METRICS = ("rmse_mm", "mean_marker_error_mm", "coordinate_nll",
           "coordinate_90_coverage", "mean_full_90_width_mm")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def implementation() -> dict[str, str]:
    return {p.name: digest(p) for p in sorted(HERE.glob("*.py"))}


def source_record(args):
    case, protocol, scale = args
    inputs = input_view(case, protocol, scale)
    prediction = predict(inputs, protocol)
    return case.specimen, case.path.name, prediction, scoring_view(case, inputs)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing an empty result table")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(root: Path, output: Path, protocol: dict[str, Any], stage: str,
            workers: int = 1) -> None:
    if stage not in ("inventory", "source", "predict"):
        raise ValueError("Unknown preparation stage")
    root = root.resolve(strict=True)
    output = output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output and dataset must be disjoint directory trees")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    provenance = {
        "created_at": now(), "protocol_id": object_digest(protocol),
        "implementation_sha256": implementation(), "python": sys.version,
        "numpy": np.__version__, "platform": platform.platform(),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "target_numeric_outcomes_read": False,
        "evidence_class": protocol["evidence_class"],
        "paper_claim_authorized": False,
    }
    write_json(output / "run_manifest.json", provenance)
    cases, inventory = audit_dataset(root, protocol)
    write_json(output / "dataset_manifest.json", inventory)
    (output / "DATA_LICENSE.txt").write_text(inventory["included_license_text"])
    report = ["# Tracking Cloth Deformation: public-data pilot", "",
              f"Study: `{protocol['study_id']}`", "",
              "120 verified CSVs; 32 shaking source recordings, 32 twisting targets,",
              "and 56 collision recordings reserved and not numerically read.", "",
              "Dataset cache is read-only. Archive/extracted-byte hashing is not",
              "numeric outcome evaluation. Included noncommercial license governs",
              "this run pending author clarification; raw recordings are not uploaded.", "",
              "This is a reduced spring-mesh pilot, not a PhysTwin/FEM reproduction.",
              "No new acquisition or paper claim is created.", ""]
    (output / "report.md").write_text("\n".join(report))
    if stage == "inventory":
        return
    source = [c for c in cases if c.motion == "shake"]
    scales = [infer_source_scale(c, read_prefix(c, protocol["prefix_seconds"])[1])
              for c in source]
    if len(set(scales)) != 1:
        raise ValueError("Source recordings disagree about metric coordinate units")
    scale = scales[0]
    tasks = [(c, protocol, scale) for c in source]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(source_record, tasks))
    else:
        records = [source_record(task) for task in tasks]
    fitted = {}
    source_rows = []
    for specimen in sorted({c.specimen for c in source}):
        subset = [(name, pred, truth) for group, name, pred, truth in records if group == specimen]
        fitted[specimen] = fit_specimen([(pred, truth) for _, pred, truth in subset], protocol)
        fitted[specimen]["source_recordings"] = [name for name, _, _ in subset]
        for name, oof in zip(fitted[specimen]["source_recordings"],
                             fitted[specimen]["oof_record_rmse_m"], strict=True):
            source_rows.extend({"recording": name, "specimen": specimen, "arm": arm,
                                "oof_rmse_mm": 1000 * float(value)}
                               for arm, value in oof.items())
    freeze = {
        "protocol_id": object_digest(protocol), "inventory_id": inventory["inventory_id"],
        "implementation_sha256": implementation(), "coordinate_scale_to_m": scale,
        "fitted_at": now(), "specimens": fitted, "target_outcomes_used": False,
        "guard_is_empirical_source_rule_not_safety_certificate": True,
    }
    write_json(output / "source_fit.json", freeze)
    save_csv(output / "source_scores.csv", source_rows)
    accepted = sum(int(f["guard_accepts"]) for f in fitted.values())
    report.extend(["## Source-only qualification", "",
                   f"Coordinate scale to metres: `{scale}` (inferred from source initialization only).",
                   f"Empirical source guard accepts {accepted}/8 specimen candidates.",
                   "Each specimen uses four-fold leave-one-speed/grasp-recording-out fitting.",
                   "These folds select the model/guard; they are not independent confirmation.",
                   "No twisting free-marker forecast outcome has been evaluated.", ""])
    (output / "report.md").write_text("\n".join(report))
    if stage == "source":
        return
    private = output / "private_predictions"
    private.mkdir(mode=0o700)
    predictions = {}
    for case in (c for c in cases if c.motion == "twist"):
        inputs = input_view(case, protocol, scale)
        beliefs = complete_beliefs(predict(inputs, protocol), fitted[case.specimen], protocol)
        arrays = {f"{arm}_mean": beliefs[arm][0] for arm in ARMS}
        arrays.update({f"{arm}_variance": beliefs[arm][1] for arm in ARMS})
        arrays.update({"times": inputs.times, "order": inputs.order, "corners": inputs.corners,
                       "cutoff": np.array(inputs.cutoff), "scale": np.array(scale)})
        artifact = private / f"{case.path.stem}.npz"
        np.savez_compressed(artifact, **arrays)
        predictions[case.path.name] = {
            "artifact": str(artifact.relative_to(output)), "sha256": digest(artifact),
            "specimen": case.specimen, "guard_accepts": fitted[case.specimen]["guard_accepts"],
            "corner_raw_column_indices": inputs.order[inputs.corners].tolist(),
            "causal_cutoff_seconds": float(inputs.times[inputs.cutoff]),
        }
    if len(predictions) != 32:
        raise ValueError("Refusing an incomplete target prediction seal")
    seal = {
        "sealed_at": now(), "protocol_id": object_digest(protocol),
        "inventory_id": inventory["inventory_id"], "source_fit_sha256": digest(output / "source_fit.json"),
        "implementation_sha256": implementation(), "predictions": predictions,
        "future_free_marker_outcomes_read": False,
        "future_driven_corner_coordinates_used": True,
        "initialization_prefix_all_markers_used": True,
        "prior_public_outcome_exposure": "unknown; no fresh-confirmation claim",
    }
    write_json(output / "prediction_seal.json", seal)
    report.extend(["## Predictions sealed", "", "All 32 target batches are sealed before scoring.",
                   "Only timestamps, the initialization prefix and future prescribed corners",
                   "entered prediction. Forecasts stay local and are not in the upload bundle.", ""])
    (output / "report.md").write_text("\n".join(report))


def aggregate(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specimens = sorted({row["specimen"] for row in rows})
    if len(specimens) != 8 or len(rows) != 32 * len(ARMS):
        raise ValueError("Incomplete roster; no pooled partial result is authorized")
    table = []
    for specimen in specimens:
        for arm in ARMS:
            subset = [row for row in rows if row["specimen"] == specimen and row["arm"] == arm]
            if len(subset) != 4 or len({row["recording"] for row in subset}) != 4:
                raise ValueError("Missing or duplicate speed/grasp condition")
            table.append({"specimen": specimen, "arm": arm,
                          **{metric: float(np.mean([row[metric] for row in subset])) for metric in METRICS}})
    summary = {arm: {metric: float(np.mean([row[metric] for row in table if row["arm"] == arm]))
                     for metric in METRICS} for arm in ARMS}
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    resamples = rng.integers(0, 8, size=(protocol["bootstrap_repetitions"], 8))
    contrasts = {}
    for comparator in ("nominal_physics", "last_residual", "map_physics"):
        diffs = np.array([next(r["rmse_mm"] for r in table if r["specimen"] == s and r["arm"] == "guarded_bayesian_physics")
                          - next(r["rmse_mm"] for r in table if r["specimen"] == s and r["arm"] == comparator)
                          for s in specimens])
        material_diffs = np.array([np.mean([diffs[i] for i, s in enumerate(specimens)
                                           if s.startswith(material + "_")]) for material in protocol["materials"]])
        material_samples = rng.integers(0, 4, size=(protocol["bootstrap_repetitions"], 4))
        contrasts[comparator] = {
            "guarded_minus_comparator_rmse_mm": float(diffs.mean()),
            "specimen_bootstrap_95_interval_mm": np.quantile(diffs[resamples].mean(axis=1), [0.025, 0.975]).tolist(),
            "material_cluster_sensitivity_95_interval_mm": np.quantile(material_diffs[material_samples].mean(axis=1), [0.025, 0.975]).tolist(),
            "specimen_wins": int((diffs < 0).sum()), "specimen_ties": int((diffs == 0).sum()),
            "specimen_losses": int((diffs > 0).sum()),
            "worst_specimen_regret_mm": float(diffs.max()),
        }
    return table, {"arms": summary, "contrasts": contrasts,
                   "inferential_unit": "8 material-size specimens; 4-material sensitivity also reported",
                   "interval_interpretation": "exploratory paired percentile bootstrap; not simultaneous; small cluster counts",
                   "aggregation": "equal recordings within specimen, then equal specimens; no frame pseudoreplication"}


def score_run(root: Path, output: Path) -> None:
    root, output = root.resolve(strict=True), output.resolve(strict=True)
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output and dataset must be disjoint directory trees")
    if (output / "target_access.json").exists():
        raise ValueError("This run already started target scoring; use a separately identified pilot run")
    protocol = json.loads((output / "protocol.json").read_text())
    seal = json.loads((output / "prediction_seal.json").read_text())
    if seal["protocol_id"] != object_digest(protocol) or seal["implementation_sha256"] != implementation():
        raise ValueError("Protocol or implementation changed after prediction sealing")
    if seal["source_fit_sha256"] != digest(output / "source_fit.json"):
        raise ValueError("Source fit changed after sealing")
    cases, inventory = audit_dataset(root, protocol)
    if inventory["inventory_id"] != seal["inventory_id"]:
        raise ValueError("Dataset changed after sealing")
    for entry in seal["predictions"].values():
        path = (output / entry["artifact"]).resolve()
        if not path.is_relative_to((output / "private_predictions").resolve()) or digest(path) != entry["sha256"]:
            raise ValueError("Prediction artifact identity mismatch")
    write_json(output / "target_access.json", {"started_at": now(), "prediction_seal_sha256": digest(output / "prediction_seal.json"),
                                               "authorized_recordings": sorted(seal["predictions"]), "purpose": "fixed public-data pilot scoring"})
    rows = []
    fallback_records = 0
    harmful_accepted_records = 0
    for case in (c for c in cases if c.motion == "twist"):
        entry = seal["predictions"][case.path.name]
        with np.load(output / entry["artifact"], allow_pickle=False) as arrays:
            inputs = Inputs(arrays["times"], np.empty((0, case.markers, 3)), np.empty((0, 2, 3)),
                            arrays["order"], arrays["corners"], int(arrays["cutoff"]),
                            float(arrays["times"][0]), float(arrays["scale"]))
            truth = scoring_view(case, inputs)
            case_scores = {}
            for arm in ARMS:
                mean, variance = arrays[f"{arm}_mean"], arrays[f"{arm}_variance"]
                case_scores[arm] = score(mean, variance, truth, inputs)
                rows.append({"recording": case.path.name, "specimen": case.specimen, "material": case.material,
                             "speed": case.speed, "grasp": case.grasp, "arm": arm,
                             "guard_accepted": entry["guard_accepts"], **case_scores[arm]})
            if not entry["guard_accepts"]:
                fallback_records += 1
                for field in ("mean", "variance"):
                    if not np.array_equal(arrays[f"guarded_bayesian_physics_{field}"], arrays[f"nominal_physics_{field}"]):
                        raise ValueError("Exact fallback violated")
                if case_scores["guarded_bayesian_physics"] != case_scores["nominal_physics"]:
                    raise ValueError("Exact fallback score violated")
            elif case_scores["guarded_bayesian_physics"]["rmse_mm"] > case_scores["nominal_physics"]["rmse_mm"]:
                harmful_accepted_records += 1
    table, metrics = aggregate(rows, protocol)
    metrics.update({"fallback_recordings": fallback_records, "accepted_recordings": 32 - fallback_records,
                    "harmful_accepted_recordings_vs_nominal": harmful_accepted_records,
                    "exact_fallback_violations": 0, "target_recordings": 32,
                    "evidence_class": protocol["evidence_class"], "paper_claim_authorized": False})
    save_csv(output / "target_scores.csv", rows)
    save_csv(output / "specimen_scores.csv", table)
    write_json(output / "metrics.json", metrics)
    manifest = json.loads((output / "run_manifest.json").read_text())
    manifest.update({"completed_at": now(), "target_numeric_outcomes_read": True,
                     "prediction_seal_sha256": digest(output / "prediction_seal.json"),
                     "metrics_sha256": digest(output / "metrics.json"),
                     "status": "completed-pilot-not-claim-promoted"})
    write_json(output / "run_manifest.json", manifest)
    report = (output / "report.md").read_text()
    report += "\n## Held-out twisting results\n\n"
    report += "| Arm | Specimen-balanced RMSE [mm] | Coordinate NLL | 90% coverage | Full width [mm] |\n"
    report += "| --- | ---: | ---: | ---: | ---: |\n"
    for arm in ARMS:
        values = metrics["arms"][arm]
        report += (f"| {arm} | {values['rmse_mm']:.4f} | {values['coordinate_nll']:.4f} | "
                   f"{100 * values['coordinate_90_coverage']:.2f}% | {values['mean_full_90_width_mm']:.4f} |\n")
    report += (f"\nGuarded candidate: {32 - fallback_records}/32 accepted; {fallback_records}/32 exact fallbacks. "
               f"{harmful_accepted_records} accepted records worsen RMSE versus nominal physics.\n\n"
               "Intervals in metrics.json resample eight specimens, with a four-material sensitivity check. "
               "They are exploratory, non-simultaneous intervals with few clusters. "
               "Diagonal moment-matched Gaussian scores do not validate joint trajectory covariance.\n\n"
               "The primary endpoint is free-marker Euclidean RMSE, not the paper's mass-matrix metric. "
               "Known future measured corner positions are prescribed inputs; this is not command-conditioned "
               "or fully online forecasting. No unseen-object, material-identification, causal or safety claim follows.\n")
    (output / "report.md").write_text(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("inventory", "source", "predict", "score"), default="source")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    try:
        if args.stage == "score":
            score_run(args.dataset_root, args.output)
        else:
            protocol = json.loads((HERE / "protocol.json").read_text())
            prepare(args.dataset_root, args.output, protocol, args.stage, args.workers)
    except Exception as exc:
        if args.output.is_dir() and not args.output.resolve().is_relative_to(args.dataset_root.resolve()):
            write_json(args.output / "failure.json", {"failed_at": now(), "stage": args.stage,
                       "exception": type(exc).__name__, "message": str(exc),
                       "target_scoring_started": (args.output / "target_access.json").exists(),
                       "scientific_decision": "not-evaluated-or-incomplete; no claim"})
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
