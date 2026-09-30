"""Gate-bound DLO4/DLO5 retrospective replication on native-checkpoint caches.

The cache adapter must bind each native replay, original recording and source
split. This module cannot authorize data access from an unpassed source gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .conditioning import GRID, calibrate, covariance_shapes, predict_open
from .controls import INDEPENDENT
from .study import (
    HORIZONS,
    METHODS,
    ModelBank,
    apply_selectors,
    attach_outcome,
    compact_row,
    fit_selectors,
    folds,
    make_contexts,
    selections,
    sha256,
    source_identity,
    summarize,
    write_json,
)

SPLIT_DOMAIN = "conditional-query-posterior-v1-20260906"


def authorize_source(path):
    report = json.loads(Path(path).read_text())
    if (
        report.get("contract") != "deform-posterior-regret-source-result-v1"
        or report.get("source_gate", {}).get("passed") is not True
        or report.get("source_gate", {}).get("replication_authorized") is not True
        or report.get("source") != source_identity(Path(__file__).parent)
    ):
        raise ValueError("exact-source gate does not authorize replication")
    return report["source_gate"]["strongest_independent_control"]


def load_replication_cache(root, object_name):
    """Validate membership, native backend and hashes BEFORE reading arrays."""
    root = Path(root)
    manifest = json.loads((root / "replication_manifest.json").read_text())
    if (
        manifest.get("contract") != "deform-posterior-regret-native-source-cache-v1"
        or manifest.get("split_domain") != SPLIT_DOMAIN
        or manifest.get("official_evaluation_read") is not False
    ):
        raise ValueError("unapproved replication cache contract")
    entry = manifest["objects"][object_name]
    checkpoint = entry["native_checkpoint"]
    if (
        checkpoint["backend"] != "official-DEFORM-PBD"
        or checkpoint["upstream_commit"] != "b73b8b8ecc033caefa693fab7898741d4e6dbeff"
        or len(checkpoint["sha256"]) != 64
    ):
        raise ValueError("native checkpoint provenance missing")
    checkpoint_path = root / checkpoint["file"]
    if sha256(checkpoint_path) != checkpoint["sha256"]:
        raise ValueError("native checkpoint checksum differs")
    records = entry["recordings"]
    if len(records) != 56 or len({r["name"] for r in records}) != 56:
        raise ValueError("replication source membership differs")
    ordered = sorted(
        records,
        key=lambda r: hashlib.sha256(
            f"{SPLIT_DOMAIN}/{object_name}/{r['name']}".encode()
        ).digest(),
    )
    expected = {
        "fit": [r["name"] for r in ordered[:32]],
        "calibration": [r["name"] for r in ordered[32:44]],
        "test": [r["name"] for r in ordered[44:]],
    }
    for record in records:
        if sha256(root / record["file"]) != record["sha256"]:
            raise ValueError("original-recording hash differs")
    bundles = {}
    for label in ("fit", "calibration", "test"):
        item = entry["caches"][label]
        if (
            item["names"] != expected[label]
            or sha256(root / item["file"]) != item["sha256"]
        ):
            raise ValueError("replication split/cache checksum differs")
        with np.load(root / item["file"], allow_pickle=False) as archive:
            value = {k: archive[k] for k in archive.files}
        if value["names"].tolist() != expected[label]:
            raise ValueError("cache membership differs")
        if value["features"].shape != (len(expected[label]), 498, 8, 92):
            raise ValueError("native feature contract differs")
        if value["action"].shape != value["baseline"][:, :, [0, 1, -2, -1]].shape:
            raise ValueError("native clamped-node/action dimensions differ")
        if not np.allclose(
            value["frames"].swapaxes(1, 2) @ value["frames"], np.eye(3), atol=1e-10
        ):
            raise ValueError("nonorthonormal native frame")
        for key, array in value.items():
            if key != "names" and not np.isfinite(array).all():
                raise ValueError("nonfinite native cache")
        bundles[label] = value
    return bundles, manifest


def stratified_bootstrap(rows_by_object, comparator, repetitions=10000):
    """Paired complete-recording sampling; fixed objects have equal weight."""
    rng = np.random.default_rng(20260907)
    summaries = []
    object_differences = {}
    for object_name, rows in rows_by_object.items():
        names = sorted({r["recording"] for r in rows})
        values = []
        for name in names:
            current = [r for r in rows if r["recording"] == name]
            harmful = np.asarray([r["regret_mm2"] > 1 for r in current])
            a = np.asarray([r["accepted50"]["posterior"] for r in current])
            b = np.asarray([r["accepted50"][comparator] for r in current])
            mse_a = np.mean(
                [
                    r["mse_c_mm2"] if take else r["mse_b_mm2"]
                    for r, take in zip(current, a, strict=True)
                ]
            )
            mse_b = np.mean(
                [
                    r["mse_c_mm2"] if take else r["mse_b_mm2"]
                    for r, take in zip(current, b, strict=True)
                ]
            )
            values.append([np.sum(a & harmful), np.sum(b & harmful), mse_a, mse_b])
        values = np.asarray(values)
        object_differences[object_name] = float(np.mean(values[:, 0] - values[:, 1]))
        samples = rng.integers(0, len(values), size=(repetitions, len(values)))
        summaries.append((values.mean(0), values[samples].mean(1)))
    observed = np.mean([s[0] for s in summaries], axis=0)
    sampled = np.mean([s[1] for s in summaries], axis=0)
    ratio = np.sqrt(sampled[:, 2] / sampled[:, 3])
    return {
        "harm_difference": float(observed[0] - observed[1]),
        "harm_difference_upper975": float(
            np.quantile(sampled[:, 0] - sampled[:, 1], 0.975)
        ),
        "harm_reduction_fraction": float(1 - observed[0] / observed[1])
        if observed[1]
        else None,
        "rmse_ratio": float(np.sqrt(observed[2] / observed[3])),
        "rmse_ratio_upper975": float(np.quantile(ratio, 0.975)),
        "object_harm_differences": object_differences,
        "repetitions": repetitions,
        "unit": "complete_recording_with_all_contexts",
    }


def replication_gate(rows_by_object, comparator):
    audit = stratified_bootstrap(rows_by_object, comparator)
    all_rows = [r for rows in rows_by_object.values() for r in rows]
    summary = summarize(all_rows, use_global_selection=True)
    brier_gain = summary[comparator]["brier"] - summary["posterior"]["brier"]
    passed = (
        audit["harm_reduction_fraction"] is not None
        and audit["harm_reduction_fraction"] >= 0.2
        and audit["harm_difference_upper975"] < 0
        and audit["rmse_ratio_upper975"] <= 1.01
        and all(d < 0 for d in audit["object_harm_differences"].values())
        and brier_gain > 0
        and all(r["valid"] for r in all_rows)
    )
    stronger = [
        m
        for m in INDEPENDENT
        if summary[m]["harmful_accepted"] < summary["posterior"]["harmful_accepted"]
    ]
    return {
        "passed": bool(passed),
        "primary_control": comparator,
        "bootstrap": audit,
        "brier_improvement": brier_gain,
        "stronger_alternatives": stronger,
        "claim_limited_by_stronger_alternative": bool(stronger),
        "all_comparators": summary,
    }


def run_replication(cache, output, source_report):
    comparator = authorize_source(source_report)  # BEFORE any cache listing/read.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    object_queries, target_bundles = {}, {}
    for object_name in ("DLO4", "DLO5"):
        bundle, manifest = load_replication_cache(cache, object_name)
        fitting, calibration, test = (bundle[k] for k in ("fit", "calibration", "test"))
        bank = ModelBank(fitting)
        indices = np.arange(32)
        _, p, sp, oof = bank.crossfit(indices, 4, 100)
        training = make_contexts(fitting, indices, oof, p, sp, outcomes=True)
        panels = []
        for k, held in enumerate(folds(indices, 4, 100)):
            fit = np.setdiff1d(indices, held)
            _, ip, isp, inner_oof = bank.crossfit(fit, 3, 1000 + k)
            panels.append(
                (
                    make_contexts(fitting, fit, inner_oof, ip, isp, outcomes=True),
                    make_contexts(fitting, held, bank.get(fit), ip, isp, outcomes=True),
                )
            )
        settings, calibrators, _, _ = fit_selectors(training, panels)
        models = bank.get(indices)
        error = np.asarray(
            [
                calibration["local_target"][i]
                - predict_open(models, calibration["features"][i])
                for i in range(12)
            ]
        )
        shapes = np.asarray(
            [
                covariance_shapes(models, calibration["features"][i], GRID)
                for i in range(12)
            ]
        )
        robust = np.asarray(
            [
                covariance_shapes(models, calibration["features"][i], GRID, robust=True)
                for i in range(12)
            ]
        )
        p, sp = calibrate(error, shapes), calibrate(error, robust)
        cal_rows = make_contexts(
            calibration, np.arange(12), models, p, sp, outcomes=True
        )
        apply_selectors(training, cal_rows, settings, calibrators)
        from .controls import fit_probability_calibration

        for method in METHODS:
            calibrators[method] = fit_probability_calibration(
                [r["raw"][method] for r in cal_rows],
                [r["regret_mm2"] > 1 for r in cal_rows],
            )
        apply_selectors(training, cal_rows, settings, calibrators)
        queries = make_contexts(test, np.arange(12), models, p, sp, outcomes=False)
        apply_selectors(training, queries, settings, calibrators)
        thresholds = {}
        for horizon in HORIZONS:
            cal = [r for r in cal_rows if r["horizon"] == horizon]
            rows = [r for r in queries if r["horizon"] == horizon]
            chosen = selections(rows, 0.5)
            for m in METHODS:
                threshold = sorted((r["score"][m], r["key"]) for r in cal)[
                    len(cal) // 2 - 1
                ]
                thresholds[f"{horizon}:{m}"] = threshold
                for k, row in enumerate(rows):
                    row.setdefault("accepted50", {})[m] = bool(chosen[m][k])
                    row.setdefault("accepted_threshold", {})[m] = bool(
                        row["valid"] and (row["score"][m], row["key"]) <= threshold
                    )
        object_queries[object_name], target_bundles[object_name] = queries, test
        directory = output / object_name
        directory.mkdir()
        np.savez_compressed(
            directory / "forecast_pair.npz",
            **{
                f"{i}_{which}": r[which]
                for i, r in enumerate(queries)
                for which in ("b", "c")
            },
        )
        write_json(
            directory / "risks_and_selections.json", [compact_row(r) for r in queries]
        )
        write_json(
            directory / "fit.json",
            {
                "settings": settings,
                "calibrators": calibrators,
                "covariance": p,
                "sandwich_covariance": sp,
                "thresholds": thresholds,
                "native_cache_manifest": manifest,
                "model_lineage": bank.lineage,
            },
        )
    write_json(
        output / "joint_prediction_seal.json",
        {
            "source_report_sha256": sha256(source_report),
            "source": source_identity(Path(__file__).parent),
            "test_scoring_started": False,
            "artifacts": {
                f"{o}/{f}": sha256(output / o / f)
                for o in object_queries
                for f in ("forecast_pair.npz", "risks_and_selections.json", "fit.json")
            },
        },
    )
    for object_name, queries in object_queries.items():
        test = target_bundles[object_name]
        for row in queries:
            i = int(np.flatnonzero(test["names"] == row["recording"])[0])
            attach_outcome(
                row,
                test["local_target"][
                    i, row["endpoint"] : row["endpoint"] + row["horizon"]
                ],
            )
    primary = {
        o: [r for r in rows if r["horizon"] == 100]
        for o, rows in object_queries.items()
    }
    report = {
        "contract": "deform-posterior-regret-retrospective-replication-v1",
        "replication_gate": replication_gate(primary, comparator),
        "historical_exposure": True,
        "fresh_confirmation": False,
        "all_contexts": {
            o: [compact_row(r) for r in rows] for o, rows in object_queries.items()
        },
        "threshold_deployment": {
            o: {
                m: {
                    "actual_acceptance": float(
                        np.mean([r["accepted_threshold"][m] for r in rows])
                    ),
                    "exact_fallback_count": sum(
                        not r["accepted_threshold"][m] for r in rows
                    ),
                }
                for m in METHODS
            }
            for o, rows in primary.items()
        },
    }
    write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    arguments = parser.parse_args()
    run_replication(arguments.cache, arguments.output, arguments.source_report)
