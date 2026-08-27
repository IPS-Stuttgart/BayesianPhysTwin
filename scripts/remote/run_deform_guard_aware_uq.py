#!/usr/bin/env python3
"""Seal a fixed-mean UQ comparison using only previously opened DEFORM carriers."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import run_deform_weak_constraint_belief as parent_runner

from bayesian_phystwin_experiments.deform_guard_aware_uq import (
    EXPERIMENT,
    FAMILIES,
    PROTOCOL,
    RAW_ARMS,
    VARIANTS,
    build_prediction,
    calibrate_source,
    calibrated_covariance,
    load_protocol,
    primary_decision,
    validate_covariance,
)
from bayesian_phystwin_experiments.deform_multiobject_restart import config_for_object
from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)
from bayesian_phystwin_experiments.deform_weak_constraint_belief import summarize_uq

ROOT = Path(__file__).resolve().parents[2]
storage = parent_runner.previous


def freeze(output: Path) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("commit the complete method before freezing")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            "configs/sota/deform_*",
            "scripts/remote/run_deform*",
            "scripts/verify_deform*",
            "tests/test_deform*",
            "docs/deform_*",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(
        output / "source_receipt.json",
        {
            "schema": "deform-state-restart-source-receipt-v1",
            "experiment": EXPERIMENT,
            "revision": revision,
            "git_clean": True,
            "files": {name: file_digest(ROOT / name) for name in paths},
            "new_method_outcomes_read": False,
            "protected_data_access": False,
        },
    )
    print(
        json.dumps({"source_revision": revision, "bound_files": len(paths)}), flush=True
    )


def input_barrier(protocol: dict[str, Any]) -> dict[str, Any]:
    path = Path(protocol["input_root"]) / "prediction_barrier.json"
    if file_digest(path) != protocol["input_prediction_barrier_sha256"]:
        raise ValueError("sealed input prediction barrier changed")
    verification = Path(protocol["input_verification_path"])
    if file_digest(verification) != protocol["input_verification_sha256"]:
        raise ValueError("independently verified parent evidence changed")
    check = json.loads(verification.read_text())
    if check["verified"] is not True or check["old_means_byte_identical"] is not True:
        raise ValueError("parent predictions were not independently verified")
    record = json.loads(path.read_text())
    if (
        record["source_revision"] != protocol["input_source_revision"]
        or record["ordinary_success"] != 30
        or record["protected_data_access"] is not False
        or record["new_metrics_computed"] is not False
        or record["retained_technical_failure"] != 0
        or record["unsealable"] != 0
        or set(record["objects"]) != {"DLO1", "DLO2", "DLO3"}
    ):
        raise ValueError("input denominator or information boundary changed")
    return record


def parent_arrays(
    item: dict[str, Any], protocol: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], str]:
    barrier = input_barrier(protocol)
    root = Path(protocol["input_root"]) / item["object"]
    seal_path = root / "prediction_seal.json"
    if file_digest(seal_path) != barrier["objects"][item["object"]]["seal_sha256"]:
        raise ValueError("input object seal changed")
    seal = json.loads(seal_path.read_text())
    if any(
        seal.get(k) != v
        for k, v in {
            "object": item["object"],
            "names": item["names"],
            "future_free_node_truth_used": False,
            "protected_data_access": False,
            "new_metrics_computed": False,
            "previous_paired_prediction_byte_identical": True,
        }.items()
    ):
        raise ValueError("input object order or causal boundary changed")
    values = [
        storage.verified_arrays(root / key, seal["files"][key])
        for key in ("model.npz", "fits.npz", "predictions.npz")
    ]
    if (
        values[0]["names"].tolist() != item["names"]
        or values[2]["names"].tolist() != item["names"]
    ):
        raise ValueError("input identity order changed")
    digest = seal["files"]["predictions.npz"]["arrays"]["previous_paired_8"]
    return values[0], values[1], values[2], digest


def expected_arrays(
    item: dict[str, Any], protocol: dict[str, Any], parent: dict[str, Any]
) -> tuple[dict[str, np.ndarray], str]:
    model, fits, predictions, mean_digest = parent_arrays(item, protocol)
    config = config_for_object(parent, item)
    batch, nodes = len(item["names"]), config.node_count
    if (
        model["response"].shape != (batch, 145, nodes, 3, 60)
        or model["incumbent"].shape != (batch, 170, nodes, 3)
        or predictions["previous_paired_8"].shape != (batch, 120, nodes, 3)
        or np.any(model["response"][:, :, config.clamped_nodes])
        or array_digest(model["incumbent"][:, 50:])
        != array_digest(predictions["incumbent"])
    ):
        raise ValueError("input physical time, clamp, or identity alignment changed")
    value = build_prediction(
        model["incumbent"][:, 50:],
        predictions["previous_paired_8"],
        model["response"][:, 25:, :, :, :24],
        fits["strong_8__coefficients"],
        fits["strong_8__posterior"],
        fits["strong_8__gain"],
        registered_mean_sha256=mean_digest,
    )
    return {"names": np.asarray(item["names"]), **value}, mean_digest


def consume_attempt(
    protocol: dict[str, Any], output: Path, receipt: dict[str, Any]
) -> Path:
    if output.resolve() != Path(protocol["run_root"]).resolve():
        raise ValueError("only the registered fresh output root is authorized")
    ledger = Path(protocol["attempt_ledger"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    write_json_once(
        ledger,
        {
            "schema": EXPERIMENT + "-attempt",
            "source_revision": receipt["revision"],
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "run_root": str(output.resolve()),
            "attempt": 1,
            "no_retry": True,
            "consumed_before_prediction_inputs": True,
            "protected_data_access": False,
        },
    )
    return ledger


def predict(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or np.__version__ != "1.24.3":
        raise ValueError("registered NumPy 1.24.3 CPU-only runtime required")
    ledger = consume_attempt(protocol, args.output, receipt)
    args.output.mkdir(exist_ok=False)
    completed, stage, start = {}, "input-verification", time.perf_counter()
    try:
        input_barrier(protocol)
        for item in parent["objects"]:
            stage = "covariance-" + item["object"]
            arrays, mean_digest = expected_arrays(item, protocol, parent)
            directory = args.output / item["object"]
            directory.mkdir(exist_ok=False)
            files = {
                "prediction.npz": storage.save_arrays(
                    directory / "prediction.npz", arrays
                )
            }
            write_json_once(
                directory / "prediction_seal.json",
                {
                    "schema": EXPERIMENT + "-object-seal",
                    "object": item["object"],
                    "names": item["names"],
                    "files": files,
                    "registered_mean_sha256": mean_digest,
                    "point_mean_byte_identical": True,
                    "new_metrics_computed": False,
                    "protected_data_access": False,
                    "future_free_node_truth_used": False,
                },
            )
            completed[item["object"]] = {
                "seal_sha256": file_digest(directory / "prediction_seal.json"),
                "ordinary_success": len(item["names"]),
            }
            print(
                json.dumps(
                    {
                        "stage": "sealed",
                        "object": item["object"],
                        "cases": len(item["names"]),
                    }
                ),
                flush=True,
            )
        write_json_once(
            args.output / "prediction_barrier.json",
            {
                "schema": EXPERIMENT + "-prediction-barrier",
                "source_revision": receipt["revision"],
                "source_receipt_sha256": file_digest(args.source_receipt),
                "protocol_sha256": file_digest(ROOT / PROTOCOL),
                "input_prediction_barrier_sha256": protocol[
                    "input_prediction_barrier_sha256"
                ],
                "attempt_ledger_sha256": file_digest(ledger),
                "objects": completed,
                "ordinary_success": 30,
                "analysis_case_count": 29,
                "retained_technical_failure": 0,
                "unsealable": 0,
                "point_mean_byte_identical": True,
                "new_metrics_computed": False,
                "protected_data_access": False,
                "new_native_rollouts": False,
                "runtime": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "device": "cpu",
                },
                "elapsed_seconds": time.perf_counter() - start,
            },
        )
    except Exception as error:
        write_json_once(
            args.output / "failure.json",
            {
                "stage": stage,
                "type": type(error).__name__,
                "message": str(error),
                "completed_objects": completed,
                "no_retry": True,
                "protected_data_access": False,
            },
        )
        raise


def validate_barrier(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    if (
        args.output.resolve() != Path(protocol["run_root"]).resolve()
        or (args.output / "failure.json").exists()
    ):
        raise ValueError(
            "unregistered root or retained failure blocks calibration/scoring"
        )
    record = json.loads((args.output / "prediction_barrier.json").read_text())
    expected = {
        "schema": EXPERIMENT + "-prediction-barrier",
        "source_revision": receipt["revision"],
        "source_receipt_sha256": file_digest(args.source_receipt),
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "input_prediction_barrier_sha256": protocol["input_prediction_barrier_sha256"],
        "attempt_ledger_sha256": file_digest(Path(protocol["attempt_ledger"])),
        "ordinary_success": 30,
        "analysis_case_count": 29,
        "retained_technical_failure": 0,
        "unsealable": 0,
        "point_mean_byte_identical": True,
        "new_metrics_computed": False,
        "protected_data_access": False,
        "new_native_rollouts": False,
    }
    if any(record.get(k) != v for k, v in expected.items()) or set(
        record["objects"]
    ) != {"DLO1", "DLO2", "DLO3"}:
        raise ValueError("prediction denominator, identity, or custody differs")
    ledger = json.loads(Path(protocol["attempt_ledger"]).read_text())
    if ledger != {
        "schema": EXPERIMENT + "-attempt",
        "source_revision": receipt["revision"],
        "protocol_sha256": file_digest(ROOT / PROTOCOL),
        "run_root": str(args.output.resolve()),
        "attempt": 1,
        "no_retry": True,
        "consumed_before_prediction_inputs": True,
        "protected_data_access": False,
    }:
        raise ValueError("write-once attempt receipt differs")
    result = {}
    for item in parent["objects"]:
        directory = args.output / item["object"]
        entry = record["objects"][item["object"]]
        if entry != {
            "seal_sha256": file_digest(directory / "prediction_seal.json"),
            "ordinary_success": len(item["names"]),
        }:
            raise ValueError("object accounting or seal differs")
        seal = json.loads((directory / "prediction_seal.json").read_text())
        expected_values, digest = expected_arrays(item, protocol, parent)
        if set(seal["files"]) != {"prediction.npz"} or any(
            seal.get(k) != v
            for k, v in {
                "schema": EXPERIMENT + "-object-seal",
                "object": item["object"],
                "names": item["names"],
                "registered_mean_sha256": digest,
                "point_mean_byte_identical": True,
                "new_metrics_computed": False,
                "protected_data_access": False,
                "future_free_node_truth_used": False,
            }.items()
        ):
            raise ValueError("object identity, mean, or information boundary changed")
        arrays = storage.verified_arrays(
            directory / "prediction.npz", seal["files"]["prediction.npz"]
        )
        if set(arrays) != set(expected_values) or any(
            array_digest(arrays[key]) != array_digest(value)
            for key, value in expected_values.items()
        ):
            raise ValueError(
                "sealed prediction differs from registered deterministic inputs"
            )
        result[item["object"]] = arrays
    return result


def analysis_inputs(
    arrays: dict[str, np.ndarray],
    truth: np.ndarray,
    item: dict[str, Any],
    parent: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    config = config_for_object(parent, item)
    keep = [i for i, name in enumerate(item["names"]) if name != config.design_case]
    error = arrays["mean"][keep][:, :, config.hidden_nodes].astype(np.float64) - truth[
        keep
    ][:, :, config.hidden_nodes].astype(np.float64)
    return error, {
        arm: arrays[arm][keep][:, :, config.hidden_nodes] for arm in RAW_ARMS
    }


def calibrate(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    arrays = validate_barrier(args, protocol, parent, receipt)
    item = next(x for x in parent["objects"] if x["object"] == "DLO2")
    truth = parent_runner.truth_for(item, parent)
    error, raw = analysis_inputs(arrays["DLO2"], truth, item, parent)
    calibration = calibrate_source(error, raw, object_name="DLO2")
    write_json_once(
        args.output / "calibration.json",
        {
            "schema": EXPERIMENT + "-calibration",
            "source_revision": receipt["revision"],
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "prediction_barrier_sha256": file_digest(
                args.output / "prediction_barrier.json"
            ),
            "source_truth_sha256": array_digest(truth),
            "object": "DLO2",
            "case_count": 13,
            "excluded_design_case": "103.pkl",
            "transfer_metrics_computed": False,
            "protected_data_access": False,
            **calibration,
        },
    )
    print(
        json.dumps(
            {
                "stage": "calibration-sealed",
                "sha256": file_digest(args.output / "calibration.json"),
            }
        ),
        flush=True,
    )


def validate_calibration(
    args: argparse.Namespace, receipt: dict[str, Any]
) -> dict[str, Any]:
    path = args.output / "calibration.json"
    if not args.calibration_sha256 or file_digest(path) != args.calibration_sha256:
        raise ValueError(
            "explicit sealed source-calibration digest required before transfer"
        )
    value = json.loads(path.read_text())
    if any(
        value.get(k) != v
        for k, v in {
            "schema": EXPERIMENT + "-calibration",
            "source_revision": receipt["revision"],
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "prediction_barrier_sha256": file_digest(
                args.output / "prediction_barrier.json"
            ),
            "object": "DLO2",
            "case_count": 13,
            "excluded_design_case": "103.pkl",
            "transfer_metrics_computed": False,
            "protected_data_access": False,
        }.items()
    ) or set(value["scales"]) != set(FAMILIES):
        raise ValueError("source calibration fields or boundary differ")
    matrices = np.asarray(value["source_full_matrices_m2"])
    if matrices.shape != (3, 3, 3):
        raise ValueError("source-full horizon matrices differ")
    validate_covariance(matrices)
    for scales in value["scales"].values():
        if set(scales) != set(VARIANTS) or any(
            len(v) != 3 or any(not np.isfinite(x) or x <= 0 for x in v)
            for v in scales.values()
        ):
            raise ValueError("source-only calibration scale is invalid")
    return value


def score(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    parent: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    arrays = validate_barrier(args, protocol, parent, receipt)
    calibration = validate_calibration(args, receipt)
    results = {}
    for item in parent["objects"]:
        truth = parent_runner.truth_for(item, parent)
        error, raw = analysis_inputs(arrays[item["object"]], truth, item, parent)
        if (
            item["object"] == "DLO2"
            and array_digest(truth) != calibration["source_truth_sha256"]
        ):
            raise ValueError("source truth changed after calibration")
        uq = {
            f"{arm}__{variant}": summarize_uq(
                error, calibrated_covariance(raw, calibration, arm, variant)
            )
            for arm in FAMILIES
            for variant in VARIANTS
        }
        per_case = {
            "coordinate_l1_mm": np.abs(error).mean(axis=(1, 2, 3)) * 1000,
            "point_rmse_mm": np.sqrt(np.square(error).sum(axis=-1).mean(axis=(1, 2)))
            * 1000,
        }
        results[item["object"]] = {
            "uq": uq,
            "point_mean_sha256": array_digest(arrays[item["object"]]["mean"]),
            "point": {k: float(v.mean()) for k, v in per_case.items()},
            "point_per_case": {k: v.tolist() for k, v in per_case.items()},
        }
    decision = primary_decision(results, mean_identity=True, accounted_cases=30)
    aggregate = {
        f"{arm}__{variant}": {
            metric: float(
                np.mean(
                    [
                        results[name]["uq"][f"{arm}__{variant}"]["summary"][metric]
                        for name in ("DLO1", "DLO3")
                    ]
                )
            )
            for metric in results["DLO1"]["uq"][f"{arm}__{variant}"]["summary"]
        }
        for arm in FAMILIES
        for variant in VARIANTS
    }
    write_json_once(
        args.output / "result.json",
        {
            "schema": EXPERIMENT + "-result",
            "source_revision": receipt["revision"],
            "source_receipt_sha256": file_digest(args.source_receipt),
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "prediction_barrier_sha256": file_digest(
                args.output / "prediction_barrier.json"
            ),
            "calibration_sha256": args.calibration_sha256,
            "objects": results,
            "equal_object_transfer": aggregate,
            "decision": decision,
            "ordinary_success": 30,
            "calibration_cases": 13,
            "transfer_cases": 16,
            "retained_technical_failure": 0,
            "unsealable": 0,
            "new_native_rollouts": 0,
            "point_mean_byte_identical": True,
            "original_results_modified": False,
            "protected_data_access": False,
            "population_confirmation_or_sota_claim": False,
            "outcome_publication": "local-or-private-paper-evidence-only",
        },
    )
    print(
        json.dumps({"stage": "scored", "decision": decision, "transfer": aggregate}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("freeze", "predict", "validate", "calibrate", "score")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--source-receipt-sha256")
    parser.add_argument("--calibration-sha256", default="")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.output)
        return
    if args.source_receipt is None or not args.source_receipt_sha256:
        parser.error("source receipt and expected digest required")
    receipt = parent_runner.multi.native.verify_source(
        args.source_receipt, args.source_receipt_sha256
    )
    if receipt.get("experiment") != EXPERIMENT:
        raise ValueError("source receipt is for another experiment")
    protocol, parent = load_protocol(ROOT / PROTOCOL, ROOT)
    if args.command == "predict":
        predict(args, protocol, parent, receipt)
    elif args.command == "calibrate":
        calibrate(args, protocol, parent, receipt)
    elif args.command == "score":
        score(args, protocol, parent, receipt)
    else:
        validate_barrier(args, protocol, parent, receipt)
        print(
            json.dumps({"barrier_valid": True, "new_metrics_computed": False}),
            flush=True,
        )


if __name__ == "__main__":
    main()
