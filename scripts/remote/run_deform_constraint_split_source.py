#!/usr/bin/env python3
"""Local CPU-only prediction/seal/score stages for the fixed DLO2 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_constraint_split import (
    ARMS,
    PRIMARY,
    SCHEMA,
    SplitConfig,
    config_record,
    content_id,
    score_arrays,
    split_forecast,
)
from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "configs/sota/deform_constraint_split_source_v1.json"
BOUND_SOURCE = (
    PROTOCOL,
    "src/bayesian_phystwin_experiments/deform_constraint_split.py",
    "src/bayesian_phystwin_experiments/deform_state_restart.py",
    "scripts/remote/run_deform_constraint_split_source.py",
    "tests/test_deform_constraint_split.py",
    "docs/deform_constraint_split_source_v1.md",
)
OLD_KEYS = {
    "incumbent": "incumbent",
    "paired": "incumbent_propagated_pose_velocity",
    "readout": "readout_sparse_pose",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_identity(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    result = {**value, "artifact_id": content_id(value)}
    write_json_once(path, result)
    return result


def read_identity(path: Path) -> dict[str, Any]:
    result = read_json(path)
    identity = {k: v for k, v in result.items() if k != "artifact_id"}
    if result.get("artifact_id") != content_id(identity):
        raise ValueError("artifact content binding changed")
    return result


def input_path(protocol: dict[str, Any], label: str) -> Path:
    path = Path(protocol["inputs"][label]["path"])
    return path if path.is_absolute() else ROOT / path


def checked_input(protocol: dict[str, Any], label: str) -> Path:
    path = input_path(protocol, label)
    if file_digest(path) != protocol["inputs"][label]["sha256"]:
        raise ValueError(f"frozen {label} changed")
    return path


def load_protocol() -> dict[str, Any]:
    protocol = read_json(ROOT / PROTOCOL)
    expected = {
        "schema": SCHEMA,
        "scope": "already-open-DLO2-only-exploratory-screen",
        "primary_arm": PRIMARY,
        "arms": list(ARMS),
        "config": config_record(SplitConfig()),
        "prediction_count": 14,
        "analysis_count": 13,
        "excluded_design_case": "103.pkl",
        "prediction_frames": [50, 170],
        "observations": {"frames": [41, 49], "identities": [2, 4, 6, 8], "count": 8},
        "geometry": "unchanged_native_nominal_future",
        "inner_product": "equal_weight_Euclidean_on_free_xyz_coordinates",
        "method": "incumbent + P_t(paired-incumbent) + (I-P_t)last_prefix_offset",
        "all_predictions_sealed_before_new_source_metrics": True,
        "attempts": 1,
    }
    false_flags = (
        "native_replays_performed",
        "upstream_or_incumbent_modified",
        "new_recordings",
        "gpu_access",
        "transfer_objects_accessed",
        "protected_data_access",
        "official_sota_claim",
        "automatic_promotion",
    )
    if any(protocol.get(k) != v for k, v in expected.items()) or any(
        protocol.get(k) is not False for k in false_flags
    ):
        raise ValueError("method, roster, or information boundary changed")
    if protocol["gate"] != {
        "minimum_rmse_gain_over_every_control_percent": 2.0,
        "lower_l1_than_every_control": True,
        "minimum_joint_wins_over_paired": 9,
        "late_rmse_no_worse_than_incumbent_and_paired": True,
        "maximum_case_rmse_ratio_to_incumbent": 1.05,
        "rmse_difference_ci95_upper_below_zero_against_every_control": True,
        "all_predictions_ordinary": True,
    }:
        raise ValueError("source gate changed")
    if protocol["inputs"]["source_truth"]["prediction_index_offset"] != 0:
        raise ValueError("the existing archive already uses prediction-frame indices")
    return protocol


def registered_names(protocol: dict[str, Any]) -> list[str]:
    parent = read_json(checked_input(protocol, "parent_protocol"))
    item = next(item for item in parent["objects"] if item["object"] == "DLO2")
    return item["names"]


def freeze(protocol: dict[str, Any]) -> None:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True):
        raise ValueError("commit the complete source before freezing")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    output = Path(protocol["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    receipt = write_identity(
        output / "source-lock.json",
        {
            "schema": SCHEMA,
            "revision": revision,
            "source_files": {name: file_digest(ROOT / name) for name in BOUND_SOURCE},
            "protocol_sha256": file_digest(ROOT / PROTOCOL),
            "output_root": str(output),
            "source_truth_opened": False,
            "protected_data_access": False,
            "git_clean": True,
        },
    )
    print(
        json.dumps({"stage": "frozen", "artifact_id": receipt["artifact_id"]}),
        flush=True,
    )


def validate_lock(protocol: dict[str, Any]) -> dict[str, Any]:
    output = Path(protocol["output_root"])
    receipt = read_identity(output / "source-lock.json")
    if (
        receipt["schema"] != SCHEMA
        or receipt["output_root"] != str(output)
        or receipt["protocol_sha256"] != file_digest(ROOT / PROTOCOL)
        or set(receipt["source_files"]) != set(BOUND_SOURCE)
        or receipt["source_truth_opened"] is not False
        or receipt["protected_data_access"] is not False
        or receipt["git_clean"] is not True
    ):
        raise ValueError("source lock or boundary changed")
    for path, digest in receipt["source_files"].items():
        if file_digest(ROOT / path) != digest:
            raise ValueError(f"bound implementation changed: {path}")
    revision = receipt["revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("source revision must be a full Git commit")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"], cwd=ROOT, check=True
    )
    for path, digest in receipt["source_files"].items():
        blob = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != digest:
            raise ValueError(
                "source receipt does not bind committed implementation bytes"
            )
    if (
        platform.python_version() != protocol["runtime"]["python"]
        or np.__version__ != protocol["runtime"]["numpy"]
        or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.environ.get("OPENBLAS_NUM_THREADS") != "1"
        or os.environ.get("OMP_NUM_THREADS") != "1"
    ):
        raise ValueError("frozen single-threaded CPU runtime required")
    return receipt


def load_forecasts(protocol: dict[str, Any]) -> tuple[list[str], dict[str, np.ndarray]]:
    names = registered_names(protocol)
    checked_input(protocol, "source_receipt")
    seal = read_json(checked_input(protocol, "prediction_seal"))
    if seal["names"] != names or seal["object"] != "DLO2":
        raise ValueError("only the registered opened DLO2 roster is permitted")
    archive = checked_input(protocol, "clean")
    if seal["files"]["clean"]["sha256"] != file_digest(archive):
        raise ValueError("parent prediction seal does not bind the archive")
    with np.load(archive, allow_pickle=False) as data:
        keys = ("names", *OLD_KEYS.values(), "physical_nominal")
        arrays = {key: data[key].copy() for key in keys}
    if arrays["names"].tolist() != names or any(
        array_digest(value) != seal["files"]["clean"]["arrays"][key]
        for key, value in arrays.items()
    ):
        raise ValueError("parent array identity or roster changed")
    return names, {arm: arrays[key] for arm, key in OLD_KEYS.items()} | {
        "nominal": arrays["physical_nominal"]
    }


def predict(protocol: dict[str, Any], receipt: dict[str, Any]) -> None:
    output = Path(protocol["output_root"])
    write_identity(
        output / "prediction-attempt.json",
        {
            "source_lock_id": receipt["artifact_id"],
            "attempt": 1,
            "output_root": str(output),
        },
    )
    names, inputs = load_forecasts(protocol)
    predictions: dict[str, list[np.ndarray]] = {arm: [] for arm in ARMS}
    diagnostics = []
    for i, name in enumerate(names):
        arms, info = split_forecast(
            inputs["incumbent"][i],
            inputs["paired"][i],
            inputs["readout"][i],
            inputs["nominal"][i],
            SplitConfig(),
        )
        for arm in ARMS:
            predictions[arm].append(arms[arm])
        diagnostics.append({"name": name, **info})
    arrays = {"names": np.asarray(names)} | {
        key: np.stack(value) for key, value in predictions.items()
    }
    for arm in OLD_KEYS:
        if array_digest(arrays[arm]) != array_digest(inputs[arm]):
            raise ValueError("an unchanged comparator was modified")
    path = output / "predictions.npz"
    with path.open("xb") as stream:
        payload: dict[str, Any] = arrays
        np.savez_compressed(stream, allow_pickle=False, **payload)
    seal = write_identity(
        output / "prediction-seal.json",
        {
            "schema": SCHEMA,
            "source_lock_id": receipt["artifact_id"],
            "attempt_sha256": file_digest(output / "prediction-attempt.json"),
            "names": names,
            "prediction_count": len(names),
            "ordinary_success": sum(item["ordinary_success"] for item in diagnostics),
            "exact_fallback_technical_failure": sum(
                item["exact_fallback"] for item in diagnostics
            ),
            "unsealable": 0,
            "predictions_sha256": file_digest(path),
            "array_sha256s": {
                key: array_digest(value) for key, value in arrays.items()
            },
            "diagnostics": diagnostics,
            "source_truth_opened": False,
            "source_metrics_computed": False,
            "protected_data_access": False,
        },
    )
    print(
        json.dumps(
            {
                "stage": "predictions-sealed",
                "artifact_id": seal["artifact_id"],
                "ordinary_success": seal["ordinary_success"],
                "fallback": seal["exact_fallback_technical_failure"],
            }
        ),
        flush=True,
    )


def validated_predictions(
    protocol: dict[str, Any], receipt: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    output = Path(protocol["output_root"])
    seal = read_identity(output / "prediction-seal.json")
    attempt = read_identity(output / "prediction-attempt.json")
    names = registered_names(protocol)
    if (
        seal["schema"] != SCHEMA
        or seal["source_lock_id"] != receipt["artifact_id"]
        or seal["names"] != names
        or seal["prediction_count"] != 14
        or seal["unsealable"] != 0
        or seal["ordinary_success"] + seal["exact_fallback_technical_failure"] != 14
        or seal["attempt_sha256"] != file_digest(output / "prediction-attempt.json")
        or attempt["source_lock_id"] != receipt["artifact_id"]
        or attempt["attempt"] != 1
        or attempt["output_root"] != str(output)
        or any(
            seal[key] is not False
            for key in (
                "source_truth_opened",
                "source_metrics_computed",
                "protected_data_access",
            )
        )
        or [item["name"] for item in seal["diagnostics"]] != names
        or sum(item["ordinary_success"] for item in seal["diagnostics"])
        != seal["ordinary_success"]
    ):
        raise ValueError("complete prediction barrier or information boundary failed")
    path = output / "predictions.npz"
    if file_digest(path) != seal["predictions_sha256"]:
        raise ValueError("prediction file changed after seal")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if (
        set(arrays) != {"names", *ARMS}
        or set(seal["array_sha256s"]) != set(arrays)
        or arrays["names"].tolist() != names
        or any(
            array_digest(value) != seal["array_sha256s"][key]
            for key, value in arrays.items()
        )
        or any(
            value.shape != (14, 120, 12, 3)
            or value.dtype != np.dtype("float64")
            or not np.isfinite(value).all()
            for key, value in arrays.items()
            if key != "names"
        )
    ):
        raise ValueError("prediction arrays or denominator differ")
    _, original = load_forecasts(protocol)
    for arm in OLD_KEYS:
        if array_digest(arrays[arm]) != array_digest(original[arm]):
            raise ValueError("registered comparator no longer byte identical")
    return arrays, seal


def score(protocol: dict[str, Any], receipt: dict[str, Any]) -> None:
    output = Path(protocol["output_root"])
    if (output / "result.json").exists():
        raise ValueError("source result already exists; no replacement or rerun")
    arrays, seal = validated_predictions(protocol, receipt)
    # This is the only truth-member access, after the entire prediction barrier.
    with np.load(
        checked_input(protocol, "source_truth"), allow_pickle=False
    ) as archive:
        if archive["names"].tolist() != seal["names"]:
            raise ValueError("source truth identities differ")
        truth = archive[protocol["inputs"]["source_truth"]["key"]][:, 50:170].copy()
    result = score_arrays(
        {arm: arrays[arm] for arm in ARMS},
        truth,
        seal["names"],
        [item["ordinary_success"] for item in seal["diagnostics"]],
        SplitConfig(),
    )
    result = write_identity(
        output / "result.json",
        {
            "schema": SCHEMA,
            "source_lock_id": receipt["artifact_id"],
            "prediction_seal_id": seal["artifact_id"],
            "prediction_seal_sha256": file_digest(output / "prediction-seal.json"),
            "source_truth_sha256": protocol["inputs"]["source_truth"]["sha256"],
            "accounting": {
                key: seal[key]
                for key in (
                    "prediction_count",
                    "ordinary_success",
                    "exact_fallback_technical_failure",
                    "unsealable",
                )
            },
            "transfer_objects_accessed": False,
            "protected_data_access": False,
            "native_replays_performed": False,
            **result,
        },
    )
    print(
        json.dumps(
            {
                "stage": "scored",
                "artifact_id": result["artifact_id"],
                "gate": result["gate"],
                "means": {
                    arm: value["mean"] for arm, value in result["metrics"].items()
                },
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "predict", "score"))
    args = parser.parse_args()
    protocol = load_protocol()
    if args.stage == "freeze":
        freeze(protocol)
    else:
        receipt = validate_lock(protocol)
        if args.stage == "predict":
            predict(protocol, receipt)
        else:
            score(protocol, receipt)


if __name__ == "__main__":
    main()
