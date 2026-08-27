#!/usr/bin/env python3
"""Independently recompute every child-branch metric from sealed source arrays."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)
from bayesian_phystwin_experiments.deft_cross_branch_source import (
    ARMS,
    PRIMARY,
    SOURCE_FILE_SHA256,
    _NumericUnpickler,
)


def independent_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    errors = prediction - target
    lengths = np.linalg.norm(errors, axis=-1)
    return {
        "coordinate_l1_mm": float(np.mean(np.abs(errors)) * 1000),
        "point_rmse_mm": float(np.sqrt(np.mean(lengths**2)) * 1000),
        "fde_mm": float(np.mean(lengths[-1]) * 1000),
        "late_rmse_mm": float(np.sqrt(np.mean(lengths[80:] ** 2)) * 1000),
    }


def verify(root: Path, source: Path) -> dict[str, object]:
    result = json.loads((root / "result.json").read_text())
    barrier_path = root / "prediction_barrier.json"
    if file_digest(barrier_path) != result["prediction_barrier_sha256"]:
        raise ValueError("result does not bind its exact prediction barrier")
    barrier = json.loads(barrier_path.read_text())
    if file_digest(root / "predictions.npz") != barrier["prediction_file_sha256"]:
        raise ValueError("sealed prediction archive changed")
    if file_digest(source) != SOURCE_FILE_SHA256:
        raise ValueError("source recording changed")
    raw = np.asarray(
        _NumericUnpickler(io.BytesIO(source.read_bytes())).load(), dtype=np.float64
    ).reshape(3, 500, 20)
    # Independent material-identity/axis mapping; no padded branch construction.
    truth = np.stack((-raw[2], -raw[0], raw[1]), axis=-1)[52:172]
    comparisons = 0
    recomputed = {}
    with np.load(root / "predictions.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(ARMS):
            raise ValueError("prediction arm denominator changed")
        for arm in ARMS:
            points = archive[arm]
            if array_digest(points) != barrier["array_sha256s"][arm]:
                raise ValueError("prediction array identity changed")
            rows = {}
            for branch, material_slice, native_slice in (
                (1, slice(13, 17), slice(1, 5)),
                (2, slice(17, 20), slice(1, 4)),
            ):
                values = independent_metrics(
                    points[:, branch, native_slice], truth[:, material_slice]
                )
                rows[f"child{branch}"] = values
                for name, value in values.items():
                    if not np.isclose(
                        value,
                        result["per_arm"][arm][f"child{branch}"][name],
                        rtol=1e-12,
                        atol=1e-9,
                    ):
                        raise ValueError(f"metric differs: {arm}/child{branch}/{name}")
                    comparisons += 1
            for name in rows["child1"]:
                value = (rows["child1"][name] + rows["child2"][name]) / 2
                if not np.isclose(
                    value,
                    result["per_arm"][arm]["equal_child_branch"][name],
                    rtol=1e-12,
                    atol=1e-9,
                ):
                    raise ValueError("equal-branch aggregate differs")
                comparisons += 1
            recomputed[arm] = rows
    checks = {}
    for child in ("child1", "child2"):
        candidate = recomputed[PRIMARY][child]
        for arm in ("native_full", "readout_persistence", "readout_linear_velocity"):
            baseline = recomputed[arm][child]
            checks[f"{child}_rmse_at_least_5pct_better_than_{arm}"] = (
                baseline["point_rmse_mm"] > 0
                and candidate["point_rmse_mm"] <= 0.95 * baseline["point_rmse_mm"]
            )
            checks[f"{child}_l1_nonworsening_vs_{arm}"] = (
                candidate["coordinate_l1_mm"] <= baseline["coordinate_l1_mm"]
            )
            checks[f"{child}_late_nonworsening_vs_{arm}"] = (
                candidate["late_rmse_mm"] <= baseline["late_rmse_mm"]
            )
    if (
        checks != result["checks"]
        or all(checks.values()) != result["source_pilot_gate_passed"]
    ):
        raise ValueError("primary gate decision differs")
    if (
        result["ordinary_successful_recordings"] != 1
        or result["technical_failures"] != 0
        or result["unsealable"] != 0
        or result["protected_data_read"] is not False
        or result["public_evaluation_or_test_split_read"] is not False
    ):
        raise ValueError("case accounting or boundary differs")
    return {
        "schema": "deft-cross-branch-source-verification-v1",
        "verification_passed": True,
        "result_sha256": file_digest(root / "result.json"),
        "barrier_sha256": file_digest(barrier_path),
        "verified_child_and_aggregate_metrics": comparisons,
        "verified_gate_checks": len(checks),
        "verified_prediction_arms": len(ARMS),
        "hidden_child_identity_events_per_arm": 120 * 7,
        "source_pilot_gate_passed": result["source_pilot_gate_passed"],
        "protected_data_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_root, args.training_source)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
