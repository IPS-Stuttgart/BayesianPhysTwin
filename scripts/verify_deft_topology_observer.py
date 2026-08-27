#!/usr/bin/env python3
"""Independent raw-identity mapping, metric arithmetic, and source gate replay."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin_experiments.deform_state_restart import (
    array_digest,
    file_digest,
    write_json_once,
)
from bayesian_phystwin_experiments.deft_cross_branch_source import _NumericUnpickler
from bayesian_phystwin_experiments.deft_topology_observer import (
    ARMS,
    CASE_IDS,
    COMPARATORS,
    PRIMARY,
)


def independent_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = prediction - truth
    squared_distance = np.sum(error * error, axis=-1)
    return {
        "point_rmse_mm": float(np.sqrt(np.mean(squared_distance)) * 1000),
        "coordinate_l1_mm": float(np.mean(np.abs(error)) * 1000),
        "late_rmse_mm": float(np.sqrt(np.mean(squared_distance[80:])) * 1000),
        "fde_mm": float(np.mean(np.sqrt(squared_distance[-1])) * 1000),
    }


def verify(root: Path, training: Path, protocol_path: Path) -> dict[str, Any]:
    result = json.loads((root / "result.json").read_text())
    if file_digest(protocol_path) != result["protocol_sha256"]:
        raise ValueError("protocol binding changed")
    protocol = json.loads(protocol_path.read_text())
    if tuple(x["id"] for x in protocol["source_cases"]) != CASE_IDS:
        raise ValueError("original recording denominator changed")
    if (
        file_digest(root / "prediction_barrier.json")
        != result["prediction_barrier_sha256"]
    ):
        raise ValueError("prediction barrier binding changed")
    barrier = json.loads((root / "prediction_barrier.json").read_text())
    if (
        set(barrier["cases"]) != set(CASE_IDS)
        or barrier["source_future_scoring_opened"] is not False
    ):
        raise ValueError("the full roster was not sealed before scoring")
    predictions = {}
    for case in CASE_IDS:
        record = barrier["cases"][case]
        case_root = root / case
        if (
            record["status"] != "ordinary-success"
            or file_digest(case_root / "prediction_seal.json") != record["seal_sha256"]
        ):
            raise ValueError("case seal identity or status changed")
        seal = json.loads((case_root / "prediction_seal.json").read_text())
        if (
            file_digest(case_root / "predictions.npz") != seal["prediction_sha256"]
            or seal["controls"]["zero_update_byte_identical"] is not True
        ):
            raise ValueError("prediction archive or exact-zero fallback changed")
        with np.load(case_root / "predictions.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if (
            set(arrays) != set(ARMS)
            or any(
                x.shape != (120, 3, 13, 3) or not np.isfinite(x).all()
                for x in arrays.values()
            )
            or {key: array_digest(x) for key, x in arrays.items()}
            != seal["array_sha256s"]
        ):
            raise ValueError("prediction array identities changed")
        predictions[case] = arrays
    rows: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    metric_count = 0
    for spec in protocol["source_cases"]:
        path = training / spec["filename"]
        if file_digest(path) != spec["sha256"]:
            raise ValueError("source recording checksum changed")
        raw = np.asarray(
            _NumericUnpickler(io.BytesIO(path.read_bytes())).load(), dtype=np.float64
        ).reshape(3, 500, 20)
        truth = np.stack((-raw[2], -raw[0], raw[1]), axis=-1)[52:172]
        case = spec["id"]
        rows[case] = {}
        for arm in ARMS:
            regions = {}
            for region, branch, native_nodes, raw_nodes in (
                ("parent", 0, [3, 5, 7, 9, 10], [3, 5, 7, 9, 10]),
                ("child1", 1, [1, 2, 3], [13, 14, 15]),
                ("child2", 2, [1, 2], [17, 18]),
            ):
                regions[region] = independent_metrics(
                    predictions[case][arm][:, branch][:, native_nodes],
                    truth[:, raw_nodes],
                )
            regions["equal_child_branch"] = {
                key: (regions["child1"][key] + regions["child2"][key]) / 2
                for key in regions["child1"]
            }
            for region, metrics in regions.items():
                for key, value in metrics.items():
                    if not np.isclose(
                        value,
                        result["per_recording"][case][arm][region][key],
                        rtol=1e-12,
                        atol=1e-9,
                    ):
                        raise ValueError("independent per-recording metric differs")
                    metric_count += 1
            rows[case][arm] = regions
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for region in ("parent", "child1", "child2", "equal_child_branch"):
            aggregate[arm][region] = {}
            for key in ("point_rmse_mm", "coordinate_l1_mm", "late_rmse_mm", "fde_mm"):
                value = sum(rows[case][arm][region][key] for case in CASE_IDS) / 3
                if not np.isclose(
                    value,
                    result["equal_recording_mean"][arm][region][key],
                    rtol=1e-12,
                    atol=1e-9,
                ):
                    raise ValueError("independent equal-recording aggregate differs")
                aggregate[arm][region][key] = value
                metric_count += 1
    checks = {}
    wins = {}
    for comparator in COMPARATORS:
        wins[comparator] = sum(
            rows[case][PRIMARY]["equal_child_branch"]["point_rmse_mm"]
            < rows[case][comparator]["equal_child_branch"]["point_rmse_mm"]
            and rows[case][PRIMARY]["equal_child_branch"]["coordinate_l1_mm"]
            <= rows[case][comparator]["equal_child_branch"]["coordinate_l1_mm"]
            for case in CASE_IDS
        )
        checks[f"at_least_two_recording_joint_wins_vs_{comparator}"] = (
            wins[comparator] >= 2
        )
        for child in ("child1", "child2"):
            candidate, baseline = (
                aggregate[PRIMARY][child],
                aggregate[comparator][child],
            )
            checks[f"{child}_rmse_gain_5pct_vs_{comparator}"] = (
                baseline["point_rmse_mm"] > 0
                and candidate["point_rmse_mm"] <= 0.95 * baseline["point_rmse_mm"]
            )
            for key in ("coordinate_l1_mm", "late_rmse_mm"):
                checks[f"{child}_{key}_nonincreasing_vs_{comparator}"] = (
                    candidate[key] <= baseline[key]
                )
    checks["no_recording_more_than_10pct_worse_than_native"] = all(
        rows[case][PRIMARY]["equal_child_branch"]["point_rmse_mm"]
        <= 1.10 * rows[case]["native_full"]["equal_child_branch"]["point_rmse_mm"]
        for case in CASE_IDS
    )
    if (
        checks != result["checks"]
        or wins != result["recording_joint_wins"]
        or all(checks.values()) != result["source_gate_passed"]
    ):
        raise ValueError("independent source decision differs")
    if (
        result["ordinary_successful_recordings"] != 3
        or result["technical_failures"] != 0
        or result["unsealable"] != 0
        or result["protected_data_read"] is not False
        or result["independent_confirmation"] is not False
    ):
        raise ValueError("recording accounting or claim boundary changed")
    return {
        "schema": "deft-topology-observer-independent-verification-v1",
        "verification_passed": True,
        "metrics_verified": metric_count,
        "gate_checks_verified": len(checks),
        "prediction_arms_verified": 3 * len(ARMS),
        "ordinary_recordings": 3,
        "physical_objects": 1,
        "source_gate_passed": result["source_gate_passed"],
        "result_sha256": file_digest(root / "result.json"),
        "prediction_barrier_sha256": file_digest(root / "prediction_barrier.json"),
        "protected_data_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run_root, args.training_root, args.protocol)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
