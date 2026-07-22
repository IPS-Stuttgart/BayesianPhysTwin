#!/usr/bin/env python3
"""Evaluate the frozen two-case MatPhys continuation gate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from aggregate_matphys_transductive_sweep import METRICS, _load_result
from bayesian_phystwin.matphys_causal_bridge import sha256_file


GATE_BASELINES = {
    "double_lift_zebra": {
        "chamfer_distance_m": 0.0142304847,
        "track_error_m": 0.0258773045,
    },
    "double_lift_cloth_1": {
        "chamfer_distance_m": 0.0132650528,
        "track_error_m": 0.0232141205,
    },
}


def _validate_bound_file(result: Mapping[str, object], key: str) -> dict[str, object]:
    record = result.get(key)
    if not isinstance(record, Mapping):
        raise ValueError(f"{key} provenance is missing")
    path = Path(str(record.get("path", ""))).resolve()
    expected_sha256 = str(record.get("sha256", ""))
    expected_size = record.get("size_bytes")
    if not path.is_file():
        raise ValueError(f"{key} provenance path is missing: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{key} provenance hash changed: {path}")
    if expected_size is None or path.stat().st_size != int(expected_size):
        raise ValueError(f"{key} provenance size changed: {path}")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "size_bytes": int(expected_size),
    }


def evaluate_gate(
    result_paths: Iterable[Path],
    *,
    baselines: Mapping[str, Mapping[str, float]] = GATE_BASELINES,
    max_regression_fraction: float = 0.10,
) -> dict[str, object]:
    if not 0.0 <= max_regression_fraction < 1.0:
        raise ValueError("max regression fraction must be in [0, 1)")

    records: dict[str, dict[str, object]] = {}
    claim_boundaries = set()
    for raw_path in result_paths:
        path = Path(raw_path).resolve()
        result = _load_result(path)
        case_name = str(result["case_name"])
        if case_name in records:
            raise ValueError(f"duplicate result for {case_name}")
        test = result["official_evaluation"]["evaluation"]["test"]
        values = {metric: float(test[metric]) for metric in METRICS}
        if int(test["frame_count"]) <= 0 or not np.isfinite(list(values.values())).all():
            raise ValueError(f"{case_name}: invalid official test metrics")
        claim_boundaries.add(str(result.get("claim_boundary", "")))
        records[case_name] = {
            "metrics": values,
            "result": {"path": str(path), "sha256": sha256_file(path)},
            "checkpoint": _validate_bound_file(result, "checkpoint"),
            "trajectory": _validate_bound_file(result, "trajectory"),
            "training_audit": _validate_bound_file(result, "training_audit"),
        }

    expected = set(baselines)
    observed = set(records)
    if observed != expected:
        raise ValueError(
            "gate cohort mismatch; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    if len(claim_boundaries) != 1 or not next(iter(claim_boundaries)):
        raise ValueError("result claim boundaries are missing or inconsistent")

    mean_checks = {}
    per_case_checks = {}
    for metric in METRICS:
        candidate_mean = float(
            np.mean([records[case]["metrics"][metric] for case in sorted(expected)])
        )
        baseline_mean = float(
            np.mean([float(baselines[case][metric]) for case in sorted(expected)])
        )
        mean_checks[metric] = {
            "candidate_case_balanced_mean_m": candidate_mean,
            "released_case_balanced_mean_m": baseline_mean,
            "improves": candidate_mean < baseline_mean,
        }

    for case_name in sorted(expected):
        case_checks = {}
        for metric in METRICS:
            candidate = float(records[case_name]["metrics"][metric])
            baseline = float(baselines[case_name][metric])
            relative_change = candidate / baseline - 1.0
            case_checks[metric] = {
                "candidate_m": candidate,
                "released_m": baseline,
                "relative_change": relative_change,
                "within_regression_cap": relative_change <= max_regression_fraction,
            }
        per_case_checks[case_name] = case_checks

    passed = all(item["improves"] for item in mean_checks.values()) and all(
        check["within_regression_cap"]
        for case_checks in per_case_checks.values()
        for check in case_checks.values()
    )
    return {
        "schema_version": 1,
        "contract": "matphys-offline-all-frame-reconstruction-gate-v1",
        "claim_boundary": next(iter(claim_boundaries)),
        "future_observations_used": True,
        "released_test_outcomes_used_in_objective": True,
        "max_regression_fraction": float(max_regression_fraction),
        "mean_checks": mean_checks,
        "per_case_checks": per_case_checks,
        "inputs": records,
        "passed": bool(passed),
        "decision": "continue_full_22" if passed else "stop_before_full_22",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-regression-fraction", type=float, default=0.10)
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    result_paths = list(
        sweep_root.glob("cases/*/export/transductive_reconstruction_result.json")
    )
    decision = evaluate_gate(
        result_paths,
        max_regression_fraction=args.max_regression_fraction,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **decision}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
