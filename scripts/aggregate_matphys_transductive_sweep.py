#!/usr/bin/env python3
"""Aggregate audited MatPhys all-frame controls without changing their meaning."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from bayesian_phystwin.matphys_causal_bridge import sha256_file


TRANSDUCTIVE_CONTRACT = "matphys-offline-all-frame-reconstruction-v1"
METRICS = ("chamfer_distance_m", "track_error_m")


def _load_result(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("contract") != TRANSDUCTIVE_CONTRACT:
        raise ValueError(f"{path}: wrong artifact contract")
    if result.get("future_observations_used") is not True:
        raise ValueError(f"{path}: future-observation disclosure is missing")
    if result.get("released_test_outcomes_used_in_objective") is not True:
        raise ValueError(f"{path}: fitted-test disclosure is missing")
    evaluation = result.get("official_evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{path}: official evaluation is missing")
    test = evaluation.get("evaluation", {}).get("test")
    if not isinstance(test, Mapping):
        raise ValueError(f"{path}: official test metrics are missing")
    if any(metric not in test for metric in METRICS) or "frame_count" not in test:
        raise ValueError(f"{path}: official test metric set is incomplete")
    return result


def aggregate_results(
    result_paths: Iterable[Path],
    *,
    expected_cases: set[str],
    reference_cd_m: float = 0.008,
    reference_track_m: float = 0.015,
) -> dict[str, object]:
    records = []
    seen = set()
    claim_boundaries = set()
    for path in sorted(Path(item).resolve() for item in result_paths):
        result = _load_result(path)
        case_name = str(result["case_name"])
        if case_name in seen:
            raise ValueError(f"duplicate result for {case_name}")
        seen.add(case_name)
        claim_boundaries.add(str(result.get("claim_boundary", "")))
        test = result["official_evaluation"]["evaluation"]["test"]
        records.append(
            {
                "case_name": case_name,
                "frame_count": int(test["frame_count"]),
                **{metric: float(test[metric]) for metric in METRICS},
                "result": {
                    "path": str(path),
                    "sha256": sha256_file(path),
                },
                "checkpoint": result["checkpoint"],
                "trajectory": result["trajectory"],
            }
        )
    if seen != expected_cases:
        missing = sorted(expected_cases - seen)
        extra = sorted(seen - expected_cases)
        raise ValueError(f"result cohort mismatch; missing={missing}, extra={extra}")
    if len(claim_boundaries) != 1 or not next(iter(claim_boundaries)):
        raise ValueError("result claim boundaries are missing or inconsistent")

    frame_counts = np.asarray([record["frame_count"] for record in records], dtype=float)
    if np.any(frame_counts <= 0):
        raise ValueError("all test intervals must contain frames")
    aggregates = {}
    references = {
        "chamfer_distance_m": float(reference_cd_m),
        "track_error_m": float(reference_track_m),
    }
    for metric in METRICS:
        values = np.asarray([record[metric] for record in records], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{metric} contains non-finite values")
        case_mean = float(values.mean())
        frame_mean = float(np.average(values, weights=frame_counts))
        reference = references[metric]
        aggregates[metric] = {
            "case_balanced_mean_m": case_mean,
            "frame_weighted_mean_m": frame_mean,
            "published_reference_m": reference,
            "case_balanced_percent_vs_reference": 100.0 * (case_mean / reference - 1.0),
            "frame_weighted_percent_vs_reference": 100.0 * (frame_mean / reference - 1.0),
        }
    return {
        "schema_version": 1,
        "contract": "matphys-offline-all-frame-reconstruction-aggregate-v1",
        "claim_boundary": next(iter(claim_boundaries)),
        "future_observations_used": True,
        "released_test_outcomes_used_in_objective": True,
        "case_count": len(records),
        "test_frame_count": int(frame_counts.sum()),
        "aggregation": {
            "primary": "case-balanced mean of official per-case test metrics",
            "secondary": "test-frame-count-weighted mean",
        },
        "metrics": aggregates,
        "cases": records,
    }


def _expected_cases(data_root: Path) -> set[str]:
    cases = {path.parent.name for path in data_root.glob("*/split.json")}
    if not cases:
        raise FileNotFoundError(f"no PhysTwin cases under {data_root}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-cd-m", type=float, default=0.008)
    parser.add_argument("--reference-track-m", type=float, default=0.015)
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    result_paths = list(
        sweep_root.glob("cases/*/export/transductive_reconstruction_result.json")
    )
    summary = aggregate_results(
        result_paths,
        expected_cases=_expected_cases(Path(args.data_root).resolve()),
        reference_cd_m=args.reference_cd_m,
        reference_track_m=args.reference_track_m,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
