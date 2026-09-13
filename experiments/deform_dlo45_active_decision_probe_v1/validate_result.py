"""Validate internal accounting of the committed DEFORM active-probe result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

DLOS = ("DLO4", "DLO5")
METHODS = (
    "fallback",
    "no_probe_certificate",
    "active_minimum_cost",
    "fixed_probe_3",
    "fixed_probe_6",
    "fixed_probe_12",
    "max_outcome_entropy",
    "oracle_probe_action",
)
ATOL = 1e-12


def _count(fraction: float, total: int) -> int:
    value = fraction * total
    rounded = round(value)
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"fraction {fraction} is not an integer count over {total}")
    return int(rounded)


def _aggregate(record: dict[str, Any], method: str) -> dict[str, Any]:
    aggregate = record["aggregate"]
    if not isinstance(aggregate, dict) or method not in aggregate:
        raise ValueError(f"missing aggregate method {method}")
    result = aggregate[method]
    if not isinstance(result, dict):
        raise ValueError(f"aggregate method {method} must be an object")
    return result


def validate(path: Path) -> dict[str, object]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != "bayesian-phystwin/deform-dlo45-active-probe-pilot-v1":
        raise ValueError("unexpected result schema")
    if result.get("status") != "retrospective-development-mechanism-pilot":
        raise ValueError("unexpected evidence status")
    records = result.get("dlos")
    pooled = result.get("pooled")
    protocol = result.get("pilot_protocol")
    if not isinstance(records, dict) or not isinstance(pooled, dict):
        raise ValueError("missing DLO or pooled results")
    if not isinstance(protocol, dict):
        raise ValueError("missing pilot protocol")
    probe_frames = tuple(int(value) for value in protocol["probe_frames"])
    if not probe_frames or probe_frames[0] != 0:
        raise ValueError("probe portfolio must begin with no probe")

    total_decisions = 0
    total_certified = 0
    total_no_probe = 0
    total_positive_probe = 0
    total_ambiguous_probe = 0
    for dlo in DLOS:
        record = records.get(dlo)
        if not isinstance(record, dict):
            raise ValueError(f"missing {dlo} result")
        decisions = int(record["decision_count"])
        total_decisions += decisions
        active = record.get("active")
        if not isinstance(active, dict):
            raise ValueError(f"missing {dlo} active diagnostics")
        duration_counts = active.get("duration_counts")
        if not isinstance(duration_counts, dict):
            raise ValueError(f"missing {dlo} duration counts")
        certified = sum(int(duration_counts[str(frame)]) for frame in probe_frames)
        fallback = int(duration_counts["fallback_no_certified_probe"])
        if certified + fallback != decisions:
            raise ValueError(f"{dlo} probe-selection counts do not sum to decisions")
        no_probe = int(duration_counts["0"])
        positive_probe = sum(
            int(duration_counts[str(frame)]) for frame in probe_frames if frame > 0
        )
        expected_frames = (
            sum(frame * int(duration_counts[str(frame)]) for frame in probe_frames)
            / decisions
        )
        active_aggregate = _aggregate(record, "active_minimum_cost")
        if not math.isclose(
            expected_frames,
            float(active_aggregate["mean_probe_frames"]),
            rel_tol=0.0,
            abs_tol=ATOL,
        ):
            raise ValueError(f"{dlo} mean probe cost is inconsistent")
        checks = (
            ("certified_probe_fraction", certified),
            ("no_probe_certified_fraction", no_probe),
            ("probe_required_fraction", positive_probe),
        )
        for key, expected_count in checks:
            if _count(float(active[key]), decisions) != expected_count:
                raise ValueError(f"{dlo} {key} is inconsistent")
        ambiguous = _count(
            float(active["probed_decision_with_multiple_supported_states_fraction"]),
            decisions,
        )
        if ambiguous > positive_probe:
            raise ValueError(f"{dlo} ambiguous probe count exceeds probes")
        for method in METHODS:
            aggregate = _aggregate(record, method)
            action_counts = [int(value) for value in aggregate["action_counts"]]
            if sum(action_counts) != decisions:
                raise ValueError(f"{dlo} {method} action counts are inconsistent")
            _count(float(aggregate["harm_fraction_vs_fallback"]), decisions)
            _count(float(aggregate["fallback_action_fraction"]), decisions)
        total_certified += certified
        total_no_probe += no_probe
        total_positive_probe += positive_probe
        total_ambiguous_probe += ambiguous

    for method in METHODS:
        pooled_method = pooled.get(method)
        if not isinstance(pooled_method, dict):
            raise ValueError(f"missing pooled method {method}")
        component = [_aggregate(records[dlo], method) for dlo in DLOS]
        expected_actions = [
            sum(int(item["action_counts"][action]) for item in component)
            for action in range(len(pooled_method["action_counts"]))
        ]
        if expected_actions != [int(value) for value in pooled_method["action_counts"]]:
            raise ValueError(f"pooled {method} action counts are inconsistent")
        expected_rmse = math.sqrt(
            sum(
                int(records[dlo]["decision_count"])
                * (float(_aggregate(records[dlo], method)["terminal_rmse_mm"]) ** 2)
                for dlo in DLOS
            )
            / total_decisions
        )
        if not math.isclose(
            expected_rmse,
            float(pooled_method["terminal_rmse_mm"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"pooled {method} RMSE is inconsistent")

    fallback_rmse = float(pooled["fallback"]["terminal_rmse_mm"])
    passive_rmse = float(pooled["no_probe_certificate"]["terminal_rmse_mm"])
    active_rmse = float(pooled["active_minimum_cost"]["terminal_rmse_mm"])
    active_harm = _count(
        float(pooled["active_minimum_cost"]["harm_fraction_vs_fallback"]),
        total_decisions,
    )
    entropy_harm = _count(
        float(pooled["max_outcome_entropy"]["harm_fraction_vs_fallback"]),
        total_decisions,
    )
    if total_certified != total_no_probe + total_positive_probe:
        raise ValueError("certified-probe decomposition is inconsistent")
    return {
        "decision_count": total_decisions,
        "certified_decisions": total_certified,
        "certified_without_probe": total_no_probe,
        "additional_certified_by_positive_probe": total_positive_probe,
        "positive_probe_outcomes_retaining_multiple_hypotheses": total_ambiguous_probe,
        "active_terminal_rmse_mm": active_rmse,
        "active_gain_vs_fallback_pct": 100.0 * (1.0 - active_rmse / fallback_rmse),
        "active_gain_vs_passive_certificate_pct": 100.0
        * (1.0 - active_rmse / passive_rmse),
        "mean_active_probe_frames": float(
            pooled["active_minimum_cost"]["mean_probe_frames"]
        ),
        "active_harmful_decisions_vs_fallback": active_harm,
        "max_entropy_harmful_decisions_vs_fallback": entropy_harm,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
