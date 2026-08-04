#!/usr/bin/env python3
"""Finalize a Deform360 source-hull probe as a complete-cohort decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_probe_sha256", None)
    payload.pop("probe_sha256", None)
    payload.pop("repository_revision", None)
    payload.pop("dataset_root", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _probe_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("probe_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def finalize_source_hull_feasibility(value: Mapping[str, Any]) -> dict[str, Any]:
    """Add a strict complete-cohort scoring decision and refresh result IDs."""

    result = dict(value)
    _require(
        result.get("schema") == "bayesian-phystwin/deform360-source-hull-probe-v1",
        "unexpected source-hull probe schema",
    )
    _require(
        result.get("amendment_id") == "deform360-source-hull-contract-probe-v2",
        "the target-blind empty-frame amendment was not applied",
    )
    boundary = result.get("information_boundary")
    _require(isinstance(boundary, dict), "probe information boundary is missing")
    for field in (
        "points_world_m_coordinate_values_decoded",
        "model_prediction_run",
        "score_bearing_outcome_computed",
        "reserved_target_outcomes_opened",
    ):
        _require(boundary.get(field) is False, f"probe boundary violation: {field}")

    episode_count = result.get("episode_count")
    eligible_count = result.get("prediction_eligible_episode_count")
    ineligible_count = result.get("prediction_ineligible_episode_count")
    _require(
        isinstance(episode_count, int)
        and not isinstance(episode_count, bool)
        and episode_count > 0,
        "episode_count must be a positive integer",
    )
    _require(
        isinstance(eligible_count, int)
        and not isinstance(eligible_count, bool)
        and eligible_count >= 0,
        "eligible count must be a nonnegative integer",
    )
    _require(
        isinstance(ineligible_count, int)
        and not isinstance(ineligible_count, bool)
        and ineligible_count >= 0,
        "ineligible count must be a nonnegative integer",
    )
    _require(
        eligible_count + ineligible_count == episode_count,
        "prediction eligibility accounting does not cover the locked cohort",
    )

    archives = result.get("archives")
    _require(
        isinstance(archives, list) and len(archives) == episode_count,
        "archive records do not cover the locked cohort",
    )
    actual_ineligible = sorted(
        str(record["relative_path"])
        for record in archives
        if isinstance(record, dict) and record.get("prediction_eligible") is False
    )
    declared_ineligible = result.get("prediction_ineligible_archives")
    _require(
        isinstance(declared_ineligible, list)
        and sorted(str(path) for path in declared_ineligible) == actual_ineligible,
        "prediction-ineligible archive accounting changed",
    )
    _require(
        len(actual_ineligible) == ineligible_count,
        "prediction-ineligible archive count changed",
    )

    scoring_feasible = ineligible_count == 0
    result["all_locked_archives_prediction_eligible"] = scoring_feasible
    result["partial_source_support_available"] = eligible_count > 0
    result["scoring_feasible"] = scoring_feasible
    result["scoring_decision"] = (
        "complete-cohort-source-scoring-supported"
        if scoring_feasible
        else "complete-cohort-source-scoring-infeasible"
    )
    result["scoring_infeasibility_reasons"] = (
        []
        if scoring_feasible
        else [
            {
                "archives": actual_ineligible,
                "code": "prediction-ineligible-locked-archives",
                "count": ineligible_count,
                "criterion": "at least three usable sampled hulls per archive",
            }
        ]
    )
    result["content_probe_sha256"] = _content_sha256(result)
    result["probe_sha256"] = _probe_sha256(result)
    return result


def write_result(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "probe result must be a JSON object")
    result = finalize_source_hull_feasibility(value)
    write_result(args.output, result)
    print(
        json.dumps(
            {
                "content_probe_sha256": result["content_probe_sha256"],
                "probe_sha256": result["probe_sha256"],
                "scoring_decision": result["scoring_decision"],
                "scoring_feasible": result["scoring_feasible"],
                "prediction_eligible_episode_count": result[
                    "prediction_eligible_episode_count"
                ],
                "prediction_ineligible_episode_count": result[
                    "prediction_ineligible_episode_count"
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
