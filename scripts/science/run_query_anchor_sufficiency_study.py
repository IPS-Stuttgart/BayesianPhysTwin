#!/usr/bin/env python3
"""Run a controlled query-anchor support--precision planning study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.nuisance_aware_information import (
    NuisanceAwareInformationState,
    greedy_nuisance_aware_selection,
)
from bayesian_phystwin.query_anchor_sufficiency import (
    evaluate_query_anchor_sufficiency,
)
from bayesian_phystwin.query_aware_anchor_planning import (
    greedy_query_aware_selection,
)

_SCHEMA = "bayesian-phystwin-query-anchor-sufficiency-controlled-v1"


def _controlled_protocol() -> dict[str, Any]:
    return {
        "schema": _SCHEMA,
        "semantics": (
            "controlled source-side planning study; covariance is divided by the "
            "declared precision multiplier"
        ),
        "claim_boundary": (
            "The study tests planner mechanics only. It does not establish real "
            "sensor competence, physical-query improvement, calibrated coverage, "
            "deployment safety, Causal4D benefit, or state of the art."
        ),
        "state_precision": np.eye(3, dtype=np.float64).tolist(),
        "nuisance_precision": np.diag([1e-3, 1e-3]).tolist(),
        "query_jacobian": [[1.0, 0.0, 0.0]],
        "precision_multipliers": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "maximum_count": 4,
        "minimum_trace_reduction": 0.0,
        "target_remaining_variance_fraction": 0.25,
        "candidates": [
            {
                "id": "shared-visual-a",
                "state_jacobian": [[4.0, 0.0, 0.0]],
                "nuisance_jacobian": [[4.0, 0.0]],
                "observation_covariance": [[1.0]],
                "reliability": 1.0,
                "cost": 1.0,
                "dependence_group": "shared-visual-capture",
            },
            {
                "id": "shared-visual-b",
                "state_jacobian": [[3.0, 0.0, 0.0]],
                "nuisance_jacobian": [[3.0, 0.0]],
                "observation_covariance": [[1.0]],
                "reliability": 1.0,
                "cost": 1.0,
                "dependence_group": "shared-visual-capture",
            },
            {
                "id": "independent-metric-x-strong",
                "state_jacobian": [[2.0, 0.0, 0.0]],
                "nuisance_jacobian": [[0.0, 0.0]],
                "observation_covariance": [[1.0]],
                "reliability": 1.0,
                "cost": 2.0,
                "dependence_group": None,
            },
            {
                "id": "query-irrelevant-y",
                "state_jacobian": [[0.0, 8.0, 0.0]],
                "nuisance_jacobian": [[0.0, 0.0]],
                "observation_covariance": [[1.0]],
                "reliability": 1.0,
                "cost": 1.0,
                "dependence_group": None,
            },
            {
                "id": "clock-confounded-x",
                "state_jacobian": [[3.0, 0.0, 0.0]],
                "nuisance_jacobian": [[0.0, 3.0]],
                "observation_covariance": [[1.0]],
                "reliability": 1.0,
                "cost": 1.0,
                "dependence_group": None,
            },
            {
                "id": "independent-metric-x-efficient",
                "state_jacobian": [[1.5, 0.0, 0.0]],
                "nuisance_jacobian": [[0.0, 0.0]],
                "observation_covariance": [[1.0]],
                "reliability": 1.0,
                "cost": 1.5,
                "dependence_group": None,
            },
            {
                "id": "independent-mixed-xz",
                "state_jacobian": [[1.0, 0.0, 1.0]],
                "nuisance_jacobian": [[0.0, 0.0]],
                "observation_covariance": [[1.0]],
                "reliability": 1.0,
                "cost": 1.0,
                "dependence_group": None,
            },
        ],
    }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _study_inputs(
    protocol: dict[str, Any],
) -> tuple[
    NuisanceAwareInformationState,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[float],
    list[str | None],
    list[str],
]:
    candidates = protocol["candidates"]
    prior = NuisanceAwareInformationState.from_independent_priors(
        np.asarray(protocol["state_precision"], dtype=np.float64),
        np.asarray(protocol["nuisance_precision"], dtype=np.float64),
    )
    return (
        prior,
        np.asarray(protocol["query_jacobian"], dtype=np.float64),
        [
            np.asarray(candidate["state_jacobian"], dtype=np.float64)
            for candidate in candidates
        ],
        [
            np.asarray(candidate["nuisance_jacobian"], dtype=np.float64)
            for candidate in candidates
        ],
        [
            np.asarray(candidate["observation_covariance"], dtype=np.float64)
            for candidate in candidates
        ],
        [float(candidate["reliability"]) for candidate in candidates],
        [float(candidate["cost"]) for candidate in candidates],
        [candidate["dependence_group"] for candidate in candidates],
        [str(candidate["id"]) for candidate in candidates],
    )


def build_controlled_study() -> tuple[dict[str, Any], str, str]:
    """Build JSON, CSV, and Markdown results without writing files."""

    protocol = _controlled_protocol()
    protocol_sha256 = hashlib.sha256(_canonical_json_bytes(protocol)).hexdigest()
    (
        prior,
        query,
        state_jacobians,
        nuisance_jacobians,
        covariances,
        reliabilities,
        costs,
        dependence_groups,
        candidate_ids,
    ) = _study_inputs(protocol)

    curve = evaluate_query_anchor_sufficiency(
        prior,
        query,
        state_jacobians,
        nuisance_jacobians,
        covariances,
        precision_multipliers=protocol["precision_multipliers"],
        reliabilities=reliabilities,
        costs=costs,
        dependence_groups=dependence_groups,
        maximum_count=int(protocol["maximum_count"]),
        minimum_trace_reduction=float(protocol["minimum_trace_reduction"]),
        target_remaining_variance_fraction=float(
            protocol["target_remaining_variance_fraction"]
        ),
    )
    query_plan = greedy_query_aware_selection(
        prior,
        query,
        state_jacobians,
        nuisance_jacobians,
        covariances,
        reliabilities=reliabilities,
        costs=costs,
        dependence_groups=dependence_groups,
        count=int(protocol["maximum_count"]),
    )
    full_state_plan = greedy_nuisance_aware_selection(
        prior,
        state_jacobians,
        nuisance_jacobians,
        covariances,
        reliabilities=reliabilities,
        count=int(protocol["maximum_count"]),
    )

    query_ids = [candidate_ids[int(index)] for index in query_plan.selected_indices]
    full_state_ids = [
        candidate_ids[int(index)] for index in full_state_plan.selected_indices
    ]
    final_fractions = curve.remaining_variance_fractions[:, -1]
    directional_checks = {
        "query_aware_avoids_query_irrelevant_first": bool(
            query_ids and query_ids[0] != "query-irrelevant-y"
        ),
        "full_state_information_prefers_query_irrelevant_first": bool(
            full_state_ids and full_state_ids[0] == "query-irrelevant-y"
        ),
        "maximum_support_variance_is_nonincreasing_with_precision": bool(
            np.all(np.diff(final_fractions) <= 1e-12)
        ),
    }

    records = []
    for record in curve.records():
        selected_ids = [
            candidate_ids[int(index)] for index in record["selected_indices"]
        ]
        records.append({**record, "selected_candidate_ids": selected_ids})

    result: dict[str, Any] = {
        "schema": _SCHEMA,
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
        "query_aware_selected_candidate_ids_at_unit_precision": query_ids,
        "full_state_information_selected_candidate_ids_at_unit_precision": (
            full_state_ids
        ),
        "first_sufficient_support_by_precision": [
            {
                "precision_multiplier": float(multiplier),
                "first_sufficient_support": int(support),
            }
            for multiplier, support in zip(
                curve.precision_multipliers,
                curve.first_sufficient_support,
                strict=True,
            )
        ],
        "directional_checks": directional_checks,
        "all_directional_checks_passed": bool(all(directional_checks.values())),
        "records": records,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["result_sha256"] = hashlib.sha256(
        _canonical_json_bytes(result)
    ).hexdigest()

    csv_buffer = io.StringIO()
    fieldnames = [
        "precision_multiplier",
        "support_count",
        "selected_count",
        "selected_candidate_ids",
        "query_variance_trace",
        "remaining_variance_fraction",
        "cumulative_cost",
        "target_met",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                **{
                    key: record[key]
                    for key in fieldnames
                    if key != "selected_candidate_ids"
                },
                "selected_candidate_ids": ";".join(record["selected_candidate_ids"]),
            }
        )

    report_lines = [
        "# Controlled query-anchor sufficiency study",
        "",
        f"Protocol SHA-256: `{protocol_sha256}`",
        f"Result SHA-256: `{result['result_sha256']}`",
        "",
        (
            "| Precision multiplier | First sufficient support | "
            "Final variance fraction | Selected order |"
        ),
        "|---:|---:|---:|---|",
    ]
    for precision_index, multiplier in enumerate(curve.precision_multipliers):
        selected = curve.selected_prefix(
            precision_index,
            int(curve.selected_counts[precision_index]),
        )
        selected_ids = [candidate_ids[int(index)] for index in selected]
        report_lines.append(
            "| "
            f"{float(multiplier):.2f} | "
            f"{int(curve.first_sufficient_support[precision_index])} | "
            f"{float(final_fractions[precision_index]):.6f} | "
            f"{', '.join(selected_ids) or 'none'} |"
        )
    report_lines.extend(
        [
            "",
            "## Directional controls",
            "",
            *[
                f"- `{name}`: **{'pass' if passed else 'fail'}**"
                for name, passed in directional_checks.items()
            ],
            "",
            "## Claim boundary",
            "",
            str(protocol["claim_boundary"]),
            "",
        ]
    )
    return result, csv_buffer.getvalue(), "\n".join(report_lines)


def _atomic_write_text(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for summary.json, curve.csv, and report.md",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing study outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_paths = (
        args.output_dir / "summary.json",
        args.output_dir / "curve.csv",
        args.output_dir / "report.md",
    )
    if not args.force:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "refusing to replace existing output: "
                + ", ".join(str(path) for path in existing)
            )
    result, curve_csv, report = build_controlled_study()
    _atomic_write_text(
        output_paths[0],
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        force=args.force,
    )
    _atomic_write_text(
        output_paths[1],
        curve_csv,
        force=args.force,
    )
    _atomic_write_text(
        output_paths[2],
        report,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
