#!/usr/bin/env python3
"""Render the frozen DLO4/DLO5 source-replication meta-analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin_experiments.deform_dlo45_source_meta_v1 import (
    DLOS,
    evaluate_source_meta_analysis,
    load_source_meta_protocol,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dlo4-source-result", type=Path, required=True)
    parser.add_argument("--dlo5-source-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _identity(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    return {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "size_bytes": source.stat().st_size,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_cases(path: Path, result: Mapping[str, object]) -> None:
    cases = cast(Sequence[Mapping[str, object]], result["cases"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "dlo",
                "name",
                "baseline_l1_m",
                "candidate_l1_m",
                "absolute_improvement_m",
                "candidate_to_baseline_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(cases)


def _write_report(path: Path, result: Mapping[str, object]) -> None:
    pooled = _mapping(result["pooled_equal_trajectory"], label="pooled result")
    bootstrap = _mapping(
        result["dlo_stratified_bootstrap"], label="stratified bootstrap"
    )
    relative_interval = cast(
        Sequence[float], bootstrap["relative_improvement_95_interval"]
    )
    absolute_interval = cast(
        Sequence[float], bootstrap["absolute_improvement_95_interval_m"]
    )
    lines = [
        "# DLO4/DLO5 joint source-replication meta-analysis",
        "",
        f"- Decision: **{result['decision']}**",
        "- Statistical unit: complete source-test trajectory",
        "- Aggregation frozen after source opening and before target outcome",
        "",
        "## Joint point result",
        "",
        (
            "- Pooled physical/candidate mean L1: "
            f"**{1000.0 * float(cast(Any, pooled['baseline_mean_l1_m'])):.4f} / "
            f"{1000.0 * float(cast(Any, pooled['candidate_mean_l1_m'])):.4f} mm**"
        ),
        (
            "- Relative improvement: "
            f"**{100.0 * float(cast(Any, pooled['relative_improvement'])):.2f}%**"
        ),
        (
            "- Wins/ties/losses: "
            f"**{pooled['wins']}/{pooled['ties']}/{pooled['losses']}**"
        ),
        (
            "- Exact upper sign probability under a fair-direction null: "
            f"**{float(cast(Any, pooled['exact_upper_sign_probability'])):.8g}**"
        ),
        (
            "- DLO-stratified 95% bootstrap interval, relative improvement: "
            f"**[{100.0 * relative_interval[0]:.2f}%, "
            f"{100.0 * relative_interval[1]:.2f}%]**"
        ),
        (
            "- DLO-stratified 95% bootstrap interval, absolute improvement: "
            f"**[{1000.0 * absolute_interval[0]:.4f}, "
            f"{1000.0 * absolute_interval[1]:.4f}] mm**"
        ),
        "",
        "## Per-DLO source gates",
        "",
        "| DLO | Physical (mm) | Candidate (mm) | Improvement | W/T/L |",
        "|---|---:|---:|---:|---:|",
    ]
    per_dlo = _mapping(result["per_dlo"], label="per-DLO results")
    for dlo in DLOS:
        row = _mapping(per_dlo[dlo], label=f"{dlo} result")
        lines.append(
            "| {dlo} | {baseline:.4f} | {candidate:.4f} | {gain:.2f}% | "
            "{wins}/{ties}/{losses} |".format(
                dlo=dlo,
                baseline=1000.0
                * float(cast(Any, row["baseline_mean_l1_m"])),
                candidate=1000.0
                * float(cast(Any, row["candidate_mean_l1_m"])),
                gain=100.0 * float(cast(Any, row["relative_improvement"])),
                wins=row["wins"],
                ties=row["ties"],
                losses=row["losses"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_source_meta_protocol(protocol_path)
    result_paths = {
        "DLO4": args.dlo4_source_result.resolve(strict=True),
        "DLO5": args.dlo5_source_result.resolve(strict=True),
    }
    identities = {dlo: _identity(path) for dlo, path in result_paths.items()}
    expected = _mapping(protocol["source_results"], label="source results")
    for dlo in DLOS:
        registered = _mapping(expected[dlo], label=f"{dlo} registered identity")
        if (
            identities[dlo]["sha256"] != registered["sha256"]
            or identities[dlo]["size_bytes"] != registered["size_bytes"]
        ):
            raise ValueError(f"{dlo} source result does not match the frozen identity")

    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    method_seal = {
        "schema_version": 1,
        "contract": "deform-dlo45-source-meta-analysis-method-seal-v1",
        "protocol": _identity(protocol_path),
        "source_results": identities,
        "target_scores_used": False,
        "paper_claim_authorized": False,
    }
    method_seal_path = output_root / "method_seal.json"
    _write_json(method_seal_path, method_seal)

    source_results = {dlo: _read_json(path) for dlo, path in result_paths.items()}
    result = evaluate_source_meta_analysis(
        protocol=protocol,
        source_results=source_results,
    )
    result.update(
        {
            "protocol": _identity(protocol_path),
            "method_seal": _identity(method_seal_path),
            "source_results": identities,
            "source_payloads_loaded_after_method_seal": True,
        }
    )
    _write_json(output_root / "result.json", result)
    _write_cases(output_root / "trajectory-results.csv", result)
    _write_report(output_root / "report.md", result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "pooled_equal_trajectory": result["pooled_equal_trajectory"],
                "dlo_stratified_bootstrap": result["dlo_stratified_bootstrap"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
