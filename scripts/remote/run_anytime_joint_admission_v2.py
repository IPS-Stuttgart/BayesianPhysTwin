#!/usr/bin/env python3
"""Run the frozen shared-alpha joint-admission efficiency study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin_experiments.anytime_joint_admission_v2 import (
    run_joint_admission_study,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_summary_csv(path: Path, result: dict[str, object]) -> None:
    scenarios = cast(dict[str, object], result["scenarios"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario",
                "group",
                "null_component",
                "expected_gain_score",
                "expected_harm_rate",
                "method",
                "component_alpha",
                "e_value_threshold",
                "replication_count",
                "crossing_count",
                "crossing_probability",
                "wilson_95_lower",
                "wilson_95_upper",
                "median_first_crossing",
                "first_crossing_q10",
                "first_crossing_q90",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for scenario_name, raw_scenario in scenarios.items():
            scenario = cast(dict[str, object], raw_scenario)
            for method in ("shared_alpha_iut", "bonferroni_split"):
                summary = cast(dict[str, object], scenario[method])
                interval = cast(list[float], summary["wilson_95_interval"])
                quantiles = cast(
                    list[float] | None,
                    summary["first_crossing_quantiles_10_90"],
                )
                writer.writerow(
                    {
                        "scenario": scenario_name,
                        "group": scenario["group"],
                        "null_component": scenario["null_component"],
                        "expected_gain_score": scenario["expected_gain_score"],
                        "expected_harm_rate": scenario["expected_harm_rate"],
                        "method": method,
                        "component_alpha": summary["component_alpha"],
                        "e_value_threshold": summary["e_value_threshold"],
                        "replication_count": summary["replication_count"],
                        "crossing_count": summary["crossing_count"],
                        "crossing_probability": summary["crossing_probability"],
                        "wilson_95_lower": interval[0],
                        "wilson_95_upper": interval[1],
                        "median_first_crossing": summary["median_first_crossing"],
                        "first_crossing_q10": (
                            None if quantiles is None else quantiles[0]
                        ),
                        "first_crossing_q90": (
                            None if quantiles is None else quantiles[1]
                        ),
                    }
                )


def _write_report(path: Path, result: dict[str, object]) -> None:
    design = cast(dict[str, object], result["design"])
    scenarios = cast(dict[str, object], result["scenarios"])
    derived = cast(dict[str, object], result["derived_comparison"])
    gate = cast(dict[str, object], result["mechanism_gate"])
    lines = [
        "# Shared-alpha joint anytime admission v2",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Mechanism gate: **{gate['passed']}**",
        f"- Shared first-epoch alpha: **{design['shared_first_epoch_alpha']}**",
        f"- Shared component threshold: **{design['shared_component_threshold']}**",
        f"- Split component threshold: **{design['split_component_threshold']}**",
        "",
        "## Controlled scenarios",
        "",
        "| Scenario | Method | Crossing probability | Wilson 95% | Median crossing |",
        "|---|---|---:|---:|---:|",
    ]
    for scenario_name, raw_scenario in scenarios.items():
        scenario = cast(dict[str, object], raw_scenario)
        for method in ("shared_alpha_iut", "bonferroni_split"):
            summary = cast(dict[str, object], scenario[method])
            interval = cast(list[float], summary["wilson_95_interval"])
            lines.append(
                f"| `{scenario_name}` | `{method}` | "
                f"{float(cast(Any, summary['crossing_probability'])):.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
                f"{summary['median_first_crossing']} |"
            )
    lines.extend(
        [
            "",
            "## Efficiency comparison",
            "",
            (
                "- Maximum shared-IUT null Wilson upper bound: "
                f"**{float(cast(Any, derived['maximum_null_wilson_upper'])):.4f}**."
            ),
            (
                "- Moderate-alternative power gain, shared minus split: "
                f"**{float(cast(Any, derived['moderate_power_gain_shared_minus_split'])):+.4f}**."
            ),
            (
                "- Moderate-alternative median crossing ratio, shared over split: "
                f"**{float(cast(Any, derived['moderate_median_crossing_ratio_shared_over_split'])):.4f}**."
            ),
            "",
            "## Theorem boundary",
            "",
            (
                "The invalid-candidate null is the union of insufficient mean gain "
                "and excessive harm rate. Joint admission requires both latched "
                "component e-processes to cross the same epoch-wise alpha boundary. "
                "If either fixed component null holds throughout the epoch, false "
                "admission is bounded by that alpha without a Bonferroni split."
            ),
            "",
            str(result["claim_boundary"]),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a JSON object")
    result = run_joint_admission_study(protocol)
    result["protocol_sha256"] = _sha256(protocol_path)
    result["runtime"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    _write_json(output_root / "effective-protocol.json", protocol)
    _write_json(output_root / "result.json", result)
    _write_summary_csv(output_root / "comparison-summary.csv", result)
    _write_report(output_root / "report.md", result)
    (output_root / "environment.txt").write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    retained = sorted(
        path
        for path in output_root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in retained),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "derived_comparison": result["derived_comparison"],
                "mechanism_gate": result["mechanism_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
