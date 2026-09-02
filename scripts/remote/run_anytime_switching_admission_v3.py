#!/usr/bin/env python3
"""Run the frozen switching-null anytime-admission stress study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin_experiments.anytime_switching_admission_v3 import (
    run_switching_admission_study,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_summary_csv(path: Path, result: dict[str, object]) -> None:
    scenarios = cast(dict[str, object], result["scenarios"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario",
                "group",
                "registered_null",
                "method",
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
        for name, raw_scenario in scenarios.items():
            scenario = cast(dict[str, object], raw_scenario)
            for method in (
                "latched_shared_alpha_iut",
                "switching_union_min_score",
            ):
                summary = cast(dict[str, object], scenario[method])
                interval = cast(list[float], summary["wilson_95_interval"])
                quantiles = cast(
                    list[float] | None,
                    summary["first_crossing_quantiles_10_90"],
                )
                writer.writerow(
                    {
                        "scenario": name,
                        "group": scenario["group"],
                        "registered_null": scenario["registered_null"],
                        "method": method,
                        "replication_count": summary["replication_count"],
                        "crossing_count": summary["crossing_count"],
                        "crossing_probability": summary[
                            "crossing_probability"
                        ],
                        "wilson_95_lower": interval[0],
                        "wilson_95_upper": interval[1],
                        "median_first_crossing": summary[
                            "median_first_crossing"
                        ],
                        "first_crossing_q10": (
                            None if quantiles is None else quantiles[0]
                        ),
                        "first_crossing_q90": (
                            None if quantiles is None else quantiles[1]
                        ),
                    }
                )


def _write_phase_csv(path: Path, result: dict[str, object]) -> None:
    scenarios = cast(dict[str, object], result["scenarios"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "scenario",
                "phase_index",
                "phase_name",
                "duration",
                "active_null_component",
                "expected_gain_score",
                "expected_harm_rate",
                "expected_robust_score",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for name, raw_scenario in scenarios.items():
            scenario = cast(dict[str, object], raw_scenario)
            phases = cast(list[dict[str, object]], scenario["phase_expectations"])
            for phase in phases:
                writer.writerow(
                    {
                        "scenario": name,
                        "phase_index": phase["phase_index"],
                        "phase_name": phase["name"],
                        "duration": phase["duration"],
                        "active_null_component": phase[
                            "active_null_component"
                        ],
                        "expected_gain_score": phase["expected_gain_score"],
                        "expected_harm_rate": phase["expected_harm_rate"],
                        "expected_robust_score": phase[
                            "expected_robust_score"
                        ],
                    }
                )


def _write_report(path: Path, result: dict[str, object]) -> None:
    design = cast(dict[str, object], result["design"])
    scenarios = cast(dict[str, object], result["scenarios"])
    derived = cast(dict[str, object], result["derived_comparison"])
    gate = cast(dict[str, object], result["mechanism_gate"])
    lines = [
        "# Switching-null anytime admission v3",
        "",
        f"- Decision: **{result['decision']}**",
        f"- Mechanism gate: **{gate['passed']}**",
        f"- First-epoch alpha: **{design['first_epoch_alpha']}**",
        f"- E-value threshold: **{design['e_value_threshold']}**",
        "",
        "## Controlled scenarios",
        "",
        "| Scenario | Method | Crossing probability | Wilson 95% | Median crossing |",
        "|---|---|---:|---:|---:|",
    ]
    for name, raw_scenario in scenarios.items():
        scenario = cast(dict[str, object], raw_scenario)
        for method in (
            "latched_shared_alpha_iut",
            "switching_union_min_score",
        ):
            summary = cast(dict[str, object], scenario[method])
            interval = cast(list[float], summary["wilson_95_interval"])
            lines.append(
                f"| `{name}` | `{method}` | "
                f"{float(cast(Any, summary['crossing_probability'])):.4f} | "
                f"[{interval[0]:.4f}, {interval[1]:.4f}] | "
                f"{summary['median_first_crossing']} |"
            )
    lines.extend(
        [
            "",
            "## Assumption stress test",
            "",
            (
                "- Latched IUT crossing under switching invalidity: "
                f"**{float(cast(Any, derived['switching_null_latched_iut_crossing_probability'])):.4f}**."
            ),
            (
                "- Robust minimum-score crossing under the same stream: "
                f"**{float(cast(Any, derived['switching_null_robust_crossing_probability'])):.4f}**."
            ),
            (
                "- Maximum robust null Wilson upper bound: "
                f"**{float(cast(Any, derived['maximum_robust_null_wilson_upper'])):.4f}**."
            ),
            (
                "- Moderate safe-benefit robust power: "
                f"**{float(cast(Any, derived['moderate_robust_power'])):.4f}**."
            ),
            (
                "- Strong safe-benefit robust power: "
                f"**{float(cast(Any, derived['strong_robust_power'])):.4f}**."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The efficient latched intersection--union rule is valid only when "
                "one component null holds throughout an epoch. In the registered "
                "switching stream, gain evidence is accumulated while harm is "
                "excessive and harm evidence is accumulated later while mean gain "
                "is negative. Latching can therefore authorize outside its theorem "
                "boundary. The minimum-score process remains valid because its "
                "score is no larger than whichever component has nonpositive "
                "conditional expectation at each reveal."
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
    result = run_switching_admission_study(protocol)
    result["protocol_sha256"] = _sha256(protocol_path)
    result["runtime"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    _write_json(output_root / "effective-protocol.json", protocol)
    _write_json(output_root / "result.json", result)
    _write_summary_csv(output_root / "comparison-summary.csv", result)
    _write_phase_csv(output_root / "phase-expectations.csv", result)
    _write_report(output_root / "report.md", result)
    (output_root / "environment.txt").write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    retained = sorted(
        item
        for item in output_root.iterdir()
        if item.is_file() and item.name != "SHA256SUMS"
    )
    (output_root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(item)}  {item.name}\n" for item in retained),
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
