#!/usr/bin/env python3
"""Run the frozen anytime-valid simulator-admission experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bayesian_phystwin_experiments.anytime_recursive_admission_v1 import (
    canonical_result_digest,
    load_anytime_recursive_protocol,
    run_anytime_recursive_admission_v1,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    source = path.resolve(strict=True)
    return {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report(result: Mapping[str, object]) -> str:
    stream = cast(Mapping[str, Any], result["fresh_stream"])
    gain = cast(Mapping[str, Any], result["gain_null_calibration"])
    harm = cast(Mapping[str, Any], result["harm_null_calibration"])
    decision = cast(Mapping[str, Any], result["decision"])
    first = stream["first_authorized_issue_index"]
    first_text = "never" if first is None else str(first)
    lines = [
        "# Anytime-valid simulator admission v1",
        "",
        "## Fresh delayed-outcome stream",
        "",
        "| Quantity | Result |",
        "|---|---:|",
        f"| Independent seed-domains | {stream['seed_count']} |",
        f"| Physical fallback mean RMSE | {1000.0 * stream['fallback_mean_loss_m']:.4f} mm |",
        f"| Shadow correction mean RMSE | {1000.0 * stream['candidate_mean_loss_m']:.4f} mm |",
        f"| Anytime-selected mean RMSE | {1000.0 * stream['selected_mean_loss_m']:.4f} mm |",
        f"| Shadow correction gain | {100.0 * stream['candidate_relative_improvement_over_fallback']:.2f}% |",
        f"| Anytime-selected gain | {100.0 * stream['selected_relative_improvement_over_fallback']:.2f}% |",
        f"| Candidate wins/ties/losses | {stream['candidate_wins']}/{stream['candidate_ties']}/{stream['candidate_loss_count']} |",
        f"| First authorized issue index | {first_text} |",
        f"| Candidate deployments | {stream['authorized_deployment_count']} |",
        f"| Exact fallback deployments | {stream['fallback_deployment_count']} |",
        f"| Selected harmful episodes | {stream['selected_harmful_episode_count']} |",
        f"| Exact fallback identity violations | {stream['exact_fallback_violation_count']} |",
        "",
        "The current e-values are evaluated only from outcomes that matured before",
        "the next deployment decision. Candidate and fallback forecasts are registered",
        "before each fresh seed-domain is generated.",
        "",
        "## Optional-stopping implementation calibration",
        "",
        "| Null | False promotions | Fraction | Total alpha | 95% Wilson interval |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Mean gain at boundary | {gain['false_promotion_count']}/{gain['world_count']} | "
            f"{100.0 * gain['false_promotion_fraction']:.3f}% | "
            f"{100.0 * gain['total_alpha']:.2f}% | "
            f"[{100.0 * gain['wilson_95_interval'][0]:.3f}%, "
            f"{100.0 * gain['wilson_95_interval'][1]:.3f}%] |"
        ),
        (
            f"| Harm rate at ceiling | {harm['false_promotion_count']}/{harm['world_count']} | "
            f"{100.0 * harm['false_promotion_fraction']:.3f}% | "
            f"{100.0 * harm['total_alpha']:.2f}% | "
            f"[{100.0 * harm['wilson_95_interval'][0]:.3f}%, "
            f"{100.0 * harm['wilson_95_interval'][1]:.3f}%] |"
        ),
        "",
        "Each null world is monitored after every outcome across four externally",
        "restarted epochs. The epoch thresholds use a geometric allocation whose",
        "infinite sum equals the registered total alpha.",
        "",
        "## Decision",
        "",
    ]
    for key, value in decision.items():
        lines.append(f"- `{key}`: **{value}**")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(result["claim_boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    repository = Path(__file__).resolve().parents[2]
    core_path = repository / "src/bayesian_phystwin/anytime_admission_v1.py"
    experiment_path = (
        repository
        / "src/bayesian_phystwin_experiments/anytime_recursive_admission_v1.py"
    )
    benchmark_path = (
        repository
        / "src/bayesian_phystwin_experiments/recursive_corruption_benchmark_v2.py"
    )
    config = load_anytime_recursive_protocol(protocol_path)
    method_seal = {
        "schema": "bayesian-phystwin.anytime-admission-method-seal-v1",
        "protocol": _identity(protocol_path),
        "core": _identity(core_path),
        "experiment": _identity(experiment_path),
        "base_benchmark": _identity(benchmark_path),
        "source_revision": os.environ.get("GITHUB_SHA"),
        "fresh_seed_start": config.seed_start,
        "fresh_seed_count": config.seed_count,
        "delay_seed": config.delay_seed,
        "null_seed": config.null_seed,
        "fresh_seed_outcomes_opened": False,
        "target_dependent_retuning": False,
        "paper_claim_authorized": False,
    }
    method_seal_path = output / "method-seal.json"
    _write_json(method_seal_path, method_seal)

    result = run_anytime_recursive_admission_v1(config)
    result["method_seal"] = _identity(method_seal_path)
    result["canonical_result_sha256_before_identity"] = canonical_result_digest(result)
    result_path = output / "result.json"
    _write_json(result_path, result)
    _write_rows(
        output / "stream.csv",
        cast(Sequence[Mapping[str, object]], result["records"]),
    )
    _write_rows(
        output / "resolutions.csv",
        cast(Sequence[Mapping[str, object]], result["resolution_records"]),
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    (output / "environment.txt").write_text(
        "\n".join(
            (
                f"python={sys.version.replace(chr(10), ' ')}",
                f"platform={platform.platform()}",
                f"numpy={np.__version__}",
                f"github_sha={os.environ.get('GITHUB_SHA', '')}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    print((output / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
