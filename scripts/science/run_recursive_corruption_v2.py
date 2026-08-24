#!/usr/bin/env python3
"""Run and retain the registered recursive-corruption v2 benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from analyze_recursive_corruption_v2 import analyze

from bayesian_phystwin.recursive_corruption_benchmark_v2 import (
    CONDITIONS,
    RecursiveCorruptionV2Config,
    run_recursive_corruption_benchmark_v2,
    sha256_file,
    write_deterministic_trace_npz,
    write_json,
    write_records_csv,
)


def _parse_seeds(value: str) -> tuple[int, ...]:
    if ":" in value:
        parts = value.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("seed ranges must use START:STOP")
        try:
            start, stop = (int(part) for part in parts)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "seed ranges must contain integers"
            ) from error
        if start < 0 or stop <= start:
            raise argparse.ArgumentTypeError("seed ranges require 0 <= START < STOP")
        return tuple(range(start, stop))
    try:
        seeds = tuple(int(part) for part in value.split(",") if part)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds or any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be unique nonnegative integers")
    return seeds


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=_parse_seeds("100000:100200"),
        help="Fresh evidence roster; the registered default is 100000:100200.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = RecursiveCorruptionV2Config()
    result, traces = run_recursive_corruption_benchmark_v2(
        seeds=args.seeds,
        conditions=CONDITIONS,
        config=config,
        retain_traces=True,
    )
    if traces is None:
        raise RuntimeError("registered run must retain traces")

    result_path = output_dir / "result.json"
    records_path = output_dir / "records.csv"
    trace_path = output_dir / "traces.npz"
    write_json(result, result_path)
    write_records_csv(result, records_path)
    write_deterministic_trace_npz(
        arrays=traces,
        result=result,
        path=trace_path,
    )
    analysis = analyze(
        result_path=result_path,
        trace_path=trace_path,
        output_dir=output_dir,
    )

    run_manifest = {
        "schema": "bayesian-phystwin.recursive-corruption-run-manifest-v2",
        "schema_version": 2,
        "source": {
            "repository": os.environ.get(
                "GITHUB_REPOSITORY",
                "IPS-Stuttgart/BayesianPhysTwin",
            ),
            "commit": os.environ.get("GITHUB_SHA"),
        },
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "invocation": {
            "seeds": list(args.seeds),
            "conditions": list(CONDITIONS),
            "output_dir": str(output_dir),
        },
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "run-manifest.json"
        },
        "all_coequal_criteria_passed": analysis["coequal_review"][
            "all_criteria_passed"
        ],
        "access_boundary": result["access_boundary"],
        "scientific_boundary": result["scientific_boundary"],
    }
    write_json(run_manifest, output_dir / "run-manifest.json")
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "seed_count": len(args.seeds),
                "record_count": len(result["records"]),
                "all_coequal_criteria_passed": analysis["coequal_review"][
                    "all_criteria_passed"
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
