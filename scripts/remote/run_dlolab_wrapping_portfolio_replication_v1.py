#!/usr/bin/env python3
"""Run the fresh Wrapping component of portfolio replication v1."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from bayesian_phystwin.query_portfolio_replication_v1 import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SENSOR_SEEDS,
    WORLD_COUNT,
    WORLD_SEEDS,
)
from bayesian_phystwin_experiments import dlolab_wrapping_certified_guard_v9 as method

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "scripts/remote/run_dlolab_wrapping_certified_guard_v9.py"
OUTPUT = Path(
    "/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1/wrapping"
)
ATTEMPT = Path(
    "/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1/"
    "wrapping.attempt.json"
)


def _configure_method() -> None:
    method.WORLD_COUNT = WORLD_COUNT
    method.PREFIX_BATCH_COUNT = (WORLD_COUNT + 8) // 9
    method.WORLD_SEED = WORLD_SEEDS["dlolab_wrapping_v9"]
    method.SENSOR_SEED = SENSOR_SEEDS["dlolab_wrapping_v9"]
    method.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    method.BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES


def _load_runner() -> Any:
    _configure_method()
    spec = importlib.util.spec_from_file_location("wrapping_portfolio_v1_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Wrapping runner")
    runner: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.WORLD_COUNT = WORLD_COUNT
    runner.PREFIX_BATCH_COUNT = (WORLD_COUNT + 8) // 9
    runner.OUTPUT = OUTPUT
    runner.ATTEMPT = ATTEMPT
    runner.NEW_SOURCES = (
        *runner.NEW_SOURCES,
        "src/bayesian_phystwin/query_portfolio_replication_v1.py",
        "scripts/remote/run_dlolab_wrapping_portfolio_replication_v1.py",
        "configs/experiments/query_portfolio_replication_v1.json",
    )
    runner.__file__ = str(Path(__file__).resolve())
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker-kind", choices=("prefix", "future"))
    parser.add_argument("--worker-index", type=int)
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT:
        raise ValueError("registered Wrapping output root required")
    runner = _load_runner()
    worker = args.worker_kind is not None or args.worker_index is not None
    if worker:
        if args.worker_kind is None or args.worker_index is None:
            raise ValueError("complete worker specification required")
        runner._worker(args.output, args.worker_kind, args.worker_index)
    else:
        runner._run(args.output)


if __name__ == "__main__":
    main()
