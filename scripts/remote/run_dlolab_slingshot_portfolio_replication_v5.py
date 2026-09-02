#!/usr/bin/env python3
"""Run the fresh Slingshot v5 transitive-count recovery component."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from bayesian_phystwin.query_portfolio_replication_v5 import (
    SLINGSHOT_SENSOR_SEEDS,
    SLINGSHOT_WORLD_SEEDS,
)

ROOT = Path(__file__).resolve().parents[2]
V4_PATH = ROOT / "scripts/remote/run_dlolab_slingshot_portfolio_replication_v4.py"
BASE = Path("/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1")
OUTPUT = BASE / "slingshot-v5"
ATTEMPT = BASE / "slingshot-v5.attempt.json"


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("slingshot_portfolio_v5_base", V4_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registered Slingshot v4 adapter")
    v4: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v4)
    wrapper = v4._load_runner()
    wrapper.method.WORLD_SEEDS.update(SLINGSHOT_WORLD_SEEDS)
    wrapper.method.SENSOR_SEEDS.update(SLINGSHOT_SENSOR_SEEDS)
    wrapper.OUTPUT_ROOT = OUTPUT
    wrapper.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.OUTPUT_ROOT = OUTPUT
    wrapper.runner.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.WORKER_RUNNER_PATH = Path(__file__).resolve()
    extra = (
        "src/bayesian_phystwin/query_portfolio_replication_v5.py",
        "scripts/remote/run_dlolab_slingshot_portfolio_replication_v5.py",
        "configs/experiments/query_portfolio_replication_v5.json",
    )
    wrapper.SOURCES = (*wrapper.SOURCES, *extra)
    wrapper.runner.SOURCES = wrapper.SOURCES
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--worker-role", choices=("calibration", "evaluation"))
    parser.add_argument("--worker-kind", choices=("prefix", "future"))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-action", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() != OUTPUT:
        raise ValueError("registered Slingshot v5 output root required")
    wrapper = _load_runner()
    values = (args.worker_role, args.worker_kind, args.worker_index)
    if args.verify_only:
        if any(value is not None for value in (*values, args.worker_action)):
            raise ValueError("verification cannot be a worker")
        wrapper.runner.verify_result(args.output)
    elif all(value is not None for value in values):
        wrapper.runner.worker(
            args.output,
            args.worker_role,
            args.worker_kind,
            args.worker_index,
            args.worker_action,
        )
    elif any(value is not None for value in (*values, args.worker_action)):
        raise ValueError("complete worker specification required")
    else:
        wrapper.runner.run(args.output)


if __name__ == "__main__":
    main()
