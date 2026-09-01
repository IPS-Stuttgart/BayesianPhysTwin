#!/usr/bin/env python3
"""Run the fresh Slingshot component of portfolio replication v1."""

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
from bayesian_phystwin_experiments import (
    dlolab_slingshot_policy_certificate_source_v3 as method_v3,
)
from bayesian_phystwin_experiments import (
    dlolab_slingshot_policy_certificate_source_v4 as method_v4,
)

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "scripts/remote/run_dlolab_slingshot_policy_certificate_source_v4.py"
OUTPUT = Path(
    "/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1/slingshot"
)
ATTEMPT = Path(
    "/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1/"
    "slingshot.attempt.json"
)


def _configure_methods() -> None:
    method_v4.COUNTS["calibration"] = 128
    method_v4.COUNTS["evaluation"] = WORLD_COUNT
    method_v4.WORLD_SEEDS["calibration"] = WORLD_SEEDS["dlolab_slingshot_v4"] - 1
    method_v4.WORLD_SEEDS["evaluation"] = WORLD_SEEDS["dlolab_slingshot_v4"]
    method_v4.SENSOR_SEEDS["calibration"] = SENSOR_SEEDS["dlolab_slingshot_v4"] - 1
    method_v4.SENSOR_SEEDS["evaluation"] = SENSOR_SEEDS["dlolab_slingshot_v4"]
    method_v3.BOOTSTRAP_SEED = BOOTSTRAP_SEED + 1
    method_v3.BOOTSTRAP_REPLICATES = BOOTSTRAP_REPLICATES


def _load_runner() -> Any:
    _configure_methods()
    spec = importlib.util.spec_from_file_location("slingshot_portfolio_v1_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen Slingshot runner")
    wrapper: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper.OUTPUT_ROOT = OUTPUT
    wrapper.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.OUTPUT_ROOT = OUTPUT
    wrapper.runner.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.WORKER_RUNNER_PATH = Path(__file__).resolve()
    extra = (
        "src/bayesian_phystwin/query_portfolio_replication_v1.py",
        "scripts/remote/run_dlolab_slingshot_portfolio_replication_v1.py",
        "configs/experiments/query_portfolio_replication_v1.json",
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
        raise ValueError("registered Slingshot output root required")
    wrapper = _load_runner()
    worker_values = (args.worker_role, args.worker_kind, args.worker_index)
    if args.verify_only:
        if any(value is not None for value in (*worker_values, args.worker_action)):
            raise ValueError("verification cannot be a worker")
        wrapper.runner.verify_result(args.output)
    elif all(value is not None for value in worker_values):
        wrapper.runner.worker(
            args.output,
            args.worker_role,
            args.worker_kind,
            args.worker_index,
            args.worker_action,
        )
    elif any(value is not None for value in (*worker_values, args.worker_action)):
        raise ValueError("complete worker specification required")
    else:
        wrapper.runner.run(args.output)


if __name__ == "__main__":
    main()
