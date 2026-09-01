#!/usr/bin/env python3
"""Run the fresh Slingshot component of portfolio replication v1."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from typing import Any, cast

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
PARENT_ROOT = Path(
    "/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1/"
    "frozen-parent"
)
POLICY_V1_ROOT = Path(
    "/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1/"
    "frozen-policy-v1"
)
RUNTIME_ROOT = Path(
    "/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1/"
    "frozen-runtime"
)
CANONICAL_PARENT_LIBRARY_PATH = (
    "/home/fpfaff/source-only/dlo-lab-decision-v1-assets/"
    "native-libs/root/usr/lib/x86_64-linux-gnu"
)


def _configure_methods() -> None:
    method_v4.COUNTS["calibration"] = 128
    method_v4.COUNTS["evaluation"] = WORLD_COUNT
    world_seeds = WORLD_SEEDS["dlolab_slingshot_v4"]
    sensor_seeds = SENSOR_SEEDS["dlolab_slingshot_v4"]
    if isinstance(world_seeds, dict) and isinstance(sensor_seeds, dict):
        method_v4.WORLD_SEEDS.update(world_seeds)
        method_v4.SENSOR_SEEDS.update(sensor_seeds)
    elif isinstance(world_seeds, int) and isinstance(sensor_seeds, int):
        method_v4.WORLD_SEEDS["calibration"] = world_seeds - 1
        method_v4.WORLD_SEEDS["evaluation"] = world_seeds
        method_v4.SENSOR_SEEDS["calibration"] = sensor_seeds - 1
        method_v4.SENSOR_SEEDS["evaluation"] = sensor_seeds
    else:
        raise ValueError("Slingshot calibration/evaluation seeds changed")
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
    wrapper.runner.V2_RUNNER.PARENT_ROOT = PARENT_ROOT
    wrapper.runner.V2_RUNNER.POLICY_V1_ROOT = POLICY_V1_ROOT
    native_runtime = wrapper.runner.runtime

    def parent_canonical_runtime() -> dict[str, Any]:
        value = copy.deepcopy(cast(dict[str, Any], native_runtime()))
        expected_library = RUNTIME_ROOT / "native-libs/root/usr/lib/x86_64-linux-gnu"
        if Path(value["environment"]["LD_LIBRARY_PATH"]) != expected_library:
            raise ValueError("exact staged parent native-library root required")
        value["environment"]["LD_LIBRARY_PATH"] = CANONICAL_PARENT_LIBRARY_PATH
        return value

    wrapper.runner.runtime = parent_canonical_runtime
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
