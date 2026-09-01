#!/usr/bin/env python3
"""Run the fresh Slingshot v2 recovery component."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, cast

from bayesian_phystwin.query_portfolio_replication_v2 import (
    SLINGSHOT_SENSOR_SEEDS,
    SLINGSHOT_WORLD_SEEDS,
)

ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "scripts/remote/run_dlolab_slingshot_portfolio_replication_v1.py"
BASE = Path("/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1")
OUTPUT = BASE / "slingshot-v2"
ATTEMPT = BASE / "slingshot-v2.attempt.json"
STAGED_LIBRARY_PATH = BASE / "frozen-runtime/native-libs/root/usr/lib/x86_64-linux-gnu"
CANONICAL_PARENT_LIBRARY_PATH = (
    "/home/fpfaff/source-only/dlo-lab-decision-v1-assets/"
    "native-libs/root/usr/lib/x86_64-linux-gnu"
)


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("slingshot_portfolio_v2_base", V1_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registered Slingshot v1 adapter")
    v1: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    wrapper = v1._load_runner()
    wrapper.method.WORLD_SEEDS.update(SLINGSHOT_WORLD_SEEDS)
    wrapper.method.SENSOR_SEEDS.update(SLINGSHOT_SENSOR_SEEDS)
    wrapper.OUTPUT_ROOT = OUTPUT
    wrapper.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.OUTPUT_ROOT = OUTPUT
    wrapper.runner.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.WORKER_RUNNER_PATH = Path(__file__).resolve()
    inherited_environment = wrapper.runner.native_worker_environment

    def staged_worker_environment(registered: dict[str, Any]) -> dict[str, str]:
        value = cast(dict[str, str], inherited_environment(registered))
        if (
            v1.CANONICAL_PARENT_LIBRARY_PATH != CANONICAL_PARENT_LIBRARY_PATH
            or value["LD_LIBRARY_PATH"] != CANONICAL_PARENT_LIBRARY_PATH
        ):
            raise ValueError("canonical parent worker environment changed")
        if not (STAGED_LIBRARY_PATH / "libOSMesa.so.8").is_file():
            raise ValueError("staged parent OSMesa library missing")
        value["LD_LIBRARY_PATH"] = str(STAGED_LIBRARY_PATH)
        return value

    wrapper.runner.native_worker_environment = staged_worker_environment
    extra = (
        "src/bayesian_phystwin/query_portfolio_replication_v2.py",
        "scripts/remote/run_dlolab_slingshot_portfolio_replication_v2.py",
        "configs/experiments/query_portfolio_replication_v2.json",
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
        raise ValueError("registered Slingshot v2 output root required")
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
