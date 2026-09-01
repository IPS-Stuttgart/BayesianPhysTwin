#!/usr/bin/env python3
"""Run the fresh Slingshot v4 recovery component."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

from bayesian_phystwin.query_portfolio_replication_v4 import (
    SLINGSHOT_SENSOR_SEEDS,
    SLINGSHOT_WORLD_SEEDS,
)

ROOT = Path(__file__).resolve().parents[2]
V3_PATH = ROOT / "scripts/remote/run_dlolab_slingshot_portfolio_replication_v3.py"
BASE = Path("/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1")
OUTPUT = BASE / "slingshot-v4"
ATTEMPT = BASE / "slingshot-v4.attempt.json"
COMPLETE_UPSTREAM = Path(
    "/home/florianpfaff/source-only/dlolab-runtime-linux-v7-assets/upstream"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_complete_upstream() -> None:
    experiments = COMPLETE_UPSTREAM / "experiments"
    if experiments not in map(Path, sys.path) or COMPLETE_UPSTREAM not in map(
        Path, sys.path
    ):
        raise ValueError("complete frozen upstream paths required")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=COMPLETE_UPSTREAM, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=COMPLETE_UPSTREAM, text=True
    ).strip()
    plane = COMPLETE_UPSTREAM / "genesis/assets/urdf/plane/plane.urdf"
    if (
        revision != "c5026a9416b03c6bc5186eba13cd4ffd4c0e7796"
        or status
        or _sha256(plane)
        != "be1a566d558bd89cabfee5b65d13f3c76acd4c009e3eb8830b369b2dfa079d29"
    ):
        raise ValueError("complete frozen upstream tree changed")


def _load_runner() -> Any:
    _verify_complete_upstream()
    spec = importlib.util.spec_from_file_location("slingshot_portfolio_v4_base", V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registered Slingshot v3 adapter")
    v3: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3)
    wrapper = v3._load_runner()
    wrapper.method.WORLD_SEEDS.update(SLINGSHOT_WORLD_SEEDS)
    wrapper.method.SENSOR_SEEDS.update(SLINGSHOT_SENSOR_SEEDS)
    wrapper.OUTPUT_ROOT = OUTPUT
    wrapper.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.OUTPUT_ROOT = OUTPUT
    wrapper.runner.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.WORKER_RUNNER_PATH = Path(__file__).resolve()
    extra = (
        "src/bayesian_phystwin/query_portfolio_replication_v4.py",
        "scripts/remote/run_dlolab_slingshot_portfolio_replication_v4.py",
        "configs/experiments/query_portfolio_replication_v4.json",
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
        raise ValueError("registered Slingshot v4 output root required")
    wrapper = _load_runner()
    values = (args.worker_role, args.worker_kind, args.worker_index)
    if args.verify_only:
        if any(value is not None for value in (*values, args.worker_action)):
            raise ValueError("verification cannot be a worker")
        wrapper.runner.verify_result(args.output)
    elif all(value is not None for value in values):
        wrapper.runner.worker(
            args.output, args.worker_role, args.worker_kind, args.worker_index,
            args.worker_action,
        )
    elif any(value is not None for value in (*values, args.worker_action)):
        raise ValueError("complete worker specification required")
    else:
        wrapper.runner.run(args.output)


if __name__ == "__main__":
    main()
