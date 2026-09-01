#!/usr/bin/env python3
"""Run the fresh Slingshot v3 recovery component."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from bayesian_phystwin.query_portfolio_replication_v3 import (
    SLINGSHOT_SENSOR_SEEDS,
    SLINGSHOT_WORLD_SEEDS,
)

ROOT = Path(__file__).resolve().parents[2]
V2_PATH = ROOT / "scripts/remote/run_dlolab_slingshot_portfolio_replication_v2.py"
BASE = Path("/home/florianpfaff/source-only/dlolab-query-portfolio-replication-v1")
OUTPUT = BASE / "slingshot-v3"
ATTEMPT = BASE / "slingshot-v3.attempt.json"
UPSTREAM_ROOT = BASE / "frozen-runtime/upstream"
ADDITIONS = BASE / "frozen-runtime-additions-v3"
ADDITIONS_MANIFEST = BASE / "frozen-runtime-additions-v3.sha256"
NATIVE_DEPENDENCIES = Path(
    "/home/florianpfaff/source-only/dlolab-runtime-linux-v7-assets/native-libs"
)
EXPECTED_ADDITIONS_MANIFEST_SHA256 = (
    "f10841ae78a89aa0375f60f8f3da3bd0331c5434bd7f0e75ebc22076f4651a03"
)
EXPECTED_PRELOAD = ":".join(
    str(NATIVE_DEPENDENCIES / name)
    for name in ("libLLVM-15.so.1", "libglapi.so.0")
)
EXPECTED_NATIVE_SHA256 = {
    "libLLVM-15.so.1": "de2e35a4f9b3f6a06d2a8a3342b3f62a3842b1923b8dfc2a6ce48e0cc2d1e85d",
    "libglapi.so.0": "6b0b3d9623ca09ae7d16d3320d8866dc0557d67e9cbb63c12752fe723444a0a1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_runtime_additions() -> None:
    required_paths = (UPSTREAM_ROOT / "experiments", UPSTREAM_ROOT, ADDITIONS)
    if any(str(path) not in sys.path for path in required_paths):
        raise ValueError("exact staged DLO-Lab Python paths required")
    if (
        _sha256(ADDITIONS_MANIFEST) != EXPECTED_ADDITIONS_MANIFEST_SHA256
        or importlib.metadata.version("mediapy") != "1.2.7"
        or importlib.metadata.version("ipython") != "9.17.0"
    ):
        raise ValueError("staged Python runtime additions changed")
    if os.environ.get("LD_PRELOAD") != EXPECTED_PRELOAD or any(
        _sha256(NATIVE_DEPENDENCIES / name) != digest
        for name, digest in EXPECTED_NATIVE_SHA256.items()
    ):
        raise ValueError("staged native runtime dependencies changed")


def _load_runner() -> Any:
    _verify_runtime_additions()
    spec = importlib.util.spec_from_file_location("slingshot_portfolio_v3_base", V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load registered Slingshot v2 adapter")
    v2: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2)
    wrapper = v2._load_runner()
    wrapper.method.WORLD_SEEDS.update(SLINGSHOT_WORLD_SEEDS)
    wrapper.method.SENSOR_SEEDS.update(SLINGSHOT_SENSOR_SEEDS)
    wrapper.OUTPUT_ROOT = OUTPUT
    wrapper.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.OUTPUT_ROOT = OUTPUT
    wrapper.runner.ATTEMPT_LEDGER = ATTEMPT
    wrapper.runner.WORKER_RUNNER_PATH = Path(__file__).resolve()
    extra = (
        "src/bayesian_phystwin/query_portfolio_replication_v3.py",
        "scripts/remote/run_dlolab_slingshot_portfolio_replication_v3.py",
        "configs/experiments/query_portfolio_replication_v3.json",
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
        raise ValueError("registered Slingshot v3 output root required")
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
