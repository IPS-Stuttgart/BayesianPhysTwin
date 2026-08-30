#!/usr/bin/env python3
"""Verify the frozen DLO-Lab Slingshot policy-certificate source result."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, cast


def _runner() -> ModuleType:
    path = Path(__file__).with_name(
        "run_dlolab_slingshot_policy_certificate_source_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        "dlolab_slingshot_policy_certificate_source_v2_runner", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("policy-certificate runner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(output: Path) -> dict[str, Any]:
    """Load the exact runner and reproduce its complete result."""

    runner = _runner()
    return cast(dict[str, Any], runner.verify_result(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.output)
    print(
        f"verified Slingshot policy-certificate result {result['artifact_id']}",
        flush=True,
    )
