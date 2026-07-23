#!/usr/bin/env python3
"""Seal one completed Deform360 held-v8.1 role outcome."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute and integrity-seal a Deform360 held-v8.1 outcome"
    )
    parser.add_argument(
        "--role", choices=("calibration", "confirmation"), required=True
    )
    parser.add_argument("--lock", required=True)
    parser.add_argument("--deployed-code", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    deployed_code = Path(arguments.deployed_code).resolve(strict=True)
    source = deployed_code / "src"
    if not source.is_dir() or source.is_symlink():
        raise ValueError("deployed package source is absent or linked")
    sys.path.insert(0, str(source))
    from bayesian_phystwin import deform360_held_v8_outcome_integrity as integrity

    expected_module = (
        source / "bayesian_phystwin" / "deform360_held_v8_outcome_integrity.py"
    )
    if Path(integrity.__file__).resolve(strict=True) != expected_module:
        raise ValueError("integrity module escaped the deployed source")
    completion = integrity.seal_role_outcome(
        lock_path=arguments.lock,
        role=arguments.role,
        deployed_code=deployed_code,
        operator_source=Path(__file__).resolve(strict=True),
    )
    terminal = completion["terminal_root_finalization"]
    print(
        json.dumps(
            {
                "event": "DEFORM360_V8_ROLE_OUTCOME_INTEGRITY_COMPLETE",
                "role": arguments.role,
                "terminal_outcome": completion["terminal_outcome"],
                "role_completion_path": str(
                    integrity.canonical_role_completion_path(
                        completion["held_root"], arguments.role
                    )
                ),
                "role_completion_artifact_sha256": completion["artifact_sha256"],
                "terminal_root_finalized": terminal["required"],
                "terminal_completion_path": terminal["completion_path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "FAIL_CLOSED",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2) from error
