#!/usr/bin/env python3
"""Seal one source-only Deform360 process-isolation qualification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys


EXPECTED_HOST = "workstation2"
RELATIVE_SOURCE = Path(
    "scripts/held/seal_deform360_process_isolation_qualification.py"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if not (
        sys.flags.isolated == 1
        and sys.flags.ignore_environment == 1
        and sys.flags.no_user_site == 1
        and sys.dont_write_bytecode
    ):
        raise RuntimeError("run the process-isolation sealer with Python -I -B")
    if socket.gethostname() != EXPECTED_HOST:
        raise RuntimeError("process-isolation sealer host changed")
    code = Path(os.path.abspath(os.fspath(arguments.code_root)))
    source = (code / RELATIVE_SOURCE).resolve(strict=True)
    if source != Path(__file__).resolve(strict=True):
        raise RuntimeError("process-isolation sealer escaped the bound code root")
    sys.path.insert(0, os.fspath(code / "src"))
    from bayesian_phystwin import deform360_process_isolation_qualification as gate

    lineage = gate.seal_process_isolation_qualification(
        arguments.qualification_root,
        arguments.completion,
        sealer_source_path=source,
    )
    print(
        json.dumps(
            {
                "operation": "sealed-process-isolation-qualification",
                "qualification_root": os.fspath(
                    Path(arguments.qualification_root).resolve(strict=True)
                ),
                "completion": lineage[
                    "process_isolation_qualification_integrity_completion"
                ],
                "source_head": lineage[
                    "process_isolation_qualification_integrity"
                ]["source_head"],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
