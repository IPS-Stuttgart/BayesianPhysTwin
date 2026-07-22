#!/usr/bin/env python3
"""Run the fresh two-barrier held-v8 calibration outcome phase."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main() -> int:
    source = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(source))
    from bayesian_phystwin.deform360_held_v8_outcome_driver import main_for_role

    return main_for_role("calibration")


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
