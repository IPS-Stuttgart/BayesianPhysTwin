#!/usr/bin/env python3
"""Generate or verify the controlled adaptive routing result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.adaptive_act_sense_fallback_certificate_v1 import (
    controlled_adaptive_router_demo,
)

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "result.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(
        controlled_adaptive_router_demo(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if args.check:
        if RESULT.read_text(encoding="utf-8") != text:
            raise SystemExit("result.json is stale")
    else:
        RESULT.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
