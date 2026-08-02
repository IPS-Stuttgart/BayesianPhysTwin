#!/usr/bin/env python3
"""Seal a declared exact-persistence fallback after a provider failure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bayesian_phystwin.deform360_tactile_guard_outcome_sealed import (  # noqa: E402
    build_technical_fallback,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--backbone-case-dir", type=Path, required=True)
    parser.add_argument("--failure-stage", required=True)
    parser.add_argument("--failure-type", required=True)
    parser.add_argument("--failure-message", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_technical_fallback(
        args.output_dir,
        protocol_path=args.protocol,
        backbone_dir=args.backbone_case_dir,
        failure_stage=args.failure_stage,
        failure_type=args.failure_type,
        failure_message=args.failure_message,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
