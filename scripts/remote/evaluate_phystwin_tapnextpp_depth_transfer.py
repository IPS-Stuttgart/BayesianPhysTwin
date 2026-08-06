#!/usr/bin/env python3
"""Apply the frozen aggregate gate to TAPNext++ source-transfer results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.tapnextpp_transfer_evaluation import (
    evaluate_transfer_panel,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--case-result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = evaluate_transfer_panel(
        args.protocol,
        args.source_manifest,
        args.case_result_root,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
