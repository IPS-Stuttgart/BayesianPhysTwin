#!/usr/bin/env python3
"""Derive the frozen trust-aligned point-MAP control from source evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_reusable_ensemble import (
    derive_source_trusted_point_map_control,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gibbs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    gibbs = json.loads(args.source_gibbs.read_text(encoding="utf-8"))
    result = derive_source_trusted_point_map_control(gibbs)
    if args.output.exists():
        raise FileExistsError(f"trusted point-MAP artifact exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "selected_pooled_candidate_label": result[
                    "selected_pooled_candidate_label"
                ],
                "source_diagnostics": result["source_diagnostics"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
