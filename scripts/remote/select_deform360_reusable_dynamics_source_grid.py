#!/usr/bin/env python3
"""Freeze pooled and single-source Deform360 physical-grid selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    select_reusable_dynamics_source_grid,
    validate_reusable_dynamics_source_selection,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
    )
    config = load_reusable_dynamics_config(config_path)
    result = select_reusable_dynamics_source_grid(config, grid_root=args.grid_root)
    validate_reusable_dynamics_source_selection(result, config=config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"source selection already exists: {args.output}")
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": True,
                "candidate_count": result["candidate_count"],
                "selected_pooled_physical_parameters": result[
                    "selected_pooled_physical_parameters"
                ],
                "selected_single_source_physical_parameters": result[
                    "selected_single_source_physical_parameters"
                ],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
