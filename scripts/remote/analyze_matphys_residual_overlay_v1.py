#!/usr/bin/env python3
"""Evaluate frozen residual overlays on selected MatPhys physical families."""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

from bayesian_phystwin_experiments.matphys_residual_overlay_audit import (
    build_matphys_residual_overlay_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root")
    parser.add_argument("selection_summary")
    parser.add_argument("future_summary")
    parser.add_argument("output")
    parser.add_argument("--selection-sha256")
    parser.add_argument("--future-sha256")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-block-length", type=int, default=5)
    parser.add_argument("--bootstrap-seed", type=int, default=20260822)
    args = parser.parse_args()
    result = build_matphys_residual_overlay_audit(
        args.data_root,
        args.selection_summary,
        args.future_summary,
        args.output,
        expected_selection_sha256=args.selection_sha256,
        expected_future_sha256=args.future_sha256,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_seed=args.bootstrap_seed,
    )
    primary = cast(dict[str, Any], result["primary_nonzero_matphys_subset"])
    print(
        json.dumps(
            {
                "output_path": result["output_path"],
                "output_sha256": result["output_sha256"],
                "primary_nonzero_matphys_subset": primary["equal_case_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
