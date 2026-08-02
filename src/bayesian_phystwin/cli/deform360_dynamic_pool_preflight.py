"""CLI for target-free Open27 dynamic-pool feasibility."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_dynamic_pool_preflight import (
    run_dynamic_pool_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check the frozen 64-point Open27 frame-zero contract without "
            "reading RGB frames or outcomes."
        )
    )
    parser.add_argument("panel_root")
    parser.add_argument("processed_root")
    parser.add_argument("output_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_dynamic_pool_preflight(
        args.panel_root,
        args.processed_root,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
