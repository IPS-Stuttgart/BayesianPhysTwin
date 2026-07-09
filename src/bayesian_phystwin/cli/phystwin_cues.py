"""CLI for continuous motion-consistency cues from PhysTwin tracks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.phystwin_adapter import (
    PhysTwinMotionCueConfig,
    build_phystwin_motion_cues,
    write_export_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build simulator-independent motion cues from PhysTwin final_data.pkl."
    )
    parser.add_argument("final_data_pickle")
    parser.add_argument("output_npz")
    parser.add_argument("--neighbor-count", type=int, default=16)
    parser.add_argument("--minimum-valid-neighbors", type=int, default=4)
    parser.add_argument("--neighbor-radius", type=float, default=0.01)
    parser.add_argument("--neighbor-reference", choices=("first", "current"), default="current")
    parser.add_argument("--insufficient-neighbor-value", type=float, default=0.10)
    parser.add_argument("--summary-json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_phystwin_motion_cues(
        args.final_data_pickle,
        args.output_npz,
        config=PhysTwinMotionCueConfig(
            neighbor_count=args.neighbor_count,
            minimum_valid_neighbors=args.minimum_valid_neighbors,
            neighbor_radius=args.neighbor_radius,
            neighbor_reference=args.neighbor_reference,
            insufficient_neighbor_value=args.insufficient_neighbor_value,
        ),
    )
    if args.summary_json:
        write_export_summary(summary, args.summary_json)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
