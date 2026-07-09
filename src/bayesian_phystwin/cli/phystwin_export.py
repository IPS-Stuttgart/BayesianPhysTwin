"""CLI for exporting official PhysTwin tracked-point residuals."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.phystwin_adapter import (
    PhysTwinExportConfig,
    export_phystwin_residuals,
    write_export_summary,
)
from bayesian_phystwin.pseudo_measurements import ReliabilityConfig
from bayesian_phystwin.residual_replay import replay_residual_csv
from bayesian_phystwin.robust_likelihood import RobustLikelihoodConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export canonical residuals from PhysTwin final_data/inference pickles."
    )
    parser.add_argument("final_data_pickle")
    parser.add_argument("trajectory_pickle")
    parser.add_argument("output_csv")
    parser.add_argument("--cues-npz")
    parser.add_argument("--variance", type=float, default=1e-4)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--correspondence", choices=("direct", "nearest"), default="direct")
    parser.add_argument("--summary-json")
    parser.add_argument("--replay-summary-json")
    parser.add_argument("--scored-csv")
    parser.add_argument("--replay-flow-scale", type=float, default=0.10)
    parser.add_argument("--replay-boundary-scale", type=float, default=0.03)
    parser.add_argument("--replay-model-discrepancy-variance", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = export_phystwin_residuals(
        args.final_data_pickle,
        args.trajectory_pickle,
        args.output_csv,
        cues_path=args.cues_npz,
        config=PhysTwinExportConfig(
            variance=args.variance,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            include_invalid=args.include_invalid,
            correspondence=args.correspondence,
        ),
    )
    if args.summary_json:
        write_export_summary(summary, args.summary_json)

    output: dict[str, object] = {"export": summary}
    if args.replay_summary_json or args.scored_csv:
        replay = replay_residual_csv(
            args.output_csv,
            reliability_config=ReliabilityConfig(
                flow_scale=args.replay_flow_scale,
                boundary_scale=args.replay_boundary_scale,
            ),
            likelihood_config=RobustLikelihoodConfig(
                model_discrepancy_variance=args.replay_model_discrepancy_variance,
            ),
        )
        if args.replay_summary_json:
            replay.write_summary_json(args.replay_summary_json)
        if args.scored_csv:
            replay.write_scored_csv(args.scored_csv)
        output["replay"] = replay.summary
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
