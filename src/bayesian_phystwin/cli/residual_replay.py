"""CLI for reliability-aware replay of exported PhysTwin residuals."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from bayesian_phystwin.pseudo_measurements import ReliabilityConfig
from bayesian_phystwin.residual_replay import replay_residual_csv
from bayesian_phystwin.robust_likelihood import RobustLikelihoodConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score exported PhysTwin residuals with a robust reliability model."
    )
    parser.add_argument("input_csv", help="Canonical residual CSV to replay")
    parser.add_argument("--summary-json", help="Write aggregate metrics as JSON")
    parser.add_argument("--scored-csv", help="Write original rows plus reliability scores")
    parser.add_argument("--default-variance", type=float, default=1e-4)
    parser.add_argument("--min-weight", type=float, default=1e-3)
    parser.add_argument("--confidence-power", type=float, default=1.0)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="Enable residual-gated baseline; omit for cue-only prior reliability",
    )
    parser.add_argument("--boundary-scale", type=float, default=0.03)
    parser.add_argument("--flow-scale", type=float, default=0.10)
    parser.add_argument("--occlusion-weight", type=float, default=0.05)
    parser.add_argument("--covariance-inflation-cap", type=float, default=100.0)
    parser.add_argument("--outlier-variance-multiplier", type=float, default=100.0)
    parser.add_argument("--model-discrepancy-variance", type=float, default=0.0)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reliability_config = ReliabilityConfig(
        min_weight=args.min_weight,
        confidence_power=args.confidence_power,
        residual_scale=args.residual_scale,
        boundary_scale=args.boundary_scale,
        flow_scale=args.flow_scale,
        occlusion_weight=args.occlusion_weight,
        covariance_inflation_at_min_weight=args.covariance_inflation_cap,
    )
    likelihood_config = RobustLikelihoodConfig(
        outlier_variance_multiplier=args.outlier_variance_multiplier,
        model_discrepancy_variance=args.model_discrepancy_variance,
    )
    result = replay_residual_csv(
        args.input_csv,
        reliability_config=reliability_config,
        likelihood_config=likelihood_config,
        default_variance=args.default_variance,
        calibration_bins=args.calibration_bins,
    )
    if args.summary_json:
        result.write_summary_json(args.summary_json)
    if args.scored_csv:
        result.write_scored_csv(args.scored_csv)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
