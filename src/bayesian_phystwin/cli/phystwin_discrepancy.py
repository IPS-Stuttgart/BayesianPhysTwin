"""CLI for causal calibration of PhysTwin profile model discrepancy."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_discrepancy import (
    PhysTwinDiscrepancyConfig,
    calibrate_phystwin_profile_discrepancy,
    write_discrepancy_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate causal model discrepancy for a saved PhysTwin profile."
    )
    parser.add_argument("final_data")
    parser.add_argument("parameter_profile")
    parser.add_argument("output_json")
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--test-start-frame", type=int, required=True)
    parser.add_argument("--observation-variance", type=float, default=2.5e-5)
    parser.add_argument("--decays", default="0,0.5,0.8,0.9,0.95,0.98,0.99")
    parser.add_argument("--reference-trajectory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    decays = tuple(float(value) for value in args.decays.split(",") if value)
    summary = calibrate_phystwin_profile_discrepancy(
        args.final_data,
        args.parameter_profile,
        config=PhysTwinDiscrepancyConfig(
            fit_end_frame=args.fit_end_frame,
            test_start_frame=args.test_start_frame,
            observation_variance=args.observation_variance,
            decay_candidates=decays,
        ),
        reference_trajectory_path=args.reference_trajectory,
    )
    write_discrepancy_summary(summary, args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
