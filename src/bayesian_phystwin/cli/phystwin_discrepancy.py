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
    parser.add_argument("--final-data-sha256", required=True)
    parser.add_argument("--parameter-profile-sha256", required=True)
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--test-start-frame", type=int, required=True)
    parser.add_argument("--observation-variance", type=float, default=2.5e-5)
    parser.add_argument("--decays", default="0,0.5,0.8,0.9,0.95,0.98,0.99")
    parser.add_argument("--reference-trajectory")
    parser.add_argument("--reference-trajectory-sha256")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.reference_trajectory) != bool(args.reference_trajectory_sha256):
        parser.error(
            "--reference-trajectory and --reference-trajectory-sha256 "
            "must be supplied together"
        )
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
        final_data_sha256=args.final_data_sha256,
        profile_sha256=args.parameter_profile_sha256,
        reference_trajectory_path=args.reference_trajectory,
        reference_trajectory_sha256=args.reference_trajectory_sha256,
    )
    write_discrepancy_summary(summary, args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
