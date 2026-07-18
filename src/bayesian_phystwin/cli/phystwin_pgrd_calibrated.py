"""CLI for prefix-calibrated PGRD residual velocity."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_pgrd_calibrated import (
    PhysTwinPGRDCalibrationConfig,
    fit_prefix_calibrated_pgrd,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a 3x3 PGRD readout on a causal prefix.")
    parser.add_argument("final_data")
    parser.add_argument("baseline_trajectory")
    parser.add_argument("gt_track_3d")
    parser.add_argument("output_dir")
    parser.add_argument("--pgrd-checkout", required=True)
    parser.add_argument("--pgrd-checkpoint", required=True)
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--number-of-points", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = fit_prefix_calibrated_pgrd(
        args.final_data,
        args.baseline_trajectory,
        args.gt_track_3d,
        args.output_dir,
        config=PhysTwinPGRDCalibrationConfig(
            fit_end_frame=args.fit_end_frame,
            train_end_frame=args.train_end_frame,
            number_of_points=args.number_of_points,
        ),
        pgrd_checkout=args.pgrd_checkout,
        pgrd_checkpoint=args.pgrd_checkpoint,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
