"""CLI for trust-gated official PGRD inference on PhysTwin trajectories."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_pgrd_adapter import (
    PhysTwinPGRDAdapterConfig,
    fit_pgrd_residual_adapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pinned PGRD residual checkpoint behind a causal gate."
    )
    parser.add_argument("final_data")
    parser.add_argument("baseline_trajectory")
    parser.add_argument("gt_track_3d")
    parser.add_argument("output_dir")
    parser.add_argument("--pgrd-checkout", required=True)
    parser.add_argument("--pgrd-checkpoint", required=True)
    parser.add_argument("--fit-end-frame", type=int, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--number-of-points", type=int, default=512)
    parser.add_argument("--normalized-extent", type=float, action="append", dest="extents")
    parser.add_argument("--yaw-degrees", type=float, action="append", dest="yaw_degrees")
    parser.add_argument("--trust", type=float, action="append", dest="trust")
    parser.add_argument("--model-frame-stride", type=int, default=3)
    parser.add_argument("--maximum-residual-m", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    defaults = PhysTwinPGRDAdapterConfig(
        fit_end_frame=args.fit_end_frame,
        train_end_frame=args.train_end_frame,
    )
    summary = fit_pgrd_residual_adapter(
        args.final_data,
        args.baseline_trajectory,
        args.gt_track_3d,
        args.output_dir,
        config=PhysTwinPGRDAdapterConfig(
            fit_end_frame=args.fit_end_frame,
            train_end_frame=args.train_end_frame,
            normalized_extent_candidates=(
                defaults.normalized_extent_candidates
                if args.extents is None
                else tuple(args.extents)
            ),
            yaw_candidates_degrees=(
                defaults.yaw_candidates_degrees
                if args.yaw_degrees is None
                else tuple(args.yaw_degrees)
            ),
            trust_candidates=(
                defaults.trust_candidates if args.trust is None else tuple(args.trust)
            ),
            number_of_points=args.number_of_points,
            model_frame_stride=args.model_frame_stride,
            maximum_residual_m=args.maximum_residual_m,
        ),
        pgrd_checkout=args.pgrd_checkout,
        pgrd_checkpoint=args.pgrd_checkpoint,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
