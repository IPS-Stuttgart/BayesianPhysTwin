"""Command-line entry point for headless official PhysTwin refits."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_headless_refit import (
    HeadlessPhysTwinRefitConfig,
    run_headless_phystwin_refit,
)
from bayesian_phystwin.phystwin_refit import REFIT_VARIANTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refit a released PhysTwin checkpoint with reliability-aware tracks."
    )
    parser.add_argument("official_repo")
    parser.add_argument("final_data")
    parser.add_argument("optimal_params")
    parser.add_argument("checkpoint")
    parser.add_argument("cues")
    parser.add_argument("output_dir")
    parser.add_argument("--variant", choices=REFIT_VARIANTS, required=True)
    parser.add_argument("--train-end-frame", type=int, required=True)
    parser.add_argument("--fit-end-frame", type=int)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--observation-variance", type=float, default=2.5e-5)
    parser.add_argument("--model-discrepancy-variance", type=float, default=0.0)
    parser.add_argument("--outlier-variance-multiplier", type=float, default=100.0)
    parser.add_argument("--flow-scale", type=float, default=0.005)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--num-substeps", type=int, default=667)
    parser.add_argument("--track-weight", type=float, default=1.0)
    parser.add_argument("--acceleration-weight", type=float, default=0.01)
    parser.add_argument("--freeze-collision", action="store_true")
    parser.add_argument(
        "--spring-parameterization",
        choices=("dense", "grouped"),
        default="dense",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--profile-grid-count", type=int, default=0)
    parser.add_argument(
        "--profile-object-log-scale-half-width",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--profile-controller-log-scale-half-width",
        type=float,
        default=0.50,
    )
    parser.add_argument("--profile-object-prior-std", type=float, default=0.15)
    parser.add_argument("--profile-controller-prior-std", type=float, default=0.50)
    parser.add_argument("--profile-likelihood-temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--released-trajectory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_headless_phystwin_refit(
        official_repo=args.official_repo,
        final_data_path=args.final_data,
        optimal_params_path=args.optimal_params,
        checkpoint_path=args.checkpoint,
        cues_path=args.cues,
        output_dir=args.output_dir,
        released_trajectory_path=args.released_trajectory,
        config=HeadlessPhysTwinRefitConfig(
            variant=args.variant,
            train_end_frame=args.train_end_frame,
            fit_end_frame=args.fit_end_frame,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            observation_variance=args.observation_variance,
            model_discrepancy_variance=args.model_discrepancy_variance,
            outlier_variance_multiplier=args.outlier_variance_multiplier,
            flow_scale=args.flow_scale,
            dt=args.dt,
            num_substeps=args.num_substeps,
            track_weight=args.track_weight,
            acceleration_weight=args.acceleration_weight,
            optimize_collision=not args.freeze_collision,
            spring_parameterization=args.spring_parameterization,
            early_stopping_patience=args.early_stopping_patience,
            profile_grid_count=args.profile_grid_count,
            profile_object_log_scale_half_width=(
                args.profile_object_log_scale_half_width
            ),
            profile_controller_log_scale_half_width=(
                args.profile_controller_log_scale_half_width
            ),
            profile_object_prior_std=args.profile_object_prior_std,
            profile_controller_prior_std=args.profile_controller_prior_std,
            profile_likelihood_temperature=args.profile_likelihood_temperature,
            device=args.device,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
