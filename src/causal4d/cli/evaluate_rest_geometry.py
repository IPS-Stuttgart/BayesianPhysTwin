"""CLI for graph-regularized PhysTwin rest-geometry experiments."""

from __future__ import annotations

import argparse
import json

from causal4d.phystwin_rest_geometry import (
    run_phystwin_rest_geometry_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Infer frame/rest geometry from O-minus, inject it into Warp, and "
            "evaluate the untouched future."
        )
    )
    parser.add_argument("official_repo")
    parser.add_argument("data_root")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--cohort",
        choices=("all", "development", "confirmation"),
        default="development",
    )
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--frame-mode",
        choices=("none", "translation", "se3"),
        default="se3",
    )
    parser.add_argument(
        "--frame-scale",
        action="append",
        type=float,
        dest="frame_scales",
    )
    parser.add_argument(
        "--rest-geometry-scale",
        action="append",
        type=float,
        dest="rest_scales",
    )
    parser.add_argument(
        "--controller-rest-mode",
        action="append",
        choices=("preserve", "recompute"),
        dest="controller_rest_modes",
    )
    parser.add_argument("--graph-prior-strength", type=float, default=0.1)
    parser.add_argument("--inner-validation-frames", type=int, default=8)
    parser.add_argument("--velocity-history-frames", type=int, default=3)
    parser.add_argument("--maximum-frame-rotation-deg", type=float, default=5.0)
    parser.add_argument("--maximum-frame-translation-m", type=float, default=0.02)
    parser.add_argument("--maximum-nonrigid-norm-m", type=float, default=0.01)
    parser.add_argument("--maximum-rest-ratio", type=float, default=1.15)
    parser.add_argument("--atomic-spring-forces", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.maximum_rest_ratio <= 1.0:
        parser.error("--maximum-rest-ratio must be greater than one")

    import numpy as np

    result = run_phystwin_rest_geometry_comparison(
        args.official_repo,
        args.data_root,
        args.output_dir,
        cohort=args.cohort,
        cases=args.cases,
        frame_mode=args.frame_mode,
        frame_scale_grid=(
            (0.0, 0.5, 1.0) if args.frame_scales is None else args.frame_scales
        ),
        rest_geometry_scale_grid=(
            (0.0, 0.25, 0.5, 1.0)
            if args.rest_scales is None
            else args.rest_scales
        ),
        controller_rest_mode_grid=(
            ("preserve", "recompute")
            if args.controller_rest_modes is None
            else args.controller_rest_modes
        ),
        graph_prior_strength=args.graph_prior_strength,
        inner_validation_frames=args.inner_validation_frames,
        velocity_history_frames=args.velocity_history_frames,
        maximum_frame_rotation_rad=np.deg2rad(args.maximum_frame_rotation_deg),
        maximum_frame_translation_m=args.maximum_frame_translation_m,
        maximum_nonrigid_norm_m=args.maximum_nonrigid_norm_m,
        maximum_rest_log_ratio=np.log(args.maximum_rest_ratio),
        deterministic_spring_forces=not args.atomic_spring_forces,
        bootstrap_samples=args.bootstrap_samples,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
