"""CLI for automatic MotionCrafter-to-PhysTwin graph association."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_motioncrafter_association import (
    MotionCrafterAssociationConfig,
    associate_motioncrafter_case,
)


def _parse_additional_views(values: list[str]) -> dict[int, str]:
    views: dict[int, str] = {}
    for value in values:
        try:
            camera_text, path = value.split("=", 1)
            camera = int(camera_text)
        except ValueError as error:
            raise SystemExit(
                f"invalid --additional-view {value!r}; expected CAMERA=NPZ"
            ) from error
        if camera < 0 or not path:
            raise SystemExit(
                f"invalid --additional-view {value!r}; expected CAMERA=NPZ"
            )
        if camera in views:
            raise SystemExit(f"duplicate --additional-view camera {camera}")
        views[camera] = path
    return views


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Associate dense MotionCrafter motion with a PhysTwin graph."
    )
    parser.add_argument("case_dir")
    parser.add_argument("raw_case_dir")
    parser.add_argument("motioncrafter_npz")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--reverse-motioncrafter-npz",
        help="optional reverse-video output for offline future identity coverage",
    )
    parser.add_argument("--train-end-frame", type=int)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--additional-view",
        action="append",
        default=[],
        metavar="CAMERA=NPZ",
        help="add another calibrated MotionCrafter camera view",
    )
    parser.add_argument("--process-stride", type=int, default=1)
    parser.add_argument("--seed-stride-pixels", type=int, default=4)
    parser.add_argument("--alignment-stride-pixels", type=int, default=4)
    parser.add_argument("--alignment-trim-fraction", type=float, default=0.8)
    parser.add_argument("--maximum-transport-error-m", type=float, default=0.02)
    parser.add_argument("--transport-candidate-count", type=int, default=4)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--position-scale-m", type=float, default=0.01)
    parser.add_argument("--motion-scale-m", type=float, default=0.02)
    parser.add_argument("--motion-strength", type=float, default=1.0)
    parser.add_argument("--graph-scale-m", type=float, default=0.015)
    parser.add_argument("--graph-strength", type=float, default=0.3)
    parser.add_argument("--collision-strength", type=float, default=0.1)
    parser.add_argument("--mean-field-iterations", type=int, default=5)
    parser.add_argument(
        "--minimum-trajectory-valid-fraction", type=float, default=0.5
    )
    parser.add_argument("--minimum-observation-mass", type=float, default=0.5)
    args = parser.parse_args()
    additional_views = _parse_additional_views(args.additional_view)
    result = associate_motioncrafter_case(
        args.case_dir,
        args.raw_case_dir,
        args.motioncrafter_npz,
        args.output_dir,
        train_end_frame=args.train_end_frame,
        additional_views=additional_views,
        reverse_motioncrafter_npz_path=args.reverse_motioncrafter_npz,
        config=MotionCrafterAssociationConfig(
            camera_index=args.camera_index,
            process_stride=args.process_stride,
            seed_stride_pixels=args.seed_stride_pixels,
            alignment_stride_pixels=args.alignment_stride_pixels,
            alignment_trim_fraction=args.alignment_trim_fraction,
            maximum_transport_error_m=args.maximum_transport_error_m,
            transport_candidate_count=args.transport_candidate_count,
            candidate_count=args.candidate_count,
            position_scale_m=args.position_scale_m,
            motion_scale_m=args.motion_scale_m,
            motion_strength=args.motion_strength,
            graph_scale_m=args.graph_scale_m,
            graph_strength=args.graph_strength,
            collision_strength=args.collision_strength,
            mean_field_iterations=args.mean_field_iterations,
            minimum_trajectory_valid_fraction=(
                args.minimum_trajectory_valid_fraction
            ),
            minimum_observation_mass=args.minimum_observation_mass,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
