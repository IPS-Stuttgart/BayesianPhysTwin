"""CLI for physics-guided anonymous MotionCrafter assimilation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_motioncrafter_assimilation import (
    AnonymousSceneFlowConfig,
    assimilate_motioncrafter_case,
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
        if camera < 0 or not path or camera in views:
            raise SystemExit(
                f"invalid --additional-view {value!r}; expected unique CAMERA=NPZ"
            )
        views[camera] = path
    return views


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-associate anonymous MotionCrafter positions/flow to a persistent "
            "PhysTwin graph every frame."
        )
    )
    parser.add_argument("case_dir")
    parser.add_argument("raw_case_dir")
    parser.add_argument("motioncrafter_npz")
    parser.add_argument("output_dir")
    parser.add_argument("--train-end-frame", type=int)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--additional-view",
        action="append",
        default=[],
        metavar="CAMERA=NPZ",
    )
    parser.add_argument("--process-stride", type=int, default=1)
    parser.add_argument("--measurement-stride-pixels", type=int, default=4)
    parser.add_argument("--alignment-stride-pixels", type=int, default=4)
    parser.add_argument("--alignment-trim-fraction", type=float, default=0.8)
    parser.add_argument("--alignment-iterations", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--position-scale-m", type=float, default=0.01)
    parser.add_argument("--flow-scale-m", type=float, default=0.02)
    parser.add_argument("--flow-strength", type=float, default=1.0)
    parser.add_argument("--maximum-position-error-m", type=float, default=0.04)
    parser.add_argument("--maximum-flow-endpoint-error-m", type=float, default=0.06)
    parser.add_argument("--entropy-strength", type=float, default=0.5)
    parser.add_argument("--minimum-observation-mass", type=float, default=0.2)
    parser.add_argument("--multiview-consistency-scale-m", type=float, default=0.015)
    parser.add_argument("--minimum-multiview-reliability", type=float, default=0.05)
    parser.add_argument("--graph-prior-strength", type=float, default=0.3)
    parser.add_argument("--graph-zero-prior-strength", type=float, default=0.0)
    parser.add_argument("--graph-ridge", type=float, default=1e-8)
    parser.add_argument("--graph-solver-relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--graph-solver-maximum-iterations", type=int, default=5000)
    parser.add_argument("--graph-covariance-probes", type=int, default=0)
    parser.add_argument("--graph-covariance-manual-track-audit", action="store_true")
    parser.add_argument("--maximum-graph-correction-m", type=float, default=0.01)
    parser.add_argument(
        "--reliability-mode",
        choices=("legacy", "decoupled_robust"),
        default="legacy",
    )
    parser.add_argument(
        "--multiview-fusion-mode",
        choices=("legacy_independent", "covariance_intersection"),
        default="legacy_independent",
    )
    parser.add_argument("--correlation-block-pixels", type=int, default=16)
    parser.add_argument("--boundary-reliability-scale-pixels", type=float, default=8.0)
    parser.add_argument("--boundary-reliability-floor", type=float, default=0.25)
    parser.add_argument("--observation-variance-floor-m2", type=float, default=4e-6)
    parser.add_argument(
        "--robust-outlier-variance-multiplier", type=float, default=100.0
    )
    parser.add_argument(
        "--robust-model-discrepancy-variance-m2", type=float, default=0.0
    )
    args = parser.parse_args()
    result = assimilate_motioncrafter_case(
        args.case_dir,
        args.raw_case_dir,
        args.motioncrafter_npz,
        args.output_dir,
        train_end_frame=args.train_end_frame,
        additional_views=_parse_additional_views(args.additional_view),
        config=AnonymousSceneFlowConfig(
            camera_index=args.camera_index,
            process_stride=args.process_stride,
            measurement_stride_pixels=args.measurement_stride_pixels,
            alignment_stride_pixels=args.alignment_stride_pixels,
            alignment_trim_fraction=args.alignment_trim_fraction,
            alignment_iterations=args.alignment_iterations,
            candidate_count=args.candidate_count,
            position_scale_m=args.position_scale_m,
            flow_scale_m=args.flow_scale_m,
            flow_strength=args.flow_strength,
            maximum_position_error_m=args.maximum_position_error_m,
            maximum_flow_endpoint_error_m=(args.maximum_flow_endpoint_error_m),
            entropy_strength=args.entropy_strength,
            minimum_observation_mass=args.minimum_observation_mass,
            multiview_consistency_scale_m=(args.multiview_consistency_scale_m),
            minimum_multiview_reliability=(args.minimum_multiview_reliability),
            graph_prior_strength=args.graph_prior_strength,
            graph_zero_prior_strength=args.graph_zero_prior_strength,
            graph_ridge=args.graph_ridge,
            graph_solver_relative_tolerance=(args.graph_solver_relative_tolerance),
            graph_solver_maximum_iterations=(args.graph_solver_maximum_iterations),
            graph_covariance_probes=args.graph_covariance_probes,
            graph_covariance_manual_track_audit=(
                args.graph_covariance_manual_track_audit
            ),
            maximum_graph_correction_m=args.maximum_graph_correction_m,
            reliability_mode=args.reliability_mode,
            multiview_fusion_mode=args.multiview_fusion_mode,
            correlation_block_pixels=args.correlation_block_pixels,
            boundary_reliability_scale_pixels=(args.boundary_reliability_scale_pixels),
            boundary_reliability_floor=args.boundary_reliability_floor,
            observation_variance_floor_m2=args.observation_variance_floor_m2,
            robust_outlier_variance_multiplier=(
                args.robust_outlier_variance_multiplier
            ),
            robust_model_discrepancy_variance_m2=(
                args.robust_model_discrepancy_variance_m2
            ),
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
