"""CLI for causal AllTracker Deform360 observations and evaluation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_raw_camera_observation import (
    ALLTRACKER_MOLMOMOTION_REVISION,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    build_raw_camera_measurement_cohort,
    evaluate_raw_camera_measurement_cohort,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build outcome-unaware raw-camera measurements or evaluate their "
            "already-hashed open-27 cohort."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-cohort",
        help="Run causal AllTracker prefixes for one deterministic cohort shard.",
    )
    build.add_argument("panel_root")
    build.add_argument("processed_root")
    build.add_argument("output_root")
    build.add_argument("alltracker_source")
    build.add_argument("checkpoint")
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--shard-index", type=int, default=0)
    build.add_argument("--shard-count", type=int, default=1)
    build.add_argument(
        "--center-count",
        type=int,
        default=16,
        help="Causal frame-zero observation pool size; frozen default remains 16.",
    )
    build.add_argument("--camera-count", type=int, default=8)
    build.add_argument("--max-side", type=int, default=512)
    build.add_argument(
        "--tracker-revision",
        default=ALLTRACKER_MOLMOMOTION_REVISION,
        help="Audit-only lock; must match the vendored MolmoMotion revision.",
    )

    evaluate = subparsers.add_parser(
        "evaluate-cohort",
        help="Open outcomes only after all 27 measurement manifests exist.",
    )
    evaluate.add_argument("panel_root")
    evaluate.add_argument("measurement_root")
    evaluate.add_argument("output_dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-cohort":
        if args.tracker_revision != ALLTRACKER_MOLMOMOTION_REVISION:
            raise ValueError(
                "AllTracker source revision differs from the protocol lock"
            )
        config = RawCameraObservationConfig(
            center_count=args.center_count,
            selected_camera_count=args.camera_count,
            alltracker_max_side=args.max_side,
        )
        runtime = AllTrackerPrefixRuntime(
            args.alltracker_source,
            args.checkpoint,
            device=args.device,
            config=config,
        )
        try:
            result = build_raw_camera_measurement_cohort(
                args.panel_root,
                args.processed_root,
                args.output_root,
                runtime,
                config=config,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            )
        finally:
            runtime.close()
    else:
        result = evaluate_raw_camera_measurement_cohort(
            args.panel_root,
            args.measurement_root,
            args.output_dir,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
