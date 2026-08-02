"""Opt-in CLI for a larger causal AllTracker observation pool."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_raw_camera_observation import (
    ALLTRACKER_MOLMOMOTION_REVISION,
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
    build_raw_camera_measurement_cohort,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the frozen 64-identity Open27 causal measurement pool."
    )
    parser.add_argument("panel_root")
    parser.add_argument("processed_root")
    parser.add_argument("output_root")
    parser.add_argument("alltracker_source")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--camera-count", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--center-count", type=int, default=64)
    parser.add_argument(
        "--tracker-revision",
        default=ALLTRACKER_MOLMOMOTION_REVISION,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.center_count != 64:
        raise ValueError("the frozen dynamic-pool source protocol requires 64 centers")
    if args.tracker_revision != ALLTRACKER_MOLMOMOTION_REVISION:
        raise ValueError("AllTracker source revision differs from the protocol lock")
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
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
