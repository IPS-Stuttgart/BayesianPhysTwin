"""CLI for causal forward/backward AllTracker cycle sidecars."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_raw_camera_cycle_uncertainty import (
    RawCameraCycleUncertaintyConfig,
    build_raw_camera_cycle_uncertainty_cohort,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel_root")
    parser.add_argument("processed_root")
    parser.add_argument("measurement_root")
    parser.add_argument("uncertainty_root")
    parser.add_argument("output_root")
    parser.add_argument("alltracker_source")
    parser.add_argument("alltracker_checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source,
        args.alltracker_checkpoint,
        device=args.device,
        config=RawCameraObservationConfig(),
    )
    try:
        result = build_raw_camera_cycle_uncertainty_cohort(
            args.panel_root,
            args.processed_root,
            args.measurement_root,
            args.uncertainty_root,
            args.output_root,
            runtime,
            config=RawCameraCycleUncertaintyConfig(),
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    finally:
        runtime.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
