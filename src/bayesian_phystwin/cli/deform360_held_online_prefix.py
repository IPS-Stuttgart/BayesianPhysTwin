"""Build the sealed outcome-blind online prefix stage for one held case."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_held_online_prefix import (
    run_held_online_prefix_case,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
    RawCameraObservationConfig,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--frame-zero-manifest", required=True)
    parser.add_argument("--physical-prior-seal", required=True)
    parser.add_argument("--prefix-authorization", required=True)
    parser.add_argument("--aligned-episode-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument(
        "--role", choices=("calibration", "confirmation"), default="calibration"
    )
    parser.add_argument("--alltracker-source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RawCameraObservationConfig()
    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source,
        args.checkpoint,
        device=args.device,
        config=config,
    )
    try:
        result = run_held_online_prefix_case(
            args.lock,
            args.frame_zero_manifest,
            args.physical_prior_seal,
            args.prefix_authorization,
            args.aligned_episode_dir,
            args.output_dir,
            runtime,
            case_name=args.case_name,
            role=args.role,
            observation_config=config,
        )
    finally:
        runtime.close()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
