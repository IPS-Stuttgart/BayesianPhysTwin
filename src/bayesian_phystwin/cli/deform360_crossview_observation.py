"""CLI for opt-in per-camera Deform360 prefix-track supplements."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from bayesian_phystwin.deform360_crossview_observation import (
    build_crossview_track_supplement,
    load_source_raw_camera_config,
)
from bayesian_phystwin.deform360_raw_camera_observation import (
    AllTrackerPrefixRuntime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one frozen causal camera plan and retain camera-level "
            "tracks for post-open disjoint-view validation."
        )
    )
    parser.add_argument("measurement_dir")
    parser.add_argument("output_dir")
    parser.add_argument("alltracker_source")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_source_raw_camera_config(args.measurement_dir)
    runtime = AllTrackerPrefixRuntime(
        args.alltracker_source,
        args.checkpoint,
        device=args.device,
        config=config,
    )
    try:
        result = build_crossview_track_supplement(
            args.measurement_dir,
            args.output_dir,
            runtime,
        )
    finally:
        runtime.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
