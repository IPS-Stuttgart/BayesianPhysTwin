"""Build future-blind PhysTwin source-family input artifacts."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_prefix_artifact import (
    build_phystwin_prefix_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a prefix-only PhysTwin fitting payload."
    )
    parser.add_argument("final_data")
    parser.add_argument("gt_track_3d")
    parser.add_argument("released_trajectory")
    parser.add_argument("output_dir")
    parser.add_argument("--prefix-end-frame", type=int, required=True)
    args = parser.parse_args()
    result = build_phystwin_prefix_artifact(
        args.final_data,
        args.gt_track_3d,
        args.released_trajectory,
        args.output_dir,
        prefix_end_frame=args.prefix_end_frame,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
