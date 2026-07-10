"""CLI for the official PhysTwin 3D evaluation metrics."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_official_evaluation import (
    evaluate_official_phystwin_files,
    write_official_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trajectory with PhysTwin's released CD and track metrics."
    )
    parser.add_argument("trajectory")
    parser.add_argument("final_data")
    parser.add_argument("gt_track_3d")
    parser.add_argument("split")
    parser.add_argument("output_json")
    args = parser.parse_args()
    summary = evaluate_official_phystwin_files(
        args.trajectory,
        args.final_data,
        args.gt_track_3d,
        args.split,
    )
    write_official_evaluation(summary, args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
