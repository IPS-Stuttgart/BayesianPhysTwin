"""CLI for the post-hoc PhysTwin future-horizon analysis."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_horizon_analysis import (
    run_phystwin_horizon_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze early, middle, and late PhysTwin future frames."
    )
    parser.add_argument("data_root")
    parser.add_argument("action_run_dir")
    parser.add_argument("persistent_run_dir")
    parser.add_argument("output_json")
    args = parser.parse_args()
    result = run_phystwin_horizon_analysis(
        args.data_root,
        args.action_run_dir,
        args.persistent_run_dir,
        args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
