"""CLI for the open-27 covariance-gated raw-camera evaluation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.deform360_raw_camera_gated_evaluation import (
    evaluate_covariance_gated_cohort,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel_root")
    parser.add_argument("measurement_root")
    parser.add_argument("uncertainty_root")
    parser.add_argument("output_dir")
    parser.add_argument("--cycle-root")
    args = parser.parse_args()
    result = evaluate_covariance_gated_cohort(
        args.panel_root,
        args.measurement_root,
        args.uncertainty_root,
        args.output_dir,
        cycle_root=args.cycle_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
