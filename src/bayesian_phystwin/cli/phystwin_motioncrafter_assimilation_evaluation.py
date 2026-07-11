"""CLI for locked-cohort MotionCrafter assimilation evaluation."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_motioncrafter_assimilation_evaluation import (
    evaluate_motioncrafter_assimilation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate a locked MotionCrafter assimilation cohort."
    )
    parser.add_argument("output_dir")
    parser.add_argument("summary", nargs="+")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    args = parser.parse_args()
    result = evaluate_motioncrafter_assimilation(
        args.summary,
        args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
