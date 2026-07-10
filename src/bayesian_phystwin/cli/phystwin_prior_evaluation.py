"""CLI for PhysTwin static/Markov cue-prior calibration."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_prior_evaluation import (
    evaluate_phystwin_prior_files,
    write_prior_evaluation,
)
from bayesian_phystwin.phystwin_refit import PhysTwinRefitReliabilityConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate static and causal cue priors on PhysTwin refit support."
    )
    parser.add_argument("final_data")
    parser.add_argument("cues")
    parser.add_argument("output_json")
    parser.add_argument("--flow-scale", type=float, default=0.005)
    parser.add_argument("--inlier-persistence", type=float, default=0.98)
    parser.add_argument("--outlier-persistence", type=float, default=0.90)
    args = parser.parse_args()
    summary = evaluate_phystwin_prior_files(
        args.final_data,
        args.cues,
        config=PhysTwinRefitReliabilityConfig(
            flow_scale=args.flow_scale,
            markov_inlier_persistence=args.inlier_persistence,
            markov_outlier_persistence=args.outlier_persistence,
        ),
    )
    write_prior_evaluation(summary, args.output_json)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
