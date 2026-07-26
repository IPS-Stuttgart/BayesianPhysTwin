"""Validate and optionally score an ObservationBeliefV1 artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from bayesian_phystwin.grouped_likelihood import (
    GroupedStudentTLikelihoodConfig,
    grouped_student_t_mixture_likelihood,
)
from bayesian_phystwin.observation_belief import load_observation_belief


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation_belief", type=Path)
    parser.add_argument("--predicted-npz", type=Path)
    parser.add_argument("--predicted-key", default="predicted_xyz_m")
    parser.add_argument("--degrees-of-freedom", type=float, default=5.0)
    parser.add_argument(
        "--outlier-covariance-multiplier", type=float, default=25.0
    )
    parser.add_argument(
        "--model-discrepancy-variance-m2", type=float, default=0.0
    )
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    belief = load_observation_belief(args.observation_belief)
    result = belief.summary()
    result["status"] = "valid"
    if args.predicted_npz is not None:
        with np.load(args.predicted_npz, allow_pickle=False) as archive:
            if args.predicted_key not in archive:
                raise ValueError(
                    f"{args.predicted_npz} has no {args.predicted_key!r} array"
                )
            predicted = np.asarray(archive[args.predicted_key])
        likelihood = grouped_student_t_mixture_likelihood(
            belief,
            predicted,
            config=GroupedStudentTLikelihoodConfig(
                degrees_of_freedom=args.degrees_of_freedom,
                outlier_covariance_multiplier=(
                    args.outlier_covariance_multiplier
                ),
                model_discrepancy_variance_m2=(
                    args.model_discrepancy_variance_m2
                ),
            ),
        )
        result["likelihood"] = {
            "total_negative_log_likelihood": (
                likelihood.total_negative_log_likelihood
            ),
            "mean_posterior_nominal_probability": (
                likelihood.mean_posterior_nominal_probability
            ),
            "group_negative_log_likelihood": (
                likelihood.negative_log_likelihood.tolist()
            ),
            "group_weighted_negative_log_likelihood": (
                likelihood.weighted_negative_log_likelihood.tolist()
            ),
            "group_posterior_nominal_probability": (
                likelihood.posterior_nominal_probability.tolist()
            ),
        }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
