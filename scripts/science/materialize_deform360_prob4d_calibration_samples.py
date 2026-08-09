#!/usr/bin/env python3
"""Materialize source-only Prob4D calibration rows from public Deform360."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_prob4d_sample_materializer import (
    Deform360Prob4DMaterializationConfig,
    materialize_deform360_prob4d_calibration_samples,
)
from bayesian_phystwin.deform360_prob4d_source_calibration import (
    load_pinned_prob4d_api,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-provider-spec", type=Path, required=True)
    parser.add_argument("--metric-prior-policy", type=Path, required=True)
    parser.add_argument("--camera-eligibility-policy", type=Path)
    parser.add_argument("--prob4d-checkout", type=Path, required=True)
    parser.add_argument("--prob4d-revision", required=True)
    parser.add_argument("--processing-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--covariance-cluster-size-pixels", type=int, default=32)
    parser.add_argument(
        "--maximum-metric-fit-correspondences", type=int, default=100_000
    )
    parser.add_argument("--maximum-point-rows-per-window", type=int, default=4_096)
    parser.add_argument("--minimum-point-rows-per-window", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    api = load_pinned_prob4d_api(
        arguments.prob4d_checkout,
        expected_revision=arguments.prob4d_revision,
    )
    result = materialize_deform360_prob4d_calibration_samples(
        plan_path=arguments.plan,
        production_result_path=arguments.production_result,
        production_root=arguments.production_root,
        prediction_root=arguments.prediction_root,
        metric_root=arguments.metric_root,
        selection_path=arguments.selection,
        visual_provider_spec_path=arguments.visual_provider_spec,
        metric_prior_policy_path=arguments.metric_prior_policy,
        camera_eligibility_policy_path=arguments.camera_eligibility_policy,
        expected_processing_revision=arguments.processing_revision,
        api=api,
        output_directory=arguments.output_dir,
        config=Deform360Prob4DMaterializationConfig(
            covariance_cluster_size_pixels=(arguments.covariance_cluster_size_pixels),
            maximum_metric_fit_correspondences=(
                arguments.maximum_metric_fit_correspondences
            ),
            maximum_point_rows_per_window=(arguments.maximum_point_rows_per_window),
            minimum_point_rows_per_window=(arguments.minimum_point_rows_per_window),
        ),
    )
    print(json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
