#!/usr/bin/env python3
"""Materialize the target-free Deform360 Prob4D sample-admissibility plan."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_prob4d_sample_admissibility import (
    materialize_deform360_prob4d_sample_admissibility,
)
from bayesian_phystwin.deform360_prob4d_source_calibration import (
    load_pinned_prob4d_api,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--metric-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-provider-spec", type=Path, required=True)
    parser.add_argument("--metric-prior-policy", type=Path, required=True)
    parser.add_argument("--camera-eligibility-policy", type=Path, required=True)
    parser.add_argument("--sample-admissibility-policy", type=Path, required=True)
    parser.add_argument("--prob4d-checkout", type=Path, required=True)
    parser.add_argument("--prob4d-revision", required=True)
    parser.add_argument("--processing-revision", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    api = load_pinned_prob4d_api(
        arguments.prob4d_checkout,
        expected_revision=arguments.prob4d_revision,
    )
    result = materialize_deform360_prob4d_sample_admissibility(
        plan_path=arguments.plan,
        prediction_root=arguments.prediction_root,
        metric_root=arguments.metric_root,
        selection_path=arguments.selection,
        visual_provider_spec_path=arguments.visual_provider_spec,
        metric_prior_policy_path=arguments.metric_prior_policy,
        camera_eligibility_policy_path=arguments.camera_eligibility_policy,
        sample_admissibility_policy_path=arguments.sample_admissibility_policy,
        expected_processing_revision=arguments.processing_revision,
        implementation_revision=arguments.implementation_revision,
        api=api,
        output_directory=arguments.output_dir,
    )
    print(json.dumps(dict(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
