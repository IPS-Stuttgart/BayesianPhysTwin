#!/usr/bin/env python3
"""Validate or fit source-only Prob4D calibration from public Deform360."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesian_phystwin.deform360_prob4d_source_calibration import (
    fit_and_publish_deform360_prob4d_source_calibration,
    load_deform360_prob4d_calibration_samples,
    load_pinned_prob4d_api,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "fit"):
        stage = subparsers.add_parser(command)
        stage.add_argument("--samples", type=Path, required=True)
        stage.add_argument("--selection", type=Path, required=True)
        stage.add_argument("--visual-provider-spec", type=Path, required=True)
        stage.add_argument("--metric-prior-policy", type=Path, required=True)
        stage.add_argument("--prediction-root", type=Path, required=True)
        if command == "fit":
            stage.add_argument("--prob4d-checkout", type=Path, required=True)
            stage.add_argument("--output-dir", type=Path, required=True)
            stage.add_argument("--trim-quantile", type=float, default=0.99)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    samples = load_deform360_prob4d_calibration_samples(
        arguments.samples,
        selection_path=arguments.selection,
        visual_provider_spec_path=arguments.visual_provider_spec,
        metric_prior_policy_path=arguments.metric_prior_policy,
        prediction_root=arguments.prediction_root,
    )
    result: Mapping[str, Any]
    if arguments.command == "validate":
        result = {
            "bundle_id": samples.bundle_id,
            "protocol_id": samples.protocol_id,
            "physical_object_count": len(samples.object_ids),
            "prediction_manifest_count": sum(
                len(paths) for paths in samples.prediction_manifest_paths
            ),
            "point_row_count": len(samples.arrays["point_errors_m"]),
            "gauge_row_count": len(samples.arrays["gauge_errors"]),
            "confirmation_payloads_opened": False,
            "target_outcomes_used": False,
        }
    else:
        api = load_pinned_prob4d_api(
            arguments.prob4d_checkout,
            expected_revision=samples.prob4d_revision,
        )
        result = fit_and_publish_deform360_prob4d_source_calibration(
            samples,
            api=api,
            output_directory=arguments.output_dir,
            trim_quantile=arguments.trim_quantile,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
