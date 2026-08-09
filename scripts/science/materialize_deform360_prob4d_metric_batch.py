#!/usr/bin/env python3
"""Materialize released robot gauges for every sealed Deform360 source stream."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.deform360_prob4d_metric_batch import (
    materialize_deform360_prob4d_metric_batch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-source-inventory", type=Path, required=True)
    parser.add_argument("--production-result", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-provider-spec", type=Path, required=True)
    parser.add_argument("--metric-prior-policy", type=Path, required=True)
    parser.add_argument("--processing-revision", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = materialize_deform360_prob4d_metric_batch(
        prepared_source_inventory_path=arguments.prepared_source_inventory,
        production_result_path=arguments.production_result,
        production_root=arguments.production_root,
        prediction_root=arguments.prediction_root,
        processed_root=arguments.processed_root,
        selection_path=arguments.selection,
        visual_provider_spec_path=arguments.visual_provider_spec,
        metric_prior_policy_path=arguments.metric_prior_policy,
        expected_processing_revision=arguments.processing_revision,
        implementation_revision=arguments.implementation_revision,
        output_directory=arguments.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
