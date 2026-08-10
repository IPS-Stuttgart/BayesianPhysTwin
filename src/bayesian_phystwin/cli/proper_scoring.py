"""Convert predictive distributions into matched proper-score evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.proper_scoring import (
    DEFAULT_MAXIMUM_ARRAY_ELEMENTS,
    DEFAULT_MAXIMUM_DIMENSION,
    DEFAULT_MAXIMUM_ENERGY_PAIR_EVALUATIONS,
    DEFAULT_MAXIMUM_RECORDS,
    DEFAULT_MAXIMUM_SAMPLES_PER_FORECAST,
    DEFAULT_MAXIMUM_VARIOGRAM_EVALUATIONS,
    DEFAULT_MAXIMUM_VARIOGRAM_PAIRS,
    build_proper_score_evidence,
)

from ._json_report import (
    DEFAULT_MAXIMUM_INPUT_BYTES,
    load_json_object,
    publish_json_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument(
        "--maximum-input-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_INPUT_BYTES,
    )
    parser.add_argument(
        "--maximum-records",
        type=int,
        default=DEFAULT_MAXIMUM_RECORDS,
    )
    parser.add_argument(
        "--maximum-samples",
        type=int,
        default=DEFAULT_MAXIMUM_SAMPLES_PER_FORECAST,
    )
    parser.add_argument(
        "--maximum-dimension",
        type=int,
        default=DEFAULT_MAXIMUM_DIMENSION,
    )
    parser.add_argument(
        "--maximum-variogram-pairs",
        type=int,
        default=DEFAULT_MAXIMUM_VARIOGRAM_PAIRS,
    )
    parser.add_argument(
        "--maximum-energy-pairs",
        type=int,
        default=DEFAULT_MAXIMUM_ENERGY_PAIR_EVALUATIONS,
    )
    parser.add_argument(
        "--maximum-variogram-evaluations",
        type=int,
        default=DEFAULT_MAXIMUM_VARIOGRAM_EVALUATIONS,
    )
    parser.add_argument(
        "--maximum-array-elements",
        type=int,
        default=DEFAULT_MAXIMUM_ARRAY_ELEMENTS,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload, input_artifact = load_json_object(
        arguments.input_json,
        maximum_input_bytes=arguments.maximum_input_bytes,
    )
    evidence = build_proper_score_evidence(
        payload,
        maximum_records=arguments.maximum_records,
        maximum_samples_per_forecast=arguments.maximum_samples,
        maximum_dimension=arguments.maximum_dimension,
        maximum_variogram_pairs=arguments.maximum_variogram_pairs,
        maximum_energy_pair_evaluations=arguments.maximum_energy_pairs,
        maximum_variogram_evaluations=(
            arguments.maximum_variogram_evaluations
        ),
        maximum_array_elements=arguments.maximum_array_elements,
    )
    emitted = publish_json_report(
        arguments.output_json,
        evidence,
        input_artifact=input_artifact,
        overwrite=arguments.overwrite,
    )
    metadata = emitted["proper_scoring"]
    if not isinstance(metadata, dict):
        raise AssertionError("proper-scoring metadata changed type")
    query_metrics = metadata["query_metrics"]
    if not isinstance(query_metrics, dict):
        raise AssertionError("proper-scoring query metrics changed type")
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(arguments.output_json.resolve()),
                "evidence_id": metadata["evidence_id"],
                "query_count": len(query_metrics),
                "metric_count": sum(
                    len(value)
                    for value in query_metrics.values()
                    if isinstance(value, list)
                ),
                "record_count": len(emitted["records"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
