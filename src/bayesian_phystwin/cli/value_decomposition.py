"""Decompose predictive value across four frozen comparison arms."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bayesian_phystwin.decisive_evidence_bootstrap import (
    DEFAULT_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
)
from bayesian_phystwin.value_decomposition import (
    DEFAULT_RAW_EQUALITY_TOLERANCE,
    analyze_bayesian_value_decomposition,
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
    parser.add_argument("--deterministic-reference", required=True)
    parser.add_argument("--guarded-reference", required=True)
    parser.add_argument("--bayesian-mean", required=True)
    parser.add_argument("--full-belief", required=True)
    parser.add_argument(
        "--raw-equality-tolerance",
        type=float,
        default=DEFAULT_RAW_EQUALITY_TOLERANCE,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=DEFAULT_BOOTSTRAP_CONFIDENCE,
    )
    parser.add_argument(
        "--maximum-input-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_INPUT_BYTES,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload, input_artifact = load_json_object(
        arguments.input_json,
        maximum_input_bytes=arguments.maximum_input_bytes,
    )
    report = analyze_bayesian_value_decomposition(
        payload,
        deterministic_reference=arguments.deterministic_reference,
        guarded_reference=arguments.guarded_reference,
        bayesian_mean=arguments.bayesian_mean,
        full_belief=arguments.full_belief,
        raw_equality_tolerance=arguments.raw_equality_tolerance,
        bootstrap_replicates=arguments.bootstrap_replicates,
        bootstrap_seed=arguments.bootstrap_seed,
        bootstrap_confidence=arguments.bootstrap_confidence,
    )
    emitted = publish_json_report(
        arguments.output_json,
        report,
        input_artifact=input_artifact,
        overwrite=arguments.overwrite,
    )
    metrics = emitted["metrics"]
    if not isinstance(metrics, dict):
        raise AssertionError("decomposition metrics changed type")
    print(
        json.dumps(
            {
                "status": "written",
                "output": str(arguments.output_json.resolve()),
                "report_id": emitted["report_id"],
                "metric_count": len(metrics),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
