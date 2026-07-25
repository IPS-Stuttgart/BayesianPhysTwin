"""CLI for the open-source Deform360 candidate-metric audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_phystwin.deform360_candidate_metric_sensitivity import (
    evaluate_open_source_candidate_metric_sensitivity,
    write_candidate_metric_sensitivity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute explicit candidate Deform360 metrics on the already-open "
            "source panel. The output is never official evaluator parity."
        )
    )
    parser.add_argument("prediction_root", type=Path)
    parser.add_argument("source_panel_root", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate_open_source_candidate_metric_sensitivity(
        args.prediction_root, args.source_panel_root
    )
    write_candidate_metric_sensitivity(result, args.output_json)
    compact = {
        "claim_label": result["claim_label"],
        "case_count": result["case_count"],
        "physical_object_count": result["physical_object_count"],
        "primary_method": result["primary_method"],
        "metric_robustness_gate": result["metric_robustness_gate"],
        "decision": result["decision"],
        "result_sha256": result["result_sha256"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
