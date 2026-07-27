#!/usr/bin/env python3
"""Run the frozen source-only graph-spectral discrepancy gate."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_graph_spectral_source_gate import (
    run_graph_spectral_source_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_graph_spectral_source_gate(
        args.data_root,
        args.protocol,
        args.output,
    )
    selected = result["selection"]["selected_candidate"]
    print(
        json.dumps(
            {
                "source_gate_passed": result["source_gate_passed"],
                "target_future_opened": result["target_future_opened"],
                "selected": {
                    key: selected[key]
                    for key in (
                        "rank",
                        "temporal_smoothing",
                        "local_prior_strength",
                        "blend",
                        "balanced_improvement",
                        "aggregate_ratios_relative_to_persistence",
                        "both_win_fold_count",
                        "maximum_case_metric_ratio",
                    )
                },
                "summary_artifact": result["summary_artifact"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["source_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
