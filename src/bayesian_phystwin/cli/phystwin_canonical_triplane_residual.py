"""CLI for the source-only canonical triplane residual gate."""

from __future__ import annotations

import argparse
import json

from bayesian_phystwin.phystwin_canonical_triplane_residual import (
    fit_canonical_triplane_residual_source_gate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("protocol_json")
    parser.add_argument("output_dir")
    parser.add_argument("--device")
    args = parser.parse_args()
    result = fit_canonical_triplane_residual_source_gate(
        args.data_root,
        args.protocol_json,
        args.output_dir,
        device=args.device,
    )
    selected = result["selection"]["selected_candidate"]
    print(
        json.dumps(
            {
                "source_gate_passed": result["source_gate_passed"],
                "target_future_opened": result["target_future_opened"],
                "selected_blend": selected["blend"],
                "balanced_improvement": selected["balanced_improvement"],
                "aggregate_ratios_relative_to_persistence": selected[
                    "aggregate_ratios_relative_to_persistence"
                ],
                "both_win_fold_count": selected["both_win_fold_count"],
                "maximum_case_metric_ratio": selected[
                    "maximum_case_metric_ratio"
                ],
                "summary_artifact": result["summary_artifact"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
