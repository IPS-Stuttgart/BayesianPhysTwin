"""CLI for open-source bias-aware Deform360 development."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from bayesian_phystwin.deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
    evaluate_deform360_bias_aware_development,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate guarded bias-aware belief on the already-open Deform360 "
            "source panel."
        )
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--measurement-root", required=True)
    parser.add_argument("--uncertainty-root", required=True)
    parser.add_argument("--selected-baseline-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-physical-response-mm", type=float, default=0.5)
    parser.add_argument("--minimum-observed-motion-mm", type=float, default=0.5)
    parser.add_argument("--physical-response-rank", type=int, default=4)
    parser.add_argument("--minimum-physical-agreement-gain", type=float, default=0.40)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Deform360BiasAwareDevelopmentConfig(
        minimum_physical_response_m=args.minimum_physical_response_mm / 1000.0,
        minimum_observed_motion_m=args.minimum_observed_motion_mm / 1000.0,
        physical_response_rank=args.physical_response_rank,
        minimum_physical_agreement_gain=args.minimum_physical_agreement_gain,
    )
    result = evaluate_deform360_bias_aware_development(
        args.source_root,
        args.measurement_root,
        args.uncertainty_root,
        args.selected_baseline_root,
        args.output,
        config=config,
    )
    compact = {
        "aggregate": result["aggregate"],
        "comparisons_to_selected_raw_baseline": result[
            "comparisons_to_selected_raw_baseline"
        ],
        "source_transfer_gates": result["source_transfer_gates"],
        "larger_preregistered_run_justified": result[
            "larger_preregistered_run_justified"
        ],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
