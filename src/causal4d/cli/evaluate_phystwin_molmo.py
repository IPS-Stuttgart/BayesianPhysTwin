"""Evaluate MolmoMotion evidence over a saved PhysTwin rollout bank."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from causal4d.molmo_adapter import load_molmo_forecasts
from causal4d.phystwin_backend import load_rollout_bank
from causal4d.phystwin_evaluation import (
    evaluate_phystwin_rollout_bank,
    write_phystwin_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare physics, MolmoMotion, online, and combined forecasts."
    )
    parser.add_argument("rollout_bank")
    parser.add_argument("final_data")
    parser.add_argument("molmo_forecasts")
    parser.add_argument("output_json")
    parser.add_argument("--observation-fraction", type=float, default=0.20)
    parser.add_argument("--observation-scale-m", type=float, default=0.006)
    parser.add_argument("--observation-likelihood-power", type=float, default=8.0)
    parser.add_argument("--dynamic-likelihood-weight", type=float, default=0.5)
    parser.add_argument("--molmo-scale-m", type=float, default=0.10)
    parser.add_argument("--molmo-likelihood-weight", type=float, default=12.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bank, manifest = load_rollout_bank(args.rollout_bank)
    molmo = load_molmo_forecasts(args.molmo_forecasts)
    result = evaluate_phystwin_rollout_bank(
        bank,
        manifest,
        args.final_data,
        molmo,
        observation_fraction=args.observation_fraction,
        observation_scale_m=args.observation_scale_m,
        observation_likelihood_power=args.observation_likelihood_power,
        dynamic_likelihood_weight=args.dynamic_likelihood_weight,
        molmo_scale_m=args.molmo_scale_m,
        molmo_likelihood_weight=args.molmo_likelihood_weight,
    )
    write_phystwin_evaluation(args.output_json, result)
    print(json.dumps(result["headline"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

