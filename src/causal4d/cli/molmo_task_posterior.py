"""Build a separate MolmoMotion-conditioned TaskPosterior."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from causal4d.contracts import PhysicalPosterior, load_contract, save_contract
from causal4d.molmo_adapter import load_molmo_forecasts
from causal4d.semantic_posterior import (
    build_task_posterior,
    molmo_task_evidence,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score MolmoMotion only against H_Q(X) and save an intention-"
            "conditioned posterior separate from the physical posterior."
        )
    )
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("molmo_forecast_npz")
    parser.add_argument("forecast_id")
    parser.add_argument("output_task_npz")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--scale-m", type=float, default=0.10)
    parser.add_argument("--degrees-of-freedom", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = load_contract(args.physical_posterior_npz)
    if not isinstance(artifact, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    bundle = load_molmo_forecasts(args.molmo_forecast_npz)
    evidence = molmo_task_evidence(
        bundle,
        args.forecast_id,
        artifact,
        scale_m=args.scale_m,
        degrees_of_freedom=args.degrees_of_freedom,
    )
    task = build_task_posterior(artifact, evidence, beta=args.beta)
    save_contract(args.output_task_npz, task)
    positive = task.task_weights > 0.0
    effective_components = 1.0 / float(np.sum(np.square(task.task_weights)))
    kl_from_physical = float(
        np.sum(
            task.task_weights[positive]
            * np.log(
                task.task_weights[positive]
                / np.maximum(task.physical_weights[positive], 1e-300)
            )
        )
    )
    print(
        json.dumps(
            {
                "beta": task.beta,
                "effective_components": effective_components,
                "forecast_id": args.forecast_id,
                "kl_task_from_physical": kl_from_physical,
                "output": str(Path(args.output_task_npz).resolve()),
                "physical_posterior_id": task.physical_posterior_id,
                "task_posterior_id": task.artifact_id,
                "weights_bit_identical": bool(
                    task.task_weights.tobytes() == task.physical_weights.tobytes()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
