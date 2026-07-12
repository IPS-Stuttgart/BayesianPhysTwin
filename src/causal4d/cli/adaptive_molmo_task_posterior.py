"""Apply source-validated semantic trust with target-side OOD rejection."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from causal4d.contracts import PhysicalPosterior, load_contract, save_contract
from causal4d.molmo_adapter import load_molmo_forecasts
from causal4d.semantic_posterior import molmo_task_evidence
from causal4d.semantic_trust import (
    apply_adaptive_semantic_trust,
    load_semantic_trust_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gate MolmoMotion trust without reading the target future."
    )
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("molmo_forecast_npz")
    parser.add_argument("forecast_id")
    parser.add_argument("semantic_trust_json")
    parser.add_argument("output_task_npz")
    parser.add_argument("output_decision_json")
    parser.add_argument("--scale-m", type=float, default=0.10)
    parser.add_argument("--degrees-of-freedom", type=float, default=3.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    physical_artifact = load_contract(args.physical_posterior_npz)
    if not isinstance(physical_artifact, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    bundle = load_molmo_forecasts(args.molmo_forecast_npz)
    evidence = molmo_task_evidence(
        bundle,
        args.forecast_id,
        physical_artifact,
        scale_m=args.scale_m,
        degrees_of_freedom=args.degrees_of_freedom,
    )
    calibration = load_semantic_trust_calibration(args.semantic_trust_json)
    task, decision = apply_adaptive_semantic_trust(
        physical_artifact,
        evidence,
        calibration,
    )
    save_contract(args.output_task_npz, task)
    decision_payload = {
        **asdict(decision),
        "physical_posterior_id": physical_artifact.artifact_id,
        "task_posterior_id": task.artifact_id,
        "weights_bit_identical": bool(
            task.task_weights.tobytes() == task.physical_weights.tobytes()
        ),
    }
    decision_path = Path(args.output_decision_json)
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
