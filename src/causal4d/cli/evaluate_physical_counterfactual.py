"""Evaluate a PhysicalPosterior without semantic evidence."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_residual_dynamics import _target_validity
from causal4d.contracts import PhysicalPosterior, load_contract
from causal4d.physical_validation import evaluate_beta_zero_physical_posterior


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a discrepancy-aware physical posterior at beta=0."
    )
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("final_data_pickle")
    parser.add_argument("output_json")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--confidence-level", type=float, default=0.90)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = load_contract(args.physical_posterior_npz)
    if not isinstance(artifact, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    with Path(args.final_data_pickle).open("rb") as handle:
        data = pickle.load(handle)
    observed = np.asarray(data["object_points"], dtype=float)
    valid = _target_validity(
        np.asarray(data["object_visibilities"], dtype=bool),
        np.asarray(data["object_motions_valid"], dtype=bool),
    )
    endpoint = artifact.context.o_minus.frame_stop - 1
    state_count = artifact.readout_trajectories_m.shape[2]
    truth = observed[endpoint:, :state_count]
    mask = valid[endpoint:, :state_count]
    result = evaluate_beta_zero_physical_posterior(
        artifact,
        truth,
        mask=mask,
        start_frame=args.start_frame,
        confidence_level=args.confidence_level,
    )
    result["case"] = artifact.context.case_id
    result["causal_context"] = artifact.context.as_dict()
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
