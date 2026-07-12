"""Infer factual actuation/contact variables from a real PhysTwin O+ prefix."""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_residual_dynamics import _target_validity
from causal4d.contracts import TwinBelief, load_contract, save_contract
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
    evaluate_factual_abduction,
)
from causal4d.phystwin_backend import load_rollout_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score observed-action PhysTwin rollouts against an early O+ prefix, "
            "infer phi/kappa_obs, and evaluate the untouched remainder."
        )
    )
    parser.add_argument("rollout_bank_npz")
    parser.add_argument("twin_belief_npz")
    parser.add_argument("final_data_pickle")
    parser.add_argument("output_factual_npz")
    parser.add_argument("output_evaluation_json")
    parser.add_argument("--o-plus-prefix-frames", type=int, default=6)
    parser.add_argument("--observation-scale-m", type=float, default=0.01)
    parser.add_argument("--likelihood-power", type=float, default=12.0)
    parser.add_argument("--dynamic-likelihood-weight", type=float, default=0.25)
    parser.add_argument("--degrees-of-freedom", type=float, default=4.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.o_plus_prefix_frames < 1:
        raise ValueError("--o-plus-prefix-frames must be positive")
    bank, manifest = load_rollout_bank(args.rollout_bank_npz)
    artifact = load_contract(args.twin_belief_npz)
    if not isinstance(artifact, TwinBelief):
        raise TypeError("twin_belief_npz must contain a TwinBelief")
    with Path(args.final_data_pickle).open("rb") as handle:
        data = pickle.load(handle)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    valid = _target_validity(visible, motion_valid)
    endpoint = artifact.endpoint_frame
    observations_from_endpoint = observed[endpoint:]
    mask_from_endpoint = valid[endpoint:]
    prefix_frame_count = args.o_plus_prefix_frames + 1
    settings = FactualAbductionConfig(
        observation_scale_m=args.observation_scale_m,
        likelihood_power=args.likelihood_power,
        dynamic_likelihood_weight=args.dynamic_likelihood_weight,
        degrees_of_freedom=args.degrees_of_freedom,
    )
    factual = abduct_factual_intervention(
        bank,
        artifact,
        observations_from_endpoint,
        prefix_frame_count=prefix_frame_count,
        observation_mask=mask_from_endpoint,
        config=settings,
    )
    evaluation = evaluate_factual_abduction(
        bank,
        artifact,
        factual,
        observations_from_endpoint,
        observation_mask=mask_from_endpoint,
        prefix_frame_count=prefix_frame_count,
        config=settings,
    )
    evaluation.update(
        {
            "case": artifact.context.case_id,
            "causal_context": artifact.context.as_dict(),
            "factual_intervention_id": factual.artifact_id,
            "rollout_bank_manifest": manifest,
            "twin_belief_id": artifact.artifact_id,
        }
    )
    save_contract(args.output_factual_npz, factual)
    result_path = Path(args.output_evaluation_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "evaluation": str(result_path.resolve()),
                "factual_intervention": str(
                    Path(args.output_factual_npz).resolve()
                ),
                "factual_intervention_id": factual.artifact_id,
                "map_hypothesis_id": evaluation["map_hypothesis_id"],
                "relative_track_error_improvement": evaluation[
                    "relative_track_error_improvement"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
