"""Run the leakage-explicit real-case oracle-gap and variance audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.phystwin_residual_dynamics import _target_validity
from causal4d.contracts import PhysicalPosterior, TwinBelief, load_contract
from causal4d.intervention_abduction import FactualAbductionConfig
from causal4d.phystwin_backend import load_rollout_bank
from causal4d.real_oracle_audit import (
    HoldoutOracleProtocol,
    audit_oracle_bank,
    bpt_nominal_prediction,
    causal4d_posterior_prediction,
    evaluate_prediction,
    oracle_gap_report,
    protocol_dict,
    released_phystwin_prediction,
    variance_decomposition,
    verify_nested_rollout_banks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose inference, intervention-proposal, and model gaps on a real "
            "PhysTwin holdout. Oracle outputs are labeled diagnostic-only."
        )
    )
    parser.add_argument("current_bank_npz")
    parser.add_argument("expanded_bank_npz")
    parser.add_argument("twin_belief_npz")
    parser.add_argument("physical_posterior_npz")
    parser.add_argument("final_data_pickle")
    parser.add_argument("released_trajectory_pickle")
    parser.add_argument("output_json")
    parser.add_argument("output_components_csv")
    parser.add_argument("--o-plus-prefix-frames", type=int, default=6)
    parser.add_argument("--observation-scale-m", type=float, default=0.01)
    parser.add_argument("--likelihood-power", type=float, default=12.0)
    parser.add_argument("--dynamic-likelihood-weight", type=float, default=0.25)
    parser.add_argument("--degrees-of-freedom", type=float, default=4.0)
    parser.add_argument("--oracle-discrepancy-cap-m", type=float, default=0.01)
    parser.add_argument(
        "--selection-metric",
        choices=("track_error_m", "coordinate_rmse_m"),
        default="track_error_m",
    )
    return parser


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _write_component_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        flattened.append(
            {
                key: (
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
        )
    preferred = [
        "bank",
        "component_id",
        "hypothesis_id",
        "particle_id",
        "hypothesis_index",
        "particle_index",
        "contact",
        "action",
    ]
    fieldnames = preferred + sorted(
        set().union(*(row.keys() for row in flattened)) - set(preferred)
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flattened)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.o_plus_prefix_frames < 1:
        raise ValueError("--o-plus-prefix-frames must be positive")
    current_bank, current_manifest = load_rollout_bank(args.current_bank_npz)
    expanded_bank, expanded_manifest = load_rollout_bank(args.expanded_bank_npz)
    belief_artifact = load_contract(args.twin_belief_npz)
    physical_artifact = load_contract(args.physical_posterior_npz)
    if not isinstance(belief_artifact, TwinBelief):
        raise TypeError("twin_belief_npz must contain a TwinBelief")
    if not isinstance(physical_artifact, PhysicalPosterior):
        raise TypeError("physical_posterior_npz must contain a PhysicalPosterior")
    if physical_artifact.source_twin_belief_id != belief_artifact.artifact_id:
        raise ValueError("PhysicalPosterior and TwinBelief provenance disagree")
    nesting = verify_nested_rollout_banks(current_bank, expanded_bank)
    if physical_artifact.readout_trajectories_m.shape[1:] != current_bank.trajectories.shape[2:]:
        raise ValueError("PhysicalPosterior and current rollout bank shapes disagree")

    data = _load_pickle(args.final_data_pickle)
    observed = np.asarray(data["object_points"], dtype=float)
    valid = _target_validity(
        np.asarray(data["object_visibilities"], dtype=bool),
        np.asarray(data["object_motions_valid"], dtype=bool),
    )
    endpoint = belief_artifact.endpoint_frame
    frame_count = current_bank.frame_count
    node_count = current_bank.node_count
    truth = observed[endpoint : endpoint + frame_count, :node_count]
    mask = valid[endpoint : endpoint + frame_count, :node_count]
    if truth.shape != current_bank.trajectories.shape[2:]:
        raise ValueError("real observations do not cover the rollout bank")
    prefix_frame_count = args.o_plus_prefix_frames + 1
    protocol = HoldoutOracleProtocol(
        start_frame=prefix_frame_count,
        stop_frame=frame_count,
        selection_metric=args.selection_metric,
    )
    config = FactualAbductionConfig(
        observation_scale_m=args.observation_scale_m,
        likelihood_power=args.likelihood_power,
        dynamic_likelihood_weight=args.dynamic_likelihood_weight,
        degrees_of_freedom=args.degrees_of_freedom,
    )

    released = released_phystwin_prediction(
        np.asarray(_load_pickle(args.released_trajectory_pickle), dtype=float),
        endpoint_frame=endpoint,
        frame_count=frame_count,
        node_count=node_count,
    )
    bpt_prediction, bpt_weights = bpt_nominal_prediction(
        current_bank,
        belief_artifact,
        truth[:prefix_frame_count],
        prefix_mask=mask[:prefix_frame_count],
        config=config,
    )
    causal4d_prediction = causal4d_posterior_prediction(physical_artifact)
    predictors = {
        "nominal_phystwin": {
            **evaluate_prediction(released, truth, mask, protocol),
            "label_use_for_prediction": False,
        },
        "bayesian_phystwin_mixture_nominal_z": {
            **evaluate_prediction(bpt_prediction, truth, mask, protocol),
            "label_use_for_prediction": False,
            "o_plus_prefix_frames_used": args.o_plus_prefix_frames,
            "effective_component_count": float(
                1.0 / np.sum(np.square(bpt_weights))
            ),
        },
        "current_causal4d_posterior": {
            **evaluate_prediction(causal4d_prediction, truth, mask, protocol),
            "label_use_for_prediction": False,
            "o_plus_prefix_frames_used": args.o_plus_prefix_frames,
            "effective_component_count": float(
                1.0 / np.sum(np.square(physical_artifact.weights))
            ),
        },
    }

    current_oracle, current_rows = audit_oracle_bank(
        current_bank,
        belief_artifact,
        truth,
        mask,
        protocol,
        bank_name="current_bank",
        discrepancy_cap_m=args.oracle_discrepancy_cap_m,
    )
    expanded_oracle, expanded_rows = audit_oracle_bank(
        expanded_bank,
        belief_artifact,
        truth,
        mask,
        protocol,
        bank_name="expanded_bank",
        discrepancy_cap_m=args.oracle_discrepancy_cap_m,
    )
    gaps = oracle_gap_report(
        predictors["current_causal4d_posterior"],
        current_oracle,
        expanded_oracle,
    )
    variance = variance_decomposition(
        physical_artifact,
        truth,
        mask,
        protocol,
        variance_floor_m2=current_bank.variance_floor_m2,
    )

    result = {
        "schema_version": 1,
        "case": belief_artifact.context.case_id,
        "experiment": "real_oracle_gap_and_variance_audit",
        "information_boundary": {
            "protocol": protocol_dict(protocol),
            "o_plus_prefix_frame_interval": [0, prefix_frame_count],
            "holdout_labels_used_for_prediction": False,
            "holdout_labels_used_for_evaluation": True,
            "holdout_labels_used_for_oracle_selection": True,
            "oracle_outputs_deployable": False,
            "warning": (
                "Current-bank, expanded-bank, and discrepancy oracles use the "
                "same holdout for selection/fitting and evaluation. They are "
                "diagnostic ceilings, not estimators."
            ),
        },
        "causal_context": belief_artifact.context.as_dict(),
        "predictors": predictors,
        "current_bank_oracle": current_oracle,
        "expanded_bank_oracle": expanded_oracle,
        "gaps": gaps,
        "bank_nesting": nesting,
        "variance_decomposition": variance,
        "abduction_likelihood": {
            "observation_scale_m": config.observation_scale_m,
            "likelihood_power": config.likelihood_power,
            "dynamic_likelihood_weight": config.dynamic_likelihood_weight,
            "degrees_of_freedom": config.degrees_of_freedom,
        },
        "artifacts": {
            "twin_belief_id": belief_artifact.artifact_id,
            "physical_posterior_id": physical_artifact.artifact_id,
            "current_bank_manifest": current_manifest,
            "expanded_bank_manifest": expanded_manifest,
            "inputs": {
                "current_bank": _input_record(args.current_bank_npz),
                "expanded_bank": _input_record(args.expanded_bank_npz),
                "twin_belief": _input_record(args.twin_belief_npz),
                "physical_posterior": _input_record(args.physical_posterior_npz),
                "final_data": _input_record(args.final_data_pickle),
                "released_trajectory": _input_record(
                    args.released_trajectory_pickle
                ),
            },
            "component_metrics_csv": str(
                Path(args.output_components_csv).resolve()
            ),
        },
    }
    _write_component_csv(
        args.output_components_csv,
        [*current_rows, *expanded_rows],
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "component_metrics": str(
                    Path(args.output_components_csv).resolve()
                ),
                "predictors": predictors,
                "track_error_gaps": gaps["track_error_m"],
                "variance_closure": variance["all_holdout"]["closure"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
