#!/usr/bin/env python3
"""Check fixed action trust on source-selected reusable PhysTwin tuples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_phystwin_trust import (
    evaluate_cardinality_normalized_fixed_trust,
    load_official_phystwin_trust_episode,
)
from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    reusable_dynamics_result_sha256,
    validate_reusable_dynamics_source_selection,
)


def _parameter_label(parameters: Mapping[str, float]) -> str:
    return (
        f"y{int(parameters['init_spring_Y'])}"
        f"-drag{int(parameters['drag_damping'])}"
        f"-dash{int(parameters['dashpot_damping'])}"
    )


def _data_path(root: Path, object_id: str, episode_id: int) -> Path:
    object_dir = object_id if episode_id == 1 else f"{object_id}-ep{episode_id:04d}"
    return (
        root
        / object_dir
        / "final_data_sparse_controller_k1_per_group_support-positive-y-to-positive-z.pkl"
    )


def _aggregate_tail(by_episode: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    names = (
        "track_rmse_m",
        "chamfer_m",
        "persistence_track_rmse_m",
        "persistence_chamfer_m",
    )
    aggregate = {
        name: float(
            np.mean(
                [
                    float(record["metrics"]["untouched_tail"][name])
                    for record in by_episode.values()
                ]
            )
        )
        for name in names
    }
    aggregate["track_improvement_fraction_vs_persistence"] = float(
        (
            aggregate["persistence_track_rmse_m"]
            - aggregate["track_rmse_m"]
        )
        / aggregate["persistence_track_rmse_m"]
    )
    aggregate["chamfer_improvement_fraction_vs_persistence"] = float(
        (
            aggregate["persistence_chamfer_m"] - aggregate["chamfer_m"]
        )
        / aggregate["persistence_chamfer_m"]
    )
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--grid-root", type=Path, required=True)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
    )
    config = load_reusable_dynamics_config(config_path)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    validate_reusable_dynamics_source_selection(selection, config=config)
    frozen = config["config"]
    trust = frozen["fixed_action_trust"]
    source_ids = [
        int(value) for value in frozen["episode_partition"]["source_selection"]
    ]
    roles = {
        "pooled": selection["selected_pooled_physical_parameters"],
        **{
            f"single-source-{episode_id}": parameters
            for episode_id, parameters in selection[
                "selected_single_source_physical_parameters"
            ].items()
        },
    }
    role_results: dict[str, Any] = {}
    for role, parameters in roles.items():
        label = _parameter_label(parameters)
        by_episode: dict[str, Any] = {}
        for episode_id in source_ids:
            driven_result = (
                args.grid_root
                / f"ep{episode_id}"
                / label
                / "official_phystwin_smoke.json"
            )
            zero_result = (
                args.zero_root
                / f"ep{episode_id}"
                / label
                / "official_phystwin_smoke.json"
            )
            episode = load_official_phystwin_trust_episode(
                str(episode_id),
                _data_path(args.controller_root, frozen["object_id"], episode_id),
                driven_result,
                zero_result,
                args.split_json,
            )
            metrics = evaluate_cardinality_normalized_fixed_trust(
                episode,
                base_action_response=float(trust["base_action_response"]),
                autonomous_drift=float(trust["autonomous_drift"]),
            )
            tail = metrics["untouched_tail"]
            by_episode[str(episode_id)] = {
                "metrics": metrics,
                "joint_win": (
                    float(tail["track_rmse_m"])
                    < float(tail["persistence_track_rmse_m"])
                    and float(tail["chamfer_m"])
                    < float(tail["persistence_chamfer_m"])
                ),
                "driven_result_sha256": sha256_file(driven_result),
                "zero_result_sha256": sha256_file(zero_result),
            }
        role_results[role] = {
            "physical_parameters": parameters,
            "by_episode": by_episode,
            "execution_balanced_untouched_tail": _aggregate_tail(by_episode),
        }

    pooled = role_results["pooled"]
    pooled_tail = pooled["execution_balanced_untouched_tail"]
    pooled_episodes = pooled["by_episode"]
    gates_config = frozen["source_compatibility_gates"]
    joint_wins = sum(int(record["joint_win"]) for record in pooled_episodes.values())
    maximum_degradation = float(
        max(
            (
                float(metrics[metric]) - float(metrics[persistence_metric])
            )
            / float(metrics[persistence_metric])
            for record in pooled_episodes.values()
            for metrics in (record["metrics"]["untouched_tail"],)
            for metric, persistence_metric in (
                ("track_rmse_m", "persistence_track_rmse_m"),
                ("chamfer_m", "persistence_chamfer_m"),
            )
        )
    )
    gates = {
        "positive_execution_balanced_track_transfer": (
            pooled_tail["track_improvement_fraction_vs_persistence"]
            >= gates_config["minimum_untouched_tail_track_improvement_fraction"]
        ),
        "positive_execution_balanced_cd_transfer": (
            pooled_tail["chamfer_improvement_fraction_vs_persistence"]
            >= gates_config["minimum_untouched_tail_cd_improvement_fraction"]
        ),
        "minimum_joint_win_count": (
            joint_wins >= gates_config["minimum_joint_win_episode_count"]
        ),
        "maximum_per_episode_degradation": (
            maximum_degradation
            <= gates_config["maximum_per_episode_degradation_fraction_per_metric"]
        ),
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableDynamicsSourceTrustCompatibility",
        "protocol_id": frozen["protocol_id"],
        "config_sha256": config["config_sha256"],
        "source_selection_result_sha256": selection["result_sha256"],
        "source_selection_file_sha256": sha256_file(args.selection),
        "fixed_action_trust": trust,
        "roles": role_results,
        "pooled_joint_win_episode_count": joint_wins,
        "pooled_maximum_per_episode_degradation_fraction": maximum_degradation,
        "gates": gates,
        "passed": passed,
        "information_boundary": {
            "source_train_and_tail_outcomes_read": True,
            "calibration_episode_read": False,
            "target_episode_read": False,
            "method_or_hyperparameter_changes_allowed": False,
        },
        "claim_boundary": (
            "source compatibility only; no independent transfer or SOTA claim"
        ),
    }
    result["result_sha256"] = reusable_dynamics_result_sha256(result)
    if args.output.exists():
        raise FileExistsError(f"source trust result already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "pooled_tail": pooled_tail,
                "joint_win_episode_count": joint_wins,
                "maximum_per_episode_degradation_fraction": maximum_degradation,
                "gates": gates,
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
