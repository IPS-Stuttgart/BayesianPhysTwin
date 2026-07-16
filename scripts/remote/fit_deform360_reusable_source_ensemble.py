#!/usr/bin/env python3
"""Fit the frozen source-only Gibbs posterior over reusable PhysTwin tuples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_phystwin_trust import (
    CausalTrustEpisode,
    load_official_phystwin_trust_episode,
)
from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    validate_reusable_dynamics_source_selection,
)
from causal4d_public.deform360_reusable_ensemble import (
    fit_source_gibbs_ensemble,
    load_reusable_ensemble_config,
    reusable_ensemble_result_sha256,
    validate_source_gibbs_ensemble_artifact,
)


def _parameter_label(parameters: Mapping[str, float]) -> str:
    return (
        f"y{int(parameters['init_spring_Y'])}"
        f"-drag{int(parameters['drag_damping'])}"
        f"-dash{int(parameters['dashpot_damping'])}"
    )


def _data_path(root: Path, object_id: str, episode_id: int) -> Path:
    object_dir = (
        object_id if episode_id == 1 else f"{object_id}-ep{episode_id:04d}"
    )
    return (
        root
        / object_dir
        / (
            "final_data_sparse_controller_k1_per_group_"
            "support-positive-y-to-positive-z.pkl"
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--driven-root", type=Path, required=True)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    parent_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
    )
    ensemble_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_ensemble_081_v1.json"
    )
    parent = load_reusable_dynamics_config(parent_path)
    ensemble = load_reusable_ensemble_config(ensemble_path)
    parent_config = parent["config"]
    ensemble_config = ensemble["config"]
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    validate_reusable_dynamics_source_selection(selection, config=parent)
    expected_parent = ensemble_config["parent_reusable_dynamics"]
    if (
        selection["result_sha256"]
        != expected_parent["source_selection_result_sha256"]
        or sha256_file(args.selection)
        != expected_parent["source_selection_file_sha256"]
    ):
        raise ValueError("ensemble uses another source-selection artifact")
    if (
        sha256_file(args.split_json)
        != parent_config["official_phystwin"]["source_split_sha256"]
    ):
        raise ValueError("ensemble source split changed")

    eligible = [row for row in selection["candidate_table"] if row["eligible"]]
    if len(eligible) != 18:
        raise ValueError("ensemble candidate support changed")
    source_ids = [
        int(value) for value in ensemble_config["source_fit"]["episode_ids"]
    ]
    candidates: dict[str, dict[str, CausalTrustEpisode]] = {}
    physical_parameters: dict[str, Mapping[str, float]] = {}
    controller_springs: dict[str, int] = {}
    input_files: dict[str, Any] = {}
    for row in eligible:
        parameters = row["physical_parameters"]
        label = _parameter_label(parameters)
        if label != row["candidate_label"]:
            raise ValueError("ensemble source candidate label changed")
        physical_parameters[label] = parameters
        candidates[label] = {}
        input_files[label] = {}
        for episode_id in source_ids:
            driven_path = (
                args.driven_root
                / f"ep{episode_id}"
                / label
                / "official_phystwin_smoke.json"
            )
            zero_path = (
                args.zero_root
                / f"ep{episode_id}"
                / label
                / "official_phystwin_smoke.json"
            )
            data_path = _data_path(
                args.controller_root, parent_config["object_id"], episode_id
            )
            episode = load_official_phystwin_trust_episode(
                str(episode_id),
                data_path,
                driven_path,
                zero_path,
                args.split_json,
            )
            candidates[label][str(episode_id)] = episode
            driven = json.loads(driven_path.read_text(encoding="utf-8"))
            spring_count = int(driven["num_controller_springs"])
            previous = controller_springs.setdefault(str(episode_id), spring_count)
            if previous != spring_count:
                raise ValueError("controller support changes across physical tuples")
            input_files[label][str(episode_id)] = {
                "driven_result_sha256": sha256_file(driven_path),
                "zero_result_sha256": sha256_file(zero_path),
            }

    source = ensemble_config["source_fit"]
    trust = ensemble_config["fixed_action_trust"]
    result = fit_source_gibbs_ensemble(
        candidates,
        physical_parameters=physical_parameters,
        controller_springs=controller_springs,
        base_action_response=float(trust["base_action_response"]),
        autonomous_drift=float(trust["autonomous_drift"]),
        frame_range=tuple(int(value) for value in source["frame_range_half_open"]),
        temperature_grid=tuple(float(value) for value in source["temperature_grid"]),
        minimum_effective_candidate_count=float(
            source["minimum_effective_candidate_count"]
        ),
    )
    result.update(
        {
            "protocol_id": ensemble_config["protocol_id"],
            "config_sha256": ensemble["config_sha256"],
            "parent_config_sha256": parent["config_sha256"],
            "source_selection_result_sha256": selection["result_sha256"],
            "source_selection_file_sha256": sha256_file(args.selection),
            "split_sha256": sha256_file(args.split_json),
            "input_result_files": input_files,
        }
    )
    result["result_sha256"] = reusable_ensemble_result_sha256(result)
    validate_source_gibbs_ensemble_artifact(result)
    if args.output.exists():
        raise FileExistsError(f"source ensemble artifact exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": result["source_gate"]["passed"],
                "selected_temperature": result["selected_temperature"],
                "posterior_diagnostics": result["posterior_diagnostics"],
                "source_gate": result["source_gate"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["source_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
