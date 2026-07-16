#!/usr/bin/env python3
"""Evaluate trust-aligned point-MAP controls on opened calibration data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_dense_source import sha256_file
from causal4d_public.deform360_phystwin_trust import (
    CausalTrustEpisode,
    cardinality_normalized_causal_prediction,
    load_official_phystwin_trust_episode,
    score_causal_trust_interval,
)
from causal4d_public.deform360_reusable_dynamics import (
    load_reusable_dynamics_config,
    reusable_dynamics_result_sha256,
)
from causal4d_public.deform360_reusable_ensemble import (
    load_reusable_ensemble_config,
    reusable_ensemble_result_sha256,
    trusted_candidate_prediction,
)


def _parameter_label(parameters: Mapping[str, float]) -> str:
    return (
        f"y{int(parameters['init_spring_Y'])}"
        f"-drag{int(parameters['drag_damping'])}"
        f"-dash{int(parameters['dashpot_damping'])}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--point-map", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _controller_data(root: Path, episode_id: int) -> Path:
    candidates = sorted(
        (root / f"ep{episode_id}" / "controller_bundle").glob("*.meta.json")
    )
    if len(candidates) != 1:
        raise ValueError("exploratory episode needs one controller bundle")
    metadata = json.loads(candidates[0].read_text(encoding="utf-8"))
    data_path = Path(metadata["output_final_data"])
    if sha256_file(data_path) != metadata["output_final_data_sha256"]:
        raise ValueError("exploratory controller data checksum changed")
    return data_path


def _load_episode(
    *,
    episode_id: int,
    label: str,
    data_path: Path,
    rollout_root: Path,
    split_path: Path,
) -> tuple[CausalTrustEpisode, dict[str, Any], Path, Path]:
    driven_path = (
        rollout_root
        / f"ep{episode_id}"
        / label
        / "driven"
        / "official_phystwin_smoke.json"
    )
    zero_path = (
        rollout_root
        / f"ep{episode_id}"
        / label
        / "zero"
        / "official_phystwin_smoke.json"
    )
    episode = load_official_phystwin_trust_episode(
        str(episode_id),
        data_path,
        driven_path,
        zero_path,
        split_path,
        evidence_scope="reusable-calibration",
    )
    driven = json.loads(driven_path.read_text(encoding="utf-8"))
    return episode, driven, driven_path, zero_path


def _score_ranges(
    episode: CausalTrustEpisode,
    prediction: np.ndarray,
    ranges: Mapping[str, list[int]],
) -> dict[str, Any]:
    return {
        name: score_causal_trust_interval(
            episode, prediction, int(bounds[0]), int(bounds[1])
        )
        for name, bounds in ranges.items()
    }


def _aggregate(
    by_episode: Mapping[str, Mapping[str, Any]],
    interval: str,
) -> dict[str, float]:
    names = (
        "track_rmse_m",
        "chamfer_m",
        "persistence_track_rmse_m",
        "persistence_chamfer_m",
    )
    result = {
        name: float(
            np.mean(
                [
                    float(record["metrics"][interval][name])
                    for record in by_episode.values()
                ]
            )
        )
        for name in names
    }
    result["track_improvement_fraction_vs_persistence"] = float(
        (result["persistence_track_rmse_m"] - result["track_rmse_m"])
        / result["persistence_track_rmse_m"]
    )
    result["chamfer_improvement_fraction_vs_persistence"] = float(
        (result["persistence_chamfer_m"] - result["chamfer_m"])
        / result["persistence_chamfer_m"]
    )
    return result


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
    point_map = json.loads(args.point_map.read_text(encoding="utf-8"))
    if not (
        point_map.get("artifact_kind")
        == "Deform360ReusableTwinSourceTrustedPointMapControl"
        and point_map.get("config_sha256") == ensemble["config_sha256"]
        and point_map.get("result_sha256")
        == reusable_ensemble_result_sha256(point_map)
    ):
        raise ValueError("trusted point-MAP control is invalid")
    parent_calibration = json.loads(
        args.parent_calibration.read_text(encoding="utf-8")
    )
    if not (
        parent_calibration.get("artifact_kind")
        == "Deform360ReusableDynamicsCalibration"
        and parent_calibration.get("result_sha256")
        == reusable_dynamics_result_sha256(parent_calibration)
        and parent_calibration.get("passed") is False
    ):
        raise ValueError("parent negative calibration artifact is invalid")
    frame = parent["config"]["frame_protocol"]
    ranges = {
        "full": frame["independent_calibration_prediction_range_half_open"],
        **frame["horizon_ranges_half_open"],
    }
    trust = ensemble["config"]["fixed_action_trust"]
    episode_ids = [0, 2, 8]
    parameters_by_role = {
        "pooled": point_map["selected_pooled_physical_parameters"],
        **{
            f"single-source-{episode_id}": parameters
            for episode_id, parameters in point_map[
                "selected_single_source_physical_parameters"
            ].items()
        },
    }
    role_results: dict[str, Any] = {}
    for role, parameters in parameters_by_role.items():
        label = _parameter_label(parameters)
        by_episode: dict[str, Any] = {}
        for episode_id in episode_ids:
            episode, driven, driven_path, zero_path = _load_episode(
                episode_id=episode_id,
                label=label,
                data_path=_controller_data(args.controller_root, episode_id),
                rollout_root=args.rollout_root,
                split_path=args.split_json,
            )
            spring_count = int(driven["num_controller_springs"])
            supported = trusted_candidate_prediction(
                episode,
                base_action_response=float(trust["base_action_response"]),
                autonomous_drift=float(trust["autonomous_drift"]),
                controller_spring_count=spring_count,
            )
            commanded = cardinality_normalized_causal_prediction(
                episode,
                base_action_response=float(trust["base_action_response"]),
                autonomous_drift=float(trust["autonomous_drift"]),
            )
            by_episode[str(episode_id)] = {
                "metrics": _score_ranges(episode, supported, ranges),
                "commanded_count_metrics": _score_ranges(
                    episode, commanded, ranges
                ),
                "controller_count": episode.controller_count,
                "controller_spring_count": spring_count,
                "driven_result_sha256": sha256_file(driven_path),
                "zero_result_sha256": sha256_file(zero_path),
            }
        role_results[role] = {
            "physical_parameters": parameters,
            "by_episode": by_episode,
            "execution_balanced": {
                interval: _aggregate(by_episode, interval) for interval in ranges
            },
        }

    pooled = role_results["pooled"]
    pooled_commanded_by_episode = {
        episode_id: {"metrics": record["commanded_count_metrics"]}
        for episode_id, record in pooled["by_episode"].items()
    }
    pooled_commanded_aggregate = {
        interval: _aggregate(pooled_commanded_by_episode, interval)
        for interval in ranges
    }
    single_roles = [role for role in role_results if role.startswith("single-source-")]
    pooled_vs_single = {}
    for episode_id in episode_ids:
        key = str(episode_id)
        pooled_metrics = pooled["by_episode"][key]["metrics"]["full"]
        median_track = float(
            np.median(
                [
                    role_results[role]["by_episode"][key]["metrics"]["full"][
                        "track_rmse_m"
                    ]
                    for role in single_roles
                ]
            )
        )
        median_cd = float(
            np.median(
                [
                    role_results[role]["by_episode"][key]["metrics"]["full"][
                        "chamfer_m"
                    ]
                    for role in single_roles
                ]
            )
        )
        pooled_vs_single[key] = {
            "median_single_source_track_rmse_m": median_track,
            "median_single_source_chamfer_m": median_cd,
            "pooled_matches_or_beats_both": (
                float(pooled_metrics["track_rmse_m"]) <= median_track
                and float(pooled_metrics["chamfer_m"]) <= median_cd
            ),
        }
    old_full = parent_calibration["pooled_trusted"]["execution_balanced"]["full"]
    new_full = pooled["execution_balanced"]["full"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTrustedMapExploratoryEvaluation",
        "ensemble_config_sha256": ensemble["config_sha256"],
        "parent_config_sha256": parent["config_sha256"],
        "point_map_result_sha256": point_map["result_sha256"],
        "parent_calibration_result_sha256": parent_calibration["result_sha256"],
        "registered_ranges_half_open": ranges,
        "pooled_supported_count": pooled,
        "pooled_commanded_count_execution_balanced": pooled_commanded_aggregate,
        "single_source_supported_count_controls": {
            role: record for role, record in role_results.items() if role != "pooled"
        },
        "pooled_vs_single_source": pooled_vs_single,
        "pooled_matches_or_beats_single_source_episode_count": sum(
            int(record["pooled_matches_or_beats_both"])
            for record in pooled_vs_single.values()
        ),
        "comparison_to_frozen_parent": {
            "parent_full": old_full,
            "trusted_map_full": new_full,
            "track_improvement_fraction_over_parent": float(
                (old_full["track_rmse_m"] - new_full["track_rmse_m"])
                / old_full["track_rmse_m"]
            ),
            "chamfer_improvement_fraction_over_parent": float(
                (old_full["chamfer_m"] - new_full["chamfer_m"])
                / old_full["chamfer_m"]
            ),
        },
        "information_boundary": {
            "previously_opened_calibration_used_for_exploratory_evaluation": True,
            "confirmatory_claim_allowed": False,
            "sealed_target_episode_read": False,
        },
        "claim_boundary": (
            "post hoc mechanism-development result; a fresh multi-object panel "
            "is required before reusable-twin or SOTA claims"
        ),
    }
    result["result_sha256"] = reusable_ensemble_result_sha256(result)
    if args.output.exists():
        raise FileExistsError(f"exploratory evaluation exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pooled_full": new_full,
                "comparison_to_frozen_parent": result[
                    "comparison_to_frozen_parent"
                ],
                "pooled_matches_or_beats_single_source_episode_count": result[
                    "pooled_matches_or_beats_single_source_episode_count"
                ],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
