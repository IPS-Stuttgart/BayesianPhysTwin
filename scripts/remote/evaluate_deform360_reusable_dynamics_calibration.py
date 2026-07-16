#!/usr/bin/env python3
"""Score the frozen reusable-PhysTwin method once on calibration episodes."""

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
    validate_reusable_dynamics_calibration_request,
    validate_reusable_dynamics_source_selection,
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
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--repeat-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_controller_metadata(
    root: Path,
    *,
    episode_id: int,
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Path]:
    candidates = sorted(
        (root / f"ep{episode_id}" / "controller_bundle").glob("*.meta.json")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one controller metadata artifact for episode {episode_id}"
        )
    path = candidates[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_only") is not False:
        raise ValueError("calibration controller bundle is mislabeled as source-only")
    request = payload.get("reusable_dynamics_request", {})
    expected = validate_reusable_dynamics_calibration_request(
        config,
        object_id=config["config"]["object_id"],
        episode_id=episode_id,
        operation="one-shot-scoring",
    )
    if request != expected:
        raise ValueError("calibration controller request changed")
    canonical = dict(payload)
    observed = canonical.pop("result_sha256", None)
    if observed != reusable_dynamics_result_sha256(canonical):
        raise ValueError("calibration controller metadata checksum mismatch")
    data_path = Path(payload["output_final_data"])
    if sha256_file(data_path) != payload["output_final_data_sha256"]:
        raise ValueError("calibration controller bundle checksum mismatch")
    return path, payload, data_path


def _observation_depth_qa(
    root: Path,
    *,
    episode_id: int,
    controller_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    episode_dir = root / f"ep{episode_id}" / "staged" / "episode_0000"
    observation_path = episode_dir / "reusable_dynamics_observations.json"
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    if sha256_file(observation_path) != controller_metadata["source_manifest_sha256"]:
        raise ValueError("controller bundle does not match observation artifact")
    cameras = list(observation["accepted_cameras"])
    available = [
        camera
        for camera in cameras
        if (episode_dir / camera / "rendered_urdf.h5").exists()
    ]
    return {
        "observation_path": str(observation_path),
        "observation_sha256": sha256_file(observation_path),
        "cotracker_revision": observation["implementation_revision"]["cotracker"],
        "accepted_camera_count": len(cameras),
        "gripper_depth_exclusion_available_camera_count": len(available),
        "gripper_depth_exclusion_missing_cameras": sorted(
            set(cameras) - set(available)
        ),
        "claim_boundary": (
            "the public aligned release lacks rendered URDF depth masks; any "
            "gripper depth inside the propagated object mask remains"
        ),
    }


def _load_episode(
    *,
    episode_id: int,
    data_path: Path,
    rollout_root: Path,
    label: str,
    split_path: Path,
) -> tuple[CausalTrustEpisode, Path, Path]:
    driven = (
        rollout_root
        / f"ep{episode_id}"
        / label
        / "driven"
        / "official_phystwin_smoke.json"
    )
    zero = (
        rollout_root
        / f"ep{episode_id}"
        / label
        / "zero"
        / "official_phystwin_smoke.json"
    )
    episode = load_official_phystwin_trust_episode(
        str(episode_id),
        data_path,
        driven,
        zero,
        split_path,
        evidence_scope="reusable-calibration",
    )
    return episode, driven, zero


def _score_ranges(
    episode: CausalTrustEpisode,
    predicted: np.ndarray,
    ranges: Mapping[str, list[int]],
) -> dict[str, Any]:
    return {
        name: score_causal_trust_interval(
            episode, predicted, int(bounds[0]), int(bounds[1])
        )
        for name, bounds in ranges.items()
    }


def _aggregate(
    by_episode: Mapping[str, Mapping[str, Any]],
    *,
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


def _material_point_errors(
    episode: CausalTrustEpisode,
    prediction: np.ndarray,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    values = []
    for frame in range(max(1, start), stop):
        mask = episode.visibility[frame] & episode.validity[frame]
        if not np.any(mask):
            mask = np.ones(episode.target_m.shape[1], dtype=bool)
        values.append(
            np.linalg.norm(
                prediction[frame, mask] - episode.target_m[frame, mask], axis=1
            )
        )
    return np.concatenate(values)


def _rollout_diagnostics(
    driven_path: Path,
    zero_path: Path,
    controller_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    driven = json.loads(driven_path.read_text(encoding="utf-8"))
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    strain = max(
        float(result["object_edge_strain"]["p99_absolute_relative_strain"])
        for result in (driven, zero)
    )
    initial_distances = [
        float(value)
        for group in controller_metadata["groups"]
        for value in group["initial_distances_m"]
    ]
    controller_springs = int(driven["num_controller_springs"])
    object_points = int(driven["num_original_points"])
    return {
        "controller_point_count": int(driven["num_controller_points"]),
        "controller_spring_count": controller_springs,
        "object_point_count": object_points,
        "direct_graph_support_fraction": float(controller_springs / object_points),
        "maximum_initial_association_distance_m": float(max(initial_distances)),
        "maximum_driven_or_zero_p99_relative_edge_strain": strain,
    }


def _repeat_rmse(
    reference_result_path: Path,
    repeat_result_path: Path,
) -> float:
    reference = json.loads(reference_result_path.read_text(encoding="utf-8"))
    repeat = json.loads(repeat_result_path.read_text(encoding="utf-8"))
    for result in (reference, repeat):
        if not (
            result.get("passed") is True
            and result.get("source_only_smoke") is False
            and result.get("reusable_dynamics_calibration") is True
        ):
            raise ValueError("deterministic repeat is not calibration evidence")
    for key in (
        "official_phystwin_revision",
        "data_sha256",
        "config_sha256",
        "split_sha256",
        "config_overrides",
        "support_dynamics",
        "effective_inertia",
        "realized_actuation",
        "contact_transmission",
        "frame_count",
        "num_original_points",
        "num_controller_points",
    ):
        if reference.get(key) != repeat.get(key):
            raise ValueError(f"deterministic repeat differs in {key}")
    reference_trajectory_path = reference_result_path.with_name(
        "official_phystwin_trajectory.npz"
    )
    repeat_trajectory_path = repeat_result_path.with_name(
        "official_phystwin_trajectory.npz"
    )
    for payload, path in (
        (reference, reference_trajectory_path),
        (repeat, repeat_trajectory_path),
    ):
        if payload.get("trajectory_sha256") != sha256_file(path):
            raise ValueError("deterministic-repeat trajectory checksum mismatch")
    reference_vertices = np.load(reference_trajectory_path)["vertices"]
    repeat_vertices = np.load(repeat_trajectory_path)["vertices"]
    if reference_vertices.shape != repeat_vertices.shape:
        raise ValueError("deterministic-repeat trajectory shape changed")
    return float(np.sqrt(np.mean((reference_vertices - repeat_vertices) ** 2)))


def main() -> int:
    args = _parse_args()
    config_path = (
        args.repo / "configs/causal4d_public/deform360_reusable_dynamics_081_v1.json"
    )
    config = load_reusable_dynamics_config(config_path)
    frozen = config["config"]
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    validate_reusable_dynamics_source_selection(selection, config=config)
    calibration_ids = [
        int(value) for value in frozen["episode_partition"]["independent_calibration"]
    ]
    if calibration_ids != [0, 2, 8]:
        raise ValueError("calibration episode order changed")
    frame = frozen["frame_protocol"]
    ranges = {
        "full": frame["independent_calibration_prediction_range_half_open"],
        **frame["horizon_ranges_half_open"],
    }
    trust = frozen["fixed_action_trust"]
    pooled_parameters = selection["selected_pooled_physical_parameters"]
    single_parameters = selection["selected_single_source_physical_parameters"]
    pooled_label = _parameter_label(pooled_parameters)

    controller_artifacts: dict[str, Any] = {}
    data_by_episode: dict[int, Path] = {}
    for episode_id in calibration_ids:
        meta_path, metadata, data_path = _load_controller_metadata(
            args.controller_root,
            episode_id=episode_id,
            config=config,
        )
        controller_artifacts[str(episode_id)] = {
            "metadata_path": str(meta_path),
            "metadata_sha256": sha256_file(meta_path),
            "metadata_result_sha256": metadata["result_sha256"],
            "data_path": str(data_path),
            "data_sha256": sha256_file(data_path),
            "observation_depth_qa": _observation_depth_qa(
                args.controller_root,
                episode_id=episode_id,
                controller_metadata=metadata,
            ),
        }
        data_by_episode[episode_id] = data_path

    roles = {
        "pooled": pooled_parameters,
        **{
            f"single-source-{source_episode}": parameters
            for source_episode, parameters in single_parameters.items()
        },
    }
    role_results: dict[str, Any] = {}
    loaded_by_role_episode: dict[tuple[str, int], CausalTrustEpisode] = {}
    result_paths: dict[tuple[str, int], tuple[Path, Path]] = {}
    for role, parameters in roles.items():
        label = _parameter_label(parameters)
        by_episode: dict[str, Any] = {}
        for episode_id in calibration_ids:
            episode, driven_path, zero_path = _load_episode(
                episode_id=episode_id,
                data_path=data_by_episode[episode_id],
                rollout_root=args.rollout_root,
                label=label,
                split_path=args.split_json,
            )
            loaded_by_role_episode[(role, episode_id)] = episode
            result_paths[(role, episode_id)] = (driven_path, zero_path)
            trusted = cardinality_normalized_causal_prediction(
                episode,
                base_action_response=float(trust["base_action_response"]),
                autonomous_drift=float(trust["autonomous_drift"]),
            )
            by_episode[str(episode_id)] = {
                "metrics": _score_ranges(episode, trusted, ranges),
                "effective_action_response": float(
                    trust["base_action_response"] / episode.controller_count
                ),
                "driven_result_sha256": sha256_file(driven_path),
                "zero_result_sha256": sha256_file(zero_path),
            }
        role_results[role] = {
            "physical_parameters": parameters,
            "by_episode": by_episode,
            "execution_balanced": {
                interval: _aggregate(by_episode, interval=interval)
                for interval in ranges
            },
        }

    primary_by_episode = role_results["pooled"]["by_episode"]
    raw_by_episode: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    nonconformity: dict[str, float] = {}
    repeat_rmse: dict[str, Any] = {}
    for episode_id in calibration_ids:
        episode = loaded_by_role_episode[("pooled", episode_id)]
        raw_by_episode[str(episode_id)] = {
            "metrics": _score_ranges(episode, episode.driven_m, ranges)
        }
        driven_path, zero_path = result_paths[("pooled", episode_id)]
        controller_meta = json.loads(
            Path(controller_artifacts[str(episode_id)]["metadata_path"]).read_text(
                encoding="utf-8"
            )
        )
        diagnostics[str(episode_id)] = _rollout_diagnostics(
            driven_path, zero_path, controller_meta
        )
        trusted = cardinality_normalized_causal_prediction(
            episode,
            base_action_response=float(trust["base_action_response"]),
            autonomous_drift=float(trust["autonomous_drift"]),
        )
        errors = _material_point_errors(
            episode,
            trusted,
            start=int(ranges["full"][0]),
            stop=int(ranges["full"][1]),
        )
        nonconformity[str(episode_id)] = float(
            np.quantile(errors, 0.9, method="higher")
        )
        repeated = {}
        for arm, reference_path in (("driven", driven_path), ("zero", zero_path)):
            repeat_path = (
                args.repeat_root
                / f"ep{episode_id}"
                / pooled_label
                / arm
                / "official_phystwin_smoke.json"
            )
            repeated[arm] = _repeat_rmse(reference_path, repeat_path)
        repeat_rmse[str(episode_id)] = {
            "driven_rmse_m": repeated["driven"],
            "zero_rmse_m": repeated["zero"],
            "maximum_rmse_m": max(repeated.values()),
        }

    raw_result = {
        "physical_parameters": pooled_parameters,
        "by_episode": raw_by_episode,
        "execution_balanced": {
            interval: _aggregate(raw_by_episode, interval=interval)
            for interval in ranges
        },
    }
    full = role_results["pooled"]["execution_balanced"]["full"]
    late = role_results["pooled"]["execution_balanced"]["late"]
    gates_config = frozen["calibration_gates"]
    joint_wins = 0
    maximum_degradation = -np.inf
    pooled_beats_single_count = 0
    per_episode_comparison: dict[str, Any] = {}
    single_roles = [name for name in roles if name.startswith("single-source-")]
    for episode_id in calibration_ids:
        key = str(episode_id)
        pooled_metrics = primary_by_episode[key]["metrics"]["full"]
        track_delta = (
            float(pooled_metrics["track_rmse_m"])
            - float(pooled_metrics["persistence_track_rmse_m"])
        ) / float(pooled_metrics["persistence_track_rmse_m"])
        cd_delta = (
            float(pooled_metrics["chamfer_m"])
            - float(pooled_metrics["persistence_chamfer_m"])
        ) / float(pooled_metrics["persistence_chamfer_m"])
        joint_win = track_delta < 0.0 and cd_delta < 0.0
        joint_wins += int(joint_win)
        maximum_degradation = max(maximum_degradation, track_delta, cd_delta)
        median_single_track = float(
            np.median(
                [
                    role_results[role]["by_episode"][key]["metrics"]["full"][
                        "track_rmse_m"
                    ]
                    for role in single_roles
                ]
            )
        )
        median_single_cd = float(
            np.median(
                [
                    role_results[role]["by_episode"][key]["metrics"]["full"][
                        "chamfer_m"
                    ]
                    for role in single_roles
                ]
            )
        )
        pooled_beats_single = (
            float(pooled_metrics["track_rmse_m"]) <= median_single_track
            and float(pooled_metrics["chamfer_m"]) <= median_single_cd
        )
        pooled_beats_single_count += int(pooled_beats_single)
        per_episode_comparison[key] = {
            "joint_win_vs_persistence": joint_win,
            "maximum_metric_degradation_fraction_vs_persistence": max(
                track_delta, cd_delta
            ),
            "median_single_source_track_rmse_m": median_single_track,
            "median_single_source_chamfer_m": median_single_cd,
            "pooled_matches_or_beats_median_single_source_both_metrics": (
                pooled_beats_single
            ),
        }

    conformal_radius = float(max(nonconformity.values()))
    coverage = {}
    for episode_id in calibration_ids:
        episode = loaded_by_role_episode[("pooled", episode_id)]
        prediction = cardinality_normalized_causal_prediction(
            episode,
            base_action_response=float(trust["base_action_response"]),
            autonomous_drift=float(trust["autonomous_drift"]),
        )
        errors = _material_point_errors(
            episode,
            prediction,
            start=int(ranges["full"][0]),
            stop=int(ranges["full"][1]),
        )
        coverage[str(episode_id)] = float(np.mean(errors <= conformal_radius))
    maximum_repeat_rmse = float(
        max(record["maximum_rmse_m"] for record in repeat_rmse.values())
    )
    maximum_p99_strain = float(
        max(
            record["maximum_driven_or_zero_p99_relative_edge_strain"]
            for record in diagnostics.values()
        )
    )
    gates = {
        "execution_balanced_track_transfer": (
            full["track_improvement_fraction_vs_persistence"]
            >= gates_config["minimum_execution_balanced_track_improvement_fraction"]
        ),
        "execution_balanced_cd_transfer": (
            full["chamfer_improvement_fraction_vs_persistence"]
            >= gates_config["minimum_execution_balanced_cd_improvement_fraction"]
        ),
        "minimum_joint_win_episode_count": (
            joint_wins >= gates_config["minimum_joint_win_episode_count"]
        ),
        "maximum_per_episode_degradation": (
            maximum_degradation
            <= gates_config["maximum_per_episode_degradation_fraction_per_metric"]
        ),
        "late_horizon_track_transfer": (
            late["track_improvement_fraction_vs_persistence"]
            >= gates_config["minimum_late_horizon_track_improvement_fraction"]
        ),
        "late_horizon_cd_transfer": (
            late["chamfer_improvement_fraction_vs_persistence"]
            >= gates_config["minimum_late_horizon_cd_improvement_fraction"]
        ),
        "pooled_matches_or_beats_single_source_controls": (
            pooled_beats_single_count
            >= gates_config[
                "pooled_must_match_or_beat_median_single_source_control_episode_count"
            ]
        ),
        "conformal_radius": (
            conformal_radius
            <= gates_config["conformal"]["maximum_allowed_radius_m"]
        ),
        "deterministic_replay": (
            maximum_repeat_rmse
            <= frozen["official_phystwin"]["deterministic_repeat_max_rmse_m"]
        ),
        "physical_strain": (
            maximum_p99_strain
            <= frozen["official_phystwin"]["maximum_p99_relative_edge_strain"]
        ),
    }
    passed = all(gates.values())
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableDynamicsCalibration",
        "protocol_id": frozen["protocol_id"],
        "config_sha256": config["config_sha256"],
        "source_selection_result_sha256": selection["result_sha256"],
        "source_selection_file_sha256": sha256_file(args.selection),
        "split_sha256": sha256_file(args.split_json),
        "primary_method": frozen["primary_method"],
        "fixed_action_trust": trust,
        "registered_ranges_half_open": ranges,
        "controller_artifacts": controller_artifacts,
        "pooled_trusted": role_results.pop("pooled"),
        "pooled_raw": raw_result,
        "single_source_trusted_controls": role_results,
        "per_episode_comparison": per_episode_comparison,
        "joint_win_episode_count": joint_wins,
        "maximum_per_episode_degradation_fraction": float(maximum_degradation),
        "pooled_matches_or_beats_single_source_episode_count": (
            pooled_beats_single_count
        ),
        "direct_graph_support_and_physical_diagnostics": diagnostics,
        "deterministic_repeat": {
            "by_episode": repeat_rmse,
            "maximum_rmse_m": maximum_repeat_rmse,
        },
        "conformal": {
            "nonconformity": gates_config["conformal"]["nonconformity"],
            "quantile_method": "higher",
            "by_episode_m": nonconformity,
            "order_statistic_rank": gates_config["conformal"][
                "order_statistic_rank"
            ],
            "radius_m": conformal_radius,
            "interval_width_m": 2.0 * conformal_radius,
            "calibration_coverage_by_episode": coverage,
            "coverage_claim_before_target": False,
        },
        "maximum_p99_relative_edge_strain": maximum_p99_strain,
        "gates": gates,
        "passed": passed,
        "information_boundary": {
            "source_selection_frozen_before_scoring": True,
            "calibration_outcomes_read_once": True,
            "method_or_hyperparameter_changes_allowed": False,
            "target_episode_read": False,
        },
        "claim_boundary": (
            "independent same-object calibration only; target episode 5 remains "
            "sealed unless every conjunctive gate passes; no multi-object or SOTA claim"
        ),
    }
    result["result_sha256"] = reusable_dynamics_result_sha256(result)
    if args.output.exists():
        raise FileExistsError(f"calibration result already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "pooled_full": full,
                "pooled_late": late,
                "joint_win_episode_count": joint_wins,
                "pooled_matches_or_beats_single_source_episode_count": (
                    pooled_beats_single_count
                ),
                "conformal_radius_m": conformal_radius,
                "maximum_deterministic_repeat_rmse_m": maximum_repeat_rmse,
                "maximum_p99_relative_edge_strain": maximum_p99_strain,
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
