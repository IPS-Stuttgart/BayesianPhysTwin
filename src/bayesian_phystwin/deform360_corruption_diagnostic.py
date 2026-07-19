#!/usr/bin/env python3
"""Deterministic corruption stress for the open Deform360 belief panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from bayesian_phystwin.deform360_online_belief_evaluation import (
    _validate_deform360_outcome_manifest,
)
from bayesian_phystwin.phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    deterministic_farthest_point_ids,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)


CENTER_COUNT = 16
UPDATE_FRAMES = (19, 38, 57)
MINIMUM_UPDATE_CENTER_COUNT = 9
MINIMUM_HISTORY_POINT_COUNT = 3
MINIMUM_DISPERSION_THRESHOLD_M = 0.010
HISTORY_DISPERSION_QUANTILE = 0.95
HISTORY_DISPERSION_MULTIPLIER = 1.5
SEEDS = tuple(range(8))
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0

EXPECTED_SOURCE_EPISODES: dict[str, tuple[int, ...]] = {
    "002-rope-silk": (2, 5, 6, 7, 9),
    "083-blanket-cloth": (1, 2, 4, 5, 8, 9),
    "085-scarf-cloth": (3, 4, 6, 8, 9),
    "092-squirrel": (4, 5, 7, 8, 9),
    "170-spider": (0, 1, 3, 5, 8, 9),
}

CONDITIONS: dict[str, dict[str, float | str]] = {
    "clean": {"kind": "clean", "amount": 0.0},
    "gaussian_5mm": {"kind": "gaussian", "amount": 0.005},
    "mismatch_25pct": {"kind": "mismatch", "amount": 0.25},
    "mismatch_50pct": {"kind": "mismatch", "amount": 0.50},
}


def stable_rng(*parts: object) -> np.random.Generator:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    return np.random.default_rng(value)


def radial_residuals(residual_m: np.ndarray) -> np.ndarray:
    location = np.median(residual_m, axis=0)
    return np.linalg.norm(residual_m - location, axis=1)


def risk_threshold(history_radial_m: np.ndarray) -> tuple[float, float | None]:
    if len(history_radial_m) < MINIMUM_HISTORY_POINT_COUNT:
        return MINIMUM_DISPERSION_THRESHOLD_M, None
    reference = float(np.quantile(history_radial_m, HISTORY_DISPERSION_QUANTILE))
    return (
        max(
            MINIMUM_DISPERSION_THRESHOLD_M,
            HISTORY_DISPERSION_MULTIPLIER * reference,
        ),
        reference,
    )


def frozen_history_dispersion(
    prior_m: np.ndarray,
    observation_m: np.ndarray,
    available: np.ndarray,
    center_ids: np.ndarray,
) -> np.ndarray:
    """Match the locked gate: one median dispersion per frame in [0, 19)."""

    dispersion: list[float] = []
    for frame in range(UPDATE_FRAMES[0]):
        frame_available = available[frame] & np.all(
            np.isfinite(prior_m[frame, center_ids]), axis=1
        )
        if int(np.sum(frame_available)) < MINIMUM_HISTORY_POINT_COUNT:
            continue
        residual = (
            observation_m[frame, frame_available]
            - prior_m[frame, center_ids[frame_available]]
        )
        dispersion.append(float(np.median(radial_residuals(residual))))
    return np.asarray(dispersion, dtype=float)


def corrupt_center_stream(
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
    *,
    condition: str,
    seed: int,
    case_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Corrupt only selected-centre observations, never scoring targets.

    Mismatch corruption deranges the requested fraction of the currently
    available centres within each frame.  Thus every counted mismatch is an
    actual wrong identity assignment, while observation support is unchanged.
    """

    spec = CONDITIONS[condition]
    selected = np.asarray(target_m[:, center_ids], dtype=float).copy()
    available = (
        visibility[:, center_ids]
        & validity[:, center_ids]
        & np.all(np.isfinite(selected), axis=2)
    )
    rng = stable_rng("deform360-corruption-v1", case_name, condition, seed)
    corrupted_count = 0
    available_count = int(np.sum(available))
    if spec["kind"] == "gaussian":
        noise = rng.normal(0.0, float(spec["amount"]), size=selected.shape)
        selected[available] += noise[available]
        corrupted_count = available_count
    elif spec["kind"] == "mismatch":
        fraction = float(spec["amount"])
        for frame in range(len(selected)):
            frame_available = np.flatnonzero(available[frame])
            if len(frame_available) < 2:
                continue
            mismatch_count = int(np.floor(fraction * len(frame_available)))
            mismatch_count = min(len(frame_available), max(2, mismatch_count))
            destinations = rng.choice(
                frame_available, size=mismatch_count, replace=False
            )
            sources = np.roll(destinations, 1)
            selected[frame, destinations] = target_m[frame, center_ids[sources]]
            corrupted_count += mismatch_count
    elif spec["kind"] != "clean":
        raise ValueError(f"unsupported condition: {condition}")
    return (
        selected,
        available,
        {
            "available_observation_count": float(available_count),
            "corrupted_observation_count": float(corrupted_count),
            "realized_corruption_fraction": (
                0.0 if available_count == 0 else corrupted_count / available_count
            ),
        },
    )


def scored_frames(frame_count: int) -> tuple[int, ...]:
    frames: list[int] = []
    for index, update in enumerate(UPDATE_FRAMES):
        stop = UPDATE_FRAMES[index + 1] if index < 2 else frame_count
        frames.extend(range(update + 1, stop))
    return tuple(frames)


def score_hidden(
    trajectory_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
) -> dict[str, float]:
    hidden = np.ones(target_m.shape[1], dtype=bool)
    hidden[center_ids] = False
    identity: list[float] = []
    chamfer: list[float] = []
    for frame in scored_frames(len(target_m)):
        supported = (
            hidden
            & visibility[frame]
            & validity[frame]
            & np.all(np.isfinite(trajectory_m[frame]), axis=1)
            & np.all(np.isfinite(target_m[frame]), axis=1)
        )
        if not np.any(supported):
            raise ValueError(f"no hidden scoring support at frame {frame}")
        prediction = np.asarray(trajectory_m[frame, supported], dtype=float)
        truth = np.asarray(target_m[frame, supported], dtype=float)
        identity.append(float(np.sqrt(np.mean(np.square(prediction - truth)))))
        target_to_prediction = cKDTree(prediction).query(truth)[0]
        prediction_to_target = cKDTree(truth).query(prediction)[0]
        chamfer.append(
            0.5
            * (
                float(np.mean(target_to_prediction))
                + float(np.mean(prediction_to_target))
            )
        )
    return {
        "hidden_identity_rmse_m": float(np.mean(identity)),
        "hidden_symmetric_chamfer_m": float(np.mean(chamfer)),
    }


def run_filter(
    prior_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
    update_observation_m: np.ndarray,
    update_available: np.ndarray,
    history_observation_m: np.ndarray,
    history_available: np.ndarray,
) -> dict[str, Any]:
    config = RecursiveRbfBeliefConfig(local_blend=0.25)
    risk_belief = initialize_recursive_rbf_belief(
        center_ids,
        prior_m[0, center_ids],
        prior_m[0],
        config=config,
    )
    ungated_belief = initialize_recursive_rbf_belief(
        center_ids,
        prior_m[0, center_ids],
        prior_m[0],
        config=config,
    )
    risk = prior_m.copy()
    ungated = prior_m.copy()
    records: list[dict[str, Any]] = []
    rejected_exact_fallback_count = 0
    history_values = frozen_history_dispersion(
        prior_m,
        history_observation_m,
        history_available,
        center_ids,
    )
    threshold, history_p95 = risk_threshold(history_values)
    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = UPDATE_FRAMES[update_index + 1] if update_index < 2 else len(prior_m)
        current_available = update_available[update].copy()
        count = int(np.sum(current_available))
        residual = np.full((CENTER_COUNT, 3), np.nan, dtype=float)
        residual[current_available] = (
            update_observation_m[update, current_available]
            - prior_m[update, center_ids[current_available]]
        )
        current_dispersion = (
            None
            if count == 0
            else float(np.median(radial_residuals(residual[current_available])))
        )

        accepted = (
            count >= MINIMUM_UPDATE_CENTER_COUNT
            and current_dispersion is not None
            and current_dispersion <= threshold
        )

        if count:
            ungated_belief, _ = update_recursive_rbf_belief(
                ungated_belief,
                update,
                prior_m[update, center_ids],
                residual,
                current_available,
                config=config,
            )
            for frame in range(update + 1, stop):
                correction = decode_recursive_rbf_belief(
                    ungated_belief,
                    prior_m[update],
                    forecast_frames=frame - update,
                    config=config,
                ).mean_m
                ungated[frame] = (prior_m[frame] + correction).astype(
                    prior_m.dtype, copy=False
                )

        if accepted:
            risk_belief, _ = update_recursive_rbf_belief(
                risk_belief,
                update,
                prior_m[update, center_ids],
                residual,
                current_available,
                config=config,
            )
            for frame in range(update + 1, stop):
                correction = decode_recursive_rbf_belief(
                    risk_belief,
                    prior_m[update],
                    forecast_frames=frame - update,
                    config=config,
                ).mean_m
                risk[frame] = (prior_m[frame] + correction).astype(
                    prior_m.dtype, copy=False
                )
        else:
            if not np.array_equal(risk[update + 1 : stop], prior_m[update + 1 : stop]):
                raise AssertionError("risk-limited rejection was not exact fallback")
            rejected_exact_fallback_count += 1

        records.append(
            {
                "frame": update,
                "available_center_count": count,
                "current_median_radial_residual_m": current_dispersion,
                "frozen_history_p95_dispersion_m": history_p95,
                "dispersion_threshold_m": threshold,
                "frozen_history_dispersion_count": int(len(history_values)),
                "accepted": bool(accepted),
                "decision": (
                    "accepted"
                    if accepted
                    else (
                        "insufficient_support"
                        if count < MINIMUM_UPDATE_CENTER_COUNT
                        else "incoherent_residual"
                    )
                ),
            }
        )
    return {
        "scores": {
            "recursive_rbf_ungated": score_hidden(
                ungated, target_m, visibility, validity, center_ids
            ),
            "recursive_rbf_risk_limited": score_hidden(
                risk, target_m, visibility, validity, center_ids
            ),
        },
        "updates": records,
        "rejected_exact_fallback_count": rejected_exact_fallback_count,
    }


def load_case(case_dir: Path) -> dict[str, Any]:
    seal_path = case_dir / "prediction_seal.json"
    target_path = case_dir / "target_data.pkl"
    outcome_path = case_dir / "outcome.json"
    seal = json.loads(seal_path.read_text())
    outcome = json.loads(outcome_path.read_text())
    _validate_deform360_outcome_manifest(
        seal_path,
        target_path,
        seal,
        outcome,
    )
    archive_path = Path(seal["prediction_archive"]["path"])
    if not archive_path.is_file():
        archive_path = case_dir / archive_path.name
    with np.load(archive_path, allow_pickle=False) as stored:
        prior = np.asarray(stored["prediction_m"]).copy()
        frame_zero = np.asarray(stored["frame_zero_points_m"]).copy()
    with target_path.open("rb") as handle:
        data = pickle.load(handle)
    target = np.asarray(data["object_points"])
    visibility = np.asarray(data["object_visibilities"], dtype=bool)
    validity = np.asarray(data["object_motions_valid"], dtype=bool)
    if prior.shape != target.shape:
        raise ValueError(f"shape mismatch: {case_dir.name}")
    if not np.array_equal(frame_zero.astype(np.float32), target[0].astype(np.float32)):
        raise ValueError(f"frame-zero mismatch: {case_dir.name}")
    candidates = np.flatnonzero(
        visibility[0]
        & validity[0]
        & np.all(np.isfinite(prior[0]), axis=1)
        & np.all(np.isfinite(target[0]), axis=1)
    )
    centers = deterministic_farthest_point_ids(prior[0], candidates, CENTER_COUNT)
    return {
        "case": case_dir.name,
        "object_id": seal["object_id"],
        "prior": prior,
        "target": target,
        "visibility": visibility,
        "validity": validity,
        "centers": centers,
    }


def cluster_bootstrap(
    differences: dict[str, float],
    groups: dict[str, str],
) -> dict[str, float]:
    object_ids = sorted(set(groups.values()))
    group_means = np.asarray(
        [
            np.mean(
                [
                    differences[case]
                    for case, object_id in groups.items()
                    if object_id == group
                ]
            )
            for group in object_ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        bootstrap[draw] = np.mean(
            group_means[rng.integers(0, len(group_means), size=len(group_means))]
        )
    return {
        "episode_mean_difference_m": float(np.mean(list(differences.values()))),
        "object_balanced_mean_difference_m": float(np.mean(group_means)),
        "object_cluster_lower_95_m": float(np.quantile(bootstrap, 0.025)),
        "object_cluster_upper_95_m": float(np.quantile(bootstrap, 0.975)),
        "object_cluster_probability_improved": float(np.mean(bootstrap < 0.0)),
    }


def aggregate_setting(
    records: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]],
    groups: dict[str, str],
) -> dict[str, Any]:
    arms = ("recursive_rbf_ungated", "recursive_rbf_risk_limited")
    metrics = ("hidden_identity_rmse_m", "hidden_symmetric_chamfer_m")
    cases = sorted(baselines)
    summary: dict[str, Any] = {}
    for arm in arms:
        episode_values: dict[str, dict[str, float]] = {}
        for case in cases:
            case_records = [record for record in records if record["case"] == case]
            episode_values[case] = {
                metric: float(
                    np.mean([record["scores"][arm][metric] for record in case_records])
                )
                for metric in metrics
            }
        metric_summary: dict[str, Any] = {}
        for metric in metrics:
            baseline_mean = float(np.mean([baselines[case][metric] for case in cases]))
            candidate_mean = float(
                np.mean([episode_values[case][metric] for case in cases])
            )
            differences = {
                case: episode_values[case][metric] - baselines[case][metric]
                for case in cases
            }
            mean_relative_regression = {
                case: episode_values[case][metric] / baselines[case][metric] - 1.0
                for case in cases
            }
            all_record_regressions = [
                record["scores"][arm][metric] / baselines[record["case"]][metric] - 1.0
                for record in records
            ]
            metric_summary[metric] = {
                "physical_prior_mean_m": baseline_mean,
                "candidate_mean_m": candidate_mean,
                "relative_change_vs_physical_prior": candidate_mean / baseline_mean
                - 1.0,
                "episode_wins": int(sum(value < 0.0 for value in differences.values())),
                "maximum_episode_mean_relative_regression": float(
                    max(mean_relative_regression.values())
                ),
                "maximum_episode_seed_relative_regression": float(
                    max(all_record_regressions)
                ),
                "physical_object_cluster_bootstrap": cluster_bootstrap(
                    differences, groups
                ),
            }
        summary[arm] = {
            "metrics": metric_summary,
            "joint_two_metric_episode_wins": int(
                sum(
                    episode_values[case][metrics[0]] < baselines[case][metrics[0]]
                    and episode_values[case][metrics[1]] < baselines[case][metrics[1]]
                    for case in cases
                )
            ),
        }

    total_updates = len(records) * len(UPDATE_FRAMES)
    accepted_updates = sum(
        update["accepted"] for record in records for update in record["updates"]
    )
    rejection_reasons: dict[str, int] = {}
    for record in records:
        for update in record["updates"]:
            decision = update["decision"]
            rejection_reasons[decision] = rejection_reasons.get(decision, 0) + 1
    summary["coverage"] = {
        "record_count": len(records),
        "deterministic_seed_count": len(set(record["seed"] for record in records)),
        "risk_limited_accepted_update_count": int(accepted_updates),
        "total_update_count": total_updates,
        "risk_limited_acceptance_fraction": accepted_updates / total_updates,
        "risk_limited_scored_frame_correction_coverage": accepted_updates
        / total_updates,
        "ungated_update_fraction": float(
            np.mean(
                [
                    update["available_center_count"] > 0
                    for record in records
                    for update in record["updates"]
                ]
            )
        ),
        "risk_gate_decisions": rejection_reasons,
        "rejected_exact_fallback_count": int(
            sum(record["rejected_exact_fallback_count"] for record in records)
        ),
    }
    summary["realized_corruption_fraction"] = float(
        np.mean(
            [record["corruption"]["realized_corruption_fraction"] for record in records]
        )
    )
    return summary


def run(root: Path) -> dict[str, Any]:
    case_names = [
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episodes in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episodes
    ]
    cases = [load_case(root / case_name) for case_name in case_names]
    if len(cases) != 27:
        raise AssertionError("fixed panel must contain 27 cases")
    baselines = {
        case["case"]: score_hidden(
            case["prior"],
            case["target"],
            case["visibility"],
            case["validity"],
            case["centers"],
        )
        for case in cases
    }
    groups = {case["case"]: case["object_id"] for case in cases}
    detailed: dict[str, dict[str, list[dict[str, Any]]]] = {
        "scheduled_only": {},
        "full_stream": {},
    }

    for history_mode in detailed:
        conditions = (
            tuple(CONDITIONS)
            if history_mode == "scheduled_only"
            else tuple(CONDITIONS)[1:]
        )
        for condition in conditions:
            condition_records: list[dict[str, Any]] = []
            seeds = (0,) if condition == "clean" else SEEDS
            for case in cases:
                clean_observations, clean_available, _ = corrupt_center_stream(
                    case["target"],
                    case["visibility"],
                    case["validity"],
                    case["centers"],
                    condition="clean",
                    seed=0,
                    case_name=case["case"],
                )
                for seed in seeds:
                    corrupt_observations, corrupt_available, corruption = (
                        corrupt_center_stream(
                            case["target"],
                            case["visibility"],
                            case["validity"],
                            case["centers"],
                            condition=condition,
                            seed=seed,
                            case_name=case["case"],
                        )
                    )
                    result = run_filter(
                        case["prior"],
                        case["target"],
                        case["visibility"],
                        case["validity"],
                        case["centers"],
                        corrupt_observations,
                        corrupt_available,
                        (
                            clean_observations
                            if history_mode == "scheduled_only"
                            else corrupt_observations
                        ),
                        (
                            clean_available
                            if history_mode == "scheduled_only"
                            else corrupt_available
                        ),
                    )
                    condition_records.append(
                        {
                            "case": case["case"],
                            "object_id": case["object_id"],
                            "condition": condition,
                            "history_mode": history_mode,
                            "seed": seed,
                            "corruption": corruption,
                            **result,
                        }
                    )
            detailed[history_mode][condition] = condition_records

    aggregates = {
        history_mode: {
            condition: aggregate_setting(records, baselines, groups)
            for condition, records in conditions.items()
        }
        for history_mode, conditions in detailed.items()
    }
    return {
        "schema_version": 1,
        "protocol": {
            "panel": "27 already-open independent-source Deform360 episodes",
            "center_count": CENTER_COUNT,
            "center_selection": "deterministic frame-zero FPS from smallest valid ID",
            "update_frames": list(UPDATE_FRAMES),
            "local_blend": 0.25,
            "risk_gate": (
                "at least 9/16 centres and current median radial residual <= "
                "max(10 mm, 1.5 * frozen [0,19) per-frame-median-dispersion p95)"
            ),
            "scoring": (
                "all 16 centres permanently excluded from identity RMSE and both "
                "directions of symmetric Euclidean Chamfer"
            ),
            "scheduled_only": (
                "primary: only observations at frames 19/38/57 are corrupted; "
                "causal history used by the locked detector remains clean"
            ),
            "full_stream": (
                "secondary threshold-poisoning diagnostic: detector calibration "
                "history [0,19) and all scheduled updates are corrupted"
            ),
            "corruption_seeds": list(SEEDS),
            "mismatch_definition": (
                "per-frame derangement of floor(fraction * currently available) "
                "centres, with a minimum of two; support is unchanged"
            ),
            "bootstrap": (
                "10,000 equal-weight physical-object-cluster resamples over five objects"
            ),
        },
        "physical_prior_by_case": baselines,
        "aggregates": aggregates,
        "records": detailed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["aggregates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
