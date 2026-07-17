#!/usr/bin/env python3
"""Diagnose the frozen Deform360 source failure without opening later panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_independent_source import (
    EXPECTED_INDEPENDENT_SOURCE_EPISODES,
    load_independent_source_lock,
    sha256_file,
    validate_independent_source_prediction_seal,
)
from causal4d_public.deform360_phystwin_trust import (
    CausalTrustEpisode,
    score_causal_trust_interval,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    _require(isinstance(value, Mapping), f"{path} must contain a mapping")
    return value


def _episode_directory(root: Path, object_id: str, episode_id: int) -> Path:
    return root / f"{object_id}-ep{episode_id:04d}"


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"none of {names} exists in {directory}")


def _closure_features(openings_m: np.ndarray, frame_count: int) -> dict[str, float]:
    openings = np.asarray(openings_m, dtype=np.float64)
    if openings.ndim == 1:
        openings = openings[:, None]
    _require(
        openings.ndim == 2 and len(openings) >= frame_count,
        "robot openings must contain the prediction horizon",
    )
    low = np.quantile(openings, 0.1, axis=0)
    high = np.quantile(openings, 0.9, axis=0)
    scale = np.maximum(high - low, 1e-9)
    closure = np.clip((high - openings[:frame_count]) / scale, 0.0, 1.0)
    minimum = np.min(closure, axis=1)
    return {
        "mean_closure": float(np.mean(closure)),
        "mean_minimum_gripper_closure": float(np.mean(minimum)),
        "all_grippers_closed_fraction_075": float(np.mean(minimum >= 0.75)),
    }


def _score_prediction(
    episode: CausalTrustEpisode, prediction_m: np.ndarray, interval: str
) -> dict[str, float | int]:
    bounds = {"future": (1, 76), "late": (51, 76)}
    start, stop = bounds[interval]
    return score_causal_trust_interval(episode, prediction_m, start, stop)


def _improvements(metrics: Mapping[str, float | int]) -> dict[str, float]:
    return {
        "track_improvement_fraction": 1.0
        - float(metrics["track_rmse_m"]) / float(metrics["persistence_track_rmse_m"]),
        "chamfer_improvement_fraction": 1.0
        - float(metrics["chamfer_m"]) / float(metrics["persistence_chamfer_m"]),
    }


def _aggregate_selected(
    rows: list[dict[str, Any]], selected_alpha: Mapping[str, float], interval: str
) -> dict[str, Any]:
    chosen = [
        row["alpha_metrics"][str(selected_alpha[row["episode_key"]])][interval]
        for row in rows
    ]
    track = 1.0 - float(np.mean([item["track_rmse_m"] for item in chosen])) / float(
        np.mean([item["persistence_track_rmse_m"] for item in chosen])
    )
    chamfer = 1.0 - float(np.mean([item["chamfer_m"] for item in chosen])) / float(
        np.mean([item["persistence_chamfer_m"] for item in chosen])
    )
    wins = sum(
        item["track_rmse_m"] < item["persistence_track_rmse_m"]
        and item["chamfer_m"] < item["persistence_chamfer_m"]
        for item in chosen
    )
    return {
        "track_improvement_fraction": track,
        "chamfer_improvement_fraction": chamfer,
        "joint_win_count": int(wins),
    }


def _pooled_relative_score(
    rows: list[dict[str, Any]], selected_alpha: Mapping[str, float]
) -> float:
    metrics = [
        row["alpha_metrics"][str(selected_alpha[row["episode_key"]])]["future"]
        for row in rows
    ]
    return 0.5 * (
        float(np.mean([item["track_rmse_m"] for item in metrics]))
        / float(np.mean([item["persistence_track_rmse_m"] for item in metrics]))
        + float(np.mean([item["chamfer_m"] for item in metrics]))
        / float(np.mean([item["persistence_chamfer_m"] for item in metrics]))
    )


def _cross_fitted_closure_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    objects = sorted({row["object_id"] for row in rows})
    feature = "mean_minimum_gripper_closure"
    values = sorted({float(row["action_features"][feature]) for row in rows})
    thresholds = [-np.inf]
    thresholds.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    thresholds.append(np.inf)
    selected: dict[str, float] = {}
    folds = []
    for held_out in objects:
        train = [row for row in rows if row["object_id"] != held_out]
        best: tuple[float, float] | None = None
        for threshold in thresholds:
            policy = {
                row["episode_key"]: (
                    0.9 if row["action_features"][feature] >= threshold else 0.0
                )
                for row in train
            }
            candidate = (_pooled_relative_score(train, policy), float(threshold))
            if best is None or candidate < best:
                best = candidate
        _require(best is not None, "closure-gate fit failed")
        for row in rows:
            if row["object_id"] == held_out:
                selected[row["episode_key"]] = (
                    0.9 if row["action_features"][feature] >= best[1] else 0.0
                )
        folds.append(
            {
                "held_out_object": held_out,
                "selected_threshold": best[1],
                "training_relative_score": best[0],
            }
        )
    return {
        "feature": feature,
        "folds": folds,
        "selected_alpha_by_episode": selected,
        "future": _aggregate_selected(rows, selected, "future"),
        "late": _aggregate_selected(rows, selected, "late"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-alpha", type=float, default=1.2)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    load_independent_source_lock(args.lock)
    _require(args.maximum_alpha >= 0.9, "alpha grid must include the frozen value")
    _require(args.alpha_step > 0.0, "alpha step must be positive")
    alpha_grid = np.arange(
        0.0, args.maximum_alpha + 0.5 * args.alpha_step, args.alpha_step
    )
    _require(np.any(np.isclose(alpha_grid, 0.9)), "alpha grid omits 0.9")
    rows: list[dict[str, Any]] = []
    for object_id, episode_ids in EXPECTED_INDEPENDENT_SOURCE_EPISODES.items():
        for episode_id in episode_ids:
            directory = _episode_directory(args.result_root, object_id, episode_id)
            stage = (
                _episode_directory(args.stage_root, object_id, episode_id)
                / "episode_0000"
            )
            seal = _load_json(directory / "prediction_seal.json")
            validate_independent_source_prediction_seal(seal, verify_archive=True)
            evaluation = _load_json(directory / "evaluation.json")
            _require(
                evaluation.get("prediction_seal_sha256") == seal["result_sha256"],
                "evaluation and prediction seal differ",
            )
            archive = _first_existing(
                directory, ("prediction.npz", "sealed_prediction.npz")
            )
            with np.load(archive, allow_pickle=False) as stored:
                persistence = np.asarray(stored["persistence_m"], dtype=np.float64)
                frozen = np.asarray(stored["prediction_m"], dtype=np.float64)
                driven = np.asarray(stored["driven_readout_m"], dtype=np.float64)
                zero = np.asarray(stored["zero_action_readout_m"], dtype=np.float64)
            target_path = directory / "target_data.pkl"
            target = _load_pickle(target_path)
            episode = CausalTrustEpisode(
                episode_id=f"{object_id}/{episode_id}",
                target_m=np.asarray(target["object_points"], dtype=np.float64),
                visibility=np.asarray(target["object_visibilities"], dtype=bool),
                validity=np.asarray(target["object_motions_valid"], dtype=bool),
                driven_m=driven,
                zero_action_m=zero,
                train_stop_frame=60,
                source_data_sha256=sha256_file(target_path),
                driven_trajectory_sha256=str(seal["input_sha256"]["driven_trajectory"]),
                zero_action_trajectory_sha256=str(
                    seal["input_sha256"]["zero_action_trajectory"]
                ),
            )
            response = (frozen - persistence) / 0.9
            alpha_metrics: dict[str, Any] = {}
            for alpha in alpha_grid:
                prediction = persistence + float(alpha) * response
                alpha_metrics[str(float(alpha))] = {
                    interval: _score_prediction(episode, prediction, interval)
                    for interval in ("future", "late")
                }
            best_alpha = min(
                alpha_grid,
                key=lambda alpha: (
                    alpha_metrics[str(float(alpha))]["future"][
                        "relative_score_vs_persistence"
                    ],
                    float(alpha),
                ),
            )
            fixed_score = alpha_metrics["0.9"]["future"][
                "relative_score_vs_persistence"
            ]
            with np.load(stage / "robot" / "robot.npz", allow_pickle=False) as robot:
                openings = np.asarray(robot["openings"], dtype=np.float64)
            rows.append(
                {
                    "object_id": object_id,
                    "episode_id": episode_id,
                    "episode_key": f"{object_id}/{episode_id}",
                    "action_features": _closure_features(openings, len(persistence)),
                    "fixed_relative_score": float(fixed_score),
                    "fixed_joint_win": bool(evaluation["joint_future_win"]),
                    "best_nonnegative_alpha": float(best_alpha),
                    "best_relative_score": float(
                        alpha_metrics[str(float(best_alpha))]["future"][
                            "relative_score_vs_persistence"
                        ]
                    ),
                    "best_improvements": _improvements(
                        alpha_metrics[str(float(best_alpha))]["future"]
                    ),
                    "alpha_metrics": alpha_metrics,
                }
            )

    fixed = {row["episode_key"]: 0.9 for row in rows}
    persistence = {row["episode_key"]: 0.0 for row in rows}
    oracle_alpha = {row["episode_key"]: row["best_nonnegative_alpha"] for row in rows}
    binary_oracle = {
        row["episode_key"]: (0.9 if row["fixed_relative_score"] < 1.0 else 0.0)
        for row in rows
    }
    compact_rows = []
    for row in rows:
        compact = dict(row)
        compact.pop("alpha_metrics")
        compact_rows.append(compact)
    payload = {
        "schema_version": 1,
        "artifact_kind": "Deform360IndependentSourceFailureDiagnosis",
        "protocol_id": "deform360-graph-action-support-independent-source-v1",
        "lock_sha256": sha256_file(args.lock),
        "episode_count": len(rows),
        "fixed_predictor": {
            "future": _aggregate_selected(rows, fixed, "future"),
            "late": _aggregate_selected(rows, fixed, "late"),
        },
        "persistence": {
            "future": _aggregate_selected(rows, persistence, "future"),
            "late": _aggregate_selected(rows, persistence, "late"),
        },
        "non_deployable_diagnostics": {
            "per_episode_nonnegative_alpha_oracle": {
                "zero_alpha_episode_count": sum(
                    value == 0.0 for value in oracle_alpha.values()
                ),
                "future": _aggregate_selected(rows, oracle_alpha, "future"),
                "late": _aggregate_selected(rows, oracle_alpha, "late"),
            },
            "fixed_or_persistence_binary_oracle": {
                "physics_selected_episode_count": sum(
                    value == 0.9 for value in binary_oracle.values()
                ),
                "future": _aggregate_selected(rows, binary_oracle, "future"),
                "late": _aggregate_selected(rows, binary_oracle, "late"),
            },
        },
        "exploratory_group_cross_fit": _cross_fitted_closure_gate(rows),
        "episodes": compact_rows,
        "information_boundary": {
            "all_27_independent_source_outcomes_read": True,
            "calibration_outcomes_read": False,
            "target_initial_frames_read": False,
            "target_actions_read": False,
            "target_outcomes_read": False,
        },
        "claim_boundary": (
            "post-failure source-only diagnosis; oracle quantities are not deployable, "
            "and any next method requires a new lock over unused objects"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
