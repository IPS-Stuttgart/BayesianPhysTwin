"""Outcome-separated pooling controls for reusable Deform360 twins."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalized_physics_score(
    track_error_m: np.ndarray,
    chamfer_m: np.ndarray,
    persistence_track_error_m: np.ndarray,
    persistence_chamfer_m: np.ndarray,
) -> np.ndarray:
    """Return an equal-weight, dimensionless score relative to persistence."""

    track = np.asarray(track_error_m, dtype=np.float64)
    chamfer = np.asarray(chamfer_m, dtype=np.float64)
    persistence_track = np.asarray(persistence_track_error_m, dtype=np.float64)
    persistence_chamfer = np.asarray(persistence_chamfer_m, dtype=np.float64)
    _require(track.shape == chamfer.shape, "candidate metric shapes differ")
    _require(track.ndim == 2, "candidate metrics must have shape (C,E)")
    _require(
        persistence_track.shape == persistence_chamfer.shape == (track.shape[1],),
        "persistence metric shapes differ",
    )
    _require(
        np.all(np.isfinite(persistence_track))
        and np.all(np.isfinite(persistence_chamfer))
        and np.all(persistence_track > 0.0)
        and np.all(persistence_chamfer > 0.0),
        "persistence errors must be finite and positive",
    )
    return 0.5 * (track / persistence_track[None] + chamfer / persistence_chamfer[None])


def _select_candidate(
    labels: Sequence[str], scores: np.ndarray, episode_indices: Sequence[int]
) -> int:
    columns = np.asarray(tuple(episode_indices), dtype=np.int64)
    _require(columns.ndim == 1 and len(columns) >= 1, "selection fold is empty")
    _require(
        np.all((0 <= columns) & (columns < scores.shape[1])),
        "selection fold contains an invalid episode",
    )
    valid = np.all(np.isfinite(scores[:, columns]), axis=1)
    _require(np.any(valid), "no candidate is finite on the selection fold")
    valid_indices = np.flatnonzero(valid)
    return int(
        min(
            valid_indices,
            key=lambda index: (
                float(np.mean(scores[index, columns])),
                str(labels[index]),
            ),
        )
    )


def fit_pooling_controls(
    candidate_labels: Sequence[str],
    fit_episode_ids: Sequence[int],
    *,
    track_error_m: np.ndarray,
    chamfer_m: np.ndarray,
    persistence_track_error_m: np.ndarray,
    persistence_chamfer_m: np.ndarray,
) -> dict[str, Any]:
    """Freeze pooled and single-fit selections without held-episode outcomes."""

    labels = tuple(str(value) for value in candidate_labels)
    episodes = tuple(int(value) for value in fit_episode_ids)
    _require(len(labels) >= 2 and len(set(labels)) == len(labels), "invalid candidates")
    _require(
        len(episodes) >= 3 and len(set(episodes)) == len(episodes),
        "at least three distinct fit episodes are required",
    )
    score = normalized_physics_score(
        track_error_m,
        chamfer_m,
        persistence_track_error_m,
        persistence_chamfer_m,
    )
    _require(
        score.shape == (len(labels), len(episodes)),
        "candidate support differs from the declared fit panel",
    )
    all_columns = tuple(range(len(episodes)))
    pooled_index = _select_candidate(labels, score, all_columns)
    single_indices = tuple(
        _select_candidate(labels, score, (episode,)) for episode in range(len(episodes))
    )
    folds = []
    for held_index, held_episode in enumerate(episodes):
        training = tuple(index for index in all_columns if index != held_index)
        pooled_loo = _select_candidate(labels, score, training)
        fold_single_indices = tuple(
            _select_candidate(labels, score, (source,)) for source in training
        )
        fold_single_scores = [
            float(score[index, held_index]) for index in fold_single_indices
        ]
        pooled_score = float(score[pooled_loo, held_index])
        folds.append(
            {
                "held_out_fit_episode_id": held_episode,
                "fit_episode_ids": [episodes[index] for index in training],
                "pooled_candidate_index": pooled_loo,
                "pooled_candidate_label": labels[pooled_loo],
                "pooled_normalized_score": pooled_score,
                "single_source_candidate_indices": list(fold_single_indices),
                "single_source_candidate_labels": [
                    labels[index] for index in fold_single_indices
                ],
                "single_source_normalized_scores": fold_single_scores,
                "single_source_median_normalized_score": float(
                    np.median(fold_single_scores)
                ),
                "pooled_beats_single_source_median": pooled_score
                < float(np.median(fold_single_scores)),
                "pooled_beats_persistence": pooled_score < 1.0,
            }
        )
    return {
        "schema_version": 1,
        "selection_metric": (
            "0.5 * future_track/persistence_track + "
            "0.5 * future_chamfer/persistence_chamfer"
        ),
        "fit_episode_ids": list(episodes),
        "candidate_labels": list(labels),
        "pooled_candidate_index": pooled_index,
        "pooled_candidate_label": labels[pooled_index],
        "single_source_candidate_indices": list(single_indices),
        "single_source_candidate_labels": [labels[index] for index in single_indices],
        "source_normalized_score_matrix": score.tolist(),
        "leave_one_fit_action_out": folds,
        "leave_one_out_persistence_win_fraction": float(
            np.mean([row["pooled_beats_persistence"] for row in folds])
        ),
        "leave_one_out_single_median_win_fraction": float(
            np.mean([row["pooled_beats_single_source_median"] for row in folds])
        ),
        "held_episode_outcomes_used": False,
    }


def evaluate_frozen_pooling_controls(
    selection: dict[str, Any],
    held_episode_ids: Sequence[int],
    *,
    track_error_m: np.ndarray,
    chamfer_m: np.ndarray,
    persistence_track_error_m: np.ndarray,
    persistence_chamfer_m: np.ndarray,
) -> dict[str, Any]:
    """Evaluate source-frozen pooled and single selections on held episodes."""

    labels = tuple(str(value) for value in selection["candidate_labels"])
    episodes = tuple(int(value) for value in held_episode_ids)
    _require(
        selection.get("held_episode_outcomes_used") is False,
        "selection artifact already used held outcomes",
    )
    _require(
        len(episodes) >= 1 and len(set(episodes)) == len(episodes),
        "held episode panel is invalid",
    )
    track = np.asarray(track_error_m, dtype=np.float64)
    chamfer = np.asarray(chamfer_m, dtype=np.float64)
    persistence_track = np.asarray(persistence_track_error_m, dtype=np.float64)
    persistence_chamfer = np.asarray(persistence_chamfer_m, dtype=np.float64)
    score = normalized_physics_score(
        track,
        chamfer,
        persistence_track,
        persistence_chamfer,
    )
    _require(
        score.shape == (len(labels), len(episodes)),
        "held candidate support differs from the frozen selection",
    )
    pooled_index = int(selection["pooled_candidate_index"])
    single_indices = tuple(
        int(value) for value in selection["single_source_candidate_indices"]
    )
    _require(
        0 <= pooled_index < len(labels)
        and len(single_indices) == len(selection["fit_episode_ids"])
        and all(0 <= value < len(labels) for value in single_indices),
        "frozen candidate indices are invalid",
    )
    rows = []
    for held_index, episode_id in enumerate(episodes):
        pooled_score = float(score[pooled_index, held_index])
        single_scores = [float(score[index, held_index]) for index in single_indices]
        single_median = float(np.median(single_scores))
        rows.append(
            {
                "held_episode_id": episode_id,
                "pooled_normalized_score": pooled_score,
                "single_source_normalized_scores": single_scores,
                "single_source_median_normalized_score": single_median,
                "pooled_beats_persistence": pooled_score < 1.0,
                "pooled_beats_single_source_median": pooled_score < single_median,
                "pooled_track_error_m": float(track[pooled_index, held_index]),
                "pooled_chamfer_m": float(chamfer[pooled_index, held_index]),
                "persistence_track_error_m": float(persistence_track[held_index]),
                "persistence_chamfer_m": float(persistence_chamfer[held_index]),
            }
        )
    return {
        "schema_version": 1,
        "pooled_candidate_index": pooled_index,
        "pooled_candidate_label": labels[pooled_index],
        "single_source_candidate_indices": list(single_indices),
        "held_episode_ids": list(episodes),
        "episodes": rows,
        "persistence_win_fraction": float(
            np.mean([row["pooled_beats_persistence"] for row in rows])
        ),
        "single_source_median_win_fraction": float(
            np.mean([row["pooled_beats_single_source_median"] for row in rows])
        ),
        "mean_pooled_normalized_score": float(
            np.mean([row["pooled_normalized_score"] for row in rows])
        ),
        "selection_refit_on_held_outcomes": False,
    }


__all__ = [
    "evaluate_frozen_pooling_controls",
    "fit_pooling_controls",
    "normalized_physics_score",
]
