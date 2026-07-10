"""Diagnose whether simulator residuals are safe to interpret as track bias."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .drift_bias import RandomWalkBiasConfig, filter_random_walk_bias
from .phystwin_official_evaluation import _nearest_distances


@dataclass(frozen=True)
class PhysTwinBiasDiagnosticConfig:
    fit_end_frame: int
    train_end_frame: int
    observation_variance: float = 2.5e-5
    inlier_prior: float = 0.99
    minimum_fit_measurements: int = 3
    bias: RandomWalkBiasConfig = RandomWalkBiasConfig()


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _target_validity(visible: np.ndarray, motion_valid: np.ndarray) -> np.ndarray:
    result = np.zeros_like(visible, dtype=bool)
    result[0] = visible[0]
    result[1:] = motion_valid[: len(visible) - 1]
    return result


def _error_summary(
    raw: np.ndarray,
    corrected: np.ndarray,
    manual: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    raw_error = np.linalg.norm(raw - manual, axis=2)[mask]
    corrected_error = np.linalg.norm(corrected - manual, axis=2)[mask]
    if len(raw_error) == 0:
        return {"count": 0}
    raw_mean = float(np.mean(raw_error))
    corrected_mean = float(np.mean(corrected_error))
    percent_change = (
        None
        if raw_mean == 0.0
        else 100.0 * (corrected_mean / raw_mean - 1.0)
    )
    return {
        "count": int(len(raw_error)),
        "raw_error_m": raw_mean,
        "corrected_error_m": corrected_mean,
        "percent_change": percent_change,
    }


def diagnose_phystwin_bias_forecast(
    final_data_path: str | Path,
    baseline_trajectory_path: str | Path,
    gt_track_path: str | Path,
    *,
    config: PhysTwinBiasDiagnosticConfig,
) -> dict[str, object]:
    """Infer fit residual bias, hold it fixed, and test manual-track correction."""

    if not 1 < config.fit_end_frame < config.train_end_frame:
        raise ValueError("expected 1 < fit_end_frame < train_end_frame")
    if config.observation_variance <= 0.0:
        raise ValueError("observation_variance must be positive")
    if not 0.0 < config.inlier_prior < 1.0:
        raise ValueError("inlier_prior must lie in (0, 1)")
    data = _load_pickle(final_data_path)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    baseline = np.asarray(_load_pickle(baseline_trajectory_path), dtype=float)
    gt_track = np.asarray(_load_pickle(gt_track_path), dtype=float)
    if not config.train_end_frame < len(observed):
        raise ValueError("train_end_frame must be below the frame count")
    finite_initial = np.isfinite(gt_track[0]).all(axis=1)
    manual = gt_track[:, finite_initial]
    match_distance, matched_tracks = _nearest_distances(
        observed[0],
        gt_track[0, finite_initial],
        p=2,
    )
    valid = _target_validity(visible, motion_valid)
    inferred_bias = np.zeros((len(matched_tracks), 3), dtype=float)
    retained: list[int] = []
    for manual_index, track in enumerate(matched_tracks):
        frames = np.flatnonzero(valid[1 : config.fit_end_frame, track]) + 1
        if len(frames) < config.minimum_fit_measurements:
            continue
        retained.append(manual_index)
        for coordinate in range(3):
            residual = (
                observed[frames, track, coordinate]
                - baseline[frames, track, coordinate]
            )
            result = filter_random_walk_bias(
                np.full(len(frames), config.inlier_prior),
                residual,
                config.observation_variance,
                sequence_ids=[f"{manual_index}:{coordinate}"] * len(frames),
                time_values=frames,
                config=config.bias,
            )
            inferred_bias[manual_index, coordinate] = result.bias_mean[-1]

    retained_array = np.asarray(retained, dtype=int)
    matched_tracks = matched_tracks[retained_array]
    match_distance = match_distance[retained_array]
    manual = manual[:, retained_array]
    inferred_bias = inferred_bias[retained_array]
    raw = observed[:, matched_tracks]
    corrected = raw - inferred_bias[None]
    result: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": (
            "fit-only random-walk simulator residual bias held fixed after fit; "
            "corrected pseudo measurement equals pseudo measurement minus bias"
        ),
        "manual_track_count": int(len(matched_tracks)),
        "initial_match_max_m": float(np.max(match_distance, initial=0.0)),
        "inferred_bias_rms_m": float(
            np.sqrt(np.mean(np.sum(np.square(inferred_bias), axis=1)))
        ),
    }
    for name, start, stop in (
        ("validation", config.fit_end_frame, config.train_end_frame),
        ("test", config.train_end_frame, len(observed)),
    ):
        mask = visible[start:stop, matched_tracks] & np.isfinite(
            manual[start:stop]
        ).all(axis=2)
        result[name] = _error_summary(
            raw[start:stop],
            corrected[start:stop],
            manual[start:stop],
            mask,
        )
    return result


def write_bias_diagnostic(summary: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
