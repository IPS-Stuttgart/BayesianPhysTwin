"""Causal model-discrepancy calibration for saved PhysTwin profile artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .phystwin_profile import (
    causal_model_discrepancy_variance,
    predictive_observation_calibration,
)


@dataclass(frozen=True)
class PhysTwinDiscrepancyConfig:
    fit_end_frame: int
    test_start_frame: int
    observation_variance: float = 2.5e-5
    decay_candidates: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99)


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _target_frame_validity(
    visible: np.ndarray,
    motion_valid: np.ndarray,
) -> np.ndarray:
    frame_count, track_count = visible.shape
    if motion_valid.shape not in {
        (frame_count, track_count),
        (frame_count - 1, track_count),
    }:
        raise ValueError("motion_valid has an incompatible shape")
    target = np.zeros_like(visible)
    target[0] = visible[0]
    target[1:] = motion_valid[: frame_count - 1]
    return target


def _split_mask(
    base_mask: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    result = np.zeros_like(base_mask)
    result[start:stop] = base_mask[start:stop]
    return result


def _calibration_by_split(
    observed: np.ndarray,
    mean: np.ndarray,
    epistemic: np.ndarray,
    valid: np.ndarray,
    discrepancy: float | np.ndarray,
    config: PhysTwinDiscrepancyConfig,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name, start, stop in (
        ("fit", 1, config.fit_end_frame),
        ("validation", config.fit_end_frame, config.test_start_frame),
        ("test", config.test_start_frame, len(observed)),
    ):
        result[name] = predictive_observation_calibration(
            observed,
            mean,
            epistemic,
            _split_mask(valid, start, stop),
            observation_variance=config.observation_variance,
            model_discrepancy_variance=discrepancy,
        )
    return result


def _calibrate_candidate(
    observed: np.ndarray,
    mean: np.ndarray,
    epistemic: np.ndarray,
    valid: np.ndarray,
    config: PhysTwinDiscrepancyConfig,
) -> dict[str, object]:
    static = _calibration_by_split(
        observed,
        mean,
        epistemic,
        valid,
        0.0,
        config,
    )
    candidates: list[dict[str, object]] = []
    selected: tuple[float, float, np.ndarray, dict[str, dict[str, float | int]]] | None = None
    for decay in config.decay_candidates:
        discrepancy = causal_model_discrepancy_variance(
            observed,
            mean,
            epistemic,
            valid,
            observation_variance=config.observation_variance,
            decay=decay,
        )
        calibration = _calibration_by_split(
            observed,
            mean,
            epistemic,
            valid,
            discrepancy,
            config,
        )
        validation_nees = float(
            calibration["validation"]["mean_nees_per_coordinate"]
        )
        score = abs(float(np.log(validation_nees)))
        candidates.append(
            {
                "decay": decay,
                "validation": calibration["validation"],
            }
        )
        ranking = (score, -decay)
        if selected is None or ranking < (selected[0], selected[1]):
            selected = (score, -decay, discrepancy, calibration)
    assert selected is not None
    _, negative_decay, discrepancy, calibration = selected
    discrepancy_stats: dict[str, dict[str, float]] = {}
    for name, start, stop in (
        ("fit", 1, config.fit_end_frame),
        ("validation", config.fit_end_frame, config.test_start_frame),
        ("test", config.test_start_frame, len(observed)),
    ):
        values = discrepancy[start:stop]
        discrepancy_stats[name] = {
            "mean_variance": float(np.mean(values)),
            "root_mean_variance_m": float(np.sqrt(np.mean(values))),
        }
    return {
        "static": static,
        "causal_one_step": {
            "warning": (
                "Frame t uses residuals only through t-1; this is online one-step "
                "calibration, not open-loop future uncertainty."
            ),
            "selected_decay": -negative_decay,
            "selection_metric": "absolute log validation NEES per coordinate",
            "candidates": candidates,
            "calibration": calibration,
            "discrepancy": discrepancy_stats,
        },
    }


def calibrate_phystwin_profile_discrepancy(
    final_data_path: str | Path,
    profile_path: str | Path,
    *,
    config: PhysTwinDiscrepancyConfig,
    reference_trajectory_path: str | Path | None = None,
) -> dict[str, object]:
    """Calibrate saved posterior and optional reference trajectories causally."""

    if not 1 < config.fit_end_frame < config.test_start_frame:
        raise ValueError("frame split must satisfy 1 < fit_end < test_start")
    if config.observation_variance <= 0.0:
        raise ValueError("observation_variance must be positive")
    if not config.decay_candidates:
        raise ValueError("at least one decay candidate is required")
    if any(not 0.0 <= value < 1.0 for value in config.decay_candidates):
        raise ValueError("decay candidates must be in [0, 1)")

    data = _load_pickle(final_data_path)
    observed = np.asarray(data["object_points"], dtype=float)
    visible = np.asarray(data["object_visibilities"], dtype=bool)
    motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
    if not config.test_start_frame < len(observed):
        raise ValueError("test_start_frame must be below the frame count")
    valid = _target_frame_validity(visible, motion_valid)
    with np.load(profile_path) as profile:
        posterior_mean = np.asarray(profile["posterior_mean_trajectory"], dtype=float)
        epistemic = np.asarray(profile["epistemic_variance"], dtype=float)
    result: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "contract": {
            "target": "hard-valid direct track coordinates",
            "observation_variance": "fixed perception term",
            "model_discrepancy_variance": "causal residual moment after removing observation and epistemic terms",
        },
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "profile": {
                "path": str(Path(profile_path).resolve()),
                "sha256": _sha256(profile_path),
            },
        },
        "posterior": _calibrate_candidate(
            observed,
            posterior_mean,
            epistemic,
            valid,
            config,
        ),
    }
    if reference_trajectory_path is not None:
        reference = np.asarray(_load_pickle(reference_trajectory_path), dtype=float)
        reference = reference[: len(observed), : observed.shape[1]]
        result["inputs"]["reference_trajectory"] = {
            "path": str(Path(reference_trajectory_path).resolve()),
            "sha256": _sha256(reference_trajectory_path),
        }
        result["reference"] = _calibrate_candidate(
            observed,
            reference,
            np.zeros_like(observed),
            valid,
            config,
        )
    return result


def write_discrepancy_summary(summary: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
