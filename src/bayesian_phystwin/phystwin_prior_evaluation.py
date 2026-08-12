"""Evaluate static and causal cue priors on PhysTwin's refit support."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import binary_calibration_metrics
from .phystwin_refit import (
    PhysTwinRefitReliabilityConfig,
    _frame_arrays,
    build_phystwin_track_objective,
)


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config(
    value: PhysTwinRefitReliabilityConfig | None,
) -> PhysTwinRefitReliabilityConfig:
    if value is None:
        return PhysTwinRefitReliabilityConfig()
    if isinstance(value, PhysTwinRefitReliabilityConfig):
        return value
    raise TypeError("config must be a PhysTwinRefitReliabilityConfig or None")


def evaluate_phystwin_prior_arrays(
    visible: np.ndarray,
    motion_valid: np.ndarray,
    cues: dict[str, np.ndarray],
    *,
    config: PhysTwinRefitReliabilityConfig | None = None,
) -> dict[str, object]:
    """Compare cue priors with hard-gate labels on target-visible tracks."""

    cfg = _config(config)
    visible_array, valid_by_target_frame = _frame_arrays(visible, motion_valid)
    support = visible_array[1:]
    labels = valid_by_target_frame[1:][support]
    variants: dict[str, object] = {}
    for variant in ("mixture", "markov_mixture"):
        objective = build_phystwin_track_objective(
            visible_array,
            motion_valid,
            cues=cues,
            variant=variant,
            config=cfg,
        )
        probability = objective.prior_inlier_probability[1:][support].astype(float)
        variants[variant] = {
            "calibration": binary_calibration_metrics(probability, labels).as_dict(),
            "mean_prior_hard_valid": float(np.mean(probability[labels])),
            "mean_prior_hard_invalid": float(
                np.mean(probability[np.logical_not(labels)])
            ),
        }
    return {
        "support_contract": "target frame visible; motion gate indexed at t-1",
        "warning": (
            "PhysTwin's motion gate is a related heuristic, not corruption "
            "ground truth."
        ),
        "measurement_count": int(len(labels)),
        "hard_valid_rate": float(np.mean(labels)),
        "variants": variants,
    }


def evaluate_phystwin_prior_files(
    final_data_path: str | Path,
    cues_path: str | Path,
    *,
    config: PhysTwinRefitReliabilityConfig | None = None,
) -> dict[str, object]:
    """Load trusted official artifacts and return prior calibration provenance."""

    cfg = _config(config)
    data = _load_pickle(final_data_path)
    with np.load(cues_path) as archive:
        cues = {name: np.asarray(archive[name]) for name in archive.files}
    result = evaluate_phystwin_prior_arrays(
        data["object_visibilities"],
        data["object_motions_valid"],
        cues,
        config=cfg,
    )
    return {
        "schema_version": 1,
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "cues": {
                "path": str(Path(cues_path).resolve()),
                "sha256": _sha256(cues_path),
            },
        },
        "config": {
            "reliability": vars(cfg),
            "static_variant": "mixture",
            "temporal_variant": "markov_mixture",
        },
        **result,
    }


def write_prior_evaluation(summary: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
