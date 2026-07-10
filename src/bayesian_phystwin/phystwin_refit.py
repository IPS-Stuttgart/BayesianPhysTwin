"""Pure NumPy support for reliability-aware refits of official PhysTwin cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


REFIT_VARIANTS = ("hard", "visible", "cue", "mixture")


@dataclass(frozen=True)
class PhysTwinRefitReliabilityConfig:
    """Cue-to-prior parameters frozen before simulator residuals are observed."""

    minimum_probability: float = 1e-3
    confidence_power: float = 1.0
    boundary_scale: float = 0.03
    flow_scale: float = 0.005
    occlusion_probability: float = 1e-3


@dataclass(frozen=True)
class PhysTwinTrackObjective:
    """Frame-aligned arrays consumed by the differentiable Warp objective."""

    variant: str
    prior_inlier_probability: np.ndarray
    support: np.ndarray
    weights: np.ndarray
    normalizer: np.ndarray


def _frame_arrays(
    visible: np.ndarray,
    motion_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    visible_array = np.asarray(visible, dtype=bool)
    valid_array = np.asarray(motion_valid, dtype=bool)
    if visible_array.ndim != 2:
        raise ValueError("visible must have shape (T, N)")
    frame_count, track_count = visible_array.shape
    if valid_array.shape not in {
        (frame_count, track_count),
        (frame_count - 1, track_count),
    }:
        raise ValueError("motion_valid must have shape (T, N) or (T-1, N)")
    valid_by_target_frame = np.zeros_like(visible_array)
    valid_by_target_frame[0] = visible_array[0]
    valid_by_target_frame[1:] = valid_array[: frame_count - 1]
    return visible_array, valid_by_target_frame


def _aligned_cue(
    cues: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, int],
    *,
    default: float | np.ndarray,
) -> np.ndarray:
    frame_count, track_count = shape
    if name not in cues:
        return np.broadcast_to(np.asarray(default, dtype=float), shape).copy()
    values = np.asarray(cues[name], dtype=float)
    if values.shape == shape:
        aligned = values.copy()
    elif values.shape == (frame_count - 1, track_count):
        aligned = np.broadcast_to(np.asarray(default, dtype=float), shape).copy()
        aligned[1:] = values
    else:
        raise ValueError(
            f"cue {name} must have shape {shape} or "
            f"({frame_count - 1}, {track_count}), got {values.shape}"
        )
    if not np.all(np.isfinite(aligned)):
        raise ValueError(f"cue {name} must contain finite values")
    return aligned


def build_phystwin_track_objective(
    visible: np.ndarray,
    motion_valid: np.ndarray,
    *,
    cues: Mapping[str, np.ndarray] | None = None,
    variant: str,
    config: PhysTwinRefitReliabilityConfig | None = None,
) -> PhysTwinTrackObjective:
    """Build residual-independent per-track priors for a refit variant."""

    if variant not in REFIT_VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(REFIT_VARIANTS)}")
    cfg = config or PhysTwinRefitReliabilityConfig()
    if not 0.0 < cfg.minimum_probability < 0.5:
        raise ValueError("minimum_probability must be in (0, 0.5)")
    if cfg.confidence_power < 0.0:
        raise ValueError("confidence_power must be nonnegative")
    if cfg.boundary_scale <= 0.0 or cfg.flow_scale <= 0.0:
        raise ValueError("cue scales must be positive")
    if not 0.0 <= cfg.occlusion_probability <= 1.0:
        raise ValueError("occlusion_probability must be in [0, 1]")

    visible_array, valid_by_frame = _frame_arrays(visible, motion_valid)
    shape = visible_array.shape
    cue_arrays = cues or {}
    confidence = np.clip(
        _aligned_cue(cue_arrays, "confidence", shape, default=1.0),
        0.0,
        1.0,
    )
    occluded = _aligned_cue(
        cue_arrays,
        "occluded",
        shape,
        default=np.logical_not(visible_array),
    ).astype(bool)
    boundary_distance = np.maximum(
        _aligned_cue(cue_arrays, "boundary_distance", shape, default=1e6),
        0.0,
    )
    flow_inconsistency = np.maximum(
        _aligned_cue(cue_arrays, "flow_inconsistency", shape, default=0.0),
        0.0,
    )
    cue_prior = (
        np.power(confidence, cfg.confidence_power)
        * (1.0 - np.exp(-boundary_distance / cfg.boundary_scale))
        * np.exp(-flow_inconsistency / cfg.flow_scale)
        * np.where(occluded, cfg.occlusion_probability, 1.0)
    )
    cue_prior = np.clip(
        cue_prior,
        cfg.minimum_probability,
        1.0 - cfg.minimum_probability,
    )

    if variant == "hard":
        support = valid_by_frame
        prior = np.where(support, 1.0 - cfg.minimum_probability, cfg.minimum_probability)
        weights = support.astype(float)
    elif variant == "visible":
        support = visible_array
        prior = np.where(support, 1.0 - cfg.minimum_probability, cfg.minimum_probability)
        weights = support.astype(float)
    else:
        support = visible_array
        prior = cue_prior
        weights = support.astype(float) * cue_prior

    if variant == "mixture":
        normalizer = np.sum(support, axis=1, dtype=float)
    else:
        normalizer = np.sum(weights, axis=1, dtype=float)
    normalizer = np.maximum(normalizer, 1.0)
    return PhysTwinTrackObjective(
        variant=variant,
        prior_inlier_probability=prior.astype(np.float32),
        support=support.astype(np.int32),
        weights=weights.astype(np.float32),
        normalizer=normalizer.astype(np.float32),
    )


def phystwin_tracking_metrics(
    observed: np.ndarray,
    trajectory: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    """Summarize direct-correspondence tracking error for a selected group."""

    observed_array = np.asarray(observed, dtype=float)
    trajectory_array = np.asarray(trajectory, dtype=float)
    mask_array = np.asarray(mask, dtype=bool)
    if observed_array.ndim != 3 or observed_array.shape[2] != 3:
        raise ValueError("observed must have shape (T, N, 3)")
    if trajectory_array.ndim != 3 or trajectory_array.shape[2] != 3:
        raise ValueError("trajectory must have shape (T, M, 3)")
    if trajectory_array.shape[0] < observed_array.shape[0]:
        raise ValueError("trajectory has fewer frames than observed")
    if trajectory_array.shape[1] < observed_array.shape[1]:
        raise ValueError("trajectory has fewer vertices than observed tracks")
    if mask_array.shape != observed_array.shape[:2]:
        raise ValueError("mask must match observed's first two axes")
    residual = observed_array - trajectory_array[
        : observed_array.shape[0], : observed_array.shape[1]
    ]
    selected = residual[mask_array]
    if len(selected) == 0:
        return {"count": 0}
    norms = np.linalg.norm(selected, axis=1)
    return {
        "count": int(len(selected)),
        "coordinate_rmse_m": float(np.sqrt(np.mean(np.square(selected)))),
        "vector_rmse_m": float(np.sqrt(np.mean(np.square(norms)))),
        "mean_norm_m": float(np.mean(norms)),
        "median_norm_m": float(np.median(norms)),
        "p95_norm_m": float(np.quantile(norms, 0.95)),
        "max_norm_m": float(np.max(norms)),
    }


def evaluate_phystwin_trajectory(
    observed: np.ndarray,
    trajectory: np.ndarray,
    visible: np.ndarray,
    motion_valid: np.ndarray,
    *,
    train_end_frame: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Evaluate train/test errors for visible, hard-valid, and rejected tracks."""

    visible_array, valid_by_frame = _frame_arrays(visible, motion_valid)
    frame_count = visible_array.shape[0]
    if not 1 < train_end_frame < frame_count:
        raise ValueError("train_end_frame must be between 2 and T-1")
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for split, start, stop in (
        ("train", 1, train_end_frame),
        ("test", train_end_frame, frame_count),
    ):
        split_mask = np.zeros_like(visible_array)
        split_mask[start:stop] = True
        groups = {
            "visible": visible_array & split_mask,
            "hard_valid": valid_by_frame & split_mask,
            "visible_hard_invalid": visible_array & ~valid_by_frame & split_mask,
        }
        result[split] = {
            name: phystwin_tracking_metrics(observed, trajectory, group_mask)
            for name, group_mask in groups.items()
        }
    return result
