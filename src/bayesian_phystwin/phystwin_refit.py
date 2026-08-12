"""Pure NumPy support for reliability-aware refits of official PhysTwin cases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

REFIT_VARIANTS = (
    "hard",
    "visible",
    "cue",
    "mixture",
    "markov_cue",
    "markov_mixture",
)


def _finite_scalar(value: object, *, name: str) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer_scalar(value: object, *, name: str) -> int:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer")
    return int(raw)


def _numeric_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _boolean_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must contain only booleans")
    return np.array(raw, dtype=np.bool_, copy=True, order="C")


def _positive_optional_scalar(value: object, *, name: str) -> float | None:
    if value is None:
        return None
    result = _finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive when enabled")
    return result


@dataclass(frozen=True)
class PhysTwinRefitReliabilityConfig:
    """Cue-to-prior parameters frozen before simulator residuals are observed."""

    minimum_probability: float = 1e-3
    confidence_power: float = 1.0
    visibility_power: float = 1.0
    boundary_scale: float | None = 0.03
    flow_scale: float | None = 0.005
    forward_backward_scale_px: float | None = None
    multiview_scale_px: float | None = None
    occlusion_probability: float = 1e-3
    markov_inlier_persistence: float = 0.98
    markov_outlier_persistence: float = 0.90

    def __post_init__(self) -> None:
        minimum_probability = _finite_scalar(
            self.minimum_probability,
            name="minimum_probability",
        )
        confidence_power = _finite_scalar(
            self.confidence_power,
            name="confidence_power",
        )
        visibility_power = _finite_scalar(
            self.visibility_power,
            name="visibility_power",
        )
        boundary_scale = _positive_optional_scalar(
            self.boundary_scale,
            name="boundary_scale",
        )
        flow_scale = _positive_optional_scalar(
            self.flow_scale,
            name="flow_scale",
        )
        forward_backward_scale = _positive_optional_scalar(
            self.forward_backward_scale_px,
            name="forward_backward_scale_px",
        )
        multiview_scale = _positive_optional_scalar(
            self.multiview_scale_px,
            name="multiview_scale_px",
        )
        occlusion_probability = _finite_scalar(
            self.occlusion_probability,
            name="occlusion_probability",
        )
        inlier_persistence = _finite_scalar(
            self.markov_inlier_persistence,
            name="markov_inlier_persistence",
        )
        outlier_persistence = _finite_scalar(
            self.markov_outlier_persistence,
            name="markov_outlier_persistence",
        )

        if not 0.0 < minimum_probability < 0.5:
            raise ValueError("minimum_probability must be in (0, 0.5)")
        if confidence_power < 0.0 or visibility_power < 0.0:
            raise ValueError("confidence and visibility powers must be nonnegative")
        if not 0.0 <= occlusion_probability <= 1.0:
            raise ValueError("occlusion_probability must be in [0, 1]")
        if not 0.0 < inlier_persistence < 1.0:
            raise ValueError("markov_inlier_persistence must be in (0, 1)")
        if not 0.0 < outlier_persistence < 1.0:
            raise ValueError("markov_outlier_persistence must be in (0, 1)")

        object.__setattr__(self, "minimum_probability", minimum_probability)
        object.__setattr__(self, "confidence_power", confidence_power)
        object.__setattr__(self, "visibility_power", visibility_power)
        object.__setattr__(self, "boundary_scale", boundary_scale)
        object.__setattr__(self, "flow_scale", flow_scale)
        object.__setattr__(
            self,
            "forward_backward_scale_px",
            forward_backward_scale,
        )
        object.__setattr__(self, "multiview_scale_px", multiview_scale)
        object.__setattr__(
            self,
            "occlusion_probability",
            occlusion_probability,
        )
        object.__setattr__(
            self,
            "markov_inlier_persistence",
            inlier_persistence,
        )
        object.__setattr__(
            self,
            "markov_outlier_persistence",
            outlier_persistence,
        )


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
    visible_array = _boolean_array(visible, name="visible")
    valid_array = _boolean_array(motion_valid, name="motion_valid")
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


def _aligned_numeric_cue(
    cues: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, int],
    *,
    default: float | np.ndarray,
) -> np.ndarray:
    frame_count, track_count = shape
    default_array = _numeric_array(default, name=f"default {name}")
    if name not in cues:
        return np.broadcast_to(default_array, shape).copy()
    values = _numeric_array(cues[name], name=f"cue {name}")
    if values.shape == shape:
        return values.copy()
    if values.shape == (frame_count - 1, track_count):
        aligned = np.broadcast_to(default_array, shape).copy()
        aligned[1:] = values
        return aligned
    raise ValueError(
        f"cue {name} must have shape {shape} or "
        f"({frame_count - 1}, {track_count}), got {values.shape}"
    )


def _aligned_boolean_cue(
    cues: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, int],
    *,
    default: bool | np.ndarray,
) -> np.ndarray:
    frame_count, track_count = shape
    default_array = _boolean_array(default, name=f"default {name}")
    if name not in cues:
        return np.broadcast_to(default_array, shape).copy()
    values = _boolean_array(cues[name], name=f"cue {name}")
    if values.shape == shape:
        return values.copy()
    if values.shape == (frame_count - 1, track_count):
        aligned = np.broadcast_to(default_array, shape).copy()
        aligned[1:] = values
        return aligned
    raise ValueError(
        f"cue {name} must have shape {shape} or "
        f"({frame_count - 1}, {track_count}), got {values.shape}"
    )


def _unit_interval_cue(
    cues: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, int],
    *,
    default: float | np.ndarray,
) -> np.ndarray:
    values = _aligned_numeric_cue(cues, name, shape, default=default)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"cue {name} must lie in [0, 1]")
    return values


def _nonnegative_cue(
    cues: Mapping[str, np.ndarray],
    name: str,
    shape: tuple[int, int],
    *,
    default: float | np.ndarray,
) -> np.ndarray:
    values = _aligned_numeric_cue(cues, name, shape, default=default)
    if np.any(values < 0.0):
        raise ValueError(f"cue {name} must be nonnegative")
    return values


def causal_markov_cue_reliability(
    prior_reliability: np.ndarray,
    *,
    inlier_persistence: float = 0.98,
    outlier_persistence: float = 0.90,
    probability_floor: float = 1e-3,
) -> np.ndarray:
    """Filter persistent inlier states using current and previous cues only."""

    prior = _numeric_array(prior_reliability, name="prior_reliability")
    inlier = _finite_scalar(inlier_persistence, name="inlier_persistence")
    outlier = _finite_scalar(outlier_persistence, name="outlier_persistence")
    floor = _finite_scalar(probability_floor, name="probability_floor")
    if prior.ndim != 2:
        raise ValueError("prior_reliability must have shape (T, N)")
    if np.any((prior < 0.0) | (prior > 1.0)):
        raise ValueError("prior_reliability must lie in [0, 1]")
    if not 0.0 < inlier < 1.0:
        raise ValueError("inlier_persistence must be in (0, 1)")
    if not 0.0 < outlier < 1.0:
        raise ValueError("outlier_persistence must be in (0, 1)")
    if not 0.0 < floor < 0.5:
        raise ValueError("probability_floor must be in (0, 0.5)")

    clipped = np.clip(prior, floor, 1.0 - floor)
    transition = np.array(
        [
            [outlier, 1.0 - outlier],
            [1.0 - inlier, inlier],
        ],
        dtype=float,
    )
    log_transition = np.log(transition)
    stationary_inlier = (1.0 - outlier) / (2.0 - inlier - outlier)
    log_initial = np.log([1.0 - stationary_inlier, stationary_inlier])
    filtered = np.empty_like(clipped)
    alpha_outlier = log_initial[0] + np.log1p(-clipped[0])
    alpha_inlier = log_initial[1] + np.log(clipped[0])
    normalizer = np.logaddexp(alpha_outlier, alpha_inlier)
    alpha_outlier -= normalizer
    alpha_inlier -= normalizer
    filtered[0] = np.exp(alpha_inlier)
    for frame in range(1, len(clipped)):
        predicted_outlier = np.logaddexp(
            alpha_outlier + log_transition[0, 0],
            alpha_inlier + log_transition[1, 0],
        )
        predicted_inlier = np.logaddexp(
            alpha_outlier + log_transition[0, 1],
            alpha_inlier + log_transition[1, 1],
        )
        alpha_outlier = predicted_outlier + np.log1p(-clipped[frame])
        alpha_inlier = predicted_inlier + np.log(clipped[frame])
        normalizer = np.logaddexp(alpha_outlier, alpha_inlier)
        alpha_outlier -= normalizer
        alpha_inlier -= normalizer
        filtered[frame] = np.exp(alpha_inlier)
    return np.clip(filtered, floor, 1.0 - floor)


def build_phystwin_track_objective(
    visible: np.ndarray,
    motion_valid: np.ndarray,
    *,
    cues: Mapping[str, np.ndarray] | None = None,
    variant: str,
    config: PhysTwinRefitReliabilityConfig | None = None,
) -> PhysTwinTrackObjective:
    """Build residual-independent per-track priors for a refit variant."""

    if not isinstance(variant, str) or variant not in REFIT_VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(REFIT_VARIANTS)}")
    if config is None:
        cfg = PhysTwinRefitReliabilityConfig()
    elif isinstance(config, PhysTwinRefitReliabilityConfig):
        cfg = config
    else:
        raise TypeError("config must be a PhysTwinRefitReliabilityConfig or None")
    if cues is None:
        cue_arrays: Mapping[str, np.ndarray] = {}
    elif isinstance(cues, Mapping):
        cue_arrays = cues
    else:
        raise TypeError("cues must be a mapping or None")

    visible_array, valid_by_frame = _frame_arrays(visible, motion_valid)
    shape = visible_array.shape
    confidence = _unit_interval_cue(
        cue_arrays,
        "confidence",
        shape,
        default=1.0,
    )
    visibility_probability = _unit_interval_cue(
        cue_arrays,
        "visibility_probability",
        shape,
        default=1.0,
    )
    occluded = _aligned_boolean_cue(
        cue_arrays,
        "occluded",
        shape,
        default=np.logical_not(visible_array),
    )
    boundary_distance = _nonnegative_cue(
        cue_arrays,
        "boundary_distance",
        shape,
        default=1e6,
    )
    flow_inconsistency = _nonnegative_cue(
        cue_arrays,
        "flow_inconsistency",
        shape,
        default=0.0,
    )
    forward_backward_error = _nonnegative_cue(
        cue_arrays,
        "forward_backward_error_px",
        shape,
        default=0.0,
    )
    forward_backward_valid = _aligned_boolean_cue(
        cue_arrays,
        "forward_backward_valid",
        shape,
        default=False,
    )
    multiview_error = _nonnegative_cue(
        cue_arrays,
        "multiview_reprojection_error_px",
        shape,
        default=0.0,
    )
    multiview_valid = _aligned_boolean_cue(
        cue_arrays,
        "multiview_valid",
        shape,
        default=False,
    )
    boundary_factor = np.ones(shape, dtype=float)
    if cfg.boundary_scale is not None:
        boundary_factor = 1.0 - np.exp(-boundary_distance / cfg.boundary_scale)
    flow_factor = np.ones(shape, dtype=float)
    if cfg.flow_scale is not None:
        flow_factor = np.exp(-flow_inconsistency / cfg.flow_scale)
    forward_backward_factor = np.ones(shape, dtype=float)
    if cfg.forward_backward_scale_px is not None:
        forward_backward_factor[forward_backward_valid] = np.exp(
            -forward_backward_error[forward_backward_valid]
            / cfg.forward_backward_scale_px
        )
    multiview_factor = np.ones(shape, dtype=float)
    if cfg.multiview_scale_px is not None:
        multiview_factor[multiview_valid] = np.exp(
            -multiview_error[multiview_valid] / cfg.multiview_scale_px
        )
    cue_prior = (
        np.power(confidence, cfg.confidence_power)
        * np.power(visibility_probability, cfg.visibility_power)
        * boundary_factor
        * flow_factor
        * forward_backward_factor
        * multiview_factor
        * np.where(occluded, cfg.occlusion_probability, 1.0)
    )
    cue_prior = np.clip(
        cue_prior,
        cfg.minimum_probability,
        1.0 - cfg.minimum_probability,
    )
    if variant.startswith("markov_"):
        cue_prior = causal_markov_cue_reliability(
            cue_prior,
            inlier_persistence=cfg.markov_inlier_persistence,
            outlier_persistence=cfg.markov_outlier_persistence,
            probability_floor=cfg.minimum_probability,
        )

    if variant == "hard":
        support = valid_by_frame
        prior = np.where(
            support,
            1.0 - cfg.minimum_probability,
            cfg.minimum_probability,
        )
        weights: np.ndarray = support.astype(float)
    elif variant == "visible":
        support = visible_array
        prior = np.where(
            support,
            1.0 - cfg.minimum_probability,
            cfg.minimum_probability,
        )
        weights = support.astype(float)
    else:
        support = visible_array
        prior = cue_prior
        weights = support.astype(float) * cue_prior

    if variant.endswith("mixture"):
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

    observed_array = _numeric_array(observed, name="observed")
    trajectory_array = _numeric_array(trajectory, name="trajectory")
    mask_array = _boolean_array(mask, name="mask")
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
    residual = (
        observed_array
        - trajectory_array[: observed_array.shape[0], : observed_array.shape[1]]
    )
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

    visible_array = _boolean_array(visible, name="visible")
    if visible_array.ndim != 2:
        raise ValueError("visible must have shape (T, N)")
    frame_count = visible_array.shape[0]
    train_end = _integer_scalar(train_end_frame, name="train_end_frame")
    if not 1 < train_end < frame_count:
        raise ValueError("train_end_frame must be between 2 and T-1")
    return evaluate_phystwin_trajectory_splits(
        observed,
        trajectory,
        visible_array,
        motion_valid,
        splits={
            "train": (1, train_end),
            "test": (train_end, frame_count),
        },
    )


def evaluate_phystwin_trajectory_splits(
    observed: np.ndarray,
    trajectory: np.ndarray,
    visible: np.ndarray,
    motion_valid: np.ndarray,
    *,
    splits: Mapping[str, tuple[int, int]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Evaluate direct tracking groups over named half-open frame intervals."""

    if not isinstance(splits, Mapping):
        raise TypeError("splits must be a mapping")
    visible_array, valid_by_frame = _frame_arrays(visible, motion_valid)
    frame_count = visible_array.shape[0]
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for split, bounds in splits.items():
        if not isinstance(split, str) or not split:
            raise TypeError("split names must be nonempty strings")
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise TypeError(f"split {split} bounds must be a two-integer tuple")
        start = _integer_scalar(bounds[0], name=f"split {split} start")
        stop = _integer_scalar(bounds[1], name=f"split {split} stop")
        if not 0 <= start < stop <= frame_count:
            raise ValueError(
                f"split {split} must satisfy 0 <= start < stop <= {frame_count}"
            )
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
