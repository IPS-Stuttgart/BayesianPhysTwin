"""Causal tactile guard for baseline-relative Deform360 belief updates.

The tactile channel is independent of the multiview camera gauge, but the
released episode-normalized arrays use a future-dependent peak scale.  This
module therefore works with raw, baseline-subtracted taxel values and never
normalizes by an episode-wide statistic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

TACTILE_REGRET_FEATURE_NAMES = (
    "total_energy_over_initial",
    "total_energy_change_fraction_3",
    "mean_pattern_change_fraction_3",
    "maximum_pattern_change_fraction_3",
    "mean_pattern_cosine_3",
    "energy_concentration",
    "segment_energy_cv",
    "cumulative_energy_change_from_frame0_fraction",
    "sensor_ratio_min",
    "sensor_ratio_max",
    "sensor_ratio_std",
    "active_taxel_mean",
    "active_taxel_std",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class AlignedTactileResponse:
    """Raw baseline-subtracted tactile values on a target camera timeline."""

    response: np.ndarray
    source_indices: np.ndarray
    signed_delta_us: np.ndarray


@dataclass(frozen=True)
class TactileRegretGuardModel:
    """Source-fitted linear benefit score with a conservative admission gate."""

    feature_center: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    ridge_penalty: float
    admission_threshold: float
    source_object_count: int
    source_row_count: int

    def __post_init__(self) -> None:
        feature_count = len(TACTILE_REGRET_FEATURE_NAMES)
        _require(
            len(self.feature_center) == feature_count
            and len(self.feature_scale) == feature_count,
            "tactile guard feature scaling has the wrong length",
        )
        _require(
            len(self.coefficients) == feature_count + 1,
            "tactile guard coefficients must include one intercept",
        )
        _require(
            all(np.isfinite(value) for value in self.feature_center)
            and all(np.isfinite(value) and value > 0.0 for value in self.feature_scale)
            and all(np.isfinite(value) for value in self.coefficients),
            "tactile guard model contains invalid values",
        )
        _require(
            np.isfinite(self.ridge_penalty) and self.ridge_penalty > 0.0,
            "ridge penalty must be finite and positive",
        )
        _require(
            np.isfinite(self.admission_threshold),
            "admission threshold must be finite",
        )
        _require(
            self.source_object_count >= 2 and self.source_row_count >= 2,
            "tactile guard needs at least two source objects and rows",
        )


def align_baseline_subtracted_tactile(
    raw_frames: np.ndarray,
    source_timestamps_us: np.ndarray,
    baseline: np.ndarray,
    target_timestamps_us: np.ndarray,
    *,
    invalid_columns: Sequence[int] = (-1,),
) -> AlignedTactileResponse:
    """Align raw taxels causally without future-dependent normalization."""

    frames = np.asarray(raw_frames)
    source = np.asarray(source_timestamps_us, dtype=np.int64)
    reference = np.asarray(baseline)
    target = np.asarray(target_timestamps_us, dtype=np.int64)
    _require(
        frames.ndim == 3 and frames.shape[1:] == reference.shape,
        "raw tactile frames and baseline have incompatible shapes",
    )
    _require(
        len(frames) == len(source) and len(frames) > 0,
        "raw tactile frames and timestamps differ",
    )
    _require(
        target.ndim == source.ndim == 1 and len(target) > 0,
        "tactile timestamps must be nonempty vectors",
    )
    _require(
        np.all(np.diff(source) >= 0) and np.all(np.diff(target) >= 0),
        "tactile timelines must be sorted",
    )
    _require(
        np.all(np.isfinite(frames)) and np.all(np.isfinite(reference)),
        "raw tactile input contains non-finite values",
    )

    following = np.searchsorted(source, target)
    following = np.clip(following, 0, len(source) - 1)
    preceding = np.maximum(following - 1, 0)
    # An exact tie uses the earlier sample, matching the released alignment rule.
    choose_preceding = np.abs(source[preceding] - target) <= np.abs(
        source[following] - target
    )
    indices = np.where(choose_preceding, preceding, following)
    response = frames[indices].astype(np.float64) - reference
    width = response.shape[2]
    columns = []
    for column in invalid_columns:
        resolved = int(column)
        if resolved < 0:
            resolved += width
        _require(0 <= resolved < width, "invalid tactile column is out of range")
        if resolved not in columns:
            columns.append(resolved)
    if columns:
        response[:, :, columns] = 0.0
    return AlignedTactileResponse(
        response=response,
        source_indices=np.asarray(indices, dtype=np.int64),
        signed_delta_us=np.asarray(source[indices] - target, dtype=np.int64),
    )


def causal_tactile_regret_features(
    tactile_response: np.ndarray,
    *,
    update_frames: Sequence[int] = (19, 38, 57),
    initial_reference_frame_count: int = 6,
    history_frame_count: int = 3,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Extract update-local features from ``(sensor,time,row,column)`` taxels."""

    response = np.asarray(tactile_response, dtype=np.float64)
    updates = tuple(int(frame) for frame in update_frames)
    _require(
        response.ndim == 4 and response.shape[0] > 0,
        "tactile response must have shape (sensor,time,row,column)",
    )
    _require(np.all(np.isfinite(response)), "tactile response is non-finite")
    _require(
        1 <= initial_reference_frame_count <= response.shape[1],
        "invalid tactile reference-frame count",
    )
    _require(history_frame_count >= 1, "tactile history must be positive")
    _require(
        updates
        and tuple(sorted(set(updates))) == updates
        and updates[0] >= max(initial_reference_frame_count, history_frame_count)
        and updates[-1] < response.shape[1],
        "invalid tactile update frames",
    )

    positive = np.maximum(response, 0.0)
    energy = positive.sum(axis=(2, 3)).T
    active_count = np.count_nonzero(positive > 0.0, axis=(2, 3)).T
    initial_energy = np.maximum(
        np.median(energy[:initial_reference_frame_count], axis=0),
        1.0,
    )
    rows = []
    diagnostics: list[dict[str, Any]] = []
    for update_index, update in enumerate(updates):
        history_start = update - history_frame_count
        previous_pattern = positive[:, history_start:update].mean(axis=1)
        current_pattern = positive[:, update]
        previous_flat = previous_pattern.reshape(len(previous_pattern), -1)
        current_flat = current_pattern.reshape(len(current_pattern), -1)
        previous_norm = np.linalg.norm(previous_flat, axis=1)
        current_norm = np.linalg.norm(current_flat, axis=1)
        pattern_change = np.linalg.norm(
            current_flat - previous_flat,
            axis=1,
        ) / np.maximum(previous_norm, 1e-9)
        pattern_cosine = np.sum(current_flat * previous_flat, axis=1) / np.maximum(
            current_norm * previous_norm,
            1e-9,
        )
        segment_start = 0 if update_index == 0 else updates[update_index - 1]
        segment_energy = energy[segment_start : update + 1].sum(axis=1)
        current_energy = energy[update]
        total_energy = float(np.sum(current_energy))
        previous_total = float(
            np.sum(np.mean(energy[history_start:update], axis=0))
        )
        sensor_ratio = current_energy / initial_energy
        current_active = active_count[update]
        values = np.asarray(
            (
                total_energy / float(np.sum(initial_energy)),
                (total_energy - previous_total) / max(previous_total, 1.0),
                float(np.mean(pattern_change)),
                float(np.max(pattern_change)),
                float(np.mean(pattern_cosine)),
                float(np.max(current_energy) / max(total_energy, 1.0)),
                float(
                    np.std(segment_energy) / max(np.mean(segment_energy), 1.0)
                ),
                float(
                    (total_energy - np.sum(energy[0]))
                    / max(float(np.sum(energy[0])), 1.0)
                ),
                float(np.min(sensor_ratio)),
                float(np.max(sensor_ratio)),
                float(np.std(sensor_ratio)),
                float(np.mean(current_active)),
                float(np.std(current_active)),
            ),
            dtype=np.float64,
        )
        _require(np.all(np.isfinite(values)), "tactile features are non-finite")
        rows.append(values)
        report = dict(
            zip(TACTILE_REGRET_FEATURE_NAMES, values.tolist(), strict=True)
        )
        report.update(
            {
                "update_frame": update,
                "sensor_energy": current_energy.tolist(),
                "sensor_active_taxels": current_active.astype(int).tolist(),
                "sensor_energy_over_initial": sensor_ratio.tolist(),
            }
        )
        diagnostics.append(report)
    return np.asarray(rows), diagnostics


def fit_object_balanced_tactile_regret_guard(
    feature_vectors: np.ndarray,
    maximum_regret_m: np.ndarray,
    object_ids: Sequence[str],
    *,
    ridge_penalty: float = 10.0,
    admission_threshold: float = 0.7,
) -> TactileRegretGuardModel:
    """Fit a source-only linear benefit score with equal object loss mass."""

    features = np.asarray(feature_vectors, dtype=np.float64)
    regret = np.asarray(maximum_regret_m, dtype=np.float64)
    groups = tuple(str(value) for value in object_ids)
    feature_count = len(TACTILE_REGRET_FEATURE_NAMES)
    _require(
        features.ndim == 2 and features.shape[1] == feature_count,
        "tactile source features have the wrong shape",
    )
    _require(
        len(features) == len(regret) == len(groups) and len(features) >= 2,
        "tactile source rows differ",
    )
    _require(
        np.all(np.isfinite(features)) and np.all(np.isfinite(regret)),
        "tactile source rows contain non-finite values",
    )
    unique_groups = tuple(sorted(set(groups)))
    _require(len(unique_groups) >= 2, "tactile guard needs multiple source objects")
    _require(
        np.isfinite(ridge_penalty) and ridge_penalty > 0.0,
        "ridge penalty must be finite and positive",
    )
    _require(np.isfinite(admission_threshold), "admission threshold must be finite")

    center = np.median(features, axis=0)
    scale = np.maximum(
        np.quantile(features, 0.75, axis=0)
        - np.quantile(features, 0.25, axis=0),
        1e-9,
    )
    standardized = (features - center) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    counts = {group: groups.count(group) for group in unique_groups}
    weights = np.asarray([1.0 / counts[group] for group in groups])
    weights *= len(weights) / float(np.sum(weights))
    labels = np.asarray(regret < 0.0, dtype=np.float64)
    regularizer = np.diag((0.0,) + (ridge_penalty,) * feature_count)
    coefficients = np.linalg.solve(
        design.T @ (weights[:, None] * design) + regularizer,
        design.T @ (weights * labels),
    )
    return TactileRegretGuardModel(
        feature_center=tuple(center.tolist()),
        feature_scale=tuple(scale.tolist()),
        coefficients=tuple(coefficients.tolist()),
        ridge_penalty=float(ridge_penalty),
        admission_threshold=float(admission_threshold),
        source_object_count=len(unique_groups),
        source_row_count=len(features),
    )


def tactile_benefit_scores(
    feature_vectors: np.ndarray,
    model: TactileRegretGuardModel,
) -> np.ndarray:
    """Return the linear source-fitted score; it is not a probability."""

    features = np.asarray(feature_vectors, dtype=np.float64)
    _require(
        features.ndim == 2
        and features.shape[1] == len(TACTILE_REGRET_FEATURE_NAMES),
        "tactile guard features have the wrong shape",
    )
    _require(np.all(np.isfinite(features)), "tactile guard features are non-finite")
    standardized = (
        features - np.asarray(model.feature_center)
    ) / np.asarray(model.feature_scale)
    design = np.column_stack((np.ones(len(standardized)), standardized))
    return design @ np.asarray(model.coefficients)


def apply_tactile_regret_guard(
    baseline: np.ndarray,
    candidate: np.ndarray,
    feature_vectors: np.ndarray,
    model: TactileRegretGuardModel,
    *,
    update_frames: Sequence[int] = (19, 38, 57),
) -> tuple[dict[str, Any], np.ndarray]:
    """Select candidate intervals or preserve the baseline bit exactly."""

    baseline_input = np.asarray(baseline)
    candidate_input = np.asarray(candidate)
    updates = tuple(int(frame) for frame in update_frames)
    _require(
        baseline_input.shape == candidate_input.shape and baseline_input.ndim >= 1,
        "tactile guard candidate and baseline differ",
    )
    _require(
        updates and tuple(sorted(set(updates))) == updates and updates[-1] < len(baseline_input),
        "invalid tactile guard update frames",
    )
    scores = tactile_benefit_scores(feature_vectors, model)
    _require(len(scores) == len(updates), "tactile guard update count changed")

    guarded = baseline_input.copy()
    decisions = []
    for index, (update, score) in enumerate(zip(updates, scores, strict=True)):
        stop = updates[index + 1] if index + 1 < len(updates) else len(guarded)
        accepted = bool(score >= model.admission_threshold)
        if accepted:
            guarded[update + 1 : stop] = candidate_input[update + 1 : stop]
        exact_fallback = bool(
            not accepted
            and np.array_equal(
                guarded[update + 1 : stop],
                baseline_input[update + 1 : stop],
            )
        )
        if not accepted and not exact_fallback:
            raise AssertionError("tactile rejection changed the exact baseline")
        decisions.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "candidate_accepted": accepted,
                "benefit_score": float(score),
                "admission_threshold": model.admission_threshold,
                "reason": (
                    "source-fitted-tactile-benefit"
                    if accepted
                    else "exact-baseline-fallback"
                ),
                "bit_exact_baseline_fallback": exact_fallback,
            }
        )
    return {
        "arm": "dual_backbone_pairwise_tactile_guarded",
        "feature_names": list(TACTILE_REGRET_FEATURE_NAMES),
        "updates": decisions,
        "information_boundary": {
            "target_argument_accepted": False,
            "future_tactile_read": False,
            "episode_wide_tactile_normalization_used": False,
            "source_outcomes_used_only_to_fit_guard": True,
            "rejection_is_bit_exact_selected_backbone": True,
        },
    }, guarded


__all__ = [
    "AlignedTactileResponse",
    "TACTILE_REGRET_FEATURE_NAMES",
    "TactileRegretGuardModel",
    "align_baseline_subtracted_tactile",
    "apply_tactile_regret_guard",
    "causal_tactile_regret_features",
    "fit_object_balanced_tactile_regret_guard",
    "tactile_benefit_scores",
]
