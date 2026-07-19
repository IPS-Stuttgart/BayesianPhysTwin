"""Risk-limited online belief evaluation on the open Deform360 source panel.

The adapter is deliberately separate from the held Deform360 protocols.  It
only accepts the 27 public source episodes whose physical predictions were
hashed before their source futures were opened.  The sealed ``prediction_m``
array is the physical prior; sparse material-point measurements are read from
the subsequently constructed ``target_data.pkl`` outcome.

Assimilation centres are never scored.  Both the identity-aware RMSE and both
directions of the symmetric Chamfer metric operate on the permanently hidden
material identities.  A rejected update leaves every risk-limited trajectory
bit-for-bit equal to the sealed physical prior over that update interval.

This is an independent-source transfer protocol, not parity with the official
Deform360 Table-4 evaluator.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .phystwin_official_evaluation import _nearest_distances
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    deterministic_farthest_point_ids,
    finite_sample_absolute_residual_quantile_m,
    initialize_recursive_rbf_belief,
    robust_huber_continuation_gain,
    update_recursive_rbf_belief,
)


PROTOCOL_ID = "deform360-open-source-online-belief-v2-development"
CENTER_COUNT = 16
UPDATE_FRAMES = (19, 38, 57)
MINIMUM_UPDATE_CENTER_COUNT = 9
MINIMUM_HISTORY_POINT_COUNT = 3
MINIMUM_DISPERSION_THRESHOLD_M = 0.010
HISTORY_DISPERSION_QUANTILE = 0.95
HISTORY_DISPERSION_MULTIPLIER = 1.5
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
CONFORMAL_STYLE_COVERAGES = (0.50, 0.90, 0.95)
CONTINUATION_GAIN_THRESHOLD = 0.25
CORRESPONDENCE_SAFE_INLIER_COUNT = 13

EXPECTED_SOURCE_EPISODES: Mapping[str, tuple[int, ...]] = {
    "002-rope-silk": (2, 5, 6, 7, 9),
    "083-blanket-cloth": (1, 2, 4, 5, 8, 9),
    "085-scarf-cloth": (3, 4, 6, 8, 9),
    "092-squirrel": (4, 5, 7, 8, 9),
    "170-spider": (0, 1, 3, 5, 8, 9),
}

ARMS = (
    "physical_prior",
    "persistence",
    "recursive_global_translation",
    "recursive_rbf_ungated",
    "recursive_rbf_risk_limited",
    "recursive_rbf_causal_continuation",
    "recursive_rbf_correspondence_safe",
    "risk_limited_frozen_current_state",
)
PRIMARY_METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
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


def _expected_episode_directories() -> tuple[str, ...]:
    return tuple(
        f"{object_id}-ep{episode_id:04d}"
        for object_id, episode_ids in EXPECTED_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    )


def _post_update_scored_frames(frame_count: int) -> tuple[int, ...]:
    if frame_count <= UPDATE_FRAMES[-1] + 1:
        raise ValueError("trajectory does not cover the fixed Deform360 updates")
    scored: list[int] = []
    for index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[index + 1] if index + 1 < len(UPDATE_FRAMES) else frame_count
        )
        scored.extend(range(update + 1, stop))
    if not scored:
        raise ValueError("fixed online protocol has no scored frame")
    return tuple(scored)


def _radial_residuals_m(residual_m: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual_m, dtype=float)
    if residual.ndim != 2 or residual.shape[1] != 3 or len(residual) == 0:
        raise ValueError("residual_m must have nonempty shape (N, 3)")
    if not np.all(np.isfinite(residual)):
        raise ValueError("residual_m must be finite")
    location = np.median(residual, axis=0)
    return np.linalg.norm(residual - location, axis=1)


def _frozen_history_dispersion_m(
    physical_prior_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    center_ids: np.ndarray,
    *,
    first_update_frame: int,
) -> np.ndarray:
    """Return one robust dispersion scalar per calibration-history frame.

    The history is frozen to ``[0, first_update_frame)``.  A frame contributes
    only when at least three selected centres are supported.  Its scalar is
    ``median_i ||r_i - coordinate_median(r)||``.  Later between-update
    observations are therefore consumed by neither the gate nor the filter.
    """

    dispersion: list[float] = []
    for frame in range(first_update_frame):
        available = (
            visibility[frame, center_ids]
            & validity[frame, center_ids]
            & np.all(np.isfinite(target_m[frame, center_ids]), axis=1)
            & np.all(np.isfinite(physical_prior_m[frame, center_ids]), axis=1)
        )
        if int(np.sum(available)) < MINIMUM_HISTORY_POINT_COUNT:
            continue
        residual = (
            target_m[frame, center_ids[available]]
            - physical_prior_m[frame, center_ids[available]]
        )
        dispersion.append(float(np.median(_radial_residuals_m(residual))))
    return np.asarray(dispersion, dtype=float)


def _risk_dispersion_threshold_m(
    history_dispersion_m: np.ndarray,
) -> tuple[float, float | None]:
    history = np.asarray(history_dispersion_m, dtype=float)
    if history.ndim != 1 or not np.all(np.isfinite(history)):
        raise ValueError("history dispersions must be a finite vector")
    if not len(history):
        return MINIMUM_DISPERSION_THRESHOLD_M, None
    reference = float(np.quantile(history, HISTORY_DISPERSION_QUANTILE))
    return (
        max(
            MINIMUM_DISPERSION_THRESHOLD_M,
            HISTORY_DISPERSION_MULTIPLIER * reference,
        ),
        reference,
    )


def _symmetric_euclidean_chamfer_m(
    predicted_m: np.ndarray,
    target_m: np.ndarray,
) -> float:
    predicted = np.asarray(predicted_m, dtype=float)
    target = np.asarray(target_m, dtype=float)
    if (
        predicted.ndim != 2
        or target.ndim != 2
        or predicted.shape[1:] != (3,)
        or target.shape[1:] != (3,)
        or len(predicted) == 0
        or len(target) == 0
    ):
        raise ValueError("Chamfer inputs must have nonempty shape (N, 3)")
    target_to_prediction, _ = _nearest_distances(predicted, target, p=2)
    prediction_to_target, _ = _nearest_distances(target, predicted, p=2)
    return 0.5 * (
        float(np.mean(target_to_prediction)) + float(np.mean(prediction_to_target))
    )


def score_deform360_hidden_trajectory(
    trajectory_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    scored_frames: tuple[int, ...],
) -> dict[str, object]:
    """Score only identities that were never used as measurements."""

    trajectory = np.asarray(trajectory_m, dtype=float)
    target = np.asarray(target_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    if trajectory.shape != target.shape or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("trajectory and target must share shape (T, N, 3)")
    if visible.shape != target.shape[:2] or valid.shape != target.shape[:2]:
        raise ValueError("visibility and validity must have shape (T, N)")
    centers = np.asarray(center_ids, dtype=np.int64)
    if centers.ndim != 1 or len(np.unique(centers)) != len(centers):
        raise ValueError("center_ids must be a unique vector")
    if np.any(centers < 0) or np.any(centers >= target.shape[1]):
        raise ValueError("center ID exceeds the material trajectory")
    hidden_identity = np.ones(target.shape[1], dtype=bool)
    hidden_identity[centers] = False
    if not np.any(hidden_identity):
        raise ValueError("centre exclusion leaves no hidden material identity")

    identity_by_frame: list[float] = []
    chamfer_by_frame: list[float] = []
    hidden_count_by_frame: list[int] = []
    for frame in scored_frames:
        if not 0 <= frame < len(target):
            raise ValueError("scored frame exceeds the trajectory")
        mask = (
            hidden_identity
            & visible[frame]
            & valid[frame]
            & np.all(np.isfinite(target[frame]), axis=1)
            & np.all(np.isfinite(trajectory[frame]), axis=1)
        )
        if not np.any(mask):
            raise ValueError(f"no supported hidden identities at frame {frame}")
        predicted_hidden = trajectory[frame, mask]
        target_hidden = target[frame, mask]
        residual = predicted_hidden - target_hidden
        identity_by_frame.append(float(np.sqrt(np.mean(np.square(residual)))))
        chamfer_by_frame.append(
            _symmetric_euclidean_chamfer_m(predicted_hidden, target_hidden)
        )
        hidden_count_by_frame.append(int(np.sum(mask)))

    if not identity_by_frame:
        raise ValueError("no scored frame")
    return {
        "frame_count": len(scored_frames),
        "scored_frames": list(scored_frames),
        "permanently_excluded_center_count": int(len(centers)),
        "post_update_hidden_identity_rmse_m": float(np.mean(identity_by_frame)),
        "post_update_hidden_symmetric_chamfer_m": float(np.mean(chamfer_by_frame)),
        "hidden_identity_count_per_frame": {
            "minimum": int(np.min(hidden_count_by_frame)),
            "mean": float(np.mean(hidden_count_by_frame)),
            "maximum": int(np.max(hidden_count_by_frame)),
        },
        "by_frame": {
            "hidden_identity_rmse_m": identity_by_frame,
            "hidden_symmetric_chamfer_m": chamfer_by_frame,
        },
    }


def _corrected_frame(
    physical_prior_frame: np.ndarray,
    correction_m: np.ndarray,
    *,
    dtype: np.dtype[Any],
) -> np.ndarray:
    return (
        np.asarray(physical_prior_frame, dtype=float)
        + np.asarray(correction_m, dtype=float)
    ).astype(dtype, copy=False)


def evaluate_deform360_online_belief_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    target_m: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    *,
    measurement_m: np.ndarray | None = None,
    measurement_visibility: np.ndarray | None = None,
    measurement_validity: np.ndarray | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Apply the fixed online protocol to one registered material trajectory."""

    prior_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    if not np.issubdtype(prior_input.dtype, np.floating):
        prior_input = prior_input.astype(np.float64)
    if not np.issubdtype(persistence_input.dtype, np.floating):
        persistence_input = persistence_input.astype(np.float64)
    prior = prior_input.copy()
    persistence = persistence_input.copy()
    target = np.asarray(target_m, dtype=float)
    visible = np.asarray(visibility, dtype=bool)
    valid = np.asarray(validity, dtype=bool)
    measurement = (
        target if measurement_m is None else np.asarray(measurement_m, dtype=float)
    )
    measurement_visible = (
        visible
        if measurement_visibility is None
        else np.asarray(measurement_visibility, dtype=bool)
    )
    measurement_valid = (
        valid
        if measurement_validity is None
        else np.asarray(measurement_validity, dtype=bool)
    )
    if (
        prior.shape != persistence.shape
        or prior.shape != target.shape
        or target.ndim != 3
        or target.shape[2] != 3
    ):
        raise ValueError("prior, persistence, and target must share shape (T, N, 3)")
    if visible.shape != target.shape[:2] or valid.shape != target.shape[:2]:
        raise ValueError("visibility and validity must have shape (T, N)")
    if measurement.shape != target.shape:
        raise ValueError("measurement_m must match target shape")
    if (
        measurement_visible.shape != target.shape[:2]
        or measurement_valid.shape != target.shape[:2]
    ):
        raise ValueError("measurement masks must have shape (T, N)")
    if target.shape[1] < CENTER_COUNT:
        raise ValueError("trajectory has fewer than 16 material identities")
    scored_frames = _post_update_scored_frames(len(target))
    if not np.array_equal(prior[0].astype(np.float32), target[0].astype(np.float32)):
        raise ValueError("physical prior and target frame-zero identities differ")

    frame_zero_candidates = np.flatnonzero(
        measurement_visible[0]
        & measurement_valid[0]
        & np.all(np.isfinite(prior[0]), axis=1)
        & np.all(np.isfinite(measurement[0]), axis=1)
    )
    if len(frame_zero_candidates) < CENTER_COUNT:
        raise ValueError("too few supported frame-zero identities for 16 centres")
    centers = deterministic_farthest_point_ids(
        prior[0], frame_zero_candidates, CENTER_COUNT
    )
    config = belief_config or RecursiveRbfBeliefConfig()
    global_config = replace(config, local_blend=0.0)
    risk_belief = initialize_recursive_rbf_belief(
        centers,
        prior[0, centers],
        prior[0],
        config=config,
    )
    ungated_belief = initialize_recursive_rbf_belief(
        centers,
        prior[0, centers],
        prior[0],
        config=config,
    )
    correspondence_safe_belief = initialize_recursive_rbf_belief(
        centers,
        prior[0, centers],
        prior[0],
        config=config,
    )

    output_dtype = prior.dtype
    risk_trajectory = prior.copy()
    ungated_trajectory = prior.copy()
    global_trajectory = prior.copy()
    frozen_current_trajectory = prior.copy()
    causal_continuation_trajectory = prior.copy()
    correspondence_safe_trajectory = prior.copy()
    risk_variance = np.full(prior.shape, np.nan, dtype=np.float32)
    ungated_variance = np.full(prior.shape, np.nan, dtype=np.float32)
    conformal_half_width = {
        coverage: np.full(len(prior), np.nan, dtype=np.float32)
        for coverage in CONFORMAL_STYLE_COVERAGES
    }
    update_records: list[dict[str, object]] = []
    history_dispersion = _frozen_history_dispersion_m(
        prior,
        measurement,
        measurement_visible,
        measurement_valid,
        centers,
        first_update_frame=UPDATE_FRAMES[0],
    )
    dispersion_threshold, history_p95 = _risk_dispersion_threshold_m(history_dispersion)
    last_trusted_observation_frame = 0

    for update_index, update in enumerate(UPDATE_FRAMES):
        stop = (
            UPDATE_FRAMES[update_index + 1]
            if update_index + 1 < len(UPDATE_FRAMES)
            else len(target)
        )
        available = (
            measurement_visible[update, centers]
            & measurement_valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(prior[update, centers]), axis=1)
        )
        available_count = int(np.sum(available))
        residual = np.full((len(centers), 3), np.nan, dtype=float)
        residual[available] = (
            measurement[update, centers[available]] - prior[update, centers[available]]
        )
        current_radial = (
            np.empty(0, dtype=float)
            if available_count == 0
            else _radial_residuals_m(residual[available])
        )
        current_dispersion = (
            None if not len(current_radial) else float(np.median(current_radial))
        )
        has_support = available_count >= MINIMUM_UPDATE_CENTER_COUNT
        has_coherent_residual = (
            current_dispersion is not None
            and current_dispersion <= dispersion_threshold
        )
        accepted = has_support and has_coherent_residual
        correspondence_safe_inlier_count = int(
            np.sum(current_radial <= dispersion_threshold)
        )
        correspondence_safe_accepted = (
            has_support
            and has_coherent_residual
            and correspondence_safe_inlier_count >= CORRESPONDENCE_SAFE_INLIER_COUNT
        )

        ungated_reliability = np.zeros(len(centers), dtype=float)
        if available_count:
            ungated_belief, ungated_reliability = update_recursive_rbf_belief(
                ungated_belief,
                update,
                prior[update, centers],
                residual,
                available,
                config=config,
            )
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    ungated_belief,
                    prior[update],
                    forecast_frames=frame - update,
                    config=config,
                )
                ungated_trajectory[frame] = _corrected_frame(
                    prior[frame], decoded.mean_m, dtype=output_dtype
                )
                ungated_variance[frame] = decoded.variance_m2.astype(np.float32)

        risk_reliability = np.zeros(len(centers), dtype=float)
        interval_half_widths: dict[str, float] | None = None
        continuation_gain: float | None = None
        continuation_selected: bool | None = None
        continuation_support_count = 0
        previous_trusted_observation_frame: int | None = None
        if accepted:
            previous_trusted_observation_frame = last_trusted_observation_frame
            continuation_support = (
                available
                & measurement_visible[last_trusted_observation_frame, centers]
                & measurement_valid[last_trusted_observation_frame, centers]
                & np.all(
                    np.isfinite(measurement[last_trusted_observation_frame, centers]),
                    axis=1,
                )
                & np.all(
                    np.isfinite(prior[last_trusted_observation_frame, centers]),
                    axis=1,
                )
            )
            continuation_support_count = int(np.sum(continuation_support))
            continuation_ids = centers[continuation_support]
            continuation_gain = robust_huber_continuation_gain(
                prior[update, continuation_ids]
                - prior[last_trusted_observation_frame, continuation_ids],
                measurement[update, continuation_ids]
                - measurement[last_trusted_observation_frame, continuation_ids],
                fallback=0.0,
            )
            continuation_selected = continuation_gain > CONTINUATION_GAIN_THRESHOLD
            interval_half_widths = {
                f"{coverage:.2f}": finite_sample_absolute_residual_quantile_m(
                    residual,
                    available,
                    coverage,
                )
                for coverage in CONFORMAL_STYLE_COVERAGES
            }
            risk_belief, risk_reliability = update_recursive_rbf_belief(
                risk_belief,
                update,
                prior[update, centers],
                residual,
                available,
                config=config,
            )
            frozen_decoded = decode_recursive_rbf_belief(
                risk_belief,
                prior[update],
                forecast_frames=0,
                config=config,
            )
            frozen_state = _corrected_frame(
                prior[update], frozen_decoded.mean_m, dtype=output_dtype
            )
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    risk_belief,
                    prior[update],
                    forecast_frames=frame - update,
                    config=config,
                )
                global_decoded = decode_recursive_rbf_belief(
                    risk_belief,
                    prior[update],
                    forecast_frames=frame - update,
                    config=global_config,
                )
                risk_trajectory[frame] = _corrected_frame(
                    prior[frame], decoded.mean_m, dtype=output_dtype
                )
                global_trajectory[frame] = _corrected_frame(
                    prior[frame], global_decoded.mean_m, dtype=output_dtype
                )
                frozen_current_trajectory[frame] = frozen_state
                causal_continuation_trajectory[frame] = (
                    risk_trajectory[frame] if continuation_selected else frozen_state
                )
                risk_variance[frame] = decoded.variance_m2.astype(np.float32)
                for coverage in CONFORMAL_STYLE_COVERAGES:
                    conformal_half_width[coverage][frame] = interval_half_widths[
                        f"{coverage:.2f}"
                    ]
            last_trusted_observation_frame = update
        else:
            # These checks make exact fallback part of the executable contract.
            for name, trajectory in {
                "recursive_rbf_risk_limited": risk_trajectory,
                "recursive_global_translation": global_trajectory,
                "recursive_rbf_causal_continuation": (causal_continuation_trajectory),
                "risk_limited_frozen_current_state": frozen_current_trajectory,
            }.items():
                if not np.array_equal(
                    trajectory[update + 1 : stop], prior[update + 1 : stop]
                ):
                    raise AssertionError(f"{name} violated exact prior fallback")

        if correspondence_safe_accepted:
            correspondence_safe_belief, _ = update_recursive_rbf_belief(
                correspondence_safe_belief,
                update,
                prior[update, centers],
                residual,
                available,
                config=config,
            )
            for frame in range(update + 1, stop):
                safe_decoded = decode_recursive_rbf_belief(
                    correspondence_safe_belief,
                    prior[update],
                    forecast_frames=frame - update,
                    config=config,
                )
                correspondence_safe_trajectory[frame] = _corrected_frame(
                    prior[frame],
                    safe_decoded.mean_m,
                    dtype=output_dtype,
                )
        elif not np.array_equal(
            correspondence_safe_trajectory[update + 1 : stop],
            prior[update + 1 : stop],
        ):
            raise AssertionError(
                "recursive_rbf_correspondence_safe violated exact prior fallback"
            )

        risk_selected = risk_reliability[available]
        ungated_selected = ungated_reliability[available]
        update_records.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "available_center_count": available_count,
                "available_center_ids": centers[available].tolist(),
                "strict_majority_required": MINIMUM_UPDATE_CENTER_COUNT,
                "current_median_radial_residual_m": current_dispersion,
                "frozen_history_frame_range_half_open": [0, UPDATE_FRAMES[0]],
                "frozen_history_dispersion_count": int(len(history_dispersion)),
                "frozen_history_p95_dispersion_m": history_p95,
                "maximum_median_radial_residual_m": dispersion_threshold,
                "accepted": accepted,
                "conformal_style_absolute_residual_half_width_m": (
                    interval_half_widths
                ),
                "previous_trusted_observation_frame": (
                    previous_trusted_observation_frame
                ),
                "continuation_projection_center_count": (continuation_support_count),
                "causal_huber_continuation_gain": continuation_gain,
                "causal_continuation_gain_threshold": (CONTINUATION_GAIN_THRESHOLD),
                "causal_continuation_selected": continuation_selected,
                "correspondence_safe_inlier_count": (correspondence_safe_inlier_count),
                "correspondence_safe_required_inlier_count": (
                    CORRESPONDENCE_SAFE_INLIER_COUNT
                ),
                "correspondence_safe_accepted": correspondence_safe_accepted,
                "correspondence_safe_decision": (
                    "accepted"
                    if correspondence_safe_accepted
                    else "exact_prior_fallback"
                ),
                "decision": (
                    "accepted"
                    if accepted
                    else (
                        "insufficient_support_exact_prior"
                        if not has_support
                        else "incoherent_residual_exact_prior"
                    )
                ),
                "risk_mean_reliability": (
                    None
                    if not accepted or not len(risk_selected)
                    else float(np.mean(risk_selected))
                ),
                "ungated_mean_reliability": (
                    None
                    if not available_count or not len(ungated_selected)
                    else float(np.mean(ungated_selected))
                ),
            }
        )

    trajectories = {
        "physical_prior": prior,
        "persistence": persistence,
        "recursive_global_translation": global_trajectory,
        "recursive_rbf_ungated": ungated_trajectory,
        "recursive_rbf_risk_limited": risk_trajectory,
        "recursive_rbf_causal_continuation": causal_continuation_trajectory,
        "recursive_rbf_correspondence_safe": correspondence_safe_trajectory,
        "risk_limited_frozen_current_state": frozen_current_trajectory,
    }
    scores = {
        name: score_deform360_hidden_trajectory(
            trajectory,
            target,
            visible,
            valid,
            center_ids=centers,
            scored_frames=scored_frames,
        )
        for name, trajectory in trajectories.items()
    }
    report: dict[str, object] = {
        "protocol_id": PROTOCOL_ID,
        "center_ids": centers.tolist(),
        "center_count": int(len(centers)),
        "update_frames": list(UPDATE_FRAMES),
        "scored_frames": list(scored_frames),
        "risk_gate": {
            "minimum_update_center_count": MINIMUM_UPDATE_CENTER_COUNT,
            "support_rule": "strict majority of 16 deterministic FPS centres",
            "dispersion_statistic": (
                "median radial residual around the current coordinate-wise median"
            ),
            "history_rule": (
                "p95 of one median-radial-residual scalar per supported frame in "
                "the frozen [0, 19) history; each frame requires at least three "
                "supported centres"
            ),
            "frozen_history_frame_range_half_open": [0, UPDATE_FRAMES[0]],
            "frozen_history_dispersion_count": int(len(history_dispersion)),
            "frozen_history_p95_dispersion_m": history_p95,
            "maximum_median_radial_residual_m": dispersion_threshold,
            "history_quantile": HISTORY_DISPERSION_QUANTILE,
            "history_multiplier": HISTORY_DISPERSION_MULTIPLIER,
            "minimum_history_point_count": MINIMUM_HISTORY_POINT_COUNT,
            "minimum_threshold_m": MINIMUM_DISPERSION_THRESHOLD_M,
            "rejected_interval_behavior": "bit-exact sealed physical prior",
            "accepted_update_count": int(
                sum(bool(record["accepted"]) for record in update_records)
            ),
        },
        "updates": update_records,
        "scores": scores,
        "metric_contract": {
            "identity": (
                "per-frame coordinate RMSE over visible, valid, permanently hidden "
                "material identities, then frame mean"
            ),
            "chamfer": (
                "per-frame symmetric mean Euclidean nearest-neighbour distance "
                "between hidden predicted and hidden target identities, then frame mean"
            ),
            "assimilation_center_exclusion": (
                "all 16 centre identities are excluded from both metric directions "
                "at every scored frame"
            ),
        },
        "belief_config": asdict(config),
        "observation_contract": {
            "measurement_stream_is_scoring_target": (
                measurement_m is None
                and measurement_visibility is None
                and measurement_validity is None
            ),
            "separation": (
                "filter updates, gate decisions, continuation selection, and "
                "conformal-style widths consume only the measurement stream; "
                "metrics consume only the scoring target"
            ),
        },
        "causal_continuation_contract": {
            "hybrid": (
                "corrected current state plus selected physical-prior future "
                "displacement"
            ),
            "gain": (
                "Huber-IRLS scalar projection of currently observed sparse "
                "displacement onto sealed physical-prior displacement since the "
                "previous accepted observation"
            ),
            "threshold": CONTINUATION_GAIN_THRESHOLD,
            "decision": "continue iff gain is strictly greater than threshold",
            "insufficient_support_gain": 0.0,
            "status": (
                "post-hoc development arm; threshold was inspected after the "
                "open PhysTwin22 and Deform360-27 outcomes"
            ),
        },
        "correspondence_safe_contract": {
            "inlier": (
                "current radial residual around the coordinate-wise median is "
                "no larger than the frozen dispersion threshold"
            ),
            "required_inliers": CORRESPONDENCE_SAFE_INLIER_COUNT,
            "selected_center_count": CENTER_COUNT,
            "rejected_interval": "bit-exact sealed physical prior",
            "status": (
                "post-hoc risk-coverage control; not a replacement under natural "
                "occlusion and requires newly held-out transfer"
            ),
        },
        "uncertainty_contract": {
            "raw_variance": (
                "coordinate-wise marginal decoder variance; not calibrated"
            ),
            "conformal_style_interval": (
                "recursive mean plus/minus the finite-sample absolute-residual "
                "quantile from currently available centre coordinates"
            ),
            "finite_sample_rank": "min(n, ceil((n + 1) * nominal_coverage))",
            "nominal_coverages": list(CONFORMAL_STYLE_COVERAGES),
            "dependence_warning": (
                "coordinates share frames and material identities; this is a "
                "conformal-style diagnostic, not a formal iid coverage guarantee"
            ),
            "rejected_interval": "no interval; exact physical-prior fallback",
        },
    }
    arrays = {
        "center_ids": centers,
        "physical_prior_m": prior,
        "persistence_m": persistence,
        "recursive_global_translation_m": global_trajectory,
        "recursive_rbf_ungated_m": ungated_trajectory,
        "recursive_rbf_risk_limited_m": risk_trajectory,
        "recursive_rbf_causal_continuation_m": causal_continuation_trajectory,
        "recursive_rbf_correspondence_safe_m": correspondence_safe_trajectory,
        "risk_limited_frozen_current_state_m": frozen_current_trajectory,
        "recursive_rbf_risk_limited_variance_m2": risk_variance,
        "recursive_rbf_ungated_variance_m2": ungated_variance,
        "recursive_rbf_risk_limited_conformal_q50_half_width_m": (
            conformal_half_width[0.50]
        ),
        "recursive_rbf_risk_limited_conformal_q90_half_width_m": (
            conformal_half_width[0.90]
        ),
        "recursive_rbf_risk_limited_conformal_q95_half_width_m": (
            conformal_half_width[0.95]
        ),
    }
    return report, arrays


def _resolve_prediction_archive(
    episode_dir: Path,
    prediction_seal: Mapping[str, Any],
) -> Path:
    archive = prediction_seal.get("prediction_archive", {})
    declared = Path(str(archive.get("path", "")))
    candidates = (declared, episode_dir / declared.name)
    for candidate in candidates:
        if candidate.is_file():
            expected = archive.get("file_sha256")
            if expected is not None and _sha256(candidate) != expected:
                raise ValueError(f"prediction archive checksum changed: {candidate}")
            return candidate.resolve()
    raise FileNotFoundError(declared)


def _validate_deform360_outcome_manifest(
    seal_path: Path,
    target_path: Path,
    prediction_seal: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> None:
    """Bind an opened outcome to its sealed prediction and target payload."""

    if outcome.get("artifact_kind") != "Deform360IndependentSourceOutcome":
        raise ValueError("unsupported Deform360 source outcome")
    for key in ("object_id", "episode_id", "episode_key"):
        if outcome.get(key) != prediction_seal.get(key):
            raise ValueError(f"source outcome {key} differs from prediction seal")

    expected_seal_sha256 = outcome.get("input_sha256", {}).get("prediction_seal")
    if not isinstance(expected_seal_sha256, str) or not expected_seal_sha256:
        raise ValueError("source outcome lacks the prediction-seal checksum")
    if _sha256(seal_path) != expected_seal_sha256:
        raise ValueError("source outcome refers to a different prediction seal")

    expected_target_sha256 = outcome.get("output_sha256", {}).get("target_data")
    if not isinstance(expected_target_sha256, str) or not expected_target_sha256:
        raise ValueError("source outcome lacks the target-data checksum")
    if _sha256(target_path) != expected_target_sha256:
        raise ValueError("source target-data checksum changed")


def evaluate_deform360_online_belief_case(
    episode_dir: str | Path,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load and evaluate one already-open independent-source episode."""

    directory = Path(episode_dir).resolve()
    seal_path = directory / "prediction_seal.json"
    target_path = directory / "target_data.pkl"
    outcome_path = directory / "outcome.json"
    for path in (seal_path, target_path, outcome_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("artifact_kind") != "Deform360IndependentSourcePredictionSeal":
        raise ValueError("unsupported Deform360 prediction seal")
    boundary = seal.get("information_boundary", {})
    if not (
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("future_object_track_read") is False
        and boundary.get("prediction_hashed_before_future_outcome_scoring") is True
    ):
        raise ValueError("physical prior crossed the source-future boundary")
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    if (
        outcome.get("information_boundary", {}).get(
            "source_future_opened_for_outcome_construction"
        )
        is not True
    ):
        raise ValueError("source outcome has not been opened")
    _validate_deform360_outcome_manifest(seal_path, target_path, seal, outcome)

    archive_path = _resolve_prediction_archive(directory, seal)
    with np.load(archive_path, allow_pickle=False) as stored:
        required = {
            "prediction_m",
            "persistence_m",
            "frame_zero_points_m",
        }
        if not required.issubset(stored.files):
            raise ValueError("sealed archive lacks required prediction arrays")
        physical_prior = np.asarray(stored["prediction_m"]).copy()
        persistence = np.asarray(stored["persistence_m"]).copy()
        frame_zero = np.asarray(stored["frame_zero_points_m"]).copy()
    target_data = _load_pickle(target_path)
    target = np.asarray(target_data["object_points"])
    visible = np.asarray(target_data["object_visibilities"], dtype=bool)
    valid = np.asarray(target_data["object_motions_valid"], dtype=bool)
    if len(target) != 76:
        raise ValueError("fixed Deform360 source protocol requires 76 frames")
    if not np.array_equal(target[0].astype(np.float32), frame_zero.astype(np.float32)):
        raise ValueError("sealed frame-zero identities differ from the source outcome")

    report, arrays = evaluate_deform360_online_belief_arrays(
        physical_prior,
        persistence,
        target,
        visible,
        valid,
    )
    report.update(
        {
            "case": directory.name,
            "object_id": str(seal["object_id"]),
            "episode_id": int(seal["episode_id"]),
            "episode_key": str(seal["episode_key"]),
            "inputs": {
                "prediction_seal": {
                    "path": str(seal_path),
                    "sha256": _sha256(seal_path),
                },
                "prediction_archive": {
                    "path": str(archive_path),
                    "sha256": _sha256(archive_path),
                },
                "target_data": {
                    "path": str(target_path),
                    "sha256": _sha256(target_path),
                },
                "outcome": {
                    "path": str(outcome_path),
                    "sha256": _sha256(outcome_path),
                },
            },
            "information_boundary": {
                "physical_prediction_sealed_before_source_future": True,
                "source_future_already_open": True,
                "official_target_outcome_used": False,
                "online_measurement_kind": (
                    "sparse identities from the multiview-fused material trajectory"
                ),
            },
        }
    )
    return report, arrays


def _physical_object_cluster_bootstrap(
    differences: Mapping[str, float],
    groups: Mapping[str, str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Bootstrap equal-weight physical-object cluster means."""

    object_ids = tuple(sorted(set(groups.values())))
    if len(object_ids) < 2:
        raise ValueError("physical-object bootstrap requires multiple objects")
    group_means = {
        object_id: float(
            np.mean(
                [
                    differences[case]
                    for case, group in groups.items()
                    if group == object_id
                ]
            )
        )
        for object_id in object_ids
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=float)
    means = np.asarray([group_means[value] for value in object_ids], dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, len(means), size=len(means))
        bootstrap[draw] = float(np.mean(means[selected]))
    return {
        "episode_mean_difference_m": float(np.mean(list(differences.values()))),
        "object_balanced_mean_difference_m": float(np.mean(means)),
        "object_cluster_lower_95_m": float(np.quantile(bootstrap, 0.025)),
        "object_cluster_upper_95_m": float(np.quantile(bootstrap, 0.975)),
        "object_cluster_probability_improved": float(np.mean(bootstrap < 0.0)),
    }


def _relative_change(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return candidate / baseline - 1.0


def evaluate_deform360_online_belief_cohort(
    root: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Evaluate and persist the fixed, open 27-episode Deform360 panel."""

    cohort_root = Path(root).resolve()
    output_dir = Path(output).resolve()
    expected = _expected_episode_directories()
    if len(expected) != 27:
        raise AssertionError("fixed source panel no longer contains 27 episodes")
    missing = [name for name in expected if not (cohort_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing fixed Deform360 episodes: {missing}")
    output_dir.mkdir(parents=True, exist_ok=False)

    reports: list[dict[str, object]] = []
    groups: dict[str, str] = {}
    artifacts: list[dict[str, object]] = []
    for case_name in expected:
        report, arrays = evaluate_deform360_online_belief_case(cohort_root / case_name)
        groups[case_name] = str(report["object_id"])
        report_path = output_dir / f"{case_name}.json"
        arrays_path = output_dir / f"{case_name}.npz"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(arrays_path, **arrays)
        artifacts.append(
            {
                "case": case_name,
                "report_sha256": _sha256(report_path),
                "arrays_sha256": _sha256(arrays_path),
            }
        )
        reports.append(report)

    aggregate = {
        arm: {
            metric: float(
                np.mean([report["scores"][arm][metric] for report in reports])
            )
            for metric in PRIMARY_METRICS
        }
        for arm in ARMS
    }
    candidate_comparators = {
        "recursive_rbf_risk_limited": (
            "physical_prior",
            "persistence",
            "recursive_global_translation",
            "recursive_rbf_ungated",
            "risk_limited_frozen_current_state",
        ),
        "recursive_rbf_causal_continuation": (
            "physical_prior",
            "persistence",
            "recursive_global_translation",
            "recursive_rbf_risk_limited",
            "risk_limited_frozen_current_state",
        ),
        "recursive_rbf_correspondence_safe": (
            "physical_prior",
            "recursive_rbf_risk_limited",
        ),
    }
    comparisons: dict[str, object] = {}
    for candidate_arm, comparators in candidate_comparators.items():
        for comparator in comparators:
            metrics: dict[str, object] = {}
            for metric in PRIMARY_METRICS:
                differences = {
                    str(report["case"]): float(
                        report["scores"][candidate_arm][metric]
                        - report["scores"][comparator][metric]
                    )
                    for report in reports
                }
                result = _physical_object_cluster_bootstrap(differences, groups)
                result["relative_change"] = _relative_change(
                    aggregate[candidate_arm][metric], aggregate[comparator][metric]
                )
                result["episode_wins"] = int(
                    np.sum(np.asarray(list(differences.values())) < 0.0)
                )
                relative_regressions = [
                    _relative_change(
                        float(report["scores"][candidate_arm][metric]),
                        float(report["scores"][comparator][metric]),
                    )
                    for report in reports
                ]
                finite_regressions = [
                    value for value in relative_regressions if value is not None
                ]
                result["maximum_episode_relative_regression"] = (
                    None if not finite_regressions else float(max(finite_regressions))
                )
                metrics[metric] = result
            comparisons[f"{candidate_arm}_vs_{comparator}"] = {
                "metrics": metrics,
                "joint_two_metric_episode_wins": int(
                    sum(
                        all(
                            report["scores"][candidate_arm][metric]
                            < report["scores"][comparator][metric]
                            for metric in PRIMARY_METRICS
                        )
                        for report in reports
                    )
                ),
            }

    accepted_updates = int(
        sum(
            bool(update["accepted"])
            for report in reports
            for update in report["updates"]
        )
    )
    correspondence_safe_accepted_updates = int(
        sum(
            bool(update["correspondence_safe_accepted"])
            for report in reports
            for update in report["updates"]
        )
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cohort_root": str(cohort_root),
        "episode_count": len(reports),
        "physical_object_count": len(set(groups.values())),
        "physical_objects": {
            key: list(value) for key, value in EXPECTED_SOURCE_EPISODES.items()
        },
        "fixed_protocol": {
            "center_count": CENTER_COUNT,
            "center_selection": "deterministic frame-zero farthest-point sampling",
            "update_frames": list(UPDATE_FRAMES),
            "minimum_update_center_count": MINIMUM_UPDATE_CENTER_COUNT,
            "frozen_history_frame_range_half_open": [0, UPDATE_FRAMES[0]],
            "minimum_dispersion_threshold_m": MINIMUM_DISPERSION_THRESHOLD_M,
            "history_dispersion_quantile": HISTORY_DISPERSION_QUANTILE,
            "history_dispersion_multiplier": HISTORY_DISPERSION_MULTIPLIER,
            "minimum_history_point_count": MINIMUM_HISTORY_POINT_COUNT,
            "causal_continuation_gain_threshold": CONTINUATION_GAIN_THRESHOLD,
            "causal_continuation_selector_status": (
                "retrospective development; requires a newly held-out transfer"
            ),
            "correspondence_safe_required_inlier_count": (
                CORRESPONDENCE_SAFE_INLIER_COUNT
            ),
            "correspondence_safe_status": (
                "post-hoc exact-safety control; requires newly held-out transfer"
            ),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "belief_config": asdict(RecursiveRbfBeliefConfig()),
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "risk_gate": {
            "accepted_update_count": accepted_updates,
            "total_update_count": len(reports) * len(UPDATE_FRAMES),
            "acceptance_fraction": accepted_updates
            / (len(reports) * len(UPDATE_FRAMES)),
            "correspondence_safe_accepted_update_count": (
                correspondence_safe_accepted_updates
            ),
            "correspondence_safe_acceptance_fraction": (
                correspondence_safe_accepted_updates
                / (len(reports) * len(UPDATE_FRAMES))
            ),
        },
        "artifacts": artifacts,
        "claim_boundary": (
            "open independent-source Deform360 transfer with fused material-track "
            "measurements; the causal continuation threshold is post-hoc; not an "
            "official held-target or Table-4 SOTA result"
        ),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = [
    "EXPECTED_SOURCE_EPISODES",
    "PROTOCOL_ID",
    "evaluate_deform360_online_belief_arrays",
    "evaluate_deform360_online_belief_case",
    "evaluate_deform360_online_belief_cohort",
    "score_deform360_hidden_trajectory",
]
