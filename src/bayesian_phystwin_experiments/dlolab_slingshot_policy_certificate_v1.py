"""Development-only Slingshot adapter for a policy-level gain certificate."""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from bayesian_phystwin.policy_gain_certificate import (
    apply_policy_gain_guard,
    calibrate_policy_gain_lower_bound,
    fit_local_policy_gain_predictor,
    predict_local_policy_gain,
)

from .dlolab_slingshot_belief import BASELINE, ORDER, REWARD_MARGIN

Array: TypeAlias = NDArray[Any]
DEVELOPMENT_COUNT = 51
NEIGHBOR_COUNT = 5
MISCOVERAGE = 0.10


def bias_invariant_features(observation: object) -> NDArray[np.float64]:
    """Return relative geometry and temporal increments from a Slingshot prefix."""

    value = np.asarray(observation, dtype=np.float64)
    if value.ndim not in (3, 4) or value.shape[-3:] != (3, 4, 3):
        raise ValueError("Slingshot observations must end in the registered 3x4x3 layout")
    if not np.all(np.isfinite(value)):
        raise ValueError("Slingshot observations must be finite")
    batch = value.reshape((-1, 3, 4, 3))
    relative = batch[:, :, :3] - batch[:, :, 3:4]
    temporal = np.diff(batch, axis=1)
    result = np.concatenate(
        (relative.reshape(len(batch), -1), temporal.reshape(len(batch), -1)),
        axis=1,
    )
    return result[0] if value.ndim == 3 else result


def posterior_policy_action(expected_losses: object) -> NDArray[np.int64]:
    """Map frozen posterior expected losses to the fixed candidate policy action."""

    value = np.asarray(expected_losses, dtype=np.float64)
    if value.ndim not in (1, 2) or value.shape[-1] != len(ORDER):
        raise ValueError("expected losses must end in the registered action order")
    if not np.all(np.isfinite(value)):
        raise ValueError("expected losses must be finite")
    result = np.asarray(ORDER, dtype=np.int64)[np.argmin(value, axis=-1)]
    return np.asarray(result, dtype=np.int64)


def leave_one_out_capacity_diagnostic(
    *,
    case_ids: tuple[str, ...],
    observations: object,
    expected_losses: object,
    action_gains: object,
) -> dict[str, Any]:
    """Measure opened-data capacity without making a prospective coverage claim."""

    observed = np.asarray(observations, dtype=np.float64)
    losses = np.asarray(expected_losses, dtype=np.float64)
    gains = np.asarray(action_gains, dtype=np.float64)
    if (
        len(case_ids) != DEVELOPMENT_COUNT
        or observed.shape != (DEVELOPMENT_COUNT, 3, 4, 3)
        or losses.shape != (DEVELOPMENT_COUNT, len(ORDER))
        or gains.shape != (DEVELOPMENT_COUNT, 7)
        or len(set(case_ids)) != DEVELOPMENT_COUNT
        or any(not value for value in case_ids)
        or not all(np.all(np.isfinite(value)) for value in (observed, losses, gains))
    ):
        raise ValueError("complete finite opened Slingshot development data required")

    features = bias_invariant_features(observed)
    actions = posterior_policy_action(losses)
    predicted = np.empty(DEVELOPMENT_COUNT, dtype=np.float64)
    neighbor_ids: list[list[str]] = []
    for query in range(DEVELOPMENT_COUNT):
        reference = np.arange(DEVELOPMENT_COUNT) != query
        model = fit_local_policy_gain_predictor(
            reference_ids=tuple(
                case_ids[index] for index in np.flatnonzero(reference)
            ),
            reference_features=features[reference],
            reference_action_gains=gains[reference],
            neighbor_count=NEIGHBOR_COUNT,
        )
        prediction = predict_local_policy_gain(
            model,
            query_features=features[query : query + 1],
            candidate_actions=actions[query : query + 1],
        )
        predicted[query] = prediction.predicted_gain[0]
        neighbor_ids.append(
            [model.reference_ids[index] for index in prediction.neighbor_indices[0]]
        )

    realized = gains[np.arange(DEVELOPMENT_COUNT), actions]
    calibration = calibrate_policy_gain_lower_bound(
        predicted_gain=predicted,
        realized_gain=realized,
        miscoverage=MISCOVERAGE,
    )
    guarded = apply_policy_gain_guard(
        candidate_actions=actions,
        predicted_gain=predicted,
        calibration=calibration,
        fallback_action=BASELINE,
        harm_margin=REWARD_MARGIN,
    )
    guarded_gain = np.where(guarded.accepted_mask, realized, 0.0)
    return {
        "schema": "dlolab-slingshot-policy-certificate-development-v1",
        "status": "retrospective_leave_one_out_capacity_diagnostic_only",
        "prospective_coverage_claim": False,
        "policy_selected_on_calibration_outcomes": False,
        "closed_288_world_panel_used": False,
        "case_count": DEVELOPMENT_COUNT,
        "feature": "shared-bias-invariant-relative-geometry-and-temporal-increments",
        "candidate_policy": "posterior-predictive-mean-action",
        "local_predictor": "standardized-five-nearest-neighbor-mean-gain",
        "miscoverage": calibration.miscoverage,
        "calibration_rank": calibration.rank,
        "calibration_offset": calibration.offset,
        "accepted_count": int(np.count_nonzero(guarded.accepted_mask)),
        "mean_guarded_gain": float(guarded_gain.mean()),
        "harmful_guarded_count": int(
            np.count_nonzero(guarded_gain < -REWARD_MARGIN)
        ),
        "candidate_mean_gain": float(realized.mean()),
        "candidate_harmful_count": int(np.count_nonzero(realized < -REWARD_MARGIN)),
        "case_ids": list(case_ids),
        "candidate_actions": actions.tolist(),
        "predicted_gain": predicted.tolist(),
        "realized_gain": realized.tolist(),
        "lower_gain_bound": guarded.lower_gain_bound.tolist(),
        "accepted_mask": guarded.accepted_mask.tolist(),
        "neighbor_ids": neighbor_ids,
    }
