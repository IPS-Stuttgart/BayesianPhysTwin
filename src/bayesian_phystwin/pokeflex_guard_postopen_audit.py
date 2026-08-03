"""Post-open diagnostics for an already sealed PokeFlex guard result."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _policy_summary(
    rows: Sequence[Mapping[str, Any]],
    keep: np.ndarray,
) -> dict[str, Any]:
    by_object: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_object[str(row["object_name"])].append(index)
    baseline_means = []
    candidate_means = []
    per_object = []
    for object_name in sorted(by_object):
        indices = np.asarray(by_object[object_name], dtype=np.int64)
        baseline = np.asarray(
            [float(rows[index]["baseline_error_mm"]) for index in indices],
            dtype=np.float64,
        )
        sealed_candidate = np.asarray(
            [float(rows[index]["candidate_error_mm"]) for index in indices],
            dtype=np.float64,
        )
        selected = np.where(keep[indices], sealed_candidate, baseline)
        baseline_mean = float(np.mean(baseline))
        candidate_mean = float(np.mean(selected))
        baseline_means.append(baseline_mean)
        candidate_means.append(candidate_mean)
        per_object.append(
            {
                "object_name": object_name,
                "baseline_mean_error_mm": baseline_mean,
                "candidate_mean_error_mm": candidate_mean,
                "difference_mm": candidate_mean - baseline_mean,
                "kept_frame_count": int(np.sum(keep[indices])),
            }
        )
    baseline_array = np.asarray(baseline_means, dtype=np.float64)
    candidate_array = np.asarray(candidate_means, dtype=np.float64)
    difference = candidate_array - baseline_array
    tolerance = 1e-12
    wins = int(np.sum(difference < -tolerance))
    losses = int(np.sum(difference > tolerance))
    ties = len(difference) - wins - losses
    baseline_mean = float(np.mean(baseline_array))
    candidate_mean = float(np.mean(candidate_array))
    return {
        "baseline_object_balanced_error_mm": baseline_mean,
        "candidate_object_balanced_error_mm": candidate_mean,
        "object_balanced_relative_improvement": (
            float((baseline_mean - candidate_mean) / baseline_mean)
        ),
        "object_win_count": wins,
        "object_tie_count": ties,
        "object_loss_count": losses,
        "kept_frame_count": int(np.sum(keep)),
        "kept_object_count": int(
            sum(row["kept_frame_count"] > 0 for row in per_object)
        ),
        "objects": per_object,
    }


def audit_sealed_guard_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit stricter subsets of sealed candidate updates after outcomes open."""

    _require(bool(rows), "guard audit has no scored rows")
    accepted = np.asarray([bool(row["accepted"]) for row in rows], dtype=np.bool_)
    baseline = np.asarray(
        [float(row["baseline_error_mm"]) for row in rows], dtype=np.float64
    )
    candidate = np.asarray(
        [float(row["candidate_error_mm"]) for row in rows], dtype=np.float64
    )
    upper = np.asarray(
        [float(row["upper_regret_mm"]) if row["accepted"] else np.nan for row in rows],
        dtype=np.float64,
    )
    _require(np.all(np.isfinite(baseline)), "baseline errors are non-finite")
    _require(np.all(np.isfinite(candidate)), "candidate errors are non-finite")
    _require(np.all(np.isfinite(upper[accepted])), "accepted upper regret is non-finite")
    _require(np.all(upper[accepted] < 0.0), "sealed guard accepted a nonnegative bound")
    _require(
        np.array_equal(candidate[~accepted], baseline[~accepted]),
        "sealed fallback is not exact",
    )

    actual_regret = candidate - baseline
    accepted_actual = actual_regret[accepted]
    accepted_upper = upper[accepted]
    _require(bool(len(accepted_upper)), "guard audit has no accepted scored rows")
    unique_cutoffs = sorted({float(value) for value in accepted_upper})
    empty_cutoff = float(np.nextafter(unique_cutoffs[0], -np.inf))
    cutoffs = [empty_cutoff, *unique_cutoffs, 0.0]
    policies = []
    for cutoff in cutoffs:
        keep = accepted & (upper < cutoff)
        summary = _policy_summary(rows, keep)
        summary["upper_regret_cutoff_mm"] = cutoff
        policies.append(summary)

    zero_loss = [policy for policy in policies if policy["object_loss_count"] == 0]
    best_zero_loss = max(
        zero_loss,
        key=lambda policy: (
            int(policy["object_win_count"]),
            float(policy["object_balanced_relative_improvement"]),
        ),
    )
    frame_oracle_keep = accepted & (actual_regret < 0.0)
    frame_oracle = _policy_summary(rows, frame_oracle_keep)
    affected_objects = sorted(
        {str(row["object_name"]) for row in rows if bool(row["accepted"])}
    )
    all_objects = sorted({str(row["object_name"]) for row in rows})
    return {
        "scored_frame_count": len(rows),
        "accepted_scored_frame_count": int(np.sum(accepted)),
        "accepted_improving_frame_count": int(np.sum(accepted_actual < 0.0)),
        "accepted_tied_frame_count": int(np.sum(accepted_actual == 0.0)),
        "accepted_harmful_frame_count": int(np.sum(accepted_actual > 0.0)),
        "accepted_false_safe_rate": float(np.mean(accepted_actual > 0.0)),
        "accepted_upper_bound_coverage": float(
            np.mean(accepted_actual <= accepted_upper)
        ),
        "objects_with_sealed_candidate_effect": affected_objects,
        "objects_without_sealed_candidate_effect": sorted(
            set(all_objects) - set(affected_objects)
        ),
        "current_policy": next(
            policy for policy in policies if policy["upper_regret_cutoff_mm"] == 0.0
        ),
        "best_zero_loss_stricter_policy": best_zero_loss,
        "maximum_zero_loss_win_count_from_sealed_candidates": int(
            best_zero_loss["object_win_count"]
        ),
        "frame_oracle_within_sealed_candidates": frame_oracle,
        "stricter_policy_count": len(policies),
        "stricter_policies": policies,
        "claim_boundary": (
            "Post-open diagnostic over subsets of already sealed candidates; "
            "it cannot support a prospective claim or authorize retuning."
        ),
    }
