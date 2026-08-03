"""Guard pairwise Deform360 updates with causal support and regret bounds.

The pairwise-consensus RBF decoder is useful on the open high-motion source
panel, but a prospective action-only study showed that coherent camera bias can
pass its geometry gate and badly damage an already accurate persistence
baseline.  This module keeps that decoder unchanged and adds three admission
layers:

* the camera panel must contain independent identities, while two-view rows
  remain admissible with their lack of three-view redundancy exposed to the
  regret model rather than hidden behind a hard camera-plan gate;
* a non-trivial causal or action-conditioned physical response must exist, and
  the decoded correction is radially shrunk relative to that response;
* a source-calibrated upper bound on regret must be negative.

Rejected intervals return the selected physical/persistence baseline
bit-for-bit.  Candidate construction accepts no target or outcome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .bias_aware_belief import (
    SourceRegretCertificate,
    apply_regret_guard,
)
from .deform360_bias_aware_prospective_artifacts import (
    select_raw_backbone_arrays,
)
from .deform360_selective_virtual_sensing_prediction import (
    predict_persistence_pairwise_rbf_arrays,
)
from .phystwin_online_belief import robust_huber_continuation_gain

FEATURE_NAMES = (
    "physical_response_rms_over_scale",
    "observed_motion_rms_over_scale",
    "future_physical_response_rms_over_scale",
    "innovation_rms_over_scale",
    "correction_rms_over_scale",
    "correction_to_physical_response",
    "physical_to_observed_motion_ratio",
    "physical_agreement_gain",
    "physical_observed_cosine",
    "common_mode_fraction",
    "redundant_center_fraction",
    "median_reprojection_over_scale",
    "pairwise_inlier_fraction",
    "pairwise_compatible_pair_fraction",
    "selected_backbone_is_physical",
    "selected_backbone_chamfer_margin_over_scale",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _radial_rms(value_m: np.ndarray) -> float:
    value = np.asarray(value_m, dtype=np.float64)
    if not len(value):
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))


def _object_scale_m(frame_zero_points_m: np.ndarray) -> float:
    points = np.asarray(frame_zero_points_m, dtype=np.float64)
    center = np.median(points, axis=0)
    return max(1e-6, float(2.0 * np.max(np.linalg.norm(points - center, axis=1))))


def _vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left_flat, right_flat) / denominator, -1.0, 1.0))


@dataclass(frozen=True)
class PairwiseRegretGuardConfig:
    """Target-free eligibility thresholds fixed before a fresh evaluation."""

    update_frames: tuple[int, ...] = (19, 38, 57)
    minimum_camera_panel_count: int = 3
    minimum_triangulation_view_count: int = 2
    redundant_view_count: int = 3
    minimum_redundant_center_count: int = 0
    minimum_motion_center_count: int = 3
    minimum_physical_support_m: float = 0.0005
    minimum_observed_motion_m: float = 0.0005
    minimum_physical_agreement_gain: float = 0.0
    maximum_correction_to_physical_response: float = 2.0
    reprojection_scale_px: float = 3.0

    def __post_init__(self) -> None:
        _require(
            tuple(sorted(set(self.update_frames))) == self.update_frames
            and bool(self.update_frames),
            "update frames must be strictly increasing",
        )
        _require(
            self.minimum_camera_panel_count >= 3,
            "the camera panel must contain at least three identities",
        )
        _require(
            2 <= self.minimum_triangulation_view_count <= self.redundant_view_count,
            "triangulation and redundancy view counts are inconsistent",
        )
        _require(
            self.minimum_redundant_center_count >= 0,
            "redundant centre count must be nonnegative",
        )
        _require(
            self.minimum_motion_center_count >= 1,
            "motion centre count must be positive",
        )
        positive = (
            self.minimum_physical_support_m,
            self.minimum_observed_motion_m,
            self.maximum_correction_to_physical_response,
            self.reprojection_scale_px,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "pairwise regret scales must be positive",
        )
        _require(
            0.0 <= self.minimum_physical_agreement_gain <= 1.0,
            "physical agreement gain must lie in [0, 1]",
        )


def _validate_inputs(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    center_ids: np.ndarray,
    selected_camera_ids: Sequence[str],
    triangulation_inlier_view_count: np.ndarray,
    triangulation_median_reprojection_px: np.ndarray,
    config: PairwiseRegretGuardConfig,
) -> None:
    _require(
        physical_prior_m.ndim == 3 and physical_prior_m.shape[2] == 3,
        "physical prior must have shape (T, N, 3)",
    )
    _require(
        physical_prior_m.shape == persistence_m.shape == measurement_m.shape,
        "trajectory shapes changed",
    )
    _require(
        measurement_visibility.shape
        == measurement_validity.shape
        == physical_prior_m.shape[:2],
        "measurement mask shape changed",
    )
    _require(
        center_ids.ndim == 1
        and len(center_ids) == len(np.unique(center_ids))
        and np.all((center_ids >= 0) & (center_ids < physical_prior_m.shape[1])),
        "center IDs must be unique and in range",
    )
    cameras = tuple(str(value) for value in selected_camera_ids)
    _require(
        len(cameras) == len(set(cameras)),
        "selected camera identities are not independent",
    )
    _require(
        len(cameras) >= config.minimum_camera_panel_count,
        "too few independently named cameras",
    )
    expected = (len(config.update_frames), len(center_ids))
    _require(
        triangulation_inlier_view_count.shape == expected,
        "triangulation inlier-count shape changed",
    )
    _require(
        triangulation_median_reprojection_px.shape == expected,
        "triangulation reprojection shape changed",
    )
    _require(
        np.all(np.isfinite(physical_prior_m))
        and np.all(np.isfinite(persistence_m)),
        "physical trajectory contains non-finite values",
    )
    _require(
        np.all(
            (triangulation_inlier_view_count >= 0)
            & (triangulation_inlier_view_count <= len(cameras))
        ),
        "triangulation view count exceeds the independent camera panel",
    )
    _require(
        config.update_frames[-1] < len(physical_prior_m),
        "update frame exceeds the trajectory",
    )


def build_pairwise_regret_candidate_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    selected_camera_ids: Sequence[str],
    triangulation_inlier_view_count: np.ndarray,
    triangulation_median_reprojection_px: np.ndarray,
    config: PairwiseRegretGuardConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build an eligible pairwise candidate without reading any outcome."""

    cfg = config or PairwiseRegretGuardConfig()
    physical_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    physical = np.asarray(physical_input, dtype=np.float64)
    persistence = np.asarray(persistence_input, dtype=np.float64)
    measurement = np.asarray(measurement_m, dtype=np.float64)
    visible = np.asarray(measurement_visibility, dtype=bool)
    valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    view_count = np.asarray(triangulation_inlier_view_count, dtype=np.int64)
    reprojection = np.asarray(
        triangulation_median_reprojection_px, dtype=np.float64
    )
    cameras = tuple(str(value) for value in selected_camera_ids)
    _validate_inputs(
        physical,
        persistence,
        measurement,
        visible,
        valid,
        centers,
        cameras,
        view_count,
        reprojection,
        cfg,
    )

    baseline_report, baseline = select_raw_backbone_arrays(
        physical_input,
        persistence_input,
        measurement,
        visible,
        valid,
        center_ids=centers,
        update_frames=cfg.update_frames,
    )
    redundant_valid = valid.copy()
    for update_index, update in enumerate(cfg.update_frames):
        redundant = (
            view_count[update_index] >= cfg.minimum_triangulation_view_count
        ) & np.isfinite(reprojection[update_index])
        redundant_valid[update, centers] &= redundant
    pairwise_report, unguarded = predict_persistence_pairwise_rbf_arrays(
        baseline,
        measurement,
        visible,
        redundant_valid,
        center_ids=centers,
        update_frames=cfg.update_frames,
    )

    candidate = baseline.copy()
    scale = _object_scale_m(physical[0])
    previous_update = 0
    interval_records: list[dict[str, Any]] = []
    for update_index, (baseline_update, pairwise_update) in enumerate(
        zip(
            baseline_report["updates"],
            pairwise_report["updates"],
            strict=True,
        )
    ):
        update = int(pairwise_update["frame"])
        stop = int(pairwise_update["interval_end_exclusive"])
        current_supported = (
            view_count[update_index] >= cfg.minimum_triangulation_view_count
        ) & np.isfinite(reprojection[update_index])
        current_redundant = (
            view_count[update_index] >= cfg.redundant_view_count
        ) & current_supported
        if update_index == 0:
            previous_supported = np.ones(len(centers), dtype=bool)
        else:
            previous_supported = (
                view_count[update_index - 1]
                >= cfg.minimum_triangulation_view_count
            ) & np.isfinite(reprojection[update_index - 1])
        available = (
            current_supported
            & previous_supported
            & visible[update, centers]
            & valid[update, centers]
            & visible[previous_update, centers]
            & valid[previous_update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(measurement[previous_update, centers]), axis=1)
        )
        available_ids = centers[available]
        physical_motion = (
            physical[update, available_ids]
            - physical[previous_update, available_ids]
        )
        observed_motion = (
            measurement[update, available_ids]
            - measurement[previous_update, available_ids]
        )
        physical_rms = _radial_rms(physical_motion)
        observed_rms = _radial_rms(observed_motion)
        agreement = robust_huber_continuation_gain(
            physical_motion,
            observed_motion,
            minimum_point_count=cfg.minimum_motion_center_count,
            fallback=0.0,
        )
        future_physical_rms = _radial_rms(
            physical[stop - 1] - persistence[stop - 1]
        )
        innovation = (
            measurement[update, available_ids]
            - np.asarray(baseline[update, available_ids], dtype=np.float64)
        )
        innovation_rms = _radial_rms(innovation)
        correction = (
            np.asarray(unguarded[update + 1], dtype=np.float64)
            - np.asarray(baseline[update + 1], dtype=np.float64)
        )
        correction_rms = _radial_rms(correction)
        response_denominator = max(physical_rms, future_physical_rms, 1e-12)
        correction_ratio = correction_rms / response_denominator
        motion_ratio = physical_rms / max(observed_rms, 1e-12)
        residual_motion = observed_motion - agreement * physical_motion
        shared_bias = (
            np.zeros(3, dtype=np.float64)
            if not len(residual_motion)
            else np.median(residual_motion, axis=0)
        )
        common_mode_fraction = float(
            np.linalg.norm(shared_bias) / max(observed_rms, 1e-12)
        )
        selected = str(baseline_update["selected_backbone"])
        chamfer = baseline_update["current_observation_chamfer_m"]
        selected_chamfer = chamfer[selected]
        other = "persistence" if selected == "physical_prior" else "physical_prior"
        other_chamfer = chamfer[other]
        chamfer_margin = (
            0.0
            if selected_chamfer is None or other_chamfer is None
            else float(other_chamfer) - float(selected_chamfer)
        )
        redundant_count = int(np.sum(current_redundant))
        reprojection_values = reprojection[update_index, current_supported]
        median_reprojection = (
            cfg.reprojection_scale_px
            if not len(reprojection_values)
            else float(np.median(reprojection_values))
        )
        features = np.asarray(
            [
                physical_rms / scale,
                observed_rms / scale,
                future_physical_rms / scale,
                innovation_rms / scale,
                correction_rms / scale,
                correction_ratio,
                motion_ratio,
                agreement,
                _vector_cosine(physical_motion, observed_motion),
                common_mode_fraction,
                redundant_count / len(centers),
                median_reprojection / cfg.reprojection_scale_px,
                float(pairwise_update["inlier_fraction"]),
                float(pairwise_update["compatible_pair_fraction"]),
                float(selected == "physical_prior"),
                chamfer_margin / scale,
            ],
            dtype=np.float64,
        )
        correction_scale = min(
            1.0,
            cfg.maximum_correction_to_physical_response
            / max(correction_ratio, 1e-12),
        )
        reasons: list[str] = []
        if not bool(pairwise_update["accepted"]):
            reasons.append("pairwise-consensus-rejected")
        if (
            cfg.minimum_redundant_center_count > 0
            and redundant_count < cfg.minimum_redundant_center_count
        ):
            reasons.append("insufficient-three-view-redundancy")
        if len(available_ids) < cfg.minimum_motion_center_count:
            reasons.append("insufficient-causal-motion-support")
        if response_denominator < cfg.minimum_physical_support_m:
            reasons.append("physical-support-too-small")
        if observed_rms < cfg.minimum_observed_motion_m:
            reasons.append("observed-motion-too-small")
        if agreement < cfg.minimum_physical_agreement_gain:
            reasons.append("physical-observation-agreement-too-small")
        eligible = not reasons
        if eligible:
            bounded = np.asarray(baseline[update + 1 : stop], dtype=np.float64) + (
                correction_scale
                * (
                    np.asarray(unguarded[update + 1 : stop], dtype=np.float64)
                    - np.asarray(baseline[update + 1 : stop], dtype=np.float64)
                )
            )
            candidate[update + 1 : stop] = bounded.astype(
                baseline.dtype, copy=False
            )
        elif not np.array_equal(
            candidate[update + 1 : stop], baseline[update + 1 : stop]
        ):
            raise AssertionError("ineligible interval changed the exact baseline")
        interval_records.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "candidate_available": eligible,
                "decision": (
                    "pairwise-regret-candidate-available"
                    if eligible
                    else "causal-support-exact-baseline-fallback"
                ),
                "rejection_reasons": reasons,
                "bit_exact_baseline_fallback": bool(
                    eligible
                    or np.array_equal(
                        candidate[update + 1 : stop],
                        baseline[update + 1 : stop],
                    )
                ),
                "feature_names": list(FEATURE_NAMES),
                "features": features.tolist(),
                "redundant_center_count": redundant_count,
                "causal_motion_center_count": int(len(available_ids)),
                "physical_response_rms_m": physical_rms,
                "observed_motion_rms_m": observed_rms,
                "future_physical_response_rms_m": future_physical_rms,
                "physical_agreement_gain": agreement,
                "correction_rms_m": correction_rms,
                "correction_to_physical_response": correction_ratio,
                "applied_correction_scale": correction_scale,
                "common_mode_fraction": common_mode_fraction,
                "selected_backbone": selected,
                "pairwise_update": pairwise_update,
            }
        )
        previous_update = update

    report = {
        "arm": "selected_pairwise_causal_regret_candidate",
        "config": asdict(cfg),
        "feature_names": list(FEATURE_NAMES),
        "baseline_selection": baseline_report,
        "pairwise_candidate": pairwise_report,
        "updates": interval_records,
        "candidate_available_count": int(
            sum(record["candidate_available"] for record in interval_records)
        ),
        "information_boundary": {
            "target_argument_accepted": False,
            "outcome_argument_accepted": False,
            "future_observation_read": False,
            "future_physical_rollout_read": True,
            "future_physical_rollout_role": (
                "action-conditioned support and correction bound only"
            ),
            "camera_panel_is_bound_by_unique_identity": True,
            "two_view_rows_are_not_counted_as_three_view_redundancy": True,
        },
    }
    return report, baseline, candidate


def apply_pairwise_regret_guard(
    baseline_m: np.ndarray,
    candidate_m: np.ndarray,
    candidate_report: Mapping[str, Any],
    certificate: SourceRegretCertificate | None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Apply a source regret certificate, with exact fallback when absent."""

    baseline = np.asarray(baseline_m)
    candidate = np.asarray(candidate_m)
    _require(baseline.shape == candidate.shape, "candidate shape changed")
    selected = baseline.copy()
    decisions: list[dict[str, Any]] = []
    for update in candidate_report["updates"]:
        start = int(update["frame"]) + 1
        stop = int(update["interval_end_exclusive"])
        available = bool(update["candidate_available"])
        if not available or certificate is None:
            accepted = False
            reason = (
                "candidate-unavailable-exact-baseline-fallback"
                if not available
                else "missing-regret-certificate-exact-baseline-fallback"
            )
            predicted_regret = None
            upper_regret = None
            in_source_support = None
        else:
            features = np.asarray(update["features"], dtype=np.float64)
            decision = apply_regret_guard(
                baseline[start:stop], candidate[start:stop], features, certificate
            )
            selected[start:stop] = decision.selected_value
            accepted = decision.candidate_accepted
            reason = decision.reason
            predicted_regret = decision.predicted_regret
            upper_regret = (
                decision.upper_regret
                if np.isfinite(decision.upper_regret)
                else None
            )
            in_source_support = certificate.in_source_support(features)
        exact_fallback = bool(
            accepted
            or np.array_equal(selected[start:stop], baseline[start:stop])
        )
        if not exact_fallback:
            raise AssertionError("regret rejection changed the exact baseline")
        decisions.append(
            {
                "frame": int(update["frame"]),
                "interval_end_exclusive": stop,
                "candidate_available": available,
                "candidate_accepted": accepted,
                "reason": reason,
                "predicted_regret_m": predicted_regret,
                "upper_regret_m": upper_regret,
                "in_source_support": in_source_support,
                "bit_exact_baseline_fallback": exact_fallback,
            }
        )
    return {
        "arm": "selected_pairwise_causal_regret_guarded",
        "decisions": decisions,
        "accepted_count": int(
            sum(decision["candidate_accepted"] for decision in decisions)
        ),
        "exact_fallback_count": int(
            sum(
                not decision["candidate_accepted"]
                and decision["bit_exact_baseline_fallback"]
                for decision in decisions
            )
        ),
        "information_boundary": {
            "target_argument_accepted": False,
            "outcome_argument_accepted": False,
            "missing_certificate_policy": "bit-exact baseline fallback",
        },
    }, selected


__all__ = [
    "FEATURE_NAMES",
    "PairwiseRegretGuardConfig",
    "apply_pairwise_regret_guard",
    "build_pairwise_regret_candidate_arrays",
]
