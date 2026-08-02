"""Baseline-relative regret guard for dual-backbone Deform360 updates.

The candidate path is target free. Source outcomes are used only to fit a
``SourceRegretCertificate`` outside this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from .bias_aware_belief import SourceRegretCertificate, apply_regret_guard
from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
from .deform360_online_belief_evaluation import robust_huber_continuation_gain
from .deform360_raw_pairwise_correspondence_diagnostic import (
    MINIMUM_SELECTOR_SUPPORT,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    _corrected_frame,
)
from .deform360_selective_virtual_sensing_prediction import (
    _validate_prediction_inputs,
)
from .phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
    detect_pairwise_consensus_correspondences,
)
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
    update_recursive_rbf_belief,
)

DUAL_BACKBONE_ARM = "dual_backbone_pairwise_consensus_rbf"
SELECTED_BACKBONE_ARM = "selected_raw_backbone"
GUARDED_ARM = "dual_backbone_pairwise_regret_guarded"
PRIOR_CORRECTION_AT_UPDATE = "prior_correction_at_update_m"
PRIOR_VARIANCE_AT_UPDATE = "prior_variance_at_update_m2"
SELECTED_BACKBONE_AT_UPDATE = "selected_backbone_at_update_m"
PAIRWISE_REGRET_FEATURE_NAMES = (
    "physical_motion_rms_over_object_scale",
    "observed_motion_rms_over_object_scale",
    "correction_rms_over_object_scale",
    "log_correction_to_physical_motion_ratio",
    "physical_observation_agreement_gain",
    "correction_coherence",
    "available_center_fraction",
    "mean_inlier_view_fraction",
    "correction_change_rms_over_object_scale",
    "correction_temporal_cosine",
    "prior_innovation_rms_over_object_scale",
    "log_prior_innovation_to_residual_ratio",
    "prior_residual_cosine",
    "prior_consistency_gain",
    "prior_standardized_innovation_rms",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _radial_rms(value_m: np.ndarray) -> float:
    value = np.asarray(value_m, dtype=np.float64)
    if not len(value):
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=-1))))


def _object_scale_m(frame_zero_points_m: np.ndarray) -> float:
    points = np.asarray(frame_zero_points_m, dtype=np.float64)
    center = np.median(points, axis=0)
    return max(1e-6, float(2.0 * np.max(np.linalg.norm(points - center, axis=1))))


def predict_dual_backbone_pairwise_rbf_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_frames: Sequence[int] = (19, 38, 57),
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the exact dual-backbone pairwise candidate without a target."""

    physical_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    physical = np.asarray(physical_input, dtype=np.float64)
    persistence = np.asarray(persistence_input, dtype=np.float64)
    measurement = np.asarray(measurement_m, dtype=np.float64)
    measurement_visible = np.asarray(measurement_visibility, dtype=bool)
    measurement_valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    updates_tuple = tuple(int(frame) for frame in update_frames)
    _validate_prediction_inputs(
        persistence,
        measurement,
        measurement_visible,
        measurement_valid,
        centers,
        updates_tuple,
    )
    _require(physical.shape == persistence.shape, "physical and persistence differ")
    _require(np.all(np.isfinite(physical)), "physical prior contains non-finite values")

    gate_cfg = gate_config or PairwiseCorrespondenceGateConfig()
    belief_cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    backbones = {PHYSICAL_ARM: physical, PERSISTENCE_ARM: persistence}
    states = {
        name: initialize_recursive_rbf_belief(
            centers,
            backbone[0, centers],
            backbone[0],
            config=belief_cfg,
        )
        for name, backbone in backbones.items()
    }
    selected_trajectory = physical_input.copy()
    candidate = physical_input.copy()
    prior_correction_at_update = np.full_like(physical, np.nan)
    prior_variance_at_update = np.full_like(physical, np.nan)
    selected_backbone_at_update = np.full_like(physical, np.nan)
    output_dtype = persistence_input.dtype
    update_reports: list[dict[str, Any]] = []

    for update_index, update in enumerate(updates_tuple):
        stop = (
            updates_tuple[update_index + 1]
            if update_index + 1 < len(updates_tuple)
            else len(persistence)
        )
        available = (
            measurement_visible[update, centers]
            & measurement_valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(physical[update, centers]), axis=1)
            & np.all(np.isfinite(persistence[update, centers]), axis=1)
        )
        available_ids = centers[available]
        supported = len(available_ids) >= MINIMUM_SELECTOR_SUPPORT
        if supported:
            observed = measurement[update, available_ids]
            current_chamfer = {
                name: _symmetric_set_chamfer_m(
                    backbone[update, available_ids],
                    observed,
                )
                for name, backbone in backbones.items()
            }
            selected_name = min(
                (PHYSICAL_ARM, PERSISTENCE_ARM),
                key=lambda name: (
                    current_chamfer[name],
                    0 if name == PHYSICAL_ARM else 1,
                ),
            )
        else:
            current_chamfer = {PHYSICAL_ARM: None, PERSISTENCE_ARM: None}
            selected_name = PERSISTENCE_ARM
        selected = backbones[selected_name]
        selected_state = states[selected_name]
        forecast_frames = (
            0
            if selected_state.last_update_frame is None
            else update - selected_state.last_update_frame
        )
        prior_prediction = decode_recursive_rbf_belief(
            selected_state,
            selected[update],
            forecast_frames=forecast_frames,
            config=belief_cfg,
        )
        prior_correction_at_update[update] = prior_prediction.mean_m
        prior_variance_at_update[update] = prior_prediction.variance_m2
        selected_backbone_at_update[update] = selected[update]
        selected_trajectory[update + 1 : stop] = selected[update + 1 : stop]
        candidate[update + 1 : stop] = selected[update + 1 : stop]

        gates = {}
        for name, backbone in backbones.items():
            residual = np.full((len(centers), 3), np.nan, dtype=np.float64)
            residual[available] = (
                measurement[update, available_ids]
                - backbone[update, available_ids]
            )
            gate = detect_pairwise_consensus_correspondences(
                backbone[update, centers],
                measurement[update, centers],
                available,
                material_ids=centers,
                config=gate_cfg,
            )
            gates[name] = gate
            if supported and gate.accepted:
                states[name], _ = update_recursive_rbf_belief(
                    states[name],
                    update,
                    backbone[update, centers],
                    residual,
                    gate.inlier_mask.copy(),
                    config=belief_cfg,
                )

        selected_gate = gates[selected_name]
        accepted = supported and selected_gate.accepted
        if accepted:
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    states[selected_name],
                    selected[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                candidate[frame] = _corrected_frame(
                    selected[frame],
                    decoded.mean_m,
                    dtype=output_dtype,
                )
        elif not np.array_equal(
            candidate[update + 1 : stop],
            selected_trajectory[update + 1 : stop],
        ):
            raise AssertionError("pairwise abstention changed the selected backbone")

        update_reports.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "available_center_count": int(len(available_ids)),
                "selector_support_sufficient": supported,
                "selected_backbone": selected_name,
                "current_observation_chamfer_m": current_chamfer,
                "prior_belief_last_update_frame": selected_state.last_update_frame,
                "prior_belief_forecast_frames": forecast_frames,
                "pairwise_gate": {
                    "accepted": accepted,
                    "decision": (
                        selected_gate.decision
                        if supported
                        else "insufficient_selector_support"
                    ),
                    "inlier_count": selected_gate.inlier_count,
                    "inlier_fraction": selected_gate.inlier_fraction,
                    "compatible_pair_fraction": (
                        selected_gate.compatible_pair_fraction
                    ),
                },
                "bit_exact_selected_backbone_fallback": bool(
                    not accepted
                    and np.array_equal(
                        candidate[update + 1 : stop],
                        selected_trajectory[update + 1 : stop],
                    )
                ),
            }
        )

    report = {
        "arm": DUAL_BACKBONE_ARM,
        "center_ids": centers.tolist(),
        "update_frames": list(updates_tuple),
        "gate_config": asdict(gate_cfg),
        "belief_config": asdict(belief_cfg),
        "updates": update_reports,
        "information_boundary": {
            "target_argument_accepted": False,
            "future_dense_reconstruction_read": False,
            "future_particle_tracks_read": False,
            "state_innovation_likelihood_count": 1,
            "prior_reliability_uses_state_innovation": False,
        },
    }
    return report, {
        SELECTED_BACKBONE_ARM: selected_trajectory,
        DUAL_BACKBONE_ARM: candidate,
        PRIOR_CORRECTION_AT_UPDATE: prior_correction_at_update,
        PRIOR_VARIANCE_AT_UPDATE: prior_variance_at_update,
        SELECTED_BACKBONE_AT_UPDATE: selected_backbone_at_update,
    }


def pairwise_regret_features(
    physical_prior_m: np.ndarray,
    selected_backbone_m: np.ndarray,
    candidate_m: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    update_frame: int,
    previous_update_frame: int,
    interval_end_exclusive: int,
    inlier_view_count: np.ndarray | None = None,
    prior_correction_at_update_m: np.ndarray | None = None,
    prior_variance_at_update_m2: np.ndarray | None = None,
    selected_backbone_at_update_m: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract target-free features for one baseline-relative regret decision."""

    physical = np.asarray(physical_prior_m, dtype=np.float64)
    baseline = np.asarray(selected_backbone_m, dtype=np.float64)
    candidate = np.asarray(candidate_m, dtype=np.float64)
    measurement = np.asarray(measurement_m, dtype=np.float64)
    visible = np.asarray(measurement_visibility, dtype=bool)
    valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    _require(
        physical.shape == baseline.shape == candidate.shape == measurement.shape,
        "regret-feature trajectories differ",
    )
    _require(physical.ndim == 3 and physical.shape[2] == 3, "invalid trajectory")
    _require(
        visible.shape == valid.shape == physical.shape[:2],
        "regret-feature masks differ",
    )
    _require(
        0 <= previous_update_frame < update_frame < interval_end_exclusive <= len(physical),
        "invalid regret-feature interval",
    )
    available = (
        visible[update_frame, centers]
        & valid[update_frame, centers]
        & np.all(np.isfinite(measurement[update_frame, centers]), axis=1)
    )
    available_ids = centers[available]
    if len(available_ids):
        previous_observed = (
            visible[previous_update_frame, available_ids]
            & valid[previous_update_frame, available_ids]
            & np.all(
                np.isfinite(measurement[previous_update_frame, available_ids]),
                axis=1,
            )
        )
        observed_origin = physical[previous_update_frame, available_ids].copy()
        observed_origin[previous_observed] = measurement[
            previous_update_frame,
            available_ids[previous_observed],
        ]
        physical_delta = (
            physical[update_frame, available_ids]
            - physical[previous_update_frame, available_ids]
        )
        observed_delta = measurement[update_frame, available_ids] - observed_origin
    else:
        previous_observed = np.zeros(0, dtype=bool)
        physical_delta = (
            physical[update_frame, centers]
            - physical[previous_update_frame, centers]
        )
        observed_delta = np.zeros_like(physical_delta)
    correction = (
        candidate[update_frame + 1]
        - baseline[update_frame + 1]
    )
    if previous_update_frame == 0:
        previous_correction = np.zeros_like(correction)
    else:
        previous_correction = (
            candidate[update_frame - 1]
            - baseline[update_frame - 1]
        )
    correction_change = correction - previous_correction
    scale = _object_scale_m(physical[0])
    physical_rms = _radial_rms(physical_delta)
    observed_rms = _radial_rms(observed_delta)
    correction_rms = _radial_rms(correction)
    correction_mean = np.mean(correction, axis=0)
    temporal_denominator = float(
        np.linalg.norm(correction) * np.linalg.norm(previous_correction)
    )
    temporal_cosine = (
        0.0
        if temporal_denominator <= 1e-12
        else float(
            np.sum(correction * previous_correction) / temporal_denominator
        )
    )
    agreement = robust_huber_continuation_gain(
        physical_delta,
        observed_delta,
        minimum_point_count=3,
        fallback=0.0,
    )
    if selected_backbone_at_update_m is None:
        selected_current = baseline[update_frame]
    else:
        selected_current = np.asarray(
            selected_backbone_at_update_m,
            dtype=np.float64,
        )
        _require(
            selected_current.shape == physical.shape[1:],
            "selected update backbone shape changed",
        )
    if prior_correction_at_update_m is None:
        prior_correction = np.zeros_like(selected_current)
    else:
        prior_correction = np.asarray(
            prior_correction_at_update_m,
            dtype=np.float64,
        )
        _require(
            prior_correction.shape == physical.shape[1:],
            "prior update correction shape changed",
        )
    if prior_variance_at_update_m2 is None:
        prior_variance = np.full_like(selected_current, 1.0)
    else:
        prior_variance = np.asarray(
            prior_variance_at_update_m2,
            dtype=np.float64,
        )
        _require(
            prior_variance.shape == physical.shape[1:],
            "prior update variance shape changed",
        )
        _require(
            np.all(np.isfinite(prior_variance)) and np.all(prior_variance > 0.0),
            "prior update variance must be finite and positive",
        )
    current_residual = (
        measurement[update_frame, available_ids]
        - selected_current[available_ids]
    )
    predicted_residual = prior_correction[available_ids]
    prior_innovation = current_residual - predicted_residual
    residual_rms = _radial_rms(current_residual)
    prior_innovation_rms = _radial_rms(prior_innovation)
    residual_denominator = float(
        np.linalg.norm(current_residual) * np.linalg.norm(predicted_residual)
    )
    prior_residual_cosine = (
        0.0
        if residual_denominator <= 1e-12
        else float(
            np.sum(current_residual * predicted_residual)
            / residual_denominator
        )
    )
    prior_consistency_gain = float(
        1.0 - prior_innovation_rms / max(residual_rms, 1e-9)
    )
    standardized_innovation = prior_innovation / np.sqrt(
        prior_variance[available_ids]
    )
    prior_standardized_innovation_rms = _radial_rms(standardized_innovation)
    if inlier_view_count is None:
        view_fraction = 0.0
    else:
        views = np.asarray(inlier_view_count, dtype=np.float64)
        _require(views.shape == centers.shape, "inlier-view shape changed")
        selected_views = views[available]
        finite_views = selected_views[np.isfinite(selected_views)]
        view_fraction = (
            0.0
            if not len(finite_views)
            else float(np.mean(np.clip(finite_views / 3.0, 0.0, 1.0)))
        )
    values = np.asarray(
        (
            physical_rms / scale,
            observed_rms / scale,
            correction_rms / scale,
            np.log(
                (correction_rms + 1e-6)
                / (physical_rms + 1e-6)
            ),
            agreement,
            float(np.linalg.norm(correction_mean) / max(correction_rms, 1e-9)),
            float(len(available_ids) / len(centers)),
            view_fraction,
            _radial_rms(correction_change) / scale,
            temporal_cosine,
            prior_innovation_rms / scale,
            np.log(
                (prior_innovation_rms + 1e-6)
                / (residual_rms + 1e-6)
            ),
            prior_residual_cosine,
            prior_consistency_gain,
            prior_standardized_innovation_rms,
        ),
        dtype=np.float64,
    )
    _require(np.all(np.isfinite(values)), "regret features are non-finite")
    diagnostics = dict(zip(PAIRWISE_REGRET_FEATURE_NAMES, values.tolist(), strict=True))
    diagnostics["object_scale_m"] = scale
    diagnostics["available_center_count"] = float(len(available_ids))
    diagnostics["previous_observation_fraction"] = (
        0.0 if not len(previous_observed) else float(np.mean(previous_observed))
    )
    diagnostics["current_residual_rms_m"] = residual_rms
    diagnostics["prior_predicted_residual_rms_m"] = _radial_rms(
        predicted_residual
    )
    return values, diagnostics


def apply_pairwise_regret_certificate(
    selected_backbone_m: np.ndarray,
    candidate_m: np.ndarray,
    feature_vectors: np.ndarray,
    certificate: SourceRegretCertificate,
    *,
    update_frames: Sequence[int] = (19, 38, 57),
) -> tuple[dict[str, Any], np.ndarray]:
    """Guard candidate intervals and preserve rejected baselines bit exactly."""

    baseline_input = np.asarray(selected_backbone_m)
    candidate_input = np.asarray(candidate_m)
    _require(baseline_input.shape == candidate_input.shape, "candidate shape changed")
    updates = tuple(int(frame) for frame in update_frames)
    features = np.asarray(feature_vectors, dtype=np.float64)
    _require(
        features.shape == (len(updates), len(certificate.feature_center)),
        "regret feature matrix changed",
    )
    guarded = baseline_input.copy()
    decisions = []
    for index, update in enumerate(updates):
        stop = updates[index + 1] if index + 1 < len(updates) else len(guarded)
        decision = apply_regret_guard(
            baseline_input[update + 1 : stop],
            candidate_input[update + 1 : stop],
            features[index],
            certificate,
        )
        guarded[update + 1 : stop] = decision.selected_value
        if not decision.candidate_accepted and not np.array_equal(
            guarded[update + 1 : stop],
            baseline_input[update + 1 : stop],
        ):
            raise AssertionError("regret rejection changed the exact baseline")
        decisions.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "candidate_accepted": decision.candidate_accepted,
                "predicted_regret_m": decision.predicted_regret,
                "upper_regret_m": (
                    float(decision.upper_regret)
                    if np.isfinite(decision.upper_regret)
                    else None
                ),
                "reason": decision.reason,
                "bit_exact_baseline_fallback": bool(
                    not decision.candidate_accepted
                    and np.array_equal(
                        guarded[update + 1 : stop],
                        baseline_input[update + 1 : stop],
                    )
                ),
            }
        )
    return {
        "arm": GUARDED_ARM,
        "feature_names": list(PAIRWISE_REGRET_FEATURE_NAMES),
        "updates": decisions,
        "information_boundary": {
            "target_argument_accepted": False,
            "source_outcomes_used_only_in_bound_certificate": True,
            "candidate_residual_used_as_prior_reliability": False,
            "rejection_is_bit_exact_selected_backbone": True,
        },
    }, guarded


__all__ = [
    "DUAL_BACKBONE_ARM",
    "GUARDED_ARM",
    "PAIRWISE_REGRET_FEATURE_NAMES",
    "PRIOR_CORRECTION_AT_UPDATE",
    "PRIOR_VARIANCE_AT_UPDATE",
    "SELECTED_BACKBONE_ARM",
    "SELECTED_BACKBONE_AT_UPDATE",
    "apply_pairwise_regret_certificate",
    "pairwise_regret_features",
    "predict_dual_backbone_pairwise_rbf_arrays",
]
