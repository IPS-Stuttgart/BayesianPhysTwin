"""Pairwise-consistent, nuisance-aware state updates for Deform360 development.

This module composes three existing target-free mechanisms without changing
their frozen implementations:

1. pairwise geometry rejects material-identity inconsistencies;
2. nuisance-marginalized information selects a compact observation subset; and
3. the bias-aware physical-response update separates reachable state from
   coherent observation bias.

The adapter accepts no target trajectory.  Open outcomes belong only in a
separate development evaluator and must never influence the update itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .bias_aware_belief import (
    BiasAwareStateUpdateConfig,
    build_physical_response_basis,
    decode_bias_aware_state,
    restrict_state_basis_to_identifiable_subspace,
    update_bias_aware_state,
)
from .deform360_online_belief_evaluation import (
    UPDATE_FRAMES,
    robust_huber_continuation_gain,
)
from .nuisance_aware_information import (
    NuisanceAwareInformationState,
    greedy_nuisance_aware_selection,
)
from .phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
    detect_pairwise_consensus_correspondences,
)

PROTOCOL_ID = "deform360-pairwise-bias-aware-open27-v1-development"


@dataclass(frozen=True)
class PairwiseBiasAwareDevelopmentConfig:
    """Target-free choices for one development candidate."""

    update_frames: tuple[int, ...] = UPDATE_FRAMES
    selected_center_count: int = 12
    physical_response_rank: int = 4
    minimum_motion_center_count: int = 3
    minimum_physical_response_m: float = 0.0005
    minimum_observed_motion_m: float = 0.0005
    minimum_physical_agreement_gain: float = 0.40
    minimum_identifiable_fraction: float = 0.10
    observation_variance_floor_m2: float = 0.005**2
    reprojection_scale_px: float = 3.0
    information_effective_sample_cap: float = 8.0
    minimum_information_gain_nats: float = 0.0
    pairwise_gate: PairwiseCorrespondenceGateConfig = field(
        default_factory=PairwiseCorrespondenceGateConfig
    )
    state_update: BiasAwareStateUpdateConfig = field(
        default_factory=BiasAwareStateUpdateConfig
    )

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.update_frames))) != self.update_frames:
            raise ValueError("update_frames must be strictly increasing")
        if not self.update_frames:
            raise ValueError("update_frames must not be empty")
        if self.selected_center_count < self.pairwise_gate.minimum_inlier_count:
            raise ValueError("selected center count cannot satisfy the pairwise gate")
        if self.physical_response_rank < 1:
            raise ValueError("physical response rank must be positive")
        if self.minimum_motion_center_count < 1:
            raise ValueError("minimum motion center count must be positive")
        positive = (
            self.minimum_physical_response_m,
            self.minimum_observed_motion_m,
            self.minimum_identifiable_fraction,
            self.observation_variance_floor_m2,
            self.reprojection_scale_px,
            self.information_effective_sample_cap,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("development scales must be positive")
        if self.minimum_identifiable_fraction > 1.0:
            raise ValueError("minimum identifiable fraction exceeds one")
        if not 0.0 <= self.minimum_physical_agreement_gain <= 1.0:
            raise ValueError("minimum physical agreement gain must lie in [0, 1]")
        if (
            not np.isfinite(self.minimum_information_gain_nats)
            or self.minimum_information_gain_nats < 0.0
        ):
            raise ValueError("minimum information gain must be nonnegative")


def _validate_inputs(
    baseline_m: np.ndarray,
    physical_response_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    action_support: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    center_ids: np.ndarray,
    prior_reliability: np.ndarray,
    observation_variance_m2: np.ndarray,
    update_frames: tuple[int, ...],
) -> None:
    if baseline_m.ndim != 3 or baseline_m.shape[2] != 3:
        raise ValueError("baseline must have shape (T, N, 3)")
    if physical_response_m.shape != baseline_m.shape:
        raise ValueError("physical response must match baseline")
    if frame_zero_points_m.shape != baseline_m.shape[1:]:
        raise ValueError("frame-zero point shape changed")
    if action_support.shape != (baseline_m.shape[1],):
        raise ValueError("action support shape changed")
    if measurement_m.shape != baseline_m.shape:
        raise ValueError("measurement must match baseline")
    for name, value in (
        ("measurement_visibility", measurement_visibility),
        ("measurement_validity", measurement_validity),
    ):
        if value.shape != baseline_m.shape[:2]:
            raise ValueError(f"{name} shape changed")
    if center_ids.ndim != 1 or len(center_ids) != len(np.unique(center_ids)):
        raise ValueError("center_ids must be a unique vector")
    if np.any(center_ids < 0) or np.any(center_ids >= baseline_m.shape[1]):
        raise ValueError("center ID exceeds trajectory")
    expected = (len(update_frames), len(center_ids))
    if prior_reliability.shape != expected:
        raise ValueError("prior reliability shape changed")
    if observation_variance_m2.shape != expected:
        raise ValueError("observation variance shape changed")
    if update_frames[-1] >= len(baseline_m):
        raise ValueError("update frame exceeds trajectory")
    finite_inputs = (
        baseline_m,
        physical_response_m,
        frame_zero_points_m,
        action_support,
        prior_reliability,
        observation_variance_m2,
    )
    if any(not np.all(np.isfinite(value)) for value in finite_inputs):
        raise ValueError("prediction input contains non-finite values")
    if np.any((action_support < 0.0) | (action_support > 1.0)):
        raise ValueError("action support must lie in [0, 1]")
    if np.any((prior_reliability < 0.0) | (prior_reliability > 1.0)):
        raise ValueError("prior reliability must lie in [0, 1]")
    if np.any(observation_variance_m2 <= 0.0):
        raise ValueError("observation variance must be positive")


def _spatial_bias_basis(frame_zero_points_m: np.ndarray) -> np.ndarray:
    points = np.asarray(frame_zero_points_m, dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0.0:
        return np.zeros((len(points), 0), dtype=np.float64)
    tolerance = max(centered.shape) * np.finfo(float).eps * singular_values[0]
    count = int(np.sum(singular_values > tolerance))
    basis = left[:, :count].copy()
    for mode in range(count):
        pivot = int(np.argmax(np.abs(basis[:, mode])))
        if basis[pivot, mode] < 0.0:
            basis[:, mode] *= -1.0
        basis[:, mode] /= np.max(np.abs(basis[:, mode]))
    return basis


def _radial_rms(value_m: np.ndarray) -> float:
    value = np.asarray(value_m, dtype=np.float64)
    if not len(value):
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))


def _capped_reliability(
    reliability: np.ndarray,
    effective_sample_cap: float,
) -> np.ndarray:
    values = np.asarray(reliability, dtype=np.float64).copy()
    total = float(np.sum(values))
    if total > effective_sample_cap:
        values *= effective_sample_cap / total
    return values


def _select_informative_centers(
    state_basis: np.ndarray,
    nuisance_basis: np.ndarray,
    variance_m2: np.ndarray,
    reliability: np.ndarray,
    *,
    count: int,
    config: PairwiseBiasAwareDevelopmentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_dimension = state_basis.shape[1]
    nuisance_dimension = nuisance_basis.shape[1]
    state_prior = np.eye(state_dimension) / config.state_update.state_prior_std_m**2
    nuisance_scale = np.concatenate(
        (
            np.full(
                max(0, nuisance_dimension - 1),
                config.state_update.shared_bias_prior_std_m,
            ),
            np.asarray([config.state_update.camera_bias_prior_std_m]),
        )
    )
    nuisance_prior = np.diag(1.0 / np.square(nuisance_scale))
    prior = NuisanceAwareInformationState.from_independent_priors(
        state_prior,
        nuisance_prior,
    )
    effective_reliability = _capped_reliability(
        reliability,
        config.information_effective_sample_cap,
    )
    selection = greedy_nuisance_aware_selection(
        prior,
        tuple(row[None] for row in state_basis),
        tuple(row[None] for row in nuisance_basis),
        tuple(np.asarray([[value]], dtype=np.float64) for value in variance_m2),
        reliabilities=tuple(float(value) for value in effective_reliability),
        count=count,
        minimum_gain_nats=config.minimum_information_gain_nats,
    )
    return (
        selection.selected_indices,
        selection.mutual_information_nats,
        effective_reliability,
    )


def predict_pairwise_bias_aware_candidate_arrays(
    baseline_m: np.ndarray,
    physical_response_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    action_support: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    center_ids: np.ndarray,
    prior_reliability: np.ndarray,
    observation_variance_m2: np.ndarray,
    config: PairwiseBiasAwareDevelopmentConfig | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build a target-free pairwise and bias-aware state candidate."""

    cfg = config or PairwiseBiasAwareDevelopmentConfig()
    baseline_input = np.asarray(baseline_m)
    baseline = np.asarray(baseline_input, dtype=np.float64)
    response = np.asarray(physical_response_m, dtype=np.float64)
    frame_zero = np.asarray(frame_zero_points_m, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
    measurement = np.asarray(measurement_m, dtype=np.float64)
    visible = np.asarray(measurement_visibility, dtype=bool)
    valid = np.asarray(measurement_validity, dtype=bool)
    centers = np.asarray(center_ids, dtype=np.int64)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    variance = np.asarray(observation_variance_m2, dtype=np.float64)
    _validate_inputs(
        baseline,
        response,
        frame_zero,
        support,
        measurement,
        visible,
        valid,
        centers,
        reliability,
        variance,
        cfg.update_frames,
    )

    candidate = baseline_input.copy()
    spatial_bias = _spatial_bias_basis(frame_zero)
    support_mask = support > 0.0
    previous_update = 0
    update_reports: list[dict[str, Any]] = []

    for update_index, update in enumerate(cfg.update_frames):
        stop = (
            cfg.update_frames[update_index + 1]
            if update_index + 1 < len(cfg.update_frames)
            else len(baseline)
        )
        candidate[update + 1 : stop] = baseline_input[update + 1 : stop]
        available = (
            visible[update, centers]
            & valid[update, centers]
            & np.all(np.isfinite(measurement[update, centers]), axis=1)
            & np.all(np.isfinite(baseline[update, centers]), axis=1)
            & (reliability[update_index] > 0.0)
        )
        gate = detect_pairwise_consensus_correspondences(
            baseline[update, centers],
            measurement[update, centers],
            available,
            material_ids=centers,
            config=cfg.pairwise_gate,
        )
        previous_available = (
            visible[previous_update, centers]
            & valid[previous_update, centers]
            & np.all(np.isfinite(measurement[previous_update, centers]), axis=1)
        )
        motion_available = gate.inlier_mask & previous_available
        physical_delta = response[update] - response[previous_update]
        physical_response_rms = _radial_rms(physical_delta[support_mask])
        observed_motion_rms = _radial_rms(
            measurement[update, centers[motion_available]]
            - measurement[previous_update, centers[motion_available]]
        )
        physical_agreement_gain = robust_huber_continuation_gain(
            physical_delta[centers[motion_available]],
            measurement[update, centers[motion_available]]
            - measurement[previous_update, centers[motion_available]],
            minimum_point_count=cfg.minimum_motion_center_count,
            fallback=0.0,
        )
        dynamic_selected = (
            gate.accepted
            and int(np.sum(motion_available)) >= cfg.minimum_motion_center_count
            and physical_response_rms >= cfg.minimum_physical_response_m
            and observed_motion_rms >= cfg.minimum_observed_motion_m
            and physical_agreement_gain >= cfg.minimum_physical_agreement_gain
        )
        record: dict[str, Any] = {
            "frame": int(update),
            "interval_end_exclusive": int(stop),
            "available_center_count": int(np.sum(available)),
            "pairwise_gate": {
                "accepted": gate.accepted,
                "decision": gate.decision,
                "inlier_count": gate.inlier_count,
                "inlier_fraction": gate.inlier_fraction,
                "compatible_pair_fraction": gate.compatible_pair_fraction,
            },
            "motion_center_count": int(np.sum(motion_available)),
            "physical_response_rms_m": physical_response_rms,
            "observed_motion_rms_m": observed_motion_rms,
            "causal_physical_agreement_gain": physical_agreement_gain,
            "dynamic_window_selected": dynamic_selected,
            "candidate_available": False,
            "decision": "target-free-gate-exact-baseline-fallback",
            "bit_exact_baseline_fallback": True,
            "selected_center_ids": [],
            "information_gain_nats": [],
        }
        if dynamic_selected:
            try:
                physical_basis = build_physical_response_basis(
                    response[: update + 1],
                    action_support=support,
                    rank=cfg.physical_response_rank,
                    minimum_response_m=cfg.minimum_physical_response_m,
                )
                inlier_offsets = np.flatnonzero(gate.inlier_mask)
                inlier_order = np.argsort(
                    centers[inlier_offsets],
                    kind="mergesort",
                )
                inlier_offsets = inlier_offsets[inlier_order]
                inlier_ids = centers[inlier_offsets]
                inlier_bias = np.column_stack(
                    (spatial_bias[inlier_ids], np.ones(len(inlier_ids)))
                )
                identifiable_all = restrict_state_basis_to_identifiable_subspace(
                    physical_basis.basis,
                    physical_basis.basis[inlier_ids],
                    inlier_bias,
                    minimum_identifiable_fraction=cfg.minimum_identifiable_fraction,
                )
                selected_local, information_gain, effective_reliability = (
                    _select_informative_centers(
                        identifiable_all.observation_basis,
                        inlier_bias,
                        variance[update_index, inlier_offsets],
                        reliability[update_index, inlier_offsets],
                        count=min(cfg.selected_center_count, len(inlier_ids)),
                        config=cfg,
                    )
                )
                if len(selected_local) < cfg.pairwise_gate.minimum_inlier_count:
                    raise ValueError("insufficient nuisance-distinguishable support")
                selected_ids = inlier_ids[selected_local]
                selected_bias = np.column_stack(
                    (spatial_bias[selected_ids], np.ones(len(selected_ids)))
                )
                identifiable = restrict_state_basis_to_identifiable_subspace(
                    physical_basis.basis,
                    physical_basis.basis[selected_ids],
                    selected_bias,
                    minimum_identifiable_fraction=cfg.minimum_identifiable_fraction,
                )
                innovation = (
                    measurement[update, selected_ids]
                    - baseline[update, selected_ids]
                )
                update_result = update_bias_aware_state(
                    innovation[None],
                    np.ones((1, len(selected_ids)), dtype=bool),
                    identifiable.observation_basis,
                    spatial_bias[selected_ids],
                    prior_reliability=reliability[
                        update_index, inlier_offsets
                    ][selected_local][None],
                    observation_variance_m2=variance[
                        update_index, inlier_offsets
                    ][selected_local][None],
                    config=cfg.state_update,
                )
                if not update_result.accepted:
                    raise ValueError(update_result.reason)
                correction = decode_bias_aware_state(
                    update_result,
                    identifiable.query_basis,
                )
                correction *= physical_agreement_gain
                candidate[update + 1 : stop] = (
                    baseline[update + 1 : stop] + correction[None]
                ).astype(baseline_input.dtype, copy=False)
                record.update(
                    {
                        "candidate_available": True,
                        "decision": "pairwise-bias-aware-candidate-available",
                        "bit_exact_baseline_fallback": False,
                        "selected_center_ids": selected_ids.tolist(),
                        "information_gain_nats": information_gain.tolist(),
                        "effective_reliability_sum": float(
                            np.sum(effective_reliability)
                        ),
                        "physical_basis_rank": int(
                            physical_basis.basis.shape[1]
                        ),
                        "identifiable_basis_rank": int(
                            identifiable.query_basis.shape[1]
                        ),
                        "minimum_identifiable_fraction": float(
                            np.min(identifiable.identifiable_fractions)
                        ),
                        "maximum_correction_m": float(
                            np.max(np.linalg.norm(correction, axis=1))
                        ),
                        "state_update_diagnostics": update_result.diagnostics,
                    }
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                record["decision"] = "pairwise-bias-aware-exact-baseline-fallback"
                record["fallback_reason"] = f"{type(error).__name__}: {error}"
        if not record["candidate_available"] and not np.array_equal(
            candidate[update + 1 : stop],
            baseline_input[update + 1 : stop],
        ):
            raise AssertionError("candidate fallback changed the exact baseline")
        update_reports.append(record)
        previous_update = update

    report = {
        "protocol_id": PROTOCOL_ID,
        "arm": "pairwise_nuisance_selected_bias_aware_state",
        "config": asdict(cfg),
        "center_ids": centers.tolist(),
        "update_frames": list(cfg.update_frames),
        "updates": update_reports,
        "candidate_update_count": int(
            sum(bool(record["candidate_available"]) for record in update_reports)
        ),
        "information_boundary": {
            "target_argument_accepted": False,
            "future_observation_read": False,
            "physical_response_frames_by_update": "causal prefix [0, update] only",
            "association_gate_uses_target": False,
            "prior_reliability_uses_state_innovation": False,
            "state_innovation_likelihood_count": 1,
            "unknown_cross_center_correlation_treatment": (
                "explicit shared nuisance basis plus capped effective information mass"
            ),
        },
    }
    return report, candidate


__all__ = [
    "PROTOCOL_ID",
    "PairwiseBiasAwareDevelopmentConfig",
    "predict_pairwise_bias_aware_candidate_arrays",
]
