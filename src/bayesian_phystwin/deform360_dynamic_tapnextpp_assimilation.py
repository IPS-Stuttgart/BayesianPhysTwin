"""Target-free assimilation for physics-guided dynamic TAPNext++ tracks.

The absolute multiview lift can contain a coherent camera/world offset.  This
module therefore updates the physical forecast from motion relative to each
query's causal birth, not from its absolute lifted position.  A conservative
unknown-correlation covariance bound, pairwise correspondence gate, and robust
recursive RBF belief sit in front of an exact selected-backbone fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
from .deform360_dynamic_query import DynamicQuerySchedule
from .phystwin_correspondence_gate import (
    PairwiseCorrespondenceGateConfig,
    detect_pairwise_consensus_correspondences,
)
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    RecursiveRbfBeliefSnapshot,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
)
from .pseudo_measurements import PseudoMeasurementBatch
from .robust_likelihood import robust_mixture_likelihood
from .tapnextpp_dynamic_multiview import DynamicMultiviewResult

UPDATE_FRAMES = (19, 38, 57)
MINIMUM_SELECTOR_SUPPORT = 3
PHYSICAL_ARM = "physical_prior"
PERSISTENCE_ARM = "persistence"
SELECTED_BACKBONE_ARM = "selected_physical_or_persistence"
CANDIDATE_ARM = "dynamic_birth_anchored_covariance_aware_rbf"
LEGACY_RELIABILITY_ASSIMILATION = "legacy-reliability-only-v2"
SET_VALUED_MIXTURE_ASSIMILATION = "set-valued-association-mixture-v3"
_ASSIMILATION_MODES = frozenset(
    {
        LEGACY_RELIABILITY_ASSIMILATION,
        SET_VALUED_MIXTURE_ASSIMILATION,
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class BirthAnchoredMeasurements:
    """Sparse endpoint measurements derived from within-birth displacement."""

    measurement_m: np.ndarray
    covariance_m2: np.ndarray
    prior_reliability: np.ndarray
    association_probability: np.ndarray
    available: np.ndarray
    entity_ids: np.ndarray

    def __post_init__(self) -> None:
        measurement = np.asarray(self.measurement_m, dtype=np.float64).copy()
        covariance = np.asarray(self.covariance_m2, dtype=np.float64).copy()
        reliability = np.asarray(
            self.prior_reliability,
            dtype=np.float64,
        ).copy()
        association = np.asarray(
            self.association_probability,
            dtype=np.float64,
        ).copy()
        available = np.asarray(self.available, dtype=bool).copy()
        entities = np.asarray(self.entity_ids, dtype=np.int64).copy()
        _require(
            measurement.ndim == 3 and measurement.shape[2] == 3,
            "measurement must have shape (T, N, 3)",
        )
        _require(
            covariance.shape == (*measurement.shape[:2], 3, 3),
            "measurement covariance shape changed",
        )
        _require(
            reliability.shape == association.shape == available.shape
            == measurement.shape[:2],
            "measurement metadata shape changed",
        )
        _require(
            entities.ndim == 1
            and len(entities) > 0
            and len(np.unique(entities)) == len(entities)
            and np.all((entities >= 0) & (entities < measurement.shape[1])),
            "measurement entity IDs are invalid",
        )
        _require(
            np.all(np.isfinite(measurement[available]))
            and np.all(np.isnan(measurement[~available])),
            "measurement finiteness differs from availability",
        )
        _require(
            np.all(np.isfinite(covariance[available]))
            and np.all(np.isnan(covariance[~available])),
            "covariance finiteness differs from availability",
        )
        _require(
            np.all(np.isfinite(reliability))
            and np.all((reliability >= 0.0) & (reliability <= 1.0))
            and np.all(np.isfinite(association))
            and np.all((association >= 0.0) & (association <= 1.0)),
            "measurement probabilities must lie in [0, 1]",
        )
        for matrix in covariance[available]:
            _require(
                np.min(np.linalg.eigvalsh(matrix)) > 0.0,
                "available covariance must be positive definite",
            )
        for value in (
            measurement,
            covariance,
            reliability,
            association,
            available,
            entities,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "measurement_m", measurement)
        object.__setattr__(self, "covariance_m2", covariance)
        object.__setattr__(self, "prior_reliability", reliability)
        object.__setattr__(self, "association_probability", association)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "entity_ids", entities)


def build_birth_anchored_measurements(
    result: DynamicMultiviewResult,
    schedule: DynamicQuerySchedule,
    physical_prediction_m: np.ndarray,
) -> BirthAnchoredMeasurements:
    """Cancel a birth-wave gauge by observing displacement from each birth.

    With unknown temporal correlation, ``2 (Sigma_birth + Sigma_update)`` is a
    conservative covariance bound for the difference.  The coherent low-rank
    camera factor is common within an update interval and cancels algebraically;
    it is not added again as independent noise.
    """

    physical = np.asarray(physical_prediction_m, dtype=np.float64)
    _require(
        physical.ndim == 3
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical)),
        "physical prediction must have finite shape (T, N, 3)",
    )
    entities = np.asarray(schedule.entity_ids, dtype=np.int64)
    births = np.asarray(schedule.birth_frames, dtype=np.int64)
    updates = np.asarray(schedule.update_frames, dtype=np.int64)
    _require(
        result.trajectory_world_m.shape
        == (max(UPDATE_FRAMES) + 1, len(entities), 3),
        "provider result differs from the causal schedule",
    )
    _require(
        np.all((entities >= 0) & (entities < physical.shape[1]))
        and np.all((births >= 0) & (births < physical.shape[0]))
        and np.all((updates >= births) & (updates < physical.shape[0])),
        "schedule exceeds the physical prediction",
    )
    measurement = np.full(physical.shape, np.nan, dtype=np.float64)
    covariance = np.full((*physical.shape[:2], 3, 3), np.nan, dtype=np.float64)
    reliability = np.zeros(physical.shape[:2], dtype=np.float64)
    association = np.zeros(physical.shape[:2], dtype=np.float64)
    available = np.zeros(physical.shape[:2], dtype=bool)
    for row, (entity, birth, update) in enumerate(
        zip(entities, births, updates, strict=True)
    ):
        if not (
            result.accepted_support[birth, row]
            and result.accepted_support[update, row]
        ):
            continue
        observed_displacement = (
            result.trajectory_world_m[update, row]
            - result.trajectory_world_m[birth, row]
        )
        measurement[update, entity] = (
            physical[birth, entity] + observed_displacement
        )
        covariance[update, entity] = 2.0 * (
            result.local_covariance_m2[birth, row]
            + result.local_covariance_m2[update, row]
        )
        reliability[update, entity] = float(
            np.sqrt(
                result.prior_reliability[birth, row]
                * result.prior_reliability[update, row]
            )
        )
        association[update, entity] = float(
            np.sqrt(
                result.association_probability[birth, row]
                * result.association_probability[update, row]
            )
        )
        available[update, entity] = True
    return BirthAnchoredMeasurements(
        measurement_m=measurement,
        covariance_m2=covariance,
        prior_reliability=reliability,
        association_probability=association,
        available=available,
        entity_ids=entities,
    )


def _scalar_covariance_bound(covariance_m2: np.ndarray) -> np.ndarray:
    matrices = np.asarray(covariance_m2, dtype=np.float64)
    return np.max(np.linalg.eigvalsh(matrices), axis=1)


def _update_covariance_aware_rbf_belief(
    prior: RecursiveRbfBeliefSnapshot,
    frame_index: int,
    center_positions_m: np.ndarray,
    measured_residual_m: np.ndarray,
    available: np.ndarray,
    observation_covariance_m2: np.ndarray,
    prior_reliability: np.ndarray,
    *,
    config: RecursiveRbfBeliefConfig,
    association_probability: np.ndarray | None = None,
) -> tuple[RecursiveRbfBeliefSnapshot, np.ndarray]:
    """Apply one residual-independent-reliability, robust metric update."""

    positions = np.asarray(center_positions_m, dtype=np.float64)
    residual = np.asarray(measured_residual_m, dtype=np.float64)
    mask = np.asarray(available, dtype=bool).copy()
    covariance = np.asarray(observation_covariance_m2, dtype=np.float64)
    reliability_prior = np.asarray(prior_reliability, dtype=np.float64)
    association = (
        None
        if association_probability is None
        else np.asarray(association_probability, dtype=np.float64)
    )
    center_count = len(prior.center_ids)
    _require(
        frame_index >= 0
        and (
            prior.last_update_frame is None
            or frame_index > prior.last_update_frame
        ),
        "RBF updates must have increasing nonnegative frames",
    )
    _require(
        positions.shape == residual.shape == (center_count, 3)
        and covariance.shape == (center_count, 3, 3)
        and mask.shape == reliability_prior.shape == (center_count,),
        "covariance-aware RBF inputs changed shape",
    )
    _require(
        np.all(np.isfinite(reliability_prior))
        and np.all((reliability_prior >= 0.0) & (reliability_prior <= 1.0)),
        "prior reliability must lie in [0, 1]",
    )
    if association is not None:
        _require(
            association.shape == (center_count,)
            and np.all(np.isfinite(association))
            and np.all((association >= 0.0) & (association <= 1.0)),
            "association probability must lie in [0, 1]",
        )
        mask &= association > 0.0
    mask &= np.all(np.isfinite(positions), axis=1)
    mask &= np.all(np.isfinite(residual), axis=1)
    mask &= np.all(np.isfinite(covariance), axis=(1, 2))
    mask &= reliability_prior > 0.0

    elapsed = (
        0
        if prior.last_update_frame is None
        else frame_index - prior.last_update_frame
    )
    process_variance = elapsed * config.process_std_m_per_sqrt_frame**2
    global_variance = prior.global_variance_m2 + process_variance
    local_variance = prior.local_variance_m2 + process_variance
    global_mean = prior.global_mean_m.copy()
    local_mean = prior.local_mean_m.copy()
    update_count = prior.update_count.copy()
    posterior_reliability = np.zeros(center_count, dtype=np.float64)

    if np.any(mask):
        selected = residual[mask]
        selected_covariance = covariance[mask]
        selected_prior_reliability = reliability_prior[mask]
        covariance_bound = np.maximum(
            _scalar_covariance_bound(selected_covariance),
            config.observation_std_m**2,
        )
        robust_location = np.median(selected, axis=0)
        absolute_deviation = np.abs(selected - robust_location)
        robust_scale = np.maximum(
            1.4826 * np.median(absolute_deviation, axis=0),
            config.observation_std_m,
        )
        if association is None:
            posterior_inlier = None
            effective_count = max(
                config.minimum_reliability,
                float(np.sum(selected_prior_reliability)),
            )
        else:
            selected_association = association[mask]
            nominal_inlier = np.clip(
                selected_prior_reliability * selected_association,
                0.0,
                1.0,
            )
            mixture = robust_mixture_likelihood(
                PseudoMeasurementBatch(
                    observed=selected,
                    predicted=np.repeat(
                        robust_location[None],
                        len(selected),
                        axis=0,
                    ),
                    variance=(
                        covariance_bound[:, None]
                        + np.square(robust_scale)[None]
                    ),
                ),
                prior_reliability=nominal_inlier,
            )
            posterior_inlier = mixture.posterior_inlier_probability
            effective_count = max(
                1e-6,
                float(np.sum(posterior_inlier)),
            )
        global_observation_variance = (
            np.square(robust_scale) + np.median(covariance_bound)
        ) / effective_count
        global_gain = global_variance / (
            global_variance + global_observation_variance
        )
        global_mean += global_gain * (robust_location - global_mean)
        global_variance *= 1.0 - global_gain

        local_observation = selected - global_mean
        if posterior_inlier is None:
            squared_radius = np.sum(
                np.square(local_observation)
                / (
                    covariance_bound[:, None]
                    + np.square(robust_scale)[None]
                ),
                axis=1,
            )
            robust_reliability = np.clip(
                (config.degrees_of_freedom + 3.0)
                / (config.degrees_of_freedom + squared_radius),
                config.minimum_reliability,
                1.0,
            )
            combined_reliability = np.clip(
                selected_prior_reliability * robust_reliability,
                config.minimum_reliability,
                1.0,
            )
        else:
            combined_reliability = posterior_inlier
        selected_ids = np.flatnonzero(mask)
        posterior_reliability[selected_ids] = combined_reliability
        for local_index, center_index in enumerate(selected_ids):
            observation_variance = (
                covariance_bound[local_index]
                / max(combined_reliability[local_index], 1e-6)
            )
            gain = local_variance[center_index] / (
                local_variance[center_index] + observation_variance
            )
            local_mean[center_index] += gain * (
                local_observation[local_index]
                - local_mean[center_index]
            )
            local_variance[center_index] *= 1.0 - gain
            update_count[center_index] += 1

    posterior = RecursiveRbfBeliefSnapshot(
        center_ids=prior.center_ids,
        center_positions_m=np.where(
            mask[:, None],
            positions,
            prior.center_positions_m,
        ),
        global_mean_m=global_mean,
        global_variance_m2=np.maximum(global_variance, 1e-12),
        local_mean_m=local_mean,
        local_variance_m2=np.maximum(local_variance, 1e-12),
        update_count=update_count,
        last_update_frame=frame_index,
        object_scale_m=prior.object_scale_m,
    )
    posterior_reliability.setflags(write=False)
    return posterior, posterior_reliability


def predict_dynamic_tapnextpp_candidate(
    physical_prediction_m: np.ndarray,
    persistence_prediction_m: np.ndarray,
    measurements: BirthAnchoredMeasurements,
    *,
    update_frames: Sequence[int] = UPDATE_FRAMES,
    gate_config: PairwiseCorrespondenceGateConfig | None = None,
    belief_config: RecursiveRbfBeliefConfig | None = None,
    assimilation_mode: str = LEGACY_RELIABILITY_ASSIMILATION,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Produce a complete target-free candidate with exact rejection fallback."""

    physical_input = np.asarray(physical_prediction_m)
    persistence_input = np.asarray(persistence_prediction_m)
    physical = np.asarray(physical_input, dtype=np.float64)
    persistence = np.asarray(persistence_input, dtype=np.float64)
    _require(
        physical.shape == persistence.shape == measurements.measurement_m.shape
        and physical.ndim == 3
        and physical.shape[2] == 3
        and np.all(np.isfinite(physical))
        and np.all(np.isfinite(persistence)),
        "candidate backbones differ or are invalid",
    )
    _require(
        np.array_equal(physical_input[0], persistence_input[0]),
        "candidate backbones changed frame-zero identities",
    )
    updates = tuple(map(int, update_frames))
    _require(
        updates == UPDATE_FRAMES and updates[-1] < len(physical),
        "dynamic update frames changed",
    )
    _require(
        assimilation_mode in _ASSIMILATION_MODES,
        f"unsupported assimilation mode {assimilation_mode!r}",
    )
    centers = measurements.entity_ids
    gate_cfg = gate_config or PairwiseCorrespondenceGateConfig()
    belief_cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    backbones = {
        PHYSICAL_ARM: physical,
        PERSISTENCE_ARM: persistence,
    }
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
    candidate_trajectory = physical_input.copy()
    candidate_variance = np.zeros_like(physical, dtype=np.float64)
    update_records: list[dict[str, Any]] = []
    output_dtype = physical_input.dtype

    for update_index, update in enumerate(updates):
        stop = updates[update_index + 1] if update_index + 1 < len(updates) else len(
            physical
        )
        available = measurements.available[update, centers].copy()
        available &= (
            measurements.prior_reliability[update, centers]
            >= belief_cfg.minimum_reliability
        )
        if assimilation_mode == SET_VALUED_MIXTURE_ASSIMILATION:
            available &= (
                measurements.association_probability[update, centers] > 0.0
            )
        available_ids = centers[available]
        observed = measurements.measurement_m[update, available_ids]
        selector_support = len(available_ids) >= MINIMUM_SELECTOR_SUPPORT
        if selector_support:
            chamfer = {
                name: _symmetric_set_chamfer_m(
                    backbone[update, available_ids],
                    observed,
                )
                for name, backbone in backbones.items()
            }
            selected_name = min(
                (PHYSICAL_ARM, PERSISTENCE_ARM),
                key=lambda name: (
                    chamfer[name],
                    0 if name == PHYSICAL_ARM else 1,
                ),
            )
        else:
            chamfer = (
                {
                    name: _symmetric_set_chamfer_m(
                        backbone[update, available_ids],
                        observed,
                    )
                    for name, backbone in backbones.items()
                }
                if len(available_ids)
                else {PHYSICAL_ARM: None, PERSISTENCE_ARM: None}
            )
            selected_name = PERSISTENCE_ARM
        selected = backbones[selected_name]
        selected_trajectory[update + 1 : stop] = selected[update + 1 : stop]
        candidate_trajectory[update + 1 : stop] = selected[update + 1 : stop]

        gates: dict[str, Any] = {}
        residuals: dict[str, np.ndarray] = {}
        for backbone_name, backbone in backbones.items():
            residual = np.full((len(centers), 3), np.nan, dtype=np.float64)
            residual[available] = observed - backbone[update, available_ids]
            residuals[backbone_name] = residual
            gates[backbone_name] = detect_pairwise_consensus_correspondences(
                backbone[update, centers],
                measurements.measurement_m[update, centers],
                available,
                material_ids=centers,
                config=gate_cfg,
            )

        selected_gate = gates[selected_name]
        accepted = bool(selector_support and selected_gate.accepted)
        robust_reliability = np.zeros(len(centers), dtype=np.float64)
        if accepted:
            covariance = measurements.covariance_m2[update, centers].copy()
            covariance[~selected_gate.inlier_mask] = np.eye(3)
            states[selected_name], robust_reliability = (
                _update_covariance_aware_rbf_belief(
                    states[selected_name],
                    update,
                    selected[update, centers],
                    residuals[selected_name],
                    selected_gate.inlier_mask.copy(),
                    covariance,
                    measurements.prior_reliability[update, centers],
                    config=belief_cfg,
                    association_probability=(
                        measurements.association_probability[update, centers]
                        if assimilation_mode
                        == SET_VALUED_MIXTURE_ASSIMILATION
                        else None
                    ),
                )
            )
            for frame in range(update + 1, stop):
                decoded = decode_recursive_rbf_belief(
                    states[selected_name],
                    selected[update],
                    forecast_frames=frame - update,
                    config=belief_cfg,
                )
                candidate_trajectory[frame] = (
                    np.asarray(selected[frame], dtype=np.float64)
                    + decoded.mean_m
                ).astype(output_dtype, copy=False)
                candidate_variance[frame] = decoded.variance_m2

        exact_fallback = bool(
            accepted
            or np.array_equal(
                candidate_trajectory[update + 1 : stop],
                selected[update + 1 : stop],
            )
        )
        _require(exact_fallback, "rejected dynamic update changed the backbone")
        update_records.append(
            {
                "frame": update,
                "interval_end_exclusive": stop,
                "available_center_count": int(len(available_ids)),
                "selector_support_sufficient": selector_support,
                "selected_backbone": selected_name,
                "current_observation_chamfer_m": chamfer,
                "pairwise_gate": {
                    "accepted": accepted,
                    "decision": (
                        selected_gate.decision
                        if selector_support
                        else "insufficient-selector-support"
                    ),
                    "inlier_count": int(selected_gate.inlier_count),
                    "inlier_fraction": float(selected_gate.inlier_fraction),
                    "compatible_pair_fraction": float(
                        selected_gate.compatible_pair_fraction
                    ),
                    "bit_exact_fallback": bool(not accepted and exact_fallback),
                },
                "mean_prior_reliability": (
                    float(
                        np.mean(
                            measurements.prior_reliability[
                                update,
                                available_ids,
                            ]
                        )
                    )
                    if len(available_ids)
                    else 0.0
                ),
                "mean_posterior_robust_reliability": (
                    float(np.mean(robust_reliability[selected_gate.inlier_mask]))
                    if accepted
                    else 0.0
                ),
            }
        )

    arrays = {
        PHYSICAL_ARM: physical_input.copy(),
        PERSISTENCE_ARM: persistence_input.copy(),
        SELECTED_BACKBONE_ARM: selected_trajectory,
        CANDIDATE_ARM: candidate_trajectory,
        "candidate_correction_variance_m2": candidate_variance,
    }
    report = {
        "schema_version": 1,
        "artifact_kind": "Deform360DynamicTAPNextPPAssimilation",
        "center_ids": centers.tolist(),
        "update_frames": list(updates),
        "gate_config": asdict(gate_cfg),
        "belief_config": asdict(belief_cfg),
        "assimilation_mode": assimilation_mode,
        "updates": update_records,
        "method_contract": {
            "absolute_multiview_position_used_as_state_innovation": False,
            "birth_anchored_displacement_innovation": True,
            "unknown_temporal_correlation_covariance_bound": (
                "2*(Sigma_birth+Sigma_update)"
            ),
            "metric_covariance_propagated_into_rbf_update": True,
            "prior_reliability_uses_physical_innovation": False,
            "association_probability_used_as_prior_reliability": False,
            "association_probability_role": (
                "separate nominal assignment event inside robust mixture"
                if assimilation_mode == SET_VALUED_MIXTURE_ASSIMILATION
                else "recorded but unused by frozen v2"
            ),
            "innovation_robustified_once_inside_rbf_update": True,
            "pairwise_gate_role": "identity-consistency admission",
            "rejection": "bit-exact selected physical-or-persistence backbone",
        },
        "information_boundary": {
            "future_target_read": False,
            "future_object_geometry_read": False,
            "prediction_depends_on": (
                "sealed physical/persistence backbones and causal RGB/depth/mask "
                "prefix observations through each registered update"
            ),
        },
    }
    return report, arrays


__all__ = [
    "BirthAnchoredMeasurements",
    "CANDIDATE_ARM",
    "LEGACY_RELIABILITY_ASSIMILATION",
    "PHYSICAL_ARM",
    "PERSISTENCE_ARM",
    "SELECTED_BACKBONE_ARM",
    "SET_VALUED_MIXTURE_ASSIMILATION",
    "UPDATE_FRAMES",
    "build_birth_anchored_measurements",
    "predict_dynamic_tapnextpp_candidate",
]
