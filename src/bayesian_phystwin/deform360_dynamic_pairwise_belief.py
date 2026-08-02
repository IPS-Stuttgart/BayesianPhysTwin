"""Dynamic, nuisance-aware sparse observations for recursive Deform360 belief.

The predictor consumes a causal pool of registered material observations.  It
never accepts a target trajectory.  At each update it first chooses a compact
candidate set without using the innovation, applies the exact pairwise
association gate, then updates a dual physical/persistence RBF belief only when
the observed motion and decoded correction are supported by the known physical
response.  Every rejection preserves the selected backbone and prior belief
state exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .bias_aware_belief import build_physical_response_basis
from .deform360_cpd_diagnostic import _symmetric_set_chamfer_m
from .deform360_online_belief_evaluation import (
    UPDATE_FRAMES,
    robust_huber_continuation_gain,
)
from .deform360_pairwise_bias_aware_development import _spatial_bias_basis
from .deform360_raw_pairwise_correspondence_diagnostic import (
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    _corrected_frame,
)
from .nuisance_aware_information import (
    NuisanceAwareInformationState,
    greedy_nuisance_aware_selection,
)
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

PROTOCOL_ID = "deform360-dynamic-pairwise-belief-open27-v1-development"
SELECTED_BACKBONE_ARM = "dynamic_pool_selected_backbone"
DYNAMIC_PAIRWISE_ARM = "dynamic_nuisance_pairwise_recursive_rbf"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _radial_rms(value_m: np.ndarray) -> float:
    value = np.asarray(value_m, dtype=np.float64)
    if not len(value):
        return 0.0
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))


def _vector_cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _student_t_reliability(
    residual_m: np.ndarray,
    scale_m: np.ndarray,
    *,
    degrees_of_freedom: float,
    minimum: float,
) -> np.ndarray:
    standardized = residual_m / scale_m[None]
    squared_radius = np.sum(np.square(standardized), axis=1)
    dimension = residual_m.shape[1]
    reliability = (degrees_of_freedom + dimension) / (
        degrees_of_freedom + squared_radius
    )
    return np.clip(reliability, minimum, 1.0)


def update_metric_recursive_rbf_belief(
    prior: RecursiveRbfBeliefSnapshot,
    frame_index: int,
    center_positions_m: np.ndarray,
    measured_residual_m: np.ndarray,
    available: np.ndarray,
    *,
    prior_reliability: np.ndarray,
    observation_variance_m2: np.ndarray,
    config: RecursiveRbfBeliefConfig,
) -> tuple[RecursiveRbfBeliefSnapshot, np.ndarray]:
    """Apply one metric, reliability-aware robust update to a frozen RBF prior."""

    if frame_index < 0:
        raise ValueError("frame_index must be nonnegative")
    if prior.last_update_frame is not None and frame_index <= prior.last_update_frame:
        raise ValueError("updates must have strictly increasing frame indices")
    positions = np.asarray(center_positions_m, dtype=np.float64)
    residual = np.asarray(measured_residual_m, dtype=np.float64)
    mask = np.asarray(available, dtype=bool).copy()
    reliability_prior = np.asarray(prior_reliability, dtype=np.float64)
    metric_variance = np.asarray(observation_variance_m2, dtype=np.float64)
    center_count = len(prior.center_ids)
    expected_vector = (center_count,)
    if positions.shape != (center_count, 3) or residual.shape != (center_count, 3):
        raise ValueError("centre positions and residuals must have shape (K, 3)")
    if (
        mask.shape != expected_vector
        or reliability_prior.shape != expected_vector
        or metric_variance.shape != expected_vector
    ):
        raise ValueError("metric update vectors must have shape (K,)")
    if not np.all(np.isfinite(reliability_prior)) or np.any(
        (reliability_prior < 0.0) | (reliability_prior > 1.0)
    ):
        raise ValueError("prior reliability must lie in [0, 1]")
    if not np.all(np.isfinite(metric_variance)) or np.any(metric_variance <= 0.0):
        raise ValueError("metric observation variance must be positive")
    mask &= np.all(np.isfinite(positions), axis=1)
    mask &= np.all(np.isfinite(residual), axis=1)
    mask &= reliability_prior > 0.0

    elapsed = (
        0 if prior.last_update_frame is None else frame_index - prior.last_update_frame
    )
    process_variance = elapsed * config.process_std_m_per_sqrt_frame**2
    global_variance = prior.global_variance_m2 + process_variance
    local_variance = prior.local_variance_m2 + process_variance
    global_mean = prior.global_mean_m.copy()
    local_mean = prior.local_mean_m.copy()
    update_count = prior.update_count.copy()
    combined_reliability = np.zeros(center_count, dtype=np.float64)

    if np.any(mask):
        selected = residual[mask]
        selected_prior = reliability_prior[mask]
        selected_variance = metric_variance[mask]
        robust_location = np.median(selected, axis=0)
        absolute_deviation = np.abs(selected - robust_location)
        robust_scale = 1.4826 * np.median(absolute_deviation, axis=0)
        robust_scale = np.maximum(robust_scale, config.observation_std_m)

        effective_count = max(float(np.sum(selected_prior)), 1e-15)
        metric_variance_floor = float(
            np.average(selected_variance, weights=selected_prior)
        )
        global_observation_variance = np.maximum(
            np.square(robust_scale), metric_variance_floor
        ) / effective_count
        global_gain = global_variance / (global_variance + global_observation_variance)
        global_mean += global_gain * (robust_location - global_mean)
        global_variance *= 1.0 - global_gain

        local_observation = selected - global_mean
        robust_reliability = _student_t_reliability(
            local_observation,
            np.maximum(robust_scale, config.observation_std_m),
            degrees_of_freedom=config.degrees_of_freedom,
            minimum=config.minimum_reliability,
        )
        selected_reliability = selected_prior * robust_reliability
        selected_ids = np.flatnonzero(mask)
        combined_reliability[selected_ids] = selected_reliability
        for local_index, center_index in enumerate(selected_ids):
            observation_variance = (
                selected_variance[local_index] / selected_reliability[local_index]
            )
            gain = local_variance[center_index] / (
                local_variance[center_index] + observation_variance
            )
            local_mean[center_index] += gain * (
                local_observation[local_index] - local_mean[center_index]
            )
            local_variance[center_index] *= 1.0 - gain
            update_count[center_index] += 1

    posterior = RecursiveRbfBeliefSnapshot(
        center_ids=prior.center_ids,
        center_positions_m=np.where(mask[:, None], positions, prior.center_positions_m),
        global_mean_m=global_mean,
        global_variance_m2=np.maximum(global_variance, 1e-12),
        local_mean_m=local_mean,
        local_variance_m2=np.maximum(local_variance, 1e-12),
        update_count=update_count,
        last_update_frame=frame_index,
        object_scale_m=prior.object_scale_m,
    )
    combined_reliability.setflags(write=False)
    return posterior, combined_reliability


@dataclass(frozen=True)
class DynamicPairwiseBeliefConfig:
    """Frozen source-development choices for the dynamic observation arm."""

    update_frames: tuple[int, ...] = UPDATE_FRAMES
    observation_pool_count: int = 64
    association_candidate_count: int = 24
    active_center_count: int = 16
    minimum_inlier_view_count: int = 3
    minimum_action_support: float = 0.10
    physical_response_rank: int = 4
    minimum_motion_center_count: int = 3
    minimum_physical_response_m: float = 0.0005
    minimum_observed_motion_m: float = 0.0005
    minimum_physical_agreement_gain: float = 0.40
    minimum_correction_physical_cosine: float = 0.0
    maximum_correction_to_physical_motion_ratio: float = 2.0
    observation_variance_floor_m2: float = 0.005**2
    information_effective_sample_cap: float = 8.0
    state_prior_std_m: float = 0.02
    nuisance_prior_std_m: float = 0.02
    minimum_information_gain_nats: float = 0.0
    pairwise_gate: PairwiseCorrespondenceGateConfig = field(
        default_factory=PairwiseCorrespondenceGateConfig
    )
    belief: RecursiveRbfBeliefConfig = field(
        default_factory=lambda: RecursiveRbfBeliefConfig(
            length_scale_fraction=0.10,
            local_blend=1.0,
        )
    )

    def __post_init__(self) -> None:
        _require(
            tuple(sorted(set(self.update_frames))) == self.update_frames
            and bool(self.update_frames),
            "update_frames must be nonempty and strictly increasing",
        )
        _require(self.observation_pool_count >= 1, "pool count must be positive")
        _require(
            self.pairwise_gate.minimum_inlier_count
            <= self.active_center_count
            <= self.association_candidate_count
            <= self.pairwise_gate.maximum_exact_center_count,
            "dynamic center counts violate the exact pairwise gate contract",
        )
        _require(
            self.association_candidate_count <= self.observation_pool_count,
            "association candidate count exceeds the observation pool",
        )
        _require(
            self.minimum_inlier_view_count >= 2,
            "at least two views are required",
        )
        _require(
            0.0 <= self.minimum_action_support <= 1.0,
            "minimum action support must lie in [0, 1]",
        )
        _require(self.physical_response_rank >= 1, "physical rank must be positive")
        _require(
            self.minimum_motion_center_count >= 1,
            "minimum motion center count must be positive",
        )
        positive = (
            self.minimum_physical_response_m,
            self.minimum_observed_motion_m,
            self.maximum_correction_to_physical_motion_ratio,
            self.observation_variance_floor_m2,
            self.information_effective_sample_cap,
            self.state_prior_std_m,
            self.nuisance_prior_std_m,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "dynamic belief scales must be positive",
        )
        _require(
            0.0 <= self.minimum_physical_agreement_gain <= 1.0,
            "physical agreement gain must lie in [0, 1]",
        )
        _require(
            -1.0 <= self.minimum_correction_physical_cosine <= 1.0,
            "correction cosine threshold must lie in [-1, 1]",
        )
        _require(
            np.isfinite(self.minimum_information_gain_nats)
            and self.minimum_information_gain_nats >= 0.0,
            "minimum information gain must be nonnegative",
        )


@dataclass(frozen=True)
class DynamicCenterSelection:
    """One deterministic residual-independent center selection."""

    selected_pool_offsets: np.ndarray
    mutual_information_nats: np.ndarray
    effective_reliability: np.ndarray

    def __post_init__(self) -> None:
        offsets = np.asarray(self.selected_pool_offsets, dtype=np.int64).copy()
        gain = np.asarray(self.mutual_information_nats, dtype=np.float64).copy()
        reliability = np.asarray(self.effective_reliability, dtype=np.float64).copy()
        _require(offsets.ndim == 1, "selected offsets must be a vector")
        _require(gain.shape == offsets.shape, "information gain shape changed")
        _require(reliability.ndim == 1, "effective reliability must be a vector")
        _require(len(np.unique(offsets)) == len(offsets), "selected offsets repeat")
        for value in (offsets, gain, reliability):
            value.setflags(write=False)
        object.__setattr__(self, "selected_pool_offsets", offsets)
        object.__setattr__(self, "mutual_information_nats", gain)
        object.__setattr__(self, "effective_reliability", reliability)


def select_dynamic_centers(
    state_basis: np.ndarray,
    nuisance_basis: np.ndarray,
    observation_variance_m2: np.ndarray,
    prior_reliability: np.ndarray,
    available: np.ndarray,
    *,
    count: int,
    config: DynamicPairwiseBeliefConfig,
) -> DynamicCenterSelection:
    """Select centers without reading their innovation against the twin."""

    state = np.asarray(state_basis, dtype=np.float64)
    nuisance = np.asarray(nuisance_basis, dtype=np.float64)
    variance = np.asarray(observation_variance_m2, dtype=np.float64)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    row_count = len(state)
    _require(state.ndim == 2 and state.shape[1] >= 1, "state basis is empty")
    _require(nuisance.ndim == 2 and len(nuisance) == row_count, "nuisance shape changed")
    _require(variance.shape == (row_count,), "variance shape changed")
    _require(reliability.shape == mask.shape == (row_count,), "selection masks differ")
    _require(np.all(np.isfinite(state)) and np.all(np.isfinite(nuisance)), "basis is not finite")
    _require(
        np.all(np.isfinite(variance)) and np.all(variance > 0.0),
        "variance must be finite and positive",
    )
    _require(
        np.all(np.isfinite(reliability))
        and np.all((reliability >= 0.0) & (reliability <= 1.0)),
        "reliability must lie in [0, 1]",
    )
    eligible = np.flatnonzero(mask & (reliability > 0.0))
    if not len(eligible) or count <= 0:
        return DynamicCenterSelection(
            selected_pool_offsets=np.empty(0, dtype=np.int64),
            mutual_information_nats=np.empty(0, dtype=np.float64),
            effective_reliability=np.zeros(row_count, dtype=np.float64),
        )
    capped = np.zeros_like(reliability)
    capped[eligible] = reliability[eligible]
    total = float(np.sum(capped))
    if total > config.information_effective_sample_cap:
        capped[eligible] *= config.information_effective_sample_cap / total
    prior = NuisanceAwareInformationState.from_independent_priors(
        np.eye(state.shape[1], dtype=np.float64) / config.state_prior_std_m**2,
        (
            np.eye(nuisance.shape[1], dtype=np.float64)
            / config.nuisance_prior_std_m**2
            if nuisance.shape[1]
            else None
        ),
    )
    result = greedy_nuisance_aware_selection(
        prior,
        tuple(state[index][None] for index in eligible),
        tuple(nuisance[index][None] for index in eligible),
        tuple(np.asarray([[variance[index]]], dtype=np.float64) for index in eligible),
        reliabilities=tuple(float(capped[index]) for index in eligible),
        count=min(int(count), len(eligible)),
        minimum_gain_nats=config.minimum_information_gain_nats,
    )
    return DynamicCenterSelection(
        selected_pool_offsets=eligible[result.selected_indices],
        mutual_information_nats=result.mutual_information_nats,
        effective_reliability=capped,
    )


def _validate_inputs(
    physical: np.ndarray,
    persistence: np.ndarray,
    response: np.ndarray,
    frame_zero: np.ndarray,
    action_support: np.ndarray,
    measurement: np.ndarray,
    visible: np.ndarray,
    valid: np.ndarray,
    pool_ids: np.ndarray,
    reliability: np.ndarray,
    variance: np.ndarray,
    inlier_views: np.ndarray,
    config: DynamicPairwiseBeliefConfig,
) -> None:
    _require(
        physical.ndim == 3 and physical.shape[2] == 3,
        "physical prior must have shape (T, N, 3)",
    )
    _require(
        persistence.shape == response.shape == measurement.shape == physical.shape,
        "trajectory inputs differ",
    )
    _require(frame_zero.shape == physical.shape[1:], "frame-zero shape changed")
    _require(action_support.shape == (physical.shape[1],), "action support shape changed")
    _require(visible.shape == valid.shape == physical.shape[:2], "measurement masks differ")
    _require(pool_ids.shape == (config.observation_pool_count,), "observation pool count changed")
    _require(len(np.unique(pool_ids)) == len(pool_ids), "observation pool IDs repeat")
    _require(np.all((pool_ids >= 0) & (pool_ids < physical.shape[1])), "pool ID exceeds graph")
    expected = (len(config.update_frames), len(pool_ids))
    _require(reliability.shape == variance.shape == inlier_views.shape == expected, "pool diagnostics differ")
    _require(config.update_frames[-1] < len(physical), "update exceeds trajectory")
    finite = (physical, persistence, response, frame_zero, action_support, reliability, variance)
    _require(all(np.all(np.isfinite(value)) for value in finite), "finite input contract failed")
    _require(np.all((action_support >= 0.0) & (action_support <= 1.0)), "action support range changed")
    _require(np.all((reliability >= 0.0) & (reliability <= 1.0)), "reliability range changed")
    _require(np.all(variance > 0.0), "variance must be positive")


def _propose_belief_updates(
    states: dict[str, RecursiveRbfBeliefSnapshot],
    backbones: dict[str, np.ndarray],
    measurement: np.ndarray,
    pool_ids: np.ndarray,
    selected_pool_offsets: np.ndarray,
    gates: dict[str, Any],
    effective_reliability: np.ndarray,
    observation_variance_m2: np.ndarray,
    update: int,
    config: DynamicPairwiseBeliefConfig,
) -> dict[str, RecursiveRbfBeliefSnapshot]:
    proposed: dict[str, RecursiveRbfBeliefSnapshot] = {}
    selected_mask = np.zeros(len(pool_ids), dtype=bool)
    selected_mask[selected_pool_offsets] = True
    for name, backbone in backbones.items():
        available = selected_mask & gates[name].inlier_mask
        residual = measurement[update, pool_ids] - backbone[update, pool_ids]
        proposed[name], _ = update_metric_recursive_rbf_belief(
            states[name],
            update,
            backbone[update, pool_ids],
            residual,
            available,
            prior_reliability=effective_reliability,
            observation_variance_m2=observation_variance_m2,
            config=config.belief,
        )
    return proposed


def predict_dynamic_pairwise_belief_arrays(
    physical_prior_m: np.ndarray,
    persistence_m: np.ndarray,
    physical_response_m: np.ndarray,
    frame_zero_points_m: np.ndarray,
    action_support: np.ndarray,
    measurement_m: np.ndarray,
    measurement_visibility: np.ndarray,
    measurement_validity: np.ndarray,
    *,
    pool_ids: np.ndarray,
    prior_reliability: np.ndarray,
    observation_variance_m2: np.ndarray,
    inlier_view_count: np.ndarray,
    config: DynamicPairwiseBeliefConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build one target-free dynamic observation candidate."""

    cfg = config or DynamicPairwiseBeliefConfig()
    physical_input = np.asarray(physical_prior_m)
    persistence_input = np.asarray(persistence_m)
    physical = np.asarray(physical_input, dtype=np.float64)
    persistence = np.asarray(persistence_input, dtype=np.float64)
    response = np.asarray(physical_response_m, dtype=np.float64)
    frame_zero = np.asarray(frame_zero_points_m, dtype=np.float64)
    support = np.asarray(action_support, dtype=np.float64)
    measurement = np.asarray(measurement_m, dtype=np.float64)
    visible = np.asarray(measurement_visibility, dtype=bool)
    valid = np.asarray(measurement_validity, dtype=bool)
    pool = np.asarray(pool_ids, dtype=np.int64)
    reliability = np.asarray(prior_reliability, dtype=np.float64)
    variance = np.asarray(observation_variance_m2, dtype=np.float64)
    view_count = np.asarray(inlier_view_count, dtype=np.int64)
    _validate_inputs(
        physical,
        persistence,
        response,
        frame_zero,
        support,
        measurement,
        visible,
        valid,
        pool,
        reliability,
        variance,
        view_count,
        cfg,
    )

    # Archive row order is not evidence. Canonicalize material identities before
    # every tie-broken selection and carry their diagnostics along with them.
    pool_order = np.argsort(pool, kind="mergesort")
    pool = pool[pool_order]
    reliability = reliability[:, pool_order]
    variance = variance[:, pool_order]
    view_count = view_count[:, pool_order]

    backbones = {PHYSICAL_ARM: physical, PERSISTENCE_ARM: persistence}
    states = {
        name: initialize_recursive_rbf_belief(
            pool,
            backbone[0, pool],
            backbone[0],
            config=cfg.belief,
        )
        for name, backbone in backbones.items()
    }
    selected_trajectory = physical_input.copy()
    candidate = physical_input.copy()
    output_dtype = physical_input.dtype
    spatial_bias = _spatial_bias_basis(frame_zero)
    previous_update = 0
    update_reports: list[dict[str, Any]] = []

    for update_index, update in enumerate(cfg.update_frames):
        stop = (
            cfg.update_frames[update_index + 1]
            if update_index + 1 < len(cfg.update_frames)
            else len(physical)
        )
        selected_trajectory[update + 1 : stop] = persistence_input[update + 1 : stop]
        candidate[update + 1 : stop] = persistence_input[update + 1 : stop]
        record: dict[str, Any] = {
            "frame": int(update),
            "interval_end_exclusive": int(stop),
            "accepted": False,
            "decision": "exact-selected-backbone-fallback",
            "bit_exact_selected_backbone_fallback": True,
            "preassociation_pool_ids": [],
            "active_center_ids": [],
            "selected_backbone": PERSISTENCE_ARM,
        }
        try:
            physical_basis = build_physical_response_basis(
                response[: update + 1],
                action_support=support,
                rank=cfg.physical_response_rank,
                minimum_response_m=cfg.minimum_physical_response_m,
            )
            available = (
                visible[update, pool]
                & valid[update, pool]
                & np.all(np.isfinite(measurement[update, pool]), axis=1)
                & (view_count[update_index] >= cfg.minimum_inlier_view_count)
                & (support[pool] >= cfg.minimum_action_support)
                & (reliability[update_index] > 0.0)
            )
            nuisance = np.column_stack((spatial_bias[pool], np.ones(len(pool))))
            preliminary = select_dynamic_centers(
                physical_basis.basis[pool],
                nuisance,
                variance[update_index],
                reliability[update_index],
                available,
                count=cfg.association_candidate_count,
                config=cfg,
            )
            preliminary_mask = np.zeros(len(pool), dtype=bool)
            preliminary_mask[preliminary.selected_pool_offsets] = True
            record["eligible_pool_count"] = int(np.sum(available))
            record["preassociation_pool_ids"] = pool[
                preliminary.selected_pool_offsets
            ].tolist()
            _require(
                len(preliminary.selected_pool_offsets)
                >= cfg.pairwise_gate.minimum_inlier_count,
                "insufficient residual-independent association support",
            )

            current_chamfer = {
                name: _symmetric_set_chamfer_m(
                    backbone[update, pool[preliminary.selected_pool_offsets]],
                    measurement[update, pool[preliminary.selected_pool_offsets]],
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
            selected = backbones[selected_name]
            selected_trajectory[update + 1 : stop] = selected[update + 1 : stop]
            candidate[update + 1 : stop] = selected[update + 1 : stop]
            record["selected_backbone"] = selected_name
            record["current_observation_chamfer_m"] = current_chamfer

            gates = {
                name: detect_pairwise_consensus_correspondences(
                    backbone[update, pool],
                    measurement[update, pool],
                    preliminary_mask,
                    material_ids=pool,
                    config=cfg.pairwise_gate,
                )
                for name, backbone in backbones.items()
            }
            selected_gate = gates[selected_name]
            record["pairwise_gate"] = {
                "accepted": selected_gate.accepted,
                "decision": selected_gate.decision,
                "inlier_count": selected_gate.inlier_count,
                "inlier_fraction": selected_gate.inlier_fraction,
                "compatible_pair_fraction": selected_gate.compatible_pair_fraction,
            }
            _require(selected_gate.accepted, selected_gate.decision)

            previous_available = (
                visible[previous_update, pool]
                & valid[previous_update, pool]
                & np.all(np.isfinite(measurement[previous_update, pool]), axis=1)
            )
            motion_mask = selected_gate.inlier_mask & previous_available
            physical_delta = response[update, pool] - response[previous_update, pool]
            observed_delta = (
                measurement[update, pool] - measurement[previous_update, pool]
            )
            physical_rms = _radial_rms(physical_delta[motion_mask])
            observed_rms = _radial_rms(observed_delta[motion_mask])
            agreement = robust_huber_continuation_gain(
                physical_delta[motion_mask],
                observed_delta[motion_mask],
                minimum_point_count=cfg.minimum_motion_center_count,
                fallback=0.0,
            )
            record.update(
                {
                    "motion_center_count": int(np.sum(motion_mask)),
                    "physical_response_rms_m": physical_rms,
                    "observed_motion_rms_m": observed_rms,
                    "causal_physical_agreement_gain": agreement,
                }
            )
            _require(
                int(np.sum(motion_mask)) >= cfg.minimum_motion_center_count,
                "insufficient causal motion support",
            )
            _require(
                physical_rms >= cfg.minimum_physical_response_m,
                "physical response is below threshold",
            )
            _require(
                observed_rms >= cfg.minimum_observed_motion_m,
                "observed motion is below threshold",
            )
            _require(
                agreement >= cfg.minimum_physical_agreement_gain,
                "observed motion disagrees with physical response",
            )

            active = select_dynamic_centers(
                physical_basis.basis[pool],
                nuisance,
                variance[update_index],
                reliability[update_index],
                selected_gate.inlier_mask,
                count=cfg.active_center_count,
                config=cfg,
            )
            _require(
                len(active.selected_pool_offsets)
                >= cfg.pairwise_gate.minimum_inlier_count,
                "insufficient nuisance-distinguishable pairwise support",
            )
            proposed = _propose_belief_updates(
                states,
                backbones,
                measurement,
                pool,
                active.selected_pool_offsets,
                gates,
                active.effective_reliability,
                variance[update_index],
                update,
                cfg,
            )
            decoded = decode_recursive_rbf_belief(
                proposed[selected_name],
                selected[update],
                forecast_frames=1,
                config=cfg.belief,
            )
            active_ids = pool[active.selected_pool_offsets]
            correction_rms = _radial_rms(decoded.mean_m[active_ids])
            physical_continuation = (
                physical[min(stop - 1, len(physical) - 1), active_ids]
                - physical[update, active_ids]
            )
            continuation_rms = _radial_rms(physical_continuation)
            correction_ratio = correction_rms / max(continuation_rms, 1e-12)
            correction_cosine = _vector_cosine(
                decoded.mean_m[active_ids], physical_continuation
            )
            record.update(
                {
                    "active_center_ids": active_ids.tolist(),
                    "active_information_gain_nats": active.mutual_information_nats.tolist(),
                    "effective_reliability_sum": float(
                        np.sum(
                            active.effective_reliability[
                                active.selected_pool_offsets
                            ]
                        )
                    ),
                    "correction_rms_m": correction_rms,
                    "physical_continuation_rms_m": continuation_rms,
                    "correction_to_physical_motion_ratio": correction_ratio,
                    "correction_physical_cosine": correction_cosine,
                }
            )
            _require(
                continuation_rms >= cfg.minimum_physical_response_m,
                "future physical continuation is below threshold",
            )
            _require(
                correction_ratio <= cfg.maximum_correction_to_physical_motion_ratio,
                "correction exceeds physical-motion ratio",
            )
            _require(
                correction_cosine >= cfg.minimum_correction_physical_cosine,
                "correction opposes physical continuation",
            )

            states = proposed
            for frame in range(update + 1, stop):
                field = decode_recursive_rbf_belief(
                    states[selected_name],
                    selected[update],
                    forecast_frames=frame - update,
                    config=cfg.belief,
                )
                candidate[frame] = _corrected_frame(
                    selected[frame], field.mean_m, dtype=output_dtype
                )
            record.update(
                {
                    "accepted": True,
                    "decision": "dynamic-pairwise-belief-update",
                    "bit_exact_selected_backbone_fallback": False,
                }
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            record["fallback_reason"] = f"{type(error).__name__}: {error}"

        if not record["accepted"] and not np.array_equal(
            candidate[update + 1 : stop], selected_trajectory[update + 1 : stop]
        ):
            raise AssertionError("dynamic rejection changed the selected backbone")
        update_reports.append(record)
        previous_update = update

    report = {
        "protocol_id": PROTOCOL_ID,
        "arm": DYNAMIC_PAIRWISE_ARM,
        "config": asdict(cfg),
        "observation_pool_ids": pool.tolist(),
        "update_frames": list(cfg.update_frames),
        "accepted_update_count": int(
            sum(bool(record["accepted"]) for record in update_reports)
        ),
        "updates": update_reports,
        "information_boundary": {
            "target_argument_accepted": False,
            "future_observation_read": False,
            "all_observation_pool_identities_must_be_excluded_from_scoring": True,
            "prior_reliability_uses_state_innovation": False,
            "association_geometry_precedes_state_innovation": True,
            "state_innovation_likelihood_count": 1,
            "rejected_belief_state_is_unchanged": True,
            "rejected_trajectory_is_bit_exact_selected_backbone": True,
        },
    }
    return report, {
        SELECTED_BACKBONE_ARM: selected_trajectory,
        DYNAMIC_PAIRWISE_ARM: candidate,
    }


__all__ = [
    "DYNAMIC_PAIRWISE_ARM",
    "PROTOCOL_ID",
    "SELECTED_BACKBONE_ARM",
    "DynamicCenterSelection",
    "DynamicPairwiseBeliefConfig",
    "predict_dynamic_pairwise_belief_arrays",
    "select_dynamic_centers",
    "update_metric_recursive_rbf_belief",
]
