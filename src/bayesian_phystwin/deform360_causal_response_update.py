"""Bias-aware sparse belief update for an admitted causal response event."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform360_causal_response_admission import (
    CausalResponseAdmission,
    direct_depth_observation_sha256,
)
from .deform360_direct_depth_provider import DirectDepthEndpointObservations
from .deform360_dynamic_tapnextpp_assimilation import (
    BirthAnchoredMeasurements,
    _update_covariance_aware_rbf_belief,
)
from .observation_belief import array_sha256
from .phystwin_online_belief import (
    RecursiveRbfBeliefConfig,
    decode_recursive_rbf_belief,
    initialize_recursive_rbf_belief,
)

CANDIDATE_ARM = "causal_response_bias_aware_rbf"
BASELINE_ARM = "selected_baseline"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class CausalResponseMeasurementConfig:
    """Frozen nuisance, reliability, and covariance settings."""

    maximum_translation_m: float = 0.05
    maximum_rotation_degrees: float = 10.0
    maximum_scale_deviation: float = 0.10
    support_saturation_count: float = 4.0
    scatter_scale_m: float = 0.01
    temporal_covariance_multiplier: float = 2.0
    shared_bias_variance_m2: float = 25e-6
    covariance_floor_m2: float = 1e-10

    def __post_init__(self) -> None:
        positive = (
            self.maximum_translation_m,
            self.maximum_rotation_degrees,
            self.maximum_scale_deviation,
            self.support_saturation_count,
            self.scatter_scale_m,
            self.temporal_covariance_multiplier,
            self.shared_bias_variance_m2,
            self.covariance_floor_m2,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "causal-response measurement settings must be positive",
        )
        _require(
            self.maximum_rotation_degrees < 180.0
            and self.maximum_scale_deviation < 1.0,
            "similarity nuisance limits are invalid",
        )


@dataclass(frozen=True)
class SimilarityNuisanceEstimate:
    """One endpoint's weighted observed-to-physical Sim(3) fit."""

    scale: float
    rotation: np.ndarray
    translation_m: np.ndarray
    rotation_degrees: float
    translation_norm_m: float
    fit_rms_m: float
    within_limits: bool

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64).copy()
        translation = np.asarray(self.translation_m, dtype=np.float64).copy()
        _require(
            rotation.shape == (3, 3)
            and translation.shape == (3,)
            and np.all(np.isfinite(rotation))
            and np.all(np.isfinite(translation)),
            "similarity nuisance arrays are invalid",
        )
        numeric = (
            self.scale,
            self.rotation_degrees,
            self.translation_norm_m,
            self.fit_rms_m,
        )
        _require(
            all(np.isfinite(value) for value in numeric)
            and self.scale > 0.0
            and self.rotation_degrees >= 0.0
            and self.translation_norm_m >= 0.0
            and self.fit_rms_m >= 0.0,
            "similarity nuisance diagnostics are invalid",
        )
        _require(
            np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
            and np.linalg.det(rotation) > 0.0,
            "similarity nuisance rotation is not proper orthogonal",
        )
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", translation)

    def descriptor(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation_m": self.translation_m.tolist(),
            "rotation_degrees": self.rotation_degrees,
            "translation_norm_m": self.translation_norm_m,
            "fit_rms_m": self.fit_rms_m,
            "within_limits": self.within_limits,
        }


@dataclass(frozen=True)
class CausalResponseMeasurements:
    """Metric sparse measurements plus fitted nuisance provenance."""

    measurements: BirthAnchoredMeasurements
    endpoint_nuisance: tuple[
        SimilarityNuisanceEstimate,
        SimilarityNuisanceEstimate,
    ]
    accepted: bool
    reason: str
    config: CausalResponseMeasurementConfig
    physical_prefix_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.reason.strip()), "measurement decision is empty")
        _require(
            len(self.endpoint_nuisance) == 2,
            "two endpoint nuisance estimates are required",
        )
        _require(
            len(self.physical_prefix_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in self.physical_prefix_sha256
            ),
            "physical prefix digest is invalid",
        )


def _selected_local_indices(
    observations: DirectDepthEndpointObservations,
    selected_entity_ids: np.ndarray,
) -> np.ndarray:
    lookup = {
        int(entity): index for index, entity in enumerate(observations.entity_ids)
    }
    _require(
        all(int(entity) in lookup for entity in selected_entity_ids),
        "admitted entity is absent from proposal observations",
    )
    return np.asarray(
        [lookup[int(entity)] for entity in selected_entity_ids],
        dtype=np.int64,
    )


def _residual_independent_weights(
    observations: DirectDepthEndpointObservations,
    selected_local: np.ndarray,
    config: CausalResponseMeasurementConfig,
) -> np.ndarray:
    support = np.min(
        observations.support_count[:, selected_local],
        axis=0,
    )
    redundancy = np.minimum(
        support / config.support_saturation_count,
        1.0,
    )
    scatter = np.max(
        observations.maximum_view_scatter_m[:, selected_local],
        axis=0,
    )
    scatter_reliability = np.exp(-scatter / config.scatter_scale_m)
    association = np.sqrt(
        np.prod(
            observations.association_probability[:, selected_local],
            axis=0,
        )
    )
    raw = redundancy * scatter_reliability * association
    _require(
        np.all(np.isfinite(raw)) and np.any(raw > 0.0),
        "proposal has no residual-independent nuisance-fit support",
    )
    return raw / np.sum(raw)


def _fit_similarity(
    observed_m: np.ndarray,
    physical_m: np.ndarray,
    weights: np.ndarray,
    config: CausalResponseMeasurementConfig,
) -> tuple[np.ndarray, SimilarityNuisanceEstimate]:
    observed = np.asarray(observed_m, dtype=np.float64)
    physical = np.asarray(physical_m, dtype=np.float64)
    _require(
        observed.shape == physical.shape == (len(weights), 3)
        and len(weights) >= 3
        and np.all(np.isfinite(observed))
        and np.all(np.isfinite(physical)),
        "similarity fit points are invalid",
    )
    observed_mean = np.sum(weights[:, None] * observed, axis=0)
    physical_mean = np.sum(weights[:, None] * physical, axis=0)
    observed_centered = observed - observed_mean
    physical_centered = physical - physical_mean
    cross_covariance = np.einsum(
        "n,ni,nj->ij",
        weights,
        physical_centered,
        observed_centered,
        optimize=True,
    )
    left, singular_values, right_transpose = np.linalg.svd(cross_covariance)
    sign = np.ones(3)
    sign[-1] = np.sign(np.linalg.det(left @ right_transpose))
    rotation = left @ np.diag(sign) @ right_transpose
    observed_variance = float(
        np.sum(weights * np.sum(np.square(observed_centered), axis=1))
    )
    _require(
        observed_variance > config.covariance_floor_m2,
        "similarity fit is geometrically degenerate",
    )
    scale = float(np.sum(singular_values * sign) / observed_variance)
    _require(scale > 0.0, "similarity fit produced a nonpositive scale")
    translation = physical_mean - scale * (rotation @ observed_mean)
    aligned = scale * (rotation @ observed.T).T + translation
    fit_rms = float(
        np.sqrt(np.sum(weights * np.sum(np.square(aligned - physical), axis=1)))
    )
    trace_value = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_degrees = float(np.degrees(np.arccos(trace_value)))
    translation_norm = float(np.linalg.norm(translation))
    within_limits = bool(
        abs(scale - 1.0) <= config.maximum_scale_deviation
        and rotation_degrees <= config.maximum_rotation_degrees
        and translation_norm <= config.maximum_translation_m
    )
    return aligned, SimilarityNuisanceEstimate(
        scale=scale,
        rotation=rotation,
        translation_m=translation,
        rotation_degrees=rotation_degrees,
        translation_norm_m=translation_norm,
        fit_rms_m=fit_rms,
        within_limits=within_limits,
    )


def build_causal_response_measurements(
    physical_positions_m: np.ndarray,
    proposal: DirectDepthEndpointObservations,
    admission: CausalResponseAdmission,
    *,
    config: CausalResponseMeasurementConfig | None = None,
) -> CausalResponseMeasurements:
    """Remove endpoint Sim(3) nuisance and form metric current-state evidence."""

    cfg = config or CausalResponseMeasurementConfig()
    physical = np.asarray(physical_positions_m, dtype=np.float64)
    selected_entities = np.asarray(
        admission.selected_entity_ids,
        dtype=np.int64,
    )
    entities = (
        selected_entities
        if len(selected_entities)
        else np.asarray(proposal.entity_ids, dtype=np.int64)
    )
    birth, update = admission.birth_frame, admission.update_frame
    _require(
        direct_depth_observation_sha256(proposal)
        == admission.proposal_observation_sha256,
        "proposal observations differ from the admitted artifact",
    )
    _require(
        np.array_equal(proposal.endpoint_frames, [birth, update])
        and update < len(physical)
        and np.all((entities >= 0) & (entities < physical.shape[1])),
        "measurement endpoints or entities changed after admission",
    )
    physical_digest = array_sha256(physical[: update + 1])
    _require(
        physical_digest == admission.physical_prefix_sha256,
        "physical prefix differs from the admitted artifact",
    )
    full_measurement = np.full(physical.shape, np.nan, dtype=np.float64)
    full_covariance = np.full((*physical.shape[:2], 3, 3), np.nan)
    prior_reliability = np.zeros(physical.shape[:2], dtype=np.float64)
    association_probability = np.zeros(physical.shape[:2], dtype=np.float64)
    available = np.zeros(physical.shape[:2], dtype=bool)
    if not admission.admitted:
        empty = BirthAnchoredMeasurements(
            measurement_m=full_measurement,
            covariance_m2=full_covariance,
            prior_reliability=prior_reliability,
            association_probability=association_probability,
            available=available,
            entity_ids=entities,
        )
        identity = SimilarityNuisanceEstimate(
            scale=1.0,
            rotation=np.eye(3),
            translation_m=np.zeros(3),
            rotation_degrees=0.0,
            translation_norm_m=0.0,
            fit_rms_m=0.0,
            within_limits=True,
        )
        return CausalResponseMeasurements(
            measurements=empty,
            endpoint_nuisance=(identity, identity),
            accepted=False,
            reason="admission-rejected-exact-baseline-fallback",
            config=cfg,
            physical_prefix_sha256=physical_digest,
        )

    selected_local = _selected_local_indices(proposal, selected_entities)
    _require(
        np.all(proposal.accepted_support[:, selected_local]),
        "admitted proposal support changed",
    )
    weights = _residual_independent_weights(proposal, selected_local, cfg)
    aligned_points: list[np.ndarray] = []
    nuisance: list[SimilarityNuisanceEstimate] = []
    for endpoint_index, frame in enumerate((birth, update)):
        aligned, estimate = _fit_similarity(
            proposal.point_world_m[endpoint_index, selected_local],
            physical[frame, selected_entities],
            weights,
            cfg,
        )
        aligned_points.append(aligned)
        nuisance.append(estimate)
    nuisance_pair = (nuisance[0], nuisance[1])
    if not all(estimate.within_limits for estimate in nuisance_pair):
        empty = BirthAnchoredMeasurements(
            measurement_m=full_measurement,
            covariance_m2=full_covariance,
            prior_reliability=prior_reliability,
            association_probability=association_probability,
            available=available,
            entity_ids=entities,
        )
        return CausalResponseMeasurements(
            measurements=empty,
            endpoint_nuisance=nuisance_pair,
            accepted=False,
            reason="similarity-nuisance-outside-limits-exact-baseline-fallback",
            config=cfg,
            physical_prefix_sha256=physical_digest,
        )

    observed_displacement = aligned_points[1] - aligned_points[0]
    full_measurement[update, selected_entities] = (
        physical[birth, selected_entities] + observed_displacement
    )
    base_covariance = cfg.temporal_covariance_multiplier * np.sum(
        proposal.covariance_m2[:, selected_local],
        axis=0,
    )
    nuisance_variance = (
        sum(estimate.fit_rms_m**2 for estimate in nuisance_pair)
        + cfg.shared_bias_variance_m2
    )
    full_covariance[update, selected_entities] = (
        base_covariance
        + (nuisance_variance + cfg.covariance_floor_m2) * np.eye(3)[None]
    )
    support_count = np.min(
        proposal.support_count[:, selected_local],
        axis=0,
    )
    redundancy = np.minimum(
        support_count / cfg.support_saturation_count,
        1.0,
    )
    scatter = np.max(
        proposal.maximum_view_scatter_m[:, selected_local],
        axis=0,
    )
    prior_reliability[update, selected_entities] = redundancy * np.exp(
        -scatter / cfg.scatter_scale_m
    )
    association_probability[update, selected_entities] = np.sqrt(
        np.prod(
            proposal.association_probability[:, selected_local],
            axis=0,
        )
    )
    available[update, selected_entities] = True
    measurements = BirthAnchoredMeasurements(
        measurement_m=full_measurement,
        covariance_m2=full_covariance,
        prior_reliability=prior_reliability,
        association_probability=association_probability,
        available=available,
        entity_ids=selected_entities,
    )
    return CausalResponseMeasurements(
        measurements=measurements,
        endpoint_nuisance=nuisance_pair,
        accepted=True,
        reason="sim3-debiased-metric-prefix-measurements",
        config=cfg,
        physical_prefix_sha256=physical_digest,
    )


def predict_causal_response_candidate(
    baseline_prediction_m: np.ndarray,
    response: CausalResponseMeasurements,
    admission: CausalResponseAdmission,
    *,
    belief_config: RecursiveRbfBeliefConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Apply one admitted robust RBF update or return the exact baseline."""

    baseline_input = np.asarray(baseline_prediction_m)
    baseline = np.asarray(baseline_input, dtype=np.float64)
    _require(
        baseline.ndim == 3 and baseline.shape[2] == 3 and np.all(np.isfinite(baseline)),
        "selected baseline prediction is invalid",
    )
    _require(
        response.measurements.measurement_m.shape == baseline.shape,
        "response measurement shape differs from the selected baseline",
    )
    update = admission.update_frame
    _require(update + 1 < len(baseline), "response has no forecast continuation")
    _require(
        array_sha256(baseline[: update + 1]) == response.physical_prefix_sha256,
        "candidate baseline prefix differs from the measured physical prefix",
    )
    candidate = baseline_input.copy()
    variance = np.zeros_like(baseline, dtype=np.float64)
    cfg = belief_config or RecursiveRbfBeliefConfig(
        length_scale_fraction=0.10,
        local_blend=1.0,
    )
    accepted = bool(admission.admitted and response.accepted)
    posterior_reliability = np.zeros(
        len(response.measurements.entity_ids),
        dtype=np.float64,
    )
    if accepted:
        centers = response.measurements.entity_ids
        state = initialize_recursive_rbf_belief(
            centers,
            baseline[update, centers],
            baseline[update],
            config=cfg,
        )
        measurement = response.measurements
        residual = (
            measurement.measurement_m[update, centers] - baseline[update, centers]
        )
        state, posterior_reliability = _update_covariance_aware_rbf_belief(
            state,
            update,
            baseline[update, centers],
            residual,
            measurement.available[update, centers],
            measurement.covariance_m2[update, centers],
            measurement.prior_reliability[update, centers],
            config=cfg,
            association_probability=measurement.association_probability[
                update, centers
            ],
        )
        for frame in range(update + 1, len(baseline)):
            decoded = decode_recursive_rbf_belief(
                state,
                baseline[update],
                forecast_frames=frame - update,
                config=cfg,
            )
            candidate[frame] = (baseline[frame] + decoded.mean_m).astype(
                baseline_input.dtype, copy=False
            )
            variance[frame] = decoded.variance_m2

    exact_fallback = bool(accepted or np.array_equal(candidate, baseline_input))
    _require(exact_fallback, "rejected causal response changed the baseline")
    report = {
        "schema_version": 1,
        "artifact_kind": "Deform360CausalResponseCandidate",
        "admission_artifact_sha256": admission.artifact_sha256,
        "admission_accepted": admission.admitted,
        "measurement_accepted": response.accepted,
        "candidate_applied": accepted,
        "decision": ("causal-response-rbf-update" if accepted else response.reason),
        "update_frame": update,
        "center_ids": response.measurements.entity_ids.tolist(),
        "measurement_config": asdict(response.config),
        "belief_config": asdict(cfg),
        "endpoint_similarity_nuisance": [
            estimate.descriptor() for estimate in response.endpoint_nuisance
        ],
        "mean_prior_reliability": (
            float(
                np.mean(
                    response.measurements.prior_reliability[
                        update,
                        response.measurements.entity_ids,
                    ]
                )
            )
            if accepted
            else 0.0
        ),
        "mean_posterior_inlier_probability": (
            float(np.mean(posterior_reliability)) if accepted else 0.0
        ),
        "bit_exact_baseline_fallback": bool(not accepted and exact_fallback),
        "method_contract": {
            "global_similarity_nuisance_removed_per_endpoint": True,
            "metric_covariance_m2_propagated": True,
            "assignment_probability_separate_from_prior_reliability": True,
            "state_innovation_changes_prior_reliability": False,
            "innovation_robustified_once": True,
            "future_observation_read": False,
            "rejection": "bit-exact selected baseline",
        },
    }
    return report, {
        BASELINE_ARM: baseline_input.copy(),
        CANDIDATE_ARM: candidate,
        "candidate_correction_variance_m2": variance,
    }


__all__ = [
    "BASELINE_ARM",
    "CANDIDATE_ARM",
    "CausalResponseMeasurementConfig",
    "CausalResponseMeasurements",
    "SimilarityNuisanceEstimate",
    "build_causal_response_measurements",
    "predict_causal_response_candidate",
]
