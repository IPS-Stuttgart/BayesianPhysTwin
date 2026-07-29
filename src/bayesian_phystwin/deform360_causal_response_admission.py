"""Causal, cross-panel admission for sparse direct-depth state updates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .deform360_direct_depth_provider import DirectDepthEndpointObservations
from .observation_belief import array_sha256

CONTRACT = "deform360-causal-response-admission-v12"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    array.setflags(write=False)
    return array


def _canonical_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-admission-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CausalResponseAdmissionConfig:
    """Frozen target-free evidence requirements for one response event."""

    spatial_group_count: int = 3
    minimum_supported_count: int = 6
    minimum_supported_per_group: int = 1
    minimum_action_support: float = 0.10
    minimum_tactile_contact_probability: float = 0.50
    minimum_actuator_displacement_m: float = 0.001
    minimum_physical_centered_rms_m: float = 0.0005
    minimum_observed_centered_rms_m: float = 0.001
    minimum_pairwise_residual_rms_m: float = 0.0005
    minimum_physical_observed_cosine: float = 0.20
    minimum_cross_panel_pairwise_cosine: float = 0.60
    minimum_cross_panel_vector_cosine: float = 0.40
    minimum_validation_improvement_fraction: float = 0.10
    maximum_cross_panel_reduced_nis: float = 9.0
    maximum_effective_count: float = 3.0
    shared_bias_variance_m2: float = 25e-6
    variance_floor_m2: float = 1e-10

    def __post_init__(self) -> None:
        _require(self.spatial_group_count >= 2, "too few spatial groups")
        _require(
            self.minimum_supported_count
            >= self.spatial_group_count * self.minimum_supported_per_group,
            "global support is weaker than grouped support",
        )
        probabilities = (
            self.minimum_action_support,
            self.minimum_tactile_contact_probability,
        )
        _require(
            all(np.isfinite(value) and 0.0 < value <= 1.0 for value in probabilities),
            "support probabilities must lie in (0, 1]",
        )
        cosines = (
            self.minimum_physical_observed_cosine,
            self.minimum_cross_panel_pairwise_cosine,
            self.minimum_cross_panel_vector_cosine,
        )
        _require(
            all(np.isfinite(value) and -1.0 <= value <= 1.0 for value in cosines),
            "cosine thresholds must lie in [-1, 1]",
        )
        positive = (
            self.minimum_actuator_displacement_m,
            self.minimum_physical_centered_rms_m,
            self.minimum_observed_centered_rms_m,
            self.minimum_pairwise_residual_rms_m,
            self.maximum_cross_panel_reduced_nis,
            self.maximum_effective_count,
            self.shared_bias_variance_m2,
            self.variance_floor_m2,
        )
        _require(
            all(np.isfinite(value) and value > 0.0 for value in positive),
            "response scales must be positive",
        )
        _require(
            np.isfinite(self.minimum_validation_improvement_fraction)
            and 0.0 <= self.minimum_validation_improvement_fraction <= 1.0,
            "validation improvement must lie in [0, 1]",
        )


@dataclass(frozen=True)
class CausalResponseMetrics:
    """Prefix-only diagnostics entering the admission decision."""

    supported_count: int
    supported_group_count: int
    effective_count: float
    physical_centered_rms_m: float
    proposal_observed_centered_rms_m: float
    validation_observed_centered_rms_m: float
    proposal_pairwise_residual_rms_m: float
    validation_pairwise_residual_rms_m: float
    physical_observed_cosine: float
    cross_panel_pairwise_cosine: float
    cross_panel_vector_cosine: float
    validation_improvement_fraction: float
    cross_panel_reduced_nis: float

    def __post_init__(self) -> None:
        _require(self.supported_count >= 0, "supported count is negative")
        _require(
            0 <= self.supported_group_count <= self.supported_count,
            "supported group count is invalid",
        )
        numeric = (
            self.effective_count,
            self.physical_centered_rms_m,
            self.proposal_observed_centered_rms_m,
            self.validation_observed_centered_rms_m,
            self.proposal_pairwise_residual_rms_m,
            self.validation_pairwise_residual_rms_m,
            self.physical_observed_cosine,
            self.cross_panel_pairwise_cosine,
            self.cross_panel_vector_cosine,
            self.validation_improvement_fraction,
            self.cross_panel_reduced_nis,
        )
        _require(
            all(np.isfinite(value) for value in numeric),
            "causal-response metrics are not finite",
        )
        _require(self.effective_count >= 0.0, "effective count is negative")
        for value in (
            self.physical_observed_cosine,
            self.cross_panel_pairwise_cosine,
            self.cross_panel_vector_cosine,
        ):
            _require(-1.0 <= value <= 1.0, "reported cosine is invalid")
        _require(
            self.cross_panel_reduced_nis >= 0.0,
            "cross-panel NIS is negative",
        )


@dataclass(frozen=True)
class CausalResponseAdmission:
    """Immutable response-event decision with complete prefix provenance."""

    case_id: str
    birth_frame: int
    update_frame: int
    proposal_camera_ids: tuple[str, ...]
    validation_camera_ids: tuple[str, ...]
    admitted: bool
    reason: str
    tactile_contact_probability: float
    actuator_displacement_m: float
    config: CausalResponseAdmissionConfig
    metrics: CausalResponseMetrics
    selected_entity_ids: np.ndarray
    spatial_group_assignments: np.ndarray
    physical_prefix_sha256: str
    action_conditioning_prefix_sha256: str
    proposal_observation_sha256: str
    validation_observation_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        entities = _readonly(self.selected_entity_ids, dtype=np.int64)
        assignments = _readonly(self.spatial_group_assignments, dtype=np.int64)
        _require(bool(self.case_id.strip()), "case ID is empty")
        _require(
            0 <= self.birth_frame < self.update_frame,
            "response endpoint order is invalid",
        )
        _require(bool(self.reason.strip()), "admission reason is empty")
        _require(
            len(self.proposal_camera_ids) >= 2
            and len(self.validation_camera_ids) >= 2
            and len(set(self.proposal_camera_ids)) == len(self.proposal_camera_ids)
            and len(set(self.validation_camera_ids)) == len(self.validation_camera_ids)
            and not set(self.proposal_camera_ids).intersection(
                self.validation_camera_ids
            ),
            "camera panels must be disjoint and independently supported",
        )
        _require(
            entities.ndim == assignments.ndim == 1
            and len(entities) == len(assignments)
            and len(np.unique(entities)) == len(entities),
            "selected entities or spatial groups are invalid",
        )
        if len(assignments):
            _require(
                np.all(
                    (assignments >= 0) & (assignments < self.config.spatial_group_count)
                ),
                "spatial group assignment is out of range",
            )
        _require(
            np.isfinite(self.tactile_contact_probability)
            and 0.0 <= self.tactile_contact_probability <= 1.0
            and np.isfinite(self.actuator_displacement_m)
            and self.actuator_displacement_m >= 0.0,
            "causal support values are invalid",
        )
        for digest in (
            self.physical_prefix_sha256,
            self.action_conditioning_prefix_sha256,
            self.proposal_observation_sha256,
            self.validation_observation_sha256,
            self.artifact_sha256,
        ):
            _require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                "admission digest is invalid",
            )
        object.__setattr__(self, "selected_entity_ids", entities)
        object.__setattr__(self, "spatial_group_assignments", assignments)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalResponseAdmission",
            "contract": CONTRACT,
            "case_id": self.case_id,
            "birth_frame": self.birth_frame,
            "update_frame": self.update_frame,
            "proposal_camera_ids": list(self.proposal_camera_ids),
            "validation_camera_ids": list(self.validation_camera_ids),
            "admitted": self.admitted,
            "reason": self.reason,
            "tactile_contact_probability": self.tactile_contact_probability,
            "actuator_displacement_m": self.actuator_displacement_m,
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "selected_entity_ids": self.selected_entity_ids.tolist(),
            "spatial_group_assignments": self.spatial_group_assignments.tolist(),
            "physical_prefix_sha256": self.physical_prefix_sha256,
            "action_conditioning_prefix_sha256": (
                self.action_conditioning_prefix_sha256
            ),
            "proposal_observation_sha256": self.proposal_observation_sha256,
            "validation_observation_sha256": self.validation_observation_sha256,
            "information_boundary": {
                "maximum_observation_frame": self.update_frame,
                "future_observation_read": False,
                "future_identity_read": False,
                "future_metric_read": False,
                "candidate_future_read": False,
                "proposal_and_validation_camera_panels_disjoint": True,
                "global_translation_treated_as_nuisance": True,
                "global_similarity_scale_treated_as_nuisance": True,
                "pairwise_distance_evidence_is_rigid_invariant": True,
                "state_innovation_changes_prior_reliability": False,
                "camera_rows_counted_as_independent": False,
                "exact_baseline_fallback_required": True,
                "final_source_calibrated_regret_guard_required": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def direct_depth_observation_sha256(
    observations: DirectDepthEndpointObservations,
) -> str:
    digests = (
        array_sha256(observations.endpoint_frames),
        array_sha256(observations.entity_ids),
        array_sha256(
            np.nan_to_num(
                observations.point_world_m,
                nan=np.finfo(np.float64).max,
            )
        ),
        array_sha256(
            np.nan_to_num(
                observations.covariance_m2,
                nan=np.finfo(np.float64).max,
            )
        ),
        array_sha256(observations.accepted_support),
        array_sha256(observations.association_probability),
        array_sha256(observations.support_count),
        array_sha256(observations.maximum_view_scatter_m),
        hashlib.sha256(
            json.dumps(
                observations.config.evidence_descriptor(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    )
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


def _balanced_spatial_groups(
    positions_m: np.ndarray,
    entity_ids: np.ndarray,
    group_count: int,
) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    entities = np.asarray(entity_ids, dtype=np.int64)
    _require(
        positions.shape == (len(entities), 3)
        and len(entities) >= group_count
        and np.all(np.isfinite(positions)),
        "spatial positions are invalid",
    )
    centroid = np.mean(positions, axis=0)
    radial = np.linalg.norm(positions - centroid, axis=1)
    first_candidates = np.flatnonzero(
        np.isclose(radial, np.max(radial), rtol=0.0, atol=1e-15)
    )
    first = int(first_candidates[np.argmin(entities[first_candidates])])
    order = [first]
    remaining = set(range(len(entities)))
    remaining.remove(first)
    while remaining:
        candidates = np.asarray(sorted(remaining), dtype=np.int64)
        nearest = np.min(
            np.linalg.norm(
                positions[candidates, None] - positions[np.asarray(order)][None],
                axis=2,
            ),
            axis=1,
        )
        best = float(np.max(nearest))
        tied = candidates[np.isclose(nearest, best, rtol=0.0, atol=1e-15)]
        selected = int(tied[np.argmin(entities[tied])])
        order.append(selected)
        remaining.remove(selected)
    assignments = np.empty(len(entities), dtype=np.int64)
    for rank, point_index in enumerate(order):
        assignments[point_index] = rank % group_count
    return assignments


def _normalized_weights(
    proposal: DirectDepthEndpointObservations,
    validation: DirectDepthEndpointObservations,
    selected_local: np.ndarray,
    action_support: np.ndarray,
) -> tuple[np.ndarray, float]:
    proposal_covariance = np.sum(
        proposal.covariance_m2[:, selected_local],
        axis=0,
    )
    validation_covariance = np.sum(
        validation.covariance_m2[:, selected_local],
        axis=0,
    )
    uncertainty = np.trace(
        proposal_covariance + validation_covariance,
        axis1=1,
        axis2=2,
    )
    association = np.sqrt(
        np.prod(
            proposal.association_probability[:, selected_local],
            axis=0,
        )
        * np.prod(
            validation.association_probability[:, selected_local],
            axis=0,
        )
    )
    raw = action_support * association / np.maximum(uncertainty, 1e-12)
    _require(
        np.all(np.isfinite(raw)) and np.any(raw > 0.0),
        "response evidence has no positive residual-independent weight",
    )
    weights = raw / np.sum(raw)
    effective = float(np.square(np.sum(raw)) / np.sum(np.square(raw)))
    return weights, effective


def _weighted_center(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return values - np.sum(weights[:, None] * values, axis=0)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.sum(np.square(values), axis=1))))


def _weighted_cosine(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    dot = float(np.sum(weights * np.sum(first * second, axis=1)))
    first_energy = float(np.sum(weights * np.sum(np.square(first), axis=1)))
    second_energy = float(np.sum(weights * np.sum(np.square(second), axis=1)))
    denominator = np.sqrt(first_energy * second_energy)
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(dot / denominator, -1.0, 1.0))


def _pairwise_residual(
    endpoint_points_m: np.ndarray,
    physical_endpoints_m: np.ndarray,
) -> np.ndarray:
    pair_i, pair_j = np.triu_indices(endpoint_points_m.shape[1], k=1)
    observed_distance = np.linalg.norm(
        endpoint_points_m[:, pair_i] - endpoint_points_m[:, pair_j],
        axis=2,
    )
    physical_distance = np.linalg.norm(
        physical_endpoints_m[:, pair_i] - physical_endpoints_m[:, pair_j],
        axis=2,
    )
    return (
        observed_distance[1]
        - observed_distance[0]
        - physical_distance[1]
        + physical_distance[0]
    )


def _pairwise_weights(weights: np.ndarray) -> np.ndarray:
    pair_i, pair_j = np.triu_indices(len(weights), k=1)
    pair_weights = np.sqrt(weights[pair_i] * weights[pair_j])
    return pair_weights / np.sum(pair_weights)


def _scalar_cosine(
    first: np.ndarray,
    second: np.ndarray,
    weights: np.ndarray,
) -> float:
    dot = float(np.sum(weights * first * second))
    denominator = np.sqrt(
        float(np.sum(weights * np.square(first)))
        * float(np.sum(weights * np.square(second)))
    )
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(dot / denominator, -1.0, 1.0))


def _remove_weighted_modes(
    values: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    modes = np.asarray(basis, dtype=np.float64)
    _require(
        modes.ndim == 2
        and modes.shape[0] == len(values)
        and np.all(np.isfinite(modes)),
        "similarity nuisance basis is invalid",
    )
    if modes.shape[1] == 0:
        return values.copy()
    square_root_weight = np.sqrt(weights)
    weighted_modes = square_root_weight[:, None] * modes
    weighted_values = square_root_weight * values
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_modes,
        weighted_values,
        rcond=1e-12,
    )
    return values - modes @ coefficients


def _reduced_cross_panel_nis(
    proposal_centered_residual: np.ndarray,
    validation_centered_residual: np.ndarray,
    proposal: DirectDepthEndpointObservations,
    validation: DirectDepthEndpointObservations,
    selected_local: np.ndarray,
    weights: np.ndarray,
    config: CausalResponseAdmissionConfig,
) -> float:
    difference = proposal_centered_residual - validation_centered_residual
    covariance = (
        np.sum(proposal.covariance_m2[:, selected_local], axis=0)
        + np.sum(validation.covariance_m2[:, selected_local], axis=0)
        + 2.0 * config.shared_bias_variance_m2 * np.eye(3)[None]
    )
    nis = np.empty(len(difference), dtype=np.float64)
    for index, (vector, matrix) in enumerate(zip(difference, covariance, strict=True)):
        nis[index] = float(vector @ np.linalg.solve(matrix, vector)) / 3.0
    return float(np.sum(weights * nis))


def _empty_metrics(
    supported_count: int = 0,
    supported_group_count: int = 0,
) -> CausalResponseMetrics:
    return CausalResponseMetrics(
        supported_count=supported_count,
        supported_group_count=supported_group_count,
        effective_count=0.0,
        physical_centered_rms_m=0.0,
        proposal_observed_centered_rms_m=0.0,
        validation_observed_centered_rms_m=0.0,
        proposal_pairwise_residual_rms_m=0.0,
        validation_pairwise_residual_rms_m=0.0,
        physical_observed_cosine=0.0,
        cross_panel_pairwise_cosine=0.0,
        cross_panel_vector_cosine=0.0,
        validation_improvement_fraction=0.0,
        cross_panel_reduced_nis=0.0,
    )


def evaluate_causal_response_admission(
    case_id: str,
    physical_positions_m: np.ndarray,
    proposal: DirectDepthEndpointObservations,
    validation: DirectDepthEndpointObservations,
    action_support: np.ndarray,
    *,
    proposal_camera_ids: tuple[str, ...],
    validation_camera_ids: tuple[str, ...],
    tactile_contact_probability: float,
    actuator_displacement_m: float,
    action_conditioning_positions_m: np.ndarray | None = None,
    config: CausalResponseAdmissionConfig | None = None,
) -> CausalResponseAdmission:
    """Certify one prefix response without constructing or scoring a future."""

    cfg = config or CausalResponseAdmissionConfig()
    physical = np.asarray(physical_positions_m, dtype=np.float64)
    action_conditioning = (
        physical
        if action_conditioning_positions_m is None
        else np.asarray(action_conditioning_positions_m, dtype=np.float64)
    )
    support = np.asarray(action_support, dtype=np.float64)
    _require(
        physical.ndim == 3 and physical.shape[2] == 3 and np.all(np.isfinite(physical)),
        "physical prefix is invalid",
    )
    _require(
        action_conditioning.shape == physical.shape
        and np.all(np.isfinite(action_conditioning))
        and np.array_equal(action_conditioning[0], physical[0]),
        "action-conditioning trajectory differs in shape or frame-zero identity",
    )
    _require(
        support.shape == (physical.shape[1],)
        and np.all(np.isfinite(support))
        and np.all((support >= 0.0) & (support <= 1.0)),
        "action support is invalid",
    )
    _require(
        np.array_equal(proposal.endpoint_frames, validation.endpoint_frames)
        and np.array_equal(proposal.entity_ids, validation.entity_ids),
        "proposal and validation observations do not share an endpoint contract",
    )
    frames = np.asarray(proposal.endpoint_frames, dtype=np.int64)
    birth, update = map(int, frames)
    entities = np.asarray(proposal.entity_ids, dtype=np.int64)
    _require(
        update < len(physical)
        and np.all((entities >= 0) & (entities < physical.shape[1])),
        "response endpoints or entities lie outside the physical prefix",
    )
    _require(
        len(proposal_camera_ids) >= 2
        and len(validation_camera_ids) >= 2
        and len(set(proposal_camera_ids)) == len(proposal_camera_ids)
        and len(set(validation_camera_ids)) == len(validation_camera_ids)
        and not set(proposal_camera_ids).intersection(validation_camera_ids),
        "proposal and validation camera panels must be disjoint",
    )
    common_support = (
        np.all(proposal.accepted_support, axis=0)
        & np.all(validation.accepted_support, axis=0)
        & (support[entities] >= cfg.minimum_action_support)
    )
    selected_local = np.flatnonzero(common_support)
    selected_entities = entities[selected_local]
    assignments = np.empty(0, dtype=np.int64)
    if len(selected_entities) >= cfg.spatial_group_count:
        assignments = _balanced_spatial_groups(
            physical[birth, selected_entities],
            selected_entities,
            cfg.spatial_group_count,
        )
    supported_groups = (
        0
        if not len(assignments)
        else sum(
            int(np.sum(assignments == group)) >= cfg.minimum_supported_per_group
            for group in range(cfg.spatial_group_count)
        )
    )
    metrics = _empty_metrics(len(selected_entities), supported_groups)
    if len(selected_entities) >= cfg.minimum_supported_count:
        weights, raw_effective = _normalized_weights(
            proposal,
            validation,
            selected_local,
            support[selected_entities],
        )
        baseline_endpoints = physical[frames][:, selected_entities]
        baseline_displacement = baseline_endpoints[1] - baseline_endpoints[0]
        action_endpoints = action_conditioning[frames][:, selected_entities]
        physical_displacement = action_endpoints[1] - action_endpoints[0]
        proposal_displacement = (
            proposal.point_world_m[1, selected_local]
            - proposal.point_world_m[0, selected_local]
        )
        validation_displacement = (
            validation.point_world_m[1, selected_local]
            - validation.point_world_m[0, selected_local]
        )
        physical_centered = _weighted_center(physical_displacement, weights)
        proposal_observed_centered = _weighted_center(
            proposal_displacement,
            weights,
        )
        validation_observed_centered = _weighted_center(
            validation_displacement,
            weights,
        )
        proposal_residual = _weighted_center(
            proposal_displacement - baseline_displacement,
            weights,
        )
        validation_residual = _weighted_center(
            validation_displacement - baseline_displacement,
            weights,
        )
        proposal_pairwise = _pairwise_residual(
            proposal.point_world_m[:, selected_local],
            baseline_endpoints,
        )
        validation_pairwise = _pairwise_residual(
            validation.point_world_m[:, selected_local],
            baseline_endpoints,
        )
        pair_weights = _pairwise_weights(weights)
        pair_i, pair_j = np.triu_indices(len(selected_entities), k=1)
        physical_birth_distance = np.linalg.norm(
            baseline_endpoints[0, pair_i] - baseline_endpoints[0, pair_j],
            axis=1,
        )
        physical_update_distance = np.linalg.norm(
            baseline_endpoints[1, pair_i] - baseline_endpoints[1, pair_j],
            axis=1,
        )
        similarity_modes = np.column_stack(
            (physical_birth_distance, physical_update_distance)
        )
        proposal_pairwise = _remove_weighted_modes(
            proposal_pairwise,
            similarity_modes,
            pair_weights,
        )
        validation_pairwise = _remove_weighted_modes(
            validation_pairwise,
            similarity_modes,
            pair_weights,
        )
        proposal_pairwise_rms = float(
            np.sqrt(np.sum(pair_weights * np.square(proposal_pairwise)))
        )
        validation_pairwise_rms = float(
            np.sqrt(np.sum(pair_weights * np.square(validation_pairwise)))
        )
        validation_baseline_loss = float(
            np.sum(pair_weights * np.square(validation_pairwise))
        )
        validation_candidate_loss = float(
            np.sum(pair_weights * np.square(validation_pairwise - proposal_pairwise))
        )
        validation_improvement = (
            0.0
            if validation_baseline_loss <= cfg.variance_floor_m2
            else 1.0 - validation_candidate_loss / validation_baseline_loss
        )
        physical_observed_cosine = min(
            _weighted_cosine(
                physical_centered,
                proposal_observed_centered,
                weights,
            ),
            _weighted_cosine(
                physical_centered,
                validation_observed_centered,
                weights,
            ),
        )
        metrics = CausalResponseMetrics(
            supported_count=len(selected_entities),
            supported_group_count=supported_groups,
            effective_count=min(raw_effective, cfg.maximum_effective_count),
            physical_centered_rms_m=_weighted_rms(
                physical_centered,
                weights,
            ),
            proposal_observed_centered_rms_m=_weighted_rms(
                proposal_observed_centered,
                weights,
            ),
            validation_observed_centered_rms_m=_weighted_rms(
                validation_observed_centered,
                weights,
            ),
            proposal_pairwise_residual_rms_m=proposal_pairwise_rms,
            validation_pairwise_residual_rms_m=validation_pairwise_rms,
            physical_observed_cosine=physical_observed_cosine,
            cross_panel_pairwise_cosine=_scalar_cosine(
                proposal_pairwise,
                validation_pairwise,
                pair_weights,
            ),
            cross_panel_vector_cosine=_weighted_cosine(
                proposal_residual,
                validation_residual,
                weights,
            ),
            validation_improvement_fraction=validation_improvement,
            cross_panel_reduced_nis=_reduced_cross_panel_nis(
                proposal_residual,
                validation_residual,
                proposal,
                validation,
                selected_local,
                weights,
                cfg,
            ),
        )

    if tactile_contact_probability < cfg.minimum_tactile_contact_probability:
        admitted = False
        reason = "insufficient-tactile-contact"
    elif actuator_displacement_m < cfg.minimum_actuator_displacement_m:
        admitted = False
        reason = "insufficient-actuator-displacement"
    elif metrics.supported_count < cfg.minimum_supported_count:
        admitted = False
        reason = "insufficient-cross-panel-support"
    elif metrics.supported_group_count < cfg.spatial_group_count:
        admitted = False
        reason = "insufficient-spatial-coverage"
    elif metrics.physical_centered_rms_m < cfg.minimum_physical_centered_rms_m:
        admitted = False
        reason = "insufficient-physical-response"
    elif (
        min(
            metrics.proposal_observed_centered_rms_m,
            metrics.validation_observed_centered_rms_m,
        )
        < cfg.minimum_observed_centered_rms_m
    ):
        admitted = False
        reason = "insufficient-observed-response"
    elif (
        min(
            metrics.proposal_pairwise_residual_rms_m,
            metrics.validation_pairwise_residual_rms_m,
        )
        < cfg.minimum_pairwise_residual_rms_m
    ):
        admitted = False
        reason = "insufficient-nonrigid-update-headroom"
    elif metrics.physical_observed_cosine < cfg.minimum_physical_observed_cosine:
        admitted = False
        reason = "response-not-action-aligned"
    elif metrics.cross_panel_pairwise_cosine < cfg.minimum_cross_panel_pairwise_cosine:
        admitted = False
        reason = "cross-panel-pairwise-disagreement"
    elif metrics.cross_panel_vector_cosine < cfg.minimum_cross_panel_vector_cosine:
        admitted = False
        reason = "cross-panel-vector-disagreement"
    elif metrics.cross_panel_reduced_nis > cfg.maximum_cross_panel_reduced_nis:
        admitted = False
        reason = "cross-panel-uncertainty-inconsistent"
    elif (
        metrics.validation_improvement_fraction
        < cfg.minimum_validation_improvement_fraction
    ):
        admitted = False
        reason = "no-heldout-prefix-improvement"
    else:
        admitted = True
        reason = "causal-cross-panel-response"

    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": CONTRACT,
        "case_id": str(case_id),
        "birth_frame": birth,
        "update_frame": update,
        "proposal_camera_ids": list(proposal_camera_ids),
        "validation_camera_ids": list(validation_camera_ids),
        "admitted": admitted,
        "reason": reason,
        "tactile_contact_probability": float(tactile_contact_probability),
        "actuator_displacement_m": float(actuator_displacement_m),
        "config": asdict(cfg),
        "metrics": asdict(metrics),
        "selected_entity_ids": selected_entities.tolist(),
        "spatial_group_assignments": assignments.tolist(),
        "physical_prefix_sha256": array_sha256(physical[: update + 1]),
        "action_conditioning_prefix_sha256": array_sha256(
            action_conditioning[: update + 1]
        ),
        "proposal_observation_sha256": direct_depth_observation_sha256(proposal),
        "validation_observation_sha256": direct_depth_observation_sha256(validation),
    }
    digest = _canonical_sha256(payload)
    return CausalResponseAdmission(
        case_id=str(case_id),
        birth_frame=birth,
        update_frame=update,
        proposal_camera_ids=proposal_camera_ids,
        validation_camera_ids=validation_camera_ids,
        admitted=admitted,
        reason=reason,
        tactile_contact_probability=float(tactile_contact_probability),
        actuator_displacement_m=float(actuator_displacement_m),
        config=cfg,
        metrics=metrics,
        selected_entity_ids=selected_entities,
        spatial_group_assignments=assignments,
        physical_prefix_sha256=payload["physical_prefix_sha256"],
        action_conditioning_prefix_sha256=payload["action_conditioning_prefix_sha256"],
        proposal_observation_sha256=payload["proposal_observation_sha256"],
        validation_observation_sha256=payload["validation_observation_sha256"],
        artifact_sha256=digest,
    )


__all__ = [
    "CONTRACT",
    "CausalResponseAdmission",
    "CausalResponseAdmissionConfig",
    "CausalResponseMetrics",
    "direct_depth_observation_sha256",
    "evaluate_causal_response_admission",
]
