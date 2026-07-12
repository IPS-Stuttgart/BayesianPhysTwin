"""Hierarchical low-rank MAP fitting for structural PhysTwin calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .structural_artifact import (
    STRUCTURAL_RANK_CANDIDATES,
    StructuralSessionCorrection,
    StructuralTwinCorrection,
    nominal_rest_geometry_sha256,
)


STRUCTURAL_VARIANTS = (
    "baseline",
    "frame_only",
    "initial_state_only",
    "rest_geometry_only",
    "rest_state",
    "hierarchical",
)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _readonly(values: np.ndarray, *, dtype: Any = float) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StructuralLinearizedSession:
    """O-minus evidence and simulator sensitivities for one acquisition session."""

    session_id: str
    observations_m: np.ndarray
    nominal_prediction_m: np.ndarray
    observation_weights: np.ndarray
    persistent_response: np.ndarray
    settled_state_response: np.ndarray
    gravity_response: np.ndarray
    fit_frame_mask: np.ndarray
    validation_frame_mask: np.ndarray
    frame_origin_m: np.ndarray
    frame_response: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be nonempty")
        observations = _readonly(self.observations_m)
        nominal = _readonly(self.nominal_prediction_m)
        weights = _readonly(self.observation_weights)
        persistent = _readonly(self.persistent_response)
        settled = _readonly(self.settled_state_response)
        gravity = _readonly(self.gravity_response)
        fit_mask = _readonly(self.fit_frame_mask, dtype=bool)
        validation_mask = _readonly(self.validation_frame_mask, dtype=bool)
        origin = _readonly(self.frame_origin_m)
        frame_response = (
            None
            if self.frame_response is None
            else _readonly(self.frame_response)
        )
        if observations.ndim != 3 or observations.shape[2] != 3:
            raise ValueError("observations_m must have shape (T, N, 3)")
        if nominal.shape != observations.shape:
            raise ValueError("nominal prediction must match observations")
        frame_count, node_count, _ = observations.shape
        if weights.shape not in {(frame_count, node_count), observations.shape}:
            raise ValueError("observation_weights must have shape (T, N) or (T, N, 3)")
        rank = persistent.shape[-1] if persistent.ndim == 4 else -1
        expected_response = (frame_count, node_count, 3, rank)
        if persistent.shape != expected_response or settled.shape != expected_response:
            raise ValueError("persistent and settled responses must have shape (T, N, 3, R)")
        if gravity.shape != (frame_count, node_count, 3, 3):
            raise ValueError("gravity_response must have shape (T, N, 3, 3)")
        if fit_mask.shape != (frame_count,) or validation_mask.shape != (frame_count,):
            raise ValueError("fit and validation masks must match O-minus frames")
        if not np.any(fit_mask) or not np.any(validation_mask):
            raise ValueError("fit and validation masks must both be nonempty")
        if np.any(fit_mask & validation_mask):
            raise ValueError("fit and validation frames must be disjoint")
        if origin.shape != (3,):
            raise ValueError("frame_origin_m must be a 3-vector")
        if frame_response is not None and frame_response.shape != (
            frame_count,
            node_count,
            3,
            6,
        ):
            raise ValueError("frame_response must have shape (T, N, 3, 6)")
        arrays = (observations, nominal, weights, persistent, settled, gravity, origin)
        if frame_response is not None:
            arrays = (*arrays, frame_response)
        if any(not np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("linearized session arrays must be finite")
        if np.any(weights < 0.0):
            raise ValueError("observation weights must be nonnegative")
        try:
            metadata = json.loads(
                json.dumps(dict(self.metadata), sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("session metadata must be finite JSON data") from error
        object.__setattr__(self, "observations_m", observations)
        object.__setattr__(self, "nominal_prediction_m", nominal)
        object.__setattr__(self, "observation_weights", weights)
        object.__setattr__(self, "persistent_response", persistent)
        object.__setattr__(self, "settled_state_response", settled)
        object.__setattr__(self, "gravity_response", gravity)
        object.__setattr__(self, "fit_frame_mask", fit_mask)
        object.__setattr__(self, "validation_frame_mask", validation_mask)
        object.__setattr__(self, "frame_origin_m", origin)
        object.__setattr__(self, "frame_response", frame_response)
        object.__setattr__(self, "metadata", metadata)

    @property
    def rank(self) -> int:
        return self.persistent_response.shape[3]

    @property
    def source_checksum(self) -> str:
        """Hash only O-minus inputs admitted by the fitting contract."""

        digest = hashlib.sha256()
        for name, values in (
            ("observations_m", self.observations_m),
            ("nominal_prediction_m", self.nominal_prediction_m),
            ("observation_weights", self.observation_weights),
            ("persistent_response", self.persistent_response),
            ("settled_state_response", self.settled_state_response),
            ("gravity_response", self.gravity_response),
            ("fit_frame_mask", self.fit_frame_mask),
            ("validation_frame_mask", self.validation_frame_mask),
            ("frame_origin_m", self.frame_origin_m),
        ):
            digest.update(name.encode("ascii"))
            digest.update(_array_sha256(values).encode("ascii"))
        if self.frame_response is not None:
            digest.update(b"frame_response")
            digest.update(_array_sha256(self.frame_response).encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class StructuralMAPConfig:
    """One preregisterable structural mechanism and regularization setting."""

    variant: str = "hierarchical"
    rank: int = 8
    persistent_prior_strength: float = 1e-2
    settled_state_prior_strength: float = 5e-2
    frame_rotation_prior_strength: float = 1e-2
    frame_translation_prior_strength: float = 1e-2
    gravity_prior_strength: float = 5e-2
    edge_strain_prior_strength: float = 1e-2
    graph_frequency_power: float = 1.0
    huber_delta_m: float = 0.005
    robust_iterations: int = 3
    allowed_edge_strain: float = 0.10

    def __post_init__(self) -> None:
        if self.variant not in STRUCTURAL_VARIANTS:
            raise ValueError(f"variant must lie in {STRUCTURAL_VARIANTS}")
        if self.rank not in STRUCTURAL_RANK_CANDIDATES:
            raise ValueError(f"rank must lie in {STRUCTURAL_RANK_CANDIDATES}")
        strengths = (
            self.persistent_prior_strength,
            self.settled_state_prior_strength,
            self.frame_rotation_prior_strength,
            self.frame_translation_prior_strength,
            self.gravity_prior_strength,
            self.edge_strain_prior_strength,
        )
        if any(value < 0.0 or not np.isfinite(value) for value in strengths):
            raise ValueError("MAP prior strengths must be finite and nonnegative")
        if self.graph_frequency_power < 0.0 or self.huber_delta_m <= 0.0:
            raise ValueError("frequency power and Huber scale must be valid")
        if self.robust_iterations < 1 or not 0.0 < self.allowed_edge_strain < 1.0:
            raise ValueError("robust iterations and allowed strain must be valid")

    @property
    def includes_rest(self) -> bool:
        return self.variant in {"rest_geometry_only", "rest_state", "hierarchical"}

    @property
    def includes_frame(self) -> bool:
        return self.variant in {"frame_only", "hierarchical"}

    @property
    def includes_state(self) -> bool:
        return self.variant in {"initial_state_only", "rest_state", "hierarchical"}

    @property
    def includes_gravity(self) -> bool:
        return self.variant == "hierarchical"


@dataclass(frozen=True)
class _ParameterLayout:
    count: int
    persistent: slice | None
    gravity: slice | None
    frame_rotation: Mapping[str, slice]
    frame_translation: Mapping[str, slice]
    settled: Mapping[str, slice]


def _parameter_layout(
    session_ids: Sequence[str], rank: int, config: StructuralMAPConfig
) -> _ParameterLayout:
    cursor = 0

    def allocate(size: int) -> slice:
        nonlocal cursor
        result = slice(cursor, cursor + size)
        cursor += size
        return result

    persistent = allocate(rank) if config.includes_rest else None
    gravity = allocate(3) if config.includes_gravity else None
    rotations = {
        session_id: allocate(3) for session_id in session_ids if config.includes_frame
    }
    translations = {
        session_id: allocate(3) for session_id in session_ids if config.includes_frame
    }
    settled = {
        session_id: allocate(rank) for session_id in session_ids if config.includes_state
    }
    return _ParameterLayout(
        count=cursor,
        persistent=persistent,
        gravity=gravity,
        frame_rotation=rotations,
        frame_translation=translations,
        settled=settled,
    )


def _frame_design(points: np.ndarray, origin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    rotation = np.empty((*centered.shape, 3), dtype=float)
    for axis_index, axis in enumerate(np.eye(3)):
        rotation[..., axis_index] = np.cross(axis, centered)
    translation = np.broadcast_to(np.eye(3), rotation.shape)
    return rotation, translation


def _frame_matrix(
    session: StructuralLinearizedSession,
    frame: int,
    layout: _ParameterLayout,
    config: StructuralMAPConfig,
) -> np.ndarray:
    node_count = session.observations_m.shape[1]
    matrix = np.zeros((node_count * 3, layout.count), dtype=float)
    if layout.persistent is not None:
        matrix[:, layout.persistent] = session.persistent_response[frame].reshape(
            node_count * 3, session.rank
        )
    if layout.gravity is not None:
        matrix[:, layout.gravity] = session.gravity_response[frame].reshape(
            node_count * 3, 3
        )
    if config.includes_frame:
        if session.frame_response is None:
            rotation, translation = _frame_design(
                session.nominal_prediction_m[frame], session.frame_origin_m
            )
            frame_response = np.concatenate((rotation, translation), axis=2)
        else:
            frame_response = session.frame_response[frame]
        matrix[:, layout.frame_rotation[session.session_id]] = frame_response[
            ..., :3
        ].reshape(node_count * 3, 3)
        matrix[:, layout.frame_translation[session.session_id]] = frame_response[
            ..., 3:
        ].reshape(node_count * 3, 3)
    if layout.settled:
        matrix[:, layout.settled[session.session_id]] = session.settled_state_response[
            frame
        ].reshape(node_count * 3, session.rank)
    return matrix


def relative_edge_strain_jacobian(
    nominal_rest_positions: np.ndarray,
    springs: np.ndarray,
    nominal_rest_lengths: np.ndarray,
    graph_basis: np.ndarray,
    *,
    num_object_springs: int,
) -> np.ndarray:
    """Linearized relative rest-length change per persistent coefficient."""

    positions = np.asarray(nominal_rest_positions, dtype=float)
    edges = np.asarray(springs, dtype=np.int64)[:num_object_springs]
    lengths = np.asarray(nominal_rest_lengths, dtype=float)[:num_object_springs]
    basis = np.asarray(graph_basis, dtype=float)
    if basis.shape[:2] != positions.shape or len(edges) != len(lengths):
        raise ValueError("edge-strain inputs disagree")
    edge_vector = positions[edges[:, 0]] - positions[edges[:, 1]]
    direction = edge_vector / np.maximum(np.linalg.norm(edge_vector, axis=1)[:, None], 1e-12)
    basis_difference = basis[edges[:, 0]] - basis[edges[:, 1]]
    return np.einsum("si,sir->sr", direction, basis_difference) / lengths[:, None]


def _prior_precision(
    layout: _ParameterLayout,
    graph_frequencies: np.ndarray,
    config: StructuralMAPConfig,
) -> np.ndarray:
    precision = np.zeros(layout.count, dtype=float)
    frequency = np.maximum(
        np.asarray(graph_frequencies, dtype=float)
        / float(np.min(graph_frequencies)),
        1.0,
    ) ** config.graph_frequency_power
    if layout.persistent is not None:
        precision[layout.persistent] = config.persistent_prior_strength * frequency
    if layout.gravity is not None:
        precision[layout.gravity] = config.gravity_prior_strength
    for value in layout.frame_rotation.values():
        precision[value] = config.frame_rotation_prior_strength
    for value in layout.frame_translation.values():
        precision[value] = config.frame_translation_prior_strength
    for value in layout.settled.values():
        precision[value] = config.settled_state_prior_strength * frequency
    return precision


def _measurement_weights(session: StructuralLinearizedSession, frame: int) -> np.ndarray:
    weights = session.observation_weights[frame]
    if weights.ndim == 1:
        weights = np.repeat(weights[:, None], 3, axis=1)
    return weights.reshape(-1)


def _solve_map(
    sessions: Sequence[StructuralLinearizedSession],
    graph_frequencies: np.ndarray,
    edge_jacobian: np.ndarray,
    layout: _ParameterLayout,
    config: StructuralMAPConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if layout.count == 0:
        return np.empty(0), {"normal_condition_number": 1.0, "irls_iterations": 0}
    parameters = np.zeros(layout.count, dtype=float)
    precision = _prior_precision(layout, graph_frequencies, config)
    normal_condition = 1.0
    for iteration in range(config.robust_iterations):
        normal = np.diag(precision + 1e-12)
        right = np.zeros(layout.count, dtype=float)
        for session in sessions:
            for frame in np.flatnonzero(session.fit_frame_mask):
                design = _frame_matrix(session, int(frame), layout, config)
                target = (
                    session.observations_m[frame] - session.nominal_prediction_m[frame]
                ).reshape(-1)
                base_weight = _measurement_weights(session, int(frame))
                residual = target - design @ parameters
                robust = np.minimum(
                    1.0,
                    config.huber_delta_m / np.maximum(np.abs(residual), 1e-12),
                )
                weight = base_weight * robust
                normal += design.T @ (weight[:, None] * design)
                right += design.T @ (weight * target)
        if layout.persistent is not None and config.edge_strain_prior_strength > 0.0:
            strain = edge_jacobian @ parameters[layout.persistent]
            robust = np.minimum(
                1.0,
                config.allowed_edge_strain / np.maximum(np.abs(strain), 1e-12),
            )
            weighted = config.edge_strain_prior_strength * robust
            block = layout.persistent
            normal[block, block] += edge_jacobian.T @ (
                weighted[:, None] * edge_jacobian
            )
        normal_condition = float(np.linalg.cond(normal))
        try:
            parameters = np.linalg.solve(normal, right)
        except np.linalg.LinAlgError:
            parameters = np.linalg.lstsq(normal, right, rcond=None)[0]
    return parameters, {
        "normal_condition_number": normal_condition,
        "irls_iterations": config.robust_iterations,
    }


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(values))
    if angle <= 1e-15:
        return np.eye(3)
    axis = values / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    column_rotation = (
        np.eye(3)
        + np.sin(angle) * skew
        + (1.0 - np.cos(angle)) * (skew @ skew)
    )
    return column_rotation.T


def _exact_maximum_edge_strain(
    positions: np.ndarray,
    edges: np.ndarray,
    lengths: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    corrected = positions + np.tensordot(basis, coefficients, axes=(2, 0))
    geometric = np.linalg.norm(
        corrected[edges[:, 0]] - corrected[edges[:, 1]], axis=1
    )
    return float(np.max(np.abs(geometric / lengths - 1.0), initial=0.0))


def _constrain_edge_strain(
    positions: np.ndarray,
    edges: np.ndarray,
    lengths: np.ndarray,
    basis: np.ndarray,
    coefficients: np.ndarray,
    allowed: float,
) -> tuple[np.ndarray, float, float]:
    unconstrained = _exact_maximum_edge_strain(
        positions, edges, lengths, basis, coefficients
    )
    if unconstrained <= allowed:
        return coefficients, 1.0, unconstrained
    lower, upper = 0.0, 1.0
    for _ in range(50):
        middle = 0.5 * (lower + upper)
        value = _exact_maximum_edge_strain(
            positions, edges, lengths, basis, middle * coefficients
        )
        if value <= allowed:
            lower = middle
        else:
            upper = middle
    constrained = lower * coefficients
    return constrained, lower, _exact_maximum_edge_strain(
        positions, edges, lengths, basis, constrained
    )


def _predict_session(
    session: StructuralLinearizedSession,
    parameters: np.ndarray,
    layout: _ParameterLayout,
    config: StructuralMAPConfig,
) -> np.ndarray:
    result = session.nominal_prediction_m.copy()
    for frame in range(len(result)):
        result[frame] += (
            _frame_matrix(session, frame, layout, config) @ parameters
        ).reshape(result.shape[1:])
    return result


@dataclass(frozen=True)
class StructuralMAPResult:
    correction: StructuralTwinCorrection
    config: StructuralMAPConfig
    predictions_m: Mapping[str, np.ndarray]
    diagnostics: Mapping[str, Any]


def fit_hierarchical_structural_map(
    sessions: Sequence[StructuralLinearizedSession],
    nominal_rest_positions: np.ndarray,
    springs: np.ndarray,
    nominal_rest_lengths: np.ndarray,
    *,
    num_object_springs: int,
    graph_basis: np.ndarray,
    graph_frequencies: np.ndarray,
    support_node_indices: Sequence[int] = (),
    surface_triangles: np.ndarray | None = None,
    validity_tetrahedra: np.ndarray | None = None,
    support_model: Mapping[str, Any] | None = None,
    config: StructuralMAPConfig | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> StructuralMAPResult:
    """Fit one hierarchical MAP artifact using O-minus evidence only."""

    settings = config or StructuralMAPConfig()
    evidence = tuple(sessions)
    if not evidence or len({value.session_id for value in evidence}) != len(evidence):
        raise ValueError("sessions must be nonempty and uniquely identified")
    if any(value.rank != settings.rank for value in evidence):
        raise ValueError("session response rank differs from the MAP configuration")
    basis = np.asarray(graph_basis, dtype=float)
    frequencies = np.asarray(graph_frequencies, dtype=float)
    if basis.shape != (len(nominal_rest_positions), 3, settings.rank):
        raise ValueError("graph basis differs from the MAP configuration")
    if frequencies.shape != (settings.rank,):
        raise ValueError("graph frequencies differ from the MAP configuration")
    object_edges = np.asarray(springs, dtype=np.int64)[:num_object_springs]
    object_lengths = np.asarray(nominal_rest_lengths, dtype=float)[
        :num_object_springs
    ]
    edge_jacobian = relative_edge_strain_jacobian(
        nominal_rest_positions,
        springs,
        nominal_rest_lengths,
        graph_basis,
        num_object_springs=num_object_springs,
    )
    layout = _parameter_layout(
        [value.session_id for value in evidence], settings.rank, settings
    )
    parameters, solver_diagnostics = _solve_map(
        evidence, frequencies, edge_jacobian, layout, settings
    )
    persistent = (
        np.zeros(settings.rank)
        if layout.persistent is None
        else parameters[layout.persistent].copy()
    )
    persistent, strain_scale, maximum_strain = _constrain_edge_strain(
        np.asarray(nominal_rest_positions, dtype=float),
        object_edges,
        object_lengths,
        basis,
        persistent,
        settings.allowed_edge_strain,
    )
    if layout.persistent is not None:
        parameters[layout.persistent] = persistent
    session_corrections = []
    predictions = {}
    session_metrics = {}
    for session in evidence:
        rotation_vector = (
            np.zeros(3)
            if session.session_id not in layout.frame_rotation
            else parameters[layout.frame_rotation[session.session_id]]
        )
        translation = (
            np.zeros(3)
            if session.session_id not in layout.frame_translation
            else parameters[layout.frame_translation[session.session_id]]
        )
        settled = (
            np.zeros(settings.rank)
            if session.session_id not in layout.settled
            else parameters[layout.settled[session.session_id]]
        )
        gravity = (
            np.zeros(3)
            if layout.gravity is None
            else parameters[layout.gravity]
        )
        session_corrections.append(
            StructuralSessionCorrection(
                session_id=session.session_id,
                frame_linear=_rotation_from_vector(rotation_vector),
                frame_translation_m=translation,
                settled_state_coefficients=settled,
                gravity_correction_mps2=gravity,
            )
        )
        prediction = _predict_session(session, parameters, layout, settings)
        predictions[session.session_id] = prediction
        residual = np.linalg.norm(prediction - session.observations_m, axis=2)
        session_metrics[session.session_id] = {
            "fit_rmse_m": float(
                np.sqrt(np.mean(np.square(residual[session.fit_frame_mask])))
            ),
            "validation_rmse_m": float(
                np.sqrt(np.mean(np.square(residual[session.validation_frame_mask])))
            ),
        }
    source_checksums = {
        f"o_minus_session:{session.session_id}": session.source_checksum
        for session in evidence
    }
    source_checksums["nominal_rest_geometry"] = nominal_rest_geometry_sha256(
        nominal_rest_positions,
        springs,
        nominal_rest_lengths,
        num_object_springs=num_object_springs,
    )
    correction = StructuralTwinCorrection(
        nominal_rest_geometry_hash=source_checksums["nominal_rest_geometry"],
        graph_basis=basis,
        graph_frequencies=frequencies,
        persistent_rest_coefficients=persistent,
        persistent_coefficient_covariance=None,
        sessions=tuple(session_corrections),
        support_node_indices=np.asarray(tuple(support_node_indices), dtype=np.int64),
        surface_triangles=np.empty((0, 3), dtype=np.int64)
        if surface_triangles is None
        else np.asarray(surface_triangles, dtype=np.int64),
        validity_tetrahedra=np.empty((0, 4), dtype=np.int64)
        if validity_tetrahedra is None
        else np.asarray(validity_tetrahedra, dtype=np.int64),
        support_model=dict(support_model or {"kind": "declared_node_anchors"}),
        allowed_edge_strain=settings.allowed_edge_strain,
        fit_session_ids=tuple(value.session_id for value in evidence),
        information_boundary={
            "persistent_fit_uses_o_minus_only": True,
            "future_frames_used_for_fit": False,
            "target_outcomes_used_for_selection": False,
            "fit_mode": "map",
            "rank_selection_domain": "source_pre_action_only",
        },
        source_checksums=source_checksums,
        metadata={
            "variant": settings.variant,
            "rank": settings.rank,
            "posterior_stage": "deferred_until_mean_transfer",
            **dict(metadata or {}),
        },
    )
    validation_values = [
        value["validation_rmse_m"] for value in session_metrics.values()
    ]
    diagnostics = {
        **solver_diagnostics,
        "variant": settings.variant,
        "rank": settings.rank,
        "parameter_count": layout.count,
        "persistent_coefficient_norm": float(np.linalg.norm(persistent)),
        "persistent_edge_strain_scale": strain_scale,
        "maximum_absolute_edge_strain": maximum_strain,
        "mean_validation_rmse_m": float(np.mean(validation_values)),
        "session_metrics": session_metrics,
        "posterior_uncertainty_estimated": False,
        "information_boundary": correction.information_boundary,
    }
    return StructuralMAPResult(
        correction=correction,
        config=settings,
        predictions_m=predictions,
        diagnostics=diagnostics,
    )


def select_structural_map_result(
    results: Sequence[StructuralMAPResult],
) -> StructuralMAPResult:
    """Select only by source O-minus validation error, then lower complexity."""

    candidates = tuple(results)
    if not candidates:
        raise ValueError("at least one structural MAP result is required")
    candidates = tuple(
        sorted(
            candidates,
            key=lambda value: (
                value.diagnostics["mean_validation_rmse_m"],
                value.diagnostics["parameter_count"],
                value.config.rank,
                STRUCTURAL_VARIANTS.index(value.config.variant),
            ),
        )
    )
    return candidates[0]
