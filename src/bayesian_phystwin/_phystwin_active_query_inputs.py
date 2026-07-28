"""Validated inputs for nuisance-aware physics-guided query planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phystwin_query_plan_v1 import (
    PhysicsGuidedQueryConfigV2,
    readonly,
    require,
)


@dataclass(frozen=True)
class _ScoredNode:
    node_id: int
    motion_score: float
    visibility_score: float
    mode_information_gain: float
    spatial_diversity_score: float
    contact_distance_score: float
    total_score: float


@dataclass(frozen=True)
class _PlannerInputs:
    rollout: np.ndarray
    pixels: np.ndarray
    predicted_support: np.ndarray
    observation_precision: np.ndarray
    mode_basis: np.ndarray
    nuisance_basis: np.ndarray
    contact_positions: np.ndarray | None
    candidate_ids: np.ndarray
    config: PhysicsGuidedQueryConfigV2
    source_revision: str
    support_model_id: str


def _normalized_basis(
    basis_values: np.ndarray | None,
    node_count: int,
    *,
    name: str,
) -> np.ndarray:
    if basis_values is None:
        return readonly(np.empty((node_count, 0), dtype=np.float64))
    basis = np.asarray(basis_values, dtype=np.float64)
    require(
        basis.ndim == 2 and basis.shape[0] == node_count,
        f"{name} must have shape (N, R)",
    )
    require(basis.shape[1] >= 1, f"{name} must retain at least one mode")
    require(np.all(np.isfinite(basis)), f"{name} contains non-finite values")
    norm = np.linalg.norm(basis, axis=0)
    require(np.all(norm > 0.0), f"{name} contains an empty mode")
    return readonly(basis / norm[None], dtype=np.float64)


def _validate_inputs(
    physical_rollout_m: np.ndarray,
    projected_pixels_xy: np.ndarray,
    predicted_support_probability: np.ndarray,
    *,
    mode_basis: np.ndarray | None,
    nuisance_basis: np.ndarray | None,
    observation_precision: np.ndarray | None,
    contact_position_m: np.ndarray | None,
    candidate_ids: np.ndarray | None,
    config: PhysicsGuidedQueryConfigV2 | None,
    source_revision: str,
    support_model_id: str,
) -> _PlannerInputs:
    cfg = config or PhysicsGuidedQueryConfigV2()
    if not isinstance(cfg, PhysicsGuidedQueryConfigV2):
        raise TypeError("config must be a PhysicsGuidedQueryConfigV2")
    require(bool(source_revision), "source_revision must be nonempty")
    require(bool(support_model_id), "support_model_id must be nonempty")
    rollout = readonly(physical_rollout_m, dtype=np.float64)
    pixels = readonly(projected_pixels_xy, dtype=np.float64)
    predicted_support = readonly(
        predicted_support_probability,
        dtype=np.float64,
    )
    require(
        rollout.ndim == 3 and rollout.shape[2] == 3,
        "physical_rollout_m must have shape (T, N, 3)",
    )
    frame_count, node_count, _ = rollout.shape
    require(frame_count >= 1 and node_count >= 1, "physical rollout is empty")
    require(
        pixels.ndim == 4
        and pixels.shape[1:3] == (frame_count, node_count)
        and pixels.shape[3] == 2,
        "projected_pixels_xy must have shape (C, T, N, 2)",
    )
    camera_count = pixels.shape[0]
    require(
        camera_count >= cfg.minimum_camera_support,
        "camera count is below minimum_camera_support",
    )
    require(
        predicted_support.shape == (camera_count, frame_count, node_count),
        "predicted_support_probability shape changed",
    )
    require(
        np.all(np.isfinite(predicted_support))
        and np.all((predicted_support >= 0.0) & (predicted_support <= 1.0)),
        "predicted support probabilities must lie in [0, 1]",
    )
    if observation_precision is None:
        precision = np.ones_like(predicted_support)
    else:
        precision = np.asarray(observation_precision, dtype=np.float64)
        require(precision.shape == predicted_support.shape, "precision shape changed")
        require(
            np.all(np.isfinite(precision)) and np.all(precision > 0.0),
            "observation_precision must be finite and positive",
        )
    normalized_mode = _normalized_basis(mode_basis, node_count, name="mode_basis")
    normalized_nuisance = _normalized_basis(
        nuisance_basis,
        node_count,
        name="nuisance_basis",
    )
    if candidate_ids is None:
        candidates = np.arange(node_count, dtype=np.int64)
    else:
        candidates = np.asarray(candidate_ids, dtype=np.int64)
        require(
            candidates.ndim == 1 and len(candidates) >= 1,
            "candidate_ids is empty",
        )
        require(
            np.all((candidates >= 0) & (candidates < node_count)),
            "candidate ID exceeds the physical rollout",
        )
        require(
            len(np.unique(candidates)) == len(candidates),
            "candidate_ids must be unique",
        )
        candidates = np.sort(candidates, kind="mergesort")
    if contact_position_m is None:
        contact_positions = None
    else:
        contact = np.asarray(contact_position_m, dtype=np.float64)
        if contact.shape == (3,):
            contact = np.repeat(contact[None], frame_count, axis=0)
        require(
            contact.shape == (frame_count, 3),
            "contact_position_m must have shape (3,) or (T, 3)",
        )
        require(np.all(np.isfinite(contact)), "contact position is not finite")
        contact_positions = readonly(contact, dtype=np.float64)
    return _PlannerInputs(
        rollout=rollout,
        pixels=pixels,
        predicted_support=predicted_support,
        observation_precision=readonly(precision, dtype=np.float64),
        mode_basis=normalized_mode,
        nuisance_basis=normalized_nuisance,
        contact_positions=contact_positions,
        candidate_ids=readonly(candidates, dtype=np.int64),
        config=cfg,
        source_revision=source_revision,
        support_model_id=support_model_id,
    )


def _remaining_motion_m(rollout: np.ndarray, frame: int) -> np.ndarray:
    start = rollout[frame]
    delta = rollout[frame:] - start[None]
    finite = np.all(np.isfinite(delta), axis=2)
    distance = np.linalg.norm(np.where(finite[..., None], delta, 0.0), axis=2)
    distance = np.where(finite, distance, -np.inf)
    motion = np.max(distance, axis=0)
    motion[~np.any(finite, axis=0)] = np.nan
    return motion


def _object_scale_m(positions: np.ndarray, candidate_ids: np.ndarray) -> float:
    selected = positions[candidate_ids]
    finite = selected[np.all(np.isfinite(selected), axis=1)]
    if len(finite) < 2:
        return 1.0
    diagonal = float(np.linalg.norm(np.max(finite, axis=0) - np.min(finite, axis=0)))
    return max(diagonal, 1e-6)


def _query_view_mask(inputs: _PlannerInputs, frame: int, node_id: int) -> np.ndarray:
    return (
        inputs.predicted_support[:, frame, node_id]
        >= inputs.config.support_probability_threshold
    ) & np.all(np.isfinite(inputs.pixels[:, frame, node_id]), axis=1)


def _expected_query_precision(
    inputs: _PlannerInputs,
    frame: int,
    node_id: int,
) -> float:
    view_mask = _query_view_mask(inputs, frame, node_id)
    if int(np.sum(view_mask)) < inputs.config.minimum_camera_support:
        return 0.0
    values = (
        inputs.observation_precision[view_mask, frame, node_id]
        * inputs.predicted_support[view_mask, frame, node_id]
    )
    return float(np.mean(values))


__all__ = [
    "_PlannerInputs",
    "_ScoredNode",
    "_expected_query_precision",
    "_object_scale_m",
    "_query_view_mask",
    "_remaining_motion_m",
    "_validate_inputs",
]
