"""MolmoMotion reweighting through the sparse physical readout H_Q."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from causal4d.contracts import PhysicalPosterior, TaskPosterior, array_sha256

if TYPE_CHECKING:
    from causal4d.molmo_adapter import MolmoForecastBundle


@dataclass(frozen=True)
class SparseSemanticEvidence:
    """Sparse desired trajectory aligned to physical posterior frames."""

    positions_m: np.ndarray
    node_indices: np.ndarray
    physical_frame_indices: np.ndarray
    scale_m: float = 0.10
    degrees_of_freedom: float = 3.0
    compare_displacements: bool = True
    anchor_positions_m: np.ndarray | None = None
    anchor_physical_frame: int = 0
    valid: np.ndarray | None = None
    source: str = "semantic_trajectory"

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=float)
        nodes = np.asarray(self.node_indices, dtype=np.int64)
        frames = np.asarray(self.physical_frame_indices, dtype=float)
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError("positions_m must have shape (F, Q, 3)")
        if nodes.shape != (positions.shape[1],):
            raise ValueError("node_indices must identify every semantic point")
        if frames.shape != (positions.shape[0],):
            raise ValueError("physical_frame_indices must identify every forecast frame")
        if np.any(nodes < 0) or not np.all(np.isfinite(frames)):
            raise ValueError("semantic node and frame indices must be valid")
        if self.scale_m <= 0.0 or self.degrees_of_freedom <= 0.0:
            raise ValueError("semantic scale and degrees of freedom must be positive")
        if not self.source:
            raise ValueError("semantic source must be nonempty")
        valid = np.isfinite(positions)
        if self.valid is not None:
            supplied = np.asarray(self.valid, dtype=bool)
            if supplied.shape == positions.shape[:2]:
                supplied = np.repeat(supplied[:, :, None], 3, axis=2)
            if supplied.shape != positions.shape:
                raise ValueError("semantic valid mask must have shape (F, Q) or (F, Q, 3)")
            valid &= supplied
        if not np.any(valid):
            raise ValueError("semantic evidence has no valid coordinates")
        anchor = None
        if self.compare_displacements:
            if self.anchor_positions_m is None:
                raise ValueError("displacement evidence requires anchor_positions_m")
            anchor = np.asarray(self.anchor_positions_m, dtype=float)
            if anchor.shape != positions.shape[1:] or not np.all(np.isfinite(anchor)):
                raise ValueError("anchor_positions_m must have finite shape (Q, 3)")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "physical_frame_indices", frames)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "anchor_positions_m", anchor)


def molmo_task_evidence(
    bundle: MolmoForecastBundle,
    forecast_id: str,
    physical: PhysicalPosterior,
    *,
    scale_m: float = 0.10,
    degrees_of_freedom: float = 3.0,
) -> SparseSemanticEvidence:
    """Align one released MolmoMotion forecast to ``H_Q(X)`` frames."""

    if forecast_id not in bundle.forecast_ids:
        raise ValueError(f"unknown MolmoMotion forecast id {forecast_id!r}")
    if bundle.query.case_name != physical.context.case_id:
        raise ValueError("MolmoMotion and PhysicalPosterior cases differ")
    forecast_index = bundle.forecast_ids.index(forecast_id)
    physical_horizon = physical.state_trajectories_m.shape[1] - 1
    available = min(
        bundle.future_horizon,
        physical_horizon // bundle.query.frame_stride,
    )
    if available < 1:
        raise ValueError("MolmoMotion and PhysicalPosterior have no common future")
    future_world = bundle.future_world_m[forecast_index, :, :available]
    return SparseSemanticEvidence(
        positions_m=np.transpose(future_world, (1, 0, 2)),
        node_indices=bundle.query.node_indices,
        physical_frame_indices=(
            np.arange(1, available + 1, dtype=float) * bundle.query.frame_stride
        ),
        scale_m=scale_m,
        degrees_of_freedom=degrees_of_freedom,
        compare_displacements=True,
        anchor_positions_m=bundle.query.anchor_positions_world_m,
        anchor_physical_frame=0,
        source=f"MolmoMotion:{forecast_id}:{bundle.checkpoint}",
    )


def query_point_readout(
    posterior: PhysicalPosterior,
    node_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    """Apply ``H_Q`` to every dense physical rollout component."""

    nodes = np.asarray(node_indices, dtype=np.int64)
    frames = np.asarray(frame_indices, dtype=float)
    trajectory = posterior.readout_trajectories_m
    if nodes.ndim != 1 or not len(nodes) or np.any(nodes < 0) or np.any(
        nodes >= trajectory.shape[2]
    ):
        raise ValueError("node_indices exceed the physical posterior")
    if frames.ndim != 1 or not len(frames) or np.min(frames) < 0.0 or np.max(
        frames
    ) > trajectory.shape[1] - 1:
        raise ValueError("frame_indices exceed the physical posterior")
    lower = np.floor(frames).astype(int)
    upper = np.ceil(frames).astype(int)
    alpha = (frames - lower).reshape(1, -1, 1, 1)
    selected_lower = trajectory[:, lower][:, :, nodes]
    selected_upper = trajectory[:, upper][:, :, nodes]
    return (1.0 - alpha) * selected_lower + alpha * selected_upper


def semantic_component_log_scores(
    posterior: PhysicalPosterior,
    evidence: SparseSemanticEvidence,
) -> np.ndarray:
    """Score only ``H_Q(X)`` with a robust Student-t product of experts."""

    predicted = query_point_readout(
        posterior,
        evidence.node_indices,
        evidence.physical_frame_indices,
    ).astype(float)
    target = evidence.positions_m
    if evidence.compare_displacements:
        physical_anchor = posterior.readout_trajectories_m[
            :, evidence.anchor_physical_frame, evidence.node_indices
        ].astype(float)
        predicted = predicted - physical_anchor[:, None]
        target = target - evidence.anchor_positions_m[None]
    valid = np.asarray(evidence.valid, dtype=bool)
    residual = predicted - target[None]
    standardized = residual / evidence.scale_m
    terms = -0.5 * (evidence.degrees_of_freedom + 1.0) * np.log1p(
        np.square(standardized) / evidence.degrees_of_freedom
    )
    valid_count = int(np.sum(valid))
    return np.sum(np.where(valid[None], terms, 0.0), axis=(1, 2, 3)) / valid_count


def build_task_posterior(
    physical: PhysicalPosterior,
    evidence: SparseSemanticEvidence,
    *,
    beta: float,
) -> TaskPosterior:
    """Create a separate intention-conditioned posterior over physical rollouts."""

    if beta < 0.0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative")
    scores = semantic_component_log_scores(physical, evidence)
    if beta == 0.0:
        task_weights = physical.weights.copy()
    else:
        log_weights = np.log(np.maximum(physical.weights, 1e-300)) + beta * scores
        maximum = float(np.max(log_weights))
        task_weights = np.exp(log_weights - maximum)
        task_weights /= np.sum(task_weights)
    return TaskPosterior(
        context=physical.context,
        physical_posterior_id=physical.artifact_id,
        component_ids=physical.component_ids,
        physical_weights=physical.weights,
        task_weights=task_weights,
        semantic_log_scores=scores,
        beta=float(beta),
        query_node_indices=evidence.node_indices,
        semantic_source=evidence.source,
        metadata={
            "semantic_interface": "q_MM(H_Q(X) | I, language)",
            "physical_state_updated_by_semantics": False,
            "physical_parameters_updated_by_semantics": False,
            "positions_sha256": array_sha256(evidence.positions_m),
            "physical_frame_indices": evidence.physical_frame_indices.tolist(),
            "compare_displacements": evidence.compare_displacements,
            "scale_m": evidence.scale_m,
            "degrees_of_freedom": evidence.degrees_of_freedom,
        },
    )


def task_posterior_mean(
    physical: PhysicalPosterior,
    task: TaskPosterior,
) -> np.ndarray:
    """Return the intention-conditioned mean without mutating physical support."""

    if task.physical_posterior_id != physical.artifact_id:
        raise ValueError("TaskPosterior does not reference this PhysicalPosterior")
    if task.component_ids != physical.component_ids:
        raise ValueError("task and physical component identities differ")
    return np.einsum(
        "k,ktnc->tnc",
        task.task_weights,
        physical.readout_trajectories_m,
    )
