"""Official PhysTwin rollout backend for Causal4D joint inference."""

from __future__ import annotations

import gc
import json
import pickle
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bayesian_phystwin.phystwin_controller_sensitivity import (
    controller_hand_count,
    infer_controller_groups,
)
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraph,
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)

from causal4d.rollout_bank import JointRolloutBank


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


@dataclass(frozen=True)
class BayesianPhysTwinParticles:
    """Selected object/controller spring-scale particles from a saved profile."""

    log_scales: np.ndarray
    weights: np.ndarray
    grid_indices: np.ndarray
    source_weight_key: str
    retained_probability_mass: float

    def __post_init__(self) -> None:
        particles = np.asarray(self.log_scales, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        indices = np.asarray(self.grid_indices, dtype=int)
        if particles.ndim != 2 or particles.shape[1] != 2:
            raise ValueError("log_scales must have shape (P, 2)")
        if weights.shape != (len(particles),) or indices.shape != (len(particles), 2):
            raise ValueError("particle weights and grid indices must match log_scales")
        if not np.all(np.isfinite(particles)) or not np.all(np.isfinite(weights)):
            raise ValueError("particle arrays must be finite")
        if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("particle weights must be nonnegative and sum to one")
        if not 0.0 < self.retained_probability_mass <= 1.0 + 1e-12:
            raise ValueError("retained probability mass must lie in (0, 1]")
        object.__setattr__(self, "log_scales", particles)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "grid_indices", indices)


def load_bayesian_phystwin_particles(
    profile_path: str | Path,
    *,
    maximum_count: int,
    weight_key: str | None = None,
) -> BayesianPhysTwinParticles:
    """Load the highest-mass spring particles from a Bayesian-PhysTwin grid."""

    if maximum_count < 1:
        raise ValueError("maximum_count must be positive")
    with np.load(profile_path, allow_pickle=False) as archive:
        required = {"object_log_scales", "controller_log_scales"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError("parameter profile is missing: " + ", ".join(sorted(missing)))
        object_grid = np.asarray(archive["object_log_scales"], dtype=float)
        controller_grid = np.asarray(archive["controller_log_scales"], dtype=float)
        if weight_key is None:
            available = [
                key
                for key in ("prediction_weights", "posterior_weights")
                if key in archive.files
            ]
            if not available:
                raise ValueError("parameter profile has no posterior weight grid")
            selected_weight_key = available[0]
        else:
            selected_weight_key = weight_key
            if selected_weight_key not in archive.files:
                raise ValueError(f"parameter profile has no {selected_weight_key!r}")
        weight_grid = np.asarray(archive[selected_weight_key], dtype=float)
    expected = (len(object_grid), len(controller_grid))
    if weight_grid.shape != expected:
        raise ValueError(f"profile weight grid must have shape {expected}")
    if np.any(weight_grid < 0.0) or not np.all(np.isfinite(weight_grid)):
        raise ValueError("profile weights must be finite and nonnegative")
    total = float(np.sum(weight_grid))
    if total <= 0.0:
        raise ValueError("profile weights must have positive mass")
    normalized = weight_grid / total
    object_mesh, controller_mesh = np.meshgrid(
        object_grid,
        controller_grid,
        indexing="ij",
    )
    particles = np.column_stack((object_mesh.reshape(-1), controller_mesh.reshape(-1)))
    grid_indices = np.column_stack(
        np.unravel_index(np.arange(weight_grid.size), weight_grid.shape)
    )
    flat_weights = normalized.reshape(-1)
    order = np.lexsort((np.arange(len(flat_weights)), -flat_weights))
    selected = order[: min(maximum_count, len(order))]
    retained = float(np.sum(flat_weights[selected]))
    weights = flat_weights[selected] / retained
    return BayesianPhysTwinParticles(
        log_scales=particles[selected],
        weights=weights,
        grid_indices=grid_indices[selected],
        source_weight_key=selected_weight_key,
        retained_probability_mass=retained,
    )


@dataclass(frozen=True)
class PhysTwinActionProposal:
    """One candidate controller trajectory for known or hidden future actions."""

    proposal_id: str
    controller_points_m: np.ndarray
    prior_weight: float
    future_action_observed: bool
    provenance: str

    def __post_init__(self) -> None:
        controls = np.asarray(self.controller_points_m, dtype=float)
        if controls.ndim != 3 or controls.shape[2] != 3:
            raise ValueError("controller_points_m must have shape (T, C, 3)")
        if not np.all(np.isfinite(controls)):
            raise ValueError("controller points must be finite")
        if not self.proposal_id or self.prior_weight <= 0.0:
            raise ValueError("action proposal id and prior weight must be valid")
        object.__setattr__(self, "controller_points_m", controls)

    def metadata(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "prior_weight": float(self.prior_weight),
            "future_action_observed": self.future_action_observed,
            "provenance": self.provenance,
        }


def known_action_proposal(controller_points_m: np.ndarray) -> PhysTwinActionProposal:
    return PhysTwinActionProposal(
        proposal_id="known_action",
        controller_points_m=np.asarray(controller_points_m, dtype=float).copy(),
        prior_weight=1.0,
        future_action_observed=True,
        provenance="released future controller trajectory",
    )


def hidden_action_proposals(
    controller_points_m: np.ndarray,
    *,
    start_frame: int,
    history_frames: int = 4,
    damping: float = 0.94,
) -> tuple[PhysTwinActionProposal, ...]:
    """Build action proposals using controller history only, never future controls."""

    controls = np.asarray(controller_points_m, dtype=float)
    if controls.ndim != 3 or controls.shape[2] != 3:
        raise ValueError("controller_points_m must have shape (T, C, 3)")
    if not 2 <= history_frames <= start_frame < len(controls):
        raise ValueError("hidden action history and start frame are inconsistent")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")
    anchor = controls[start_frame - 1].copy()
    velocity = np.mean(
        np.diff(controls[start_frame - history_frames : start_frame], axis=0),
        axis=0,
    )

    def proposal(identifier: str, mode: str, prior: float) -> PhysTwinActionProposal:
        values = controls.copy()
        values[start_frame:] = anchor
        current = anchor.copy()
        for offset, frame in enumerate(range(start_frame, len(values)), start=1):
            if mode == "persist":
                delta = np.zeros_like(velocity)
            elif mode == "continue":
                delta = velocity * damping ** (offset - 1)
            elif mode == "reverse":
                delta = -velocity * damping ** (offset - 1)
            elif mode == "orthogonal":
                delta = velocity[:, [1, 0, 2]].copy()
                delta[:, 0] *= -1.0
                delta[:, 2] = velocity[:, 2]
                delta *= damping ** (offset - 1)
            else:
                raise ValueError(f"unknown hidden action mode {mode!r}")
            current = current + delta
            values[frame] = current
        return PhysTwinActionProposal(
            proposal_id=identifier,
            controller_points_m=values,
            prior_weight=prior,
            future_action_observed=False,
            provenance=f"history-only {mode} proposal",
        )

    return (
        proposal("history_continue", "continue", 0.40),
        proposal("history_persist", "persist", 0.25),
        proposal("history_reverse", "reverse", 0.20),
        proposal("history_orthogonal", "orthogonal", 0.15),
    )


@dataclass(frozen=True)
class PhysTwinContactState:
    """Realized attachment and controller-transfer hypothesis."""

    attachment_shifts: tuple[int, ...]
    gain_multiplier: float
    delay_steps: int
    slip_fraction: float
    rotation_degrees: float
    prior_weight: float

    def __post_init__(self) -> None:
        if not self.attachment_shifts or any(value not in {-1, 0, 1} for value in self.attachment_shifts):
            raise ValueError("attachment shifts must be -1, 0, or 1 per hand")
        if self.gain_multiplier <= 0.0 or self.delay_steps < 0:
            raise ValueError("contact gain and delay must be valid")
        if not 0.0 <= self.slip_fraction < 1.0:
            raise ValueError("slip_fraction must lie in [0, 1)")
        if not np.isfinite(self.rotation_degrees) or self.prior_weight <= 0.0:
            raise ValueError("rotation and prior weight must be valid")

    @property
    def state_id(self) -> str:
        shifts = "_".join(f"{value:+d}" for value in self.attachment_shifts)
        return (
            f"shift_{shifts}__gain_{self.gain_multiplier:.3f}"
            f"__delay_{self.delay_steps}__slip_{self.slip_fraction:.3f}"
            f"__rot_{self.rotation_degrees:+.1f}"
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "attachment_shifts": list(self.attachment_shifts),
            "gain_multiplier": float(self.gain_multiplier),
            "delay_steps": int(self.delay_steps),
            "slip_fraction": float(self.slip_fraction),
            "rotation_degrees": float(self.rotation_degrees),
            "contact_prior_weight": float(self.prior_weight),
        }


@dataclass(frozen=True)
class PhysTwinHypothesisConfig:
    attachment_shift_values: tuple[int, ...] = (-1, 0, 1)
    gain_values: tuple[float, ...] = (0.85, 1.0, 1.15)
    delay_values: tuple[int, ...] = (0, 2)
    slip_values: tuple[float, ...] = (0.0, 0.20)
    rotation_values_degrees: tuple[float, ...] = (-8.0, 0.0, 8.0)
    maximum_contact_states: int = 12

    def __post_init__(self) -> None:
        if set(self.attachment_shift_values) - {-1, 0, 1} or 0 not in self.attachment_shift_values:
            raise ValueError("attachment shift values must include zero and use -1/0/1")
        if not self.gain_values or min(self.gain_values) <= 0.0 or 1.0 not in self.gain_values:
            raise ValueError("gain values must be positive and include 1")
        if not self.delay_values or min(self.delay_values) < 0 or 0 not in self.delay_values:
            raise ValueError("delay values must be nonnegative and include 0")
        if not self.slip_values or min(self.slip_values) < 0.0 or max(self.slip_values) >= 1.0 or 0.0 not in self.slip_values:
            raise ValueError("slip values must lie in [0, 1) and include 0")
        if not self.rotation_values_degrees or 0.0 not in self.rotation_values_degrees:
            raise ValueError("rotation values must include zero")
        if self.maximum_contact_states < 1:
            raise ValueError("maximum_contact_states must be positive")


def _contact_prior_score(
    shifts: tuple[int, ...],
    gain: float,
    delay: int,
    slip: float,
    rotation: float,
    *,
    shift_value_count: int,
) -> float:
    nonzero_shift_probability = 0.30 / max(shift_value_count - 1, 1)
    shift_score = float(
        np.prod([0.70 if value == 0 else nonzero_shift_probability for value in shifts])
    )
    gain_score = float(np.exp(-0.5 * ((gain - 1.0) / 0.12) ** 2))
    delay_score = float(np.exp(-delay / 1.5))
    slip_score = float(np.exp(-slip / 0.15))
    rotation_score = float(np.exp(-0.5 * (rotation / 6.0) ** 2))
    return shift_score * gain_score * delay_score * slip_score * rotation_score


def build_contact_states(
    hand_count: int,
    config: PhysTwinHypothesisConfig | None = None,
) -> tuple[PhysTwinContactState, ...]:
    """Build a prior-ranked beam while retaining every latent contact channel."""

    cfg = config or PhysTwinHypothesisConfig()
    if hand_count < 1:
        raise ValueError("hand_count must be positive")
    candidates: dict[
        tuple[tuple[int, ...], float, int, float, float], float
    ] = {}
    for shifts, gain, delay, slip, rotation in product(
        product(cfg.attachment_shift_values, repeat=hand_count),
        cfg.gain_values,
        cfg.delay_values,
        cfg.slip_values,
        cfg.rotation_values_degrees,
    ):
        key = (tuple(shifts), float(gain), int(delay), float(slip), float(rotation))
        candidates[key] = _contact_prior_score(
            *key,
            shift_value_count=len(cfg.attachment_shift_values),
        )

    nominal = ((0,) * hand_count, 1.0, 0, 0.0, 0.0)
    required = [nominal]
    for hand in range(hand_count):
        for shift in cfg.attachment_shift_values:
            if shift:
                values = [0] * hand_count
                values[hand] = shift
                required.append((tuple(values), 1.0, 0, 0.0, 0.0))
    required.extend(
        ((0,) * hand_count, value, 0, 0.0, 0.0)
        for value in cfg.gain_values
        if value != 1.0
    )
    required.extend(
        ((0,) * hand_count, 1.0, value, 0.0, 0.0)
        for value in cfg.delay_values
        if value != 0
    )
    required.extend(
        ((0,) * hand_count, 1.0, 0, value, 0.0)
        for value in cfg.slip_values
        if value != 0.0
    )
    required.extend(
        ((0,) * hand_count, 1.0, 0, 0.0, value)
        for value in cfg.rotation_values_degrees
        if value != 0.0
    )
    required = list(dict.fromkeys(required))
    if cfg.maximum_contact_states < len(required):
        raise ValueError(
            "maximum_contact_states is too small to retain every contact channel; "
            f"need at least {len(required)}"
        )
    ranked = sorted(candidates, key=lambda key: (-candidates[key], key))
    selected = required + [key for key in ranked if key not in set(required)]
    selected = selected[: cfg.maximum_contact_states]
    selected_scores = np.asarray([candidates[key] for key in selected], dtype=float)
    selected_scores /= np.sum(selected_scores)
    return tuple(
        PhysTwinContactState(
            attachment_shifts=key[0],
            gain_multiplier=key[1],
            delay_steps=key[2],
            slip_fraction=key[3],
            rotation_degrees=key[4],
            prior_weight=float(weight),
        )
        for key, weight in zip(selected, selected_scores, strict=True)
    )


@dataclass(frozen=True)
class PhysTwinRolloutHypothesis:
    hypothesis_id: str
    action_proposal_id: str
    action_prior_weight: float
    contact: PhysTwinContactState
    prior_weight: float

    def metadata(self, proposal: PhysTwinActionProposal) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "prior_weight": float(self.prior_weight),
            "action": proposal.metadata(),
            "contact": self.contact.metadata(),
        }


def build_rollout_hypotheses(
    proposals: Sequence[PhysTwinActionProposal],
    contact_states: Sequence[PhysTwinContactState],
) -> tuple[PhysTwinRolloutHypothesis, ...]:
    if not proposals or not contact_states:
        raise ValueError("rollout hypotheses require actions and contact states")
    action_total = float(sum(proposal.prior_weight for proposal in proposals))
    hypotheses = []
    for proposal, contact in product(proposals, contact_states):
        prior = (proposal.prior_weight / action_total) * contact.prior_weight
        hypotheses.append(
            PhysTwinRolloutHypothesis(
                hypothesis_id=f"{proposal.proposal_id}__{contact.state_id}",
                action_proposal_id=proposal.proposal_id,
                action_prior_weight=proposal.prior_weight / action_total,
                contact=contact,
                prior_weight=prior,
            )
        )
    total = float(sum(hypothesis.prior_weight for hypothesis in hypotheses))
    return tuple(
        PhysTwinRolloutHypothesis(
            hypothesis_id=value.hypothesis_id,
            action_proposal_id=value.action_proposal_id,
            action_prior_weight=value.action_prior_weight,
            contact=value.contact,
            prior_weight=value.prior_weight / total,
        )
        for value in hypotheses
    )


def transform_controller_trajectory(
    controller_points_m: np.ndarray,
    groups: np.ndarray,
    contact: PhysTwinContactState,
    *,
    start_frame: int,
) -> np.ndarray:
    """Apply delay, slip, and direction error to future controller targets."""

    controls = np.asarray(controller_points_m, dtype=float)
    labels = np.asarray(groups, dtype=int)
    if controls.ndim != 3 or controls.shape[2] != 3:
        raise ValueError("controller_points_m must have shape (T, C, 3)")
    if labels.shape != (controls.shape[1],):
        raise ValueError("groups must label every controller point")
    if len(contact.attachment_shifts) != int(np.max(labels)) + 1:
        raise ValueError("contact hand count and controller groups differ")
    if not 1 <= start_frame < len(controls):
        raise ValueError("start_frame must leave a future controller interval")
    delayed = controls.copy()
    for frame in range(start_frame, len(controls)):
        source = max(start_frame - 1, frame - contact.delay_steps)
        delayed[frame] = controls[source]
    anchor = controls[start_frame - 1]
    radians = float(np.deg2rad(contact.rotation_degrees))
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    transformed = delayed.copy()
    future_displacement = delayed[start_frame:] - anchor[None]
    future_displacement = future_displacement @ rotation.T
    future_displacement *= 1.0 - contact.slip_fraction
    transformed[start_frame:] = anchor[None] + future_displacement
    return transformed.astype(np.float32)


@dataclass(frozen=True)
class AttachmentGraphVariant:
    graph: PhysTwinSpringGraph
    attachment_shifts: tuple[int, ...]
    changed_controller_springs: int


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= 0.0:
        raise ValueError("object points have no principal-axis extent")
    axis = right[0]
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0.0:
        axis = -axis
    return axis


def shift_phystwin_attachment_graph(
    graph: PhysTwinSpringGraph,
    controller_groups: np.ndarray,
    attachment_shifts: Sequence[int],
) -> AttachmentGraphVariant:
    """Move every controller spring endpoint by one coherent object-graph hop."""

    groups = np.asarray(controller_groups, dtype=int)
    shifts = tuple(int(value) for value in attachment_shifts)
    if groups.ndim != 1 or not len(groups):
        raise ValueError("controller_groups must be a nonempty vector")
    if len(shifts) != int(np.max(groups)) + 1 or any(value not in {-1, 0, 1} for value in shifts):
        raise ValueError("attachment_shifts must provide -1/0/1 for every group")
    object_count = len(graph.vertices) - len(groups)
    if object_count <= 0:
        raise ValueError("graph must contain object and controller vertices")
    adjacency: list[list[int]] = [[] for _ in range(object_count)]
    for first, second in graph.springs[: graph.num_object_springs]:
        first_i, second_i = int(first), int(second)
        adjacency[first_i].append(second_i)
        adjacency[second_i].append(first_i)
    for values in adjacency:
        values.sort()
    axis = _principal_axis(np.asarray(graph.vertices[:object_count], dtype=float))
    springs = np.asarray(graph.springs, dtype=np.int32).copy()
    rest_lengths = np.asarray(graph.rest_lengths, dtype=np.float32).copy()
    changed = 0
    for spring_index in range(graph.num_object_springs, len(springs)):
        first, second = map(int, springs[spring_index])
        if first >= object_count and second < object_count:
            control_vertex, object_vertex, object_column = first, second, 1
        elif second >= object_count and first < object_count:
            control_vertex, object_vertex, object_column = second, first, 0
        else:
            raise ValueError("controller spring must connect one object and one control")
        control_index = control_vertex - object_count
        shift = shifts[int(groups[control_index])]
        if shift and adjacency[object_vertex]:
            candidates = np.asarray(adjacency[object_vertex], dtype=int)
            delta = graph.vertices[candidates] - graph.vertices[object_vertex]
            scores = shift * (delta @ axis)
            best_order = np.lexsort((candidates, -scores))
            shifted_vertex = int(candidates[best_order[0]])
            springs[spring_index, object_column] = shifted_vertex
            changed += int(shifted_vertex != object_vertex)
        object_endpoint = int(springs[spring_index, object_column])
        rest_lengths[spring_index] = float(
            np.linalg.norm(
                graph.vertices[control_vertex].astype(float)
                - graph.vertices[object_endpoint].astype(float)
            )
        )
    variant = PhysTwinSpringGraph(
        vertices=np.asarray(graph.vertices, dtype=np.float32).copy(),
        springs=springs,
        rest_lengths=rest_lengths,
        masses=np.asarray(graph.masses, dtype=np.float32).copy(),
        num_object_springs=graph.num_object_springs,
    )
    return AttachmentGraphVariant(variant, shifts, changed)


@dataclass(frozen=True)
class OfficialPhysTwinBackendConfig:
    dt: float = 5e-5
    num_substeps: int = 667
    velocity_history_frames: int = 3
    deterministic_spring_forces: bool = True
    self_collision: bool | None = None
    device: str = "cuda:0"
    variance_floor_m2: float = 2.5e-5
    confidence_level: float = 0.90

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.num_substeps < 1:
            raise ValueError("PhysTwin time step and substeps must be positive")
        if self.velocity_history_frames < 2:
            raise ValueError("velocity_history_frames must be at least two")
        if self.variance_floor_m2 <= 0.0:
            raise ValueError("variance_floor_m2 must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must lie in (0, 1)")


class OfficialPhysTwinBackend:
    """Run Causal4D hypotheses through the pinned official Warp simulator."""

    def __init__(
        self,
        *,
        official_repo: str | Path,
        final_data_path: str | Path,
        optimal_params_path: str | Path,
        checkpoint_path: str | Path,
        baseline_trajectory_path: str | Path,
        profile_path: str | Path,
        train_end_frame: int,
        parameter_particle_count: int,
        config: OfficialPhysTwinBackendConfig | None = None,
    ) -> None:
        self.official_repo = Path(official_repo)
        self.final_data_path = Path(final_data_path)
        self.optimal_params_path = Path(optimal_params_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.baseline_trajectory_path = Path(baseline_trajectory_path)
        self.profile_path = Path(profile_path)
        self.config = config or OfficialPhysTwinBackendConfig()
        self.data = _load_pickle(self.final_data_path)
        self.optimal = _load_pickle(self.optimal_params_path)
        self.baseline = np.asarray(
            _load_pickle(self.baseline_trajectory_path), dtype=np.float32
        )
        self.object_points = np.asarray(self.data["object_points"], dtype=np.float32)
        self.visible = np.asarray(self.data["object_visibilities"], dtype=bool)
        self.motion_valid = np.asarray(self.data["object_motions_valid"], dtype=bool)
        self.controller_points = np.asarray(
            self.data["controller_points"], dtype=np.float32
        )
        self.surface_points = np.asarray(self.data["surface_points"], dtype=np.float32)
        self.interior_points = np.asarray(self.data["interior_points"], dtype=np.float32)
        self.frame_count, self.original_count, coordinate_count = self.object_points.shape
        if coordinate_count != 3 or not self.config.velocity_history_frames <= train_end_frame < self.frame_count:
            raise ValueError("train_end_frame is incompatible with the PhysTwin case")
        self.train_end_frame = int(train_end_frame)
        structure = np.concatenate(
            (self.object_points[0], self.surface_points, self.interior_points),
            axis=0,
        )
        if self.baseline.shape != (self.frame_count, len(structure), 3):
            raise ValueError("baseline trajectory does not match the PhysTwin state")
        self.graph = build_phystwin_spring_graph(
            structure,
            self.controller_points[0],
            config=PhysTwinSpringGraphConfig(
                object_radius=float(self.optimal["object_radius"]),
                object_max_neighbours=int(self.optimal["object_max_neighbours"]),
                controller_radius=float(self.optimal["controller_radius"]),
                controller_max_neighbours=int(self.optimal["controller_max_neighbours"]),
            ),
        )
        self.case_name = self.final_data_path.resolve().parent.name
        self.hand_count = controller_hand_count(self.case_name)
        self.controller_groups = infer_controller_groups(
            self.controller_points[0], group_count=self.hand_count
        )
        self.particles = load_bayesian_phystwin_particles(
            self.profile_path,
            maximum_count=parameter_particle_count,
        )

    @property
    def observations_from_endpoint(self) -> np.ndarray:
        return self.object_points[self.train_end_frame - 1 :]

    @property
    def observation_mask_from_endpoint(self) -> np.ndarray:
        return (
            self.visible[self.train_end_frame - 1 :]
            & self.motion_valid[self.train_end_frame - 1 :]
        )

    def default_manifest(self) -> dict[str, Any]:
        return {
            "backend": "official_phystwin_warp",
            "case": self.case_name,
            "train_end_frame": self.train_end_frame,
            "source_paths": {
                "official_repo": str(self.official_repo.resolve()),
                "final_data": str(self.final_data_path.resolve()),
                "optimal_params": str(self.optimal_params_path.resolve()),
                "checkpoint": str(self.checkpoint_path.resolve()),
                "baseline_trajectory": str(self.baseline_trajectory_path.resolve()),
                "parameter_profile": str(self.profile_path.resolve()),
            },
            "parameter_particles": {
                "count": len(self.particles.weights),
                "log_scale_names": ["object_springs", "controller_springs"],
                "source_weight_key": self.particles.source_weight_key,
                "retained_probability_mass": self.particles.retained_probability_mass,
                "grid_indices": self.particles.grid_indices.tolist(),
                "log_scales": self.particles.log_scales.tolist(),
                "weights": self.particles.weights.tolist(),
            },
            "controller_groups": self.controller_groups.tolist(),
            "runtime": asdict(self.config),
        }

    def build_rollout_bank(
        self,
        action_proposals: Sequence[PhysTwinActionProposal],
        *,
        hypothesis_config: PhysTwinHypothesisConfig | None = None,
    ) -> tuple[JointRolloutBank, dict[str, Any]]:
        """Simulate all action/contact hypotheses under selected theta particles."""

        proposal_by_id = {proposal.proposal_id: proposal for proposal in action_proposals}
        if len(proposal_by_id) != len(action_proposals):
            raise ValueError("action proposal ids must be unique")
        if any(proposal.controller_points_m.shape != self.controller_points.shape for proposal in action_proposals):
            raise ValueError("action proposal controls must match the PhysTwin case")
        contact_states = build_contact_states(self.hand_count, hypothesis_config)
        hypotheses = build_rollout_hypotheses(action_proposals, contact_states)
        trajectory_shape = (
            len(hypotheses),
            len(self.particles.weights),
            self.frame_count - self.train_end_frame + 1,
            self.original_count,
            3,
        )
        trajectories = np.empty(trajectory_shape, dtype=np.float32)
        endpoint_index = self.train_end_frame - 1
        frame_dt = self.config.dt * self.config.num_substeps
        from bayesian_phystwin.phystwin_state_injection import (
            _initialize_simulator,
            _released_self_collision_for_case,
            _rollout_restart,
            estimate_endpoint_velocity_delta,
        )

        endpoint_velocity = estimate_endpoint_velocity_delta(
            self.baseline[
                self.train_end_frame
                - self.config.velocity_history_frames : self.train_end_frame
            ],
            frame_dt=frame_dt,
        )
        self_collision = (
            _released_self_collision_for_case(self.case_name)
            if self.config.self_collision is None
            else self.config.self_collision
        )
        shift_diagnostics: dict[str, Any] = {}
        unique_shifts = tuple(
            dict.fromkeys(hypothesis.contact.attachment_shifts for hypothesis in hypotheses)
        )
        for shifts in unique_shifts:
            variant = shift_phystwin_attachment_graph(
                self.graph,
                self.controller_groups,
                shifts,
            )
            shift_key = ",".join(map(str, shifts))
            shift_diagnostics[shift_key] = {
                "changed_controller_springs": variant.changed_controller_springs,
                "controller_spring_count": len(variant.graph.springs)
                - variant.graph.num_object_springs,
            }
            simulator, torch, wp, _ = _initialize_simulator(
                self.official_repo,
                self.data,
                self.optimal,
                self.checkpoint_path,
                variant.graph,
                num_surface_points=self.original_count + len(self.surface_points),
                original_count=self.original_count,
                dt=self.config.dt,
                num_substeps=self.config.num_substeps,
                self_collision=bool(self_collision),
                deterministic_spring_forces=self.config.deterministic_spring_forces,
                spring_parameterization="grouped",
                device=self.config.device,
            )
            selected_hypotheses = [
                (index, hypothesis)
                for index, hypothesis in enumerate(hypotheses)
                if hypothesis.contact.attachment_shifts == shifts
            ]
            for hypothesis_index, hypothesis in selected_hypotheses:
                proposal = proposal_by_id[hypothesis.action_proposal_id]
                controls = transform_controller_trajectory(
                    proposal.controller_points_m,
                    self.controller_groups,
                    hypothesis.contact,
                    start_frame=self.train_end_frame,
                )
                simulator.controller_points = torch.as_tensor(
                    controls,
                    dtype=torch.float32,
                    device=self.config.device,
                ).contiguous()
                for particle_index, particle in enumerate(self.particles.log_scales):
                    group_scales = np.asarray(
                        [
                            particle[0],
                            particle[1] + np.log(hypothesis.contact.gain_multiplier),
                        ],
                        dtype=np.float32,
                    )
                    with torch.no_grad():
                        simulator.group_log_scale_tensor.copy_(
                            torch.as_tensor(
                                group_scales,
                                dtype=torch.float32,
                                device=self.config.device,
                            )
                        )
                    future = _rollout_restart(
                        simulator,
                        torch,
                        wp,
                        self.baseline[endpoint_index],
                        endpoint_velocity,
                        start_frame=self.train_end_frame,
                        stop_frame=self.frame_count,
                        device=self.config.device,
                    )
                    trajectories[hypothesis_index, particle_index, 0] = self.baseline[
                        endpoint_index, : self.original_count
                    ]
                    trajectories[hypothesis_index, particle_index, 1:] = future[
                        :, : self.original_count
                    ]
            del simulator
            gc.collect()
            torch.cuda.empty_cache()

        metadata = tuple(
            hypothesis.metadata(proposal_by_id[hypothesis.action_proposal_id])
            for hypothesis in hypotheses
        )
        bank = JointRolloutBank(
            hypothesis_ids=tuple(hypothesis.hypothesis_id for hypothesis in hypotheses),
            hypothesis_metadata=metadata,
            hypothesis_prior_weights=np.asarray(
                [hypothesis.prior_weight for hypothesis in hypotheses], dtype=float
            ),
            parameter_particles=self.particles.log_scales,
            parameter_weights=self.particles.weights,
            trajectories=trajectories,
            variance_floor_m2=self.config.variance_floor_m2,
            confidence_level=self.config.confidence_level,
        )
        manifest = self.default_manifest()
        manifest.update(
            {
                "hypothesis_count": len(hypotheses),
                "contact_state_count": len(contact_states),
                "action_proposals": [proposal.metadata() for proposal in action_proposals],
                "hypotheses": list(metadata),
                "attachment_shift_diagnostics": shift_diagnostics,
                "rollout_shape": list(trajectories.shape),
                "rollout_frame_interval": [endpoint_index, self.frame_count],
            }
        )
        return bank, manifest


def save_rollout_bank(
    path: str | Path,
    bank: JointRolloutBank,
    manifest: dict[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        hypothesis_ids=np.asarray(bank.hypothesis_ids),
        hypothesis_metadata_json=np.asarray(
            [json.dumps(value, sort_keys=True) for value in bank.hypothesis_metadata]
        ),
        hypothesis_prior_weights=bank.hypothesis_prior_weights,
        parameter_particles=bank.parameter_particles,
        parameter_weights=bank.parameter_weights,
        trajectories=bank.trajectories,
        variance_floor_m2=np.asarray(bank.variance_floor_m2),
        confidence_level=np.asarray(bank.confidence_level),
        manifest_json=np.asarray(json.dumps(manifest, sort_keys=True)),
    )


def load_rollout_bank(path: str | Path) -> tuple[JointRolloutBank, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = tuple(
            json.loads(str(value)) for value in archive["hypothesis_metadata_json"]
        )
        bank = JointRolloutBank(
            hypothesis_ids=tuple(map(str, archive["hypothesis_ids"])),
            hypothesis_metadata=metadata,
            hypothesis_prior_weights=np.asarray(
                archive["hypothesis_prior_weights"], dtype=float
            ),
            parameter_particles=np.asarray(archive["parameter_particles"], dtype=float),
            parameter_weights=np.asarray(archive["parameter_weights"], dtype=float),
            trajectories=np.asarray(archive["trajectories"], dtype=np.float32),
            variance_floor_m2=float(archive["variance_floor_m2"]),
            confidence_level=float(archive["confidence_level"]),
        )
        manifest = json.loads(str(archive["manifest_json"]))
    return bank, manifest
