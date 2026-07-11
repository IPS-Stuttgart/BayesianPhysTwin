"""Small deterministic graph simulator used by the counterfactual benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np


PARAMETER_NAMES = ("stiffness", "damping", "contact_gain")


@dataclass(frozen=True)
class PhysicalParameters:
    """Unknown object and contact parameters exposed to inference."""

    stiffness: float
    damping: float
    contact_gain: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.stiffness, self.damping, self.contact_gain], dtype=float
        )

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_array(cls, values: np.ndarray) -> PhysicalParameters:
        values = np.asarray(values, dtype=float)
        if values.shape != (3,):
            raise ValueError("physical parameter vector must have shape (3,)")
        return cls(*map(float, values))


@dataclass(frozen=True)
class GraphObject:
    """A deformable graph with known geometry, topology, mass, and true parameters."""

    name: str
    rest_positions: np.ndarray
    edges: tuple[tuple[int, int], ...]
    mass: float
    support_stiffness: float
    true_parameters: PhysicalParameters
    sensor_nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        positions = np.asarray(self.rest_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("rest_positions must have shape (node, 2)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("rest_positions must be finite")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.support_stiffness < 0.0:
            raise ValueError("support_stiffness must be non-negative")
        node_count = positions.shape[0]
        if not self.edges:
            raise ValueError("at least one graph edge is required")
        for first, second in self.edges:
            if first == second or not (
                0 <= first < node_count and 0 <= second < node_count
            ):
                raise ValueError("edges must connect distinct valid nodes")
        if not self.sensor_nodes or any(
            node < 0 or node >= node_count for node in self.sensor_nodes
        ):
            raise ValueError("sensor_nodes must contain valid node indices")
        positions = positions.copy()
        positions.setflags(write=False)
        object.__setattr__(self, "rest_positions", positions)

    @property
    def node_count(self) -> int:
        return int(self.rest_positions.shape[0])

    @property
    def characteristic_length(self) -> float:
        lengths = [
            np.linalg.norm(self.rest_positions[first] - self.rest_positions[second])
            for first, second in self.edges
        ]
        return float(np.median(lengths))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rest_positions_m": self.rest_positions.tolist(),
            "edges": [list(edge) for edge in self.edges],
            "known_mass_kg": float(self.mass),
            "known_support_stiffness": float(self.support_stiffness),
            "true_parameters": self.true_parameters.as_dict(),
            "sensor_nodes": list(self.sensor_nodes),
        }


@dataclass(frozen=True)
class Action:
    """Known commanded forces and their nominal material contact nodes."""

    action_id: str
    split: str
    contact_nodes: tuple[int, ...]
    commanded_forces: np.ndarray

    def __post_init__(self) -> None:
        forces = np.asarray(self.commanded_forces, dtype=float)
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if forces.ndim != 3 or forces.shape[1:] != (len(self.contact_nodes), 2):
            raise ValueError(
                "commanded_forces must have shape (transition, contact, 2)"
            )
        if forces.shape[0] == 0 or not np.all(np.isfinite(forces)):
            raise ValueError("commanded_forces must be non-empty and finite")
        forces = forces.copy()
        forces.setflags(write=False)
        object.__setattr__(self, "commanded_forces", forces)

    @property
    def frame_count(self) -> int:
        return int(self.commanded_forces.shape[0] + 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "split": self.split,
            "contact_nodes": list(self.contact_nodes),
            "commanded_forces_n": self.commanded_forces.tolist(),
        }


@dataclass(frozen=True)
class WorldCondition:
    """Observed contact regime and optional plan/world model discrepancy."""

    name: str
    contact_gain_multiplier: float = 1.0
    contact_delay_steps: int = 0
    shift_contact_nodes: bool = False
    contact_spread: float = 0.0
    control_rotation_radians: float = 0.0
    nonlinear_stiffening: float = 0.0

    def __post_init__(self) -> None:
        if self.contact_gain_multiplier <= 0.0:
            raise ValueError("contact_gain_multiplier must be positive")
        if self.contact_delay_steps < 0:
            raise ValueError("contact_delay_steps must be non-negative")
        if not 0.0 <= self.contact_spread < 1.0:
            raise ValueError("contact_spread must be in [0, 1)")
        if not np.isfinite(self.control_rotation_radians):
            raise ValueError("control_rotation_radians must be finite")
        if self.nonlinear_stiffening < 0.0:
            raise ValueError("nonlinear_stiffening must be non-negative")

    def plan_model(self) -> WorldCondition:
        """Return the nominal contact model available to all three baselines."""

        return replace(
            self,
            contact_gain_multiplier=1.0,
            contact_delay_steps=0,
            shift_contact_nodes=False,
            contact_spread=0.0,
            control_rotation_radians=0.0,
            nonlinear_stiffening=0.0,
        )

    def oracle_contact_model(self) -> WorldCondition:
        """Expose realized contact while retaining plan-model structural mismatch."""

        return replace(self, nonlinear_stiffening=0.0)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SimulatorConfig:
    frame_count: int = 56
    dt: float = 0.03
    velocity_drag: float = 0.18

    def __post_init__(self) -> None:
        if self.frame_count < 4:
            raise ValueError("frame_count must be at least four")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.velocity_drag < 0.0:
            raise ValueError("velocity_drag must be non-negative")


def graph_laplacian(graph_object: GraphObject) -> np.ndarray:
    """Return the unnormalised graph Laplacian."""

    laplacian = np.zeros(
        (graph_object.node_count, graph_object.node_count), dtype=float
    )
    for first, second in graph_object.edges:
        laplacian[first, first] += 1.0
        laplacian[second, second] += 1.0
        laplacian[first, second] -= 1.0
        laplacian[second, first] -= 1.0
    return laplacian


def resolved_contact_nodes(
    graph_object: GraphObject,
    action: Action,
    condition: WorldCondition,
) -> tuple[int, ...]:
    """Resolve a deterministic one-edge contact shift for the shifted condition."""

    if not condition.shift_contact_nodes:
        return action.contact_nodes
    adjacency: list[list[int]] = [[] for _ in range(graph_object.node_count)]
    for first, second in graph_object.edges:
        adjacency[first].append(second)
        adjacency[second].append(first)

    shifted: list[int] = []
    occupied = set(action.contact_nodes)
    for node in action.contact_nodes:
        candidates = sorted(
            adjacency[node],
            key=lambda candidate: (
                candidate in occupied,
                np.linalg.norm(
                    graph_object.rest_positions[candidate]
                    - graph_object.rest_positions[node]
                ),
                candidate,
            ),
        )
        shifted.append(candidates[0] if candidates else node)
    return tuple(shifted)


def graph_adjacency(graph_object: GraphObject) -> tuple[tuple[int, ...], ...]:
    """Return sorted one-hop graph neighbours for every material node."""

    adjacency: list[list[int]] = [[] for _ in range(graph_object.node_count)]
    for first, second in graph_object.edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    return tuple(tuple(sorted(neighbours)) for neighbours in adjacency)


def _nonlinear_force(
    displacement: np.ndarray,
    graph_object: GraphObject,
    parameters: PhysicalParameters,
    condition: WorldCondition,
) -> np.ndarray:
    if condition.nonlinear_stiffening == 0.0:
        return np.zeros_like(displacement)
    force = np.zeros_like(displacement)
    scale_squared = max(graph_object.characteristic_length**2, 1e-8)
    coefficient = condition.nonlinear_stiffening * parameters.stiffness / scale_squared
    for first, second in graph_object.edges:
        relative = displacement[second] - displacement[first]
        edge_force = coefficient * float(relative @ relative) * relative
        force[first] += edge_force
        force[second] -= edge_force
    return force


def simulate(
    graph_object: GraphObject,
    action: Action,
    parameters: PhysicalParameters,
    condition: WorldCondition,
    config: SimulatorConfig,
) -> np.ndarray:
    """Simulate positions for one known intervention using semi-implicit Euler."""

    if action.frame_count != config.frame_count:
        raise ValueError("action and simulator frame counts differ")
    if any(
        node < 0 or node >= graph_object.node_count for node in action.contact_nodes
    ):
        raise ValueError("action contains an invalid contact node")
    if min(parameters.as_array()) <= 0.0:
        raise ValueError("all physical parameters must be positive")

    laplacian = graph_laplacian(graph_object)
    displacement = np.zeros_like(graph_object.rest_positions)
    velocity = np.zeros_like(displacement)
    trajectory = np.empty((config.frame_count, graph_object.node_count, 2), dtype=float)
    trajectory[0] = graph_object.rest_positions
    contact_nodes = resolved_contact_nodes(graph_object, action, condition)
    adjacency = graph_adjacency(graph_object)

    for frame in range(1, config.frame_count):
        external_force = np.zeros_like(displacement)
        control_index = frame - 1 - condition.contact_delay_steps
        if control_index >= 0:
            scaled_forces = (
                parameters.contact_gain
                * condition.contact_gain_multiplier
                * action.commanded_forces[control_index]
            )
            if condition.control_rotation_radians:
                cosine = np.cos(condition.control_rotation_radians)
                sine = np.sin(condition.control_rotation_radians)
                rotation = np.asarray(((cosine, -sine), (sine, cosine)))
                scaled_forces = scaled_forces @ rotation.T
            for contact_index, node in enumerate(contact_nodes):
                contact_force = scaled_forces[contact_index]
                neighbours = adjacency[node]
                if condition.contact_spread and neighbours:
                    external_force[node] += (
                        1.0 - condition.contact_spread
                    ) * contact_force
                    shared_force = (
                        condition.contact_spread / len(neighbours)
                    ) * contact_force
                    for neighbour in neighbours:
                        external_force[neighbour] += shared_force
                else:
                    external_force[node] += contact_force

        force = external_force
        force -= parameters.stiffness * (laplacian @ displacement)
        force -= parameters.damping * (laplacian @ velocity)
        force -= config.velocity_drag * velocity
        force -= graph_object.support_stiffness * displacement
        force += _nonlinear_force(displacement, graph_object, parameters, condition)
        acceleration = force / graph_object.mass
        velocity = velocity + config.dt * acceleration
        displacement = displacement + config.dt * velocity
        trajectory[frame] = graph_object.rest_positions + displacement

    if not np.all(np.isfinite(trajectory)):
        raise FloatingPointError("simulation produced non-finite states")
    return trajectory


def simulate_particles(
    graph_object: GraphObject,
    action: Action,
    particles: np.ndarray,
    condition: WorldCondition,
    config: SimulatorConfig,
) -> np.ndarray:
    """Simulate one action for a bank of candidate parameter vectors."""

    particles = np.asarray(particles, dtype=float)
    if particles.ndim != 2 or particles.shape[1] != len(PARAMETER_NAMES):
        raise ValueError("particles must have shape (particle, 3)")
    return np.stack(
        [
            simulate(
                graph_object,
                action,
                PhysicalParameters.from_array(particle),
                condition,
                config,
            )
            for particle in particles
        ],
        axis=0,
    )
