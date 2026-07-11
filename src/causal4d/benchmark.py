"""Protocol construction for the Causal4D controlled counterfactual benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any

import numpy as np

from causal4d.simulator import (
    Action,
    GraphObject,
    PhysicalParameters,
    SimulatorConfig,
    WorldCondition,
    simulate,
)


@dataclass(frozen=True)
class CounterfactualBenchmarkConfig:
    """All choices required to reproduce the first controlled milestone."""

    frame_count: int = 56
    dt: float = 0.03
    velocity_drag: float = 0.18
    training_repeats: int = 2
    observation_noise_std_m: float = 0.0015
    inference_noise_std_m: float = 0.006
    likelihood_power: float = 0.12
    fit_frame_stride: int = 4
    parameter_grid_count: int = 5
    stiffness_relative_width: float = 0.40
    damping_relative_width: float = 0.55
    contact_gain_relative_width: float = 0.30
    world_control_rotation_deg: float = 8.0
    world_nonlinear_stiffening: float = 0.18
    generative_ridge: float = 0.35
    predictive_variance_floor_m2: float = 4e-6
    confidence_level: float = 0.90
    gross_failure_threshold_m: float = 0.04

    def __post_init__(self) -> None:
        if self.training_repeats < 1:
            raise ValueError("training_repeats must be positive")
        if self.observation_noise_std_m <= 0.0 or self.inference_noise_std_m <= 0.0:
            raise ValueError("noise scales must be positive")
        if not 0.0 < self.likelihood_power <= 1.0:
            raise ValueError("likelihood_power must be in (0, 1]")
        if self.fit_frame_stride < 1:
            raise ValueError("fit_frame_stride must be positive")
        if self.parameter_grid_count < 3 or self.parameter_grid_count % 2 == 0:
            raise ValueError(
                "parameter_grid_count must be an odd integer of at least three"
            )
        if self.generative_ridge <= 0.0:
            raise ValueError("generative_ridge must be positive")
        if self.predictive_variance_floor_m2 <= 0.0:
            raise ValueError("predictive_variance_floor_m2 must be positive")
        if not np.isfinite(self.world_control_rotation_deg):
            raise ValueError("world_control_rotation_deg must be finite")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        SimulatorConfig(
            frame_count=self.frame_count,
            dt=self.dt,
            velocity_drag=self.velocity_drag,
        )

    @property
    def simulator(self) -> SimulatorConfig:
        return SimulatorConfig(
            frame_count=self.frame_count,
            dt=self.dt,
            velocity_drag=self.velocity_drag,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Episode:
    """One observed or counterfactual interaction for a fixed object."""

    graph_object: GraphObject
    action: Action
    condition: WorldCondition
    repeat_id: int
    truth: np.ndarray
    observations: np.ndarray
    descriptor: np.ndarray

    @property
    def episode_id(self) -> str:
        return (
            f"{self.graph_object.name}/{self.action.action_id}/"
            f"{self.condition.name}/r{self.repeat_id}"
        )


@dataclass(frozen=True)
class ObjectProtocol:
    graph_object: GraphObject
    actions: tuple[Action, ...]
    training_conditions: tuple[WorldCondition, ...]
    validation_condition: WorldCondition
    test_conditions: tuple[WorldCondition, ...]

    @property
    def train_actions(self) -> tuple[Action, ...]:
        return tuple(action for action in self.actions if action.split == "train")

    @property
    def validation_action(self) -> Action:
        return next(action for action in self.actions if action.split == "validation")

    @property
    def test_action(self) -> Action:
        return next(action for action in self.actions if action.split == "test")

    def as_dict(self) -> dict[str, Any]:
        return {
            "object": self.graph_object.as_dict(),
            "actions": [action.as_dict() for action in self.actions],
            "split": {
                "train": [action.action_id for action in self.train_actions],
                "validation": [self.validation_action.action_id],
                "held_out_test": [self.test_action.action_id],
            },
            "training_contact_conditions": [
                condition.as_dict() for condition in self.training_conditions
            ],
            "validation_contact_condition": self.validation_condition.as_dict(),
            "held_out_world_conditions": [
                condition.as_dict() for condition in self.test_conditions
            ],
        }


def _grid_edges(
    rows: int, columns: int, *, diagonals: bool
) -> tuple[tuple[int, int], ...]:
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for column in range(columns):
            node = row * columns + column
            if column + 1 < columns:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + columns))
            if diagonals and row + 1 < rows and column + 1 < columns:
                edges.append((node, node + columns + 1))
                edges.append((node + 1, node + columns))
    return tuple(edges)


def make_objects() -> tuple[GraphObject, ...]:
    """Create three topologically and physically distinct deformable objects."""

    rope_positions = np.column_stack(
        [np.linspace(-0.30, 0.30, 7), np.zeros(7, dtype=float)]
    )
    rope = GraphObject(
        name="rope",
        rest_positions=rope_positions,
        edges=tuple((index, index + 1) for index in range(6)),
        mass=0.82,
        support_stiffness=0.50,
        true_parameters=PhysicalParameters(8.5, 0.62, 0.92),
        sensor_nodes=(0, 3, 6),
    )

    cloth_positions = np.asarray(
        [
            (column * 0.12 - 0.12, row * 0.12 - 0.12)
            for row in range(3)
            for column in range(3)
        ],
        dtype=float,
    )
    cloth = GraphObject(
        name="cloth",
        rest_positions=cloth_positions,
        edges=_grid_edges(3, 3, diagonals=True),
        mass=1.08,
        support_stiffness=0.42,
        true_parameters=PhysicalParameters(6.4, 0.84, 0.78),
        sensor_nodes=(0, 4, 8),
    )

    block_positions = np.asarray(
        [
            (column * 0.11 - 0.165, row * 0.13 - 0.065)
            for row in range(2)
            for column in range(4)
        ],
        dtype=float,
    )
    soft_block = GraphObject(
        name="soft_block",
        rest_positions=block_positions,
        edges=_grid_edges(2, 4, diagonals=True),
        mass=1.34,
        support_stiffness=0.68,
        true_parameters=PhysicalParameters(10.6, 1.08, 1.06),
        sensor_nodes=(0, 3, 6),
    )
    return rope, cloth, soft_block


def _envelope(frame_count: int, profile: str) -> np.ndarray:
    phase = np.linspace(0.0, 1.0, frame_count - 1)
    if profile == "smooth":
        return np.sin(np.pi * phase) ** 2
    if profile == "hold":
        return np.clip(phase / 0.20, 0.0, 1.0) * np.clip((1.0 - phase) / 0.20, 0.0, 1.0)
    if profile == "double":
        return np.sin(2.0 * np.pi * phase) ** 2
    if profile == "impulse":
        return np.exp(-0.5 * np.square((phase - 0.22) / 0.09))
    raise ValueError(f"unknown force profile: {profile}")


def _action_forces(
    frame_count: int,
    vectors: tuple[tuple[float, float], ...],
    profile: str,
    *,
    rotate: float = 0.0,
) -> np.ndarray:
    envelope = _envelope(frame_count, profile)
    base = np.asarray(vectors, dtype=float)
    forces = envelope[:, None, None] * base[None, :, :]
    if rotate:
        phase = np.linspace(0.0, 1.0, frame_count - 1)
        angles = rotate * (phase - 0.5)
        cosine = np.cos(angles)
        sine = np.sin(angles)
        x = forces[..., 0].copy()
        y = forces[..., 1].copy()
        forces[..., 0] = cosine[:, None] * x - sine[:, None] * y
        forces[..., 1] = sine[:, None] * x + cosine[:, None] * y
    return forces


def make_actions(graph_object: GraphObject, frame_count: int) -> tuple[Action, ...]:
    """Create four fitting actions, one validation action, and one held-out action."""

    positions = graph_object.rest_positions
    centre = np.mean(positions, axis=0)
    left = int(np.argmin(positions[:, 0]))
    right = int(np.argmax(positions[:, 0]))
    middle = int(np.argmin(np.linalg.norm(positions - centre, axis=1)))
    upper = int(np.argmax(positions[:, 1] + 0.05 * positions[:, 0]))

    return (
        Action(
            "left_lift",
            "train",
            (left,),
            _action_forces(frame_count, ((0.08, 0.48),), "smooth"),
        ),
        Action(
            "right_drag",
            "train",
            (right,),
            _action_forces(frame_count, ((0.43, 0.10),), "hold"),
        ),
        Action(
            "centre_pulse",
            "train",
            (middle,),
            _action_forces(frame_count, ((-0.08, -0.36),), "double"),
        ),
        Action(
            "dual_stretch",
            "train",
            (left, right),
            _action_forces(
                frame_count,
                ((-0.31, 0.12), (0.31, 0.12)),
                "smooth",
            ),
        ),
        Action(
            "reverse_sweep",
            "validation",
            (upper,),
            _action_forces(frame_count, ((-0.34, 0.27),), "hold", rotate=0.8),
        ),
        Action(
            "diagonal_hook",
            "test",
            (right,),
            _action_forces(frame_count, ((-0.30, 0.52),), "impulse", rotate=-0.5),
        ),
    )


def _training_conditions(
    config: CounterfactualBenchmarkConfig,
) -> tuple[WorldCondition, ...]:
    conditions: list[WorldCondition] = []
    for repeat in range(config.training_repeats):
        if repeat % 2 == 0:
            gain, delay, name = 1.0, 0, f"firm_r{repeat}"
        else:
            gain, delay, name = 0.82, 1, f"compliant_r{repeat}"
        conditions.append(
            WorldCondition(
                name=name,
                contact_gain_multiplier=gain,
                contact_delay_steps=delay,
                control_rotation_radians=np.deg2rad(config.world_control_rotation_deg),
                nonlinear_stiffening=config.world_nonlinear_stiffening,
            )
        )
    return tuple(conditions)


def build_protocol(
    config: CounterfactualBenchmarkConfig | None = None,
) -> tuple[ObjectProtocol, ...]:
    """Build the locked object/action/contact split."""

    cfg = config or CounterfactualBenchmarkConfig()
    training_conditions = _training_conditions(cfg)
    validation_condition = WorldCondition(
        name="validation_compliant",
        contact_gain_multiplier=0.90,
        contact_delay_steps=1,
        control_rotation_radians=np.deg2rad(cfg.world_control_rotation_deg),
        nonlinear_stiffening=cfg.world_nonlinear_stiffening,
    )
    test_conditions = (
        WorldCondition(
            name="matched_contact",
            control_rotation_radians=np.deg2rad(cfg.world_control_rotation_deg),
            nonlinear_stiffening=cfg.world_nonlinear_stiffening,
        ),
        WorldCondition(
            name="shifted_contact",
            contact_gain_multiplier=0.72,
            contact_delay_steps=2,
            shift_contact_nodes=True,
            control_rotation_radians=np.deg2rad(cfg.world_control_rotation_deg),
            nonlinear_stiffening=cfg.world_nonlinear_stiffening,
        ),
    )
    return tuple(
        ObjectProtocol(
            graph_object=graph_object,
            actions=make_actions(graph_object, cfg.frame_count),
            training_conditions=training_conditions,
            validation_condition=validation_condition,
            test_conditions=test_conditions,
        )
        for graph_object in make_objects()
    )


def action_descriptor(
    graph_object: GraphObject,
    action: Action,
    condition: WorldCondition,
) -> np.ndarray:
    """Describe an intervention without using its resulting object trajectory."""

    del condition
    # Baselines receive the commanded intervention, not the latent contact
    # transmission realised by the independent world model.
    contact_nodes = action.contact_nodes
    contact_position = np.mean(graph_object.rest_positions[list(contact_nodes)], axis=0)
    extent = np.ptp(graph_object.rest_positions, axis=0)
    extent = np.where(extent > 1e-8, extent, graph_object.characteristic_length)
    centre = np.mean(graph_object.rest_positions, axis=0)
    normalized_contact = (contact_position - centre) / extent
    effective_forces = action.commanded_forces
    impulse = np.sum(effective_forces, axis=(0, 1))
    impulse *= 1.0 / max(action.commanded_forces.shape[0], 1)
    peak = float(np.max(np.linalg.norm(effective_forces, axis=-1)))
    force_magnitude = np.sum(np.linalg.norm(effective_forces, axis=-1), axis=1)
    phase = np.linspace(0.0, 1.0, force_magnitude.size)
    temporal_centroid = float(
        np.sum(phase * force_magnitude) / max(np.sum(force_magnitude), 1e-12)
    )
    return np.asarray(
        [
            normalized_contact[0],
            normalized_contact[1],
            impulse[0],
            impulse[1],
            peak,
            float(len(contact_nodes)),
            temporal_centroid,
        ],
        dtype=float,
    )


def make_parameter_grid(
    graph_object: GraphObject,
    config: CounterfactualBenchmarkConfig,
) -> np.ndarray:
    """Create a grid containing the truth and deliberately confounded alternatives."""

    truth = graph_object.true_parameters.as_array()
    widths = np.asarray(
        [
            config.stiffness_relative_width,
            config.damping_relative_width,
            config.contact_gain_relative_width,
        ]
    )
    axes = [
        value
        * np.linspace(
            1.0 - width,
            1.0 + width,
            config.parameter_grid_count,
        )
        for value, width in zip(truth, widths, strict=True)
    ]
    return np.asarray(list(product(*axes)), dtype=float)


def generate_episodes(
    protocol: ObjectProtocol,
    config: CounterfactualBenchmarkConfig,
    *,
    seed: int,
) -> tuple[tuple[Episode, ...], Episode, tuple[Episode, ...]]:
    """Generate noisy observed episodes and noiseless held-out worlds."""

    rng = np.random.default_rng(seed)

    def make_episode(
        action: Action,
        condition: WorldCondition,
        repeat_id: int,
        noisy: bool,
    ) -> Episode:
        truth = simulate(
            protocol.graph_object,
            action,
            protocol.graph_object.true_parameters,
            condition,
            config.simulator,
        )
        observations = truth.copy()
        if noisy:
            observations += rng.normal(
                scale=config.observation_noise_std_m,
                size=observations.shape,
            )
        return Episode(
            graph_object=protocol.graph_object,
            action=action,
            condition=condition,
            repeat_id=repeat_id,
            truth=truth,
            observations=observations,
            descriptor=action_descriptor(protocol.graph_object, action, condition),
        )

    training = tuple(
        make_episode(action, condition, repeat, True)
        for action in protocol.train_actions
        for repeat, condition in enumerate(protocol.training_conditions)
    )
    validation = make_episode(
        protocol.validation_action,
        protocol.validation_condition,
        0,
        True,
    )
    held_out = tuple(
        make_episode(protocol.test_action, condition, 0, False)
        for condition in protocol.test_conditions
    )
    return training, validation, held_out


def protocol_manifest(
    protocols: tuple[ObjectProtocol, ...],
    config: CounterfactualBenchmarkConfig,
) -> dict[str, Any]:
    """Return a serializable record of controls, truths, and locked splits."""

    return {
        "schema_version": 1,
        "benchmark": "causal4d-controlled-counterfactual-v1",
        "units": {
            "position": "metre",
            "force": "newton",
            "mass": "kilogram",
            "time": "second",
        },
        "config": config.as_dict(),
        "objects": [protocol.as_dict() for protocol in protocols],
        "evaluation_rule": {
            "fit": "four action templates with repeated contact conditions",
            "model_selection": "one validation interaction",
            "test": "one untouched action under matched and shifted contact worlds",
            "available_to_models": "commanded forces and nominal contact nodes",
            "evaluator_only": "realised contact gain, delay, node shift, frame bias, and world nonlinearity",
            "aggregation": "equal weight per object, seed, world, and method",
        },
    }
