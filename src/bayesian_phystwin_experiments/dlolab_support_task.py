"""Native unilateral-support task and source-only decision-sensitivity gate.

This is not the previous contact-free task and does not change its frozen
outcomes. The contact implementation is the pinned DLO-Lab ROD solver.
"""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin._portable_contracts import content_id

from .deform_state_restart import array_digest
from .dlolab_native import DloLabConfig, DloLabRuntime, verify_upstream


@dataclass(frozen=True)
class SupportTaskConfig:
    schema: str = "dlolab-unilateral-support-source-v1"
    rod: DloLabConfig = field(default_factory=lambda: DloLabConfig(node_count=25))
    support_nodes: int = 7
    support_interval_m: float = 0.06
    support_height_m: float = 0.48
    support_radius_m: float = 0.02
    prefix_steps: int = 125
    horizon_steps: int = 250
    loss_tail_steps: int = 50
    effort_weight: float = 0.02

    def __post_init__(self) -> None:
        if self.schema != "dlolab-unilateral-support-source-v1":
            raise ValueError("invalid support task schema")
        for name in (
            "support_nodes",
            "prefix_steps",
            "horizon_steps",
            "loss_tail_steps",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"invalid count {name}")
        if self.support_nodes < 3 or self.loss_tail_steps > self.horizon_steps:
            raise ValueError("invalid support or horizon geometry")
        for name in (
            "support_interval_m",
            "support_height_m",
            "support_radius_m",
            "effort_weight",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid positive value {name}")

    @property
    def identity(self) -> str:
        return content_id(dataclasses.asdict(self))


def source_worlds() -> tuple[np.ndarray, np.ndarray]:
    return (100000.0 * np.repeat([0.2, 1.0, 5.0], 2), np.tile([0.20, 0.40], 3))


def task_actions() -> np.ndarray:
    values = [[0.0, 0.0, 0.0]]
    for y in (-0.06, 0.0, 0.06):
        for z in (0.0, 0.05, 0.10, 0.15):
            if y != 0 or z != 0:
                values.append([0.0, y, z])
    return np.asarray(values, dtype=np.float64)


def task_goals() -> np.ndarray:
    return np.asarray(
        [[x, 0.0, z] for x in (0.35, 0.50, 0.60) for z in (0.35, 0.45, 0.55)]
    )


def qualification_protocol() -> dict[str, Any]:
    bending, support_x = source_worlds()
    return {
        "schema": "dlolab-support-decision-qualification-v1",
        "config": dataclasses.asdict(SupportTaskConfig()),
        "bending_settings": bending.tolist(),
        "support_x_m": support_x.tolist(),
        "action_offsets_m": task_actions().tolist(),
        "goals_m": task_goals().tolist(),
        "replay_action_indices": [0, 11],
        "monolithic_action_index": 11,
        "loss": "last-50-frame mean squared tip-goal distance plus 0.02 squared root displacement",
        "unit": "fixed six-world source grid, not independent physical objects",
        "source_only": True,
        "method_comparison": False,
        "new_recordings": False,
        "protected_data_read": False,
        "automatic_evaluation_authorization": False,
        "geometry_gate": {
            "maximum_relative_length_error": 0.10,
            "maximum_root_error_m": 1e-10,
            "maximum_support_penetration_m": 0.003,
            "support_unchanged": True,
            "all_state_replays_byte_identical": True,
            "minimum_contact_world_count": 2,
            "geometric_contact_tolerance_m": 0.002,
        },
        "decision_gate": {
            "minimum_distinct_oracle_actions_per_goal": 2,
            "minimum_oracle_gain_over_best_world_blind_action_relative_to_hold": 0.10,
            "minimum_passing_goals_of_nine": 3,
            "minimum_absolute_oracle_gap_m2": 0.000025,
        },
    }


def support_positions(config: SupportTaskConfig, support_x: np.ndarray) -> np.ndarray:
    x = np.asarray(support_x, dtype=np.float64)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("invalid support-world vector")
    length = (config.rod.node_count - 1) * config.rod.interval_m
    if np.any(x <= 2 * config.rod.interval_m) or np.any(
        x >= length - config.rod.interval_m
    ):
        raise ValueError("support must lie inside the free rod span")
    positions = np.empty((len(x), config.support_nodes, 3), dtype=np.float64)
    positions[:, :, 0] = x[:, None]
    positions[:, :, 1] = (
        np.arange(config.support_nodes) - (config.support_nodes - 1) / 2
    ) * config.support_interval_m
    positions[:, :, 2] = config.support_height_m
    return positions


class NativeSupportRuntime(DloLabRuntime):
    """Reuse the qualified full-state/reset/control API with native ROD supports."""

    def __init__(
        self,
        upstream: Path,
        task: SupportTaskConfig,
        bending: np.ndarray,
        support_x: np.ndarray,
    ):
        support = support_positions(task, support_x)
        values = np.asarray(bending, dtype=np.float64)
        if (
            values.shape != (len(support),)
            or not np.isfinite(values).all()
            or np.any(values <= 0)
        ):
            raise ValueError("invalid native material bank")
        self.provenance = verify_upstream(upstream)
        if "genesis" in sys.modules:
            raise ValueError("Genesis must follow pinned-source validation")
        sys.path.insert(0, str(upstream.resolve()))
        import genesis as gs
        import torch

        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        gs.init(
            backend=gs.cpu,
            precision="64",
            seed=task.rod.seed,
            logging_level="error",
            theme="dumb",
        )
        self.gs = gs
        self.task = task
        self.config = task.rod
        self.batch_size = len(values)
        self.step_index = 0
        self.model_id = content_id(
            {
                "task": task.identity,
                "bending": array_digest(values),
                "support": array_digest(support),
            }
        )
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.config.dt_s, substeps=self.config.substeps, requires_grad=False
            ),
            rod_options=gs.options.RODOptions(
                gravity=(0.0, 0.0, -9.81),
                floor_height=-5.0,
                damping=self.config.damping,
                angular_damping=self.config.angular_damping,
                n_pbd_iters=self.config.constraint_iterations,
            ),
            show_viewer=False,
        )
        self.rod = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                E=self.config.bending_modulus,
                G=self.config.twisting_modulus,
                segment_mass=self.config.segment_mass_kg,
                segment_radius=self.config.segment_radius_m,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=self.config.node_count,
                interval=self.config.interval_m,
                axis="x",
                pos=(0.0, 0.0, self.config.height_m),
            ),
        )
        self.support = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                segment_radius=task.support_radius_m, segment_mass=1.0
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=task.support_nodes,
                interval=task.support_interval_m,
                axis="y",
                pos=tuple(support[0, 0]),
            ),
        )
        self.scene.build(n_envs=self.batch_size)
        self.rod.set_fixed_states(fixed_ids=[0, 1])
        self.support.set_fixed_states(fixed_ids=list(range(task.support_nodes)))
        self.rod.set_bending_stiffness(torch.as_tensor(values))
        self.support.set_pos(self.scene.sim.cur_substep_local, support)
        if not np.array_equal(
            self.rod.get_all_bending_stiffness_tc().detach().cpu().numpy(), values
        ):
            raise RuntimeError("native material binding failed")
        if not np.array_equal(self.support.get_all_verts(), support):
            raise RuntimeError("native support binding failed")
        self.initial_positions = self.positions()
        self.initial_support = support.copy()

    def support_unchanged(self) -> bool:
        return array_digest(self.support.get_all_verts()) == array_digest(
            self.initial_support
        )


def action_commands(
    config: SupportTaskConfig, initial: np.ndarray, index: int
) -> np.ndarray:
    actions = task_actions()
    if type(index) is not int or not 0 <= index < len(actions):
        raise ValueError("unknown action index")
    if (
        initial.ndim != 3
        or initial.shape[1:] != (config.rod.node_count, 3)
        or not np.isfinite(initial).all()
    ):
        raise ValueError("invalid action starting carrier")
    phase = np.linspace(0, 1, config.horizon_steps)
    ramp = 3 * phase**2 - 2 * phase**3
    return (
        initial[None, :, :2]
        + ramp[:, None, None, None] * actions[index][None, None, None]
    )


def segment_distance(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> np.ndarray:
    """Exact finite-segment distance via interior stationary point and boundaries."""
    a0, a1, b0, b1 = np.broadcast_arrays(a0, a1, b0, b1)
    if a0.shape[-1] != 3 or not all(np.isfinite(x).all() for x in (a0, a1, b0, b1)):
        raise ValueError("invalid geometry")
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    uu, vv, uv = np.sum(u * u, axis=-1), np.sum(v * v, axis=-1), np.sum(u * v, axis=-1)
    uw, vw = np.sum(u * w, axis=-1), np.sum(v * w, axis=-1)
    if np.any(uu <= 0) or np.any(vv <= 0):
        raise ValueError("zero-length segment")
    candidates = []
    for endpoint in (a0, a1):
        t = np.clip(np.sum((endpoint - b0) * v, axis=-1) / vv, 0, 1)
        candidates.append(np.sum((endpoint - b0 - t[..., None] * v) ** 2, axis=-1))
    for endpoint in (b0, b1):
        t = np.clip(np.sum((endpoint - a0) * u, axis=-1) / uu, 0, 1)
        candidates.append(np.sum((endpoint - a0 - t[..., None] * u) ** 2, axis=-1))
    denominator = uu * vv - uv**2
    nonparallel = denominator > 1e-12 * uu * vv
    safe = np.where(nonparallel, denominator, 1.0)
    s, t = (uv * vw - vv * uw) / safe, (uu * vw - uv * uw) / safe
    interior = nonparallel & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    square = np.sum((w + s[..., None] * u - t[..., None] * v) ** 2, axis=-1)
    candidates.append(np.where(interior, square, np.inf))
    return np.sqrt(np.min(candidates, axis=0))


def contact_clearance(
    trajectory: np.ndarray, support: np.ndarray, config: SupportTaskConfig
) -> np.ndarray:
    if trajectory.ndim != 4 or trajectory.shape[1:] != (
        len(support),
        config.rod.node_count,
        3,
    ):
        raise ValueError("trajectory must be time-by-world-by-node")
    distances = segment_distance(
        trajectory[:, :, :-1, None],
        trajectory[:, :, 1:, None],
        support[None, :, None, :-1],
        support[None, :, None, 1:],
    )
    return (
        distances.min(axis=(2, 3))
        - config.rod.segment_radius_m
        - config.support_radius_m
    )


def source_task_losses(futures: np.ndarray, config: SupportTaskConfig) -> np.ndarray:
    expected = (12, config.horizon_steps, 6, config.rod.node_count, 3)
    if futures.shape != expected or not np.isfinite(futures).all():
        raise ValueError("complete twelve-action, six-world native forecasts required")
    tip = futures[:, -config.loss_tail_steps :, :, -1]
    difference = tip[None] - task_goals()[:, None, None, None]
    losses = np.sum(difference**2, axis=-1).mean(axis=2).transpose(0, 2, 1)
    return (
        losses + config.effort_weight * np.sum(task_actions() ** 2, axis=1)[None, None]
    )


def sensitivity_gate(losses: np.ndarray) -> dict[str, Any]:
    if (
        losses.shape != (9, 6, 12)
        or not np.isfinite(losses).all()
        or np.any(losses < 0)
    ):
        raise ValueError("complete fixed goal-by-world-by-action loss cube required")
    reports = []
    for index, table in enumerate(losses):
        hold_loss = float(table[:, 0].mean())
        if hold_loss <= 0:
            raise ValueError("zero hold loss makes the relative gate undefined")
        oracle_actions = np.argmin(table, axis=1)
        oracle = float(np.min(table, axis=1).mean())
        blind = float(np.min(table.mean(axis=0)))
        gap = blind - oracle
        count = len(np.unique(oracle_actions))
        passed = count >= 2 and gap >= 0.10 * hold_loss and gap >= 0.000025
        reports.append(
            {
                "goal_index": index,
                "goal_m": task_goals()[index].tolist(),
                "oracle_action_indices": oracle_actions.tolist(),
                "distinct_oracle_actions": count,
                "hold_loss_m2": hold_loss,
                "best_world_blind_loss_m2": blind,
                "oracle_loss_m2": oracle,
                "value_of_world_information_m2": gap,
                "value_relative_to_hold": gap / hold_loss,
                "passed": bool(passed),
            }
        )
    passing = sum(value["passed"] for value in reports)
    return {
        "goals": reports,
        "passing_goals": passing,
        "decision_sensitivity_passed": passing >= 3,
        "method_comparison": False,
        "automatic_evaluation_authorization": False,
    }
