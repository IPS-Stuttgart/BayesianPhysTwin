"""Isolated native DLO-Lab rod interface for synthetic decision experiments."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform_state_restart import array_digest, file_digest

UPSTREAM_REVISION = "c5026a9416b03c6bc5186eba13cd4ffd4c0e7796"
STATE_FIELDS = (
    "pos",
    "vel",
    "fixed",
    "theta",
    "omega",
    "edge",
    "length",
    "d1",
    "d2",
    "d3",
    "d1_ref",
    "d2_ref",
    "kb",
    "twist",
    "kappa_rest",
)


@dataclass(frozen=True)
class DloLabConfig:
    schema: str = "dlolab-procedural-native-config-v1"
    node_count: int = 16
    interval_m: float = 0.025
    height_m: float = 0.6
    dt_s: float = 0.002
    substeps: int = 10
    constraint_iterations: int = 10
    bending_modulus: float = 100000.0
    twisting_modulus: float = 10000.0
    segment_mass_kg: float = 0.002
    segment_radius_m: float = 0.003
    damping: float = 10.0
    angular_damping: float = 5.0
    seed: int = 260828

    def __post_init__(self) -> None:
        if self.schema != "dlolab-procedural-native-config-v1":
            raise ValueError("invalid DLO-Lab schema")
        for name in ("node_count", "substeps", "constraint_iterations", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"invalid integer field {name}")
        if self.node_count < 6:
            raise ValueError("at least six nodes are required")
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"invalid positive parameter {field.name}")

    @property
    def identity(self) -> str:
        data = json.dumps(
            dataclasses.asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(data).hexdigest()


def verify_upstream(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError("DLO-Lab upstream revision changed")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    ).strip():
        raise ValueError("DLO-Lab tracked source is dirty")
    names = subprocess.check_output(
        ["git", "ls-files", "*.py", "pyproject.toml", "LICENSE"],
        cwd=root,
        text=True,
    ).splitlines()
    hashes = {name: file_digest(root / name) for name in sorted(names)}
    return {"revision": revision, "source_sha256": hashes}


def clamp_only_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    clamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace only the two prescribed root positions; preserve free dynamics."""
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)
    clamps = np.asarray(clamps)
    if (
        positions.ndim != 3
        or positions.shape[-1] != 3
        or positions.shape[1] < 6
        or velocities.shape != positions.shape
        or clamps.shape != (positions.shape[0], 2, 3)
    ):
        raise ValueError("only batched two-clamp commands are accepted")
    if any(not np.isfinite(value).all() for value in (positions, velocities, clamps)):
        raise ValueError("state and clamp commands must be finite")
    result = np.array(positions, dtype=np.float64, order="C", copy=True)
    velocity = np.array(velocities, dtype=np.float64, order="C", copy=True)
    result[:, :2] = clamps
    velocity[:, :2] = 0.0
    return result, velocity


def native_state_arrays(native_state: Any) -> dict[str, np.ndarray]:
    active = [state for state in native_state.solvers_state if state is not None]
    if len(active) != 1 or not all(hasattr(active[0], name) for name in STATE_FIELDS):
        raise ValueError("this qualification supports exactly one native rod solver")
    result: dict[str, np.ndarray] = {}
    for name in STATE_FIELDS:
        value = getattr(active[0], name)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.array(value, copy=True, order="C")
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite native memory {name}")
        result[name] = array
    return result


def native_state_digests(native_state: Any) -> dict[str, str]:
    return {
        name: array_digest(value)
        for name, value in native_state_arrays(native_state).items()
    }


@dataclass(frozen=True)
class NativeSnapshot:
    step_index: int
    config_id: str
    native_state: Any
    field_digests: dict[str, str]

    def validate(self, config: DloLabConfig) -> None:
        if type(self.step_index) is not int or self.step_index < 0:
            raise ValueError("invalid snapshot index")
        if self.config_id != config.identity:
            raise ValueError("snapshot model configuration changed")
        if self.field_digests != native_state_digests(self.native_state):
            raise ValueError("native snapshot was mutated")


class DloLabRuntime:
    """Native CPU/float64 procedural rod, without learned weights or raw data."""

    def __init__(self, upstream: Path, config: DloLabConfig, *, batch_size: int = 1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("invalid batch size")
        self.provenance = verify_upstream(upstream)
        if "genesis" in sys.modules:
            raise ValueError("Genesis must not be imported before source validation")
        sys.path.insert(0, str(upstream.resolve()))
        import genesis as gs
        import torch

        if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 8):
            raise ValueError("pinned upstream requires Torch >= 2.8")
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        gs.init(
            backend=gs.cpu,
            precision="64",
            seed=config.seed,
            logging_level="error",
            theme="dumb",
        )
        self.gs = gs
        self.config = config
        self.batch_size = batch_size
        self.step_index = 0
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=config.dt_s,
                substeps=config.substeps,
                requires_grad=False,
            ),
            rod_options=gs.options.RODOptions(
                damping=config.damping,
                angular_damping=config.angular_damping,
                n_pbd_iters=config.constraint_iterations,
            ),
            show_viewer=False,
        )
        self.rod = self.scene.add_entity(
            material=gs.materials.ROD.Base(
                E=config.bending_modulus,
                G=config.twisting_modulus,
                segment_mass=config.segment_mass_kg,
                segment_radius=config.segment_radius_m,
            ),
            morph=gs.morphs.ParameterizedRod(
                type="rod",
                n_vertices=config.node_count,
                interval=config.interval_m,
                axis="x",
                pos=(0.0, 0.0, config.height_m),
            ),
        )
        self.scene.build(n_envs=batch_size)
        self.rod.set_fixed_states(fixed_ids=[0, 1])
        self.initial_positions = self.positions()

    def positions(self) -> np.ndarray:
        return np.asarray(self.rod.get_all_verts()).copy()

    def capture(self) -> NativeSnapshot:
        state = self.scene.get_state()
        return NativeSnapshot(
            self.step_index,
            self.config.identity,
            state,
            native_state_digests(state),
        )

    def restore(self, snapshot: NativeSnapshot) -> None:
        snapshot.validate(self.config)
        self.scene.reset(snapshot.native_state)
        self.step_index = snapshot.step_index
        if native_state_digests(self.scene.get_state()) != snapshot.field_digests:
            raise RuntimeError("native state restore is not byte-identical")

    def step(self, clamps: np.ndarray) -> np.ndarray:
        positions, velocities = clamp_only_state(
            self.positions(),
            self.rod.get_all_vels(),
            clamps,
        )
        self.rod.set_position(positions)
        self.rod.set_velocity(velocities)
        self.scene.step(update_visualizer=False)
        self.step_index += 1
        result = self.positions()
        if not np.isfinite(result).all():
            raise RuntimeError("native rod produced nonfinite positions")
        return result

    def rollout(self, commands: np.ndarray) -> np.ndarray:
        value = np.asarray(commands, dtype=np.float64)
        if (
            value.ndim != 4
            or value.shape[1:] != (self.batch_size, 2, 3)
            or not len(value)
        ):
            raise ValueError("commands must be (time, batch, two clamps, xyz)")
        if not np.isfinite(value).all():
            raise ValueError("commands must be finite")
        return np.stack([self.step(command) for command in value])

    def close(self) -> None:
        self.gs.destroy()
