"""Optional Genesis runtime for the synthetic elastic-MPM compatibility smoke."""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import genesis as gs
import numpy as np
import torch

from ._portable_contracts import content_id, write_atomic_json
from .genesis_mpm_backend_v1 import (
    GENESIS_MPM_BACKEND_KIND,
    GENESIS_MPM_ENGINE_REPOSITORY,
    GENESIS_MPM_RUNTIME_SCHEMA,
    GENESIS_MPM_SCHEMA_VERSION,
    file_sha256,
    validate_genesis_mpm_runtime_manifest,
)
from .physical_rollout_v1 import write_deterministic_npz
from .phystwin_online_belief import deterministic_farthest_point_ids

_IMPLEMENTATION_PATHS = (
    "src/bayesian_phystwin/_genesis_mpm_runtime.py",
    "src/bayesian_phystwin/genesis_mpm_backend_v1.py",
    "src/bayesian_phystwin/physical_rollout_v1.py",
)


@dataclass(frozen=True, slots=True)
class GenesisMpmSmokeConfig:
    """Frozen parameters for a small elastic beam with compliant attachments."""

    frame_count: int = 40
    query_count: int = 64
    fps: float = 120.0
    substeps: int = 32
    grid_density: int = 64
    density_kg_m3: float = 1000.0
    young_modulus_pa: float = 50_000.0
    poisson_ratio: float = 0.30
    attachment_stiffness: float = 500.0
    beam_length_m: float = 0.30
    beam_width_m: float = 0.05
    beam_height_m: float = 0.05
    action_displacement_m: float = 0.010
    elastic_model: str = "corotation"
    seed: int = 260811

    def validate(self) -> None:
        if self.frame_count < 2:
            raise ValueError("frame_count must be at least two")
        if self.query_count < 1:
            raise ValueError("query_count must be positive")
        if self.fps <= 0.0 or not np.isfinite(self.fps):
            raise ValueError("fps must be finite and positive")
        if self.substeps < 1:
            raise ValueError("substeps must be positive")
        if self.grid_density < 8:
            raise ValueError("grid_density must be at least eight")
        positive = {
            "density_kg_m3": self.density_kg_m3,
            "young_modulus_pa": self.young_modulus_pa,
            "attachment_stiffness": self.attachment_stiffness,
            "beam_length_m": self.beam_length_m,
            "beam_width_m": self.beam_width_m,
            "beam_height_m": self.beam_height_m,
            "action_displacement_m": self.action_displacement_m,
        }
        for name, value in positive.items():
            if value <= 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must be in [0, 0.5)")
        if self.elastic_model not in {"corotation", "neohooken"}:
            raise ValueError("elastic_model must be corotation or neohooken")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def _positions_numpy(entity: Any) -> np.ndarray:
    value = entity.get_particles_pos()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value.numpy() if hasattr(value, "numpy") else value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 3:
        raise RuntimeError("Genesis returned an unexpected particle-position shape")
    return np.ascontiguousarray(array, dtype=np.float32)


def _backend_value(backend: str) -> Any:
    if backend == "gpu":
        return gs.gpu
    if backend == "cpu":
        return gs.cpu
    raise ValueError("backend must be gpu or cpu")


def _implementation_record() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    revision = os.environ.get("BPT_IMPLEMENTATION_REVISION")
    if revision is None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Genesis smoke requires a Git revision or BPT_IMPLEMENTATION_REVISION"
            ) from error
        revision = completed.stdout.strip()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError("BPT implementation revision is not a lowercase Git SHA-1")
    source_hashes: dict[str, str] = {}
    for relative_path in _IMPLEMENTATION_PATHS:
        source_path = repository_root / relative_path
        if not source_path.is_file() or source_path.is_symlink():
            raise RuntimeError("BPT implementation source file is unavailable")
        source_hashes[relative_path] = file_sha256(source_path)
    return {
        "repository": "IPS-Stuttgart/BayesianPhysTwin",
        "revision": revision,
        "source_files_sha256": source_hashes,
    }


def _simulate_one(  # pragma: no cover - exercised by the native runtime smoke
    config: GenesisMpmSmokeConfig,
    *,
    backend: str,
    driven_action: bool,
) -> np.ndarray:
    gs.init(
        backend=_backend_value(backend),
        precision="32",
        logging_level=logging.WARNING,
        seed=config.seed,
    )
    scene: Any | None = None
    try:
        half_length = 0.5 * config.beam_length_m
        center_z = 0.30
        # Genesis contracts each requested MPM bound by several grid cells.
        # Keep the sampled beam and both attachment patches inside that
        # effective domain even in deliberately coarse smoke configurations.
        padding = max(0.12, 4.0 / config.grid_density)
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1.0 / config.fps,
                substeps=config.substeps,
                gravity=(0.0, 0.0, 0.0),
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=(-half_length - padding, -0.15, 0.10),
                upper_bound=(half_length + padding, 0.15, 0.55),
                grid_density=config.grid_density,
            ),
            show_viewer=False,
        )
        gripper_size = (0.04, 0.08, 0.08)
        left = scene.add_entity(
            gs.morphs.Box(
                pos=(-half_length, 0.0, center_z),
                size=gripper_size,
                fixed=True,
            )
        )
        right = scene.add_entity(
            gs.morphs.Box(
                pos=(half_length, 0.0, center_z),
                size=gripper_size,
                fixed=False,
            )
        )
        beam = scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=config.young_modulus_pa,
                nu=config.poisson_ratio,
                rho=config.density_kg_m3,
                model=config.elastic_model,
            ),
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, center_z),
                size=(
                    config.beam_length_m,
                    config.beam_width_m,
                    config.beam_height_m,
                ),
            ),
        )
        scene.build()
        attachment_width = max(2.0 / config.grid_density, 0.0125)
        y_half = 0.6 * config.beam_width_m
        z_half = 0.6 * config.beam_height_m
        left_mask = beam.get_particles_in_bbox(
            (-half_length - attachment_width, -y_half, center_z - z_half),
            (-half_length + attachment_width, y_half, center_z + z_half),
        )
        right_mask = beam.get_particles_in_bbox(
            (half_length - attachment_width, -y_half, center_z - z_half),
            (half_length + attachment_width, y_half, center_z + z_half),
        )
        if not bool(left_mask.any()) or not bool(right_mask.any()):
            raise RuntimeError(
                "Genesis beam discretization did not create attachment particles"
            )
        beam.set_particle_constraints(
            left_mask,
            left.links[0].idx,
            stiffness=config.attachment_stiffness,
        )
        beam.set_particle_constraints(
            right_mask,
            right.links[0].idx,
            stiffness=config.attachment_stiffness,
        )

        trajectory = np.empty(
            (config.frame_count, beam.n_particles, 3), dtype=np.float32
        )
        trajectory[0] = _positions_numpy(beam)
        initial_qpos = torch.tensor(
            [half_length, 0.0, center_z, 1.0, 0.0, 0.0, 0.0],
            device=gs.device,
            dtype=torch.float32,
        )
        for frame_index in range(1, config.frame_count):
            alpha = frame_index / (config.frame_count - 1)
            displacement = (
                config.action_displacement_m * alpha if driven_action else 0.0
            )
            target_qpos = initial_qpos.clone()
            target_qpos[2] += displacement
            right.set_qpos(target_qpos)
            scene.step()
            trajectory[frame_index] = _positions_numpy(beam)
        if not np.all(np.isfinite(trajectory)):
            raise RuntimeError("Genesis MPM generated non-finite particle positions")
        return trajectory
    finally:
        if scene is not None:
            scene.destroy()
        gs.destroy()


def run_genesis_mpm_smoke(
    *,
    raw_rollout_path: str | Path,
    runtime_manifest_path: str | Path,
    backend: str = "gpu",
    config: GenesisMpmSmokeConfig | None = None,
) -> dict[str, Any]:
    """Run driven and zero-action beams and seal a raw particle artifact."""

    if config is None:
        smoke = GenesisMpmSmokeConfig()
    elif isinstance(config, GenesisMpmSmokeConfig):
        smoke = config
    else:
        raise TypeError("config must be a GenesisMpmSmokeConfig")
    smoke.validate()
    _backend_value(backend)
    raw_path = Path(raw_rollout_path)
    runtime_path = Path(runtime_manifest_path)
    if raw_path.exists() or runtime_path.exists():
        raise FileExistsError("Genesis smoke output already exists")

    driven = _simulate_one(smoke, backend=backend, driven_action=True)
    zero = _simulate_one(smoke, backend=backend, driven_action=False)
    if not np.array_equal(driven[0], zero[0]):
        raise RuntimeError("driven and zero-action Genesis runs differ at frame zero")
    query_count = min(smoke.query_count, driven.shape[1])
    query_indices = deterministic_farthest_point_ids(
        driven[0], np.arange(driven.shape[1], dtype=np.int64), query_count
    )
    query_response = np.linalg.norm(
        driven[:, query_indices] - zero[:, query_indices], axis=2
    )
    maximum_query_response = np.max(query_response, axis=0)
    query_normalization = float(np.max(maximum_query_response))
    if not np.isfinite(query_normalization) or query_normalization <= 0.0:
        raise RuntimeError(
            "driven Genesis smoke produced no action-conditioned response"
        )
    maximum_response = float(np.max(np.linalg.norm(driven - zero, axis=2)))
    response_ratio = maximum_response / smoke.action_displacement_m
    stability_cap_ratio = 3.0
    if response_ratio > stability_cap_ratio:
        raise RuntimeError("Genesis MPM response exceeded the frozen stability cap")
    maximum_particle_step = float(
        np.max(np.linalg.norm(np.diff(driven, axis=0), axis=2))
    )
    if not np.isfinite(maximum_particle_step) or maximum_particle_step <= 0.0:
        raise RuntimeError("driven Genesis smoke produced no finite particle motion")
    action_support = np.asarray(
        maximum_query_response / query_normalization, dtype=np.float32
    )
    raw_arrays = {
        "driven_particle_positions_m": np.ascontiguousarray(driven),
        "zero_action_particle_positions_m": np.ascontiguousarray(zero),
        "material_query_indices": np.ascontiguousarray(query_indices),
        "action_support": np.ascontiguousarray(action_support),
    }
    write_deterministic_npz(raw_path, raw_arrays)

    device_name = (
        torch.cuda.get_device_name(0)
        if backend == "gpu" and torch.cuda.is_available()
        else platform.processor() or "cpu"
    )
    simulation = {
        "scene": "compliant-gripper-beam-bend-v1",
        "beam_extents_m": [
            smoke.beam_length_m,
            smoke.beam_width_m,
            smoke.beam_height_m,
        ],
        "action_displacement_m": [0.0, 0.0, smoke.action_displacement_m],
        "gravity_m_s2": [0.0, 0.0, 0.0],
        "density_kg_m3": smoke.density_kg_m3,
        "young_modulus_pa": smoke.young_modulus_pa,
        "poisson_ratio": smoke.poisson_ratio,
        "elastic_model": smoke.elastic_model,
        "grid_density": smoke.grid_density,
        "substeps": smoke.substeps,
        "attachment_stiffness": smoke.attachment_stiffness,
        "solver": "genesis-mpm",
    }
    boundary = {
        "synthetic_scene": True,
        "dataset_payload_read": False,
        "future_observations_read": False,
        "outcomes_read": False,
        "known_action_used": True,
    }
    identity = {
        "schema": GENESIS_MPM_RUNTIME_SCHEMA,
        "schema_version": GENESIS_MPM_SCHEMA_VERSION,
        "backend_kind": GENESIS_MPM_BACKEND_KIND,
        "engine_repository": GENESIS_MPM_ENGINE_REPOSITORY,
        "engine_version": str(gs.__version__),
        "torch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "device": backend,
        "device_name": device_name,
        "coordinate_frame": "right-handed-z-up-world-v1",
        "position_units": "m",
        "time_units": "s",
        "frame_count": smoke.frame_count,
        "particle_count": int(driven.shape[1]),
        "query_count": int(len(query_indices)),
        "time_step_s": 1.0 / smoke.fps,
        "simulation": simulation,
        "diagnostics": {
            "maximum_action_response_m": maximum_response,
            "maximum_particle_step_m": maximum_particle_step,
            "response_to_action_ratio": response_ratio,
            "stability_cap_ratio": stability_cap_ratio,
            "stability_gate_passed": True,
        },
        "implementation": _implementation_record(),
        "information_boundary": boundary,
        "raw_rollout_sha256": file_sha256(raw_path),
    }
    runtime = {**identity, "runtime_id": content_id(identity)}
    write_atomic_json(runtime, runtime_path, overwrite=False)
    validate_genesis_mpm_runtime_manifest(runtime, raw_rollout_path=raw_path)
    return {
        "runtime": runtime,
        "config": asdict(smoke),
        "maximum_action_response_m": maximum_response,
    }
