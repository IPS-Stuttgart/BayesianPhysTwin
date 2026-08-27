"""Contracts for an unchanged public DLO-Lab task, not a new physics model."""

from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .deform_state_restart import array_digest, file_digest
from .dlolab_native import STATE_FIELDS, verify_upstream

ASSET_SHA256 = "acd483e232f1bb1fbf34078b154825fab3d2ee63b0aa4efc253c4411b368e421"
MUSHROOM_REVISION = "ec3364740627da945b8bab6e01d8151edb0f83f1"
RIGID_FIELDS = (
    "qpos",
    "dofs_vel",
    "dofs_acc",
    "links_pos",
    "links_quat",
    "i_pos_shift",
    "mass_shift",
    "friction_ratio",
)


def slingshot_actions() -> np.ndarray:
    actions = np.zeros((2, 3, 6), dtype=np.float64)
    actions[1, :, 1] = -0.04
    return actions


def protocol() -> dict[str, Any]:
    return {
        "schema": "dlolab-native-slingshot-qualification-v1",
        "task": "slingshot",
        "n_envs": 1,
        "actions": slingshot_actions().tolist(),
        "execution_order": [0, 1, 1],
        "native_steps_per_rollout": 900,
        "controller_substeps": 10,
        "native_env_reward_unchanged": True,
        "native_physics_unchanged": True,
        "runtime_override": "CPU float64, single Torch/BLAS thread, headless OSMesa",
        "minimum_gripper_motion_m": 0.01,
        "minimum_band_motion_m": 0.01,
        "maximum_replay_position_error_m": 1e-6,
        "maximum_replay_memory_absolute_error": 1e-9,
        "maximum_replay_memory_relative_error": 1e-6,
        "maximum_fixed_endpoint_error_m": 1e-9,
        "method_comparison": False,
        "automatic_method_evaluation_authorized": False,
        "new_recordings": False,
        "protected_data_read": False,
        "source_only": True,
    }


def source_identity(upstream: Path, mushroom: Path, archive: Path) -> dict[str, Any]:
    source = verify_upstream(upstream)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=mushroom, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=mushroom,
        text=True,
    ).strip()
    if revision != MUSHROOM_REVISION or dirty:
        raise ValueError("Mushroom-RL source changed")
    if file_digest(archive) != ASSET_SHA256:
        raise ValueError("official asset archive changed")
    assets = {}
    with zipfile.ZipFile(archive) as data:
        for member in data.infolist():
            if member.is_dir():
                continue
            path = Path(member.filename)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "dlo-lab":
                raise ValueError("noncanonical native asset path")
            target = upstream / "genesis/assets" / path
            digest = hashlib.sha256(data.read(member)).hexdigest()
            if target.is_symlink() or file_digest(target) != digest:
                raise ValueError("installed official asset differs from archive")
            assets[member.filename] = digest
    names = subprocess.check_output(
        ["git", "ls-files", "*.py", "pyproject.toml"], cwd=mushroom, text=True
    ).splitlines()
    return {
        "dlolab": source,
        "mushroom_revision": revision,
        "mushroom_source_sha256": {n: file_digest(mushroom / n) for n in names},
        "asset_archive_sha256": ASSET_SHA256,
        "asset_sha256": assets,
        "robot_asset_sha256": {
            str(p.relative_to(upstream)): file_digest(p)
            for parent in ("panda_bullet", "plane")
            for p in sorted((upstream / "genesis/assets/urdf" / parent).rglob("*"))
            if p.is_file()
        },
    }


def native_memory(state: Any) -> dict[str, np.ndarray]:
    arrays = {}
    active = [x for x in state.solvers_state if x is not None]
    expected = {"RigidSolverState": RIGID_FIELDS, "RODSolverState": STATE_FIELDS}
    if {type(x).__name__ for x in active} != set(expected) or len(active) != 2:
        raise ValueError("exactly the native rigid and rod solvers are required")
    for solver in active:
        kind = type(solver).__name__
        for field in expected[kind]:
            value = getattr(solver, field)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.array(value, order="C", copy=True)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite native memory")
            arrays[f"{kind}.{field}"] = array
    return arrays


def memory_comparison(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray]
) -> dict[str, Any]:
    if set(a) != set(b):
        raise ValueError("native memory fields changed")
    if any(a[k].shape != b[k].shape or a[k].dtype != b[k].dtype for k in a):
        raise ValueError("native memory layout changed")
    return {
        "byte_identical": all(array_digest(a[k]) == array_digest(b[k]) for k in a),
        "within_tolerance": all(
            np.allclose(a[k], b[k], rtol=1e-6, atol=1e-9) for k in a
        ),
        "maximum_absolute_difference": max(
            float(np.max(np.abs(a[k] - b[k]))) for k in a if a[k].size
        ),
        "field_count": len(a),
    }


def fixed_endpoint_error(traces: list[np.ndarray]) -> float:
    if not traces or any(
        x.ndim != 4 or x.shape[1:] != (1, 12, 3) or not len(x) for x in traces
    ):
        raise ValueError(
            "native slingshot traces must be time-by-one-world-by-12-nodes"
        )
    reference = np.take(traces[0][:1], [0, 1, 10, 11], axis=2)
    return max(
        float(np.max(np.abs(np.take(x, [0, 1, 10, 11], axis=2) - reference)))
        for x in traces
    )
