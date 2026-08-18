#!/usr/bin/env python3
"""Run the pinned synthetic native Genesis MPM backend smoke for issue #664."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.genesis_mpm_replay_v1 import GenesisMPMEntityReplayV1
from bayesian_phystwin.material_trajectory_producer_v1 import (
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive

GENESIS_REVISION = "0796d27667087d0087fe09d903f8aadf7fa9adeb"
GENESIS_VERSION = "1.3.3"
GENESIS_REPOSITORY = "https://github.com/Genesis-Embodied-AI/genesis-world"
GENESIS_SOURCE_BLOBS = {
    "__init__.py": "6313cf06d94a8203ecc77810eea5121bbeae9d99",
    "engine/entities/mpm_entity.py": "f700601b4abb37985d4b256d54661dbd6dc1f525",
    "engine/solvers/mpm_solver.py": "4cf9df95858d5af114ed428d4bf302b81b4daceb",
    "engine/materials/MPM/elastic.py": "98ad7b8e0f19aadb1bfaf6b3ec4bb98a94fefc39",
    "options/solvers.py": "0ef3c50de61ae3754384329949ea3a0f6a077916",
}
SMOKE_SCHEMA = "bayesian-phystwin.genesis-mpm-native-smoke-v1"
PORTABLE_MEMBERS = (
    "SHA256SUMS",
    "material-trajectory-backend.json",
    "physical-prediction.npz",
    "provenance/material-trajectory-rollout.npz",
    "provenance/material-runtime.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()


def _content_id(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _repository_revision(script_path: Path) -> str:
    repository_root = script_path.parents[2]
    completed = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip().lower()
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None:
        raise RuntimeError("unable to resolve an exact BayesianPhysTwin revision")
    return revision


@dataclass(frozen=True, slots=True)
class _NativeModules:
    gs: Any
    torch: Any
    package_root: Path
    source_records: Mapping[str, Mapping[str, str]]


def _load_native_modules() -> _NativeModules:
    installed_version = importlib.metadata.version("genesis-world")
    if installed_version != GENESIS_VERSION:
        raise RuntimeError(
            f"genesis-world version mismatch: expected {GENESIS_VERSION}, "
            f"found {installed_version}"
        )

    gs = importlib.import_module("genesis")
    torch = importlib.import_module("torch")
    reported_version = str(getattr(gs, "__version__", ""))
    if reported_version and reported_version != GENESIS_VERSION:
        raise RuntimeError(
            f"Genesis reported version mismatch: expected {GENESIS_VERSION}, "
            f"found {reported_version}"
        )
    package_file = getattr(gs, "__file__", None)
    if not package_file:
        raise RuntimeError("genesis package does not expose __file__")
    package_root = Path(package_file).resolve().parent

    source_records: dict[str, Mapping[str, str]] = {}
    for relative_path, expected_blob in GENESIS_SOURCE_BLOBS.items():
        path = package_root / relative_path
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Genesis source path is not an ordinary file: {path}")
        observed_blob = _git_blob_sha1(path)
        if observed_blob != expected_blob:
            raise RuntimeError(
                f"Genesis source mismatch for {relative_path}: expected Git blob "
                f"{expected_blob}, found {observed_blob}"
            )
        source_records[f"genesis/{relative_path}"] = {
            "git_blob_sha1": observed_blob,
            "sha256": _sha256_file(path),
        }
    return _NativeModules(
        gs=gs,
        torch=torch,
        package_root=package_root,
        source_records=source_records,
    )


def _build_replay(
    native: _NativeModules,
    *,
    seed: int,
    time_step_s: float,
    substeps: int,
    grid_density: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    density_kg_m3: float,
) -> GenesisMPMEntityReplayV1:
    gs = native.gs
    gs.set_random_seed(seed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=time_step_s,
            substeps=substeps,
            gravity=(0.0, 0.0, 0.0),
        ),
        mpm_options=gs.options.MPMOptions(
            grid_density=grid_density,
            lower_bound=(-0.25, -0.25, 0.0),
            upper_bound=(0.25, 0.25, 0.5),
        ),
        show_viewer=False,
        show_FPS=False,
    )
    entity = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=young_modulus_pa,
            nu=poisson_ratio,
            rho=density_kg_m3,
            model="corotation",
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.2),
            size=(0.08, 0.08, 0.08),
        ),
    )
    scene.build()
    return GenesisMPMEntityReplayV1(scene=scene, entity=entity, context=entity)


def _set_uniform_x_velocity(
    native: _NativeModules,
    replay: GenesisMPMEntityReplayV1,
    velocity_m_s: float,
) -> None:
    entity = replay.entity
    current = entity.get_particles_pos()
    velocity = native.torch.zeros_like(current)
    velocity[..., 0] = velocity_m_s
    entity.set_particles_vel(velocity)


def _portable_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for member in PORTABLE_MEMBERS:
        path = directory / member
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing portable member: {member}")
        hashes[member] = _sha256_file(path)
    return hashes


def _run_once(
    native: _NativeModules,
    output_dir: Path,
    *,
    frame_count: int,
    time_step_s: float,
    substeps: int,
    grid_density: int,
    velocity_m_s: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    density_kg_m3: float,
    seed: int,
    script_path: Path,
    adapter_path: Path,
    producer_revision: str,
) -> Mapping[str, Any]:
    gs = native.gs
    gs.init(
        backend=gs.cpu,
        precision="64",
        seed=seed,
        use_deterministic_algorithms=True,
    )
    try:

        def replay_factory() -> GenesisMPMEntityReplayV1:
            return _build_replay(
                native,
                seed=seed,
                time_step_s=time_step_s,
                substeps=substeps,
                grid_density=grid_density,
                young_modulus_pa=young_modulus_pa,
                poisson_ratio=poisson_ratio,
                density_kg_m3=density_kg_m3,
            )

        def driven_control(
            _: int,
            replay: GenesisMPMEntityReplayV1,
        ) -> None:
            _set_uniform_x_velocity(native, replay, velocity_m_s)

        def zero_control(
            _: int,
            replay: GenesisMPMEntityReplayV1,
        ) -> None:
            _set_uniform_x_velocity(native, replay, 0.0)

        scene_spec = {
            "morph": "box",
            "box_position_m": [0.0, 0.0, 0.2],
            "box_size_m": [0.08, 0.08, 0.08],
            "mpm_bounds_m": [[-0.25, -0.25, 0.0], [0.25, 0.25, 0.5]],
            "grid_density": grid_density,
            "seed": seed,
        }
        source_artifacts = {
            "scripts/remote/run_genesis_mpm_native_smoke.py": _sha256_file(script_path),
            "src/bayesian_phystwin/genesis_mpm_replay_v1.py": _sha256_file(
                adapter_path
            ),
        }
        source_artifacts.update(
            {
                f"native/{path}": record["sha256"]
                for path, record in native.source_records.items()
            }
        )
        artifact = produce_material_trajectory_backend(
            output_dir=output_dir,
            backend_kind="genesis-mpm-v1",
            replay_factory=replay_factory,
            driven_control=driven_control,
            zero_action_control=zero_control,
            frame_count=frame_count,
            material_query_indices=np.array([0], dtype=np.int64),
            action_support=np.array([1.0], dtype=np.float64),
            engine_revision=GENESIS_REVISION,
            engine_version=GENESIS_VERSION,
            producer_repository="IPS-Stuttgart/BayesianPhysTwin",
            producer_revision=producer_revision,
            producer_version="genesis-mpm-native-smoke-v1",
            producer_artifacts=source_artifacts,
            topology_sha256=_content_id(scene_spec),
            device="cpu",
            device_name=str(gs.device),
            time_step_s=time_step_s,
            scene_id="synthetic-zero-gravity-elastic-box-v1",
            model_kind="mpm-elastic-box",
            constitutive_model="Genesis MPM Elastic corotation",
            integrator="Genesis native MPM time stepping",
            solver="Genesis native MPM solver",
            substeps=substeps,
            engine_parameters={
                **scene_spec,
                "gravity_m_s2": [0.0, 0.0, 0.0],
                "young_modulus_pa": young_modulus_pa,
                "poisson_ratio": poisson_ratio,
                "density_kg_m3": density_kg_m3,
                "driven_uniform_x_velocity_m_s": velocity_m_s,
                "precision": "64",
                "deterministic_algorithms": True,
            },
        )
        arrays = load_physical_rollout_archive(
            output_dir / "physical-prediction.npz",
            expected_frame_count=frame_count,
        )
        zero_delta = (
            arrays["zero_action_readout_m"] - arrays["frame_zero_points_m"][None]
        )
        response = arrays["driven_readout_m"] - arrays["zero_action_readout_m"]
        return {
            "artifact_id": artifact["artifact_id"],
            "runtime_id": artifact["runtime_id"],
            "maximum_zero_action_drift_m": float(
                np.max(np.linalg.norm(zero_delta, axis=-1))
            ),
            "maximum_driven_minus_zero_response_m": float(
                np.max(np.linalg.norm(response, axis=-1))
            ),
            "portable_sha256": _portable_hashes(output_dir),
        }
    finally:
        gs.destroy()


def run_smoke(
    output_dir: str | Path,
    *,
    frame_count: int = 5,
    time_step_s: float = 0.004,
    substeps: int = 10,
    grid_density: int = 64,
    velocity_m_s: float = 0.05,
    young_modulus_pa: float = 100_000.0,
    poisson_ratio: float = 0.3,
    density_kg_m3: float = 1000.0,
    seed: int = 0,
) -> Mapping[str, Any]:
    if (
        isinstance(frame_count, (bool, np.bool_))
        or not isinstance(frame_count, (int, np.integer))
        or int(frame_count) < 2
    ):
        raise ValueError("frame_count must be an integer >= 2")
    frame_count = int(frame_count)
    if time_step_s <= 0.0 or not np.isfinite(time_step_s):
        raise ValueError("time_step_s must be finite and positive")
    if (
        isinstance(substeps, (bool, np.bool_))
        or not isinstance(substeps, (int, np.integer))
        or int(substeps) < 1
    ):
        raise ValueError("substeps must be an integer >= 1")
    substeps = int(substeps)
    if (
        isinstance(grid_density, (bool, np.bool_))
        or not isinstance(grid_density, (int, np.integer))
        or int(grid_density) < 2
    ):
        raise ValueError("grid_density must be an integer >= 2")
    grid_density = int(grid_density)
    if velocity_m_s <= 0.0 or not np.isfinite(velocity_m_s):
        raise ValueError("velocity_m_s must be finite and positive")
    if young_modulus_pa <= 0.0 or not np.isfinite(young_modulus_pa):
        raise ValueError("young_modulus_pa must be finite and positive")
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1,0.5)")
    if density_kg_m3 <= 0.0 or not np.isfinite(density_kg_m3):
        raise ValueError("density_kg_m3 must be finite and positive")
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)

    root = Path(output_dir).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    native = _load_native_modules()
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]
    adapter_path = (
        repository_root / "src" / "bayesian_phystwin" / "genesis_mpm_replay_v1.py"
    )
    if not adapter_path.is_file() or adapter_path.is_symlink():
        raise RuntimeError("Genesis replay adapter source is unavailable")
    producer_revision = _repository_revision(script_path)

    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.staging.",
        dir=root.parent,
    ) as temporary:
        staging = Path(temporary) / root.name
        staging.mkdir()
        kwargs = {
            "frame_count": frame_count,
            "time_step_s": time_step_s,
            "substeps": substeps,
            "grid_density": grid_density,
            "velocity_m_s": velocity_m_s,
            "young_modulus_pa": young_modulus_pa,
            "poisson_ratio": poisson_ratio,
            "density_kg_m3": density_kg_m3,
            "seed": seed,
            "script_path": script_path,
            "adapter_path": adapter_path,
            "producer_revision": producer_revision,
        }
        run_a = _run_once(native, staging / "run-a", **kwargs)
        run_b = _run_once(native, staging / "run-b", **kwargs)
        deterministic = run_a["portable_sha256"] == run_b["portable_sha256"]
        if not deterministic:
            raise RuntimeError("native Genesis MPM replay is not byte-deterministic")

        response = float(run_a["maximum_driven_minus_zero_response_m"])
        zero_drift = float(run_a["maximum_zero_action_drift_m"])
        minimum_response = velocity_m_s * time_step_s * 0.1
        if response <= minimum_response:
            raise RuntimeError("Genesis MPM driven arm did not produce a response")
        if zero_drift > 1e-8:
            raise RuntimeError(
                "Genesis MPM zero-action drift exceeds the synthetic smoke bound"
            )

        descriptor: dict[str, Any] = {
            "schema": SMOKE_SCHEMA,
            "claim_boundary": (
                "Synthetic native-execution and provenance smoke only; no "
                "source-value, fresh-object, calibration, or Causal4D benefit claim."
            ),
            "backend_profile": "genesis-mpm-v1",
            "engine": {
                "repository": GENESIS_REPOSITORY,
                "revision": GENESIS_REVISION,
                "version": GENESIS_VERSION,
                "source_records": native.source_records,
            },
            "producer_revision": producer_revision,
            "runtime": {
                "python_version": platform.python_version(),
                "torch_version": str(getattr(native.torch, "__version__", "unknown")),
                "backend": "cpu",
                "precision": "64",
                "deterministic_algorithms": True,
            },
            "problem": {
                "frame_count": frame_count,
                "time_step_s": time_step_s,
                "substeps": substeps,
                "grid_density": grid_density,
                "uniform_x_velocity_m_s": velocity_m_s,
                "young_modulus_pa": young_modulus_pa,
                "poisson_ratio": poisson_ratio,
                "density_kg_m3": density_kg_m3,
                "gravity_m_s2": [0.0, 0.0, 0.0],
                "seed": seed,
            },
            "checks": {
                "installed_source_matches_pinned_git_blobs": True,
                "portable_replay_byte_deterministic": deterministic,
                "maximum_zero_action_drift_m": zero_drift,
                "maximum_driven_minus_zero_response_m": response,
                "minimum_required_response_m": minimum_response,
            },
            "run_a": run_a,
            "run_b": run_b,
            "future_outcomes_read": False,
            "dataset_payload_read": False,
        }
        descriptor["smoke_id"] = _content_id(descriptor)
        result_path = staging / "genesis-mpm-native-smoke.json"
        result_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, root)
    return descriptor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--time-step-s", type=float, default=0.004)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--grid-density", type=int, default=64)
    parser.add_argument("--velocity-m-s", type=float, default=0.05)
    parser.add_argument("--young-modulus-pa", type=float, default=100_000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--density-kg-m3", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_smoke(
        args.output_dir,
        frame_count=args.frame_count,
        time_step_s=args.time_step_s,
        substeps=args.substeps,
        grid_density=args.grid_density,
        velocity_m_s=args.velocity_m_s,
        young_modulus_pa=args.young_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        density_kg_m3=args.density_kg_m3,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
