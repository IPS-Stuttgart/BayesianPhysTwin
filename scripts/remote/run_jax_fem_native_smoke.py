#!/usr/bin/env python3
"""Run the pinned synthetic native JAX-FEM backend smoke for issue #664."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.jax_fem_producer_v1 import produce_jax_fem_backend
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive

JAX_FEM_REVISION = "82c6993c16704e38611f9cb91a5b70f1c690daee"
JAX_FEM_VERSION = "0.0.12"
JAX_FEM_REPOSITORY = "https://github.com/deepmodeling/jax-fem"
JAX_FEM_SOURCE_BLOBS = {
    "problem.py": "8a20d24fc2e98aa33d4bd76e543f00c471740551",
    "solver.py": "f0f64cb629e202f2d179710b745ea4d682f1ace2",
    "generate_mesh.py": "bd564c8f4a049ae28bc3592e21d9547a5f509629",
}
SMOKE_SCHEMA = "bayesian-phystwin.jax-fem-native-smoke-v1"
PORTABLE_MEMBERS = (
    "SHA256SUMS",
    "lagrangian-backend.json",
    "physical-prediction.npz",
    "provenance/lagrangian-rollout.npz",
    "provenance/lagrangian-runtime.json",
)

_POINTS_M = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.01],
        [0.0, 0.01, 0.0],
        [0.0, 0.01, 0.01],
        [0.04, 0.0, 0.0],
        [0.04, 0.0, 0.01],
        [0.04, 0.01, 0.0],
        [0.04, 0.01, 0.01],
    ],
    dtype=np.float64,
)
_CELLS = np.array([[0, 4, 6, 2, 1, 5, 7, 3]], dtype=np.int32)


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


@dataclass(frozen=True, slots=True)
class _NativeModules:
    jax: Any
    jnp: Any
    Mesh: Any
    Problem: Any
    solver: Any
    package_root: Path
    source_records: Mapping[str, Mapping[str, str]]


def _load_native_modules() -> _NativeModules:
    installed_version = importlib.metadata.version("jax-fem")
    if installed_version != JAX_FEM_VERSION:
        raise RuntimeError(
            f"jax-fem version mismatch: expected {JAX_FEM_VERSION}, "
            f"found {installed_version}"
        )

    jax = importlib.import_module("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = importlib.import_module("jax.numpy")
    jax_fem = importlib.import_module("jax_fem")
    generate_mesh = importlib.import_module("jax_fem.generate_mesh")
    problem_module = importlib.import_module("jax_fem.problem")
    solver_module = importlib.import_module("jax_fem.solver")

    package_file = getattr(jax_fem, "__file__", None)
    if not package_file:
        raise RuntimeError("jax_fem package does not expose __file__")
    package_root = Path(package_file).resolve().parent

    source_records: dict[str, Mapping[str, str]] = {}
    for name, expected_blob in JAX_FEM_SOURCE_BLOBS.items():
        path = package_root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"jax-fem source path is not an ordinary file: {path}")
        observed_blob = _git_blob_sha1(path)
        if observed_blob != expected_blob:
            raise RuntimeError(
                f"jax-fem source mismatch for {name}: expected Git blob "
                f"{expected_blob}, found {observed_blob}"
            )
        source_records[f"jax_fem/{name}"] = {
            "git_blob_sha1": observed_blob,
            "sha256": _sha256_file(path),
        }

    return _NativeModules(
        jax=jax,
        jnp=jnp,
        Mesh=generate_mesh.Mesh,
        Problem=problem_module.Problem,
        solver=solver_module.solver,
        package_root=package_root,
        source_records=source_records,
    )


class _NativeReplay:
    def __init__(
        self,
        native: _NativeModules,
        *,
        young_modulus_pa: float,
        poisson_ratio: float,
        maximum_displacement_m: float,
    ) -> None:
        self.native = native
        self.young_modulus_pa = float(young_modulus_pa)
        self.poisson_ratio = float(poisson_ratio)
        self.maximum_displacement_m = float(maximum_displacement_m)
        self.load_fraction = 0.0
        self.reference = native.jnp.asarray(_POINTS_M)
        self.mesh = native.Mesh(
            self.reference,
            native.jnp.asarray(_CELLS),
            ele_type="HEX8",
        )

    def get_reference_points_m(self) -> object:
        return self.reference

    def set_load_fraction(self, fraction: float) -> None:
        value = float(fraction)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("load fraction must be finite and in [0,1]")
        self.load_fraction = value

    def solve(self) -> object:
        jnp = self.native.jnp
        base_problem = self.native.Problem
        young = self.young_modulus_pa
        poisson = self.poisson_ratio
        target = self.maximum_displacement_m * self.load_fraction

        class LinearElasticity(base_problem):
            def get_tensor_map(self) -> Any:
                def stress(u_grad: Any) -> Any:
                    mu = young / (2.0 * (1.0 + poisson))
                    lmbda = (
                        young
                        * poisson
                        / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
                    )
                    epsilon = 0.5 * (u_grad + u_grad.T)
                    return (
                        lmbda * jnp.trace(epsilon) * jnp.eye(self.dim)
                        + 2.0 * mu * epsilon
                    )

                return stress

        def left(point: Any) -> Any:
            return jnp.isclose(point[0], 0.0, atol=1e-9)

        def right(point: Any) -> Any:
            return jnp.isclose(point[0], 0.04, atol=1e-9)

        def zero(_: Any) -> float:
            return 0.0

        def prescribed_x(_: Any) -> float:
            return target

        problem = LinearElasticity(
            mesh=self.mesh,
            vec=3,
            dim=3,
            ele_type="HEX8",
            dirichlet_bc_info=[
                [left, left, left, right],
                [0, 1, 2, 0],
                [zero, zero, zero, prescribed_x],
            ],
        )
        solution = self.native.solver(problem)
        if not isinstance(solution, (list, tuple)) or len(solution) != 1:
            raise RuntimeError("JAX-FEM solver returned an unexpected solution structure")
        return solution[0]


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
    maximum_displacement_m: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    script_path: Path,
) -> Mapping[str, Any]:
    def replay_factory() -> _NativeReplay:
        return _NativeReplay(
            native,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
            maximum_displacement_m=maximum_displacement_m,
        )

    def driven_control(index: int, replay: Any) -> None:
        replay.set_load_fraction((index + 1) / (frame_count - 1))

    def zero_control(_: int, replay: Any) -> None:
        replay.set_load_fraction(0.0)

    source_artifacts = {
        "scripts/remote/run_jax_fem_native_smoke.py": _sha256_file(script_path),
    }
    source_artifacts.update(
        {
            f"native/{path}": record["sha256"]
            for path, record in native.source_records.items()
        }
    )
    devices = native.jax.devices()
    if not devices:
        raise RuntimeError("JAX reports no execution device")
    primary = devices[0]
    device = (
        f"{getattr(primary, 'platform', 'unknown')}:"
        f"{getattr(primary, 'id', 'unknown')}:"
        f"{getattr(primary, 'device_kind', 'unknown')}"
    )

    artifact = produce_jax_fem_backend(
        output_dir=output_dir,
        replay_factory=replay_factory,
        driven_control=driven_control,
        zero_action_control=zero_control,
        frame_count=frame_count,
        material_query_indices=np.arange(len(_POINTS_M), dtype=np.int64),
        action_support=_POINTS_M[:, 0] / 0.04,
        engine_revision=JAX_FEM_REVISION,
        engine_version=JAX_FEM_VERSION,
        source_artifacts=source_artifacts,
        device=device,
        load_step_size=1.0 / (frame_count - 1),
        element_type="HEX8",
        constitutive_model="small-strain-isotropic-linear-elasticity",
        nonlinear_solver="jax-fem-default-newton",
        source_kind="synthetic",
    )
    arrays = load_physical_rollout_archive(
        output_dir / "physical-prediction.npz",
        expected_frame_count=frame_count,
    )
    zero_delta = arrays["zero_action_readout_m"] - arrays["frame_zero_points_m"][None]
    response = arrays["driven_readout_m"] - arrays["zero_action_readout_m"]
    right_face = _POINTS_M[:, 0] == 0.04
    return {
        "artifact_id": artifact["artifact_id"],
        "maximum_zero_action_drift_m": float(
            np.max(np.linalg.norm(zero_delta, axis=-1))
        ),
        "maximum_driven_minus_zero_response_m": float(
            np.max(np.linalg.norm(response, axis=-1))
        ),
        "final_right_face_x_displacement_m": float(
            np.max(
                arrays["driven_readout_m"][-1, right_face, 0]
                - arrays["frame_zero_points_m"][right_face, 0]
            )
        ),
        "portable_sha256": _portable_hashes(output_dir),
    }


def run_smoke(
    output_dir: str | Path,
    *,
    frame_count: int = 5,
    maximum_displacement_m: float = 0.004,
    young_modulus_pa: float = 100_000.0,
    poisson_ratio: float = 0.3,
) -> Mapping[str, Any]:
    if frame_count < 2:
        raise ValueError("frame_count must be >= 2")
    if maximum_displacement_m <= 0.0 or not np.isfinite(maximum_displacement_m):
        raise ValueError("maximum_displacement_m must be finite and positive")
    if young_modulus_pa <= 0.0 or not np.isfinite(young_modulus_pa):
        raise ValueError("young_modulus_pa must be finite and positive")
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1,0.5)")

    root = Path(output_dir).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()

    native = _load_native_modules()
    script_path = Path(__file__).resolve()
    run_a = _run_once(
        native,
        root / "run-a",
        frame_count=frame_count,
        maximum_displacement_m=maximum_displacement_m,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
        script_path=script_path,
    )
    run_b = _run_once(
        native,
        root / "run-b",
        frame_count=frame_count,
        maximum_displacement_m=maximum_displacement_m,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
        script_path=script_path,
    )
    deterministic = run_a["portable_sha256"] == run_b["portable_sha256"]
    if not deterministic:
        raise RuntimeError("native JAX-FEM replay is not byte-deterministic")

    response = float(run_a["maximum_driven_minus_zero_response_m"])
    zero_drift = float(run_a["maximum_zero_action_drift_m"])
    final_right = float(run_a["final_right_face_x_displacement_m"])
    response_tolerance = max(1e-8, maximum_displacement_m * 1e-5)
    if abs(final_right - maximum_displacement_m) > response_tolerance:
        raise RuntimeError("JAX-FEM prescribed displacement was not reproduced")
    if response <= maximum_displacement_m * 0.5:
        raise RuntimeError("JAX-FEM driven arm did not produce a material response")
    if zero_drift > 1e-10:
        raise RuntimeError("JAX-FEM zero-action drift exceeds the synthetic smoke bound")

    devices = [
        {
            "platform": str(getattr(device, "platform", "unknown")),
            "id": str(getattr(device, "id", "unknown")),
            "device_kind": str(getattr(device, "device_kind", "unknown")),
        }
        for device in native.jax.devices()
    ]
    descriptor: dict[str, Any] = {
        "schema": SMOKE_SCHEMA,
        "claim_boundary": (
            "Synthetic native-execution and provenance smoke only; no source-value, "
            "fresh-object, calibration, or Causal4D benefit claim."
        ),
        "backend_profile": "jax-fem-quasistatic-v1",
        "engine": {
            "repository": JAX_FEM_REPOSITORY,
            "revision": JAX_FEM_REVISION,
            "version": JAX_FEM_VERSION,
            "source_records": native.source_records,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "jax_version": str(getattr(native.jax, "__version__", "unknown")),
            "jax_enable_x64": bool(native.jax.config.jax_enable_x64),
            "devices": devices,
        },
        "problem": {
            "element_type": "HEX8",
            "point_count": len(_POINTS_M),
            "cell_count": len(_CELLS),
            "frame_count": frame_count,
            "maximum_displacement_m": maximum_displacement_m,
            "young_modulus_pa": young_modulus_pa,
            "poisson_ratio": poisson_ratio,
        },
        "checks": {
            "installed_source_matches_pinned_git_blobs": True,
            "portable_replay_byte_deterministic": deterministic,
            "maximum_zero_action_drift_m": zero_drift,
            "maximum_driven_minus_zero_response_m": response,
            "final_right_face_x_displacement_m": final_right,
        },
        "run_a": run_a,
        "run_b": run_b,
        "future_outcomes_read": False,
        "dataset_payload_read": False,
    }
    descriptor["smoke_id"] = _content_id(descriptor)
    result_path = root / "jax-fem-native-smoke.json"
    result_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return descriptor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--maximum-displacement-m", type=float, default=0.004)
    parser.add_argument("--young-modulus-pa", type=float, default=100_000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_smoke(
        args.output_dir,
        frame_count=args.frame_count,
        maximum_displacement_m=args.maximum_displacement_m,
        young_modulus_pa=args.young_modulus_pa,
        poisson_ratio=args.poisson_ratio,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
