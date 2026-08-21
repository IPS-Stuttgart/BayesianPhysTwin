#!/usr/bin/env python3
"""Run the pinned synthetic native MuJoCo volumetric-flex smoke."""

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
from typing import Any, cast

import numpy as np

from bayesian_phystwin.material_trajectory_engine_replays_v1 import (
    MuJoCoFlexReplayV1,
)
from bayesian_phystwin.material_trajectory_producer_v1 import (
    produce_material_trajectory_backend,
)
from bayesian_phystwin.physical_rollout_v1 import load_physical_rollout_archive

MUJOCO_REVISION = "237c17e48539b6c90bf90d3161547cbdcbfaa1e0"
MUJOCO_VERSION = "3.9.0"
MUJOCO_REPOSITORY = "https://github.com/google-deepmind/mujoco"
MUJOCO_WHEEL_FILENAME = (
    "mujoco-3.9.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
)
MUJOCO_WHEEL_SHA256 = "c148824d73487fe5ee29c371eff981645f372ccada1f20ea331288323e37c65e"
MUJOCO_INSTALLED_FILE_SHA256 = {
    "mujoco/__init__.py": (
        "f8e5b528617004b6215e16cb0c945faf1f8c7b5d798e5b92e3aabce19c838497"
    ),
    "mujoco/_structs.cpython-310-x86_64-linux-gnu.so": (
        "b6875dcfef3f895f8c293f9d5e20d4da8e7e37df1f3bda201124c1f2951b6c63"
    ),
    "mujoco/libmujoco.so.3.9.0": (
        "526773636a795dad11e094c8655d2375984a5cd7090f254d86bb71074651b852"
    ),
}
SMOKE_SCHEMA = "bayesian-phystwin.mujoco-flex-native-smoke-v1"
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
    mujoco: Any
    package_root: Path
    installed_records: Mapping[str, Mapping[str, str]]


def _load_native_modules(wheel_path: Path) -> _NativeModules:
    wheel = wheel_path.absolute()
    if not wheel.is_file() or wheel.is_symlink():
        raise RuntimeError("MuJoCo wheel must be an ordinary file")
    if wheel.name != MUJOCO_WHEEL_FILENAME:
        raise RuntimeError("MuJoCo wheel filename differs from the frozen runtime")
    if _sha256_file(wheel) != MUJOCO_WHEEL_SHA256:
        raise RuntimeError("MuJoCo wheel SHA-256 differs from the frozen runtime")
    if platform.python_implementation() != "CPython" or platform.python_version_tuple()[
        :2
    ] != ("3", "10"):
        raise RuntimeError("MuJoCo smoke requires the frozen CPython 3.10 ABI")

    installed_version = importlib.metadata.version("mujoco")
    if installed_version != MUJOCO_VERSION:
        raise RuntimeError(
            f"MuJoCo version mismatch: expected {MUJOCO_VERSION}, "
            f"found {installed_version}"
        )
    mujoco = importlib.import_module("mujoco")
    reported_version = str(getattr(mujoco, "__version__", ""))
    if reported_version != MUJOCO_VERSION:
        raise RuntimeError(
            f"MuJoCo reported version mismatch: expected {MUJOCO_VERSION}, "
            f"found {reported_version}"
        )
    distribution = importlib.metadata.distribution("mujoco")
    package_root = Path(str(distribution.locate_file(""))).resolve()
    records: dict[str, Mapping[str, str]] = {}
    for relative_path, expected_sha256 in MUJOCO_INSTALLED_FILE_SHA256.items():
        path = Path(str(distribution.locate_file(relative_path))).resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"MuJoCo runtime member is unavailable: {relative_path}")
        observed = _sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"MuJoCo runtime member changed: {relative_path}; "
                f"expected {expected_sha256}, found {observed}"
            )
        records[relative_path] = {"sha256": observed}
    return _NativeModules(
        mujoco=mujoco,
        package_root=package_root,
        installed_records=records,
    )


def _scene_xml(
    *,
    integrator_time_step_s: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    total_mass_kg: float,
) -> str:
    return f"""<mujoco model="bpt_mujoco_volumetric_flex_v1">
  <option timestep="{integrator_time_step_s:.17g}" gravity="0 0 0"
          integrator="implicitfast" iterations="100" tolerance="1e-12"/>
  <worldbody>
    <flexcomp name="soft" type="grid" dim="3" count="5 3 3"
              spacing="0.1 0.1 0.1" radius="0.001" mass="{total_mass_kg:.17g}">
      <contact contype="0" conaffinity="0" selfcollide="none"/>
      <edge damping="2"/>
      <elasticity young="{young_modulus_pa:.17g}"
                  poisson="{poisson_ratio:.17g}" damping="2"/>
      <pin gridrange="0 0 0 0 2 2"/>
    </flexcomp>
  </worldbody>
</mujoco>"""


@dataclass(slots=True)
class _ForceControlledReplay:
    replay: MuJoCoFlexReplayV1
    right_body_ids: np.ndarray
    force_per_vertex_n: float = 0.0

    @property
    def context(self) -> object:
        return self

    def synchronize(self) -> object:
        return self.replay.synchronize()

    def get_material_positions_m(self) -> object:
        return self.replay.get_material_positions_m()

    def step(self) -> None:
        data = self.replay.data
        data.xfrc_applied.fill(0.0)
        data.xfrc_applied[self.right_body_ids, 0] = self.force_per_vertex_n
        self.replay.step()


def _build_replay(
    native: _NativeModules,
    *,
    integrator_time_step_s: float,
    integrator_substeps: int,
    young_modulus_pa: float,
    poisson_ratio: float,
    total_mass_kg: float,
) -> _ForceControlledReplay:
    mujoco = native.mujoco
    model = mujoco.MjModel.from_xml_string(
        _scene_xml(
            integrator_time_step_s=integrator_time_step_s,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
            total_mass_kg=total_mass_kg,
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "soft")
    if flex_id != 0 or int(model.nflexvert) != 45 or int(model.nflexelem) != 96:
        raise RuntimeError(
            "MuJoCo compiled flex topology differs from the frozen scene"
        )
    positions = np.asarray(data.flexvert_xpos)
    right_vertices = np.flatnonzero(
        np.isclose(positions[:, 0], float(np.max(positions[:, 0])), atol=1e-12)
    )
    body_ids = np.asarray(model.flex_vertbodyid, dtype=np.int64)[right_vertices]
    if (
        len(right_vertices) != 9
        or len(np.unique(body_ids)) != 9
        or np.any(body_ids <= 0)
    ):
        raise RuntimeError("MuJoCo driven flex-face identity changed")

    def step(observed_model: object, observed_data: object) -> None:
        if observed_model is not model or observed_data is not data:
            raise RuntimeError("MuJoCo replay model/data identity changed")
        for _ in range(integrator_substeps):
            mujoco.mj_step(model, data)
        native_data = cast(Any, data)
        if not np.all(np.isfinite(native_data.qpos)) or not np.all(
            np.isfinite(native_data.qvel)
        ):
            raise RuntimeError("MuJoCo volumetric flex produced non-finite state")

    return _ForceControlledReplay(
        replay=MuJoCoFlexReplayV1(
            model=model,
            data=data,
            flex_id=flex_id,
            step_callback=step,
        ),
        right_body_ids=np.ascontiguousarray(body_ids),
    )


def _portable_hashes(directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for member in PORTABLE_MEMBERS:
        path = directory / member
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing portable member: {member}")
        hashes[member] = _sha256_file(path)
    return hashes


def _topology_descriptor() -> dict[str, object]:
    return {
        "flex_name": "soft",
        "dimension": 3,
        "grid_count": [5, 3, 3],
        "grid_spacing_m": [0.1, 0.1, 0.1],
        "vertex_count": 45,
        "tetrahedron_count": 96,
        "pinned_grid_range": [0, 0, 0, 0, 2, 2],
        "driven_face": "maximum-x-nine-vertices",
    }


def _run_once(
    native: _NativeModules,
    output_dir: Path,
    *,
    frame_count: int,
    output_time_step_s: float,
    integrator_time_step_s: float,
    integrator_substeps: int,
    force_per_vertex_n: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    total_mass_kg: float,
    script_path: Path,
    adapter_path: Path,
    producer_revision: str,
) -> Mapping[str, Any]:
    def replay_factory() -> _ForceControlledReplay:
        return _build_replay(
            native,
            integrator_time_step_s=integrator_time_step_s,
            integrator_substeps=integrator_substeps,
            young_modulus_pa=young_modulus_pa,
            poisson_ratio=poisson_ratio,
            total_mass_kg=total_mass_kg,
        )

    def driven_control(_: int, replay: Any) -> None:
        replay.force_per_vertex_n = force_per_vertex_n

    def zero_control(_: int, replay: Any) -> None:
        replay.force_per_vertex_n = 0.0

    topology = _topology_descriptor()
    source_artifacts = {
        "scripts/remote/run_mujoco_flex_native_smoke.py": _sha256_file(script_path),
        "src/bayesian_phystwin/material_trajectory_engine_replays_v1.py": (
            _sha256_file(adapter_path)
        ),
        **{
            f"native/{path}": record["sha256"]
            for path, record in native.installed_records.items()
        },
    }
    artifact = produce_material_trajectory_backend(
        output_dir=output_dir,
        backend_kind="mujoco-flex-v1",
        replay_factory=replay_factory,
        driven_control=driven_control,
        zero_action_control=zero_control,
        frame_count=frame_count,
        material_query_indices=np.arange(45, dtype=np.int64),
        action_support=np.repeat(
            np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64), 9
        ),
        engine_revision=MUJOCO_REVISION,
        engine_version=MUJOCO_VERSION,
        producer_repository="IPS-Stuttgart/BayesianPhysTwin",
        producer_revision=producer_revision,
        producer_version="mujoco-flex-native-smoke-v1",
        producer_artifacts=source_artifacts,
        topology_sha256=_content_id(topology),
        device="cpu",
        device_name=platform.processor() or platform.machine(),
        time_step_s=output_time_step_s,
        scene_id="synthetic-pinned-volumetric-flex-v1",
        model_kind="three-dimensional-tetrahedral-flex-grid",
        constitutive_model="MuJoCo isotropic linear elasticity",
        integrator="implicitfast",
        solver="MuJoCo native flex constraint solver",
        substeps=integrator_substeps,
        engine_parameters={
            **topology,
            "integrator_time_step_s": integrator_time_step_s,
            "output_time_step_s": output_time_step_s,
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "force_per_driven_vertex_n": force_per_vertex_n,
            "young_modulus_pa": young_modulus_pa,
            "poisson_ratio": poisson_ratio,
            "total_mass_kg": total_mass_kg,
            "contact_enabled": False,
        },
    )
    arrays = load_physical_rollout_archive(
        output_dir / "physical-prediction.npz",
        expected_frame_count=frame_count,
    )
    zero_delta = arrays["zero_action_readout_m"] - arrays["frame_zero_points_m"][None]
    response = arrays["driven_readout_m"] - arrays["zero_action_readout_m"]
    final_x = response[-1, :, 0]
    return {
        "artifact_id": artifact["artifact_id"],
        "runtime_id": artifact["runtime_id"],
        "maximum_zero_action_drift_m": float(
            np.max(np.linalg.norm(zero_delta, axis=-1))
        ),
        "maximum_driven_minus_zero_response_m": float(
            np.max(np.linalg.norm(response, axis=-1))
        ),
        "final_maximum_x_response_m": float(np.max(final_x)),
        "portable_sha256": _portable_hashes(output_dir),
    }


def _stiffness_probe(
    native: _NativeModules,
    *,
    frame_count: int,
    integrator_time_step_s: float,
    integrator_substeps: int,
    force_per_vertex_n: float,
    young_modulus_pa: float,
    poisson_ratio: float,
    total_mass_kg: float,
) -> float:
    replay = _build_replay(
        native,
        integrator_time_step_s=integrator_time_step_s,
        integrator_substeps=integrator_substeps,
        young_modulus_pa=young_modulus_pa,
        poisson_ratio=poisson_ratio,
        total_mass_kg=total_mass_kg,
    )
    initial = np.asarray(replay.get_material_positions_m()).copy()
    replay.force_per_vertex_n = force_per_vertex_n
    for _ in range(frame_count - 1):
        replay.step()
    response = np.asarray(replay.get_material_positions_m()) - initial
    return float(np.max(response[:, 0]))


def run_smoke(
    output_dir: str | Path,
    *,
    wheel_path: str | Path,
    frame_count: int = 5,
    output_time_step_s: float = 0.025,
    integrator_time_step_s: float = 0.00001,
    force_per_vertex_n: float = 1.0,
    young_modulus_pa: float = 1000.0,
    poisson_ratio: float = 0.3,
    total_mass_kg: float = 7.0,
) -> Mapping[str, Any]:
    if type(frame_count) is not int or frame_count < 2:
        raise ValueError("frame_count must be an integer >= 2")
    for name, value in (
        ("output_time_step_s", output_time_step_s),
        ("integrator_time_step_s", integrator_time_step_s),
        ("force_per_vertex_n", force_per_vertex_n),
        ("young_modulus_pa", young_modulus_pa),
        ("total_mass_kg", total_mass_kg),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not np.isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in (-1,0.5)")
    ratio = output_time_step_s / integrator_time_step_s
    integrator_substeps = int(round(ratio))
    if integrator_substeps < 1 or not np.isclose(
        integrator_substeps * integrator_time_step_s,
        output_time_step_s,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("output time step must be an exact integrator-step multiple")

    root = Path(output_dir).absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    native = _load_native_modules(Path(wheel_path))
    script_path = Path(__file__).resolve()
    repository_root = script_path.parents[2]
    adapter_path = (
        repository_root
        / "src"
        / "bayesian_phystwin"
        / "material_trajectory_engine_replays_v1.py"
    )
    if not adapter_path.is_file() or adapter_path.is_symlink():
        raise RuntimeError("MuJoCo replay adapter source is unavailable")
    producer_revision = _repository_revision(script_path)

    with tempfile.TemporaryDirectory(
        prefix=f".{root.name}.staging.",
        dir=root.parent,
    ) as temporary:
        staging = Path(temporary) / root.name
        staging.mkdir()

        def run_once(name: str) -> Mapping[str, Any]:
            return _run_once(
                native,
                staging / name,
                frame_count=frame_count,
                output_time_step_s=output_time_step_s,
                integrator_time_step_s=integrator_time_step_s,
                integrator_substeps=integrator_substeps,
                force_per_vertex_n=force_per_vertex_n,
                young_modulus_pa=young_modulus_pa,
                poisson_ratio=poisson_ratio,
                total_mass_kg=total_mass_kg,
                script_path=script_path,
                adapter_path=adapter_path,
                producer_revision=producer_revision,
            )

        run_a = run_once("run-a")
        run_b = run_once("run-b")
        deterministic = run_a["portable_sha256"] == run_b["portable_sha256"]
        if not deterministic:
            raise RuntimeError("native MuJoCo Flex replay is not byte-deterministic")

        response = float(run_a["maximum_driven_minus_zero_response_m"])
        zero_drift = float(run_a["maximum_zero_action_drift_m"])
        minimum_response = 1e-4
        if response <= minimum_response:
            raise RuntimeError("MuJoCo Flex driven arm did not produce a response")
        if zero_drift > 1e-12:
            raise RuntimeError("MuJoCo Flex zero-action drift exceeds the smoke bound")
        low_young = young_modulus_pa / 2.0
        high_young = young_modulus_pa * 2.0
        low_response = _stiffness_probe(
            native,
            frame_count=frame_count,
            integrator_time_step_s=integrator_time_step_s,
            integrator_substeps=integrator_substeps,
            force_per_vertex_n=force_per_vertex_n,
            young_modulus_pa=low_young,
            poisson_ratio=poisson_ratio,
            total_mass_kg=total_mass_kg,
        )
        high_response = _stiffness_probe(
            native,
            frame_count=frame_count,
            integrator_time_step_s=integrator_time_step_s,
            integrator_substeps=integrator_substeps,
            force_per_vertex_n=force_per_vertex_n,
            young_modulus_pa=high_young,
            poisson_ratio=poisson_ratio,
            total_mass_kg=total_mass_kg,
        )
        if not low_response > high_response * 1.25:
            raise RuntimeError("MuJoCo Flex response lacks Young-modulus sensitivity")

        descriptor: dict[str, Any] = {
            "schema": SMOKE_SCHEMA,
            "claim_boundary": (
                "Synthetic native-execution, volumetric-flex, constitutive-sensitivity, "
                "and provenance smoke only; no source-value, fresh-object, calibration, "
                "or Causal4D benefit claim."
            ),
            "backend_profile": "mujoco-flex-v1",
            "engine": {
                "repository": MUJOCO_REPOSITORY,
                "revision": MUJOCO_REVISION,
                "version": MUJOCO_VERSION,
                "wheel_filename": MUJOCO_WHEEL_FILENAME,
                "wheel_sha256": MUJOCO_WHEEL_SHA256,
                "installed_records": native.installed_records,
            },
            "producer_revision": producer_revision,
            "runtime": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "numpy_version": np.__version__,
                "device": "cpu",
            },
            "problem": {
                **_topology_descriptor(),
                "frame_count": frame_count,
                "output_time_step_s": output_time_step_s,
                "integrator_time_step_s": integrator_time_step_s,
                "integrator_substeps": integrator_substeps,
                "force_per_driven_vertex_n": force_per_vertex_n,
                "young_modulus_pa": young_modulus_pa,
                "poisson_ratio": poisson_ratio,
                "total_mass_kg": total_mass_kg,
                "gravity_m_s2": [0.0, 0.0, 0.0],
            },
            "checks": {
                "wheel_matches_pinned_sha256": True,
                "installed_runtime_matches_pinned_sha256": True,
                "portable_replay_byte_deterministic": deterministic,
                "maximum_zero_action_drift_m": zero_drift,
                "maximum_driven_minus_zero_response_m": response,
                "minimum_required_response_m": minimum_response,
                "low_young_modulus_pa": low_young,
                "low_young_response_m": low_response,
                "high_young_modulus_pa": high_young,
                "high_young_response_m": high_response,
                "minimum_low_over_high_response_ratio": 1.25,
                "observed_low_over_high_response_ratio": low_response / high_response,
            },
            "run_a": run_a,
            "run_b": run_b,
            "future_outcomes_read": False,
            "dataset_payload_read": False,
        }
        descriptor["smoke_id"] = _content_id(descriptor)
        result_path = staging / "mujoco-flex-native-smoke.json"
        result_path.write_text(
            json.dumps(descriptor, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, root)
    return descriptor


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--output-time-step-s", type=float, default=0.025)
    parser.add_argument("--integrator-time-step-s", type=float, default=0.00001)
    parser.add_argument("--force-per-vertex-n", type=float, default=1.0)
    parser.add_argument("--young-modulus-pa", type=float, default=1000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.3)
    parser.add_argument("--total-mass-kg", type=float, default=7.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_smoke(
        args.output_dir,
        wheel_path=args.wheel,
        frame_count=args.frame_count,
        output_time_step_s=args.output_time_step_s,
        integrator_time_step_s=args.integrator_time_step_s,
        force_per_vertex_n=args.force_per_vertex_n,
        young_modulus_pa=args.young_modulus_pa,
        poisson_ratio=args.poisson_ratio,
        total_mass_kg=args.total_mass_kg,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
