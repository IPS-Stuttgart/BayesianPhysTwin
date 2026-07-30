#!/usr/bin/env python3
"""Generate an observation-free Cloth Sim2Real physical baseline.

This exploratory adapter keeps the frozen v1 guarded readout update unchanged
while allowing its physical baseline to come from either MuJoCo 3 or SOFA.
It never reads benchmark point clouds or future observations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_COMMIT = "178a9b9722191c51cf0dcbc3cf0dc03701b09eb3"
SOFA_ARCHIVE_SHA256 = (
    "de1ab962978f1b77db97d9925e6fef6b2bc924aff6aa04956a59d9e1bd0e3720"
)
SIMULATORS = ("mujoco3", "sofa")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _case_parts(case_id: str) -> tuple[str, str]:
    parts = case_id.split("/", maxsplit=1)
    _require(len(parts) == 2, "case id must be CLOTH/TASK")
    cloth_sample, task = parts
    _require(task in {"dynamic", "quasi_static"}, "unknown task")
    return cloth_sample, task


def _config_path(benchmark_root: Path, simulator: str) -> Path:
    _require(simulator in SIMULATORS, f"unsupported simulator {simulator}")
    return benchmark_root / f"bcm/conf/envs/{simulator}.yaml"


def _initialize_environment(
    environment: Any,
    simulator: str,
    trajectory: list[np.ndarray],
    pretrajectory_steps: int,
) -> None:
    if simulator == "sofa":
        environment.start_simulation(
            trajectory,
            pretrajectory_steps,
            None,
            None,
            None,
            None,
            None,
        )
    else:
        environment.reset()


def _extract_faces(environment: Any, info: dict[str, Any], simulator: str) -> np.ndarray:
    if simulator == "sofa":
        topology = environment.SquareGravity.getObject("topo")
        faces = np.asarray(topology.triangles.array(), dtype=np.int64)
    else:
        faces = np.asarray(info["faces"], dtype=np.int64)
    _require(
        faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0,
        "simulator returned invalid triangle faces",
    )
    return faces.copy()


def _sofa_provenance() -> dict[str, Any]:
    sofa_root_value = os.environ.get("SOFA_ROOT")
    _require(sofa_root_value is not None, "SOFA_ROOT is required for SOFA")
    sofa_root = Path(sofa_root_value).resolve()
    git_info = sofa_root / "git-info.txt"
    _require(git_info.is_file(), "SOFA git-info.txt is missing")
    text = git_info.read_text(encoding="utf-8")
    matches = re.findall(r"\b[0-9a-f]{40}\b", text)
    _require(matches, "SOFA git-info.txt has no full commit")
    return {
        "name": "sofa",
        "version": "23.06.00",
        "commit": matches[0],
        "git_info_sha256": _sha256(git_info),
        "official_archive_sha256": SOFA_ARCHIVE_SHA256,
    }


def _simulator_provenance(simulator: str) -> dict[str, Any]:
    if simulator == "sofa":
        return _sofa_provenance()
    return {
        "name": "mujoco3",
        "version": importlib.import_module("mujoco").__version__,
    }


def _simulate(args: argparse.Namespace) -> int:
    benchmark_root = args.benchmark_code_root.resolve()
    _require((benchmark_root / "bcm").is_dir(), "benchmark code root has no bcm")
    git_head = subprocess.check_output(
        ["git", "-C", str(benchmark_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    _require(git_head == BENCHMARK_COMMIT, "benchmark checkout commit changed")

    simulator = str(args.simulator)
    cloth_sample, task = _case_parts(args.case_id)
    sys.path.insert(0, str(benchmark_root))
    omega_conf = importlib.import_module("omegaconf").OmegaConf
    get_env = importlib.import_module("bcm.envs").get_env
    generate_full_trajectory = importlib.import_module(
        "bcm.manipulation_utils"
    ).generate_full_trajectory

    full = omega_conf.load(_config_path(benchmark_root, simulator))
    parameter_name = f"params_{task}_{cloth_sample}"
    _require(hasattr(full, parameter_name), f"missing parameters {parameter_name}")
    environment_config = omega_conf.create(
        {
            "name": simulator,
            "render_mode": "None",
            "depth": False,
            "width": 320,
            "height": 288,
            "params": omega_conf.to_container(
                getattr(full, parameter_name),
                resolve=True,
            ),
        }
    )
    real_setup = {
        "table": {
            "xmin": -0.4,
            "xmax": 0.4,
            "ymin": -0.1,
            "ymax": 0.8,
            "zmax": 0.195,
        },
        "gripper_start": {
            "left": [0.0, 0.0, 1.0],
            "right": [0.5, 0.0, 1.0],
        },
    }
    environment = get_env(
        environment_config,
        real_setup=real_setup,
        target=None,
    )
    dt_s = float(environment.trajectory_dt)
    stabilization_steps = int(1.0 / dt_s)
    trajectory, pretrajectory_steps = generate_full_trajectory(
        dt_s,
        cloth_sample,
        "unused",
        stabilization_steps,
        task,
        simulator,
    )
    _initialize_environment(
        environment,
        simulator,
        trajectory,
        pretrajectory_steps,
    )

    vertices: list[np.ndarray] = []
    info: dict[str, Any] | None = None
    for index, action in enumerate(trajectory):
        _, _, _, _, info = environment.step(action)
        if index >= pretrajectory_steps:
            state = np.asarray(info["vertices"], dtype=np.float64)
            _require(
                state.ndim == 2
                and state.shape[1] == 3
                and np.isfinite(state).all(),
                "simulator returned invalid vertices",
            )
            vertices.append(state.copy())
    _require(info is not None and vertices, "simulator returned no trajectory")
    faces = _extract_faces(environment, info, simulator)
    environment.close()

    output = args.output.resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        vertices_m=np.stack(vertices),
        faces=faces,
        actions_m=np.asarray(
            trajectory[pretrajectory_steps:],
            dtype=np.float64,
        ),
        dt_s=np.asarray(dt_s),
    )
    _write_json_once(
        output.with_suffix(".json"),
        {
            "schema_version": 2,
            "artifact_kind": "ClothSim2RealPhysicalBaseline",
            "case_id": args.case_id,
            "benchmark_commit": BENCHMARK_COMMIT,
            "simulator": simulator,
            "simulator_provenance": _simulator_provenance(simulator),
            "pretrajectory_steps": pretrajectory_steps,
            "simulator_frame_count": len(vertices),
            "node_count": int(vertices[0].shape[0]),
            "face_count": int(len(faces)),
            "dt_s": dt_s,
            "npz_sha256": _sha256(output),
            "point_cloud_coordinates_read": False,
            "prefix_observations_read": False,
            "future_outcomes_read": False,
            "claim_boundary": (
                "observation-free physical baseline generation; evaluation on "
                "the previously opened benchmark repeats is exploratory"
            ),
        },
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-code-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--simulator",
        choices=SIMULATORS,
        default="mujoco3",
        help="opt-in physical backend; the default preserves the v1 baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(function=_simulate)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
