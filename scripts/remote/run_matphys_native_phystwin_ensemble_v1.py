#!/usr/bin/env python3
"""Replay a MatPhys fold ensemble on one native released PhysTwin case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bayesian_phystwin.matphys_warp_ensemble_v1 import (
    baseline_relative_trajectory_ensemble_arrays,
    file_sha256,
    load_matphys_spring_ensemble,
)
from bayesian_phystwin.phystwin_graph import (
    PhysTwinSpringGraphConfig,
    build_phystwin_spring_graph,
)
from bayesian_phystwin.phystwin_state_injection import (
    _initialize_simulator,
    _released_self_collision_for_case,
    _rollout_initial,
)

SCHEMA = "bayesian-phystwin.matphys-native-phystwin-trajectory-ensemble"
VERSION = 1
PROTOCOL = "baseline-relative-target-excluded-matphys-native-phystwin-v1"


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical_id(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_bound_file(record: object, *, name: str) -> Path:
    _require(isinstance(record, dict), f"{name} record is missing")
    path = Path(record.get("path", ""))
    _require(path.is_file() and not path.is_symlink(), f"{name} is not an ordinary file")
    path = path.resolve(strict=True)
    _require(file_sha256(path) == record.get("sha256"), f"{name} SHA-256 changed")
    _require(path.stat().st_size == record.get("byte_count"), f"{name} size changed")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-repository", type=Path, required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--expected-official-revision", required=True)
    parser.add_argument("--case-input-manifest", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--spring-ensemble", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--incumbent-replays", type=int, default=4)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--num-substeps", type=int, default=667)
    parser.add_argument("--maximum-reference-rmse-m", type=float, default=0.002)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    execution = args.execution_repository.resolve(strict=True)
    _require(
        _git_revision(execution) == args.expected_execution_revision,
        "execution repository revision changed",
    )
    _require(
        not subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=execution,
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
        "execution repository is dirty",
    )
    official = args.official_phystwin_repo.resolve(strict=True)
    _require(
        _git_revision(official) == args.expected_official_revision,
        "official PhysTwin revision changed",
    )
    _require(args.incumbent_replays >= 2, "incumbent replays must be at least two")
    _require(args.dt > 0.0 and args.num_substeps > 0, "time integration is invalid")
    _require(
        args.maximum_reference_rmse_m > 0.0,
        "reference tolerance must be positive",
    )

    case_input = json.loads(args.case_input_manifest.read_text(encoding="utf-8"))
    _require(
        case_input.get("schema") == "bayesian-phystwin.matphys-native-phystwin-case"
        and case_input.get("schema_version") == 1,
        "native case manifest changed",
    )
    prediction = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    _require(
        prediction.get("schema") == "bayesian-phystwin.matphys-fold-ensemble-prediction"
        and prediction.get("schema_version") == 1,
        "MatPhys prediction manifest changed",
    )
    boundary = prediction.get("information_boundary", {})
    _require(
        boundary.get("target_future_observations_used") is False
        and boundary.get("target_future_outcomes_opened") is False
        and boundary.get("target_object_used_for_checkpoint_training") is False,
        "MatPhys prediction crossed its information boundary",
    )
    outputs = case_input["outputs"]
    inputs = case_input["inputs"]
    final_data_path = _load_bound_file(inputs["final_data"], name="final data")
    optimal_path = _load_bound_file(inputs["optimal_params"], name="optimal params")
    checkpoint_path = _load_bound_file(inputs["checkpoint"], name="checkpoint")
    baseline_path = _load_bound_file(
        inputs["baseline_trajectory"], name="baseline trajectory"
    )
    graph_path = _load_bound_file(outputs["episode_graph"], name="episode graph")
    complete_field_path = _load_bound_file(
        outputs["incumbent_complete_spring_field"], name="complete spring field"
    )
    prediction_graph = prediction.get("inputs", {}).get("episode_graph", {})
    prediction_incumbent = prediction.get("inputs", {}).get(
        "incumbent_spring_field", {}
    )
    _require(
        prediction_graph.get("sha256") == file_sha256(graph_path),
        "prediction and native case graph differ",
    )
    _require(
        prediction_incumbent.get("sha256")
        == outputs["incumbent_object_spring_field"]["sha256"],
        "prediction and native incumbent spring fields differ",
    )
    output_record = prediction.get("output", {})
    _require(
        file_sha256(args.spring_ensemble) == output_record.get("sha256"),
        "MatPhys spring ensemble changed",
    )
    member_count = prediction.get("member_count")
    _require(type(member_count) is int and member_count >= 2, "member count is invalid")
    fields = load_matphys_spring_ensemble(
        args.spring_ensemble, expected_member_count=member_count
    )

    with final_data_path.open("rb") as stream:
        data = pickle.load(stream)
    with optimal_path.open("rb") as stream:
        optimal = pickle.load(stream)
    with baseline_path.open("rb") as stream:
        released_baseline = np.asarray(pickle.load(stream), dtype=np.float32)
    observed = np.asarray(data["object_points"], dtype=np.float32)
    surface = np.asarray(data["surface_points"], dtype=np.float32)
    interior = np.asarray(data["interior_points"], dtype=np.float32)
    controller = np.asarray(data["controller_points"], dtype=np.float32)
    structure = np.concatenate((observed[0], surface, interior), axis=0)
    graph = build_phystwin_spring_graph(
        structure,
        controller[0],
        config=PhysTwinSpringGraphConfig(
            object_radius=float(optimal["object_radius"]),
            object_max_neighbours=int(optimal["object_max_neighbours"]),
            controller_radius=float(optimal["controller_radius"]),
            controller_max_neighbours=int(optimal["controller_max_neighbours"]),
        ),
    )
    with np.load(graph_path, allow_pickle=False) as archive:
        registered_points = np.asarray(archive["vertices"], dtype=np.float32)
        registered_edges = np.asarray(archive["springs"], dtype=np.int64)
    _require(
        np.array_equal(registered_points, fields.graph_points_m)
        and np.array_equal(registered_points, structure),
        "native graph vertices changed",
    )
    _require(
        np.array_equal(registered_edges, fields.graph_edges)
        and np.array_equal(
            registered_edges,
            np.asarray(graph.springs[: graph.num_object_springs], dtype=np.int64),
        ),
        "native graph edge order changed",
    )
    complete_field = np.load(complete_field_path, allow_pickle=False)
    _require(
        complete_field.dtype == np.float32
        and complete_field.shape == (len(graph.springs),)
        and np.array_equal(
            complete_field[: graph.num_object_springs],
            fields.incumbent_spring_y_pa,
        ),
        "complete and object incumbent fields disagree",
    )
    _require(
        released_baseline.shape == (len(observed), len(structure), 3),
        "released baseline shape changed",
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    simulator, torch, wp, _ = _initialize_simulator(
        official,
        data,
        optimal,
        checkpoint_path,
        graph,
        num_surface_points=observed.shape[1] + len(surface),
        original_count=observed.shape[1],
        dt=float(args.dt),
        num_substeps=int(args.num_substeps),
        self_collision=_released_self_collision_for_case(str(case_input["case_id"])),
        deterministic_spring_forces=True,
        device=str(args.device),
    )

    def rollout(object_field: np.ndarray) -> np.ndarray:
        full = complete_field.copy()
        full[: graph.num_object_springs] = object_field
        log_y = torch.log(
            torch.as_tensor(full, dtype=torch.float32, device=args.device)
        )
        simulator.set_spring_Y(log_y)
        wp.synchronize()
        positions, _ = _rollout_initial(simulator, wp, frame_count=len(observed))
        return positions.astype(np.float32, copy=False)

    started = time.perf_counter()
    incumbent_replicates = np.stack(
        [
            rollout(fields.incumbent_spring_y_pa)
            for _ in range(int(args.incumbent_replays))
        ]
    )
    member_trajectories = np.stack(
        [rollout(field) for field in fields.member_spring_y_pa]
    )
    rollout_seconds = time.perf_counter() - started
    arrays = baseline_relative_trajectory_ensemble_arrays(
        incumbent_replicates, member_trajectories
    )
    reference_delta = arrays["incumbent_replay_mean_m"] - released_baseline
    reference_rmse = float(np.sqrt(np.mean(reference_delta**2)))
    reference_max = float(np.max(np.abs(reference_delta)))
    passed = bool(reference_rmse <= args.maximum_reference_rmse_m)
    archive_path = args.output_dir / "matphys_native_trajectory_ensemble.npz"
    np.savez_compressed(archive_path, **arrays)
    identity = {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "protocol": PROTOCOL,
        "case_id": case_input["case_id"],
        "case_input_id": case_input["case_input_id"],
        "source_prediction_id": prediction["prediction_id"],
        "member_count": member_count,
        "incumbent_replay_count": int(args.incumbent_replays),
        "replay_estimator": (
            "baseline-relative-member-second-moment-plus-shared-incumbent-floor-v1"
        ),
        "parity": {
            "released_baseline_sha256": inputs["baseline_trajectory"]["sha256"],
            "coordinate_rmse_m": reference_rmse,
            "maximum_absolute_error_m": reference_max,
            "maximum_allowed_rmse_m": float(args.maximum_reference_rmse_m),
            "passed": passed,
        },
        "runtime": {
            "execution_revision": _git_revision(execution),
            "official_phystwin_revision": _git_revision(official),
            "runner_sha256": file_sha256(Path(__file__)),
            "device": str(args.device),
            "dt": float(args.dt),
            "num_substeps": int(args.num_substeps),
            "rollout_seconds": rollout_seconds,
        },
        "output": {
            "path": str(archive_path.resolve(strict=True)),
            "sha256": file_sha256(archive_path),
        },
        "information_boundary": {
            "source_only": True,
            "future_observations_used_by_matphys": False,
            "future_outcomes_opened_by_runner": False,
            "point_mean_changed": False,
            "calibration_claim_authorized": False,
        },
        "claim_boundary": (
            "Target-excluded MatPhys fold fields were propagated through the "
            "native released PhysTwin simulator. The stronger released point mean "
            "remains unchanged. Raw baseline-relative second moments are not a "
            "calibrated posterior until the source gate passes."
        ),
        "passed": passed,
    }
    result = {**identity, "result_id": _canonical_id(identity)}
    result_path = args.output_dir / "matphys_native_trajectory_ensemble.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
