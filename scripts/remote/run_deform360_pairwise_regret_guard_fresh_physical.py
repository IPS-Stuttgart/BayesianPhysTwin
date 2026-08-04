#!/usr/bin/env python3
"""Build one fresh frame-zero-only physical/persistence backbone."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_bias_aware_prospective_physical import (
    AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE,
    CANONICAL_NODE_COUNT,
    OFFICIAL_PHYSTWIN_REVISION,
    OFFICIAL_REAL_CONFIG_SHA256,
    UPSTREAM_FILE_SHA256,
    WARP_DYNAMICS,
    build_persistence_backbone_arrays,
    build_prediction_only_bundle,
    build_warp_backbone_arrays,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_artifacts import (
    build_fresh_physical_seal,
    validate_fresh_processing_cohort,
    write_fresh_physical_artifacts,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_processing import (
    fresh_processing_case,
    validate_fresh_processing_protocol,
)
from bayesian_phystwin.deform360_pairwise_regret_guard_fresh_protocol import (
    file_sha256,
    validate_fresh_technical_lock,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_repository(repository: Path) -> str:
    revision = _git_revision(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status.strip(), f"repository is dirty: {repository}")
    return revision


def _validate_runtime(upstream: Path, official: Path, config: Path) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = upstream / relative
        _require(path.is_file(), f"frozen upstream file is missing: {relative}")
        digest = file_sha256(path)
        _require(digest == expected, f"frozen upstream file changed: {relative}")
        observed[relative] = digest
    _require(
        config.is_file() and file_sha256(config) == OFFICIAL_REAL_CONFIG_SHA256,
        "official PhysTwin real config changed",
    )
    revision = _git_revision(official)
    _require(revision == OFFICIAL_PHYSTWIN_REVISION, "official PhysTwin changed")
    _require(
        not subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=official,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "official PhysTwin checkout is dirty",
    )
    return {
        "official_phystwin_revision": revision,
        "official_config_sha256": file_sha256(config),
        "upstream_file_sha256": observed,
    }


def _read_frame_zero_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)
    _require(
        points.ndim == 2
        and points.shape[1] == 3
        and colors.shape == points.shape
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(colors)),
        "frame-zero point cloud is invalid",
    )
    return points, colors


def _run_logged(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    log_path: Path,
) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return completed.returncode, time.perf_counter() - started


def _command_failure(command: Sequence[str], log_path: Path, returncode: int) -> str:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return f"command failed with exit {returncode}: {' '.join(command)}\n" + "\n".join(
        lines[-80:]
    )


def _validate_twin_summary(
    path: Path,
    *,
    lock: Mapping[str, Any],
    protocol: Mapping[str, Any],
    case: Mapping[str, Any],
    prediction_data: Path,
    graph_path: Path,
    simulator_data: Path,
    state_path: Path,
    passed: bool,
) -> dict[str, Any]:
    summary = _load_json(path)
    _require(
        summary.get("artifact_kind")
        == "Deform360PairwiseRegretGuardFreshAutomaticEpisodeTwin"
        and summary.get("protocol_id") == lock["protocol_id"]
        and summary.get("technical_lock_sha256") == lock["lock_sha256"]
        and summary.get("processing_protocol_sha256") == protocol["protocol_sha256"]
        and all(summary.get(key) == value for key, value in case.items()),
        "automatic-twin summary is incompatible",
    )
    _require(
        summary.get("passed") is passed
        and summary.get("state_metrics", {}).get("passed") is passed,
        "automatic-twin admission status changed",
    )
    _require(
        summary.get("input_sha256", {}).get("episode_final_data")
        == file_sha256(prediction_data),
        "automatic twin used another prediction bundle",
    )
    outputs = summary.get("output_sha256", {})
    for key, output in (
        ("episode_graph", graph_path),
        ("simulator_final_data", simulator_data),
        ("state_artifact", state_path),
    ):
        _require(
            output.is_file() and outputs.get(key) == file_sha256(output),
            f"automatic-twin {key} changed",
        )
    boundary = summary.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("post_initial_object_observation_used") is False
        and boundary.get("target_access") is False
        and boundary.get("future_object_tracks_present") is False
        and boundary.get("held_v8_runtime_or_target_artifact_access") is False,
        "automatic twin crossed its prediction boundary",
    )
    return summary


def _expected_warp_overrides() -> dict[str, Any]:
    return {
        "controller_max_neighbours": WARP_DYNAMICS["controller_max_neighbours"],
        "controller_radius": WARP_DYNAMICS["controller_radius_m"],
        "dashpot_damping": WARP_DYNAMICS["dashpot_damping"],
        "drag_damping": WARP_DYNAMICS["drag_damping"],
        "init_spring_Y": WARP_DYNAMICS["init_spring_y"],
    }


def _load_warp_trajectory(
    result_path: Path,
    *,
    label: str,
    scale: float,
    simulator_data: Path,
    graph_path: Path,
    vertex_count: int,
) -> np.ndarray:
    result = _load_json(result_path)
    _require(result.get("passed") is True, f"{label} Warp rollout failed")
    _require("external_target_scoring" not in result, "Warp read a target")
    _require(
        result.get("data_sha256") == file_sha256(simulator_data)
        and result.get("official_phystwin_revision") == OFFICIAL_PHYSTWIN_REVISION
        and result.get("config_sha256") == OFFICIAL_REAL_CONFIG_SHA256
        and result.get("config_overrides") == _expected_warp_overrides(),
        f"{label} Warp numerical contract changed",
    )
    _require(
        result.get("support_dynamics", {}).get("mode")
        == WARP_DYNAMICS["support_dynamics"],
        f"{label} support dynamics changed",
    )
    graph_record = result.get("canonical_reusable_graph", {})
    _require(
        graph_record.get("file_sha256") == file_sha256(graph_path)
        and int(graph_record.get("controller_patch_size_per_anchor", -1))
        == WARP_DYNAMICS["canonical_controller_patch_size"],
        f"{label} Warp graph changed",
    )
    _require(
        float(
            result.get("realized_actuation", {}).get(
                "controller_displacement_scale", -1.0
            )
        )
        == scale,
        f"{label} action scale changed",
    )
    trajectory_path = result_path.with_name("official_phystwin_trajectory.npz")
    _require(
        result.get("trajectory_sha256") == file_sha256(trajectory_path),
        f"{label} trajectory checksum changed",
    )
    with np.load(trajectory_path, allow_pickle=False) as stored:
        trajectory = np.asarray(stored["vertices"], dtype=np.float64)
    _require(
        trajectory.ndim == 3
        and trajectory.shape[0] == 76
        and trajectory.shape[1] >= vertex_count
        and trajectory.shape[2] == 3
        and np.all(np.isfinite(trajectory)),
        f"invalid {label} trajectory",
    )
    return trajectory[:, :vertex_count]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--technical-lock", type=Path, required=True)
    parser.add_argument("--processing-protocol", type=Path, required=True)
    parser.add_argument("--processing-cohort", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--deform360-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    code_revision = _require_clean_repository(repo)
    lock_path = args.technical_lock.resolve()
    protocol_path = args.processing_protocol.resolve()
    cohort_path = args.processing_cohort.resolve()
    lock = _load_json(lock_path)
    protocol = _load_json(protocol_path)
    cohort = _load_json(cohort_path)
    validate_fresh_technical_lock(lock)
    validate_fresh_processing_protocol(protocol)
    validate_fresh_processing_cohort(cohort, lock=lock, protocol=protocol)
    processed = args.processed_episode_dir.resolve()
    _require(processed.name.startswith("episode_"), "processed episode name changed")
    episode_id = int(processed.name.removeprefix("episode_"))
    object_id = processed.parent.name
    case = fresh_processing_case(lock, object_id, episode_id)
    disposition = next(row for row in cohort["cases"] if row["case"] == case["case"])
    _require(disposition["status"] == "admitted", "source case is not admitted")
    upstream = args.upstream_repo.resolve()
    official = args.official_phystwin_repo.resolve()
    official_config = args.official_config.resolve()
    provenance = _validate_runtime(upstream, official, official_config)
    provenance["bayesian_phystwin_revision"] = code_revision
    provenance["physical_runner_sha256"] = file_sha256(Path(__file__).resolve())
    auto_script = (
        repo
        / "scripts/remote/build_deform360_pairwise_regret_guard_fresh_automatic_twin.py"
    )
    provenance["automatic_twin_wrapper_sha256"] = file_sha256(auto_script)

    root = args.work_root.resolve() / str(case["case"])
    _require(not root.exists(), "physical work directory already exists")
    root.mkdir(parents=True)
    frame_zero_source = processed / "start_obj_pcd.ply"
    known_action = processed / "robot.npz"
    _require(frame_zero_source.is_file(), "frame-zero PLY is missing")
    _require(known_action.is_file(), "known action is missing")
    points, colors = _read_frame_zero_ply(frame_zero_source)
    expected_point_count = disposition["admission"]["observed_source_contract"][
        "frame_zero_point_count"
    ]
    _require(
        disposition["admission"].get("accepted") is True
        and len(points) == expected_point_count,
        "frame-zero point count differs from source admission",
    )
    geometry = root / "frame_zero_points.npz"
    np.savez_compressed(geometry, points_m=points, colors=colors)
    prediction_data = root / "prediction_only_input.pkl"
    prediction_summary = build_prediction_only_bundle(
        geometry,
        known_action,
        prediction_data,
        object_id=object_id,
        episode_id=episode_id,
        case=str(case["case"]),
    )
    prediction_summary_path = root / "prediction_only_input.json"
    prediction_summary_path.write_text(
        json.dumps(prediction_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    graph_path = root / "episode_graph.npz"
    simulator_data = root / "simulator_final_data.pkl"
    state_path = root / "state_artifact.npz"
    twin_summary_path = root / "twin_summary.json"
    common_env = dict(os.environ)
    common_env.update(
        {
            "PYNPUT_BACKEND": "dummy",
            "PYOPENGL_PLATFORM": "egl",
            "WANDB_MODE": "disabled",
        }
    )
    automatic_env = dict(common_env)
    automatic_env["PYTHONPATH"] = os.pathsep.join(
        (str(repo / "src"), str(args.deform360_repo.resolve()))
    )
    twin_command = [
        str(args.python),
        str(auto_script),
        "--upstream-repo",
        str(upstream),
        "--technical-lock",
        str(lock_path),
        "--processing-protocol",
        str(protocol_path),
        "--object-id",
        object_id,
        "--episode-id",
        str(episode_id),
        "--episode-final-data",
        str(prediction_data),
        "--episode-graph",
        str(graph_path),
        "--simulator-final-data",
        str(simulator_data),
        "--state-artifact",
        str(state_path),
        "--summary",
        str(twin_summary_path),
        "--canonical-node-count",
        str(CANONICAL_NODE_COUNT),
    ]
    automatic_log = root / "logs/automatic_twin.log"
    twin_exit, twin_runtime = _run_logged(
        twin_command, env=automatic_env, log_path=automatic_log
    )
    runtimes: dict[str, float] = {"automatic_twin": twin_runtime}
    archive_path = root / "physical_prediction_source.npz"
    physical_manifest_path = root / "physical_prediction_source.json"
    common_inputs: dict[str, Path] = {
        "technical_lock": lock_path,
        "processing_protocol": protocol_path,
        "processing_cohort": cohort_path,
        "source_processing": processed / "fresh_pairwise_processing.json",
        "source_admission": processed / "fresh_pairwise_admission.json",
        "frame_zero_source": frame_zero_source,
        "frame_zero_geometry": geometry,
        "known_action": known_action,
        "prediction_only_input": prediction_data,
        "prediction_only_summary": prediction_summary_path,
        "episode_graph": graph_path,
        "simulator_final_data": simulator_data,
        "state_artifact": state_path,
        "twin_summary": twin_summary_path,
        "automatic_twin_log": automatic_log,
    }
    if twin_exit == AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE:
        twin = _validate_twin_summary(
            twin_summary_path,
            lock=lock,
            protocol=protocol,
            case=case,
            prediction_data=prediction_data,
            graph_path=graph_path,
            simulator_data=simulator_data,
            state_path=state_path,
            passed=False,
        )
        arrays = build_persistence_backbone_arrays(points)
        physical_manifest = write_fresh_physical_artifacts(
            archive_path,
            physical_manifest_path,
            arrays,
            case=case,
            technical_lock=lock,
            processing_protocol=protocol,
            physical_mode="persistence_fallback",
            input_files=common_inputs,
            runtime_provenance={**provenance, "runtime_seconds": runtimes},
            fallback_diagnostics={
                "reason": "automatic_twin_source_admission_failed",
                "automatic_twin_exit_code": twin_exit,
                "automatic_twin_result_sha256": twin["result_sha256"],
                "automatic_twin_state_metrics": twin["state_metrics"],
                "warp_attempted": False,
            },
        )
    else:
        if twin_exit:
            raise RuntimeError(_command_failure(twin_command, automatic_log, twin_exit))
        _validate_twin_summary(
            twin_summary_path,
            lock=lock,
            protocol=protocol,
            case=case,
            prediction_data=prediction_data,
            graph_path=graph_path,
            simulator_data=simulator_data,
            state_path=state_path,
            passed=True,
        )
        official_env = dict(common_env)
        official_env["PYTHONPATH"] = os.pathsep.join(
            (str(upstream / "src"), str(args.deform360_repo.resolve()))
        )
        smoke_script = (
            upstream / "scripts/remote/run_deform360_official_phystwin_smoke.py"
        )
        split_path = (
            upstream
            / "configs/causal4d_public/deform360_independent_source_split_v1.json"
        )
        result_paths: dict[str, Path] = {}
        for label, scale in (("driven", 1.0), ("zero_action", 0.0)):
            rollout_dir = root / f"warp_{label}"
            command = [
                str(args.python),
                str(smoke_script),
                "--official-phystwin-repo",
                str(official),
                "--data",
                str(simulator_data),
                "--config",
                str(official_config),
                "--split-json",
                str(split_path),
                "--output-dir",
                str(rollout_dir),
                "--canonical-reusable-graph",
                str(graph_path),
                "--device",
                args.device,
                "--controller-radius-m",
                str(WARP_DYNAMICS["controller_radius_m"]),
                "--controller-max-neighbours",
                str(WARP_DYNAMICS["controller_max_neighbours"]),
                "--canonical-controller-patch-size",
                str(WARP_DYNAMICS["canonical_controller_patch_size"]),
                "--init-spring-y",
                str(WARP_DYNAMICS["init_spring_y"]),
                "--drag-damping",
                str(WARP_DYNAMICS["drag_damping"]),
                "--dashpot-damping",
                str(WARP_DYNAMICS["dashpot_damping"]),
                "--controller-displacement-scale",
                str(scale),
                "--support-dynamics",
                str(WARP_DYNAMICS["support_dynamics"]),
                "--report-edge-strain",
            ]
            log_path = root / "logs" / f"warp_{label}.log"
            returncode, elapsed = _run_logged(
                command, env=official_env, log_path=log_path
            )
            if returncode:
                raise RuntimeError(_command_failure(command, log_path, returncode))
            runtimes[f"warp_{label}"] = elapsed
            result_paths[label] = rollout_dir / "official_phystwin_smoke.json"
        with np.load(graph_path, allow_pickle=False) as graph:
            vertices = np.asarray(graph["vertices"], dtype=np.float64)
            springs = np.asarray(graph["springs"], dtype=np.int64)
            rest_lengths = np.asarray(graph["rest_lengths"], dtype=np.float64)
            anchors = np.asarray(graph["contact_anchor_indices"], dtype=np.int64)
            graph_semantic = str(np.asarray(graph["reusable_graph_sha256"]).item())
        with np.load(state_path, allow_pickle=False) as state:
            weights = np.asarray(state["readout_weights"], dtype=np.float64)
            state_graph_semantic = str(
                np.asarray(state["canonical_graph_sha256"]).item()
            )
        _require(
            graph_semantic == state_graph_semantic, "state readout uses another graph"
        )
        driven = _load_warp_trajectory(
            result_paths["driven"],
            label="driven",
            scale=1.0,
            simulator_data=simulator_data,
            graph_path=graph_path,
            vertex_count=len(vertices),
        )
        zero = _load_warp_trajectory(
            result_paths["zero_action"],
            label="zero_action",
            scale=0.0,
            simulator_data=simulator_data,
            graph_path=graph_path,
            vertex_count=len(vertices),
        )
        arrays = build_warp_backbone_arrays(
            points,
            vertices=vertices,
            springs=springs,
            rest_lengths=rest_lengths,
            contact_anchor_indices=anchors,
            readout_weights=weights,
            driven_vertices_m=driven,
            zero_action_vertices_m=zero,
        )
        warp_inputs = dict(common_inputs)
        for label, result_path in result_paths.items():
            warp_inputs[f"{label}_result"] = result_path
            warp_inputs[f"{label}_trajectory"] = result_path.with_name(
                "official_phystwin_trajectory.npz"
            )
        physical_manifest = write_fresh_physical_artifacts(
            archive_path,
            physical_manifest_path,
            arrays,
            case=case,
            technical_lock=lock,
            processing_protocol=protocol,
            physical_mode="warp_twin",
            input_files=warp_inputs,
            runtime_provenance={**provenance, "runtime_seconds": runtimes},
        )
    backbone_dir = args.backbone_root.resolve() / str(case["case"])
    seal = build_fresh_physical_seal(
        lock_path,
        protocol_path,
        cohort_path,
        backbone_dir,
        object_id=object_id,
        episode_id=episode_id,
        physical_archive=archive_path,
        physical_manifest=physical_manifest_path,
    )
    print(
        json.dumps(
            {
                "case": case["case"],
                "physical_mode": physical_manifest["physical_mode"],
                "physical_manifest_sha256": physical_manifest["result_sha256"],
                "backbone_seal_sha256": seal["result_sha256"],
                "runtime_seconds": runtimes,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
