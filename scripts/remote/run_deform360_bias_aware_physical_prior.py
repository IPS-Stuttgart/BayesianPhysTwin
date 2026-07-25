#!/usr/bin/env python3
"""Build and seal one fresh frame-zero-only Deform360 physical backbone."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    build_prospective_backbone_seal,
    canonical_sha256,
    file_sha256,
    prospective_case_record,
)
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
    frame_zero_physical_policy,
    write_physical_artifacts,
)
from bayesian_phystwin.deform360_bias_aware_prospective_protocol import (
    PROTOCOL_ID,
    load_bias_aware_prospective_protocol,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object expected: {path}")
    return value


def _validate_stage(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    staged: Path,
    record: Mapping[str, Any],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    stage_path = staged / "prediction_prefix_manifest.json"
    frame_zero_path = staged / "frame_zero_reconstruction_manifest.json"
    stage = _load_json(stage_path)
    frame_zero = _load_json(frame_zero_path)
    for label, value, kind in (
        ("prefix", stage, "Deform360BiasAwarePredictionPrefix"),
        (
            "frame-zero",
            frame_zero,
            "Deform360BiasAwareFrameZeroReconstruction",
        ),
    ):
        _require(
            value.get("artifact_kind") == kind
            and value.get("protocol_id") == PROTOCOL_ID
            and value.get("protocol_config_sha256") == protocol["config_sha256"]
            and value.get("result_sha256")
            == canonical_sha256(value, digest_key="result_sha256"),
            f"{label} manifest is incompatible",
        )
        _require(
            all(value.get(key) == expected for key, expected in record.items()),
            f"{label} case identity changed",
        )
    stage_boundary = stage.get("information_boundary", {})
    frame_boundary = frame_zero.get("information_boundary", {})
    _require(
        stage_boundary.get("source_object_frames_after_prefix_read") is False
        and stage_boundary.get("future_dense_reconstruction_read") is False
        and stage_boundary.get("future_particle_tracks_read") is False
        and stage_boundary.get("target_metric_read") is False,
        "prefix staging crossed its prediction boundary",
    )
    _require(
        frame_boundary.get("object_observation_frames_used") == [0]
        and frame_boundary.get("future_object_rgb_read") is False
        and frame_boundary.get("future_dense_reconstruction_read") is False
        and frame_boundary.get("future_particle_tracks_read") is False
        and frame_boundary.get("target_metric_read") is False,
        "frame-zero reconstruction crossed its prediction boundary",
    )
    geometry = staged / "frame_zero_points.npz"
    action = staged / "known-action" / "robot.npz"
    _require(
        file_sha256(geometry) == frame_zero["outputs_sha256"]["frame_zero_points"],
        "frame-zero geometry checksum changed",
    )
    _require(
        file_sha256(action) == stage["staged_robot_sha256"]["known_action"],
        "known action checksum changed",
    )
    _require(
        file_sha256(protocol_path) == stage["inputs_sha256"]["protocol"],
        "staging used another protocol file",
    )
    return geometry, action, frame_zero_path, frame_zero


def _validate_runtime(
    upstream: Path,
    official: Path,
    official_config: Path,
) -> dict[str, Any]:
    observed: dict[str, str] = {}
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = upstream / relative
        _require(path.is_file(), f"frozen upstream file is missing: {relative}")
        digest = file_sha256(path)
        _require(digest == expected, f"frozen upstream file changed: {relative}")
        observed[relative] = digest
    _require(
        official_config.is_file()
        and file_sha256(official_config) == OFFICIAL_REAL_CONFIG_SHA256,
        "official PhysTwin real config changed",
    )
    revision = _git_revision(official)
    _require(revision == OFFICIAL_PHYSTWIN_REVISION, "official PhysTwin changed")
    return {
        "official_phystwin_revision": revision,
        "official_config_sha256": file_sha256(official_config),
        "upstream_file_sha256": observed,
    }


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
    return (
        f"command failed with exit {returncode}: {' '.join(command)}\n"
        + "\n".join(lines[-80:])
    )


def _validate_twin_summary(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    record: Mapping[str, Any],
    prediction_data: Path,
    graph_path: Path,
    simulator_data: Path,
    state_path: Path,
    passed: bool,
) -> dict[str, Any]:
    summary = _load_json(path)
    _require(
        summary.get("artifact_kind") == "Deform360BiasAwareAutomaticEpisodeTwin"
        and summary.get("protocol_id") == PROTOCOL_ID
        and summary.get("protocol_config_sha256") == protocol["config_sha256"]
        and summary.get("result_sha256")
        == canonical_sha256(summary, digest_key="result_sha256"),
        "automatic-twin summary is incompatible",
    )
    _require(
        all(summary.get(key) == value for key, value in record.items()),
        "automatic-twin case identity changed",
    )
    _require(summary.get("passed") is passed, "automatic-twin status changed")
    _require(
        summary.get("state_metrics", {}).get("passed") is passed,
        "automatic-twin admission metrics disagree",
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
        and boundary.get("future_object_tracks_present") is False,
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--staged-case-dir", type=Path, required=True)
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
    protocol_path = args.protocol.resolve()
    protocol = load_bias_aware_prospective_protocol(protocol_path)
    staged = args.staged_case_dir.resolve()
    record = prospective_case_record(
        protocol_path,
        object_id=staged.name.rsplit("-ep", 1)[0],
        episode_id=int(staged.name.rsplit("-ep", 1)[1]),
    )
    geometry, action, frame_zero_manifest, frame_zero = _validate_stage(
        protocol, protocol_path, staged, record
    )
    policy = frame_zero_physical_policy(frame_zero)
    upstream = args.upstream_repo.resolve()
    official = args.official_phystwin_repo.resolve()
    official_config = args.official_config.resolve()
    provenance = (
        _validate_runtime(upstream, official, official_config)
        if policy == "automatic_twin"
        else {"physical_runtime_required": False}
    )
    provenance["bayesian_phystwin_revision"] = code_revision
    provenance["physical_runner_sha256"] = file_sha256(Path(__file__).resolve())
    provenance["automatic_twin_wrapper_sha256"] = file_sha256(
        repo / "scripts/remote/build_deform360_bias_aware_automatic_twin.py"
    )

    root = args.work_root.resolve() / str(record["case"])
    _require(not root.exists(), "physical work directory already exists")
    root.mkdir(parents=True)
    prediction_data = root / "prediction_only_input.pkl"
    prediction_summary = build_prediction_only_bundle(
        geometry,
        action,
        prediction_data,
        object_id=str(record["object_id"]),
        episode_id=int(record["episode_id"]),
        case=str(record["case"]),
    )
    prediction_summary_path = root / "prediction_only_input.json"
    prediction_summary_path.write_text(
        json.dumps(prediction_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if policy == "persistence_only":
        with np.load(geometry, allow_pickle=False) as stored:
            arrays = build_persistence_backbone_arrays(stored["points_m"])
        archive_path = root / "prediction.npz"
        physical_manifest_path = root / "physical_prediction_manifest.json"
        physical_manifest = write_physical_artifacts(
            archive_path,
            physical_manifest_path,
            arrays,
            case_record=record,
            protocol_config_sha256=str(protocol["config_sha256"]),
            physical_mode="persistence_fallback",
            input_files={
                "protocol": protocol_path,
                "prediction_prefix_manifest": (
                    staged / "prediction_prefix_manifest.json"
                ),
                "frame_zero_manifest": frame_zero_manifest,
                "frame_zero_geometry": geometry,
                "known_action": action,
                "prediction_only_input": prediction_data,
                "prediction_only_summary": prediction_summary_path,
            },
            runtime_provenance={
                **provenance,
                "runtime_seconds": {"automatic_twin": 0.0, "warp": 0.0},
            },
            fallback_diagnostics={
                "reason": "frame_zero_reconstruction_persistence_only",
                "material_point_source": frame_zero["material_point_source"],
                "fallback_source_config_sha256": frame_zero[
                    "fallback_source_config_sha256"
                ],
                "automatic_twin_attempted": False,
                "warp_attempted": False,
                "state_update_available": False,
            },
        )
        backbone_dir = args.backbone_root.resolve() / str(record["case"])
        seal = build_prospective_backbone_seal(
            protocol_path,
            backbone_dir,
            object_id=str(record["object_id"]),
            episode_id=int(record["episode_id"]),
            physical_archive=archive_path,
            physical_manifest=physical_manifest_path,
        )
        print(
            json.dumps(
                {
                    "case": record["case"],
                    "physical_mode": physical_manifest["physical_mode"],
                    "physical_manifest_sha256": physical_manifest[
                        "result_sha256"
                    ],
                    "backbone_seal_sha256": seal["result_sha256"],
                    "runtime_seconds": {
                        "automatic_twin": 0.0,
                        "warp": 0.0,
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    graph_path = root / "episode_graph.npz"
    simulator_data = root / "simulator_final_data.pkl"
    state_path = root / "state_artifact.npz"
    twin_summary_path = root / "twin_summary.json"
    python_path = args.python
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
        str(python_path),
        str(repo / "scripts/remote/build_deform360_bias_aware_automatic_twin.py"),
        "--repo",
        str(upstream),
        "--protocol",
        str(protocol_path),
        "--object-id",
        str(record["object_id"]),
        "--episode-id",
        str(record["episode_id"]),
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
    automatic_log = root / "logs" / "automatic_twin.log"
    twin_exit, twin_runtime = _run_logged(
        twin_command, env=automatic_env, log_path=automatic_log
    )
    runtimes: dict[str, float] = {"automatic_twin": twin_runtime}
    archive_path = root / "prediction.npz"
    physical_manifest_path = root / "physical_prediction_manifest.json"
    common_inputs: dict[str, Path] = {
        "protocol": protocol_path,
        "prediction_prefix_manifest": staged / "prediction_prefix_manifest.json",
        "frame_zero_manifest": frame_zero_manifest,
        "frame_zero_geometry": geometry,
        "known_action": action,
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
            protocol=protocol,
            record=record,
            prediction_data=prediction_data,
            graph_path=graph_path,
            simulator_data=simulator_data,
            state_path=state_path,
            passed=False,
        )
        with np.load(geometry, allow_pickle=False) as stored:
            arrays = build_persistence_backbone_arrays(stored["points_m"])
        physical_manifest = write_physical_artifacts(
            archive_path,
            physical_manifest_path,
            arrays,
            case_record=record,
            protocol_config_sha256=str(protocol["config_sha256"]),
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
        twin = _validate_twin_summary(
            twin_summary_path,
            protocol=protocol,
            record=record,
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
                str(python_path),
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
            graph_semantic == state_graph_semantic,
            "state readout uses another graph",
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
        with np.load(geometry, allow_pickle=False) as stored:
            initial = np.asarray(stored["points_m"], dtype=np.float64)
        arrays = build_warp_backbone_arrays(
            initial,
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
        physical_manifest = write_physical_artifacts(
            archive_path,
            physical_manifest_path,
            arrays,
            case_record=record,
            protocol_config_sha256=str(protocol["config_sha256"]),
            physical_mode="warp_twin",
            input_files=warp_inputs,
            runtime_provenance={**provenance, "runtime_seconds": runtimes},
        )
    backbone_dir = args.backbone_root.resolve() / str(record["case"])
    seal = build_prospective_backbone_seal(
        protocol_path,
        backbone_dir,
        object_id=str(record["object_id"]),
        episode_id=int(record["episode_id"]),
        physical_archive=archive_path,
        physical_manifest=physical_manifest_path,
    )
    output = {
        "case": record["case"],
        "physical_mode": physical_manifest["physical_mode"],
        "physical_manifest_sha256": physical_manifest["result_sha256"],
        "backbone_seal_sha256": seal["result_sha256"],
        "runtime_seconds": runtimes,
    }
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
