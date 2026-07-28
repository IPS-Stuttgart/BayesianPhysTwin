#!/usr/bin/env python3
"""Run and seal one fresh-object 384-node physical backbone."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from bayesian_phystwin.deform360_fresh_pairwise_physical import (
    AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE,
    CANONICAL_NODE_COUNT,
    OFFICIAL_PHYSTWIN_REVISION,
    OFFICIAL_REAL_CONFIG_SHA256,
    UPSTREAM_FILE_SHA256,
    WARP_DYNAMICS,
    build_persistence_backbone_arrays,
    build_prediction_only_bundle,
    build_warp_backbone_arrays,
    load_frame_zero_ply,
    write_physical_artifacts,
)
from bayesian_phystwin.deform360_fresh_pairwise_protocol import (
    build_backbone_seal,
    canonical_sha256,
    file_sha256,
    fresh_case_record,
    load_bound_cohort,
    load_fresh_pairwise_protocol,
    load_json,
)
from bayesian_phystwin.deform360_fresh_source_lock import (
    validate_fresh_source_admission,
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


def _failure(command: Sequence[str], log_path: Path, returncode: int) -> str:
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    return (
        f"command failed with exit {returncode}: {' '.join(command)}\n"
        + "\n".join(tail)
    )


def _validate_runtime(
    source_repo: Path,
    official_repo: Path,
    official_config: Path,
) -> dict[str, Any]:
    observed = {}
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = source_repo / relative
        _require(path.is_file(), f"frozen source file is missing: {relative}")
        digest = file_sha256(path)
        _require(digest == expected, f"frozen source file changed: {relative}")
        observed[relative] = digest
    _require(
        official_config.is_file()
        and file_sha256(official_config) == OFFICIAL_REAL_CONFIG_SHA256,
        "official PhysTwin config changed",
    )
    revision = _git_revision(official_repo)
    _require(revision == OFFICIAL_PHYSTWIN_REVISION, "official PhysTwin changed")
    return {
        "official_phystwin_revision": revision,
        "official_config_sha256": file_sha256(official_config),
        "source_file_sha256": observed,
    }


def _validate_twin_summary(
    path: Path,
    *,
    protocol_sha256: str,
    cohort_sha256: str,
    case_record: Mapping[str, Any],
    prediction_data: Path,
    graph_path: Path,
    simulator_data: Path,
    state_path: Path,
    passed: bool,
) -> dict[str, Any]:
    summary = load_json(path)
    _require(
        summary.get("artifact_kind")
        == "Deform360FreshPairwiseAutomaticEpisodeTwin"
        and summary.get("protocol_config_sha256") == protocol_sha256
        and summary.get("cohort_lock_sha256") == cohort_sha256
        and summary.get("result_sha256")
        == canonical_sha256(summary, digest_key="result_sha256"),
        "automatic-twin summary is incompatible",
    )
    _require(
        all(summary.get(key) == value for key, value in case_record.items()),
        "automatic-twin identity changed",
    )
    _require(summary.get("passed") is passed, "automatic-twin status changed")
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
    result = load_json(result_path)
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
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--processed-episode-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
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
    protocol = load_fresh_pairwise_protocol(
        args.protocol,
        repository_root=repo,
    )
    cohort = load_bound_cohort(args.cohort_lock, protocol)
    admission = load_json(args.admission)
    validate_fresh_source_admission(admission)
    _require(admission["accepted"] is True, "source admission did not pass")
    record = fresh_case_record(
        cohort,
        object_id=str(admission["object_id"]),
        episode_id=int(admission["episode_id"]),
    )
    _require(
        admission["admission_sha256"] == record["admission_sha256"],
        "admission differs from cohort lock",
    )
    processed = args.processed_episode_dir.resolve()
    frame_zero = processed / "start_obj_pcd.ply"
    robot = processed / "robot/robot.npz"
    control_meta = processed / "control_points.meta.json"
    _require(
        file_sha256(frame_zero)
        == admission["source_files"]["frame_zero"]["sha256"],
        "frame-zero PLY differs from admission",
    )
    control = load_json(control_meta)
    _require(
        file_sha256(control_meta)
        == admission["source_files"]["control_meta"]["sha256"],
        "control metadata differs from admission",
    )
    _require(
        file_sha256(robot) == control["inputs"]["robot_sha256"],
        "known robot action differs from source provenance",
    )
    source_repo = args.source_repo.resolve()
    official_repo = args.official_phystwin_repo.resolve()
    official_config = args.official_config.resolve()
    provenance = _validate_runtime(source_repo, official_repo, official_config)
    provenance.update(
        {
            "bayesian_phystwin_revision": code_revision,
            "physical_runner_sha256": file_sha256(Path(__file__).resolve()),
            "automatic_twin_wrapper_sha256": file_sha256(
                repo
                / "scripts/remote/build_deform360_fresh_pairwise_automatic_twin.py"
            ),
        }
    )

    root = args.output_dir.resolve()
    _require(not root.exists(), "physical output directory already exists")
    root.mkdir(parents=True)
    prediction_data = root / "prediction_only_input.pkl"
    prediction_summary = build_prediction_only_bundle(
        frame_zero,
        robot,
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
    graph_path = root / "episode_graph.npz"
    simulator_data = root / "simulator_final_data.pkl"
    state_path = root / "state_artifact.npz"
    twin_summary_path = root / "twin_summary.json"
    python_path = args.python.absolute()
    _require(python_path.is_file(), "requested Python runtime is missing")
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
        (
            str(repo / "src"),
            str(source_repo / "src"),
            str(args.deform360_repo.resolve()),
        )
    )
    twin_command = [
        str(python_path),
        str(repo / "scripts/remote/build_deform360_fresh_pairwise_automatic_twin.py"),
        "--source-repo",
        str(source_repo),
        "--protocol",
        str(args.protocol.resolve()),
        "--cohort-lock",
        str(args.cohort_lock.resolve()),
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
    automatic_log = root / "logs/automatic_twin.log"
    twin_exit, twin_runtime = _run_logged(
        twin_command,
        env=automatic_env,
        log_path=automatic_log,
    )
    runtimes: dict[str, float] = {"automatic_twin": twin_runtime}
    archive_path = root / "prediction.npz"
    manifest_path = root / "physical_prediction_manifest.json"
    common_inputs: dict[str, Path] = {
        "protocol": args.protocol.resolve(),
        "cohort_lock": args.cohort_lock.resolve(),
        "source_admission": args.admission.resolve(),
        "frame_zero_ply": frame_zero,
        "known_action": robot,
        "control_metadata": control_meta,
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
            protocol_sha256=protocol["config_file_sha256"],
            cohort_sha256=cohort["cohort_lock_sha256"],
            case_record=record,
            prediction_data=prediction_data,
            graph_path=graph_path,
            simulator_data=simulator_data,
            state_path=state_path,
            passed=False,
        )
        points, _ = load_frame_zero_ply(frame_zero)
        arrays = build_persistence_backbone_arrays(points)
        physical_manifest = write_physical_artifacts(
            archive_path,
            manifest_path,
            arrays,
            protocol_config_sha256=protocol["config_file_sha256"],
            cohort_lock_sha256=cohort["cohort_lock_sha256"],
            case_record=record,
            physical_mode="source_admission_persistence_fallback",
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
            raise RuntimeError(_failure(twin_command, automatic_log, twin_exit))
        _validate_twin_summary(
            twin_summary_path,
            protocol_sha256=protocol["config_file_sha256"],
            cohort_sha256=cohort["cohort_lock_sha256"],
            case_record=record,
            prediction_data=prediction_data,
            graph_path=graph_path,
            simulator_data=simulator_data,
            state_path=state_path,
            passed=True,
        )
        official_env = dict(common_env)
        official_env["PYTHONPATH"] = os.pathsep.join(
            (str(source_repo / "src"), str(args.deform360_repo.resolve()))
        )
        smoke_script = (
            source_repo / "scripts/remote/run_deform360_official_phystwin_smoke.py"
        )
        split_path = (
            source_repo
            / "configs/causal4d_public/deform360_independent_source_split_v1.json"
        )
        result_paths: dict[str, Path] = {}
        for label, scale in (("driven", 1.0), ("zero_action", 0.0)):
            rollout_dir = root / f"warp_{label}"
            command = [
                str(python_path),
                str(smoke_script),
                "--official-phystwin-repo",
                str(official_repo),
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
            log_path = root / f"logs/warp_{label}.log"
            returncode, elapsed = _run_logged(
                command,
                env=official_env,
                log_path=log_path,
            )
            if returncode:
                raise RuntimeError(_failure(command, log_path, returncode))
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
        points, _ = load_frame_zero_ply(frame_zero)
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
        physical_manifest = write_physical_artifacts(
            archive_path,
            manifest_path,
            arrays,
            protocol_config_sha256=protocol["config_file_sha256"],
            cohort_lock_sha256=cohort["cohort_lock_sha256"],
            case_record=record,
            physical_mode="warp_twin",
            input_files=warp_inputs,
            runtime_provenance={**provenance, "runtime_seconds": runtimes},
        )
    seal = build_backbone_seal(
        root / "prediction_seal.json",
        protocol_path=args.protocol,
        cohort_path=args.cohort_lock,
        case_record=record,
        admission_path=args.admission,
        prediction_archive=archive_path,
        physical_manifest=manifest_path,
    )
    print(
        json.dumps(
            {
                "case": record["case"],
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
