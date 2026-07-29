#!/usr/bin/env python3
"""Run and seal one pre-lock V14 frame-zero physical backbone."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_causal_response_direct_depth_cohort import (
    validate_v14_staging_queue,
)
from bayesian_phystwin.deform360_causal_response_direct_depth_physical import (
    PRELOCK_PROTOCOL_ID,
    build_v14_physical_arrays,
    build_v14_prediction_only_bundle,
    load_v14_physical_prelock_protocol,
    v14_physical_case_record,
    write_v14_physical_artifacts,
)
from bayesian_phystwin.deform360_causal_response_prefix_geometry import (
    load_v14_prefix_geometry_protocol,
)
from bayesian_phystwin.deform360_causal_response_prefix_geometry_validation import (
    load_v14_prefix_geometry_validation,
    validate_v14_prefix_geometry_bundle,
)
from bayesian_phystwin.deform360_causal_response_prefix_geometry_validation_v2 import (
    load_v14_prefix_geometry_validation_v2,
    validate_v14_prefix_geometry_bundle_v2,
)
from bayesian_phystwin.deform360_fresh_pairwise_physical import (
    AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE,
    CANONICAL_NODE_COUNT,
    WARP_DYNAMICS,
    build_persistence_backbone_arrays,
    build_warp_backbone_arrays,
    load_frame_zero_ply,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "JSON artifact must contain an object")
    return payload


def _load_runner_helpers(repo: Path) -> ModuleType:
    path = repo / "scripts/remote/run_deform360_fresh_pairwise_physical.py"
    spec = importlib.util.spec_from_file_location(
        "_v14_pairwise_physical_helpers",
        path,
    )
    _require(spec is not None and spec.loader is not None, "cannot load helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_parent_file(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    digest_key: str,
) -> None:
    _require(
        path.is_file()
        and file_sha256(path) == protocol["parent_artifacts"][digest_key],
        f"V14 physical parent changed: {digest_key}",
    )


def _validate_bound_geometry(
    *,
    protocol: Mapping[str, Any],
    case_record: Mapping[str, Any],
    geometry_protocol_path: Path,
    runtime_v1_path: Path,
    validation_v1_path: Path,
    runtime_v2_path: Path,
    validation_v2_path: Path,
    manifest_path: Path,
    result_path: Path,
    runtime_application_path: Path,
    geometry_episode: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    geometry_protocol = load_v14_prefix_geometry_protocol(geometry_protocol_path)
    runtime_v1 = _load_json(runtime_v1_path)
    validation_v1 = load_v14_prefix_geometry_validation(validation_v1_path)
    runtime_v2 = _load_json(runtime_v2_path)
    validation_v2 = load_v14_prefix_geometry_validation_v2(validation_v2_path)
    rank = int(case_record["queue_rank"])
    if rank == 3:
        manifest, result, application = validate_v14_prefix_geometry_bundle(
            manifest_path=manifest_path,
            result_path=result_path,
            runtime_application_path=runtime_application_path,
            geometry_protocol=geometry_protocol,
            runtime_amendment=runtime_v1,
            validation_amendment=validation_v1,
            geometry_episode=geometry_episode,
        )
    else:
        manifest, result, application = validate_v14_prefix_geometry_bundle_v2(
            manifest_path=manifest_path,
            result_path=result_path,
            runtime_application_path=runtime_application_path,
            geometry_protocol=geometry_protocol,
            runtime_amendment_v2=runtime_v2,
            validation_amendment_v2=validation_v2,
            geometry_episode=geometry_episode,
        )
    expected = next(
        record
        for record in protocol["geometry_cases"]
        if int(record["queue_rank"]) == rank
    )
    observed = {
        "object_hash": manifest["object_hash"],
        "case_hash": manifest["case_hash"],
        "physical_node_count": int(manifest["physical_node_count"]),
        "successful_camera_count": len(manifest["cameras"]),
        "geometry_manifest_artifact_sha256": manifest["artifact_sha256"],
        "geometry_manifest_file_sha256": file_sha256(manifest_path),
        "geometry_result_artifact_sha256": result["artifact_sha256"],
        "geometry_result_file_sha256": file_sha256(result_path),
        "runtime_application_artifact_sha256": application["artifact_sha256"],
        "runtime_application_file_sha256": file_sha256(runtime_application_path),
        "runtime_contract_version": 1 if rank == 3 else 2,
    }
    _require(
        all(expected.get(key) == value for key, value in observed.items()),
        "V14 physical geometry differs from the pre-lock ledger",
    )
    _require(
        observed["object_hash"] == case_record["object_hash"]
        and observed["case_hash"] == case_record["case_hash"]
        and observed["physical_node_count"] == case_record["physical_node_count"],
        "V14 physical geometry differs from its hash-only case record",
    )
    return manifest, result, application


def _validate_twin_summary(
    path: Path,
    *,
    protocol: Mapping[str, Any],
    case_record: Mapping[str, Any],
    prediction_data: Path,
    graph_path: Path,
    simulator_data: Path,
    state_path: Path,
    passed: bool,
) -> dict[str, Any]:
    summary = _load_json(path)
    _require(
        summary.get("artifact_kind")
        == "Deform360CausalResponseDirectDepthAutomaticEpisodeTwinV14"
        and summary.get("protocol_id") == PRELOCK_PROTOCOL_ID
        and summary.get("physical_prelock_config_sha256") == protocol["config_sha256"]
        and summary.get("artifact_sha256")
        == hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in summary.items()
                    if key != "artifact_sha256"
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "V14 automatic-twin summary is incompatible",
    )
    for key in (
        "queue_rank",
        "object_hash",
        "case_hash",
        "metadata_sha256",
        "physical_node_count",
        "successful_camera_count",
    ):
        _require(
            summary.get(key) == case_record[key],
            f"V14 automatic-twin {key} changed",
        )
    _require(
        summary.get("passed") is passed
        and summary.get("state_metrics", {}).get("passed") is passed,
        "V14 automatic-twin status changed",
    )
    _require(
        summary.get("input_sha256", {}).get("episode_final_data")
        == file_sha256(prediction_data),
        "V14 automatic twin used another prediction bundle",
    )
    outputs = summary.get("output_sha256", {})
    for key, output in (
        ("episode_graph", graph_path),
        ("simulator_final_data", simulator_data),
        ("state_artifact", state_path),
    ):
        _require(
            output.is_file() and outputs.get(key) == file_sha256(output),
            f"V14 automatic-twin {key} changed",
        )
    boundary = summary.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("post_initial_object_observation_used") is False
        and boundary.get("prefix_or_future_tactile_read") is False
        and boundary.get("identity_or_metric_outcome_read") is False
        and boundary.get("source_lock_read") is False
        and boundary.get("plaintext_object_or_episode_identity_retained") is False
        and boundary.get("held_v8_artifact_or_process_access") is False,
        "V14 automatic twin crossed its prediction boundary",
    )
    return summary


def _load_graph_state(
    graph_path: Path,
    state_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(graph_path, allow_pickle=False) as graph:
        vertices = np.asarray(graph["vertices"], dtype=np.float64)
        springs = np.asarray(graph["springs"], dtype=np.int64)
        rest_lengths = np.asarray(graph["rest_lengths"], dtype=np.float64)
        anchors = np.asarray(graph["contact_anchor_indices"], dtype=np.int64)
        graph_semantic = str(np.asarray(graph["reusable_graph_sha256"]).item())
    with np.load(state_path, allow_pickle=False) as state:
        weights = np.asarray(state["readout_weights"], dtype=np.float64)
        state_graph_semantic = str(np.asarray(state["canonical_graph_sha256"]).item())
    _require(
        graph_semantic == state_graph_semantic,
        "V14 state readout uses another graph",
    )
    return vertices, springs, rest_lengths, anchors, weights


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prelock-protocol", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-rank", type=int, required=True)
    parser.add_argument("--geometry-protocol", type=Path, required=True)
    parser.add_argument("--runtime-v1", type=Path, required=True)
    parser.add_argument("--validation-v1", type=Path, required=True)
    parser.add_argument("--runtime-v2", type=Path, required=True)
    parser.add_argument("--validation-v2", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--geometry-result", type=Path, required=True)
    parser.add_argument("--runtime-application", type=Path, required=True)
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
    helpers = _load_runner_helpers(repo)
    code_revision = helpers._require_clean_repository(repo)
    prelock_path = args.prelock_protocol.resolve()
    protocol = load_v14_physical_prelock_protocol(prelock_path)
    implementation_files = {
        "artifact_module": (
            repo / "src/bayesian_phystwin/"
            "deform360_causal_response_direct_depth_physical.py"
        ),
        "automatic_twin": (
            repo / "scripts/remote/"
            "build_deform360_causal_response_direct_depth_v14_automatic_twin.py"
        ),
        "physical_runner": Path(__file__).resolve(),
    }
    _require(
        all(
            file_sha256(path) == protocol["implementation"]["file_sha256"][name]
            for name, path in implementation_files.items()
        ),
        "V14 physical implementation differs from the pre-lock protocol",
    )
    queue_path = args.queue.resolve()
    queue = validate_v14_staging_queue(queue_path)
    _verify_parent_file(
        queue_path,
        protocol=protocol,
        digest_key="staging_queue_file_sha256",
    )
    case_record = v14_physical_case_record(
        protocol,
        queue,
        queue_rank=args.queue_rank,
    )
    parent_paths = {
        "geometry_protocol_file_sha256": args.geometry_protocol.resolve(),
        "runtime_v1_file_sha256": args.runtime_v1.resolve(),
        "validation_v1_file_sha256": args.validation_v1.resolve(),
        "runtime_v2_file_sha256": args.runtime_v2.resolve(),
        "validation_v2_file_sha256": args.validation_v2.resolve(),
    }
    for digest_key, path in parent_paths.items():
        _verify_parent_file(path, protocol=protocol, digest_key=digest_key)
    processed = args.processed_episode_dir.resolve()
    manifest_path = args.geometry_manifest.resolve()
    result_path = args.geometry_result.resolve()
    runtime_application_path = args.runtime_application.resolve()
    _validate_bound_geometry(
        protocol=protocol,
        case_record=case_record,
        geometry_protocol_path=parent_paths["geometry_protocol_file_sha256"],
        runtime_v1_path=parent_paths["runtime_v1_file_sha256"],
        validation_v1_path=parent_paths["validation_v1_file_sha256"],
        runtime_v2_path=parent_paths["runtime_v2_file_sha256"],
        validation_v2_path=parent_paths["validation_v2_file_sha256"],
        manifest_path=manifest_path,
        result_path=result_path,
        runtime_application_path=runtime_application_path,
        geometry_episode=processed,
    )
    frame_zero = processed / "start_obj_pcd.ply"
    robot = processed / "robot/robot.npz"
    _require(
        frame_zero.is_file() and robot.is_file(),
        "V14 physical frame-zero geometry or action is missing",
    )

    source_repo = args.source_repo.resolve()
    official_repo = args.official_phystwin_repo.resolve()
    official_config = args.official_config.resolve()
    provenance = helpers._validate_runtime(
        source_repo,
        official_repo,
        official_config,
    )
    automatic_wrapper = (
        repo / "scripts/remote/"
        "build_deform360_causal_response_direct_depth_v14_automatic_twin.py"
    )
    pairwise_helpers = repo / "scripts/remote/run_deform360_fresh_pairwise_physical.py"
    provenance.update(
        {
            "bayesian_phystwin_revision": code_revision,
            "physical_runner_sha256": file_sha256(Path(__file__).resolve()),
            "automatic_twin_wrapper_sha256": file_sha256(automatic_wrapper),
            "pairwise_runner_helpers_sha256": file_sha256(pairwise_helpers),
        }
    )

    root = args.output_dir.resolve()
    _require(not root.exists(), "V14 physical work directory already exists")
    root.mkdir(parents=True)
    prediction_data = root / "prediction_only_input.pkl"
    prediction_summary = build_v14_prediction_only_bundle(
        frame_zero,
        robot,
        prediction_data,
        case_record=case_record,
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
        str(automatic_wrapper),
        "--source-repo",
        str(source_repo),
        "--prelock-protocol",
        str(prelock_path),
        "--queue",
        str(queue_path),
        "--queue-rank",
        str(case_record["queue_rank"]),
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
    twin_exit, twin_runtime = helpers._run_logged(
        twin_command,
        env=automatic_env,
        log_path=automatic_log,
    )
    runtimes: dict[str, float] = {"automatic_twin": twin_runtime}
    common_inputs: dict[str, Path] = {
        "staging_queue": queue_path,
        "geometry_protocol": parent_paths["geometry_protocol_file_sha256"],
        "runtime_v1": parent_paths["runtime_v1_file_sha256"],
        "validation_v1": parent_paths["validation_v1_file_sha256"],
        "runtime_v2": parent_paths["runtime_v2_file_sha256"],
        "validation_v2": parent_paths["validation_v2_file_sha256"],
        "geometry_manifest": manifest_path,
        "geometry_result": result_path,
        "runtime_application": runtime_application_path,
        "frame_zero_ply": frame_zero,
        "known_action": robot,
        "prediction_only_input": prediction_data,
        "prediction_only_summary": prediction_summary_path,
        "episode_graph": graph_path,
        "simulator_final_data": simulator_data,
        "state_artifact": state_path,
        "twin_summary": twin_summary_path,
        "automatic_twin_log": automatic_log,
    }
    admitted = twin_exit == 0
    if twin_exit not in {0, AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE}:
        raise RuntimeError(helpers._failure(twin_command, automatic_log, twin_exit))
    twin = _validate_twin_summary(
        twin_summary_path,
        protocol=protocol,
        case_record=case_record,
        prediction_data=prediction_data,
        graph_path=graph_path,
        simulator_data=simulator_data,
        state_path=state_path,
        passed=admitted,
    )
    vertices, springs, rest_lengths, anchors, weights = _load_graph_state(
        graph_path,
        state_path,
    )
    fallback_diagnostics: dict[str, Any] | None = None
    if not admitted:
        points, _ = load_frame_zero_ply(frame_zero)
        base = build_persistence_backbone_arrays(points)
        fallback_diagnostics = {
            "reason": "automatic_twin_source_admission_failed",
            "automatic_twin_exit_code": twin_exit,
            "automatic_twin_artifact_sha256": twin["artifact_sha256"],
            "automatic_twin_state_metrics": twin["state_metrics"],
            "warp_attempted": False,
        }
    else:
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
            returncode, elapsed = helpers._run_logged(
                command,
                env=official_env,
                log_path=log_path,
            )
            if returncode:
                raise RuntimeError(helpers._failure(command, log_path, returncode))
            runtimes[f"warp_{label}"] = elapsed
            result_paths[label] = rollout_dir / "official_phystwin_smoke.json"
        driven = helpers._load_warp_trajectory(
            result_paths["driven"],
            label="driven",
            scale=1.0,
            simulator_data=simulator_data,
            graph_path=graph_path,
            vertex_count=len(vertices),
        )
        zero = helpers._load_warp_trajectory(
            result_paths["zero_action"],
            label="zero_action",
            scale=0.0,
            simulator_data=simulator_data,
            graph_path=graph_path,
            vertex_count=len(vertices),
        )
        points, _ = load_frame_zero_ply(frame_zero)
        base = build_warp_backbone_arrays(
            points,
            vertices=vertices,
            springs=springs,
            rest_lengths=rest_lengths,
            contact_anchor_indices=anchors,
            readout_weights=weights,
            driven_vertices_m=driven,
            zero_action_vertices_m=zero,
        )
        for label, result_path in result_paths.items():
            common_inputs[f"{label}_result"] = result_path
            common_inputs[f"{label}_trajectory"] = result_path.with_name(
                "official_phystwin_trajectory.npz"
            )
    arrays = build_v14_physical_arrays(
        base,
        vertices=vertices,
        springs=springs,
        readout_weights=weights,
    )
    manifest = write_v14_physical_artifacts(
        root / "sealed_physical",
        arrays,
        prelock_protocol_path=prelock_path,
        case_record=case_record,
        physical_mode=(
            "warp_twin" if admitted else "automatic_twin_persistence_fallback"
        ),
        code_revision=code_revision,
        input_files=common_inputs,
        runtime_provenance={**provenance, "runtime_seconds": runtimes},
        fallback_diagnostics=fallback_diagnostics,
    )
    print(
        json.dumps(
            {
                "queue_rank": case_record["queue_rank"],
                "case_hash": case_record["case_hash"],
                "physical_mode": manifest["physical_mode"],
                "physical_artifact_sha256": manifest["artifact_sha256"],
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
