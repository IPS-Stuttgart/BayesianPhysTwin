#!/usr/bin/env python3
"""Run and seal one dynamic-protocol Deform360 physical backbone."""

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

from bayesian_phystwin.deform360_dynamic_tapnextpp_cohort import (
    dynamic_provider_case_record,
    load_dynamic_provider_cohort_lock,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_physical import (
    build_readout_graph_basis,
    write_dynamic_physical_artifacts,
)
from bayesian_phystwin.deform360_fresh_pairwise_physical import (
    AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE,
    CANONICAL_NODE_COUNT,
    WARP_DYNAMICS,
    build_persistence_backbone_arrays,
    build_prediction_only_bundle,
    build_warp_backbone_arrays,
    load_frame_zero_ply,
)
from bayesian_phystwin.deform360_fresh_source_lock import (
    validate_fresh_source_admission,
)
from bayesian_phystwin.deform360_object_exclusion import file_sha256
from bayesian_phystwin.tapnextpp_dynamic_multiview import PROTOCOL_ID


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
        "_dynamic_tapnextpp_pairwise_physical_helpers",
        path,
    )
    _require(spec is not None and spec.loader is not None, "cannot load runner helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_protocol(path: Path, cohort: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _load_json(path)
    _require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol ID changed")
    _require(
        file_sha256(path)
        == cohort["bindings"]["provider_protocol_file_sha256"],
        "protocol differs from cohort lock",
    )
    return protocol


def _validate_twin_summary(
    path: Path,
    *,
    protocol_sha256: str,
    cohort: Mapping[str, Any],
    cohort_path: Path,
    case_record: Mapping[str, Any],
    partition: str,
    prediction_data: Path,
    graph_path: Path,
    simulator_data: Path,
    state_path: Path,
    passed: bool,
) -> dict[str, Any]:
    summary = _load_json(path)
    _require(
        summary.get("artifact_kind")
        == "Deform360DynamicTAPNextPPAutomaticEpisodeTwin"
        and summary.get("protocol_id") == PROTOCOL_ID
        and summary.get("provider_protocol_file_sha256") == protocol_sha256
        and summary.get("cohort_lock_sha256") == cohort["cohort_lock_sha256"]
        and summary.get("cohort_lock_file_sha256") == file_sha256(cohort_path)
        and summary.get("partition") == partition,
        "dynamic automatic-twin summary is incompatible",
    )
    _require(
        all(summary.get(key) == value for key, value in case_record.items()),
        "dynamic automatic-twin identity changed",
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
        and boundary.get("future_object_tracks_present") is False
        and boundary.get(
            "held_v8_target_query_score_barrier_or_outcome_access"
        )
        is False,
        "automatic twin crossed its prediction boundary",
    )
    expected = dict(summary)
    result_sha256 = expected.pop("result_sha256", None)
    canonical = hashlib.sha256(
        json.dumps(
            expected,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _require(result_sha256 == canonical, "automatic-twin summary checksum changed")
    return summary


def _load_graph_state(
    graph_path: Path,
    state_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
    return vertices, springs, rest_lengths, anchors, weights


def _dynamic_arrays(
    base: Mapping[str, np.ndarray],
    *,
    vertices: np.ndarray,
    springs: np.ndarray,
    readout_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "action_support": np.asarray(base["action_support"]),
        "driven_readout_m": np.asarray(base["driven_readout_m"]),
        "frame_zero_points_m": np.asarray(base["frame_zero_points_m"]),
        "graph_basis": build_readout_graph_basis(
            vertices,
            springs,
            readout_weights,
        ).astype(np.float32),
        "persistence_prediction_m": np.asarray(base["persistence_m"]),
        "physical_prediction_m": np.asarray(base["prediction_m"]),
        "zero_action_readout_m": np.asarray(base["zero_action_readout_m"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cohort-lock", type=Path, required=True)
    parser.add_argument("--partition", choices=("source", "target"), required=True)
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
    helpers = _load_runner_helpers(repo)
    code_revision = helpers._require_clean_repository(repo)
    cohort_path = args.cohort_lock.resolve()
    cohort = load_dynamic_provider_cohort_lock(cohort_path)
    protocol_path = args.protocol.resolve()
    _load_protocol(protocol_path, cohort)
    admission_path = args.admission.resolve()
    admission = _load_json(admission_path)
    validate_fresh_source_admission(admission)
    _require(admission["accepted"] is True, "source admission did not pass")
    record = dynamic_provider_case_record(
        cohort,
        object_id=str(admission["object_id"]),
        episode_id=int(admission["episode_id"]),
        partition=args.partition,
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
    control = _load_json(control_meta)
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
    provenance = helpers._validate_runtime(
        source_repo,
        official_repo,
        official_config,
    )
    dynamic_wrapper = (
        repo
        / "scripts/remote/build_deform360_dynamic_tapnextpp_automatic_twin.py"
    )
    pairwise_helpers = (
        repo / "scripts/remote/run_deform360_fresh_pairwise_physical.py"
    )
    provenance.update(
        {
            "bayesian_phystwin_revision": code_revision,
            "physical_runner_sha256": file_sha256(Path(__file__).resolve()),
            "automatic_twin_wrapper_sha256": file_sha256(dynamic_wrapper),
            "pairwise_runner_helpers_sha256": file_sha256(pairwise_helpers),
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
        str(dynamic_wrapper),
        "--source-repo",
        str(source_repo),
        "--protocol",
        str(protocol_path),
        "--cohort-lock",
        str(cohort_path),
        "--partition",
        args.partition,
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
    twin_exit, twin_runtime = helpers._run_logged(
        twin_command,
        env=automatic_env,
        log_path=automatic_log,
    )
    runtimes: dict[str, float] = {"automatic_twin": twin_runtime}
    common_inputs: dict[str, Path] = {
        "source_admission": admission_path,
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
    admitted = twin_exit == 0
    if twin_exit not in {0, AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE}:
        raise RuntimeError(helpers._failure(twin_command, automatic_log, twin_exit))
    twin = _validate_twin_summary(
        twin_summary_path,
        protocol_sha256=file_sha256(protocol_path),
        cohort=cohort,
        cohort_path=cohort_path,
        case_record=record,
        partition=args.partition,
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
            "automatic_twin_result_sha256": twin["result_sha256"],
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
    arrays = _dynamic_arrays(
        base,
        vertices=vertices,
        springs=springs,
        readout_weights=weights,
    )
    physical_manifest = write_dynamic_physical_artifacts(
        root / "sealed_physical",
        arrays,
        protocol_path=protocol_path,
        cohort_lock_path=cohort_path,
        case_record=record,
        partition=args.partition,
        physical_mode=(
            "warp_twin"
            if admitted
            else "source_admission_persistence_fallback"
        ),
        code_revision=code_revision,
        input_files=common_inputs,
        runtime_provenance={**provenance, "runtime_seconds": runtimes},
        fallback_diagnostics=fallback_diagnostics,
    )
    print(
        json.dumps(
            {
                "case": record["case"],
                "partition": args.partition,
                "physical_mode": physical_manifest["physical_mode"],
                "physical_manifest_sha256": physical_manifest["result_sha256"],
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
