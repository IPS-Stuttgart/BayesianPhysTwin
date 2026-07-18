#!/usr/bin/env python3
"""Seal every locked physical candidate before a held outcome is revealed."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from causal4d_public.deform360_reusable_sota_method import (
    load_reusable_sota_method,
    reusable_sota_physical_candidates,
)
from causal4d_public.deform360_reusable_sota_protocol import (
    load_reusable_sota_config,
)
from causal4d_public.deform360_reusable_sota_window import (
    authorize_development_held_prediction_window,
    load_reusable_sota_window,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--window-addendum", type=Path, required=True)
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--object-id", required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--prediction-input", type=Path, required=True)
    parser.add_argument("--prediction-summary", type=Path, required=True)
    parser.add_argument("--simulator-final-data", type=Path, required=True)
    parser.add_argument("--episode-graph", type=Path, required=True)
    parser.add_argument("--state-artifact", type=Path, required=True)
    parser.add_argument("--twin-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _validate_summary(
    path: Path,
    *,
    artifact_kind: str,
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == artifact_kind
        and payload.get("result_sha256") == _canonical_sha256(payload)
        and payload.get("object_id") == object_id
        and int(payload.get("episode_id", -1)) == episode_id
        and payload.get("passed") is True,
        f"incompatible {artifact_kind} summary",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("post_initial_object_observation_used") is False
        and boundary.get("target_access") is False,
        f"{artifact_kind} crossed the held information boundary",
    )
    return payload


def _run_rollout(
    *,
    runner: Path,
    official_repo: Path,
    data_path: Path,
    graph_path: Path,
    real_config: Path,
    output_dir: Path,
    device: str,
    candidate: Mapping[str, Any],
    controller_scale: float,
    warp: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = output_dir / "official_phystwin_smoke.json"
    trajectory_path = output_dir / "official_phystwin_trajectory.npz"
    if not result_path.is_file():
        command = [
            sys.executable,
            str(runner),
            "--official-phystwin-repo",
            str(official_repo),
            "--data",
            str(data_path),
            "--config",
            str(real_config),
            "--output-dir",
            str(output_dir),
            "--canonical-reusable-graph",
            str(graph_path),
            "--device",
            device,
            "--controller-radius-m",
            str(warp["controller_radius_m"]),
            "--controller-max-neighbours",
            str(warp["controller_max_neighbours"]),
            "--canonical-controller-patch-size",
            str(warp["canonical_controller_patch_size"]),
            "--init-spring-y",
            str(candidate["init_spring_y"]),
            "--drag-damping",
            str(candidate["drag_damping"]),
            "--dashpot-damping",
            str(candidate["dashpot_damping"]),
            "--controller-displacement-scale",
            str(controller_scale),
            "--support-dynamics",
            str(warp["support_dynamics"]),
            "--report-edge-strain",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                f"Warp rollout failed for {candidate['label']} at scale "
                f"{controller_scale}\nSTDOUT:\n{completed.stdout[-4000:]}\n"
                f"STDERR:\n{completed.stderr[-4000:]}"
            )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected_overrides = {
        "controller_max_neighbours": int(warp["controller_max_neighbours"]),
        "controller_radius": float(warp["controller_radius_m"]),
        "dashpot_damping": float(candidate["dashpot_damping"]),
        "drag_damping": float(candidate["drag_damping"]),
        "init_spring_Y": float(candidate["init_spring_y"]),
    }
    _require(
        result.get("passed") is True
        and result.get("official_phystwin_revision") == warp["revision"]
        and result.get("config_sha256") == warp["real_config_sha256"]
        and result.get("config_overrides") == expected_overrides
        and result.get("data_sha256") == _sha256_file(data_path)
        and result.get("trajectory_sha256") == _sha256_file(trajectory_path)
        and float(
            result.get("realized_actuation", {}).get(
                "controller_displacement_scale", -1.0
            )
        )
        == controller_scale,
        f"incompatible Warp rollout for {candidate['label']}",
    )
    return {
        "result_path": result_path,
        "result_sha256": _sha256_file(result_path),
        "trajectory_path": trajectory_path,
        "trajectory_sha256": result["trajectory_sha256"],
    }


def main() -> int:
    args = _parse_args()
    protocol = load_reusable_sota_config(args.protocol)
    window = load_reusable_sota_window(args.window_addendum)
    method = load_reusable_sota_method(args.method)
    authorization = authorize_development_held_prediction_window(
        protocol,
        window,
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    _require(
        method["config"]["parent_config_sha256"] == protocol["config_sha256"]
        and method["config"]["window_config_sha256"] == window["config_sha256"],
        "method lock belongs to another protocol",
    )
    prediction_summary = _validate_summary(
        args.prediction_summary,
        artifact_kind="Deform360PredictionOnlyInput",
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    twin_summary = _validate_summary(
        args.twin_summary,
        artifact_kind="Deform360AutomaticEpisodeTwin",
        object_id=args.object_id,
        episode_id=args.episode_id,
    )
    _require(
        prediction_summary.get("sota_authorization", {}).get("window")
        == authorization
        and twin_summary.get("sota_authorization", {}).get("window")
        == authorization,
        "held inputs use another SOTA authorization",
    )
    _require(
        prediction_summary.get("output_sha256")
        == _sha256_file(args.prediction_input)
        and twin_summary.get("output_sha256", {}).get("simulator_final_data")
        == _sha256_file(args.simulator_final_data)
        and twin_summary.get("output_sha256", {}).get("episode_graph")
        == _sha256_file(args.episode_graph)
        and twin_summary.get("output_sha256", {}).get("state_artifact")
        == _sha256_file(args.state_artifact),
        "held input hashes changed",
    )

    warp = method["config"]["official_warp"]
    real_config = args.official_phystwin_repo / "configs" / "real.yaml"
    _require(
        _git_revision(args.official_phystwin_repo) == warp["revision"]
        and _sha256_file(real_config) == warp["real_config_sha256"],
        "official PhysTwin checkout changed",
    )
    with args.prediction_input.open("rb") as stream:
        prediction_input = pickle.load(stream)
    persistence = np.asarray(prediction_input["object_points"], dtype=np.float32)
    with np.load(args.state_artifact, allow_pickle=False) as archive:
        readout_weights = np.asarray(archive["readout_weights"], dtype=np.float64)
    _require(
        persistence.shape[0] == method["config"]["prediction_bank"]["frame_count"]
        and readout_weights.shape[0] == persistence.shape[1],
        "held readout shape changed",
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    runner = args.repo / "scripts/remote/run_deform360_official_phystwin_smoke.py"
    records = []
    for candidate in reusable_sota_physical_candidates(method):
        candidate_root = args.output_root / candidate["label"]
        driven = _run_rollout(
            runner=runner,
            official_repo=args.official_phystwin_repo,
            data_path=args.simulator_final_data,
            graph_path=args.episode_graph,
            real_config=real_config,
            output_dir=candidate_root / "driven",
            device=args.device,
            candidate=candidate,
            controller_scale=1.0,
            warp=warp,
        )
        zero = _run_rollout(
            runner=runner,
            official_repo=args.official_phystwin_repo,
            data_path=args.simulator_final_data,
            graph_path=args.episode_graph,
            real_config=real_config,
            output_dir=candidate_root / "zero",
            device=args.device,
            candidate=candidate,
            controller_scale=0.0,
            warp=warp,
        )
        with np.load(driven["trajectory_path"], allow_pickle=False) as archive:
            driven_graph = np.asarray(archive["vertices"], dtype=np.float64)
        with np.load(zero["trajectory_path"], allow_pickle=False) as archive:
            zero_graph = np.asarray(archive["vertices"], dtype=np.float64)
        _require(
            driven_graph.shape == zero_graph.shape
            and driven_graph.shape[0] == persistence.shape[0]
            and driven_graph.shape[1] == readout_weights.shape[1]
            and np.all(np.isfinite(driven_graph))
            and np.all(np.isfinite(zero_graph)),
            f"invalid graph response for {candidate['label']}",
        )
        graph_response = driven_graph - zero_graph
        dense_response = np.einsum(
            "mn,tnc->tmc", readout_weights, graph_response, optimize=True
        )
        dense_prediction = persistence.astype(np.float64) + dense_response
        _require(
            np.all(np.isfinite(dense_prediction)),
            f"nonfinite dense prediction for {candidate['label']}",
        )
        prediction_path = candidate_root / "prediction.npz"
        np.savez_compressed(
            prediction_path,
            prediction_m=dense_prediction.astype(np.float32),
            persistence_m=persistence,
            graph_response_m=graph_response.astype(np.float32),
            candidate_label=np.asarray(candidate["label"]),
        )
        records.append(
            {
                **candidate,
                "driven_result_sha256": driven["result_sha256"],
                "driven_trajectory_sha256": driven["trajectory_sha256"],
                "zero_result_sha256": zero["result_sha256"],
                "zero_trajectory_sha256": zero["trajectory_sha256"],
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": _sha256_file(prediction_path),
                "maximum_dense_response_m": float(
                    np.max(np.linalg.norm(dense_response, axis=-1))
                ),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableSotaHeldPredictionBank",
        "protocol_id": method["config"]["protocol_id"],
        "method_config_sha256": method["config_sha256"],
        "object_id": args.object_id,
        "episode_id": args.episode_id,
        "authorization": authorization,
        "candidate_count": len(records),
        "records": records,
        "input_sha256": {
            "prediction_input": _sha256_file(args.prediction_input),
            "prediction_summary": _sha256_file(args.prediction_summary),
            "simulator_final_data": _sha256_file(args.simulator_final_data),
            "episode_graph": _sha256_file(args.episode_graph),
            "state_artifact": _sha256_file(args.state_artifact),
            "twin_summary": _sha256_file(args.twin_summary),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_robot_action_used": True,
            "future_object_outcome_read": False,
            "future_tactile_read": False,
            "candidate_selection_performed": False,
            "all_locked_candidates_sealed": True,
            "confirmatory_object_read": False,
        },
        "passed": len(records) == 18,
        "claim_boundary": (
            "development held prediction bank only; candidate selection remains "
            "source-only and no Deform360 Table 4 parity is claimed"
        ),
    }
    manifest["result_sha256"] = _canonical_sha256(manifest)
    manifest_path = args.output_root / "prediction_bank.json"
    _require(not manifest_path.exists(), f"prediction bank exists: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": manifest["passed"],
                "object_id": args.object_id,
                "episode_id": args.episode_id,
                "candidate_count": len(records),
                "result_sha256": manifest["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
