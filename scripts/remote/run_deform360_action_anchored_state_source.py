#!/usr/bin/env python3
"""Stage, predict, seal, or score one action-anchored Deform360 source case."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_action_anchored_state import (
    ActionAnchoredStateConfig,
    estimate_action_anchored_chain_state,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "predict", "seal", "evaluate"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--causal4d-root", type=Path)
    parser.add_argument("--official-phystwin-repo", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _result_sha256(value: dict[str, Any]) -> str:
    canonical = dict(value)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    output = dict(payload)
    output["result_sha256"] = _result_sha256(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if "result_sha256" in value:
        _require(
            value["result_sha256"] == _result_sha256(value),
            f"JSON result checksum differs: {path}",
        )
    return value


def _git_head(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _add_causal4d_source(root: Path) -> None:
    source = root.resolve() / "src"
    _require(source.is_dir(), f"Causal4D source directory is missing: {source}")
    sys.path.insert(0, str(source))


def _unpack_hulls(path: Path) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    with np.load(path, allow_pickle=False) as stored:
        frames = np.asarray(stored["frame_indices"], dtype=np.int32)
        offsets = np.asarray(stored["point_offsets"], dtype=np.int64)
        points = np.asarray(stored["points_world_m"], dtype=np.float64)
    _require(
        frames.ndim == 1
        and offsets.shape == (len(frames) + 1,)
        and offsets[0] == 0
        and offsets[-1] == len(points),
        "visual-hull archive has inconsistent offsets",
    )
    hulls = tuple(
        points[int(offsets[index]) : int(offsets[index + 1])].copy()
        for index in range(len(frames))
    )
    _require(
        all(
            hull.ndim == 2
            and hull.shape[1] == 3
            and len(hull) >= 21
            and np.all(np.isfinite(hull))
            for hull in hulls
        ),
        "visual-hull archive contains an invalid frame",
    )
    return frames, hulls


def _pack_hulls(
    path: Path,
    frames: np.ndarray,
    hulls: tuple[np.ndarray, ...],
) -> None:
    offsets = np.zeros(len(hulls) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(hull) for hull in hulls])
    points = np.concatenate(hulls, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_indices=np.asarray(frames, dtype=np.int32),
        point_offsets=offsets,
        points_world_m=points,
    )


def _case_paths(
    config: dict[str, Any],
    data_root: Path,
) -> tuple[Path, Path, Path]:
    case = config["case"]
    object_id = str(case["object_id"])
    episode_id = int(case["episode_id"])
    episode = data_root.resolve() / "aligned" / object_id / f"episode_{episode_id:04d}"
    hull = (
        data_root.resolve()
        / "observations"
        / object_id
        / f"episode_{episode_id:04d}"
        / "sampled_hulls.npz"
    )
    contact = data_root.resolve() / "observations" / object_id / "contact_model.json"
    return episode, hull, contact


def _stage(
    config: dict[str, Any],
    data_root: Path,
    output_root: Path,
) -> None:
    episode, source_hull_path, contact_model_path = _case_paths(config, data_root)
    _require(episode.is_dir(), "source episode is missing")
    _require(source_hull_path.is_file(), "source hull archive is missing")
    _require(contact_model_path.is_file(), "source contact model is missing")
    frames, hulls = _unpack_hulls(source_hull_path)
    required = np.asarray(config["case"]["prefix_hull_frames"], dtype=np.int32)
    indices = []
    for frame in required:
        matches = np.flatnonzero(frames == frame)
        _require(len(matches) == 1, f"required prefix hull is unavailable: {frame}")
        indices.append(int(matches[0]))
    _require(
        np.all(np.diff(required) > 0),
        "prefix hull frames must be strictly increasing",
    )
    prefix_hulls = tuple(hulls[index] for index in indices)
    archive = output_root.resolve() / "staged" / "prefix_hulls.npz"
    _pack_hulls(archive, required, prefix_hulls)
    robot = episode / "robot" / "robot.npz"
    manifest = {
        "artifact_kind": "Deform360ActionAnchoredPrefixCustody",
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "case": config["case"],
        "source": {
            "hull_archive_path": str(source_hull_path.resolve()),
            "hull_archive_sha256": _file_sha256(source_hull_path),
            "robot_sha256": _file_sha256(robot),
            "contact_model_sha256": _file_sha256(contact_model_path),
        },
        "staged_prefix": {
            "archive_path": str(archive),
            "archive_sha256": _file_sha256(archive),
            "frame_indices": required.astype(int).tolist(),
            "point_counts": [len(hull) for hull in prefix_hulls],
            "points_sha256": [_array_sha256(hull) for hull in prefix_hulls],
        },
        "custody_boundary": {
            "trusted_stager_read_source_archive": True,
            "prediction_receives_only_staged_prefix_hulls": True,
            "future_geometry_values_written_to_prediction_stage": False,
            "source_outcomes_previously_open": True,
            "claim": "post-open source development only",
        },
    }
    _write_json(output_root.resolve() / "staged" / "manifest.json", manifest)


def _robot_arrays(state: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    openings = np.asarray(state.openings, dtype=np.float64)
    transforms = np.asarray(state.T_worlds, dtype=np.float64)
    if openings.ndim == 1:
        openings = openings[:, None]
    if transforms.ndim == 3:
        transforms = transforms[:, None]
    _require(
        openings.ndim == 2 and transforms.shape == (*openings.shape, 4, 4),
        "robot arrays have unexpected shape",
    )
    return openings, transforms, np.asarray(state.actions)


def _previous_controller_positions(
    state: Any,
    previous_frame: int,
    associations: tuple[dict[str, Any], ...],
) -> np.ndarray:
    from deform360.processing.control_points_stage import gripper_taxel_points

    openings, transforms, _ = _robot_arrays(state)
    positions = []
    for association in associations:
        axis = int(association["robot_axis"])
        selected = np.asarray(
            association["selected_taxel_indices"],
            dtype=np.int64,
        )
        taxels = gripper_taxel_points(
            float(openings[previous_frame, axis]),
            transforms[previous_frame, axis],
        )
        offset = np.asarray(association["contact_offset_m"], dtype=np.float64)
        positions.append(np.mean(taxels[selected], axis=0) + offset)
    return np.asarray(positions, dtype=np.float64)


def _prediction_case_with_velocity(case: Any, velocity: np.ndarray) -> Any:
    from causal4d_public.deform360_replication_warp import Deform360WarpForecastCase

    return Deform360WarpForecastCase(
        episode_id=case.episode_id,
        graph=case.graph,
        controller_positions_m=case.controller_positions_m,
        contact_active=case.contact_active,
        contact_node_indices=case.contact_node_indices,
        contact_rest_lengths_m=case.contact_rest_lengths_m,
        dt_seconds=case.dt_seconds,
        initial_velocities_m_s=velocity,
    )


def _predict(
    config: dict[str, Any],
    data_root: Path,
    causal4d_root: Path,
    official_phystwin_repo: Path,
    output_root: Path,
    device: str,
) -> None:
    _add_causal4d_source(causal4d_root)
    from causal4d_public.deform360_phystwin_feasibility import (
        WarpRopeCandidate,
        WarpRopeFeasibilityConfig,
    )
    from causal4d_public.deform360_replication_case import (
        build_replication_warp_observation,
    )
    from causal4d_public.deform360_replication_contact import causal_confirmed
    from causal4d_public.deform360_replication_graph import (
        build_sparse_graph_for_stratum,
    )
    from causal4d_public.deform360_replication_warp import (
        OfficialWarpSparseGraphRunner,
    )
    from deform360.robot import load_robot_state

    root = output_root.resolve()
    custody_path = root / "staged" / "manifest.json"
    custody = _load_json(custody_path)
    prefix_path = Path(custody["staged_prefix"]["archive_path"])
    _require(
        _file_sha256(prefix_path) == custody["staged_prefix"]["archive_sha256"],
        "staged prefix archive differs from custody manifest",
    )
    frames, hulls = _unpack_hulls(prefix_path)
    required = np.asarray(config["case"]["prefix_hull_frames"], dtype=np.int32)
    _require(np.array_equal(frames, required), "staged prefix frames differ")
    previous_frame, branch_frame = map(int, frames)
    previous_hull, current_hull = hulls

    episode, _, contact_model_path = _case_paths(config, data_root)
    state = load_robot_state(episode / "robot" / "robot.npz")
    openings, _, _ = _robot_arrays(state)
    contact_payload = json.loads(contact_model_path.read_text(encoding="utf-8"))
    threshold = float(contact_payload["opening_threshold_m"])
    confirmation = int(contact_payload["confirmation_frames"])
    visual_schedule = np.column_stack(
        [
            causal_confirmed(openings[:, axis] <= threshold, confirmation)
            for axis in range(openings.shape[1])
        ]
    )
    observation = build_replication_warp_observation(
        episode,
        f"{config['case']['object_id']}/episode_{config['case']['episode_id']:04d}",
        "filament",
        [branch_frame],
        [current_hull],
        visual_schedule,
        dt_seconds=float(config["state_estimator"]["dt_seconds"]),
    )
    previous_graph = build_sparse_graph_for_stratum(previous_hull, "filament")
    current_graph = observation.case.graph
    _require(
        len(previous_graph.positions_m) == len(current_graph.positions_m),
        "prefix graph node counts differ",
    )
    previous_controllers = _previous_controller_positions(
        state,
        previous_frame,
        observation.contact_associations,
    )
    current_controllers = observation.case.controller_positions_m[0]
    anchor_indices = np.asarray(
        observation.case.contact_node_indices,
        dtype=np.int64,
    )
    state_config = ActionAnchoredStateConfig(**config["state_estimator"]["config"])
    estimate = estimate_action_anchored_chain_state(
        previous_graph.positions_m,
        current_graph.positions_m,
        previous_controllers,
        current_controllers,
        anchor_indices,
        dt_seconds=float(config["state_estimator"]["dt_seconds"]),
        config=state_config,
    )
    zero = np.zeros_like(current_graph.positions_m)
    candidate_velocities = {
        "physical_zero_velocity": zero,
        "camera_topology_velocity": estimate.camera_smoothed_velocity_m_s,
        "action_harmonic_velocity": estimate.action_harmonic_velocity_m_s,
        "bias_aware_action_anchored_velocity": (
            estimate.bias_corrected_action_velocity_m_s if estimate.accepted else zero
        ),
    }
    routes = {
        name: (
            "candidate"
            if name != "bias_aware_action_anchored_velocity" or estimate.accepted
            else "exact-zero-velocity-fallback"
        )
        for name in candidate_velocities
    }

    index = int(config["physical_model"]["leave_one_source_candidate_index"])
    parameters = dict(config["physical_model"]["candidate_parameters"])
    candidate = WarpRopeCandidate(**parameters)
    feasibility = WarpRopeFeasibilityConfig(
        **config["physical_model"]["feasibility_config"]
    )

    trajectories: dict[str, np.ndarray] = {}
    for name, velocity in candidate_velocities.items():
        case = _prediction_case_with_velocity(observation.case, velocity)
        runner = OfficialWarpSparseGraphRunner(
            official_phystwin_repo,
            case,
            feasibility,
            device=device,
        )
        trajectories[name] = runner.rollout(candidate)
        del runner
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    trajectories["exact_persistence"] = np.repeat(
        current_graph.positions_m[None],
        len(observation.case.controller_positions_m),
        axis=0,
    )

    prediction_path = root / "predictions" / "prediction.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        prediction_path,
        **{f"trajectory__{name}": value for name, value in trajectories.items()},
        **{
            f"initial_velocity__{name}": value
            for name, value in candidate_velocities.items()
        },
        graph_positions_m=current_graph.positions_m,
        graph_spring_edges=current_graph.spring_edges,
        graph_spring_families=current_graph.spring_families,
        graph_masses=current_graph.masses,
        branch_frame=np.asarray(branch_frame, dtype=np.int32),
    )
    report = {
        "artifact_kind": "Deform360ActionAnchoredStatePrediction",
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "case": config["case"],
        "implementation": {
            "bayesian_phystwin_commit": _git_head(Path(__file__).resolve().parents[2]),
            "causal4d_commit": _git_head(causal4d_root.resolve()),
            "official_phystwin_commit": _git_head(official_phystwin_repo.resolve()),
            "trackdeform3d_reference_commit": config["references"][
                "trackdeform3d_commit"
            ],
            "adapter_scope": (
                "new Bayesian-PhysTwin endpoint-state adapter; no upstream "
                "TrackDeform3D code is vendored or claimed as evaluated"
            ),
        },
        "prefix": {
            "custody_manifest_sha256": _file_sha256(custody_path),
            "staged_archive_sha256": _file_sha256(prefix_path),
            "frames": frames.astype(int).tolist(),
            "branch_frame": branch_frame,
        },
        "physical_model": {
            "source_lock": {
                "pooled_fit_relative_path": config["physical_model"][
                    "pooled_fit_relative_path"
                ],
                "pooled_fit_file_sha256": config["physical_model"][
                    "pooled_fit_file_sha256"
                ],
                "pooled_fit_result_sha256": config["physical_model"][
                    "pooled_fit_result_sha256"
                ],
                "source_grid_file_sha256": config["physical_model"][
                    "source_grid_file_sha256"
                ],
            },
            "leave_one_source_candidate_index": index,
            "parameters": parameters,
            "feasibility_config": asdict(feasibility),
        },
        "state_estimate": {
            "config": asdict(state_config),
            "accepted": estimate.accepted,
            "routes": routes,
            "diagnostics": estimate.diagnostics,
            "initial_velocity_sha256": {
                name: _array_sha256(value)
                for name, value in candidate_velocities.items()
            },
        },
        "contact_associations": list(observation.contact_associations),
        "prediction": {
            "archive_path": str(prediction_path),
            "archive_sha256": _file_sha256(prediction_path),
            "arms": sorted(trajectories),
            "full_rate_frame_count": len(observation.case.controller_positions_m),
        },
        "information_boundary": {
            "prediction_read_staged_prefix_hulls_only": True,
            "known_future_robot_trajectory_read": True,
            "future_geometry_read": False,
            "future_tactile_read": False,
            "source_metrics_read": False,
            "score_bearing_fit_or_grid_artifact_read": False,
            "held_v8_read": False,
            "fresh_target_read": False,
        },
    }
    _write_json(root / "predictions" / "prediction.json", report)


def _seal(config: dict[str, Any], output_root: Path) -> None:
    root = output_root.resolve()
    report_path = root / "predictions" / "prediction.json"
    report = _load_json(report_path)
    archive = Path(report["prediction"]["archive_path"])
    _require(
        _file_sha256(archive) == report["prediction"]["archive_sha256"],
        "prediction archive differs from report",
    )
    seal = {
        "artifact_kind": "Deform360ActionAnchoredStatePredictionSeal",
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "case": config["case"],
        "prediction_report_path": str(report_path),
        "prediction_report_sha256": _file_sha256(report_path),
        "prediction_archive_path": str(archive),
        "prediction_archive_sha256": _file_sha256(archive),
        "outcome_open_authorized_after_seal": True,
        "claim": "post-open source development; seal prevents same-run outcome tuning",
    }
    _write_json(root / "prediction_seal.json", seal)


def _relative_improvement(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / baseline


def _evaluate(
    config: dict[str, Any],
    data_root: Path,
    causal4d_root: Path,
    output_root: Path,
    output: Path,
) -> None:
    _add_causal4d_source(causal4d_root)
    from causal4d_public.deform360_replication_graph import Deform360SparseGraph
    from causal4d_public.deform360_replication_warp import (
        sparse_graph_strain_summary,
        sparse_trajectory_chamfer_m,
    )

    root = output_root.resolve()
    seal_path = root / "prediction_seal.json"
    seal = _load_json(seal_path)
    report_path = Path(seal["prediction_report_path"])
    archive_path = Path(seal["prediction_archive_path"])
    _require(
        _file_sha256(report_path) == seal["prediction_report_sha256"]
        and _file_sha256(archive_path) == seal["prediction_archive_sha256"],
        "sealed prediction differs",
    )
    report = _load_json(report_path)
    _, source_hull_path, _ = _case_paths(config, data_root)
    frames, hulls = _unpack_hulls(source_hull_path)
    branch_frame = int(report["prefix"]["branch_frame"])
    future_mask = frames > branch_frame
    future_frames = frames[future_mask]
    future_hulls = tuple(hull for hull, keep in zip(hulls, future_mask) if keep)
    _require(
        len(future_frames) >= 2 and int(future_frames[0]) > branch_frame,
        "source outcome has no untouched future",
    )
    with np.load(archive_path, allow_pickle=False) as stored:
        graph = Deform360SparseGraph(
            positions_m=np.asarray(stored["graph_positions_m"], dtype=np.float64),
            spring_edges=np.asarray(stored["graph_spring_edges"], dtype=np.int32),
            spring_families=np.asarray(
                stored["graph_spring_families"],
                dtype=np.int8,
            ),
            masses=np.asarray(stored["graph_masses"], dtype=np.float64),
            stratum="filament",
            diagnostics={"construction": "sealed prefix graph"},
        )
        arms = {
            name.removeprefix("trajectory__"): np.asarray(
                stored[name],
                dtype=np.float64,
            )
            for name in stored.files
            if name.startswith("trajectory__")
        }
    relative = future_frames - branch_frame
    scores = {}
    for name, trajectory in arms.items():
        _require(int(relative[-1]) < len(trajectory), f"{name} ends before outcome")
        selected = trajectory[relative]
        score = sparse_trajectory_chamfer_m(future_hulls, selected)
        score["strain"] = sparse_graph_strain_summary(graph, trajectory)
        scores[name] = score

    fused_name = "bias_aware_action_anchored_velocity"
    physical_name = "physical_zero_velocity"
    persistence_name = "exact_persistence"
    fused = scores[fused_name]
    physical = scores[physical_name]
    persistence = scores[persistence_name]
    gains = {
        "fused_vs_physical_mean": _relative_improvement(
            float(fused["mean_m"]),
            float(physical["mean_m"]),
        ),
        "fused_vs_persistence_mean": _relative_improvement(
            float(fused["mean_m"]),
            float(persistence["mean_m"]),
        ),
        "fused_vs_physical_late": _relative_improvement(
            float(fused["late_mean_m"]),
            float(physical["late_mean_m"]),
        ),
        "fused_vs_persistence_late": _relative_improvement(
            float(fused["late_mean_m"]),
            float(persistence["late_mean_m"]),
        ),
    }
    gate = config["promotion_gate"]
    checks = {
        "state_gate_accepted": bool(report["state_estimate"]["accepted"]),
        "mean_improvement_vs_physical": (
            gains["fused_vs_physical_mean"]
            >= float(gate["minimum_mean_improvement_fraction"])
        ),
        "mean_improvement_vs_persistence": (
            gains["fused_vs_persistence_mean"]
            >= float(gate["minimum_mean_improvement_fraction"])
        ),
        "late_nonregression_vs_physical": (
            gains["fused_vs_physical_late"]
            >= -float(gate["maximum_late_degradation_fraction"])
        ),
        "late_nonregression_vs_persistence": (
            gains["fused_vs_persistence_late"]
            >= -float(gate["maximum_late_degradation_fraction"])
        ),
        "physical_strain_gate": (
            float(fused["strain"]["p99"])
            <= float(gate["maximum_p99_relative_edge_strain"])
        ),
    }
    passed = all(checks.values())
    result = {
        "artifact_kind": "Deform360ActionAnchoredStateSourceResult",
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "case": config["case"],
        "prediction_seal_sha256": _file_sha256(seal_path),
        "outcome": {
            "source_hull_archive_sha256": _file_sha256(source_hull_path),
            "future_frame_indices": future_frames.astype(int).tolist(),
            "future_frame_count": len(future_frames),
        },
        "scores": scores,
        "relative_improvements": gains,
        "promotion_gate": {
            "criteria": gate,
            "checks": checks,
            "passed": passed,
        },
        "decision": (
            "justify-multi-episode-source-panel"
            if passed
            else "stop-action-anchored-state-route"
        ),
        "claim_boundary": {
            "post_open_source_development": True,
            "single_object_single_episode": True,
            "upstream_trackdeform3d_evaluated": False,
            "object_level_transfer_established": False,
            "state_of_the_art_claim_authorized": False,
            "held_v8_read": False,
        },
    }
    _write_json(output.resolve(), result)


def main() -> int:
    args = _parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    output_root = args.output_root.resolve()
    if args.phase == "stage":
        _require(args.data_root is not None, "--data-root is required for stage")
        _stage(config, args.data_root, output_root)
    elif args.phase == "predict":
        _require(args.data_root is not None, "--data-root is required for predict")
        _require(
            args.causal4d_root is not None,
            "--causal4d-root is required for predict",
        )
        _require(
            args.official_phystwin_repo is not None,
            "--official-phystwin-repo is required for predict",
        )
        _predict(
            config,
            args.data_root,
            args.causal4d_root,
            args.official_phystwin_repo,
            output_root,
            args.device,
        )
    elif args.phase == "seal":
        _seal(config, output_root)
    else:
        _require(args.data_root is not None, "--data-root is required for evaluate")
        _require(
            args.causal4d_root is not None,
            "--causal4d-root is required for evaluate",
        )
        _require(args.output is not None, "--output is required for evaluate")
        _evaluate(
            config,
            args.data_root,
            args.causal4d_root,
            output_root,
            args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
