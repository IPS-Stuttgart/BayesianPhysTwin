#!/usr/bin/env python3
"""Replay target-excluded MatPhys spring fields in official PhysTwin Warp."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.matphys_warp_ensemble_v1 import (
    MATPHYS_WARP_ENSEMBLE_PROTOCOL,
    MATPHYS_WARP_ENSEMBLE_SCHEMA,
    MATPHYS_WARP_ENSEMBLE_VERSION,
    file_sha256,
    hierarchical_trajectory_ensemble_arrays,
    load_matphys_spring_ensemble,
    load_registered_replay_graph,
)

EXPECTED_OFFICIAL_WARP_VERSION = "1.16.0"
EXPECTED_REFERENCE_RUNNER_SHA256 = (
    "e7bf6a6c06e074ac3cdefe259c1cf5eecf8cd905dae1b710a81107ab166ca535"
)
EXPECTED_REPLAY_RUNTIME = {
    "python_version": "3.10.20",
    "numpy_version": "1.26.4",
    "torch_version": "2.4.0+cu121",
    "torch_cuda_version": "12.1",
    "warp_version": EXPECTED_OFFICIAL_WARP_VERSION,
}


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _validate_warp_runtime(observed_version: str) -> str:
    _require(
        observed_version == EXPECTED_OFFICIAL_WARP_VERSION,
        "official Warp runtime version changed",
    )
    return observed_version


def _validate_replay_runtime(observed: dict[str, str]) -> dict[str, str]:
    _require(observed == EXPECTED_REPLAY_RUNTIME, "official replay runtime changed")
    return observed


def _validate_independent_reference(
    *,
    result_path: Path,
    trajectory_path: Path,
    runner_path: Path,
    data_path: Path,
    config_sha256: str,
    official_revision: str,
    registered_graph_path: Path,
    controller_max_neighbours: int,
    controller_radius_m: float,
    controller_patch_size: int,
    init_spring_y: float,
    drag_damping: float,
    dashpot_damping: float,
) -> dict[str, Any]:
    _require(
        file_sha256(runner_path) == EXPECTED_REFERENCE_RUNNER_SHA256,
        "official independent-reference runner changed",
    )
    _require(
        result_path.resolve(strict=True).with_name("official_phystwin_trajectory.npz")
        == trajectory_path.resolve(strict=True),
        "independent-reference result and trajectory are not colocated",
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(
        result.get("passed") is True
        and result.get("source_only_smoke") is True
        and "external_target_scoring" not in result
        and result.get("official_phystwin_revision") == official_revision
        and result.get("config_sha256") == config_sha256
        and result.get("data_sha256") == file_sha256(data_path)
        and result.get("trajectory_sha256") == file_sha256(trajectory_path),
        "independent official reference changed",
    )
    _require(
        result.get("config_overrides")
        == {
            "controller_max_neighbours": controller_max_neighbours,
            "controller_radius": controller_radius_m,
            "dashpot_damping": dashpot_damping,
            "drag_damping": drag_damping,
            "init_spring_Y": init_spring_y,
        },
        "independent official reference dynamics changed",
    )
    reference_graph = result.get("canonical_reusable_graph")
    reference_support = result.get("support_dynamics")
    reference_actuation = result.get("realized_actuation")
    _require(
        isinstance(reference_graph, dict)
        and reference_graph.get("file_sha256")
        == file_sha256(registered_graph_path)
        and reference_graph.get("controller_patch_size_per_anchor")
        == controller_patch_size,
        "independent official reference graph changed",
    )
    _require(
        isinstance(reference_support, dict)
        and reference_support.get("mode") == "official-ground"
        and reference_support.get("reverse_factor") == -1.0
        and reference_support.get("uses_official_cuda_graph") is True,
        "independent official support dynamics changed",
    )
    _require(
        isinstance(reference_actuation, dict)
        and reference_actuation.get("controller_displacement_scale") == 1.0,
        "independent official actuation changed",
    )
    return result


def _unavailable_render_symbol(name: str):
    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"render-only PhysTwin symbol was called: {name}")

    unavailable.__name__ = name
    return unavailable


def _stub_module(name: str, symbols: tuple[str, ...]) -> None:
    module = types.ModuleType(name)
    module.__dict__["__all__"] = list(symbols)
    for symbol in symbols:
        setattr(module, symbol, _unavailable_render_symbol(f"{name}.{symbol}"))
    sys.modules[name] = module


def _install_headless_render_stubs() -> None:
    gaussian = types.ModuleType("gaussian_splatting")
    gaussian.__dict__["__path__"] = []
    scene = types.ModuleType("gaussian_splatting.scene")
    scene.__dict__["__path__"] = []
    utils = types.ModuleType("gaussian_splatting.utils")
    utils.__dict__["__path__"] = []
    sys.modules[gaussian.__name__] = gaussian
    sys.modules[scene.__name__] = scene
    sys.modules[utils.__name__] = utils
    _stub_module("gaussian_splatting.scene.gaussian_model", ("GaussianModel",))
    _stub_module("gaussian_splatting.scene.cameras", ("Camera",))
    _stub_module("gaussian_splatting.gaussian_renderer", ("render",))
    _stub_module(
        "gaussian_splatting.dynamic_utils",
        (
            "interpolate_motions_speedup",
            "knn_weights",
            "knn_weights_sparse",
            "get_topk_indices",
            "calc_weights_vals_from_indices",
        ),
    )
    _stub_module(
        "gaussian_splatting.utils.graphics_utils",
        ("getWorld2View2", "focal2fov", "fov2focal"),
    )
    _stub_module(
        "gaussian_splatting.rotation_utils",
        ("quaternion_multiply", "matrix_to_quaternion"),
    )
    _stub_module(
        "gs_render",
        (
            "remove_gaussians_with_low_opacity",
            "remove_gaussians_with_point_mesh_distance",
        ),
    )


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
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-repository", type=Path, required=True)
    parser.add_argument("--expected-execution-revision", required=True)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--expected-official-revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--registered-graph", type=Path, required=True)
    parser.add_argument("--prediction-manifest", type=Path, required=True)
    parser.add_argument("--spring-ensemble", type=Path, required=True)
    parser.add_argument("--reference-trajectory", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-runner", type=Path, required=True)
    parser.add_argument(
        "--historical-reference-trajectory", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--controller-radius-m", type=float, default=0.03)
    parser.add_argument("--controller-patch-size", type=int, default=16)
    parser.add_argument("--controller-max-neighbours", type=int, default=1)
    parser.add_argument("--init-spring-y", type=float, default=10000.0)
    parser.add_argument("--drag-damping", type=float, default=10.0)
    parser.add_argument("--dashpot-damping", type=float, default=100.0)
    parser.add_argument("--replays-per-field", type=int, default=4)
    parser.add_argument("--max-reference-to-replay-ratio", type=float, default=3.0)
    parser.add_argument("--min-member-to-replay-ratio", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    execution_repository = args.execution_repository.resolve(strict=True)
    _require(
        _git_revision(execution_repository) == args.expected_execution_revision,
        "execution repository revision changed",
    )
    _require(
        not subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=execution_repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout,
        "execution repository is dirty",
    )
    official_repository = args.official_phystwin_repo.resolve(strict=True)
    _require(
        _git_revision(official_repository) == args.expected_official_revision,
        "official PhysTwin revision changed",
    )
    _require(
        file_sha256(args.config) == args.expected_config_sha256,
        "official PhysTwin config changed",
    )
    _validate_independent_reference(
        result_path=args.reference_result,
        trajectory_path=args.reference_trajectory,
        runner_path=args.reference_runner,
        data_path=args.data,
        config_sha256=args.expected_config_sha256,
        official_revision=args.expected_official_revision,
        registered_graph_path=args.registered_graph,
        controller_max_neighbours=args.controller_max_neighbours,
        controller_radius_m=args.controller_radius_m,
        controller_patch_size=args.controller_patch_size,
        init_spring_y=args.init_spring_y,
        drag_damping=args.drag_damping,
        dashpot_damping=args.dashpot_damping,
    )
    _require(args.controller_radius_m > 0.0, "controller radius must be positive")
    _require(args.controller_patch_size > 0, "controller patch size must be positive")
    _require(
        args.controller_max_neighbours > 0, "controller neighbours must be positive"
    )
    _require(args.init_spring_y > 0.0, "initial spring stiffness must be positive")
    _require(args.drag_damping >= 0.0, "drag damping must be nonnegative")
    _require(args.dashpot_damping >= 0.0, "dashpot damping must be nonnegative")
    _require(args.replays_per_field >= 2, "replays per field must be at least two")
    _require(
        args.max_reference_to_replay_ratio > 0.0,
        "reference-to-replay ratio must be positive",
    )
    _require(
        args.min_member_to_replay_ratio > 0.0,
        "member-to-replay ratio must be positive",
    )

    prediction = json.loads(args.prediction_manifest.read_text(encoding="utf-8"))
    _require(
        prediction.get("schema") == "bayesian-phystwin.matphys-fold-ensemble-prediction"
        and prediction.get("schema_version") == 1,
        "MatPhys prediction manifest identity changed",
    )
    boundary = prediction.get("information_boundary")
    _require(isinstance(boundary, dict), "MatPhys prediction boundary is missing")
    _require(
        boundary.get("target_future_observations_used") is False
        and boundary.get("target_future_outcomes_opened") is False
        and boundary.get("target_object_used_for_checkpoint_training") is False,
        "MatPhys prediction crossed its information boundary",
    )
    member_count = prediction.get("member_count")
    _require(type(member_count) is int and member_count > 1, "member count is invalid")
    output_record = prediction.get("output")
    _require(isinstance(output_record, dict), "MatPhys spring output record is missing")
    _require(
        file_sha256(args.spring_ensemble) == output_record.get("sha256"),
        "MatPhys spring ensemble changed",
    )
    graph_input = prediction.get("inputs", {}).get("episode_graph", {})
    _require(
        file_sha256(args.registered_graph) == graph_input.get("sha256"),
        "registered graph changed",
    )
    fields = load_matphys_spring_ensemble(
        args.spring_ensemble,
        expected_member_count=member_count,
    )

    with args.data.open("rb") as stream:
        data = pickle.load(stream)
    object_points = np.asarray(data["object_points"], dtype=np.float32)
    controller_points = np.asarray(data["controller_points"], dtype=np.float32)
    _require(
        object_points.ndim == 3
        and object_points.shape[1:] == fields.graph_points_m.shape,
        "simulator object trajectory shape changed",
    )
    _require(
        np.array_equal(object_points[0], fields.graph_points_m),
        "simulator frame-zero points differ from the MatPhys graph",
    )
    _require(
        controller_points.ndim == 3
        and controller_points.shape[0] == object_points.shape[0]
        and controller_points.shape[2] == 3,
        "simulator controller trajectory shape changed",
    )
    replay_graph = load_registered_replay_graph(
        args.registered_graph,
        expected_points_m=fields.graph_points_m,
        expected_edges=fields.graph_edges,
        controller_reference_m=controller_points[0],
        controller_radius_m=args.controller_radius_m,
        controller_patch_size=args.controller_patch_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    _install_headless_render_stubs()
    sys.path.insert(0, str(official_repository))

    import torch
    import warp as wp
    from qqtt.engine.trainer_warp import InvPhyTrainerWarp
    from qqtt.utils import cfg

    warp_version = _validate_warp_runtime(importlib.metadata.version("warp-lang"))
    replay_runtime = _validate_replay_runtime(
        {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": importlib.metadata.version("torch"),
            "torch_cuda_version": str(torch.version.cuda),
            "warp_version": warp_version,
        }
    )

    cfg.load_from_yaml(str(args.config))
    cfg.controller_radius = args.controller_radius_m
    cfg.controller_max_neighbours = args.controller_max_neighbours
    cfg.init_spring_Y = args.init_spring_y
    cfg.drag_damping = args.drag_damping
    cfg.dashpot_damping = args.dashpot_damping
    cfg.device = args.device
    torch_device = torch.device(args.device)
    torch.cuda.set_device(torch_device)
    torch.manual_seed(260811)
    np.random.seed(260811)

    def _init_registered_graph(
        _trainer: object,
        runtime_object_points: Any,
        runtime_controller_points: Any,
        **_kwargs: Any,
    ) -> tuple[Any, Any, Any, Any, int]:
        runtime_points = runtime_object_points.detach().cpu().numpy()
        runtime_controls = runtime_controller_points.detach().cpu().numpy()
        _require(
            np.array_equal(runtime_points, fields.graph_points_m),
            "runtime object points differ from the registered graph",
        )
        _require(
            np.array_equal(runtime_controls, controller_points[0]),
            "runtime controller points differ from the registered action",
        )
        return (
            torch.as_tensor(
                replay_graph.vertices, dtype=torch.float32, device=args.device
            ),
            torch.as_tensor(
                replay_graph.springs, dtype=torch.int32, device=args.device
            ),
            torch.as_tensor(
                replay_graph.rest_lengths,
                dtype=torch.float32,
                device=args.device,
            ),
            torch.as_tensor(
                replay_graph.masses, dtype=torch.float32, device=args.device
            ),
            replay_graph.num_object_springs,
        )

    InvPhyTrainerWarp._init_start = _init_registered_graph
    build_started = time.perf_counter()
    trainer = InvPhyTrainerWarp(
        str(args.data),
        str(args.output_dir),
        pure_inference_mode=True,
        device=args.device,
    )
    torch.cuda.synchronize(args.device)
    build_seconds = time.perf_counter() - build_started
    simulator = trainer.simulator
    _require(
        trainer.num_object_springs == replay_graph.num_object_springs,
        "official Warp object-spring count changed",
    )
    base_log_y = wp.to_torch(simulator.wp_spring_Y, requires_grad=False).clone()
    _require(
        len(base_log_y)
        == len(fields.incumbent_spring_y_pa) + replay_graph.num_controller_springs,
        "official Warp spring count changed",
    )

    def rollout(object_spring_y_pa: np.ndarray | None) -> np.ndarray:
        spring_log_y = base_log_y.clone()
        if object_spring_y_pa is not None:
            spring_log_y[: trainer.num_object_springs] = torch.log(
                torch.as_tensor(
                    object_spring_y_pa,
                    dtype=spring_log_y.dtype,
                    device=args.device,
                )
            )
        simulator.set_spring_Y(spring_log_y)
        simulator.set_init_state(
            simulator.wp_init_vertices, simulator.wp_init_velocities
        )
        frames = [
            wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False)
            .cpu()
            .numpy()
            .copy()
        ]
        for frame in range(1, trainer.dataset.frame_len):
            simulator.set_controller_target(frame, pure_inference=True)
            if simulator.object_collision_flag:
                simulator.update_collision_graph()
            if cfg.use_graph:
                wp.capture_launch(simulator.forward_graph)
            else:
                simulator.step()
            frames.append(
                wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False)
                .cpu()
                .numpy()
                .copy()
            )
            simulator.set_init_state(
                simulator.wp_states[-1].wp_x,
                simulator.wp_states[-1].wp_v,
            )
        torch.cuda.synchronize(args.device)
        result = np.stack(frames).astype(np.float32, copy=False)
        _require(np.all(np.isfinite(result)), "official Warp replay is nonfinite")
        return result

    rollout_started = time.perf_counter()
    incumbent_replicates = np.stack(
        [rollout(None) for _ in range(args.replays_per_field)]
    )
    member_replicates = np.stack(
        [
            np.stack([rollout(field) for _ in range(args.replays_per_field)])
            for field in fields.member_spring_y_pa
        ]
    )
    rollout_seconds = time.perf_counter() - rollout_started
    arrays = hierarchical_trajectory_ensemble_arrays(
        incumbent_replicates,
        member_replicates,
    )
    incumbent_mean = arrays["incumbent_replay_mean_m"]
    with np.load(args.reference_trajectory, allow_pickle=False) as archive:
        reference = np.asarray(archive["vertices"], dtype=np.float32)
    with np.load(args.historical_reference_trajectory, allow_pickle=False) as archive:
        historical_reference = np.asarray(archive["vertices"], dtype=np.float32)
    _require(
        reference.shape == incumbent_mean.shape
        and historical_reference.shape == incumbent_mean.shape,
        "reference trajectory shapes changed",
    )
    reference_difference = incumbent_mean - reference.astype(np.float64)
    reference_rmse = float(np.sqrt(np.mean(reference_difference**2)))
    reference_max = float(np.max(np.abs(reference_difference)))
    historical_difference = reference.astype(np.float64) - historical_reference
    historical_reference_rmse = float(
        np.sqrt(np.mean(historical_difference**2))
    )
    historical_reference_max = float(np.max(np.abs(historical_difference)))
    replay_coordinate_std = float(
        np.sqrt(
            np.mean(
                np.trace(
                    arrays["incumbent_replay_covariance_m2"],
                    axis1=-2,
                    axis2=-1,
                )
            )
            / 3.0
        )
    )
    member_coordinate_std = float(
        np.sqrt(
            np.mean(
                np.trace(
                    arrays["between_member_covariance_m2"],
                    axis1=-2,
                    axis2=-1,
                )
            )
            / 3.0
        )
    )
    reference_to_replay_ratio = reference_rmse / max(replay_coordinate_std, 1e-12)
    effective_replay_floor = max(
        replay_coordinate_std,
        historical_reference_rmse,
        1e-12,
    )
    member_to_replay_ratio = member_coordinate_std / effective_replay_floor
    replay_passed = bool(
        reference_to_replay_ratio <= args.max_reference_to_replay_ratio
        and member_to_replay_ratio >= args.min_member_to_replay_ratio
    )

    archive_path = args.output_dir / "matphys_warp_trajectory_ensemble.npz"
    np.savez_compressed(archive_path, **arrays)
    identity = {
        "schema": MATPHYS_WARP_ENSEMBLE_SCHEMA,
        "schema_version": MATPHYS_WARP_ENSEMBLE_VERSION,
        "protocol": MATPHYS_WARP_ENSEMBLE_PROTOCOL,
        "case_id": prediction.get("case_id"),
        "target_object_id": prediction.get("target_object_id"),
        "source_prediction_id": prediction.get("prediction_id"),
        "source_ensemble_id": prediction.get("source_ensemble_id"),
        "member_count": member_count,
        "replays_per_field": args.replays_per_field,
        "replay_strategy": "law-of-total-variance-between-plus-within-v1",
        "registered_graph": {
            "path": str(args.registered_graph.resolve(strict=True)),
            "sha256": file_sha256(args.registered_graph),
            "object_node_count": len(fields.graph_points_m),
            "object_spring_count": replay_graph.num_object_springs,
            "controller_spring_count": replay_graph.num_controller_springs,
            "controller_group_count": replay_graph.controller_group_count,
        },
        "runtime": {
            "execution_revision": _git_revision(execution_repository),
            "runner_sha256": file_sha256(Path(__file__)),
            "official_phystwin_revision": _git_revision(official_repository),
            "official_config_sha256": file_sha256(args.config),
            **replay_runtime,
            "device": args.device,
            "build_seconds": build_seconds,
            "rollout_seconds": rollout_seconds,
        },
        "parity": {
            "reference_trajectory_sha256": file_sha256(args.reference_trajectory),
            "reference_result_sha256": file_sha256(args.reference_result),
            "reference_runner_sha256": file_sha256(args.reference_runner),
            "historical_reference_trajectory_sha256": file_sha256(
                args.historical_reference_trajectory
            ),
            "reference_byte_identical": bool(
                incumbent_replicates[0].tobytes() == reference.tobytes()
            ),
            "reference_rmse_m": reference_rmse,
            "reference_max_abs_m": reference_max,
            "historical_reference_rmse_m": historical_reference_rmse,
            "historical_reference_max_abs_m": historical_reference_max,
            "effective_replay_floor_m": effective_replay_floor,
            "incumbent_replay_coordinate_std_m": replay_coordinate_std,
            "reference_to_replay_ratio": reference_to_replay_ratio,
            "maximum_reference_to_replay_ratio": (args.max_reference_to_replay_ratio),
            "member_between_coordinate_std_m": member_coordinate_std,
            "member_to_replay_ratio": member_to_replay_ratio,
            "minimum_member_to_replay_ratio": args.min_member_to_replay_ratio,
            "all_incumbent_replays_byte_identical": bool(
                all(
                    item.tobytes() == incumbent_replicates[0].tobytes()
                    for item in incumbent_replicates[1:]
                )
            ),
            "passed": replay_passed,
        },
        "output": {
            "path": str(archive_path.resolve(strict=True)),
            "sha256": file_sha256(archive_path),
        },
        "information_boundary": {
            "official_warp_replay_completed": True,
            "source_only_smoke": True,
            "target_future_observations_used": False,
            "target_future_outcomes_opened": False,
            "calibration_claim_authorized": False,
            "point_accuracy_claim_authorized": False,
        },
        "claim_boundary": (
            "Target-excluded MatPhys checkpoint disagreement has been propagated "
            "through repeated official PhysTwin Warp replays. Between-checkpoint "
            "and within-checkpoint replay covariance are separated before being "
            "combined. This source-only replay does not establish calibrated "
            "uncertainty or improved accuracy."
        ),
        "passed": replay_passed,
    }
    payload = {**identity, "result_id": _canonical_id(identity)}
    manifest_path = args.output_dir / "matphys_warp_trajectory_ensemble.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if replay_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
