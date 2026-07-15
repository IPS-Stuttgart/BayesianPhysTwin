#!/usr/bin/env python3
"""Run a headless official-PhysTwin rollout on source-only Deform360 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360_dense_source import (
    DEFORM360_SUPPORT_DYNAMICS,
    support_dynamics_reverse_factor,
)


def _unavailable_render_symbol(name: str):
    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"render-only PhysTwin symbol was called: {name}")

    unavailable.__name__ = name
    return unavailable


def _stub_module(name: str, symbols: tuple[str, ...]) -> None:
    module = types.ModuleType(name)
    module.__all__ = list(symbols)
    for symbol in symbols:
        setattr(module, symbol, _unavailable_render_symbol(f"{name}.{symbol}"))
    sys.modules[name] = module


def _install_headless_render_stubs() -> None:
    """Bypass optional rendering dependencies without changing dynamics code."""
    gaussian = types.ModuleType("gaussian_splatting")
    gaussian.__path__ = []  # type: ignore[attr-defined]
    scene = types.ModuleType("gaussian_splatting.scene")
    scene.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("gaussian_splatting.utils")
    utils.__path__ = []  # type: ignore[attr-defined]
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _symmetric_chamfer_m(predicted: np.ndarray, target: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    pred_to_target = cKDTree(target).query(predicted, workers=-1)[0]
    target_to_pred = cKDTree(predicted).query(target, workers=-1)[0]
    return float(0.5 * (pred_to_target.mean() + target_to_pred.mean()))


def _score_trajectory(
    predicted: np.ndarray,
    target: np.ndarray,
    visibility: np.ndarray,
    validity: np.ndarray,
    intervals: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    persistence = np.repeat(target[:1], len(target), axis=0)
    per_frame: list[dict[str, float | int | bool | None]] = []
    for frame in range(len(target)):
        mask = np.asarray(visibility[frame] & validity[frame], dtype=bool)
        if not np.any(mask):
            mask = np.ones(target.shape[1], dtype=bool)
        pred = predicted[frame, mask]
        truth = target[frame, mask]
        persist = persistence[frame, mask]
        finite = bool(np.isfinite(pred).all() and np.isfinite(truth).all())
        if not finite:
            per_frame.append(
                {
                    "frame": frame,
                    "valid_points": int(mask.sum()),
                    "prediction_finite": False,
                    "track_rmse_m": None,
                    "persistence_track_rmse_m": float(
                        np.sqrt(np.mean((persist - truth) ** 2))
                    ),
                    "chamfer_m": None,
                    "persistence_chamfer_m": _symmetric_chamfer_m(persist, truth),
                }
            )
            continue
        per_frame.append(
            {
                "frame": frame,
                "valid_points": int(mask.sum()),
                "prediction_finite": True,
                "track_rmse_m": float(np.sqrt(np.mean((pred - truth) ** 2))),
                "persistence_track_rmse_m": float(
                    np.sqrt(np.mean((persist - truth) ** 2))
                ),
                "chamfer_m": _symmetric_chamfer_m(pred, truth),
                "persistence_chamfer_m": _symmetric_chamfer_m(persist, truth),
            }
        )
    future = per_frame[1:]
    all_future_finite = all(bool(row["prediction_finite"]) for row in future)
    result = {
        "per_frame": per_frame,
        "future_prediction_finite": all_future_finite,
        "future_track_rmse_m": (
            float(np.mean([float(row["track_rmse_m"]) for row in future]))
            if all_future_finite
            else None
        ),
        "future_persistence_track_rmse_m": float(
            np.mean([float(row["persistence_track_rmse_m"]) for row in future])
        ),
        "future_chamfer_m": (
            float(np.mean([float(row["chamfer_m"]) for row in future]))
            if all_future_finite
            else None
        ),
        "future_persistence_chamfer_m": float(
            np.mean([float(row["persistence_chamfer_m"]) for row in future])
        ),
    }
    interval_metrics = {}
    for name, (start, stop) in (intervals or {}).items():
        selected = per_frame[max(1, start) : stop]
        if not selected:
            raise ValueError(f"score interval {name!r} has no future frames")
        finite = all(bool(row["prediction_finite"]) for row in selected)
        interval_metrics[name] = {
            "frame_range": [max(1, start), stop],
            "frame_count": len(selected),
            "prediction_finite": finite,
            "track_rmse_m": (
                float(np.mean([float(row["track_rmse_m"]) for row in selected]))
                if finite
                else None
            ),
            "persistence_track_rmse_m": float(
                np.mean([float(row["persistence_track_rmse_m"]) for row in selected])
            ),
            "chamfer_m": (
                float(np.mean([float(row["chamfer_m"]) for row in selected]))
                if finite
                else None
            ),
            "persistence_chamfer_m": float(
                np.mean([float(row["persistence_chamfer_m"]) for row in selected])
            ),
        }
    result["intervals"] = interval_metrics
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-phystwin-repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--controller-radius-m", type=float)
    parser.add_argument("--controller-max-neighbours", type=int)
    parser.add_argument("--init-spring-y", type=float)
    parser.add_argument("--drag-damping", type=float)
    parser.add_argument("--dashpot-damping", type=float)
    parser.add_argument(
        "--particle-mass-scale",
        type=float,
        default=None,
        help=(
            "Opt-in source-only effective inertia multiplier. The official "
            "PhysTwin default leaves the mass array untouched."
        ),
    )
    parser.add_argument(
        "--controller-displacement-scale",
        type=float,
        default=None,
        help=(
            "Opt-in source-only realized-actuation gain. Controller displacement "
            "from frame zero is scaled before the official Warp rollout."
        ),
    )
    parser.add_argument(
        "--controller-spring-stiffness-scale",
        type=float,
        default=None,
        help=(
            "Opt-in source-only multiplier for virtual controller-attachment "
            "spring stiffness, separate from object material stiffness."
        ),
    )
    parser.add_argument(
        "--reverse-z",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override which side of z=0 contains free space and gravity.",
    )
    parser.add_argument(
        "--support-dynamics",
        choices=DEFORM360_SUPPORT_DYNAMICS,
        default="official-ground",
        help=(
            "Opt-in support regime. gravity-neutral-planar is a source-only "
            "diagnostic for already settled, support-confined motion."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.controller_radius_m is not None and args.controller_radius_m <= 0.0:
        raise ValueError("controller radius must be positive")
    if (
        args.controller_max_neighbours is not None
        and args.controller_max_neighbours < 1
    ):
        raise ValueError("controller neighbour count must be positive")
    if args.init_spring_y is not None and args.init_spring_y <= 0.0:
        raise ValueError("initial spring stiffness must be positive")
    if args.drag_damping is not None and args.drag_damping < 0.0:
        raise ValueError("drag damping must be non-negative")
    if args.dashpot_damping is not None and args.dashpot_damping < 0.0:
        raise ValueError("dashpot damping must be non-negative")
    if args.particle_mass_scale is not None and (
        not np.isfinite(args.particle_mass_scale) or args.particle_mass_scale <= 0.0
    ):
        raise ValueError("particle mass scale must be positive")
    if args.controller_displacement_scale is not None and (
        not np.isfinite(args.controller_displacement_scale)
        or args.controller_displacement_scale < 0.0
    ):
        raise ValueError("controller displacement scale must be non-negative")
    if args.controller_spring_stiffness_scale is not None and (
        not np.isfinite(args.controller_spring_stiffness_scale)
        or args.controller_spring_stiffness_scale <= 0.0
    ):
        raise ValueError("controller spring stiffness scale must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("WANDB_MODE", "disabled")
    _install_headless_render_stubs()
    sys.path.insert(0, str(args.official_phystwin_repo.resolve()))

    import torch
    import warp as wp

    from qqtt.engine.trainer_warp import InvPhyTrainerWarp
    from qqtt.utils import cfg

    cfg.load_from_yaml(str(args.config))
    overrides = {
        "controller_radius": args.controller_radius_m,
        "controller_max_neighbours": args.controller_max_neighbours,
        "init_spring_Y": args.init_spring_y,
        "drag_damping": args.drag_damping,
        "dashpot_damping": args.dashpot_damping,
        "reverse_z": args.reverse_z,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(cfg, name, value)
    if args.support_dynamics != "official-ground":
        # The official CUDA graph captures reverse_factor by value during
        # construction. Eager stepping lets the opt-in factor be applied after
        # the untouched trainer builds its simulator.
        cfg.use_graph = False
    torch_device = torch.device(args.device)
    torch.cuda.set_device(torch_device)
    torch.cuda.reset_peak_memory_stats(torch_device)
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
    controller_points = simulator.controller_points
    if controller_points is None:
        raise ValueError("Deform360 rollout has no controller trajectory")
    controller_origin = controller_points[:1]
    original_controller_displacement = controller_points - controller_origin
    applied_controller_displacement_scale = (
        args.controller_displacement_scale
        if args.controller_displacement_scale is not None
        else 1.0
    )
    if args.controller_displacement_scale is not None:
        scaled_controller_points = controller_origin + (
            applied_controller_displacement_scale * original_controller_displacement
        )
        simulator.controller_points = scaled_controller_points
        trainer.controller_points = scaled_controller_points
    applied_controller_points = simulator.controller_points
    original_max_controller_displacement_m = float(
        torch.linalg.vector_norm(original_controller_displacement, dim=-1).max().item()
    )
    applied_max_controller_displacement_m = float(
        torch.linalg.vector_norm(
            applied_controller_points - applied_controller_points[:1], dim=-1
        )
        .max()
        .item()
    )
    masses = wp.to_torch(simulator.wp_masses, requires_grad=False)
    if args.particle_mass_scale is not None:
        masses.mul_(args.particle_mass_scale)
    spring_log_stiffness = wp.to_torch(simulator.wp_spring_Y, requires_grad=False)
    num_controller_springs = simulator.n_springs - trainer.num_object_springs
    if num_controller_springs < 1:
        raise ValueError("Deform360 rollout has no controller attachment springs")
    original_controller_spring_y = torch.exp(
        spring_log_stiffness[-num_controller_springs:]
    )
    if args.controller_spring_stiffness_scale is not None:
        spring_log_stiffness[-num_controller_springs:].add_(
            float(np.log(args.controller_spring_stiffness_scale))
        )
    applied_controller_spring_y = torch.clamp(
        torch.exp(spring_log_stiffness[-num_controller_springs:]),
        min=float(simulator.spring_Y_min),
        max=float(simulator.spring_Y_max),
    )
    simulator.reverse_factor = support_dynamics_reverse_factor(
        args.support_dynamics,
        reverse_z=bool(cfg.reverse_z),
    )
    frame_len = trainer.dataset.frame_len
    simulator.set_init_state(simulator.wp_init_vertices, simulator.wp_init_velocities)
    vertices = [
        wp.to_torch(simulator.wp_states[0].wp_x, requires_grad=False).cpu().numpy()
    ]
    rollout_started = time.perf_counter()
    for frame in range(1, frame_len):
        simulator.set_controller_target(frame, pure_inference=True)
        if simulator.object_collision_flag:
            simulator.update_collision_graph()
        if cfg.use_graph:
            wp.capture_launch(simulator.forward_graph)
        else:
            simulator.step()
        vertices.append(
            wp.to_torch(simulator.wp_states[-1].wp_x, requires_grad=False).cpu().numpy()
        )
        simulator.set_init_state(
            simulator.wp_states[-1].wp_x,
            simulator.wp_states[-1].wp_v,
        )
    torch.cuda.synchronize(args.device)
    rollout_seconds = time.perf_counter() - rollout_started
    trajectory = np.stack(vertices)
    predicted = trajectory[:, : trainer.num_original_points]
    target = trainer.object_points.detach().cpu().numpy()
    visibility = trainer.object_visibilities.detach().cpu().numpy()
    validity = trainer.object_motions_valid.detach().cpu().numpy()
    intervals = None
    if args.split_json is not None:
        split = json.loads(args.split_json.read_text(encoding="utf-8"))
        if int(split.get("frame_len", -1)) != frame_len:
            raise ValueError("score split frame count differs from PhysTwin data")
        intervals = {
            name: (int(bounds[0]), int(bounds[1]))
            for name, bounds in split.items()
            if name != "frame_len"
        }
    metrics = _score_trajectory(
        predicted,
        target,
        visibility,
        validity,
        intervals=intervals,
    )

    trajectory_path = args.output_dir / "official_phystwin_trajectory.npz"
    np.savez_compressed(trajectory_path, vertices=trajectory)
    finite_by_frame = np.isfinite(trajectory).all(axis=(1, 2))
    invalid_frames = np.flatnonzero(~finite_by_frame)
    finite_values = trajectory[np.isfinite(trajectory)]
    payload = {
        "passed": bool(finite_by_frame.all()),
        "source_only_smoke": True,
        "official_phystwin_revision": _git_revision(args.official_phystwin_repo),
        "data_sha256": _sha256_file(args.data),
        "config_sha256": _sha256_file(args.config),
        "split_sha256": (
            _sha256_file(args.split_json) if args.split_json is not None else None
        ),
        "config_overrides": {
            name: value for name, value in overrides.items() if value is not None
        },
        "support_dynamics": {
            "mode": args.support_dynamics,
            "reverse_factor": float(simulator.reverse_factor),
            "uses_official_cuda_graph": bool(cfg.use_graph),
            "claim_boundary": (
                "official-ground is the frozen PhysTwin path; "
                "gravity-neutral-planar is an exploratory source-only "
                "reduced-order support diagnostic"
            ),
        },
        "effective_inertia": {
            "particle_mass_scale": float(args.particle_mass_scale or 1.0),
            "override_applied": args.particle_mass_scale is not None,
            "minimum_particle_mass": float(masses.min().item()),
            "maximum_particle_mass": float(masses.max().item()),
            "claim_boundary": (
                "source-only effective stiffness-to-inertia diagnostic; "
                "not an identified SI material mass"
            ),
        },
        "realized_actuation": {
            "controller_displacement_scale": float(
                applied_controller_displacement_scale
            ),
            "override_applied": args.controller_displacement_scale is not None,
            "original_max_controller_displacement_m": (
                original_max_controller_displacement_m
            ),
            "applied_max_controller_displacement_m": (
                applied_max_controller_displacement_m
            ),
            "claim_boundary": (
                "source-only realized controller-displacement gain; interpreted "
                "as an effective actuation-transmission latent, not a calibrated "
                "robot gain"
            ),
        },
        "contact_transmission": {
            "controller_spring_stiffness_scale": float(
                args.controller_spring_stiffness_scale or 1.0
            ),
            "override_applied": args.controller_spring_stiffness_scale is not None,
            "num_controller_springs": int(num_controller_springs),
            "original_controller_spring_y_min": float(
                original_controller_spring_y.min().item()
            ),
            "original_controller_spring_y_max": float(
                original_controller_spring_y.max().item()
            ),
            "effective_controller_spring_y_min": float(
                applied_controller_spring_y.min().item()
            ),
            "effective_controller_spring_y_max": float(
                applied_controller_spring_y.max().item()
            ),
            "claim_boundary": (
                "source-only virtual attachment compliance; separates contact "
                "transmission from object material stiffness but is not a "
                "measured grasp stiffness"
            ),
        },
        "frame_count": int(frame_len),
        "num_original_points": int(trainer.num_original_points),
        "num_surface_points": int(trainer.num_surface_points),
        "num_all_points": int(trainer.num_all_points),
        "num_controller_points": int(trainer.controller_points.shape[1]),
        "num_springs": int(simulator.n_springs),
        "num_object_springs": int(trainer.num_object_springs),
        "num_controller_springs": int(num_controller_springs),
        "finite_vertex_fraction": float(np.mean(np.isfinite(trajectory))),
        "first_nonfinite_frame": (
            int(invalid_frames[0]) if len(invalid_frames) else None
        ),
        "maximum_absolute_finite_coordinate_m": (
            float(np.max(np.abs(finite_values))) if len(finite_values) else None
        ),
        "build_seconds": build_seconds,
        "rollout_seconds": rollout_seconds,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(torch_device)),
        "trajectory_sha256": _sha256_file(trajectory_path),
        "metrics": metrics,
    }
    output_path = args.output_dir / "official_phystwin_smoke.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
