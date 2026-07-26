#!/usr/bin/env python3
"""Render a prefix-only Gaussian observation Jacobian for one PhysTwin case.

This runner is deliberately a source-development utility. It reads only the
declared RGB/mask prefix frames, a pre-existing physical trajectory, frame-zero
Gaussian appearance, and a frame-zero graph basis. It never opens manual
tracks, future RGB, future masks, or evaluation targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.photometric_graph_update import (
    PhotometricGraphConfig,
    select_photometric_graph_update,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _json_write(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _load_gaussians(path: Path, *, device: str) -> dict[str, Any]:
    import torch
    from plyfile import PlyData

    vertex = PlyData.read(path).elements[0]

    def stack(names: list[str]) -> np.ndarray:
        return np.stack([np.asarray(vertex[name]) for name in names], axis=1)

    names = [property_.name for property_ in vertex.properties]
    xyz = stack(["x", "y", "z"]).astype(np.float32)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    dc = stack(["f_dc_0", "f_dc_1", "f_dc_2"]).astype(np.float32)
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    _require(len(rest_names) == 45, "expected degree-three Gaussian SH features")
    rest = stack(rest_names).reshape(len(xyz), 3, 15).transpose(0, 2, 1)
    colors = np.concatenate((dc[:, None, :], rest), axis=1).astype(np.float32)
    scale_names = sorted(
        (name for name in names if name.startswith("scale_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    raw_scale = stack(scale_names).astype(np.float32)
    _require(raw_scale.shape[1] in {1, 3}, "unsupported Gaussian scale shape")
    scale = np.exp(raw_scale)
    if scale.shape[1] == 1:
        scale = np.repeat(scale, 3, axis=1)
    rotation_names = sorted(
        (name for name in names if name.startswith("rot_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    rotation = stack(rotation_names).astype(np.float32)
    rotation /= np.maximum(np.linalg.norm(rotation, axis=1, keepdims=True), 1e-12)
    opacity = 1.0 / (1.0 + np.exp(-opacity))
    retained = opacity >= 0.1
    _require(np.any(retained), "opacity filter removed every Gaussian")
    return {
        "xyz": xyz[retained],
        "xyz_torch": torch.as_tensor(xyz[retained], device=device),
        "colors": torch.as_tensor(colors[retained], device=device),
        "scales": torch.as_tensor(scale[retained], device=device),
        "rotations": torch.as_tensor(rotation[retained], device=device),
        "opacities": torch.as_tensor(opacity[retained], device=device),
        "retained_count": int(np.sum(retained)),
        "original_count": int(len(retained)),
    }


def _load_cameras(
    path: Path,
    *,
    names: list[str],
    downsample: int,
    device: str,
) -> tuple[Any, Any, int, int]:
    import torch

    rows = json.loads(path.read_text(encoding="utf-8"))
    by_name = {str(row["img_name"]): row for row in rows}
    missing = sorted(set(names) - set(by_name))
    _require(not missing, f"camera JSON is missing {missing}")
    width = int(by_name[names[0]]["width"]) // downsample
    height = int(by_name[names[0]]["height"]) // downsample
    _require(width >= 16 and height >= 16, "downsampled camera is too small")
    view_matrices = []
    intrinsic_matrices = []
    for name in names:
        row = by_name[name]
        _require(
            int(row["width"]) // downsample == width
            and int(row["height"]) // downsample == height,
            "camera resolutions disagree",
        )
        camera_to_world = np.eye(4, dtype=np.float32)
        camera_to_world[:3, :3] = np.asarray(row["rotation"], dtype=np.float32)
        camera_to_world[:3, 3] = np.asarray(row["position"], dtype=np.float32)
        view_matrices.append(np.linalg.inv(camera_to_world))
        intrinsic = np.asarray(
            [
                [float(row["fx"]) / downsample, 0.0, 0.5 * width],
                [0.0, float(row["fy"]) / downsample, 0.5 * height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        intrinsic_matrices.append(intrinsic)
    return (
        torch.as_tensor(np.stack(view_matrices), device=device),
        torch.as_tensor(np.stack(intrinsic_matrices), device=device),
        width,
        height,
    )


def _render(
    gaussian: dict[str, Any],
    xyz: Any,
    view_matrices: Any,
    intrinsic_matrices: Any,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from gsplat import rasterization

    camera_count = len(view_matrices)
    background = torch.ones((camera_count, 3), device=xyz.device)
    colors, alpha, _ = rasterization(
        means=xyz,
        quats=gaussian["rotations"],
        scales=gaussian["scales"],
        opacities=gaussian["opacities"],
        colors=gaussian["colors"],
        viewmats=view_matrices,
        Ks=intrinsic_matrices,
        width=width,
        height=height,
        sh_degree=3,
        packed=False,
        backgrounds=background,
        render_mode="RGB",
        rasterize_mode="antialiased",
    )
    return (
        colors.detach().cpu().numpy().astype(np.float32),
        alpha[..., 0].detach().cpu().numpy().astype(np.float32),
    )


def _read_prefix_images(
    pattern: str,
    mask_pattern: str,
    *,
    frames: list[int],
    camera_count: int,
    width: int,
    height: int,
    mask_erosion_pixels: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    import cv2

    images = np.empty(
        (len(frames), camera_count, height, width, 3),
        dtype=np.float32,
    )
    masks = np.empty((len(frames), camera_count, height, width), dtype=bool)
    source_hashes: dict[str, str] = {}
    kernel = np.ones(
        (2 * mask_erosion_pixels + 1, 2 * mask_erosion_pixels + 1),
        dtype=np.uint8,
    )
    for frame_position, frame in enumerate(frames):
        for camera in range(camera_count):
            image_path = Path(pattern.format(camera=camera, frame=frame))
            mask_path = Path(mask_pattern.format(camera=camera, frame=frame))
            _require(image_path.is_file(), f"missing prefix RGB {image_path}")
            _require(mask_path.is_file(), f"missing prefix mask {mask_path}")
            source_hashes[str(image_path)] = _sha256(image_path)
            source_hashes[str(mask_path)] = _sha256(mask_path)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            _require(image is not None and mask is not None, "OpenCV read failed")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)
            if mask_erosion_pixels:
                mask = cv2.erode(mask, kernel)
            images[frame_position, camera] = image.astype(np.float32) / 255.0
            masks[frame_position, camera] = mask.astype(bool)
    return images, masks, source_hashes


def _fixed_gaussian_attachment(
    node_xyz: np.ndarray,
    gaussian_xyz: np.ndarray,
    *,
    neighbour_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    _require(neighbour_count >= 1, "neighbour_count must be positive")
    distances, indices = cKDTree(node_xyz).query(
        gaussian_xyz,
        k=neighbour_count,
        workers=-1,
    )
    if neighbour_count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = 1.0 / np.maximum(distances, 1e-6)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return indices.astype(np.int64), weights.astype(np.float32)


def _apply_attachment(
    values: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    selected = values[indices]
    return np.sum(weights[..., None] * selected, axis=1)


def _parameter_fields(
    graph_basis: np.ndarray,
    *,
    position_step_m: float,
) -> np.ndarray:
    basis = np.asarray(graph_basis, dtype=np.float64)
    maximum = np.max(np.abs(basis), axis=0)
    _require(np.all(maximum > 0.0), "graph basis contains an empty mode")
    fields = np.zeros((len(basis), 3, 3 * basis.shape[1]), dtype=np.float32)
    for mode in range(basis.shape[1]):
        normalized = basis[:, mode] * position_step_m / maximum[mode]
        for coordinate in range(3):
            fields[:, coordinate, 3 * mode + coordinate] = normalized
    return fields


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--graph-basis", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--rgb-pattern", required=True)
    parser.add_argument("--mask-pattern", required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--camera-names", nargs="+", default=["cam0", "cam1", "cam2"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--mask-erosion-pixels", type=int, default=2)
    parser.add_argument("--alpha-threshold", type=float, default=0.1)
    parser.add_argument("--attachment-neighbours", type=int, default=16)
    parser.add_argument("--position-step-m", type=float, default=0.005)
    parser.add_argument("--fit-frame-count", type=int, default=4)
    parser.add_argument("--correlation-block-size", type=int, default=8)
    parser.add_argument("--state-ridge", type=float, default=0.05)
    parser.add_argument("--maximum-weight-norm", type=float, default=2.0)
    parser.add_argument(
        "--minimum-validation-improvement-fraction",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--minimum-validation-improvement-absolute",
        type=float,
        default=0.0005,
    )
    args = parser.parse_args()

    _require(not args.output.exists(), "output already exists")
    _require(args.downsample >= 1, "downsample must be positive")
    _require(len(args.frames) >= 3, "at least three prefix frames are required")
    _require(args.frames == sorted(set(args.frames)), "frames must be unique and sorted")
    for path in (
        args.trajectory,
        args.graph_basis,
        args.gaussian_ply,
        args.camera_json,
    ):
        _require(path.is_file(), f"missing input {path}")

    args.output.mkdir(parents=True)
    with args.trajectory.open("rb") as handle:
        trajectory = np.asarray(pickle.load(handle), dtype=np.float32)
    basis_archive = np.load(args.graph_basis)
    _require("graph_basis" in basis_archive, "graph basis archive is missing graph_basis")
    graph_basis = np.asarray(basis_archive["graph_basis"], dtype=np.float64)
    _require(
        trajectory.ndim == 3
        and trajectory.shape[1:] == (len(graph_basis), 3),
        "trajectory and graph basis disagree",
    )
    _require(max(args.frames) < len(trajectory), "prefix frame exceeds trajectory")

    gaussian = _load_gaussians(args.gaussian_ply, device=args.device)
    view_matrices, intrinsic_matrices, width, height = _load_cameras(
        args.camera_json,
        names=args.camera_names,
        downsample=args.downsample,
        device=args.device,
    )
    observed, object_mask, prefix_hashes = _read_prefix_images(
        args.rgb_pattern,
        args.mask_pattern,
        frames=args.frames,
        camera_count=len(args.camera_names),
        width=width,
        height=height,
        mask_erosion_pixels=args.mask_erosion_pixels,
    )

    attachment_indices, attachment_weights = _fixed_gaussian_attachment(
        trajectory[0],
        gaussian["xyz"],
        neighbour_count=args.attachment_neighbours,
    )
    node_fields = _parameter_fields(
        graph_basis,
        position_step_m=args.position_step_m,
    )
    gaussian_parameter_fields = np.stack(
        [
            _apply_attachment(
                node_fields[..., parameter],
                attachment_indices,
                attachment_weights,
            )
            for parameter in range(node_fields.shape[-1])
        ],
        axis=-1,
    ).astype(np.float32)

    import torch

    baseline_rgb = np.empty_like(observed)
    baseline_alpha = np.empty(object_mask.shape, dtype=np.float32)
    jacobian = np.empty((*observed.shape, node_fields.shape[-1]), dtype=np.float32)
    frame_zero_nodes = trajectory[0]
    for position, frame in enumerate(args.frames):
        gaussian_displacement = _apply_attachment(
            trajectory[frame] - frame_zero_nodes,
            attachment_indices,
            attachment_weights,
        )
        baseline_xyz = gaussian["xyz"] + gaussian_displacement
        baseline_render, alpha = _render(
            gaussian,
            torch.as_tensor(baseline_xyz, device=args.device),
            view_matrices,
            intrinsic_matrices,
            width=width,
            height=height,
        )
        baseline_rgb[position] = baseline_render
        baseline_alpha[position] = alpha
        for parameter in range(node_fields.shape[-1]):
            perturbed_render, _ = _render(
                gaussian,
                torch.as_tensor(
                    baseline_xyz + gaussian_parameter_fields[..., parameter],
                    device=args.device,
                ),
                view_matrices,
                intrinsic_matrices,
                width=width,
                height=height,
            )
            jacobian[position, ..., parameter] = (
                perturbed_render - baseline_render
            )

    valid_mask = object_mask & (baseline_alpha >= args.alpha_threshold)
    settings = PhotometricGraphConfig(
        fit_frame_count=args.fit_frame_count,
        correlation_block_size=args.correlation_block_size,
        state_ridge=args.state_ridge,
        maximum_weight_norm=args.maximum_weight_norm,
        minimum_validation_improvement_fraction=(
            args.minimum_validation_improvement_fraction
        ),
        minimum_validation_improvement_absolute=(
            args.minimum_validation_improvement_absolute
        ),
    )
    selection = select_photometric_graph_update(
        observed,
        baseline_rgb,
        jacobian,
        valid_mask,
        config=settings,
    )
    selected_node_correction = np.einsum(
        "ncp,p->nc",
        node_fields,
        selection.state_weights,
    )
    artifact = args.output / "photometric_prefix.npz"
    np.savez_compressed(
        artifact,
        frames=np.asarray(args.frames, dtype=np.int64),
        observed_rgb=observed.astype(np.float16),
        baseline_rgb=baseline_rgb.astype(np.float16),
        baseline_alpha=baseline_alpha.astype(np.float16),
        valid_mask=valid_mask,
        state_jacobian_rgb=jacobian.astype(np.float16),
        graph_basis=graph_basis.astype(np.float32),
        node_parameter_fields_m=node_fields,
        selected_state_weights=selection.state_weights,
        selected_node_correction_m=selected_node_correction,
        posterior_covariance=selection.posterior_covariance,
    )
    report = {
        "schema_version": 1,
        "status": (
            "prefix_gate_passed" if selection.accepted else "prefix_gate_failed"
        ),
        "accepted": selection.accepted,
        "reason": selection.reason,
        "frames_read": args.frames,
        "future_rgb_or_mask_read": False,
        "manual_tracks_read": False,
        "trajectory_sha256": _sha256(args.trajectory),
        "graph_basis_sha256": _sha256(args.graph_basis),
        "gaussian_ply_sha256": _sha256(args.gaussian_ply),
        "camera_json_sha256": _sha256(args.camera_json),
        "prefix_source_sha256": prefix_hashes,
        "artifact_sha256": _sha256(artifact),
        "selected_state_weights_sha256": _array_sha256(
            selection.state_weights
        ),
        "selected_node_correction_sha256": _array_sha256(
            selected_node_correction
        ),
        "gaussian_counts": {
            "original": gaussian["original_count"],
            "retained": gaussian["retained_count"],
        },
        "render_resolution": [height, width],
        "valid_fraction": float(np.mean(valid_mask)),
        "maximum_node_correction_m": float(
            np.max(np.linalg.norm(selected_node_correction, axis=1))
        ),
        "selection": {
            "state_weights": selection.state_weights.tolist(),
            "posterior_covariance": selection.posterior_covariance.tolist(),
            "diagnostics": selection.diagnostics,
        },
        "claim_boundary": (
            "Opened-source prefix competence only. Passing does not authorize "
            "a target, held-v8 access, or a state-of-the-art claim."
        ),
    }
    report_path = args.output / "report.json"
    _json_write(report_path, report)
    (args.output / "PREFIX_SELECTION_COMPLETE").write_text(
        _sha256(report_path) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
