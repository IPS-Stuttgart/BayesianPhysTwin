"""Causal multi-keyframe DINO features on PhysTwin material nodes."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from .matphys_causal_bridge import numeric_frame_paths, sha256_file


DINO_HUB_REPOSITORY = "facebookresearch/dinov2"
DINO_NODE_FEATURE_CONTRACT = "tracked-mask-depth-multiview-fit-prefix-v1"


def _camera_data(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (case_dir / "calibrate.pkl").open("rb") as handle:
        c2ws = np.asarray(pickle.load(handle), dtype=np.float32)
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float32)
    if np.max(np.abs(intrinsics[:, :2, :])) <= 2.0:
        wh = metadata["WH"]
        if isinstance(wh[0], list):
            widths_heights = wh
        else:
            widths_heights = [wh for _ in range(len(intrinsics))]
        for camera, (width, height) in enumerate(widths_heights):
            intrinsics[camera, 0, :] *= float(width)
            intrinsics[camera, 1, :] *= float(height)
    return np.linalg.inv(c2ws).astype(np.float32), intrinsics


def project_world_points(
    points: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project metric world points with the calibrated pinhole camera."""

    xyz = np.asarray(points, dtype=np.float64)
    w2c = np.asarray(world_to_camera, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if w2c.shape != (4, 4) or camera_matrix.shape != (3, 3):
        raise ValueError("camera transforms must have shape (4,4) and (3,3)")
    homogeneous = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
    camera = (w2c @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    projected = (camera_matrix @ camera.T).T
    uv = projected[:, :2] / np.maximum(depth[:, None], 1e-12)
    return uv.astype(np.float32), depth.astype(np.float32)


def transfer_observed_features(
    observed_rest_points: np.ndarray,
    structure_points: np.ndarray,
    observed_features: np.ndarray,
    observed_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill unseen/added nodes from the nearest directly observed material node."""

    observed = np.asarray(observed_rest_points, dtype=np.float64)
    structure = np.asarray(structure_points, dtype=np.float64)
    features = np.asarray(observed_features, dtype=np.float64)
    counts = np.asarray(observed_counts, dtype=np.int64).reshape(-1)
    if observed.shape != (len(features), 3) or len(counts) != len(features):
        raise ValueError("observed feature arrays are inconsistent")
    direct = np.flatnonzero(counts > 0)
    if len(direct) == 0:
        raise ValueError("no material node has a direct DINO observation")
    try:
        from scipy.spatial import cKDTree

        nearest = direct[cKDTree(observed[direct]).query(structure, k=1)[1]]
    except ImportError:
        nearest_chunks = []
        for start in range(0, len(structure), 512):
            delta = (
                structure[start : start + 512, None, :]
                - observed[direct][None, :, :]
            )
            nearest_chunks.append(
                direct[np.argmin(np.einsum("nki,nki->nk", delta, delta), axis=1)]
            )
        nearest = np.concatenate(nearest_chunks)
    result = features[nearest]
    direct_structure_count = min(len(observed), len(structure))
    direct_structure = np.arange(direct_structure_count)
    has_direct = counts[:direct_structure_count] > 0
    result[direct_structure[has_direct]] = features[direct_structure[has_direct]]
    contributor_count = np.zeros(len(structure), dtype=np.int32)
    contributor_count[direct_structure[has_direct]] = counts[:direct_structure_count][
        has_direct
    ].astype(np.int32)
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("transferred DINO features contain a zero vector")
    return (
        (result / norm).astype(np.float32),
        contributor_count,
        nearest.astype(np.int64),
    )


def _metric_depth(path: Path) -> np.ndarray:
    depth = np.asarray(np.load(path))
    if np.issubdtype(depth.dtype, np.integer) or float(np.nanmax(depth)) > 50.0:
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32)
    return depth


class CausalDinoNodeExtractor:
    """Load one pinned DINO model and extract node features case by case."""

    def __init__(
        self,
        *,
        model_name: str = "dinov2_vitl14_reg",
        image_size: int = 518,
        device: str = "cuda:0",
        depth_tolerance_m: float = 0.02,
        relative_depth_tolerance: float = 0.03,
    ) -> None:
        import torch
        from torchvision import transforms

        if image_size < 14 or image_size % 14 != 0:
            raise ValueError("DINO image size must be a positive multiple of 14")
        if depth_tolerance_m <= 0.0 or relative_depth_tolerance < 0.0:
            raise ValueError("depth tolerances must be nonnegative")
        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.image_size = image_size
        self.depth_tolerance_m = float(depth_tolerance_m)
        self.relative_depth_tolerance = float(relative_depth_tolerance)
        self.model = torch.hub.load(
            DINO_HUB_REPOSITORY,
            model_name,
            trust_repo=True,
        ).eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        checkpoint_candidates = sorted(
            (Path(torch.hub.get_dir()) / "checkpoints").glob(
                f"*{model_name.replace('dinov2_', '')}*"
            )
        )
        if not checkpoint_candidates:
            checkpoint_candidates = sorted(
                (Path(torch.hub.get_dir()) / "checkpoints").glob("*dinov2*")
            )
        if not checkpoint_candidates:
            raise RuntimeError("DINO checkpoint bytes could not be located for provenance")
        self.checkpoints = [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in checkpoint_candidates
        ]

    def _patch_tokens(self, image):
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            output = self.model.forward_features(tensor)
        tokens = output.get("x_norm_patchtokens")
        if tokens is None:
            raise RuntimeError("DINO model does not expose normalized patch tokens")
        patch_side = self.image_size // 14
        if tokens.shape[1] != patch_side * patch_side:
            raise RuntimeError("DINO patch-token shape disagrees with image size")
        return tokens.transpose(1, 2).reshape(1, -1, patch_side, patch_side)

    def extract_case(
        self,
        case_dir: str | Path,
        frame_ids: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        from PIL import Image
        import torch.nn.functional as functional

        root = Path(case_dir).resolve()
        with (root / "final_data.pkl").open("rb") as handle:
            data = pickle.load(handle)
        object_points = np.asarray(data["object_points"], dtype=np.float32)
        object_visibility = np.asarray(data["object_visibilities"], dtype=bool)
        motion_valid = np.asarray(data["object_motions_valid"], dtype=bool)
        structure_points = np.concatenate(
            (
                object_points[0],
                np.asarray(data["surface_points"], dtype=np.float32),
                np.asarray(data["interior_points"], dtype=np.float32),
            ),
            axis=0,
        )
        w2cs, intrinsics = _camera_data(root)
        feature_sum = None
        counts = np.zeros(object_points.shape[1], dtype=np.int32)
        source_records: list[dict[str, object]] = []
        for frame_id in sorted(set(int(value) for value in frame_ids)):
            if not 0 <= frame_id < len(object_points):
                raise ValueError(f"frame {frame_id} exceeds tracked object data")
            for camera in range(len(w2cs)):
                color_files = numeric_frame_paths(root / "color" / str(camera))
                if frame_id not in color_files:
                    raise FileNotFoundError(
                        root / "color" / str(camera) / f"{frame_id}.png"
                    )
                color_path = color_files[frame_id]
                depth_path = root / "depth" / str(camera) / f"{frame_id}.npy"
                mask_path = root / "mask" / str(camera) / "0" / f"{frame_id}.png"
                if not depth_path.is_file() or not mask_path.is_file():
                    raise FileNotFoundError(
                        f"missing causal visibility input for {root.name} frame "
                        f"{frame_id} camera {camera}"
                    )
                image = Image.open(color_path).convert("RGB")
                width, height = image.size
                depth_image = _metric_depth(depth_path)
                mask = np.asarray(Image.open(mask_path)) > 0
                if mask.ndim == 3:
                    mask = np.any(mask, axis=2)
                if depth_image.shape != (height, width) or mask.shape[:2] != (
                    height,
                    width,
                ):
                    raise ValueError("color, depth, and mask image shapes disagree")
                uv, projected_depth = project_world_points(
                    object_points[frame_id],
                    w2cs[camera],
                    intrinsics[camera],
                )
                pixel = np.rint(uv).astype(np.int64)
                in_image = (
                    (pixel[:, 0] >= 0)
                    & (pixel[:, 0] < width)
                    & (pixel[:, 1] >= 0)
                    & (pixel[:, 1] < height)
                    & (projected_depth > 0.0)
                )
                sampled_depth = np.zeros(len(pixel), dtype=np.float32)
                sampled_mask = np.zeros(len(pixel), dtype=bool)
                valid_pixel = np.flatnonzero(in_image)
                sampled_depth[valid_pixel] = depth_image[
                    pixel[valid_pixel, 1], pixel[valid_pixel, 0]
                ]
                sampled_mask[valid_pixel] = mask[
                    pixel[valid_pixel, 1], pixel[valid_pixel, 0]
                ]
                tolerance = np.maximum(
                    self.depth_tolerance_m,
                    self.relative_depth_tolerance * np.maximum(sampled_depth, 0.0),
                )
                valid = (
                    in_image
                    & sampled_mask
                    & (sampled_depth > 0.0)
                    & (np.abs(sampled_depth - projected_depth) <= tolerance)
                    & object_visibility[frame_id]
                    & motion_valid[frame_id]
                )
                tokens = self._patch_tokens(image)
                uv_normalized = uv.copy()
                uv_normalized[:, 0] = uv_normalized[:, 0] / max(width - 1, 1) * 2 - 1
                uv_normalized[:, 1] = uv_normalized[:, 1] / max(height - 1, 1) * 2 - 1
                grid = self.torch.from_numpy(uv_normalized).to(
                    self.device, dtype=tokens.dtype
                ).view(1, -1, 1, 2)
                sampled = (
                    functional.grid_sample(
                        tokens,
                        grid,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .squeeze(-1)
                    .transpose(0, 1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                sampled /= np.maximum(
                    np.linalg.norm(sampled, axis=1, keepdims=True), 1e-12
                )
                if feature_sum is None:
                    feature_sum = np.zeros_like(sampled, dtype=np.float64)
                feature_sum[valid] += sampled[valid]
                counts[valid] += 1
                source_records.append(
                    {
                        "frame_id": frame_id,
                        "camera": camera,
                        "direct_node_count": int(np.sum(valid)),
                        "color": {
                            "path": str(color_path),
                            "sha256": sha256_file(color_path),
                        },
                        "depth": {
                            "path": str(depth_path),
                            "sha256": sha256_file(depth_path),
                        },
                        "mask": {
                            "path": str(mask_path),
                            "sha256": sha256_file(mask_path),
                        },
                    }
                )
        if feature_sum is None:
            raise RuntimeError("no DINO keyframe was processed")
        observed_features = feature_sum / np.maximum(counts[:, None], 1)
        node_features, contributor_count, nearest = transfer_observed_features(
            object_points[0],
            structure_points,
            observed_features,
            counts,
        )
        provenance = {
            "contract": DINO_NODE_FEATURE_CONTRACT,
            "model_repository": DINO_HUB_REPOSITORY,
            "model_name": self.model_name,
            "model_checkpoints": self.checkpoints,
            "image_size": self.image_size,
            "frame_ids": sorted(set(int(value) for value in frame_ids)),
            "depth_tolerance_m": self.depth_tolerance_m,
            "relative_depth_tolerance": self.relative_depth_tolerance,
            "direct_observed_node_count": int(np.sum(counts > 0)),
            "structure_node_count": len(structure_points),
            "nearest_material_transfer_sha256": _integer_array_sha256(nearest),
            "sources": source_records,
            "final_data": {
                "path": str((root / "final_data.pkl").resolve()),
                "sha256": sha256_file(root / "final_data.pkl"),
            },
            "calibration": {
                "path": str((root / "calibrate.pkl").resolve()),
                "sha256": sha256_file(root / "calibrate.pkl"),
            },
            "metadata": {
                "path": str((root / "metadata.json").resolve()),
                "sha256": sha256_file(root / "metadata.json"),
            },
        }
        return node_features, contributor_count, provenance


def _integer_array_sha256(value: np.ndarray) -> str:
    import hashlib

    array = np.ascontiguousarray(value, dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()
