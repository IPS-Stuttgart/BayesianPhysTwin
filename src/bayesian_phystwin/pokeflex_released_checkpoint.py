"""Causal adapter for the released PokeFlex Kinect point-cloud checkpoint."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .pokeflex_bayesian_registration import voxel_cluster_centroids


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _points(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2 and result.shape[1] == 3, f"{name} must be Nx3")
    _require(len(result) > 0, f"{name} is empty")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


@dataclass(frozen=True)
class PokeFlexCheckpointPointCloud:
    """One normalized, padded frame and its deterministic preprocessing record."""

    points: np.ndarray
    input_point_count: int
    retained_point_count: int
    voxel_size_m: float


@dataclass(frozen=True)
class PokeFlexCheckpointPrediction:
    """Released-checkpoint prediction in the original metric world frame."""

    vertices_m: np.ndarray
    history_retained_point_counts: tuple[int, ...]
    history_voxel_sizes_m: tuple[float, ...]


def prepare_pokeflex_checkpoint_point_cloud(
    observation_views_m: Sequence[np.ndarray],
    *,
    template_center_m: np.ndarray,
    template_scale_m: float,
    maximum_points: int = 5000,
    initial_voxel_size_m: float = 0.001,
    voxel_step_m: float = 0.001,
) -> PokeFlexCheckpointPointCloud:
    """Match the released Kinect preprocessing without target-frame information."""

    _require(bool(observation_views_m), "at least one observation view is required")
    _require(maximum_points >= 16, "point-cloud capacity is too small")
    _require(initial_voxel_size_m > 0.0, "initial voxel size must be positive")
    _require(voxel_step_m > 0.0, "voxel step must be positive")
    center = np.asarray(template_center_m, dtype=np.float64)
    _require(center.shape == (3,), "template center must have shape (3,)")
    _require(np.all(np.isfinite(center)), "template center is non-finite")
    _require(np.isfinite(template_scale_m) and template_scale_m > 0.0, "template scale is invalid")

    views = tuple(
        _points(view, f"observation view {index}")
        for index, view in enumerate(observation_views_m)
    )
    fused = np.concatenate(views, axis=0)
    input_count = len(fused)
    voxel_size = initial_voxel_size_m
    retained = fused
    while len(retained) > maximum_points:
        retained = voxel_cluster_centroids(fused, voxel_size)
        voxel_size += voxel_step_m
    used_voxel_size = 0.0 if retained is fused else voxel_size - voxel_step_m
    normalized = (retained - center[None, :]) / template_scale_m
    padded = np.zeros((maximum_points, 3), dtype=np.float32)
    padded[: len(normalized)] = normalized.astype(np.float32)
    return PokeFlexCheckpointPointCloud(
        points=padded,
        input_point_count=input_count,
        retained_point_count=len(normalized),
        voxel_size_m=float(used_voxel_size),
    )


class PokeFlexReleasedCheckpoint:
    """Thin inference wrapper around the three official serialized modules."""

    history_frame_count = 5
    maximum_template_vertices = 11000

    def __init__(
        self,
        template_vertices_m: np.ndarray,
        *,
        pointcloud_encoder: object,
        attention_model: object,
        decoder: object,
        torch_module: object,
        device: str,
    ) -> None:
        template = _points(template_vertices_m, "template vertices")
        _require(
            len(template) <= self.maximum_template_vertices,
            "template exceeds the released checkpoint capacity",
        )
        self.template_vertices_m = template
        self.template_center_m = template.mean(axis=0)
        self.template_scale_m = float(
            np.max(np.linalg.norm(template - self.template_center_m[None, :], axis=1))
        )
        _require(self.template_scale_m > 0.0, "template scale is zero")
        self.pointcloud_encoder = pointcloud_encoder
        self.attention_model = attention_model
        self.decoder = decoder
        self.torch = torch_module
        self.device = device
        normalized = (template - self.template_center_m[None, :]) / self.template_scale_m
        padded = np.zeros((self.maximum_template_vertices, 3), dtype=np.float32)
        padded[: len(template)] = normalized.astype(np.float32)
        self._template_tensor = self.torch.as_tensor(
            padded, dtype=self.torch.float32, device=device
        ).unsqueeze(0)

    @classmethod
    def load(
        cls,
        template_vertices_m: np.ndarray,
        *,
        upstream_checkout: Path,
        checkpoint_root: Path,
        device: str | None = None,
    ) -> "PokeFlexReleasedCheckpoint":
        """Load trusted official checkpoint bytes after registering their code path."""

        checkout = Path(upstream_checkout).resolve()
        checkpoint = Path(checkpoint_root).resolve()
        _require((checkout / "models").is_dir(), "upstream model package is missing")
        for filename in (
            "pointcloud_encoder.pth",
            "attention_model.pth",
            "decoder.pth",
        ):
            _require((checkpoint / filename).is_file(), f"missing checkpoint: {filename}")
        checkout_text = str(checkout)
        if checkout_text not in sys.path:
            sys.path.insert(0, checkout_text)
        import torch

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        models = tuple(
            torch.load(checkpoint / filename, map_location=selected_device, weights_only=False)
            for filename in (
                "pointcloud_encoder.pth",
                "attention_model.pth",
                "decoder.pth",
            )
        )
        for model in models:
            model.eval()
        return cls(
            template_vertices_m,
            pointcloud_encoder=models[0],
            attention_model=models[1],
            decoder=models[2],
            torch_module=torch,
            device=selected_device,
        )

    def encode_frame(
        self, observation_views_m: Sequence[np.ndarray]
    ) -> tuple[object, PokeFlexCheckpointPointCloud]:
        """Encode one causal Kinect frame once for reuse in overlapping histories."""

        prepared = prepare_pokeflex_checkpoint_point_cloud(
            observation_views_m,
            template_center_m=self.template_center_m,
            template_scale_m=self.template_scale_m,
        )
        tensor = self.torch.as_tensor(
            prepared.points, dtype=self.torch.float32, device=self.device
        ).unsqueeze(0)
        with self.torch.no_grad():
            feature = self.pointcloud_encoder.encoder.forward(tensor.permute(0, 2, 1))[0]
        return feature, prepared

    def predict_from_encoded_history(
        self,
        encoded_history: Sequence[object],
        preprocessing_history: Sequence[PokeFlexCheckpointPointCloud],
    ) -> PokeFlexCheckpointPrediction:
        """Predict frame f from exactly the five encoded frames f-5 through f-1."""

        _require(
            len(encoded_history) == self.history_frame_count,
            "released checkpoint requires exactly five history frames",
        )
        _require(
            len(preprocessing_history) == self.history_frame_count,
            "preprocessing history length differs from encoded history",
        )
        stacked = self.torch.stack(tuple(encoded_history), dim=0).unsqueeze(1)
        with self.torch.no_grad():
            feature = self.attention_model.forward(stacked)
            normalized = self.decoder.forward(feature, self._template_tensor)
            normalized = normalized.reshape(1, self.maximum_template_vertices, 3)
            normalized = normalized[0, : len(self.template_vertices_m)].cpu().numpy()
        vertices = (
            normalized * self.template_scale_m + self.template_center_m[None, :]
        )
        return PokeFlexCheckpointPrediction(
            vertices_m=np.asarray(vertices, dtype=np.float64),
            history_retained_point_counts=tuple(
                item.retained_point_count for item in preprocessing_history
            ),
            history_voxel_sizes_m=tuple(
                item.voxel_size_m for item in preprocessing_history
            ),
        )
