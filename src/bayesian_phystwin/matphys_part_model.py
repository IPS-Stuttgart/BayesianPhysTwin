"""Opt-in DINO part conditioning for MatPhys's released simple decoder."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


PART_AWARE_MODEL_CONTRACT = "simple-videomae-dino-part-conditioning-v1"


def summarize_part_spring_ratios(
    log_spring_y: np.ndarray,
    teacher_log_spring_y: np.ndarray,
    edge_part_index: np.ndarray,
) -> dict[str, object]:
    """Summarize the physically applied stiffness ratio by semantic part."""

    predicted = np.asarray(log_spring_y, dtype=float).reshape(-1)
    teacher = np.asarray(teacher_log_spring_y, dtype=float).reshape(-1)
    part_index = np.asarray(edge_part_index, dtype=int).reshape(-1)
    if len(predicted) == 0 or predicted.shape != teacher.shape:
        raise ValueError("predicted and teacher spring fields must have equal size")
    if len(part_index) != len(predicted) or np.any(part_index < 0):
        raise ValueError("edge part indices must cover the spring field")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(teacher)):
        raise ValueError("spring fields must be finite")
    ratio = np.exp(predicted - teacher)

    def statistics(values: np.ndarray) -> dict[str, float | int]:
        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }

    return {
        "overall": statistics(ratio),
        "by_part": [
            {"part": int(part), **statistics(ratio[part_index == part])}
            for part in sorted(set(part_index.tolist()))
        ],
    }


def install_part_aware_simple_model(
    training: Any,
    *,
    part_feature_dim: int = 1024,
    part_feature_scale: float = 1.0,
) -> type:
    """Patch a pinned MatPhys trainer so part descriptors affect spring fields.

    The public simple decoder loads ``part_features`` but never passes them to
    the model. This adapter adds a small zero-initialized projection into the
    existing per-part material embedding. A zero feature scale therefore
    reproduces the upstream decoder exactly, while a positive scale lets DINO
    descriptors distinguish parts that share the same material class.
    """

    import torch
    from torch import nn

    if part_feature_dim < 1:
        raise ValueError("part feature dimension must be positive")
    if not math.isfinite(float(part_feature_scale)) or part_feature_scale < 0.0:
        raise ValueError("part feature scale must be nonnegative")
    base_model = training.SimpleVideoMaterialPhysicsModel
    base_forward_case = training.forward_case

    class PartAwareSimpleVideoMaterialPhysicsModel(base_model):
        matphys_part_model_contract = PART_AWARE_MODEL_CONTRACT

        def __init__(self, *args, **kwargs):
            configured_dim = int(kwargs.pop("part_feature_dim", part_feature_dim))
            configured_scale = float(
                kwargs.pop("part_feature_scale", part_feature_scale)
            )
            super().__init__(*args, **kwargs)
            if (
                configured_dim < 1
                or not math.isfinite(configured_scale)
                or configured_scale < 0.0
            ):
                raise ValueError("invalid part feature adapter configuration")
            d_mat = int(self.material_codebook.codebook.shape[1])
            self.part_feature_dim = configured_dim
            self.part_feature_scale = configured_scale
            self.part_feature_encoder = nn.Sequential(
                nn.LayerNorm(configured_dim),
                nn.Linear(configured_dim, d_mat),
            )
            nn.init.zeros_(self.part_feature_encoder[-1].weight)
            nn.init.zeros_(self.part_feature_encoder[-1].bias)
            self._active_part_features = None

        def _global_hidden(self, pixel_values, material_dist, geo_stats):
            z_motion = self.motion_encoder(pixel_values)
            z_mat_part = self.material_codebook(material_dist)
            part_features = self._active_part_features
            if part_features is None:
                raise RuntimeError("part-aware MatPhys forward omitted part features")
            if part_features.ndim != 2 or part_features.shape[0] != z_mat_part.shape[0]:
                raise ValueError("part feature count must match material parts")
            if part_features.shape[1] != self.part_feature_dim:
                raise ValueError("part feature width disagrees with the adapter")
            part_delta = torch.tanh(self.part_feature_encoder(part_features))
            z_mat_part = z_mat_part + self.part_feature_scale * part_delta
            z_mat_global = z_mat_part.mean(dim=0)
            if geo_stats is None:
                z_geo_global = torch.zeros(
                    self.geo_stats_encoder[-1].out_features,
                    device=z_motion.device,
                    dtype=z_motion.dtype,
                )
            else:
                z_geo_global = self.geo_stats_encoder(
                    geo_stats.view(1, -1)
                ).squeeze(0)
            hidden = self.global_context(
                torch.cat([z_motion, z_mat_global, z_geo_global], dim=-1)
            )
            return hidden, z_mat_part

        def forward(self, *args, part_features=None, **kwargs):
            if part_features is None:
                raise ValueError("part_features are required by the part-aware decoder")
            if self._active_part_features is not None:
                raise RuntimeError("nested part-aware MatPhys forward is unsupported")
            self._active_part_features = part_features
            try:
                return super().forward(*args, **kwargs)
            finally:
                self._active_part_features = None

    def forward_case(model, batch, idx, device, pixel_values):
        impl = training._unwrap_model(model)
        ctrl_rest = batch["ctrl_rest_length"][idx].to(device)
        ctrl_rest = ctrl_rest.view(-1, 1) if ctrl_rest.numel() > 0 else None
        ctrl_part = batch.get("ctrl_part_idx")
        ctrl_part = (
            ctrl_part[idx].to(device)
            if ctrl_part is not None and ctrl_rest is not None
            else None
        )
        return impl(
            pixel_values=pixel_values,
            z_geo=batch["z_geo"][idx].to(device),
            material_dist=batch["material_dist"][idx].to(device),
            edge_part_idx=batch["edge_part_idx"][idx].to(device),
            part_features=batch["part_features"][idx].to(device),
            geo_stats=(
                batch["geo_stats"][idx].to(device)
                if "geo_stats" in batch
                else None
            ),
            ctrl_rest_length=ctrl_rest,
            ctrl_part_idx=ctrl_part,
        )

    PartAwareSimpleVideoMaterialPhysicsModel.__name__ = (
        "PartAwareSimpleVideoMaterialPhysicsModel"
    )
    training.SimpleVideoMaterialPhysicsModel = PartAwareSimpleVideoMaterialPhysicsModel
    training.forward_case = forward_case
    training._part_aware_upstream_forward_case = base_forward_case
    return PartAwareSimpleVideoMaterialPhysicsModel
