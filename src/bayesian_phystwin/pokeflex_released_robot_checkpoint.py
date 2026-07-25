"""Causal adapter for the released PokeFlex robot-data checkpoint."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


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
class PokeFlexRobotHistory:
    """Five normalized force/tool records consumed by the official checkpoint."""

    values: np.ndarray
    maximum_force_n: float


@dataclass(frozen=True)
class PokeFlexRobotPrediction:
    """Released robot-data prediction in the original metric world frame."""

    vertices_m: np.ndarray
    history_frame_count: int


def prepare_pokeflex_robot_history(
    records: Sequence[Mapping[str, object]],
    *,
    template_center_m: np.ndarray,
    template_scale_m: float,
    maximum_force_n: float = 100.0,
) -> PokeFlexRobotHistory:
    """Reproduce the official force and tool-position normalization."""

    _require(bool(records), "robot history is empty")
    center = np.asarray(template_center_m, dtype=np.float64)
    _require(center.shape == (3,), "template center must have shape (3,)")
    _require(np.all(np.isfinite(center)), "template center is non-finite")
    _require(
        np.isfinite(template_scale_m) and template_scale_m > 0.0,
        "template scale is invalid",
    )
    _require(
        np.isfinite(maximum_force_n) and maximum_force_n > 0.0,
        "force normalization is invalid",
    )

    values = np.zeros((len(records), 6), dtype=np.float32)
    for index, record in enumerate(records):
        wrench = np.asarray(record.get("forces"), dtype=np.float64)
        transform = np.asarray(record.get("T_WT"), dtype=np.float64)
        _require(
            wrench.ndim == 1 and len(wrench) >= 3,
            f"robot record {index} has no 3D force",
        )
        _require(
            transform.shape == (4, 4),
            f"robot record {index} has an invalid tool transform",
        )
        _require(
            np.all(np.isfinite(wrench[:3])) and np.all(np.isfinite(transform)),
            f"robot record {index} is non-finite",
        )
        values[index, :3] = (wrench[:3] / maximum_force_n).astype(np.float32)
        values[index, 3:] = (
            (transform[:3, 3] - center) / template_scale_m
        ).astype(np.float32)
    return PokeFlexRobotHistory(
        values=values,
        maximum_force_n=float(maximum_force_n),
    )


class PokeFlexReleasedRobotCheckpoint:
    """Thin inference wrapper around the official force-only checkpoint."""

    history_frame_count = 5
    maximum_template_vertices = 11000
    maximum_force_n = 100.0

    def __init__(
        self,
        template_vertices_m: np.ndarray,
        *,
        force_encoder: object,
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
            np.max(
                np.linalg.norm(
                    template - self.template_center_m[None, :],
                    axis=1,
                )
            )
        )
        _require(self.template_scale_m > 0.0, "template scale is zero")
        self.force_encoder = force_encoder
        self.attention_model = attention_model
        self.decoder = decoder
        self.torch = torch_module
        self.device = device

        normalized = (
            template - self.template_center_m[None, :]
        ) / self.template_scale_m
        padded = np.zeros((self.maximum_template_vertices, 3), dtype=np.float32)
        padded[: len(template)] = normalized.astype(np.float32)
        self._template_tensor = self.torch.as_tensor(
            padded,
            dtype=self.torch.float32,
            device=device,
        ).unsqueeze(0)

    @classmethod
    def load(
        cls,
        template_vertices_m: np.ndarray,
        *,
        upstream_checkout: Path,
        checkpoint_root: Path,
        device: str | None = None,
    ) -> "PokeFlexReleasedRobotCheckpoint":
        """Load trusted official checkpoint bytes after registering their code path."""

        checkout = Path(upstream_checkout).resolve()
        checkpoint = Path(checkpoint_root).resolve()
        _require((checkout / "models").is_dir(), "upstream model package is missing")
        filenames = (
            "force_encoder.pth",
            "attention_model.pth",
            "decoder.pth",
        )
        for filename in filenames:
            _require(
                (checkpoint / filename).is_file(),
                f"missing checkpoint: {filename}",
            )
        checkout_text = str(checkout)
        if checkout_text not in sys.path:
            sys.path.insert(0, checkout_text)
        import torch

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        models = tuple(
            torch.load(
                checkpoint / filename,
                map_location=selected_device,
                weights_only=False,
            )
            for filename in filenames
        )
        for model in models:
            model.eval()
        return cls(
            template_vertices_m,
            force_encoder=models[0],
            attention_model=models[1],
            decoder=models[2],
            torch_module=torch,
            device=selected_device,
        )

    def predict_from_records(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> PokeFlexRobotPrediction:
        """Predict frame f from exactly the robot records f-5 through f-1."""

        _require(
            len(records) == self.history_frame_count,
            "released robot checkpoint requires exactly five history frames",
        )
        history = prepare_pokeflex_robot_history(
            records,
            template_center_m=self.template_center_m,
            template_scale_m=self.template_scale_m,
            maximum_force_n=self.maximum_force_n,
        )
        tensor = self.torch.as_tensor(
            history.values,
            dtype=self.torch.float32,
            device=self.device,
        )
        with self.torch.no_grad():
            features = self.force_encoder.forward(tensor)
            features = features.view(
                self.history_frame_count,
                1,
                features.shape[-1],
            )
            features = self.attention_model.forward(features)
            normalized = self.decoder.forward(features, self._template_tensor)
            normalized = normalized.reshape(
                1,
                self.maximum_template_vertices,
                3,
            )
            normalized = normalized[
                0,
                : len(self.template_vertices_m),
            ].cpu().numpy()
        vertices = (
            normalized * self.template_scale_m
            + self.template_center_m[None, :]
        )
        return PokeFlexRobotPrediction(
            vertices_m=np.asarray(vertices, dtype=np.float64),
            history_frame_count=self.history_frame_count,
        )
