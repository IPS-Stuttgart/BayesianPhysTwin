"""Target-independent candidate construction for PokeFlex robot-data fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _field(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    _require(result.ndim == 2 and result.shape[1] == 3, f"{name} must be Nx3")
    _require(len(result) > 0, f"{name} is empty")
    _require(np.all(np.isfinite(result)), f"{name} contains non-finite values")
    return result


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=1))))


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.reshape(-1)
    second_flat = second.reshape(-1)
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(first_flat, second_flat) / denominator)


@dataclass(frozen=True)
class PokeFlexRobotFusionConfig:
    """Frozen source-development proposal bank with exact baseline fallback."""

    scales: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2)

    def __post_init__(self) -> None:
        _require(bool(self.scales), "fusion scale bank is empty")
        _require(self.scales[0] == 0.0, "first fusion scale must be exact fallback")
        _require(
            all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in self.scales),
            "fusion scale is outside [0, 1]",
        )
        _require(
            len(set(self.scales)) == len(self.scales),
            "fusion scale bank contains duplicates",
        )


def pokeflex_robot_fusion_candidates(
    baseline_vertices_m: np.ndarray,
    robot_vertices_m: np.ndarray,
    *,
    config: PokeFlexRobotFusionConfig | None = None,
) -> dict[str, np.ndarray]:
    """Interpolate toward an independent robot-data proposal."""

    baseline = _field(baseline_vertices_m, "baseline vertices")
    robot = _field(robot_vertices_m, "robot vertices")
    _require(baseline.shape == robot.shape, "robot proposal topology changed")
    cfg = config or PokeFlexRobotFusionConfig()
    candidates = {}
    for scale in cfg.scales:
        name = f"robot_convex_scale_{scale:g}"
        if scale == 0.0:
            candidate = baseline.copy()
            _require(
                np.array_equal(candidate, baseline),
                "fusion fallback changed baseline bytes",
            )
        else:
            candidate = baseline + scale * (robot - baseline)
        candidates[name] = candidate
    return candidates


def pokeflex_robot_fusion_features(
    baseline_vertices_m: np.ndarray,
    robot_vertices_m: np.ndarray,
    template_vertices_m: np.ndarray,
    robot_records: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    """Return causal diagnostics without reading target-frame geometry."""

    baseline = _field(baseline_vertices_m, "baseline vertices")
    robot = _field(robot_vertices_m, "robot vertices")
    template = _field(template_vertices_m, "template vertices")
    _require(
        baseline.shape == robot.shape == template.shape,
        "fusion inputs have different topology",
    )
    _require(len(robot_records) >= 2, "at least two robot records are required")
    forces = np.asarray(
        [np.asarray(record.get("forces"), dtype=np.float64)[:3] for record in robot_records],
        dtype=np.float64,
    )
    tools = np.asarray(
        [
            np.asarray(record.get("T_WT"), dtype=np.float64)[:3, 3]
            for record in robot_records
        ],
        dtype=np.float64,
    )
    _require(
        forces.shape == (len(robot_records), 3)
        and tools.shape == (len(robot_records), 3),
        "robot diagnostics have invalid shape",
    )
    _require(
        np.all(np.isfinite(forces)) and np.all(np.isfinite(tools)),
        "robot diagnostics are non-finite",
    )
    baseline_deformation = baseline - template
    robot_deformation = robot - template
    disagreement = robot - baseline
    return {
        "baseline_deformation_rms_m": _rms(baseline_deformation),
        "robot_deformation_rms_m": _rms(robot_deformation),
        "model_disagreement_rms_m": _rms(disagreement),
        "deformation_cosine": _cosine(
            baseline_deformation,
            robot_deformation,
        ),
        "force_norm_n": float(np.linalg.norm(forces[-1])),
        "force_delta_norm_n": float(np.linalg.norm(forces[-1] - forces[-2])),
        "tool_step_m": float(np.linalg.norm(tools[-1] - tools[-2])),
    }
