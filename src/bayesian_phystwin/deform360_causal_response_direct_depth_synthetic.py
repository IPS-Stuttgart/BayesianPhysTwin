"""Production-path synthetic controls for V14 adaptive causal direct depth."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .deform360_causal_response_adaptive_query import (
    INFLATED_FALLBACK_ARM,
    STRICT_ARM,
    AdaptiveCausalResponseQueryConfig,
    build_adaptive_causal_response_query_schedule,
)
from .deform360_causal_response_direct_depth import (
    predict_adaptive_direct_depth_v14,
    scan_adaptive_direct_depth_v14,
)
from .deform360_causal_response_event import CausalResponseEventConfig
from .deform360_causal_response_update import BASELINE_ARM, CANDIDATE_ARM

CONTRACT = "deform360-causal-response-direct-depth-synthetic-v14"
POSITIVE = "positive-nonrigid"
PLACEBO_BIAS = "placebo-common-depth-bias"
PLACEBO_CROSS_PANEL = "placebo-cross-panel"
PLACEBO_NO_CONTACT = "placebo-no-contact"
CONTROL_KINDS = frozenset(
    {POSITIVE, PLACEBO_BIAS, PLACEBO_CROSS_PANEL, PLACEBO_NO_CONTACT}
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-direct-depth-synthetic-v14\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AdaptiveDirectDepthSyntheticConfigV14:
    """Frozen V14 synthetic-control dimensions and gate."""

    trial_count_per_arm: int = 6
    node_count: int = 16
    frame_count: int = 14
    prefix_frame_count: int = 11
    image_size: int = 256
    focal_length_px: float = 1600.0
    response_start_frame: int = 7
    minimum_positive_improvement_fraction: float = 0.10

    def __post_init__(self) -> None:
        _require(self.trial_count_per_arm >= 3, "too few V14 synthetic trials")
        _require(self.node_count >= 16, "too few V14 synthetic nodes")
        _require(
            self.frame_count > self.prefix_frame_count > self.response_start_frame,
            "V14 synthetic frame contract is invalid",
        )
        _require(self.image_size >= 128, "V14 synthetic image is too small")
        _require(
            np.isfinite(self.focal_length_px) and self.focal_length_px > 0.0,
            "V14 synthetic focal length is invalid",
        )
        _require(
            0.0 < self.minimum_positive_improvement_fraction < 1.0,
            "V14 synthetic improvement gate is invalid",
        )


@dataclass(frozen=True)
class AdaptiveDirectDepthSyntheticTrialV14:
    """One V14 production-path synthetic disposition."""

    trial_index: int
    carrier_arm: str
    control_kind: str
    candidate_applied: bool
    exact_baseline_fallback: bool
    baseline_future_rmse_m: float | None
    candidate_future_rmse_m: float | None

    def __post_init__(self) -> None:
        _require(self.trial_index >= 0, "V14 synthetic trial index is negative")
        _require(
            self.carrier_arm in {STRICT_ARM, INFLATED_FALLBACK_ARM},
            "V14 synthetic carrier arm is invalid",
        )
        _require(
            self.control_kind in CONTROL_KINDS,
            "V14 synthetic control kind is invalid",
        )
        for value in (
            self.baseline_future_rmse_m,
            self.candidate_future_rmse_m,
        ):
            _require(
                value is None or (np.isfinite(value) and value >= 0.0),
                "V14 synthetic outcome is invalid",
            )
        _require(
            (self.baseline_future_rmse_m is None)
            == (self.candidate_future_rmse_m is None),
            "V14 synthetic outcome pair is incomplete",
        )


@dataclass(frozen=True)
class AdaptiveDirectDepthSyntheticResultV14:
    """Checksummed V14 positive-control and placebo evidence."""

    config: AdaptiveDirectDepthSyntheticConfigV14
    trials: tuple[AdaptiveDirectDepthSyntheticTrialV14, ...]
    positive_detection_count: int
    placebo_admission_count: int
    placebo_exact_fallback_count: int
    mean_positive_baseline_rmse_m: float
    mean_positive_candidate_rmse_m: float
    gate_passed: bool
    artifact_sha256: str

    def __post_init__(self) -> None:
        expected = 4 * self.config.trial_count_per_arm
        _require(len(self.trials) == expected, "V14 synthetic trial count changed")
        _require(
            min(
                self.positive_detection_count,
                self.placebo_admission_count,
                self.placebo_exact_fallback_count,
            )
            >= 0,
            "V14 synthetic aggregate count is negative",
        )
        _require(
            np.isfinite(self.mean_positive_baseline_rmse_m)
            and np.isfinite(self.mean_positive_candidate_rmse_m)
            and self.mean_positive_baseline_rmse_m >= 0.0
            and self.mean_positive_candidate_rmse_m >= 0.0,
            "V14 synthetic aggregate RMSE is invalid",
        )
        _require(
            isinstance(self.artifact_sha256, str)
            and len(self.artifact_sha256) == 64
            and all(
                character in "0123456789abcdef"
                for character in self.artifact_sha256
            ),
            "V14 synthetic result digest is invalid",
        )

    @property
    def positive_trial_count(self) -> int:
        return 2 * self.config.trial_count_per_arm

    @property
    def placebo_trial_count(self) -> int:
        return 2 * self.config.trial_count_per_arm

    @property
    def positive_improvement_fraction(self) -> float:
        return 1.0 - (
            self.mean_positive_candidate_rmse_m
            / max(self.mean_positive_baseline_rmse_m, 1e-12)
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360AdaptiveDirectDepthSyntheticResultV14",
            "contract": CONTRACT,
            "config": asdict(self.config),
            "trials": [asdict(trial) for trial in self.trials],
            "positive_trial_count": self.positive_trial_count,
            "positive_detection_count": self.positive_detection_count,
            "placebo_trial_count": self.placebo_trial_count,
            "placebo_admission_count": self.placebo_admission_count,
            "placebo_exact_fallback_count": self.placebo_exact_fallback_count,
            "mean_positive_baseline_rmse_m": self.mean_positive_baseline_rmse_m,
            "mean_positive_candidate_rmse_m": self.mean_positive_candidate_rmse_m,
            "positive_improvement_fraction": self.positive_improvement_fraction,
            "gate_passed": self.gate_passed,
            "information_boundary": {
                "real_object_observation_read": False,
                "real_identity_or_metric_read": False,
                "production_v14_scan_and_prediction_wrappers_used": True,
                "strict_and_inflated_carriers_exercised": True,
                "held_v8_artifact_or_process_access": False,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def _geometry(
    config: AdaptiveDirectDepthSyntheticConfigV14,
    trial_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinate = np.linspace(-1.0, 1.0, config.node_count)
    frame_zero = np.column_stack(
        (
            0.12 * coordinate,
            0.045 * np.where(np.arange(config.node_count) % 2, 1.0, -1.0),
            np.full(config.node_count, 2.0),
        )
    )
    physical_mode = np.column_stack(
        (
            np.linspace(-0.0007, 0.0007, config.node_count),
            np.linspace(0.0005, -0.0005, config.node_count),
            np.zeros(config.node_count),
        )
    )
    physical = np.stack(
        [
            frame_zero + frame * physical_mode
            for frame in range(config.frame_count)
        ]
    )
    local_scale = 0.4 + 1.6 * (
        0.5 + 0.5 * np.sin(2.0 * np.pi * coordinate)
    )
    residual_per_frame = 2.5 * physical_mode * local_scale[:, None]
    graph_basis = np.zeros((config.node_count, 3, 8), dtype=np.float64)
    for mode in range(8):
        graph_basis[:, mode % 3, mode] = coordinate ** (mode % 4 + 1)
    return physical, residual_per_frame, graph_basis, np.full(config.node_count, 0.8)


def _cameras(
    config: AdaptiveDirectDepthSyntheticConfigV14,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    intrinsics = np.repeat(np.eye(3)[None], 8, axis=0)
    intrinsics[:, 0, 0] = config.focal_length_px
    intrinsics[:, 1, 1] = config.focal_length_px
    intrinsics[:, 0, 2] = config.image_size / 2
    intrinsics[:, 1, 2] = config.image_size / 2
    poses = np.repeat(np.eye(4)[None], 8, axis=0)
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    poses[:, 0, 3] = 0.01 * np.cos(angles)
    poses[:, 1, 3] = 0.01 * np.sin(angles)
    return intrinsics, poses, tuple(f"synthetic-camera-{index}" for index in range(8))


def _render_points(
    points_world_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count, node_count = points_world_m.shape[:2]
    camera_count = len(intrinsics)
    depths = np.zeros(
        (camera_count, frame_count, image_size, image_size),
        dtype=np.float64,
    )
    masks = np.zeros_like(depths, dtype=bool)
    homogeneous = np.concatenate(
        (points_world_m, np.ones((frame_count, node_count, 1))),
        axis=2,
    )
    for camera in range(camera_count):
        world_to_camera = np.linalg.inv(camera_to_world[camera])
        camera_points = np.einsum(
            "ij,fkj->fki",
            world_to_camera,
            homogeneous,
        )[..., :3]
        pixels = np.einsum("ij,fkj->fki", intrinsics[camera], camera_points)
        pixels = pixels[..., :2] / pixels[..., 2:]
        for frame in range(frame_count):
            for node in range(node_count):
                column, row = np.rint(pixels[frame, node]).astype(int)
                _require(
                    0 <= row < image_size and 0 <= column < image_size,
                    "V14 synthetic point left the image",
                )
                masks[camera, frame, row, column] = True
                depths[camera, frame, row, column] = camera_points[frame, node, 2]
    return depths, masks


def _trial(
    config: AdaptiveDirectDepthSyntheticConfigV14,
    trial_index: int,
    carrier_arm: str,
    control_kind: str,
) -> AdaptiveDirectDepthSyntheticTrialV14:
    physical, residual_per_frame, graph_basis, action_support = _geometry(
        config,
        trial_index,
    )
    intrinsics, poses, camera_ids = _cameras(config)
    frame_zero_depth, frame_zero_mask = _render_points(
        physical[:1],
        intrinsics,
        poses,
        config.image_size,
    )
    if carrier_arm == INFLATED_FALLBACK_ARM:
        frame_zero_depth[[2, 3, 6, 7]] = 0.0
        frame_zero_mask[[2, 3, 6, 7]] = False
    carrier = build_adaptive_causal_response_query_schedule(
        physical[0],
        graph_basis,
        action_support,
        intrinsics,
        poses,
        frame_zero_depth[:, 0],
        frame_zero_mask[:, 0],
        camera_ids=camera_ids,
        config=AdaptiveCausalResponseQueryConfig(
            prefix_frame_count=config.prefix_frame_count,
        ),
    )
    _require(carrier.arm == carrier_arm, "V14 synthetic carrier arm changed")

    observed_plus = physical[: config.prefix_frame_count].copy()
    for frame in range(config.response_start_frame, config.prefix_frame_count):
        observed_plus[frame] += (
            frame - config.response_start_frame + 1
        ) * residual_per_frame
    observed_minus = physical[: config.prefix_frame_count].copy()
    for frame in range(config.response_start_frame, config.prefix_frame_count):
        observed_minus[frame] -= (
            frame - config.response_start_frame + 1
        ) * residual_per_frame
    plus_depth, plus_mask = _render_points(
        observed_plus,
        intrinsics,
        poses,
        config.image_size,
    )
    if control_kind == PLACEBO_CROSS_PANEL:
        minus_depth, minus_mask = _render_points(
            observed_minus,
            intrinsics,
            poses,
            config.image_size,
        )
        depths = plus_depth
        masks = plus_mask
        validation = carrier.panels.validation_indices
        depths[validation] = minus_depth[validation]
        masks[validation] = minus_mask[validation]
    elif control_kind == PLACEBO_BIAS:
        depths, masks = _render_points(
            physical[: config.prefix_frame_count],
            intrinsics,
            poses,
            config.image_size,
        )
        depths[:, config.response_start_frame :] += np.where(
            masks[:, config.response_start_frame :],
            0.02,
            0.0,
        )
    else:
        depths, masks = plus_depth, plus_mask
    if carrier_arm == INFLATED_FALLBACK_ARM:
        depths[[2, 3, 6, 7]] = 0.0
        masks[[2, 3, 6, 7]] = False

    tactile = np.zeros(config.prefix_frame_count)
    if control_kind != PLACEBO_NO_CONTACT:
        tactile[config.response_start_frame :] = 1.0
    actuator = np.zeros((config.prefix_frame_count, 1, 3))
    actuator[:, 0, 0] = 0.002 * np.arange(config.prefix_frame_count)
    scan = scan_adaptive_direct_depth_v14(
        f"synthetic-v14-{carrier_arm}-{trial_index:02d}",
        physical,
        carrier,
        intrinsics,
        poses,
        depths,
        masks,
        action_support,
        tactile,
        actuator,
        event_config=CausalResponseEventConfig(
            endpoint_lag_frames=6,
            first_candidate_update_frame=8,
            last_candidate_update_frame=config.prefix_frame_count - 1,
        ),
    )
    report, arrays = predict_adaptive_direct_depth_v14(physical, scan)
    exact_fallback = (
        arrays[CANDIDATE_ARM].dtype == arrays[BASELINE_ARM].dtype
        and arrays[CANDIDATE_ARM].shape == arrays[BASELINE_ARM].shape
        and arrays[CANDIDATE_ARM].tobytes() == arrays[BASELINE_ARM].tobytes()
    )
    if control_kind == POSITIVE:
        update = scan.scan.selected_admission.update_frame
        persistent_residual = (
            update - config.response_start_frame + 1
        ) * residual_per_frame
        truth = physical.copy()
        truth[update + 1 :] += persistent_residual
        future = slice(update + 1, None)
        baseline_rmse = float(
            np.sqrt(np.mean(np.square(arrays[BASELINE_ARM][future] - truth[future])))
        )
        candidate_rmse = float(
            np.sqrt(np.mean(np.square(arrays[CANDIDATE_ARM][future] - truth[future])))
        )
    else:
        baseline_rmse = None
        candidate_rmse = None
    return AdaptiveDirectDepthSyntheticTrialV14(
        trial_index=trial_index,
        carrier_arm=carrier_arm,
        control_kind=control_kind,
        candidate_applied=bool(report["candidate_applied"]),
        exact_baseline_fallback=exact_fallback,
        baseline_future_rmse_m=baseline_rmse,
        candidate_future_rmse_m=candidate_rmse,
    )


def run_adaptive_direct_depth_synthetic_v14(
    config: AdaptiveDirectDepthSyntheticConfigV14 | None = None,
) -> AdaptiveDirectDepthSyntheticResultV14:
    """Exercise both V14 carrier arms through the production wrappers."""

    cfg = config or AdaptiveDirectDepthSyntheticConfigV14()
    trials: list[AdaptiveDirectDepthSyntheticTrialV14] = []
    placebo_kinds = (PLACEBO_BIAS, PLACEBO_CROSS_PANEL, PLACEBO_NO_CONTACT)
    for arm in (STRICT_ARM, INFLATED_FALLBACK_ARM):
        for index in range(cfg.trial_count_per_arm):
            trials.append(_trial(cfg, index, arm, POSITIVE))
        for index in range(cfg.trial_count_per_arm):
            trials.append(
                _trial(
                    cfg,
                    index,
                    arm,
                    placebo_kinds[index % len(placebo_kinds)],
                )
            )
    positives = [trial for trial in trials if trial.control_kind == POSITIVE]
    placebos = [trial for trial in trials if trial.control_kind != POSITIVE]
    positive_detection_count = sum(trial.candidate_applied for trial in positives)
    placebo_admission_count = sum(trial.candidate_applied for trial in placebos)
    placebo_exact_fallback_count = sum(
        trial.exact_baseline_fallback for trial in placebos
    )
    mean_baseline = float(
        np.mean([trial.baseline_future_rmse_m for trial in positives])
    )
    mean_candidate = float(
        np.mean([trial.candidate_future_rmse_m for trial in positives])
    )
    improvement = 1.0 - mean_candidate / max(mean_baseline, 1e-12)
    gate_passed = bool(
        positive_detection_count == len(positives)
        and placebo_admission_count == 0
        and placebo_exact_fallback_count == len(placebos)
        and improvement >= cfg.minimum_positive_improvement_fraction
    )
    provisional = AdaptiveDirectDepthSyntheticResultV14(
        config=cfg,
        trials=tuple(trials),
        positive_detection_count=positive_detection_count,
        placebo_admission_count=placebo_admission_count,
        placebo_exact_fallback_count=placebo_exact_fallback_count,
        mean_positive_baseline_rmse_m=mean_baseline,
        mean_positive_candidate_rmse_m=mean_candidate,
        gate_passed=gate_passed,
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor())
    return AdaptiveDirectDepthSyntheticResultV14(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )


def write_adaptive_direct_depth_synthetic_v14(
    path: str | Path,
    result: AdaptiveDirectDepthSyntheticResultV14,
) -> None:
    """Write one immutable V14 synthetic-control result."""

    output = Path(path)
    _require(not output.exists(), "V14 synthetic result already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.descriptor(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    validate_adaptive_direct_depth_synthetic_v14(output)


def validate_adaptive_direct_depth_synthetic_v14(
    path: str | Path,
) -> AdaptiveDirectDepthSyntheticResultV14:
    """Validate V14 production-path synthetic controls and their gate."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind")
        == "Deform360AdaptiveDirectDepthSyntheticResultV14"
        and payload.get("contract") == CONTRACT
        and payload.get("artifact_sha256") == _canonical_sha256(payload),
        "V14 synthetic result is invalid",
    )
    result = AdaptiveDirectDepthSyntheticResultV14(
        config=AdaptiveDirectDepthSyntheticConfigV14(**payload["config"]),
        trials=tuple(
            AdaptiveDirectDepthSyntheticTrialV14(**trial)
            for trial in payload["trials"]
        ),
        positive_detection_count=payload["positive_detection_count"],
        placebo_admission_count=payload["placebo_admission_count"],
        placebo_exact_fallback_count=payload["placebo_exact_fallback_count"],
        mean_positive_baseline_rmse_m=payload[
            "mean_positive_baseline_rmse_m"
        ],
        mean_positive_candidate_rmse_m=payload[
            "mean_positive_candidate_rmse_m"
        ],
        gate_passed=payload["gate_passed"],
        artifact_sha256=payload["artifact_sha256"],
    )
    _require(
        result.descriptor() == payload and result.gate_passed,
        "V14 synthetic controls did not pass their frozen gate",
    )
    return result


__all__ = [
    "CONTRACT",
    "AdaptiveDirectDepthSyntheticConfigV14",
    "AdaptiveDirectDepthSyntheticResultV14",
    "AdaptiveDirectDepthSyntheticTrialV14",
    "run_adaptive_direct_depth_synthetic_v14",
    "validate_adaptive_direct_depth_synthetic_v14",
    "write_adaptive_direct_depth_synthetic_v14",
]
