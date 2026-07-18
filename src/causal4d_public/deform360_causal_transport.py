"""Causal contact transport for Deform360 reusable-twin prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class CausalContactTransportConfig:
    """Outcome-independent controls for one contact-transport expert."""

    controller_group_size: int = 768
    maximum_contact_distance_m: float = 0.01
    opening_contact_threshold_m: float = 0.0795396
    confirmation_frames: int = 1
    base_support_scale_m: float = 0.003
    support_growth_per_travel: float = 0.1
    initial_contact_gain: float = 0.5
    acquired_contact_gain: float = 0.0
    transform_mode: str = "translation"

    def validate(self) -> None:
        _require(self.controller_group_size >= 1, "controller group size is invalid")
        _require(
            np.isfinite(self.maximum_contact_distance_m)
            and self.maximum_contact_distance_m > 0.0,
            "contact distance must be finite and positive",
        )
        _require(
            np.isfinite(self.opening_contact_threshold_m)
            and self.opening_contact_threshold_m > 0.0,
            "opening threshold must be finite and positive",
        )
        _require(self.confirmation_frames >= 1, "confirmation count is invalid")
        _require(
            np.isfinite(self.base_support_scale_m)
            and self.base_support_scale_m > 0.0,
            "base support scale must be finite and positive",
        )
        _require(
            np.isfinite(self.support_growth_per_travel)
            and self.support_growth_per_travel >= 0.0,
            "support growth must be finite and non-negative",
        )
        for name, value in (
            ("initial contact gain", self.initial_contact_gain),
            ("acquired contact gain", self.acquired_contact_gain),
        ):
            _require(
                np.isfinite(value) and 0.0 <= value <= 1.0,
                f"{name} must lie in [0, 1]",
            )
        _require(
            self.transform_mode in {"translation", "se3"},
            "transport mode must be translation or se3",
        )


@dataclass(frozen=True)
class CausalContactTransportResult:
    """One prediction and its auditable causal-contact diagnostics."""

    prediction_m: np.ndarray
    contact_active: np.ndarray
    controller_to_initial_object_distance_m: np.ndarray
    onset_frames: tuple[int | None, ...]
    maximum_transport_weight: float
    exact_persistence: bool

    def __post_init__(self) -> None:
        prediction = np.asarray(self.prediction_m, dtype=np.float64)
        active = np.asarray(self.contact_active, dtype=bool)
        distance = np.asarray(
            self.controller_to_initial_object_distance_m, dtype=np.float64
        )
        _require(
            prediction.ndim == 3
            and prediction.shape[2] == 3
            and np.all(np.isfinite(prediction)),
            "transport prediction must be finite (T,N,3)",
        )
        _require(
            active.ndim == 2
            and distance.shape == active.shape
            and active.shape[0] == prediction.shape[0],
            "contact diagnostics do not share the prediction frame axis",
        )
        _require(
            len(self.onset_frames) == active.shape[1]
            and np.all(np.isfinite(distance))
            and np.all(distance >= 0.0),
            "contact onset or distance diagnostics are invalid",
        )
        for name, values in (
            ("prediction_m", prediction),
            ("contact_active", active),
            ("controller_to_initial_object_distance_m", distance),
        ):
            copied = values.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "onset_frames": [
                None if value is None else int(value) for value in self.onset_frames
            ],
            "contact_fraction_by_group": np.mean(
                self.contact_active, axis=0
            ).tolist(),
            "minimum_controller_to_initial_object_distance_m_by_group": np.min(
                self.controller_to_initial_object_distance_m, axis=0
            ).tolist(),
            "maximum_transport_weight": self.maximum_transport_weight,
            "exact_persistence": self.exact_persistence,
        }


def _validated_inputs(
    initial_object_points_m: np.ndarray,
    controller_points_m: np.ndarray,
    openings_m: np.ndarray,
    config: CausalContactTransportConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config.validate()
    initial = np.asarray(initial_object_points_m, dtype=np.float64)
    controllers = np.asarray(controller_points_m, dtype=np.float64)
    openings = np.asarray(openings_m, dtype=np.float64)
    if openings.ndim == 1:
        openings = openings[:, None]
    _require(
        initial.ndim == 2
        and initial.shape[1] == 3
        and len(initial) >= 1
        and np.all(np.isfinite(initial)),
        "initial object points must be finite (N,3)",
    )
    _require(
        controllers.ndim == 3
        and controllers.shape[0] >= 2
        and controllers.shape[2] == 3
        and np.all(np.isfinite(controllers)),
        "controller points must be finite (T,P,3)",
    )
    _require(
        openings.ndim == 2
        and openings.shape[0] == controllers.shape[0]
        and np.all(np.isfinite(openings)),
        "openings must be finite (T,G)",
    )
    _require(
        controllers.shape[1]
        == openings.shape[1] * config.controller_group_size,
        "controller points do not match opening groups",
    )
    return initial, controllers, openings


def infer_latched_contact_schedule(
    initial_object_points_m: np.ndarray,
    controller_points_m: np.ndarray,
    openings_m: np.ndarray,
    *,
    config: CausalContactTransportConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[int | None, ...]]:
    """Infer causal contact onset from frame-zero geometry and known actions."""

    initial, controllers, openings = _validated_inputs(
        initial_object_points_m, controller_points_m, openings_m, config
    )
    frame_count = len(controllers)
    group_count = openings.shape[1]
    object_tree = cKDTree(initial)
    distance = np.empty((frame_count, group_count), dtype=np.float64)
    active = np.zeros((frame_count, group_count), dtype=bool)
    onsets: list[int | None] = []
    for group in range(group_count):
        start = group * config.controller_group_size
        stop = start + config.controller_group_size
        for frame in range(frame_count):
            distance[frame, group] = float(
                np.min(object_tree.query(controllers[frame, start:stop], k=1)[0])
            )
        evidence = (
            distance[:, group] <= config.maximum_contact_distance_m
        ) & (openings[:, group] <= config.opening_contact_threshold_m)
        run = 0
        onset = None
        for frame, supported in enumerate(evidence):
            run = run + 1 if supported else 0
            if run >= config.confirmation_frames:
                # Activation starts when the confirming evidence is available.
                onset = frame
                break
        if onset is not None:
            active[onset:, group] = True
        onsets.append(onset)
    return active, distance, tuple(onsets)


def _rigid_transform(
    source: np.ndarray, target: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    rotation = np.eye(3)
    if mode == "se3":
        left, _, right_t = np.linalg.svd(
            (source - source_center).T @ (target - target_center)
        )
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1] *= -1.0
            rotation = right_t.T @ left.T
    return rotation, source_center, target_center


def causal_contact_transport_prediction(
    initial_object_points_m: np.ndarray,
    controller_points_m: np.ndarray,
    openings_m: np.ndarray,
    *,
    config: CausalContactTransportConfig,
) -> CausalContactTransportResult:
    """Skin frame-zero points to causally supported gripper motion."""

    initial, controllers, openings = _validated_inputs(
        initial_object_points_m, controller_points_m, openings_m, config
    )
    active, contact_distance, onsets = infer_latched_contact_schedule(
        initial,
        controllers,
        openings,
        config=config,
    )
    frame_count = len(controllers)
    prediction = np.repeat(initial[None], frame_count, axis=0)
    weighted_delta = np.zeros_like(prediction)
    total_weight = np.zeros((frame_count, len(initial)), dtype=np.float64)
    strongest_weight = np.zeros_like(total_weight)
    maximum_weight = 0.0
    for group, onset in enumerate(onsets):
        if onset is None:
            continue
        gain = (
            config.initial_contact_gain
            if onset == 0
            else config.acquired_contact_gain
        )
        if gain == 0.0:
            continue
        start = group * config.controller_group_size
        stop = start + config.controller_group_size
        reference = controllers[onset, start:stop]
        point_distance = cKDTree(reference).query(initial, k=1)[0]
        reference_center = np.mean(reference, axis=0)
        for frame in range(onset, frame_count):
            rotation, source_center, target_center = _rigid_transform(
                reference,
                controllers[frame, start:stop],
                config.transform_mode,
            )
            travel = float(np.linalg.norm(target_center - reference_center))
            scale = (
                config.base_support_scale_m
                + config.support_growth_per_travel * travel
            )
            weight = gain * np.exp(-point_distance / scale)
            moved = (initial - source_center) @ rotation.T + target_center
            weighted_delta[frame] += weight[:, None] * (moved - initial)
            total_weight[frame] += weight
            strongest_weight[frame] = np.maximum(strongest_weight[frame], weight)
            maximum_weight = max(maximum_weight, float(np.max(weight)))
    mean_delta = np.zeros_like(weighted_delta)
    np.divide(
        weighted_delta,
        total_weight[:, :, None],
        out=mean_delta,
        where=total_weight[:, :, None] > 0.0,
    )
    prediction += strongest_weight[:, :, None] * mean_delta
    exact_persistence = bool(
        np.array_equal(prediction, np.repeat(initial[None], frame_count, axis=0))
    )
    return CausalContactTransportResult(
        prediction_m=prediction,
        contact_active=active,
        controller_to_initial_object_distance_m=contact_distance,
        onset_frames=onsets,
        maximum_transport_weight=maximum_weight,
        exact_persistence=exact_persistence,
    )


__all__ = [
    "CausalContactTransportConfig",
    "CausalContactTransportResult",
    "causal_contact_transport_prediction",
    "infer_latched_contact_schedule",
]
