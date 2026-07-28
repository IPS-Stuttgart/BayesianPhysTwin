"""Causal TAPNext++ execution for dynamically born material queries.

The pinned TAPNext++ runtime cannot append queries to an existing recurrent
state. Each birth wave is therefore an independent online rollout that starts
at its causal birth frame and stops at the associated belief-update frame.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .deform360_dynamic_query import (
    DynamicQuerySchedule,
    projection_matrices,
)
from .tapnextpp_birth_association import (
    BirthAssociationConfig,
    propose_birth_query_pixels,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DynamicTAPNextPPRuntimeConfig:
    """Frozen tracker execution choices from the provider protocol."""

    input_resolution: int = 512
    support_points_per_query: int = 64
    support_radius_model_px: float = 32.0

    def __post_init__(self) -> None:
        _require(
            self.input_resolution > 0 and self.input_resolution % 16 == 0,
            "input resolution must be a positive patch multiple",
        )
        _require(
            self.support_points_per_query >= 0,
            "support-point count must be nonnegative",
        )
        _require(
            np.isfinite(self.support_radius_model_px)
            and self.support_radius_model_px > 0.0,
            "support radius must be finite and positive",
        )


@dataclass(frozen=True)
class DynamicBirthAssociations:
    """Causal pixel associations for every scheduled material identity."""

    query_points_world_m: np.ndarray
    query_points_xy: np.ndarray
    valid: np.ndarray
    association_probability: np.ndarray
    association_entropy: np.ndarray
    candidate_pixel_covariance_px2: np.ndarray
    candidate_count: np.ndarray
    camera_indices: np.ndarray
    camera_names: tuple[str, ...]

    def __post_init__(self) -> None:
        world = _readonly(self.query_points_world_m, dtype=np.float64)
        pixels = _readonly(self.query_points_xy, dtype=np.float64)
        valid = _readonly(self.valid, dtype=bool)
        probability = _readonly(
            self.association_probability,
            dtype=np.float64,
        )
        entropy = _readonly(self.association_entropy, dtype=np.float64)
        covariance = _readonly(
            self.candidate_pixel_covariance_px2,
            dtype=np.float64,
        )
        count = _readonly(self.candidate_count, dtype=np.int64)
        cameras = _readonly(self.camera_indices, dtype=np.int64)
        _require(
            world.ndim == 2 and world.shape[1] == 3,
            "query points must have shape (N, 3)",
        )
        camera_count, entity_count = valid.shape
        _require(
            pixels.shape == (camera_count, entity_count, 2),
            "query pixels must have shape (C, N, 2)",
        )
        for name, values in (
            ("association_probability", probability),
            ("association_entropy", entropy),
            ("candidate_count", count),
        ):
            _require(
                values.shape == valid.shape,
                f"{name} shape differs from validity",
            )
        _require(
            covariance.shape == (camera_count, entity_count, 2, 2),
            "candidate covariance must have shape (C, N, 2, 2)",
        )
        _require(
            world.shape[0] == entity_count,
            "world-query count differs from pixel-query count",
        )
        _require(
            cameras.shape == (camera_count,)
            and len(self.camera_names) == camera_count,
            "camera identities differ from association cameras",
        )
        _require(
            len(set(map(int, cameras))) == camera_count
            and len(set(self.camera_names)) == camera_count,
            "association cameras repeat",
        )
        _require(np.all(np.isfinite(world)), "world queries are not finite")
        _require(
            np.all(np.isfinite(probability))
            and np.all((probability >= 0.0) & (probability <= 1.0)),
            "association probability must lie in [0, 1]",
        )
        _require(
            np.all(np.isfinite(entropy))
            and np.all((entropy >= 0.0) & (entropy <= 1.0)),
            "association entropy must lie in [0, 1]",
        )
        _require(
            np.all(count >= 0),
            "candidate counts must be nonnegative",
        )
        _require(
            np.all(np.isfinite(pixels[valid]))
            and np.all(np.isfinite(covariance[valid])),
            "valid associations must have finite geometry",
        )
        _require(
            np.all(count[valid] > 0),
            "valid associations must have candidates",
        )
        object.__setattr__(self, "query_points_world_m", world)
        object.__setattr__(self, "query_points_xy", pixels)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "association_probability", probability)
        object.__setattr__(self, "association_entropy", entropy)
        object.__setattr__(
            self,
            "candidate_pixel_covariance_px2",
            covariance,
        )
        object.__setattr__(self, "candidate_count", count)
        object.__setattr__(self, "camera_indices", cameras)


@dataclass(frozen=True)
class DynamicTAPNextPPRuntimeResult:
    """Per-camera tracks with explicit inactive regions."""

    tracks_xy: np.ndarray
    visibility_probability: np.ndarray
    active: np.ndarray
    rollout_count: int
    model_frame_count: int
    elapsed_seconds: float

    def __post_init__(self) -> None:
        tracks = _readonly(self.tracks_xy, dtype=np.float64)
        visibility = _readonly(
            self.visibility_probability,
            dtype=np.float64,
        )
        active = _readonly(self.active, dtype=bool)
        _require(
            tracks.ndim == 4 and tracks.shape[-1] == 2,
            "tracks must have shape (C, T, N, 2)",
        )
        _require(
            visibility.shape == tracks.shape[:-1],
            "visibility shape differs from tracks",
        )
        _require(
            active.shape == tracks.shape[:-1],
            "active mask shape differs from tracks",
        )
        _require(
            np.all(np.isfinite(tracks[active])),
            "active tracks are not finite",
        )
        _require(
            np.all(np.isnan(tracks[~active])),
            "inactive tracks must remain NaN",
        )
        _require(
            np.all(np.isfinite(visibility))
            and np.all((visibility >= 0.0) & (visibility <= 1.0)),
            "visibility probabilities must lie in [0, 1]",
        )
        _require(
            np.all(visibility[~active] == 0.0),
            "inactive visibility must remain zero",
        )
        _require(self.rollout_count >= 0, "rollout count is negative")
        _require(self.model_frame_count >= 0, "model-frame count is negative")
        _require(
            np.isfinite(self.elapsed_seconds) and self.elapsed_seconds >= 0.0,
            "runtime must be finite and nonnegative",
        )
        object.__setattr__(self, "tracks_xy", tracks)
        object.__setattr__(self, "visibility_probability", visibility)
        object.__setattr__(self, "active", active)


def _grid_support_points(
    count: int,
    width: float,
    height: float,
) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    columns = max(1, round(float(np.sqrt(count * width / height))))
    rows = max(1, int(np.ceil(count / columns)))
    xs = (np.arange(columns) + 0.5) * (width / columns)
    ys = (np.arange(rows) + 0.5) * (height / rows)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack(
        [grid_x.ravel(), grid_y.ravel()],
        axis=-1,
    ).astype(np.float32)[:count]


def _local_support_points(
    query_xy: np.ndarray,
    count_per_query: int,
    radius_x: float,
    radius_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    if count_per_query <= 0 or not len(query_xy):
        return np.zeros((0, 2), dtype=np.float32)
    output: list[np.ndarray] = []
    for query_x, query_y in np.asarray(query_xy, dtype=np.float32):
        local = _grid_support_points(
            count_per_query,
            2.0 * radius_x,
            2.0 * radius_y,
        )
        local -= np.asarray([radius_x, radius_y], dtype=np.float32)
        local += np.asarray([query_x, query_y], dtype=np.float32)
        local[:, 0] = np.clip(local[:, 0], 0, width - 1)
        local[:, 1] = np.clip(local[:, 1], 0, height - 1)
        output.append(local)
    return np.concatenate(output, axis=0).astype(np.float32)


def build_dynamic_birth_associations(
    schedule: DynamicQuerySchedule,
    physical_positions_m: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    *,
    config: BirthAssociationConfig | None = None,
) -> DynamicBirthAssociations:
    """Associate each birth using only that frame's causal depth and mask."""

    positions = np.asarray(physical_positions_m, dtype=np.float64)
    matrices = np.asarray(intrinsics, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    depths = np.asarray(depths_m, dtype=np.float64)
    masks = np.asarray(object_masks, dtype=bool)
    _require(
        positions.ndim == 3 and positions.shape[2] == 3,
        "physical positions must have shape (T, N, 3)",
    )
    _require(
        matrices.ndim == 3 and matrices.shape[1:] == (3, 3),
        "intrinsics must have shape (C, 3, 3)",
    )
    _require(
        poses.shape == (len(matrices), 4, 4),
        "camera poses must have shape (C, 4, 4)",
    )
    _require(
        depths.ndim == 4 and depths.shape[:2] == (len(matrices), len(positions)),
        "depths must have shape (C, T, H, W)",
    )
    _require(masks.shape == depths.shape, "object masks differ from depths")
    entities = np.asarray(schedule.entity_ids, dtype=np.int64)
    births = np.asarray(schedule.birth_frames, dtype=np.int64)
    _require(
        np.all((entities >= 0) & (entities < positions.shape[1])),
        "scheduled entity exceeds the physical state",
    )
    _require(
        np.all((births >= 0) & (births < len(positions))),
        "scheduled birth exceeds the causal inputs",
    )
    camera_indices = np.asarray(
        schedule.camera_panel.camera_indices,
        dtype=np.int64,
    )
    _require(
        np.all((camera_indices >= 0) & (camera_indices < len(matrices))),
        "selected camera exceeds the causal inputs",
    )
    selected_intrinsics = matrices[camera_indices]
    selected_poses = poses[camera_indices]
    selected_projections = projection_matrices(
        selected_intrinsics,
        selected_poses,
    )
    entity_count = len(entities)
    camera_count = len(camera_indices)
    query_world = positions[births, entities]
    query_xy = np.full((camera_count, entity_count, 2), np.nan)
    valid = np.zeros((camera_count, entity_count), dtype=bool)
    probability = np.zeros((camera_count, entity_count))
    entropy = np.ones((camera_count, entity_count))
    covariance = np.full((camera_count, entity_count, 2, 2), np.nan)
    candidate_count = np.zeros((camera_count, entity_count), dtype=np.int64)
    for birth in sorted(set(map(int, births))):
        rows = np.flatnonzero(births == birth)
        proposal = propose_birth_query_pixels(
            query_world[rows],
            selected_projections,
            selected_poses,
            depths[camera_indices, birth],
            masks[camera_indices, birth],
            config=config,
        )
        query_xy[:, rows] = proposal["query_points_xy"]
        valid[:, rows] = proposal["valid"]
        probability[:, rows] = proposal["association_probability"]
        entropy[:, rows] = proposal["association_entropy"]
        covariance[:, rows] = proposal["candidate_pixel_covariance_px2"]
        candidate_count[:, rows] = proposal["candidate_count"]
    return DynamicBirthAssociations(
        query_points_world_m=query_world,
        query_points_xy=query_xy,
        valid=valid,
        association_probability=probability,
        association_entropy=entropy,
        candidate_pixel_covariance_px2=covariance,
        candidate_count=candidate_count,
        camera_indices=camera_indices,
        camera_names=schedule.camera_panel.camera_names,
    )


def _track_frame_with_probability(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    query_points_xy: np.ndarray | None,
    state: Any,
    tapnext_utils: Any,
) -> tuple[np.ndarray, np.ndarray, Any]:
    import torch

    if query_points_xy is None and state is None:
        raise ValueError("query points are required for a fresh tracker state")
    height, width = frame_bgr.shape[:2]
    frame_tensor = tapnext_utils.preprocess_frame(
        frame_bgr,
        model.device,
        model.input_resolution,
    )
    query_tensor = None
    if query_points_xy is not None:
        model_points = tapnext_utils.display_to_model(
            query_points_xy,
            height,
            width,
            model.MODEL_SIZE,
        )
        query_tensor = tapnext_utils.make_query_tensor(
            model_points,
            model.device,
        )
    context = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if model.device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )
    with torch.no_grad(), context:
        tracks, _, visibility_logits, new_state = model._model(
            video=frame_tensor,
            query_points=query_tensor,
            state=state,
        )
    tracks_xy = tracks[0, 0].detach().float().cpu().numpy()[:, ::-1].copy()
    positions_xy = tapnext_utils.model_to_display(
        tracks_xy,
        height,
        width,
        model.MODEL_SIZE,
    )
    probability = (
        torch.sigmoid(visibility_logits[0, 0, :, 0])
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    return (
        positions_xy.astype(np.float32),
        probability.astype(np.float32),
        new_state,
    )


TrackerStep = Callable[
    [Any, np.ndarray, np.ndarray | None, Any, Any],
    tuple[np.ndarray, np.ndarray, Any],
]


def run_dynamic_tapnextpp_births(
    model: Any,
    rgbs: np.ndarray,
    associations: DynamicBirthAssociations,
    birth_frames: np.ndarray,
    update_frames: np.ndarray,
    tapnext_utils: Any,
    *,
    config: DynamicTAPNextPPRuntimeConfig | None = None,
    tracker_step: TrackerStep | None = None,
) -> DynamicTAPNextPPRuntimeResult:
    """Run one independent recurrent rollout per camera and birth wave."""

    cfg = config or DynamicTAPNextPPRuntimeConfig()
    frames = np.asarray(rgbs, dtype=np.uint8)
    births = np.asarray(birth_frames, dtype=np.int64)
    updates = np.asarray(update_frames, dtype=np.int64)
    _require(
        frames.ndim == 5 and frames.shape[-1] == 3,
        "RGB frames must have shape (C, T, H, W, 3)",
    )
    camera_count, frame_count, height, width, _ = frames.shape
    entity_count = associations.query_points_xy.shape[1]
    _require(
        camera_count == len(associations.camera_indices),
        "RGB cameras differ from association cameras",
    )
    _require(
        births.shape == updates.shape == (entity_count,),
        "birth and update frames must match association entities",
    )
    _require(
        np.all((births >= 0) & (births < frame_count)),
        "birth frame is outside the causal video",
    )
    _require(
        np.all((updates >= births) & (updates < frame_count)),
        "update frame is outside the causal video",
    )
    step = tracker_step or (
        lambda model_arg,
        frame_arg,
        query_arg,
        state_arg,
        utils_arg: _track_frame_with_probability(
            model_arg,
            frame_arg,
            query_points_xy=query_arg,
            state=state_arg,
            tapnext_utils=utils_arg,
        )
    )
    tracks = np.full(
        (camera_count, frame_count, entity_count, 2),
        np.nan,
        dtype=np.float64,
    )
    visibility = np.zeros(
        (camera_count, frame_count, entity_count),
        dtype=np.float64,
    )
    active = np.zeros_like(visibility, dtype=bool)
    rollout_count = 0
    model_frame_count = 0
    radius_x = cfg.support_radius_model_px * (
        width / cfg.input_resolution
    )
    radius_y = cfg.support_radius_model_px * (
        height / cfg.input_resolution
    )
    started = time.perf_counter()
    for birth, update in sorted(
        set(zip(map(int, births), map(int, updates), strict=True))
    ):
        group = np.flatnonzero((births == birth) & (updates == update))
        for camera in range(camera_count):
            rows = group[associations.valid[camera, group]]
            if not len(rows):
                continue
            real_queries = associations.query_points_xy[camera, rows].astype(
                np.float32,
            )
            supports = _local_support_points(
                real_queries,
                cfg.support_points_per_query,
                radius_x,
                radius_y,
                width,
                height,
            )
            all_queries = np.concatenate([real_queries, supports], axis=0)
            state = None
            rollout_count += 1
            for frame in range(birth, update + 1):
                frame_bgr = frames[camera, frame, :, :, ::-1].copy()
                positions, probabilities, state = step(
                    model,
                    frame_bgr,
                    all_queries if frame == birth else None,
                    state,
                    tapnext_utils,
                )
                _require(
                    positions.shape == (len(all_queries), 2)
                    and probabilities.shape == (len(all_queries),),
                    "tracker output differs from submitted query count",
                )
                _require(
                    np.all(np.isfinite(positions))
                    and np.all(np.isfinite(probabilities))
                    and np.all(
                        (probabilities >= 0.0) & (probabilities <= 1.0)
                    ),
                    "tracker output is invalid",
                )
                tracks[camera, frame, rows] = positions[: len(rows)]
                visibility[camera, frame, rows] = probabilities[: len(rows)]
                active[camera, frame, rows] = True
                model_frame_count += 1
    return DynamicTAPNextPPRuntimeResult(
        tracks_xy=tracks,
        visibility_probability=visibility,
        active=active,
        rollout_count=rollout_count,
        model_frame_count=model_frame_count,
        elapsed_seconds=time.perf_counter() - started,
    )


__all__ = [
    "DynamicBirthAssociations",
    "DynamicTAPNextPPRuntimeConfig",
    "DynamicTAPNextPPRuntimeResult",
    "build_dynamic_birth_associations",
    "run_dynamic_tapnextpp_births",
]
