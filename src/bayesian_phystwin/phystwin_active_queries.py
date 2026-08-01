"""Causal physics-guided query planning for multiview point trackers.

The planner uses only an action-conditioned physical rollout, projected view
support, and optional prefix-time tracker support.  It never reads future
measurements, residuals, or evaluation targets.  Selected graph identities can
therefore seed a causal tracker before a bias-aware Bayesian state update.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def _integer_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    integer = int(value)
    _require(integer >= minimum, f"{name} must be an integer >= {minimum}")
    return integer


def _integer_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    integer_dtype = np.issubdtype(raw.dtype, np.integer) and not np.issubdtype(
        raw.dtype, np.bool_
    )
    _require(
        raw.size == 0 or integer_dtype,
        f"{name} must contain integers",
    )
    return np.asarray(raw, dtype=np.int64).copy()


@dataclass(frozen=True)
class PhysicsGuidedQueryConfig:
    """Selection and causal reseeding settings for one rollout prefix."""

    query_count: int = 8
    maximum_reseeds: int = 8
    minimum_motion_m: float = 0.002
    minimum_camera_support: int = 2
    support_probability_threshold: float = 0.5
    contact_exclusion_radius_m: float = 0.0
    contact_exclusion_fraction: float = 0.05
    motion_weight: float = 1.0
    visibility_weight: float = 0.5
    mode_information_weight: float = 1.0
    spatial_diversity_weight: float = 1.0
    contact_distance_weight: float = 0.25
    mode_regularization: float = 1e-3
    reseed_patience_frames: int = 2
    minimum_reseed_interval_frames: int = 2

    def __post_init__(self) -> None:
        discrete_values = {
            "query_count": _integer_at_least(
                self.query_count, name="query_count", minimum=1
            ),
            "maximum_reseeds": _integer_at_least(
                self.maximum_reseeds, name="maximum_reseeds", minimum=0
            ),
            "minimum_camera_support": _integer_at_least(
                self.minimum_camera_support,
                name="minimum_camera_support",
                minimum=2,
            ),
            "reseed_patience_frames": _integer_at_least(
                self.reseed_patience_frames,
                name="reseed_patience_frames",
                minimum=1,
            ),
            "minimum_reseed_interval_frames": _integer_at_least(
                self.minimum_reseed_interval_frames,
                name="minimum_reseed_interval_frames",
                minimum=1,
            ),
        }
        for name, value in discrete_values.items():
            object.__setattr__(self, name, value)
        _require(
            np.isfinite(self.minimum_motion_m) and self.minimum_motion_m >= 0.0,
            "minimum_motion_m must be finite and nonnegative",
        )
        _require(
            np.isfinite(self.support_probability_threshold)
            and 0.0 < self.support_probability_threshold <= 1.0,
            "support_probability_threshold must lie in (0, 1]",
        )
        _require(
            np.isfinite(self.contact_exclusion_radius_m)
            and self.contact_exclusion_radius_m >= 0.0,
            "contact_exclusion_radius_m must be finite and nonnegative",
        )
        _require(
            np.isfinite(self.contact_exclusion_fraction)
            and self.contact_exclusion_fraction >= 0.0,
            "contact_exclusion_fraction must be finite and nonnegative",
        )
        weights = (
            self.motion_weight,
            self.visibility_weight,
            self.mode_information_weight,
            self.spatial_diversity_weight,
            self.contact_distance_weight,
        )
        _require(
            all(np.isfinite(value) and value >= 0.0 for value in weights),
            "query score weights must be finite and nonnegative",
        )
        _require(any(value > 0.0 for value in weights), "one score weight is required")
        _require(
            np.isfinite(self.mode_regularization) and self.mode_regularization > 0.0,
            "mode_regularization must be positive",
        )


@dataclass(frozen=True)
class PhysicsGuidedQueryPlan:
    """Immutable per-camera query events for a causal tracker provider."""

    node_ids: np.ndarray
    seed_frames: np.ndarray
    replaces_node_ids: np.ndarray
    camera_mask: np.ndarray
    seed_pixels_xy: np.ndarray
    motion_score: np.ndarray
    visibility_score: np.ndarray
    mode_information_gain: np.ndarray
    spatial_diversity_score: np.ndarray
    contact_distance_score: np.ndarray
    total_score: np.ndarray
    requested_active_queries: int
    minimum_camera_support: int
    prefix_frame_count: int

    def __post_init__(self) -> None:
        requested_active_queries = _integer_at_least(
            self.requested_active_queries,
            name="requested_active_queries",
            minimum=1,
        )
        minimum_camera_support = _integer_at_least(
            self.minimum_camera_support,
            name="minimum_camera_support",
            minimum=2,
        )
        prefix_frame_count = _integer_at_least(
            self.prefix_frame_count,
            name="prefix_frame_count",
            minimum=1,
        )
        object.__setattr__(self, "requested_active_queries", requested_active_queries)
        object.__setattr__(self, "minimum_camera_support", minimum_camera_support)
        object.__setattr__(self, "prefix_frame_count", prefix_frame_count)

        arrays = {
            "node_ids": _integer_vector(self.node_ids, name="node_ids"),
            "seed_frames": _integer_vector(self.seed_frames, name="seed_frames"),
            "replaces_node_ids": _integer_vector(
                self.replaces_node_ids, name="replaces_node_ids"
            ),
            "camera_mask": np.asarray(self.camera_mask, dtype=bool).copy(),
            "seed_pixels_xy": np.asarray(self.seed_pixels_xy, dtype=np.float64).copy(),
            "motion_score": np.asarray(self.motion_score, dtype=np.float64).copy(),
            "visibility_score": np.asarray(
                self.visibility_score, dtype=np.float64
            ).copy(),
            "mode_information_gain": np.asarray(
                self.mode_information_gain, dtype=np.float64
            ).copy(),
            "spatial_diversity_score": np.asarray(
                self.spatial_diversity_score, dtype=np.float64
            ).copy(),
            "contact_distance_score": np.asarray(
                self.contact_distance_score, dtype=np.float64
            ).copy(),
            "total_score": np.asarray(self.total_score, dtype=np.float64).copy(),
        }
        event_count = len(arrays["node_ids"])
        vector_names = (
            "node_ids",
            "seed_frames",
            "replaces_node_ids",
            "motion_score",
            "visibility_score",
            "mode_information_gain",
            "spatial_diversity_score",
            "contact_distance_score",
            "total_score",
        )
        for name in vector_names:
            _require(arrays[name].shape == (event_count,), f"{name} shape changed")
        camera_mask = arrays["camera_mask"]
        _require(camera_mask.ndim == 2, "camera_mask must have shape (Q, C)")
        _require(
            camera_mask.shape[0] == event_count,
            "camera_mask event count changed",
        )
        camera_count = camera_mask.shape[1]
        _require(
            camera_count >= self.minimum_camera_support,
            "camera count is too small",
        )
        _require(
            arrays["seed_pixels_xy"].shape == (event_count, camera_count, 2),
            "seed_pixels_xy must have shape (Q, C, 2)",
        )
        if event_count:
            node_ids = arrays["node_ids"]
            seed_frames = arrays["seed_frames"]
            replacements = arrays["replaces_node_ids"]
            _require(np.all(node_ids >= 0), "node IDs must be nonnegative")
            _require(
                len(np.unique(node_ids)) == event_count,
                "a graph identity may be seeded only once",
            )
            _require(
                np.all((seed_frames >= 0) & (seed_frames < self.prefix_frame_count)),
                "seed frame lies outside the prefix",
            )
            _require(
                np.all(np.diff(seed_frames) >= 0),
                "query events must be ordered causally",
            )
            _require(
                np.all(replacements >= -1),
                "replacement IDs must use -1 or a graph identity",
            )
            for event_index, replaced in enumerate(replacements):
                if replaced < 0:
                    continue
                _require(
                    seed_frames[event_index] > 0,
                    "frame-zero query cannot replace",
                )
                _require(
                    int(replaced) in set(node_ids[:event_index]),
                    "replacement identity was not seeded earlier",
                )
                _require(
                    int(replaced) != int(node_ids[event_index]),
                    "query event cannot replace itself",
                )
            _require(
                np.all(np.sum(camera_mask, axis=1) >= self.minimum_camera_support),
                "query event lacks independent multiview support",
            )
            supported_pixels = arrays["seed_pixels_xy"][camera_mask]
            _require(
                np.all(np.isfinite(supported_pixels)),
                "supported seed pixels must be finite",
            )
            unsupported_pixels = arrays["seed_pixels_xy"][~camera_mask]
            _require(
                np.all(np.isnan(unsupported_pixels)),
                "unsupported seed pixels must be NaN",
            )
            for name in vector_names[3:]:
                _require(np.all(np.isfinite(arrays[name])), f"{name} is not finite")
        for name, value in arrays.items():
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def initial_query_count(self) -> int:
        """Number of identities seeded at prefix frame zero."""

        return int(np.sum(self.seed_frames == 0))

    @property
    def reseed_count(self) -> int:
        """Number of query events that replace a support-lost identity."""

        return int(np.sum(self.replaces_node_ids >= 0))

    @property
    def initial_budget_met(self) -> bool:
        """Whether frame zero supplied the requested active-query budget."""

        return self.initial_query_count >= self.requested_active_queries

    def camera_queries_txy(
        self,
        camera_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return node IDs, ``[seed_frame, x, y]``, and replacement IDs."""

        camera_count = self.camera_mask.shape[1]
        index = _integer_at_least(camera_index, name="camera_index", minimum=0)
        if index >= camera_count:
            raise ValueError("camera_index lies outside the query plan")
        keep = self.camera_mask[:, index]
        node_ids = self.node_ids[keep].copy()
        queries = np.column_stack(
            (
                self.seed_frames[keep].astype(np.float64),
                self.seed_pixels_xy[keep, index],
            )
        )
        replaces = self.replaces_node_ids[keep].copy()
        for value in (node_ids, queries, replaces):
            value.setflags(write=False)
        return node_ids, queries, replaces


@dataclass(frozen=True)
class _ScoredNode:
    node_id: int
    motion_score: float
    visibility_score: float
    mode_information_gain: float
    spatial_diversity_score: float
    contact_distance_score: float
    total_score: float


def geometric_view_support(
    projected_pixels_xy: np.ndarray,
    projected_depth_m: np.ndarray,
    image_size_hw: np.ndarray,
    *,
    border_margin_px: float = 0.0,
) -> np.ndarray:
    """Return in-front, in-frame geometric support for every camera and node."""

    pixels = np.asarray(projected_pixels_xy, dtype=np.float64)
    depth = np.asarray(projected_depth_m, dtype=np.float64)
    _require(
        pixels.ndim == 4 and pixels.shape[3] == 2,
        "projected_pixels_xy must have shape (C, T, N, 2)",
    )
    _require(depth.shape == pixels.shape[:3], "projected_depth_m shape changed")
    _require(
        np.isfinite(border_margin_px) and border_margin_px >= 0.0,
        "border_margin_px must be finite and nonnegative",
    )
    image_size = np.asarray(image_size_hw, dtype=np.float64)
    if image_size.shape == (2,):
        image_size = np.repeat(image_size[None], pixels.shape[0], axis=0)
    _require(
        image_size.shape == (pixels.shape[0], 2),
        "image_size_hw must have shape (2,) or (C, 2)",
    )
    _require(
        np.all(np.isfinite(image_size)) and np.all(image_size > 0.0),
        "image sizes must be finite and positive",
    )
    _require(
        np.all(2.0 * border_margin_px < image_size),
        "border margin removes the complete image",
    )
    height = image_size[:, 0, None, None]
    width = image_size[:, 1, None, None]
    x = pixels[..., 0]
    y = pixels[..., 1]
    result = (
        np.all(np.isfinite(pixels), axis=3)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (x >= border_margin_px)
        & (x <= width - 1.0 - border_margin_px)
        & (y >= border_margin_px)
        & (y <= height - 1.0 - border_margin_px)
    )
    result.setflags(write=False)
    return result


def _remaining_motion_m(rollout: np.ndarray, frame: int) -> np.ndarray:
    start = rollout[frame]
    delta = rollout[frame:] - start[None]
    finite = np.all(np.isfinite(delta), axis=2)
    distance = np.linalg.norm(np.where(finite[..., None], delta, 0.0), axis=2)
    distance = np.where(finite, distance, -np.inf)
    motion = np.max(distance, axis=0)
    motion[~np.any(finite, axis=0)] = np.nan
    return motion


def _object_scale_m(positions: np.ndarray, candidate_ids: np.ndarray) -> float:
    selected = positions[candidate_ids]
    finite = selected[np.all(np.isfinite(selected), axis=1)]
    if len(finite) < 2:
        return 1.0
    diagonal = float(np.linalg.norm(np.max(finite, axis=0) - np.min(finite, axis=0)))
    return max(diagonal, 1e-6)


def _normalized_mode_basis(
    mode_basis: np.ndarray | None,
    node_count: int,
) -> np.ndarray:
    if mode_basis is None:
        return np.empty((node_count, 0), dtype=np.float64)
    basis = np.asarray(mode_basis, dtype=np.float64)
    _require(
        basis.ndim == 2 and basis.shape[0] == node_count,
        "mode_basis must have shape (N, R)",
    )
    _require(basis.shape[1] >= 1, "mode_basis must retain at least one mode")
    _require(np.all(np.isfinite(basis)), "mode_basis contains non-finite values")
    norm = np.linalg.norm(basis, axis=0)
    _require(np.all(norm > 0.0), "mode_basis contains an empty mode")
    return basis / norm[None]


def _select_at_frame(
    *,
    frame: int,
    count: int,
    rollout: np.ndarray,
    pixels: np.ndarray,
    predicted_support: np.ndarray,
    mode_basis: np.ndarray,
    contact_positions: np.ndarray | None,
    candidate_ids: np.ndarray,
    unavailable_ids: set[int],
    active_ids: list[int],
    config: PhysicsGuidedQueryConfig,
) -> list[_ScoredNode]:
    if count <= 0:
        return []
    camera_count = pixels.shape[0]
    threshold = config.support_probability_threshold
    view_mask = (predicted_support[:, frame] >= threshold) & np.all(
        np.isfinite(pixels[:, frame]), axis=2
    )
    view_count = np.sum(view_mask, axis=0)
    motion_m = _remaining_motion_m(rollout, frame)
    object_scale = _object_scale_m(rollout[frame], candidate_ids)
    if contact_positions is None:
        contact_distance_m = np.full(rollout.shape[1], np.nan)
        contact_exclusion_m = 0.0
    else:
        contact_distance_m = np.linalg.norm(
            rollout[frame] - contact_positions[frame], axis=1
        )
        contact_exclusion_m = max(
            config.contact_exclusion_radius_m,
            config.contact_exclusion_fraction * object_scale,
        )
    finite_position = np.all(np.isfinite(rollout[frame]), axis=1)
    eligible_mask = (
        finite_position
        & np.isfinite(motion_m)
        & (motion_m >= config.minimum_motion_m)
        & (view_count >= config.minimum_camera_support)
    )
    if contact_positions is not None:
        eligible_mask &= contact_distance_m >= contact_exclusion_m
    eligible = np.asarray(
        [
            int(node_id)
            for node_id in candidate_ids
            if eligible_mask[node_id] and int(node_id) not in unavailable_ids
        ],
        dtype=np.int64,
    )
    if not len(eligible):
        return []

    maximum_motion = max(float(np.max(motion_m[eligible])), 1e-12)
    motion_score = motion_m / maximum_motion
    support_count = np.sum(
        (predicted_support[:, frame:] >= threshold)
        & np.all(np.isfinite(pixels[:, frame:]), axis=3),
        axis=0,
    )
    visibility_score = np.mean(support_count / camera_count, axis=0)
    if contact_positions is None:
        contact_score = np.zeros(rollout.shape[1], dtype=np.float64)
    else:
        contact_score = np.clip(contact_distance_m / object_scale, 0.0, 1.0)

    mode_count = mode_basis.shape[1]
    information = config.mode_regularization * np.eye(mode_count)
    if mode_count:
        for node_id in active_ids:
            row = mode_basis[node_id]
            information += np.outer(row, row)
    diversity_ids = list(active_ids)
    selected: list[_ScoredNode] = []
    selected_ids: set[int] = set()
    while len(selected) < min(count, len(eligible)):
        remaining = np.asarray(
            [node_id for node_id in eligible if int(node_id) not in selected_ids],
            dtype=np.int64,
        )
        if not len(remaining):
            break
        if diversity_ids:
            reference = rollout[frame, np.asarray(diversity_ids, dtype=np.int64)]
            distance = np.linalg.norm(
                rollout[frame, remaining, None] - reference[None], axis=2
            )
            spatial_score = np.clip(np.min(distance, axis=1) / object_scale, 0.0, 1.0)
        else:
            spatial_score = np.ones(len(remaining), dtype=np.float64)
        if mode_count:
            raw_mode_gain = np.asarray(
                [
                    np.log1p(
                        float(
                            mode_basis[node_id]
                            @ np.linalg.solve(information, mode_basis[node_id])
                        )
                    )
                    for node_id in remaining
                ],
                dtype=np.float64,
            )
            maximum_gain = float(np.max(raw_mode_gain))
            if maximum_gain > 0.0:
                normalized_mode_gain = raw_mode_gain / maximum_gain
            else:
                normalized_mode_gain = np.zeros_like(raw_mode_gain)
        else:
            raw_mode_gain = np.zeros(len(remaining), dtype=np.float64)
            normalized_mode_gain = raw_mode_gain
        total = (
            config.motion_weight * motion_score[remaining]
            + config.visibility_weight * visibility_score[remaining]
            + config.mode_information_weight * normalized_mode_gain
            + config.spatial_diversity_weight * spatial_score
            + config.contact_distance_weight * contact_score[remaining]
        )
        maximum_total = float(np.max(total))
        tied = remaining[np.isclose(total, maximum_total, rtol=0.0, atol=1e-12)]
        chosen = int(np.min(tied))
        chosen_offset = int(np.flatnonzero(remaining == chosen)[0])
        selected.append(
            _ScoredNode(
                node_id=chosen,
                motion_score=float(motion_score[chosen]),
                visibility_score=float(visibility_score[chosen]),
                mode_information_gain=float(raw_mode_gain[chosen_offset]),
                spatial_diversity_score=float(spatial_score[chosen_offset]),
                contact_distance_score=float(contact_score[chosen]),
                total_score=float(total[chosen_offset]),
            )
        )
        selected_ids.add(chosen)
        diversity_ids.append(chosen)
        if mode_count:
            row = mode_basis[chosen]
            information += np.outer(row, row)
    return selected


def plan_physics_guided_queries(
    physical_rollout_m: np.ndarray,
    projected_pixels_xy: np.ndarray,
    predicted_support_probability: np.ndarray,
    *,
    mode_basis: np.ndarray | None = None,
    tracker_support_probability: np.ndarray | None = None,
    contact_position_m: np.ndarray | None = None,
    candidate_ids: np.ndarray | None = None,
    config: PhysicsGuidedQueryConfig | None = None,
) -> PhysicsGuidedQueryPlan:
    """Plan initial graph queries and causal support-triggered replacements.

    ``physical_rollout_m`` and ``predicted_support_probability`` may span the
    complete allowed prefix because both are action-conditioned predictions.
    In contrast, tracker support is consumed sequentially: the decision at frame
    ``t`` reads only support at frames up to and including ``t``.
    """

    cfg = config or PhysicsGuidedQueryConfig()
    rollout = np.asarray(physical_rollout_m, dtype=np.float64)
    pixels = np.asarray(projected_pixels_xy, dtype=np.float64)
    predicted_support = np.asarray(predicted_support_probability, dtype=np.float64)
    _require(
        rollout.ndim == 3 and rollout.shape[2] == 3,
        "physical_rollout_m must have shape (T, N, 3)",
    )
    frame_count, node_count, _ = rollout.shape
    _require(frame_count >= 1 and node_count >= 1, "physical rollout is empty")
    _require(
        pixels.ndim == 4
        and pixels.shape[1:3] == (frame_count, node_count)
        and pixels.shape[3] == 2,
        "projected_pixels_xy must have shape (C, T, N, 2)",
    )
    camera_count = pixels.shape[0]
    _require(
        camera_count >= cfg.minimum_camera_support,
        "camera count is below minimum_camera_support",
    )
    _require(
        predicted_support.shape == (camera_count, frame_count, node_count),
        "predicted_support_probability shape changed",
    )
    _require(
        np.all(np.isfinite(predicted_support))
        and np.all((predicted_support >= 0.0) & (predicted_support <= 1.0)),
        "predicted support probabilities must lie in [0, 1]",
    )
    if tracker_support_probability is None:
        tracker_support = predicted_support
    else:
        tracker_support = np.asarray(tracker_support_probability, dtype=np.float64)
        _require(
            tracker_support.shape == predicted_support.shape,
            "tracker_support_probability shape changed",
        )
        finite_tracker = tracker_support[np.isfinite(tracker_support)]
        _require(
            np.all((finite_tracker >= 0.0) & (finite_tracker <= 1.0)),
            "finite tracker support probabilities must lie in [0, 1]",
        )
        tracker_support = np.where(np.isfinite(tracker_support), tracker_support, 0.0)
    normalized_basis = _normalized_mode_basis(mode_basis, node_count)
    candidates: np.ndarray
    if candidate_ids is None:
        candidates = np.arange(node_count, dtype=np.int64)
    else:
        candidates = _integer_vector(candidate_ids, name="candidate_ids")
        _require(
            candidates.ndim == 1 and len(candidates) >= 1,
            "candidate_ids is empty",
        )
        _require(
            np.all((candidates >= 0) & (candidates < node_count)),
            "candidate ID exceeds the physical rollout",
        )
        _require(
            len(np.unique(candidates)) == len(candidates),
            "candidate_ids must be unique",
        )
        candidates = np.sort(candidates, kind="mergesort")
    if contact_position_m is None:
        contact_positions = None
    else:
        contact = np.asarray(contact_position_m, dtype=np.float64)
        if contact.shape == (3,):
            contact = np.repeat(contact[None], frame_count, axis=0)
        _require(
            contact.shape == (frame_count, 3),
            "contact_position_m must have shape (3,) or (T, 3)",
        )
        _require(np.all(np.isfinite(contact)), "contact position is not finite")
        contact_positions = contact

    event_nodes: list[int] = []
    event_frames: list[int] = []
    event_replacements: list[int] = []
    event_masks: list[np.ndarray] = []
    event_pixels: list[np.ndarray] = []
    event_scores: list[_ScoredNode] = []
    ever_seeded: set[int] = set()
    active: dict[int, tuple[int, int]] = {}
    pending_replacements: list[int] = []

    def append_event(scored: _ScoredNode, frame: int, replaces: int) -> None:
        node_id = scored.node_id
        mask = (
            predicted_support[:, frame, node_id] >= cfg.support_probability_threshold
        ) & np.all(np.isfinite(pixels[:, frame, node_id]), axis=1)
        _require(
            int(np.sum(mask)) >= cfg.minimum_camera_support,
            "selected query lost multiview support",
        )
        seed_pixels = np.where(mask[:, None], pixels[:, frame, node_id], np.nan)
        event_nodes.append(node_id)
        event_frames.append(frame)
        event_replacements.append(replaces)
        event_masks.append(mask)
        event_pixels.append(seed_pixels)
        event_scores.append(scored)
        ever_seeded.add(node_id)
        active[node_id] = (frame, 0)

    initial = _select_at_frame(
        frame=0,
        count=cfg.query_count,
        rollout=rollout,
        pixels=pixels,
        predicted_support=predicted_support,
        mode_basis=normalized_basis,
        contact_positions=contact_positions,
        candidate_ids=candidates,
        unavailable_ids=ever_seeded,
        active_ids=[],
        config=cfg,
    )
    for scored in initial:
        append_event(scored, 0, -1)

    maximum_event_count = len(initial) + cfg.maximum_reseeds
    for frame in range(1, frame_count):
        lost: list[int] = []
        for node_id in sorted(active):
            seed_frame, unsupported_streak = active[node_id]
            support_count = int(
                np.sum(
                    (
                        tracker_support[:, frame, node_id]
                        >= cfg.support_probability_threshold
                    )
                    & np.all(
                        np.isfinite(pixels[:, frame, node_id]),
                        axis=1,
                    )
                )
            )
            if support_count >= cfg.minimum_camera_support:
                unsupported_streak = 0
            else:
                unsupported_streak += 1
            active[node_id] = (seed_frame, unsupported_streak)
            if (
                unsupported_streak >= cfg.reseed_patience_frames
                and frame - seed_frame >= cfg.minimum_reseed_interval_frames
            ):
                lost.append(node_id)
        for node_id in lost:
            del active[node_id]
            pending_replacements.append(node_id)
        remaining_events = maximum_event_count - len(event_nodes)
        vacancy_count = min(cfg.query_count - len(active), remaining_events)
        if vacancy_count <= 0:
            continue
        replacements = _select_at_frame(
            frame=frame,
            count=vacancy_count,
            rollout=rollout,
            pixels=pixels,
            predicted_support=predicted_support,
            mode_basis=normalized_basis,
            contact_positions=contact_positions,
            candidate_ids=candidates,
            unavailable_ids=ever_seeded,
            active_ids=sorted(active),
            config=cfg,
        )
        for scored in replacements:
            replaced = pending_replacements.pop(0) if pending_replacements else -1
            append_event(scored, frame, replaced)

    event_count = len(event_nodes)
    if event_count:
        camera_mask = np.stack(event_masks, axis=0)
        seed_pixels = np.stack(event_pixels, axis=0)
    else:
        camera_mask = np.empty((0, camera_count), dtype=bool)
        seed_pixels = np.empty((0, camera_count, 2), dtype=np.float64)
    return PhysicsGuidedQueryPlan(
        node_ids=np.asarray(event_nodes, dtype=np.int64),
        seed_frames=np.asarray(event_frames, dtype=np.int64),
        replaces_node_ids=np.asarray(event_replacements, dtype=np.int64),
        camera_mask=camera_mask,
        seed_pixels_xy=seed_pixels,
        motion_score=np.asarray(
            [score.motion_score for score in event_scores], dtype=np.float64
        ),
        visibility_score=np.asarray(
            [score.visibility_score for score in event_scores], dtype=np.float64
        ),
        mode_information_gain=np.asarray(
            [score.mode_information_gain for score in event_scores], dtype=np.float64
        ),
        spatial_diversity_score=np.asarray(
            [score.spatial_diversity_score for score in event_scores], dtype=np.float64
        ),
        contact_distance_score=np.asarray(
            [score.contact_distance_score for score in event_scores], dtype=np.float64
        ),
        total_score=np.asarray(
            [score.total_score for score in event_scores], dtype=np.float64
        ),
        requested_active_queries=cfg.query_count,
        minimum_camera_support=cfg.minimum_camera_support,
        prefix_frame_count=frame_count,
    )
