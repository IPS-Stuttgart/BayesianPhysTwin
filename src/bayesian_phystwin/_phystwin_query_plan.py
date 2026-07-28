"""Immutable content-addressed physics-guided query plan contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ._phystwin_query_core import (
    PHYSICS_GUIDED_QUERY_PLAN_SCHEMA,
    PHYSICS_GUIDED_QUERY_PLAN_VERSION,
    PhysicsGuidedQueryConfigV2,
    _validate_sha256,
    array_sha256,
    readonly,
    require,
)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class PhysicsGuidedQueryPlanV1:
    """Immutable query events and provenance for one causal prefix."""

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
    config: PhysicsGuidedQueryConfigV2
    source_revision: str
    support_model_id: str
    physical_rollout_sha256: str
    projected_pixels_sha256: str
    predicted_support_sha256: str
    mode_basis_sha256: str
    nuisance_basis_sha256: str
    observation_precision_sha256: str
    candidate_ids_sha256: str
    contact_position_sha256: str
    tracker_support_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.config, PhysicsGuidedQueryConfigV2):
            raise TypeError("config must be a PhysicsGuidedQueryConfigV2")
        require(bool(self.source_revision), "source_revision must be nonempty")
        require(bool(self.support_model_id), "support_model_id must be nonempty")
        digest_names = (
            "physical_rollout_sha256",
            "projected_pixels_sha256",
            "predicted_support_sha256",
            "mode_basis_sha256",
            "nuisance_basis_sha256",
            "observation_precision_sha256",
            "candidate_ids_sha256",
            "contact_position_sha256",
            "tracker_support_sha256",
        )
        for name in digest_names:
            _validate_sha256(str(getattr(self, name)), name=name)
        arrays = {
            "node_ids": readonly(self.node_ids, dtype=np.int64),
            "seed_frames": readonly(self.seed_frames, dtype=np.int64),
            "replaces_node_ids": readonly(self.replaces_node_ids, dtype=np.int64),
            "camera_mask": readonly(self.camera_mask, dtype=bool),
            "seed_pixels_xy": readonly(self.seed_pixels_xy, dtype=np.float64),
            "motion_score": readonly(self.motion_score, dtype=np.float64),
            "visibility_score": readonly(self.visibility_score, dtype=np.float64),
            "mode_information_gain": readonly(
                self.mode_information_gain,
                dtype=np.float64,
            ),
            "spatial_diversity_score": readonly(
                self.spatial_diversity_score,
                dtype=np.float64,
            ),
            "contact_distance_score": readonly(
                self.contact_distance_score,
                dtype=np.float64,
            ),
            "total_score": readonly(self.total_score, dtype=np.float64),
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
            require(arrays[name].shape == (event_count,), f"{name} shape changed")
        camera_mask = arrays["camera_mask"]
        require(camera_mask.ndim == 2, "camera_mask must have shape (Q, C)")
        require(camera_mask.shape[0] == event_count, "camera event count changed")
        camera_count = camera_mask.shape[1]
        require(
            camera_count >= self.minimum_camera_support,
            "camera count is too small",
        )
        require(
            arrays["seed_pixels_xy"].shape == (event_count, camera_count, 2),
            "seed_pixels_xy must have shape (Q, C, 2)",
        )
        require(self.prefix_frame_count >= 1, "prefix frame count is invalid")
        require(
            self.requested_active_queries == self.config.query_count,
            "requested query count differs from the bound config",
        )
        require(
            self.minimum_camera_support == self.config.minimum_camera_support,
            "minimum camera support differs from the bound config",
        )
        if event_count:
            node_ids = arrays["node_ids"]
            seed_frames = arrays["seed_frames"]
            replacements = arrays["replaces_node_ids"]
            require(np.all(node_ids >= 0), "node IDs must be nonnegative")
            require(
                len(np.unique(node_ids)) == event_count,
                "a graph identity may be seeded only once",
            )
            require(
                np.all((seed_frames >= 0) & (seed_frames < self.prefix_frame_count)),
                "seed frame lies outside the prefix",
            )
            require(
                np.all(np.diff(seed_frames) >= 0),
                "query events must be ordered causally",
            )
            require(np.all(replacements >= -1), "replacement IDs are invalid")
            for event_index, replaced in enumerate(replacements):
                if replaced < 0:
                    continue
                require(seed_frames[event_index] > 0, "frame-zero query cannot replace")
                require(
                    int(replaced) in set(node_ids[:event_index]),
                    "replacement identity was not seeded earlier",
                )
                require(
                    int(replaced) != int(node_ids[event_index]),
                    "query event cannot replace itself",
                )
            require(
                np.all(np.sum(camera_mask, axis=1) >= self.minimum_camera_support),
                "query event lacks independent multiview support",
            )
            require(
                np.all(np.isfinite(arrays["seed_pixels_xy"][camera_mask])),
                "supported seed pixels must be finite",
            )
            require(
                np.all(np.isnan(arrays["seed_pixels_xy"][~camera_mask])),
                "unsupported seed pixels must be NaN",
            )
            for name in vector_names[3:]:
                require(np.all(np.isfinite(arrays[name])), f"{name} is not finite")
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    def descriptor(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible plan descriptor."""

        return {
            "schema_name": PHYSICS_GUIDED_QUERY_PLAN_SCHEMA,
            "schema_version": PHYSICS_GUIDED_QUERY_PLAN_VERSION,
            "requested_active_queries": self.requested_active_queries,
            "minimum_camera_support": self.minimum_camera_support,
            "prefix_frame_count": self.prefix_frame_count,
            "config": asdict(self.config),
            "source_revision": self.source_revision,
            "support_model_id": self.support_model_id,
            "input_sha256": {
                "candidate_ids": self.candidate_ids_sha256,
                "contact_position_m": self.contact_position_sha256,
                "mode_basis": self.mode_basis_sha256,
                "nuisance_basis": self.nuisance_basis_sha256,
                "observation_precision": self.observation_precision_sha256,
                "physical_rollout_m": self.physical_rollout_sha256,
                "predicted_support_probability": self.predicted_support_sha256,
                "projected_pixels_xy": self.projected_pixels_sha256,
                "tracker_support_probability": self.tracker_support_sha256,
            },
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the exact numeric payload included in the content address."""

        return {
            "node_ids": self.node_ids,
            "seed_frames": self.seed_frames,
            "replaces_node_ids": self.replaces_node_ids,
            "camera_mask": self.camera_mask,
            "seed_pixels_xy": self.seed_pixels_xy,
            "motion_score": self.motion_score,
            "visibility_score": self.visibility_score,
            "mode_information_gain": self.mode_information_gain,
            "spatial_diversity_score": self.spatial_diversity_score,
            "contact_distance_score": self.contact_distance_score,
            "total_score": self.total_score,
        }

    @property
    def artifact_id(self) -> str:
        """Content address of the descriptor and every event array."""

        digest = hashlib.sha256()
        digest.update(_canonical_json(self.descriptor()))
        for name, values in sorted(self.arrays().items()):
            digest.update(name.encode("utf-8"))
            digest.update(array_sha256(values).encode("ascii"))
        return digest.hexdigest()

    @property
    def initial_query_count(self) -> int:
        return int(np.sum(self.seed_frames == 0))

    @property
    def reseed_count(self) -> int:
        return int(np.sum(self.replaces_node_ids >= 0))

    @property
    def initial_budget_met(self) -> bool:
        return self.initial_query_count >= self.requested_active_queries

    def camera_queries_txy(
        self,
        camera_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return node IDs, ``[seed_frame, x, y]``, and replacement IDs."""

        camera_count = self.camera_mask.shape[1]
        if not 0 <= camera_index < camera_count:
            raise ValueError("camera_index lies outside the query plan")
        keep = self.camera_mask[:, camera_index]
        node_ids = self.node_ids[keep].copy()
        queries = np.column_stack(
            (
                self.seed_frames[keep].astype(np.float64),
                self.seed_pixels_xy[keep, camera_index],
            )
        )
        replacements = self.replaces_node_ids[keep].copy()
        for value in (node_ids, queries, replacements):
            value.setflags(write=False)
        return node_ids, queries, replacements


__all__ = ["PhysicsGuidedQueryPlanV1"]
