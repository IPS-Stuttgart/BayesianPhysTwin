"""Content-addressed artifacts for physics-guided active query plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


PHYSICS_GUIDED_QUERY_PLAN_SCHEMA = "bayesian_phystwin.query_plan"
PHYSICS_GUIDED_QUERY_PLAN_VERSION = 1


def require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


def array_sha256(values: np.ndarray) -> str:
    """Hash an array including its dtype and shape."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def readonly(
    values: np.ndarray,
    *,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    """Return a defensive C-order read-only array copy."""

    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhysicsGuidedQueryConfigV2:
    """Nuisance-aware selection and causal reseeding settings."""

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
    nuisance_regularization: float = 1e-3
    reseed_patience_frames: int = 2
    minimum_reseed_interval_frames: int = 2

    def __post_init__(self) -> None:
        require(self.query_count >= 1, "query_count must be positive")
        require(self.maximum_reseeds >= 0, "maximum_reseeds must be nonnegative")
        require(
            self.minimum_camera_support >= 2,
            "minimum_camera_support must retain independent multiview support",
        )
        require(
            np.isfinite(self.minimum_motion_m) and self.minimum_motion_m >= 0.0,
            "minimum_motion_m must be finite and nonnegative",
        )
        require(
            np.isfinite(self.support_probability_threshold)
            and 0.0 < self.support_probability_threshold <= 1.0,
            "support_probability_threshold must lie in (0, 1]",
        )
        nonnegative = (
            self.contact_exclusion_radius_m,
            self.contact_exclusion_fraction,
            self.motion_weight,
            self.visibility_weight,
            self.mode_information_weight,
            self.spatial_diversity_weight,
            self.contact_distance_weight,
        )
        require(
            all(np.isfinite(value) and value >= 0.0 for value in nonnegative),
            "query scales and weights must be finite and nonnegative",
        )
        weights = nonnegative[2:]
        require(any(value > 0.0 for value in weights), "one score weight is required")
        require(
            np.isfinite(self.mode_regularization) and self.mode_regularization > 0.0,
            "mode_regularization must be positive",
        )
        require(
            np.isfinite(self.nuisance_regularization)
            and self.nuisance_regularization > 0.0,
            "nuisance_regularization must be positive",
        )
        require(
            self.reseed_patience_frames >= 1,
            "reseed_patience_frames must be positive",
        )
        require(
            self.minimum_reseed_interval_frames >= 1,
            "minimum_reseed_interval_frames must be positive",
        )


@dataclass(frozen=True)
class PhysicsGuidedQueryStepV1:
    """One causal batch of tracker queries emitted at one prefix frame."""

    frame: int
    node_ids: np.ndarray
    replaces_node_ids: np.ndarray
    camera_mask: np.ndarray
    seed_pixels_xy: np.ndarray

    def __post_init__(self) -> None:
        require(self.frame >= 0, "query step frame must be nonnegative")
        node_ids = readonly(self.node_ids, dtype=np.int64)
        replacements = readonly(self.replaces_node_ids, dtype=np.int64)
        camera_mask = readonly(self.camera_mask, dtype=bool)
        seed_pixels = readonly(self.seed_pixels_xy, dtype=np.float64)
        event_count = len(node_ids)
        require(replacements.shape == (event_count,), "replacement shape changed")
        require(
            camera_mask.ndim == 2 and camera_mask.shape[0] == event_count,
            "camera_mask must have shape (Q, C)",
        )
        require(
            seed_pixels.shape == (event_count, camera_mask.shape[1], 2),
            "seed_pixels_xy must have shape (Q, C, 2)",
        )
        if event_count:
            require(np.all(node_ids >= 0), "node IDs must be nonnegative")
            require(
                len(np.unique(node_ids)) == event_count,
                "query step node IDs must be unique",
            )
            require(np.all(replacements >= -1), "replacement IDs are invalid")
            require(
                np.all(np.isfinite(seed_pixels[camera_mask])),
                "supported seed pixels must be finite",
            )
            require(
                np.all(np.isnan(seed_pixels[~camera_mask])),
                "unsupported seed pixels must be NaN",
            )
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "replaces_node_ids", replacements)
        object.__setattr__(self, "camera_mask", camera_mask)
        object.__setattr__(self, "seed_pixels_xy", seed_pixels)


__all__ = [
    "PHYSICS_GUIDED_QUERY_PLAN_SCHEMA",
    "PHYSICS_GUIDED_QUERY_PLAN_VERSION",
    "PhysicsGuidedQueryConfigV2",
    "PhysicsGuidedQueryStepV1",
    "array_sha256",
    "readonly",
    "require",
]
