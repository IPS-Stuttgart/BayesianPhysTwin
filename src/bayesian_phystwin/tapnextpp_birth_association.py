"""Causal pixel proposals for dynamically born TAPNext++ queries.

Physical geometry is used only to define an association distribution over
nearby depth-supported object pixels. It does not emit perception reliability;
that quantity is computed later from tracker and multiview observation cues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deform360_dynamic_query import project_visibility

LEGACY_EXACT_PIXEL_ASSOCIATION = "legacy-exact-pixel-entropy-v1"
SET_VALUED_COVARIANCE_ASSOCIATION = "set-valued-covariance-v1"
_ASSOCIATION_MODES = frozenset(
    {
        LEGACY_EXACT_PIXEL_ASSOCIATION,
        SET_VALUED_COVARIANCE_ASSOCIATION,
    }
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True)
class BirthAssociationConfig:
    """Frozen local association choices from the v1 protocol."""

    search_radius_px: int = 12
    depth_scale_m: float = 0.03
    minimum_candidate_count: int = 1
    association_mode: str = LEGACY_EXACT_PIXEL_ASSOCIATION

    def __post_init__(self) -> None:
        _require(self.search_radius_px >= 1, "search radius must be positive")
        _require(self.depth_scale_m > 0.0, "depth scale must be positive")
        _require(
            self.minimum_candidate_count >= 1,
            "minimum candidate count must be positive",
        )
        _require(
            self.association_mode in _ASSOCIATION_MODES,
            f"unsupported association mode {self.association_mode!r}",
        )


def _camera_depth_m(
    point_world_m: np.ndarray,
    camera_to_world: np.ndarray,
) -> float:
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = world_to_camera @ np.append(point_world_m, 1.0)
    return float(camera[2])


def _candidate_distribution(
    center_xy: np.ndarray,
    expected_depth_m: float,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    config: BirthAssociationConfig,
) -> (
    tuple[
        np.ndarray,
        float,
        float,
        np.ndarray,
        int,
        np.ndarray,
        float,
        np.ndarray,
    ]
    | None
):
    height, width = depth_m.shape
    center_x, center_y = np.asarray(center_xy, dtype=np.float64)
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        return None
    x0 = max(0, int(np.floor(center_x)) - config.search_radius_px)
    x1 = min(width, int(np.floor(center_x)) + config.search_radius_px + 2)
    y0 = max(0, int(np.floor(center_y)) - config.search_radius_px)
    y1 = min(height, int(np.floor(center_y)) + config.search_radius_px + 2)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pixels = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    radius2 = np.sum((pixels - center_xy[None]) ** 2, axis=1)
    patch_depth = np.asarray(depth_m[y0:y1, x0:x1], dtype=np.float64).ravel()
    patch_mask = np.asarray(object_mask[y0:y1, x0:x1], dtype=bool).ravel()
    valid = (
        patch_mask
        & np.isfinite(patch_depth)
        & (patch_depth > 0.0)
        & (radius2 <= config.search_radius_px**2)
    )
    if int(np.sum(valid)) < config.minimum_candidate_count:
        return None
    candidates = pixels[valid]
    candidate_depth = patch_depth[valid]
    pixel_term = radius2[valid] / (config.search_radius_px**2)
    depth_term = ((candidate_depth - expected_depth_m) / config.depth_scale_m) ** 2
    cost = pixel_term + depth_term
    minimum = float(np.min(cost))
    weights = np.exp(-0.5 * (cost - minimum))
    weights /= np.sum(weights)
    best = int(np.argmax(weights))
    query_xy = candidates[best]
    candidate_xyd = np.column_stack((candidates, candidate_depth))
    mixture_mean = np.sum(weights[:, None] * candidate_xyd, axis=0)
    centered = candidate_xyd - mixture_mean
    mixture_covariance = np.einsum(
        "n,ni,nj->ij",
        weights,
        centered,
        centered,
    )
    covariance = mixture_covariance[:2, :2]
    if len(weights) == 1:
        normalized_entropy = 0.0
    else:
        entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-300))))
        normalized_entropy = entropy / np.log(len(weights))
    geometry_evidence = float(np.exp(-0.5 * minimum))
    if config.association_mode == LEGACY_EXACT_PIXEL_ASSOCIATION:
        association_probability = float(
            np.clip(
                geometry_evidence * (1.0 - normalized_entropy),
                0.0,
                1.0,
            )
        )
    else:
        # The latent event is that the material projection lies in this
        # depth-supported patch. Pixel ambiguity is represented by the
        # assignment covariance below and must not also collapse that event.
        association_probability = float(np.clip(geometry_evidence, 0.0, 1.0))
    return (
        query_xy,
        association_probability,
        normalized_entropy,
        covariance,
        len(weights),
        mixture_mean[:2],
        float(mixture_mean[2]),
        mixture_covariance,
    )


def propose_birth_query_pixels(
    predicted_points_world_m: np.ndarray,
    projection_matrices: np.ndarray,
    camera_to_world: np.ndarray,
    depths_m: np.ndarray,
    object_masks: np.ndarray,
    *,
    config: BirthAssociationConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return per-camera causal query proposals and association uncertainty."""

    cfg = config or BirthAssociationConfig()
    points = np.asarray(predicted_points_world_m, dtype=np.float64)
    projections = np.asarray(projection_matrices, dtype=np.float64)
    poses = np.asarray(camera_to_world, dtype=np.float64)
    depth = np.asarray(depths_m, dtype=np.float64)
    masks = np.asarray(object_masks, dtype=bool)
    _require(
        points.ndim == 2 and points.shape[1] == 3,
        "predicted points must have shape (N, 3)",
    )
    _require(
        projections.ndim == 3 and projections.shape[1:] == (3, 4),
        "projection matrices must have shape (C, 3, 4)",
    )
    camera_count = len(projections)
    _require(
        poses.shape == (camera_count, 4, 4),
        "camera poses must have shape (C, 4, 4)",
    )
    _require(
        depth.ndim == 3 and depth.shape[0] == camera_count,
        "depth must have shape (C, H, W)",
    )
    _require(masks.shape == depth.shape, "object masks differ from depth")
    _require(
        np.all(np.isfinite(points))
        and np.all(np.isfinite(projections))
        and np.all(np.isfinite(poses)),
        "association geometry is not finite",
    )
    shapes = np.asarray(
        [[frame.shape[0], frame.shape[1]] for frame in depth],
        dtype=np.int64,
    )
    projected, _, visible = project_visibility(points, projections, shapes)

    point_count = len(points)
    query_xy: np.ndarray = np.full(
        (camera_count, point_count, 2),
        np.nan,
        dtype=np.float64,
    )
    valid: np.ndarray = np.zeros((camera_count, point_count), dtype=bool)
    probability: np.ndarray = np.zeros(
        (camera_count, point_count),
        dtype=np.float64,
    )
    entropy: np.ndarray = np.ones(
        (camera_count, point_count),
        dtype=np.float64,
    )
    covariance: np.ndarray = np.full(
        (camera_count, point_count, 2, 2),
        np.nan,
        dtype=np.float64,
    )
    candidate_count: np.ndarray = np.zeros(
        (camera_count, point_count),
        dtype=np.int64,
    )
    candidate_mean_xy: np.ndarray = np.full(
        (camera_count, point_count, 2),
        np.nan,
        dtype=np.float64,
    )
    candidate_mean_depth_m: np.ndarray = np.full(
        (camera_count, point_count),
        np.nan,
        dtype=np.float64,
    )
    candidate_xyd_covariance: np.ndarray = np.full(
        (camera_count, point_count, 3, 3),
        np.nan,
        dtype=np.float64,
    )
    for camera in range(camera_count):
        for entity in range(point_count):
            if not visible[camera, entity]:
                continue
            expected_depth = _camera_depth_m(points[entity], poses[camera])
            if not np.isfinite(expected_depth) or expected_depth <= 0.0:
                continue
            proposal = _candidate_distribution(
                projected[camera, entity],
                expected_depth,
                depth[camera],
                masks[camera],
                cfg,
            )
            if proposal is None:
                continue
            (
                query_xy[camera, entity],
                probability[camera, entity],
                entropy[camera, entity],
                covariance[camera, entity],
                candidate_count[camera, entity],
                candidate_mean_xy[camera, entity],
                candidate_mean_depth_m[camera, entity],
                candidate_xyd_covariance[camera, entity],
            ) = proposal
            valid[camera, entity] = True

    return {
        "query_points_xy": query_xy,
        "valid": valid,
        "association_probability": probability,
        "association_entropy": entropy,
        "candidate_pixel_covariance_px2": covariance,
        "candidate_count": candidate_count,
        "candidate_mean_xy": candidate_mean_xy,
        "candidate_mean_depth_m": candidate_mean_depth_m,
        "candidate_xyd_covariance": candidate_xyd_covariance,
    }


__all__ = [
    "BirthAssociationConfig",
    "LEGACY_EXACT_PIXEL_ASSOCIATION",
    "SET_VALUED_COVARIANCE_ASSOCIATION",
    "propose_birth_query_pixels",
]
