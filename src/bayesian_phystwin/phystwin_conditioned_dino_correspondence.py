"""PhysTwin-conditioned, residual-independent material correspondence.

The physical rollout is allowed to restrict the association search region, but
it must not become perception reliability.  Appearance uniqueness, local patch
agreement, mask/depth support, and cross-view consistency determine reliability
before the state innovation is evaluated by the robust Bayesian likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


def _readonly(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ConditionedDinoConfig:
    """Frozen numerical choices for one descriptor correspondence search."""

    search_radius_px: float = 56.0
    descriptor_temperature: float = 0.04
    minimum_cosine_similarity: float = 0.55
    maximum_normalized_entropy: float = 0.98
    minimum_candidate_count: int = 3
    pixel_standard_deviation_floor: float = 1.5
    patch_radius_px: int = 5
    patch_search_radius_px: int = 10
    patch_temperature: float = 0.08
    minimum_patch_correlation: float = 0.20
    minimum_patch_standard_deviation: float = 2.0 / 255.0
    minimum_views: int = 2
    maximum_cross_view_disagreement_m: float = 0.030
    depth_standard_deviation_m: float = 0.005
    shared_bias_standard_deviation_m: float = 0.005

    def __post_init__(self) -> None:
        if not np.isfinite(self.search_radius_px) or self.search_radius_px <= 0.0:
            raise ValueError("search_radius_px must be positive")
        if (
            not np.isfinite(self.descriptor_temperature)
            or self.descriptor_temperature <= 0.0
        ):
            raise ValueError("descriptor_temperature must be positive")
        if not -1.0 <= self.minimum_cosine_similarity <= 1.0:
            raise ValueError("minimum_cosine_similarity must lie in [-1, 1]")
        if not 0.0 <= self.maximum_normalized_entropy <= 1.0:
            raise ValueError("maximum_normalized_entropy must lie in [0, 1]")
        if self.minimum_candidate_count < 1:
            raise ValueError("minimum_candidate_count must be positive")
        if self.pixel_standard_deviation_floor <= 0.0:
            raise ValueError("pixel_standard_deviation_floor must be positive")
        if self.patch_radius_px < 1 or self.patch_search_radius_px < 0:
            raise ValueError("patch radii are invalid")
        if self.patch_temperature <= 0.0:
            raise ValueError("patch_temperature must be positive")
        if not -1.0 <= self.minimum_patch_correlation <= 1.0:
            raise ValueError("minimum_patch_correlation must lie in [-1, 1]")
        if self.minimum_patch_standard_deviation < 0.0:
            raise ValueError("minimum_patch_standard_deviation must be nonnegative")
        if self.minimum_views < 1:
            raise ValueError("minimum_views must be positive")
        if self.maximum_cross_view_disagreement_m <= 0.0:
            raise ValueError("maximum_cross_view_disagreement_m must be positive")
        if (
            self.depth_standard_deviation_m <= 0.0
            or self.shared_bias_standard_deviation_m < 0.0
        ):
            raise ValueError("metric uncertainty scales are invalid")


@dataclass(frozen=True)
class DescriptorMatch:
    """One descriptor association and its assignment-mixture uncertainty."""

    uv_px: np.ndarray
    covariance_px2: np.ndarray
    cosine_similarity: float
    normalized_entropy: float
    association_probability: float
    prior_reliability: float
    candidate_count: int
    accepted: bool
    decision: str

    def __post_init__(self) -> None:
        uv = _readonly(self.uv_px, dtype=np.float64)
        covariance = _readonly(self.covariance_px2, dtype=np.float64)
        if uv.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("descriptor match geometry is invalid")
        _require_finite("uv_px", uv)
        _require_finite("covariance_px2", covariance)
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("covariance_px2 must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
            raise ValueError("covariance_px2 must be positive semidefinite")
        if not -1.0 <= self.cosine_similarity <= 1.0:
            raise ValueError("cosine_similarity must lie in [-1, 1]")
        if not 0.0 <= self.normalized_entropy <= 1.0:
            raise ValueError("normalized_entropy must lie in [0, 1]")
        if not 0.0 <= self.association_probability <= 1.0:
            raise ValueError("association_probability must lie in [0, 1]")
        if not 0.0 <= self.prior_reliability <= 1.0:
            raise ValueError("prior_reliability must lie in [0, 1]")
        if self.candidate_count < 0 or not self.decision:
            raise ValueError("descriptor match diagnostics are invalid")
        object.__setattr__(self, "uv_px", uv)
        object.__setattr__(self, "covariance_px2", covariance)


@dataclass(frozen=True)
class PatchMatch:
    """Pixel-level appearance refinement around a descriptor association."""

    uv_px: np.ndarray
    covariance_px2: np.ndarray
    correlation: float
    normalized_entropy: float
    association_probability: float
    prior_reliability: float
    candidate_count: int
    accepted: bool
    decision: str

    def __post_init__(self) -> None:
        uv = _readonly(self.uv_px, dtype=np.float64)
        covariance = _readonly(self.covariance_px2, dtype=np.float64)
        if uv.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("patch match geometry is invalid")
        _require_finite("uv_px", uv)
        _require_finite("covariance_px2", covariance)
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("covariance_px2 must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
            raise ValueError("covariance_px2 must be positive semidefinite")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must lie in [-1, 1]")
        if not 0.0 <= self.normalized_entropy <= 1.0:
            raise ValueError("normalized_entropy must lie in [0, 1]")
        if not 0.0 <= self.association_probability <= 1.0:
            raise ValueError("association_probability must lie in [0, 1]")
        if not 0.0 <= self.prior_reliability <= 1.0:
            raise ValueError("prior_reliability must lie in [0, 1]")
        if self.candidate_count < 0 or not self.decision:
            raise ValueError("patch match diagnostics are invalid")
        object.__setattr__(self, "uv_px", uv)
        object.__setattr__(self, "covariance_px2", covariance)


@dataclass(frozen=True)
class MetricViewObservation:
    """One RGB-D view observation before unknown-correlation fusion."""

    mean_world_m: np.ndarray
    covariance_world_m2: np.ndarray
    prior_reliability: float
    accepted: bool

    def __post_init__(self) -> None:
        mean = _readonly(self.mean_world_m, dtype=np.float64)
        covariance = _readonly(self.covariance_world_m2, dtype=np.float64)
        if mean.shape != (3,) or covariance.shape != (3, 3):
            raise ValueError("metric view observation geometry is invalid")
        _require_finite("mean_world_m", mean)
        _require_finite("covariance_world_m2", covariance)
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("covariance_world_m2 must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("covariance_world_m2 must be positive definite")
        if not 0.0 <= self.prior_reliability <= 1.0:
            raise ValueError("prior_reliability must lie in [0, 1]")
        object.__setattr__(self, "mean_world_m", mean)
        object.__setattr__(self, "covariance_world_m2", covariance)


@dataclass(frozen=True)
class FusedMetricObservation:
    """Correlation-conservative multiview observation or exact abstention."""

    mean_world_m: np.ndarray
    covariance_world_m2: np.ndarray
    prior_reliability: float
    accepted_view_mask: np.ndarray
    accepted: bool
    decision: str
    maximum_pair_disagreement_m: float | None

    def __post_init__(self) -> None:
        mean = _readonly(self.mean_world_m, dtype=np.float64)
        covariance = _readonly(self.covariance_world_m2, dtype=np.float64)
        mask = _readonly(self.accepted_view_mask, dtype=bool)
        if mean.shape != (3,) or covariance.shape != (3, 3) or mask.ndim != 1:
            raise ValueError("fused observation geometry is invalid")
        _require_finite("mean_world_m", mean)
        _require_finite("covariance_world_m2", covariance)
        if not np.allclose(covariance, covariance.T, atol=1e-12):
            raise ValueError("covariance_world_m2 must be symmetric")
        if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("covariance_world_m2 must be positive definite")
        if not 0.0 <= self.prior_reliability <= 1.0:
            raise ValueError("prior_reliability must lie in [0, 1]")
        if not self.decision:
            raise ValueError("decision must be nonempty")
        object.__setattr__(self, "mean_world_m", mean)
        object.__setattr__(self, "covariance_world_m2", covariance)
        object.__setattr__(self, "accepted_view_mask", mask)


def _normalized_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    _require_finite(name, array)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero descriptor")
    return array / norms


def _softmax_weights(scores: np.ndarray, temperature: float) -> np.ndarray:
    shifted = (np.asarray(scores, dtype=np.float64) - float(np.max(scores))) / (
        temperature
    )
    shifted = np.clip(shifted, -700.0, 0.0)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def _normalized_entropy(weights: np.ndarray) -> float:
    probabilities = np.asarray(weights, dtype=np.float64)
    if len(probabilities) <= 1:
        return 0.0
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny))
    )
    return float(np.clip(entropy / np.log(len(probabilities)), 0.0, 1.0))


def _mixture_covariance(
    positions: np.ndarray,
    weights: np.ndarray,
    *,
    floor_variance: float,
) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    mean = np.sum(probabilities[:, None] * points, axis=0)
    centered = points - mean
    covariance = np.einsum(
        "n,ni,nj->ij",
        probabilities,
        centered,
        centered,
    )
    covariance += floor_variance * np.eye(points.shape[1])
    return covariance


def match_descriptor_near_prediction(
    reference_descriptor: np.ndarray,
    feature_map: np.ndarray,
    valid_mask: np.ndarray,
    predicted_uv_px: np.ndarray,
    *,
    image_width: int,
    image_height: int,
    config: ConditionedDinoConfig | None = None,
) -> DescriptorMatch:
    """Match one material descriptor without using the state innovation.

    ``predicted_uv_px`` controls only which descriptor cells are considered.
    Distance from the physical prediction is deliberately absent from the
    returned reliability and covariance.
    """

    cfg = config or ConditionedDinoConfig()
    features = np.asarray(feature_map, dtype=np.float64)
    validity = np.asarray(valid_mask, dtype=bool)
    predicted = np.asarray(predicted_uv_px, dtype=np.float64)
    if features.ndim != 3 or validity.shape != features.shape[:2]:
        raise ValueError("feature_map and valid_mask shapes differ")
    if predicted.shape != (2,) or image_width < 1 or image_height < 1:
        raise ValueError("image geometry is invalid")
    _require_finite("predicted_uv_px", predicted)
    reference = _normalized_rows(
        np.asarray(reference_descriptor, dtype=np.float64)[None],
        name="reference_descriptor",
    )[0]
    normalized = _normalized_rows(features, name="feature_map")
    rows, columns = features.shape[:2]
    x = (np.arange(columns, dtype=np.float64) + 0.5) * image_width / columns
    y = (np.arange(rows, dtype=np.float64) + 0.5) * image_height / rows
    grid_x, grid_y = np.meshgrid(x, y)
    distance = np.hypot(grid_x - predicted[0], grid_y - predicted[1])
    candidate_mask = validity & (distance <= cfg.search_radius_px)
    candidate_indices = np.argwhere(candidate_mask)
    candidate_count = len(candidate_indices)
    fallback_covariance = (
        cfg.search_radius_px**2 + cfg.pixel_standard_deviation_floor**2
    ) * np.eye(2)
    if candidate_count < cfg.minimum_candidate_count:
        return DescriptorMatch(
            uv_px=predicted,
            covariance_px2=fallback_covariance,
            cosine_similarity=-1.0,
            normalized_entropy=1.0,
            association_probability=0.0,
            prior_reliability=0.0,
            candidate_count=candidate_count,
            accepted=False,
            decision="insufficient_descriptor_candidates",
        )

    descriptors = normalized[candidate_mask]
    scores = np.clip(descriptors @ reference, -1.0, 1.0)
    weights = _softmax_weights(scores, cfg.descriptor_temperature)
    entropy = _normalized_entropy(weights)
    candidate_uv = np.column_stack(
        (
            x[candidate_indices[:, 1]],
            y[candidate_indices[:, 0]],
        )
    )
    best = int(np.argmax(scores))
    best_similarity = float(scores[best])
    best_probability = float(weights[best])
    similarity_support = np.clip(
        (best_similarity - cfg.minimum_cosine_similarity)
        / max(1.0 - cfg.minimum_cosine_similarity, 1e-12),
        0.0,
        1.0,
    )
    uniqueness_support = np.clip(
        (cfg.maximum_normalized_entropy - entropy)
        / max(cfg.maximum_normalized_entropy, 1e-12),
        0.0,
        1.0,
    )
    reliability = float(np.sqrt(similarity_support * uniqueness_support))
    accepted = (
        best_similarity >= cfg.minimum_cosine_similarity
        and entropy <= cfg.maximum_normalized_entropy
    )
    if best_similarity < cfg.minimum_cosine_similarity:
        decision = "descriptor_similarity_below_threshold"
    elif entropy > cfg.maximum_normalized_entropy:
        decision = "descriptor_assignment_too_ambiguous"
    else:
        decision = "accepted"
    covariance = _mixture_covariance(
        candidate_uv,
        weights,
        floor_variance=cfg.pixel_standard_deviation_floor**2,
    )
    return DescriptorMatch(
        uv_px=candidate_uv[best],
        covariance_px2=covariance,
        cosine_similarity=best_similarity,
        normalized_entropy=entropy,
        association_probability=best_probability,
        prior_reliability=reliability,
        candidate_count=candidate_count,
        accepted=accepted,
        decision=decision,
    )


def _grayscale(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 2:
        result = value.astype(np.float64)
    elif value.ndim == 3 and value.shape[2] == 3:
        result = np.tensordot(
            value.astype(np.float64),
            np.asarray([0.299, 0.587, 0.114]),
            axes=([2], [0]),
        )
    else:
        raise ValueError("image must have shape (H,W) or (H,W,3)")
    if np.issubdtype(value.dtype, np.integer) or np.max(result) > 1.5:
        result /= 255.0
    _require_finite("image", result)
    return result


def _integer_patch(
    image: np.ndarray,
    center_uv: np.ndarray,
    radius: int,
) -> np.ndarray | None:
    center = np.rint(np.asarray(center_uv, dtype=np.float64)).astype(np.int64)
    x, y = (int(center[0]), int(center[1]))
    if (
        x - radius < 0
        or x + radius >= image.shape[1]
        or y - radius < 0
        or y + radius >= image.shape[0]
    ):
        return None
    return image[y - radius : y + radius + 1, x - radius : x + radius + 1]


def refine_patch_correlation(
    reference_image: np.ndarray,
    current_image: np.ndarray,
    reference_uv_px: np.ndarray,
    coarse_uv_px: np.ndarray,
    valid_mask: np.ndarray,
    *,
    config: ConditionedDinoConfig | None = None,
) -> PatchMatch:
    """Refine a descriptor match by local normalized patch correlation."""

    cfg = config or ConditionedDinoConfig()
    reference = _grayscale(reference_image)
    current = _grayscale(current_image)
    validity = np.asarray(valid_mask, dtype=bool)
    reference_uv = np.asarray(reference_uv_px, dtype=np.float64)
    coarse_uv = np.asarray(coarse_uv_px, dtype=np.float64)
    if reference.shape != current.shape or validity.shape != current.shape:
        raise ValueError("reference, current, and valid mask shapes differ")
    if reference_uv.shape != (2,) or coarse_uv.shape != (2,):
        raise ValueError("patch coordinates must have shape (2,)")
    reference_patch = _integer_patch(reference, reference_uv, cfg.patch_radius_px)
    fallback_covariance = (
        cfg.patch_search_radius_px**2 + cfg.pixel_standard_deviation_floor**2
    ) * np.eye(2)
    if reference_patch is None:
        return PatchMatch(
            uv_px=coarse_uv,
            covariance_px2=fallback_covariance,
            correlation=-1.0,
            normalized_entropy=1.0,
            association_probability=0.0,
            prior_reliability=0.0,
            candidate_count=0,
            accepted=False,
            decision="reference_patch_outside_image",
        )
    reference_centered = reference_patch - np.mean(reference_patch)
    reference_norm = float(np.linalg.norm(reference_centered))
    if np.std(reference_patch) < cfg.minimum_patch_standard_deviation:
        return PatchMatch(
            uv_px=coarse_uv,
            covariance_px2=fallback_covariance,
            correlation=-1.0,
            normalized_entropy=1.0,
            association_probability=0.0,
            prior_reliability=0.0,
            candidate_count=0,
            accepted=False,
            decision="reference_patch_has_insufficient_texture",
        )

    coarse_integer = np.rint(coarse_uv).astype(np.int64)
    positions: list[np.ndarray] = []
    scores: list[float] = []
    for offset_y in range(
        -cfg.patch_search_radius_px,
        cfg.patch_search_radius_px + 1,
    ):
        for offset_x in range(
            -cfg.patch_search_radius_px,
            cfg.patch_search_radius_px + 1,
        ):
            uv = coarse_integer + np.asarray([offset_x, offset_y])
            x, y = int(uv[0]), int(uv[1])
            if not (0 <= x < current.shape[1] and 0 <= y < current.shape[0]):
                continue
            if not validity[y, x]:
                continue
            patch = _integer_patch(current, uv, cfg.patch_radius_px)
            if patch is None or np.std(patch) < cfg.minimum_patch_standard_deviation:
                continue
            centered = patch - np.mean(patch)
            denominator = reference_norm * float(np.linalg.norm(centered))
            if denominator <= 1e-12:
                continue
            positions.append(uv.astype(np.float64))
            correlation = np.sum(reference_centered * centered) / denominator
            scores.append(
                float(np.clip(correlation, -1.0, 1.0))
            )
    candidate_count = len(scores)
    if candidate_count == 0:
        return PatchMatch(
            uv_px=coarse_uv,
            covariance_px2=fallback_covariance,
            correlation=-1.0,
            normalized_entropy=1.0,
            association_probability=0.0,
            prior_reliability=0.0,
            candidate_count=0,
            accepted=False,
            decision="no_valid_patch_candidate",
        )
    score_array = np.asarray(scores, dtype=np.float64)
    position_array = np.asarray(positions, dtype=np.float64)
    weights = _softmax_weights(score_array, cfg.patch_temperature)
    entropy = _normalized_entropy(weights)
    best = int(np.argmax(score_array))
    correlation = float(score_array[best])
    support = np.clip(
        (correlation - cfg.minimum_patch_correlation)
        / max(1.0 - cfg.minimum_patch_correlation, 1e-12),
        0.0,
        1.0,
    )
    reliability = float(np.sqrt(support * max(1.0 - entropy, 0.0)))
    accepted = correlation >= cfg.minimum_patch_correlation
    covariance = _mixture_covariance(
        position_array,
        weights,
        floor_variance=cfg.pixel_standard_deviation_floor**2,
    )
    return PatchMatch(
        uv_px=position_array[best],
        covariance_px2=covariance,
        correlation=correlation,
        normalized_entropy=entropy,
        association_probability=float(weights[best]),
        prior_reliability=reliability,
        candidate_count=candidate_count,
        accepted=accepted,
        decision="accepted" if accepted else "patch_correlation_below_threshold",
    )


def unproject_rgbd_observation(
    uv_px: np.ndarray,
    covariance_px2: np.ndarray,
    depth_m: float,
    intrinsics: np.ndarray,
    camera_to_world: np.ndarray,
    *,
    prior_reliability: float,
    depth_standard_deviation_m: float,
    accepted: bool = True,
) -> MetricViewObservation:
    """Unproject a pixel and propagate pixel/depth covariance into metres."""

    uv = np.asarray(uv_px, dtype=np.float64)
    pixel_covariance = np.asarray(covariance_px2, dtype=np.float64)
    camera_matrix = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    if (
        uv.shape != (2,)
        or pixel_covariance.shape != (2, 2)
        or camera_matrix.shape != (3, 3)
        or transform.shape != (4, 4)
    ):
        raise ValueError("unprojection geometry is invalid")
    _require_finite("unprojection inputs", np.concatenate((uv, pixel_covariance.ravel())))
    if not np.isfinite(depth_m) or depth_m <= 0.0:
        raise ValueError("depth_m must be positive")
    if depth_standard_deviation_m <= 0.0:
        raise ValueError("depth_standard_deviation_m must be positive")
    if not 0.0 <= prior_reliability <= 1.0:
        raise ValueError("prior_reliability must lie in [0, 1]")
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    camera_point = np.asarray(
        [
            (uv[0] - cx) * depth_m / fx,
            (uv[1] - cy) * depth_m / fy,
            depth_m,
        ]
    )
    world_point = transform[:3, :3] @ camera_point + transform[:3, 3]
    jacobian = np.asarray(
        [
            [depth_m / fx, 0.0, (uv[0] - cx) / fx],
            [0.0, depth_m / fy, (uv[1] - cy) / fy],
            [0.0, 0.0, 1.0],
        ]
    )
    source_covariance = np.zeros((3, 3), dtype=np.float64)
    source_covariance[:2, :2] = pixel_covariance
    source_covariance[2, 2] = depth_standard_deviation_m**2
    camera_covariance = jacobian @ source_covariance @ jacobian.T
    rotation = transform[:3, :3]
    world_covariance = rotation @ camera_covariance @ rotation.T
    world_covariance = 0.5 * (world_covariance + world_covariance.T)
    world_covariance += np.finfo(float).eps * np.eye(3)
    return MetricViewObservation(
        mean_world_m=world_point,
        covariance_world_m2=world_covariance,
        prior_reliability=prior_reliability,
        accepted=accepted,
    )


def covariance_intersection(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
    *,
    grid_size: int = 101,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fuse two estimates without assuming independent errors."""

    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    first_mean = np.asarray(mean_a, dtype=np.float64)
    second_mean = np.asarray(mean_b, dtype=np.float64)
    first_covariance = np.asarray(covariance_a, dtype=np.float64)
    second_covariance = np.asarray(covariance_b, dtype=np.float64)
    expected_covariance_shape = (len(first_mean), len(first_mean))
    if (
        first_mean.shape != second_mean.shape
        or first_covariance.shape != expected_covariance_shape
        or second_covariance.shape != expected_covariance_shape
    ):
        raise ValueError("covariance-intersection shapes differ")
    _require_finite("covariance-intersection means", np.concatenate((first_mean, second_mean)))
    _require_finite(
        "covariance-intersection covariances",
        np.concatenate((first_covariance.ravel(), second_covariance.ravel())),
    )
    if (
        not np.allclose(first_covariance, first_covariance.T, atol=1e-12)
        or not np.allclose(second_covariance, second_covariance.T, atol=1e-12)
        or np.min(np.linalg.eigvalsh(first_covariance)) <= 0.0
        or np.min(np.linalg.eigvalsh(second_covariance)) <= 0.0
    ):
        raise ValueError("covariance-intersection inputs must be positive definite")
    first_precision = np.linalg.inv(first_covariance)
    second_precision = np.linalg.inv(second_covariance)
    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for weight in np.linspace(0.0, 1.0, grid_size):
        precision = weight * first_precision + (1.0 - weight) * second_precision
        covariance = np.linalg.inv(precision)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0.0:
            continue
        information_mean = (
            weight * first_precision @ first_mean
            + (1.0 - weight) * second_precision @ second_mean
        )
        mean = covariance @ information_mean
        candidate = (float(logdet), float(weight), mean, covariance)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise ValueError("covariance intersection found no positive covariance")
    return best[2], best[3], best[1]


def _maximum_consensus_indices(
    points: np.ndarray,
    maximum_distance: float,
) -> np.ndarray:
    count = len(points)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    compatible = distance <= maximum_distance
    best: tuple[int, ...] = ()
    best_spread = float("inf")
    for size in range(1, count + 1):
        for subset in combinations(range(count), size):
            selected = np.asarray(subset, dtype=np.int64)
            local = compatible[np.ix_(selected, selected)]
            if not np.all(local):
                continue
            spread = float(np.max(distance[np.ix_(selected, selected)]))
            if len(subset) > len(best) or (
                len(subset) == len(best)
                and (spread, subset) < (best_spread, best)
            ):
                best = subset
                best_spread = spread
    return np.asarray(best, dtype=np.int64)


def fuse_unknown_correlation(
    observations: list[MetricViewObservation] | tuple[MetricViewObservation, ...],
    *,
    config: ConditionedDinoConfig | None = None,
) -> FusedMetricObservation:
    """Fuse a cross-view consensus by covariance intersection.

    Repeating an identical camera does not divide its covariance by the number
    of copies.  A shared-bias floor is added once after fusion.
    """

    cfg = config or ConditionedDinoConfig()
    count = len(observations)
    eligible_indices = np.asarray(
        [
            index
            for index, observation in enumerate(observations)
            if observation.accepted and observation.prior_reliability > 0.0
        ],
        dtype=np.int64,
    )
    accepted_mask = np.zeros(count, dtype=bool)
    fallback_covariance = (
        cfg.maximum_cross_view_disagreement_m**2
        + cfg.shared_bias_standard_deviation_m**2
    ) * np.eye(3)
    if len(eligible_indices) < cfg.minimum_views:
        return FusedMetricObservation(
            mean_world_m=np.zeros(3),
            covariance_world_m2=fallback_covariance,
            prior_reliability=0.0,
            accepted_view_mask=accepted_mask,
            accepted=False,
            decision="insufficient_accepted_views",
            maximum_pair_disagreement_m=None,
        )
    eligible_points = np.asarray(
        [observations[index].mean_world_m for index in eligible_indices]
    )
    consensus_local = _maximum_consensus_indices(
        eligible_points,
        cfg.maximum_cross_view_disagreement_m,
    )
    consensus = eligible_indices[consensus_local]
    if len(consensus) < cfg.minimum_views:
        return FusedMetricObservation(
            mean_world_m=np.zeros(3),
            covariance_world_m2=fallback_covariance,
            prior_reliability=0.0,
            accepted_view_mask=accepted_mask,
            accepted=False,
            decision="insufficient_cross_view_consensus",
            maximum_pair_disagreement_m=None,
        )
    accepted_mask[consensus] = True
    mean = observations[int(consensus[0])].mean_world_m.copy()
    covariance = observations[int(consensus[0])].covariance_world_m2.copy()
    for index in consensus[1:]:
        observation = observations[int(index)]
        mean, covariance, _ = covariance_intersection(
            mean,
            covariance,
            observation.mean_world_m,
            observation.covariance_world_m2,
        )
    covariance = covariance + (
        cfg.shared_bias_standard_deviation_m**2 * np.eye(3)
    )
    selected_points = np.asarray(
        [observations[int(index)].mean_world_m for index in consensus]
    )
    maximum_disagreement = float(
        np.max(
            np.linalg.norm(
                selected_points[:, None] - selected_points[None, :],
                axis=2,
            )
        )
    )
    reliability = float(
        min(observations[int(index)].prior_reliability for index in consensus)
    )
    return FusedMetricObservation(
        mean_world_m=mean,
        covariance_world_m2=covariance,
        prior_reliability=reliability,
        accepted_view_mask=accepted_mask,
        accepted=True,
        decision="accepted",
        maximum_pair_disagreement_m=maximum_disagreement,
    )


def exact_fallback_points(
    baseline_points_m: np.ndarray,
    candidate_points_m: np.ndarray,
    accepted: np.ndarray,
) -> np.ndarray:
    """Route accepted rows and preserve rejected baseline bytes exactly."""

    baseline = np.asarray(baseline_points_m)
    candidate = np.asarray(candidate_points_m)
    mask = np.asarray(accepted, dtype=bool)
    if baseline.shape != candidate.shape or baseline.shape[-1] != 3:
        raise ValueError("baseline and candidate must share shape (..., 3)")
    if mask.shape != baseline.shape[:-1]:
        raise ValueError("accepted mask shape differs from point arrays")
    result = baseline.copy()
    result[mask] = candidate[mask]
    if not np.array_equal(result[~mask], baseline[~mask]):
        raise AssertionError("rejected rows are not exact baseline fallback")
    return result


__all__ = [
    "ConditionedDinoConfig",
    "DescriptorMatch",
    "FusedMetricObservation",
    "MetricViewObservation",
    "PatchMatch",
    "covariance_intersection",
    "exact_fallback_points",
    "fuse_unknown_correlation",
    "match_descriptor_near_prediction",
    "refine_patch_correlation",
    "unproject_rgbd_observation",
]
