"""Deterministic reductions of weighted physical-parameter support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


SupportMethod = Literal["top_mass", "weighted_coreset"]


def _normalized_weights(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a nonempty vector")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have positive mass")
    return weights / total


def weighted_parameter_moments(
    particles: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted mean and population covariance."""

    values = np.asarray(particles, dtype=float)
    normalized = _normalized_weights(weights)
    if values.ndim != 2 or values.shape[0] != len(normalized):
        raise ValueError("particles must have shape (particle, parameter)")
    if not np.all(np.isfinite(values)):
        raise ValueError("particles must be finite")
    mean = np.einsum("p,pd->d", normalized, values)
    centered = values - mean
    covariance = np.einsum("p,pi,pj->ij", normalized, centered, centered)
    return mean, covariance


@dataclass(frozen=True)
class ParameterSupportReduction:
    """A selected support and the source mass represented by its weights."""

    method: SupportMethod
    indices: np.ndarray
    weights: np.ndarray
    source_particle_count: int
    directly_retained_probability_mass: float
    represented_probability_mass: float
    mean_error_l2: float
    covariance_error_frobenius: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        weights = _normalized_weights(self.weights)
        if indices.ndim != 1 or len(indices) != len(weights):
            raise ValueError("support indices and weights must be matching vectors")
        if len(set(map(int, indices))) != len(indices):
            raise ValueError("support indices must be unique")
        if np.any(indices < 0) or np.any(indices >= self.source_particle_count):
            raise ValueError("support index exceeds the source particles")
        for value in (
            self.directly_retained_probability_mass,
            self.represented_probability_mass,
        ):
            if not 0.0 < value <= 1.0 + 1e-12:
                raise ValueError("support probability masses must lie in (0, 1]")
        if self.mean_error_l2 < 0.0 or self.covariance_error_frobenius < 0.0:
            raise ValueError("support moment errors must be nonnegative")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "weights", weights)

    @property
    def count(self) -> int:
        return len(self.indices)

    @property
    def effective_support(self) -> float:
        return float(1.0 / np.sum(np.square(self.weights)))

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "count": self.count,
            "source_particle_count": self.source_particle_count,
            "indices": self.indices.tolist(),
            "weights": self.weights.tolist(),
            "directly_retained_probability_mass": (
                self.directly_retained_probability_mass
            ),
            "represented_probability_mass": self.represented_probability_mass,
            "effective_support": self.effective_support,
            "parameter_mean_error_l2": self.mean_error_l2,
            "parameter_covariance_error_frobenius": (self.covariance_error_frobenius),
        }


def _top_mass_reduction(
    particles: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    order = np.lexsort((np.arange(len(weights)), -weights))
    selected = order[:count]
    retained = float(np.sum(weights[selected]))
    return selected, weights[selected] / retained, retained, retained


def _weighted_coreset_reduction(
    particles: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if count == len(weights):
        return np.arange(count), weights.copy(), 1.0, 1.0

    mean, covariance = weighted_parameter_moments(particles, weights)
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    normalized = (particles - mean) / scale
    differences = normalized[:, None, :] - normalized[None, :, :]
    squared_distance = np.einsum("ijd,ijd->ij", differences, differences)

    medoids = [int(np.argmax(weights))]
    nearest = squared_distance[:, medoids[0]].copy()
    while len(medoids) < count:
        scores = weights * nearest
        scores[np.asarray(medoids, dtype=int)] = -1.0
        candidate = int(np.argmax(scores))
        medoids.append(candidate)
        nearest = np.minimum(nearest, squared_distance[:, candidate])

    medoids_array = np.asarray(medoids, dtype=np.int64)
    for _ in range(100):
        assignment = np.argmin(squared_distance[:, medoids_array], axis=1)
        updated = medoids_array.copy()
        for cluster_index in range(count):
            members = np.flatnonzero(assignment == cluster_index)
            if len(members) == 0:
                nearest = np.min(squared_distance[:, updated], axis=1)
                score = weights * nearest
                score[updated] = -1.0
                updated[cluster_index] = int(np.argmax(score))
                continue
            objective = squared_distance[np.ix_(members, members)].T @ weights[members]
            updated[cluster_index] = int(members[int(np.argmin(objective))])
        if np.array_equal(updated, medoids_array):
            break
        medoids_array = updated

    assignment = np.argmin(squared_distance[:, medoids_array], axis=1)
    reduced_weights = np.asarray(
        [np.sum(weights[assignment == index]) for index in range(count)],
        dtype=float,
    )
    order = np.lexsort((medoids_array, -reduced_weights))
    medoids_array = medoids_array[order]
    reduced_weights = reduced_weights[order]
    direct_mass = float(np.sum(weights[medoids_array]))
    return medoids_array, reduced_weights, direct_mass, 1.0


def reduce_parameter_support(
    particles: np.ndarray,
    weights: np.ndarray,
    *,
    maximum_count: int,
    method: SupportMethod = "top_mass",
) -> ParameterSupportReduction:
    """Reduce a weighted posterior without consulting predictive targets.

    ``top_mass`` truncates and renormalizes the selected source cells.
    ``weighted_coreset`` assigns every source cell to a deterministic weighted
    medoid and transfers its mass, preserving full represented probability.
    """

    values = np.asarray(particles, dtype=float)
    normalized = _normalized_weights(weights)
    if values.ndim != 2 or values.shape[0] != len(normalized):
        raise ValueError("particles must have shape (particle, parameter)")
    if not np.all(np.isfinite(values)):
        raise ValueError("particles must be finite")
    if maximum_count < 1:
        raise ValueError("maximum_count must be positive")
    count = min(int(maximum_count), len(normalized))
    if method == "top_mass":
        selected, reduced, retained, represented = _top_mass_reduction(
            values,
            normalized,
            count,
        )
    elif method == "weighted_coreset":
        selected, reduced, retained, represented = _weighted_coreset_reduction(
            values,
            normalized,
            count,
        )
    else:
        raise ValueError(f"unknown parameter support method {method!r}")

    full_mean, full_covariance = weighted_parameter_moments(values, normalized)
    reduced_mean, reduced_covariance = weighted_parameter_moments(
        values[selected],
        reduced,
    )
    return ParameterSupportReduction(
        method=method,
        indices=selected,
        weights=reduced,
        source_particle_count=len(normalized),
        directly_retained_probability_mass=retained,
        represented_probability_mass=represented,
        mean_error_l2=float(np.linalg.norm(reduced_mean - full_mean)),
        covariance_error_frobenius=float(
            np.linalg.norm(reduced_covariance - full_covariance)
        ),
    )
