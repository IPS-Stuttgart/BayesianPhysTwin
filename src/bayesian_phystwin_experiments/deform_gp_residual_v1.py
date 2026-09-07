"""Bounded Nyström GP residual diagnostic; owns issue #946, not a stable API.

The finite-rank kernel is K_xu (K_uu + nugget I)^-1 K_ux. The nugget
is an explicit approximation parameter, never an emergency numerical repair.
All rows are used with equal total likelihood weight per trajectory/node.
Posterior covariance is conditional on this working likelihood, not calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def checked(values: np.ndarray, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not array.size or not np.isfinite(array).all():
        raise ValueError("expected a nonempty finite array with the declared rank")
    return array


def matern_material_kernel(
    x: np.ndarray,
    z: np.ndarray,
    length: float,
    x_arc: np.ndarray | None = None,
    z_arc: np.ndarray | None = None,
    material_length: float = 0.35,
) -> np.ndarray:
    """Matérn-3/2 state kernel, optionally times a material-coordinate RBF."""
    x, z = checked(x, 2), checked(z, 2)
    if x.shape[1] != z.shape[1] or not np.isfinite(length) or length <= 0:
        raise ValueError("invalid kernel dimensions or length")
    # Explicit differences avoid cancellation and hidden distance clipping.
    distance2 = np.zeros((len(x), len(z)))
    for column in range(x.shape[1]):
        distance2 += (x[:, column, None] - z[None, :, column]) ** 2
    radius = np.sqrt(3.0 * distance2) / length
    kernel = (1.0 + radius) * np.exp(-radius)
    if (x_arc is None) != (z_arc is None):
        raise ValueError("material coordinates must be supplied together")
    if x_arc is not None:
        a, b = checked(x_arc, 1), checked(z_arc, 1)
        if len(a) != len(x) or len(b) != len(z) or material_length <= 0:
            raise ValueError("invalid material coordinates")
        kernel *= np.exp(-0.5 * ((a[:, None] - b[None]) / material_length) ** 2)
    return kernel


@dataclass
class FiniteGP:
    location: np.ndarray
    scale: np.ndarray
    anchors: np.ndarray
    anchor_arc: np.ndarray | None
    factor: np.ndarray
    length: float
    material_length: float
    output_scale: np.ndarray
    weights: np.ndarray
    weight_covariance: np.ndarray
    noise: float

    def predict(self, x: np.ndarray, arc: np.ndarray | None = None) -> np.ndarray:
        x = checked(x, 2)
        result = []
        for start in range(0, len(x), 2048):
            stop = start + 2048
            standardized = (x[start:stop] - self.location) / self.scale
            local_arc = None if arc is None else arc[start:stop]
            cross = matern_material_kernel(
                standardized,
                self.anchors,
                self.length,
                local_arc,
                self.anchor_arc,
                self.material_length,
            )
            basis = np.linalg.solve(self.factor, cross.T).T
            result.append((basis @ self.weights) * self.output_scale)
        return np.concatenate(result)

    def archive(self) -> dict[str, np.ndarray]:
        return {
            "location": self.location,
            "scale": self.scale,
            "anchors": self.anchors,
            "anchor_arc": np.array([]) if self.anchor_arc is None else self.anchor_arc,
            "factor": self.factor,
            "length": np.array(self.length),
            "material_length": np.array(self.material_length),
            "output_scale": self.output_scale,
            "weights": self.weights,
            "weight_covariance": self.weight_covariance,
            "noise": np.array(self.noise),
        }


def fit_gp_bank(
    x: np.ndarray,
    y: np.ndarray,
    *,
    arc: np.ndarray | None,
    anchor_indices: np.ndarray,
    length_multiplier: float,
    noises: list[float],
    row_weight: float,
    nugget: float = 1e-6,
    material_length: float = 0.35,
) -> dict[float, FiniteGP]:
    """Fit several noise levels from the same training-only sufficient statistics."""
    x, y = checked(x, 2), checked(y, 2)
    indices = np.asarray(anchor_indices)
    if len(x) != len(y) or indices.ndim != 1 or len(indices) < 2:
        raise ValueError("invalid training arrays or anchors")
    if (
        indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(x))
    ):
        raise ValueError("invalid anchor indices")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("anchor indices repeat")
    if any(
        not np.isfinite(v) or v <= 0
        for v in [row_weight, nugget, length_multiplier, *noises]
    ):
        raise ValueError("positive finite hyperparameters required")
    location = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-10, scale, 1.0)
    anchors = (x[indices] - location) / scale
    # Training-only median pair distance defines the state length scale.
    distances = np.sqrt(np.sum((anchors[:, None] - anchors[None]) ** 2, axis=2))
    positive = distances[distances > 1e-10]
    length = (float(np.median(positive)) if positive.size else 1.0) * length_multiplier
    anchor_arc = None if arc is None else checked(arc, 1)[indices]
    kuu = matern_material_kernel(
        anchors, anchors, length, anchor_arc, anchor_arc, material_length
    )
    factor = np.linalg.cholesky(kuu + nugget * np.eye(len(indices)))
    # Declared scale floor is 1 mm; this is not target calibration.
    output_scale = np.maximum(np.sqrt(np.mean(y * y, axis=0)), 0.001)
    gram = np.zeros((len(indices), len(indices)))
    rhs = np.zeros((len(indices), y.shape[1]))
    for start in range(0, len(x), 2048):
        stop = start + 2048
        local_arc = None if arc is None else arc[start:stop]
        cross = matern_material_kernel(
            (x[start:stop] - location) / scale,
            anchors,
            length,
            local_arc,
            anchor_arc,
            material_length,
        )
        basis = np.linalg.solve(factor, cross.T).T
        gram += row_weight * (basis.T @ basis)
        rhs += row_weight * (basis.T @ (y[start:stop] / output_scale))
    result = {}
    for noise in noises:
        normal = gram + noise * np.eye(len(indices))
        chol = np.linalg.cholesky(normal)
        weights = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
        covariance = noise * np.linalg.solve(
            chol.T, np.linalg.solve(chol, np.eye(len(indices)))
        )
        result[noise] = FiniteGP(
            location,
            scale,
            anchors,
            anchor_arc,
            factor,
            length,
            material_length,
            output_scale,
            weights,
            covariance,
            noise,
        )
    return result


def balanced_anchor_indices(
    trajectories: int, horizon: int, nodes: int, count: int
) -> np.ndarray:
    """Deterministic, output-independent landmarks with every trajectory represented."""
    if min(trajectories, horizon, nodes) < 1 or count < trajectories:
        raise ValueError("invalid landmark budget")
    count = min(count, trajectories * horizon * nodes)
    indices = []
    for trajectory in range(trajectories):
        allocated = count // trajectories + int(trajectory < count % trajectories)
        local = np.linspace(0, horizon * nodes - 1, allocated, dtype=np.int64)
        indices.extend((trajectory * horizon * nodes + local).tolist())
    return np.asarray(indices, dtype=np.int64)


def add_residual(
    baseline: np.ndarray, canonical: np.ndarray, frames: np.ndarray, shrinkage: float
) -> np.ndarray:
    baseline, canonical, frames = (
        checked(baseline, 4),
        checked(canonical, 4),
        checked(frames, 3),
    )
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("invalid shrinkage")
    if canonical.shape != (*baseline.shape[:2], baseline.shape[2] - 4, 3):
        raise ValueError("residual shape mismatch")
    result = baseline.copy()
    if shrinkage:
        result[:, :, 2:-2] += shrinkage * np.einsum("ntvj,nij->ntvi", canonical, frames)
    return result
