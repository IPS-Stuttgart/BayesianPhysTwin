"""Exact task-uncertainty certificates for affine capability regions.

A decision-capability region has half-spaces ``normal @ theta <= offset``.
For a centered convex task-uncertainty set ``U``, every objective in
``center + U`` is capable exactly when

    normal[j] @ center + sigma_U(normal[j]) <= offset[j]

for every half-space, where ``sigma_U`` is the support function. Boxes and
ellipsoids therefore have closed forms. Norm-ball margins additionally report
the largest objective perturbation around a capable nominal task.

The result is conditional on the supplied atlas and task-uncertainty set. It
does not validate the objective, quantify model misspecification, certify
safety, or authorize deployment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

from .decision_capability_atlas_v1 import AffineCapabilityHalfspacesV1

FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
TaskNorm: TypeAlias = Literal["l1", "l2", "linf"]

TASK_UNCERTAINTY_CERTIFICATE_VERSION: Final = 1
TASK_UNCERTAINTY_CERTIFICATE_SEMANTICS: Final = (
    "exact-convex-task-set-containment-in-affine-capability-region-v1"
)
TASK_UNCERTAINTY_CERTIFICATE_CLAIM_BOUNDARY: Final = (
    "The task-set certificate is exact only for the supplied affine capability "
    "half-spaces and declared convex task-uncertainty set. It inherits the "
    "finite-support, quotient, loss-family, action-set, and regret-tolerance "
    "conditions of the underlying atlas. It does not validate the objective or "
    "uncertainty set, cover model misspecification, certify safety, or authorize "
    "deployment."
)

_NUMERICAL_ATOL: Final = 1e-12


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_int64(value: object) -> IntArray:
    array = np.ascontiguousarray(value, dtype=np.int64)
    array.setflags(write=False)
    return array


def _immutable_bool(value: object) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    array.setflags(write=False)
    return array


def _halfspaces(value: object) -> AffineCapabilityHalfspacesV1:
    if not isinstance(value, AffineCapabilityHalfspacesV1):
        raise TypeError("halfspaces must be AffineCapabilityHalfspacesV1")
    normal = np.asarray(value.normal)
    offset = np.asarray(value.offset)
    if normal.ndim != 2 or offset.ndim != 1 or normal.shape[0] != offset.size:
        raise ValueError("halfspaces contain inconsistent normal and offset arrays")
    if normal.shape[1] < 1:
        raise ValueError("halfspaces must have a positive task dimension")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(offset)):
        raise ValueError("halfspaces must be finite")
    return value


def _centers(value: object, *, dimension: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("task_centers must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != dimension:
        raise ValueError(
            "task_centers must have shape (task_set_count, task_dimension)"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("task_centers must be finite")
    return _immutable_float64(array)


def _batched_nonnegative(
    value: object,
    *,
    name: str,
    count: int,
    width: int,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (width,):
            raise ValueError(f"{name} must contain one value per task dimension")
        array = np.broadcast_to(array[None, :], (count, width)).copy()
    elif array.ndim == 2:
        if array.shape != (count, width):
            raise ValueError(
                f"{name} must have shape (task_set_count, task_dimension)"
            )
    else:
        raise ValueError(f"{name} must be one- or two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return _immutable_float64(array)


class TaskSetCapabilityV1(NamedTuple):
    """Exact containment result for one action and uncertain-task roster."""

    action_index: int
    uncertainty_kind: str
    task_centers: FloatArray
    support_value: FloatArray
    worst_excess: FloatArray
    critical_halfspace_index: IntArray
    capable_mask: BoolArray

    @property
    def all_capable(self) -> bool:
        return bool(np.all(self.capable_mask))

    @property
    def minimum_slack(self) -> FloatArray:
        return _immutable_float64(-self.worst_excess)

    def summary(self) -> dict[str, object]:
        return {
            "version": TASK_UNCERTAINTY_CERTIFICATE_VERSION,
            "semantics": TASK_UNCERTAINTY_CERTIFICATE_SEMANTICS,
            "action_index": self.action_index,
            "uncertainty_kind": self.uncertainty_kind,
            "task_set_count": int(self.task_centers.shape[0]),
            "task_dimension": int(self.task_centers.shape[1]),
            "halfspace_count": int(self.support_value.shape[1]),
            "capable_task_set_count": int(np.count_nonzero(self.capable_mask)),
            "claim_boundary": TASK_UNCERTAINTY_CERTIFICATE_CLAIM_BOUNDARY,
        }


class TaskCapabilityMarginV1(NamedTuple):
    """Normalized constraint margin and exact inscribed-ball radius."""

    action_index: int
    task_norm: str
    task_centers: FloatArray
    normalized_constraint_margin: FloatArray
    guaranteed_radius: FloatArray
    critical_halfspace_index: IntArray
    center_capable_mask: BoolArray

    @property
    def task_count(self) -> int:
        return int(self.task_centers.shape[0])

    @property
    def task_dimension(self) -> int:
        return int(self.task_centers.shape[1])

    def summary(self) -> dict[str, object]:
        return {
            "version": TASK_UNCERTAINTY_CERTIFICATE_VERSION,
            "semantics": TASK_UNCERTAINTY_CERTIFICATE_SEMANTICS,
            "action_index": self.action_index,
            "task_norm": self.task_norm,
            "task_count": self.task_count,
            "task_dimension": self.task_dimension,
            "capable_center_count": int(np.count_nonzero(self.center_capable_mask)),
            "claim_boundary": TASK_UNCERTAINTY_CERTIFICATE_CLAIM_BOUNDARY,
        }


def _task_set_result(
    halfspaces: AffineCapabilityHalfspacesV1,
    centers: FloatArray,
    support: FloatArray,
    *,
    uncertainty_kind: str,
) -> TaskSetCapabilityV1:
    expected = (centers.shape[0], halfspaces.halfspace_count)
    if support.shape != expected:
        raise ValueError("support value has the wrong task-set or half-space shape")
    if not np.all(np.isfinite(support)) or np.any(support < 0.0):
        raise ValueError("support value must be finite and nonnegative")
    if halfspaces.halfspace_count:
        excess = centers @ halfspaces.normal.T + support - halfspaces.offset[None, :]
        critical = np.argmax(excess, axis=1).astype(np.int64, copy=False)
        worst = excess[np.arange(centers.shape[0]), critical]
    else:
        critical = np.full(centers.shape[0], -1, dtype=np.int64)
        worst = np.full(centers.shape[0], -np.inf, dtype=np.float64)
    return TaskSetCapabilityV1(
        action_index=halfspaces.action_index,
        uncertainty_kind=uncertainty_kind,
        task_centers=centers,
        support_value=_immutable_float64(support),
        worst_excess=_immutable_float64(worst),
        critical_halfspace_index=_immutable_int64(critical),
        capable_mask=_immutable_bool(worst <= _NUMERICAL_ATOL),
    )


def box_task_set_capability(
    halfspaces: AffineCapabilityHalfspacesV1,
    task_centers: object,
    half_widths: object,
) -> TaskSetCapabilityV1:
    """Certify ``abs(theta-center) <= half_width`` boxes exactly."""

    region = _halfspaces(halfspaces)
    centers = _centers(task_centers, dimension=region.task_dimension)
    widths = _batched_nonnegative(
        half_widths,
        name="half_widths",
        count=centers.shape[0],
        width=region.task_dimension,
    )
    return _task_set_result(
        region,
        centers,
        widths @ np.abs(region.normal).T,
        uncertainty_kind="axis-aligned-box",
    )


def ellipsoid_task_set_capability(
    halfspaces: AffineCapabilityHalfspacesV1,
    task_centers: object,
    generators: object,
) -> TaskSetCapabilityV1:
    """Certify ``center + G u, ||u||_2 <= 1`` ellipsoids exactly."""

    region = _halfspaces(halfspaces)
    centers = _centers(task_centers, dimension=region.task_dimension)
    raw = np.asarray(generators)
    if raw.dtype.kind not in "iuf":
        raise ValueError("generators must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim == 2:
        if array.shape[0] != region.task_dimension or array.shape[1] < 1:
            raise ValueError(
                "generators must have shape (task_dimension, latent_dimension)"
            )
        array = np.broadcast_to(
            array[None, :, :],
            (centers.shape[0], *array.shape),
        ).copy()
    elif array.ndim == 3:
        if (
            array.shape[0] != centers.shape[0]
            or array.shape[1] != region.task_dimension
            or array.shape[2] < 1
        ):
            raise ValueError(
                "batched generators must have shape "
                "(task_set_count, task_dimension, latent_dimension)"
            )
    else:
        raise ValueError("generators must be two- or three-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("generators must be finite")
    projected = np.einsum("hd,ndk->nhk", region.normal, array)
    return _task_set_result(
        region,
        centers,
        np.linalg.norm(projected, axis=2),
        uncertainty_kind="centered-ellipsoid",
    )


def norm_ball_capability_margin(
    halfspaces: AffineCapabilityHalfspacesV1,
    task_centers: object,
    *,
    task_norm: TaskNorm = "l2",
) -> TaskCapabilityMarginV1:
    """Return the exact inscribed radius for capable centers.

    Outside the region, the negative normalized constraint margin is a violation
    diagnostic rather than the distance to the polyhedron.
    """

    region = _halfspaces(halfspaces)
    centers = _centers(task_centers, dimension=region.task_dimension)
    if task_norm == "l2":
        denominator = np.linalg.norm(region.normal, ord=2, axis=1)
    elif task_norm == "linf":
        denominator = np.linalg.norm(region.normal, ord=1, axis=1)
    elif task_norm == "l1":
        denominator = np.linalg.norm(region.normal, ord=np.inf, axis=1)
    else:
        raise ValueError("task_norm must be 'l1', 'l2', or 'linf'")

    if region.halfspace_count == 0:
        margin = np.full(centers.shape[0], np.inf, dtype=np.float64)
        critical = np.full(centers.shape[0], -1, dtype=np.int64)
    else:
        slack = region.offset[None, :] - centers @ region.normal.T
        ratio = np.empty_like(slack)
        active = denominator > _NUMERICAL_ATOL
        ratio[:, active] = slack[:, active] / denominator[active]
        if np.any(~active):
            ratio[:, ~active] = np.where(
                slack[:, ~active] >= -_NUMERICAL_ATOL,
                np.inf,
                -np.inf,
            )
        critical = np.argmin(ratio, axis=1).astype(np.int64, copy=False)
        margin = ratio[np.arange(centers.shape[0]), critical]
    capable = margin >= -_NUMERICAL_ATOL
    return TaskCapabilityMarginV1(
        action_index=region.action_index,
        task_norm=task_norm,
        task_centers=centers,
        normalized_constraint_margin=_immutable_float64(margin),
        guaranteed_radius=_immutable_float64(
            np.where(capable, np.maximum(margin, 0.0), 0.0)
        ),
        critical_halfspace_index=_immutable_int64(critical),
        center_capable_mask=_immutable_bool(capable),
    )


def _shifted_halfspaces(
    halfspaces: AffineCapabilityHalfspacesV1,
    support: FloatArray,
) -> AffineCapabilityHalfspacesV1:
    return AffineCapabilityHalfspacesV1(
        action_index=halfspaces.action_index,
        regret_tolerance=halfspaces.regret_tolerance,
        active_class_index=_immutable_int64(halfspaces.active_class_index),
        normal=_immutable_float64(halfspaces.normal),
        offset=_immutable_float64(halfspaces.offset - support),
        benchmark_action_index=_immutable_int64(halfspaces.benchmark_action_index),
        witness_hypothesis_index=_immutable_int64(
            halfspaces.witness_hypothesis_index
        ),
    )


def box_robust_center_halfspaces(
    halfspaces: AffineCapabilityHalfspacesV1,
    half_widths: object,
) -> AffineCapabilityHalfspacesV1:
    """Return half-spaces for centers whose complete task box is capable."""

    region = _halfspaces(halfspaces)
    widths = _batched_nonnegative(
        half_widths,
        name="half_widths",
        count=1,
        width=region.task_dimension,
    )[0]
    return _shifted_halfspaces(region, np.abs(region.normal) @ widths)


def ellipsoid_robust_center_halfspaces(
    halfspaces: AffineCapabilityHalfspacesV1,
    generator: object,
) -> AffineCapabilityHalfspacesV1:
    """Return half-spaces for centers of one translated ellipsoid."""

    region = _halfspaces(halfspaces)
    raw = np.asarray(generator)
    if raw.dtype.kind not in "iuf":
        raise ValueError("generator must contain real numeric values")
    matrix = np.ascontiguousarray(raw, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != region.task_dimension
        or matrix.shape[1] < 1
    ):
        raise ValueError(
            "generator must have shape (task_dimension, latent_dimension)"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("generator must be finite")
    return _shifted_halfspaces(
        region,
        np.linalg.norm(region.normal @ matrix, axis=1),
    )


def task_uncertainty_action_mask(
    reports: Sequence[TaskSetCapabilityV1],
) -> BoolArray:
    """Stack one report per contiguous action without resolving overlaps."""

    if isinstance(reports, (str, bytes)) or not isinstance(reports, Sequence):
        raise TypeError("reports must be a sequence of TaskSetCapabilityV1 values")
    if not reports:
        raise ValueError("reports must be nonempty")
    for report in reports:
        if not isinstance(report, TaskSetCapabilityV1):
            raise TypeError("reports must contain TaskSetCapabilityV1 values")
    ordered = tuple(sorted(reports, key=lambda item: item.action_index))
    actions = np.asarray([item.action_index for item in ordered], dtype=np.int64)
    if not np.array_equal(actions, np.arange(actions.size, dtype=np.int64)):
        raise ValueError(
            "report action indices must be unique and contiguous from zero"
        )
    reference = ordered[0]
    for item in ordered[1:]:
        if item.uncertainty_kind != reference.uncertainty_kind:
            raise ValueError("reports must use the same uncertainty kind")
        same_shape = item.task_centers.shape == reference.task_centers.shape
        same_centers = same_shape and np.array_equal(
            item.task_centers,
            reference.task_centers,
        )
        if not same_centers:
            raise ValueError("reports must use identical task centers")
    return _immutable_bool(np.column_stack([item.capable_mask for item in ordered]))


__all__ = [
    "TASK_UNCERTAINTY_CERTIFICATE_CLAIM_BOUNDARY",
    "TASK_UNCERTAINTY_CERTIFICATE_SEMANTICS",
    "TASK_UNCERTAINTY_CERTIFICATE_VERSION",
    "TaskCapabilityMarginV1",
    "TaskSetCapabilityV1",
    "box_robust_center_halfspaces",
    "box_task_set_capability",
    "ellipsoid_robust_center_halfspaces",
    "ellipsoid_task_set_capability",
    "norm_ball_capability_margin",
    "task_uncertainty_action_mask",
]
