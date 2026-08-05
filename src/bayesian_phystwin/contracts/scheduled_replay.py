"""Immutable contracts for continuous replay of joint contact schedules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Protocol, runtime_checkable

import numpy as np


SCHEDULED_CONTACT_REPLAY_SCHEMA_VERSION = 1
CONTACT_REGIME_SEMANTICS_V1 = (
    "inactive",
    "sticking",
    "slipping",
    "detached",
)


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _identifiers(
    values: object,
    *,
    name: str,
    expected_count: int | None = None,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{name} must be a sequence of identifiers")
    try:
        result = tuple(_identifier(value, name=name) for value in values)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of identifiers") from error
    if expected_count is not None and len(result) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} identifiers")
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _numeric_array(
    values: object,
    *,
    name: str,
    dtype: np.dtype,
    ndim: int | None = None,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuIf":
        raise ValueError(f"{name} must contain only real numeric values")
    array = np.array(raw, dtype=dtype, copy=True, order="C")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if trailing_shape and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(f"{name} must end in shape {trailing_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _float_array(
    values: object,
    *,
    name: str,
    ndim: int | None = None,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    return _numeric_array(
        values,
        name=name,
        dtype=np.dtype(np.float64),
        ndim=ndim,
        trailing_shape=trailing_shape,
    )


def _integer_array(
    values: object,
    *,
    name: str,
    ndim: int | None = None,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain only integer values")
    array = np.array(raw, dtype=np.int64, copy=True, order="C")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


def _probability(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real probability")
    result = float(value)
    lower_ok = result > 0.0 if positive else result >= 0.0
    if not np.isfinite(result) or not lower_ok or result > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _array_record(values: object, *, dtype: str) -> dict[str, object]:
    canonical = np.ascontiguousarray(np.asarray(values, dtype=np.dtype(dtype)))
    return {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest(),
    }


def _content_identity(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contact_parameter(
    values: object,
    *,
    name: str,
    path_count: int,
    contact_count: int,
    frame_count: int,
) -> np.ndarray:
    array = _float_array(values, name=name)
    target = (path_count, contact_count, frame_count)
    broadcast: np.ndarray
    if array.shape == ():
        broadcast = np.full(target, float(array), dtype=np.float64)
    elif array.shape == (contact_count,):
        broadcast = np.broadcast_to(array[None, :, None], target).copy()
    elif array.shape == (path_count, contact_count):
        broadcast = np.broadcast_to(array[:, :, None], target).copy()
    elif array.shape == (contact_count, frame_count):
        broadcast = np.broadcast_to(array[None, :, :], target).copy()
    elif array.shape == target:
        broadcast = np.array(array, dtype=np.float64, copy=True, order="C")
    else:
        raise ValueError(
            f"{name} must be scalar or have shape (G,), (K, G), "
            "(G, T), or (K, G, T)"
        )
    if np.any(broadcast < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    broadcast.setflags(write=False)
    return broadcast


def _variance_array(
    values: object,
    *,
    path_count: int,
    frame_count: int,
    node_count: int,
) -> np.ndarray:
    variance = _float_array(values, name="conditional_variance_m2")
    allowed = {
        (),
        (node_count, 3),
        (path_count, node_count, 3),
        (path_count, frame_count, node_count, 3),
    }
    if variance.shape not in allowed:
        raise ValueError(
            "conditional_variance_m2 must be scalar or have shape (N, 3), "
            "(K, N, 3), or (K, T, N, 3)"
        )
    if np.any(variance < 0.0):
        raise ValueError("conditional_variance_m2 must be nonnegative")
    return variance


@dataclass(frozen=True, slots=True)
class ScheduledContactReplayRequestV1:
    """Complete batch request for continuously simulated joint contact schedules."""

    request_id: str
    schedule_identity: str
    simulator_configuration_id: str
    initial_state_id: str
    contact_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    regime_paths: np.ndarray
    prior_weights: np.ndarray
    retained_prior_mass: float
    group_log_scales: np.ndarray
    controller_points_m: np.ndarray
    position_m: np.ndarray
    velocity_mps: np.ndarray
    frame_times_s: np.ndarray
    contact_node_indices: np.ndarray
    contact_node_weights: np.ndarray
    normal_stiffness_npm: np.ndarray | float
    tangential_stiffness_npm: np.ndarray | float
    friction_coefficient: np.ndarray | float

    def __post_init__(self) -> None:
        request_id = _identifier(self.request_id, name="request_id")
        schedule_identity = _identifier(
            self.schedule_identity,
            name="schedule_identity",
        )
        simulator_configuration_id = _identifier(
            self.simulator_configuration_id,
            name="simulator_configuration_id",
        )
        initial_state_id = _identifier(self.initial_state_id, name="initial_state_id")

        paths = _integer_array(self.regime_paths, name="regime_paths", ndim=3)
        if paths.shape[0] < 1 or paths.shape[1] < 1 or paths.shape[2] < 1:
            raise ValueError("regime_paths must have shape (K>=1, G>=1, T>=1)")
        path_count, contact_count, frame_count = paths.shape
        if np.any(paths < 0) or np.any(
            paths >= len(CONTACT_REGIME_SEMANTICS_V1)
        ):
            raise ValueError("regime_paths contain an unknown contact regime")

        contact_ids = _identifiers(
            self.contact_ids,
            name="contact_ids",
            expected_count=contact_count,
        )
        path_ids = _identifiers(
            self.path_ids,
            name="path_ids",
            expected_count=path_count,
        )

        prior_weights = _float_array(
            self.prior_weights,
            name="prior_weights",
            ndim=1,
        )
        if prior_weights.shape != (path_count,):
            raise ValueError("prior_weights must identify every joint path")
        if np.any(prior_weights < 0.0) or not np.isclose(
            np.sum(prior_weights),
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("prior_weights must be nonnegative and sum to one")

        group_log_scales = _float_array(
            self.group_log_scales,
            name="group_log_scales",
            ndim=1,
        )
        if not len(group_log_scales):
            raise ValueError("group_log_scales must be nonempty")

        controller_points = _float_array(
            self.controller_points_m,
            name="controller_points_m",
            ndim=3,
            trailing_shape=(3,),
        )
        if (
            controller_points.shape[0] != frame_count
            or controller_points.shape[1] < 1
        ):
            raise ValueError("controller_points_m must have shape (T, C>=1, 3)")

        position = _float_array(
            self.position_m,
            name="position_m",
            ndim=2,
            trailing_shape=(3,),
        )
        velocity = _float_array(
            self.velocity_mps,
            name="velocity_mps",
            ndim=2,
            trailing_shape=(3,),
        )
        if position.shape != velocity.shape or position.shape[0] < 1:
            raise ValueError(
                "position_m and velocity_mps must have matching shape (N>=1, 3)"
            )
        node_count = position.shape[0]

        frame_times = _float_array(
            self.frame_times_s,
            name="frame_times_s",
            ndim=1,
        )
        if frame_times.shape != (frame_count,):
            raise ValueError("frame_times_s must identify every schedule frame")
        if frame_count > 1 and np.any(np.diff(frame_times) <= 0.0):
            raise ValueError("frame_times_s must be strictly increasing")

        contact_indices = _integer_array(
            self.contact_node_indices,
            name="contact_node_indices",
            ndim=4,
        )
        contact_weights = _float_array(
            self.contact_node_weights,
            name="contact_node_weights",
            ndim=4,
        )
        if (
            contact_indices.shape != contact_weights.shape
            or contact_indices.shape[:3] != paths.shape
            or contact_indices.shape[3] < 1
        ):
            raise ValueError(
                "contact_node_indices and contact_node_weights must have shape "
                "(K, G, T, M>=1)"
            )
        if np.any(contact_indices < -1) or np.any(contact_indices >= node_count):
            raise ValueError("contact_node_indices lie outside the physical state")
        if np.any(contact_weights < 0.0):
            raise ValueError("contact_node_weights must be nonnegative")
        if np.any(contact_weights[contact_indices < 0] != 0.0):
            raise ValueError("padded contact nodes must have zero weight")

        active = (paths == 1) | (paths == 2)
        weight_sums = np.sum(contact_weights, axis=3)
        valid_counts = np.sum(contact_indices >= 0, axis=3)
        if np.any(valid_counts[active] < 1) or not np.allclose(
            weight_sums[active],
            1.0,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "sticking and slipping contacts require a normalized finite-area "
                "patch"
            )
        if np.any(valid_counts[~active] != 0) or np.any(
            weight_sums[~active] != 0.0
        ):
            raise ValueError(
                "inactive and detached contacts must not apply a contact patch"
            )
        valid_contact_weights = contact_weights[contact_indices >= 0]
        if np.any(valid_contact_weights <= 0.0):
            raise ValueError("every retained contact node must have positive weight")
        for row in contact_indices.reshape(-1, contact_indices.shape[-1]):
            valid = row[row >= 0]
            if len(valid) != len(np.unique(valid)):
                raise ValueError("one contact patch must not repeat a node index")

        normal = _contact_parameter(
            self.normal_stiffness_npm,
            name="normal_stiffness_npm",
            path_count=path_count,
            contact_count=contact_count,
            frame_count=frame_count,
        )
        tangential = _contact_parameter(
            self.tangential_stiffness_npm,
            name="tangential_stiffness_npm",
            path_count=path_count,
            contact_count=contact_count,
            frame_count=frame_count,
        )
        friction = _contact_parameter(
            self.friction_coefficient,
            name="friction_coefficient",
            path_count=path_count,
            contact_count=contact_count,
            frame_count=frame_count,
        )

        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "schedule_identity", schedule_identity)
        object.__setattr__(
            self,
            "simulator_configuration_id",
            simulator_configuration_id,
        )
        object.__setattr__(self, "initial_state_id", initial_state_id)
        object.__setattr__(self, "contact_ids", contact_ids)
        object.__setattr__(self, "path_ids", path_ids)
        object.__setattr__(self, "regime_paths", paths)
        object.__setattr__(self, "prior_weights", prior_weights)
        object.__setattr__(
            self,
            "retained_prior_mass",
            _probability(
                self.retained_prior_mass,
                name="retained_prior_mass",
                positive=True,
            ),
        )
        object.__setattr__(self, "group_log_scales", group_log_scales)
        object.__setattr__(self, "controller_points_m", controller_points)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_mps", velocity)
        object.__setattr__(self, "frame_times_s", frame_times)
        object.__setattr__(self, "contact_node_indices", contact_indices)
        object.__setattr__(self, "contact_node_weights", contact_weights)
        object.__setattr__(self, "normal_stiffness_npm", normal)
        object.__setattr__(self, "tangential_stiffness_npm", tangential)
        object.__setattr__(self, "friction_coefficient", friction)

    @property
    def request_identity(self) -> str:
        """Content identity for physical, schedule, contact, and timebase inputs."""

        return _content_identity(
            {
                "schema_version": SCHEDULED_CONTACT_REPLAY_SCHEMA_VERSION,
                "request_id": self.request_id,
                "schedule_identity": self.schedule_identity,
                "simulator_configuration_id": self.simulator_configuration_id,
                "initial_state_id": self.initial_state_id,
                "contact_ids": list(self.contact_ids),
                "path_ids": list(self.path_ids),
                "retained_prior_mass": self.retained_prior_mass,
                "regime_paths": _array_record(self.regime_paths, dtype="<i8"),
                "prior_weights": _array_record(self.prior_weights, dtype="<f8"),
                "group_log_scales": _array_record(
                    self.group_log_scales,
                    dtype="<f8",
                ),
                "controller_points_m": _array_record(
                    self.controller_points_m,
                    dtype="<f8",
                ),
                "position_m": _array_record(self.position_m, dtype="<f8"),
                "velocity_mps": _array_record(self.velocity_mps, dtype="<f8"),
                "frame_times_s": _array_record(self.frame_times_s, dtype="<f8"),
                "contact_node_indices": _array_record(
                    self.contact_node_indices,
                    dtype="<i8",
                ),
                "contact_node_weights": _array_record(
                    self.contact_node_weights,
                    dtype="<f8",
                ),
                "normal_stiffness_npm": _array_record(
                    self.normal_stiffness_npm,
                    dtype="<f8",
                ),
                "tangential_stiffness_npm": _array_record(
                    self.tangential_stiffness_npm,
                    dtype="<f8",
                ),
                "friction_coefficient": _array_record(
                    self.friction_coefficient,
                    dtype="<f8",
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class ScheduledContactReplayResultV1:
    """Complete continuously simulated trajectory bank for one schedule request."""

    positions_m: np.ndarray
    velocities_mps: np.ndarray
    conditional_variance_m2: np.ndarray | float
    request_id: str
    request_identity: str
    schedule_identity: str
    simulator_configuration_id: str
    initial_state_id: str
    contact_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    regime_paths: np.ndarray
    frame_times_s: np.ndarray
    provider_name: str
    provider_version: str
    provider_revision: str

    def __post_init__(self) -> None:
        positions = _float_array(
            self.positions_m,
            name="positions_m",
            ndim=4,
            trailing_shape=(3,),
        )
        velocities = _float_array(
            self.velocities_mps,
            name="velocities_mps",
            ndim=4,
            trailing_shape=(3,),
        )
        if positions.shape != velocities.shape or any(
            size < 1 for size in positions.shape[:3]
        ):
            raise ValueError(
                "positions_m and velocities_mps must have matching shape "
                "(K>=1, T>=1, N>=1, 3)"
            )
        path_count, frame_count, node_count, _ = positions.shape
        paths = _integer_array(self.regime_paths, name="regime_paths", ndim=3)
        if paths.shape[0] != path_count or paths.shape[2] != frame_count:
            raise ValueError("regime_paths must match the result paths and frames")
        if np.any(paths < 0) or np.any(
            paths >= len(CONTACT_REGIME_SEMANTICS_V1)
        ):
            raise ValueError("regime_paths contain an unknown contact regime")
        contact_count = paths.shape[1]
        contact_ids = _identifiers(
            self.contact_ids,
            name="contact_ids",
            expected_count=contact_count,
        )
        path_ids = _identifiers(
            self.path_ids,
            name="path_ids",
            expected_count=path_count,
        )
        frame_times = _float_array(
            self.frame_times_s,
            name="frame_times_s",
            ndim=1,
        )
        if frame_times.shape != (frame_count,):
            raise ValueError("frame_times_s must identify every result frame")
        if frame_count > 1 and np.any(np.diff(frame_times) <= 0.0):
            raise ValueError("frame_times_s must be strictly increasing")
        variance = _variance_array(
            self.conditional_variance_m2,
            path_count=path_count,
            frame_count=frame_count,
            node_count=node_count,
        )

        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_mps", velocities)
        object.__setattr__(self, "conditional_variance_m2", variance)
        object.__setattr__(
            self,
            "request_id",
            _identifier(self.request_id, name="request_id"),
        )
        object.__setattr__(
            self,
            "request_identity",
            _identifier(self.request_identity, name="request_identity"),
        )
        object.__setattr__(
            self,
            "schedule_identity",
            _identifier(self.schedule_identity, name="schedule_identity"),
        )
        object.__setattr__(
            self,
            "simulator_configuration_id",
            _identifier(
                self.simulator_configuration_id,
                name="simulator_configuration_id",
            ),
        )
        object.__setattr__(
            self,
            "initial_state_id",
            _identifier(self.initial_state_id, name="initial_state_id"),
        )
        object.__setattr__(self, "contact_ids", contact_ids)
        object.__setattr__(self, "path_ids", path_ids)
        object.__setattr__(self, "regime_paths", paths)
        object.__setattr__(self, "frame_times_s", frame_times)
        object.__setattr__(
            self,
            "provider_name",
            _identifier(self.provider_name, name="provider_name"),
        )
        object.__setattr__(
            self,
            "provider_version",
            _identifier(self.provider_version, name="provider_version"),
        )
        object.__setattr__(
            self,
            "provider_revision",
            _identifier(self.provider_revision, name="provider_revision"),
        )

    @classmethod
    def from_request(
        cls,
        request: ScheduledContactReplayRequestV1,
        *,
        positions_m: np.ndarray,
        velocities_mps: np.ndarray,
        conditional_variance_m2: np.ndarray | float,
        provider_name: str,
        provider_version: str,
        provider_revision: str,
    ) -> ScheduledContactReplayResultV1:
        """Bind a complete trajectory bank to the exact validated request."""

        if not isinstance(request, ScheduledContactReplayRequestV1):
            raise TypeError("request must be a ScheduledContactReplayRequestV1")
        return cls(
            positions_m=positions_m,
            velocities_mps=velocities_mps,
            conditional_variance_m2=conditional_variance_m2,
            request_id=request.request_id,
            request_identity=request.request_identity,
            schedule_identity=request.schedule_identity,
            simulator_configuration_id=request.simulator_configuration_id,
            initial_state_id=request.initial_state_id,
            contact_ids=request.contact_ids,
            path_ids=request.path_ids,
            regime_paths=request.regime_paths,
            frame_times_s=request.frame_times_s,
            provider_name=provider_name,
            provider_version=provider_version,
            provider_revision=provider_revision,
        )

    @property
    def result_identity(self) -> str:
        """Content identity for the request binding and complete replay output."""

        return _content_identity(
            {
                "schema_version": SCHEDULED_CONTACT_REPLAY_SCHEMA_VERSION,
                "request_id": self.request_id,
                "request_identity": self.request_identity,
                "schedule_identity": self.schedule_identity,
                "simulator_configuration_id": self.simulator_configuration_id,
                "initial_state_id": self.initial_state_id,
                "contact_ids": list(self.contact_ids),
                "path_ids": list(self.path_ids),
                "provider_name": self.provider_name,
                "provider_version": self.provider_version,
                "provider_revision": self.provider_revision,
                "regime_paths": _array_record(self.regime_paths, dtype="<i8"),
                "frame_times_s": _array_record(self.frame_times_s, dtype="<f8"),
                "positions_m": _array_record(self.positions_m, dtype="<f8"),
                "velocities_mps": _array_record(self.velocities_mps, dtype="<f8"),
                "conditional_variance_m2": _array_record(
                    self.conditional_variance_m2,
                    dtype="<f8",
                ),
            }
        )

    @property
    def replay_result_identity(self) -> str:
        """Alias used by Causal4D rollout-bank contracts."""

        return self.result_identity


def validate_scheduled_contact_replay_result(
    request: ScheduledContactReplayRequestV1,
    result: ScheduledContactReplayResultV1,
) -> ScheduledContactReplayResultV1:
    """Require a replay result to bind the complete request without semantic drift."""

    if not isinstance(request, ScheduledContactReplayRequestV1):
        raise TypeError("request must be a ScheduledContactReplayRequestV1")
    if not isinstance(result, ScheduledContactReplayResultV1):
        raise TypeError("result must be a ScheduledContactReplayResultV1")
    for name in (
        "request_id",
        "schedule_identity",
        "simulator_configuration_id",
        "initial_state_id",
        "contact_ids",
        "path_ids",
    ):
        if getattr(result, name) != getattr(request, name):
            raise ValueError(
                f"scheduled replay result {name} does not match the request"
            )
    if result.request_identity != request.request_identity:
        raise ValueError("scheduled replay result request identity does not match")
    if not np.array_equal(result.regime_paths, request.regime_paths):
        raise ValueError("scheduled replay result regime paths do not match the request")
    if not np.array_equal(result.frame_times_s, request.frame_times_s):
        raise ValueError("scheduled replay result timebase does not match the request")
    return result


@runtime_checkable
class ScheduledContactReplayProviderV1(Protocol):
    """Provider for one continuous physical rollout per complete joint schedule."""

    @property
    def simulator_configuration_id(self) -> str:
        """Identifier for the fixed physical simulator configuration."""

        ...

    @property
    def provider_revision(self) -> str:
        """Immutable source or installed-distribution revision."""

        ...

    def replay_scheduled_contacts(
        self,
        request: ScheduledContactReplayRequestV1,
    ) -> ScheduledContactReplayResultV1:
        """Replay every requested schedule without segment splicing or omissions."""

        ...

    def close(self) -> None:
        """Release simulator resources."""

        ...


__all__ = [
    "CONTACT_REGIME_SEMANTICS_V1",
    "SCHEDULED_CONTACT_REPLAY_SCHEMA_VERSION",
    "ScheduledContactReplayProviderV1",
    "ScheduledContactReplayRequestV1",
    "ScheduledContactReplayResultV1",
    "validate_scheduled_contact_replay_result",
]
