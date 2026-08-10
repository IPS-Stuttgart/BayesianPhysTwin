"""Component and configuration contracts for dynamic endpoint averaging."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Literal, TypeAlias

import numpy as np

from .contracts.fixed_anchor import FixedBayesianAnchorConfigV1

DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION = 2
EvidencePoolingV2: TypeAlias = Literal["per_track", "object"]


class DynamicEndpointNumericalError(FloatingPointError):
    """Raised when dynamic endpoint covariance leaves its finite PSD contract."""


def _real(
    value: object,
    *,
    name: str,
    minimum: float,
    strict: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be a finite real number {relation} {minimum}")
    result = float(value)
    valid = result > minimum if strict else result >= minimum
    if not np.isfinite(result) or not valid:
        relation = ">" if strict else ">="
        raise ValueError(f"{name} must be a finite real number {relation} {minimum}")
    return result


def _mixture_parameters(
    observation_std_m: object,
    inlier_prior: object,
    outlier_variance_multiplier: object,
) -> tuple[float, float, float]:
    observation_std = _real(
        observation_std_m,
        name="observation_std_m",
        minimum=0.0,
        strict=True,
    )
    prior = _real(
        inlier_prior,
        name="inlier_prior",
        minimum=0.0,
        strict=True,
    )
    if prior >= 1.0:
        raise ValueError("inlier_prior must lie strictly between zero and one")
    multiplier = _real(
        outlier_variance_multiplier,
        name="outlier_variance_multiplier",
        minimum=1.0,
        strict=True,
    )
    return observation_std, prior, multiplier


@dataclass(frozen=True, slots=True)
class PersistenceEndpointComponentV2:
    """Exact last-valid-residual mean with robust predictive evidence."""

    process_std_m: float = 0.001
    observation_std_m: float = 0.0025
    initial_std_m: float = 0.01
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0

    def __post_init__(self) -> None:
        process = _real(
            self.process_std_m,
            name="process_std_m",
            minimum=0.0,
        )
        initial = _real(
            self.initial_std_m,
            name="initial_std_m",
            minimum=0.0,
            strict=True,
        )
        observation, prior, multiplier = _mixture_parameters(
            self.observation_std_m,
            self.inlier_prior,
            self.outlier_variance_multiplier,
        )
        object.__setattr__(self, "process_std_m", process)
        object.__setattr__(self, "observation_std_m", observation)
        object.__setattr__(self, "initial_std_m", initial)
        object.__setattr__(self, "inlier_prior", prior)
        object.__setattr__(self, "outlier_variance_multiplier", multiplier)


@dataclass(frozen=True, slots=True)
class DampedTrendEndpointComponentV2:
    """Isotropic robust local-linear-trend discrepancy component."""

    velocity_retention: float = 0.95
    level_process_std_m: float = 0.0005
    velocity_process_std_m_per_step: float = 0.00025
    observation_std_m: float = 0.0025
    initial_level_std_m: float = 0.01
    initial_velocity_std_m_per_step: float = 0.0025
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0

    def __post_init__(self) -> None:
        retention = _real(
            self.velocity_retention,
            name="velocity_retention",
            minimum=0.0,
        )
        if retention > 1.0:
            raise ValueError("velocity_retention must lie in [0, 1]")
        level_process = _real(
            self.level_process_std_m,
            name="level_process_std_m",
            minimum=0.0,
        )
        velocity_process = _real(
            self.velocity_process_std_m_per_step,
            name="velocity_process_std_m_per_step",
            minimum=0.0,
        )
        initial_level = _real(
            self.initial_level_std_m,
            name="initial_level_std_m",
            minimum=0.0,
            strict=True,
        )
        initial_velocity = _real(
            self.initial_velocity_std_m_per_step,
            name="initial_velocity_std_m_per_step",
            minimum=0.0,
            strict=True,
        )
        observation, prior, multiplier = _mixture_parameters(
            self.observation_std_m,
            self.inlier_prior,
            self.outlier_variance_multiplier,
        )
        object.__setattr__(self, "velocity_retention", retention)
        object.__setattr__(self, "level_process_std_m", level_process)
        object.__setattr__(
            self,
            "velocity_process_std_m_per_step",
            velocity_process,
        )
        object.__setattr__(self, "observation_std_m", observation)
        object.__setattr__(self, "initial_level_std_m", initial_level)
        object.__setattr__(
            self,
            "initial_velocity_std_m_per_step",
            initial_velocity,
        )
        object.__setattr__(self, "inlier_prior", prior)
        object.__setattr__(self, "outlier_variance_multiplier", multiplier)


DynamicEndpointComponentV2: TypeAlias = (
    FixedBayesianAnchorConfigV1
    | PersistenceEndpointComponentV2
    | DampedTrendEndpointComponentV2
)


def _default_components() -> tuple[DynamicEndpointComponentV2, ...]:
    return (
        PersistenceEndpointComponentV2(),
        FixedBayesianAnchorConfigV1(
            process_std_m=0.0,
            observation_std_m=0.0025,
        ),
        FixedBayesianAnchorConfigV1(
            process_std_m=0.001,
            observation_std_m=0.0025,
        ),
        FixedBayesianAnchorConfigV1(
            process_std_m=0.005,
            observation_std_m=0.0025,
        ),
        DampedTrendEndpointComponentV2(velocity_retention=0.8),
        DampedTrendEndpointComponentV2(velocity_retention=0.95),
        DampedTrendEndpointComponentV2(velocity_retention=0.995),
    )


def _component_identity(component: DynamicEndpointComponentV2) -> tuple[object, ...]:
    if isinstance(component, FixedBayesianAnchorConfigV1):
        return (
            "local-level",
            component.process_std_m,
            component.observation_std_m,
            component.initial_std_m,
            component.inlier_prior,
            component.outlier_variance_multiplier,
        )
    if isinstance(component, PersistenceEndpointComponentV2):
        return (
            "persistence",
            component.process_std_m,
            component.observation_std_m,
            component.initial_std_m,
            component.inlier_prior,
            component.outlier_variance_multiplier,
        )
    return (
        "damped-trend",
        component.velocity_retention,
        component.level_process_std_m,
        component.velocity_process_std_m_per_step,
        component.observation_std_m,
        component.initial_level_std_m,
        component.initial_velocity_std_m_per_step,
        component.inlier_prior,
        component.outlier_variance_multiplier,
    )


def component_kind(component: DynamicEndpointComponentV2) -> str:
    """Return the stable dynamics-family label for a component."""

    if isinstance(component, FixedBayesianAnchorConfigV1):
        return "local-level"
    if isinstance(component, PersistenceEndpointComponentV2):
        return "persistence"
    if isinstance(component, DampedTrendEndpointComponentV2):
        return "damped-trend"
    raise TypeError("unsupported dynamic endpoint component")


@dataclass(frozen=True, slots=True)
class DynamicEndpointModelAverageConfigV2:
    """Finite causal-prefix model family and evidence-pooling policy."""

    components: tuple[DynamicEndpointComponentV2, ...] = field(
        default_factory=_default_components
    )
    component_prior_probability: tuple[float, ...] | None = None
    balance_component_families: bool = True
    evidence_pooling: EvidencePoolingV2 = "per_track"

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("at least one dynamic endpoint component is required")
        supported = (
            FixedBayesianAnchorConfigV1,
            PersistenceEndpointComponentV2,
            DampedTrendEndpointComponentV2,
        )
        if not all(isinstance(component, supported) for component in components):
            raise TypeError("components contain an unsupported endpoint model")
        identities = {_component_identity(component) for component in components}
        if len(identities) != len(components):
            raise ValueError("dynamic endpoint components must be unique")
        if type(self.balance_component_families) is not bool:
            raise TypeError("balance_component_families must be a bool")
        if self.component_prior_probability is None:
            if self.balance_component_families:
                kinds = tuple(component_kind(component) for component in components)
                family_count = len(set(kinds))
                members = {kind: kinds.count(kind) for kind in set(kinds)}
                prior = np.asarray(
                    [1.0 / (family_count * members[kind]) for kind in kinds],
                    dtype=np.float64,
                )
            else:
                prior = np.full(len(components), 1.0 / len(components))
        else:
            prior = np.asarray(
                self.component_prior_probability,
                dtype=np.float64,
            )
            if prior.shape != (len(components),):
                raise ValueError(
                    "component_prior_probability must match the component count"
                )
            if not np.all(np.isfinite(prior)) or np.any(prior <= 0.0):
                raise ValueError(
                    "component_prior_probability must be finite and positive"
                )
            prior = prior / np.sum(prior)
        if self.evidence_pooling not in {"per_track", "object"}:
            raise ValueError("evidence_pooling must be 'per_track' or 'object'")
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self,
            "component_prior_probability",
            tuple(float(value) for value in prior),
        )

    @property
    def component_kinds(self) -> tuple[str, ...]:
        return tuple(component_kind(component) for component in self.components)


DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2 = DynamicEndpointModelAverageConfigV2()


def _component_matrices(
    component: DynamicEndpointComponentV2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, bool]:
    if isinstance(component, FixedBayesianAnchorConfigV1):
        transition = np.eye(2)
        process = np.diag([component.process_std_m**2, 0.0])
        initial = np.diag([component.initial_std_m**2, 0.0])
        return (
            transition,
            process,
            initial,
            component.observation_std_m**2,
            component.inlier_prior,
            component.outlier_variance_multiplier,
            False,
        )
    if isinstance(component, PersistenceEndpointComponentV2):
        transition = np.eye(2)
        process = np.diag([component.process_std_m**2, 0.0])
        initial = np.diag([component.initial_std_m**2, 0.0])
        return (
            transition,
            process,
            initial,
            component.observation_std_m**2,
            component.inlier_prior,
            component.outlier_variance_multiplier,
            True,
        )
    transition = np.asarray(
        [[1.0, 1.0], [0.0, component.velocity_retention]],
        dtype=np.float64,
    )
    process = np.diag(
        [
            component.level_process_std_m**2,
            component.velocity_process_std_m_per_step**2,
        ]
    )
    initial = np.diag(
        [
            component.initial_level_std_m**2,
            component.initial_velocity_std_m_per_step**2,
        ]
    )
    return (
        transition,
        process,
        initial,
        component.observation_std_m**2,
        component.inlier_prior,
        component.outlier_variance_multiplier,
        False,
    )


__all__ = [
    "DEFAULT_DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONFIG_V2",
    "DYNAMIC_ENDPOINT_MODEL_AVERAGE_CONTRACT_VERSION",
    "DampedTrendEndpointComponentV2",
    "DynamicEndpointComponentV2",
    "DynamicEndpointModelAverageConfigV2",
    "DynamicEndpointNumericalError",
    "EvidencePoolingV2",
    "PersistenceEndpointComponentV2",
    "component_kind",
]
