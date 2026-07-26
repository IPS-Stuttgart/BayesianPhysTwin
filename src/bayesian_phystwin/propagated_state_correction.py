"""Guarded artifacts for action-propagated PhysTwin state corrections."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .dynamic_discrepancy import (
    project_prefix_graph_coefficients,
    scale_coefficients_to_field_limit,
)
from .propagated_state_belief import (
    PropagatedStateBeliefConfig,
    PropagatedStateBeliefResult,
    infer_propagated_state_belief,
    propagated_state_readout,
)


PROPAGATED_STATE_CORRECTION_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(value: np.ndarray, *, dtype: object = np.float64) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    _require(np.all(np.isfinite(result)), "array contains non-finite values")
    result.setflags(write=False)
    return result


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_data(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def modal_state_parameter_fields(
    graph_basis: np.ndarray,
    *,
    position_step_m: float,
    velocity_step_mps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mode-coordinate perturbations with fixed maximum node norms.

    Parameter order is all position mode-coordinate pairs followed by all
    velocity mode-coordinate pairs. One unit weight always means exactly one
    declared perturbation step at the mode's maximum-amplitude node.
    """

    basis = np.asarray(graph_basis, dtype=np.float64)
    _require(basis.ndim == 2 and basis.shape[1] >= 1, "graph basis is empty")
    _require(np.all(np.isfinite(basis)), "graph basis contains non-finite values")
    _require(position_step_m > 0.0, "position step must be positive")
    _require(velocity_step_mps > 0.0, "velocity step must be positive")
    maximum = np.max(np.abs(basis), axis=0)
    _require(np.all(maximum > 0.0), "graph basis contains an empty mode")
    position_steps = position_step_m / maximum
    velocity_steps = velocity_step_mps / maximum
    rank = basis.shape[1]
    position_fields = np.zeros((len(basis), 3, 3 * rank), dtype=np.float64)
    velocity_fields = np.zeros_like(position_fields)
    for mode in range(rank):
        for coordinate in range(3):
            parameter = 3 * mode + coordinate
            position_fields[:, coordinate, parameter] = (
                basis[:, mode] * position_steps[mode]
            )
            velocity_fields[:, coordinate, parameter] = (
                basis[:, mode] * velocity_steps[mode]
            )
    return position_fields, velocity_fields, position_steps, velocity_steps


def decode_limited_state_weights(
    state_weights: np.ndarray,
    graph_basis: np.ndarray,
    position_coefficient_steps_m: np.ndarray,
    velocity_coefficient_steps_mps: np.ndarray,
    *,
    maximum_position_update_m: float,
    maximum_velocity_update_mps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode and radially limit position and velocity state updates."""

    basis = np.asarray(graph_basis, dtype=np.float64)
    position_steps = np.asarray(position_coefficient_steps_m, dtype=np.float64)
    velocity_steps = np.asarray(velocity_coefficient_steps_mps, dtype=np.float64)
    weights = np.asarray(state_weights, dtype=np.float64)
    rank = basis.shape[1]
    _require(weights.shape == (6 * rank,), "state weight shape changed")
    _require(position_steps.shape == (rank,), "position step shape changed")
    _require(velocity_steps.shape == (rank,), "velocity step shape changed")
    _require(np.all(np.isfinite(weights)), "state weights are non-finite")
    split = 3 * rank
    position_coefficients = weights[:split].reshape(rank, 3) * position_steps[:, None]
    velocity_coefficients = weights[split:].reshape(rank, 3) * velocity_steps[:, None]
    position_coefficients, position_limit = scale_coefficients_to_field_limit(
        basis,
        position_coefficients,
        maximum_node_norm=maximum_position_update_m,
    )
    velocity_coefficients, velocity_limit = scale_coefficients_to_field_limit(
        basis,
        velocity_coefficients,
        maximum_node_norm=maximum_velocity_update_mps,
    )
    limited_weights = weights.copy()
    limited_weights[:split] *= float(position_limit["radial_scale"])
    limited_weights[split:] *= float(velocity_limit["radial_scale"])
    return (
        limited_weights,
        basis @ position_coefficients,
        basis @ velocity_coefficients,
        {"position": position_limit, "velocity": velocity_limit},
    )


def scale_posterior_covariance_for_state_limits(
    posterior_covariance: np.ndarray,
    *,
    graph_rank: int,
    position_scale: float,
    velocity_scale: float,
    shared_bias_scale: float = 1.0,
) -> np.ndarray:
    """Apply deterministic radial state and bias limits to a covariance."""

    covariance = np.asarray(posterior_covariance, dtype=np.float64)
    dimension = 9 * graph_rank
    _require(covariance.shape == (dimension, dimension), "covariance shape changed")
    _require(np.all(np.isfinite(covariance)), "covariance is non-finite")
    _require(0.0 < position_scale <= 1.0, "position scale must lie in (0, 1]")
    _require(0.0 < velocity_scale <= 1.0, "velocity scale must lie in (0, 1]")
    _require(
        0.0 < shared_bias_scale <= 1.0,
        "shared bias scale must lie in (0, 1]",
    )
    scale = np.ones(dimension, dtype=np.float64)
    scale[: 3 * graph_rank] = position_scale
    scale[3 * graph_rank : 6 * graph_rank] = velocity_scale
    scale[6 * graph_rank :] = shared_bias_scale
    return scale[:, None] * covariance * scale[None]


def _weighted_rmse(
    residual: np.ndarray,
    available: np.ndarray,
    reliability: np.ndarray,
) -> float:
    selected = np.asarray(available, dtype=bool)
    weights = np.asarray(reliability, dtype=np.float64)
    finite = np.all(np.isfinite(residual), axis=2)
    node_weight = np.where(selected & finite, weights, 0.0)
    denominator = 3.0 * float(np.sum(node_weight))
    _require(denominator > 0.0, "validation interval has no observation support")
    return float(
        np.sqrt(np.sum(node_weight[:, :, None] * np.square(residual)) / denominator)
    )


@dataclass(frozen=True)
class PropagatedStateSelectionConfig:
    """Frozen development gate applied inside the allowed response prefix."""

    fit_frame_count: int = 4
    minimum_validation_improvement_fraction: float = 0.05
    minimum_validation_improvement_m: float = 0.00025
    projection_ridge: float = 1e-5
    maximum_position_update_m: float = 0.05
    maximum_velocity_update_mps: float = 0.25
    maximum_shared_bias_m: float = 0.05

    def __post_init__(self) -> None:
        _require(self.fit_frame_count >= 2, "fit frame count must be at least two")
        _require(
            0.0 <= self.minimum_validation_improvement_fraction < 1.0,
            "minimum validation improvement must lie in [0, 1)",
        )
        _require(
            self.minimum_validation_improvement_m >= 0.0,
            "minimum absolute validation improvement must be nonnegative",
        )
        _require(self.projection_ridge > 0.0, "projection ridge must be positive")
        _require(
            self.maximum_position_update_m > 0.0,
            "maximum position update must be positive",
        )
        _require(
            self.maximum_velocity_update_mps > 0.0,
            "maximum velocity update must be positive",
        )
        _require(
            self.maximum_shared_bias_m > 0.0,
            "maximum shared bias must be positive",
        )


@dataclass(frozen=True)
class PropagatedStateSelection:
    """Prefix-only selection result with an exact persistence fallback."""

    accepted: bool
    reason: str
    full_belief: PropagatedStateBeliefResult
    state_weights: np.ndarray
    position_update_m: np.ndarray
    velocity_update_mps: np.ndarray
    shared_bias_coefficients_m: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        weights = _readonly(self.state_weights)
        position = _readonly(self.position_update_m)
        velocity = _readonly(self.velocity_update_mps)
        bias = _readonly(self.shared_bias_coefficients_m)
        _require(weights.ndim == 1, "state weights must be a vector")
        _require(position.ndim == 2 and position.shape[1] == 3, "position changed")
        _require(velocity.shape == position.shape, "velocity shape changed")
        _require(bias.ndim == 2 and bias.shape[1] == 3, "shared bias changed")
        if not self.accepted:
            _require(
                np.array_equal(weights, np.zeros_like(weights)),
                "rejected selection must carry an exact zero state update",
            )
            _require(
                np.array_equal(position, np.zeros_like(position))
                and np.array_equal(velocity, np.zeros_like(velocity)),
                "rejected selection must carry exact zero state fields",
            )
            _require(
                np.array_equal(bias, np.zeros_like(bias)),
                "rejected selection must carry exact zero shared bias",
            )
        object.__setattr__(self, "state_weights", weights)
        object.__setattr__(self, "position_update_m", position)
        object.__setattr__(self, "velocity_update_mps", velocity)
        object.__setattr__(self, "shared_bias_coefficients_m", bias)
        object.__setattr__(
            self, "diagnostics", _json_data(self.diagnostics, "diagnostics")
        )


def select_propagated_state_update(
    innovation_m: np.ndarray,
    available: np.ndarray,
    state_response_at_step_m: np.ndarray,
    observed_graph_basis: np.ndarray,
    full_graph_basis: np.ndarray,
    position_coefficient_steps_m: np.ndarray,
    velocity_coefficient_steps_mps: np.ndarray,
    *,
    prior_reliability: np.ndarray | None = None,
    observation_variance_m2: np.ndarray | None = None,
    belief_config: PropagatedStateBeliefConfig | None = None,
    selection_config: PropagatedStateSelectionConfig | None = None,
) -> PropagatedStateSelection:
    """Select a propagated state update without inspecting forecast frames.

    The state model is fitted on an early subset of the allowed prefix. Its
    predictions on the remaining prefix frames must beat a graph-persistence
    correction fitted at the fit-subset endpoint. A rejection returns exact
    zero state fields; the caller must then reuse the frozen persistence
    trajectory byte-for-byte.
    """

    cfg = selection_config or PropagatedStateSelectionConfig()
    innovation = np.asarray(innovation_m, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    response = np.asarray(state_response_at_step_m, dtype=np.float64)
    observed_basis = np.asarray(observed_graph_basis, dtype=np.float64)
    full_basis = np.asarray(full_graph_basis, dtype=np.float64)
    _require(innovation.ndim == 3 and innovation.shape[2] == 3, "innovation changed")
    _require(mask.shape == innovation.shape[:2], "availability changed")
    _require(response.shape[:3] == innovation.shape, "state response changed")
    _require(observed_basis.shape[0] == innovation.shape[1], "basis coverage changed")
    _require(
        full_basis.shape[1] == observed_basis.shape[1],
        "observed and full graph ranks differ",
    )
    _require(
        cfg.fit_frame_count < len(innovation),
        "selection requires held-out prefix frames",
    )
    if prior_reliability is None:
        reliability = np.ones(mask.shape, dtype=np.float64)
    else:
        reliability = np.asarray(prior_reliability, dtype=np.float64)
        _require(reliability.shape == mask.shape, "reliability shape changed")
    variance = None
    if observation_variance_m2 is not None:
        variance = np.asarray(observation_variance_m2, dtype=np.float64)
        _require(
            variance.shape in {mask.shape, (*mask.shape, 3)},
            "observation variance shape changed",
        )

    fit = slice(0, cfg.fit_frame_count)
    validation = slice(cfg.fit_frame_count, len(innovation))
    fit_variance = None if variance is None else variance[fit]
    fit_belief = infer_propagated_state_belief(
        innovation[fit],
        mask[fit],
        response[fit],
        observed_basis,
        prior_reliability=reliability[fit],
        observation_variance_m2=fit_variance,
        config=belief_config,
    )
    persistence_coefficients = project_prefix_graph_coefficients(
        innovation[fit],
        mask[fit],
        observed_basis,
        ridge=cfg.projection_ridge,
    )[-1]
    persistence_coefficients, persistence_limit = scale_coefficients_to_field_limit(
        observed_basis,
        persistence_coefficients,
        maximum_node_norm=cfg.maximum_position_update_m,
    )
    persistence_prediction = observed_basis @ persistence_coefficients
    persistence_residual = innovation[validation] - persistence_prediction[None]
    persistence_rmse = _weighted_rmse(
        persistence_residual,
        mask[validation],
        reliability[validation],
    )

    zero_weights = np.zeros(response.shape[3], dtype=np.float64)
    zero_position = np.zeros((len(full_basis), 3), dtype=np.float64)
    zero_bias = np.zeros((observed_basis.shape[1], 3), dtype=np.float64)
    diagnostics: dict[str, Any] = {
        "selection_uses_forecast_frames": False,
        "fit_frame_count": cfg.fit_frame_count,
        "validation_frame_count": len(innovation) - cfg.fit_frame_count,
        "persistence_validation_rmse_m": persistence_rmse,
        "minimum_validation_improvement_fraction": (
            cfg.minimum_validation_improvement_fraction
        ),
        "minimum_validation_improvement_m": cfg.minimum_validation_improvement_m,
        "persistence_fit_limit": persistence_limit,
        "development_gate_not_source_calibrated": True,
    }
    if not fit_belief.accepted:
        diagnostics["fit_belief_reason"] = fit_belief.reason
        return PropagatedStateSelection(
            accepted=False,
            reason=f"fit-{fit_belief.reason}",
            full_belief=fit_belief,
            state_weights=zero_weights,
            position_update_m=zero_position,
            velocity_update_mps=zero_position,
            shared_bias_coefficients_m=zero_bias,
            diagnostics=diagnostics,
        )

    limited_fit_weights, _, _, fit_limits = decode_limited_state_weights(
        fit_belief.state_weights,
        full_basis,
        position_coefficient_steps_m,
        velocity_coefficient_steps_mps,
        maximum_position_update_m=cfg.maximum_position_update_m,
        maximum_velocity_update_mps=cfg.maximum_velocity_update_mps,
    )
    limited_fit_bias, fit_bias_limit = scale_coefficients_to_field_limit(
        full_basis,
        fit_belief.shared_bias_coefficients_m,
        maximum_node_norm=cfg.maximum_shared_bias_m,
    )
    joint_prediction = propagated_state_readout(
        response[validation],
        limited_fit_weights,
        observed_basis,
        limited_fit_bias,
    )
    joint_rmse = _weighted_rmse(
        innovation[validation] - joint_prediction,
        mask[validation],
        reliability[validation],
    )
    improvement = 1.0 - joint_rmse / persistence_rmse if persistence_rmse else 0.0
    absolute_improvement = persistence_rmse - joint_rmse
    diagnostics.update(
        {
            "joint_validation_rmse_m": joint_rmse,
            "validation_improvement_fraction": improvement,
            "validation_improvement_m": absolute_improvement,
            "fit_state_limits": fit_limits,
            "fit_shared_bias_limit": fit_bias_limit,
            "fit_belief_diagnostics": fit_belief.diagnostics,
        }
    )
    if (
        improvement < cfg.minimum_validation_improvement_fraction
        or absolute_improvement < cfg.minimum_validation_improvement_m
    ):
        return PropagatedStateSelection(
            accepted=False,
            reason="prefix-validation-regret-guard",
            full_belief=fit_belief,
            state_weights=zero_weights,
            position_update_m=zero_position,
            velocity_update_mps=zero_position,
            shared_bias_coefficients_m=zero_bias,
            diagnostics=diagnostics,
        )

    full_belief = infer_propagated_state_belief(
        innovation,
        mask,
        response,
        observed_basis,
        prior_reliability=reliability,
        observation_variance_m2=variance,
        config=belief_config,
    )
    if not full_belief.accepted:
        diagnostics["full_belief_reason"] = full_belief.reason
        return PropagatedStateSelection(
            accepted=False,
            reason=f"full-{full_belief.reason}",
            full_belief=full_belief,
            state_weights=zero_weights,
            position_update_m=zero_position,
            velocity_update_mps=zero_position,
            shared_bias_coefficients_m=zero_bias,
            diagnostics=diagnostics,
        )
    limited_weights, position, velocity, full_limits = decode_limited_state_weights(
        full_belief.state_weights,
        full_basis,
        position_coefficient_steps_m,
        velocity_coefficient_steps_mps,
        maximum_position_update_m=cfg.maximum_position_update_m,
        maximum_velocity_update_mps=cfg.maximum_velocity_update_mps,
    )
    limited_bias, full_bias_limit = scale_coefficients_to_field_limit(
        full_basis,
        full_belief.shared_bias_coefficients_m,
        maximum_node_norm=cfg.maximum_shared_bias_m,
    )
    diagnostics.update(
        {
            "full_state_limits": full_limits,
            "full_shared_bias_limit": full_bias_limit,
            "full_belief_diagnostics": full_belief.diagnostics,
        }
    )
    return PropagatedStateSelection(
        accepted=True,
        reason="prefix-validation-gate-passed",
        full_belief=full_belief,
        state_weights=limited_weights,
        position_update_m=position,
        velocity_update_mps=velocity,
        shared_bias_coefficients_m=limited_bias,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class PropagatedStateCorrection:
    """Typed record of one guarded action-propagated state update."""

    case_id: str
    graph_basis: np.ndarray
    graph_eigenvalues: np.ndarray
    position_coefficient_steps_m: np.ndarray
    velocity_coefficient_steps_mps: np.ndarray
    state_weights: np.ndarray
    shared_bias_coefficients_m: np.ndarray
    posterior_covariance: np.ndarray
    accepted_state_update: bool
    selection_reason: str
    prefix_frame_start: int
    prefix_frame_stop: int
    fit_frame_stop: int
    information_boundary: Mapping[str, Any]
    source_checksums: Mapping[str, str]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.case_id), "case id is empty")
        basis = _readonly(self.graph_basis)
        eigenvalues = _readonly(self.graph_eigenvalues)
        position_steps = _readonly(self.position_coefficient_steps_m)
        velocity_steps = _readonly(self.velocity_coefficient_steps_mps)
        weights = _readonly(self.state_weights)
        bias = _readonly(self.shared_bias_coefficients_m)
        covariance = _readonly(self.posterior_covariance)
        rank = basis.shape[1]
        _require(basis.ndim == 2 and rank >= 1, "graph basis is empty")
        _require(eigenvalues.shape == (rank,), "graph eigenvalues changed")
        _require(position_steps.shape == (rank,), "position steps changed")
        _require(velocity_steps.shape == (rank,), "velocity steps changed")
        _require(weights.shape == (6 * rank,), "state weights changed")
        _require(bias.shape == (rank, 3), "shared bias changed")
        dimension = len(weights) + 3 * rank
        _require(covariance.shape == (dimension, dimension), "covariance changed")
        _require(
            0 <= self.prefix_frame_start < self.fit_frame_stop < self.prefix_frame_stop,
            "prefix boundaries changed",
        )
        if not self.accepted_state_update:
            _require(
                np.array_equal(weights, np.zeros_like(weights)),
                "rejected artifact must contain exact zero state weights",
            )
        boundary = _json_data(self.information_boundary, "information_boundary")
        _require(
            boundary.get("forecast_frames_used_for_fit_or_selection") is False,
            "artifact violates the forecast boundary",
        )
        checksums = dict(self.source_checksums)
        _require(
            bool(checksums)
            and all(
                len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in checksums.values()
            ),
            "source checksums must be SHA-256 digests",
        )
        for name, value in (
            ("graph_basis", basis),
            ("graph_eigenvalues", eigenvalues),
            ("position_coefficient_steps_m", position_steps),
            ("velocity_coefficient_steps_mps", velocity_steps),
            ("state_weights", weights),
            ("shared_bias_coefficients_m", bias),
            ("posterior_covariance", covariance),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "information_boundary", boundary)
        object.__setattr__(self, "source_checksums", dict(sorted(checksums.items())))
        object.__setattr__(
            self, "diagnostics", _json_data(self.diagnostics, "diagnostics")
        )

    def position_update_m(self) -> np.ndarray:
        _, position, _, _ = decode_limited_state_weights(
            self.state_weights,
            self.graph_basis,
            self.position_coefficient_steps_m,
            self.velocity_coefficient_steps_mps,
            maximum_position_update_m=float("inf"),
            maximum_velocity_update_mps=float("inf"),
        )
        return position

    def velocity_update_mps(self) -> np.ndarray:
        _, _, velocity, _ = decode_limited_state_weights(
            self.state_weights,
            self.graph_basis,
            self.position_coefficient_steps_m,
            self.velocity_coefficient_steps_mps,
            maximum_position_update_m=float("inf"),
            maximum_velocity_update_mps=float("inf"),
        )
        return velocity

    def shared_bias_field_m(self) -> np.ndarray:
        return self.graph_basis @ self.shared_bias_coefficients_m

    def _arrays(self) -> dict[str, np.ndarray]:
        return {
            "graph_basis": self.graph_basis,
            "graph_eigenvalues": self.graph_eigenvalues,
            "position_coefficient_steps_m": self.position_coefficient_steps_m,
            "velocity_coefficient_steps_mps": self.velocity_coefficient_steps_mps,
            "state_weights": self.state_weights,
            "shared_bias_coefficients_m": self.shared_bias_coefficients_m,
            "posterior_covariance": self.posterior_covariance,
        }

    def _scalars(self) -> dict[str, Any]:
        return {
            "schema_version": PROPAGATED_STATE_CORRECTION_SCHEMA_VERSION,
            "artifact_kind": "PropagatedStateCorrection",
            "case_id": self.case_id,
            "accepted_state_update": self.accepted_state_update,
            "selection_reason": self.selection_reason,
            "prefix_frame_start": self.prefix_frame_start,
            "prefix_frame_stop": self.prefix_frame_stop,
            "fit_frame_stop": self.fit_frame_stop,
            "information_boundary": self.information_boundary,
            "source_checksums": self.source_checksums,
            "diagnostics": self.diagnostics,
        }

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                self._scalars(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
        for name, value in sorted(self._arrays().items()):
            digest.update(name.encode("ascii"))
            digest.update(_array_sha256(value).encode("ascii"))
        return digest.hexdigest()


def write_propagated_state_correction(
    path: str | Path,
    correction: PropagatedStateCorrection,
) -> dict[str, str]:
    """Write a checksummed, non-pickled correction artifact."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = target.with_suffix(".json")
    arrays_path = target.with_suffix(".npz")
    np.savez_compressed(arrays_path, **correction._arrays())
    manifest = {
        **correction._scalars(),
        "artifact_id": correction.artifact_id,
        "arrays_path": arrays_path.name,
        "arrays_sha256": _file_sha256(arrays_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "artifact_id": correction.artifact_id,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _file_sha256(manifest_path),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": str(manifest["arrays_sha256"]),
    }


__all__ = [
    "PROPAGATED_STATE_CORRECTION_SCHEMA_VERSION",
    "PropagatedStateCorrection",
    "PropagatedStateSelection",
    "PropagatedStateSelectionConfig",
    "decode_limited_state_weights",
    "modal_state_parameter_fields",
    "scale_posterior_covariance_for_state_limits",
    "select_propagated_state_update",
    "write_propagated_state_correction",
]
