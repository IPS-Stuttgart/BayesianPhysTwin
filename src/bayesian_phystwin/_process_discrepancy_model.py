"""Typed model, state, and provenance for process discrepancy."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ._process_discrepancy_basis import ProcessDiscrepancyBasisV1
from ._process_discrepancy_common import array_sha256, json_data, readonly

PROCESS_DISCREPANCY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProcessDiscrepancyDynamicsV1:
    """Stable AR(1) prior and observation regularization settings."""

    autoregressive_coefficient: float = 0.95
    stationary_coefficient_std_n: float = 0.1
    graph_roughness_strength: float = 0.0
    observation_noise_floor_n: float = 1e-6
    local_power_prior_std_w: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.autoregressive_coefficient) or abs(
            self.autoregressive_coefficient
        ) >= 1.0:
            raise ValueError("autoregressive_coefficient must have magnitude < 1")
        if (
            not np.isfinite(self.stationary_coefficient_std_n)
            or self.stationary_coefficient_std_n <= 0.0
        ):
            raise ValueError("stationary_coefficient_std_n must be positive")
        if (
            not np.isfinite(self.graph_roughness_strength)
            or self.graph_roughness_strength < 0.0
        ):
            raise ValueError("graph_roughness_strength must be nonnegative")
        if (
            not np.isfinite(self.observation_noise_floor_n)
            or self.observation_noise_floor_n <= 0.0
        ):
            raise ValueError("observation_noise_floor_n must be positive")
        if self.local_power_prior_std_w is not None and (
            not np.isfinite(self.local_power_prior_std_w)
            or self.local_power_prior_std_w <= 0.0
        ):
            raise ValueError("local_power_prior_std_w must be positive when set")

    def stationary_variance_n2(
        self,
        basis: ProcessDiscrepancyBasisV1,
    ) -> np.ndarray:
        denominator = 1.0 + self.graph_roughness_strength * basis.latent_graph_roughness
        return np.square(self.stationary_coefficient_std_n) / denominator


@dataclass(frozen=True)
class ProcessDiscrepancyFitBoundaryV1:
    """Fail-closed declaration for source-frozen model selection."""

    method_freeze_id: str
    split_id: str
    baseline_id: str
    readout_comparator_id: str
    future_outcomes_used_for_fit_or_selection: bool = False
    target_outcomes_used_for_fit_or_selection: bool = False

    def __post_init__(self) -> None:
        identifiers = (
            self.method_freeze_id,
            self.split_id,
            self.baseline_id,
            self.readout_comparator_id,
        )
        if any(
            not isinstance(value, str) or not value.strip() for value in identifiers
        ):
            raise ValueError("fit-boundary identifiers must be nonempty strings")
        if self.future_outcomes_used_for_fit_or_selection:
            raise ValueError("future outcomes must not be used for fit or selection")
        if self.target_outcomes_used_for_fit_or_selection:
            raise ValueError("target outcomes must not be used for fit or selection")


@dataclass(frozen=True)
class ProcessDiscrepancyStateV1:
    """Gaussian posterior over constrained force coefficients."""

    mean_coefficients_n: np.ndarray
    covariance_n2: np.ndarray
    step_index: int = 0

    def __post_init__(self) -> None:
        mean = readonly(self.mean_coefficients_n)
        covariance = readonly(self.covariance_n2)
        if mean.ndim != 1 or len(mean) < 1:
            raise ValueError("mean_coefficients_n must be a nonempty vector")
        if covariance.shape != (len(mean), len(mean)):
            raise ValueError("covariance_n2 must match coefficient dimension")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("state mean and covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10):
            raise ValueError("covariance_n2 must be symmetric")
        try:
            np.linalg.cholesky(0.5 * (covariance + covariance.T))
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance_n2 must be positive definite") from error
        if self.step_index < 0:
            raise ValueError("step_index must be nonnegative")
        object.__setattr__(self, "mean_coefficients_n", mean)
        object.__setattr__(self, "covariance_n2", covariance)


@dataclass(frozen=True)
class ProcessDiscrepancyUpdateV1:
    """One Bayesian conditioning result with audit diagnostics."""

    prior: ProcessDiscrepancyStateV1
    posterior: ProcessDiscrepancyStateV1
    observed_coordinate_count: int
    power_pseudo_observation_count: int
    standardized_residual_rms: float | None
    information_gain_nats: float
    total_mechanical_power_mean_w: float | None
    total_mechanical_power_std_w: float | None
    constraint_residual_l2_n: float


@dataclass(frozen=True)
class ProcessDiscrepancyModelV1:
    """Content-addressable process-discrepancy model definition."""

    basis: ProcessDiscrepancyBasisV1
    dynamics: ProcessDiscrepancyDynamicsV1
    fit_boundary: ProcessDiscrepancyFitBoundaryV1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", json_data(self.metadata, name="metadata"))

    @property
    def model_id(self) -> str:
        scalar_payload = {
            "schema_version": PROCESS_DISCREPANCY_SCHEMA_VERSION,
            "artifact_kind": "ProcessDiscrepancyModelV1",
            "dynamics": {
                "autoregressive_coefficient": self.dynamics.autoregressive_coefficient,
                "stationary_coefficient_std_n": (
                    self.dynamics.stationary_coefficient_std_n
                ),
                "graph_roughness_strength": self.dynamics.graph_roughness_strength,
                "observation_noise_floor_n": self.dynamics.observation_noise_floor_n,
                "local_power_prior_std_w": self.dynamics.local_power_prior_std_w,
            },
            "fit_boundary": {
                "method_freeze_id": self.fit_boundary.method_freeze_id,
                "split_id": self.fit_boundary.split_id,
                "baseline_id": self.fit_boundary.baseline_id,
                "readout_comparator_id": self.fit_boundary.readout_comparator_id,
                "future_outcomes_used_for_fit_or_selection": (
                    self.fit_boundary.future_outcomes_used_for_fit_or_selection
                ),
                "target_outcomes_used_for_fit_or_selection": (
                    self.fit_boundary.target_outcomes_used_for_fit_or_selection
                ),
            },
            "enforce_zero_net_force": self.basis.enforce_zero_net_force,
            "enforce_zero_net_torque": self.basis.enforce_zero_net_torque,
            "metadata": self.metadata,
        }
        digest = hashlib.sha256(
            json.dumps(
                scalar_payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        arrays = {
            "graph_basis": self.basis.graph_basis,
            "graph_eigenvalues": self.basis.graph_eigenvalues,
            "node_positions_m": self.basis.node_positions_m,
            "support_weights": self.basis.support_weights,
            "externally_supported": self.basis.externally_supported,
            "force_operator": self.basis.force_operator,
            "latent_to_graph_coefficients": (
                self.basis.latent_to_graph_coefficients
            ),
            "constraint_matrix": self.basis.constraint_matrix,
            "latent_graph_roughness": self.basis.latent_graph_roughness,
        }
        for name, values in sorted(arrays.items()):
            digest.update(name.encode("ascii"))
            digest.update(array_sha256(values).encode("ascii"))
        return digest.hexdigest()


def initial_process_discrepancy_state(
    model: ProcessDiscrepancyModelV1,
) -> ProcessDiscrepancyStateV1:
    variance = model.dynamics.stationary_variance_n2(model.basis)
    return ProcessDiscrepancyStateV1(
        mean_coefficients_n=np.zeros(model.basis.latent_dimension, dtype=float),
        covariance_n2=np.diag(variance),
        step_index=0,
    )


def predict_process_discrepancy(
    model: ProcessDiscrepancyModelV1,
    state: ProcessDiscrepancyStateV1,
) -> ProcessDiscrepancyStateV1:
    if len(state.mean_coefficients_n) != model.basis.latent_dimension:
        raise ValueError("state does not match process-discrepancy basis")
    coefficient = model.dynamics.autoregressive_coefficient
    stationary_variance = model.dynamics.stationary_variance_n2(model.basis)
    innovation_covariance = (1.0 - coefficient**2) * np.diag(stationary_variance)
    predicted_covariance = (
        coefficient**2 * state.covariance_n2 + innovation_covariance
    )
    predicted_covariance = 0.5 * (
        predicted_covariance + predicted_covariance.T
    )
    return ProcessDiscrepancyStateV1(
        mean_coefficients_n=coefficient * state.mean_coefficients_n,
        covariance_n2=predicted_covariance,
        step_index=state.step_index + 1,
    )

def process_discrepancy_force_moments(
    model: ProcessDiscrepancyModelV1,
    state: ProcessDiscrepancyStateV1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nodal posterior force mean and marginal standard deviation."""

    if len(state.mean_coefficients_n) != model.basis.latent_dimension:
        raise ValueError("state does not match process-discrepancy basis")
    mean = model.basis.force_from_coefficients(state.mean_coefficients_n)
    variance = model.basis.marginal_force_variance_n2(state.covariance_n2)
    return mean, np.sqrt(variance)

