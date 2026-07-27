"""Stable Bayesian-PhysTwin compatibility surface for Causal4D.

The module preserves the provider-v1 import path while forwarding owned replay,
geometry, hashing, and metadata operations to stable core modules.  Historical
diagnostic helpers remain isolated here for frozen Causal4D consumers.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from .contracts.provider import (
    installed_distribution_revision,
    installed_distribution_version,
)
from .contracts.replay import PhysTwinReplayProviderV1 as PhysTwinReplayProvider
from .phystwin.artifacts import sha256_file
from .phystwin.geometry import build_lift_map, lift_residual, target_validity
from .phystwin.replay import (
    _rollout_initial_trajectory,
    _rollout_restart_trajectory,
    _state_numpy as _owned_state_numpy,
)

CAUSAL4D_PROVIDER_API_VERSION = 1
CAUSAL4D_PROVIDER_PACKAGE_VERSION = "0.4.0"
CAUSAL4D_PROVIDER_CAPABILITIES = (
    "artifact_checksums",
    "bayesian_anchor_endpoint",
    "diagnostic_comparison",
    "diagnostic_compatibility",
    "diagnostic_discrepancy",
    "diagnostic_observation_audit",
    "diagnostic_propagated_state",
    "diagnostic_rest_geometry",
    "particle_endpoint_position",
    "particle_endpoint_velocity",
    "physical_parameter_particles",
    "phystwin_replay",
    "residual_lifting",
    "target_validity",
)
CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS = {
    "GraphBelief": 1,
    "TwinBelief": 1,
}


@dataclass(frozen=True)
class FixedBayesianAnchorConfigV1:
    """Immutable fixed-anchor hyperparameters consumed by Causal4D."""

    process_std_m: float = 0.005
    observation_std_m: float = 0.001
    initial_std_m: float = 0.01
    inlier_prior: float = 0.95
    outlier_variance_multiplier: float = 100.0

    def __post_init__(self) -> None:
        if self.process_std_m < 0.0:
            raise ValueError("process_std_m must be nonnegative")
        if self.observation_std_m <= 0.0 or self.initial_std_m <= 0.0:
            raise ValueError("observation and initial scales must be positive")
        if not 0.0 < self.inlier_prior < 1.0:
            raise ValueError("inlier_prior must lie in (0, 1)")
        if self.outlier_variance_multiplier <= 1.0:
            raise ValueError("outlier_variance_multiplier must exceed one")


FIXED_BAYESIAN_ANCHOR_CONFIG_V1 = FixedBayesianAnchorConfigV1()
FIXED_PROCESS_STD_M = FIXED_BAYESIAN_ANCHOR_CONFIG_V1.process_std_m
FIXED_OBSERVATION_STD_M = FIXED_BAYESIAN_ANCHOR_CONFIG_V1.observation_std_m
FIXED_INITIAL_STD_M = FIXED_BAYESIAN_ANCHOR_CONFIG_V1.initial_std_m
FIXED_INLIER_PRIOR = FIXED_BAYESIAN_ANCHOR_CONFIG_V1.inlier_prior
FIXED_OUTLIER_VARIANCE_MULTIPLIER = (
    FIXED_BAYESIAN_ANCHOR_CONFIG_V1.outlier_variance_multiplier
)


def _readonly_array(values: np.ndarray, *, dtype: Any) -> np.ndarray:
    array = np.asarray(values, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BayesianAnchorEndpointV1:
    """Immutable robust random-walk endpoint posterior."""

    mean: np.ndarray
    variance: np.ndarray
    final_inlier_probability: np.ndarray
    update_count: np.ndarray

    def __post_init__(self) -> None:
        mean = _readonly_array(self.mean, dtype=float)
        variance = _readonly_array(self.variance, dtype=float)
        probability = _readonly_array(self.final_inlier_probability, dtype=float)
        update_count = _readonly_array(self.update_count, dtype=np.int64)
        if mean.ndim != 2 or mean.shape[1] != 3:
            raise ValueError("mean must have shape (N, 3)")
        expected = (len(mean),)
        if variance.shape != expected or probability.shape != expected:
            raise ValueError("variance and inlier probability must have shape (N,)")
        if update_count.shape != expected:
            raise ValueError("update_count must have shape (N,)")
        if (
            not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(variance))
            or not np.all(np.isfinite(probability))
        ):
            raise ValueError("endpoint posterior arrays must be finite")
        if np.any(variance < 0.0):
            raise ValueError("endpoint variances must be nonnegative")
        if np.any((probability < 0.0) | (probability > 1.0)):
            raise ValueError("inlier probabilities must lie in [0, 1]")
        if np.any(update_count < 0):
            raise ValueError("update counts must be nonnegative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "final_inlier_probability", probability)
        object.__setattr__(self, "update_count", update_count)


_LAZY_DIAGNOSTIC_EXPORTS: dict[str, tuple[str, str]] = {
    "DEVELOPMENT_CASES": ("phystwin_confirmatory", "DEVELOPMENT_CASES"),
    "DynamicDiscrepancyCorrection": (
        "dynamic_discrepancy",
        "DynamicDiscrepancyCorrection",
    ),
    "LOCALIZATION_GRAPH_RANK": ("dynamic_discrepancy", "LOCALIZATION_GRAPH_RANK"),
    "PropagatedStateBeliefConfig": (
        "propagated_state_belief",
        "PropagatedStateBeliefConfig",
    ),
    "PropagatedStateCorrection": (
        "propagated_state_correction",
        "PropagatedStateCorrection",
    ),
    "PropagatedStateSelectionConfig": (
        "propagated_state_correction",
        "PropagatedStateSelectionConfig",
    ),
    "cross_view_residual_audit": (
        "observation_model_audit",
        "cross_view_residual_audit",
    ),
    "decode_limited_state_weights": (
        "propagated_state_correction",
        "decode_limited_state_weights",
    ),
    "estimate_endpoint_velocity_delta": (
        "phystwin_state_injection",
        "estimate_endpoint_velocity_delta",
    ),
    "fit_dimensionless_linearized_correction": (
        "dynamic_discrepancy",
        "fit_dimensionless_linearized_correction",
    ),
    "graph_discrepancy_diagnostics": (
        "phystwin_graph_discrepancy",
        "graph_discrepancy_diagnostics",
    ),
    "graph_smoothed_discrepancy_posterior": (
        "phystwin_graph_discrepancy",
        "graph_smoothed_discrepancy_posterior",
    ),
    "infer_propagated_state_belief": (
        "propagated_state_belief",
        "infer_propagated_state_belief",
    ),
    "load_dynamic_discrepancy_correction": (
        "dynamic_discrepancy",
        "load_dynamic_discrepancy_correction",
    ),
    "metric_agreement_audit": (
        "observation_model_audit",
        "metric_agreement_audit",
    ),
    "modal_state_parameter_fields": (
        "propagated_state_correction",
        "modal_state_parameter_fields",
    ),
    "normalized_spring_laplacian": (
        "phystwin_graph_discrepancy",
        "normalized_spring_laplacian",
    ),
    "official_metrics_by_frame": (
        "phystwin_comparison",
        "official_metrics_by_frame",
    ),
    "paired_block_bootstrap": ("phystwin_comparison", "paired_block_bootstrap"),
    "phystwin_physical_object_cluster": (
        "phystwin_comparison",
        "phystwin_physical_object_cluster",
    ),
    "prefix_position_velocity_coefficients": (
        "dynamic_discrepancy",
        "prefix_position_velocity_coefficients",
    ),
    "released_observation_capability_audit": (
        "observation_model_audit",
        "released_observation_capability_audit",
    ),
    "scale_coefficients_to_field_limit": (
        "dynamic_discrepancy",
        "scale_coefficients_to_field_limit",
    ),
    "scale_posterior_covariance_for_state_limits": (
        "propagated_state_correction",
        "scale_posterior_covariance_for_state_limits",
    ),
    "select_propagated_state_update": (
        "propagated_state_correction",
        "select_propagated_state_update",
    ),
    "write_dynamic_discrepancy_correction": (
        "dynamic_discrepancy",
        "write_dynamic_discrepancy_correction",
    ),
    "write_propagated_state_correction": (
        "propagated_state_correction",
        "write_propagated_state_correction",
    ),
}


def causal4d_provider_manifest(
    *,
    provider_revision: str | None = None,
) -> dict[str, object]:
    """Return the versioned provider descriptor consumed by Causal4D.

    ``provider_revision`` should be the exact Git revision for frozen experiments.
    Normal editable or package-index installs may omit it; their compatibility is
    decided from ``provider_version`` and the API/schema versions instead.
    """

    revision = (
        provider_revision
        or os.environ.get("BAYESIAN_PHYSTWIN_REVISION")
        or installed_distribution_revision("bayesian-phystwin")
        or "unversioned-install"
    )
    return {
        "provider_name": "bayesian-phystwin",
        "provider_version": installed_distribution_version(
            "bayesian-phystwin",
            fallback=CAUSAL4D_PROVIDER_PACKAGE_VERSION,
        ),
        "provider_revision": revision,
        "schema_version": CAUSAL4D_PROVIDER_API_VERSION,
        "capabilities": list(CAUSAL4D_PROVIDER_CAPABILITIES),
        "artifact_schema_versions": dict(CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS),
        "metadata": {
            "provider_api": "bayesian_phystwin.causal4d_provider_v1",
            "provider_api_version": CAUSAL4D_PROVIDER_API_VERSION,
        },
    }


class OfficialPhysTwinReplayProvider:
    """Adapter around the released Warp simulator implementing provider API v1."""

    def __init__(self, simulator: Any, torch: Any, wp: Any, *, device: str) -> None:
        self._simulator = simulator
        self._torch = torch
        self._wp = wp
        self._device = str(device)
        self._closed = False

    @property
    def device(self) -> str:
        return self._device

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PhysTwin replay provider is closed")

    def set_group_log_scales(self, values: np.ndarray) -> None:
        self._require_open()
        array = np.asarray(values, dtype=np.float32)
        target = self._simulator.group_log_scale_tensor
        expected = tuple(int(value) for value in target.shape)
        if array.shape != expected:
            raise ValueError(f"group log-scales must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("group log-scales must be finite")
        with self._torch.no_grad():
            target.copy_(
                self._torch.as_tensor(
                    array,
                    dtype=self._torch.float32,
                    device=self._device,
                )
            )
        self._wp.synchronize()

    def set_controller_points(self, values: np.ndarray) -> None:
        self._require_open()
        array = np.asarray(values, dtype=np.float32)
        current = self._simulator.controller_points
        expected = tuple(int(value) for value in current.shape)
        if array.shape != expected:
            raise ValueError(f"controller points must have shape {expected}")
        if not np.all(np.isfinite(array)):
            raise ValueError("controller points must be finite")
        self._simulator.controller_points = self._torch.as_tensor(
            array,
            dtype=self._torch.float32,
            device=self._device,
        ).contiguous()
        self._wp.synchronize()

    def replay_initial(self, *, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
        self._require_open()
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        positions, velocities = _rollout_initial(
            self._simulator,
            self._wp,
            frame_count=frame_count,
        )
        return np.asarray(positions), np.asarray(velocities)

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> np.ndarray:
        self._require_open()
        position = np.asarray(position_m, dtype=np.float32)
        velocity = np.asarray(velocity_mps, dtype=np.float32)
        if (
            position.ndim != 2
            or position.shape[1] != 3
            or velocity.shape != position.shape
        ):
            raise ValueError("restart position and velocity must have shape (N, 3)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("restart state must be finite")
        if not 0 <= start_frame < stop_frame:
            raise ValueError("restart frame interval must be nonempty")
        return np.asarray(
            rollout_restart(
                self._simulator,
                self._torch,
                self._wp,
                position,
                velocity,
                start_frame=start_frame,
                stop_frame=stop_frame,
                device=self._device,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._simulator = None
        gc.collect()
        cuda = getattr(self._torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "empty_cache"):
            cuda.empty_cache()

    def __enter__(self) -> OfficialPhysTwinReplayProvider:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def create_official_replay_provider(
    official_repo: str | Path,
    data: Mapping[str, object],
    optimal: Mapping[str, object],
    checkpoint_path: str | Path,
    graph: Any,
    *,
    num_surface_points: int,
    original_count: int,
    dt: float,
    num_substeps: int,
    self_collision: bool,
    deterministic_spring_forces: bool = False,
    spring_parameterization: str = "dense",
    device: str,
) -> OfficialPhysTwinReplayProvider:
    """Construct the official replay adapter without exposing simulator internals."""

    simulator, torch, wp, _ = initialize_simulator(
        official_repo,
        dict(data),
        dict(optimal),
        checkpoint_path,
        graph,
        num_surface_points=num_surface_points,
        original_count=original_count,
        dt=dt,
        num_substeps=num_substeps,
        self_collision=self_collision,
        deterministic_spring_forces=deterministic_spring_forces,
        spring_parameterization=spring_parameterization,
        device=device,
    )
    return OfficialPhysTwinReplayProvider(simulator, torch, wp, device=device)


def _delegate(module: str, name: str, *args: Any, **kwargs: Any) -> Any:
    function = getattr(import_module(f"bayesian_phystwin.{module}"), name)
    return function(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Resolve explicitly registered diagnostic compatibility names lazily."""

    target = _LAZY_DIAGNOSTIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(f"bayesian_phystwin.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_DIAGNOSTIC_EXPORTS))


def robust_random_walk_endpoint(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    process_variance: float,
    observation_variance: float,
    initial_variance: float,
    inlier_prior: float,
    outlier_variance_multiplier: float,
) -> BayesianAnchorEndpointV1:
    """Run the historical robust endpoint filter behind an immutable DTO."""

    posterior = _delegate(
        "phystwin_bayesian_anchor",
        "robust_random_walk_endpoint",
        residual,
        valid,
        end_frame=end_frame,
        process_variance=process_variance,
        observation_variance=observation_variance,
        initial_variance=initial_variance,
        inlier_prior=inlier_prior,
        outlier_variance_multiplier=outlier_variance_multiplier,
    )
    return BayesianAnchorEndpointV1(
        mean=posterior.mean,
        variance=posterior.variance,
        final_inlier_probability=posterior.final_inlier_probability,
        update_count=posterior.update_count,
    )


def infer_fixed_bayesian_anchor_endpoint(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    end_frame: int,
    config: FixedBayesianAnchorConfigV1 = FIXED_BAYESIAN_ANCHOR_CONFIG_V1,
) -> BayesianAnchorEndpointV1:
    """Run the frozen Bayesian-anchor endpoint contract."""

    return robust_random_walk_endpoint(
        residual,
        valid,
        end_frame=end_frame,
        process_variance=config.process_std_m**2,
        observation_variance=config.observation_std_m**2,
        initial_variance=config.initial_std_m**2,
        inlier_prior=config.inlier_prior,
        outlier_variance_multiplier=config.outlier_variance_multiplier,
    )


# Legacy artifact compatibility. New code should use the hash-locked artifact API.
def load_pickle(path: str | Path) -> Any:
    return _delegate("phystwin_residual_dynamics", "_load_pickle", path)


# Publicly named compatibility operations for advanced Causal4D diagnostics.
def chamfer_by_frame(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_additional_confirmation", "_chamfer_by_frame", *args, **kwargs
    )


def lock_protocol(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_confirmatory", "_lock_protocol", *args, **kwargs)


def git_commit(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_git_commit", *args, **kwargs)


def initialize_simulator(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_state_injection", "_initialize_simulator", *args, **kwargs
    )


def metric_summary(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_metric_summary", *args, **kwargs)


def released_self_collision_for_case(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_state_injection",
        "_released_self_collision_for_case",
        *args,
        **kwargs,
    )


def _rollout_initial(*args: Any, **kwargs: Any) -> Any:
    return _rollout_initial_trajectory(*args, **kwargs)


def rollout_restart(*args: Any, **kwargs: Any) -> Any:
    positions, _ = _rollout_restart_trajectory(*args, **kwargs)
    return positions


def simulator_runtime(*args: Any, **kwargs: Any) -> Any:
    return _delegate("phystwin_state_injection", "_simulator_runtime", *args, **kwargs)


def state_numpy(*args: Any, **kwargs: Any) -> Any:
    return _owned_state_numpy(*args, **kwargs)


def load_official_spring_mass_module(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "_phystwin_warp_backend",
        "load_official_spring_mass_module",
        *args,
        **kwargs,
    )


def make_reliability_simulator_class(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "_phystwin_warp_backend",
        "make_reliability_simulator_class",
        *args,
        **kwargs,
    )


def measurement_target_audit(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "deform360_selective_virtual_sensing_evaluation",
        "_measurement_target_audit",
        *args,
        **kwargs,
    )


def attachment_support_nodes(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic",
        "_attachment_support_nodes",
        *args,
        **kwargs,
    )


def far_graph_observation_error(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic",
        "_far_graph_observation_error",
        *args,
        **kwargs,
    )


def graph_distance(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic", "_graph_distance", *args, **kwargs
    )


def horizon_summary(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic", "_horizon_summary", *args, **kwargs
    )


def object_rest_lengths(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic",
        "_object_rest_lengths",
        *args,
        **kwargs,
    )


def set_simulator_arrays(*args: Any, **kwargs: Any) -> Any:
    return _delegate(
        "phystwin_structural_diagnostic",
        "_set_simulator_arrays",
        *args,
        **kwargs,
    )


__all__ = [
    "CAUSAL4D_ARTIFACT_SCHEMA_VERSIONS",
    "CAUSAL4D_PROVIDER_API_VERSION",
    "CAUSAL4D_PROVIDER_CAPABILITIES",
    "CAUSAL4D_PROVIDER_PACKAGE_VERSION",
    "BayesianAnchorEndpointV1",
    "FIXED_BAYESIAN_ANCHOR_CONFIG_V1",
    "FIXED_INITIAL_STD_M",
    "FIXED_INLIER_PRIOR",
    "FIXED_OBSERVATION_STD_M",
    "FIXED_OUTLIER_VARIANCE_MULTIPLIER",
    "FIXED_PROCESS_STD_M",
    "FixedBayesianAnchorConfigV1",
    "OfficialPhysTwinReplayProvider",
    "PhysTwinReplayProvider",
    "attachment_support_nodes",
    "build_lift_map",
    "causal4d_provider_manifest",
    "chamfer_by_frame",
    "create_official_replay_provider",
    "far_graph_observation_error",
    "git_commit",
    "graph_distance",
    "horizon_summary",
    "infer_fixed_bayesian_anchor_endpoint",
    "initialize_simulator",
    "lift_residual",
    "load_official_spring_mass_module",
    "load_pickle",
    "lock_protocol",
    "make_reliability_simulator_class",
    "measurement_target_audit",
    "metric_summary",
    "object_rest_lengths",
    "released_self_collision_for_case",
    "robust_random_walk_endpoint",
    "rollout_restart",
    "set_simulator_arrays",
    "sha256_file",
    "simulator_runtime",
    "state_numpy",
    "target_validity",
]

__all__ += sorted(_LAZY_DIAGNOSTIC_EXPORTS)
