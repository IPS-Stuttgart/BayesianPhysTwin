"""Versioned public compatibility facade for Causal4D consumers.

Causal4D should depend on this module instead of underscore-prefixed implementation
details spread across Bayesian-PhysTwin.  The facade keeps imports lazy so the
contract remains available in lightweight installations without Torch, Warp, OpenCV,
or an official PhysTwin checkout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
from importlib import import_module
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import subprocess
from typing import Any, Protocol, runtime_checkable

import numpy as np

PROVIDER_CONTRACT_SCHEMA_VERSION = 1
PROVIDER_API_VERSION = "1.0.0"
PROVIDER_NAME = "bayesian-phystwin"

BASE_CAUSAL4D_PROVIDER_CAPABILITIES = (
    "artifact_checksums",
    "particle_endpoint_position",
    "particle_endpoint_velocity",
    "physical_parameter_particles",
)
CAUSAL4D_PROVIDER_CAPABILITIES = BASE_CAUSAL4D_PROVIDER_CAPABILITIES + (
    "causal_observation_belief_v1",
    "dynamic_discrepancy_artifact_v1",
    "full_covariance_observation_update",
    "official_warp_initial_replay",
    "official_warp_restart_replay",
    "physical_graph_construction",
    "residual_lifting",
)
ARTIFACT_SCHEMA_VERSIONS = {
    "observation_belief": 1,
    "dynamic_discrepancy_correction": 1,
}

# Public facade name -> (implementation module, implementation attribute).
# Existing implementation modules remain authoritative; only these names form the
# compatibility contract.
_ALIAS_TARGETS: dict[str, tuple[str, str]] = {
    # Frozen endpoint-filter settings.
    "FIXED_INITIAL_STD_M": (
        ".phystwin_additional_bayesian_confirmation",
        "FIXED_INITIAL_STD_M",
    ),
    "FIXED_INLIER_PRIOR": (
        ".phystwin_additional_bayesian_confirmation",
        "FIXED_INLIER_PRIOR",
    ),
    "FIXED_OBSERVATION_STD_M": (
        ".phystwin_additional_bayesian_confirmation",
        "FIXED_OBSERVATION_STD_M",
    ),
    "FIXED_OUTLIER_VARIANCE_MULTIPLIER": (
        ".phystwin_additional_bayesian_confirmation",
        "FIXED_OUTLIER_VARIANCE_MULTIPLIER",
    ),
    "FIXED_PROCESS_STD_M": (
        ".phystwin_additional_bayesian_confirmation",
        "FIXED_PROCESS_STD_M",
    ),
    # Physical graph and controller helpers.
    "PhysTwinSpringGraph": (".phystwin_graph", "PhysTwinSpringGraph"),
    "PhysTwinSpringGraphConfig": (".phystwin_graph", "PhysTwinSpringGraphConfig"),
    "build_phystwin_spring_graph": (
        ".phystwin_graph",
        "build_phystwin_spring_graph",
    ),
    "controller_hand_count": (
        ".phystwin_controller_sensitivity",
        "controller_hand_count",
    ),
    "infer_controller_groups": (
        ".phystwin_controller_sensitivity",
        "infer_controller_groups",
    ),
    # Residual loading, validity, lifting, and hashing.
    "build_lift_map": (".phystwin_residual_dynamics", "_lift_map"),
    "clip_residual": (".phystwin_residual_dynamics", "_clip_residual"),
    "lift_residual": (".phystwin_residual_dynamics", "_lift_residual"),
    "load_pickle": (".phystwin_residual_dynamics", "_load_pickle"),
    "sha256_file": (".phystwin_residual_dynamics", "_sha256"),
    "target_validity": (".phystwin_residual_dynamics", "_target_validity"),
    # Official Warp initialization and replay.
    "git_commit": (".phystwin_state_injection", "_git_commit"),
    "initialize_simulator": (".phystwin_state_injection", "_initialize_simulator"),
    "metric_summary": (".phystwin_state_injection", "_metric_summary"),
    "released_self_collision_for_case": (
        ".phystwin_state_injection",
        "_released_self_collision_for_case",
    ),
    "rollout_initial": (".phystwin_state_injection", "_rollout_initial"),
    "rollout_restart": (".phystwin_state_injection", "_rollout_restart"),
    "simulator_runtime": (".phystwin_state_injection", "_simulator_runtime"),
    "state_numpy": (".phystwin_state_injection", "_state_numpy"),
    "estimate_endpoint_velocity_delta": (
        ".phystwin_state_injection",
        "estimate_endpoint_velocity_delta",
    ),
    "load_official_spring_mass_module": (
        "._phystwin_warp_backend",
        "load_official_spring_mass_module",
    ),
    "make_reliability_simulator_class": (
        "._phystwin_warp_backend",
        "make_reliability_simulator_class",
    ),
    "build_phystwin_track_objective": (
        ".phystwin_refit",
        "build_phystwin_track_objective",
    ),
    # Metrics, protocol locks, and robust endpoint inference.
    "chamfer_by_frame": (".phystwin_additional_confirmation", "_chamfer_by_frame"),
    "official_metrics_by_frame": (
        ".phystwin_comparison",
        "official_metrics_by_frame",
    ),
    "paired_block_bootstrap": (".phystwin_comparison", "paired_block_bootstrap"),
    "phystwin_physical_object_cluster": (
        ".phystwin_comparison",
        "phystwin_physical_object_cluster",
    ),
    "DEVELOPMENT_CASES": (".phystwin_confirmatory", "DEVELOPMENT_CASES"),
    "lock_protocol": (".phystwin_confirmatory", "_lock_protocol"),
    "robust_random_walk_endpoint": (
        ".phystwin_bayesian_anchor",
        "robust_random_walk_endpoint",
    ),
    # Graph discrepancy and structural-diagnostic helpers.
    "graph_discrepancy_diagnostics": (
        ".phystwin_graph_discrepancy",
        "graph_discrepancy_diagnostics",
    ),
    "graph_smoothed_discrepancy_posterior": (
        ".phystwin_graph_discrepancy",
        "graph_smoothed_discrepancy_posterior",
    ),
    "normalized_spring_laplacian": (
        ".phystwin_graph_discrepancy",
        "normalized_spring_laplacian",
    ),
    "attachment_support_nodes": (
        ".phystwin_structural_diagnostic",
        "_attachment_support_nodes",
    ),
    "far_graph_observation_error": (
        ".phystwin_structural_diagnostic",
        "_far_graph_observation_error",
    ),
    "graph_distance": (".phystwin_structural_diagnostic", "_graph_distance"),
    "horizon_summary": (".phystwin_structural_diagnostic", "_horizon_summary"),
    "object_rest_lengths": (
        ".phystwin_structural_diagnostic",
        "_object_rest_lengths",
    ),
    "set_simulator_arrays": (
        ".phystwin_structural_diagnostic",
        "_set_simulator_arrays",
    ),
    # Versioned discrepancy and observation-audit artifacts.
    "LOCALIZATION_GRAPH_RANK": (".dynamic_discrepancy", "LOCALIZATION_GRAPH_RANK"),
    "DynamicDiscrepancyCorrection": (
        ".dynamic_discrepancy",
        "DynamicDiscrepancyCorrection",
    ),
    "fit_dimensionless_linearized_correction": (
        ".dynamic_discrepancy",
        "fit_dimensionless_linearized_correction",
    ),
    "load_dynamic_discrepancy_correction": (
        ".dynamic_discrepancy",
        "load_dynamic_discrepancy_correction",
    ),
    "prefix_position_velocity_coefficients": (
        ".dynamic_discrepancy",
        "prefix_position_velocity_coefficients",
    ),
    "scale_coefficients_to_field_limit": (
        ".dynamic_discrepancy",
        "scale_coefficients_to_field_limit",
    ),
    "write_dynamic_discrepancy_correction": (
        ".dynamic_discrepancy",
        "write_dynamic_discrepancy_correction",
    ),
    "cross_view_residual_audit": (
        ".observation_model_audit",
        "cross_view_residual_audit",
    ),
    "metric_agreement_audit": (
        ".observation_model_audit",
        "metric_agreement_audit",
    ),
    "released_observation_capability_audit": (
        ".observation_model_audit",
        "released_observation_capability_audit",
    ),
}


def _validated_json_mapping(values: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(dict(values), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error


def _installed_version() -> str:
    try:
        return importlib_metadata.version(PROVIDER_NAME)
    except importlib_metadata.PackageNotFoundError:
        # Source-tree imports remain deterministic before installation.
        return "0.4.0"


def _repository_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "unavailable"


@dataclass(frozen=True)
class PhysicalBeliefProviderManifest:
    """Provider identity, semantic capabilities, and artifact schema versions."""

    provider_name: str
    provider_version: str
    provider_revision: str
    schema_version: int = PROVIDER_CONTRACT_SCHEMA_VERSION
    capabilities: tuple[str, ...] = CAUSAL4D_PROVIDER_CAPABILITIES
    artifact_schema_versions: Mapping[str, int] = field(
        default_factory=lambda: dict(ARTIFACT_SCHEMA_VERSIONS)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider_name or not self.provider_version or not self.provider_revision:
            raise ValueError("provider identity fields must be nonempty")
        if self.schema_version < 1:
            raise ValueError("schema_version must be positive")
        capabilities = tuple(sorted(map(str, self.capabilities)))
        if not capabilities or any(not value for value in capabilities):
            raise ValueError("capabilities must contain nonempty names")
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("capabilities must be unique")
        artifact_versions = {
            str(name): int(version)
            for name, version in dict(self.artifact_schema_versions).items()
        }
        if not artifact_versions or any(
            not name or version < 1 for name, version in artifact_versions.items()
        ):
            raise ValueError("artifact schema versions must be positive and named")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "artifact_schema_versions",
            dict(sorted(artifact_versions.items())),
        )
        object.__setattr__(
            self,
            "metadata",
            _validated_json_mapping(self.metadata, name="metadata"),
        )

    @property
    def manifest_id(self) -> str:
        descriptor = {
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "provider_revision": self.provider_revision,
            "schema_version": self.schema_version,
            "capabilities": list(self.capabilities),
            "artifact_schema_versions": dict(self.artifact_schema_versions),
            "metadata": dict(self.metadata),
        }
        return hashlib.sha256(
            json.dumps(
                descriptor,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "provider_name": self.provider_name,
            "provider_version": self.provider_version,
            "provider_revision": self.provider_revision,
            "schema_version": self.schema_version,
            "capabilities": list(self.capabilities),
            "artifact_schema_versions": dict(self.artifact_schema_versions),
            "metadata": dict(self.metadata),
        }


def provider_manifest(
    provider_revision: str | None = None,
) -> PhysicalBeliefProviderManifest:
    """Return the installed provider's content-addressed semantic manifest."""

    revision = _repository_revision() if provider_revision is None else str(provider_revision)
    if not revision:
        raise ValueError("provider_revision must be nonempty")
    return PhysicalBeliefProviderManifest(
        provider_name=PROVIDER_NAME,
        provider_version=_installed_version(),
        provider_revision=revision,
        metadata={
            "provider_api_version": PROVIDER_API_VERSION,
            "provider_module": __name__,
            "implementation_imports_are_lazy": True,
        },
    )


@dataclass(frozen=True)
class ReplayTrajectory:
    """Validated position and velocity trajectory returned by a replay provider."""

    position_m: np.ndarray
    velocity_mps: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64).copy()
        velocity = np.asarray(self.velocity_mps, dtype=np.float64).copy()
        if position.ndim != 3 or position.shape[-1] != 3 or len(position) < 1:
            raise ValueError("position_m must have nonempty shape (T, N, 3)")
        if velocity.shape != position.shape:
            raise ValueError("velocity_mps must match position_m")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("replay trajectory arrays must be finite")
        position.setflags(write=False)
        velocity.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_mps", velocity)
        object.__setattr__(
            self,
            "metadata",
            _validated_json_mapping(self.metadata, name="metadata"),
        )

    @property
    def frame_count(self) -> int:
        return int(self.position_m.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.position_m.shape[1])


@runtime_checkable
class PhysTwinReplayProvider(Protocol):
    """Execution boundary used by Causal4D instead of a simulator implementation."""

    @property
    def manifest(self) -> PhysicalBeliefProviderManifest:
        """Return the semantic provider manifest for this implementation."""

    def replay_initial(self, *, frame_count: int) -> ReplayTrajectory:
        """Replay the physical twin from its registered initial state."""

    def replay_restart(
        self,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        start_frame: int,
        stop_frame: int,
    ) -> ReplayTrajectory:
        """Replay from an explicit branch-time state over a half-open frame range."""


def __getattr__(name: str) -> Any:
    target = _ALIAS_TARGETS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(module_name, package=__package__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ALIAS_TARGETS))


__all__ = [
    "ARTIFACT_SCHEMA_VERSIONS",
    "BASE_CAUSAL4D_PROVIDER_CAPABILITIES",
    "CAUSAL4D_PROVIDER_CAPABILITIES",
    "PROVIDER_API_VERSION",
    "PROVIDER_CONTRACT_SCHEMA_VERSION",
    "PROVIDER_NAME",
    "PhysicalBeliefProviderManifest",
    "PhysTwinReplayProvider",
    "ReplayTrajectory",
    "provider_manifest",
    *_ALIAS_TARGETS,
]
