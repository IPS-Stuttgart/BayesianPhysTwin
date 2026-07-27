"""Versioned scientific compatibility surface for Causal4D.

The replay and graph contracts live in ``causal4d_provider_v2`` and
``causal4d_graph_provider_v1``. This module owns the remaining scientific
operations used by Causal4D so downstream production code never imports
unversioned Bayesian-PhysTwin implementation modules directly.

Exports are resolved lazily because several diagnostics depend on optional
simulation or vision packages. Importing this provider therefore remains
lightweight while each requested symbol still resolves to one explicit owned
implementation.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

import numpy as np

CAUSAL4D_SCIENTIFIC_PROVIDER_API_VERSION: Final = 1
CAUSAL4D_SCIENTIFIC_PROVIDER_PACKAGE_VERSION: Final = "0.4.0"

# Public facade name -> (owned implementation module, implementation symbol).
# Private implementation names are intentionally confined to this provider.
_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "ARM_TO_ARCHIVE_KEY": (
        "deform360_selective_virtual_sensing_evaluation",
        "ARM_TO_ARCHIVE_KEY",
    ),
    "DEVELOPMENT_CASES": ("phystwin_confirmatory", "DEVELOPMENT_CASES"),
    "DynamicDiscrepancyCorrection": (
        "dynamic_discrepancy",
        "DynamicDiscrepancyCorrection",
    ),
    "FIXED_INITIAL_STD_M": (
        "phystwin_additional_bayesian_confirmation",
        "FIXED_INITIAL_STD_M",
    ),
    "FIXED_INLIER_PRIOR": (
        "phystwin_additional_bayesian_confirmation",
        "FIXED_INLIER_PRIOR",
    ),
    "FIXED_OBSERVATION_STD_M": (
        "phystwin_additional_bayesian_confirmation",
        "FIXED_OBSERVATION_STD_M",
    ),
    "FIXED_OUTLIER_VARIANCE_MULTIPLIER": (
        "phystwin_additional_bayesian_confirmation",
        "FIXED_OUTLIER_VARIANCE_MULTIPLIER",
    ),
    "FIXED_PROCESS_STD_M": (
        "phystwin_additional_bayesian_confirmation",
        "FIXED_PROCESS_STD_M",
    ),
    "LOCALIZATION_GRAPH_RANK": (
        "dynamic_discrepancy",
        "LOCALIZATION_GRAPH_RANK",
    ),
    "MANIFEST_FILENAME": (
        "deform360_raw_camera_observation",
        "MANIFEST_FILENAME",
    ),
    "MEASUREMENT_FILENAME": (
        "deform360_raw_camera_observation",
        "MEASUREMENT_FILENAME",
    ),
    "PROTOCOL_ID": (
        "deform360_selective_virtual_sensing_protocol",
        "PROTOCOL_ID",
    ),
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
    "SCORED_FRAMES": (
        "deform360_selective_virtual_sensing_evaluation",
        "SCORED_FRAMES",
    ),
    "VIRTUAL_SENSING_ARCHIVE_FILENAME": (
        "deform360_selective_virtual_sensing_artifacts",
        "VIRTUAL_SENSING_ARCHIVE_FILENAME",
    ),
    "VIRTUAL_SENSING_REPORT_FILENAME": (
        "deform360_selective_virtual_sensing_artifacts",
        "VIRTUAL_SENSING_REPORT_FILENAME",
    ),
    "VIRTUAL_SENSING_SEAL_FILENAME": (
        "deform360_selective_virtual_sensing_artifacts",
        "VIRTUAL_SENSING_SEAL_FILENAME",
    ),
    "attachment_support_nodes": (
        "phystwin_structural_diagnostic",
        "_attachment_support_nodes",
    ),
    "build_phystwin_track_objective": (
        "phystwin_refit",
        "build_phystwin_track_objective",
    ),
    "chamfer_by_frame": (
        "phystwin_additional_confirmation",
        "_chamfer_by_frame",
    ),
    "closure_confidence": (
        "deform360_selective_virtual_sensing_staging",
        "closure_confidence",
    ),
    "cross_view_residual_audit": (
        "observation_model_audit",
        "cross_view_residual_audit",
    ),
    "decode_limited_state_weights": (
        "propagated_state_correction",
        "decode_limited_state_weights",
    ),
    "dynamic_window_source_case": (
        "deform360_selective_virtual_sensing_staging",
        "dynamic_window_source_case",
    ),
    "end_effector_origins": (
        "deform360_selective_virtual_sensing_staging",
        "end_effector_origins",
    ),
    "estimate_endpoint_velocity_delta": (
        "phystwin_state_injection",
        "estimate_endpoint_velocity_delta",
    ),
    "far_graph_observation_error": (
        "phystwin_structural_diagnostic",
        "_far_graph_observation_error",
    ),
    "fit_dimensionless_linearized_correction": (
        "dynamic_discrepancy",
        "fit_dimensionless_linearized_correction",
    ),
    "git_commit": ("phystwin_state_injection", "_git_commit"),
    "graph_discrepancy_diagnostics": (
        "phystwin_graph_discrepancy",
        "graph_discrepancy_diagnostics",
    ),
    "graph_distance": ("phystwin_structural_diagnostic", "_graph_distance"),
    "graph_smoothed_discrepancy_posterior": (
        "phystwin_graph_discrepancy",
        "graph_smoothed_discrepancy_posterior",
    ),
    "horizon_summary": ("phystwin_structural_diagnostic", "_horizon_summary"),
    "infer_propagated_state_belief": (
        "propagated_state_belief",
        "infer_propagated_state_belief",
    ),
    "initialize_simulator": (
        "phystwin_state_injection",
        "_initialize_simulator",
    ),
    "load_dynamic_discrepancy_correction": (
        "dynamic_discrepancy",
        "load_dynamic_discrepancy_correction",
    ),
    "load_official_spring_mass_module": (
        "_phystwin_warp_backend",
        "load_official_spring_mass_module",
    ),
    "load_phystwin_raw_track_map": (
        "phystwin_raw_cues",
        "load_phystwin_raw_track_map",
    ),
    "load_pickle": ("phystwin_residual_dynamics", "_load_pickle"),
    "load_selective_virtual_sensing_protocol": (
        "deform360_selective_virtual_sensing_protocol",
        "load_selective_virtual_sensing_protocol",
    ),
    "lock_protocol": ("phystwin_confirmatory", "_lock_protocol"),
    "make_reliability_simulator_class": (
        "_phystwin_warp_backend",
        "make_reliability_simulator_class",
    ),
    "measurement_target_audit": (
        "deform360_selective_virtual_sensing_evaluation",
        "_measurement_target_audit",
    ),
    "metric_agreement_audit": (
        "observation_model_audit",
        "metric_agreement_audit",
    ),
    "metric_summary": ("phystwin_state_injection", "_metric_summary"),
    "modal_state_parameter_fields": (
        "propagated_state_correction",
        "modal_state_parameter_fields",
    ),
    "normalized_spring_laplacian": (
        "phystwin_graph_discrepancy",
        "normalized_spring_laplacian",
    ),
    "object_rest_lengths": (
        "phystwin_structural_diagnostic",
        "_object_rest_lengths",
    ),
    "official_metrics_by_frame": (
        "phystwin_comparison",
        "official_metrics_by_frame",
    ),
    "paired_block_bootstrap": (
        "phystwin_comparison",
        "paired_block_bootstrap",
    ),
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
    "released_self_collision_for_case": (
        "phystwin_state_injection",
        "_released_self_collision_for_case",
    ),
    "robust_random_walk_endpoint": (
        "phystwin_bayesian_anchor",
        "robust_random_walk_endpoint",
    ),
    "scale_coefficients_to_field_limit": (
        "dynamic_discrepancy",
        "scale_coefficients_to_field_limit",
    ),
    "scale_posterior_covariance_for_state_limits": (
        "propagated_state_correction",
        "scale_posterior_covariance_for_state_limits",
    ),
    "score_selective_virtual_sensing_arrays": (
        "deform360_selective_virtual_sensing_evaluation",
        "score_selective_virtual_sensing_arrays",
    ),
    "select_action_only_window": (
        "deform360_selective_virtual_sensing_staging",
        "select_action_only_window",
    ),
    "select_propagated_state_update": (
        "propagated_state_correction",
        "select_propagated_state_update",
    ),
    "select_translation_contact_window": (
        "deform360_selective_virtual_sensing_staging",
        "select_translation_contact_window",
    ),
    "selective_case_records": (
        "deform360_selective_virtual_sensing_artifacts",
        "selective_case_records",
    ),
    "set_simulator_arrays": (
        "phystwin_structural_diagnostic",
        "_set_simulator_arrays",
    ),
    "simulator_runtime": ("phystwin_state_injection", "_simulator_runtime"),
    "state_numpy": ("phystwin.replay", "_state_numpy"),
    "validate_selective_prediction_seal": (
        "deform360_selective_virtual_sensing_artifacts",
        "validate_selective_prediction_seal",
    ),
    "write_dynamic_discrepancy_correction": (
        "dynamic_discrepancy",
        "write_dynamic_discrepancy_correction",
    ),
    "write_propagated_state_correction": (
        "propagated_state_correction",
        "write_propagated_state_correction",
    ),
    # Private target used only by ``rollout_restart``.
    "_rollout_restart_trajectory": (
        "phystwin.replay",
        "_rollout_restart_trajectory",
    ),
}


def _resolve(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = target
    value = getattr(import_module(f"bayesian_phystwin.{module_name}"), symbol_name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    """Resolve one declared scientific export without eager optional imports."""

    return _resolve(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


def rollout_restart(*args: Any, **kwargs: Any) -> np.ndarray:
    """Replay a restart and retain the provider-v1 position-only compatibility."""

    function = _resolve("_rollout_restart_trajectory")
    positions, _ = function(*args, **kwargs)
    return np.asarray(positions)


def causal4d_scientific_provider_manifest() -> dict[str, object]:
    """Describe this explicitly versioned scientific compatibility surface."""

    return {
        "provider_name": "bayesian-phystwin-scientific",
        "provider_version": CAUSAL4D_SCIENTIFIC_PROVIDER_PACKAGE_VERSION,
        "schema_version": CAUSAL4D_SCIENTIFIC_PROVIDER_API_VERSION,
        "provider_api": "bayesian_phystwin.causal4d_scientific_provider_v1",
        "exports": list(__all__),
    }


__all__ = sorted(
    [
        *[name for name in _EXPORTS if not name.startswith("_")],
        "CAUSAL4D_SCIENTIFIC_PROVIDER_API_VERSION",
        "CAUSAL4D_SCIENTIFIC_PROVIDER_PACKAGE_VERSION",
        "causal4d_scientific_provider_manifest",
        "rollout_restart",
    ]
)
