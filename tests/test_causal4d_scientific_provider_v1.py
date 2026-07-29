"""Contract tests for the versioned Causal4D scientific facade."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import bayesian_phystwin.causal4d_scientific_provider_v1 as provider

_EXPECTED_CAUSAL4D_EXPORTS = {
    "ARM_TO_ARCHIVE_KEY",
    "DEVELOPMENT_CASES",
    "DynamicDiscrepancyCorrection",
    "FIXED_INITIAL_STD_M",
    "FIXED_INLIER_PRIOR",
    "FIXED_OBSERVATION_STD_M",
    "FIXED_OUTLIER_VARIANCE_MULTIPLIER",
    "FIXED_PROCESS_STD_M",
    "LOCALIZATION_GRAPH_RANK",
    "MANIFEST_FILENAME",
    "MEASUREMENT_FILENAME",
    "PROTOCOL_ID",
    "PropagatedStateBeliefConfig",
    "PropagatedStateCorrection",
    "PropagatedStateSelectionConfig",
    "SCORED_FRAMES",
    "VIRTUAL_SENSING_ARCHIVE_FILENAME",
    "VIRTUAL_SENSING_REPORT_FILENAME",
    "VIRTUAL_SENSING_SEAL_FILENAME",
    "attachment_support_nodes",
    "build_phystwin_track_objective",
    "chamfer_by_frame",
    "closure_confidence",
    "cross_view_residual_audit",
    "decode_limited_state_weights",
    "dynamic_window_source_case",
    "end_effector_origins",
    "estimate_endpoint_velocity_delta",
    "far_graph_observation_error",
    "fit_dimensionless_linearized_correction",
    "git_commit",
    "graph_discrepancy_diagnostics",
    "graph_distance",
    "graph_smoothed_discrepancy_posterior",
    "horizon_summary",
    "infer_propagated_state_belief",
    "initialize_simulator",
    "load_dynamic_discrepancy_correction",
    "load_official_spring_mass_module",
    "load_phystwin_raw_track_map",
    "load_pickle",
    "load_selective_virtual_sensing_protocol",
    "lock_protocol",
    "make_reliability_simulator_class",
    "measurement_target_audit",
    "metric_agreement_audit",
    "metric_summary",
    "modal_state_parameter_fields",
    "normalized_spring_laplacian",
    "object_rest_lengths",
    "official_metrics_by_frame",
    "paired_block_bootstrap",
    "phystwin_physical_object_cluster",
    "prefix_position_velocity_coefficients",
    "released_observation_capability_audit",
    "released_self_collision_for_case",
    "robust_random_walk_endpoint",
    "rollout_restart",
    "scale_coefficients_to_field_limit",
    "scale_posterior_covariance_for_state_limits",
    "score_selective_virtual_sensing_arrays",
    "select_action_only_window",
    "select_propagated_state_update",
    "select_translation_contact_window",
    "selective_case_records",
    "set_simulator_arrays",
    "simulator_runtime",
    "state_numpy",
    "validate_selective_prediction_seal",
    "write_dynamic_discrepancy_correction",
    "write_propagated_state_correction",
}


def test_manifest_and_export_inventory_are_explicit() -> None:
    manifest = provider.causal4d_scientific_provider_manifest()

    assert manifest["schema_version"] == 1
    assert manifest["provider_api"] == (
        "bayesian_phystwin.causal4d_scientific_provider_v1"
    )
    assert _EXPECTED_CAUSAL4D_EXPORTS <= set(provider.__all__)
    assert all(not name.startswith("_") for name in provider.__all__)


def test_facade_import_is_lazy() -> None:
    module_name = "bayesian_phystwin.phystwin_bayesian_anchor"
    sys.modules.pop(module_name, None)
    reloaded = importlib.reload(provider)

    assert module_name not in sys.modules
    assert reloaded.robust_random_walk_endpoint is not None
    assert module_name in sys.modules


def test_rollout_restart_preserves_position_only_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = np.arange(6, dtype=float).reshape(1, 2, 3)

    def fake_resolve(name: str) -> Any:
        assert name == "_rollout_restart_trajectory"
        return lambda *args, **kwargs: (expected, np.zeros_like(expected))

    monkeypatch.setattr(provider, "_resolve", fake_resolve)

    result = provider.rollout_restart(object())

    np.testing.assert_array_equal(result, expected)


def test_unknown_export_fails_closed() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        provider.__getattr__("not_a_public_export")


def test_provider_v1_delegates_historical_scientific_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bayesian_phystwin.causal4d_provider_v1 as replay_provider

    calls: list[tuple[str, str, tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    def fake_delegate(
        module: str,
        name: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        calls.append((module, name, args, kwargs))
        return sentinel

    monkeypatch.setattr(replay_provider, "_delegate", fake_delegate)

    assert replay_provider.load_pickle(Path("legacy.pkl")) is sentinel
    assert calls[-1][:2] == ("phystwin_residual_dynamics", "_load_pickle")

    wrappers = (
        (
            "chamfer_by_frame",
            "phystwin_additional_confirmation",
            "_chamfer_by_frame",
        ),
        ("lock_protocol", "phystwin_confirmatory", "_lock_protocol"),
        ("git_commit", "phystwin_state_injection", "_git_commit"),
        (
            "initialize_simulator",
            "phystwin_state_injection",
            "_initialize_simulator",
        ),
        ("metric_summary", "phystwin_state_injection", "_metric_summary"),
        (
            "released_self_collision_for_case",
            "phystwin_state_injection",
            "_released_self_collision_for_case",
        ),
        ("simulator_runtime", "phystwin_state_injection", "_simulator_runtime"),
        (
            "load_official_spring_mass_module",
            "_phystwin_warp_backend",
            "load_official_spring_mass_module",
        ),
        (
            "make_reliability_simulator_class",
            "_phystwin_warp_backend",
            "make_reliability_simulator_class",
        ),
        (
            "measurement_target_audit",
            "deform360_selective_virtual_sensing_evaluation",
            "_measurement_target_audit",
        ),
        (
            "attachment_support_nodes",
            "phystwin_structural_diagnostic",
            "_attachment_support_nodes",
        ),
        (
            "far_graph_observation_error",
            "phystwin_structural_diagnostic",
            "_far_graph_observation_error",
        ),
        (
            "graph_distance",
            "phystwin_structural_diagnostic",
            "_graph_distance",
        ),
        (
            "horizon_summary",
            "phystwin_structural_diagnostic",
            "_horizon_summary",
        ),
        (
            "object_rest_lengths",
            "phystwin_structural_diagnostic",
            "_object_rest_lengths",
        ),
        (
            "set_simulator_arrays",
            "phystwin_structural_diagnostic",
            "_set_simulator_arrays",
        ),
    )
    for wrapper_name, module_name, target_name in wrappers:
        wrapper = getattr(replay_provider, wrapper_name)
        assert wrapper("argument", option=True) is sentinel
        assert calls[-1][:2] == (module_name, target_name)

    assert "FIXED_PROCESS_STD_M" in replay_provider.__dir__()
    assert replay_provider.__getattr__("FIXED_PROCESS_STD_M") == (
        provider.FIXED_PROCESS_STD_M
    )
    with pytest.raises(AttributeError, match="has no attribute"):
        replay_provider.__getattr__("not_a_compatibility_export")
