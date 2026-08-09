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


def _dynamic_discrepancy_correction() -> provider.DynamicDiscrepancyCorrection:
    node_count = 12
    basis_seed = np.column_stack(
        (
            np.ones(node_count),
            np.linspace(-1.0, 1.0, node_count),
            np.cos(np.linspace(0.0, np.pi, node_count)),
            np.sin(np.linspace(0.0, 2.0 * np.pi, node_count)),
        )
    )
    basis = np.linalg.qr(basis_seed, mode="reduced")[0]
    coefficients = np.arange(12, dtype=float).reshape(4, 3) * 1e-4
    return provider.DynamicDiscrepancyCorrection(
        case_id="stable_coverage_case",
        graph_basis=basis,
        graph_eigenvalues=np.asarray((0.0, 0.1, 0.2, 0.3)),
        position_coefficients_m=coefficients,
        velocity_coefficients_mps=2.0 * coefficients,
        generalized_force_coefficients_n=3.0 * coefficients,
        structural_coefficients_m=4.0 * coefficients,
        prefix_frame_start=19,
        prefix_frame_stop=26,
        frame_dt_s=0.05,
        information_boundary={
            "o_plus_prefix_frames": 6,
            "future_frames_used_for_fit_or_selection": False,
            "manual_tracks_used_for_fit_or_selection": False,
            "graph_rank": 4,
        },
        regularization={"dimensionless_ridge": 1e-4},
        source_checksums={"source": "a" * 64},
    )


def test_dynamic_discrepancy_stable_paths_and_roundtrip(tmp_path: Path) -> None:
    correction = _dynamic_discrepancy_correction()

    for values in (
        correction.graph_basis,
        correction.graph_eigenvalues,
        correction.position_coefficients_m,
        correction.velocity_coefficients_mps,
        correction.generalized_force_coefficients_n,
        correction.structural_coefficients_m,
    ):
        assert values.flags.writeable is False
        with pytest.raises(ValueError):
            values.setflags(write=True)

    limited, diagnostics = provider.scale_coefficients_to_field_limit(
        correction.graph_basis,
        correction.position_coefficients_m,
        maximum_node_norm=1e-5,
    )
    assert diagnostics["limit_applied"] is True
    assert np.max(np.linalg.norm(correction.graph_basis @ limited, axis=1)) == (
        pytest.approx(1e-5)
    )

    written = provider.write_dynamic_discrepancy_correction(
        tmp_path / "correction",
        correction,
    )
    loaded = provider.load_dynamic_discrepancy_correction(written["manifest_path"])
    assert loaded.artifact_id == correction.artifact_id
    np.testing.assert_allclose(
        loaded.position_field_m(),
        correction.position_field_m(),
    )
