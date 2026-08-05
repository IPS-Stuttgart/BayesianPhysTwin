from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "run_prob4d_real_camera_validation_v1.py"
SPEC = importlib.util.spec_from_file_location("prob4d_real_camera_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _protocol() -> dict:
    return json.loads(
        (ROOT / "protocols" / "prob4d_real_camera_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )


def test_prior_reliability_is_independent_of_physical_innovation() -> None:
    signature = inspect.signature(MODULE.source_only_prior_reliability)
    assert "residual" not in signature.parameters
    cues = {
        "parallel_disagreement": np.asarray([0.0, 0.4, 2.0]),
        "lateral_disagreement": np.asarray([0.1, 0.2, 1.0]),
        "parallel_variance": np.asarray([1.0, 1.0, 2.0]),
        "lateral_variance": np.asarray([1.0, 2.0, 2.0]),
        "overlap_count": np.asarray([1, 2, 0]),
        "minimum": 0.05,
    }

    low_state_residual_m = np.zeros((3, 3))
    high_state_residual_m = np.full((3, 3), 100.0)
    low = MODULE.source_only_prior_reliability(**cues)
    high = MODULE.source_only_prior_reliability(**cues)

    assert not np.array_equal(low_state_residual_m, high_state_residual_m)
    np.testing.assert_array_equal(low, high)


def test_correlated_duplicate_evidence_has_a_fixed_effective_sample_cap() -> None:
    count = 4096
    association = np.full(count, 0.8)
    weights = MODULE.group_capped_composite_weights(
        ["same-camera-frame"] * count,
        association,
        effective_samples_per_group=64.0,
    )

    assert np.sum(weights) <= 64.0
    assert np.sum(weights) < np.sum(association)
    np.testing.assert_allclose(np.unique(weights), [64.0 * 0.8 / count])


def test_assignment_mixture_spread_increases_metric_covariance() -> None:
    floor = 2.5e-7
    certain = MODULE.assignment_mixture_covariance(
        np.asarray([[0.0, 0.0, 0.0]]),
        np.asarray([1.0]),
        floor_m2=floor,
    )
    ambiguous = MODULE.assignment_mixture_covariance(
        np.asarray([[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        np.asarray([0.5, 0.5]),
        floor_m2=floor,
    )

    np.testing.assert_allclose(certain, np.eye(3) * floor)
    assert ambiguous[0, 0] > certain[0, 0]
    assert np.trace(ambiguous) > np.trace(certain)


def test_reserved_identity_selection_uses_frame_zero_only() -> None:
    frame_zero = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    future = np.zeros((8, len(frame_zero), 3))
    selected_before = MODULE.deterministic_farthest_point_indices(
        frame_zero, np.arange(len(frame_zero)), 2
    )
    future[:] = 1e6
    selected_after = MODULE.deterministic_farthest_point_indices(
        frame_zero, np.arange(len(frame_zero)), 2
    )

    np.testing.assert_array_equal(selected_before, selected_after)
    np.testing.assert_array_equal(selected_before, [0, 3])


def test_query_covariance_is_metric_symmetric_positive_semidefinite() -> None:
    jacobian = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            [[0.2, 0.1], [0.3, -0.2], [0.0, 1.0]],
        ]
    )
    state_covariance = np.asarray([[4e-4, 1e-4], [1e-4, 3e-4]])
    result = MODULE.query_covariance(jacobian, state_covariance)

    assert result.shape == (2, 3, 3)
    np.testing.assert_allclose(result, np.swapaxes(result, 1, 2), atol=1e-15)
    assert np.all(np.linalg.eigvalsh(result) >= -1e-15)


def test_guard_rejection_is_exact_zero_fallback() -> None:
    raw = np.asarray([[0.001, -0.002, 0.003]])
    accepted, deployed = MODULE.exact_fallback_selection(
        raw,
        inference_admissible=True,
        risk_score=0.8,
        risk_threshold=0.7,
    )

    assert accepted is False
    np.testing.assert_array_equal(deployed, np.zeros_like(raw))


def test_protocol_keeps_real_camera_claim_and_transfer_gates_frozen() -> None:
    protocol = _protocol()

    assert protocol["status"] == (
        "retrospective-source-locked-before-method-specific-scoring"
    )
    assert protocol["prob4d"]["revision"] == (
        "364f216c14f7770c1b360bb1b836b11ecf0c18b8"
    )
    assert protocol["prob4d"]["complete_overlap_window_count"] == 2
    assert protocol["identity"][
        "reserved_graph_nodes_excluded_from_camera_association"
    ]
    assert protocol["advancement_gates"][
        "deployed_mean_improvement_fraction_at_least"
    ] == 0.10
    assert "not prospective" in protocol["claim_boundary"]
