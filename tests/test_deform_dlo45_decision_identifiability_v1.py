from __future__ import annotations

import numpy as np

from experiments.deform_dlo45_decision_identifiability_v1.evaluate import (
    FRAME_COUNT,
    NODE_COUNT,
    Model,
    Protocol,
    decide,
    deterministic_kmeans,
    extract_observation,
    fit_model,
    partition_names,
)


def protocol() -> Protocol:
    return Protocol(
        prefix_frames=5,
        horizon_frames=25,
        stride_frames=25,
        action_scales=(0.0, 0.5, 1.0),
        neighbor_grid=(8,),
        cluster_grid=(4,),
        temperature_grid=(1.0,),
        regret_tolerance_grid=(0.0, 0.05),
        kmeans_iterations=20,
        source_fit_count=39,
        source_calibration_count=9,
        source_test_count=8,
        partition_domain="test-domain",
        source_gate_mean_ratio=1.2,
        source_gate_worst_trajectory_ratio=1.5,
        source_gate_minimum_nonfallback_fraction=0.0,
        bootstrap_replicates=100,
        bootstrap_seed=42,
    )


def trajectory(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros((FRAME_COUNT, NODE_COUNT, 3), dtype=np.float64)
    time = np.arange(FRAME_COUNT, dtype=np.float64)
    left = np.column_stack(
        (
            0.0005 * time,
            0.002 * np.sin(time / 50.0),
            0.1 + np.zeros_like(time),
        )
    )
    right = np.column_stack(
        (
            0.2 + 0.0002 * time,
            0.02 * np.sin(time / 70.0),
            0.1 + np.zeros_like(time),
        )
    )
    for node, weight in enumerate(np.linspace(0.0, 1.0, NODE_COUNT)):
        result[:, node] = (1.0 - weight) * left + weight * right
        result[:, node, 2] += 0.01 * np.sin(time / 20.0 + 2.0 * weight)
        result[:, node, 2] += 0.0002 * rng.normal(size=FRAME_COUNT)
    return result


def test_observation_is_invariant_to_future_internal_outcomes() -> None:
    frozen = protocol()
    first = trajectory()
    second = first.copy()
    second[30:55, 2:-2] += 100.0

    observation_a = extract_observation(first, 29, frozen)
    observation_b = extract_observation(second, 29, frozen)

    np.testing.assert_allclose(observation_a.feature, observation_b.feature)
    np.testing.assert_allclose(observation_a.baseline, observation_b.baseline)
    assert observation_a.length_scale == observation_b.length_scale


def test_partition_is_deterministic_and_complete() -> None:
    frozen = protocol()
    names = tuple(f"trajectory-{index:02d}.pkl" for index in range(56))

    first = partition_names(names, "DLO4", frozen)
    second = partition_names(tuple(reversed(names)), "DLO4", frozen)

    assert first == second
    assert len(first["fit"]) == 39
    assert len(first["calibration"]) == 9
    assert len(first["source_test"]) == 8
    assert set().union(*map(set, first.values())) == set(names)


def test_kmeans_is_deterministic_and_contiguous() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(40, 7))

    first = deterministic_kmeans(values, 5, 20)
    second = deterministic_kmeans(values, 5, 20)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.unique(first), np.arange(np.max(first) + 1))


def test_decision_outputs_registered_action_and_finite_certificate() -> None:
    frozen = protocol()
    rng = np.random.default_rng(11)
    features = rng.normal(size=(48, 81))
    residuals = rng.normal(scale=0.02, size=(48, 25 * 8 * 3))
    model = fit_model(
        features,
        residuals,
        cluster_count=4,
        neighbors=8,
        temperature_scale=1.0,
        regret_tolerance=0.05,
        protocol=frozen,
    )

    result = decide(features[0], model, frozen)

    assert isinstance(model, Model)
    assert 0 <= result.certificate_action < 3
    assert 0 <= result.jeffrey_action < 3
    assert 0 <= result.kernel_action < 3
    assert 0 <= result.map_action < 3
    assert np.all(np.isfinite(result.worst_case_regret))
    assert result.ambiguity_width >= 0.0
    assert result.unsupported_specificity_nats >= 0.0
