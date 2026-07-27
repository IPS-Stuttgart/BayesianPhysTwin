from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.gauge_aware_belief import (
    GaugeAwareBeliefConfig,
    GaugeAwareObservationBatch,
    select_gauge_aware_candidate,
    update_gauge_aware_belief,
)
from bayesian_phystwin.observation_belief import (
    ObservationBeliefV1,
    load_observation_belief,
    save_observation_belief,
)
from bayesian_phystwin.observation_belief_gauge_adapter import (
    build_gauge_aware_batch_from_observation_belief,
)
from bayesian_phystwin.phystwin_profile import truncate_profile_prediction_weights


def _proper_rotation(rng: np.random.Generator) -> np.ndarray:
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _dense_local_covariance(belief: ObservationBeliefV1) -> np.ndarray:
    count = belief.observation_count
    dense = np.zeros((3 * count, 3 * count), dtype=np.float64)
    for index, covariance in enumerate(belief.local_covariance_m2):
        row = slice(3 * index, 3 * index + 3)
        dense[row, row] = covariance
    return dense


def _dense_observation_covariance(belief: ObservationBeliefV1) -> np.ndarray:
    count = belief.observation_count
    dense = _dense_local_covariance(belief)
    factor_blocks = np.zeros(
        (count, 3, belief.factor_rank),
        dtype=np.float64,
    )
    for group_id in np.unique(belief.factor_group_ids):
        factor_blocks.fill(0.0)
        selected = belief.factor_group_ids == group_id
        factor_blocks[selected] = belief.low_rank_factor_m[selected]
        factor = factor_blocks.reshape(3 * count, belief.factor_rank)
        dense += factor @ factor.T
    return dense


def _random_observation_belief(seed: int) -> ObservationBeliefV1:
    rng = np.random.default_rng(seed)
    count = 7
    covariance_root = rng.normal(size=(count, 3, 3))
    local_covariance = np.einsum(
        "nij,nkj->nik", covariance_root, covariance_root
    )
    local_covariance *= 1e-5
    local_covariance += np.eye(3)[None] * 1e-6
    correlation_groups = np.arange(count, dtype=np.int64) % 3
    group_ids = np.unique(correlation_groups)
    return ObservationBeliefV1(
        case_id="quality-ratchet",
        stream_id="randomized-contract",
        causal_frame_stop=count + 1,
        view_names=("view-0",),
        window_names=("window-0",),
        factor_names=("shared-x", "shared-y"),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.arange(count, dtype=np.int64),
        mean_xyz_m=rng.normal(scale=0.05, size=(count, 3)),
        frame_ids=np.arange(count, dtype=np.int64),
        entity_ids=np.arange(count, dtype=np.int64),
        view_indices=np.zeros(count, dtype=np.int64),
        window_indices=np.zeros(count, dtype=np.int64),
        correlation_group_ids=correlation_groups,
        factor_group_ids=np.zeros(count, dtype=np.int64),
        prior_reliability=rng.uniform(0.2, 1.0, size=count),
        association_probability=rng.uniform(0.2, 1.0, size=count),
        local_covariance_m2=local_covariance,
        low_rank_factor_m=rng.normal(scale=1e-3, size=(count, 3, 2)),
        group_ids=group_ids,
        group_prior_nominal_probability=rng.uniform(
            0.5, 1.0, size=len(group_ids)
        ),
        group_composite_weight=rng.uniform(0.5, 1.0, size=len(group_ids)),
        metadata={"seed": seed, "nested": {"beta": 2, "alpha": 1}},
    )


def _adapt_random_belief(
    belief: ObservationBeliefV1,
    *,
    seed: int,
):
    rng = np.random.default_rng(seed)
    state_count = 2
    count = belief.observation_count
    empty_nuisance = np.zeros((count, 3, 0), dtype=np.float64)
    return build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=np.zeros_like(belief.mean_xyz_m),
        state_jacobian=rng.normal(size=(count, 3, state_count)),
        query_state_jacobian=rng.normal(size=(4, 3, state_count)),
        physical_response_scale_m=0.25,
        shared_bias_jacobian=empty_nuisance,
        view_bias_jacobian=empty_nuisance,
    )


@pytest.mark.parametrize("seed", range(8))
def test_observation_contract_digest_and_covariance_properties(seed: int) -> None:
    belief = _random_observation_belief(seed)
    minimum_eigenvalues = np.linalg.eigvalsh(belief.local_covariance_m2)[:, 0]
    assert np.all(minimum_eigenvalues > 0.0)
    assert not belief.mean_xyz_m.flags.writeable
    assert not belief.local_covariance_m2.flags.writeable

    reordered_metadata = replace(
        belief,
        metadata={"nested": {"alpha": 1, "beta": 2}, "seed": seed},
    )
    assert reordered_metadata.artifact_id == belief.artifact_id

    changed_mean = belief.mean_xyz_m.copy()
    changed_mean[0, 0] = np.nextafter(changed_mean[0, 0], np.inf)
    changed = replace(belief, mean_xyz_m=changed_mean)
    assert changed.artifact_id != belief.artifact_id


@pytest.mark.parametrize("seed", range(5))
def test_random_observation_round_trip_is_content_exact(
    seed: int, tmp_path: Path
) -> None:
    belief = _random_observation_belief(seed + 50)
    path = tmp_path / f"belief-{seed}.npz"

    save_observation_belief(path, belief)
    loaded = load_observation_belief(path)

    assert loaded.artifact_id == belief.artifact_id
    assert loaded.metadata == belief.metadata
    for name, expected in belief._arrays().items():
        np.testing.assert_array_equal(loaded._arrays()[name], expected)
        assert not loaded._arrays()[name].flags.writeable


@pytest.mark.parametrize("seed", range(6))
def test_random_sim3_transform_preserves_covariance_semantics(seed: int) -> None:
    rng = np.random.default_rng(seed + 100)
    belief = _random_observation_belief(seed)
    rotation = _proper_rotation(rng)
    translation = rng.normal(scale=0.2, size=3)
    scale = float(np.exp(rng.uniform(-1.0, 1.0)))

    transformed = belief.transformed(
        rotation=rotation,
        translation_m=translation,
        scale=scale,
        stream_id="transformed",
    )
    expected_mean = scale * np.einsum(
        "ij,nj->ni", rotation, belief.mean_xyz_m
    ) + translation
    expected_covariance = scale**2 * np.einsum(
        "ij,njk,lk->nil",
        rotation,
        belief.local_covariance_m2,
        rotation,
    )
    expected_factor = scale * np.einsum(
        "ij,njr->nir", rotation, belief.low_rank_factor_m
    )
    coordinate_transform = np.kron(
        np.eye(belief.observation_count),
        scale * rotation,
    )
    expected_dense_covariance = (
        coordinate_transform
        @ _dense_observation_covariance(belief)
        @ coordinate_transform.T
    )

    np.testing.assert_allclose(transformed.mean_xyz_m, expected_mean)
    np.testing.assert_allclose(
        transformed.local_covariance_m2, expected_covariance
    )
    np.testing.assert_allclose(transformed.low_rank_factor_m, expected_factor)
    np.testing.assert_allclose(
        _dense_observation_covariance(transformed),
        expected_dense_covariance,
        atol=1e-15,
        rtol=1e-12,
    )
    assert np.all(
        np.linalg.eigvalsh(transformed.local_covariance_m2)[:, 0] > 0.0
    )
    assert transformed.metadata["metric_transform"]["source_artifact_id"] == (
        belief.artifact_id
    )


@pytest.mark.parametrize("seed", range(5))
def test_adapter_represents_low_rank_covariance_exactly_once(seed: int) -> None:
    belief = _random_observation_belief(seed + 500)
    adapted = _adapt_random_belief(belief, seed=seed + 600)
    batch = adapted.batch

    np.testing.assert_array_equal(
        batch.observation_covariance_m2,
        belief.local_covariance_m2,
    )
    expanded_factor = batch.gauge_jacobian.reshape(
        3 * belief.observation_count,
        -1,
    )
    represented_covariance = _dense_local_covariance(belief)
    represented_covariance += (
        expanded_factor
        @ batch.gauge_prior_covariance
        @ expanded_factor.T
    )
    np.testing.assert_allclose(
        represented_covariance,
        _dense_observation_covariance(belief),
        atol=1e-15,
        rtol=1e-12,
    )
    assert batch.metadata["low_rank_covariance_double_counted"] is False


@pytest.mark.parametrize("seed", range(5))
def test_future_source_suffix_does_not_change_causal_numerics(seed: int) -> None:
    belief = _random_observation_belief(seed + 700)
    with_suffix = replace(
        belief,
        source_artifact_sha256="c" * 64,
        metadata={
            **belief.metadata,
            "unobserved_source_suffix_frame_count": 100 + seed,
        },
    )
    original = _adapt_random_belief(belief, seed=seed + 800)
    appended = _adapt_random_belief(with_suffix, seed=seed + 800)

    assert original.observation_artifact_id != appended.observation_artifact_id
    for name in (
        "innovation_m",
        "observation_covariance_m2",
        "state_jacobian",
        "gauge_jacobian",
        "shared_bias_jacobian",
        "view_bias_jacobian",
        "query_state_jacobian",
        "gauge_prior_covariance",
        "prior_reliability",
        "prior_nominal_probability",
        "composite_weight",
    ):
        np.testing.assert_array_equal(
            getattr(original.batch, name),
            getattr(appended.batch, name),
        )

    config = GaugeAwareBeliefConfig(
        effective_samples_per_correlation_group=belief.observation_count,
        maximum_state_update_m=1.0,
        maximum_update_to_physical_response_ratio=10.0,
    )
    original_result = update_gauge_aware_belief(original.batch, config=config)
    appended_result = update_gauge_aware_belief(appended.batch, config=config)
    assert original_result.inference_admissible
    assert appended_result.inference_admissible
    np.testing.assert_allclose(
        appended_result.state_coefficients,
        original_result.state_coefficients,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        appended_result.posterior_covariance,
        original_result.posterior_covariance,
        atol=1e-12,
        rtol=1e-12,
    )


def _random_identifiable_batch(seed: int) -> GaugeAwareObservationBatch:
    rng = np.random.default_rng(seed)
    count = 14
    raw_modes = rng.normal(size=(count, 2))
    modes, _ = np.linalg.qr(raw_modes)
    modes *= np.sqrt(count)
    state = np.zeros((count, 3, 2), dtype=np.float64)
    state[:, 0, :] = modes
    true_coefficients = rng.uniform(-0.004, 0.004, size=2)
    innovation = np.einsum("mcs,s->mc", state, true_coefficients)
    covariance = np.zeros((count, 3, 3), dtype=np.float64)
    for index in range(count):
        covariance[index] = np.diag(rng.uniform(1e-6, 4e-6, size=3))
    prior_root = rng.normal(size=(2, 2))
    prior = prior_root @ prior_root.T
    prior *= 0.008 / np.max(np.linalg.eigvalsh(prior))
    prior += np.eye(2) * 0.002
    empty = np.zeros((count, 3, 0), dtype=np.float64)
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=covariance,
        state_jacobian=state,
        gauge_jacobian=empty,
        shared_bias_jacobian=empty,
        view_bias_jacobian=empty,
        query_state_jacobian=state.copy(),
        gauge_prior_covariance=np.zeros((0, 0), dtype=np.float64),
        correlation_group_ids=tuple(f"window-{index % 3}" for index in range(count)),
        prior_reliability=np.ones(count, dtype=np.float64),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=prior,
        metadata={"seed": seed},
    )


def _permuted_batch(
    batch: GaugeAwareObservationBatch, permutation: np.ndarray
) -> GaugeAwareObservationBatch:
    return GaugeAwareObservationBatch(
        innovation_m=batch.innovation_m[permutation],
        observation_covariance_m2=batch.observation_covariance_m2[permutation],
        state_jacobian=batch.state_jacobian[permutation],
        gauge_jacobian=batch.gauge_jacobian[permutation],
        shared_bias_jacobian=batch.shared_bias_jacobian[permutation],
        view_bias_jacobian=batch.view_bias_jacobian[permutation],
        query_state_jacobian=batch.query_state_jacobian,
        gauge_prior_covariance=batch.gauge_prior_covariance,
        correlation_group_ids=tuple(
            batch.correlation_group_ids[index] for index in permutation
        ),
        prior_reliability=batch.prior_reliability[permutation],
        prior_nominal_probability=batch.prior_nominal_probability[permutation],
        composite_weight=batch.composite_weight[permutation],
        physical_response_scale_m=batch.physical_response_scale_m,
        state_prior_covariance_m2=batch.state_prior_covariance_m2,
        metadata=batch.metadata,
    )


@pytest.mark.parametrize("seed", range(8))
def test_gauge_update_is_permutation_invariant_and_covariance_psd(seed: int) -> None:
    batch = _random_identifiable_batch(seed)
    config = GaugeAwareBeliefConfig(
        effective_samples_per_correlation_group=14.0
    )
    result = update_gauge_aware_belief(batch, config=config)
    permutation = np.random.default_rng(seed + 1000).permutation(
        len(batch.innovation_m)
    )
    permuted = update_gauge_aware_belief(
        _permuted_batch(batch, permutation), config=config
    )

    assert result.inference_admissible
    assert permuted.inference_admissible
    np.testing.assert_allclose(
        permuted.state_coefficients,
        result.state_coefficients,
        atol=1e-9,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        permuted.posterior_covariance,
        result.posterior_covariance,
        atol=1e-9,
        rtol=1e-9,
    )
    eigenvalues = np.linalg.eigvalsh(result.posterior_covariance)
    assert np.min(eigenvalues) >= -1e-10
    assert np.trace(result.posterior_covariance[:2, :2]) <= (
        np.trace(batch.state_prior_covariance_m2) + 1e-10
    )


@pytest.mark.parametrize("seed", range(6))
def test_unsupported_state_mode_retains_prior_variance(seed: int) -> None:
    rng = np.random.default_rng(seed + 1100)
    count = 12
    raw_modes = rng.normal(size=(count, 2))
    modes, _ = np.linalg.qr(raw_modes)
    modes *= np.sqrt(count)
    state = np.zeros((count, 3, 3), dtype=np.float64)
    state[:, 0, 0] = modes[:, 0]
    state[:, 1, 1] = modes[:, 1]
    true_coefficients = np.asarray([0.003, -0.002, 0.0])
    innovation = np.einsum("mcs,s->mc", state, true_coefficients)
    covariance = np.repeat(
        np.diag([2e-6, 3e-6, 4e-6])[None],
        count,
        axis=0,
    )
    prior = np.diag([0.002, 0.003, 0.007])
    empty = np.zeros((count, 3, 0), dtype=np.float64)
    batch = GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=covariance,
        state_jacobian=state,
        gauge_jacobian=empty,
        shared_bias_jacobian=empty,
        view_bias_jacobian=empty,
        query_state_jacobian=state.copy(),
        gauge_prior_covariance=np.zeros((0, 0), dtype=np.float64),
        correlation_group_ids=tuple(f"group-{index % 3}" for index in range(count)),
        prior_reliability=np.ones(count, dtype=np.float64),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=prior,
    )
    result = update_gauge_aware_belief(
        batch,
        config=GaugeAwareBeliefConfig(
            effective_samples_per_correlation_group=float(count),
            maximum_state_update_m=1.0,
            maximum_update_to_physical_response_ratio=10.0,
        ),
    )

    assert result.inference_admissible
    assert result.identifiable_state_transform.shape == (3, 2)
    assert result.state_coefficients[2] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        result.posterior_covariance[2],
        prior[2],
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        result.posterior_covariance[:, 2],
        prior[:, 2],
        atol=1e-12,
        rtol=1e-12,
    )
    assert np.all(np.diag(result.posterior_covariance)[:2] < np.diag(prior)[:2])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_missing_guard_fallback_preserves_random_baseline_bytes(
    dtype: type[np.generic],
) -> None:
    result = update_gauge_aware_belief(_random_identifiable_batch(200))
    baseline = np.asarray([0.0, -0.0, 1.5, -2.25], dtype=dtype)
    candidate = np.asarray([9.0, 8.0, 7.0, 6.0], dtype=dtype)

    selection = select_gauge_aware_candidate(baseline, candidate, result)

    assert result.inference_admissible
    assert not selection.candidate_accepted
    assert selection.reason == "missing-regret-guard-exact-baseline-fallback"
    assert selection.selected_value.dtype == baseline.dtype
    assert selection.selected_value.shape == baseline.shape
    assert selection.selected_value.tobytes() == baseline.tobytes()


@pytest.mark.parametrize("seed", range(12))
def test_truncation_mass_composes_relative_to_original_posterior(seed: int) -> None:
    rng = np.random.default_rng(seed + 300)
    original = rng.lognormal(mean=0.0, sigma=1.0, size=(5, 7))
    normalized = original / np.sum(original)
    requested_mass = float(rng.uniform(0.25, 0.95))

    truncated, retained_mass, selected_count = truncate_profile_prediction_weights(
        original,
        retained_mass=requested_mass,
    )

    assert np.sum(truncated) == pytest.approx(1.0, abs=1e-14)
    assert retained_mass >= requested_mass
    assert selected_count == int(np.count_nonzero(truncated))

    descending = np.argsort(-normalized.reshape(-1), kind="stable")
    selected = descending[:selected_count]
    np.testing.assert_array_equal(
        np.flatnonzero(truncated.reshape(-1) > 0.0), np.sort(selected)
    )
    if selected_count > 1:
        mass_without_last = float(np.sum(normalized.reshape(-1)[selected[:-1]]))
        assert mass_without_last < requested_mass

    subset_count = int(rng.integers(1, selected_count + 1))
    subset = rng.choice(selected, size=subset_count, replace=False)
    downstream_retained_mass = float(np.sum(truncated.reshape(-1)[subset]))
    original_retained_mass = float(np.sum(normalized.reshape(-1)[subset]))
    assert original_retained_mass == pytest.approx(
        retained_mass * downstream_retained_mass,
        rel=1e-12,
        abs=1e-15,
    )
