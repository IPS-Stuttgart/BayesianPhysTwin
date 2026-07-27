import numpy as np

from bayesian_phystwin import (
    PseudoMeasurementBatch,
    robust_mixture_likelihood,
)
from bayesian_phystwin.phystwin_conditioned_dino_correspondence import (
    ConditionedDinoConfig,
    MetricViewObservation,
    covariance_intersection,
    exact_fallback_points,
    fuse_unknown_correlation,
    match_descriptor_near_prediction,
    refine_patch_correlation,
)


def _unit_feature_map() -> np.ndarray:
    features = np.zeros((3, 3, 4), dtype=np.float64)
    features[..., 0] = 1.0
    features[1, 1] = [0.0, 1.0, 0.0, 0.0]
    features[1, 2] = [0.0, 0.8, 0.6, 0.0]
    return features


def _descriptor_config(**overrides: object) -> ConditionedDinoConfig:
    values: dict[str, object] = {
        "search_radius_px": 100.0,
        "descriptor_temperature": 0.04,
        "minimum_cosine_similarity": 0.5,
        "maximum_normalized_entropy": 0.99,
        "minimum_candidate_count": 3,
    }
    values.update(overrides)
    return ConditionedDinoConfig(**values)


def _observation(
    point: list[float],
    covariance: np.ndarray,
    *,
    reliability: float = 0.8,
) -> MetricViewObservation:
    return MetricViewObservation(
        mean_world_m=np.asarray(point),
        covariance_world_m2=np.asarray(covariance),
        prior_reliability=reliability,
        accepted=True,
    )


def test_descriptor_match_recovers_unique_material_identity() -> None:
    match = match_descriptor_near_prediction(
        np.array([0.0, 1.0, 0.0, 0.0]),
        _unit_feature_map(),
        np.ones((3, 3), dtype=bool),
        np.array([50.0, 50.0]),
        image_width=90,
        image_height=90,
        config=_descriptor_config(),
    )

    assert match.accepted
    np.testing.assert_array_equal(match.uv_px, [45.0, 45.0])
    assert match.cosine_similarity == 1.0
    assert match.association_probability > 0.99
    assert match.prior_reliability > 0.5


def test_state_residual_does_not_enter_prior_perception_reliability() -> None:
    common = dict(
        reference_descriptor=np.array([0.0, 1.0, 0.0, 0.0]),
        feature_map=_unit_feature_map(),
        valid_mask=np.ones((3, 3), dtype=bool),
        image_width=90,
        image_height=90,
        config=_descriptor_config(),
    )
    close_prediction = match_descriptor_near_prediction(
        predicted_uv_px=np.array([45.0, 45.0]),
        **common,
    )
    far_prediction = match_descriptor_near_prediction(
        predicted_uv_px=np.array([70.0, 70.0]),
        **common,
    )

    assert close_prediction.candidate_count == far_prediction.candidate_count
    assert close_prediction.prior_reliability == far_prediction.prior_reliability
    assert (
        close_prediction.association_probability
        == far_prediction.association_probability
    )
    np.testing.assert_array_equal(close_prediction.covariance_px2, far_prediction.covariance_px2)


def test_assignment_ambiguity_increases_observation_covariance() -> None:
    unique_features = _unit_feature_map()
    ambiguous_features = unique_features.copy()
    ambiguous_features[1, 2] = [0.0, 1.0, 0.0, 0.0]
    common = dict(
        reference_descriptor=np.array([0.0, 1.0, 0.0, 0.0]),
        valid_mask=np.ones((3, 3), dtype=bool),
        predicted_uv_px=np.array([45.0, 45.0]),
        image_width=90,
        image_height=90,
        config=_descriptor_config(),
    )

    unique = match_descriptor_near_prediction(
        feature_map=unique_features,
        **common,
    )
    ambiguous = match_descriptor_near_prediction(
        feature_map=ambiguous_features,
        **common,
    )

    assert ambiguous.normalized_entropy > unique.normalized_entropy
    assert ambiguous.prior_reliability < unique.prior_reliability
    assert np.trace(ambiguous.covariance_px2) > np.trace(unique.covariance_px2)


def test_patch_refinement_recovers_known_image_translation() -> None:
    rng = np.random.default_rng(7)
    reference = np.zeros((41, 41), dtype=np.float64)
    patch = rng.normal(size=(7, 7))
    reference[17:24, 17:24] = patch
    current = np.zeros_like(reference)
    current[20:27, 12:19] = patch
    config = ConditionedDinoConfig(
        patch_radius_px=3,
        patch_search_radius_px=8,
        minimum_patch_standard_deviation=1e-4,
        minimum_patch_correlation=0.8,
    )

    match = refine_patch_correlation(
        reference,
        current,
        np.array([20.0, 20.0]),
        np.array([15.0, 23.0]),
        np.ones_like(reference, dtype=bool),
        config=config,
    )

    assert match.accepted
    np.testing.assert_array_equal(match.uv_px, [15.0, 23.0])
    assert match.correlation > 0.999


def test_covariance_intersection_is_not_independence_fusion() -> None:
    covariance = np.diag([4.0, 9.0, 16.0])
    _, fused_covariance, _ = covariance_intersection(
        np.zeros(3),
        covariance,
        np.zeros(3),
        covariance,
    )
    independent_covariance = np.linalg.inv(
        np.linalg.inv(covariance) + np.linalg.inv(covariance)
    )

    np.testing.assert_allclose(fused_covariance, covariance)
    assert np.all(
        np.linalg.eigvalsh(fused_covariance - independent_covariance) >= -1e-12
    )


def test_duplicate_correlated_view_does_not_create_arbitrary_confidence() -> None:
    covariance = 1e-4 * np.eye(3)
    observation = _observation([0.1, 0.2, 0.3], covariance)
    config = ConditionedDinoConfig(
        minimum_views=2,
        shared_bias_standard_deviation_m=0.0,
    )

    two_views = fuse_unknown_correlation(
        [observation, observation],
        config=config,
    )
    three_views = fuse_unknown_correlation(
        [observation, observation, observation],
        config=config,
    )

    assert two_views.accepted and three_views.accepted
    np.testing.assert_allclose(two_views.covariance_world_m2, covariance)
    np.testing.assert_allclose(three_views.covariance_world_m2, covariance)


def test_shared_bias_floor_is_added_once_after_multiview_fusion() -> None:
    covariance = 1e-4 * np.eye(3)
    observation = _observation([0.1, 0.2, 0.3], covariance)
    bias_standard_deviation = 0.005
    config = ConditionedDinoConfig(
        minimum_views=2,
        shared_bias_standard_deviation_m=bias_standard_deviation,
    )

    result = fuse_unknown_correlation(
        [observation, observation, observation],
        config=config,
    )

    expected = covariance + bias_standard_deviation**2 * np.eye(3)
    np.testing.assert_allclose(result.covariance_world_m2, expected)


def test_exact_fallback_preserves_rejected_baseline_bytes() -> None:
    baseline = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    candidate = baseline + np.float32(0.125)

    result = exact_fallback_points(
        baseline,
        candidate,
        np.array([True, False]),
    )

    np.testing.assert_array_equal(result[0], candidate[0])
    assert result[1].tobytes() == baseline[1].tobytes()
    assert result.dtype == baseline.dtype


def test_gross_innovation_is_processed_once_by_robust_likelihood() -> None:
    match = match_descriptor_near_prediction(
        np.array([0.0, 1.0, 0.0, 0.0]),
        _unit_feature_map(),
        np.ones((3, 3), dtype=bool),
        np.array([45.0, 45.0]),
        image_width=90,
        image_height=90,
        config=_descriptor_config(),
    )
    assert match.prior_reliability > 0.5
    batch = PseudoMeasurementBatch(
        observed=[[0.25, 0.0, 0.0]],
        predicted=[[0.0, 0.0, 0.0]],
        variance=1e-5,
    )

    result = robust_mixture_likelihood(
        batch,
        prior_reliability=np.array([match.prior_reliability]),
    )

    assert result.posterior_inlier_probability[0] < 1e-10
    assert match.prior_reliability > 0.5
