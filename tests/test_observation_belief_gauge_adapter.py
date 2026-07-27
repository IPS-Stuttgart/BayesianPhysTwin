from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.gauge_aware_belief import (
    COMPOSITE_WEIGHT_MODE_CONSUMER_CAP,
    COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL,
    GaugeAwareBeliefConfig,
    GaugeAwareObservationBatch,
    update_gauge_aware_belief,
)
from bayesian_phystwin.observation_belief import ObservationBeliefV1
from bayesian_phystwin.observation_belief_gauge_adapter import (
    build_gauge_aware_batch_from_observation_belief,
    centered_view_translation_bias_jacobian,
    global_translation_bias_jacobian,
)


def _belief(
    *,
    association_probability: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> ObservationBeliefV1:
    local = np.repeat(np.eye(3)[None], 4, axis=0) * 4e-6
    factors = np.zeros((4, 3, 1))
    factors[:2, 0, 0] = 0.003
    factors[2:, 1, 0] = 0.004
    return ObservationBeliefV1(
        case_id="case",
        stream_id="prob4d:unfused",
        causal_frame_stop=10,
        view_names=("camera-0", "camera-1"),
        window_names=("window-0", "window-1"),
        factor_names=("gauge-translation",),
        source_repository="FlorianPfaff/Prob4D",
        source_revision="a" * 40,
        source_artifact_sha256="b" * 64,
        declared_frame_ids=np.asarray([8, 9]),
        mean_xyz_m=np.asarray(
            [
                [0.01, 0.00, 1.0],
                [0.01, 0.00, 1.0],
                [0.00, 0.02, 1.0],
                [0.00, 0.02, 1.0],
            ]
        ),
        frame_ids=np.asarray([8, 8, 9, 9]),
        entity_ids=np.asarray([0, 0, 0, 0]),
        view_indices=np.asarray([0, 1, 0, 1]),
        window_indices=np.asarray([0, 0, 1, 1]),
        correlation_group_ids=np.asarray([4, 4, 9, 9]),
        factor_group_ids=np.asarray([11, 11, 13, 13]),
        prior_reliability=np.asarray([0.9, 0.8, 0.7, 0.6]),
        association_probability=(
            np.asarray([0.2, 0.3, 0.4, 0.5])
            if association_probability is None
            else association_probability
        ),
        local_covariance_m2=local,
        low_rank_factor_m=factors,
        group_ids=np.asarray([4, 9]),
        group_prior_nominal_probability=np.asarray([0.85, 0.65]),
        group_composite_weight=np.asarray([0.5, 0.25]),
        metadata={} if metadata is None else metadata,
    )


def _state_design(count: int) -> np.ndarray:
    design = np.zeros((count, 3, 1))
    design[:, 0, 0] = np.asarray([-1.0, -1.0, 1.0, 1.0])[:count]
    return design


def _adapt(
    belief: ObservationBeliefV1,
    *,
    predicted: np.ndarray | None = None,
):
    state = _state_design(belief.observation_count)
    return build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=(
            np.zeros_like(belief.mean_xyz_m) if predicted is None else predicted
        ),
        state_jacobian=state,
        query_state_jacobian=state[:2],
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.asarray([[4e-4]]),
    )


def test_low_rank_covariance_becomes_one_nuisance_per_factor_group() -> None:
    belief = _belief()
    adapted = _adapt(belief)
    batch = adapted.batch

    np.testing.assert_array_equal(
        batch.observation_covariance_m2,
        belief.local_covariance_m2,
    )
    assert batch.gauge_jacobian.shape == (4, 3, 2)
    assert adapted.gauge_parameter_names == (
        "factor-group-11:gauge-translation",
        "factor-group-13:gauge-translation",
    )
    for first in range(belief.observation_count):
        for second in range(belief.observation_count):
            represented = (
                batch.gauge_jacobian[first]
                @ batch.gauge_prior_covariance
                @ batch.gauge_jacobian[second].T
            )
            expected = (
                belief.low_rank_factor_m[first] @ belief.low_rank_factor_m[second].T
                if belief.factor_group_ids[first] == belief.factor_group_ids[second]
                else np.zeros((3, 3))
            )
            np.testing.assert_allclose(represented, expected)


def test_adapter_keeps_association_separate_from_all_reliability_inputs() -> None:
    first = _adapt(_belief(association_probability=np.zeros(4)))
    second = _adapt(
        _belief(association_probability=np.ones(4)),
        predicted=np.full((4, 3), 100.0),
    )

    np.testing.assert_array_equal(
        first.batch.prior_reliability,
        second.batch.prior_reliability,
    )
    np.testing.assert_allclose(
        first.batch.prior_nominal_probability,
        [0.85, 0.85, 0.65, 0.65],
    )
    np.testing.assert_allclose(
        first.batch.composite_weight,
        [0.5, 0.5, 0.25, 0.25],
    )
    assert np.all(first.association_probability == 0.0)
    assert np.all(second.association_probability == 1.0)
    assert first.summary()["association_used_as_prior_reliability"] is False


def test_default_bias_design_is_full_rank_without_mean_duplication() -> None:
    views = np.asarray([0, 1, 0, 1])
    shared = global_translation_bias_jacobian(len(views))
    centered = centered_view_translation_bias_jacobian(
        views,
        view_count=2,
    )
    combined = np.concatenate((shared, centered), axis=2).reshape(12, 6)

    assert np.linalg.matrix_rank(combined) == 6
    np.testing.assert_allclose(
        centered[views == 0].mean(axis=0) + centered[views == 1].mean(axis=0),
        0.0,
        atol=1e-15,
    )


def test_group_weights_cap_duplicate_correlated_evidence() -> None:
    count = 8
    mode = np.linspace(-1.0, 1.0, count)
    state = np.zeros((count, 3, 1))
    state[:, 0, 0] = mode
    innovation = np.zeros((count, 3))
    innovation[:, 0] = 0.01 * mode

    def run(repetitions: int):
        repeated_state = np.tile(state, (repetitions, 1, 1))
        repeated_innovation = np.tile(innovation, (repetitions, 1))
        batch = GaugeAwareObservationBatch(
            innovation_m=repeated_innovation,
            observation_covariance_m2=np.tile(
                np.eye(3)[None] * 1e-6,
                (count * repetitions, 1, 1),
            ),
            state_jacobian=repeated_state,
            gauge_jacobian=np.zeros((count * repetitions, 3, 0)),
            shared_bias_jacobian=np.zeros((count * repetitions, 3, 0)),
            view_bias_jacobian=np.zeros((count * repetitions, 3, 0)),
            query_state_jacobian=state,
            gauge_prior_covariance=np.zeros((0, 0)),
            correlation_group_ids=tuple(
                "one-correlated-window" for _ in range(count * repetitions)
            ),
            prior_reliability=np.ones(count * repetitions),
            prior_nominal_probability=np.full(
                count * repetitions,
                0.5,
            ),
            composite_weight=np.full(count * repetitions, 0.5),
            physical_response_scale_m=0.05,
        )
        return update_gauge_aware_belief(
            batch,
            config=GaugeAwareBeliefConfig(
                effective_samples_per_correlation_group=float(count),
            ),
        )

    original = run(1)
    duplicated = run(2)
    assert original.accepted and duplicated.accepted
    assert original.diagnostics["effective_observation_information_mass"] == 2.0
    assert duplicated.diagnostics["effective_observation_information_mass"] == 2.0
    np.testing.assert_allclose(
        original.posterior_covariance,
        duplicated.posterior_covariance,
        rtol=1e-12,
        atol=1e-15,
    )


def test_unanchored_global_state_translation_abstains() -> None:
    belief = _belief()
    state = global_translation_bias_jacobian(belief.observation_count)
    adapted = build_gauge_aware_batch_from_observation_belief(
        belief,
        physical_prediction_xyz_m=np.zeros_like(belief.mean_xyz_m),
        state_jacobian=state,
        query_state_jacobian=state[:2],
        physical_response_scale_m=0.05,
    )

    result = update_gauge_aware_belief(adapted.batch)

    assert not result.accepted
    assert result.reason == "no-identifiable-query-state"


def test_adapter_respects_explicit_prob4d_final_group_power() -> None:
    adapted = _adapt(
        _belief(
            metadata={
                "group_composite_weight_semantics": (
                    "final-per-row-effective-sample-cap-v1"
                ),
                "effective_samples_per_group": 64.0,
            }
        )
    )

    assert adapted.batch.composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    assert adapted.batch.metadata["composite_weight_mode_source"] == (
        "artifact-metadata"
    )
    assert adapted.summary()["composite_weight_mode"] == (
        COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    )


def test_adapter_recognizes_legacy_prob4d_effective_sample_metadata() -> None:
    adapted = _adapt(_belief(metadata={"effective_samples_per_group": 64.0}))

    assert adapted.batch.composite_weight_mode == COMPOSITE_WEIGHT_MODE_PROVIDER_FINAL
    assert adapted.batch.metadata["composite_weight_mode_source"] == (
        "legacy-prob4d-export-metadata"
    )


def test_adapter_rejects_unknown_prob4d_composite_weight_semantics() -> None:
    belief = _belief(
        metadata={"group_composite_weight_semantics": "unsupported-semantics"}
    )

    with pytest.raises(ValueError, match="unsupported Prob4D"):
        _adapt(belief)


def test_non_prob4d_without_weight_semantics_uses_consumer_cap() -> None:
    belief = replace(
        _belief(),
        source_repository="Example/ObservationProvider",
        metadata={},
    )

    adapted = _adapt(belief)

    assert adapted.batch.composite_weight_mode == COMPOSITE_WEIGHT_MODE_CONSUMER_CAP
    assert adapted.batch.metadata["composite_weight_mode_source"] == "consumer-default"


def test_zero_rank_belief_has_no_gauge_parameters() -> None:
    belief = replace(
        _belief(),
        factor_names=(),
        low_rank_factor_m=np.zeros((4, 3, 0)),
    )

    adapted = _adapt(belief)

    assert adapted.batch.gauge_jacobian.shape == (4, 3, 0)
    assert adapted.gauge_parameter_names == ()
    assert adapted.gauge_parameter_group_ids.shape == (0,)


def test_global_translation_bias_rejects_empty_observation_set() -> None:
    with pytest.raises(ValueError, match="observation_count must be positive"):
        global_translation_bias_jacobian(0)
