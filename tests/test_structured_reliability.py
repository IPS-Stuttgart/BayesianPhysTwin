import numpy as np
import pytest

from bayesian_phystwin import (
    MarkovReliabilityConfig,
    markov_log_evidence_batch,
    smooth_markov_reliability,
)
from bayesian_phystwin.structured_reliability import MarkovReliabilityResult


def test_markov_smoothing_uses_future_evidence_for_persistent_corruption() -> None:
    prior = np.full(8, 0.75)
    log_inlier = np.array([0.0, 0.0, -0.2, -0.3, -0.5, -1.0, -8.0, -10.0])
    log_outlier = np.array([-5.0, -5.0, -0.1, -0.1, -0.1, 0.0, 0.0, 0.0])
    iid = np.exp(np.log(prior) + log_inlier)
    iid /= iid + np.exp(np.log1p(-prior) + log_outlier)

    result = smooth_markov_reliability(
        prior,
        log_inlier,
        log_outlier,
        sequence_ids=["track-1"] * 8,
        time_values=np.arange(8),
        config=MarkovReliabilityConfig(
            inlier_persistence=0.98,
            outlier_persistence=0.98,
        ),
    )

    assert result.posterior_inlier_probability[5] < iid[5]
    assert result.posterior_inlier_probability[-1] < 1e-4
    assert result.sequence_count == 1
    assert np.isfinite(result.total_log_evidence)


def test_sequences_are_smoothed_independently_and_returned_in_input_order() -> None:
    result = smooth_markov_reliability(
        prior_reliability=np.full(4, 0.8),
        log_inlier_density=np.array([0.0, -10.0, 0.0, 0.0]),
        log_outlier_density=np.array([-5.0, 0.0, -5.0, -5.0]),
        sequence_ids=["a", "a", "b", "b"],
        time_values=[1, 0, 1, 0],
    )

    assert result.posterior_inlier_probability.shape == (4,)
    assert set(result.sequence_log_evidence) == {"a", "b"}
    assert (
        result.posterior_inlier_probability[1] < result.posterior_inlier_probability[0]
    )
    assert result.posterior_inlier_probability[2] > 0.9


def test_invalid_persistence_raises() -> None:
    with pytest.raises(ValueError):
        smooth_markov_reliability(
            np.array([0.5]),
            np.array([0.0]),
            np.array([0.0]),
            ["a"],
            [0],
            config=MarkovReliabilityConfig(inlier_persistence=1.0),
        )


def test_batched_evidence_matches_scalar_sequence_evidence() -> None:
    prior = np.array([0.8, 0.7, 0.6, 0.9])
    log_inlier = np.array(
        [
            [0.0, -0.1, -2.0, -3.0],
            [-1.0, -0.5, -0.2, -0.1],
        ]
    )
    log_outlier = np.array(
        [
            [-2.0, -1.0, -0.1, 0.0],
            [-0.2, -0.2, -1.0, -2.0],
        ]
    )
    ids = ["a", "a", "b", "b"]
    times = [0, 1, 0, 1]

    batched = markov_log_evidence_batch(prior, log_inlier, log_outlier, ids, times)
    scalar = np.array(
        [
            smooth_markov_reliability(
                prior,
                log_inlier[index],
                log_outlier[index],
                ids,
                times,
            ).total_log_evidence
            for index in range(2)
        ]
    )

    assert np.allclose(batched, scalar)


def test_order_only_mode_preserves_historical_gap_semantics() -> None:
    kwargs = {
        "prior_reliability": np.array([0.8, 0.8, 0.8]),
        "log_inlier_density": np.array([0.0, -0.2, -3.0]),
        "log_outlier_density": np.array([-3.0, -1.0, -0.1]),
        "sequence_ids": ["track"] * 3,
    }
    unit = smooth_markov_reliability(time_values=[0, 1, 2], **kwargs)
    gapped = smooth_markov_reliability(time_values=[0, 10, 100], **kwargs)

    assert np.allclose(
        unit.posterior_inlier_probability,
        gapped.posterior_inlier_probability,
    )
    assert unit.total_log_evidence == pytest.approx(gapped.total_log_evidence)


def test_integer_step_mode_uses_elapsed_transitions() -> None:
    prior = np.array([0.5, 0.5])
    log_inlier = np.array([0.0, -1.5])
    log_outlier = np.array([-8.0, -1.4])
    config = MarkovReliabilityConfig(
        inlier_persistence=0.99,
        outlier_persistence=0.95,
        time_delta_mode="integer-steps",
    )
    adjacent = smooth_markov_reliability(
        prior,
        log_inlier,
        log_outlier,
        ["track", "track"],
        [0, 1],
        config=config,
    )
    gapped = smooth_markov_reliability(
        prior,
        log_inlier,
        log_outlier,
        ["track", "track"],
        [0, 100],
        config=config,
    )

    stationary_inlier = (1.0 - config.outlier_persistence) / (
        2.0 - config.inlier_persistence - config.outlier_persistence
    )
    assert abs(gapped.posterior_inlier_probability[1] - stationary_inlier) < abs(
        adjacent.posterior_inlier_probability[1] - stationary_inlier
    )


def test_integer_step_batched_evidence_matches_scalar() -> None:
    prior = np.array([0.8, 0.7, 0.6, 0.9])
    log_inlier = np.array(
        [
            [0.0, -0.1, -2.0, -3.0],
            [-1.0, -0.5, -0.2, -0.1],
        ]
    )
    log_outlier = np.array(
        [
            [-2.0, -1.0, -0.1, 0.0],
            [-0.2, -0.2, -1.0, -2.0],
        ]
    )
    ids = ["a", "a", "b", "b"]
    times = [0, 3, 1, 5]
    config = MarkovReliabilityConfig(time_delta_mode="integer-steps")

    batched = markov_log_evidence_batch(
        prior,
        log_inlier,
        log_outlier,
        ids,
        times,
        config=config,
    )
    scalar = np.array(
        [
            smooth_markov_reliability(
                prior,
                log_inlier[index],
                log_outlier[index],
                ids,
                times,
                config=config,
            ).total_log_evidence
            for index in range(2)
        ]
    )

    assert np.allclose(batched, scalar)


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ([0.0, 1.5], "integer multiples"),
        ([1.0, 1.0], "strictly increasing"),
        ([0.0, np.inf], "finite"),
        (["first", "second"], "numeric"),
    ],
)
def test_integer_step_mode_rejects_invalid_time_values(
    times: list[object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        smooth_markov_reliability(
            np.array([0.5, 0.5]),
            np.zeros(2),
            np.zeros(2),
            ["track", "track"],
            times,
            config=MarkovReliabilityConfig(time_delta_mode="integer-steps"),
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (MarkovReliabilityConfig(outlier_persistence=1.0), "outlier_persistence"),
        (MarkovReliabilityConfig(probability_floor=0.5), "probability_floor"),
        (MarkovReliabilityConfig(time_delta_mode="unsupported"), "time_delta_mode"),
        (MarkovReliabilityConfig(time_step=0.0), "time_step"),
    ],
)
def test_additional_invalid_markov_configurations(
    config: MarkovReliabilityConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        smooth_markov_reliability(
            np.array([0.5]),
            np.array([0.0]),
            np.array([0.0]),
            ["track"],
            [0],
            config=config,
        )


def test_transition_helper_accepts_constant_matrix() -> None:
    import bayesian_phystwin.structured_reliability as reliability_module

    transition = np.log(np.array([[0.9, 0.1], [0.2, 0.8]]))
    assert reliability_module._transition_at(transition, 0) is transition


@pytest.mark.parametrize(
    ("prior", "log_inlier", "log_outlier", "ids", "times", "message"),
    [
        (np.ones(2), np.zeros(1), np.zeros(2), ["a", "a"], [0, 1], "shape"),
        (np.empty(0), np.empty(0), np.empty(0), [], [], "at least one"),
        (np.array([np.nan]), np.zeros(1), np.zeros(1), ["a"], [0], "prior"),
        (np.ones(1), np.array([np.nan]), np.zeros(1), ["a"], [0], "densities"),
    ],
)
def test_scalar_markov_input_validation(
    prior: np.ndarray,
    log_inlier: np.ndarray,
    log_outlier: np.ndarray,
    ids: list[str],
    times: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        smooth_markov_reliability(
            prior,
            log_inlier,
            log_outlier,
            ids,
            times,
        )


@pytest.mark.parametrize(
    ("prior", "log_inlier", "log_outlier", "ids", "times", "message"),
    [
        (np.ones(2), np.zeros(2), np.zeros(2), ["a", "a"], [0, 1], "equal shape"),
        (
            np.ones(2),
            np.zeros((1, 2)),
            np.zeros((2, 2)),
            ["a", "a"],
            [0, 1],
            "equal shape",
        ),
        (
            np.ones(1),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            ["a", "a"],
            [0, 1],
            "prior_reliability",
        ),
        (np.empty(0), np.zeros((0, 0)), np.zeros((0, 0)), [], [], "at least one"),
        (
            np.array([np.nan]),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            ["a"],
            [0],
            "prior_reliability",
        ),
        (
            np.ones(1),
            np.array([[np.nan]]),
            np.zeros((1, 1)),
            ["a"],
            [0],
            "densities",
        ),
    ],
)
def test_batched_markov_input_validation(
    prior: np.ndarray,
    log_inlier: np.ndarray,
    log_outlier: np.ndarray,
    ids: list[str],
    times: list[int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        markov_log_evidence_batch(
            prior,
            log_inlier,
            log_outlier,
            ids,
            times,
        )


def test_mixed_type_sequence_ids_fail_instead_of_merging_tracks() -> None:
    with pytest.raises(ValueError, match="distinct typed identities"):
        smooth_markov_reliability(
            np.full(2, 0.5),
            np.zeros(2),
            np.zeros(2),
            [1, "1"],
            [0, 1],
        )
    with pytest.raises(ValueError, match="distinct typed identities"):
        markov_log_evidence_batch(
            np.full(2, 0.5),
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            ["1", 1],
            [0, 1],
        )


def test_integer_sequence_ids_retain_the_historical_string_output_key() -> None:
    result = smooth_markov_reliability(
        np.array([0.5]),
        np.array([0.0]),
        np.array([0.0]),
        [np.int64(7)],
        [0],
    )

    assert set(result.sequence_log_evidence) == {"7"}


@pytest.mark.parametrize("sequence_id", [True, np.bool_(False), 1.5, object(), ""])
def test_sequence_ids_fail_closed_on_unsupported_values(sequence_id: object) -> None:
    with pytest.raises((TypeError, ValueError), match="sequence_ids"):
        smooth_markov_reliability(
            np.array([0.5]),
            np.array([0.0]),
            np.array([0.0]),
            [sequence_id],  # type: ignore[list-item]
            [0],
        )


@pytest.mark.parametrize("prior", [-0.1, 1.1])
def test_prior_reliability_outside_probability_range_is_rejected(
    prior: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        smooth_markov_reliability(
            np.array([prior]),
            np.zeros(1),
            np.zeros(1),
            ["track"],
            [0],
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        markov_log_evidence_batch(
            np.array([prior]),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            ["track"],
            [0],
        )


def test_falsey_invalid_config_is_not_replaced_by_the_default() -> None:
    with pytest.raises(TypeError, match="MarkovReliabilityConfig"):
        smooth_markov_reliability(
            np.array([0.5]),
            np.zeros(1),
            np.zeros(1),
            ["track"],
            [0],
            config=0,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="MarkovReliabilityConfig"):
        markov_log_evidence_batch(
            np.array([0.5]),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            ["track"],
            [0],
            config=False,  # type: ignore[arg-type]
        )


def test_markov_result_defensively_owns_and_freezes_outputs() -> None:
    posterior = np.array([0.25, 0.75])
    evidence = {"track": -1.5}
    result = MarkovReliabilityResult(posterior, evidence)

    posterior[:] = 0.5
    evidence["track"] = 0.0

    np.testing.assert_array_equal(
        result.posterior_inlier_probability,
        [0.25, 0.75],
    )
    assert result.sequence_log_evidence == {"track": -1.5}
    with pytest.raises(ValueError):
        result.posterior_inlier_probability[0] = 0.0
    with pytest.raises(TypeError):
        result.sequence_log_evidence["track"] = 0.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("posterior", "evidence", "error", "message"),
    [
        (np.zeros((1, 1)), {"track": 0.0}, ValueError, "nonempty vector"),
        (np.array([np.nan]), {"track": 0.0}, ValueError, "finite values"),
        (np.array([1.1]), {"track": 0.0}, ValueError, r"\[0, 1\]"),
        (np.array([0.5]), [], TypeError, "mapping"),
        (np.array([0.5]), {"": 0.0}, ValueError, "nonempty strings"),
        (np.array([0.5]), {"track": np.inf}, ValueError, "finite numbers"),
        (np.array([0.5]), {"track": True}, ValueError, "finite numbers"),
        (np.array([0.5]), {"track": object()}, ValueError, "finite numbers"),
    ],
)
def test_markov_result_rejects_invalid_manual_construction(
    posterior: np.ndarray,
    evidence: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        MarkovReliabilityResult(
            posterior,
            evidence,  # type: ignore[arg-type]
        )


def test_integer_step_mode_rejects_unrepresentable_scaled_gap() -> None:
    with pytest.raises(ValueError, match="supported integer-step range"):
        smooth_markov_reliability(
            np.array([0.5, 0.5]),
            np.zeros(2),
            np.zeros(2),
            ["track", "track"],
            [0.0, 1e20],
            config=MarkovReliabilityConfig(time_delta_mode="integer-steps"),
        )
