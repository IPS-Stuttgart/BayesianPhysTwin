import numpy as np
import pytest

from bayesian_phystwin import (
    RandomWalkBiasConfig,
    RandomWalkBiasResult,
    filter_random_walk_bias,
    robust_random_walk_log_evidence_batch,
)


def test_random_walk_bias_tracks_gradual_drift_without_following_gross_outlier() -> (
    None
):
    rng = np.random.default_rng(2)
    drift = np.concatenate([np.zeros(10), np.linspace(0.0, 0.06, 20)])
    residual = drift + rng.normal(scale=0.003, size=drift.size)
    residual[15] += 0.20
    prior = np.full(drift.size, 0.95)
    prior[15] = 0.05

    result = filter_random_walk_bias(
        prior,
        residual,
        0.003**2,
        sequence_ids=["track"] * drift.size,
        time_values=np.arange(drift.size),
        config=RandomWalkBiasConfig(
            process_variance=2.5e-5,
            initial_variance=1e-6,
        ),
        bias_probability=np.concatenate([np.zeros(10), np.ones(20)]),
    )

    assert abs(result.bias_mean[-1] - drift[-1]) < 0.01
    assert result.inlier_probability[15] < 0.1
    assert abs(result.bias_mean[15] - result.bias_mean[14]) < 0.03
    assert np.isfinite(result.total_log_evidence)


def test_batched_bias_evidence_prefers_correct_shared_physics_hypothesis() -> None:
    time = np.arange(25)
    drift = np.linspace(0.0, 0.04, time.size)
    correct = drift
    wrong = drift + 0.08
    residual = np.vstack([correct, wrong])

    evidence = robust_random_walk_log_evidence_batch(
        np.full(time.size, 0.9),
        residual,
        0.005**2,
        sequence_ids=["track"] * time.size,
        time_values=time,
    )

    assert evidence.shape == (2,)
    assert evidence[0] > evidence[1]


def test_multiple_sequences_have_independent_bias_states() -> None:
    result = filter_random_walk_bias(
        np.full(6, 0.9),
        np.array([0.0, 0.01, 0.02, 0.0, -0.01, -0.02]),
        0.002**2,
        sequence_ids=["a", "a", "a", "b", "b", "b"],
        time_values=[0, 1, 2, 0, 1, 2],
    )

    assert result.bias_mean[2] > 0.0
    assert result.bias_mean[5] < 0.0
    assert set(result.sequence_log_evidence) == {"a", "b"}


def test_mixed_type_sequence_ids_fail_instead_of_merging_bias_states() -> None:
    with pytest.raises(ValueError, match="distinct typed identities"):
        filter_random_walk_bias(
            np.full(2, 0.9),
            np.zeros(2),
            1e-4,
            sequence_ids=[1, "1"],
            time_values=[0, 1],
        )
    with pytest.raises(ValueError, match="distinct typed identities"):
        robust_random_walk_log_evidence_batch(
            np.full(2, 0.9),
            np.zeros((1, 2)),
            1e-4,
            sequence_ids=["1", 1],
            time_values=[0, 1],
        )


def test_integer_sequence_id_retains_historical_string_evidence_key() -> None:
    result = filter_random_walk_bias(
        np.asarray([0.9]),
        np.asarray([0.0]),
        1e-4,
        sequence_ids=[np.int64(7)],
        time_values=[0],
    )

    assert set(result.sequence_log_evidence) == {"7"}


@pytest.mark.parametrize("sequence_id", [True, np.bool_(False), 1.5, object(), ""])
def test_bias_filter_rejects_unsupported_sequence_ids(sequence_id: object) -> None:
    with pytest.raises((TypeError, ValueError), match="sequence_ids"):
        filter_random_walk_bias(
            np.asarray([0.9]),
            np.asarray([0.0]),
            1e-4,
            sequence_ids=[sequence_id],  # type: ignore[list-item]
            time_values=[0],
        )


@pytest.mark.parametrize("prior", [-0.1, 1.1])
def test_bias_filter_rejects_prior_reliability_outside_probability_range(
    prior: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        filter_random_walk_bias(
            np.asarray([prior]),
            np.asarray([0.0]),
            1e-4,
            sequence_ids=["track"],
            time_values=[0],
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        robust_random_walk_log_evidence_batch(
            np.asarray([prior]),
            np.zeros((1, 1)),
            1e-4,
            sequence_ids=["track"],
            time_values=[0],
        )


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_bias_filter_rejects_bias_probability_outside_range(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        filter_random_walk_bias(
            np.asarray([0.9]),
            np.asarray([0.0]),
            1e-4,
            sequence_ids=["track"],
            time_values=[0],
            bias_probability=np.asarray([probability]),
        )


def test_bias_filter_rejects_nonfinite_bias_probability() -> None:
    with pytest.raises(ValueError, match="bias_probability.*finite"):
        filter_random_walk_bias(
            np.asarray([0.9]),
            np.asarray([0.0]),
            1e-4,
            sequence_ids=["track"],
            time_values=[0],
            bias_probability=np.asarray([np.nan]),
        )


def test_bias_filter_rejects_falsey_invalid_config() -> None:
    with pytest.raises(TypeError, match="RandomWalkBiasConfig"):
        filter_random_walk_bias(
            np.asarray([0.9]),
            np.asarray([0.0]),
            1e-4,
            sequence_ids=["track"],
            time_values=[0],
            config=0,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="RandomWalkBiasConfig"):
        robust_random_walk_log_evidence_batch(
            np.asarray([0.9]),
            np.zeros((1, 1)),
            1e-4,
            sequence_ids=["track"],
            time_values=[0],
            config=False,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (RandomWalkBiasConfig(process_variance=np.nan), "process_variance"),
        (RandomWalkBiasConfig(process_variance=-1.0), "process_variance"),
        (RandomWalkBiasConfig(base_process_variance=np.inf), "base_process_variance"),
        (RandomWalkBiasConfig(initial_variance=np.nan), "initial_variance"),
        (
            RandomWalkBiasConfig(outlier_variance_multiplier=np.inf),
            "outlier_variance_multiplier",
        ),
        (
            RandomWalkBiasConfig(outlier_variance_multiplier=1.0),
            "outlier_variance_multiplier",
        ),
        (RandomWalkBiasConfig(probability_floor=np.nan), "probability_floor"),
        (RandomWalkBiasConfig(probability_floor=0.5), "probability_floor"),
    ],
)
def test_bias_filter_rejects_nonfinite_configuration(
    config: RandomWalkBiasConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        filter_random_walk_bias(
            np.asarray([0.9]),
            np.asarray([0.0]),
            1e-4,
            sequence_ids=["track"],
            time_values=[0],
            config=config,
        )


def test_bias_result_defensively_owns_and_freezes_outputs() -> None:
    mean = np.asarray([0.1, 0.2])
    variance = np.asarray([0.01, 0.02])
    probability = np.asarray([0.8, 0.9])
    evidence = {"track": -1.5}
    result = RandomWalkBiasResult(mean, variance, probability, evidence)

    mean[:] = 0.0
    variance[:] = 0.0
    probability[:] = 0.0
    evidence["track"] = 0.0

    np.testing.assert_array_equal(result.bias_mean, [0.1, 0.2])
    np.testing.assert_array_equal(result.bias_variance, [0.01, 0.02])
    np.testing.assert_array_equal(result.inlier_probability, [0.8, 0.9])
    assert result.sequence_log_evidence == {"track": -1.5}
    with pytest.raises(ValueError):
        result.bias_mean[0] = 0.0
    with pytest.raises(TypeError):
        result.sequence_log_evidence["track"] = 0.0  # type: ignore[index]


def test_batched_bias_evidence_is_read_only() -> None:
    evidence = robust_random_walk_log_evidence_batch(
        np.asarray([0.9]),
        np.zeros((1, 1)),
        1e-4,
        sequence_ids=["track"],
        time_values=[0],
    )

    with pytest.raises(ValueError):
        evidence[0] = 0.0


@pytest.mark.parametrize(
    ("mean", "variance", "probability", "evidence", "error", "message"),
    [
        (np.zeros((1, 1)), np.zeros(1), np.ones(1), {}, ValueError, "vector shape"),
        (np.zeros(0), np.zeros(0), np.zeros(0), {}, ValueError, "nonempty"),
        (np.zeros(1), np.zeros(2), np.ones(1), {}, ValueError, "vector shape"),
        (np.zeros(1), np.zeros(1), np.ones(2), {}, ValueError, "vector shape"),
        (np.asarray([np.nan]), np.zeros(1), np.ones(1), {}, ValueError, "bias_mean"),
        (
            np.zeros(1),
            np.asarray([np.inf]),
            np.ones(1),
            {},
            ValueError,
            "bias_variance",
        ),
        (np.zeros(1), np.asarray([-1.0]), np.ones(1), {}, ValueError, "bias_variance"),
        (np.zeros(1), np.zeros(1), np.asarray([np.nan]), {}, ValueError, r"\[0, 1\]"),
        (np.zeros(1), np.zeros(1), np.asarray([1.1]), {}, ValueError, r"\[0, 1\]"),
        (np.zeros(1), np.zeros(1), np.ones(1), [], TypeError, "mapping"),
        (np.zeros(1), np.zeros(1), np.ones(1), {"": 0.0}, ValueError, "nonempty"),
        (
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            {"track": True},
            ValueError,
            "finite numbers",
        ),
        (
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            {"track": object()},
            ValueError,
            "finite numbers",
        ),
        (
            np.zeros(1),
            np.zeros(1),
            np.ones(1),
            {"track": np.inf},
            ValueError,
            "finite numbers",
        ),
    ],
)
def test_bias_result_rejects_invalid_manual_construction(
    mean: np.ndarray,
    variance: np.ndarray,
    probability: np.ndarray,
    evidence: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        RandomWalkBiasResult(
            mean,
            variance,
            probability,
            evidence,  # type: ignore[arg-type]
        )


def test_bias_filter_rejects_empty_batches() -> None:
    with pytest.raises(ValueError, match="at least one"):
        filter_random_walk_bias(
            np.empty(0),
            np.empty(0),
            1e-4,
            sequence_ids=[],
            time_values=[],
        )
    with pytest.raises(ValueError, match="at least one"):
        robust_random_walk_log_evidence_batch(
            np.empty(0),
            np.empty((0, 0)),
            1e-4,
            sequence_ids=[],
            time_values=[],
        )


@pytest.mark.parametrize("time_value", [np.nan, np.inf, -np.inf])
def test_bias_filter_rejects_nonfinite_numeric_time_values(time_value: float) -> None:
    with pytest.raises(ValueError, match="time_values.*finite"):
        filter_random_walk_bias(
            np.asarray([0.9]),
            np.asarray([0.0]),
            1e-4,
            sequence_ids=["track"],
            time_values=[time_value],
        )


def test_bias_filter_fails_closed_on_nonfinite_likelihood_numerics() -> None:
    with pytest.raises(FloatingPointError, match="likelihood"):
        filter_random_walk_bias(
            np.asarray([0.9]),
            np.asarray([np.finfo(np.float64).max]),
            np.finfo(np.float64).tiny,
            sequence_ids=["track"],
            time_values=[0],
        )


def test_bias_filter_supports_order_only_nonnumeric_times() -> None:
    result = filter_random_walk_bias(
        np.full(2, 0.9),
        np.asarray([0.0, 0.01]),
        1e-4,
        sequence_ids=["track", "track"],
        time_values=["first", "second"],
    )

    assert np.all(np.isfinite(result.bias_mean))


def test_batched_bias_filter_rejects_zero_measurements_with_particles() -> None:
    with pytest.raises(ValueError, match="at least one"):
        robust_random_walk_log_evidence_batch(
            np.empty(0),
            np.empty((1, 0)),
            1e-4,
            sequence_ids=[],
            time_values=[],
        )
