from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.simulation_based_calibration import (
    SIMULATION_BASED_CALIBRATION_SCHEMA,
    SimulationBasedCalibrationSummaryV1,
    posterior_pit_matrix,
    weighted_randomized_pit,
)


def test_weighted_randomized_pit_retains_weighted_tie_mass() -> None:
    value = weighted_randomized_pit(
        [0.0, 1.0, 1.0, 2.0],
        1.0,
        weights=[0.1, 0.2, 0.3, 0.4],
        tie_breaker=0.25,
    )
    assert value == pytest.approx(0.225)


def test_weighted_randomized_pit_validates_probability_inputs() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_randomized_pit([0.0, 1.0], 0.5, weights=[1.0, -1.0])
    with pytest.raises(ValueError, match="positive finite total"):
        weighted_randomized_pit([0.0, 1.0], 0.5, weights=[0.0, 0.0])
    with pytest.raises(ValueError, match="tie_breaker"):
        weighted_randomized_pit([0.0, 1.0], 0.5, tie_breaker=1.1)
    with pytest.raises(ValueError, match="nonnegative"):
        weighted_randomized_pit(
            [0.0, 1.0],
            0.5,
            absolute_tolerance=-1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        weighted_randomized_pit([0.0, np.nan], 0.5)


def test_posterior_pit_matrix_supports_shared_and_per_group_weights() -> None:
    samples = np.asarray(
        [
            [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
            [[1.0, -1.0], [2.0, 0.0], [3.0, 1.0]],
        ]
    )
    truths = np.asarray([[1.0, 2.5], [2.5, 0.0]])
    ties = np.asarray([[0.25, 0.5], [0.5, 0.75]])

    shared = posterior_pit_matrix(
        samples,
        truths,
        weights=np.asarray([1.0, 2.0, 1.0]),
        tie_breakers=ties,
    )
    per_group = posterior_pit_matrix(
        samples,
        truths,
        weights=np.asarray([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]]),
        tie_breakers=ties,
    )

    np.testing.assert_allclose(shared, per_group)
    np.testing.assert_allclose(shared[0], [0.375, 0.75])
    np.testing.assert_allclose(shared[1], [0.75, 0.625])
    assert not shared.flags.writeable
    with pytest.raises(ValueError):
        shared.setflags(write=True)


def test_posterior_pit_matrix_rejects_shape_and_tie_drift() -> None:
    samples = np.zeros((2, 3, 1))
    truths = np.zeros((2, 1))
    with pytest.raises(ValueError, match="truths"):
        posterior_pit_matrix(samples, np.zeros((1, 1)))
    with pytest.raises(ValueError, match="weights"):
        posterior_pit_matrix(samples, truths, weights=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="tie_breakers"):
        posterior_pit_matrix(samples, truths, tie_breakers=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="tie_breakers"):
        posterior_pit_matrix(samples, truths, tie_breakers=np.full((2, 1), 2.0))


def _uniform_summary() -> SimulationBasedCalibrationSummaryV1:
    midpoints = (np.arange(10, dtype=np.float64) + 0.5) / 10.0
    return SimulationBasedCalibrationSummaryV1(
        group_ids=tuple(f"simulation-{index}" for index in range(10)),
        parameter_names=("stiffness", "damping"),
        pit_values=np.column_stack((midpoints, midpoints[::-1])),
        bin_count=10,
        metadata={"generator_revision": "a" * 40, "target_outcomes_used": False},
    )


def test_summary_reports_uniformity_without_claiming_a_p_value() -> None:
    summary = _uniform_summary()

    np.testing.assert_array_equal(summary.histogram_counts, np.ones((2, 10)))
    np.testing.assert_allclose(summary.mean_pit, [0.5, 0.5])
    np.testing.assert_allclose(summary.ks_distance, [0.05, 0.05])
    np.testing.assert_allclose(summary.central_50_coverage, [0.6, 0.6])
    np.testing.assert_allclose(summary.central_90_coverage, [1.0, 1.0])
    np.testing.assert_allclose(summary.central_95_coverage, [1.0, 1.0])
    np.testing.assert_allclose(summary.tail_5_rates, np.zeros((2, 2)))
    assert summary.expected_histogram_count == 1.0
    assert summary.independent_group_count == 10
    assert summary.parameter_count == 2

    record = summary.to_record()
    assert record["schema"] == SIMULATION_BASED_CALIBRATION_SCHEMA
    assert record["artifact_id"] == summary.artifact_id
    assert "p_value" not in record


def test_summary_is_content_addressed_and_irreversibly_immutable() -> None:
    source = np.column_stack(
        (
            (np.arange(10, dtype=np.float64) + 0.5) / 10.0,
            (np.arange(10, dtype=np.float64) + 0.5) / 10.0,
        )
    )
    summary = SimulationBasedCalibrationSummaryV1(
        group_ids=tuple(f"simulation-{index}" for index in range(10)),
        parameter_names=("x", "y"),
        pit_values=source,
        metadata={"nested": {"seed": 7}},
    )
    artifact_id = summary.artifact_id
    source[:] = 0.0
    assert summary.artifact_id == artifact_id
    assert summary.metadata["nested"]["seed"] == 7

    for values in (
        summary.pit_values,
        summary.histogram_counts,
        summary.mean_pit,
        summary.ks_distance,
        summary.cramer_von_mises,
        summary.central_50_coverage,
        summary.central_90_coverage,
        summary.central_95_coverage,
        summary.tail_5_rates,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.setflags(write=True)

    restored = replace(summary, artifact_id=artifact_id)
    assert restored.artifact_id == artifact_id
    with pytest.raises(ValueError, match="does not match"):
        replace(summary, artifact_id="0" * 64, pit_values=summary.pit_values * 0.5)


def test_summary_rejects_nonindependent_or_invalid_inputs() -> None:
    values = np.asarray([[0.25], [0.75]])
    with pytest.raises(ValueError, match="unique independent"):
        SimulationBasedCalibrationSummaryV1(
            group_ids=("same", "same"),
            parameter_names=("x",),
            pit_values=values,
        )
    with pytest.raises(ValueError, match="parameter_names"):
        SimulationBasedCalibrationSummaryV1(
            group_ids=("a", "b"),
            parameter_names=("x", "x"),
            pit_values=np.column_stack((values, values)),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        SimulationBasedCalibrationSummaryV1(
            group_ids=("a", "b"),
            parameter_names=("x",),
            pit_values=np.asarray([[-0.1], [0.5]]),
        )
    with pytest.raises(ValueError, match="bin_count"):
        SimulationBasedCalibrationSummaryV1(
            group_ids=("a", "b"),
            parameter_names=("x",),
            pit_values=values,
            bin_count=True,
        )
