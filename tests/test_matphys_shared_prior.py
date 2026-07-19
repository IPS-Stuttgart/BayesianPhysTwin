import numpy as np
import pytest

from bayesian_phystwin.matphys_shared_prior import (
    MATPHYS_MATERIAL_NAMES,
    assess_matphys_prediction_competence,
    build_matphys_spring_direction,
    material_distribution_from_weights,
    validate_material_distributions,
)


def test_named_material_weights_follow_matphys_class_order():
    row = material_distribution_from_weights({"fabric": 3.0, "fur": 1.0})

    assert row.shape == (10,)
    assert row[MATPHYS_MATERIAL_NAMES.index("fabric")] == pytest.approx(0.75)
    assert row[MATPHYS_MATERIAL_NAMES.index("fur")] == pytest.approx(0.25)
    assert row.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "values, message",
    [
        (np.ones((2, 9)), "shape"),
        (np.zeros((2, 10)), "positive mass"),
        (np.array([[1.0] + [-0.1] * 9]), "nonnegative"),
        (np.array([[np.nan] + [0.0] * 9]), "finite"),
    ],
)
def test_material_distribution_validation_fails_closed(values, message):
    with pytest.raises(ValueError, match=message):
        validate_material_distributions(values)


def test_material_distribution_validation_normalizes_each_part():
    values = np.zeros((2, 10), dtype=np.float64)
    values[0, 0] = 2.0
    values[1, 7] = 1.0
    values[1, 8] = 3.0

    normalized = validate_material_distributions(values, expected_parts=2)

    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
    np.testing.assert_allclose(normalized[1, [7, 8]], [0.25, 0.75])


def test_matphys_direction_has_exact_teacher_and_prediction_endpoints():
    teacher = np.log(np.array([1.0e4, 2.0e4, 3.0e4, 8.0e4]))
    predicted_object = np.log(np.array([2.0e4, 1.0e4, 4.5e4]))
    predicted_controller = np.log(np.array([4.0e4]))

    direction = build_matphys_spring_direction(
        teacher_log_y=teacher,
        predicted_object_log_y=predicted_object,
        predicted_controller_log_y=predicted_controller,
        object_spring_count=3,
    )

    np.testing.assert_array_equal(direction.reconstruct(teacher, 0.0), teacher)
    np.testing.assert_allclose(
        direction.reconstruct(teacher, direction.prior_coefficient),
        np.concatenate((predicted_object, predicted_controller)),
        rtol=1e-7,
        atol=1e-7,
    )
    assert np.max(np.abs(direction.weights)) == pytest.approx(1.0)
    assert direction.diagnostics()["controller_spring_count"] == 1


def test_matphys_direction_rejects_spring_count_mismatch_and_zero_direction():
    with pytest.raises(ValueError, match="different spring counts"):
        build_matphys_spring_direction(
            teacher_log_y=[1.0, 2.0],
            predicted_object_log_y=[1.0],
        )
    with pytest.raises(ValueError, match="indistinguishable"):
        build_matphys_spring_direction(
            teacher_log_y=[1.0, 2.0],
            predicted_object_log_y=[1.0, 2.0],
        )


def test_competence_gate_rejects_lower_bound_collapse():
    teacher = np.log(np.array([2.0e3, 4.0e3, 8.0e3, 1.6e4]))
    predicted = np.full(4, np.log(1.0e3))

    result = assess_matphys_prediction_competence(
        teacher_object_log_y=teacher,
        predicted_object_log_y=predicted,
    )

    assert result.competent_direction is False
    assert result.failure_reasons == (
        "lower-bound-saturation",
        "spatially-constant-output",
    )
    assert result.lower_bound_fraction == pytest.approx(1.0)
    assert result.direction_teacher_correlation == pytest.approx(-1.0)


def test_competence_gate_accepts_nondegenerate_spatial_prediction():
    teacher = np.log(np.array([2.0e3, 4.0e3, 8.0e3, 1.6e4]))
    predicted = np.log(np.array([3.0e3, 5.0e3, 7.0e3, 1.2e4]))

    result = assess_matphys_prediction_competence(
        teacher_object_log_y=teacher,
        predicted_object_log_y=predicted,
    )

    assert result.competent_direction is True
    assert result.failure_reasons == ()
    assert result.predicted_log_std > result.minimum_spatial_log_std


def test_competence_gate_rejects_invalid_bounds_and_shapes():
    with pytest.raises(ValueError, match="same spring field"):
        assess_matphys_prediction_competence(
            teacher_object_log_y=[1.0, 2.0],
            predicted_object_log_y=[1.0],
        )
    with pytest.raises(ValueError, match="ordered"):
        assess_matphys_prediction_competence(
            teacher_object_log_y=[1.0, 2.0],
            predicted_object_log_y=[1.0, 2.0],
            stiffness_minimum=1.0e3,
            stiffness_maximum=1.0e3,
        )
