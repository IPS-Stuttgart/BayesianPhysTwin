from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.structured_point_covariance import (
    StructuredPointCovarianceV1,
)
from bayesian_phystwin.structured_point_covariance_operator_v1 import (
    STRUCTURED_POINT_COVARIANCE_OPERATOR_CLAIM_BOUNDARY,
    STRUCTURED_POINT_COVARIANCE_OPERATOR_SCHEMA,
    StructuredPointCovarianceOperatorV1,
)


def _covariance(*, singular_local: bool = False) -> StructuredPointCovarianceV1:
    first = np.diag([0.7, 0.8, 0.9])
    if singular_local:
        first[0, 0] = 0.0
    local = np.stack((first, np.diag([1.1, 1.2, 1.3])))
    gauge = np.zeros((2, 3, 1), dtype=np.float64)
    gauge[:, 0, 0] = [0.2, -0.1]
    process = np.zeros((2, 3, 2), dtype=np.float64)
    process[0, 1, 0] = 0.3
    process[1, 2, 1] = 0.4
    return StructuredPointCovarianceV1(
        point_ids=("point:0", "point:1"),
        local_covariance_m2=local,
        shared_factors_m={"gauge": gauge, "process": process},
        coordinate_frame="world",
        source_artifact_id="1" * 64,
        calibration_artifact_id="2" * 64,
        metadata={"source": "unit-test"},
    )


def _operator(*, singular_local: bool = False) -> StructuredPointCovarianceOperatorV1:
    return StructuredPointCovarianceOperatorV1(
        _covariance(singular_local=singular_local),
        metadata={"consumer": "Causal4D"},
    )


def test_operator_is_content_addressed_and_immutable() -> None:
    covariance = _covariance()
    operator = StructuredPointCovarianceOperatorV1(
        covariance,
        metadata={"consumer": "Causal4D"},
    )
    duplicate = StructuredPointCovarianceOperatorV1(
        covariance,
        metadata={"consumer": "Causal4D"},
        artifact_id=operator.artifact_id,
    )

    assert duplicate.artifact_id == operator.artifact_id
    assert operator.point_count == 2
    assert operator.state_dimension == 6
    assert operator.shared_rank == 3
    assert operator.shared_component_names == ("gauge", "process")
    assert operator.supports_woodbury_solve is True
    assert operator.to_record()["artifact_id"] == operator.artifact_id
    assert operator.summary()["schema"] == STRUCTURED_POINT_COVARIANCE_OPERATOR_SCHEMA
    assert operator.summary()["claim_boundary"] == (
        STRUCTURED_POINT_COVARIANCE_OPERATOR_CLAIM_BOUNDARY
    )
    with pytest.raises(TypeError):
        operator.metadata["consumer"] = "tampered"  # type: ignore[index]


def test_operator_identity_binds_covariance_and_metadata() -> None:
    covariance = _covariance()
    first = StructuredPointCovarianceOperatorV1(covariance, metadata={"arm": "a"})
    second = StructuredPointCovarianceOperatorV1(covariance, metadata={"arm": "b"})
    changed_covariance = StructuredPointCovarianceV1(
        point_ids=covariance.point_ids,
        local_covariance_m2=covariance.local_covariance_m2,
        shared_factors_m=covariance.shared_factors_m,
        coordinate_frame=covariance.coordinate_frame,
        source_artifact_id="7" * 64,
        calibration_artifact_id=covariance.calibration_artifact_id,
        metadata=covariance.metadata,
    )
    changed = StructuredPointCovarianceOperatorV1(
        changed_covariance,
        metadata={"arm": "a"},
    )

    assert first.artifact_id != second.artifact_id
    assert first.artifact_id != changed.artifact_id
    with pytest.raises(ValueError, match="does not match content"):
        StructuredPointCovarianceOperatorV1(
            covariance,
            artifact_id="0" * 64,
        )
    with pytest.raises(TypeError, match="StructuredPointCovarianceV1"):
        StructuredPointCovarianceOperatorV1(cast(Any, object()))
    with pytest.raises(ValueError, match="metadata"):
        StructuredPointCovarianceOperatorV1(covariance, metadata=cast(Any, 1))


def test_matrix_actions_match_bounded_dense_covariance() -> None:
    operator = _operator()
    dense = operator.covariance.dense_covariance_m2()
    vector = np.arange(1.0, 7.0)
    matrix = np.column_stack((vector, vector[::-1], np.ones(6)))

    np.testing.assert_allclose(operator.matmul(vector), dense @ vector)
    np.testing.assert_allclose(operator.matmul(matrix), dense @ matrix)
    assert operator.matmul(vector).flags.writeable is False
    assert operator.matmul(matrix).flags.writeable is False

    component_sum = operator.component_matmul("local", matrix)
    for component in operator.shared_component_names:
        component_sum = component_sum + operator.component_matmul(component, matrix)
    np.testing.assert_allclose(component_sum, operator.matmul(matrix))

    np.testing.assert_allclose(
        operator.quadratic_form(vector),
        vector @ dense @ vector,
    )


def test_component_actions_fail_closed_on_unknown_or_invalid_names() -> None:
    operator = _operator()
    vector = np.ones(operator.state_dimension)

    with pytest.raises(ValueError, match="nonempty literal text"):
        operator.component_matmul(cast(Any, 1), vector)
    with pytest.raises(ValueError, match="retained shared component"):
        operator.component_matmul("missing", vector)
    with pytest.raises(ValueError, match="requires one vector"):
        operator.quadratic_form(np.ones((operator.state_dimension, 1)))


@pytest.mark.parametrize(
    ("rhs", "match"),
    [
        (np.ones(5), "first dimension"),
        (np.ones((6, 1, 1)), "must have shape"),
        (np.ones((6, 0)), "at least one right-hand side"),
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]), "must be finite"),
        (np.array(["x"] * 6), "real numeric"),
    ],
)
def test_matrix_actions_reject_malformed_right_hand_sides(
    rhs: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _operator().matmul(rhs)
