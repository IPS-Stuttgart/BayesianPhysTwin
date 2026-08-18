from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.structured_point_covariance import (
    StructuredPointCovarianceV1,
)
from bayesian_phystwin.structured_point_covariance_operator_v1 import (
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


def test_woodbury_solve_and_logdet_match_dense_without_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operator = _operator()
    dense = operator.covariance.dense_covariance_m2()
    vector = np.arange(1.0, 7.0)
    matrix = np.column_stack((vector, vector[::-1]))
    expected_vector = np.linalg.solve(dense, vector)
    expected_matrix = np.linalg.solve(dense, matrix)
    sign, expected_logdet = np.linalg.slogdet(dense)
    assert sign == 1.0

    def forbidden_inverse(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("explicit inverse must not be used")

    monkeypatch.setattr(np.linalg, "inv", forbidden_inverse)
    np.testing.assert_allclose(operator.solve(vector), expected_vector)
    np.testing.assert_allclose(operator.solve(matrix), expected_matrix)
    assert operator.solve(vector).flags.writeable is False
    assert operator.solve(matrix).flags.writeable is False
    assert operator.logdet() == pytest.approx(expected_logdet)


def test_woodbury_empty_shared_path_matches_block_diagonal() -> None:
    covariance = _covariance()
    block_only = StructuredPointCovarianceV1(
        point_ids=covariance.point_ids,
        local_covariance_m2=covariance.local_covariance_m2,
        shared_factors_m={},
        coordinate_frame=covariance.coordinate_frame,
        source_artifact_id="8" * 64,
    )
    operator = StructuredPointCovarianceOperatorV1(block_only)
    dense = block_only.dense_covariance_m2()
    rhs = np.arange(1.0, 7.0)

    np.testing.assert_allclose(operator.solve(rhs), np.linalg.solve(dense, rhs))
    assert operator.logdet() == pytest.approx(np.linalg.slogdet(dense)[1])
    assert operator.shared_rank == 0


def test_semidefinite_local_blocks_retain_actions_but_reject_woodbury() -> None:
    operator = _operator(singular_local=True)
    vector = np.ones(operator.state_dimension)

    assert operator.supports_woodbury_solve is False
    np.testing.assert_allclose(
        operator.matmul(vector),
        operator.covariance.dense_covariance_m2() @ vector,
    )
    with pytest.raises(ValueError, match="strictly positive-definite"):
        operator.solve(vector)
    with pytest.raises(ValueError, match="strictly positive-definite"):
        operator.logdet()


def test_sampling_is_deterministic_finite_and_component_preserving() -> None:
    operator = _operator(singular_local=True)
    first = operator.sample(np.random.default_rng(17), 4000)
    second = operator.sample(np.random.default_rng(17), 4000)

    assert first.shape == (4000, operator.state_dimension)
    assert first.flags.writeable is False
    np.testing.assert_array_equal(first, second)
    assert np.all(np.isfinite(first))
    empirical = np.cov(first, rowvar=False, bias=True)
    expected = operator.covariance.dense_covariance_m2()
    np.testing.assert_allclose(empirical, expected, rtol=0.12, atol=0.04)

    with pytest.raises(TypeError, match="numpy.random.Generator"):
        operator.sample(cast(Any, object()), 1)
    with pytest.raises(ValueError, match="integer >= 1"):
        operator.sample(np.random.default_rng(1), cast(Any, True))
    with pytest.raises(ValueError, match="integer >= 1"):
        operator.sample(np.random.default_rng(1), 0)


def test_query_projection_delegates_exact_component_decomposition() -> None:
    operator = _operator()
    jacobian = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    projected = operator.project_query_covariance(jacobian)
    expected = jacobian @ operator.covariance.dense_covariance_m2() @ jacobian.T

    np.testing.assert_allclose(projected.total_covariance_m2, expected)
    total = np.array(projected.local_covariance_m2, copy=True)
    for component in projected.shared_component_covariances_m2.values():
        total += component
    np.testing.assert_allclose(total, projected.total_covariance_m2)


def test_matrix_action_detects_nonrepresentable_overflow() -> None:
    factor = np.full((1, 3, 1), 1e150)
    covariance = StructuredPointCovarianceV1(
        point_ids=("point:0",),
        local_covariance_m2=np.eye(3)[None],
        shared_factors_m={"gauge": factor},
        coordinate_frame="world",
        source_artifact_id="9" * 64,
    )
    operator = StructuredPointCovarianceOperatorV1(covariance)

    with pytest.raises(ValueError, match="overflowed finite"):
        operator.matmul(np.full(3, 1e100))
