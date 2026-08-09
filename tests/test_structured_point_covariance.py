from __future__ import annotations

import operator
from copy import deepcopy

import numpy as np
import pytest

import bayesian_phystwin.structured_point_covariance as structured
from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.structured_point_covariance import (
    StructuredPointCovarianceV1,
)


def _local() -> np.ndarray:
    return np.asarray(
        [
            [[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.16]],
            [[0.25, 0.0, 0.02], [0.0, 0.36, 0.0], [0.02, 0.0, 0.49]],
        ],
        dtype=np.float64,
    )


def _factors() -> dict[str, np.ndarray]:
    discrepancy = np.zeros((2, 3, 1), dtype=np.float64)
    discrepancy[0, :, 0] = [0.10, 0.02, 0.00]
    discrepancy[1, :, 0] = [0.20, -0.01, 0.03]

    gauge = np.zeros((2, 3, 2), dtype=np.float64)
    gauge[0, :, 0] = [0.03, 0.00, 0.01]
    gauge[1, :, 0] = [0.03, 0.00, 0.01]
    gauge[0, :, 1] = [0.00, 0.04, 0.00]
    gauge[1, :, 1] = [0.00, 0.05, 0.00]
    return {"gauge": gauge, "discrepancy": discrepancy}


def _covariance(**changes: object) -> StructuredPointCovarianceV1:
    values: dict[str, object] = {
        "point_ids": ("point-0", "point-1"),
        "local_covariance_m2": _local(),
        "shared_factors_m": _factors(),
        "coordinate_frame": "phystwin-world",
        "source_artifact_id": "a" * 64,
        "metadata": {"selection": "source-only", "rank_trace": 0.999},
    }
    values.update(changes)
    return StructuredPointCovarianceV1(**values)


def test_structured_covariance_is_content_addressed_and_immutable() -> None:
    local = _local()
    factors = _factors()
    covariance = _covariance(
        local_covariance_m2=local,
        shared_factors_m=factors,
        calibration_artifact_id="b" * 64,
    )

    local[:] = 100.0
    factors["gauge"][:] = 100.0

    assert covariance.point_count == 2
    assert covariance.state_dimension == 6
    assert covariance.shared_rank == 3
    assert covariance.shared_component_names == ("discrepancy", "gauge")
    assert covariance.artifact_id == content_id(covariance.descriptor())
    assert covariance.summary() == {
        "schema": structured.STRUCTURED_POINT_COVARIANCE_SCHEMA,
        "schema_version": structured.STRUCTURED_POINT_COVARIANCE_VERSION,
        "artifact_id": covariance.artifact_id,
        "point_count": 2,
        "state_dimension": 6,
        "shared_rank": 3,
        "shared_component_ranks": {"discrepancy": 1, "gauge": 2},
        "coordinate_frame": "phystwin-world",
        "source_artifact_id": "a" * 64,
        "calibration_artifact_id": "b" * 64,
        "local_blocks_exclude_shared_components": True,
    }
    assert not covariance.local_covariance_m2.flags.writeable
    assert not covariance.shared_factors_m["gauge"].flags.writeable
    with pytest.raises(TypeError):
        operator.setitem(
            covariance.shared_factors_m,
            "process",
            np.zeros((2, 3, 1)),
        )


def test_content_identity_changes_with_each_covariance_component() -> None:
    baseline = _covariance()

    changed_local = _local()
    changed_local[0, 0, 0] += 0.01
    changed_factors = _factors()
    changed_factors["gauge"][0, 0, 0] += 0.01

    variants = (
        _covariance(local_covariance_m2=changed_local),
        _covariance(shared_factors_m=changed_factors),
        _covariance(coordinate_frame="other-world"),
        _covariance(source_artifact_id="c" * 64),
        _covariance(calibration_artifact_id="d" * 64),
        _covariance(metadata={"selection": "different"}),
    )
    assert all(item.artifact_id != baseline.artifact_id for item in variants)

    replay = _covariance(artifact_id=baseline.artifact_id)
    assert replay.artifact_id == baseline.artifact_id
    with pytest.raises(ValueError, match="artifact_id does not match"):
        _covariance(artifact_id="0" * 64)


def test_marginal_and_cross_covariances_match_dense_representation() -> None:
    covariance = _covariance()
    dense = covariance.dense_covariance_m2()
    marginal = covariance.marginal_covariance_m2()

    np.testing.assert_allclose(marginal[0], dense[:3, :3])
    np.testing.assert_allclose(marginal[1], dense[3:, 3:])
    np.testing.assert_allclose(
        covariance.cross_covariance_m2("point-0", "point-1"),
        dense[:3, 3:],
    )
    np.testing.assert_allclose(
        covariance.cross_covariance_m2("point-1", "point-1"),
        dense[3:, 3:],
    )
    assert not dense.flags.writeable
    assert not marginal.flags.writeable

    with pytest.raises(ValueError, match="point ID is not present"):
        covariance.cross_covariance_m2("point-0", "missing")
    with pytest.raises(ValueError, match="literal string"):
        covariance.cross_covariance_m2(" point-0", "point-1")


def test_query_projection_matches_dense_covariance_and_decomposes_components() -> None:
    covariance = _covariance()
    jacobian = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 0.25], [0.0, 0.0, 0.75]],
        ],
        dtype=np.float64,
    )

    projected = covariance.project_query_covariance(jacobian)
    flat = jacobian.reshape((3, 6))
    expected = flat @ covariance.dense_covariance_m2() @ flat.T

    np.testing.assert_allclose(projected.total_covariance_m2, expected)
    summed = np.array(projected.local_covariance_m2, copy=True)
    for name in projected.shared_component_names:
        factor = projected.shared_component_factors_m[name]
        component = projected.shared_component_covariances_m2[name]
        np.testing.assert_allclose(component, factor @ factor.T)
        summed += component
    np.testing.assert_allclose(summed, projected.total_covariance_m2)
    assert projected.query_dimension == 3
    assert projected.shared_component_names == ("discrepancy", "gauge")
    assert not projected.total_covariance_m2.flags.writeable

    flat_projected = covariance.project_query_covariance(flat)
    np.testing.assert_allclose(
        flat_projected.total_covariance_m2,
        projected.total_covariance_m2,
    )


def test_empty_shared_factor_mapping_reduces_to_block_diagonal_covariance() -> None:
    covariance = _covariance(shared_factors_m={})
    dense = covariance.dense_covariance_m2()

    assert covariance.shared_rank == 0
    assert covariance.shared_component_names == ()
    np.testing.assert_allclose(dense[:3, :3], _local()[0])
    np.testing.assert_allclose(dense[3:, 3:], _local()[1])
    np.testing.assert_allclose(dense[:3, 3:], 0.0)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"point_ids": ("point-0", "point-0")}, "unique"),
        ({"point_ids": ()}, "must not be empty"),
        ({"coordinate_frame": " padded "}, "literal string"),
        ({"source_artifact_id": "a" * 63}, "source_artifact_id"),
        ({"calibration_artifact_id": "b" * 63}, "calibration_artifact_id"),
        ({"metadata": {"value": float("nan")}}, "finite"),
    ],
)
def test_metadata_and_identity_contracts_fail_closed(
    changes: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _covariance(**changes)


@pytest.mark.parametrize(
    ("local", "match"),
    [
        (np.zeros((2, 3)), "shape"),
        (np.zeros((1, 3, 3)), "shape"),
        (np.asarray([[[True] * 3] * 3] * 2), "real numeric"),
        (np.full((2, 3, 3), np.nan), "finite"),
        (
            np.asarray(
                [
                    [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    np.eye(3),
                ]
            ),
            "symmetric",
        ),
        (
            np.asarray(
                [
                    np.diag([-1.0, 1.0, 1.0]),
                    np.eye(3),
                ]
            ),
            "positive semidefinite",
        ),
    ],
)
def test_local_covariance_contract_fails_closed(
    local: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _covariance(local_covariance_m2=local)


def test_tiny_roundoff_asymmetry_and_negative_eigenvalue_are_admitted() -> None:
    local = _local()
    local[0, 0, 1] += 1e-13
    local[1] = np.diag([-1e-13, 1.0, 1.0])

    covariance = _covariance(local_covariance_m2=local)

    np.testing.assert_allclose(
        covariance.local_covariance_m2,
        np.swapaxes(covariance.local_covariance_m2, 1, 2),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("factors", "match"),
    [
        ([], "must be a mapping"),
        ({"unknown": np.zeros((2, 3, 1))}, "must be one of"),
        ({"gauge": np.zeros((2, 3))}, "shape"),
        ({"gauge": np.zeros((1, 3, 1))}, "shape"),
        ({"gauge": np.zeros((2, 3, 0))}, "positive retained rank"),
        ({"gauge": np.full((2, 3, 1), np.inf)}, "finite"),
        ({"gauge": np.ones((2, 3, 1), dtype=bool)}, "real numeric"),
    ],
)
def test_shared_factor_contract_fails_closed(
    factors: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _covariance(shared_factors_m=factors)


@pytest.mark.parametrize(
    ("jacobian", "match"),
    [
        (np.zeros(6), "must have shape"),
        (np.zeros((1, 5)), "flat query_jacobian"),
        (np.zeros((1, 1, 3)), "point_count"),
        (np.zeros((0, 2, 3)), "at least one"),
        (np.full((1, 2, 3), np.nan), "finite"),
        (np.ones((1, 2, 3), dtype=bool), "real numeric"),
    ],
)
def test_query_jacobian_contract_fails_closed(
    jacobian: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _covariance().project_query_covariance(jacobian)


def test_dense_materialization_requires_explicit_bounded_dimension() -> None:
    covariance = _covariance()

    with pytest.raises(ValueError, match="dimension limit"):
        covariance.dense_covariance_m2(maximum_dimension=5)
    with pytest.raises(ValueError, match="integer"):
        covariance.dense_covariance_m2(maximum_dimension=True)


def test_descriptor_is_order_invariant_over_shared_component_mapping() -> None:
    first = _covariance(shared_factors_m=_factors())
    reversed_factors = dict(reversed(list(_factors().items())))
    second = _covariance(shared_factors_m=reversed_factors)

    assert first.artifact_id == second.artifact_id
    assert first.descriptor() == second.descriptor()


def test_all_declared_shared_component_labels_are_admitted() -> None:
    factors = {
        name: np.full((2, 3, 1), index / 100.0)
        for index, name in enumerate(
            structured.SHARED_COVARIANCE_COMPONENTS,
            start=1,
        )
    }

    covariance = _covariance(shared_factors_m=factors)

    assert covariance.shared_component_names == tuple(
        sorted(structured.SHARED_COVARIANCE_COMPONENTS)
    )
    assert covariance.shared_rank == len(structured.SHARED_COVARIANCE_COMPONENTS)


def test_descriptor_is_detached_from_mutable_metadata() -> None:
    metadata = {"nested": {"values": [1, 2, 3]}}
    covariance = _covariance(metadata=metadata)
    descriptor = deepcopy(covariance.descriptor())

    metadata["nested"]["values"].append(4)

    assert covariance.descriptor() == descriptor
