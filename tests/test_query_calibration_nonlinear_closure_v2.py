from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.nonlinear_closure_certificate_v2 import (
    NONLINEAR_CLOSURE_CERTIFICATE_CLAIM_BOUNDARY,
    NonlinearClosureCertificateV2,
    NonlinearClosureStatus,
)
from bayesian_phystwin.physical_linearization import (
    PhysicalLinearizationV1,
    evaluate_nonlinear_closure,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64


def _linearization(query_count: int = 3) -> PhysicalLinearizationV1:
    state = np.zeros((2, 3, 2), dtype=np.float64)
    state[0, 0, 0] = 1.0
    state[1, 1, 1] = 1.0
    query = np.zeros((query_count, 3, 2), dtype=np.float64)
    for index in range(query_count):
        query[index, index % 2, index % 2] = 1.0 + index
    response = np.zeros((query_count, 3), dtype=np.float64)
    response[:, 0] = np.arange(1, query_count + 1) * 0.01
    return PhysicalLinearizationV1(
        observation_artifact_id=A,
        baseline_belief_id=B,
        action_prefix_id=C,
        simulator_revision="sim-v2",
        frame_ids=np.asarray([0, 1], dtype=np.int64),
        entity_ids=np.asarray([0, 0], dtype=np.int64),
        view_indices=np.asarray([0, 0], dtype=np.int64),
        window_indices=np.asarray([0, 1], dtype=np.int64),
        state_jacobian=state,
        query_state_jacobian=query,
        physical_response_m=response,
    )


def _replays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = np.zeros((3, 3), dtype=np.float64)
    linearized = np.zeros((2, 3, 3), dtype=np.float64)
    linearized[0, :, 0] = [0.01, 0.02, 0.03]
    linearized[1, :, 0] = [-0.01, -0.02, -0.03]
    nonlinear = linearized.copy()
    nonlinear[:, 0, 1] += 0.0002
    nonlinear[:, 1, 1] += 0.0004
    nonlinear[:, 2, 1] += 0.0006
    return baseline, linearized, nonlinear


def _certificate(
    *,
    linearization: PhysicalLinearizationV1 | None = None,
    perturbation_set_id: str = D,
    query_set_id: str = E,
    perturbation_indices: np.ndarray | None = None,
    query_indices: np.ndarray | None = None,
    horizon_indices: np.ndarray | None = None,
    baseline_query_m: np.ndarray | None = None,
    linearized_query_m: np.ndarray | None = None,
    nonlinear_query_m: np.ndarray | None = None,
    absolute_tolerance_m: np.ndarray | None = None,
    relative_tolerance: np.ndarray | None = None,
    prediction_floor_m: float = 1e-6,
    closure_ratio_limit: float = 1.0,
    comparison_tolerance: float = 1e-12,
    metadata: dict[str, Any] | None = None,
    artifact_id: str | None = None,
) -> NonlinearClosureCertificateV2:
    baseline, linearized, nonlinear = _replays()
    return NonlinearClosureCertificateV2(
        linearization=linearization or _linearization(),
        perturbation_set_id=perturbation_set_id,
        query_set_id=query_set_id,
        perturbation_indices=(
            np.asarray([10, 20], dtype=np.int64)
            if perturbation_indices is None
            else perturbation_indices
        ),
        query_indices=(
            np.asarray([100, 200, 300], dtype=np.int64)
            if query_indices is None
            else query_indices
        ),
        horizon_indices=(
            np.asarray([1, 1, 2], dtype=np.int64)
            if horizon_indices is None
            else horizon_indices
        ),
        baseline_query_m=(baseline if baseline_query_m is None else baseline_query_m),
        linearized_query_m=(
            linearized if linearized_query_m is None else linearized_query_m
        ),
        nonlinear_query_m=(
            nonlinear if nonlinear_query_m is None else nonlinear_query_m
        ),
        absolute_tolerance_m=(
            np.asarray([0.001, 0.001, 0.001], dtype=np.float64)
            if absolute_tolerance_m is None
            else absolute_tolerance_m
        ),
        relative_tolerance=(
            np.asarray([0.1, 0.1, 0.1], dtype=np.float64)
            if relative_tolerance is None
            else relative_tolerance
        ),
        prediction_floor_m=prediction_floor_m,
        closure_ratio_limit=closure_ratio_limit,
        comparison_tolerance=comparison_tolerance,
        metadata={} if metadata is None else metadata,
        artifact_id=artifact_id,
    )


def test_nonlinear_closure_v2_passes_and_reports_query_horizon_maxima() -> None:
    certificate = _certificate()

    assert certificate.status is NonlinearClosureStatus.LOCALLY_CLOSED
    assert certificate.locally_closed
    assert certificate.passes_closure_gate
    assert certificate.perturbation_count == 2
    assert certificate.query_count == 3
    assert certificate.horizon_count == 2
    np.testing.assert_array_equal(certificate.unique_horizon_indices, [1, 2])
    np.testing.assert_allclose(
        certificate.per_query_maximum_closure_ratio,
        [0.1, 0.4 / 3.0, 0.15],
    )
    np.testing.assert_allclose(
        certificate.per_horizon_maximum_closure_ratio,
        [0.4 / 3.0, 0.15],
    )
    assert certificate.maximum_closure_ratio == pytest.approx(0.15)
    assert certificate.worst_query_index == 300
    assert certificate.worst_horizon_index == 2
    assert certificate.summary()["claim_boundary"] == (
        NONLINEAR_CLOSURE_CERTIFICATE_CLAIM_BOUNDARY
    )


def test_nonlinear_closure_v2_localizes_worst_failure() -> None:
    baseline, linearized, nonlinear = _replays()
    nonlinear[1, 2, 2] += 0.02

    certificate = _certificate(
        baseline_query_m=baseline,
        linearized_query_m=linearized,
        nonlinear_query_m=nonlinear,
    )

    assert certificate.status is NonlinearClosureStatus.CLOSURE_VIOLATION
    assert not certificate.passes_closure_gate
    assert certificate.worst_perturbation_index == 20
    assert certificate.worst_query_index == 300
    assert certificate.worst_horizon_index == 2
    assert certificate.worst_nonlinear_remainder_m == pytest.approx(
        np.hypot(0.0006, 0.02)
    )
    assert certificate.maximum_closure_ratio > 1.0
    assert certificate.closure_ratio_margin < 0.0


def test_nonlinear_closure_v2_catches_failure_hidden_by_aggregate_norm() -> None:
    linearization = _linearization()
    baseline = np.zeros((3, 3), dtype=np.float64)
    linearized_single = np.asarray(
        [[100.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.001, 0.0, 0.0]],
        dtype=np.float64,
    )
    nonlinear_single = linearized_single.copy()
    nonlinear_single[1, 1] = 0.002

    aggregate = evaluate_nonlinear_closure(
        linearization.artifact_id,
        baseline_query_m=baseline,
        linearized_query_m=linearized_single,
        nonlinear_query_m=nonlinear_single,
        absolute_tolerance_m=0.0,
        relative_tolerance=0.01,
    )
    certificate = _certificate(
        linearization=linearization,
        perturbation_indices=np.asarray([10], dtype=np.int64),
        baseline_query_m=baseline,
        linearized_query_m=linearized_single[None, :, :],
        nonlinear_query_m=nonlinear_single[None, :, :],
        absolute_tolerance_m=np.asarray([0.0001] * 3),
        relative_tolerance=np.asarray([0.1] * 3),
    )

    assert aggregate.candidate_valid
    assert not certificate.locally_closed
    assert certificate.worst_query_index == 200
    assert certificate.maximum_closure_ratio == pytest.approx(10.0)


def test_nonlinear_closure_v2_comparison_tolerance_is_explicit() -> None:
    baseline, linearized, nonlinear = _replays()
    allowed = 0.001 + 0.1 * 0.01
    nonlinear[0, 0, 1] = allowed * (1.0 + 5e-7)

    strict = _certificate(
        baseline_query_m=baseline,
        linearized_query_m=linearized,
        nonlinear_query_m=nonlinear,
        comparison_tolerance=0.0,
    )
    tolerant = _certificate(
        baseline_query_m=baseline,
        linearized_query_m=linearized,
        nonlinear_query_m=nonlinear,
        comparison_tolerance=1e-6,
    )

    assert not strict.locally_closed
    assert tolerant.locally_closed
    assert tolerant.admission_bound == pytest.approx(1.000001)


def test_nonlinear_closure_v2_binds_every_input_and_derived_array() -> None:
    source = _certificate()
    _, linearized, nonlinear = _replays()
    nonlinear[0, 0, 2] += 1e-9
    changed_replay = _certificate(
        linearized_query_m=linearized,
        nonlinear_query_m=nonlinear,
    )
    changed_horizon = _certificate(
        horizon_indices=np.asarray([1, 2, 2], dtype=np.int64)
    )

    assert changed_replay.artifact_id != source.artifact_id
    assert changed_horizon.artifact_id != source.artifact_id
    assert source.to_record()["artifact_id"] == source.artifact_id
    for name, array in source.arrays().items():
        record = source.descriptor()[name]
        assert record["shape"] == list(array.shape)
        assert record["dtype"] == array.dtype.str
        assert len(record["sha256"]) == 64


def test_nonlinear_closure_v2_rejects_tampered_supplied_id() -> None:
    source = _certificate()
    accepted = _certificate(artifact_id=source.artifact_id)
    assert accepted.artifact_id == source.artifact_id

    with pytest.raises(ValueError, match="artifact_id does not match content"):
        _certificate(artifact_id="0" * 64)


def test_nonlinear_closure_v2_arrays_and_metadata_are_immutable() -> None:
    baseline, linearized, nonlinear = _replays()
    metadata = {"source": {"groups": ["a", "b"]}}
    certificate = _certificate(
        baseline_query_m=baseline,
        linearized_query_m=linearized,
        nonlinear_query_m=nonlinear,
        metadata=metadata,
    )
    artifact_id = certificate.artifact_id

    baseline[0, 0] = 9.0
    linearized[0, 0, 0] = 9.0
    nonlinear[0, 0, 0] = 9.0
    metadata["source"]["groups"].append("mutated")

    assert certificate.baseline_query_m[0, 0] == 0.0
    assert certificate.linearized_query_m[0, 0, 0] == 0.01
    assert certificate.metadata["source"]["groups"] == ["a", "b"]
    assert certificate.artifact_id == artifact_id
    for name, array in certificate.arrays().items():
        assert not array.flags.writeable, name
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        certificate.metadata["source"]["groups"].append("forbidden")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("linearization", cast(Any, object()), "PhysicalLinearizationV1"),
        ("perturbation_set_id", "invalid", "lowercase hexadecimal"),
        ("query_set_id", cast(str, 7), "lowercase hexadecimal"),
    ),
)
def test_nonlinear_closure_v2_rejects_invalid_bound_identities(
    field: str,
    value: Any,
    message: str,
) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=message):
        _certificate(**kwargs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "perturbation_indices",
            np.asarray([10, 10], dtype=np.int64),
            "must be unique",
        ),
        (
            "query_indices",
            np.asarray([100, -1, 300], dtype=np.int64),
            "must be nonnegative",
        ),
        (
            "horizon_indices",
            np.asarray([1, -1, 2], dtype=np.int64),
            "must be nonnegative",
        ),
        (
            "query_indices",
            np.asarray([100, 200], dtype=np.int64),
            "identify every query row",
        ),
    ),
)
def test_nonlinear_closure_v2_rejects_invalid_index_contracts(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _certificate(**{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("baseline_query_m", np.zeros((3, 2)), r"shape \(Q, 3\)"),
        ("linearized_query_m", np.zeros((2, 3, 2)), r"shape \(P, Q, 3\)"),
        ("nonlinear_query_m", np.zeros((1, 3, 3)), r"shape \(P, Q, 3\)"),
        ("absolute_tolerance_m", np.ones(2), r"shape \(Q,\)"),
        ("relative_tolerance", np.ones(2), r"shape \(Q,\)"),
    ),
)
def test_nonlinear_closure_v2_rejects_shape_drift(
    field: str,
    value: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _certificate(**{field: value})


def test_nonlinear_closure_v2_rejects_nonfinite_or_nonnumeric_replays() -> None:
    baseline, _, _ = _replays()
    baseline[0, 0] = np.nan
    with pytest.raises(ValueError, match="baseline_query_m must be finite"):
        _certificate(baseline_query_m=baseline)

    with pytest.raises(ValueError, match="must contain real numeric values"):
        _certificate(baseline_query_m=cast(Any, [["bad"] * 3] * 3))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("absolute_tolerance_m", np.asarray([0.0, 0.0, 0.0]), "positive"),
        ("relative_tolerance", np.asarray([-0.1, 0.1, 0.1]), "nonnegative"),
        ("prediction_floor_m", 0.0, "must be positive"),
        ("prediction_floor_m", cast(float, True), "nonnegative real number"),
        ("closure_ratio_limit", -1.0, "nonnegative real number"),
        ("comparison_tolerance", np.inf, "nonnegative real number"),
    ),
)
def test_nonlinear_closure_v2_rejects_invalid_policy_values(
    field: str,
    value: Any,
    message: str,
) -> None:
    kwargs: dict[str, Any] = {field: value}
    if field == "absolute_tolerance_m":
        kwargs["relative_tolerance"] = np.zeros(3)
    with pytest.raises(ValueError, match=message):
        _certificate(**kwargs)
