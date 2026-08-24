from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.query_identifying_action_design_v1 import (
    QueryIdentifyingActionCandidateV1,
    QueryIdentifyingActionDesignV1,
    QueryIdentifyingActionStatus,
    QueryIdentifyingDesignStatus,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _candidate(
    action_id: str,
    state_jacobian: np.ndarray,
    *,
    nuisance_jacobian: np.ndarray | None = None,
    observation_variance: float = 1.0,
    reliability: float | np.ndarray = 1.0,
    cost: float = 0.0,
    risk: float = 0.0,
    safe: bool = True,
) -> QueryIdentifyingActionCandidateV1:
    state_jacobian = np.asarray(state_jacobian, dtype=np.float64)
    if state_jacobian.ndim == 1:
        state_jacobian = state_jacobian[None, :]
    if nuisance_jacobian is None:
        nuisance_jacobian = np.zeros((state_jacobian.shape[0], 0))
    return QueryIdentifyingActionCandidateV1(
        action_id=action_id,
        state_jacobian=state_jacobian,
        nuisance_jacobian=nuisance_jacobian,
        observation_covariance=(
            np.eye(state_jacobian.shape[0]) * observation_variance
        ),
        reliability=reliability,
        dimensionless_cost=cost,
        dimensionless_risk=risk,
        safety_admissible=safe,
    )


def _decision(
    candidates: list[QueryIdentifyingActionCandidateV1],
    *,
    state_precision: np.ndarray | None = None,
    nuisance_precision: np.ndarray | None = None,
    cross_precision: np.ndarray | None = None,
    query: np.ndarray | None = None,
    query_scale: np.ndarray | None = None,
    cost_weight: float = 0.0,
    risk_weight: float = 0.0,
    maximum_risk: float = 1.0,
    minimum_improvement: float = 0.0,
) -> QueryIdentifyingActionDesignV1:
    state_precision = (
        np.eye(2)
        if state_precision is None
        else np.asarray(state_precision, dtype=np.float64)
    )
    nuisance_precision = (
        np.zeros((0, 0))
        if nuisance_precision is None
        else np.asarray(nuisance_precision, dtype=np.float64)
    )
    cross_precision = (
        np.zeros((state_precision.shape[0], nuisance_precision.shape[0]))
        if cross_precision is None
        else np.asarray(cross_precision, dtype=np.float64)
    )
    query = np.eye(2) if query is None else np.asarray(query, dtype=np.float64)
    query_scale = (
        np.eye(query.shape[0])
        if query_scale is None
        else np.asarray(query_scale, dtype=np.float64)
    )
    return QueryIdentifyingActionDesignV1(
        prior_belief_id=_digest("prior"),
        query_id=_digest("query"),
        query_scale_id=_digest("query-scale"),
        protocol_id=_digest("protocol"),
        prior_state_precision=state_precision,
        prior_nuisance_precision=nuisance_precision,
        prior_state_nuisance_precision=cross_precision,
        query_jacobian=query,
        query_scale=query_scale,
        candidates=candidates,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
        maximum_risk=maximum_risk,
        minimum_objective_improvement=minimum_improvement,
    )


def test_selects_action_with_largest_query_contraction() -> None:
    weak = _candidate("weak", np.array([0.2, 0.0]))
    strong = _candidate("strong", np.array([2.0, 0.0]))
    decision = _decision([weak, strong], query=np.array([[1.0, 0.0]]))

    assert decision.status is QueryIdentifyingDesignStatus.ACTION_SELECTED
    assert decision.selected_action_id == "strong"
    assert decision.selected_evaluation is not None
    assert decision.selected_evaluation.query_trace_reduction > 0.0
    assert decision.selected_evaluation.marginal_state_information_gain_nats > 0.0
    assert decision.action_selected is True
    assert decision.no_action_recommended is False


def test_nuisance_confounding_reduces_action_value() -> None:
    clean = _candidate(
        "clean",
        np.array([1.0, 0.0]),
        nuisance_jacobian=np.zeros((1, 1)),
    )
    confounded = _candidate(
        "confounded",
        np.array([1.0, 0.0]),
        nuisance_jacobian=np.array([[1.0]]),
    )
    decision = _decision(
        [clean, confounded],
        nuisance_precision=np.array([[0.01]]),
        query=np.array([[1.0, 0.0]]),
    )
    evaluations = {item.candidate.action_id: item for item in decision.evaluations}

    assert decision.selected_action_id == "clean"
    assert evaluations["clean"].nuisance_trace_effect == pytest.approx(0.0)
    assert evaluations["confounded"].nuisance_trace_effect > 0.0
    assert (
        evaluations["confounded"].query_trace_reduction
        < evaluations["clean"].query_trace_reduction
    )


def test_safety_and_risk_gates_fail_closed() -> None:
    unsafe = _candidate("unsafe", np.array([10.0, 0.0]), safe=False)
    risky = _candidate("risky", np.array([10.0, 0.0]), risk=0.8)
    decision = _decision(
        [unsafe, risky],
        query=np.array([[1.0, 0.0]]),
        maximum_risk=0.5,
    )
    statuses = {item.candidate.action_id: item.status for item in decision.evaluations}

    assert decision.status is QueryIdentifyingDesignStatus.NO_ELIGIBLE_ACTION
    assert decision.selected_action_id is None
    assert decision.no_action_recommended is True
    assert statuses == {
        "risky": QueryIdentifyingActionStatus.RISK_REJECTED,
        "unsafe": QueryIdentifyingActionStatus.SAFETY_REJECTED,
    }


def test_cost_can_make_information_gain_insufficient() -> None:
    action = _candidate("costly", np.array([2.0, 0.0]), cost=10.0)
    decision = _decision(
        [action],
        query=np.array([[1.0, 0.0]]),
        cost_weight=1.0,
        minimum_improvement=0.01,
    )

    assert decision.status is QueryIdentifyingDesignStatus.INSUFFICIENT_GAIN
    assert decision.evaluations[0].status is (
        QueryIdentifyingActionStatus.INSUFFICIENT_GAIN
    )
    assert decision.evaluations[0].marginal_state_information_gain_nats > 0.0
    assert decision.evaluations[0].objective_improvement < 0.0


def test_order_invariance_and_lexicographic_tie_break() -> None:
    beta = _candidate("beta", np.array([1.0, 0.0]))
    alpha = _candidate("alpha", np.array([1.0, 0.0]))
    first = _decision([beta, alpha], query=np.array([[1.0, 0.0]]))
    second = _decision([alpha, beta], query=np.array([[1.0, 0.0]]))

    assert first.artifact_id == second.artifact_id
    assert first.selected_action_id == second.selected_action_id == "alpha"
    assert [item.candidate.action_id for item in first.evaluations] == [
        "alpha",
        "beta",
    ]


def test_reliability_scales_expected_action_value() -> None:
    low = _candidate("low", np.array([1.0, 0.0]), reliability=0.1)
    high = _candidate("high", np.array([1.0, 0.0]), reliability=0.9)
    decision = _decision([low, high], query=np.array([[1.0, 0.0]]))
    evaluations = {item.candidate.action_id: item for item in decision.evaluations}

    assert decision.selected_action_id == "high"
    assert evaluations["high"].query_trace_reduction > (
        evaluations["low"].query_trace_reduction
    )


def test_physical_reparameterization_preserves_scores() -> None:
    covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    precision = np.linalg.solve(covariance, np.eye(2))
    query = np.array([[1.0, -0.5]])
    jacobian = np.array([[1.2, 0.7], [-0.3, 0.9]])
    original = _decision(
        [_candidate("probe", jacobian)],
        state_precision=precision,
        query=query,
    )

    transform = np.array([[2.0, 0.4], [0.0, 0.5]])
    inverse = np.linalg.solve(transform, np.eye(2))
    transformed_covariance = transform @ covariance @ transform.T
    transformed_precision = np.linalg.solve(
        transformed_covariance,
        np.eye(2),
    )
    transformed = _decision(
        [_candidate("probe", jacobian @ inverse)],
        state_precision=transformed_precision,
        query=query @ inverse,
    )

    left = original.evaluations[0]
    right = transformed.evaluations[0]
    assert left.posterior_normalized_query_trace == pytest.approx(
        right.posterior_normalized_query_trace,
        rel=1e-11,
        abs=1e-12,
    )
    assert left.query_trace_reduction == pytest.approx(
        right.query_trace_reduction,
        rel=1e-11,
        abs=1e-12,
    )
    assert left.marginal_state_information_gain_nats == pytest.approx(
        right.marginal_state_information_gain_nats,
        rel=1e-11,
        abs=1e-12,
    )


def test_trivial_query_recommends_no_action() -> None:
    decision = _decision(
        [_candidate("probe", np.array([1.0, 0.0]))],
        query=np.zeros((1, 2)),
    )

    assert decision.status is QueryIdentifyingDesignStatus.TRIVIAL_QUERY
    assert decision.evaluations[0].status is (
        QueryIdentifyingActionStatus.TRIVIAL_QUERY
    )
    assert decision.selected_action_id is None
    assert decision.baseline_normalized_query_trace == 0.0


def test_contracts_are_immutable_and_content_addressed() -> None:
    candidate = _candidate("probe", np.array([1.0, 0.0]))
    decision = _decision([candidate], query=np.array([[1.0, 0.0]]))

    assert len(candidate.artifact_id or "") == 64
    assert len(decision.artifact_id or "") == 64
    with pytest.raises(ValueError, match="read-only"):
        candidate.state_jacobian[0, 0] = 0.0
    with pytest.raises(TypeError):
        candidate.metadata["changed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="artifact_id"):
        replace(candidate, artifact_id=_digest("wrong"))
    with pytest.raises(ValueError, match="artifact_id"):
        replace(decision, artifact_id=_digest("wrong"))


def test_validation_rejects_dimension_reliability_and_covariance_errors() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        _candidate(
            "bad",
            np.array([1.0, 0.0]),
            observation_variance=0.0,
        )
    with pytest.raises(ValueError, match="reliability"):
        _candidate("bad-r", np.array([1.0, 0.0]), reliability=1.1)
    with pytest.raises(ValueError, match="unique"):
        _decision(
            [
                _candidate("same", np.array([1.0, 0.0])),
                _candidate("same", np.array([0.0, 1.0])),
            ]
        )
    with pytest.raises(ValueError, match="nuisance_precision"):
        _decision(
            [
                _candidate(
                    "bad-n",
                    np.array([1.0, 0.0]),
                    nuisance_jacobian=np.array([[1.0]]),
                )
            ],
            nuisance_precision=np.array([[-1.0]]),
        )


def test_candidate_validation_rejects_malformed_scalars_and_arrays() -> None:
    base = {
        "action_id": "probe",
        "state_jacobian": np.array([[1.0, 0.0]]),
        "nuisance_jacobian": np.zeros((1, 0)),
        "observation_covariance": np.eye(1),
    }
    for action_id in ("", " probe", 1):
        with pytest.raises(ValueError, match="action_id"):
            QueryIdentifyingActionCandidateV1(**{**base, "action_id": action_id})
    for cost in (True, -1.0, np.inf):
        with pytest.raises(ValueError, match="dimensionless_cost"):
            QueryIdentifyingActionCandidateV1(
                **base,
                dimensionless_cost=cost,  # type: ignore[arg-type]
            )
    with pytest.raises(ValueError, match="real numeric"):
        QueryIdentifyingActionCandidateV1(
            **{**base, "state_jacobian": np.array([["x"]])}
        )
    with pytest.raises(ValueError, match="must be a matrix"):
        QueryIdentifyingActionCandidateV1(
            **{**base, "state_jacobian": np.array([1.0, 0.0])}
        )
    with pytest.raises(ValueError, match="must be finite"):
        QueryIdentifyingActionCandidateV1(
            **{**base, "state_jacobian": np.array([[np.nan, 0.0]])}
        )
    with pytest.raises(ValueError, match="must be square"):
        QueryIdentifyingActionCandidateV1(
            **{**base, "observation_covariance": np.ones((1, 2))}
        )
    with pytest.raises(ValueError, match="exactly symmetric"):
        QueryIdentifyingActionCandidateV1(
            **{
                **base,
                "state_jacobian": np.eye(2),
                "nuisance_jacobian": np.zeros((2, 0)),
                "observation_covariance": np.array([[1.0, 0.2], [0.0, 1.0]]),
            }
        )
    with pytest.raises(ValueError, match="must be nonempty"):
        QueryIdentifyingActionCandidateV1(
            **{
                **base,
                "state_jacobian": np.empty((0, 2)),
                "nuisance_jacobian": np.empty((0, 0)),
                "observation_covariance": np.empty((0, 0)),
            }
        )


def test_candidate_validation_rejects_shape_and_reliability_drift() -> None:
    with pytest.raises(ValueError, match="state_jacobian must be nonempty"):
        _candidate("empty-state", np.empty((1, 0)))
    with pytest.raises(ValueError, match="same observation rows"):
        QueryIdentifyingActionCandidateV1(
            action_id="bad-nuisance-rows",
            state_jacobian=np.ones((2, 1)),
            nuisance_jacobian=np.zeros((1, 0)),
            observation_covariance=np.eye(2),
        )
    with pytest.raises(ValueError, match="one row and column"):
        QueryIdentifyingActionCandidateV1(
            action_id="bad-covariance-shape",
            state_jacobian=np.ones((2, 1)),
            nuisance_jacobian=np.zeros((2, 0)),
            observation_covariance=np.eye(1),
        )
    with pytest.raises(ValueError, match="real numeric"):
        _candidate("bad-reliability-type", np.ones((2, 1)), reliability="bad")
    with pytest.raises(ValueError, match="one value per observation row"):
        _candidate(
            "bad-reliability-shape",
            np.ones((2, 1)),
            reliability=np.ones(3),
        )


def test_candidate_records_and_dimension_properties_are_consistent() -> None:
    candidate = _candidate(
        "block",
        np.eye(2),
        nuisance_jacobian=np.ones((2, 1)),
        reliability=np.array([0.2, 0.8]),
    )
    record = candidate.to_record()

    assert candidate.observation_dimension == 2
    assert candidate.state_dimension == 2
    assert candidate.nuisance_dimension == 1
    assert candidate.reliability_vector.tolist() == [0.2, 0.8]
    assert record["artifact_id"] == candidate.artifact_id
    assert record["action_id"] == "block"


def test_evaluation_validation_and_records_fail_closed() -> None:
    decision = _decision(
        [_candidate("probe", np.array([1.0, 0.0]))],
        query=np.array([[1.0, 0.0]]),
    )
    evaluation = decision.evaluations[0]
    assert evaluation.summary()["artifact_id"] == evaluation.artifact_id
    assert evaluation.to_record()["artifact_id"] == evaluation.artifact_id

    with pytest.raises(TypeError, match="candidate"):
        replace(evaluation, candidate=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="status"):
        replace(evaluation, status="eligible")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite nonnegative"):
        replace(evaluation, query_trace_reduction=-1.0)
    with pytest.raises(ValueError, match="finite real"):
        replace(evaluation, objective_improvement=np.inf)
    with pytest.raises(ValueError, match="artifact_id"):
        replace(evaluation, artifact_id=_digest("wrong-evaluation"))


def test_design_validation_rejects_invalid_roster_query_and_thresholds() -> None:
    candidate = _candidate("probe", np.array([1.0, 0.0]))
    kwargs = {
        "prior_belief_id": _digest("prior"),
        "query_id": _digest("query"),
        "query_scale_id": _digest("scale"),
        "protocol_id": _digest("protocol"),
        "prior_state_precision": np.eye(2),
        "prior_nuisance_precision": np.zeros((0, 0)),
        "prior_state_nuisance_precision": np.zeros((2, 0)),
        "query_jacobian": np.array([[1.0, 0.0]]),
        "query_scale": np.eye(1),
        "candidates": [candidate],
    }
    with pytest.raises(ValueError, match="numerical_tolerance"):
        QueryIdentifyingActionDesignV1(**kwargs, numerical_tolerance=0.0)
    with pytest.raises(ValueError, match="selection_tolerance"):
        QueryIdentifyingActionDesignV1(**kwargs, selection_tolerance=True)
    with pytest.raises(ValueError, match="at least one query row"):
        QueryIdentifyingActionDesignV1(
            **{
                **kwargs,
                "query_jacobian": np.empty((0, 2)),
                "query_scale": np.empty((0, 0)),
            }
        )
    with pytest.raises(ValueError, match="one row and column"):
        QueryIdentifyingActionDesignV1(**{**kwargs, "query_scale": np.eye(2)})
    with pytest.raises(ValueError, match="at least one action"):
        QueryIdentifyingActionDesignV1(**{**kwargs, "candidates": []})
    with pytest.raises(TypeError, match="every candidate"):
        QueryIdentifyingActionDesignV1(**{**kwargs, "candidates": [object()]})
    with pytest.raises(ValueError, match="state_jacobian dimension"):
        QueryIdentifyingActionDesignV1(
            **{**kwargs, "candidates": [_candidate("wrong-state", np.ones(3))]}
        )
    with pytest.raises(ValueError, match="nuisance_jacobian dimension"):
        QueryIdentifyingActionDesignV1(
            **{
                **kwargs,
                "candidates": [
                    _candidate(
                        "wrong-nuisance",
                        np.array([1.0, 0.0]),
                        nuisance_jacobian=np.ones((1, 1)),
                    )
                ],
            }
        )


def test_design_summary_records_and_no_selection_accessor() -> None:
    decision = _decision(
        [_candidate("unsafe", np.array([1.0, 0.0]), safe=False)],
        query=np.array([[1.0, 0.0]]),
    )
    summary = decision.summary()
    record = decision.to_record()

    assert decision.selected_evaluation is None
    assert summary["artifact_id"] == decision.artifact_id
    assert summary["selected_action_id"] is None
    assert record["artifact_id"] == decision.artifact_id
    assert record["claim_boundary"]


def test_signed_nuisance_effect_preserves_prior_state_nuisance_correlation() -> None:
    nuisance_only_probe = _candidate(
        "nuisance-probe",
        np.array([0.0]),
        nuisance_jacobian=np.array([[1.0]]),
    )
    decision = _decision(
        [nuisance_only_probe],
        state_precision=np.array([[1.0]]),
        nuisance_precision=np.array([[1.0]]),
        cross_precision=np.array([[0.5]]),
        query=np.array([[1.0]]),
    )
    evaluation = decision.evaluations[0]

    assert evaluation.query_trace_reduction > 0.0
    assert evaluation.ideal_query_trace_reduction == pytest.approx(0.0)
    assert evaluation.nuisance_trace_effect < 0.0
