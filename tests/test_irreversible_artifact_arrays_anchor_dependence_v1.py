from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

import bayesian_phystwin.inference._guarded as guarded_module
from bayesian_phystwin.inference.v1 import (
    AnchorDependenceV1,
    ClaimBearingProb4DCandidateV1,
    ObservationBeliefV1,
    PhysicalLinearizationV1,
    infer_prob4d_candidate,
)


class _ArrayFailure:
    def __array__(self) -> np.ndarray:
        raise TypeError("not an array")


def _base_kwargs(*, count: int = 1) -> dict[str, Any]:
    return {
        "correlation_group_ids": tuple("contact:shared" for _ in range(count)),
        "prior_reliability": np.full(count, 0.9),
        "prior_nominal_probability": np.full(count, 0.95),
        "composite_weight": np.full(count, 1.0 / max(count, 1)),
        "bias_jacobian": np.ones((count, 3, 1), dtype=np.float64),
        "bias_prior_covariance": np.array([[1e-6]], dtype=np.float64),
        "metadata": {"source": "unit-test"},
    }


def _dependence(*, count: int = 2) -> AnchorDependenceV1:
    return AnchorDependenceV1(**_base_kwargs(count=count))


def test_anchor_dependence_is_content_addressed_and_immutable() -> None:
    reliability = np.array([0.9, 0.8], dtype=np.float64)
    kwargs = _base_kwargs(count=2)
    kwargs["prior_reliability"] = reliability
    dependence = AnchorDependenceV1(**kwargs)
    duplicate = AnchorDependenceV1(
        **kwargs,
        artifact_id=dependence.artifact_id,
    )

    reliability[0] = 0.0
    assert dependence.prior_reliability[0] == 0.9
    assert dependence.prior_reliability.flags.writeable is False
    assert dependence.bias_jacobian is not None
    assert dependence.bias_jacobian.flags.writeable is False
    assert duplicate.artifact_id == dependence.artifact_id
    assert dependence.to_record()["artifact_id"] == dependence.artifact_id
    assert dependence.anchor_count == 2
    assert dependence.bias_dimension == 1
    with pytest.raises(TypeError, match="immutable"):
        dependence.metadata["source"] = "tampered"  # type: ignore[index]


def test_anchor_dependence_without_bias_has_explicit_zero_dimension() -> None:
    dependence = AnchorDependenceV1(
        correlation_group_ids=("contact:single",),
        prior_reliability=np.ones(1),
        prior_nominal_probability=np.ones(1),
        composite_weight=np.ones(1),
    )

    assert dependence.bias_dimension == 0
    assert set(dependence.arrays()) == {
        "prior_reliability",
        "prior_nominal_probability",
        "composite_weight",
    }
    dependence.require_anchor_count(1)
    with pytest.raises(TypeError, match="must be an integer"):
        dependence.require_anchor_count(cast(Any, True))
    with pytest.raises(ValueError, match="row count"):
        dependence.require_anchor_count(2)


def test_anchor_dependence_identity_covers_groups_arrays_and_metadata() -> None:
    baseline = _dependence()
    changes = [
        {"correlation_group_ids": ("contact:a", "contact:b")},
        {"composite_weight": np.array([0.4, 0.5])},
        {"metadata": {"source": "other"}},
    ]

    for change in changes:
        kwargs = _base_kwargs(count=2)
        kwargs.update(change)
        assert AnchorDependenceV1(**kwargs).artifact_id != baseline.artifact_id


@pytest.mark.parametrize(
    ("overrides", "exception", "match"),
    [
        (
            {
                "correlation_group_ids": (),
                "prior_reliability": np.zeros(0),
                "prior_nominal_probability": np.zeros(0),
                "composite_weight": np.zeros(0),
                "bias_jacobian": None,
                "bias_prior_covariance": None,
            },
            ValueError,
            "must not be empty",
        ),
        ({"correlation_group_ids": ["anchor"]}, TypeError, "tuple of exact"),
        ({"correlation_group_ids": ("anchor", 1)}, TypeError, "tuple of exact"),
        ({"correlation_group_ids": (" anchor",)}, ValueError, "whitespace"),
        ({"prior_reliability": _ArrayFailure()}, ValueError, "real numeric"),
        ({"prior_reliability": ["bad"]}, ValueError, "real numeric"),
        ({"prior_reliability": np.array([np.nan])}, ValueError, "finite"),
        ({"prior_reliability": np.ones(2)}, ValueError, "shape"),
        ({"prior_reliability": np.array([1.1])}, ValueError, "reliability"),
        (
            {"prior_nominal_probability": np.array([-0.1])},
            ValueError,
            "nominal_probability",
        ),
        ({"composite_weight": np.zeros(1)}, ValueError, "composite_weight"),
        ({"bias_jacobian": None}, ValueError, "supplied together"),
        ({"bias_prior_covariance": None}, ValueError, "supplied together"),
        (
            {"bias_jacobian": np.ones((2, 3, 1))},
            ValueError,
            "bias_jacobian must have shape",
        ),
        (
            {
                "bias_jacobian": np.ones((1, 3, 0)),
                "bias_prior_covariance": np.zeros((0, 0)),
            },
            ValueError,
            "at least one bias mode",
        ),
        ({"bias_prior_covariance": np.ones(1)}, ValueError, "2 dimensions"),
        ({"bias_prior_covariance": np.eye(2)}, ValueError, r"shape \(B, B\)"),
        (
            {
                "bias_jacobian": np.ones((1, 3, 2)),
                "bias_prior_covariance": np.array([[1.0, 1.0], [0.0, 1.0]]),
            },
            ValueError,
            "symmetric",
        ),
        (
            {"bias_jacobian": np.array([[[np.nan], [0.0], [0.0]]])},
            ValueError,
            "finite",
        ),
        (
            {"bias_prior_covariance": np.array([[-1.0]])},
            ValueError,
            "positive semidefinite",
        ),
        ({"metadata": 1}, ValueError, "metadata"),
        ({"artifact_id": "bad"}, ValueError, "lowercase SHA-256"),
        ({"artifact_id": "0" * 64}, ValueError, "does not match content"),
    ],
)
def test_anchor_dependence_rejects_ambiguous_or_invalid_inputs(
    overrides: dict[str, Any],
    exception: type[Exception],
    match: str,
) -> None:
    kwargs = _base_kwargs()
    kwargs.update(overrides)
    with pytest.raises(exception, match=match):
        AnchorDependenceV1(**kwargs)


def test_anchor_dependence_maps_to_exact_frozen_solver_keywords() -> None:
    dependence = _dependence()
    keywords = dependence.inference_kwargs()

    assert set(keywords) == {
        "anchor_correlation_group_ids",
        "anchor_prior_reliability",
        "anchor_prior_nominal_probability",
        "anchor_composite_weight",
        "anchor_bias_jacobian",
        "anchor_bias_prior_covariance",
    }
    assert keywords["anchor_correlation_group_ids"] == (
        "contact:shared",
        "contact:shared",
    )
    assert keywords["anchor_prior_reliability"] is dependence.prior_reliability
    assert keywords["anchor_bias_jacobian"] is dependence.bias_jacobian


@dataclass(frozen=True)
class _DummyResult:
    input_lineage: dict[str, object]


@dataclass(frozen=True)
class _DummyUpdate:
    result: _DummyResult


@dataclass(frozen=True)
class _DummyCandidate:
    update_v1: _DummyUpdate

    @property
    def result(self) -> _DummyResult:
        return self.update_v1.result


def test_typed_anchor_identity_is_bound_without_mutating_candidate() -> None:
    dependence = _dependence()
    candidate = _DummyCandidate(_DummyUpdate(_DummyResult({"existing": True})))

    bound = guarded_module._bind_anchor_dependence_identity(  # noqa: SLF001
        cast(ClaimBearingProb4DCandidateV1, candidate),
        dependence,
    )

    assert candidate.result.input_lineage == {"existing": True}
    assert bound.result.input_lineage == {
        "existing": True,
        "anchor_dependence_artifact_id": dependence.artifact_id,
    }


def test_typed_anchor_identity_rejects_conflicting_lineage() -> None:
    dependence = _dependence()
    candidate = _DummyCandidate(
        _DummyUpdate(_DummyResult({"anchor_dependence_artifact_id": "0" * 64}))
    )

    with pytest.raises(ValueError, match="contradicts"):
        guarded_module._bind_anchor_dependence_identity(  # noqa: SLF001
            cast(ClaimBearingProb4DCandidateV1, candidate),
            dependence,
        )


def test_guarded_api_delegates_typed_dependence_and_binds_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = object.__new__(ObservationBeliefV1)
    linearization = object.__new__(PhysicalLinearizationV1)
    dependence = _dependence()
    prediction = np.zeros((1, 3), dtype=np.float64)
    candidate = object.__new__(ClaimBearingProb4DCandidateV1)
    bound_candidate = object.__new__(ClaimBearingProb4DCandidateV1)
    captured: dict[str, Any] = {}

    def fake_infer(*args: object, **kwargs: object) -> ClaimBearingProb4DCandidateV1:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return candidate

    def fake_bind(
        value: ClaimBearingProb4DCandidateV1,
        value_dependence: AnchorDependenceV1,
    ) -> ClaimBearingProb4DCandidateV1:
        captured["candidate"] = value
        captured["dependence"] = value_dependence
        return bound_candidate

    monkeypatch.setattr(
        guarded_module,
        "infer_claim_bearing_prob4d_candidate_from_artifacts",
        fake_infer,
    )
    monkeypatch.setattr(
        guarded_module,
        "_bind_anchor_dependence_identity",
        fake_bind,
    )

    result = infer_prob4d_candidate(
        observation,
        linearization,
        physical_prediction_xyz_m=prediction,
        anchor_innovation_m=np.zeros((2, 3)),
        anchor_covariance_m2=np.repeat(np.eye(3)[None], 2, axis=0),
        anchor_state_jacobian=np.zeros((2, 3, 1)),
        anchor_dependence=dependence,
    )

    assert result is bound_candidate
    assert captured["args"] == (observation, linearization)
    assert captured["candidate"] is candidate
    assert captured["dependence"] is dependence
    kwargs = cast(dict[str, object], captured["kwargs"])
    for name, value in dependence.inference_kwargs().items():
        assert kwargs[name] is value


def test_guarded_api_rejects_mixed_or_misaligned_typed_dependence() -> None:
    observation = object.__new__(ObservationBeliefV1)
    linearization = object.__new__(PhysicalLinearizationV1)
    prediction = np.zeros((1, 3), dtype=np.float64)
    dependence = _dependence()

    cases: list[tuple[type[Exception], str, dict[str, Any]]] = [
        (
            ValueError,
            "cannot be combined",
            {
                "anchor_innovation_m": np.zeros((2, 3)),
                "anchor_dependence": dependence,
                "anchor_prior_reliability": np.ones(2),
            },
        ),
        (
            ValueError,
            "row count",
            {
                "anchor_innovation_m": np.zeros((1, 3)),
                "anchor_dependence": dependence,
            },
        ),
        (ValueError, "requires anchor_innovation_m", {"anchor_dependence": dependence}),
        (
            ValueError,
            r"shape \(A, 3\)",
            {
                "anchor_innovation_m": np.zeros(2),
                "anchor_dependence": dependence,
            },
        ),
        (
            TypeError,
            "AnchorDependenceV1 or None",
            {
                "anchor_innovation_m": np.zeros((2, 3)),
                "anchor_dependence": object(),
            },
        ),
        (
            TypeError,
            "unknown legacy",
            {
                "anchor_innovation_m": np.zeros((2, 3)),
                "anchor_reliabilty": np.ones(2),
            },
        ),
    ]

    for exception, match, extra in cases:
        with pytest.raises(exception, match=match):
            infer_prob4d_candidate(
                observation,
                linearization,
                physical_prediction_xyz_m=prediction,
                **extra,
            )
