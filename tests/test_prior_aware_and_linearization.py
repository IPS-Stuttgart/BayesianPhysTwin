import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin._gauge_aware_contracts import GaugeAwareObservationBatch
from bayesian_phystwin.complete_belief_selection import (
    CompleteBeliefGuardDecisionV1,
    CompleteBeliefSelectionV1,
    select_complete_belief,
)
from bayesian_phystwin.physical_linearization import (
    NonlinearClosureV1,
    PhysicalLinearizationV1,
    evaluate_nonlinear_closure,
    load_physical_linearization,
    save_physical_linearization,
    validate_observation_linearization_alignment,
)
from bayesian_phystwin.prior_aware_gauge_belief import (
    PriorAwareGaugeConfigV1,
    update_prior_aware_gauge_belief,
)
from bayesian_phystwin.propagated_state_belief import (
    PropagatedStateBeliefConfig,
    infer_propagated_state_belief,
)

A = "a" * 64
B = "b" * 64
C = "c" * 64


def _empty(count: int) -> np.ndarray:
    return np.zeros((count, 3, 0), dtype=float)


def _confounded_batch(
    gauge_variance: float, *, outlier: bool = False
) -> GaugeAwareObservationBatch:
    count = 12
    state = np.zeros((count, 3, 1))
    state[:, 0, 0] = 1.0
    innovation = np.zeros((count, 3))
    innovation[:, 0] = 0.01
    if outlier:
        innovation[-4:, 0] = 0.25
    groups = tuple("g0" if index < 8 else "g1" for index in range(count))
    return GaugeAwareObservationBatch(
        innovation_m=innovation,
        observation_covariance_m2=np.tile(np.eye(3) * 1e-6, (count, 1, 1)),
        state_jacobian=state,
        gauge_jacobian=state.copy(),
        shared_bias_jacobian=_empty(count),
        view_bias_jacobian=_empty(count),
        query_state_jacobian=state.copy(),
        gauge_prior_covariance=np.asarray([[gauge_variance]]),
        correlation_group_ids=groups,
        prior_reliability=np.ones(count),
        prior_nominal_probability=np.asarray([0.95] * 8 + [0.80] * 4),
        composite_weight=np.ones(count),
        physical_response_scale_m=0.05,
        state_prior_covariance_m2=np.asarray([[0.01]]),
        metadata={"observation_artifact_id": C},
    )


def test_tight_gauge_prior_approaches_known_gauge_solution() -> None:
    result = update_prior_aware_gauge_belief(
        _confounded_batch(1e-10),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.01,
        ),
    )
    assert result.inference_admissible
    assert result.state_coefficients[0] == pytest.approx(0.01, abs=6e-4)
    assert result.diagnostics["identifiability_mode"] == "prior-aware-schur-v1"


def test_diffuse_gauge_prior_falls_back_or_suppresses_state() -> None:
    result = update_prior_aware_gauge_belief(
        _confounded_batch(1e6),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.10,
        ),
    )
    assert not result.inference_admissible or abs(result.state_coefficients[0]) < 0.005


def test_group_mixture_downweights_corrupted_group() -> None:
    result = update_prior_aware_gauge_belief(
        _confounded_batch(1e-10, outlier=True),
        config=PriorAwareGaugeConfigV1(
            effective_samples_per_correlation_group=12,
            minimum_identifiable_fraction=0.01,
        ),
    )
    posterior = result.diagnostics["observation_group_posterior_nominal_probability"]
    assert posterior[1] < posterior[0]
    assert result.diagnostics["prior_nominal_probability_used_inside_mixture"]


@dataclass
class Observation:
    artifact_id: str
    frame_ids: np.ndarray
    entity_ids: np.ndarray
    view_indices: np.ndarray
    window_indices: np.ndarray


def _linearization() -> PhysicalLinearizationV1:
    state = np.zeros((3, 3, 1))
    state[:, 0, 0] = [-1.0, 0.0, 1.0]
    return PhysicalLinearizationV1(
        observation_artifact_id=A,
        baseline_belief_id=B,
        action_prefix_id=C,
        simulator_revision="sim-1",
        frame_ids=np.asarray([1, 1, 2]),
        entity_ids=np.asarray([0, 1, 0]),
        view_indices=np.asarray([0, 0, 0]),
        window_indices=np.asarray([0, 0, 1]),
        state_jacobian=state,
        query_state_jacobian=state.copy(),
        physical_response_m=np.asarray([[0.01, 0.0, 0.0]] * 3),
    )


def _write_linearization_archive(
    path: Path,
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    payload: dict[str, Any] = {
        "descriptor_json": np.asarray(
            json.dumps(
                dict(descriptor),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    }
    payload.update(arrays)
    np.savez_compressed(path, **payload)


def test_linearization_rejects_row_permutation() -> None:
    linearization = _linearization()
    permutation = np.asarray([1, 0, 2])
    observation = Observation(
        artifact_id=A,
        frame_ids=linearization.frame_ids[permutation],
        entity_ids=linearization.entity_ids[permutation],
        view_indices=linearization.view_indices[permutation],
        window_indices=linearization.window_indices[permutation],
    )
    with pytest.raises(ValueError, match="differ"):
        validate_observation_linearization_alignment(observation, linearization)


def test_linearization_rejects_artifact_mismatch() -> None:
    linearization = _linearization()
    observation = Observation(
        artifact_id="0" * 64,
        frame_ids=linearization.frame_ids,
        entity_ids=linearization.entity_ids,
        view_indices=linearization.view_indices,
        window_indices=linearization.window_indices,
    )

    with pytest.raises(ValueError, match="does not identify"):
        validate_observation_linearization_alignment(observation, linearization)


def test_response_scale_is_bound_to_linearization() -> None:
    assert _linearization().physical_response_scale_m == pytest.approx(0.01)


@pytest.mark.parametrize(
    "field",
    ("frame_ids", "entity_ids", "view_indices", "window_indices"),
)
def test_linearization_rejects_lossy_integer_identity_coercion(field: str) -> None:
    source = _linearization()
    values = np.asarray(getattr(source, field), dtype=np.float64)

    with pytest.raises(ValueError, match=f"{field} must contain integers"):
        PhysicalLinearizationV1(
            **{
                **source.__dict__,
                field: values,
            }
        )


def test_linearization_rejects_nonvector_frame_ids() -> None:
    source = _linearization()

    with pytest.raises(ValueError, match="frame_ids must have shape"):
        PhysicalLinearizationV1(
            **{
                **source.__dict__,
                "frame_ids": np.asarray([[1, 1, 2]], dtype=np.int64),
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("observation_artifact_id", "invalid", "lowercase SHA-256"),
        ("baseline_belief_id", cast(str, 1), "lowercase SHA-256"),
        ("simulator_revision", "", "simulator_revision must be nonempty"),
        ("simulator_revision", cast(str, 1), "simulator_revision must be nonempty"),
    ),
)
def test_linearization_rejects_invalid_descriptor_fields(
    field: str,
    value: str,
    message: str,
) -> None:
    source = _linearization()

    with pytest.raises(ValueError, match=message):
        PhysicalLinearizationV1(
            **{
                **source.__dict__,
                field: value,
            }
        )


def test_linearization_metadata_is_deeply_immutable_and_id_stable() -> None:
    metadata_input = {"nested": {"items": [1, {"accepted": True}]}}
    linearization = PhysicalLinearizationV1(
        **{
            **_linearization().__dict__,
            "metadata": metadata_input,
        }
    )
    artifact_id = linearization.artifact_id

    metadata_input["nested"]["items"][1]["accepted"] = False
    assert linearization.metadata["nested"]["items"][1]["accepted"] is True
    assert linearization.artifact_id == artifact_id

    with pytest.raises(TypeError):
        linearization.metadata["nested"]["items"][1]["accepted"] = False
    with pytest.raises(TypeError):
        linearization.metadata["nested"]["items"].append("mutated")

    copied = copy.deepcopy(linearization.metadata)
    copied["nested"]["items"].append("copy-only")
    assert "copy-only" not in linearization.metadata["nested"]["items"]


def test_linearization_round_trip_revalidates_content_address(tmp_path: Path) -> None:
    source = _linearization()
    path = tmp_path / "linearization.npz"

    save_physical_linearization(path, source)
    restored = load_physical_linearization(path)

    assert restored.artifact_id == source.artifact_id
    np.testing.assert_array_equal(restored.frame_ids, source.frame_ids)
    assert not restored.state_jacobian.flags.writeable


def test_linearization_loader_rejects_missing_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "missing-descriptor.npz"
    np.savez_compressed(path, frame_ids=np.asarray([1], dtype=np.int64))

    with pytest.raises(ValueError, match="no descriptor_json"):
        load_physical_linearization(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"schema_name": "unsupported"}, "unsupported physical-linearization schema"),
        ({"schema_version": 2}, "unsupported physical-linearization version"),
        ({"schema_version": True}, "schema_version must be an integer"),
    ),
)
def test_linearization_loader_rejects_descriptor_drift(
    tmp_path: Path,
    changes: Mapping[str, Any],
    message: str,
) -> None:
    source = _linearization()
    descriptor = {
        **source.descriptor(),
        "artifact_id": source.artifact_id,
        **changes,
    }
    path = tmp_path / "descriptor-drift.npz"
    _write_linearization_archive(path, descriptor, source.arrays())

    with pytest.raises(ValueError, match=message):
        load_physical_linearization(path)


def test_linearization_loader_rejects_array_set_drift(tmp_path: Path) -> None:
    source = _linearization()
    descriptor = {**source.descriptor(), "artifact_id": source.artifact_id}
    arrays = source.arrays()
    arrays.pop("physical_response_m")
    path = tmp_path / "array-drift.npz"
    _write_linearization_archive(path, descriptor, arrays)

    with pytest.raises(ValueError, match="array set changed"):
        load_physical_linearization(path)


def test_linearization_loader_rejects_payload_digest_mismatch(tmp_path: Path) -> None:
    source = _linearization()
    descriptor = {**source.descriptor(), "artifact_id": "0" * 64}
    path = tmp_path / "digest-mismatch.npz"
    _write_linearization_archive(path, descriptor, source.arrays())

    with pytest.raises(ValueError, match="digest does not match"):
        load_physical_linearization(path)


def test_nonlinear_closure_fails_large_remainder() -> None:
    linearization = _linearization()
    baseline = np.zeros((3, 3))
    linear = np.ones((3, 3)) * 0.01
    nonlinear = linear.copy()
    nonlinear[0, 0] += 0.1
    closure = evaluate_nonlinear_closure(
        linearization.artifact_id,
        baseline_query_m=baseline,
        linearized_query_m=linear,
        nonlinear_query_m=nonlinear,
        absolute_tolerance_m=0.02,
        relative_tolerance=0.2,
    )
    assert not closure.candidate_valid


def test_nonlinear_closure_metadata_is_immutable_and_id_stable() -> None:
    metadata_input = {"checks": [{"name": "replay", "accepted": True}]}
    closure = evaluate_nonlinear_closure(
        _linearization().artifact_id,
        baseline_query_m=np.zeros((1, 3)),
        linearized_query_m=np.asarray([[0.01, 0.0, 0.0]]),
        nonlinear_query_m=np.asarray([[0.01, 0.0, 0.0]]),
        absolute_tolerance_m=0.0,
        relative_tolerance=0.0,
        metadata=metadata_input,
    )
    closure_id = closure.closure_id

    metadata_input["checks"][0]["accepted"] = False
    assert closure.metadata["checks"][0]["accepted"] is True
    assert closure.closure_id == closure_id
    with pytest.raises(TypeError):
        closure.metadata["checks"][0]["accepted"] = False


def test_nonlinear_closure_requires_a_genuine_boolean_decision() -> None:
    with pytest.raises(ValueError, match="candidate_valid must be a boolean"):
        NonlinearClosureV1(
            linearization_artifact_id=A,
            absolute_error_m=0.0,
            relative_error=0.0,
            absolute_tolerance_m=0.0,
            relative_tolerance=0.0,
            candidate_valid=cast(bool, 1),
        )


def test_nonlinear_closure_rejects_inconsistent_decision() -> None:
    with pytest.raises(ValueError, match="does not match"):
        NonlinearClosureV1(
            linearization_artifact_id=A,
            absolute_error_m=0.0,
            relative_error=0.0,
            absolute_tolerance_m=0.0,
            relative_tolerance=0.0,
            candidate_valid=False,
        )


@pytest.mark.parametrize(
    ("absolute_tolerance", "relative_tolerance", "floor"),
    (
        (-1.0, 0.0, 1e-12),
        (0.0, np.nan, 1e-12),
        (0.0, 0.0, 0.0),
    ),
)
def test_nonlinear_closure_rejects_invalid_tolerances(
    absolute_tolerance: float,
    relative_tolerance: float,
    floor: float,
) -> None:
    with pytest.raises(ValueError, match="closure tolerances"):
        evaluate_nonlinear_closure(
            A,
            baseline_query_m=np.zeros((1, 3)),
            linearized_query_m=np.zeros((1, 3)),
            nonlinear_query_m=np.zeros((1, 3)),
            absolute_tolerance_m=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            denominator_floor_m=floor,
        )


@dataclass
class Belief:
    artifact_id: str
    payload: np.ndarray


def _guard_decision(
    *,
    inference_admissible: object = True,
    regret_guard_accepted: object = False,
    reason: object = "source certificate rejected",
    metadata: Mapping[str, Any] | None = None,
) -> CompleteBeliefGuardDecisionV1:
    return CompleteBeliefGuardDecisionV1(
        baseline_belief_id="d" * 64,
        candidate_belief_id="e" * 64,
        common_domain_id="f" * 64,
        certificate_id="9" * 64,
        inference_admissible=cast(bool, inference_admissible),
        regret_guard_accepted=cast(bool, regret_guard_accepted),
        reason=cast(str, reason),
        metadata={} if metadata is None else metadata,
    )


def test_complete_belief_fallback_reuses_exact_baseline_object() -> None:
    baseline = Belief("d" * 64, np.asarray([0.0, -0.0], dtype=np.float32))
    candidate = Belief("e" * 64, np.asarray([1.0, 1.0], dtype=np.float32))
    selected, manifest = select_complete_belief(
        baseline,
        candidate,
        _guard_decision(),
    )
    assert selected is baseline
    assert manifest.selected_belief_id == baseline.artifact_id
    assert not manifest.selected_candidate


def test_complete_belief_accepts_candidate_only_after_both_gates() -> None:
    baseline = Belief("d" * 64, np.asarray([0.0]))
    candidate = Belief("e" * 64, np.asarray([1.0]))

    selected, manifest = select_complete_belief(
        baseline,
        candidate,
        _guard_decision(
            inference_admissible=True,
            regret_guard_accepted=True,
        ),
    )

    assert selected is candidate
    assert manifest.selected_candidate
    assert manifest.reason == "guard-accepted"


def test_complete_belief_reports_numerical_rejection() -> None:
    baseline = Belief("d" * 64, np.asarray([0.0]))
    candidate = Belief("e" * 64, np.asarray([1.0]))

    selected, manifest = select_complete_belief(
        baseline,
        candidate,
        _guard_decision(inference_admissible=False),
    )

    assert selected is baseline
    assert manifest.reason == "inference-rejected"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("inference_admissible", 1),
        ("inference_admissible", np.int64(1)),
        ("regret_guard_accepted", 0.0),
    ),
)
def test_complete_belief_guard_requires_genuine_booleans(
    field: str,
    value: object,
) -> None:
    settings = {
        "inference_admissible": True,
        "regret_guard_accepted": False,
        field: value,
    }
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        _guard_decision(**settings)


@pytest.mark.parametrize("reason", ("", 1))
def test_complete_belief_guard_rejects_invalid_reason(reason: object) -> None:
    with pytest.raises(ValueError, match="reason must be nonempty"):
        _guard_decision(reason=reason)


def test_complete_belief_guard_rejects_acceptance_after_inference_failure() -> None:
    with pytest.raises(ValueError, match="requires inference_admissible"):
        _guard_decision(
            inference_admissible=False,
            regret_guard_accepted=True,
        )


def test_complete_belief_selection_rejects_unbound_beliefs() -> None:
    baseline = Belief("0" * 64, np.asarray([0.0]))
    candidate = Belief("e" * 64, np.asarray([1.0]))

    with pytest.raises(ValueError, match="does not bind the baseline"):
        select_complete_belief(baseline, candidate, _guard_decision())

    baseline = Belief("d" * 64, np.asarray([0.0]))
    candidate = Belief("0" * 64, np.asarray([1.0]))
    with pytest.raises(ValueError, match="does not bind the candidate"):
        select_complete_belief(baseline, candidate, _guard_decision())


def test_complete_belief_selection_contract_rejects_contradictions() -> None:
    decision = _guard_decision()
    base = {
        "baseline_belief_id": "d" * 64,
        "candidate_belief_id": "e" * 64,
        "common_domain_id": "f" * 64,
        "guard_decision_id": decision.decision_id,
        "selected_belief_id": "d" * 64,
        "selected_candidate": False,
        "reason": "regret-guard-rejected",
    }

    with pytest.raises(ValueError, match="contradicts routing decision"):
        CompleteBeliefSelectionV1(
            **{
                **base,
                "selected_belief_id": "e" * 64,
            }
        )
    with pytest.raises(ValueError, match="reason must be nonempty"):
        CompleteBeliefSelectionV1(**{**base, "reason": ""})
    with pytest.raises(ValueError, match="selected_candidate must be a boolean"):
        CompleteBeliefSelectionV1(
            **{
                **base,
                "selected_candidate": cast(bool, 0),
            }
        )


def test_complete_belief_metadata_is_immutable_and_ids_are_stable() -> None:
    metadata_input = {"certificate": {"groups": ["object-1"]}}
    decision = _guard_decision(metadata=metadata_input)
    decision_id = decision.decision_id

    metadata_input["certificate"]["groups"].append("object-2")
    assert decision.metadata["certificate"]["groups"] == ["object-1"]
    assert decision.decision_id == decision_id
    with pytest.raises(TypeError):
        decision.metadata["certificate"]["groups"].append("mutated")

    baseline = Belief("d" * 64, np.asarray([0.0]))
    candidate = Belief("e" * 64, np.asarray([1.0]))
    selection_metadata = {"routing": {"reasons": ["source-only"]}}
    _, selection = select_complete_belief(
        baseline,
        candidate,
        decision,
        metadata=selection_metadata,
    )
    selection_id = selection.selection_id
    selection_metadata["routing"]["reasons"].append("target-informed")
    assert selection.metadata["routing"]["reasons"] == ["source-only"]
    assert selection.selection_id == selection_id
    with pytest.raises(TypeError):
        selection.metadata["routing"]["reasons"].append("mutated")


def _one_step_propagated_state_problem() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    innovation = np.zeros((1, 3, 3), dtype=np.float64)
    innovation[0, :, 0] = np.asarray([1.0, 1.0, 10.0])
    available = np.ones((1, 3), dtype=bool)
    response = np.zeros((1, 3, 3, 1), dtype=np.float64)
    response[0, :, 0, 0] = 1.0
    bias_basis = np.zeros((3, 0), dtype=np.float64)
    return innovation, available, response, bias_basis


def test_propagated_state_final_system_uses_returned_robust_weights() -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()
    config = PropagatedStateBeliefConfig(
        observation_std_m=1.0,
        state_weight_prior_std=10.0,
        effective_samples_per_frame=3.0,
        effective_frame_count=1.0,
        maximum_iterations=1,
        reject_unidentifiable_state=False,
    )

    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        config=config,
    )

    assert result.accepted
    robust = result.robust_weights[0]
    expected_precision = 1.0 / config.state_weight_prior_std**2 + np.sum(robust)
    expected_right = float(robust @ innovation[0, :, 0])
    assert result.state_weights[0] == pytest.approx(
        expected_right / expected_precision,
        rel=1e-12,
        abs=1e-12,
    )
    assert result.posterior_covariance[0, 0] == pytest.approx(
        1.0 / expected_precision,
        rel=1e-12,
        abs=1e-12,
    )
    assert result.diagnostics["final_system_uses_returned_robust_weights"] is True
    assert result.diagnostics["posterior_solver"] == "cholesky"


def test_propagated_state_spd_paths_do_not_use_numpy_inverse(monkeypatch) -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()

    def fail_inverse(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("np.linalg.inv must not be used for SPD systems")

    monkeypatch.setattr(np.linalg, "inv", fail_inverse)
    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        state_prior_covariance=np.asarray([[4.0]]),
        config=PropagatedStateBeliefConfig(
            maximum_iterations=2,
            reject_unidentifiable_state=False,
        ),
    )

    assert result.accepted
    np.testing.assert_allclose(
        result.posterior_covariance,
        result.posterior_covariance.T,
        atol=0.0,
        rtol=0.0,
    )


def test_propagated_state_rejects_non_positive_definite_prior() -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()

    with pytest.raises(ValueError, match="positive definite"):
        infer_propagated_state_belief(
            innovation,
            available,
            response,
            bias_basis,
            observation_variance_m2=np.ones(available.shape),
            state_prior_covariance=np.asarray([[0.0]]),
            config=PropagatedStateBeliefConfig(
                maximum_iterations=1,
                reject_unidentifiable_state=False,
            ),
        )


def test_propagated_state_ill_conditioned_posterior_falls_back() -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()

    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        config=PropagatedStateBeliefConfig(
            maximum_iterations=1,
            maximum_condition_number=0.5,
            reject_unidentifiable_state=False,
        ),
    )

    assert not result.accepted
    assert result.reason == "ill-conditioned-posterior"


def test_propagated_state_final_cholesky_failure_falls_back(monkeypatch) -> None:
    innovation, available, response, bias_basis = _one_step_propagated_state_problem()
    original_cholesky = np.linalg.cholesky
    call_count = 0

    def fail_second_cholesky(matrix: np.ndarray) -> np.ndarray:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise np.linalg.LinAlgError("forced final-system failure")
        return original_cholesky(matrix)

    monkeypatch.setattr(np.linalg, "cholesky", fail_second_cholesky)
    result = infer_propagated_state_belief(
        innovation,
        available,
        response,
        bias_basis,
        observation_variance_m2=np.ones(available.shape),
        config=PropagatedStateBeliefConfig(
            maximum_iterations=1,
            reject_unidentifiable_state=False,
        ),
    )

    assert call_count == 2
    assert not result.accepted
    assert result.reason == "singular-posterior"
