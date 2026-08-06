from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.persistent_prob4d_visual_bias import (
    PersistentVisualBiasCandidateV1,
    apply_persistent_visual_bias_update,
    predict_persistent_visual_bias_run,
    propose_persistent_visual_bias_update,
    select_persistent_visual_bias_candidate,
    start_persistent_visual_bias_run,
)
from bayesian_phystwin.prob4d_visual_bias_stream import (
    PROB4D_VISUAL_BIAS_MODEL_SCHEMA,
    Prob4DVisualBiasStreamConsumptionBindingV1,
    Prob4DVisualBiasStreamUpdateBindingV1,
    ValidatedProb4DVisualBiasStreamV1,
)
from bayesian_phystwin.prob4d_visual_bias_update import (
    PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
    _array_descriptor,
    _canonical_id,
)


def _sha(character: str) -> str:
    return character * 64


def _model_id(covariance: np.ndarray) -> str:
    return _canonical_id(
        {
            "schema": PROB4D_VISUAL_BIAS_MODEL_SCHEMA,
            "bias_ids": ["camera-0"],
            "basis_names": ["tx"],
            "joint_bias_covariance": _array_descriptor(covariance),
            "orthogonalization_semantics": (PROB4D_VISUAL_BIAS_ORTHOGONALIZATION),
            "gauge_projection_tolerance": 1e-6,
            "model_metadata": {"calibration": "source-only"},
        }
    )


def _binding(
    row_counts: tuple[int, ...],
    *,
    bias_covariance: np.ndarray | None = None,
) -> Prob4DVisualBiasStreamConsumptionBindingV1:
    covariance = (
        np.array([[0.25]], dtype=np.float64)
        if bias_covariance is None
        else bias_covariance
    )
    model_id = _model_id(covariance)
    updates: list[Prob4DVisualBiasStreamUpdateBindingV1] = []
    previous: str | None = None
    row_start = 0
    update_characters = ("1", "2", "3")
    sidecar_characters = ("a", "b", "c")
    observation_characters = ("d", "e", "f")
    identity_characters = ("7", "8", "9")
    for index, row_count in enumerate(row_counts):
        update = Prob4DVisualBiasStreamUpdateBindingV1(
            bias_model_id=model_id,
            observation_stream_update_id=_sha(update_characters[index]),
            visual_bias_artifact_id=_sha(sidecar_characters[index]),
            observation_artifact_id=_sha(observation_characters[index]),
            observation_identity_sha256=_sha(identity_characters[index]),
            frame_start=2 * index,
            frame_stop_exclusive=2 * index + 2,
            row_start=row_start,
            row_stop_exclusive=row_start + row_count,
            maximum_gauge_projection=0.0,
            previous_update_id=previous,
        )
        updates.append(update)
        previous = update.update_id
        row_start += row_count

    row_update_indices = np.concatenate(
        [
            np.full(row_count, index, dtype=np.int64)
            for index, row_count in enumerate(row_counts)
        ]
    )
    row_bias_indices = np.zeros(row_start, dtype=np.int64)
    bias_jacobian = np.zeros((row_start, 3, 1), dtype=np.float64)
    bias_jacobian[:, 0, 0] = 1.0
    stream = ValidatedProb4DVisualBiasStreamV1(
        stream_key="case-a/camera-0",
        bias_ids=("camera-0",),
        basis_names=("tx",),
        orthogonalization_semantics=PROB4D_VISUAL_BIAS_ORTHOGONALIZATION,
        gauge_projection_tolerance=1e-6,
        updates=tuple(updates),
        row_update_indices=row_update_indices,
        row_bias_indices=row_bias_indices,
        bias_jacobian=bias_jacobian,
        joint_bias_covariance=covariance,
        model_metadata={"calibration": "source-only"},
        metadata={"protocol": "persistent-solver-test"},
        bias_model_id=model_id,
    )
    return Prob4DVisualBiasStreamConsumptionBindingV1(
        visual_bias_stream=stream,
        factor_stream_artifact_id=_sha("4"),
        factor_stream_update_ids=tuple(
            update.observation_stream_update_id for update in updates
        ),
        observation_binding_ids=tuple(
            _sha(character) for character in ("5", "6", "0")[: len(updates)]
        ),
        recursive_nuisance_policy_id=_sha("a"),
        nuisance_family_id=stream.nuisance_family_id,
    )


def _measurement_arrays(
    row_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    measurement = np.zeros((row_count, 3), dtype=np.float64)
    measurement[:, 0] = 1.0
    physical_jacobian = np.zeros((row_count, 3, 1), dtype=np.float64)
    physical_jacobian[:, 0, 0] = 1.0
    covariance = np.repeat(
        (0.1 * np.eye(3, dtype=np.float64))[None],
        row_count,
        axis=0,
    )
    return measurement, physical_jacobian, covariance


def _innovation(run: object, measurement: np.ndarray) -> np.ndarray:
    belief = run.belief
    prediction = np.zeros_like(measurement)
    prediction[:, 0] = belief.physical_mean[0] + belief.provider_bias_mean[0]
    return measurement - prediction


def _run_all(row_counts: tuple[int, ...]):
    binding = _binding(row_counts)
    run = start_persistent_visual_bias_run(
        binding,
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.zeros(1, dtype=np.float64),
        physical_covariance=np.eye(1, dtype=np.float64),
    )
    measurement, jacobian, covariance = _measurement_arrays(sum(row_counts))
    row_start = 0
    for index, row_count in enumerate(row_counts):
        row_slice = slice(row_start, row_start + row_count)
        run, _ = apply_persistent_visual_bias_update(
            run,
            innovation_xyz=_innovation(run, measurement[row_slice]),
            physical_jacobian=jacobian[row_slice],
            conditional_covariance=covariance[row_slice],
            physical_linearization_id=_sha(str(index + 1)),
            accepted=True,
            reason="accepted-by-test-guard",
        )
        row_start += row_count
    return run


def test_recursive_partition_matches_stacked_update() -> None:
    stacked = _run_all((4,))
    recursive = _run_all((2, 2))

    np.testing.assert_allclose(
        stacked.belief.physical_mean,
        recursive.belief.physical_mean,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        stacked.belief.provider_bias_mean,
        recursive.belief.provider_bias_mean,
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        stacked.belief.joint_covariance,
        recursive.belief.joint_covariance,
        atol=1e-12,
        rtol=1e-12,
    )


def test_rejection_is_exact_object_fallback_and_consumes_update() -> None:
    binding = _binding((1, 1))
    run = start_persistent_visual_bias_run(
        binding,
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.zeros(1, dtype=np.float64),
        physical_covariance=np.eye(1, dtype=np.float64),
    )
    measurement, jacobian, covariance = _measurement_arrays(1)
    candidate = propose_persistent_visual_bias_update(
        run,
        innovation_xyz=measurement,
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("1"),
    )
    prior = run.belief
    selected = select_persistent_visual_bias_candidate(
        run,
        candidate,
        accepted=False,
        reason="frozen-regret-guard-rejected",
    )

    assert selected.belief is prior
    assert selected.next_update_index == 1
    assert selected.events[-1].exact_fallback_reproduced is True
    with pytest.raises(ValueError, match="stale or out of order"):
        select_persistent_visual_bias_candidate(
            selected,
            candidate,
            accepted=True,
            reason="replay",
        )


def test_prediction_retains_bias_marginal_and_propagates_cross_covariance() -> None:
    binding = _binding((1, 1))
    run = start_persistent_visual_bias_run(
        binding,
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.zeros(1, dtype=np.float64),
        physical_covariance=np.eye(1, dtype=np.float64),
    )
    measurement, jacobian, covariance = _measurement_arrays(1)
    run, _ = apply_persistent_visual_bias_update(
        run,
        innovation_xyz=measurement,
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("1"),
        accepted=True,
        reason="accepted-by-test-guard",
    )
    bias_covariance = np.array(run.belief.provider_bias_covariance, copy=True)
    cross_covariance = np.array(
        run.belief.physical_bias_cross_covariance,
        copy=True,
    )

    predicted = predict_persistent_visual_bias_run(
        run,
        physical_transition=np.array([[2.0]], dtype=np.float64),
        process_covariance=np.array([[0.2]], dtype=np.float64),
        transition_id=_sha("b"),
    )

    np.testing.assert_allclose(
        predicted.belief.provider_bias_covariance,
        bias_covariance,
    )
    np.testing.assert_allclose(
        predicted.belief.physical_bias_cross_covariance,
        2.0 * cross_covariance,
    )
    assert predicted.next_update_index == run.next_update_index


def test_singular_source_bias_prior_is_supported_exactly() -> None:
    binding = _binding(
        (1,),
        bias_covariance=np.zeros((1, 1), dtype=np.float64),
    )
    run = start_persistent_visual_bias_run(
        binding,
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.zeros(1, dtype=np.float64),
        physical_covariance=np.eye(1, dtype=np.float64),
    )
    measurement, jacobian, covariance = _measurement_arrays(1)
    run, _ = apply_persistent_visual_bias_update(
        run,
        innovation_xyz=measurement,
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("1"),
        accepted=True,
        reason="accepted-by-test-guard",
    )

    np.testing.assert_array_equal(
        run.belief.provider_bias_mean,
        np.zeros(1, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        run.belief.provider_bias_covariance,
        np.zeros((1, 1), dtype=np.float64),
    )


def test_reinstantiating_bias_prior_is_an_overconfident_negative_control() -> None:
    binding = _binding((1, 1))
    run = start_persistent_visual_bias_run(
        binding,
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.zeros(1, dtype=np.float64),
        physical_covariance=np.eye(1, dtype=np.float64),
    )
    measurement, jacobian, covariance = _measurement_arrays(1)
    run, _ = apply_persistent_visual_bias_update(
        run,
        innovation_xyz=measurement,
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("1"),
        accepted=True,
        reason="accepted-by-test-guard",
    )
    correct, _ = apply_persistent_visual_bias_update(
        run,
        innovation_xyz=_innovation(run, measurement),
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("2"),
        accepted=True,
        reason="accepted-by-test-guard",
    )

    reset = start_persistent_visual_bias_run(
        _binding((1,)),
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.array(run.belief.physical_mean, copy=True),
        physical_covariance=np.array(run.belief.physical_covariance, copy=True),
    )
    naive, _ = apply_persistent_visual_bias_update(
        reset,
        innovation_xyz=_innovation(reset, measurement),
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("2"),
        accepted=True,
        reason="negative-control",
    )

    assert (
        naive.belief.physical_covariance[0, 0]
        < correct.belief.physical_covariance[0, 0]
    )


def test_solver_does_not_materialize_complete_global_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding((2,))

    def _forbidden_global_design(
        self: ValidatedProb4DVisualBiasStreamV1,
    ) -> np.ndarray:
        raise AssertionError("dense global design must not be materialized")

    monkeypatch.setattr(
        ValidatedProb4DVisualBiasStreamV1,
        "global_design",
        _forbidden_global_design,
    )
    run = start_persistent_visual_bias_run(
        binding,
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.zeros(1, dtype=np.float64),
        physical_covariance=np.eye(1, dtype=np.float64),
    )
    measurement, jacobian, covariance = _measurement_arrays(2)
    candidate = propose_persistent_visual_bias_update(
        run,
        innovation_xyz=measurement,
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("1"),
    )

    assert candidate.information_gain_nats > 0.0


def test_candidate_identity_is_tamper_evident() -> None:
    binding = _binding((1,))
    run = start_persistent_visual_bias_run(
        binding,
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.zeros(1, dtype=np.float64),
        physical_covariance=np.eye(1, dtype=np.float64),
    )
    measurement, jacobian, covariance = _measurement_arrays(1)
    candidate = propose_persistent_visual_bias_update(
        run,
        innovation_xyz=measurement,
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("1"),
    )

    assert isinstance(candidate, PersistentVisualBiasCandidateV1)
    with pytest.raises(ValueError, match="candidate ID mismatch"):
        replace(candidate, candidate_id=_sha("0"))


def _single_candidate():
    binding = _binding((1,))
    run = start_persistent_visual_bias_run(
        binding,
        physical_state_domain_id="physical-state-v1",
        physical_mean=np.zeros(1, dtype=np.float64),
        physical_covariance=np.eye(1, dtype=np.float64),
    )
    measurement, jacobian, covariance = _measurement_arrays(1)
    candidate = propose_persistent_visual_bias_update(
        run,
        innovation_xyz=measurement,
        physical_jacobian=jacobian,
        conditional_covariance=covariance,
        physical_linearization_id=_sha("1"),
    )
    return run, candidate


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("factor_stream_update_id", "factor-stream update differs"),
        ("observation_binding_id", "observation binding differs"),
    ],
)
def test_selection_rejects_forged_candidate_bindings(
    field: str,
    message: str,
) -> None:
    run, candidate = _single_candidate()
    forged = replace(candidate, **{field: _sha("f")}, candidate_id=None)
    with pytest.raises(ValueError, match=message):
        select_persistent_visual_bias_candidate(
            run,
            forged,
            accepted=False,
            reason="reject-forged-binding",
        )


def test_candidate_rejects_forged_posterior_lineage() -> None:
    _, candidate = _single_candidate()
    lineage = dict(candidate.posterior_belief.metadata)
    lineage["source_update_index"] = 7
    posterior = replace(
        candidate.posterior_belief,
        metadata=lineage,
        belief_id=None,
    )
    with pytest.raises(ValueError, match="source update index"):
        replace(candidate, posterior_belief=posterior, candidate_id=None)

    lineage = dict(candidate.posterior_belief.metadata)
    lineage["physical_linearization_id"] = _sha("f")
    posterior = replace(
        candidate.posterior_belief,
        metadata=lineage,
        belief_id=None,
    )
    with pytest.raises(ValueError, match="physical linearization"):
        replace(candidate, posterior_belief=posterior, candidate_id=None)


def test_selection_rejects_incompatible_posterior_contract() -> None:
    run, candidate = _single_candidate()
    wrong_domain = replace(
        candidate.posterior_belief,
        physical_state_domain_id="different-state-domain",
        belief_id=None,
    )
    with pytest.raises(ValueError, match="physical state domain"):
        select_persistent_visual_bias_candidate(
            run,
            replace(
                candidate,
                posterior_belief=wrong_domain,
                candidate_id=None,
            ),
            accepted=False,
            reason="reject-domain-mismatch",
        )

    wrong_root = replace(
        candidate.posterior_belief,
        bias_covariance_root=np.zeros_like(
            candidate.posterior_belief.bias_covariance_root
        ),
        belief_id=None,
    )
    with pytest.raises(ValueError, match="covariance root"):
        select_persistent_visual_bias_candidate(
            run,
            replace(
                candidate,
                posterior_belief=wrong_root,
                candidate_id=None,
            ),
            accepted=False,
            reason="reject-root-mismatch",
        )


def test_selection_rejects_noncontracting_or_misreported_candidate() -> None:
    run, candidate = _single_candidate()
    expanded = replace(
        candidate.posterior_belief,
        joint_covariance=2.0 * np.asarray(run.belief.joint_covariance),
        belief_id=None,
    )
    with pytest.raises(ValueError, match="measurement contraction"):
        select_persistent_visual_bias_candidate(
            run,
            replace(
                candidate,
                posterior_belief=expanded,
                information_gain_nats=0.0,
                candidate_id=None,
            ),
            accepted=False,
            reason="reject-expanded-covariance",
        )

    with pytest.raises(ValueError, match="information gain"):
        select_persistent_visual_bias_candidate(
            run,
            replace(
                candidate,
                information_gain_nats=candidate.information_gain_nats + 1.0,
                candidate_id=None,
            ),
            accepted=False,
            reason="reject-misreported-gain",
        )


def test_persistent_belief_views_are_irreversibly_immutable() -> None:
    run, _ = _single_candidate()
    for array in (
        run.belief.physical_mean,
        run.belief.bias_latent_mean,
        run.belief.joint_covariance,
        run.belief.bias_covariance_root,
        run.belief.physical_covariance,
        run.belief.physical_bias_cross_covariance,
        run.belief.bias_latent_covariance,
        run.belief.provider_bias_mean,
        run.belief.provider_bias_covariance,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
