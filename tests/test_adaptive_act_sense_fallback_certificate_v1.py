from __future__ import annotations

import itertools

import numpy as np
import pytest

from bayesian_phystwin.adaptive_act_sense_fallback_certificate_v1 import (
    ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY,
    AdaptivePlanV1,
    adaptive_act_sense_fallback_certificate,
    controlled_adaptive_router_demo,
    controlled_stochastic_xor_demo,
)


def problem():
    hypotheses = tuple(itertools.product((0, 1), repeat=3))
    correct = np.asarray([x ^ y for x, y, _ in hypotheses], dtype=np.int64)
    losses = np.empty((8, 3), dtype=float)
    losses[:, 0] = (correct != 0).astype(float)
    losses[:, 1] = (correct != 1).astype(float)
    losses[:, 2] = 0.45

    def noisy_bit(axis: int, accuracy: float):
        result = np.empty((8, 2), dtype=float)
        for index, hypothesis in enumerate(hypotheses):
            bit = hypothesis[axis]
            result[index, bit] = accuracy
            result[index, 1 - bit] = 1.0 - accuracy
        return result

    return dict(
        prior_weights=np.full(8, 1 / 8),
        quotient_weights=[1.0],
        class_index=np.zeros(8, dtype=int),
        terminal_loss_by_hypothesis_action=losses,
        probe_outcome_probability=(
            noisy_bit(0, 0.95),
            noisy_bit(1, 0.95),
            noisy_bit(2, 0.99),
        ),
        probe_costs=[0.05, 0.05, 0.01],
        fallback_action_index=2,
        regret_tolerance=0.25,
        max_frontier_count=20_000,
    )


def test_one_probe_falls_back_but_two_probe_policy_is_certified():
    one = adaptive_act_sense_fallback_certificate(max_depth=1, **problem())
    two = adaptive_act_sense_fallback_certificate(max_depth=2, **problem())

    assert one.output_mode == "fallback"
    assert one.output_plan.action_index == 2
    assert two.output_mode == "sense"
    assert two.output_plan.probe_depth == 2
    assert two.worst_case_regret[two.output_plan_index] <= 0.25
    assert set(two.output_plan.canonical_key().__repr__())  # deterministic key exists


def test_two_probe_plan_ignores_cheaper_nuisance_probe():
    two = adaptive_act_sense_fallback_certificate(max_depth=2, **problem())
    key = repr(two.output_plan.canonical_key())
    # Probe index 2 is the nuisance bit. It must not appear in the chosen tree.
    assert "(1, 2," not in key


def test_expected_loss_matches_noisy_xor_formula():
    two = adaptive_act_sense_fallback_certificate(max_depth=2, **problem())
    losses = two.loss_by_hypothesis_plan[:, two.output_plan_index]
    # Two independent 95%-accurate bits give XOR error probability
    # 2*0.95*0.05 = 0.095; two probes cost 0.10.
    np.testing.assert_allclose(losses, 0.195, atol=1e-12)


def test_output_arrays_are_bytes_backed_and_do_not_alias_inputs():
    values = problem()
    prior = values["prior_weights"]
    first_probe = values["probe_outcome_probability"][0]
    certificate = adaptive_act_sense_fallback_certificate(max_depth=2, **values)

    prior[:] = 0.0
    first_probe[:] = 0.0
    assert np.all(np.isfinite(certificate.loss_by_hypothesis_plan))
    for array in (
        certificate.loss_by_hypothesis_plan,
        certificate.pairwise_worst_case_gap,
        certificate.worst_case_regret,
        certificate.minimizer_mask,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_nonunique_minimizer_returns_exact_fallback():
    certificate = adaptive_act_sense_fallback_certificate(
        prior_weights=[0.5, 0.5],
        quotient_weights=[1.0],
        class_index=[0, 0],
        terminal_loss_by_hypothesis_action=[[0.0, 0.0], [0.0, 0.0]],
        probe_outcome_probability=[[[1.0], [1.0]]],
        probe_costs=[1.0],
        fallback_action_index=1,
        max_depth=1,
        regret_tolerance=0.0,
    )
    assert certificate.candidate_plan_index is None
    assert certificate.output_mode == "fallback"
    assert certificate.output_plan.action_index == 1


def test_validation_rejects_bad_probe_model():
    values = problem()
    probes = list(values["probe_outcome_probability"])
    probes[0] = np.full((8, 2), 0.6)
    values["probe_outcome_probability"] = probes
    with pytest.raises(ValueError, match="sum to one"):
        adaptive_act_sense_fallback_certificate(max_depth=2, **values)


def test_demo_is_content_addressed_and_preserves_state_ambiguity():
    result = controlled_stochastic_xor_demo()
    assert result["one_probe"]["output_mode"] == "fallback"
    assert result["two_probe"]["output_mode"] == "sense"
    assert result["state_remains_unidentified"] is True
    assert len(result["result_id"]) == 64


def test_zero_depth_allows_a_direct_only_certificate():
    certificate = adaptive_act_sense_fallback_certificate(max_depth=0, **problem())
    assert certificate.output_mode == "fallback"
    assert certificate.output_plan.probe_depth == 0


def test_adaptive_router_selects_different_second_probes_by_outcome():
    result = controlled_adaptive_router_demo()
    assert result["direct_only"]["output_mode"] == "fallback"
    assert result["one_probe"]["output_mode"] == "fallback"
    assert result["two_probe"]["output_mode"] == "sense"
    assert result["selected_second_probe_by_router_outcome"] == (1, 2)
    assert result["nuisance_probe_selected"] is False
    assert result["two_probe"]["candidate_worst_case_regret"] == pytest.approx(0.129)
    assert result["best_nonadaptive_fixed_two_probe"]["worst_case_loss"] == (
        pytest.approx(0.475)
    )
    assert result["adaptive_worst_case_improvement_over_best_fixed_two_probe"] == (
        pytest.approx(0.346)
    )
    assert result["state_remains_unidentified"] is True


def test_multiclass_quotient_gap_matches_explicit_vertex_enumeration():
    losses = np.array(
        [
            [0.0, 2.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 3.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    )
    class_index = np.array([0, 0, 1, 1])
    quotient = np.array([0.25, 0.75])
    certificate = adaptive_act_sense_fallback_certificate(
        prior_weights=np.full(4, 0.25),
        quotient_weights=quotient,
        class_index=class_index,
        terminal_loss_by_hypothesis_action=losses,
        probe_outcome_probability=(),
        probe_costs=[],
        fallback_action_index=2,
        max_depth=0,
        regret_tolerance=1.0,
    )

    vertices = []
    for first in (0, 1):
        for second in (2, 3):
            weight = np.zeros(4)
            weight[first] = quotient[0]
            weight[second] = quotient[1]
            vertices.append(weight)
    expected = np.empty((3, 3))
    for candidate in range(3):
        for comparator in range(3):
            expected[candidate, comparator] = max(
                float(weight @ (losses[:, candidate] - losses[:, comparator]))
                for weight in vertices
            )
    np.fill_diagonal(expected, 0.0)

    np.testing.assert_allclose(certificate.pairwise_worst_case_gap, expected)
    np.testing.assert_allclose(
        certificate.worst_case_regret,
        np.maximum(np.max(expected, axis=1), 0.0),
    )


def test_probability_masses_are_not_silently_renormalized() -> None:
    values = problem()
    values["prior_weights"] = np.ones(8)
    with pytest.raises(ValueError, match="sum to one"):
        adaptive_act_sense_fallback_certificate(max_depth=2, **values)


def test_public_plan_constructor_rejects_malformed_trees() -> None:
    with pytest.raises(ValueError, match="act plans"):
        AdaptivePlanV1(mode="act", action_index=0, probe_index=0)
    with pytest.raises(ValueError, match="sense plans"):
        AdaptivePlanV1(mode="sense", probe_index=0)
    with pytest.raises(ValueError, match="mode"):
        AdaptivePlanV1(mode="unknown")  # type: ignore[arg-type]


def test_claim_boundary_exposes_conditional_independence() -> None:
    assert "conditionally independent" in (
        ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY
    )
    assert "does not validate" in (
        ADAPTIVE_ACT_SENSE_FALLBACK_CERTIFICATE_CLAIM_BOUNDARY
    )
