from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.interventional_cause_estimation_v1 import (
    CauseQueryEstimateStatus,
    InterventionPlanStatus,
    estimate_cause_queries,
    plan_diagnostic_interventions,
)
from bayesian_phystwin_experiments.interventional_cause_identifiability_v1 import (
    CauseResponseSignatureV1,
    InterventionalCauseIdentifiabilityCertificateV1,
    InterventionResponseBlockV1,
)

SHA = "a" * 64


def _cause(
    cause_id: str,
    intervention_ids: tuple[str, ...],
    blocks: tuple[np.ndarray, ...],
    query: np.ndarray,
) -> CauseResponseSignatureV1:
    return CauseResponseSignatureV1(
        cause_id=cause_id,
        latent_coordinates_id=SHA,
        cause_query_id=SHA,
        intervention_blocks=tuple(
            InterventionResponseBlockV1(
                intervention_id=intervention_id,
                response_signature_id=SHA,
                whitened_response_signature=block,
            )
            for intervention_id, block in zip(
                intervention_ids,
                blocks,
                strict=True,
            )
        ),
        cause_query_map=query,
    )


def _certificate(
    causes: tuple[CauseResponseSignatureV1, ...],
    nuisance: np.ndarray | None = None,
) -> InterventionalCauseIdentifiabilityCertificateV1:
    row_count = sum(
        block.observation_dimension for block in causes[0].intervention_blocks
    )
    if nuisance is None:
        nuisance = np.empty((row_count, 0), dtype=np.float64)
    return InterventionalCauseIdentifiabilityCertificateV1(
        observation_whitening_id=SHA,
        declared_nuisance_id=SHA,
        cause_family_id=SHA,
        cause_signatures=causes,
        joint_whitened_nuisance_design=nuisance,
    )


def test_quantitative_estimate_recovers_two_simultaneous_causes() -> None:
    interventions = ("action-a", "action-b")
    certificate = _certificate(
        (
            _cause(
                "cause-a",
                interventions,
                (np.asarray([[1.0]]), np.asarray([[1.0]])),
                np.eye(1),
            ),
            _cause(
                "cause-b",
                interventions,
                (np.asarray([[1.0]]), np.asarray([[-1.0]])),
                np.eye(1),
            ),
        )
    )
    residual = np.asarray([2.0, 4.0], dtype=np.float64)
    bundle = estimate_cause_queries(
        certificate,
        residual,
        residual_noise_variance=0.25,
        residual_noise_radius=0.1,
    )

    assert bundle.by_cause("cause-a").status is CauseQueryEstimateStatus.IDENTIFIABLE
    assert bundle.by_cause("cause-b").status is CauseQueryEstimateStatus.IDENTIFIABLE
    assert bundle.by_cause("cause-a").estimate == pytest.approx([3.0])
    assert bundle.by_cause("cause-b").estimate == pytest.approx([-1.0])
    np.testing.assert_allclose(
        bundle.by_cause("cause-a").covariance, np.asarray([[0.125]])
    )
    assert bundle.by_cause("cause-a").deterministic_error_radius == pytest.approx(
        0.1 / np.sqrt(2.0)
    )
    for estimate in bundle.cause_estimates:
        assert estimate.full_query_interval_valid
        for array in (
            estimate.estimate,
            estimate.covariance,
            estimate.identified_query_map,
            estimate.unresolved_query_map,
            estimate.factor_operator,
        ):
            assert not array.flags.writeable


def test_partial_query_reports_unsupported_component() -> None:
    interventions = ("action-a", "action-b")
    certificate = _certificate(
        (
            _cause(
                "cause-a",
                interventions,
                (
                    np.asarray([[1.0, 0.0]]),
                    np.asarray([[0.0, 0.0]]),
                ),
                np.eye(2),
            ),
            _cause(
                "cause-b",
                interventions,
                (
                    np.asarray([[0.0]]),
                    np.asarray([[1.0]]),
                ),
                np.eye(1),
            ),
        )
    )
    estimate = estimate_cause_queries(
        certificate,
        np.asarray([2.0, 0.0]),
    ).by_cause("cause-a")

    assert estimate.status is CauseQueryEstimateStatus.PARTIALLY_IDENTIFIABLE
    assert estimate.identifiable_energy_fraction == pytest.approx(0.5)
    assert estimate.estimate == pytest.approx([2.0, 0.0])
    assert not estimate.full_query_interval_valid
    assert np.linalg.norm(estimate.unresolved_query_map) == pytest.approx(1.0)


def test_exact_confounding_abstains_instead_of_forcing_a_label() -> None:
    interventions = ("action-a", "action-b")
    shared = (np.asarray([[1.0]]), np.asarray([[1.0]]))
    certificate = _certificate(
        (
            _cause("cause-a", interventions, shared, np.eye(1)),
            _cause("cause-b", interventions, shared, np.eye(1)),
        )
    )
    bundle = estimate_cause_queries(certificate, np.asarray([1.0, 1.0]))

    assert all(
        item.status is CauseQueryEstimateStatus.CONFOUNDED
        for item in bundle.cause_estimates
    )
    assert all(not item.full_query_interval_valid for item in bundle.cause_estimates)
    assert all(
        item.identifiable_energy_fraction == 0.0 for item in bundle.cause_estimates
    )


def test_minimum_cost_planner_rejects_a_cheap_redundant_probe() -> None:
    interventions = (
        "action-0-source",
        "action-1-diagnostic-a",
        "action-2-diagnostic-b",
        "action-3-decoy",
    )
    columns = {
        "cause-a": (1.0, 1.0, 0.0, 0.2),
        "cause-b": (1.0, 0.0, 1.0, 0.2),
        "cause-c": (1.0, -1.0, -1.0, 0.2),
    }
    certificate = _certificate(
        tuple(
            _cause(
                cause_id,
                interventions,
                tuple(np.asarray([[value]]) for value in values),
                np.eye(1),
            )
            for cause_id, values in sorted(columns.items())
        )
    )
    costs = {
        "action-0-source": 0.0,
        "action-1-diagnostic-a": 1.0,
        "action-2-diagnostic-b": 0.8,
        "action-3-decoy": 0.1,
    }
    plan = plan_diagnostic_interventions(
        certificate,
        costs,
        required_intervention_ids=("action-0-source",),
        maximum_interventions=3,
    )

    assert plan.status is InterventionPlanStatus.FULL_IDENTIFICATION
    assert plan.selected_intervention_ids == (
        "action-0-source",
        "action-1-diagnostic-a",
        "action-2-diagnostic-b",
    )
    assert plan.selected_total_cost == pytest.approx(1.8)
    assert "action-3-decoy" not in plan.selected_intervention_ids

    limited = plan_diagnostic_interventions(
        certificate,
        costs,
        required_intervention_ids=("action-0-source",),
        maximum_interventions=2,
    )
    assert limited.status is InterventionPlanStatus.BUDGET_LIMITED_PARTIAL
    assert not limited.selected_score.all_required_causes_identified


def test_augmented_intervention_cannot_increase_blue_variance_in_example() -> None:
    interventions = ("action-a", "action-b", "action-c")
    certificate = _certificate(
        (
            _cause(
                "cause-a",
                interventions,
                (
                    np.asarray([[1.0]]),
                    np.asarray([[1.0]]),
                    np.asarray([[1.0]]),
                ),
                np.eye(1),
            ),
            _cause(
                "cause-b",
                interventions,
                (
                    np.asarray([[1.0]]),
                    np.asarray([[-1.0]]),
                    np.asarray([[0.0]]),
                ),
                np.eye(1),
            ),
        )
    )
    residual = np.zeros(3)
    two = estimate_cause_queries(
        certificate,
        residual,
        intervention_ids=("action-a", "action-b"),
    ).by_cause("cause-a")
    three = estimate_cause_queries(certificate, residual).by_cause("cause-a")

    assert two.status is CauseQueryEstimateStatus.IDENTIFIABLE
    assert three.status is CauseQueryEstimateStatus.IDENTIFIABLE
    assert three.covariance[0, 0] <= two.covariance[0, 0] + 1e-12


def test_invalid_intervention_order_and_cost_roster_fail_closed() -> None:
    interventions = ("action-a", "action-b")
    certificate = _certificate(
        (
            _cause(
                "cause-a",
                interventions,
                (np.asarray([[1.0]]), np.asarray([[1.0]])),
                np.eye(1),
            ),
            _cause(
                "cause-b",
                interventions,
                (np.asarray([[1.0]]), np.asarray([[-1.0]])),
                np.eye(1),
            ),
        )
    )
    with pytest.raises(ValueError, match="certificate order"):
        estimate_cause_queries(
            certificate,
            np.asarray([0.0, 0.0]),
            intervention_ids=("action-b", "action-a"),
        )
    with pytest.raises(ValueError, match="cover exactly"):
        plan_diagnostic_interventions(certificate, {"action-a": 0.0})
