from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin_experiments.explain_transport_probe_abstain_v1 import (
    DiagnosticDisposition,
    ExplainTransportProbeAbstainV1,
)
from bayesian_phystwin_experiments.interventional_cause_adequacy_v1 import (
    CauseFamilyAdequacyStatus,
    InterventionalCauseFamilyAdequacyV1,
)

SHA = "a" * 64


def _report(
    *,
    cause_signatures: dict[str, np.ndarray],
    residual: np.ndarray,
    target_maps: dict[str, np.ndarray],
    candidate_designs: dict[str, np.ndarray] | None = None,
    intervention_costs: dict[str, float] | None = None,
    noise_radius: float = 0.05,
    artifact_id: str | None = None,
) -> ExplainTransportProbeAbstainV1:
    if candidate_designs is None:
        candidate_designs = {}
    if intervention_costs is None:
        intervention_costs = {candidate: 1.0 for candidate in candidate_designs}
    adequacy = InterventionalCauseFamilyAdequacyV1(
        residual_id=SHA,
        intervention_roster_id=SHA,
        whitening_id=SHA,
        cause_signature_ids={cause: SHA for cause in cause_signatures},
        cause_signatures=cause_signatures,
        whitened_residual=residual,
        noise_radius=noise_radius,
    )
    return ExplainTransportProbeAbstainV1(
        adequacy_certificate=adequacy,
        target_intervention_roster_id=SHA,
        target_transport_ids={target: SHA for target in target_maps},
        target_maps=target_maps,
        candidate_roster_id=SHA,
        candidate_intervention_ids={candidate: SHA for candidate in candidate_designs},
        candidate_designs=candidate_designs,
        intervention_costs=intervention_costs,
        artifact_id=artifact_id,
    )


def _ambiguous_report(
    *,
    target_maps: dict[str, np.ndarray],
    include_informative_probes: bool = True,
) -> ExplainTransportProbeAbstainV1:
    # Sorted coefficient order is gauge, material, state.
    cause_signatures = {
        "gauge": np.asarray([[1.0]]),
        "material": np.asarray([[0.0]]),
        "state": np.asarray([[1.0]]),
    }
    if include_informative_probes:
        candidates = {
            "material-probe": np.asarray([[0.0, 1.0, 0.0]]),
            "redundant-probe": np.asarray([[2.0, 0.0, 2.0]]),
            "state-gauge-probe": np.asarray([[1.0, 0.0, -1.0]]),
        }
        costs = {
            "material-probe": 1.0,
            "redundant-probe": 0.1,
            "state-gauge-probe": 1.0,
        }
    else:
        candidates = {
            "redundant-probe": np.asarray([[2.0, 0.0, 2.0]]),
        }
        costs = {"redundant-probe": 0.1}
    return _report(
        cause_signatures=cause_signatures,
        residual=np.asarray([2.0]),
        target_maps=target_maps,
        candidate_designs=candidates,
        intervention_costs=costs,
    )


def test_unmodeled_cause_returns_none_of_the_above_without_probe() -> None:
    report = _report(
        cause_signatures={
            "gauge": np.asarray([[1.0], [0.0], [0.0]]),
            "state": np.asarray([[0.0], [1.0], [0.0]]),
        },
        residual=np.asarray([0.0, 0.0, 2.0]),
        target_maps={"state-target": np.asarray([[0.0, 1.0]])},
        candidate_designs={
            "state-gauge-probe": np.asarray([[1.0, -1.0]]),
        },
        noise_radius=0.1,
    )

    decision = report.decision_for("state-target")
    assert decision.adequacy_status is CauseFamilyAdequacyStatus.UNMODELED_CAUSE
    assert decision.disposition is DiagnosticDisposition.NONE_OF_THE_ABOVE
    assert decision.none_of_the_above is True
    assert decision.selected_interventions == ()
    assert decision.transport_permitted is False
    fallback = object()
    assert (
        report.deploy_or_exact_fallback("state-target", fallback=fallback) is fallback
    )


def test_no_detectable_error_does_not_invent_a_correction() -> None:
    report = _report(
        cause_signatures={
            "gauge": np.asarray([[1.0]]),
            "state": np.asarray([[1.0]]),
        },
        residual=np.asarray([0.01]),
        target_maps={"sum": np.asarray([[1.0, 1.0]])},
        noise_radius=0.05,
    )

    decision = report.decision_for("sum")
    assert decision.disposition is DiagnosticDisposition.NO_DETECTABLE_ERROR
    assert decision.transport_permitted is False
    fallback = object()
    assert report.deploy_or_exact_fallback("sum", fallback=fallback) is fallback


def test_unique_registered_explanation_can_explain_and_transport() -> None:
    # Sorted coefficient order is gauge, state.
    report = _report(
        cause_signatures={
            "gauge": np.asarray([[0.0], [1.0]]),
            "state": np.asarray([[1.0], [0.0]]),
        },
        residual=np.asarray([2.0, -1.0]),
        target_maps={"state": np.asarray([[0.0, 1.0]])},
    )

    decision = report.decision_for("state")
    assert decision.disposition is DiagnosticDisposition.EXPLAIN_AND_TRANSPORT
    assert decision.registered_explanation_unique is True
    assert decision.transport_permitted is True
    deployed = report.deploy_or_exact_fallback("state", fallback=object())
    np.testing.assert_allclose(deployed, np.asarray([2.0]))


def test_target_can_transport_without_unique_cause_identification() -> None:
    report = _ambiguous_report(
        target_maps={"sum": np.asarray([[1.0, 0.0, 1.0]])},
    )

    decision = report.decision_for("sum")
    assert decision.disposition is DiagnosticDisposition.TRANSPORT_WITHOUT_CAUSE
    assert decision.registered_explanation_unique is False
    assert decision.coefficient_ambiguity_dimension == 2
    assert decision.transport_permitted is True
    assert decision.selected_interventions == ()
    assert decision.selected_intervention_cost == 0.0
    assert decision.minimum_full_cause_identification_cost == pytest.approx(2.0)
    assert decision.target_cost_saving_vs_full_cause_identification == pytest.approx(
        2.0
    )
    np.testing.assert_allclose(
        report.deploy_or_exact_fallback("sum", fallback=object()),
        np.asarray([2.0]),
    )


def test_ambiguous_target_selects_only_the_target_identifying_probe() -> None:
    report = _ambiguous_report(
        target_maps={"difference": np.asarray([[1.0, 0.0, -1.0]])},
    )

    decision = report.decision_for("difference")
    assert decision.disposition is DiagnosticDisposition.PROBE_THEN_REASSESS
    assert decision.transport_permitted is False
    assert decision.fallback_required_now is True
    assert decision.selected_interventions == ("state-gauge-probe",)
    assert decision.selected_intervention_cost == pytest.approx(1.0)
    assert decision.minimum_full_cause_identification_cost == pytest.approx(2.0)
    assert decision.target_cost_saving_vs_full_cause_identification == pytest.approx(
        1.0
    )
    fallback = object()
    assert report.deploy_or_exact_fallback("difference", fallback=fallback) is fallback


def test_unresolvable_target_abstains_instead_of_forcing_a_cause() -> None:
    report = _ambiguous_report(
        target_maps={"difference": np.asarray([[1.0, 0.0, -1.0]])},
        include_informative_probes=False,
    )

    decision = report.decision_for("difference")
    assert decision.disposition is DiagnosticDisposition.ABSTAIN
    assert decision.transport_permitted is False
    assert decision.selected_interventions == ()


def test_partial_target_is_reported_but_not_deployed_as_a_full_target() -> None:
    report = _ambiguous_report(
        target_maps={
            "sum-and-difference": np.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
        },
        include_informative_probes=False,
    )

    decision = report.decision_for("sum-and-difference")
    assert decision.disposition is DiagnosticDisposition.PARTIAL_ONLY_FALLBACK
    assert decision.partial_target_available is True
    assert decision.target_identifiable_dimension == 1
    assert decision.target_ambiguity_dimension == 1
    assert decision.transport_permitted is False
    fallback = object()
    assert (
        report.deploy_or_exact_fallback(
            "sum-and-difference",
            fallback=fallback,
        )
        is fallback
    )


def test_report_is_content_addressed_and_rejects_wrong_artifact_id() -> None:
    report = _ambiguous_report(
        target_maps={"sum": np.asarray([[1.0, 0.0, 1.0]])},
    )
    assert len(report.artifact_id) == 64
    assert report.to_record()["artifact_id"] == report.artifact_id

    with pytest.raises(ValueError, match="artifact_id does not match"):
        _report(
            cause_signatures={
                "gauge": np.asarray([[1.0]]),
                "state": np.asarray([[1.0]]),
            },
            residual=np.asarray([2.0]),
            target_maps={"sum": np.asarray([[1.0, 1.0]])},
            artifact_id="0" * 64,
        )


def test_target_roster_can_contain_all_operational_dispositions() -> None:
    report = _ambiguous_report(
        target_maps={
            "difference": np.asarray([[1.0, 0.0, -1.0]]),
            "material": np.asarray([[0.0, 1.0, 0.0]]),
            "sum": np.asarray([[1.0, 0.0, 1.0]]),
        },
    )
    assert report.decision_for("sum").disposition is (
        DiagnosticDisposition.TRANSPORT_WITHOUT_CAUSE
    )
    assert report.decision_for("difference").selected_interventions == (
        "state-gauge-probe",
    )
    assert report.decision_for("material").selected_interventions == ("material-probe",)
    assert report.to_record()["disposition_counts"] == {
        "abstain": 0,
        "explain_and_transport": 0,
        "no_detectable_error": 0,
        "none_of_the_above": 0,
        "partial_only_fallback": 0,
        "probe_then_reassess": 2,
        "transport_without_cause": 1,
    }
