from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from bayesian_phystwin.calibration_domain_guard import (
    CalibrationDomainGuardCertificateV1,
    CalibrationDomainGuardConfigV1,
    fit_calibration_domain_guard,
    select_calibration_domain_guarded_belief,
)

PARTITION_ID = "d" * 64
COMMON_DOMAIN_ID = "c" * 64


@dataclass(frozen=True)
class _Belief:
    artifact_id: str


def _cloth_style_guard(
    *,
    frozen_before: bool = True,
    application_used: bool = False,
    independent: bool = True,
    metadata: dict[str, object] | None = None,
) -> CalibrationDomainGuardCertificateV1:
    improvements = np.asarray(
        [0.10, 0.0905, 0.0, 0.03, -0.0663, -0.186],
        dtype=np.float64,
    )
    fallback = np.ones(6, dtype=np.float64)
    return fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-physical-cloth-trial",
        metric="symmetric-l1-chamfer-m",
        group_ids=(
            "dynamic-1",
            "dynamic-2",
            "dynamic-3",
            "static-1",
            "static-2",
            "static-3",
        ),
        domain_ids=(
            "dynamic",
            "dynamic",
            "dynamic",
            "quasi-static",
            "quasi-static",
            "quasi-static",
        ),
        candidate_losses=fallback * (1.0 - improvements),
        fallback_losses=fallback,
        guard_frozen_before_application_outcomes=frozen_before,
        application_outcomes_used_for_guard_selection=application_used,
        calibration_groups_independent=independent,
        metadata={} if metadata is None else metadata,
    )


def test_cloth_style_calibration_authorizes_only_dynamic_continuation() -> None:
    certificate = _cloth_style_guard()

    dynamic = certificate.decision_for_domain("dynamic")
    quasi_static = certificate.decision_for_domain("quasi-static")

    assert certificate.deployment_admissible
    assert certificate.supported_domains == ("dynamic",)
    assert dynamic is not None
    assert dynamic.calibration_supported
    assert dynamic.mean_relative_improvement == pytest.approx(0.0635)
    assert dynamic.win_count == 2
    assert dynamic.required_win_count == 2
    assert dynamic.worst_relative_improvement == pytest.approx(0.0)
    assert quasi_static is not None
    assert not quasi_static.calibration_supported
    assert quasi_static.mean_relative_improvement == pytest.approx(-0.0741)
    assert quasi_static.win_count == 1
    assert quasi_static.worst_relative_improvement == pytest.approx(-0.186)
    assert set(quasi_static.reasons) == {
        "insufficient-calibration-wins",
        "mean-improvement-below-threshold",
        "single-group-regression-exceeds-limit",
    }


def test_calibration_certificate_is_permutation_invariant() -> None:
    first = _cloth_style_guard()
    permutation = np.asarray([5, 2, 0, 4, 1, 3])
    improvements = np.asarray(
        [0.10, 0.0905, 0.0, 0.03, -0.0663, -0.186],
    )
    groups = np.asarray(
        ["dynamic-1", "dynamic-2", "dynamic-3", "static-1", "static-2", "static-3"],
    )
    domains = np.asarray(
        [
            "dynamic",
            "dynamic",
            "dynamic",
            "quasi-static",
            "quasi-static",
            "quasi-static",
        ],
    )
    fallback = np.ones(6)
    second = fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-physical-cloth-trial",
        metric="symmetric-l1-chamfer-m",
        group_ids=tuple(str(value) for value in groups[permutation]),
        domain_ids=tuple(str(value) for value in domains[permutation]),
        candidate_losses=(fallback * (1.0 - improvements))[permutation],
        fallback_losses=fallback[permutation],
        guard_frozen_before_application_outcomes=True,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
    )

    assert second.calibration_data_id == first.calibration_data_id
    assert second.artifact_id == first.artifact_id


def test_supported_domain_selects_candidate_complete_belief() -> None:
    baseline = _Belief("a" * 64)
    candidate = _Belief("b" * 64)

    selected, record = select_calibration_domain_guarded_belief(
        baseline,
        candidate,
        _cloth_style_guard(),
        domain_id="dynamic",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
    )

    assert selected is candidate
    assert record.selected_candidate
    assert record.selected_belief_id == candidate.artifact_id


@pytest.mark.parametrize("domain_id", ["quasi-static", "unseen-regime"])
def test_rejected_or_unknown_domain_returns_exact_baseline_object(
    domain_id: str,
) -> None:
    baseline = _Belief("a" * 64)
    candidate = _Belief("b" * 64)

    selected, record = select_calibration_domain_guarded_belief(
        baseline,
        candidate,
        _cloth_style_guard(),
        domain_id=domain_id,
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
    )

    assert selected is baseline
    assert not record.selected_candidate
    assert record.selected_belief_id == baseline.artifact_id


@pytest.mark.parametrize(
    ("frozen_before", "application_used", "independent"),
    [(False, False, True), (True, True, True), (True, False, False)],
)
def test_nonprospective_information_boundary_forces_exact_fallback(
    frozen_before: bool,
    application_used: bool,
    independent: bool,
) -> None:
    baseline = _Belief("a" * 64)
    candidate = _Belief("b" * 64)
    certificate = _cloth_style_guard(
        frozen_before=frozen_before,
        application_used=application_used,
        independent=independent,
    )

    selected, record = select_calibration_domain_guarded_belief(
        baseline,
        candidate,
        certificate,
        domain_id="dynamic",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
    )

    assert not certificate.deployment_admissible
    assert selected is baseline
    assert not record.selected_candidate


def test_inference_rejection_overrides_supported_domain() -> None:
    baseline = _Belief("a" * 64)
    candidate = _Belief("b" * 64)

    selected, record = select_calibration_domain_guarded_belief(
        baseline,
        candidate,
        _cloth_style_guard(),
        domain_id="dynamic",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=False,
    )

    assert selected is baseline
    assert not record.selected_candidate


def test_under_supported_domain_fails_closed() -> None:
    certificate = fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit="physical-trial",
        metric="loss-m",
        group_ids=("g1", "g2"),
        domain_ids=("dynamic", "dynamic"),
        candidate_losses=np.asarray([0.8, 0.8]),
        fallback_losses=np.asarray([1.0, 1.0]),
        guard_frozen_before_application_outcomes=True,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
    )

    decision = certificate.decision_for_domain("dynamic")
    assert decision is not None
    assert not decision.calibration_supported
    assert "insufficient-calibration-groups" in decision.reasons


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("group_ids", ("g", "g", "h"), "duplicates"),
        ("domain_ids", ("d", "d"), "equal lengths"),
        ("candidate_losses", np.asarray([0.8, np.nan, 0.9]), "finite"),
        ("candidate_losses", np.asarray([0.8, -0.1, 0.9]), "nonnegative"),
        ("fallback_losses", np.asarray([1.0, 0.0, 1.0]), "strictly positive"),
    ],
)
def test_guard_rejects_malformed_calibration_inputs(
    field: str,
    value: object,
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "calibration_partition_id": PARTITION_ID,
        "statistical_unit": "physical-trial",
        "metric": "loss-m",
        "group_ids": ("g1", "g2", "g3"),
        "domain_ids": ("d", "d", "d"),
        "candidate_losses": np.asarray([0.8, 0.9, 0.85]),
        "fallback_losses": np.ones(3),
        "guard_frozen_before_application_outcomes": True,
        "application_outcomes_used_for_guard_selection": False,
        "calibration_groups_independent": True,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        fit_calibration_domain_guard(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        CalibrationDomainGuardConfigV1(minimum_group_count=np.int64(3)),
        CalibrationDomainGuardConfigV1(minimum_win_fraction=np.float64(2.0 / 3.0)),
    ],
)
def test_guard_accepts_numpy_scalar_configuration(config: object) -> None:
    assert isinstance(config, CalibrationDomainGuardConfigV1)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("minimum_group_count", True),
        ("minimum_win_fraction", False),
        ("minimum_mean_relative_improvement", np.nan),
        ("maximum_single_group_relative_regression", -0.1),
    ],
)
def test_guard_config_rejects_lossy_or_invalid_values(
    name: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        CalibrationDomainGuardConfigV1(**{name: value})  # type: ignore[arg-type]


def test_certificate_rejects_forged_domain_decision() -> None:
    certificate = _cloth_style_guard()
    decisions = list(certificate.decisions)
    decisions[0] = replace(
        decisions[0],
        calibration_supported=False,
        reasons=("forged",),
        artifact_id=None,
    )

    with pytest.raises(ValueError, match="does not match"):
        CalibrationDomainGuardCertificateV1(
            calibration_partition_id=certificate.calibration_partition_id,
            statistical_unit=certificate.statistical_unit,
            metric=certificate.metric,
            config=certificate.config,
            calibration_data_id=certificate.calibration_data_id,
            decisions=decisions,
            guard_frozen_before_application_outcomes=True,
            application_outcomes_used_for_guard_selection=False,
            calibration_groups_independent=True,
        )


def test_certificate_metadata_is_defensively_copied_and_frozen() -> None:
    metadata = {"source": {"groups": ["calibration-only"]}}
    certificate = _cloth_style_guard(metadata=metadata)
    source = metadata["source"]
    assert isinstance(source, dict)
    groups = source["groups"]
    assert isinstance(groups, list)
    groups.append("mutated")

    assert list(certificate.metadata["source"]["groups"]) == ["calibration-only"]
    with pytest.raises(TypeError, match="immutable"):
        certificate.metadata["new"] = True  # type: ignore[index]


def test_supplied_certificate_identity_must_match_content() -> None:
    certificate = _cloth_style_guard()

    with pytest.raises(ValueError, match="artifact_id does not match"):
        replace(certificate, artifact_id="0" * 64)


def test_threshold_boundaries_are_inclusive() -> None:
    certificate = fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit="physical-trial",
        metric="loss-m",
        group_ids=("g1", "g2", "g3"),
        domain_ids=("boundary", "boundary", "boundary"),
        candidate_losses=np.asarray([0.9, 0.9, 1.05]),
        fallback_losses=np.ones(3),
        guard_frozen_before_application_outcomes=True,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
    )

    decision = certificate.decision_for_domain("boundary")
    assert decision is not None
    assert decision.calibration_supported
    assert decision.mean_relative_improvement == pytest.approx(0.05)
    assert decision.worst_relative_improvement == pytest.approx(-0.05)


def test_decision_identity_round_trip_and_record() -> None:
    decision = _cloth_style_guard().decision_for_domain("dynamic")
    assert decision is not None

    reconstructed = replace(decision, artifact_id=decision.artifact_id)

    assert reconstructed.artifact_id == decision.artifact_id
    assert reconstructed.to_record()["artifact_id"] == decision.artifact_id


def test_certificate_identity_round_trip_and_record() -> None:
    certificate = _cloth_style_guard()

    reconstructed = replace(certificate, artifact_id=certificate.artifact_id)

    assert reconstructed.artifact_id == certificate.artifact_id
    assert reconstructed.to_record()["artifact_id"] == certificate.artifact_id


def test_selection_copies_caller_metadata() -> None:
    baseline = _Belief("a" * 64)
    candidate = _Belief("b" * 64)
    metadata = {"request": {"source": "frozen-prefix"}}

    _, record = select_calibration_domain_guarded_belief(
        baseline,
        candidate,
        _cloth_style_guard(),
        domain_id="dynamic",
        common_domain_id=COMMON_DOMAIN_ID,
        inference_admissible=True,
        metadata=metadata,
    )
    request = metadata["request"]
    assert isinstance(request, dict)
    request["source"] = "mutated"

    assert record.metadata["caller"]["request"]["source"] == "frozen-prefix"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("statistical_unit", " bad", "canonical string"),
        ("metric", "", "canonical string"),
        ("group_ids", "g1", "sequence of strings"),
        ("group_ids", 1, "sequence of strings"),
        ("group_ids", (), "must not be empty"),
        ("domain_ids", ("domain", " bad", "domain"), "canonical string"),
        ("candidate_losses", np.asarray(0.8), "one-dimensional"),
        ("candidate_losses", np.empty((0,)), "one-dimensional"),
        ("candidate_losses", np.asarray([[0.8, 0.9, 0.7]]), "one-dimensional"),
        ("candidate_losses", np.asarray(["0.8", "0.9", "0.7"]), "numeric"),
    ],
)
def test_guard_rejects_noncanonical_shapes_and_identifiers(
    field: str,
    value: object,
    match: str,
) -> None:
    arguments: dict[str, object] = {
        "calibration_partition_id": PARTITION_ID,
        "statistical_unit": "physical-trial",
        "metric": "loss-m",
        "group_ids": ("g1", "g2", "g3"),
        "domain_ids": ("domain", "domain", "domain"),
        "candidate_losses": np.asarray([0.8, 0.9, 0.7]),
        "fallback_losses": np.ones(3),
        "guard_frozen_before_application_outcomes": True,
        "application_outcomes_used_for_guard_selection": False,
        "calibration_groups_independent": True,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        fit_calibration_domain_guard(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_mean_relative_improvement", 1.1),
        ("minimum_win_fraction", 0.0),
        ("minimum_win_fraction", 1.1),
        ("maximum_single_group_relative_regression", 1.1),
        ("numerical_tolerance", -1e-6),
        ("numerical_tolerance", 1.0),
        ("minimum_mean_relative_improvement", [0.05]),
    ],
)
def test_guard_config_rejects_out_of_range_or_nonscalar_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        CalibrationDomainGuardConfigV1(**{field: value})  # type: ignore[arg-type]


def test_fit_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="CalibrationDomainGuardConfigV1"):
        fit_calibration_domain_guard(
            calibration_partition_id=PARTITION_ID,
            statistical_unit="physical-trial",
            metric="loss-m",
            group_ids=("g1", "g2", "g3"),
            domain_ids=("domain", "domain", "domain"),
            candidate_losses=np.asarray([0.8, 0.9, 0.7]),
            fallback_losses=np.ones(3),
            guard_frozen_before_application_outcomes=True,
            application_outcomes_used_for_guard_selection=False,
            calibration_groups_independent=True,
            config=object(),  # type: ignore[arg-type]
        )


def test_selection_rejects_wrong_certificate_type() -> None:
    with pytest.raises(TypeError, match="CalibrationDomainGuardCertificateV1"):
        select_calibration_domain_guarded_belief(
            _Belief("a" * 64),
            _Belief("b" * 64),
            object(),  # type: ignore[arg-type]
            domain_id="dynamic",
            common_domain_id=COMMON_DOMAIN_ID,
            inference_admissible=True,
        )


def test_decision_rejects_malformed_internal_fields() -> None:
    decision = _cloth_style_guard().decision_for_domain("dynamic")
    assert decision is not None

    invalid_updates = (
        {"domain_id": " bad", "artifact_id": None},
        {
            "group_ids": ("g", "g"),
            "relative_improvements": (0.1, 0.1),
            "artifact_id": None,
        },
        {"group_ids": ("g",), "relative_improvements": (), "artifact_id": None},
        {"group_ids": ("g",), "relative_improvements": (1.1,), "artifact_id": None},
        {"group_ids": ("g", "h"), "relative_improvements": (0.1,), "artifact_id": None},
        {"win_count": 4, "artifact_id": None},
        {"reasons": ("duplicate", "duplicate"), "artifact_id": None},
        {"artifact_id": "0" * 64},
    )
    for updates in invalid_updates:
        with pytest.raises(ValueError):
            replace(decision, **updates)


def test_domain_evaluator_rejects_mismatched_lengths() -> None:
    import bayesian_phystwin.calibration_domain_guard as guard_module

    with pytest.raises(ValueError, match="length must match"):
        guard_module._evaluate_domain(
            "domain",
            ("g1", "g2"),
            (0.1,),
            CalibrationDomainGuardConfigV1(),
        )


def test_certificate_rejects_malformed_decision_roster_and_config() -> None:
    certificate = _cloth_style_guard()
    first = certificate.decisions[0]

    invalid_updates: tuple[dict[str, object], ...] = (
        {"config": object(), "artifact_id": None},
        {"decisions": (), "artifact_id": None},
        {"decisions": (object(),), "artifact_id": None},
        {"decisions": (first, first), "artifact_id": None},
    )
    for updates in invalid_updates:
        with pytest.raises((TypeError, ValueError)):
            replace(certificate, **updates)
