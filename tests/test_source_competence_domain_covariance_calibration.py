from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bayesian_phystwin.calibration_domain_guard import (
    fit_calibration_domain_guard,
)
from bayesian_phystwin.domain_covariance_calibration import (
    DomainCovarianceCalibrationApplicationV1,
    DomainCovarianceCalibrationCertificateV1,
    DomainCovarianceCalibrationConfigV1,
    apply_domain_covariance_calibration,
    fit_domain_covariance_calibration,
)

PARTITION_ID = "a" * 64
PREDICTOR_ID = "b" * 64


def _guard(
    *,
    frozen_before: bool = True,
    application_used: bool = False,
    independent: bool = True,
):
    groups = tuple(f"dynamic-{index}" for index in range(4)) + tuple(
        f"static-{index}" for index in range(4)
    )
    domains = ("dynamic",) * 4 + ("quasi-static",) * 4
    fallback = np.ones(8, dtype=np.float64)
    candidate = np.asarray([0.9] * 4 + [1.1] * 4, dtype=np.float64)
    return fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-physical-session",
        metric="point-loss-m",
        group_ids=groups,
        domain_ids=domains,
        candidate_losses=candidate,
        fallback_losses=fallback,
        guard_frozen_before_application_outcomes=frozen_before,
        application_outcomes_used_for_guard_selection=application_used,
        calibration_groups_independent=independent,
    )


def _calibration_arrays(
    *,
    dynamic_residuals: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0),
    events_per_group: int = 1,
):
    event_ids: list[str] = []
    group_ids: list[str] = []
    domain_ids: list[str] = []
    residuals: list[list[float]] = []
    covariances: list[list[list[float]]] = []
    for domain, prefix, values in (
        ("dynamic", "dynamic", dynamic_residuals),
        ("quasi-static", "static", (1.0, 1.0, 1.0, 1.0)),
    ):
        for group_index, value in enumerate(values):
            for event_index in range(events_per_group):
                event_ids.append(f"{prefix}-{group_index}-event-{event_index}")
                group_ids.append(f"{prefix}-{group_index}")
                domain_ids.append(domain)
                residuals.append([value])
                covariances.append([[1.0]])
    return (
        tuple(event_ids),
        tuple(group_ids),
        tuple(domain_ids),
        np.asarray(residuals, dtype=np.float64),
        np.asarray(covariances, dtype=np.float64),
    )


def _config(**overrides: object) -> DomainCovarianceCalibrationConfigV1:
    values: dict[str, object] = {
        "covariance_scales": (1.0, 4.0),
        "isotropic_variances": (0.0, 3.0),
        "minimum_group_count": 4,
        "minimum_mean_loo_nll_improvement": 0.0,
        "maximum_single_group_loo_nll_regression": 0.0,
    }
    values.update(overrides)
    return DomainCovarianceCalibrationConfigV1(**values)  # type: ignore[arg-type]


def _certificate(
    *,
    dynamic_residuals: tuple[float, ...] = (2.0, 2.0, 2.0, 2.0),
    events_per_group: int = 1,
    predictor_frozen: bool = True,
    grid_frozen: bool = True,
    application_used: bool = False,
    independent: bool = True,
    guard=None,
    config: DomainCovarianceCalibrationConfigV1 | None = None,
    metadata: dict[str, object] | None = None,
) -> DomainCovarianceCalibrationCertificateV1:
    event_ids, group_ids, domain_ids, residuals, covariances = _calibration_arrays(
        dynamic_residuals=dynamic_residuals,
        events_per_group=events_per_group,
    )
    return fit_domain_covariance_calibration(
        predictor_id=PREDICTOR_ID,
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-physical-session",
        residual_semantics="prediction-error-m",
        covariance_semantics="raw-predictive-covariance-m2",
        event_ids=event_ids,
        group_ids=group_ids,
        domain_ids=domain_ids,
        residuals=residuals,
        covariances=covariances,
        domain_guard=_guard() if guard is None else guard,
        predictor_frozen_before_calibration_outcomes=predictor_frozen,
        transform_grid_frozen_before_calibration_outcomes=grid_frozen,
        application_outcomes_used_for_calibration_selection=application_used,
        calibration_groups_independent=independent,
        config=_config() if config is None else config,
        metadata={} if metadata is None else metadata,
    )


def test_scale_plus_floor_selection_uses_conservative_tie_break() -> None:
    certificate = _certificate()
    dynamic = certificate.decision_for_domain("dynamic")
    quasi_static = certificate.decision_for_domain("quasi-static")

    assert certificate.deployment_admissible
    assert certificate.supported_domains == ("dynamic",)
    assert dynamic is not None
    assert dynamic.calibration_supported
    assert dynamic.selected_covariance_scale == 4.0
    assert dynamic.selected_isotropic_variance == 0.0
    assert dynamic.calibrated_equal_group_mean_nll < dynamic.raw_equal_group_mean_nll
    assert dynamic.mean_loo_nll_improvement > 0.0
    assert dynamic.worst_loo_nll_regression == pytest.approx(0.0)
    assert quasi_static is not None
    assert not quasi_static.calibration_supported
    assert "calibration-domain-guard-rejected" in quasi_static.reasons


def test_supported_domain_applies_read_only_transform_without_mutating_input() -> None:
    raw = np.asarray([[1.0, 0.2], [0.2, 2.0]])
    snapshot = raw.copy()

    output, record = apply_domain_covariance_calibration(
        raw,
        _certificate(),
        domain_id="dynamic",
        inference_admissible=True,
    )

    np.testing.assert_array_equal(raw, snapshot)
    np.testing.assert_allclose(output, 4.0 * raw)
    assert output is not raw
    assert not output.flags.writeable
    assert record.applied
    assert not record.exact_fallback
    assert record.reason == "calibration-domain-authorized"
    assert record.raw_covariance_sha256 != record.output_covariance_sha256


def test_supported_domain_applies_transform_to_covariance_batch() -> None:
    raw = np.stack([np.eye(2), np.eye(2) * 2.0])

    output, record = apply_domain_covariance_calibration(
        raw,
        _certificate(),
        domain_id="dynamic",
        inference_admissible=True,
    )

    np.testing.assert_allclose(output, 4.0 * raw)
    assert output.shape == raw.shape
    assert record.covariance_scale == 4.0
    assert record.isotropic_variance == 0.0


@pytest.mark.parametrize(
    ("domain_id", "inference_admissible", "reason"),
    [
        ("quasi-static", True, "calibration-domain-rejected"),
        ("unseen", True, "unknown-calibration-domain"),
        ("dynamic", False, "inference-rejected"),
    ],
)
def test_rejection_returns_exact_raw_covariance_object(
    domain_id: str,
    inference_admissible: bool,
    reason: str,
) -> None:
    raw = np.asarray([[1.0]])

    output, record = apply_domain_covariance_calibration(
        raw,
        _certificate(),
        domain_id=domain_id,
        inference_admissible=inference_admissible,
    )

    assert output is raw
    assert record.exact_fallback
    assert not record.applied
    assert record.reason == reason
    assert record.covariance_scale == 1.0
    assert record.isotropic_variance == 0.0
    assert record.raw_covariance_sha256 == record.output_covariance_sha256


@pytest.mark.parametrize(
    ("predictor_frozen", "grid_frozen", "application_used", "independent"),
    [
        (False, True, False, True),
        (True, False, False, True),
        (True, True, True, True),
        (True, True, False, False),
    ],
)
def test_nonprospective_information_boundary_forces_exact_fallback(
    predictor_frozen: bool,
    grid_frozen: bool,
    application_used: bool,
    independent: bool,
) -> None:
    certificate = _certificate(
        predictor_frozen=predictor_frozen,
        grid_frozen=grid_frozen,
        application_used=application_used,
        independent=independent,
    )
    raw = np.asarray([[1.0]])

    output, record = apply_domain_covariance_calibration(
        raw,
        certificate,
        domain_id="dynamic",
        inference_admissible=True,
    )

    assert not certificate.deployment_admissible
    assert certificate.supported_domains == ()
    assert output is raw
    assert record.reason == "calibration-information-boundary-rejected"


def test_nonprospective_domain_guard_forces_exact_fallback() -> None:
    certificate = _certificate(guard=_guard(frozen_before=False))
    raw = np.asarray([[1.0]])

    output, record = apply_domain_covariance_calibration(
        raw,
        certificate,
        domain_id="dynamic",
        inference_admissible=True,
    )

    assert not certificate.guard_deployment_admissible
    assert not certificate.deployment_admissible
    assert output is raw
    assert record.reason == "calibration-information-boundary-rejected"


def test_leave_one_group_out_regression_rejects_outlier_domain() -> None:
    certificate = _certificate(dynamic_residuals=(2.0, 2.0, 2.0, 0.1))
    decision = certificate.decision_for_domain("dynamic")

    assert decision is not None
    assert decision.selected_covariance_scale == 4.0
    assert decision.worst_loo_nll_regression > 0.0
    assert not decision.calibration_supported
    assert (
        "single-group-loo-nll-regression-exceeds-limit" in decision.reasons
    )


def test_calibration_is_permutation_invariant() -> None:
    first = _certificate()
    event_ids, group_ids, domain_ids, residuals, covariances = _calibration_arrays()
    permutation = np.asarray([7, 0, 5, 2, 4, 1, 6, 3])
    second = fit_domain_covariance_calibration(
        predictor_id=PREDICTOR_ID,
        calibration_partition_id=PARTITION_ID,
        statistical_unit="independent-physical-session",
        residual_semantics="prediction-error-m",
        covariance_semantics="raw-predictive-covariance-m2",
        event_ids=tuple(str(value) for value in np.asarray(event_ids)[permutation]),
        group_ids=tuple(str(value) for value in np.asarray(group_ids)[permutation]),
        domain_ids=tuple(str(value) for value in np.asarray(domain_ids)[permutation]),
        residuals=residuals[permutation],
        covariances=covariances[permutation],
        domain_guard=_guard(),
        predictor_frozen_before_calibration_outcomes=True,
        transform_grid_frozen_before_calibration_outcomes=True,
        application_outcomes_used_for_calibration_selection=False,
        calibration_groups_independent=True,
        config=_config(),
    )

    assert second.calibration_data_id == first.calibration_data_id
    assert second.artifact_id == first.artifact_id


def test_equal_group_scoring_is_not_changed_by_within_group_replication() -> None:
    single = _certificate(events_per_group=1)
    repeated = _certificate(events_per_group=5)
    single_decision = single.decision_for_domain("dynamic")
    repeated_decision = repeated.decision_for_domain("dynamic")

    assert single_decision is not None
    assert repeated_decision is not None
    assert repeated_decision.artifact_id == single_decision.artifact_id
    assert repeated.calibration_data_id != single.calibration_data_id


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("event_ids", ("event",) * 8, "duplicates"),
        ("group_ids", ("group",) * 7, "equal event counts"),
        ("residuals", np.zeros((8, 2, 1)), "two-dimensional"),
        ("residuals", np.asarray([[np.nan]] * 8), "finite"),
        ("covariances", np.ones((8, 2, 3)), "square"),
        ("covariances", np.asarray([[[np.nan]]] * 8), "finite"),
    ],
)
def test_fit_rejects_malformed_event_inputs(
    field: str,
    value: object,
    match: str,
) -> None:
    event_ids, group_ids, domain_ids, residuals, covariances = _calibration_arrays()
    arguments: dict[str, object] = {
        "predictor_id": PREDICTOR_ID,
        "calibration_partition_id": PARTITION_ID,
        "statistical_unit": "independent-physical-session",
        "residual_semantics": "prediction-error-m",
        "covariance_semantics": "raw-predictive-covariance-m2",
        "event_ids": event_ids,
        "group_ids": group_ids,
        "domain_ids": domain_ids,
        "residuals": residuals,
        "covariances": covariances,
        "domain_guard": _guard(),
        "predictor_frozen_before_calibration_outcomes": True,
        "transform_grid_frozen_before_calibration_outcomes": True,
        "application_outcomes_used_for_calibration_selection": False,
        "calibration_groups_independent": True,
        "config": _config(),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "covariances",
    [
        np.asarray([[[1.0, 0.3], [0.1, 1.0]]] * 8),
        np.asarray([[[1.0, 2.0], [2.0, 1.0]]] * 8),
    ],
)
def test_fit_rejects_nonsymmetric_or_indefinite_covariance(
    covariances: np.ndarray,
) -> None:
    event_ids, group_ids, domain_ids, _, _ = _calibration_arrays()

    with pytest.raises(ValueError, match="symmetric|positive semidefinite"):
        fit_domain_covariance_calibration(
            predictor_id=PREDICTOR_ID,
            calibration_partition_id=PARTITION_ID,
            statistical_unit="independent-physical-session",
            residual_semantics="prediction-error-m",
            covariance_semantics="raw-predictive-covariance-m2",
            event_ids=event_ids,
            group_ids=group_ids,
            domain_ids=domain_ids,
            residuals=np.zeros((8, 2)),
            covariances=covariances,
            domain_guard=_guard(),
            predictor_frozen_before_calibration_outcomes=True,
            transform_grid_frozen_before_calibration_outcomes=True,
            application_outcomes_used_for_calibration_selection=False,
            calibration_groups_independent=True,
            config=_config(),
        )


def test_group_cannot_cross_calibration_domains() -> None:
    event_ids, group_ids, domain_ids, residuals, covariances = _calibration_arrays()
    mutable_domains = list(domain_ids)
    mutable_groups = list(group_ids)
    mutable_groups[4] = mutable_groups[0]

    with pytest.raises(ValueError, match="one domain"):
        fit_domain_covariance_calibration(
            predictor_id=PREDICTOR_ID,
            calibration_partition_id=PARTITION_ID,
            statistical_unit="independent-physical-session",
            residual_semantics="prediction-error-m",
            covariance_semantics="raw-predictive-covariance-m2",
            event_ids=event_ids,
            group_ids=mutable_groups,
            domain_ids=mutable_domains,
            residuals=residuals,
            covariances=covariances,
            domain_guard=_guard(),
            predictor_frozen_before_calibration_outcomes=True,
            transform_grid_frozen_before_calibration_outcomes=True,
            application_outcomes_used_for_calibration_selection=False,
            calibration_groups_independent=True,
            config=_config(),
        )


def test_group_roster_must_match_domain_guard() -> None:
    event_ids, group_ids, domain_ids, residuals, covariances = _calibration_arrays()
    mutable_groups = list(group_ids)
    mutable_groups[0] = "dynamic-replacement"

    with pytest.raises(ValueError, match="differs from domain guard"):
        fit_domain_covariance_calibration(
            predictor_id=PREDICTOR_ID,
            calibration_partition_id=PARTITION_ID,
            statistical_unit="independent-physical-session",
            residual_semantics="prediction-error-m",
            covariance_semantics="raw-predictive-covariance-m2",
            event_ids=event_ids,
            group_ids=mutable_groups,
            domain_ids=domain_ids,
            residuals=residuals,
            covariances=covariances,
            domain_guard=_guard(),
            predictor_frozen_before_calibration_outcomes=True,
            transform_grid_frozen_before_calibration_outcomes=True,
            application_outcomes_used_for_calibration_selection=False,
            calibration_groups_independent=True,
            config=_config(),
        )


@pytest.mark.parametrize(
    "config",
    [
        {"covariance_scales": (0.5, 2.0)},
        {"isotropic_variances": (1e-6,)},
        {"covariance_scales": (1.0, 1.0)},
        {"isotropic_variances": (0.0, -1.0)},
        {"minimum_group_count": True},
        {"minimum_mean_loo_nll_improvement": -0.1},
        {"scoring_eigenvalue_floor": 0.0},
    ],
)
def test_config_rejects_ambiguous_or_unsafe_values(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _config(**config)


def test_partition_and_statistical_unit_must_match_domain_guard() -> None:
    event_ids, group_ids, domain_ids, residuals, covariances = _calibration_arrays()

    with pytest.raises(ValueError, match="calibration_partition_id differs"):
        fit_domain_covariance_calibration(
            predictor_id=PREDICTOR_ID,
            calibration_partition_id="f" * 64,
            statistical_unit="independent-physical-session",
            residual_semantics="prediction-error-m",
            covariance_semantics="raw-predictive-covariance-m2",
            event_ids=event_ids,
            group_ids=group_ids,
            domain_ids=domain_ids,
            residuals=residuals,
            covariances=covariances,
            domain_guard=_guard(),
            predictor_frozen_before_calibration_outcomes=True,
            transform_grid_frozen_before_calibration_outcomes=True,
            application_outcomes_used_for_calibration_selection=False,
            calibration_groups_independent=True,
            config=_config(),
        )


def test_certificate_and_application_metadata_are_defensively_frozen() -> None:
    metadata = {"source": {"groups": ["calibration-only"]}}
    certificate = _certificate(metadata=metadata)
    source = metadata["source"]
    assert isinstance(source, dict)
    groups = source["groups"]
    assert isinstance(groups, list)
    groups.append("mutated")
    assert list(certificate.metadata["source"]["groups"]) == ["calibration-only"]

    raw = np.asarray([[1.0]])
    application_metadata = {"request": {"source": "frozen-prefix"}}
    _, record = apply_domain_covariance_calibration(
        raw,
        certificate,
        domain_id="dynamic",
        inference_admissible=True,
        metadata=application_metadata,
    )
    request = application_metadata["request"]
    assert isinstance(request, dict)
    request["source"] = "mutated"
    assert record.metadata["request"]["source"] == "frozen-prefix"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: replace(_certificate(), artifact_id="0" * 64),
        lambda: replace(
            _certificate().decision_for_domain("dynamic"),
            artifact_id="0" * 64,
        ),
    ],
)
def test_forged_content_id_is_rejected(factory) -> None:
    with pytest.raises(ValueError, match="artifact_id does not match"):
        factory()


def test_application_record_round_trip_identity() -> None:
    raw = np.asarray([[1.0]])
    _, record = apply_domain_covariance_calibration(
        raw,
        _certificate(),
        domain_id="dynamic",
        inference_admissible=True,
    )

    reconstructed = replace(record, artifact_id=record.artifact_id)

    assert isinstance(reconstructed, DomainCovarianceCalibrationApplicationV1)
    assert reconstructed.artifact_id == record.artifact_id
    assert reconstructed.to_record()["artifact_id"] == record.artifact_id


def test_application_requires_numpy_array_and_valid_covariance() -> None:
    certificate = _certificate()

    with pytest.raises(TypeError, match="numpy.ndarray"):
        apply_domain_covariance_calibration(
            [[1.0]],  # type: ignore[arg-type]
            certificate,
            domain_id="dynamic",
            inference_admissible=True,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        apply_domain_covariance_calibration(
            np.asarray([[1.0, 2.0], [2.0, 1.0]]),
            certificate,
            domain_id="dynamic",
            inference_admissible=True,
        )
