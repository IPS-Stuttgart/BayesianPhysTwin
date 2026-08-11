from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bayesian_phystwin.domain_covariance_calibration as calibration
from bayesian_phystwin.calibration_domain_guard import (
    fit_calibration_domain_guard,
)
from bayesian_phystwin.domain_covariance_calibration import (
    DomainCovarianceCalibrationConfigV1,
    apply_domain_covariance_calibration,
    fit_domain_covariance_calibration,
)

PARTITION_ID = "a" * 64
PREDICTOR_ID = "b" * 64
STATISTICAL_UNIT = "independent-physical-session"


def _guard(
    group_ids: tuple[str, ...],
    domain_ids: tuple[str, ...],
):
    return fit_calibration_domain_guard(
        calibration_partition_id=PARTITION_ID,
        statistical_unit=STATISTICAL_UNIT,
        metric="point-loss-m",
        group_ids=group_ids,
        domain_ids=domain_ids,
        candidate_losses=np.full(len(group_ids), 0.9),
        fallback_losses=np.ones(len(group_ids)),
        guard_frozen_before_application_outcomes=True,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
    )


def _fit(
    *,
    group_ids: tuple[str, ...] = ("g0", "g1", "g2", "g3"),
    domain_ids: tuple[str, ...] = ("dynamic",) * 4,
    residuals: np.ndarray | None = None,
    covariances: np.ndarray | None = None,
    domain_guard: object | None = None,
    config: object | None = None,
):
    event_ids = tuple(f"event-{index}" for index in range(len(group_ids)))
    residual_values = (
        np.full((len(group_ids), 1), 2.0) if residuals is None else residuals
    )
    covariance_values = (
        np.ones((len(group_ids), 1, 1)) if covariances is None else covariances
    )
    guard = _guard(group_ids, domain_ids) if domain_guard is None else domain_guard
    settings = (
        DomainCovarianceCalibrationConfigV1(
            covariance_scales=(1.0, 4.0),
            isotropic_variances=(0.0,),
            minimum_group_count=2,
        )
        if config is None
        else config
    )
    return fit_domain_covariance_calibration(
        predictor_id=PREDICTOR_ID,
        calibration_partition_id=PARTITION_ID,
        statistical_unit=STATISTICAL_UNIT,
        residual_semantics="prediction-error-m",
        covariance_semantics="raw-predictive-covariance-m2",
        event_ids=event_ids,
        group_ids=group_ids,
        domain_ids=domain_ids,
        residuals=residual_values,
        covariances=covariance_values,
        domain_guard=guard,  # type: ignore[arg-type]
        predictor_frozen_before_calibration_outcomes=True,
        transform_grid_frozen_before_calibration_outcomes=True,
        application_outcomes_used_for_calibration_selection=False,
        calibration_groups_independent=True,
        config=settings,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: calibration._canonical_string(" bad ", name="value"),
            "canonical string",
        ),
        (lambda: calibration._strings(1, name="values"), "sequence"),
        (lambda: calibration._strings((), name="values"), "must not be empty"),
        (lambda: calibration._number(True, name="value"), "finite number"),
        (
            lambda: calibration._number(np.ones(2), name="value"),
            "finite number",
        ),
        (lambda: calibration._number(np.nan, name="value"), "finite number"),
        (lambda: calibration._grid(1, name="grid", positive=True), "sequence"),
        (
            lambda: calibration._grid((), name="grid", positive=True),
            "must not be empty",
        ),
        (lambda: calibration._residuals(np.empty((0, 1))), "must not be empty"),
        (
            lambda: calibration._covariances(
                np.asarray([["bad"]]),
                calibration=False,
            ),
            "real numeric",
        ),
        (
            lambda: calibration._covariances(
                np.ones((1, 1, 1, 1)),
                calibration=False,
            ),
            "shape",
        ),
        (
            lambda: calibration._covariances(
                np.eye(1),
                calibration=True,
            ),
            "calibration covariances",
        ),
        (
            lambda: calibration._covariances(
                np.empty((0, 1, 1)),
                calibration=False,
            ),
            "must not be empty",
        ),
        (
            lambda: calibration._transform_score((1.0, 0.0)),
            "three fields",
        ),
        (
            lambda: calibration._held_score(("group", 1.0, 0.0, 1.0)),
            "five fields",
        ),
    ],
)
def test_private_contract_edges_fail_closed(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()


def test_nonfinite_transformed_covariance_is_rejected() -> None:
    covariance = np.asarray([[[np.finfo(np.float64).max]]])

    with pytest.raises(ValueError, match="transformed covariance must be finite"):
        calibration._transform(covariance, 2.0, 0.0)


@pytest.mark.parametrize(
    ("config", "guard", "match"),
    [
        (object(), None, "config must be"),
        (None, object(), "domain_guard must be"),
    ],
)
def test_fit_rejects_wrong_contract_types(
    config: object | None,
    guard: object | None,
    match: str,
) -> None:
    with pytest.raises(TypeError, match=match):
        _fit(config=config, domain_guard=guard)


def test_fit_rejects_residual_covariance_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions differ"):
        _fit(
            residuals=np.ones((4, 2)),
            covariances=np.ones((4, 1, 1)),
        )


def test_one_group_domain_uses_raw_leave_one_group_out_fallback() -> None:
    certificate = _fit(group_ids=("only",), domain_ids=("single",))
    decision = certificate.decision_for_domain("single")

    assert decision is not None
    assert len(decision.leave_one_group_out) == 1
    held = decision.leave_one_group_out[0]
    assert held[1:3] == (1.0, 0.0)
    assert not decision.calibration_supported


def test_domain_missing_from_guard_fails_closed_without_roster_error() -> None:
    guard = _guard(("other-0", "other-1", "other-2"), ("other",) * 3)
    certificate = _fit(domain_guard=guard)
    decision = certificate.decision_for_domain("dynamic")

    assert decision is not None
    assert not decision.guard_supported
    assert not decision.calibration_supported
    assert "calibration-domain-guard-rejected" in decision.reasons


def test_apply_rejects_wrong_certificate_type() -> None:
    with pytest.raises(TypeError, match="certificate must be"):
        apply_domain_covariance_calibration(
            np.eye(1),
            object(),  # type: ignore[arg-type]
            domain_id="dynamic",
            inference_admissible=True,
        )


def test_application_digests_bind_actual_single_matrix_shapes() -> None:
    certificate = _fit()
    raw = np.asarray([[1.0, 0.1], [0.1, 2.0]])

    output, record = apply_domain_covariance_calibration(
        raw,
        certificate,
        domain_id="dynamic",
        inference_admissible=True,
    )

    assert record.raw_covariance_sha256 == calibration._array_digest(raw)
    assert record.output_covariance_sha256 == calibration._array_digest(output)
    assert output.shape == raw.shape


def test_identity_transform_retains_exact_raw_object() -> None:
    certificate = _fit(residuals=np.ones((4, 1)))
    decision = certificate.decision_for_domain("dynamic")
    raw = np.asarray([[1.0]])

    output, record = apply_domain_covariance_calibration(
        raw,
        certificate,
        domain_id="dynamic",
        inference_admissible=True,
    )

    assert decision is not None
    assert decision.calibration_supported
    assert decision.selected_covariance_scale == 1.0
    assert decision.selected_isotropic_variance == 0.0
    assert output is raw
    assert record.exact_fallback
    assert not record.applied
    assert record.reason == "calibration-identity-transform-retained"


def test_application_record_rejects_internally_inconsistent_states() -> None:
    applied_output, applied = apply_domain_covariance_calibration(
        np.asarray([[1.0]]),
        _fit(),
        domain_id="dynamic",
        inference_admissible=True,
    )
    assert applied_output.shape == (1, 1)
    raw = np.asarray([[1.0]])
    _, fallback = apply_domain_covariance_calibration(
        raw,
        _fit(),
        domain_id="missing",
        inference_admissible=True,
    )

    invalid_factories = (
        lambda: replace(applied, inference_admissible=False, artifact_id=None),
        lambda: replace(applied, reason="wrong", artifact_id=None),
        lambda: replace(
            applied,
            covariance_scale=1.0,
            isotropic_variance=0.0,
            artifact_id=None,
        ),
        lambda: replace(fallback, covariance_scale=2.0, artifact_id=None),
        lambda: replace(fallback, calibration_supported=True, artifact_id=None),
        lambda: replace(fallback, reason="wrong", artifact_id=None),
    )

    for factory in invalid_factories:
        with pytest.raises(ValueError):
            factory()
