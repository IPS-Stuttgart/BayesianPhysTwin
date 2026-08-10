from __future__ import annotations

import numpy as np
import pytest
from domain_covariance_calibration_test_helpers import _certificate, _inputs

from bayesian_phystwin.domain_covariance_calibration import (
    apply_domain_covariance_calibration,
    fit_domain_covariance_calibration,
)


def test_cross_fitted_calibration_authorizes_only_undercovered_domain() -> None:
    certificate = _certificate()

    assert certificate.deployment_admissible
    assert certificate.supported_domains == ("dynamic",)
    dynamic = certificate.transform_for_domain("dynamic")
    quasi_static = certificate.transform_for_domain("quasi-static")
    assert dynamic is not None
    assert dynamic.scale > 1.0
    assert dynamic.calibrated_group_balanced_nll_per_dimension < (
        dynamic.raw_group_balanced_nll_per_dimension
    )
    assert dynamic.calibrated_mean_normalized_energy == pytest.approx(1.0, abs=0.1)
    assert quasi_static is not None
    assert quasi_static.scale == 1.0
    assert quasi_static.isotropic_floor_variance == 0.0


def test_each_predictive_fold_excludes_its_held_out_group() -> None:
    certificate = _certificate()

    assert len(certificate.fold_records) == 6
    for fold in certificate.fold_records:
        assert fold.held_out_group_id not in fold.training_group_ids
        assert len(fold.training_group_ids) == 2
    dynamic = [fold for fold in certificate.fold_records if fold.domain_id == "dynamic"]
    assert all(fold.log_loss_ratio < 0.0 for fold in dynamic)
    assert all(
        fold.calibrated_normalized_energy < fold.raw_normalized_energy
        for fold in dynamic
    )


def test_supported_domain_applies_immutable_covariance_inflation() -> None:
    raw = np.eye(3)

    calibrated, record = apply_domain_covariance_calibration(
        raw,
        _certificate(),
        domain_id="dynamic",
    )

    assert calibrated is not raw
    assert record.calibration_applied
    assert record.reason == "domain-covariance-calibration-applied"
    assert np.all(np.linalg.eigvalsh(calibrated) > np.linalg.eigvalsh(raw))
    assert not calibrated.flags.writeable
    with pytest.raises(ValueError):
        calibrated[0, 0] = 0.0


def test_rejected_domain_returns_exact_covariance_object() -> None:
    raw = np.eye(3)

    selected, record = apply_domain_covariance_calibration(
        raw,
        _certificate(),
        domain_id="quasi-static",
    )

    assert selected is raw
    assert not record.calibration_applied
    assert record.input_covariance_id == record.output_covariance_id
    assert record.reason == "calibration-domain-rejected"


def test_unknown_domain_returns_exact_covariance_object() -> None:
    raw = np.eye(3)

    selected, record = apply_domain_covariance_calibration(
        raw,
        _certificate(),
        domain_id="unseen-regime",
    )

    assert selected is raw
    assert not record.calibration_applied
    assert record.reason == "unknown-calibration-domain"


@pytest.mark.parametrize(
    ("frozen_before", "application_used", "independent"),
    ((False, False, True), (True, True, True), (True, False, False)),
)
def test_nonprospective_certificate_forces_exact_array_fallback(
    frozen_before: bool,
    application_used: bool,
    independent: bool,
) -> None:
    certificate = _certificate(
        guard_frozen_before_application_outcomes=frozen_before,
        application_outcomes_used_for_guard_selection=application_used,
        calibration_groups_independent=independent,
    )
    raw = np.eye(3)

    selected, record = apply_domain_covariance_calibration(
        raw,
        certificate,
        domain_id="dynamic",
    )

    assert not certificate.deployment_admissible
    assert selected is raw
    assert not record.calibration_applied
    assert record.reason == "calibration-information-boundary-rejected"


def test_group_and_sample_permutations_preserve_certificate_identity() -> None:
    first = _certificate()
    arguments = _inputs()
    group_permutation = np.asarray([5, 1, 3, 0, 4, 2])
    sample_permutation = np.asarray([5, 2, 0, 4, 1, 3])
    for name in ("group_ids", "domain_ids"):
        values = np.asarray(arguments[name])
        arguments[name] = tuple(str(value) for value in values[group_permutation])
    for name in ("sample_ids", "residuals", "covariances"):
        values = list(arguments[name])
        reordered = []
        for group_index in group_permutation:
            value = values[int(group_index)]
            if name == "sample_ids":
                reordered.append(
                    tuple(str(item) for item in np.asarray(value)[sample_permutation])
                )
            else:
                reordered.append(np.asarray(value)[sample_permutation])
        arguments[name] = tuple(reordered)

    second = fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

    assert second.calibration_data_id == first.calibration_data_id
    assert second.artifact_id == first.artifact_id


def test_group_balancing_prevents_long_group_from_changing_equal_group_result() -> None:
    first = _certificate()
    arguments = _inputs()
    residuals = list(arguments["residuals"])
    covariances = list(arguments["covariances"])
    sample_ids = list(arguments["sample_ids"])
    residuals[0] = np.repeat(np.asarray(residuals[0]), 20, axis=0)
    covariances[0] = np.repeat(np.asarray(covariances[0]), 20, axis=0)
    sample_ids[0] = tuple(f"repeat-{index:03d}" for index in range(len(residuals[0])))
    arguments["residuals"] = tuple(residuals)
    arguments["covariances"] = tuple(covariances)
    arguments["sample_ids"] = tuple(sample_ids)

    second = fit_domain_covariance_calibration(**arguments)  # type: ignore[arg-type]

    first_dynamic = first.transform_for_domain("dynamic")
    second_dynamic = second.transform_for_domain("dynamic")
    assert first_dynamic is not None and second_dynamic is not None
    assert second_dynamic.scale == first_dynamic.scale
    assert second_dynamic.isotropic_floor_variance == (
        first_dynamic.isotropic_floor_variance
    )
