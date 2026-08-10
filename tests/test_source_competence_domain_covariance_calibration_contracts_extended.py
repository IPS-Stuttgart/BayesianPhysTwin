from __future__ import annotations

from dataclasses import replace

import pytest
from domain_covariance_calibration_test_helpers import (
    _certificate,
    _certificate_arguments,
    _guard_for_folds,
)

from bayesian_phystwin.domain_covariance_calibration import (
    DomainCovarianceCalibrationCertificateV1,
)


def test_transform_and_fold_contracts_reject_tampering() -> None:
    certificate = _certificate()
    transform = certificate.transforms[0]
    assert replace(transform, artifact_id=transform.artifact_id) == transform
    with pytest.raises(ValueError, match="artifact_id"):
        replace(transform, artifact_id="0" * 64)
    with pytest.raises(ValueError, match="duplicates"):
        replace(
            transform,
            group_ids=(transform.group_ids[0], transform.group_ids[0]),
            artifact_id=None,
        )

    fold = certificate.fold_records[0]
    assert replace(fold, artifact_id=fold.artifact_id) == fold
    with pytest.raises(ValueError, match="artifact_id"):
        replace(fold, artifact_id="0" * 64)
    with pytest.raises(ValueError, match="duplicates"):
        replace(
            fold,
            training_group_ids=(fold.training_group_ids[0],) * 2,
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="held-out"):
        replace(
            fold,
            training_group_ids=(fold.held_out_group_id,),
            artifact_id=None,
        )
    with pytest.raises(ValueError, match="log_loss_ratio"):
        replace(fold, log_loss_ratio=0.0, artifact_id=None)


def test_certificate_roster_and_grid_invariants_reject_tampering() -> None:
    certificate = _certificate()
    base = _certificate_arguments(certificate)

    invalid_cases: list[tuple[str, object, str]] = [
        ("config", object(), "config must"),
        ("transforms", (), "transforms must"),
        ("transforms", (object(),), "transforms must"),
        ("fold_records", (), "fold_records must"),
        ("fold_records", (object(),), "fold_records must"),
        ("guard_certificate", object(), "guard_certificate must"),
    ]
    for field, value, match in invalid_cases:
        arguments = dict(base)
        arguments[field] = value
        with pytest.raises((TypeError, ValueError), match=match):
            DomainCovarianceCalibrationCertificateV1(  # type: ignore[arg-type]
                **arguments
            )

    transforms = list(certificate.transforms)
    transforms[1] = replace(
        transforms[1],
        domain_id=transforms[0].domain_id,
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="duplicate domains"):
        DomainCovarianceCalibrationCertificateV1(
            **{**base, "transforms": transforms}  # type: ignore[arg-type]
        )

    transforms = list(certificate.transforms)
    transforms[0] = replace(transforms[0], dimension=2, artifact_id=None)
    with pytest.raises(ValueError, match="certificate dimension"):
        DomainCovarianceCalibrationCertificateV1(
            **{**base, "transforms": transforms}  # type: ignore[arg-type]
        )

    folds = list(certificate.fold_records)
    foreign_index = next(
        index for index, fold in enumerate(folds) if fold.domain_id == "quasi-static"
    )
    folds[foreign_index] = replace(
        folds[foreign_index],
        held_out_group_id=folds[0].held_out_group_id,
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="unique held-out"):
        DomainCovarianceCalibrationCertificateV1(
            **{**base, "fold_records": folds}  # type: ignore[arg-type]
        )

    folds = list(certificate.fold_records)
    folds[0] = replace(folds[0], domain_id="foreign", artifact_id=None)
    with pytest.raises(ValueError, match="domain rosters"):
        DomainCovarianceCalibrationCertificateV1(
            **{**base, "fold_records": folds}  # type: ignore[arg-type]
        )


def test_certificate_transform_and_fold_values_must_match_frozen_grid() -> None:
    certificate = _certificate()
    base = _certificate_arguments(certificate)

    transforms = list(certificate.transforms)
    transforms[0] = replace(
        transforms[0],
        group_ids=transforms[0].group_ids[:-1],
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="group roster"):
        DomainCovarianceCalibrationCertificateV1(
            **{**base, "transforms": transforms}  # type: ignore[arg-type]
        )

    for field, value, match in (
        ("scale", 1.23, "transform scale"),
        (
            "isotropic_floor_variance",
            0.123 * certificate.transforms[0].reference_variance,
            "transform floor",
        ),
        (
            "calibrated_group_balanced_nll_per_dimension",
            certificate.transforms[0].raw_group_balanced_nll_per_dimension + 1.0,
            "worsens fitted NLL",
        ),
    ):
        transforms = list(certificate.transforms)
        transforms[0] = replace(
            transforms[0],
            **{field: value, "artifact_id": None},
        )
        with pytest.raises(ValueError, match=match):
            DomainCovarianceCalibrationCertificateV1(
                **{**base, "transforms": transforms}  # type: ignore[arg-type]
            )

    folds = list(certificate.fold_records)
    folds[0] = replace(
        folds[0],
        training_group_ids=(folds[0].training_group_ids[0],),
        artifact_id=None,
    )
    with pytest.raises(ValueError, match="training roster"):
        DomainCovarianceCalibrationCertificateV1(
            **{**base, "fold_records": folds}  # type: ignore[arg-type]
        )

    for field, value, match in (
        ("scale", 1.23, "fold scale"),
        (
            "isotropic_floor_variance",
            0.123 * certificate.fold_records[0].reference_variance,
            "fold floor",
        ),
    ):
        folds = list(certificate.fold_records)
        folds[0] = replace(
            folds[0],
            **{field: value, "artifact_id": None},
        )
        with pytest.raises(ValueError, match=match):
            DomainCovarianceCalibrationCertificateV1(
                **{**base, "fold_records": folds}  # type: ignore[arg-type]
            )


def test_certificate_rejects_mismatched_embedded_guard() -> None:
    certificate = _certificate()
    base = _certificate_arguments(certificate)
    folds = certificate.fold_records

    mismatched = (
        (
            _guard_for_folds(
                certificate,
                folds,
                calibration_partition_id="e" * 64,
            ),
            "partitions differ",
        ),
        (
            _guard_for_folds(certificate, folds, statistical_unit="other-unit"),
            "statistical units differ",
        ),
        (
            _guard_for_folds(certificate, folds, metric="other-metric"),
            "guard metric",
        ),
        (
            _guard_for_folds(
                certificate,
                folds,
                domain_ids=("foreign",) + tuple(item.domain_id for item in folds[1:]),
            ),
            "domain rosters",
        ),
    )
    for guard, match in mismatched:
        with pytest.raises(ValueError, match=match):
            DomainCovarianceCalibrationCertificateV1(
                **{**base, "guard_certificate": guard}  # type: ignore[arg-type]
            )
