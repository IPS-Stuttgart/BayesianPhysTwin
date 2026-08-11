from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.calibration_domain_guard import (
    CalibrationDomainGuardCertificateV1,
    fit_calibration_domain_guard,
)
from bayesian_phystwin.posterior_covariance_semantics import (
    exact_prior_fallback_covariance_semantics,
    working_irls_covariance_semantics,
)
from bayesian_phystwin.regime_covariance_calibration import (
    CovarianceCalibrationGroupV1,
    RegimeCovarianceCalibrationV1,
    RegimeCovarianceTransformV1,
    calibrate_guarded_regime_covariance,
    calibrate_regime_covariance,
    calibrated_covariance_semantics,
    fit_regime_covariance_calibration,
    load_regime_covariance_calibration,
    save_regime_covariance_calibration,
)

_IDS = {
    "predictor_id": "1" * 64,
    "query_set_id": "2" * 64,
    "grouping_rule_id": "3" * 64,
    "calibration_evidence_id": "4" * 64,
}


def _groups() -> tuple[CovarianceCalibrationGroupV1, ...]:
    result: list[CovarianceCalibrationGroupV1] = []
    specifications = {
        "dynamic": (
            np.asarray([[4.0, 1.5], [1.5, 1.0]]),
            np.eye(2),
            100,
        ),
        "quasi_static": (
            np.asarray([[0.9, 0.0], [0.0, 1.1]]),
            np.eye(2),
            200,
        ),
    }
    for regime_id, (truth_covariance, raw_covariance, seed_offset) in (
        specifications.items()
    ):
        for index in range(4):
            rng = np.random.default_rng(seed_offset + index)
            residuals = rng.multivariate_normal(
                np.zeros(2),
                truth_covariance,
                size=48,
            )
            covariances = np.repeat(raw_covariance[None, :, :], 48, axis=0)
            result.append(
                CovarianceCalibrationGroupV1(
                    group_id=f"{regime_id}-{index}",
                    regime_id=regime_id,
                    residuals=residuals,
                    covariances=covariances,
                    metadata={"physical_unit": "object-session"},
                )
            )
    return tuple(result)


def _guard(
    groups: tuple[CovarianceCalibrationGroupV1, ...] | None = None,
    *,
    frozen: bool = True,
) -> CalibrationDomainGuardCertificateV1:
    source = groups or _groups()
    candidate_losses = [
        0.75 if group.regime_id == "dynamic" else 1.10
        for group in source
    ]
    return fit_calibration_domain_guard(
        calibration_partition_id="5" * 64,
        statistical_unit="independent-object-session",
        metric="gaussian-nll",
        group_ids=[group.group_id for group in source],
        domain_ids=[group.regime_id for group in source],
        candidate_losses=candidate_losses,
        fallback_losses=np.ones(len(source)),
        guard_frozen_before_application_outcomes=frozen,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
    )


def _fit(
    groups: tuple[CovarianceCalibrationGroupV1, ...] | None = None,
    *,
    application_outcomes_used_for_fit: bool = False,
) -> RegimeCovarianceCalibrationV1:
    return fit_regime_covariance_calibration(
        groups or _groups(),
        scales=(0.5, 1.0, 2.0, 4.0),
        floor_fractions=(0.0, 0.05),
        shrinkages=(0.0, 1.0),
        max_rank=2,
        predictor_frozen_before_calibration_outcomes=True,
        transform_family_frozen_before_calibration_outcomes=True,
        calibration_groups_independent=True,
        application_outcomes_used_for_fit=application_outcomes_used_for_fit,
        metadata={"protocol": "source-only-loo-v1"},
        **_IDS,
    )


def test_fit_improves_dynamic_heldout_score_and_preserves_identity_option() -> None:
    calibration = _fit()
    dynamic = calibration.transform_for("dynamic")

    assert calibration.deployment_admissible
    assert dynamic.calibrated_loo_nll < dynamic.raw_loo_nll
    assert dynamic.effective_rank >= 1
    assert 1.0 in dynamic.scale_grid
    assert 0.0 in dynamic.floor_fraction_grid
    assert 0.0 in dynamic.shrinkage_grid
    assert abs(dynamic.calibrated_refit_normalized_nees - 1.0) < abs(
        dynamic.raw_normalized_nees - 1.0
    )

    calibrated = calibrate_regime_covariance(
        np.eye(2),
        regime_id="dynamic",
        calibration=calibration,
    )
    assert calibrated[0, 1] > 0.5
    assert np.linalg.eigvalsh(calibrated).min() > 0.0
    assert not calibrated.flags.writeable
    with pytest.raises(ValueError):
        calibrated.setflags(write=True)


def test_group_and_input_order_are_content_invariant() -> None:
    groups = _groups()
    forward = _fit(groups)
    reverse = _fit(tuple(reversed(groups)))

    assert reverse.artifact_id == forward.artifact_id
    for transform in forward.transforms:
        assert transform.calibration_group_ids == tuple(
            sorted(transform.calibration_group_ids)
        )


def test_group_constructor_detaches_arrays_and_binds_content() -> None:
    residuals = np.ones((3, 2))
    covariances = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
    group = CovarianceCalibrationGroupV1(
        group_id="object-1",
        regime_id="dynamic",
        residuals=residuals,
        covariances=covariances,
    )
    artifact_id = group.artifact_id

    residuals[:] = 100.0
    covariances[:] = 100.0

    np.testing.assert_array_equal(group.residuals, np.ones((3, 2)))
    np.testing.assert_array_equal(
        group.covariances,
        np.repeat(np.eye(2)[None, :, :], 3, axis=0),
    )
    assert group.artifact_id == artifact_id
    for array in (group.residuals, group.covariances):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_unknown_regime_and_wrong_dimension_fail_closed() -> None:
    calibration = _fit()
    with pytest.raises(KeyError, match="unknown"):
        calibrate_regime_covariance(
            np.eye(2),
            regime_id="unregistered",
            calibration=calibration,
        )
    with pytest.raises(ValueError, match="dimension"):
        calibrate_regime_covariance(
            np.eye(3),
            regime_id="dynamic",
            calibration=calibration,
        )


def test_diagnostic_information_order_cannot_route_deployment() -> None:
    diagnostic = _fit(application_outcomes_used_for_fit=True)
    assert not diagnostic.deployment_admissible

    with pytest.raises(PermissionError, match="diagnostic"):
        calibrate_regime_covariance(
            np.eye(2),
            regime_id="dynamic",
            calibration=diagnostic,
        )
    exploratory = calibrate_regime_covariance(
        np.eye(2),
        regime_id="dynamic",
        calibration=diagnostic,
        require_deployment_admissible=False,
    )
    assert exploratory.shape == (2, 2)


def test_merged_domain_guard_authorizes_only_supported_matching_regime() -> None:
    groups = _groups()
    calibration = _fit(groups)
    certificate = _guard(groups)

    guarded = calibrate_guarded_regime_covariance(
        np.eye(2),
        regime_id="dynamic",
        calibration=calibration,
        certificate=certificate,
    )
    direct = calibrate_regime_covariance(
        np.eye(2),
        regime_id="dynamic",
        calibration=calibration,
    )
    np.testing.assert_allclose(guarded, direct)
    with pytest.raises(PermissionError, match="rejects"):
        calibrate_guarded_regime_covariance(
            np.eye(2),
            regime_id="quasi_static",
            calibration=calibration,
            certificate=certificate,
        )
    with pytest.raises(KeyError, match="unknown"):
        calibrate_guarded_regime_covariance(
            np.eye(2),
            regime_id="unknown",
            calibration=calibration,
            certificate=certificate,
        )
    with pytest.raises(PermissionError, match="diagnostic"):
        calibrate_guarded_regime_covariance(
            np.eye(2),
            regime_id="dynamic",
            calibration=calibration,
            certificate=_guard(groups, frozen=False),
        )


def test_domain_guard_and_covariance_calibration_group_rosters_must_match() -> None:
    groups = _groups()
    calibration = _fit(groups)
    changed_groups = tuple(
        CovarianceCalibrationGroupV1(
            group_id=(
                "replacement-dynamic-0"
                if group.group_id == "dynamic-0"
                else group.group_id
            ),
            regime_id=group.regime_id,
            residuals=group.residuals,
            covariances=group.covariances,
        )
        for group in groups
    )
    with pytest.raises(ValueError, match="different group rosters"):
        calibrate_guarded_regime_covariance(
            np.eye(2),
            regime_id="dynamic",
            calibration=calibration,
            certificate=_guard(changed_groups),
        )


def test_covariance_semantics_bind_exact_calibration_and_regime() -> None:
    calibration = _fit()
    raw = working_irls_covariance_semantics(np.eye(2))
    certificate = _guard()
    calibrated = calibrated_covariance_semantics(
        raw,
        regime_id="dynamic",
        calibration=calibration,
        certificate=certificate,
    )

    assert calibrated.calibrated
    assert calibrated.calibration_artifact_id == calibration.artifact_id
    assert calibrated.metadata["regime_id"] == "dynamic"
    assert calibrated.metadata["regime_covariance_transform_id"] == (
        calibration.transform_for("dynamic").artifact_id
    )
    assert calibrated.metadata["calibration_domain_guard_certificate_id"] == (
        certificate.artifact_id
    )
    with pytest.raises(ValueError, match="already calibrated"):
        calibrated_covariance_semantics(
            calibrated,
            regime_id="dynamic",
            calibration=calibration,
        )
    fallback = exact_prior_fallback_covariance_semantics(
        np.eye(2),
        reason="guard-rejected",
    )
    with pytest.raises(ValueError, match="fallback"):
        calibrated_covariance_semantics(
            fallback,
            regime_id="dynamic",
            calibration=calibration,
        )


def test_save_load_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    calibration = _fit()
    path = tmp_path / "regime-calibration.json"
    save_regime_covariance_calibration(calibration, path)
    loaded = load_regime_covariance_calibration(path)

    assert loaded.artifact_id == calibration.artifact_id
    assert loaded.to_record() == calibration.to_record()
    with pytest.raises(FileExistsError):
        save_regime_covariance_calibration(calibration, path)

    record = json.loads(path.read_text(encoding="utf-8"))
    record["transforms"][0]["covariance_scale"] *= 2.0
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id|grid"):
        load_regime_covariance_calibration(path)


def test_duplicate_json_keys_and_admissibility_tampering_fail_closed(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_regime_covariance_calibration(duplicate)

    calibration = _fit()
    record = calibration.to_record()
    record["deployment_admissible"] = False
    with pytest.raises(ValueError, match="contradicts"):
        RegimeCovarianceCalibrationV1.from_mapping(record)


@pytest.mark.parametrize(
    ("residuals", "covariances", "message"),
    [
        (np.ones(2), np.eye(2), "shape"),
        (np.asarray([[np.nan, 0.0]]), np.eye(2)[None], "finite"),
        (
            np.ones((1, 2)),
            np.asarray([[[1.0, 1.0], [0.0, 1.0]]]),
            "symmetric",
        ),
        (
            np.ones((1, 2)),
            np.asarray([[[1.0, 0.0], [0.0, 0.0]]]),
            "positive definite",
        ),
    ],
)
def test_invalid_group_geometry_fails_closed(
    residuals: np.ndarray,
    covariances: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CovarianceCalibrationGroupV1(
            group_id="object-1",
            regime_id="dynamic",
            residuals=residuals,
            covariances=covariances,
        )


def test_fit_requires_identity_grid_unique_groups_and_three_groups() -> None:
    groups = _groups()
    common = {
        **_IDS,
        "predictor_frozen_before_calibration_outcomes": True,
        "transform_family_frozen_before_calibration_outcomes": True,
        "calibration_groups_independent": True,
        "application_outcomes_used_for_fit": False,
    }
    with pytest.raises(ValueError, match="identity"):
        fit_regime_covariance_calibration(
            groups,
            scales=(2.0,),
            floor_fractions=(0.0,),
            shrinkages=(0.0,),
            **common,
        )
    with pytest.raises(ValueError, match="globally unique"):
        fit_regime_covariance_calibration(
            groups + (groups[0],),
            **common,
        )
    with pytest.raises(ValueError, match="at least three"):
        fit_regime_covariance_calibration(
            tuple(group for group in groups if group.group_id.endswith(("-0", "-1"))),
            **common,
        )


def test_group_literal_numeric_and_artifact_validation_fail_closed() -> None:
    residuals = np.ones((2, 2))
    covariances = np.repeat(np.eye(2)[None, :, :], 2, axis=0)
    with pytest.raises(ValueError, match="canonical literal"):
        CovarianceCalibrationGroupV1(
            group_id=" object-1",
            regime_id="dynamic",
            residuals=residuals,
            covariances=covariances,
        )
    with pytest.raises(ValueError, match="real numeric"):
        CovarianceCalibrationGroupV1(
            group_id="object-1",
            regime_id="dynamic",
            residuals=np.asarray([["x", "y"]]),
            covariances=np.eye(2)[None],
        )
    with pytest.raises(ValueError, match="real numeric"):
        CovarianceCalibrationGroupV1(
            group_id="object-1",
            regime_id="dynamic",
            residuals=np.ones((1, 2)),
            covariances=np.asarray([[['x', 'y'], ['z', 'w']]]),
        )
    with pytest.raises(ValueError, match="matching residuals"):
        CovarianceCalibrationGroupV1(
            group_id="object-1",
            regime_id="dynamic",
            residuals=np.ones((2, 2)),
            covariances=np.eye(2)[None],
        )
    with pytest.raises(ValueError, match="finite"):
        CovarianceCalibrationGroupV1(
            group_id="object-1",
            regime_id="dynamic",
            residuals=np.ones((1, 2)),
            covariances=np.asarray([[[np.nan, 0.0], [0.0, 1.0]]]),
        )

    valid = CovarianceCalibrationGroupV1(
        group_id="object-1",
        regime_id="dynamic",
        residuals=residuals,
        covariances=covariances,
    )
    rebound = CovarianceCalibrationGroupV1(
        group_id=valid.group_id,
        regime_id=valid.regime_id,
        residuals=valid.residuals,
        covariances=valid.covariances,
        artifact_id=valid.artifact_id,
    )
    assert rebound.artifact_id == valid.artifact_id
    with pytest.raises(ValueError, match="does not match content"):
        CovarianceCalibrationGroupV1(
            group_id=valid.group_id,
            regime_id=valid.regime_id,
            residuals=valid.residuals,
            covariances=valid.covariances,
            artifact_id="0" * 64,
        )


def test_transform_constructor_rejects_grid_and_factor_defects() -> None:
    groups = _groups()
    transform = _fit(groups).transform_for("dynamic")
    base = dict(
        regime_id=transform.regime_id,
        dimension=transform.dimension,
        covariance_scale=transform.covariance_scale,
        isotropic_variance=transform.isotropic_variance,
        additive_factor=transform.additive_factor,
        selected_floor_fraction=transform.selected_floor_fraction,
        selected_shrinkage=transform.selected_shrinkage,
        max_rank=transform.max_rank,
        scale_grid=transform.scale_grid,
        floor_fraction_grid=transform.floor_fraction_grid,
        shrinkage_grid=transform.shrinkage_grid,
        calibration_group_ids=transform.calibration_group_ids,
        calibration_group_artifact_ids=transform.calibration_group_artifact_ids,
        raw_loo_nll=transform.raw_loo_nll,
        calibrated_loo_nll=transform.calibrated_loo_nll,
        refit_nll=transform.refit_nll,
        raw_normalized_nees=transform.raw_normalized_nees,
        calibrated_refit_normalized_nees=transform.calibrated_refit_normalized_nees,
    )
    with pytest.raises(ValueError, match="rank"):
        RegimeCovarianceTransformV1(**base, max_rank=0, additive_factor=np.ones((2, 1))
    with pytest.raises(ValueError, match="blong to scale_grid"):
        RegimeCovarianceTransformV1(**base, covariance_scale=3.0)
    with pytest.raises(ValueError, match="duplicates"):
        RegimeCovarianceTransformV1(**base, scale_grid=(1.0, 1.0))


def test_calibration_constructor_rejects_duplicate_regimes_and_identity_tampering() -> None:
    calibration = _fit()
    transform = calibration.transform_for("dynamic")
    with pytest.raises(ValueError, match="one entry per regime"):
        RegimeCovarianceCalibrationV1(
            predictor_id=calibration.predictor_id,
            query_set_id=calibration.query_set_id,
            grouping_rule_id=calibration.grouping_rule_id,
            calibration_evidence_id=calibration.calibration_evidence_id,
            transforms=(transform, transform),
            predictor_frozen_before_calibration_outcomes=True,
            transform_family_frozen_before_calibration_outcomes=True,
            calibration_groups_independent=True,
            application_outcomes_used_for_fit=False,
        )
    record = calibration.to_record()
    record["predictor_id"] = "9"*64
    with pytest.raises(ValueError, match="artifact_id|content"):
        RegimeCovarianceCalibrationV1.from_mapping(record)

def test_calibrate_rejects_batch_geometry_and_nonnumeric_content() -> None:
    calibration = _fit()
    with pytest.raises(ValueError, match="symmetric"):
        calibrate_regime_covariance(
            np.asarray([[[1.0, 1.0], [0.0, 1.0]]]),
            regime_id="dynamic",
            calibration=calibration,
         )
    with pytest.raises(ValueError, match="real numeric"):
        calibrate_regime_covariance(
            np.asarray([["x", "y"], ["z", "w"]]),
            regime_id="dynamic",
            calibration=calibration,
        )

def test_semantics_contracts_reject_dimension_and_type_defects() -> None:
    calibration = _fit()
    with pytest.raises(TypeError, match="semantics"):
        calibrated_covariance_semantics(
            object(),
            regime_id="dynamic",
            calibration=calibration,
        )
    small = RegimeCovarianceCalibrationV1(
        predictor_id=calibration.predictor_id,
        query_set_id=calibration.query_set_id,
        grouping_rule_id=calibration.grouping_rule_id,
        calibration_evidence_id=calibration.calibration_evidence_id,
        transforms=(tuple(
            RegimeCovarianceTransformV1(
                regime_id="tiny",
                dimension=1,
                covariance_scale=1.0,
                isotropic_variance=0.0,
                additive_factor=np.empty((1, 0)),
                selected_floor_fraction=0.0,
                selected_shrinkage=0.0,
                max_rank=1,
                scale_grid=(1.0,),
                floor_fraction_grid=(0.0,),
                shrinkage_grid=(0.0,),
                calibration_group_ids=("a", "b", "c"),
                calibration_group_artifact_ids=("1"*64, "2"*64, "3"*64),
                raw_loo_nll=1.0,
                calibrated_loo_nll=1.0,
                refit_nll=1.0,
                raw_normalized_nees=1.0,
                calibrated_refit_normalized_nees=1.0,
            ),
        ),
        predictor_frozen_before_calibration_outcomes=True,
        transform_family_frozen_before_calibration_outcomes=True,
        calibration_groups_independent=True,
        application_outcomes_used_for_fit=False,
    )
    with pytest.raises(ValueError, match="dimension"):
        calibrated_covariance_semantics(
            working_irls_covariance_semantics(np.eye(2)),
            regime_id="tiny",
            calibration=small,
        )

def test_calibration_save_and_calibrate_parameter_type_defects(tmp_path: Path) -> None:
    calibration = _fit()
    with pytest.raises(TypeError, match="calibration"):
        save_regime_covariance_calibration(object(), tmp_path / "bad.json")
    with pytest.raises(TypeError, match="calibration"):
        calibrate_regime_covariance(
            np.eye(2),
            regime_id="dynamic",
            calibration=object(),
        )


def test_fit_rejects_boolean_grid_entries() -> None:
    with pytest.raises(ValueError, match="finite real"):
        fit_regime_covariance_calibration(
            _groups(),
            scales=(True, 1.0),
            floor_fractions=(0.0,),
            shrinkages=(0.0,),
            max_rank=2,
            predictor_frozen_before_calibration_outcomes=True,
            transform_family_frozen_before_calibration_outcomes=True,
            calibration_groups_independent=True,
            application_outcomes_used_for_fit=False,
            **_IDS,
        )

def test_group_artifact_ids_are_order_invariant_in_transform() -> None:
    groups = _groups()
    forward = _fit(groups)
    reversed = _fit(tuple(reversed(groups)))

    for left, right in zip(forward.transforms, reversed.transforms, strict=True):
        assert left.calibration_group_artifact_ids == right.calibration_group_artifact_ids

def test_calibration_detaches_metadata_and_transforms() -> None:
    calibration = _fit()
    metadata = dict(calibration.metadata)
    transform = calibration.transform_for("dynamic")
    with pytest.raises(TypeError):
        metadata["x"] = object()
    with pytest.raises(ValueError):
        transform.additive_factor.setflags(write=True)

def test_calibrated_semantics_require_deployment_admissible_calibration() -> None:
    diagnostic = _fit(application_outcomes_used_for_fit=True)
    with pytest.raises(PermissionError, match="diagnostic"):
        calibrated_covariance_semantics(
            working_irls_covariance_semantics(np.eye(2)),
            regime_id="dynamic",
            calibration=diagnostic,
        )

def test_guarded_calibration_rejects_inconsistent_certificate_roster() -> None:
    groups = _groups()
    calibration = _fit(groups)
    certificate = _guard(groups)
    tampered = CalibrationDomainGuardCertificateV1(
        calibration_partition_id=certificate.calibration_partition_id,
        statistical_unit=certificate.statistical_unit,
        metric=certificate.metric,
        decisions=certificate.decisions,
        guard_frozen_before_application_outcomes=True,
        application_outcomes_used_for_guard_selection=False,
        calibration_groups_independent=True,
     )
    with pytest.raises(ValueError, match="different group rosters"):
        calibrate_guarded_regime_covariance(
            np.eye(2),
            regime_id="dynamic",
            calibration=calibration,
            certificate=tampered,
        )

def test_guarded_calibration_requires_valid_certificate_type() -> None:
    calibration = _fit()
    with pytest.raises(TypeError, match="certificate"):
        calibrate_guarded_regime_covariance(
            np.eye(2),
            regime_id="dynamic",
            calibration=calibration,
            certificate=object(),
        )
