from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.provider_physical_mapping_audit_v1 import (
    PROVIDER_PHYSICAL_MAPPING_AUDIT_INFORMATION_BOUNDARY,
    PROVIDER_PHYSICAL_MAPPING_AUDIT_SCHEMA,
    PROVIDER_PHYSICAL_MAPPING_AUDIT_VERSION,
    ProviderPhysicalMappingAuditV1,
    ProviderPhysicalMappingCaseV1,
    ProviderPhysicalMappingPolicyV1,
    audit_provider_physical_mapping,
)


def _digest(character: str) -> str:
    return character * 64


def _case(**updates: Any) -> ProviderPhysicalMappingCaseV1:
    values: dict[str, Any] = {
        "case_id": "source-case-01",
        "provider_artifact_id": _digest("0"),
        "physical_query_id": _digest("1"),
        "mapping_protocol_id": _digest("2"),
        "provider_frame": "camera_native",
        "physical_frame": "robot_world",
        "points_native": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [900.0, 900.0, 900.0],
            ]
        ),
        "valid_mask": np.array([True, True, True, False]),
        "provider_unit_scale_m": 0.01,
        "provider_to_physical": np.eye(4),
        "query_bounds_m": np.array(
            [
                [-0.001, -0.001, -0.001],
                [0.021, 0.001, 0.001],
            ]
        ),
        "timestamps_s": np.array([0.0, 0.1, 0.2, np.nan]),
        "query_time_window_s": np.array([0.0, 0.2]),
        "covariances_native": np.repeat(np.eye(3)[None], 4, axis=0),
        "metadata": {"information_split": "source-only"},
    }
    values.update(updates)
    return ProviderPhysicalMappingCaseV1(**values)


def _policy(**updates: Any) -> ProviderPhysicalMappingPolicyV1:
    values: dict[str, Any] = {
        "minimum_valid_point_count": 3,
        "minimum_valid_fraction": 0.75,
        "minimum_mapped_point_count": 3,
        "minimum_mapped_fraction": 1.0,
        "require_timestamps": True,
        "require_covariance": True,
    }
    values.update(updates)
    return ProviderPhysicalMappingPolicyV1(**values)


def _manual_audit(**updates: Any) -> ProviderPhysicalMappingAuditV1:
    values: dict[str, Any] = {
        "case_id": "source-case-01",
        "case_artifact_id": _digest("3"),
        "provider_artifact_id": _digest("0"),
        "physical_query_id": _digest("1"),
        "mapping_protocol_id": _digest("2"),
        "provider_frame": "camera_native",
        "physical_frame": "robot_world",
        "policy_id": _digest("4"),
        "mapping_admissible": True,
        "technical_valid": True,
        "provider_support_complete": True,
        "query_support_sufficient": True,
        "result_reason": "provider-physical-mapping-admissible",
        "rejection_reasons": (),
        "diagnostics": {},
    }
    values.update(updates)
    return ProviderPhysicalMappingAuditV1(**values)


def _diagnostics(audit: ProviderPhysicalMappingAuditV1) -> dict[str, Any]:
    return cast(dict[str, Any], audit.to_dict()["diagnostics"])


def test_admissible_mapping_binds_all_inputs_and_reports_accounting() -> None:
    case = _case()
    policy = _policy()

    audit = audit_provider_physical_mapping(case, policy)

    assert audit.mapping_admissible is True
    assert audit.technical_valid is True
    assert audit.provider_support_complete is True
    assert audit.query_support_sufficient is True
    assert audit.result_reason == "provider-physical-mapping-admissible"
    assert audit.rejection_reasons == ()
    assert audit.provider_artifact_id == case.provider_artifact_id
    assert audit.physical_query_id == case.physical_query_id
    assert audit.mapping_protocol_id == case.mapping_protocol_id
    assert audit.provider_frame == "camera_native"
    assert audit.physical_frame == "robot_world"
    assert audit.policy_id == policy.policy_id
    assert audit.provider_failure_signal_patch() == {
        "technical_valid": True,
        "provider_support_complete": True,
    }

    payload = audit.to_dict()
    assert payload["schema"] == PROVIDER_PHYSICAL_MAPPING_AUDIT_SCHEMA
    assert payload["schema_version"] == PROVIDER_PHYSICAL_MAPPING_AUDIT_VERSION
    assert payload["information_boundary"] == (
        PROVIDER_PHYSICAL_MAPPING_AUDIT_INFORMATION_BOUNDARY
    )
    assert len(cast(str, case.artifact_id)) == 64
    assert len(cast(str, audit.audit_id)) == 64
    json.dumps(payload, sort_keys=True, allow_nan=False)

    diagnostics = _diagnostics(audit)
    points = diagnostics["point_accounting"]
    assert points["point_count"] == 4
    assert points["declared_valid_count"] == 3
    assert points["declared_valid_fraction"] == pytest.approx(0.75)
    assert points["mapped_point_count"] == 3
    assert points["mapped_fraction_of_declared_valid"] == pytest.approx(1.0)
    assert diagnostics["provider_bbox_native"] == {
        "lower": [0.0, 0.0, 0.0],
        "upper": [2.0, 0.0, 0.0],
    }
    assert diagnostics["physical_bbox_m"] == {
        "lower": [0.0, 0.0, 0.0],
        "upper": [0.02, 0.0, 0.0],
    }
    assert diagnostics["mapped_bbox_m"] == diagnostics["physical_bbox_m"]
    assert diagnostics["time"]["provider_timestamp_range_s"] == [0.0, 0.2]
    covariance = diagnostics["covariance"]
    assert covariance["minimum_eigenvalue_m2"] == pytest.approx(1e-4)
    assert covariance["maximum_eigenvalue_m2"] == pytest.approx(1e-4)
    assert covariance["invalid_declared_valid_count"] == 0


def test_rigid_transform_and_boundary_tolerance_are_applied_in_meters() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    case = _case(
        provider_to_physical=transform,
        query_bounds_m=np.array(
            [
                [0.999, 1.999, 2.999],
                [1.0195, 2.001, 3.001],
            ]
        ),
    )
    policy = _policy(boundary_tolerance_m=0.001)

    audit = audit_provider_physical_mapping(case, policy)

    assert audit.mapping_admissible is True
    diagnostics = _diagnostics(audit)
    assert diagnostics["mapped_bbox_m"] == {
        "lower": [1.0, 2.0, 3.0],
        "upper": [1.02, 2.0, 3.0],
    }
    assert diagnostics["query_bounds_m"]["boundary_tolerance_m"] == 0.001


def test_provider_validity_and_query_overlap_remain_separate_decisions() -> None:
    case = _case(
        query_bounds_m=np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
    )

    audit = audit_provider_physical_mapping(case, _policy())

    assert audit.technical_valid is True
    assert audit.provider_support_complete is True
    assert audit.query_support_sufficient is False
    assert audit.mapping_admissible is False
    assert audit.result_reason == "insufficient-physical-query-overlap"
    assert audit.rejection_reasons == ("insufficient-physical-query-overlap",)
    assert audit.provider_failure_signal_patch() == {
        "technical_valid": True,
        "provider_support_complete": False,
    }
    assert _diagnostics(audit)["mapped_bbox_m"] is None


def test_invalid_and_numerically_nonfinite_transforms_fail_closed() -> None:
    reflection = np.eye(4)
    reflection[0, 0] = -1.0
    reflected = audit_provider_physical_mapping(
        _case(provider_to_physical=reflection),
        _policy(),
    )
    assert reflected.technical_valid is False
    assert reflected.rejection_reasons[0] == (
        "invalid-provider-to-physical-transform"
    )
    assert _diagnostics(reflected)["transform"]["rotation_determinant"] == -1.0

    huge = np.eye(4)
    huge[:3, :3] *= 1e308
    overflowed = audit_provider_physical_mapping(
        _case(provider_to_physical=huge),
        _policy(),
    )
    transform = _diagnostics(overflowed)["transform"]
    assert transform["finite"] is False
    assert transform["rotation_determinant"] is None
    assert transform["rotation_orthogonality_error"] is None
    json.dumps(overflowed.to_dict(), allow_nan=False)


def test_declared_valid_nonfinite_points_and_transform_overflow_are_distinct() -> None:
    points = np.array(
        [
            [np.nan, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
        ]
    )
    nonfinite = audit_provider_physical_mapping(
        _case(points_native=points),
        _policy(minimum_mapped_point_count=1, minimum_mapped_fraction=0.0),
    )
    assert "nonfinite-declared-valid-provider-points" in (
        nonfinite.rejection_reasons
    )
    assert _diagnostics(nonfinite)["point_accounting"][
        "nonfinite_declared_valid_count"
    ] == 1

    points[0] = [0.0, 0.0, 0.0]
    masked_nonfinite = audit_provider_physical_mapping(
        _case(points_native=points),
        _policy(),
    )
    assert masked_nonfinite.technical_valid is True

    overflow_points = np.array(
        [
            [1e308, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    overflowed = audit_provider_physical_mapping(
        _case(
            points_native=overflow_points,
            provider_unit_scale_m=2.0,
        ),
        _policy(minimum_mapped_point_count=1, minimum_mapped_fraction=0.0),
    )
    assert "nonfinite-transformed-provider-points" in (
        overflowed.rejection_reasons
    )


def test_timestamp_requirements_finiteness_and_overlap_fail_closed() -> None:
    missing = audit_provider_physical_mapping(
        _case(timestamps_s=None, query_time_window_s=None),
        _policy(),
    )
    assert missing.result_reason == "required-provider-timestamps-missing"
    assert _diagnostics(missing)["time"] == {
        "available": False,
        "required": True,
        "declared_valid_count": 3,
    }

    timestamps = np.array([0.0, np.nan, 0.2, np.nan])
    nonfinite = audit_provider_physical_mapping(
        _case(timestamps_s=timestamps),
        _policy(minimum_mapped_point_count=1, minimum_mapped_fraction=0.0),
    )
    assert "nonfinite-declared-valid-provider-timestamps" in (
        nonfinite.rejection_reasons
    )
    assert _diagnostics(nonfinite)["time"][
        "nonfinite_declared_valid_timestamp_count"
    ] == 1

    outside = audit_provider_physical_mapping(
        _case(query_time_window_s=np.array([1.0, 2.0])),
        _policy(),
    )
    assert outside.technical_valid is True
    assert outside.query_support_sufficient is False
    assert outside.result_reason == "insufficient-physical-query-overlap"


def test_optional_timestamps_and_covariance_can_be_absent() -> None:
    audit = audit_provider_physical_mapping(
        _case(
            timestamps_s=None,
            query_time_window_s=None,
            covariances_native=None,
        ),
        _policy(require_timestamps=False, require_covariance=False),
    )

    assert audit.mapping_admissible is True
    diagnostics = _diagnostics(audit)
    assert diagnostics["time"]["available"] is False
    assert diagnostics["time"]["required"] is False
    assert diagnostics["covariance"]["available"] is False
    assert diagnostics["covariance"]["required"] is False


def test_required_or_invalid_covariance_is_a_technical_failure() -> None:
    missing = audit_provider_physical_mapping(
        _case(covariances_native=None),
        _policy(),
    )
    assert "required-provider-covariance-missing" in missing.rejection_reasons

    nonfinite_covariance = np.repeat(np.eye(3)[None], 4, axis=0)
    nonfinite_covariance[1, 0, 0] = np.nan
    nonfinite_covariance[3] = np.nan
    nonfinite = audit_provider_physical_mapping(
        _case(covariances_native=nonfinite_covariance),
        _policy(minimum_mapped_point_count=1, minimum_mapped_fraction=0.0),
    )
    diagnostics = _diagnostics(nonfinite)["covariance"]
    assert diagnostics["nonfinite_native_count"] == 1
    assert diagnostics["invalid_declared_valid_count"] == 1

    asymmetric_covariance = np.repeat(np.eye(3)[None], 4, axis=0)
    asymmetric_covariance[0, 0, 1] = 1.0
    asymmetric = audit_provider_physical_mapping(
        _case(covariances_native=asymmetric_covariance),
        _policy(minimum_mapped_point_count=1, minimum_mapped_fraction=0.0),
    )
    assert _diagnostics(asymmetric)["covariance"][
        "symmetry_failure_count"
    ] == 1

    indefinite_covariance = np.repeat(np.eye(3)[None], 4, axis=0)
    indefinite_covariance[0, 0, 0] = -1.0
    indefinite = audit_provider_physical_mapping(
        _case(covariances_native=indefinite_covariance),
        _policy(minimum_mapped_point_count=1, minimum_mapped_fraction=0.0),
    )
    assert _diagnostics(indefinite)["covariance"][
        "eigenvalue_failure_count"
    ] == 1


def test_covariance_condition_and_transform_overflow_are_reported() -> None:
    ill_conditioned = np.repeat(np.eye(3)[None], 4, axis=0)
    ill_conditioned[:, 2, 2] = 1e-9
    conditioned = audit_provider_physical_mapping(
        _case(covariances_native=ill_conditioned),
        _policy(
            maximum_covariance_condition_number=1e6,
            minimum_mapped_point_count=1,
            minimum_mapped_fraction=0.0,
        ),
    )
    covariance = _diagnostics(conditioned)["covariance"]
    assert covariance["condition_failure_count"] == 3
    assert covariance["maximum_finite_condition_number"] == pytest.approx(1e9)

    singular = np.repeat(np.eye(3)[None], 4, axis=0)
    singular[:, 2, 2] = 0.0
    singular_audit = audit_provider_physical_mapping(
        _case(covariances_native=singular),
        _policy(
            maximum_covariance_condition_number=1e6,
            minimum_mapped_point_count=1,
            minimum_mapped_fraction=0.0,
        ),
    )
    assert _diagnostics(singular_audit)["covariance"][
        "condition_failure_count"
    ] == 3

    huge_rotation = np.eye(4)
    huge_rotation[:3, :3] *= 1e156
    overflowed = audit_provider_physical_mapping(
        _case(provider_to_physical=huge_rotation),
        _policy(
            minimum_mapped_point_count=1,
            minimum_mapped_fraction=0.0,
        ),
    )
    assert _diagnostics(overflowed)["covariance"][
        "nonfinite_transformed_count"
    ] == 3


def test_covariance_eigendecomposition_failure_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(_: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("synthetic failure")

    monkeypatch.setattr(np.linalg, "eigvalsh", _fail)
    audit = audit_provider_physical_mapping(
        _case(),
        _policy(minimum_mapped_point_count=1, minimum_mapped_fraction=0.0),
    )
    covariance = _diagnostics(audit)["covariance"]
    assert covariance["eigendecomposition_failure_count"] == 3
    assert covariance["minimum_eigenvalue_m2"] is None
    assert covariance["maximum_eigenvalue_m2"] is None


def test_provider_support_thresholds_use_declared_point_units() -> None:
    valid = np.array([False, False, False, False])
    audit = audit_provider_physical_mapping(
        _case(valid_mask=valid),
        _policy(
            minimum_valid_point_count=1,
            minimum_valid_fraction=0.1,
            minimum_mapped_point_count=1,
            minimum_mapped_fraction=0.1,
        ),
    )

    assert audit.provider_support_complete is False
    assert audit.query_support_sufficient is False
    assert audit.rejection_reasons[-2:] == (
        "insufficient-provider-valid-support",
        "insufficient-physical-query-overlap",
    )
    points = _diagnostics(audit)["point_accounting"]
    assert points["declared_valid_fraction"] == 0.0
    assert points["mapped_fraction_of_declared_valid"] == 0.0
    assert _diagnostics(audit)["provider_bbox_native"] is None
    assert _diagnostics(audit)["physical_bbox_m"] is None


def test_case_arrays_and_metadata_are_immutable_and_content_addressed() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [900.0, 900.0, 900.0],
        ]
    )
    metadata = {"nested": {"value": 1}}
    case = _case(points_native=points, metadata=metadata)
    points[0, 0] = 99.0
    metadata["nested"]["value"] = 2

    assert case.points_native[0, 0] == 0.0
    assert case.metadata["nested"]["value"] == 1
    with pytest.raises(ValueError):
        case.points_native.setflags(write=True)
    with pytest.raises(TypeError):
        cast(dict[str, Any], case.metadata)["new"] = True

    same = _case(metadata={"nested": {"value": 1}})
    bound = _case(
        metadata={"nested": {"value": 1}},
        artifact_id=case.artifact_id,
    )
    changed = _case(
        points_native=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [900.0, 900.0, 900.0],
            ]
        )
    )
    assert case.artifact_id == same.artifact_id
    assert case.artifact_id == bound.artifact_id
    assert case.artifact_id != changed.artifact_id
    with pytest.raises(ValueError, match="does not match"):
        _case(artifact_id=_digest("f"))


def test_policy_identity_and_validation_are_strict() -> None:
    first = _policy()
    second = _policy()
    changed = _policy(minimum_mapped_fraction=0.9)
    assert first.policy_id == second.policy_id
    assert first.policy_id != changed.policy_id

    invalid_values = (
        {"minimum_valid_point_count": True},
        {"minimum_valid_point_count": 0},
        {"minimum_mapped_fraction": 1.1},
        {"minimum_valid_fraction": -0.1},
        {"boundary_tolerance_m": -1.0},
        {"require_timestamps": 1},
        {"require_covariance": 1},
        {"maximum_covariance_condition_number": 0.9},
        {"maximum_covariance_condition_number": float("inf")},
        {"minimum_covariance_eigenvalue_m2": float("nan")},
    )
    for updates in invalid_values:
        with pytest.raises(ValueError):
            _policy(**updates)


def test_case_validation_rejects_ambiguous_or_malformed_inputs() -> None:
    invalid_cases = (
        {"case_id": " source-case-01"},
        {"provider_artifact_id": "not-a-digest"},
        {"provider_frame": ""},
        {"points_native": np.zeros((2, 2))},
        {"points_native": np.zeros((0, 3))},
        {"valid_mask": np.array([1, 1, 1, 0])},
        {"valid_mask": np.array([True, True])},
        {"provider_unit_scale_m": 0.0},
        {"provider_unit_scale_m": 1e200},
        {"provider_to_physical": np.eye(3)},
        {"provider_to_physical": np.full((4, 4), np.nan)},
        {"query_bounds_m": np.zeros((3, 3))},
        {"query_bounds_m": np.full((2, 3), np.nan)},
        {
            "query_bounds_m": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                ]
            )
        },
        {"timestamps_s": np.zeros(4), "query_time_window_s": None},
        {"timestamps_s": np.zeros(3)},
        {"query_time_window_s": np.array([1.0, 0.0])},
        {"query_time_window_s": np.array([np.nan, 1.0])},
        {"covariances_native": np.zeros((3, 3, 3))},
        {"metadata": {"bad": float("nan")}},
    )
    for updates in invalid_cases:
        with pytest.raises(ValueError):
            _case(**updates)


def test_manual_audit_validation_rejects_inconsistent_or_mutable_claims() -> None:
    valid = _manual_audit()
    same = _manual_audit(audit_id=valid.audit_id)
    assert valid.audit_id == same.audit_id
    with pytest.raises(ValueError, match="does not match"):
        _manual_audit(audit_id=_digest("f"))

    invalid_values = (
        {"case_id": " bad"},
        {"case_artifact_id": "bad"},
        {"provider_frame": ""},
        {"mapping_admissible": 1},
        {"result_reason": " bad"},
        {"rejection_reasons": "not-a-sequence"},
        {
            "mapping_admissible": False,
            "technical_valid": False,
            "result_reason": "invalid-provider-to-physical-transform",
            "rejection_reasons": [
                "invalid-provider-to-physical-transform",
                "invalid-provider-to-physical-transform",
            ],
        },
        {
            "mapping_admissible": False,
            "technical_valid": False,
            "result_reason": "unsupported",
            "rejection_reasons": ["unsupported"],
        },
        {
            "mapping_admissible": False,
            "technical_valid": False,
            "provider_support_complete": False,
            "query_support_sufficient": False,
            "result_reason": "insufficient-physical-query-overlap",
            "rejection_reasons": [
                "insufficient-physical-query-overlap",
                "insufficient-provider-valid-support",
            ],
        },
        {
            "mapping_admissible": False,
            "technical_valid": True,
            "result_reason": "invalid-provider-to-physical-transform",
            "rejection_reasons": ["invalid-provider-to-physical-transform"],
        },
        {
            "mapping_admissible": False,
            "provider_support_complete": True,
            "result_reason": "insufficient-provider-valid-support",
            "rejection_reasons": ["insufficient-provider-valid-support"],
        },
        {
            "mapping_admissible": False,
            "query_support_sufficient": True,
            "result_reason": "insufficient-physical-query-overlap",
            "rejection_reasons": ["insufficient-physical-query-overlap"],
        },
        {
            "mapping_admissible": False,
            "query_support_sufficient": False,
            "result_reason": "provider-physical-mapping-admissible",
            "rejection_reasons": ["insufficient-physical-query-overlap"],
        },
        {
            "mapping_admissible": False,
            "technical_valid": True,
            "provider_support_complete": True,
            "query_support_sufficient": True,
        },
        {"diagnostics": {"bad": float("nan")}},
    )
    for updates in invalid_values:
        with pytest.raises(ValueError):
            _manual_audit(**updates)


def test_audit_rejects_wrong_contract_types() -> None:
    with pytest.raises(TypeError, match="case must be"):
        audit_provider_physical_mapping(cast(Any, {}), _policy())
    with pytest.raises(TypeError, match="policy must be"):
        audit_provider_physical_mapping(_case(), cast(Any, {}))
