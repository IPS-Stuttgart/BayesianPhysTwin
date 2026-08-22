from __future__ import annotations

import json
import math
from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.provider_physical_mapping_audit_v1 import (
    ProviderPhysicalMappingAuditV1,
    ProviderPhysicalMappingCaseV1,
    ProviderPhysicalMappingPolicyV1,
    audit_provider_physical_mapping,
)


def _digest(character: str) -> str:
    return character * 64


def _case(
    *,
    provider_unit_scale_m: float,
    query_bounds_m: np.ndarray,
) -> ProviderPhysicalMappingCaseV1:
    return ProviderPhysicalMappingCaseV1(
        case_id="numerical-guard-source-case",
        provider_artifact_id=_digest("0"),
        physical_query_id=_digest("1"),
        mapping_protocol_id=_digest("2"),
        provider_frame="camera_native",
        physical_frame="robot_world",
        points_native=np.zeros((1, 3)),
        valid_mask=np.array([True]),
        provider_unit_scale_m=provider_unit_scale_m,
        provider_to_physical=np.eye(4),
        query_bounds_m=query_bounds_m,
        covariances_native=np.eye(3)[None],
        metadata={"information_split": "source-only"},
    )


def _policy(**updates: Any) -> ProviderPhysicalMappingPolicyV1:
    values: dict[str, Any] = {
        "minimum_valid_point_count": 1,
        "minimum_valid_fraction": 1.0,
        "minimum_mapped_point_count": 1,
        "minimum_mapped_fraction": 1.0,
        "require_covariance": True,
    }
    values.update(updates)
    return ProviderPhysicalMappingPolicyV1(**values)


def _diagnostics(audit: ProviderPhysicalMappingAuditV1) -> dict[str, Any]:
    return cast(dict[str, Any], audit.to_dict()["diagnostics"])


def test_unit_scale_square_cannot_underflow_to_zero() -> None:
    minimum_safe = math.sqrt(float(np.finfo(float).tiny))
    bounds = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    with pytest.raises(ValueError, match="provider_unit_scale_m must be at least"):
        _case(
            provider_unit_scale_m=math.nextafter(minimum_safe, 0.0),
            query_bounds_m=bounds,
        )

    audit = audit_provider_physical_mapping(
        _case(
            provider_unit_scale_m=minimum_safe,
            query_bounds_m=bounds,
        ),
        _policy(),
    )

    assert audit.mapping_admissible is True
    covariance = _diagnostics(audit)["covariance"]
    scale_squared = covariance["provider_unit_scale_squared_m2"]
    assert scale_squared == pytest.approx(float(np.finfo(float).tiny))
    assert scale_squared > 0.0


def test_tolerance_expansion_overflow_is_a_technical_rejection() -> None:
    maximum = float(np.finfo(float).max)
    audit = audit_provider_physical_mapping(
        _case(
            provider_unit_scale_m=1.0,
            query_bounds_m=np.array(
                [
                    [-maximum, -maximum, -maximum],
                    [maximum, maximum, maximum],
                ]
            ),
        ),
        _policy(boundary_tolerance_m=maximum),
    )

    assert audit.mapping_admissible is False
    assert audit.technical_valid is False
    assert audit.query_support_sufficient is False
    assert audit.result_reason == "nonfinite-effective-physical-query-bounds"
    assert audit.rejection_reasons == (
        "nonfinite-effective-physical-query-bounds",
        "insufficient-physical-query-overlap",
    )
    bounds = _diagnostics(audit)["query_bounds_m"]
    assert bounds["effective_finite"] is False
    assert bounds["effective_lower"] == [None, None, None]
    assert bounds["effective_upper"] == [None, None, None]
    json.dumps(audit.to_dict(), sort_keys=True, allow_nan=False)
