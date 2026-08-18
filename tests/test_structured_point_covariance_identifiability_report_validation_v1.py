from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from bayesian_phystwin.bias_aware_belief import (
    IdentifiableStateBasis,
    PhysicalResponseBasis,
)
from bayesian_phystwin.identifiability_report_v1 import (
    IDENTIFIABILITY_REPORT_CLAIM_BOUNDARY,
    IDENTIFIABILITY_REPORT_SCHEMA,
    IdentifiabilityReportV1,
    identifiability_report_from_bases,
)


def _physical_response() -> PhysicalResponseBasis:
    return PhysicalResponseBasis(
        basis=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ]
        ),
        singular_values_m=np.array([3.0, 2.0, 1.0]),
        explained_energy_fraction=0.9,
        supported_point_count=4,
        maximum_response_m=0.08,
    )


def _identifiable_basis() -> IdentifiableStateBasis:
    return IdentifiableStateBasis(
        query_basis=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
                [0.0, 0.5],
            ]
        ),
        observation_basis=np.array(
            [
                [0.8, 0.0],
                [0.0, 0.6],
                [0.4, 0.3],
                [0.0, 0.3],
            ]
        ),
        coefficient_transform=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        identifiable_fractions=np.array([0.8, 0.6]),
    )


def _report(**overrides: Any) -> IdentifiabilityReportV1:
    values: dict[str, Any] = {
        "physical_response_id": "3" * 64,
        "observation_mapping_id": "4" * 64,
        "bias_design_id": "5" * 64,
        "query_id": "6" * 64,
        "physical_singular_values_m": np.array([3.0, 2.0, 1.0]),
        "coefficient_transform": np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        ),
        "identifiable_fractions": np.array([0.8, 0.6]),
        "supported_point_count": 4,
        "maximum_response_m": 0.08,
        "explained_energy_fraction": 0.9,
        "minimum_identifiable_fraction_required": 0.5,
        "metadata": {"protocol": "source-only"},
    }
    values.update(overrides)
    return IdentifiabilityReportV1(**values)


@pytest.mark.parametrize(
    ("overrides", "exception", "match"),
    [
        ({"physical_response_id": "bad"}, ValueError, "lowercase"),
        (
            {"physical_singular_values_m": np.array([])},
            ValueError,
            "nonempty vector",
        ),
        (
            {"physical_singular_values_m": np.ones((3, 1))},
            ValueError,
            "nonempty vector",
        ),
        (
            {"physical_singular_values_m": np.array([3.0, 0.0, 1.0])},
            ValueError,
            "must be positive",
        ),
        (
            {"physical_singular_values_m": np.array([3.0, 1.0, 2.0])},
            ValueError,
            "nonincreasing",
        ),
        (
            {"physical_singular_values_m": np.array([3.0, np.nan, 1.0])},
            ValueError,
            "must be finite",
        ),
        (
            {"physical_singular_values_m": np.array(["x", "y", "z"])},
            ValueError,
            "real numeric",
        ),
        (
            {"coefficient_transform": np.ones(3)},
            ValueError,
            "one row per physical response mode",
        ),
        (
            {"coefficient_transform": np.ones((2, 1))},
            ValueError,
            "one row per physical response mode",
        ),
        (
            {"coefficient_transform": np.ones((3, 0))},
            ValueError,
            "between one and all",
        ),
        (
            {"coefficient_transform": np.ones((3, 4))},
            ValueError,
            "between one and all",
        ),
        (
            {"coefficient_transform": np.array([[np.nan], [0.0], [0.0]])},
            ValueError,
            "must be finite",
        ),
        (
            {"identifiable_fractions": np.ones(3)},
            ValueError,
            "one entry per retained mode",
        ),
        (
            {"identifiable_fractions": np.array([0.8, 0.0])},
            ValueError,
            r"lie in \(0, 1\]",
        ),
        (
            {"identifiable_fractions": np.array([0.8, 1.1])},
            ValueError,
            r"lie in \(0, 1\]",
        ),
        (
            {"identifiable_fractions": np.array([0.8, 0.4])},
            ValueError,
            "contradict the registered minimum",
        ),
        ({"supported_point_count": True}, ValueError, "must be an integer"),
        ({"supported_point_count": 0}, ValueError, "integer >= 1"),
        ({"maximum_response_m": 0.0}, ValueError, "must be > 0.0"),
        ({"maximum_response_m": np.inf}, ValueError, "finite real"),
        ({"explained_energy_fraction": 0.0}, ValueError, "must be > 0.0"),
        ({"explained_energy_fraction": 1.1}, ValueError, "must be <= 1.0"),
        (
            {"minimum_identifiable_fraction_required": True},
            ValueError,
            "finite real",
        ),
        (
            {"minimum_identifiable_fraction_required": 0.0},
            ValueError,
            "must be > 0.0",
        ),
        (
            {"minimum_identifiable_fraction_required": 1.1},
            ValueError,
            "must be <= 1.0",
        ),
        ({"metadata": 1}, ValueError, "metadata"),
        ({"artifact_id": "bad"}, ValueError, "lowercase"),
    ],
)
def test_identifiability_report_rejects_invalid_contract_inputs(
    overrides: dict[str, Any],
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        _report(**overrides)


def test_identifiability_builder_rejects_wrong_or_misaligned_bases() -> None:
    kwargs = {
        "physical_response_id": "3" * 64,
        "observation_mapping_id": "4" * 64,
        "bias_design_id": "5" * 64,
        "query_id": "6" * 64,
        "minimum_identifiable_fraction_required": 0.5,
    }
    with pytest.raises(TypeError, match="PhysicalResponseBasis"):
        identifiability_report_from_bases(
            cast(Any, object()),
            _identifiable_basis(),
            **kwargs,
        )
    with pytest.raises(TypeError, match="IdentifiableStateBasis"):
        identifiability_report_from_bases(
            _physical_response(),
            cast(Any, object()),
            **kwargs,
        )

    mismatched = IdentifiableStateBasis(
        query_basis=np.ones((4, 1)),
        observation_basis=np.ones((4, 1)),
        coefficient_transform=np.ones((2, 1)),
        identifiable_fractions=np.ones(1),
    )
    with pytest.raises(ValueError, match="does not match physical modes"):
        identifiability_report_from_bases(
            _physical_response(),
            mismatched,
            **kwargs,
        )
