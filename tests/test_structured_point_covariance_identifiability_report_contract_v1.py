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


def test_identifiability_report_from_existing_bases_is_content_addressed() -> None:
    report = identifiability_report_from_bases(
        _physical_response(),
        _identifiable_basis(),
        physical_response_id="3" * 64,
        observation_mapping_id="4" * 64,
        bias_design_id="5" * 64,
        query_id="6" * 64,
        minimum_identifiable_fraction_required=0.5,
        metadata={"protocol": "source-only"},
    )
    duplicate = _report(artifact_id=report.artifact_id)

    assert duplicate.artifact_id == report.artifact_id
    assert report.physical_mode_count == 3
    assert report.retained_mode_count == 2
    assert report.discarded_mode_count == 1
    assert report.retained_mode_fraction == pytest.approx(2.0 / 3.0)
    assert report.discarded_mode_fraction == pytest.approx(1.0 / 3.0)
    np.testing.assert_allclose(report.state_bias_overlap_fractions, [0.6, 0.8])
    assert report.minimum_identifiable_fraction_observed == pytest.approx(0.6)
    assert report.mean_identifiable_fraction == pytest.approx(0.7)
    assert report.maximum_state_bias_overlap == pytest.approx(0.8)
    assert report.summary()["schema"] == IDENTIFIABILITY_REPORT_SCHEMA
    assert report.summary()["claim_boundary"] == IDENTIFIABILITY_REPORT_CLAIM_BOUNDARY
    assert report.to_record()["derived"]["retained_mode_count"] == 2
    assert set(report.arrays()) == {
        "physical_singular_values_m",
        "coefficient_transform",
        "identifiable_fractions",
    }
    assert report.physical_singular_values_m.flags.writeable is False
    assert report.coefficient_transform.flags.writeable is False
    assert report.identifiable_fractions.flags.writeable is False
    assert report.state_bias_overlap_fractions.flags.writeable is False
    with pytest.raises(TypeError):
        report.metadata["protocol"] = "tampered"  # type: ignore[index]
    with pytest.raises(ValueError):
        report.identifiable_fractions.setflags(write=True)


def test_identifiability_report_copies_inputs_and_binds_every_identity() -> None:
    singular_values = np.array([3.0, 2.0, 1.0])
    transform = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    fractions = np.array([0.8, 0.6])
    report = _report(
        physical_singular_values_m=singular_values,
        coefficient_transform=transform,
        identifiable_fractions=fractions,
    )
    changed_query = _report(query_id="7" * 64)
    changed_fraction = _report(identifiable_fractions=np.array([0.8, 0.7]))
    changed_metadata = _report(metadata={"protocol": "other"})

    singular_values[:] = 1.0
    transform[:] = 0.0
    fractions[:] = 1.0
    np.testing.assert_array_equal(report.physical_singular_values_m, [3.0, 2.0, 1.0])
    np.testing.assert_array_equal(
        report.coefficient_transform,
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
    )
    np.testing.assert_array_equal(report.identifiable_fractions, [0.8, 0.6])
    assert report.artifact_id != changed_query.artifact_id
    assert report.artifact_id != changed_fraction.artifact_id
    assert report.artifact_id != changed_metadata.artifact_id
    with pytest.raises(ValueError, match="does not match content"):
        _report(artifact_id="0" * 64)


