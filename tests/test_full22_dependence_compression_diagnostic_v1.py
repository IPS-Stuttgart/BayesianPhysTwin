from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "science" / "analyze_full22_dependence_compression_v1.py"
PROTOCOL = ROOT / "protocols" / "full22_dependence_compression_diagnostic_v1.json"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "analyze_full22_dependence_compression_v1",
        SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_script()


def test_protocol_is_content_addressed_target_free_and_block_scoped() -> None:
    protocol = MODULE.load_protocol(PROTOCOL)

    assert protocol["status"] == "frozen-before-covariance-value-read"
    assert protocol["source"]["case_count"] == 22
    assert protocol["source"]["candidate_id"] == "independent_endpoint_v1"
    assert protocol["analysis"]["dense_cross_track_covariance_available"] is False
    assert protocol["information_boundary"]["future_outcomes_required"] is False
    assert (
        protocol["proceed_gate"]["outcome_comparison_authorized_by_this_diagnostic"]
        is False
    )


def test_total_correlation_matches_closed_form_for_correlated_pair() -> None:
    rho = 0.6
    covariance = np.asarray([[4.0, 1.2], [1.2, 1.0]], dtype=np.float64)

    observed = MODULE.total_correlation_nats(covariance)

    assert float(observed) == pytest.approx(-0.5 * math.log(1.0 - rho**2))


def test_total_correlation_is_invariant_to_coordinate_rescaling() -> None:
    covariance = np.asarray(
        [[2.0, 0.4, -0.2], [0.4, 1.0, 0.1], [-0.2, 0.1, 0.5]],
        dtype=np.float64,
    )
    scale = np.diag([3.0, 0.5, 7.0])

    first = MODULE.total_correlation_nats(covariance)
    second = MODULE.total_correlation_nats(scale @ covariance @ scale)

    assert float(second) == pytest.approx(float(first), abs=1e-14)


def test_rank1_diagnostic_is_exact_for_isotropic_plus_one_factor() -> None:
    factor = np.asarray([0.3, -0.2, 0.4], dtype=np.float64)
    covariance = 0.7 * np.eye(3) + np.outer(factor, factor)

    reconstructed = MODULE._rank1_marginal_reconstruction(covariance)

    np.testing.assert_allclose(reconstructed, covariance, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(
        np.diag(reconstructed),
        np.diag(covariance),
        atol=2e-16,
        rtol=0.0,
    )


def test_case_analysis_uses_frozen_horizon_scales_and_common_noise() -> None:
    covariance = np.broadcast_to(
        np.asarray(
            [[1.0, 0.7, 0.0], [0.7, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        (6, 2, 3, 3),
    ).copy()

    result = MODULE.analyze_case_covariance(
        covariance,
        scales=(1.0, 2.0, 4.0),
        observation_std_m=0.5,
    )

    assert result["block_count"] == 12
    assert result["future_frame_count"] == 6
    assert result["track_count"] == 2
    assert result["mean_total_correlation_nats"] > (
        result["mean_raw_total_correlation_nats"]
    )
    assert result["rank1_relative_total_correlation_error"] == pytest.approx(0.0)


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "covariance",
    [
        np.eye(3, dtype=np.float64),
        np.zeros((2, 3, 2, 2), dtype=np.float64),
        np.full((2, 3, 3, 3), np.nan, dtype=np.float64),
    ],
)
def test_case_analysis_rejects_wrong_or_nonpositive_covariance(
    covariance: NDArray[np.float64],
) -> None:
    with pytest.raises(ValueError):
        MODULE.analyze_case_covariance(
            covariance,
            scales=(1.0, 2.0, 4.0),
            observation_std_m=0.005,
        )
