import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin._portable_contracts import content_id
from bayesian_phystwin.matphys_native_source_v1 import (
    NativeMatPhysCaseEvidence,
    calibrated_covariance,
    frame_zero_farthest_point_indices,
    gaussian_case_metrics,
    native_case_evidence,
    select_candidate_calibration,
    select_isotropic_calibration,
)


def _scorer():
    path = (
        Path(__file__).parents[1]
        / "scripts"
        / "science"
        / "score_matphys_native_phystwin_source_v1.py"
    )
    spec = importlib.util.spec_from_file_location("matphys_native_scorer_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_zero_farthest_points_are_deterministic_and_unique() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.05, 0.05, 0.0]]
    )

    selected = frame_zero_farthest_point_indices(points, count=3)

    np.testing.assert_array_equal(selected, [0, 1, 2])


def test_baseline_relative_orientation_can_beat_isotropic_at_lower_volume() -> None:
    residual = np.tile(np.array([[0.01, 0.0, 0.0]]), (20, 1))
    raw = np.broadcast_to(
        np.diag([0.01**2, 0.001**2, 0.001**2]), (len(residual), 3, 3)
    ).copy()
    candidate = gaussian_case_metrics(
        residual,
        calibrated_covariance(raw, scale=1.0, isotropic_std_m=0.0005),
    )
    isotropic = gaussian_case_metrics(
        residual,
        np.broadcast_to(np.eye(3) * 0.01**2, (len(residual), 3, 3)),
    )

    assert candidate["nll_nats_per_event"] < isotropic["nll_nats_per_event"]
    assert candidate["mean_ellipsoid_volume_m3"] < isotropic["mean_ellipsoid_volume_m3"]


@pytest.mark.parametrize(
    "invalid_mask",
    (
        np.ones((2, 2), dtype=np.int64),
        np.array([[1.0, 0.0], [np.nan, 1.0]], dtype=np.float64),
    ),
)
def test_native_case_evidence_rejects_non_boolean_validity_masks(
    invalid_mask: np.ndarray,
) -> None:
    observed = np.zeros((2, 2, 3), dtype=np.float64)
    covariance = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()

    with pytest.raises(ValueError, match="valid_mask must have boolean dtype"):
        native_case_evidence(
            case_id="mask-contract",
            observed_m=observed,
            baseline_mean_m=observed,
            valid_mask=invalid_mask,
            raw_covariance_m2=covariance,
            future_start=1,
            future_stop=2,
        )


def test_source_calibration_selects_anisotropic_donor_and_isotropic_scale() -> None:
    residual = np.tile(np.array([[0.01, 0.0, 0.0]]), (12, 1))
    raw = np.broadcast_to(
        np.diag([0.01**2, 0.0005**2, 0.0005**2]), (len(residual), 3, 3)
    ).copy()
    cases = (
        NativeMatPhysCaseEvidence("a", residual, raw),
        NativeMatPhysCaseEvidence("b", residual, raw),
    )

    assert select_candidate_calibration(
        cases,
        scale_grid=(0.0, 1.0),
        isotropic_std_grid_m=(0.0005, 0.01),
    ) == (1.0, 0.0005)
    assert (
        select_isotropic_calibration(cases, isotropic_std_grid_m=(0.0005, 0.01)) == 0.01
    )


def test_source_scorer_rejects_mutated_content_identity() -> None:
    scorer = _scorer()
    identity = {
        "schema": "synthetic",
        "schema_version": 1,
        "value": 7,
    }
    record = {**identity, "prediction_id": content_id(identity)}

    scorer._validate_content_id(
        record, id_field="prediction_id", name="synthetic prediction"
    )
    record["value"] = 8
    with pytest.raises(ValueError, match="content identity changed"):
        scorer._validate_content_id(
            record, id_field="prediction_id", name="synthetic prediction"
        )
