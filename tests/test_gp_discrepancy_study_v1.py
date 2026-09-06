"""Recording-level information-boundary and end-to-end study regressions."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin_experiments.gp_discrepancy_study_v1 import run_study, synthetic_data


def protocol():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/diagnostics/gp_discrepancy_development_v1.json").read_text())


def test_future_score_outcomes_cannot_change_fitting_or_predictions():
    data = synthetic_data(7)
    first, arrays = run_study(data, protocol())
    changed_truth = data.truth.copy()
    changed_truth[data.split == "score"] += 10.0
    second, other = run_study(replace(data, truth=changed_truth), protocol())
    for key in ("prediction_sha256", "selected_config", "gp_mean_accepted_on_select",
                "covariance_scale_from_calibration", "contrast_threshold_from_calibration_m"):
        assert first[key] == second[key]
    np.testing.assert_array_equal(arrays["selected_predictions"], other["selected_predictions"])
    assert second["point_summary"]["selected_mean_l1_m"] > 9.0


def test_same_mean_controls_have_identical_marginal_width_and_coverage():
    result, _ = run_study(synthetic_data(17), protocol())
    full = result["covariance_summary"]["gp_scaled"]
    for control in ("gp_frame_block", "gp_temporal_sign_control"):
        other = result["covariance_summary"][control]
        assert full["mean_full_90_width"] == other["mean_full_90_width"]
        assert full["marginal_90_coverage"] == other["marginal_90_coverage"]


def test_clamped_points_are_exact_even_after_gp_correction():
    data = synthetic_data(29)
    _, arrays = run_study(data, protocol())
    clamped = np.all(data.basis == 0, axis=1)
    np.testing.assert_array_equal(arrays["selected_predictions"][:, :, clamped], arrays["baseline_predictions"][:, :, clamped])


def test_rotated_canonical_frames_preserve_mode_uncertainty():
    data = synthetic_data(43)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = replace(data, baseline=data.baseline @ rotation.T, ridge=data.ridge @ rotation.T,
                      truth=data.truth @ rotation.T, frames=rotation[None] @ data.frames)
    first, arrays = run_study(data, protocol())
    second, other = run_study(rotated, protocol())
    np.testing.assert_allclose(other["selected_predictions"], arrays["selected_predictions"] @ rotation.T, atol=1e-12)
    assert first["covariance_summary"]["gp_scaled"]["nll_per_dimension"] == pytest.approx(second["covariance_summary"]["gp_scaled"]["nll_per_dimension"], abs=1e-10)


def test_repeated_execution_is_reproducible():
    first, _ = run_study(synthetic_data(71), protocol())
    second, _ = run_study(synthetic_data(71), protocol())
    assert first == second


def test_source_positive_does_not_imply_shifted_regime_calibration():
    result, _ = run_study(synthetic_data(7, adverse=True), protocol())
    # This is a deliberately adverse implementation fixture, not a fitted gate.
    assert result["covariance_summary"]["gp_scaled"]["marginal_90_coverage"] < 0.75
    assert result["real_improvement_established"] is False
    assert result["official_evaluation_opened"] is False


@pytest.mark.parametrize("field", ["ids", "split", "frames", "truth"])
def test_invalid_study_inputs_fail_closed(field):
    data = synthetic_data(7)
    if field == "ids":
        value = np.repeat("same-recording", len(data.ids))
    elif field == "split":
        value = np.repeat("fit", len(data.split))
    else:
        value = getattr(data, field).copy()
        value.flat[0] = np.nan
    with pytest.raises(ValueError):
        replace(data, **{field: value}).validate()


def test_promotion_labels_are_forbidden():
    with pytest.raises(ValueError, match="confirmatory"):
        replace(synthetic_data(7), kind="fresh-confirmation").validate()
