from __future__ import annotations

import numpy as np
import pytest

from bayesian_phystwin.deform360_adaptive_covariance_rbf import (
    FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
    normalized_covariance_dispersion,
    predict_adaptive_covariance_selected_backbone_rbf,
)
from bayesian_phystwin.deform360_held_online_prefix import (
    FRAME_COUNT,
    HELD_RBF_CONFIG,
    UPDATE_FRAMES,
    predict_support_gated_selected_backbone_rbf,
)


def _inputs() -> dict[str, object]:
    grid = np.stack(
        np.meshgrid(
            np.linspace(0.0, 0.03, 4),
            np.linspace(0.0, 0.03, 4),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)
    frame_zero = np.column_stack((grid, np.zeros(len(grid)))).astype(np.float32)
    physical = np.repeat(frame_zero[None], FRAME_COUNT, axis=0)
    physical[:, :, 0] += np.arange(FRAME_COUNT, dtype=np.float32)[:, None] * 2e-4
    persistence = np.repeat(frame_zero[None], FRAME_COUNT, axis=0)
    centers = np.arange(len(frame_zero), dtype=np.int64)
    measurements: dict[int, np.ndarray] = {}
    validity: dict[int, np.ndarray] = {}
    covariance: dict[int, np.ndarray] = {}
    covariance_validity: dict[int, np.ndarray] = {}
    for budget in (4, 8):
        observed = np.full_like(physical, np.nan)
        observed_validity = np.zeros(physical.shape[:2], dtype=bool)
        for update in UPDATE_FRAMES:
            observed[update, centers] = physical[update, centers]
            observed[update, centers, 2] += 0.001
            observed_validity[update, centers] = True
        measurements[budget] = observed
        validity[budget] = observed_validity
        covariance[budget] = np.zeros((*physical.shape[:2], 3, 3), dtype=float)
        covariance[budget][observed_validity] = np.eye(3) * 1e-12
        covariance_validity[budget] = observed_validity.copy()
    return {
        "physical": physical,
        "persistence": persistence,
        "centers": centers,
        "selected_cameras": {
            4: tuple(f"camera-{index:02d}" for index in range(4)),
            8: tuple(f"camera-{index:02d}" for index in range(8)),
        },
        "measurements": measurements,
        "validity": validity,
        "covariance": covariance,
        "covariance_validity": covariance_validity,
    }


def _predict(inputs: dict[str, object]):
    return predict_adaptive_covariance_selected_backbone_rbf(
        inputs["physical"],
        inputs["persistence"],
        inputs["selected_cameras"],
        inputs["measurements"],
        inputs["validity"],
        inputs["covariance"],
        inputs["covariance_validity"],
        center_ids=inputs["centers"],
        config=FROZEN_ADAPTIVE_COVARIANCE_CONFIG,
        rbf_config=HELD_RBF_CONFIG,
    )


def test_normalized_covariance_dispersion_uses_radial_p90_and_bbox() -> None:
    inputs = _inputs()
    physical = np.asarray(inputs["physical"])
    centers = np.asarray(inputs["centers"])
    covariance = np.asarray(inputs["covariance"][4])
    validity = np.asarray(inputs["covariance_validity"][4])
    diagonal = float(
        np.linalg.norm(np.max(physical[0], axis=0) - np.min(physical[0], axis=0))
    )
    radial_standard_deviation = 0.005 * diagonal
    covariance[UPDATE_FRAMES[0], centers] = np.eye(3) * (
        radial_standard_deviation**2 / 3.0
    )

    result = normalized_covariance_dispersion(
        covariance,
        validity,
        centers,
        UPDATE_FRAMES[0],
        physical[0],
    )

    assert result["valid_covariance_center_count"] == 16
    assert result["radial_standard_deviation_quantile_m"] == pytest.approx(
        radial_standard_deviation
    )
    assert result["normalized_covariance_dispersion"] == pytest.approx(0.005)
    assert result["probabilistic_calibration_claimed"] is False


def test_reliable_four_view_route_matches_frozen_single_budget_predictor() -> None:
    inputs = _inputs()
    prediction, selected_raw, diagnostic = _predict(inputs)
    reference, reference_raw, _ = predict_support_gated_selected_backbone_rbf(
        inputs["physical"],
        inputs["persistence"],
        inputs["measurements"][4],
        inputs["validity"][4],
        center_ids=inputs["centers"],
        rbf_config=HELD_RBF_CONFIG,
    )

    assert np.array_equal(prediction, reference)
    assert np.array_equal(selected_raw, reference_raw)
    assert [record["route"] for record in diagnostic["updates"]] == [
        "4_view_rbf"
    ] * len(UPDATE_FRAMES)
    assert [record["tracked_camera_count"] for record in diagnostic["updates"]] == [
        4
    ] * len(UPDATE_FRAMES)


def test_unreliable_four_view_routes_to_eight_views() -> None:
    inputs = _inputs()
    four_validity = np.asarray(inputs["covariance_validity"][4])
    four_validity[:] = False

    prediction, selected_raw, diagnostic = _predict(inputs)
    reference, reference_raw, _ = predict_support_gated_selected_backbone_rbf(
        inputs["physical"],
        inputs["persistence"],
        inputs["measurements"][8],
        inputs["validity"][8],
        center_ids=inputs["centers"],
        rbf_config=HELD_RBF_CONFIG,
    )

    assert np.array_equal(prediction, reference)
    assert np.array_equal(selected_raw, reference_raw)
    assert [record["route"] for record in diagnostic["updates"]] == [
        "8_view_rbf"
    ] * len(UPDATE_FRAMES)
    assert all(
        record["budget_diagnostics"]["4"]["reliable"] is False
        and record["budget_diagnostics"]["8"]["reliable"] is True
        for record in diagnostic["updates"]
    )


def test_double_rejection_is_bit_exact_physical_and_skips_state_update() -> None:
    inputs = _inputs()
    for budget in (4, 8):
        np.asarray(inputs["covariance_validity"][budget])[:] = False

    prediction, selected_raw, diagnostic = _predict(inputs)

    assert np.array_equal(prediction, inputs["physical"])
    assert np.array_equal(selected_raw, inputs["physical"])
    assert all(
        record["route"] == "physical_prior_fallback"
        and record["tracked_camera_count"] == 8
        and record["rbf_correction_applied"] is False
        and record["state_updated"] is False
        for record in diagnostic["updates"]
    )


def test_accepted_fallback_accepted_sequence_has_exact_fallback_interval() -> None:
    inputs = _inputs()
    middle_update = UPDATE_FRAMES[1]
    for budget in (4, 8):
        np.asarray(inputs["covariance_validity"][budget])[middle_update] = False

    prediction, selected_raw, diagnostic = _predict(inputs)

    assert [record["route"] for record in diagnostic["updates"]] == [
        "4_view_rbf",
        "physical_prior_fallback",
        "4_view_rbf",
    ]
    fallback_slice = slice(middle_update + 1, UPDATE_FRAMES[2])
    np.testing.assert_array_equal(
        prediction[fallback_slice],
        np.asarray(inputs["physical"])[fallback_slice],
    )
    np.testing.assert_array_equal(
        selected_raw[fallback_slice],
        np.asarray(inputs["physical"])[fallback_slice],
    )
    assert diagnostic["updates"][1]["state_updated"] is False


@pytest.mark.parametrize("route_to_eight", [False, True])
def test_distinct_tracked_camera_count_handles_nonnested_plans(
    route_to_eight: bool,
) -> None:
    inputs = _inputs()
    inputs["selected_cameras"] = {
        4: ("camera-00", "camera-01", "camera-02", "camera-extra"),
        8: tuple(f"camera-{index:02d}" for index in range(8)),
    }
    np.asarray(inputs["covariance_validity"][4])[:] = False
    if not route_to_eight:
        np.asarray(inputs["covariance_validity"][8])[:] = False

    _, _, diagnostic = _predict(inputs)

    expected_route = "8_view_rbf" if route_to_eight else "physical_prior_fallback"
    assert all(
        record["route"] == expected_route
        and record["tracked_camera_count"] == 9
        and record["tracked_cameras"]
        == [
            "camera-00",
            "camera-01",
            "camera-02",
            "camera-03",
            "camera-04",
            "camera-05",
            "camera-06",
            "camera-07",
            "camera-extra",
        ]
        for record in diagnostic["updates"]
    )


def test_predictor_is_deterministic_and_does_not_mutate_inputs() -> None:
    inputs = _inputs()
    snapshots = {
        "physical": np.asarray(inputs["physical"]).copy(),
        "persistence": np.asarray(inputs["persistence"]).copy(),
        **{
            f"{label}-{budget}": np.asarray(inputs[label][budget]).copy()
            for label in (
                "measurements",
                "validity",
                "covariance",
                "covariance_validity",
            )
            for budget in (4, 8)
        },
    }

    first = _predict(inputs)
    second = _predict(inputs)

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert np.array_equal(inputs["physical"], snapshots["physical"])
    assert np.array_equal(inputs["persistence"], snapshots["persistence"])
    for label in (
        "measurements",
        "validity",
        "covariance",
        "covariance_validity",
    ):
        for budget in (4, 8):
            assert np.array_equal(
                inputs[label][budget],
                snapshots[f"{label}-{budget}"],
                equal_nan=True,
            )
