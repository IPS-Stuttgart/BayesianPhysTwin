from __future__ import annotations

import numpy as np

from causal4d_public.deform360_causal_transport import (
    CausalContactTransportConfig,
    causal_contact_transport_prediction,
    infer_latched_contact_schedule,
)


def _controllers(frame_count: int = 4) -> np.ndarray:
    base = np.asarray(
        [
            [-0.01, -0.001, 0.0],
            [-0.01, 0.001, 0.0],
            [-0.012, -0.001, 0.0],
            [-0.012, 0.001, 0.0],
        ]
    )
    return np.stack([base + [0.01 * frame, 0.0, 0.0] for frame in range(frame_count)])


def _config(**overrides: object) -> CausalContactTransportConfig:
    values = {
        "controller_group_size": 4,
        "maximum_contact_distance_m": 0.02,
        "opening_contact_threshold_m": 0.08,
        "confirmation_frames": 1,
        "base_support_scale_m": 0.01,
        "support_growth_per_travel": 0.0,
        "initial_contact_gain": 1.0,
        "acquired_contact_gain": 0.0,
        "transform_mode": "translation",
    }
    values.update(overrides)
    return CausalContactTransportConfig(**values)


def test_initial_contact_transports_near_points_more_than_far_points() -> None:
    initial = np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0]])
    controllers = _controllers()
    result = causal_contact_transport_prediction(
        initial,
        controllers,
        np.full((4, 1), 0.04),
        config=_config(),
    )

    displacement = np.linalg.norm(result.prediction_m[-1] - initial, axis=1)
    assert result.onset_frames == (0,)
    assert displacement[0] > 10.0 * displacement[1]
    assert displacement[0] > 0.005
    assert not result.exact_persistence


def test_acquired_contact_defaults_to_exact_persistence() -> None:
    initial = np.asarray([[0.05, 0.0, 0.0], [0.10, 0.0, 0.0]])
    controllers = _controllers(frame_count=8)
    openings = np.asarray([0.10, 0.10, 0.10, 0.07, 0.06, 0.05, 0.05, 0.05])
    result = causal_contact_transport_prediction(
        initial,
        controllers,
        openings,
        config=_config(maximum_contact_distance_m=0.05),
    )

    assert result.onset_frames[0] is not None
    assert result.onset_frames[0] > 0
    assert result.exact_persistence
    np.testing.assert_array_equal(
        result.prediction_m, np.repeat(initial[None], len(controllers), axis=0)
    )


def test_confirmation_is_causal_and_not_backdated() -> None:
    initial = np.asarray([[0.0, 0.0, 0.0]])
    schedule, _, onset = infer_latched_contact_schedule(
        initial,
        _controllers(),
        np.full((4, 1), 0.04),
        config=_config(confirmation_frames=3),
    )

    assert onset == (2,)
    np.testing.assert_array_equal(schedule[:, 0], [False, False, True, True])


def test_support_growth_expands_the_transported_region() -> None:
    initial = np.asarray([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]])
    controllers = _controllers()
    openings = np.full((4, 1), 0.04)
    fixed = causal_contact_transport_prediction(
        initial,
        controllers,
        openings,
        config=_config(base_support_scale_m=0.003),
    )
    growing = causal_contact_transport_prediction(
        initial,
        controllers,
        openings,
        config=_config(
            base_support_scale_m=0.003,
            support_growth_per_travel=2.0,
        ),
    )

    fixed_far = np.linalg.norm(fixed.prediction_m[-1, 1] - initial[1])
    growing_far = np.linalg.norm(growing.prediction_m[-1, 1] - initial[1])
    assert growing_far > 5.0 * fixed_far


def test_duplicate_correlated_grippers_do_not_double_transport() -> None:
    initial = np.asarray([[0.0, 0.0, 0.0]])
    one = _controllers()
    duplicated = np.concatenate((one, one), axis=1)
    single = causal_contact_transport_prediction(
        initial,
        one,
        np.full((4, 1), 0.04),
        config=_config(initial_contact_gain=1.0),
    )
    double = causal_contact_transport_prediction(
        initial,
        duplicated,
        np.full((4, 2), 0.04),
        config=_config(initial_contact_gain=1.0),
    )

    np.testing.assert_allclose(double.prediction_m, single.prediction_m, atol=1e-12)
