import numpy as np

from bayesian_phystwin.bias_aware_belief import BiasAwareStateUpdateConfig
from bayesian_phystwin.deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
    predict_bias_aware_candidate_arrays,
)


def _synthetic_inputs() -> dict[str, np.ndarray]:
    frame_count = 8
    point_count = 12
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.column_stack(
        (0.10 * np.cos(angle), 0.10 * np.sin(angle), np.zeros(point_count))
    )
    baseline = np.repeat(frame_zero[None], frame_count, axis=0).astype(np.float32)
    local_mode = np.sin(2.0 * angle)
    response = np.zeros_like(baseline)
    response[:, :, 2] = np.linspace(0.0, 0.006, frame_count)[:, None] * local_mode
    measurement = np.full_like(baseline, np.nan)
    visibility = np.zeros((frame_count, point_count), dtype=bool)
    validity = np.zeros_like(visibility)
    measurement[0] = frame_zero
    visibility[0] = True
    validity[0] = True
    for frame in (3, 6):
        measurement[frame] = baseline[frame]
        measurement[frame, :, 0] += 0.010
        local_amplitude = 0.004 if frame == 3 else 0.006
        measurement[frame, :, 2] += local_amplitude * local_mode
        visibility[frame] = True
        validity[frame] = True
    return {
        "baseline": baseline,
        "response": response,
        "frame_zero": frame_zero,
        "action_support": np.ones(point_count),
        "measurement": measurement,
        "visibility": visibility,
        "validity": validity,
        "center_ids": np.arange(point_count),
        "reliability": np.ones((2, point_count)),
        "variance": np.full((2, point_count), 0.002**2),
    }


def _config() -> Deform360BiasAwareDevelopmentConfig:
    return Deform360BiasAwareDevelopmentConfig(
        update_frames=(3, 6),
        minimum_available_center_count=8,
        minimum_motion_center_count=3,
        physical_response_rank=2,
        minimum_physical_response_m=0.0005,
        minimum_observed_motion_m=0.0005,
        state_update=BiasAwareStateUpdateConfig(
            observation_std_m=0.002,
            state_prior_std_m=0.05,
            shared_bias_prior_std_m=0.05,
            camera_bias_prior_std_m=0.05,
        ),
    )


def test_target_free_candidate_recovers_local_state_not_global_bias() -> None:
    inputs = _synthetic_inputs()

    report, candidate = predict_bias_aware_candidate_arrays(
        inputs["baseline"],
        inputs["response"],
        inputs["frame_zero"],
        inputs["action_support"],
        inputs["measurement"],
        inputs["visibility"],
        inputs["validity"],
        center_ids=inputs["center_ids"],
        prior_reliability=inputs["reliability"],
        observation_variance_m2=inputs["variance"],
        config=_config(),
    )

    assert report["candidate_update_count"] == 2
    assert all(update["candidate_available"] for update in report["updates"])
    correction = candidate[4] - inputs["baseline"][4]
    assert np.sqrt(np.mean(np.square(correction[:, 2]))) > 0.002
    assert np.max(np.abs(correction[:, 0])) < 0.001
    assert report["information_boundary"]["target_argument_accepted"] is False


def test_actionless_window_is_bit_exact_baseline_fallback() -> None:
    inputs = _synthetic_inputs()
    inputs["response"][:] = 0.0

    report, candidate = predict_bias_aware_candidate_arrays(
        inputs["baseline"],
        inputs["response"],
        inputs["frame_zero"],
        inputs["action_support"],
        inputs["measurement"],
        inputs["visibility"],
        inputs["validity"],
        center_ids=inputs["center_ids"],
        prior_reliability=inputs["reliability"],
        observation_variance_m2=inputs["variance"],
        config=_config(),
    )

    assert report["candidate_update_count"] == 0
    assert candidate.tobytes() == inputs["baseline"].tobytes()
    assert all(update["bit_exact_baseline_fallback"] for update in report["updates"])
