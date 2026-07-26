from dataclasses import asdict

import numpy as np

from bayesian_phystwin.bias_aware_belief import BiasAwareStateUpdateConfig
from bayesian_phystwin.deform360_bias_aware_belief_development import (
    Deform360BiasAwareDevelopmentConfig,
)
from bayesian_phystwin.phystwin_prob4d_bias_guard import (
    Prob4DBiasGuardConfig,
    build_guarded_prob4d_prefix_candidate,
)


def _source_lock() -> dict[str, object]:
    source = Deform360BiasAwareDevelopmentConfig(
        update_frames=(3,),
        minimum_available_center_count=8,
        minimum_motion_center_count=3,
        physical_response_rank=2,
        minimum_physical_response_m=0.0005,
        minimum_observed_motion_m=0.0005,
        minimum_physical_agreement_gain=0.4,
        state_update=BiasAwareStateUpdateConfig(
            observation_std_m=0.002,
            state_prior_std_m=0.05,
            shared_bias_prior_std_m=0.05,
            camera_bias_prior_std_m=0.05,
        ),
    )
    return {
        "protocol_id": "source-v4-test",
        "candidate_certified": True,
        "upper_regret_m": -1e-6,
        "config": asdict(source),
    }


def _synthetic_inputs(*, common_bias: bool = False) -> dict[str, np.ndarray]:
    frame_count = 16
    train_end = 12
    point_count = 12
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.column_stack(
        (0.10 * np.cos(angle), 0.10 * np.sin(angle), np.zeros(point_count))
    )
    mode = np.sin(2.0 * angle)
    physical = np.repeat(frame_zero[None], frame_count, axis=0)
    physical[:, :, 2] += np.linspace(0.0, 0.010, frame_count)[:, None] * mode
    selected = physical.copy().astype(np.float32)
    measurement = physical[:train_end].copy()
    truth = physical.copy()
    if common_bias:
        measurement[:, :, 0] += 0.004
    else:
        measurement[:, :, 2] += 0.003 * mode
        truth[:, :, 2] += 0.003 * mode
    covariance = np.zeros((train_end, point_count, 3, 3), dtype=np.float64)
    covariance[:] = np.eye(3) * 0.001**2
    return {
        "selected": selected,
        "physical_prefix": physical[:train_end],
        "measurement": measurement,
        "validity": np.ones((train_end, point_count), dtype=bool),
        "reliability": np.full((train_end, point_count), 0.9),
        "covariance": covariance,
        "object_points": truth[:train_end],
        "visibility": np.ones((train_end, point_count), dtype=bool),
        "motion_validity": np.ones((train_end, point_count), dtype=bool),
    }


def _run(inputs: dict[str, np.ndarray]):
    return build_guarded_prob4d_prefix_candidate(
        inputs["selected"],
        inputs["physical_prefix"],
        inputs["measurement"],
        inputs["validity"],
        inputs["reliability"],
        inputs["covariance"],
        inputs["object_points"],
        inputs["visibility"],
        inputs["motion_validity"],
        num_surface_points=inputs["selected"].shape[1],
        source_lock=_source_lock(),
        config=Prob4DBiasGuardConfig(
            fit_fraction=0.75,
            minimum_validation_frame_count=3,
            minimum_balanced_validation_improvement_fraction=0.001,
        ),
    )


def test_physical_mode_update_passes_disjoint_prefix_gate() -> None:
    inputs = _synthetic_inputs()

    report, candidate, guarded = _run(inputs)

    assert report["candidate_available"] is True
    assert report["candidate_accepted"] is True
    assert report["bit_exact_selected_baseline_fallback"] is False
    assert report["validation"]["no_primary_regression"] is True
    assert not np.array_equal(candidate, inputs["selected"])
    assert not np.array_equal(guarded, inputs["selected"])
    assert report["information_boundary"]["future_manual_track_read"] is False


def test_common_camera_bias_falls_back_bit_exactly() -> None:
    inputs = _synthetic_inputs(common_bias=True)

    report, _, guarded = _run(inputs)

    assert report["candidate_accepted"] is False
    assert report["bit_exact_selected_baseline_fallback"] is True
    assert guarded.tobytes() == inputs["selected"].tobytes()


def test_future_baseline_mutation_does_not_change_prefix_decision() -> None:
    inputs = _synthetic_inputs()
    first_report, _, _ = _run(inputs)
    mutated = {key: value.copy() for key, value in inputs.items()}
    mutated["selected"][12:] += 10.0

    second_report, _, _ = _run(mutated)

    assert first_report["candidate_accepted"] == second_report["candidate_accepted"]
    assert first_report["validation"] == second_report["validation"]


def test_sparse_unobserved_placeholders_do_not_reach_strict_update() -> None:
    inputs = _synthetic_inputs()
    inputs["validity"][:, :2] = False
    inputs["measurement"][:, :2] = np.nan
    inputs["reliability"][:, :2] = np.nan
    inputs["covariance"][:, :2] = np.nan

    report, _, _ = _run(inputs)

    assert report["candidate_available"] is True
