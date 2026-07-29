from __future__ import annotations

import numpy as np

from bayesian_phystwin.bias_aware_belief import BiasAwareStateUpdateConfig
from bayesian_phystwin.deform360_pairwise_bias_aware_development import (
    PairwiseBiasAwareDevelopmentConfig,
    predict_pairwise_bias_aware_candidate_arrays,
)


def _synthetic_inputs() -> dict[str, np.ndarray]:
    frame_count = 7
    point_count = 16
    angle = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    frame_zero = np.column_stack(
        (0.10 * np.cos(angle), 0.10 * np.sin(angle), np.zeros(point_count))
    )
    local_mode = np.sin(2.0 * angle)
    baseline = np.repeat(frame_zero[None], frame_count, axis=0).astype(np.float32)
    response = np.zeros_like(baseline)
    response[:, :, 2] = np.linspace(0.0, 0.006, frame_count)[:, None] * local_mode
    measurement = np.full_like(baseline, np.nan)
    visibility = np.zeros((frame_count, point_count), dtype=bool)
    validity = np.zeros_like(visibility)
    measurement[0] = frame_zero
    visibility[0] = True
    validity[0] = True
    update = 3
    measurement[update] = baseline[update]
    measurement[update, :, 0] += 0.010
    measurement[update, :, 2] += response[update, :, 2]
    corrupt = np.asarray([1, 6, 11])
    measurement[update, corrupt] += np.asarray(
        [[0.14, 0.08, 0.03], [-0.12, 0.11, -0.04], [0.09, -0.15, 0.05]]
    )
    visibility[update] = True
    validity[update] = True
    return {
        "baseline": baseline,
        "response": response,
        "frame_zero": frame_zero,
        "action_support": np.ones(point_count),
        "measurement": measurement,
        "visibility": visibility,
        "validity": validity,
        "center_ids": np.arange(point_count),
        "reliability": np.ones((1, point_count)),
        "variance": np.full((1, point_count), 0.002**2),
        "local_mode": local_mode,
        "corrupt": corrupt,
    }


def _config() -> PairwiseBiasAwareDevelopmentConfig:
    return PairwiseBiasAwareDevelopmentConfig(
        update_frames=(3,),
        selected_center_count=12,
        physical_response_rank=2,
        minimum_motion_center_count=3,
        minimum_physical_response_m=0.0005,
        minimum_observed_motion_m=0.0005,
        minimum_physical_agreement_gain=0.40,
        information_effective_sample_cap=8.0,
        state_update=BiasAwareStateUpdateConfig(
            observation_std_m=0.002,
            state_prior_std_m=0.05,
            shared_bias_prior_std_m=0.05,
            camera_bias_prior_std_m=0.05,
        ),
    )


def _predict(
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray]:
    return predict_pairwise_bias_aware_candidate_arrays(
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


def test_pairwise_bias_aware_update_rejects_swaps_and_common_bias() -> None:
    inputs = _synthetic_inputs()

    report, candidate = _predict(inputs)

    assert report["candidate_update_count"] == 1
    update = report["updates"][0]
    assert update["pairwise_gate"]["accepted"] is True
    assert update["pairwise_gate"]["inlier_count"] == 13
    assert not set(update["selected_center_ids"]) & set(inputs["corrupt"].tolist())
    assert len(update["selected_center_ids"]) == 12
    assert update["effective_reliability_sum"] <= 8.0 + 1e-12
    correction = candidate[4] - inputs["baseline"][4]
    assert np.sqrt(np.mean(np.square(correction[:, 2]))) > 0.001
    assert np.max(np.abs(correction[:, 0])) < 0.001


def test_common_bias_without_physical_response_is_exact_fallback() -> None:
    inputs = _synthetic_inputs()
    update = 3
    inputs["measurement"][update] = inputs["baseline"][update]
    inputs["measurement"][update, :, 0] += 0.010

    report, candidate = _predict(inputs)

    assert report["candidate_update_count"] == 0
    assert candidate.tobytes() == inputs["baseline"].tobytes()
    assert report["updates"][0]["bit_exact_baseline_fallback"] is True
    assert report["updates"][0]["causal_physical_agreement_gain"] == 0.0


def test_global_innovation_does_not_change_prior_selection_reliability() -> None:
    inputs = _synthetic_inputs()
    first_report, _ = _predict(inputs)
    shifted = _synthetic_inputs()
    shifted["measurement"][3, :, 0] += 0.020
    second_report, _ = _predict(shifted)

    first_update = first_report["updates"][0]
    second_update = second_report["updates"][0]
    assert first_update["selected_center_ids"] == second_update["selected_center_ids"]
    assert (
        first_update["effective_reliability_sum"]
        == second_update["effective_reliability_sum"]
    )
    assert (
        first_report["information_boundary"][
            "prior_reliability_uses_state_innovation"
        ]
        is False
    )


def test_center_input_order_does_not_change_selected_material_ids() -> None:
    inputs = _synthetic_inputs()
    original_report, original_candidate = _predict(inputs)
    permutation = np.asarray([8, 3, 14, 1, 12, 5, 10, 0, 15, 6, 4, 11, 2, 13, 7, 9])
    inputs["center_ids"] = inputs["center_ids"][permutation]
    inputs["reliability"] = inputs["reliability"][:, permutation]
    inputs["variance"] = inputs["variance"][:, permutation]

    permuted_report, permuted_candidate = _predict(inputs)

    assert (
        original_report["updates"][0]["selected_center_ids"]
        == permuted_report["updates"][0]["selected_center_ids"]
    )
    np.testing.assert_allclose(original_candidate, permuted_candidate)


def test_insufficient_pairwise_support_is_exact_fallback() -> None:
    inputs = _synthetic_inputs()
    inputs["validity"][3, 8:] = False

    report, candidate = _predict(inputs)

    assert report["candidate_update_count"] == 0
    assert candidate.tobytes() == inputs["baseline"].tobytes()
    assert report["updates"][0]["pairwise_gate"]["accepted"] is False
