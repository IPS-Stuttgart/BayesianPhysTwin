from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from causal4d_public.deform360_phystwin_trust import CausalTrustEpisode
from causal4d_public.deform360_reusable_ensemble import (
    ensemble_prediction,
    derive_source_trusted_point_map_control,
    fit_source_gibbs_ensemble,
    gibbs_weights,
    load_reusable_ensemble_config,
    supported_controller_count,
    trusted_candidate_prediction,
    validate_source_gibbs_ensemble_artifact,
)


def _config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_reusable_ensemble_081_v1.json"
    )


def _episode(
    episode_id: str,
    *,
    response_sign: float,
) -> CausalTrustEpisode:
    frames = 7
    nodes = 5
    initial = np.column_stack(
        (np.linspace(0.0, 0.2, nodes), np.zeros(nodes), np.zeros(nodes))
    )
    progress = np.linspace(0.0, 1.0, frames)[:, None, None]
    response = np.zeros((frames, nodes, 3))
    response[..., 1] = 0.08 * progress[..., 0]
    drift = np.zeros_like(response)
    drift[..., 2] = 0.01 * progress[..., 0]
    zero = initial[None] + drift
    driven = zero + response_sign * response
    target = initial[None] + 0.5 * response + 0.2 * drift
    return CausalTrustEpisode(
        episode_id=episode_id,
        target_m=target,
        visibility=np.ones((frames, nodes), dtype=bool),
        validity=np.ones((frames, nodes), dtype=bool),
        driven_m=driven,
        zero_action_m=zero,
        train_stop_frame=5,
        source_data_sha256="a" * 64,
        driven_trajectory_sha256=("b" if response_sign > 0 else "c") * 64,
        zero_action_trajectory_sha256="d" * 64,
        controller_count=2,
    )


def _candidate_bank() -> dict[str, dict[str, CausalTrustEpisode]]:
    return {
        label: {
            episode_id: _episode(episode_id, response_sign=sign)
            for episode_id in ("1", "4", "6")
        }
        for label, sign in (("good", 1.0), ("bad", -1.0))
    }


def test_reusable_ensemble_config_is_canonically_locked(tmp_path: Path) -> None:
    config = load_reusable_ensemble_config(_config_path())

    assert config["config"]["sealed_target"]["may_open_under_this_protocol"] is False
    changed = json.loads(json.dumps(config))
    changed["config"]["sealed_target"]["may_open_under_this_protocol"] = True
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")

    try:
        load_reusable_ensemble_config(changed_path)
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("mutated ensemble config was accepted")


def test_gibbs_weights_are_stable_and_favor_lower_loss() -> None:
    weights = gibbs_weights({"bad": 1_000_000.0, "good": 0.0}, 0.1)

    assert weights["good"] == 1.0
    assert weights["bad"] == 0.0
    assert (
        supported_controller_count(controller_count=2, controller_spring_count=1)
        == 1
    )


def test_ensemble_prediction_reports_parameter_spread() -> None:
    predictions = {
        "a": np.zeros((2, 3, 3)),
        "b": np.ones((2, 3, 3)),
    }
    mean, variance = ensemble_prediction(predictions, {"a": 0.25, "b": 0.75})

    np.testing.assert_allclose(mean, 0.75)
    np.testing.assert_allclose(variance, 0.1875)


def test_supported_trust_ignores_unattached_controller_groups() -> None:
    episode = _episode("8", response_sign=1.0)

    prediction = trusted_candidate_prediction(
        episode,
        base_action_response=0.4,
        autonomous_drift=0.1,
        controller_spring_count=1,
    )

    expected = (
        episode.target_m[:1]
        + 0.4 * (episode.driven_m - episode.zero_action_m)
        + 0.1 * (episode.zero_action_m - episode.target_m[:1])
    )
    np.testing.assert_allclose(prediction, expected)


def test_source_gibbs_fit_uses_only_registered_source_frames() -> None:
    candidates = _candidate_bank()
    parameters = {
        "good": {"init_spring_Y": 10_000.0},
        "bad": {"init_spring_Y": 80_000.0},
    }
    kwargs = {
        "physical_parameters": parameters,
        "controller_springs": {"1": 1, "4": 1, "6": 1},
        "base_action_response": 0.5,
        "autonomous_drift": 0.2,
        "frame_range": (1, 5),
        "temperature_grid": (0.01, 0.1, 1.0),
        "minimum_effective_candidate_count": 1.1,
    }
    result = fit_source_gibbs_ensemble(candidates, **kwargs)

    assert result["posterior_weights"]["good"] > 0.75
    assert result["posterior_diagnostics"]["effective_candidate_count"] >= 1.1
    assert len(result["temperature_table"]) == 3
    assert all(len(row["folds"]) == 3 for row in result["temperature_table"])
    validate_source_gibbs_ensemble_artifact(result)
    point_map = derive_source_trusted_point_map_control(result)
    assert point_map["selected_pooled_candidate_label"] == "good"
    assert point_map["source_diagnostics"]["joint_win_episode_count"] == 3
    assert point_map["source_diagnostics"][
        "same_candidate_selected_in_every_outer_fold"
    ]

    mutated = {}
    for label, by_episode in candidates.items():
        mutated[label] = {}
        for episode_id, episode in by_episode.items():
            target = episode.target_m.copy()
            target[5:] += 10.0
            mutated[label][episode_id] = replace(episode, target_m=target)
    changed = fit_source_gibbs_ensemble(mutated, **kwargs)

    assert changed["selected_temperature"] == result["selected_temperature"]
    assert changed["posterior_weights"] == result["posterior_weights"]
    assert changed["result_sha256"] == result["result_sha256"]
