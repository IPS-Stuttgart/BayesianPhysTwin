import json

import numpy as np

from causal4d_public.deform360_causal_expert_router import (
    CausalExpertEpisode,
    CausalExpertRouterModel,
    build_causal_expert_features,
    cross_fit_causal_expert_router,
    fit_causal_expert_router,
    load_causal_expert_router,
)


def _episode(object_id: str, episode_id: int) -> CausalExpertEpisode:
    return CausalExpertEpisode(
        object_id=object_id,
        episode_id=episode_id,
        labels=("persistence", "helpful", "harmful"),
        features=np.asarray([[0.0], [1.0]]),
        normalized_scores=np.asarray([1.0, 0.8, 1.2]),
    )


def test_candidate_features_are_outcome_independent():
    features = build_causal_expert_features(
        {
            "base_support_scale_m": 0.003,
            "support_growth_per_travel": 0.1,
            "initial_contact_gain": 0.5,
            "acquired_contact_gain": 0.0,
            "transform_mode": "translation",
            "trust_features": {"response_rms_m": 0.002},
            "diagnostics": {
                "maximum_transport_weight": 0.4,
                "contact_fraction_by_group": [1.0, 0.5],
                "minimum_controller_to_initial_object_distance_m_by_group": [
                    0.001,
                    0.002,
                ],
                "onset_frames": [0, None],
            },
            "metrics": {"must_not_be_read": object()},
        }
    )

    assert features["response_rms_m"] == 0.002
    assert features["candidate_initial_contact_fraction"] == 0.5
    assert features["candidate_missing_contact_fraction"] == 0.5
    assert "metrics" not in features


def test_router_returns_exact_persistence_when_upper_bound_is_not_better():
    episode = _episode("object-a", 1)
    model = CausalExpertRouterModel(
        feature_names=("x",),
        candidate_labels=episode.labels,
        coefficients=np.asarray([-0.2, 0.0]),
        feature_mean=np.asarray([0.5]),
        feature_scale=np.asarray([0.5]),
        ridge=1.0,
        selected_residual_quantile=0.3,
        minimum_improvement_fraction=0.0,
        calibration_level=0.9,
    )

    decision = model.decide(episode)

    assert decision.accepted is False
    assert decision.selected_label == "persistence"
    assert decision.selected_index == 0


def test_object_cross_fit_learns_helpful_expert_and_preserves_safety():
    episodes = [
        _episode(object_id, episode_id)
        for object_id in ("a", "b", "c")
        for episode_id in (1, 2)
    ]

    result = cross_fit_causal_expert_router(
        episodes,
        feature_names=("x",),
        ridge_grid=(0.1, 1.0),
        calibration_level=0.8,
    )

    assert result["mean_normalized_score"] < 1.0
    assert result["maximum_normalized_score"] <= 1.0
    assert result["accepted_fraction"] == 1.0
    assert {row["selected_label"] for row in result["rows"]} == {"helpful"}


def test_router_artifact_round_trip(tmp_path):
    episodes = [_episode(object_id, 1) for object_id in ("a", "b", "c")]
    model, _ = fit_causal_expert_router(
        episodes,
        feature_names=("x",),
        ridge_grid=(0.1,),
        calibration_level=0.8,
    )
    payload = model.to_payload(source={"panel": "synthetic"})
    path = tmp_path / "router.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_causal_expert_router(path)

    assert loaded.result_sha256 == payload["result_sha256"]
    assert loaded.decide(episodes[0]).selected_label == "helpful"
