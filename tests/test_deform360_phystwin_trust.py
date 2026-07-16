from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_phystwin_trust import (
    CausalTrustEpisode,
    CausalTrustWeights,
    PhysicalTrustParameters,
    apply_cardinality_physical_grid_source_gate,
    cardinality_normalized_causal_prediction,
    causal_control_variate_prediction,
    evaluate_cardinality_normalized_fixed_trust,
    fit_cardinality_normalized_physical_grid_source_trust,
    fit_cardinality_normalized_source_causal_trust,
    fit_regime_gated_source_causal_trust,
    fit_source_causal_trust,
    load_cardinality_source_execution_protocol,
    load_cardinality_trust_protocol,
    load_contact_anchored_causal_trust_protocol,
    load_official_phystwin_trust_episode,
    score_causal_trust_interval,
    validate_cardinality_normalized_source_causal_trust_artifact,
    validate_cardinality_physical_grid_source_trust_artifact,
    validate_regime_gated_source_causal_trust_artifact,
    validate_source_causal_trust_artifact,
)


def _episode(
    episode_id: str,
    *,
    offset: float = 0.0,
    controller_count: int = 1,
) -> CausalTrustEpisode:
    frames = 7
    nodes = 4
    initial = np.column_stack(
        (np.linspace(0.0, 0.3, nodes), np.zeros(nodes), np.zeros(nodes))
    )
    progress = np.linspace(0.0, 1.0, frames)[:, None, None]
    autonomous = np.zeros((frames, nodes, 3))
    autonomous[..., 2] = 0.02 * progress[..., 0] + offset * progress[..., 0]
    response = np.zeros((frames, nodes, 3))
    response[..., 1] = 0.08 * progress[..., 0]
    zero = initial[None] + autonomous
    driven = zero + controller_count * response
    target = initial[None] + 0.5 * response + 0.2 * autonomous
    return CausalTrustEpisode(
        episode_id=episode_id,
        target_m=target,
        visibility=np.ones((frames, nodes), dtype=bool),
        validity=np.ones((frames, nodes), dtype=bool),
        driven_m=driven,
        zero_action_m=zero,
        train_stop_frame=5,
        source_data_sha256="a" * 64,
        driven_trajectory_sha256="b" * 64,
        zero_action_trajectory_sha256="c" * 64,
        controller_count=controller_count,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cardinality_protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_cardinality_trust_002_rope_silk_v1.json"
    )


def _cardinality_source_execution_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_cardinality_source_execution_002_rope_silk_v1.json"
    )


def _contact_anchored_protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "deform360_contact_anchored_causal_trust_002_rope_silk_v1.json"
    )


def test_independent_cardinality_protocol_is_canonically_locked(
    tmp_path: Path,
) -> None:
    protocol = load_cardinality_trust_protocol(_cardinality_protocol_path())
    assert protocol["config"]["source_episode_ids"] == [0, 2, 5, 6, 7, 9]
    assert protocol["config"]["sealed_target_episode_id"] == 1

    changed = json.loads(json.dumps(protocol))
    changed["config"]["source_episode_ids"] = [0, 2]
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_cardinality_trust_protocol(changed_path)


def test_independent_source_execution_is_canonically_locked(
    tmp_path: Path,
) -> None:
    protocol = load_cardinality_source_execution_protocol(
        _cardinality_source_execution_path()
    )
    frame_slice = protocol["config"]["frame_slice"]
    assert frame_slice["train_frame_range"] == [0, 64]
    assert frame_slice["untouched_tail_frame_range"] == [64, 81]
    assert (
        protocol["config"]["physical_arm_roles"]["primary_gate_arm"]
        == "source_pooled_grid"
    )

    changed = json.loads(json.dumps(protocol))
    changed["config"]["frame_slice"]["train_frame_range"] = [0, 65]
    changed_path = tmp_path / "changed-source-execution.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_cardinality_source_execution_protocol(changed_path)


def test_contact_anchored_source_discovery_is_locked_before_calibration(
    tmp_path: Path,
) -> None:
    protocol = load_contact_anchored_causal_trust_protocol(
        _contact_anchored_protocol_path()
    )
    config = protocol["config"]
    assert config["information_boundary"]["all_source_outcomes_read_before_this_freeze"]
    assert not config["information_boundary"][
        "calibration_episode_outcomes_read_before_this_freeze"
    ]
    assert config["causal_trust"]["support_tangential_policy"] == (
        "exact-persistence-fallback"
    )

    changed = json.loads(json.dumps(protocol))
    changed["config"]["causal_trust"]["autonomous_drift_weight"] = 0.1
    changed_path = tmp_path / "changed-contact-anchored.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_contact_anchored_causal_trust_protocol(changed_path)


def test_causal_control_variate_contains_exact_control_arms() -> None:
    episode = _episode("e0")
    persistence = causal_control_variate_prediction(
        episode.target_m[:1],
        episode.driven_m,
        episode.zero_action_m,
        CausalTrustWeights(0.0, 0.0),
    )
    raw_phystwin = causal_control_variate_prediction(
        episode.target_m[:1],
        episode.driven_m,
        episode.zero_action_m,
        CausalTrustWeights(1.0, 1.0),
    )
    intervention_only = causal_control_variate_prediction(
        episode.target_m[:1],
        episode.driven_m,
        episode.zero_action_m,
        CausalTrustWeights(1.0, 0.0),
    )

    np.testing.assert_array_equal(
        persistence, np.repeat(episode.target_m[:1], len(episode.target_m), axis=0)
    )
    np.testing.assert_allclose(raw_phystwin, episode.driven_m, atol=1e-15)
    np.testing.assert_allclose(
        intervention_only,
        episode.target_m[:1] + episode.driven_m - episode.zero_action_m,
        atol=1e-15,
    )


def test_fixed_cardinality_policy_normalizes_bimanual_action_response() -> None:
    episode = _episode("bimanual", controller_count=2)

    predicted = cardinality_normalized_causal_prediction(
        episode, base_action_response=0.5, autonomous_drift=0.2
    )
    metrics = evaluate_cardinality_normalized_fixed_trust(
        episode, base_action_response=0.5, autonomous_drift=0.2
    )

    np.testing.assert_allclose(predicted, episode.target_m, atol=1e-15)
    assert metrics["effective_action_response"] == 0.25
    assert metrics["train"]["track_rmse_m"] < 1e-14
    assert metrics["untouched_tail"]["chamfer_m"] < 1e-14

    full = score_causal_trust_interval(episode, predicted, 1, 7)
    assert full["track_rmse_m"] < 1e-14
    assert full["relative_score_vs_persistence"] < 1e-12


def test_source_fit_recovers_separate_action_and_drift_trust() -> None:
    episodes = [_episode(f"e{index}", offset=index * 0.001) for index in range(3)]

    result = fit_source_causal_trust(
        episodes,
        action_response_grid=(0.0, 0.5, 1.0),
        autonomous_drift_grid=(0.0, 0.2, 1.0),
    )

    assert result["selected_weights"] == {
        "action_response": 0.5,
        "autonomous_drift": 0.2,
    }
    assert len(result["leave_one_action_out"]) == 3
    assert all(
        fold["beats_persistence_track"] and fold["beats_persistence_chamfer"]
        for fold in result["leave_one_action_out"]
    )
    validate_source_causal_trust_artifact(result)


def test_cardinality_normalization_transfers_to_bimanual_response() -> None:
    episodes = [
        _episode("unimanual-0"),
        _episode("unimanual-1", offset=0.001),
        _episode("bimanual", offset=0.002, controller_count=2),
    ]

    result = fit_cardinality_normalized_source_causal_trust(
        episodes,
        action_response_grid=(0.0, 0.5, 1.0),
        autonomous_drift_grid=(0.0, 0.2, 1.0),
    )

    assert result["selected_weights"] == {
        "action_response": 0.5,
        "autonomous_drift": 0.2,
    }
    assert result["controller_counts"]["bimanual"] == 2
    assert result["effective_selected_action_response_by_episode"]["bimanual"] == 0.25
    assert all(
        fold["beats_persistence_track"] and fold["beats_persistence_chamfer"]
        for fold in result["leave_one_action_out"]
    )
    bimanual_fold = next(
        fold
        for fold in result["leave_one_action_out"]
        if fold["held_out_episode_id"] == "bimanual"
    )
    assert bimanual_fold["controller_count"] == 2
    assert bimanual_fold["effective_action_response"] == 0.25
    validate_cardinality_normalized_source_causal_trust_artifact(result)


def test_physical_grid_is_selected_inside_each_outer_source_fold() -> None:
    source_ids = ("0", "2", "5", "6", "7", "9")
    controller_counts = (1, 1, 2, 2, 2, 2)
    good = PhysicalTrustParameters(10_000.0, 1.0, 50.0)
    bad = PhysicalTrustParameters(80_000.0, 10.0, 100.0)
    good_episodes = tuple(
        _episode(
            episode_id,
            offset=index * 0.001,
            controller_count=controller_count,
        )
        for index, (episode_id, controller_count) in enumerate(
            zip(source_ids, controller_counts, strict=True)
        )
    )
    bad_episodes = tuple(
        replace(
            episode,
            driven_m=episode.zero_action_m
            - 0.5 * (episode.driven_m - episode.zero_action_m),
            driven_trajectory_sha256="d" * 64,
        )
        for episode in good_episodes
    )

    result = fit_cardinality_normalized_physical_grid_source_trust(
        {bad: bad_episodes, good: good_episodes},
        action_response_grid=(0.0, 0.5, 1.0),
        autonomous_drift_grid=(0.0, 0.2, 1.0),
    )

    assert result["selected_physical_parameters"] == good.as_dict()
    assert all(
        fold["selected_physical_parameters"] == good.as_dict()
        and fold["beats_persistence_track"]
        and fold["beats_persistence_chamfer"]
        for fold in result["leave_one_action_out"]
    )
    validate_cardinality_physical_grid_source_trust_artifact(result)

    mutated_candidates = {}
    for physical, episodes in {bad: bad_episodes, good: good_episodes}.items():
        mutated = []
        for episode in episodes:
            target = episode.target_m.copy()
            target[episode.train_stop_frame :] += 0.5
            mutated.append(replace(episode, target_m=target))
        mutated_candidates[physical] = tuple(mutated)
    mutated_result = fit_cardinality_normalized_physical_grid_source_trust(
        mutated_candidates,
        action_response_grid=(0.0, 0.5, 1.0),
        autonomous_drift_grid=(0.0, 0.2, 1.0),
    )
    assert (
        mutated_result["selected_physical_parameters"]
        == result["selected_physical_parameters"]
    )
    assert mutated_result["selected_weights"] == result["selected_weights"]
    assert [
        (
            fold["selected_physical_parameters"],
            fold["selected_weights"],
        )
        for fold in mutated_result["leave_one_action_out"]
    ] == [
        (
            fold["selected_physical_parameters"],
            fold["selected_weights"],
        )
        for fold in result["leave_one_action_out"]
    ]

    gate = apply_cardinality_physical_grid_source_gate(
        result,
        load_cardinality_trust_protocol(_cardinality_protocol_path()),
        load_cardinality_source_execution_protocol(
            _cardinality_source_execution_path()
        ),
        registered_qa_by_episode={episode_id: True for episode_id in source_ids},
        tail_mutation_invariant=True,
    )
    assert gate["passed"]
    assert gate["joint_win_count"] == 6
    assert gate["bimanual_joint_win_count"] == 4


def test_regime_gate_cross_fits_prehensile_and_falls_back_exactly() -> None:
    episodes = [_episode(f"e{index}", offset=index * 0.001) for index in range(5)]
    regimes = {
        "e0": "prehensile",
        "e1": "prehensile",
        "e2": "prehensile",
        "e3": "nonprehensile",
        "e4": "nonprehensile",
    }

    result = fit_regime_gated_source_causal_trust(
        episodes,
        regimes,
        action_response_grid=(0.0, 0.5, 1.0),
        autonomous_drift_grid=(0.0, 0.2, 1.0),
    )

    assert result["policy"]["prehensile"]["selected_weights"] == {
        "action_response": 0.5,
        "autonomous_drift": 0.2,
    }
    fallback_folds = [
        fold
        for fold in result["leave_one_action_out"]
        if fold["contact_regime"] == "nonprehensile"
    ]
    assert len(fallback_folds) == 2
    assert all(
        fold["selected_weights"] == {"action_response": 0.0, "autonomous_drift": 0.0}
        and fold["exact_persistence_fallback"]
        for fold in fallback_folds
    )
    assert result["prospective_source_gate"]["passed"]
    validate_regime_gated_source_causal_trust_artifact(result)


def test_tail_mutation_cannot_change_source_selected_weights() -> None:
    episodes = [_episode(f"e{index}", offset=index * 0.001) for index in range(3)]
    original = fit_source_causal_trust(
        episodes,
        action_response_grid=(0.0, 0.5, 1.0),
        autonomous_drift_grid=(0.0, 0.2, 1.0),
    )
    changed_target = episodes[-1].target_m.copy()
    changed_target[episodes[-1].train_stop_frame :] += 0.5
    mutated = [*episodes[:-1], replace(episodes[-1], target_m=changed_target)]

    changed = fit_source_causal_trust(
        mutated,
        action_response_grid=(0.0, 0.5, 1.0),
        autonomous_drift_grid=(0.0, 0.2, 1.0),
    )

    assert changed["selected_weights"] == original["selected_weights"]
    assert changed["result_sha256"] != original["result_sha256"]


def test_causal_trust_artifact_rejects_boundary_tampering() -> None:
    result = fit_source_causal_trust(
        [_episode("e0"), _episode("e1", offset=0.001)],
        action_response_grid=(0.0, 0.5, 1.0),
        autonomous_drift_grid=(0.0, 0.2, 1.0),
    )
    changed = dict(result)
    changed["information_boundary"] = dict(result["information_boundary"])
    changed["information_boundary"]["target_episode_read"] = True

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source_causal_trust_artifact(changed)


def test_load_official_warp_pair_checks_matched_configuration(tmp_path: Path) -> None:
    episode = _episode("e0")
    data_path = tmp_path / "data.pkl"
    split_path = tmp_path / "split.json"
    with data_path.open("wb") as stream:
        pickle.dump(
            {
                "object_points": episode.target_m,
                "object_visibilities": episode.visibility,
                "object_motions_valid": episode.validity,
            },
            stream,
        )
    split_path.write_text(
        json.dumps({"frame_len": 7, "train": [0, 5], "test": [5, 7]}),
        encoding="utf-8",
    )
    result_paths = []
    for name, scale, trajectory in (
        ("driven", 1.0, episode.driven_m),
        ("zero", 0.0, episode.zero_action_m),
    ):
        root = tmp_path / name
        root.mkdir()
        trajectory_path = root / "official_phystwin_trajectory.npz"
        np.savez_compressed(trajectory_path, vertices=trajectory)
        payload = {
            "passed": True,
            "source_only_smoke": True,
            "official_phystwin_revision": "revision",
            "data_sha256": _sha256(data_path),
            "config_sha256": "d" * 64,
            "split_sha256": _sha256(split_path),
            "config_overrides": {"init_spring_Y": 1.0},
            "support_dynamics": {"mode": "official-ground"},
            "effective_inertia": {"particle_mass_scale": 1.0},
            "contact_transmission": {"scale": 1.0},
            "realized_actuation": {"controller_displacement_scale": scale},
            "frame_count": 7,
            "num_controller_points": 1,
            "num_original_points": 4,
            "trajectory_sha256": _sha256(trajectory_path),
        }
        result_path = root / "official_phystwin_smoke.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        result_paths.append(result_path)

    loaded = load_official_phystwin_trust_episode(
        "e0", data_path, result_paths[0], result_paths[1], split_path
    )
    np.testing.assert_allclose(loaded.driven_m, episode.driven_m)

    source_payloads = []
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        source_payloads.append(dict(payload))
        payload["source_only_smoke"] = False
        payload["reusable_dynamics_calibration"] = True
        result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not source-only"):
        load_official_phystwin_trust_episode(
            "e0", data_path, result_paths[0], result_paths[1], split_path
        )
    loaded_calibration = load_official_phystwin_trust_episode(
        "e0",
        data_path,
        result_paths[0],
        result_paths[1],
        split_path,
        evidence_scope="reusable-calibration",
    )
    np.testing.assert_allclose(loaded_calibration.driven_m, episode.driven_m)
    for result_path, payload in zip(result_paths, source_payloads, strict=True):
        result_path.write_text(json.dumps(payload), encoding="utf-8")

    legacy = json.loads(result_paths[1].read_text(encoding="utf-8"))
    legacy.pop("contact_transmission")
    result_paths[1].write_text(json.dumps(legacy), encoding="utf-8")
    loaded_legacy = load_official_phystwin_trust_episode(
        "e0", data_path, result_paths[0], result_paths[1], split_path
    )
    np.testing.assert_allclose(loaded_legacy.zero_action_m, episode.zero_action_m)

    legacy_driven = json.loads(result_paths[0].read_text(encoding="utf-8"))
    legacy_driven.pop("realized_actuation")
    result_paths[0].write_text(json.dumps(legacy_driven), encoding="utf-8")
    loaded_legacy_driven = load_official_phystwin_trust_episode(
        "e0", data_path, result_paths[0], result_paths[1], split_path
    )
    np.testing.assert_allclose(
        loaded_legacy_driven.driven_m, episode.driven_m
    )

    changed = json.loads(result_paths[1].read_text(encoding="utf-8"))
    changed["config_sha256"] = "e" * 64
    result_paths[1].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="config_sha256"):
        load_official_phystwin_trust_episode(
            "e0", data_path, result_paths[0], result_paths[1], split_path
        )
