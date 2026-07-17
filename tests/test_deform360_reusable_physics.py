from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import causal4d_public.deform360_reusable_physics as reusable
from causal4d_public.deform360_independent_source import sha256_file
from causal4d_public.deform360_reusable_trust_protocol import (
    authorize_reusable_trust_episode,
    load_reusable_trust_protocol,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "configs/causal4d_public/deform360_reusable_trust_fresh_v1.json"
ADDENDUM = (
    ROOT
    / "configs/causal4d_public/deform360_reusable_trust_physics_addendum_v1.json"
)
EXECUTION = (
    ROOT / "configs/causal4d_public/deform360_reusable_trust_execution_v1.json"
)


def _pickle(path: Path, value: object) -> None:
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_physical_response_seal_is_outcome_blind_and_checksummed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM)
    object_id = "003-cable"
    episode_id = 1
    authorization = authorize_reusable_trust_episode(
        protocol, object_id=object_id, episode_id=episode_id, operation="fit"
    )
    frame_count = 4
    point_count = 3
    node_count = 4
    initial = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    data = {
        "object_points": np.repeat(initial[None], frame_count, axis=0),
        "object_colors": np.zeros((frame_count, point_count, 3), dtype=np.float32),
        "object_visibilities": np.ones((frame_count, point_count), dtype=bool),
        "object_motions_valid": np.ones((frame_count, point_count), dtype=bool),
        "controller_points": np.zeros((frame_count, 2, 3), dtype=np.float32),
        "prediction_only_input": {
            "schema_version": 1,
            "object_id": object_id,
            "episode_id": episode_id,
            "object_observation_frames_used": [0],
            "known_future_robot_trajectory_used": True,
            "future_object_observations_present": False,
            "future_tactile_used": False,
        },
    }
    data_path = tmp_path / "prediction.pkl"
    simulator_path = tmp_path / "simulator.pkl"
    graph_path = tmp_path / "graph.npz"
    readout_path = tmp_path / "readout.npz"
    summary_path = tmp_path / "summary.json"
    _pickle(data_path, data)
    graph_path.write_bytes(b"graph")
    graph = SimpleNamespace(
        vertices=np.zeros((node_count, 3)),
        observed_node_count=node_count,
        sha256="a" * 64,
    )
    monkeypatch.setattr(reusable, "load_canonical_deform360_graph", lambda _: graph)
    monkeypatch.setattr(
        reusable,
        "graph_contact_distance_m",
        lambda _: np.zeros(node_count, dtype=np.float64),
    )
    monkeypatch.setattr(
        reusable,
        "graph_readout_action_support",
        lambda weights, _distance, *, length_scale_m: np.ones(len(weights)),
    )
    _pickle(
        simulator_path,
        {"reusable_graph_registration": {"canonical_graph_sha256": graph.sha256}},
    )
    weights = np.zeros((point_count, node_count), dtype=np.float32)
    weights[np.arange(point_count), np.arange(point_count)] = 1.0
    np.savez_compressed(
        readout_path,
        readout_weights=weights,
        canonical_graph_sha256=np.asarray(graph.sha256),
    )
    _json(
        summary_path,
        {
            "passed": True,
            "object_id": object_id,
            "episode_id": episode_id,
            "fresh_authorization": authorization,
            "input_sha256": {"episode_final_data": sha256_file(data_path)},
            "output_sha256": {
                "simulator_final_data": sha256_file(simulator_path)
            },
            "information_boundary": {
                "target_access": False,
                "post_initial_object_observation_used": False,
            },
        },
    )

    fixed = protocol["addendum"]["object_level_physical_grid"][
        "fixed_warp_settings"
    ]
    parameters = {
        "init_spring_y": 10000.0,
        "drag_damping": 1.0,
        "dashpot_damping": 50.0,
    }
    overrides = {
        "controller_max_neighbours": 1,
        "controller_radius": 0.03,
        "dashpot_damping": 50.0,
        "drag_damping": 1.0,
        "init_spring_Y": 10000.0,
    }
    result_paths = {}
    for name, scale in (("driven", 1.0), ("zero", 0.0)):
        result_dir = tmp_path / name
        result_dir.mkdir()
        trajectory_path = result_dir / "official_phystwin_trajectory.npz"
        trajectory = np.zeros((frame_count, node_count, 3), dtype=np.float32)
        if name == "driven":
            trajectory[:, :, 2] = np.arange(frame_count)[:, None] * 0.01
        np.savez_compressed(trajectory_path, vertices=trajectory)
        result_path = result_dir / "result.json"
        _json(
            result_path,
            {
                "passed": True,
                "data_sha256": sha256_file(simulator_path),
                "official_phystwin_revision": fixed["official_phystwin_revision"],
                "config_sha256": fixed["config_sha256"],
                "config_overrides": overrides,
                "support_dynamics": {"mode": "official-ground"},
                "canonical_reusable_graph": {
                    "file_sha256": sha256_file(graph_path),
                    "reusable_graph_sha256": graph.sha256,
                    "controller_patch_size_per_anchor": 16,
                },
                "trajectory_sha256": sha256_file(trajectory_path),
                "realized_actuation": {"controller_displacement_scale": scale},
                "split_sha256": None,
            },
        )
        result_paths[name] = result_path

    archive = tmp_path / "response.npz"
    payload = reusable.seal_reusable_physics_response(
        archive,
        protocol=protocol,
        object_id=object_id,
        episode_id=episode_id,
        operation="fit",
        parameters=parameters,
        prediction_data_path=data_path,
        simulator_data_path=simulator_path,
        graph_path=graph_path,
        readout_path=readout_path,
        twin_summary_path=summary_path,
        driven_result_path=result_paths["driven"],
        zero_result_path=result_paths["zero"],
    )
    validated = reusable.validate_reusable_physics_response(
        payload, protocol=protocol, verify_archive=True
    )

    assert validated["candidate_index"] == 0
    assert payload["information_boundary"]["object_outcome_used"] is False
    with np.load(archive, allow_pickle=False) as stored:
        assert stored["prediction_m"][-1, :, 2] == pytest.approx(0.027)

    with np.load(archive, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    arrays["prediction_m"][1, 0, 0] = 1.0
    np.savez_compressed(archive, **arrays)
    with pytest.raises(ValueError, match="archive checksum changed"):
        reusable.validate_reusable_physics_response(
            payload, protocol=protocol, verify_archive=True
        )


def test_physical_response_rejects_tuple_outside_frozen_grid() -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM)

    with pytest.raises(ValueError, match="outside the frozen grid"):
        reusable._candidate_index(
            protocol,
            {
                "init_spring_y": 80000.0,
                "drag_damping": 1.0,
                "dashpot_damping": 50.0,
            },
        )


def test_physical_selection_pools_only_fit_episodes_and_freezes_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = load_reusable_trust_protocol(PARENT, ADDENDUM, EXECUTION)
    object_id = "003-cable"
    fit_ids = protocol["splits"][object_id]["fit_episode_ids"]
    candidates = protocol["physical_candidates"]
    selected_index = 7
    frames = 76
    points = 3
    base = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        dtype=np.float32,
    )
    displacement = np.linspace(0.0, 0.02, frames, dtype=np.float32)
    target_paths = {}
    robot_paths = {}
    response_paths = []
    for episode_id in fit_ids:
        target = np.repeat(base[None], frames, axis=0)
        target[:, :, 2] += displacement[:, None]
        target_path = tmp_path / f"target-{episode_id}.pkl"
        _pickle(
            target_path,
            {
                "object_points": target,
                "object_visibilities": np.ones((frames, points), dtype=bool),
                "object_motions_valid": np.ones((frames, points), dtype=bool),
            },
        )
        target_paths[episode_id] = target_path
        robot_path = tmp_path / f"robot-{episode_id}.npz"
        actions = np.zeros((81, 1, 5, 3), dtype=np.float32)
        actions[:, :, :, 2] = np.linspace(0.0, 0.03, 81)[:, None, None]
        openings = np.linspace(0.05, 0.01, 81, dtype=np.float32)[:, None]
        np.savez_compressed(robot_path, actions=actions, openings=openings)
        robot_paths[episode_id] = robot_path
        authorization = authorize_reusable_trust_episode(
            protocol,
            object_id=object_id,
            episode_id=episode_id,
            operation="fit",
        )
        persistence = np.repeat(base[None], frames, axis=0)
        for candidate_index, parameters in enumerate(candidates):
            prediction = persistence.copy()
            if candidate_index == selected_index:
                prediction[:, :, 2] += (0.9 / 0.45) * displacement[:, None]
            archive = tmp_path / f"response-{episode_id}-{candidate_index}.npz"
            arrays = {
                "prediction_m": prediction,
                "persistence_m": persistence,
                "driven_readout_m": persistence,
                "zero_action_readout_m": persistence,
                "action_support": np.ones(points, dtype=np.float32),
                "frame_zero_points_m": base,
            }
            np.savez_compressed(archive, **arrays)
            payload = {
                "schema_version": 1,
                "artifact_kind": "Deform360ReusableTwinPhysicalResponse",
                "prospective_authorization": authorization,
                "object_id": object_id,
                "episode_id": episode_id,
                "episode_key": f"{object_id}/{episode_id}",
                "candidate_index": candidate_index,
                "physical_parameters": parameters,
                "canonical_graph": {
                    "file_sha256": "d" * 64,
                    "reusable_graph_sha256": "e" * 64,
                    "shared_object_topology": True,
                },
                "prediction_archive": {
                    "path": str(archive),
                    "file_sha256": sha256_file(archive),
                    "array_sha256": {
                        name: reusable.sha256_array(value)
                        for name, value in arrays.items()
                    },
                },
                "input_sha256": {
                    "driven_trajectory": "b" * 64,
                    "zero_action_trajectory": "c" * 64,
                },
                "information_boundary": {
                    "object_observation_frames_used": [0],
                    "post_initial_object_observation_used": False,
                    "future_object_track_read": False,
                    "future_object_visibility_read": False,
                    "future_tactile_read": False,
                    "external_target_scoring_in_warp": False,
                    "object_outcome_used": False,
                },
            }
            payload["result_sha256"] = reusable._result_sha256(payload)
            response_path = tmp_path / f"response-{episode_id}-{candidate_index}.json"
            _json(response_path, payload)
            response_paths.append(response_path)

    trust_path = tmp_path / "trust.json"
    trust_path.write_text("{}", encoding="utf-8")

    class TrustModel:
        result_sha256 = protocol["addendum"]["reference_trust_response"][
            "candidate_result_sha256"
        ]
        reference_response_alpha = 0.9
        feature_names = ("mean_minimum_gripper_closure",)

        @staticmethod
        def decide(_features: object) -> SimpleNamespace:
            return SimpleNamespace(
                alpha=0.45,
                raw_alpha=0.45,
                closure_accepted=True,
                closure_value=0.5,
            )

    monkeypatch.setattr(
        reusable, "load_reusable_twin_trust_candidate", lambda _: TrustModel()
    )
    first_fit_responses = [
        path
        for path in response_paths
        if f"response-{fit_ids[0]}-" in path.name
    ]
    grid_seal = reusable.build_reusable_physics_fit_grid_seal(
        first_fit_responses,
        protocol=protocol,
        object_id=object_id,
        episode_id=fit_ids[0],
    )
    validated_grid = reusable.validate_reusable_physics_fit_grid_seal(
        grid_seal, protocol=protocol, verify_responses=True
    )
    result = reusable.fit_reusable_physics_selection(
        response_paths,
        target_paths=target_paths,
        robot_paths=robot_paths,
        protocol=protocol,
        trust_artifact_path=trust_path,
        object_id=object_id,
    )
    validated = reusable.validate_reusable_physics_selection(
        result, protocol=protocol
    )

    assert validated_grid["response_count"] == 18
    assert validated["selected_candidate_index"] == selected_index
    assert len(result["leave_one_fit_episode_out"]) == 6
    assert len(result["single_fit_episode_controls"]) == 6
    assert result["information_boundary"]["held_out_outcomes_read"] is False
    assert result["canonical_graph"]["reusable_graph_sha256"] == "e" * 64

    changed = json.loads(response_paths[0].read_text(encoding="utf-8"))
    changed["canonical_graph"]["reusable_graph_sha256"] = "f" * 64
    changed.pop("result_sha256")
    changed["result_sha256"] = reusable._result_sha256(changed)
    _json(response_paths[0], changed)
    with pytest.raises(ValueError, match="rebuilt the canonical graph"):
        reusable.fit_reusable_physics_selection(
            response_paths,
            target_paths=target_paths,
            robot_paths=robot_paths,
            protocol=protocol,
            trust_artifact_path=trust_path,
            object_id=object_id,
        )
