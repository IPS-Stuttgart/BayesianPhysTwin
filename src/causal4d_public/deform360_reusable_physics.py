"""Outcome-blind physical responses for the fresh reusable-twin panel."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_action_support import (
    graph_contact_distance_m,
    graph_readout_action_support,
)
from .deform360_independent_source import (
    sha256_array,
    sha256_file,
    validate_prediction_only_bundle,
)
from .deform360_phystwin_trust import (
    CausalTrustEpisode,
    score_causal_trust_interval,
)
from .deform360_reusable_graph import load_canonical_deform360_graph
from .deform360_reusable_trust import (
    build_deform360_trust_features,
    load_reusable_twin_trust_candidate,
)
from .deform360_reusable_trust_protocol import authorize_reusable_trust_episode


REUSABLE_PHYSICS_RESPONSE_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    _require(isinstance(value, Mapping), f"{path.name} must contain a mapping")
    return value


def _physical_parameters(value: Mapping[str, Any]) -> dict[str, float]:
    try:
        normalized = {
            "init_spring_y": float(value["init_spring_y"]),
            "drag_damping": float(value["drag_damping"]),
            "dashpot_damping": float(value["dashpot_damping"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("physical parameter tuple is incomplete") from error
    _require(
        all(np.isfinite(item) for item in normalized.values()),
        "physical parameter tuple is non-finite",
    )
    return normalized


def _candidate_index(
    protocol: Mapping[str, Any], parameters: Mapping[str, Any]
) -> int:
    candidate = _physical_parameters(parameters)
    candidates = [
        _physical_parameters(item) for item in protocol.get("physical_candidates", ())
    ]
    _require(candidate in candidates, "physical tuple is outside the frozen grid")
    return candidates.index(candidate)


def _trajectory_from_result(
    result_path: Path,
    *,
    result: Mapping[str, Any],
    frame_count: int,
    node_count: int,
) -> tuple[Path, np.ndarray]:
    trajectory_path = result_path.with_name("official_phystwin_trajectory.npz")
    _require(
        result.get("trajectory_sha256") == sha256_file(trajectory_path),
        "Warp trajectory checksum changed",
    )
    with np.load(trajectory_path, allow_pickle=False) as stored:
        trajectory = np.asarray(stored["vertices"], dtype=np.float64)
    _require(
        trajectory.ndim == 3
        and trajectory.shape[0] == frame_count
        and trajectory.shape[1] >= node_count
        and trajectory.shape[2] == 3,
        "Warp trajectory shape is invalid",
    )
    return trajectory_path, trajectory[:, :node_count]


def seal_reusable_physics_response(
    archive_path: str | Path,
    *,
    protocol: Mapping[str, Any],
    object_id: str,
    episode_id: int,
    operation: str,
    parameters: Mapping[str, Any],
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    readout_path: str | Path,
    twin_summary_path: str | Path,
    driven_result_path: str | Path,
    zero_result_path: str | Path,
) -> dict[str, Any]:
    """Seal one matched driven/zero response without opening object outcomes."""

    authorization = authorize_reusable_trust_episode(
        protocol,
        object_id=object_id,
        episode_id=episode_id,
        operation=operation,
    )
    candidate = _physical_parameters(parameters)
    candidate_index = _candidate_index(protocol, candidate)
    data_file = Path(prediction_data_path).resolve()
    simulator_file = Path(simulator_data_path).resolve()
    graph_file = Path(graph_path).resolve()
    readout_file = Path(readout_path).resolve()
    summary_file = Path(twin_summary_path).resolve()
    driven_file = Path(driven_result_path).resolve()
    zero_file = Path(zero_result_path).resolve()

    data = _load_pickle(data_file)
    simulator_data = _load_pickle(simulator_file)
    bundle = validate_prediction_only_bundle(
        data, object_id=object_id, episode_id=episode_id
    )
    graph = load_canonical_deform360_graph(graph_file)
    twin = _load_json(summary_file)
    _require(twin.get("passed") is True, "episode state failed admission")
    _require(
        twin.get("object_id") == object_id
        and int(twin.get("episode_id", -1)) == int(episode_id),
        "automatic twin belongs to another episode",
    )
    _require(
        twin.get("fresh_authorization") == authorization,
        "episode state uses another fresh authorization",
    )
    _require(
        twin.get("information_boundary", {}).get("target_access") is False
        and twin.get("information_boundary", {}).get(
            "post_initial_object_observation_used"
        )
        is False,
        "episode state crossed the prediction boundary",
    )

    data_sha = sha256_file(data_file)
    simulator_sha = sha256_file(simulator_file)
    graph_sha = sha256_file(graph_file)
    readout_sha = sha256_file(readout_file)
    _require(
        twin.get("input_sha256", {}).get("episode_final_data") == data_sha,
        "episode state uses another prediction-only bundle",
    )
    _require(
        twin.get("output_sha256", {}).get("simulator_final_data") == simulator_sha,
        "episode state simulator bundle hash changed",
    )
    _require(
        simulator_data.get("reusable_graph_registration", {}).get(
            "canonical_graph_sha256"
        )
        == graph.sha256,
        "simulator bundle uses another graph",
    )
    with np.load(readout_file, allow_pickle=False) as state:
        weights = np.asarray(state["readout_weights"], dtype=np.float64)
        state_graph_sha = str(np.asarray(state["canonical_graph_sha256"]).item())
    _require(state_graph_sha == graph.sha256, "readout uses another graph")
    _require(
        weights.shape == (bundle["point_count"], len(graph.vertices)),
        "readout shape differs from frame-zero identities and graph",
    )

    addendum = protocol["addendum"]
    fixed = addendum["object_level_physical_grid"]["fixed_warp_settings"]
    expected_overrides = {
        "controller_max_neighbours": int(fixed["controller_max_neighbours"]),
        "controller_radius": float(fixed["controller_radius_m"]),
        "dashpot_damping": candidate["dashpot_damping"],
        "drag_damping": candidate["drag_damping"],
        "init_spring_Y": candidate["init_spring_y"],
    }
    results = {
        "driven": _load_json(driven_file),
        "zero_action": _load_json(zero_file),
    }
    trajectories: dict[str, np.ndarray] = {}
    trajectory_files: dict[str, Path] = {}
    for name, result_file in (("driven", driven_file), ("zero_action", zero_file)):
        result = results[name]
        _require(result.get("passed") is True, f"{name} Warp rollout failed")
        _require(
            "external_target_scoring" not in result,
            f"{name} Warp rollout read an object outcome",
        )
        _require(
            result.get("data_sha256") == simulator_sha,
            f"{name} simulator data hash differs",
        )
        _require(
            result.get("official_phystwin_revision")
            == fixed["official_phystwin_revision"],
            f"{name} PhysTwin revision differs from the addendum",
        )
        _require(
            result.get("config_sha256") == fixed["config_sha256"]
            and result.get("config_overrides") == expected_overrides,
            f"{name} Warp dynamics differ from the frozen tuple",
        )
        _require(
            result.get("support_dynamics", {}).get("mode")
            == fixed["support_dynamics"],
            f"{name} support dynamics differ from the addendum",
        )
        canonical = result.get("canonical_reusable_graph", {})
        _require(
            isinstance(canonical, Mapping)
            and canonical.get("file_sha256") == graph_sha
            and canonical.get("reusable_graph_sha256") == graph.sha256,
            f"{name} rollout uses another graph",
        )
        _require(
            int(canonical.get("controller_patch_size_per_anchor", -1))
            == int(fixed["canonical_controller_patch_size"]),
            f"{name} controller patch differs from the addendum",
        )
        if "execution" in protocol:
            attachment = protocol["execution"]["dynamic_controller_attachment"]
            _require(
                canonical.get("object_topology_rebuilt_per_episode") is False
                and int(canonical.get("contact_anchor_count", -1)) == 0
                and canonical.get("controller_attachment_mode")
                == "episode_dynamic_grouped_anchors"
                and int(canonical.get("controller_group_size", -1))
                == int(attachment["controller_group_size"]),
                f"{name} rollout violates the reusable execution lock",
            )
        trajectory_file, trajectory = _trajectory_from_result(
            result_file,
            result=result,
            frame_count=int(bundle["frame_count"]),
            node_count=len(graph.vertices),
        )
        _require(np.all(np.isfinite(trajectory)), f"{name} trajectory is non-finite")
        trajectory_files[name] = trajectory_file
        trajectories[name] = trajectory

    reference = addendum["reference_trust_response"]
    _require(
        float(
            results["driven"]
            .get("realized_actuation", {})
            .get("controller_displacement_scale", -1.0)
        )
        == float(reference["controller_displacement_scale_driven"]),
        "driven rollout does not use the frozen action scale",
    )
    _require(
        float(
            results["zero_action"]
            .get("realized_actuation", {})
            .get("controller_displacement_scale", -1.0)
        )
        == float(reference["controller_displacement_scale_zero"]),
        "zero-action rollout moves the controller",
    )
    _require(
        results["driven"].get("config_sha256")
        == results["zero_action"].get("config_sha256")
        and results["driven"].get("split_sha256")
        == results["zero_action"].get("split_sha256"),
        "matched rollouts use different config or frame splits",
    )

    distance = graph_contact_distance_m(graph)
    length_scale = float(reference["graph_action_support_length_scale_m"])
    response_alpha = float(reference["reference_response_alpha"])
    support = graph_readout_action_support(
        weights, distance, length_scale_m=length_scale
    )
    driven_readout = np.einsum(
        "mn,tnc->tmc", weights, trajectories["driven"], optimize=True
    )
    zero_readout = np.einsum(
        "mn,tnc->tmc", weights, trajectories["zero_action"], optimize=True
    )
    initial = np.asarray(data["object_points"][0], dtype=np.float64)
    offset = initial - zero_readout[0]
    driven_readout += offset[None]
    zero_readout += offset[None]
    prediction = initial[None] + response_alpha * support[None, :, None] * (
        driven_readout - zero_readout
    )
    persistence = np.repeat(initial[None], int(bundle["frame_count"]), axis=0)
    _require(np.all(np.isfinite(prediction)), "physical response is non-finite")

    arrays = {
        "prediction_m": prediction.astype(np.float32),
        "persistence_m": persistence.astype(np.float32),
        "driven_readout_m": driven_readout.astype(np.float32),
        "zero_action_readout_m": zero_readout.astype(np.float32),
        "action_support": support.astype(np.float32),
        "frame_zero_points_m": initial.astype(np.float32),
    }
    archive_file = Path(archive_path).resolve()
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive_file, **arrays)
    payload: dict[str, Any] = {
        "schema_version": REUSABLE_PHYSICS_RESPONSE_SCHEMA_VERSION,
        "artifact_kind": "Deform360ReusableTwinPhysicalResponse",
        "prospective_authorization": authorization,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "episode_key": f"{object_id}/{int(episode_id)}",
        "candidate_index": candidate_index,
        "physical_parameters": candidate,
        "canonical_graph": {
            "file_sha256": graph_sha,
            "reusable_graph_sha256": graph.sha256,
            "shared_object_topology": True,
        },
        "response": {
            "graph_action_support_length_scale_m": length_scale,
            "reference_response_alpha": response_alpha,
            "observed_graph_node_count": int(graph.observed_node_count),
            "total_graph_node_count": len(graph.vertices),
            "frame_count": int(bundle["frame_count"]),
            "point_count": int(bundle["point_count"]),
        },
        "prediction_archive": {
            "path": str(archive_file),
            "file_sha256": sha256_file(archive_file),
            "array_sha256": {
                name: sha256_array(value) for name, value in arrays.items()
            },
        },
        "input_sha256": {
            "prediction_data": data_sha,
            "simulator_data": simulator_sha,
            "graph": graph_sha,
            "readout": readout_sha,
            "twin_summary": sha256_file(summary_file),
            "driven_result": sha256_file(driven_file),
            "zero_action_result": sha256_file(zero_file),
            "driven_trajectory": sha256_file(trajectory_files["driven"]),
            "zero_action_trajectory": sha256_file(
                trajectory_files["zero_action"]
            ),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_used": True,
            "post_initial_object_observation_used": False,
            "future_object_track_read": False,
            "future_object_visibility_read": False,
            "future_tactile_read": False,
            "symbolic_action_label_read": False,
            "external_target_scoring_in_warp": False,
            "object_outcome_used": False,
        },
        "claim_boundary": (
            "outcome-blind physical response; trust and physical selection are "
            "separate frozen stages"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def validate_reusable_physics_response(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    verify_archive: bool = False,
) -> dict[str, Any]:
    """Validate one fresh physical response and optionally every stored array."""

    _require(
        payload.get("schema_version") == REUSABLE_PHYSICS_RESPONSE_SCHEMA_VERSION
        and payload.get("artifact_kind")
        == "Deform360ReusableTwinPhysicalResponse",
        "reusable physical response identity changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "reusable physical response checksum changed",
    )
    object_id = str(payload.get("object_id"))
    episode_id = int(payload.get("episode_id", -1))
    authorization = payload.get("prospective_authorization", {})
    role = authorization.get("role")
    operation = "fit" if role == "object-level-fit" else "held-prediction"
    expected = authorize_reusable_trust_episode(
        protocol,
        object_id=object_id,
        episode_id=episode_id,
        operation=operation,
    )
    _require(authorization == expected, "physical response authorization changed")
    candidate = _physical_parameters(payload.get("physical_parameters", {}))
    _require(
        int(payload.get("candidate_index", -1))
        == _candidate_index(protocol, candidate),
        "physical response candidate identity changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("post_initial_object_observation_used") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("future_object_visibility_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("external_target_scoring_in_warp") is False
        and boundary.get("object_outcome_used") is False,
        "physical response crossed the outcome boundary",
    )
    if "execution" in protocol:
        graph = payload.get("canonical_graph", {})
        _require(
            graph.get("shared_object_topology") is True
            and isinstance(graph.get("file_sha256"), str)
            and isinstance(graph.get("reusable_graph_sha256"), str),
            "physical response lacks a shared canonical graph",
        )
    if verify_archive:
        archive = payload.get("prediction_archive", {})
        path = Path(str(archive.get("path", "")))
        _require(path.is_file(), "physical response archive is missing")
        _require(
            sha256_file(path) == archive.get("file_sha256"),
            "physical response archive checksum changed",
        )
        expected_arrays = archive.get("array_sha256", {})
        with np.load(path, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(expected_arrays),
                "physical response arrays changed",
            )
            for name in stored.files:
                _require(
                    sha256_array(stored[name]) == expected_arrays[name],
                    f"physical response array {name} checksum changed",
                )
    return {
        "episode_key": f"{object_id}/{episode_id}",
        "candidate_index": int(payload["candidate_index"]),
        "physical_parameters": candidate,
        "result_sha256": str(payload["result_sha256"]),
    }


def build_reusable_physics_fit_grid_seal(
    response_paths: list[str | Path],
    *,
    protocol: Mapping[str, Any],
    object_id: str,
    episode_id: int,
) -> dict[str, Any]:
    """Seal all 18 fit responses before opening that episode's outcome."""

    authorization = authorize_reusable_trust_episode(
        protocol,
        object_id=object_id,
        episode_id=episode_id,
        operation="fit",
    )
    expected_indices = set(range(len(protocol["physical_candidates"])))
    responses: dict[str, Any] = {}
    for value in response_paths:
        path = Path(value).resolve()
        payload = _load_json(path)
        validated = validate_reusable_physics_response(
            payload, protocol=protocol, verify_archive=True
        )
        _require(
            validated["episode_key"] == authorization["episode_key"],
            "fit grid mixes episodes",
        )
        index = int(validated["candidate_index"])
        key = str(index)
        _require(key not in responses, "fit grid repeats a physical candidate")
        responses[key] = {
            "candidate_index": index,
            "physical_parameters": validated["physical_parameters"],
            "response_json_path": str(path),
            "response_json_sha256": sha256_file(path),
            "response_result_sha256": validated["result_sha256"],
        }
    _require(
        {int(value) for value in responses} == expected_indices,
        "fit grid seal requires all 18 frozen physical responses",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTwinFitGridSeal",
        "protocol_id": protocol["parent"]["protocol_id"],
        "physics_addendum_id": protocol["addendum"]["protocol_id"],
        "parent_file_sha256": protocol["parent_file_sha256"],
        "addendum_file_sha256": protocol["addendum_file_sha256"],
        "prospective_authorization": authorization,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "episode_key": authorization["episode_key"],
        "response_count": len(responses),
        "responses": dict(sorted(responses.items(), key=lambda item: int(item[0]))),
        "information_boundary": {
            "all_physical_responses_hashed": True,
            "fit_outcome_read": False,
            "post_initial_object_observation_used": False,
        },
        "claim_boundary": (
            "fit-response grid seal only; permits outcome construction but no "
            "physical selection by itself"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def validate_reusable_physics_fit_grid_seal(
    payload: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    verify_responses: bool = True,
) -> dict[str, Any]:
    """Validate one complete, outcome-blind 18-response fit grid."""

    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind") == "Deform360ReusableTwinFitGridSeal",
        "reusable fit-grid seal identity changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "reusable fit-grid seal checksum changed",
    )
    _require(
        payload.get("parent_file_sha256") == protocol["parent_file_sha256"]
        and payload.get("addendum_file_sha256") == protocol["addendum_file_sha256"],
        "fit-grid seal uses another protocol",
    )
    object_id = str(payload.get("object_id"))
    episode_id = int(payload.get("episode_id", -1))
    authorization = authorize_reusable_trust_episode(
        protocol,
        object_id=object_id,
        episode_id=episode_id,
        operation="fit",
    )
    _require(
        payload.get("prospective_authorization") == authorization,
        "fit-grid authorization changed",
    )
    responses = payload.get("responses", {})
    expected_indices = set(range(len(protocol["physical_candidates"])))
    _require(
        isinstance(responses, Mapping)
        and {int(value) for value in responses} == expected_indices
        and int(payload.get("response_count", -1)) == len(expected_indices),
        "fit-grid seal is incomplete",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("all_physical_responses_hashed") is True
        and boundary.get("fit_outcome_read") is False
        and boundary.get("post_initial_object_observation_used") is False,
        "fit-grid seal crossed the outcome boundary",
    )
    if verify_responses:
        for key, record in responses.items():
            path = Path(str(record.get("response_json_path", "")))
            _require(
                path.is_file()
                and sha256_file(path) == record.get("response_json_sha256"),
                f"fit response changed for candidate {key}",
            )
            validated = validate_reusable_physics_response(
                _load_json(path), protocol=protocol, verify_archive=True
            )
            _require(
                validated["episode_key"] == authorization["episode_key"]
                and validated["candidate_index"] == int(key)
                and validated["result_sha256"]
                == record.get("response_result_sha256"),
                f"fit response identity changed for candidate {key}",
            )
    return {
        "object_id": object_id,
        "episode_id": episode_id,
        "episode_key": authorization["episode_key"],
        "response_count": len(expected_indices),
        "result_sha256": str(payload["result_sha256"]),
    }


def _load_response_arrays(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = Path(str(payload["prediction_archive"]["path"]))
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]) for name in stored.files}


def _fit_episode(
    target_path: Path,
    *,
    episode_key: str,
    reference_payload: Mapping[str, Any],
) -> CausalTrustEpisode:
    target = _load_pickle(target_path)
    arrays = _load_response_arrays(reference_payload)
    points = np.asarray(target["object_points"], dtype=np.float64)
    visibility = np.asarray(target["object_visibilities"], dtype=bool)
    validity = np.asarray(target["object_motions_valid"], dtype=bool)
    persistence = np.asarray(arrays["persistence_m"], dtype=np.float64)
    _require(points.shape == persistence.shape, "fit target and response shapes differ")
    _require(
        np.array_equal(
            points[0].astype(np.float32),
            np.asarray(arrays["frame_zero_points_m"], dtype=np.float32),
        ),
        "fit target frame zero differs from the sealed response",
    )
    _require(
        visibility.shape == points.shape[:2] and validity.shape == points.shape[:2],
        "fit target masks differ from points",
    )
    return CausalTrustEpisode(
        episode_id=episode_key,
        target_m=points,
        visibility=visibility,
        validity=validity,
        driven_m=persistence,
        zero_action_m=persistence,
        train_stop_frame=60,
        source_data_sha256=sha256_file(target_path),
        driven_trajectory_sha256=str(
            reference_payload["input_sha256"]["driven_trajectory"]
        ),
        zero_action_trajectory_sha256=str(
            reference_payload["input_sha256"]["zero_action_trajectory"]
        ),
    )


def _candidate_tie_key(row: Mapping[str, Any], score: float) -> tuple[float, ...]:
    parameters = row["physical_parameters"]
    return (
        score,
        float(parameters["init_spring_y"]),
        float(parameters["drag_damping"]),
        float(parameters["dashpot_damping"]),
    )


def fit_reusable_physics_selection(
    response_paths: list[str | Path],
    *,
    target_paths: Mapping[int | str, str | Path],
    robot_paths: Mapping[int | str, str | Path],
    protocol: Mapping[str, Any],
    trust_artifact_path: str | Path,
    object_id: str,
) -> dict[str, Any]:
    """Select one object-level physical tuple using only its six fit episodes."""

    split = protocol.get("splits", {}).get(object_id)
    _require(split is not None, "selection object is outside the fresh panel")
    fit_ids = tuple(int(value) for value in split["fit_episode_ids"])
    candidates = [
        _physical_parameters(value) for value in protocol["physical_candidates"]
    ]
    response_by_episode: dict[int, dict[int, dict[str, Any]]] = {
        episode_id: {} for episode_id in fit_ids
    }
    response_file_hashes: dict[str, str] = {}
    canonical_graph_identities: set[tuple[str, str]] = set()
    for value in response_paths:
        path = Path(value).resolve()
        payload = _load_json(path)
        validated = validate_reusable_physics_response(
            payload, protocol=protocol, verify_archive=True
        )
        _require(
            payload["object_id"] == object_id,
            "physical selection mixes objects",
        )
        episode_id = int(payload["episode_id"])
        candidate_index = int(validated["candidate_index"])
        _require(episode_id in response_by_episode, "response is not from a fit episode")
        _require(
            candidate_index not in response_by_episode[episode_id],
            "physical response is duplicated",
        )
        response_by_episode[episode_id][candidate_index] = payload
        response_file_hashes[
            f"{episode_id}/{candidate_index}"
        ] = sha256_file(path)
        if "execution" in protocol:
            graph = payload.get("canonical_graph", {})
            canonical_graph_identities.add(
                (
                    str(graph.get("file_sha256")),
                    str(graph.get("reusable_graph_sha256")),
                )
            )
    expected_count = len(fit_ids) * len(candidates)
    _require(
        len(response_file_hashes) == expected_count
        and all(
            set(responses) == set(range(len(candidates)))
            for responses in response_by_episode.values()
        ),
        "physical selection requires every frozen candidate on all six fit episodes",
    )
    if "execution" in protocol:
        _require(
            len(canonical_graph_identities) == 1,
            "physical selection rebuilt the canonical graph across episodes",
        )

    normalized_targets = {int(key): Path(value).resolve() for key, value in target_paths.items()}
    normalized_robots = {int(key): Path(value).resolve() for key, value in robot_paths.items()}
    _require(set(normalized_targets) == set(fit_ids), "fit target set changed")
    _require(set(normalized_robots) == set(fit_ids), "fit robot set changed")
    model = load_reusable_twin_trust_candidate(trust_artifact_path)
    expected_model = protocol["addendum"]["reference_trust_response"][
        "candidate_result_sha256"
    ]
    _require(model.result_sha256 == expected_model, "selection uses another trust model")
    reference_parameters = _physical_parameters(
        protocol["addendum"]["reference_trust_response"]
    )
    reference_index = candidates.index(reference_parameters)
    horizon = tuple(
        int(value) for value in protocol["addendum"]["selection"]["fit_horizon_half_open"]
    )

    episodes: dict[int, CausalTrustEpisode] = {}
    alpha_by_episode: dict[int, float] = {}
    trust_by_episode: dict[str, Any] = {}
    persistence_by_episode: dict[int, np.ndarray] = {}
    for episode_id in fit_ids:
        reference_payload = response_by_episode[episode_id][reference_index]
        reference_arrays = _load_response_arrays(reference_payload)
        reference_prediction = np.asarray(
            reference_arrays["prediction_m"], dtype=np.float64
        )
        persistence = np.asarray(reference_arrays["persistence_m"], dtype=np.float64)
        _require(
            reference_prediction.shape == persistence.shape,
            "reference prediction and persistence differ",
        )
        with np.load(normalized_robots[episode_id], allow_pickle=False) as robot:
            actions = np.asarray(robot["actions"], dtype=np.float64)
            openings = np.asarray(robot["openings"], dtype=np.float64)
        response = (reference_prediction - persistence) / model.reference_response_alpha
        features = build_deform360_trust_features(
            actions, openings, response, persistence
        )
        decision = model.decide(features)
        alpha_by_episode[episode_id] = float(decision.alpha)
        trust_by_episode[str(episode_id)] = {
            "alpha": float(decision.alpha),
            "raw_alpha": float(decision.raw_alpha),
            "closure_accepted": bool(decision.closure_accepted),
            "closure_value": float(decision.closure_value),
            "features": {
                name: float(features[name]) for name in model.feature_names
            },
        }
        episode_key = f"{object_id}/{episode_id}"
        episodes[episode_id] = _fit_episode(
            normalized_targets[episode_id],
            episode_key=episode_key,
            reference_payload=reference_payload,
        )
        persistence_by_episode[episode_id] = persistence

    candidate_table: list[dict[str, Any]] = []
    for candidate_index, parameters in enumerate(candidates):
        by_episode: dict[str, Any] = {}
        scores = []
        for episode_id in fit_ids:
            payload = response_by_episode[episode_id][candidate_index]
            arrays = _load_response_arrays(payload)
            candidate_prediction = np.asarray(arrays["prediction_m"], dtype=np.float64)
            persistence = np.asarray(arrays["persistence_m"], dtype=np.float64)
            _require(
                np.array_equal(persistence, persistence_by_episode[episode_id]),
                "persistence changes across physical candidates",
            )
            if alpha_by_episode[episode_id] == 0.0:
                trusted = persistence.copy()
                _require(np.array_equal(trusted, persistence), "zero trust is not exact")
            else:
                raw_response = (
                    candidate_prediction - persistence
                ) / model.reference_response_alpha
                trusted = persistence + alpha_by_episode[episode_id] * raw_response
            metrics = score_causal_trust_interval(
                episodes[episode_id], trusted, horizon[0], horizon[1]
            )
            scores.append(float(metrics["relative_score_vs_persistence"]))
            by_episode[str(episode_id)] = metrics
        candidate_table.append(
            {
                "candidate_index": candidate_index,
                "physical_parameters": parameters,
                "eligible": True,
                "pooled_relative_score_vs_persistence": float(np.mean(scores)),
                "by_episode": by_episode,
            }
        )

    def select(selected_ids: tuple[int, ...]) -> dict[str, Any]:
        return min(
            candidate_table,
            key=lambda row: _candidate_tie_key(
                row,
                float(
                    np.mean(
                        [
                            row["by_episode"][str(episode_id)][
                                "relative_score_vs_persistence"
                            ]
                            for episode_id in selected_ids
                        ]
                    )
                ),
            ),
        )

    pooled = select(fit_ids)
    leave_one_out = []
    for held_id in fit_ids:
        selected = select(tuple(value for value in fit_ids if value != held_id))
        leave_one_out.append(
            {
                "held_out_fit_episode_id": held_id,
                "selected_candidate_index": int(selected["candidate_index"]),
                "selected_physical_parameters": selected["physical_parameters"],
                "held_out_metrics": selected["by_episode"][str(held_id)],
            }
        )
    single = {
        str(episode_id): {
            "selected_candidate_index": int(select((episode_id,))["candidate_index"]),
            "selected_physical_parameters": select((episode_id,))[
                "physical_parameters"
            ],
        }
        for episode_id in fit_ids
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360ReusableTwinPhysicalSelection",
        "protocol_id": protocol["parent"]["protocol_id"],
        "physics_addendum_id": protocol["addendum"]["protocol_id"],
        "parent_file_sha256": protocol["parent_file_sha256"],
        "addendum_file_sha256": protocol["addendum_file_sha256"],
        "object_id": object_id,
        "topology": split["topology"],
        "fit_episode_ids": list(fit_ids),
        "candidate_count": len(candidates),
        "reference_candidate_index": reference_index,
        "selected_candidate_index": int(pooled["candidate_index"]),
        "selected_physical_parameters": pooled["physical_parameters"],
        "selected_pooled_relative_score_vs_persistence": pooled[
            "pooled_relative_score_vs_persistence"
        ],
        "trust_by_episode": trust_by_episode,
        "leave_one_fit_episode_out": leave_one_out,
        "single_fit_episode_controls": single,
        "candidate_table": candidate_table,
        "input_sha256": {
            "trust_artifact": sha256_file(trust_artifact_path),
            "responses": dict(sorted(response_file_hashes.items())),
            "targets": {
                str(key): sha256_file(path)
                for key, path in sorted(normalized_targets.items())
            },
            "robots": {
                str(key): sha256_file(path)
                for key, path in sorted(normalized_robots.items())
            },
        },
        "information_boundary": {
            "fit_outcomes_used": list(fit_ids),
            "held_out_actions_read": False,
            "held_out_initial_geometry_read": False,
            "held_out_outcomes_read": False,
            "trust_inferred_only_from_fixed_reference_response": True,
            "candidate_physics_cannot_change_trust": True,
        },
        "claim_boundary": (
            "object-level fit only; held prediction and evaluation remain sealed"
        ),
    }
    if "execution" in protocol:
        graph_file_sha256, reusable_graph_sha256 = next(
            iter(canonical_graph_identities)
        )
        payload.update(
            {
                "execution_file_sha256": protocol["execution_file_sha256"],
                "canonical_graph": {
                    "file_sha256": graph_file_sha256,
                    "reusable_graph_sha256": reusable_graph_sha256,
                    "shared_across_all_fit_and_held_episodes": True,
                },
            }
        )
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def validate_reusable_physics_selection(
    payload: Mapping[str, Any], *, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a frozen object-level physical selection."""

    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_kind")
        == "Deform360ReusableTwinPhysicalSelection",
        "reusable physical selection identity changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "reusable physical selection checksum changed",
    )
    _require(
        payload.get("parent_file_sha256") == protocol["parent_file_sha256"]
        and payload.get("addendum_file_sha256") == protocol["addendum_file_sha256"],
        "physical selection uses another protocol",
    )
    if "execution" in protocol:
        graph = payload.get("canonical_graph", {})
        _require(
            payload.get("execution_file_sha256")
            == protocol["execution_file_sha256"]
            and graph.get("shared_across_all_fit_and_held_episodes") is True
            and isinstance(graph.get("file_sha256"), str)
            and isinstance(graph.get("reusable_graph_sha256"), str),
            "physical selection uses another reusable execution",
        )
    object_id = str(payload.get("object_id"))
    split = protocol.get("splits", {}).get(object_id)
    _require(split is not None, "physical selection object is outside the panel")
    _require(
        payload.get("fit_episode_ids") == list(split["fit_episode_ids"]),
        "physical selection fit episodes changed",
    )
    parameters = _physical_parameters(payload.get("selected_physical_parameters", {}))
    index = _candidate_index(protocol, parameters)
    _require(
        int(payload.get("selected_candidate_index", -1)) == index,
        "selected physical tuple identity changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("held_out_actions_read") is False
        and boundary.get("held_out_initial_geometry_read") is False
        and boundary.get("held_out_outcomes_read") is False
        and boundary.get("trust_inferred_only_from_fixed_reference_response") is True
        and boundary.get("candidate_physics_cannot_change_trust") is True,
        "physical selection crossed the held-out boundary",
    )
    return {
        "object_id": object_id,
        "selected_candidate_index": index,
        "selected_physical_parameters": parameters,
        "result_sha256": str(payload["result_sha256"]),
    }


__all__ = [
    "REUSABLE_PHYSICS_RESPONSE_SCHEMA_VERSION",
    "build_reusable_physics_fit_grid_seal",
    "fit_reusable_physics_selection",
    "seal_reusable_physics_response",
    "validate_reusable_physics_fit_grid_seal",
    "validate_reusable_physics_selection",
    "validate_reusable_physics_response",
]
