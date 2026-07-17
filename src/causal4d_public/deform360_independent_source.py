"""Prediction-first independent-source gate for Deform360 graph action support."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_action_support import (
    graph_contact_distance_m,
    graph_readout_action_support,
)
from .deform360_phystwin_trust import (
    CausalTrustEpisode,
    score_causal_trust_interval,
)
from .deform360_reusable_graph import load_canonical_deform360_graph


INDEPENDENT_SOURCE_SCHEMA_VERSION = 1
INDEPENDENT_SOURCE_PROTOCOL_ID = "deform360-graph-action-support-independent-source-v1"
EXPECTED_INDEPENDENT_SOURCE_EPISODES = {
    "002-rope-silk": (2, 5, 6, 7, 9),
    "085-scarf-cloth": (3, 4, 6, 8, 9),
    "083-blanket-cloth": (1, 2, 4, 5, 8, 9),
    "092-squirrel": (4, 5, 7, 8, 9),
    "170-spider": (0, 1, 3, 5, 8, 9),
}
DISCOVERY_EPISODES = {
    "002-rope-silk/0",
    "085-scarf-cloth/1",
    "092-squirrel/0",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_independent_source_lock(path: str | Path) -> dict[str, Any]:
    """Load and validate the prospectively frozen 27-episode source gate."""

    lock_path = Path(path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "independent-source lock must be an object")
    _require(
        payload.get("schema_version") == INDEPENDENT_SOURCE_SCHEMA_VERSION,
        "independent-source schema changed",
    )
    _require(
        payload.get("protocol_id") == INDEPENDENT_SOURCE_PROTOCOL_ID,
        "independent-source protocol id changed",
    )
    panel = payload.get("independent_source_panel", {})
    observed = {
        str(object_id): tuple(int(value) for value in episode_ids)
        for object_id, episode_ids in panel.get("episodes_by_object", {}).items()
    }
    _require(
        observed == EXPECTED_INDEPENDENT_SOURCE_EPISODES,
        "independent-source episode panel changed",
    )
    _require(
        int(panel.get("episode_count", -1))
        == sum(len(values) for values in observed.values())
        == 27,
        "independent-source episode count changed",
    )
    _require(
        set(panel.get("discovery_episodes_excluded", ())) == DISCOVERY_EPISODES,
        "discovery exclusion changed",
    )
    _require(
        panel.get("prediction_must_be_hashed_before_future_outcome_scoring") is True
        and panel.get("post_initial_object_observations_used_for_prediction") is False
        and panel.get("all_episodes_must_be_scored") is True,
        "prediction-first source boundary changed",
    )
    predictor = payload.get("frozen_predictor", {})
    prediction_input = predictor.get("prediction_input", {})
    _require(
        int(predictor.get("maximum_observed_graph_node_count", -1)) == 384
        and int(predictor.get("minimum_observed_graph_node_count", -1)) == 128,
        "observed graph capacity changed",
    )
    _require(
        predictor.get("future_object_observations_used_for_twin") is False
        and prediction_input.get("future_object_tracks_present") is False
        and prediction_input.get("future_object_visibility_present") is False
        and prediction_input.get("future_tactile_used") is False
        and prediction_input.get(
            "driven_and_zero_predictions_hashed_before_future_scoring"
        )
        is True,
        "prediction input contains forbidden future evidence",
    )
    support = predictor.get("graph_action_support", {})
    _require(
        float(support.get("length_scale_m", -1.0)) == 0.12
        and float(support.get("action_response", -1.0)) == 0.9
        and float(support.get("autonomous_drift_response", -1.0)) == 0.0,
        "frozen graph action-support predictor changed",
    )
    dynamics = predictor.get("warp_dynamics", {})
    _require(
        predictor.get("official_phystwin_revision")
        == "2b6630528141b9cba5a7677c8b88b2129b4a8390"
        and dynamics
        == {
            "config_sha256": (
                "a40a5ec2f5c978c1290810f20ed56db7cab99dc0c227adfe6b7434dfc95ead48"
            ),
            "init_spring_y": 10000.0,
            "drag_damping": 10.0,
            "dashpot_damping": 100.0,
            "controller_radius_m": 0.03,
            "controller_max_neighbours": 1,
            "canonical_controller_patch_size": 16,
            "driven_controller_displacement_scale": 1.0,
            "zero_controller_displacement_scale": 0.0,
            "support_dynamics": "official-ground",
        },
        "frozen Warp dynamics changed",
    )
    return payload


def authorize_independent_source_episode(
    lock: Mapping[str, Any], object_id: str, episode_id: int
) -> dict[str, Any]:
    """Reject discovery, calibration, and target episodes before any I/O."""

    expected = EXPECTED_INDEPENDENT_SOURCE_EPISODES.get(str(object_id))
    _require(expected is not None, "object is outside the independent source panel")
    episode = int(episode_id)
    _require(
        episode in expected,
        "episode is not authorized for independent-source evaluation",
    )
    _require(
        f"{object_id}/{episode}" not in DISCOVERY_EPISODES,
        "discovery episode cannot enter the independent source gate",
    )
    _require(
        lock.get("protocol_id") == INDEPENDENT_SOURCE_PROTOCOL_ID,
        "authorization uses another protocol",
    )
    return {
        "protocol_id": INDEPENDENT_SOURCE_PROTOCOL_ID,
        "object_id": str(object_id),
        "episode_id": episode,
        "episode_key": f"{object_id}/{episode}",
    }


def validate_prediction_only_bundle(
    payload: Mapping[str, Any],
    *,
    object_id: str | None = None,
    episode_id: int | None = None,
) -> dict[str, Any]:
    """Validate a PhysTwin input that contains no future object observation."""

    marker = payload.get("prediction_only_input", {})
    _require(
        isinstance(marker, Mapping)
        and marker.get("schema_version") == 1
        and marker.get("future_object_observations_present") is False
        and marker.get("object_observation_frames_used") == [0]
        and marker.get("known_future_robot_trajectory_used") is True
        and marker.get("future_tactile_used") is False,
        "PhysTwin bundle is not prediction-only",
    )
    if object_id is not None:
        _require(marker.get("object_id") == object_id, "prediction object differs")
    if episode_id is not None:
        _require(
            int(marker.get("episode_id", -1)) == int(episode_id),
            "prediction episode differs",
        )
    points = np.asarray(payload.get("object_points"))
    colors = np.asarray(payload.get("object_colors"))
    visibility = np.asarray(payload.get("object_visibilities"), dtype=bool)
    validity = np.asarray(payload.get("object_motions_valid"), dtype=bool)
    controllers = np.asarray(payload.get("controller_points"))
    _require(
        points.ndim == 3 and points.shape[0] >= 2 and points.shape[2] == 3,
        "prediction object points must have shape (T,N,3)",
    )
    _require(colors.shape == points.shape, "prediction colors differ from points")
    _require(
        visibility.shape == points.shape[:2] and validity.shape == points.shape[:2],
        "prediction object masks differ from points",
    )
    _require(
        controllers.ndim == 3
        and controllers.shape[0] == points.shape[0]
        and controllers.shape[2] == 3,
        "prediction controller trajectory differs from object frame axis",
    )
    _require(
        np.array_equal(points, np.repeat(points[:1], len(points), axis=0))
        and np.array_equal(colors, np.repeat(colors[:1], len(colors), axis=0))
        and np.array_equal(
            visibility, np.repeat(visibility[:1], len(visibility), axis=0)
        )
        and np.array_equal(validity, np.repeat(validity[:1], len(validity), axis=0)),
        "prediction bundle contains changing future object observations",
    )
    _require(
        np.all(np.isfinite(points))
        and np.all(np.isfinite(colors))
        and np.all(np.isfinite(controllers)),
        "prediction bundle is non-finite",
    )
    return {
        "frame_count": int(points.shape[0]),
        "point_count": int(points.shape[1]),
        "controller_point_count": int(controllers.shape[1]),
        "frame_zero_points_sha256": sha256_array(points[0]),
        "controller_trajectory_sha256": sha256_array(controllers),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path.name} must contain an object")
    return value


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    _require(isinstance(value, Mapping), f"{path.name} must contain a mapping")
    return value


def seal_independent_source_prediction(
    archive_path: str | Path,
    *,
    lock_path: str | Path,
    object_id: str,
    episode_id: int,
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    readout_path: str | Path,
    twin_summary_path: str | Path,
    driven_result_path: str | Path,
    zero_result_path: str | Path,
) -> dict[str, Any]:
    """Build and hash the deployable prediction before a future target is read."""

    lock_file = Path(lock_path).resolve()
    lock = load_independent_source_lock(lock_file)
    authorization = authorize_independent_source_episode(lock, object_id, episode_id)
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
    _require(twin.get("passed") is True, "automatic twin failed admission")
    _require(
        twin.get("object_id") == object_id
        and int(twin.get("episode_id", -1)) == int(episode_id),
        "automatic twin belongs to another episode",
    )
    _require(
        twin.get("information_boundary", {}).get("target_access") is False
        and twin.get("information_boundary", {}).get(
            "post_initial_object_observation_used"
        )
        is False,
        "automatic twin crossed the prediction boundary",
    )
    data_sha = sha256_file(data_file)
    simulator_sha = sha256_file(simulator_file)
    graph_sha = sha256_file(graph_file)
    readout_sha = sha256_file(readout_file)
    _require(
        twin.get("input_sha256", {}).get("episode_final_data") == data_sha,
        "automatic twin uses another prediction-only bundle",
    )
    _require(
        twin.get("output_sha256", {}).get("simulator_final_data") == simulator_sha,
        "automatic twin simulator bundle hash changed",
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

    results = {"driven": _load_json(driven_file), "zero_action": _load_json(zero_file)}
    trajectories: dict[str, np.ndarray] = {}
    trajectory_files: dict[str, Path] = {}
    expected_revision = lock["frozen_predictor"]["official_phystwin_revision"]
    dynamics = lock["frozen_predictor"]["warp_dynamics"]
    expected_overrides = {
        "controller_max_neighbours": dynamics["controller_max_neighbours"],
        "controller_radius": dynamics["controller_radius_m"],
        "dashpot_damping": dynamics["dashpot_damping"],
        "drag_damping": dynamics["drag_damping"],
        "init_spring_Y": dynamics["init_spring_y"],
    }
    for name, result_file in (("driven", driven_file), ("zero_action", zero_file)):
        result = results[name]
        _require(result.get("passed") is True, f"{name} Warp rollout failed")
        _require(
            "external_target_scoring" not in result,
            f"{name} Warp rollout read the future target before sealing",
        )
        _require(
            result.get("data_sha256") == simulator_sha,
            f"{name} simulator data hash differs",
        )
        _require(
            result.get("official_phystwin_revision") == expected_revision,
            f"{name} PhysTwin revision differs from the lock",
        )
        _require(
            result.get("config_sha256") == dynamics["config_sha256"]
            and result.get("config_overrides") == expected_overrides,
            f"{name} Warp dynamics differ from the lock",
        )
        _require(
            result.get("support_dynamics", {}).get("mode")
            == dynamics["support_dynamics"],
            f"{name} support dynamics differ from the lock",
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
            == int(dynamics["canonical_controller_patch_size"]),
            f"{name} controller patch differs from the lock",
        )
        trajectory_file = result_file.with_name("official_phystwin_trajectory.npz")
        _require(
            result.get("trajectory_sha256") == sha256_file(trajectory_file),
            f"{name} trajectory hash changed",
        )
        with np.load(trajectory_file, allow_pickle=False) as stored:
            trajectory = np.asarray(stored["vertices"], dtype=np.float64)
        _require(
            trajectory.ndim == 3
            and trajectory.shape[0] == bundle["frame_count"]
            and trajectory.shape[1] >= len(graph.vertices)
            and trajectory.shape[2] == 3
            and np.all(np.isfinite(trajectory)),
            f"{name} trajectory is invalid",
        )
        trajectory_files[name] = trajectory_file
        trajectories[name] = trajectory[:, : len(graph.vertices)]
    _require(
        float(
            results["driven"]
            .get("realized_actuation", {})
            .get("controller_displacement_scale", -1.0)
        )
        == float(dynamics["driven_controller_displacement_scale"]),
        "driven rollout does not use the frozen action",
    )
    _require(
        float(
            results["zero_action"]
            .get("realized_actuation", {})
            .get("controller_displacement_scale", -1.0)
        )
        == float(dynamics["zero_controller_displacement_scale"]),
        "zero-action rollout moves the controller",
    )
    _require(
        results["driven"].get("config_sha256")
        == results["zero_action"].get("config_sha256"),
        "matched rollouts use different PhysTwin configs",
    )
    _require(
        results["driven"].get("split_sha256")
        == results["zero_action"].get("split_sha256"),
        "matched rollouts use different frame splits",
    )
    predictor = lock["frozen_predictor"]
    maximum_nodes = int(predictor["maximum_observed_graph_node_count"])
    minimum_nodes = int(predictor["minimum_observed_graph_node_count"])
    observed_nodes = int(graph.observed_node_count)
    _require(
        minimum_nodes <= observed_nodes <= maximum_nodes,
        "observed graph capacity is outside the frozen range",
    )
    distance = graph_contact_distance_m(graph)
    length_scale = float(predictor["graph_action_support"]["length_scale_m"])
    action_response = float(predictor["graph_action_support"]["action_response"])
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
    prediction = initial[None] + (
        action_response * support[None, :, None] * (driven_readout - zero_readout)
    )
    persistence = np.repeat(initial[None], bundle["frame_count"], axis=0)
    _require(np.all(np.isfinite(prediction)), "sealed prediction is non-finite")

    archive_file = Path(archive_path).resolve()
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive_file,
        prediction_m=prediction.astype(np.float32),
        persistence_m=persistence.astype(np.float32),
        driven_readout_m=driven_readout.astype(np.float32),
        zero_action_readout_m=zero_readout.astype(np.float32),
        action_support=support.astype(np.float32),
        frame_zero_points_m=initial.astype(np.float32),
    )
    arrays = {
        "prediction_m": sha256_array(prediction.astype(np.float32)),
        "persistence_m": sha256_array(persistence.astype(np.float32)),
        "driven_readout_m": sha256_array(driven_readout.astype(np.float32)),
        "zero_action_readout_m": sha256_array(zero_readout.astype(np.float32)),
        "action_support": sha256_array(support.astype(np.float32)),
        "frame_zero_points_m": sha256_array(initial.astype(np.float32)),
    }
    payload = {
        "schema_version": INDEPENDENT_SOURCE_SCHEMA_VERSION,
        "artifact_kind": "Deform360IndependentSourcePredictionSeal",
        **authorization,
        "lock_sha256": sha256_file(lock_file),
        "frozen_predictor": {
            "length_scale_m": length_scale,
            "action_response": action_response,
            "autonomous_drift_response": 0.0,
            "observed_graph_node_count": observed_nodes,
            "total_graph_node_count": len(graph.vertices),
            "frame_count": bundle["frame_count"],
            "point_count": bundle["point_count"],
        },
        "prediction_archive": {
            "path": str(archive_file),
            "file_sha256": sha256_file(archive_file),
            "array_sha256": arrays,
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
            "zero_action_trajectory": sha256_file(trajectory_files["zero_action"]),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_used": True,
            "future_object_track_read": False,
            "future_object_visibility_read": False,
            "future_tactile_read": False,
            "external_target_scoring_in_warp": False,
            "prediction_hashed_before_future_outcome_scoring": True,
            "calibration_outcome_read": False,
            "target_outcome_read": False,
        },
        "claim_boundary": (
            "deployable independent-source prediction sealed before its public "
            "future outcome is scored"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def validate_independent_source_prediction_seal(
    payload: Mapping[str, Any], *, verify_archive: bool = False
) -> dict[str, Any]:
    """Validate a prediction seal and optionally re-hash every stored array."""

    _require(
        payload.get("schema_version") == INDEPENDENT_SOURCE_SCHEMA_VERSION
        and payload.get("artifact_kind") == "Deform360IndependentSourcePredictionSeal"
        and payload.get("protocol_id") == INDEPENDENT_SOURCE_PROTOCOL_ID,
        "independent-source prediction seal identity changed",
    )
    _require(
        payload.get("result_sha256") == _result_sha256(payload),
        "independent-source prediction seal checksum changed",
    )
    _require(
        _valid_sha256(payload.get("lock_sha256")),
        "prediction seal has no lock checksum",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("future_object_track_read") is False
        and boundary.get("future_object_visibility_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("external_target_scoring_in_warp") is False
        and boundary.get("prediction_hashed_before_future_outcome_scoring") is True,
        "prediction seal crossed the future-outcome boundary",
    )
    if verify_archive:
        archive = payload.get("prediction_archive", {})
        path = Path(str(archive.get("path", "")))
        _require(path.is_file(), "prediction archive is missing")
        _require(
            sha256_file(path) == archive.get("file_sha256"),
            "prediction archive checksum changed",
        )
        expected = archive.get("array_sha256", {})
        with np.load(path, allow_pickle=False) as stored:
            _require(set(stored.files) == set(expected), "prediction arrays changed")
            for name in stored.files:
                _require(
                    sha256_array(stored[name]) == expected[name],
                    f"prediction array {name} checksum changed",
                )
    return {
        "passed": True,
        "episode_key": payload.get("episode_key"),
        "result_sha256": payload.get("result_sha256"),
    }


def evaluate_independent_source_prediction(
    prediction_seal: Mapping[str, Any],
    target_data_path: str | Path,
    *,
    lock_path: str | Path,
) -> dict[str, Any]:
    """Open and score one public source future only after prediction sealing."""

    validate_independent_source_prediction_seal(prediction_seal, verify_archive=True)
    lock_file = Path(lock_path).resolve()
    lock = load_independent_source_lock(lock_file)
    _require(
        prediction_seal.get("lock_sha256") == sha256_file(lock_file),
        "prediction was sealed under another lock",
    )
    authorize_independent_source_episode(
        lock,
        str(prediction_seal["object_id"]),
        int(prediction_seal["episode_id"]),
    )
    target_file = Path(target_data_path).resolve()
    target = _load_pickle(target_file)
    points = np.asarray(target["object_points"], dtype=np.float64)
    visibility = np.asarray(target["object_visibilities"], dtype=bool)
    validity = np.asarray(target["object_motions_valid"], dtype=bool)
    archive_path = Path(prediction_seal["prediction_archive"]["path"])
    with np.load(archive_path, allow_pickle=False) as stored:
        prediction = np.asarray(stored["prediction_m"], dtype=np.float64)
        persistence = np.asarray(stored["persistence_m"], dtype=np.float64)
        frame_zero = np.asarray(stored["frame_zero_points_m"], dtype=np.float64)
        driven = np.asarray(stored["driven_readout_m"], dtype=np.float64)
        zero = np.asarray(stored["zero_action_readout_m"], dtype=np.float64)
    _require(
        points.shape == prediction.shape == persistence.shape,
        "target and sealed prediction shapes differ",
    )
    _require(
        visibility.shape == points.shape[:2] and validity.shape == points.shape[:2],
        "target masks differ from target points",
    )
    _require(
        np.array_equal(points[0].astype(np.float32), frame_zero.astype(np.float32)),
        "target frame-zero identities differ from the sealed prediction",
    )
    _require(
        np.array_equal(persistence, np.repeat(frame_zero[None], len(points), axis=0)),
        "sealed persistence baseline changed",
    )
    episode = CausalTrustEpisode(
        episode_id=str(prediction_seal["episode_key"]),
        target_m=points,
        visibility=visibility,
        validity=validity,
        driven_m=driven,
        zero_action_m=zero,
        train_stop_frame=60,
        source_data_sha256=sha256_file(target_file),
        driven_trajectory_sha256=str(
            prediction_seal["input_sha256"]["driven_trajectory"]
        ),
        zero_action_trajectory_sha256=str(
            prediction_seal["input_sha256"]["zero_action_trajectory"]
        ),
    )
    frame_count = len(points)
    _require(frame_count == 76, "independent source target must have 76 frames")
    intervals = {
        "future": (1, 76),
        "early": (1, 26),
        "middle": (26, 51),
        "late": (51, 76),
    }
    metrics: dict[str, Any] = {}
    for name, (start, stop) in intervals.items():
        scored = score_causal_trust_interval(episode, prediction, start, stop)
        scored["track_improvement_fraction"] = 1.0 - (
            float(scored["track_rmse_m"]) / float(scored["persistence_track_rmse_m"])
        )
        scored["chamfer_improvement_fraction"] = 1.0 - (
            float(scored["chamfer_m"]) / float(scored["persistence_chamfer_m"])
        )
        metrics[name] = scored
    payload = {
        "schema_version": INDEPENDENT_SOURCE_SCHEMA_VERSION,
        "artifact_kind": "Deform360IndependentSourceEpisodeEvaluation",
        "protocol_id": INDEPENDENT_SOURCE_PROTOCOL_ID,
        "object_id": prediction_seal["object_id"],
        "episode_id": int(prediction_seal["episode_id"]),
        "episode_key": prediction_seal["episode_key"],
        "prediction_seal_sha256": prediction_seal["result_sha256"],
        "target_data_sha256": sha256_file(target_file),
        "metrics": metrics,
        "joint_future_win": bool(
            metrics["future"]["track_improvement_fraction"] > 0.0
            and metrics["future"]["chamfer_improvement_fraction"] > 0.0
        ),
        "information_boundary": {
            "deployable_prediction_previously_sealed": True,
            "source_future_opened_for_scoring": True,
            "calibration_outcome_read": False,
            "target_outcome_read": False,
        },
        "claim_boundary": (
            "independent public source evaluation; the 27-episode conjunctive "
            "gate, calibration gate, and official targets remain outstanding"
        ),
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


def _balanced_improvement(
    rows: Sequence[Mapping[str, Any]], interval: str, metric: str
) -> float:
    predicted = np.asarray([float(row["metrics"][interval][metric]) for row in rows])
    baseline = np.asarray(
        [float(row["metrics"][interval][f"persistence_{metric}"]) for row in rows]
    )
    return float(1.0 - np.mean(predicted) / np.mean(baseline))


def aggregate_independent_source_gate(
    evaluations: Sequence[Mapping[str, Any]], *, lock_path: str | Path
) -> dict[str, Any]:
    """Apply every predeclared source transfer gate conjunctively."""

    lock_file = Path(lock_path).resolve()
    lock = load_independent_source_lock(lock_file)
    rows = tuple(evaluations)
    expected_keys = {
        f"{object_id}/{episode_id}"
        for object_id, episode_ids in EXPECTED_INDEPENDENT_SOURCE_EPISODES.items()
        for episode_id in episode_ids
    }
    observed_keys = {str(row.get("episode_key")) for row in rows}
    _require(len(rows) == len(observed_keys), "independent evaluations repeat")
    _require(
        observed_keys == expected_keys,
        "independent source gate requires exactly the locked 27 episodes",
    )
    for row in rows:
        _require(
            row.get("artifact_kind") == "Deform360IndependentSourceEpisodeEvaluation"
            and row.get("protocol_id") == INDEPENDENT_SOURCE_PROTOCOL_ID
            and row.get("result_sha256") == _result_sha256(row),
            "independent episode evaluation is invalid",
        )
    panel = lock["independent_source_panel"]
    track = _balanced_improvement(rows, "future", "track_rmse_m")
    chamfer = _balanced_improvement(rows, "future", "chamfer_m")
    late_track = _balanced_improvement(rows, "late", "track_rmse_m")
    late_chamfer = _balanced_improvement(rows, "late", "chamfer_m")
    joint_wins = sum(bool(row["joint_future_win"]) for row in rows)
    by_object: dict[str, Any] = {}
    no_object_median_degradation = True
    maximum_degradation = {"track": -np.inf, "chamfer": -np.inf}
    per_object_win_gate = True
    for object_id in EXPECTED_INDEPENDENT_SOURCE_EPISODES:
        selected = [row for row in rows if row["object_id"] == object_id]
        track_changes = np.asarray(
            [
                -float(row["metrics"]["future"]["track_improvement_fraction"])
                for row in selected
            ]
        )
        chamfer_changes = np.asarray(
            [
                -float(row["metrics"]["future"]["chamfer_improvement_fraction"])
                for row in selected
            ]
        )
        wins = sum(bool(row["joint_future_win"]) for row in selected)
        win_fraction = wins / len(selected)
        medians_non_degrading = bool(
            np.median(track_changes) <= 0.0 and np.median(chamfer_changes) <= 0.0
        )
        no_object_median_degradation &= medians_non_degrading
        per_object_win_gate &= win_fraction >= float(
            panel["minimum_joint_win_fraction_per_object"]
        )
        maximum_degradation["track"] = max(
            maximum_degradation["track"], float(np.max(track_changes))
        )
        maximum_degradation["chamfer"] = max(
            maximum_degradation["chamfer"], float(np.max(chamfer_changes))
        )
        by_object[object_id] = {
            "episode_count": len(selected),
            "joint_win_count": wins,
            "joint_win_fraction": win_fraction,
            "median_track_change_fraction": float(np.median(track_changes)),
            "median_chamfer_change_fraction": float(np.median(chamfer_changes)),
            "median_non_degradation_passed": medians_non_degrading,
        }
    gates = {
        "execution_balanced_track": track
        >= float(panel["minimum_execution_balanced_track_improvement_fraction"]),
        "execution_balanced_chamfer": chamfer
        >= float(panel["minimum_execution_balanced_chamfer_improvement_fraction"]),
        "late_track": late_track
        >= float(panel["minimum_late_track_improvement_fraction"]),
        "late_chamfer": late_chamfer
        >= float(panel["minimum_late_chamfer_improvement_fraction"]),
        "joint_win_count": joint_wins >= int(panel["minimum_joint_win_episode_count"]),
        "per_object_joint_win_fraction": per_object_win_gate,
        "no_object_median_degradation": no_object_median_degradation,
        "maximum_per_episode_track_degradation": maximum_degradation["track"]
        <= float(panel["maximum_per_episode_degradation_fraction_per_metric"]),
        "maximum_per_episode_chamfer_degradation": maximum_degradation["chamfer"]
        <= float(panel["maximum_per_episode_degradation_fraction_per_metric"]),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": INDEPENDENT_SOURCE_SCHEMA_VERSION,
        "artifact_kind": "Deform360IndependentSourceGate",
        "protocol_id": INDEPENDENT_SOURCE_PROTOCOL_ID,
        "lock_sha256": sha256_file(lock_file),
        "episode_count": len(rows),
        "metrics": {
            "execution_balanced_track_improvement_fraction": track,
            "execution_balanced_chamfer_improvement_fraction": chamfer,
            "late_track_improvement_fraction": late_track,
            "late_chamfer_improvement_fraction": late_chamfer,
            "joint_win_episode_count": joint_wins,
            "maximum_per_episode_track_degradation_fraction": maximum_degradation[
                "track"
            ],
            "maximum_per_episode_chamfer_degradation_fraction": maximum_degradation[
                "chamfer"
            ],
            "by_object": by_object,
        },
        "gates": gates,
        "passed": passed,
        "next_step": (
            "freeze implementation and open registered calibration outcomes"
            if passed
            else "freeze source failure and keep calibration and targets sealed"
        ),
        "evaluation_result_sha256": {
            str(row["episode_key"]): str(row["result_sha256"])
            for row in sorted(rows, key=lambda value: str(value["episode_key"]))
        },
        "information_boundary": {
            "independent_source_outcomes_read": True,
            "calibration_outcomes_read": False,
            "target_initial_frames_read": False,
            "target_actions_read": False,
            "target_outcomes_read": False,
        },
    }
    payload["result_sha256"] = _result_sha256(payload)
    return payload


__all__ = [
    "DISCOVERY_EPISODES",
    "EXPECTED_INDEPENDENT_SOURCE_EPISODES",
    "INDEPENDENT_SOURCE_PROTOCOL_ID",
    "aggregate_independent_source_gate",
    "authorize_independent_source_episode",
    "evaluate_independent_source_prediction",
    "load_independent_source_lock",
    "seal_independent_source_prediction",
    "sha256_array",
    "sha256_file",
    "validate_independent_source_prediction_seal",
    "validate_prediction_only_bundle",
]
