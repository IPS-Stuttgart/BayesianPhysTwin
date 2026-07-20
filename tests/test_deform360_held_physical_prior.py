from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_phystwin.deform360_held_physical_prior as physical_prior

from deform360_held_test_helpers import (
    default_frame_zero_config,
    dummy_immutable_bindings,
)

from bayesian_phystwin.deform360_held_physical_prior import (
    AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256,
    AUTOMATIC_TWIN_PROTOCOL_ID,
    CANONICAL_NODE_COUNT,
    FRAME_COUNT,
    OFFICIAL_PHYSTWIN_REVISION,
    OFFICIAL_REAL_CONFIG_SHA256,
    WARP_DYNAMICS,
    build_persistence_fallback_archive,
    build_physical_prediction_archive,
    build_prediction_only_artifacts,
    run_held_physical_prior,
    select_action_window,
    sha256_file,
    validate_physical_prediction_manifest,
    validate_python_runtime,
)
from bayesian_phystwin.deform360_held_protocol import (
    create_held_protocol_lock,
    held_artifact_sha256,
)


CASE_NAME = "083-blanket-cloth-ep0000"


def test_v2_uses_fixed_1024_node_capacity() -> None:
    assert CANONICAL_NODE_COUNT == 1024
    assert physical_prior.HELD_PHYSICAL_NUMERIC_CONTRACT["contract_id"].endswith("-v2")
    assert physical_prior.HELD_PHYSICAL_NUMERIC_CONTRACT["canonical_node_count"] == 1024


def test_physical_runner_rejects_lock_numeric_contract_mismatch(tmp_path: Path) -> None:
    bindings = dummy_immutable_bindings()
    bindings["held_physical_numeric_contract"] = "f" * 64
    lock_path = tmp_path / "mismatched-lock.json"
    create_held_protocol_lock(lock_path, immutable_bindings=bindings)

    with pytest.raises(ValueError, match="physical numeric contract"):
        run_held_physical_prior(
            tmp_path / "unread-frame-zero.json",
            lock_path,
            tmp_path / "output",
            case_name=CASE_NAME,
            role="calibration",
            upstream_repo=tmp_path / "upstream",
            official_phystwin_repo=tmp_path / "phystwin",
            official_config=tmp_path / "real.yaml",
            deform360_repo=tmp_path / "deform360",
            python=tmp_path / "python",
        )

    assert not (tmp_path / "output").exists()


def _bound_file(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_robot(path: Path, frame_count: int = FRAME_COUNT) -> None:
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], frame_count, axis=0)
    poses[:, 0, 3] = np.linspace(0.0, 0.02, frame_count)
    actions = np.zeros((frame_count, 5, 3), dtype=np.float64)
    actions[:, 0] = poses[:, :3, 3]
    actions[:, 1:4] = poses[:, :3, :3]
    openings = np.linspace(0.10, 0.05, frame_count)
    actions[:, 4, 0] = openings
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.uint16),
        actions=actions,
        T_worlds=poses,
        openings=openings,
        bimanual=np.asarray(False, dtype=np.bool_),
    )


def _make_frame_zero_bundle(
    path: Path, *, encoded_frames: tuple[int, ...] = (0,)
) -> None:
    camera_count = 2
    point_count = 128
    points = np.column_stack(
        (
            np.linspace(0.0, 0.1, point_count),
            np.zeros(point_count),
            np.ones(point_count) * 0.2,
        )
    ).astype(np.float32)
    colors = np.tile(np.array([[0.2, 0.4, 0.6]], dtype=np.float32), (point_count, 1))
    rgb = np.zeros((camera_count, 2, 3, 3), dtype=np.uint8)
    mask = np.ones(rgb.shape[:3], dtype=bool)
    depth = np.ones(rgb.shape[:3], dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float64)[None], camera_count, axis=0)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], camera_count, axis=0)
    projections = np.repeat(
        np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)[None],
        camera_count,
        axis=0,
    )
    np.savez_compressed(
        path,
        frame_indices=np.asarray(encoded_frames, dtype=np.int64),
        camera_names=np.asarray(["cam0", "cam1"]),
        rgb_frame0=rgb,
        mask_frame0=mask,
        depth_frame0_m=depth,
        depth_valid_frame0=mask,
        intrinsics=intrinsics,
        camera_to_world=poses,
        projection_world_to_pixel=projections,
        object_points_world_m=points,
        object_colors_rgb=colors,
        object_color_support_count=np.ones(point_count, dtype=np.uint8) * 2,
        visual_hull_points_world_m=points,
    )


def _make_locked_frame_zero(
    tmp_path: Path, *, encoded_frames: tuple[int, ...] = (0,)
) -> tuple[Path, Path]:
    lock_path = tmp_path / "held_lock.json"
    create_held_protocol_lock(lock_path, immutable_bindings=dummy_immutable_bindings())
    bundle_path = tmp_path / "frame_zero.npz"
    _make_frame_zero_bundle(bundle_path, encoded_frames=encoded_frames)
    robot_path = tmp_path / "robot.npz"
    _make_robot(robot_path, frame_count=100)
    selected_robot_path = tmp_path / "known_action_76.npz"
    _make_robot(selected_robot_path)
    robot_metadata_path = tmp_path / "robot.meta.json"
    robot_metadata_path.write_text("{}\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldFrameZeroBundle",
        "protocol_id": "deform360-held-online-belief-v1",
        "case_name": CASE_NAME,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "role": "calibration",
        "frame_indices": [0],
        "lock_sha256": sha256_file(lock_path),
        "lock_artifact_sha256": json.loads(lock_path.read_text(encoding="utf-8"))[
            "artifact_sha256"
        ],
        "config": default_frame_zero_config(),
        "bundle": _bound_file(bundle_path),
        "action_inputs": {
            "robot_trajectory": _bound_file(robot_path),
            "robot_metadata": _bound_file(robot_metadata_path),
        },
        "action_alignment": {
            "selected_raw_frame_range_half_open": [8, 89],
            "prediction_raw_frame_range_half_open": [8, 84],
            "selected_action_bundle": _bound_file(selected_robot_path),
        },
        "information_boundary": {
            "maximum_object_rgb_frame_read": 0,
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_depth_or_mask_read": False,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
            "whole_future_container_hashed_or_read": False,
        },
    }
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    manifest_path = tmp_path / "frame_zero_manifest.json"
    _write_json(manifest_path, manifest)
    return lock_path, manifest_path


def test_prediction_input_repeats_only_frame_zero(tmp_path: Path) -> None:
    lock_path, manifest_path = _make_locked_frame_zero(tmp_path)
    data_path = tmp_path / "prediction.pkl"
    summary_path = tmp_path / "prediction.json"
    summary = build_prediction_only_artifacts(
        manifest_path,
        lock_path,
        data_path,
        summary_path,
        case_name=CASE_NAME,
        role="calibration",
    )
    with data_path.open("rb") as stream:
        data = pickle.load(stream)
    assert data["object_points"].shape == (FRAME_COUNT, 128, 3)
    assert np.array_equal(
        data["object_points"],
        np.repeat(data["object_points"][:1], FRAME_COUNT, axis=0),
    )
    assert data["controller_points"].shape == (FRAME_COUNT, 768, 3)
    assert summary["information_boundary"]["future_object_geometry_read"] is False
    assert summary["point_count"] == 128


def test_prediction_input_rejects_multiframe_bundle(tmp_path: Path) -> None:
    lock_path, manifest_path = _make_locked_frame_zero(tmp_path, encoded_frames=(0, 1))
    with pytest.raises(ValueError, match="nonzero frame"):
        build_prediction_only_artifacts(
            manifest_path,
            lock_path,
            tmp_path / "prediction.pkl",
            tmp_path / "prediction.json",
            case_name=CASE_NAME,
            role="calibration",
        )


def test_action_window_uses_earliest_closed_path_maximum() -> None:
    frame_count = 100
    actions = np.zeros((frame_count, 5, 3), dtype=np.float64)
    openings = np.ones(frame_count, dtype=np.float64) * 0.05
    # Identical action path in two overlapping candidates; the first must win.
    actions[:, 0, 0] = np.arange(frame_count, dtype=np.float64)
    result = select_action_window(actions, openings)
    assert result["prediction_frame_range_half_open"] == [8, 84]
    assert result["staging_frame_range_half_open"] == [8, 89]


def _warp_result(
    path: Path,
    *,
    simulator_path: Path,
    graph_path: Path,
    semantic_sha256: str,
    scale: float,
    trajectory: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path = path.with_name("official_phystwin_trajectory.npz")
    np.savez_compressed(trajectory_path, vertices=trajectory)
    value = {
        "passed": True,
        "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
        "data_sha256": sha256_file(simulator_path),
        "config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
        "config_overrides": {
            "controller_max_neighbours": 1,
            "controller_radius": 0.03,
            "dashpot_damping": 100.0,
            "drag_damping": 10.0,
            "init_spring_Y": 10_000.0,
        },
        "support_dynamics": {"mode": "official-ground"},
        "canonical_reusable_graph": {
            "file_sha256": sha256_file(graph_path),
            "reusable_graph_sha256": semantic_sha256,
            "controller_patch_size_per_anchor": 16,
        },
        "realized_actuation": {"controller_displacement_scale": scale},
        "trajectory_sha256": sha256_file(trajectory_path),
    }
    _write_json(path, value)


def test_physical_archive_matches_frozen_driven_minus_zero_formula(
    tmp_path: Path,
) -> None:
    lock_path, frame_zero_manifest = _make_locked_frame_zero(tmp_path)
    prediction_path = tmp_path / "prediction.pkl"
    prediction_summary = tmp_path / "prediction_summary.json"
    build_prediction_only_artifacts(
        frame_zero_manifest,
        lock_path,
        prediction_path,
        prediction_summary,
        case_name=CASE_NAME,
        role="calibration",
    )
    simulator_path = tmp_path / "simulator.pkl"
    simulator_path.write_bytes(b"simulator")
    semantic_sha256 = "a" * 64
    graph_path = tmp_path / "graph.npz"
    vertex_count = 128
    vertices = np.column_stack(
        (
            np.arange(vertex_count) * 0.001,
            np.zeros(vertex_count),
            np.zeros(vertex_count),
        )
    ).astype(np.float32)
    springs = np.column_stack(
        (np.arange(vertex_count - 1), np.arange(1, vertex_count))
    ).astype(np.int32)
    np.savez_compressed(
        graph_path,
        vertices=vertices,
        springs=springs,
        rest_lengths=np.ones(vertex_count - 1, dtype=np.float32) * 0.001,
        contact_anchor_indices=np.asarray([0], dtype=np.int64),
        observed_node_count=np.asarray(vertex_count),
        reusable_graph_sha256=np.asarray(semantic_sha256),
    )
    state_path = tmp_path / "state.npz"
    np.savez_compressed(
        state_path,
        readout_weights=np.eye(vertex_count, dtype=np.float32),
        canonical_graph_sha256=np.asarray(semantic_sha256),
    )
    twin_path = tmp_path / "twin.json"
    twin = {
        "passed": True,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "input_sha256": {"episode_final_data": sha256_file(prediction_path)},
        "output_sha256": {"simulator_final_data": sha256_file(simulator_path)},
        "information_boundary": {
            "target_access": False,
            "post_initial_object_observation_used": False,
        },
    }
    _write_json(twin_path, twin)
    zero = np.repeat(vertices[None], FRAME_COUNT, axis=0)
    driven = zero.copy()
    driven[:, :, 1] += np.arange(FRAME_COUNT, dtype=np.float32)[:, None] * 0.001
    driven_result = tmp_path / "driven" / "official_phystwin_smoke.json"
    zero_result = tmp_path / "zero" / "official_phystwin_smoke.json"
    _warp_result(
        driven_result,
        simulator_path=simulator_path,
        graph_path=graph_path,
        semantic_sha256=semantic_sha256,
        scale=1.0,
        trajectory=driven,
    )
    _warp_result(
        zero_result,
        simulator_path=simulator_path,
        graph_path=graph_path,
        semantic_sha256=semantic_sha256,
        scale=0.0,
        trajectory=zero,
    )
    archive_path = tmp_path / "physical_prediction.npz"
    manifest_path = tmp_path / "physical_prediction.json"
    result = build_physical_prediction_archive(
        prediction_path,
        simulator_path,
        graph_path,
        state_path,
        twin_path,
        driven_result,
        zero_result,
        archive_path,
        manifest_path,
        frame_zero_manifest_path=frame_zero_manifest,
        lock_path=lock_path,
        case_name=CASE_NAME,
        role="calibration",
        runtime_provenance={},
        stage_runtime_seconds={"test": 0.0},
    )
    assert result["frozen_predictor"]["warp_dynamics"] == WARP_DYNAMICS
    with np.load(archive_path, allow_pickle=False) as stored:
        assert stored["prediction_m"].shape == (FRAME_COUNT, vertex_count, 3)
        assert np.allclose(stored["prediction_m"][0], stored["frame_zero_points_m"])
        assert np.all(stored["action_support"] <= 1.0)
        assert np.all(stored["action_support"] > 0.0)

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["information_boundary"]["future_object_geometry_read"] = True
    tampered["artifact_sha256"] = held_artifact_sha256(tampered)
    _write_json(manifest_path, tampered)
    with pytest.raises(ValueError, match="information boundary"):
        validate_physical_prediction_manifest(manifest_path, verify_archive=True)


def _upstream_result_sha256(value: dict[str, object]) -> str:
    canonical = dict(value)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _make_inadmissible_twin_artifacts(
    root: Path,
    prediction_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    with prediction_path.open("rb") as stream:
        prediction = pickle.load(stream)
    points = np.asarray(prediction["object_points"])[0]
    controllers = np.asarray(prediction["controller_points"])
    point_count = len(points)
    graph_sha256 = "b" * 64
    graph_path = root / "episode_graph.npz"
    springs = np.column_stack(
        (np.arange(point_count - 1), np.arange(1, point_count))
    ).astype(np.int32)
    rest_lengths = np.linalg.norm(
        points[springs[:, 1]] - points[springs[:, 0]], axis=1
    ).astype(np.float32)
    np.savez_compressed(
        graph_path,
        vertices=points.astype(np.float32),
        colors=np.asarray(prediction["object_colors"])[0].astype(np.float32),
        source_indices=np.arange(point_count, dtype=np.int64),
        springs=springs,
        rest_lengths=rest_lengths,
        masses=np.ones(point_count, dtype=np.float32),
        bridge_spring_count=np.asarray(0, dtype=np.int64),
        observed_node_count=np.asarray(point_count, dtype=np.int64),
        latent_node_count=np.asarray(0, dtype=np.int64),
        contact_anchor_indices=np.asarray([0], dtype=np.int64),
        contact_chain_spring_count=np.asarray(0, dtype=np.int64),
        reusable_graph_sha256=np.asarray(graph_sha256),
    )
    reliability = np.full(point_count, 0.69, dtype=np.float64)
    state_path = root / "state_artifact.npz"
    np.savez_compressed(
        state_path,
        vertices=points.astype(np.float32),
        readout_weights=np.eye(point_count, dtype=np.float64),
        readout_covariance_m2=np.zeros((point_count, 3, 3), dtype=np.float64),
        target_prior_reliability=reliability,
        state_covariance_m2=np.zeros((point_count, 3, 3), dtype=np.float64),
        source_to_target_distance_m=np.zeros(point_count, dtype=np.float64),
        target_to_source_distance_m=np.zeros(point_count, dtype=np.float64),
        relative_edge_strain=np.zeros(len(springs), dtype=np.float64),
        canonical_graph_sha256=np.asarray(graph_sha256),
        state_frame=np.asarray(0, dtype=np.int64),
    )
    simulator_path = root / "simulator_final_data.pkl"
    simulator_path.write_bytes(b"checksummed automatic-twin simulator input")
    metrics: dict[str, object] = {
        "passed": False,
        "finite": True,
        "symmetric_chamfer_m": 0.0,
        "source_to_target_p95_m": 0.0,
        "target_to_source_p95_m": 0.0,
        "observed_target_fraction": 1.0,
        "canonical_supported_fraction": 1.0,
        "effective_target_reliability": 0.69,
        "initial_readout_rmse_m": 0.0,
        "p99_absolute_relative_edge_strain": 0.0,
        "maximum_absolute_relative_edge_strain": 0.0,
        "maximum_bridge_absolute_relative_edge_strain": 0.0,
        "maximum_contact_anchor_error_m": 0.0,
    }
    summary: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360AutomaticEpisodeTwin",
        "protocol_id": AUTOMATIC_TWIN_PROTOCOL_ID,
        "protocol_config_sha256": AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "phase": "calibration",
        "graph_mode": "episode_specific_frame_zero_control",
        "capacity_diagnostic": {
            "configured_canonical_node_count": 192,
            "requested_canonical_node_count": CANONICAL_NODE_COUNT,
            "effective_canonical_node_count": point_count,
            "source_only_override": point_count != 192,
            "capacity_is_a_maximum": True,
        },
        "graph": {
            "schema_version": 1,
            "artifact_kind": "Deform360CanonicalReusableGraph",
            "path": str(graph_path.resolve()),
            "reusable_graph_sha256": graph_sha256,
            "node_count": point_count,
            "object_spring_count": len(springs),
            "bridge_spring_count": 0,
            "observed_node_count": point_count,
            "latent_node_count": 0,
            "contact_anchor_count": 1,
            "contact_chain_spring_count": 0,
        },
        "state_metrics": metrics,
        "input_sha256": {
            "episode_final_data": sha256_file(prediction_path),
            "development_observations": None,
            "contact_conditioned_action": None,
        },
        "output_sha256": {
            "episode_graph": sha256_file(graph_path),
            "simulator_final_data": sha256_file(simulator_path),
            "state_artifact": sha256_file(state_path),
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_robot_action_available": True,
            "post_initial_object_observation_used": False,
            "simulator_residual_used": False,
            "target_access": False,
            "prediction_only_input_required": True,
            "future_object_tracks_present": False,
            "contact_conditioned_action_used": False,
            "contact_conditioned_action_result_sha256": None,
        },
        "prediction_input_validation": {
            "frame_count": FRAME_COUNT,
            "point_count": point_count,
            "controller_point_count": controllers.shape[1],
            "frame_zero_points_sha256": physical_prior.sha256_array(points),
            "controller_trajectory_sha256": physical_prior.sha256_array(controllers),
        },
        "sota_input_validation": None,
        "passed": False,
        "claim_boundary": (
            "benchmark-fair automatic frame-zero episode-twin control; physical "
            "parameters may be pooled across source episodes"
        ),
    }
    summary["result_sha256"] = _upstream_result_sha256(summary)
    summary_path = root / "twin_summary.json"
    _write_json(summary_path, summary)
    log_path = root / "automatic_twin.log"
    _write_json(log_path, summary)
    return simulator_path, graph_path, state_path, summary_path, log_path


def test_inadmissible_twin_seals_explicit_persistence_fallback(
    tmp_path: Path,
) -> None:
    lock_path, frame_zero_manifest = _make_locked_frame_zero(tmp_path)
    prediction_path = tmp_path / "prediction.pkl"
    prediction_summary = tmp_path / "prediction_summary.json"
    build_prediction_only_artifacts(
        frame_zero_manifest,
        lock_path,
        prediction_path,
        prediction_summary,
        case_name=CASE_NAME,
        role="calibration",
    )
    simulator, graph, state, twin, log = _make_inadmissible_twin_artifacts(
        tmp_path, prediction_path
    )
    archive = tmp_path / "fallback.npz"
    manifest_path = tmp_path / "fallback.json"
    manifest = build_persistence_fallback_archive(
        prediction_path,
        simulator,
        graph,
        state,
        twin,
        log,
        archive,
        manifest_path,
        frame_zero_manifest_path=frame_zero_manifest,
        lock_path=lock_path,
        case_name=CASE_NAME,
        role="calibration",
        automatic_twin_exit_code=2,
        runtime_provenance={"test": True},
        stage_runtime_seconds={"automatic_twin": 0.1},
    )
    assert manifest["physical_mode"] == "persistence_fallback"
    assert manifest["physical_admitted"] is False
    assert manifest["fallback_diagnostics"]["warp_attempted"] is False
    assert set(manifest["input_files"]) == {
        "prediction_only_input",
        "simulator_final_data",
        "episode_graph",
        "state_artifact",
        "twin_summary",
        "automatic_twin_log",
    }
    with np.load(archive, allow_pickle=False) as stored:
        persistence = stored["persistence_m"]
        assert np.array_equal(stored["prediction_m"], persistence)
        assert np.array_equal(stored["driven_readout_m"], persistence)
        assert np.array_equal(stored["zero_action_readout_m"], persistence)
        assert np.count_nonzero(stored["action_support"]) == 0

    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["physical_admitted"] = True
    tampered["artifact_sha256"] = held_artifact_sha256(tampered)
    _write_json(manifest_path, tampered)
    with pytest.raises(ValueError, match="admission flag"):
        validate_physical_prediction_manifest(manifest_path, verify_archive=True)


def test_persistence_fallback_rejects_passing_or_unchecksummed_twin(
    tmp_path: Path,
) -> None:
    lock_path, frame_zero_manifest = _make_locked_frame_zero(tmp_path)
    prediction_path = tmp_path / "prediction.pkl"
    build_prediction_only_artifacts(
        frame_zero_manifest,
        lock_path,
        prediction_path,
        tmp_path / "prediction_summary.json",
        case_name=CASE_NAME,
        role="calibration",
    )
    simulator, graph, state, twin, log = _make_inadmissible_twin_artifacts(
        tmp_path, prediction_path
    )
    summary = json.loads(twin.read_text(encoding="utf-8"))
    summary["state_metrics"]["effective_target_reliability"] = 0.71
    summary["state_metrics"]["passed"] = True
    summary["passed"] = True
    summary["result_sha256"] = _upstream_result_sha256(summary)
    _write_json(twin, summary)
    with pytest.raises(ValueError, match="inadmissible result"):
        build_persistence_fallback_archive(
            prediction_path,
            simulator,
            graph,
            state,
            twin,
            log,
            tmp_path / "fallback.npz",
            tmp_path / "fallback.json",
            frame_zero_manifest_path=frame_zero_manifest,
            lock_path=lock_path,
            case_name=CASE_NAME,
            role="calibration",
            automatic_twin_exit_code=2,
            runtime_provenance={},
            stage_runtime_seconds={"automatic_twin": 0.1},
        )


def test_python_runtime_preserves_supplied_venv_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "python-real"
    executable.write_bytes(b"locked interpreter bytes")
    executable.chmod(0o755)
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    supplied = venv_bin / "python"
    supplied.symlink_to(executable)
    freeze = b"zeta==2\nalpha==1\npip==24\n"
    expected_freeze = hashlib.sha256(b"alpha==1\npip==24\nzeta==2\n").hexdigest()
    observed_command: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed_command.extend(command)
        assert kwargs == {"check": True, "capture_output": True}
        return SimpleNamespace(stdout=freeze)

    monkeypatch.setattr(physical_prior.subprocess, "run", fake_run)
    result = validate_python_runtime(
        supplied,
        {
            "python_executable": sha256_file(executable),
            "python_pip_freeze_sorted": expected_freeze,
        },
    )
    assert observed_command[0] == str(supplied.absolute())
    assert result["supplied_python_path"] == str(supplied.absolute())
    assert result["resolved_python_path"] == str(executable.resolve())


def test_runner_does_not_convert_arbitrary_subprocess_failure_to_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, frame_zero_manifest = _make_locked_frame_zero(tmp_path)
    monkeypatch.setattr(
        physical_prior,
        "validate_python_runtime",
        lambda *_: {"supplied_python_path": str(tmp_path / "python")},
    )
    monkeypatch.setattr(
        physical_prior, "validate_upstream_runtime", lambda *_: {"test": True}
    )

    def fail(*args: object, **kwargs: object) -> float:
        raise physical_prior._LoggedCommandError(
            "not an admission rejection", returncode=1, elapsed_seconds=0.1
        )

    monkeypatch.setattr(physical_prior, "_run_logged", fail)
    with pytest.raises(RuntimeError, match="not an admission rejection"):
        run_held_physical_prior(
            frame_zero_manifest,
            lock_path,
            tmp_path / "output",
            case_name=CASE_NAME,
            role="calibration",
            upstream_repo=tmp_path / "upstream",
            official_phystwin_repo=tmp_path / "phystwin",
            official_config=tmp_path / "real.yaml",
            deform360_repo=tmp_path / "deform360",
            python=tmp_path / "venv" / "bin" / "python",
        )
    assert not (tmp_path / "output" / "prediction.npz").exists()


def test_runner_converts_only_valid_exit_two_admission_to_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, frame_zero_manifest = _make_locked_frame_zero(tmp_path)
    supplied_python = tmp_path / "venv" / "bin" / "python"
    monkeypatch.setattr(
        physical_prior,
        "validate_python_runtime",
        lambda *_: {"supplied_python_path": str(supplied_python)},
    )
    monkeypatch.setattr(
        physical_prior, "validate_upstream_runtime", lambda *_: {"test": True}
    )
    calls: list[list[str]] = []

    def fail_with_valid_admission(
        command: list[str], *, env: object, log_path: Path
    ) -> float:
        del env
        calls.append(command)
        assert command[command.index("--canonical-node-count") + 1] == "1024"
        prediction = Path(command[command.index("--episode-final-data") + 1])
        output = prediction.parent
        _, _, _, summary, _ = _make_inadmissible_twin_artifacts(output, prediction)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
        raise physical_prior._LoggedCommandError(
            "explicit source admission rejection", returncode=2, elapsed_seconds=0.1
        )

    monkeypatch.setattr(physical_prior, "_run_logged", fail_with_valid_admission)
    result = run_held_physical_prior(
        frame_zero_manifest,
        lock_path,
        tmp_path / "output",
        case_name=CASE_NAME,
        role="calibration",
        upstream_repo=tmp_path / "upstream",
        official_phystwin_repo=tmp_path / "phystwin",
        official_config=tmp_path / "real.yaml",
        deform360_repo=tmp_path / "deform360",
        python=supplied_python,
    )
    assert len(calls) == 1
    assert result["physical_prediction_manifest"]["physical_mode"] == (
        "persistence_fallback"
    )
    assert result["physical_prediction_manifest"]["physical_admitted"] is False
    assert not (tmp_path / "output" / "warp_driven").exists()


def test_runner_keeps_warp_failure_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, frame_zero_manifest = _make_locked_frame_zero(tmp_path)
    monkeypatch.setattr(
        physical_prior,
        "validate_python_runtime",
        lambda *_: {"supplied_python_path": str(tmp_path / "python")},
    )
    monkeypatch.setattr(
        physical_prior, "validate_upstream_runtime", lambda *_: {"test": True}
    )
    call_count = 0

    def fail_warp(*args: object, **kwargs: object) -> float:
        nonlocal call_count
        del args, kwargs
        call_count += 1
        if call_count == 1:
            return 0.1
        raise physical_prior._LoggedCommandError(
            "Warp rollout failed", returncode=2, elapsed_seconds=0.2
        )

    monkeypatch.setattr(physical_prior, "_run_logged", fail_warp)
    with pytest.raises(RuntimeError, match="Warp rollout failed"):
        run_held_physical_prior(
            frame_zero_manifest,
            lock_path,
            tmp_path / "output",
            case_name=CASE_NAME,
            role="calibration",
            upstream_repo=tmp_path / "upstream",
            official_phystwin_repo=tmp_path / "phystwin",
            official_config=tmp_path / "real.yaml",
            deform360_repo=tmp_path / "deform360",
            python=tmp_path / "venv" / "bin" / "python",
        )
    assert call_count == 2
    assert not (tmp_path / "output" / "prediction.npz").exists()
