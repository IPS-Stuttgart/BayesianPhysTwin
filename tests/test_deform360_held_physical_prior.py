from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

import bayesian_phystwin.deform360_held_physical_prior as physical_prior

from deform360_held_test_helpers import (
    bound_file,
    default_frame_zero_config,
    dummy_immutable_bindings,
    write_robot_kinematics_fixture,
)

from bayesian_phystwin.deform360_frame_zero_assets import (
    FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
    FRAME_ZERO_CAMERA_SELECTION_RULE,
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
    load_controller_trajectory,
    run_held_physical_prior,
    sha256_file,
    validate_physical_prediction_manifest,
    validate_python_runtime,
)
from bayesian_phystwin.deform360_robot_kinematics import (
    ROBOT_KINEMATICS_WINDOW_CONTRACT,
    ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
    ROBOT_KINEMATICS_WINDOW_POLICY_ID,
)
from bayesian_phystwin.deform360_held_protocol import (
    create_held_protocol_lock,
    held_artifact_sha256,
)


CASE_NAME = "083-blanket-cloth-ep0000"
TEST_CAMERAS = tuple(
    sorted(["brics-odroid-001_cam0", *(f"camera-{index:02d}" for index in range(7))])
)


def _run_git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["/usr/bin/git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout


def _official_worktree_fixture(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    repository = tmp_path / "official-phystwin"
    for relative in physical_prior._QQTT_IMPORTED_PROVENANCE.values():
        source = repository / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# locked test source: {relative}\n", encoding="utf-8")
    (repository / "README.md").write_text("locked fixture\n", encoding="utf-8")
    _run_git(repository, "init", "-q")
    _run_git(repository, "add", ".")
    _run_git(
        repository,
        "-c",
        "user.name=Held Test",
        "-c",
        "user.email=held@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    revision = _run_git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    tree_lines = _run_git(
        repository,
        "ls-tree",
        "-r",
        "--full-tree",
        "HEAD",
    ).splitlines()
    tree_manifest = b"".join(line + b"\n" for line in sorted(tree_lines))
    bindings = {
        "official_phystwin_revision_literal": hashlib.sha256(
            revision.encode("ascii")
        ).hexdigest(),
        "official_phystwin_commit_object": hashlib.sha256(
            _run_git(repository, "cat-file", "commit", "HEAD")
        ).hexdigest(),
        "official_phystwin_git_tree_manifest": hashlib.sha256(
            tree_manifest
        ).hexdigest(),
    }
    return repository, revision, bindings


def _mock_runtime_provenance(tmp_path: Path) -> dict[str, object]:
    upstream = tmp_path / "upstream"
    official = tmp_path / "phystwin"
    deform360 = tmp_path / "deform360"
    upstream.mkdir(exist_ok=True)
    official.mkdir(exist_ok=True)
    deform360.mkdir(exist_ok=True)
    config = tmp_path / "real.yaml"
    config.touch()
    return {
        "upstream_repository_root": str(upstream),
        "official_phystwin_repository_root": str(official),
        "official_config_path": str(config),
        "test": True,
    }


def test_v3_uses_fixed_1024_node_capacity_and_shared_robot_contract() -> None:
    assert CANONICAL_NODE_COUNT == 1024
    assert physical_prior.HELD_PHYSICAL_NUMERIC_CONTRACT["contract_id"].endswith("-v3")
    assert physical_prior.HELD_PHYSICAL_NUMERIC_CONTRACT["canonical_node_count"] == 1024
    assert physical_prior.HELD_PHYSICAL_NUMERIC_CONTRACT["robot_kinematics"] == {
        "policy_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
        "contract_sha256": ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
        "trajectory_semantics": (
            "aligned absolute end-effector pose/opening annotation in the "
            "Deform360 world frame; not a delta command"
        ),
        "controller_source": "T_worlds absolute end-effector pose and openings",
    }


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
    poses[:, 0, 3] = np.arange(frame_count, dtype=np.float64) * 0.001
    actions = np.zeros((frame_count, 5, 3), dtype=np.float64)
    actions[:, 0] = poses[:, :3, 3]
    actions[:, 1:4] = poses[:, :3, :3]
    openings = np.full(frame_count, 0.05, dtype=np.float64)
    actions[:, 4, 0] = openings
    np.savez_compressed(
        path,
        format_version=np.asarray(1, dtype=np.uint16),
        actions=actions,
        T_worlds=poses,
        openings=openings,
        bimanual=np.asarray(False, dtype=np.bool_),
    )


def _slice_robot(source: Path, destination: Path, *, start: int, count: int) -> None:
    with np.load(source, allow_pickle=False) as stored:
        np.savez_compressed(
            destination,
            format_version=stored["format_version"],
            actions=stored["actions"][start : start + count],
            T_worlds=stored["T_worlds"][start : start + count],
            openings=stored["openings"][start : start + count],
            bimanual=stored["bimanual"],
        )


def _make_frame_zero_bundle(
    path: Path, *, encoded_frames: tuple[int, ...] = (0,)
) -> None:
    camera_count = len(TEST_CAMERAS)
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
        camera_names=np.asarray(TEST_CAMERAS),
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
    robot_path, _selected_robot_path, action_alignment = (
        write_robot_kinematics_fixture(
            tmp_path,
            source_frame_count=100,
            selected_start=8,
        )
    )
    robot_metadata_path = tmp_path / "robot.meta.json"
    robot_metadata_path.write_text("{}\n", encoding="utf-8")
    config = default_frame_zero_config()
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360HeldFrameZeroBundle",
        "protocol_id": "deform360-held-online-belief-v5",
        "case_name": CASE_NAME,
        "object_id": "083-blanket-cloth",
        "episode_id": 0,
        "role": "calibration",
        "frame_indices": [0],
        "lock_sha256": sha256_file(lock_path),
        "lock_artifact_sha256": json.loads(lock_path.read_text(encoding="utf-8"))[
            "artifact_sha256"
        ],
        "config": config,
        "bundle": _bound_file(bundle_path),
        "action_inputs": {
            "robot_trajectory": bound_file(robot_path),
            "robot_metadata": _bound_file(robot_metadata_path),
        },
        "action_alignment": action_alignment,
        "camera_policy": {
            "policy_id": FRAME_ZERO_CAMERA_SELECTION_POLICY_ID,
            "rule": FRAME_ZERO_CAMERA_SELECTION_RULE,
            "reference_camera": config["reference_camera"],
            "minimum_selected_camera_count": config["minimum_camera_count"],
            "candidate_cameras": list(TEST_CAMERAS),
            "candidate_camera_count": len(TEST_CAMERAS),
            "selected_cameras": list(TEST_CAMERAS),
            "selected_camera_count": len(TEST_CAMERAS),
            "abstained_cameras": [],
            "abstained_camera_count": 0,
        },
        "camera_frame_zero_access": [
            {
                "camera": camera,
                "path": str((tmp_path / f"{camera}.mp4").resolve()),
                "decoded_frame_count": 1,
                "maximum_rgb_frame_read": 0,
                "action_window_frame_index": 0,
                "source_aligned_frame_index": 8,
                "decoded_rgb_sha256": "d" * 64,
                "whole_file_hashed_or_read": False,
            }
            for camera in TEST_CAMERAS
        ],
        "information_boundary": {
            "maximum_object_rgb_frame_read": 0,
            "object_observation_frames_used": [0],
            "known_aligned_realized_robot_kinematics_read": True,
            "known_robot_trajectory_semantics": ROBOT_KINEMATICS_WINDOW_CONTRACT[
                "trajectory_semantics"
            ],
            "robot_delta_command_read": False,
            "commanded_control_read": False,
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
    assert summary["information_boundary"][
        "known_future_aligned_robot_kinematics_read"
    ] is True
    audit = summary["robot_kinematics_window"]
    assert audit["policy_id"] == ROBOT_KINEMATICS_WINDOW_POLICY_ID
    assert audit["contract_sha256"] == ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256
    assert audit["prediction_raw_frame_range_half_open"] == [8, 84]
    assert audit["exact_source_slice_verified"] is True
    assert data["prediction_only_input"][
        "known_future_realized_robot_kinematics_used"
    ] is True
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


def test_raw_controller_trajectory_uses_shared_realized_kinematics_selector(
    tmp_path: Path,
) -> None:
    robot_path = tmp_path / "raw_robot.npz"
    _make_robot(robot_path, frame_count=100)

    controllers, audit = load_controller_trajectory(robot_path)

    assert controllers.shape == (FRAME_COUNT, 768, 3)
    assert audit["policy_id"] == ROBOT_KINEMATICS_WINDOW_POLICY_ID
    assert audit["contract_sha256"] == ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256
    assert audit["input_mode"] == "raw_episode_robot_kinematics"
    assert audit["selected_raw_frame_range_half_open"] == [8, 89]
    assert audit["prediction_raw_frame_range_half_open"] == [8, 84]
    assert audit["exact_source_slice_verified"] is True
    # With identity EEF rotations and a fixed opening, the whole taxel cloud
    # must translate exactly with T_worlds[..., :3, 3].
    expected_dx = np.arange(FRAME_COUNT, dtype=np.float64) * 0.001
    observed_dx = np.mean(controllers[:, :, 0], axis=1) - np.mean(
        controllers[0, :, 0]
    )
    assert np.allclose(observed_dx, expected_dx, atol=1e-7)


def test_preselected_controller_slice_is_verified_against_raw_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw_robot.npz"
    selected = tmp_path / "selected_robot.npz"
    _make_robot(source, frame_count=100)
    _slice_robot(source, selected, start=8, count=FRAME_COUNT)

    _, audit = load_controller_trajectory(
        selected,
        source_robot_path=source,
        expected_selected_raw_frame_range=[8, 89],
        expected_prediction_raw_frame_range=[8, 84],
    )

    assert audit["input_mode"] == "preselected_exact_prediction_slice"
    assert audit["selected_raw_frame_range_half_open"] == [8, 89]
    assert audit["prediction_raw_frame_range_half_open"] == [8, 84]
    assert audit["selected_prediction_frame_range_half_open"] == [0, FRAME_COUNT]
    assert audit["exact_source_slice_verified"] is True
    assert audit["selected_bundle_validation"]["exact_source_slice"] is True

    _, standalone = load_controller_trajectory(selected)
    assert standalone["selection_performed"] is False
    assert standalone["selected_raw_frame_range_half_open"] is None
    assert standalone["exact_source_slice_verified"] is False


def test_controller_trajectory_fails_closed_on_schema_parity_and_slice_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "raw_robot.npz"
    _make_robot(source, frame_count=100)
    with np.load(source, allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name], copy=True) for name in stored.files}

    extra = tmp_path / "extra_field.npz"
    np.savez_compressed(extra, **arrays, unexpected=np.asarray(1))
    with pytest.raises(ValueError, match="field set changed"):
        load_controller_trajectory(extra)

    parity = tmp_path / "bad_action_parity.npz"
    parity_arrays = {name: np.array(value, copy=True) for name, value in arrays.items()}
    parity_arrays["actions"][10, 0, 0] += 0.1
    np.savez_compressed(parity, **parity_arrays)
    with pytest.raises(ValueError, match="row 0 does not match"):
        load_controller_trajectory(parity)

    selected = tmp_path / "wrong_selected_slice.npz"
    _slice_robot(source, selected, start=8, count=FRAME_COUNT)
    with np.load(selected, allow_pickle=False) as stored:
        selected_arrays = {
            name: np.array(stored[name], copy=True) for name in stored.files
        }
    selected_arrays["actions"][:, 0, 0] += 0.1
    selected_arrays["T_worlds"][:, 0, 3] += 0.1
    np.savez_compressed(selected, **selected_arrays)
    with pytest.raises(ValueError, match="not the exact source slice"):
        load_controller_trajectory(
            selected,
            source_robot_path=source,
            expected_selected_raw_frame_range=[8, 89],
            expected_prediction_raw_frame_range=[8, 84],
        )


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
    tampered["robot_kinematics_window"]["contract_sha256"] = "f" * 64
    tampered["artifact_sha256"] = held_artifact_sha256(tampered)
    _write_json(manifest_path, tampered)
    with pytest.raises(ValueError, match="robot kinematics contract"):
        validate_physical_prediction_manifest(manifest_path, verify_archive=True)

    _write_json(manifest_path, result)
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


def _test_runtime_entries(root: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(directory) / name
            relative = os.path.relpath(path, root)
            observed = os.lstat(path)
            entry: dict[str, object] = {
                "path": relative,
                "mode": format(stat.S_IMODE(observed.st_mode), "04o"),
            }
            if stat.S_ISDIR(observed.st_mode):
                entry["type"] = "directory"
            elif stat.S_ISREG(observed.st_mode):
                entry.update(
                    {
                        "type": "file",
                        "size": observed.st_size,
                        "sha256": sha256_file(path),
                    }
                )
            else:
                assert stat.S_ISLNK(observed.st_mode)
                entry.update({"type": "symlink", "target": os.readlink(path)})
            entries.append(entry)
    return sorted(entries, key=lambda entry: os.fsencode(str(entry["path"])))


def _write_test_runtime_manifest(
    path: Path,
    *,
    root: Path,
    freeze_sha256: str,
) -> dict[str, object]:
    entries = _test_runtime_entries(root)
    counts = {"directory": 0, "file": 0, "symlink": 0}
    total_bytes = 0
    for entry in entries:
        counts[str(entry["type"])] += 1
        if entry["type"] == "file":
            total_bytes += int(entry["size"])
    entries_bytes = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    payload: dict[str, object] = {
        "artifact_kind": physical_prior.HELD_PYTHON_RUNTIME_MANIFEST_KIND,
        "root_path": str(root),
        "python_pip_freeze_sorted_sha256": freeze_sha256,
        "entry_counts": counts,
        "total_regular_file_bytes": total_bytes,
        "tree_sha256": hashlib.sha256(entries_bytes).hexdigest(),
        "entries": entries,
    }
    path.write_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )
    path.chmod(0o400)
    return payload


@pytest.fixture
def frozen_python_runtime(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> dict[str, object]:
    workspace = Path(tempfile.mkdtemp(prefix="bpt-frozen-runtime-", dir="/tmp"))
    executable = workspace / "python-real"
    executable.write_bytes(b"locked interpreter bytes")
    executable.chmod(0o755)
    root = workspace / "frozen-runtime"
    venv_bin = root / "bin"
    package_dir = root / "lib/python3.12/site-packages/example"
    package_dir.mkdir(parents=True)
    venv_bin.mkdir(parents=True)
    package_file = package_dir / "__init__.py"
    package_file.write_bytes(b"VALUE = 1\n")
    package_file.chmod(0o444)
    supplied = venv_bin / "python"
    supplied.symlink_to(executable)
    (venv_bin / "python3").symlink_to("python")
    (venv_bin / "python3.12").symlink_to("python")
    for directory, directories, _files in os.walk(root):
        Path(directory).chmod(0o555)
        for name in directories:
            path = Path(directory) / name
            if not path.is_symlink():
                path.chmod(0o555)
    freeze = b"zeta==2\nalpha==1\npip==24\n"
    expected_freeze = hashlib.sha256(
        b"alpha==1\npip==24\nzeta==2\n"
    ).hexdigest()
    manifest_path = root.parent / f"{root.name}.tree-manifest.json"
    manifest = _write_test_runtime_manifest(
        manifest_path,
        root=root,
        freeze_sha256=expected_freeze,
    )
    monkeypatch.setattr(physical_prior, "HELD_PYTHON_RUNTIME", root)
    monkeypatch.setattr(
        physical_prior,
        "HELD_PYTHON_RUNTIME_MANIFEST",
        manifest_path,
    )
    monkeypatch.setattr(
        physical_prior,
        "HELD_PYTHON_RUNTIME_SYMLINKS",
        {
            "bin/python": str(executable),
            "bin/python3": "python",
            "bin/python3.12": "python",
        },
    )
    bindings = {
        "held_frozen_runtime_manifest": sha256_file(manifest_path),
        "python_executable": sha256_file(executable),
        "python_pip_freeze_sorted": expected_freeze,
    }

    def restore_permissions() -> None:
        if manifest_path.exists():
            manifest_path.chmod(0o600)
        for directory, directories, files in os.walk(root, topdown=False):
            for name in files:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(0o600)
            for name in directories:
                path = Path(directory) / name
                if not path.is_symlink():
                    path.chmod(0o700)
            Path(directory).chmod(0o700)
        shutil.rmtree(workspace)

    request.addfinalizer(restore_permissions)
    return {
        "root": root,
        "supplied": supplied,
        "executable": executable,
        "package_file": package_file,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "freeze": freeze,
        "bindings": bindings,
    }


def test_python_runtime_preserves_supplied_venv_symlink(
    frozen_python_runtime: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = frozen_python_runtime["executable"]
    supplied = frozen_python_runtime["supplied"]
    freeze = frozen_python_runtime["freeze"]
    bindings = frozen_python_runtime["bindings"]
    assert isinstance(executable, Path)
    assert isinstance(supplied, Path)
    assert isinstance(freeze, bytes)
    assert isinstance(bindings, dict)
    observed_command: list[str] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed_command.extend(command)
        assert kwargs == {"check": True, "capture_output": True}
        return SimpleNamespace(stdout=freeze)

    monkeypatch.setattr(physical_prior.subprocess, "run", fake_run)
    result = validate_python_runtime(supplied, bindings)
    assert observed_command == [
        str(supplied.absolute()),
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={physical_prior.HELD_PYCACHE_PREFIX}",
        "-m",
        "pip",
        "freeze",
        "--all",
    ]
    assert result["supplied_python_path"] == str(supplied.absolute())
    assert result["resolved_python_path"] == str(executable.resolve())
    assert result["runtime_root"] == str(frozen_python_runtime["root"])
    assert result["runtime_manifest_sha256"] == bindings[
        "held_frozen_runtime_manifest"
    ]


def test_python_runtime_rejects_interpreter_outside_frozen_root(
    frozen_python_runtime: dict[str, object],
) -> None:
    executable = frozen_python_runtime["executable"]
    bindings = frozen_python_runtime["bindings"]
    assert isinstance(executable, Path)
    assert isinstance(bindings, dict)

    with pytest.raises(ValueError, match="outside the frozen runtime"):
        validate_python_runtime(executable, bindings)


def test_python_runtime_rejects_aliased_interpreter_path(
    frozen_python_runtime: dict[str, object],
) -> None:
    root = frozen_python_runtime["root"]
    bindings = frozen_python_runtime["bindings"]
    assert isinstance(root, Path)
    assert isinstance(bindings, dict)

    with pytest.raises(ValueError, match="not exact and absolute"):
        validate_python_runtime(str(root / "bin/../bin/python"), bindings)


def test_python_runtime_rejects_wrong_manifest_binding(
    frozen_python_runtime: dict[str, object],
) -> None:
    supplied = frozen_python_runtime["supplied"]
    bindings = dict(frozen_python_runtime["bindings"])
    assert isinstance(supplied, Path)
    bindings["held_frozen_runtime_manifest"] = "f" * 64

    with pytest.raises(ValueError, match="manifest differs"):
        validate_python_runtime(supplied, bindings)


def test_python_runtime_rejects_writable_runtime_root(
    frozen_python_runtime: dict[str, object],
) -> None:
    root = frozen_python_runtime["root"]
    supplied = frozen_python_runtime["supplied"]
    bindings = frozen_python_runtime["bindings"]
    assert isinstance(root, Path)
    assert isinstance(supplied, Path)
    assert isinstance(bindings, dict)
    root.chmod(0o755)

    with pytest.raises(ValueError, match="root mode differs"):
        validate_python_runtime(supplied, bindings)


def test_python_runtime_rejects_unlisted_symlink(
    frozen_python_runtime: dict[str, object],
) -> None:
    root = frozen_python_runtime["root"]
    supplied = frozen_python_runtime["supplied"]
    bindings = frozen_python_runtime["bindings"]
    assert isinstance(root, Path)
    assert isinstance(supplied, Path)
    assert isinstance(bindings, dict)
    bin_dir = root / "bin"
    bin_dir.chmod(0o755)
    (bin_dir / "unlisted-python").symlink_to("python")
    bin_dir.chmod(0o555)

    with pytest.raises(ValueError, match="paths differ"):
        validate_python_runtime(supplied, bindings)


def test_python_runtime_rejects_manifested_file_tamper(
    frozen_python_runtime: dict[str, object],
) -> None:
    supplied = frozen_python_runtime["supplied"]
    package_file = frozen_python_runtime["package_file"]
    bindings = frozen_python_runtime["bindings"]
    assert isinstance(supplied, Path)
    assert isinstance(package_file, Path)
    assert isinstance(bindings, dict)
    package_file.chmod(0o644)
    package_file.write_bytes(b"VALUE = 2\n")
    package_file.chmod(0o444)

    with pytest.raises(ValueError, match="file checksum changed"):
        validate_python_runtime(supplied, bindings)


def test_python_runtime_rejects_manifested_symlink_tamper(
    frozen_python_runtime: dict[str, object],
) -> None:
    root = frozen_python_runtime["root"]
    supplied = frozen_python_runtime["supplied"]
    bindings = frozen_python_runtime["bindings"]
    assert isinstance(root, Path)
    assert isinstance(supplied, Path)
    assert isinstance(bindings, dict)
    bin_dir = root / "bin"
    python3 = bin_dir / "python3"
    bin_dir.chmod(0o755)
    python3.unlink()
    python3.symlink_to("python3.12")
    bin_dir.chmod(0o555)

    with pytest.raises(ValueError, match="symlink target changed"):
        validate_python_runtime(supplied, bindings)


def test_python_runtime_rejects_wrong_tree_checksum_even_when_manifest_is_bound(
    frozen_python_runtime: dict[str, object],
) -> None:
    supplied = frozen_python_runtime["supplied"]
    manifest_path = frozen_python_runtime["manifest_path"]
    manifest = dict(frozen_python_runtime["manifest"])
    bindings = dict(frozen_python_runtime["bindings"])
    assert isinstance(supplied, Path)
    assert isinstance(manifest_path, Path)
    manifest["tree_sha256"] = "f" * 64
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )
    manifest_path.chmod(0o400)
    bindings["held_frozen_runtime_manifest"] = sha256_file(manifest_path)

    with pytest.raises(ValueError, match="tree checksum changed"):
        validate_python_runtime(supplied, bindings)


def test_official_phystwin_worktree_matches_all_locked_git_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision, bindings = _official_worktree_fixture(tmp_path)
    monkeypatch.setattr(physical_prior, "OFFICIAL_PHYSTWIN_REVISION", revision)

    result = physical_prior._validate_official_phystwin_worktree(
        repository,
        bindings,
    )

    assert result["revision"] == revision
    assert result["revision_literal_sha256"] == bindings[
        "official_phystwin_revision_literal"
    ]
    assert result["commit_object_sha256"] == bindings[
        "official_phystwin_commit_object"
    ]
    assert result["git_tree_manifest_sha256"] == bindings[
        "official_phystwin_git_tree_manifest"
    ]
    assert result["qqtt_imported_provenance"] == {
        name: str(repository / relative)
        for name, relative in physical_prior._QQTT_IMPORTED_PROVENANCE.items()
    }


@pytest.mark.parametrize(
    ("binding", "message"),
    (
        ("official_phystwin_revision_literal", "revision differs"),
        ("official_phystwin_commit_object", "commit object differs"),
        ("official_phystwin_git_tree_manifest", "Git tree differs"),
    ),
)
def test_official_phystwin_worktree_rejects_wrong_locked_git_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: str,
    message: str,
) -> None:
    repository, revision, bindings = _official_worktree_fixture(tmp_path)
    monkeypatch.setattr(physical_prior, "OFFICIAL_PHYSTWIN_REVISION", revision)
    bindings[binding] = "f" * 64

    with pytest.raises(ValueError, match=message):
        physical_prior._validate_official_phystwin_worktree(repository, bindings)


def test_official_phystwin_worktree_rejects_dirty_tracked_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision, bindings = _official_worktree_fixture(tmp_path)
    monkeypatch.setattr(physical_prior, "OFFICIAL_PHYSTWIN_REVISION", revision)
    (repository / "qqtt/engine/trainer_warp.py").write_text(
        "# malicious tracked edit\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tracked worktree is dirty"):
        physical_prior._validate_official_phystwin_worktree(repository, bindings)


def test_official_phystwin_worktree_rejects_untracked_import_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision, bindings = _official_worktree_fixture(tmp_path)
    monkeypatch.setattr(physical_prior, "OFFICIAL_PHYSTWIN_REVISION", revision)
    (repository / "qqtt/engine/torch.py").write_text(
        "raise RuntimeError('shadowed')\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="untracked importable file"):
        physical_prior._validate_official_phystwin_worktree(repository, bindings)


def test_official_phystwin_worktree_rejects_untracked_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, revision, bindings = _official_worktree_fixture(tmp_path)
    monkeypatch.setattr(physical_prior, "OFFICIAL_PHYSTWIN_REVISION", revision)
    (repository / "qqtt/engine/shadow.py").symlink_to(
        repository / "qqtt/engine/trainer_warp.py"
    )

    with pytest.raises(ValueError, match="contains a symlink"):
        physical_prior._validate_official_phystwin_worktree(repository, bindings)


def test_isolated_runpy_ignores_malicious_cwd_pythonpath_and_sitecustomize(
    tmp_path: Path,
) -> None:
    trusted = tmp_path / "trusted"
    malicious = tmp_path / "malicious"
    trusted.mkdir()
    malicious.mkdir()
    output = tmp_path / "result.txt"
    startup_marker = tmp_path / "startup-ran.txt"
    (trusted / "trusted_module.py").write_text("VALUE = 'locked'\n", encoding="utf-8")
    script = trusted / "runner.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "import trusted_module\n"
        "Path(sys.argv[1]).write_text(trusted_module.VALUE, encoding='utf-8')\n",
        encoding="utf-8",
    )
    (malicious / "trusted_module.py").write_text(
        "VALUE = 'shadowed'\n",
        encoding="utf-8",
    )
    (malicious / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(startup_marker)!r}).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(malicious)
    command = physical_prior._isolated_runpy_command(
        sys.executable,
        script,
        import_roots=(trusted,),
        arguments=(str(output),),
    )

    completed = subprocess.run(
        command,
        cwd=malicious,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text(encoding="utf-8") == "locked"
    assert not startup_marker.exists()


def test_isolated_runpy_verifies_qqtt_module_file_provenance(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    official = tmp_path / "official"
    wrong_official = tmp_path / "wrong-official"
    trusted.mkdir()
    wrong_official.mkdir()
    for relative in (
        "qqtt/__init__.py",
        "qqtt/engine/__init__.py",
        "qqtt/engine/trainer_warp.py",
        "qqtt/model/__init__.py",
        "qqtt/model/diff_simulator/__init__.py",
        "qqtt/model/diff_simulator/spring_mass_warp.py",
        "qqtt/utils/__init__.py",
    ):
        source = official / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("LOCKED = True\n", encoding="utf-8")
    output = tmp_path / "qqtt-provenance.txt"
    script = trusted / "smoke.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import qqtt\n"
        "import qqtt.engine.trainer_warp\n"
        "import qqtt.model.diff_simulator.spring_mass_warp\n"
        "import qqtt.utils\n"
        "Path(sys.argv[2]).write_text('passed', encoding='utf-8')\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    command = physical_prior._isolated_runpy_command(
        sys.executable,
        script,
        import_roots=(trusted,),
        arguments=(str(official), str(output)),
        provenance_root=official,
    )
    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert output.read_text(encoding="utf-8") == "passed"

    wrong_command = physical_prior._isolated_runpy_command(
        sys.executable,
        script,
        import_roots=(trusted,),
        arguments=(str(official), str(output)),
        provenance_root=wrong_official,
    )
    rejected = subprocess.run(
        wrong_command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "official PhysTwin import provenance changed" in rejected.stderr


def test_runner_does_not_convert_arbitrary_subprocess_failure_to_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path, frame_zero_manifest = _make_locked_frame_zero(tmp_path)
    runtime = _mock_runtime_provenance(tmp_path)
    monkeypatch.setattr(
        physical_prior,
        "validate_python_runtime",
        lambda *_: {"supplied_python_path": str(tmp_path / "python")},
    )
    monkeypatch.setattr(
        physical_prior, "validate_upstream_runtime", lambda *_: dict(runtime)
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
    runtime = _mock_runtime_provenance(tmp_path)
    supplied_python = tmp_path / "venv" / "bin" / "python"
    monkeypatch.setattr(
        physical_prior,
        "validate_python_runtime",
        lambda *_: {"supplied_python_path": str(supplied_python)},
    )
    monkeypatch.setattr(
        physical_prior, "validate_upstream_runtime", lambda *_: dict(runtime)
    )
    calls: list[list[str]] = []

    def fail_with_valid_admission(
        command: list[str], *, env: object, log_path: Path
    ) -> float:
        assert isinstance(env, dict)
        assert "PYTHONPATH" not in env
        assert "PYTHONSAFEPATH" not in env
        calls.append(command)
        assert command[1:6] == [
            "-I",
            "-B",
            "-X",
            f"pycache_prefix={physical_prior.HELD_PYCACHE_PREFIX}",
            "-c",
        ]
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
    runtime = _mock_runtime_provenance(tmp_path)
    monkeypatch.setattr(
        physical_prior,
        "validate_python_runtime",
        lambda *_: {"supplied_python_path": str(tmp_path / "python")},
    )
    monkeypatch.setattr(
        physical_prior, "validate_upstream_runtime", lambda *_: dict(runtime)
    )
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fail_warp(
        command: list[str], *, env: dict[str, str], log_path: Path
    ) -> float:
        del log_path
        calls.append((command, env))
        if len(calls) == 1:
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
    assert len(calls) == 2
    expected_flags = [
        "-I",
        "-B",
        "-X",
        f"pycache_prefix={physical_prior.HELD_PYCACHE_PREFIX}",
        "-c",
    ]
    assert all(command[1:6] == expected_flags for command, _ in calls)
    assert all("PYTHONPATH" not in env for _, env in calls)
    assert all("PYTHONSAFEPATH" not in env for _, env in calls)
    assert any(
        value.endswith("build_deform360_automatic_episode_twin.py")
        for value in calls[0][0]
    )
    assert any(
        value.endswith("run_deform360_official_phystwin_smoke.py")
        for value in calls[1][0]
    )
    assert not (tmp_path / "output" / "prediction.npz").exists()
