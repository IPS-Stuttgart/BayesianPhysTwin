"""Exact 384-node physical-backbone utilities for the fresh Deform360 study."""

from __future__ import annotations

import heapq
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from .deform360_fresh_pairwise_protocol import (
    EXPECTED_FRAME_COUNT,
    PROTOCOL_ID,
    array_sha256,
    canonical_sha256,
    file_sha256,
    write_json,
)


OFFICIAL_PHYSTWIN_REVISION = "2b6630528141b9cba5a7677c8b88b2129b4a8390"
OFFICIAL_REAL_CONFIG_SHA256 = (
    "a40a5ec2f5c978c1290810f20ed56db7cab99dc0c227adfe6b7434dfc95ead48"
)
LENGTH_SCALE_M = 0.12
ACTION_RESPONSE = 0.9
AUTONOMOUS_DRIFT_RESPONSE = 0.0
CANONICAL_NODE_COUNT = 384
MINIMUM_NODE_COUNT = 128
AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE = 2

WARP_DYNAMICS = {
    "init_spring_y": 10_000.0,
    "drag_damping": 10.0,
    "dashpot_damping": 100.0,
    "controller_radius_m": 0.03,
    "controller_max_neighbours": 1,
    "canonical_controller_patch_size": 16,
    "support_dynamics": "official-ground",
}

UPSTREAM_FILE_SHA256 = {
    "scripts/remote/build_deform360_automatic_episode_twin.py": (
        "dd43bfeaa0ddb53252e3b2d9c907c147379b2cce6b4c5d5dfa14f310fdacfa9a"
    ),
    "scripts/remote/run_deform360_official_phystwin_smoke.py": (
        "e7bf6a6c06e074ac3cdefe259c1cf5eecf8cd905dae1b710a81107ab166ca535"
    ),
    "src/causal4d_public/deform360_reusable_graph.py": (
        "97b93e32c5009f5783b2f36be7e03d4acda33f0608c9694797e8e5c72d3dd8a5"
    ),
    "src/causal4d_public/deform360_partial_graph_state.py": (
        "81536d81ce4cfd0e61074d2f4096b3160624b6afa2e1dda1d0dab16c113192a3"
    ),
    "src/causal4d_public/deform360_dense_reusable_panel.py": (
        "0861831b9ab3cf6d64833efe533073f4f444f2315c04057377f243efffd8b17e"
    ),
    "src/causal4d_public/deform360_action_support.py": (
        "132283722400ac102ec84e9b7d21974edcdac0ff750168d70860cd89c8446783"
    ),
    "src/bayesian_phystwin/phystwin_graph.py": (
        "f6f1ef8d3a1fb95fc069a550ae7db12d6b32efe80582f479efb411452062b6fb"
    ),
    "configs/causal4d_public/deform360_dense_reusable_panel_v1.json": (
        "8a90705dd38c6c90b042ed8f450e2bc7e3cffc54b965765b004d0385999d40ea"
    ),
    "configs/causal4d_public/deform360_independent_source_split_v1.json": (
        "c150b2c8ea3947fe2ffe359c5da45d321b5086cd67141c2da9f912aac154ff4a"
    ),
}

_FINGER_BASE_LEFT = np.array([-0.04246242, 0.0835, 0.0097])
_FINGER_BASE_RIGHT = np.array([0.04246242, 0.0835, 0.0107])
_TAXEL_X_M = 0.007
_TAXEL_Y0_M = -0.056
_TAXEL_Y_STEP_M = -0.002
_TAXEL_Z_PITCH_M = 0.025 / 12.0
_TAXEL_ROWS = 12
_TAXEL_COLS = 32


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_frame_zero_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the exact XYZ/RGB PLY contract emitted by pinned Deform360."""

    source = Path(path).resolve()
    with source.open("rb") as handle:
        header_lines: list[str] = []
        while True:
            raw = handle.readline()
            _require(raw, "PLY header is incomplete")
            line = raw.decode("ascii", errors="strict").rstrip("\r\n")
            header_lines.append(line)
            if line == "end_header":
                break
            _require(len(header_lines) <= 64, "PLY header is unexpectedly long")
        payload_offset = handle.tell()
    _require(header_lines[0] == "ply", "invalid PLY magic")
    _require(
        header_lines[1] in {
            "format binary_little_endian 1.0",
            "format ascii 1.0",
        },
        "unsupported PLY encoding",
    )
    vertex_lines = [
        line for line in header_lines if line.startswith("element vertex ")
    ]
    _require(len(vertex_lines) == 1, "PLY vertex count is ambiguous")
    vertex_count = int(vertex_lines[0].split()[2])
    _require(vertex_count >= MINIMUM_NODE_COUNT, "frame-zero PLY is below 128 points")
    properties = [
        line
        for line in header_lines
        if line.startswith("property ") and " list " not in f" {line} "
    ]
    _require(
        properties
        == [
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ],
        "frame-zero PLY properties changed",
    )
    if header_lines[1] == "format binary_little_endian 1.0":
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        with source.open("rb") as handle:
            handle.seek(payload_offset)
            vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
        _require(len(vertices) == vertex_count, "PLY payload is truncated")
        points = np.column_stack(
            (vertices["x"], vertices["y"], vertices["z"])
        ).astype(np.float32)
        colors = np.column_stack(
            (vertices["red"], vertices["green"], vertices["blue"])
        ).astype(np.float32)
    else:
        values = np.loadtxt(source, skiprows=len(header_lines), ndmin=2)
        _require(
            values.shape == (vertex_count, 6), "ASCII PLY payload shape changed"
        )
        points = values[:, :3].astype(np.float32)
        colors = values[:, 3:].astype(np.float32)
    colors /= np.float32(255.0)
    _require(
        points.shape == colors.shape == (vertex_count, 3)
        and np.all(np.isfinite(points))
        and np.all(np.isfinite(colors))
        and float(np.min(colors)) >= 0.0
        and float(np.max(colors)) <= 1.0,
        "frame-zero PLY values are invalid",
    )
    return points, colors


def write_frame_zero_archive(
    ply_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    points, colors = load_frame_zero_ply(ply_path)
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, points_m=points, colors=colors)
    return {
        "point_count": int(len(points)),
        "points_sha256": array_sha256(points),
        "colors_sha256": array_sha256(colors),
        "source_ply_sha256": file_sha256(ply_path),
        "archive_sha256": file_sha256(destination),
    }


def _taxel_grid_root_frame(joint: float) -> np.ndarray:
    rows, columns = np.meshgrid(
        np.arange(_TAXEL_ROWS), np.arange(_TAXEL_COLS), indexing="ij"
    )
    rows = rows.reshape(-1).astype(np.float64)
    columns = columns.reshape(-1).astype(np.float64)
    y_root = -(_TAXEL_Y0_M + _TAXEL_Y_STEP_M * columns)
    z_root = -_TAXEL_Z_PITCH_M * (11.5 - rows)
    left = np.stack(
        (
            np.full_like(y_root, _FINGER_BASE_LEFT[0] + joint + _TAXEL_X_M),
            _FINGER_BASE_LEFT[1] + y_root,
            _FINGER_BASE_LEFT[2] + z_root,
        ),
        axis=1,
    )
    right = np.stack(
        (
            np.full_like(y_root, _FINGER_BASE_RIGHT[0] - joint - _TAXEL_X_M),
            _FINGER_BASE_RIGHT[1] + y_root,
            _FINGER_BASE_RIGHT[2] + z_root,
        ),
        axis=1,
    )
    interleaved = np.empty((2 * _TAXEL_ROWS * _TAXEL_COLS, 3), dtype=np.float64)
    interleaved[0::2] = left
    interleaved[1::2] = right
    return interleaved


def _gripper_taxel_points(opening_m: float, world_from_eef: np.ndarray) -> np.ndarray:
    clamped = float(np.clip(opening_m, 0.04, 0.112))
    normalized = (clamped - 0.04) / (0.112 - 0.04)
    joint = 0.038 - normalized * (0.038 - 0.005)
    points = _taxel_grid_root_frame(joint)
    pose = np.asarray(world_from_eef, dtype=np.float64)
    _require(pose.shape == (4, 4), "end-effector pose shape changed")
    return points @ pose[:3, :3].T + pose[:3, 3]


def load_controller_trajectory(
    robot_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reproduce Deform360's controller taxel cloud from the known action."""

    source = Path(robot_path).resolve()
    with np.load(source, allow_pickle=False) as robot:
        required = {"format_version", "actions", "T_worlds", "openings", "bimanual"}
        _require(required <= set(robot.files), "robot archive is incomplete")
        _require(
            int(np.asarray(robot["format_version"]).item()) == 1,
            "robot archive format changed",
        )
        actions = np.asarray(robot["actions"], dtype=np.float64)
        poses = np.asarray(robot["T_worlds"], dtype=np.float64)
        openings = np.asarray(robot["openings"], dtype=np.float64)
        bimanual_value = np.asarray(robot["bimanual"])
    _require(
        bimanual_value.shape == () and bimanual_value.dtype == np.bool_,
        "invalid bimanual flag",
    )
    _require(
        len(actions) == len(poses) == len(openings) == EXPECTED_FRAME_COUNT,
        "known action is not the frozen 76-frame window",
    )
    _require(
        np.all(np.isfinite(actions))
        and np.all(np.isfinite(poses))
        and np.all(np.isfinite(openings)),
        "robot archive is non-finite",
    )
    bimanual = bool(bimanual_value.item())
    controllers: list[np.ndarray] = []
    for frame in range(EXPECTED_FRAME_COUNT):
        blocks = []
        for gripper in range(2 if bimanual else 1):
            pose = poses[frame, gripper] if bimanual else poses[frame]
            opening = openings[frame, gripper] if bimanual else openings[frame]
            blocks.append(_gripper_taxel_points(float(opening), pose))
        controllers.append(np.concatenate(blocks, axis=0))
    trajectory = np.stack(controllers).astype(np.float32)
    _require(np.all(np.isfinite(trajectory)), "controller trajectory is non-finite")
    return trajectory, {
        "selection_rule": "preselected_action_only_prediction_window",
        "prediction_frame_range_half_open": [0, EXPECTED_FRAME_COUNT],
        "controller_point_count": int(trajectory.shape[1]),
        "controller_trajectory_sha256": array_sha256(trajectory),
        "source_robot_sha256": file_sha256(source),
        "bimanual": bimanual,
    }


def build_prediction_only_bundle(
    frame_zero_ply: str | Path,
    known_action_archive: str | Path,
    output_path: str | Path,
    *,
    object_id: str,
    episode_id: int,
    case: str,
) -> dict[str, Any]:
    """Write the constant-object payload used by the exact automatic twin."""

    points, colors = load_frame_zero_ply(frame_zero_ply)
    controllers, action = load_controller_trajectory(known_action_archive)
    object_points = np.repeat(points[None], EXPECTED_FRAME_COUNT, axis=0)
    object_colors = np.repeat(colors[None], EXPECTED_FRAME_COUNT, axis=0)
    observed = np.ones(object_points.shape[:2], dtype=bool)
    marker = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "case": case,
        "object_id": object_id,
        "episode_id": int(episode_id),
        "object_observation_frames_used": [0],
        "known_future_robot_trajectory_used": True,
        "future_object_observations_present": False,
        "future_tactile_used": False,
        "frame_zero_ply_sha256": file_sha256(frame_zero_ply),
        "known_action_sha256": file_sha256(known_action_archive),
        "action_window": action,
    }
    payload = {
        "object_points": object_points,
        "object_colors": object_colors,
        "object_visibilities": observed,
        "object_motions_valid": observed.copy(),
        "controller_points": controllers,
        "surface_points": np.empty((0, 3), dtype=np.float32),
        "interior_points": np.empty((0, 3), dtype=np.float32),
        "prediction_only_input": marker,
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "frame_count": EXPECTED_FRAME_COUNT,
        "point_count": int(len(points)),
        "controller_point_count": int(controllers.shape[1]),
        "frame_zero_points_sha256": array_sha256(points),
        "frame_zero_colors_sha256": array_sha256(colors),
        "controller_trajectory_sha256": array_sha256(controllers),
        "output_sha256": file_sha256(destination),
        "action_window": action,
        "information_boundary": {
            "object_observation_frames_used": [0],
            "future_object_observations_present": False,
            "future_tactile_used": False,
        },
    }


def _graph_contact_distances(
    vertex_count: int,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(vertex_count)]
    for edge, length_value in zip(springs, rest_lengths, strict=True):
        first, second = map(int, edge)
        length = float(length_value)
        _require(
            0 <= first < vertex_count
            and 0 <= second < vertex_count
            and np.isfinite(length)
            and length >= 0.0,
            "invalid graph edge",
        )
        adjacency[first].append((second, length))
        adjacency[second].append((first, length))
    distances = np.full(vertex_count, np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for anchor_value in np.asarray(anchors, dtype=np.int64).reshape(-1):
        anchor = int(anchor_value)
        _require(0 <= anchor < vertex_count, "invalid contact anchor")
        distances[anchor] = 0.0
        heapq.heappush(queue, (0.0, anchor))
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for neighbour, edge_length in adjacency[node]:
            proposed = distance + edge_length
            if proposed < distances[neighbour]:
                distances[neighbour] = proposed
                heapq.heappush(queue, (proposed, neighbour))
    _require(np.all(np.isfinite(distances)), "contact does not reach the graph")
    return distances


def build_warp_backbone_arrays(
    frame_zero_points_m: np.ndarray,
    *,
    vertices: np.ndarray,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    contact_anchor_indices: np.ndarray,
    readout_weights: np.ndarray,
    driven_vertices_m: np.ndarray,
    zero_action_vertices_m: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create the exact open-27 driven-minus-zero graph-support backbone."""

    initial = np.asarray(frame_zero_points_m, dtype=np.float64)
    graph_vertices = np.asarray(vertices, dtype=np.float64)
    weights = np.asarray(readout_weights, dtype=np.float64)
    driven = np.asarray(driven_vertices_m, dtype=np.float64)
    zero = np.asarray(zero_action_vertices_m, dtype=np.float64)
    _require(initial.ndim == 2 and initial.shape[1] == 3, "invalid material points")
    _require(graph_vertices.ndim == 2 and graph_vertices.shape[1] == 3, "bad graph")
    _require(
        weights.shape == (len(initial), len(graph_vertices))
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6),
        "invalid graph readout",
    )
    expected = (EXPECTED_FRAME_COUNT, len(graph_vertices), 3)
    _require(
        driven.shape == zero.shape == expected
        and np.all(np.isfinite(driven))
        and np.all(np.isfinite(zero)),
        "invalid Warp trajectories",
    )
    distances = _graph_contact_distances(
        len(graph_vertices), springs, rest_lengths, contact_anchor_indices
    )
    node_support = np.exp(-distances / LENGTH_SCALE_M)
    support = np.clip(weights @ node_support, 0.0, 1.0)
    driven_readout = np.einsum("mn,tnc->tmc", weights, driven, optimize=True)
    zero_readout = np.einsum("mn,tnc->tmc", weights, zero, optimize=True)
    offset = initial - zero_readout[0]
    driven_readout += offset[None]
    zero_readout += offset[None]
    prediction = initial[None] + ACTION_RESPONSE * support[None, :, None] * (
        driven_readout - zero_readout
    )
    persistence = np.repeat(initial[None], EXPECTED_FRAME_COUNT, axis=0)
    arrays = {
        "prediction_m": prediction.astype(np.float32),
        "persistence_m": persistence.astype(np.float32),
        "driven_readout_m": driven_readout.astype(np.float32),
        "zero_action_readout_m": zero_readout.astype(np.float32),
        "action_support": support.astype(np.float32),
        "frame_zero_points_m": initial.astype(np.float32),
    }
    _require(
        np.array_equal(arrays["prediction_m"][0], arrays["frame_zero_points_m"])
        and np.array_equal(arrays["persistence_m"], persistence.astype(np.float32)),
        "frame-zero material identity changed",
    )
    return arrays


def build_persistence_backbone_arrays(
    frame_zero_points_m: np.ndarray,
) -> dict[str, np.ndarray]:
    initial = np.asarray(frame_zero_points_m, dtype=np.float32)
    _require(
        initial.ndim == 2
        and initial.shape[1] == 3
        and len(initial) >= MINIMUM_NODE_COUNT
        and np.all(np.isfinite(initial)),
        "invalid persistence fallback points",
    )
    persistence = np.repeat(initial[None], EXPECTED_FRAME_COUNT, axis=0)
    return {
        "prediction_m": persistence.copy(),
        "persistence_m": persistence.copy(),
        "driven_readout_m": persistence.copy(),
        "zero_action_readout_m": persistence.copy(),
        "action_support": np.zeros(len(initial), dtype=np.float32),
        "frame_zero_points_m": initial.copy(),
    }


def write_physical_artifacts(
    archive_path: str | Path,
    manifest_path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    protocol_config_sha256: str,
    cohort_lock_sha256: str,
    case_record: Mapping[str, Any],
    physical_mode: str,
    input_files: Mapping[str, str | Path],
    runtime_provenance: Mapping[str, Any],
    fallback_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal a successful Warp backbone or preregistered persistence fallback."""

    _require(
        physical_mode in {"warp_twin", "source_admission_persistence_fallback"},
        "physical mode changed",
    )
    _require(
        (physical_mode == "source_admission_persistence_fallback")
        == (fallback_diagnostics is not None),
        "fallback diagnostics disagree with physical mode",
    )
    expected_names = {
        "prediction_m",
        "persistence_m",
        "driven_readout_m",
        "zero_action_readout_m",
        "action_support",
        "frame_zero_points_m",
    }
    _require(set(arrays) == expected_names, "physical archive arrays changed")
    stored = {name: np.asarray(arrays[name]) for name in sorted(expected_names)}
    archive = Path(archive_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **stored)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "Deform360FreshPairwisePhysicalPrediction",
        "protocol_id": PROTOCOL_ID,
        "protocol_config_sha256": protocol_config_sha256,
        "cohort_lock_sha256": cohort_lock_sha256,
        **dict(case_record),
        "physical_mode": physical_mode,
        "physical_admitted": physical_mode == "warp_twin",
        "fallback_diagnostics": (
            None if fallback_diagnostics is None else dict(fallback_diagnostics)
        ),
        "frozen_predictor": {
            "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
            "official_real_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
            "length_scale_m": LENGTH_SCALE_M,
            "action_response": ACTION_RESPONSE,
            "autonomous_drift_response": AUTONOMOUS_DRIFT_RESPONSE,
            "canonical_observed_node_count": CANONICAL_NODE_COUNT,
            "minimum_observed_node_count": MINIMUM_NODE_COUNT,
            "frame_count": EXPECTED_FRAME_COUNT,
            "warp_dynamics": dict(WARP_DYNAMICS),
        },
        "physical_prediction_archive": {
            "path": str(archive),
            "file_sha256": file_sha256(archive),
            "array_sha256": {
                name: array_sha256(value) for name, value in stored.items()
            },
        },
        "input_files": {
            name: {
                "path": str(Path(path).resolve()),
                "file_sha256": file_sha256(path),
            }
            for name, path in sorted(input_files.items())
        },
        "runtime_provenance": dict(runtime_provenance),
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "future_tactile_read": False,
            "outcome_manifest_read": False,
            "prediction_hashed_before_future_outcome_scoring": True,
        },
        "passed": True,
    }
    payload["result_sha256"] = canonical_sha256(
        payload, digest_key="result_sha256"
    )
    write_json(manifest_path, payload)
    return payload


__all__ = [
    "ACTION_RESPONSE",
    "AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE",
    "AUTONOMOUS_DRIFT_RESPONSE",
    "CANONICAL_NODE_COUNT",
    "LENGTH_SCALE_M",
    "MINIMUM_NODE_COUNT",
    "OFFICIAL_PHYSTWIN_REVISION",
    "OFFICIAL_REAL_CONFIG_SHA256",
    "UPSTREAM_FILE_SHA256",
    "WARP_DYNAMICS",
    "build_persistence_backbone_arrays",
    "build_prediction_only_bundle",
    "build_warp_backbone_arrays",
    "load_controller_trajectory",
    "load_frame_zero_ply",
    "write_frame_zero_archive",
    "write_physical_artifacts",
]
