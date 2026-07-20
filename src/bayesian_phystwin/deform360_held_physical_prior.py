"""Prediction-only Deform360 physical priors for the held protocol.

The numerical twin and Warp rollout are deliberately reused from the frozen
``reusable-trust-fresh-code/Bayesian-PhysTwin`` tree.  This module adds the
missing security boundary around that code:

* the case must be in the exact calibration or confirmation whitelist;
* the only object input is a validated :class:`Deform360HeldFrameZeroBundle`;
* the known robot action is selected without reading object motion;
* the upstream source files, official PhysTwin revision, and Warp config are
  checked before a subprocess is started; and
* driven and zero-action trajectories are converted into the frozen
  graph-support prediction and hashed before any outcome operation.

No function in this module accepts an outcome path.  The resulting four files
are inputs to :func:`bayesian_phystwin.deform360_held_protocol.create_physical_prior_seal`.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
from pathlib import Path
import pickle
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_held_protocol import (
    FRAME_COUNT,
    PROTOCOL_ID,
    create_physical_prior_seal,
    held_artifact_sha256,
    held_contract_sha256,
    load_held_protocol_lock,
    validate_frame_zero_bundle_manifest,
)


ARTIFACT_KIND = "Deform360HeldPhysicalPrediction"
PREDICTION_INPUT_KIND = "Deform360PredictionOnlyInput"
SCHEMA_VERSION = 1

OFFICIAL_PHYSTWIN_REVISION = "2b6630528141b9cba5a7677c8b88b2129b4a8390"
OFFICIAL_REAL_CONFIG_SHA256 = (
    "a40a5ec2f5c978c1290810f20ed56db7cab99dc0c227adfe6b7434dfc95ead48"
)
LENGTH_SCALE_M = 0.12
ACTION_RESPONSE = 0.9
AUTONOMOUS_DRIFT_RESPONSE = 0.0
CANONICAL_NODE_COUNT = 384
MINIMUM_NODE_COUNT = 128

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
    "src/causal4d_public/deform360_contact_conditioned_action.py": (
        "1d4e2bbd4389d8d7055d0803f3feda3ea540d45123e0aa3f646bccf2cfa6c57e"
    ),
    "src/causal4d_public/deform360_dense_source.py": (
        "6c9ffa0043302079acf303f23af9e9ebb895f0aa8cf03930effe8936a879bb29"
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

UPSTREAM_LOCK_BINDING_BY_PATH = {
    "scripts/remote/build_deform360_automatic_episode_twin.py": (
        "upstream_automatic_twin_builder"
    ),
    "scripts/remote/run_deform360_official_phystwin_smoke.py": (
        "upstream_official_phystwin_smoke"
    ),
    "src/causal4d_public/deform360_reusable_graph.py": (
        "upstream_reusable_graph_source"
    ),
    "src/causal4d_public/deform360_partial_graph_state.py": (
        "upstream_partial_graph_state_source"
    ),
    "src/causal4d_public/deform360_dense_reusable_panel.py": (
        "upstream_dense_reusable_panel_source"
    ),
    "src/causal4d_public/deform360_action_support.py": (
        "upstream_action_support_source"
    ),
    "src/causal4d_public/deform360_contact_conditioned_action.py": (
        "upstream_contact_conditioned_action_source"
    ),
    "src/causal4d_public/deform360_dense_source.py": "upstream_dense_source",
    "src/bayesian_phystwin/phystwin_graph.py": "upstream_phystwin_graph_source",
    "configs/causal4d_public/deform360_dense_reusable_panel_v1.json": (
        "upstream_dense_reusable_panel_config"
    ),
    "configs/causal4d_public/deform360_independent_source_split_v1.json": (
        "upstream_independent_source_split_config"
    ),
}

UPSTREAM_RUNTIME_BUNDLE_CONTRACT = {
    "artifact_kind": "Deform360HeldUpstreamRuntimeBundleV1",
    "files": [
        {"path": path, "sha256": UPSTREAM_FILE_SHA256[path]}
        for path in sorted(UPSTREAM_FILE_SHA256)
    ],
}

HELD_PHYSICAL_NUMERIC_CONTRACT = {
    "contract_id": "deform360-held-physical-prior-v1",
    "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
    "official_real_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
    "length_scale_m": LENGTH_SCALE_M,
    "action_response": ACTION_RESPONSE,
    "autonomous_drift_response": AUTONOMOUS_DRIFT_RESPONSE,
    "canonical_node_count": CANONICAL_NODE_COUNT,
    "minimum_node_count": MINIMUM_NODE_COUNT,
    "warp_dynamics": WARP_DYNAMICS,
    "upstream_file_sha256": UPSTREAM_FILE_SHA256,
}

FRAME_ZERO_ARRAYS = frozenset(
    {
        "frame_indices",
        "camera_names",
        "rgb_frame0",
        "mask_frame0",
        "depth_frame0_m",
        "depth_valid_frame0",
        "intrinsics",
        "camera_to_world",
        "projection_world_to_pixel",
        "object_points_world_m",
        "object_colors_rgb",
        "object_color_support_count",
        "visual_hull_points_world_m",
    }
)

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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash an array exactly like the frozen independent-source code."""

    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination.resolve()


def _bound_file(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    _require(
        source.is_file() and not source.is_symlink(), f"missing regular file: {source}"
    )
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }


def _validate_bound_file(
    record: Mapping[str, Any], *, label: str, allow_metadata: bool = False
) -> Path:
    _require(isinstance(record, Mapping), f"{label} binding is missing")
    path = record.get("path")
    _require(isinstance(path, str) and bool(path), f"{label} path is missing")
    observed = _bound_file(path)
    bound_record = {key: record.get(key) for key in observed}
    _require(observed == bound_record, f"{label} binding changed")
    if not allow_metadata:
        _require(set(record) == set(observed), f"{label} binding has unexpected fields")
    return Path(observed["path"])


def validate_upstream_runtime(
    upstream_repo: str | Path,
    official_phystwin_repo: str | Path,
    official_config: str | Path,
) -> dict[str, Any]:
    """Fail before computation if any frozen numerical implementation moved."""

    upstream = Path(upstream_repo).resolve()
    official = Path(official_phystwin_repo).resolve()
    config = Path(official_config).resolve()
    observed_files: dict[str, str] = {}
    for relative, expected in UPSTREAM_FILE_SHA256.items():
        path = upstream / relative
        _require(path.is_file(), f"missing frozen upstream file: {relative}")
        observed = sha256_file(path)
        _require(observed == expected, f"frozen upstream file changed: {relative}")
        observed_files[relative] = observed
    _require(config.is_file(), "official PhysTwin config is missing")
    _require(
        sha256_file(config) == OFFICIAL_REAL_CONFIG_SHA256,
        "official PhysTwin real.yaml changed",
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=official,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(
        revision == OFFICIAL_PHYSTWIN_REVISION, "official PhysTwin revision changed"
    )
    return {
        "official_phystwin_revision": revision,
        "official_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
        "upstream_file_sha256": observed_files,
    }


def _load_frame_zero_geometry(
    manifest_path: str | Path,
    lock_path: str | Path,
    *,
    case_name: str,
    role: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest = validate_frame_zero_bundle_manifest(
        manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    bundle_path = _validate_bound_file(manifest["bundle"], label="frame-zero bundle")
    with np.load(bundle_path, allow_pickle=False) as stored:
        _require(
            set(stored.files) == FRAME_ZERO_ARRAYS,
            "frame-zero bundle array set changed",
        )
        frame_indices = np.asarray(stored["frame_indices"])
        points = np.asarray(stored["object_points_world_m"], dtype=np.float32)
        colors = np.asarray(stored["object_colors_rgb"])
        camera_names = np.asarray(stored["camera_names"])
        rgb = np.asarray(stored["rgb_frame0"])
        masks = np.asarray(stored["mask_frame0"])
        depth = np.asarray(stored["depth_frame0_m"])
        valid = np.asarray(stored["depth_valid_frame0"])
        intrinsics = np.asarray(stored["intrinsics"])
        camera_to_world = np.asarray(stored["camera_to_world"])
        projections = np.asarray(stored["projection_world_to_pixel"])
    _require(
        np.array_equal(frame_indices, np.array([0])), "bundle contains a nonzero frame"
    )
    _require(
        points.ndim == 2 and points.shape[1] == 3, "invalid frame-zero object points"
    )
    _require(len(points) >= MINIMUM_NODE_COUNT, "frame-zero point count is below 128")
    _require(colors.shape == points.shape, "frame-zero point colors differ from points")
    _require(np.all(np.isfinite(points)), "frame-zero object points are non-finite")
    _require(np.all(np.isfinite(colors)), "frame-zero object colors are non-finite")
    camera_count = len(camera_names)
    _require(camera_count >= 2, "frame-zero bundle has fewer than two cameras")
    _require(
        rgb.ndim == 4 and rgb.shape[0] == camera_count, "invalid frame-zero RGB stack"
    )
    _require(masks.shape == rgb.shape[:3], "frame-zero masks differ from RGB")
    _require(
        depth.shape == masks.shape and valid.shape == masks.shape, "invalid depth stack"
    )
    _require(intrinsics.shape == (camera_count, 3, 3), "invalid camera intrinsics")
    _require(camera_to_world.shape == (camera_count, 4, 4), "invalid camera poses")
    _require(projections.shape == (camera_count, 3, 4), "invalid camera projections")
    _require(np.all(np.isfinite(intrinsics)), "non-finite camera intrinsics")
    _require(np.all(np.isfinite(camera_to_world)), "non-finite camera poses")
    _require(np.all(np.isfinite(projections)), "non-finite camera projections")
    if colors.dtype.kind in "ui":
        colors = colors.astype(np.float32) / float(np.iinfo(colors.dtype).max)
    else:
        colors = colors.astype(np.float32)
    _require(
        float(np.min(colors)) >= -1e-6 and float(np.max(colors)) <= 1.0 + 1e-6,
        "frame-zero colors must lie in [0,1]",
    )
    return manifest, points, np.clip(colors, 0.0, 1.0)


def _controller_centres(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    _require(np.all(np.isfinite(values)), "robot actions are non-finite")
    if values.ndim == 3:
        values = values[:, None, :, :]
    _require(
        values.ndim == 4 and values.shape[-1] == 3 and values.shape[2] >= 1,
        "robot actions have an invalid shape",
    )
    return np.mean(values, axis=2)


def _closure_confidence(openings: np.ndarray) -> np.ndarray:
    aperture = np.asarray(openings, dtype=np.float64)
    if aperture.ndim == 1:
        aperture = aperture[:, None]
    _require(aperture.ndim == 2 and np.all(np.isfinite(aperture)), "invalid openings")
    low = np.quantile(aperture, 0.1, axis=0)
    high = np.quantile(aperture, 0.9, axis=0)
    span = high - low
    confidence = np.ones_like(aperture)
    varying = span > 1e-9
    confidence[:, varying] = np.clip(
        (high[varying] - aperture[:, varying]) / span[varying], 0.0, 1.0
    )
    return confidence


def select_action_window(
    actions: np.ndarray,
    openings: np.ndarray,
    *,
    frame_count: int = FRAME_COUNT,
) -> dict[str, Any]:
    """Port the frozen action-only 81-frame selection, then drop its 5-frame tail."""

    centres = _controller_centres(actions)
    closed = _closure_confidence(openings)
    _require(closed.shape == centres.shape[:2], "openings differ from action groups")
    staging_length = frame_count + 5
    _require(
        len(centres) >= staging_length, "robot trajectory is shorter than 81 frames"
    )
    starts = np.arange(8, len(centres) - staging_length + 1, 6, dtype=np.int64)
    _require(len(starts) > 0, "robot trajectory has no locked action candidate")
    candidates: list[tuple[float, int]] = []
    for start_value in starts:
        start = int(start_value)
        selected = centres[start : start + staging_length]
        selected_closed = closed[start : start + staging_length]
        step = np.linalg.norm(np.diff(selected, axis=0), axis=-1)
        weighted = step * np.minimum(selected_closed[:-1], selected_closed[1:])
        candidates.append((float(np.mean(np.sum(weighted, axis=0))), start))
    # ``max`` preserves the first candidate on equal values, the locked earliest tie break.
    best_score = max(score for score, _ in candidates)
    selected_start = next(start for score, start in candidates if score == best_score)
    return {
        "selection_rule": "maximum_mean_closed_weighted_gripper_path",
        "candidate_start_frame": 8,
        "candidate_stride_frames": 6,
        "candidate_count": len(candidates),
        "staging_frame_range_half_open": [
            selected_start,
            selected_start + staging_length,
        ],
        "prediction_frame_range_half_open": [
            selected_start,
            selected_start + frame_count,
        ],
        "tracking_tail_frames_skipped": 5,
        "mean_closed_weighted_path_length_m": best_score,
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
    return points @ pose[:3, :3].T + pose[:3, 3]


def load_controller_trajectory(
    robot_path: str | Path,
    *,
    frame_count: int = FRAME_COUNT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the bound action and reproduce Deform360's controller taxel cloud."""

    with np.load(robot_path, allow_pickle=False) as robot:
        required = {"format_version", "actions", "T_worlds", "openings", "bimanual"}
        _require(
            required.issubset(robot.files), "robot archive is missing required arrays"
        )
        _require(
            int(np.asarray(robot["format_version"]).item()) == 1, "robot format changed"
        )
        actions = np.asarray(robot["actions"], dtype=np.float64)
        poses = np.asarray(robot["T_worlds"], dtype=np.float64)
        openings = np.asarray(robot["openings"], dtype=np.float64)
        bimanual_value = np.asarray(robot["bimanual"])
    _require(
        bimanual_value.shape == () and bimanual_value.dtype == np.bool_,
        "invalid bimanual flag",
    )
    bimanual = bool(bimanual_value.item())
    _require(len(actions) == len(poses) == len(openings), "robot frame axes differ")
    _require(np.all(np.isfinite(actions)), "robot actions are non-finite")
    _require(np.all(np.isfinite(poses)), "robot poses are non-finite")
    _require(np.all(np.isfinite(openings)), "robot openings are non-finite")
    selection = (
        {
            "selection_rule": "preselected_exact_prediction_window",
            "candidate_start_frame": 0,
            "candidate_stride_frames": None,
            "candidate_count": 1,
            "staging_frame_range_half_open": [0, len(actions)],
            "prediction_frame_range_half_open": [0, frame_count],
            "tracking_tail_frames_skipped": max(0, len(actions) - frame_count),
            "mean_closed_weighted_path_length_m": None,
        }
        if len(actions) in {frame_count, frame_count + 5}
        else select_action_window(actions, openings, frame_count=frame_count)
    )
    start, stop = selection["prediction_frame_range_half_open"]
    poses = poses[start:stop]
    openings = openings[start:stop]
    _require(len(poses) == frame_count, "selected robot window is not 76 frames")
    controllers = []
    for frame in range(frame_count):
        blocks = []
        gripper_count = 2 if bimanual else 1
        for gripper in range(gripper_count):
            pose = poses[frame, gripper] if bimanual else poses[frame]
            opening = openings[frame, gripper] if bimanual else openings[frame]
            blocks.append(_gripper_taxel_points(float(opening), pose))
        controllers.append(np.concatenate(blocks, axis=0))
    trajectory = np.stack(controllers).astype(np.float32)
    _require(np.all(np.isfinite(trajectory)), "controller trajectory is non-finite")
    selection["controller_point_count"] = int(trajectory.shape[1])
    selection["controller_trajectory_sha256"] = sha256_array(trajectory)
    return trajectory, selection


def build_prediction_only_artifacts(
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path,
    *,
    case_name: str,
    role: str = "calibration",
) -> dict[str, Any]:
    """Create the exact constant-object PhysTwin contract from one frame."""

    manifest, points, colors = _load_frame_zero_geometry(
        frame_zero_manifest_path,
        lock_path,
        case_name=case_name,
        role=role,
    )
    alignment = manifest.get("action_alignment", {})
    _require(isinstance(alignment, Mapping), "frame-zero action alignment is missing")
    staging_range = alignment.get("selected_raw_frame_range_half_open")
    prediction_range = alignment.get("prediction_raw_frame_range_half_open")
    _require(
        isinstance(staging_range, list)
        and len(staging_range) == 2
        and int(staging_range[1]) - int(staging_range[0]) == FRAME_COUNT + 5,
        "frame-zero action alignment is not the frozen 81-frame window",
    )
    _require(
        isinstance(prediction_range, list)
        and len(prediction_range) == 2
        and int(prediction_range[0]) == int(staging_range[0])
        and int(prediction_range[1]) - int(prediction_range[0]) == FRAME_COUNT,
        "frame-zero selected action is not the frozen 76-frame window",
    )
    robot_path = _validate_bound_file(
        alignment.get("selected_action_bundle", {}),
        label="selected robot trajectory",
    )
    controllers, action_window = load_controller_trajectory(robot_path)
    object_points = np.repeat(points[None], FRAME_COUNT, axis=0).astype(np.float32)
    object_colors = np.repeat(colors[None], FRAME_COUNT, axis=0).astype(np.float32)
    observed = np.ones(object_points.shape[:2], dtype=bool)
    marker = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": manifest["object_id"],
        "episode_id": int(manifest["episode_id"]),
        "role": role,
        "object_observation_frames_used": [0],
        "known_future_robot_trajectory_used": True,
        "future_object_observations_present": False,
        "future_tactile_used": False,
        "frame_zero_manifest_artifact_sha256": manifest["artifact_sha256"],
        "frame_zero_bundle_sha256": manifest["bundle"]["sha256"],
        "source_robot_trajectory_sha256": manifest["action_inputs"]["robot_trajectory"][
            "sha256"
        ],
        "selected_robot_trajectory_sha256": sha256_file(robot_path),
        "action_window": action_window,
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
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": PREDICTION_INPUT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": manifest["object_id"],
        "episode_id": int(manifest["episode_id"]),
        "role": role,
        "frame_count": FRAME_COUNT,
        "point_count": len(points),
        "controller_point_count": int(controllers.shape[1]),
        "frame_zero_points_sha256": sha256_array(points),
        "frame_zero_colors_sha256": sha256_array(colors),
        "controller_trajectory_sha256": sha256_array(controllers),
        "action_window": action_window,
        "input_sha256": {
            "held_lock": sha256_file(lock_path),
            "frame_zero_manifest": sha256_file(frame_zero_manifest_path),
            "frame_zero_bundle": manifest["bundle"]["sha256"],
            "robot_trajectory": manifest["action_inputs"]["robot_trajectory"]["sha256"],
            "robot_metadata": manifest["action_inputs"]["robot_metadata"]["sha256"],
            "selected_robot_trajectory": sha256_file(robot_path),
        },
        "output_sha256": sha256_file(destination),
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_tactile_read": False,
            "outcome_created": False,
            "outcome_read": False,
        },
        "passed": True,
    }
    summary["artifact_sha256"] = held_artifact_sha256(summary)
    _write_json(summary_path, summary)
    return summary


def _run_logged(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    log_path: Path,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            list(command),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n"
            + "\n".join(tail)
        )
    return elapsed


def _expected_warp_overrides() -> dict[str, Any]:
    return {
        "controller_max_neighbours": WARP_DYNAMICS["controller_max_neighbours"],
        "controller_radius": WARP_DYNAMICS["controller_radius_m"],
        "dashpot_damping": WARP_DYNAMICS["dashpot_damping"],
        "drag_damping": WARP_DYNAMICS["drag_damping"],
        "init_spring_Y": WARP_DYNAMICS["init_spring_y"],
    }


def _load_prediction_pickle(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open("rb") as stream:
        value = pickle.load(stream)
    _require(isinstance(value, Mapping), "prediction-only pickle is not a mapping")
    return value


def _graph_contact_distances(
    vertex_count: int,
    springs: np.ndarray,
    rest_lengths: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(vertex_count)]
    for edge, length in zip(springs, rest_lengths):
        first, second = map(int, edge)
        distance = float(length)
        _require(
            0 <= first < vertex_count and 0 <= second < vertex_count,
            "invalid graph edge",
        )
        _require(np.isfinite(distance) and distance >= 0.0, "invalid graph rest length")
        adjacency[first].append((second, distance))
        adjacency[second].append((first, distance))
    distances = np.full(vertex_count, np.inf, dtype=np.float64)
    queue: list[tuple[float, int]] = []
    for anchor_value in anchors:
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
    _require(
        np.all(np.isfinite(distances)), "contact does not reach the material graph"
    )
    return distances


def build_physical_prediction_archive(
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    readout_path: str | Path,
    twin_summary_path: str | Path,
    driven_result_path: str | Path,
    zero_result_path: str | Path,
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    case_name: str,
    role: str,
    runtime_provenance: Mapping[str, Any],
    stage_runtime_seconds: Mapping[str, float],
) -> dict[str, Any]:
    """Port the frozen driven-minus-zero graph-support sealer for held cases."""

    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    prediction_data = _load_prediction_pickle(prediction_data_path)
    points = np.asarray(prediction_data["object_points"], dtype=np.float64)
    controllers = np.asarray(prediction_data["controller_points"], dtype=np.float64)
    marker = prediction_data.get("prediction_only_input", {})
    _require(
        points.ndim == 3 and points.shape[0] == FRAME_COUNT, "invalid prediction points"
    )
    _require(
        controllers.ndim == 3 and len(controllers) == FRAME_COUNT, "invalid controllers"
    )
    _require(
        np.array_equal(points, np.repeat(points[:1], FRAME_COUNT, axis=0)),
        "prediction input contains changing future object geometry",
    )
    _require(
        marker.get("object_observation_frames_used") == [0], "prediction marker changed"
    )
    _require(
        marker.get("future_object_observations_present") is False,
        "future object data present",
    )
    _require(marker.get("future_tactile_used") is False, "future tactile present")
    twin = _load_json(twin_summary_path)
    _require(twin.get("passed") is True, "automatic twin failed admission")
    _require(twin.get("object_id") == frame_zero["object_id"], "twin object changed")
    _require(
        int(twin.get("episode_id", -1)) == int(frame_zero["episode_id"]),
        "twin episode changed",
    )
    boundary = twin.get("information_boundary", {})
    _require(boundary.get("target_access") is False, "automatic twin accessed target")
    _require(
        boundary.get("post_initial_object_observation_used") is False,
        "twin used future object",
    )
    _require(
        twin.get("input_sha256", {}).get("episode_final_data")
        == sha256_file(prediction_data_path),
        "twin used another prediction input",
    )
    _require(
        twin.get("output_sha256", {}).get("simulator_final_data")
        == sha256_file(simulator_data_path),
        "simulator data changed after twin construction",
    )
    with np.load(graph_path, allow_pickle=False) as graph:
        vertices = np.asarray(graph["vertices"], dtype=np.float64)
        springs = np.asarray(graph["springs"], dtype=np.int64)
        rest_lengths = np.asarray(graph["rest_lengths"], dtype=np.float64)
        anchors = np.asarray(graph["contact_anchor_indices"], dtype=np.int64)
        observed_nodes = int(np.asarray(graph["observed_node_count"]).item())
        graph_semantic_sha256 = str(np.asarray(graph["reusable_graph_sha256"]).item())
    _require(
        MINIMUM_NODE_COUNT <= observed_nodes <= CANONICAL_NODE_COUNT,
        "observed graph capacity is outside the frozen range",
    )
    with np.load(readout_path, allow_pickle=False) as state:
        weights = np.asarray(state["readout_weights"], dtype=np.float64)
        state_graph_sha256 = str(np.asarray(state["canonical_graph_sha256"]).item())
    _require(state_graph_sha256 == graph_semantic_sha256, "readout uses another graph")
    _require(weights.shape == (points.shape[1], len(vertices)), "readout shape changed")
    _require(
        np.all(np.isfinite(weights)) and np.all(weights >= 0.0),
        "invalid readout weights",
    )
    _require(
        np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6), "readout is not convex"
    )

    trajectories: dict[str, np.ndarray] = {}
    result_files = {
        "driven": Path(driven_result_path),
        "zero_action": Path(zero_result_path),
    }
    expected_scales = {"driven": 1.0, "zero_action": 0.0}
    for label, result_file in result_files.items():
        result = _load_json(result_file)
        _require(result.get("passed") is True, f"{label} Warp rollout failed")
        _require(
            "external_target_scoring" not in result, f"{label} rollout read a target"
        )
        _require(
            result.get("data_sha256") == sha256_file(simulator_data_path),
            "Warp data changed",
        )
        _require(
            result.get("official_phystwin_revision") == OFFICIAL_PHYSTWIN_REVISION,
            "Warp used another PhysTwin revision",
        )
        _require(
            result.get("config_sha256") == OFFICIAL_REAL_CONFIG_SHA256,
            "Warp config changed",
        )
        _require(
            result.get("config_overrides") == _expected_warp_overrides(),
            "Warp overrides changed",
        )
        _require(
            result.get("support_dynamics", {}).get("mode")
            == WARP_DYNAMICS["support_dynamics"],
            "Warp support dynamics changed",
        )
        graph_record = result.get("canonical_reusable_graph", {})
        _require(
            graph_record.get("file_sha256") == sha256_file(graph_path),
            "Warp graph file changed",
        )
        _require(
            graph_record.get("reusable_graph_sha256") == graph_semantic_sha256,
            "Warp graph changed",
        )
        _require(
            int(graph_record.get("controller_patch_size_per_anchor", -1))
            == WARP_DYNAMICS["canonical_controller_patch_size"],
            "Warp controller patch changed",
        )
        _require(
            float(
                result.get("realized_actuation", {}).get(
                    "controller_displacement_scale", -1.0
                )
            )
            == expected_scales[label],
            f"{label} action scale changed",
        )
        trajectory_path = result_file.with_name("official_phystwin_trajectory.npz")
        _require(
            result.get("trajectory_sha256") == sha256_file(trajectory_path),
            "trajectory changed",
        )
        with np.load(trajectory_path, allow_pickle=False) as trajectory_file:
            trajectory = np.asarray(trajectory_file["vertices"], dtype=np.float64)
        _require(
            trajectory.ndim == 3
            and trajectory.shape[0] == FRAME_COUNT
            and trajectory.shape[1] >= len(vertices)
            and trajectory.shape[2] == 3
            and np.all(np.isfinite(trajectory)),
            f"invalid {label} trajectory",
        )
        trajectories[label] = trajectory[:, : len(vertices)]

    distances = _graph_contact_distances(len(vertices), springs, rest_lengths, anchors)
    node_support = np.exp(-distances / LENGTH_SCALE_M)
    support = np.clip(weights @ node_support, 0.0, 1.0)
    driven_readout = np.einsum(
        "mn,tnc->tmc", weights, trajectories["driven"], optimize=True
    )
    zero_readout = np.einsum(
        "mn,tnc->tmc", weights, trajectories["zero_action"], optimize=True
    )
    initial = points[0]
    offset = initial - zero_readout[0]
    driven_readout += offset[None]
    zero_readout += offset[None]
    prediction = initial[None] + ACTION_RESPONSE * support[None, :, None] * (
        driven_readout - zero_readout
    )
    persistence = np.repeat(initial[None], FRAME_COUNT, axis=0)
    _require(np.all(np.isfinite(prediction)), "physical prediction is non-finite")
    arrays = {
        "prediction_m": prediction.astype(np.float32),
        "persistence_m": persistence.astype(np.float32),
        "driven_readout_m": driven_readout.astype(np.float32),
        "zero_action_readout_m": zero_readout.astype(np.float32),
        "action_support": support.astype(np.float32),
        "frame_zero_points_m": initial.astype(np.float32),
    }
    archive = Path(archive_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **arrays)
    input_paths = {
        "prediction_only_input": prediction_data_path,
        "simulator_final_data": simulator_data_path,
        "episode_graph": graph_path,
        "state_artifact": readout_path,
        "twin_summary": twin_summary_path,
        "driven_result": driven_result_path,
        "zero_action_result": zero_result_path,
        "driven_trajectory": Path(driven_result_path).with_name(
            "official_phystwin_trajectory.npz"
        ),
        "zero_action_trajectory": Path(zero_result_path).with_name(
            "official_phystwin_trajectory.npz"
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": frame_zero["object_id"],
        "episode_id": int(frame_zero["episode_id"]),
        "role": role,
        "frozen_predictor": {
            "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
            "official_real_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
            "length_scale_m": LENGTH_SCALE_M,
            "action_response": ACTION_RESPONSE,
            "autonomous_drift_response": AUTONOMOUS_DRIFT_RESPONSE,
            "frame_count": FRAME_COUNT,
            "observed_graph_node_count": observed_nodes,
            "total_graph_node_count": len(vertices),
            "point_count": points.shape[1],
            "warp_dynamics": dict(WARP_DYNAMICS),
        },
        "physical_prediction_archive": {
            **_bound_file(archive),
            "array_sha256": {
                name: sha256_array(value) for name, value in arrays.items()
            },
        },
        "input_files": {name: _bound_file(path) for name, path in input_paths.items()},
        "held_lock_sha256": sha256_file(lock_path),
        "frame_zero_manifest_sha256": sha256_file(frame_zero_manifest_path),
        "frame_zero_manifest_artifact_sha256": frame_zero["artifact_sha256"],
        "runtime_provenance": dict(runtime_provenance),
        "stage_runtime_seconds": {
            key: float(value) for key, value in stage_runtime_seconds.items()
        },
        "information_boundary": {
            "object_observation_frames_used": [0],
            "known_future_robot_action_read": True,
            "future_object_rgb_read": False,
            "future_object_geometry_read": False,
            "future_object_track_read": False,
            "future_object_visibility_read": False,
            "future_tactile_read": False,
            "external_target_scoring_in_warp": False,
            "outcome_created": False,
            "outcome_read": False,
            "physical_prediction_hashed_before_outcome": True,
        },
        "passed": True,
    }
    manifest["artifact_sha256"] = held_artifact_sha256(manifest)
    _write_json(manifest_path, manifest)
    return validate_physical_prediction_manifest(manifest_path, verify_archive=True)


def validate_physical_prediction_manifest(
    manifest_path: str | Path,
    *,
    verify_archive: bool = False,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "unsupported prediction schema",
    )
    _require(
        manifest.get("artifact_kind") == ARTIFACT_KIND,
        "unsupported prediction artifact",
    )
    _require(manifest.get("protocol_id") == PROTOCOL_ID, "prediction protocol changed")
    _require(manifest.get("passed") is True, "physical prediction did not pass")
    frozen = manifest.get("frozen_predictor", {})
    _require(
        frozen.get("official_phystwin_revision") == OFFICIAL_PHYSTWIN_REVISION,
        "revision changed",
    )
    _require(
        frozen.get("official_real_config_sha256") == OFFICIAL_REAL_CONFIG_SHA256,
        "config changed",
    )
    _require(
        float(frozen.get("length_scale_m", -1.0)) == LENGTH_SCALE_M,
        "length scale changed",
    )
    _require(
        float(frozen.get("action_response", -1.0)) == ACTION_RESPONSE,
        "action response changed",
    )
    _require(
        float(frozen.get("autonomous_drift_response", -1.0))
        == AUTONOMOUS_DRIFT_RESPONSE,
        "autonomous drift changed",
    )
    _require(int(frozen.get("frame_count", -1)) == FRAME_COUNT, "frame count changed")
    _require(frozen.get("warp_dynamics") == WARP_DYNAMICS, "Warp dynamics changed")
    archive_record = manifest.get("physical_prediction_archive", {})
    archive_path = _validate_bound_file(
        archive_record,
        label="physical prediction archive",
        allow_metadata=True,
    )
    inputs = manifest.get("input_files", {})
    _require(
        isinstance(inputs, Mapping) and bool(inputs), "prediction inputs are missing"
    )
    for label, record in inputs.items():
        _validate_bound_file(record, label=str(label))
    boundary = manifest.get("information_boundary", {})
    _require(
        boundary.get("object_observation_frames_used") == [0]
        and boundary.get("known_future_robot_action_read") is True
        and boundary.get("future_object_rgb_read") is False
        and boundary.get("future_object_geometry_read") is False
        and boundary.get("future_object_track_read") is False
        and boundary.get("future_object_visibility_read") is False
        and boundary.get("future_tactile_read") is False
        and boundary.get("external_target_scoring_in_warp") is False
        and boundary.get("outcome_created") is False
        and boundary.get("outcome_read") is False
        and boundary.get("physical_prediction_hashed_before_outcome") is True,
        "physical prediction crossed its information boundary",
    )
    _require(
        manifest.get("artifact_sha256") == held_artifact_sha256(manifest),
        "physical prediction manifest checksum changed",
    )
    if verify_archive:
        expected = archive_record.get("array_sha256")
        _require(isinstance(expected, Mapping), "archive array checksums are missing")
        with np.load(archive_path, allow_pickle=False) as stored:
            _require(
                set(stored.files) == set(expected),
                "prediction archive array set changed",
            )
            for name in stored.files:
                _require(
                    sha256_array(stored[name]) == expected[name],
                    f"{name} checksum changed",
                )
    return manifest


def run_held_physical_prior(
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    output_dir: str | Path,
    *,
    case_name: str,
    role: str = "calibration",
    upstream_repo: str | Path,
    official_phystwin_repo: str | Path,
    official_config: str | Path,
    deform360_repo: str | Path,
    python: str | Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run and seal one prediction-only held/calibration physical forecast."""

    # Both authorization and all source/config hashes are checked before a GPU process.
    lock = load_held_protocol_lock(lock_path)
    _require(
        lock["immutable_bindings"]["held_physical_numeric_contract"]
        == held_contract_sha256(HELD_PHYSICAL_NUMERIC_CONTRACT),
        "physical numeric contract differs from the immutable lock",
    )
    _require(
        lock["immutable_bindings"]["upstream_runtime_bundle_tree"]
        == held_contract_sha256(UPSTREAM_RUNTIME_BUNDLE_CONTRACT),
        "upstream runtime bundle differs from the immutable lock",
    )
    for relative_path, binding_key in UPSTREAM_LOCK_BINDING_BY_PATH.items():
        _require(
            lock["immutable_bindings"][binding_key]
            == UPSTREAM_FILE_SHA256[relative_path],
            f"upstream source binding changed: {relative_path}",
        )
    validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    provenance = validate_upstream_runtime(
        upstream_repo,
        official_phystwin_repo,
        official_config,
    )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    prediction_data = root / "prediction_only_input.pkl"
    prediction_summary = root / "prediction_only_input.json"
    build_prediction_only_artifacts(
        frame_zero_manifest_path,
        lock_path,
        prediction_data,
        prediction_summary,
        case_name=case_name,
        role=role,
    )
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path, lock_path
    )
    upstream = Path(upstream_repo).resolve()
    python_path = Path(python).resolve()
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (str(upstream / "src"), str(Path(deform360_repo).resolve()))
            ),
            "PYNPUT_BACKEND": "dummy",
            "PYOPENGL_PLATFORM": "egl",
            "WANDB_MODE": "disabled",
        }
    )
    graph_path = root / "episode_graph.npz"
    simulator_data = root / "simulator_final_data.pkl"
    state_path = root / "state_artifact.npz"
    twin_summary = root / "twin_summary.json"
    runtimes: dict[str, float] = {}
    twin_command = [
        str(python_path),
        str(upstream / "scripts/remote/build_deform360_automatic_episode_twin.py"),
        "--repo",
        str(upstream),
        "--object-id",
        str(frame_zero["object_id"]),
        "--episode-id",
        str(frame_zero["episode_id"]),
        "--phase",
        "calibration" if role == "calibration" else "source",
        "--episode-final-data",
        str(prediction_data),
        "--episode-graph",
        str(graph_path),
        "--simulator-final-data",
        str(simulator_data),
        "--state-artifact",
        str(state_path),
        "--summary",
        str(twin_summary),
        "--prediction-only-input",
        "--canonical-node-count",
        str(CANONICAL_NODE_COUNT),
    ]
    if role == "calibration":
        twin_command.append("--source-admission-passed")
    runtimes["automatic_twin"] = _run_logged(
        twin_command,
        env=env,
        log_path=root / "logs/automatic_twin.log",
    )

    smoke_script = upstream / "scripts/remote/run_deform360_official_phystwin_smoke.py"
    split_path = (
        upstream / "configs/causal4d_public/deform360_independent_source_split_v1.json"
    )
    result_paths: dict[str, Path] = {}
    for label, scale in (("driven", 1.0), ("zero_action", 0.0)):
        rollout_dir = root / f"warp_{label}"
        command = [
            str(python_path),
            str(smoke_script),
            "--official-phystwin-repo",
            str(Path(official_phystwin_repo).resolve()),
            "--data",
            str(simulator_data),
            "--config",
            str(Path(official_config).resolve()),
            "--split-json",
            str(split_path),
            "--output-dir",
            str(rollout_dir),
            "--canonical-reusable-graph",
            str(graph_path),
            "--device",
            device,
            "--controller-radius-m",
            str(WARP_DYNAMICS["controller_radius_m"]),
            "--controller-max-neighbours",
            str(WARP_DYNAMICS["controller_max_neighbours"]),
            "--canonical-controller-patch-size",
            str(WARP_DYNAMICS["canonical_controller_patch_size"]),
            "--init-spring-y",
            str(WARP_DYNAMICS["init_spring_y"]),
            "--drag-damping",
            str(WARP_DYNAMICS["drag_damping"]),
            "--dashpot-damping",
            str(WARP_DYNAMICS["dashpot_damping"]),
            "--controller-displacement-scale",
            str(scale),
            "--support-dynamics",
            str(WARP_DYNAMICS["support_dynamics"]),
            "--report-edge-strain",
        ]
        runtimes[f"warp_{label}"] = _run_logged(
            command,
            env=env,
            log_path=root / f"logs/warp_{label}.log",
        )
        result_paths[label] = rollout_dir / "official_phystwin_smoke.json"

    prediction_archive = root / "prediction.npz"
    physical_manifest = root / "physical_prediction_manifest.json"
    seal_started = time.perf_counter()
    prediction_manifest = build_physical_prediction_archive(
        prediction_data,
        simulator_data,
        graph_path,
        state_path,
        twin_summary,
        result_paths["driven"],
        result_paths["zero_action"],
        prediction_archive,
        physical_manifest,
        frame_zero_manifest_path=frame_zero_manifest_path,
        lock_path=lock_path,
        case_name=case_name,
        role=role,
        runtime_provenance=provenance,
        stage_runtime_seconds=runtimes,
    )
    runtimes["prediction_seal"] = time.perf_counter() - seal_started
    physical_seal_path = root / "physical_prior_seal.json"
    physical_seal = create_physical_prior_seal(
        physical_seal_path,
        lock_path,
        frame_zero_manifest_path,
        {
            "prediction_only_input": prediction_data,
            "prediction_only_summary": prediction_summary,
            "physical_prediction_archive": prediction_archive,
            "physical_prediction_manifest": physical_manifest,
        },
        case_name=case_name,
        role=role,
    )
    return {
        "case_name": case_name,
        "role": role,
        "physical_prediction_manifest": prediction_manifest,
        "physical_prior_seal": physical_seal,
        "runtime_seconds": runtimes,
    }


__all__ = [
    "ACTION_RESPONSE",
    "ARTIFACT_KIND",
    "AUTONOMOUS_DRIFT_RESPONSE",
    "HELD_PHYSICAL_NUMERIC_CONTRACT",
    "LENGTH_SCALE_M",
    "OFFICIAL_PHYSTWIN_REVISION",
    "OFFICIAL_REAL_CONFIG_SHA256",
    "UPSTREAM_FILE_SHA256",
    "UPSTREAM_LOCK_BINDING_BY_PATH",
    "UPSTREAM_RUNTIME_BUNDLE_CONTRACT",
    "WARP_DYNAMICS",
    "build_physical_prediction_archive",
    "build_prediction_only_artifacts",
    "load_controller_trajectory",
    "run_held_physical_prior",
    "select_action_window",
    "sha256_array",
    "sha256_file",
    "validate_physical_prediction_manifest",
    "validate_upstream_runtime",
]
