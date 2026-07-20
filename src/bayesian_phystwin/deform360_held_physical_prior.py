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
CANONICAL_NODE_COUNT = 1024
MINIMUM_NODE_COUNT = 128

PHYSICAL_MODE_WARP_TWIN = "warp_twin"
PHYSICAL_MODE_PERSISTENCE_FALLBACK = "persistence_fallback"
PERSISTENCE_FALLBACK_REASON = "automatic_twin_source_admission_failed"
AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE = 2
AUTOMATIC_TWIN_PROTOCOL_ID = "deform360-dense-reusable-panel-v1"
AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256 = (
    "1a78b8d74679ebf65768cc5078b34d034a2fcac55f7e0c0a00e50e1967a1c9bd"
)
AUTOMATIC_TWIN_ADMISSION_THRESHOLDS = {
    "maximum_supported_distance_m": 0.02,
    "minimum_observed_target_fraction": 0.95,
    "minimum_effective_target_reliability": 0.70,
    "maximum_p99_relative_edge_strain": 0.50,
    "maximum_bridge_relative_edge_strain": 0.50,
    "maximum_contact_anchor_error_m": 0.015,
}

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
    "contract_id": "deform360-held-physical-prior-v2",
    "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
    "official_real_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
    "length_scale_m": LENGTH_SCALE_M,
    "action_response": ACTION_RESPONSE,
    "autonomous_drift_response": AUTONOMOUS_DRIFT_RESPONSE,
    "canonical_node_count": CANONICAL_NODE_COUNT,
    "minimum_node_count": MINIMUM_NODE_COUNT,
    "automatic_twin_admission_thresholds": AUTOMATIC_TWIN_ADMISSION_THRESHOLDS,
    "persistence_fallback": {
        "physical_mode": PHYSICAL_MODE_PERSISTENCE_FALLBACK,
        "reason": PERSISTENCE_FALLBACK_REASON,
        "required_automatic_twin_exit_code": AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE,
        "requires_valid_checksummed_inadmissible_twin": True,
        "warp_attempted": False,
        "prediction_equals_persistence": True,
        "driven_equals_persistence": True,
        "zero_action_equals_persistence": True,
        "action_support": "all_zero",
        "physical_admitted": False,
    },
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

PHYSICAL_ARCHIVE_ARRAYS = frozenset(
    {
        "prediction_m",
        "persistence_m",
        "driven_readout_m",
        "zero_action_readout_m",
        "action_support",
        "frame_zero_points_m",
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


class _LoggedCommandError(RuntimeError):
    """A subprocess failure with its exact exit status and elapsed runtime."""

    def __init__(
        self, message: str, *, returncode: int, elapsed_seconds: float
    ) -> None:
        super().__init__(message)
        self.returncode = int(returncode)
        self.elapsed_seconds = float(elapsed_seconds)


def _sorted_pip_freeze_sha256(stdout: bytes) -> str:
    lines = stdout.splitlines()
    canonical = b"\n".join(sorted(lines)) + b"\n"
    return hashlib.sha256(canonical).hexdigest()


def validate_python_runtime(
    python: str | Path,
    immutable_bindings: Mapping[str, Any],
) -> dict[str, str]:
    """Validate a venv interpreter without destroying its symlink semantics.

    A virtualenv's ``bin/python`` is commonly a symlink.  Executing the resolved
    target bypasses Python's virtualenv discovery, so the supplied absolute path
    is retained for every subprocess.  Only the executable-byte checksum follows
    the symlink to its resolved regular file.
    """

    supplied = Path(os.path.abspath(os.fspath(Path(python).expanduser())))
    _require(supplied.is_file(), "supplied Python interpreter is missing")
    resolved = supplied.resolve(strict=True)
    _require(resolved.is_file(), "resolved Python interpreter is not a file")
    executable_sha256 = sha256_file(resolved)
    _require(
        executable_sha256 == immutable_bindings.get("python_executable"),
        "Python executable bytes differ from the immutable lock",
    )
    completed = subprocess.run(
        [str(supplied), "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
    )
    freeze_sha256 = _sorted_pip_freeze_sha256(completed.stdout)
    _require(
        freeze_sha256 == immutable_bindings.get("python_pip_freeze_sorted"),
        "Python pip freeze differs from the immutable lock",
    )
    return {
        "supplied_python_path": str(supplied),
        "resolved_python_path": str(resolved),
        "python_executable_sha256": executable_sha256,
        "python_pip_freeze_sorted_sha256": freeze_sha256,
    }


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
        raise _LoggedCommandError(
            f"command failed with exit {completed.returncode}: {' '.join(command)}\n"
            + "\n".join(tail),
            returncode=completed.returncode,
            elapsed_seconds=elapsed,
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


def _upstream_result_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _metric_matches(observed: Any, expected: float) -> bool:
    return isinstance(observed, (int, float)) and np.isclose(
        float(observed), float(expected), rtol=1e-12, atol=1e-12
    )


def _validate_inadmissible_automatic_twin(
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    state_path: str | Path,
    twin_summary_path: str | Path,
    *,
    case_name: str,
    object_id: str,
    episode_id: int,
    role: str,
) -> dict[str, Any]:
    """Validate the one upstream exit-2 result eligible for persistence fallback."""

    prediction_path = Path(prediction_data_path).resolve()
    simulator_path = Path(simulator_data_path).resolve()
    graph_file = Path(graph_path).resolve()
    state_file = Path(state_path).resolve()
    summary_file = Path(twin_summary_path).resolve()
    for label, path in (
        ("prediction-only input", prediction_path),
        ("automatic-twin simulator data", simulator_path),
        ("automatic-twin graph", graph_file),
        ("automatic-twin state", state_file),
        ("automatic-twin summary", summary_file),
    ):
        _require(path.is_file() and not path.is_symlink(), f"missing {label}")

    prediction = _load_prediction_pickle(prediction_path)
    points = np.asarray(prediction.get("object_points"))
    colors = np.asarray(prediction.get("object_colors"))
    controllers = np.asarray(prediction.get("controller_points"))
    _require(
        points.ndim == 3
        and points.shape[0] == FRAME_COUNT
        and points.shape[2] == 3
        and colors.shape == points.shape
        and controllers.ndim == 3
        and controllers.shape[0] == FRAME_COUNT
        and controllers.shape[2] == 3,
        "inadmissible twin used an invalid prediction-only input",
    )
    _require(
        np.array_equal(points, np.repeat(points[:1], FRAME_COUNT, axis=0)),
        "inadmissible twin input contains future object geometry",
    )

    with np.load(graph_file, allow_pickle=False) as stored:
        expected_graph_arrays = {
            "vertices",
            "colors",
            "source_indices",
            "springs",
            "rest_lengths",
            "masses",
            "bridge_spring_count",
            "observed_node_count",
            "latent_node_count",
            "contact_anchor_indices",
            "contact_chain_spring_count",
            "reusable_graph_sha256",
        }
        _require(
            set(stored.files) == expected_graph_arrays,
            "inadmissible twin graph array set changed",
        )
        vertices = np.asarray(stored["vertices"], dtype=np.float64)
        springs = np.asarray(stored["springs"], dtype=np.int64)
        rest_lengths = np.asarray(stored["rest_lengths"], dtype=np.float64)
        bridge_count = int(np.asarray(stored["bridge_spring_count"]).item())
        observed_count = int(np.asarray(stored["observed_node_count"]).item())
        latent_count = int(np.asarray(stored["latent_node_count"]).item())
        anchors = np.asarray(stored["contact_anchor_indices"], dtype=np.int64)
        contact_chain_count = int(
            np.asarray(stored["contact_chain_spring_count"]).item()
        )
        graph_sha256 = str(np.asarray(stored["reusable_graph_sha256"]).item())
    effective_count = min(CANONICAL_NODE_COUNT, points.shape[1])
    _require(
        vertices.ndim == 2
        and vertices.shape[1] == 3
        and springs.ndim == 2
        and springs.shape[1] == 2
        and rest_lengths.shape == (len(springs),)
        and observed_count == effective_count
        and latent_count == len(vertices) - observed_count
        and 0 <= bridge_count <= len(springs)
        and 0 <= contact_chain_count <= bridge_count
        and np.all(np.isfinite(vertices))
        and np.all(np.isfinite(rest_lengths))
        and np.all(rest_lengths > 0.0)
        and np.all((springs >= 0) & (springs < len(vertices)))
        and np.all((anchors >= 0) & (anchors < len(vertices))),
        "inadmissible twin graph failed structural validation",
    )

    with np.load(state_file, allow_pickle=False) as stored:
        expected_state_arrays = {
            "vertices",
            "readout_weights",
            "readout_covariance_m2",
            "target_prior_reliability",
            "state_covariance_m2",
            "source_to_target_distance_m",
            "target_to_source_distance_m",
            "relative_edge_strain",
            "canonical_graph_sha256",
            "state_frame",
        }
        _require(
            set(stored.files) == expected_state_arrays,
            "inadmissible twin state array set changed",
        )
        state_vertices = np.asarray(stored["vertices"], dtype=np.float64)
        weights = np.asarray(stored["readout_weights"], dtype=np.float64)
        readout_covariance = np.asarray(
            stored["readout_covariance_m2"], dtype=np.float64
        )
        reliability = np.asarray(stored["target_prior_reliability"], dtype=np.float64)
        state_covariance = np.asarray(stored["state_covariance_m2"], dtype=np.float64)
        source_distance = np.asarray(
            stored["source_to_target_distance_m"], dtype=np.float64
        )
        target_distance = np.asarray(
            stored["target_to_source_distance_m"], dtype=np.float64
        )
        strain = np.asarray(stored["relative_edge_strain"], dtype=np.float64)
        state_graph_sha256 = str(np.asarray(stored["canonical_graph_sha256"]).item())
        state_frame = int(np.asarray(stored["state_frame"]).item())
    point_count = points.shape[1]
    _require(
        state_vertices.shape == vertices.shape
        and np.array_equal(state_vertices, vertices)
        and weights.shape == (point_count, len(vertices))
        and readout_covariance.shape == (point_count, 3, 3)
        and reliability.shape == (point_count,)
        and state_covariance.shape == (len(vertices), 3, 3)
        and source_distance.shape == (len(vertices),)
        and target_distance.shape == (point_count,)
        and strain.shape == (len(springs),)
        and state_graph_sha256 == graph_sha256
        and state_frame == 0
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6)
        and np.all(np.isfinite(reliability))
        and np.all((0.0 <= reliability) & (reliability <= 1.0))
        and np.all(np.isfinite(source_distance))
        and np.all(np.isfinite(target_distance))
        and np.all(np.isfinite(strain)),
        "inadmissible twin state failed structural validation",
    )

    twin = _load_json(summary_file)
    expected_summary_keys = {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "protocol_config_sha256",
        "object_id",
        "episode_id",
        "phase",
        "graph_mode",
        "capacity_diagnostic",
        "graph",
        "state_metrics",
        "input_sha256",
        "output_sha256",
        "information_boundary",
        "prediction_input_validation",
        "sota_input_validation",
        "passed",
        "claim_boundary",
        "result_sha256",
    }
    _require(
        set(twin) == expected_summary_keys,
        "inadmissible automatic-twin summary schema changed",
    )
    _require(
        twin.get("schema_version") == 1
        and twin.get("artifact_kind") == "Deform360AutomaticEpisodeTwin"
        and twin.get("protocol_id") == AUTOMATIC_TWIN_PROTOCOL_ID
        and twin.get("protocol_config_sha256") == AUTOMATIC_TWIN_PROTOCOL_CONFIG_SHA256
        and twin.get("object_id") == object_id
        and int(twin.get("episode_id", -1)) == episode_id
        and twin.get("phase") == ("calibration" if role == "calibration" else "source")
        and twin.get("graph_mode") == "episode_specific_frame_zero_control"
        and twin.get("passed") is False
        and twin.get("sota_input_validation") is None
        and twin.get("result_sha256") == _upstream_result_sha256(twin),
        "automatic twin is not a checksummed explicit inadmissible result",
    )
    capacity = twin.get("capacity_diagnostic", {})
    _require(
        capacity
        == {
            "configured_canonical_node_count": 192,
            "requested_canonical_node_count": CANONICAL_NODE_COUNT,
            "effective_canonical_node_count": effective_count,
            "source_only_override": effective_count != 192,
            "capacity_is_a_maximum": True,
        },
        "inadmissible twin used another graph capacity",
    )
    graph = twin.get("graph", {})
    _require(
        graph
        == {
            "schema_version": 1,
            "artifact_kind": "Deform360CanonicalReusableGraph",
            "path": str(graph_file),
            "reusable_graph_sha256": graph_sha256,
            "node_count": len(vertices),
            "object_spring_count": len(springs),
            "bridge_spring_count": bridge_count,
            "observed_node_count": observed_count,
            "latent_node_count": latent_count,
            "contact_anchor_count": len(anchors),
            "contact_chain_spring_count": contact_chain_count,
        },
        "inadmissible twin summary binds another graph",
    )
    _require(
        twin.get("input_sha256")
        == {
            "episode_final_data": sha256_file(prediction_path),
            "development_observations": None,
            "contact_conditioned_action": None,
        }
        and twin.get("output_sha256")
        == {
            "episode_graph": sha256_file(graph_file),
            "simulator_final_data": sha256_file(simulator_path),
            "state_artifact": sha256_file(state_file),
        },
        "inadmissible twin input/output checksums changed",
    )
    _require(
        twin.get("prediction_input_validation")
        == {
            "frame_count": FRAME_COUNT,
            "point_count": point_count,
            "controller_point_count": controllers.shape[1],
            "frame_zero_points_sha256": sha256_array(points[0]),
            "controller_trajectory_sha256": sha256_array(controllers),
        },
        "inadmissible twin prediction-input validation changed",
    )
    _require(
        twin.get("information_boundary")
        == {
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
        "inadmissible twin crossed the prediction-only boundary",
    )

    metrics = twin.get("state_metrics", {})
    expected_metric_keys = {
        "passed",
        "finite",
        "symmetric_chamfer_m",
        "source_to_target_p95_m",
        "target_to_source_p95_m",
        "observed_target_fraction",
        "canonical_supported_fraction",
        "effective_target_reliability",
        "initial_readout_rmse_m",
        "p99_absolute_relative_edge_strain",
        "maximum_absolute_relative_edge_strain",
        "maximum_bridge_absolute_relative_edge_strain",
        "maximum_contact_anchor_error_m",
    }
    _require(
        isinstance(metrics, Mapping)
        and set(metrics) == expected_metric_keys
        and metrics.get("passed") is False,
        "automatic twin lacks explicit failed state metrics",
    )
    finite = bool(
        np.all(np.isfinite(state_vertices))
        and np.all(np.isfinite(readout_covariance))
        and np.all(np.isfinite(state_covariance))
    )
    readout = weights @ state_vertices
    bridge_strain = strain[-bridge_count:] if bridge_count else np.empty(0)
    expected_metrics = {
        "symmetric_chamfer_m": 0.5
        * (float(np.mean(source_distance)) + float(np.mean(target_distance))),
        "source_to_target_p95_m": float(np.quantile(source_distance, 0.95)),
        "target_to_source_p95_m": float(np.quantile(target_distance, 0.95)),
        "observed_target_fraction": float(
            np.mean(
                target_distance
                <= AUTOMATIC_TWIN_ADMISSION_THRESHOLDS["maximum_supported_distance_m"]
            )
        ),
        "canonical_supported_fraction": float(
            np.mean(
                source_distance
                <= AUTOMATIC_TWIN_ADMISSION_THRESHOLDS["maximum_supported_distance_m"]
            )
        ),
        "effective_target_reliability": float(np.mean(reliability)),
        "initial_readout_rmse_m": float(np.sqrt(np.mean((readout - points[0]) ** 2))),
        "p99_absolute_relative_edge_strain": float(np.quantile(strain, 0.99)),
        "maximum_absolute_relative_edge_strain": float(np.max(strain)),
        "maximum_bridge_absolute_relative_edge_strain": float(
            np.max(bridge_strain, initial=0.0)
        ),
    }
    _require(metrics.get("finite") is finite, "automatic twin finite metric changed")
    for name, expected in expected_metrics.items():
        _require(
            _metric_matches(metrics.get(name), expected),
            f"automatic twin metric changed: {name}",
        )
    contact_error = metrics.get("maximum_contact_anchor_error_m")
    _require(
        isinstance(contact_error, (int, float))
        and np.isfinite(float(contact_error))
        and float(contact_error) >= 0.0,
        "automatic twin contact metric is invalid",
    )
    threshold = AUTOMATIC_TWIN_ADMISSION_THRESHOLDS
    recomputed_pass = bool(
        finite
        and float(metrics["observed_target_fraction"])
        >= threshold["minimum_observed_target_fraction"]
        and float(metrics["effective_target_reliability"])
        >= threshold["minimum_effective_target_reliability"]
        and float(metrics["p99_absolute_relative_edge_strain"])
        <= threshold["maximum_p99_relative_edge_strain"]
        and float(metrics["maximum_bridge_absolute_relative_edge_strain"])
        <= threshold["maximum_bridge_relative_edge_strain"]
        and float(contact_error) <= threshold["maximum_contact_anchor_error_m"]
    )
    _require(not recomputed_pass, "automatic twin metrics actually pass admission")
    _require(
        isinstance(twin.get("claim_boundary"), str)
        and "frame-zero episode-twin control" in twin["claim_boundary"],
        "automatic twin claim boundary changed",
    )
    return twin


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
        "physical_mode": PHYSICAL_MODE_WARP_TWIN,
        "physical_admitted": True,
        "fallback_diagnostics": None,
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


def build_persistence_fallback_archive(
    prediction_data_path: str | Path,
    simulator_data_path: str | Path,
    graph_path: str | Path,
    state_path: str | Path,
    twin_summary_path: str | Path,
    automatic_twin_log_path: str | Path,
    archive_path: str | Path,
    manifest_path: str | Path,
    *,
    frame_zero_manifest_path: str | Path,
    lock_path: str | Path,
    case_name: str,
    role: str,
    automatic_twin_exit_code: int,
    runtime_provenance: Mapping[str, Any],
    stage_runtime_seconds: Mapping[str, float],
) -> dict[str, Any]:
    """Seal persistence only after a genuine automatic-twin admission rejection."""

    _require(
        automatic_twin_exit_code == AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE,
        "persistence fallback requires the frozen inadmissible exit code",
    )
    frame_zero = validate_frame_zero_bundle_manifest(
        frame_zero_manifest_path,
        lock_path,
        expected_case_name=case_name,
        expected_role=role,
    )
    twin = _validate_inadmissible_automatic_twin(
        prediction_data_path,
        simulator_data_path,
        graph_path,
        state_path,
        twin_summary_path,
        case_name=case_name,
        object_id=str(frame_zero["object_id"]),
        episode_id=int(frame_zero["episode_id"]),
        role=role,
    )
    prediction_data = _load_prediction_pickle(prediction_data_path)
    points = np.asarray(prediction_data["object_points"], dtype=np.float32)
    _require(
        points.shape[0] == FRAME_COUNT
        and np.array_equal(points, np.repeat(points[:1], FRAME_COUNT, axis=0)),
        "persistence fallback input contains changing object geometry",
    )
    persistence = np.repeat(points[:1], FRAME_COUNT, axis=0).astype(
        np.float32, copy=False
    )
    zeros = np.zeros(points.shape[1], dtype=np.float32)
    arrays = {
        "prediction_m": persistence.copy(),
        "persistence_m": persistence.copy(),
        "driven_readout_m": persistence.copy(),
        "zero_action_readout_m": persistence.copy(),
        "action_support": zeros,
        "frame_zero_points_m": points[0].copy(),
    }
    archive = Path(archive_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **arrays)
    with np.load(graph_path, allow_pickle=False) as graph:
        observed_nodes = int(np.asarray(graph["observed_node_count"]).item())
        total_nodes = len(np.asarray(graph["vertices"]))
    input_paths = {
        "prediction_only_input": prediction_data_path,
        "simulator_final_data": simulator_data_path,
        "episode_graph": graph_path,
        "state_artifact": state_path,
        "twin_summary": twin_summary_path,
        "automatic_twin_log": automatic_twin_log_path,
    }
    summary_record = _bound_file(twin_summary_path)
    log_record = _bound_file(automatic_twin_log_path)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "case_name": case_name,
        "object_id": frame_zero["object_id"],
        "episode_id": int(frame_zero["episode_id"]),
        "role": role,
        "physical_mode": PHYSICAL_MODE_PERSISTENCE_FALLBACK,
        "physical_admitted": False,
        "fallback_diagnostics": {
            "reason": PERSISTENCE_FALLBACK_REASON,
            "automatic_twin_exit_code": automatic_twin_exit_code,
            "automatic_twin_result_sha256": twin["result_sha256"],
            "automatic_twin_summary_sha256": summary_record["sha256"],
            "automatic_twin_log_sha256": log_record["sha256"],
            "automatic_twin_state_metrics": dict(twin["state_metrics"]),
            "warp_attempted": False,
        },
        "frozen_predictor": {
            "official_phystwin_revision": OFFICIAL_PHYSTWIN_REVISION,
            "official_real_config_sha256": OFFICIAL_REAL_CONFIG_SHA256,
            "length_scale_m": LENGTH_SCALE_M,
            "action_response": ACTION_RESPONSE,
            "autonomous_drift_response": AUTONOMOUS_DRIFT_RESPONSE,
            "frame_count": FRAME_COUNT,
            "observed_graph_node_count": observed_nodes,
            "total_graph_node_count": total_nodes,
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
    mode = manifest.get("physical_mode")
    _require(
        mode in {PHYSICAL_MODE_WARP_TWIN, PHYSICAL_MODE_PERSISTENCE_FALLBACK},
        "physical prediction mode changed",
    )
    admitted = manifest.get("physical_admitted")
    _require(
        admitted is (mode == PHYSICAL_MODE_WARP_TWIN),
        "physical admission flag disagrees with prediction mode",
    )
    fallback = manifest.get("fallback_diagnostics")
    if mode == PHYSICAL_MODE_WARP_TWIN:
        _require(fallback is None, "Warp-twin prediction carries fallback diagnostics")
    else:
        _require(
            isinstance(fallback, Mapping)
            and set(fallback)
            == {
                "reason",
                "automatic_twin_exit_code",
                "automatic_twin_result_sha256",
                "automatic_twin_summary_sha256",
                "automatic_twin_log_sha256",
                "automatic_twin_state_metrics",
                "warp_attempted",
            }
            and fallback.get("reason") == PERSISTENCE_FALLBACK_REASON
            and fallback.get("automatic_twin_exit_code")
            == AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE
            and fallback.get("warp_attempted") is False,
            "persistence fallback diagnostics changed",
        )
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
    _require(
        isinstance(frozen.get("point_count"), int)
        and int(frozen.get("point_count")) >= MINIMUM_NODE_COUNT
        and MINIMUM_NODE_COUNT
        <= int(frozen.get("observed_graph_node_count", -1))
        <= CANONICAL_NODE_COUNT
        and int(frozen.get("total_graph_node_count", -1))
        >= int(frozen.get("observed_graph_node_count", -1)),
        "physical graph capacity changed",
    )
    archive_record = manifest.get("physical_prediction_archive", {})
    archive_path = _validate_bound_file(
        archive_record,
        label="physical prediction archive",
        allow_metadata=True,
    )
    inputs = manifest.get("input_files", {})
    expected_input_roles = (
        {
            "prediction_only_input",
            "simulator_final_data",
            "episode_graph",
            "state_artifact",
            "twin_summary",
            "driven_result",
            "zero_action_result",
            "driven_trajectory",
            "zero_action_trajectory",
        }
        if mode == PHYSICAL_MODE_WARP_TWIN
        else {
            "prediction_only_input",
            "simulator_final_data",
            "episode_graph",
            "state_artifact",
            "twin_summary",
            "automatic_twin_log",
        }
    )
    _require(
        isinstance(inputs, Mapping) and set(inputs) == expected_input_roles,
        "prediction input roles changed",
    )
    for label, record in inputs.items():
        _validate_bound_file(record, label=str(label))
    if mode == PHYSICAL_MODE_PERSISTENCE_FALLBACK:
        twin = _validate_inadmissible_automatic_twin(
            inputs["prediction_only_input"]["path"],
            inputs["simulator_final_data"]["path"],
            inputs["episode_graph"]["path"],
            inputs["state_artifact"]["path"],
            inputs["twin_summary"]["path"],
            case_name=str(manifest.get("case_name")),
            object_id=str(manifest.get("object_id")),
            episode_id=int(manifest.get("episode_id", -1)),
            role=str(manifest.get("role")),
        )
        _require(
            fallback.get("automatic_twin_result_sha256") == twin["result_sha256"]
            and fallback.get("automatic_twin_summary_sha256")
            == inputs["twin_summary"]["sha256"]
            and fallback.get("automatic_twin_log_sha256")
            == inputs["automatic_twin_log"]["sha256"]
            and fallback.get("automatic_twin_state_metrics") == twin["state_metrics"],
            "persistence fallback diagnostics do not bind the failed twin",
        )
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
        _require(
            isinstance(expected, Mapping) and set(expected) == PHYSICAL_ARCHIVE_ARRAYS,
            "archive array checksums are missing or changed",
        )
        with np.load(archive_path, allow_pickle=False) as stored:
            _require(
                set(stored.files) == PHYSICAL_ARCHIVE_ARRAYS,
                "prediction archive array set changed",
            )
            for name in stored.files:
                _require(
                    sha256_array(stored[name]) == expected[name],
                    f"{name} checksum changed",
                )
            if mode == PHYSICAL_MODE_PERSISTENCE_FALLBACK:
                persistence = np.asarray(stored["persistence_m"])
                frame_zero = np.asarray(stored["frame_zero_points_m"])
                _require(
                    persistence.dtype == np.dtype(np.float32)
                    and persistence.shape
                    == (FRAME_COUNT, int(frozen["point_count"]), 3)
                    and frame_zero.dtype == np.dtype(np.float32)
                    and frame_zero.shape == (int(frozen["point_count"]), 3)
                    and np.array_equal(
                        persistence,
                        np.repeat(frame_zero[None], FRAME_COUNT, axis=0),
                    )
                    and np.array_equal(stored["prediction_m"], persistence)
                    and np.array_equal(stored["driven_readout_m"], persistence)
                    and np.array_equal(stored["zero_action_readout_m"], persistence)
                    and stored["action_support"].dtype == np.dtype(np.float32)
                    and stored["action_support"].shape == (int(frozen["point_count"]),)
                    and np.count_nonzero(stored["action_support"]) == 0,
                    "persistence fallback arrays changed",
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
    python_runtime = validate_python_runtime(python, lock["immutable_bindings"])
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
    provenance["python_runtime"] = python_runtime
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
    # Do not resolve this path: executing a venv symlink through its base
    # interpreter silently disables the virtualenv's prefix/site-packages.
    python_path = Path(python_runtime["supplied_python_path"])
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
    automatic_twin_log = root / "logs/automatic_twin.log"
    try:
        runtimes["automatic_twin"] = _run_logged(
            twin_command,
            env=env,
            log_path=automatic_twin_log,
        )
    except _LoggedCommandError as error:
        if error.returncode != AUTOMATIC_TWIN_EXIT_CODE_INADMISSIBLE:
            raise
        runtimes["automatic_twin"] = error.elapsed_seconds
        prediction_archive = root / "prediction.npz"
        physical_manifest = root / "physical_prediction_manifest.json"
        seal_started = time.perf_counter()
        prediction_manifest = build_persistence_fallback_archive(
            prediction_data,
            simulator_data,
            graph_path,
            state_path,
            twin_summary,
            automatic_twin_log,
            prediction_archive,
            physical_manifest,
            frame_zero_manifest_path=frame_zero_manifest_path,
            lock_path=lock_path,
            case_name=case_name,
            role=role,
            automatic_twin_exit_code=error.returncode,
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
    "CANONICAL_NODE_COUNT",
    "HELD_PHYSICAL_NUMERIC_CONTRACT",
    "LENGTH_SCALE_M",
    "OFFICIAL_PHYSTWIN_REVISION",
    "OFFICIAL_REAL_CONFIG_SHA256",
    "PHYSICAL_MODE_PERSISTENCE_FALLBACK",
    "PHYSICAL_MODE_WARP_TWIN",
    "UPSTREAM_FILE_SHA256",
    "UPSTREAM_LOCK_BINDING_BY_PATH",
    "UPSTREAM_RUNTIME_BUNDLE_CONTRACT",
    "WARP_DYNAMICS",
    "build_persistence_fallback_archive",
    "build_physical_prediction_archive",
    "build_prediction_only_artifacts",
    "load_controller_trajectory",
    "run_held_physical_prior",
    "select_action_window",
    "sha256_array",
    "sha256_file",
    "validate_physical_prediction_manifest",
    "validate_python_runtime",
    "validate_upstream_runtime",
]
