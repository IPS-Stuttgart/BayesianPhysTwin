"""Future-blind source utilities for a public PGND dynamics backbone."""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from .phystwin_official_evaluation import (
    evaluate_official_phystwin_interval,
    official_phystwin_metrics_by_frame,
)

PGND_UPSTREAM_COMMIT = "ae050d1342faa0bceb2a10f4b0ab2e11682351cb"
PGND_SLOTH_CHECKPOINT_SHA256 = (
    "1ce7f86a40058c2680784ac40f633a67e00e9ce8af8a6111acc3362d71d3b052"
)
PGND_SLOTH_CONFIG_SHA256 = (
    "01e33f8152b25ac80998f0ddaadd182f88ca96b18432bc288edc4723a943a915"
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def verify_clean_git_checkout(
    checkout: str | Path,
    *,
    expected_commit: str | None = None,
) -> dict[str, object]:
    """Return immutable checkout provenance or reject source-tree drift."""

    checkout_path = Path(checkout).resolve()
    if not checkout_path.is_dir():
        raise FileNotFoundError(f"Git checkout does not exist: {checkout_path}")
    commit = subprocess.run(
        ["git", "-C", str(checkout_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        [
            "git",
            "-C",
            str(checkout_path),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(
            f"Git commit mismatch: expected {expected_commit}, got {commit}"
        )
    if status:
        raise ValueError(f"Git checkout is dirty: {checkout_path}")
    return {
        "checkout": str(checkout_path),
        "commit": commit,
        "clean": True,
    }


def verify_pgnd_assets(
    checkout: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    *,
    expected_commit: str = PGND_UPSTREAM_COMMIT,
    expected_checkpoint_sha256: str = PGND_SLOTH_CHECKPOINT_SHA256,
    expected_config_sha256: str = PGND_SLOTH_CONFIG_SHA256,
) -> dict[str, object]:
    """Reject upstream, checkpoint, or configuration drift."""

    checkout_path = Path(checkout).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    config_path = Path(config).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PGND checkpoint does not exist: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"PGND config does not exist: {config_path}")
    checkout_state = verify_clean_git_checkout(
        checkout_path,
        expected_commit=expected_commit,
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    config_sha256 = sha256_file(config_path)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "PGND checkpoint mismatch: expected "
            f"{expected_checkpoint_sha256}, got {checkpoint_sha256}"
        )
    if config_sha256 != expected_config_sha256:
        raise ValueError(
            f"PGND config mismatch: expected {expected_config_sha256}, "
            f"got {config_sha256}"
        )
    return {
        "checkout": str(checkout_path),
        "commit": str(checkout_state["commit"]),
        "clean": True,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "config": str(config_path),
        "config_sha256": config_sha256,
    }


@dataclass(frozen=True)
class PGNDFrameSelection:
    """Prefix history and future model frames at PGND's 10 Hz cadence."""

    initialization_frame: int
    history_frames: tuple[int, ...]
    prediction_frames: tuple[int, ...]
    model_frame_stride: int


def select_pgnd_frames(
    *,
    train_end_exclusive: int,
    frame_count: int,
    history_length: int = 2,
    model_frame_stride: int = 3,
) -> PGNDFrameSelection:
    """Select a causal cadence whose final model step stays inside the episode."""

    if not 1 < train_end_exclusive < frame_count:
        raise ValueError("expected 1 < train_end_exclusive < frame_count")
    if history_length < 1:
        raise ValueError("history_length must be positive")
    if model_frame_stride < 1:
        raise ValueError("model_frame_stride must be positive")
    initialization_frame = (
        (train_end_exclusive - 1) // model_frame_stride
    ) * model_frame_stride
    history_frames = tuple(
        initialization_frame - offset * model_frame_stride
        for offset in range(history_length, 0, -1)
    )
    if history_frames[0] < 0:
        raise ValueError("the observed prefix is too short for PGND history")
    prediction_frames = tuple(
        range(
            initialization_frame + model_frame_stride,
            frame_count,
            model_frame_stride,
        )
    )
    if not prediction_frames:
        raise ValueError("the episode has no future PGND model step")
    if prediction_frames[-1] != frame_count - 1:
        raise ValueError(
            "the frozen cadence must land exactly on the final evaluation frame"
        )
    return PGNDFrameSelection(
        initialization_frame=initialization_frame,
        history_frames=history_frames,
        prediction_frames=prediction_frames,
        model_frame_stride=model_frame_stride,
    )


@dataclass(frozen=True)
class PGNDMetricTransform:
    """Prefix-only metric transform used by PGND's public plush demo."""

    rotation_model_from_world: np.ndarray
    translation_model: np.ndarray

    @classmethod
    def fit(
        cls,
        current_world_m: np.ndarray,
        *,
        grid_spacing_m: float = 0.02,
        clip_bound_cells: float = 1.5,
    ) -> PGNDMetricTransform:
        points = np.asarray(current_world_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("current_world_m must have shape (N, 3), N > 0")
        if not np.all(np.isfinite(points)):
            raise ValueError("current_world_m must be finite")
        if grid_spacing_m <= 0.0:
            raise ValueError("grid_spacing_m must be positive")
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        rotated = points @ rotation.T
        translation = np.array(
            [
                0.5 - float(np.mean(rotated[:, 0])),
                grid_spacing_m * (clip_bound_cells + 0.5)
                + 1e-5
                - float(np.min(rotated[:, 1])),
                0.5 - float(np.mean(rotated[:, 2])),
            ]
        )
        return cls(
            rotation_model_from_world=rotation,
            translation_model=translation,
        )

    def positions_to_model(self, points_world_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_m, dtype=float)
        return points @ self.rotation_model_from_world.T + self.translation_model

    def positions_to_world(self, points_model_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_model_m, dtype=float)
        return (points - self.translation_model) @ self.rotation_model_from_world

    def velocities_to_model(self, velocities_world_mps: np.ndarray) -> np.ndarray:
        velocities = np.asarray(velocities_world_mps, dtype=float)
        return velocities @ self.rotation_model_from_world.T


def physically_supported_contact_trajectory(
    controller_points_world_m: np.ndarray,
    physical_world_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose known-action points nearest the unchanged physical prior."""

    controller = np.asarray(controller_points_world_m, dtype=float)
    physical = np.asarray(physical_world_m, dtype=float)
    if controller.ndim != 3 or controller.shape[2] != 3:
        raise ValueError("controller_points_world_m must have shape (T, C, 3)")
    if physical.ndim != 3 or physical.shape[2] != 3:
        raise ValueError("physical_world_m must have shape (T, N, 3)")
    if controller.shape[0] != physical.shape[0]:
        raise ValueError("controller and physical trajectories must share T")
    if controller.shape[1] == 0 or physical.shape[1] == 0:
        raise ValueError("controller and physical trajectories must be nonempty")
    if not np.all(np.isfinite(controller)) or not np.all(np.isfinite(physical)):
        raise ValueError("controller and physical trajectories must be finite")

    indices = np.empty(controller.shape[0], dtype=np.int64)
    selected = np.empty((controller.shape[0], 3), dtype=float)
    for frame in range(controller.shape[0]):
        delta = controller[frame, :, None] - physical[frame, None]
        squared = np.sum(np.square(delta), axis=2)
        controller_index = int(np.argmin(np.min(squared, axis=1)))
        indices[frame] = controller_index
        selected[frame] = controller[frame, controller_index]
    return selected, indices


def build_pgnd_gripper_actions(
    positions_model_m: np.ndarray,
    *,
    dt_s: float,
    radius_m: float,
) -> np.ndarray:
    """Build the public PGND 15-value spherical-gripper action rows."""

    positions = np.asarray(positions_model_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("positions_model_m must have shape (T, 3), T >= 2")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_model_m must be finite")
    if dt_s <= 0.0 or radius_m <= 0.0:
        raise ValueError("dt_s and radius_m must be positive")
    actions: np.ndarray = np.zeros((len(positions), 1, 15), dtype=np.float32)
    actions[:, 0, :3] = positions
    actions[:-1, 0, 3:6] = np.diff(positions, axis=0) / dt_s
    actions[-1, 0, 3:6] = actions[-2, 0, 3:6]
    actions[:, 0, 6] = 1.0
    actions[:, 0, 13] = radius_m
    actions[:, 0, 14] = 0.0
    return actions


def interpolate_model_steps(
    *,
    physical_prefix: np.ndarray,
    model_prediction_frames: tuple[int, ...],
    model_predictions: np.ndarray,
    initialization_frame: int,
    frame_count: int,
) -> np.ndarray:
    """Build a full-rate trajectory without reading future observations."""

    prefix = np.asarray(physical_prefix, dtype=float)
    predictions = np.asarray(model_predictions, dtype=float)
    if prefix.ndim != 3 or prefix.shape[2] != 3:
        raise ValueError("physical_prefix must have shape (T, N, 3)")
    if predictions.shape != (
        len(model_prediction_frames),
        prefix.shape[1],
        3,
    ):
        raise ValueError("model_predictions do not match the frozen frames")
    if prefix.shape[0] <= initialization_frame:
        raise ValueError("physical_prefix omits the initialization frame")
    if frame_count <= initialization_frame:
        raise ValueError("frame_count must exceed initialization_frame")

    trajectory = np.empty((frame_count, prefix.shape[1], 3), dtype=float)
    trajectory[: initialization_frame + 1] = prefix[: initialization_frame + 1]
    anchor_frames = (initialization_frame,) + model_prediction_frames
    anchor_values = np.concatenate(
        [prefix[initialization_frame : initialization_frame + 1], predictions],
        axis=0,
    )
    for segment in range(len(anchor_frames) - 1):
        start = anchor_frames[segment]
        stop = anchor_frames[segment + 1]
        for frame in range(start + 1, stop + 1):
            fraction = (frame - start) / (stop - start)
            trajectory[frame] = (1.0 - fraction) * anchor_values[
                segment
            ] + fraction * anchor_values[segment + 1]
    if anchor_frames[-1] != frame_count - 1:
        raise ValueError("model prediction frames do not cover the episode")
    return trajectory


def prepare_pgnd_source_input(
    *,
    final_data_path: str | Path,
    physical_trajectory_path: str | Path,
    split_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Extract only known actions and a model prior into a prediction carrier."""

    final_data = _load_pickle(final_data_path)
    physical = np.asarray(_load_pickle(physical_trajectory_path), dtype=np.float32)
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    controller = np.asarray(final_data["controller_points"], dtype=np.float32)
    if physical.ndim != 3 or physical.shape[2] != 3:
        raise ValueError("physical trajectory must have shape (T, N, 3)")
    if controller.ndim != 3 or controller.shape[2] != 3:
        raise ValueError("controller trajectory must have shape (T, C, 3)")
    if physical.shape[0] != controller.shape[0]:
        raise ValueError("physical and controller trajectories must share T")
    object_points = np.asarray(final_data["object_points"])
    surface_points = np.asarray(final_data["surface_points"])
    if (
        object_points.ndim != 3
        or object_points.shape[0] != physical.shape[0]
        or object_points.shape[2] != 3
    ):
        raise ValueError("object_points must have shape (T, M, 3)")
    if surface_points.ndim != 2 or surface_points.shape[1] != 3:
        raise ValueError("surface_points must have shape (S, 3)")
    num_surface_points = object_points.shape[1] + surface_points.shape[0]
    if not 0 < num_surface_points <= physical.shape[1]:
        raise ValueError("frame-zero surface layout disagrees with physical nodes")
    train_end = int(split["train"][1])
    test_end = int(split["test"][1])
    if test_end != physical.shape[0]:
        raise ValueError("split test end disagrees with the source trajectories")
    selection = select_pgnd_frames(
        train_end_exclusive=train_end,
        frame_count=test_end,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        physical_trajectory=physical,
        controller_points=controller,
        train_end_exclusive=np.asarray(train_end, dtype=np.int64),
        test_end_exclusive=np.asarray(test_end, dtype=np.int64),
        initialization_frame=np.asarray(selection.initialization_frame, dtype=np.int64),
        history_frames=np.asarray(selection.history_frames, dtype=np.int64),
        prediction_frames=np.asarray(selection.prediction_frames, dtype=np.int64),
        num_surface_points=np.asarray(num_surface_points, dtype=np.int64),
    )
    return {
        "schema_version": 1,
        "claim_boundary": (
            "Prediction carrier contains only an unchanged physical trajectory, "
            "known controller actions, and the released split. It contains no "
            "future object observations or manual tracks."
        ),
        "output": {
            "path": str(output.resolve()),
            "sha256": sha256_file(output),
        },
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": sha256_file(final_data_path),
                "fields_read": [
                    "controller_points",
                    "object_points shape only",
                    "surface_points shape only",
                ],
            },
            "physical_trajectory": {
                "path": str(Path(physical_trajectory_path).resolve()),
                "sha256": sha256_file(physical_trajectory_path),
            },
            "split": {
                "path": str(Path(split_path).resolve()),
                "sha256": sha256_file(split_path),
            },
        },
        "selection": {
            "initialization_frame": selection.initialization_frame,
            "history_frames": list(selection.history_frames),
            "prediction_frames": list(selection.prediction_frames),
            "train_end_exclusive": train_end,
            "test_end_exclusive": test_end,
        },
    }


def _horizon_intervals(start: int, end: int) -> dict[str, tuple[int, int]]:
    edges = np.linspace(start, end, 4, dtype=int)
    return {
        "early": (int(edges[0]), int(edges[1])),
        "middle": (int(edges[1]), int(edges[2])),
        "late": (int(edges[2]), int(edges[3])),
    }


def evaluate_pgnd_source_prediction(
    *,
    candidate_trajectory: np.ndarray,
    equal_support_physical: np.ndarray,
    equal_support_surface_count: int,
    full_physical: np.ndarray,
    persistence_trajectory: np.ndarray,
    final_data: dict[str, object],
    gt_track_3d: np.ndarray,
    train_end_exclusive: int,
    test_end_exclusive: int,
    required_relative_improvement: float = 0.02,
) -> dict[str, object]:
    """Evaluate one opened source case under the official PhysTwin metrics."""

    candidate = np.asarray(candidate_trajectory, dtype=float)
    equal_physical = np.asarray(equal_support_physical, dtype=float)
    full = np.asarray(full_physical, dtype=float)
    persistence = np.asarray(persistence_trajectory, dtype=float)
    object_points = np.asarray(final_data["object_points"], dtype=float)
    visibility = np.asarray(final_data["object_visibilities"], dtype=bool)
    tracks = np.asarray(gt_track_3d, dtype=float)
    if not (
        candidate.shape == equal_physical.shape == persistence.shape
        and candidate.shape[0] == full.shape[0]
        and candidate.shape[2] == full.shape[2] == 3
    ):
        raise ValueError("candidate and comparator trajectories are incompatible")
    if not all(
        np.all(np.isfinite(values))
        for values in (candidate, equal_physical, full, persistence)
    ):
        raise ValueError("prediction trajectories must be finite")
    if not 0 < equal_support_surface_count <= candidate.shape[1]:
        raise ValueError("equal_support_surface_count is invalid")

    num_surface_points_full = min(
        int(object_points.shape[1])
        + int(np.asarray(final_data["surface_points"]).shape[0]),
        full.shape[1],
    )
    trajectories = {
        "pgnd_candidate": (candidate, equal_support_surface_count),
        "equal_support_physical": (
            equal_physical,
            equal_support_surface_count,
        ),
        "full_physical": (full, num_surface_points_full),
        "endpoint_persistence": (
            persistence,
            equal_support_surface_count,
        ),
    }
    metrics: dict[str, dict[str, object]] = {}
    by_frame: dict[str, dict[str, np.ndarray]] = {}
    for name, (trajectory, surface_count) in trajectories.items():
        metrics[name] = evaluate_official_phystwin_interval(
            trajectory,
            object_points,
            visibility,
            tracks,
            num_surface_points=surface_count,
            start_frame=train_end_exclusive,
            end_frame=test_end_exclusive,
        )
        by_frame[name] = official_phystwin_metrics_by_frame(
            trajectory,
            object_points,
            visibility,
            tracks,
            num_surface_points=surface_count,
            start_frame=train_end_exclusive,
            end_frame=test_end_exclusive,
        )

    horizons: dict[str, object] = {}
    for horizon, (start, end) in _horizon_intervals(
        train_end_exclusive, test_end_exclusive
    ).items():
        offset_start = start - train_end_exclusive
        offset_end = end - train_end_exclusive
        horizons[horizon] = {
            name: {
                metric_name: float(
                    np.mean(values[metric_name][offset_start:offset_end])
                )
                for metric_name in ("chamfer_distance_m", "track_error_m")
            }
            for name, values in by_frame.items()
        }

    candidate_metrics = metrics["pgnd_candidate"]
    full_metrics = metrics["full_physical"]
    ratios = {
        name: cast(float, candidate_metrics[name]) / cast(float, full_metrics[name])
        for name in ("chamfer_distance_m", "track_error_m")
    }
    passed = all(
        ratio <= 1.0 - required_relative_improvement for ratio in ratios.values()
    )
    return {
        "metric_contract": {
            "chamfer_distance_m": (
                "Released PhysTwin one-way visible-observation to predicted-surface "
                "L1 nearest-neighbor distance."
            ),
            "track_error_m": (
                "Released PhysTwin frame-zero nearest-particle manual identity "
                "readout and future Euclidean error."
            ),
        },
        "metrics": metrics,
        "horizons": horizons,
        "gate": {
            "required_relative_improvement": required_relative_improvement,
            "candidate_to_full_physical_ratio": ratios,
            "passed": passed,
            "next_step": (
                "Run a wider frozen opened-source panel."
                if passed
                else "Close raw PGND replacement without a wider run."
            ),
        },
    }
