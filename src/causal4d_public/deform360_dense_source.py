"""Source-only helpers for dense Deform360-to-PhysTwin feasibility runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360_replication import load_deform360_replication_protocol


DEFORM360_SUPPORT_DYNAMICS = (
    "official-ground",
    "gravity-neutral-planar",
)


def support_dynamics_reverse_factor(
    mode: str,
    *,
    reverse_z: bool,
) -> float:
    """Return the official simulator's gravity/ground orientation factor.

    ``gravity-neutral-planar`` is an exploratory reduced-order regime for an
    object already settled on a horizontal support. It disables both the
    official gravity term and its z=0 collision response. The caller must keep
    it opt-in because it is inappropriate when gravity-driven free motion is a
    material part of the evaluated action.
    """
    if mode not in DEFORM360_SUPPORT_DYNAMICS:
        raise ValueError(f"unknown Deform360 support dynamics mode: {mode!r}")
    if mode == "gravity-neutral-planar":
        return 0.0
    return -1.0 if reverse_z else 1.0


@dataclass(frozen=True)
class SparseControllerPatch:
    """A frame-zero, geometry-only association to gripper surface points."""

    controller_indices: np.ndarray
    nearest_object_indices: np.ndarray
    initial_distances_m: np.ndarray

    def __post_init__(self) -> None:
        controllers = np.asarray(self.controller_indices, dtype=np.int32)
        objects = np.asarray(self.nearest_object_indices, dtype=np.int32)
        distances = np.asarray(self.initial_distances_m, dtype=np.float64)
        if (
            controllers.ndim != 1
            or objects.shape != controllers.shape
            or distances.shape != controllers.shape
            or len(controllers) < 1
        ):
            raise ValueError("sparse controller association arrays differ")
        if len(np.unique(controllers)) != len(controllers):
            raise ValueError("sparse controller association repeats a point")
        if np.any(controllers < 0) or np.any(objects < 0):
            raise ValueError("sparse controller association index is negative")
        if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
            raise ValueError("sparse controller association distance is invalid")
        for name, value in (
            ("controller_indices", controllers),
            ("nearest_object_indices", objects),
            ("initial_distances_m", distances),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


@dataclass(frozen=True)
class PhysTwinSupportFrame:
    """A proper rigid map from an annotation support frame into z-up Warp."""

    rotation_world_to_sim: np.ndarray
    translation_sim_m: np.ndarray
    support_axis: int
    free_space_sign: int
    support_location_world_m: float
    clearance_m: float

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_world_to_sim, dtype=np.float64)
        translation = np.asarray(self.translation_sim_m, dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("support-frame transform has an invalid shape")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10):
            raise ValueError("support-frame rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10):
            raise ValueError("support-frame transform is not a proper rotation")
        if self.support_axis not in (0, 1, 2) or self.free_space_sign not in (-1, 1):
            raise ValueError("support-frame axis or sign is invalid")
        if not np.isfinite(self.support_location_world_m) or self.clearance_m < 0.0:
            raise ValueError("support-frame offset is invalid")
        for name, value in (
            ("rotation_world_to_sim", rotation),
            ("translation_sim_m", translation),
        ):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)

    def transform(self, points_world_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_m, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.all(np.isfinite(points)):
            raise ValueError("support-frame input points are invalid")
        return points @ self.rotation_world_to_sim.T + self.translation_sim_m


def fit_phystwin_support_frame(
    initial_object_points_world_m: np.ndarray,
    *,
    support_axis: int,
    free_space_sign: int = 1,
    support_quantile: float = 0.01,
    clearance_m: float = 0.002,
) -> PhysTwinSupportFrame:
    """Map a declared support normal to +z using frame-zero geometry only."""

    points = np.asarray(initial_object_points_world_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 1:
        raise ValueError("support-frame object points must have shape (N,3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("support-frame object points are non-finite")
    if support_axis not in (0, 1, 2) or free_space_sign not in (-1, 1):
        raise ValueError("support-frame axis or sign is invalid")
    if not 0.0 <= support_quantile <= 0.5 or clearance_m < 0.0:
        raise ValueError("support-frame quantile or clearance is invalid")
    normal = np.zeros(3, dtype=np.float64)
    normal[support_axis] = free_space_sign
    candidates = np.eye(3)
    reference = candidates[np.argmin(np.abs(candidates @ normal))]
    x_axis = reference - normal * np.dot(reference, normal)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    rotation = np.stack((x_axis, y_axis, normal))
    projected = points @ normal
    support_location = float(np.quantile(projected, support_quantile))
    translation = np.asarray((0.0, 0.0, -support_location + clearance_m))
    return PhysTwinSupportFrame(
        rotation_world_to_sim=rotation,
        translation_sim_m=translation,
        support_axis=support_axis,
        free_space_sign=free_space_sign,
        support_location_world_m=support_location,
        clearance_m=clearance_m,
    )


def select_sparse_controller_patch(
    initial_object_points_m: np.ndarray,
    initial_controller_points_m: np.ndarray,
    *,
    count: int,
    minimum_separation_m: float = 0.004,
) -> SparseControllerPatch:
    """Select a deterministic contact patch without reading future motion."""

    objects = np.asarray(initial_object_points_m, dtype=np.float64)
    controllers = np.asarray(initial_controller_points_m, dtype=np.float64)
    if (
        objects.ndim != 2
        or controllers.ndim != 2
        or objects.shape[1:] != (3,)
        or controllers.shape[1:] != (3,)
        or len(objects) < 1
        or len(controllers) < 1
    ):
        raise ValueError("object and controller points must have shape (N,3)")
    if not np.all(np.isfinite(objects)) or not np.all(np.isfinite(controllers)):
        raise ValueError("object or controller points are non-finite")
    if not 1 <= count <= len(controllers):
        raise ValueError("sparse controller count is invalid")
    if minimum_separation_m < 0.0:
        raise ValueError("minimum controller separation is negative")
    squared = np.sum(
        (controllers[:, None, :] - objects[None, :, :]) ** 2,
        axis=2,
    )
    nearest_objects = np.argmin(squared, axis=1)
    nearest_distances = np.sqrt(squared[np.arange(len(controllers)), nearest_objects])
    order = np.lexsort((np.arange(len(controllers)), nearest_distances))
    selected: list[int] = []
    for index in order:
        if selected and np.any(
            np.linalg.norm(controllers[selected] - controllers[index], axis=1)
            < minimum_separation_m
        ):
            continue
        selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(
            int(index)
            for index in order
            if int(index) not in selected_set
        )
        selected = selected[:count]
    indices = np.asarray(selected, dtype=np.int32)
    return SparseControllerPatch(
        controller_indices=indices,
        nearest_object_indices=nearest_objects[indices],
        initial_distances_m=nearest_distances[indices],
    )


def associate_controller_material_patch(
    initial_object_points_m: np.ndarray,
    initial_controller_points_m: np.ndarray,
    controller_indices: np.ndarray,
) -> SparseControllerPatch:
    """Attach a source-learned gripper patch to frame-zero object geometry.

    Controller indices identify material locations on the gripper and may be
    transferred between episodes. The object-side attachment remains
    episode-specific and is inferred from the new episode's first frame only.
    """
    objects = np.asarray(initial_object_points_m, dtype=np.float64)
    controllers = np.asarray(initial_controller_points_m, dtype=np.float64)
    indices = np.asarray(controller_indices, dtype=np.int64)
    if (
        objects.ndim != 2
        or controllers.ndim != 2
        or objects.shape[1:] != (3,)
        or controllers.shape[1:] != (3,)
        or len(objects) < 1
        or len(controllers) < 1
    ):
        raise ValueError("object and controller points must have shape (N,3)")
    if not np.all(np.isfinite(objects)) or not np.all(np.isfinite(controllers)):
        raise ValueError("object or controller points are non-finite")
    if (
        indices.ndim != 1
        or len(indices) < 1
        or np.any(indices < 0)
        or np.any(indices >= len(controllers))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("controller material indices are invalid")
    selected = controllers[indices]
    squared = np.sum((selected[:, None, :] - objects[None, :, :]) ** 2, axis=2)
    nearest_objects = np.argmin(squared, axis=1)
    nearest_distances = np.sqrt(
        squared[np.arange(len(selected)), nearest_objects]
    )
    return SparseControllerPatch(
        controller_indices=indices.astype(np.int32),
        nearest_object_indices=nearest_objects.astype(np.int32),
        initial_distances_m=nearest_distances,
    )


def fit_source_controller_patch(
    object_points_m: np.ndarray,
    controller_points_m: np.ndarray,
    *,
    count: int,
    maximum_initial_distance_m: float = 0.02,
    proximity_weight: float = 0.25,
    minimum_separation_m: float = 0.004,
) -> tuple[SparseControllerPatch, dict[str, Any]]:
    """Fit a reusable gripper patch from source-prefix co-motion only.

    Every controller taxel is paired with its nearest frame-zero object node,
    matching the one-neighbour official PhysTwin attachment used by the smoke
    runner. Candidates are ranked by displacement agreement over the supplied
    source frames plus a small frame-zero proximity penalty. Callers must pass
    only their declared fitting interval.
    """
    objects = np.asarray(object_points_m, dtype=np.float64)
    controllers = np.asarray(controller_points_m, dtype=np.float64)
    if (
        objects.ndim != 3
        or controllers.ndim != 3
        or objects.shape[0] != controllers.shape[0]
        or objects.shape[2:] != (3,)
        or controllers.shape[2:] != (3,)
        or len(objects) < 2
        or objects.shape[1] < 1
        or controllers.shape[1] < 1
    ):
        raise ValueError("source association trajectories have invalid shapes")
    if not np.all(np.isfinite(objects)) or not np.all(np.isfinite(controllers)):
        raise ValueError("source association trajectories are non-finite")
    if not 1 <= count <= controllers.shape[1]:
        raise ValueError("source association controller count is invalid")
    if maximum_initial_distance_m <= 0.0:
        raise ValueError("source association distance gate must be positive")
    if proximity_weight < 0.0 or minimum_separation_m < 0.0:
        raise ValueError("source association regularizer is negative")

    initial_objects = objects[0]
    initial_controllers = controllers[0]
    squared = np.sum(
        (initial_controllers[:, None, :] - initial_objects[None, :, :]) ** 2,
        axis=2,
    )
    nearest_objects = np.argmin(squared, axis=1)
    indices = np.arange(len(initial_controllers))
    nearest_distances = np.sqrt(squared[indices, nearest_objects])
    object_displacements = objects - objects[:1]
    controller_displacements = controllers - controllers[:1]
    paired_displacements = object_displacements[:, nearest_objects]
    mismatch = paired_displacements - controller_displacements
    motion_rmse = np.sqrt(np.mean(mismatch**2, axis=(0, 2)))
    scores = motion_rmse + proximity_weight * nearest_distances
    eligible = nearest_distances <= maximum_initial_distance_m
    if int(np.count_nonzero(eligible)) < count:
        raise ValueError("too few source association candidates pass distance gate")
    order = np.lexsort(
        (indices, nearest_distances, motion_rmse, scores, ~eligible)
    )
    selected: list[int] = []
    for index in order:
        if not eligible[index]:
            continue
        if selected and np.any(
            np.linalg.norm(
                initial_controllers[selected] - initial_controllers[index], axis=1
            )
            < minimum_separation_m
        ):
            continue
        selected.append(int(index))
        if len(selected) == count:
            break
    if len(selected) < count:
        selected_set = set(selected)
        selected.extend(
            int(index)
            for index in order
            if eligible[index] and int(index) not in selected_set
        )
        selected = selected[:count]
    selected_indices = np.asarray(selected, dtype=np.int32)
    patch = SparseControllerPatch(
        controller_indices=selected_indices,
        nearest_object_indices=nearest_objects[selected_indices],
        initial_distances_m=nearest_distances[selected_indices],
    )
    diagnostics = {
        "fit_frame_range": [0, len(objects)],
        "fit_frame_count": len(objects),
        "candidate_count": int(np.count_nonzero(eligible)),
        "maximum_initial_distance_m": maximum_initial_distance_m,
        "proximity_weight": proximity_weight,
        "minimum_separation_m": minimum_separation_m,
        "selected_motion_rmse_m": motion_rmse[selected_indices].tolist(),
        "selected_score_m": scores[selected_indices].tolist(),
        "selection_rule": (
            "nearest frame-zero object node per taxel, ranked by source-prefix "
            "displacement RMSE plus proximity penalty"
        ),
        "held_out_motion_read": False,
    }
    return patch, diagnostics


def require_source_episode(
    protocol_path: str | Path,
    object_id: str,
    episode_index: int,
) -> Mapping[str, Any]:
    """Return the locked object record only when the episode is a source case."""

    protocol = load_deform360_replication_protocol(protocol_path)
    records = {
        str(record["object_id"]): record for record in protocol["config"]["cohort"]
    }
    if object_id not in records:
        raise ValueError(f"{object_id!r} is not in the locked replication cohort")
    record = records[object_id]
    source_ids = tuple(int(value) for value in record["source_episode_ids"])
    if int(episode_index) not in source_ids:
        raise ValueError(
            f"episode {episode_index} is not a source episode for {object_id}; "
            "calibration and target episodes are forbidden"
        )
    return record


def unpack_sampled_mask(
    archive: Mapping[str, np.ndarray],
    camera: str,
    frame_index: int,
) -> np.ndarray:
    """Decode one sealed source mask from a packed sampled-mask archive."""

    cameras = np.asarray(archive["cameras"]).astype(str)
    frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
    image_shape = np.asarray(archive["image_shape"], dtype=np.int64)
    packed = np.asarray(archive["packed_masks"], dtype=np.uint8)
    if image_shape.shape != (2,) or np.any(image_shape <= 0):
        raise ValueError("sampled mask image_shape must contain positive height/width")
    camera_matches = np.flatnonzero(cameras == str(camera))
    frame_matches = np.flatnonzero(frame_indices == int(frame_index))
    if len(camera_matches) != 1:
        raise ValueError(f"camera {camera!r} is not unique in sampled masks")
    if len(frame_matches) != 1:
        raise ValueError(f"frame {frame_index} is not unique in sampled masks")
    height, width = (int(value) for value in image_shape)
    encoded = packed[int(camera_matches[0]), int(frame_matches[0])]
    if encoded.shape != (height, (width + 7) // 8):
        raise ValueError("packed sampled mask shape does not match image_shape")
    return np.unpackbits(encoded, axis=-1, count=width).astype(bool, copy=False)


def sha256_file(path: str | Path) -> str:
    """Hash a source artifact without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dense_source_manifest(
    path: str | Path,
    *,
    protocol_path: str | Path,
    object_id: str,
    episode_index: int,
    source_episode_dir: str | Path,
    sampled_masks_path: str | Path,
    start_frame: int,
    frame_count: int,
    cameras: list[str],
    outputs: Mapping[str, Any],
) -> None:
    """Record the information boundary and checksums for a staged smoke run."""

    manifest = {
        "schema": "causal4d/deform360-dense-source-smoke/v1",
        "object_id": object_id,
        "episode_index": int(episode_index),
        "source_only": True,
        "calibration_episode_accessed": False,
        "target_episode_accessed": False,
        "future_outcomes_used_for_fitting": False,
        "frame_range": [int(start_frame), int(start_frame + frame_count)],
        "cameras": sorted(cameras),
        "inputs": {
            "protocol_path": str(Path(protocol_path).resolve()),
            "protocol_sha256": sha256_file(protocol_path),
            "source_episode_dir": str(Path(source_episode_dir).resolve()),
            "sampled_masks_path": str(Path(sampled_masks_path).resolve()),
            "sampled_masks_sha256": sha256_file(sampled_masks_path),
        },
        "outputs": dict(outputs),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DEFORM360_SUPPORT_DYNAMICS",
    "PhysTwinSupportFrame",
    "SparseControllerPatch",
    "associate_controller_material_patch",
    "fit_source_controller_patch",
    "fit_phystwin_support_frame",
    "require_source_episode",
    "select_sparse_controller_patch",
    "sha256_file",
    "support_dynamics_reverse_factor",
    "unpack_sampled_mask",
    "write_dense_source_manifest",
]
