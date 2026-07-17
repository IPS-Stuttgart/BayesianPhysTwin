"""Contact-conditioned Deform360 actions for reusable PhysTwin rollouts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CONTACT_CONDITIONED_ACTION_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ContactConditionedControllerAction:
    """Controller groups aligned to their first predicted material contact."""

    controller_points_m: np.ndarray
    contact_active: np.ndarray
    source_group_indices: tuple[int, ...]
    onset_frames: tuple[int, ...]
    initial_contact_distance_m: tuple[float, ...]
    source_group_size: int
    source_group_count: int
    maximum_contact_distance_m: float

    def __post_init__(self) -> None:
        controls = np.asarray(self.controller_points_m, dtype=np.float64)
        active = np.asarray(self.contact_active, dtype=bool)
        group_count = len(self.source_group_indices)
        _require(
            controls.ndim == 3 and controls.shape[0] >= 2 and controls.shape[2] == 3,
            "conditioned controller points must have shape (T,P,3)",
        )
        _require(
            self.source_group_size >= 1 and self.source_group_count >= 1,
            "source controller grouping is invalid",
        )
        _require(
            controls.shape[1] == group_count * self.source_group_size,
            "conditioned controller points differ from retained groups",
        )
        _require(
            active.shape == (len(controls), group_count),
            "conditioned contact schedule differs from retained groups",
        )
        _require(
            len(self.onset_frames)
            == len(self.initial_contact_distance_m)
            == group_count,
            "conditioned group metadata has inconsistent cardinality",
        )
        _require(
            tuple(sorted(set(self.source_group_indices))) == self.source_group_indices
            and all(
                0 <= index < self.source_group_count
                for index in self.source_group_indices
            ),
            "retained source group indices are invalid",
        )
        _require(
            all(
                0 <= onset < len(controls)
                and active[onset, group]
                and not np.any(active[:onset, group])
                for group, onset in enumerate(self.onset_frames)
            ),
            "conditioned onset does not match the first active frame",
        )
        _require(
            np.isfinite(self.maximum_contact_distance_m)
            and self.maximum_contact_distance_m > 0.0
            and all(
                np.isfinite(distance)
                and 0.0 <= distance <= self.maximum_contact_distance_m
                for distance in self.initial_contact_distance_m
            ),
            "conditioned contact distance is invalid",
        )
        _require(np.all(np.isfinite(controls)), "conditioned controls are nonfinite")
        copied_controls = controls.copy()
        copied_active = active.copy()
        copied_controls.setflags(write=False)
        copied_active.setflags(write=False)
        object.__setattr__(self, "controller_points_m", copied_controls)
        object.__setattr__(self, "contact_active", copied_active)

    @property
    def retained_group_count(self) -> int:
        return len(self.source_group_indices)

    @property
    def falls_back_to_persistence(self) -> bool:
        return self.retained_group_count == 0

    @property
    def controller_reference_points_m(self) -> np.ndarray:
        return self.controller_points_m[0]


def _minimum_group_distance_m(
    group_points_m: np.ndarray, initial_object_points_m: np.ndarray
) -> float:
    group = np.asarray(group_points_m, dtype=np.float64)
    object_points = np.asarray(initial_object_points_m, dtype=np.float64)
    minimum = np.inf
    block_size = 256
    for start in range(0, len(group), block_size):
        difference = (
            group[start : start + block_size, None, :] - object_points[None, :, :]
        )
        minimum = min(minimum, float(np.min(np.linalg.norm(difference, axis=2))))
    return minimum


def condition_controller_action(
    controller_points_m: np.ndarray,
    contact_active: np.ndarray,
    initial_object_points_m: np.ndarray,
    *,
    controller_group_size: int,
    maximum_contact_distance_m: float,
) -> ContactConditionedControllerAction:
    """Align retained controller groups to predicted onset using frame-zero geometry.

    The function uses the known controller trajectory, a contact schedule predicted
    without target tactile, and only the initial object geometry. Groups without a
    predicted contact or with an implausible onset separation are removed. Before a
    retained group's onset, its controller geometry is held at the onset pose so an
    inactive attachment has no approach-induced displacement.
    """

    controls = np.asarray(controller_points_m, dtype=np.float64)
    schedule = np.asarray(contact_active, dtype=bool)
    initial = np.asarray(initial_object_points_m, dtype=np.float64)
    _require(
        controls.ndim == 3 and controls.shape[0] >= 2 and controls.shape[2] == 3,
        "controller points must have shape (T,P,3)",
    )
    _require(
        initial.ndim == 2 and initial.shape[1] == 3 and len(initial) >= 1,
        "initial object points must have shape (N,3)",
    )
    _require(
        controller_group_size >= 1 and controls.shape[1] % controller_group_size == 0,
        "controller points do not form complete groups",
    )
    source_group_count = controls.shape[1] // controller_group_size
    _require(
        schedule.shape == (len(controls), source_group_count),
        "contact schedule must have shape (T,G)",
    )
    _require(
        np.all(np.isfinite(controls)) and np.all(np.isfinite(initial)),
        "controller and object points must be finite",
    )
    _require(
        np.isfinite(maximum_contact_distance_m) and maximum_contact_distance_m > 0.0,
        "maximum contact distance must be finite and positive",
    )

    retained_controls = []
    retained_active = []
    source_indices = []
    onset_frames = []
    distances = []
    for source_group in range(source_group_count):
        active_frames = np.flatnonzero(schedule[:, source_group])
        if not len(active_frames):
            continue
        onset = int(active_frames[0])
        start = source_group * controller_group_size
        stop = start + controller_group_size
        group = controls[:, start:stop].copy()
        reference = group[onset].copy()
        distance = _minimum_group_distance_m(reference, initial)
        if distance > maximum_contact_distance_m:
            continue
        group[: onset + 1] = reference
        retained_controls.append(group)
        retained_active.append(schedule[:, source_group])
        source_indices.append(source_group)
        onset_frames.append(onset)
        distances.append(distance)

    if retained_controls:
        conditioned = np.concatenate(retained_controls, axis=1)
        active = np.column_stack(retained_active)
    else:
        conditioned = np.empty((len(controls), 0, 3), dtype=np.float64)
        active = np.empty((len(controls), 0), dtype=bool)
    return ContactConditionedControllerAction(
        controller_points_m=conditioned,
        contact_active=active,
        source_group_indices=tuple(source_indices),
        onset_frames=tuple(onset_frames),
        initial_contact_distance_m=tuple(distances),
        source_group_size=controller_group_size,
        source_group_count=source_group_count,
        maximum_contact_distance_m=float(maximum_contact_distance_m),
    )


def controller_spring_group_indices(
    springs: np.ndarray,
    *,
    num_object_springs: int,
    controller_vertex_start: int,
    controller_point_count: int,
    controller_group_size: int,
    retained_group_count: int,
) -> np.ndarray:
    """Map trailing controller springs to packed retained controller groups."""

    edges = np.asarray(springs, dtype=np.int64)
    _require(
        edges.ndim == 2 and edges.shape[1] == 2,
        "springs must have shape (S,2)",
    )
    _require(
        0 < num_object_springs < len(edges),
        "object spring count must leave controller springs",
    )
    _require(
        controller_vertex_start >= 1
        and controller_point_count >= 1
        and controller_group_size >= 1
        and retained_group_count >= 1,
        "controller spring grouping is invalid",
    )
    _require(
        controller_point_count == controller_group_size * retained_group_count,
        "packed controller points differ from retained groups",
    )
    controller_stop = controller_vertex_start + controller_point_count
    result = []
    for edge in edges[num_object_springs:]:
        controller_endpoints = edge[
            (edge >= controller_vertex_start) & (edge < controller_stop)
        ]
        _require(
            len(controller_endpoints) == 1,
            "each controller spring must have exactly one controller endpoint",
        )
        local_index = int(controller_endpoints[0]) - controller_vertex_start
        group = local_index // controller_group_size
        _require(group < retained_group_count, "controller spring group is invalid")
        result.append(group)
    groups = np.asarray(result, dtype=np.int64)
    _require(
        set(groups.tolist()) == set(range(retained_group_count)),
        "not every retained controller group has an attachment spring",
    )
    return groups


def write_contact_conditioned_action_artifact(
    archive_path: str | Path,
    action: ContactConditionedControllerAction,
    *,
    object_id: str,
    episode_id: int,
    source_controller_sha256: str,
    contact_model_result_sha256: str,
    information_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Write a checksum-bound action artifact for an opt-in Warp rollout."""

    _require(len(source_controller_sha256) == 64, "source controller hash is invalid")
    _require(len(contact_model_result_sha256) == 64, "contact model hash is invalid")
    boundary = dict(information_boundary)
    _require(
        boundary.get("future_object_observations_used") is False
        and boundary.get("target_tactile_used") is False
        and boundary.get("known_future_robot_action_used") is True,
        "contact-conditioned action crossed its prediction boundary",
    )
    archive = Path(archive_path).resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive,
        controller_points_m=action.controller_points_m,
        contact_active=action.contact_active.astype(np.uint8),
        source_group_indices=np.asarray(action.source_group_indices, dtype=np.int64),
        onset_frames=np.asarray(action.onset_frames, dtype=np.int64),
        initial_contact_distance_m=np.asarray(
            action.initial_contact_distance_m, dtype=np.float64
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": CONTACT_CONDITIONED_ACTION_SCHEMA_VERSION,
        "artifact_kind": "Deform360ContactConditionedControllerAction",
        "object_id": str(object_id),
        "episode_id": int(episode_id),
        "frame_count": len(action.controller_points_m),
        "source_group_size": action.source_group_size,
        "source_group_count": action.source_group_count,
        "retained_group_count": action.retained_group_count,
        "source_group_indices": list(action.source_group_indices),
        "onset_frames": list(action.onset_frames),
        "initial_contact_distance_m": list(action.initial_contact_distance_m),
        "maximum_contact_distance_m": action.maximum_contact_distance_m,
        "falls_back_to_persistence": action.falls_back_to_persistence,
        "source_controller_sha256": source_controller_sha256,
        "contact_model_result_sha256": contact_model_result_sha256,
        "archive": {
            "path": str(archive),
            "sha256": _file_sha256(archive),
            "array_sha256": {
                "controller_points_m": _array_sha256(action.controller_points_m),
                "contact_active": _array_sha256(action.contact_active.astype(np.uint8)),
            },
        },
        "information_boundary": boundary,
        "claim_boundary": (
            "source-trained contact realization from known action and frame-zero "
            "geometry; no target tactile or future object observation"
        ),
    }
    payload["result_sha256"] = _artifact_sha256(payload)
    return payload


def load_contact_conditioned_action_artifact(
    payload: Mapping[str, Any],
) -> ContactConditionedControllerAction:
    """Validate and load a contact-conditioned action artifact."""

    _require(
        payload.get("schema_version") == CONTACT_CONDITIONED_ACTION_SCHEMA_VERSION
        and payload.get("artifact_kind")
        == "Deform360ContactConditionedControllerAction",
        "contact-conditioned action identity changed",
    )
    _require(
        payload.get("result_sha256") == _artifact_sha256(payload),
        "contact-conditioned action checksum changed",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("future_object_observations_used") is False
        and boundary.get("target_tactile_used") is False
        and boundary.get("known_future_robot_action_used") is True,
        "contact-conditioned action crossed its prediction boundary",
    )
    archive = Path(str(payload.get("archive", {}).get("path", "")))
    _require(
        archive.is_file() and _file_sha256(archive) == payload["archive"]["sha256"],
        "contact-conditioned action archive changed",
    )
    with np.load(archive, allow_pickle=False) as stored:
        action = ContactConditionedControllerAction(
            controller_points_m=np.asarray(
                stored["controller_points_m"], dtype=np.float64
            ),
            contact_active=np.asarray(stored["contact_active"], dtype=bool),
            source_group_indices=tuple(
                map(int, np.asarray(stored["source_group_indices"]).tolist())
            ),
            onset_frames=tuple(map(int, np.asarray(stored["onset_frames"]).tolist())),
            initial_contact_distance_m=tuple(
                map(
                    float,
                    np.asarray(stored["initial_contact_distance_m"]).tolist(),
                )
            ),
            source_group_size=int(payload["source_group_size"]),
            source_group_count=int(payload["source_group_count"]),
            maximum_contact_distance_m=float(payload["maximum_contact_distance_m"]),
        )
    _require(
        action.retained_group_count == int(payload["retained_group_count"])
        and action.falls_back_to_persistence
        is bool(payload["falls_back_to_persistence"]),
        "contact-conditioned action metadata changed",
    )
    return action


__all__ = [
    "CONTACT_CONDITIONED_ACTION_SCHEMA_VERSION",
    "ContactConditionedControllerAction",
    "condition_controller_action",
    "controller_spring_group_indices",
    "load_contact_conditioned_action_artifact",
    "write_contact_conditioned_action_artifact",
]
