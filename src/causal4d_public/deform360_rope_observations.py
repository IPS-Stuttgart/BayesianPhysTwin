"""Source-only rope dynamics observations with tactile contact registration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360 import Deform360ProtocolConfig
from .deform360_contact import validate_contact_artifact
from .deform360_rope_dynamics import RopeDynamicsObservation
from .deform360_rope_sequence import validate_rope_sequence_artifact


DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION = 3
TAXEL_ROWS_USED = 12
TAXEL_COLUMNS = 32
TAXEL_COUNT_PER_GRIPPER = 2 * TAXEL_ROWS_USED * TAXEL_COLUMNS


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class RopeSourceObservationConfig:
    selected_taxel_count: int = 8
    minimum_selected_taxel_count: int = 2
    minimum_active_sample_count: int = 4
    contact_offset_estimation_sample_count: int = 5
    maximum_patch_to_centerline_median_m: float = 0.04
    maximum_surface_to_centerline_p95_m: float = 0.03

    def __post_init__(self) -> None:
        _require(
            1 <= self.selected_taxel_count <= TAXEL_COUNT_PER_GRIPPER,
            "invalid selected-taxel count",
        )
        _require(
            2 <= self.minimum_selected_taxel_count <= self.selected_taxel_count,
            "invalid minimum selected-taxel count",
        )
        _require(
            self.minimum_active_sample_count >= 1,
            "minimum active-sample count must be positive",
        )
        _require(
            self.contact_offset_estimation_sample_count >= 1,
            "contact-offset sample count must be positive",
        )
        _require(
            self.maximum_patch_to_centerline_median_m > 0.0,
            "patch-distance gate must be positive",
        )
        _require(
            self.maximum_surface_to_centerline_p95_m > 0.0,
            "surface-distance gate must be positive",
        )


def rope_source_observation_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _interleaved_tactile(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    _require(left_values.shape == right_values.shape, "tactile-side shapes differ")
    _require(
        left_values.ndim == 3
        and left_values.shape[1] >= TAXEL_ROWS_USED
        and left_values.shape[2] == TAXEL_COLUMNS,
        "tactile arrays must have shape (T,>=12,32)",
    )
    cropped_left = left_values[:, :TAXEL_ROWS_USED].reshape(len(left_values), -1)
    cropped_right = right_values[:, :TAXEL_ROWS_USED].reshape(len(right_values), -1)
    output = np.empty((len(left_values), TAXEL_COUNT_PER_GRIPPER), dtype=np.float64)
    output[:, 0::2] = cropped_left
    output[:, 1::2] = cropped_right
    return output


def select_contact_taxels(
    left: np.ndarray,
    right: np.ndarray,
    contact_active: np.ndarray,
    *,
    selected_taxel_count: int,
    minimum_selected_taxel_count: int,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a stable tactile patch from source contact frames only."""

    values = _interleaved_tactile(left, right)
    active = np.asarray(contact_active, dtype=bool)
    _require(active.shape == (len(values),), "contact/tactile length mismatch")
    _require(np.any(active), "contact schedule has no active frame")
    counts = np.count_nonzero(values[active] > threshold, axis=0)
    strengths = np.maximum(values[active] - threshold, 0.0).sum(axis=0)
    indices = np.arange(values.shape[1], dtype=np.int64)
    order = np.lexsort((indices, -strengths, -counts))
    active_taxel_count = int(np.count_nonzero(counts > 0))
    _require(
        active_taxel_count >= minimum_selected_taxel_count,
        "fewer than two tactile taxels are active in the source contact window",
    )
    retained_count = min(selected_taxel_count, active_taxel_count)
    selected = order[:retained_count]
    return selected.astype(np.int32), {
        "maximum_selected_taxel_count": selected_taxel_count,
        "minimum_selected_taxel_count": minimum_selected_taxel_count,
        "available_active_taxel_count": active_taxel_count,
        "selected_indices": selected.astype(int).tolist(),
        "active_frame_count_by_selected_taxel": counts[selected].astype(int).tolist(),
        "positive_response_sum_by_selected_taxel": strengths[selected].tolist(),
        "selection_rule": (
            "descending active-frame count, then positive response sum, then "
            "ascending official interleaved taxel index"
        ),
    }


def _source_record(
    contact_model: Mapping[str, Any], episode_index: int
) -> Mapping[str, Any]:
    episode_id = f"001-rope/episode_{episode_index:04d}"
    matches = [
        record
        for record in contact_model.get("inputs", {}).get("source", [])
        if record.get("episode_id") == episode_id
    ]
    _require(len(matches) == 1, "contact model has no unique source episode record")
    return matches[0]


def _contact_groups(
    record: Mapping[str, Any], contact_model: Mapping[str, Any], gripper_count: int
) -> list[tuple[int, str]]:
    mapping = contact_model["model"]["tactile_group_to_robot_axis"]
    if not record["bimanual"]:
        group = record.get("mono_event_group")
        _require(isinstance(group, str), "monomanual tactile event group is missing")
        _require(
            mapping.get(group) == 0, "monomanual tactile group is not robot axis 0"
        )
        return [(0, group)]
    pairs = sorted((int(axis), str(group)) for group, axis in mapping.items())
    _require(
        [axis for axis, _ in pairs] == list(range(gripper_count)),
        "bimanual tactile groups do not cover every robot axis",
    )
    return pairs


def build_source_rope_observation(
    processed_root: str | Path,
    protocol: Deform360ProtocolConfig,
    contact_model: Mapping[str, Any],
    rope_sequence: Mapping[str, Any],
    output_archive_path: str | Path,
    *,
    config: RopeSourceObservationConfig | None = None,
) -> dict[str, Any]:
    """Register source tactile patches to one reconstructed rope sequence."""

    cfg = config or RopeSourceObservationConfig()
    validate_contact_artifact(contact_model, expected_kind="Deform360ContactModel")
    validate_rope_sequence_artifact(rope_sequence, verify_archive=True)
    episode_index = int(rope_sequence["episode_index"])
    _require(
        episode_index in protocol.source_episode_ids,
        "rope observation episode is not in the locked source split",
    )
    _require(
        rope_sequence["protocol_id"] == protocol.protocol_id
        and contact_model["protocol_id"] == protocol.protocol_id,
        "source artifacts belong to a different protocol",
    )
    record = _source_record(contact_model, episode_index)
    episode_dir = Path(processed_root).resolve() / f"episode_{episode_index:04d}"
    robot_path = episode_dir / "robot" / "robot.npz"
    _require(robot_path.is_file(), "source robot trajectory is missing")
    _require(
        _sha256_file(robot_path) == record["robot_sha256"],
        "source robot trajectory differs from the contact-model input",
    )
    try:
        from deform360.processing.control_points_stage import gripper_taxel_points
        from deform360.robot import load_robot_state
    except ImportError as error:  # pragma: no cover - pinned host integration
        raise RuntimeError(
            "the pinned Deform360 processing environment is required"
        ) from error

    state = load_robot_state(robot_path)
    with np.load(rope_sequence["archive"]["path"], allow_pickle=False) as stored:
        frame_indices = np.asarray(stored["frame_indices"], dtype=np.int32)
        positions = np.asarray(stored["centerlines_m"], dtype=np.float64)
    _require(
        len(frame_indices) == len(positions) and len(frame_indices) >= 4,
        "rope sequence frame/position count mismatch",
    )
    _require(
        np.all(np.diff(frame_indices) > 0)
        and int(frame_indices[-1]) < len(state.actions),
        "rope sequence frames are invalid for the robot trajectory",
    )
    frame_steps = np.diff(frame_indices)
    _require(
        np.all(frame_steps == frame_steps[0]),
        "rope sequence must use a uniform frame stride",
    )
    gripper_count = state.num_grippers
    pairs = _contact_groups(record, contact_model, gripper_count)
    controllers = np.empty((len(frame_indices), gripper_count, 3), dtype=np.float64)
    contact_active = np.zeros((len(frame_indices), gripper_count), dtype=bool)
    contact_offsets = np.empty((gripper_count, 3), dtype=np.float64)
    associations = []
    contact_nodes = []
    tactile_input_by_sensor = {
        item["sensor"]: item for item in record["tactile_inputs"]
    }
    for axis, group in pairs:
        window = record["group_windows"][group]
        start = window["contact_start_frame"]
        end = window["contact_end_frame"]
        _require(
            start is not None and end is not None, "source contact window is empty"
        )
        full_active = np.zeros(len(state.actions), dtype=bool)
        full_active[int(start) : int(end) + 1] = True
        contact_active[:, axis] = full_active[frame_indices]
        selected_count = int(np.count_nonzero(contact_active[:, axis]))
        _require(
            selected_count >= cfg.minimum_active_sample_count,
            "source contact window has too few sampled active frames",
        )
        tactile_values = []
        tactile_provenance = []
        for side in ("left", "right"):
            sensor = f"{group}_{side}"
            path = episode_dir / sensor / "synced_tactile.npy"
            _require(path.is_file(), f"source tactile stream is missing: {sensor}")
            _require(
                sensor in tactile_input_by_sensor, "unexpected source tactile stream"
            )
            _require(
                _sha256_file(path) == tactile_input_by_sensor[sensor]["file_sha256"],
                "source tactile stream differs from the contact-model input",
            )
            values = np.load(path, allow_pickle=False)
            _require(len(values) == len(state.actions), "robot/tactile length mismatch")
            tactile_values.append(values)
            tactile_provenance.append(
                {
                    "sensor": sensor,
                    "sha256": tactile_input_by_sensor[sensor]["file_sha256"],
                }
            )
        selected_taxels, selection = select_contact_taxels(
            tactile_values[0],
            tactile_values[1],
            full_active,
            selected_taxel_count=cfg.selected_taxel_count,
            minimum_selected_taxel_count=cfg.minimum_selected_taxel_count,
            threshold=protocol.tactile_contact_threshold,
        )
        nearest_surface = []
        for output_frame, frame_index in enumerate(frame_indices):
            if state.bimanual:
                transform = state.T_worlds[frame_index, axis]
                opening = float(state.openings[frame_index, axis])
            else:
                transform = state.T_worlds[frame_index]
                opening = float(state.openings[frame_index])
            taxels = gripper_taxel_points(opening, transform)
            _require(
                taxels.shape == (TAXEL_COUNT_PER_GRIPPER, 3),
                "official gripper taxel geometry has an unexpected shape",
            )
            controllers[output_frame, axis] = np.mean(taxels[selected_taxels], axis=0)
            nearest_surface.append(
                float(
                    np.min(
                        np.linalg.norm(
                            taxels[:, None, :] - positions[output_frame][None, :, :],
                            axis=2,
                        )
                    )
                )
            )
        active = contact_active[:, axis]
        patch_distance = np.linalg.norm(
            controllers[:, axis, None, :] - positions, axis=2
        )
        node = int(np.argmin(np.median(patch_distance[active], axis=0)))
        contact_nodes.append(node)
        active_indices = np.flatnonzero(active)
        offset_indices = active_indices[: cfg.contact_offset_estimation_sample_count]
        contact_offsets[axis] = np.median(
            positions[offset_indices, node] - controllers[offset_indices, axis],
            axis=0,
        )
        selected_distance = patch_distance[active, node]
        surface_distance = np.asarray(nearest_surface)[active]
        patch_median = float(np.median(selected_distance))
        surface_p95 = float(np.quantile(surface_distance, 0.95))
        associations.append(
            {
                "robot_axis": axis,
                "tactile_group": group,
                "contact_window_full_frames": {"start": int(start), "end": int(end)},
                "sampled_active_frame_count": selected_count,
                "selected_taxels": selection,
                "tactile_inputs": tactile_provenance,
                "contact_node_index": node,
                "contact_offset_m": contact_offsets[axis].tolist(),
                "contact_offset_estimation_sampled_frames": frame_indices[
                    offset_indices
                ]
                .astype(int)
                .tolist(),
                "patch_to_contact_node_distance_m": {
                    "median": patch_median,
                    "p95": float(np.quantile(selected_distance, 0.95)),
                    "maximum": float(np.max(selected_distance)),
                },
                "gripper_surface_to_centerline_distance_m": {
                    "median": float(np.median(surface_distance)),
                    "p95": surface_p95,
                    "maximum": float(np.max(surface_distance)),
                },
                "gates": {
                    "patch_median": {
                        "value": patch_median,
                        "maximum": cfg.maximum_patch_to_centerline_median_m,
                        "passed": patch_median
                        <= cfg.maximum_patch_to_centerline_median_m,
                    },
                    "surface_p95": {
                        "value": surface_p95,
                        "maximum": cfg.maximum_surface_to_centerline_p95_m,
                        "passed": surface_p95
                        <= cfg.maximum_surface_to_centerline_p95_m,
                    },
                },
            }
        )
    distinct_nodes = len(set(contact_nodes)) == len(contact_nodes)
    gate_values = [
        gate["passed"]
        for association in associations
        for gate in association["gates"].values()
    ]
    quality = {
        "passed": bool(all(gate_values) and (gripper_count == 1 or distinct_nodes)),
        "distinct_bimanual_contact_nodes": {
            "value": distinct_nodes,
            "required": gripper_count > 1,
            "passed": gripper_count == 1 or distinct_nodes,
        },
    }
    output = Path(output_archive_path).resolve()
    _require(output.suffix == ".npz", "rope observation archive must end in .npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    contact_node_array = np.asarray(contact_nodes, dtype=np.int32)
    np.savez_compressed(
        output,
        frame_indices=frame_indices,
        positions_m=positions,
        controller_positions_m=controllers,
        contact_active=contact_active,
        contact_node_indices=contact_node_array,
        contact_offsets_m=contact_offsets,
    )
    payload: dict[str, Any] = {
        "schema_version": DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION,
        "artifact_kind": "Deform360SourceRopeDynamicsObservation",
        "protocol_id": protocol.protocol_id,
        "episode_index": episode_index,
        "episode_id": record["episode_id"],
        "action": record["action"],
        "split": "source",
        "parameters": asdict(cfg),
        "frame_indices": frame_indices.astype(int).tolist(),
        "frame_stride": int(frame_steps[0]),
        "dt_seconds": float(
            frame_steps[0] * protocol.nominal_frame_interval_us / 1_000_000.0
        ),
        "contact_associations": associations,
        "quality": quality,
        "inputs": {
            "rope_sequence_result_sha256": rope_sequence["result_sha256"],
            "contact_model_result_sha256": contact_model["result_sha256"],
            "robot_sha256": record["robot_sha256"],
        },
        "archive": {
            "path": str(output),
            "sha256": _sha256_file(output),
            "bytes": output.stat().st_size,
            "positions_sha256": _sha256_array(positions),
            "controller_positions_sha256": _sha256_array(controllers),
            "contact_active_sha256": _sha256_array(contact_active),
            "contact_offsets_sha256": _sha256_array(contact_offsets),
        },
        "information_boundary": {
            "source_episode_only": True,
            "source_tactile_read": True,
            "target_files_read": False,
            "target_metrics_computed": False,
        },
        "measurement_semantics": {
            "object_nodes": (
                "normalized-arc-length silhouette pseudo-correspondences, not "
                "verified material identities"
            ),
            "controller": (
                "official Deform360 gripper taxel geometry restricted to a "
                "source-tactile-selected stable patch"
            ),
            "contact": "source tactile contact window; unitless normal response",
        },
    }
    payload["result_sha256"] = rope_source_observation_artifact_sha256(payload)
    return payload


def validate_source_rope_observation_artifact(
    payload: Mapping[str, Any], *, verify_archive: bool = True
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION,
        "unsupported rope-observation artifact schema",
    )
    _require(
        payload.get("artifact_kind") == "Deform360SourceRopeDynamicsObservation",
        "unexpected rope-observation artifact kind",
    )
    _require(
        payload.get("result_sha256")
        == rope_source_observation_artifact_sha256(payload),
        "rope-observation artifact checksum mismatch",
    )
    _require(payload.get("split") == "source", "rope observation is not source-only")
    _require(
        payload.get("information_boundary", {}).get("target_files_read") is False,
        "rope observation read target files",
    )
    _require(
        payload.get("quality", {}).get("passed") is True,
        "rope observation failed contact-registration gates",
    )
    if verify_archive:
        archive = Path(payload["archive"]["path"])
        _require(archive.is_file(), "rope observation archive is missing")
        _require(
            _sha256_file(archive) == payload["archive"]["sha256"],
            "rope observation archive checksum mismatch",
        )
        with np.load(archive, allow_pickle=False) as stored:
            expected = {
                "frame_indices",
                "positions_m",
                "controller_positions_m",
                "contact_active",
                "contact_node_indices",
                "contact_offsets_m",
            }
            _require(set(stored.files) == expected, "rope observation fields differ")
            _require(
                _sha256_array(stored["positions_m"])
                == payload["archive"]["positions_sha256"],
                "rope observation positions checksum mismatch",
            )
            _require(
                _sha256_array(stored["controller_positions_m"])
                == payload["archive"]["controller_positions_sha256"],
                "rope observation controller checksum mismatch",
            )
            _require(
                _sha256_array(stored["contact_active"])
                == payload["archive"]["contact_active_sha256"],
                "rope observation contact checksum mismatch",
            )
            _require(
                _sha256_array(stored["contact_offsets_m"])
                == payload["archive"]["contact_offsets_sha256"],
                "rope observation contact-offset checksum mismatch",
            )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "episode_id": payload["episode_id"],
    }


def load_source_rope_dynamics_observation(
    payload: Mapping[str, Any],
) -> RopeDynamicsObservation:
    validate_source_rope_observation_artifact(payload, verify_archive=True)
    with np.load(payload["archive"]["path"], allow_pickle=False) as stored:
        return RopeDynamicsObservation(
            episode_id=str(payload["episode_id"]),
            positions_m=np.asarray(stored["positions_m"], dtype=np.float64),
            controller_positions_m=np.asarray(
                stored["controller_positions_m"], dtype=np.float64
            ),
            contact_active=np.asarray(stored["contact_active"], dtype=bool),
            contact_node_indices=tuple(
                map(int, np.asarray(stored["contact_node_indices"]).tolist())
            ),
            contact_offsets_m=np.asarray(stored["contact_offsets_m"], dtype=np.float64),
            dt_seconds=float(payload["dt_seconds"]),
        )


def write_source_rope_observation_artifact(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION",
    "RopeSourceObservationConfig",
    "build_source_rope_observation",
    "load_source_rope_dynamics_observation",
    "rope_source_observation_artifact_sha256",
    "select_contact_taxels",
    "validate_source_rope_observation_artifact",
    "write_source_rope_observation_artifact",
]
