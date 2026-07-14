"""Seal deployable Deform360 rope rollouts before opening target outcomes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .deform360 import Deform360ProtocolConfig
from .deform360_contact import validate_contact_artifact
from .deform360_rope_dynamics import rollout_rope_dynamics
from .deform360_rope_evaluation import (
    build_oracle_tactile_rope_prediction,
    seal_held_out_rope_predictions,
    validate_held_out_rope_prediction_seal,
)
from .deform360_rope_fit import (
    load_forward_rope_fit_parameters,
    validate_forward_rope_fit_artifact,
)
from .deform360_rope_prefix import validate_target_prefix_rope_geometry


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class RopeTargetPredictionConfig:
    visual_patch_taxel_count: int = 8
    maximum_patch_to_centerline_m: float = 0.02

    def __post_init__(self) -> None:
        _require(self.visual_patch_taxel_count >= 2, "visual patch needs two taxels")
        _require(
            self.maximum_patch_to_centerline_m > 0.0,
            "contact association gate must be positive",
        )


def propagate_prefix_contact_state(
    visual_contact_active: np.ndarray, prefix_end_state: np.ndarray
) -> np.ndarray:
    """Propagate prefix contact until a future visual transition occurs."""

    visual = np.asarray(visual_contact_active, dtype=bool)
    state = np.asarray(prefix_end_state, dtype=bool)
    _require(visual.ndim == 2 and len(visual) >= 1, "visual schedule must be (T,C)")
    _require(state.shape == (visual.shape[1],), "prefix contact-state shape mismatch")
    output = np.empty_like(visual)
    output[0] = state
    for frame in range(1, len(visual)):
        transition = visual[frame] != visual[frame - 1]
        state = np.where(transition, visual[frame], state)
        output[frame] = state
    return output


def select_visual_contact_patch(
    taxel_points_m: np.ndarray,
    centerline_m: np.ndarray,
    *,
    taxel_count: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, dict[str, float]]:
    """Associate one gripper to the nearest rope endpoint using prefix geometry."""

    taxels = np.asarray(taxel_points_m, dtype=np.float64)
    centerline = np.asarray(centerline_m, dtype=np.float64)
    _require(taxels.ndim == 2 and taxels.shape[1] == 3, "taxels must be (K,3)")
    _require(
        centerline.ndim == 2 and centerline.shape[1] == 3,
        "centerline must be (N,3)",
    )
    _require(2 <= taxel_count <= len(taxels), "invalid visual patch size")
    surface_distance = np.min(
        np.linalg.norm(taxels[:, None, :] - centerline[None, :, :], axis=2), axis=1
    )
    indices = np.arange(len(taxels), dtype=np.int64)
    selected = np.lexsort((indices, surface_distance))[:taxel_count]
    patch = np.mean(taxels[selected], axis=0)
    node_distances = np.linalg.norm(centerline - patch, axis=1)
    node = int(np.argmin(node_distances))
    offset = centerline[node] - patch
    return (
        selected.astype(np.int32),
        patch,
        node,
        offset,
        {
            "nearest_surface_distance_m": float(surface_distance[selected[0]]),
            "patch_to_node_distance_m": float(node_distances[node]),
        },
    )


def build_and_seal_target_rope_predictions(
    processed_root: str | Path,
    protocol: Deform360ProtocolConfig,
    contact_prediction_seal: Mapping[str, Any],
    shared_fit: Mapping[str, Any],
    prefix_geometry: Mapping[str, Any],
    prediction_archive_path: str | Path,
    *,
    config: RopeTargetPredictionConfig = RopeTargetPredictionConfig(),
) -> dict[str, Any]:
    """Roll out and seal both deployable methods without target-future outcomes."""

    validate_contact_artifact(
        contact_prediction_seal,
        expected_kind="Deform360TargetContactPredictionSeal",
    )
    fit_validation = validate_forward_rope_fit_artifact(shared_fit)
    _require(
        fit_validation["source_competence_passed"] is True,
        "shared dynamics failed the frozen source competence gate",
    )
    validate_target_prefix_rope_geometry(prefix_geometry, verify_archive=True)
    _require(
        contact_prediction_seal["protocol_id"]
        == shared_fit["protocol_id"]
        == prefix_geometry["protocol_id"]
        == protocol.protocol_id,
        "target rollout artifacts belong to different protocols",
    )
    target_index = protocol.target_episode_ids[0]
    episode_dir = Path(processed_root).resolve() / f"episode_{target_index:04d}"
    robot_path = episode_dir / "robot" / "robot.npz"
    _require(robot_path.is_file(), "target robot trajectory is missing")
    _require(
        _sha256_file(robot_path)
        == contact_prediction_seal["inputs"]["target_robot_sha256"],
        "target robot trajectory differs from the contact seal",
    )
    try:
        from deform360.processing.control_points_stage import gripper_taxel_points
        from deform360.robot import load_robot_state
    except ImportError as error:  # pragma: no cover - pinned host integration
        raise RuntimeError(
            "the pinned Deform360 processing environment is required"
        ) from error
    state = load_robot_state(robot_path)
    _require(state.bimanual and state.num_grippers == 2, "target must be bimanual")
    with np.load(prefix_geometry["archive"]["path"], allow_pickle=False) as stored:
        prefix_frames = np.asarray(stored["frame_indices"], dtype=np.int32)
        prefix_centerlines = np.asarray(stored["centerlines_m"], dtype=np.float64)
    prefix_end = int(prefix_frames[-1])
    future_start = prefix_end + 1
    _require(
        future_start
        == int(contact_prediction_seal["target_prefix"]["stop_frame_exclusive"]),
        "prefix geometry and contact seal end at different frames",
    )
    _require(future_start < len(state.actions), "target prefix has no future")
    initial = prefix_centerlines[-1]
    controller_frame_indices = np.arange(prefix_end, len(state.actions), dtype=np.int32)
    selected_taxels = []
    contact_nodes = []
    offsets = []
    associations = []
    for axis in range(state.num_grippers):
        taxels = gripper_taxel_points(
            float(state.openings[prefix_end, axis]),
            state.T_worlds[prefix_end, axis],
        )
        selected, _, node, offset, diagnostics = select_visual_contact_patch(
            taxels,
            initial,
            taxel_count=config.visual_patch_taxel_count,
        )
        _require(
            diagnostics["patch_to_node_distance_m"]
            <= config.maximum_patch_to_centerline_m,
            "target visual contact patch is too far from the rope",
        )
        selected_taxels.append(selected)
        contact_nodes.append(node)
        offsets.append(offset)
        associations.append(
            {
                "robot_axis": axis,
                "selected_taxel_indices": selected.astype(int).tolist(),
                "contact_node_index": node,
                "contact_offset_m": offset.tolist(),
                **diagnostics,
            }
        )
    _require(len(set(contact_nodes)) == 2, "target grippers map to one rope node")
    controllers = np.empty(
        (len(controller_frame_indices), state.num_grippers, 3), dtype=np.float64
    )
    for output_index, frame_index in enumerate(controller_frame_indices):
        for axis in range(state.num_grippers):
            taxels = gripper_taxel_points(
                float(state.openings[frame_index, axis]),
                state.T_worlds[frame_index, axis],
            )
            controllers[output_index, axis] = np.mean(
                taxels[selected_taxels[axis]], axis=0
            )
    visual_full = np.asarray(
        contact_prediction_seal["visual_only"]["state_by_robot_axis"], dtype=bool
    ).T
    _require(
        visual_full.shape == (len(state.actions), state.num_grippers),
        "visual contact schedule and target robot trajectory disagree",
    )
    visual = visual_full[prefix_end:]
    tactile_prefix_state = np.zeros(state.num_grippers, dtype=bool)
    for record in contact_prediction_seal["target_prefix"]["tactile_conditioned_z"]:
        tactile_prefix_state[int(record["robot_axis"])] = bool(
            record["contact_at_prefix_end"]
        )
    tactile_conditioned = propagate_prefix_contact_state(visual, tactile_prefix_state)
    parameters = load_forward_rope_fit_parameters(shared_fit)
    fit_config = shared_fit["config"]
    rest_lengths = np.linalg.norm(np.diff(initial, axis=0), axis=1)
    common = {
        "initial_positions_m": initial,
        "initial_velocities_m_s": np.zeros_like(initial),
        "controller_positions_m": controllers,
        "contact_node_indices": tuple(contact_nodes),
        "contact_offsets_m": np.asarray(offsets, dtype=np.float64),
        "rest_lengths_m": rest_lengths,
        "parameters": parameters,
        "dt_seconds": protocol.nominal_frame_interval_us / 1_000_000.0,
        "gravity_m_s2": np.zeros(3),
        "substeps": int(fit_config["substeps"]),
        "constraint_iterations": int(fit_config["constraint_iterations"]),
    }
    visual_prediction = rollout_rope_dynamics(contact_active=visual, **common)[1:]
    tactile_prediction = rollout_rope_dynamics(
        contact_active=tactile_conditioned, **common
    )[1:]
    rollout_configuration = {
        "parameters": asdict(config),
        "target_robot_sha256": _sha256_file(robot_path),
        "controller_frame_indices": controller_frame_indices.astype(int).tolist(),
        "controller_positions_sha256": _sha256_array(controllers),
        "contact_associations": associations,
        "visual_contact_schedule_sha256": _sha256_array(visual),
        "tactile_conditioned_contact_schedule_sha256": _sha256_array(
            tactile_conditioned
        ),
        "tactile_prefix_end_state": tactile_prefix_state.tolist(),
        "tactile_schedule_policy": (
            "hold the six-frame prefix posterior until the source-calibrated "
            "visual schedule exhibits a transition"
        ),
        "initial_velocity_policy": shared_fit["initial_velocity_policy"],
        "effective_gravity_m_s2": shared_fit["effective_gravity_m_s2"],
        "shared_parameters": parameters.as_dict(),
        "substeps": int(fit_config["substeps"]),
        "constraint_iterations": int(fit_config["constraint_iterations"]),
        "information_boundary": {
            "target_robot_trajectory_read": True,
            "target_visual_prefix_read": True,
            "target_tactile_prefix_read": True,
            "target_future_geometry_read": False,
            "target_tactile_oracle_read": False,
        },
    }
    return seal_held_out_rope_predictions(
        prediction_archive_path,
        {
            "visual_only": visual_prediction,
            "tactile_conditioned_z": tactile_prediction,
        },
        protocol_id=protocol.protocol_id,
        contact_prediction_seal=contact_prediction_seal,
        shared_dynamics_fit_sha256=shared_fit["result_sha256"],
        target_prefix_geometry_sha256=prefix_geometry["result_sha256"],
        future_start_frame=future_start,
        rollout_configuration=rollout_configuration,
    )


def write_target_rope_prediction_seal(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    validate_held_out_rope_prediction_seal(payload, verify_archive=True)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def build_target_oracle_tactile_rope_prediction(
    processed_root: str | Path,
    protocol: Deform360ProtocolConfig,
    held_out_prediction_seal: Mapping[str, Any],
    target_contact_oracle: Mapping[str, Any],
    shared_fit: Mapping[str, Any],
    prefix_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Roll out the offline full-tactile contact upper bound after sealing."""

    validate_held_out_rope_prediction_seal(held_out_prediction_seal)
    validate_contact_artifact(
        target_contact_oracle,
        expected_kind="Deform360TargetContactOracleEvaluation",
    )
    _require(
        target_contact_oracle["held_out_prediction_seal_sha256"]
        == held_out_prediction_seal["result_sha256"],
        "contact oracle and held-out prediction seal differ",
    )
    _require(
        held_out_prediction_seal["shared_dynamics_fit_sha256"]
        == shared_fit["result_sha256"],
        "oracle rollout uses another shared fit",
    )
    _require(
        held_out_prediction_seal["target_prefix_geometry_sha256"]
        == prefix_geometry["result_sha256"],
        "oracle rollout uses another target-prefix geometry",
    )
    validate_forward_rope_fit_artifact(shared_fit)
    validate_target_prefix_rope_geometry(prefix_geometry)
    target_index = protocol.target_episode_ids[0]
    episode_dir = Path(processed_root).resolve() / f"episode_{target_index:04d}"
    robot_path = episode_dir / "robot" / "robot.npz"
    try:
        from deform360.processing.control_points_stage import gripper_taxel_points
        from deform360.robot import load_robot_state
    except ImportError as error:  # pragma: no cover - pinned host integration
        raise RuntimeError(
            "the pinned Deform360 processing environment is required"
        ) from error
    state = load_robot_state(robot_path)
    with np.load(prefix_geometry["archive"]["path"], allow_pickle=False) as stored:
        prefix_frames = np.asarray(stored["frame_indices"], dtype=np.int32)
        initial = np.asarray(stored["centerlines_m"][-1], dtype=np.float64)
    prefix_end = int(prefix_frames[-1])
    configuration = held_out_prediction_seal["rollout_configuration"]
    controller_frame_indices = np.asarray(
        configuration["controller_frame_indices"], dtype=np.int32
    )
    _require(
        int(controller_frame_indices[0]) == prefix_end,
        "oracle controller trajectory does not start at the prefix endpoint",
    )
    associations = sorted(
        configuration["contact_associations"], key=lambda row: row["robot_axis"]
    )
    controllers = np.empty(
        (len(controller_frame_indices), state.num_grippers, 3), dtype=np.float64
    )
    for output_index, frame_index in enumerate(controller_frame_indices):
        for association in associations:
            axis = int(association["robot_axis"])
            taxels = gripper_taxel_points(
                float(state.openings[frame_index, axis]),
                state.T_worlds[frame_index, axis],
            )
            controllers[output_index, axis] = np.mean(
                taxels[np.asarray(association["selected_taxel_indices"], dtype=int)],
                axis=0,
            )
    _require(
        _sha256_array(controllers) == configuration["controller_positions_sha256"],
        "reconstructed oracle controller trajectory differs from the seal",
    )
    oracle_full = np.asarray(
        target_contact_oracle["contact_state_by_robot_axis"]["oracle_tactile"],
        dtype=bool,
    ).T
    _require(
        oracle_full.shape == (len(state.actions), state.num_grippers),
        "oracle contact schedule and target robot trajectory disagree",
    )
    oracle_schedule = oracle_full[prefix_end:]
    parameters = load_forward_rope_fit_parameters(shared_fit)
    fit_config = shared_fit["config"]
    trajectory = rollout_rope_dynamics(
        initial,
        np.zeros_like(initial),
        controllers,
        oracle_schedule,
        tuple(int(row["contact_node_index"]) for row in associations),
        np.asarray([row["contact_offset_m"] for row in associations]),
        np.linalg.norm(np.diff(initial, axis=0), axis=1),
        parameters,
        dt_seconds=protocol.nominal_frame_interval_us / 1_000_000.0,
        gravity_m_s2=np.zeros(3),
        substeps=int(fit_config["substeps"]),
        constraint_iterations=int(fit_config["constraint_iterations"]),
    )[1:]
    return build_oracle_tactile_rope_prediction(
        trajectory,
        held_out_prediction_seal=held_out_prediction_seal,
        target_contact_oracle=target_contact_oracle,
        contact_schedule_sha256=_sha256_array(oracle_schedule),
    )


def write_target_oracle_tactile_rope_prediction(
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
    "RopeTargetPredictionConfig",
    "build_and_seal_target_rope_predictions",
    "build_target_oracle_tactile_rope_prediction",
    "propagate_prefix_contact_state",
    "select_visual_contact_patch",
    "write_target_rope_prediction_seal",
    "write_target_oracle_tactile_rope_prediction",
]
