"""Strict Deform360 robot kinematics and action-window selection.

The public Deform360 ``robot.npz`` contract stores the same absolute gripper
state in two representations.  ``T_worlds`` is the end-effector-to-world rigid
transform, while ``actions`` encodes its translation in row zero, its rotation
in rows one through three, and ``[opening_m, 0, 0]`` in row four.  These rows
are heterogeneous fields; they must never be averaged as if they were points.

This module is intentionally independent of frame-zero geometry, the physical
prior, and held-out protocol code.  Those call sites can therefore share one
validated, deterministic robot-only window contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROBOT_STATE_FORMAT_VERSION = 1
ROBOT_STATE_ARCHIVE_FIELDS = frozenset(
    {"format_version", "actions", "T_worlds", "openings", "bimanual"}
)

ROBOT_KINEMATICS_WINDOW_POLICY_ID = "deform360-tworld-eef-translation-closed-path-v1"

_ACTION_PARITY_RTOL = 1e-6
_ACTION_PARITY_ATOL = 1e-8
_HOMOGENEOUS_ROW_ATOL = 1e-7
_ROTATION_ATOL = 1e-5
_CONSTANT_OPENING_SPAN_M = 1e-9
_OPENING_LOW_QUANTILE = 0.10
_OPENING_HIGH_QUANTILE = 0.90
_QUANTILE_METHOD = "linear"

ROBOT_KINEMATICS_WINDOW_CONTRACT: dict[str, Any] = {
    "contract_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
    "robot_state_format_version": ROBOT_STATE_FORMAT_VERSION,
    "trajectory_semantics": (
        "aligned absolute end-effector pose/opening annotation in the "
        "Deform360 world frame; not a delta command"
    ),
    "translation_units": "metres",
    "opening_units": "metres",
    "selection_inputs": [
        "robot.npz:T_worlds[..., :3, 3]",
        "robot.npz:openings",
        "robot.npz:bimanual",
    ],
    "canonical_translation_shapes": {
        "monomanual": "(T,1,3)",
        "bimanual": "(T,2,3)",
    },
    "closure_confidence": {
        "low_quantile": _OPENING_LOW_QUANTILE,
        "high_quantile": _OPENING_HIGH_QUANTILE,
        "quantile_axis": "time independently per gripper",
        "quantile_method": _QUANTILE_METHOD,
        "constant_span_threshold_m": _CONSTANT_OPENING_SPAN_M,
        "varying_formula": "clip((q90-opening)/(q90-q10),0,1)",
        "constant_formula": "one",
    },
    "candidate_score": (
        "mean over grippers of the sum over adjacent steps of Euclidean "
        "T_worlds translation displacement times the minimum endpoint "
        "closure confidence"
    ),
    "candidate_score_units": "metres",
    "tie_break": "earliest candidate start",
    "calculation_dtype": "float64",
    "default_candidate_first_frame": 8,
    "default_candidate_stride_frames": 6,
    "default_window_length_frames": 81,
    "default_prediction_frame_count": 76,
    "action_redundancy_tolerance": {
        "rtol": _ACTION_PARITY_RTOL,
        "atol": _ACTION_PARITY_ATOL,
    },
}


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


def artifact_sha256(value: Mapping[str, Any]) -> str:
    canonical = dict(value)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return hashlib.sha256(
        descriptor + b"\0" + array.view(np.uint8).tobytes()
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256 = hashlib.sha256(
    _canonical_bytes(ROBOT_KINEMATICS_WINDOW_CONTRACT)
).hexdigest()


@dataclass(frozen=True)
class Deform360RobotKinematics:
    """Validated, pickle-free Deform360 robot-state arrays."""

    actions: np.ndarray
    T_worlds: np.ndarray
    openings: np.ndarray
    bimanual: bool

    @property
    def frame_count(self) -> int:
        return int(len(self.actions))

    @property
    def gripper_count(self) -> int:
        return 2 if self.bimanual else 1

    @property
    def eef_translations_world_m(self) -> np.ndarray:
        if self.bimanual:
            return np.asarray(self.T_worlds[:, :, :3, 3], dtype=np.float64)
        return np.asarray(self.T_worlds[:, None, :3, 3], dtype=np.float64)

    @property
    def canonical_openings_m(self) -> np.ndarray:
        if self.bimanual:
            return np.asarray(self.openings, dtype=np.float64)
        return np.asarray(self.openings[:, None], dtype=np.float64)

    def archive_arrays(self) -> dict[str, np.ndarray]:
        """Return exact-format copies suitable for a safe selected NPZ."""

        return {
            "format_version": np.asarray(ROBOT_STATE_FORMAT_VERSION, dtype=np.uint16),
            "actions": np.array(self.actions, copy=True),
            "T_worlds": np.array(self.T_worlds, copy=True),
            "openings": np.array(self.openings, copy=True),
            "bimanual": np.asarray(self.bimanual, dtype=np.bool_),
        }


def _immutable_float64_array(value: object, *, name: str) -> np.ndarray:
    source = np.asarray(value)
    _require(source.dtype == np.dtype(np.float64), f"{name} must have dtype float64")
    _require(np.all(np.isfinite(source)), f"{name} contains non-finite values")
    result = np.array(source, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def validate_robot_kinematics_arrays(
    *,
    format_version: object,
    actions: object,
    T_worlds: object,
    openings: object,
    bimanual: object,
) -> Deform360RobotKinematics:
    """Validate the exact public Deform360 ``robot.npz`` array contract."""

    version = np.asarray(format_version)
    _require(
        version.shape == () and version.dtype == np.dtype(np.uint16),
        "format_version must be a uint16 scalar",
    )
    _require(
        int(version.item()) == ROBOT_STATE_FORMAT_VERSION,
        "unsupported robot state format version",
    )
    bimanual_array = np.asarray(bimanual)
    _require(
        bimanual_array.shape == () and bimanual_array.dtype == np.dtype(np.bool_),
        "bimanual must be a bool scalar",
    )
    is_bimanual = bool(bimanual_array.item())

    action_array = _immutable_float64_array(actions, name="actions")
    transform_array = _immutable_float64_array(T_worlds, name="T_worlds")
    opening_array = _immutable_float64_array(openings, name="openings")

    if is_bimanual:
        _require(
            action_array.ndim == 4 and action_array.shape[1:] == (2, 5, 3),
            "bimanual actions must have shape (T,2,5,3)",
        )
        expected_transforms = (len(action_array), 2, 4, 4)
        expected_openings = (len(action_array), 2)
    else:
        _require(
            action_array.ndim == 3 and action_array.shape[1:] == (5, 3),
            "monomanual actions must have shape (T,5,3)",
        )
        expected_transforms = (len(action_array), 4, 4)
        expected_openings = (len(action_array),)
    _require(len(action_array) > 0, "robot state has no frames")
    _require(
        transform_array.shape == expected_transforms,
        "T_worlds shape does not match the robot mode",
    )
    _require(
        opening_array.shape == expected_openings,
        "openings shape does not match the robot mode",
    )
    _require(np.all(opening_array >= 0.0), "openings must be non-negative metres")

    flat_transforms = transform_array.reshape(-1, 4, 4)
    _require(
        np.allclose(
            flat_transforms[:, 3, :],
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=_HOMOGENEOUS_ROW_ATOL,
        ),
        "T_worlds has an invalid homogeneous bottom row",
    )
    rotations = flat_transforms[:, :3, :3]
    gram = np.swapaxes(rotations, 1, 2) @ rotations
    _require(
        np.allclose(
            gram,
            np.eye(3, dtype=np.float64),
            rtol=0.0,
            atol=_ROTATION_ATOL,
        ),
        "T_worlds rotation is not orthonormal",
    )
    _require(
        np.allclose(
            np.linalg.det(rotations),
            1.0,
            rtol=0.0,
            atol=_ROTATION_ATOL,
        ),
        "T_worlds rotation must have determinant +1",
    )

    _require(
        np.allclose(
            action_array[..., 0, :],
            transform_array[..., :3, 3],
            rtol=_ACTION_PARITY_RTOL,
            atol=_ACTION_PARITY_ATOL,
        ),
        "actions row 0 does not match T_worlds translation",
    )
    _require(
        np.allclose(
            action_array[..., 1:4, :],
            transform_array[..., :3, :3],
            rtol=_ACTION_PARITY_RTOL,
            atol=_ACTION_PARITY_ATOL,
        ),
        "actions rows 1:4 do not match T_worlds rotation",
    )
    expected_opening_rows = np.zeros_like(action_array[..., 4, :])
    expected_opening_rows[..., 0] = opening_array
    _require(
        np.allclose(
            action_array[..., 4, :],
            expected_opening_rows,
            rtol=_ACTION_PARITY_RTOL,
            atol=_ACTION_PARITY_ATOL,
        ),
        "actions row 4 does not match openings",
    )
    return Deform360RobotKinematics(
        actions=action_array,
        T_worlds=transform_array,
        openings=opening_array,
        bimanual=is_bimanual,
    )


def load_robot_kinematics_archive(
    path: str | Path,
    *,
    expected_frame_count: int | None = None,
) -> Deform360RobotKinematics:
    """Load and strictly validate an exact-field, pickle-free robot archive."""

    source = Path(path)
    _require(source.suffix.lower() == ".npz", "robot archive must use .npz")
    _require(not source.is_symlink(), "robot archive must not be a symlink")
    resolved = source.resolve()
    _require(resolved.is_file(), "robot archive is missing")
    with np.load(resolved, allow_pickle=False) as stored:
        _require(
            set(stored.files) == set(ROBOT_STATE_ARCHIVE_FIELDS),
            "robot archive field set changed",
        )
        state = validate_robot_kinematics_arrays(
            format_version=stored["format_version"],
            actions=stored["actions"],
            T_worlds=stored["T_worlds"],
            openings=stored["openings"],
            bimanual=stored["bimanual"],
        )
    if expected_frame_count is not None:
        _require(
            state.frame_count == expected_frame_count,
            "robot archive frame count changed",
        )
    return state


def robot_kinematics_array_records(
    state: Deform360RobotKinematics,
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": sha256_array(value),
        }
        for name, value in sorted(state.archive_arrays().items())
    }


def _closure_confidence(openings_m: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    openings = np.asarray(openings_m, dtype=np.float64)
    _require(
        openings.ndim == 2 and openings.shape[1] in (1, 2),
        "canonical openings must have shape (T,1) or (T,2)",
    )
    low = np.quantile(
        openings,
        _OPENING_LOW_QUANTILE,
        axis=0,
        method=_QUANTILE_METHOD,
    )
    high = np.quantile(
        openings,
        _OPENING_HIGH_QUANTILE,
        axis=0,
        method=_QUANTILE_METHOD,
    )
    span = high - low
    varying = span > _CONSTANT_OPENING_SPAN_M
    confidence = np.ones_like(openings, dtype=np.float64)
    confidence[:, varying] = np.clip(
        (high[varying] - openings[:, varying]) / span[varying],
        0.0,
        1.0,
    )
    return confidence, {
        "low_quantile": _OPENING_LOW_QUANTILE,
        "high_quantile": _OPENING_HIGH_QUANTILE,
        "quantile_method": _QUANTILE_METHOD,
        "constant_span_threshold_m": _CONSTANT_OPENING_SPAN_M,
        "q10_opening_m": low.tolist(),
        "q90_opening_m": high.tolist(),
        "q90_minus_q10_m": span.tolist(),
        "varying_opening_by_gripper": varying.tolist(),
        "confidence_sha256": sha256_array(confidence),
    }


def select_robot_kinematics_window(
    state: Deform360RobotKinematics,
    *,
    window_length_frames: int = 81,
    prediction_frame_count: int = 76,
    candidate_first_frame: int = 8,
    candidate_stride_frames: int = 6,
) -> dict[str, Any]:
    """Select the deterministic closed-weighted EEF translation window."""

    _require(isinstance(state, Deform360RobotKinematics), "robot state is invalid")
    _require(
        2 <= prediction_frame_count < window_length_frames <= state.frame_count,
        "robot trajectory is shorter than the requested window",
    )
    _require(candidate_first_frame >= 0, "candidate first frame is negative")
    _require(candidate_stride_frames >= 1, "candidate stride must be positive")

    starts = np.arange(
        candidate_first_frame,
        state.frame_count - window_length_frames + 1,
        candidate_stride_frames,
        dtype=np.int64,
    )
    _require(bool(len(starts)), "robot trajectory has no complete candidate")
    positions = state.eef_translations_world_m
    openings = state.canonical_openings_m
    confidence, closure_audit = _closure_confidence(openings)

    candidates: list[dict[str, Any]] = []
    for candidate_index, start_value in enumerate(starts):
        start = int(start_value)
        stop = start + window_length_frames
        displacement_m = np.linalg.norm(np.diff(positions[start:stop], axis=0), axis=-1)
        endpoint_confidence = np.minimum(
            confidence[start : stop - 1], confidence[start + 1 : stop]
        )
        per_gripper_path_m = np.sum(
            displacement_m * endpoint_confidence,
            axis=0,
            dtype=np.float64,
        )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "frame_range_half_open": [start, stop],
                "prediction_frame_range_half_open": [
                    start,
                    start + prediction_frame_count,
                ],
                "per_gripper_closed_weighted_translation_path_length_m": (
                    per_gripper_path_m.tolist()
                ),
                "mean_closed_weighted_translation_path_length_m": float(
                    np.mean(per_gripper_path_m, dtype=np.float64)
                ),
                "maximum_closed_weighted_translation_path_length_m": float(
                    np.max(per_gripper_path_m)
                ),
            }
        )

    selected_index = min(
        range(len(candidates)),
        key=lambda index: (
            -float(candidates[index]["mean_closed_weighted_translation_path_length_m"]),
            int(candidates[index]["frame_range_half_open"][0]),
        ),
    )
    selected = candidates[selected_index]
    start, stop = selected["frame_range_half_open"]
    audit: dict[str, Any] = {
        "policy_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
        "contract_sha256": ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
        "trajectory_semantics": ROBOT_KINEMATICS_WINDOW_CONTRACT[
            "trajectory_semantics"
        ],
        "selection_rule": (
            "maximize mean per-gripper closed-weighted T_worlds translation "
            "path; earliest candidate breaks exact ties"
        ),
        "selection_inputs": list(ROBOT_KINEMATICS_WINDOW_CONTRACT["selection_inputs"]),
        "source_robot_frame_count": state.frame_count,
        "bimanual": state.bimanual,
        "gripper_count": state.gripper_count,
        "candidate_first_frame": candidate_first_frame,
        "candidate_stride_frames": candidate_stride_frames,
        "window_length_frames": window_length_frames,
        "prediction_frame_count": prediction_frame_count,
        "tracking_tail_frame_count": window_length_frames - prediction_frame_count,
        "input_array_sha256": {
            name: record["sha256"]
            for name, record in robot_kinematics_array_records(state).items()
        },
        "eef_translations_world_m_sha256": sha256_array(positions),
        "canonical_openings_m_sha256": sha256_array(openings),
        "closure_confidence": closure_audit,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_records_sha256": hashlib.sha256(
            _canonical_bytes(candidates)
        ).hexdigest(),
        "selected_candidate_index": selected_index,
        "selected_raw_frame_range_half_open": [int(start), int(stop)],
        "prediction_raw_frame_range_half_open": [
            int(start),
            int(start) + prediction_frame_count,
        ],
        "selected_score": dict(selected),
        "object_geometry_used_for_selection": False,
        "object_rgb_used_for_selection": False,
        "tactile_used_for_selection": False,
        "outcome_used_for_selection": False,
    }
    audit["artifact_sha256"] = artifact_sha256(audit)
    return audit


def validate_robot_kinematics_selection_audit(
    audit: Mapping[str, Any],
    state: Deform360RobotKinematics,
    *,
    window_length_frames: int = 81,
    prediction_frame_count: int = 76,
    candidate_first_frame: int = 8,
    candidate_stride_frames: int = 6,
) -> dict[str, Any]:
    """Recompute and require an exact selection audit."""

    expected = select_robot_kinematics_window(
        state,
        window_length_frames=window_length_frames,
        prediction_frame_count=prediction_frame_count,
        candidate_first_frame=candidate_first_frame,
        candidate_stride_frames=candidate_stride_frames,
    )
    _require(
        _canonical_bytes(dict(audit)) == _canonical_bytes(expected),
        "robot kinematics selection audit changed",
    )
    return expected


def slice_robot_kinematics(
    state: Deform360RobotKinematics,
    *,
    start_frame: int,
    frame_count: int,
) -> Deform360RobotKinematics:
    """Return a validated exact-format temporal slice of a robot state."""

    _require(isinstance(start_frame, int) and start_frame >= 0, "invalid slice start")
    _require(isinstance(frame_count, int) and frame_count >= 1, "invalid slice length")
    stop = start_frame + frame_count
    _require(stop <= state.frame_count, "robot slice exceeds the source trajectory")
    return validate_robot_kinematics_arrays(
        format_version=np.asarray(ROBOT_STATE_FORMAT_VERSION, dtype=np.uint16),
        actions=state.actions[start_frame:stop],
        T_worlds=state.T_worlds[start_frame:stop],
        openings=state.openings[start_frame:stop],
        bimanual=np.asarray(state.bimanual, dtype=np.bool_),
    )


def validate_selected_robot_kinematics_bundle(
    selected: str | Path | Deform360RobotKinematics,
    *,
    source_state: Deform360RobotKinematics,
    prediction_start_frame: int,
    prediction_frame_count: int = 76,
) -> dict[str, Any]:
    """Validate that a selected bundle is the exact source prediction slice."""

    _require(
        isinstance(source_state, Deform360RobotKinematics),
        "source robot state is invalid",
    )
    selected_state = (
        selected
        if isinstance(selected, Deform360RobotKinematics)
        else load_robot_kinematics_archive(
            selected, expected_frame_count=prediction_frame_count
        )
    )
    expected = slice_robot_kinematics(
        source_state,
        start_frame=prediction_start_frame,
        frame_count=prediction_frame_count,
    )
    expected_arrays = expected.archive_arrays()
    selected_arrays = selected_state.archive_arrays()
    _require(
        set(expected_arrays) == set(selected_arrays) == set(ROBOT_STATE_ARCHIVE_FIELDS),
        "selected robot bundle field set changed",
    )
    for name in sorted(expected_arrays):
        observed = selected_arrays[name]
        wanted = expected_arrays[name]
        _require(
            observed.dtype == wanted.dtype
            and observed.shape == wanted.shape
            and np.array_equal(observed, wanted),
            f"selected robot bundle is not the exact source slice: {name}",
        )
    audit: dict[str, Any] = {
        "policy_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
        "contract_sha256": ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256,
        "source_frame_count": source_state.frame_count,
        "prediction_raw_frame_range_half_open": [
            prediction_start_frame,
            prediction_start_frame + prediction_frame_count,
        ],
        "prediction_frame_count": prediction_frame_count,
        "bimanual": selected_state.bimanual,
        "gripper_count": selected_state.gripper_count,
        "exact_source_slice": True,
        "source_array_sha256": {
            name: sha256_array(value)
            for name, value in sorted(source_state.archive_arrays().items())
        },
        "selected_array_sha256": {
            name: sha256_array(value) for name, value in sorted(selected_arrays.items())
        },
    }
    audit["artifact_sha256"] = artifact_sha256(audit)
    return audit


__all__ = [
    "Deform360RobotKinematics",
    "ROBOT_KINEMATICS_WINDOW_CONTRACT",
    "ROBOT_KINEMATICS_WINDOW_CONTRACT_SHA256",
    "ROBOT_KINEMATICS_WINDOW_POLICY_ID",
    "ROBOT_STATE_ARCHIVE_FIELDS",
    "ROBOT_STATE_FORMAT_VERSION",
    "artifact_sha256",
    "load_robot_kinematics_archive",
    "robot_kinematics_array_records",
    "select_robot_kinematics_window",
    "sha256_array",
    "sha256_file",
    "slice_robot_kinematics",
    "validate_robot_kinematics_arrays",
    "validate_robot_kinematics_selection_audit",
    "validate_selected_robot_kinematics_bundle",
]
