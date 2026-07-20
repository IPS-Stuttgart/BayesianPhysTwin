from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from bayesian_phystwin.deform360_frame_zero_assets import (
    FrameZeroAssetConfig,
    artifact_sha256,
)
from bayesian_phystwin.deform360_held_outcome_scoring import (
    OUTCOME_RECONSTRUCTION_CONTRACT,
)
from bayesian_phystwin.deform360_held_physical_prior import (
    HELD_PHYSICAL_NUMERIC_CONTRACT,
    UPSTREAM_FILE_SHA256,
    UPSTREAM_LOCK_BINDING_BY_PATH,
    UPSTREAM_RUNTIME_BUNDLE_CONTRACT,
)
from bayesian_phystwin.deform360_held_protocol import (
    REQUIRED_IMMUTABLE_BINDING_KEYS,
    SOURCE_FEASIBILITY_AMENDMENT_CONTRACT,
    V6_OUTCOME_WITHDRAWAL_REPORT_FILE_SHA256,
    held_contract_sha256,
)
from bayesian_phystwin.deform360_held_gsplat_runtime import (
    GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256,
)
from bayesian_phystwin.deform360_robot_kinematics import (
    ROBOT_KINEMATICS_WINDOW_CONTRACT,
    ROBOT_KINEMATICS_WINDOW_POLICY_ID,
    load_robot_kinematics_archive,
    robot_kinematics_array_records,
    select_robot_kinematics_window,
    slice_robot_kinematics,
    validate_selected_robot_kinematics_bundle,
)


def dummy_immutable_bindings() -> dict[str, str]:
    """Return a complete deterministic test-only held binding set."""

    bindings = {
        key: hashlib.sha256(f"test-only:{key}".encode()).hexdigest()
        for key in REQUIRED_IMMUTABLE_BINDING_KEYS
    }
    bindings["frame_zero_default_config"] = artifact_sha256(
        asdict(FrameZeroAssetConfig())
    )
    bindings["outcome_reconstruction_contract"] = held_contract_sha256(
        OUTCOME_RECONSTRUCTION_CONTRACT
    )
    bindings["held_physical_numeric_contract"] = held_contract_sha256(
        HELD_PHYSICAL_NUMERIC_CONTRACT
    )
    bindings["held_source_feasibility_amendment_contract"] = held_contract_sha256(
        SOURCE_FEASIBILITY_AMENDMENT_CONTRACT
    )
    bindings["held_outcome_cuda_smoke_contract"] = GSPLAT_RUNTIME_SMOKE_CONTRACT_SHA256
    bindings["v6_outcome_withdrawal_report"] = V6_OUTCOME_WITHDRAWAL_REPORT_FILE_SHA256
    bindings["upstream_runtime_bundle_tree"] = held_contract_sha256(
        UPSTREAM_RUNTIME_BUNDLE_CONTRACT
    )
    for path, binding_key in UPSTREAM_LOCK_BINDING_BY_PATH.items():
        bindings[binding_key] = UPSTREAM_FILE_SHA256[path]
    return bindings


def default_frame_zero_config() -> dict[str, object]:
    return asdict(FrameZeroAssetConfig())


def bound_file(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def write_robot_kinematics_fixture(
    directory: Path,
    *,
    source_frame_count: int = 150,
    selected_start: int = 62,
    window_length_frames: int = 81,
    prediction_frame_count: int = 76,
    candidate_first_frame: int = 8,
    candidate_stride_frames: int = 6,
) -> tuple[Path, Path, dict[str, object]]:
    """Write a strict monomanual archive with a uniquely selected window."""

    directory.mkdir(parents=True, exist_ok=True)
    transforms = np.repeat(
        np.eye(4, dtype=np.float64)[None], source_frame_count, axis=0
    )
    # Motion exists on every transition inside the requested 81-frame window.
    for frame in range(selected_start + 1, selected_start + window_length_frames):
        transforms[frame:, 0, 3] += 0.001
    openings = np.zeros(source_frame_count, dtype=np.float64)
    actions = np.zeros((source_frame_count, 5, 3), dtype=np.float64)
    actions[:, 0, :] = transforms[:, :3, 3]
    actions[:, 1:4, :] = transforms[:, :3, :3]
    actions[:, 4, 0] = openings
    raw_path = directory / "robot.npz"
    np.savez_compressed(
        raw_path,
        format_version=np.asarray(1, dtype=np.uint16),
        actions=actions,
        T_worlds=transforms,
        openings=openings,
        bimanual=np.asarray(False, dtype=np.bool_),
    )
    source_state = load_robot_kinematics_archive(raw_path)
    selection = select_robot_kinematics_window(
        source_state,
        window_length_frames=window_length_frames,
        prediction_frame_count=prediction_frame_count,
        candidate_first_frame=candidate_first_frame,
        candidate_stride_frames=candidate_stride_frames,
    )
    assert selection["selected_raw_frame_range_half_open"][0] == selected_start
    selected_state = slice_robot_kinematics(
        source_state,
        start_frame=selected_start,
        frame_count=prediction_frame_count,
    )
    selected_path = directory / "known_action_76.npz"
    np.savez_compressed(selected_path, **selected_state.archive_arrays())
    exact_slice = validate_selected_robot_kinematics_bundle(
        selected_path,
        source_state=source_state,
        prediction_start_frame=selected_start,
        prediction_frame_count=prediction_frame_count,
    )
    selected_record = bound_file(selected_path)
    alignment: dict[str, object] = {
        "policy_id": ROBOT_KINEMATICS_WINDOW_POLICY_ID,
        "trajectory_semantics": ROBOT_KINEMATICS_WINDOW_CONTRACT[
            "trajectory_semantics"
        ],
        "selection_audit": selection,
        "selected_raw_frame_range_half_open": list(
            selection["selected_raw_frame_range_half_open"]
        ),
        "prediction_raw_frame_range_half_open": list(
            selection["prediction_raw_frame_range_half_open"]
        ),
        "tracking_tail_frame_count": window_length_frames - prediction_frame_count,
        "source_robot_frame_count": source_frame_count,
        "prediction_frame_count": prediction_frame_count,
        "selected_robot_kinematics_bundle": selected_record,
        "selected_action_bundle": selected_record,
        "selected_action_bundle_is_compatibility_alias": True,
        "selected_action_arrays": robot_kinematics_array_records(selected_state),
        "selected_bundle_exact_slice_audit": exact_slice,
    }
    return raw_path, selected_path, alignment


def write_robot_metadata_fixture(
    path: Path,
    *,
    source_frame_count: int,
    cameras: tuple[str, ...],
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "deform360.processing/robot/v1",
                "outputs": {"num_frames": source_frame_count},
                "inputs": {
                    "aligned_timestamps_sha256": "b" * 64,
                    "video_sha256": {camera: "c" * 64 for camera in cameras},
                },
                "parameters": {"cameras": list(cameras)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path
