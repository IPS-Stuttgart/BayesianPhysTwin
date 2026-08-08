from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.causal4d_artifacts_v2 import ReleasedPhysTwinVisualInputsV2
from bayesian_phystwin.dynamic_discrepancy import DynamicDiscrepancyCorrection


def _dynamic_correction() -> DynamicDiscrepancyCorrection:
    values = np.column_stack(
        (
            np.ones(8),
            np.linspace(-1.0, 1.0, 8),
            np.cos(np.linspace(0.0, np.pi, 8)),
            np.sin(np.linspace(0.0, 2.0 * np.pi, 8)),
        )
    )
    basis = np.linalg.qr(values, mode="reduced")[0]
    zeros = np.zeros((4, 3), dtype=np.float64)
    return DynamicDiscrepancyCorrection(
        case_id="immutable-test",
        graph_basis=basis,
        graph_eigenvalues=np.asarray([0.0, 0.1, 0.2, 0.3]),
        position_coefficients_m=zeros,
        velocity_coefficients_mps=zeros,
        generalized_force_coefficients_n=zeros,
        structural_coefficients_m=zeros,
        prefix_frame_start=10,
        prefix_frame_stop=17,
        frame_dt_s=0.05,
        information_boundary={
            "o_plus_prefix_frames": 6,
            "future_frames_used_for_fit_or_selection": False,
            "manual_tracks_used_for_fit_or_selection": False,
            "graph_rank": 4,
        },
        regularization={"ridge": 1e-4},
        source_checksums={"source": "a" * 64},
    )


def _released_visual_inputs() -> ReleasedPhysTwinVisualInputsV2:
    root = Path("/tmp/bayesian-phystwin-immutable-test")
    return ReleasedPhysTwinVisualInputsV2(
        raw_case_dir=root,
        final_data_sha256="a" * 64,
        metadata_sha256="b" * 64,
        pcd_sha256="c" * 64,
        calibration_sha256="d" * 64,
        cotracker_sha256=(("cotracker/camera0.npz", "e" * 64),),
        initial_match_tolerance_m=1e-3,
        object_points_m=np.zeros((2, 1, 3), dtype=np.float64),
        object_visibility=np.ones((2, 1), dtype=bool),
        object_motion_valid=np.ones((2, 1), dtype=bool),
        track_paths=(root / "cotracker" / "camera0.npz",),
        tracks_by_camera=(np.zeros((2, 1, 2), dtype=np.float64),),
        visibility_by_camera=(np.ones((2, 1), dtype=bool),),
        source_camera=np.asarray([0], dtype=np.int64),
        source_track=np.asarray([0], dtype=np.int64),
        source_world_points_m=np.zeros((1, 3), dtype=np.float64),
        initial_match_distance_m=np.asarray([0.0], dtype=np.float64),
        intrinsics=np.eye(3, dtype=np.float64)[None],
        camera_to_world=np.eye(4, dtype=np.float64)[None],
        source_fps=30.0,
        image_width=640,
        image_height=480,
    )


def _assert_irreversibly_immutable(array: np.ndarray) -> None:
    assert not array.flags.writeable
    with pytest.raises(ValueError):
        array.setflags(write=True)


def test_dynamic_discrepancy_arrays_cannot_be_reenabled_for_writes() -> None:
    correction = _dynamic_correction()
    artifact_id = correction.artifact_id

    for array in correction._array_payload().values():
        _assert_irreversibly_immutable(array)

    assert correction.artifact_id == artifact_id


def test_released_visual_inputs_cannot_mutate_behind_artifact_identity() -> None:
    inputs = _released_visual_inputs()
    artifact_id = inputs.artifact_id

    arrays = (
        inputs.object_points_m,
        inputs.object_visibility,
        inputs.object_motion_valid,
        *inputs.tracks_by_camera,
        *inputs.visibility_by_camera,
        inputs.source_camera,
        inputs.source_track,
        inputs.source_world_points_m,
        inputs.initial_match_distance_m,
        inputs.intrinsics,
        inputs.camera_to_world,
    )
    for array in arrays:
        _assert_irreversibly_immutable(array)

    assert inputs.artifact_id == artifact_id
