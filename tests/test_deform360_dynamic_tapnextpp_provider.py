from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_dynamic_query import (
    CameraPanel,
    DynamicQueryConfig,
    DynamicQuerySchedule,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_assimilation import (
    CANDIDATE_ARM,
    PERSISTENCE_ARM,
    PHYSICAL_ARM,
    SELECTED_BACKBONE_ARM,
)
from bayesian_phystwin.deform360_dynamic_tapnextpp_provider import (
    CAUSAL_FRAME_STOP_EXCLUSIVE,
    PROVIDER_ARCHIVE_FILENAME,
    PROVIDER_REPORT_FILENAME,
    CausalCameraInputs,
    validate_query_schedule_artifact,
    write_assimilation_artifacts,
    write_provider_artifacts,
    write_query_schedule_artifact,
)
from bayesian_phystwin.tapnextpp_dynamic_multiview import (
    DynamicMultiviewConfig,
    DynamicMultiviewResult,
)
from bayesian_phystwin.tapnextpp_dynamic_runtime import (
    DynamicBirthAssociations,
    DynamicTAPNextPPRuntimeResult,
)


def _schedule(*, update_frame: int = 19) -> DynamicQuerySchedule:
    config = DynamicQueryConfig()
    panel = CameraPanel(
        camera_indices=np.arange(8),
        camera_names=tuple(f"camera-{index}" for index in range(8)),
        frame_zero_coverage=np.ones(8),
        selection_scores=np.ones(8),
    )
    return DynamicQuerySchedule(
        update_frames=np.asarray([update_frame]),
        birth_frames=np.asarray([0]),
        entity_ids=np.asarray([2]),
        predicted_motion_m=np.asarray([0.01]),
        predicted_visible_views=np.asarray([8]),
        information_gain=np.asarray([1.0]),
        config=config,
        camera_panel=panel,
        physical_prefix_sha256="a" * 64,
        graph_basis_sha256="b" * 64,
        artifact_sha256="c" * 64,
    )


def _causal_inputs(*, frame_count: int = CAUSAL_FRAME_STOP_EXCLUSIVE) -> dict:
    return {
        "camera_indices": np.arange(8),
        "camera_names": tuple(f"camera-{index}" for index in range(8)),
        "rgbs": np.zeros((8, frame_count, 2, 3, 3), dtype=np.uint8),
        "depths_m": np.ones((8, frame_count, 2, 3), dtype=np.float32),
        "object_masks": np.ones((8, frame_count, 2, 3), dtype=bool),
        "intrinsics": np.repeat(np.eye(3)[None], 8, axis=0),
        "camera_to_world": np.repeat(np.eye(4)[None], 8, axis=0),
        "provenance": {"maximum_frame_read": 57},
    }


def test_causal_camera_inputs_reject_future_frames() -> None:
    valid = CausalCameraInputs(**_causal_inputs())
    assert valid.rgbs.shape[1] == 58
    with pytest.raises(ValueError, match="shape"):
        CausalCameraInputs(**_causal_inputs(frame_count=59))


def test_query_schedule_artifact_binds_hash_only_case(tmp_path: Path) -> None:
    path = tmp_path / "schedule.json"
    artifact = write_query_schedule_artifact(
        path,
        _schedule(),
        case_hash="d" * 64,
        input_sha256={"physical": "e" * 64},
    )

    assert artifact["case_hash"] == "d" * 64
    assert artifact["information_boundary"]["maximum_physical_frame_read"] == 57
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["entity_ids"] = [7]
    with pytest.raises(ValueError, match="checksum"):
        validate_query_schedule_artifact(changed)


def test_assimilation_output_excludes_all_measurement_identities(
    tmp_path: Path,
) -> None:
    baseline = np.zeros((76, 10, 3), dtype=np.float32)
    candidate = baseline.copy()
    candidate[20:, :, 0] = 0.01
    arrays = {
        PHYSICAL_ARM: baseline,
        PERSISTENCE_ARM: baseline,
        SELECTED_BACKBONE_ARM: baseline,
        CANDIDATE_ARM: candidate,
        "candidate_correction_variance_m2": np.ones_like(baseline) * 1e-6,
    }
    prediction_path, assimilation_path, report = write_assimilation_artifacts(
        tmp_path,
        {
            "schema_version": 1,
            "artifact_kind": "Deform360DynamicTAPNextPPAssimilation",
        },
        arrays,
        case_hash="f" * 64,
        measurement_entity_ids=np.asarray([3, 1, 3]),
        update_frames=np.asarray([19, 38, 57]),
        input_sha256={"belief": "a" * 64},
    )

    assert assimilation_path.is_file()
    assert report["prediction_input"]["measurement_identity_count"] == 2
    with np.load(prediction_path, allow_pickle=False) as stored:
        np.testing.assert_array_equal(
            stored["measurement_entity_ids"],
            np.asarray([1, 3]),
        )
        assert set(stored["hidden_entity_ids"]) == set(range(10)) - {1, 3}
        np.testing.assert_array_equal(
            stored["candidate_prediction_m"],
            candidate,
        )


def test_provider_artifacts_bind_runtime_and_covariance(
    tmp_path: Path,
) -> None:
    camera_count = 3
    frame_count = 4
    entity_count = 1
    scalar_shape = (frame_count, entity_count)
    covariance = np.repeat(
        (np.eye(3) * 1e-6)[None, None],
        frame_count,
        axis=0,
    )
    result = DynamicMultiviewResult(
        trajectory_world_m=np.zeros((*scalar_shape, 3)),
        proposal_available=np.ones(scalar_shape, dtype=bool),
        accepted_support=np.ones(scalar_shape, dtype=bool),
        prior_reliability=np.full(scalar_shape, 0.8),
        association_probability=np.full(scalar_shape, 0.9),
        local_covariance_m2=covariance,
        naive_independent_covariance_m2=covariance * 0.5,
        assignment_mixture_spread_m2=covariance * 0.1,
        independent_support_count=np.full(scalar_shape, camera_count),
        raw_support_count=np.full(scalar_shape, camera_count),
        reprojection_rmse_px=np.full(scalar_shape, 0.5),
        depth_residual_rmse_m=np.full(scalar_shape, 0.001),
        inlier_camera_mask=np.ones(
            (camera_count, *scalar_shape),
            dtype=bool,
        ),
        camera_cluster_ids=np.arange(camera_count),
        shared_bias_standard_deviation_m=0.005,
        config=DynamicMultiviewConfig(),
    )
    runtime = DynamicTAPNextPPRuntimeResult(
        tracks_xy=np.zeros((camera_count, frame_count, entity_count, 2)),
        visibility_probability=np.ones(
            (camera_count, frame_count, entity_count)
        ),
        active=np.ones(
            (camera_count, frame_count, entity_count),
            dtype=bool,
        ),
        rollout_count=camera_count,
        model_frame_count=frame_count,
        elapsed_seconds=1.0,
    )
    associations = DynamicBirthAssociations(
        query_points_world_m=np.zeros((entity_count, 3)),
        query_points_xy=np.zeros((camera_count, entity_count, 2)),
        valid=np.ones((camera_count, entity_count), dtype=bool),
        association_probability=np.full(
            (camera_count, entity_count),
            0.9,
        ),
        association_entropy=np.full(
            (camera_count, entity_count),
            0.1,
        ),
        candidate_pixel_covariance_px2=np.repeat(
            (np.eye(2) * 0.25)[None, None],
            camera_count,
            axis=0,
        ),
        candidate_count=np.ones(
            (camera_count, entity_count),
            dtype=np.int64,
        ),
        camera_indices=np.arange(camera_count),
        camera_names=tuple(
            f"camera-{index}" for index in range(camera_count)
        ),
    )

    archive, report_path, report = write_provider_artifacts(
        tmp_path,
        result,
        runtime,
        associations,
        _schedule(update_frame=3),
        case_hash="d" * 64,
        input_sha256={"schedule": "e" * 64},
        runtime_provenance={"device": "synthetic"},
    )

    assert archive == tmp_path / PROVIDER_ARCHIVE_FILENAME
    assert report_path == tmp_path / PROVIDER_REPORT_FILENAME
    assert report["support"]["birth_and_update_supported_fraction"] == 1.0
    assert report["information_boundary"]["maximum_rgb_depth_mask_frame_read"] == 57
    with np.load(archive, allow_pickle=False) as stored:
        np.testing.assert_array_equal(
            stored["local_covariance_m2"],
            covariance,
        )
