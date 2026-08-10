from __future__ import annotations

import json

import numpy as np

from bayesian_phystwin.deform360_joint_sparse_endpoint_v5 import (
    Deform360JointSparseEndpointConfigV5,
    Deform360ReservedViewGeometryV5,
    load_deform360_joint_sparse_endpoint_report_v5,
    score_deform360_joint_sparse_endpoint_v5,
    select_reserved_endpoint_views_v5,
    validate_deform360_joint_sparse_endpoint_report_v5,
)
from bayesian_phystwin.deform360_joint_sparse_prediction_v5 import (
    RAW_METHOD_IDS,
    VT2_VISUOTACTILE_UNGUARDED,
)


def _config() -> Deform360JointSparseEndpointConfigV5:
    return Deform360JointSparseEndpointConfigV5(
        evaluation_frame_range_half_open=(1, 3),
        maximum_target_points_per_frame_view=100,
        minimum_target_points_per_frame_view=4,
        distance_chunk_size=8,
    )


def _points() -> np.ndarray:
    rows, columns = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    return np.column_stack(
        (
            (columns.reshape(-1) - 2.0) / 100.0,
            (rows.reshape(-1) - 2.0) / 100.0,
            np.ones(25),
        )
    ).astype(np.float32)


def _views(*, empty: bool = False):
    all_cameras = ("cam-a", "cam-b", "cam-c", "cam-d")
    selected = select_reserved_endpoint_views_v5("001-test", all_cameras)
    mask = np.zeros((2, 5, 5), dtype=bool) if empty else np.ones((2, 5, 5), dtype=bool)
    intrinsics = np.asarray(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    )
    views = tuple(
        Deform360ReservedViewGeometryV5(
            object_id="001-test",
            episode_id=0,
            camera_id=camera,
            frame_indices=np.asarray([1, 2]),
            depth_m=np.ones((2, 5, 5), dtype=np.float32),
            object_mask=mask,
            intrinsics=intrinsics,
            camera_to_world=np.eye(4),
            source_artifact_ids={f"depth/{camera}.h5": "a" * 64},
        )
        for camera in selected
    )
    return all_cameras, views


def _trajectories() -> dict[str, np.ndarray]:
    trajectory = np.repeat(_points()[None], 3, axis=0)
    values = {method_id: trajectory.copy() for method_id in RAW_METHOD_IDS}
    values[VT2_VISUOTACTILE_UNGUARDED][:, :, 0] += 0.01
    return values


def test_reserved_view_selection_uses_identities_only() -> None:
    cameras = ("cam-a", "cam-b", "cam-c", "cam-d")
    first = select_reserved_endpoint_views_v5("001-test", cameras)
    second = select_reserved_endpoint_views_v5("001-test", tuple(reversed(cameras)))

    assert first == second
    assert len(first) == 2


def test_endpoint_scores_exact_geometry_after_prediction_seal() -> None:
    cameras, views = _views()
    report = score_deform360_joint_sparse_endpoint_v5(
        object_id="001-test",
        episode_id=0,
        stratum="sheet",
        prediction_seal_id="b" * 64,
        trajectories_m=_trajectories(),
        reserved_views=views,
        all_camera_ids=cameras,
        evaluation_role="development_source",
        config=_config(),
    )

    assert report["technical_failure"] is False
    assert report["cell_count_per_method"] == 4
    assert report["method_loss_mm"][RAW_METHOD_IDS[0]] < 1e-6
    assert report["method_loss_mm"][VT2_VISUOTACTILE_UNGUARDED] > 0.0
    assert report["information_boundary"]["future_geometry_used_for_prediction"] is False
    assert report["information_boundary"]["tactile_used_to_define_target"] is False
    assert report["information_boundary"]["development_suffix_used_for_scoring"] is True
    assert (
        report["information_boundary"][
            "confirmation_payloads_opened_for_authorized_scoring"
        ]
        is False
    )
    assert len(report["endpoint_report_id"]) == 64
    validated = validate_deform360_joint_sparse_endpoint_report_v5(
        report,
        expected_evaluation_role="development_source",
    )
    assert validated["endpoint_report_id"] == report["endpoint_report_id"]


def test_endpoint_report_loader_rejects_tampering(tmp_path) -> None:
    cameras, views = _views()
    report = score_deform360_joint_sparse_endpoint_v5(
        object_id="001-test",
        episode_id=0,
        stratum="sheet",
        prediction_seal_id="b" * 64,
        trajectories_m=_trajectories(),
        reserved_views=views,
        all_camera_ids=cameras,
        evaluation_role="development_source",
        config=_config(),
    )
    path = tmp_path / "endpoint-report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = load_deform360_joint_sparse_endpoint_report_v5(
        path,
        expected_evaluation_role="development_source",
    )
    assert loaded["endpoint_report_id"] == report["endpoint_report_id"]

    changed = dict(report)
    changed["technical_failure"] = 1
    path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        load_deform360_joint_sparse_endpoint_report_v5(path)
    except ValueError as error:
        assert "endpoint_report_id changed" in str(error)
    else:
        raise AssertionError("tampered endpoint report was accepted")


def test_missing_target_cell_retains_object_with_fixed_penalty() -> None:
    cameras, views = _views(empty=True)
    report = score_deform360_joint_sparse_endpoint_v5(
        object_id="001-test",
        episode_id=0,
        stratum="sheet",
        prediction_seal_id="b" * 64,
        trajectories_m=_trajectories(),
        reserved_views=views,
        all_camera_ids=cameras,
        evaluation_role="development_source",
        config=_config(),
    )

    assert report["technical_failure"] is True
    assert len(report["missing_target_cells"]) == 4
    assert set(report["method_loss_mm"].values()) == {1000.0}


def test_unregistered_endpoint_camera_is_rejected() -> None:
    cameras, views = _views()
    replacement = Deform360ReservedViewGeometryV5(
        object_id="001-test",
        episode_id=0,
        camera_id="cam-unregistered",
        frame_indices=np.asarray([1, 2]),
        depth_m=np.ones((2, 5, 5), dtype=np.float32),
        object_mask=np.ones((2, 5, 5), dtype=bool),
        intrinsics=np.asarray(
            [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
        ),
        camera_to_world=np.eye(4),
        source_artifact_ids={"depth/replacement.h5": "a" * 64},
    )
    changed = (replacement, views[1])
    try:
        score_deform360_joint_sparse_endpoint_v5(
            object_id="001-test",
            episode_id=0,
            stratum="sheet",
            prediction_seal_id="b" * 64,
            trajectories_m=_trajectories(),
            reserved_views=changed,
            all_camera_ids=cameras,
            evaluation_role="development_source",
            config=_config(),
        )
    except ValueError as error:
        assert "view roster changed" in str(error)
    else:
        raise AssertionError("an unregistered endpoint camera was accepted")


def test_confirmation_scoring_requires_machine_authorization() -> None:
    cameras, views = _views()
    try:
        score_deform360_joint_sparse_endpoint_v5(
            object_id="001-test",
            episode_id=0,
            stratum="sheet",
            prediction_seal_id="b" * 64,
            trajectories_m=_trajectories(),
            reserved_views=views,
            all_camera_ids=cameras,
            evaluation_role="independent_confirmation",
            config=_config(),
        )
    except (TypeError, ValueError) as error:
        assert "opening_authorization_id" in str(error)
    else:
        raise AssertionError("confirmation scoring opened without machine authorization")

    report = score_deform360_joint_sparse_endpoint_v5(
        object_id="001-test",
        episode_id=0,
        stratum="sheet",
        prediction_seal_id="b" * 64,
        trajectories_m=_trajectories(),
        reserved_views=views,
        all_camera_ids=cameras,
        evaluation_role="independent_confirmation",
        opening_authorization_id="c" * 64,
        config=_config(),
    )
    boundary = report["information_boundary"]
    assert boundary["confirmation_payloads_opened_for_authorized_scoring"] is True
    assert boundary["confirmation_target_outcomes_used_for_scoring"] is True
    assert boundary["confirmation_outcomes_used_before_authorization"] is False
