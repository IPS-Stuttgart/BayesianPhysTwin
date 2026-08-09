import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.phystwin_alltracker_cues as alltracker
import bayesian_phystwin.phystwin_cotracker3_cues as cotracker
import bayesian_phystwin.phystwin_motioncrafter_assimilation as motion
from bayesian_phystwin.mask_distance import (
    _interior_mask_distance_fallback,
    interior_mask_distance,
)
from bayesian_phystwin.phystwin_alltracker_cues import (
    AllTrackerDensePrediction,
    PhysTwinAllTrackerCueConfig,
)
from bayesian_phystwin.phystwin_cotracker3_cues import (
    CoTracker3CueConfig,
    CoTracker3Prediction,
    project_world_points,
    triangulate_multiview_tracks,
)
from bayesian_phystwin.phystwin_motioncrafter_assimilation import (
    _mask_boundary_distance,
)
from bayesian_phystwin.phystwin_raw_cues import PhysTwinRawTrackMap


def test_boundary_distance_treats_image_exterior_as_background() -> None:
    mask = np.ones((3, 4), dtype=bool)
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    np.testing.assert_allclose(interior_mask_distance(mask), expected)
    np.testing.assert_allclose(_mask_boundary_distance(mask), expected)


def test_numpy_boundary_fallback_is_exact_euclidean() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    expected = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )

    fallback = _interior_mask_distance_fallback(mask)
    np.testing.assert_allclose(fallback, expected)
    np.testing.assert_allclose(interior_mask_distance(mask), fallback)


def test_triangulation_rejects_behind_camera_support() -> None:
    intrinsics = np.repeat(np.eye(3)[None], 2, axis=0)
    camera_to_world = np.repeat(np.eye(4)[None], 2, axis=0)
    camera_to_world[1, 0, 3] = 1.0
    camera_to_world[1, 2, 3] = 1.0
    point = np.array([[0.0, 0.0, 0.5]])
    tracks = np.empty((2, 1, 1, 2), dtype=float)
    depths = []
    for camera in range(2):
        tracks[camera, 0], depth = project_world_points(
            point, intrinsics[camera], camera_to_world[camera]
        )
        depths.append(float(depth[0]))

    assert depths[0] > 0.0
    assert depths[1] < 0.0
    reconstructed, error, count = triangulate_multiview_tracks(
        tracks,
        np.ones((2, 1, 1), dtype=bool),
        np.ones((2, 1, 1), dtype=float),
        intrinsics,
        camera_to_world,
    )

    np.testing.assert_array_equal(count, [[1]])
    assert np.all(np.isnan(reconstructed[0, 0]))
    assert np.isnan(error[0, 0])


def _single_camera_mapping(root: Path) -> PhysTwinRawTrackMap:
    frame_count = 4
    world = np.asarray([[2.0, 3.0, 1.0]], dtype=float)
    final_points = np.repeat(world[None], frame_count, axis=0)
    camera_points = np.zeros((1, 8, 8, 3), dtype=float)
    camera_points[0, 3, 2] = world[0]
    archived_tracks = np.repeat(
        np.asarray([[[3.0, 2.0]]], dtype=float), frame_count, axis=0
    )
    return PhysTwinRawTrackMap(
        final_points=final_points,
        final_visible=np.ones((frame_count, 1), dtype=bool),
        camera_points=camera_points,
        track_paths=(root / "camera-0.npz",),
        tracks_by_camera=(archived_tracks,),
        visibility_by_camera=(np.ones((frame_count, 1), dtype=bool),),
        source_camera=np.asarray([0], dtype=np.int16),
        source_track=np.asarray([0], dtype=np.int32),
        initial_match_distance_m=np.zeros(1, dtype=float),
        source_world_points=world,
    )


def _write_single_camera_raw_case(root: Path) -> None:
    (root / "mask").mkdir(parents=True)
    (root / "metadata.json").write_text(
        json.dumps({"intrinsics": [np.eye(3).tolist()]}), encoding="utf-8"
    )
    with (root / "calibrate.pkl").open("wb") as handle:
        pickle.dump(np.eye(4)[None], handle)
    masks = [[{"object": np.ones((8, 8), dtype=bool)}] for _ in range(4)]
    with (root / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)


class _FakeAllTrackerRunner:
    source_sha256 = "source"
    checkpoint_sha256 = "checkpoint"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def track(
        self, video_rgb: np.ndarray, query_pixels_xy: np.ndarray
    ) -> AllTrackerDensePrediction:
        queries = np.asarray(query_pixels_xy, dtype=np.float32)
        tracks = np.repeat(queries[None], len(video_rgb), axis=0)
        return AllTrackerDensePrediction(
            tracks_xy=tracks,
            quality_probability=np.full(tracks.shape[:2], 0.9, dtype=np.float32),
        )

    def close(self) -> None:
        pass


class _FakeCoTrackerRunner:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def track(
        self, video_rgb: np.ndarray, query_pixels_xy: np.ndarray
    ) -> CoTracker3Prediction:
        queries = np.asarray(query_pixels_xy, dtype=np.float32)
        tracks = np.repeat(queries[None], len(video_rgb), axis=0)
        probability = np.full(tracks.shape[:2], 0.9, dtype=np.float32)
        return CoTracker3Prediction(
            tracks_xy=tracks,
            visibility_probability=probability,
            confidence_probability=probability,
        )


def test_alltracker_source_builder_uses_border_aware_distance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_case = tmp_path / "raw"
    _write_single_camera_raw_case(raw_case)
    final_data = tmp_path / "final.pkl"
    final_data.write_bytes(b"final")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "alltracker.npz"

    monkeypatch.setattr(
        alltracker,
        "load_phystwin_raw_track_map",
        lambda *_args, **_kwargs: _single_camera_mapping(tmp_path),
    )
    monkeypatch.setattr(
        alltracker,
        "_load_video_prefix",
        lambda *_args, **_kwargs: np.zeros((3, 8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(alltracker, "PhysTwinAllTrackerRunner", _FakeAllTrackerRunner)

    summary = alltracker.build_phystwin_alltracker_cues(
        final_data,
        raw_case,
        tmp_path / "alltracker-source",
        checkpoint,
        output,
        config=PhysTwinAllTrackerCueConfig(train_end_frame=3, window_length=4),
        device="cpu",
    )

    with np.load(output) as archive:
        assert archive["boundary_distance"][0, 0] > 0.0
        assert np.all(archive["cue_available"][:3])
        assert not archive["cue_available"][3, 0]
    assert summary["artifact_kind"] == "PhysTwinAllTrackerCues"


def test_cotracker_source_builder_exercises_boundary_and_mapping_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_case = tmp_path / "raw"
    _write_single_camera_raw_case(raw_case)
    final_data = tmp_path / "final.pkl"
    final_data.write_bytes(b"final")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    tracker_root = tmp_path / "cotracker"
    (tracker_root / ".git").mkdir(parents=True)
    output = tmp_path / "cotracker.npz"

    monkeypatch.setattr(
        cotracker,
        "load_phystwin_raw_track_map",
        lambda *_args, **_kwargs: _single_camera_mapping(tmp_path),
    )
    monkeypatch.setattr(
        cotracker,
        "_load_video_prefix",
        lambda *_args, **_kwargs: np.zeros((3, 8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cotracker, "CoTracker3OnlineRunner", _FakeCoTrackerRunner)
    monkeypatch.setattr(cotracker, "_git_revision", lambda *_args: "0" * 40)

    summary = cotracker.build_phystwin_cotracker3_cues(
        final_data,
        raw_case,
        checkpoint,
        tracker_root,
        output,
        config=CoTracker3CueConfig(train_end_frame=3, window_length=4),
        device="cpu",
    )

    with np.load(output) as archive:
        assert archive["boundary_distance"][0, 0] > 0.0
        assert archive["source_tracks_xy"].shape == (4, 1, 2)
        assert not archive["cue_available"][3, 0]
    assert summary["archive_track_parity_error_px"]["0"]["maximum"] == 0.0


def test_variant_summary_covers_manual_uncertainty_and_absent_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_initial = np.zeros((1, 3), dtype=float)
    trajectory = np.zeros((2, 1, 3), dtype=float)
    positions = trajectory.copy()
    valid = np.ones((2, 1), dtype=bool)
    reliability = np.ones((2, 1), dtype=float)
    frame_indices = np.asarray([0, 1], dtype=np.int64)
    train = np.asarray([True, False])
    future = np.asarray([False, True])
    visible = np.ones((2, 1), dtype=bool)

    monkeypatch.setattr(
        motion,
        "dense_graph_error_by_frame",
        lambda *_args, **_kwargs: np.zeros(2, dtype=float),
    )
    monkeypatch.setattr(
        motion,
        "_nearest_distances",
        lambda first, *_args, **_kwargs: (
            np.zeros(len(first), dtype=float),
            np.zeros(len(first), dtype=np.int64),
        ),
    )

    no_manual, no_audit = motion._variant_summary(
        graph_initial,
        trajectory,
        positions,
        valid,
        reliability,
        None,
        frame_indices,
        train,
        future,
        trajectory,
        visible,
    )
    assert no_audit is None
    assert no_manual["manual_identity_audit"]["available"] is False

    monkeypatch.setattr(
        motion,
        "manual_track_association_audit",
        lambda *_args, **_kwargs: {
            "error_by_sampled_frame_m": [0.0, 0.0],
            "graph_vertex_indices": [0],
        },
    )
    covariance = np.broadcast_to(np.eye(3), (2, 1, 3, 3)).copy()
    manual, audit = motion._variant_summary(
        graph_initial,
        trajectory,
        positions,
        valid,
        reliability,
        trajectory.copy(),
        frame_indices,
        train,
        future,
        trajectory,
        visible,
        covariance,
    )

    assert audit is not None
    assert manual["manual_identity_audit"]["available"] is True
    assert manual["manual_identity_uncertainty_audit"]["available"] is True
