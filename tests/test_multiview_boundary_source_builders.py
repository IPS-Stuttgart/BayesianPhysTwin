from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.phystwin_alltracker_cues as alltracker
import bayesian_phystwin.phystwin_cotracker3_cues as cotracker
import bayesian_phystwin.phystwin_motioncrafter_assimilation as assimilation
from bayesian_phystwin.phystwin_alltracker_cues import (
    AllTrackerDensePrediction,
    PhysTwinAllTrackerCueConfig,
    build_phystwin_alltracker_cues,
)
from bayesian_phystwin.phystwin_cotracker3_cues import (
    CoTracker3CueConfig,
    CoTracker3Prediction,
    build_phystwin_cotracker3_cues,
)
from bayesian_phystwin.phystwin_raw_cues import PhysTwinRawTrackMap


def _mapping(root: Path, *, frame_count: int) -> PhysTwinRawTrackMap:
    archived = np.zeros((frame_count, 1, 2), dtype=float)
    return PhysTwinRawTrackMap(
        final_points=np.zeros((frame_count, 1, 3), dtype=float),
        final_visible=np.ones((frame_count, 1), dtype=bool),
        camera_points=np.zeros((1, 5, 5, 3), dtype=float),
        track_paths=(root / "camera-0.npz",),
        tracks_by_camera=(archived,),
        visibility_by_camera=(np.ones((frame_count, 1), dtype=bool),),
        source_camera=np.asarray([0], dtype=np.int16),
        source_track=np.asarray([0], dtype=np.int32),
        initial_match_distance_m=np.zeros(1, dtype=float),
        source_world_points=np.zeros((1, 3), dtype=float),
    )


def _write_raw_case(root: Path, *, frame_count: int) -> None:
    (root / "mask").mkdir(parents=True)
    (root / "metadata.json").write_text(
        json.dumps({"intrinsics": np.eye(3)[None].tolist()}),
        encoding="utf-8",
    )
    with (root / "calibrate.pkl").open("wb") as handle:
        pickle.dump(np.eye(4)[None], handle)
    masks = [
        [{"object": np.ones((5, 5), dtype=bool)}]
        for _ in range(frame_count)
    ]
    with (root / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)


class _AllTrackerRunner:
    source_sha256 = "a" * 64
    checkpoint_sha256 = "b" * 64

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def track(
        self,
        video_rgb: np.ndarray,
        query_pixels_xy: np.ndarray,
    ) -> AllTrackerDensePrediction:
        queries = np.asarray(query_pixels_xy, dtype=np.float32)
        tracks = np.repeat(queries[None], len(video_rgb), axis=0)
        return AllTrackerDensePrediction(
            tracks_xy=tracks,
            quality_probability=np.full(tracks.shape[:2], 0.9, dtype=np.float32),
        )

    def close(self) -> None:
        pass


class _CoTrackerRunner:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def track(
        self,
        video: np.ndarray,
        queries_xy: np.ndarray,
    ) -> CoTracker3Prediction:
        queries = np.asarray(queries_xy, dtype=np.float32)
        tracks = np.repeat(queries[None], len(video), axis=0)
        probability = np.full(tracks.shape[:2], 0.9, dtype=np.float32)
        return CoTracker3Prediction(
            tracks_xy=tracks,
            visibility_probability=probability,
            confidence_probability=probability,
        )


def test_alltracker_source_builder_uses_border_aware_distance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    _write_raw_case(raw, frame_count=3)
    final_data = tmp_path / "final.pkl"
    final_data.write_bytes(b"final")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "alltracker.npz"

    monkeypatch.setattr(
        alltracker,
        "load_phystwin_raw_track_map",
        lambda *_args, **_kwargs: _mapping(tmp_path, frame_count=3),
    )
    monkeypatch.setattr(
        alltracker,
        "_load_video_prefix",
        lambda *_args, **_kwargs: np.zeros((2, 5, 5, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(alltracker, "PhysTwinAllTrackerRunner", _AllTrackerRunner)

    build_phystwin_alltracker_cues(
        final_data,
        raw,
        tmp_path / "alltracker-source",
        checkpoint,
        output,
        config=PhysTwinAllTrackerCueConfig(train_end_frame=2),
        device="cpu",
    )

    with np.load(output) as archive:
        np.testing.assert_allclose(archive["boundary_distance"][:2, 0], 0.2)
        assert not archive["cue_available"][2, 0]


def test_cotracker_source_builder_uses_border_aware_distance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    _write_raw_case(raw, frame_count=4)
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
        lambda *_args, **_kwargs: _mapping(tmp_path, frame_count=4),
    )
    monkeypatch.setattr(
        cotracker,
        "_load_video_prefix",
        lambda *_args, **_kwargs: np.zeros((3, 5, 5, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(cotracker, "CoTracker3OnlineRunner", _CoTrackerRunner)
    monkeypatch.setattr(cotracker, "_git_revision", lambda _path: "c" * 40)
    monkeypatch.setattr(
        cotracker,
        "_initial_multiview_eligibility",
        lambda *_args, **_kwargs: (
            np.zeros((1, 2), dtype=float),
            np.ones(1, dtype=bool),
            np.zeros(1, dtype=float),
        ),
    )

    build_phystwin_cotracker3_cues(
        final_data,
        raw,
        checkpoint,
        tracker_root,
        output,
        config=CoTracker3CueConfig(train_end_frame=3, window_length=4),
        device="cpu",
    )

    with np.load(output) as archive:
        np.testing.assert_allclose(archive["boundary_distance"][:3, 0], 0.2)
        assert not archive["cue_available"][3, 0]


def test_variant_summary_covers_manual_uncertainty_and_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assimilation,
        "dense_graph_error_by_frame",
        lambda *_args, **_kwargs: np.zeros(2, dtype=float),
    )
    monkeypatch.setattr(
        assimilation,
        "manual_track_association_audit",
        lambda *_args, **_kwargs: {
            "error_by_sampled_frame_m": [0.0, 0.0],
            "graph_vertex_indices": [0],
        },
    )
    monkeypatch.setattr(
        assimilation,
        "_nearest_distances",
        lambda left, _right, **_kwargs: (np.zeros(len(left)), np.zeros(len(left))),
    )

    trajectory = np.zeros((2, 1, 3), dtype=float)
    valid = np.ones((2, 1), dtype=bool)
    reliability = np.ones((2, 1), dtype=float)
    frames = np.asarray([0, 1], dtype=np.int64)
    train = np.asarray([True, False])
    future = np.asarray([False, True])
    manual = np.zeros((2, 1, 3), dtype=float)
    covariance = np.broadcast_to(0.1 * np.eye(3), (2, 1, 3, 3)).copy()

    summary, audit = assimilation._variant_summary(
        trajectory[0],
        trajectory,
        trajectory,
        valid,
        reliability,
        manual,
        frames,
        train,
        future,
        trajectory,
        valid,
        covariance,
    )
    assert audit is not None
    assert summary["manual_identity_audit"]["available"] is True
    assert summary["manual_identity_uncertainty_audit"]["available"] is True
    assert summary["manual_identity_uncertainty_audit"][
        "track_count_by_sampled_frame"
    ] == [1, 1]

    absent, absent_audit = assimilation._variant_summary(
        trajectory[0],
        trajectory,
        trajectory,
        valid,
        reliability,
        None,
        frames,
        train,
        future,
        trajectory,
        valid,
        None,
    )
    assert absent_audit is None
    assert absent["manual_identity_audit"]["available"] is False
