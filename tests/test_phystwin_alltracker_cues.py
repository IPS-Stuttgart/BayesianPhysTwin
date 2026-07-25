from dataclasses import asdict
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import bayesian_phystwin.phystwin_alltracker_cues as alltracker
from bayesian_phystwin.deform360_raw_camera_observation import (
    ALLTRACKER_CHECKPOINT_SHA256,
    ALLTRACKER_RUNTIME_SOURCE_SHA256,
)
from bayesian_phystwin.phystwin_alltracker_cues import (
    AllTrackerDensePrediction,
    PhysTwinAllTrackerMultiviewCueConfig,
    build_phystwin_alltracker_multiview_cues,
)
from bayesian_phystwin.phystwin_raw_cues import PhysTwinRawTrackMap


def _mapping(root: Path) -> PhysTwinRawTrackMap:
    frame_count = 4
    track_count = 2
    camera_count = 3
    initial = np.asarray(
        [[-0.1, 0.0, 1.0], [0.1, 0.0, 1.0]],
        dtype=float,
    )
    points = np.repeat(initial[None], frame_count, axis=0)
    return PhysTwinRawTrackMap(
        final_points=points,
        final_visible=np.ones((frame_count, track_count), dtype=bool),
        camera_points=np.zeros((camera_count, 8, 8, 3), dtype=float),
        track_paths=tuple(root / f"camera-{index}.npz" for index in range(3)),
        tracks_by_camera=tuple(
            np.zeros((frame_count, 1, 2), dtype=float)
            for _ in range(camera_count)
        ),
        visibility_by_camera=tuple(
            np.ones((frame_count, 1), dtype=bool)
            for _ in range(camera_count)
        ),
        source_camera=np.asarray([0, 1], dtype=np.int16),
        source_track=np.asarray([0, 0], dtype=np.int32),
        initial_match_distance_m=np.zeros(track_count, dtype=float),
        source_world_points=initial,
    )


def _write_raw_case(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "mask").mkdir()
    intrinsics = np.repeat(np.eye(3)[None], 3, axis=0)
    (root / "metadata.json").write_text(
        json.dumps({"intrinsics": intrinsics.tolist()}),
        encoding="utf-8",
    )
    with (root / "calibrate.pkl").open("wb") as handle:
        pickle.dump(np.repeat(np.eye(4)[None], 3, axis=0), handle)
    masks = [
        [{"object": np.ones((8, 8), dtype=bool)} for _ in range(3)]
        for _ in range(4)
    ]
    with (root / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)


def _write_source_cues(
    path: Path,
    config: PhysTwinAllTrackerMultiviewCueConfig,
    *,
    future_quality: float = 0.0,
) -> dict[str, np.ndarray]:
    tracks = np.full((4, 2, 2), np.nan, dtype=np.float32)
    tracks[:3, 0] = np.asarray([[2.0, 2.0], [2.1, 2.0], [2.2, 2.0]])
    tracks[:3, 1] = np.asarray([[4.0, 4.0], [4.1, 4.0], [4.2, 4.0]])
    quality = np.zeros((4, 2), dtype=np.float32)
    quality[:3] = 0.9
    quality[3] = future_quality
    arrays = {
        "source_tracks_xy": tracks,
        "source_quality_probability": quality,
        "forward_backward_error_px": np.asarray(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=np.float32,
        ),
        "forward_backward_valid": np.asarray(
            [[True, True], [True, True], [True, True], [False, False]],
        ),
        "boundary_distance": np.asarray(
            [[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.0, 0.0]],
            dtype=np.float32,
        ),
        "cue_available": np.asarray(
            [[True, True], [True, True], [True, True], [False, False]],
        ),
        "source_camera": np.asarray([0, 1], dtype=np.int16),
        "source_track": np.asarray([0, 0], dtype=np.int32),
        "initial_match_distance_m": np.zeros(2, dtype=np.float32),
    }
    np.savez_compressed(path, **arrays)
    summary = {
        "artifact_kind": "PhysTwinAllTrackerCues",
        "config": asdict(config.source_config()),
        "tracker": {
            "runtime_source_sha256": ALLTRACKER_RUNTIME_SOURCE_SHA256,
            "checkpoint_sha256": ALLTRACKER_CHECKPOINT_SHA256,
        },
        "output": {"sha256": alltracker._sha256(path)},
    }
    path.with_suffix(".summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return arrays


class _FakeRunner:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def track(
        self,
        video_rgb: np.ndarray,
        query_pixels_xy: np.ndarray,
    ) -> AllTrackerDensePrediction:
        queries = np.asarray(query_pixels_xy, dtype=np.float32)
        tracks = np.repeat(queries[None], len(video_rgb), axis=0)
        tracks[:, :, 0] += np.arange(len(video_rgb))[:, None] * 0.1
        return AllTrackerDensePrediction(
            tracks_xy=tracks,
            quality_probability=np.full(
                tracks.shape[:2],
                0.9,
                dtype=np.float32,
            ),
        )

    def close(self) -> None:
        pass


def _fake_eligibility(
    world_points: np.ndarray,
    *_args: object,
    **_kwargs: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(world_points)
    return (
        np.asarray([[2.0, 2.0], [4.0, 4.0]], dtype=float)[:count],
        np.ones(count, dtype=bool),
        np.full(count, 0.001, dtype=float),
    )


def _fake_triangulation(
    tracks_xy: np.ndarray,
    valid: np.ndarray,
    *_args: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count, track_count = tracks_xy.shape[1:3]
    support = np.sum(valid, axis=0).astype(np.int16)
    points = np.zeros((frame_count, track_count, 3), dtype=float)
    points[:, :, 0] = np.arange(frame_count)[:, None] * 0.01
    points[:, :, 1] = np.arange(track_count)[None] * 0.1
    error = np.full((frame_count, track_count), 0.25, dtype=float)
    points[support < 2] = np.nan
    error[support < 2] = np.nan
    return points, error, support


def test_alltracker_multiview_augmentation_preserves_source_and_future(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PhysTwinAllTrackerMultiviewCueConfig(train_end_frame=3)
    raw_case = tmp_path / "raw"
    _write_raw_case(raw_case)
    final_data = tmp_path / "final.pkl"
    final_data.write_bytes(b"final")
    base = tmp_path / "base.npz"
    base_arrays = _write_source_cues(base, config)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "multiview.npz"

    monkeypatch.setattr(
        alltracker,
        "load_phystwin_raw_track_map",
        lambda *_args, **_kwargs: _mapping(tmp_path),
    )
    monkeypatch.setattr(
        alltracker,
        "_load_video_prefix",
        lambda *_args, **_kwargs: np.zeros((3, 8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        alltracker,
        "_initial_multiview_eligibility",
        _fake_eligibility,
    )
    monkeypatch.setattr(alltracker, "PhysTwinAllTrackerRunner", _FakeRunner)
    monkeypatch.setattr(
        alltracker,
        "triangulate_multiview_tracks",
        _fake_triangulation,
    )

    summary = build_phystwin_alltracker_multiview_cues(
        final_data,
        raw_case,
        tmp_path / "alltracker",
        checkpoint,
        base,
        output,
        config=config,
        device="cpu",
    )

    with np.load(output) as archive:
        for name, expected in base_arrays.items():
            assert np.array_equal(archive[name], expected, equal_nan=True)
        assert archive["multiview_tracks_xy_prefix"].shape == (3, 3, 2, 2)
        assert np.all(archive["multiview_camera_count"][:3] == 3)
        assert np.all(archive["multiview_point_valid"][:3])
        assert not np.any(archive["multiview_point_valid"][3:])
        assert np.all(np.isnan(archive["multiview_points_world_m"][3:]))
    assert summary["compatibility"]["source_arrays_preserved_exactly"]
    assert summary["information_boundary"]["future_rgb_read"] is False
    assert summary["multiview"]["three_view_visible_fraction"] == 1.0


def test_alltracker_multiview_rejects_future_source_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PhysTwinAllTrackerMultiviewCueConfig(train_end_frame=3)
    raw_case = tmp_path / "raw"
    _write_raw_case(raw_case)
    base = tmp_path / "base.npz"
    _write_source_cues(base, config, future_quality=0.1)
    monkeypatch.setattr(
        alltracker,
        "load_phystwin_raw_track_map",
        lambda *_args, **_kwargs: _mapping(tmp_path),
    )

    with pytest.raises(ValueError, match="future quality"):
        build_phystwin_alltracker_multiview_cues(
            tmp_path / "final.pkl",
            raw_case,
            tmp_path / "alltracker",
            tmp_path / "checkpoint.pth",
            base,
            tmp_path / "output.npz",
            config=config,
        )


def test_alltracker_multiview_rejects_changed_source_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = PhysTwinAllTrackerMultiviewCueConfig(train_end_frame=3)
    raw_case = tmp_path / "raw"
    _write_raw_case(raw_case)
    base = tmp_path / "base.npz"
    _write_source_cues(base, config)
    summary_path = base.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config"]["max_side"] = 256
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(
        alltracker,
        "load_phystwin_raw_track_map",
        lambda *_args, **_kwargs: _mapping(tmp_path),
    )

    with pytest.raises(ValueError, match="config differs"):
        build_phystwin_alltracker_multiview_cues(
            tmp_path / "final.pkl",
            raw_case,
            tmp_path / "alltracker",
            tmp_path / "checkpoint.pth",
            base,
            tmp_path / "output.npz",
            config=config,
        )


def test_alltracker_multiview_config_rejects_invalid_cycle_threshold() -> None:
    with pytest.raises(ValueError, match="maximum_cycle_error_px"):
        PhysTwinAllTrackerMultiviewCueConfig(
            train_end_frame=3,
            maximum_cycle_error_px=0.0,
        )
