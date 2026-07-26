from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.phystwin_static_scene_gauge import (
    PHYSTWIN_STATIC_SCENE_GAUGE_ARTIFACT_KIND,
    PhysTwinStaticSceneGaugeConfig,
    build_phystwin_static_scene_gauge,
    load_static_scene_corrected_multiview_tracks,
    load_static_scene_corrected_source_tracks,
)
from bayesian_phystwin.phystwin_static_scene_gauge_competence import (
    StaticSceneGaugeCompetenceConfig,
    evaluate_phystwin_static_scene_gauge_prefix,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _artifacts(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    source = np.array(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[12.0, 21.0], [31.0, 42.0]],
            [[14.0, 22.0], [32.0, 44.0]],
            [[16.0, 23.0], [33.0, 46.0]],
        ],
        dtype=np.float32,
    )
    multiview = np.stack((source, source + 100.0))
    cues = tmp_path / "cues.npz"
    np.savez_compressed(
        cues,
        source_tracks_xy=source,
        source_camera=np.array([0, 1], dtype=np.int16),
        multiview_tracks_xy_prefix=multiview,
    )

    source_correction = np.zeros((3, 2, 2), dtype=np.float32)
    source_correction[:, 0] = [1.0, 2.0]
    source_supported = np.zeros((3, 2), dtype=bool)
    source_supported[:, 0] = True
    multiview_correction = np.zeros((2, 3, 2, 2), dtype=np.float32)
    multiview_correction[0, :, 0] = [1.0, 2.0]
    multiview_supported = np.zeros((2, 3, 2), dtype=bool)
    multiview_supported[0, :, 0] = True
    gauge = tmp_path / "gauge.npz"
    np.savez_compressed(
        gauge,
        source_track_correction_px=source_correction,
        source_track_variance_px2=np.ones((3, 2), dtype=np.float32),
        source_track_supported=source_supported,
        multiview_track_correction_px=multiview_correction,
        multiview_track_variance_px2=np.ones(
            (2, 3, 2),
            dtype=np.float32,
        ),
        multiview_track_supported=multiview_supported,
        camera_accepted=np.array([True, False]),
    )
    summary = {
        "schema_version": 1,
        "artifact_kind": PHYSTWIN_STATIC_SCENE_GAUGE_ARTIFACT_KIND,
        "inputs": {"cues_sha256": _sha256(cues)},
        "output": {"sha256": _sha256(gauge)},
    }
    gauge.with_suffix(".summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    return cues, gauge, source, multiview


def test_bound_source_loader_applies_only_supported_prefix_rows(
    tmp_path: Path,
) -> None:
    cues, gauge, source, _ = _artifacts(tmp_path)

    corrected, variance, supported, _ = (
        load_static_scene_corrected_source_tracks(cues, gauge)
    )

    expected = source.copy()
    expected[:3, 0] -= [1.0, 2.0]
    np.testing.assert_array_equal(corrected, expected)
    np.testing.assert_array_equal(corrected[:, 1], source[:, 1])
    np.testing.assert_array_equal(corrected[3], source[3])
    assert variance.shape == (3, 2)
    assert np.all(supported[:, 0])
    assert not np.any(supported[:, 1])


def test_bound_multiview_loader_preserves_rejected_camera(
    tmp_path: Path,
) -> None:
    cues, gauge, _, multiview = _artifacts(tmp_path)

    corrected, _, supported, _ = (
        load_static_scene_corrected_multiview_tracks(cues, gauge)
    )

    expected = multiview.copy()
    expected[0, :3, 0] -= [1.0, 2.0]
    np.testing.assert_array_equal(corrected, expected)
    np.testing.assert_array_equal(corrected[1], multiview[1])
    assert not np.any(supported[1])


def test_loader_rejects_changed_source_cues(tmp_path: Path) -> None:
    cues, gauge, source, multiview = _artifacts(tmp_path)
    np.savez_compressed(
        cues,
        source_tracks_xy=source + 1.0,
        source_camera=np.array([0, 1], dtype=np.int16),
        multiview_tracks_xy_prefix=multiview,
    )

    with pytest.raises(ValueError, match="different source cues"):
        load_static_scene_corrected_source_tracks(cues, gauge)


def test_builder_exactly_falls_back_without_static_queries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = tmp_path / "raw"
    (raw / "mask").mkdir(parents=True)
    (raw / "depth" / "0").mkdir(parents=True)
    (raw / "color" / "0").mkdir(parents=True)
    masks = [
        [
            {
                "object": np.ones((8, 8), dtype=bool),
                "controller": np.zeros((8, 8), dtype=bool),
            }
        ]
        for _ in range(2)
    ]
    with (raw / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)
    np.save(
        raw / "depth" / "0" / "0.npy",
        np.full((8, 8), 1000, dtype=np.uint16),
    )
    for frame in range(2):
        (raw / "color" / "0" / f"{frame}.png").write_bytes(
            f"frame-{frame}".encode()
        )
    cues = tmp_path / "cues.npz"
    tracks = np.full((2, 1, 2), 4.0, dtype=np.float32)
    np.savez_compressed(
        cues,
        source_tracks_xy=tracks,
        source_camera=np.array([0], dtype=np.int16),
        multiview_tracks_xy_prefix=tracks[None],
    )
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    cotracker = tmp_path / "cotracker"
    cotracker.mkdir()
    monkeypatch.setattr(
        "bayesian_phystwin.phystwin_static_scene_gauge.CoTracker3OnlineRunner",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        "bayesian_phystwin.phystwin_static_scene_gauge._git_revision",
        lambda path: "test-revision",
    )
    gauge = tmp_path / "gauge.npz"

    summary = build_phystwin_static_scene_gauge(
        cues,
        raw,
        checkpoint,
        cotracker,
        gauge,
        config=PhysTwinStaticSceneGaugeConfig(train_end_frame=2),
        device="cpu",
    )
    corrected, _, supported, _ = load_static_scene_corrected_source_tracks(
        cues,
        gauge,
    )

    np.testing.assert_array_equal(corrected, tracks)
    assert not np.any(supported)
    assert summary["camera_summaries"]["0"]["source"]["reason"] == (
        "insufficient-static-scene-queries"
    )


def test_prefix_competence_scores_only_allowed_frames(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "depth" / "0").mkdir(parents=True)
    (raw / "metadata.json").write_text(
        json.dumps(
            {
                "intrinsics": [
                    [[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]
                ]
            }
        ),
        encoding="utf-8",
    )
    with (raw / "calibrate.pkl").open("wb") as handle:
        pickle.dump([np.eye(4)], handle)
    for frame in range(4):
        np.save(
            raw / "depth" / "0" / f"{frame}.npy",
            np.full((9, 9), 1000, dtype=np.uint16),
        )

    source = np.array(
        [
            [[4.0, 4.0]],
            [[5.0, 4.0]],
            [[6.0, 4.0]],
            [[7.0, 4.0]],
            [[100.0, 100.0]],
        ],
        dtype=np.float32,
    )
    cues = tmp_path / "competence_cues.npz"
    np.savez_compressed(
        cues,
        source_tracks_xy=source,
        source_camera=np.array([0], dtype=np.int16),
        multiview_tracks_xy_prefix=source[None, :4],
        cotracker_quality_probability=np.ones((5, 1), dtype=np.float32),
        forward_backward_error_px=np.zeros((5, 1), dtype=np.float32),
        forward_backward_valid=np.ones((5, 1), dtype=bool),
        boundary_distance=np.ones((5, 1), dtype=np.float32),
        cue_available=np.array([[True], [True], [True], [True], [False]]),
    )
    correction = np.zeros((4, 1, 2), dtype=np.float32)
    correction[:, 0, 0] = np.arange(4)
    gauge = tmp_path / "competence_gauge.npz"
    np.savez_compressed(
        gauge,
        source_track_correction_px=correction,
        source_track_variance_px2=np.ones((4, 1), dtype=np.float32),
        source_track_supported=np.ones((4, 1), dtype=bool),
        multiview_track_correction_px=correction[None],
        multiview_track_variance_px2=np.ones((1, 4, 1), dtype=np.float32),
        multiview_track_supported=np.ones((1, 4, 1), dtype=bool),
        camera_accepted=np.array([True]),
    )
    gauge.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_kind": PHYSTWIN_STATIC_SCENE_GAUGE_ARTIFACT_KIND,
                "inputs": {"cues_sha256": _sha256(cues)},
                "output": {"sha256": _sha256(gauge)},
                "camera_summaries": {"0": {"accepted": True}},
            }
        ),
        encoding="utf-8",
    )
    final_data = tmp_path / "final_data.pkl"
    with final_data.open("wb") as handle:
        pickle.dump(
            {"object_points": np.array([[[0.0, 0.0, 1.0]]])},
            handle,
        )
    manual = tmp_path / "manual.pkl"
    manual_values = np.zeros((5, 1, 3), dtype=float)
    manual_values[:, :, 2] = 1.0
    manual_values[4, 0, 0] = 10_000.0
    with manual.open("wb") as handle:
        pickle.dump(manual_values, handle)

    result = evaluate_phystwin_static_scene_gauge_prefix(
        cues,
        gauge,
        raw,
        final_data,
        manual,
        case="fixture",
        config=StaticSceneGaugeCompetenceConfig(
            train_end_frame=4,
            late_frame_count=2,
        ),
    )

    assert result["raw"]["mean_error_mm"] > 0.0
    assert result["static_scene_gauge"]["mean_error_mm"] == pytest.approx(0.0)
    assert result["relative_improvement"]["mean_error_mm"] == pytest.approx(1.0)
    assert result["information_boundary"]["future_manual_track_read"] is False
