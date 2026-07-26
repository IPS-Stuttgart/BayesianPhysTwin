from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.cli.phystwin_spatialtrackerv2_competence import (
    build_parser,
)
from bayesian_phystwin.phystwin_spatialtrackerv2_competence import (
    SpatialTrackerV2Prediction,
    camera_tracks_to_world,
    load_canonical_spatialtrackerv2_prediction,
    load_spatialtrackerv2_prediction,
    project_world_queries_to_pixels,
    save_canonical_spatialtrackerv2_prediction,
    validate_spatialtrackerv2_prediction_contract,
)


def _query_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.1, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _query_pixels() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 4.0, 4.0],
            [0.0, 4.4, 4.0],
        ],
        dtype=np.float32,
    )


def _coords(frame_count: int = 4) -> np.ndarray:
    coords = np.repeat(_query_points()[None, :, 1:], frame_count, axis=0)
    coords[:, :, 0] += 0.01 * np.arange(frame_count)[:, None]
    return coords


def test_projection_uses_world_to_camera_calibration() -> None:
    intrinsics = np.asarray(
        [
            [4.0, 0.0, 4.0],
            [0.0, 4.0, 4.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world_to_camera = np.eye(4)

    pixels, depths = project_world_queries_to_pixels(
        _query_points()[:, 1:],
        intrinsics,
        world_to_camera,
    )

    np.testing.assert_allclose(pixels, _query_pixels())
    np.testing.assert_allclose(depths, [1.0, 1.0])


def test_camera_tracks_transform_into_world_frame() -> None:
    tracks = np.asarray([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]])
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]

    world = camera_tracks_to_world(tracks, camera_to_world)

    np.testing.assert_allclose(
        world,
        [[[1.0, 2.0, 4.0], [1.1, 2.0, 4.0]]],
    )


def test_official_archive_loads_and_round_trips(tmp_path: Path) -> None:
    source = tmp_path / "official.npz"
    np.savez(
        source,
        coords_world_m=_coords(),
        valid=np.ones((4, 2), dtype=bool),
        visibility_probability=np.full((4, 2), 0.9),
        confidence=np.full((4, 2), 0.8),
        query_points=_query_points(),
        query_pixels_xyt=_query_pixels(),
    )

    prediction = load_spatialtrackerv2_prediction(source)
    validate_spatialtrackerv2_prediction_contract(
        prediction,
        _query_points(),
        _query_pixels(),
        expected_frame_count=4,
    )
    canonical = tmp_path / "canonical.npz"
    save_canonical_spatialtrackerv2_prediction(canonical, prediction)
    restored = load_canonical_spatialtrackerv2_prediction(canonical)

    np.testing.assert_allclose(restored.coords_world_m, _coords())
    np.testing.assert_array_equal(restored.valid, np.ones((4, 2), dtype=bool))
    np.testing.assert_allclose(restored.query_points, _query_points())
    np.testing.assert_allclose(restored.query_pixels_xyt, _query_pixels())


def test_archive_rejects_non_boolean_validity(tmp_path: Path) -> None:
    source = tmp_path / "malformed.npz"
    np.savez(
        source,
        coords_world_m=_coords(),
        valid=np.ones((4, 2), dtype=np.float32),
        visibility_probability=np.full((4, 2), 0.9),
        confidence=np.full((4, 2), 0.8),
        query_points=_query_points(),
        query_pixels_xyt=_query_pixels(),
    )

    with pytest.raises(ValueError, match="valid must be boolean"):
        load_spatialtrackerv2_prediction(source)


def test_prediction_contract_rejects_pixel_query_mismatch() -> None:
    prediction = SpatialTrackerV2Prediction(
        coords_world_m=_coords(),
        valid=np.ones((4, 2), dtype=bool),
        visibility_probability=np.full((4, 2), 0.9),
        confidence=np.full((4, 2), 0.8),
        query_points=_query_points(),
        query_pixels_xyt=_query_pixels(),
    )
    changed = _query_pixels()
    changed[1, 1] += 1.0

    with pytest.raises(ValueError, match="pixel queries differ"):
        validate_spatialtrackerv2_prediction_contract(
            prediction,
            _query_points(),
            changed,
            expected_frame_count=4,
        )


def test_invalid_coordinates_are_removed_from_support(tmp_path: Path) -> None:
    coords = _coords().astype(float)
    coords[2, 1] = np.nan
    source = tmp_path / "invalid-coordinate.npz"
    np.savez(
        source,
        coords_world_m=coords,
        valid=np.ones((4, 2), dtype=bool),
        visibility_probability=np.full((4, 2), 0.9),
        confidence=np.full((4, 2), 0.8),
        query_points=_query_points(),
        query_pixels_xyt=_query_pixels(),
    )

    prediction = load_spatialtrackerv2_prediction(source)

    assert not prediction.valid[2, 1]


def test_seal_parser_has_no_score_target_or_cue_argument() -> None:
    args = build_parser().parse_args(
        [
            "seal-prediction",
            "--protocol",
            "protocol.json",
            "--input-manifest",
            "input.json",
            "--spatialtrackerv2-result",
            "result.npz",
            "--output-dir",
            "output",
        ]
    )

    assert not hasattr(args, "case_dir")
    assert not hasattr(args, "cues")
    assert not hasattr(args, "gt_track")
    assert not hasattr(args, "manual_tracks")
