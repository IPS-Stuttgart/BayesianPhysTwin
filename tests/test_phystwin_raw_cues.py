import pickle
from pathlib import Path

import numpy as np

from bayesian_phystwin.phystwin_raw_cues import build_phystwin_raw_camera_cues


def test_raw_camera_cues_map_tracks_and_measure_boundary(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "cotracker").mkdir(parents=True)
    (raw / "pcd").mkdir()
    (raw / "mask").mkdir()
    tracks = np.array(
        [
            [[1.0, 1.0], [2.0, 2.0]],
            [[1.0, 2.0], [2.0, 2.0]],
        ]
    )
    visibility = np.array([[True, True], [True, False]])
    np.savez(raw / "cotracker" / "0.npz", tracks=tracks, visibility=visibility)
    points = np.zeros((1, 4, 4, 3))
    points[0, 1, 1] = [0.01, 0.01, 0.0]
    points[0, 2, 2] = [0.02, 0.02, 0.0]
    np.savez(raw / "pcd" / "0.npz", points=points)
    object_mask = np.zeros((4, 4), dtype=bool)
    object_mask[1:3, 1:3] = True
    masks = {
        frame: {0: {"object": object_mask, "controller": ~object_mask}}
        for frame in range(2)
    }
    with (raw / "mask" / "processed_masks.pkl").open("wb") as handle:
        pickle.dump(masks, handle)
    final = {
        "object_points": np.array(
            [
                [[0.02, 0.02, 0.0], [0.01, 0.01, 0.0]],
                [[0.02, 0.02, 0.0], [0.01, 0.01, 0.0]],
            ]
        ),
        "object_visibilities": np.ones((2, 2), dtype=bool),
    }
    final_path = tmp_path / "final.pkl"
    with final_path.open("wb") as handle:
        pickle.dump(final, handle)

    summary = build_phystwin_raw_camera_cues(
        final_path,
        raw,
        tmp_path / "cues.npz",
    )

    with np.load(tmp_path / "cues.npz") as cues:
        np.testing.assert_array_equal(cues["source_track"], [1, 0])
        np.testing.assert_array_equal(cues["raw_visibility"][1], [False, True])
        assert np.all(cues["boundary_distance"] >= 0.0)
    assert summary["mapping"]["maximum_distance_m"] == 0.0
