import copy
import hashlib
import pickle

import numpy as np
import pytest

from bayesian_phystwin.phystwin_prefix_artifact import (
    build_phystwin_prefix_artifact,
)


def _write_pickle(path, value):
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path, suffix, future_shift=0.0):
    frames = 7
    points = np.arange(frames * 2 * 3, dtype=np.float32).reshape(frames, 2, 3)
    points[4:] += future_shift
    controller = np.arange(frames * 3, dtype=np.float32).reshape(frames, 1, 3)
    controller[4:] += future_shift
    final_data = {
        "object_points": points,
        "object_visibilities": np.ones((frames, 2), dtype=bool),
        "object_motions_valid": np.ones((frames, 2), dtype=bool),
        "controller_points": controller,
        "surface_points": np.ones((3, 3), dtype=np.float32),
        "interior_points": np.ones((1, 3), dtype=np.float32),
        "unused_future_field": np.full((frames, 4), future_shift),
    }
    gt = copy.deepcopy(points)
    released = copy.deepcopy(points)
    final_path = tmp_path / f"final_{suffix}.pkl"
    gt_path = tmp_path / f"gt_{suffix}.pkl"
    released_path = tmp_path / f"released_{suffix}.pkl"
    _write_pickle(final_path, final_data)
    _write_pickle(gt_path, gt)
    _write_pickle(released_path, released)
    return final_path, gt_path, released_path


def test_prefix_artifact_is_future_blind_and_masks_hold_frame(tmp_path):
    first = _inputs(tmp_path, "first", future_shift=0.0)
    second = _inputs(tmp_path, "second", future_shift=1000.0)

    build_phystwin_prefix_artifact(
        *first,
        tmp_path / "out_first",
        prefix_end_frame=4,
    )
    build_phystwin_prefix_artifact(
        *second,
        tmp_path / "out_second",
        prefix_end_frame=4,
    )

    for name in (
        "final_data_prefix.pkl",
        "gt_track_3d_prefix.pkl",
        "released_trajectory_prefix.pkl",
    ):
        assert _digest(tmp_path / "out_first" / name) == _digest(
            tmp_path / "out_second" / name
        )
    with (tmp_path / "out_first" / "final_data_prefix.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert len(payload["object_points"]) == 5
    np.testing.assert_array_equal(payload["object_points"][-1], payload["object_points"][-2])
    assert not np.any(payload["object_visibilities"][-1])
    assert not np.any(payload["object_motions_valid"][-1])
    assert "unused_future_field" not in payload


def test_prefix_artifact_rejects_inconsistent_frame_counts(tmp_path):
    paths = _inputs(tmp_path, "bad")
    with paths[0].open("rb") as handle:
        payload = pickle.load(handle)
    payload["controller_points"] = payload["controller_points"][:-1]
    _write_pickle(paths[0], payload)

    with pytest.raises(ValueError, match="controller_points"):
        build_phystwin_prefix_artifact(
            *paths,
            tmp_path / "out",
            prefix_end_frame=4,
        )
