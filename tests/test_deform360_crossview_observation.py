from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_bias_aware_prospective_artifacts import (
    canonical_sha256,
    file_sha256,
)
from bayesian_phystwin.deform360_crossview_observation import (
    ARCHIVE_FILENAME,
    ARTIFACT_KIND,
    MANIFEST_FILENAME,
    PROTOCOL_ID,
    align_camera_tracks,
    load_crossview_track_supplement,
)


def test_align_camera_tracks_preserves_common_material_order() -> None:
    tracks, visible = align_camera_tracks(
        np.asarray([11, 4, 9]),
        np.asarray([9, 11]),
        np.asarray([[90.0, 91.0], [110.0, 111.0]]),
        np.asarray([True, False]),
    )

    np.testing.assert_allclose(tracks[0], [110.0, 111.0])
    assert np.all(np.isnan(tracks[1]))
    np.testing.assert_allclose(tracks[2], [90.0, 91.0])
    np.testing.assert_array_equal(visible, [False, False, True])


def _write_artifact(root: Path) -> None:
    arrays = {
        "track_pixels_xy": np.zeros((2, 4, 3, 2), dtype=np.float32),
        "track_visibility": np.ones((2, 4, 3), dtype=bool),
        "frame_zero_pixels_xy": np.zeros((4, 3, 2), dtype=np.float32),
        "frame_zero_support": np.ones((4, 3), dtype=bool),
        "center_ids": np.asarray([2, 5, 8]),
        "selected_cameras": np.asarray(["a", "b", "c", "d"]),
        "update_frames": np.asarray([1, 2]),
        "center_frame_zero_points_m": np.zeros((3, 3), dtype=np.float32),
        "intrinsics": np.repeat(np.eye(3)[None], 4, axis=0),
        "camera_to_world": np.repeat(np.eye(4)[None], 4, axis=0),
    }
    root.mkdir()
    archive = root / ARCHIVE_FILENAME
    np.savez_compressed(archive, **arrays)
    manifest = {
        "artifact_kind": ARTIFACT_KIND,
        "protocol_id": PROTOCOL_ID,
        "output": {"archive_file_sha256": file_sha256(archive)},
    }
    manifest["result_sha256"] = canonical_sha256(
        manifest, digest_key="result_sha256"
    )
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )


def test_crossview_supplement_loads_only_checksummed_arrays(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)

    manifest, arrays = load_crossview_track_supplement(artifact)

    assert manifest["protocol_id"] == PROTOCOL_ID
    assert arrays["track_pixels_xy"].shape == (2, 4, 3, 2)


def test_crossview_supplement_rejects_archive_mutation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    _write_artifact(artifact)
    with (artifact / ARCHIVE_FILENAME).open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="archive checksum"):
        load_crossview_track_supplement(artifact)
