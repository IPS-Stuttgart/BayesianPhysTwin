from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from bayesian_phystwin.deform360_joint_sparse_geometric_npz_v4 import (
    _load_prediction_support_windows,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    ROOT
    / "protocols/locks/"
    "deform360_official_hub_joint_sparse_geometric_materializer_v4.json"
)


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def _write_prediction_fixture(root: Path) -> tuple[Path, str]:
    revision = str(json.loads(POLICY_PATH.read_text())["motioncrafter_revision"])
    window = root / "window.npz"
    with zipfile.ZipFile(window, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("window_id.npy", _npy_bytes(np.asarray("window-0")))
        archive.writestr(
            "frame_indices.npy",
            _npy_bytes(np.asarray([10, 11], dtype=np.int64)),
        )
        archive.writestr(
            "valid_mask.npy",
            _npy_bytes(np.ones((2, 2, 3), dtype=np.bool_)),
        )
    run_spec = {"fixture": "causal-coverage"}
    canonical = json.dumps(
        run_spec,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest = {
        "format_version": 1,
        "motioncrafter_commit": revision,
        "overlap_windows": [
            {
                "window_id": "window-0",
                "path": window.name,
                "start_frame": 10,
                "stop_frame": 12,
            }
        ],
        "artifact_integrity": {
            "schema": "prob4d.motioncrafter-artifact-integrity.v1",
            "run_spec": run_spec,
            "run_spec_sha256": hashlib.sha256(canonical).hexdigest(),
            "members": [
                {
                    "path": window.name,
                    "sha256": hashlib.sha256(window.read_bytes()).hexdigest(),
                    "bytes": window.stat().st_size,
                    "kind": "independently_decoded_overlap_window",
                }
            ],
        },
    }
    path = root / "predictions.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path, revision


def test_prediction_windows_cover_every_registered_causal_frame(
    tmp_path: Path,
) -> None:
    path, revision = _write_prediction_fixture(tmp_path)
    windows, _ = _load_prediction_support_windows(
        path,
        causal_range=(10, 12),
        image_shape=(2, 3),
        expected_motioncrafter_revision=revision,
    )
    assert [window.window_id for window in windows] == ["window-0"]

    with pytest.raises(ValueError, match="complete causal range"):
        _load_prediction_support_windows(
            path,
            causal_range=(10, 13),
            image_shape=(2, 3),
            expected_motioncrafter_revision=revision,
        )
