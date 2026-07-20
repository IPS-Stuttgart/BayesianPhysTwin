from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "remote"
    / "repair_deform360_selective_prediction_prefix_cadence.py"
)
SPEC = importlib.util.spec_from_file_location("deform360_cadence_repair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


def _write_test_video(path: Path, frame_count: int, ffmpeg: Path) -> None:
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=30",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )


def _read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        okay, frame = capture.read()
    finally:
        capture.release()
    assert okay
    return frame


def test_exact_trim_materializes_every_selected_frame(tmp_path: Path) -> None:
    executable = os.environ.get("DEFORM360_TEST_FFMPEG") or shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("FFmpeg is unavailable")
    ffmpeg = Path(executable)
    source = tmp_path / "source.mp4"
    destination = tmp_path / "trimmed.mp4"
    _write_test_video(source, 70, ffmpeg)

    REPAIR._trim_video_exact(
        ffmpeg,
        source,
        destination,
        start=5,
        count=58,
    )

    assert REPAIR._decoded_frame_count(destination) == 58
    assert len(REPAIR._decoded_rgb_sha256(destination, 58)) == 64
    source_first = _read_frame(source, 5).astype(np.float32)
    output_first = _read_frame(destination, 0).astype(np.float32)
    assert float(np.mean(np.abs(source_first - output_first))) < 8.0
