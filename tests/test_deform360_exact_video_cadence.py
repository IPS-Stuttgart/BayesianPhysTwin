from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

from bayesian_phystwin.deform360_exact_video_cadence import (
    FROZEN_AUTHORIZED_FUTURE_STAGE_SHA256,
    append_tail_to_prefix_exact_30hz,
    decoded_prefix_sha256,
    decoded_frame_count,
    frozen_authorized_future_stage_path,
    sha256,
    trim_video_exact_30hz,
)


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


def test_frozen_authorized_future_stage_hash_is_unchanged() -> None:
    assert (
        sha256(frozen_authorized_future_stage_path())
        == FROZEN_AUTHORIZED_FUTURE_STAGE_SHA256
    )


def test_exact_tail_trim_materializes_all_23_frames(tmp_path: Path) -> None:
    executable = os.environ.get("DEFORM360_TEST_FFMPEG") or shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("FFmpeg is unavailable")
    ffmpeg = Path(executable)
    source = tmp_path / "source.mp4"
    tail = tmp_path / "tail.mp4"
    _write_test_video(source, 100, ffmpeg)

    trim_video_exact_30hz(ffmpeg, source, tail, start=58, count=23)

    assert decoded_frame_count(tail) == 23


def test_exact_append_preserves_prefix_and_materializes_81_frames(
    tmp_path: Path,
) -> None:
    executable = os.environ.get("DEFORM360_TEST_FFMPEG") or shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("FFmpeg is unavailable")
    ffmpeg = Path(executable)
    source = tmp_path / "source.mp4"
    prefix = tmp_path / "prefix.mp4"
    authorized = tmp_path / "authorized.mp4"
    _write_test_video(source, 100, ffmpeg)
    trim_video_exact_30hz(ffmpeg, source, prefix, start=0, count=58)
    prefix_digest = decoded_prefix_sha256(ffmpeg, prefix, 58)

    append_tail_to_prefix_exact_30hz(
        ffmpeg,
        prefix,
        source,
        authorized,
        source_start=0,
        prefix_frame_count=58,
        raw_frame_count=81,
    )

    assert decoded_frame_count(authorized) == 81
    assert decoded_prefix_sha256(ffmpeg, authorized, 58) == prefix_digest


@pytest.mark.parametrize("start,count", [(-1, 23), (58, 0)])
def test_exact_tail_trim_rejects_invalid_ranges(
    tmp_path: Path,
    start: int,
    count: int,
) -> None:
    with pytest.raises(ValueError, match="frame range"):
        trim_video_exact_30hz(
            Path("ffmpeg"),
            tmp_path / "missing.mp4",
            tmp_path / "tail.mp4",
            start=start,
            count=count,
        )
