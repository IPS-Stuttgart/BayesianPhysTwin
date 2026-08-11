"""Exact source-frame materialization for pinned Deform360 video windows."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

import cv2


FROZEN_AUTHORIZED_FUTURE_STAGE_SHA256 = (
    "2ed2ffb0cd6ceeb2f08a485d578a7257701e92d94c1f4d8e9063843aacff778c"
)
FRAME_RATE_HZ = 30


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_authorized_future_stage_path() -> Path:
    """Locate the target-staging script guarded by the operational wrapper."""
    return (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "remote"
        / "stage_deform360_selective_authorized_future.py"
    )


def decoded_frame_count(path: Path) -> int:
    """Count frames that can actually be decoded, rather than trusting metadata."""
    capture = cv2.VideoCapture(str(path))
    try:
        return sum(1 for _ in iter(capture.grab, False))
    finally:
        capture.release()


def trim_video_exact_30hz(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    start: int,
    count: int,
) -> None:
    """Encode an exact inclusive source-frame selection at an explicit cadence."""
    if start < 0 or count < 1:
        raise ValueError("video frame range is invalid")
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            (
                f"select='between(n,{start},{start + count - 1})',"
                f"setpts=N/({FRAME_RATE_HZ}*TB)"
            ),
            "-frames:v",
            str(count),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "12",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(FRAME_RATE_HZ),
            "-fps_mode",
            "cfr",
            str(destination),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    actual = decoded_frame_count(destination)
    if actual != count:
        raise ValueError(
            f"exact video trim produced {actual} rather than {count} frames: "
            f"{destination}"
        )


def decoded_prefix_sha256(ffmpeg: Path, path: Path, frame_count: int) -> str:
    """Hash decoded RGB bytes for exactly the requested video prefix."""
    process = subprocess.Popen(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            str(frame_count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    if process.stdout is None:
        raise RuntimeError("FFmpeg output pipe is unavailable")
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    return_code = process.wait()
    if return_code != 0:
        raise ValueError(f"cannot decode video prefix: {path}")
    return digest.hexdigest()


def append_tail_to_prefix_exact_30hz(
    ffmpeg: Path,
    prefix: Path,
    source: Path,
    destination: Path,
    *,
    source_start: int,
    prefix_frame_count: int,
    raw_frame_count: int,
) -> None:
    """Append an exact source tail while preserving the encoded prefix bytes."""
    if not 0 < prefix_frame_count < raw_frame_count:
        raise ValueError("prefix and raw frame counts are invalid")
    prefix_segment = destination.with_name("sealed_prefix.mp4")
    tail_segment = destination.with_name("authorized_tail.mp4")
    concat_list = destination.with_name("concat.txt")
    shutil.copy2(prefix, prefix_segment)
    try:
        trim_video_exact_30hz(
            ffmpeg,
            source,
            tail_segment,
            source_start + prefix_frame_count,
            raw_frame_count - prefix_frame_count,
        )
        concat_list.write_text(
            f"file '{prefix_segment}'\nfile '{tail_segment}'\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(destination),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
        )
    finally:
        prefix_segment.unlink(missing_ok=True)
        tail_segment.unlink(missing_ok=True)
        concat_list.unlink(missing_ok=True)
    actual = decoded_frame_count(destination)
    if actual != raw_frame_count:
        raise ValueError(
            f"authorized video has {actual} rather than {raw_frame_count} frames: "
            f"{destination}"
        )


__all__ = [
    "FRAME_RATE_HZ",
    "FROZEN_AUTHORIZED_FUTURE_STAGE_SHA256",
    "append_tail_to_prefix_exact_30hz",
    "decoded_prefix_sha256",
    "decoded_frame_count",
    "frozen_authorized_future_stage_path",
    "sha256",
    "trim_video_exact_30hz",
]
