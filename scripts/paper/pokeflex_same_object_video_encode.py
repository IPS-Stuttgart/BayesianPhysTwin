"""H.264 encoder for the deterministic PokeFlex paper video."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_video(
    path: Path,
    rgb_frames: Iterable[np.ndarray],
    *,
    fps: int,
    repeats_per_state: int,
) -> dict[str, Any]:
    import imageio_ffmpeg

    iterator = iter(rgb_frames)
    try:
        first = next(iterator)
    except StopIteration as error:
        raise ValueError("no rendered frames are available") from error
    height, width, channels = first.shape
    _require(channels == 3, "rendered frame is not RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=16,
        ffmpeg_log_level="warning",
        output_params=["-movflags", "+faststart", "-crf", "20"],
    )
    writer.send(None)
    start_hold = fps
    end_hold = fps
    for _ in range(start_hold):
        writer.send(first.tobytes())
    state_count = 0
    last = first
    for frame in chain((first,), iterator):
        _require(frame.shape == first.shape, "rendered frame dimensions changed")
        for _ in range(repeats_per_state):
            writer.send(frame.tobytes())
        state_count += 1
        last = frame
    for _ in range(end_hold):
        writer.send(last.tobytes())
    writer.close()
    frame_count = start_hold + state_count * repeats_per_state + end_hold
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "source_state_count": state_count,
        "encoded_frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "codec": "H.264/libx264",
        "pixel_format": "yuv420p",
    }
