#!/usr/bin/env python3
"""Run frozen target staging with exact FFmpeg source-frame cadence."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import ModuleType

from bayesian_phystwin.deform360_exact_video_cadence import (
    FROZEN_AUTHORIZED_FUTURE_STAGE_SHA256,
    append_tail_to_prefix_exact_30hz,
    decoded_prefix_sha256,
    decoded_frame_count,
    frozen_authorized_future_stage_path,
    sha256,
    trim_video_exact_30hz,
)


PREFIX_FRAME_COUNT = 58
RAW_FRAME_COUNT = 81


def _load_frozen_stage(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deform360_selective_authorized_future_frozen_stage",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen authorized-future stage: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ffmpeg_version(ffmpeg: Path) -> str:
    return subprocess.run(
        [str(ffmpeg), "-version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()[0]


def main() -> int:
    frozen_path = frozen_authorized_future_stage_path()
    frozen_sha256 = sha256(frozen_path)
    if frozen_sha256 != FROZEN_AUTHORIZED_FUTURE_STAGE_SHA256:
        raise ValueError("authorized-future cadence base differs from the frozen stage")
    executable = os.environ.get("DEFORM360_FFMPEG") or shutil.which("ffmpeg")
    if executable is None:
        raise FileNotFoundError("FFmpeg is unavailable on PATH")
    ffmpeg = Path(executable).expanduser().resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"FFmpeg executable is missing: {ffmpeg}")
    frozen = _load_frozen_stage(frozen_path)
    def exact_trim(source: Path, destination: Path, start: int, count: int) -> None:
        trim_video_exact_30hz(ffmpeg, source, destination, start, count)

    def checked_append(
        prefix: Path,
        source: Path,
        destination: Path,
        *,
        source_start: int,
    ) -> None:
        prefix_count = decoded_frame_count(prefix)
        if prefix_count != PREFIX_FRAME_COUNT:
            raise ValueError(
                f"sealed prefix has {prefix_count} rather than "
                f"{PREFIX_FRAME_COUNT} frames: {prefix}"
            )
        append_tail_to_prefix_exact_30hz(
            ffmpeg,
            prefix,
            source,
            destination,
            source_start=source_start,
            prefix_frame_count=PREFIX_FRAME_COUNT,
            raw_frame_count=RAW_FRAME_COUNT,
        )

    def exact_decoded_prefix_sha256(path: Path, frame_count: int) -> str:
        return decoded_prefix_sha256(ffmpeg, path, frame_count)

    frozen._trim_video = exact_trim
    frozen._append_tail_to_prefix = checked_append
    frozen._decoded_prefix_sha256 = exact_decoded_prefix_sha256
    print(
        json.dumps(
            {
                "artifact_kind": (
                    "Deform360AuthorizedFutureCadenceOperationalAmendment"
                ),
                "frozen_authorized_future_stage_sha256": frozen_sha256,
                "cadence_helper_source_sha256": sha256(
                    Path(trim_video_exact_30hz.__code__.co_filename)
                ),
                "wrapper_source_sha256": sha256(Path(__file__).resolve()),
                "source_frame_selection_changed": False,
                "target_method_changed": False,
                "prefix_decoded_frame_count": PREFIX_FRAME_COUNT,
                "authorized_decoded_frame_count": RAW_FRAME_COUNT,
                "ffmpeg_path": str(ffmpeg),
                "ffmpeg_sha256": sha256(ffmpeg),
                "ffmpeg_version": _ffmpeg_version(ffmpeg),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return int(frozen.main())


if __name__ == "__main__":
    raise SystemExit(main())
