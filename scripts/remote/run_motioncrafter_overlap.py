"""Run pinned MotionCrafter while honoring its advertised window overlap."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from typing import Any

import fire

import run as motioncrafter_run
from motioncrafter.determ_ppl import MotionCrafterDetermPipeline
from motioncrafter.diff_ppl import MotionCrafterDiffPipeline


MOTIONCRAFTER_REVISION = "1d6a8947ec6ebabbcf4fc1e0f6d06828fcf6f257"


def _verify_revision(motioncrafter_root: str | Path) -> None:
    root = Path(motioncrafter_root).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != MOTIONCRAFTER_REVISION:
        raise RuntimeError(
            f"MotionCrafter revision {revision} does not match "
            f"{MOTIONCRAFTER_REVISION}"
        )


def main(
    *,
    motioncrafter_root: str = "/home/florianpfaff/MotionCrafter",
    overlap: int = 5,
    **kwargs: Any,
) -> None:
    """Forward Fire arguments to upstream while repairing overlap forwarding."""

    if overlap < 0:
        raise ValueError("overlap must be nonnegative")
    _verify_revision(motioncrafter_root)
    pipeline_class = (
        MotionCrafterDetermPipeline
        if kwargs.get("model_type", "diff") == "determ"
        else MotionCrafterDiffPipeline
    )
    original_call = pipeline_class.__call__

    @functools.wraps(original_call)
    def call_with_overlap(self: Any, *args: Any, **call_kwargs: Any) -> Any:
        call_kwargs["overlap"] = overlap
        return original_call(self, *args, **call_kwargs)

    pipeline_class.__call__ = call_with_overlap
    try:
        motioncrafter_run.main(overlap=overlap, **kwargs)
    finally:
        pipeline_class.__call__ = original_call


if __name__ == "__main__":
    fire.Fire(main)
