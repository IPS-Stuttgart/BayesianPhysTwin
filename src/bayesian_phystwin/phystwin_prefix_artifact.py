"""Future-blind PhysTwin inputs for source-family fitting and validation."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


PREFIX_ARTIFACT_CONTRACT = "phystwin-observation-prefix-plus-hold-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _prefix_plus_hold(value: Any, prefix_end_frame: int) -> np.ndarray:
    array = np.asarray(value)
    return np.concatenate(
        (array[:prefix_end_frame], array[prefix_end_frame - 1 : prefix_end_frame]),
        axis=0,
    )


def build_phystwin_prefix_artifact(
    final_data_path: str | Path,
    gt_track_path: str | Path,
    released_trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    prefix_end_frame: int,
) -> dict[str, object]:
    """Write a fitting payload that contains no future object observation.

    One synthetic hold frame follows the prefix so existing split-aware refit
    code can keep ``train_end_frame < frame_count``. The hold repeats the last
    permitted state and control, while visibility and motion-valid masks are
    false. It is a software sentinel, not an observed or scored future frame.
    """

    final_data = _load_pickle(final_data_path)
    gt_track = np.asarray(_load_pickle(gt_track_path))
    released = np.asarray(_load_pickle(released_trajectory_path))
    if not isinstance(final_data, dict):
        raise ValueError("final_data must contain a dictionary")
    required = {
        "object_points",
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
        "surface_points",
        "interior_points",
    }
    missing = required - set(final_data)
    if missing:
        raise ValueError("final_data is missing keys: " + ", ".join(sorted(missing)))
    object_points = np.asarray(final_data["object_points"])
    frame_count = len(object_points)
    if not 2 < prefix_end_frame < frame_count:
        raise ValueError("prefix_end_frame must lie in (2, frame_count)")
    for name in (
        "object_visibilities",
        "object_motions_valid",
        "controller_points",
    ):
        if len(np.asarray(final_data[name])) != frame_count:
            raise ValueError(f"{name} must share the object frame count")
    if len(gt_track) != frame_count or len(released) < prefix_end_frame:
        raise ValueError("track and released trajectory must cover the prefix")

    prefix_data = {
        "object_points": _prefix_plus_hold(object_points, prefix_end_frame),
        "object_visibilities": _prefix_plus_hold(
            final_data["object_visibilities"], prefix_end_frame
        ),
        "object_motions_valid": _prefix_plus_hold(
            final_data["object_motions_valid"], prefix_end_frame
        ),
        "controller_points": _prefix_plus_hold(
            final_data["controller_points"], prefix_end_frame
        ),
        "surface_points": np.asarray(final_data["surface_points"]).copy(),
        "interior_points": np.asarray(final_data["interior_points"]).copy(),
    }
    prefix_data["object_visibilities"][-1] = False
    prefix_data["object_motions_valid"][-1] = False
    prefix_track = _prefix_plus_hold(gt_track, prefix_end_frame)
    prefix_released = _prefix_plus_hold(released, prefix_end_frame)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "final_data": output / "final_data_prefix.pkl",
        "gt_track_3d": output / "gt_track_3d_prefix.pkl",
        "released_trajectory": output / "released_trajectory_prefix.pkl",
    }
    for name, value in (
        ("final_data", prefix_data),
        ("gt_track_3d", prefix_track),
        ("released_trajectory", prefix_released),
    ):
        with paths[name].open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "schema_version": 1,
        "contract": PREFIX_ARTIFACT_CONTRACT,
        "prefix_end_frame": int(prefix_end_frame),
        "output_frame_count": int(prefix_end_frame + 1),
        "hold_frame_index": int(prefix_end_frame),
        "claim_boundary": (
            "The fitting payload contains permitted prefix observations plus one "
            "unobserved repeated hold sentinel; no future object observation or "
            "future controller value is copied."
        ),
        "inputs": {
            "final_data": {
                "path": str(Path(final_data_path).resolve()),
                "sha256": _sha256(final_data_path),
            },
            "gt_track_3d": {
                "path": str(Path(gt_track_path).resolve()),
                "sha256": _sha256(gt_track_path),
            },
            "released_trajectory": {
                "path": str(Path(released_trajectory_path).resolve()),
                "sha256": _sha256(released_trajectory_path),
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
