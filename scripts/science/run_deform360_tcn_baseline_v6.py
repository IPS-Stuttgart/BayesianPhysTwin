#!/usr/bin/env python3
"""Common-coordinate wrapper for the registered Deform360 TCN baseline.

The frozen same-object predictors legitimately operate on each object's available
tactile carriers.  A global temporal network instead requires one shared coordinate
system.  This wrapper maps every released tactile carrier into four named 96-value
sensor slots and fills absent slots with zeros.  It does not modify the frozen v3
predictor, target selection, horizon, or metric.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "run_deform360_tcn_baseline_v6_base.py"
SPEC = importlib.util.spec_from_file_location(
    "deform360_tcn_baseline_v6_base",
    BASELINE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import registered TCN baseline: {BASELINE_PATH}")
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)

SENSOR_SUFFIXES = (
    "tactilel_left",
    "tactilel_right",
    "tactiler_left",
    "tactiler_right",
)
SLOT_WIDTH = 96
COMMON_TACTILE_WIDTH = len(SENSOR_SUFFIXES) * SLOT_WIDTH


def canonicalize_episode(episode: Any) -> Any:
    paths = tuple(episode.descriptor.tactile_paths)
    if not paths:
        raise core.base.EvaluationError("episode has no tactile carrier")
    width = int(episode.tactile.shape[1])
    if width % len(paths):
        raise core.base.EvaluationError(
            f"tactile width {width} is not divisible by {len(paths)} carriers"
        )
    block_width = width // len(paths)
    if block_width != SLOT_WIDTH:
        raise core.base.EvaluationError(
            f"unexpected pooled tactile carrier width: {block_width}"
        )

    values = np.zeros(
        (len(episode.tactile), COMMON_TACTILE_WIDTH),
        dtype=np.float64,
    )
    occupied: set[int] = set()
    for block_index, path in enumerate(paths):
        name = path.parent.name.lower()
        matches = [
            slot
            for slot, suffix in enumerate(SENSOR_SUFFIXES)
            if name.endswith(suffix)
        ]
        if len(matches) != 1:
            raise core.base.EvaluationError(
                f"cannot assign tactile carrier to a canonical slot: {path}"
            )
        slot = matches[0]
        if slot in occupied:
            raise core.base.EvaluationError(
                f"duplicate canonical tactile slot {slot}: {path}"
            )
        occupied.add(slot)
        source = slice(block_index * SLOT_WIDTH, (block_index + 1) * SLOT_WIDTH)
        target = slice(slot * SLOT_WIDTH, (slot + 1) * SLOT_WIDTH)
        values[:, target] = episode.tactile[:, source]

    return core.base.EpisodeData(
        descriptor=episode.descriptor,
        tactile=values,
        robot_actions=episode.robot_actions,
        bimanual=episode.bimanual,
        fingerprints={
            **episode.fingerprints,
            "common_tactile_layout": {
                "sensor_suffixes": list(SENSOR_SUFFIXES),
                "slot_width": SLOT_WIDTH,
                "present_slots": sorted(occupied),
                "absent_slots_zero_filled": sorted(set(range(4)) - occupied),
            },
        },
    )


def discover_source_object(
    root: Path,
    object_id: str,
    minimum_episodes: int,
    base_protocol: Mapping[str, Any],
    horizon: int,
    history_frames: int,
    action_samples: int,
) -> Any:
    descriptors = core.base.discover_object(root, object_id, minimum_episodes)
    if len(descriptors) < minimum_episodes:
        raise core.base.EvaluationError(
            f"object {object_id} lacks required carriers"
        )
    target_descriptor = max(descriptors, key=lambda item: item.episode_id)
    source_descriptors = tuple(
        descriptor for descriptor in descriptors if descriptor is not target_descriptor
    )
    source = tuple(
        canonicalize_episode(core.base.load_episode(descriptor))
        for descriptor in source_descriptors
    )
    model = base_protocol["model"]
    feature_scale = core.base.feature_scale(
        source,
        float(model["source_feature_scale_quantile"]),
    )
    source_samples = core.concatenate_samples(
        [
            core.temporal_samples(
                episode,
                feature_scale,
                base_protocol,
                horizon,
                history_frames,
                action_samples,
            )
            for episode in source
        ]
    )
    return core.ObjectSource(
        object_id=object_id,
        descriptors=tuple(descriptors),
        source_descriptors=source_descriptors,
        target_descriptor=target_descriptor,
        source=source,
        feature_scale=feature_scale,
        source_samples=source_samples,
    )


def load_target_samples(
    source_object: Any,
    base_protocol: Mapping[str, Any],
    horizon: int,
    history_frames: int,
    action_samples: int,
) -> Any:
    target = canonicalize_episode(
        core.base.load_episode(source_object.target_descriptor)
    )
    return core.temporal_samples(
        target,
        source_object.feature_scale,
        base_protocol,
        horizon,
        history_frames,
        action_samples,
    )


core.discover_source_object = discover_source_object
core.load_target_samples = load_target_samples

if __name__ == "__main__":
    raise SystemExit(core.main())
