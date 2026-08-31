#!/usr/bin/env python3
"""Common-coordinate wrapper for the registered Deform360 TCN baseline.

The frozen same-object predictors legitimately operate on each object's available
tactile carriers. A global temporal network instead requires one shared coordinate
system. This wrapper maps every released tactile carrier into four named 96-value
sensor slots and fills absent slots with zeros. It also binds every source/target
episode to the exact development and confirmation rosters retained by the earlier
frozen evidence. It does not modify the frozen v3 predictor, horizon, or metric.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / "run_deform360_tcn_baseline_v6_base.py"
BINDING_SHA256 = "60f27cb3addcbd0bb7e353a6c36d9d1082baf53d1dd7b7942786157c86ffca5d"
BINDING_ZLIB_BASE64 = "eNrlnc1y2zYQgO95Co7P3pb/pHps7z31lulwQHApogIBFgDlyJm8ewFaipxpM7UjRVK89gU2QBIfdoFdLEDw47souutwi1JPIyrXWD0bjo2ZVSO6u1+izP/EeVHUcX0fyuKWyZk5odV/Fi2qalWV5VJUqF6b8alsq2fVMbPzxT76vHCjSVjdob/UNh0ascWu6Y0eG/wgrBNq3egJlf8nbkWHiqO/1JkZ758uH5lQzYjOCN5o0wzaiEf/HD4wtcZQm55JeyhscJKM4wLIpNQP/yrhmFmja/TsuB7RNrP1T3a6sSiRu+ZZZQ8X+us+LZi6/SsU4Z7Q+cwkLp/9OxR/vzziCdtncO0rG0o+b/e7+0P2/nZLi97FcQLGt8Mxe9/oX1bo/T43iuL7z8nkmEyPyeyYzI/J4pgsj8nqmKz3qT8/V2TfYseK+HqslsxP96cApwswWCE3hKgz4KyVhORcpcC1c1oBl3pAK4X6PvRfQb4UZ52A9YOU12hi/biGSahNkK4b6GCnJVjNyWFnybPeTAq8BMWmjaAHXoN37QhSt96T3EDL1udhf1HNs9NrXmRg/54RH9GcWOVbl1FRe1Mr5b5f0tHNMoM+TG3MG0E+zjS/6mPkYOa2ReO7o+roYBfgBoOMEHEJ/UxIryvokbmBUldewQ5DZOYsVvXGYZMYtAlxqjflPr2AO4G1QVTUsFMY3pir/ALoDNZSb8mpeL7H3gq1k1eFvxRxAUyYs82HfhxJl2AHYcj16wr4gNa2mpmOGnsNzjA7EFT21ZF8ksw6wYm1QBoDn11Yl7xc3OvKxAnQU/Q0+Gqq26Dhg8CeDnfmBY1MBf0mJvHgsTFrn/jRUMMvll5uRyYlNfQK5rE1KCWjONLV8CgmqS/PfQ3YFfSajcREnMVLYJyRCzR58EXcvSQXd8lScMiuMZxdKuCQZd4Dtxtqcs0/T7yubKsuRlzAQ9g3RUzOFYx6uv3AYX066Qraju1CbPxbQNPblmMew8QmctOJPIEHwyaa7Cko5BsnkELcP8+g02s6ws09rpR0eAvg3zYw/5i4JdhJqzXSIa7A+D+ZWkskx15DOzvnLRQ1cO9zsVeMYje+RFHE0OoPQp3RDN2i1IoE+NzS0dIiPezr7Ga+OXFycOusOTjtncXJz+1PlXBxoZlbUYBE52Z+IUf3GmIpYWQbnKfXWohbV7cKNsLxARU101eswA6aDm/pbWPYJE5sEl4GWznRwU3DEo//pUOcB5+PDm4Bk3CMZl+uQe2kVm/p1en/p67K52812Utyv4iwOp2wggHH6SJivYYAa+iFHcJWqu92qsFNKu4K+MCEghEdoxMprVPvcRjHBLX3WuosvAytNniePRY3iZgDm8wVjjW4GGB5OLjBcmZ6ahpcfbGpopUzte0GdQ1WsQ0dI7WKwZKS8CoNBz8IY5COSV4dDrsIZ2pBb2ZBZyVzlYM564rIrfMW8MAcmhGlVnSoS9Dc6Wm2dJC9sRbhiAA6xDW0yPiwrNnSckySOIZpniZ0dIjTcFIgWvu6JfofHjsDwREmxjfUVNxPLv14tvsW4Oy1q75nqG7h5cN2BLYmJ3Hp++Bm/67vm95ln8QVjMI5al1vBZOevWWlhZ3EYQ38IezMZhM19gQGZF1YUzk3ef4ihPJ0hBQejLDuezBkVx2Fkgy4lpIZOuqYQ+8NjO+Ky/sSbzS+nCTFIb68ZvMjUhtzShh2ndFrlN7BdafYm/Q8i7en+0ZJBe3chrjaVRT3ygKtAdVyEis5VV6BRXaFKMRr9iS825e7m2YzaRs+GXL3q1Bd5AaMuB4ndMKJLUZ//PZ7xOZOuMjpJRM/MO6iJ5qIhSuWp0T7p0RGW4fGRuG7IFG7W67pjX5EFfmR/KmpomeH9C73WKWHHK5VL/bfQokOHzT5aWnGO8sHHFmoast2aAVTMA076x6E+rnD8AmVrIzBcQX7ykDrkcLkZFs+v0Oz9RUMEgyuxqd3/wBQHoDW"

SPEC = importlib.util.spec_from_file_location(
    "deform360_tcn_baseline_v6_base",
    BASELINE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import registered TCN baseline: {BASELINE_PATH}")
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)

binding_bytes = zlib.decompress(base64.b64decode(BINDING_ZLIB_BASE64))
if hashlib.sha256(binding_bytes).hexdigest() != BINDING_SHA256:
    raise RuntimeError("Deform360 episode binding digest changed")
BINDING = json.loads(binding_bytes)
if BINDING.get("schema") != "bayesian-phystwin/deform360-tcn-episode-binding-v6":
    raise RuntimeError("unexpected Deform360 episode binding schema")
if BINDING.get("object_count") != 106:
    raise RuntimeError("Deform360 episode binding object count changed")
EPISODE_BINDING = {
    str(item["object_id"]): {
        "cohort": str(item["cohort"]),
        "source_episode_ids": tuple(int(value) for value in item["source_episode_ids"]),
        "target_episode_id": int(item["target_episode_id"]),
    }
    for item in BINDING["objects"]
}
if len(EPISODE_BINDING) != 106:
    raise RuntimeError("Deform360 episode binding contains duplicate objects")

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


def bound_descriptors(
    root: Path,
    object_id: str,
    minimum_episodes: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], Any]:
    binding = EPISODE_BINDING.get(object_id)
    if binding is None:
        raise core.base.EvaluationError(
            f"object absent from frozen episode binding: {object_id}"
        )
    discovered = core.base.discover_object(root, object_id, minimum_episodes)
    by_id = {int(item.episode_id): item for item in discovered}
    source_ids = tuple(binding["source_episode_ids"])
    target_id = int(binding["target_episode_id"])
    required_ids = source_ids + (target_id,)
    missing = [episode_id for episode_id in required_ids if episode_id not in by_id]
    if missing:
        raise core.base.EvaluationError(
            f"frozen episode binding unavailable for {object_id}: missing={missing}"
        )
    source_descriptors = tuple(by_id[episode_id] for episode_id in source_ids)
    target_descriptor = by_id[target_id]
    descriptors = source_descriptors + (target_descriptor,)
    return descriptors, source_descriptors, target_descriptor


def discover_source_object(
    root: Path,
    object_id: str,
    minimum_episodes: int,
    base_protocol: Mapping[str, Any],
    horizon: int,
    history_frames: int,
    action_samples: int,
) -> Any:
    descriptors, source_descriptors, target_descriptor = bound_descriptors(
        root,
        object_id,
        minimum_episodes,
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
        descriptors=descriptors,
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


_original_run = core.run


def run(protocol_path: Path, root: Path) -> dict[str, Any]:
    result = _original_run(protocol_path, root)
    result["episode_binding"] = {
        "embedded_sha256": BINDING_SHA256,
        "object_count": 106,
        "development_source_run_id": BINDING["development_source_run_id"],
        "evaluation_source_run_id": BINDING["evaluation_source_run_id"],
        "source_and_target_episode_rosters_changed": False,
    }
    return result


core.discover_source_object = discover_source_object
core.load_target_samples = load_target_samples
core.run = run

if __name__ == "__main__":
    raise SystemExit(core.main())
