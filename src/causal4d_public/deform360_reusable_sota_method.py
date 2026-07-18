"""Locked numerical method for reusable-PhysTwin Deform360 evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping


METHOD_SCHEMA_VERSION = 1
METHOD_PROTOCOL_ID = "deform360-reusable-sota-method-v1"
PARENT_CONFIG_SHA256 = (
    "64e41773d3e333987dc97a11c52e3474fd34fe65139e6c7d799b3e8f4db188cd"
)
WINDOW_CONFIG_SHA256 = (
    "331faa06dd3d81d1ddea615828eae2b4a040c3a56b92058e5b037a6129f1f9c8"
)
CANONICAL_METHOD_CONFIG_SHA256 = (
    "54a865cb4382410c07501d7b6b8c512ed60a2f60d43c92be774e22f0b1427b5c"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reusable_sota_method_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reusable_sota_physical_candidates(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    grid = payload["config"]["physical_grid"]
    candidates = []
    for spring_y, drag, dashpot in itertools.product(
        grid["init_spring_y"],
        grid["drag_damping"],
        grid["dashpot_damping"],
    ):
        candidates.append(
            {
                "label": (
                    f"y{int(spring_y)}-drag{int(drag)}-dash{int(dashpot)}"
                ),
                "init_spring_y": float(spring_y),
                "drag_damping": float(drag),
                "dashpot_damping": float(dashpot),
            }
        )
    return tuple(candidates)


def validate_reusable_sota_method(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == METHOD_SCHEMA_VERSION,
        "unsupported reusable-SOTA method schema",
    )
    observed = reusable_sota_method_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "reusable-SOTA method checksum mismatch",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == METHOD_PROTOCOL_ID
        and config.get("parent_config_sha256") == PARENT_CONFIG_SHA256
        and config.get("window_config_sha256") == WINDOW_CONFIG_SHA256,
        "reusable-SOTA method parent changed",
    )
    timing = config.get("lock_timing", {})
    _require(
        timing.get("fit_prediction_metrics_read") is False
        and timing.get("held_future_outcomes_read") is False
        and timing.get("confirmatory_objects_opened") is False,
        "reusable-SOTA method was not locked outcome-blind",
    )
    grid = config.get("physical_grid", {})
    _require(
        tuple(grid.get("init_spring_y", ())) == (10000.0, 30000.0, 50000.0)
        and tuple(grid.get("drag_damping", ())) == (1.0, 3.0, 10.0)
        and tuple(grid.get("dashpot_damping", ())) == (50.0, 100.0),
        "reusable-SOTA physical grid changed",
    )
    candidates = reusable_sota_physical_candidates(payload)
    _require(
        grid.get("candidate_count") == len(candidates) == 18,
        "reusable-SOTA candidate count changed",
    )
    warp = config.get("official_warp", {})
    _require(
        warp.get("revision") == "2b6630528141b9cba5a7677c8b88b2129b4a8390"
        and warp.get("real_config_sha256")
        == "a40a5ec2f5c978c1290810f20ed56db7cab99dc0c227adfe6b7434dfc95ead48"
        and warp.get("controller_radius_m") == 0.03
        and warp.get("controller_max_neighbours") == 1
        and warp.get("canonical_controller_patch_size") == 16
        and warp.get("support_dynamics") == "official-ground",
        "reusable-SOTA Warp settings changed",
    )
    bank = config.get("prediction_bank", {})
    selection = config.get("selection", {})
    _require(
        bank.get("frame_count") == 76
        and bank.get("candidate_selection_uses_fit_episodes_only") is True
        and bank.get("all_held_candidate_predictions_hashed_before_outcome_reveal")
        is True
        and bank.get("held_future_object_or_tactile_used") is False
        and selection.get("held_outcome_refit_forbidden") is True,
        "reusable-SOTA prediction boundary changed",
    )
    _require(
        observed == CANONICAL_METHOD_CONFIG_SHA256,
        "reusable-SOTA method differs from the canonical lock",
    )
    return {
        "passed": True,
        "protocol_id": METHOD_PROTOCOL_ID,
        "config_sha256": observed,
        "candidate_count": len(candidates),
    }


def load_reusable_sota_method(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "reusable-SOTA method must be an object")
    validate_reusable_sota_method(payload)
    return payload


__all__ = [
    "CANONICAL_METHOD_CONFIG_SHA256",
    "METHOD_PROTOCOL_ID",
    "load_reusable_sota_method",
    "reusable_sota_method_sha256",
    "reusable_sota_physical_candidates",
    "validate_reusable_sota_method",
]
