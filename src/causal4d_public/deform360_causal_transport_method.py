"""Locked source-development method for Deform360 causal contact transport."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping


METHOD_SCHEMA_VERSION = 1
METHOD_PROTOCOL_ID = "deform360-causal-contact-transport-source-v1"
PARENT_CONFIG_SHA256 = (
    "64e41773d3e333987dc97a11c52e3474fd34fe65139e6c7d799b3e8f4db188cd"
)
WINDOW_CONFIG_SHA256 = (
    "331faa06dd3d81d1ddea615828eae2b4a040c3a56b92058e5b037a6129f1f9c8"
)
CANONICAL_METHOD_CONFIG_SHA256 = (
    "6f70329aa2ecf0c7b3d7e7cc648f2d701ad6092c655d8a3970fc9a7f4adf3206"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def causal_transport_method_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    return hashlib.sha256(
        json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def causal_transport_candidates(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    config = payload["config"]
    candidates: list[dict[str, Any]] = [
        {
            "label": "persistence",
            "base_support_scale_m": float(config["candidate_grid"]["base_scale_m"][0]),
            "support_growth_per_travel": 0.0,
            "initial_contact_gain": 0.0,
            "acquired_contact_gain": 0.0,
            "transform_mode": "translation",
        }
    ]
    for (
        base_scale,
        growth,
        initial_gain,
        mode,
    ) in itertools.product(
        config["candidate_grid"]["base_scale_m"],
        config["candidate_grid"]["support_growth_per_travel"],
        config["candidate_grid"]["initial_contact_gain"],
        config["candidate_grid"]["transform_mode"],
    ):
        candidates.append(
            {
                "label": (
                    f"{mode}-s{int(round(1000 * base_scale))}mm-"
                    f"g{growth:g}-a{initial_gain:g}"
                ),
                "base_support_scale_m": float(base_scale),
                "support_growth_per_travel": float(growth),
                "initial_contact_gain": float(initial_gain),
                "acquired_contact_gain": float(
                    config["contact_policy"]["acquired_contact_gain"]
                ),
                "transform_mode": str(mode),
            }
        )
    return tuple(candidates)


def validate_causal_transport_method(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == METHOD_SCHEMA_VERSION,
        "unsupported causal-transport method schema",
    )
    observed = causal_transport_method_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "causal-transport method checksum mismatch",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == METHOD_PROTOCOL_ID
        and config.get("parent_config_sha256") == PARENT_CONFIG_SHA256
        and config.get("window_config_sha256") == WINDOW_CONFIG_SHA256,
        "causal-transport parent changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("source_outcomes_used_for_method_discovery") is True
        and boundary.get("development_held_future_read") is False
        and boundary.get("confirmatory_data_read") is False
        and boundary.get("pokeflex_target_read") is False,
        "causal-transport information boundary changed",
    )
    contact = config.get("contact_policy", {})
    _require(
        contact.get("controller_group_size") == 768
        and contact.get("maximum_contact_distance_m") == 0.01
        and contact.get("opening_contact_threshold_m") == 0.0795396
        and contact.get("confirmation_frames") == 1
        and contact.get("acquired_contact_gain") == 0.0,
        "causal contact policy changed",
    )
    grid = config.get("candidate_grid", {})
    _require(
        tuple(grid.get("base_scale_m", ())) == (0.003, 0.005, 0.01)
        and tuple(grid.get("support_growth_per_travel", ()))
        == (0.0, 0.1, 0.5, 2.0)
        and tuple(grid.get("initial_contact_gain", ())) == (0.5, 1.0)
        and tuple(grid.get("transform_mode", ())) == ("translation", "se3"),
        "causal-transport candidate grid changed",
    )
    candidates = causal_transport_candidates(payload)
    _require(
        len(candidates) == grid.get("candidate_count") == 49,
        "causal-transport candidate count changed",
    )
    gate = config.get("source_gate", {})
    _require(
        gate.get("minimum_leave_one_action_out_persistence_win_fraction") == 2 / 3
        and gate.get("minimum_leave_one_action_out_single_median_win_fraction")
        == 2 / 3
        and gate.get("maximum_mean_normalized_score") == 0.98,
        "causal-transport source gate changed",
    )
    _require(
        observed == CANONICAL_METHOD_CONFIG_SHA256,
        "causal-transport method differs from the canonical source lock",
    )
    return {
        "passed": True,
        "protocol_id": METHOD_PROTOCOL_ID,
        "config_sha256": observed,
        "candidate_count": len(candidates),
    }


def load_causal_transport_method(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "causal-transport method must be an object")
    validate_causal_transport_method(payload)
    return payload


__all__ = [
    "CANONICAL_METHOD_CONFIG_SHA256",
    "METHOD_PROTOCOL_ID",
    "causal_transport_candidates",
    "causal_transport_method_sha256",
    "load_causal_transport_method",
    "validate_causal_transport_method",
]
