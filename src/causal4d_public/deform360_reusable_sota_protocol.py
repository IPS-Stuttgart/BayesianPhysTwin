"""Validation helpers for the prospective reusable-PhysTwin SOTA protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REUSABLE_SOTA_SCHEMA_VERSION = 1
REUSABLE_SOTA_PROTOCOL_ID = "deform360-reusable-sota-v1"
CANONICAL_REUSABLE_SOTA_CONFIG_SHA256 = (
    "64e41773d3e333987dc97a11c52e3474fd34fe65139e6c7d799b3e8f4db188cd"
)

EXPECTED_DEVELOPMENT_OBJECTS = {
    "1d": (
        "004-rubber-band",
        "067-paracord",
        "073-shoelace",
        "079-chain-metal",
    ),
    "2d": (
        "008-pink-cloth",
        "016-shirt-cloth",
        "021-bag-cloth",
        "040-paper-cloth",
    ),
    "3d": (
        "043-dog",
        "046-sponge",
        "052-rubber-duck",
        "096-octopus",
    ),
}
EXPECTED_CONFIRMATORY_OBJECTS = {
    "1d": ("068-nylon-rope", "074-string"),
    "2d": ("033-mask-cloth", "117-bubble-wrap-cloth"),
    "3d": ("090-sloth", "145-rubber-toy"),
}
EXPECTED_FIT_EPISODES = (1, 3, 4, 6, 7, 9)
EXPECTED_HELD_EPISODES = (0, 2, 5, 8)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reusable_sota_config_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the canonical payload while excluding its self-declared digest."""

    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_object_tuples(value: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {category: tuple(value.get(category, ())) for category in ("1d", "2d", "3d")}


def validate_reusable_sota_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any change to the prospective objects or information boundary."""

    _require(
        payload.get("schema_version") == REUSABLE_SOTA_SCHEMA_VERSION,
        "unsupported reusable-SOTA schema",
    )
    observed = reusable_sota_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "reusable-SOTA config checksum mismatch",
    )
    config = payload.get("config", {})
    _require(
        config.get("protocol_id") == REUSABLE_SOTA_PROTOCOL_ID,
        "reusable-SOTA protocol id changed",
    )
    dataset = config.get("dataset", {})
    _require(
        tuple(dataset.get("fit_episode_ids", ())) == EXPECTED_FIT_EPISODES
        and tuple(dataset.get("held_episode_ids", ())) == EXPECTED_HELD_EPISODES,
        "reusable-SOTA episode split changed",
    )
    development = _as_object_tuples(config.get("development_objects", {}))
    confirmatory = _as_object_tuples(config.get("confirmatory_objects", {}))
    _require(development == EXPECTED_DEVELOPMENT_OBJECTS, "development panel changed")
    _require(confirmatory == EXPECTED_CONFIRMATORY_OBJECTS, "confirmatory panel changed")
    all_objects = {
        object_id
        for panel in (development, confirmatory)
        for object_ids in panel.values()
        for object_id in object_ids
    }
    _require(len(all_objects) == 18, "reusable-SOTA object panel is not disjoint")

    claim = config.get("claim", {})
    _require(
        claim.get("primary_setting")
        == "multi-episode transfer to held actions on the same object"
        and claim.get("zero_shot_multi_object_claim") is False
        and claim.get("learned_temporal_residual_in_primary_method") is False
        and claim.get("causal4d_frozen_claim_may_change") is False,
        "reusable-SOTA claim boundary changed",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("confirmatory_held_input_frame_count") == 1
        and boundary.get("confirmatory_future_object_frames_allowed_before_prediction_seal")
        is False
        and boundary.get("confirmatory_future_particle_tracks_allowed_before_prediction_seal")
        is False
        and boundary.get("confirmatory_future_point_clouds_allowed_before_prediction_seal")
        is False
        and boundary.get("confirmatory_tactile_allowed_before_prediction_seal") is False
        and boundary.get("held_predictions_must_be_checksummed_before_outcome_reveal")
        is True
        and boundary.get("future_outcome_opened_early_invalidates_object_without_replacement")
        is True,
        "reusable-SOTA information boundary changed",
    )
    evaluation = config.get("evaluation", {})
    _require(
        evaluation.get("confirmatory_object_count") == 6
        and evaluation.get("confirmatory_held_episode_count") == 24
        and evaluation.get("unit_of_replication")
        == "object with episodes nested within object",
        "reusable-SOTA evaluation design changed",
    )
    reference = config.get("published_reference", {})
    _require(
        reference.get("direct_sota_claim_requires_identical_split_horizon_and_evaluator")
        is True
        and reference.get("pgrd_cross_benchmark_result_is_not_a_direct_comparator")
        is True,
        "reusable-SOTA comparison boundary changed",
    )
    _require(
        observed == CANONICAL_REUSABLE_SOTA_CONFIG_SHA256,
        "reusable-SOTA config differs from the canonical lock",
    )
    return {
        "passed": True,
        "protocol_id": REUSABLE_SOTA_PROTOCOL_ID,
        "config_sha256": observed,
        "development_object_count": 12,
        "confirmatory_object_count": 6,
        "confirmatory_held_episode_count": 24,
    }


def load_reusable_sota_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "reusable-SOTA config must be an object")
    validate_reusable_sota_config(payload)
    return payload


__all__ = [
    "CANONICAL_REUSABLE_SOTA_CONFIG_SHA256",
    "EXPECTED_CONFIRMATORY_OBJECTS",
    "EXPECTED_DEVELOPMENT_OBJECTS",
    "EXPECTED_FIT_EPISODES",
    "EXPECTED_HELD_EPISODES",
    "REUSABLE_SOTA_PROTOCOL_ID",
    "load_reusable_sota_config",
    "reusable_sota_config_sha256",
    "validate_reusable_sota_config",
]
