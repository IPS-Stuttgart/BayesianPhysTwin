"""Validation for the frozen PokeFlex robot-data fusion source study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .pokeflex_registration_protocol import POKEFLEX_REGISTRATION_PROTOCOL_SHA256


POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_ID = "pokeflex-robot-fusion-source-v1"
POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_SHA256 = (
    "e1ca97dfab720f427561f40c1eac3958fc00edbd65319bb74a8c5302621900ce"
)
EXPECTED_DEVELOPMENT_OBJECTS = (
    "FoamDice",
    "MemoryFoam",
    "PlushOctopus",
    "3dPrintedHeart",
    "ToiletPaperRoll",
)
EXPECTED_ROBOT_CHECKPOINT_SHA256 = {
    "attention_model.pth": (
        "311dd1e9585cdb1d22048638f82b38131b7743e9f62ebccedd32b210cfb7d0eb"
    ),
    "decoder.pth": (
        "67832bc7ba9c49e89f5ac5fc6c77d80adbfa9d14cb84eba9a0fea1e7aa54c186"
    ),
    "force_encoder.pth": (
        "0550d1f46dfd599be386b76bd9ee4fa49f3df663606bf93f2db53c3400cfbee0"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pokeflex_robot_fusion_source_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical checksum without the embedded digest."""

    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_pokeflex_robot_fusion_source_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate source-only evidence and exact-fallback boundaries."""

    _require(payload.get("schema_version") == 1, "unsupported protocol schema")
    _require(
        payload.get("artifact_kind") == "PokeFlexRobotFusionSourceProtocol",
        "unexpected robot-fusion protocol kind",
    )
    _require(
        payload.get("protocol_id") == POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_ID,
        "robot-fusion protocol id changed",
    )
    observed = pokeflex_robot_fusion_source_sha256(payload)
    _require(
        payload.get("protocol_sha256") == observed,
        "robot-fusion protocol checksum mismatch",
    )
    if POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_SHA256 != "TO_BE_FILLED":
        _require(
            observed == POKEFLEX_ROBOT_FUSION_SOURCE_PROTOCOL_SHA256,
            "robot-fusion protocol differs from canonical lock",
        )

    parent = payload.get("parent_protocol")
    _require(isinstance(parent, Mapping), "parent protocol is missing")
    _require(
        parent.get("protocol_sha256") == POKEFLEX_REGISTRATION_PROTOCOL_SHA256,
        "parent protocol checksum changed",
    )
    upstream = payload.get("upstream")
    _require(isinstance(upstream, Mapping), "upstream provenance is missing")
    _require(
        upstream.get("code_commit")
        == "aaa8726072834a95bbe97e1a113588968c36e185",
        "upstream code commit changed",
    )
    checkpoint = upstream.get("released_robot_checkpoint")
    _require(isinstance(checkpoint, Mapping), "robot checkpoint provenance is missing")
    observed_checkpoint = {
        str(name): str(value.get("sha256", ""))
        for name, value in checkpoint.items()
        if isinstance(value, Mapping)
    }
    _require(
        observed_checkpoint == EXPECTED_ROBOT_CHECKPOINT_SHA256,
        "robot checkpoint hashes changed",
    )

    boundary = payload.get("evidence_boundary")
    _require(isinstance(boundary, Mapping), "evidence boundary is missing")
    _require(
        boundary.get("method_design_take") == "FoamDice_T1",
        "method-design take changed",
    )
    _require(
        tuple(boundary.get("development_objects", ()))
        == EXPECTED_DEVELOPMENT_OBJECTS,
        "development object boundary changed",
    )
    _require(
        boundary.get("source_takes") == ["T1", "T4", "T5", "T6"],
        "source-take inventory changed",
    )
    _require(
        boundary.get("target_objects_remain_sealed") is True,
        "target objects were unsealed",
    )
    _require(
        boundary.get("replacement_allowed") is False,
        "replacement policy changed",
    )

    causal = payload.get("causal_input_contract")
    _require(isinstance(causal, Mapping), "causal input contract is missing")
    _require(causal.get("history_frame_count") == 5, "history length changed")
    _require(
        causal.get("future_observation_used") is False,
        "future observation was enabled",
    )

    candidate = payload.get("candidate_lock")
    _require(isinstance(candidate, Mapping), "candidate lock is missing")
    _require(
        candidate.get("scales") == [0.0, 0.05, 0.1, 0.2],
        "candidate scale bank changed",
    )
    _require(
        "byte-for-byte" in str(candidate.get("exact_fallback", "")),
        "exact fallback changed",
    )
    _require(
        candidate.get("fixed_blend_is_not_final_method") is True,
        "fixed blend was promoted",
    )

    evaluation = payload.get("source_evaluation")
    _require(isinstance(evaluation, Mapping), "source evaluation is missing")
    _require(
        float(evaluation.get("minimum_object_balanced_relative_improvement", 0.0))
        >= 0.05,
        "transfer gate was weakened",
    )
    _require(
        int(evaluation.get("minimum_object_wins", 0)) >= 4,
        "object-win gate was weakened",
    )
    _require(
        float(evaluation.get("maximum_false_safe_rate", 1.0)) <= 0.1,
        "false-safe gate was weakened",
    )
    _require(
        evaluation.get("all_required_before_fresh_object_protocol") is True,
        "fresh-object gate changed",
    )
    return {
        "passed": True,
        "protocol_sha256": observed,
        "development_objects": EXPECTED_DEVELOPMENT_OBJECTS,
        "source_takes": ("T1", "T4", "T5", "T6"),
    }


def load_pokeflex_robot_fusion_source_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the canonical robot-fusion source protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = validate_pokeflex_robot_fusion_source_protocol(payload)
    result["path"] = str(source)
    result["payload"] = payload
    return result
