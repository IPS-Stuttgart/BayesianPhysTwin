"""Frozen protocol boundary for reusable Deform360 material association."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REUSABLE_ASSOCIATION_SCHEMA_VERSION = 1
REUSABLE_ASSOCIATION_PROTOCOL_ID = "deform360-reusable-material-association-v2"
CANONICAL_REUSABLE_ASSOCIATION_CONFIG_SHA256 = (
    "fa3c20b90d08c1fee11e7490c48c10d0a275d1850dd966084359eb93f1506118"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reusable_association_config_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_reusable_association_config(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == REUSABLE_ASSOCIATION_SCHEMA_VERSION,
        "unsupported reusable-association schema",
    )
    observed = reusable_association_config_sha256(payload)
    _require(
        payload.get("config_sha256") == observed,
        "reusable-association checksum mismatch",
    )
    _require(
        observed == CANONICAL_REUSABLE_ASSOCIATION_CONFIG_SHA256,
        "reusable-association protocol differs from the canonical lock",
    )
    config = payload.get("config")
    _require(isinstance(config, Mapping), "reusable-association config is missing")
    _require(
        config.get("protocol_id") == REUSABLE_ASSOCIATION_PROTOCOL_ID,
        "reusable-association protocol id changed",
    )
    candidates = config.get("mask_candidates", {})
    _require(
        candidates.get("simulator_residual_allowed") is False,
        "simulator residual leaked into association reliability",
    )
    _require(
        candidates.get("future_frame_allowed") is False,
        "future frame leaked into material association",
    )
    selection = config.get("joint_selection", {})
    _require(
        selection.get("top_appearance_set_retained_when_geometry_passes") is True,
        "geometry may override an already valid appearance identity",
    )
    gate = config.get("calibration_gate", {})
    _require(
        gate.get("future_prediction_metrics_allowed") is False,
        "calibration gate may read future outcomes",
    )
    _require(
        gate.get("conjunctive_all_episodes_required") is True,
        "calibration gate is not conjunctive",
    )
    boundary = config.get("information_boundary", {})
    _require(
        boundary.get("development_calibration_media_read") is False,
        "development read calibration media",
    )
    _require(
        boundary.get("development_target_media_read") is False,
        "development read target media",
    )
    _require(
        boundary.get("target_media_allowed_before_calibration_pass") is False,
        "target may open before calibration passes",
    )
    return {
        "passed": True,
        "protocol_id": REUSABLE_ASSOCIATION_PROTOCOL_ID,
        "config_sha256": observed,
    }


def load_reusable_association_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "reusable-association file must be an object")
    validate_reusable_association_config(payload)
    return payload


def reusable_association_source_evidence_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_reusable_association_source_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        payload.get("artifact_kind") == "Deform360ReusableAssociationSourceEvidence",
        "unexpected reusable-association evidence kind",
    )
    _require(
        payload.get("config_sha256") == CANONICAL_REUSABLE_ASSOCIATION_CONFIG_SHA256,
        "source evidence uses another reusable-association config",
    )
    _require(
        payload.get("result_sha256")
        == reusable_association_source_evidence_sha256(payload),
        "reusable-association source-evidence checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(
        boundary.get("calibration_media_read_for_v2_development") is False,
        "source evidence read calibration media",
    )
    _require(boundary.get("target_media_read") is False, "source evidence read target")
    conclusion = payload.get("conclusion", {})
    _require(
        conclusion.get("independent_calibration_passed") is False,
        "source evidence claims independent calibration",
    )
    _require(
        conclusion.get("state_of_the_art_claim") is False,
        "source evidence claims state of the art",
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "mask_case_count": len(payload.get("mask_cases", [])),
    }


def load_reusable_association_source_evidence(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "source-evidence file must be an object")
    validate_reusable_association_source_evidence(payload)
    return payload


__all__ = [
    "CANONICAL_REUSABLE_ASSOCIATION_CONFIG_SHA256",
    "REUSABLE_ASSOCIATION_PROTOCOL_ID",
    "load_reusable_association_config",
    "load_reusable_association_source_evidence",
    "reusable_association_config_sha256",
    "reusable_association_source_evidence_sha256",
    "validate_reusable_association_config",
    "validate_reusable_association_source_evidence",
]
