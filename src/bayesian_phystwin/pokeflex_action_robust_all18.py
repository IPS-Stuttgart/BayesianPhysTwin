"""Source-only extension of PokeFlex robust scales to all 18 objects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .pokeflex_action_robust_scale import (
    ACTION_ROBUST_SCALE_FILE_SHA256,
    ACTION_ROBUST_SCALE_SHA256,
    BASE_EFFECTIVE_SCALE,
    CANDIDATE_MULTIPLIERS,
    GLOBAL_MULTIPLIER,
    action_robust_control_summary,
    select_action_robust_multiplier,
    validate_action_robust_scale_calibration,
)

ALL18_SOURCE_PROTOCOL_KIND = "PokeFlexActionRobustAll18SourceProtocol"
ALL18_SOURCE_PROTOCOL_ID = "pokeflex-action-robust-all18-source-v4"
ALL18_CALIBRATION_KIND = "PokeFlexActionRobustScaleCalibrationExtension"
ALL18_CALIBRATION_ID = "pokeflex-action-robust-scale-all18-v4"
SOURCE_SMOKE_KIND = "PokeFlexCheckpointBayesianRegistrationDevelopmentSmoke"
SOURCE_FIELD = "action_local_state_relative_0.4"
SELECTION_SALT = "pokeflex-action-robust-all18-source-v4"
NEW_OBJECTS = (
    "3dPrintedBunny",
    "3dPrintedHeart",
    "FoamDice",
    "MemoryFoam",
    "PlushOctopus",
    "ToiletPaperRoll",
)
EXPECTED_ALL18_OBJECTS = (
    "3dPrintedBunny",
    "3dPrintedCylinder",
    "3dPrintedHeart",
    "3dPrintedPizza",
    "3dPrintedPyramid",
    "Beanbag",
    "FoamCylinder",
    "FoamDice",
    "FoamHalfSphere",
    "MemoryFoam",
    "Pillow",
    "PlushDice",
    "PlushMoon",
    "PlushOctopus",
    "PlushTurtle",
    "PlushVolleyball",
    "Sponge",
    "ToiletPaperRoll",
)


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(payload: Mapping[str, Any], digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_name(take_id: str) -> str:
    object_name, separator, take_number = str(take_id).rpartition("_T")
    _require(bool(separator) and take_number.isdigit(), "invalid source take id")
    return object_name


def _selection_digest(take_id: str) -> str:
    return hashlib.sha256(
        SELECTION_SALT.encode("ascii") + b"\0" + take_id.encode("ascii")
    ).hexdigest()


def protocol_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of the source-only v4 protocol."""

    return _canonical_sha256(payload, "protocol_sha256")


def calibration_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of the all-18 scale calibration."""

    return _canonical_sha256(payload, "calibration_sha256")


def validate_all18_source_protocol(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate source selection and the no-target evidence boundary."""

    _require(payload.get("schema_version") == 1, "source protocol schema changed")
    _require(
        payload.get("artifact_kind") == ALL18_SOURCE_PROTOCOL_KIND,
        "source protocol kind changed",
    )
    _require(
        payload.get("protocol_id") == ALL18_SOURCE_PROTOCOL_ID,
        "source protocol id changed",
    )
    _require(
        payload.get("protocol_sha256") == protocol_sha256(payload),
        "source protocol checksum mismatch",
    )
    parent = payload.get("parent_calibration")
    _require(isinstance(parent, Mapping), "parent calibration binding is missing")
    assert isinstance(parent, Mapping)
    _require(
        parent.get("calibration_sha256") == ACTION_ROBUST_SCALE_SHA256,
        "parent calibration changed",
    )
    _require(
        parent.get("file_sha256") == ACTION_ROBUST_SCALE_FILE_SHA256,
        "parent calibration bytes changed",
    )
    method = payload.get("method")
    _require(isinstance(method, Mapping), "source method is missing")
    assert isinstance(method, Mapping)
    _require(method.get("field") == SOURCE_FIELD, "source field changed")
    _require(
        float(method.get("checkpoint_control_scale", np.nan)) == 0.0,
        "checkpoint control scale changed",
    )
    _require(
        float(method.get("base_effective_scale", -1.0)) == BASE_EFFECTIVE_SCALE,
        "base effective scale changed",
    )
    _require(
        tuple(float(value) for value in method.get("candidate_multipliers", ()))
        == CANDIDATE_MULTIPLIERS,
        "candidate multiplier bank changed",
    )
    expected_scales = tuple(
        BASE_EFFECTIVE_SCALE * value for value in CANDIDATE_MULTIPLIERS
    )
    _require(
        tuple(float(value) for value in method.get("effective_scales", ()))
        == expected_scales,
        "effective scale bank changed",
    )

    selection = payload.get("source_selection")
    _require(isinstance(selection, Mapping), "source selection is missing")
    assert isinstance(selection, Mapping)
    _require(selection.get("salt") == SELECTION_SALT, "selection salt changed")
    objects = selection.get("objects")
    _require(isinstance(objects, Mapping), "source object map is missing")
    assert isinstance(objects, Mapping)
    _require(
        tuple(sorted(objects)) == tuple(sorted(NEW_OBJECTS)), "source objects changed"
    )
    selected_take_ids: list[str] = []
    for object_name in NEW_OBJECTS:
        raw = objects[object_name]
        _require(isinstance(raw, Mapping), "source object row is invalid")
        assert isinstance(raw, Mapping)
        candidates = tuple(str(value) for value in raw.get("eligible_take_ids", ()))
        selected = tuple(str(value) for value in raw.get("selected_take_ids", ()))
        official_target = str(raw.get("official_target_take_id", ""))
        _require(len(candidates) >= 2, "source candidate inventory is too small")
        _require(len(candidates) == len(set(candidates)), "source candidates repeat")
        _require(
            all(_object_name(value) == object_name for value in candidates),
            "source candidates change object",
        )
        _require(
            official_target not in candidates, "official target entered source pool"
        )
        expected = tuple(sorted(candidates, key=_selection_digest)[:2])
        _require(selected == expected, "source pair is not the frozen salted selection")
        _require(official_target not in selected, "official target selected as source")
        selected_take_ids.extend(selected)
    _require(len(selected_take_ids) == 12, "source take count changed")
    _require(len(set(selected_take_ids)) == 12, "source takes repeat")

    boundary = payload.get("evidence_boundary")
    _require(isinstance(boundary, Mapping), "evidence boundary is missing")
    assert isinstance(boundary, Mapping)
    _require(
        boundary.get("official_target_outcomes_used_for_v4_selection") is False,
        "official target outcomes entered v4 selection",
    )
    _require(
        boundary.get("official18_evaluation_authorized") is False,
        "official18 evaluation was prematurely authorized",
    )
    _require(
        boundary.get("author_mapping_required_before_official18") is True,
        "author mapping requirement was removed",
    )
    return {
        "passed": True,
        "protocol_sha256": payload["protocol_sha256"],
        "selected_take_ids": tuple(selected_take_ids),
    }


def _scale_name(scale: float) -> str:
    return f"checkpoint_{SOURCE_FIELD}_residual_scale_{scale:g}"


def source_row_from_smoke(
    payload: Mapping[str, Any],
    *,
    expected_take_id: str,
    expected_protocol_sha256: str,
    artifact_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Extract the frozen multiplier bank from one causal source smoke."""

    _require(payload.get("schema_version") == 1, "source smoke schema changed")
    _require(payload.get("artifact_kind") == SOURCE_SMOKE_KIND, "source smoke changed")
    _require(
        payload.get("all18_source_protocol_sha256") == expected_protocol_sha256,
        "source smoke uses another v4 protocol",
    )
    take = payload.get("take")
    _require(isinstance(take, Mapping), "source take metadata is missing")
    assert isinstance(take, Mapping)
    _require(take.get("id") == expected_take_id, "source take identity changed")
    _require(
        payload.get("future_observation_used") is False, "source prediction leaked"
    )
    fields = tuple(str(value) for value in payload.get("correction_fields", ()))
    _require(SOURCE_FIELD in fields, "source field is missing")
    aggregates = payload.get("aggregates")
    _require(isinstance(aggregates, Mapping), "source aggregates are missing")
    assert isinstance(aggregates, Mapping)
    scores: dict[str, float] = {}
    for multiplier in CANDIDATE_MULTIPLIERS:
        scale = BASE_EFFECTIVE_SCALE * multiplier
        raw = aggregates.get(_scale_name(scale))
        _require(isinstance(raw, Mapping), "source scale aggregate is missing")
        assert isinstance(raw, Mapping)
        value = float(raw.get("mean_CD_UL1_mm", np.nan))
        _require(np.isfinite(value) and value > 0.0, "source scale score is invalid")
        scores[str(multiplier)] = value
    updates = payload.get("updates")
    _require(isinstance(updates, list), "source update records are missing")
    assert isinstance(updates, list)
    support = sum(
        bool(row.get("accepted")) and bool(row.get("action_supported"))
        for row in updates
        if isinstance(row, Mapping)
    )
    row: dict[str, Any] = {
        "take_id": expected_take_id,
        "supported_frame_count": int(support),
        "mean_CD_UL1_mm_by_multiplier": scores,
    }
    if artifact_file_sha256 is not None:
        _require(
            len(artifact_file_sha256) == 64,
            "source artifact file checksum is invalid",
        )
        row["source_artifact_file_sha256"] = artifact_file_sha256
    return row


def build_all18_calibration(
    parent_calibration: Mapping[str, Any],
    source_protocol: Mapping[str, Any],
    source_artifacts: Mapping[str, Mapping[str, Any]],
    *,
    source_artifact_file_sha256s: Mapping[str, str],
) -> dict[str, Any]:
    """Extend the frozen 12-object calibration with six source-only objects."""

    validate_action_robust_scale_calibration(parent_calibration)
    protocol_validation = validate_all18_source_protocol(source_protocol)
    expected = tuple(protocol_validation["selected_take_ids"])
    _require(
        set(source_artifacts) == set(expected), "source artifact inventory changed"
    )
    _require(
        set(source_artifact_file_sha256s) == set(expected),
        "source artifact checksum inventory changed",
    )
    for digest in source_artifact_file_sha256s.values():
        _require(
            len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
            "source artifact file checksum is invalid",
        )
    rows = {
        take_id: source_row_from_smoke(
            source_artifacts[take_id],
            expected_take_id=take_id,
            expected_protocol_sha256=str(source_protocol["protocol_sha256"]),
            artifact_file_sha256=source_artifact_file_sha256s[take_id],
        )
        for take_id in expected
    }
    selected_by_object = source_protocol["source_selection"]["objects"]
    new_objects = {
        object_name: select_action_robust_multiplier(
            [
                rows[take_id]
                for take_id in selected_by_object[object_name]["selected_take_ids"]
            ]
        )
        for object_name in NEW_OBJECTS
    }
    objects = deepcopy(dict(parent_calibration["objects"]))
    _require(not (set(objects) & set(new_objects)), "new objects overlap parent map")
    objects.update(new_objects)
    _require(
        tuple(sorted(objects)) == tuple(sorted(EXPECTED_ALL18_OBJECTS)),
        "all18 map incomplete",
    )

    gains = [
        float(gain)
        for row in new_objects.values()
        for gain in row["source_relative_improvements"]
    ]
    controls = action_robust_control_summary()
    source_gate = {
        "new_source_object_count": len(new_objects),
        "new_source_action_count": len(gains),
        "adjusted_new_object_count": sum(
            row["multiplier"] != GLOBAL_MULTIPLIER for row in new_objects.values()
        ),
        "source_action_regression_count": sum(gain < -1e-12 for gain in gains),
        "minimum_source_action_relative_improvement": float(min(gains)),
        "mean_source_action_relative_improvement": float(np.mean(gains)),
        "controls_passed": bool(controls["passed"]),
    }
    source_gate["passed"] = bool(
        source_gate["adjusted_new_object_count"] >= 3
        and source_gate["source_action_regression_count"] == 0
        and source_gate["mean_source_action_relative_improvement"] > 0.0
        and controls["passed"]
    )
    _require(source_gate["passed"], "all18 source extension gate failed")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ALL18_CALIBRATION_KIND,
        "calibration_id": ALL18_CALIBRATION_ID,
        "parent_calibration_sha256": ACTION_ROBUST_SCALE_SHA256,
        "parent_calibration_file_sha256": ACTION_ROBUST_SCALE_FILE_SHA256,
        "source_protocol_sha256": source_protocol["protocol_sha256"],
        "source_artifact_file_sha256s": dict(
            sorted(source_artifact_file_sha256s.items())
        ),
        "base_effective_scale": BASE_EFFECTIVE_SCALE,
        "global_multiplier": GLOBAL_MULTIPLIER,
        "candidate_multipliers": list(CANDIDATE_MULTIPLIERS),
        "objects": objects,
        "new_objects": new_objects,
        "source_gate": source_gate,
        "synthetic_controls": controls,
        "official_target_outcomes_used_for_v4_selection": False,
        "official18_evaluation_authorized": False,
        "claim_boundary": (
            "This source-only extension completes a scale map for all eighteen "
            "public object identities. It is not an official-split result, and "
            "the five unresolved official records still require author mapping."
        ),
    }
    payload["calibration_sha256"] = calibration_sha256(payload)
    return payload


def validate_all18_calibration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a generated all-18 source calibration."""

    _require(payload.get("schema_version") == 1, "calibration schema changed")
    _require(
        payload.get("artifact_kind") == ALL18_CALIBRATION_KIND,
        "calibration kind changed",
    )
    _require(
        payload.get("calibration_id") == ALL18_CALIBRATION_ID, "calibration id changed"
    )
    _require(
        payload.get("calibration_sha256") == calibration_sha256(payload),
        "calibration checksum mismatch",
    )
    _require(
        payload.get("parent_calibration_sha256") == ACTION_ROBUST_SCALE_SHA256,
        "parent calibration changed",
    )
    _require(
        payload.get("parent_calibration_file_sha256")
        == ACTION_ROBUST_SCALE_FILE_SHA256,
        "parent calibration bytes changed",
    )
    _require(
        payload.get("official_target_outcomes_used_for_v4_selection") is False,
        "target outcome entered v4 selection",
    )
    _require(
        payload.get("official18_evaluation_authorized") is False,
        "official18 evaluation was prematurely authorized",
    )
    source_hashes = payload.get("source_artifact_file_sha256s")
    _require(isinstance(source_hashes, Mapping), "source artifact hashes are missing")
    assert isinstance(source_hashes, Mapping)
    _require(len(source_hashes) == 12, "source artifact hash inventory changed")
    for take_id, digest in source_hashes.items():
        _require(
            isinstance(take_id, str)
            and isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest),
            "source artifact file checksum is invalid",
        )
    objects = payload.get("objects")
    _require(isinstance(objects, Mapping), "calibration object map is missing")
    assert isinstance(objects, Mapping)
    _require(
        tuple(sorted(objects)) == tuple(sorted(EXPECTED_ALL18_OBJECTS)),
        "all18 map changed",
    )
    new_objects = payload.get("new_objects")
    _require(isinstance(new_objects, Mapping), "new object map is missing")
    assert isinstance(new_objects, Mapping)
    _require(
        tuple(sorted(new_objects)) == tuple(sorted(NEW_OBJECTS)), "new objects changed"
    )
    expected_source_takes = {
        str(take_id)
        for row in new_objects.values()
        for take_id in row.get("source_take_ids", ())
    }
    _require(
        set(source_hashes) == expected_source_takes,
        "source artifact hash inventory changed",
    )
    gate = payload.get("source_gate")
    _require(isinstance(gate, Mapping), "source gate is missing")
    assert isinstance(gate, Mapping)
    _require(gate.get("passed") is True, "source gate failed")
    return {
        "passed": True,
        "calibration_sha256": payload["calibration_sha256"],
        "multipliers": {
            str(name): float(row["multiplier"]) for name, row in objects.items()
        },
    }


def load_all18_source_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_all18_source_protocol(payload)
    return payload


def load_source_artifacts(
    root: str | Path, take_ids: Sequence[str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load ``<take_id>.json`` source artifacts and bind their bytes."""

    source_root = Path(root)
    payloads = {}
    digests = {}
    for take_id in take_ids:
        path = source_root / f"{take_id}.json"
        _require(path.is_file(), f"missing source artifact: {take_id}")
        payloads[take_id] = json.loads(path.read_text(encoding="utf-8"))
        digests[take_id] = _file_sha256(path)
    return payloads, digests


__all__ = [
    "ALL18_CALIBRATION_ID",
    "ALL18_CALIBRATION_KIND",
    "ALL18_SOURCE_PROTOCOL_ID",
    "ALL18_SOURCE_PROTOCOL_KIND",
    "EXPECTED_ALL18_OBJECTS",
    "NEW_OBJECTS",
    "SELECTION_SALT",
    "SOURCE_FIELD",
    "build_all18_calibration",
    "calibration_sha256",
    "load_all18_source_protocol",
    "load_source_artifacts",
    "protocol_sha256",
    "source_row_from_smoke",
    "validate_all18_calibration",
    "validate_all18_source_protocol",
]
