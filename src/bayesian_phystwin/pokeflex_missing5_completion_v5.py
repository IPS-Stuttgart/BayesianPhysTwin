"""Pretarget V5 scale amendment for PokeFlex's five unavailable official takes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .pokeflex_action_robust_all18 import SOURCE_FIELD
from .pokeflex_action_robust_official18_v4 import (
    ALL18_MULTIPLIERS,
    OFFICIAL18_MISSING_PUBLIC_TAKE_IDS,
    SELECTED_ARM,
    validate_official18_v4_protocol,
)
from .pokeflex_action_robust_official18_v4 import (
    EXPECTED_PROTOCOL_SHA256 as PARENT_PROTOCOL_SHA256,
)
from .pokeflex_missing5_scale import (
    BASE_EFFECTIVE_SCALE,
    GLOBAL_MULTIPLIER,
    result_sha256,
    validate_source_protocol,
)

PROTOCOL_KIND = "PokeFlexMissingFiveScaleCompletionV5Protocol"
PROTOCOL_ID = "pokeflex-missing5-scale-completion-v5"
PARENT_PROTOCOL_FILE_SHA256 = (
    "2e51c38305a345200485a49be1c82d5816d3d1c4c5867026115fbecd27d9d141"
)
SOURCE_PROTOCOL_SHA256 = (
    "83737068dca8621e331bcd30c76bc2852509872e59d034d984dc931d7bf5e27a"
)
SOURCE_PROTOCOL_FILE_SHA256 = (
    "0671df8beaaa4e560a264599ab5edbedd2e66ad2a7e1f9181f1b71fdea5fc70a"
)
SOURCE_RESULT_SHA256 = (
    "49658508e9531abd43d966c0eeb56f4deec43db3234e0ea530f756955b6deee7"
)
SOURCE_RESULT_FILE_SHA256 = (
    "2f666ce4060a488f036745ff9471acd39a79e2c2a7e0799c7c03e65075e75bf1"
)
TARGET_MULTIPLIERS = {
    "3dPrintedCylinder_T7": 2.0,
    "3dPrintedHeart_T14": 1.5,
    "3dPrintedPizza_T13": 1.0,
    "Pillow_T8": 1.0,
    "Sponge_T10": 1.0,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: str | Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical V5 completion-protocol digest."""

    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _source_result_multipliers(
    source_result: Mapping[str, Any],
    source_protocol: Mapping[str, Any],
) -> dict[str, float]:
    validate_source_protocol(source_protocol)
    _require(
        source_protocol.get("protocol_sha256") == SOURCE_PROTOCOL_SHA256,
        "registered source protocol changed",
    )
    _require(source_result.get("schema_version") == 1, "source result schema changed")
    _require(
        source_result.get("result_sha256") == result_sha256(source_result),
        "source result checksum changed",
    )
    _require(
        source_result.get("result_sha256") == SOURCE_RESULT_SHA256,
        "registered source result changed",
    )
    _require(
        source_result.get("protocol_sha256") == SOURCE_PROTOCOL_SHA256,
        "source result protocol changed",
    )
    _require(
        source_result.get("official_target_outcomes_used") is False,
        "source result used an official target outcome",
    )
    _require(
        source_result.get("held_v8_accessed") is False,
        "source result accessed held-v8",
    )
    gate = source_result.get("source_gate")
    _require(isinstance(gate, Mapping), "source result gate is missing")
    _require(gate.get("passed") is True, "source result gate did not pass")
    _require(int(gate.get("complete_take_count", -1)) == 30, "source result is incomplete")
    _require(
        int(gate.get("source_action_regression_count", -1)) == 0,
        "source result has a deployed action regression",
    )
    _require(
        int(gate.get("deployed_loo_held_action_regression_count", -1)) == 0,
        "source result has a deployed LOO regression",
    )
    rows = source_result.get("objects")
    _require(isinstance(rows, Mapping), "source object rows are missing")
    expected_objects = {take.rpartition("_T")[0] for take in TARGET_MULTIPLIERS}
    _require(set(rows) == expected_objects, "source object inventory changed")
    multipliers = {name: float(row["multiplier"]) for name, row in rows.items()}
    expected = {
        take.rpartition("_T")[0]: multiplier
        for take, multiplier in TARGET_MULTIPLIERS.items()
    }
    _require(multipliers == expected, "source-selected multiplier map changed")
    return multipliers


def build_completion_protocol(
    parent_protocol: Mapping[str, Any],
    source_protocol: Mapping[str, Any],
    source_result: Mapping[str, Any],
    *,
    locked_at_utc: str,
    parent_protocol_file_sha256: str,
    source_protocol_file_sha256: str,
    source_result_file_sha256: str,
) -> dict[str, Any]:
    """Build the immutable pretarget V5 scale amendment."""

    validate_official18_v4_protocol(parent_protocol)
    _require(
        parent_protocol_file_sha256 == PARENT_PROTOCOL_FILE_SHA256,
        "parent protocol bytes changed",
    )
    _require(
        source_protocol_file_sha256 == SOURCE_PROTOCOL_FILE_SHA256,
        "source protocol bytes changed",
    )
    _require(
        source_result_file_sha256 == SOURCE_RESULT_FILE_SHA256,
        "source result bytes changed",
    )
    object_multipliers = _source_result_multipliers(source_result, source_protocol)
    targets = tuple(OFFICIAL18_MISSING_PUBLIC_TAKE_IDS)
    _require(set(targets) == set(TARGET_MULTIPLIERS), "target inventory changed")
    effective_scales = {
        take: BASE_EFFECTIVE_SCALE * TARGET_MULTIPLIERS[take] for take in targets
    }
    v4_scales = {
        take: BASE_EFFECTIVE_SCALE * ALL18_MULTIPLIERS[take.rpartition("_T")[0]]
        for take in targets
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PROTOCOL_KIND,
        "protocol_id": PROTOCOL_ID,
        "locked_at_utc": locked_at_utc,
        "status": "pretarget V5 amendment; author archives remain unavailable",
        "claim_boundary": (
            "The source gate selects scales before any of the five official target "
            "archives are available. This amendment is not target evidence and does "
            "not alter the immutable V4 result or protocol."
        ),
        "parent_v4": {
            "path": "configs/sota/pokeflex_action_robust_official18_v4.json",
            "protocol_sha256": PARENT_PROTOCOL_SHA256,
            "protocol_file_sha256": parent_protocol_file_sha256,
            "unchanged": True,
        },
        "source_calibration": {
            "protocol_path": "configs/sota/pokeflex_missing5_scale_source_v5.json",
            "protocol_sha256": SOURCE_PROTOCOL_SHA256,
            "protocol_file_sha256": source_protocol_file_sha256,
            "result_path": (
                "results/sota/pokeflex_missing5_scale_source_v5/source_result.json"
            ),
            "result_sha256": SOURCE_RESULT_SHA256,
            "result_file_sha256": source_result_file_sha256,
            "source_outcomes_previously_opened": True,
            "official_target_outcomes_used": False,
            "object_multipliers": dict(sorted(object_multipliers.items())),
        },
        "target_cohort": {
            "take_ids": list(targets),
            "take_count": len(targets),
            "replacement_allowed": False,
            "author_archives_available_at_lock": False,
            "target_outcomes_opened_at_lock": False,
        },
        "method": {
            "selected_arm": SELECTED_ARM,
            "field": SOURCE_FIELD,
            "base_effective_scale": BASE_EFFECTIVE_SCALE,
            "global_multiplier": GLOBAL_MULTIPLIER,
            "target_multipliers": {
                take: TARGET_MULTIPLIERS[take] for take in targets
            },
            "target_effective_scales": effective_scales,
            "v4_target_effective_scales": v4_scales,
            "unsupported_frame_action": "byte-identical released checkpoint",
            "target_outcome_adaptation": "forbidden",
        },
        "custody": {
            "inherits_v4_author_source_manifest": True,
            "inherits_v4_all-five_prediction_barrier": True,
            "required_prospective_prediction_seal_count": len(targets),
            "target_mesh_access_before_barrier": "forbidden",
            "execution_implementation_lock_before_author_archive_access": True,
            "all_prediction_revisions_must_match": True,
        },
        "evaluation": {
            "primary_metric": "CD_UL1_mm",
            "surface_sample_count": 10_000,
            "surface_sample_seed": 20_260_720,
            "references": [
                "released checkpoint",
                "global effective scale 0.125",
                "frozen V4 object-specific scale",
                "published official-18 checkpoint value 6.498 mm",
            ],
            "bootstrap": {
                "unit": "physical object",
                "replicates": 20_000,
                "seed": 20_260_806,
                "upper_quantile": 0.975,
            },
        },
        "gates": {
            "prospective_v5_vs_v4": {
                "object_balanced_relative_improvement_above": 0.0,
                "minimum_per_object_relative_improvement": 0.0,
                "paired_bootstrap_upper_difference_mm_below": 0.0,
            },
            "official18": {
                "v5_below_v4": True,
                "v5_below_published_6_498_mm": True,
                "paired_bootstrap_upper_v5_minus_v4_mm_below": 0.0,
            },
        },
        "forbidden": [
            "opening a target mesh before all five V5 prediction seals pass",
            "changing any V5 multiplier from a target outcome",
            "substituting an alternate take for an unavailable official target",
            "weakening a V5-versus-V4 gate after author archive access",
            "altering or relabeling the frozen V4 protocol or result",
            "using frame f or later observations to predict frame f",
            "touching any held-v8 runtime, target, query, score, barrier, or outcome artifact",
        ],
        "held_v8_accessed": False,
    }
    payload["protocol_sha256"] = protocol_sha256(payload)
    validate_completion_protocol(payload, bind_registered_digest=False)
    return payload


def validate_completion_protocol(
    payload: Mapping[str, Any],
    *,
    bind_registered_digest: bool = True,
) -> dict[str, Any]:
    """Validate the immutable V5 target map and evidence boundary."""

    _require(payload.get("schema_version") == 1, "completion protocol schema changed")
    _require(payload.get("artifact_kind") == PROTOCOL_KIND, "completion kind changed")
    _require(payload.get("protocol_id") == PROTOCOL_ID, "completion id changed")
    observed = protocol_sha256(payload)
    _require(payload.get("protocol_sha256") == observed, "completion checksum changed")
    if bind_registered_digest:
        from .pokeflex_missing5_completion_v5_lock import EXPECTED_PROTOCOL_SHA256

        _require(observed == EXPECTED_PROTOCOL_SHA256, "registered completion changed")
    parent = payload.get("parent_v4")
    _require(isinstance(parent, Mapping), "parent V4 binding is missing")
    _require(parent.get("protocol_sha256") == PARENT_PROTOCOL_SHA256, "parent V4 changed")
    _require(
        parent.get("protocol_file_sha256") == PARENT_PROTOCOL_FILE_SHA256,
        "parent V4 bytes changed",
    )
    _require(parent.get("unchanged") is True, "parent V4 was not preserved")
    source = payload.get("source_calibration")
    _require(isinstance(source, Mapping), "source calibration binding is missing")
    _require(source.get("protocol_sha256") == SOURCE_PROTOCOL_SHA256, "source protocol changed")
    _require(
        source.get("protocol_file_sha256") == SOURCE_PROTOCOL_FILE_SHA256,
        "source protocol bytes changed",
    )
    _require(source.get("result_sha256") == SOURCE_RESULT_SHA256, "source result changed")
    _require(
        source.get("result_file_sha256") == SOURCE_RESULT_FILE_SHA256,
        "source result bytes changed",
    )
    _require(
        source.get("official_target_outcomes_used") is False,
        "source calibration used target outcomes",
    )
    target = payload.get("target_cohort")
    _require(isinstance(target, Mapping), "target cohort is missing")
    _require(
        tuple(target.get("take_ids", ())) == tuple(OFFICIAL18_MISSING_PUBLIC_TAKE_IDS),
        "target cohort changed",
    )
    _require(target.get("replacement_allowed") is False, "target replacement enabled")
    _require(
        target.get("author_archives_available_at_lock") is False,
        "author archives were available at lock",
    )
    _require(
        target.get("target_outcomes_opened_at_lock") is False,
        "target outcomes were opened at lock",
    )
    method = payload.get("method")
    _require(isinstance(method, Mapping), "method is missing")
    _require(method.get("selected_arm") == SELECTED_ARM, "selected arm changed")
    _require(method.get("field") == SOURCE_FIELD, "correction field changed")
    _require(
        dict(method.get("target_multipliers", {}))
        == {take: TARGET_MULTIPLIERS[take] for take in OFFICIAL18_MISSING_PUBLIC_TAKE_IDS},
        "target multiplier map changed",
    )
    for take, multiplier in TARGET_MULTIPLIERS.items():
        scale = float(method["target_effective_scales"][take])
        _require(
            math.isclose(scale, BASE_EFFECTIVE_SCALE * multiplier),
            "target effective scale changed",
        )
    _require(method.get("target_outcome_adaptation") == "forbidden", "adaptation enabled")
    _require(
        method.get("unsupported_frame_action") == "byte-identical released checkpoint",
        "unsupported-frame fallback changed",
    )
    custody = payload.get("custody")
    _require(isinstance(custody, Mapping), "custody is missing")
    _require(
        custody.get("execution_implementation_lock_before_author_archive_access")
        is True,
        "execution lock was disabled",
    )
    _require(
        int(custody.get("required_prospective_prediction_seal_count", -1)) == 5,
        "prediction barrier count changed",
    )
    _require(
        custody.get("target_mesh_access_before_barrier") == "forbidden",
        "target custody weakened",
    )
    gates = payload.get("gates")
    _require(isinstance(gates, Mapping), "target gates are missing")
    prospective = gates.get("prospective_v5_vs_v4")
    _require(isinstance(prospective, Mapping), "V5-versus-V4 gate is missing")
    _require(
        float(prospective.get("minimum_per_object_relative_improvement", -1.0)) == 0.0,
        "per-object target safety gate changed",
    )
    _require(
        float(prospective.get("object_balanced_relative_improvement_above", -1.0))
        == 0.0,
        "object-balanced target gate changed",
    )
    _require(
        float(prospective.get("paired_bootstrap_upper_difference_mm_below", 1.0))
        == 0.0,
        "prospective bootstrap gate changed",
    )
    official = gates.get("official18")
    _require(isinstance(official, Mapping), "official-18 gate is missing")
    _require(official.get("v5_below_v4") is True, "V5-versus-V4 gate changed")
    _require(
        official.get("v5_below_published_6_498_mm") is True,
        "published-reference gate changed",
    )
    _require(
        float(official.get("paired_bootstrap_upper_v5_minus_v4_mm_below", 1.0))
        == 0.0,
        "official-18 bootstrap gate changed",
    )
    _require(payload.get("held_v8_accessed") is False, "held-v8 boundary changed")
    return {
        "passed": True,
        "protocol_sha256": observed,
        "target_take_ids": tuple(OFFICIAL18_MISSING_PUBLIC_TAKE_IDS),
        "target_multipliers": {
            take: TARGET_MULTIPLIERS[take] for take in OFFICIAL18_MISSING_PUBLIC_TAKE_IDS
        },
    }


__all__ = [
    "PARENT_PROTOCOL_FILE_SHA256",
    "PROTOCOL_ID",
    "PROTOCOL_KIND",
    "SOURCE_PROTOCOL_FILE_SHA256",
    "SOURCE_PROTOCOL_SHA256",
    "SOURCE_RESULT_FILE_SHA256",
    "SOURCE_RESULT_SHA256",
    "TARGET_MULTIPLIERS",
    "build_completion_protocol",
    "file_sha256",
    "protocol_sha256",
    "validate_completion_protocol",
]
