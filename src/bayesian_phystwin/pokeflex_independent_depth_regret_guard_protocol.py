"""Validation for the prospective PokeFlex D405 regret-guard replication."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .pokeflex_independent_depth_regret_guard import FEATURE_NAMES
from .pokeflex_independent_depth_source_validation_protocol import (
    POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_SHA256,
)


POKEFLEX_REGRET_GUARD_PROSPECTIVE_PROTOCOL_ID = (
    "pokeflex-independent-depth-regret-guard-prospective-v1"
)
POKEFLEX_REGRET_GUARD_PROSPECTIVE_PROTOCOL_SHA256 = (
    "be2bbf6f2e1ac1ce0a536bd02a09633d5607677fe3f7ce8d51cfd8e7d533c447"
)
EXPECTED_PROSPECTIVE_TAKES = (
    "FoamDice_T7",
    "FoamDice_T8",
    "PlushOctopus_T7",
)
EXPECTED_SOURCE_RESULT_SHA256 = (
    "6065d1178796eba949b0411fbd57b53184e39d38f6559357c740d63b8b47398b"
)
EXPECTED_CANDIDATE_RUNNER_SHA256 = (
    "7927deb862dac8783b5415197ff65854ec3c0235a01db88689997c9b97f22e25"
)
EXPECTED_CODE_LOCK = {
    "regret_guard_module_sha256": (
        "485091f5b310eac330dc3f20fb0853ed831b06aafae01c0b2bb42eaf0fcef840"
    ),
    "source_evaluator_sha256": (
        "567241adc83681f0a3df7f046d1c30755ad67d1e1f6f921fae6ca6534d7bd11e"
    ),
    "prospective_runner_sha256": (
        "4942760e7ddca8a4bf1dff0efa543588fd50fd65289d1ba51f1eeddec1326b11"
    ),
    "prospective_evaluator_sha256": (
        "97e781c176cc428f4a043762688867e93cf5b12d34ecd2a48fa9c3a10d9336a1"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pokeflex_regret_guard_prospective_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical protocol checksum without its embedded digest."""

    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_pokeflex_regret_guard_prospective_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable three-take prospective development replication."""

    _require(payload.get("schema_version") == 1, "unsupported protocol schema")
    _require(
        payload.get("artifact_kind")
        == "PokeFlexIndependentDepthRegretGuardProspectiveProtocol",
        "unexpected prospective protocol kind",
    )
    _require(
        payload.get("protocol_id") == POKEFLEX_REGRET_GUARD_PROSPECTIVE_PROTOCOL_ID,
        "prospective protocol id changed",
    )
    observed = pokeflex_regret_guard_prospective_sha256(payload)
    _require(
        payload.get("protocol_sha256") == observed,
        "prospective protocol checksum mismatch",
    )
    if POKEFLEX_REGRET_GUARD_PROSPECTIVE_PROTOCOL_SHA256 != "TO_BE_FILLED":
        _require(
            observed == POKEFLEX_REGRET_GUARD_PROSPECTIVE_PROTOCOL_SHA256,
            "prospective protocol differs from canonical lock",
        )

    parent = payload.get("parent_method")
    _require(isinstance(parent, Mapping), "parent method is missing")
    _require(
        parent.get("protocol_sha256")
        == POKEFLEX_INDEPENDENT_DEPTH_SOURCE_VALIDATION_PROTOCOL_SHA256,
        "parent method protocol changed",
    )
    _require(
        parent.get("candidate_runner_sha256") == EXPECTED_CANDIDATE_RUNNER_SHA256,
        "candidate runner changed",
    )
    source = payload.get("source_evidence")
    _require(isinstance(source, Mapping), "source evidence is missing")
    _require(
        source.get("result_sha256") == EXPECTED_SOURCE_RESULT_SHA256,
        "source result changed",
    )
    _require(source.get("cross_object_gate_passed") is True, "source gate failed")

    cohort = payload.get("prospective_cohort")
    _require(isinstance(cohort, Mapping), "prospective cohort is missing")
    _require(
        tuple(cohort.get("take_ids", ())) == EXPECTED_PROSPECTIVE_TAKES,
        "prospective take inventory changed",
    )
    _require(cohort.get("replacement_allowed") is False, "replacement policy changed")
    _require(
        cohort.get("archive_members_or_outcomes_read_before_lock") is False,
        "prospective outcomes were opened before lock",
    )
    _require(
        cohort.get("calibration_objects_remain_sealed") is True
        and cohort.get("target_objects_remain_sealed") is True,
        "calibration or target objects were unsealed",
    )

    causal = payload.get("causal_input_contract")
    _require(isinstance(causal, Mapping), "causal input contract is missing")
    _require(
        causal.get("frame_f_kinect_or_realsense_allowed_before_prediction") is False,
        "future sensor access changed",
    )
    _require(
        causal.get("frame_f_mesh_allowed_before_scoring") is False,
        "future mesh access changed",
    )

    method = payload.get("method_lock")
    _require(isinstance(method, Mapping), "method lock is missing")
    _require(tuple(method.get("feature_names", ())) == FEATURE_NAMES, "features changed")
    expected_values = {
        "candidate_nominal_coverage": 0.9,
        "candidate_within_take_coverage": 0.8,
        "selector_nominal_coverage": 0.9,
        "selector_within_take_coverage": 0.8,
        "ridge_penalty": 10.0,
        "support_margin_std": 0.25,
        "minimum_improvement_mm": 0.0,
    }
    for name, expected in expected_values.items():
        _require(float(method.get(name, -1.0)) == expected, f"method changed: {name}")
    _require(method.get("no_refit_on_prospective_takes") is True, "refit policy changed")
    _require(
        method.get("exact_fallback")
        == "released Kinect checkpoint vertices byte-for-byte",
        "exact fallback changed",
    )

    evaluation = payload.get("evaluation")
    _require(isinstance(evaluation, Mapping), "evaluation gates are missing")
    _require(
        float(evaluation.get("minimum_object_balanced_relative_improvement", 0.0))
        >= 0.01,
        "improvement gate weakened",
    )
    _require(int(evaluation.get("minimum_object_wins", 0)) >= 2, "win gate weakened")
    _require(
        float(evaluation.get("maximum_object_regression", 1.0)) <= 0.0,
        "regression gate weakened",
    )
    _require(
        evaluation.get("all_required_for_replication_success") is True,
        "joint gate changed",
    )
    code_lock = payload.get("code_lock")
    _require(isinstance(code_lock, Mapping), "code lock is missing")
    _require(dict(code_lock) == EXPECTED_CODE_LOCK, "code lock changed")
    return {
        "passed": True,
        "protocol_sha256": observed,
        "take_ids": EXPECTED_PROSPECTIVE_TAKES,
    }


def load_pokeflex_regret_guard_prospective_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the canonical prospective protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = validate_pokeflex_regret_guard_prospective_protocol(payload)
    result["path"] = str(source)
    result["payload"] = payload
    return result
