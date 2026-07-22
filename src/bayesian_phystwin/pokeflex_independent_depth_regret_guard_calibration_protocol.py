"""Validation for the independent-object PokeFlex regret-guard calibration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


POKEFLEX_REGRET_GUARD_CALIBRATION_PROTOCOL_ID = (
    "pokeflex-independent-depth-regret-guard-calibration-v1"
)
POKEFLEX_REGRET_GUARD_CALIBRATION_PROTOCOL_SHA256 = (
    "cef28df26d7710670aa3d73883462e896ec2d903c3be5faf00fee6a8d00a2644"
)
EXPECTED_CALIBRATION_TAKES = (
    "3dPrintedPyramid_T2",
    "Beanbag_T2",
    "FoamCylinder_T2",
    "PlushMoon_T2",
)
EXPECTED_PARENT_PROTOCOL_SHA256 = (
    "be2bbf6f2e1ac1ce0a536bd02a09633d5607677fe3f7ce8d51cfd8e7d533c447"
)
EXPECTED_PARENT_RESULT_SHA256 = (
    "0fc409a1bf85d4ef0e697bdb5604689094914ac57d32652660685007bfd67b98"
)
EXPECTED_SOURCE_RESULT_SHA256 = (
    "6065d1178796eba949b0411fbd57b53184e39d38f6559357c740d63b8b47398b"
)
EXPECTED_CANDIDATE_RUNNER_SHA256 = (
    "7927deb862dac8783b5415197ff65854ec3c0235a01db88689997c9b97f22e25"
)
EXPECTED_CODE_LOCK = {
    "calibration_runner_sha256": (
        "42d5a88e9ffa6d673898151e68b2174ac071d2fc974e5ca29bd375661944d270"
    ),
    "calibration_evaluator_sha256": (
        "a115dfabe6a716ed5bf9cfda97319d9bb22dbb8e88dc85ac9f78a4d7a742b4a6"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def pokeflex_regret_guard_calibration_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical protocol checksum without its embedded digest."""

    canonical = dict(payload)
    canonical.pop("protocol_sha256", None)
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_pokeflex_regret_guard_calibration_protocol(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the immutable four-object calibration evaluation."""

    _require(payload.get("schema_version") == 1, "unsupported protocol schema")
    _require(
        payload.get("artifact_kind")
        == "PokeFlexIndependentDepthRegretGuardCalibrationProtocol",
        "unexpected calibration protocol kind",
    )
    _require(
        payload.get("protocol_id") == POKEFLEX_REGRET_GUARD_CALIBRATION_PROTOCOL_ID,
        "calibration protocol id changed",
    )
    observed = pokeflex_regret_guard_calibration_sha256(payload)
    _require(payload.get("protocol_sha256") == observed, "protocol checksum mismatch")
    if POKEFLEX_REGRET_GUARD_CALIBRATION_PROTOCOL_SHA256 != "TO_BE_FILLED":
        _require(
            observed == POKEFLEX_REGRET_GUARD_CALIBRATION_PROTOCOL_SHA256,
            "calibration protocol differs from canonical lock",
        )
    parent = payload.get("parent_replication")
    _require(isinstance(parent, Mapping), "parent replication is missing")
    _require(
        parent.get("protocol_sha256") == EXPECTED_PARENT_PROTOCOL_SHA256,
        "parent protocol changed",
    )
    _require(
        parent.get("result_sha256") == EXPECTED_PARENT_RESULT_SHA256,
        "parent result changed",
    )
    _require(parent.get("gate_passed") is True, "parent replication failed")
    deployment = payload.get("frozen_deployment")
    _require(isinstance(deployment, Mapping), "frozen deployment is missing")
    _require(
        deployment.get("source_result_sha256") == EXPECTED_SOURCE_RESULT_SHA256,
        "source result changed",
    )
    _require(
        deployment.get("candidate_runner_sha256")
        == EXPECTED_CANDIDATE_RUNNER_SHA256,
        "candidate runner changed",
    )
    _require(
        deployment.get("no_refit_or_recalibration") is True,
        "calibration refit policy changed",
    )
    cohort = payload.get("calibration_cohort")
    _require(isinstance(cohort, Mapping), "calibration cohort is missing")
    _require(
        tuple(cohort.get("take_ids", ())) == EXPECTED_CALIBRATION_TAKES,
        "calibration take inventory changed",
    )
    _require(cohort.get("replacement_allowed") is False, "replacement policy changed")
    _require(
        cohort.get("archive_members_or_outcomes_read_before_lock") is False,
        "calibration outcomes were opened before lock",
    )
    _require(
        cohort.get("target_objects_remain_sealed") is True,
        "target objects were unsealed",
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
    evaluation = payload.get("evaluation")
    _require(isinstance(evaluation, Mapping), "evaluation gates are missing")
    _require(
        float(evaluation.get("minimum_object_balanced_relative_improvement", 0.0))
        >= 0.01,
        "improvement gate weakened",
    )
    _require(int(evaluation.get("minimum_object_wins", 0)) >= 3, "win gate weakened")
    _require(
        float(evaluation.get("maximum_object_regression", 1.0)) <= 0.01,
        "regression gate weakened",
    )
    _require(
        float(evaluation.get("maximum_false_safe_rate", 1.0)) <= 0.10,
        "false-safe gate weakened",
    )
    _require(
        evaluation.get("all_required_before_target_protocol_drafting") is True,
        "joint gate changed",
    )
    code_lock = payload.get("code_lock")
    _require(isinstance(code_lock, Mapping), "code lock is missing")
    if "TO_BE_FILLED" not in EXPECTED_CODE_LOCK.values():
        _require(dict(code_lock) == EXPECTED_CODE_LOCK, "code lock changed")
    return {
        "passed": True,
        "protocol_sha256": observed,
        "take_ids": EXPECTED_CALIBRATION_TAKES,
    }


def load_pokeflex_regret_guard_calibration_protocol(
    path: str | Path,
) -> dict[str, Any]:
    """Load and validate the canonical calibration protocol."""

    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = validate_pokeflex_regret_guard_calibration_protocol(payload)
    result["path"] = str(source)
    result["payload"] = payload
    return result
