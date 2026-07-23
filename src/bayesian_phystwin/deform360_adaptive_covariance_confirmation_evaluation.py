"""Target-opening-only evaluation for adaptive-covariance confirmation.

The evaluator has one deliberate capability boundary: the pinned production
prediction-barrier validator must successfully replay the complete H2-locked
34-case target-free cohort before the first invocation of ``target_loader``.
It then reopens every hash-bound target-free diagnostic and derives the route
evidence against which target-scoring results are checked.  This module never
searches for, selects, replaces, or silently drops a case.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bayesian_phystwin.deform360_adaptive_covariance_confirmation_lock import (
    PROTOCOL_ID,
    validate_confirmation_cohort_lock,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_external_runtime import (
    validate_confirmation_h2_loaded_runtime,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_scoring import (
    validate_confirmation_case_target_loader_attestation,
)
from bayesian_phystwin.deform360_adaptive_covariance_confirmation_seal import (
    CASE_DIAGNOSTIC_KIND,
    DIAGNOSTIC_FILENAME,
    TARGET_FREE_BOUNDARY,
    validate_confirmation_prediction_barrier,
)
from bayesian_phystwin.deform360_adaptive_covariance_rbf import (
    ADAPTIVE_COVARIANCE_PROTOCOL_ID,
)


SCHEMA_VERSION = 1
RESULT_ARTIFACT_KIND = "Deform360AdaptiveCovarianceConfirmationEvaluationResultV1"
DEVELOPMENT_RESULT_ARTIFACT_KIND = (
    "Deform360AdaptiveCovarianceConfirmationDevelopmentEvaluationV1"
)
BARRIER_ARTIFACT_KIND = "Deform360AdaptiveCovarianceConfirmationPredictionBarrierV1"
BARRIER_STATUS = "complete-target-free-cohort-prediction-barrier"
PRODUCTION_EVALUATION_MODE = "production_attested"
DEVELOPMENT_EVALUATION_MODE = "development_unattested"
EVALUATION_SOURCE_REPOSITORY_PATH = (
    "src/bayesian_phystwin/deform360_adaptive_covariance_confirmation_evaluation.py"
)
_EXACT_H2_RUNTIME_VALIDATOR = validate_confirmation_h2_loaded_runtime
_EXACT_SCORING_ATTESTATION_VALIDATOR = (
    validate_confirmation_case_target_loader_attestation
)
_EXACT_PREDICTION_BARRIER_VALIDATOR = validate_confirmation_prediction_barrier

EXPECTED_OBJECT_COUNT = 17
EXPECTED_CASE_COUNT = 34
EXPECTED_EPISODES_PER_OBJECT = 2
EXPECTED_UPDATE_FRAMES = (19, 38, 57)
EXPECTED_UPDATE_COUNT = EXPECTED_CASE_COUNT * len(EXPECTED_UPDATE_FRAMES)

ARMS = ("adaptive", "fixed8", "fixed4")
METRICS = (
    "post_update_hidden_identity_rmse_m",
    "post_update_hidden_symmetric_chamfer_m",
)
ROUTES = ("4_view_rbf", "8_view_rbf", "physical_prior_fallback")

NONINFERIORITY_RATIO_MARGIN = 1.05
HARMFUL_OBJECT_RATIO = 1.10
SEVERE_CASE_RATIO = 1.25
MAXIMUM_HARMFUL_OBJECT_COUNT = 1
MAXIMUM_SEVERE_CASE_COUNT = 0
MAXIMUM_MEAN_CHARGED_CAMERAS = 6.4
MAXIMUM_PHYSICAL_FALLBACK_COUNT = 25
MAXIMUM_RETAINED_TECHNICAL_FAILURE_CASE_COUNT = 0
MINIMUM_JOINT_SIGN_SUCCESSES = 13

BOOTSTRAP_REPLICATE_COUNT = 200_000
BOOTSTRAP_UPPER_QUANTILE = 0.95
BOOTSTRAP_UPPER_INDEX = (
    math.ceil(BOOTSTRAP_UPPER_QUANTILE * BOOTSTRAP_REPLICATE_COUNT) - 1
)
BOOTSTRAP_DOMAIN = b"deform360-confirmation-object-bootstrap-index-v1"
BOOTSTRAP_SEED_LABEL = "paired-object-bootstrap-v1-B200000"

_FULL_SHA1 = re.compile(r"[0-9a-f]{40}")
_FULL_SHA256 = re.compile(r"[0-9a-f]{64}")

_BARRIER_INFORMATION_BOUNDARY = {
    "sealer_target_or_outcome_argument_accepted": False,
    "sealer_target_or_outcome_path_opened": False,
    "metric_or_score_computed": False,
    "prediction_content_only": True,
    "all_case_predictions_must_seal_before_barrier": True,
}

_BARRIER_KEYS = {
    "schema_version",
    "artifact_kind",
    "protocol_id",
    "status",
    "lock_binding",
    "exact_case_ids",
    "case_count",
    "ordered_case_seals",
    "information_boundary",
    "artifact_sha256",
}
_LOCK_BINDING_KEYS = {
    "file_sha256",
    "artifact_sha256",
    "implementation_commit_h1",
    "cohort_lock_commit_h2",
}
_BARRIER_CASE_KEYS = {
    "case_id",
    "manifest_file_sha256",
    "manifest_artifact_sha256",
    "prediction_archive_sha256",
    "diagnostic_file_sha256",
    "diagnostic_artifact_sha256",
}
_OUTCOME_KEYS = {
    "case_id",
    "diagnostic_file_sha256",
    "diagnostic_artifact_sha256",
    "target_file_sha256",
    "target_arrays_sha256",
    "frame_zero_scale_m",
    "metrics",
    "updates",
}
_UPDATE_KEYS = {
    "update_frame",
    "route",
    "attempted_camera_ids",
    "future_visual_update_applied",
    "rbf_state_updated",
    "fallback_reason",
}
_DIAGNOSTIC_KEYS = {
    "schema_version",
    "artifact_kind",
    "protocol_id",
    "case_identity",
    "nested_selected_cameras",
    "covariance_routing",
    "technical_disposition",
    "information_boundary",
    "artifact_sha256",
}
_EXPECTED_UPDATE_STOPS = (38, 57, 76)
_CENTER_COUNT = 16
_FALLBACK_CONTRACT = {
    "trajectory": "physical_prior",
    "rbf_state_update": False,
    "bit_exact": True,
}
_RETAINED_FAILURE_FALLBACK_CONTRACT = {
    "trajectory": "persistence",
    "rbf_state_update": False,
    "bit_exact": True,
}
_RETAINED_FAILURE_CODES = {
    "automatic_twin_backend_failure",
    "prediction_runtime_failure",
    "resource_exhaustion",
}


@dataclass(frozen=True)
class _CaseSpec:
    case_id: str
    stratum: str
    object_id: str
    episode_id: int


TargetLoader = Callable[[str, Path, Mapping[str, Any]], Mapping[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any], *, digest_key: str) -> str:
    canonical = dict(value)
    canonical.pop(digest_key, None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _is_full_lower_sha1(value: object) -> bool:
    return (
        isinstance(value, str)
        and _FULL_SHA1.fullmatch(value) is not None
        and value != "0" * 40
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _FULL_SHA256.fullmatch(value) is not None


def _finite_nonnegative(value: object, *, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"{label} is invalid")
    return result


def _finite_positive(value: object, *, label: str) -> float:
    result = _finite_nonnegative(value, label=label)
    _require(result > 0.0, f"{label} must be positive")
    return result


def _load_lock(
    path: str | Path,
    *,
    expected_h1: str,
) -> tuple[dict[str, Any], str]:
    source = Path(path)
    payload_bytes = source.read_bytes()
    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("confirmation cohort lock is invalid JSON") from error
    _require(isinstance(payload, dict), "confirmation cohort lock is not an object")
    validate_confirmation_cohort_lock(
        payload,
        expected_implementation_commit_h1=expected_h1,
    )
    return payload, hashlib.sha256(payload_bytes).hexdigest()


def _case_specs(lock: Mapping[str, Any]) -> tuple[_CaseSpec, ...]:
    cohort = lock.get("cohort")
    _require(isinstance(cohort, Mapping), "lock cohort is not an object")
    specifications: list[_CaseSpec] = []
    object_ids: list[str] = []
    for stratum in ("filament", "sheet", "volumetric"):
        records = cohort.get(stratum)
        _require(isinstance(records, list), f"lock {stratum} cohort is not a list")
        for record in records:
            _require(isinstance(record, Mapping), "lock object record is invalid")
            object_id = record.get("object_id")
            _require(
                isinstance(object_id, str) and bool(object_id),
                "lock object ID is invalid",
            )
            object_ids.append(object_id)
            episodes = record.get("episodes")
            _require(
                isinstance(episodes, list)
                and len(episodes) == EXPECTED_EPISODES_PER_OBJECT,
                f"{object_id} must have exactly two locked episodes",
            )
            for episode in episodes:
                _require(
                    isinstance(episode, Mapping),
                    f"{object_id} episode record is invalid",
                )
                episode_id = episode.get("episode_id")
                case_id = episode.get("case_id")
                _require(
                    type(episode_id) is int and 0 <= episode_id <= 9,
                    f"{object_id} episode ID is invalid",
                )
                _require(
                    case_id == f"{object_id}-ep{episode_id:04d}",
                    f"{object_id} case ID changed",
                )
                specifications.append(
                    _CaseSpec(
                        case_id=case_id,
                        stratum=stratum,
                        object_id=object_id,
                        episode_id=episode_id,
                    )
                )
    case_ids = tuple(specification.case_id for specification in specifications)
    _require(
        len(object_ids) == len(set(object_ids)) == EXPECTED_OBJECT_COUNT,
        "lock must contain exactly 17 distinct physical objects",
    )
    _require(
        len(case_ids) == len(set(case_ids)) == EXPECTED_CASE_COUNT,
        "lock must contain exactly 34 distinct cases",
    )
    _require(
        list(case_ids) == lock.get("selected_case_ids"),
        "lock selected case order differs from cohort order",
    )
    _require(lock.get("case_count") == EXPECTED_CASE_COUNT, "lock case count changed")
    return tuple(specifications)


def _validate_barrier(
    barrier: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    lock_file_sha256: str,
    expected_h1: str,
    h2_commit: str,
    case_specs: tuple[_CaseSpec, ...],
) -> tuple[dict[str, Any], ...]:
    _require(isinstance(barrier, Mapping), "barrier validator returned no object")
    _require(set(barrier) == _BARRIER_KEYS, "prediction barrier schema changed")
    _require(barrier["schema_version"] == 1, "prediction barrier schema changed")
    _require(
        barrier["artifact_kind"] == BARRIER_ARTIFACT_KIND,
        "prediction barrier artifact kind changed",
    )
    _require(barrier["protocol_id"] == PROTOCOL_ID, "barrier protocol changed")
    _require(barrier["status"] == BARRIER_STATUS, "prediction barrier is incomplete")
    _require(
        barrier["information_boundary"] == _BARRIER_INFORMATION_BOUNDARY,
        "prediction barrier target boundary changed",
    )
    declared_digest = barrier["artifact_sha256"]
    _require(_is_sha256(declared_digest), "prediction barrier digest is invalid")
    _require(
        _canonical_sha256(barrier, digest_key="artifact_sha256") == declared_digest,
        "prediction barrier checksum mismatch",
    )

    lock_binding = barrier["lock_binding"]
    _require(
        isinstance(lock_binding, Mapping) and set(lock_binding) == _LOCK_BINDING_KEYS,
        "prediction barrier lock binding changed",
    )
    _require(
        lock_binding["file_sha256"] == lock_file_sha256,
        "prediction barrier lock file changed",
    )
    _require(
        lock_binding["artifact_sha256"] == lock["artifact_sha256"],
        "prediction barrier lock artifact changed",
    )
    _require(
        lock_binding["implementation_commit_h1"] == expected_h1,
        "prediction barrier H1 changed",
    )
    _require(
        lock_binding["cohort_lock_commit_h2"] == h2_commit,
        "prediction barrier H2 changed",
    )

    expected_case_ids = [specification.case_id for specification in case_specs]
    _require(
        barrier["exact_case_ids"] == expected_case_ids,
        "prediction barrier case order or closure changed",
    )
    _require(
        barrier["case_count"] == EXPECTED_CASE_COUNT,
        "prediction barrier case count changed",
    )
    ordered_case_seals = barrier["ordered_case_seals"]
    _require(
        isinstance(ordered_case_seals, list)
        and len(ordered_case_seals) == EXPECTED_CASE_COUNT,
        "prediction barrier must contain all 34 case seals",
    )
    normalized: list[dict[str, Any]] = []
    for expected_case_id, record in zip(
        expected_case_ids,
        ordered_case_seals,
        strict=True,
    ):
        _require(
            isinstance(record, Mapping) and set(record) == _BARRIER_CASE_KEYS,
            f"prediction barrier case schema changed: {expected_case_id}",
        )
        _require(
            record["case_id"] == expected_case_id,
            "prediction barrier case order changed",
        )
        for key in _BARRIER_CASE_KEYS - {"case_id"}:
            _require(
                _is_sha256(record[key]),
                f"prediction barrier {expected_case_id} {key} is invalid",
            )
        normalized.append(dict(record))
    return tuple(normalized)


def _stable_regular_file_bytes(path: Path, *, label: str) -> bytes:
    source = path.absolute()
    before = os.lstat(source)
    _require(not stat.S_ISLNK(before.st_mode), f"{label} may not be a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    _require(
        source.resolve(strict=True) == source,
        f"{label} has a symlinked or noncanonical path",
    )
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{label} changed while opening",
        )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.fstat(descriptor)
        current = os.lstat(source)
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        _require(
            identity
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            == (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ),
            f"{label} changed while reading",
        )
        _require(len(payload) == opened.st_size, f"{label} read was incomplete")
        return payload
    finally:
        os.close(descriptor)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_sealed_diagnostic_evidence(
    case_dir: str | Path,
    *,
    specification: _CaseSpec,
    barrier_case: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen one sealed diagnostic and derive its target-free route evidence."""

    root = Path(case_dir).absolute()
    _require(
        root.resolve(strict=True) == root
        and root.is_dir()
        and not root.is_symlink()
        and root.name == specification.case_id,
        f"{specification.case_id} sealed case directory changed after barrier",
    )
    raw = _stable_regular_file_bytes(
        root / DIAGNOSTIC_FILENAME,
        label=f"{specification.case_id} target-free diagnostic",
    )
    file_sha256 = hashlib.sha256(raw).hexdigest()
    _require(
        file_sha256 == barrier_case["diagnostic_file_sha256"],
        f"{specification.case_id} diagnostic file hash changed after barrier",
    )
    try:
        diagnostic = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"{specification.case_id} target-free diagnostic is invalid JSON"
        ) from error
    _require(
        isinstance(diagnostic, Mapping) and set(diagnostic) == _DIAGNOSTIC_KEYS,
        f"{specification.case_id} target-free diagnostic schema changed",
    )
    artifact_digest = diagnostic["artifact_sha256"]
    _require(
        _is_sha256(artifact_digest)
        and artifact_digest == barrier_case["diagnostic_artifact_sha256"]
        and _canonical_sha256(diagnostic, digest_key="artifact_sha256")
        == artifact_digest,
        f"{specification.case_id} diagnostic artifact hash changed after barrier",
    )
    expected_identity = {
        "case_id": specification.case_id,
        "stratum": specification.stratum,
        "object_id": specification.object_id,
        "episode_id": specification.episode_id,
    }
    _require(
        diagnostic["schema_version"] == SCHEMA_VERSION
        and diagnostic["artifact_kind"] == CASE_DIAGNOSTIC_KIND
        and diagnostic["protocol_id"] == PROTOCOL_ID
        and diagnostic["case_identity"] == expected_identity
        and diagnostic["information_boundary"] == TARGET_FREE_BOUNDARY,
        f"{specification.case_id} target-free diagnostic identity changed",
    )

    selected = diagnostic["nested_selected_cameras"]
    _require(
        isinstance(selected, Mapping) and set(selected) == {"4", "8"},
        f"{specification.case_id} nested camera schema changed",
    )
    selected_four = selected["4"]
    selected_eight = selected["8"]
    _require(
        isinstance(selected_four, list)
        and isinstance(selected_eight, list)
        and len(selected_four) == len(set(selected_four)) == 4
        and len(selected_eight) == len(set(selected_eight)) == 8
        and all(
            isinstance(camera, str) and bool(camera)
            for camera in selected_four + selected_eight
        )
        and selected_eight[:4] == selected_four,
        f"{specification.case_id} nested selected cameras changed",
    )

    disposition = diagnostic["technical_disposition"]
    _require(
        isinstance(disposition, Mapping)
        and disposition.get("status")
        in {"prediction_complete", "retained_technical_failure"}
        and disposition.get("case_retained") is True
        and disposition.get("disposition_based_on_target_or_outcome") is False,
        f"{specification.case_id} technical disposition changed",
    )
    center_ids = disposition.get("center_ids")
    _require(
        isinstance(center_ids, list)
        and len(center_ids) == len(set(center_ids)) == _CENTER_COUNT
        and all(type(center_id) is int and center_id >= 0 for center_id in center_ids),
        f"{specification.case_id} assimilation center IDs changed",
    )
    retained_failure = disposition["status"] == "retained_technical_failure"
    failure_code = disposition.get("failure_code")
    if retained_failure:
        _require(
            failure_code in _RETAINED_FAILURE_CODES,
            f"{specification.case_id} retained failure code is missing",
        )

    routing = diagnostic["covariance_routing"]
    expected_fallback_contract = (
        _RETAINED_FAILURE_FALLBACK_CONTRACT if retained_failure else _FALLBACK_CONTRACT
    )
    _require(
        isinstance(routing, Mapping)
        and routing.get("protocol_id") == ADAPTIVE_COVARIANCE_PROTOCOL_ID
        and routing.get("fallback") == expected_fallback_contract,
        f"{specification.case_id} covariance routing contract changed",
    )
    updates = routing.get("updates")
    _require(
        isinstance(updates, list) and len(updates) == len(EXPECTED_UPDATE_FRAMES),
        f"{specification.case_id} sealed routing update count changed",
    )
    derived_updates: list[dict[str, Any]] = []
    for expected_frame, expected_stop, update in zip(
        EXPECTED_UPDATE_FRAMES,
        _EXPECTED_UPDATE_STOPS,
        updates,
        strict=True,
    ):
        _require(
            isinstance(update, Mapping)
            and update.get("frame") == expected_frame
            and update.get("stop_frame_exclusive") == expected_stop,
            f"{specification.case_id} sealed routing frames changed",
        )
        route = update.get("route")
        _require(route in ROUTES, f"{specification.case_id} sealed route is invalid")
        if retained_failure:
            _require(
                route == "physical_prior_fallback",
                f"{specification.case_id} retained failure used a visual route",
            )
        expected_budget = {
            "4_view_rbf": 4,
            "8_view_rbf": 8,
            "physical_prior_fallback": None,
        }[route]
        _require(
            update.get("selected_camera_budget") == expected_budget,
            f"{specification.case_id} sealed route budget changed",
        )
        attempted_budgets = (4,) if route == "4_view_rbf" else (4, 8)
        budget_diagnostics = update.get("budget_diagnostics")
        _require(
            isinstance(budget_diagnostics, Mapping)
            and set(budget_diagnostics)
            == {str(budget) for budget in attempted_budgets},
            f"{specification.case_id} sealed attempted budgets changed",
        )
        expected_reliability = {
            "4_view_rbf": {"4": True},
            "8_view_rbf": {"4": False, "8": True},
            "physical_prior_fallback": {"4": False, "8": False},
        }[route]
        for budget in attempted_budgets:
            record = budget_diagnostics[str(budget)]
            valid_center_ids = (
                record.get("valid_covariance_center_ids")
                if isinstance(record, Mapping)
                else None
            )
            _require(
                isinstance(record, Mapping)
                and type(record.get("valid_covariance_center_count")) is int
                and 0 <= record["valid_covariance_center_count"] <= _CENTER_COUNT
                and isinstance(valid_center_ids, list)
                and len(valid_center_ids) == record["valid_covariance_center_count"]
                and len(valid_center_ids) == len(set(valid_center_ids))
                and all(
                    type(center_id) is int and center_id in center_ids
                    for center_id in valid_center_ids
                )
                and record.get("reliable") is expected_reliability[str(budget)],
                f"{specification.case_id} sealed covariance routing changed",
            )
        attempted = selected_four if route == "4_view_rbf" else selected_eight
        _require(
            update.get("tracked_cameras") == attempted
            and update.get("tracked_camera_count") == len(attempted),
            f"{specification.case_id} sealed attempted cameras changed",
        )
        applied = route != "physical_prior_fallback"
        _require(
            update.get("rbf_correction_applied") is applied
            and update.get("state_updated") is applied,
            f"{specification.case_id} sealed visual/state update flags changed",
        )
        if route == "physical_prior_fallback":
            _require(
                update.get("selected_backbone")
                == ("persistence" if retained_failure else "physical_prior"),
                f"{specification.case_id} sealed fallback backbone changed",
            )
            fallback_reason = (
                str(failure_code) if retained_failure else "covariance_abstention"
            )
        else:
            _require(
                update.get("selected_backbone") in {"physical_prior", "persistence"},
                f"{specification.case_id} sealed selected backbone changed",
            )
            fallback_reason = None
        derived_updates.append(
            {
                "update_frame": expected_frame,
                "route": route,
                "attempted_camera_ids": list(attempted),
                "future_visual_update_applied": applied,
                "rbf_state_updated": applied,
                "fallback_reason": fallback_reason,
            }
        )
    return {
        "diagnostic_file_sha256": file_sha256,
        "diagnostic_artifact_sha256": artifact_digest,
        "center_ids": list(center_ids),
        "updates": derived_updates,
    }


def _normalize_case_outcome(
    value: Mapping[str, Any],
    *,
    specification: _CaseSpec,
    barrier_case: Mapping[str, Any],
    sealed_diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "target loader returned no case object")
    _require(
        isinstance(sealed_diagnostic, Mapping)
        and set(sealed_diagnostic)
        == {
            "diagnostic_file_sha256",
            "diagnostic_artifact_sha256",
            "center_ids",
            "updates",
        }
        and sealed_diagnostic["diagnostic_file_sha256"]
        == barrier_case["diagnostic_file_sha256"]
        and sealed_diagnostic["diagnostic_artifact_sha256"]
        == barrier_case["diagnostic_artifact_sha256"],
        f"{specification.case_id} sealed diagnostic evidence is unbound",
    )
    _require(
        set(value) == _OUTCOME_KEYS, f"{specification.case_id} outcome schema changed"
    )
    _require(value["case_id"] == specification.case_id, "target loader changed case ID")
    for key in (
        "diagnostic_file_sha256",
        "diagnostic_artifact_sha256",
    ):
        _require(
            value[key] == barrier_case[key],
            f"{specification.case_id} target-free diagnostic changed after barrier",
        )
    for key in ("target_file_sha256", "target_arrays_sha256"):
        _require(
            _is_sha256(value[key]),
            f"{specification.case_id} {key} is invalid",
        )
    scale = _finite_positive(
        value["frame_zero_scale_m"],
        label=f"{specification.case_id} frame-zero scale",
    )

    metrics = value["metrics"]
    _require(
        isinstance(metrics, Mapping) and set(metrics) == set(ARMS),
        f"{specification.case_id} metric arms changed",
    )
    normalized_metrics: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_metrics = metrics[arm]
        _require(
            isinstance(arm_metrics, Mapping) and set(arm_metrics) == set(METRICS),
            f"{specification.case_id} {arm} metrics changed",
        )
        normalized_metrics[arm] = {
            metric: _finite_nonnegative(
                arm_metrics[metric],
                label=f"{specification.case_id} {arm} {metric}",
            )
            for metric in METRICS
        }
    for comparator in ("fixed8", "fixed4"):
        for metric in METRICS:
            _require(
                normalized_metrics[comparator][metric] > 0.0,
                f"{specification.case_id} {comparator} {metric} denominator is zero",
            )

    updates = value["updates"]
    _require(
        isinstance(updates, list) and len(updates) == len(EXPECTED_UPDATE_FRAMES),
        f"{specification.case_id} must have exactly three update routes",
    )
    normalized_updates: list[dict[str, Any]] = []
    for expected_frame, update in zip(
        EXPECTED_UPDATE_FRAMES,
        updates,
        strict=True,
    ):
        _require(
            isinstance(update, Mapping) and set(update) == _UPDATE_KEYS,
            f"{specification.case_id} update schema changed",
        )
        _require(
            update["update_frame"] == expected_frame,
            f"{specification.case_id} update frame order changed",
        )
        route = update["route"]
        _require(route in ROUTES, f"{specification.case_id} route is invalid")
        attempted = update["attempted_camera_ids"]
        _require(
            isinstance(attempted, list)
            and all(isinstance(camera, str) and camera for camera in attempted)
            and len(attempted) == len(set(attempted)),
            f"{specification.case_id} attempted camera IDs are invalid",
        )
        charged = 4 if route == "4_view_rbf" else 8
        _require(
            len(attempted) == charged,
            f"{specification.case_id} route camera charge changed",
        )
        future_applied = update["future_visual_update_applied"]
        state_updated = update["rbf_state_updated"]
        _require(
            type(future_applied) is bool and type(state_updated) is bool,
            f"{specification.case_id} update state flags are invalid",
        )
        fallback_reason = update["fallback_reason"]
        if route == "physical_prior_fallback":
            _require(
                future_applied is False and state_updated is False,
                f"{specification.case_id} physical fallback updated visual state",
            )
            _require(
                isinstance(fallback_reason, str) and bool(fallback_reason),
                f"{specification.case_id} fallback reason is missing",
            )
        else:
            _require(
                future_applied is True and state_updated is True,
                f"{specification.case_id} accepted route did not update visual state",
            )
            _require(
                fallback_reason is None,
                f"{specification.case_id} accepted route has a fallback reason",
            )
        normalized_updates.append(
            {
                "update_frame": expected_frame,
                "route": route,
                "attempted_camera_ids": list(attempted),
                "counterfactual_policy_charged_camera_streams": charged,
                "future_visual_update_applied": future_applied,
                "rbf_state_updated": state_updated,
                "fallback_reason": fallback_reason,
            }
        )
    reported_route_evidence = [
        {
            "update_frame": update["update_frame"],
            "route": update["route"],
            "attempted_camera_ids": update["attempted_camera_ids"],
            "future_visual_update_applied": update["future_visual_update_applied"],
            "rbf_state_updated": update["rbf_state_updated"],
            "fallback_reason": update["fallback_reason"],
        }
        for update in normalized_updates
    ]
    _require(
        reported_route_evidence == sealed_diagnostic["updates"],
        f"{specification.case_id} target loader route evidence differs from "
        "the sealed target-free diagnostic",
    )

    return {
        "case_id": specification.case_id,
        "stratum": specification.stratum,
        "object_id": specification.object_id,
        "episode_id": specification.episode_id,
        "prediction_barrier_case": dict(barrier_case),
        "target_evidence": {
            "target_file_sha256": value["target_file_sha256"],
            "target_arrays_sha256": value["target_arrays_sha256"],
        },
        "assimilation_center_ids": list(sealed_diagnostic["center_ids"]),
        "frame_zero_scale_m": scale,
        "metrics": normalized_metrics,
        "updates": normalized_updates,
    }


def _framed_sha256_bytes(*frames: bytes) -> bytes:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(len(frame).to_bytes(8, "big", signed=False))
        digest.update(frame)
    return digest.digest()


def bootstrap_seed_sha256(
    *,
    h1_commit: str,
    h2_commit: str,
    lock_artifact_sha256: str,
) -> str:
    """Return the outcome-independent preregistered object-bootstrap seed."""

    _require(_is_full_lower_sha1(h1_commit), "bootstrap H1 is invalid")
    _require(_is_full_lower_sha1(h2_commit), "bootstrap H2 is invalid")
    _require(_is_sha256(lock_artifact_sha256), "bootstrap lock digest is invalid")
    preimage = b"\0".join(
        (
            PROTOCOL_ID.encode("ascii"),
            h1_commit.encode("ascii"),
            h2_commit.encode("ascii"),
            lock_artifact_sha256.encode("ascii"),
            BOOTSTRAP_SEED_LABEL.encode("ascii"),
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def _bootstrap_index(
    seed_sha256: str,
    replicate_index: int,
    draw_index: int,
    *,
    object_count: int,
) -> int:
    _require(_is_sha256(seed_sha256), "bootstrap seed is invalid")
    _require(object_count >= 1, "bootstrap object count must be positive")
    _require(replicate_index >= 0 and draw_index >= 0, "bootstrap index is negative")
    modulus = 1 << 256
    rejection_limit = modulus - (modulus % object_count)
    counter = 0
    while True:
        digest = _framed_sha256_bytes(
            BOOTSTRAP_DOMAIN,
            bytes.fromhex(seed_sha256),
            replicate_index.to_bytes(8, "big", signed=False),
            draw_index.to_bytes(8, "big", signed=False),
            counter.to_bytes(8, "big", signed=False),
        )
        integer = int.from_bytes(digest, "big", signed=False)
        if integer < rejection_limit:
            return integer % object_count
        counter += 1


def _bootstrap_indices(
    seed_sha256: str,
    *,
    replicate_count: int,
    object_count: int,
) -> tuple[np.ndarray, str]:
    _require(replicate_count >= 1, "bootstrap replicate count must be positive")
    _require(1 <= object_count <= 65535, "bootstrap object count is invalid")
    encoded = bytearray(replicate_count * object_count * 2)
    indices = np.empty((replicate_count, object_count), dtype=np.uint16)
    cursor = 0
    for replicate in range(replicate_count):
        for draw in range(object_count):
            index = _bootstrap_index(
                seed_sha256,
                replicate,
                draw,
                object_count=object_count,
            )
            indices[replicate, draw] = index
            encoded[cursor : cursor + 2] = index.to_bytes(2, "big")
            cursor += 2
    return indices, hashlib.sha256(encoded).hexdigest()


def _empirical_upper_95(values: list[float]) -> float:
    _require(
        len(values) == BOOTSTRAP_REPLICATE_COUNT,
        "bootstrap distribution has the wrong replicate count",
    )
    _require(all(math.isfinite(value) for value in values), "bootstrap is non-finite")
    values.sort()
    return float(values[BOOTSTRAP_UPPER_INDEX])


def _bootstrap_ratio_upper(
    indices: np.ndarray,
    numerator: tuple[float, ...],
    denominator: tuple[float, ...],
) -> float:
    distribution: list[float] = []
    for row in indices:
        numerator_sum = math.fsum(numerator[int(index)] for index in row)
        denominator_sum = math.fsum(denominator[int(index)] for index in row)
        _require(denominator_sum > 0.0, "bootstrap ratio denominator is zero")
        distribution.append(numerator_sum / denominator_sum)
    return _empirical_upper_95(distribution)


def _bootstrap_mean_upper(
    indices: np.ndarray,
    values: tuple[float, ...],
) -> float:
    object_count = len(values)
    distribution = [
        math.fsum(values[int(index)] for index in row) / object_count for row in indices
    ]
    return _empirical_upper_95(distribution)


def _bootstrap_analysis(
    *,
    seed_sha256: str,
    object_values: Mapping[str, Any],
) -> dict[str, Any]:
    indices, matrix_sha256 = _bootstrap_indices(
        seed_sha256,
        replicate_count=BOOTSTRAP_REPLICATE_COUNT,
        object_count=EXPECTED_OBJECT_COUNT,
    )
    fixed8: dict[str, float] = {}
    fixed4: dict[str, float] = {}
    normalized: dict[str, float] = {}
    for metric in METRICS:
        adaptive = tuple(object_values["metrics"]["adaptive"][metric])
        fixed8_values = tuple(object_values["metrics"]["fixed8"][metric])
        fixed4_values = tuple(object_values["metrics"]["fixed4"][metric])
        fixed8[metric] = _bootstrap_ratio_upper(
            indices,
            adaptive,
            fixed8_values,
        )
        fixed4[metric] = _bootstrap_ratio_upper(
            indices,
            adaptive,
            fixed4_values,
        )
        normalized[metric] = _bootstrap_mean_upper(
            indices,
            tuple(object_values["scale_normalized_difference"][metric]),
        )
    camera_upper = _bootstrap_mean_upper(
        indices,
        tuple(object_values["mean_charged_cameras"]),
    )
    return {
        "algorithm_id": "sha256-rejection-paired-object-bootstrap-v1",
        "replicate_count": BOOTSTRAP_REPLICATE_COUNT,
        "object_count": EXPECTED_OBJECT_COUNT,
        "seed_sha256": seed_sha256,
        "index_framing": (
            "uint64-big-endian-length-framed domain, raw seed, uint64 replicate, "
            "uint64 draw, uint64 rejection counter; uniform 256-bit rejection "
            "before modulo 17"
        ),
        "resample_index_matrix_encoding": "row-major uint16 big-endian",
        "resample_index_matrix_sha256": matrix_sha256,
        "upper_quantile": BOOTSTRAP_UPPER_QUANTILE,
        "upper_order_statistic_zero_based_index": BOOTSTRAP_UPPER_INDEX,
        "summation": "Python math.fsum in lock object order",
        "one_sided_upper_95": {
            "adaptive_vs_fixed8_ratio": fixed8,
            "adaptive_vs_fixed4_ratio": fixed4,
            "scale_normalized_difference": normalized,
            "mean_counterfactual_policy_charged_camera_streams": camera_upper,
        },
    }


def _exact_sign_tail(success_count: int) -> dict[str, Any]:
    _require(
        type(success_count) is int and 0 <= success_count <= EXPECTED_OBJECT_COUNT,
        "sign-test success count is invalid",
    )
    numerator = sum(
        math.comb(EXPECTED_OBJECT_COUNT, index)
        for index in range(success_count, EXPECTED_OBJECT_COUNT + 1)
    )
    denominator = 2**EXPECTED_OBJECT_COUNT
    return {
        "success_count": success_count,
        "trial_count": EXPECTED_OBJECT_COUNT,
        "null_success_probability": 0.5,
        "tail_numerator": numerator,
        "tail_denominator": denominator,
        "one_sided_p_value": numerator / denominator,
        "critical_success_count": MINIMUM_JOINT_SIGN_SUCCESSES,
        "passed": success_count >= MINIMUM_JOINT_SIGN_SUCCESSES,
    }


def _tail_distribution(values: list[float]) -> dict[str, Any]:
    _require(bool(values), "tail distribution is empty")
    ordered = sorted(values)
    p90_index = math.ceil(0.90 * len(ordered)) - 1
    thresholds = (0.05, 0.10, 0.20, 0.25)
    return {
        "count": len(ordered),
        "maximum_relative_change": ordered[-1],
        "median_relative_change": float(statistics.median(ordered)),
        "p90_nearest_rank_relative_change": ordered[p90_index],
        "count_strictly_above": {
            format(threshold, ".2f"): sum(value > threshold for value in ordered)
            for threshold in thresholds
        },
    }


def _object_and_case_summaries(
    cases: tuple[dict[str, Any], ...],
    case_specs: tuple[_CaseSpec, ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    by_case = {case["case_id"]: case for case in cases}
    by_object: dict[str, list[_CaseSpec]] = {}
    for specification in case_specs:
        by_object.setdefault(specification.object_id, []).append(specification)
    _require(
        all(
            len(specifications) == EXPECTED_EPISODES_PER_OBJECT
            for specifications in by_object.values()
        ),
        "every object must retain exactly two episodes",
    )

    case_rows: list[dict[str, Any]] = []
    severe_case_ids: list[str] = []
    retained_technical_failure_case_ids: list[str] = []
    route_counts = {route: 0 for route in ROUTES}
    route_counts_by_update_frame = {
        str(frame): {route: 0 for route in ROUTES} for frame in EXPECTED_UPDATE_FRAMES
    }
    route_counts_by_stratum = {
        stratum: {route: 0 for route in ROUTES}
        for stratum in ("filament", "sheet", "volumetric")
    }
    fallback_reasons: dict[str, int] = {}
    case_relative_values = {metric: [] for metric in METRICS}
    for specification in case_specs:
        case = by_case[specification.case_id]
        relative: dict[str, dict[str, float]] = {"fixed8": {}, "fixed4": {}}
        for comparator in ("fixed8", "fixed4"):
            for metric in METRICS:
                relative[comparator][metric] = (
                    case["metrics"]["adaptive"][metric]
                    / case["metrics"][comparator][metric]
                    - 1.0
                )
        severe = any(
            1.0 + relative["fixed8"][metric] > SEVERE_CASE_RATIO for metric in METRICS
        )
        if severe:
            severe_case_ids.append(specification.case_id)
        for metric in METRICS:
            case_relative_values[metric].append(relative["fixed8"][metric])
        for update in case["updates"]:
            route = update["route"]
            route_counts[route] += 1
            route_counts_by_update_frame[str(update["update_frame"])][route] += 1
            route_counts_by_stratum[specification.stratum][route] += 1
            if route == "physical_prior_fallback":
                reason = update["fallback_reason"]
                fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
        if any(
            update["fallback_reason"] in _RETAINED_FAILURE_CODES
            for update in case["updates"]
        ):
            retained_technical_failure_case_ids.append(specification.case_id)
        case_rows.append(
            {
                **case,
                "relative_change": relative,
                "severe_vs_fixed8": severe,
            }
        )

    object_rows: list[dict[str, Any]] = []
    object_values: dict[str, Any] = {
        "metrics": {arm: {metric: [] for metric in METRICS} for arm in ARMS},
        "scale_normalized_difference": {metric: [] for metric in METRICS},
        "mean_paired_relative_change": {
            "fixed8": {metric: [] for metric in METRICS},
            "fixed4": {metric: [] for metric in METRICS},
        },
        "mean_charged_cameras": [],
    }
    harmful_object_ids: list[str] = []
    joint_fixed8_success_ids: list[str] = []
    joint_fixed4_strict_win_ids: list[str] = []
    object_relative_values = {metric: [] for metric in METRICS}
    for object_id, specifications in by_object.items():
        object_cases = [
            by_case[specification.case_id] for specification in specifications
        ]
        arm_metrics: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            arm_metrics[arm] = {}
            for metric in METRICS:
                value = (
                    math.fsum(case["metrics"][arm][metric] for case in object_cases)
                    / EXPECTED_EPISODES_PER_OBJECT
                )
                arm_metrics[arm][metric] = value
                object_values["metrics"][arm][metric].append(value)

        fixed8_ratios = {
            metric: arm_metrics["adaptive"][metric] / arm_metrics["fixed8"][metric]
            for metric in METRICS
        }
        fixed4_ratios = {
            metric: arm_metrics["adaptive"][metric] / arm_metrics["fixed4"][metric]
            for metric in METRICS
        }
        joint_fixed8_success = all(
            fixed8_ratios[metric] <= NONINFERIORITY_RATIO_MARGIN for metric in METRICS
        )
        joint_fixed4_strict_win = all(fixed4_ratios[metric] < 1.0 for metric in METRICS)
        harmful = any(
            fixed8_ratios[metric] > HARMFUL_OBJECT_RATIO for metric in METRICS
        )
        if joint_fixed8_success:
            joint_fixed8_success_ids.append(object_id)
        if joint_fixed4_strict_win:
            joint_fixed4_strict_win_ids.append(object_id)
        if harmful:
            harmful_object_ids.append(object_id)
        for metric in METRICS:
            object_relative_values[metric].append(fixed8_ratios[metric] - 1.0)

        normalized: dict[str, float] = {}
        paired_relative = {"fixed8": {}, "fixed4": {}}
        for metric in METRICS:
            normalized[metric] = (
                math.fsum(
                    (
                        case["metrics"]["adaptive"][metric]
                        - case["metrics"]["fixed8"][metric]
                    )
                    / case["frame_zero_scale_m"]
                    for case in object_cases
                )
                / EXPECTED_EPISODES_PER_OBJECT
            )
            object_values["scale_normalized_difference"][metric].append(
                normalized[metric]
            )
            for comparator in ("fixed8", "fixed4"):
                paired_relative[comparator][metric] = (
                    arm_metrics["adaptive"][metric] / arm_metrics[comparator][metric]
                    - 1.0
                )
                object_values["mean_paired_relative_change"][comparator][metric].append(
                    paired_relative[comparator][metric]
                )

        charged_values = [
            update["counterfactual_policy_charged_camera_streams"]
            for case in object_cases
            for update in case["updates"]
        ]
        mean_cameras = math.fsum(charged_values) / len(charged_values)
        object_values["mean_charged_cameras"].append(mean_cameras)
        object_route_counts = {
            route: sum(
                update["route"] == route
                for case in object_cases
                for update in case["updates"]
            )
            for route in ROUTES
        }
        object_rows.append(
            {
                "object_id": object_id,
                "stratum": specifications[0].stratum,
                "case_ids": [specification.case_id for specification in specifications],
                "metrics": arm_metrics,
                "ratios": {
                    "adaptive_vs_fixed8": fixed8_ratios,
                    "adaptive_vs_fixed4": fixed4_ratios,
                },
                "scale_normalized_difference_vs_fixed8": normalized,
                "paired_relative_change": paired_relative,
                "joint_fixed8_noninferiority_success": joint_fixed8_success,
                "joint_fixed4_strict_win": joint_fixed4_strict_win,
                "harmful_vs_fixed8": harmful,
                "mean_counterfactual_policy_charged_camera_streams": mean_cameras,
                "route_counts": object_route_counts,
            }
        )

    route_summary = {
        "update_count": EXPECTED_UPDATE_COUNT,
        "counts": route_counts,
        "counts_by_update_frame": route_counts_by_update_frame,
        "counts_by_stratum": route_counts_by_stratum,
        "fallback_route_count_including_retained_technical_failures": route_counts[
            "physical_prior_fallback"
        ],
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "minimum_four_view_count_implied_by_camera_gate": 41,
    }
    tail_summary = {
        "harmful_object_ids": harmful_object_ids,
        "harmful_object_count": len(harmful_object_ids),
        "harmful_object_ratio_threshold": HARMFUL_OBJECT_RATIO,
        "maximum_harmful_object_count": MAXIMUM_HARMFUL_OBJECT_COUNT,
        "severe_case_ids": severe_case_ids,
        "severe_case_count": len(severe_case_ids),
        "severe_case_ratio_threshold": SEVERE_CASE_RATIO,
        "maximum_severe_case_count": MAXIMUM_SEVERE_CASE_COUNT,
        "retained_technical_failure_case_ids": retained_technical_failure_case_ids,
        "retained_technical_failure_case_count": len(
            retained_technical_failure_case_ids
        ),
        "maximum_retained_technical_failure_case_count": (
            MAXIMUM_RETAINED_TECHNICAL_FAILURE_CASE_COUNT
        ),
        "joint_fixed8_noninferiority_success_ids": joint_fixed8_success_ids,
        "joint_fixed4_strict_win_ids": joint_fixed4_strict_win_ids,
        "object_relative_change_distributions": {
            metric: _tail_distribution(object_relative_values[metric])
            for metric in METRICS
        },
        "case_relative_change_distributions": {
            metric: _tail_distribution(case_relative_values[metric])
            for metric in METRICS
        },
    }
    return (
        case_rows,
        object_rows,
        object_values,
        {
            "routes": route_summary,
            "tails": tail_summary,
        },
    )


def _mean(values: list[float] | tuple[float, ...]) -> float:
    _require(bool(values), "cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _aggregate(
    *,
    object_values: Mapping[str, Any],
    auxiliary: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    arm_means: dict[str, dict[str, float]] = {
        arm: {
            metric: _mean(object_values["metrics"][arm][metric]) for metric in METRICS
        }
        for arm in ARMS
    }
    ratios = {
        "adaptive_vs_fixed8": {
            metric: arm_means["adaptive"][metric] / arm_means["fixed8"][metric]
            for metric in METRICS
        },
        "adaptive_vs_fixed4": {
            metric: arm_means["adaptive"][metric] / arm_means["fixed4"][metric]
            for metric in METRICS
        },
    }
    scale_normalized = {
        metric: _mean(object_values["scale_normalized_difference"][metric])
        for metric in METRICS
    }
    paired_relative = {
        comparator: {
            metric: _mean(
                object_values["mean_paired_relative_change"][comparator][metric]
            )
            for metric in METRICS
        }
        for comparator in ("fixed8", "fixed4")
    }
    mean_cameras = _mean(object_values["mean_charged_cameras"])

    upper = bootstrap["one_sided_upper_95"]
    fixed8_metric_gates = {
        metric: {
            "point_ratio": ratios["adaptive_vs_fixed8"][metric],
            "one_sided_upper_95_ratio": upper["adaptive_vs_fixed8_ratio"][metric],
            "noninferiority_margin": NONINFERIORITY_RATIO_MARGIN,
            "passed": (
                ratios["adaptive_vs_fixed8"][metric] < NONINFERIORITY_RATIO_MARGIN
                and upper["adaptive_vs_fixed8_ratio"][metric]
                < NONINFERIORITY_RATIO_MARGIN
            ),
        }
        for metric in METRICS
    }
    fixed8_ni_passed = all(gate["passed"] for gate in fixed8_metric_gates.values())
    sign_test = _exact_sign_tail(
        len(auxiliary["tails"]["joint_fixed8_noninferiority_success_ids"])
    )
    camera_gate = {
        "mean_counterfactual_policy_charged_camera_streams": mean_cameras,
        "maximum_mean": MAXIMUM_MEAN_CHARGED_CAMERAS,
        "one_sided_upper_95_diagnostic": upper[
            "mean_counterfactual_policy_charged_camera_streams"
        ],
        "passed": mean_cameras <= MAXIMUM_MEAN_CHARGED_CAMERAS,
        "bootstrap_bound_is_gating": False,
    }
    fallback_gate = {
        "fallback_route_count_including_retained_technical_failures": auxiliary[
            "routes"
        ]["fallback_route_count_including_retained_technical_failures"],
        "maximum_count": MAXIMUM_PHYSICAL_FALLBACK_COUNT,
        "passed": (
            auxiliary["routes"][
                "fallback_route_count_including_retained_technical_failures"
            ]
            <= MAXIMUM_PHYSICAL_FALLBACK_COUNT
        ),
    }
    harmful_gate = {
        "harmful_object_count": auxiliary["tails"]["harmful_object_count"],
        "maximum_count": MAXIMUM_HARMFUL_OBJECT_COUNT,
        "passed": (
            auxiliary["tails"]["harmful_object_count"] <= MAXIMUM_HARMFUL_OBJECT_COUNT
        ),
    }
    severe_gate = {
        "severe_case_count": auxiliary["tails"]["severe_case_count"],
        "maximum_count": MAXIMUM_SEVERE_CASE_COUNT,
        "passed": (
            auxiliary["tails"]["severe_case_count"] <= MAXIMUM_SEVERE_CASE_COUNT
        ),
    }
    technical_failure_gate = {
        "retained_technical_failure_case_count": auxiliary["tails"][
            "retained_technical_failure_case_count"
        ],
        "maximum_count": MAXIMUM_RETAINED_TECHNICAL_FAILURE_CASE_COUNT,
        "passed": (
            auxiliary["tails"]["retained_technical_failure_case_count"]
            <= MAXIMUM_RETAINED_TECHNICAL_FAILURE_CASE_COUNT
        ),
        "failure_event_evidence_is_operator_declared": True,
        "positive_confirmation_with_any_retained_technical_failure": False,
    }
    primary_passed = all(
        (
            fixed8_ni_passed,
            sign_test["passed"],
            camera_gate["passed"],
            fallback_gate["passed"],
            harmful_gate["passed"],
            severe_gate["passed"],
            technical_failure_gate["passed"],
        )
    )

    fixed4_metric_gates = {
        metric: {
            "point_ratio": ratios["adaptive_vs_fixed4"][metric],
            "one_sided_upper_95_ratio": upper["adaptive_vs_fixed4_ratio"][metric],
            "superiority_margin": 1.0,
            "passed": (
                ratios["adaptive_vs_fixed4"][metric] < 1.0
                and upper["adaptive_vs_fixed4_ratio"][metric] < 1.0
            ),
        }
        for metric in METRICS
    }
    fixed4_sign_test = _exact_sign_tail(
        len(auxiliary["tails"]["joint_fixed4_strict_win_ids"])
    )
    fixed4_supported = (
        primary_passed
        and all(gate["passed"] for gate in fixed4_metric_gates.values())
        and fixed4_sign_test["passed"]
    )
    return {
        "object_balanced_arm_means_m": arm_means,
        "ratios": ratios,
        "diagnostics": {
            "mean_scale_normalized_paired_difference_vs_fixed8": scale_normalized,
            "one_sided_upper_95_scale_normalized_paired_difference_vs_fixed8": (
                upper["scale_normalized_difference"]
            ),
            "mean_paired_relative_change": paired_relative,
        },
        "routes": auxiliary["routes"],
        "tails": auxiliary["tails"],
        "primary_confirmation": {
            "fixed8_noninferiority": {
                "intersection_union_no_multiplicity_adjustment": True,
                "metrics": fixed8_metric_gates,
                "passed": fixed8_ni_passed,
            },
            "joint_exact_sign_test": sign_test,
            "camera_budget": camera_gate,
            "fallback_route_including_retained_technical_failures": fallback_gate,
            "harmful_object_tail": harmful_gate,
            "severe_case_tail": severe_gate,
            "retained_technical_failure": technical_failure_gate,
            "passed": primary_passed,
            "decision": (
                "CONFIRMED_FIXED8_NI_ROUTED_CAMERA_BUDGET"
                if primary_passed
                else "NOT_CONFIRMED"
            ),
        },
        "fixed4_secondary": {
            "does_not_affect_primary_confirmation": True,
            "gatekept_after_primary_confirmation": True,
            "intersection_union_no_multiplicity_adjustment": True,
            "metrics": fixed4_metric_gates,
            "joint_exact_strict_win_sign_test": fixed4_sign_test,
            "supported": fixed4_supported,
            "decision": (
                "ADAPTIVE_SUPERIORITY_OVER_FIXED4_SUPPORTED"
                if fixed4_supported
                else "ADAPTIVE_SUPERIORITY_OVER_FIXED4_NOT_SUPPORTED"
            ),
        },
    }


def _withhold_unattested_confirmation_decisions(
    aggregate: dict[str, Any],
) -> None:
    """Keep development diagnostics while removing production claim fields."""

    primary = aggregate["primary_confirmation"]
    primary["development_statistical_gates_satisfied"] = primary.pop("passed")
    primary.pop("decision")
    primary["production_decision_authorized"] = False
    primary["production_decision_withheld_reason"] = (
        "unattested development target loader"
    )

    secondary = aggregate["fixed4_secondary"]
    secondary["development_statistical_gates_satisfied"] = secondary.pop("supported")
    secondary.pop("decision")
    secondary["production_decision_authorized"] = False
    secondary["production_decision_withheld_reason"] = (
        "unattested development target loader"
    )


def evaluate_adaptive_covariance_confirmation(
    lock_path: str | Path,
    barrier_path: str | Path,
    h2_commit: str,
    case_seal_dirs: Mapping[str, str | Path],
    *,
    expected_h1: str,
    target_loader: TargetLoader,
    evaluation_mode: str = DEVELOPMENT_EVALUATION_MODE,
    scoring_attestation: Mapping[str, Any] | None = None,
    adapter_repository: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the complete barrier, then open and score exactly 34 targets."""

    _require(_is_full_lower_sha1(expected_h1), "expected H1 is invalid")
    _require(_is_full_lower_sha1(h2_commit), "H2 commit is invalid")
    _require(h2_commit != expected_h1, "H2 must differ from implementation H1")
    _require(
        evaluation_mode in {DEVELOPMENT_EVALUATION_MODE, PRODUCTION_EVALUATION_MODE},
        "evaluation mode is invalid",
    )
    _require(
        evaluation_mode == PRODUCTION_EVALUATION_MODE or scoring_attestation is None,
        "development evaluation cannot accept a production scoring attestation",
    )
    _require(
        evaluation_mode == PRODUCTION_EVALUATION_MODE or adapter_repository is None,
        "development evaluation cannot accept a production adapter repository",
    )
    evaluator_repository_provenance: dict[str, Any] | None = None
    if evaluation_mode == PRODUCTION_EVALUATION_MODE:
        _require(
            validate_confirmation_h2_loaded_runtime is _EXACT_H2_RUNTIME_VALIDATOR,
            "production H2 runtime validator capability changed",
        )
        _require(
            validate_confirmation_case_target_loader_attestation
            is _EXACT_SCORING_ATTESTATION_VALIDATOR,
            "production scoring-attestation validator capability changed",
        )
        _require(
            validate_confirmation_prediction_barrier
            is _EXACT_PREDICTION_BARRIER_VALIDATOR,
            "production prediction-barrier validator capability changed",
        )
        _require(
            isinstance(adapter_repository, (str, os.PathLike)),
            "production evaluation requires the canonical adapter repository",
        )
        evaluator_repository_provenance = validate_confirmation_h2_loaded_runtime(
            adapter_repository,
            lock_path,
            h2_commit,
            expected_h1=expected_h1,
            source_file=__file__,
            source_repository_path=EVALUATION_SOURCE_REPOSITORY_PATH,
        )
    lock, lock_file_sha256 = _load_lock(lock_path, expected_h1=expected_h1)
    case_specs = _case_specs(lock)
    expected_case_ids = tuple(specification.case_id for specification in case_specs)
    _require(
        set(case_seal_dirs) == set(expected_case_ids)
        and len(case_seal_dirs) == EXPECTED_CASE_COUNT,
        "case seal directories do not have exact 34-case closure",
    )

    # This replay is the capability barrier.  No target-loader reference is
    # invoked, passed, or otherwise exposed before it returns successfully.
    barrier = validate_confirmation_prediction_barrier(
        barrier_path,
        lock_path,
        h2_commit,
        case_seal_dirs,
        expected_h1=expected_h1,
    )
    barrier_cases = _validate_barrier(
        barrier,
        lock=lock,
        lock_file_sha256=lock_file_sha256,
        expected_h1=expected_h1,
        h2_commit=h2_commit,
        case_specs=case_specs,
    )

    # Reopen every target-free diagnostic after the complete barrier replay and
    # before any target capability is invoked.  Target loaders may report route
    # evidence, but cannot author it.
    sealed_diagnostics = tuple(
        _load_sealed_diagnostic_evidence(
            case_seal_dirs[specification.case_id],
            specification=specification,
            barrier_case=barrier_case,
        )
        for specification, barrier_case in zip(
            case_specs,
            barrier_cases,
            strict=True,
        )
    )

    production_scoring_attestation: dict[str, Any] | None = None
    if evaluation_mode == PRODUCTION_EVALUATION_MODE:
        production_scoring_attestation = (
            validate_confirmation_case_target_loader_attestation(
                target_loader,
                lock_path,
                h2_commit,
                expected_h1=expected_h1,
                require_production=True,
            )
        )
        _require(
            isinstance(scoring_attestation, Mapping)
            and dict(scoring_attestation) == production_scoring_attestation,
            "production scoring attestation was not explicitly passed or changed",
        )

    outcomes: list[dict[str, Any]] = []
    for specification, barrier_case, sealed_diagnostic in zip(
        case_specs,
        barrier_cases,
        sealed_diagnostics,
        strict=True,
    ):
        loaded = target_loader(
            specification.case_id,
            Path(case_seal_dirs[specification.case_id]),
            barrier_case,
        )
        outcomes.append(
            _normalize_case_outcome(
                loaded,
                specification=specification,
                barrier_case=barrier_case,
                sealed_diagnostic=sealed_diagnostic,
            )
        )
    _require(
        len(outcomes) == EXPECTED_CASE_COUNT,
        "target loader did not return every locked case",
    )

    case_rows, object_rows, object_values, auxiliary = _object_and_case_summaries(
        tuple(outcomes),
        case_specs,
    )
    seed = bootstrap_seed_sha256(
        h1_commit=expected_h1,
        h2_commit=h2_commit,
        lock_artifact_sha256=lock["artifact_sha256"],
    )
    bootstrap = _bootstrap_analysis(
        seed_sha256=seed,
        object_values=object_values,
    )
    aggregate = _aggregate(
        object_values=object_values,
        auxiliary=auxiliary,
        bootstrap=bootstrap,
    )
    production_authorized = evaluation_mode == PRODUCTION_EVALUATION_MODE
    if not production_authorized:
        _withhold_unattested_confirmation_decisions(aggregate)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": (
            RESULT_ARTIFACT_KIND
            if production_authorized
            else DEVELOPMENT_RESULT_ARTIFACT_KIND
        ),
        "protocol_id": PROTOCOL_ID,
        "status": (
            "complete-target-opened-confirmation-evaluation"
            if production_authorized
            else "complete-target-opened-unattested-development-evaluation"
        ),
        "scoring_attestation": (
            {
                "status": "validated-production-attested",
                "production_confirmation_authorized": True,
                "attestation": production_scoring_attestation,
                "evaluator_repository_provenance": (evaluator_repository_provenance),
            }
            if production_authorized
            else {
                "status": "unattested-development-only",
                "production_confirmation_authorized": False,
                "attestation": None,
                "evaluator_repository_provenance": None,
            }
        ),
        "lock_binding": {
            "file_sha256": lock_file_sha256,
            "artifact_sha256": lock["artifact_sha256"],
            "implementation_commit_h1": expected_h1,
            "cohort_lock_commit_h2": h2_commit,
        },
        "prediction_barrier": {
            "artifact_sha256": barrier["artifact_sha256"],
            "status": barrier["status"],
            "validated_before_first_target_loader_invocation": True,
            "all_sealed_diagnostics_revalidated_before_first_target_loader_invocation": (
                True
            ),
            "case_count": barrier["case_count"],
        },
        "cohort": {
            "unit_of_replication": "physical object",
            "object_count": EXPECTED_OBJECT_COUNT,
            "episodes_per_object": EXPECTED_EPISODES_PER_OBJECT,
            "case_count": EXPECTED_CASE_COUNT,
            "update_count": EXPECTED_UPDATE_COUNT,
            "case_ids": list(expected_case_ids),
            "object_ids": [row["object_id"] for row in object_rows],
            "no_replacement_or_outcome_exclusion": True,
        },
        "metric_contract": {
            "metrics": list(METRICS),
            "case_metric_role": (
                (
                    "predeclared post-update hidden-identity metric returned by "
                    "the attested H1-bound native target scorer"
                )
                if production_authorized
                else (
                    "unattested development callback metric; not authorized for "
                    "the production confirmation decision"
                )
            ),
            "episode_aggregation": "arithmetic mean of exactly two episodes",
            "object_aggregation": "equal arithmetic weight for all 17 objects",
            "primary_estimand": "ratio of object-balanced arithmetic means",
            "noninferiority_margin": NONINFERIORITY_RATIO_MARGIN,
            "mean_paired_relative_change_is_diagnostic_only": True,
            "frame_zero_scale_normalized_difference_is_diagnostic_only": True,
        },
        "camera_budget_contract": {
            "four_view_rbf_charge": 4,
            "eight_view_rbf_charge": 8,
            "physical_or_technical_fallback_charge": 8,
            "all_camera_frame_zero_planning_excluded_and_disclosed": True,
            "offline_precomputation_does_not_establish_realized_compute_savings": True,
        },
        "bootstrap": bootstrap,
        "cases": case_rows,
        "objects": object_rows,
        "aggregate": aggregate,
        "claim_boundary": (
            (
                "A passing result supports prospective noninferiority to the "
                "frozen fixed-eight RBF comparator with a routed mean camera "
                "budget on this H2-locked causal-prefix cohort. It is not "
                "official Deform360 open-loop parity, a hardware-acquisition "
                "saving, or by itself a state-of-the-art claim."
            )
            if production_authorized
            else (
                "Development-only target-loader exercise. No production "
                "confirmation, noninferiority, superiority, camera-budget, "
                "parity, or state-of-the-art decision is authorized."
            )
        ),
    }
    result["result_sha256"] = _canonical_sha256(
        result,
        digest_key="result_sha256",
    )
    return result


__all__ = [
    "ARMS",
    "BARRIER_ARTIFACT_KIND",
    "BARRIER_STATUS",
    "BOOTSTRAP_REPLICATE_COUNT",
    "BOOTSTRAP_UPPER_INDEX",
    "DEVELOPMENT_EVALUATION_MODE",
    "DEVELOPMENT_RESULT_ARTIFACT_KIND",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_OBJECT_COUNT",
    "EXPECTED_UPDATE_COUNT",
    "METRICS",
    "MINIMUM_JOINT_SIGN_SUCCESSES",
    "NONINFERIORITY_RATIO_MARGIN",
    "PRODUCTION_EVALUATION_MODE",
    "RESULT_ARTIFACT_KIND",
    "bootstrap_seed_sha256",
    "evaluate_adaptive_covariance_confirmation",
]
