"""Shared contracts for Deform360 calibration execution records."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

DEFORM360_CALIBRATION_SOURCE_RUN_SCHEMA: Final = (
    "bayesian-phystwin.deform360-calibration-source-run"
)
DEFORM360_CALIBRATION_SOURCE_RUN_VERSION: Final = 1
DEFORM360_CALIBRATION_SOURCE_RUN_SEMANTICS: Final = (
    "non-sensitive-direct-calibration-source-completion-v1"
)
DEFORM360_CALIBRATION_SOURCE_PLAN_SCHEMA: Final = (
    "bayesian-phystwin/deform360-calibration-source-plan-v1"
)
DEFORM360_CALIBRATION_DOWNLOAD_SCHEMA: Final = (
    "bayesian-phystwin/deform360-calibration-download-v1"
)
DEFORM360_CALIBRATION_SOURCE_RESULT_SCHEMA: Final = (
    "bayesian-phystwin/deform360-calibration-source-result-v1"
)
DEFORM360_CALIBRATION_SOURCE_PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin/deform360-official-hub-calibration-source-v1"
)
DEFORM360_STAGE0_SELECTION_SCHEMA: Final = (
    "bayesian-phystwin/deform360-official-hub-selection-v1"
)
DEFORM360_VISUAL_PROVIDER_LOCK_SCHEMA: Final = (
    "bayesian-phystwin.deform360-visual-provider-lock"
)
DEFORM360_CALIBRATION_SOURCE_PROTOCOL_ID: Final = (
    "deform360-official-hub-calibration-source-v1"
)
DEFORM360_PARENT_PROTOCOL_ID: Final = "deform360-official-hub-visuotactile-v1"
DEFORM360_DATASET_REPOSITORY: Final = "brownu/deform360"
DEFORM360_DATASET_REVISION: Final = "f804696d7a133908c7497ffdab43819d879b5cbc"
DEFORM360_PROCESSING_REPOSITORY: Final = "lhy0807/deform360"
DEFORM360_VISUAL_PROVIDER_LOCK_SEMANTICS: Final = (
    "target-blind-prob4d-motioncrafter-producer-lock-v1"
)
DEFORM360_EXPECTED_TACTILE_BASELINE_POLICY: Final = {
    "policy_id": "nearest-filename-timestamp-v1",
    "maximum_absolute_distance_us": 600_000_000,
    "minimum_runner_up_margin_us": 60_000_000,
    "maximum_cross_sensor_span_us": 5_000_000,
    "single_candidate_is_accepted_without_timestamp": True,
}
DEFORM360_CALIBRATION_SOURCE_RUN_CLAIM_BOUNDARY: Final = (
    "Execution and information-boundary evidence only. This record does not "
    "establish observation-provider competence, physical-query benefit, "
    "calibration, independent-object transfer, deployment safety, or state of "
    "the art."
)
ARTIFACT_CONTRACT_EXIT_CODE: Final = 4
SUPPORT_GATE_EXIT_CODE: Final = 3
RECORD_WRITE_EXIT_CODE: Final = 70
EXPECTED_OBJECT_COUNT: Final = 10
EXPECTED_OBJECTS_PER_STRATUM: Final = 5
EXPECTED_CONFIRMATION_OBJECT_COUNT: Final = 12
EXPECTED_CONFIRMATION_OBJECTS_PER_STRATUM: Final = 6
MINIMUM_SUPPORTED_OBJECTS: Final = 8
MINIMUM_SUPPORTED_PER_STRATUM: Final = 4
MINIMUM_CAMERA_STREAMS: Final = 8
MINIMUM_ALIGNED_FRAMES: Final = 81
EXPECTED_STRATA: Final = ("sheet", "volumetric")
ALLOWED_RESULT_OBJECT_STATUSES: Final = frozenset(
    {
        "source_prepared",
        "technical_failure_without_replacement",
        "unsupported_without_replacement",
    }
)
EXPECTED_PLAN_INFORMATION_BOUNDARY: Final = {
    "repository_names_opened": True,
    "calibration_payloads_opened": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}
EXPECTED_DOWNLOAD_INFORMATION_BOUNDARY: Final = {
    "calibration_payloads_opened": True,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}
EXPECTED_RESULT_INFORMATION_BOUNDARY: Final = {
    "calibration_camera_payloads_opened": True,
    "calibration_tactile_payloads_opened": True,
    "calibration_robot_state_derived": True,
    "calibration_target_metrics_computed": False,
    "confirmation_payloads_opened": False,
    "target_outcomes_used": False,
    "replacement_allowed": False,
}
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ObjectIdentity = tuple[str, int, str]
ObjectIdentitySet = frozenset[ObjectIdentity]
ExpectedUnit = tuple[int, str, str, str]
ExpectedUnitMap = dict[str, ExpectedUnit]
ObjectSupportSummary = tuple[
    ObjectIdentitySet,
    frozenset[str],
    int,
    dict[str, int],
]


class InvalidJsonError(ValueError):
    """Strict JSON could not be parsed after its bytes were identified."""

    def __init__(self, file_sha256: str) -> None:
        super().__init__("JSON is not strict UTF-8 JSON")
        self.file_sha256 = file_sha256


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def load_json_object(path: Path) -> tuple[dict[str, Any], str]:
    """Load one strict JSON object and return its exact file digest."""

    encoded = path.read_bytes()
    file_sha256 = hashlib.sha256(encoded).hexdigest()
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except ValueError as error:
        raise InvalidJsonError(file_sha256) from error
    if not isinstance(value, dict):
        raise InvalidJsonError(file_sha256)
    return value, file_sha256


def content_sha256(value: Mapping[str, Any]) -> str:
    """Hash one complete canonical JSON object."""

    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_sha256(
    value: Mapping[str, Any],
    *,
    digest_key: str = "record_sha256",
) -> str:
    """Hash canonical JSON after excluding its self-digest field."""

    payload = dict(value)
    payload.pop(digest_key, None)
    return content_sha256(payload)


def exit_code(value: int, *, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 255:
        raise ValueError(f"{name} must be an integer in [0, 255]")
    return value


def positive_integer(value: int, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def revision(value: str, *, name: str) -> str:
    if type(value) is not str or _REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-character revision")
    return value


def sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def integer_field(
    value: object,
    *,
    name: str,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} exceeds {maximum}")
    return value


def string_sequence(
    value: object,
    *,
    name: str,
    minimum_length: int = 0,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise ValueError(f"{name} must be an array of strings")
    result = tuple(value)
    if len(result) < minimum_length or len(set(result)) != len(result):
        raise ValueError(f"{name} has invalid support or duplicates")
    return result


def raw_object_path(value: object, *, name: str) -> tuple[str, str]:
    """Validate a confined ``raw/<object>/...`` POSIX path."""

    if type(value) is not str or "\\" in value:
        raise ValueError(f"{name} must be a confined POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) < 3
        or path.parts[0] != "raw"
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{name} must remain below raw/<object>")
    return value, path.parts[1]


def object_support_counts(
    value: Mapping[str, Any],
    *,
    artifact: str,
    allowed_statuses: frozenset[str],
    supported_status: str,
) -> ObjectSupportSummary:
    """Validate the frozen ten-object cohort and count supported rows."""

    objects = value.get("objects")
    if not isinstance(objects, list) or len(objects) != EXPECTED_OBJECT_COUNT:
        raise ValueError(f"{artifact} must contain exactly ten object rows")
    identities: set[ObjectIdentity] = set()
    object_ids: set[str] = set()
    supported_ids: set[str] = set()
    cohort_by_stratum = dict.fromkeys(EXPECTED_STRATA, 0)
    supported_by_stratum = dict.fromkeys(EXPECTED_STRATA, 0)
    for row in objects:
        if not isinstance(row, Mapping):
            raise ValueError(f"{artifact} object row is malformed")
        object_id = row.get("object_id")
        if type(object_id) is not str or not object_id:
            raise ValueError(f"{artifact} object identity is invalid")
        episode_id = row.get("episode_id")
        if type(episode_id) is not int or episode_id < 0:
            raise ValueError(f"{artifact} episode identity is invalid")
        stratum = row.get("stratum")
        if stratum not in EXPECTED_STRATA:
            raise ValueError(f"{artifact} object stratum changed")
        identity = (object_id, episode_id, stratum)
        if identity in identities or object_id in object_ids:
            raise ValueError(f"{artifact} object identities are duplicated")
        identities.add(identity)
        object_ids.add(object_id)
        status = row.get("status")
        if status not in allowed_statuses:
            raise ValueError(f"{artifact} object status changed")
        cohort_by_stratum[stratum] += 1
        if status == supported_status:
            supported_ids.add(object_id)
            supported_by_stratum[stratum] += 1
    if any(
        count != EXPECTED_OBJECTS_PER_STRATUM for count in cohort_by_stratum.values()
    ):
        raise ValueError(f"{artifact} cohort strata changed")
    return (
        frozenset(identities),
        frozenset(supported_ids),
        sum(supported_by_stratum.values()),
        supported_by_stratum,
    )


def validated_support_gate(
    value: Mapping[str, Any],
    *,
    artifact: str,
    object_supported: int,
    object_supported_by_stratum: Mapping[str, int],
) -> dict[str, Any]:
    """Validate the exact frozen 8/10 and 4/5 support gate."""

    gate = value.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError(f"{artifact} gate is missing")
    expected_keys = {
        "supported_object_count",
        "supported_by_stratum",
        "minimum_supported_objects",
        "minimum_supported_per_stratum",
        "support_passed",
    }
    if set(gate) != expected_keys:
        raise ValueError(f"{artifact} gate fields changed")
    supported = integer_field(
        gate.get("supported_object_count"),
        name="supported_object_count",
        maximum=EXPECTED_OBJECT_COUNT,
    )
    by_stratum = gate.get("supported_by_stratum")
    if not isinstance(by_stratum, Mapping) or set(by_stratum) != set(EXPECTED_STRATA):
        raise ValueError(f"{artifact} supported_by_stratum changed")
    sheet = integer_field(
        by_stratum.get("sheet"),
        name="supported_by_stratum.sheet",
        maximum=EXPECTED_OBJECTS_PER_STRATUM,
    )
    volumetric = integer_field(
        by_stratum.get("volumetric"),
        name="supported_by_stratum.volumetric",
        maximum=EXPECTED_OBJECTS_PER_STRATUM,
    )
    observed_by_stratum = {"sheet": sheet, "volumetric": volumetric}
    if supported != sheet + volumetric:
        raise ValueError(f"{artifact} supported object counts disagree")
    if supported != object_supported or observed_by_stratum != dict(
        object_supported_by_stratum
    ):
        raise ValueError(f"{artifact} gate disagrees with object rows")
    minimum = integer_field(
        gate.get("minimum_supported_objects"),
        name="minimum_supported_objects",
        maximum=EXPECTED_OBJECT_COUNT,
    )
    minimum_per_stratum = integer_field(
        gate.get("minimum_supported_per_stratum"),
        name="minimum_supported_per_stratum",
        maximum=EXPECTED_OBJECTS_PER_STRATUM,
    )
    if minimum != MINIMUM_SUPPORTED_OBJECTS:
        raise ValueError(f"{artifact} minimum supported object count changed")
    if minimum_per_stratum != MINIMUM_SUPPORTED_PER_STRATUM:
        raise ValueError(f"{artifact} minimum supported per stratum changed")
    support_passed = gate.get("support_passed")
    if type(support_passed) is not bool:
        raise ValueError(f"{artifact} support_passed must be a boolean")
    expected_pass = (
        supported >= MINIMUM_SUPPORTED_OBJECTS
        and min(sheet, volumetric) >= MINIMUM_SUPPORTED_PER_STRATUM
    )
    if support_passed is not expected_pass:
        raise ValueError(f"{artifact} gate decision disagrees with its counts")
    return {
        "supported_object_count": supported,
        "supported_by_stratum": observed_by_stratum,
        "minimum_supported_objects": minimum,
        "minimum_supported_per_stratum": minimum_per_stratum,
        "support_passed": support_passed,
    }
