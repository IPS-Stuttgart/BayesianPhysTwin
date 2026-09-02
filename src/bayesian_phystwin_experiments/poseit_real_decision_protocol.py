"""Validate the source-only pre-outcome PoseIt real-decision protocol."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SCHEMA_VERSION = 1
PROTOCOL_ID = "poseit-real-decision-probe-protocol-v1"
POSEIT_REPOSITORY_REVISION = "5e290eb024f25b1f4aa602724e6869e512aca434"
POSEIT_DRIVE_FOLDER_ID = "1CQiMPBEVvRMrDBSIRVeuwyuUOCOesfMc"
POSEIT_GELSIGHT_FILE_ID = "1EitCcpHoPEQKnlpqWKp02io2WpIxBrEe"
POSEIT_WEISS_FILE_ID = "124lVr6WTHDo5XbIGO-DWWcNpq7BBm8Mg"
OBJECT_COUNT = 26
FIT_COUNT = 10
CALIBRATION_COUNT = 5
SOURCE_TEST_COUNT = 5
CONFIRMATION_COUNT = 6
SPLIT_DOMAIN = "poseit-real-decision-object-split-v1"
MANDATORY_ANCHOR = 1
SELECTABLE_POSES = tuple(range(2, 17))
BUDGETS = (0, 1, 2, 3)
_CANONICAL_CONFIG_SHA256 = (
    "fa49b7c2d20d02d554f9d38b6025839d583493bf2a28dbea29a17cb804e66504"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, message: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), message)
    return cast(Mapping[str, Any], value)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def poseit_protocol_config_sha256(payload: Mapping[str, Any]) -> str:
    """Return the canonical digest of a parsed PoseIt protocol."""

    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def poseit_protocol_file_sha256(path: str | Path) -> str:
    """Return the byte digest of a PoseIt protocol file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_object_token(value: str) -> str:
    """Canonicalize an archive-derived object token for split assignment."""

    token = unicodedata.normalize("NFKC", value).casefold().strip()
    _require(bool(token), "object token is empty")
    return token


@dataclass(frozen=True)
class PoseItObjectCohort:
    """Object roles assigned without reading any PoseIt phase label."""

    fit: tuple[str, ...]
    calibration: tuple[str, ...]
    source_test: tuple[str, ...]
    confirmation: tuple[str, ...]

    @property
    def all_objects(self) -> tuple[str, ...]:
        return self.fit + self.calibration + self.source_test + self.confirmation

    def role(self, object_token: str) -> str:
        """Return the frozen role of one canonical object token."""

        token = canonical_object_token(object_token)
        if token in self.fit:
            return "fit"
        if token in self.calibration:
            return "calibration"
        if token in self.source_test:
            return "source_test"
        if token in self.confirmation:
            return "confirmation"
        raise KeyError(f"unregistered PoseIt object token: {token}")


def derive_poseit_object_cohort(object_tokens: Sequence[str]) -> PoseItObjectCohort:
    """Assign the exact hash-ordered object split without consulting outcomes."""

    canonical = tuple(canonical_object_token(value) for value in object_tokens)
    _require(len(canonical) == OBJECT_COUNT, "PoseIt object count changed")
    _require(len(set(canonical)) == OBJECT_COUNT, "PoseIt object tokens are not unique")

    def key(token: str) -> tuple[str, str]:
        payload = f"{SPLIT_DOMAIN}\0{token}".encode()
        return hashlib.sha256(payload).hexdigest(), token

    ordered = tuple(sorted(canonical, key=key))
    fit_end = FIT_COUNT
    calibration_end = fit_end + CALIBRATION_COUNT
    source_end = calibration_end + SOURCE_TEST_COUNT
    return PoseItObjectCohort(
        fit=ordered[:fit_end],
        calibration=ordered[fit_end:calibration_end],
        source_test=ordered[calibration_end:source_end],
        confirmation=ordered[source_end:],
    )


def _validate_dataset(payload: Mapping[str, Any]) -> None:
    dataset = _mapping(payload.get("dataset"), message="dataset lock is missing")
    _require(
        dataset.get("release_repository_revision") == POSEIT_REPOSITORY_REVISION,
        "PoseIt repository revision changed",
    )
    counts = dataset.get("counts")
    _require(
        counts
        == {
            "grasp_datapoints": 1840,
            "holding_poses": 16,
            "household_objects": 26,
        },
        "PoseIt release counts changed",
    )
    archive_root = _mapping(
        dataset.get("archive_root"), message="archive root is missing"
    )
    _require(
        archive_root.get("google_drive_folder_id") == POSEIT_DRIVE_FOLDER_ID,
        "PoseIt archive root changed",
    )
    archive = _mapping(dataset.get("archive"), message="primary archive is missing")
    _require(archive.get("file_name") == "gelsight.zip", "primary archive changed")
    _require(
        archive.get("google_drive_file_id") == POSEIT_GELSIGHT_FILE_ID,
        "primary archive file ID changed",
    )
    _require(archive.get("expected_sha256") is None, "archive hash was set early")
    _require(archive.get("size_bytes") is None, "archive size was set early")
    _require(
        archive.get("status")
        == "public-locator-visible-byte-acquisition-quota-blocked",
        "preaccess archive status changed",
    )
    secondary = _mapping(
        dataset.get("secondary_archive"),
        message="secondary archive boundary is missing",
    )
    _require(secondary.get("admitted") is False, "secondary archive was admitted")
    _require(
        secondary.get("google_drive_file_id") == POSEIT_WEISS_FILE_ID,
        "secondary archive file ID changed",
    )


def _validate_cohort(payload: Mapping[str, Any]) -> None:
    cohort = _mapping(payload.get("cohort"), message="cohort rule is missing")
    _require(
        cohort.get("expected_object_count") == OBJECT_COUNT, "object count changed"
    )
    _require(cohort.get("pairwise_disjoint") is True, "cohort disjointness changed")
    assignment = _mapping(
        cohort.get("assignment"), message="cohort assignment is missing"
    )
    _require(assignment.get("domain_separator") == SPLIT_DOMAIN, "split domain changed")
    _require(assignment.get("fit_count") == FIT_COUNT, "fit count changed")
    _require(
        assignment.get("calibration_count") == CALIBRATION_COUNT,
        "calibration count changed",
    )
    _require(
        assignment.get("source_test_count") == SOURCE_TEST_COUNT,
        "source-test count changed",
    )
    _require(
        assignment.get("confirmation_count") == CONFIRMATION_COUNT,
        "confirmation count changed",
    )


def _validate_method(payload: Mapping[str, Any]) -> None:
    method = _mapping(payload.get("method"), message="method lock is missing")
    anchor = _mapping(method.get("mandatory_anchor"), message="anchor is missing")
    _require(anchor.get("holding_pose") == MANDATORY_ANCHOR, "anchor changed")
    probes = _mapping(method.get("selectable_probes"), message="probes are missing")
    _require(
        tuple(probes.get("holding_poses", ())) == SELECTABLE_POSES,
        "selectable poses changed",
    )
    _require(probes.get("shake_outcome_revealed") is False, "probe outcome opened")
    model = _mapping(method.get("model_family"), message="model family is missing")
    _require(model.get("prob4d_used") is False, "Prob4D declaration changed")
    selectors = _mapping(
        method.get("probe_selectors"), message="probe selectors are missing"
    )
    decision_directed = _mapping(
        selectors.get("decision_directed"),
        message="decision-directed selector is missing",
    )
    _require(
        decision_directed.get("predictive_draw_count") == 4096,
        "predictive draw count changed",
    )
    _require(
        decision_directed.get("predictive_seed") == 20260902,
        "predictive seed changed",
    )
    certificate = _mapping(
        method.get("shared_certificate"), message="certificate is missing"
    )
    _require(certificate.get("coverage") == 0.8, "certificate coverage changed")
    _require(
        certificate.get("shared_across_methods") is True,
        "certificate is not shared",
    )
    _require(
        certificate.get("uses_confirmation") is False, "confirmation entered guard"
    )


def _validate_decision_and_evaluation(payload: Mapping[str, Any]) -> None:
    decision = _mapping(payload.get("decision"), message="decision rule is missing")
    _require(decision.get("stable_label") == "Pass", "stable label changed")
    _require(
        tuple(decision.get("unstable_labels", ())) == ("Slip", "Drop", "Not present"),
        "unstable label mapping changed",
    )
    _require(decision.get("stable_utility") == 1.0, "stable utility changed")
    _require(decision.get("unstable_utility") == -1.0, "unstable utility changed")
    _require(decision.get("abstention_utility") == 0.0, "abstention changed")
    evaluation = _mapping(payload.get("evaluation"), message="evaluation is missing")
    budgets = _mapping(evaluation.get("budgets"), message="budgets are missing")
    _require(
        tuple(budgets.get("additional_probe_counts", ())) == BUDGETS,
        "probe budgets changed",
    )
    inference = _mapping(
        evaluation.get("statistical_inference"), message="inference is missing"
    )
    _require(inference.get("statistical_unit") == "object", "unit changed")
    _require(inference.get("bootstrap_seed") == 20260902, "bootstrap seed changed")


def _validate_boundary(payload: Mapping[str, Any]) -> None:
    boundary = _mapping(
        payload.get("information_boundary"), message="information boundary is missing"
    )
    for key in (
        "archive_member_names_open_before_protocol_freeze",
        "archive_payload_bytes_acquired_before_protocol_freeze",
        "confirmation_outcomes_open_before_authorization",
        "held_v8_access_allowed",
        "method_changes_after_source_test",
        "object_tokens_open_before_protocol_freeze",
        "phase_labels_open_before_protocol_freeze",
        "repository_object_image_bytes_opened",
        "repository_readme_object_catalog_used_for_assignment_or_method_design",
    ):
        _require(boundary.get(key) is False, f"information boundary changed: {key}")
    _require(
        boundary.get("repository_readme_object_catalog_opened") is True,
        "public README disclosure record changed",
    )
    for key in (
        "target_open_requires_archive_hash",
        "target_open_requires_exact_method_seal",
        "target_open_requires_source_gate",
        "target_open_requires_write_once_authorization",
    ):
        _require(boundary.get(key) is True, f"information boundary changed: {key}")
    promotion = _mapping(payload.get("promotion"), message="promotion lock is missing")
    _require(promotion.get("target_authorized") is False, "target was authorized early")
    _require(promotion.get("target_attempt_limit") == 1, "attempt limit changed")


def load_poseit_real_decision_protocol(path: str | Path) -> dict[str, Any]:
    """Load the exact pre-outcome PoseIt protocol and reject drift."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "protocol must be a JSON object")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema changed")
    _require(payload.get("contract") == PROTOCOL_ID, "protocol ID changed")
    _require(
        payload.get("status") == "frozen-before-archive-outcome-access",
        "protocol status changed",
    )
    _validate_dataset(payload)
    _validate_cohort(payload)
    _validate_method(payload)
    _validate_decision_and_evaluation(payload)
    _validate_boundary(payload)
    digest = poseit_protocol_config_sha256(payload)
    _require(digest == _CANONICAL_CONFIG_SHA256, "protocol configuration drifted")
    return dict(payload)


__all__ = [
    "BUDGETS",
    "CALIBRATION_COUNT",
    "CONFIRMATION_COUNT",
    "FIT_COUNT",
    "MANDATORY_ANCHOR",
    "OBJECT_COUNT",
    "POSEIT_DRIVE_FOLDER_ID",
    "POSEIT_GELSIGHT_FILE_ID",
    "POSEIT_REPOSITORY_REVISION",
    "PROTOCOL_ID",
    "PoseItObjectCohort",
    "SELECTABLE_POSES",
    "SOURCE_TEST_COUNT",
    "SPLIT_DOMAIN",
    "canonical_object_token",
    "derive_poseit_object_cohort",
    "load_poseit_real_decision_protocol",
    "poseit_protocol_config_sha256",
    "poseit_protocol_file_sha256",
]
