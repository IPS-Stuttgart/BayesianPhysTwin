"""Access-closed feasibility contract for a 4D-DRESS competence study."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
)

PROTOCOL_SCHEMA: Final = (
    "bayesian-phystwin.fourddress-query-competence-feasibility-boundary"
)
PROTOCOL_VERSION: Final = 1
PARTICIPANT_SPLIT_SCHEMA: Final = (
    "bayesian-phystwin.fourddress-query-competence-participant-split"
)
PARTICIPANT_SPLIT_VERSION: Final = 1

FOURDDRESS_REPOSITORY: Final = "eth-ait/4d-dress"
FOURDDRESS_REVISION: Final = "d1685e18b438587f00227df41ec7659e67f04df1"
HOOD_REPOSITORY: Final = "Dolorousrtur/HOOD"
HOOD_REVISION: Final = "9bc1076195979ac6c027fdd729c6e960cad62f2a"
PARTICIPANT_SELECTION_SALT: Final = "fourddress-query-competence-participant-v1"
REQUIRED_PARTICIPANT_COUNT: Final = 32
METHOD_SELECTION_COUNT: Final = 8
CERTIFICATION_COUNT: Final = 24

_PROTOCOL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "protocol_id",
        "protocol_label",
        "claim_boundary",
        "upstreams",
        "intended_study",
        "access_audit",
        "unresolved_prerequisites",
        "information_boundary",
    }
)
_UPSTREAM_FIELDS: Final = frozenset(
    {
        "dataset_repository",
        "dataset_revision",
        "dataset_project_url",
        "simulator_repository",
        "simulator_revision",
    }
)
_STUDY_FIELDS: Final = frozenset(
    {
        "physical_group",
        "required_participant_count",
        "method_selection_count",
        "certification_count",
        "participant_selection_salt",
        "participant_selection_method",
        "candidate_family",
        "fallback_family",
        "risk_signal_family",
        "query_functional",
        "horizons_seconds",
        "primary_loss",
        "confidence_level",
        "maximum_accepted_harm_upper_bound",
        "minimum_accepted_certification_participants",
        "method_selection_precedes_certification",
        "participant_balanced_inference_required",
    }
)
_AUDIT_FIELDS: Final = frozenset(
    {
        "audit_date_utc",
        "hosts_checked",
        "active_lane_found",
        "reserved_cohort_found",
        "dataset_payload_found",
        "license_or_access_receipt_found",
        "active_process_found",
        "code_only_checkout_found",
    }
)
_PREREQUISITE_FIELDS: Final = frozenset(
    {
        "user_accepted_dataset_license_receipt_sha256",
        "dataset_payload_manifest_sha256",
        "participant_metadata_manifest_sha256",
        "hood_checkpoint_sha256",
        "smpl_asset_receipt_sha256",
        "source_adapter_qualification_id",
        "participant_split_id",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "public_metadata_read",
        "license_terms_accepted_by_agent",
        "dataset_payload_read",
        "participant_roster_read",
        "physical_outcomes_read",
        "source_execution_authorized",
        "certification_execution_authorized",
        "replacement_allowed",
    }
)

_EXPECTED_UPSTREAMS: Final = {
    "dataset_repository": FOURDDRESS_REPOSITORY,
    "dataset_revision": FOURDDRESS_REVISION,
    "dataset_project_url": "https://eth-ait.github.io/4d-dress/",
    "simulator_repository": HOOD_REPOSITORY,
    "simulator_revision": HOOD_REVISION,
}
_EXPECTED_STUDY: Final = {
    "physical_group": "participant",
    "required_participant_count": REQUIRED_PARTICIPANT_COUNT,
    "method_selection_count": METHOD_SELECTION_COUNT,
    "certification_count": CERTIFICATION_COUNT,
    "participant_selection_salt": PARTICIPANT_SELECTION_SALT,
    "participant_selection_method": "metadata-only-salted-sha256-ranking-v1",
    "candidate_family": "hood-general-garment-dynamics-v1",
    "fallback_family": "linear-blend-skinning-observed-pose-v1",
    "risk_signal_family": (
        "preoutcome-cross-runtime-disagreement-and-runtime-diagnostics-v1"
    ),
    "query_functional": "garment-surface-pointset-forecast-v1",
    "horizons_seconds": [0.25, 0.5, 1.0],
    "primary_loss": ("participant-balanced-symmetric-garment-surface-distance-mm-v1"),
    "confidence_level": 0.95,
    "maximum_accepted_harm_upper_bound": 0.2,
    "minimum_accepted_certification_participants": 14,
    "method_selection_precedes_certification": True,
    "participant_balanced_inference_required": True,
}
_EXPECTED_AUDIT: Final = {
    "audit_date_utc": "2026-08-30",
    "hosts_checked": ["gpuserver4090", "gpuserver6000"],
    "active_lane_found": False,
    "reserved_cohort_found": False,
    "dataset_payload_found": False,
    "license_or_access_receipt_found": False,
    "active_process_found": False,
    "code_only_checkout_found": True,
}
_EXPECTED_PREREQUISITES: Final = {
    "user_accepted_dataset_license_receipt_sha256": None,
    "dataset_payload_manifest_sha256": None,
    "participant_metadata_manifest_sha256": None,
    "hood_checkpoint_sha256": None,
    "smpl_asset_receipt_sha256": None,
    "source_adapter_qualification_id": None,
    "participant_split_id": None,
}
_EXPECTED_BOUNDARY: Final = {
    "public_metadata_read": True,
    "license_terms_accepted_by_agent": False,
    "dataset_payload_read": False,
    "participant_roster_read": False,
    "physical_outcomes_read": False,
    "source_execution_authorized": False,
    "certification_execution_authorized": False,
    "replacement_allowed": False,
}


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, Any], value)


def _exact_mapping(
    value: object,
    *,
    name: str,
    fields: frozenset[str],
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    mapping = _mapping(value, name=name)
    require_exact_fields(mapping, expected=fields, name=name)
    if dict(mapping) != dict(expected):
        raise ValueError(f"{name} changed")
    return mapping


@dataclass(frozen=True, slots=True)
class FourDDressCompetenceFeasibilityV1:
    """Frozen evidence that licensed-data execution is currently closed."""

    value: Mapping[str, Any]
    protocol_id: str

    @property
    def source_execution_authorized(self) -> bool:
        boundary = _mapping(
            self.value["information_boundary"],
            name="information boundary",
        )
        return bool(boundary["source_execution_authorized"])

    @property
    def certification_execution_authorized(self) -> bool:
        boundary = _mapping(
            self.value["information_boundary"],
            name="information boundary",
        )
        return bool(boundary["certification_execution_authorized"])

    @property
    def unresolved_prerequisites(self) -> tuple[str, ...]:
        prerequisites = _mapping(
            self.value["unresolved_prerequisites"],
            name="unresolved prerequisites",
        )
        return tuple(
            sorted(key for key, value in prerequisites.items() if value is None)
        )


def load_fourddress_competence_feasibility_v1(
    path: str | Path,
) -> FourDDressCompetenceFeasibilityV1:
    """Load and verify the exact access-closed feasibility artifact."""

    value = load_strict_json_object(path, label="4D-DRESS feasibility protocol")
    require_exact_fields(
        value,
        expected=_PROTOCOL_FIELDS,
        name="4D-DRESS feasibility protocol",
    )
    if value["schema"] != PROTOCOL_SCHEMA:
        raise ValueError("4D-DRESS feasibility schema changed")
    if value["schema_version"] != PROTOCOL_VERSION:
        raise ValueError("4D-DRESS feasibility version changed")

    protocol_id = exact_revision(value["protocol_id"], name="protocol_id")
    descriptor = dict(value)
    descriptor.pop("protocol_id")
    if content_id(descriptor) != protocol_id:
        raise ValueError("4D-DRESS feasibility protocol_id changed")

    _exact_mapping(
        value["upstreams"],
        name="4D-DRESS upstreams",
        fields=_UPSTREAM_FIELDS,
        expected=_EXPECTED_UPSTREAMS,
    )
    _exact_mapping(
        value["intended_study"],
        name="4D-DRESS intended study",
        fields=_STUDY_FIELDS,
        expected=_EXPECTED_STUDY,
    )
    _exact_mapping(
        value["access_audit"],
        name="4D-DRESS access audit",
        fields=_AUDIT_FIELDS,
        expected=_EXPECTED_AUDIT,
    )
    _exact_mapping(
        value["unresolved_prerequisites"],
        name="4D-DRESS unresolved prerequisites",
        fields=_PREREQUISITE_FIELDS,
        expected=_EXPECTED_PREREQUISITES,
    )
    _exact_mapping(
        value["information_boundary"],
        name="4D-DRESS information boundary",
        fields=_BOUNDARY_FIELDS,
        expected=_EXPECTED_BOUNDARY,
    )

    nonempty_string(value["protocol_label"], name="protocol_label")
    nonempty_string(value["claim_boundary"], name="claim_boundary")
    return FourDDressCompetenceFeasibilityV1(
        value=value,
        protocol_id=protocol_id,
    )


def _participant_rank(participant_id: str) -> tuple[str, str]:
    payload = f"{PARTICIPANT_SELECTION_SALT}\0{participant_id}".encode()
    return hashlib.sha256(payload).hexdigest(), participant_id


@dataclass(frozen=True, slots=True)
class FourDDressParticipantSplitV1:
    """Outcome-blind participant split derived from a names-only roster."""

    participant_manifest_id: str
    method_selection_participants: tuple[str, ...]
    certification_participants: tuple[str, ...]
    split_id: str

    def descriptor(self) -> dict[str, object]:
        return {
            "schema": PARTICIPANT_SPLIT_SCHEMA,
            "schema_version": PARTICIPANT_SPLIT_VERSION,
            "participant_manifest_id": self.participant_manifest_id,
            "participant_selection_salt": PARTICIPANT_SELECTION_SALT,
            "participant_selection_method": ("metadata-only-salted-sha256-ranking-v1"),
            "method_selection_participants": list(self.method_selection_participants),
            "certification_participants": list(self.certification_participants),
        }


def build_fourddress_participant_split_v1(
    participant_ids: Sequence[str],
) -> FourDDressParticipantSplitV1:
    """Hash-split exactly 32 names-only participant identifiers."""

    if isinstance(participant_ids, (str, bytes)):
        raise ValueError("participant_ids must be a sequence of strings")
    participants = tuple(
        nonempty_string(value, name="participant id") for value in participant_ids
    )
    if len(participants) != REQUIRED_PARTICIPANT_COUNT:
        raise ValueError(
            f"participant roster must contain {REQUIRED_PARTICIPANT_COUNT} entries"
        )
    if len(set(participants)) != len(participants):
        raise ValueError("participant roster must contain unique entries")
    if any(value.strip() != value for value in participants):
        raise ValueError("participant ids must be canonical strings")

    canonical = tuple(sorted(participants))
    participant_manifest_id = content_id(
        {
            "schema": "bayesian-phystwin.fourddress-participant-manifest",
            "schema_version": 1,
            "participant_ids": list(canonical),
        }
    )
    ranked = tuple(sorted(canonical, key=_participant_rank))
    method = ranked[:METHOD_SELECTION_COUNT]
    certification = ranked[METHOD_SELECTION_COUNT:]
    descriptor = {
        "schema": PARTICIPANT_SPLIT_SCHEMA,
        "schema_version": PARTICIPANT_SPLIT_VERSION,
        "participant_manifest_id": participant_manifest_id,
        "participant_selection_salt": PARTICIPANT_SELECTION_SALT,
        "participant_selection_method": ("metadata-only-salted-sha256-ranking-v1"),
        "method_selection_participants": list(method),
        "certification_participants": list(certification),
    }
    return FourDDressParticipantSplitV1(
        participant_manifest_id=participant_manifest_id,
        method_selection_participants=method,
        certification_participants=certification,
        split_id=content_id(descriptor),
    )
