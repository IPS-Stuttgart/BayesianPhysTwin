from __future__ import annotations

import copy
import hashlib
from typing import Any, cast

import pytest

from bayesian_phystwin.prospective_protocol_authority_v1 import (
    ProspectiveProtocolAuthorityEntryV1,
    ProspectiveProtocolAuthorityRegistryV1,
    build_prospective_protocol_authority_entry,
)
from bayesian_phystwin.prospective_study_lifecycle_v1 import (
    ProspectiveStudyProtocolV1,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protocol() -> ProspectiveStudyProtocolV1:
    return ProspectiveStudyProtocolV1(
        protocol_id="coverage-v1",
        method_set_id=_digest("method"),
        decision_rule_id=_digest("decision"),
        fallback_identity_id=_digest("fallback"),
        information_boundary_id=_digest("boundary"),
        statistical_unit="physical object session",
        development_group_ids=("source",),
        calibration_group_ids=("calibration",),
        target_group_ids=("target",),
    )


def _authority() -> ProspectiveProtocolAuthorityEntryV1:
    protocol = _protocol()
    return build_prospective_protocol_authority_entry(
        protocol,
        claim_id="claim",
        authority_status="authoritative",
        authority_decision_id=_digest("authority"),
    )


def _payload() -> dict[str, Any]:
    registry = ProspectiveProtocolAuthorityRegistryV1(entries=(_authority(),))
    return cast(dict[str, Any], registry.as_dict())


def test_registry_parser_rejects_noncanonical_text_and_container_shapes() -> None:
    bad_claim = copy.deepcopy(_payload())
    bad_claim["entries"][0]["claim_id"] = " claim"
    with pytest.raises(ValueError, match="canonical single-line text"):
        ProspectiveProtocolAuthorityRegistryV1.from_mapping(bad_claim)

    bad_metadata = copy.deepcopy(_payload())
    bad_metadata["entries"][0]["metadata"] = []
    with pytest.raises(ValueError, match="literal string object keys"):
        ProspectiveProtocolAuthorityRegistryV1.from_mapping(bad_metadata)

    bad_entries = copy.deepcopy(_payload())
    bad_entries["entries"] = "not-an-array"
    with pytest.raises(ValueError, match="must be a JSON array"):
        ProspectiveProtocolAuthorityRegistryV1.from_mapping(bad_entries)


def test_entry_rejects_missing_and_self_successors() -> None:
    protocol = _protocol()
    common = {
        "claim_id": "claim",
        "protocol_id": protocol.protocol_id,
        "protocol_content_id": protocol.protocol_content_id,
        "authority_status": "superseded",
        "authority_decision_id": _digest("authority"),
    }

    with pytest.raises(ValueError, match="requires a successor"):
        ProspectiveProtocolAuthorityEntryV1(**common)

    with pytest.raises(ValueError, match="cannot point to itself"):
        ProspectiveProtocolAuthorityEntryV1(
            **common,
            superseded_by_protocol_content_id=protocol.protocol_content_id,
        )


def test_registry_rejects_empty_and_nonentry_sequences() -> None:
    with pytest.raises(ValueError, match="must be nonempty"):
        ProspectiveProtocolAuthorityRegistryV1(entries=())

    with pytest.raises(ValueError, match="authority-entry values"):
        ProspectiveProtocolAuthorityRegistryV1(
            entries=cast(Any, (object(),)),
        )


def test_parsers_reject_wrong_schema_names() -> None:
    wrong_registry_schema = copy.deepcopy(_payload())
    wrong_registry_schema["schema_name"] = "other.registry"
    with pytest.raises(ValueError, match="authority registry schema"):
        ProspectiveProtocolAuthorityRegistryV1.from_mapping(wrong_registry_schema)

    wrong_entry_schema = copy.deepcopy(_payload())
    wrong_entry_schema["entries"][0]["schema_name"] = "other.entry"
    with pytest.raises(ValueError, match="authority entry schema"):
        ProspectiveProtocolAuthorityRegistryV1.from_mapping(wrong_entry_schema)
