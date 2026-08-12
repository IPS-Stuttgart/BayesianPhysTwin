from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from bayesian_phystwin.prospective_protocol_authority_v1 import (
    AuthorityStatusV1,
    ProspectiveProtocolAuthorityEntryV1,
    ProspectiveProtocolAuthorityRegistryV1,
    build_prospective_protocol_authority_entry,
    load_prospective_protocol_authority_registry,
    lock_authoritative_prospective_study,
    require_authoritative_protocol,
    validate_authoritative_prospective_study_chain,
    write_prospective_protocol_authority_registry,
)
from bayesian_phystwin.prospective_study_lifecycle_v1 import (
    ProspectiveStudyProtocolV1,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _protocol(name: str) -> ProspectiveStudyProtocolV1:
    return ProspectiveStudyProtocolV1(
        protocol_id=f"{name}-v1",
        method_set_id=_digest(f"method:{name}"),
        decision_rule_id=_digest(f"decision:{name}"),
        fallback_identity_id=_digest(f"fallback:{name}"),
        information_boundary_id=_digest(f"boundary:{name}"),
        statistical_unit="physical object session",
        development_group_ids=(f"source-{name}",),
        calibration_group_ids=(f"calibration-{name}",),
        target_group_ids=(f"target-{name}",),
    )


def _entry(
    protocol: ProspectiveStudyProtocolV1,
    *,
    claim: str,
    status: AuthorityStatusV1,
    successor: str | None = None,
) -> ProspectiveProtocolAuthorityEntryV1:
    return build_prospective_protocol_authority_entry(
        protocol,
        claim_id=claim,
        authority_status=status,
        authority_decision_id=_digest(f"authority:{claim}:{protocol.protocol_id}"),
        superseded_by_protocol_content_id=successor,
    )


def test_registry_resolves_authority_and_canonicalizes_entry_order(tmp_path) -> None:
    old = _protocol("old")
    current = _protocol("current")
    archive = _protocol("archive")
    registry = ProspectiveProtocolAuthorityRegistryV1(
        entries=(
            _entry(current, claim="fresh-transfer", status="authoritative"),
            _entry(archive, claim="fresh-transfer", status="historical"),
            _entry(
                old,
                claim="fresh-transfer",
                status="superseded",
                successor=current.protocol_content_id,
            ),
        ),
        metadata={"owner": "paper claim registry"},
    )

    assert registry.authoritative_entry("fresh-transfer").protocol_id == "current-v1"
    assert [
        entry.protocol_id
        for entry in registry.supersession_chain(
            claim_id="fresh-transfer",
            protocol_content_id=old.protocol_content_id,
        )
    ] == ["old-v1", "current-v1"]
    assert registry.claim_ids == ("fresh-transfer",)

    reordered = ProspectiveProtocolAuthorityRegistryV1(
        entries=tuple(reversed(registry.entries)),
        metadata={"owner": "paper claim registry"},
    )
    assert reordered.registry_id == registry.registry_id

    path = tmp_path / "authority.json"
    write_prospective_protocol_authority_registry(registry, path)
    assert load_prospective_protocol_authority_registry(path) == registry
    with pytest.raises(FileExistsError):
        write_prospective_protocol_authority_registry(registry, path)


def test_registry_requires_exactly_one_authoritative_protocol_per_claim() -> None:
    first = _protocol("first")
    second = _protocol("second")

    with pytest.raises(ValueError, match="exactly one authoritative"):
        ProspectiveProtocolAuthorityRegistryV1(
            entries=(
                _entry(first, claim="claim", status="historical"),
                _entry(second, claim="claim", status="historical"),
            )
        )

    with pytest.raises(ValueError, match="exactly one authoritative"):
        ProspectiveProtocolAuthorityRegistryV1(
            entries=(
                _entry(first, claim="claim", status="authoritative"),
                _entry(second, claim="claim", status="authoritative"),
            )
        )


def test_registry_rejects_dangling_historical_and_cross_claim_successors() -> None:
    old = _protocol("old")
    current = _protocol("current")
    historical = _protocol("historical")

    with pytest.raises(ValueError, match="unregistered successor"):
        ProspectiveProtocolAuthorityRegistryV1(
            entries=(
                _entry(current, claim="claim", status="authoritative"),
                _entry(
                    old,
                    claim="claim",
                    status="superseded",
                    successor=_digest("missing"),
                ),
            )
        )

    with pytest.raises(ValueError, match="historical protocol"):
        ProspectiveProtocolAuthorityRegistryV1(
            entries=(
                _entry(current, claim="claim", status="authoritative"),
                _entry(historical, claim="claim", status="historical"),
                _entry(
                    old,
                    claim="claim",
                    status="superseded",
                    successor=historical.protocol_content_id,
                ),
            )
        )

    with pytest.raises(ValueError, match="unregistered successor"):
        ProspectiveProtocolAuthorityRegistryV1(
            entries=(
                _entry(current, claim="claim-a", status="authoritative"),
                _entry(historical, claim="claim-b", status="authoritative"),
                _entry(
                    old,
                    claim="claim-a",
                    status="superseded",
                    successor=historical.protocol_content_id,
                ),
            )
        )


def test_registry_rejects_supersession_cycles() -> None:
    first = _protocol("first")
    second = _protocol("second")
    current = _protocol("current")

    with pytest.raises(ValueError, match="supersession cycle"):
        ProspectiveProtocolAuthorityRegistryV1(
            entries=(
                _entry(current, claim="claim", status="authoritative"),
                _entry(
                    first,
                    claim="claim",
                    status="superseded",
                    successor=second.protocol_content_id,
                ),
                _entry(
                    second,
                    claim="claim",
                    status="superseded",
                    successor=first.protocol_content_id,
                ),
            )
        )


def test_registry_rejects_duplicate_protocol_names_and_content_ids() -> None:
    protocol = _protocol("same")
    authority = _entry(protocol, claim="claim", status="authoritative")

    with pytest.raises(ValueError, match="content identity"):
        ProspectiveProtocolAuthorityRegistryV1(entries=(authority, authority))

    renamed = replace(
        authority,
        protocol_content_id=_digest("different-content"),
        authority_status="historical",
    )
    with pytest.raises(ValueError, match="claim/protocol id"):
        ProspectiveProtocolAuthorityRegistryV1(entries=(authority, renamed))


def test_registry_mapping_roundtrip_detects_tampering() -> None:
    protocol = _protocol("current")
    registry = ProspectiveProtocolAuthorityRegistryV1(
        entries=(_entry(protocol, claim="claim", status="authoritative"),)
    )
    payload = registry.as_dict()
    restored = ProspectiveProtocolAuthorityRegistryV1.from_mapping(payload)
    assert restored.registry_id == registry.registry_id

    tampered = dict(payload)
    entries = [dict(item) for item in payload["entries"]]  # type: ignore[index]
    entries[0]["authority_decision_id"] = _digest("changed")
    tampered["entries"] = entries
    with pytest.raises(ValueError, match="entry identity mismatch"):
        ProspectiveProtocolAuthorityRegistryV1.from_mapping(tampered)


def test_require_authoritative_protocol_matches_both_name_and_content() -> None:
    current = _protocol("current")
    other = _protocol("other")
    registry = ProspectiveProtocolAuthorityRegistryV1(
        entries=(_entry(current, claim="claim", status="authoritative"),)
    )

    authority = require_authoritative_protocol(
        registry,
        claim_id="claim",
        protocol=current,
    )
    assert authority.protocol_content_id == current.protocol_content_id

    with pytest.raises(ValueError, match="not authoritative"):
        require_authoritative_protocol(
            registry,
            claim_id="claim",
            protocol=other,
        )


def test_same_protocol_may_be_authoritative_for_distinct_claims() -> None:
    protocol = _protocol("shared")
    registry = ProspectiveProtocolAuthorityRegistryV1(
        entries=(
            _entry(protocol, claim="accuracy", status="authoritative"),
            _entry(protocol, claim="calibration", status="authoritative"),
        )
    )

    assert registry.claim_ids == ("accuracy", "calibration")


def test_authoritative_design_lock_binds_registry_and_entry_identity() -> None:
    current = _protocol("current")
    registry = ProspectiveProtocolAuthorityRegistryV1(
        entries=(_entry(current, claim="claim", status="authoritative"),)
    )

    state = lock_authoritative_prospective_study(
        registry,
        claim_id="claim",
        protocol=current,
        metadata={"operator": "source-only"},
    )
    authority = registry.authoritative_entry("claim")
    assert state.metadata["authority_registry_id"] == registry.registry_id
    assert state.metadata["authority_entry_id"] == authority.entry_id
    assert state.metadata["authority_claim_id"] == "claim"
    validate_authoritative_prospective_study_chain(
        registry,
        claim_id="claim",
        protocol=current,
        states=(state,),
    )


def test_authoritative_design_lock_rejects_override_and_tampering() -> None:
    current = _protocol("current")
    registry = ProspectiveProtocolAuthorityRegistryV1(
        entries=(_entry(current, claim="claim", status="authoritative"),)
    )

    with pytest.raises(ValueError, match="cannot override authority bindings"):
        lock_authoritative_prospective_study(
            registry,
            claim_id="claim",
            protocol=current,
            metadata={"authority_registry_id": _digest("forged")},
        )

    state = lock_authoritative_prospective_study(
        registry,
        claim_id="claim",
        protocol=current,
    )
    tampered = replace(
        state,
        metadata={
            **dict(state.metadata),
            "authority_entry_id": _digest("forged"),
        },
    )
    with pytest.raises(ValueError, match="does not bind the supplied authority"):
        validate_authoritative_prospective_study_chain(
            registry,
            claim_id="claim",
            protocol=current,
            states=(tampered,),
        )


def test_entry_and_registry_reject_coercion_and_invalid_status_shape() -> None:
    current = _protocol("current")
    authority = _entry(current, claim="claim", status="authoritative")

    with pytest.raises(ValueError, match="cannot bind a successor"):
        replace(
            authority,
            superseded_by_protocol_content_id=_digest("successor"),
        )

    payload = ProspectiveProtocolAuthorityRegistryV1(entries=(authority,)).as_dict()
    wrong_version = dict(payload)
    wrong_version["schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        ProspectiveProtocolAuthorityRegistryV1.from_mapping(wrong_version)

    wrong_status = dict(payload)
    entries = [dict(item) for item in payload["entries"]]  # type: ignore[index]
    entries[0]["authority_status"] = 1
    wrong_status["entries"] = entries
    with pytest.raises(ValueError, match="authority status"):
        ProspectiveProtocolAuthorityRegistryV1.from_mapping(wrong_status)
