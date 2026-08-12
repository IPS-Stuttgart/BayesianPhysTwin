"""Content-addressed authority registry for prospective study protocols.

The registry resolves which immutable protocol is authoritative for each
scientific claim without changing the protocol or its target-access lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import (
    content_id,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .prospective_study_lifecycle_v1 import (
    ProspectiveStudyProtocolV1,
    ProspectiveStudyStateV1,
    lock_prospective_study,
    validate_prospective_study_chain,
)

PROSPECTIVE_PROTOCOL_AUTHORITY_ENTRY_SCHEMA: Final = (
    "bayesian-phystwin.prospective-protocol-authority-entry-v1"
)
PROSPECTIVE_PROTOCOL_AUTHORITY_REGISTRY_SCHEMA: Final = (
    "bayesian-phystwin.prospective-protocol-authority-registry-v1"
)
PROSPECTIVE_PROTOCOL_AUTHORITY_SCHEMA_VERSION: Final = 1

AuthorityStatusV1 = Literal["authoritative", "superseded", "historical"]

_STATUSES: Final = frozenset({"authoritative", "superseded", "historical"})
_ENTRY_FIELDS: Final = frozenset(
    {
        "entry_id",
        "schema_name",
        "schema_version",
        "claim_id",
        "protocol_id",
        "protocol_content_id",
        "authority_status",
        "authority_decision_id",
        "superseded_by_protocol_content_id",
        "metadata",
    }
)
_AUTHORITY_METADATA_FIELDS: Final = frozenset(
    {
        "authority_claim_id",
        "authority_entry_id",
        "authority_registry_id",
    }
)
_REGISTRY_FIELDS: Final = frozenset(
    {
        "registry_id",
        "schema_name",
        "schema_version",
        "entries",
        "metadata",
    }
)


def _text(value: object, *, name: str) -> str:
    result = nonempty_string(value, name=name)
    if result.strip() != result or any(character in result for character in "\x00\r\n"):
        raise ValueError(f"{name} must be canonical single-line text")
    return result


def _version(value: object, *, name: str) -> int:
    if type(value) is not int or value != PROSPECTIVE_PROTOCOL_AUTHORITY_SCHEMA_VERSION:
        raise ValueError(f"unexpected {name} schema version")
    return value


def _status(value: object) -> AuthorityStatusV1:
    if type(value) is not str or value not in _STATUSES:
        raise ValueError("unsupported prospective-protocol authority status")
    return cast(AuthorityStatusV1, value)


def _optional_digest(value: object, *, name: str) -> str | None:
    return None if value is None else sha256_digest(value, name=name)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string object keys")
    return cast(Mapping[str, Any], value)


def _entry_sequence(value: object) -> tuple[ProspectiveProtocolAuthorityEntryV1, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("authority registry entries must be a JSON array")
    return tuple(
        ProspectiveProtocolAuthorityEntryV1.from_mapping(
            _mapping(item, name=f"authority registry entry {index}")
        )
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class ProspectiveProtocolAuthorityEntryV1:
    """Authority classification for one immutable protocol and one claim."""

    claim_id: str
    protocol_id: str
    protocol_content_id: str
    authority_status: AuthorityStatusV1
    authority_decision_id: str
    superseded_by_protocol_content_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, name="claim_id"))
        object.__setattr__(
            self,
            "protocol_id",
            _text(self.protocol_id, name="protocol_id"),
        )
        object.__setattr__(
            self,
            "protocol_content_id",
            sha256_digest(self.protocol_content_id, name="protocol_content_id"),
        )
        object.__setattr__(
            self,
            "authority_status",
            _status(self.authority_status),
        )
        object.__setattr__(
            self,
            "authority_decision_id",
            sha256_digest(self.authority_decision_id, name="authority_decision_id"),
        )
        object.__setattr__(
            self,
            "superseded_by_protocol_content_id",
            _optional_digest(
                self.superseded_by_protocol_content_id,
                name="superseded_by_protocol_content_id",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="authority entry metadata"),
        )
        self._validate_status_shape()

    def _validate_status_shape(self) -> None:
        successor = self.superseded_by_protocol_content_id
        if self.authority_status == "superseded":
            if successor is None:
                raise ValueError("superseded protocol requires a successor")
            if successor == self.protocol_content_id:
                raise ValueError("superseded protocol cannot point to itself")
            return
        if successor is not None:
            raise ValueError(
                f"{self.authority_status} protocol cannot bind a successor"
            )

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": PROSPECTIVE_PROTOCOL_AUTHORITY_ENTRY_SCHEMA,
            "schema_version": PROSPECTIVE_PROTOCOL_AUTHORITY_SCHEMA_VERSION,
            "claim_id": self.claim_id,
            "protocol_id": self.protocol_id,
            "protocol_content_id": self.protocol_content_id,
            "authority_status": self.authority_status,
            "authority_decision_id": self.authority_decision_id,
            "superseded_by_protocol_content_id": (
                self.superseded_by_protocol_content_id
            ),
            "metadata": plain_json(self.metadata),
        }

    @property
    def entry_id(self) -> str:
        return content_id(self.descriptor())

    def as_dict(self) -> dict[str, object]:
        return {"entry_id": self.entry_id, **self.descriptor()}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> ProspectiveProtocolAuthorityEntryV1:
        require_exact_fields(value, expected=_ENTRY_FIELDS, name="authority entry")
        if (
            _text(value["schema_name"], name="schema_name")
            != PROSPECTIVE_PROTOCOL_AUTHORITY_ENTRY_SCHEMA
        ):
            raise ValueError("unexpected prospective-protocol authority entry schema")
        _version(value["schema_version"], name="authority entry")
        result = cls(
            claim_id=_text(value["claim_id"], name="claim_id"),
            protocol_id=_text(value["protocol_id"], name="protocol_id"),
            protocol_content_id=sha256_digest(
                value["protocol_content_id"], name="protocol_content_id"
            ),
            authority_status=_status(value["authority_status"]),
            authority_decision_id=sha256_digest(
                value["authority_decision_id"], name="authority_decision_id"
            ),
            superseded_by_protocol_content_id=_optional_digest(
                value["superseded_by_protocol_content_id"],
                name="superseded_by_protocol_content_id",
            ),
            metadata=_mapping(value["metadata"], name="authority entry metadata"),
        )
        supplied = sha256_digest(value["entry_id"], name="entry_id")
        if supplied != result.entry_id:
            raise ValueError("prospective-protocol authority entry identity mismatch")
        return result


@dataclass(frozen=True)
class ProspectiveProtocolAuthorityRegistryV1:
    """One canonical registry with exactly one authority per scientific claim."""

    entries: tuple[ProspectiveProtocolAuthorityEntryV1, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.entries, (str, bytes))
            or not isinstance(self.entries, Sequence)
            or not self.entries
        ):
            raise ValueError("authority registry entries must be nonempty")
        if any(
            not isinstance(entry, ProspectiveProtocolAuthorityEntryV1)
            for entry in self.entries
        ):
            raise ValueError(
                "authority registry entries must contain authority-entry values"
            )
        entries = tuple(
            sorted(
                self.entries,
                key=lambda entry: (
                    entry.claim_id,
                    entry.protocol_id,
                    entry.protocol_content_id,
                ),
            )
        )
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="authority registry metadata",
            ),
        )
        self._validate_registry()

    def _validate_registry(self) -> None:
        by_key: dict[tuple[str, str], ProspectiveProtocolAuthorityEntryV1] = {}
        protocol_names: set[tuple[str, str]] = set()
        claims: dict[str, list[ProspectiveProtocolAuthorityEntryV1]] = {}
        for entry in self.entries:
            key = (entry.claim_id, entry.protocol_content_id)
            if key in by_key:
                raise ValueError(
                    "authority registry repeats a claim/protocol content identity"
                )
            by_key[key] = entry
            name_key = (entry.claim_id, entry.protocol_id)
            if name_key in protocol_names:
                raise ValueError("authority registry repeats a claim/protocol id")
            protocol_names.add(name_key)
            claims.setdefault(entry.claim_id, []).append(entry)
        for claim_id, entries in claims.items():
            authoritative = [
                entry for entry in entries if entry.authority_status == "authoritative"
            ]
            if len(authoritative) != 1:
                raise ValueError(
                    f"claim {claim_id!r} requires exactly one authoritative protocol"
                )
            authoritative_id = authoritative[0].protocol_content_id
            for entry in entries:
                if entry.authority_status != "superseded":
                    continue
                terminal_id = self._resolve_successor_chain(
                    claim_id=claim_id,
                    start=entry,
                    by_key=by_key,
                )
                if terminal_id != authoritative_id:
                    raise ValueError(
                        f"claim {claim_id!r} supersession chain does not end "
                        "at its authoritative protocol"
                    )

    @staticmethod
    def _resolve_successor_chain(
        *,
        claim_id: str,
        start: ProspectiveProtocolAuthorityEntryV1,
        by_key: Mapping[tuple[str, str], ProspectiveProtocolAuthorityEntryV1],
    ) -> str:
        seen: set[str] = set()
        current = start
        while current.authority_status == "superseded":
            if current.protocol_content_id in seen:
                raise ValueError(f"claim {claim_id!r} contains a supersession cycle")
            seen.add(current.protocol_content_id)
            successor_id = current.superseded_by_protocol_content_id
            if successor_id is None:
                raise ValueError("superseded protocol requires a successor")
            successor = by_key.get((claim_id, successor_id))
            if successor is None:
                raise ValueError(
                    f"claim {claim_id!r} references an unregistered successor"
                )
            if successor.authority_status == "historical":
                raise ValueError(
                    f"claim {claim_id!r} cannot supersede into a historical protocol"
                )
            current = successor
        return current.protocol_content_id

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": PROSPECTIVE_PROTOCOL_AUTHORITY_REGISTRY_SCHEMA,
            "schema_version": PROSPECTIVE_PROTOCOL_AUTHORITY_SCHEMA_VERSION,
            "entries": [entry.as_dict() for entry in self.entries],
            "metadata": plain_json(self.metadata),
        }

    @property
    def registry_id(self) -> str:
        return content_id(self.descriptor())

    @property
    def claim_ids(self) -> tuple[str, ...]:
        return tuple(sorted({entry.claim_id for entry in self.entries}))

    def as_dict(self) -> dict[str, object]:
        return {"registry_id": self.registry_id, **self.descriptor()}

    def authoritative_entry(
        self,
        claim_id: str,
    ) -> ProspectiveProtocolAuthorityEntryV1:
        claim = _text(claim_id, name="claim_id")
        matches = [
            entry
            for entry in self.entries
            if entry.claim_id == claim and entry.authority_status == "authoritative"
        ]
        if not matches:
            raise KeyError(claim)
        return matches[0]

    def entry(
        self,
        *,
        claim_id: str,
        protocol_content_id: str,
    ) -> ProspectiveProtocolAuthorityEntryV1:
        claim = _text(claim_id, name="claim_id")
        protocol = sha256_digest(
            protocol_content_id,
            name="protocol_content_id",
        )
        for entry in self.entries:
            if entry.claim_id == claim and entry.protocol_content_id == protocol:
                return entry
        raise KeyError((claim, protocol))

    def supersession_chain(
        self,
        *,
        claim_id: str,
        protocol_content_id: str,
    ) -> tuple[ProspectiveProtocolAuthorityEntryV1, ...]:
        current = self.entry(
            claim_id=claim_id,
            protocol_content_id=protocol_content_id,
        )
        chain = [current]
        while current.authority_status == "superseded":
            successor_id = current.superseded_by_protocol_content_id
            if successor_id is None:
                raise ValueError("superseded protocol requires a successor")
            current = self.entry(
                claim_id=current.claim_id,
                protocol_content_id=successor_id,
            )
            chain.append(current)
        return tuple(chain)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> ProspectiveProtocolAuthorityRegistryV1:
        require_exact_fields(
            value,
            expected=_REGISTRY_FIELDS,
            name="authority registry",
        )
        if (
            _text(value["schema_name"], name="schema_name")
            != PROSPECTIVE_PROTOCOL_AUTHORITY_REGISTRY_SCHEMA
        ):
            raise ValueError(
                "unexpected prospective-protocol authority registry schema"
            )
        _version(value["schema_version"], name="authority registry")
        result = cls(
            entries=_entry_sequence(value["entries"]),
            metadata=_mapping(value["metadata"], name="authority registry metadata"),
        )
        supplied = sha256_digest(value["registry_id"], name="registry_id")
        if supplied != result.registry_id:
            raise ValueError(
                "prospective-protocol authority registry identity mismatch"
            )
        return result


def build_prospective_protocol_authority_entry(
    protocol: ProspectiveStudyProtocolV1,
    *,
    claim_id: str,
    authority_status: AuthorityStatusV1,
    authority_decision_id: str,
    superseded_by_protocol_content_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveProtocolAuthorityEntryV1:
    """Bind authority metadata to one exact prospective-study protocol."""

    if not isinstance(protocol, ProspectiveStudyProtocolV1):
        raise TypeError("protocol must be a ProspectiveStudyProtocolV1")
    return ProspectiveProtocolAuthorityEntryV1(
        claim_id=claim_id,
        protocol_id=protocol.protocol_id,
        protocol_content_id=protocol.protocol_content_id,
        authority_status=authority_status,
        authority_decision_id=authority_decision_id,
        superseded_by_protocol_content_id=superseded_by_protocol_content_id,
        metadata={} if metadata is None else metadata,
    )


def require_authoritative_protocol(
    registry: ProspectiveProtocolAuthorityRegistryV1,
    *,
    claim_id: str,
    protocol: ProspectiveStudyProtocolV1,
) -> ProspectiveProtocolAuthorityEntryV1:
    """Require an exact protocol to be the sole authority for one claim."""

    if not isinstance(registry, ProspectiveProtocolAuthorityRegistryV1):
        raise TypeError("registry must be a ProspectiveProtocolAuthorityRegistryV1")
    if not isinstance(protocol, ProspectiveStudyProtocolV1):
        raise TypeError("protocol must be a ProspectiveStudyProtocolV1")
    authority = registry.authoritative_entry(claim_id)
    if (
        authority.protocol_id != protocol.protocol_id
        or authority.protocol_content_id != protocol.protocol_content_id
    ):
        raise ValueError("supplied protocol is not authoritative for the claim")
    return authority


def lock_authoritative_prospective_study(
    registry: ProspectiveProtocolAuthorityRegistryV1,
    *,
    claim_id: str,
    protocol: ProspectiveStudyProtocolV1,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveStudyStateV1:
    """Create a design lock bound to the sole protocol authority."""

    authority = require_authoritative_protocol(
        registry,
        claim_id=claim_id,
        protocol=protocol,
    )
    supplied = (
        {} if metadata is None else dict(_mapping(metadata, name="study lock metadata"))
    )
    collisions = sorted(_AUTHORITY_METADATA_FIELDS & set(supplied))
    if collisions:
        raise ValueError(
            f"study lock metadata cannot override authority bindings: {collisions}"
        )
    supplied.update(
        {
            "authority_claim_id": authority.claim_id,
            "authority_entry_id": authority.entry_id,
            "authority_registry_id": registry.registry_id,
        }
    )
    return lock_prospective_study(protocol, metadata=supplied)


def validate_authoritative_prospective_study_chain(
    registry: ProspectiveProtocolAuthorityRegistryV1,
    *,
    claim_id: str,
    protocol: ProspectiveStudyProtocolV1,
    states: Sequence[ProspectiveStudyStateV1],
) -> None:
    """Validate lifecycle ancestry and its exact authority design-lock binding."""

    authority = require_authoritative_protocol(
        registry,
        claim_id=claim_id,
        protocol=protocol,
    )
    validate_prospective_study_chain(protocol, states)
    first = states[0]
    metadata = plain_json(first.metadata)
    expected = {
        "authority_claim_id": authority.claim_id,
        "authority_entry_id": authority.entry_id,
        "authority_registry_id": registry.registry_id,
    }
    actual = {name: metadata.get(name) for name in _AUTHORITY_METADATA_FIELDS}
    if actual != expected:
        raise ValueError(
            "prospective-study design lock does not bind the supplied authority"
        )


def load_prospective_protocol_authority_registry(
    path: str | Path,
) -> ProspectiveProtocolAuthorityRegistryV1:
    """Load and fully revalidate one authority registry."""

    return ProspectiveProtocolAuthorityRegistryV1.from_mapping(
        load_strict_json_object(path, label="prospective-protocol authority registry")
    )


def write_prospective_protocol_authority_registry(
    registry: ProspectiveProtocolAuthorityRegistryV1,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Publish one canonical registry with atomic no-clobber semantics."""

    if not isinstance(registry, ProspectiveProtocolAuthorityRegistryV1):
        raise TypeError("registry must be a ProspectiveProtocolAuthorityRegistryV1")
    write_atomic_json(registry.as_dict(), path, overwrite=overwrite)


__all__ = [
    "PROSPECTIVE_PROTOCOL_AUTHORITY_ENTRY_SCHEMA",
    "PROSPECTIVE_PROTOCOL_AUTHORITY_REGISTRY_SCHEMA",
    "PROSPECTIVE_PROTOCOL_AUTHORITY_SCHEMA_VERSION",
    "AuthorityStatusV1",
    "ProspectiveProtocolAuthorityEntryV1",
    "ProspectiveProtocolAuthorityRegistryV1",
    "build_prospective_protocol_authority_entry",
    "load_prospective_protocol_authority_registry",
    "lock_authoritative_prospective_study",
    "require_authoritative_protocol",
    "validate_authoritative_prospective_study_chain",
    "write_prospective_protocol_authority_registry",
]
