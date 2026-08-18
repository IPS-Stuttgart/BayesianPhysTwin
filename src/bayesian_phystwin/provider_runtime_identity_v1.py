"""Typed, content-addressed Prob4D runtime identity for downstream handoffs.

A runtime evidence *source* such as ``installed_vcs_metadata`` describes how a
revision was verified; it is not itself a source revision.  This module keeps
those concepts separate and binds the independently observed commit to the
provider manifest before BayesianPhysTwin hands evidence to another repository.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ._canonical_contracts import (
    frozen_finite_json_mapping,
    literal_lower_hex,
    plain_json,
)
from .prob4d_provider_attestation import (
    PROB4D_PROVIDER_SOURCE_REPOSITORY,
    validate_prob4d_provider_attestation,
)

PROB4D_RUNTIME_IDENTITY_SCHEMA = "bayesian_phystwin.prob4d_runtime_identity"
PROB4D_RUNTIME_IDENTITY_VERSION = 1
PROB4D_PROJECT_ID = "prob4d"
PROB4D_REPOSITORY_IDENTITIES = frozenset(
    {
        PROB4D_PROVIDER_SOURCE_REPOSITORY,
        "IPS-Stuttgart/Prob4D",
    }
)
_RUNTIME_EVIDENCE_SOURCES = frozenset(
    {
        "installed_vcs_metadata",
        "source_checkout",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "project_id",
        "source_repository",
        "provider_manifest_id",
        "expected_revision",
        "observed_revision",
        "revision_evidence_source",
        "clean_checkout",
        "independently_verified",
        "metadata",
        "identity_id",
    }
)


def _require_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: object, *, name: str) -> str:
    try:
        return literal_lower_hex(value, name=name, lengths={64})
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error


def _require_revision(value: object, *, name: str) -> str:
    try:
        return literal_lower_hex(value, name=name, lengths={40, 64})
    except ValueError as error:
        raise ValueError(f"{name} must be an exact lowercase Git commit") from error


def _canonical_id(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Prob4DRuntimeIdentityV1:
    """Bind a claim-bearing Prob4D manifest to its executing Git commit."""

    project_id: str
    source_repository: str
    provider_manifest_id: str
    expected_revision: str
    observed_revision: str
    revision_evidence_source: str
    clean_checkout: bool | None
    independently_verified: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        project_id = _require_string(self.project_id, name="project_id")
        if project_id != PROB4D_PROJECT_ID:
            raise ValueError("project_id must identify the stable Prob4D project")
        source_repository = _require_string(
            self.source_repository,
            name="source_repository",
        )
        if source_repository not in PROB4D_REPOSITORY_IDENTITIES:
            raise ValueError("source_repository is not a registered Prob4D identity")
        provider_manifest_id = _require_sha256(
            self.provider_manifest_id,
            name="provider_manifest_id",
        )
        expected_revision = _require_revision(
            self.expected_revision,
            name="expected_revision",
        )
        observed_revision = _require_revision(
            self.observed_revision,
            name="observed_revision",
        )
        if observed_revision != expected_revision:
            raise ValueError("observed_revision differs from expected_revision")
        evidence_source = _require_string(
            self.revision_evidence_source,
            name="revision_evidence_source",
        )
        if evidence_source not in _RUNTIME_EVIDENCE_SOURCES:
            raise ValueError(
                "claim-bearing runtime identity requires independent VCS evidence"
            )
        clean_checkout = self.clean_checkout
        if evidence_source == "source_checkout":
            if type(clean_checkout) is not bool:
                raise TypeError(
                    "source-checkout runtime identity must declare cleanliness"
                )
            if clean_checkout is not True:
                raise ValueError("claim-bearing source checkout must be clean")
        elif clean_checkout is not None:
            raise ValueError(
                "non-checkout runtime evidence cannot declare checkout cleanliness"
            )
        if type(self.independently_verified) is not bool:
            raise TypeError("independently_verified must be a bool")
        if self.independently_verified is not True:
            raise ValueError(
                "claim-bearing runtime identity must be independently verified"
            )
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "source_repository", source_repository)
        object.__setattr__(self, "provider_manifest_id", provider_manifest_id)
        object.__setattr__(self, "expected_revision", expected_revision)
        object.__setattr__(self, "observed_revision", observed_revision)
        object.__setattr__(self, "revision_evidence_source", evidence_source)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(
                self.metadata,
                name="runtime identity metadata",
            ),
        )

    @property
    def runtime_revision(self) -> str:
        """Return the executing commit, never the evidence-source label."""

        return self.observed_revision

    @property
    def runtime_revision_source(self) -> str:
        """Compatibility alias for the VCS evidence-source label."""

        return self.revision_evidence_source

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema": PROB4D_RUNTIME_IDENTITY_SCHEMA,
            "schema_version": PROB4D_RUNTIME_IDENTITY_VERSION,
            "project_id": self.project_id,
            "source_repository": self.source_repository,
            "provider_manifest_id": self.provider_manifest_id,
            "expected_revision": self.expected_revision,
            "observed_revision": self.observed_revision,
            "revision_evidence_source": self.revision_evidence_source,
            "clean_checkout": self.clean_checkout,
            "independently_verified": self.independently_verified,
            "metadata": plain_json(self.metadata),
        }

    @property
    def identity_id(self) -> str:
        return _canonical_id(self._identity_payload())

    @property
    def artifact_id(self) -> str:
        return self.identity_id

    def to_record(self) -> dict[str, object]:
        return {**self._identity_payload(), "identity_id": self.identity_id}

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> Prob4DRuntimeIdentityV1:
        if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
            raise ValueError("runtime identity record must be a string-keyed mapping")
        missing = sorted(_RECORD_FIELDS - set(value))
        unexpected = sorted(set(value) - _RECORD_FIELDS)
        if missing or unexpected:
            raise ValueError(
                "runtime identity fields changed; "
                f"missing={missing}, unexpected={unexpected}"
            )
        if value["schema"] != PROB4D_RUNTIME_IDENTITY_SCHEMA:
            raise ValueError("unsupported runtime identity schema")
        if value["schema_version"] != PROB4D_RUNTIME_IDENTITY_VERSION:
            raise ValueError("unsupported runtime identity schema version")
        identity = cls(
            project_id=value["project_id"],
            source_repository=value["source_repository"],
            provider_manifest_id=value["provider_manifest_id"],
            expected_revision=value["expected_revision"],
            observed_revision=value["observed_revision"],
            revision_evidence_source=value["revision_evidence_source"],
            clean_checkout=value["clean_checkout"],
            independently_verified=value["independently_verified"],
            metadata=value["metadata"],
        )
        if value["identity_id"] != identity.identity_id:
            raise ValueError("runtime identity content address changed")
        return identity

    @classmethod
    def from_provider_attestation(
        cls,
        attestation: Mapping[str, Any],
        *,
        source_repository: str = PROB4D_PROVIDER_SOURCE_REPOSITORY,
        metadata: Mapping[str, Any] | None = None,
    ) -> Prob4DRuntimeIdentityV1:
        """Build an identity from an independently revalidated attestation."""

        if not isinstance(attestation, Mapping):
            raise TypeError("attestation must be a mapping")
        source_revision = _require_revision(
            attestation.get("provider_revision"),
            name="provider_revision",
        )
        validated = validate_prob4d_provider_attestation(
            attestation,
            source_revision=source_revision,
            require_claim_bearing=True,
        )
        runtime = validated["runtime_revision"]
        if not isinstance(runtime, Mapping):
            raise AssertionError("validated runtime attestation lost mapping type")
        observed = runtime.get("observed_revision")
        if observed is None:
            raise ValueError(
                "claim-bearing runtime attestation omits observed revision"
            )
        return cls(
            project_id=PROB4D_PROJECT_ID,
            source_repository=source_repository,
            provider_manifest_id=validated["provider_manifest_id"],
            expected_revision=runtime["expected_revision"],
            observed_revision=observed,
            revision_evidence_source=runtime["source"],
            clean_checkout=runtime["clean_checkout"],
            independently_verified=runtime["independently_verified"],
            metadata=metadata or {},
        )


__all__ = [
    "PROB4D_PROJECT_ID",
    "PROB4D_REPOSITORY_IDENTITIES",
    "PROB4D_RUNTIME_IDENTITY_SCHEMA",
    "PROB4D_RUNTIME_IDENTITY_VERSION",
    "Prob4DRuntimeIdentityV1",
]
