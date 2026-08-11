"""Accepted and exact-fallback artifacts for the three-repository golden path."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from ._canonical_contracts import frozen_finite_json_mapping, plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    require_exact_fields,
    sha256_digest,
    write_atomic_json,
)
from .gauge_aware_belief import GaugeAwareSelection

GOLDEN_PATH_SELECTION_SCHEMA = (
    "bayesian_phystwin.three_repository_golden_path_selection"
)
GOLDEN_PATH_SELECTION_SCHEMA_VERSION = 1
GOLDEN_PATH_BUNDLE_SCHEMA = (
    "bayesian_phystwin.three_repository_golden_path_bundle"
)
GOLDEN_PATH_BUNDLE_SCHEMA_VERSION = 1

GoldenPathDecision = Literal["accepted", "rejected"]
_COMPONENT_KEYS = ("bayesian_phystwin", "prob4d", "causal4d")
_ARRAY_FIELDS = frozenset(
    {"array_id", "dtype", "shape", "nbytes", "payload_sha256"}
)
_SELECTION_FIELDS = frozenset(
    {
        "artifact_id",
        "schema_name",
        "schema_version",
        "case_id",
        "protocol_id",
        "decision",
        "reason",
        "inference_admissible",
        "regret_guard_present",
        "regret_guard_accepted",
        "candidate_accepted",
        "observation_artifact_id",
        "twin_belief_id",
        "physical_posterior_id",
        "provider_manifest_id",
        "run_manifest_id",
        "evidence_fingerprint",
        "repository_revisions",
        "wheel_sha256",
        "package_versions",
        "baseline_identity",
        "candidate_identity",
        "selected_identity",
        "exact_fallback_identity",
        "metadata",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "bundle_id",
        "schema_name",
        "schema_version",
        "accepted",
        "rejected",
    }
)


def _literal_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a literal boolean")
    return cast(bool, value)


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return cast(int, value)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} must use literal string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[Any], value)


def _component_mapping(
    value: Mapping[str, str],
    *,
    name: str,
    validator: Any,
) -> Mapping[str, Any]:
    payload = _mapping(value, name=name)
    if tuple(sorted(payload)) != tuple(sorted(_COMPONENT_KEYS)):
        raise ValueError(f"{name} must contain the complete component roster")
    normalized = {
        key: validator(payload[key], name=f"{name}.{key}")
        for key in _COMPONENT_KEYS
    }
    return frozen_finite_json_mapping(normalized, name=name)


def _package_version(value: object, *, name: str) -> str:
    version = nonempty_string(value, name=name)
    if version != version.strip() or any(ord(character) < 32 for character in version):
        raise ValueError(f"{name} must be canonical text")
    return version


@dataclass(frozen=True, slots=True)
class ArrayByteIdentityV1:
    """Exact dtype, shape, and byte identity of one finite real array."""

    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    payload_sha256: str

    def __post_init__(self) -> None:
        dtype_text = nonempty_string(self.dtype, name="array dtype")
        try:
            dtype = np.dtype(dtype_text)
        except TypeError as error:
            raise ValueError("array dtype is invalid") from error
        if dtype.str != dtype_text or dtype.kind not in "biuf":
            raise ValueError("array dtype must be canonical finite real storage")
        if isinstance(self.shape, (str, bytes)):
            raise ValueError("array shape must be an integer tuple")
        shape = tuple(
            _positive_integer(value, name=f"array shape[{index}]")
            for index, value in enumerate(self.shape)
        )
        if not shape or any(dimension == 0 for dimension in shape):
            raise ValueError("array shape must be nonempty and strictly positive")
        nbytes = _positive_integer(self.nbytes, name="array nbytes")
        expected_nbytes = math.prod(shape) * dtype.itemsize
        if nbytes != expected_nbytes:
            raise ValueError("array nbytes does not match dtype and shape")
        object.__setattr__(self, "dtype", dtype_text)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "nbytes", nbytes)
        object.__setattr__(
            self,
            "payload_sha256",
            sha256_digest(self.payload_sha256, name="array payload SHA-256"),
        )

    @classmethod
    def from_array(cls, value: object) -> ArrayByteIdentityV1:
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
            raise ValueError("golden-path arrays must contain finite real values")
        canonical = np.ascontiguousarray(array)
        import hashlib

        return cls(
            dtype=canonical.dtype.str,
            shape=canonical.shape,
            nbytes=canonical.nbytes,
            payload_sha256=hashlib.sha256(
                canonical.tobytes(order="C")
            ).hexdigest(),
        )

    @property
    def array_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
            "payload_sha256": self.payload_sha256,
        }

    def as_dict(self) -> dict[str, object]:
        return {"array_id": self.array_id, **self.descriptor()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ArrayByteIdentityV1:
        payload = _mapping(value, name="array identity")
        require_exact_fields(
            payload,
            expected=_ARRAY_FIELDS,
            name="array identity",
        )
        shape = tuple(
            _positive_integer(item, name=f"array shape[{index}]")
            for index, item in enumerate(
                _sequence(payload["shape"], name="array shape")
            )
        )
        identity = cls(
            dtype=nonempty_string(payload["dtype"], name="array dtype"),
            shape=shape,
            nbytes=_positive_integer(payload["nbytes"], name="array nbytes"),
            payload_sha256=sha256_digest(
                payload["payload_sha256"],
                name="array payload SHA-256",
            ),
        )
        expected = sha256_digest(payload["array_id"], name="array ID")
        if identity.array_id != expected:
            raise ValueError("array ID does not match its descriptor")
        return identity


@dataclass(frozen=True, slots=True)
class GoldenPathSelectionArtifactV1:
    """One admitted candidate or exact baseline-fallback decision."""

    case_id: str
    protocol_id: str
    decision: GoldenPathDecision
    reason: str
    inference_admissible: bool
    regret_guard_present: bool
    regret_guard_accepted: bool
    candidate_accepted: bool
    observation_artifact_id: str
    twin_belief_id: str
    physical_posterior_id: str
    provider_manifest_id: str
    run_manifest_id: str
    evidence_fingerprint: str
    repository_revisions: Mapping[str, str]
    wheel_sha256: Mapping[str, str]
    package_versions: Mapping[str, str]
    baseline_identity: ArrayByteIdentityV1
    candidate_identity: ArrayByteIdentityV1
    selected_identity: ArrayByteIdentityV1
    exact_fallback_identity: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = nonempty_string(self.case_id, name="case_id")
        protocol_id = nonempty_string(self.protocol_id, name="protocol_id")
        reason = nonempty_string(self.reason, name="reason")
        if any(value != value.strip() for value in (case_id, protocol_id, reason)):
            raise ValueError("golden-path identifiers and reason must be canonical")
        decision = self.decision
        if type(decision) is not str or decision not in {"accepted", "rejected"}:
            raise ValueError("unsupported golden-path decision")
        inference_admissible = _literal_boolean(
            self.inference_admissible,
            name="inference_admissible",
        )
        guard_present = _literal_boolean(
            self.regret_guard_present,
            name="regret_guard_present",
        )
        guard_accepted = _literal_boolean(
            self.regret_guard_accepted,
            name="regret_guard_accepted",
        )
        candidate_accepted = _literal_boolean(
            self.candidate_accepted,
            name="candidate_accepted",
        )
        baseline = self.baseline_identity
        candidate = self.candidate_identity
        selected = self.selected_identity
        if any(
            type(value) is not ArrayByteIdentityV1
            for value in (baseline, candidate, selected)
        ):
            raise ValueError("selection identities must be ArrayByteIdentityV1")
        if baseline.array_id == candidate.array_id:
            raise ValueError("golden-path baseline and candidate must be distinct")

        fallback = self.exact_fallback_identity
        if decision == "accepted":
            if not (
                inference_admissible
                and guard_present
                and guard_accepted
                and candidate_accepted
            ):
                raise ValueError("accepted decision requires every admission gate")
            if reason != "candidate-accepted":
                raise ValueError("accepted decision reason changed")
            if selected.array_id != candidate.array_id:
                raise ValueError("accepted decision must select exact candidate bytes")
            if fallback is not None:
                raise ValueError("accepted decision cannot claim an exact fallback")
        else:
            if candidate_accepted:
                raise ValueError("rejected decision cannot accept the candidate")
            if not reason.endswith("exact-baseline-fallback"):
                raise ValueError("rejected decision must declare exact baseline fallback")
            if selected.array_id != baseline.array_id:
                raise ValueError("rejected decision changed the baseline bytes")
            fallback = sha256_digest(
                fallback,
                name="exact_fallback_identity",
            )
            if fallback != baseline.array_id:
                raise ValueError("exact fallback identity must equal the baseline ID")

        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "inference_admissible", inference_admissible)
        object.__setattr__(self, "regret_guard_present", guard_present)
        object.__setattr__(self, "regret_guard_accepted", guard_accepted)
        object.__setattr__(self, "candidate_accepted", candidate_accepted)
        for name in (
            "observation_artifact_id",
            "twin_belief_id",
            "physical_posterior_id",
            "provider_manifest_id",
            "run_manifest_id",
            "evidence_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                sha256_digest(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "repository_revisions",
            _component_mapping(
                self.repository_revisions,
                name="repository_revisions",
                validator=exact_revision,
            ),
        )
        object.__setattr__(
            self,
            "wheel_sha256",
            _component_mapping(
                self.wheel_sha256,
                name="wheel_sha256",
                validator=sha256_digest,
            ),
        )
        object.__setattr__(
            self,
            "package_versions",
            _component_mapping(
                self.package_versions,
                name="package_versions",
                validator=_package_version,
            ),
        )
        object.__setattr__(self, "exact_fallback_identity", fallback)
        object.__setattr__(
            self,
            "metadata",
            frozen_finite_json_mapping(self.metadata, name="metadata"),
        )

    @property
    def artifact_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": GOLDEN_PATH_SELECTION_SCHEMA,
            "schema_version": GOLDEN_PATH_SELECTION_SCHEMA_VERSION,
            "case_id": self.case_id,
            "protocol_id": self.protocol_id,
            "decision": self.decision,
            "reason": self.reason,
            "inference_admissible": self.inference_admissible,
            "regret_guard_present": self.regret_guard_present,
            "regret_guard_accepted": self.regret_guard_accepted,
            "candidate_accepted": self.candidate_accepted,
            "observation_artifact_id": self.observation_artifact_id,
            "twin_belief_id": self.twin_belief_id,
            "physical_posterior_id": self.physical_posterior_id,
            "provider_manifest_id": self.provider_manifest_id,
            "run_manifest_id": self.run_manifest_id,
            "evidence_fingerprint": self.evidence_fingerprint,
            "repository_revisions": plain_json(self.repository_revisions),
            "wheel_sha256": plain_json(self.wheel_sha256),
            "package_versions": plain_json(self.package_versions),
            "baseline_identity": self.baseline_identity.as_dict(),
            "candidate_identity": self.candidate_identity.as_dict(),
            "selected_identity": self.selected_identity.as_dict(),
            "exact_fallback_identity": self.exact_fallback_identity,
            "metadata": plain_json(self.metadata),
        }

    def as_dict(self) -> dict[str, object]:
        return {"artifact_id": self.artifact_id, **self.descriptor()}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> GoldenPathSelectionArtifactV1:
        payload = _mapping(value, name="golden-path selection artifact")
        require_exact_fields(
            payload,
            expected=_SELECTION_FIELDS,
            name="golden-path selection artifact",
        )
        if payload["schema_name"] != GOLDEN_PATH_SELECTION_SCHEMA:
            raise ValueError("unsupported golden-path selection schema")
        if (
            payload["schema_version"]
            != GOLDEN_PATH_SELECTION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported golden-path selection version")
        artifact = cls(
            case_id=nonempty_string(payload["case_id"], name="case_id"),
            protocol_id=nonempty_string(
                payload["protocol_id"],
                name="protocol_id",
            ),
            decision=cast(GoldenPathDecision, payload["decision"]),
            reason=nonempty_string(payload["reason"], name="reason"),
            inference_admissible=_literal_boolean(
                payload["inference_admissible"],
                name="inference_admissible",
            ),
            regret_guard_present=_literal_boolean(
                payload["regret_guard_present"],
                name="regret_guard_present",
            ),
            regret_guard_accepted=_literal_boolean(
                payload["regret_guard_accepted"],
                name="regret_guard_accepted",
            ),
            candidate_accepted=_literal_boolean(
                payload["candidate_accepted"],
                name="candidate_accepted",
            ),
            observation_artifact_id=sha256_digest(
                payload["observation_artifact_id"],
                name="observation_artifact_id",
            ),
            twin_belief_id=sha256_digest(
                payload["twin_belief_id"],
                name="twin_belief_id",
            ),
            physical_posterior_id=sha256_digest(
                payload["physical_posterior_id"],
                name="physical_posterior_id",
            ),
            provider_manifest_id=sha256_digest(
                payload["provider_manifest_id"],
                name="provider_manifest_id",
            ),
            run_manifest_id=sha256_digest(
                payload["run_manifest_id"],
                name="run_manifest_id",
            ),
            evidence_fingerprint=sha256_digest(
                payload["evidence_fingerprint"],
                name="evidence_fingerprint",
            ),
            repository_revisions=cast(
                Mapping[str, str],
                _mapping(
                    payload["repository_revisions"],
                    name="repository_revisions",
                ),
            ),
            wheel_sha256=cast(
                Mapping[str, str],
                _mapping(payload["wheel_sha256"], name="wheel_sha256"),
            ),
            package_versions=cast(
                Mapping[str, str],
                _mapping(
                    payload["package_versions"],
                    name="package_versions",
                ),
            ),
            baseline_identity=ArrayByteIdentityV1.from_mapping(
                _mapping(
                    payload["baseline_identity"],
                    name="baseline_identity",
                )
            ),
            candidate_identity=ArrayByteIdentityV1.from_mapping(
                _mapping(
                    payload["candidate_identity"],
                    name="candidate_identity",
                )
            ),
            selected_identity=ArrayByteIdentityV1.from_mapping(
                _mapping(
                    payload["selected_identity"],
                    name="selected_identity",
                )
            ),
            exact_fallback_identity=cast(
                str | None,
                payload["exact_fallback_identity"],
            ),
            metadata=_mapping(payload["metadata"], name="metadata"),
        )
        expected = sha256_digest(payload["artifact_id"], name="artifact_id")
        if artifact.artifact_id != expected:
            raise ValueError("selection artifact ID does not match its payload")
        return artifact


@dataclass(frozen=True, slots=True)
class GoldenPathEvidenceBundleV1:
    """One accepted and one rejected decision from the same exact run."""

    accepted: GoldenPathSelectionArtifactV1
    rejected: GoldenPathSelectionArtifactV1

    def __post_init__(self) -> None:
        accepted = self.accepted
        rejected = self.rejected
        if type(accepted) is not GoldenPathSelectionArtifactV1 or type(
            rejected
        ) is not GoldenPathSelectionArtifactV1:
            raise ValueError("golden-path bundle entries have invalid types")
        if accepted.decision != "accepted" or rejected.decision != "rejected":
            raise ValueError("golden-path bundle requires accepted and rejected entries")
        shared_fields = (
            "case_id",
            "protocol_id",
            "observation_artifact_id",
            "twin_belief_id",
            "physical_posterior_id",
            "provider_manifest_id",
            "run_manifest_id",
            "evidence_fingerprint",
            "repository_revisions",
            "wheel_sha256",
            "package_versions",
            "baseline_identity",
            "candidate_identity",
        )
        for name in shared_fields:
            if getattr(accepted, name) != getattr(rejected, name):
                raise ValueError(f"bundle entries disagree on {name}")
        if rejected.exact_fallback_identity != rejected.baseline_identity.array_id:
            raise ValueError("rejected entry does not bind exact fallback identity")
        if accepted.artifact_id == rejected.artifact_id:
            raise ValueError("accepted and rejected artifact IDs must differ")

    @property
    def bundle_id(self) -> str:
        return content_id(self.descriptor())

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_name": GOLDEN_PATH_BUNDLE_SCHEMA,
            "schema_version": GOLDEN_PATH_BUNDLE_SCHEMA_VERSION,
            "accepted": self.accepted.as_dict(),
            "rejected": self.rejected.as_dict(),
        }

    def as_dict(self) -> dict[str, object]:
        return {"bundle_id": self.bundle_id, **self.descriptor()}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> GoldenPathEvidenceBundleV1:
        payload = _mapping(value, name="golden-path evidence bundle")
        require_exact_fields(
            payload,
            expected=_BUNDLE_FIELDS,
            name="golden-path evidence bundle",
        )
        if payload["schema_name"] != GOLDEN_PATH_BUNDLE_SCHEMA:
            raise ValueError("unsupported golden-path bundle schema")
        if payload["schema_version"] != GOLDEN_PATH_BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported golden-path bundle version")
        bundle = cls(
            accepted=GoldenPathSelectionArtifactV1.from_mapping(
                _mapping(payload["accepted"], name="accepted artifact")
            ),
            rejected=GoldenPathSelectionArtifactV1.from_mapping(
                _mapping(payload["rejected"], name="rejected artifact")
            ),
        )
        expected = sha256_digest(payload["bundle_id"], name="bundle_id")
        if bundle.bundle_id != expected:
            raise ValueError("golden-path bundle ID does not match its payload")
        return bundle


def build_golden_path_selection_artifact_v1(
    *,
    selection: GaugeAwareSelection,
    baseline: object,
    candidate: object,
    case_id: str,
    protocol_id: str,
    observation_artifact_id: str,
    twin_belief_id: str,
    physical_posterior_id: str,
    provider_manifest_id: str,
    run_manifest_id: str,
    evidence_fingerprint: str,
    repository_revisions: Mapping[str, str],
    wheel_sha256: Mapping[str, str],
    package_versions: Mapping[str, str],
    metadata: Mapping[str, Any] | None = None,
) -> GoldenPathSelectionArtifactV1:
    """Bind one selection to exact arrays and exact cross-repository evidence."""

    if type(selection) is not GaugeAwareSelection:
        raise ValueError("selection must be GaugeAwareSelection")
    baseline_identity = ArrayByteIdentityV1.from_array(baseline)
    candidate_identity = ArrayByteIdentityV1.from_array(candidate)
    selected_identity = ArrayByteIdentityV1.from_array(selection.selected_value)
    return GoldenPathSelectionArtifactV1(
        case_id=case_id,
        protocol_id=protocol_id,
        decision="accepted" if selection.candidate_accepted else "rejected",
        reason=selection.reason,
        inference_admissible=selection.inference_admissible,
        regret_guard_present=selection.regret_guard_present,
        regret_guard_accepted=selection.regret_guard_accepted,
        candidate_accepted=selection.candidate_accepted,
        observation_artifact_id=observation_artifact_id,
        twin_belief_id=twin_belief_id,
        physical_posterior_id=physical_posterior_id,
        provider_manifest_id=provider_manifest_id,
        run_manifest_id=run_manifest_id,
        evidence_fingerprint=evidence_fingerprint,
        repository_revisions=repository_revisions,
        wheel_sha256=wheel_sha256,
        package_versions=package_versions,
        baseline_identity=baseline_identity,
        candidate_identity=candidate_identity,
        selected_identity=selected_identity,
        exact_fallback_identity=(
            None if selection.candidate_accepted else baseline_identity.array_id
        ),
        metadata={} if metadata is None else metadata,
    )


def write_golden_path_evidence_bundle_v1(
    directory: str | Path,
    bundle: GoldenPathEvidenceBundleV1,
    *,
    overwrite: bool = False,
) -> Mapping[str, Path]:
    """Write the accepted, rejected, and bundle JSON records atomically."""

    if type(bundle) is not GoldenPathEvidenceBundleV1:
        raise ValueError("bundle must be GoldenPathEvidenceBundleV1")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "accepted": root / "accepted-selection.json",
        "rejected": root / "rejected-selection.json",
        "bundle": root / "golden-path-bundle.json",
    }
    write_atomic_json(
        bundle.accepted.as_dict(),
        paths["accepted"],
        overwrite=overwrite,
    )
    write_atomic_json(
        bundle.rejected.as_dict(),
        paths["rejected"],
        overwrite=overwrite,
    )
    write_atomic_json(bundle.as_dict(), paths["bundle"], overwrite=overwrite)
    return frozen_finite_json_mapping(
        {name: path.as_posix() for name, path in paths.items()},
        name="golden-path output paths",
    )


def load_golden_path_evidence_bundle_v1(
    directory: str | Path,
) -> GoldenPathEvidenceBundleV1:
    """Load the three records and require exact pair-to-bundle agreement."""

    root = Path(directory)
    accepted = GoldenPathSelectionArtifactV1.from_mapping(
        load_strict_json_object(
            root / "accepted-selection.json",
            label="accepted golden-path selection",
        )
    )
    rejected = GoldenPathSelectionArtifactV1.from_mapping(
        load_strict_json_object(
            root / "rejected-selection.json",
            label="rejected golden-path selection",
        )
    )
    bundle = GoldenPathEvidenceBundleV1.from_mapping(
        load_strict_json_object(
            root / "golden-path-bundle.json",
            label="golden-path evidence bundle",
        )
    )
    if bundle.accepted != accepted or bundle.rejected != rejected:
        raise ValueError("golden-path pair does not match the bundle")
    return bundle


__all__ = [
    "GOLDEN_PATH_BUNDLE_SCHEMA",
    "GOLDEN_PATH_BUNDLE_SCHEMA_VERSION",
    "GOLDEN_PATH_SELECTION_SCHEMA",
    "GOLDEN_PATH_SELECTION_SCHEMA_VERSION",
    "ArrayByteIdentityV1",
    "GoldenPathEvidenceBundleV1",
    "GoldenPathSelectionArtifactV1",
    "build_golden_path_selection_artifact_v1",
    "load_golden_path_evidence_bundle_v1",
    "write_golden_path_evidence_bundle_v1",
]
