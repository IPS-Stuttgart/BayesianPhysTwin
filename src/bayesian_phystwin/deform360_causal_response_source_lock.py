"""Fresh-object source lock for the prospective V12 experiment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .deform360_causal_response_preflight import (
    CausalResponseSourcePreflight,
    CausalResponseSourcePreflightConfig,
    validate_causal_response_source_preflight,
)

SOURCE_LOCK_CONTRACT = "deform360-causal-response-source-lock-v12"
OBJECT_HASH_NAMESPACE = b"deform360-fresh-object-exclusion-v1\0"
REQUIRED_EXCLUSION_SCOPES = frozenset(
    {
        "prob4d",
        "molmomotion_field",
        "held_v8_all_attempts",
        "bayesian_phystwin_through_v11",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_sha256", None)
    return hashlib.sha256(
        b"deform360-causal-response-source-lock-v12\0"
        + json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def deform360_object_hash(object_id: str) -> str:
    """Hash an object identity in the shared fresh-object namespace."""

    _require(bool(str(object_id).strip()), "object ID is empty")
    return hashlib.sha256(
        OBJECT_HASH_NAMESPACE + str(object_id).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class CausalResponseSourceCase:
    """One metadata-selected source case before observations or outcomes."""

    case_id: str
    case_hash: str
    object_hash: str
    metadata_sha256: str
    source_preflight_sha256: str
    fold: int

    def __post_init__(self) -> None:
        _require(bool(self.case_id.strip()), "case ID is empty")
        _require(
            all(
                _valid_digest(value)
                for value in (
                    self.case_hash,
                    self.object_hash,
                    self.metadata_sha256,
                    self.source_preflight_sha256,
                )
            ),
            "source case digest is invalid",
        )
        _require(0 <= self.fold < 3, "source fold is invalid")


@dataclass(frozen=True)
class CausalResponseSourceLock:
    """Immutable twelve-object source panel and exclusion provenance."""

    protocol_id: str
    repository_revision: str
    method_config_sha256: str
    exclusion_manifest_sha256: dict[str, str]
    excluded_object_hashes: tuple[str, ...]
    cases: tuple[CausalResponseSourceCase, ...]
    selection_metadata_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require(bool(self.protocol_id.strip()), "protocol ID is empty")
        _require(
            len(self.repository_revision) == 40
            and all(
                character in "0123456789abcdef"
                for character in self.repository_revision
            ),
            "repository revision is invalid",
        )
        _require(
            _valid_digest(self.method_config_sha256)
            and _valid_digest(self.selection_metadata_sha256)
            and _valid_digest(self.artifact_sha256),
            "source lock digest is invalid",
        )
        manifests = dict(sorted(self.exclusion_manifest_sha256.items()))
        _require(
            set(manifests) == REQUIRED_EXCLUSION_SCOPES
            and all(_valid_digest(value) for value in manifests.values()),
            "source lock lacks one or more required exclusion scopes",
        )
        excluded = tuple(sorted(set(self.excluded_object_hashes)))
        _require(
            len(excluded) == len(self.excluded_object_hashes)
            and all(_valid_digest(value) for value in excluded),
            "excluded object hashes are invalid or duplicated",
        )
        _require(len(self.cases) == 12, "source panel must contain 12 cases")
        _require(
            len({case.case_id for case in self.cases}) == 12
            and len({case.case_hash for case in self.cases}) == 12
            and len({case.object_hash for case in self.cases}) == 12,
            "source cases or physical objects are duplicated",
        )
        _require(
            not set(case.object_hash for case in self.cases).intersection(excluded),
            "source panel overlaps an excluded physical object",
        )
        _require(
            Counter(case.fold for case in self.cases) == Counter({0: 4, 1: 4, 2: 4}),
            "source cross-fit folds must contain four objects each",
        )
        object.__setattr__(self, "exclusion_manifest_sha256", manifests)
        object.__setattr__(self, "excluded_object_hashes", excluded)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "Deform360CausalResponseSourceLock",
            "contract": SOURCE_LOCK_CONTRACT,
            "protocol_id": self.protocol_id,
            "repository_revision": self.repository_revision,
            "method_config_sha256": self.method_config_sha256,
            "exclusion_manifest_sha256": dict(self.exclusion_manifest_sha256),
            "excluded_object_count": len(self.excluded_object_hashes),
            "excluded_object_hashes": list(self.excluded_object_hashes),
            "selection_metadata_sha256": self.selection_metadata_sha256,
            "cases": [asdict(case) for case in self.cases],
            "source_gate": {
                "physical_object_count": 12,
                "episode_count_per_object": 1,
                "cross_fit_fold_count": 3,
                "objects_per_fold": 4,
            },
            "information_boundary": {
                "selection_inputs": "released metadata only",
                "object_rgb_depth_mask_tactile_or_action_read": False,
                "physical_prediction_read": False,
                "identity_or_metric_outcome_read": False,
                "target_cohort_read": False,
                "held_v8_object_ids_or_outcomes_read": False,
                "held_v8_hash_only_exclusion_used": True,
                "accepted_source_preflight_required": True,
            },
            "artifact_sha256": self.artifact_sha256,
        }


def build_causal_response_source_lock(
    cases: Iterable[CausalResponseSourceCase],
    *,
    protocol_id: str,
    repository_revision: str,
    method_config_sha256: str,
    exclusion_manifest_sha256: Mapping[str, str],
    excluded_object_hashes: Iterable[str],
    selection_metadata_sha256: str,
    source_preflights: Iterable[CausalResponseSourcePreflight],
) -> CausalResponseSourceLock:
    """Build a source lock only after the complete exclusion union exists."""

    ordered_cases = tuple(
        sorted(
            cases,
            key=lambda case: (case.fold, case.object_hash, case.case_hash),
        )
    )
    _require(
        len({case.case_hash for case in ordered_cases}) == len(ordered_cases)
        and len({case.object_hash for case in ordered_cases}) == len(ordered_cases),
        "source cases or physical objects are duplicated",
    )
    preflights = tuple(source_preflights)
    for preflight in preflights:
        validate_causal_response_source_preflight(preflight)
    _require(
        len(preflights) == len(ordered_cases)
        and len({preflight.case_hash for preflight in preflights}) == len(preflights),
        "source preflight set does not match the source panel",
    )
    preflight_by_case = {preflight.case_hash: preflight for preflight in preflights}
    for case in ordered_cases:
        preflight = preflight_by_case.get(case.case_hash)
        _require(
            preflight is not None
            and preflight.admitted
            and preflight.config == CausalResponseSourcePreflightConfig()
            and preflight.object_hash == case.object_hash
            and preflight.artifact_sha256 == case.source_preflight_sha256,
            "source case lacks a matching accepted V12 preflight",
        )
    provisional = CausalResponseSourceLock(
        protocol_id=protocol_id,
        repository_revision=repository_revision,
        method_config_sha256=method_config_sha256,
        exclusion_manifest_sha256=dict(exclusion_manifest_sha256),
        excluded_object_hashes=tuple(excluded_object_hashes),
        cases=ordered_cases,
        selection_metadata_sha256=selection_metadata_sha256,
        artifact_sha256="0" * 64,
    )
    digest = _canonical_sha256(provisional.descriptor())
    result = CausalResponseSourceLock(
        **{**provisional.__dict__, "artifact_sha256": digest}
    )
    _require(
        _canonical_sha256(result.descriptor()) == result.artifact_sha256,
        "source lock changed after construction",
    )
    return result


def write_causal_response_source_lock(
    path: str | Path,
    lock: CausalResponseSourceLock,
) -> None:
    """Write a source lock once without an outcome-bearing side channel."""

    output = Path(path)
    _require(not output.exists(), "source lock output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            lock.descriptor(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    validate_causal_response_source_lock(output)


def validate_causal_response_source_lock(
    path: str | Path,
) -> CausalResponseSourceLock:
    """Validate source cardinality, exclusions, folds, and artifact checksum."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(
        payload.get("artifact_kind") == "Deform360CausalResponseSourceLock"
        and payload.get("contract") == SOURCE_LOCK_CONTRACT,
        "source lock kind or contract changed",
    )
    lock = CausalResponseSourceLock(
        protocol_id=payload["protocol_id"],
        repository_revision=payload["repository_revision"],
        method_config_sha256=payload["method_config_sha256"],
        exclusion_manifest_sha256=payload["exclusion_manifest_sha256"],
        excluded_object_hashes=tuple(payload["excluded_object_hashes"]),
        cases=tuple(CausalResponseSourceCase(**record) for record in payload["cases"]),
        selection_metadata_sha256=payload["selection_metadata_sha256"],
        artifact_sha256=payload["artifact_sha256"],
    )
    _require(
        lock.descriptor() == payload
        and _canonical_sha256(payload) == lock.artifact_sha256,
        "source lock checksum or descriptor changed",
    )
    return lock


__all__ = [
    "OBJECT_HASH_NAMESPACE",
    "REQUIRED_EXCLUSION_SCOPES",
    "SOURCE_LOCK_CONTRACT",
    "CausalResponseSourceCase",
    "CausalResponseSourceLock",
    "build_causal_response_source_lock",
    "deform360_object_hash",
    "validate_causal_response_source_lock",
    "write_causal_response_source_lock",
]
