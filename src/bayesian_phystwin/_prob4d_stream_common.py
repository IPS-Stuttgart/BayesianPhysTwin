"""Shared contracts and helpers for recursive Prob4D factor streams."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

from ._canonical_contracts import (
    canonical_relative_posix_path,
    frozen_finite_json_mapping,
    literal_lower_hex,
)
from ._portable_contracts import write_atomic_json


PROB4D_OBSERVATION_FACTOR_STREAM_SCHEMA = "prob4d.observation-factor-stream"
PROB4D_OBSERVATION_FACTOR_STREAM_VERSION = 1
PROB4D_OBSERVATION_FACTOR_BUNDLE_SCHEMA = "prob4d.observation-factor-bundle"
PROB4D_OBSERVATION_FACTOR_BUNDLE_VERSION = 4
PROB4D_LEGACY_SOURCE_REPOSITORY = "FlorianPfaff/Prob4D"
PROB4D_CANONICAL_SOURCE_REPOSITORY = "IPS-Stuttgart/Prob4D"
PROB4D_SOURCE_REPOSITORY_ALIASES = frozenset(
    {
        PROB4D_LEGACY_SOURCE_REPOSITORY,
        PROB4D_CANONICAL_SOURCE_REPOSITORY,
    }
)
PROB4D_PROJECT_ID = "github-repository-id:1295794737"

PROB4D_STREAM_OBSERVATION_BINDING_SCHEMA = (
    "bayesian_phystwin.prob4d_stream_observation_binding"
)
PROB4D_STREAM_OBSERVATION_BINDING_VERSION = 1
RECURSIVE_NUISANCE_POLICY_SCHEMA = (
    "bayesian_phystwin.prob4d_recursive_nuisance_policy"
)
RECURSIVE_NUISANCE_POLICY_VERSION = 1
RecursiveNuisanceMode = Literal[
    "persistent_explicit_state",
    "conditionally_independent_increments",
]
RECURSIVE_NUISANCE_MODES: tuple[RecursiveNuisanceMode, ...] = (
    "persistent_explicit_state",
    "conditionally_independent_increments",
)
CLAIM_BEARING_PROB4D_STREAM_STEP_SCHEMA = (
    "bayesian_phystwin.claim_bearing_prob4d_stream_step"
)
CLAIM_BEARING_PROB4D_STREAM_STEP_VERSION = 1
CLAIM_BEARING_PROB4D_STREAM_RUN_SCHEMA = (
    "bayesian_phystwin.claim_bearing_prob4d_stream_run"
)
CLAIM_BEARING_PROB4D_STREAM_RUN_VERSION = 1

_UPDATE_FIELDS = frozenset(
    {
        "update_index",
        "admitted_frame_start",
        "causal_frame_stop",
        "bundle_manifest_path",
        "bundle_manifest_sha256",
        "bundle_payload_sha256",
        "bundle_sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "factor_count",
        "observation_count",
        "persistent_identity_count",
        "observation_identity_sha256",
        "gauge_ids",
        "previous_update_id",
        "update_id",
    }
)
_STREAM_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "metadata",
        "updates",
        "artifact_id",
    }
)
_NUISANCE_POLICY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "policy_id",
        "mode",
        "state_domain_id",
        "nuisance_family_ids",
        "conditional_independence_evidence_id",
        "metadata",
    }
)
_STEP_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "step_id",
        "stream_artifact_id",
        "stream_update_id",
        "observation_binding_id",
        "update_index",
        "admitted_frame_start",
        "causal_frame_stop",
        "prior_belief_id",
        "observation_artifact_id",
        "linearization_artifact_id",
        "claim_update_id",
        "candidate_belief_id",
        "guard_decision_id",
        "selection_id",
        "selected_belief_id",
        "selected_candidate",
        "exact_fallback",
        "provider_manifest_id",
        "calibration_artifact_ids",
        "runtime_revision_source",
        "runtime_revision_independently_verified",
        "covariance_semantics_id",
        "covariance_policy_id",
        "recursive_nuisance_policy_id",
        "previous_step_id",
        "reason",
        "metadata",
    }
)
_RUN_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "run_id",
        "stream_artifact_id",
        "initial_belief_id",
        "provider_manifest_id",
        "calibration_artifact_ids",
        "runtime_revision_source",
        "runtime_revision_independently_verified",
        "covariance_policy_id",
        "recursive_nuisance_policy_id",
        "steps",
        "metadata",
    }
)


class ArtifactBelief(Protocol):
    @property
    def artifact_id(self) -> str: ...


BeliefT = TypeVar("BeliefT", bound=ArtifactBelief)


def _sha256(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={64})


def _revision(value: object, *, name: str) -> str:
    return literal_lower_hex(value, name=name, lengths={40, 64})


def _nonempty_literal_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty literal string")
    return value


def _repository(value: object, *, name: str) -> str:
    repository = _nonempty_literal_string(value, name=name)
    if repository not in PROB4D_SOURCE_REPOSITORY_ALIASES:
        raise ValueError(f"{name} does not identify Prob4D")
    return repository


def _string_tuple(
    value: object,
    *,
    name: str,
    require_unique: bool = True,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(
        _nonempty_literal_string(item, name=f"{name} item") for item in value
    )
    if not result:
        raise ValueError(f"{name} must not be empty")
    if require_unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _calibration_ids(value: object) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("calibration_artifact_ids must be a nonempty mapping")
    result: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        name = _nonempty_literal_string(
            raw_name,
            name="calibration artifact name",
        )
        result[name] = _sha256(
            raw_digest,
            name=f"calibration artifact {name}",
        )
    return cast(
        Mapping[str, str],
        frozen_finite_json_mapping(
            dict(sorted(result.items())),
            name="calibration_artifact_ids",
        ),
    )


def _optional_calibration_ids(value: object) -> Mapping[str, str]:
    if value is None or (isinstance(value, Mapping) and not value):
        return cast(
            Mapping[str, str],
            frozen_finite_json_mapping({}, name="calibration_artifact_ids"),
        )
    return _calibration_ids(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinary_confined_file(
    root: Path,
    relative_path: str,
    *,
    name: str,
) -> Path:
    safe = canonical_relative_posix_path(relative_path, name=name)
    root_resolved = root.resolve()
    candidate = root_resolved
    for part in safe.split("/"):
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"{name} must not traverse a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as error:
        raise ValueError(f"{name} escapes its containing directory") from error
    if not resolved.is_file():
        raise ValueError(f"{name} must identify an ordinary file")
    return resolved


def _write_atomic_json(
    value: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    output = Path(path)
    if output.is_symlink():
        raise ValueError(f"refusing to replace symlink {output}")
    write_atomic_json(value, output, overwrite=overwrite)
    return output
