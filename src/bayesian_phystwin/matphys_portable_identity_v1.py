"""Path-independent identity certificate for official MatPhys producer bundles.

The frozen producer-v1 contract intentionally records exact local source paths.
This additive certificate strips those host-specific locations from the
scientific identity while retaining them in a separate, self-addressed source
verification receipt.  Existing producer-v1 artifacts remain unchanged.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from ._artifact_custody import (
    file_sha256,
    new_staging_directory,
    ordinary_directory,
    ordinary_file,
    publish_staging_directory,
    regular_file_roster,
    validate_checksum_manifest,
    write_checksum_manifest,
)
from ._canonical_contracts import plain_json
from ._portable_contracts import (
    content_id,
    exact_revision,
    load_strict_json_object,
    nonempty_string,
    repository_name,
    require_exact_fields,
    sha256_digest,
    source_artifact_mapping,
    write_atomic_json,
)
from .matphys_backend_v1 import (
    MATPHYS_BACKEND_CLAIM_BOUNDARY,
    MATPHYS_BACKEND_KIND,
    MATPHYS_BACKEND_PROPOSAL_SCHEMA,
    MATPHYS_BACKEND_VERSION,
    MATPHYS_PARAMETERIZATION,
    MATPHYS_ROLLOUT_BACKEND,
    validate_matphys_backend_proposal,
)
from .matphys_official_producer_v1 import (
    ARTIFACT_FILENAME as SOURCE_ARTIFACT_FILENAME,
)
from .matphys_official_producer_v1 import (
    CAUSAL_PROPOSAL_FILENAME as SOURCE_CAUSAL_PROPOSAL_FILENAME,
)
from .matphys_official_producer_v1 import (
    MATPHYS_CAUSAL_PREFIX_MODE,
    MATPHYS_OFFICIAL_BACKEND_KIND,
    MATPHYS_OFFICIAL_CLAIM_BOUNDARY,
    MATPHYS_OFFICIAL_PARAMETERIZATION,
    MATPHYS_OFFICIAL_PIPELINE_COMPONENTS,
    MATPHYS_OFFICIAL_PRODUCER_PROTOCOL,
    MATPHYS_OFFICIAL_PRODUCER_SCHEMA,
    MATPHYS_OFFICIAL_PRODUCER_VERSION,
    MATPHYS_OFFICIAL_ROLLOUT_BACKEND,
    MATPHYS_PUBLISHED_PARITY_MODE,
    validate_matphys_official_producer_artifact,
)

PORTABLE_IDENTITY_SCHEMA: Final = "bayesian-phystwin.matphys-portable-identity"
PORTABLE_IDENTITY_VERSION: Final = 1
PORTABLE_PROPOSAL_SCHEMA: Final = "bayesian-phystwin.matphys-backend-proposal-portable"
PORTABLE_PROPOSAL_VERSION: Final = 1
SOURCE_VERIFICATION_SCHEMA: Final = "bayesian-phystwin.matphys-source-verification"
SOURCE_VERIFICATION_VERSION: Final = 1

PORTABLE_IDENTITY_FILENAME: Final = "matphys-portable-identity.json"
SOURCE_VERIFICATION_FILENAME: Final = "source-verification.json"
CHECKSUMS_FILENAME: Final = "SHA256SUMS"

_INPUT_ROLES: Final = (
    "candidate_parameters",
    "checkpoint",
    "identity_parameters",
    "replay_input",
    "spring_field",
)
_PORTABLE_FILE_FIELDS: Final = frozenset({"role", "sha256", "byte_count"})
_PORTABLE_SPRING_FIELDS: Final = frozenset({"role", "sha256", "byte_count", "count"})
_LOCAL_FILE_FIELDS: Final = frozenset({"role", "path", "sha256", "byte_count"})
_LOCAL_SPRING_FIELDS: Final = frozenset(
    {"role", "path", "sha256", "byte_count", "count"}
)
_SOURCE_BUNDLE_FIELDS: Final = frozenset({"path", "artifact_path", "artifact_sha256"})
_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "portable_artifact_id",
        "source_bundle",
        "source_locations",
        "host_status_id",
    }
)
_PORTABLE_PROPOSAL_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "source_schema",
        "source_schema_version",
        "backend_kind",
        "parameterization",
        "rollout_backend",
        "source_repository",
        "source_revision",
        "simulator_repository",
        "simulator_revision",
        "target_object_id",
        "training_object_ids",
        "target_object_excluded",
        "target_evidence_end_frame_exclusive",
        "target_future_observations_used",
        "known_future_robot_action_used",
        "proposal_strength",
        "checkpoint",
        "spring_field",
        "source_artifacts",
        "claim_boundary",
        "portable_proposal_id",
    }
)
_CERTIFICATE_FIELDS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "source_schema",
        "source_schema_version",
        "source_producer_protocol",
        "mode",
        "backend_kind",
        "parameterization",
        "rollout_backend",
        "coordinate_frame",
        "position_unit",
        "source_repository",
        "source_revision",
        "simulator_repository",
        "simulator_revision",
        "case_id",
        "target_object_id",
        "checkpoint_training_object_ids",
        "target_fit_frame_range_half_open",
        "future_frame_start",
        "proposal_strength",
        "pipeline_components",
        "pipeline_component_artifacts",
        "inputs",
        "replay_summary",
        "source_artifacts",
        "outputs",
        "information_boundary",
        "claim_boundary",
        "portable_artifact_id",
    }
)
_OUTPUT_FIELDS: Final = frozenset(
    {"candidate_archive", "identity_replay_archive", "causal_proposal"}
)
_REPLAY_SUMMARY_FIELDS: Final = frozenset(
    {
        "frame_count",
        "state_count",
        "query_count",
        "dtype",
        "frame_indices",
        "frame_indices_sha256",
        "material_query_indices_sha256",
    }
)
_BOUNDARY_FIELDS: Final = frozenset(
    {
        "target_prefix_used_for_parameter_fit",
        "target_object_used_for_checkpoint_training",
        "target_future_observations_used",
        "future_outcomes_opened",
        "known_future_robot_action_used",
        "causal_backend_eligible",
        "published_benchmark_control_only",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a JSON array")
    return cast(Sequence[object], value)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not result == result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{name} must be a finite number")
    return result


def _frame_range(value: object, *, name: str) -> tuple[int, int]:
    raw = _sequence(value, name=name)
    if len(raw) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw
    ):
        raise ValueError(f"{name} must contain exactly two integer indices")
    start, stop = cast(int, raw[0]), cast(int, raw[1])
    if not 0 <= start < stop:
        raise ValueError(f"{name} must be a nonempty nonnegative half-open range")
    return start, stop


def _portable_file_record(
    role: str,
    value: object,
    *,
    spring: bool = False,
) -> dict[str, object]:
    record = _mapping(value, name=role)
    expected = (
        frozenset({"path", "sha256", "byte_count", "count"})
        if spring
        else frozenset({"path", "sha256", "byte_count"})
    )
    require_exact_fields(record, expected=expected, name=role)
    portable: dict[str, object] = {
        "role": role,
        "sha256": sha256_digest(record.get("sha256"), name=f"{role}.sha256"),
        "byte_count": _positive_integer(
            record.get("byte_count"), name=f"{role}.byte_count"
        ),
    }
    if spring:
        portable["count"] = _positive_integer(record.get("count"), name=f"{role}.count")
    return portable


def _normalize_portable_file(
    value: object,
    *,
    role: str,
    spring: bool = False,
) -> dict[str, object]:
    record = _mapping(value, name=role)
    expected = _PORTABLE_SPRING_FIELDS if spring else _PORTABLE_FILE_FIELDS
    require_exact_fields(record, expected=expected, name=role)
    _require(record.get("role") == role, f"{role} role changed")
    normalized: dict[str, object] = {
        "role": role,
        "sha256": sha256_digest(record.get("sha256"), name=f"{role}.sha256"),
        "byte_count": _positive_integer(
            record.get("byte_count"), name=f"{role}.byte_count"
        ),
    }
    if spring:
        normalized["count"] = _positive_integer(
            record.get("count"), name=f"{role}.count"
        )
    return normalized


def _local_file_record(
    role: str,
    value: object,
    *,
    spring: bool = False,
) -> dict[str, object]:
    source = _mapping(value, name=role)
    portable = _portable_file_record(role, source, spring=spring)
    result = {
        **portable,
        "path": nonempty_string(source.get("path"), name=f"{role}.path"),
    }
    return result


def _normalize_local_file(
    value: object,
    *,
    role: str,
    spring: bool = False,
    verify: bool = False,
) -> dict[str, object]:
    record = _mapping(value, name=role)
    expected = _LOCAL_SPRING_FIELDS if spring else _LOCAL_FILE_FIELDS
    require_exact_fields(record, expected=expected, name=role)
    portable_record = dict(record)
    portable_record.pop("path")
    portable = _normalize_portable_file(portable_record, role=role, spring=spring)
    path = nonempty_string(record.get("path"), name=f"{role}.path")
    if verify:
        source = ordinary_file(path, name=role)
        _require(
            source.stat().st_size == portable["byte_count"],
            f"{role} byte count changed",
        )
        _require(
            file_sha256(source) == portable["sha256"],
            f"{role} SHA-256 changed",
        )
        path = str(source)
    return {**portable, "path": path}


def _portable_output_record(role: str, value: object) -> dict[str, object]:
    record = _mapping(value, name=role)
    require_exact_fields(
        record,
        expected=frozenset({"path", "sha256", "byte_count"}),
        name=role,
    )
    return {
        "role": role,
        "sha256": sha256_digest(record.get("sha256"), name=f"{role}.sha256"),
        "byte_count": _positive_integer(
            record.get("byte_count"), name=f"{role}.byte_count"
        ),
    }


def _normalize_object_ids(value: object, *, name: str) -> tuple[str, ...]:
    raw = _sequence(value, name=name)
    values = tuple(nonempty_string(item, name=f"{name} entry") for item in raw)
    _require(values == tuple(sorted(set(values))), f"{name} must be sorted and unique")
    return values


def _portable_proposal(
    source_root: Path,
    official: Mapping[str, Any],
    *,
    verify_sources: bool,
) -> dict[str, object] | None:
    outputs = _mapping(official.get("outputs"), name="official outputs")
    if outputs.get("causal_proposal") is None:
        return None
    proposal = validate_matphys_backend_proposal(
        load_strict_json_object(
            source_root / SOURCE_CAUSAL_PROPOSAL_FILENAME,
            label="source MatPhys proposal",
        ),
        verify_files=verify_sources,
    )
    checkpoint = _portable_file_record("checkpoint", official.get("checkpoint"))
    spring = _portable_file_record(
        "spring_field", official.get("spring_field"), spring=True
    )
    proposal_checkpoint = _mapping(
        proposal.get("checkpoint"), name="proposal checkpoint"
    )
    proposal_spring = _mapping(proposal.get("spring_field"), name="proposal spring")
    _require(
        proposal_checkpoint.get("sha256") == checkpoint["sha256"],
        "portable proposal checkpoint identity disagrees with producer artifact",
    )
    _require(
        proposal_spring.get("sha256") == spring["sha256"]
        and proposal_spring.get("count") == spring["count"],
        "portable proposal spring identity disagrees with producer artifact",
    )
    identity: dict[str, object] = {
        "schema": PORTABLE_PROPOSAL_SCHEMA,
        "schema_version": PORTABLE_PROPOSAL_VERSION,
        "source_schema": MATPHYS_BACKEND_PROPOSAL_SCHEMA,
        "source_schema_version": MATPHYS_BACKEND_VERSION,
        "backend_kind": proposal["backend_kind"],
        "parameterization": proposal["parameterization"],
        "rollout_backend": proposal["rollout_backend"],
        "source_repository": proposal["source_repository"],
        "source_revision": proposal["source_revision"],
        "simulator_repository": proposal["simulator_repository"],
        "simulator_revision": proposal["simulator_revision"],
        "target_object_id": proposal["target_object_id"],
        "training_object_ids": proposal["training_object_ids"],
        "target_object_excluded": proposal["target_object_excluded"],
        "target_evidence_end_frame_exclusive": proposal[
            "target_evidence_end_frame_exclusive"
        ],
        "target_future_observations_used": proposal["target_future_observations_used"],
        "known_future_robot_action_used": proposal["known_future_robot_action_used"],
        "proposal_strength": proposal["proposal_strength"],
        "checkpoint": checkpoint,
        "spring_field": spring,
        "source_artifacts": proposal["source_artifacts"],
        "claim_boundary": proposal["claim_boundary"],
    }
    return {**identity, "portable_proposal_id": content_id(identity)}


def validate_portable_matphys_proposal(value: object) -> dict[str, object]:
    """Validate one path-independent MatPhys proposal identity."""

    proposal = _mapping(value, name="portable MatPhys proposal")
    require_exact_fields(
        proposal,
        expected=_PORTABLE_PROPOSAL_FIELDS,
        name="portable MatPhys proposal",
    )
    _require(
        proposal.get("schema") == PORTABLE_PROPOSAL_SCHEMA
        and proposal.get("schema_version") == PORTABLE_PROPOSAL_VERSION,
        "portable MatPhys proposal schema changed",
    )
    _require(
        proposal.get("source_schema") == MATPHYS_BACKEND_PROPOSAL_SCHEMA
        and proposal.get("source_schema_version") == MATPHYS_BACKEND_VERSION,
        "portable MatPhys proposal source contract changed",
    )
    _require(
        proposal.get("backend_kind") == MATPHYS_BACKEND_KIND
        and proposal.get("parameterization") == MATPHYS_PARAMETERIZATION
        and proposal.get("rollout_backend") == MATPHYS_ROLLOUT_BACKEND,
        "portable MatPhys proposal semantics changed",
    )
    training = _normalize_object_ids(
        proposal.get("training_object_ids"), name="training_object_ids"
    )
    target = nonempty_string(proposal.get("target_object_id"), name="target_object_id")
    _require(target not in training, "portable proposal training includes target")
    _require(
        proposal.get("target_object_excluded") is True
        and proposal.get("target_future_observations_used") is False
        and proposal.get("known_future_robot_action_used") is True,
        "portable MatPhys proposal information boundary changed",
    )
    artifacts = source_artifact_mapping(
        _mapping(proposal.get("source_artifacts"), name="source_artifacts"),
        name="source_artifacts",
    )
    identity: dict[str, object] = {
        "schema": PORTABLE_PROPOSAL_SCHEMA,
        "schema_version": PORTABLE_PROPOSAL_VERSION,
        "source_schema": MATPHYS_BACKEND_PROPOSAL_SCHEMA,
        "source_schema_version": MATPHYS_BACKEND_VERSION,
        "backend_kind": MATPHYS_BACKEND_KIND,
        "parameterization": MATPHYS_PARAMETERIZATION,
        "rollout_backend": MATPHYS_ROLLOUT_BACKEND,
        "source_repository": repository_name(
            proposal.get("source_repository"), name="source_repository"
        ),
        "source_revision": exact_revision(
            proposal.get("source_revision"), name="source_revision"
        ),
        "simulator_repository": repository_name(
            proposal.get("simulator_repository"), name="simulator_repository"
        ),
        "simulator_revision": exact_revision(
            proposal.get("simulator_revision"), name="simulator_revision"
        ),
        "target_object_id": target,
        "training_object_ids": list(training),
        "target_object_excluded": True,
        "target_evidence_end_frame_exclusive": _nonnegative_integer(
            proposal.get("target_evidence_end_frame_exclusive"),
            name="target_evidence_end_frame_exclusive",
        ),
        "target_future_observations_used": False,
        "known_future_robot_action_used": True,
        "proposal_strength": _finite_number(
            proposal.get("proposal_strength"), name="proposal_strength"
        ),
        "checkpoint": _normalize_portable_file(
            proposal.get("checkpoint"), role="checkpoint"
        ),
        "spring_field": _normalize_portable_file(
            proposal.get("spring_field"), role="spring_field", spring=True
        ),
        "source_artifacts": dict(artifacts),
        "claim_boundary": nonempty_string(
            proposal.get("claim_boundary"), name="claim_boundary"
        ),
    }
    _require(
        identity["claim_boundary"] == MATPHYS_BACKEND_CLAIM_BOUNDARY,
        "portable MatPhys proposal claim boundary changed",
    )
    normalized = {**identity, "portable_proposal_id": content_id(identity)}
    _require(
        proposal.get("portable_proposal_id") == normalized["portable_proposal_id"],
        "portable MatPhys proposal identity changed",
    )
    return cast(dict[str, object], plain_json(normalized))


def _normalize_replay_summary(value: object) -> dict[str, object]:
    summary = _mapping(value, name="replay_summary")
    require_exact_fields(
        summary, expected=_REPLAY_SUMMARY_FIELDS, name="replay_summary"
    )
    frames_raw = _sequence(summary.get("frame_indices"), name="frame_indices")
    frames = [
        int(item)
        for item in frames_raw
        if not isinstance(item, bool) and isinstance(item, int)
    ]
    _require(
        len(frames) == len(frames_raw)
        and len(frames)
        == _positive_integer(summary.get("frame_count"), name="frame_count"),
        "replay frame indices changed",
    )
    _require(
        bool(frames)
        and frames[0] >= 0
        and all(right > left for left, right in zip(frames, frames[1:], strict=False)),
        "replay frame indices must be strictly increasing",
    )
    return {
        "frame_count": len(frames),
        "state_count": _positive_integer(
            summary.get("state_count"), name="state_count"
        ),
        "query_count": _positive_integer(
            summary.get("query_count"), name="query_count"
        ),
        "dtype": nonempty_string(summary.get("dtype"), name="dtype"),
        "frame_indices": frames,
        "frame_indices_sha256": sha256_digest(
            summary.get("frame_indices_sha256"), name="frame_indices_sha256"
        ),
        "material_query_indices_sha256": sha256_digest(
            summary.get("material_query_indices_sha256"),
            name="material_query_indices_sha256",
        ),
    }


def _expected_boundary(*, mode: str, target_in_training: bool) -> dict[str, bool]:
    causal = mode == MATPHYS_CAUSAL_PREFIX_MODE
    return {
        "target_prefix_used_for_parameter_fit": True,
        "target_object_used_for_checkpoint_training": target_in_training,
        "target_future_observations_used": False,
        "future_outcomes_opened": False,
        "known_future_robot_action_used": True,
        "causal_backend_eligible": causal,
        "published_benchmark_control_only": not causal,
    }


def _derive_certificate(
    source_root: Path,
    *,
    verify_sources: bool,
) -> tuple[dict[str, object], dict[str, Any]]:
    official = validate_matphys_official_producer_artifact(
        source_root, verify_sources=verify_sources
    )
    inputs = {
        "checkpoint": _portable_file_record("checkpoint", official["checkpoint"]),
        "spring_field": _portable_file_record(
            "spring_field", official["spring_field"], spring=True
        ),
        "candidate_parameters": _portable_file_record(
            "candidate_parameters", official["candidate_parameters"]
        ),
        "identity_parameters": _portable_file_record(
            "identity_parameters", official["identity_parameters"]
        ),
        "replay_input": _portable_file_record("replay_input", official["replay_input"]),
    }
    outputs = _mapping(official["outputs"], name="official outputs")
    identity: dict[str, object] = {
        "schema": PORTABLE_IDENTITY_SCHEMA,
        "schema_version": PORTABLE_IDENTITY_VERSION,
        "source_schema": MATPHYS_OFFICIAL_PRODUCER_SCHEMA,
        "source_schema_version": MATPHYS_OFFICIAL_PRODUCER_VERSION,
        "source_producer_protocol": MATPHYS_OFFICIAL_PRODUCER_PROTOCOL,
        "mode": official["mode"],
        "backend_kind": official["backend_kind"],
        "parameterization": official["parameterization"],
        "rollout_backend": official["rollout_backend"],
        "coordinate_frame": official["coordinate_frame"],
        "position_unit": official["position_unit"],
        "source_repository": official["source_repository"],
        "source_revision": official["source_revision"],
        "simulator_repository": official["simulator_repository"],
        "simulator_revision": official["simulator_revision"],
        "case_id": official["case_id"],
        "target_object_id": official["target_object_id"],
        "checkpoint_training_object_ids": official["checkpoint_training_object_ids"],
        "target_fit_frame_range_half_open": official[
            "target_fit_frame_range_half_open"
        ],
        "future_frame_start": official["future_frame_start"],
        "proposal_strength": official["proposal_strength"],
        "pipeline_components": official["pipeline_components"],
        "pipeline_component_artifacts": official["pipeline_component_artifacts"],
        "inputs": inputs,
        "replay_summary": official["replay_summary"],
        "source_artifacts": official["source_artifacts"],
        "outputs": {
            "candidate_archive": _portable_output_record(
                "candidate_archive", outputs["candidate_archive"]
            ),
            "identity_replay_archive": _portable_output_record(
                "identity_replay_archive", outputs["identity_replay_archive"]
            ),
            "causal_proposal": _portable_proposal(
                source_root,
                official,
                verify_sources=verify_sources,
            ),
        },
        "information_boundary": official["information_boundary"],
        "claim_boundary": official["claim_boundary"],
    }
    return (
        {**identity, "portable_artifact_id": content_id(identity)},
        official,
    )


def validate_matphys_portable_certificate(value: object) -> dict[str, object]:
    """Validate a path-independent official MatPhys identity certificate."""

    certificate = _mapping(value, name="MatPhys portable identity")
    require_exact_fields(
        certificate,
        expected=_CERTIFICATE_FIELDS,
        name="MatPhys portable identity",
    )
    _require(
        certificate.get("schema") == PORTABLE_IDENTITY_SCHEMA
        and certificate.get("schema_version") == PORTABLE_IDENTITY_VERSION,
        "MatPhys portable identity schema changed",
    )
    _require(
        certificate.get("source_schema") == MATPHYS_OFFICIAL_PRODUCER_SCHEMA
        and certificate.get("source_schema_version")
        == MATPHYS_OFFICIAL_PRODUCER_VERSION
        and certificate.get("source_producer_protocol")
        == MATPHYS_OFFICIAL_PRODUCER_PROTOCOL,
        "MatPhys portable source contract changed",
    )
    mode = nonempty_string(certificate.get("mode"), name="mode")
    _require(
        mode in {MATPHYS_CAUSAL_PREFIX_MODE, MATPHYS_PUBLISHED_PARITY_MODE},
        "unknown MatPhys portable mode",
    )
    _require(
        certificate.get("backend_kind") == MATPHYS_OFFICIAL_BACKEND_KIND
        and certificate.get("parameterization") == MATPHYS_OFFICIAL_PARAMETERIZATION
        and certificate.get("rollout_backend") == MATPHYS_OFFICIAL_ROLLOUT_BACKEND,
        "MatPhys portable backend semantics changed",
    )
    target = nonempty_string(
        certificate.get("target_object_id"), name="target_object_id"
    )
    training = _normalize_object_ids(
        certificate.get("checkpoint_training_object_ids"),
        name="checkpoint_training_object_ids",
    )
    if mode == MATPHYS_CAUSAL_PREFIX_MODE:
        _require(target not in training, "causal portable identity includes target")
    fit_start, fit_stop = _frame_range(
        certificate.get("target_fit_frame_range_half_open"),
        name="target_fit_frame_range_half_open",
    )
    future_start = _positive_integer(
        certificate.get("future_frame_start"), name="future_frame_start"
    )
    _require(fit_stop <= future_start, "portable fitting crosses future boundary")
    if mode == MATPHYS_CAUSAL_PREFIX_MODE:
        _require(fit_stop < future_start, "causal portable identity has no gate frame")

    components = tuple(
        nonempty_string(item, name="pipeline component")
        for item in _sequence(
            certificate.get("pipeline_components"), name="pipeline_components"
        )
    )
    _require(
        components == MATPHYS_OFFICIAL_PIPELINE_COMPONENTS,
        "MatPhys portable pipeline component roster changed",
    )
    component_artifacts = _mapping(
        certificate.get("pipeline_component_artifacts"),
        name="pipeline_component_artifacts",
    )
    _require(
        set(component_artifacts) == set(MATPHYS_OFFICIAL_PIPELINE_COMPONENTS),
        "MatPhys portable component artifact roster changed",
    )
    normalized_components = {
        name: sha256_digest(component_artifacts.get(name), name=f"component {name}")
        for name in MATPHYS_OFFICIAL_PIPELINE_COMPONENTS
    }

    inputs_raw = _mapping(certificate.get("inputs"), name="inputs")
    _require(set(inputs_raw) == set(_INPUT_ROLES), "portable input roster changed")
    inputs = {
        role: _normalize_portable_file(
            inputs_raw.get(role), role=role, spring=role == "spring_field"
        )
        for role in _INPUT_ROLES
    }
    outputs_raw = _mapping(certificate.get("outputs"), name="outputs")
    require_exact_fields(outputs_raw, expected=_OUTPUT_FIELDS, name="outputs")
    outputs: dict[str, object] = {
        "candidate_archive": _normalize_portable_file(
            outputs_raw.get("candidate_archive"), role="candidate_archive"
        ),
        "identity_replay_archive": _normalize_portable_file(
            outputs_raw.get("identity_replay_archive"),
            role="identity_replay_archive",
        ),
    }
    proposal_value = outputs_raw.get("causal_proposal")
    if mode == MATPHYS_CAUSAL_PREFIX_MODE:
        _require(proposal_value is not None, "causal portable proposal is missing")
        proposal = validate_portable_matphys_proposal(proposal_value)
        _require(
            proposal["checkpoint"] == inputs["checkpoint"]
            and proposal["spring_field"] == inputs["spring_field"],
            "portable proposal input identity differs from producer identity",
        )
        outputs["causal_proposal"] = proposal
    else:
        _require(proposal_value is None, "published portable control has a proposal")
        outputs["causal_proposal"] = None

    artifacts = source_artifact_mapping(
        _mapping(certificate.get("source_artifacts"), name="source_artifacts"),
        name="source_artifacts",
    )
    boundary = _mapping(
        certificate.get("information_boundary"), name="information_boundary"
    )
    require_exact_fields(
        boundary, expected=_BOUNDARY_FIELDS, name="information_boundary"
    )
    expected_boundary = _expected_boundary(
        mode=mode,
        target_in_training=target in training,
    )
    _require(
        dict(boundary) == expected_boundary, "portable information boundary changed"
    )
    identity: dict[str, object] = {
        "schema": PORTABLE_IDENTITY_SCHEMA,
        "schema_version": PORTABLE_IDENTITY_VERSION,
        "source_schema": MATPHYS_OFFICIAL_PRODUCER_SCHEMA,
        "source_schema_version": MATPHYS_OFFICIAL_PRODUCER_VERSION,
        "source_producer_protocol": MATPHYS_OFFICIAL_PRODUCER_PROTOCOL,
        "mode": mode,
        "backend_kind": MATPHYS_OFFICIAL_BACKEND_KIND,
        "parameterization": MATPHYS_OFFICIAL_PARAMETERIZATION,
        "rollout_backend": MATPHYS_OFFICIAL_ROLLOUT_BACKEND,
        "coordinate_frame": nonempty_string(
            certificate.get("coordinate_frame"), name="coordinate_frame"
        ),
        "position_unit": nonempty_string(
            certificate.get("position_unit"), name="position_unit"
        ),
        "source_repository": repository_name(
            certificate.get("source_repository"), name="source_repository"
        ),
        "source_revision": exact_revision(
            certificate.get("source_revision"), name="source_revision"
        ),
        "simulator_repository": repository_name(
            certificate.get("simulator_repository"), name="simulator_repository"
        ),
        "simulator_revision": exact_revision(
            certificate.get("simulator_revision"), name="simulator_revision"
        ),
        "case_id": nonempty_string(certificate.get("case_id"), name="case_id"),
        "target_object_id": target,
        "checkpoint_training_object_ids": list(training),
        "target_fit_frame_range_half_open": [fit_start, fit_stop],
        "future_frame_start": future_start,
        "proposal_strength": _finite_number(
            certificate.get("proposal_strength"), name="proposal_strength"
        ),
        "pipeline_components": list(components),
        "pipeline_component_artifacts": normalized_components,
        "inputs": inputs,
        "replay_summary": _normalize_replay_summary(certificate.get("replay_summary")),
        "source_artifacts": dict(artifacts),
        "outputs": outputs,
        "information_boundary": expected_boundary,
        "claim_boundary": nonempty_string(
            certificate.get("claim_boundary"), name="claim_boundary"
        ),
    }
    _require(
        identity["coordinate_frame"] == "right-handed-z-up-world-v1"
        and identity["position_unit"] == "m",
        "MatPhys portable coordinate contract changed",
    )
    _require(
        identity["claim_boundary"] == MATPHYS_OFFICIAL_CLAIM_BOUNDARY,
        "MatPhys portable claim boundary changed",
    )
    normalized = {**identity, "portable_artifact_id": content_id(identity)}
    _require(
        certificate.get("portable_artifact_id") == normalized["portable_artifact_id"],
        "MatPhys portable artifact identity changed",
    )
    return cast(dict[str, object], plain_json(normalized))


def _derive_receipt(
    source_root: Path,
    certificate: Mapping[str, object],
    official: Mapping[str, Any],
) -> dict[str, object]:
    source_artifact = ordinary_file(
        source_root / SOURCE_ARTIFACT_FILENAME,
        name="source producer artifact",
    )
    locations = {
        "checkpoint": _local_file_record("checkpoint", official["checkpoint"]),
        "spring_field": _local_file_record(
            "spring_field", official["spring_field"], spring=True
        ),
        "candidate_parameters": _local_file_record(
            "candidate_parameters", official["candidate_parameters"]
        ),
        "identity_parameters": _local_file_record(
            "identity_parameters", official["identity_parameters"]
        ),
        "replay_input": _local_file_record("replay_input", official["replay_input"]),
    }
    identity: dict[str, object] = {
        "schema": SOURCE_VERIFICATION_SCHEMA,
        "schema_version": SOURCE_VERIFICATION_VERSION,
        "portable_artifact_id": certificate["portable_artifact_id"],
        "source_bundle": {
            "path": str(source_root),
            "artifact_path": SOURCE_ARTIFACT_FILENAME,
            "artifact_sha256": file_sha256(source_artifact),
        },
        "source_locations": locations,
    }
    return {**identity, "host_status_id": content_id(identity)}


def validate_matphys_source_verification(
    value: object,
    *,
    verify_sources: bool = False,
) -> dict[str, object]:
    """Validate host-local source locations separately from portable identity."""

    receipt = _mapping(value, name="MatPhys source verification")
    require_exact_fields(
        receipt,
        expected=_RECEIPT_FIELDS,
        name="MatPhys source verification",
    )
    _require(
        receipt.get("schema") == SOURCE_VERIFICATION_SCHEMA
        and receipt.get("schema_version") == SOURCE_VERIFICATION_VERSION,
        "MatPhys source verification schema changed",
    )
    bundle_raw = _mapping(receipt.get("source_bundle"), name="source_bundle")
    require_exact_fields(
        bundle_raw, expected=_SOURCE_BUNDLE_FIELDS, name="source_bundle"
    )
    bundle_path = nonempty_string(bundle_raw.get("path"), name="source_bundle.path")
    artifact_path = nonempty_string(
        bundle_raw.get("artifact_path"), name="source_bundle.artifact_path"
    )
    _require(
        artifact_path == SOURCE_ARTIFACT_FILENAME,
        "source producer artifact path changed",
    )
    bundle = {
        "path": bundle_path,
        "artifact_path": artifact_path,
        "artifact_sha256": sha256_digest(
            bundle_raw.get("artifact_sha256"), name="source_bundle.artifact_sha256"
        ),
    }
    locations_raw = _mapping(receipt.get("source_locations"), name="source_locations")
    _require(
        set(locations_raw) == set(_INPUT_ROLES),
        "source verification location roster changed",
    )
    locations = {
        role: _normalize_local_file(
            locations_raw.get(role),
            role=role,
            spring=role == "spring_field",
            verify=verify_sources,
        )
        for role in _INPUT_ROLES
    }
    if verify_sources:
        source_root = ordinary_directory(bundle_path, name="source producer bundle")
        artifact = ordinary_file(
            source_root / artifact_path,
            name="source producer artifact",
        )
        _require(
            file_sha256(artifact) == bundle["artifact_sha256"],
            "source producer artifact SHA-256 changed",
        )
        bundle["path"] = str(source_root)
    identity: dict[str, object] = {
        "schema": SOURCE_VERIFICATION_SCHEMA,
        "schema_version": SOURCE_VERIFICATION_VERSION,
        "portable_artifact_id": sha256_digest(
            receipt.get("portable_artifact_id"), name="portable_artifact_id"
        ),
        "source_bundle": bundle,
        "source_locations": locations,
    }
    normalized = {**identity, "host_status_id": content_id(identity)}
    _require(
        receipt.get("host_status_id") == normalized["host_status_id"],
        "MatPhys host source status identity changed",
    )
    return cast(dict[str, object], plain_json(normalized))


def materialize_matphys_portable_identity(
    source_bundle_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Publish a portable certificate plus a separate local-source receipt."""

    source_root = ordinary_directory(
        source_bundle_dir, name="source official MatPhys bundle"
    )
    certificate, official = _derive_certificate(source_root, verify_sources=True)
    receipt = _derive_receipt(source_root, certificate, official)
    output, staging = new_staging_directory(output_dir)
    try:
        write_atomic_json(
            certificate,
            staging / PORTABLE_IDENTITY_FILENAME,
            overwrite=False,
        )
        write_atomic_json(
            receipt,
            staging / SOURCE_VERIFICATION_FILENAME,
            overwrite=False,
        )
        write_checksum_manifest(staging, (PORTABLE_IDENTITY_FILENAME,))
        publish_staging_directory(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_matphys_portable_identity(output, verify_sources=True)


def validate_matphys_portable_identity(
    output_dir: str | Path,
    *,
    verify_sources: bool = False,
) -> dict[str, object]:
    """Validate portable identity and optionally reverify its host-local sources."""

    root = ordinary_directory(output_dir, name="MatPhys portable identity bundle")
    expected_roster = frozenset(
        {
            PORTABLE_IDENTITY_FILENAME,
            SOURCE_VERIFICATION_FILENAME,
            CHECKSUMS_FILENAME,
        }
    )
    _require(
        regular_file_roster(root) == expected_roster,
        "MatPhys portable identity bundle roster changed",
    )
    certificate = validate_matphys_portable_certificate(
        load_strict_json_object(
            root / PORTABLE_IDENTITY_FILENAME,
            label="MatPhys portable identity",
        )
    )
    receipt = validate_matphys_source_verification(
        load_strict_json_object(
            root / SOURCE_VERIFICATION_FILENAME,
            label="MatPhys source verification",
        ),
        verify_sources=verify_sources,
    )
    _require(
        receipt["portable_artifact_id"] == certificate["portable_artifact_id"],
        "source verification names another portable artifact",
    )
    validate_checksum_manifest(root, (PORTABLE_IDENTITY_FILENAME,))
    if verify_sources:
        source_bundle = cast(Mapping[str, object], receipt["source_bundle"])
        source_root = ordinary_directory(
            nonempty_string(source_bundle.get("path"), name="source_bundle.path"),
            name="source official MatPhys bundle",
        )
        expected_certificate, official = _derive_certificate(
            source_root, verify_sources=True
        )
        _require(
            expected_certificate == certificate,
            "portable certificate no longer derives from source bundle",
        )
        expected_receipt = _derive_receipt(source_root, certificate, official)
        _require(
            expected_receipt == receipt,
            "host source receipt no longer derives from source bundle",
        )
    return {
        "portable_artifact": certificate,
        "source_verification": receipt,
    }


__all__ = [
    "CHECKSUMS_FILENAME",
    "PORTABLE_IDENTITY_FILENAME",
    "PORTABLE_IDENTITY_SCHEMA",
    "PORTABLE_IDENTITY_VERSION",
    "SOURCE_VERIFICATION_FILENAME",
    "SOURCE_VERIFICATION_SCHEMA",
    "SOURCE_VERIFICATION_VERSION",
    "materialize_matphys_portable_identity",
    "validate_matphys_portable_certificate",
    "validate_matphys_portable_identity",
    "validate_matphys_source_verification",
    "validate_portable_matphys_proposal",
]
